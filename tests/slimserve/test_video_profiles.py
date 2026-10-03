# SPDX-License-Identifier: Apache-2.0
"""The LTX-2.5 video profiles: registry shape, request envelope, and the
one-at-a-time serving queue. No weights and no GPU are needed here; the
measured gates live in perf/ltx25_harness/."""

import threading
import time

import pytest

from slimserve import registry
from slimserve.registry import files_for, resolve
from slimserve.video import server

VIDEO_IDS = ("ltx25-distilled", "ltx25-dev", "ltx25-dfr")
GIB = 1 << 30


def _plan(profile_id):
    return resolve(profile_id, "metal", 1, None, 128 * GIB)


def test_video_profiles_resolve_on_metal_only():
    for profile_id in VIDEO_IDS:
        entry = registry.describe(profile_id)
        assert entry["platforms"] == ["metal"]
        plan = _plan(profile_id)
        assert plan.engine["pipeline"] == profile_id.split("-")[1]
        assert not registry.is_language_model(plan.source)


def test_video_profiles_carry_no_chat_serving_defaults():
    for profile_id in VIDEO_IDS:
        engine = _plan(profile_id).engine
        for key in (
            "enable_prefix_caching",
            "enable_auto_tool_choice",
            "default_chat_template_kwargs",
        ):
            assert key not in engine


def test_video_profiles_are_gated_on_the_validated_memory_size():
    with pytest.raises(registry.ProfileError):
        resolve("ltx25-distilled", "metal", 1, None, 64 * GIB)


def test_each_pipeline_fetches_its_own_transformer_and_adapter():
    paths = {
        pid: {entry["path"] for entry in files_for(_plan(pid))} for pid in VIDEO_IDS
    }
    dev = "diffusion_models/ltx-2.5-22b-dev-transformer-bf16.safetensors"
    distilled = "diffusion_models/ltx-2.5-22b-distilled-transformer-bf16.safetensors"
    assert distilled in paths["ltx25-distilled"] and dev not in paths["ltx25-distilled"]
    assert dev in paths["ltx25-dev"] and distilled not in paths["ltx25-dev"]
    assert "loras/ltx-2.5-22b-distilled-lora-450-bf16.safetensors" in paths["ltx25-dev"]
    detail = "loras/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
    assert detail in paths["ltx25-dfr"] and detail not in paths["ltx25-distilled"]
    for pid in VIDEO_IDS:
        assert (
            "text_encoders/gemma4-12b-with-proj-ltx-2.5-bf16.safetensors" in paths[pid]
        )
        assert "vae/ltx-2.5-video-vae-conv-bf16.safetensors" in paths[pid]


def test_the_detailing_adapter_downloads_from_its_own_repository():
    entry = next(e for e in files_for(_plan("ltx25-dfr")) if "ic-lora" in e["path"])
    assert "LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler" in entry["url"]
    assert entry["url"].endswith(
        "/ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
    )


CFG = {
    "pipeline": "distilled",
    "width": 1536,
    "height": 1024,
    "num_frames": 121,
    "fps": 24.0,
    "max_video_tokens": 24576,
}


def test_request_defaults_to_the_profile_clip():
    params = server.normalize_request({"prompt": "a fox"}, CFG)
    assert (
        params["width"],
        params["height"],
        params["num_frames"],
        params["seed"],
    ) == (1536, 1024, 121, 42)


def test_request_accepts_size_and_seconds():
    params = server.normalize_request(
        {"prompt": "a fox", "size": "768x512", "seconds": 2}, CFG
    )
    assert (params["width"], params["height"], params["num_frames"]) == (768, 512, 49)


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"prompt": " "},
        {"prompt": "x", "size": "770x512"},
        {"prompt": "x", "num_frames": 120},
        {"prompt": "x", "size": "3072x2048"},  # outside the validated memory envelope
        {"prompt": "x", "num_frames": 241},
        {"prompt": "x", "negative_prompt": "blurry"},  # distilled has no CFG
    ],
)
def test_request_outside_the_envelope_is_refused(body):
    with pytest.raises(server.BadRequest):
        server.normalize_request(body, CFG)


def test_decoder_defaults_to_diffusion_and_accepts_conv():
    assert server.normalize_request({"prompt": "x"}, CFG)["decoder"] == "diffusion"
    req = {"prompt": "x", "decoder": "conv"}
    assert server.normalize_request(req, CFG)["decoder"] == "conv"
    with pytest.raises(server.BadRequest):
        server.normalize_request({"prompt": "x", "decoder": "fast"}, CFG)


def test_negative_prompt_reaches_the_dev_pipeline():
    params = server.normalize_request(
        {"prompt": "x", "negative_prompt": "blurry"}, {**CFG, "pipeline": "dev"}
    )
    assert params["negative_prompt"] == "blurry"


class _FakeEngine:
    """Records overlap: the real engine must never run two generations at once."""

    def __init__(self):
        self.active = 0
        self.max_active = 0
        self.calls = 0

    def distilled(self, prompt, on_step=None, **params):
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls += 1
        if "fail" in prompt:
            self.active -= 1
            raise RuntimeError("boom")
        on_step("stage1", 0, 1.0)
        time.sleep(0.02)
        self.active -= 1

        class _Timings:
            spans = {"stage1": 0.02}
            memory = {"stage1": (GIB, GIB, 0)}

        class _Result:
            timings = _Timings()

        return _Result()

    def render(self, result, path, seed=42, decoder=None):
        self.decoders = getattr(self, "decoders", []) + [decoder]
        path.write_bytes(b"mp4")
        return path


def _service(tmp_path, engine):
    service = server.VideoService(
        CFG, "LTX-2.5", tmp_path, engine_factory=lambda: engine
    )
    assert service.ready.wait(5)
    return service


def test_concurrent_requests_run_one_at_a_time_and_all_complete(tmp_path):
    engine = _FakeEngine()
    service = _service(tmp_path, engine)
    jobs, errors = [], []

    def client(i):
        try:
            jobs.append(service.submit({"prompt": f"clip {i}", "size": "768x512"}))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=client, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    for job in jobs:
        assert job.done.wait(10)
        assert job.status == "completed" and job.path.read_bytes() == b"mp4"
    assert engine.calls == 8 and engine.max_active == 1
    assert service.health()["completed"] == 8
    service.stop()


def test_a_failed_generation_does_not_stop_the_queue(tmp_path):
    engine = _FakeEngine()
    service = _service(tmp_path, engine)
    bad = service.submit({"prompt": "fail please"})
    good = service.submit({"prompt": "fine"})
    assert bad.done.wait(10) and good.done.wait(10)
    assert bad.status == "failed" and "boom" in bad.error
    assert good.status == "completed"
    service.stop()


def test_the_queue_is_bounded(tmp_path):
    gate = threading.Event()

    class _Blocked(_FakeEngine):
        def distilled(self, prompt, on_step=None, **params):
            gate.wait(10)
            return super().distilled(prompt, on_step=on_step, **params)

    service = _service(tmp_path, _Blocked())
    accepted = [
        service.submit({"prompt": "x"}) for _ in range(server.MAX_QUEUED + 1)
    ]  # one running
    time.sleep(0.05)
    with pytest.raises(OverflowError):
        for _ in range(3):
            service.submit({"prompt": "x"})
    gate.set()
    for job in accepted:
        assert job.done.wait(20)
    service.stop()


def test_an_engine_that_fails_to_load_is_reported_not_hidden(tmp_path):
    def broken():
        raise FileNotFoundError("weights missing")

    service = server.VideoService(CFG, "LTX-2.5", tmp_path, engine_factory=broken)
    assert service.ready.wait(5)
    health = service.health()
    assert health["status"] == "error" and "weights missing" in health["error"]


def test_dfr_canvas_and_dev_schedule():
    mx = pytest.importorskip("mlx.core")  # noqa: F841 - the sampling module needs MLX
    from slimserve.video.ltx25 import sampling

    assert sampling.dfr_canvas(121) == (121, 24, [24, 48, 72, 96, 120])
    assert sampling.dfr_canvas(97) == (97, 32, [32, 64, 96])
    assert sampling.dfr_canvas(49)[0] == 49
    sigmas = sampling.ltx2_schedule(30, 1536)
    assert len(sigmas) == 31 and sigmas[0] == 1.0 and sigmas[-1] == 0.0
    assert abs(sigmas[-2] - 0.1) < 1e-9
    assert all(a > b for a, b in zip(sigmas, sigmas[1:]))


def test_samplers_reach_the_denoisers_answer():
    mx = pytest.importorskip("mlx.core")
    from slimserve.video.ltx25 import sampling

    target_v, target_a = mx.full((1, 6, 4), 2.0), mx.full((1, 3, 4), -1.0)
    video = sampling.noised_state(
        (1, 6, 4), mx.zeros((1, 6, 3)), seed=1, tokens_per_frame=2
    )
    audio = sampling.noised_state((1, 3, 4), mx.zeros((1, 3, 1)), seed=2)
    perfect = lambda vs, au, vx, ax, sigma: (target_v, target_a)  # noqa: E731
    for out in (
        sampling.euler_loop(perfect, video, audio, sampling.STAGE_2_DISTILLED_SIGMAS),
        sampling.euler_ancestral_loop(
            perfect, video, audio, sampling.DISTILLED_SIGMAS, noise_seed=3
        ),
    ):
        assert float(mx.abs(out[0] - target_v).max()) < 1e-5
        assert float(mx.abs(out[1] - target_a).max()) < 1e-5
