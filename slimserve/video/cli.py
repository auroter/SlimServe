# SPDX-License-Identifier: Apache-2.0
"""`slimserve <video profile>`: serve, or render one clip with --prompt."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from slimserve import term
from slimserve.registry import Plan


def engine_config(plan: Plan) -> dict[str, Any]:
    cfg = dict(plan.engine)
    cfg["root"] = plan.model_dir
    return cfg


def run(plan: Plan, args: Any) -> int:
    cfg = engine_config(plan)
    name = cfg.get("served_model_name", plan.profile_id)
    if args.chat:
        term.fail(
            f"{plan.profile_id} generates video; use --prompt for one clip or serve it"
        )
        return 2
    if args.prompt:
        return _one_clip(cfg, args)
    from slimserve.video import server

    term.ok(
        f"serving {name} ({cfg['pipeline']}) on http://{args.host}:{args.port}"
        "  POST /v1/videos"
    )
    return server.serve(cfg, name, args.host, args.port)


def _add_images(body: dict[str, Any], args: Any) -> None:
    """--image PATH [FRAME [STRENGTH [CRF]]] (repeatable, upstream's form): a
    lone frame-0 still is the `image` shorthand (with --image-strength), the
    rest go through `images`."""
    groups = getattr(args, "image", None) or []
    strength_flag = getattr(args, "image_strength", None)
    if strength_flag is not None and not groups:
        raise ValueError("--image-strength needs --image")
    stills = []
    for group in groups:
        if not 1 <= len(group) <= 4:
            raise ValueError("--image takes PATH [FRAME [STRENGTH [CRF]]]")
        path = Path(group[0]).expanduser()
        if path.suffix.lower() == ".exr":  # HDR stills travel by path
            if not path.is_file():
                raise ValueError(f"cannot read --image {path}")
            data = None
        else:
            try:
                data = path.read_bytes()
            except OSError as error:
                raise ValueError(f"cannot read --image {path}: {error}") from error
        try:
            frame = int(group[1]) if len(group) > 1 else 0
            strength = float(group[2]) if len(group) > 2 else None
            crf = int(group[3]) if len(group) > 3 else None
        except ValueError as error:
            raise ValueError(f"--image {path}: {error}") from error
        entry = {"frame": frame, "strength": strength, "crf": crf}
        entry["image" if data is not None else "path"] = (
            data if data is not None else str(path)
        )
        stills.append(entry)
    if not stills:
        return
    first = stills[0]
    if first["strength"] is None:
        first["strength"] = 1.0 if strength_flag is None else strength_flag
    elif strength_flag is not None:
        raise ValueError("give the strength inline or with --image-strength, not both")
    for still in stills[1:]:
        if still["strength"] is None:
            still["strength"] = 1.0
    if (
        len(stills) == 1
        and first["frame"] == 0
        and first["crf"] is None
        and "image" in first
    ):
        body["image"], body["image_strength"] = first["image"], first["strength"]
        return
    body["images"] = stills


def _add_guidance(body: dict[str, Any], args: Any) -> None:
    """Upstream's guider flags -> `steps`, `guidance`, `lora_strengths`."""
    if getattr(args, "num_inference_steps", None) is not None:
        body["steps"] = args.num_inference_steps
    guidance: dict[str, dict[str, Any]] = {}
    for modality, other in (("video", "a2v"), ("audio", "v2a")):
        fields = {
            "cfg": getattr(args, f"{modality}_cfg_guidance_scale", None),
            "stg": getattr(args, f"{modality}_stg_guidance_scale", None),
            "stg_blocks": getattr(args, f"{modality}_stg_blocks", None),
            "rescale": getattr(args, f"{modality}_rescale_scale", None),
            "modality": getattr(args, f"{other}_guidance_scale", None),
            "skip_step": getattr(args, f"{modality}_skip_step", None),
        }
        fields = {k: v for k, v in fields.items() if v is not None}
        if fields:
            guidance[modality] = fields
    if guidance:
        body["guidance"] = guidance
    s1 = getattr(args, "distilled_lora_strength_stage_1", None)
    s2 = getattr(args, "distilled_lora_strength_stage_2", None)
    if s1 is not None or s2 is not None:
        from slimserve.video.ltx25 import pipeline as pl

        body["lora_strengths"] = [
            pl.HQ_LORA_STAGE_1 if s1 is None else s1,
            pl.HQ_LORA_STAGE_2 if s2 is None else s2,
        ]


def _add_loras(body: dict[str, Any], args: Any) -> None:
    """--lora PATH [STRENGTH], repeatable (upstream's form)."""
    groups = getattr(args, "lora", None) or []
    loras = []
    for group in groups:
        if not 1 <= len(group) <= 2:
            raise ValueError("--lora takes PATH [STRENGTH]")
        try:
            strength = float(group[1]) if len(group) > 1 else 1.0
        except ValueError as error:
            raise ValueError(f"--lora {group[0]}: {error}") from error
        loras.append({"path": str(Path(group[0]).expanduser()), "strength": strength})
    if loras:
        body["loras"] = loras


def _one_clip(cfg: dict[str, Any], args: Any) -> int:
    from slimserve.video import server
    from slimserve.video.ltx25.pipeline import LTX25Engine

    body = {"prompt": args.prompt}
    if getattr(args, "frame_rate", None) is not None:
        body["fps"] = args.frame_rate
    for key in (
        "size",
        "seconds",
        "seed",
        "negative_prompt",
        "decoder",
        "temporal_upscalings",
        "spatial_upscalings",
        "video_path",
        "start_time",
        "end_time",
        "tile_height",
        "tile_width",
        "audio_path",
        "audio_start_time",
        "audio_max_duration",
        "reference_video",
        "reference_strength",
        "chunk_pixel_frames",
        "chunk_carry_frames",
        "chunk_blend_frames",
        "hdr",
    ):
        if getattr(args, key, None) is not None:
            body[key] = getattr(args, key)
    for key in (
        "skip_stage_2",
        "stage_2_ic_lora",
        "tile",
        "decode_with_keyframes",
        "chunked",
    ):
        if getattr(args, key, False):
            body[key] = True
    if getattr(args, "num_generated_keyframes", None):
        body["generated_keyframes"] = args.num_generated_keyframes
    if getattr(args, "video_conditioning", None):
        body["video_conditioning"] = []
        for group in args.video_conditioning:
            if not 1 <= len(group) <= 2:
                term.fail("--video-conditioning takes PATH [STRENGTH]")
                return 2
            body["video_conditioning"].append(
                {
                    "path": str(Path(group[0]).expanduser()),
                    "strength": float(group[1]) if len(group) > 1 else 1.0,
                }
            )
    if getattr(args, "conditioning_attention_mask", None):
        group = args.conditioning_attention_mask
        if len(group) != 2:
            term.fail("--conditioning-attention-mask takes MASK_PATH STRENGTH")
            return 2
        body["attention_mask"] = str(Path(group[0]).expanduser())
        body["attention_strength"] = float(group[1])
    if getattr(args, "enhance_prompt", False):
        body["enhance_prompt"] = True
    try:
        _add_images(body, args)
        _add_guidance(body, args)
        _add_loras(body, args)
        params = server.normalize_request(body, cfg)
    except ValueError as error:
        term.fail(str(error))
        return 2
    ext = ".wav" if cfg["pipeline"] == "t2a" else ".mp4"
    out = Path(args.output or f"{cfg['pipeline']}-{params['seed']}{ext}").expanduser()
    started = time.perf_counter()
    engine = LTX25Engine(
        root=cfg["root"],
        variant="dev" if cfg["pipeline"] in server.GUIDED_PIPELINES else "distilled",
    )
    source = params.pop("source", None)
    prompt = params.pop("prompt")
    extra = ""
    if params.pop("enhance_prompt", False):
        t0 = time.perf_counter()
        prompt = engine.enhance(prompt, server._first_still(params))
        extra = f"enhance {time.perf_counter() - t0:.1f}s, "
        term.note(f"enhanced prompt: {prompt}")
        engine.unload_enhancer()
    decoder = params.pop("decoder", None)
    if params.get("chunk") is not None:
        params["decoder"] = decoder
    if source:
        term.note(
            f"source {source['width']}x{source['height']}x{source['num_frames']} "
            f"at {source['fps']:g} fps"
        )
    fast = server.fast_settings(cfg)
    if fast is not None:
        params["fast"] = fast
    loras = (
        None
        if cfg["pipeline"] in ("ic_lora", "dubit", "alpha")
        else params.pop("loras", None)
    )
    with engine.user_loras(loras):
        result = getattr(engine, cfg["pipeline"])(
            prompt,
            keep_text=False,
            on_step=lambda stage, i, s: term.note(f"{stage} step {i + 1}"),
            **params,
        )
    out = engine.render(
        result, out, seed=params["seed"], decoder=decoder, hdr=params.get("hdr")
    )
    spans = ", ".join(f"{k} {v:.1f}s" for k, v in result.timings.spans.items())
    auto = (
        f"  (duration head: {result.predicted_seconds:.1f} s)"
        if result.predicted_seconds is not None
        else ""
    )
    term.ok(
        f"{out}  {result.width}x{result.height}x{result.num_frames}{auto}  "
        f"{time.perf_counter() - started:.1f}s  ({extra}{spans})"
    )
    return 0
