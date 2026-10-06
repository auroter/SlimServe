# SPDX-License-Identifier: Apache-2.0
"""HTTP serving for the video profiles.

One GPU, one generation at a time: requests queue and a single worker thread
owns the engine (and every MLX call), with the weights resident between
requests. The API is job-shaped because a clip takes minutes:

    POST   /v1/videos                 {"prompt": ..., "size": "1536x1024",
                                       "seconds": 5, "seed": 42,
                                       "image": <base64 or data: URL>,
                                       "image_strength": 1.0}
    GET    /v1/videos/<id>            status, progress, timings
    GET    /v1/videos/<id>/content    the mp4 (H.264 + AAC)
    DELETE /v1/videos/<id>
    GET    /v1/models, /health

`"wait": true` in the POST body holds the request open and returns the
finished job. `image` (image-to-video) is an encoded still (PNG, JPEG, ...)
as base64, optionally wrapped in a `data:image/...;base64,` URL; it becomes
the clip's first frame, pinned at `image_strength` (0-1, default 1.0).
Requests outside the profile's validated envelope are refused: the envelope
is what was measured to fit this machine's memory.
"""

from __future__ import annotations

import base64
import binascii
import io
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MAX_QUEUED = 16
KEEP_FINISHED = 32


class BadRequest(ValueError):
    pass


@dataclass
class Job:
    id: str
    params: dict[str, Any]
    status: str = "queued"  # queued | in_progress | completed | failed
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    progress: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    path: Path | None = None
    timings: dict[str, Any] = field(default_factory=dict)
    done: threading.Event = field(default_factory=threading.Event)

    def public(self, model: str) -> dict[str, Any]:
        out = {
            "id": self.id,
            "object": "video",
            "model": model,
            "status": self.status,
            "created_at": int(self.created_at),
            "progress": self.progress,
            **self.params,
        }
        if isinstance(out.get("image"), (bytes, bytearray)):
            out["image"] = f"<{len(out['image'])} bytes>"
        if self.completed_at:
            out["completed_at"] = int(self.completed_at)
            out["seconds_elapsed"] = round(
                self.completed_at - (self.started_at or self.created_at), 2
            )
        if self.error:
            out["error"] = {"message": self.error}
        if self.timings:
            out["timings"] = self.timings
        return out


def normalize_request(body: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate a request against the profile's envelope; returns engine parameters."""
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise BadRequest("`prompt` is required")
    width, height = cfg["width"], cfg["height"]
    if size := body.get("size"):
        try:
            width, height = (int(x) for x in str(size).lower().split("x"))
        except ValueError as exc:
            raise BadRequest("`size` must look like 1536x1024") from exc
    width, height = int(body.get("width", width)), int(body.get("height", height))
    fps = float(body.get("fps", cfg["fps"]))
    if "num_frames" in body:
        frames = int(body["num_frames"])
    elif "seconds" in body:
        frames = int(round(float(body["seconds"]) * fps / 8)) * 8 + 1
    else:
        frames = cfg["num_frames"]
    if width % 64 or height % 64 or width < 256 or height < 256:
        raise BadRequest("width and height must be multiples of 64, at least 256")
    if frames < 9 or (frames - 1) % 8:
        raise BadRequest("num_frames must be 8k + 1 (9, 17, ..., 121)")
    if not 1.0 <= fps <= 60.0:
        raise BadRequest("fps must be between 1 and 60")
    tokens = ((frames - 1) // 8 + 1) * (height // 32) * (width // 32)
    if tokens > cfg["max_video_tokens"]:
        raise BadRequest(
            f"{width}x{height}x{frames} is {tokens} latent tokens; "
            "this profile is validated up to "
            f"{cfg['max_video_tokens']} "
            f"({cfg['width']}x{cfg['height']}x{cfg['num_frames']})"
        )
    params: dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "num_frames": frames,
        "fps": fps,
        "seed": int(body.get("seed", 42)),
    }
    decoder = str(body.get("decoder", cfg.get("decoder", "diffusion")))
    if decoder not in ("diffusion", "conv"):
        raise BadRequest(
            "decoder must be 'diffusion' (default, sharper) or 'conv' (faster)"
        )
    params["decoder"] = decoder
    if "negative_prompt" in body:
        if cfg["pipeline"] != "dev":
            raise BadRequest(
                "negative_prompt applies to the dev pipeline only "
                "(the distilled flows have no CFG)"
            )
        params["negative_prompt"] = str(body["negative_prompt"])
    if body.get("image") is not None:
        params["image"] = decode_image_field(body["image"])
        strength = body.get("image_strength", 1.0)
        try:
            strength = float(strength)
        except (TypeError, ValueError) as exc:
            raise BadRequest("image_strength must be a number in [0, 1]") from exc
        if not 0.0 <= strength <= 1.0:
            raise BadRequest("image_strength must be a number in [0, 1]")
        params["image_strength"] = strength
    elif "image_strength" in body:
        raise BadRequest("image_strength needs an image")
    return params


def fast_settings(cfg: dict[str, Any]):
    """The profile's fast tier (`engine.fast`), or None for the exact pipeline."""
    if not cfg.get("fast"):
        return None
    from slimserve.video.ltx25.pipeline import Fast

    fast = Fast.from_config(cfg["fast"])
    return fast if fast.active else None


def decode_image_field(value: Any) -> bytes:
    """`image`: encoded image bytes as base64, optionally a data: URL; raw bytes
    (the CLI's file contents) pass through. The bytes are decoded as an image by
    the engine; here they must at least be a non-empty, valid base64 payload."""
    if isinstance(value, (bytes, bytearray)):
        data = bytes(value)
    elif isinstance(value, str):
        text = value.strip()
        if text.startswith("data:"):
            header, sep, text = text.partition(",")
            if not sep or ";base64" not in header:
                raise BadRequest("image data: URL must be base64 encoded")
        try:
            data = base64.b64decode(text, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise BadRequest("image must be base64 (optionally a data: URL)") from exc
    else:
        raise BadRequest("image must be a base64 string")
    if not data:
        raise BadRequest("image is empty")
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            im.verify()
    except ImportError:
        pass  # the engine's decoder reports it instead
    except Exception as exc:
        raise BadRequest(f"image is not a decodable image: {exc}") from exc
    return data


class VideoService:
    """The queue, the worker and the job table. HTTP-free so tests can drive it."""

    def __init__(
        self,
        cfg: dict[str, Any],
        model_name: str,
        output_dir: Path,
        engine_factory=None,
    ):
        self.cfg, self.model_name, self.output_dir = cfg, model_name, output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.jobs: dict[str, Job] = {}
        self.order: list[str] = []
        self.lock = threading.Lock()
        self.queue: queue.Queue[Job | None] = queue.Queue()
        self.ready = threading.Event()
        self.load_error: str | None = None
        self.completed = 0
        self.failed = 0
        self._engine_factory = engine_factory or self._default_engine
        self.worker = threading.Thread(
            target=self._run, name="ltx25-worker", daemon=True
        )
        self.worker.start()

    def _default_engine(self):
        from slimserve.video.ltx25.pipeline import LTX25Engine

        pipeline = self.cfg["pipeline"]
        engine = LTX25Engine(
            root=self.cfg.get("root"),
            variant="dev" if pipeline == "dev" else "distilled",
        )
        engine.load_text()
        engine.load_dit()
        engine.load_vae()
        engine.load_upscaler()
        engine.load_audio()
        if self.cfg.get("decoder", "diffusion") == "diffusion":
            engine.load_diffvae()
        if pipeline == "dev":
            engine.load_distilled_lora()
        elif pipeline == "dfr":
            engine.load_detail_lora()
        return engine

    # ---- worker (the only thread that touches the engine) -----------------
    def _run(self) -> None:
        try:
            engine = self._engine_factory()
        except (
            Exception
        ) as exc:  # surfaced through /health; the process stays up to report it
            self.load_error = f"{type(exc).__name__}: {exc}"
            self.ready.set()
            return
        self.ready.set()
        while True:
            job = self.queue.get()
            if job is None:
                return
            job.status, job.started_at = "in_progress", time.time()
            try:
                self._generate(engine, job)
                job.status = "completed"
                self.completed += 1
            except Exception as exc:
                job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"
                self.failed += 1
            job.completed_at = time.time()
            job.done.set()
            self._trim()

    def _generate(self, engine, job: Job) -> None:
        p = dict(job.params)
        prompt = p.pop("prompt")

        def on_step(stage: str, index: int, sigma: float) -> None:
            job.progress = {"stage": stage, "step": index + 1}

        decoder = p.pop("decoder", None)
        fast = fast_settings(self.cfg)
        if fast is not None:
            p["fast"] = fast
        result = getattr(engine, self.cfg["pipeline"])(prompt, on_step=on_step, **p)
        job.progress = {"stage": "decode"}
        path = self.output_dir / f"{job.id}.mp4"
        engine.render(result, path, seed=p["seed"], decoder=decoder)
        job.path = path
        tm = result.timings
        job.timings = {
            "spans_s": {k: round(v, 2) for k, v in tm.spans.items()},
            "peak_gib": round(
                max((m[0] for m in tm.memory.values()), default=0) / 2**30, 1
            ),
        }
        job.progress = {"stage": "done"}

    def _trim(self) -> None:
        with self.lock:
            finished = [
                i for i in self.order if self.jobs[i].status in ("completed", "failed")
            ]
            for old in finished[:-KEEP_FINISHED]:
                self._drop(old)

    def _drop(self, job_id: str) -> None:
        job = self.jobs.pop(job_id, None)
        if job_id in self.order:
            self.order.remove(job_id)
        if job and job.path:
            job.path.unlink(missing_ok=True)

    # ---- API ---------------------------------------------------------------
    def submit(self, body: dict[str, Any]) -> Job:
        params = normalize_request(body, self.cfg)
        with self.lock:
            waiting = sum(1 for j in self.jobs.values() if j.status == "queued")
            if waiting >= MAX_QUEUED:
                raise OverflowError(f"{waiting} requests already queued")
            job = Job(id="video_" + uuid.uuid4().hex[:24], params=params)
            self.jobs[job.id] = job
            self.order.append(job.id)
        self.queue.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def delete(self, job_id: str) -> bool:
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None or job.status in ("queued", "in_progress"):
                return False
            self._drop(job_id)
            return True

    def health(self) -> dict[str, Any]:
        state = (
            "error" if self.load_error else ("ok" if self.ready.is_set() else "loading")
        )
        counts = {
            s: sum(1 for j in self.jobs.values() if j.status == s)
            for s in ("queued", "in_progress")
        }
        out = {
            "status": state,
            "model": self.model_name,
            "pipeline": self.cfg["pipeline"],
            **counts,
            "completed": self.completed,
            "failed": self.failed,
        }
        if self.load_error:
            out["error"] = self.load_error
        try:  # counters only; safe from any thread
            import mlx.core as mx

            out["memory_gib"] = {
                "active": round(mx.get_active_memory() / 2**30, 1),
                "cache": round(mx.get_cache_memory() / 2**30, 1),
            }
        except ImportError:
            pass
        return out

    def stop(self) -> None:
        self.queue.put(None)


def make_handler(service: VideoService):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(
            self, fmt: str, *args: Any
        ) -> None:  # quiet; the CLI prints job lines
            pass

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, message: str) -> None:
            self._json(code, {"error": {"message": message, "code": code}})

        def do_GET(self) -> None:
            path = self.path.split("?")[0].rstrip("/")
            if path == "/health":
                health = service.health()
                return self._json(200 if health["status"] == "ok" else 503, health)
            if path == "/v1/models":
                return self._json(
                    200,
                    {
                        "object": "list",
                        "data": [{"id": service.model_name, "object": "model"}],
                    },
                )
            parts = path.split("/")
            if len(parts) >= 4 and parts[1:3] == ["v1", "videos"]:
                job = service.get(parts[3])
                if job is None:
                    return self._error(404, "no such video")
                if len(parts) == 4:
                    return self._json(200, job.public(service.model_name))
                if parts[4] == "content":
                    if job.status != "completed" or job.path is None:
                        return self._error(409, f"video is {job.status}")
                    data = job.path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
            self._error(404, "not found")

        def do_POST(self) -> None:
            if self.path.split("?")[0].rstrip("/") != "/v1/videos":
                return self._error(404, "not found")
            if service.load_error:
                return self._error(503, service.load_error)
            try:
                body = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length", "0")))
                    or b"{}"
                )
                job = service.submit(body)
            except (BadRequest, json.JSONDecodeError) as exc:
                return self._error(400, str(exc))
            except OverflowError as exc:
                return self._error(429, str(exc))
            if body.get("wait"):
                job.done.wait()
            self._json(200, job.public(service.model_name))

        def do_DELETE(self) -> None:
            parts = self.path.split("?")[0].rstrip("/").split("/")
            if len(parts) == 4 and parts[1:3] == ["v1", "videos"]:
                if service.delete(parts[3]):
                    return self._json(200, {"id": parts[3], "deleted": True})
                return self._error(
                    409, "no such video, or it is still queued or running"
                )
            self._error(404, "not found")

    return Handler


def output_dir() -> Path:
    return Path(
        os.environ.get("SLIMSERVE_VIDEO_DIR", "~/.cache/slimserve/videos")
    ).expanduser()


def serve(cfg: dict[str, Any], model_name: str, host: str, port: int) -> int:
    service = VideoService(cfg, model_name, output_dir())
    httpd = ThreadingHTTPServer((host, port), make_handler(service))
    httpd.daemon_threads = True
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        service.stop()
        httpd.server_close()
    return 0
