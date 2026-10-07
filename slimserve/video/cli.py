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


def _one_clip(cfg: dict[str, Any], args: Any) -> int:
    from slimserve.video import server
    from slimserve.video.ltx25.pipeline import LTX25Engine

    body = {"prompt": args.prompt}
    for key in (
        "size",
        "seconds",
        "seed",
        "negative_prompt",
        "decoder",
        "temporal_upscalings",
        "spatial_upscalings",
    ):
        if getattr(args, key, None) is not None:
            body[key] = getattr(args, key)
    if getattr(args, "image", None):
        try:
            body["image"] = Path(args.image).expanduser().read_bytes()
        except OSError as error:
            term.fail(f"cannot read --image: {error}")
            return 2
        if getattr(args, "image_strength", None) is not None:
            body["image_strength"] = args.image_strength
    elif getattr(args, "image_strength", None) is not None:
        term.fail("--image-strength needs --image")
        return 2
    try:
        params = server.normalize_request(body, cfg)
    except server.BadRequest as error:
        term.fail(str(error))
        return 2
    out = Path(args.output or f"{cfg['pipeline']}-{params['seed']}.mp4").expanduser()
    started = time.perf_counter()
    engine = LTX25Engine(
        root=cfg["root"],
        variant="dev" if cfg["pipeline"] in ("dev", "hq") else "distilled",
    )
    prompt = params.pop("prompt")
    decoder = params.pop("decoder")
    fast = server.fast_settings(cfg)
    if fast is not None:
        params["fast"] = fast
    result = getattr(engine, cfg["pipeline"])(
        prompt,
        keep_text=False,
        on_step=lambda stage, i, s: term.note(f"{stage} step {i + 1}"),
        **params,
    )
    engine.render(result, out, seed=params["seed"], decoder=decoder)
    spans = ", ".join(f"{k} {v:.1f}s" for k, v in result.timings.spans.items())
    auto = (
        f"  (duration head: {result.predicted_seconds:.1f} s)"
        if result.predicted_seconds is not None
        else ""
    )
    term.ok(
        f"{out}  {params['width']}x{params['height']}x{result.num_frames}{auto}  "
        f"{time.perf_counter() - started:.1f}s  ({spans})"
    )
    return 0
