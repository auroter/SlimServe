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
        stills.append({"image": data, "frame": frame, "strength": strength, "crf": crf})
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
    if len(stills) == 1 and first["frame"] == 0 and first["crf"] is None:
        body["image"], body["image_strength"] = first["image"], first["strength"]
        return
    body["images"] = stills


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
    if getattr(args, "enhance_prompt", False):
        body["enhance_prompt"] = True
    try:
        _add_images(body, args)
        params = server.normalize_request(body, cfg)
    except ValueError as error:
        term.fail(str(error))
        return 2
    out = Path(args.output or f"{cfg['pipeline']}-{params['seed']}.mp4").expanduser()
    started = time.perf_counter()
    engine = LTX25Engine(
        root=cfg["root"],
        variant="dev" if cfg["pipeline"] in server.GUIDED_PIPELINES else "distilled",
    )
    prompt = params.pop("prompt")
    extra = ""
    if params.pop("enhance_prompt", False):
        t0 = time.perf_counter()
        prompt = engine.enhance(prompt, server._first_still(params))
        extra = f"enhance {time.perf_counter() - t0:.1f}s, "
        term.note(f"enhanced prompt: {prompt}")
        engine.unload_enhancer()
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
        f"{time.perf_counter() - started:.1f}s  ({extra}{spans})"
    )
    return 0
