# SPDX-License-Identifier: Apache-2.0
"""Runtime low-rank adapters for the LTX-2.5 DiT.

An adapter is applied as y += strength * (x A^T) B^T next to the base GEMM
instead of being fused into the weights. The base checkpoint stays exactly
the official one, adapters attach and detach per stage or per request with no
reload, and the rank-450 distilled adapter costs 8.3 GiB instead of a second
39 GiB transformer. Official files use `diffusion_model.<linear>.lora_{A,B}.weight`
with alpha == rank, so the scale is the strength alone.
"""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from slimserve.video.ltx25 import checkpoints
from slimserve.video.ltx25.dit import LTX25DiT

PREFIX = "diffusion_model."


class Lora:
    def __init__(self, component: str, root: Path | None = None):
        self.path = checkpoints.path_of(component, root)
        self.pairs: dict[str, tuple[mx.array, mx.array]] = {}

    def load(self) -> "Lora":
        if self.pairs:
            return self
        header = checkpoints.read_header(self.path)
        rank, alpha = header.metadata.get("lora_rank"), header.metadata.get("lora_alpha")
        if rank != alpha:
            raise ValueError(f"{self.path.name}: lora_alpha {alpha} != lora_rank {rank}; scaling not implemented")
        raw = checkpoints.cast_operands(checkpoints.load_raw(self.path))
        for key in [k for k in raw if k.endswith(".lora_A.weight")]:
            name = key[len(PREFIX) : -len(".lora_A.weight")]
            self.pairs[name] = (raw[key], raw[key.replace(".lora_A.", ".lora_B.")])
        return self

    def attach(self, dit: LTX25DiT, strength: float = 1.0) -> None:
        known = {k[: -len(".weight")] for k in dit.w if k.endswith(".weight")} | set(dit.split_k)
        missing = [n for n in self.pairs if n not in known]
        if missing:
            raise KeyError(f"{self.path.name}: {len(missing)} adapter targets not in the DiT, e.g. {missing[:3]}")
        for name, (a, b) in self.pairs.items():
            dit.lora.setdefault(name, []).append((a, b, strength))

    def detach(self, dit: LTX25DiT) -> None:
        for name, (a, _b) in self.pairs.items():
            terms = [t for t in dit.lora.get(name, []) if t[0] is not a]
            if terms:
                dit.lora[name] = terms
            else:
                dit.lora.pop(name, None)
