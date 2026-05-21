# SPDX-License-Identifier: MIT
"""Portable reusable composable primitives for MLX + PyTorch sister backends.

OVERNIGHT-WW per operator directive 2026-05-21 verbatim *"perhaps we should
start writing portable reusable composable primitives in MLX and PyTorch as
well and experimenting with cuda t4 with eval against weights and substrates
trained using MLX; is they possible?"*

This package operationalizes a canonical 2-backend (MLX on Apple Silicon Metal
+ PyTorch on CUDA / CPU / MPS) primitives layer so substrate trainers can be
authored ONCE against the portable API and dispatched to either:

- ``backend="mlx"`` — Apple Silicon Metal GPU via MLX framework (FREE local
  development; non-promotable per Catalog #1 + #192 + #317)
- ``backend="pytorch"`` — CUDA T4 / A100 / CPU / MPS via PyTorch (canonical
  contest-axis promotion path per CLAUDE.md "Submission auth eval — BOTH CPU
  AND CUDA")

The "train anywhere, eval anywhere" pattern:

    1. Author substrate against portable primitives (single source)
    2. Train via MLX backend locally on M5 Max ($0 paid GPU)
    3. Export weights via canonical ``tac.local_acceleration.mlx_to_pytorch_export``
    4. Eval on CUDA T4 via canonical ``experiments/contest_auth_eval.py``
       (paid Modal A100 / T4; contest-axis authoritative)

Per CLAUDE.md non-negotiables PRESERVED:
- **MPS auth eval is NOISE** (Catalog #1): MLX-backend scores remain
  non-promotable per Catalog #192 sister discipline at the MLX surface.
- **Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
  HARDWARE**: portable primitives enable cross-backend authoring, but
  contest-axis promotion still requires paired Linux x86_64 + NVIDIA via
  the PyTorch backend evaluated on Modal T4 / A100.
- **Apples-to-apples evidence discipline**: tests verify numerical
  equivalence MLX-vs-PyTorch within documented ε bands so backend
  swapping doesn't silently change weights.
- **Beauty, simplicity, and developer experience** (CLAUDE.md
  non-negotiable): single canonical interface per primitive; thin
  adapters; no hidden complexity; reviewable in 30 seconds per primitive.

Sister of:
- :mod:`tac.local_acceleration.mlx_integration` (canonical MLX scaffold)
- :mod:`tac.local_acceleration.mlx_to_pytorch_export` (canonical weight
  export pipeline; lands sister to this package)
- :mod:`tac.substrates.grayscale_lut.mlx_native` (canonical MLX-native
  variant of Selfcomp grayscale_lut Phase 2 BUILD per OVERNIGHT-TT)

Canonical surfaces in this package:

- :mod:`tac.portable_primitives.backend` — backend enum + capability detection
- :mod:`tac.portable_primitives.tensor` — :class:`PortableTensor` wrapper
- :mod:`tac.portable_primitives.nn` — canonical neural network primitives
  (Conv2d, Linear, LayerNorm, GELU, sigmoid, bilinear upsample) with
  sister implementations per backend
- :mod:`tac.portable_primitives.optim` — :class:`PortableAdam` optimizer
- :mod:`tac.portable_primitives.loss` — :func:`mse_loss` +
  :func:`cross_entropy_loss`

Per the canonical contract: every primitive constructed with ``backend=...``
must return numerically-equivalent results across MLX + PyTorch within ε:

- ``fp32``: ε ≤ 1e-5 (per ATOL convention; sister tests in
  ``src/tac/portable_primitives/tests/``)
- ``bf16`` / ``fp16``: ε ≤ 1e-3 (mixed-precision tolerance)

Non-equivalence is a structural bug (a future refactor would silently change
trained weights when swapping backends); tests pin the contract.
"""

from __future__ import annotations

from tac.portable_primitives.backend import (
    Backend,
    BackendUnavailableError,
    is_mlx_available,
    is_pytorch_available,
    resolve_backend,
)
from tac.portable_primitives.nn_extended import (
    PortableAvgPool2d,
    PortableBatchNorm2d,
    PortableDepthwiseConv2d,
    PortableMaxPool2d,
    silu,
)
from tac.portable_primitives.nn_attention import (
    PortableLayerScale,
    PortableMHSA,
    PortableRepMixer,
    PortableTokenMixer,
)

__all__ = [
    "SCHEMA_VERSION",
    "Backend",
    "BackendUnavailableError",
    "is_mlx_available",
    "is_pytorch_available",
    "resolve_backend",
    # MLX-ARCH-1 extended primitives (foundational ops for FastViT-T12 +
    # EfficientNet-B2-UNet architecture port).
    "PortableBatchNorm2d",
    "PortableDepthwiseConv2d",
    "PortableMaxPool2d",
    "PortableAvgPool2d",
    "silu",
    # MLX-ARCH-2 attention primitives (FastViT-T12 stages 1-4 building blocks).
    "PortableLayerScale",
    "PortableMHSA",
    "PortableTokenMixer",
    "PortableRepMixer",
]

SCHEMA_VERSION = "portable_primitives.v1"
