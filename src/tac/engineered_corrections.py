# SPDX-License-Identifier: MIT
"""Lane EC: Engineered Corrections (canonical tac-side API).

Lane EC produces small per-frame correction deltas that ship with the
submission archive and are applied additively at inflate time. Per the
2026-04-28 skunkworks council "EC-first composition" rationale
(.omx/research/lane_g_v3_stacking_skunkworks_20260428.md), engineered
corrections compose well with every other lane because they are a pure
additive correction applied AFTER the renderer's forward pass.

Predicted band [contest-CUDA]: [0.85, 1.10] solo from a Lane A 1.15 anchor [prediction].
The downside is bounded by --max-artifact-bytes (council Quantizr).

Strict-scorer-rule: corrections.bin is DATA, not MODEL. The inflate path
loads + applies corrections without ever importing PoseNet or SegNet.
The compress-time gradient search (experiments/engineered_quant_noise.py
or experiments/precompute_gradient_corrections.py) is the only place
scorers are touched.

This module is the canonical tac-side API; the heavy compute-time
implementation lives in experiments/precompute_gradient_corrections.py
and the entire test suite (test_lane_ec_v2_greedy.py + Lane I sparse
codec tests) imports from there. We re-export the load-bearing functions
here so downstream consumers can do::

    from tac.engineered_corrections import (
        compute_per_frame_corrections,
        apply_corrections_at_inflate,
        serialize_corrections,
        load_corrections,
    )

without reaching into experiments/.

MPS-vs-CUDA engineering corrections
-----------------------------------

Per the standing CLAUDE.md "MPS auth eval is NOISE — NON-NEGOTIABLE,
HIGHEST EMPHASIS" rule and CLAUDE.md "Forbidden MPS-derived strategic
decision (the MPS-falsification trap)" forbidden pattern, every MPS forward
pass through SegNet / PoseNet / renderer drifts substantially vs CUDA
(empirically: PoseNet 23x worse, SegNet 2x worse, final score 2.5x worse
on the canonical 2026-04-25 paired measurement). The drift is calibrated
in the canonical equations registry per Catalog #344 as
``mps_drift_architecture_class_dependent_v1`` with three modeled noise
sources (``conv2d_accumulation`` / ``softmax_numerics`` / ``matmul_fp16``).

This module surfaces the three canonical engineering corrections targeting
each noise source as a single import surface alongside the rest of the
engineered-corrections public API. The heavy implementations live under
``tac.mps_diagnostic.*`` and are re-exported here so callers can do::

    from tac.engineered_corrections import (
        kahan_summation,             # conv2d_accumulation noise
        softmax_with_epsilon,        # softmax_numerics noise
        fp32_matmul,                 # matmul_fp16 noise
    )

without reaching across two namespaces. Per Catalog #344 these helpers are
the canonical_consumers of equation #2; per Catalog #287 / #323 every
result they emit is non-promotable (axis tag ``[macOS-MPS-*-PyTorch]``)
until a paired Linux x86_64 anchor lands. The corrections are COMPRESS-TIME
ONLY — the strict-scorer-rule (Catalog #6) forbids loading PoseNet / SegNet
at inflate time, and these helpers operate on the SCORER FORWARD PASS that
only the compress-time path touches.

  - ``kahan_summation(summands, dim=-1, keepdim=False)`` — Kahan-compensated
    sum reducing fp32 accumulation drift from O(eps * sqrt(N)) to O(eps).
    Anchor: slot 9 formalization predicts ~10x aggregate-weighted gap
    reduction for Conv2d-heavy networks (SegNet / PoseNet / NeRV) [prediction].
    Sister: ``kahan_conv2d`` is the full Conv2d drop-in replacement.
  - ``softmax_with_epsilon(logits, dim=-1, intermediate_dtype=fp64)`` —
    pinned-epsilon softmax with fp64 log-sum-exp stabilization. Anchor:
    slot 9 §4.2 predicts ~50% SegNet 5-class boundary-flip reduction vs
    naive ``F.softmax`` on MPS [prediction].
  - ``fp32_matmul(...)`` — context manager pinning ``torch.backends.cuda.matmul
    .allow_tf32=False`` + ``torch.backends.cudnn.allow_tf32=False`` so the
    cross-backend gap shrinks symmetrically. Anchor: slot 9 predicts 2x
    drift reduction for the matmul-accumulation component [prediction].

All three helpers preserve the input dtype and degrade gracefully (no-op
or fallback) when run on a backend that does not exhibit the noise source
they target.

Cross-references:
  - Canonical equation: ``tac.canonical_equations.builtins.build_mps_drift_architecture_class_dependent_v1``
    (equation #2 per ``feedback_canonical_equations_and_models_registry_formalization_landed_20260519.md``).
  - CLAUDE.md "MPS auth eval is NOISE — NON-NEGOTIABLE, HIGHEST EMPHASIS".
  - CLAUDE.md FORBIDDEN_PATTERNS "Forbidden MPS-derived strategic decision
    (the MPS-falsification trap)".
  - Catalog #6 (strict-scorer-rule), #287 (no-phantom-API), #323 (canonical
    Provenance), #344 (canonical equations registry).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Re-export the canonical compress-time + inflate-time helpers from
# experiments/precompute_gradient_corrections.py. That module is the
# single source of truth and has 444 lines of regression tests against
# the sparse pack/unpack format, the int8 quant scale, and the EC-V2
# greedy water-fill selector. Re-exporting keeps the API surface small
# and avoids the canonical-import-drift bug class.
from experiments.precompute_gradient_corrections import (  # noqa: E402
    apply_corrections as _apply_corrections,
    estimated_sparse_bytes,
    greedy_waterfill_correction_map,
    pack_sparse_corrections,
    sparse_to_dense_int8,
    sparsify_and_quantize,
    unpack_sparse_corrections,
)

# Re-export the canonical MPS-vs-CUDA engineering corrections from
# tac.mps_diagnostic.*. The heavy implementations + monkey-patch wiring
# live there; we surface a single canonical import name here so callers
# get the three corrections alongside the rest of the engineered-corrections
# public API. The aliasing pattern (canonical_name = tac.mps_diagnostic.X)
# matches the operator-facing names cited in the canonical equation
# registry entry #2 docstring AND the task spec; the underlying primitives
# retain their full-precision names + monkey-patch surfaces.
#
# Per Catalog #287 phantom-API guard: every aliased name resolves to a
# real importable function via this module body at import time.
from tac.mps_diagnostic.fp32_matmul_override import (  # noqa: E402
    MPS_BLAS_PREFERENCE_API_AVAILABLE,
    STRICT_FP32_FLAGS,
    enable_fp32_matmul_accumulation_strict,
    restore_fp32_matmul_accumulation_state,
    strict_fp32_matmul_accumulation as fp32_matmul,
)
from tac.mps_diagnostic.kahan_conv2d import (  # noqa: E402
    KAHAN_CONV2D_AXIS_TAG,
    KAHAN_CONV2D_EVIDENCE_GRADE,
    kahan_conv2d,
    kahan_sum as kahan_summation,
    patch_conv2d_to_kahan_for_mps_globally,
    restore_torch_conv2d,
)
from tac.mps_diagnostic.pinned_softmax import (  # noqa: E402
    PINNED_SOFTMAX_AXIS_TAG,
    PINNED_SOFTMAX_EVIDENCE_GRADE,
    patch_softmax_to_pinned_for_mps_globally,
    pinned_softmax as softmax_with_epsilon,
    restore_torch_softmax,
)

if TYPE_CHECKING:
    import torch


__all__ = [
    "compute_per_frame_corrections",
    "apply_corrections_at_inflate",
    "serialize_corrections",
    "load_corrections",
    # Re-exports kept here so callers don't need a second import line.
    "estimated_sparse_bytes",
    "greedy_waterfill_correction_map",
    "pack_sparse_corrections",
    "unpack_sparse_corrections",
    "sparse_to_dense_int8",
    "sparsify_and_quantize",
    # MPS-vs-CUDA engineering corrections (Catalog #344 equation #2 consumers).
    "kahan_summation",
    "softmax_with_epsilon",
    "fp32_matmul",
    # Heavy sister surfaces (drop-in conv2d / global patches / state).
    "kahan_conv2d",
    "patch_conv2d_to_kahan_for_mps_globally",
    "restore_torch_conv2d",
    "KAHAN_CONV2D_AXIS_TAG",
    "KAHAN_CONV2D_EVIDENCE_GRADE",
    "patch_softmax_to_pinned_for_mps_globally",
    "restore_torch_softmax",
    "PINNED_SOFTMAX_AXIS_TAG",
    "PINNED_SOFTMAX_EVIDENCE_GRADE",
    "enable_fp32_matmul_accumulation_strict",
    "restore_fp32_matmul_accumulation_state",
    "STRICT_FP32_FLAGS",
    "MPS_BLAS_PREFERENCE_API_AVAILABLE",
]


def compute_per_frame_corrections(
    gradients: np.ndarray,
    *,
    top_k_pct: float = 5.0,
    quantize_bits: int = 8,
    allocation_strategy: str = "greedy",
    rate_cap_bytes: int = 50_000,
) -> dict[str, Any]:
    """Compute per-frame correction deltas from a per-pixel scorer-gradient
    tensor.

    The gradient is the compress-time signal: ``d(score)/d(pixel)``. We
    pick the most cost-effective subset of pixels (greedy water-fill by
    default; ``fixed-budget`` for the V1 top-K behaviour) and quantize
    the kept values to int8.

    Args:
        gradients: ``(N_frames, H, W, C)`` float32 gradient tensor. ``C``
            is typically 3 (RGB) but may be 5 (per-class SegNet logit
            corrections) — the library is shape-agnostic.
        top_k_pct: fixed-budget percentage (V1 only). Ignored for greedy.
        quantize_bits: 4, 8, or 16. 8 is the default; ~30% smaller than
            16 with no measurable score impact at ``--max-delta=2``.
        allocation_strategy: ``"greedy"`` (EC-V2) or ``"fixed-budget"`` (V1).
        rate_cap_bytes: hard cap on the compressed size (greedy only).
            Council Quantizr: bounded downside.

    Returns:
        Sparse-correction dict — same shape as
        ``sparsify_and_quantize`` so downstream code (serialize, apply,
        archive build) is identical for V1 and V2.
    """
    return sparsify_and_quantize(
        np.asarray(gradients),
        top_k_pct=top_k_pct,
        quantize_bits=quantize_bits,
        allocation_strategy=allocation_strategy,
        rate_cap_bytes=rate_cap_bytes,
    )


def apply_corrections_at_inflate(
    rendered: "np.ndarray | torch.Tensor",
    corrections: dict[str, Any] | bytes | str | Path,
    *,
    alpha: float = 1.0,
) -> np.ndarray:
    """Apply pre-computed corrections to rendered frames at INFLATE time.

    Strict-scorer-rule: this function loads NO scorer. It is a pure
    additive overlay (``rendered + alpha * dequant_corrections``) clipped
    to ``[0, 255]``.

    Args:
        rendered: ``(N, H, W, C)`` float32 array of rendered frames.
        corrections: either a sparse dict (already unpacked), raw bytes,
            or a filesystem path to a packed corrections file.
        alpha: step-size multiplier (typically 1.0; smaller values blunt
            corrections during smoke tests).

    Returns:
        ``(N, H, W, C)`` float32 corrected frames.
    """
    # Normalize input into the unpacked-dict form expected by
    # apply_corrections in the canonical impl.
    if isinstance(corrections, (str, Path)):
        with open(corrections, "rb") as fh:
            corrections = unpack_sparse_corrections(fh.read(), compressed=True)
    elif isinstance(corrections, (bytes, bytearray)):
        corrections = unpack_sparse_corrections(bytes(corrections), compressed=True)

    # The canonical impl operates on numpy. If a torch tensor is passed,
    # detach + move to CPU. We do NOT introduce a torch dep at module
    # import time (apply_corrections_at_inflate runs in inflate path
    # which CLAUDE.md strict-scorer-rule says must not touch torch
    # autograd or scorer code; numpy-only is the safest path).
    try:
        import torch as _torch  # local import to avoid module-load cost

        if isinstance(rendered, _torch.Tensor):
            rendered = rendered.detach().cpu().numpy()
    except ImportError:  # pragma: no cover — torch is a hard dep here, but be safe
        pass
    rendered = np.asarray(rendered, dtype=np.float32)
    return _apply_corrections(rendered, corrections, alpha=alpha)


def serialize_corrections(
    corrections: dict[str, Any],
    output_path: str | Path,
    *,
    compression: str = "zlib",
) -> int:
    """Serialize a sparse-correction dict to a single binary file
    (``gradient_corrections.bin`` is the canonical archive name) using
    the same packed format the inflate dispatcher expects.

    Returns the on-disk byte size (so the caller can enforce
    ``--max-artifact-bytes``).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = pack_sparse_corrections(corrections, compression=compression)
    output_path.write_bytes(payload)
    return output_path.stat().st_size


def load_corrections(
    path: str | Path,
    *,
    compressed: bool = True,
) -> dict[str, Any]:
    """Load + unpack a serialized corrections file. Inverse of
    :func:`serialize_corrections`.
    """
    data = Path(path).read_bytes()
    return unpack_sparse_corrections(data, compressed=compressed)
