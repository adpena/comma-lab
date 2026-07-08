"""GPU (MLX) verdict device + CPU-torch anchor drift instrument — PURE helpers.

Operator directive 2026-07-08: *"Shouldn't the verdict be run on gpu now that it's
deterministic and way faster? Even if launched in a separate process temporarily the
contention is probably worth it."* Design law (HYBRID, main-orchestrator binding): a
GPU verdict at the FAST cadence + a CPU-torch ANCHOR verdict at a SLOW cadence. The
CPU-torch verdict is the independent **positive-control sentinel** (training gradients
already flow through the MLX scorer, so a GPU-only verdict would fold the instrument
into what it measures) AND every prior trajectory baseline is a CPU-verdict number
(comparability). This module holds the DEVICE-FREE parts — the anchor-cadence math, the
argmax flip-disagreement counter, and the paired drift-row schema — so they are unit
testable at $0 WITHOUT importing torch or MLX. The heavy per-chunk scorer forwards live
next to the CPU primitives in ``experiments/train_witness_realized_through_R_mlx.py``
(``gpu_verdict_d_seg_argmax_batch`` / ``gpu_verdict_d_pose_batch`` /
``gpu_verdict_dseg_dpose_chunked`` / ``paired_anchor_verdict``).

AUTHORITY (CLAUDE.md NON-NEGOTIABLE — MPS/MLX is NEVER a score): BOTH verdict flavours
are ADVISORY, NON-PROMOTABLE. The GPU verdict rows carry ``[macOS-MLX advisory]``; the
anchor's CPU rows carry ``[macOS-CPU advisory]``. Only a byte-closed ``upstream/evaluate.py``
n600 exact row (contest-CPU/CUDA) moves the pointer. This module never claims a score.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# ── canonical advisory axis tags (NON-PROMOTABLE per CLAUDE.md "MPS/MLX NEVER a score") ──
AXIS_TAG_GPU = "[macOS-MLX advisory]"
AXIS_TAG_CPU = "[macOS-CPU advisory]"

__all__ = [
    "AXIS_TAG_CPU",
    "AXIS_TAG_GPU",
    "build_paired_drift_row",
    "flip_disagreement_count",
    "gpu_verdict_conflicts",
    "should_anchor",
]


def gpu_verdict_conflicts(*, async_verdict: bool, curriculum_nucleus_guard: bool,
                          ladder_island_homotopy: bool) -> list[str]:
    """Return the FAIL-CLOSED conflicts that forbid ``--verdict-device gpu``.

    The GPU verdict is an ADVISORY monitor and must NOT (a) run MLX off the main thread
    (it would race the training GPU stream — the async worker is pure numpy+torch for
    exactly this reason), or (b) feed a training-affecting controller on MLX numbers. The
    nucleus guard feeds the CE->tau trigger; the ladder homotopy feeds per-class λ — both
    consume the verdict argmax INTO training, so they stay on CPU authority. Empty list =>
    the gpu monitor is admissible. Pure predicate (no argparse/trainer import) so the guard
    is unit-testable at $0."""
    conflicts: list[str] = []
    if async_verdict:
        conflicts.append("--async-verdict (MLX off the main thread races the training GPU stream)")
    if curriculum_nucleus_guard:
        conflicts.append("--curriculum-nucleus-guard (feeds the CE->tau trigger into training)")
    if ladder_island_homotopy:
        conflicts.append("--ladder-island-homotopy (feeds per-class λ into training)")
    return conflicts


def should_anchor(verdict_count: int, anchor_every: int) -> bool:
    """Fire the CPU-torch ANCHOR on every ``anchor_every``-th GPU verdict.

    ``verdict_count`` is the 1-based count of GPU verdicts emitted so far (increment
    BEFORE calling). ``anchor_every <= 0`` disables the anchor (GPU-only monitoring, no
    positive-control) — the default. ``anchor_every == 1`` anchors every verdict.

    Pure integer predicate (no device, no state) so the cadence is unit-testable at $0.
    """
    n = int(anchor_every)
    if n <= 0:
        return False
    return int(verdict_count) % n == 0


def flip_disagreement_count(cpu_realized: Any, gpu_realized: Any) -> int:
    """Total pixels where the CPU-torch argmax and the MLX-GPU argmax DISAGREE.

    Accepts either a stacked ``(N,h,w)`` array or a per-pair list of ``(h,w)`` maps for
    EACH side (they must be the same shape/order). This is the core drift signal: if the
    GPU forward is deterministic AND agrees with CPU, this count is small/stable; a large
    or drifting count means the GPU verdict cannot stand in for the CPU authority.

    Pure numpy (no torch/MLX) so it is testable at $0 on synthetic argmax maps.
    """
    cpu_list = _as_pair_list(cpu_realized)
    gpu_list = _as_pair_list(gpu_realized)
    if len(cpu_list) != len(gpu_list):
        raise ValueError(
            f"flip_disagreement_count: pair-count mismatch cpu={len(cpu_list)} gpu={len(gpu_list)}")
    total = 0
    for c, g in zip(cpu_list, gpu_list):
        c_a = np.asarray(c)
        g_a = np.asarray(g)
        if c_a.shape != g_a.shape:
            raise ValueError(
                f"flip_disagreement_count: per-pair shape mismatch {c_a.shape} vs {g_a.shape}")
        total += int(np.count_nonzero(c_a != g_a))
    return total


def _as_pair_list(realized: Any) -> list:
    """Normalise a stacked ``(N,h,w)`` array or a per-pair list to a per-pair list."""
    if isinstance(realized, (list, tuple)):
        return list(realized)
    arr = np.asarray(realized)
    if arr.ndim == 3:
        return [arr[i] for i in range(arr.shape[0])]
    if arr.ndim == 2:
        return [arr]
    raise ValueError(f"_as_pair_list: expected (N,h,w) or (h,w) or list, got ndim={arr.ndim}")


def build_paired_drift_row(paired: dict[str, Any], *, epoch: int, verdict_batch: int) -> dict[str, Any]:
    """Assemble the printable paired DRIFT row from a ``paired_anchor_verdict`` result.

    ``paired`` must carry ``d_seg_cpu``/``d_seg_gpu``/``d_pose_cpu``/``d_pose_gpu``/
    ``argmax_flip_disagreement_count``/``max_abs_dpose_delta`` (all device-computed). The
    row is OBSERVABILITY-ONLY (a companion to the ``{stage:verdict}`` row) — it is never
    read back into training/parity/resume/history/result.json, so it is additive-safe.
    Both device numbers carry their NON-PROMOTABLE axis tags. Deltas are gpu-minus-cpu.
    """
    ds_cpu = float(paired["d_seg_cpu"])
    ds_gpu = float(paired["d_seg_gpu"])
    dp_cpu = float(paired["d_pose_cpu"])
    dp_gpu = float(paired["d_pose_gpu"])
    return {
        "stage": "verdict_anchor",
        "epoch": int(epoch),
        "verdict_batch": int(verdict_batch),
        # advisory numbers, both axes labelled (NON-PROMOTABLE)
        "d_seg_gpu": round(ds_gpu, 6),
        "d_seg_cpu": round(ds_cpu, 6),
        "d_pose_gpu": round(dp_gpu, 8),
        "d_pose_cpu": round(dp_cpu, 8),
        # the drift instrument
        "d_seg_delta": round(ds_gpu - ds_cpu, 8),
        "d_pose_delta": round(dp_gpu - dp_cpu, 10),
        "argmax_flip_disagreement_count": int(paired["argmax_flip_disagreement_count"]),
        "max_abs_dpose_delta": float(paired["max_abs_dpose_delta"]),
        # authority (both flavours advisory; the pointer moves only via byte-closed exact eval)
        "axis_gpu": AXIS_TAG_GPU,
        "axis_cpu": AXIS_TAG_CPU,
        "promotable": False,
    }
