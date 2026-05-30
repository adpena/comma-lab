# SPDX-License-Identifier: MIT
"""PoseNet MAE-V (Multi-Axis Variant) canonical surrogate package.

The canonical PoseNet equivalent of the existing Hinton-distilled SegNet
surrogate at ``tac.residual_basis.hinton_distilled_scorer_surrogate``,
closing the canonical-scorer-surrogate gap surfaced 2026-05-30 per the
operator-validated Yousfi-voice critique: PoseNet (FastViT-T12 on
12-channel YUV6) is 2.71x more marginally important at the PR106
frontier per CLAUDE.md "SegNet vs PoseNet importance" empirical table,
but the only existing surrogate sister was for SegNet.

**Architecture**: a NUMPY-PORTABLE Hinton-distilled student head that
maps a decoded frame pair ``(rgb_0, rgb_1)`` each ``(B, H, W, 3)`` in
``[0, 1]`` to a ``(B, pose_dims=6)`` pose vector via the canonical
coarse-spatial-pool + linear-projection architecture from
``tac.substrates.hinton_distilled_scorer_surrogate.LearnablePoseStudentHead``
— but EXPORTED to numpy so per-byte gradient extraction tooling +
cathedral consumers can consult the surrogate WITHOUT bootstrapping the
MLX runtime.

**Training pipeline** (canonical, NOT redone here per
UNIQUE-AND-COMPLETE-PER-METHOD; this package wraps the existing
primitives):

  1. Build :class:`RealPoseNetTeacherCache` from contest video pairs via
     ``tac.substrates.hinton_distilled_scorer_surrogate.RealPoseNetTeacherCache``
     (REAL PoseNet inference, gradient-free, indexed by pair).
  2. Train a :class:`LearnablePoseStudentHead` against the teacher cache
     via ``pose_distillation_mse_loss`` per the canonical sister training
     loop at ``tac.local_acceleration.pr95_hnerv_mlx_long_training`` /
     ``tac.substrates._shared.mlx_score_aware``.
  3. Export the trained MLX weights via
     :func:`PoseNetMaeVSurrogate.from_mlx_student_head` (or directly via
     :func:`PoseNetMaeVSurrogate.from_numpy_weights` for hand-built
     fixtures).
  4. Verify forward parity vs the canonical PoseNet via
     :func:`compute_forward_parity_max_abs` per Catalog #1265 contest-
     equivalence drift discipline (target max_abs < 3e-5 per Slot 1303
     T3 GRAND COUNCIL ULP boundary).

**Per-byte gradient extraction** (the canonical purpose of this
deployable surrogate): :func:`compute_per_byte_pose_jacobian` decodes
the archive bytes via the caller-supplied decoder, runs the surrogate
forward on each frame pair, and returns a per-byte estimate of
``d(pose_dist)/d(byte_i)`` via finite-difference (1-bit perturbation
per byte). This is sister of the canonical per-byte master-gradient
extraction at ``tools/extract_master_gradient.py`` but RUNS in numpy +
operates on a CHEAP surrogate — cost-discrimination-only, NOT
contest-axis authoritative.

**Catalog #341 Tier A canonical-routing markers** are emitted in every
returned :class:`PoseJacobianResult` per CLAUDE.md "MPS auth eval is
NOISE" + Catalog #192 + Catalog #317: ``score_claim=False`` +
``promotable=False`` + ``axis_tag="[macOS-CPU advisory]"``.

**Canonical Provenance per Catalog #323** propagates source-artifact
SHA256 + training-anchor reference + canonical-helper-invocation +
``evidence_grade="predicted"`` so downstream cathedral consumers
inherit the non-promotability invariant.

**Slot EEE NO FAKE IMPLEMENTATIONS gate** (per the 2026-05-29 honesty
discipline + CLAUDE.md HIGHEST-EMPHASIS non-negotiable 2026-05-30):
this surrogate genuinely runs the canonical PR95 coarse-spatial-pool +
linear-projection forward; it is NOT a stub that returns canonical
markers. Tests verify (a) different weights produce different outputs,
(b) the forward matches a hand-computed reference, (c) per-byte
Jacobian on a perturbed byte produces a finite nonzero estimate.

**Public API**:

* :class:`PoseNetMaeVSurrogate` — numpy-portable deployable surrogate.
* :func:`compute_per_byte_pose_jacobian` — finite-difference per-byte
  gradient extraction.
* :func:`compute_forward_parity_max_abs` — Catalog #1265 sister
  drift-discipline helper.
* :func:`build_surrogate_from_numpy_weights` — convenience constructor.
* :class:`PoseJacobianResult` — typed per-byte Jacobian result.
"""
from __future__ import annotations

from tac.scorer_surrogate.posenet_mae_v.surrogate import (
    CANONICAL_POSE_DIMS,
    CANONICAL_POSE_POOL_GRID,
    PARITY_MAX_ABS_CANONICAL_THRESHOLD,
    PoseJacobianResult,
    PoseNetMaeVSurrogate,
    PoseNetMaeVSurrogateInvalidError,
    build_canonical_provenance_for_surrogate,
    build_surrogate_from_numpy_weights,
    compute_forward_parity_max_abs,
    compute_per_byte_pose_jacobian,
)

__all__ = [
    "CANONICAL_POSE_DIMS",
    "CANONICAL_POSE_POOL_GRID",
    "PARITY_MAX_ABS_CANONICAL_THRESHOLD",
    "PoseJacobianResult",
    "PoseNetMaeVSurrogate",
    "PoseNetMaeVSurrogateInvalidError",
    "build_canonical_provenance_for_surrogate",
    "build_surrogate_from_numpy_weights",
    "compute_forward_parity_max_abs",
    "compute_per_byte_pose_jacobian",
]
