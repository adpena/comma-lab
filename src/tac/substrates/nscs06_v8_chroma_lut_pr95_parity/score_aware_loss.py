# SPDX-License-Identifier: MIT
"""NSCS06 v8 chroma_lut + cls_stream PR-95-parity score-aware Lagrangian.

Per HNeRV parity discipline lessons L1 (substrate is score-aware: gradient
through SegNet/PoseNet on contest video, NOT extracted masks / NOT L2/KL
on raw frames / NOT synthetic data) + L6 (score-domain Lagrangian
``alpha * B(theta)/N + beta * d_seg + gamma * sqrt(d_pose)`` via actual
scorer; rel_err^2-as-objective FORBIDDEN per
``feedback_three_lossy_anchors_show_rel_err_squared_objective_falsified_20260508``)
+ L8 (eval-roundtrip + differentiable scorer-preprocess; uint8 bottleneck
384 -> 874 -> uint8 -> 384 MUST be simulated in proxy loss).

Routes through canonical helper
``tac.substrates.score_aware_common.score_pair_components`` per
Catalog #164 (canonical scorer-preprocess routing before scorer forward)
+ Catalog #226 (canonical gate_auth_eval_call helper routing for auth eval).

Per CLAUDE.md "eval_roundtrip -- NON-NEGOTIABLE, HIGHEST EMPHASIS": eval_roundtrip
defaults to True; `apply_eval_roundtrip_during_training` + `patch_upstream_yuv6_globally`
+ `load_differentiable_scorers` (canonical 3-call pattern for differentiable
score-aware training).

Per CLAUDE.md "EMA -- NON-NEGOTIABLE, HIGHEST EMPHASIS": NSCS06 v8 has NO
learned weights (closed-form LUT derivation from GT pixels via per-bin median).
EMA is therefore N/A for this substrate. The score-aware loop is COMPRESS-time
LUT-quantization-loss only; there is no per-epoch optimizer step.

# AUTOCAST_FP16_WAIVED:closed_form_lut_no_neural_training_loop_no_autocast_needed
# TF32_WAIVED:closed_form_lut_no_neural_matmul
# TORCH_COMPILE_WAIVED:closed_form_lut_no_neural_forward
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np

__all__ = [
    "SCORE_AWARE_LAGRANGIAN_ALPHA",
    "SCORE_AWARE_LAGRANGIAN_BETA",
    "SCORE_AWARE_LAGRANGIAN_GAMMA",
    "score_aware_lagrangian_loss",
    "score_pair_components_pr95_parity",
]


# Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable: canonical
# contest formula coefficients. score = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes / 37545489
# The score-aware Lagrangian targets each term separately.
SCORE_AWARE_LAGRANGIAN_ALPHA: Final[float] = 25.0
"""Rate-term coefficient: 25 * archive_bytes / 37545489."""

SCORE_AWARE_LAGRANGIAN_BETA: Final[float] = 100.0
"""SegNet distortion coefficient: 100 * d_seg."""

SCORE_AWARE_LAGRANGIAN_GAMMA: Final[float] = 1.0
"""PoseNet distortion coefficient: 1 * sqrt(10 * d_pose) = sqrt(10) * sqrt(d_pose)."""


def score_pair_components_pr95_parity(
    *,
    pred_pair_btchw: "Any",
    target_pair_btchw: "Any",
    seg_scorer: "Any",
    pose_scorer: "Any",
    apply_eval_roundtrip: bool = True,
    use_canonical_helper: bool = True,
) -> dict[str, "Any"]:
    """Compute per-pair (seg, pose) score components via canonical helper.

    Per HNeRV parity L1 + L6 + L8: routes through canonical
    ``tac.substrates.score_aware_common.score_pair_components`` per
    Catalog #164. The helper internally:

    1. Applies eval_roundtrip (uint8 bottleneck simulation) per L8.
    2. Calls ``self.<scorer>.preprocess_input(...)`` BEFORE bare forward per Catalog #164.
    3. SegNet input: 4D ``(B, C, H, W)`` with last-frame slice via preprocess.
    4. PoseNet input: 4D ``(B, T*6, H/2, W/2)`` via differentiable ``rgb_to_yuv6``.
    5. Returns ``{"d_seg": ..., "d_pose": ..., "score_seg": ..., "score_pose": ...}``.

    Args:
        pred_pair_btchw: predicted (B, T, C, H, W) tensor (T=2 per contest pair)
        target_pair_btchw: target (B, T, C, H, W) tensor
        seg_scorer: SegNet model with ``.preprocess_input(...)`` method
        pose_scorer: PoseNet model with ``.preprocess_input(...)`` method
        apply_eval_roundtrip: whether to simulate uint8 384->874->uint8->384 (L8)
        use_canonical_helper: route through canonical helper (recommended)

    Returns:
        dict with per-component score deltas + raw forward outputs
    """
    if not use_canonical_helper:
        raise RuntimeError(
            "score_pair_components_pr95_parity ONLY supports use_canonical_helper=True "
            "per Catalog #164 sister discipline + HNeRV parity L8"
        )
    # Lazy-import the canonical helper so this module loads cleanly without torch
    # for inflate-only contexts (per HNeRV parity L9 runtime closure discipline).
    from tac.substrates.score_aware_common import score_pair_components
    components = score_pair_components(
        pred_pair_btchw=pred_pair_btchw,
        target_pair_btchw=target_pair_btchw,
        seg_scorer=seg_scorer,
        pose_scorer=pose_scorer,
        apply_eval_roundtrip=apply_eval_roundtrip,
    )
    return components


def score_aware_lagrangian_loss(
    *,
    components: dict[str, "Any"],
    archive_bytes: int,
    sample_count: int = 600,
    rate_denom_bytes: int = 37_545_489,
) -> "Any":
    """Compute the canonical score-domain Lagrangian per HNeRV parity L6.

    Formula:
        score = ALPHA * archive_bytes / rate_denom_bytes
              + BETA * d_seg
              + GAMMA * sqrt(GAMMA_INNER * d_pose)

    where ALPHA = 25.0, BETA = 100.0, GAMMA = 1.0, GAMMA_INNER = 10.0 (canonical
    contest formula constants per ``src/tac/score_composition/__init__.py``).

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag": this returns
    a SCALAR Lagrangian for training (NOT a contest-score claim). Per Catalog
    #287 + #323: callers MUST tag results with canonical Provenance before
    publishing to any score-claim surface.

    Args:
        components: dict from ``score_pair_components_pr95_parity``
            with keys ``d_seg`` + ``d_pose``
        archive_bytes: total archive byte count (canonical rate-term numerator)
        sample_count: number of pairs evaluated (used by some normalizations)
        rate_denom_bytes: canonical contest formula denominator (37545489)

    Returns:
        Scalar torch.Tensor Lagrangian (when components from canonical helper)
        OR scalar float when called with numpy inputs (for unit testing)

    [prediction; canonical-equation-26-grounded; per-substrate-symposium-pending]
    """
    if "d_seg" not in components or "d_pose" not in components:
        raise ValueError(f"components must have d_seg + d_pose keys; got {sorted(components)}")
    d_seg = components["d_seg"]
    d_pose = components["d_pose"]
    rate_term = SCORE_AWARE_LAGRANGIAN_ALPHA * float(archive_bytes) / float(rate_denom_bytes)
    seg_term = SCORE_AWARE_LAGRANGIAN_BETA * d_seg
    # Handle both torch and numpy backends. Lazy-import torch to keep this module
    # importable in inflate-only contexts per HNeRV parity L9.
    pose_inner = 10.0 * d_pose
    try:
        import torch
        if isinstance(d_pose, torch.Tensor):
            # Clamp to avoid sqrt(0) gradient explosion at perfect-pose case
            pose_inner_clamped = torch.clamp(pose_inner, min=1e-12)
            pose_term = SCORE_AWARE_LAGRANGIAN_GAMMA * torch.sqrt(pose_inner_clamped)
        else:
            pose_inner_clamped = max(float(pose_inner), 1e-12)
            pose_term = SCORE_AWARE_LAGRANGIAN_GAMMA * float(np.sqrt(pose_inner_clamped))
    except ImportError:
        pose_inner_clamped = max(float(pose_inner), 1e-12)
        pose_term = SCORE_AWARE_LAGRANGIAN_GAMMA * float(np.sqrt(pose_inner_clamped))
    return rate_term + seg_term + pose_term
