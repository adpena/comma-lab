# SPDX-License-Identifier: MIT
"""L-inf pose-Fisher-led allocation onto SNeRV LF coefficients (the G3 wire-in).

The proven inverse-steganalysis allocator (``tac.analysis.inverse_steganalysis_
linf_vs_l2_gate.allocate_linf_margin_budget``, §7 GREEN: L-inf beats L2 56.9% at
equal rate on the real scorer, POSE-DOMINATED) places per-element quantizer steps
by reverse-waterfill on a per-element margin budget ``rho``. That allocator is
GENERAL (any per-element ``rho`` + a target bit budget) so it wires directly onto
the SNeRV stored LF coefficients — IF we can express each LF coefficient's margin
budget.

The G3 step: the oracle ``rho_i`` lives in the PIXEL domain (P18 SegNet flip-risk +
P19 PoseNet Fisher, at scorer resolution). To allocate on LF COEFFICIENTS we must
push the pixel saliency into the LF-coefficient domain through the EXACT synthesis
adjoint of the orthonormal DWT.

The native cropped-synthesis adjoint identity (closed by construction in
``dwt.py``, rel-residual ~0):
for the orthonormal padded-canvas synthesis operator ``S`` (coeffs -> cropped
native pixels), the sensitivity of a scorer output ``Q`` to a coefficient ``c_j``
is

    dQ/dc_j = sum_i (dQ/dx_i)(dx_i/dc_j) = [S^T (dQ/dx)]_j

Because native decode crops after padded synthesis, ``S^T`` is NOT reflect-padded
encoder analysis on odd frame sizes. It is zero-embed native pixel cotangents onto
the padded canvas, then run the orthonormal DWT analysis. The detail subbands are
GENERATED (not stored), so only the LF block of that crop-aware adjoint is the
per-coefficient cost the allocator consumes — and the finest detail subbands being
a structural ``d_seg`` dead-zone (SegNet stride-2 stem) is exactly why the LF block
carries the detector-relevant mass.

We combine P18 + P19 into one per-pixel saliency (pose-weighted per the §7
POSE-DOMINATED finding), analyze it into the coefficient domain, take the LF block,
invert it to a margin budget (large coefficient saliency -> small step), and feed
``allocate_linf_margin_budget``.

[verified-against: the L-inf gate §7 GREEN advisory; the DWT adjoint exactness
(``dwt.synthesis_adjoint_residual`` rel-residual 0.0); DeepFool margin (P18) +
Fisher-Jacobian (P19) oracle (``tac.analysis.score_exact_saliency``).]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

from tac.analysis.inverse_steganalysis_linf_vs_l2_gate import (
    StepAllocation,
    allocate_linf_margin_budget,
    margin_budget_from_saliency,
)
from tac.substrates.snerv_inverse_steg_carrier.dwt import (
    DEFAULT_WAVELET,
    dwt2_native_synthesis_adjoint,
)

__all__ = [
    "POSE_WEIGHT_DEFAULT",
    "LfSaliency",
    "SnervAllocationError",
    "allocate_lf_linf",
    "push_pixel_saliency_to_lf",
]

# §7 finding: the allocation is POSE-DOMINATED. The combined per-pixel saliency
# weights the pose-Fisher surface heavily (pose's marginal value dominates at the
# frontier operating point per CLAUDE.md "SegNet vs PoseNet importance").
POSE_WEIGHT_DEFAULT: Final[float] = 1.0


class SnervAllocationError(ValueError):
    """Raised when the SNeRV LF allocation contract is violated."""


@dataclass(frozen=True)
class LfSaliency:
    """Per-LF-coefficient saliency pushed from the pixel oracle via the DWT adjoint."""

    lf_saliency: np.ndarray  # (h_lf, w_lf) per-coefficient saliency (>= 0)
    lf_shape: tuple[int, int]
    pixel_seg_mass: float  # fraction of pixel d_seg saliency mass in the LF block
    pixel_pose_mass: float  # fraction of pixel d_pose saliency mass in the LF block


def _resize_nn(field: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbour resize a pixel saliency field to the carrier frame shape.

    The oracle surfaces are at scorer resolution (384x512); the carrier frame may
    be a different (H, W). NN keeps the saliency non-negative and avoids inventing
    structure (no interpolation artifacts that would fabricate cost).
    """
    th, tw = target_hw
    fh, fw = field.shape
    if (fh, fw) == (th, tw):
        return field
    ri = (np.arange(th) * fh // th).clip(0, fh - 1)
    ci = (np.arange(tw) * fw // tw).clip(0, fw - 1)
    return field[np.ix_(ri, ci)]


def push_pixel_saliency_to_lf(
    seg_saliency_hw: np.ndarray,
    pose_saliency_hw: np.ndarray,
    *,
    carrier_hw: tuple[int, int],
    levels: int = 3,
    wavelet: str = DEFAULT_WAVELET,
    pose_weight: float = POSE_WEIGHT_DEFAULT,
) -> LfSaliency:
    """Push the pixel-domain oracle into the LF-coefficient domain (the G3 step).

    1. Resize P18 (seg flip-risk) + P19 (pose Fisher) to the carrier frame shape.
    2. Combine into one per-pixel saliency ``s = s_seg + pose_weight * s_pose``
       (POSE-DOMINATED per §7).
    3. Push ``s`` through the EXACT native cropped-synthesis adjoint: zero-embed
       the native pixel saliency on the padded canvas, then analyze with the
       orthonormal DWT. The LF block is the per-stored-coefficient cost.

    Returns the LF-domain saliency + diagnostic mass fractions (how much of the
    detector-relevant pixel mass lands in the STORED LF block vs the GENERATED
    detail — the higher the LF mass, the more the carrier's stored bits actually
    control the detector).
    """
    seg = np.abs(np.asarray(seg_saliency_hw, dtype=np.float64))
    pose = np.abs(np.asarray(pose_saliency_hw, dtype=np.float64))
    seg = _resize_nn(seg, carrier_hw)
    pose = _resize_nn(pose, carrier_hw)
    # Normalize each surface to unit mass so the pose_weight is meaningful.
    seg_n = seg / (seg.sum() + 1e-30)
    pose_n = pose / (pose.sum() + 1e-30)
    combined = seg_n + pose_weight * pose_n  # per-pixel saliency

    # Push saliency through the native synthesis adjoint. We take |coeff| because
    # the saliency is a magnitude (the allocator wants a non-negative
    # per-coefficient cost). The LF block is coeffs[0].
    def _lf_block(field: np.ndarray) -> np.ndarray:
        pyr = dwt2_native_synthesis_adjoint(field, levels=levels, wavelet=wavelet)
        return np.abs(pyr.lf)

    lf_combined = _lf_block(combined)

    # Mass fractions: how much detector saliency lands in the stored LF block.
    # (For an orthonormal transform total energy is preserved; the LF block's
    # share of |coeff| is the meaningful "stored-bit leverage" diagnostic.)
    seg_pyr = dwt2_native_synthesis_adjoint(seg_n, levels=levels, wavelet=wavelet)
    pose_pyr = dwt2_native_synthesis_adjoint(pose_n, levels=levels, wavelet=wavelet)
    seg_total = float(np.abs(seg_pyr.lf).sum()) + sum(
        float(np.abs(d[0]).sum() + np.abs(d[1]).sum() + np.abs(d[2]).sum())
        for d in seg_pyr.details
    )
    pose_total = float(np.abs(pose_pyr.lf).sum()) + sum(
        float(np.abs(d[0]).sum() + np.abs(d[1]).sum() + np.abs(d[2]).sum())
        for d in pose_pyr.details
    )
    seg_mass = float(np.abs(seg_pyr.lf).sum()) / (seg_total + 1e-30)
    pose_mass = float(np.abs(pose_pyr.lf).sum()) / (pose_total + 1e-30)

    return LfSaliency(
        lf_saliency=lf_combined,
        lf_shape=(int(lf_combined.shape[0]), int(lf_combined.shape[1])),
        pixel_seg_mass=seg_mass,
        pixel_pose_mass=pose_mass,
    )


def allocate_lf_linf(
    lf_saliency: LfSaliency,
    *,
    target_bits: float,
    min_step: float = 0.5,
    max_step: float | None = None,
    eps: float = 1e-8,
) -> StepAllocation:
    """Wire the proven L-inf allocator onto the SNeRV LF coefficients.

    ``lf_saliency`` is the per-coefficient detector saliency (high = the detector
    cares = must quantize FINELY = small step). We invert it to a margin budget
    ``rho = 1/(saliency + eps)`` via the canonical ``margin_budget_from_saliency``
    (so high saliency -> small rho -> small step) and feed
    ``allocate_linf_margin_budget`` with the SAME ``target_bits`` as the L2
    baseline (the fairness invariant). The realized step map is one quantizer step
    per stored LF coefficient — the inverse-steganalysis allocation in the carrier
    domain.

    Returns the canonical :class:`StepAllocation` (steps reshape to ``lf_shape``).
    """
    sal = np.asarray(lf_saliency.lf_saliency, dtype=np.float64).reshape(-1)
    rho = margin_budget_from_saliency(sal, eps=eps)
    alloc = allocate_linf_margin_budget(
        rho,
        target_bits=target_bits,
        min_step=min_step,
        max_step=max_step,
        fairness_direction="disadvantage_linf",  # NO-FAKE: L-inf spends >= L2 bits
    )
    return alloc
