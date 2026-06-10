# SPDX-License-Identifier: MIT
"""Consumer hooks for the evaluator invisibility basis (task #47).

Wires the certified tier-1 basis (``tac.optimization.evaluator_invisibility_basis``)
into the two named consumers WITHOUT editing the just-rewired
``lf_payload_rate_distortion`` module (it is imported, not modified):

  (a) the #46 LF rate-distortion waterfiller — a ``null_basis`` action builder
      that produces a ``CandidateActionEvaluation`` declaring CERTIFIED zero
      distortion (``est_delta_d_seg = est_delta_d_pose = 0.0``) for any bytes that
      provably encode tier-1-invisible camera pixels.  Such bytes are free: their
      perturbation produces a bit-identical scorer input, so coarsening / dropping
      / re-painting them costs exactly zero distortion (not estimated — derived).

  (b) the PR110++ atom generator — a helper exposed via the existing
      ``tac.null_space_exploiter`` surface (extended per the orphan-inventory REUSE
      plan) that yields per-pixel certified-free perturbation directions for the
      frame1-mode family (perturb along tier-1 directions = free repair room).

Both consumers cite the basis by sha (the artifact's certified header sha).

CLAUDE.md compliance: ``planning_only_no_score_claim`` / ``promotable=false`` /
``no_mps_authoritative`` / ``no_tmp_paths``.  A null-basis recode action still
requires the waterfiller's exact re-measure surface before any score claim — the
certified zero is a CORRECTNESS fact about the scorer INPUT, not a measured score.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.optimization.evaluator_invisibility_basis import (
    EvaluatorInvisibilityBasis,
    EvaluatorInvisibilityBasisError,
    Tier1ResizeNullSpace,
)

# Imported (NOT edited) from the just-rewired waterfiller.
from tac.optimization.lf_payload_rate_distortion import (
    ACTION_DROP,
    ACTION_RECODE,
    BaselineScoreTerms,
    CandidateActionEvaluation,
    PayloadSection,
)


# ---------------------------------------------------------------------------
# Consumer (a): the #46 waterfiller null_basis action builder.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SectionTier1Accounting:
    """How many of a payload section's bytes provably encode tier-1-invisible
    camera pixels (the certified free-byte count for that section).

    ``pixel_index_map`` maps each of the section's bytes (or coefficient groups) to
    a ``(frame_role, channel, row, col)`` camera location.  A byte is tier-1-free
    iff its pixel is certified invisible (resize zero-weight, OR frame0 for SegNet
    when the byte ONLY affects the SegNet-read frame).
    """

    section_name: str
    n_section_bytes: int
    n_tier1_free_bytes: int
    basis_header_sha256: str

    @property
    def tier1_free_fraction(self) -> float:
        return (
            self.n_tier1_free_bytes / self.n_section_bytes
            if self.n_section_bytes
            else 0.0
        )

    def to_row(self) -> dict[str, Any]:
        return {
            "section_name": self.section_name,
            "n_section_bytes": int(self.n_section_bytes),
            "n_tier1_free_bytes": int(self.n_tier1_free_bytes),
            "tier1_free_fraction": self.tier1_free_fraction,
            "basis_header_sha256": self.basis_header_sha256,
            "evidence_grade": "mathematical-derivation",
            "promotable": False,
        }


def count_section_tier1_free_bytes(
    section: PayloadSection,
    basis: EvaluatorInvisibilityBasis,
    *,
    pixel_locations: Iterable[tuple[str, int, int, int]],
    basis_header_sha256: str,
) -> SectionTier1Accounting:
    """Count how many of ``section``'s bytes encode tier-1-invisible pixels.

    ``pixel_locations`` is the section's byte->pixel map: an iterable of
    ``(frame_role, channel, row, col)`` for each byte (or coefficient).  A byte is
    tier-1-free when its pixel is certified invisible to BOTH heads (the
    ``tier1_pixel_invisible`` query).  frame0 bytes that ONLY feed SegNet are also
    free (the corollary), but the conservative default requires BOTH-head
    invisibility (resize zero-weight) so a byte that secretly feeds PoseNet via
    frame0 is not mis-counted as free.
    """
    n_free = 0
    n_total = 0
    for (frame_role, channel, row, col) in pixel_locations:
        n_total += 1
        try:
            if basis.tier1_pixel_invisible(frame_role, int(row), int(col), int(channel)):
                n_free += 1
        except EvaluatorInvisibilityBasisError:
            # out-of-bounds / bad role => not certifiable => not free (fail closed)
            continue
    return SectionTier1Accounting(
        section_name=section.name,
        n_section_bytes=n_total,
        n_tier1_free_bytes=n_free,
        basis_header_sha256=basis_header_sha256,
    )


def build_null_basis_recode_action(
    section: PayloadSection,
    accounting: SectionTier1Accounting,
    baseline: BaselineScoreTerms,
    *,
    free_byte_floor: int,
) -> CandidateActionEvaluation | None:
    """A waterfiller action that re-encodes a section's TIER-1-FREE bytes to a
    smaller floor at CERTIFIED zero distortion.

    Unlike ``build_recode_action`` (which assumes a lossless recode is zero-cost),
    this action's zero distortion is CERTIFIED by the closed-form basis: the bytes
    it frees encode camera pixels whose perturbation produces a bit-identical
    scorer input.  So coarsening / re-painting them is provably zero-distortion —
    the strongest possible free-bytes branch of the reverse waterfill.

    Returns ``None`` when the section has fewer tier-1-free bytes than
    ``free_byte_floor`` (no worthwhile certified-free opportunity).
    """
    freed = int(accounting.n_tier1_free_bytes) - int(free_byte_floor)
    if freed <= 0:
        return None
    return CandidateActionEvaluation(
        action_id=f"{section.name}::null_basis_recode(tier1_free={accounting.n_tier1_free_bytes})",
        action_kind=ACTION_RECODE,
        section_name=section.name,
        # CERTIFIED zero — derived, not estimated. The bytes are tier-1-invisible.
        est_delta_d_seg=0.0,
        est_delta_d_pose=0.0,
        delta_bytes=-freed,
        atlas_scope_valid=True,
        scope_reason=(
            "CERTIFIED tier-1 invisibility (resize zero-weight pixels): freeing "
            "these bytes yields a bit-identical scorer input (residual==0.0); "
            f"basis sha {accounting.basis_header_sha256[:12]}"
        ),
        baseline=baseline,
    )


def build_null_basis_drop_action(
    section: PayloadSection,
    accounting: SectionTier1Accounting,
    baseline: BaselineScoreTerms,
) -> CandidateActionEvaluation | None:
    """A DROP action for a section that is ENTIRELY tier-1-free (every byte encodes
    a certified-invisible pixel).  Dropping it frees all its bytes at certified
    zero distortion.  Returns ``None`` unless the section is 100% tier-1-free."""
    if accounting.n_section_bytes == 0:
        return None
    if accounting.n_tier1_free_bytes != accounting.n_section_bytes:
        return None
    return CandidateActionEvaluation(
        action_id=f"{section.name}::null_basis_drop(100%_tier1_free)",
        action_kind=ACTION_DROP,
        section_name=section.name,
        est_delta_d_seg=0.0,
        est_delta_d_pose=0.0,
        delta_bytes=-int(accounting.n_section_bytes),
        atlas_scope_valid=True,
        scope_reason=(
            "section is 100% CERTIFIED tier-1-invisible; dropping it is "
            f"zero-distortion (basis sha {accounting.basis_header_sha256[:12]})"
        ),
        baseline=baseline,
    )


# ---------------------------------------------------------------------------
# Consumer (b): the PR110++ atom generator — certified free perturbation room.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CertifiedFreePerturbationAtom:
    """A PR110++ frame1-mode atom: a certified-free per-pixel perturbation.

    Perturbing along this direction (a single tier-1-invisible camera pixel, any
    amplitude up to uint8 clipping) is FREE repair room: it can carry payload /
    repair signal at zero distortion cost because the scorer input is unchanged.
    """

    frame_role: str
    channel: int
    row: int
    col: int
    max_amplitude: float  # the clipping-bounded amplitude room (uint8 -> 255)
    basis_header_sha256: str
    evidence_grade: str = "mathematical-derivation"
    promotable: bool = False

    def as_camera_delta(self, base_value: float) -> float:
        """The maximum signed amplitude usable at this pixel given its current
        value (clipping to [0,255]); the larger of headroom up or down."""
        up = 255.0 - base_value
        down = base_value
        return max(up, down)


def generate_pr110_certified_free_atoms(
    basis: EvaluatorInvisibilityBasis,
    *,
    frame_role: str,
    channels: Iterable[int] = (0, 1, 2),
    basis_header_sha256: str,
    max_atoms: int | None = None,
) -> list[CertifiedFreePerturbationAtom]:
    """Yield the PR110++ frame1-mode certified-free perturbation atoms.

    One atom per (channel, zero-weight pixel).  These are the free directions a
    PR110++ frame1-mode generator can perturb along to carry payload / repair at
    zero scorer cost.  ``max_atoms`` caps the count (the full set is ~692K per
    frame; consumers usually want a ranked subset).
    """
    if frame_role not in ("frame0", "frame1"):
        raise EvaluatorInvisibilityBasisError("frame_role must be frame0/frame1")
    t1: Tier1ResizeNullSpace = basis.tier1_resize
    mask = t1.zero_weight_pixel_mask()
    rr, cc = np.where(mask)
    atoms: list[CertifiedFreePerturbationAtom] = []
    for ch in channels:
        if not (0 <= int(ch) < 3):
            raise EvaluatorInvisibilityBasisError("channel must be in [0,3)")
        for r, c in zip(rr.tolist(), cc.tolist(), strict=True):
            atoms.append(
                CertifiedFreePerturbationAtom(
                    frame_role=frame_role,
                    channel=int(ch),
                    row=int(r),
                    col=int(c),
                    max_amplitude=255.0,
                    basis_header_sha256=basis_header_sha256,
                )
            )
            if max_atoms is not None and len(atoms) >= int(max_atoms):
                return atoms
    return atoms


def certified_free_pixel_capacity(
    basis: EvaluatorInvisibilityBasis,
    *,
    n_channels: int = 3,
) -> dict[str, Any]:
    """The total certified-free per-pixel perturbation capacity (the PR110++
    free-repair-room budget) for one frame role.

    This is the count of certified-invisible single-pixel directions BOTH scorer
    heads permit (resize zero-weight pixels x channels), plus the frame0 SegNet
    corollary capacity reported separately (SegNet-only-invisible, not both-head)."""
    t1 = basis.tier1_resize
    both_head = t1.n_zero_weight_pixels_per_channel * int(n_channels)
    return {
        "both_head_free_directions_per_frame": both_head,
        "both_head_free_fraction_per_channel": t1.zero_weight_pixel_fraction,
        "frame0_segnet_only_free_directions": (
            basis.frame0_corollary.segnet_invisible_directions
        ),
        "full_resize_null_dim_per_channel": t1.full_null_dim,
        "evidence_grade": "mathematical-derivation",
        "promotable": False,
    }


__all__ = [
    "CertifiedFreePerturbationAtom",
    "SectionTier1Accounting",
    "build_null_basis_drop_action",
    "build_null_basis_recode_action",
    "certified_free_pixel_capacity",
    "count_section_tier1_free_bytes",
    "generate_pr110_certified_free_atoms",
]
