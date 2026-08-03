# SPDX-License-Identifier: MIT
"""Per-axis gap decomposition of S against a DEMONSTRATED floor, and the denominator
every ΔS claim must be quoted against.

WHY THIS EXISTS (ddm_cv1, 2026-08-02).  The campaign repeatedly quoted ΔS values with no
denominator, which makes a 0.0077%-of-gap win and a 12%-of-gap win read identically.  This
equation makes the denominator executable: ``fraction_of_gap`` refuses to evaluate unless a
floor has been supplied, so a ΔS cannot be reported as "progress" without saying progress
toward what.

WHAT A "GAP" IS HERE.  Not the raw axis contribution.  The gap is measured against a
DEMONSTRATED floor -- an externally achieved row proving the axis value is reachable.  Our
own axis contribution over-states how much is available (part of it is the floor itself,
which nobody has beaten), and the raw contribution ordering can therefore disagree with the
gap ordering.  On the 2026-08-02 state it does not, but the ordering is a measured output,
never an assumption -- so ``rank_by_gap`` returns it rather than any caller hardcoding it.

RATE DENOMINATOR (Catalog #812 / ddm_us1).  ``upstream/evaluate.py`` sums
``rglob('*')`` over ``upstream/videos/`` DYNAMICALLY; it does NOT use a literal
37,545,489.  A stray ``._*`` or ``.DS_Store`` silently changes it.  This module therefore
REFUSES to default the denominator and requires it as a measured input with its source.

S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/rate_denominator_bytes
    (upstream/evaluate.py:92)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

EQUATION_ID = "gap_decomposition_against_demonstrated_floor_v1"

_SEG_COEFF = 100.0
_POSE_COEFF = 10.0
_RATE_COEFF = 25.0


def _require_finite_nonneg(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric, got {type(value).__name__}")
    out = float(value)
    if not math.isfinite(out) or out < 0.0:
        raise ValueError(f"{name} must be finite and non-negative, got {value!r}")
    return out


@dataclass(frozen=True)
class MeasuredScoreTriple:
    """A byte-closed (d_seg, d_pose, archive_bytes) row plus its rate denominator.

    ``source_artifact`` is mandatory: an unsourced triple cannot anchor a gap, and a gap
    without a source is exactly the unanchored-ΔS failure this equation exists to stop.
    """

    d_seg: float
    d_pose: float
    archive_bytes: int
    rate_denominator_bytes: int
    source_artifact: str
    axis_tag: str
    status: str = "MEASURED"

    def __post_init__(self) -> None:
        if self.status != "MEASURED":
            raise ValueError("only status=MEASURED is accepted; derived rows cannot anchor a gap")
        _require_finite_nonneg(self.d_seg, "d_seg")
        _require_finite_nonneg(self.d_pose, "d_pose")
        for name in ("archive_bytes", "rate_denominator_bytes"):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not isinstance(raw, int):
                raise TypeError(f"{name} must be int (a byte count), got {type(raw).__name__}")
            if raw <= 0:
                raise ValueError(f"{name} must be > 0, got {raw}")
        if not self.source_artifact.strip():
            raise ValueError("source_artifact is required")
        if not self.axis_tag.strip():
            raise ValueError("axis_tag is required (e.g. '[macOS-CPU advisory exact n600]')")

    @property
    def seg_contribution(self) -> float:
        return _SEG_COEFF * float(self.d_seg)

    @property
    def pose_contribution(self) -> float:
        return math.sqrt(_POSE_COEFF * float(self.d_pose))

    @property
    def rate_contribution(self) -> float:
        return _RATE_COEFF * float(self.archive_bytes) / float(self.rate_denominator_bytes)

    @property
    def total(self) -> float:
        """S recomputed from components -- never read a rounded 'Final score' field."""
        return self.seg_contribution + self.pose_contribution + self.rate_contribution


@dataclass(frozen=True)
class GapDecomposition:
    """Our measured row against a demonstrated floor. Both must be MEASURED."""

    ours: MeasuredScoreTriple
    floor: MeasuredScoreTriple

    def __post_init__(self) -> None:
        if self.ours.rate_denominator_bytes != self.floor.rate_denominator_bytes:
            raise ValueError(
                "rate denominators differ between our row and the floor row; the rate axis "
                "is not comparable across different upstream/videos/ contents (Catalog #812)"
            )

    @property
    def seg_gap(self) -> float:
        return self.ours.seg_contribution - self.floor.seg_contribution

    @property
    def pose_gap(self) -> float:
        return self.ours.pose_contribution - self.floor.pose_contribution

    @property
    def rate_gap(self) -> float:
        return self.ours.rate_contribution - self.floor.rate_contribution

    @property
    def total_gap(self) -> float:
        return self.ours.total - self.floor.total

    def per_axis(self) -> dict[str, float]:
        return {"seg": self.seg_gap, "pose": self.pose_gap, "rate": self.rate_gap}

    def shares(self) -> dict[str, float]:
        """Fraction of the TOTAL gap carried by each axis.

        Negative gaps (an axis where we already beat the floor) are reported as-is rather
        than clipped: a negative share is real information -- it says that axis is not
        where the remaining work is, and clipping it would silently inflate the others.
        """
        total = self.total_gap
        if abs(total) < 1e-12:
            raise ValueError("total gap is ~0; shares are undefined (we are AT the floor)")
        return {k: v / total for k, v in self.per_axis().items()}

    def rank_by_gap(self) -> tuple[str, ...]:
        """Axes ordered by remaining gap, largest first. MEASURED OUTPUT, not an assumption."""
        return tuple(
            name for name, _ in sorted(self.per_axis().items(), key=lambda kv: -kv[1])
        )

    def fraction_of_gap(self, delta_s: float) -> float:
        """THE DENOMINATOR. A ΔS expressed as a fraction of the remaining total gap.

        Sign convention matches the campaign: a score-LOWERING move has delta_s < 0 and
        returns a POSITIVE fraction (that much of the gap closed).
        """
        if isinstance(delta_s, bool) or not isinstance(delta_s, (int, float)):
            raise TypeError("delta_s must be numeric")
        value = float(delta_s)
        if not math.isfinite(value):
            raise ValueError("delta_s must be finite")
        total = self.total_gap
        if abs(total) < 1e-12:
            raise ValueError("total gap is ~0; fraction_of_gap is undefined")
        return -value / total

    def bytes_per_percent_of_gap(self) -> float:
        """How many archive bytes one percent of the remaining gap is worth.

        Converts the rate axis into the same currency as seg and pose so a byte saving and
        a distortion saving can be compared without re-deriving the exchange rate by hand.
        """
        total = self.total_gap
        if abs(total) < 1e-12:
            raise ValueError("total gap is ~0; exchange rate is undefined")
        bytes_per_s_unit = float(self.ours.rate_denominator_bytes) / _RATE_COEFF
        return abs(total) * 0.01 * bytes_per_s_unit
