# SPDX-License-Identifier: MIT
"""Fisher-mass-in-annulus observable (SPEC_v10 §13.2 trigger observable; §13.4 surface 5).

OBSERVER-ONLY: pure numpy, reads a margin field, writes nothing, never touches
training state — byte-identity of any run consuming it is preserved BY
CONSTRUCTION (per the "observability defaults ON" non-negotiable this module
carries no enable gate of its own; callers decide cadence).

The law (registered): ``fisher_curvature_equals_categorical_fisher_trace_caustic_v1``
+ ``optimal_metric_unification_v1`` —

    tr g |_{2-class} = 2 p (1 - p) = (1/2) sech^2(m / 2),   p = sigma(m)

where ``m`` is the top1-top2 scalar logit margin (the exact distance-to-flip,
``scalar_top1_top2_margin_is_exact_distance_to_flip_v1``). The Fisher trace is
the local decision-geometry density: it peaks at 0.5 on the separatrix (m=0)
and decays even-symmetrically in |m|. The RESIDUAL Fisher mass inside the
|m| < band annulus is therefore the decision-geometry-native "how much
flip-relevant geometry remains near the boundary" observable — the §13.2
trigger reading (convergence/engagement events read THIS, not the raw
Euclidean d_seg slope).

DERIVED, not measured: the sech^2 identity is the registered derivation; the
0.978 Pearson (curvature <-> -margin) is the registered measured calibration.
This module adds NO new empirical claim — it evaluates the registered law.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "FisherAnnulusReport",
    "fisher_trace_from_margin",
    "fisher_mass_in_annulus",
    "fisher_mass_report_over_pairs",
]

# The registered annulus band default (|m| < 2): the SPEC §13.4(5) "m<2 annulus".
DEFAULT_ANNULUS_BAND = 2.0


def fisher_trace_from_margin(margin: np.ndarray) -> np.ndarray:
    """Per-pixel categorical Fisher trace (two-class annulus form) from a margin field.

    ``tr g = (1/2) sech^2(m/2)`` evaluated OVERFLOW-STABLY:
    ``sech^2(x) = 4 e^{-2|x|} / (1 + e^{-2|x|})^2`` (exact for all finite x;
    even in x, so signed or absolute margins give identical output).

    Args:
        margin: any-shape float array of top1-top2 logit margins (signed or abs).

    Returns:
        Same-shape float64 array in (0, 0.5]; 0.5 exactly at m=0.
    """
    m = np.asarray(margin, dtype=np.float64)
    x = np.abs(m) * 0.5  # sech^2(m/2), |.| by evenness
    e = np.exp(-2.0 * x)  # in (0, 1], no overflow for any finite m
    sech2 = 4.0 * e / np.square(1.0 + e)
    return 0.5 * sech2


@dataclass(frozen=True)
class FisherAnnulusReport:
    """Observer-only report of residual Fisher mass in the |m| < band annulus.

    Non-promotable observability row by construction (Tier A markers carried
    explicitly so no downstream consumer can mistake this for a score claim).
    """

    band: float
    total_px: int
    annulus_px: int
    total_fisher_mass: float
    annulus_fisher_mass: float
    # annulus Fisher mass / total Fisher mass in [0, 1] — the trigger reading.
    annulus_mass_fraction: float
    # annulus px / total px (geometric occupancy, for context vs mass concentration).
    annulus_px_fraction: float
    mean_fisher_in_annulus: float
    # Tier A observability markers (Catalog #341): NEVER a score claim.
    axis_tag: str = field(default="[observer]")
    promotable: bool = field(default=False)
    score_claim: bool = field(default=False)

    def to_dict(self) -> dict:
        return {
            "band": self.band,
            "total_px": self.total_px,
            "annulus_px": self.annulus_px,
            "total_fisher_mass": self.total_fisher_mass,
            "annulus_fisher_mass": self.annulus_fisher_mass,
            "annulus_mass_fraction": self.annulus_mass_fraction,
            "annulus_px_fraction": self.annulus_px_fraction,
            "mean_fisher_in_annulus": self.mean_fisher_in_annulus,
            "axis_tag": self.axis_tag,
            "promotable": self.promotable,
            "score_claim": self.score_claim,
        }


def fisher_mass_in_annulus(
    margin: np.ndarray, band: float = DEFAULT_ANNULUS_BAND
) -> FisherAnnulusReport:
    """Residual Fisher mass inside the |m| < ``band`` annulus of a margin field.

    Args:
        margin: (H, W) or any-shape margin field (signed or absolute; the Fisher
            trace is even in m and the annulus test uses |m|).
        band: annulus half-width in margin/logit units (default 2.0, the SPEC
            §13.4(5) "m<2" reading).

    Returns:
        FisherAnnulusReport (frozen, observer-only, non-promotable).
    """
    if not np.isfinite(band) or band <= 0.0:
        raise ValueError(f"band must be a finite positive float, got {band!r}")
    m = np.asarray(margin, dtype=np.float64)
    if m.size == 0:
        raise ValueError("margin field is empty")
    if not np.all(np.isfinite(m)):
        raise ValueError("margin field contains non-finite values (fail closed)")
    tr = fisher_trace_from_margin(m)
    in_ann = np.abs(m) < band
    total_mass = float(tr.sum())
    ann_mass = float(tr[in_ann].sum())
    ann_px = int(in_ann.sum())
    return FisherAnnulusReport(
        band=float(band),
        total_px=int(m.size),
        annulus_px=ann_px,
        total_fisher_mass=total_mass,
        annulus_fisher_mass=ann_mass,
        annulus_mass_fraction=(ann_mass / total_mass) if total_mass > 0.0 else 0.0,
        annulus_px_fraction=ann_px / float(m.size),
        mean_fisher_in_annulus=(ann_mass / ann_px) if ann_px > 0 else 0.0,
    )


def fisher_mass_report_over_pairs(
    margins, band: float = DEFAULT_ANNULUS_BAND
) -> FisherAnnulusReport:
    """Aggregate the annulus report over an iterable of per-pair margin fields.

    Sums masses/pixels across pairs (equivalent to one report over the
    concatenated fields), so the fraction is the pair-population reading the
    §13.2 trigger consumes. Fail-closed on an empty iterable.
    """
    total_mass = 0.0
    ann_mass = 0.0
    total_px = 0
    ann_px = 0
    n = 0
    for m in margins:
        r = fisher_mass_in_annulus(m, band=band)
        total_mass += r.total_fisher_mass
        ann_mass += r.annulus_fisher_mass
        total_px += r.total_px
        ann_px += r.annulus_px
        n += 1
    if n == 0:
        raise ValueError("no margin fields given")
    return FisherAnnulusReport(
        band=float(band),
        total_px=total_px,
        annulus_px=ann_px,
        total_fisher_mass=total_mass,
        annulus_fisher_mass=ann_mass,
        annulus_mass_fraction=(ann_mass / total_mass) if total_mass > 0.0 else 0.0,
        annulus_px_fraction=(ann_px / float(total_px)) if total_px > 0 else 0.0,
        mean_fisher_in_annulus=(ann_mass / ann_px) if ann_px > 0 else 0.0,
    )
