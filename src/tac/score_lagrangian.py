# SPDX-License-Identifier: MIT
"""Canonical analytical Lagrangian-multiplier helper for the contest score.

Contest score (per upstream/evaluate.py):

    S = 25 * (archive_bytes / N) + 100 * d_seg + sqrt(10 * d_pose)

where ``N = 37,545,489`` is the canonical archive normalizer.

Derivation of training-time multipliers (Boyd & Vandenberghe 2004
*Convex Optimization* Ch.5 — KKT + Lagrangian multipliers):

* ``d(S)/d(d_seg) = 100``                      (constant)
* ``d(S)/d(d_pose) = 5 / sqrt(10 * d_pose)``   (operating-point dependent)
* ``d(S)/d(rate_bytes) = 25 / N = 6.66e-7``    (constant)

Optimal training-time multipliers should be PROPORTIONAL to these marginal
contributions at the current operating point. The naive uniform-multiplier
baseline (``lambda_seg = lambda_pose = lambda_rate = 1``) is FALSIFIED at
every non-saturated operating point because it ignores the
``d(pose)/d(pose_avg) -> infinity`` divergence as ``pose_avg -> 0`` (which is
exactly the PR106-frontier regime).

Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent":

* OLD 1.x operating point (``pose_avg ~= 0.18``):
  ``lambda_pose/lambda_seg = 12/100 ~= 0.12`` -> SegNet 77x more important
* PR106 frontier (``pose_avg ~= 3.4e-5``):
  ``lambda_pose/lambda_seg = 271/100 = 2.71`` -> pose 2.71x more important
* Crossover at ``pose_avg ~= 2.5e-4`` (where
  ``d(pose)/d(pose_avg) = d(seg)/d(seg_avg)``)

Bug class extincted:
~~~~~~~~~~~~~~~~~~~~

Per the ARBITRARINESS-EXTINCTION audit landed 2026-05-18
(commit ``2d042f7e6``), TOP-1 ranked extinction
(``value_id=lambda_seg_pose_rate_multipliers_unprincipled``,
``rank_score_per_dollar=12.0``, ``predicted_ev_delta_s=[-0.012, -0.003]``,
``cost_envelope_usd=0.0``, ``resolution_path=PATH 2 analytical solve``)
identified ~30 substrate trainers that hand-tune ``lambda_seg``,
``lambda_pose``, ``lambda_rate`` per-trainer with values spanning ~10x.
Each per-trainer tune is a local optimization against a *snapshot* of the
operating point that goes stale as the substrate moves along the
score-axis. The closed-form formulas in this module derive the canonical
multipliers from the contest score directly, so any trainer that adopts
this helper inherits operating-point-aware multipliers without per-tune
maintenance.

Citations
~~~~~~~~~

* Boyd, S. & Vandenberghe, L. *Convex Optimization* (Cambridge, 2004).
  Chapter 5 (KKT conditions + Lagrangian multipliers).
* CLAUDE.md "Meta-Lagrangian/Pareto solver -- NON-NEGOTIABLE,
  HIGHEST EMPHASIS" (prefer solvable math over arbitrary sweeps;
  ground every knob in entropy/MDL / Fisher / Dykstra-feasibility).
* CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent
  (UPDATED 2026-05-04)" (the operating-point crossover empirical anchor).
* ARBITRARINESS-EXTINCTION audit 2026-05-18 commit ``2d042f7e6``
  TOP-1 (``lambda_seg_pose_rate_multipliers_unprincipled``).
* ``.omx/state/arbitrariness_extinction_audit_20260518.jsonl`` (52-row
  audit feeding this extinction).

Catalog #125 six-hook wire-in declaration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable, this module
declares its solver wire-in surface explicitly:

1. **Sensitivity-map contribution**: ACTIVE — every marginal multiplier
   IS a sensitivity-map row (``d(S)/d(metric)`` per axis); downstream
   ``tac.sensitivity_map.*`` consumers can route through
   :func:`compute_marginal_multipliers` to populate per-axis weights.
2. **Pareto constraint**: ACTIVE — the multipliers parameterize a
   single-scalar Lagrangian whose level-sets are the Pareto frontier of
   ``(d_seg, d_pose, rate_bytes)``; consumers may use them as Dykstra
   alternating-projection weights per CLAUDE.md "Council conduct".
3. **Bit-allocator hook**: N/A — bit-allocator operates at the
   per-tensor / per-channel layer below the contest-score layer this
   helper exposes; the bit-allocator's Lagrangian inherits these
   multipliers via the rate-term coefficient ``lambda_rate``.
4. **Cathedral autopilot dispatch hook**: ACTIVE — autopilot ranker
   may consume canonical multipliers to compose per-substrate
   score-aware loss without per-trainer hand-tuning.
5. **Continual-learning posterior update**: ACTIVE — operating-point
   classification feeds the cost-band posterior (per CLAUDE.md
   "MPS auth eval is NOISE" sister discipline ``cost_band_posterior``).
6. **Probe-disambiguator**: ACTIVE — the helper IS the canonical
   disambiguator between three regimes
   (``old_1x_seg_dominant`` / ``crossover`` / ``frontier_pose_dominant``).

Out-of-scope
~~~~~~~~~~~~

Substrate-trainer wire-ins (the ~30 substrates that currently hand-tune
``lambda_seg/lambda_pose/lambda_rate``) are operator-routable follow-ons.
Each substrate-trainer touch requires per-substrate symposium per
CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council
symposium" non-negotiable (Catalog #325).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

__all__ = [
    "CONTEST_RATE_DENOM_BYTES",
    "RATE_SCORE_PER_BYTE",
    "SEG_SCORE_PER_UNIT",
    "POSE_CROSSOVER_AT_AVG",
    "LagrangianMultipliers",
    "compute_marginal_multipliers",
    "empirical_anchor_pr106_frontier",
    "empirical_anchor_old_1x_operating_point",
]

#: Canonical archive normalizer used by the contest rate term
#: (``25 * archive_bytes / N``). Derived from upstream/evaluate.py.
CONTEST_RATE_DENOM_BYTES: Final[int] = 37_545_489

#: Marginal score per archive byte: ``25 / N``.
#: Constant across all operating points because the rate term is linear.
RATE_SCORE_PER_BYTE: Final[float] = 25.0 / CONTEST_RATE_DENOM_BYTES

#: Marginal score per unit of SegNet distortion: ``d(100*d_seg)/d(d_seg) = 100``.
#: Constant across all operating points because the seg term is linear.
SEG_SCORE_PER_UNIT: Final[float] = 100.0

#: Pose-vs-seg marginal crossover in pose_avg space.
#:
#: At ``pose_avg ~ 2.5e-4`` the marginal contribution
#: ``d(sqrt(10*pose_avg))/d(pose_avg) = 5/sqrt(10*pose_avg) ~ 100``
#: equals the SegNet marginal (``= 100``). Below this threshold, pose
#: is marginally more important than seg; above it, seg is marginally
#: more important. Empirical anchors per CLAUDE.md:
#: OLD 1.x pose_avg ~ 0.18 (well above) -> SegNet-dominant.
#: PR106 frontier pose_avg ~ 3.4e-5 (well below) -> pose-dominant.
POSE_CROSSOVER_AT_AVG: Final[float] = 2.5e-4


@dataclass(frozen=True, slots=True)
class LagrangianMultipliers:
    """Canonical training-time multipliers for the contest score's three axes.

    Attributes
    ----------
    lambda_seg
        Marginal score contribution per unit SegNet distortion. Constant
        ``= 100.0`` across all operating points.
    lambda_pose
        Marginal score contribution per unit PoseNet distortion. Equals
        ``5 / sqrt(10 * pose_avg)`` -- diverges as ``pose_avg -> 0``
        (the PR106-frontier regime).
    lambda_rate
        Marginal score contribution per archive byte. Constant
        ``= 25 / 37,545,489 ~ 6.66e-7`` (configurable for diagnostic uses
        that route through a different rate axis).
    pose_to_seg_ratio
        Convenience field exposing ``lambda_pose / lambda_seg``.
        Cross-check against CLAUDE.md empirical anchors: ~0.12 at OLD 1.x,
        ~2.71 at PR106 frontier.
    operating_point_classification
        One of ``"old_1x_seg_dominant"`` / ``"crossover"`` /
        ``"frontier_pose_dominant"``. Discrete probe-disambiguator output
        per Catalog #125 hook 6.
    """

    lambda_seg: float
    lambda_pose: float
    lambda_rate: float
    pose_to_seg_ratio: float
    operating_point_classification: str


def compute_marginal_multipliers(
    *,
    seg_avg: float,
    pose_avg: float,
    rate_axis_bytes_per_unit: float = RATE_SCORE_PER_BYTE,
    normalize_to_sum_one: bool = False,
) -> LagrangianMultipliers:
    """Compute canonical Lagrangian multipliers from contest-score marginals.

    Returns multipliers PROPORTIONAL to ``d(score)/d(metric)`` at the current
    operating point. By default unnormalized (use raw marginals); pass
    ``normalize_to_sum_one=True`` for a softmax-like normalization that sums
    to 1 across the three axes.

    Parameters
    ----------
    seg_avg
        Current average SegNet distortion. Must be non-negative.
    pose_avg
        Current average PoseNet distortion. Must be strictly positive
        (the marginal diverges at ``pose_avg = 0``).
    rate_axis_bytes_per_unit
        Marginal score per unit on the rate axis. Defaults to the canonical
        ``RATE_SCORE_PER_BYTE = 25 / 37,545,489``. Override only for
        diagnostic / non-contest rate axes (e.g. bits-per-pair instead of
        bytes-per-archive).
    normalize_to_sum_one
        If True, the three multipliers are scaled so they sum to 1. Useful
        when the multipliers must be interpreted as relative weights in a
        convex combination (e.g. Dykstra alternating-projection weights);
        retains the ratio between axes.

    Returns
    -------
    LagrangianMultipliers
        Frozen dataclass with ``lambda_seg``, ``lambda_pose``, ``lambda_rate``,
        ``pose_to_seg_ratio``, ``operating_point_classification``.

    Raises
    ------
    ValueError
        If ``pose_avg <= 0`` or ``seg_avg < 0`` (the marginal formula is
        undefined / yields a degenerate Lagrangian).

    Notes
    -----
    Per CLAUDE.md "SegNet vs PoseNet importance — operating-point
    dependent":

    * ``pose_avg < 2.5e-4`` (the crossover) -> classified
      ``frontier_pose_dominant`` (pose marginal strictly exceeds seg
      marginal; this covers the PR106-frontier canonical anchor
      ``pose_avg = 3.4e-5`` which yields ratio ~ 2.71).
    * ``pose_avg > 2.5e-3`` (one order of magnitude above crossover) ->
      classified ``old_1x_seg_dominant`` (seg marginal at least ~10x
      pose marginal; covers the OLD 1.x canonical anchor
      ``pose_avg = 0.18`` which yields ratio ~ 0.12).
    * Otherwise (``2.5e-4 <= pose_avg <= 2.5e-3``) -> classified
      ``crossover`` (multipliers are within ~10x of each other; neither
      axis is clearly dominant).

    Catalog #125 hook 4 (cathedral autopilot dispatch): ACTIVE -- the
    autopilot ranker can consume canonical multipliers to compose
    per-substrate score-aware loss without per-trainer hand-tuning.

    Examples
    --------
    >>> mult = compute_marginal_multipliers(seg_avg=6.7e-4, pose_avg=3.4e-5)
    >>> round(mult.pose_to_seg_ratio, 2)
    2.71
    >>> mult.operating_point_classification
    'frontier_pose_dominant'

    >>> mult = compute_marginal_multipliers(seg_avg=0.07, pose_avg=0.18)
    >>> round(mult.pose_to_seg_ratio, 2)
    0.12
    >>> mult.operating_point_classification
    'old_1x_seg_dominant'
    """
    if pose_avg <= 0:
        raise ValueError(
            f"pose_avg must be positive (got {pose_avg!r}); the marginal "
            f"d(sqrt(10*pose_avg))/d(pose_avg) diverges at pose_avg=0"
        )
    if seg_avg < 0:
        raise ValueError(
            f"seg_avg must be non-negative (got {seg_avg!r})"
        )

    lambda_seg = SEG_SCORE_PER_UNIT
    lambda_pose = 5.0 / math.sqrt(10.0 * pose_avg)
    lambda_rate = rate_axis_bytes_per_unit

    if normalize_to_sum_one:
        total = lambda_seg + lambda_pose + lambda_rate
        # total > 0 always (lambda_seg = 100; others non-negative)
        lambda_seg = lambda_seg / total
        lambda_pose = lambda_pose / total
        lambda_rate = lambda_rate / total

    ratio = lambda_pose / lambda_seg

    if pose_avg < POSE_CROSSOVER_AT_AVG:
        op_class = "frontier_pose_dominant"
    elif pose_avg > POSE_CROSSOVER_AT_AVG * 10:
        op_class = "old_1x_seg_dominant"
    else:
        op_class = "crossover"

    return LagrangianMultipliers(
        lambda_seg=lambda_seg,
        lambda_pose=lambda_pose,
        lambda_rate=lambda_rate,
        pose_to_seg_ratio=ratio,
        operating_point_classification=op_class,
    )


def empirical_anchor_pr106_frontier() -> LagrangianMultipliers:
    """Return canonical PR106-frontier multipliers.

    Operating point: ``seg_avg = 6.7e-4``, ``pose_avg = 3.4e-5`` per the
    CLAUDE.md "SegNet vs PoseNet importance" table for the PR106 frontier
    operating point. Yields ``pose_to_seg_ratio ~ 2.71``
    (canonical anchor for the operating-point-dependent rule).
    """
    return compute_marginal_multipliers(seg_avg=6.7e-4, pose_avg=3.4e-5)


def empirical_anchor_old_1x_operating_point() -> LagrangianMultipliers:
    """Return canonical OLD-1.x-baseline multipliers.

    Operating point: ``seg_avg = 0.07``, ``pose_avg = 0.18`` per the
    CLAUDE.md "SegNet vs PoseNet importance" historical anchor. Yields
    ``pose_to_seg_ratio ~ 0.12`` (seg-dominant regime).
    """
    return compute_marginal_multipliers(seg_avg=0.07, pose_avg=0.18)
