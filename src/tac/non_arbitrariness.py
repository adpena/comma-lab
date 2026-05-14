# SPDX-License-Identifier: MIT
"""Non-arbitrariness probe primitive — generalized from probe_seg_loss_surrogate_disambiguator.

Per ``feedback_design_tension_ship_both_interpretations_let_math_arbitrate_20260509``
and the operator framing 2026-05-09 ("non-arbitrariness"), when the council
debates a design choice with 2+ defensible interpretations, the agent should NOT
silently pick one. Instead, it should ship both as alternatives + run a
regime-conditional probe + emit a verdict the planner consumes.

The canonical example is ``tools/probe_seg_loss_surrogate_disambiguator.py``
which evaluates {T7 Fisher-Rao, T8 Sinkhorn-W2, T11 Lovász hinge} as seg-loss
surrogates across 5 regimes (soft_disagreement, sharp_disagreement, ...) and
emits KEEP / PRUNE / ENSEMBLE / DEFER per composition.

This module abstracts the pattern so any 2+-alternative design tension can use
the same arbitration surface:

  ProbeAlternative   — one candidate interpretation (e.g. "T7", "T8")
  ProbeRegime        — one fixture regime under which alternatives differ
  ProbeVerdict       — KEEP / PRUNE / ENSEMBLE / DEFER + per-regime breakdown
  probe_alternatives — orchestration entry point

Wire-in
-------
- Used by ``tac.unified_action`` when DualVariables would otherwise need to
  choose between 2+ refinement-track contributions.
- Used by ``tools/cathedral_autopilot.py`` plan rows that have 2+ defensible
  encoder configs.
- Cross-ref ``tools/probe_seg_loss_surrogate_disambiguator.py`` for the
  empirical implementation.

CLAUDE.md compliance
--------------------
- Returns verdicts tagged ``[probe-evidence:local]`` — never claims a contest
  score. The verdict steers planning; authoritative scores come from
  ``upstream/evaluate.py``.
- Default thresholds match the disambiguator: KEEP at cos<=0.30, PRUNE at
  cos>=0.85, regime variance defer at >=0.20. Caller may tune.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations
from typing import Any

PROBE_SCHEMA_VERSION = "tac_non_arbitrariness_probe_v1"
PROBE_EVIDENCE_GRADE = "[probe-evidence:local]"

# Default thresholds (mirror probe_seg_loss_surrogate_disambiguator).
DEFAULT_COS_KEEP_THRESHOLD = 0.30
DEFAULT_COS_PRUNE_THRESHOLD = 0.85
DEFAULT_REGIME_VAR_DEFER_THRESHOLD = 0.20
DEFAULT_DELTA_REGRESSION = 0.0
DEFAULT_DELTA_KEEP_THRESHOLD = 0.0


class Verdict(str, Enum):
    """Per-composition decision after probe evaluation."""

    KEEP = "KEEP"  # alternatives are sufficiently distinct + composition reduces score
    ENSEMBLE = "ENSEMBLE"  # alternatives partially overlap but stack is still helpful
    PRUNE = "PRUNE"  # alternatives are redundant OR composition regresses score
    DEFER = "DEFER"  # regime-conditional disagreement; need more probe data
    BASELINE = "BASELINE"  # single-alternative baseline marker
    SINGLE = "SINGLE"  # single-alternative (non-baseline)


@dataclass(frozen=True)
class ProbeAlternative:
    """One candidate interpretation of a design choice.

    Examples:
      - ``ProbeAlternative("T7_fisher_rao", payload={"loss_fn": fisher_rao_fn})``
      - ``ProbeAlternative("brotli_q11", payload={"encoder": "brotli", "quality": 11})``
    """

    name: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProbeRegime:
    """One fixture regime under which alternatives may differ.

    The ``builder`` callable produces a regime-specific input fixture (e.g. a
    soft-disagreement segmentation pair). The probe evaluates each alternative
    against the same fixture sequence so that cosine similarity / score-delta
    are comparable.
    """

    name: str
    builder: Callable[[int], Any]  # seed -> fixture (any caller-defined type)


@dataclass(frozen=True)
class ProbeResult:
    """Final per-composition + per-regime probe report."""

    schema: str
    evidence_grade: str
    composition: tuple[str, ...]
    verdict: Verdict
    mean_cos_sim_pairs: dict[tuple[str, str], float]
    cos_variance_across_regimes: float
    predicted_delta: float
    delta_baseline: float
    incremental_delta: float
    n_regimes: int
    n_repeats: int
    notes: list[str] = field(default_factory=list)


def _composition_verdict(
    composition: tuple[str, ...],
    cos_sim_pairs: dict[tuple[str, str], float],
    predicted_delta: float,
    delta_baseline: float,
    *,
    cos_variance_across_regimes: float,
    cos_keep_threshold: float,
    cos_prune_threshold: float,
    regime_var_defer_threshold: float,
    delta_regression: float,
    delta_keep_threshold: float,
) -> Verdict:
    """Per-composition verdict — generalized from probe_seg_loss_surrogate_disambiguator._verdict_for_composition.

    The DEFER branch fires either:
    * by mean-cos-sim being in the no-mans-land between KEEP/ENSEMBLE/PRUNE
      thresholds, OR
    * (the more important regime-aware case) when ``cos_variance_across_regimes``
      is high (>= regime_var_defer_threshold). Variance >= 0.20 across regimes
      means the cos sim FLIPS sign between regimes; mean-only verdict is
      misleading.
    """
    if len(composition) == 1:
        return Verdict.SINGLE  # generic single-alternative marker

    incremental = predicted_delta - delta_baseline

    if not cos_sim_pairs:
        # Defensive — shouldn't happen for len > 1
        return Verdict.DEFER
    cos_vals = [cos_sim_pairs[tuple(sorted((a, b)))] for a, b in combinations(composition, 2)]
    mean_cos = sum(cos_vals) / len(cos_vals)
    max_cos = max(cos_vals)

    # High variance overrides mean-based verdict.
    if cos_variance_across_regimes >= regime_var_defer_threshold:
        return Verdict.DEFER
    if predicted_delta > -delta_regression:
        # Predicted regression (positive delta means worse score).
        return Verdict.PRUNE
    if max_cos >= cos_prune_threshold:
        return Verdict.PRUNE
    if mean_cos <= cos_keep_threshold and incremental <= -delta_keep_threshold:
        return Verdict.KEEP
    if cos_keep_threshold < mean_cos < cos_prune_threshold and incremental <= 0.0:
        return Verdict.ENSEMBLE
    return Verdict.DEFER


def probe_alternatives(
    alternatives: Sequence[ProbeAlternative],
    regimes: Sequence[ProbeRegime],
    cost_fn: Callable[[ProbeAlternative, Any], tuple[float, Any]],
    similarity_fn: Callable[[Any, Any], float],
    *,
    composition: Sequence[str] | None = None,
    n_repeats: int = 3,
    seed: int = 0,
    cos_keep_threshold: float = DEFAULT_COS_KEEP_THRESHOLD,
    cos_prune_threshold: float = DEFAULT_COS_PRUNE_THRESHOLD,
    regime_var_defer_threshold: float = DEFAULT_REGIME_VAR_DEFER_THRESHOLD,
    delta_regression: float = DEFAULT_DELTA_REGRESSION,
    delta_keep_threshold: float = DEFAULT_DELTA_KEEP_THRESHOLD,
    delta_baseline: float = 0.0,
    composition_predicted_delta: float | None = None,
) -> ProbeResult:
    """Run the non-arbitrariness probe over alternatives × regimes.

    Args:
        alternatives: list of ProbeAlternative to compare
        regimes: list of ProbeRegime fixtures spanning the operating-point space
        cost_fn: ``(alternative, fixture) -> (cost_scalar, signal)``. ``signal`` is
            an opaque artifact (e.g. a gradient tensor) that ``similarity_fn``
            will compare across alternatives. ``cost_scalar`` is the score
            contribution of this alternative on this fixture.
        similarity_fn: ``(signal_a, signal_b) -> cos_sim_in_[-1, 1]`` — measures
            whether two alternatives produce the same-direction signal.
        composition: subset of alternatives to evaluate as a stacked composition.
            None ⇒ evaluate the full set ``[a.name for a in alternatives]``.
        n_repeats: per-regime repeat count (averaged for noise).
        seed: deterministic seed; per-repeat seeds are derived as
            ``seed + 7919 * r``.
        cos_*_threshold + regime_var_defer_threshold + delta_*: thresholds for
            the verdict (defaults match probe_seg_loss_surrogate_disambiguator).
        delta_baseline: predicted score delta for the SINGLE-alternative
            baseline (caller supplies — e.g. T7-alone).
        composition_predicted_delta: predicted score delta for the composition.
            If None, defaults to delta_baseline (i.e. assume no incremental
            improvement, which biases toward PRUNE).

    Returns:
        :class:`ProbeResult` with verdict + diagnostic fields.
    """
    if not alternatives:
        raise ValueError("probe_alternatives requires >= 1 alternative")
    if not regimes:
        raise ValueError("probe_alternatives requires >= 1 regime")
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")
    if cos_keep_threshold < 0 or cos_prune_threshold > 1:
        raise ValueError(
            "cos thresholds must be in [-1, 1] (got "
            f"keep={cos_keep_threshold}, prune={cos_prune_threshold})"
        )
    if cos_keep_threshold >= cos_prune_threshold:
        raise ValueError(
            f"cos_keep_threshold ({cos_keep_threshold}) must be < "
            f"cos_prune_threshold ({cos_prune_threshold})"
        )

    composition_names = tuple(composition) if composition is not None else tuple(a.name for a in alternatives)
    name_to_alt = {a.name: a for a in alternatives}
    for cname in composition_names:
        if cname not in name_to_alt:
            raise ValueError(
                f"composition references unknown alternative: {cname!r}"
            )

    # Per-regime per-pair cosine similarities.
    per_regime_pair_cos: dict[str, dict[tuple[str, str], list[float]]] = {}
    notes: list[str] = []

    for regime in regimes:
        per_regime_pair_cos[regime.name] = {
            tuple(sorted((a, b))): [] for a, b in combinations(composition_names, 2)
        }
        for r in range(n_repeats):
            fixture = regime.builder(seed + 7919 * r)
            signals: dict[str, Any] = {}
            for cname in composition_names:
                cost_result = cost_fn(name_to_alt[cname], fixture)
                # R2-Yousfi: validate cost_fn returns (cost_scalar, signal); fail
                # loud rather than crash on tuple-unpack with bad arity.
                if not isinstance(cost_result, tuple) or len(cost_result) != 2:
                    raise ValueError(
                        f"cost_fn must return a 2-tuple (cost_scalar, signal); "
                        f"got {type(cost_result).__name__} of length "
                        f"{len(cost_result) if isinstance(cost_result, tuple) else 'N/A'} "
                        f"for alternative={cname!r} regime={regime.name!r}"
                    )
                _cost, sig = cost_result
                signals[cname] = sig
            # All pairwise cos sims for this fixture.
            for a, b in combinations(composition_names, 2):
                key = tuple(sorted((a, b)))
                per_regime_pair_cos[regime.name][key].append(similarity_fn(signals[a], signals[b]))

    # Aggregate per-regime mean cos.
    per_regime_mean_cos: dict[str, dict[tuple[str, str], float]] = {}
    for regime_name, pair_dict in per_regime_pair_cos.items():
        per_regime_mean_cos[regime_name] = {
            k: (sum(v) / len(v) if v else 0.0) for k, v in pair_dict.items()
        }

    # Aggregate cross-regime: per-pair mean (used for verdict) + per-pair variance
    # (used to detect regime flip).
    pair_keys = list({k for d in per_regime_mean_cos.values() for k in d.keys()})
    mean_cos_sim_pairs: dict[tuple[str, str], float] = {}
    pair_variances: list[float] = []
    for pk in pair_keys:
        regime_means = [per_regime_mean_cos[r][pk] for r in per_regime_mean_cos.keys()]
        mu = sum(regime_means) / len(regime_means)
        mean_cos_sim_pairs[pk] = mu
        var = sum((m - mu) ** 2 for m in regime_means) / len(regime_means)
        pair_variances.append(var)
    cos_variance_across_regimes = (
        max(pair_variances) if pair_variances else 0.0
    )

    predicted_delta = (
        composition_predicted_delta
        if composition_predicted_delta is not None
        else delta_baseline
    )
    if composition_predicted_delta is None:
        notes.append(
            "composition_predicted_delta=None; defaulted to delta_baseline "
            "(biases verdict toward PRUNE; caller should supply explicit prediction)"
        )

    verdict = _composition_verdict(
        composition_names,
        mean_cos_sim_pairs,
        predicted_delta=predicted_delta,
        delta_baseline=delta_baseline,
        cos_variance_across_regimes=cos_variance_across_regimes,
        cos_keep_threshold=cos_keep_threshold,
        cos_prune_threshold=cos_prune_threshold,
        regime_var_defer_threshold=regime_var_defer_threshold,
        delta_regression=delta_regression,
        delta_keep_threshold=delta_keep_threshold,
    )

    if cos_variance_across_regimes >= regime_var_defer_threshold:
        notes.append(
            f"regime cos-variance {cos_variance_across_regimes:.3f} >= "
            f"defer threshold {regime_var_defer_threshold:.3f} → DEFER"
        )

    return ProbeResult(
        schema=PROBE_SCHEMA_VERSION,
        evidence_grade=PROBE_EVIDENCE_GRADE,
        composition=composition_names,
        verdict=verdict,
        mean_cos_sim_pairs=mean_cos_sim_pairs,
        cos_variance_across_regimes=cos_variance_across_regimes,
        predicted_delta=predicted_delta,
        delta_baseline=delta_baseline,
        incremental_delta=predicted_delta - delta_baseline,
        n_regimes=len(regimes),
        n_repeats=n_repeats,
        notes=notes,
    )


__all__ = [
    "PROBE_SCHEMA_VERSION",
    "PROBE_EVIDENCE_GRADE",
    "DEFAULT_COS_KEEP_THRESHOLD",
    "DEFAULT_COS_PRUNE_THRESHOLD",
    "DEFAULT_REGIME_VAR_DEFER_THRESHOLD",
    "Verdict",
    "ProbeAlternative",
    "ProbeRegime",
    "ProbeResult",
    "probe_alternatives",
]
