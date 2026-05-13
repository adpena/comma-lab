"""Tests for tac.non_arbitrariness — generalized probe-disambiguator primitive.

Mirrors the test discipline of probe_seg_loss_surrogate_disambiguator: cover
KEEP / PRUNE / ENSEMBLE / DEFER paths + edge cases + threshold validation.
"""
from __future__ import annotations


import pytest

from tac.non_arbitrariness import (
    DEFAULT_REGIME_VAR_DEFER_THRESHOLD,
    PROBE_EVIDENCE_GRADE,
    PROBE_SCHEMA_VERSION,
    ProbeAlternative,
    ProbeRegime,
    ProbeResult,
    Verdict,
    probe_alternatives,
)


# ── Synthetic probe fixtures ────────────────────────────────────────────────


def _identical_signal(_alt, fixture):
    """Cost-fn that returns the same signal regardless of alternative.
    Used to test PRUNE (high cos sim → alternatives are redundant)."""
    return 0.5, fixture


def _opposite_signal_factory(name_to_value: dict[str, float]):
    """Cost-fn factory: each alternative produces a constant signal value.
    Used to test KEEP (low/negative cos sim → alternatives are distinct)."""

    def cost_fn(alt, fixture):
        # signal = (alt-specific scalar, fixture seed). similarity will operate
        # on the scalar.
        return 0.5, name_to_value[alt.name]

    return cost_fn


def _scalar_cosine(a: float, b: float) -> float:
    """Signed similarity for scalars: +1 if same sign, -1 if opposite, 0 if either is 0."""
    if a == 0 or b == 0:
        return 0.0
    if a * b > 0:
        return 1.0
    return -1.0


def _identity_cosine(a, b) -> float:
    """Always 1.0 — used with _identical_signal to test PRUNE."""
    return 1.0


def _make_constant_regime(name: str, value):
    """Regime whose builder ignores seed and returns a constant fixture."""

    def builder(_seed):
        return value

    return ProbeRegime(name=name, builder=builder)


# ── Basic verdict paths ────────────────────────────────────────────────────


def test_prune_when_alternatives_are_identical():
    """High cos sim across regimes → PRUNE (alternatives are redundant)."""
    alts = [ProbeAlternative("a"), ProbeAlternative("b")]
    regimes = [_make_constant_regime("r1", 1.0), _make_constant_regime("r2", 2.0)]
    result = probe_alternatives(
        alts, regimes,
        cost_fn=_identical_signal,
        similarity_fn=_identity_cosine,
        delta_baseline=-0.1,
        composition_predicted_delta=-0.11,
    )
    assert result.verdict == Verdict.PRUNE
    assert result.mean_cos_sim_pairs[("a", "b")] == pytest.approx(1.0)


def test_keep_when_alternatives_are_orthogonal_AND_composition_improves():
    """Low cos sim + composition reduces score → KEEP."""
    alts = [ProbeAlternative("a"), ProbeAlternative("b")]
    regimes = [_make_constant_regime("r1", 1.0)]
    cost_fn = _opposite_signal_factory({"a": 1.0, "b": -1.0})
    result = probe_alternatives(
        alts, regimes,
        cost_fn=cost_fn,
        similarity_fn=_scalar_cosine,
        delta_baseline=-0.05,
        composition_predicted_delta=-0.10,
        delta_keep_threshold=0.04,
    )
    # cos = -1.0 (opposite) → mean_cos = -1.0 ≤ keep threshold (0.30)
    # incremental = -0.10 - (-0.05) = -0.05; need ≤ -0.04 ✓
    assert result.verdict == Verdict.KEEP
    assert result.mean_cos_sim_pairs[("a", "b")] == pytest.approx(-1.0)


def test_ensemble_when_partial_overlap_and_composition_helps():
    """Mid cos sim + composition improves → ENSEMBLE."""
    alts = [ProbeAlternative("a"), ProbeAlternative("b")]
    regimes = [_make_constant_regime("r1", 1.0)]

    # cos = 0.5 (mid range, between keep=0.30 and prune=0.85)
    def custom_sim(a, b):
        return 0.5

    result = probe_alternatives(
        alts, regimes,
        cost_fn=lambda alt, fix: (0.5, 1.0),
        similarity_fn=custom_sim,
        delta_baseline=-0.05,
        composition_predicted_delta=-0.06,  # incremental = -0.01 (helps)
    )
    assert result.verdict == Verdict.ENSEMBLE


def test_defer_on_regime_variance_overrides_mean():
    """Regime variance >= 0.20 → DEFER even if mean cos suggests KEEP."""
    alts = [ProbeAlternative("a"), ProbeAlternative("b")]
    # Two regimes; first cos=0.0, second cos=1.0.
    # mean=0.5, variance=0.25 (>=0.20 → DEFER overrides).
    sim_calls = {"call_count": 0}

    def variable_sim(a, b):
        # Alternates between 0 and 1 on each call (deterministic by call order).
        # Within a regime n_repeats=1 so each regime sees one sim call.
        sim_calls["call_count"] += 1
        return 0.0 if sim_calls["call_count"] % 2 == 1 else 1.0

    regimes = [_make_constant_regime("r1", 1.0), _make_constant_regime("r2", 2.0)]
    result = probe_alternatives(
        alts, regimes,
        cost_fn=lambda alt, fix: (0.5, fix),
        similarity_fn=variable_sim,
        delta_baseline=-0.05,
        composition_predicted_delta=-0.10,
        n_repeats=1,
    )
    assert result.verdict == Verdict.DEFER
    assert result.cos_variance_across_regimes >= DEFAULT_REGIME_VAR_DEFER_THRESHOLD


def test_prune_when_predicted_regression_overrides_low_cos():
    """Low cos sim BUT composition regresses score → PRUNE (regression-first rule)."""
    alts = [ProbeAlternative("a"), ProbeAlternative("b")]
    regimes = [_make_constant_regime("r1", 1.0)]
    cost_fn = _opposite_signal_factory({"a": 1.0, "b": -1.0})
    result = probe_alternatives(
        alts, regimes,
        cost_fn=cost_fn,
        similarity_fn=_scalar_cosine,
        delta_baseline=-0.05,
        composition_predicted_delta=+0.01,  # POSITIVE = worse score
    )
    assert result.verdict == Verdict.PRUNE


def test_single_alternative_returns_single_verdict():
    """Composition of size 1 returns Verdict.SINGLE."""
    alts = [ProbeAlternative("a")]
    regimes = [_make_constant_regime("r1", 1.0)]
    result = probe_alternatives(
        alts, regimes,
        cost_fn=lambda alt, fix: (0.5, fix),
        similarity_fn=lambda a, b: 0.0,
        delta_baseline=-0.05,
    )
    assert result.verdict == Verdict.SINGLE
    assert result.composition == ("a",)


def test_defer_when_in_no_mans_land_with_no_improvement():
    """Mid cos sim + no improvement → ENSEMBLE if delta<=0; DEFER otherwise.

    Edge: incremental == 0.0 ⇒ ENSEMBLE branch ('<= 0.0').
    """
    alts = [ProbeAlternative("a"), ProbeAlternative("b")]
    regimes = [_make_constant_regime("r1", 1.0)]
    result = probe_alternatives(
        alts, regimes,
        cost_fn=lambda alt, fix: (0.5, 1.0),
        similarity_fn=lambda a, b: 0.5,  # mid range
        delta_baseline=-0.05,
        composition_predicted_delta=-0.05,  # incremental = 0.0 (no change)
    )
    # mean=0.5 ∈ (0.30, 0.85), incremental=0 → ENSEMBLE branch.
    assert result.verdict == Verdict.ENSEMBLE


# ── Validation ─────────────────────────────────────────────────────────────


def test_empty_alternatives_raises():
    with pytest.raises(ValueError, match=">= 1 alternative"):
        probe_alternatives(
            [], [_make_constant_regime("r1", 1.0)],
            cost_fn=lambda a, f: (0.0, 0.0),
            similarity_fn=lambda a, b: 0.0,
        )


def test_empty_regimes_raises():
    with pytest.raises(ValueError, match=">= 1 regime"):
        probe_alternatives(
            [ProbeAlternative("a")], [],
            cost_fn=lambda a, f: (0.0, 0.0),
            similarity_fn=lambda a, b: 0.0,
        )


def test_invalid_n_repeats_raises():
    with pytest.raises(ValueError, match="n_repeats"):
        probe_alternatives(
            [ProbeAlternative("a")], [_make_constant_regime("r1", 1.0)],
            cost_fn=lambda a, f: (0.0, 0.0),
            similarity_fn=lambda a, b: 0.0,
            n_repeats=0,
        )


def test_invalid_threshold_ordering_raises():
    with pytest.raises(ValueError, match="must be <"):
        probe_alternatives(
            [ProbeAlternative("a"), ProbeAlternative("b")],
            [_make_constant_regime("r1", 1.0)],
            cost_fn=lambda a, f: (0.0, 0.0),
            similarity_fn=lambda a, b: 0.0,
            cos_keep_threshold=0.85,
            cos_prune_threshold=0.30,
        )


def test_invalid_cos_threshold_range_raises():
    with pytest.raises(ValueError, match="cos thresholds must be in"):
        probe_alternatives(
            [ProbeAlternative("a"), ProbeAlternative("b")],
            [_make_constant_regime("r1", 1.0)],
            cost_fn=lambda a, f: (0.0, 0.0),
            similarity_fn=lambda a, b: 0.0,
            cos_keep_threshold=-0.5,  # invalid
            cos_prune_threshold=0.85,
        )


def test_unknown_composition_alternative_raises():
    with pytest.raises(ValueError, match="unknown alternative"):
        probe_alternatives(
            [ProbeAlternative("a")],
            [_make_constant_regime("r1", 1.0)],
            cost_fn=lambda a, f: (0.0, 0.0),
            similarity_fn=lambda a, b: 0.0,
            composition=["a", "z_does_not_exist"],
        )


def test_cost_fn_must_return_two_tuple_R2_yousfi():
    """R2-Yousfi: cost_fn returning bad arity must fail loud, not silently crash."""
    with pytest.raises(ValueError, match="cost_fn must return a 2-tuple"):
        probe_alternatives(
            [ProbeAlternative("a"), ProbeAlternative("b")],
            [_make_constant_regime("r1", 1.0)],
            cost_fn=lambda alt, fix: 0.5,  # returns scalar, not tuple
            similarity_fn=lambda a, b: 0.0,
        )

    with pytest.raises(ValueError, match="cost_fn must return a 2-tuple"):
        probe_alternatives(
            [ProbeAlternative("a"), ProbeAlternative("b")],
            [_make_constant_regime("r1", 1.0)],
            cost_fn=lambda alt, fix: (0.5, 0.0, 0.0),  # 3-tuple
            similarity_fn=lambda a, b: 0.0,
        )


# ── Reproducibility + determinism ──────────────────────────────────────────


def test_seed_determinism():
    """Same seed + same inputs → same verdict + same numeric pair cosines."""
    alts = [ProbeAlternative("a"), ProbeAlternative("b"), ProbeAlternative("c")]
    regimes = [_make_constant_regime("r1", 1.0), _make_constant_regime("r2", 2.0)]

    def deterministic_cost_fn(alt, fix):
        return float(hash(alt.name + str(fix))) / 1e18, alt.name

    def deterministic_sim(a, b):
        # Hash-based similarity that doesn't care about fixture/seed.
        return -1.0 if a != b else 1.0

    result_a = probe_alternatives(
        alts, regimes,
        cost_fn=deterministic_cost_fn,
        similarity_fn=deterministic_sim,
        seed=42,
        n_repeats=2,
        delta_baseline=-0.05,
        composition_predicted_delta=-0.10,
    )
    result_b = probe_alternatives(
        alts, regimes,
        cost_fn=deterministic_cost_fn,
        similarity_fn=deterministic_sim,
        seed=42,
        n_repeats=2,
        delta_baseline=-0.05,
        composition_predicted_delta=-0.10,
    )
    assert result_a.verdict == result_b.verdict
    assert result_a.mean_cos_sim_pairs == result_b.mean_cos_sim_pairs


# ── Schema + provenance ────────────────────────────────────────────────────


def test_probe_result_schema_and_evidence_grade():
    alts = [ProbeAlternative("a"), ProbeAlternative("b")]
    regimes = [_make_constant_regime("r1", 1.0)]
    result = probe_alternatives(
        alts, regimes,
        cost_fn=lambda alt, fix: (0.5, alt.name),
        similarity_fn=lambda a, b: 0.0,
    )
    assert result.schema == PROBE_SCHEMA_VERSION
    assert result.evidence_grade == PROBE_EVIDENCE_GRADE
    assert isinstance(result, ProbeResult)


def test_default_composition_is_all_alternatives():
    alts = [ProbeAlternative("a"), ProbeAlternative("b"), ProbeAlternative("c")]
    regimes = [_make_constant_regime("r1", 1.0)]
    result = probe_alternatives(
        alts, regimes,
        cost_fn=lambda alt, fix: (0.5, alt.name),
        similarity_fn=lambda a, b: 0.0,
    )
    assert result.composition == ("a", "b", "c")


def test_no_predicted_delta_appends_warning_note():
    alts = [ProbeAlternative("a"), ProbeAlternative("b")]
    regimes = [_make_constant_regime("r1", 1.0)]
    result = probe_alternatives(
        alts, regimes,
        cost_fn=lambda alt, fix: (0.5, alt.name),
        similarity_fn=lambda a, b: 0.0,
        delta_baseline=-0.05,
        # composition_predicted_delta omitted → defaults to delta_baseline
    )
    assert any("composition_predicted_delta=None" in n for n in result.notes)
    assert result.predicted_delta == result.delta_baseline


# ── Smoke: 3-way composition matches probe_seg_loss_surrogate_disambiguator shape ──


def test_three_way_composition_seg_surrogate_shape():
    """Smoke test mirroring the shape of probe_seg_loss_surrogate_disambiguator's
    {T7, T8, T11} composition. Verifies that the primitive can host the
    canonical use case."""
    alts = [
        ProbeAlternative("T7_fisher_rao", payload={"loss_kind": "fisher_rao"}),
        ProbeAlternative("T8_sinkhorn_w2", payload={"loss_kind": "sinkhorn_w2"}),
        ProbeAlternative("T11_lovasz_hinge", payload={"loss_kind": "lovasz_hinge"}),
    ]
    regimes = [
        _make_constant_regime("soft_disagreement", 0.1),
        _make_constant_regime("sharp_disagreement", 0.9),
        _make_constant_regime("uniform_random", 0.5),
    ]

    def cost_fn(alt, fix):
        # Synthetic: each alternative encodes its name in the signal.
        signal = hash(alt.payload["loss_kind"]) / 1e18 + fix
        return 0.5, signal

    def cos_fn(a, b):
        # Synthetic: sign-based cosine of the synthetic signals.
        if a == 0 or b == 0:
            return 0.0
        return 1.0 if a * b > 0 else -1.0

    result = probe_alternatives(
        alts, regimes,
        cost_fn=cost_fn, similarity_fn=cos_fn,
        delta_baseline=-0.05,
        composition_predicted_delta=-0.10,
    )
    # Verdict is one of the canonical 4 (or BASELINE/SINGLE for size-1).
    assert result.verdict in {Verdict.KEEP, Verdict.PRUNE, Verdict.ENSEMBLE, Verdict.DEFER}
    # 3 pairs from a 3-way composition.
    assert len(result.mean_cos_sim_pairs) == 3
