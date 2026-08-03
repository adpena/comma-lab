# SPDX-License-Identifier: MIT
"""Tests for :mod:`tac.witness_dsl.guarded_constant`.

The four tests named ``test_positive_control_*`` are the arm's required POSITIVE
CONTROLS: each one reproduces a MEASURED 2026-08-03 misuse instance and asserts
that the guard REFUSES it.  A guard that cannot be shown firing on a real case
is the ``#831`` class (refusal gates with no positive control) — these four exist
so this type does not become a 20th such gate.
"""
from __future__ import annotations

import numpy as np
import pytest

from tac.witness_dsl import guarded_constant as gc
from tac.witness_dsl import guarded_constant_registry as reg
from tac.witness_dsl.lawref import InputRef, LawRef

# ---------------------------------------------------------------------------
# Stand-ins for the two MEASURED populations.  We do not ship the 117,964,800-site
# cached field in a unit test; instead each fixture is a QUANTILE-MATCHED
# reconstruction: log-linear inverse-CDF interpolation through the percentile
# knots MEASURED by ddm_rt2 on cached n600 GT, so the statistics the guard
# actually reads are the measured ones by construction.  Nothing is invented —
# every knot below is quoted from the memo.
#
#   GT-Lane-restricted (memo §9.2): p1 0.0103  p5 0.0517  p10 0.1047
#                                   p25 0.2766  p50 0.6013
#   GLOBAL field       (memo §4)  : p0.01 0.0036  p0.1 0.0353  p1 0.3552
#                                   p5 2.0582   p50 5.8934
#
# A first attempt used a lognormal shaped only to match p5, and the guard
# REFUSED it (its p10 was 0.0740, not the measured 0.1047).  That refusal is
# itself evidence the guard reads the distribution and not a label.
# ---------------------------------------------------------------------------
LANE_QUANTILE_KNOTS = ((1.0, 0.0103), (5.0, 0.0517), (10.0, 0.1047), (25.0, 0.2766), (50.0, 0.6013))
GLOBAL_QUANTILE_KNOTS = (
    (0.01, 0.0036), (0.1, 0.0353), (1.0, 0.3552), (5.0, 2.0582), (50.0, 5.8934),
)


def _quantile_matched_population(knots, *, n: int = 200_000):
    """Deterministic population whose percentiles reproduce the measured knots."""
    pcts = np.array([k[0] for k in knots], dtype=np.float64)
    vals = np.log(np.array([k[1] for k in knots], dtype=np.float64))
    # Evaluate the interpolated inverse-CDF on a regular percentile grid; the
    # empirical percentiles of the result reproduce the knots (log-linear between).
    grid = (np.arange(n, dtype=np.float64) + 0.5) * 100.0 / n
    return np.exp(np.interp(grid, pcts, vals))


@pytest.fixture(scope="module")
def lane_restricted_sample():
    return _quantile_matched_population(LANE_QUANTILE_KNOTS)


@pytest.fixture(scope="module")
def global_field_sample():
    return _quantile_matched_population(GLOBAL_QUANTILE_KNOTS)


def test_fixtures_reproduce_the_measured_percentiles(
    lane_restricted_sample, global_field_sample
):
    """The fixture is only admissible if it reproduces what was MEASURED.

    Tolerance 1% relative: the percentile grid has 200,000 points, so the
    sub-1% knots (global p0.01 = 20 samples) carry visible grid-resolution
    error.  Every knot above p1 lands inside 0.2%.
    """
    for pct, val in LANE_QUANTILE_KNOTS:
        assert float(np.percentile(lane_restricted_sample, pct)) == pytest.approx(val, rel=1e-2)
    for pct, val in GLOBAL_QUANTILE_KNOTS:
        assert float(np.percentile(global_field_sample, pct)) == pytest.approx(val, rel=1e-2)


# ===========================================================================
# POSITIVE CONTROL 1 (instance a) — STALENESS: a declared derivation not invoked.
# ===========================================================================
def test_positive_control_staleness_declared_derivation_not_invoked_refuses():
    """The derivation exists, is documented, and the frozen literal ships.

    ``derive_margin_floor`` says "data-derived per run, never a bare constant"
    and the shipped value is the frozen ``0.1``.  Nothing fails today (the value
    is right to 1.047x) — which is exactly why a value-comparison test cannot
    catch it and a STRUCTURAL guard must.
    """
    with pytest.raises(gc.StaleDerivation) as exc:
        reg.MARGIN_FLOOR.resolve(consumer_role=gc.ROLE_THRESHOLD)  # no sample
    msg = str(exc.value)
    assert "STALE-DERIVATION rule fired" in msg
    assert "tac.optimization.lane_guard:derive_margin_floor" in msg
    assert "FIX:" in msg


def test_staleness_control_is_not_vacuous_the_guard_passes_when_invoked(
    lane_restricted_sample,
):
    """The mirror of control 1: with the sample supplied the derivation RUNS.

    Without this, control 1 would be satisfied by a guard that refuses always.
    """
    out = reg.MARGIN_FLOOR.resolve(
        consumer_role=gc.ROLE_THRESHOLD,
        sample=lane_restricted_sample,
        sample_domain_id=reg.LANE_MARGIN_DOMAIN.domain_id,
    )
    assert out.status == gc.STATUS_PASS, out.violations
    assert out.derivation_invoked is True
    assert out.derived_value is not None
    # The live derivation's output becomes the value (class-1 derived_live).
    assert out.value == pytest.approx(out.derived_value)


def test_staleness_refuses_when_the_frozen_value_has_rotted(lane_restricted_sample):
    """If the data moves far enough, the frozen value is declared ROTTED."""
    rotted = gc.GuardedConstant(
        constant_id="rotted_probe",
        lawref=LawRef(
            equation_id="dsl_custodied_scalar_identity_v1",
            inputs={"value": InputRef.literal(0.1, provenance="frozen incumbent")},
            ladder_class="hardcoded_waiver",
            fallback=0.1,
            fallback_waiver_reason="probe",
        ),
        units=gc.Units.parse("dimensionless"),
        role=gc.ROLE_THRESHOLD,
        domain=gc.DomainSpec(domain_id="probe", description="probe population"),
        derivation=gc.DerivationRef(
            callable_path="tac.optimization.lane_guard:derive_margin_floor",
            kwargs={"pct": 10.0},
            returns_tuple_index=0,
            drift_tolerance_ratio=1.10,
            tolerance_provenance="probe tolerance mirroring the registry declaration",
        ),
        incumbent_literal=0.1,
    )
    # A population 100x hotter than the one the frozen value was derived on.
    with pytest.raises(gc.StaleDerivation) as exc:
        rotted.resolve(
            consumer_role=gc.ROLE_THRESHOLD,
            sample=lane_restricted_sample * 100.0,
            sample_domain_id="probe",
        )
    assert "ROTTED" in str(exc.value)


# ===========================================================================
# POSITIVE CONTROL 2 (instance b) — DOMAIN: the wrong input distribution.
# ===========================================================================
def test_positive_control_domain_declared_id_mismatch_refuses():
    """Line 1 of defence: the caller NAMES a different distribution."""
    with pytest.raises(gc.DomainViolation) as exc:
        reg.MARGIN_FLOOR.resolve(
            consumer_role=gc.ROLE_THRESHOLD,
            sample=np.array([0.05, 0.1, 0.2]),
            sample_domain_id="qa80_margin__global_field",
        )
    msg = str(exc.value)
    assert "DOMAIN rule fired" in msg
    assert "qa80_margin__gt_lane_restricted" in msg
    assert "The VALUE may be fine; the DOMAIN is not" in msg


def test_positive_control_domain_statistical_witness_refuses_global_field(
    global_field_sample,
):
    """Line 2 of defence: the caller MISLABELS the global field as lane-restricted.

    This is the harder half of the (b) instance — the declared id is 'correct',
    so only the statistic can tell the two populations apart.
    """
    with pytest.raises(gc.DomainViolation) as exc:
        reg.MARGIN_FLOOR.resolve(
            consumer_role=gc.ROLE_THRESHOLD,
            sample=global_field_sample,
            sample_domain_id=reg.LANE_MARGIN_DOMAIN.domain_id,
        )
    msg = str(exc.value)
    assert "DIFFERENT POPULATION" in msg
    assert "BAND PROVENANCE:" in msg


def test_domain_band_is_derived_not_asserted():
    """The band is the log-space max-margin separator of two MEASURED endpoints."""
    check = gc.DomainCheck.max_margin_separator(
        kind="quantile",
        param=5.0,
        in_domain_value=reg.LANE_MARGIN_P5,
        out_of_domain_value=reg.GLOBAL_MARGIN_P5,
        provenance="ddm_rt2 n600",
    )
    g = (reg.GLOBAL_MARGIN_P5 / reg.LANE_MARGIN_P5) ** 0.5
    assert check.lo == pytest.approx(reg.LANE_MARGIN_P5 / g)
    assert check.hi == pytest.approx(reg.LANE_MARGIN_P5 * g)
    # The in-domain statistic sits inside; the out-of-domain statistic does not.
    assert check.lo <= reg.LANE_MARGIN_P5 <= check.hi
    assert not (check.lo <= reg.GLOBAL_MARGIN_P5 <= check.hi)
    # And a statistic that does NOT separate the populations is refused outright,
    # rather than silently producing a vacuous band.
    with pytest.raises(gc.GuardedConstantError, match="does NOT separate"):
        gc.DomainCheck.max_margin_separator(
            kind="mean", in_domain_value=1.0, out_of_domain_value=1.0, provenance="p"
        )


# ===========================================================================
# POSITIVE CONTROL 3 (instance c-i) — ROLE: exchange rate used as a cost.
# ===========================================================================
def test_positive_control_exchange_rate_consumed_as_mechanism_cost_refuses():
    with pytest.raises(gc.RoleMisuse) as exc:
        reg.W_EXCHANGE_RATE.resolve(consumer_role=gc.ROLE_MECHANISM_COST)
    msg = str(exc.value)
    assert "ROLE rule fired" in msg
    assert "exchange_rate" in msg and "mechanism_cost" in msg
    assert "65x" in msg  # the measured mechanism-cost span, carried in the reason
    assert "FIX:" in msg


def test_role_control_is_not_vacuous_the_declared_role_passes():
    out = reg.W_EXCHANGE_RATE.resolve(consumer_role=gc.ROLE_EXCHANGE_RATE)
    assert out.status == gc.STATUS_PASS, out.violations
    assert out.value == pytest.approx(reg.W_BYTES_PER_FLIP)


def test_W_is_the_exact_closed_form():
    """W re-derived from the score function, not recalled: 4*DEN/(n*H*W)."""
    from fractions import Fraction

    exact = Fraction(100 * reg.CONTEST_RATE_DENOMINATOR, 25 * reg.N_PAIRS * reg.FRAME_H * reg.FRAME_W)
    assert exact == Fraction(4171721, 3276800)
    assert reg.W_BYTES_PER_FLIP == float(exact) == 1.2731082153320312


# ===========================================================================
# POSITIVE CONTROL 4 (instance c-ii) — UNITS: bits/flip vs bits/band_pixel.
# ===========================================================================
def test_positive_control_bits_per_flip_vs_bits_per_band_pixel_refuses():
    """The UNIT was wrong, not the value — and it changed the design conclusion."""
    published = gc.Quantity.of(5.91, "bits/flip")
    correct = gc.Quantity.of(0.60, "bits/band_pixel")
    with pytest.raises(gc.UnitMismatch) as exc:
        published.compare_to(correct)
    msg = str(exc.value)
    assert "UNIT rule fired" in msg
    assert "bits/flip" in msg and "bits/band_pixel" in msg
    assert "FIX:" in msg
    with pytest.raises(gc.UnitMismatch):
        published + correct


def test_units_control_is_not_vacuous_same_dimension_compares():
    a = gc.Quantity.of(5.91, "bits/flip")
    b = gc.Quantity.of(1.97, "bits/flip")
    assert a.compare_to(b) == pytest.approx(3.0)
    assert (a - b).value == pytest.approx(3.94)


def test_units_parse_and_normalize():
    assert gc.Units.parse("B/flip") == gc.Units.parse("bytes/flips")
    assert gc.Units.parse("bytes/flip") != gc.Units.parse("bytes/component")
    assert gc.Units.parse("bits/flip") != gc.Units.parse("bytes/flip")  # NOT convertible
    assert gc.Units.parse("1").is_dimensionless()
    assert gc.Units.parse("dimensionless").is_dimensionless()
    assert str(gc.Units.parse("bytes*pixel^-2")) == "bytes/pixel^2"
    with pytest.raises(gc.GuardedConstantError):
        gc.Units.parse("bytes/@flip")


# ===========================================================================
# VACUITY — an empty scope is VACUOUS, never PASS.
# ===========================================================================
def test_zero_checks_reports_vacuous_not_pass():
    bare = gc.GuardedConstant(
        constant_id="bare",
        lawref=LawRef(
            equation_id="dsl_custodied_scalar_identity_v1",
            inputs={"value": InputRef.literal(1.0, provenance="probe")},
            ladder_class="measured_anchor",
        ),
        units=gc.Units.parse("dimensionless"),
        role=gc.ROLE_SCALE,
        domain=gc.DomainSpec(domain_id="d", description="probe"),
    )
    out = bare.audit()  # no role, no sample, no domain id, no derivation
    assert out.status == gc.STATUS_VACUOUS
    assert out.checks_run == 0
    assert out.checks_declared > 0  # the denominator is REPORTED, not hidden
    assert all(c.ok is None for c in out.checks)


def test_empty_sample_refuses_rather_than_passing_vacuously():
    with pytest.raises(gc.DomainViolation, match="ZERO"):
        reg.MARGIN_FLOOR.resolve(
            consumer_role=gc.ROLE_THRESHOLD,
            sample=np.array([np.nan, np.inf]),
            sample_domain_id=reg.LANE_MARGIN_DOMAIN.domain_id,
        )


# ===========================================================================
# LOUD-on-read vs FAIL-CLOSED-on-write.
# ===========================================================================
def test_audit_is_loud_but_does_not_raise():
    out = reg.MARGIN_FLOOR.audit(consumer_role=gc.ROLE_THRESHOLD)
    assert out.status == gc.STATUS_REFUSED
    assert any("STALE-DERIVATION" in v for v in out.violations)
    # LOUD: the violation is in the record, and the value is still readable so a
    # caller can migrate byte-identically while SEEING what is unproven.
    assert out.value == reg.SHIPPED_MARGIN_FLOOR


def test_loud_mode_records_violations_without_raising():
    out = reg.MARGIN_FLOOR.resolve(consumer_role=gc.ROLE_THRESHOLD, mode=gc.MODE_LOUD)
    assert out.status == gc.STATUS_REFUSED
    assert out.violations


# ===========================================================================
# CORROBORATION — agreement is represented, disagreement refuses.
# ===========================================================================
def test_two_agreeing_routes_are_represented_not_resolved_away(lane_restricted_sample):
    out = reg.MARGIN_FLOOR.resolve(
        consumer_role=gc.ROLE_THRESHOLD,
        sample=lane_restricted_sample,
        sample_domain_id=reg.LANE_MARGIN_DOMAIN.domain_id,
    )
    assert out.corroboration_max_ratio == pytest.approx(
        reg.LANE_DERIVED_FLOOR / reg.L7_FP32_DRIFT_BOUND, rel=1e-9
    )
    assert out.corroboration_max_ratio <= reg.MARGIN_FLOOR.corroboration_tolerance_ratio
    # Both routes survive in the declaration — the L7 fp32 portability guard is
    # not discarded in favour of the "winner".
    assert {c.route_id for c in reg.MARGIN_FLOOR.corroborations} == {
        "l7_fp32_cross_hardware_drift_bound",
        "gt_lane_restricted_margin_p10",
    }


def test_disagreeing_routes_refuse():
    conflicted = gc.GuardedConstant(
        constant_id="conflicted",
        lawref=LawRef(
            equation_id="dsl_custodied_scalar_identity_v1",
            inputs={"value": InputRef.literal(1.0, provenance="probe")},
            ladder_class="measured_anchor",
        ),
        units=gc.Units.parse("dimensionless"),
        role=gc.ROLE_SCALE,
        domain=gc.DomainSpec(domain_id="d", description="probe"),
        corroborations=(
            gc.Corroboration(route_id="a", value=1.0, provenance="route a"),
            gc.Corroboration(route_id="b", value=9.0, provenance="route b"),
        ),
        corroboration_tolerance_ratio=1.5,
        corroboration_tolerance_provenance="probe",
    )
    with pytest.raises(gc.CorroborationConflict, match="do not average"):
        conflicted.resolve(consumer_role=gc.ROLE_SCALE)


# ===========================================================================
# Declaration hygiene — a band/tolerance without provenance is refused.
# ===========================================================================
def test_band_without_provenance_refuses():
    with pytest.raises(gc.GuardedConstantError, match="provenance"):
        gc.DomainCheck(kind="mean", lo=0.0, hi=1.0, provenance="")


def test_tolerance_without_provenance_refuses():
    with pytest.raises(gc.GuardedConstantError, match="tolerance_provenance"):
        gc.DerivationRef(callable_path="m:f", drift_tolerance_ratio=2.0)


def test_consumer_role_has_no_default():
    """A consumer that will not say what it is using the number FOR is the (c)
    instance; ``resolve`` therefore has no ``consumer_role`` default."""
    with pytest.raises(TypeError):
        reg.W_EXCHANGE_RATE.resolve()  # type: ignore[call-arg]


# ===========================================================================
# Determinism + migration honesty.
# ===========================================================================
def test_resolution_hash_is_deterministic_and_excludes_timestamps():
    a = reg.W_EXCHANGE_RATE.resolve(consumer_role=gc.ROLE_EXCHANGE_RATE)
    b = reg.W_EXCHANGE_RATE.resolve(consumer_role=gc.ROLE_EXCHANGE_RATE)
    assert a.resolution_hash == b.resolution_hash
    assert len(a.resolution_hash) == 64


def test_resolution_hash_changes_when_the_value_changes(lane_restricted_sample):
    frozen = reg.MARGIN_FLOOR.audit(consumer_role=gc.ROLE_THRESHOLD)
    live = reg.MARGIN_FLOOR.resolve(
        consumer_role=gc.ROLE_THRESHOLD,
        sample=lane_restricted_sample,
        sample_domain_id=reg.LANE_MARGIN_DOMAIN.domain_id,
    )
    assert frozen.resolution_hash != live.resolution_hash


def test_migration_byte_identity_is_proven_not_assumed():
    frozen = reg.MARGIN_FLOOR.audit(consumer_role=gc.ROLE_THRESHOLD)
    assert frozen.byte_identical_to_incumbent is True
    assert frozen.behaviour_change is False
    assert reg.MARGIN_FLOOR_INCUMBENT == reg.SHIPPED_MARGIN_FLOOR
    assert reg.MARGIN_FLOOR_MIGRATION_IS_BYTE_IDENTICAL is True


def test_adopting_the_derived_value_is_flagged_as_a_behaviour_change(
    lane_restricted_sample,
):
    live = reg.MARGIN_FLOOR.resolve(
        consumer_role=gc.ROLE_THRESHOLD,
        sample=lane_restricted_sample,
        sample_domain_id=reg.LANE_MARGIN_DOMAIN.domain_id,
    )
    assert live.byte_identical_to_incumbent is False
    assert live.behaviour_change is True


# ===========================================================================
# Bridge to the existing Provenanced model (no parallel SoT).
# ===========================================================================
def test_bridges_to_existing_provenanced_model(lane_restricted_sample):
    from tac.witness_dsl.typed_config import ProvenanceClass

    live = reg.MARGIN_FLOOR.resolve(
        consumer_role=gc.ROLE_THRESHOLD,
        sample=lane_restricted_sample,
        sample_domain_id=reg.LANE_MARGIN_DOMAIN.domain_id,
    )
    p = live.to_provenanced()
    assert p.provenance is ProvenanceClass.DERIVED_LIVE
    assert p.unit == "dimensionless"
    assert p.source.startswith("guarded_constant:seg_margin_hinge_floor#")

    frozen = reg.MARGIN_FLOOR.audit(consumer_role=gc.ROLE_THRESHOLD)
    pf = frozen.to_provenanced()
    assert pf.provenance is ProvenanceClass.HARDCODED_WITH_WAIVER
    assert pf.waiver and "STALE-DERIVATION" in pf.waiver


def test_to_dict_reports_the_denominator():
    d = reg.MARGIN_FLOOR.audit(consumer_role=gc.ROLE_THRESHOLD).to_dict()
    assert "checks_run" in d and "checks_declared" in d
    assert d["checks_declared"] >= d["checks_run"]


# ===========================================================================
# Registry sanity.
# ===========================================================================
def test_registry_declarations_are_wellformed():
    assert set(reg.REGISTRY) == {"seg_margin_hinge_floor", "flip_byte_exchange_rate_W"}
    for cid, const in reg.REGISTRY.items():
        assert const.constant_id == cid
        assert const.role in gc.VALID_ROLES
        assert const.domain.description.strip()
        for check in const.domain.checks:
            assert check.provenance.strip()
