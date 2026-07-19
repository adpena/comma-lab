"""#408/#404 merge-window-prep wire-in tests (p0_328_408 branch).

Two Scope-B deliverables, both READ-ONLY / score-neutral telemetry:

1. ``lever_engage`` schema unification — the canonical ``lever_engage_row`` gains an additive
   ``extra`` param so no emitter hand-rolls a divergent ``{"stage": "lever_engage", ...}`` dict;
   the base schema (stage/lever/status/epoch/via) stays authoritative + stable.
2. rate-rolling rate-proxy soft-signal — the ``--rate-rolling-telemetry`` trainer flag is real +
   DSL-held (``RateRollingTelemetry`` auto-unlocks), the row NEVER carries a kill, and the resume
   contract (proxy_tail -> baseline_from_row continuity) that the trainer's __raterolling_ resume
   registration relies on holds.

Byte-identity of training numerics is preserved BY CONSTRUCTION: every surface here READS
already-materialized state (weights / config) and only prints JSON rows — it adds no loss term,
no gradient, and no optimizer/EMA/model mutation. These are apparatus/MEANS, not a score claim.
"""
import pytest

from tac.witness_control.telemetry_producers import lever_engage_row


# ---------------------------------------------------------------------------
# 1. lever_engage schema unification
# ---------------------------------------------------------------------------
def test_lever_engage_row_base_schema_stable_without_extra():
    r = lever_engage_row("additive_margin", status="fired", epoch=3, via="setup_loss_binding")
    assert set(r) == {"stage", "lever", "status", "epoch", "via"}
    assert r["stage"] == "lever_engage"
    assert r["lever"] == "additive_margin"
    assert r["status"] == "fired"
    assert r["epoch"] == 3


def test_lever_engage_row_extra_is_additive():
    r = lever_engage_row(
        "additive_margin", status="armed", epoch=0, via="typed_configuration",
        extra={"engaged": True, "inert": False, "head": "softmax",
               "additive_margin": 0.1, "margin_field_head_weight": 0.5, "reason": "ok"})
    # base fields intact + extra diagnostics folded into the SAME row.
    assert r["stage"] == "lever_engage" and r["status"] == "armed" and r["via"] == "typed_configuration"
    assert r["engaged"] is True and r["inert"] is False and r["reason"] == "ok"
    assert r["additive_margin"] == 0.1 and r["margin_field_head_weight"] == 0.5


def test_lever_engage_row_extra_cannot_override_reserved_fields():
    for bad in ("stage", "lever", "status", "epoch", "via"):
        with pytest.raises(ValueError, match="collides with a reserved"):
            lever_engage_row("x", status="armed", epoch=0, via="v", extra={bad: "hijack"})


def test_lever_engage_row_empty_or_none_extra_is_noop():
    base = lever_engage_row("x", status="complete", epoch=1, via="v")
    assert lever_engage_row("x", status="complete", epoch=1, via="v", extra=None) == base
    assert lever_engage_row("x", status="complete", epoch=1, via="v", extra={}) == base


def test_lever_engage_row_rejects_invalid_status():
    with pytest.raises(ValueError, match="invalid lever engagement status"):
        lever_engage_row("x", status="pending", epoch=0, via="v")


# ---------------------------------------------------------------------------
# 2. rate-rolling telemetry: flag real + DSL-held + resume contract + never-kills
# ---------------------------------------------------------------------------
def test_rate_rolling_flag_is_a_real_trainer_flag():
    from tac.witness_dsl.curriculum_dsl import real_trainer_flags

    assert "--rate-rolling-telemetry" in real_trainer_flags(None)


def test_rate_rolling_dsl_lever_auto_unlocked_and_maps_the_flag():
    from tac.witness_dsl.constants_telemetry_build_wave_20260715 import RateRollingTelemetry

    lev = RateRollingTelemetry()  # no longer raises TrainerWireInQueued now that the flag landed
    assert lev.name == "rate_rolling_telemetry_row"
    assert lev.overrides == {"--rate-rolling-telemetry": True}


def test_rate_rolling_queue_row_marked_landed():
    from tac.witness_dsl.constants_telemetry_build_wave_20260715 import TRAINER_WIREIN_QUEUE

    rate = TRAINER_WIREIN_QUEUE[-1]
    assert "rate_rolling" in rate["producer"]
    assert "landed" in rate["status"].lower()
    assert "p0_328_408" in rate["status"]


def test_rate_rolling_row_never_carries_a_kill_and_is_informs_only():
    from tac.witness_control.rate_rolling_telemetry import (
        SIGNAL_STATES,
        rate_rolling_row,
    )

    # even a strongly SUSTAINED_GROWTH series only INFORMS (informs_only True, no halt/kill field).
    series = [1.0 + 0.1 * i for i in range(30)]
    row = rate_rolling_row(50, series)
    assert row["stage"] == "rate_rolling"
    assert row["informs_only"] is True
    assert row["drift_signal"] in SIGNAL_STATES
    assert not any(k in row for k in ("kill", "halt", "abort", "revert", "clamp"))


def test_rate_rolling_resume_contract_proxy_tail_roundtrips_continuously():
    """The trainer's __raterolling_ resume registration persists the proxy tail + baseline so the
    rolling mean continues bit-faithfully across a crash. This asserts the underlying contract:
    the emitted proxy_tail rebuilds a baseline that re-emits an IDENTICAL rolling_avg."""
    from tac.witness_control.rate_rolling_telemetry import (
        baseline_from_row,
        rate_rolling_row,
    )

    series = [1.0, 1.02, 0.99, 1.03, 1.05, 1.01, 1.04, 1.06, 1.02, 1.03]
    row_before = rate_rolling_row(30, series)
    # simulate a crash: only the persisted proxy_tail survives (the resume registry restores it).
    resumed_baseline = baseline_from_row(row_before)
    resumed_series = list(resumed_baseline.proxy_tail)
    row_after = rate_rolling_row(30, resumed_series, baseline=resumed_baseline)
    # the rolling average recomputed from the persisted tail matches the pre-crash value.
    assert row_after["rolling_avg"] == pytest.approx(row_before["rolling_avg"])
    assert row_after["rel_from_t0"] == pytest.approx(0.0, abs=1e-12)
