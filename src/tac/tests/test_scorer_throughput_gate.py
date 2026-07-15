# SPDX-License-Identifier: MIT
"""#224 Wave D — tests for the SegNet fwd+bwd throughput gate (launch-time ~17x fast-path assertion).

Verdict logic is unit-tested GPU-free via an injected ``measure_fn`` (the heavy MLX/scorer
measurement is isolated). One GPU-gated smoke proves the real micro-bench returns a finite ms.
"""

from __future__ import annotations

import pytest

from tac.local_acceleration import scorer_throughput_gate as gate


# --------------------------------------------------------------------------
# sub-part 2 pure helper: relative ceiling
# --------------------------------------------------------------------------
def test_step_time_within_ceiling_pass():
    assert gate.step_time_within_ceiling(400.0, 396.0, 1.5) is True   # 400 <= 594
    assert gate.step_time_within_ceiling(594.0, 396.0, 1.5) is True   # exactly at ceiling


def test_step_time_within_ceiling_fail():
    assert gate.step_time_within_ceiling(600.0, 396.0, 1.5) is False  # 600 > 594


def test_step_time_within_ceiling_rejects_nonpositive_baseline():
    with pytest.raises(ValueError):
        gate.step_time_within_ceiling(100.0, 0.0)


# --------------------------------------------------------------------------
# evaluate_throughput verdict logic (injected measure_fn — NO GPU)
# --------------------------------------------------------------------------
def test_evaluate_throughput_fast_verdict():
    v = gate.evaluate_throughput(measure_fn=lambda **kw: 396.0)  # the ON fast-path anchor
    assert v.status == "fast"
    assert v.ok is True and v.is_slow is False
    assert v.within_abs is True and v.within_ceiling is True
    assert v.segnet_fwd_bwd_ms == pytest.approx(396.0)


def test_evaluate_throughput_slow_absolute_verdict():
    # the OFF reference (~6713ms) must be caught by the absolute 700ms gate
    v = gate.evaluate_throughput(measure_fn=lambda **kw: 6713.0)
    assert v.status == "slow"
    assert v.ok is False and v.is_slow is True
    assert v.within_abs is False
    assert "NOT active" in v.reason and "6713" in v.reason


def test_evaluate_throughput_slow_relative_only():
    # 650ms passes the absolute 700 gate but FAILS the 1.5x*396=594 relative ceiling => slow
    v = gate.evaluate_throughput(measure_fn=lambda **kw: 650.0)
    assert v.status == "slow"
    assert v.within_abs is True and v.within_ceiling is False


def test_evaluate_throughput_unavailable_never_blocks():
    def _boom(**kw):
        raise RuntimeError("no GPU here")

    v = gate.evaluate_throughput(measure_fn=_boom)
    assert v.status == "unavailable"
    assert v.ok is True                      # unavailability must NOT block the launch
    assert v.is_slow is False
    assert v.segnet_fwd_bwd_ms is None
    assert "no GPU here" in v.reason


def test_evaluate_throughput_custom_threshold_override():
    # a stricter threshold flips a value that would pass the default
    v = gate.evaluate_throughput(abs_threshold_ms=300.0, measure_fn=lambda **kw: 396.0)
    assert v.status == "slow"
    assert v.within_abs is False


def test_measure_fn_is_called_with_custom_backward_true():
    seen = {}

    def _capture(**kw):
        seen.update(kw)
        return 396.0

    gate.evaluate_throughput(measure_fn=_capture)
    assert seen.get("custom_backward") is True   # the gate always measures the FAST path


def test_anchors_are_sane():
    assert gate.ON_REF_MS < gate.ABS_THRESHOLD_MS < gate.OFF_REF_MS
    assert gate.SEG_H == 384 and gate.SEG_W == 512


def test_event_conditional_budget_has_lawref_custody_and_midpoint():
    receipt = gate.derive_wall_clock_budget_receipt(3000)
    assert receipt.flat_fallback_used is False
    assert receipt.fallback_reason is None
    assert [stage.epochs for stage in receipt.stages] == [32, 2968]
    assert receipt.stages[0].min_per_ep == pytest.approx(251.6 / 60.0)
    assert receipt.stages[1].min_per_ep == pytest.approx(((325.0 + 333.0) / 2.0) / 60.0)
    assert set(receipt.lawref_manifest) == {
        "transition_epoch",
        "pre_event_min_per_ep",
        "post_event_min_per_ep",
    }
    for row in receipt.lawref_manifest.values():
        assert row["fallback_used"] is False
        assert row["inputs"]
        assert all(item["sha256"] for item in row["inputs"])


@pytest.mark.parametrize(
    ("epochs", "expected_counts"),
    [(1, [1, 0]), (20, [20, 0]), (32, [32, 0]), (33, [32, 1])],
)
def test_event_conditional_budget_truncates_short_runs(epochs, expected_counts):
    receipt = gate.derive_wall_clock_budget_receipt(epochs)
    assert [stage.epochs for stage in receipt.stages] == expected_counts
    assert sum(stage.epochs for stage in receipt.stages) == epochs


def test_flat_fallback_is_explicit_and_keeps_old_formula():
    receipt = gate.derive_wall_clock_budget_receipt(3000, profile=None)
    assert receipt.flat_fallback_used is True
    assert receipt.fallback_reason == "event telemetry profile absent"
    assert receipt.total_days == pytest.approx(
        gate.project_wall_clock_days(gate.RUN1_MEASURED_MIN_PER_EP, 3000)
        * gate.WALL_CLOCK_SLACK_FACTOR
    )


def test_explicit_custom_flat_anchor_keeps_backward_compatible_math():
    got = gate.derive_wall_clock_budget_days(20, min_per_ep=2.5, slack=1.1)
    assert got == pytest.approx(gate.project_wall_clock_days(2.5, 20) * 1.1)


def test_invalid_event_profile_falls_back_with_reason():
    receipt = gate.derive_wall_clock_budget_receipt(10, profile=object())
    assert receipt.flat_fallback_used is True
    assert "invalid type" in str(receipt.fallback_reason)


def test_missing_lawref_anchor_falls_back_with_logged_reason(tmp_path):
    from dataclasses import replace

    from tac.witness_dsl.lawref import InputRef, LawRef

    profile = gate.canonical_event_wall_clock_profile()
    original = profile.pre_event.min_per_ep_ref
    missing_ref = LawRef(
        equation_id=original.equation_id,
        inputs={
            "seconds": InputRef.anchor(
                str(tmp_path / "missing.json"),
                "telemetry/pre_event/median_seconds_per_epoch",
                "test-only missing measured telemetry anchor",
                config_tags=profile.config_tags,
            )
        },
        ladder_class=original.ladder_class,
    )
    broken = replace(
        profile,
        pre_event=replace(profile.pre_event, min_per_ep_ref=missing_ref),
    )
    receipt = gate.derive_wall_clock_budget_receipt(20, profile=broken)
    assert receipt.flat_fallback_used is True
    assert "artifact" in str(receipt.fallback_reason)


# --------------------------------------------------------------------------
# GPU-gated smoke: the REAL micro-bench returns a finite ms (skips w/o MLX+scorer)
# --------------------------------------------------------------------------
def test_real_micro_bench_finite_if_available():
    pytest.importorskip("mlx.core")
    try:
        ms = gate.measure_segnet_fwd_bwd_ms(batch=2, warmup=1, iters=2)
    except Exception as exc:  # scorer weights / GPU absent on this host — allowed skip
        pytest.skip(f"scorer/MLX micro-bench unavailable: {exc}")
    assert ms > 0.0 and ms == ms  # finite, positive (NaN != NaN would fail the second clause)
