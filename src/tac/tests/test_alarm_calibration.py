# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac.alarm_calibration import (
    PValueObservation,
    adjudicate_alarm_family,
    alarm_registry_json,
    benjamini_hochberg,
    default_alarm_registry,
    lp1_lane_guard_ratchet_null_reproduction,
    rows_for_consumer,
    split_conformal_p_value,
)


def test_registry_has_cf1_named_alarm_rows_and_repaired_a1():
    rows = {row.alarm_id: row for row in default_alarm_registry()}
    assert {
        "A1_REALIZATION_GAP_ALARM",
        "lane_guard.ratchet",
        "term_domination",
        "term_inert",
        "gnorm_hijack",
    } <= set(rows)
    assert rows["A1_REALIZATION_GAP_ALARM"].fdr_family == "realization_gap"
    assert rows["A1_REALIZATION_GAP_ALARM"].block_calibration_required is True
    assert rows["A1_REALIZATION_GAP_ALARM"].statistic_status == "ready"
    assert "realized_gate_dseg_mean_ht" in rows["A1_REALIZATION_GAP_ALARM"].score
    assert rows["lane_guard.ratchet"].block_calibration_required is True
    assert rows["term_domination"].exchangeability_grade == "partial_stage_scoped"
    assert rows["term_inert"].fdr_family == "loss_term"
    assert rows["gnorm_hijack"].p_value_direction == "greater"


def test_registry_json_is_queryable_not_just_memo_text():
    payload = alarm_registry_json()
    assert payload["schema"] == "tac_l1_alarm_registry.v1"
    assert "diagnostic alarm calibration only" in payload["authority_boundary"]
    consumers = rows_for_consumer("burn supervisor")
    assert {row.alarm_id for row in consumers} == {
        "A1_REALIZATION_GAP_ALARM",
        "term_domination",
    }
    assert {row.alarm_id for row in rows_for_consumer("a1 stage-exit")} == {
        "A1_REALIZATION_GAP_ALARM"
    }
    with pytest.raises(ValueError, match="consumer_query"):
        rows_for_consumer("")


def test_split_conformal_high_tail_positive_and_negative():
    cal = [0.1, 0.2, 0.3, 0.4]
    high = split_conformal_p_value(cal, 0.5, direction="greater")
    assert high.tail_count == 0
    assert high.p_value == pytest.approx(1.0 / 5.0)
    not_high = split_conformal_p_value(cal, 0.15, direction="greater")
    assert not_high.p_value == pytest.approx(4.0 / 5.0)
    low = split_conformal_p_value(cal, 0.05, direction="less")
    assert low.p_value == pytest.approx(1.0 / 5.0)


def test_split_conformal_superuniform_under_exchangeable_null():
    rng = np.random.default_rng(20260805)
    pvals = []
    for _ in range(2500):
        sample = rng.normal(0.0, 1.0, 33)
        pvals.append(
            split_conformal_p_value(sample[:-1], sample[-1], direction="greater").p_value
        )
    p = np.asarray(pvals)
    for alpha in (0.05, 0.10, 0.20, 0.50):
        assert float(np.mean(p <= alpha)) <= alpha + 0.025


def test_benjamini_hochberg_rejects_prefix_and_computes_q_values():
    res = benjamini_hochberg(
        [
            PValueObservation("b", 0.03, "fam"),
            PValueObservation("a", 0.001, "fam"),
            PValueObservation("c", 0.20, "fam"),
        ],
        alpha=0.05,
    )
    by_id = {r.alarm_id: r for r in res}
    assert by_id["a"].rejected is True
    assert by_id["b"].rejected is True
    assert by_id["c"].rejected is False
    assert by_id["a"].q_value <= by_id["b"].q_value <= by_id["c"].q_value


def test_bh_rejects_mixed_families():
    with pytest.raises(ValueError, match="one FDR family"):
        benjamini_hochberg(
            [
                PValueObservation("a", 0.01, "fam1"),
                PValueObservation("b", 0.01, "fam2"),
            ]
        )


def test_adjudicate_alarm_family_consumes_registry():
    out = adjudicate_alarm_family(
        {"lane_guard.ratchet": 10.0},
        {"lane_guard.ratchet": [0.0, 1.0, 2.0, 3.0]},
        fdr_family="lane_guard",
        alpha=0.25,
    )
    assert len(out) == 1
    assert out[0].registry_row.alarm_id == "lane_guard.ratchet"
    assert out[0].alarm_fires is True


def test_lp1_lane_guard_null_reproduction_suppresses_false_positive():
    rec = lp1_lane_guard_ratchet_null_reproduction()
    assert rec["schema"] == "lp1_lane_guard_ratchet_null_reproduction.v1"
    assert rec["n_trials"] == 20_000
    assert rec["verdict"] == "FALSE_POSITIVE_REPRODUCED_NO_HIGH_TAIL_ALARM"
    null = rec["null"]
    assert null["sum_rises_mean"] == pytest.approx(0.050097, abs=0.001)
    assert null["sum_rises_p5"] == pytest.approx(0.035449, abs=0.001)
    assert null["sum_rises_p95"] == pytest.approx(0.065819, abs=0.001)
    assert 0.004 <= null["sum_rises_low_tail_percentile"] <= 0.010
    adj = rec["adjudications"][0]
    assert adj["registry_row"]["alarm_id"] == "lane_guard.ratchet"
    assert adj["conformal"]["p_value"] > 0.99
    assert adj["alarm_fires"] is False
