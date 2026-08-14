from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_qs3_compensation_overlay_runtime as overlay
from experiments import ddm_qs3_saturation_compose as qs3
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def test_calibrated_efficiency_is_worker_net_over_changed_denominator() -> None:
    assert pytest.approx(32 / 189) == qs3.CALIBRATED_REALIZATION_EFFICIENCY
    assert qs3.MEASURED_CHANGED_PIXELS - qs3.MEASURED_NET_FLIPS == 157


def test_screening_applies_pose_adjusted_admission_bar() -> None:
    row = {
        "proposal_id": "p",
        "pair": 7,
        "directed_edge": "Road->Lane",
        "token_site_count": 2,
        "receiver_surface": {"exact_field_target_edge_mass_on_support": 100},
    }
    result = qs3.screening_row(row, bytes_per_pair=5.0, pose_s_per_cell=1e-9)
    expected_rate = 5.0 * qs3.RATE_S_PER_BYTE
    expected_bar = qs3.BREAKEVEN_FLIPS_PER_BYTE * (1.0 + 2e-9 / expected_rate)
    assert result["pose_adjusted_admission_bar_flips_per_byte"] == pytest.approx(expected_bar)
    assert result["predicted_flips_per_byte"] == pytest.approx((100 * 32 / 189) / 5)
    assert result["admission_status"].startswith("QUEUE_FOR_EXACT_SCHUR")


def test_unique_pair_calibration_has_one_row_per_pair() -> None:
    rows = qs3.unique_pair_calibration_rows()
    pairs = [int(row["pair"]) for row in rows]
    assert len(rows) == 9
    assert pairs == sorted(set(pairs))


def test_nine_pair_overlay_roundtrips_exact_deltas() -> None:
    rows = qs3.unique_pair_calibration_rows()
    pairs, deltas = qs3.qs2.exact_deltas(rows)
    payload = overlay.encode_compensation_overlay(pairs, deltas)
    decoded_pairs, decoded_deltas = overlay.decode_compensation_overlay(payload)
    assert np.array_equal(decoded_pairs, pairs)
    assert np.array_equal(decoded_deltas, deltas)


def test_js6_bank_is_complete_nonprefix_census() -> None:
    rows = qs3.load_js6_rows()
    assert len(rows) == 200
    assert len({json.dumps(row["proposal_id"]) for row in rows}) == 200


def test_qs3_runners_do_not_measure_and_discard_payloads() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=qs3.REPO,
        roots=[
            Path("experiments/ddm_qs3_compensation_overlay_runtime.py"),
            Path("experiments/ddm_qs3_saturation_compose.py"),
        ],
    )
    assert findings == []
