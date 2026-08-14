from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_qs4_collateral_suppression as qs4
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def test_change_classes_partition_b_h_w() -> None:
    base = np.asarray([[0, 0, 1, 1, 2]], dtype=np.uint8)
    candidate = np.asarray([[1, 2, 0, 2, 3]], dtype=np.uint8)
    gt = np.asarray([[1, 0, 1, 3, 4]], dtype=np.uint8)
    changed, beneficial, harmful, wrong = qs4.classify_changes(base, candidate, gt)
    assert np.count_nonzero(changed) == 5
    assert np.count_nonzero(beneficial) == 1
    assert np.count_nonzero(harmful) == 2
    assert np.count_nonzero(wrong) == 2


def test_pose_adjusted_admission_is_strict() -> None:
    result = qs4.admission_metrics(
        expected_net_flips=8.0, kept_sites=6, bytes_per_pair=5.7
    )
    expected_rate = 5.7 * qs4.RATE_S_PER_BYTE
    expected_bar = qs4.BREAKEVEN_FLIPS_PER_BYTE * (
        1.0 + 6 * qs4.POSE_S_PER_EDIT_CELL / expected_rate
    )
    assert result["pose_adjusted_admission_bar_flips_per_byte"] == pytest.approx(
        expected_bar
    )
    assert result["expected_flips_per_byte"] == pytest.approx(8.0 / 5.7)
    assert result["passes"] is True


def test_nearest_assignment_returns_edited_site() -> None:
    mask = np.zeros((qs4.HEIGHT, qs4.WIDTH), dtype=np.bool_)
    mask[10, 10] = True
    mask[10, 20] = True
    distances, nearest = qs4._nearest_assignment(mask)
    assert distances[10, 10] == 0.0
    assert nearest[10, 10] == 10 * qs4.WIDTH + 10
    assert nearest[10, 20] == 10 * qs4.WIDTH + 20


def test_qs1_materializer_accepts_retained_custom_token_path() -> None:
    source = Path(qs4.qs1.__file__).read_text()
    assert 'row.get(\n                "candidate_tokens_path"' in source
    assert 'expected_sites = row.get("token_site_count")' in source


def test_qs4_runner_does_not_measure_and_discard_payloads() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=qs4.REPO,
        roots=[
            Path("experiments/ddm_qs1_frame0_schur_coupled_solve.py"),
            Path("experiments/ddm_qs3_saturation_compose.py"),
            Path("experiments/ddm_qs4_collateral_suppression.py"),
        ],
    )
    assert findings == []
