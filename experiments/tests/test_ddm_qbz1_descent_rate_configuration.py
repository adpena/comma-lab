# SPDX-License-Identifier: MIT
"""Scorer-free contract tests for the qbz1 capacity probe."""

from __future__ import annotations

import math

import torch

from experiments import ddm_qbz1_descent_rate_configuration as qbz1


def test_pair_split_is_deterministic_disjoint_and_real_n() -> None:
    train_a, holdout_a = qbz1.split_pair_ids(qbz1.SEED)
    train_b, holdout_b = qbz1.split_pair_ids(qbz1.SEED)
    assert (train_a, holdout_a) == (train_b, holdout_b)
    assert len(train_a) == 480
    assert len(holdout_a) == 120
    assert set(train_a).isdisjoint(holdout_a)
    assert set(train_a) | set(holdout_a) == set(range(qbz1.N))


def test_spatial_holdout_is_pair_specific_and_near_one_fifth() -> None:
    first = qbz1.spatial_holdout_mask(0)
    second = qbz1.spatial_holdout_mask(1)
    assert first.shape == (qbz1.H, qbz1.W)
    assert first.dtype == torch.bool
    assert not torch.equal(first, second)
    for mask in (first, second):
        fraction = float(mask.float().mean())
        assert math.isclose(fraction, 0.2, abs_tol=5.0e-4)


def test_schedule_covers_declared_epochs_and_updates() -> None:
    rows, boundaries = qbz1.schedule(qbz1.SEED)
    expected = 480 * qbz1.PAIR_HOLDOUT_EPOCHS + 600 * qbz1.SPATIAL_HOLDOUT_EPOCHS
    assert len(rows) == expected
    assert boundaries[f"pair_holdout_epoch_{qbz1.PAIR_HOLDOUT_EPOCHS:02d}"] == 960
    assert boundaries[f"spatial_holdout_epoch_{qbz1.SPATIAL_HOLDOUT_EPOCHS:02d}"] == expected
    assert sum(row["phase"] == "pair_holdout" for row in rows) == 960
    assert sum(row["phase"] == "spatial_holdout" for row in rows) == 6000


def test_rederived_scores_match_retained_components_exactly() -> None:
    receipt = qbz1.rederive_source_series()
    assert receipt["level_comparison_refused"] is True
    assert [row["run"] for row in receipt["rows"]] == ["r7", "r8", "r9", "r10"]
    for row in receipt["rows"]:
        expected = (
            100.0 * row["d_seg_hat"]
            + math.sqrt(10.0 * row["d_pose_hat"])
            + 25.0 * row["B_hat"] / qbz1.RATE_DENOMINATOR
        )
        assert row["S_recomputed"] == expected
        assert row["S_reproduces_abs_error"] == 0.0
        assert row["estimator_status"] == "NO2_SECTION5_HT_COMPLETE"
        assert row["selection_count"] == 32
        assert len(row["unmet_gates"]) == 3


def test_realization_claim_gate_refuses_foreign_lane() -> None:
    try:
        qbz1.assert_active_scorer_claim("ddm_fcd3_scorer_20260829")
    except qbz1.QBZ1Error as exc:
        assert "qbz1-owned" in str(exc)
    else:
        raise AssertionError("foreign scorer claim was accepted")


def test_nonpositive_learning_rate_is_refused_before_materialization(tmp_path) -> None:
    try:
        qbz1.run_fit(tmp_path, learning_rate=0.0, resume_from=None)
    except qbz1.QBZ1Error as exc:
        assert "finite and positive" in str(exc)
    else:
        raise AssertionError("zero learning rate was accepted")


def test_claim_gate_uses_newest_lane_row_and_refuses_live_conflict(tmp_path, monkeypatch) -> None:
    registry = tmp_path / "claims.md"
    registry.write_text(
        "| 2099-08-29T19:00:00Z | MAIN | ddm_qbz1_scorer_20990829 | local_macos_cpu | qbz | | active_local_advisory | owned |\n"
        "| 2099-08-29T18:59:00Z | other | ddm_fcd3_scorer_20990829 | local_macos_cpu | fcd | | active_local_advisory | conflict |\n"
        "| 2099-08-29T18:58:00Z | old | ddm_qbz1_scorer_20990829 | local_macos_cpu | qbz-old | | completed | superseded |\n"
    )
    monkeypatch.setattr(qbz1, "ACTIVE_CLAIMS", registry)
    try:
        qbz1.assert_active_scorer_claim("ddm_qbz1_scorer_20990829")
    except qbz1.QBZ1Error as exc:
        assert "another live scorer" in str(exc)
    else:
        raise AssertionError("concurrent scorer claim was accepted")
