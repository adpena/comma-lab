from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments import ddm_mc35_micro35_union_build as mc36
from tac.payload_retention_gate import check_no_measure_and_discard_payload


def test_signed_neighbours_include_center_and_every_direction() -> None:
    center = np.zeros(mc36.qs1.DIMENSIONS, dtype=np.int32)
    candidates = mc36._signed_int12_neighbours(center)
    assert len(candidates) == 1 + 2 * mc36.qs1.DIMENSIONS
    assert np.array_equal(candidates[0], center)
    assert {tuple(row) for row in candidates} == {
        tuple(center),
        *{
            tuple(np.eye(1, mc36.qs1.DIMENSIONS, dimension, dtype=np.int32)[0] * sign)
            for dimension in range(mc36.qs1.DIMENSIONS)
            for sign in (-1, 1)
        },
    }


def test_signed_neighbours_respect_int12_endpoints() -> None:
    center = np.zeros(mc36.qs1.DIMENSIONS, dtype=np.int32)
    center[0] = 2047
    center[1] = -2048
    candidates = mc36._signed_int12_neighbours(center)
    assert len(candidates) == 1 + 2 * mc36.qs1.DIMENSIONS - 2
    assert all(np.all(row >= -2048) and np.all(row <= 2047) for row in candidates)


def test_signed_neighbours_do_not_alias_input() -> None:
    center = np.zeros(mc36.qs1.DIMENSIONS, dtype=np.int32)
    candidates = mc36._signed_int12_neighbours(center)
    candidates[0][0] = 99
    assert center[0] == 0
    assert candidates[1][0] != 99


def test_pose_delta_is_zero_for_baseline() -> None:
    baseline = np.arange(mc36.qs1.POSE_DIMENSIONS, dtype=np.float32)
    gt = baseline + 0.25
    assert mc36._pose_delta_dpose(baseline, baseline, gt) == 0.0


def test_pose_delta_uses_n600_six_scalar_denominator() -> None:
    baseline = np.zeros(mc36.qs1.POSE_DIMENSIONS, dtype=np.float32)
    gt = np.zeros_like(baseline)
    vector = np.ones_like(baseline)
    assert mc36._pose_delta_dpose(vector, baseline, gt) == pytest.approx(1.0 / 600)


def test_best_pose_feasible_prefers_lowest_baseline_objective() -> None:
    index = mc36._best_pose_feasible_index(
        np.asarray([0.3, 0.1, 0.2]),
        np.asarray([0.0, 2.0, 0.5]),
        1.0,
    )
    assert index == 2


def test_best_pose_feasible_breaks_objective_tie_by_pose_delta() -> None:
    index = mc36._best_pose_feasible_index(
        np.asarray([0.1, 0.1]), np.asarray([0.5, -0.5]), 1.0
    )
    assert index == 1


def test_best_pose_feasible_breaks_full_tie_by_stable_index() -> None:
    index = mc36._best_pose_feasible_index(
        np.asarray([0.1, 0.1]), np.asarray([0.5, 0.5]), 1.0
    )
    assert index == 0


def test_best_pose_feasible_returns_none_when_constraint_is_empty() -> None:
    assert (
        mc36._best_pose_feasible_index(
            np.asarray([0.1, 0.2]), np.asarray([2.0, 3.0]), 1.0
        )
        is None
    )


def test_best_pose_feasible_rejects_geometry_mismatch() -> None:
    with pytest.raises(mc36.MC35Error, match="different geometry"):
        mc36._best_pose_feasible_index(
            np.asarray([0.1]), np.asarray([0.2, 0.3]), 1.0
        )


def test_best_pose_feasible_rejects_nonfinite_metrics() -> None:
    with pytest.raises(mc36.MC35Error, match="non-finite"):
        mc36._best_pose_feasible_index(
            np.asarray([np.nan]), np.asarray([0.0]), 1.0
        )


def test_complementary_gate_coverage_finds_both_fire_orders() -> None:
    left_only, right_only = mc36._complementary_gate_coverage(
        {"seg": True, "rate": False, "pose": True},
        {"seg": True, "rate": True, "pose": False},
    )
    assert left_only == {"pose"}
    assert right_only == {"rate"}


def test_complementary_gate_coverage_rejects_different_gate_sets() -> None:
    with pytest.raises(mc36.MC35Error, match="gate names differ"):
        mc36._complementary_gate_coverage({"seg": True}, {"pose": True})


def test_composition_evidence_accepts_measured_metric_complementarity() -> None:
    pair = {
        "gates": {
            "net_flips_gte_35": True,
            "delta_bytes_lte_29": False,
            "delta_dpose_lte_cap": True,
            "receiver_parseback": True,
        }
    }
    drop = {
        "gates": {
            "net_flips_gte_35": True,
            "delta_bytes_lte_29": False,
            "delta_dpose_lte_cap": False,
            "receiver_parseback": True,
        }
    }
    evidence = mc36._composition_evidence(
        pair_result=pair,
        drop_result=drop,
        pair_recount={
            "seg": {"net_flip_gain": 35},
            "rate": {"candidate_archive_bytes": 186_313},
        },
        drop_recount={
            "seg": {"net_flip_gain": 37},
            "rate": {"candidate_archive_bytes": 186_292},
        },
    )
    assert evidence["justified"] is True


def test_composition_evidence_rejects_absent_rate_relief() -> None:
    pair = {
        "gates": {
            "net_flips_gte_35": True,
            "delta_bytes_lte_29": False,
            "delta_dpose_lte_cap": True,
            "receiver_parseback": True,
        }
    }
    drop = {
        "gates": {
            "net_flips_gte_35": True,
            "delta_bytes_lte_29": False,
            "delta_dpose_lte_cap": False,
            "receiver_parseback": True,
        }
    }
    evidence = mc36._composition_evidence(
        pair_result=pair,
        drop_result=drop,
        pair_recount={
            "seg": {"net_flip_gain": 35},
            "rate": {"candidate_archive_bytes": 186_300},
        },
        drop_recount={
            "seg": {"net_flip_gain": 37},
            "rate": {"candidate_archive_bytes": 186_301},
        },
    )
    assert evidence["justified"] is False


def test_variant_axis_carries_scope_and_nonpromotion_boundary() -> None:
    axis = mc36._variant_axis("successor_drop532", 7)
    assert "7 changed pairs over n600" in axis
    assert "successor_drop532" in axis
    assert axis.endswith("NON-PROMOTABLE")


def test_mc36_runner_does_not_measure_and_discard_payloads() -> None:
    assert check_no_measure_and_discard_payload(
        repo_root=mc36.REPO,
        roots=[Path("experiments/ddm_mc35_micro35_union_build.py")],
    ) == []


def test_storage_route_resume_reuses_immutable_receipt(tmp_path: Path) -> None:
    output = tmp_path / "logical"
    bulk = tmp_path / "bulk"
    first = mc36.route_variant_bulk(
        output=output,
        bulk_root=bulk,
        variant="test_variant",
        expected_bulk_bytes=0,
    )
    second = mc36.route_variant_bulk(
        output=output,
        bulk_root=bulk,
        variant="test_variant",
        expected_bulk_bytes=0,
    )
    assert second == first
    assert (output / "retained").resolve() == (bulk / "retained").resolve()


def test_storage_route_resume_rejects_variant_drift(tmp_path: Path) -> None:
    output = tmp_path / "logical"
    bulk = tmp_path / "bulk"
    mc36.route_variant_bulk(
        output=output,
        bulk_root=bulk,
        variant="first",
        expected_bulk_bytes=0,
    )
    with pytest.raises(mc36.MC35Error, match="resumed storage route differs"):
        mc36.route_variant_bulk(
            output=output,
            bulk_root=bulk,
            variant="second",
            expected_bulk_bytes=0,
        )


def test_same_device_compile_workspace_keeps_original_route(tmp_path: Path) -> None:
    output = tmp_path / "logical"
    bulk = tmp_path / "bulk"
    mc36.route_variant_bulk(
        output=output,
        bulk_root=bulk,
        variant="same_device",
        expected_bulk_bytes=0,
    )
    source = tmp_path / "source.npy"
    source.write_bytes(b"payload")
    result = mc36.migrate_compile_workspace_to_hardlink_device(
        output=output, variant="same_device", source_payload=source
    )
    assert result["migration_required"] is False
    assert (output / "compile_workspace").is_symlink()


def test_storage_route_accepts_preserved_cross_device_addendum(tmp_path: Path) -> None:
    output = tmp_path / "logical"
    bulk = tmp_path / "bulk"
    first = mc36.route_variant_bulk(
        output=output,
        bulk_root=bulk,
        variant="migrated",
        expected_bulk_bytes=0,
    )
    workspace = output / "compile_workspace"
    attempt = output / "compile_workspace_cross_device_attempt"
    workspace.replace(attempt)
    workspace.mkdir()
    mc36.qs1.retain_json(
        output / "STORAGE_ROUTING_ADDENDUM.json",
        {
            "schema": "ddm_mc36_compile_workspace_same_device_route.v1",
            "variant": "migrated",
        },
    )
    second = mc36.route_variant_bulk(
        output=output,
        bulk_root=bulk,
        variant="migrated",
        expected_bulk_bytes=0,
    )
    assert second == first
    assert attempt.resolve() == (bulk / "compile_workspace").resolve()


def test_variant_compensation_receipt_preserves_legacy_solver_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    legacy = {
        "schema": "legacy",
        "rows": [{"pair": 105, "solve": {"codes": [1, 2]}}],
    }
    mc36.qs1.retain_json(tmp_path / "FRESH_COMPENSATION.json", legacy)
    monkeypatch.setattr(mc36, "checkpoint", lambda *_args, **_kwargs: {})
    rows = [
        {
            "pair": 105,
            "solve": {"codes": [1, 2]},
            "variant_final_pose_vector": {"path": "retained.npy"},
        }
    ]
    result = mc36._retain_compensation_result(
        output=tmp_path,
        variant="successor_drop532",
        rows=rows,
        reuse_proofs=[],
    )
    assert json.loads((tmp_path / "FRESH_COMPENSATION.json").read_text()) == legacy
    assert json.loads((tmp_path / "VARIANT_COMPENSATION.json").read_text()) == result


def test_variant_compensation_receipt_rejects_legacy_row_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    mc36.qs1.retain_json(
        tmp_path / "FRESH_COMPENSATION.json",
        {"schema": "legacy", "rows": [{"pair": 105, "solve": {"codes": [1]}}]},
    )
    monkeypatch.setattr(mc36, "checkpoint", lambda *_args, **_kwargs: {})
    with pytest.raises(mc36.MC35Error, match="legacy fresh-compensation rows differ"):
        mc36._retain_compensation_result(
            output=tmp_path,
            variant="successor_drop532",
            rows=[{"pair": 105, "solve": {"codes": [2]}}],
            reuse_proofs=[],
        )
