# SPDX-License-Identifier: MIT
"""Typed config, resumability, and false-authority tests for G90."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "materialize_taskspace_projected_population_costates.py"
SPEC = importlib.util.spec_from_file_location("g90_projected_costate_tool", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

H = "a" * 64


def _identity(path: str = "/does/not/exist") -> dict[str, object]:
    return {"path": path, "bytes": 1, "sha256": H}


def _config_body() -> dict[str, object]:
    return {
        "schema": MODULE.CONFIG_SCHEMA,
        "output_root": "/Volumes/VertigoDataTier/pact/g90_test",
        "seed": 1234,
        "num_threads": 4,
        "safety_reserve_bytes": 1,
        "source_video": _identity(),
        "g85": {
            "sample_count": 600,
            "d_pose": 163.06130981,
            "d_seg": 0.0274712,
            "archive": _identity(),
            "raw": _identity(),
        },
        "semantic_archive": _identity(),
        "g46_target_labels": _identity(),
        "segnet_weights": _identity(),
        "posenet_weights": _identity(),
        "upstream_closure_sha256": H,
        "upstream_members": [_identity()],
        "g78_aggregate": _identity(),
        "g78_aggregate_self_sha256": H,
        "g87_aggregate": _identity(),
        "g87_aggregate_self_sha256": H,
    }


def test_config_is_closed_and_binds_full_n600_score_point(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_config_body()), encoding="utf-8")
    config = MODULE.load_config(path)
    assert config.global_mean_pose_dist == 163.06130981
    assert config.global_mean_seg_dist == 0.0274712
    assert config.g85_archive["sha256"] == H


def test_config_refuses_an_invented_admission_threshold(tmp_path: Path) -> None:
    body = _config_body()
    body["admission_threshold"] = 0.0
    path = tmp_path / "config.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(
        MODULE.ProjectedCostateMaterializerError,
        match="schema/key set differs",
    ):
        MODULE.load_config(path)


def test_immutable_checkpoint_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "batch.json"
    MODULE._atomic_write_json(path, {"a": 1})
    MODULE._atomic_write_json(path, {"a": 1})
    with pytest.raises(
        MODULE.ProjectedCostateMaterializerError,
        match="immutable checkpoint differs",
    ):
        MODULE._atomic_write_json(path, {"a": 2})


def test_stage_resume_order_is_fixed_and_never_skips_a_gap(tmp_path: Path) -> None:
    assert MODULE._next_incomplete_stage(tmp_path) == 0
    stage0 = tmp_path / "20_stage_00_0000_0120" / "stage_receipt.json"
    stage0.parent.mkdir(parents=True)
    stage0.write_text("{}", encoding="utf-8")
    assert MODULE._next_incomplete_stage(tmp_path) == 1


def test_incomplete_population_cannot_emit_an_aggregate(tmp_path: Path) -> None:
    body = _config_body()
    body["output_root"] = str(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    # Production config parsing correctly refuses local output.  Constructing
    # the typed value directly isolates the aggregate completeness law.
    production_path = tmp_path / "production.json"
    body["output_root"] = "/Volumes/VertigoDataTier/pact/g90_test"
    production_path.write_text(json.dumps(body), encoding="utf-8")
    config = MODULE.load_config(production_path)
    object.__setattr__(config, "output_root", tmp_path)
    assert MODULE._write_aggregate_if_complete(config) is None


def test_blocker_binds_exact_immutable_resume_frontier(tmp_path: Path) -> None:
    body = _config_body()
    production_path = tmp_path / "production.json"
    production_path.write_text(json.dumps(body), encoding="utf-8")
    config = MODULE.load_config(production_path)
    object.__setattr__(config, "output_root", tmp_path)

    batch_dir = tmp_path / "20_stage_00_0000_0120" / "batches"
    for pair_start in range(0, 96, 16):
        pair_stop = pair_start + 16
        checkpoint = MODULE._seal(
            {
                "schema": MODULE.BATCH_SCHEMA,
                "pair_range": [pair_start, pair_stop],
            },
            field="batch_checkpoint_sha256",
        )
        MODULE._atomic_write_json(
            batch_dir / f"batch_{pair_start:04d}_{pair_stop:04d}.json",
            checkpoint,
        )

    exc = MODULE.ProjectedPopulationCostateError(
        "current-base SegNet argmax differs from fresh G78 described cells",
        context={
            "failing_pair_range": [96, 112],
            "mismatch_cell_count": 3,
            "actual_cells_sha256": "b" * 64,
            "expected_cells_sha256": "c" * 64,
        },
    )
    blocker_path = MODULE._write_blocker(config, exc)
    blocker = json.loads(blocker_path.read_bytes())
    frontier = blocker["immutable_resume_frontier"]
    assert frontier["sealed_batch_count"] == 6
    assert frontier["next_pair_range"] == [96, 112]
    assert [row["pair_range"] for row in frontier["sealed_batches"]] == [
        [start, start + 16] for start in range(0, 96, 16)
    ]
    assert blocker["exception_context"]["mismatch_cell_count"] == 3
    retry = blocker["deterministic_retry_classification"]
    assert retry["class"] == "CONDITIONAL_SINGLE_RETRY_AFTER_ISOLATED_EXACT_RECHECK"
    assert retry["automatic_retry_allowed"] is False
