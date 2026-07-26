from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import measure_v10_free_predictor_floor as scorer
from tools import score_coupled_witness_raw_debt as debt


def _class_row(errors: int = 0) -> dict[str, dict[str, object]]:
    return {
        name: {
            "class_id": index,
            "errors": errors if index == 0 else 0,
            "sites": 2 if index == 0 else 0,
            "d_seg": errors / 2 if index == 0 else None,
        }
        for index, name in enumerate(scorer.CLASS_ORDER)
    }


def test_stage_hash_and_prefix_reject_mutation(tmp_path: Path) -> None:
    stage_dir = tmp_path / "stages"
    stage = debt._stage_payload(
        config_sha256="a" * 64,
        start=0,
        end=1,
        rows=[{"pair_id": 0, "d_seg": 0.0, "d_pose": 0.0, "per_class": _class_row()}],
    )
    debt._write_once(debt._stage_path(stage_dir, 0, 1), stage)
    rows, custody = debt._load_prefix(
        stage_dir=stage_dir,
        pair_count=1,
        stage_pairs=1,
        config_sha256="a" * 64,
    )
    assert rows[0]["pair_id"] == 0
    assert custody[0]["stage_sha256"] == stage["stage_sha256"]
    mutated = json.loads((stage_dir / "pairs-0000-0000.json").read_text())
    mutated["rows"][0]["d_pose"] = 1.0
    (stage_dir / "pairs-0000-0000.json").write_text(json.dumps(mutated), encoding="utf-8")
    with pytest.raises(debt.RawDebtError, match="differs from receipt body"):
        debt._load_prefix(
            stage_dir=stage_dir,
            pair_count=1,
            stage_pairs=1,
            config_sha256="a" * 64,
        )


def test_aggregate_preserves_pair_and_class_debt() -> None:
    rows = [
        {
            "pair_id": 0,
            "d_seg": 0.25,
            "d_pose": 0.01,
            "seg_mismatched_pixels": 1,
            "seg_events": [[0, 1, 0, 1]],
            "per_class": _class_row(1),
            "cache_label_mismatches": 1,
            "cache_pose_max_abs_difference": 0.125,
        },
        {
            "pair_id": 1,
            "d_seg": 0.0,
            "d_pose": 0.03,
            "seg_mismatched_pixels": 0,
            "seg_events": [],
            "per_class": _class_row(0),
            "cache_label_mismatches": 2,
            "cache_pose_max_abs_difference": 0.25,
        },
    ]
    result = debt._aggregate_rows(rows, pair_count=2)
    assert result["mean_d_seg"] == 0.125
    assert result["mean_d_pose"] == 0.02
    assert result["per_class"]["Road"]["errors"] == 1
    assert result["per_class"]["Road"]["sites"] == 4
    assert result["seg_event_count"] == 1
    assert result["cache_label_mismatches"] == 3
    assert result["cache_pose_max_abs_difference"] == 0.25
    assert len(result["pair_rows_sha256"]) == 64


def test_contest_reference_must_bind_exact_raw(tmp_path: Path) -> None:
    path = tmp_path / "reference.json"
    path.write_text(
        json.dumps(
            {
                "score_axis": "contest_cpu",
                "evidence_grade": "[contest-CPU]",
                "archive_size_bytes": 10,
                "avg_segnet_dist": 0.1,
                "avg_posenet_dist": 0.2,
                "canonical_score": 1.0,
                "provenance": {
                    "archive_sha256": "b" * 64,
                    "inflated_output_manifest": {
                        "payload": {"files": [{"sha256": "a" * 64, "bytes": 20, "relative_path": "0.raw"}]}
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    result = debt._contest_reference(path, raw_sha256="a" * 64, raw_bytes=20)
    assert result["score_axis"] == "contest_cpu"
    with pytest.raises(debt.RawDebtError, match="does not bind"):
        debt._contest_reference(path, raw_sha256="c" * 64, raw_bytes=20)


def test_target_reference_requires_exact_zero_distortion_raw(tmp_path: Path) -> None:
    path = tmp_path / "target.json"
    path.write_text(
        json.dumps(
            {
                "schema": "target.v1",
                "axis": "[local CPU advisory]",
                "score_claim": False,
                "source_custody": {"pair_count": 2},
                "candidate": {
                    "inflated_raw_sha256": "a" * 64,
                    "inflated_raw_bytes": 20,
                    "archive_sha256": "b" * 64,
                    "d_seg": 0.0,
                    "d_pose": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    result = debt._target_reference(path, raw_sha256="a" * 64, raw_bytes=20, pair_count=2)
    assert result["d_seg"] == 0.0
    with pytest.raises(debt.RawDebtError, match="does not bind"):
        debt._target_reference(path, raw_sha256="c" * 64, raw_bytes=20, pair_count=2)
