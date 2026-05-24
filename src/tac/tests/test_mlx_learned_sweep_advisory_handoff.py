# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.mlx_dynamic_learned_sweep import FALSE_AUTHORITY
from tac.optimization.mlx_dynamic_learned_sweep_local_actuator import (
    MACOS_CPU_ADVISORY_EVIDENCE_SEMANTICS,
    MACOS_CPU_ADVISORY_PAYLOAD_SCORE_AXIS,
)
from tac.optimization.mlx_effective_spend_triage_selection import (
    ROW_SCHEMA as SELECTION_ROW_SCHEMA,
)
from tac.optimization.mlx_effective_spend_triage_selection import (
    SCHEMA as SELECTION_SCHEMA,
)
from tac.optimization.mlx_learned_sweep_advisory_handoff import (
    MLXLearnedSweepAdvisoryHandoffError,
    stamp_macos_cpu_advisory_paths,
)
from tac.repo_io import read_json, write_json

REPO_ROOT = Path(__file__).resolve().parents[3]


def _sha(seed: str) -> str:
    return (seed * 64)[:64]


def _selection() -> dict:
    return {
        "schema": SELECTION_SCHEMA,
        **FALSE_AUTHORITY,
        "selected_rows": [
            {
                "schema": SELECTION_ROW_SCHEMA,
                **FALSE_AUTHORITY,
                "candidate_id": "candidate-a",
                "row_id": "row-a",
                "queue_candidate_id": "queue-a",
            }
        ],
    }


def _advisory_payload(seed: str, *, score_axis: str | None = None) -> dict:
    return {
        **FALSE_AUTHORITY,
        "score_claim_eligible": False,
        "score_axis": score_axis or MACOS_CPU_ADVISORY_PAYLOAD_SCORE_AXIS,
        "evidence_semantics": MACOS_CPU_ADVISORY_EVIDENCE_SEMANTICS,
        "archive_sha256": _sha(seed),
        "archive_size_bytes": 123456,
        "score_seg_contribution": 0.01,
        "score_pose_contribution": 0.02,
        "score_rate_contribution": 0.03,
        "provenance": {
            "archive_sha256": _sha(seed),
            "inflate_runtime_manifest": {"runtime_tree_sha256": _sha("b")},
            "inflated_output_manifest": {
                "payload": {"aggregate_sha256": _sha("c")}
            },
        },
    }


def _write_advisory_pair(tmp_path: Path) -> dict:
    write_json(tmp_path / "candidate.json", _advisory_payload("a"))
    write_json(tmp_path / "baseline.json", _advisory_payload("d"))
    return {
        "schema": "mlx_learned_sweep_macos_cpu_advisory_path_map.v1",
        "rows": [
            {
                "candidate_id": "candidate-a",
                "candidate_advisory_path": "candidate.json",
                "baseline_advisory_path": "baseline.json",
            }
        ],
    }


def test_stamp_macos_cpu_advisory_paths_validates_and_stamps_selection(
    tmp_path: Path,
) -> None:
    stamped, report = stamp_macos_cpu_advisory_paths(
        _selection(),
        _write_advisory_pair(tmp_path),
        source_artifact_root=tmp_path,
        require_all_selected=True,
    )

    row = stamped["selected_rows"][0]
    assert row["local_cpu_advisory_source_path"] == "candidate.json"
    assert row["window_baseline_local_cpu_advisory_source_path"] == "baseline.json"
    assert row["local_cpu_advisory_source_sha256"]
    assert stamped["macos_cpu_advisory_handoff"]["schema"] == report["schema"]
    assert report["stamped_row_count"] == 1
    assert report["ready_for_macos_cpu_advisory_queue"] is True
    assert report["authority_boundary"]["score_claim"] is False
    assert report["stamped_rows"][0]["candidate_advisory"]["bytes"] > 0


def test_stamp_macos_cpu_advisory_paths_rejects_non_advisory_score_axis(
    tmp_path: Path,
) -> None:
    write_json(tmp_path / "candidate.json", _advisory_payload("a", score_axis="contest"))
    write_json(tmp_path / "baseline.json", _advisory_payload("d"))

    with pytest.raises(MLXLearnedSweepAdvisoryHandoffError, match="score_axis"):
        stamp_macos_cpu_advisory_paths(
            _selection(),
            {
                "rows": [
                    {
                        "candidate_id": "candidate-a",
                        "candidate_advisory_path": "candidate.json",
                        "baseline_advisory_path": "baseline.json",
                    }
                ]
            },
            source_artifact_root=tmp_path,
        )


def test_stamp_macos_cpu_advisory_paths_rejects_missing_runtime_identity(
    tmp_path: Path,
) -> None:
    candidate = _advisory_payload("a")
    del candidate["provenance"]["inflate_runtime_manifest"]
    write_json(tmp_path / "candidate.json", candidate)
    write_json(tmp_path / "baseline.json", _advisory_payload("d"))

    with pytest.raises(MLXLearnedSweepAdvisoryHandoffError, match="runtime"):
        stamp_macos_cpu_advisory_paths(
            _selection(),
            {
                "rows": [
                    {
                        "candidate_id": "candidate-a",
                        "candidate_advisory_path": "candidate.json",
                        "baseline_advisory_path": "baseline.json",
                    }
                ]
            },
            source_artifact_root=tmp_path,
        )


def test_stamp_macos_cpu_advisory_handoff_cli_writes_guarded_artifacts(
    tmp_path: Path,
) -> None:
    selection_path = tmp_path / "selection.json"
    path_map_path = tmp_path / "path_map.json"
    output_selection_path = tmp_path / "stamped_selection.json"
    report_path = tmp_path / "report.json"
    write_json(selection_path, _selection())
    write_json(path_map_path, _write_advisory_pair(tmp_path))

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/stamp_mlx_learned_sweep_advisory_handoff.py"),
            "--selection",
            str(selection_path),
            "--path-map",
            str(path_map_path),
            "--output-selection",
            str(output_selection_path),
            "--report-output",
            str(report_path),
            "--repo-root",
            str(REPO_ROOT),
            "--source-artifact-root",
            str(tmp_path),
            "--require-all-selected",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    assert stdout["stamped_row_count"] == 1
    assert stdout["authority"]["promotion_eligible"] is False
    stamped = read_json(output_selection_path)
    assert stamped["selected_rows"][0]["local_cpu_advisory_source_path"] == (
        "candidate.json"
    )
    assert read_json(report_path)["ready_for_macos_cpu_advisory_queue"] is True
