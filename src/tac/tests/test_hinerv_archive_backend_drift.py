# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tac.analysis.hinerv_archive_backend_drift import (
    HINERV_ARCHIVE_BACKEND_DRIFT_SCHEMA,
    HinervArchiveBackendDriftError,
    build_hinerv_archive_backend_drift_report,
)
from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (
    CONTEST_BYTE_PRICE_SCORE,
)
from tools.build_hinerv_archive_backend_drift import main as tool_main


def test_hinerv_backend_drift_admits_mlx_for_local_velocity_only() -> None:
    reference = _replay_report(
        label="fallback",
        rows={"tiny": 100_000, "small": 200_000},
        backend="pytorch_portable_fallback",
    )
    candidate = _replay_report(
        label="mlx",
        rows={"tiny": 99_980, "small": 200_060},
        backend="mlx",
    )

    report = build_hinerv_archive_backend_drift_report(
        reference,
        candidate,
        reference_label="pytorch_portable_fallback",
        candidate_label="mlx_metal",
        max_abs_byte_delta=128,
    )

    assert report["schema"] == HINERV_ARCHIVE_BACKEND_DRIFT_SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["local_dev_velocity_ready"] is True
    assert report["ready_backend_for_local_iteration"] == "mlx_metal"
    assert report["max_abs_byte_delta_observed"] == 60
    assert report["sum_byte_delta_candidate_minus_reference"] == 40
    assert math.isclose(
        report["sum_rate_score_delta_candidate_minus_reference"],
        40 * CONTEST_BYTE_PRICE_SCORE,
    )
    assert report["candidate_archive_export_backend_counts"] == {"mlx": 2}
    assert "contest_cpu_cuda_exact_eval_not_executed" in report["blockers"]
    assert "hinerv_archive_backend_drift_local_dev_velocity_only" in report["blockers"]


def test_hinerv_backend_drift_blocks_missing_rows_and_excess_drift() -> None:
    reference = _replay_report(
        label="fallback",
        rows={"tiny": 100_000, "small": 200_000},
        backend="pytorch_portable_fallback",
    )
    candidate = _replay_report(
        label="mlx",
        rows={"tiny": 103_000},
        backend="mlx",
    )

    report = build_hinerv_archive_backend_drift_report(
        reference,
        candidate,
        max_abs_byte_delta=1024,
    )

    assert report["local_dev_velocity_ready"] is False
    assert report["max_abs_byte_delta_observed"] == 3000
    blockers = set(report["blockers"])
    assert "hinerv_archive_backend_drift_candidate_row_missing" in blockers
    assert "hinerv_archive_backend_drift_row_exceeds_byte_tolerance" in blockers
    assert "hinerv_archive_backend_drift_local_dev_velocity_blocked" in blockers


def test_hinerv_backend_drift_rejects_wrong_schema() -> None:
    with pytest.raises(HinervArchiveBackendDriftError, match="reference must be"):
        build_hinerv_archive_backend_drift_report(
            {"schema": "wrong"},
            _replay_report(label="mlx", rows={"tiny": 1}, backend="mlx"),
        )


def test_hinerv_backend_drift_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    reference_path = tmp_path / "fallback.json"
    candidate_path = tmp_path / "mlx.json"
    output_json = tmp_path / "drift.json"
    output_md = tmp_path / "drift.md"
    reference_path.write_text(
        json.dumps(
            _replay_report(
                label="fallback",
                rows={"tiny": 100_000},
                backend="pytorch_portable_fallback",
            )
        ),
        encoding="utf-8",
    )
    candidate_path.write_text(
        json.dumps(
            _replay_report(label="mlx", rows={"tiny": 100_001}, backend="mlx")
        ),
        encoding="utf-8",
    )

    rc = tool_main(
        [
            "--reference-json",
            str(reference_path),
            "--candidate-json",
            str(candidate_path),
            "--reference-label",
            "fallback",
            "--candidate-label",
            "mlx_metal",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema"] == HINERV_ARCHIVE_BACKEND_DRIFT_SCHEMA
    assert payload["reference_json_sha256"]
    assert payload["candidate_json_sha256"]
    assert payload["local_dev_velocity_ready"] is True
    assert "HiNeRV archive backend drift" in output_md.read_text(encoding="utf-8")


def _replay_report(*, label: str, rows: dict[str, int], backend: str) -> dict:
    return {
        "schema": "hinerv_archive_ladder_replay_actuator.v1",
        "authority": "false_authority_replay_actuator_no_scorer_claim",
        "report_path": f"/Users/adpena/Projects/pact/.omx/research/{label}.json",
        "execution_requested": label != "mlx_loaded",
        "load_existing_requested": label == "mlx_loaded",
        "rows": [
            {
                "row_id": row_id,
                "status": "executed_report_loaded_false_authority",
                "archive_bytes": archive_bytes,
                "archive_sha256": f"{index:064x}",
                "archive_path": (
                    f"/Volumes/VertigoDataTier/pact/{label}/{row_id}/archive.zip"
                ),
                "receiver_proof_ready": True,
                "archive_export_backend_counts": {backend: 1},
                "blockers": ["hinerv_archive_size_row_has_no_nonrate_score"],
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            }
            for index, (row_id, archive_bytes) in enumerate(rows.items(), start=1)
        ],
        "archive_bytes_by_row_id": dict(rows),
        "blockers": [
            "contest_cpu_cuda_exact_eval_not_executed",
            "hinerv_archive_ladder_replay_false_authority_no_nonrate_score",
        ],
        "score_claim": False,
        "score_claim_valid": False,
        "frontier_score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "production_hardened_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }
