# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tac.analysis.hinerv_archive_ladder_replay_actuator import (
    HINERV_ARCHIVE_LADDER_REPLAY_ACTUATOR_SCHEMA,
    HinervArchiveLadderReplayActuatorError,
    build_hinerv_archive_ladder_replay_actuator_report,
)
from tools.run_hinerv_archive_ladder_replay_actuator import main as tool_main


def test_hinerv_replay_actuator_plans_rows_without_execution() -> None:
    report = build_hinerv_archive_ladder_replay_actuator_report(
        _waterfill_report(Path("/tmp/unused")),
        row_ids=["hi_nerv_local_tiny"],
    )

    assert report["schema"] == HINERV_ARCHIVE_LADDER_REPLAY_ACTUATOR_SCHEMA
    assert report["execution_requested"] is False
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["row_count"] == 1
    row = report["rows"][0]
    assert row["row_id"] == "hi_nerv_local_tiny"
    assert row["status"] == "planned"
    assert row["execute_ready"] is True
    assert row["executed"] is False
    assert "contest_cpu_cuda_exact_eval_not_executed" in report["blockers"]


def test_hinerv_replay_actuator_executes_and_loads_replay_report(
    tmp_path: Path,
) -> None:
    output_json = tmp_path / "replay.json"
    waterfill = _waterfill_report(output_json)

    def fake_runner(argv: list[str], cwd: Path, timeout_seconds: int | None) -> dict:
        assert timeout_seconds == 123
        path = _flag_path(argv, "--output-json", cwd)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "schema": "hinerv_archive_size_ladder.v1",
                    "archive_export_backend_counts": {"mlx": 1},
                    "archive_rows": [
                        {
                            "row_id": "hi_nerv_local_tiny",
                            "archive_bytes": 123456,
                            "archive_sha256": "a" * 64,
                            "archive_path": "/Volumes/VertigoDataTier/pact/tiny/archive.zip",
                            "submission_dir": "/Volumes/VertigoDataTier/pact/tiny/submission",
                            "spine_manifest_path": "/Volumes/VertigoDataTier/pact/tiny/spine.json",
                            "receiver_proof_path": "/Volumes/VertigoDataTier/pact/tiny/proof.json",
                            "decoder_weight_waterfill_plan_path": "/Volumes/VertigoDataTier/pact/tiny/plan.json",
                            "runtime_consumption_proof_ready": True,
                            "blockers": [
                                "hinerv_archive_size_row_has_no_nonrate_score",
                            ],
                        }
                    ],
                    "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
                }
            ),
            encoding="utf-8",
        )
        return {"returncode": 0, "stdout_tail": "ok", "stderr_tail": ""}

    report = build_hinerv_archive_ladder_replay_actuator_report(
        waterfill,
        execute=True,
        cwd=tmp_path,
        timeout_seconds=123,
        replay_output_root="/Volumes/VertigoDataTier/pact/fresh_hinerv_replay",
        artifact_tag="unit_tag",
        runner=fake_runner,
    )

    assert report["executed_row_count"] == 1
    assert report["loaded_replay_report_count"] == 1
    assert report["receiver_proof_ready_row_count"] == 1
    row = report["rows"][0]
    assert row["status"] == "executed_report_loaded_false_authority"
    assert row["archive_bytes"] == 123456
    assert row["submission_dir"] == "/Volumes/VertigoDataTier/pact/tiny/submission"
    assert row["spine_manifest_path"].endswith("/spine.json")
    assert row["receiver_proof_path"].endswith("/proof.json")
    assert row["decoder_weight_waterfill_plan_path"].endswith("/plan.json")
    assert row["receiver_proof_ready"] is True
    assert row["archive_export_backend_counts"] == {"mlx": 1}
    assert row["replay_report_sha256"]
    assert row["command_argv"][row["command_argv"].index("--output-dir") + 1] == (
        "/Volumes/VertigoDataTier/pact/fresh_hinerv_replay/hi_nerv_local_tiny"
    )
    assert "unit_tag_hi_nerv_local_tiny" in row["output_json"]
    assert "hinerv_archive_size_row_has_no_nonrate_score" in row["blockers"]
    assert "contest_cpu_cuda_exact_eval_not_executed" in report["blockers"]


def test_hinerv_replay_actuator_loads_existing_report_without_execution(
    tmp_path: Path,
) -> None:
    output_json = tmp_path / "existing_replay.json"
    output_json.write_text(
        json.dumps(
            {
                "schema": "hinerv_archive_size_ladder.v1",
                "archive_export_backend_counts": {"pytorch_portable_fallback": 1},
                "archive_rows": [
                    {
                        "row_id": "hi_nerv_local_tiny",
                        "archive_bytes": 654321,
                        "archive_sha256": "c" * 64,
                        "archive_path": "/Volumes/VertigoDataTier/pact/existing/archive.zip",
                        "submission_dir": "/Volumes/VertigoDataTier/pact/existing/submission",
                        "spine_manifest_path": "/Volumes/VertigoDataTier/pact/existing/spine.json",
                        "receiver_proof_path": "/Volumes/VertigoDataTier/pact/existing/proof.json",
                        "decoder_weight_waterfill_plan_path": "/Volumes/VertigoDataTier/pact/existing/plan.json",
                        "runtime_consumption_proof_ready": True,
                        "blockers": ["archive_export_backend_not_mlx"],
                    }
                ],
                "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
            }
        ),
        encoding="utf-8",
    )

    report = build_hinerv_archive_ladder_replay_actuator_report(
        _waterfill_report(output_json),
        load_existing=True,
        execute=False,
        cwd=tmp_path,
    )

    assert report["execution_requested"] is False
    assert report["load_existing_requested"] is True
    assert report["loaded_replay_report_count"] == 1
    assert report["executed_row_count"] == 0
    row = report["rows"][0]
    assert row["status"] == "existing_report_loaded_false_authority"
    assert row["archive_bytes"] == 654321
    assert row["archive_path"].endswith("/existing/archive.zip")
    assert row["submission_dir"].endswith("/existing/submission")
    assert row["spine_manifest_path"].endswith("/existing/spine.json")
    assert row["receiver_proof_path"].endswith("/existing/proof.json")
    assert row["decoder_weight_waterfill_plan_path"].endswith("/existing/plan.json")
    assert row["receiver_proof_ready"] is True
    assert row["archive_export_backend_counts"] == {"pytorch_portable_fallback": 1}
    assert row["executed"] is False
    assert row["score_claim"] is False


def test_hinerv_replay_actuator_fails_closed_on_bad_command() -> None:
    waterfill = _waterfill_report(Path("/tmp/replay.json"))
    waterfill["rows"][0]["archive_ladder_replay_command_argv"] = [
        ".venv/bin/python",
        "tools/not_the_ladder.py",
    ]

    report = build_hinerv_archive_ladder_replay_actuator_report(
        waterfill,
        execute=True,
    )

    row = report["rows"][0]
    assert row["executed"] is False
    assert row["status"] == "blocked_not_executed"
    assert "hinerv_archive_ladder_replay_command_tool_unexpected" in row["blockers"]


def test_hinerv_replay_actuator_rejects_wrong_schema() -> None:
    with pytest.raises(
        HinervArchiveLadderReplayActuatorError,
        match=r"expected hinerv_archive_ladder_waterfill\.v1",
    ):
        build_hinerv_archive_ladder_replay_actuator_report({"schema": "wrong"})


def test_hinerv_replay_actuator_cli_plan_smoke(tmp_path: Path) -> None:
    waterfill_path = tmp_path / "waterfill.json"
    output_json = tmp_path / "actuator.json"
    output_md = tmp_path / "actuator.md"
    waterfill_path.write_text(
        json.dumps(_waterfill_report(tmp_path / "replay.json")),
        encoding="utf-8",
    )

    rc = tool_main(
        [
            "--waterfill-json",
            str(waterfill_path),
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--row-id",
            "hi_nerv_local_tiny",
        ]
    )

    assert rc == 0
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["row_count"] == 1
    assert payload["rows"][0]["status"] == "planned"
    assert "HiNeRV archive ladder replay actuator" in output_md.read_text(
        encoding="utf-8"
    )


def _waterfill_report(output_json: Path) -> dict[str, Any]:
    return {
        "schema": "hinerv_archive_ladder_waterfill.v1",
        "candidate_id": "unit",
        "report_path": "/Users/adpena/Projects/pact/.omx/research/unit_waterfill.json",
        "rows": [
            {
                "row_id": "hi_nerv_local_tiny",
                "archive_ladder_replay_output_dir": (
                    "/Volumes/VertigoDataTier/pact/hinerv_archive_ladder_waterfill_replay/"
                    "hi_nerv_local_tiny"
                ),
                "archive_ladder_replay_command_argv": [
                    ".venv/bin/python",
                    "tools/build_hinerv_archive_size_ladder.py",
                    "--output-dir",
                    "/Volumes/VertigoDataTier/pact/hinerv_archive_ladder_waterfill_replay/hi_nerv_local_tiny",
                    "--output-json",
                    output_json.as_posix(),
                    "--output-md",
                    output_json.with_suffix(".md").as_posix(),
                    "--num-pairs",
                    "600",
                    "--row-id",
                    "hi_nerv_local_tiny",
                    "--decoder-codec",
                    "int8_mixed",
                    "--emit-receiver-proof",
                    "--emit-decoder-weight-waterfill-plan",
                    "--decoder-weight-saliency-json",
                    ".omx/research/unit_saliency.json",
                    "--decoder-weight-waterfill-action-bits",
                    "0,2,4,8,16,32",
                ],
            }
        ],
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _flag_path(argv: list[str], flag: str, cwd: Path) -> Path:
    value = Path(argv[argv.index(flag) + 1])
    if not value.is_absolute():
        value = cwd / value
    return value
