# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tools import smoke_staircase_ssh_input_custody as smoke


def test_ssh_input_custody_smoke_exercises_directory_push_and_pullback(
    tmp_path: Path,
) -> None:
    args = smoke.parse_args(
        [
            "--run-dir",
            str(tmp_path / "smoke"),
            "--queue-id",
            "ssh_input_custody_smoke_fixture",
        ]
    )

    report = smoke.run_smoke(args)

    assert report["schema"] == smoke.SMOKE_SCHEMA
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["rank_or_kill_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert "local_fake_transport_no_network" in report["dispatch_blockers"]
    assert report["observations"]["success_count"] == 1
    assert report["observations"]["failure_count"] == 0
    assert report["observations"]["directory_push_used_delete"] is True
    assert report["observations"]["output_artifact_exists"] is True
    assert report["observations"]["output_false_authority"] is True
    manifest = report["input_manifest"]
    assert manifest["local_manifest"]["is_dir"] is True
    assert manifest["local_manifest"]["recursive_entry_count"] == 3
    assert len(manifest["local_manifest"]["recursive_sha256"]) == 64
    assert len(manifest["local_manifest_sha256"]) == 64
    output_path = tmp_path / "smoke" / "outputs" / "remote_result.json"
    output_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert output_payload["schema"] == smoke.OUTPUT_SCHEMA
    assert output_payload["dispatch_attempted"] is False


def test_ssh_input_custody_smoke_rejects_nonpositive_max_steps(tmp_path: Path) -> None:
    args = smoke.parse_args(
        [
            "--run-dir",
            str(tmp_path / "smoke"),
            "--max-steps",
            "0",
        ]
    )

    try:
        smoke.run_smoke(args)
    except SystemExit as exc:
        assert "--max-steps must be >= 1" in str(exc)
    else:
        raise AssertionError("smoke accepted --max-steps=0")
