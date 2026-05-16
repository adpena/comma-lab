# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path


def test_tt5l_timing_smoke_tool_builds_valid_artifact(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    tool = repo_root / "tools" / "build_tt5l_first_anchor_timing_smoke_artifact.py"
    result_relpath = (
        "experiments/results/time_traveler_l5_v2/tt5l_timing_smoke_result.json"
    )
    result_path = tmp_path / result_relpath
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps({"schema": "tt5l_timing_result_v1", "score_claim": False}) + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--repo-root",
            str(tmp_path),
            "--result-artifact",
            result_relpath,
            "--provider",
            "modal",
            "--hardware",
            "A100",
            "--provider-call-id",
            "fc-test",
            "--elapsed-seconds",
            "12.5",
            "--seconds-per-epoch",
            "1.25",
            "--command-argv-json",
            json.dumps(["python", "experiments/train_substrate_time_traveler_l5.py"]),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    artifact_path = tmp_path / ".omx/state/tt5l_first_anchor_timing_smoke.json"
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["schema"] == "tt5l_first_anchor_timing_smoke_v1"
    assert payload["predicate_id"] == "tt5l_first_anchor_timing_smoke_rate_v1"
    assert payload["required_axes"] == ["contest_cpu", "contest_cuda"]
    assert payload["provider_call_id"] == "fc-test"
    assert payload["result_artifact_path"] == result_relpath
    assert payload["result_artifact_sha256"] == hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest()
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    stdout = json.loads(proc.stdout)
    assert stdout["status"]["artifact_valid"] is True
