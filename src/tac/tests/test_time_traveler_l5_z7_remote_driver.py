# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
REMOTE_DRIVER = (
    REPO
    / "scripts"
    / "remote_lane_substrate_time_traveler_l5_z7_lstm_predictive_coding.sh"
)


def _write_claim_helper(workspace: Path) -> None:
    (workspace / "tools").mkdir(parents=True, exist_ok=True)
    (workspace / "tools" / "claim_lane_dispatch.py").write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys

cmd = sys.argv[1] if len(sys.argv) > 1 else ""
if cmd == "summary":
    print(json.dumps({
        "active": [{
            "lane_id": os.environ["Z7_GRU_LANE_ID"],
            "instance_job_id": os.environ["Z7_GRU_DISPATCH_INSTANCE_JOB_ID"],
        }]
    }))
elif cmd == "claim":
    claim_args_path = os.environ.get("CLAIM_ARGS_PATH")
    if claim_args_path:
        with open(claim_args_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(sys.argv[1:]) + "\\n")
    print("terminal-claim-ok")
else:
    raise SystemExit(f"unexpected claim helper command: {cmd}")
""",
        encoding="utf-8",
    )


def _write_bootstrap(workspace: Path) -> None:
    (workspace / "scripts").mkdir(parents=True, exist_ok=True)
    (workspace / "scripts" / "remote_archive_only_eval.sh").write_text(
        "bootstrap_runtime_deps() { return 0; }\n",
        encoding="utf-8",
    )


def test_z7_remote_driver_bash_syntax_clean() -> None:
    result = subprocess.run(
        ["bash", "-n", str(REMOTE_DRIVER)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_z7_remote_driver_threads_timing_smoke_defaults_and_terminal_claim(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "experiments").mkdir(parents=True)
    _write_claim_helper(workspace)
    _write_bootstrap(workspace)

    trainer = (
        workspace
        / "experiments"
        / "train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py"
    )
    trainer.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

argv = sys.argv[1:]
out = Path(argv[argv.index("--output-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "argv.json").write_text(json.dumps(argv), encoding="utf-8")
(out / "z7_gru_prebuild_full_main_export_stats.json").write_text(
    json.dumps({
        "evidence_grade": "z7_remote_timing_smoke_no_score_claim",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_paid_dispatch": False,
    }),
    encoding="utf-8",
)
""",
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    log_dir = tmp_path / "logs"
    claim_args_path = tmp_path / "claim_args.jsonl"
    lane_id = "lane_per_substrate_symposium_z7_lstm_predictive_coding_20260517"
    env = os.environ.copy()
    env.update(
        {
            "WORKSPACE": str(workspace),
            "PYBIN": sys.executable,
            "CLAIM_ARGS_PATH": str(claim_args_path),
            "LOG_DIR": str(log_dir),
            "OUTPUT_DIR": str(output_dir),
            "Z7_GRU_OUTPUT_DIR": str(output_dir),
            "Z7_GRU_DISPATCH_INSTANCE_JOB_ID": "z7-gru-test-job",
            "Z7_GRU_LANE_ID": lane_id,
            "Z7_GRU_TRAINER_MODE": "timing_smoke",
        }
    )

    result = subprocess.run(
        ["bash", str(REMOTE_DRIVER)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, (
        "Z7 remote lane script failed\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    argv = json.loads((output_dir / "argv.json").read_text(encoding="utf-8"))
    assert "--smoke" not in argv
    expected_pairs = {
        "--epochs": "1",
        "--batch-size": "1",
        "--lr": "1e-3",
        "--latent-dim": "6",
        "--ego-motion-dim": "3",
        "--gru-hidden-dim": "8",
        "--gru-num-layers": "1",
        "--max-pairs": "1",
        "--decoder-embed-dim": "4",
        "--decoder-channels": "4,4",
        "--decoder-num-upsample-blocks": "2",
        "--decoder-initial-grid-h": "2",
        "--decoder-initial-grid-w": "2",
        "--output-height": "16",
        "--output-width": "16",
        "--loss-mode": "score_aware",
    }
    for flag, value in expected_pairs.items():
        assert flag in argv
        assert argv[argv.index(flag) + 1] == value
    assert "--inflate-verify" in argv
    assert "--emit-static-control" in argv

    provenance = json.loads((log_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["lane_id"] == lane_id
    assert provenance["mode"] == "timing_smoke"
    assert provenance["loss_mode"] == "score_aware"
    assert provenance["max_pairs"] == "1"
    assert provenance["emit_static_control"] == "true"

    claim_rows = [
        json.loads(line)
        for line in claim_args_path.read_text(encoding="utf-8").splitlines()
    ]
    assert claim_rows
    terminal_args = claim_rows[-1]
    assert terminal_args[terminal_args.index("--status") + 1] == (
        "completed_z7_gru_remote_driver_no_score_claim"
    )
    notes = terminal_args[terminal_args.index("--notes") + 1]
    assert "evidence_marker=[z7_remote_timing_smoke_no_score_claim]" in notes
    assert "score_claim=false" in notes
    assert "mode=timing_smoke" in notes


def test_z7_remote_driver_refuses_missing_stats_completion_marker(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "experiments").mkdir(parents=True)
    _write_claim_helper(workspace)
    _write_bootstrap(workspace)

    trainer = (
        workspace
        / "experiments"
        / "train_substrate_time_traveler_l5_z7_lstm_predictive_coding.py"
    )
    trainer.write_text(
        """#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

argv = sys.argv[1:]
out = Path(argv[argv.index("--output-dir") + 1])
out.mkdir(parents=True, exist_ok=True)
(out / "argv.txt").write_text(" ".join(argv), encoding="utf-8")
raise SystemExit(0)
""",
        encoding="utf-8",
    )

    output_dir = tmp_path / "output"
    log_dir = tmp_path / "logs"
    claim_args_path = tmp_path / "claim_args_missing_stats.jsonl"
    env = os.environ.copy()
    env.update(
        {
            "WORKSPACE": str(workspace),
            "PYBIN": sys.executable,
            "CLAIM_ARGS_PATH": str(claim_args_path),
            "LOG_DIR": str(log_dir),
            "OUTPUT_DIR": str(output_dir),
            "Z7_GRU_OUTPUT_DIR": str(output_dir),
            "Z7_GRU_DISPATCH_INSTANCE_JOB_ID": "z7-gru-missing-stats-job",
            "Z7_GRU_LANE_ID": "lane_z7_gru_missing_stats_test",
            "Z7_GRU_TRAINER_MODE": "timing_smoke",
        }
    )

    result = subprocess.run(
        ["bash", str(REMOTE_DRIVER)],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 31, (
        "expected missing stats failure\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    claim_rows = [
        json.loads(line)
        for line in claim_args_path.read_text(encoding="utf-8").splitlines()
    ]
    terminal_args = claim_rows[-1]
    assert terminal_args[terminal_args.index("--status") + 1] == (
        "failed_z7_gru_remote_driver_rc_31"
    )
