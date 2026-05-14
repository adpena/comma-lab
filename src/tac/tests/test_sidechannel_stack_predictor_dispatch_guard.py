# SPDX-License-Identifier: MIT
"""Dispatch-readiness guards for tools/sidechannel_stack_predictor.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PREDICTOR = REPO_ROOT / "tools" / "sidechannel_stack_predictor.py"


def test_sidechannel_predictor_json_is_forensic_not_dispatch_ready() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(PREDICTOR),
            "--bits",
            "4",
            "--sidechannels",
            "latent",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)

    assert payload["evidence_semantics"] == "prediction_only_forensic"
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["beats_pr106"] is False
    assert payload["predicted_beats_pr106"] is True
    assert "prediction_only_stack_estimate" in payload["dispatch_blockers"]


def test_sidechannel_predictor_human_output_emits_no_dispatch_one_liner() -> None:
    proc = subprocess.run(
        [sys.executable, str(PREDICTOR), "--bits", "5"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert "ready_for_exact_eval_dispatch=false" in proc.stdout
    assert "Operator one-liner" not in proc.stdout
    assert "launch_lane_on_vastai.py full" not in proc.stdout
