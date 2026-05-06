from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def test_wr01_exact_eval_packet_reports_missing_env_without_dispatch(tmp_path: Path) -> None:
    env = os.environ.copy()
    for key in (
        "LIGHTNING_SSH_TARGET",
        "LIGHTNING_REMOTE_PACT",
        "LIGHTNING_UPSTREAM_DIR",
        "LIGHTNING_TEAMSPACE",
        "LIGHTNING_STUDIO",
        "LIGHTNING_SDK_USER",
    ):
        env.pop(key, None)
    out = tmp_path / "packet.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_wr01_exact_eval_packet.py"),
            "--json-out",
            str(out),
        ],
        check=True,
        text=True,
        env=env,
    )

    payload = json.loads(out.read_text())
    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["ready_for_submit"] is False
    assert "missing_lightning_environment" in payload["blockers"]
    assert payload["preflight_ready"] is True
    assert payload["compliance_ok"] is True
    assert payload["payload_diff_ready"] is True
    assert "--stage-workspace" in payload["commands"]["submit"]
    assert "'$LIGHTNING_SSH_TARGET'" not in payload["commands"]["submit"]
    assert "--remote $LIGHTNING_SSH_TARGET" in payload["commands"]["submit"]
    assert "tools/claim_lane_dispatch.py" in payload["commands"]["claim"]
