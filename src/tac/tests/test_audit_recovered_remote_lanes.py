from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.audit_recovered_remote_lanes import LANE_CONTRACTS, audit_recovered_remote_lanes

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "audit_recovered_remote_lanes.py"


def _write_contract_script(root: Path, contract_index: int, *, omit_marker: str | None = None) -> None:
    contract = LANE_CONTRACTS[contract_index]
    path = root / contract.path
    path.parent.mkdir(parents=True, exist_ok=True)
    markers = [marker for marker in contract.required_markers if marker != omit_marker]
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + "\n".join(f"# {m}" for m in markers) + "\n", encoding="utf-8")


def test_recovered_remote_lane_audit_passes_synthetic_contracts(tmp_path: Path) -> None:
    for index in range(len(LANE_CONTRACTS)):
        _write_contract_script(tmp_path, index)

    report = audit_recovered_remote_lanes(tmp_path)
    payload = report.to_dict()

    assert payload["ready_for_operator_visibility"] is True
    assert payload["dispatch_attempted"] is False
    assert payload["score_claim"] is False
    assert payload["summary"]["lane_count"] == len(LANE_CONTRACTS)
    assert all(not row["missing_markers"] for row in payload["summary"]["scripts"])


def test_recovered_remote_lane_audit_fails_missing_marker(tmp_path: Path) -> None:
    for index in range(len(LANE_CONTRACTS)):
        _write_contract_script(
            tmp_path,
            index,
            omit_marker="scripts/remote_archive_only_eval.sh" if index == 0 else None,
        )

    report = audit_recovered_remote_lanes(tmp_path)
    payload = report.to_dict()

    assert payload["ready_for_operator_visibility"] is False
    assert "remote_archive_only_eval.sh" in "\n".join(payload["blockers"])


def test_recovered_remote_lane_audit_fails_branch_only_public_clone(tmp_path: Path) -> None:
    for index in range(len(LANE_CONTRACTS)):
        _write_contract_script(tmp_path, index)
    pr79 = tmp_path / LANE_CONTRACTS[1].path
    text = pr79.read_text(encoding="utf-8")
    text = text.replace('# git checkout --detach "$PUBLIC_COMMIT"\n', "")
    text += 'git clone --depth 1 --branch "$PUBLIC_BRANCH" "$PUBLIC_REPO" "$WORK"\n'
    pr79.write_text(text, encoding="utf-8")

    report = audit_recovered_remote_lanes(tmp_path)
    payload = report.to_dict()

    assert payload["ready_for_operator_visibility"] is False
    assert "branch-only" in "\n".join(payload["blockers"])


def test_recovered_remote_lane_audit_cli_json_contract() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--format", "json"],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["audit"] == "recovered_remote_lane_canonicalization"
    assert payload["ready_for_operator_visibility"] is True
    assert payload["dispatch_attempted"] is False
    assert payload["score_claim"] is False
