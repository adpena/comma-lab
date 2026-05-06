from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tools.audit_tooling_consolidation import audit_tooling

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "tools" / "audit_tooling_consolidation.py"


def test_audit_tooling_detects_duplicate_helper_patterns(tmp_path: Path) -> None:
    sample_dir = tmp_path / "tools"
    sample_dir.mkdir()
    sample = sample_dir / "sample.py"
    sample.write_text(
        "\n".join(
            [
                "import json",
                "from pathlib import Path",
                "REPO = Path(__file__).resolve().parents[1]",
                "def _sha256_file(path):",
                "    return 'x'",
                "print(json.dumps({'score_claim': False}, indent=2, sort_keys=True))",
            ]
        ),
        encoding="utf-8",
    )

    report = audit_tooling(tmp_path, ("tools",))
    payload = report.to_dict()

    assert payload["ready_for_incremental_consolidation"] is True
    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    counts = payload["summary"]["pattern_counts"]
    assert counts["local_sha256_helper"] == 1
    assert counts["local_json_dump"] == 1
    assert counts["manual_repo_root_parents"] == 1
    assert counts["manual_audit_score_dispatch_metadata"] == 1
    assert payload["summary"]["affected_file_count"] == 1
    assert payload["per_file_counts"]["tools/sample.py"]["local_json_dump"] == 1


def test_audit_tooling_cli_json_contract() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--format",
            "json",
            "--scan-root",
            "tools/audit_hnerv_frontier_scorecard.py",
        ],
        cwd=REPO,
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["audit"] == "tooling_consolidation_inventory"
    assert payload["ready_for_incremental_consolidation"] is True
    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
