# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

from tools.audit_preserved_orphans import (
    PreservedOrphanContract,
    audit_preserved_orphans,
)


def test_preserved_orphan_audit_passes_duplicate_and_reviewed_supersession(tmp_path: Path) -> None:
    root = tmp_path / ".omx/state/orphans_preserved"
    root.mkdir(parents=True)
    canonical_duplicate = tmp_path / "src/tac/tests/test_example.py"
    canonical_duplicate.parent.mkdir(parents=True)
    canonical_duplicate.write_text("same\n", encoding="utf-8")
    (root / "test_example.py.orphan").write_text("same\n", encoding="utf-8")

    canonical_superseded = tmp_path / "scripts/remote_lane_example.sh"
    canonical_superseded.parent.mkdir(parents=True)
    canonical_superseded.write_text("hardened\n", encoding="utf-8")
    (root / "remote_lane_example.sh.orphan").write_text("old\n", encoding="utf-8")

    report = audit_preserved_orphans(
        tmp_path,
        contracts=(
            PreservedOrphanContract(
                orphan_name="test_example.py.orphan",
                canonical_path="src/tac/tests/test_example.py",
            ),
            PreservedOrphanContract(
                orphan_name="remote_lane_example.sh.orphan",
                canonical_path="scripts/remote_lane_example.sh",
                reviewed_status="superseded_by_hardened_canonical_script",
            ),
        ),
    )

    payload = report.to_dict()
    assert payload["ready_for_preserved_orphan_cleanup"] is True
    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["summary"]["duplicate_count"] == 1
    assert payload["summary"]["superseded_count"] == 1


def test_preserved_orphan_audit_blocks_unknown_orphan(tmp_path: Path) -> None:
    root = tmp_path / ".omx/state/orphans_preserved"
    root.mkdir(parents=True)
    (root / "unknown.py.orphan").write_text("x\n", encoding="utf-8")

    report = audit_preserved_orphans(tmp_path, contracts=())
    payload = report.to_dict()

    assert payload["ready_for_preserved_orphan_cleanup"] is False
    assert "unknown preserved orphan" in "\n".join(payload["blockers"])


def test_preserved_orphan_audit_blocks_unreviewed_drift(tmp_path: Path) -> None:
    root = tmp_path / ".omx/state/orphans_preserved"
    root.mkdir(parents=True)
    canonical = tmp_path / "scripts/live.sh"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("live\n", encoding="utf-8")
    (root / "live.sh.orphan").write_text("old\n", encoding="utf-8")

    report = audit_preserved_orphans(
        tmp_path,
        contracts=(
            PreservedOrphanContract(
                orphan_name="live.sh.orphan",
                canonical_path="scripts/live.sh",
            ),
        ),
    )
    payload = report.to_dict()

    assert payload["ready_for_preserved_orphan_cleanup"] is False
    assert "without reviewed supersession" in "\n".join(payload["blockers"])


def test_preserved_orphan_audit_passes_absent_root(tmp_path: Path) -> None:
    report = audit_preserved_orphans(tmp_path)
    payload = report.to_dict()

    assert payload["ready_for_preserved_orphan_cleanup"] is True
    assert payload["summary"]["present_count"] == 0
