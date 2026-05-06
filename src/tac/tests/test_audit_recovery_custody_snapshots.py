from __future__ import annotations

from pathlib import Path

from tac.repo_io import write_json
from tools.audit_recovery_custody_snapshots import (
    EXPECTED_BLOCKED_RECOVERY_INPUTS,
    EXPECTED_LIVE_DIFF_PATHS,
    EXPECTED_PYC_AST_OK,
    EXPECTED_PYC_FULL_DECOMPILE,
    EXPECTED_PYC_STUBS,
    EXPECTED_PYC_TOTAL,
    EXPECTED_SIGNAL_DISPOSITIONS,
    RecoveryCustodyConfig,
    audit_recovery_custody_snapshots,
)


def _write_expected_snapshot(root: Path) -> None:
    pyc_root = root / "pyc"
    signal_root = root / "signal"
    pyc_root.mkdir(parents=True)
    signal_root.mkdir(parents=True)
    for name in (
        "RECOVERY_INDEX.md",
        "per_file_table.md",
        "per_file_table.tsv",
        "recover_orphans.py",
    ):
        (pyc_root / name).write_text("custody\n", encoding="utf-8")

    rows = []
    for index in range(EXPECTED_PYC_TOTAL):
        pyc = pyc_root / f"f{index}.pyc"
        pyc.write_bytes(b"pyc")
        rows.append(
            {
                "orphan_rel": f"missing/f{index}.py",
                "pyc_path": str(pyc),
                "stub_written": index < EXPECTED_PYC_STUBS,
                "ast_parse_ok": index < EXPECTED_PYC_AST_OK,
                "decompiled_ok": index < EXPECTED_PYC_FULL_DECOMPILE,
            }
        )
    write_json(pyc_root / "recovery_results.json", rows)

    for name in (
        "counts.txt",
        "deleted_tracked_files.txt",
        "modified_tracked_files.txt",
        "quarantine_audit.md",
        "git_status_short.txt",
        "untracked_files.txt",
        "worktree_list.txt",
    ):
        (signal_root / name).write_text("", encoding="utf-8")

    records = []
    for disposition, count in EXPECTED_SIGNAL_DISPOSITIONS.items():
        for i in range(count):
            relpath = f"{disposition}/{i}"
            if disposition == "blocked_recovery_input_needs_canonicalization_before_promotion":
                relpath = EXPECTED_BLOCKED_RECOVERY_INPUTS[i]
            if disposition == "compare_by_hand_live_diff_before_merge_or_delete":
                relpath = EXPECTED_LIVE_DIFF_PATHS[i]
            records.append({"category": "test", "disposition": disposition, "relpath": relpath})
    write_json(signal_root / "quarantine_audit.json", {"records": records})


def _config() -> RecoveryCustodyConfig:
    return RecoveryCustodyConfig(
        pyc_recovery_root="pyc",
        signal_loss_root="signal",
        resolved_dispositions_manifest=None,
    )


def test_recovery_custody_audit_passes_expected_snapshot(tmp_path: Path) -> None:
    _write_expected_snapshot(tmp_path)

    report = audit_recovery_custody_snapshots(tmp_path, config=_config())
    payload = report.to_dict()

    assert payload["ready_for_recovery_custody_preservation"] is True
    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["summary"]["pyc_recovery"]["recovery_result_count"] == EXPECTED_PYC_TOTAL
    assert payload["summary"]["signal_loss"]["deleted_tracked_count"] == 0


def test_recovery_custody_audit_blocks_missing_pyc_custody(tmp_path: Path) -> None:
    _write_expected_snapshot(tmp_path)
    (tmp_path / "pyc/f0.pyc").unlink()

    report = audit_recovery_custody_snapshots(tmp_path, config=_config())
    payload = report.to_dict()

    assert payload["ready_for_recovery_custody_preservation"] is False
    assert "lost bytecode custody" in "\n".join(payload["blockers"])


def test_recovery_custody_audit_blocks_tracked_source_loss_marker(tmp_path: Path) -> None:
    _write_expected_snapshot(tmp_path)
    (tmp_path / "signal/deleted_tracked_files.txt").write_text("src/tac/lost.py\n", encoding="utf-8")

    report = audit_recovery_custody_snapshots(tmp_path, config=_config())
    payload = report.to_dict()

    assert payload["ready_for_recovery_custody_preservation"] is False
    assert "deleted tracked files" in "\n".join(payload["blockers"])


def test_recovery_custody_audit_blocks_disposition_drift(tmp_path: Path) -> None:
    _write_expected_snapshot(tmp_path)
    payload = {
        "records": [
            {"category": "test", "disposition": "duplicate_same_safe_to_delete_after_manifest_commit", "relpath": "x"}
        ]
    }
    write_json(tmp_path / "signal/quarantine_audit.json", payload)

    report = audit_recovery_custody_snapshots(tmp_path, config=_config())
    payload = report.to_dict()

    assert payload["ready_for_recovery_custody_preservation"] is False
    assert "disposition count drifted" in "\n".join(payload["blockers"])


def test_recovery_custody_audit_subtracts_resolved_dispositions(tmp_path: Path) -> None:
    _write_expected_snapshot(tmp_path)
    write_json(
        tmp_path / "resolved.json",
        {
            "entries": [
                {
                    "historical_disposition": "compare_by_hand_live_diff_before_merge_or_delete",
                    "relpath": EXPECTED_LIVE_DIFF_PATHS[0],
                    "resolution": "resolved_keep_live_sanitized_trace",
                    "evidence": "synthetic test evidence",
                },
                *[
                    {
                        "historical_disposition": (
                            "blocked_recovery_input_needs_canonicalization_before_promotion"
                        ),
                        "relpath": relpath,
                        "resolution": "resolved_superseded",
                        "evidence": "synthetic test evidence",
                    }
                    for relpath in EXPECTED_BLOCKED_RECOVERY_INPUTS
                ],
            ]
        },
    )

    report = audit_recovery_custody_snapshots(
        tmp_path,
        config=RecoveryCustodyConfig(
            pyc_recovery_root="pyc",
            signal_loss_root="signal",
            resolved_dispositions_manifest="resolved.json",
        ),
    )
    payload = report.to_dict()

    assert payload["ready_for_recovery_custody_preservation"] is True
    assert payload["summary"]["next_required_dispositions"]["blocked_recovery_inputs"] == []
    assert payload["summary"]["next_required_dispositions"]["live_diff_paths"] == []
    assert payload["summary"]["signal_loss"]["resolved_live_diff_paths"] == [EXPECTED_LIVE_DIFF_PATHS[0]]


def test_recovery_custody_audit_blocks_nonhistorical_resolved_disposition(tmp_path: Path) -> None:
    _write_expected_snapshot(tmp_path)
    write_json(
        tmp_path / "resolved.json",
        {
            "entries": [
                {
                    "historical_disposition": "compare_by_hand_live_diff_before_merge_or_delete",
                    "relpath": "not/in/snapshot",
                    "resolution": "bad",
                    "evidence": "synthetic test evidence",
                }
            ]
        },
    )

    report = audit_recovery_custody_snapshots(
        tmp_path,
        config=RecoveryCustodyConfig(
            pyc_recovery_root="pyc",
            signal_loss_root="signal",
            resolved_dispositions_manifest="resolved.json",
        ),
    )
    payload = report.to_dict()

    assert payload["ready_for_recovery_custody_preservation"] is False
    assert "non-historical" in "\n".join(payload["blockers"])
