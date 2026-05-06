from __future__ import annotations

from tools.audit_release_index_split import (
    IndexRecord,
    document_local_custody,
    parse_status_porcelain,
)


def test_parse_status_porcelain_handles_renames_and_untracked() -> None:
    text = "\n".join(
        [
            "MM docs/paper/07_discussion.md",
            "R  old.py -> tools/new.py",
            "?? scratch.py",
        ]
    )

    assert parse_status_porcelain(text) == [
        ("MM", "docs/paper/07_discussion.md"),
        ("R ", "tools/new.py"),
        ("??", "scratch.py"),
    ]


def test_index_record_schema_is_stable() -> None:
    record = IndexRecord(
        xy="MM",
        path=".omx/state/lane_registry.json",
        kind="staged_private_runtime_state",
        severity="blocker",
        detail="provider state must be summarized",
    )

    assert record.path.endswith("lane_registry.json")
    assert record.severity == "blocker"


def test_parse_status_porcelain_keeps_gitlink_worktree_status() -> None:
    text = " m experiments/results/public_pr100_intake/source"

    assert parse_status_porcelain(text) == [
        (" m", "experiments/results/public_pr100_intake/source")
    ]


def test_document_local_custody_downgrades_matching_warning_to_info() -> None:
    record = IndexRecord(
        xy=" M",
        path="experiments/results/public_pr100_intake/source",
        kind="unstaged_local_custody_snapshot",
        severity="warning",
        detail="raw snapshot",
    )
    documented = document_local_custody(
        record,
        [
            {
                "id": "public_intake",
                "match": {
                    "kind": "unstaged_local_custody_snapshot",
                    "path_prefix": "experiments/results/public_pr",
                },
            }
        ],
    )

    assert documented.severity == "info"
    assert documented.documented_by == "public_intake"
