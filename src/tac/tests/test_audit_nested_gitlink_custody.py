from __future__ import annotations

from tools.audit_nested_gitlink_custody import (
    NestedGitlinkRecord,
    dirty_gitlink_statuses,
    parse_gitlink_paths,
    render_payload,
)


def test_parse_gitlink_paths_keeps_only_mode_160000() -> None:
    text = "\n".join(
        [
            "100644 abc123 0\tsrc/tac/file.py",
            "160000 deadbee 0\texperiments/results/public_pr100/source",
            "160000 c5e1274 0\treports/raw/kaggle_ingest/source",
        ]
    )

    assert parse_gitlink_paths(text) == {
        "experiments/results/public_pr100/source",
        "reports/raw/kaggle_ingest/source",
    }


def test_dirty_gitlink_statuses_ignores_non_gitlink_paths() -> None:
    statuses = "\n".join(
        [
            " M src/tac/file.py",
            " m experiments/results/public_pr100/source",
            " ? reports/raw/kaggle_ingest/source",
            " M reverse_engineering/orphan/file.py",
        ]
    )

    assert dirty_gitlink_statuses(
        statuses,
        {
            "experiments/results/public_pr100/source",
            "reports/raw/kaggle_ingest/source",
        },
    ) == [
        (" m", "experiments/results/public_pr100/source"),
        (" ?", "reports/raw/kaggle_ingest/source"),
    ]


def test_render_payload_blocks_undocumented_dirty_gitlink() -> None:
    payload = render_payload(
        [
            NestedGitlinkRecord(
                xy=" m",
                path="experiments/results/public_pr100/source",
                head="abc1234",
                severity="warning",
                documented_by=None,
                dirty_count=1,
                dirty_entries=(" M submissions/quantizr/compress.py",),
            )
        ]
    )

    assert payload["ready_for_public_release_split"] is False
    assert payload["summary"]["warning_count"] == 1
    assert payload["blockers"] == [
        "experiments/results/public_pr100/source: dirty nested gitlink lacks a local-custody manifest rule"
    ]


def test_render_payload_accepts_documented_dirty_gitlink() -> None:
    payload = render_payload(
        [
            NestedGitlinkRecord(
                xy=" m",
                path="experiments/results/public_pr100/source",
                head="abc1234",
                severity="info",
                documented_by="public_pr_intake_gitlinks_forensic",
                dirty_count=1,
                dirty_entries=(" M submissions/quantizr/compress.py",),
            )
        ]
    )

    assert payload["ready_for_public_release_split"] is True
    assert payload["summary"]["documented_count"] == 1
    assert payload["blockers"] == []
