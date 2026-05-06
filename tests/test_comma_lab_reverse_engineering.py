from __future__ import annotations

import json
import zipfile
from pathlib import Path

from comma_lab.reverse_engineering import (
    audit_reverse_engineering_tree,
    load_release_resolution_rules,
    release_blocking_records,
    render_json,
    render_markdown,
)


def test_reverse_engineering_audit_classifies_curated_orphan_and_raw(tmp_path: Path) -> None:
    repo = tmp_path
    reverse = repo / "reverse_engineering"
    orphan = reverse / "orphan_pyc_recovery_20260505_codex"
    (reverse / "pr95_hnerv").mkdir(parents=True)
    (reverse / "README.md").write_text("# Reverse Engineering\n", encoding="utf-8")
    (reverse / "pr95_hnerv" / "README.md").write_text("# PR95\n", encoding="utf-8")
    raw = reverse / "archive.zip"
    with zipfile.ZipFile(raw, "w") as zf:
        zf.writestr("0.bin", b"x")
    cache = reverse / "__pycache__/ignored.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"cache")

    live_tac = repo / "src/tac"
    live_tac.mkdir(parents=True)
    (live_tac / "codec.py").write_text("VALUE = 1\n", encoding="utf-8")
    orphan_tac = orphan / "src/tac/codec.py"
    orphan_tac.parent.mkdir(parents=True)
    orphan_tac.write_text("VALUE = 2\n", encoding="utf-8")
    duplicate = orphan / "src/tac/duplicate.py"
    (live_tac / "duplicate.py").write_text("VALUE = 3\n", encoding="utf-8")
    duplicate.write_text(
        "# pyc-recovery: human-reconstructed from src/tac/duplicate.py.pyc\n"
        "# This is the canonical main-repo content as of 2026-05-05.\n"
        "# Recovery spec preserved at: duplicate.recovery_spec.json\n"
        "# Original STUB has been replaced with this canonical version.\n"
        "VALUE = 3\n",
        encoding="utf-8",
    )
    spec = orphan / "src/tac/codec.recovery_spec.json"
    spec.write_text('{"pyc_path": "src/tac/__pycache__/codec.pyc"}\n', encoding="utf-8")
    damaged = orphan / "experiments/preflight_pr91_pr92_replay_contracts.py"
    damaged.parent.mkdir(parents=True, exist_ok=True)
    damaged.write_text(
        "# Source Generated with Decompyle++\n"
        "def _read_or_run(report_path):\n"
        "    report = None(report_path)\n"
        "    return report\n",
        encoding="utf-8",
    )
    live_script = repo / "scripts/remote_lane_x.sh"
    live_script.parent.mkdir(parents=True)
    live_script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
    shadow_script = orphan / "scripts/remote_lane_x.sh.PREFLIGHT_DEBT"
    shadow_script.parent.mkdir(parents=True)
    shadow_script.write_text("#!/usr/bin/env bash\nset -euo pipefail\n", encoding="utf-8")
    live_trace = repo / "docs/paper/ara/trace/events.jsonl"
    live_trace.parent.mkdir(parents=True)
    live_trace.write_text(
        '{"event_type":"observation","source_path":"<operator-memory>/MEMORY.md","summary":"MEMORY"}\n',
        encoding="utf-8",
    )
    recovered_trace = orphan / "docs/paper/ara/trace/events.jsonl"
    recovered_trace.parent.mkdir(parents=True)
    recovered_trace.write_text(
        '{"event_type":"observation","source_path":'
        '"/Users/adpena/.claude/projects/-Users-adpena-Projects-pact/memory/MEMORY.md",'
        '"summary":"MEMORY"}\n',
        encoding="utf-8",
    )
    memory = orphan / ".omx/auto_memory_snapshot_20260504T230223Z/feedback.md"
    memory.parent.mkdir(parents=True)
    memory.write_text("important memory\n", encoding="utf-8")

    records = audit_reverse_engineering_tree(repo, reverse_root=reverse)
    by_rel = {record.relpath: record for record in records}

    assert "reverse_engineering/__pycache__/ignored.pyc" not in by_rel
    assert by_rel["reverse_engineering/README.md"].disposition == "track_in_git"
    assert by_rel["reverse_engineering/archive.zip"].disposition == "externalize_with_manifest"
    assert by_rel["reverse_engineering/orphan_pyc_recovery_20260505_codex/src/tac/codec.py"].disposition == (
        "compare_and_promote_to_tac"
    )
    assert by_rel["reverse_engineering/orphan_pyc_recovery_20260505_codex/src/tac/codec.py"].live_status == (
        "differs_from_main"
    )
    assert by_rel["reverse_engineering/orphan_pyc_recovery_20260505_codex/src/tac/duplicate.py"].disposition == (
        "delete_after_manifest"
    )
    assert by_rel["reverse_engineering/orphan_pyc_recovery_20260505_codex/src/tac/duplicate.py"].live_status == (
        "same_as_main_ignoring_recovery_header"
    )
    assert by_rel[
        "reverse_engineering/orphan_pyc_recovery_20260505_codex/src/tac/codec.recovery_spec.json"
    ].disposition == "preserve_until_source_disposition"
    damaged_rel = (
        "reverse_engineering/orphan_pyc_recovery_20260505_codex/"
        "experiments/preflight_pr91_pr92_replay_contracts.py"
    )
    assert by_rel[damaged_rel].category == "orphan_damaged_decompile"
    assert by_rel[damaged_rel].disposition == "preserve_until_hand_rehydration"
    assert by_rel[damaged_rel].target == "experiments/preflight_pr91_pr92_replay_contracts.py"
    assert by_rel[
        "reverse_engineering/orphan_pyc_recovery_20260505_codex/.omx/auto_memory_snapshot_20260504T230223Z/feedback.md"
    ].disposition == "summarize_to_research_ledger"
    assert by_rel[
        "reverse_engineering/orphan_pyc_recovery_20260505_codex/scripts/remote_lane_x.sh.PREFLIGHT_DEBT"
    ].category == "orphan_operator_tool_shadow"
    assert by_rel[
        "reverse_engineering/orphan_pyc_recovery_20260505_codex/scripts/remote_lane_x.sh.PREFLIGHT_DEBT"
    ].disposition == "delete_after_manifest"
    assert by_rel[
        "reverse_engineering/orphan_pyc_recovery_20260505_codex/scripts/remote_lane_x.sh.PREFLIGHT_DEBT"
    ].target == "scripts/remote_lane_x.sh"
    assert by_rel[
        "reverse_engineering/orphan_pyc_recovery_20260505_codex/docs/paper/ara/trace/events.jsonl"
    ].category == "orphan_report_private_path_shadow"
    assert by_rel[
        "reverse_engineering/orphan_pyc_recovery_20260505_codex/docs/paper/ara/trace/events.jsonl"
    ].disposition == "delete_after_manifest"

    payload = json.loads(render_json(records))
    assert payload["total_files"] == len(records)
    assert payload["disposition_counts"]["compare_and_promote_to_tac"] == 1
    assert payload["disposition_counts"]["preserve_until_hand_rehydration"] == 1
    release_payload = json.loads(render_json(records, release_strict=True))
    assert release_payload["release_strict"] is True
    assert release_payload["blocking_count"] > payload["blocking_count"]
    release_blockers = release_blocking_records(records)
    assert any(record.disposition == "compare_and_promote_to_tac" for record in release_blockers)
    assert any(record.disposition == "preserve_until_hand_rehydration" for record in release_blockers)
    release_manifest = tmp_path / "release_manifest.json"
    release_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rules": [
                    {
                        "id": "summaries",
                        "match": {"disposition": "summarize_to_research_ledger"},
                        "action": "summarized_to_ledger",
                        "public_release": "publish_sanitized_summary",
                        "ledger_path": ".omx/research/recovery_quarantine_signal_loss_triage_20260505_codex.md",
                        "note": "Synthetic test summary rule.",
                    },
                    {
                        "id": "promote",
                        "match": {"disposition": "compare_and_promote_to_tac"},
                        "action": "promoted_or_ledgered",
                        "public_release": "publish_curated_source",
                        "ledger_path": ".omx/research/recovery_quarantine_signal_loss_triage_20260505_codex.md",
                        "note": "Synthetic test promotion rule.",
                    },
                    {
                        "id": "preserve",
                        "match": {"disposition": "preserve_until_hand_rehydration"},
                        "action": "preserved_private_with_digest",
                        "public_release": "exclude_raw",
                        "ledger_path": ".omx/research/recovery_quarantine_signal_loss_triage_20260505_codex.md",
                        "note": "Synthetic test preservation rule.",
                    },
                    {
                        "id": "externalize",
                        "match": {"disposition": "externalize_with_manifest"},
                        "action": "externalized_or_excluded",
                        "public_release": "exclude_raw",
                        "ledger_path": ".omx/research/recovery_quarantine_signal_loss_triage_20260505_codex.md",
                        "note": "Synthetic test externalization rule.",
                    },
                    {
                        "id": "spec",
                        "match": {"disposition": "preserve_until_source_disposition"},
                        "action": "preserved_private_with_digest",
                        "public_release": "exclude_raw",
                        "ledger_path": ".omx/research/recovery_quarantine_signal_loss_triage_20260505_codex.md",
                        "note": "Synthetic test recovery-spec rule.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    release_rules = load_release_resolution_rules(release_manifest)
    resolved_payload = json.loads(render_json(records, release_strict=True, release_rules=release_rules))
    assert resolved_payload["blocking_count"] == 0
    assert resolved_payload["release_resolution_rule_count"] == len(release_rules)
    markdown = render_markdown(records)
    assert "Promotion Queue" in markdown
    assert "src/tac/codec.py" in markdown
    assert "Preserved Recovery Signal" in markdown
    assert "preflight_pr91_pr92_replay_contracts.py" in markdown
