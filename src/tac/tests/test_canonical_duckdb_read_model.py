# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tac.canonical_duckdb import (
    CANONICAL_TABLES,
    audit_table_provenance,
    push_canonical_task_status_to_hf,
    push_to_hf,
    refresh_all_tables,
    run_canonical_query,
)
from tac.canonical_task_status import register_task, update_status
from tools.refresh_canonical_duckdb import main as refresh_canonical_duckdb_main


class _FakeHfApi:
    def __init__(self, *, visibility_after_update: bool = True) -> None:
        self.visibility_after_update = visibility_after_update
        self.calls: list[tuple[str, object]] = []
        self.private = False

    def create_repo(self, **kwargs) -> None:
        self.calls.append(("create_repo", kwargs))
        self.private = bool(kwargs.get("private"))

    def update_repo_settings(self, **kwargs) -> None:
        self.calls.append(("update_repo_settings", kwargs))
        self.private = self.visibility_after_update

    def repo_info(self, repo_id, repo_type):
        self.calls.append(("repo_info", {"repo_id": repo_id, "repo_type": repo_type}))
        return SimpleNamespace(private=self.private, sha="fake-hf-sha")

    def upload_file(self, **kwargs) -> None:
        self.calls.append(("upload_file", kwargs))


def _assert_no_uploaded_local_paths(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            assert key != "path"
            if key == "duckdb_path":
                assert str(nested).startswith(".omx/")
            _assert_no_uploaded_local_paths(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_uploaded_local_paths(nested)


def _seed_repo(root: Path) -> None:
    state = root / ".omx" / "state"
    research = root / ".omx" / "research"
    state.mkdir(parents=True)
    research.mkdir(parents=True)
    (state / "lane_registry.json").write_text(
        json.dumps(
            {
                "lanes": [
                    {
                        "id": "lane_test",
                        "name": "Test Lane",
                        "phase": 2,
                        "level": 0,
                        "gates": {},
                        "notes": "",
                    }
                ]
            }
        )
    )
    (state / "events.jsonl").write_text('{"event": 1}\n{"event": 2}\n')
    (state / "sample.json").write_text('{"ok": true}\n')
    (research / "test_memo.md").write_text(
        "---\n"
        "title: Test Memo\n"
        "date_utc: 2026-05-18T00:00:00Z\n"
        "lane_id: lane_test\n"
        "research_only: true\n"
        "score_claim: false\n"
        "promotion_eligible: false\n"
        "---\n"
        "\n"
        "# Test Memo\n"
    )
    register_task(
        "test_memo::ITEM_1",
        ".omx/research/test_memo.md",
        "Pending task",
        "codex",
        actor="pytest",
        session_id="test",
        repo_root=root,
    )
    register_task(
        "test_memo::ITEM_2",
        ".omx/research/test_memo.md",
        "Completed task",
        "codex",
        actor="pytest",
        session_id="test",
        repo_root=root,
    )
    update_status(
        "test_memo::ITEM_2",
        "in_progress",
        actor="pytest",
        session_id="test",
        repo_root=root,
    )
    update_status(
        "test_memo::ITEM_2",
        "completed",
        actor="pytest",
        session_id="test",
        test_status="green",
        commit_shas=("abc123",),
        notes="[empirical:src/tac/tests/test_canonical_duckdb_read_model.py]",
        repo_root=root,
    )


def test_canonical_duckdb_refreshes_source_backed_tables(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    db_path = tmp_path / ".omx" / "state" / "canonical.duckdb"

    result = refresh_all_tables(
        tmp_path,
        db_path=db_path,
        tables=(
            "lanes",
            "research_memos",
            "state_json_files",
            "state_jsonl_files",
            "canonical_task_status",
        ),
    )

    assert set(CANONICAL_TABLES) >= {"lanes", "research_memos"}
    assert result["lanes"]["row_count"] == 1
    assert result["research_memos"]["row_count"] == 1
    assert result["state_json_files"]["row_count"] >= 1
    assert result["state_jsonl_files"]["row_count"] == 2
    assert result["canonical_task_status"]["row_count"] == 4

    provenance = audit_table_provenance("research_memos", db_path=db_path)
    assert provenance["duckdb_is_source_of_truth"] is False
    assert provenance["row_count"] == 1

    rows = run_canonical_query("lanes_with_research_memos", db_path=db_path)
    assert rows[0]["lane_id"] == "lane_test"
    assert rows[0]["memo_count"] == 1

    task_summary = run_canonical_query("canonical_task_status_by_memo", db_path=db_path)
    assert task_summary[0]["source_design_memo"] == ".omx/research/test_memo.md"
    assert task_summary[0]["memo_title"] == "Test Memo"
    assert task_summary[0]["task_count"] == 2
    assert task_summary[0]["pending_count"] == 1
    assert task_summary[0]["completed_count"] == 1
    assert task_summary[0]["green_test_count"] == 1

    pending_tasks = run_canonical_query("canonical_task_status_pending_with_memo", db_path=db_path)
    assert [row["task_id"] for row in pending_tasks] == ["test_memo::ITEM_1"]
    assert pending_tasks[0]["memo_title"] == "Test Memo"


def test_canonical_duckdb_hf_push_requires_operator_approval(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    db_path = tmp_path / ".omx" / "state" / "canonical.duckdb"
    refresh_all_tables(tmp_path, db_path=db_path, tables=("lanes",))

    with pytest.raises(PermissionError, match="operator_approved"):
        push_to_hf(
            "lanes",
            "adpena/comma-canonical-test",
            db_path=db_path,
            export_dir=tmp_path / "export",
            operator_approved=False,
        )


def test_canonical_task_status_hf_export_dry_run_private_default(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    db_path = tmp_path / ".omx" / "state" / "canonical.duckdb"
    refresh_all_tables(tmp_path, db_path=db_path, tables=("canonical_task_status",))

    manifest = push_canonical_task_status_to_hf(
        db_path,
        export_dir=tmp_path / "hf_export",
        repo_root=tmp_path,
    )

    assert manifest["status"] == "dry_run_pending_operator_approval"
    assert manifest["private"] is True
    assert manifest["remote_push_fired"] is False
    assert manifest["duckdb_is_source_of_truth"] is False
    assert manifest["source_of_truth"] == ".omx/state/canonical_task_status.jsonl"
    assert manifest["history_rows"] == 4
    assert manifest["latest_rows"] == 2
    assert manifest["source"]["source_ledger_rows"] == 4
    assert len(manifest["source"]["source_ledger_sha256"]) == 64
    assert {
        file_row["path_in_repo"]
        for file_row in manifest["files"]
    } == {
        "data/canonical_task_status.parquet",
        "data/canonical_task_status_latest.parquet",
    }
    assert manifest["manifest"]["path_in_repo"] == "metadata/canonical_task_status_hf_manifest.json"
    assert manifest["dataset_card"]["path_in_repo"] == "README.md"
    card_text = (tmp_path / "hf_export" / "README.md").read_text(encoding="utf-8")
    assert "source of truth remains the append-only ledger" in card_text
    assert "private-only" in card_text
    for file_row in manifest["files"]:
        assert (tmp_path / "hf_export" / file_row["path_in_repo"]).exists()
        assert file_row["bytes"] > 0
        assert len(file_row["sha256"]) == 64
    manifest_path = tmp_path / "hf_export" / manifest["manifest"]["path_in_repo"]
    assert manifest_path.exists()
    assert manifest["manifest"]["bytes"] > 0
    assert len(manifest["manifest"]["sha256"]) == 64
    disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert disk_manifest["status"] == "dry_run_pending_operator_approval"
    assert disk_manifest["private"] is True
    assert disk_manifest["remote_push_fired"] is False
    _assert_no_uploaded_local_paths(disk_manifest)


def test_canonical_task_status_hf_export_refuses_unapproved_fire(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    db_path = tmp_path / ".omx" / "state" / "canonical.duckdb"
    refresh_all_tables(tmp_path, db_path=db_path, tables=("canonical_task_status",))

    with pytest.raises(PermissionError, match="operator_approved=True"):
        push_canonical_task_status_to_hf(
            db_path,
            export_dir=tmp_path / "hf_export",
            dry_run=False,
            operator_approved=False,
            repo_root=tmp_path,
        )


def test_canonical_task_status_hf_export_refuses_stale_duckdb_read_model(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    db_path = tmp_path / ".omx" / "state" / "canonical.duckdb"
    refresh_all_tables(tmp_path, db_path=db_path, tables=("canonical_task_status",))
    register_task(
        "test_memo::ITEM_3",
        ".omx/research/test_memo.md",
        "New pending task after refresh",
        "codex",
        actor="pytest",
        session_id="test",
        repo_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="DuckDB read-model is stale"):
        push_canonical_task_status_to_hf(
            db_path,
            export_dir=tmp_path / "hf_export",
            repo_root=tmp_path,
            refresh_before_export=False,
        )


def test_canonical_task_status_hf_push_forces_private_visibility_before_upload(
    tmp_path: Path,
) -> None:
    _seed_repo(tmp_path)
    db_path = tmp_path / ".omx" / "state" / "canonical.duckdb"
    fake_api = _FakeHfApi()

    manifest = push_canonical_task_status_to_hf(
        db_path,
        "adpena/private-test",
        export_dir=tmp_path / "hf_export",
        dry_run=False,
        operator_approved=True,
        repo_root=tmp_path,
        hf_api=fake_api,
    )

    calls = [name for name, _payload in fake_api.calls]
    assert calls[:3] == ["create_repo", "update_repo_settings", "repo_info"]
    assert calls.count("upload_file") == 4
    upload_paths = [
        payload["path_in_repo"]
        for name, payload in fake_api.calls
        if name == "upload_file"
    ]
    assert "README.md" in upload_paths
    assert upload_paths[-1] == "metadata/canonical_task_status_hf_manifest.json"
    assert manifest["status"] == "pushed"
    assert manifest["private"] is True
    assert manifest["hf_repo_private_verified_after_upload"] is True
    disk_manifest = json.loads(
        (tmp_path / "hf_export" / "metadata/canonical_task_status_hf_manifest.json")
        .read_text(encoding="utf-8")
    )
    assert disk_manifest["remote_push_fired"] is True
    _assert_no_uploaded_local_paths(disk_manifest)


def test_canonical_task_status_hf_push_refuses_visibility_mismatch_before_upload(
    tmp_path: Path,
) -> None:
    _seed_repo(tmp_path)
    db_path = tmp_path / ".omx" / "state" / "canonical.duckdb"
    fake_api = _FakeHfApi(visibility_after_update=False)

    with pytest.raises(RuntimeError, match="visibility mismatch"):
        push_canonical_task_status_to_hf(
            db_path,
            "adpena/private-test",
            export_dir=tmp_path / "hf_export",
            dry_run=False,
            operator_approved=True,
            repo_root=tmp_path,
            hf_api=fake_api,
        )
    assert [name for name, _payload in fake_api.calls].count("upload_file") == 0


def test_canonical_task_status_hf_export_is_private_only(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    db_path = tmp_path / ".omx" / "state" / "canonical.duckdb"

    with pytest.raises(PermissionError, match="private-only"):
        push_canonical_task_status_to_hf(
            db_path,
            export_dir=tmp_path / "hf_export",
            repo_root=tmp_path,
            private=False,
        )


def test_refresh_cli_can_emit_canonical_task_status_hf_dry_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _seed_repo(tmp_path)
    db_path = tmp_path / ".omx" / "state" / "canonical.duckdb"
    export_dir = tmp_path / "hf_export"

    rc = refresh_canonical_duckdb_main(
        [
            "--repo-root",
            str(tmp_path),
            "--db-path",
            str(db_path),
            "--tables",
            "canonical_task_status",
            "--push-hf-canonical-task-status",
            "--hf-export-dir",
            str(export_dir),
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["canonical_task_status"]["row_count"] == 4
    assert payload["hf_canonical_task_status"]["status"] == "dry_run_pending_operator_approval"
    assert payload["hf_canonical_task_status"]["private"] is True
    assert payload["hf_canonical_task_status"]["remote_push_fired"] is False


def test_refresh_cli_refuses_canonical_task_status_public_raw_export(tmp_path: Path) -> None:
    _seed_repo(tmp_path)
    db_path = tmp_path / ".omx" / "state" / "canonical.duckdb"

    with pytest.raises(SystemExit, match="private-only"):
        refresh_canonical_duckdb_main(
            [
                "--repo-root",
                str(tmp_path),
                "--db-path",
                str(db_path),
                "--tables",
                "canonical_task_status",
                "--push-hf-canonical-task-status",
                "--hf-public",
            ]
        )
