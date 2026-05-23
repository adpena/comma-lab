from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REVIEW_TRACKER = REPO_ROOT / "tools" / "review_tracker.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("review_tracker_under_test", REVIEW_TRACKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _insert_entity(module, *, file_path: str) -> None:
    con = module._init_db()
    con.execute(
        """
        INSERT INTO entities (
            qualified_name, module, file_path, entity_type, name,
            start_line, end_line, line_count, complexity, review_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            f"{file_path}::func",
            file_path.replace("/", ".").removesuffix(".py"),
            file_path,
            "function",
            "func",
            1,
            3,
            3,
            1,
            "unreviewed",
        ],
    )
    con.close()


def _review_status(module, *, file_path: str) -> str:
    con = module._connect_duckdb(module.TRACKER_DB)
    try:
        row = con.execute(
            "SELECT review_status FROM entities WHERE file_path = ?",
            [file_path],
        ).fetchone()
    finally:
        con.close()
    assert row is not None
    return str(row[0])


def test_greenup_import_refuses_issue_only_bullet_fallback(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "TRACKER_DB", tmp_path / "review_tracker.duckdb")
    monkeypatch.setattr(module, "TRACKER_JSON", tmp_path / "review_tracker.json")
    _insert_entity(module, file_path="src/tac/fake_impl.py")
    pass_file = tmp_path / "greenup.md"
    pass_file.write_text(
        "- src/tac/fake_impl.py -- ISSUES FOUND: fake implementation\n",
        encoding="utf-8",
    )

    module.cmd_greenup_import(str(pass_file))

    assert _review_status(module, file_path="src/tac/fake_impl.py") == "unreviewed"


def test_greenup_import_refuses_not_clean_bullet(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "TRACKER_DB", tmp_path / "review_tracker.duckdb")
    monkeypatch.setattr(module, "TRACKER_JSON", tmp_path / "review_tracker.json")
    _insert_entity(module, file_path="src/tac/not_clean_impl.py")
    pass_file = tmp_path / "greenup.md"
    pass_file.write_text(
        "- src/tac/not_clean_impl.py -- NOT CLEAN: needs follow-up\n",
        encoding="utf-8",
    )

    module.cmd_greenup_import(str(pass_file))

    assert _review_status(module, file_path="src/tac/not_clean_impl.py") == "unreviewed"


def test_greenup_import_marks_clean_verdict_only(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "TRACKER_DB", tmp_path / "review_tracker.duckdb")
    monkeypatch.setattr(module, "TRACKER_JSON", tmp_path / "review_tracker.json")
    _insert_entity(module, file_path="src/tac/clean_impl.py")
    pass_file = tmp_path / "greenup.md"
    pass_file.write_text("- src/tac/clean_impl.py -- CLEAN\n", encoding="utf-8")

    module.cmd_greenup_import(str(pass_file))

    assert _review_status(module, file_path="src/tac/clean_impl.py") == "reviewed"
