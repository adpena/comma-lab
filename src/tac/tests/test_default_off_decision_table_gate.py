"""Tests for check_default_off_decision_table_consumed (#405, 2026-07-10).

The default-off decision-table consume gate: refuses (a) rot of the decision
table (missing memo/twin, unparseable twin, bad rows/dispositions) and (b) a
NEW config-finalization artifact (crucible SPEC / authored config dated after
the table) that neither references the table nor records consumption nor
carries a non-placeholder waiver. WARN-ONLY in preflight_all; strict here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.preflight import (
    _DEFAULT_OFF_TABLE_JSONL_REL,
    _DEFAULT_OFF_TABLE_MEMO_REL,
    PreflightError,
    check_default_off_decision_table_consumed,
)

MEMO_REL = Path(_DEFAULT_OFF_TABLE_MEMO_REL)
TWIN_REL = Path(_DEFAULT_OFF_TABLE_JSONL_REL)


def _seed_repo(root: Path, *, memo: bool = True, twin_rows: list[str] | None = None) -> None:
    """Create a minimal fake repo with the decision-table artifacts."""
    (root / MEMO_REL).parent.mkdir(parents=True, exist_ok=True)
    if memo:
        (root / MEMO_REL).write_text("# DEFAULT-OFF SWEEP\n", encoding="utf-8")
    if twin_rows is not None:
        (root / TWIN_REL).write_text("\n".join(twin_rows) + "\n", encoding="utf-8")


def _meta_row() -> str:
    return json.dumps({"_meta": {"table": "default_off_decision_table", "version": "20260710"}})


def _good_row(name: str = "SomeLever", disposition: str = "measure-cheap($0/n600)") -> str:
    return json.dumps({"name": name, "surface": "duty-queue", "disposition": disposition})


class TestTableIntegrity:
    def test_live_repo_is_clean(self) -> None:
        # The real repo carries the memo + twin and no post-dated unconsuming artifacts.
        assert check_default_off_decision_table_consumed(strict=False) == []

    def test_ok_on_minimal_valid_fixture(self, tmp_path: Path) -> None:
        _seed_repo(tmp_path, twin_rows=[_meta_row(), _good_row()])
        assert check_default_off_decision_table_consumed(repo_root=tmp_path) == []

    def test_missing_memo_is_violation(self, tmp_path: Path) -> None:
        _seed_repo(tmp_path, memo=False, twin_rows=[_meta_row(), _good_row()])
        v = check_default_off_decision_table_consumed(repo_root=tmp_path)
        assert any("comprehensive_sweep" in x and "MISSING" in x for x in v)

    def test_missing_twin_is_violation(self, tmp_path: Path) -> None:
        _seed_repo(tmp_path, twin_rows=None)
        v = check_default_off_decision_table_consumed(repo_root=tmp_path)
        assert any("decision_table" in x and "MISSING" in x for x in v)

    def test_unparseable_twin_is_violation(self, tmp_path: Path) -> None:
        _seed_repo(tmp_path)
        (tmp_path / TWIN_REL).write_text("{not json\n", encoding="utf-8")
        v = check_default_off_decision_table_consumed(repo_root=tmp_path)
        assert any("unparseable" in x or "unreadable" in x for x in v)

    def test_row_missing_required_field_is_violation(self, tmp_path: Path) -> None:
        bad = json.dumps({"name": "X", "disposition": "measure-cheap($0/n600)"})  # no surface
        _seed_repo(tmp_path, twin_rows=[_meta_row(), bad])
        v = check_default_off_decision_table_consumed(repo_root=tmp_path)
        assert any("missing required" in x and "surface" in x for x in v)

    def test_disposition_outside_enum_is_violation(self, tmp_path: Path) -> None:
        bad = json.dumps({"name": "X", "surface": "s", "disposition": "maybe-later"})
        _seed_repo(tmp_path, twin_rows=[_meta_row(), bad])
        v = check_default_off_decision_table_consumed(repo_root=tmp_path)
        assert any("not in the registered enum" in x for x in v)

    def test_missing_meta_header_is_violation(self, tmp_path: Path) -> None:
        _seed_repo(tmp_path, twin_rows=[_good_row()])
        v = check_default_off_decision_table_consumed(repo_root=tmp_path)
        assert any("_meta" in x for x in v)

    def test_strict_raises(self, tmp_path: Path) -> None:
        _seed_repo(tmp_path, memo=False, twin_rows=[_meta_row(), _good_row()])
        with pytest.raises(PreflightError):
            check_default_off_decision_table_consumed(repo_root=tmp_path, strict=True)


class TestFinalizationConsumption:
    def _spec(self, root: Path, name: str, body: str) -> Path:
        d = root / ".omx" / "research" / "t5_crucible4"
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        p.write_text(body, encoding="utf-8")
        return p

    def test_newer_spec_without_reference_is_violation(self, tmp_path: Path) -> None:
        _seed_repo(tmp_path, twin_rows=[_meta_row(), _good_row()])
        self._spec(tmp_path, "SPEC_v753_20260712.md", "# v7.5.3 SPEC\nno table reference\n")
        v = check_default_off_decision_table_consumed(repo_root=tmp_path)
        assert any("does not reference" in x for x in v)

    def test_newer_spec_with_consumed_line_is_ok(self, tmp_path: Path) -> None:
        _seed_repo(tmp_path, twin_rows=[_meta_row(), _good_row()])
        self._spec(
            tmp_path, "SPEC_v753_20260712.md",
            "# v7.5.3 SPEC\nDEFAULT_OFF_TABLE_CONSUMED: default_off_decision_table_20260710\n")
        assert check_default_off_decision_table_consumed(repo_root=tmp_path) == []

    def test_newer_spec_naming_table_file_is_ok(self, tmp_path: Path) -> None:
        _seed_repo(tmp_path, twin_rows=[_meta_row(), _good_row()])
        self._spec(
            tmp_path, "SPEC_v753_20260712.md",
            "# v7.5.3 SPEC\nconsumed default_off_comprehensive_sweep_20260710.md rows 1-185\n")
        assert check_default_off_decision_table_consumed(repo_root=tmp_path) == []

    def test_waiver_with_rationale_is_ok(self, tmp_path: Path) -> None:
        _seed_repo(tmp_path, twin_rows=[_meta_row(), _good_row()])
        self._spec(
            tmp_path, "SPEC_v753_20260712.md",
            "# SPEC\n# DEFAULT_OFF_TABLE_OK:byte-close-only spec, no trainer config surface\n")
        assert check_default_off_decision_table_consumed(repo_root=tmp_path) == []

    def test_placeholder_waiver_is_rejected(self, tmp_path: Path) -> None:
        _seed_repo(tmp_path, twin_rows=[_meta_row(), _good_row()])
        self._spec(tmp_path, "SPEC_v753_20260712.md", "# SPEC\n# DEFAULT_OFF_TABLE_OK:<rationale>\n")
        v = check_default_off_decision_table_consumed(repo_root=tmp_path)
        assert any("does not reference" in x for x in v)

    def test_older_spec_is_exempt(self, tmp_path: Path) -> None:
        _seed_repo(tmp_path, twin_rows=[_meta_row(), _good_row()])
        self._spec(tmp_path, "SPEC_v75_optimal_single_trunk_20260708.md", "# old SPEC, no ref\n")
        assert check_default_off_decision_table_consumed(repo_root=tmp_path) == []

    def test_undated_artifact_is_exempt(self, tmp_path: Path) -> None:
        _seed_repo(tmp_path, twin_rows=[_meta_row(), _good_row()])
        self._spec(tmp_path, "SPEC_someday.md", "# undated SPEC, no ref\n")
        assert check_default_off_decision_table_consumed(repo_root=tmp_path) == []

    def test_authored_config_pattern_also_gated(self, tmp_path: Path) -> None:
        _seed_repo(tmp_path, twin_rows=[_meta_row(), _good_row()])
        d = tmp_path / ".omx" / "research"
        d.mkdir(parents=True, exist_ok=True)
        (d / "crucible_v753_authored_20260713.md").write_text("# authored config, no ref\n",
                                                              encoding="utf-8")
        v = check_default_off_decision_table_consumed(repo_root=tmp_path)
        assert any("crucible_v753_authored_20260713" in x for x in v)
