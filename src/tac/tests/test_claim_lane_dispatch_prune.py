# SPDX-License-Identifier: MIT
"""Tests for ``tools/claim_lane_dispatch.py prune`` (T1-E state hygiene wave).

Memory: ``feedback_state_hygiene_gc_and_prune_landed_20260512.md``.
"""
from __future__ import annotations

import importlib.util
import json
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "claim_lane_dispatch.py"


def _load_tool_module():
    import sys

    name = "claim_lane_dispatch_under_test"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TOOL_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cld_mod():
    return _load_tool_module()


@pytest.fixture
def fake_ledger(tmp_path):
    """Construct a fake live-ledger with a mix of terminal-old, terminal-recent,
    and active rows."""

    ledger = tmp_path / "active_lane_dispatch_claims.md"
    body = textwrap.dedent(
        """\
        # Active lane dispatch claims — cross-agent coordination ledger

        ## Claims (newest first)

        | timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |
        |---|---|---|---|---|---|---|---|
        | 2026-05-12T10:00:00Z | t | lane_active_a | modal | job_active_a |  | active_dispatch | recent |
        | 2026-05-11T10:00:00Z | t | lane_recent_terminal | modal | job_recent_terminal |  | completed_ok | terminal but recent (<7d) |
        | 2026-04-30T10:00:00Z | t | lane_old_terminal_a | modal | job_old_terminal_a |  | completed_ok | old terminal (>7d) |
        | 2026-04-29T10:00:00Z | t | lane_old_terminal_a | modal | job_old_terminal_a |  | active_dispatch | predecessor row |
        | 2026-04-15T10:00:00Z | t | lane_old_terminal_b | modal | job_old_terminal_b |  | failed_modal_rc | very old terminal |
        """
    )
    ledger.write_text(body)
    return ledger


# ── _plan_prune ────────────────────────────────────────────────────────────


def test_plan_prune_partitions_correctly(cld_mod, fake_ledger):
    import datetime as dt

    text = fake_ledger.read_text()
    claims = cld_mod._parse_claims(text)
    now = dt.datetime(2026, 5, 12, 11, tzinfo=dt.UTC)
    keep, prune, ambiguous = cld_mod._plan_prune(
        claims, now_utc=now, terminal_age_days=7.0, ttl_hours=24.0
    )
    # lane_active_a (active) → kept  # FAKE_LANE_OK: claim-prune unit-test fixture
    # lane_recent_terminal (1d ago terminal) → kept  # FAKE_LANE_OK: claim-prune unit-test fixture
    # lane_old_terminal_a (12d ago latest terminal) → prune BOTH rows  # FAKE_LANE_OK: claim-prune unit-test fixture
    # lane_old_terminal_b (27d ago terminal) → prune  # FAKE_LANE_OK: claim-prune unit-test fixture
    assert ambiguous == []
    pruned_lanes = {c.lane_id for c in prune}
    assert pruned_lanes == {"lane_old_terminal_a", "lane_old_terminal_b"}  # FAKE_LANE_OK: claim-prune unit-test fixture
    keep_lanes = {c.lane_id for c in keep}
    assert keep_lanes == {"lane_active_a", "lane_recent_terminal"}  # FAKE_LANE_OK: claim-prune unit-test fixture
    # lane_old_terminal_a has 2 rows; both should be pruned  # FAKE_LANE_OK: claim-prune unit-test fixture
    assert sum(1 for c in prune if c.lane_id == "lane_old_terminal_a") == 2  # FAKE_LANE_OK: claim-prune unit-test fixture


def test_plan_prune_protects_active_lanes(cld_mod, tmp_path):
    """Active lanes are kept even when they have predecessor rows older than
    the prune threshold."""

    import datetime as dt

    text = textwrap.dedent(
        """\
        # ledger

        ## Claims (newest first)

        | timestamp_utc | agent | lane_id | platform | instance/job_id | predicted_eta_utc | status | notes |
        |---|---|---|---|---|---|---|---|
        | 2026-05-12T10:00:00Z | t | lane_x | modal | job_x |  | active_dispatch | newest |
        | 2026-04-01T10:00:00Z | t | lane_x | modal | job_x_prev |  | completed_ok | 40d old terminal |
        """
    )
    claims = cld_mod._parse_claims(text)
    now = dt.datetime(2026, 5, 12, 11, tzinfo=dt.UTC)
    keep, prune, _ = cld_mod._plan_prune(
        claims, now_utc=now, terminal_age_days=7.0, ttl_hours=24.0
    )
    # lane_x active job_x is kept; lane_x:job_x_prev is a DIFFERENT (lane,
    # job_id) pair whose latest row IS terminal AND old → prune.
    assert any(c.instance_job_id == "job_x" for c in keep)
    assert any(c.instance_job_id == "job_x_prev" for c in prune)


# ── _archive_month_key / _build_archive_text_with_appended ────────────────


def test_archive_month_key_format(cld_mod):
    c = cld_mod.Claim(
        timestamp_utc="2026-05-12T10:00:00Z",
        agent="t",
        lane_id="x",
        platform="modal",
        instance_job_id="j",
        predicted_eta_utc="",
        status="completed_ok",
        notes="",
    )
    assert cld_mod._archive_month_key(c) == "2026-05"


def test_archive_month_key_unparsable(cld_mod):
    c = cld_mod.Claim(
        timestamp_utc="not-a-date",
        agent="t",
        lane_id="x",
        platform="modal",
        instance_job_id="j",
        predicted_eta_utc="",
        status="completed_ok",
        notes="",
    )
    assert cld_mod._archive_month_key(c) is None


def test_build_archive_appends_and_dedups(cld_mod):
    existing_lines = []
    c1 = cld_mod.Claim(
        "2026-05-01T10:00:00Z", "t", "lane_a", "modal", "job_a", "", "completed_ok", "n1"
    )
    c2 = cld_mod.Claim(
        "2026-05-02T10:00:00Z", "t", "lane_b", "modal", "job_b", "", "completed_ok", "n2"  # FAKE_LANE_OK: claim-prune unit-test fixture
    )
    text1 = cld_mod._build_archive_text_with_appended(
        existing_lines, [c1, c2], month_key="2026-05"
    )
    # Re-append same rows; should dedupe.
    text2 = cld_mod._build_archive_text_with_appended(
        text1.splitlines(keepends=True), [c1, c2], month_key="2026-05"
    )
    n1 = text1.count("job_a")
    n2 = text2.count("job_a")
    assert n1 == 1
    assert n2 == 1


def test_build_archive_sorts_newest_first(cld_mod):
    c_old = cld_mod.Claim(
        "2026-05-01T10:00:00Z", "t", "x", "modal", "ja", "", "completed_ok", ""
    )
    c_new = cld_mod.Claim(
        "2026-05-05T10:00:00Z", "t", "x", "modal", "jb", "", "completed_ok", ""
    )
    text = cld_mod._build_archive_text_with_appended(
        [], [c_old, c_new], month_key="2026-05"
    )
    pos_old = text.index("2026-05-01")
    pos_new = text.index("2026-05-05")
    assert pos_new < pos_old


# ── _prune end-to-end ──────────────────────────────────────────────────────


def test_prune_dry_run_makes_no_changes(cld_mod, fake_ledger, tmp_path):
    archive_dir = tmp_path / "archive"
    rc = cld_mod.main(
        [
            "prune",
            "--claims-path",
            str(fake_ledger),
            "--archive-dir",
            str(archive_dir),
            "--terminal-age-days",
            "7",
            "--now-utc",
            "2026-05-12T11:00:00Z",
            "--dry-run",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    # Live ledger unchanged.
    text_after = fake_ledger.read_text()
    assert "lane_old_terminal_a" in text_after  # FAKE_LANE_OK: claim-prune unit-test fixture
    # No archive file created.
    assert not archive_dir.exists() or not list(archive_dir.iterdir())


def test_prune_applies_archives_and_rewrites_ledger(cld_mod, fake_ledger, tmp_path):
    archive_dir = tmp_path / "archive"
    rc = cld_mod.main(
        [
            "prune",
            "--claims-path",
            str(fake_ledger),
            "--archive-dir",
            str(archive_dir),
            "--terminal-age-days",
            "7",
            "--now-utc",
            "2026-05-12T11:00:00Z",
            "--format",
            "json",
        ]
    )
    assert rc == 0
    live_text = fake_ledger.read_text()
    # Old terminal rows are pruned out of the live ledger.
    assert "lane_old_terminal_a" not in live_text  # FAKE_LANE_OK: claim-prune unit-test fixture
    assert "lane_old_terminal_b" not in live_text  # FAKE_LANE_OK: claim-prune unit-test fixture
    # Active and recent terminal kept.
    assert "lane_active_a" in live_text  # FAKE_LANE_OK: claim-prune unit-test fixture
    assert "lane_recent_terminal" in live_text  # FAKE_LANE_OK: claim-prune unit-test fixture
    # April rows landed in 2026-04 archive; April-30 also goes there.
    arch_apr = archive_dir / "dispatch_claims_2026-04.md"
    assert arch_apr.is_file()
    arch_text = arch_apr.read_text()
    assert "lane_old_terminal_a" in arch_text  # FAKE_LANE_OK: claim-prune unit-test fixture
    assert "lane_old_terminal_b" in arch_text  # FAKE_LANE_OK: claim-prune unit-test fixture


def test_prune_idempotent(cld_mod, fake_ledger, tmp_path):
    archive_dir = tmp_path / "archive"
    args_base = [
        "prune",
        "--claims-path",
        str(fake_ledger),
        "--archive-dir",
        str(archive_dir),
        "--terminal-age-days",
        "7",
        "--now-utc",
        "2026-05-12T11:00:00Z",
        "--format",
        "json",
    ]
    cld_mod.main(args_base)
    text_after_first = fake_ledger.read_text()
    cld_mod.main(args_base)
    text_after_second = fake_ledger.read_text()
    assert text_after_first == text_after_second
    # The archive file should not gain duplicate rows.
    arch_apr = archive_dir / "dispatch_claims_2026-04.md"
    if arch_apr.exists():
        count_a = arch_apr.read_text().count("lane_old_terminal_a")  # FAKE_LANE_OK: claim-prune unit-test fixture
        assert count_a == 2  # original 2 rows, no duplicates


def test_prune_archive_append_only(cld_mod, fake_ledger, tmp_path):
    """A second prune call MUST preserve rows from the first prune."""

    archive_dir = tmp_path / "archive"
    # First prune
    cld_mod.main(
        [
            "prune",
            "--claims-path",
            str(fake_ledger),
            "--archive-dir",
            str(archive_dir),
            "--terminal-age-days",
            "7",
            "--now-utc",
            "2026-05-12T11:00:00Z",
            "--format",
            "text",
        ]
    )
    arch = archive_dir / "dispatch_claims_2026-04.md"
    first_text = arch.read_text()
    # Inject another old-terminal row into the live ledger and re-prune.
    ledger_text = fake_ledger.read_text()
    new_row = (
        "| 2026-04-10T10:00:00Z | t | lane_new_old | modal | job_new_old "  # FAKE_LANE_OK: claim-prune unit-test fixture
        "|  | completed_ok | another old |\n"
    )
    # Insert after the table separator.
    ledger_lines = ledger_text.splitlines(keepends=True)
    for idx, line in enumerate(ledger_lines):
        if line.startswith("|---|"):
            ledger_lines.insert(idx + 1, new_row)
            break
    fake_ledger.write_text("".join(ledger_lines))
    # Second prune
    cld_mod.main(
        [
            "prune",
            "--claims-path",
            str(fake_ledger),
            "--archive-dir",
            str(archive_dir),
            "--terminal-age-days",
            "7",
            "--now-utc",
            "2026-05-12T11:00:00Z",
            "--format",
            "text",
        ]
    )
    second_text = arch.read_text()
    # All rows from first prune must still be present.
    for lane in ("lane_old_terminal_a", "lane_old_terminal_b"):  # FAKE_LANE_OK: claim-prune unit-test fixture
        assert lane in second_text, f"{lane} missing after second prune"
    # New row landed.
    assert "lane_new_old" in second_text  # FAKE_LANE_OK: claim-prune unit-test fixture


# ── summary archive scanning ───────────────────────────────────────────────


def test_summary_includes_archive_by_default(cld_mod, fake_ledger, tmp_path, capsys):
    archive_dir = tmp_path / "archive"
    # First, prune so terminals are moved.
    cld_mod.main(
        [
            "prune",
            "--claims-path",
            str(fake_ledger),
            "--archive-dir",
            str(archive_dir),
            "--terminal-age-days",
            "7",
            "--now-utc",
            "2026-05-12T11:00:00Z",
            "--format",
            "text",
        ]
    )
    capsys.readouterr()
    cld_mod.main(
        [
            "summary",
            "--claims-path",
            str(fake_ledger),
            "--archive-dir",
            str(archive_dir),
            "--now-utc",
            "2026-05-12T11:00:00Z",
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    # Pruned rows are visible in summary because archive is included.
    lane_ids = {row["lane_id"] for row in summary["terminal_latest"]}
    assert "lane_old_terminal_a" in lane_ids  # FAKE_LANE_OK: claim-prune unit-test fixture
    assert "lane_old_terminal_b" in lane_ids  # FAKE_LANE_OK: claim-prune unit-test fixture


def test_summary_live_only_excludes_archive(cld_mod, fake_ledger, tmp_path, capsys):
    archive_dir = tmp_path / "archive"
    cld_mod.main(
        [
            "prune",
            "--claims-path",
            str(fake_ledger),
            "--archive-dir",
            str(archive_dir),
            "--terminal-age-days",
            "7",
            "--now-utc",
            "2026-05-12T11:00:00Z",
            "--format",
            "text",
        ]
    )
    capsys.readouterr()
    cld_mod.main(
        [
            "summary",
            "--claims-path",
            str(fake_ledger),
            "--archive-dir",
            str(archive_dir),
            "--live-only",
            "--now-utc",
            "2026-05-12T11:00:00Z",
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    lane_ids = {row["lane_id"] for row in summary["terminal_latest"]}
    assert "lane_old_terminal_a" not in lane_ids  # FAKE_LANE_OK: claim-prune unit-test fixture
    assert "lane_old_terminal_b" not in lane_ids  # FAKE_LANE_OK: claim-prune unit-test fixture
