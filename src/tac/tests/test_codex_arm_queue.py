# SPDX-License-Identifier: MIT
"""Tests for the codex arm queue + saturation actuator.

This tool SPAWNS PROCESSES, so its refusal paths matter more than its happy path:
the cap, the one-scorer-owner rule, the missing-prompt refusal, and the kill switch.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[3]
_TOOL = _REPO / "tools" / "codex_arm_queue.py"


def _load():
    spec = importlib.util.spec_from_file_location("_codex_arm_queue", _TOOL)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def q():
    return _load()


def _row(name, rank=10, scorer=False, status="queued"):
    return {
        "name": name,
        "prompt_path": f".omx/tmp/codex_runs/{name}_prompt.md",
        "rank": rank,
        "owns_scorer": scorer,
        "status": status,
    }


# --- latest-row-wins (append-only status changes) ---------------------------------


def test_latest_row_per_name_wins(q):
    rows = [_row("a", status="queued"), _row("a", status="landed")]
    assert q.latest_by_name(rows)["a"]["status"] == "landed"


def test_partial_mark_row_does_not_drop_the_prompt_path(q):
    """The 2026-08-04 defect: `mark` appends {name,status} only, and naive
    last-row-wins made the charter unspawnable (`would spawn: fz4 ()`)."""
    rows = [_row("a"), {"name": "a", "status": "live"}]
    merged = q.latest_by_name(rows)["a"]
    assert merged["status"] == "live"
    assert merged["prompt_path"] == ".omx/tmp/codex_runs/a_prompt.md"
    assert merged["rank"] == 10


def test_merge_keeps_the_newest_value_of_a_field_set_twice(q):
    rows = [_row("a", rank=10), {"name": "a", "rank": 5}]
    assert q.latest_by_name(rows)["a"]["rank"] == 5


def test_torn_jsonl_tail_does_not_crash(q, tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text(json.dumps(_row("a")) + "\n{not json\n", encoding="utf-8")
    assert len(q.load_rows(path)) == 1  # the good row survives; the tear is skipped


def test_missing_queue_file_is_empty_not_an_error(q, tmp_path):
    assert q.load_rows(tmp_path / "absent.jsonl") == []


# --- selection + the scorer rule --------------------------------------------------


def test_picks_in_rank_order(q):
    rows = [_row("b", rank=20), _row("a", rank=10), _row("c", rank=30)]
    picks = q.next_charters(rows, live=set(), slots=2, scorer_taken=False)
    assert [p["name"] for p in picks] == ["a", "b"]


def test_respects_slot_count(q):
    rows = [_row(n, rank=i) for i, n in enumerate("abcd")]
    assert len(q.next_charters(rows, set(), 2, False)) == 2


def test_live_arms_are_not_respawned(q):
    rows = [_row("a", rank=10), _row("b", rank=20)]
    picks = q.next_charters(rows, live={"a"}, slots=4, scorer_taken=False)
    assert [p["name"] for p in picks] == ["b"]


def test_landed_and_dropped_are_not_respawned(q):
    rows = [_row("a", status="landed"), _row("b", status="dropped"), _row("c")]
    picks = q.next_charters(rows, set(), 4, False)
    assert [p["name"] for p in picks] == ["c"]


def test_scorer_owner_skipped_when_slot_taken(q):
    """One full-n600 job at a time, fleet-wide."""
    rows = [_row("s", rank=10, scorer=True), _row("free", rank=20)]
    picks = q.next_charters(rows, set(), 4, scorer_taken=True)
    assert [p["name"] for p in picks] == ["free"]


def test_only_one_scorer_owner_per_fill(q):
    rows = [_row("s1", rank=10, scorer=True), _row("s2", rank=20, scorer=True)]
    picks = q.next_charters(rows, set(), 4, scorer_taken=False)
    assert [p["name"] for p in picks] == ["s1"]


def test_scorer_owner_fires_when_slot_free(q):
    rows = [_row("s", rank=10, scorer=True)]
    assert [p["name"] for p in q.next_charters(rows, set(), 4, False)] == ["s"]


# --- spawn refusals ---------------------------------------------------------------


def test_spawn_refuses_missing_prompt(q, capsys):
    assert q.spawn("ghost", ".omx/tmp/codex_runs/definitely_absent_prompt.md") is False


def test_spawn_command_carries_the_ssd_add_dir(q):
    """The flag whose absence killed fz3 — pinned so it cannot silently vanish."""
    cmd = q.spawn_command("x", ".omx/tmp/codex_runs/x_prompt.md")
    assert f"--add-dir {q.SSD_ADD_DIR}" in cmd or f"--add-dir '{q.SSD_ADD_DIR}'" in cmd


def test_spawn_command_detaches_via_setsid_not_merely_disown(q):
    """MEASURED 2026-08-04: four `nohup ... & disown` arms were reaped together by a
    process-group signal. disown clears the JOB TABLE; only setsid(2) leaves the
    GROUP. macOS has no setsid(1), so it must be the Python call."""
    cmd = q.spawn_command("x", ".omx/tmp/codex_runs/x_prompt.md")
    assert "os.setsid()" in cmd
    assert "os.fork()" in cmd
    assert "disown" not in cmd  # the mechanism that failed must not creep back


def test_spawn_command_redirects_stdio_so_no_tty_dependency(q):
    cmd = q.spawn_command("x", ".omx/tmp/codex_runs/x_prompt.md")
    assert "os.devnull" in cmd  # stdin closed: codex must never wait on input
    assert "x.log" in cmd  # stdout/stderr land in the durable per-arm log


def test_spawn_command_names_the_common_contract(q):
    assert "_common_contract.md" in q.spawn_command("x", "p.md")


# --- live detection reads the OS, not the ledger ----------------------------------


def test_live_arm_names_returns_a_set(q):
    """The ledger can lie (fz3 died without marking itself); the process table cannot."""
    assert isinstance(q.live_arm_names(), set)


def test_parses_a_real_ps_line_into_arm_names(q):
    """THE regression: `pgrep -af` is Linux-only — macOS ignores -a and prints bare
    PIDs, so the old parser returned empty while four arms were running, and the
    only test (isinstance ... set) passed on the broken function."""
    ps_line = (
        "26949 codex exec --skip-git-repo-check -s workspace-write "
        "--add-dir /Volumes/VertigoDataTier/pact -m gpt-5.5 "
        "-o .omx/tmp/codex_runs/fz4.last.txt 'Read and execute the charter'"
    )
    assert q.parse_arm_names(ps_line) == {"fz4"}


def test_parses_multiple_arms_and_ignores_unrelated_processes(q):
    out = "\n".join(
        [
            "111 /usr/bin/python3 something_else.py",
            "222 codex exec -o .omx/tmp/codex_runs/rt1.last.txt 'go'",
            "333 codex exec -o .omx/tmp/codex_runs/wk3.last.txt 'go'",
            "444 grep codex exec",
        ]
    )
    assert q.parse_arm_names(out) == {"rt1", "wk3"}


def test_bare_pid_output_yields_no_names_not_a_crash(q):
    """What macOS `pgrep -af` actually returns — the shape that silently blinded it."""
    assert q.parse_arm_names("26949\n27282\n") == set()


# --- the queue file itself --------------------------------------------------------


def test_shipped_queue_parses_and_has_prompts(q):
    rows = q.load_rows()
    if not rows:
        pytest.skip("queue not seeded in this checkout")
    for name, row in q.latest_by_name(rows).items():
        if row.get("status") in {"queued", "live"}:
            prompt = _REPO / row.get("prompt_path", "")
            assert prompt.exists(), f"{name} points at a missing prompt: {prompt}"
