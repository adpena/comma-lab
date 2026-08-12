# SPDX-License-Identifier: MIT
"""Tests for the codex arm queue + saturation actuator.

This tool SPAWNS PROCESSES, so its refusal paths matter more than its happy path:
the cap, the one-scorer-owner rule, the missing-prompt refusal, and the kill switch.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from tac.subagent_contract import RETAINED_REASONING

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


def test_keeper_carries_the_ssd_add_dir(q):
    """The flag whose absence killed fz3 — pinned so it cannot silently vanish.

    Upgraded 2026-08-12 for the multi-tier grant (SSD_ADD_DIR -> SSD_ADD_DIRS):
    EVERY granted tier must ride the keeper, not just one — arms measurably
    write both (sr2 coldstore + pz4a retained payloads live on tier-2)."""
    src = q.keeper_source("x", ".omx/tmp/codex_runs/x_prompt.md")
    assert "--add-dir" in src
    assert q.SSD_ADD_DIRS, "the grant list must never be empty"
    for tier in q.SSD_ADD_DIRS:
        assert tier in src, f"granted tier missing from keeper: {tier}"


def test_spawn_command_detaches_via_setsid_not_merely_disown(q):
    """disown clears the JOB TABLE only; the child is reparented to PID 1 when the
    tool-shell exits, which is one of the launchd reaper's orphan signals. setsid
    detachment (macOS has no setsid(1) — Python call) remains required so the
    keeper survives the harness's own lifetime."""
    cmd = q.spawn_command("x", ".omx/tmp/codex_runs/x_prompt.md")
    assert "os.setsid()" in cmd
    assert "os.fork()" in cmd
    assert "disown" not in cmd  # the mechanism that failed must not creep back


def test_spawn_command_has_no_reaper_matchable_name(q):
    """THE root-cause pin (2026-08-04): com.vertigo.claude-code-reaper (launchd,
    every 60s) SIGTERMs any \\b(claude|codex)\\b process with no TTY and
    (PPID==1 or dead stdin) older than 300s — receipts: signal=TERM at
    elapsed 335/337/337s; a plain-bash control detached identically SURVIVED.
    The spawn command's ps-visible line must therefore contain NO standalone
    codex/claude word (``codex_runs`` is safe: underscore = word char)."""
    import re

    cmd = q.spawn_command("x", ".omx/tmp/codex_runs/x_prompt.md")
    assert re.search(r"\b(claude|codex)\b", cmd) is None
    assert "_keeper.py" in cmd


def test_keeper_runs_codex_as_child_with_regular_file_stdin(q):
    """The other two reaper conjuncts, broken in the keeper: codex is a normal
    CHILD (PPID != 1 — Popen, not detached) and its stdin is a REGULAR FILE,
    which stdin_is_dead() (grep '/dev/null|PIPE|FIFO') reads as alive."""
    src = q.keeper_source("x", ".omx/tmp/codex_runs/x_prompt.md")
    assert "subprocess.Popen" in src
    assert ".stdin" in src and "stdin=stdin_f" in src
    assert "os.devnull" not in src  # devnull stdin would trip stdin_is_dead()


def test_keeper_names_the_common_contract(q):
    assert "_common_contract.md" in q.keeper_source("x", "p.md")


def test_direct_entrypoint_bootstraps_repo_src(q):
    assert q._SRC == _REPO / "src"
    assert str(q._SRC) in sys.path


def test_keeper_composes_the_retained_reasoning_contract(q):
    src = q.keeper_source("x", "p.md")
    assert src.count(RETAINED_REASONING) == 2  # initial generation and relay
    assert src.count("## NEXT_IF_RESUMED") == 2


def test_spawn_writes_the_keeper_file(q, tmp_path, monkeypatch):
    monkeypatch.setattr(q, "_REPO", tmp_path)
    monkeypatch.setattr(q, "RUNS", tmp_path / ".omx" / "tmp" / "codex_runs")
    # append_row's default path binds the REAL queue at def time — stub it so a
    # unit test can never pollute live state (it did, once).
    monkeypatch.setattr(q, "append_row", lambda *a, **k: None)
    monkeypatch.setattr(q, "SPAWN_LOG", tmp_path / "spawn.jsonl")
    calls = []
    monkeypatch.setattr(q.subprocess, "run", lambda *a, **k: calls.append(a))
    prompt = tmp_path / ".omx" / "tmp" / "codex_runs" / "x_prompt.md"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("charter")
    assert q.spawn("x", ".omx/tmp/codex_runs/x_prompt.md") is True
    keeper = tmp_path / ".omx" / "tmp" / "codex_runs" / "x_keeper.py"
    assert keeper.exists() and "codex" in keeper.read_text()
    assert calls  # the detached spawn was actually invoked


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


# --- the exit receipt (2026-08-04 round 2; keeper-owned since the reaper fix) -----


def test_keeper_writes_an_exit_receipt(q):
    """`.last.txt` presence proves a clean finish; its ABSENCE proves nothing.
    Without an rc/signal receipt, death and completion are indistinguishable —
    which is how two rounds of guessing happened. The receipt identified the
    reaper on its FIRST firing (signal=TERM at 335-337s)."""
    src = q.keeper_source("x", ".omx/tmp/codex_runs/x_prompt.md")
    assert ".done" in src
    assert "rc=%d" in src
    assert "persist-final" in src
    assert "--last" in src


def test_keeper_handles_signals_so_a_reap_leaves_evidence(q):
    """Python handlers interrupt proc.wait() immediately — no bash
    foreground-trap-deferral class (round-2 positive control: the fg form
    wrote no receipt in exactly the reap case)."""
    src = q.keeper_source("x", ".omx/tmp/codex_runs/x_prompt.md")
    assert "signal.signal" in src
    for sig in ("TERM", "INT", "HUP", "QUIT"):
        assert sig in src
    assert "proc.wait()" in src
    assert "signal=%s" in src


def test_spawn_clears_stale_receipts_from_a_previous_generation(tmp_path, monkeypatch):
    """Review finding (executed control 2026-08-04): a leftover `.done` death
    receipt or `.last.txt` clean-finish marker from a prior generation
    survived respawn, so the next death-vs-completion read would consume
    GHOST evidence — the exact ambiguity the receipt instrument exists to
    remove. spawn() must unlink both before launching the keeper."""
    q = _load()
    monkeypatch.setattr(q, "_REPO", tmp_path)
    monkeypatch.setattr(q, "RUNS", tmp_path / ".omx/tmp/codex_runs")
    monkeypatch.setattr(q, "SPAWN_LOG", tmp_path / "spawn.jsonl")
    monkeypatch.setattr(q, "append_row", lambda *a, **k: None)
    monkeypatch.setattr(q.subprocess, "run", lambda *a, **k: None)
    q.RUNS.mkdir(parents=True)
    (q.RUNS / "x_prompt.md").write_text("charter")
    (q.RUNS / "x.done").write_text("signal=TERM elapsed=335")
    (q.RUNS / "x.last.txt").write_text("stale final message")
    assert q.spawn("x", ".omx/tmp/codex_runs/x_prompt.md") is True
    assert not (q.RUNS / "x.done").exists(), "stale death receipt survived respawn"
    assert not (q.RUNS / "x.last.txt").exists(), "stale finish marker survived respawn"


def test_next_if_resumed_parser_accepts_contract_and_title_case(q):
    text = """# Receipt

## NEXT_IF_RESUMED

1. First action.

## Boundary

Measured nothing.

## Next If Resumed

1. Title-case receipt block.
2. Still in the block.

## Later

Done.
"""
    blocks = q.next_if_resumed_blocks(text)
    assert [b["heading"] for b in blocks] == ["NEXT_IF_RESUMED", "Next If Resumed"]
    assert "Boundary" not in blocks[0]["text"]
    assert "Still in the block" in blocks[1]["text"]
    assert "Later" not in blocks[1]["text"]


def test_next_if_resumed_parser_ignores_incidental_title_mentions(q):
    text = """# NP1 Receipt - Arm Final-Message Persistence And NEXT_IF_RESUMED Surface

## Answer First

This describes the surface, not a resumable plan block.

## NEXT_IF_RESUMED

1. Resume here.
"""
    blocks = q.next_if_resumed_blocks(text)
    assert len(blocks) == 1
    assert blocks[0]["line_start"] == 7
    assert "Answer First" not in blocks[0]["text"]


def test_generated_contract_catches_a_block_the_old_prompt_missed(q):
    old_prompt_final = """# Final

Remaining work: route the measured row after the receiver is selected.
"""
    assert q.next_if_resumed_blocks(old_prompt_final) == []

    new_contract_final = """# Final

## NEXT_IF_RESUMED

- QUEUED-WITH-FIRE-ORDER; owner=receiver-owner; consumer_store=task-ledger;
  trigger=after receiver selection: route the measured row.
"""
    assert "## NEXT_IF_RESUMED" in q.keeper_source("x", "p.md")
    blocks = q.next_if_resumed_blocks(new_contract_final)
    assert len(blocks) == 1
    assert "owner=receiver-owner" in blocks[0]["text"]


def test_generated_contract_negative_control_omits_non_followon_prose(q):
    non_followon = """# Final

The NEXT_IF_RESUMED extractor was reviewed. All requested work is complete.
"""
    assert q.next_if_resumed_blocks(non_followon) == []


def test_extract_next_if_resumed_is_idempotent_and_has_no_phantom(q, tmp_path):
    out = tmp_path / "next.jsonl"
    receipt = tmp_path / "ddm_au1_20260805" / "AU1_RECEIPT.md"
    receipt.parent.mkdir()
    receipt.write_text(
        "# AU1\n\n## NEXT_IF_RESUMED\n\n1. Triage the candidates.\n",
        encoding="utf-8",
    )
    no_block = tmp_path / "ddm_none_20260805" / "NO_BLOCK_RECEIPT.md"
    no_block.parent.mkdir()
    no_block.write_text("# Receipt\n\n## Boundary\n\nNo continuation block.\n", encoding="utf-8")

    first = q.extract_next_if_resumed([receipt, no_block], provenance="positive-control", out_path=out)
    second = q.extract_next_if_resumed([receipt, no_block], provenance="positive-control", out_path=out)

    assert first == {"sources": 2, "blocks_seen": 1, "written": 1, "files_with_rows": 1}
    assert second == {"sources": 2, "blocks_seen": 1, "written": 0, "files_with_rows": 1}
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["name"] == "au1"
    assert rows[0]["provenance"] == "positive-control"
    assert rows[0]["source_kind"] == "arm_receipt"
    assert "Triage the candidates" in rows[0]["text"]


def test_infer_arm_name_from_standalone_ddm_memo(q, tmp_path):
    path = tmp_path / "ddm_dw1_qa75_distill_window_20260730.md"
    path.write_text("# memo\n", encoding="utf-8")
    assert q._infer_arm_name(path) == "dw1"


def test_persist_final_message_copies_full_text_indexes_and_extracts_next(q, tmp_path, monkeypatch):
    monkeypatch.setattr(q, "FINAL_MESSAGES", tmp_path / "arm_final_messages")
    monkeypatch.setattr(q, "FINAL_MESSAGE_INDEX", tmp_path / "final_messages.jsonl")
    monkeypatch.setattr(q, "NEXT_IF_RESUMED", tmp_path / "next_if_resumed.jsonl")
    final_text = "# Final\n\nFull message body.\n\n## NEXT_IF_RESUMED\n\n1. Resume here.\n"
    last = tmp_path / "x.last.txt"
    last.write_text(final_text, encoding="utf-8")

    row = q.persist_final_message("x", 0, 12, last)

    assert row is not None
    persisted = Path(row["path"])
    assert persisted.read_text(encoding="utf-8") == final_text
    index_rows = [
        json.loads(line)
        for line in (tmp_path / "final_messages.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(index_rows) == 1
    assert {k: index_rows[0][k] for k in ("name", "rc", "elapsed", "path", "sha256")} == {
        "name": "x",
        "rc": 0,
        "elapsed": 12,
        "path": row["path"],
        "sha256": row["sha256"],
    }
    next_rows = [
        json.loads(line)
        for line in (tmp_path / "next_if_resumed.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(next_rows) == 1
    assert next_rows[0]["name"] == "x"
    assert next_rows[0]["provenance"] == "harvested-final"
    assert next_rows[0]["source_kind"] == "persisted_final_message"
    assert "Resume here" in next_rows[0]["text"]


# --- fmtools charter-lint advisories (warn-only, never a gate) -------------------


def test_lint_charter_fm_advisories_absent_is_silent(q, tmp_path, monkeypatch):
    prompt = tmp_path / "x_prompt.md"
    prompt.write_text("Implement a mechanism.\n", encoding="utf-8")
    monkeypatch.setattr(q, "_fm_advisory_module", lambda: None)
    assert q.lint_charter_fm_advisories(str(prompt)) == []


def test_fm_advisory_warns_but_strict_mode_still_queues(q, tmp_path, monkeypatch, capsys):
    class FakeFM:
        @staticmethod
        def charter_class(_text, timeout=15):
            return {"charter_class": "build_race_train_measure", "rationale": "build measure"}

        @staticmethod
        def mechanism_reduction_language(_text, timeout=15):
            return {"flags": ["quick-train"]}

    prompt = tmp_path / "x_prompt.md"
    prompt.write_text(
        "Implement and measure the surface.\n\n"
        "## OPTIMAL FORM\n\n"
        "REFERENCE: source package.\n"
        "sha abcdef1234567890\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(q, "_fm_advisory_module", lambda: FakeFM)
    monkeypatch.setattr(q, "append_row", lambda *_a, **_k: None)
    monkeypatch.setenv("TAC_CHARTER_LINT_STRICT", "1")

    rc = q.cmd_add(Namespace(
        name="x",
        prompt=str(prompt),
        rank=10,
        owns_scorer=False,
        note="",
    ))

    out = capsys.readouterr().out
    assert rc == 0
    assert "charter-lint WARN [x]: fmtools advisory charter_class=build_race_train_measure" in out
    assert "mechanism_reduction_language=quick-train" in out
    assert "REFUSED" not in out
    assert "queued x" in out


def test_fm_advisory_does_not_rescue_deterministic_strict_refusal(
    q, tmp_path, monkeypatch, capsys
):
    class FakeFM:
        @staticmethod
        def charter_class(_text, timeout=15):
            return {"charter_class": "audit_analysis", "rationale": "audit"}

        @staticmethod
        def mechanism_reduction_language(_text, timeout=15):
            return {"flags": []}

    prompt = tmp_path / "x_prompt.md"
    prompt.write_text("Implement the thing without an optimal-form block.\n", encoding="utf-8")
    monkeypatch.setattr(q, "_fm_advisory_module", lambda: FakeFM)
    monkeypatch.setattr(q, "append_row", lambda *_a, **_k: None)
    monkeypatch.setenv("TAC_CHARTER_LINT_STRICT", "1")

    rc = q.cmd_add(Namespace(
        name="x",
        prompt=str(prompt),
        rank=10,
        owns_scorer=False,
        note="",
    ))

    out = capsys.readouterr().out
    assert rc == 3
    assert "charter-lint REFUSED [x]:" in out
    assert "charter-lint WARN [x]: fmtools advisory charter_class=audit_analysis" in out
    assert "queued x" not in out
