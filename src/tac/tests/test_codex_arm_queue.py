# SPDX-License-Identifier: MIT
"""Tests for the codex arm queue + saturation actuator.

This tool SPAWNS PROCESSES, so its refusal paths matter more than its happy path:
the cap, the one-scorer-owner rule, the missing-prompt refusal, and the kill switch.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
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


def test_add_refuses_inline_charter_without_raw_oserror_or_traceback():
    inline_charter = "inline-charter-" + ("x" * 5000)
    proc = subprocess.run(
        [
            sys.executable,
            str(_TOOL),
            "add",
            "--name",
            "inline-must-refuse",
            "--prompt",
            inline_charter,
        ],
        cwd=_REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    output = proc.stdout + proc.stderr
    assert proc.returncode != 0
    assert "charters must be files" in output
    assert "file path" in output
    assert "Traceback" not in output
    assert "OSError" not in output


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

    assert first == {
        "sources": 2,
        "blocks_seen": 1,
        "written": 1,
        "files_with_rows": 1,
        "auto_retracted": 0,
    }
    assert second == {
        "sources": 2,
        "blocks_seen": 1,
        "written": 0,
        "files_with_rows": 1,
        "auto_retracted": 0,
    }
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
        "sha abcdef1234567890\n"
        # Pre-existing RED repaired 2026-08-16: the negatives-accounting leg
        # landed in lint_charter_optimal_form on 08-15 and this fixture was
        # never updated, so cmd_add returned 3 at HEAD. The test's SUBJECT is
        # the FM-advisory path (advisory warns, strict still queues), so the
        # fixture conforms rather than the lint relaxing.
        "Prior negative: the parked carrier family.\n",
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


# ---------------------------------------------------------------------------
# Charter-time RECALL/VALIDATION advisories (operator 2026-08-16 correction).
# Both-direction controls: the lint must FIRE on the measured 08-16 defect
# shapes AND go SILENT on a conforming charter. A detector that only fires
# positive is a rubber stamp.
# ---------------------------------------------------------------------------


def _recall_prompt(tmp_path, body: str):
    prompt = tmp_path / "recall_prompt.md"
    prompt.write_text(body, encoding="utf-8")
    return str(prompt)


def test_recall_lint_silent_on_conforming_charter(q, tmp_path, monkeypatch):
    """NEGATIVE CONTROL: charter-dependent legs emit no stale-premise warning."""
    # Index-health advisories are corpus state, not a property of this fixture.
    monkeypatch.setattr(q, "_lint_corrections_index_freshness", lambda: [])
    path = _recall_prompt(
        tmp_path,
        "Build the carrier refit per ddm_rfo2_route_20260815.md.\n"
        "Baseline re-derived at run time from the live pointer file.\n",
    )
    assert q.lint_charter_recall_advisories(path) == []


def test_recall_lint_flags_bare_task_ids_without_memo_filename(q, tmp_path):
    """Three arms reported this on 2026-08-16: arms cannot resolve #NNNN."""
    path = _recall_prompt(tmp_path, "Execute #1074 and #1038 on the live base.\n")
    out = q.lint_charter_recall_advisories(path)
    assert any("bare task ids" in w for w in out)


def test_recall_lint_accepts_task_ids_when_a_memo_is_cited(q, tmp_path):
    """A task id is fine WITH a resolvable memo filename beside it."""
    path = _recall_prompt(
        tmp_path,
        "Execute #1074 per ddm_td1_token_drop_schur_arithmetic_20260816.md.\n",
    )
    out = q.lint_charter_recall_advisories(path)
    assert not any("bare task ids" in w for w in out)


def test_recall_lint_does_not_let_unrelated_memo_launder_bare_task_id(q, tmp_path):
    """rv15 F11: anchoring one line cannot waive a different bare-id line."""
    path = _recall_prompt(
        tmp_path,
        "Read ddm_td1_token_drop_schur_arithmetic_20260816.md.\n"
        "Then execute #1162 by its harness label.\n",
    )
    out = q.lint_charter_recall_advisories(path)
    assert any("#1162" in w and "bare task ids" in w for w in out)


def test_recall_lint_does_not_let_same_line_unrelated_memo_launder_id(q, tmp_path):
    """rv16 red control: physical-line co-location is not content ownership."""
    path = _recall_prompt(
        tmp_path,
        "Read unrelated_review.md and execute #1162 by its harness label.\n",
    )
    out = q.lint_charter_recall_advisories(path)
    assert any("#1162" in w and "bare task ids" in w for w in out)


def test_recall_lint_adjudicates_each_task_id_line_independently(q, tmp_path):
    path = _recall_prompt(
        tmp_path,
        "Execute #1074 per ddm_td1_token_drop_schur_arithmetic_20260816.md.\n"
        "Execute #1163 with no repo citation.\n",
    )
    out = q.lint_charter_recall_advisories(path)
    assert any("#1163" in w for w in out)
    assert all("#1074" not in w for w in out if "bare task ids" in w)


def test_recall_lint_flags_stale_frontier_literal(q, tmp_path, monkeypatch):
    """A score-shaped literal absent from the live pointer is the pv1 class."""
    pointer = tmp_path / "pointer.json"
    pointer.write_text(
        '{"effective_frontier": {"score": 0.15959729295498598}}', encoding="utf-8"
    )
    monkeypatch.setattr(q, "FRONTIER_POINTER", pointer)
    path = _recall_prompt(tmp_path, "Frontier: S 0.1600920261571558 on the live base.\n")
    out = q.lint_charter_recall_advisories(path)
    assert any("match NO anchor in the live" in w for w in out)


def test_recall_lint_silent_when_frontier_literal_is_current(q, tmp_path, monkeypatch):
    """NEGATIVE CONTROL for the same leg — the live value must not fire."""
    pointer = tmp_path / "pointer.json"
    pointer.write_text(
        '{"effective_frontier": {"score": 0.15959729295498598}}', encoding="utf-8"
    )
    monkeypatch.setattr(q, "FRONTIER_POINTER", pointer)
    path = _recall_prompt(tmp_path, "Frontier: S 0.15959729295498598 on the live base.\n")
    out = q.lint_charter_recall_advisories(path)
    assert not any("match NO anchor" in w for w in out)


def test_recall_lint_flags_refuted_numeric_from_corrections_index(q, tmp_path, monkeypatch):
    """The gx1/ra1 class: a charter quoting a value the corpus corrected.

    ADJUDICATED 2026-08-17 (#1085), fixture corrected rather than deleted.  This
    test passed against a fixture CLEANER than the real store: it named a single
    unambiguous correction, while the live index pairs adjacent numbers inside a
    marker-word window and records no quantity at all.  So it proved the leg
    works on a store we do not have.  The fixture now carries the `quantity`
    field the leg requires, which is what the test's own intent implied -- it
    wants the rate bar IDENTIFIED, not merely digit-matched.  Its sister,
    `test_stale_numbers_leg_is_silent_when_store_cannot_identify_quantities`,
    pins the live-store behaviour so the silence cannot go unnoticed.
    """
    index = tmp_path / "corrections.jsonl"
    index.write_text(
        json.dumps(
            {
                "quantity": "live rate bar",
                "refuted_value": "15157",
                "corrected_value": "14414",
                "source": ".omx/research/ddm_ra1_carrier_rank_refit_preproof_20260816.md",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(q, "CORRECTIONS_INDEX", index)
    path = _recall_prompt(tmp_path, "The rate rung is 15,157 B on the live base.\n")
    out = q.lint_charter_recall_advisories(path)
    assert any("REFUTED value" in w and "14414" in w for w in out)


def test_recall_lint_waiver_silences_all_legs(q, tmp_path):
    """RECALL_LINT_NA is the tracked escape, matching the sibling lints."""
    path = _recall_prompt(
        tmp_path,
        "RECALL_LINT_NA: pure-apparatus landing with no research premises.\n"
        "Execute #1074 and #1038.\n",
    )
    assert q.lint_charter_recall_advisories(path) == []


def test_recall_lint_never_raises_on_unreadable_stores(q, tmp_path, monkeypatch):
    """Advisory by construction: a missing store is silence, never a block."""
    monkeypatch.setattr(q, "CORRECTIONS_INDEX", tmp_path / "absent.jsonl")
    monkeypatch.setattr(q, "FRONTIER_POINTER", tmp_path / "absent.json")
    monkeypatch.setattr(q, "RESEARCH_DIR", tmp_path / "absent_dir")
    monkeypatch.setattr(q, "_lint_corrections_index_freshness", lambda: [])
    path = _recall_prompt(tmp_path, "This has never been run and is un-owned.\n")
    assert q.lint_charter_recall_advisories(path) == []


# --- the retraction channel (ddm_sc3, 2026-08-16) --------------------------------
# A retraction is the ONLY way a correction at the source reaches a reader that
# already serves the stale row. These tests are MUTATION tests on purpose: each one
# plants the retraction, proves the row disappears, removes the retraction, and
# proves the row comes back. A test that passes on the pre-fix code proves nothing.


def _plan_store(q, tmp_path, text="1. Fire when the archive is below 186,269 B.\n"):
    out = tmp_path / "next.jsonl"
    receipt = tmp_path / "ddm_rx1_rate_attack_20260814" / "RECEIPT.md"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(f"# RX1\n\n## NEXT_IF_RESUMED\n\n{text}", encoding="utf-8")
    q.extract_next_if_resumed([receipt], provenance="positive-control", out_path=out)
    rows = q.load_next_if_resumed(out)
    assert len(rows) == 1
    return out, receipt, rows[0]


def test_retraction_hides_a_superseded_row_and_removing_it_brings_the_row_back(q, tmp_path):
    out, _receipt, row = _plan_store(q, tmp_path)
    before = out.read_text(encoding="utf-8")

    assert row["retracted"] is False
    assert row["retraction_disposition"] is None

    q.retract_next_if_resumed_row(
        row["row_id"],
        reason="the admission bar 186,269 B is 3,510 B above the live 182,759 B frontier",
        citation=".omx/research/ddm_fb1_stale_bar_rebase_and_bank_union_20260816.md",
        retracted_by="ddm_sc3",
        path=out,
    )

    assert q.load_next_if_resumed(out) == []  # MUTATION: the stale row is gone
    with_flag = q.load_next_if_resumed(out, include_superseded=True)
    assert len(with_flag) == 1
    assert with_flag[0]["retraction_disposition"] == q.RETRACTION_SUPERSEDED
    assert "182,759" in with_flag[0]["retractions"][0]["reason"]

    out.write_text(before, encoding="utf-8")  # CONTROL: remove only the retraction
    assert len(q.load_next_if_resumed(out)) == 1


def test_amend_required_row_stays_visible_but_carries_its_notice(q, tmp_path):
    out, _receipt, row = _plan_store(q, tmp_path)
    q.retract_next_if_resumed_row(
        row["row_id"],
        reason="the third clause quotes a 15,157 B cut computed off a superseded base",
        citation=".omx/research/ddm_fb1_stale_bar_rebase_and_bank_union_20260816.md",
        retracted_by="ddm_sc3",
        disposition=q.RETRACTION_AMEND_REQUIRED,
        path=out,
    )
    rows = q.load_next_if_resumed(out)
    assert len(rows) == 1, "AMEND_REQUIRED must not hide live follow-ons"
    assert rows[0]["retraction_disposition"] == q.RETRACTION_AMEND_REQUIRED
    assert rows[0]["retractions"][0]["disposition"] == q.RETRACTION_AMEND_REQUIRED


def test_retraction_never_mutates_or_deletes_the_target_row(q, tmp_path):
    out, _receipt, row = _plan_store(q, tmp_path)
    stored_before = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    q.retract_next_if_resumed_row(
        row["row_id"],
        reason="superseded by the live canonical frontier pointer, checked at source",
        citation=".omx/state/canonical_frontier_pointer.json",
        retracted_by="ddm_sc3",
        path=out,
    )
    stored_after = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert stored_after[: len(stored_before)] == stored_before  # append-only, byte-for-byte
    assert len(stored_after) == len(stored_before) + 1
    assert stored_after[-1]["schema"] == "codex_arm_queue.next_if_resumed.retraction.v1"


def test_retraction_refuses_unknown_target_placeholder_reason_and_bad_disposition(q, tmp_path):
    out, _receipt, row = _plan_store(q, tmp_path)
    good = dict(
        reason="the admission bar is stale against the live frontier pointer",
        citation=".omx/state/canonical_frontier_pointer.json",
        retracted_by="ddm_sc3",
        path=out,
    )
    with pytest.raises(ValueError, match="targets nothing"):
        q.retract_next_if_resumed_row("deadbeef" * 8, **good)
    for placeholder in ("<reason>", "stale", "TBD", ""):
        with pytest.raises(ValueError, match="real rationale"):
            q.retract_next_if_resumed_row(
                row["row_id"],
                reason=placeholder,
                citation=good["citation"],
                retracted_by="ddm_sc3",
                path=out,
            )
    with pytest.raises(ValueError, match="disposition"):
        q.retract_next_if_resumed_row(row["row_id"], disposition="MAYBE", **good)
    with pytest.raises(ValueError, match="citation"):
        q.retract_next_if_resumed_row(
            row["row_id"],
            reason=good["reason"],
            citation="  ",
            retracted_by="ddm_sc3",
            path=out,
        )
    assert len(q.load_next_if_resumed(out)) == 1  # every refusal left the row live


def test_correcting_the_source_auto_retracts_the_pre_correction_row(q, tmp_path):
    """THE defect: a corrected memo used to leave its stale row live beside the fix."""
    out, receipt, stale = _plan_store(q, tmp_path)
    receipt.write_text(
        "# RX1\n\n## NEXT_IF_RESUMED\n\n1. Fire when the archive is at or below 168,345 B.\n",
        encoding="utf-8",
    )
    summary = q.extract_next_if_resumed([receipt], provenance="positive-control", out_path=out)

    assert summary["written"] == 1
    assert summary["auto_retracted"] == 1
    live = q.load_next_if_resumed(out)
    assert len(live) == 1, "the pre-correction row must not survive the correction"
    assert "168,345" in live[0]["text"]
    assert stale["row_id"] not in {r["row_id"] for r in live}

    everything = q.load_next_if_resumed(out, include_superseded=True)
    assert len(everything) == 2  # nothing was deleted; the stale row is retained as debt


def test_auto_retraction_does_not_fire_across_different_sources(q, tmp_path):
    out, _receipt, _row = _plan_store(q, tmp_path)
    other = tmp_path / "ddm_rx1_rate_attack_20260814" / "SECOND.md"
    other.write_text("# RX1\n\n## NEXT_IF_RESUMED\n\n1. A different plan entirely.\n", encoding="utf-8")
    summary = q.extract_next_if_resumed([other], provenance="positive-control", out_path=out)
    assert summary["auto_retracted"] == 0
    assert len(q.load_next_if_resumed(out)) == 2


def test_debt_ledger_reports_the_denominator_not_just_the_survivors(q, tmp_path):
    out, _receipt, row = _plan_store(q, tmp_path)
    q.retract_next_if_resumed_row(
        row["row_id"],
        reason="the admission bar 186,269 B sits above the live shipping archive",
        citation=".omx/research/ddm_fb1_stale_bar_rebase_and_bank_union_20260816.md",
        retracted_by="ddm_sc3",
        path=out,
    )
    debt = q.next_if_resumed_debt(out)
    assert debt["plan_rows_total"] == 1
    assert debt["plan_rows_live"] == 0
    assert debt["counts"][q.RETRACTION_SUPERSEDED] == 1
    assert debt["superseded"][0]["row_id"] == row["row_id"]


def test_legacy_rows_without_any_retraction_load_unchanged(q, tmp_path):
    """Additive: the 248 pre-channel rows must keep loading exactly as before."""
    out = tmp_path / "legacy.jsonl"
    out.write_text(
        json.dumps(
            {
                "schema": "codex_arm_queue.next_if_resumed.v1",
                "row_id": "abc123",
                "name": "au1",
                "text": "1. Resume here.",
            }
        )
        + "\n"
        + json.dumps({"schema": "some.other.surface.v1", "name": "ignored"})
        + "\n",
        encoding="utf-8",
    )
    rows = q.load_next_if_resumed(out)
    assert [r["name"] for r in rows] == ["au1"]
    assert rows[0]["retracted"] is False and rows[0]["retraction_disposition"] is None


# --- #1085: _lint_stale_numbers fails closed on a store that cannot identify quantities ---


def _stale_number_module():
    import importlib.util
    import pathlib
    import sys

    root = pathlib.Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location(
        "_caq_stale", root / "tools" / "codex_arm_queue.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_caq_stale"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_stale_numbers_leg_is_silent_when_store_cannot_identify_quantities(tmp_path):
    """The live corrections index pairs adjacent numbers in a window.

    It records no quantity name, so `3000 -> 3.37`, `3000 -> 726` and
    `3000 -> 0.05` are three unrelated quantities sharing four digits.  The leg
    must emit nothing rather than assert "recorded as a REFUTED value" from
    adjacency.  Measured 2026-08-17: 648 of 11,840 live rows are date-parse
    artifacts (a bare year paired with a month fragment).
    """

    mod = _stale_number_module()
    index = tmp_path / "no_identity.jsonl"
    index.write_text(
        json.dumps(
            {
                "refuted_value": "2026",
                "corrected_value": "05",
                "phrase": "superseded",
                "source": "some_memo_20260521.md",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    mod.CORRECTIONS_INDEX = index
    assert mod._corrections_index_identifies_quantities() is False
    assert mod._lint_stale_numbers("charter written on 2026-08-17 at line 1179") == []


def test_stale_numbers_leg_reactivates_when_the_store_names_the_quantity(tmp_path):
    """POSITIVE CONTROL: the leg is retired-pending-store-repair, not deleted.

    Rebuild the index with a quantity field and the leg must fire again on the
    real case it was built for -- the 15,157 B rate bar superseded by 14,413.4.
    Without this control the fix would be indistinguishable from deleting a
    detector because it was inconvenient.
    """

    mod = _stale_number_module()
    index = tmp_path / "with_identity.jsonl"
    index.write_text(
        json.dumps(
            {
                "quantity": "live rate bar",
                "refuted_value": "15157",
                "corrected_value": "14413.4",
                "source": ".omx/research/ddm_rfo2_rate_ladder_20260815.md",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    mod.CORRECTIONS_INDEX = index
    assert mod._corrections_index_identifies_quantities() is True
    warnings = mod._lint_stale_numbers("the bar is 15,157 B against a superseded archive")
    assert len(warnings) == 1
    assert "15157" in warnings[0]
    assert "14413.4" in warnings[0]


def test_stale_numbers_leg_tolerates_a_missing_or_unreadable_store(tmp_path):
    mod = _stale_number_module()
    mod.CORRECTIONS_INDEX = tmp_path / "absent.jsonl"
    assert mod._corrections_index_identifies_quantities() is False
    assert mod._lint_stale_numbers("15,157 B") == []


# ---------------------------------------------------------------------------
# scaffold subcommand (operator 2026-08-20 "Must self protect and fix
# permanently" — the hand-authored-charter lint-refusal class, paid on 08-17
# AND 08-20; charters are born FROM the machine that judges them)
# ---------------------------------------------------------------------------


def test_scaffold_template_carries_every_lint_required_section():
    mod = _load()
    text = mod._CHARTER_TEMPLATE.format(name="zz_t", date="20990101")
    for header in (
        "## MANDATE",
        "## SCOPE",
        "## HARD CONSTRAINTS",
        "## PRIOR NEGATIVE SIGNAL",
        "## OPTIMAL FORM",
        "## DELIVERABLE",
        "PRIOR-LAW PREDICTION",
        "FALSIFIER",
    ):
        assert header in text, f"template lost required section {header!r}"


def test_scaffold_raw_template_is_unspawnable(tmp_path):
    """The un-spawnable property: an unfilled template must FAIL the
    optimal-form lint (no sha pin can appear in placeholders), so the
    scaffold cannot be used to bypass the charter lint with empty sections."""
    mod = _load()
    charter = tmp_path / "zz_raw_20990101.md"
    charter.write_text(
        mod._CHARTER_TEMPLATE.format(name="zz_raw", date="20990101"),
        encoding="utf-8",
    )
    problems = mod.lint_charter_optimal_form(str(charter))
    assert problems, "raw scaffold template PASSED the lint — un-spawnable property broken"


def test_scaffold_refuses_overwrite(tmp_path):
    mod = _load()
    out = tmp_path / "zz_exists.md"
    out.write_text("existing charter", encoding="utf-8")
    rc = mod.cmd_scaffold(Namespace(name="zz_exists", out=str(out)))
    assert rc == 2
    assert out.read_text(encoding="utf-8") == "existing charter"


def test_scaffold_filled_template_passes_lint(tmp_path):
    """The negative direction: a genuinely filled charter must pass — the
    template must not be structurally impossible to satisfy."""
    mod = _load()
    text = mod._CHARTER_TEMPLATE.format(name="zz_fill", date="20990101")
    text = text.replace(
        "- Family exemplar: <FILL: the family's landed form — memo + commit sha. The lint REFUSES\n"
        "  this template until a real sha/commit pin appears here and the family exemplar is\n"
        "  cited with the word 'reference' or a receipt path.>",
        "- Family reference: the landed pq9 review-round form, commit fd8e6024c7.",
    )
    charter = tmp_path / "zz_fill_20990101.md"
    charter.write_text(text, encoding="utf-8")
    problems = mod.lint_charter_optimal_form(str(charter))
    assert problems == [], f"filled template still refused: {problems}"
