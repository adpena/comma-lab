"""Regression tests for checkpoint-custodied Codex transient retries."""
from __future__ import annotations

import re

from tools import codex_delegate, codex_retry_checkpoint


def _gen_launcher(tmp_path, monkeypatch):
    monkeypatch.setattr(codex_delegate, "RUNS", tmp_path)
    prompt = tmp_path / "p.txt"
    prompt.write_text("do a thing", encoding="utf-8")
    log = tmp_path / "run.log"
    last = tmp_path / "run.last.txt"
    done = tmp_path / "run.done"
    launcher = codex_delegate._write_launcher(
        "unit_test", "20260713T000000Z", prompt, "gpt-5.6-sol",
        "ultra", "workspace-write", log, last, done,
    )
    return launcher.read_text(encoding="utf-8")


def test_launcher_has_bounded_retry_loop(tmp_path, monkeypatch):
    body = _gen_launcher(tmp_path, monkeypatch)
    # A while-loop bounded by the retry cap, gated on RC != 0 AND the transient signature.
    assert 'while [ "$RC" -ne 0 ]' in body
    assert f'-lt {codex_delegate._MAX_CAPACITY_RETRIES}' in body
    assert "grep -qiE" in body
    # It may re-run codex only after exact checkpoint custody is proved.
    assert body.count("codex exec") >= 2
    assert "codex_retry_checkpoint.py" in body
    assert "RETRY-REFUSED-NO-CHECKPOINT" in body
    assert '--progress-file "$WORKDIR/.omx/state/subagent_progress.jsonl"' in body
    assert "2>&1 | tee" in body
    assert "tail -c 400" in body
    assert "resume.prompt" not in body  # compatibility call uses the same small fixture prompt
    assert "sleep $backoff" in body


def test_launcher_is_headless_no_window_keeper(tmp_path, monkeypatch):
    # HEADLESS invariant (2026-07-16): the launcher must EXIT cleanly on completion, never
    # `exec bash` (which held a GUI Terminal window open forever = the orphaned-window class).
    body = _gen_launcher(tmp_path, monkeypatch)
    assert "exec bash" not in body, "launcher must not exec bash (orphans a completed window)"
    assert "exit $RC" in body, "launcher must exit cleanly with codex's return code"
    # The isolation-failure branch must also exit, not hold a window.
    assert body.count("exit $RC") >= 2


def test_retry_classification_uses_only_current_attempt_output(tmp_path, monkeypatch):
    body = _gen_launcher(tmp_path, monkeypatch)
    # A transient in the initial/cumulative log must not authorize a later
    # retry after the immediately preceding attempt died with a fatal error.
    assert "ATTEMPT_LOG=" in body
    assert "grep -qiE" in body and '"$ATTEMPT_LOG"; do' in body
    assert ': > "$ATTEMPT_LOG"' in body
    assert re.search(r'tee -a .* "\$ATTEMPT_LOG"', body) is not None
    start = body.index('while [ "$RC" -ne 0 ]')
    retry_condition = body[start:body.index("; do", start)]
    assert "run.log" not in retry_condition


def test_transient_signature_matches_the_real_capacity_error():
    # The exact death string observed in the wild MUST match the retry signature.
    sig = codex_delegate._TRANSIENT_DEATH_SIGNATURE
    observed = "ERROR: Selected model is at capacity. Please try a different model."
    assert re.search(sig, observed, re.IGNORECASE) is not None
    # And it must NOT match a genuine fatal error (bad flag) — those should die, not loop.
    fatal = "error: unexpected argument '--rmote' found"
    assert re.search(sig, fatal, re.IGNORECASE) is None


def test_retry_constants_are_bounded_for_long_arms():
    assert 1 <= codex_delegate._MAX_CAPACITY_RETRIES <= 2
    assert codex_delegate._CAPACITY_BACKOFF_STEP_SECONDS >= 5


def test_retry_checkpoint_selects_latest_exact_custody_row(tmp_path):
    progress = tmp_path / "progress.jsonl"
    progress.write_text(
        "\n".join(
            [
                '{"parent_id_or_session":"other","status":"in_progress","step":99,"next_action":"wrong"}',
                '{"parent_id_or_session":"codex_delegate:arm:stamp","status":"in_progress","step":1,"next_action":"first"}',
                '{"parent_id_or_session":"codex_delegate:arm:stamp","status":"in_progress","step":2,"next_action":"continue"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    row = codex_retry_checkpoint.latest_resumable_checkpoint(
        progress, "codex_delegate:arm:stamp"
    )
    assert row and row["step"] == 2 and row["next_action"] == "continue"


def test_retry_checkpoint_refuses_missing_or_nonresumable_rows(tmp_path, capsys):
    progress = tmp_path / "progress.jsonl"
    progress.write_text(
        '{"parent_id_or_session":"codex_delegate:arm:stamp","status":"complete",'
        '"step":2,"next_action":""}\n',
        encoding="utf-8",
    )
    rc = codex_retry_checkpoint.main(
        [
            "--delegation-key",
            "codex_delegate:arm:stamp",
            "--progress-file",
            str(progress),
        ]
    )
    assert rc == codex_retry_checkpoint.REFUSED_RC
    assert "RETRY-REFUSED-NO-CHECKPOINT" in capsys.readouterr().out


def test_compact_prompts_externalize_large_authority(monkeypatch, tmp_path):
    monkeypatch.setattr(codex_delegate, "RUNS", tmp_path)
    authority = tmp_path / "wrapped.txt"
    authority.write_text("x" * 200_000, encoding="utf-8")
    entry, resume = codex_delegate._write_compact_prompts(
        label="arm",
        stamp="stamp",
        wrapped_prompt=authority,
        delegation_key="codex_delegate:arm:stamp",
    )
    assert entry.stat().st_size < 2_000
    assert resume.stat().st_size < 2_000
    assert "bytes=200000" in entry.read_text(encoding="utf-8")
    assert "--latest-only" in resume.read_text(encoding="utf-8")
    assert "32768-byte chunks" in resume.read_text(encoding="utf-8")
