"""Regression test for the codex_delegate transient-death auto-recovery (apparatus fix
for the codex_probe_token_limit_death / "Selected model is at capacity" bug class).

The generated launcher MUST re-run on a transient (capacity/rate-limit/disconnect) death
so the agent self-resumes from its subagent_checkpoint instead of orphaning work.
"""
from __future__ import annotations

import re

from tools import codex_delegate


def _gen_launcher(tmp_path):
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


def test_launcher_has_bounded_retry_loop(tmp_path):
    body = _gen_launcher(tmp_path)
    # A while-loop bounded by the retry cap, gated on RC != 0 AND the transient signature.
    assert 'while [ "$RC" -ne 0 ]' in body
    assert f'-lt {codex_delegate._MAX_CAPACITY_RETRIES}' in body
    assert "grep -qiE" in body
    # It re-runs codex exec (self-resume) — codex is invoked at least twice in the script.
    assert body.count("codex exec") >= 2
    assert "sleep $backoff" in body


def test_transient_signature_matches_the_real_capacity_error():
    # The exact death string observed in the wild MUST match the retry signature.
    sig = codex_delegate._TRANSIENT_DEATH_SIGNATURE
    observed = "ERROR: Selected model is at capacity. Please try a different model."
    assert re.search(sig, observed, re.IGNORECASE) is not None
    # And it must NOT match a genuine fatal error (bad flag) — those should die, not loop.
    fatal = "error: unexpected argument '--rmote' found"
    assert re.search(sig, fatal, re.IGNORECASE) is None


def test_retry_constants_are_sane():
    assert codex_delegate._MAX_CAPACITY_RETRIES >= 3
    assert codex_delegate._CAPACITY_BACKOFF_STEP_SECONDS >= 5
