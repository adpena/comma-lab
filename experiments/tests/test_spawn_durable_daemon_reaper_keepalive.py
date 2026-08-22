"""Reaper-keepalive wrap: the fleet reaper's own intentional-daemon marker.

MEASURED 2026-08-22: three consecutive jo1 r9 daemon deaths at ~5 min, no exit
receipt, 95% memory free — the launchd reaper matching `\b(claude|codex)\b`
against argv that embedded a claude-bearing PATH. See _reaper_keepalive_wrap.
"""

from __future__ import annotations

from tools.spawn_durable_daemon import REAPER_KEEPALIVE_TOKEN, _reaper_keepalive_wrap


def test_marker_lands_in_argv_not_only_env() -> None:
    # POSITIVE: the token must be visible to `ps`, i.e. present in argv itself.
    out = _reaper_keepalive_wrap(["/bin/echo", "hi"])
    assert REAPER_KEEPALIVE_TOKEN in " ".join(out)
    assert out[0] == "/usr/bin/env"
    assert out[-2:] == ["/bin/echo", "hi"]


def test_idempotent_when_caller_already_marked() -> None:
    # NEGATIVE direction: no double-wrap when the caller supplied the marker.
    already = ["/usr/bin/env", f"{REAPER_KEEPALIVE_TOKEN}=1", "/bin/echo", "hi"]
    assert _reaper_keepalive_wrap(already) == already


def test_marker_survives_a_path_bearing_env_prefix() -> None:
    # The exact live shape that died: env PATH=<...claude...> <cmd>.
    cmd = ["env", "PATH=/x/tools/host_shims:/Users/u/.claude/plugins/bin", "/bin/true"]
    out = _reaper_keepalive_wrap(cmd)
    joined = " ".join(out)
    assert REAPER_KEEPALIVE_TOKEN in joined
    assert ".claude" in joined  # the reaper-matching substring is still present…
    assert out.index(f"{REAPER_KEEPALIVE_TOKEN}=1") < out.index(cmd[1])  # …and now excluded


def test_reaper_exclusion_regex_actually_drops_the_wrapped_line() -> None:
    # Reproduce the reaper's exclusion predicate against a realistic ps line.
    import re

    exclusion = re.compile(r"codex_runs/|REAPER_KEEPALIVE|/Applications/[^ ]*\.app/")
    argv = _reaper_keepalive_wrap(
        ["env", "PATH=/Users/u/.claude/plugins/bin", ".venv/bin/python", "-m", "trainer"]
    )
    ps_line = f"76768 ?? 0:01.23 {' '.join(argv)}"
    assert exclusion.search(ps_line), "wrapped daemon must be excluded from the reaper snapshot"
    # CONTROL: the same line WITHOUT the wrap is NOT excluded (the death shape).
    bare = "76768 ?? 0:01.23 env PATH=/Users/u/.claude/plugins/bin .venv/bin/python -m trainer"
    assert not exclusion.search(bare)
    assert re.search(r"\b(claude|codex)\b", bare), "bare line matches the reaper's kill pattern"
