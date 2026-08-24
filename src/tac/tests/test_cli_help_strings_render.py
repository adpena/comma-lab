"""Every ``add_argument(help=...)`` must survive argparse's ``%``-interpolation.

WHY THIS EXISTS (ddm_df1, #908 audit, 2026-08-03).  ``argparse`` renders help text as
``self._get_help_string(action) % params`` where ``params`` is a *dict*.  A single unescaped
``%`` in a help string therefore raises at ``format_help()`` time and ``--help`` CRASHES —
silently, because nothing calls ``--help`` in CI.

MEASURED at discovery: ``experiments/train_tr1_partition_renderer_mlx.py --help`` (the live TR1
launch path) and ``experiments/train_levelset_witness_realized_through_R_mlx.py --help`` (the
canonical capstone theta* trainer named in CLAUDE.md) BOTH raised
``TypeError: %o format: an integer is required, not dict``.  The repo-wide sweep then found the
same class at 7 further call sites across 6 more files — the 6-7x bug-class spread CLAUDE.md
predicts.  All 10 were escaped in the same commit batch; this guard makes the class extinct.

This is not cosmetic.  ``--help`` is the surface the CLAUDE.md "NEVER invent CLI flags" rule
tells every agent to consult before wiring a flag into a subprocess call.  While it crashed,
``--deterministic-r`` (the #903 cure) was invisible on the trainer that needs it.

The scan is AST-based on purpose: it needs no imports, so it costs nothing and cannot be
defeated by a heavy module that will not load in a test environment.

SCOPE, stated so a green run is not over-read: this checks *literal* help strings only.  A help
value built at runtime (f-string, variable, ``.format()``) is not statically evaluable and is
SKIPPED, not passed.  The end-to-end complement is running ``--help`` itself, which is what
found the original two crashes; the three launch-path trainers were verified that way at
landing (all rc=0).
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]

#: The dict argparse actually passes; ``%(default)s`` etc. must keep working.
ARGPARSE_PARAMS: dict[str, object] = {
    "prog": "prog",
    "default": 1,
    "choices": "choices",
    "type": "type",
    "const": 1,
    "metavar": "metavar",
    "dest": "dest",
}


def _tracked_python_files() -> list[Path]:
    out = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT, capture_output=True, text=True, check=True,
    ).stdout.split()
    return [REPO_ROOT / rel for rel in out]


def _help_string_violations(path: Path) -> list[tuple[int, str, str]]:
    """Return ``(lineno, help_text, error)`` for each literal help= that fails ``%``-render."""
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except (SyntaxError, ValueError, OSError):
        return []  # not our concern; other gates own unparseable sources
    bad: list[tuple[int, str, str]] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        for kw in node.keywords:
            if kw.arg != "help":
                continue
            try:
                value = ast.literal_eval(kw.value)
            except (ValueError, TypeError, SyntaxError):
                continue  # a computed help string; not statically checkable
            if not isinstance(value, str):
                continue
            try:
                value % ARGPARSE_PARAMS
            except (TypeError, ValueError, KeyError) as exc:
                bad.append((node.lineno, value, f"{type(exc).__name__}: {exc}"))
    return bad


def _scan_repo() -> tuple[list[str], int, int]:
    """Return ``(violations, files_declaring_add_argument, literal_help_strings)``."""
    violations: list[str] = []
    files_with_args = 0
    help_count = 0
    for path in _tracked_python_files():
        try:
            tree = ast.parse(path.read_text(errors="replace"))
        except (SyntaxError, ValueError, OSError):
            continue
        declares = False
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "add_argument"):
                declares = True
                for kw in node.keywords:
                    if kw.arg == "help":
                        try:
                            value = ast.literal_eval(kw.value)
                        except (ValueError, TypeError, SyntaxError):
                            continue
                        if isinstance(value, str):
                            help_count += 1
        if declares:
            files_with_args += 1
        for lineno, text, err in _help_string_violations(path):
            rel = path.relative_to(REPO_ROOT)
            violations.append(f"{rel}:{lineno} {err} -- help={text[:110]!r}")
    return violations, files_with_args, help_count


def test_positive_control_a_stray_percent_is_caught(tmp_path: Path) -> None:
    """The instrument must FIRE on a known-bad string, or a clean scan proves nothing."""
    src = tmp_path / "bad_cli.py"
    src.write_text(
        "import argparse\n"
        "ap = argparse.ArgumentParser()\n"
        'ap.add_argument("--x", help="covers 81.7% of it inside the window")\n'
    )
    bad = _help_string_violations(src)
    assert len(bad) == 1, f"positive control did not fire: {bad}"
    # Assert an error was RECORDED, not its exact CPython wording (which varies by version).
    assert bad[0][2], "violation recorded with an empty error message"


def test_negative_control_escaped_and_named_conversions_pass(tmp_path: Path) -> None:
    """It must NOT fire on correctly escaped `%%` or on argparse's own `%(default)s`."""
    src = tmp_path / "good_cli.py"
    src.write_text(
        "import argparse\n"
        "ap = argparse.ArgumentParser()\n"
        'ap.add_argument("--a", help="covers 81.7%% of it (default: %(default)s)")\n'
        'ap.add_argument("--b", help="no percent at all")\n'
    )
    assert _help_string_violations(src) == []


@pytest.mark.parametrize(
    "rel_path",
    [
        "experiments/train_tr1_partition_renderer_mlx.py",
        "experiments/train_levelset_witness_realized_through_R_mlx.py",
        "experiments/train_witness_realized_through_R_mlx.py",
    ],
)
def test_launch_path_trainer_help_strings_render(rel_path: str) -> None:
    """The three canonical launch-path trainers, named individually so a failure is legible."""
    path = REPO_ROOT / rel_path
    assert path.is_file(), f"canonical trainer missing: {rel_path} (scope is VACUOUS, not clean)"
    bad = _help_string_violations(path)
    assert not bad, (
        f"{rel_path}: {len(bad)} help string(s) crash `--help`. "
        "argparse renders help as `text % params`; escape a literal percent as `%%`.\n"
        + "\n".join(f"  L{ln} {err} -- {txt[:110]!r}" for ln, txt, err in bad)
    )


def test_repo_wide_no_unescaped_percent_in_help_strings() -> None:
    """Whole-repo class guard. Reports its DENOMINATOR so an empty scan cannot pass silently."""
    violations, files_with_args, help_count = _scan_repo()
    assert files_with_args > 100, (
        f"scan found only {files_with_args} files declaring add_argument -- the scan scope "
        "collapsed (VACUOUS), which is not a pass"
    )
    assert help_count > 1000, (
        f"scan found only {help_count} literal help strings -- scope collapsed (VACUOUS)"
    )
    assert not violations, (
        f"{len(violations)} help string(s) crash `--help` "
        f"(scanned {files_with_args} files / {help_count} help strings). "
        "argparse renders help as `text % params`; escape a literal percent as `%%`.\n"
        + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# ddm_nt1 (2026-08-24): the guard above is good and it DID NOT FIRE for 3 weeks.
#
# `06fa0ad37d` landed it and escaped all 10 then-live sites. `08a472aa29` the SAME DAY
# reintroduced the class on `experiments/train_tr1_partition_renderer_mlx.py` — the trainer
# this file names by hand in its own parametrize list above — and it stayed broken until
# 2026-08-24. Nothing UNCONDITIONAL ran this file at commit: `tools/preflight_hook.py`
# subset-selects its pytest step from the staged diff, and that selector links a test to a
# source file via module-reference tokens, which a test naming its targets as string-literal
# paths does not produce. The instrument existed; its trigger did not reach it.
#
# The cure is `tools/preflight_hook.py::run_argparse_help_render_scan`, a staged-diff
# BLOCKING step. These tests pin that the commit-time guard and this repo-wide guard agree,
# so the pair cannot silently diverge into two different definitions of "renders".
# ---------------------------------------------------------------------------


def _load_preflight_hook():
    import importlib.util

    hook_path = REPO_ROOT / "tools" / "preflight_hook.py"
    assert hook_path.is_file(), f"preflight hook missing: {hook_path} (scope VACUOUS, not clean)"
    spec = importlib.util.spec_from_file_location("_pact_preflight_hook_for_test", hook_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_commit_time_guard_shares_the_repo_wide_params_dict() -> None:
    """Both guards must interpolate against the SAME dict or they judge different things."""
    hook = _load_preflight_hook()
    assert hook._ARGPARSE_HELP_PARAMS == ARGPARSE_PARAMS, (
        "the commit-time hook and this repo-wide test disagree about argparse's params dict; "
        "one of them would pass a string the other rejects"
    )


def test_commit_time_guard_positive_control_fires(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A staged bad help string must BLOCK the commit (rc=1), or the wiring proves nothing."""
    hook = _load_preflight_hook()
    src = tmp_path / "bad_cli.py"
    src.write_text(
        "import argparse\n"
        "ap = argparse.ArgumentParser()\n"
        'ap.add_argument("--x", help="runs for ~100% of its epochs")\n'
    )
    assert hook._argparse_help_render_violations(src), "scan did not detect the bad string"
    monkeypatch.chdir(tmp_path)  # process-global; monkeypatch restores it even on failure
    assert hook.run_argparse_help_render_scan(["bad_cli.py"]) == 1


def test_commit_time_guard_negative_controls_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Escaped `%%`, `%(default)s`, no-percent, empty staged, and non-.py must all be rc=0."""
    hook = _load_preflight_hook()
    src = tmp_path / "good_cli.py"
    src.write_text(
        "import argparse\n"
        "ap = argparse.ArgumentParser()\n"
        'ap.add_argument("--a", default=3, help="~100%% of epochs (default: %(default)s)")\n'
        'ap.add_argument("--b", help="no percent at all")\n'
    )
    monkeypatch.chdir(tmp_path)
    assert hook.run_argparse_help_render_scan(["good_cli.py"]) == 0
    assert hook.run_argparse_help_render_scan([]) == 0
    assert hook.run_argparse_help_render_scan(["README.md"]) == 0


def test_commit_time_guard_is_wired_into_hook_main() -> None:
    """An unwired guard is exactly the failure this landing exists to close."""
    hook_src = (REPO_ROOT / "tools" / "preflight_hook.py").read_text()
    main_body = hook_src.split("def main()", 1)
    assert len(main_body) == 2, "preflight_hook.main() not found"
    assert "run_argparse_help_render_scan(staged)" in main_body[1], (
        "run_argparse_help_render_scan is defined but never called from main() — "
        "a guard that does not run is decoration, which is the bug this landing closes"
    )


def test_guard_verdict_matches_real_argparse(tmp_path: Path) -> None:
    """The static verdict must agree with what argparse ACTUALLY does at format_help()."""
    import argparse

    hook = _load_preflight_hook()
    for text, should_crash in (
        ("runs for ~100% of its epochs", True),
        ("runs for ~100%% of its epochs", False),
        ("plain help", False),
    ):
        parser = argparse.ArgumentParser()
        parser.add_argument("--x", help=text)
        crashed = False
        try:
            parser.format_help()
        except (TypeError, ValueError, KeyError):
            crashed = True
        src = tmp_path / "probe_cli.py"
        src.write_text(
            "import argparse\nap = argparse.ArgumentParser()\n"
            f"ap.add_argument({'--x'!r}, help={text!r})\n"
        )
        flagged = bool(hook._argparse_help_render_violations(src))
        assert crashed == should_crash, f"argparse behaviour changed for {text!r}"
        assert flagged == crashed, (
            f"guard says flagged={flagged} but real argparse crashed={crashed} for {text!r}"
        )


def test_commit_time_guard_is_cwd_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round-2 self-review: the first draft returned a false PASS when CWD != repo root.

    `_staged_py_files()` yields repo-relative paths, so a scan that resolves them against the
    CWD scans NOTHING (and reports rc=0) whenever the caller's CWD moved. A guard whose failure
    mode is a silent pass is the exact class this landing exists to close.
    """
    hook = _load_preflight_hook()
    monkeypatch.chdir(tmp_path)  # deliberately NOT the repo root
    # A repo-relative path to a file that really exists and really is clean.
    assert hook.run_argparse_help_render_scan(
        ["tools/preflight_hook.py"]
    ) == 0, "clean repo file should pass"
    # And the scan must actually have READ it, not silently skipped it: point the guard at a
    # repo-relative file known to declare add_argument and confirm the violation helper
    # resolves it through REPO_ROOT rather than the CWD.
    resolved = REPO_ROOT / "experiments" / "train_tr1_partition_renderer_mlx.py"
    assert resolved.is_file()
    assert hook._argparse_help_render_violations(resolved) == [], (
        "the live TR1 trainer must render --help cleanly"
    )
