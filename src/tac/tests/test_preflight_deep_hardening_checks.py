# SPDX-License-Identifier: MIT
"""Tests for Preflight Checks 51, 52, 53 (deep DX hardening pass 2026-04-28).

- Check 51: check_no_bare_except — bare `except:` and `except Exception: pass`
- Check 52: check_subprocess_run_checked — unchecked subprocess.run
- Check 53: check_tools_have_argparse — CLI tools without --help wiring
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    check_no_bare_except,
    check_subprocess_run_checked,
    check_tools_have_argparse,
)


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Create a fake repo structure: src/tac/, scripts/, tools/, experiments/."""
    for d in ("src/tac", "scripts", "tools", "experiments"):
        (tmp_path / d).mkdir(parents=True, exist_ok=True)
    return tmp_path


# ── Check 51 ─────────────────────────────────────────────────────────────


def test_check_51_passes_on_clean_specific_exception(fake_repo):
    f = fake_repo / "src/tac/clean.py"
    f.write_text(
        "def foo():\n"
        "    try:\n"
        "        x = 1\n"
        "    except IndexError:\n"
        "        pass\n"
    )
    violations = check_no_bare_except(repo_root=fake_repo, strict=False, verbose=False)
    assert violations == []


def test_check_51_catches_bare_except(fake_repo):
    f = fake_repo / "src/tac/bad.py"
    f.write_text(
        "def foo():\n"
        "    try:\n"
        "        x = 1\n"
        "    except:\n"
        "        pass\n"
    )
    violations = check_no_bare_except(repo_root=fake_repo, strict=False, verbose=False)
    assert len(violations) == 1
    assert "bare `except:`" in violations[0]


def test_check_51_catches_silent_swallow(fake_repo):
    f = fake_repo / "src/tac/swallow.py"
    f.write_text(
        "def foo():\n"
        "    try:\n"
        "        x = 1\n"
        "    except Exception: pass\n"
    )
    violations = check_no_bare_except(repo_root=fake_repo, strict=False, verbose=False)
    assert len(violations) == 1
    assert "silently swallows" in violations[0]


def test_check_51_honors_waiver(fake_repo):
    f = fake_repo / "src/tac/waived.py"
    f.write_text(
        "def foo():\n"
        "    try:\n"
        "        x = 1\n"
        "    except:  # noqa: E722\n"
        "        pass\n"
    )
    violations = check_no_bare_except(repo_root=fake_repo, strict=False, verbose=False)
    assert violations == []


def test_check_51_strict_raises(fake_repo):
    f = fake_repo / "src/tac/bad.py"
    # Bare except on its own line — what the regex actually matches.
    f.write_text("try:\n    x=1\nexcept:\n    pass\n")
    with pytest.raises(MetaBugViolation):
        check_no_bare_except(repo_root=fake_repo, strict=True, verbose=False)


# ── Check 52 ─────────────────────────────────────────────────────────────


def test_check_52_passes_on_check_true(fake_repo):
    f = fake_repo / "src/tac/clean_subproc.py"
    f.write_text(
        "import subprocess\n"
        "def go():\n"
        "    subprocess.run(['ls'], check=True)\n"
    )
    violations = check_subprocess_run_checked(repo_root=fake_repo, strict=False, verbose=False)
    assert violations == []


def test_check_52_passes_on_returncode_check(fake_repo):
    f = fake_repo / "src/tac/checked.py"
    f.write_text(
        "import subprocess\n"
        "def go():\n"
        "    r = subprocess.run(['ls'])\n"
        "    if r.returncode != 0:\n"
        "        raise RuntimeError('failed')\n"
    )
    violations = check_subprocess_run_checked(repo_root=fake_repo, strict=False, verbose=False)
    assert violations == []


def test_check_52_catches_unchecked_call(fake_repo):
    f = fake_repo / "src/tac/unchecked.py"
    f.write_text(
        "import subprocess\n"
        "def go():\n"
        "    subprocess.run(['rm', '-rf', '/'])\n"
    )
    violations = check_subprocess_run_checked(repo_root=fake_repo, strict=False, verbose=False)
    assert len(violations) == 1
    assert "without check=True" in violations[0]


def test_check_52_honors_waiver(fake_repo):
    f = fake_repo / "src/tac/waived_subproc.py"
    f.write_text(
        "import subprocess\n"
        "def go():\n"
        "    subprocess.run(['ls'])  # subprocess-no-check-OK: best-effort\n"
    )
    violations = check_subprocess_run_checked(repo_root=fake_repo, strict=False, verbose=False)
    assert violations == []


def test_check_52_check_false_is_explicit_optout(fake_repo):
    f = fake_repo / "src/tac/explicit_optout.py"
    f.write_text(
        "import subprocess\n"
        "def go():\n"
        "    subprocess.run(['ls'], check=False)\n"
    )
    violations = check_subprocess_run_checked(repo_root=fake_repo, strict=False, verbose=False)
    assert violations == []


def test_check_52_ignores_docstring_mention(fake_repo):
    # AST precision layer (pf2x r80): a docstring MENTIONING subprocess.run(
    # is not a call site (process_group_kill.py's module doc, live FP).
    f = fake_repo / "src/tac/doc_mention.py"
    f.write_text(
        '"""Helper notes.\n'
        "\n"
        "``subprocess.run(cmd, timeout=N)`` kills only the direct child.\n"
        '"""\n'
        "X = 1\n"
    )
    violations = check_subprocess_run_checked(repo_root=fake_repo, strict=False, verbose=False)
    assert violations == []


def test_check_52_accepts_check_true_beyond_text_window(fake_repo):
    # AST precision layer: check=True 12 lines into a long arg list sits
    # past the 8-line text window (modal_click_polish_cpu.py, live FP).
    args = "".join(f"        'arg{i}',\n" for i in range(10))
    f = fake_repo / "src/tac/long_call.py"
    f.write_text(
        "import subprocess\n"
        "def go():\n"
        "    subprocess.run(\n"
        "        [\n"
        f"{args}"
        "        ],\n"
        "        check=True,\n"
        "    )\n"
    )
    violations = check_subprocess_run_checked(repo_root=fake_repo, strict=False, verbose=False)
    assert violations == []


def test_check_52_accepts_returned_completed_process(fake_repo):
    # AST precision layer: `return subprocess.run(...)` propagates the
    # CompletedProcess to the caller — not captured-and-ignored, not
    # discarded (auto_push_main._git wrapper shape, live FP).
    f = fake_repo / "src/tac/thin_wrapper.py"
    f.write_text(
        "import subprocess\n"
        "def _git(*args):\n"
        "    return subprocess.run(['git', *args], capture_output=True, text=True)\n"
    )
    violations = check_subprocess_run_checked(repo_root=fake_repo, strict=False, verbose=False)
    assert violations == []


def test_check_52_accepts_check_kwarg_passthrough(fake_repo):
    # AST precision layer: check=<variable> is an active choice threaded by
    # the caller (public_repo_hygiene_census check=check w/ default True).
    f = fake_repo / "src/tac/passthrough.py"
    f.write_text(
        "import subprocess\n"
        "def _run(argv, check=True):\n"
        "    r = subprocess.run(argv, check=check, capture_output=True)\n"
        "    return r.stdout\n"
    )
    violations = check_subprocess_run_checked(repo_root=fake_repo, strict=False, verbose=False)
    assert violations == []


def test_check_52_still_catches_unchecked_after_ast_layer(fake_repo):
    # POSITIVE CONTROL for the AST layer: a genuinely unchecked, captured,
    # stdout-consumed call must STILL be flagged (the bug class itself).
    f = fake_repo / "src/tac/still_bad.py"
    f.write_text(
        "import subprocess\n"
        "def go():\n"
        "    out = subprocess.run(['ls'], capture_output=True, text=True).stdout\n"
        "    return out\n"
    )
    violations = check_subprocess_run_checked(repo_root=fake_repo, strict=False, verbose=False)
    assert len(violations) == 1
    assert "without check=True" in violations[0]


# ── Check 53 ─────────────────────────────────────────────────────────────


def test_check_53_passes_on_argparse_tool(fake_repo):
    f = fake_repo / "tools/good_tool.py"
    f.write_text(
        '"""A tool."""\n'
        "import argparse\n"
        "def main():\n"
        "    p = argparse.ArgumentParser()\n"
        "    p.add_argument('x')\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    violations = check_tools_have_argparse(repo_root=fake_repo, strict=False, verbose=False)
    assert violations == []


def test_check_53_catches_no_argparse(fake_repo):
    f = fake_repo / "tools/bad_tool.py"
    f.write_text(
        '"""A tool with no argparse."""\n'
        "import sys\n"
        "def main():\n"
        "    print(sys.argv)\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    violations = check_tools_have_argparse(repo_root=fake_repo, strict=False, verbose=False)
    assert len(violations) == 1
    assert "tools/bad_tool.py" in violations[0]


def test_check_53_skips_library_helpers(fake_repo):
    """A tools/ file with no __main__ entry is a library helper, not a CLI."""
    f = fake_repo / "tools/helper.py"
    f.write_text(
        '"""A library helper."""\n'
        "def util():\n"
        "    return 42\n"
    )
    violations = check_tools_have_argparse(repo_root=fake_repo, strict=False, verbose=False)
    assert violations == []


def test_check_53_honors_waiver(fake_repo):
    f = fake_repo / "tools/waived.py"
    f.write_text(
        '"""A tool. # no-argparse-OK: subcommand dispatch via sys.argv."""\n'
        "import sys\n"
        "def main():\n"
        "    print(sys.argv[1])\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    violations = check_tools_have_argparse(repo_root=fake_repo, strict=False, verbose=False)
    assert violations == []


def test_check_53_accepts_click(fake_repo):
    f = fake_repo / "tools/click_tool.py"
    f.write_text(
        '"""A click tool."""\n'
        "import click\n"
        "@click.command()\n"
        "def main():\n"
        "    pass\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    violations = check_tools_have_argparse(repo_root=fake_repo, strict=False, verbose=False)
    assert violations == []
