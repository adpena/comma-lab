"""Council R3 regression: verify train_renderer's auth-eval-on-best wiring is REAL.

The previous wiring (R2) was dead code:
  - Invented `--auth-eval-masks` flag for auth_eval_renderer that doesn't exist
  - Skipped --archive-size-bytes → rate computed from renderer-only (~290KB)
    instead of triple-joint archive (~700KB), systematically optimistic
  - Silently no-op'd if --auth-eval-masks wasn't passed (which was every chain
    because the flag was never plumbed through any caller)

These tests verify the R3 fix actually:
  A. Reads auth_eval_renderer.py's REAL argparse args (no invented flags)
  B. Builds a REAL archive when masks + poses provided
  C. Passes --archive-size-bytes from the BUILT archive (not renderer-only)
  D. Doesn't silently skip when masks/poses missing (logs WARN, runs anyway)
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def auth_eval_argparse() -> set[str]:
    """The REAL set of --flag-names that auth_eval_renderer.py accepts.

    Computed by introspecting the file (not by guessing). Any wiring that
    passes a flag NOT in this set is dead code that argparse will reject
    or silently ignore.
    """
    src = (REPO / "experiments" / "auth_eval_renderer.py").read_text()
    # Match `parser.add_argument("--name", ...)` and `parser.add_argument( "--name", ...)`
    flags = set(re.findall(r'parser\.add_argument\(\s*"(--[a-z0-9_-]+)"', src))
    assert flags, f"no flags found in auth_eval_renderer.py — regex bug"
    return flags


def test_auth_eval_renderer_has_no_masks_flag(auth_eval_argparse):
    """Documents the bug: there is NO --masks or --auth-eval-masks flag.
    Any wiring that passes either is dead code."""
    assert "--masks" not in auth_eval_argparse
    assert "--auth-eval-masks" not in auth_eval_argparse


def test_auth_eval_renderer_required_flags(auth_eval_argparse):
    """Affirmative test: these are the flags train_renderer MUST use."""
    must_exist = {"--checkpoint", "--upstream-dir", "--device", "--poses",
                  "--archive-size-bytes", "--output-dir"}
    missing = must_exist - auth_eval_argparse
    assert not missing, (
        f"auth_eval_renderer.py is missing flags train_renderer expects: {missing}"
    )


def test_train_renderer_does_not_pass_invented_flags():
    """train_renderer.py's auth-eval-on-best block must NOT emit any flag
    that auth_eval_renderer.py doesn't accept. This catches the dead-flag
    wiring that R3 found."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    # Find the auth-eval block — between "# Auth eval on best" and "return best_scorer"
    m = re.search(r"# Auth eval on best.*?return best_scorer", src, re.DOTALL)
    assert m, "auth-eval-on-best block not found"
    block = m.group(0)

    # Get every "--flag" used in the block as a string literal
    flags_in_block = set(re.findall(r'"(--[a-z0-9_-]+)"', block))

    # Get the real auth_eval_renderer flags
    auth_src = (REPO / "experiments" / "auth_eval_renderer.py").read_text()
    real_flags = set(re.findall(r'parser\.add_argument\(\s*"(--[a-z0-9_-]+)"', auth_src))

    invented = flags_in_block - real_flags
    assert not invented, (
        f"train_renderer's auth-eval block uses flags that don't exist in "
        f"auth_eval_renderer.py argparse: {invented}. This is dead code that "
        f"argparse would reject."
    )


def test_train_renderer_passes_archive_size_bytes():
    """Council R3-2: the subprocess MUST pass --archive-size-bytes,
    otherwise rate is computed from renderer-only (~290KB) which is
    optimistic vs the real triple-joint archive (~700KB)."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    m = re.search(r"# Auth eval on best.*?return best_scorer", src, re.DOTALL)
    assert m
    block = m.group(0)
    assert '"--archive-size-bytes"' in block, (
        "train_renderer must pass --archive-size-bytes to auth_eval_renderer, "
        "otherwise the rate term is systematically optimistic by ~2x. "
        "This is Council R3-2."
    )


def test_train_renderer_builds_real_archive():
    """The fix mandate: when masks + poses are provided, the wiring
    must call build_submission_archive (NOT just pass renderer.bin
    bytes as the 'archive size')."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    m = re.search(r"# Auth eval on best.*?return best_scorer", src, re.DOTALL)
    assert m
    block = m.group(0)
    assert "build_submission_archive" in block, (
        "auth-eval-on-best must use build_submission_archive when both "
        "masks + poses are provided. Otherwise the archive size passed "
        "to auth_eval_renderer is renderer-only and the rate is biased."
    )


def test_train_renderer_does_not_silently_skip():
    """Council R3-3: previous R2 wiring SKIPPED the eval entirely if
    --auth-eval-masks wasn't passed, meaning every existing chain (none
    of which pass the flag) silently no-op'd. The fix must run the eval
    anyway, with a WARN about the rate bias."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    m = re.search(r"# Auth eval on best.*?return best_scorer", src, re.DOTALL)
    assert m
    block = m.group(0)
    # The bad pattern: an early-out/skip when args.auth_eval_masks is missing.
    # Look for `if not args.auth_eval_masks:` followed by a print/skip with
    # NO subsequent subprocess.run.
    bad = re.search(
        r"if not args\.auth_eval_masks[^\n]*:\s*\n\s*print[^\n]*(?:WARN|Skipping)[^\n]*\n\s*else:",
        block,
    )
    assert bad is None, (
        "auth-eval-on-best must not gate the entire eval on --auth-eval-masks. "
        "Run with renderer-only archive size if masks missing (with WARN), "
        "but DO run. This was Council R3-3."
    )


def test_train_renderer_uses_repo_root_constant():
    """Council R3-4: subprocess path resolution must use the
    established `_repo` constant, not brittle `..` traversal."""
    src = (REPO / "src" / "tac" / "experiments" / "train_renderer.py").read_text()
    m = re.search(r"# Auth eval on best.*?return best_scorer", src, re.DOTALL)
    assert m
    block = m.group(0)
    # Either uses _repo or another robust pathlib idiom. Specifically reject
    # the `parents[N] / ".."` antipattern.
    bad_paths = re.findall(r'parents\[\d+\]\s*/\s*"\.\."', block)
    assert not bad_paths, (
        f"train_renderer auth-eval block uses brittle `parents[N] / '..'` "
        f"path traversal: {bad_paths}. Use the existing `_repo` constant."
    )


def test_argparse_smoke_default_true():
    """The flag must default to True per CLAUDE.md non-negotiable, and
    --no-auth-eval-on-best must turn it off."""
    sys.path.insert(0, str(REPO / "src" / "tac" / "experiments"))
    if "train_renderer" in sys.modules:
        del sys.modules["train_renderer"]
    from train_renderer import parse_args  # noqa: E402
    a1 = parse_args(["--tag", "smoke"])
    assert a1.auth_eval_on_best is True
    a2 = parse_args(["--tag", "smoke", "--no-auth-eval-on-best"])
    assert a2.auth_eval_on_best is False
