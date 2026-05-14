# SPDX-License-Identifier: MIT
"""Catalog #209 — DP1 contest-video-leakage caller-routing gate tests.

Verifies ``tac.preflight.check_no_contest_video_leakage_in_distillation_callers``:

* live-repo regression guard (count must stay 0)
* positive (bare ``distill_codebook(`` callsite under in-scope dir flagged)
* negative (Comma2k19FrameIterator within 30-line lookback accepted)
* negative (canonical ``_build_frame_iterator`` helper invocation accepted)
* negative (manual ``check_no_contest_video_leakage`` accepted)
* same-line waiver acceptance / placeholder ``<reason>`` rejection
* file-level waiver acceptance / placeholder rejection
* exempt-path filter (``experiments/results/``, ``_intake_``, OSS export mirror)
* test-file exclusion (tests are out of scope by design)
* canonical implementation file self-exempt
* strict mode raises PreflightError; silent on clean
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_no_contest_video_leakage_in_distillation_callers,
)


# ---------------------------------------------------------------------------
# Live-repo regression
# ---------------------------------------------------------------------------


def test_live_repo_zero_violations() -> None:
    """The committed DP1 trainer + iterator wiring must keep #209 clean."""
    violations = check_no_contest_video_leakage_in_distillation_callers(
        strict=False
    )
    assert violations == [], (
        f"Catalog #209 live count drifted: {len(violations)} violations\n  "
        + "\n  ".join(violations[:5])
    )


# ---------------------------------------------------------------------------
# Helper to build a minimal repo skeleton with the in-scope dirs
# ---------------------------------------------------------------------------


def _make_scope_skeleton(repo_root: Path) -> None:
    """Create the in-scope directories the gate iterates over."""
    for sub in ("src/tac", "tools", "experiments", "scripts"):
        (repo_root / sub).mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Positive — bare distill_codebook caller is flagged
# ---------------------------------------------------------------------------


def test_bare_caller_flagged(tmp_path: Path) -> None:
    _make_scope_skeleton(tmp_path)
    target = tmp_path / "experiments" / "trainer.py"
    target.write_text(
        "from tac.substrates.pretrained_driving_prior import distill_codebook\n"
        "def main():\n"
        "    book = distill_codebook(cfg, frames=raw_iter)\n"
    )
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert len(v) == 1
    assert "experiments/trainer.py" in v[0]


def test_bare_caller_in_tools_flagged(tmp_path: Path) -> None:
    _make_scope_skeleton(tmp_path)
    target = tmp_path / "tools" / "distill_x.py"
    target.write_text(
        "from tac.substrates.pretrained_driving_prior import distill_codebook\n"
        "def run():\n"
        "    return distill_codebook(my_cfg)\n"
    )
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert len(v) == 1
    assert "tools/distill_x.py" in v[0]


def test_bare_caller_in_scripts_flagged(tmp_path: Path) -> None:
    _make_scope_skeleton(tmp_path)
    (tmp_path / "scripts" / "x.py").write_text(
        "import sys\n"
        "from tac.substrates.pretrained_driving_prior import distill_codebook\n"
        "distill_codebook(c)\n"
    )
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert len(v) == 1


def test_multiple_bare_callers_each_flagged(tmp_path: Path) -> None:
    _make_scope_skeleton(tmp_path)
    (tmp_path / "tools" / "a.py").write_text("distill_codebook(c)\n")
    (tmp_path / "tools" / "b.py").write_text("x = distill_codebook(c)\n")
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert len(v) == 2
    paths = [x.split(":")[0] for x in v]
    assert "tools/a.py" in paths and "tools/b.py" in paths


# ---------------------------------------------------------------------------
# Negative — iterator construction in lookback accepted
# ---------------------------------------------------------------------------


def test_iterator_in_lookback_accepted(tmp_path: Path) -> None:
    _make_scope_skeleton(tmp_path)
    target = tmp_path / "experiments" / "good.py"
    target.write_text(
        "from tac.substrates.pretrained_driving_prior import (\n"
        "    Comma2k19FrameIterator, distill_codebook,\n"
        ")\n"
        "def main():\n"
        "    it = Comma2k19FrameIterator(synthetic=True, n_frames=8)\n"
        "    book = distill_codebook(cfg, frames=iter(it))\n"
    )
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_canonical_helper_invocation_accepted(tmp_path: Path) -> None:
    """The canonical _build_frame_iterator(...) helper short-circuits the check.

    The helper itself is verified separately (its defining file lives under
    the in-scope dirs and must construct Comma2k19FrameIterator).
    """
    _make_scope_skeleton(tmp_path)
    target = tmp_path / "experiments" / "trainer_using_helper.py"
    target.write_text(
        "from tac.substrates.pretrained_driving_prior import distill_codebook\n"
        "def _build_frame_iterator(args):\n"
        "    pass  # canonical helper defined elsewhere\n"
        "def main():\n"
        "    it = _build_frame_iterator(args)\n"
        "    book = distill_codebook(cfg, frames=iter(it))\n"
    )
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_manual_guard_invocation_accepted(tmp_path: Path) -> None:
    """Manual ``check_no_contest_video_leakage([...])`` invocation accepted."""
    _make_scope_skeleton(tmp_path)
    target = tmp_path / "experiments" / "manual_guard.py"
    target.write_text(
        "from tac.substrates.pretrained_driving_prior import (\n"
        "    distill_codebook, check_no_contest_video_leakage,\n"
        ")\n"
        "def main():\n"
        "    check_no_contest_video_leakage([Path('/tmp/chunks')])\n"
        "    book = distill_codebook(cfg, frames=iter(my_raw))\n"
    )
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert v == []


# ---------------------------------------------------------------------------
# Waiver mechanism
# ---------------------------------------------------------------------------


def test_same_line_waiver_with_reason_accepted(tmp_path: Path) -> None:
    _make_scope_skeleton(tmp_path)
    target = tmp_path / "tools" / "waived.py"
    target.write_text(
        "distill_codebook(cfg)  # COMMA2K19_LEAKAGE_VERIFIED_OK:operator-audit-2026-05-14\n"
    )
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_same_line_waiver_placeholder_rejected(tmp_path: Path) -> None:
    """The literal `<reason>` placeholder cannot self-waive (docstring example)."""
    _make_scope_skeleton(tmp_path)
    (tmp_path / "tools" / "selfwaive.py").write_text(
        "distill_codebook(cfg)  # COMMA2K19_LEAKAGE_VERIFIED_OK:<reason>\n"
    )
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert len(v) == 1


def test_file_level_waiver_accepted(tmp_path: Path) -> None:
    _make_scope_skeleton(tmp_path)
    (tmp_path / "tools" / "filewaive.py").write_text(
        "# COMMA2K19_LEAKAGE_VERIFIED_OK_FILE:harness-routes-via-internal-helper-not-detectable-statically\n"
        "import x\n"
        "distill_codebook(cfg)\n"
    )
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_file_level_waiver_placeholder_rejected(tmp_path: Path) -> None:
    _make_scope_skeleton(tmp_path)
    (tmp_path / "tools" / "filewaive_bad.py").write_text(
        "# COMMA2K19_LEAKAGE_VERIFIED_OK_FILE:<reason>\n"
        "distill_codebook(cfg)\n"
    )
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert len(v) == 1


# ---------------------------------------------------------------------------
# Scope filter — exempt directories + test files
# ---------------------------------------------------------------------------


def test_experiments_results_path_exempt(tmp_path: Path) -> None:
    _make_scope_skeleton(tmp_path)
    nested = tmp_path / "experiments" / "results" / "lane_x"
    nested.mkdir(parents=True)
    (nested / "regen.py").write_text("distill_codebook(c)\n")
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_intake_clone_path_exempt(tmp_path: Path) -> None:
    _make_scope_skeleton(tmp_path)
    nested = tmp_path / "experiments" / "results" / "public_pr_intake_codex"
    nested.mkdir(parents=True)
    (nested / "vendored_caller.py").write_text("distill_codebook(c)\n")
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_oss_export_mirror_exempt(tmp_path: Path) -> None:
    _make_scope_skeleton(tmp_path)
    omx = tmp_path / "src" / "tac" / ".omx" / "oss_export"
    omx.mkdir(parents=True)
    (omx / "x.py").write_text("distill_codebook(c)\n")
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_tests_subdir_exempt(tmp_path: Path) -> None:
    """Files under */tests/* are out of scope (tests cover both modes)."""
    _make_scope_skeleton(tmp_path)
    (tmp_path / "src" / "tac" / "tests").mkdir(parents=True)
    (tmp_path / "src" / "tac" / "tests" / "test_distill.py").write_text(
        "distill_codebook(c)\n"
    )
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_test_prefix_files_exempt(tmp_path: Path) -> None:
    """test_* files at any level are exempt."""
    _make_scope_skeleton(tmp_path)
    (tmp_path / "tools" / "test_distill_smoke.py").write_text(
        "distill_codebook(c)\n"
    )
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_test_suffix_files_exempt(tmp_path: Path) -> None:
    _make_scope_skeleton(tmp_path)
    (tmp_path / "tools" / "distill_test.py").write_text(
        "distill_codebook(c)\n"
    )
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert v == []


# ---------------------------------------------------------------------------
# Self-exempt: canonical implementation file
# ---------------------------------------------------------------------------


def test_canonical_implementation_file_self_exempt(tmp_path: Path) -> None:
    """The canonical distillation.py contains def distill_codebook itself.

    The function-definition line should NOT be flagged (skipped via the
    'def ' prefix filter); but the legitimate self-exempt protects against
    any internal `distill_codebook(` recursion.
    """
    _make_scope_skeleton(tmp_path)
    canonical = (
        tmp_path
        / "src"
        / "tac"
        / "substrates"
        / "pretrained_driving_prior"
    )
    canonical.mkdir(parents=True)
    (canonical / "distillation.py").write_text(
        "def distill_codebook(cfg, *, frames=None):\n"
        "    return distill_codebook(cfg, frames=frames)\n"
    )
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    # Canonical file is self-exempt; no violations even though it has a
    # non-def call that would otherwise flag.
    assert v == []


# ---------------------------------------------------------------------------
# Other edge cases
# ---------------------------------------------------------------------------


def test_def_distill_codebook_not_flagged(tmp_path: Path) -> None:
    """A file defining a function named distill_codebook (without calling it)
    should not be flagged on the def line."""
    _make_scope_skeleton(tmp_path)
    (tmp_path / "tools" / "wrap.py").write_text(
        "def distill_codebook(*args, **kwargs):\n"
        "    return None\n"
    )
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_import_line_not_flagged(tmp_path: Path) -> None:
    _make_scope_skeleton(tmp_path)
    (tmp_path / "tools" / "imp.py").write_text(
        "from tac.substrates.pretrained_driving_prior import distill_codebook\n"
        "from tac.x import (distill_codebook, foo)\n"
    )
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_unrelated_token_not_flagged(tmp_path: Path) -> None:
    _make_scope_skeleton(tmp_path)
    (tmp_path / "tools" / "u.py").write_text(
        "# Notes about distill_codebook design but no actual call.\n"
        "x = my_distill_codebook_helper(c)\n"
    )
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    # ``my_distill_codebook_helper(`` contains the substring distill_codebook(
    # only if the helper happens to share the suffix; ours is `_helper(` which
    # does NOT match `distill_codebook(`.
    assert v == []


def test_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    _make_scope_skeleton(tmp_path)
    (tmp_path / "tools" / "violator.py").write_text("distill_codebook(c)\n")
    with pytest.raises(PreflightError) as exc:
        check_no_contest_video_leakage_in_distillation_callers(
            repo_root=tmp_path, strict=True
        )
    msg = str(exc.value)
    assert "Catalog #209" in msg
    assert "Comma2k19FrameIterator" in msg


def test_strict_mode_silent_on_clean(tmp_path: Path) -> None:
    _make_scope_skeleton(tmp_path)
    # Empty repo skeleton — no callers; strict should not raise.
    out = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=True
    )
    assert out == []


def test_unicode_decode_error_skipped_silently(tmp_path: Path) -> None:
    _make_scope_skeleton(tmp_path)
    bad = tmp_path / "tools" / "bin.py"
    bad.write_bytes(b"\x80\x81 distill_codebook(c)\n")
    # Should not raise; bad-encoding files are silently skipped.
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert v == []


def test_iterator_more_than_30_lines_above_not_accepted(tmp_path: Path) -> None:
    """Iterator construction outside the 30-line lookback should NOT count."""
    _make_scope_skeleton(tmp_path)
    body_lines = ["from x import Comma2k19FrameIterator, distill_codebook"]
    body_lines.append("it = Comma2k19FrameIterator(synthetic=True)")
    # 35 filler lines so iterator falls outside lookback.
    body_lines.extend(["# filler"] * 35)
    body_lines.append("distill_codebook(cfg, frames=iter(it))")
    (tmp_path / "tools" / "far.py").write_text("\n".join(body_lines) + "\n")
    v = check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False
    )
    assert len(v) == 1


def test_verbose_mode_prints_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _make_scope_skeleton(tmp_path)
    (tmp_path / "tools" / "a.py").write_text("distill_codebook(c)\n")
    check_no_contest_video_leakage_in_distillation_callers(
        repo_root=tmp_path, strict=False, verbose=True
    )
    captured = capsys.readouterr()
    assert "contest-video-leakage-callers" in captured.out


def test_orchestrator_callsite_strict_true(tmp_path: Path) -> None:
    """preflight_all's wired callsite uses strict=True (regression guard)."""
    src = Path(__file__).resolve().parent.parent / "preflight.py"
    text = src.read_text(encoding="utf-8")
    # Find the orchestrator callsite (one of the lambda/positional invocations
    # inside preflight_all). The pattern looks for a `strict=True` kwarg in
    # the same 5-line window as the call.
    needle = "check_no_contest_video_leakage_in_distillation_callers("
    idx = text.find(needle, text.find("def preflight_all("))
    assert idx >= 0, "Catalog #209 not wired into preflight_all()"
    window = text[idx : idx + 250]
    assert "strict=True" in window, (
        "Catalog #209 wired into preflight_all() but not strict=True; "
        "see CLAUDE.md 'Strict-flip atomicity rule'"
    )
