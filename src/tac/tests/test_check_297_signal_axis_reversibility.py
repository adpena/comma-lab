# SPDX-License-Identifier: MIT
"""Tests for Catalog #297 ``check_substrate_signal_axis_destruction_has_reversibility_probe``.

Per ``.omx/research/meta_assumption_backfill_audit_all_staircase_substrates_20260516.md``
+ ``.omx/research/grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516.md``
+ commit ``4292c8ce2`` (symposium) + commit ``b0a7ff474`` (META-assumption audit).

The gate refuses substrate trainers and codec modules under
``src/tac/substrates/*/`` or ``experiments/train_substrate_*.py`` that
contain forbidden signal-axis-destruction tokens (Y=R=G=B,
grayscale_to_rgb(...duplicate), frame.mean(...color), rgb_grey = (r+g+b)/3,
single_channel_only, _grayscale_to_rgb, _drop_chroma) without ONE of:
(a) sister probe file at ``tools/probe_<substrate>_reversibility*.py``,
(b) same-line waiver ``# SIGNAL_AXIS_DESTRUCTION_REVERSIBLE_PROBE_OK:<rationale>``,
(c) enclosing function name contains ``_compress_time_only`` marker.

Empirical anchors: NSCS06 Y=R=G=B chroma replication (seg=64.59), NSCS06
np.roll global translation (pose=149.03), Z3-G1 empty hyperprior slots
(silent 0.19869 baseline reproduction).

Sister of Catalog #220 / #272 / #139 / #105.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_signal_axis_destruction_has_reversibility_probe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _substrate_dir(repo_root: Path, substrate_id: str) -> Path:
    d = repo_root / "src" / "tac" / "substrates" / substrate_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_substrate_file(
    repo_root: Path, substrate_id: str, name: str, body: str
) -> Path:
    d = _substrate_dir(repo_root, substrate_id)
    p = d / name
    p.write_text(body, encoding="utf-8")
    return p


def _write_trainer_file(repo_root: Path, substrate_id: str, body: str) -> Path:
    d = repo_root / "experiments"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"train_substrate_{substrate_id}.py"
    p.write_text(body, encoding="utf-8")
    return p


def _write_probe(repo_root: Path, substrate_id: str, suffix: str = "") -> Path:
    d = repo_root / "tools"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"probe_{substrate_id}_reversibility{suffix}.py"
    p.write_text("# stub probe\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Live-repo regression guard
# ---------------------------------------------------------------------------


def test_297_live_repo_count_bounded() -> None:
    """Live count should be ≤ 5 per audit prediction."""
    violations = check_substrate_signal_axis_destruction_has_reversibility_probe(
        strict=False, verbose=False,
    )
    assert len(violations) <= 5, (
        f"Live-repo Catalog #297 violation count exceeded predicted ceiling "
        f"of 5. Found {len(violations)}: {violations[:3]}"
    )


# ---------------------------------------------------------------------------
# Out-of-scope behavior
# ---------------------------------------------------------------------------


def test_297_no_substrates_dir_returns_empty(tmp_path: Path) -> None:
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_297_empty_substrate_dir_returns_empty(tmp_path: Path) -> None:
    _substrate_dir(tmp_path, "foo")
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_297_substrate_file_without_destruction_tokens_clean(
    tmp_path: Path,
) -> None:
    _write_substrate_file(
        tmp_path, "foo", "inflate.py",
        "import numpy as np\n\ndef inflate():\n    return np.zeros((10,))\n",
    )
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Positive: destruction token without probe / waiver / marker = flagged
# ---------------------------------------------------------------------------


def test_297_grayscale_to_rgb_helper_flagged(tmp_path: Path) -> None:
    """NSCS06 anchor: ``_grayscale_to_rgb`` helper at compress-time."""
    _write_substrate_file(
        tmp_path, "nscs06", "inflate.py",
        "def _grayscale_to_rgb(gray):\n    return gray\n\n"
        "def inflate():\n    return _grayscale_to_rgb(None)\n",
    )
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    # Both the def line AND the call line trigger.
    assert len(v) >= 1


def test_297_y_r_g_b_chained_assignment_flagged(tmp_path: Path) -> None:
    """NSCS06 Y=R=G=B literal chained assignment."""
    _write_substrate_file(
        tmp_path, "nscs06", "codec.py",
        "def reconstruct():\n    gray = 128\n    Y = R = G = B = gray\n",
    )
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1
    assert "Y=R=G=B" in v[0] or "Y\\s*=\\s*R" in v[0]


def test_297_drop_chroma_helper_flagged(tmp_path: Path) -> None:
    _write_substrate_file(
        tmp_path, "foo", "codec.py",
        "def _drop_chroma(rgb):\n    return rgb[..., 0]\n",
    )
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_297_single_channel_only_marker_flagged(tmp_path: Path) -> None:
    _write_substrate_file(
        tmp_path, "foo", "trainer.py",
        "def train(single_channel_only=True):\n    pass\n",
    )
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_297_rgb_grey_literal_flagged(tmp_path: Path) -> None:
    """Manual chroma-to-luma reduction literal."""
    _write_substrate_file(
        tmp_path, "foo", "codec.py",
        "def compress(r, g, b):\n    rgb_grey = (r+g+b)/3\n    return rgb_grey\n",
    )
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_297_grayscale_to_rgb_call_with_duplicate_flagged(tmp_path: Path) -> None:
    _write_substrate_file(
        tmp_path, "foo", "inflate.py",
        "def inflate(gray):\n    return grayscale_to_rgb(gray, duplicate=True)\n",
    )
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_297_train_substrate_py_in_experiments_in_scope(tmp_path: Path) -> None:
    """``experiments/train_substrate_*.py`` files are scanned."""
    _write_trainer_file(
        tmp_path, "foo",
        "def train():\n    Y = R = G = B = 128\n",
    )
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


# ---------------------------------------------------------------------------
# Negative: acceptance cascade satisfied
# ---------------------------------------------------------------------------


def test_297_sister_probe_file_accepts(tmp_path: Path) -> None:
    """``tools/probe_<substrate>_reversibility.py`` short-circuits."""
    _write_substrate_file(
        tmp_path, "foo", "codec.py",
        "def codec():\n    Y = R = G = B = 128\n",
    )
    _write_probe(tmp_path, "foo")
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_297_sister_probe_with_suffix_accepts(tmp_path: Path) -> None:
    """``tools/probe_<substrate>_reversibility_chroma.py`` also accepts."""
    _write_substrate_file(
        tmp_path, "foo", "codec.py",
        "def codec():\n    Y = R = G = B = 128\n",
    )
    _write_probe(tmp_path, "foo", suffix="_chroma")
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_297_same_line_waiver_accepts(tmp_path: Path) -> None:
    _write_substrate_file(
        tmp_path, "foo", "codec.py",
        "def codec():\n"
        "    Y = R = G = B = 128  # SIGNAL_AXIS_DESTRUCTION_REVERSIBLE_PROBE_OK:"
        "explicit operator-approved compress-only single-channel lane per "
        "research_only=true scaffold contract\n",
    )
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_297_compress_time_only_function_short_circuits(tmp_path: Path) -> None:
    """Function name containing ``_compress_time_only`` accepts."""
    _write_substrate_file(
        tmp_path, "foo", "codec.py",
        "def _do_compress_time_only_grayscale():\n"
        "    Y = R = G = B = 128\n",
    )
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_297_placeholder_rationale_rejected(tmp_path: Path) -> None:
    """``<rationale>`` literal MUST NOT self-waive."""
    _write_substrate_file(
        tmp_path, "foo", "codec.py",
        "def codec():\n"
        "    Y = R = G = B = 128  # SIGNAL_AXIS_DESTRUCTION_REVERSIBLE_PROBE_OK:"
        "<rationale>\n",
    )
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_297_placeholder_reason_rejected(tmp_path: Path) -> None:
    """``<reason>`` literal MUST NOT self-waive."""
    _write_substrate_file(
        tmp_path, "foo", "codec.py",
        "def codec():\n"
        "    Y = R = G = B = 128  # SIGNAL_AXIS_DESTRUCTION_REVERSIBLE_PROBE_OK:"
        "<reason>\n",
    )
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


# ---------------------------------------------------------------------------
# Comment lines + docstring mentions must not trigger
# ---------------------------------------------------------------------------


def test_297_comment_line_with_pattern_not_flagged(tmp_path: Path) -> None:
    """A pure comment line mentioning ``Y=R=G=B`` is NOT flagged."""
    _write_substrate_file(
        tmp_path, "foo", "codec.py",
        "# This codec previously had Y = R = G = B replication but was fixed.\n"
        "def codec():\n    return None\n",
    )
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


def test_297_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    _write_substrate_file(
        tmp_path, "foo", "codec.py",
        "def codec():\n    Y = R = G = B = 128\n",
    )
    with pytest.raises(PreflightError) as exc_info:
        check_substrate_signal_axis_destruction_has_reversibility_probe(
            repo_root=tmp_path, strict=True, verbose=False,
        )
    msg = str(exc_info.value)
    assert "Catalog #297" in msg
    assert "signal-axis" in msg or "destruction" in msg


def test_297_strict_mode_silent_on_clean(tmp_path: Path) -> None:
    _write_substrate_file(
        tmp_path, "foo", "codec.py",
        "def codec():\n    pass\n",
    )
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=True, verbose=False,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Aggregation + edge cases
# ---------------------------------------------------------------------------


def test_297_aggregates_multiple_violations_one_file(tmp_path: Path) -> None:
    """Multiple destruction-token lines in one file = multiple violations."""
    _write_substrate_file(
        tmp_path, "foo", "codec.py",
        "def codec():\n"
        "    Y = R = G = B = 128\n"
        "    rgb_grey = (r+g+b)/3\n",
    )
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 2


def test_297_aggregates_across_files(tmp_path: Path) -> None:
    _write_substrate_file(
        tmp_path, "foo", "codec.py",
        "def codec():\n    Y = R = G = B = 128\n",
    )
    _write_substrate_file(
        tmp_path, "bar", "codec.py",
        "def _drop_chroma(rgb):\n    return rgb[..., 0]\n",
    )
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 2


def test_297_test_files_excluded(tmp_path: Path) -> None:
    """Files in ``/tests/`` subdir or named ``test_*.py`` are excluded."""
    d = tmp_path / "src" / "tac" / "substrates" / "foo" / "tests"
    d.mkdir(parents=True, exist_ok=True)
    (d / "test_codec.py").write_text(
        "def test_codec():\n    Y = R = G = B = 128\n",
        encoding="utf-8",
    )
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_297_string_repo_root_accepted(tmp_path: Path) -> None:
    _write_substrate_file(
        tmp_path, "foo", "codec.py",
        "def codec():\n    Y = R = G = B = 128\n",
    )
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=str(tmp_path), strict=False, verbose=False,
    )
    assert len(v) == 1


def test_297_syntax_error_file_tolerated(tmp_path: Path) -> None:
    """SyntaxError in target file is tolerated (silently skipped)."""
    # NB: our gate uses text scanning, not AST parsing, so syntax errors
    # are scanned for tokens anyway. Verify both behaviors are sensible.
    _write_substrate_file(
        tmp_path, "foo", "codec.py",
        "def codec(:::\n    Y = R = G = B = 128\n",
    )
    # Should not raise even with malformed syntax.
    v = check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    # Token-scanning still picks up the literal line.
    assert len(v) == 1


def test_297_verbose_clean_output(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _write_substrate_file(
        tmp_path, "foo", "codec.py",
        "def codec():\n    pass\n",
    )
    check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=True,
    )
    captured = capsys.readouterr()
    assert "check_substrate_signal_axis_destruction" in captured.out
    assert "OK" in captured.out


def test_297_verbose_dirty_output(
    capsys: pytest.CaptureFixture[str], tmp_path: Path
) -> None:
    _write_substrate_file(
        tmp_path, "foo", "codec.py",
        "def codec():\n    Y = R = G = B = 128\n",
    )
    check_substrate_signal_axis_destruction_has_reversibility_probe(
        repo_root=tmp_path, strict=False, verbose=True,
    )
    captured = capsys.readouterr()
    assert "1 violation" in captured.out
