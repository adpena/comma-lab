# SPDX-License-Identifier: MIT
"""Tests for Catalog #310 ``check_f_asymptote_substrate_design_is_class_shift_not_bolt_on``.

Per Z6/Z7/Z8 design memo Pattern G (commit ``aa412d2db``).

The gate refuses substrate design memos claiming F-asymptote-class /
asymptotic-pursuit status WITHOUT explicit primary class-shift
architecture declaration (vs bolt-on).

Empirical anchor: Z4 (cooperative-receiver loss) and Z5 (predictive-coding)
as designed were BOLT-ONS to Z3/A1; Z6/Z7/Z8 must be PRIMARY substrates
with predictive-coding as architectural core.

Sister of Catalog #290 / #294 / #296 / #303 / #309.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_f_asymptote_substrate_design_is_class_shift_not_bolt_on,
)


def _research_dir(repo_root: Path) -> Path:
    d = repo_root / ".omx" / "research"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write(d: Path, name: str, body: str) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    path = d / name
    path.write_text(body, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Live-repo regression guard
# ---------------------------------------------------------------------------


def test_310_live_repo_count_bounded() -> None:
    """STRICT-from-byte-one: live count expected 0."""
    violations = check_f_asymptote_substrate_design_is_class_shift_not_bolt_on(
        strict=False, verbose=False,
    )
    assert len(violations) <= 5, (
        f"Live count {len(violations)} > 5; Z6/Z7/Z8 design or sister "
        "memo regressed?"
    )


# ---------------------------------------------------------------------------
# Out-of-scope behavior
# ---------------------------------------------------------------------------


def test_310_no_research_dir_returns_empty(tmp_path: Path) -> None:
    v = check_f_asymptote_substrate_design_is_class_shift_not_bolt_on(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_310_memo_without_f_asymptote_token_out_of_scope(tmp_path: Path) -> None:
    _write(_research_dir(tmp_path), "foo_design_20260516.md", "no asymptotic claim.\n")
    v = check_f_asymptote_substrate_design_is_class_shift_not_bolt_on(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_310_pre_cutoff_exempt(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "foo_design_20260515.md",
        "f-asymptote substrate target.\n",
    )
    v = check_f_asymptote_substrate_design_is_class_shift_not_bolt_on(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_310_non_design_memo_not_scanned(tmp_path: Path) -> None:
    _write(_research_dir(tmp_path), "report_20260516.md", "asymptotic-pursuit\n")
    v = check_f_asymptote_substrate_design_is_class_shift_not_bolt_on(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Positive: f-asymptote claim without class-shift declaration = flagged
# ---------------------------------------------------------------------------


def test_310_asymptotic_pursuit_without_class_shift_flagged(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\nThis is an asymptotic-pursuit substrate. We bolt it on to A1.\n",
    )
    v = check_f_asymptote_substrate_design_is_class_shift_not_bolt_on(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_310_f_asymptote_token_without_declaration_flagged(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "y_design_20260516.md",
        "# Y\n\nF-asymptote lattice node. Bolt-on to existing substrate.\n",
    )
    v = check_f_asymptote_substrate_design_is_class_shift_not_bolt_on(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_310_multiple_violations_aggregated(tmp_path: Path) -> None:
    research = _research_dir(tmp_path)
    _write(research, "a_design_20260516.md", "asymptotic-pursuit on A1.\n")
    _write(research, "b_design_20260516.md", "F-asymptote claim.\n")
    v = check_f_asymptote_substrate_design_is_class_shift_not_bolt_on(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 2


# ---------------------------------------------------------------------------
# Negative: class-shift declaration accepts
# ---------------------------------------------------------------------------


def test_310_primary_substrate_declaration_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\nasymptotic-pursuit primary substrate. Not bolt-on.\n",
    )
    v = check_f_asymptote_substrate_design_is_class_shift_not_bolt_on(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_310_class_shift_substrate_token_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\nF-asymptote class-shift substrate; architectural core is predictive coding.\n",
    )
    v = check_f_asymptote_substrate_design_is_class_shift_not_bolt_on(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_310_scorer_relationship_class_shift_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\nasymptotic-pursuit; scorer-relationship class-shift.\n",
    )
    v = check_f_asymptote_substrate_design_is_class_shift_not_bolt_on(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_310_bolt_on_contrast_token_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\nasymptotic-pursuit. paradigm not bolt-on.\n",
    )
    v = check_f_asymptote_substrate_design_is_class_shift_not_bolt_on(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Waiver semantics
# ---------------------------------------------------------------------------


def test_310_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\nasymptotic-pursuit\n\n# F_ASYMPTOTE_CLASS_SHIFT_NOT_BOLT_ON_OK:research-only-scoping-memo-no-build\n",
    )
    v = check_f_asymptote_substrate_design_is_class_shift_not_bolt_on(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_310_placeholder_rationale_rejected(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\nasymptotic-pursuit\n\n# F_ASYMPTOTE_CLASS_SHIFT_NOT_BOLT_ON_OK:<rationale>\n",
    )
    v = check_f_asymptote_substrate_design_is_class_shift_not_bolt_on(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


def test_310_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "asymptotic-pursuit substrate.\n",
    )
    with pytest.raises(PreflightError) as exc_info:
        check_f_asymptote_substrate_design_is_class_shift_not_bolt_on(
            repo_root=tmp_path, strict=True, verbose=False,
        )
    assert "Catalog #310" in str(exc_info.value)
    assert "Pattern G" in str(exc_info.value)


def test_310_strict_mode_silent_on_clean_repo(tmp_path: Path) -> None:
    _research_dir(tmp_path)
    v = check_f_asymptote_substrate_design_is_class_shift_not_bolt_on(
        repo_root=tmp_path, strict=True, verbose=False,
    )
    assert v == []
