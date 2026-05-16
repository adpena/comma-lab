# SPDX-License-Identifier: MIT
"""Tests for Catalog #312 ``check_hierarchical_predictive_coding_has_canonical_quadruple``.

Per Z6/Z7/Z8 design memo Pattern I (commit ``aa412d2db``).

The gate refuses substrate design memos claiming hierarchical predictive
coding WITHOUT the canonical Rao-Ballard + Mallat-CDF + DreamerV3 +
Wyner-Ziv quadruple.

Empirical anchor: Z8 design memo Sections 4.3 + 11 + 13 explicitly bind all
4 primitives simultaneously. Without the quadruple, "hierarchical predictive
coding" claims degenerate to hierarchy in name only.

Sister of Catalog #290 / #294 / #310 / #311 (paired G/H/I pattern).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_hierarchical_predictive_coding_has_canonical_quadruple,
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


def _full_quadruple_body() -> str:
    return (
        "## Architecture\n\n"
        "Hierarchical predictive coding via:\n"
        "- Rao-Ballard 1999 hierarchy (multi-level prediction errors)\n"
        "- Mallat wavelet hierarchical priors / CDF coding\n"
        "- Hafner DreamerV3 latent dynamics (stochastic + deterministic)\n"
        "- Wyner-Ziv side-information coding\n"
    )


# ---------------------------------------------------------------------------
# Live-repo regression guard
# ---------------------------------------------------------------------------


def test_312_live_repo_count_bounded() -> None:
    """STRICT-from-byte-one; live count expected 0."""
    violations = check_hierarchical_predictive_coding_has_canonical_quadruple(
        strict=False, verbose=False,
    )
    assert len(violations) <= 5, (
        f"Live count {len(violations)} > 5; Z8 design memo regressed?"
    )


# ---------------------------------------------------------------------------
# Out-of-scope behavior
# ---------------------------------------------------------------------------


def test_312_no_research_dir_returns_empty(tmp_path: Path) -> None:
    v = check_hierarchical_predictive_coding_has_canonical_quadruple(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_312_memo_without_trigger_out_of_scope(tmp_path: Path) -> None:
    _write(_research_dir(tmp_path), "foo_design_20260516.md", "no claim.\n")
    v = check_hierarchical_predictive_coding_has_canonical_quadruple(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_312_pre_cutoff_exempt(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "foo_design_20260515.md",
        "hierarchical predictive coding without quadruple.\n",
    )
    v = check_hierarchical_predictive_coding_has_canonical_quadruple(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_312_non_design_memo_not_scanned(tmp_path: Path) -> None:
    _write(_research_dir(tmp_path), "report_20260516.md", "hierarchical predictive coding\n")
    v = check_hierarchical_predictive_coding_has_canonical_quadruple(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Positive: hierarchical PC claim without quadruple = flagged
# ---------------------------------------------------------------------------


def test_312_claim_without_any_primitive_flagged(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\nThe substrate uses hierarchical predictive coding.\n",
    )
    v = check_hierarchical_predictive_coding_has_canonical_quadruple(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_312_claim_with_only_rao_ballard_flagged(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\nhierarchical predictive coding via Rao-Ballard hierarchy ONLY.\n",
    )
    v = check_hierarchical_predictive_coding_has_canonical_quadruple(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1
    # Should report missing 3 primitives
    assert "3 of 4" in v[0]


def test_312_claim_with_three_of_four_flagged(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\nhierarchical predictive coding via Rao-Ballard hierarchy + Mallat wavelet + DreamerV3 latent dynamics; no side-information coder.\n",
    )
    v = check_hierarchical_predictive_coding_has_canonical_quadruple(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1
    assert "1 of 4" in v[0]


def test_312_multiple_violations_aggregated(tmp_path: Path) -> None:
    research = _research_dir(tmp_path)
    _write(research, "a_design_20260516.md", "hierarchical predictive coding\n")
    _write(research, "b_design_20260516.md", "multi-level predictive coding\n")
    v = check_hierarchical_predictive_coding_has_canonical_quadruple(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 2


# ---------------------------------------------------------------------------
# Negative: full quadruple accepts
# ---------------------------------------------------------------------------


def test_312_full_quadruple_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        _full_quadruple_body(),
    )
    v = check_hierarchical_predictive_coding_has_canonical_quadruple(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_312_alternative_primitive_tokens_accepted(tmp_path: Path) -> None:
    """Tests that case-insensitive variants accept (rao ballard / hafner / etc.)."""
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\nmulti-level predictive coding via rao ballard hierarchy + daubechies wavelet + hafner latent dynamics + wyner ziv side-information.\n",
    )
    v = check_hierarchical_predictive_coding_has_canonical_quadruple(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Waiver semantics
# ---------------------------------------------------------------------------


def test_312_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\nhierarchical predictive coding\n\n# HIERARCHICAL_PREDICTIVE_CODING_QUADRUPLE_OK:partial-design-quadruple-in-sister-memo-z8\n",
    )
    v = check_hierarchical_predictive_coding_has_canonical_quadruple(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_312_placeholder_rationale_rejected(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "# X\n\nhierarchical predictive coding\n\n# HIERARCHICAL_PREDICTIVE_CODING_QUADRUPLE_OK:<rationale>\n",
    )
    v = check_hierarchical_predictive_coding_has_canonical_quadruple(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


def test_312_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "hierarchical predictive coding without primitives.\n",
    )
    with pytest.raises(PreflightError) as exc_info:
        check_hierarchical_predictive_coding_has_canonical_quadruple(
            repo_root=tmp_path, strict=True, verbose=False,
        )
    assert "Catalog #312" in str(exc_info.value)
    assert "Pattern I" in str(exc_info.value)


def test_312_strict_mode_silent_on_clean_repo(tmp_path: Path) -> None:
    _research_dir(tmp_path)
    v = check_hierarchical_predictive_coding_has_canonical_quadruple(
        repo_root=tmp_path, strict=True, verbose=False,
    )
    assert v == []
