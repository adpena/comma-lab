# SPDX-License-Identifier: MIT
"""Tests for Catalog #309 ``check_substrate_design_memo_declares_horizon_class``.

Per FALSIFICATION-AUDIT-v2 Pattern F (commit ``c5e4953e6``) + HORIZON-CLASS
standing directive 2026-05-16.

The gate refuses substrate design memos containing predicted-band sections
WITHOUT explicit horizon_class declaration (PLATEAU-ADJACENT / FRONTIER-PURSUIT
/ ASYMPTOTIC-PURSUIT).

Empirical anchor: the 0.196-0.199 cluster IS the canonical plateau; pursuing
more plateau-adjacent substrates without ANY asymptotic-pursuit alternative
is the structural failure mode (plateau trap).

Sister of Catalog #290 / #294 / #296 / #303 / #305.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_substrate_design_memo_declares_horizon_class,
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


def test_309_live_repo_count_bounded() -> None:
    violations = check_substrate_design_memo_declares_horizon_class(
        strict=False, verbose=False,
    )
    assert len(violations) <= 20, (
        f"Live count {len(violations)} > 20; backfill regressed?"
    )


# ---------------------------------------------------------------------------
# Out-of-scope behavior
# ---------------------------------------------------------------------------


def test_309_no_research_dir_returns_empty(tmp_path: Path) -> None:
    v = check_substrate_design_memo_declares_horizon_class(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_309_memo_without_predicted_band_out_of_scope(tmp_path: Path) -> None:
    _write(_research_dir(tmp_path), "foo_design_20260516.md", "no predicted band.\n")
    v = check_substrate_design_memo_declares_horizon_class(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_309_non_design_memo_not_scanned(tmp_path: Path) -> None:
    _write(_research_dir(tmp_path), "report_20260516.md", "## Predicted ΔS band\n\n[0.10, 0.20]\n")
    v = check_substrate_design_memo_declares_horizon_class(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_309_pre_cutoff_exempt(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "foo_design_20260515.md",
        "## Predicted ΔS band\n\n[0.10, 0.20]\n",
    )
    v = check_substrate_design_memo_declares_horizon_class(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Positive: predicted-band without horizon_class = flagged
# ---------------------------------------------------------------------------


def test_309_predicted_band_without_horizon_class_flagged(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "## Predicted ΔS band\n\n[0.10, 0.20] from heuristic; no horizon classification.\n",
    )
    v = check_substrate_design_memo_declares_horizon_class(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_309_predicted_score_band_variant_flagged(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "y_design_20260516.md",
        "## Predicted score band\n\n[0.05, 0.10]\n",
    )
    v = check_substrate_design_memo_declares_horizon_class(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_309_predicted_cpu_band_variant_flagged(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "z_design_20260516.md",
        "## Predicted CPU band\n\n[0.13, 0.16]\n",
    )
    v = check_substrate_design_memo_declares_horizon_class(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


def test_309_multiple_violations_aggregated(tmp_path: Path) -> None:
    research = _research_dir(tmp_path)
    _write(research, "a_design_20260516.md", "## Predicted ΔS band\n[0.10, 0.20]\n")
    _write(research, "b_design_20260516.md", "## Predicted score band\n[0.05, 0.10]\n")
    v = check_substrate_design_memo_declares_horizon_class(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 2


# ---------------------------------------------------------------------------
# Negative: horizon_class token accepts
# ---------------------------------------------------------------------------


def test_309_horizon_class_plateau_adjacent_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "## Predicted ΔS band\n\n[0.18, 0.20]\n\nhorizon_class: plateau_adjacent\n",
    )
    v = check_substrate_design_memo_declares_horizon_class(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_309_horizon_class_frontier_pursuit_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "## Predicted ΔS band\n\n[0.12, 0.18]\n\nhorizon_class: frontier_pursuit\n",
    )
    v = check_substrate_design_memo_declares_horizon_class(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_309_horizon_class_asymptotic_pursuit_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "## Predicted ΔS band\n\n[0.05, 0.10]\n\nhorizon_class: asymptotic_pursuit\n",
    )
    v = check_substrate_design_memo_declares_horizon_class(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_309_horizon_class_section_header_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "## Predicted ΔS band\n\n[0.13, 0.16]\n\n## horizon-class:\n\nFRONTIER-PURSUIT\n",
    )
    v = check_substrate_design_memo_declares_horizon_class(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


# ---------------------------------------------------------------------------
# Waiver semantics
# ---------------------------------------------------------------------------


def test_309_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "## Predicted ΔS band\n\n[0.10, 0.20]\n\n# HORIZON_CLASS_DECLARATION_OK:design-time-deferred-per-roadmap\n",
    )
    v = check_substrate_design_memo_declares_horizon_class(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == []


def test_309_placeholder_rationale_rejected(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "## Predicted ΔS band\n\n[0.10, 0.20]\n\n# HORIZON_CLASS_DECLARATION_OK:<rationale>\n",
    )
    v = check_substrate_design_memo_declares_horizon_class(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


def test_309_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    _write(
        _research_dir(tmp_path),
        "x_design_20260516.md",
        "## Predicted ΔS band\n\n[0.10, 0.20]\n",
    )
    with pytest.raises(PreflightError) as exc_info:
        check_substrate_design_memo_declares_horizon_class(
            repo_root=tmp_path, strict=True, verbose=False,
        )
    assert "Catalog #309" in str(exc_info.value)
    assert "Pattern F" in str(exc_info.value)


def test_309_strict_mode_silent_on_clean_repo(tmp_path: Path) -> None:
    _research_dir(tmp_path)
    v = check_substrate_design_memo_declares_horizon_class(
        repo_root=tmp_path, strict=True, verbose=False,
    )
    assert v == []
