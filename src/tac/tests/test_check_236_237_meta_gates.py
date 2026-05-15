# SPDX-License-Identifier: MIT
"""Catalog #236 + #237 META preflight gate regression tests.

These tests pin the META preflight gate behavior:
- #236: refuses drift of the TAO-1 prose-negation guard surfaces
- #237: refuses drift of the BOYD-1 fallback-semantic disambiguator
        surfaces + enforces dict-disjointness invariant

Memory: feedback_r2_critical_fix_wave_tao1_boyd1_landed_20260515.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _CHECK_236_NEGATION_TOKEN_FLOOR,
    _CHECK_236_REQUIRED_PREFLIGHT_TOKENS,
    _CHECK_237_REQUIRED_TOKENS,
    check_catalog_233_gate_3_prose_negation_guard_present,
    check_d9_fallback_semantic_disambiguator_present,
)


# ----------------------------------------------------------------------------
# Catalog #236 — TAO-1 prose-negation guard META gate
# ----------------------------------------------------------------------------


def test_catalog_236_live_repo_clean() -> None:
    """Live repo must satisfy Catalog #236 (no drift)."""
    violations = check_catalog_233_gate_3_prose_negation_guard_present(
        strict=False
    )
    assert violations == [], (
        f"Catalog #236 live count is {len(violations)} — investigate drift"
    )


def test_catalog_236_strict_silent_on_clean() -> None:
    """Strict mode must NOT raise when repo is clean."""
    violations = check_catalog_233_gate_3_prose_negation_guard_present(
        strict=True
    )
    assert violations == []


def test_catalog_236_required_tokens_set_size() -> None:
    """The required-tokens set has the canonical 4 surfaces."""
    assert len(_CHECK_236_REQUIRED_PREFLIGHT_TOKENS) == 4


def test_catalog_236_negation_token_floor_is_canonical() -> None:
    """The negation token floor is set to the canonical value."""
    assert _CHECK_236_NEGATION_TOKEN_FLOOR == 10


def test_catalog_236_flags_missing_helper_function(tmp_path: Path) -> None:
    """When the helper function is removed from preflight.py, gate raises."""
    # Build a fake repo root with a stripped-down preflight.py
    repo = tmp_path / "fake_repo"
    (repo / "src" / "tac").mkdir(parents=True)
    fake_preflight = repo / "src" / "tac" / "preflight.py"
    # Missing _check_233_text_has_prose_negation_for_auth_eval
    fake_preflight.write_text(
        "# stub preflight.py\n"
        "_CHECK_233_PROSE_NEGATION_TOKENS = frozenset({'a', 'b'})\n"
        "_CHECK_233_AUTH_EVAL_100EP_STRUCTURED_PATTERNS = ()\n"
        "_CHECK_233_AUTH_EVAL_100EP_LOOSE_PATTERNS = ()\n"
    )
    violations = check_catalog_233_gate_3_prose_negation_guard_present(
        repo_root=repo, strict=False
    )
    # Should flag the missing helper.
    assert any(
        "_check_233_text_has_prose_negation_for_auth_eval" in v
        for v in violations
    )


def test_catalog_236_flags_missing_token_set(tmp_path: Path) -> None:
    """When the negation token set is removed, gate raises."""
    repo = tmp_path / "fake_repo"
    (repo / "src" / "tac").mkdir(parents=True)
    fake_preflight = repo / "src" / "tac" / "preflight.py"
    fake_preflight.write_text(
        "# stub preflight.py\n"
        "def _check_233_text_has_prose_negation_for_auth_eval(t): pass\n"
        "_CHECK_233_AUTH_EVAL_100EP_STRUCTURED_PATTERNS = ()\n"
        "_CHECK_233_AUTH_EVAL_100EP_LOOSE_PATTERNS = ()\n"
    )
    violations = check_catalog_233_gate_3_prose_negation_guard_present(
        repo_root=repo, strict=False
    )
    assert any("_CHECK_233_PROSE_NEGATION_TOKENS" in v for v in violations)


def test_catalog_236_strict_raises_on_drift(tmp_path: Path) -> None:
    """Strict mode raises PreflightError on drift."""
    repo = tmp_path / "fake_repo"
    (repo / "src" / "tac").mkdir(parents=True)
    fake_preflight = repo / "src" / "tac" / "preflight.py"
    fake_preflight.write_text("# nothing here\n")
    with pytest.raises(PreflightError) as exc:
        check_catalog_233_gate_3_prose_negation_guard_present(
            repo_root=repo, strict=True
        )
    assert "Catalog #236" in str(exc.value)


def test_catalog_236_silent_when_preflight_missing(tmp_path: Path) -> None:
    """When preflight.py is not at canonical path, gate silently OKs."""
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    violations = check_catalog_233_gate_3_prose_negation_guard_present(
        repo_root=repo, strict=False
    )
    assert violations == []


# ----------------------------------------------------------------------------
# Catalog #237 — BOYD-1 fallback-semantic disambiguator META gate
# ----------------------------------------------------------------------------


def test_catalog_237_live_repo_clean() -> None:
    """Live repo must satisfy Catalog #237 (no drift)."""
    violations = check_d9_fallback_semantic_disambiguator_present(strict=False)
    assert violations == [], (
        f"Catalog #237 live count is {len(violations)} — investigate drift"
    )


def test_catalog_237_strict_silent_on_clean() -> None:
    """Strict mode must NOT raise when repo is clean."""
    violations = check_d9_fallback_semantic_disambiguator_present(strict=True)
    assert violations == []


def test_catalog_237_required_tokens_set_size() -> None:
    """The required-tokens set has the canonical 7 surfaces."""
    assert len(_CHECK_237_REQUIRED_TOKENS) == 7


def test_catalog_237_required_tokens_include_disambiguator_surfaces() -> None:
    """The required tokens cover the canonical disambiguator surface."""
    assert "_CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS" in _CHECK_237_REQUIRED_TOKENS
    assert "_CAPACITY_OVERFLOW_FALLBACKS_PER_CLASS" in _CHECK_237_REQUIRED_TOKENS
    assert "class FallbackReason" in _CHECK_237_REQUIRED_TOKENS
    assert "capacity_overflow: bool" in _CHECK_237_REQUIRED_TOKENS


def test_catalog_237_flags_missing_cheaper_dict(tmp_path: Path) -> None:
    """When _CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS is removed, gate raises."""
    repo = tmp_path / "fake_repo"
    (repo / "src" / "tac").mkdir(parents=True)
    fake_cb = repo / "src" / "tac" / "cost_band_calibration.py"
    fake_cb.write_text(
        "# stub\n"
        "_CAPACITY_OVERFLOW_FALLBACKS_PER_CLASS = {}\n"
        "class FallbackReason:\n"
        "    CHEAPER_ALTERNATIVE = 'cheaper_alternative'\n"
        "    CAPACITY_OVERFLOW = 'capacity_overflow'\n"
        "def _fallback_reason_for(c, p, g): pass\n"
        "def select_provider_for_class(c, *, capacity_overflow: bool = False): pass\n"
    )
    violations = check_d9_fallback_semantic_disambiguator_present(
        repo_root=repo, strict=False
    )
    assert any(
        "_CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS" in v for v in violations
    )


def test_catalog_237_flags_missing_overflow_dict(tmp_path: Path) -> None:
    """When _CAPACITY_OVERFLOW_FALLBACKS_PER_CLASS is removed, gate raises."""
    repo = tmp_path / "fake_repo"
    (repo / "src" / "tac").mkdir(parents=True)
    fake_cb = repo / "src" / "tac" / "cost_band_calibration.py"
    fake_cb.write_text(
        "# stub\n"
        "_CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS = {}\n"
        "class FallbackReason:\n"
        "    CHEAPER_ALTERNATIVE = 'cheaper_alternative'\n"
        "    CAPACITY_OVERFLOW = 'capacity_overflow'\n"
        "def _fallback_reason_for(c, p, g): pass\n"
        "def select_provider_for_class(c, *, capacity_overflow: bool = False): pass\n"
    )
    violations = check_d9_fallback_semantic_disambiguator_present(
        repo_root=repo, strict=False
    )
    assert any(
        "_CAPACITY_OVERFLOW_FALLBACKS_PER_CLASS" in v for v in violations
    )


def test_catalog_237_flags_missing_capacity_overflow_param(tmp_path: Path) -> None:
    """When select_provider_for_class drops capacity_overflow, gate raises."""
    repo = tmp_path / "fake_repo"
    (repo / "src" / "tac").mkdir(parents=True)
    fake_cb = repo / "src" / "tac" / "cost_band_calibration.py"
    fake_cb.write_text(
        "# stub\n"
        "_CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS = {}\n"
        "_CAPACITY_OVERFLOW_FALLBACKS_PER_CLASS = {}\n"
        "class FallbackReason:\n"
        "    CHEAPER_ALTERNATIVE = 'cheaper_alternative'\n"
        "    CAPACITY_OVERFLOW = 'capacity_overflow'\n"
        "def _fallback_reason_for(c, p, g): pass\n"
        "def select_provider_for_class(c): pass\n"  # no capacity_overflow
    )
    violations = check_d9_fallback_semantic_disambiguator_present(
        repo_root=repo, strict=False
    )
    assert any("capacity_overflow: bool" in v for v in violations)


def test_catalog_237_strict_raises_on_drift(tmp_path: Path) -> None:
    """Strict mode raises PreflightError on drift."""
    repo = tmp_path / "fake_repo"
    (repo / "src" / "tac").mkdir(parents=True)
    fake_cb = repo / "src" / "tac" / "cost_band_calibration.py"
    fake_cb.write_text("# nothing here\n")
    with pytest.raises(PreflightError) as exc:
        check_d9_fallback_semantic_disambiguator_present(
            repo_root=repo, strict=True
        )
    assert "Catalog #237" in str(exc.value)


def test_catalog_237_silent_when_cost_band_missing(tmp_path: Path) -> None:
    """When cost_band_calibration.py is not at canonical path, gate silently OKs."""
    repo = tmp_path / "fake_repo"
    repo.mkdir()
    violations = check_d9_fallback_semantic_disambiguator_present(
        repo_root=repo, strict=False
    )
    assert violations == []


def test_catalog_237_disjointness_invariant_enforced() -> None:
    """The live repo's two dicts are disjoint per dispatch class."""
    from tac.cost_band_calibration import (
        _CAPACITY_OVERFLOW_FALLBACKS_PER_CLASS,
        _CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS,
        DISPATCH_CLASSES,
    )
    for cls in DISPATCH_CLASSES:
        cheaper = set(_CHEAPER_ALTERNATIVE_FALLBACKS_PER_CLASS.get(cls, []))
        overflow = set(_CAPACITY_OVERFLOW_FALLBACKS_PER_CLASS.get(cls, []))
        assert cheaper.isdisjoint(overflow), (
            f"Class {cls!r}: same fallback in both dicts"
        )


# ----------------------------------------------------------------------------
# Both gates — orchestrator-callsite regression guard
# ----------------------------------------------------------------------------


def test_catalog_236_237_wired_strict_in_preflight_all() -> None:
    """preflight_all() wires both checks at strict=True.

    This guard prevents a regression where a future refactor of preflight_all()
    silently switches the strict flag back to False, defeating the self-
    protection guarantee.
    """
    import inspect

    from tac.preflight import preflight_all

    src = inspect.getsource(preflight_all)
    # Catalog #236 wire-in
    assert "check_catalog_233_gate_3_prose_negation_guard_present(" in src
    # Find the strict= kwarg near the call
    pos_236 = src.find(
        "check_catalog_233_gate_3_prose_negation_guard_present("
    )
    window_236 = src[pos_236:pos_236 + 200]
    assert "strict=True" in window_236, (
        f"Catalog #236 wire-in is not strict=True: window={window_236!r}"
    )

    # Catalog #237 wire-in
    assert "check_d9_fallback_semantic_disambiguator_present(" in src
    pos_237 = src.find(
        "check_d9_fallback_semantic_disambiguator_present("
    )
    window_237 = src[pos_237:pos_237 + 200]
    assert "strict=True" in window_237, (
        f"Catalog #237 wire-in is not strict=True: window={window_237!r}"
    )
