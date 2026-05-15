"""Catalog #239 (BOYD-2 self-protection) tests.

Anchor: R2 ledger BOYD-2 finding (Boyd LOW, 2026-05-14, voice: convex
optimization-feasibility lens). ``classify_dispatch`` numeric inference
branches used CLOSED ``>=`` boundaries against
``PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["full"]`` and
``PER_CLASS_SOFT_COST_CEILING_USD["long_burn"]`` so a borderline 12.0h
wallclock estimate routed to ``long_burn`` (Lightning A100, $50+ class)
instead of ``full`` (Vast.ai 4090, $2-15 class) — a 5-10x cost-class jump
for a 1-second wallclock difference at the boundary.

The fix changes the upgrade boundaries to OPEN ``>`` (strict greater-than)
and self-protects via this STRICT preflight gate.

Lane: lane_r2_low_fix_wave_boyd2_mackay2_20260515.
Memory: feedback_r2_low_fix_wave_boyd2_mackay2_landed_20260515.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.cost_band_calibration import (
    PER_CLASS_SOFT_COST_CEILING_USD,
    PER_CLASS_SOFT_WALLCLOCK_CEILING_HR,
    classify_dispatch,
)
from tac.preflight import (
    PreflightError,
    check_classify_dispatch_no_raw_ge_boundaries,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# ----------------------------------------------------------------------------
# Behavioral tests: classify_dispatch open boundary at upgrade ceiling
# ----------------------------------------------------------------------------


def test_classify_dispatch_full_ceiling_exact_routes_to_full() -> None:
    """At exactly 12.0h (full ceiling) the safer routing is "full".

    Pre-fix, ``>=`` would have returned "long_burn" — a 5-10x cost jump.
    Post-fix, ``>`` returns "full" so operators must explicitly request
    long_burn rather than fall into it from a borderline estimate.
    """
    full_ceiling_sec = PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["full"] * 3600.0
    assert classify_dispatch(estimated_wall_clock_sec=full_ceiling_sec) == "full"


def test_classify_dispatch_full_ceiling_minus_one_routes_to_full() -> None:
    """1 second below the ceiling routes to "full" (clearly below)."""
    full_ceiling_sec = PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["full"] * 3600.0
    assert (
        classify_dispatch(estimated_wall_clock_sec=full_ceiling_sec - 1.0)
        == "full"
    )


def test_classify_dispatch_full_ceiling_plus_one_routes_to_long_burn() -> None:
    """1 second above the ceiling routes to "long_burn" (clearly above)."""
    full_ceiling_sec = PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["full"] * 3600.0
    assert (
        classify_dispatch(estimated_wall_clock_sec=full_ceiling_sec + 1.0)
        == "long_burn"
    )


def test_classify_dispatch_smoke_ceiling_exact_routes_to_smoke() -> None:
    """Smoke ceiling keeps CLOSED ``<=`` (downgrade boundary; safer to
    over-route to the cheaper class). 0.5h exact -> "smoke".
    """
    smoke_ceiling_sec = PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["smoke"] * 3600.0
    assert (
        classify_dispatch(estimated_wall_clock_sec=smoke_ceiling_sec)
        == "smoke"
    )


def test_classify_dispatch_smoke_ceiling_minus_one_routes_to_smoke() -> None:
    """Below smoke ceiling routes to "smoke"."""
    smoke_ceiling_sec = PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["smoke"] * 3600.0
    assert (
        classify_dispatch(estimated_wall_clock_sec=smoke_ceiling_sec - 1.0)
        == "smoke"
    )


def test_classify_dispatch_long_burn_cost_ceiling_exact_routes_to_full() -> None:
    """Cost-side: at the long_burn cost ceiling exactly the safer routing
    is "full" — operators must explicitly request the long_burn class.
    """
    long_burn_cost_ceiling = PER_CLASS_SOFT_COST_CEILING_USD["long_burn"]
    # Note: 100.0 USD exactly. With OPEN ``>`` boundary returns "full".
    assert (
        classify_dispatch(estimated_cost_usd=long_burn_cost_ceiling)
        == "full"
    )


def test_classify_dispatch_long_burn_cost_ceiling_plus_one_routes_to_long_burn() -> None:
    """1 USD above the long_burn cost ceiling routes to "long_burn"."""
    long_burn_cost_ceiling = PER_CLASS_SOFT_COST_CEILING_USD["long_burn"]
    assert (
        classify_dispatch(estimated_cost_usd=long_burn_cost_ceiling + 1.0)
        == "long_burn"
    )


def test_classify_dispatch_legacy_long_burn_via_wallclock_15h() -> None:
    """Pre-existing test passes: 15h wallclock is well above the 12h
    ceiling and routes to "long_burn" with both ``>=`` and ``>``.
    """
    assert classify_dispatch(estimated_wall_clock_sec=15 * 3600.0) == "long_burn"


def test_classify_dispatch_legacy_long_burn_via_cost_120usd() -> None:
    """Pre-existing test passes: 120 USD is above the 100 USD ceiling
    and routes to "long_burn" with both ``>=`` and ``>``.
    """
    assert classify_dispatch(estimated_cost_usd=120.0) == "long_burn"


def test_classify_dispatch_smoke_cost_ceiling_exact_routes_to_smoke() -> None:
    """Smoke cost ceiling keeps CLOSED ``<=`` (downgrade boundary). At
    exactly 2.0 USD routes to "smoke".
    """
    smoke_cost_ceiling = PER_CLASS_SOFT_COST_CEILING_USD["smoke"]
    assert (
        classify_dispatch(estimated_cost_usd=smoke_cost_ceiling) == "smoke"
    )


# ----------------------------------------------------------------------------
# Catalog #239 STRICT preflight gate tests
# ----------------------------------------------------------------------------


def test_check_239_clean_repo_returns_zero() -> None:
    """Live repo MUST have 0 violations at HEAD (BOYD-2 fix landed)."""
    violations = check_classify_dispatch_no_raw_ge_boundaries(
        repo_root=REPO_ROOT, strict=False
    )
    assert violations == [], (
        f"Catalog #239 found unexpected violations at HEAD: {violations}"
    )


def test_check_239_strict_mode_passes_clean() -> None:
    """STRICT mode does not raise on clean repo."""
    # Should not raise.
    check_classify_dispatch_no_raw_ge_boundaries(
        repo_root=REPO_ROOT, strict=True
    )


def test_check_239_detects_synthetic_ge_full_boundary(tmp_path: Path) -> None:
    """Synthetic regression: a fixture with raw ``>=`` against full
    ceiling is flagged.
    """
    src_dir = tmp_path / "src" / "tac"
    src_dir.mkdir(parents=True)
    target = src_dir / "cost_band_calibration.py"
    target.write_text(
        '''
"""Synthetic test fixture."""

PER_CLASS_SOFT_WALLCLOCK_CEILING_HR = {"smoke": 0.5, "full": 12.0}


def classify_dispatch(*, estimated_wall_clock_sec=None):
    if estimated_wall_clock_sec is not None:
        if estimated_wall_clock_sec >= PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["full"] * 3600.0:
            return "long_burn"
    return "full"
''',
        encoding="utf-8",
    )
    violations = check_classify_dispatch_no_raw_ge_boundaries(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1
    assert "PER_CLASS_SOFT_WALLCLOCK_CEILING_HR" in violations[0]
    assert "'full'" in violations[0]


def test_check_239_detects_synthetic_ge_long_burn_cost_boundary(
    tmp_path: Path,
) -> None:
    """Synthetic regression: ``>=`` against long_burn cost ceiling flagged."""
    src_dir = tmp_path / "src" / "tac"
    src_dir.mkdir(parents=True)
    target = src_dir / "cost_band_calibration.py"
    target.write_text(
        '''
PER_CLASS_SOFT_COST_CEILING_USD = {"smoke": 2.0, "long_burn": 100.0}


def classify_dispatch(*, estimated_cost_usd=None):
    if estimated_cost_usd is not None:
        if estimated_cost_usd >= PER_CLASS_SOFT_COST_CEILING_USD["long_burn"]:
            return "long_burn"
    return "full"
''',
        encoding="utf-8",
    )
    violations = check_classify_dispatch_no_raw_ge_boundaries(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1
    assert "PER_CLASS_SOFT_COST_CEILING_USD" in violations[0]
    assert "'long_burn'" in violations[0]


def test_check_239_skips_smoke_le_boundary(tmp_path: Path) -> None:
    """Smoke ``<=`` boundary is OUT of scope (downgrade routing is safe)."""
    src_dir = tmp_path / "src" / "tac"
    src_dir.mkdir(parents=True)
    target = src_dir / "cost_band_calibration.py"
    target.write_text(
        '''
PER_CLASS_SOFT_WALLCLOCK_CEILING_HR = {"smoke": 0.5, "full": 12.0}


def classify_dispatch(*, estimated_wall_clock_sec=None):
    if estimated_wall_clock_sec is not None:
        if estimated_wall_clock_sec <= PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["smoke"] * 3600.0:
            return "smoke"
    return "full"
''',
        encoding="utf-8",
    )
    violations = check_classify_dispatch_no_raw_ge_boundaries(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_check_239_skips_outside_classify_dispatch(tmp_path: Path) -> None:
    """``>=`` outside ``classify_dispatch`` body is NOT flagged."""
    src_dir = tmp_path / "src" / "tac"
    src_dir.mkdir(parents=True)
    target = src_dir / "cost_band_calibration.py"
    target.write_text(
        '''
PER_CLASS_SOFT_WALLCLOCK_CEILING_HR = {"smoke": 0.5, "full": 12.0}


def some_other_function(*, estimated_wall_clock_sec=None):
    if estimated_wall_clock_sec is not None:
        if estimated_wall_clock_sec >= PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["full"] * 3600.0:
            return "outside scope"
    return "ok"


def classify_dispatch(*, estimated_wall_clock_sec=None):
    if estimated_wall_clock_sec is not None:
        if estimated_wall_clock_sec > PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["full"] * 3600.0:
            return "long_burn"
    return "full"
''',
        encoding="utf-8",
    )
    violations = check_classify_dispatch_no_raw_ge_boundaries(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_check_239_honors_same_line_waiver(tmp_path: Path) -> None:
    """Same-line waiver with real rationale is honored."""
    src_dir = tmp_path / "src" / "tac"
    src_dir.mkdir(parents=True)
    target = src_dir / "cost_band_calibration.py"
    target.write_text(
        '''
PER_CLASS_SOFT_WALLCLOCK_CEILING_HR = {"smoke": 0.5, "full": 12.0}


def classify_dispatch(*, estimated_wall_clock_sec=None):
    if estimated_wall_clock_sec is not None:
        if estimated_wall_clock_sec >= PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["full"] * 3600.0:  # CLASSIFY_DISPATCH_GE_BOUNDARY_OK:legacy operator-approved closed boundary
            return "long_burn"
    return "full"
''',
        encoding="utf-8",
    )
    violations = check_classify_dispatch_no_raw_ge_boundaries(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_check_239_rejects_placeholder_waiver(tmp_path: Path) -> None:
    """Placeholder ``<reason>`` literal is rejected (cannot self-waive)."""
    src_dir = tmp_path / "src" / "tac"
    src_dir.mkdir(parents=True)
    target = src_dir / "cost_band_calibration.py"
    target.write_text(
        '''
PER_CLASS_SOFT_WALLCLOCK_CEILING_HR = {"smoke": 0.5, "full": 12.0}


def classify_dispatch(*, estimated_wall_clock_sec=None):
    if estimated_wall_clock_sec is not None:
        if estimated_wall_clock_sec >= PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["full"] * 3600.0:  # CLASSIFY_DISPATCH_GE_BOUNDARY_OK:<reason>
            return "long_burn"
    return "full"
''',
        encoding="utf-8",
    )
    violations = check_classify_dispatch_no_raw_ge_boundaries(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1
    assert "PER_CLASS_SOFT_WALLCLOCK_CEILING_HR" in violations[0]


def test_check_239_strict_raises_on_violation(tmp_path: Path) -> None:
    """STRICT mode raises ``PreflightError`` with Catalog #239 message."""
    src_dir = tmp_path / "src" / "tac"
    src_dir.mkdir(parents=True)
    target = src_dir / "cost_band_calibration.py"
    target.write_text(
        '''
PER_CLASS_SOFT_WALLCLOCK_CEILING_HR = {"smoke": 0.5, "full": 12.0}


def classify_dispatch(*, estimated_wall_clock_sec=None):
    if estimated_wall_clock_sec is not None:
        if estimated_wall_clock_sec >= PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["full"] * 3600.0:
            return "long_burn"
    return "full"
''',
        encoding="utf-8",
    )
    with pytest.raises(PreflightError, match="Catalog #239"):
        check_classify_dispatch_no_raw_ge_boundaries(
            repo_root=tmp_path, strict=True
        )


def test_check_239_missing_target_file_returns_clean(tmp_path: Path) -> None:
    """No cost_band_calibration.py present -> 0 violations."""
    violations = check_classify_dispatch_no_raw_ge_boundaries(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_check_239_string_repo_root(tmp_path: Path) -> None:
    """``repo_root`` accepts ``str`` (not just ``Path``)."""
    violations = check_classify_dispatch_no_raw_ge_boundaries(
        repo_root=str(tmp_path), strict=False
    )
    assert violations == []


def test_check_239_eval_class_also_in_scope(tmp_path: Path) -> None:
    """``eval`` is an UPGRADE class (cost increases vs canonical default).

    Per Catalog #239 ``_CHECK_239_UPGRADE_CLASSES``: ``full`` /
    ``long_burn`` / ``eval`` are all upgrade classes; smoke is the only
    downgrade class.
    """
    src_dir = tmp_path / "src" / "tac"
    src_dir.mkdir(parents=True)
    target = src_dir / "cost_band_calibration.py"
    target.write_text(
        '''
PER_CLASS_SOFT_COST_CEILING_USD = {"smoke": 2.0, "eval": 5.0, "full": 15.0}


def classify_dispatch(*, estimated_cost_usd=None):
    if estimated_cost_usd is not None:
        if estimated_cost_usd >= PER_CLASS_SOFT_COST_CEILING_USD["eval"]:
            return "eval"
    return "full"
''',
        encoding="utf-8",
    )
    violations = check_classify_dispatch_no_raw_ge_boundaries(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 1
    assert "'eval'" in violations[0]


def test_check_239_multiple_violations_aggregated(tmp_path: Path) -> None:
    """Multiple ``>=`` upgrade boundaries are all reported."""
    src_dir = tmp_path / "src" / "tac"
    src_dir.mkdir(parents=True)
    target = src_dir / "cost_band_calibration.py"
    target.write_text(
        '''
PER_CLASS_SOFT_WALLCLOCK_CEILING_HR = {"smoke": 0.5, "full": 12.0}
PER_CLASS_SOFT_COST_CEILING_USD = {"smoke": 2.0, "long_burn": 100.0}


def classify_dispatch(*, estimated_wall_clock_sec=None, estimated_cost_usd=None):
    if estimated_wall_clock_sec is not None:
        if estimated_wall_clock_sec >= PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["full"] * 3600.0:
            return "long_burn"
    if estimated_cost_usd is not None:
        if estimated_cost_usd >= PER_CLASS_SOFT_COST_CEILING_USD["long_burn"]:
            return "long_burn"
    return "full"
''',
        encoding="utf-8",
    )
    violations = check_classify_dispatch_no_raw_ge_boundaries(
        repo_root=tmp_path, strict=False
    )
    assert len(violations) == 2


def test_check_239_unrelated_dict_not_flagged(tmp_path: Path) -> None:
    """``>=`` against an unrelated dict (not a canonical ceiling) NOT flagged."""
    src_dir = tmp_path / "src" / "tac"
    src_dir.mkdir(parents=True)
    target = src_dir / "cost_band_calibration.py"
    target.write_text(
        '''
SOME_OTHER_DICT = {"full": 999.0}


def classify_dispatch(*, x=None):
    if x is not None:
        if x >= SOME_OTHER_DICT["full"]:
            return "long_burn"
    return "full"
''',
        encoding="utf-8",
    )
    violations = check_classify_dispatch_no_raw_ge_boundaries(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_check_239_smoke_cost_ceiling_le_not_flagged(tmp_path: Path) -> None:
    """``<= PER_CLASS_SOFT_COST_CEILING_USD["smoke"]`` is downgrade, OUT scope."""
    src_dir = tmp_path / "src" / "tac"
    src_dir.mkdir(parents=True)
    target = src_dir / "cost_band_calibration.py"
    target.write_text(
        '''
PER_CLASS_SOFT_COST_CEILING_USD = {"smoke": 2.0}


def classify_dispatch(*, estimated_cost_usd=None):
    if estimated_cost_usd is not None:
        if estimated_cost_usd <= PER_CLASS_SOFT_COST_CEILING_USD["smoke"]:
            return "smoke"
    return "full"
''',
        encoding="utf-8",
    )
    violations = check_classify_dispatch_no_raw_ge_boundaries(
        repo_root=tmp_path, strict=False
    )
    assert violations == []


def test_check_239_orchestrator_callsite_strict_true() -> None:
    """Regression guard: the orchestrator wires Catalog #239 strict=True.

    Per CLAUDE.md "Strict-flip atomicity rule" — the gate landed strict
    from byte one because the BOYD-2 fix lands in the same commit batch.
    """
    preflight_path = REPO_ROOT / "src" / "tac" / "preflight.py"
    text = preflight_path.read_text(encoding="utf-8")
    # Find the orchestrator wire-in.
    assert "check_classify_dispatch_no_raw_ge_boundaries(" in text
    # Find the strict=True keyword in the same block.
    idx = text.find("check_classify_dispatch_no_raw_ge_boundaries(")
    assert idx != -1
    # Look ahead 200 chars for strict=True.
    snippet = text[idx : idx + 200]
    assert "strict=True" in snippet, (
        "Catalog #239 orchestrator callsite must be strict=True per "
        "Strict-flip atomicity rule"
    )


def test_check_239_classify_dispatch_at_head_uses_strict_gt() -> None:
    """Source-level regression guard: cost_band_calibration.py at HEAD
    uses strict ``>`` for the long_burn upgrade boundary, not ``>=``.

    Sister of the structural Catalog #239 gate — directly checks the
    canonical fix surface.
    """
    target = REPO_ROOT / "src" / "tac" / "cost_band_calibration.py"
    text = target.read_text(encoding="utf-8")
    # The OPEN boundary must be present at the canonical line.
    assert (
        '> PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["full"] * 3600.0' in text
    ), "BOYD-2 fix missing: classify_dispatch wallclock long_burn boundary must use ``>``"
    assert (
        '> PER_CLASS_SOFT_COST_CEILING_USD["long_burn"]' in text
    ), "BOYD-2 fix missing: classify_dispatch cost long_burn boundary must use ``>``"
    # The CLOSED boundary on smoke (downgrade) must remain.
    assert (
        '<= PER_CLASS_SOFT_WALLCLOCK_CEILING_HR["smoke"] * 3600.0' in text
    ), "Smoke wallclock downgrade boundary must remain ``<=``"
    assert (
        '<= PER_CLASS_SOFT_COST_CEILING_USD["smoke"]' in text
    ), "Smoke cost downgrade boundary must remain ``<=``"
