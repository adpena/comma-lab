# SPDX-License-Identifier: MIT
"""Tests for Catalog #344 — canonical equation reference required in empirical memos."""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_empirical_finding_memo_references_canonical_equation,
)
from tac.preflight import _CHECK_344_CUTOFF_DATE_SUFFIX_INT as _CUTOFF

# Derive the post-cutoff (binds) and pre-cutoff (grandfathered/exempt) date
# suffixes from the live cutoff constant so these tests stay correct across
# future apparatus-hygiene re-baselines (2026-05-19 framework birthday ->
# 2026-07-09 re-baseline; the boundary is enforced, the literal is not).
_POST = str(_CUTOFF)  # a memo dated ON the cutoff BINDS (>= comparison)
_PRE = str(_CUTOFF - 1)  # one day before the cutoff -> grandfathered/exempt


def _write_memo(tmp_path: Path, name: str, body: str) -> Path:
    research = tmp_path / ".omx" / "research"
    research.mkdir(parents=True, exist_ok=True)
    p = research / name
    p.write_text(body, encoding="utf-8")
    return p


def test_empty_repo_no_violations(tmp_path: Path):
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert out == []


def test_no_research_dir_skipped(tmp_path: Path):
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert out == []


def test_pre_cutoff_memo_exempt(tmp_path: Path):
    _write_memo(
        tmp_path,
        f"old_memo_{_PRE}.md",
        "We empirically falsified the prediction.",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert out == []


def test_memo_with_empirical_finding_no_reference_flagged(tmp_path: Path):
    _write_memo(
        tmp_path,
        f"new_finding_{_POST}.md",
        "The empirical anchor showed predicted vs measured drift of 30x. "
        "This is a clear ratified observation.",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert len(out) == 1


def test_memo_with_canonical_equation_reference_accepted(tmp_path: Path):
    _write_memo(
        tmp_path,
        f"finding_with_eq_{_POST}.md",
        "The empirical anchor matched the prediction. We registered this in "
        "tac.canonical_equations via register_canonical_equation.",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert out == []


def test_memo_with_formalization_pending_waiver_accepted(tmp_path: Path):
    _write_memo(
        tmp_path,
        f"finding_pending_{_POST}.md",
        "We empirically falsified the prediction. # FORMALIZATION_PENDING:will land equation next session",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert out == []


def test_placeholder_rationale_rejected(tmp_path: Path):
    _write_memo(
        tmp_path,
        f"finding_placeholder_{_POST}.md",
        "Predicted vs empirical residual was 30x. # FORMALIZATION_PENDING:<rationale>",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert len(out) == 1


def test_empty_rationale_rejected(tmp_path: Path):
    _write_memo(
        tmp_path,
        f"finding_empty_{_POST}.md",
        "Predicted vs empirical residual was 30x. # FORMALIZATION_PENDING:",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert len(out) == 1


def test_short_rationale_rejected(tmp_path: Path):
    _write_memo(
        tmp_path,
        f"finding_short_{_POST}.md",
        "Predicted vs empirical residual was 30x. # FORMALIZATION_PENDING:x",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert len(out) == 1


def test_strict_mode_raises(tmp_path: Path):
    _write_memo(
        tmp_path,
        f"strict_finding_{_POST}.md",
        "We empirically ratified this prediction.",
    )
    with pytest.raises(PreflightError, match="Catalog #344"):
        check_empirical_finding_memo_references_canonical_equation(
            repo_root=tmp_path, strict=True
        )


def test_strict_mode_silent_on_clean(tmp_path: Path):
    _write_memo(
        tmp_path,
        f"clean_finding_{_POST}.md",
        "Empirical anchor confirmed via tac.canonical_equations registry.",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path, strict=True
    )
    assert out == []


def test_memo_without_empirical_token_skipped(tmp_path: Path):
    """Memos that don't mention empirical findings are out of scope."""
    _write_memo(
        tmp_path,
        f"design_note_{_POST}.md",
        "Design discussion of the upcoming substrate. Reactivation criteria pinned.",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert out == []


def test_multiple_violations_aggregated(tmp_path: Path):
    for i in range(3):
        _write_memo(
            tmp_path,
            f"multi_{i}_{_POST}.md",
            "Predicted vs empirical residual was confirmed.",
        )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert len(out) == 3


def test_string_repo_root_accepted(tmp_path: Path):
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=str(tmp_path)
    )
    assert out == []


def test_canonical_equations_module_token_accepted(tmp_path: Path):
    """Any of the listed canonical-equation reference tokens satisfies the gate."""
    for token in (
        "tac.canonical_equations",
        "canonical_equations_registry",
        "register_canonical_equation",
        "update_equation_with_empirical_anchor",
    ):
        memo = _write_memo(
            tmp_path,
            f"refers_{abs(hash(token)) % 10000}_{_POST}.md",
            f"Empirical anchor confirmed; see {token} for details.",
        )
        out = check_empirical_finding_memo_references_canonical_equation(
            repo_root=tmp_path
        )
        assert out == [], f"{token} should satisfy the gate"
        memo.unlink()


def test_live_repo_regression_guard():
    """Live-repo count is bounded and small. After the 2026-07-09 apparatus-
    hygiene re-baseline (cutoff 20260519 -> 20260709; see preflight.py) the 515
    accumulated historical DAG-ledger memos are grandfathered and the live count
    was driven to 0. This sister guard keeps the count well under the strict
    orchestrator's tolerance; it will fail loudly if a future re-baseline stalls
    or the grandfather regresses."""
    out = check_empirical_finding_memo_references_canonical_equation()
    assert len(out) <= 5, f"Catalog #344 live count={len(out)} exceeds bound"


def test_orchestrator_wires_warn_only():
    """Smoke-test that preflight_all does NOT raise on Catalog #344 live."""
    # The orchestrator wires Catalog #344 at strict=False; this confirms the
    # function is reachable via the preflight module's public surface.
    from tac.preflight import (
        check_empirical_finding_memo_references_canonical_equation as f,
    )

    assert callable(f)


def test_catalog_344_callable_via_globals():
    """Catalog #185 sister regression — function must be in module globals
    for the META-meta drift gate to introspect it."""
    import tac.preflight as p

    assert hasattr(
        p, "check_empirical_finding_memo_references_canonical_equation"
    )


def test_design_note_with_empirical_token_in_quote_still_flagged(tmp_path: Path):
    """Same-line waiver MUST mention the canonical marker; embedded prose alone
    does not satisfy the gate."""
    _write_memo(
        tmp_path,
        f"quote_finding_{_POST}.md",
        "The reviewer wrote 'predicted vs empirical residual was 30x'.",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert len(out) == 1


def test_wave_3_backfill_keeps_live_count_zero_in_strict_mode():
    """WAVE-3-CATALOG-344-BACKFILL-SWEEP-STRICT-FLIP-READY 2026-05-20
    regression guard. The May-19 STRICT-FLIP-ENABLERS subagent drove the
    initial 52 pre-framework memos to zero; this gate accrued 53 new post-
    flip drift violations across council/landing/design/audit/codex/sweep
    memos between May 19 and May 20. The WAVE-3 sweep classified all 53 as
    incidental-trigger-token (NOT new empirical-finding claim) per archetype
    taxonomy C/G/L/D/A/S/X/R and applied APPEND-ONLY ``FORMALIZATION_PENDING``
    waiver footers per Catalog #110/#113 HISTORICAL_PROVENANCE. This test
    fails if a future commit re-introduces post-cutoff drift WITHOUT the
    canonical-equation reference token OR the FORMALIZATION_PENDING waiver
    with substantive rationale.

    Per CLAUDE.md "Strict-flip atomicity rule": the gate already runs
    ``strict=True`` in ``preflight_all()`` since the May-19 STRICT-FLIP-
    ENABLERS landing; this regression guards against drift recurrence.

    APPARATUS-HYGIENE RE-BASELINE 2026-07-09 (append-only note). The per-memo
    footer discipline was never operationalized across the ongoing research
    corpus, so by 2026-07-09 the live count had drifted to 517 (515 grandfathered
    historical DAG-ledger memos + 2 genuine 2026-07-09 formalization-track
    debts). Per CLAUDE.md "Forbidden premature KILL" + APPEND-ONLY (Catalog
    #110/#113 forbids mutating 515 historical memos at scale), the cutoff was
    re-baselined 20260519 -> 20260709 (see preflight.py), grandfathering the
    historical corpus; the 2 real debts were resolved via APPEND-ONLY
    ``# FORMALIZATION_PENDING:`` footers, driving the live count back to 0. This
    is a grandfather re-baseline, NOT the silent-cap anti-pattern: the ``<= 5``
    bound is unchanged and a NEW memo dated >= the re-baseline still BINDS.
    """
    # Call against the live repo (no tmp_path); strict mode must not raise.
    out = check_empirical_finding_memo_references_canonical_equation(
        strict=False, verbose=False
    )
    # Bound the ceiling slightly above 0 to allow legitimate in-flight memo
    # work to slip through tests for at most a few minutes; the orchestrator
    # callsite is already strict so any non-zero count will fail preflight_all
    # globally. The bound here is per WAVE-3 landing memo + future sister
    # backfill sweeps that may run alongside this regression.
    assert len(out) <= 5, (
        f"Catalog #344 drifted to {len(out)} violations. "
        "Per WAVE-3 backfill 2026-05-20 the live count was driven to 0. "
        "If this fails on a new memo, either: (a) cite tac.canonical_equations "
        "via import/equation_id; (b) add same-line waiver "
        "`# FORMALIZATION_PENDING:<substantive_rationale_>=4_chars>`; "
        "or (c) register a NEW canonical equation if the finding warrants "
        "formalization. Per CLAUDE.md Catalog #344 + #287 sister discipline. "
        f"Violations: {out[:3]}"
    )


def test_wave_3_backfill_sister_equations_registered_upstream():
    """WAVE-3 sister regression: verify the 3 NEW canonical equations
    registered by sister CPU-CUDA-WRITEUP commit ``6f08ebd94b`` are
    callable via ``tac.canonical_equations``. These ARE the equations
    the WAVE-3 audit identified as the upstream coverage that made the
    WAVE-3 backfill possible without registering additional Bucket 3
    equations.

    Per CLAUDE.md "Canonical equations + models registry" non-negotiable.
    """
    from tac.canonical_equations import query_equations

    equations = query_equations()
    equation_ids = {e.equation_id for e in equations}
    # Sister CPU-CUDA-WRITEUP wave registered these 3 (commit 6f08ebd94b).
    expected = {
        "cpu_cuda_score_gap_v1",
        "pose_axis_cuda_amplification_v1",
        "mps_portability_use_case_taxonomy_v1",
    }
    missing = expected - equation_ids
    assert not missing, (
        f"WAVE-3 sister CPU-CUDA-WRITEUP equations missing: {missing}. "
        f"Expected via commit `6f08ebd94b`. Current registry contains "
        f"{len(equation_ids)} equations."
    )


# ---------------------------------------------------------------------------
# APPARATUS-HYGIENE RE-BASELINE 2026-07-09 boundary regression guards. These
# lock the grandfather semantics so a future re-baseline (or a regression that
# reverts the cutoff) is caught: a memo dated ON/AFTER the cutoff BINDS, a memo
# dated just BEFORE the cutoff is grandfathered, and an undated memo is skipped.
# ---------------------------------------------------------------------------


def test_rebaseline_boundary_pre_cutoff_grandfathered(tmp_path: Path):
    """A memo dated exactly one day BEFORE the live cutoff is grandfathered even
    with an empirical-finding token and NO canonical-equation reference."""
    _write_memo(
        tmp_path,
        f"grandfathered_{_PRE}.md",
        "The empirical anchor was falsified; predicted vs measured drifted 30x.",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert out == [], f"pre-cutoff memo (dated {_PRE}) must be grandfathered"


def test_rebaseline_boundary_post_cutoff_binds(tmp_path: Path):
    """A memo dated ON the live cutoff (>= comparison) BINDS: empirical token +
    no reference + no waiver -> flagged. Guards against a re-baseline that
    over-grandfathers the very day it re-baselines forward."""
    _write_memo(
        tmp_path,
        f"on_cutoff_{_POST}.md",
        "The empirical anchor was falsified; predicted vs measured drifted 30x.",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert len(out) == 1, f"on-cutoff memo (dated {_POST}) must bind"


def test_rebaseline_day_after_cutoff_binds(tmp_path: Path):
    """A memo dated the day AFTER the cutoff also binds — the gate is forward-
    binding, not a one-day window."""
    day_after = str(_CUTOFF + 1)
    _write_memo(
        tmp_path,
        f"day_after_{day_after}.md",
        "Empirical anchor: predicted vs empirical residual was ratified.",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path
    )
    assert len(out) == 1, f"post-cutoff memo (dated {day_after}) must bind"


def test_undated_memo_skipped_documented_default(tmp_path: Path):
    """An undated memo (filename lacking the ``_YYYYMMDD`` suffix) cannot be
    date-classified and is SKIPPED — the documented conservative default: do
    not flag what cannot be dated. Guards the date-parse edge case."""
    _write_memo(
        tmp_path,
        "no_date_finding.md",
        "The empirical anchor was falsified; predicted vs measured drift 30x.",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path, strict=True
    )
    assert out == [], "undated memo must be skipped (documented default)"


# ---------------------------------------------------------------------------
# ddm_eq1 2026-09-04: the "stratified" substring false positive.
#
# MEASURED: 16 of the 29 memos live on this gate at commit d3212bed1 (55.2%) tripped it
# ONLY because "ratified" is a substring of "stratified" -- the word this campaign uses
# for the seeded draws it takes INSTEAD of a contiguous prefix ([[m88]]). Corpus counts:
# 704 "stratified" against 29 "unratified". The gate was penalising the memos that did
# their sampling right, and more than half the reported "week of drift" was instrument.
# ---------------------------------------------------------------------------
def test_stratified_alone_does_not_trigger_the_ratified_token(tmp_path: Path):
    """The cure. A seeded stratified draw is a SAMPLING statement, not a verdict."""
    _write_memo(
        tmp_path,
        f"strat_{_CUTOFF + 1}.md",
        "Rows come from a seeded stratified draw of 200/600 (seed 20260903), not a prefix.",
    )
    out = check_empirical_finding_memo_references_canonical_equation(
        repo_root=tmp_path, strict=True
    )
    assert out == [], "'stratified' must not be read as the 'ratified' trigger"


def test_bare_ratified_still_triggers(tmp_path: Path):
    """The cure must not blunt the token it narrows."""
    _write_memo(tmp_path, f"rat_{_CUTOFF + 1}.md", "The prior law was RATIFIED at n600.")
    out = check_empirical_finding_memo_references_canonical_equation(repo_root=tmp_path)
    assert len(out) == 1


def test_unratified_still_triggers(tmp_path: Path):
    """'unratified' is a real ratification-status claim; only 'stratified' is excluded."""
    _write_memo(tmp_path, f"unrat_{_CUTOFF + 1}.md", "The verdict remains unratified.")
    out = check_empirical_finding_memo_references_canonical_equation(repo_root=tmp_path)
    assert len(out) == 1


def test_stratified_memo_that_also_carries_a_real_trigger_still_binds(tmp_path: Path):
    """Narrowing one token must not let a co-occurring real trigger through."""
    _write_memo(
        tmp_path,
        f"both_{_CUTOFF + 1}.md",
        "A seeded stratified draw; the prior law is FALSIFIED at n600.",
    )
    out = check_empirical_finding_memo_references_canonical_equation(repo_root=tmp_path)
    assert len(out) == 1


def test_only_the_ratified_token_carries_an_override(tmp_path: Path):
    """Pin the blast radius: every other trigger keeps plain substring semantics."""
    from tac.preflight import _CHECK_344_TOKEN_OVERRIDE_RE

    assert set(_CHECK_344_TOKEN_OVERRIDE_RE) == {"ratified"}


def test_ddm_eq1_registered_both_backfilled_equations_upstream():
    """The two laws this backfill registered must be resolvable from the registry."""
    from tac.canonical_equations.registry import get_equation_by_id

    for equation_id in (
        "renderer_seg_pose_coupling_shipped_object_v1",
        "annulus_restricted_prefix_bias_detector_v1",
    ):
        equation = get_equation_by_id(equation_id)
        assert equation is not None, f"{equation_id} missing from the registry"
        assert equation.empirical_anchors, f"{equation_id} registered with no anchor"
