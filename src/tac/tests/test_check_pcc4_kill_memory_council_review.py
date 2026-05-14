# SPDX-License-Identifier: MIT
"""Regression tests for Check PCC4 (KILL/FALSIFIED memory files require
Grand Council adversarial review).

Established 2026-04-30 after the Lane 17 IMP premature-KILL incident:
the agent recorded a KILL verdict at ~22:50 UTC based on a measurement
bug (3.47-second "200-epoch" stub loop). The user's adversarial challenge
("was the IMP results reliable and is that verdict actually hold up
acording to etreme adversarail grand councill") caught the premature
kill before it became durable memory folklore.

CLAUDE.md non-negotiable now mandates that every KILL / FALSIFIED memory
file contain a Grand Council adversarial review with internal-consistency
checks and reactivation criteria. This check enforces that mandate at
preflight time.

Reference: feedback_grand_council_pcc4_kill_memory_review_enforcement_
20260430.md.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_kill_memory_files_have_council_review,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FULL_COUNCIL_BLOCK = """\
## Grand Council adversarial review

- Shannon (LEAD): R(D) bound says this lane was at the entropy floor;
  no further byte savings possible without information loss.
- Dykstra (CO-LEAD): convex-hull intersection rules out further
  Pareto improvement at this rate.
- Yousfi (challenge designer): scoring formula confirms regression.
- Fridrich (steganalysis): texture-budget exhausted; no UNIWARD slack.
- Contrarian: no surviving counter-argument; KILL is correct.
- Quantizr: leaderboard rank unchanged by this lane.
- Hotz: 30-min audit confirms architectural ceiling.

## Internal-consistency check

- elapsed_sec >= epochs * MIN_SEC: VERIFIED
- EMA shadow used at eval: VERIFIED
- auth-eval archive matches submission archive bytes: VERIFIED

## What would change my mind

- New empirical evidence showing PoseNet sensitivity drops below 0.001.
- Hardware FP8 path becomes available (CC >= 10.0).
- Council Round 4+ unanimous reverse vote.
"""


def _write_kill_file(
    tmp_path: Path,
    filename: str,
    *,
    body: str = _FULL_COUNCIL_BLOCK,
    title: str = "Lane test KILL — synthetic regression case",
    description: str = "Synthetic kill record for PCC4 testing.",
) -> Path:
    """Write a kill memory file with frontmatter + body to a tmp memory dir."""
    memory = tmp_path / "memory"
    memory.mkdir(exist_ok=True)
    text = (
        "---\n"
        f"name: {title}\n"
        f"description: {description}\n"
        "type: project\n"
        "originSessionId: test-session-id\n"
        "---\n"
        + body
    )
    (memory / filename).write_text(text)
    return memory


# ---------------------------------------------------------------------------
# Filename-glob detection
# ---------------------------------------------------------------------------


def test_clean_kill_file_with_full_sections_passes(tmp_path: Path) -> None:
    """A `project_*killed*.md` file with all 3 sections passes."""
    mem = _write_kill_file(
        tmp_path, "project_lane_test_killed_cycle_20260430.md",
    )
    violations = check_kill_memory_files_have_council_review(
        strict=False, verbose=False, memory_dir=mem,
    )
    assert violations == []


def test_kill_file_missing_council_header_violates(tmp_path: Path) -> None:
    body = (
        "## Internal-consistency check\n- elapsed_sec sane\n\n"
        "## What would change my mind\n- new evidence\n"
    )
    mem = _write_kill_file(
        tmp_path, "project_lane_x_killed_20260430.md", body=body,
    )
    violations = check_kill_memory_files_have_council_review(
        strict=False, verbose=False, memory_dir=mem,
    )
    assert len(violations) == 1
    assert "Grand Council header" in violations[0]


def test_kill_file_missing_internal_consistency_violates(tmp_path: Path) -> None:
    body = (
        "## Grand Council\n"
        "- Shannon: vote KILL\n- Dykstra: vote KILL\n"
        "- Yousfi: vote KILL\n- Fridrich: vote KILL\n"
        "- Contrarian: vote KILL\n\n"
        "## What would change my mind\n- new evidence\n"
    )
    mem = _write_kill_file(
        tmp_path, "project_lane_x_killed_20260430.md", body=body,
    )
    violations = check_kill_memory_files_have_council_review(
        strict=False, verbose=False, memory_dir=mem,
    )
    assert len(violations) == 1
    assert "internal-consistency" in violations[0]


def test_kill_file_missing_reactivation_violates(tmp_path: Path) -> None:
    body = (
        "## Grand Council\n"
        "- Shannon: vote KILL\n- Dykstra: vote KILL\n"
        "- Yousfi: vote KILL\n- Fridrich: vote KILL\n"
        "- Contrarian: vote KILL\n\n"
        "## Internal-consistency check\n- elapsed_sec sane\n"
    )
    mem = _write_kill_file(
        tmp_path, "project_lane_x_killed_20260430.md", body=body,
    )
    violations = check_kill_memory_files_have_council_review(
        strict=False, verbose=False, memory_dir=mem,
    )
    assert len(violations) == 1
    assert "reactivation" in violations[0]


def test_kill_file_too_few_council_members_violates(tmp_path: Path) -> None:
    body = (
        "## Grand Council\n"
        "- Shannon: vote KILL\n"
        "- Dykstra: vote KILL\n"
        "- Yousfi: vote KILL\n\n"  # only 3 members
        "## Internal-consistency check\n- elapsed_sec sane\n\n"
        "## What would change my mind\n- new evidence\n"
    )
    mem = _write_kill_file(
        tmp_path, "project_lane_x_killed_20260430.md", body=body,
    )
    violations = check_kill_memory_files_have_council_review(
        strict=False, verbose=False, memory_dir=mem,
    )
    assert len(violations) == 1
    assert "3/5" in violations[0] or "council members named" in violations[0]


# ---------------------------------------------------------------------------
# Body-literal detection
# ---------------------------------------------------------------------------


def test_falsified_in_body_triggers_check(tmp_path: Path) -> None:
    """A file with `FALSIFIED` literal but no kill-glob filename still
    must comply."""
    body = (
        "The Lane PD-V2 hypothesis is FALSIFIED on Lane G v3 anchor.\n"
        "No further sections."
    )
    mem = _write_kill_file(
        tmp_path, "project_lane_pd_v2_audit_20260430.md", body=body,
    )
    violations = check_kill_memory_files_have_council_review(
        strict=False, verbose=False, memory_dir=mem,
    )
    assert len(violations) == 1
    assert "project_lane_pd_v2_audit_20260430.md" in violations[0]


def test_verdict_kill_in_body_triggers_check(tmp_path: Path) -> None:
    body = (
        "VERDICT: KILL\n\nNo other sections.\n"
    )
    mem = _write_kill_file(
        tmp_path, "project_lane_y_audit_20260430.md", body=body,
    )
    violations = check_kill_memory_files_have_council_review(
        strict=False, verbose=False, memory_dir=mem,
    )
    assert len(violations) == 1


def test_dead_alone_does_not_trigger(tmp_path: Path) -> None:
    """`DEAD` is intentionally NOT a trigger literal — too ambiguous
    (matches `dead-flag bug`, `dead resolver`, etc.)."""
    body = (
        "We fixed the dead-flag wiring bug (DEAD code removed).\n"
    )
    mem = _write_kill_file(
        tmp_path, "feedback_dead_flag_bug_20260430.md", body=body,
    )
    violations = check_kill_memory_files_have_council_review(
        strict=False, verbose=False, memory_dir=mem,
    )
    # `feedback_*` doesn't match the kill-filename glob, AND `DEAD`
    # isn't in the body literals. So no violation.
    assert violations == []


# ---------------------------------------------------------------------------
# Auto-pass: explicit user override
# ---------------------------------------------------------------------------


def test_user_override_marker_auto_passes(tmp_path: Path) -> None:
    body = (
        "COUNCIL_REVIEW_SKIPPED_USER_OVERRIDE: testing override path\n"
        "\nNo other sections required.\n"
    )
    mem = _write_kill_file(
        tmp_path, "project_lane_z_killed_20260430.md", body=body,
    )
    violations = check_kill_memory_files_have_council_review(
        strict=False, verbose=False, memory_dir=mem,
    )
    assert violations == []


def test_user_override_with_no_reason_does_not_pass(tmp_path: Path) -> None:
    """Override marker with no reason text does NOT auto-pass."""
    body = (
        "COUNCIL_REVIEW_SKIPPED_USER_OVERRIDE: \n"
        "\nNo other sections.\n"
    )
    mem = _write_kill_file(
        tmp_path, "project_lane_z_killed_20260430.md", body=body,
    )
    violations = check_kill_memory_files_have_council_review(
        strict=False, verbose=False, memory_dir=mem,
    )
    # Empty reason → regex does not match → no auto-pass → check applies.
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Auto-pass: WITHDRAWN in title (the Lane 17 IMP fixture pattern)
# ---------------------------------------------------------------------------


def test_withdrawn_in_title_auto_passes(tmp_path: Path) -> None:
    """A kill that was WITHDRAWN under adversarial scrutiny IS the
    success outcome of this check — auto-pass."""
    body = "(no sections — verdict was withdrawn)"
    mem = _write_kill_file(
        tmp_path, "project_lane_a_killed_20260430.md",
        body=body,
        title="Lane A KILL VERDICT WITHDRAWN — measurement bug, not science result",
    )
    violations = check_kill_memory_files_have_council_review(
        strict=False, verbose=False, memory_dir=mem,
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Auto-pass: legacy grandfather (date < 20260430)
# ---------------------------------------------------------------------------


def test_legacy_file_grandfathered(tmp_path: Path) -> None:
    """A kill file with timestamp suffix < 20260430 is grandfathered."""
    body = "No sections — this is a legacy 2026-04-29 kill record."
    mem = _write_kill_file(
        tmp_path, "project_lane_old_killed_20260429.md", body=body,
    )
    violations = check_kill_memory_files_have_council_review(
        strict=False, verbose=False, memory_dir=mem,
    )
    assert violations == []


def test_2026_04_30_file_NOT_grandfathered(tmp_path: Path) -> None:
    """A kill file dated exactly 2026-04-30 IS subject to the check."""
    body = "No sections."
    mem = _write_kill_file(
        tmp_path, "project_lane_new_killed_20260430.md", body=body,
    )
    violations = check_kill_memory_files_have_council_review(
        strict=False, verbose=False, memory_dir=mem,
    )
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Strict mode + edge cases
# ---------------------------------------------------------------------------


def test_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    body = "No sections."
    mem = _write_kill_file(
        tmp_path, "project_lane_x_killed_20260430.md", body=body,
    )
    with pytest.raises(PreflightError) as exc_info:
        check_kill_memory_files_have_council_review(
            strict=True, verbose=False, memory_dir=mem,
        )
    assert "GRAND COUNCIL REVIEW" in str(exc_info.value)
    assert "COUNCIL_REVIEW_SKIPPED_USER_OVERRIDE" in str(exc_info.value)


def test_no_memory_dir_returns_empty(tmp_path: Path) -> None:
    """Edge case: memory_dir does not exist → returns []."""
    nonexistent = tmp_path / "does_not_exist"
    violations = check_kill_memory_files_have_council_review(
        strict=True, verbose=False, memory_dir=nonexistent,
    )
    assert violations == []


def test_empty_memory_dir_returns_empty(tmp_path: Path) -> None:
    """Edge case: memory_dir empty → 0 violations."""
    mem = tmp_path / "memory"
    mem.mkdir()
    violations = check_kill_memory_files_have_council_review(
        strict=True, verbose=False, memory_dir=mem,
    )
    assert violations == []


def test_memory_md_index_file_skipped(tmp_path: Path) -> None:
    """The MEMORY.md index file is ALWAYS skipped (it's a TOC, not a
    kill record)."""
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "MEMORY.md").write_text(
        "FALSIFIED: list of all falsified lanes\n"
        "RETIRED: list of all retired lanes\n"
    )
    violations = check_kill_memory_files_have_council_review(
        strict=False, verbose=False, memory_dir=mem,
    )
    assert violations == []


def test_non_md_files_skipped(tmp_path: Path) -> None:
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "project_lane_x_killed_20260430.txt").write_text("FALSIFIED")
    violations = check_kill_memory_files_have_council_review(
        strict=False, verbose=False, memory_dir=mem,
    )
    assert violations == []


# ---------------------------------------------------------------------------
# The Lane 17 IMP fixture (real file): MUST PASS
# ---------------------------------------------------------------------------


def test_real_lane_17_imp_fixture_passes() -> None:
    """The real Lane 17 IMP kill file at the user's memory dir MUST
    pass — its title says WITHDRAWN, which is the auto-pass outcome
    of this check."""
    fixture = (
        Path.home() / ".claude" / "projects"
        / "-Users-adpena-Projects-pact" / "memory"
        / "project_lane_17_imp_killed_cycle_0_198_regression_20260430.md"
    )
    if not fixture.is_file():
        pytest.skip(f"fixture not present: {fixture}")
    # Audit JUST the parent dir.
    violations = check_kill_memory_files_have_council_review(
        strict=False, verbose=False, memory_dir=fixture.parent,
    )
    # The fixture itself must NOT appear in the violation list (it's
    # WITHDRAWN — auto-passes).
    fixture_violations = [v for v in violations if fixture.name in v]
    assert fixture_violations == [], (
        f"Lane 17 IMP fixture should auto-pass via WITHDRAWN-in-title "
        f"rule, but produced violations: {fixture_violations}"
    )


# ---------------------------------------------------------------------------
# Council deliberation outcomes (encoded as test invariants)
# ---------------------------------------------------------------------------


def test_council_decision_global_memory_dir_default() -> None:
    """Council vote: scan the GLOBAL memory dir (~/.claude/projects/...),
    NOT just repo-local memory. The global dir is where the agent stores
    durable kill records cross-session.

    Verify: function default `memory_dir=None` resolves to a path under
    ~/.claude/projects/-Users-adpena-Projects-pact/memory."""
    from tac.preflight import _PCC4_DEFAULT_MEMORY_DIR

    # Path components must include the user's claude memory layout.
    parts = _PCC4_DEFAULT_MEMORY_DIR.parts
    assert ".claude" in parts
    assert "projects" in parts
    assert "memory" in parts


def test_council_decision_falsified_alone_triggers() -> None:
    """Council vote: FALSIFIED alone IS a kill semantic (it's an
    explicit verdict literal, not an incidental usage). Tested above
    via test_falsified_in_body_triggers_check."""
    from tac.preflight import _PCC4_KILL_BODY_LITERALS

    assert "FALSIFIED" in _PCC4_KILL_BODY_LITERALS
    assert "RETIRED" in _PCC4_KILL_BODY_LITERALS
    # DEAD intentionally excluded — too ambiguous.
    assert "DEAD" not in _PCC4_KILL_BODY_LITERALS


def test_council_decision_reactivation_canonical_name_variants() -> None:
    """Council vote: multiple canonical names accepted for the
    "what would change my mind" subsection. The canonical headers are
    `## Reactivation criteria`, `## Conditions for retracting`, and
    `## What would change my mind`."""
    from tac.preflight import _PCC4_REACTIVATION_HEADERS

    for h in (
        "## Reactivation criteria",
        "## Conditions for retracting",
        "## What would change my mind",
        "what would change",
    ):
        assert h in _PCC4_REACTIVATION_HEADERS
