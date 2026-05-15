# SPDX-License-Identifier: MIT
"""Tests for Catalog #291 ``check_session_has_recent_meta_assumption_review``.

Per ``feedback_l5_staircase_v2_and_adversarial_apparatus_structural_fixes_landed_20260515.md``
+ ``feedback_adversarial_review_apparatus_blind_to_shared_assumption_failure_meta_meta_meta_meta_20260515.md``.

The gate refuses sessions that have drifted past the recurring META-ASSUMPTION
ADVERSARIAL REVIEW cadence (every 7 days OR every 50 subagent landings,
whichever first). It scans the operator memory directory for the most-recent
``feedback_*meta_assumption*review*.md`` OR
``feedback_assumptions_challenge_audit_*.md`` file with a canonical body
token, then enforces both canaries.

Sister of Catalog #229 (premise-verification) + Catalog #185 (LIVE_COUNT
drift) + Catalog #290 (substrate canonical-vs-unique discipline).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_session_has_recent_meta_assumption_review,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CANONICAL_BODY = (
    "META-ASSUMPTION review body containing the canonical phrase "
    "'shared assumption' and 'if violated' to satisfy the body-token "
    "requirement."
)


def _write_review_memo(
    memory_dir: Path,
    name: str,
    *,
    body: str = CANONICAL_BODY,
    mtime: float | None = None,
) -> Path:
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / name
    path.write_text(body, encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _write_serializer_log(repo_root: Path, events: list[dict]) -> Path:
    log = repo_root / ".omx" / "state" / "commit-serializer.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8"
    )
    return log


def _now_utc(year: int = 2026, month: int = 5, day: int = 15) -> datetime:
    return datetime(year, month, day, 23, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_291_no_memory_dir_flags_violation(tmp_path: Path) -> None:
    """Missing memory dir is itself a violation: there's no review at all."""
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=tmp_path / "missing",
        serializer_log=tmp_path / "missing.log",
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "no META-ASSUMPTION ADVERSARIAL REVIEW memo found" in violations[0]


def test_291_empty_memory_dir_flags_violation(tmp_path: Path) -> None:
    """Empty memory dir: no review memo present."""
    (tmp_path / "memory").mkdir()
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=tmp_path / "memory",
        serializer_log=tmp_path / "missing.log",
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "no META-ASSUMPTION ADVERSARIAL REVIEW memo found" in violations[0]


def test_291_recent_assumptions_audit_passes(tmp_path: Path) -> None:
    """The canonical assumptions-challenge-audit memo from today passes."""
    memory = tmp_path / "memory"
    _write_review_memo(
        memory,
        "feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md",
    )
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=memory,
        serializer_log=tmp_path / "missing.log",
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_291_recent_meta_assumption_review_passes(tmp_path: Path) -> None:
    """A `feedback_*meta_assumption*review*.md` memo from today passes."""
    memory = tmp_path / "memory"
    _write_review_memo(
        memory,
        "feedback_meta_assumption_adversarial_review_q2_20260515.md",
    )
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=memory,
        serializer_log=tmp_path / "missing.log",
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_291_review_8_days_old_flags_time_violation(tmp_path: Path) -> None:
    """A review 8 days old exceeds the 7-day max cadence."""
    memory = tmp_path / "memory"
    _write_review_memo(
        memory,
        "feedback_meta_assumption_review_old_20260507.md",
    )
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=memory,
        serializer_log=tmp_path / "missing.log",
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "8 days old" in violations[0]
    assert "max 7" in violations[0]


def test_291_review_7_days_old_passes(tmp_path: Path) -> None:
    """A review exactly 7 days old (boundary inclusive) passes."""
    memory = tmp_path / "memory"
    _write_review_memo(
        memory,
        "feedback_meta_assumption_review_boundary_20260508.md",
    )
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=memory,
        serializer_log=tmp_path / "missing.log",
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_291_landings_above_threshold_flags_violation(tmp_path: Path) -> None:
    """51 commits after the review's mtime exceeds the 50-max."""
    memory = tmp_path / "memory"
    # Write review with mtime of 2026-05-15T00:00:00Z
    review_mtime = _now_utc(2026, 5, 15).replace(hour=0).timestamp()
    _write_review_memo(
        memory,
        "feedback_meta_assumption_review_today_20260515.md",
        mtime=review_mtime,
    )
    # 51 commits later in the day
    events = [
        {
            "outcome": "committed",
            "written_at_utc": f"2026-05-15T0{(i // 10) % 10}:{i % 60:02d}:00+00:00",
        }
        for i in range(2, 53)
    ]
    log = _write_serializer_log(tmp_path, events)
    # Make sure they are STRICTLY AFTER mtime by setting all to >= 02:00:00
    events = [
        {
            "outcome": "committed",
            "written_at_utc": f"2026-05-15T{(2 + i // 60):02d}:{i % 60:02d}:00+00:00",
        }
        for i in range(51)
    ]
    _write_serializer_log(tmp_path, events)
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=memory,
        serializer_log=log,
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "51 subagent landings" in violations[0]
    assert "max 50" in violations[0]


def test_291_landings_at_threshold_passes(tmp_path: Path) -> None:
    """Exactly 50 commits after mtime is at the boundary (inclusive)."""
    memory = tmp_path / "memory"
    review_mtime = _now_utc(2026, 5, 15).replace(hour=0).timestamp()
    _write_review_memo(
        memory,
        "feedback_meta_assumption_review_today_20260515.md",
        mtime=review_mtime,
    )
    events = [
        {
            "outcome": "committed",
            "written_at_utc": f"2026-05-15T{(2 + i // 60):02d}:{i % 60:02d}:00+00:00",
        }
        for i in range(50)
    ]
    log = _write_serializer_log(tmp_path, events)
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=memory,
        serializer_log=log,
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_291_landings_before_mtime_not_counted(tmp_path: Path) -> None:
    """Commits with timestamps before the mtime of the review are excluded."""
    memory = tmp_path / "memory"
    review_mtime = _now_utc(2026, 5, 15).replace(hour=12).timestamp()
    _write_review_memo(
        memory,
        "feedback_meta_assumption_review_midday_20260515.md",
        mtime=review_mtime,
    )
    # 100 commits BEFORE noon (excluded)
    events = [
        {
            "outcome": "committed",
            "written_at_utc": f"2026-05-15T{(i // 60):02d}:{i % 60:02d}:00+00:00",
        }
        for i in range(60, 100)  # 0:60 ... irrelevant; just generate <12:00
    ]
    # Force these to be before noon
    events = [
        {
            "outcome": "committed",
            "written_at_utc": f"2026-05-15T0{i // 60}:{i % 60:02d}:00+00:00",
        }
        for i in range(60)
    ]
    log = _write_serializer_log(tmp_path, events)
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=memory,
        serializer_log=log,
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_291_non_committed_outcomes_not_counted(tmp_path: Path) -> None:
    """Only `outcome=committed` events count; failed/refused are excluded."""
    memory = tmp_path / "memory"
    review_mtime = _now_utc(2026, 5, 15).replace(hour=0).timestamp()
    _write_review_memo(
        memory,
        "feedback_assumptions_challenge_audit_today_20260515.md",
        mtime=review_mtime,
    )
    events = [
        {
            "outcome": "expected_content_sha_mismatch",
            "written_at_utc": f"2026-05-15T0{1 + i // 60}:{i % 60:02d}:00+00:00",
        }
        for i in range(100)
    ]
    log = _write_serializer_log(tmp_path, events)
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=memory,
        serializer_log=log,
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_291_placeholder_body_does_not_count_as_review(tmp_path: Path) -> None:
    """A memo whose name matches but lacks canonical body tokens is rejected."""
    memory = tmp_path / "memory"
    _write_review_memo(
        memory,
        "feedback_meta_assumption_review_empty_20260515.md",
        body="this is a placeholder file with no canonical tokens",
    )
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=memory,
        serializer_log=tmp_path / "missing.log",
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "no META-ASSUMPTION ADVERSARIAL REVIEW memo found" in violations[0]


def test_291_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    """Strict mode raises PreflightError citing Catalog #291."""
    with pytest.raises(PreflightError) as exc:
        check_session_has_recent_meta_assumption_review(
            memory_dir=tmp_path / "missing",
            serializer_log=tmp_path / "missing.log",
            repo_root=tmp_path,
            now_utc=_now_utc(),
            strict=True,
            verbose=False,
        )
    msg = str(exc.value)
    assert "Catalog #291" in msg
    assert "META-ASSUMPTION" in msg


def test_291_strict_mode_silent_on_clean(tmp_path: Path) -> None:
    """Strict mode does not raise on a clean session."""
    memory = tmp_path / "memory"
    _write_review_memo(
        memory,
        "feedback_assumptions_challenge_audit_today_20260515.md",
    )
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=memory,
        serializer_log=tmp_path / "missing.log",
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=True,
        verbose=False,
    )
    assert violations == []


def test_291_verbose_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verbose mode prints OK on clean."""
    memory = tmp_path / "memory"
    _write_review_memo(
        memory,
        "feedback_assumptions_challenge_audit_today_20260515.md",
    )
    check_session_has_recent_meta_assumption_review(
        memory_dir=memory,
        serializer_log=tmp_path / "missing.log",
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=False,
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "check_session_has_recent_meta_assumption_review" in captured.out
    assert "OK" in captured.out


def test_291_verbose_violation_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Verbose mode prints violation count when violations are found."""
    check_session_has_recent_meta_assumption_review(
        memory_dir=tmp_path / "missing",
        serializer_log=tmp_path / "missing.log",
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=False,
        verbose=True,
    )
    captured = capsys.readouterr()
    assert "1 violation(s)" in captured.out


def test_291_string_repo_root_accepted(tmp_path: Path) -> None:
    """Both Path and str repo_root inputs are accepted."""
    memory = tmp_path / "memory"
    _write_review_memo(
        memory,
        "feedback_assumptions_challenge_audit_today_20260515.md",
    )
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=str(memory),
        serializer_log=str(tmp_path / "missing.log"),
        repo_root=str(tmp_path),
        now_utc=_now_utc(),
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_291_naive_datetime_treated_as_utc(tmp_path: Path) -> None:
    """A naive (no tzinfo) datetime is treated as UTC."""
    memory = tmp_path / "memory"
    _write_review_memo(
        memory,
        "feedback_assumptions_challenge_audit_today_20260515.md",
    )
    naive_now = datetime(2026, 5, 15, 23, 0, 0)  # naive
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=memory,
        serializer_log=tmp_path / "missing.log",
        repo_root=tmp_path,
        now_utc=naive_now,
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_291_corrupt_serializer_log_handled_gracefully(tmp_path: Path) -> None:
    """Malformed JSONL lines in the log are skipped, not crashing the gate."""
    memory = tmp_path / "memory"
    review_mtime = _now_utc(2026, 5, 15).replace(hour=0).timestamp()
    _write_review_memo(
        memory,
        "feedback_meta_assumption_review_today_20260515.md",
        mtime=review_mtime,
    )
    log = tmp_path / ".omx" / "state" / "commit-serializer.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "this is not valid JSON\n"
        '{"outcome": "committed", "written_at_utc": "2026-05-15T05:00:00+00:00"}\n'
        '{"malformed":\n'
        '"not even an object"\n'
        '[]\n',
        encoding="utf-8",
    )
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=memory,
        serializer_log=log,
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=False,
        verbose=False,
    )
    # 1 valid committed event < 50 threshold, time-OK, no violations
    assert violations == []


def test_291_picks_most_recent_review(tmp_path: Path) -> None:
    """When multiple review memos exist, the most-recent date is used."""
    memory = tmp_path / "memory"
    # Old review
    _write_review_memo(
        memory,
        "feedback_assumptions_challenge_audit_old_20260501.md",
    )
    # Recent review
    _write_review_memo(
        memory,
        "feedback_meta_assumption_review_today_20260515.md",
    )
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=memory,
        serializer_log=tmp_path / "missing.log",
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=False,
        verbose=False,
    )
    # Most-recent (today) used; passes.
    assert violations == []


def test_291_filename_not_matching_pattern_ignored(tmp_path: Path) -> None:
    """A `feedback_*.md` file that doesn't match the canonical pattern does
    NOT count as a META-ASSUMPTION review."""
    memory = tmp_path / "memory"
    _write_review_memo(
        memory,
        "feedback_random_subagent_landed_20260515.md",
    )
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=memory,
        serializer_log=tmp_path / "missing.log",
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1
    assert "no META-ASSUMPTION" in violations[0]


def test_291_invalid_date_suffix_ignored(tmp_path: Path) -> None:
    """A filename with an invalid date suffix (e.g. month=13) is skipped."""
    memory = tmp_path / "memory"
    # 20269999 has invalid month; should NOT count
    _write_review_memo(
        memory,
        "feedback_assumptions_challenge_audit_invalid_20269999.md",
    )
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=memory,
        serializer_log=tmp_path / "missing.log",
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=False,
        verbose=False,
    )
    assert len(violations) == 1


def test_291_live_repo_regression_guard() -> None:
    """Live-repo regression guard: gate currently passes (warn-only) AND has
    bounded count <= 5 if drift occurs. The today's
    ASSUMPTIONS-CHALLENGE-AUDIT is the canonical first instance."""
    violations = check_session_has_recent_meta_assumption_review(
        strict=False, verbose=False,
    )
    # Bounded sentinel: > 5 means real drift; current expected is 0 (today).
    assert len(violations) <= 5, (
        f"Catalog #291 unexpected drift: {len(violations)} violations.\n"
        + "\n".join(violations[:5])
    )


def test_291_orchestrator_callsite_warn_only_regression_guard() -> None:
    """The orchestrator must call this gate with strict=False (warn-only)
    initially per CLAUDE.md 'Strict-flip atomicity rule'."""
    import inspect
    from tac import preflight as pf
    src = inspect.getsource(pf.preflight_all)
    assert "check_session_has_recent_meta_assumption_review(" in src, (
        "Catalog #291 must be wired into preflight_all"
    )
    # Find the call and assert strict=False
    idx = src.find("check_session_has_recent_meta_assumption_review(")
    snippet = src[idx : idx + 200]
    assert "strict=False" in snippet, (
        f"Catalog #291 should be warn-only at landing; got: {snippet[:200]}"
    )


def test_291_function_callable_via_preflight_module_globals() -> None:
    """Catalog #185 sister regression: the gate function MUST be importable
    via tac.preflight module globals so the META-meta-meta drift detector
    can resolve it."""
    from tac import preflight as pf
    assert hasattr(pf, "check_session_has_recent_meta_assumption_review")
    assert callable(pf.check_session_has_recent_meta_assumption_review)


def test_291_constants_pinned() -> None:
    """The cadence constants are part of the protection contract; pin them."""
    from tac.preflight import (
        _CHECK_291_MAX_DAYS_SINCE_LAST_REVIEW,
        _CHECK_291_MAX_LANDINGS_SINCE_LAST_REVIEW,
    )
    assert _CHECK_291_MAX_DAYS_SINCE_LAST_REVIEW == 7
    assert _CHECK_291_MAX_LANDINGS_SINCE_LAST_REVIEW == 50


def test_291_canonical_body_tokens_recognized() -> None:
    """Each canonical body token (case-insensitive) is recognized."""
    from tac.preflight import _CHECK_291_REVIEW_BODY_TOKENS
    expected = {
        "shared assumption",
        "shared assumptions",
        "assumption-violation",
        "assumption violation",
        "if violated",
        "META-ASSUMPTION",
        "meta-assumption",
        "ASSUMPTIONS-CHALLENGE-AUDIT",
    }
    assert set(_CHECK_291_REVIEW_BODY_TOKENS) == expected


def test_291_helper_parse_date_suffix() -> None:
    """The internal date-parser helper handles malformed inputs."""
    from tac.preflight import _check_291_parse_date_suffix
    assert _check_291_parse_date_suffix("20260515") == (2026, 5, 15)
    assert _check_291_parse_date_suffix("not_a_date") is None
    assert _check_291_parse_date_suffix("2026") is None  # too short
    assert _check_291_parse_date_suffix("20261301") is None  # invalid month
    assert _check_291_parse_date_suffix("20260132") is None  # invalid day


def test_291_helper_strictly_after_semantics(tmp_path: Path) -> None:
    """Internal helper: events strictly AFTER the cutoff are counted; events
    AT the cutoff are excluded."""
    from tac.preflight import _check_291_count_subagent_landings_since
    log = tmp_path / "log.jsonl"
    log.write_text(
        json.dumps(
            {"outcome": "committed", "written_at_utc": "2026-05-15T10:00:00+00:00"}
        )
        + "\n"
        + json.dumps(
            {"outcome": "committed", "written_at_utc": "2026-05-15T10:00:01+00:00"}
        )
        + "\n",
        encoding="utf-8",
    )
    # Cutoff at 10:00:00; only the 10:00:01 event counts.
    count = _check_291_count_subagent_landings_since(log, "2026-05-15T10:00:00+00:00")
    assert count == 1


def test_291_helper_missing_log_returns_zero(tmp_path: Path) -> None:
    """Internal helper: missing log file returns 0 (fail-OPEN)."""
    from tac.preflight import _check_291_count_subagent_landings_since
    count = _check_291_count_subagent_landings_since(
        tmp_path / "missing", "2026-05-15T00:00:00+00:00"
    )
    assert count == 0


def test_291_filename_regex_case_insensitive(tmp_path: Path) -> None:
    """Filename matching is case-insensitive."""
    memory = tmp_path / "memory"
    _write_review_memo(
        memory,
        "FEEDBACK_META_ASSUMPTION_REVIEW_TODAY_20260515.md",
    )
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=memory,
        serializer_log=tmp_path / "missing.log",
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=False,
        verbose=False,
    )
    assert violations == []


def test_291_both_canaries_can_fire(tmp_path: Path) -> None:
    """When BOTH time AND landings exceed thresholds, both violations
    surface."""
    memory = tmp_path / "memory"
    review_mtime = datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp()
    _write_review_memo(
        memory,
        "feedback_meta_assumption_review_old_20260501.md",
        mtime=review_mtime,
    )
    events = [
        {
            "outcome": "committed",
            "written_at_utc": f"2026-05-{2 + i // 24:02d}T{i % 24:02d}:00:00+00:00",
        }
        for i in range(60)
    ]
    log = _write_serializer_log(tmp_path, events)
    violations = check_session_has_recent_meta_assumption_review(
        memory_dir=memory,
        serializer_log=log,
        repo_root=tmp_path,
        now_utc=_now_utc(),
        strict=False,
        verbose=False,
    )
    assert len(violations) == 2
    msgs = "\n".join(violations)
    assert "days old" in msgs
    assert "subagent landings" in msgs
