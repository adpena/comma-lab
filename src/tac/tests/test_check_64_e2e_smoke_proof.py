"""Regression tests for Check 64 (canonical E2E smoke proof).

Closes the structural gap surfaced by Lane RM-d's 0.mkv crash: 63 STATIC
preflight checks all guard CODE PATTERNS, none of them actually run the
deploy → inflate → contest_auth_eval pipeline locally to prove a lane
will produce a contest score.

Check 64 enforces that every scripts/remote_lane_*.sh has an entry in
.omx/state/lane_e2e_smoke_proofs.json that is < 7 days old, written by
experiments/canonical_local_auth_eval_smoke.py.

Reference: feedback_canonical_e2e_smoke_PERMANENT_GUARD_20260428.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    SMOKE_PROOFS_REL,
    SMOKE_PROOF_MAX_AGE_DAYS,
    check_lane_scripts_have_e2e_smoke_proof,
)

REPO = Path(__file__).resolve().parents[3]


# ────────────────────────────────────────────────────────────────────────
# Test scaffolding
# ────────────────────────────────────────────────────────────────────────


def _utc_iso(when: dt.datetime) -> str:
    return when.strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_repo(tmp: Path, *, lane_scripts: dict[str, str],
               proofs: dict | None = None) -> Path:
    """Create a fake repo root at `tmp` with the given lane scripts and
    optional proofs file."""
    scripts = tmp / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    for name, body in lane_scripts.items():
        (scripts / name).write_text(body)
    if proofs is not None:
        proofs_path = tmp / SMOKE_PROOFS_REL
        proofs_path.parent.mkdir(parents=True, exist_ok=True)
        proofs_path.write_text(json.dumps(proofs))
    return tmp


# ────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────


def test_check_passes_when_proof_exists_and_recent(tmp_path: Path) -> None:
    """A lane with a fresh smoke proof passes the check."""
    now_iso = _utc_iso(dt.datetime.now(dt.timezone.utc))
    repo = _make_repo(
        tmp_path,
        lane_scripts={"remote_lane_foo.sh": "#!/bin/bash\necho hi\n"},
        proofs={
            "remote_lane_foo": {
                "timestamp_utc": now_iso,
                "archive_sha256": "deadbeef" * 8,
                "stages_passed": ["extract", "whitelist"],
                "tool_version": 1,
            }
        },
    )
    violations = check_lane_scripts_have_e2e_smoke_proof(
        repo_root=repo, strict=True, verbose=False,
    )
    assert violations == []


def test_check_fails_when_proof_missing(tmp_path: Path) -> None:
    """A lane WITHOUT a smoke proof violates the check."""
    repo = _make_repo(
        tmp_path,
        lane_scripts={"remote_lane_bar.sh": "#!/bin/bash\necho hi\n"},
        proofs={},  # empty
    )
    with pytest.raises(MetaBugViolation) as exc_info:
        check_lane_scripts_have_e2e_smoke_proof(
            repo_root=repo, strict=True, verbose=False,
        )
    msg = str(exc_info.value)
    assert "remote_lane_bar.sh" in msg
    assert "no smoke proof" in msg


def test_check_fails_when_proof_too_old(tmp_path: Path) -> None:
    """A proof older than SMOKE_PROOF_MAX_AGE_DAYS days violates."""
    too_old = dt.datetime.now(dt.timezone.utc) - dt.timedelta(
        days=SMOKE_PROOF_MAX_AGE_DAYS + 2,
    )
    repo = _make_repo(
        tmp_path,
        lane_scripts={"remote_lane_baz.sh": "#!/bin/bash\necho hi\n"},
        proofs={
            "remote_lane_baz": {
                "timestamp_utc": _utc_iso(too_old),
                "archive_sha256": "deadbeef" * 8,
                "stages_passed": ["extract"],
                "tool_version": 1,
            }
        },
    )
    with pytest.raises(MetaBugViolation) as exc_info:
        check_lane_scripts_have_e2e_smoke_proof(
            repo_root=repo, strict=True, verbose=False,
        )
    msg = str(exc_info.value)
    assert "too old" in msg
    assert "remote_lane_baz.sh" in msg


def test_check_passes_when_waiver_present(tmp_path: Path) -> None:
    """A lane with `# E2E_SMOKE_OPT_OUT:<reason>` is waived."""
    repo = _make_repo(
        tmp_path,
        lane_scripts={
            "remote_lane_waived.sh": (
                "#!/bin/bash\n"
                "# E2E_SMOKE_OPT_OUT: needs 60GB GPU memory for archive build\n"
                "echo hi\n"
            ),
        },
        proofs={},  # no proof — but waiver should pass it
    )
    violations = check_lane_scripts_have_e2e_smoke_proof(
        repo_root=repo, strict=True, verbose=False,
    )
    assert violations == []


def test_check_skips_when_no_lane_scripts(tmp_path: Path) -> None:
    """A repo with no remote_lane_*.sh scripts at all returns no violations
    (the check has nothing to enforce)."""
    repo = _make_repo(tmp_path, lane_scripts={}, proofs={})
    violations = check_lane_scripts_have_e2e_smoke_proof(
        repo_root=repo, strict=True, verbose=False,
    )
    assert violations == []


def test_check_wired_into_preflight_all() -> None:
    """The check is invoked from the canonical preflight_all() entry point.

    Anchors against a literal `check_lane_scripts_have_e2e_smoke_proof(`
    call site so a future refactor that drops the wire-in fails the test.
    """
    src = (REPO / "src/tac/preflight.py").read_text()
    # The function definition exists.
    assert "def check_lane_scripts_have_e2e_smoke_proof(" in src, (
        "Check 64 function definition missing"
    )
    # AND it's invoked from preflight_all() with strict=True.
    assert "check_lane_scripts_have_e2e_smoke_proof(strict=True" in src, (
        "Check 64 not wired into preflight_all() with strict=True"
    )


# ────────────────────────────────────────────────────────────────────────
# Additional coverage — guards against the bug class
# ────────────────────────────────────────────────────────────────────────


def test_check_rejects_waiver_with_too_short_reason(tmp_path: Path) -> None:
    """`# E2E_SMOKE_OPT_OUT:.` placeholder must NOT pass — at least 4 chars
    of reason required so operators write a real justification."""
    repo = _make_repo(
        tmp_path,
        lane_scripts={
            "remote_lane_lazy.sh": (
                "#!/bin/bash\n"
                "# E2E_SMOKE_OPT_OUT: x\n"  # reason is just "x" — too short
                "echo hi\n"
            ),
        },
        proofs={},
    )
    with pytest.raises(MetaBugViolation):
        check_lane_scripts_have_e2e_smoke_proof(
            repo_root=repo, strict=True, verbose=False,
        )


def test_check_rejects_proof_without_timestamp(tmp_path: Path) -> None:
    """A corrupt proof entry without timestamp_utc is treated as no proof."""
    repo = _make_repo(
        tmp_path,
        lane_scripts={"remote_lane_corrupt.sh": "#!/bin/bash\necho hi\n"},
        proofs={"remote_lane_corrupt": {"archive_sha256": "abc"}},  # no ts
    )
    with pytest.raises(MetaBugViolation) as exc_info:
        check_lane_scripts_have_e2e_smoke_proof(
            repo_root=repo, strict=True, verbose=False,
        )
    assert "missing 'timestamp_utc'" in str(exc_info.value)


def test_check_handles_missing_proofs_file(tmp_path: Path) -> None:
    """A repo with NO proofs file at all is treated as 'every lane unproven'.
    Catches the bootstrap case where an operator runs preflight before ever
    invoking the smoke tool."""
    repo = _make_repo(
        tmp_path,
        lane_scripts={"remote_lane_first.sh": "#!/bin/bash\necho hi\n"},
        proofs=None,  # no file written
    )
    with pytest.raises(MetaBugViolation) as exc_info:
        check_lane_scripts_have_e2e_smoke_proof(
            repo_root=repo, strict=True, verbose=False,
        )
    assert "no smoke proof" in str(exc_info.value)


def test_real_repo_has_no_violations() -> None:
    """The actual codebase must pass Check 64 (post-backfill).

    This is the Lane A → strict promotion guard: by the time Check 64 ships
    STRICT, the backfill has already run against all 70 lane scripts. A
    regression in either the smoke tool OR a lane script's name format
    would surface here.
    """
    violations = check_lane_scripts_have_e2e_smoke_proof(
        repo_root=REPO, strict=False, verbose=False,
    )
    assert violations == [], (
        "Check 64 has live violations on the real repo:\n"
        + "\n".join(f"  • {v}" for v in violations[:10])
    )
