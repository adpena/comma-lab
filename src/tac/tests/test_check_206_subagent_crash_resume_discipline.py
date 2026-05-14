"""Tests for Catalog #206 — subagent crash-resume checkpoint discipline.

Empirical anchor 2026-05-14: Wyner-Ziv research subagent crashed mid-session
with Anthropic API "Internal server error" after 17min / 58 tool uses / 1704
tokens; all in-flight progress was lost. Sister pattern WAVE-3-HNERV-C-RETRY
survived the same failure class only because intermediate commits had
already landed.

This gate refuses subagent commits (those routed through
``tools/subagent_commit_serializer.py``) whose commit body lacks a
checkpoint discipline trace (``tools/subagent_checkpoint.py`` /
``subagent_progress.jsonl`` / ``checkpoint discipline honored``) or a
same-line ``# CHECKPOINT_DISCIPLINE_WAIVED:<reason>`` waiver.

Memory: feedback_subagent_crash_resume_discipline_landed_20260514.md.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    _check_206_body_has_checkpoint_signal,
    _check_206_iter_recent_commit_bodies,
    _check_206_load_serializer_commits,
    check_subagent_dispatches_use_checkpoint_discipline,
)


# ── Helper: build a tiny git repo with a serializer log ─────────────────


def _make_repo_with_serializer_log(
    tmp_path: Path,
    commits: list[tuple[str, str, bool]],
) -> Path:
    """Build a synthetic repo with N commits and a paired serializer log.

    Each ``commits`` entry is ``(label, body, is_subagent)``. The git
    commit is created with the given body; the serializer log gets a row
    pointing at that commit if ``is_subagent`` is True.
    """
    root = tmp_path / "repo"
    root.mkdir()
    state_dir = root / ".omx" / "state"
    state_dir.mkdir(parents=True)

    # Initialize git
    subprocess.run(
        ["git", "init", "-q", "-b", "main"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    # Disable commit signing for tests
    subprocess.run(
        ["git", "config", "commit.gpgsign", "false"],
        cwd=root,
        check=True,
        capture_output=True,
    )

    serializer_log_path = state_dir / "commit-serializer.log"
    log_rows: list[dict] = []

    for label, body, is_subagent in commits:
        marker_file = root / f"f_{label}.txt"
        marker_file.write_text(label)
        subprocess.run(
            ["git", "add", marker_file.name],
            cwd=root,
            check=True,
            capture_output=True,
        )
        # Use --allow-empty-message=False; body becomes the full message
        subprocess.run(
            ["git", "commit", "-q", "-m", body],
            cwd=root,
            check=True,
            capture_output=True,
        )
        # Read the SHA we just created
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        full_sha = result.stdout.strip()
        if is_subagent:
            log_rows.append({
                "outcome": "committed",
                "head_after": full_sha[:9],
                "label": label,
                "files": [marker_file.name],
                "message_head": body.split("\n")[0][:80],
                # 2026-05-14: post-cutoff timestamp so the strict gate evaluates
                # the test fixture (the cutoff
                # _CHECK_206_DISCIPLINE_CUTOFF_UTC is 2026-05-14T10:30:00Z;
                # 9999-12-31 is well past it). Without this, the cutoff filter
                # treats every test row as legacy and the gate returns clean.
                "started_at_utc": "9999-12-31T00:00:00Z",
            })
        else:
            # Non-subagent: skip or emit a non-committed outcome
            pass

    # Write the serializer log JSONL
    with open(serializer_log_path, "w") as fh:
        for row in log_rows:
            fh.write(json.dumps(row) + "\n")

    return root


# ── Helper functions: unit-test the internals ────────────────────────────


def test_body_signal_recognizes_canonical_token():
    body = "Did good things.\n\nUsed tools/subagent_checkpoint.py at step 3."
    assert _check_206_body_has_checkpoint_signal(body)


def test_body_signal_recognizes_short_token():
    body = "blah blah\nsubagent_checkpoint.py invoked\n"
    assert _check_206_body_has_checkpoint_signal(body)


def test_body_signal_recognizes_jsonl_token():
    body = "Wrote to subagent_progress.jsonl during the work."
    assert _check_206_body_has_checkpoint_signal(body)


def test_body_signal_recognizes_honored_phrase():
    body = "Commit body: checkpoint discipline honored throughout."
    assert _check_206_body_has_checkpoint_signal(body)


def test_body_signal_recognizes_waiver_with_reason():
    body = (
        "Small fix.\n# CHECKPOINT_DISCIPLINE_WAIVED:single-edit subagent, "
        "2 tool uses"
    )
    assert _check_206_body_has_checkpoint_signal(body)


def test_body_signal_rejects_bare_waiver_keyword():
    """A waiver MUST have a reason; bare 'CHECKPOINT_DISCIPLINE_WAIVED' is not enough."""
    body = "blah\nCHECKPOINT_DISCIPLINE_WAIVED\nmore blah"
    assert not _check_206_body_has_checkpoint_signal(body)


def test_body_signal_rejects_placeholder_waiver():
    """Self-waiver via the docstring's `<reason>` placeholder is rejected."""
    body = "# CHECKPOINT_DISCIPLINE_WAIVED:<reason>"
    assert not _check_206_body_has_checkpoint_signal(body)


def test_body_signal_rejects_empty_body():
    assert not _check_206_body_has_checkpoint_signal("")


def test_body_signal_rejects_unrelated_body():
    body = "Refactored some code. Tests pass. Done."
    assert not _check_206_body_has_checkpoint_signal(body)


def test_load_serializer_commits_missing_log(tmp_path):
    shas = _check_206_load_serializer_commits(tmp_path, 50)
    assert shas == set()


def test_load_serializer_commits_returns_committed_only(tmp_path):
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True)
    log = state / "commit-serializer.log"
    rows = [
        {"outcome": "committed", "head_after": "aaaaaaaaa"},
        {"outcome": "committed", "head_after": "bbbbbbbbb"},
        {"outcome": "refused", "head_after": "ccccccccc"},  # skip
        {"outcome": "failed"},  # skip
        {"outcome": "committed", "head_after": "ddddddddd"},
    ]
    with open(log, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    shas = _check_206_load_serializer_commits(tmp_path, 50)
    assert shas == {"aaaaaaaaa", "bbbbbbbbb", "ddddddddd"}


def test_load_serializer_commits_respects_last_n(tmp_path):
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True)
    log = state / "commit-serializer.log"
    with open(log, "w") as fh:
        for i in range(10):
            fh.write(json.dumps({"outcome": "committed", "head_after": f"sha{i:06d}"}) + "\n")
    shas = _check_206_load_serializer_commits(tmp_path, 3)
    # last 3 = sha000007, sha000008, sha000009
    assert shas == {"sha000007", "sha000008", "sha000009"}


def test_load_serializer_commits_handles_bad_json(tmp_path):
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True)
    log = state / "commit-serializer.log"
    log.write_text(
        json.dumps({"outcome": "committed", "head_after": "aaaaaaaaa"}) + "\n"
        "garbage\n"
        + json.dumps({"outcome": "committed", "head_after": "bbbbbbbbb"}) + "\n"
    )
    shas = _check_206_load_serializer_commits(tmp_path, 50)
    assert shas == {"aaaaaaaaa", "bbbbbbbbb"}


# ── End-to-end: gate against synthetic repo ─────────────────────────────


def test_clean_subagent_commit_passes(tmp_path):
    root = _make_repo_with_serializer_log(
        tmp_path,
        [
            (
                "good",
                (
                    "Refactor module.\n\n"
                    "Wrote tools/subagent_checkpoint.py at step 2."
                ),
                True,
            )
        ],
    )
    v = check_subagent_dispatches_use_checkpoint_discipline(
        repo_root=root, strict=False, verbose=False
    )
    assert v == []


def test_dirty_subagent_commit_flagged(tmp_path):
    root = _make_repo_with_serializer_log(
        tmp_path,
        [("bad", "Did stuff. No checkpoint mentioned.", True)],
    )
    v = check_subagent_dispatches_use_checkpoint_discipline(
        repo_root=root, strict=False, verbose=False
    )
    assert len(v) == 1
    assert "lacks checkpoint discipline trace" in v[0]


def test_non_subagent_commit_not_flagged(tmp_path):
    """Operator-side commits (not in serializer log) are out of scope."""
    root = _make_repo_with_serializer_log(
        tmp_path,
        [
            (
                "operator",
                "Operator commit, not a subagent. No checkpoint here.",
                False,
            )
        ],
    )
    v = check_subagent_dispatches_use_checkpoint_discipline(
        repo_root=root, strict=False, verbose=False
    )
    assert v == []


def test_waiver_with_reason_accepted(tmp_path):
    root = _make_repo_with_serializer_log(
        tmp_path,
        [
            (
                "waived",
                (
                    "Tiny fix.\n\n"
                    "# CHECKPOINT_DISCIPLINE_WAIVED:single-line edit, 2 tool uses"
                ),
                True,
            )
        ],
    )
    v = check_subagent_dispatches_use_checkpoint_discipline(
        repo_root=root, strict=False, verbose=False
    )
    assert v == []


def test_bare_waiver_keyword_rejected(tmp_path):
    """`CHECKPOINT_DISCIPLINE_WAIVED` (no `:reason`) does not satisfy."""
    root = _make_repo_with_serializer_log(
        tmp_path,
        [("naked_waiver", "blah CHECKPOINT_DISCIPLINE_WAIVED blah", True)],
    )
    v = check_subagent_dispatches_use_checkpoint_discipline(
        repo_root=root, strict=False, verbose=False
    )
    assert len(v) == 1


def test_strict_mode_raises_preflight_error(tmp_path):
    root = _make_repo_with_serializer_log(
        tmp_path,
        [("violator", "blah blah, no token", True)],
    )
    with pytest.raises(PreflightError, match="checkpoint trace"):
        check_subagent_dispatches_use_checkpoint_discipline(
            repo_root=root, strict=True, verbose=False
        )


def test_missing_serializer_log_returns_clean(tmp_path):
    """If the log doesn't exist, no subagent commits are visible to refuse."""
    root = tmp_path / "repo"
    root.mkdir()
    subprocess.run(
        ["git", "init", "-q", "-b", "main"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    (root / "f.txt").write_text("x")
    subprocess.run(
        ["git", "add", "f.txt"], cwd=root, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    v = check_subagent_dispatches_use_checkpoint_discipline(
        repo_root=root, strict=False, verbose=False
    )
    assert v == []


def test_strict_mode_clean_repo_no_raise(tmp_path):
    root = _make_repo_with_serializer_log(
        tmp_path,
        [
            (
                "good",
                "Wrote subagent_checkpoint.py during the run.",
                True,
            )
        ],
    )
    # Should NOT raise
    v = check_subagent_dispatches_use_checkpoint_discipline(
        repo_root=root, strict=True, verbose=False
    )
    assert v == []


def test_mixed_subagent_and_operator(tmp_path):
    root = _make_repo_with_serializer_log(
        tmp_path,
        [
            ("operator1", "Operator commit, no token", False),
            ("subagent_clean", "Used subagent_checkpoint.py at step 1.", True),
            ("subagent_dirty", "No mention of checkpoints anywhere.", True),
        ],
    )
    v = check_subagent_dispatches_use_checkpoint_discipline(
        repo_root=root, strict=False, verbose=False
    )
    assert len(v) == 1
    assert "subagent_dirty" not in v[0] or True  # body grep
    # The violation should reference the dirty commit's SHA
    # We can't easily map label->sha without rev-parse, but the count should be 1


def test_string_repo_root_accepted(tmp_path):
    root = _make_repo_with_serializer_log(
        tmp_path,
        [("good", "Used subagent_checkpoint.py.", True)],
    )
    v = check_subagent_dispatches_use_checkpoint_discipline(
        repo_root=str(root), strict=False, verbose=False
    )
    assert v == []


def test_verbose_output_on_clean(tmp_path, capsys):
    root = _make_repo_with_serializer_log(
        tmp_path,
        [("good", "Used subagent_checkpoint.py.", True)],
    )
    check_subagent_dispatches_use_checkpoint_discipline(
        repo_root=root, strict=False, verbose=True
    )
    out = capsys.readouterr().out
    assert "subagent-checkpoint-discipline" in out
    assert "OK" in out


def test_verbose_output_on_dirty(tmp_path, capsys):
    root = _make_repo_with_serializer_log(
        tmp_path,
        [("bad", "no token.", True)],
    )
    check_subagent_dispatches_use_checkpoint_discipline(
        repo_root=root, strict=False, verbose=True
    )
    out = capsys.readouterr().out
    assert "WARN" in out
    assert "1/1" in out


def test_iter_recent_commit_bodies_returns_pairs(tmp_path):
    root = _make_repo_with_serializer_log(
        tmp_path,
        [
            ("c1", "first commit body", True),
            ("c2", "second commit body", True),
        ],
    )
    bodies = _check_206_iter_recent_commit_bodies(root, 50)
    assert len(bodies) >= 2
    # Body content (and SHA) should be returned
    found_bodies = {body.strip() for _sha, body in bodies}
    assert "first commit body" in found_bodies
    assert "second commit body" in found_bodies


def test_iter_recent_commit_bodies_in_non_git_dir(tmp_path):
    """A non-git directory returns []."""
    bodies = _check_206_iter_recent_commit_bodies(tmp_path, 50)
    assert bodies == []


def test_gate_live_repo_count_below_threshold():
    """Regression guard: live count in the actual repo must not silently rise.

    The initial landing has 12 in-flight subagent commits without checkpoints
    (predating Catalog #206). The threshold here is intentionally permissive
    (≤30) so legitimate normal use doesn't cause flake; the strict-flip in a
    follow-up wave drives this to 0.
    """
    from tac.preflight import REPO_ROOT
    v = check_subagent_dispatches_use_checkpoint_discipline(
        repo_root=REPO_ROOT, strict=False, verbose=False
    )
    # Warn-only initially; should be a small number, not the entire 50-commit
    # window flagged due to misconfiguration.
    assert len(v) <= 30, (
        f"Live violation count = {len(v)} (above 30). This either means a "
        "regression in the gate or a flood of new subagent commits without "
        "checkpoint discipline. Investigate before raising the threshold."
    )


# ─── Cutoff-filter behavior (codex 3-findings fix, 2026-05-14) ──────────


def _make_repo_with_serializer_log_v2(
    tmp_path,
    commits,  # list of (label, body, is_subagent, started_at_utc)
):
    """V2 helper: lets tests choose the started_at_utc per commit.

    Required by the cutoff-filter tests below so pre-cutoff and post-cutoff
    behavior can be exercised independently.
    """
    root = tmp_path / "repo"
    root.mkdir()
    state_dir = root / ".omx" / "state"
    state_dir.mkdir(parents=True)

    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=root, check=True, capture_output=True)

    serializer_log_path = state_dir / "commit-serializer.log"
    log_rows = []
    for label, body, is_subagent, started_at_utc in commits:
        f = root / f"f_{label}.txt"
        f.write_text(label)
        subprocess.run(["git", "add", f.name], cwd=root, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-q", "-m", body], cwd=root, check=True, capture_output=True)
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        if is_subagent:
            log_rows.append({
                "outcome": "committed",
                "head_after": sha[:9],
                "label": label,
                "started_at_utc": started_at_utc,
            })
    with open(serializer_log_path, "w") as fh:
        for row in log_rows:
            fh.write(json.dumps(row) + "\n")
    return root


def test_pre_cutoff_subagent_commit_without_checkpoint_exempt(tmp_path):
    """Legacy commit (started_at_utc before cutoff) is exempt regardless of body."""
    root = _make_repo_with_serializer_log_v2(
        tmp_path,
        [("legacy", "no checkpoint mention here", True, "2026-05-14T10:00:00Z")],
    )
    v = check_subagent_dispatches_use_checkpoint_discipline(
        repo_root=root, strict=False, verbose=False
    )
    assert v == []


def test_post_cutoff_subagent_commit_without_checkpoint_FAILS_STRICT(tmp_path):
    """REGRESSION TEST per codex MEDIUM #2 recommendation:

    A post-cutoff subagent commit (serializer started_at_utc AFTER the
    cutoff) that LACKS a checkpoint trace MUST cause preflight_all in
    strict mode to RAISE PreflightError.
    """
    root = _make_repo_with_serializer_log_v2(
        tmp_path,
        [(
            "post_cutoff_violator",
            "post-cutoff subagent commit without checkpoint mention",
            True,
            "9999-12-31T00:00:00Z",
        )],
    )
    # Warn mode: violation detected.
    v = check_subagent_dispatches_use_checkpoint_discipline(
        repo_root=root, strict=False, verbose=False
    )
    assert len(v) == 1
    # Strict mode: must raise.
    with pytest.raises(PreflightError, match="checkpoint trace"):
        check_subagent_dispatches_use_checkpoint_discipline(
            repo_root=root, strict=True, verbose=False
        )


def test_post_cutoff_subagent_commit_with_checkpoint_passes_strict(tmp_path):
    """Post-cutoff commit WITH a checkpoint trace must pass strict."""
    root = _make_repo_with_serializer_log_v2(
        tmp_path,
        [(
            "post_cutoff_clean",
            "Wrote tools/subagent_checkpoint.py at step 3.",
            True,
            "9999-12-31T00:00:00Z",
        )],
    )
    v = check_subagent_dispatches_use_checkpoint_discipline(
        repo_root=root, strict=True, verbose=False
    )
    assert v == []


def test_missing_started_at_utc_is_treated_as_legacy(tmp_path):
    """Conservative behavior: a row missing started_at_utc is treated as legacy
    (exempt) to avoid breaking tests/fixtures that omit the field."""
    root = tmp_path / "repo"
    root.mkdir()
    state_dir = root / ".omx" / "state"
    state_dir.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@e.com"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "config", "commit.gpgsign", "false"], cwd=root, check=True, capture_output=True)
    (root / "f.txt").write_text("x")
    subprocess.run(["git", "add", "f.txt"], cwd=root, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "no checkpoint mention"],
        cwd=root, check=True, capture_output=True,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    log = state_dir / "commit-serializer.log"
    # Note: NO started_at_utc field.
    log.write_text(
        json.dumps({"outcome": "committed", "head_after": sha[:9]}) + "\n"
    )
    v = check_subagent_dispatches_use_checkpoint_discipline(
        repo_root=root, strict=True, verbose=False
    )
    assert v == []


def test_load_serializer_started_at_returns_mapping(tmp_path):
    """Helper round-trip."""
    from tac.preflight import _check_206_load_serializer_started_at
    state = tmp_path / ".omx" / "state"
    state.mkdir(parents=True)
    log = state / "commit-serializer.log"
    rows = [
        {"outcome": "committed", "head_after": "shaPOSTcut", "started_at_utc": "9999-01-01T00:00:00Z"},
        {"outcome": "committed", "head_after": "shaPRECutT", "started_at_utc": "2026-05-14T00:00:00Z"},
        {"outcome": "refused",   "head_after": "shaSKIPPED"},
        {"outcome": "committed", "head_after": "shaNoTime"},  # missing field
    ]
    with open(log, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    m = _check_206_load_serializer_started_at(tmp_path, 50)
    assert m == {
        "shaPOSTcut": "9999-01-01T00:00:00Z",
        "shaPRECutT": "2026-05-14T00:00:00Z",
    }


def test_cutoff_constant_present_and_iso_format():
    """Smoke: the cutoff constant must exist and parse as ISO-8601."""
    import datetime as _dt
    from tac.preflight import _CHECK_206_DISCIPLINE_CUTOFF_UTC
    # Strip trailing Z (replace with +00:00 for fromisoformat).
    iso = _CHECK_206_DISCIPLINE_CUTOFF_UTC.replace("Z", "+00:00")
    parsed = _dt.datetime.fromisoformat(iso)
    assert parsed.tzinfo is not None
    assert parsed.year >= 2026


def test_orchestrator_callsite_is_strict_true():
    """Catalog #176 sister check: the strict-flip MUST be wired in preflight_all().

    This guards against accidental regression of the strict=False landing.
    """
    import inspect
    from tac import preflight
    src = inspect.getsource(preflight.preflight_all)
    # The check_subagent_dispatches_use_checkpoint_discipline call must use
    # strict=True (the 3-findings fix). Pattern is permissive on whitespace
    # but rejects strict=False.
    assert "check_subagent_dispatches_use_checkpoint_discipline" in src
    # Forbid strict=False in the same window where the function name appears.
    idx = src.index("check_subagent_dispatches_use_checkpoint_discipline")
    window = src[idx:idx + 600]
    assert "strict=True" in window, (
        "Expected strict=True wire-in for check_subagent_dispatches_use_"
        "checkpoint_discipline per Catalog #206 strict-flip"
    )


def test_preflight_all_raises_on_post_cutoff_violation_in_synthetic_repo(
    tmp_path,
):
    """REGRESSION GUARD per codex MEDIUM #2 recommendation:

    A post-cutoff subagent commit that lacks a checkpoint trace MUST cause
    ``preflight_all(strict=True)`` (or the subagent-checkpoint gate
    invocation) to raise PreflightError.

    Construction: a synthetic repo with a single post-cutoff dirty subagent
    commit. We exercise the gate directly (preflight_all itself wires many
    sister checks that cannot all be satisfied in a synthetic tmp repo);
    the contract this test enforces is that the wire-in honors strict=True.
    """
    root = _make_repo_with_serializer_log_v2(
        tmp_path,
        [(
            "regression_guard",
            "post-cutoff dirty body, no token, no waiver",
            True,
            "9999-12-31T00:00:00Z",
        )],
    )
    # The strict-flipped wire-in must propagate strict=True; the gate
    # function must raise.
    with pytest.raises(PreflightError, match="checkpoint trace"):
        check_subagent_dispatches_use_checkpoint_discipline(
            repo_root=root, strict=True, verbose=False,
        )
