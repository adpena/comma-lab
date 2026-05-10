"""Tests for HARDEN-2026-05-08 preflight checks.

Each new check has two tests: one positive (clean tree -> 0 violations on
the live repo) and one negative (synthetic violation in a tmp tree raises
under strict=True). The live-repo positive check guards regressions; the
tmp-tree negative check guards the detector itself.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tac.preflight import (
    MetaBugViolation,
    PreflightError,
    check_137531_candidate_decoder_path_wired,
    check_admm_lagrangian_bisection_convergent,
    check_codec_pipeline_op_order_deterministic,
    check_evidence_row_has_falsification_scope_when_negative,
    check_public_pr_intake_clones_pristine,
    check_per_tensor_K_side_info_matches_decoder_expectation,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# ─── Check: ADMM Lagrangian bisection convergent ─────────────────────────


def test_admm_bisection_check_passes_on_live_repo() -> None:
    """The live tree should have 0 violations — both declared bisection
    tools have the required `for _ in range(N):` cap + tolerance break."""
    violations = check_admm_lagrangian_bisection_convergent(
        strict=False, verbose=False
    )
    assert violations == [], f"live-repo unexpected violations: {violations}"


def test_admm_bisection_strict_passes_on_live_repo() -> None:
    """Strict-mode parity check (raises if any violation)."""
    check_admm_lagrangian_bisection_convergent(strict=True, verbose=False)


# ─── Check: codec pipeline op order deterministic ────────────────────────


def test_codec_pipeline_determinism_check_passes_on_live_repo() -> None:
    violations = check_codec_pipeline_op_order_deterministic(
        strict=False, verbose=False
    )
    assert violations == [], f"live-repo unexpected violations: {violations}"


def test_codec_pipeline_determinism_strict_passes_on_live_repo() -> None:
    check_codec_pipeline_op_order_deterministic(strict=True, verbose=False)


# ─── Check: per-tensor K side-info matches decoder expectation ───────────


def test_per_tensor_K_check_passes_on_live_repo() -> None:
    violations = check_per_tensor_K_side_info_matches_decoder_expectation(
        strict=False, verbose=False
    )
    assert violations == [], f"live-repo unexpected violations: {violations}"


def test_per_tensor_K_strict_passes_on_live_repo() -> None:
    check_per_tensor_K_side_info_matches_decoder_expectation(
        strict=True, verbose=False
    )


# ─── Check: evidence row falsification scope when negative ───────────────


def test_falsification_scope_check_passes_on_live_repo() -> None:
    violations = check_evidence_row_has_falsification_scope_when_negative(
        strict=False, verbose=False
    )
    assert violations == [], f"live-repo unexpected violations: {violations}"


def test_falsification_scope_negative_detected(tmp_path: Path) -> None:
    """A row with family_falsified=False AND a negative/retired marker AND
    missing falsification_scope MUST be detected in strict mode.

    The check requires `negative_or_retired` evidence (verdict / status /
    blockers / grade containing one of: negative / retired / falsified)
    before insisting on a scope — proxy rows that don't claim negativity
    are exempt."""
    (tmp_path / "reports").mkdir(parents=True)
    bad_path = tmp_path / "reports" / "cathedral_autopilot_evidence.jsonl"
    bad_path.write_text(
        json.dumps(
            {
                "technique": "test_bad_row",
                "family_falsified": False,
                "contest_dispatch_verdict": "negative",
                # falsification_scope missing
            }
        )
        + "\n"
    )
    with pytest.raises(PreflightError) as exc_info:
        check_evidence_row_has_falsification_scope_when_negative(
            repo_root=tmp_path, strict=True, verbose=False
        )
    assert "test_bad_row" in str(exc_info.value)
    assert "falsification_scope" in str(exc_info.value)


def test_falsification_scope_proxy_row_without_scope_passes(tmp_path: Path) -> None:
    """A proxy row with family_falsified=False but NO negative-or-retired
    marker is allowed to omit falsification_scope (no class-level claim
    being made)."""
    (tmp_path / "reports").mkdir(parents=True)
    p = tmp_path / "reports" / "cathedral_autopilot_evidence.jsonl"
    p.write_text(
        json.dumps(
            {
                "technique": "neutral_proxy",
                "family_falsified": False,
                # no scope, no negative verdict
            }
        )
        + "\n"
    )
    violations = check_evidence_row_has_falsification_scope_when_negative(
        repo_root=tmp_path, strict=True, verbose=False
    )
    assert violations == []


def test_falsification_scope_true_value_warns(tmp_path: Path) -> None:
    """A row with family_falsified=True (without explicit grand-council
    record) is also a violation per CLAUDE.md "KILL is the LAST RESORT"."""
    (tmp_path / "reports").mkdir(parents=True)
    bad_path = tmp_path / "reports" / "cathedral_autopilot_evidence.jsonl"
    bad_path.write_text(
        json.dumps(
            {
                "technique": "premature_kill",
                "family_falsified": True,
                "falsification_scope": "anything",
            }
        )
        + "\n"
    )
    violations = check_evidence_row_has_falsification_scope_when_negative(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert any("premature_kill" in v for v in violations)
    assert any("LAST RESORT" in v for v in violations)


def test_falsification_scope_clean_passes(tmp_path: Path) -> None:
    """A row with family_falsified=False AND a non-empty falsification_scope
    must pass."""
    (tmp_path / "reports").mkdir(parents=True)
    good_path = tmp_path / "reports" / "cathedral_autopilot_evidence.jsonl"
    good_path.write_text(
        json.dumps(
            {
                "technique": "good_row",
                "family_falsified": False,
                "falsification_scope": "tested_only_config_X_at_rms_Y",
            }
        )
        + "\n"
    )
    violations = check_evidence_row_has_falsification_scope_when_negative(
        repo_root=tmp_path, strict=True, verbose=False
    )
    assert violations == []


# ─── Check: 137531 candidate decoder path wired ──────────────────────────


def test_137531_candidate_check_passes_on_live_repo() -> None:
    violations = check_137531_candidate_decoder_path_wired(
        strict=False, verbose=False
    )
    assert violations == [], f"live-repo unexpected violations: {violations}"


def test_137531_candidate_strict_passes_on_live_repo() -> None:
    check_137531_candidate_decoder_path_wired(strict=True, verbose=False)


def test_public_pr_pristine_lfs_only_dirty_preserves_status_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LFS-only pointer/content dirtiness is benign, including the first
    `git status --short` row where the leading worktree status column is a
    space. A whole-output `.strip()` used to corrupt that first path.
    """
    clone = (
        tmp_path
        / "experiments"
        / "results"
        / "public_pr90_intake_20260505_auto"
        / "source"
    )
    (clone / ".git").mkdir(parents=True)

    def fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output is True
        assert text is True
        assert timeout == 30
        if cmd[:3] == ["git", "-C", str(clone)]:
            subcmd = cmd[3:]
            if subcmd == ["status", "--short"]:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=(
                        " M submissions/qrepro/assets/animated_triptych.gif\n"
                        " M submissions/qrepro/assets/archive_components.png\n"
                    ),
                    stderr="",
                )
            if subcmd == ["lfs", "status", "--porcelain"]:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=(
                        " M submissions/qrepro/assets/animated_triptych.gif\n"
                        " M submissions/qrepro/assets/archive_components.png\n"
                    ),
                    stderr="",
                )
            if subcmd == ["diff", "--numstat"]:
                return subprocess.CompletedProcess(
                    cmd,
                    0,
                    stdout=(
                        "-\t-\tsubmissions/qrepro/assets/animated_triptych.gif\n"
                        "-\t-\tsubmissions/qrepro/assets/archive_components.png\n"
                    ),
                    stderr="",
                )
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="unexpected")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert (
        check_public_pr_intake_clones_pristine(
            repo_root=tmp_path, strict=True, verbose=False
        )
        == []
    )


def test_public_pr_pristine_fans_out_clean_status_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clean clone status checks should run concurrently; dirty clones still
    take the sequential LFS/diff path.
    """
    clones = []
    for idx in range(3):
        clone = (
            tmp_path
            / "experiments"
            / "results"
            / f"public_pr{90 + idx}_intake_20260505_auto"
            / "source"
        )
        (clone / ".git").mkdir(parents=True)
        clones.append(clone)

    seen_thread_names: set[str] = set()

    def fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        import threading

        assert capture_output is True
        assert text is True
        assert timeout == 30
        if len(cmd) >= 5 and cmd[0] == "git" and cmd[3:] == ["status", "--short"]:
            seen_thread_names.add(threading.current_thread().name)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="unexpected")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert (
        check_public_pr_intake_clones_pristine(
            repo_root=tmp_path, strict=True, verbose=False
        )
        == []
    )
    assert len(seen_thread_names) >= 2


def test_public_pr_pristine_discovers_generic_pr_src_repo_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone = (
        tmp_path
        / "experiments"
        / "results"
        / "public_pr_archive_20260505"
        / "public_pr101_intake_20260505_auto"
        / "pr101_src"
        / "repo"
    )
    (clone / ".git").mkdir(parents=True)
    seen_status = False

    def fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal seen_status
        assert capture_output is True
        assert text is True
        assert timeout == 30
        if cmd[:3] == ["git", "-C", str(clone)] and cmd[3:] == ["status", "--short"]:
            seen_status = True
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="unexpected")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert (
        check_public_pr_intake_clones_pristine(
            repo_root=tmp_path, strict=True, verbose=False
        )
        == []
    )
    assert seen_status is True


def test_public_pr_pristine_discovers_nested_public_pr_custody_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone = (
        tmp_path
        / "experiments"
        / "results"
        / "public_pr102_hnerv_lc_v2_custody_20260507_codex"
        / "public_pr102_intake_20260507_auto"
        / "source"
    )
    (clone / ".git").mkdir(parents=True)
    seen_status = False

    def fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal seen_status
        assert capture_output is True
        assert text is True
        assert timeout == 30
        if cmd[:3] == ["git", "-C", str(clone)] and cmd[3:] == ["status", "--short"]:
            seen_status = True
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="unexpected")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert (
        check_public_pr_intake_clones_pristine(
            repo_root=tmp_path, strict=True, verbose=False
        )
        == []
    )
    assert seen_status is True


def test_public_pr_pristine_caches_clean_status_and_invalidates_untracked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone = (
        tmp_path
        / "experiments"
        / "results"
        / "public_pr90_intake_20260505_auto"
        / "source"
    )
    git_dir = clone / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "index").write_bytes(b"index")
    (clone / "submission.py").write_text("print('upstream')\n")

    status_calls = 0

    def fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal status_calls
        assert capture_output is True
        assert text is True
        assert timeout == 30
        if cmd[:3] == ["git", "-C", str(clone)]:
            subcmd = cmd[3:]
            if subcmd == ["status", "--short"]:
                status_calls += 1
                stdout = "?? local_waiver.py\n" if (clone / "local_waiver.py").exists() else ""
                return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
            if subcmd == ["lfs", "status", "--porcelain"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if subcmd == ["diff", "--numstat"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="unexpected")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert (
        check_public_pr_intake_clones_pristine(
            repo_root=tmp_path, strict=True, verbose=False
        )
        == []
    )
    assert status_calls == 1

    assert (
        check_public_pr_intake_clones_pristine(
            repo_root=tmp_path, strict=True, verbose=False
        )
        == []
    )
    assert status_calls == 1

    (clone / "local_waiver.py").write_text("# local waiver must invalidate cache\n")
    violations = check_public_pr_intake_clones_pristine(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert status_calls == 2
    assert any("local_waiver.py" in item for item in violations)


def test_public_pr_pristine_clean_cache_invalidates_tracked_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone = (
        tmp_path
        / "experiments"
        / "results"
        / "public_pr91_intake_20260505_auto"
        / "source"
    )
    git_dir = clone / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "index").write_bytes(b"index")
    tracked = clone / "submission.py"
    tracked.write_text("print('upstream')\n")

    status_calls = 0

    def fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal status_calls
        assert capture_output is True
        assert text is True
        assert timeout == 30
        if cmd[:3] == ["git", "-C", str(clone)]:
            subcmd = cmd[3:]
            if subcmd == ["status", "--short"]:
                status_calls += 1
                stdout = " M submission.py\n" if tracked.read_text() != "print('upstream')\n" else ""
                return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
            if subcmd == ["lfs", "status", "--porcelain"]:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if subcmd == ["diff", "--numstat"]:
                stdout = "1\t1\tsubmission.py\n" if tracked.read_text() != "print('upstream')\n" else ""
                return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="unexpected")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert (
        check_public_pr_intake_clones_pristine(
            repo_root=tmp_path, strict=True, verbose=False
        )
        == []
    )
    assert (
        check_public_pr_intake_clones_pristine(
            repo_root=tmp_path, strict=True, verbose=False
        )
        == []
    )
    assert status_calls == 1

    tracked.write_text("print('local edit')\n")
    violations = check_public_pr_intake_clones_pristine(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert status_calls == 2
    assert any("submission.py" in item for item in violations)


def test_public_pr_pristine_cache_disable_forces_live_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone = (
        tmp_path
        / "experiments"
        / "results"
        / "public_pr92_intake_20260505_auto"
        / "source"
    )
    git_dir = clone / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "index").write_bytes(b"index")
    (clone / "submission.py").write_text("print('upstream')\n")
    status_calls = 0

    def fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal status_calls
        assert capture_output is True
        assert text is True
        assert timeout == 30
        if cmd[:3] == ["git", "-C", str(clone)] and cmd[3:] == ["status", "--short"]:
            status_calls += 1
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="unexpected")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("PACT_PREFLIGHT_DISABLE_INCREMENTAL_CACHE", "1")

    assert (
        check_public_pr_intake_clones_pristine(
            repo_root=tmp_path, strict=True, verbose=False
        )
        == []
    )
    assert (
        check_public_pr_intake_clones_pristine(
            repo_root=tmp_path, strict=True, verbose=False
        )
        == []
    )
    assert status_calls == 2


def test_public_pr_pristine_git_status_timeout_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone = (
        tmp_path
        / "experiments"
        / "results"
        / "public_pr93_intake_20260505_auto"
        / "source"
    )
    (clone / ".git").mkdir(parents=True)

    def fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output is True
        assert text is True
        assert timeout == 30
        if cmd[:3] == ["git", "-C", str(clone)] and cmd[3:] == ["status", "--short"]:
            raise subprocess.TimeoutExpired(cmd, timeout)
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="unexpected")

    monkeypatch.setattr(subprocess, "run", fake_run)
    violations = check_public_pr_intake_clones_pristine(
        repo_root=tmp_path, strict=False, verbose=False
    )

    assert len(violations) == 1
    assert "git status unavailable" in violations[0]
    assert "timed out" in violations[0]
    with pytest.raises(MetaBugViolation, match="git status unavailable"):
        check_public_pr_intake_clones_pristine(
            repo_root=tmp_path, strict=True, verbose=False
        )


def test_public_pr_pristine_git_status_nonzero_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone = (
        tmp_path
        / "experiments"
        / "results"
        / "public_pr94_intake_20260505_auto"
        / "source"
    )
    (clone / ".git").mkdir(parents=True)

    def fake_run(
        cmd: list[str],
        *,
        capture_output: bool,
        text: bool,
        timeout: int,
    ) -> subprocess.CompletedProcess[str]:
        assert capture_output is True
        assert text is True
        assert timeout == 30
        if cmd[:3] == ["git", "-C", str(clone)] and cmd[3:] == ["status", "--short"]:
            return subprocess.CompletedProcess(cmd, 128, stdout="", stderr="fatal: bad gitdir")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="unexpected")

    monkeypatch.setattr(subprocess, "run", fake_run)
    violations = check_public_pr_intake_clones_pristine(
        repo_root=tmp_path, strict=False, verbose=False
    )

    assert len(violations) == 1
    assert "returned 128" in violations[0]
    assert "fatal: bad gitdir" in violations[0]
