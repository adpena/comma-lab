"""Tests for Catalog #147 — Lightning submit `cancel_pending_job_locked`
only safe pre-network.

Bug class: codex round 7+8 HIGH 1 (2026-05-09). The round-6 fix (#143) made
dispatchers register the pending row BEFORE submit, but the cancel-on-exception
logic was too aggressive — `except BaseException` around `submit_lightning_job(...)`
(which includes `Job.run(...)`) could silently delete the only harvester-visible
row for a real paid Lightning job.

Fix: split exception routing — narrow `except _PreNetworkSubmitError` calls
`cancel_pending_job_locked`, residual `except BaseException` calls
`mark_pending_failed_unknown_billing_locked` to preserve the row for forensic
recovery.

Memory: feedback_codex_round78_findings_fix_with_self_protection_landed_20260509.md
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest

from tac.preflight import check_lightning_submit_cancel_only_before_network


def _write(tmp_path: Path, rel: str, source: str) -> Path:
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dedent(source))
    return p


# ── Positive (catches violation) ──────────────────────────────────────────


def test_baseexception_around_submit_with_cancel_is_violation(tmp_path: Path) -> None:
    _write(tmp_path, "experiments/foo_lightning_dispatcher.py", """
        def main():
            register_pending_active_job(job_name="x", machine="t4")
            try:
                submit_result = submit_lightning_job(
                    job_name="x", machine="t4", command="...",
                    teamspace="t", studio="s", user="u",
                    max_runtime_sec=600, dry_run=False,
                )
            except BaseException as exc:
                cancel_pending_job_locked(job_name="x", failure_reason=str(exc))
                raise
    """)
    v = check_lightning_submit_cancel_only_before_network(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1, v
    assert "[Check 147]" in v[0]
    assert "BaseException" in v[0] or "BaseException" in v[0]


def test_bare_except_with_cancel_is_violation(tmp_path: Path) -> None:
    _write(tmp_path, "experiments/bar_lightning.py", """
        def main():
            register_pending_active_job(job_name="x", machine="t4")
            try:
                submit_lightning_job(job_name="x", machine="t4", command="c",
                    teamspace="t", studio="s", user="u",
                    max_runtime_sec=60, dry_run=False)
            except:
                cancel_pending_job_locked(job_name="x")
                raise
    """)
    v = check_lightning_submit_cancel_only_before_network(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1, v


def test_baseexception_in_tuple_with_cancel_is_violation(tmp_path: Path) -> None:
    _write(tmp_path, "experiments/baz_lightning.py", """
        def main():
            register_pending_active_job(job_name="x", machine="t4")
            try:
                submit_lightning_job(job_name="x", machine="t4", command="c",
                    teamspace="t", studio="s", user="u",
                    max_runtime_sec=60, dry_run=False)
            except (BaseException, TypeError) as exc:
                cancel_pending_job_locked(job_name="x")
                raise
    """)
    v = check_lightning_submit_cancel_only_before_network(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1, v


def test_job_run_directly_in_try_with_cancel_is_violation(tmp_path: Path) -> None:
    _write(tmp_path, "experiments/qux_lightning.py", """
        def main():
            register_pending_active_job(job_name="x", machine="t4")
            try:
                Job.run(name="x", machine="t4", command="c", studio="s",
                    teamspace="t", user="u", env={}, interruptible=False,
                    max_runtime=60)
            except BaseException as exc:
                cancel_pending_job_locked(job_name="x")
                raise
    """)
    v = check_lightning_submit_cancel_only_before_network(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1, v


def test_strict_mode_raises_on_violation(tmp_path: Path) -> None:
    from tac.preflight import PreflightError
    _write(tmp_path, "experiments/strict_lightning.py", """
        def main():
            try:
                submit_lightning_job(job_name="x", machine="t4", command="c",
                    teamspace="t", studio="s", user="u",
                    max_runtime_sec=60, dry_run=False)
            except BaseException:
                cancel_pending_job_locked(job_name="x")
    """)
    with pytest.raises(PreflightError) as excinfo:
        check_lightning_submit_cancel_only_before_network(
            repo_root=tmp_path, strict=True, verbose=False,
        )
    assert "147" in str(excinfo.value)


# ── Negative (allows non-violations) ──────────────────────────────────────


def test_pre_network_narrow_handler_with_cancel_is_allowed(tmp_path: Path) -> None:
    _write(tmp_path, "experiments/narrow_lightning.py", """
        class _PreNetworkSubmitError(RuntimeError):
            pass

        def main():
            register_pending_active_job(job_name="x", machine="t4")
            try:
                submit_lightning_job(job_name="x", machine="t4", command="c",
                    teamspace="t", studio="s", user="u",
                    max_runtime_sec=60, dry_run=False)
            except _PreNetworkSubmitError as exc:
                cancel_pending_job_locked(job_name="x")
                raise
    """)
    v = check_lightning_submit_cancel_only_before_network(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == [], v


def test_baseexception_with_mark_unknown_billing_only_is_allowed(tmp_path: Path) -> None:
    _write(tmp_path, "experiments/preserve_lightning.py", """
        def main():
            register_pending_active_job(job_name="x", machine="t4")
            try:
                submit_lightning_job(job_name="x", machine="t4", command="c",
                    teamspace="t", studio="s", user="u",
                    max_runtime_sec=60, dry_run=False)
            except BaseException as exc:
                mark_pending_failed_unknown_billing_locked(
                    job_name="x", failure_reason=str(exc),
                )
                raise
    """)
    v = check_lightning_submit_cancel_only_before_network(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == [], v


def test_split_handler_narrow_then_wide_is_allowed(tmp_path: Path) -> None:
    _write(tmp_path, "experiments/split_lightning.py", """
        class _PreNetworkSubmitError(RuntimeError):
            pass

        def main():
            register_pending_active_job(job_name="x", machine="t4")
            try:
                submit_lightning_job(job_name="x", machine="t4", command="c",
                    teamspace="t", studio="s", user="u",
                    max_runtime_sec=60, dry_run=False)
            except _PreNetworkSubmitError as pre_exc:
                cancel_pending_job_locked(job_name="x")
                raise
            except BaseException as submit_exc:
                mark_pending_failed_unknown_billing_locked(
                    job_name="x", failure_reason=str(submit_exc),
                )
                raise
    """)
    v = check_lightning_submit_cancel_only_before_network(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == [], v


def test_no_submit_in_try_block_is_allowed(tmp_path: Path) -> None:
    _write(tmp_path, "experiments/unrelated_lightning.py", """
        def main():
            try:
                some_other_call()
            except BaseException:
                cancel_pending_job_locked(job_name="x")
                raise
    """)
    v = check_lightning_submit_cancel_only_before_network(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == [], v


def test_submit_in_try_but_no_cancel_in_handler_is_allowed(tmp_path: Path) -> None:
    _write(tmp_path, "experiments/no_cancel_lightning.py", """
        def main():
            try:
                submit_lightning_job(job_name="x", machine="t4", command="c",
                    teamspace="t", studio="s", user="u",
                    max_runtime_sec=60, dry_run=False)
            except BaseException as exc:
                logger.error("submit failed: %s", exc)
                raise
    """)
    v = check_lightning_submit_cancel_only_before_network(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == [], v


# ── Waiver respect ────────────────────────────────────────────────────────


def test_waiver_on_cancel_line_is_respected(tmp_path: Path) -> None:
    _write(tmp_path, "experiments/waiver_lightning.py", """
        def main():
            try:
                submit_lightning_job(job_name="x", machine="t4", command="c",
                    teamspace="t", studio="s", user="u",
                    max_runtime_sec=60, dry_run=False)
            except BaseException as exc:
                cancel_pending_job_locked(job_name="x")  # CANCEL_PENDING_PRE_NETWORK_OK:test-fixture
                raise
    """)
    v = check_lightning_submit_cancel_only_before_network(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == [], v


def test_waiver_on_try_line_is_respected(tmp_path: Path) -> None:
    _write(tmp_path, "experiments/waiver_try_lightning.py", """
        def main():
            try:  # CANCEL_PENDING_PRE_NETWORK_OK:test-fixture
                submit_lightning_job(job_name="x", machine="t4", command="c",
                    teamspace="t", studio="s", user="u",
                    max_runtime_sec=60, dry_run=False)
            except BaseException:
                cancel_pending_job_locked(job_name="x")
                raise
    """)
    v = check_lightning_submit_cancel_only_before_network(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == [], v


def test_waiver_on_except_line_is_respected(tmp_path: Path) -> None:
    _write(tmp_path, "experiments/waiver_except_lightning.py", """
        def main():
            try:
                submit_lightning_job(job_name="x", machine="t4", command="c",
                    teamspace="t", studio="s", user="u",
                    max_runtime_sec=60, dry_run=False)
            except BaseException:  # CANCEL_PENDING_PRE_NETWORK_OK:test
                cancel_pending_job_locked(job_name="x")
                raise
    """)
    v = check_lightning_submit_cancel_only_before_network(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == [], v


# ── Edge cases ────────────────────────────────────────────────────────────


def test_non_lightning_files_are_skipped(tmp_path: Path) -> None:
    _write(tmp_path, "experiments/not_a_dispatcher.py", """
        def main():
            try:
                submit_lightning_job(job_name="x", machine="t4", command="c",
                    teamspace="t", studio="s", user="u",
                    max_runtime_sec=60, dry_run=False)
            except BaseException:
                cancel_pending_job_locked(job_name="x")
                raise
    """)
    v = check_lightning_submit_cancel_only_before_network(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == [], v


def test_test_files_are_skipped(tmp_path: Path) -> None:
    _write(tmp_path, "src/tac/tests/test_lightning_x.py", """
        def test_foo():
            try:
                submit_lightning_job(job_name="x", machine="t4", command="c",
                    teamspace="t", studio="s", user="u",
                    max_runtime_sec=60, dry_run=False)
            except BaseException:
                cancel_pending_job_locked(job_name="x")
    """)
    v = check_lightning_submit_cancel_only_before_network(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == [], v


def test_public_pr_intake_skipped(tmp_path: Path) -> None:
    _write(tmp_path, "experiments/results/public_pr101_intake_codex/lightning_x.py", """
        def main():
            try:
                submit_lightning_job(job_name="x", machine="t4", command="c",
                    teamspace="t", studio="s", user="u",
                    max_runtime_sec=60, dry_run=False)
            except BaseException:
                cancel_pending_job_locked(job_name="x")
    """)
    v = check_lightning_submit_cancel_only_before_network(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert v == [], v


def test_attribute_call_cancel_is_detected(tmp_path: Path) -> None:
    _write(tmp_path, "experiments/attr_lightning.py", """
        def main():
            try:
                submit_lightning_job(job_name="x", machine="t4", command="c",
                    teamspace="t", studio="s", user="u",
                    max_runtime_sec=60, dry_run=False)
            except BaseException:
                helper.cancel_pending_job_locked(job_name="x")
    """)
    v = check_lightning_submit_cancel_only_before_network(
        repo_root=tmp_path, strict=False, verbose=False,
    )
    assert len(v) == 1, v


def test_real_repo_live_count_zero() -> None:
    """Smoke: live count on the repo as committed is 0."""
    v = check_lightning_submit_cancel_only_before_network(
        strict=False, verbose=False,
    )
    assert v == [], (
        f"Catalog #147 live count must be 0 after the round 7+8 HIGH 1 "
        f"fix landed; got {len(v)} violation(s):\n"
        + "\n".join(v[:5])
    )


def test_strict_mode_passes_on_real_repo() -> None:
    """Smoke: strict mode does not raise on the repo as committed."""
    # If this fails the dispatchers regressed; fix the dispatcher, not the test.
    check_lightning_submit_cancel_only_before_network(
        strict=True, verbose=False,
    )
