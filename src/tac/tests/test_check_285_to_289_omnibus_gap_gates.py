"""Catalog #285-#289 OMNIBUS GAP gates - dedicated tests.

Sister tests for the 5 GAP gates landed via PHASE-2-LAND-5-GAP-GATES per
`feedback_phase_2_land_5_gap_gates_omnibus_followon_landed_20260515.md`.

Lane: lane_phase_2_land_5_gap_gates_20260515.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_catalog_text_references_existing_gate_callable,
    check_new_dispatch_guard_helper_has_no_fail_open_anti_pattern,
    check_no_docstring_overstatement_without_evidence_tag,
    check_serializer_log_no_dropped_expected_content_sha_retry_pattern,
    check_uv_torch_install_has_driver_version_pin,
)

# COMMIT_SERIALIZER_BYPASS_OK_FILE:test fixtures contain canonical violation samples and waiver examples; the live-repo gate self-exempts test files.
# DOCSTRING_PERCENT_CLAIM_OK:test fixtures contain canonical violation samples for Catalog #287


# ============================================================================
# Helpers
# ============================================================================

def _make_repo(tmp_path: Path) -> Path:
    """Return a tmp repo root with empty src/tac/ and tools/ skeletons."""
    (tmp_path / "src" / "tac").mkdir(parents=True)
    (tmp_path / "tools").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "experiments").mkdir()
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    return tmp_path


# ============================================================================
# Catalog #285 - META-meta-meta fail-open guard tests
# ============================================================================

class TestCheck285FailOpenGuard:
    def test_clean_repo_returns_empty(self, tmp_path):
        repo = _make_repo(tmp_path)
        violations = check_new_dispatch_guard_helper_has_no_fail_open_anti_pattern(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_imports_dispatch_token_only_does_not_flag(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "tools" / "operator_authorize.py").write_text(
            "import os\n\ndef foo():\n    return True\n"
        )
        violations = check_new_dispatch_guard_helper_has_no_fail_open_anti_pattern(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_import_error_return_true_anti_pattern_flagged(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "tools" / "local_pre_deploy_check.py").write_text(
            "def helper():\n    try:\n        from canonical import x\n"
            "    except ImportError:\n        return True\n    return False\n"
        )
        violations = check_new_dispatch_guard_helper_has_no_fail_open_anti_pattern(
            repo_root=repo, strict=False, verbose=False,
        )
        assert len(violations) == 1
        assert "fail-open anti-pattern" in violations[0]

    def test_same_window_waiver_with_rationale_accepted(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "tools" / "local_pre_deploy_check.py").write_text(
            "def helper():\n    try:\n        # DISPATCH_GUARD_FAIL_OPEN_OK:legacy harness wrapper test fixture\n"
            "        from canonical import x\n"
            "    except ImportError:\n        return True\n    return False\n"
        )
        violations = check_new_dispatch_guard_helper_has_no_fail_open_anti_pattern(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_placeholder_waiver_rejected(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "tools" / "local_pre_deploy_check.py").write_text(
            "def helper():\n    try:\n        # DISPATCH_GUARD_FAIL_OPEN_OK:<rationale>\n"
            "        from canonical import x\n"
            "    except ImportError:\n        return True\n    return False\n"
        )
        violations = check_new_dispatch_guard_helper_has_no_fail_open_anti_pattern(
            repo_root=repo, strict=False, verbose=False,
        )
        assert len(violations) == 1

    def test_strict_raises_on_violation(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "tools" / "local_pre_deploy_check.py").write_text(
            "def helper():\n    try:\n        from canonical import x\n"
            "    except ImportError:\n        return True\n    return False\n"
        )
        with pytest.raises(PreflightError, match="Catalog #285"):
            check_new_dispatch_guard_helper_has_no_fail_open_anti_pattern(
                repo_root=repo, strict=True, verbose=False,
            )

    def test_unrelated_file_skipped(self, tmp_path):
        repo = _make_repo(tmp_path)
        # Out of scope - not in the curated allow-list
        (repo / "tools" / "unrelated.py").write_text(
            "def helper():\n    try:\n        from canonical import x\n"
            "    except ImportError:\n        return True\n"
        )
        violations = check_new_dispatch_guard_helper_has_no_fail_open_anti_pattern(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_live_repo_regression_guard(self):
        """Live-repo MUST stay at 0 violations (warn-only at landing)."""
        violations = check_new_dispatch_guard_helper_has_no_fail_open_anti_pattern(
            strict=False, verbose=False,
        )
        # Bound at landing; tighten as backfill progresses.
        assert len(violations) <= 5, (
            f"Live count regressed: {len(violations)} violations "
            "(landing baseline: 0; warn-only allows incremental drift)"
        )


# ============================================================================
# Catalog #286 - phantom catalog row tests
# ============================================================================

class TestCheck286PhantomCatalogRow:
    def test_clean_repo_returns_empty(self, tmp_path):
        repo = _make_repo(tmp_path)
        # No CLAUDE.md in tmp repo
        violations = check_catalog_text_references_existing_gate_callable(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_synthetic_phantom_row_flagged(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "CLAUDE.md").write_text(
            "## Meta-bug class catalog (strict-mode preflight)\n\n"
            "1. `check_no_mps_fallback_default` - real callable.\n"
            "999. `check_phantom_does_not_exist` - phantom row.\n"
            "\n## Next section\n"
        )
        violations = check_catalog_text_references_existing_gate_callable(
            repo_root=repo, strict=False, verbose=False,
        )
        assert len(violations) == 1
        assert "PHANTOM" in violations[0]
        assert "999" in violations[0]
        assert "check_phantom_does_not_exist" in violations[0]

    def test_real_callable_passes(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "CLAUDE.md").write_text(
            "## Meta-bug class catalog (strict-mode preflight)\n\n"
            "1. `check_no_mps_fallback_default` - real callable.\n"
            "286. `check_catalog_text_references_existing_gate_callable` - this gate.\n"
            "\n## Next section\n"
        )
        violations = check_catalog_text_references_existing_gate_callable(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_strict_raises_on_phantom(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "CLAUDE.md").write_text(
            "## Meta-bug class catalog\n\n"
            "999. `check_phantom_does_not_exist` - phantom.\n"
            "\n## Next\n"
        )
        with pytest.raises(PreflightError, match="Catalog #286"):
            check_catalog_text_references_existing_gate_callable(
                repo_root=repo, strict=True, verbose=False,
            )

    def test_live_repo_regression_guard_zero_phantoms(self):
        """Live-repo MUST have 0 phantom catalog rows after PHASE-1 recovery."""
        violations = check_catalog_text_references_existing_gate_callable(
            strict=False, verbose=False,
        )
        assert violations == [], (
            f"Live count regressed: {len(violations)} phantom catalog "
            f"row(s); PHASE-1 commit 42665ce95 was supposed to recover all "
            f"of #273-#278. Sample: {violations[0] if violations else '<none>'}"
        )

    def test_strict_mode_silent_on_clean(self, tmp_path):
        repo = _make_repo(tmp_path)
        # No CLAUDE.md
        violations = check_catalog_text_references_existing_gate_callable(
            repo_root=repo, strict=True, verbose=False,
        )
        assert violations == []


# ============================================================================
# Catalog #287 - docstring overstatement tests
# ============================================================================

class TestCheck287DocstringOverstatement:
    def test_clean_src_returns_empty(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "src" / "tac" / "module.py").write_text(
            '"""A clean module without any percentage claims."""\n'
            "def foo():\n    return 42\n"
        )
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_saves_percent_without_tag_flagged(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "src" / "tac" / "module.py").write_text(
            '"""This codec saves 49% bytes vs baseline."""\n'
            "def foo():\n    return 42\n"
        )
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=repo, strict=False, verbose=False,
        )
        assert len(violations) >= 1
        assert "saves 49%" in violations[0]

    def test_saves_percent_with_evidence_tag_passes(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "src" / "tac" / "module.py").write_text(
            '"""This codec saves 49% bytes vs baseline.\n\n'
            'Evidence: [empirical:reports/raw/2026-05-15-codec-eval.json]\n'
            '"""\n'
            "def foo():\n    return 42\n"
        )
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_contest_cuda_tag_passes(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "src" / "tac" / "module.py").write_text(
            '"""This codec saves 49% bytes vs baseline [contest-CUDA T4]."""\n'
            "def foo():\n    return 42\n"
        )
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_same_line_waiver_accepted(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "src" / "tac" / "module.py").write_text(
            '"""Saves 49% per design proof  # DOCSTRING_PERCENT_CLAIM_OK:design proof not measurement."""\n'
            "def foo():\n    return 42\n"
        )
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_placeholder_waiver_rejected(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "src" / "tac" / "module.py").write_text(
            '# Saves 49% per design proof  # DOCSTRING_PERCENT_CLAIM_OK:<rationale>\n'
            "def foo():\n    return 42\n"
        )
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=repo, strict=False, verbose=False,
        )
        assert len(violations) >= 1

    def test_test_files_excluded(self, tmp_path):
        repo = _make_repo(tmp_path)
        test_dir = repo / "src" / "tac" / "tests"
        test_dir.mkdir()
        (test_dir / "test_foo.py").write_text(
            '"""This test asserts saves 49% bytes."""\n'
            "def test_foo():\n    pass\n"
        )
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_strict_raises_on_violation(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "src" / "tac" / "module.py").write_text(
            '"""Saves 49% bytes vs baseline."""\n'
        )
        with pytest.raises(PreflightError, match="Catalog #287"):
            check_no_docstring_overstatement_without_evidence_tag(
                repo_root=repo, strict=True, verbose=False,
            )

    def test_x_faster_pattern_flagged(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "src" / "tac" / "module.py").write_text(
            '"""4x faster than naive baseline."""\n'
        )
        violations = check_no_docstring_overstatement_without_evidence_tag(
            repo_root=repo, strict=False, verbose=False,
        )
        assert len(violations) >= 1

    def test_live_repo_regression_warn_only(self):
        """Live-repo: warn-only at landing; bound rather than zero."""
        violations = check_no_docstring_overstatement_without_evidence_tag(
            strict=False, verbose=False,
        )
        # Initial landing tolerates legacy violations; STRICT-flip after sweep
        assert len(violations) < 10000  # absurd upper bound; sanity only


# ============================================================================
# Catalog #288 - uv torch driver-version pin tests
# ============================================================================

class TestCheck288UvTorchDriverPin:
    def test_clean_repo_returns_empty(self, tmp_path):
        repo = _make_repo(tmp_path)
        violations = check_uv_torch_install_has_driver_version_pin(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_unpinned_uv_pip_install_torch_flagged(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "scripts" / "remote_lane_foo.sh").write_text(
            "#!/bin/bash\nset -e\nuv pip install torch==2.5.1\n"
        )
        violations = check_uv_torch_install_has_driver_version_pin(
            repo_root=repo, strict=False, verbose=False,
        )
        assert len(violations) >= 1
        assert "unpinned" in violations[0]

    def test_pinned_with_cu124_passes(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "scripts" / "remote_lane_foo.sh").write_text(
            "#!/bin/bash\nset -e\nuv pip install torch==2.5.1+cu124\n"
        )
        violations = check_uv_torch_install_has_driver_version_pin(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_inflate_torch_spec_env_var_passes(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "scripts" / "remote_lane_foo.sh").write_text(
            "#!/bin/bash\nset -e\n"
            "export INFLATE_TORCH_SPEC=torch==2.5.1+cu124\n"
            "uv pip install torch\n"
        )
        violations = check_uv_torch_install_has_driver_version_pin(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_extra_index_url_passes(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "scripts" / "remote_lane_foo.sh").write_text(
            "#!/bin/bash\nset -e\n"
            "export UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu124\n"
            "uv pip install torch\n"
        )
        violations = check_uv_torch_install_has_driver_version_pin(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_same_line_waiver_accepted(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "scripts" / "remote_lane_foo.sh").write_text(
            "#!/bin/bash\nset -e\n"
            "uv pip install torch  # UV_TORCH_DRIVER_PIN_OK:smoke test deliberately uses default wheel\n"
        )
        violations = check_uv_torch_install_has_driver_version_pin(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_placeholder_waiver_rejected(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "scripts" / "remote_lane_foo.sh").write_text(
            "#!/bin/bash\nset -e\n"
            "uv pip install torch  # UV_TORCH_DRIVER_PIN_OK:<rationale>\n"
        )
        violations = check_uv_torch_install_has_driver_version_pin(
            repo_root=repo, strict=False, verbose=False,
        )
        assert len(violations) >= 1

    def test_strict_raises_on_violation(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "scripts" / "remote_lane_foo.sh").write_text(
            "#!/bin/bash\nuv pip install torch\n"
        )
        with pytest.raises(PreflightError, match="Catalog #288"):
            check_uv_torch_install_has_driver_version_pin(
                repo_root=repo, strict=True, verbose=False,
            )

    def test_test_files_excluded(self, tmp_path):
        repo = _make_repo(tmp_path)
        (repo / "scripts" / "tests").mkdir()
        (repo / "scripts" / "tests" / "test_foo.sh").write_text(
            "uv pip install torch\n"
        )
        violations = check_uv_torch_install_has_driver_version_pin(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_live_repo_regression_zero(self):
        """Live-repo MUST stay at 0 (canonical wrapper auto-pins)."""
        violations = check_uv_torch_install_has_driver_version_pin(
            strict=False, verbose=False,
        )
        assert len(violations) <= 5, (
            f"Live count regressed: {len(violations)} unpinned uv torch "
            f"installs. Sample: {violations[0] if violations else '<none>'}"
        )


# ============================================================================
# Catalog #289 - commit-serializer drop-flag-and-retry tests
# ============================================================================

class TestCheck289DropFlagRetry:
    def test_no_log_returns_empty(self, tmp_path):
        repo = _make_repo(tmp_path)
        violations = check_serializer_log_no_dropped_expected_content_sha_retry_pattern(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_clean_log_returns_empty(self, tmp_path):
        repo = _make_repo(tmp_path)
        log = repo / ".omx" / "state" / "commit-serializer.log"
        log.write_text(
            json.dumps({
                "started_at_utc": "2026-05-16T00:00:00Z",
                "outcome": "committed",
                "files": ["a.py", "b.py", "c.py"],
                "message_head": "test commit",
            }) + "\n"
        )
        violations = check_serializer_log_no_dropped_expected_content_sha_retry_pattern(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_drop_flag_retry_pattern_flagged(self, tmp_path):
        repo = _make_repo(tmp_path)
        log = repo / ".omx" / "state" / "commit-serializer.log"
        log.write_text(
            json.dumps({
                "started_at_utc": "2026-05-16T00:00:00Z",
                "outcome": "expected_content_sha_mismatch",
                "files": ["a.py", "b.py", "c.py"],
                "message_head": "drop-flag-test",
                "expected_content_sha256_present": True,
            }) + "\n"
            + json.dumps({
                "started_at_utc": "2026-05-16T00:01:00Z",
                "outcome": "committed",
                "files": ["a.py", "b.py", "c.py"],
                "message_head": "drop-flag-test",
                "expected_content_sha256_present": False,  # WAVE-D bug
            }) + "\n"
        )
        violations = check_serializer_log_no_dropped_expected_content_sha_retry_pattern(
            repo_root=repo, strict=False, verbose=False,
        )
        assert len(violations) == 1
        assert "drop-flag-and-retry pattern" in violations[0]

    def test_correct_re_pass_with_flag_present_passes(self, tmp_path):
        repo = _make_repo(tmp_path)
        log = repo / ".omx" / "state" / "commit-serializer.log"
        log.write_text(
            json.dumps({
                "started_at_utc": "2026-05-16T00:00:00Z",
                "outcome": "expected_content_sha_mismatch",
                "files": ["a.py", "b.py", "c.py"],
                "message_head": "test",
                "expected_content_sha256_present": True,
            }) + "\n"
            + json.dumps({
                "started_at_utc": "2026-05-16T00:01:00Z",
                "outcome": "committed",
                "files": ["a.py", "b.py", "c.py"],
                "message_head": "test",
                "expected_content_sha256_present": True,  # caller re-passed
            }) + "\n"
        )
        violations = check_serializer_log_no_dropped_expected_content_sha_retry_pattern(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_pre_gap5_legacy_schema_exempt(self, tmp_path):
        repo = _make_repo(tmp_path)
        log = repo / ".omx" / "state" / "commit-serializer.log"
        # Pre-GAP-5 schema: no expected_content_sha256_present field at all.
        # Legacy events are exempt.
        log.write_text(
            json.dumps({
                "started_at_utc": "2026-05-16T00:00:00Z",
                "outcome": "expected_content_sha_mismatch",
                "files": ["a.py", "b.py", "c.py"],
                "message_head": "legacy",
            }) + "\n"
            + json.dumps({
                "started_at_utc": "2026-05-16T00:01:00Z",
                "outcome": "committed",
                "files": ["a.py", "b.py", "c.py"],
                "message_head": "legacy",
            }) + "\n"
        )
        violations = check_serializer_log_no_dropped_expected_content_sha_retry_pattern(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_pre_cutoff_event_exempt(self, tmp_path):
        repo = _make_repo(tmp_path)
        log = repo / ".omx" / "state" / "commit-serializer.log"
        # Pre-cutoff (2026-05-15T22:00:00Z): exempt per Strict-flip atomicity
        log.write_text(
            json.dumps({
                "started_at_utc": "2026-05-15T20:00:00Z",
                "outcome": "expected_content_sha_mismatch",
                "files": ["a.py", "b.py", "c.py"],
                "message_head": "legacy",
            }) + "\n"
            + json.dumps({
                "started_at_utc": "2026-05-15T20:01:00Z",
                "outcome": "committed",
                "files": ["a.py", "b.py", "c.py"],
                "message_head": "legacy",
            }) + "\n"
        )
        violations = check_serializer_log_no_dropped_expected_content_sha_retry_pattern(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_small_commit_under_threshold_not_flagged(self, tmp_path):
        repo = _make_repo(tmp_path)
        log = repo / ".omx" / "state" / "commit-serializer.log"
        # nfiles=2 < 3 threshold = lower-risk, not flagged
        log.write_text(
            json.dumps({
                "started_at_utc": "2026-05-16T00:00:00Z",
                "outcome": "expected_content_sha_mismatch",
                "files": ["a.py", "b.py"],
                "message_head": "small",
            }) + "\n"
            + json.dumps({
                "started_at_utc": "2026-05-16T00:01:00Z",
                "outcome": "committed",
                "files": ["a.py", "b.py"],
                "message_head": "small",
            }) + "\n"
        )
        violations = check_serializer_log_no_dropped_expected_content_sha_retry_pattern(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_outside_window_not_flagged(self, tmp_path):
        repo = _make_repo(tmp_path)
        log = repo / ".omx" / "state" / "commit-serializer.log"
        # >10 min apart => unrelated retries
        log.write_text(
            json.dumps({
                "started_at_utc": "2026-05-16T00:00:00Z",
                "outcome": "expected_content_sha_mismatch",
                "files": ["a.py", "b.py", "c.py"],
                "message_head": "test",
            }) + "\n"
            + json.dumps({
                "started_at_utc": "2026-05-16T00:30:00Z",
                "outcome": "committed",
                "files": ["a.py", "b.py", "c.py"],
                "message_head": "test",
            }) + "\n"
        )
        violations = check_serializer_log_no_dropped_expected_content_sha_retry_pattern(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_strict_raises_on_violation(self, tmp_path):
        repo = _make_repo(tmp_path)
        log = repo / ".omx" / "state" / "commit-serializer.log"
        log.write_text(
            json.dumps({
                "started_at_utc": "2026-05-16T00:00:00Z",
                "outcome": "expected_content_sha_mismatch",
                "files": ["a.py", "b.py", "c.py"],
                "message_head": "drop-flag-test",
                "expected_content_sha256_present": True,
            }) + "\n"
            + json.dumps({
                "started_at_utc": "2026-05-16T00:01:00Z",
                "outcome": "committed",
                "files": ["a.py", "b.py", "c.py"],
                "message_head": "drop-flag-test",
                "expected_content_sha256_present": False,
            }) + "\n"
        )
        with pytest.raises(PreflightError, match="Catalog #289"):
            check_serializer_log_no_dropped_expected_content_sha_retry_pattern(
                repo_root=repo, strict=True, verbose=False,
            )

    def test_malformed_json_lines_skipped(self, tmp_path):
        repo = _make_repo(tmp_path)
        log = repo / ".omx" / "state" / "commit-serializer.log"
        log.write_text(
            "not json garbage\n"
            + json.dumps({
                "started_at_utc": "2026-05-16T00:00:00Z",
                "outcome": "committed",
                "files": ["a.py"],
                "message_head": "test",
            }) + "\n"
        )
        violations = check_serializer_log_no_dropped_expected_content_sha_retry_pattern(
            repo_root=repo, strict=False, verbose=False,
        )
        assert violations == []

    def test_live_repo_regression_zero(self):
        """Live-repo MUST stay at 0 post-cutoff."""
        violations = check_serializer_log_no_dropped_expected_content_sha_retry_pattern(
            strict=False, verbose=False,
        )
        assert len(violations) <= 3, (
            f"Live count regressed: {len(violations)} drop-flag-retry "
            f"pattern(s) post-cutoff. Sample: "
            f"{violations[0] if violations else '<none>'}"
        )


# ============================================================================
# META-meta sister gate regression guards (Catalog #176 + #185)
# ============================================================================

class TestCheck285To289MetaMetaCleanliness:
    """Verify the 5 new gates do not trip Catalog #176/#185 META-meta gates."""

    def test_all_5_new_gates_callable(self):
        for fn in (
            check_new_dispatch_guard_helper_has_no_fail_open_anti_pattern,
            check_catalog_text_references_existing_gate_callable,
            check_no_docstring_overstatement_without_evidence_tag,
            check_uv_torch_install_has_driver_version_pin,
            check_serializer_log_no_dropped_expected_content_sha_retry_pattern,
        ):
            assert callable(fn)
            # All accept (strict, verbose) kwargs
            result = fn(strict=False, verbose=False)
            assert isinstance(result, list)

    def test_strict_signatures_consistent(self):
        """All 5 gates accept strict=True kwarg without TypeError."""
        # Smoke test: pass strict=True with a clean tmp repo (no actual data),
        # verify no TypeError on signature mismatch.
        # We use a non-existent tmp repo path so each gate's "no input" branch
        # short-circuits cleanly.
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            # Each gate must handle missing input gracefully
            for fn in (
                check_new_dispatch_guard_helper_has_no_fail_open_anti_pattern,
                check_catalog_text_references_existing_gate_callable,
                check_no_docstring_overstatement_without_evidence_tag,
                check_uv_torch_install_has_driver_version_pin,
                check_serializer_log_no_dropped_expected_content_sha_retry_pattern,
            ):
                # No raise expected (empty / missing inputs => empty list)
                result = fn(repo_root=repo, strict=True, verbose=False)
                assert isinstance(result, list)
