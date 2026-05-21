# SPDX-License-Identifier: MIT
"""Dedicated tests for Catalog #361 + sister vendor helper.

Catalog #361 (`check_modal_artifact_filter_preserves_submission_dir`) refuses
any state of `experiments/modal_train_lane.py` where the artifact harvester
applies `mtime_floor` to files under `output/submission/` without the canonical
bypass. The bug class anchor is OVERNIGHT-CC 99d06f967 (2026-05-21) — 4 paired
DP1 auth_eval Modal dispatches rc=1 ModuleNotFoundError because 8 vendored .py
module bodies were silently dropped by the harvester.

Also tests the canonical helper
`tac.substrates._shared.trainer_skeleton.vendor_module_with_fresh_mtime` that
provides defense-in-depth at the substrate-trainer surface.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_modal_artifact_filter_preserves_submission_dir,
)
from tac.substrates._shared.trainer_skeleton import (
    vendor_module_with_fresh_mtime,
)


# ---------------------------------------------------------------------------
# Live-repo regression guard
# ---------------------------------------------------------------------------


def test_live_repo_passes_catalog_361() -> None:
    """OVERNIGHT-GG fix landed in same commit batch — must be clean."""
    violations = check_modal_artifact_filter_preserves_submission_dir()
    assert violations == [], (
        f"Catalog #361 should be clean post-OVERNIGHT-GG; got: {violations}"
    )


def test_live_repo_passes_catalog_361_strict() -> None:
    """Strict-mode invocation against live repo."""
    violations = check_modal_artifact_filter_preserves_submission_dir(strict=True)
    assert violations == []


# ---------------------------------------------------------------------------
# Synthetic regression / positive cases
# ---------------------------------------------------------------------------


def _write_synthetic_modal_train_lane(repo_root: Path, body: str) -> Path:
    """Materialize a synthetic experiments/modal_train_lane.py for testing."""
    target = repo_root / "experiments" / "modal_train_lane.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(body, encoding="utf-8")
    return target


def test_synthetic_clean_passes(tmp_path: Path) -> None:
    """Canonical bypass pattern accepted."""
    canonical_body = (
        "# canonical filter\n"
        "artifact_mtime_floor = time.time() - 5.0\n"
        "for fp in files:\n"
        "    rel_parts = rel.parts\n"
        "    under_submission = (\n"
        "        len(rel_parts) >= 3\n"
        '        and rel_parts[0] == "output"\n'
        '        and rel_parts[1] == "submission"\n'
        "    )\n"
        "    if not under_submission and st.st_mtime < artifact_mtime_floor:\n"
        "        continue\n"
    )
    _write_synthetic_modal_train_lane(tmp_path, canonical_body)
    assert (
        check_modal_artifact_filter_preserves_submission_dir(
            repo_root=tmp_path
        )
        == []
    )


def test_synthetic_regression_no_bypass_flagged(tmp_path: Path) -> None:
    """Pre-OVERNIGHT-GG state (no bypass) flagged."""
    regression_body = (
        "artifact_mtime_floor = time.time() - 5.0\n"
        "for fp in files:\n"
        "    if st.st_mtime < artifact_mtime_floor:\n"
        "        continue\n"
    )
    _write_synthetic_modal_train_lane(tmp_path, regression_body)
    violations = check_modal_artifact_filter_preserves_submission_dir(
        repo_root=tmp_path
    )
    assert len(violations) == 1
    msg = violations[0]
    assert "[catalog-361]" in msg
    assert "OVERNIGHT-CC 99d06f967" in msg
    assert "ModuleNotFoundError" in msg
    assert "under_submission" in msg


def test_synthetic_partial_bypass_flagged(tmp_path: Path) -> None:
    """Partial fix (missing one canonical token) still flagged."""
    partial_body = (
        "artifact_mtime_floor = time.time() - 5.0\n"
        "for fp in files:\n"
        "    rel_parts = rel.parts\n"
        "    # missing under_submission guard\n"
        '    if rel_parts[0] == "output" and rel_parts[1] == "submission":\n'
        "        artifacts[rel_str] = fp.read_bytes()\n"
    )
    _write_synthetic_modal_train_lane(tmp_path, partial_body)
    violations = check_modal_artifact_filter_preserves_submission_dir(
        repo_root=tmp_path
    )
    assert len(violations) == 1


def test_strict_mode_raises(tmp_path: Path) -> None:
    """Strict mode raises PreflightError on violation."""
    regression_body = (
        "artifact_mtime_floor = time.time() - 5.0\n"
        "if st.st_mtime < artifact_mtime_floor: continue\n"
    )
    _write_synthetic_modal_train_lane(tmp_path, regression_body)
    with pytest.raises(PreflightError, match="catalog-361"):
        check_modal_artifact_filter_preserves_submission_dir(
            repo_root=tmp_path, strict=True
        )


def test_strict_silent_on_clean(tmp_path: Path) -> None:
    canonical_body = (
        "artifact_mtime_floor = time.time() - 5.0\n"
        "rel_parts = rel.parts\n"
        "under_submission = (\n"
        "    len(rel_parts) >= 3\n"
        '    and rel_parts[0] == "output"\n'
        '    and rel_parts[1] == "submission"\n'
        ")\n"
        "if not under_submission and st.st_mtime < artifact_mtime_floor:\n"
        "    continue\n"
    )
    _write_synthetic_modal_train_lane(tmp_path, canonical_body)
    assert (
        check_modal_artifact_filter_preserves_submission_dir(
            repo_root=tmp_path, strict=True
        )
        == []
    )


# ---------------------------------------------------------------------------
# Waiver semantics
# ---------------------------------------------------------------------------


def test_file_level_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    """Real rationale in first 80 lines waives."""
    waived_body = (
        "# CATALOG_361_HARVESTER_FILTER_WAIVED:explicit_alternative_harvester_per_operator_review_with_paired_test_coverage\n"
        "# no canonical bypass tokens below\n"
        "artifact_mtime_floor = time.time() - 5.0\n"
        "if st.st_mtime < artifact_mtime_floor: continue\n"
    )
    _write_synthetic_modal_train_lane(tmp_path, waived_body)
    assert (
        check_modal_artifact_filter_preserves_submission_dir(
            repo_root=tmp_path
        )
        == []
    )


def test_file_level_waiver_placeholder_rationale_rejected(
    tmp_path: Path,
) -> None:
    """Placeholder `<rationale>` rejected (gate's own docstring example)."""
    waived_body = (
        "# CATALOG_361_HARVESTER_FILTER_WAIVED:<rationale>\n"
        "artifact_mtime_floor = time.time() - 5.0\n"
        "if st.st_mtime < artifact_mtime_floor: continue\n"
    )
    _write_synthetic_modal_train_lane(tmp_path, waived_body)
    violations = check_modal_artifact_filter_preserves_submission_dir(
        repo_root=tmp_path
    )
    assert len(violations) == 1


def test_file_level_waiver_placeholder_reason_rejected(tmp_path: Path) -> None:
    waived_body = (
        "# CATALOG_361_HARVESTER_FILTER_WAIVED:<reason>\n"
        "artifact_mtime_floor = time.time() - 5.0\n"
        "if st.st_mtime < artifact_mtime_floor: continue\n"
    )
    _write_synthetic_modal_train_lane(tmp_path, waived_body)
    assert len(
        check_modal_artifact_filter_preserves_submission_dir(repo_root=tmp_path)
    ) == 1


def test_file_level_waiver_empty_rationale_rejected(tmp_path: Path) -> None:
    waived_body = (
        "# CATALOG_361_HARVESTER_FILTER_WAIVED:\n"
        "artifact_mtime_floor = time.time() - 5.0\n"
        "if st.st_mtime < artifact_mtime_floor: continue\n"
    )
    _write_synthetic_modal_train_lane(tmp_path, waived_body)
    assert len(
        check_modal_artifact_filter_preserves_submission_dir(repo_root=tmp_path)
    ) == 1


def test_file_level_waiver_short_rationale_rejected(tmp_path: Path) -> None:
    waived_body = (
        "# CATALOG_361_HARVESTER_FILTER_WAIVED:ok\n"  # 2 chars
        "artifact_mtime_floor = time.time() - 5.0\n"
        "if st.st_mtime < artifact_mtime_floor: continue\n"
    )
    _write_synthetic_modal_train_lane(tmp_path, waived_body)
    assert len(
        check_modal_artifact_filter_preserves_submission_dir(repo_root=tmp_path)
    ) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_missing_target_returns_empty(tmp_path: Path) -> None:
    """Out-of-scope: dispatcher file absent (e.g., fresh checkout)."""
    assert (
        check_modal_artifact_filter_preserves_submission_dir(repo_root=tmp_path)
        == []
    )


def test_string_repo_root_accepted(tmp_path: Path) -> None:
    """repo_root accepts str (not just Path)."""
    canonical_body = (
        "artifact_mtime_floor = time.time() - 5.0\n"
        "rel_parts = rel.parts\n"
        "under_submission = (\n"
        "    len(rel_parts) >= 3\n"
        '    and rel_parts[0] == "output"\n'
        '    and rel_parts[1] == "submission"\n'
        ")\n"
        "if not under_submission and st.st_mtime < artifact_mtime_floor:\n"
        "    continue\n"
    )
    _write_synthetic_modal_train_lane(tmp_path, canonical_body)
    assert (
        check_modal_artifact_filter_preserves_submission_dir(
            repo_root=str(tmp_path)
        )
        == []
    )


def test_verbose_output_clean(tmp_path: Path, capsys) -> None:
    canonical_body = (
        "artifact_mtime_floor = time.time() - 5.0\n"
        "rel_parts = rel.parts\n"
        "under_submission = (\n"
        "    len(rel_parts) >= 3\n"
        '    and rel_parts[0] == "output"\n'
        '    and rel_parts[1] == "submission"\n'
        ")\n"
        "if not under_submission and st.st_mtime < artifact_mtime_floor:\n"
        "    continue\n"
    )
    _write_synthetic_modal_train_lane(tmp_path, canonical_body)
    check_modal_artifact_filter_preserves_submission_dir(
        repo_root=tmp_path, verbose=True
    )
    captured = capsys.readouterr()
    assert "[catalog-361]" in captured.out
    assert "0 violation(s)" in captured.out


# ---------------------------------------------------------------------------
# vendor_module_with_fresh_mtime (canonical helper defense-in-depth)
# ---------------------------------------------------------------------------


def test_vendor_module_with_fresh_mtime_basic(tmp_path: Path) -> None:
    """Helper copies content + stamps current mtime."""
    src = tmp_path / "src.py"
    src.write_text("# canonical source\nx = 1\n", encoding="utf-8")
    # Backdate source mtime to simulate Modal copytree(symlinks=True) propagation
    old_mtime = time.time() - 86400  # 1 day old
    os.utime(src, (old_mtime, old_mtime))
    assert src.stat().st_mtime < time.time() - 1000  # confirm backdated

    dst = tmp_path / "vendored" / "dst.py"
    vendor_module_with_fresh_mtime(src, dst)

    assert dst.is_file()
    assert dst.read_text(encoding="utf-8") == "# canonical source\nx = 1\n"
    # Critical invariant: dst mtime is FRESH (within 5 seconds of now)
    assert dst.stat().st_mtime > time.time() - 5.0, (
        f"dst mtime {dst.stat().st_mtime} should be fresh; "
        f"now={time.time()}; src_mtime={src.stat().st_mtime}"
    )


def test_vendor_module_with_fresh_mtime_creates_parent_dirs(
    tmp_path: Path,
) -> None:
    src = tmp_path / "src.py"
    src.write_text("y = 2\n", encoding="utf-8")
    dst = tmp_path / "deep" / "nested" / "tree" / "dst.py"
    assert not dst.parent.is_dir()
    vendor_module_with_fresh_mtime(src, dst)
    assert dst.is_file()


def test_vendor_module_passes_mtime_floor_filter(tmp_path: Path) -> None:
    """End-to-end PV: vendored module passes Modal harvester's mtime_floor.

    Reproduces the OVERNIGHT-CC 99d06f967 anchor scenario: source file
    has old mtime (simulating Modal copytree-staged repo), filter floor
    is set BEFORE vendoring (simulating lane-start timing). Without the
    helper, shutil.copy2's preserved mtime would fail the floor check
    and the file would be silently dropped.
    """
    src = tmp_path / "old_source.py"
    src.write_text("# old\n", encoding="utf-8")
    # Backdate aggressively
    old_mtime = time.time() - 86400 * 30  # 30 days old
    os.utime(src, (old_mtime, old_mtime))

    # Simulate harvester capturing floor BEFORE vendoring (Modal pattern)
    artifact_mtime_floor = time.time() - 5.0
    time.sleep(0.05)  # ensure post-floor

    dst = tmp_path / "submission" / "dst.py"
    vendor_module_with_fresh_mtime(src, dst)

    # Critical PV: dst mtime > floor (would pass the harvester filter)
    assert dst.stat().st_mtime >= artifact_mtime_floor, (
        f"vendored body mtime {dst.stat().st_mtime} must be > floor "
        f"{artifact_mtime_floor} (OVERNIGHT-CC anchor regression)"
    )


# ---------------------------------------------------------------------------
# Orchestrator wire-in regression guard
# ---------------------------------------------------------------------------


def test_orchestrator_wires_catalog_361_strict_true() -> None:
    """preflight_all() must call Catalog #361 with strict=True (live count: 0)."""
    preflight_src = (
        Path(__file__).resolve().parents[3] / "src" / "tac" / "preflight.py"
    )
    text = preflight_src.read_text(encoding="utf-8")
    # Must mention the canonical wire-in pattern
    assert "check_modal_artifact_filter_preserves_submission_dir" in text
    assert "Catalog #361" in text
    # Must be wired strict=True in preflight_all (find the body of the orchestrator)
    # Conservative check: pattern is a Call with strict=True nearby.
    invocation_idx = text.find(
        "check_modal_artifact_filter_preserves_submission_dir(\n"
        "            strict=True"
    )
    assert invocation_idx > 0, "Catalog #361 must be wired strict=True in preflight_all()"


# ---------------------------------------------------------------------------
# Catalog #185 sister-callable regression guard
# ---------------------------------------------------------------------------


def test_catalog_361_callable_via_globals() -> None:
    """Per Catalog #185, function must be importable via tac.preflight namespace."""
    import tac.preflight as mod
    fn = getattr(mod, "check_modal_artifact_filter_preserves_submission_dir")
    assert callable(fn)
    # Signature: kwargs-only
    import inspect
    sig = inspect.signature(fn)
    for name in ("repo_root", "strict", "verbose"):
        assert name in sig.parameters, f"missing param: {name}"
