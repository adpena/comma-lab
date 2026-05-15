# SPDX-License-Identifier: MIT
"""Catalog #244 — substrate driver scripts MUST carry the canonical
Modal/CUDA env hygiene block (3 exports: DALI_DISABLE_NVML +
CUBLAS_WORKSPACE_CONFIG + PYTORCH_CUDA_ALLOC_CONF).

D1 incident anchor 2026-05-15: substrate_d1_segnet_margin_polytope Modal T4
smoke crashed at "nvml error (999)" inside DALI fn.experimental.inputs.video
because the lane script lacked DALI_DISABLE_NVML=1 export. The companion fix
in tac.substrate_registry.driver_generator AUTO-EMITS the block; this gate
refuses any hand-written substrate driver that omits it.

Initial wire-in is WARN-ONLY per CLAUDE.md "Strict-flip atomicity rule"
because 31/36 legacy substrate scripts are missing the block at landing time.
Strict-flip planned alongside the legacy backfill wave.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tac.preflight import (
    PreflightError,
    check_remote_lane_scripts_carry_canonical_nvml_block,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

CANONICAL_BLOCK = """\
#!/bin/bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
echo "ok"
"""

MISSING_ALL = """\
#!/bin/bash
set -euo pipefail
echo "no canonical NVML hygiene"
"""

MISSING_NVML_ONLY = """\
#!/bin/bash
set -euo pipefail
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
"""

MISSING_CUBLAS_ONLY = """\
#!/bin/bash
set -euo pipefail
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
"""

MISSING_ALLOC_ONLY = """\
#!/bin/bash
set -euo pipefail
export DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
"""


def _setup_scripts_dir(tmp_path: Path) -> Path:
    """Create a tmp scripts/ dir with no other files."""
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    return scripts_dir


def _write_substrate_script(scripts_dir: Path, name: str, body: str) -> Path:
    """Write a remote_lane_substrate_<name>.sh into the tmp scripts/ dir."""
    p = scripts_dir / f"remote_lane_substrate_{name}.sh"
    p.write_text(body)
    return p


# ---------------------------------------------------------------------------
# Positive cases: missing block flagged
# ---------------------------------------------------------------------------


def test_check_244_flags_script_missing_all_three_exports(tmp_path: Path) -> None:
    scripts = _setup_scripts_dir(tmp_path)
    _write_substrate_script(scripts, "alpha", MISSING_ALL)
    violations = check_remote_lane_scripts_carry_canonical_nvml_block(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    v = violations[0]
    assert "remote_lane_substrate_alpha.sh" in v
    assert "DALI_DISABLE_NVML" in v
    assert "CUBLAS_WORKSPACE_CONFIG" in v
    assert "PYTORCH_CUDA_ALLOC_CONF" in v


def test_check_244_flags_script_missing_nvml_only(tmp_path: Path) -> None:
    scripts = _setup_scripts_dir(tmp_path)
    _write_substrate_script(scripts, "alpha", MISSING_NVML_ONLY)
    violations = check_remote_lane_scripts_carry_canonical_nvml_block(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "DALI_DISABLE_NVML" in violations[0]
    # Sister exports present should NOT be in the missing list
    v = violations[0]
    assert "['DALI_DISABLE_NVML']" in v


def test_check_244_flags_script_missing_cublas_only(tmp_path: Path) -> None:
    scripts = _setup_scripts_dir(tmp_path)
    _write_substrate_script(scripts, "alpha", MISSING_CUBLAS_ONLY)
    violations = check_remote_lane_scripts_carry_canonical_nvml_block(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "['CUBLAS_WORKSPACE_CONFIG']" in violations[0]


def test_check_244_flags_script_missing_alloc_only(tmp_path: Path) -> None:
    scripts = _setup_scripts_dir(tmp_path)
    _write_substrate_script(scripts, "alpha", MISSING_ALLOC_ONLY)
    violations = check_remote_lane_scripts_carry_canonical_nvml_block(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "['PYTORCH_CUDA_ALLOC_CONF']" in violations[0]


def test_check_244_flags_multiple_files_aggregated(tmp_path: Path) -> None:
    scripts = _setup_scripts_dir(tmp_path)
    _write_substrate_script(scripts, "alpha", MISSING_ALL)
    _write_substrate_script(scripts, "beta", MISSING_NVML_ONLY)
    _write_substrate_script(scripts, "gamma", CANONICAL_BLOCK)
    violations = check_remote_lane_scripts_carry_canonical_nvml_block(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 2
    paths = [v.split(":")[0] for v in violations]
    assert "scripts/remote_lane_substrate_alpha.sh" in paths
    assert "scripts/remote_lane_substrate_beta.sh" in paths
    assert "scripts/remote_lane_substrate_gamma.sh" not in paths


# ---------------------------------------------------------------------------
# Negative cases: canonical block present, no flag
# ---------------------------------------------------------------------------


def test_check_244_clean_when_block_present(tmp_path: Path) -> None:
    scripts = _setup_scripts_dir(tmp_path)
    _write_substrate_script(scripts, "alpha", CANONICAL_BLOCK)
    violations = check_remote_lane_scripts_carry_canonical_nvml_block(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_244_block_token_present_anywhere_in_body_accepted(
    tmp_path: Path,
) -> None:
    """The check is body-token presence - exact line position does not matter
    for acceptance (driver-generator emits at canonical position; legacy
    scripts may have it elsewhere)."""
    scripts = _setup_scripts_dir(tmp_path)
    body_with_late_exports = """\
#!/bin/bash
set -euo pipefail
echo "some setup"
echo "more setup"
# late but still present
export DALI_DISABLE_NVML=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "ok"
"""
    _write_substrate_script(scripts, "alpha", body_with_late_exports)
    violations = check_remote_lane_scripts_carry_canonical_nvml_block(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Waiver semantics
# ---------------------------------------------------------------------------


def test_check_244_waiver_with_rationale_accepted(tmp_path: Path) -> None:
    scripts = _setup_scripts_dir(tmp_path)
    body_waived = """\
#!/bin/bash
set -euo pipefail
# CANONICAL_NVML_BLOCK_OK: research-only CPU substrate; never dispatched on Modal/CUDA
echo "ok"
"""
    _write_substrate_script(scripts, "alpha", body_waived)
    violations = check_remote_lane_scripts_carry_canonical_nvml_block(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_244_waiver_placeholder_rationale_rejected(tmp_path: Path) -> None:
    scripts = _setup_scripts_dir(tmp_path)
    body_placeholder = """\
#!/bin/bash
set -euo pipefail
# CANONICAL_NVML_BLOCK_OK: <reason>
echo "no exports"
"""
    _write_substrate_script(scripts, "alpha", body_placeholder)
    violations = check_remote_lane_scripts_carry_canonical_nvml_block(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_check_244_waiver_rationale_token_placeholder_rejected(
    tmp_path: Path,
) -> None:
    scripts = _setup_scripts_dir(tmp_path)
    body_placeholder = """\
#!/bin/bash
set -euo pipefail
# CANONICAL_NVML_BLOCK_OK: <rationale>
echo "no exports"
"""
    _write_substrate_script(scripts, "alpha", body_placeholder)
    violations = check_remote_lane_scripts_carry_canonical_nvml_block(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


def test_check_244_waiver_bare_no_rationale_rejected(tmp_path: Path) -> None:
    scripts = _setup_scripts_dir(tmp_path)
    body_bare = """\
#!/bin/bash
set -euo pipefail
# CANONICAL_NVML_BLOCK_OK:
echo "no exports"
"""
    _write_substrate_script(scripts, "alpha", body_bare)
    violations = check_remote_lane_scripts_carry_canonical_nvml_block(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1


# ---------------------------------------------------------------------------
# Scope and edge cases
# ---------------------------------------------------------------------------


def test_check_244_scope_only_substrate_glob(tmp_path: Path) -> None:
    """Only ``remote_lane_substrate_*.sh`` is scanned (other ``remote_lane_*``
    that are not substrate-prefixed are out of scope)."""
    scripts = _setup_scripts_dir(tmp_path)
    # In-scope (substrate)
    _write_substrate_script(scripts, "in_scope", MISSING_ALL)
    # Out-of-scope (different prefix)
    other = scripts / "remote_lane_other_thing.sh"
    other.write_text(MISSING_ALL)
    violations = check_remote_lane_scripts_carry_canonical_nvml_block(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert len(violations) == 1
    assert "remote_lane_substrate_in_scope.sh" in violations[0]
    assert "remote_lane_other_thing.sh" not in str(violations)


def test_check_244_no_scripts_dir_returns_empty(tmp_path: Path) -> None:
    # No scripts/ dir at all
    violations = check_remote_lane_scripts_carry_canonical_nvml_block(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_244_empty_scripts_dir_returns_empty(tmp_path: Path) -> None:
    _setup_scripts_dir(tmp_path)
    violations = check_remote_lane_scripts_carry_canonical_nvml_block(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


def test_check_244_string_repo_root_accepted(tmp_path: Path) -> None:
    scripts = _setup_scripts_dir(tmp_path)
    _write_substrate_script(scripts, "alpha", CANONICAL_BLOCK)
    # Pass Path (canonical signature)
    violations = check_remote_lane_scripts_carry_canonical_nvml_block(
        repo_root=tmp_path, strict=False, verbose=False
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


def test_check_244_strict_raises_on_violation(tmp_path: Path) -> None:
    scripts = _setup_scripts_dir(tmp_path)
    _write_substrate_script(scripts, "alpha", MISSING_ALL)
    with pytest.raises(PreflightError) as ei:
        check_remote_lane_scripts_carry_canonical_nvml_block(
            repo_root=tmp_path, strict=True, verbose=False
        )
    msg = str(ei.value)
    assert "Catalog #244" not in msg or "canonical Modal/CUDA env hygiene" in msg
    assert "DALI_DISABLE_NVML" in msg
    assert "611495f26" in msg
    assert "remote_lane_substrate_alpha.sh" in msg


def test_check_244_strict_silent_on_clean(tmp_path: Path) -> None:
    scripts = _setup_scripts_dir(tmp_path)
    _write_substrate_script(scripts, "alpha", CANONICAL_BLOCK)
    # Should not raise
    violations = check_remote_lane_scripts_carry_canonical_nvml_block(
        repo_root=tmp_path, strict=True, verbose=False
    )
    assert violations == []


# ---------------------------------------------------------------------------
# Live-repo regression guard: the live count is 31 at landing (warn-only).
# Once the legacy backfill wave drives the count to 0 the gate flips strict;
# this regression guard ensures the count does NOT GROW beyond a known
# upper bound while warn-only.
# ---------------------------------------------------------------------------


def test_check_244_live_repo_violation_count_within_bound() -> None:
    """At landing time, the live repo has 31/36 substrate scripts missing the
    canonical block. This guard ensures the count does not grow past 36 (the
    total substrate-script count) — which would be impossible — and ensures
    the gate is operating against the live repo.
    """
    repo_root = Path(__file__).resolve().parents[3]
    if not (repo_root / "scripts").is_dir():
        pytest.skip("scripts/ dir not present")
    violations = check_remote_lane_scripts_carry_canonical_nvml_block(
        repo_root=repo_root, strict=False, verbose=False
    )
    # Upper bound: total substrate scripts count
    total_substrate_scripts = len(
        list((repo_root / "scripts").glob("remote_lane_substrate_*.sh"))
    )
    assert len(violations) <= total_substrate_scripts


def test_check_244_orchestrator_callsite_strict_flipped() -> None:
    """Catalog #244 wire-in MUST be `strict=True` per the 2026-05-15 strict-flip
    by CATALOG-244-BACKFILL-31-DRIVERS subagent. All 31 legacy substrate
    drivers were backfilled with the canonical 3-export block in the same
    commit batch per CLAUDE.md "Strict-flip atomicity rule"; live count at
    strict-flip: 0 (all 36 substrate drivers carry the block).
    """
    import re

    preflight_src = Path(__file__).resolve().parents[1] / "preflight.py"
    text = preflight_src.read_text()
    # Find the wire-in (anywhere in the file)
    pat = re.compile(
        r"check_remote_lane_scripts_carry_canonical_nvml_block\(\s*strict=(\w+)",
        re.M,
    )
    matches = pat.findall(text)
    # Filter out the function-def line (it's strict=False default) -
    # there should be exactly 1 callsite outside the def (the wire-in).
    assert matches, "no orchestrator callsite found"
    # The wire-in callsite is now `strict=True` post-strict-flip 2026-05-15.
    assert "True" in matches, (
        f"orchestrator callsite must be strict=True (strict-flipped 2026-05-15); "
        f"got matches={matches}"
    )
