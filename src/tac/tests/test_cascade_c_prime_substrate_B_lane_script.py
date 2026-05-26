# SPDX-License-Identifier: MIT
"""Cascade C' substrate B lane script structural tests.

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable + Catalog #244
(canonical NVML 3-export) + Catalog #163 (canonical sentinel sourcing) +
Catalog #204 (canonical 3-branch Modal-aware OUTPUT_DIR) + Catalog #326
(driver mode env var consumption with multi-key precedence) + Catalog #146
(contest 3-arg inflate.sh signature).

Tests verify the canonical pattern compliance for
``scripts/remote_lane_substrate_cascade_c_prime_frame_1_segnet_waterfill.sh``
as the structural protection that any future regression breaks loudly at
import time.

Sister test of:
- ``src/tac/tests/test_check_244_remote_lane_canonical_nvml_block.py`` (catalog gate)
- ``src/tac/tests/test_check_163_remote_lane_sentinel.py`` (catalog gate)
- ``src/tac/tests/test_check_204_pr95plus_modal_durable_output.py`` (catalog gate)
- ``src/tac/tests/test_check_326_substrate_driver_consumes_trainer_mode_env_var.py`` (catalog gate)

These tests are SUBSTRATE-LEVEL structural-pattern tests; the catalog gates
above provide REPO-LEVEL structural extinction. Both are required per
CLAUDE.md "Bugs must be permanently fixed AND self-protected against".
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LANE_SCRIPT = REPO_ROOT / "scripts" / "remote_lane_substrate_cascade_c_prime_frame_1_segnet_waterfill.sh"


@pytest.fixture(scope="module")
def lane_script_text() -> str:
    """Cached lane script body for all structural pattern tests."""
    assert LANE_SCRIPT.exists(), f"lane script missing at {LANE_SCRIPT}"
    return LANE_SCRIPT.read_text(encoding="utf-8")


def test_lane_script_exists() -> None:
    """Lane script file must exist at canonical path per recipe lane_script field."""
    assert LANE_SCRIPT.is_file()


def test_lane_script_is_executable() -> None:
    """Lane script must be executable (chmod +x) per canonical bash invocation."""
    import os
    import stat

    mode = os.stat(LANE_SCRIPT).st_mode
    assert mode & stat.S_IXUSR, "lane script must be executable (chmod +x)"


def test_lane_script_bash_syntax_valid() -> None:
    """`bash -n` static parse must succeed (catches shell syntax errors).

    Skipped if bash unavailable (Windows CI).
    """
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available on this platform")
    result = subprocess.run(
        [bash, "-n", str(LANE_SCRIPT)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"bash -n failed: stderr={result.stderr}"


def test_lane_script_has_shebang(lane_script_text: str) -> None:
    """First line must be `#!/bin/bash` shebang per canonical sister pattern."""
    first_line = lane_script_text.splitlines()[0]
    assert first_line == "#!/bin/bash"


def test_lane_script_has_set_euo_pipefail(lane_script_text: str) -> None:
    """`set -euo pipefail` per Catalog #2 (no shell silent-skip cascade)."""
    assert "set -euo pipefail" in lane_script_text


def test_lane_script_carries_canonical_nvml_3_export_block(lane_script_text: str) -> None:
    """Catalog #244: 3-export NVML/CUDA env hygiene block.

    Per D1 incident 2026-05-15 (Modal T4 NVML 999 crash inside DALI). Block must
    appear early (before bootstrap_runtime_deps sourcing) so DALI sees env vars
    before any Python import.
    """
    # All 3 canonical exports must be present
    assert 'CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"' in lane_script_text
    assert 'DALI_DISABLE_NVML="${DALI_DISABLE_NVML:-1}"' in lane_script_text
    assert (
        'PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"'
        in lane_script_text
    )
    # Must appear within first 60 lines (i.e. before bootstrap source)
    lines = lane_script_text.splitlines()
    nvml_line = next(
        (i for i, line in enumerate(lines) if "DALI_DISABLE_NVML=" in line), -1
    )
    assert 0 < nvml_line < 60, (
        f"NVML export must appear in first 60 lines (got line {nvml_line + 1})"
    )


def test_lane_script_sources_canonical_bootstrap_with_sentinel(lane_script_text: str) -> None:
    """Catalog #163: canonical sentinel REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1.

    Without the sentinel, sourcing scripts/remote_archive_only_eval.sh runs the
    sourced main flow (which expects pre-built archive.zip) in the calling
    shell and exits before this lane's stages can start.
    """
    assert (
        "REMOTE_ARCHIVE_ONLY_EVAL_SOURCE_ONLY=1 source"
        in lane_script_text
    )
    assert "scripts/remote_archive_only_eval.sh" in lane_script_text


def test_lane_script_calls_bootstrap_runtime_deps(lane_script_text: str) -> None:
    """The lane script must invoke the canonical bootstrap function after sourcing."""
    assert "bootstrap_runtime_deps" in lane_script_text


def test_lane_script_canonical_3_branch_output_dir(lane_script_text: str) -> None:
    """Catalog #204: 3-branch Modal-aware OUTPUT_DIR resolution.

    (a) explicit override via CASCADE_C_PRIME_OUTPUT_DIR
    (b) Modal worker via /modal_results/${DISPATCH_INSTANCE_JOB_ID}/output
    (c) local/Vast.ai via $LOG_DIR/output
    """
    # Branch (a): explicit override
    assert 'if [ -n "${CASCADE_C_PRIME_OUTPUT_DIR:-}" ]; then' in lane_script_text
    # Branch (b): Modal canonical path
    assert 'MODAL_RUNTIME:-0' in lane_script_text
    assert '/modal_results/${DISPATCH_INSTANCE_JOB_ID}/output' in lane_script_text
    # Branch (c): local fallback
    assert 'OUTPUT_DIR="$LOG_DIR/output"' in lane_script_text


def test_lane_script_catalog_326_mode_env_multi_key_precedence(
    lane_script_text: str,
) -> None:
    """Catalog #326: mode env var consumption with multi-key precedence.

    CASCADE_C_PRIME_TRAINER_MODE > SMOKE_ONLY > default. Must FAIL-LOUD warning
    if neither key is set. Recipe env_overrides sets CASCADE_C_PRIME_TRAINER_MODE
    explicitly to avoid the Z6 Wave 2 4c dispatch bug class (driver default
    "smoke" while recipe intent "full").
    """
    # CASCADE_C_PRIME_TRAINER_MODE primary key
    assert 'CASCADE_C_PRIME_TRAINER_MODE' in lane_script_text
    # SMOKE_ONLY secondary key
    assert 'SMOKE_ONLY' in lane_script_text
    # Multi-key cascade (elif pattern)
    assert 'elif [ -n "${SMOKE_ONLY:-}" ]' in lane_script_text
    # FAIL-LOUD warning per Catalog #326 fail-loud invariant
    assert 'WARN' in lane_script_text
    assert 'Catalog #326' in lane_script_text


def test_lane_script_invokes_canonical_trainer_path(lane_script_text: str) -> None:
    """The lane script must invoke the canonical trainer wrapper path.

    Per the operator-authorize recipe `trainer_path` field. The trainer wrapper
    at experiments/train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py
    is sister subagent C's scope; THIS lane script invokes it canonically. If
    sister C has not yet built the wrapper, the lane script fails-fast with
    exit code 26 + diagnostic FATAL log.
    """
    assert (
        "experiments/train_substrate_cascade_c_prime_frame_1_segnet_waterfill.py"
        in lane_script_text
    )
    # Fail-fast probe for missing trainer
    assert 'if [ ! -f "$TRAINER_PATH" ]' in lane_script_text
    assert "exit 26" in lane_script_text


def test_lane_script_dispatch_claim_verification(lane_script_text: str) -> None:
    """Per CLAUDE.md CROSS-AGENT DISPATCH COORDINATION: claim verification."""
    assert "claim_lane_dispatch.py" in lane_script_text
    assert "DISPATCH_INSTANCE_JOB_ID" in lane_script_text
    assert "DISPATCH_CLAIMS_PATH" in lane_script_text


def test_lane_script_heartbeat_watchdog(lane_script_text: str) -> None:
    """Per CLAUDE.md "Remote code parity": heartbeat every ~5 min."""
    assert "heartbeat.log" in lane_script_text
    # Sleep 300 = 5 min
    assert "sleep 300" in lane_script_text


def test_lane_script_provenance_emission(lane_script_text: str) -> None:
    """Provenance file must be emitted with canonical fields."""
    assert "provenance.json" in lane_script_text
    # Per CLAUDE.md "Apples-to-apples evidence discipline" non-promotable defaults
    assert '"score_claim": False' in lane_script_text
    assert '"promotion_eligible": False' in lane_script_text


def test_lane_script_axis_tagged_completion_marker(lane_script_text: str) -> None:
    """Per CLAUDE.md "Apples-to-apples evidence discipline": axis-tagged marker."""
    # LANE_CASCADE_C_PRIME_DONE [contest-CUDA] / [contest-CPU] case-derived
    assert "LANE_CASCADE_C_PRIME_DONE" in lane_script_text
    assert "contest-" in lane_script_text


def test_lane_script_workspace_default(lane_script_text: str) -> None:
    """WORKSPACE defaults to /workspace/pact per canonical sister pattern."""
    assert 'WORKSPACE="${WORKSPACE:-/workspace/pact}"' in lane_script_text


def test_lane_script_lane_id_matches_recipe(lane_script_text: str) -> None:
    """LANE_ID must match the recipe's lane_id field (Catalog #126 pre-registration)."""
    assert 'LANE_ID="lane_cascade_c_prime_option_a_build_scaffold_20260526"' in lane_script_text


def test_lane_script_strips_macos_appledouble(lane_script_text: str) -> None:
    """macOS AppleDouble resource fork strip per sister NSCS06 pattern.

    Prevents `._*.mkv` files (created by macOS over SMB shares) from being
    picked up by upstream video probes.
    """
    assert "rm -f upstream/videos/._*.mkv" in lane_script_text


def test_lane_script_modal_aware_required_input_probe(lane_script_text: str) -> None:
    """Catalog #152 + #204 sister: multi-candidate probe for required input files.

    Sister of the resolve_required_input_modal_aware canonical helper pattern.
    Per the STC v2 driver fix (2026-05-17): defensive multi-candidate resolution
    is mandatory when required-input files live under Modal-IGNORED paths.
    """
    # Multi-candidate probe (Vast.ai + Modal-ro + Modal-rw)
    assert "/workspace/pact/upstream/videos/0.mkv" in lane_script_text
    assert "/tmp/pact/upstream/videos/0.mkv" in lane_script_text


def test_lane_script_stages_emit_log_markers(lane_script_text: str) -> None:
    """Each stage must emit log markers for operator-facing observability per Catalog #305."""
    expected_stages = [
        "stage_0b_nvdec_probe",
        "stage_1_bootstrap_runtime_deps",
        "stage_2_provenance",
        "stage_4_trainer_invoke",
    ]
    for stage in expected_stages:
        assert stage in lane_script_text, f"missing log marker for {stage}"


def test_lane_script_trap_cleans_heartbeat_pid(lane_script_text: str) -> None:
    """trap on EXIT must clean up heartbeat watchdog PID (no orphan background processes)."""
    assert "trap" in lane_script_text
    assert "HEARTBEAT_PID" in lane_script_text


def test_lane_script_modal_runtime_canonical_check(lane_script_text: str) -> None:
    """Modal-runtime branch must check both MODAL_RUNTIME=1 AND /modal_results dir."""
    # Catalog #204 canonical pattern: 3 conditions joined
    assert '"${MODAL_RUNTIME:-0}" = "1"' in lane_script_text
    assert '[ -d "/modal_results" ]' in lane_script_text
    assert '[ -n "${DISPATCH_INSTANCE_JOB_ID:-}" ]' in lane_script_text
