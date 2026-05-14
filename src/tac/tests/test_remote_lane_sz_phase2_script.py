# SPDX-License-Identifier: MIT
"""Structural validation for ``scripts/remote_lane_sz_phase2_full.sh``.

Per CLAUDE.md non-negotiables:
  * ``set -euo pipefail`` (Lane B post-mortem ``feedback_zip_dep_bootstrap_trap``).
  * NVDEC probe at Stage 0 (``feedback_vastai_nvdec_host_variation``).
  * Sources ``env.sh`` (``feedback_canonical_remote_bootstraps``).
  * Writes ``provenance.json`` + ``heartbeat.log``.
  * Uses python ``zipfile`` (NOT shell ``zip``) for archive build.
  * Records ``predicted_band`` in provenance.
  * Every CLI flag passed to a Python script exists in that script's argparse
    (``feedback_dead_flag_wiring_pattern``).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT = REPO_ROOT / "scripts" / "remote_lane_sz_phase2_full.sh"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


# ── Bootstrap discipline ──────────────────────────────────────────────────


def test_script_exists(script_text):
    assert SCRIPT.is_file()
    assert script_text.startswith("#!"), "missing shebang"


def test_set_euo_pipefail(script_text):
    # Must have all three: -e (fail on error), -u (fail on unset var),
    # pipefail (catch errors mid-pipeline). Lane B post-mortem.
    assert "set -euo pipefail" in script_text


def test_sources_env_sh(script_text):
    assert 'source "$WORKSPACE/env.sh"' in script_text


def test_workspace_pinned(script_text):
    assert "WORKSPACE=/workspace/pact" in script_text


def test_uses_container_python(script_text):
    # /opt/conda/bin/python is the canonical PyTorch container python
    # (feedback_canonical_remote_bootstraps).
    assert "PYBIN=/opt/conda/bin/python" in script_text


# ── Provenance + heartbeat ────────────────────────────────────────────────


def test_writes_provenance(script_text):
    assert 'PROVENANCE="$LOG_DIR/provenance.json"' in script_text
    assert "predicted_band" in script_text


def test_predicted_band_matches_target(script_text):
    # The lane-level predicted band must match the szabolcs PR#56 reproduction
    # target. If it drifts, the council needs to re-evaluate the lane premise.
    m = re.search(r"'predicted_band'\s*:\s*\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", script_text)
    assert m is not None, "provenance must record predicted_band as [low, high]"
    low, high = float(m.group(1)), float(m.group(2))
    assert (low, high) == (0.30, 0.50), (
        f"predicted_band drifted to ({low}, {high}); szabolcs target is (0.30, 0.50)."
    )


def test_writes_heartbeat(script_text):
    assert 'HEARTBEAT="$LOG_DIR/heartbeat.log"' in script_text
    # Heartbeat loop runs in background.
    assert "while true; do" in script_text


# ── NVDEC probe ───────────────────────────────────────────────────────────


def test_stage_0_nvdec_probe(script_text):
    # NVDEC probe runs at Stage 0 BEFORE GPU spend; per
    # feedback_vastai_nvdec_host_variation it MAY warn-and-continue for Lane SZ
    # because training uses pyav CPU decode, but the probe call itself must
    # still be present.
    assert "Stage 0: NVDEC probe" in script_text
    assert "probe_nvdec.sh" in script_text


# ── Archive build uses python zipfile (NOT shell zip) ────────────────────


def test_archive_uses_python_zipfile(script_text):
    # Lane B post-mortem: PyTorch container has no shell `zip` binary.
    # Python's zipfile is the canonical packaging path.
    assert "import zipfile" in script_text
    # And we should NEVER call shell `zip` directly. A bare `zip ` token
    # (followed by space) anywhere in the script body would be a regression.
    # (We allow "zipfile" / "ZIP_DEFLATED" because those are python module
    # names.)
    nonpython_zip = re.search(r"(^|\s)zip\s+-", script_text, re.MULTILINE)
    assert nonpython_zip is None, (
        f"shell `zip` invocation detected at: {nonpython_zip.group(0)!r}"
    )


def test_archive_szabolcs_paradigm(script_text):
    # Lane SZ archive contains renderer.bin ONLY. The script must NOT bundle
    # masks.mkv or optimized_poses.pt (which would defeat the entire paradigm).
    # We check the python zipfile.ZipFile invocation in Stage 3, isolated
    # from log/comment lines (which legitimately MENTION the absent files
    # for documentation).
    stage3 = re.search(
        r'log "=== Stage 3:.*?log "=== Stage 4:', script_text, re.DOTALL,
    )
    assert stage3 is not None, "missing Stage 3 banner"
    block = stage3.group(0)
    # Extract just the python heredoc that builds the zip.
    py = re.search(r'with zipfile\.ZipFile.*?print\(', block, re.DOTALL)
    assert py is not None, "Stage 3 must invoke zipfile.ZipFile to build archive"
    py_block = py.group(0)
    assert "masks.mkv" not in py_block, (
        "Lane SZ archive must NOT include masks.mkv (LUT is reconstructed in code)"
    )
    assert "optimized_poses.pt" not in py_block, (
        "Lane SZ archive must NOT include optimized_poses.pt (per-frame affine "
        "embedding lives inside the renderer state)"
    )
    assert "renderer.bin" in py_block


# ── Flag-arity (no invented CLI flags) ────────────────────────────────────


def _real_flags(path: Path) -> set[str]:
    src = path.read_text()
    return set(re.findall(r"add_argument\(\s*[\"\']--([a-z][a-z0-9-]+)", src))


def _flags_in_block(block: str) -> set[str]:
    return set(re.findall(r"\B--([a-z][a-z0-9-]+)", block))


def test_train_flags_match_argparse(script_text):
    real = _real_flags(REPO_ROOT / "experiments" / "train_szabolcs.py")
    stage1 = re.search(
        r'log "=== Stage 1:.*?log "=== Stage 2:', script_text, re.DOTALL,
    )
    assert stage1 is not None
    used = _flags_in_block(stage1.group(0))
    bogus = used - real
    assert not bogus, f"invented flags in train stage: {sorted(bogus)}"


def test_export_flags_match_argparse(script_text):
    real = _real_flags(REPO_ROOT / "experiments" / "export_szabolcs_archive.py")
    stage2 = re.search(
        r'log "=== Stage 2:.*?log "=== Stage 3:', script_text, re.DOTALL,
    )
    assert stage2 is not None
    used = _flags_in_block(stage2.group(0))
    bogus = used - real
    assert not bogus, f"invented flags in export stage: {sorted(bogus)}"


def test_auth_eval_flags_match_argparse(script_text):
    real = _real_flags(REPO_ROOT / "experiments" / "contest_auth_eval.py")
    stage4 = re.search(
        r'log "=== Stage 4:.*?(?:log "=== LANE_SZ_DONE|\Z)',
        script_text, re.DOTALL,
    )
    assert stage4 is not None
    used = _flags_in_block(stage4.group(0))
    bogus = used - real
    assert not bogus, f"invented flags in auth_eval stage: {sorted(bogus)}"


# ── Stage markers (so future updates don't silently drop a stage) ────────


def test_all_four_stages_present(script_text):
    for marker in (
        "Stage 0: NVDEC probe",
        "Stage 1: train szabolcs",
        "Stage 2: SZv1 export",
        "Stage 3: build archive",
        "Stage 4: contest_auth_eval",
        "LANE_SZ_DONE",
    ):
        assert marker in script_text, f"missing stage marker: {marker!r}"


def test_result_json_validation(script_text):
    # Lane B post-mortem: the auth eval MUST emit a RESULT_JSON line and the
    # script MUST validate it. Without this, a silent crash leaves a green
    # exit code and we never notice the eval failed.
    assert "RESULT_JSON" in script_text
    assert "auth_eval log has no RESULT_JSON" in script_text
