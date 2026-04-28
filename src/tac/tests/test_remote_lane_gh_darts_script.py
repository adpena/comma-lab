"""Script-structure tests for scripts/remote_lane_gh_darts_ratio.sh.

Pins every CLAUDE.md non-negotiable on the new DARTS-search bootstrap:
  * `set -euo pipefail`
  * `--device cuda` (no MPS / CPU fallback)
  * Stage 0 NVDEC probe BEFORE any GPU spend
  * provenance.json + heartbeat.log
  * Python zipfile (no shell `zip`)
  * Completion marker `LANE_GH_DARTS_DONE`
  * `[contest-CUDA-DARTS-search]` lane tag in the discovered-arch JSON
  * Candidate ratios match :data:`GHOST_RATIO_CANDIDATES`
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_gh_darts_ratio.sh"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


def test_script_exists():
    assert SCRIPT.exists(), f"missing Lane GH-DARTS script: {SCRIPT}"


def test_script_executable():
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be chmod +x"


def test_set_euo_pipefail(script_text: str):
    assert "set -euo pipefail" in script_text


def test_no_mps_or_cpu_device(script_text: str):
    assert "--device mps" not in script_text
    assert "--device cpu" not in script_text


def test_uses_cuda_device(script_text: str):
    # CUDA reference can be `torch.device('cuda')` or `--device cuda`.
    assert "torch.device('cuda')" in script_text or "--device cuda" in script_text


def test_nvdec_probe_present(script_text: str):
    assert "probe_nvdec.sh" in script_text
    # Must be in Stage 0 (anchor at $WORKSPACE).
    assert 'bash "$WORKSPACE/scripts/probe_nvdec.sh"' in script_text


def test_nvdec_probe_fails_loud(script_text: str):
    section = re.search(r"probe_nvdec\.sh.*?exit\s+\d+", script_text, re.DOTALL)
    assert section is not None, "NVDEC probe must abort with `exit N` on failure"


def test_provenance_json_written(script_text: str):
    assert "provenance.json" in script_text


def test_heartbeat_log_written(script_text: str):
    assert "heartbeat.log" in script_text


def test_no_shell_zip_binary(script_text: str):
    code = "\n".join(
        line for line in script_text.splitlines()
        if not line.strip().startswith("#")
    )
    bad = re.search(r"(^|[\s;&|`\(])zip\s+(?!file)", code)
    assert not bad, f"shell zip forbidden; got {bad.group(0) if bad else None!r}"


def test_completion_marker_present(script_text: str):
    assert "LANE_GH_DARTS_DONE" in script_text


def test_lane_tag_contest_cuda_darts(script_text: str):
    """Every score must carry a lane tag (CLAUDE.md non-negotiable)."""
    assert "[contest-CUDA-DARTS-search]" in script_text


def test_log_dir_namespaced(script_text: str):
    assert "lane_gh_darts_results" in script_text


def test_tag_namespaced(script_text: str):
    assert 'TAG="lane_gh_darts_ratio"' in script_text


def test_internal_lane_name_lane_gh_darts(script_text: str):
    assert "[lane-gh-darts]" in script_text
    assert "lane=GH-DARTS" in script_text


def test_provenance_records_darts_method(script_text: str):
    assert "darts_method" in script_text
    assert "first-order DARTS" in script_text


def test_provenance_records_temperature_schedule(script_text: str):
    assert "darts_temperature_schedule" in script_text
    assert "5.0" in script_text and "0.1" in script_text


def test_provenance_records_predicted_band(script_text: str):
    assert "predicted_band_post_retrain" in script_text


def test_provenance_records_anchor_baseline(script_text: str):
    assert "anchor_score_baseline" in script_text


def test_candidate_ratios_match_module_constant(script_text: str):
    """Pre-flight stage must verify the candidate ratios equal
    :data:`GHOST_RATIO_CANDIDATES` so this script + the module never
    silently drift."""
    # Locked tuple referenced in pre-flight.
    assert "(1.5, 2.0, 2.5, 3.0, 4.0)" in script_text


def test_alpha_trajectory_artifact_named(script_text: str):
    """CLAUDE.md algorithmic-rigor: each search reports the FULL α
    evolution trajectory."""
    assert "alpha_trajectory.json" in script_text


def test_discovered_arch_artifact_named(script_text: str):
    assert "discovered_arch.json" in script_text


def test_uses_workspace_pact(script_text: str):
    assert "WORKSPACE=/workspace/pact" in script_text


def test_uses_container_python(script_text: str):
    """CLAUDE.md feedback_canonical_remote_bootstraps: use container
    Python /opt/conda/bin/python (NOT a venv)."""
    assert "PYBIN=/opt/conda/bin/python" in script_text


def test_pythonhashseed_pinned(script_text: str):
    assert "PYTHONHASHSEED=1234" in script_text
