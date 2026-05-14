# SPDX-License-Identifier: MIT
"""Script-structure tests for scripts/remote_lane_k_darts_channels.sh."""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_k_darts_channels.sh"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


def test_script_exists():
    assert SCRIPT.exists(), f"missing Lane K-DARTS script: {SCRIPT}"


def test_script_executable():
    assert os.access(SCRIPT, os.X_OK)


def test_set_euo_pipefail(script_text: str):
    assert "set -euo pipefail" in script_text


def test_no_mps_or_cpu_device(script_text: str):
    assert "--device mps" not in script_text
    assert "--device cpu" not in script_text


def test_uses_cuda_device(script_text: str):
    assert "torch.device('cuda')" in script_text or "--device cuda" in script_text


def test_nvdec_probe_present(script_text: str):
    assert "probe_nvdec.sh" in script_text
    assert 'bash "$WORKSPACE/scripts/probe_nvdec.sh"' in script_text


def test_nvdec_probe_fails_loud(script_text: str):
    assert re.search(r"probe_nvdec\.sh.*?exit\s+\d+", script_text, re.DOTALL)


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
    assert not bad


def test_completion_marker_present(script_text: str):
    assert "LANE_K_DARTS_DONE" in script_text


def test_lane_tag_contest_cuda_darts(script_text: str):
    assert "[contest-CUDA-DARTS-search]" in script_text


def test_log_dir_namespaced(script_text: str):
    assert "lane_k_darts_results" in script_text


def test_tag_namespaced(script_text: str):
    assert 'TAG="lane_k_darts_channels"' in script_text


def test_internal_lane_name_lane_k_darts(script_text: str):
    assert "[lane-k-darts]" in script_text
    assert "lane=K-DARTS" in script_text


def test_provenance_records_darts_method(script_text: str):
    assert "darts_method" in script_text


def test_provenance_records_param_budget(script_text: str):
    assert "param_budget" in script_text
    assert "100000" in script_text or "100_000" in script_text


def test_provenance_records_predicted_band(script_text: str):
    assert "predicted_band_post_retrain" in script_text


def test_candidate_channels_recorded(script_text: str):
    """Pre-flight + provenance both must reference the canonical lists."""
    assert "(16, 24, 32, 48)" in script_text  # base_channels
    assert "(24, 32, 48, 64)" in script_text  # mid_channels


def test_uses_param_budget_penalty(script_text: str):
    """CLAUDE.md mathematical rigor — budget penalty must be applied
    so DARTS actually trades capacity for rate."""
    assert "param_budget_penalty" in script_text


def test_alpha_trajectory_artifact_named(script_text: str):
    assert "alpha_trajectory.json" in script_text


def test_discovered_arch_artifact_named(script_text: str):
    assert "discovered_arch.json" in script_text


def test_pythonhashseed_pinned(script_text: str):
    assert "PYTHONHASHSEED=1234" in script_text


def test_container_python_used(script_text: str):
    assert "PYBIN=/opt/conda/bin/python" in script_text
