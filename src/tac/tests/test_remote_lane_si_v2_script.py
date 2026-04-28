"""Lane SI-V2: structural tests for scripts/remote_lane_si_v2_learnable_threshold.sh.

V2-specific properties:
  - target_bytes is set in the saliency-weighted compression call
  - apply_saliency_weighted_compression is invoked with `saliency=` and `target_bytes=`
  - tac.learnable_saliency_threshold module is referenced (or imported transitively)
  - Predicted band [1.05, 1.18] is documented in provenance

Plus all standard CLAUDE.md non-negotiable checks (set -euo pipefail, NVDEC
probe, provenance, heartbeat, python zipfile, no MPS fallback, real CLI flags).
"""
from __future__ import annotations

import os
import re
import stat
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_si_v2_learnable_threshold.sh"
PROFILE_SAL = REPO / "experiments" / "profile_scorer_saliency.py"


@pytest.fixture(scope="module")
def script_src() -> str:
    return SCRIPT.read_text()


def test_script_exists() -> None:
    assert SCRIPT.is_file()


def test_script_is_executable() -> None:
    mode = os.stat(SCRIPT).st_mode
    assert mode & stat.S_IXUSR, "script must be chmod +x"


def test_set_euo_pipefail_present(script_src) -> None:
    assert "set -euo pipefail" in script_src
    bad_lines = [ln for ln in script_src.splitlines() if re.search(r"^\s*set\s+-uo\s", ln)]
    assert not bad_lines, f"Forbidden `set -uo`: {bad_lines}"


def test_nvdec_probe_stage_0(script_src) -> None:
    assert "probe_nvdec.sh" in script_src


def test_provenance_json_written(script_src) -> None:
    assert "provenance.json" in script_src
    assert "git_hash" in script_src or "GIT_HASH" in script_src
    assert "gpu_name" in script_src.lower() or "GPU_NAME" in script_src


def test_heartbeat_log_written(script_src) -> None:
    assert "heartbeat" in script_src.lower()


def test_python_zipfile_not_shell_zip(script_src) -> None:
    bad = re.findall(r"^\s*zip\s+", script_src, re.MULTILINE)
    assert not bad
    assert "zipfile.ZipFile" in script_src


def test_no_cpu_or_mps_device(script_src) -> None:
    assert "--device cpu" not in script_src
    assert "--device mps" not in script_src
    assert "--device cuda" in script_src


def test_stage_4_runs_contest_auth_eval(script_src) -> None:
    assert "contest_auth_eval.py" in script_src


def test_predicted_band_v2_recorded(script_src) -> None:
    """Lane SI-V2 band [1.05, 1.18] (slightly tighter floor than V1)."""
    assert "[1.05, 1.18]" in script_src


def test_v2_uses_target_bytes(script_src) -> None:
    """Lane SI-V2 core: target_bytes in apply_saliency_weighted_compression."""
    assert "target_bytes" in script_src, (
        "Lane SI-V2 must pass target_bytes= to "
        "apply_saliency_weighted_compression (Lagrangian threshold)."
    )


def test_v2_imports_apply_saliency_weighted_compression(script_src) -> None:
    """The V2 stage 2 block imports the V2-aware function."""
    assert "apply_saliency_weighted_compression" in script_src


def test_v2_passes_saliency_keyword(script_src) -> None:
    """V2 mode REQUIRES the raw saliency argument (not just saliency_inv)."""
    # Either `saliency=sal` or `saliency=` is fine
    assert "saliency=" in script_src


def test_uses_container_python(script_src) -> None:
    assert "/opt/conda/bin/python" in script_src
    assert ".venv/bin/python" not in script_src


def test_workspace_path_is_canonical(script_src) -> None:
    assert "WORKSPACE=/workspace/pact" in script_src


def test_env_sh_sourced(script_src) -> None:
    assert 'source "$WORKSPACE/env.sh"' in script_src


def test_documents_outstanding_inflate_decoder_todo(script_src) -> None:
    """Lane SI-V2 still defers Component 4. Header must document this."""
    has_todo = (
        "OUTSTANDING TODO" in script_src
        or "Component 4" in script_src
        or "inflate-time decoder" in script_src.lower()
    )
    assert has_todo


def test_required_artifact_preflight_present(script_src) -> None:
    assert "renderer.bin" in script_src
    assert "optimized_poses.pt" in script_src
    assert "posenet.safetensors" in script_src
    assert "segnet.safetensors" in script_src


def test_invokes_profile_scorer_saliency_with_real_flags(script_src) -> None:
    profile_block = re.search(
        r"profile_scorer_saliency\.py(.+?)2>&1", script_src, re.DOTALL
    )
    assert profile_block is not None
    flags = set(re.findall(r"--[\w-]+", profile_block.group(1)))
    expected = {
        "--checkpoint", "--poses", "--masks-mkv", "--video", "--output",
        "--device", "--upstream-dir", "--n-pairs", "--reduce",
    }
    missing = expected - flags
    assert not missing, (
        f"Lane SI-V2 doesn't pass expected flags to profile_scorer_saliency: {missing}"
    )


def test_invokes_contest_auth_eval_with_real_flags(script_src) -> None:
    eval_block = re.search(r"contest_auth_eval\.py(.+?)2>&1", script_src, re.DOTALL)
    assert eval_block is not None
    flags = set(re.findall(r"--[\w-]+", eval_block.group(1)))
    real_flags = {
        "--archive", "--inflate-sh", "--upstream-dir", "--video-names-file",
        "--device", "--work-dir", "--inflate-timeout", "--evaluate-timeout",
        "--keep-work-dir",
    }
    invented = flags - real_flags
    assert not invented
