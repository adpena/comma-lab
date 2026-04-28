"""Lane PS-V2: structural tests for scripts/remote_lane_ps_v2_learnable_class_weights.sh.

V2-specific properties (the load-bearing ones for Lane PS-V2):
  - --learnable-segnet-class-weights flag is passed
  - --segnet-class-weights "1,5,5,1,1" is the warm-start (still passed)
  - --kl-distill-weight 1.0 (mandatory — Lane PS only fires on KL distill path)
  - [lane-ps-v2] banner sanity check
  - Predicted band [1.02, 1.18] in provenance

Plus all standard CLAUDE.md non-negotiable checks.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_ps_v2_learnable_class_weights.sh"
OPTIMIZE_POSES = REPO / "experiments" / "optimize_poses.py"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


@pytest.fixture(scope="module")
def optimize_poses_argparse_flags() -> set[str]:
    src = OPTIMIZE_POSES.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


# ── Existence + bash-safety guards ─────────────────────────────────────


def test_script_exists():
    assert SCRIPT.exists()


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK)


def test_full_set_euo_pipefail(script_text):
    assert "set -euo pipefail" in script_text


# ── Stage 0 NVDEC probe ────────────────────────────────────────────────


def test_nvdec_probe_present(script_text):
    assert "probe_nvdec.sh" in script_text


def test_nvdec_probe_before_optimize(script_text):
    """The EXECUTED probe_nvdec.sh must come before the EXECUTED
    optimize_poses.py invocation. Both names appear earlier in argparse-
    verification comments — those are docs, not invocations."""
    probe_match = re.search(r'^\s*bash\s+"\$WORKSPACE/scripts/probe_nvdec\.sh"',
                            script_text, re.MULTILINE)
    optimize_match = re.search(r'^\s*"\$PYBIN"\s+-u\s+experiments/optimize_poses\.py',
                                script_text, re.MULTILINE)
    assert probe_match is not None
    assert optimize_match is not None
    assert probe_match.start() < optimize_match.start()


# ── Anchor on Lane A ──────────────────────────────────────────────────


def test_anchors_on_lane_a(script_text):
    assert "experiments/results/lane_a_landed/iter_0/renderer.bin" in script_text
    assert "experiments/results/lane_a_landed/optimized_poses.pt" in script_text
    assert "experiments/results/lane_a_landed/extracted/masks.mkv" in script_text


# ── V2-specific: learnable flag present + warm-start CSV present ──────


def test_learnable_segnet_class_weights_flag_present(script_text):
    """Lane PS-V2 core: --learnable-segnet-class-weights must appear."""
    assert "--learnable-segnet-class-weights" in script_text


def test_learnable_lr_flag_present(script_text):
    assert "--learnable-segnet-class-weights-lr" in script_text


def test_learnable_var_lambda_flag_present(script_text):
    assert "--learnable-segnet-class-weights-var-lambda" in script_text


def test_warm_start_csv_present(script_text):
    """V2 still passes the V1-style CSV as the warm-start."""
    assert '"1,5,5,1,1"' in script_text or "'1,5,5,1,1'" in script_text


def test_kl_distill_weight_present(script_text):
    """Mandatory: --kl-distill-weight 1.0 (V2 only fires on KL distill path)."""
    assert "--kl-distill-weight 1.0" in script_text


def test_kl_distill_temperature_present(script_text):
    assert "--kl-distill-temperature 2.0" in script_text


# ── Banner sanity check ───────────────────────────────────────────────


def test_lane_ps_v2_banner_check_present(script_text):
    """Script must grep for [lane-ps-v2] banner to fail loud if the
    LEARNABLE path isn't active."""
    assert 'grep -q "lane-ps-v2"' in script_text


# ── Predicted band ────────────────────────────────────────────────────


def test_predicted_band_v2_recorded(script_text):
    """Lane PS-V2 band [1.02, 1.18] (slightly tighter floor than V1)."""
    assert "[1.02, 1.18]" in script_text


# ── Provenance + heartbeat ────────────────────────────────────────────


def test_provenance_json_emitted(script_text):
    assert "provenance.json" in script_text
    assert "git_hash" in script_text
    assert "gpu_name" in script_text


def test_heartbeat_emitted(script_text):
    assert "heartbeat" in script_text.lower()


# ── No MPS / CPU ──────────────────────────────────────────────────────


def test_no_mps_or_cpu_device(script_text):
    assert "--device mps" not in script_text
    assert "--device cpu" not in script_text
    assert "--device cuda" in script_text


# ── Python zipfile not shell zip ──────────────────────────────────────


def test_python_zipfile(script_text):
    bad = re.findall(r"^[^#\n]*\bzip\b\s+(?:-[a-z]+\s+)?\S+\.zip", script_text, re.M)
    assert not bad
    assert "zipfile.ZipFile" in script_text


# ── Container python ──────────────────────────────────────────────────


def test_uses_container_python(script_text):
    assert "/opt/conda/bin/python" in script_text
    assert ".venv/bin/python" not in script_text


def test_workspace_path_is_canonical(script_text):
    assert "WORKSPACE=/workspace/pact" in script_text


def test_env_sh_sourced(script_text):
    assert 'source "$WORKSPACE/env.sh"' in script_text


# ── End with contest_auth_eval ────────────────────────────────────────


def test_ends_with_contest_auth_eval(script_text):
    assert "contest_auth_eval.py" in script_text
    auth_idx = script_text.rfind("contest_auth_eval.py")
    optimize_idx = script_text.rfind("optimize_poses.py")
    assert auth_idx > optimize_idx


# ── Argparse-grep verification ────────────────────────────────────────


def _extract_invocation_flags(src: str, marker: str) -> set[str]:
    """Extract --flags from a single shell invocation block."""
    pat = rf"{re.escape(marker)}.+?(?=\n\S|\Z)"
    m = re.search(pat, src, re.DOTALL)
    if m is None:
        return set()
    return set(re.findall(r"\s--([a-z][a-z0-9-]+)", m.group(0)))


def test_optimize_poses_flags_are_real(script_text, optimize_poses_argparse_flags):
    """Every --flag passed to optimize_poses must exist in its argparse."""
    flags = _extract_invocation_flags(script_text, "optimize_poses.py")
    assert flags
    missing = flags - optimize_poses_argparse_flags
    assert not missing, (
        f"Lane PS-V2 invokes optimize_poses with invented flags: {missing}. "
        f"argparse has: {sorted(optimize_poses_argparse_flags)}"
    )


def test_contest_auth_eval_flags_are_real(script_text):
    flags = _extract_invocation_flags(script_text, "contest_auth_eval.py")
    assert flags
    real = {
        "archive", "inflate-sh", "upstream-dir", "device",
        "keep-work-dir", "work-dir", "video-names-file",
        "inflate-timeout", "evaluate-timeout",
    }
    invented = flags - real
    assert not invented, f"Invented: {invented}"


def test_no_codex_subagent(script_text):
    assert "/codex" not in script_text
