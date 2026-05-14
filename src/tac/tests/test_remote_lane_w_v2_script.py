# SPDX-License-Identifier: MIT
"""Lane W-V2: structural tests for scripts/remote_lane_w_v2_learnable_hardness.sh.

Locks in the bug-class compliance from CLAUDE.md non-negotiables:
  - set -euo pipefail (memory: feedback_zip_dep_bootstrap_trap)
  - python zipfile not shell zip
  - --device cuda only (no MPS fallback)
  - NVDEC probe BEFORE GPU spend (memory: feedback_vastai_nvdec_host_variation)
  - provenance.json + heartbeat.log + run_record.json
    (memory: feedback_canonical_remote_bootstraps)
  - argparse-grep-verified flags (memory: feedback_dead_flag_wiring_pattern)
  - container python /opt/conda/bin/python (NOT venv)

V2-specific properties (the load-bearing ones for Lane W-V2):
  - --learnable-pair-weights flag is passed to train_renderer
  - --mode continuous is passed to profile_pair_sensitivity
  - Predicted band [0.85, 1.05] is documented in provenance
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_w_v2_learnable_hardness.sh"
TRAIN_RENDERER = REPO / "src" / "tac" / "experiments" / "train_renderer.py"
PROFILE_PAIR = REPO / "experiments" / "profile_pair_sensitivity.py"


@pytest.fixture(scope="module")
def script_src() -> str:
    return SCRIPT.read_text()


@pytest.fixture(scope="module")
def train_renderer_flags() -> set[str]:
    src = TRAIN_RENDERER.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


@pytest.fixture(scope="module")
def profile_pair_flags() -> set[str]:
    src = PROFILE_PAIR.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


def test_script_exists():
    assert SCRIPT.exists()
    assert SCRIPT.stat().st_mode & 0o111, "script must be executable"


def test_strict_mode_set(script_src):
    assert "set -euo pipefail" in script_src
    assert "set -uo pipefail" not in script_src


def test_no_shell_zip_binary(script_src):
    bad = re.findall(r"^[^#\n]*\bzip\b\s+(?:-[a-z]+\s+)?\S+\.zip", script_src, re.M)
    assert not bad, f"Forbidden shell zip binary: {bad}"
    assert "zipfile.ZipFile" in script_src


def test_device_cuda_no_mps_fallback(script_src):
    assert "--device mps" not in script_src
    assert "--device cpu" not in script_src
    cuda_count = script_src.count("--device cuda")
    assert cuda_count >= 3, f"Expected >=3 --device cuda; got {cuda_count}"


def test_nvdec_probe_before_train(script_src):
    """The EXECUTED probe_nvdec.sh call must appear before the EXECUTED
    train_renderer call. Both names appear earlier in argparse-verification
    comments — those are documentation, not invocations. Match the actual
    `bash` invocation of the probe and the `python -u -m` invocation of
    train_renderer."""
    probe_match = re.search(r'^\s*bash\s+"\$WORKSPACE/scripts/probe_nvdec\.sh"',
                            script_src, re.MULTILINE)
    train_match = re.search(r'^\s*"\$PYBIN"\s+-u\s+-m\s+tac\.experiments\.train_renderer',
                            script_src, re.MULTILINE)
    assert probe_match is not None, "probe_nvdec.sh must be invoked"
    assert train_match is not None, "train_renderer must be invoked via -u -m"
    assert probe_match.start() < train_match.start(), (
        "NVDEC probe must execute BEFORE train_renderer (memory: "
        "feedback_vastai_nvdec_host_variation)."
    )


def test_provenance_json_emitted(script_src):
    assert "provenance.json" in script_src
    assert "git_hash" in script_src
    assert "gpu_name" in script_src
    assert "predicted_band" in script_src


def test_heartbeat_emitted(script_src):
    assert "heartbeat.log" in script_src
    assert "trap" in script_src and "kill" in script_src


def test_run_record_emitted(script_src):
    assert "run_record.json" in script_src


def test_uses_container_python(script_src):
    assert "/opt/conda/bin/python" in script_src
    assert ".venv/bin/python" not in script_src


def test_anchors_on_lane_a(script_src):
    assert "experiments/results/lane_a_landed/iter_0/renderer.bin" in script_src
    assert "experiments/results/lane_a_landed/iter_0/optimized_poses.pt" in script_src
    assert "experiments/results/lane_a_landed/iter_0/masks.mkv" in script_src


def test_predicted_band_v2(script_src):
    """V2 band is tighter than V1: [0.85, 1.05]."""
    assert "[0.85, 1.05]" in script_src


def test_self_compress_flag_present(script_src):
    assert "--use-self-compress-codec" in script_src


def test_learnable_pair_weights_flag_present(script_src):
    """Lane W-V2 core: --learnable-pair-weights must appear."""
    assert "--learnable-pair-weights" in script_src


def test_pair_loss_weights_flag_present(script_src):
    """V2 still passes --pair-loss-weights as the warm-start tensor."""
    assert "--pair-loss-weights" in script_src


def test_profile_uses_continuous_mode(script_src):
    """Lane W-V2 uses --mode continuous to produce the warm-start."""
    assert "--mode continuous" in script_src


def test_no_auth_eval_on_best_in_train(script_src):
    assert "--no-auth-eval-on-best" in script_src


def test_resume_from_anchor(script_src):
    m = re.search(
        r"--resume-from\s+\"\$ANCHOR_RENDERER\"",
        script_src,
    )
    assert m is not None


def test_archive_uses_lane_a_masks_and_poses(script_src):
    assert re.search(
        r"cp\s+\"\$ANCHOR_MASKS\"\s+\"\$LOG_DIR/iter_0/masks\.mkv\"", script_src
    )
    assert re.search(
        r"cp\s+\"\$ANCHOR_POSES\"\s+\"\$LOG_DIR/iter_0/optimized_poses\.pt\"",
        script_src,
    )


def test_contest_auth_eval_at_end(script_src):
    auth_idx = script_src.rfind("contest_auth_eval.py")
    assert auth_idx > 0
    train_idx = script_src.rfind("train_renderer")
    assert auth_idx > train_idx


def test_workspace_path_is_canonical(script_src):
    assert "WORKSPACE=/workspace/pact" in script_src


def test_env_sh_sourced(script_src):
    assert 'source "$WORKSPACE/env.sh"' in script_src


def test_no_pipefail_grep_q_trap(script_src):
    assert "grep -q" not in script_src or "set +o pipefail" in script_src


def test_script_does_not_invoke_codex_or_subagent(script_src):
    assert "/codex" not in script_src


# ── Argparse-grep verification (NEVER invent CLI flags) ────────────────


def _extract_invocation_flags(src: str, marker: str) -> set[str]:
    """Extract --flags from a single shell invocation block.

    A shell invocation in our scripts is a backslash-continued list of
    lines that starts after `marker` and ends at the FIRST line not
    ending in a backslash. This regex captures exactly that block, so
    flags from later invocations don't leak in.
    """
    # Find marker, then capture everything until the first non-backslash line.
    pat = rf"{re.escape(marker)}.+?(?=\n\S|\Z)"
    m = re.search(pat, src, re.DOTALL)
    if m is None:
        return set()
    return set(re.findall(r"\s--([a-z][a-z0-9-]+)", m.group(0)))


def test_train_renderer_flags_are_real(script_src, train_renderer_flags):
    """Every --flag passed to train_renderer must exist in its argparse."""
    flags = _extract_invocation_flags(script_src, "tac.experiments.train_renderer")
    assert flags, "Could not extract flags from train_renderer invocation"
    missing = flags - train_renderer_flags
    assert not missing, (
        f"Lane W-V2 invokes train_renderer with invented flags: {missing}. "
        f"Available flags: {sorted(train_renderer_flags)[:20]}..."
    )


def test_profile_pair_sensitivity_flags_are_real(script_src, profile_pair_flags):
    """Every --flag passed to profile_pair_sensitivity must exist."""
    flags = _extract_invocation_flags(script_src, "profile_pair_sensitivity.py")
    assert flags, "Could not extract flags from profile_pair_sensitivity invocation"
    missing = flags - profile_pair_flags
    assert not missing, (
        f"Lane W-V2 invokes profile_pair_sensitivity with invented flags: "
        f"{missing}"
    )


def test_contest_auth_eval_flags_are_real(script_src):
    """Whitelist for contest_auth_eval.py argparse (verified 2026-04-27)."""
    flags = _extract_invocation_flags(script_src, "contest_auth_eval.py")
    assert flags, "Could not extract flags from contest_auth_eval invocation"
    real = {
        "archive", "inflate-sh", "upstream-dir", "device",
        "keep-work-dir", "work-dir", "video-names-file",
        "inflate-timeout", "evaluate-timeout",
    }
    invented = flags - real
    assert not invented, (
        f"Lane W-V2 invokes contest_auth_eval with invented flags: {invented}"
    )
