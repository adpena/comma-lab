# SPDX-License-Identifier: MIT
"""Lane W: structural tests for scripts/remote_lane_w_hard_pair_self_compress.sh.

Locks in the bug-class compliance from CLAUDE.md non-negotiables:
  - set -euo pipefail (memory: feedback_zip_dep_bootstrap_trap)
  - python zipfile not shell zip
  - --device cuda only (no MPS fallback)
  - NVDEC probe BEFORE GPU spend (memory: feedback_vastai_nvdec_host_variation)
  - provenance.json + heartbeat.log + run_record.json
    (memory: feedback_canonical_remote_bootstraps)
  - argparse-grep-verified flags (memory: feedback_dead_flag_wiring_pattern)
  - container python /opt/conda/bin/python (NOT venv)
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_w_hard_pair_self_compress.sh"


@pytest.fixture(scope="module")
def script_src() -> str:
    return SCRIPT.read_text()


def test_script_exists():
    assert SCRIPT.exists()
    assert SCRIPT.stat().st_mode & 0o111, "script must be executable"


def test_strict_mode_set(script_src):
    """set -euo pipefail (NOT -uo without -e)."""
    assert "set -euo pipefail" in script_src, (
        "Lane W script must use `set -euo pipefail` per "
        "feedback_zip_dep_bootstrap_trap. `set -uo` (without -e) was the "
        "exact LANE-B 6.5h+$2 silent-cascade trap."
    )
    # Make sure no one else snuck in `set -uo`.
    assert "set -uo pipefail" not in script_src


def test_no_shell_zip_binary(script_src):
    """PyTorch container has no `zip`. Use python zipfile.
    feedback_zip_dep_bootstrap_trap."""
    # Forbidden: a bare `zip ` command line invocation. We allow the word
    # `zipfile` (python module) and `unzip` (decompression — no `zip` symlink).
    bad = re.findall(r"^[^#\n]*\bzip\b\s+(?:-[a-z]+\s+)?\S+\.zip", script_src, re.M)
    assert not bad, (
        f"Forbidden shell zip binary call(s): {bad}. Use python zipfile "
        f"(memory: feedback_zip_dep_bootstrap_trap)."
    )
    assert "zipfile.ZipFile" in script_src, (
        "Archive build must use python `zipfile.ZipFile` per "
        "feedback_zip_dep_bootstrap_trap."
    )


def test_device_cuda_no_mps_fallback(script_src):
    """Every Python invocation must pass --device cuda (no MPS or CPU)."""
    assert "--device mps" not in script_src
    assert "--device cpu" not in script_src
    # And every script we invoke that takes --device must specify it.
    cuda_count = script_src.count("--device cuda")
    assert cuda_count >= 3, (
        f"Expected --device cuda on at least 3 invocations (profile, train, "
        f"contest_auth_eval); got {cuda_count}."
    )


def test_nvdec_probe_before_train(script_src):
    """probe_nvdec.sh must run BEFORE train_renderer.py is invoked."""
    probe_idx = script_src.find("probe_nvdec.sh")
    train_idx = script_src.find("train_renderer")
    assert probe_idx > 0, "probe_nvdec.sh not invoked"
    assert train_idx > 0, "train_renderer not invoked"
    assert probe_idx < train_idx, (
        "NVDEC probe must run BEFORE train_renderer (memory: "
        "feedback_vastai_nvdec_host_variation). Otherwise we burn 30 min "
        "of setup discovering a missing NVDEC at eval stage."
    )


def test_provenance_json_emitted(script_src):
    """canonical bootstrap requirement (feedback_canonical_remote_bootstraps)."""
    assert "provenance.json" in script_src
    # Must include git_hash + gpu_name (the required provenance keys)
    assert "git_hash" in script_src
    assert "gpu_name" in script_src
    assert "predicted_band" in script_src


def test_heartbeat_emitted(script_src):
    """Per CLAUDE.md "Tmux session existence is NOT a heartbeat"."""
    assert "heartbeat.log" in script_src
    assert "trap" in script_src and "kill" in script_src, (
        "heartbeat process must be cleaned up via trap-on-EXIT."
    )


def test_run_record_emitted(script_src):
    """canonical bootstrap requirement."""
    assert "run_record.json" in script_src


def test_uses_container_python(script_src):
    """Container python /opt/conda/bin/python (NOT venv).
    Per memory feedback_canonical_remote_bootstraps."""
    assert "/opt/conda/bin/python" in script_src
    # Make sure we don't accidentally use a venv path.
    assert ".venv/bin/python" not in script_src


def test_anchors_on_lane_a(script_src):
    """Lane W premise: anchored on Lane A (1.15) frontier."""
    assert "experiments/results/lane_a_landed/iter_0/renderer.bin" in script_src
    assert "experiments/results/lane_a_landed/iter_0/optimized_poses.pt" in script_src
    assert "experiments/results/lane_a_landed/iter_0/masks.mkv" in script_src


def test_predicted_band_present(script_src):
    """The predicted score band must be machine-readable in provenance."""
    assert "[0.85, 1.10]" in script_src, (
        "predicted_band must match the council-set [0.85, 1.10] for Lane W."
    )


def test_self_compress_flag_present(script_src):
    """SC codec must be enabled in the train invocation."""
    assert "--use-self-compress-codec" in script_src


def test_pair_loss_weights_flag_present(script_src):
    """The Lane W flag must be plumbed into the train invocation."""
    assert "--pair-loss-weights" in script_src


def test_no_auth_eval_on_best_in_train(script_src):
    """SC training auto-disables auth_eval_on_best (architecturally), but
    we pass --no-auth-eval-on-best explicitly so the intent is loud and
    the future maintainer doesn't accidentally re-enable it without
    realising SCv1 export is a separate path."""
    assert "--no-auth-eval-on-best" in script_src


def test_resume_from_anchor(script_src):
    """Train must resume from Lane A's renderer (the 1.15 frontier)."""
    m = re.search(
        r"--resume-from\s+\"\$ANCHOR_RENDERER\"",
        script_src,
    )
    assert m is not None, (
        "Train invocation must --resume-from $ANCHOR_RENDERER. "
        "Without it, Lane W trains from random init and discards the 1.15 anchor."
    )


def test_archive_uses_lane_a_masks_and_poses(script_src):
    """Lane W swaps ONLY the renderer; masks + poses are Lane A's exact bytes."""
    # cp $ANCHOR_MASKS into iter_0/masks.mkv
    assert re.search(r"cp\s+\"\$ANCHOR_MASKS\"\s+\"\$LOG_DIR/iter_0/masks\.mkv\"", script_src)
    # cp $ANCHOR_POSES into iter_0/optimized_poses.pt
    assert re.search(r"cp\s+\"\$ANCHOR_POSES\"\s+\"\$LOG_DIR/iter_0/optimized_poses\.pt\"", script_src)


def test_contest_auth_eval_at_end(script_src):
    """The ONLY trustworthy score is contest_auth_eval on the BUILT archive
    (CLAUDE.md non-negotiable: Auth eval EVERYWHERE)."""
    # Must be the LAST significant Python invocation
    auth_idx = script_src.rfind("contest_auth_eval.py")
    assert auth_idx > 0, "contest_auth_eval.py must be invoked"
    train_idx = script_src.rfind("train_renderer")
    assert auth_idx > train_idx, (
        "contest_auth_eval.py must run AFTER train_renderer. Otherwise we "
        "score the wrong artifact."
    )


def test_scv1_export_uses_canonical_function(script_src):
    """SCv1 packing must use tac.renderer_export.export_self_compressed_renderer
    (NOT a fictional export_scv1)."""
    assert "export_self_compressed_renderer" in script_src, (
        "SCv1 export must call tac.renderer_export.export_self_compressed_renderer. "
        "An invented export_scv1 wouldn't exist (dead-flag class)."
    )
    assert "from tac.renderer_export import export_self_compressed_renderer" in script_src


def test_uses_fp32_checkpoint_not_fp4_for_scv1(script_src):
    """SC weights are CORRUPTED by the FP4 save path. The SCv1 export must
    load the renderer_*_best_fp32.pt (which preserves SC per-channel
    weights), NOT renderer_*_best_fp4.pt."""
    assert "renderer_*_best_fp32.pt" in script_src
    # And we must NOT load the fp4 one for export.
    fp4_loads = re.findall(r"BEST_FP4=.*?renderer.*?fp4\.pt", script_src)
    assert not fp4_loads, (
        f"Lane W must not use renderer_*_best_fp4.pt for SCv1 export "
        f"(SC weights are corrupted by FP4 quantization). Found: {fp4_loads}"
    )


def test_workspace_path_is_canonical(script_src):
    """All Vast.ai bootstraps use WORKSPACE=/workspace/pact."""
    assert "WORKSPACE=/workspace/pact" in script_src


def test_env_sh_sourced(script_src):
    """env.sh provides PYTHONPATH=src:upstream:$PWD + FFMPEG_BIN."""
    assert 'source "$WORKSPACE/env.sh"' in script_src


def test_no_pipefail_grep_q_trap(script_src):
    """grep -q + pipefail = SIGPIPE trap (memory: feedback_pipefail_grep_q_trap)."""
    assert "grep -q" not in script_src or "set +o pipefail" in script_src, (
        "grep -q under pipefail can SIGPIPE-fail. Either don't use -q or "
        "wrap in `set +o pipefail` ... `set -o pipefail`."
    )


def test_script_does_not_invoke_codex_or_subagent(script_src):
    """Subagent task spec: the script must not invoke /codex or any
    skill-style command."""
    assert "/codex" not in script_src
    assert "claude code" not in script_src.lower() or "claude-code" not in script_src.lower(), (
        "Lane W bootstrap is shell-only; no subagent invocations."
    )
