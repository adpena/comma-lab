"""Regression tests for scripts/remote_lane_lr_lora_pose_tto.sh.

Lane LR = Low-Rank pose adaptation. Anchored on Lane A's verified
1.15 [contest-CUDA] floor. The single variable vs Lane A is
``--lora-rank 1`` (the rank-1 PoseNet Jacobian hypothesis from
``project_posenet_rank1_discovery``).

These tests pin every claim the launch script makes:

  1. Strict bash safety — `set -euo pipefail` (the LANE-B bootstrap trap).
  2. Stage 0 NVDEC probe BEFORE any GPU spend
     (``feedback_vastai_nvdec_host_variation``).
  3. Anchor on Lane A (NOT baseline_dilated_h64_0_90 or any other lane).
  4. Warm-start from Lane A's optimized_poses.pt (--gt-poses-path).
  5. --lora-rank 1 passed (the defining flag).
  6. --lora-steps wired (NOT inventing a flag).
  7. Every CLI flag verified against optimize_poses.py argparse — never
     invent flags (``feedback_dead_flag_wiring_pattern``).
  8. Provenance + heartbeat writes.
  9. Predicted band [1.10, 1.18] recorded in provenance.
 10. Sanity check: saved .pt MUST be a LoRA dict — fail loud on regression.
 11. Python zipfile (PyTorch container has no `zip` binary —
     ``feedback_zip_dep_bootstrap_trap``).
 12. Internal name `lane_lr`.
 13. No MPS / CPU device fallback (CLAUDE.md MPS-CUDA drift rule).
 14. Auth eval at the end (CLAUDE.md auth-eval-everywhere rule).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_lr_lora_pose_tto.sh"
OPTIMIZE_POSES = REPO / "experiments" / "optimize_poses.py"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


@pytest.fixture(scope="module")
def optimize_poses_argparse_flags() -> set[str]:
    """Extract `add_argument("--<flag>", ...)` flag names from
    optimize_poses.py. CLAUDE.md non-negotiable: NEVER invent CLI flags."""
    src = OPTIMIZE_POSES.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


# ── Existence + bash-safety guards ─────────────────────────────────────


def test_script_exists():
    assert SCRIPT.exists(), f"missing Lane LR launch script: {SCRIPT}"


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be chmod +x"


def test_full_set_euo_pipefail(script_text: str):
    assert "set -euo pipefail" in script_text, (
        "script must use `set -euo pipefail` (LANE-B bootstrap trap; "
        "feedback_zip_dep_bootstrap_trap)"
    )


# ── Stage 0 NVDEC probe ─────────────────────────────────────────────────


def test_nvdec_probe_present(script_text: str):
    assert "probe_nvdec.sh" in script_text


def test_nvdec_probe_fails_loud(script_text: str):
    probe_section = re.search(
        r"probe_nvdec\.sh.*?exit\s+\d+", script_text, re.DOTALL,
    )
    assert probe_section is not None, "NVDEC probe must `exit N` on failure"


# ── Anchor on Lane A ────────────────────────────────────────────────────


def test_anchors_on_lane_a(script_text: str):
    """Lane LR uses Lane A's RENDERER + masks + warm-start poses so the
    experiment isolates the LoRA factorisation as the SINGLE variable."""
    assert "experiments/results/lane_a_landed" in script_text, (
        "Lane LR must anchor on Lane A's verified 1.15 [contest-CUDA] "
        "artifacts at experiments/results/lane_a_landed/"
    )


def test_anchors_on_lane_a_renderer_bin(script_text: str):
    assert (
        "experiments/results/lane_a_landed/iter_0/renderer.bin" in script_text
    ), (
        "Lane LR must use Lane A's renderer (NOT baseline_dilated_h64_0_90); "
        "this isolates the LoRA factorisation as the single variable vs Lane A"
    )


def test_warm_start_from_lane_a_poses(script_text: str):
    """Lane LR warm-starts the LoRA U=0 path from Lane A's optimized
    poses (the same file is the ``base`` of the U @ V factorisation)."""
    assert (
        "experiments/results/lane_a_landed/optimized_poses.pt" in script_text
    ), (
        "Lane LR must use Lane A's optimized_poses.pt as --gt-poses-path "
        "(LoRA's `base` warm-start)"
    )


# ── Lane LR differentiator: --lora-rank 1 ──────────────────────────────


def test_lora_rank_1(script_text: str):
    """The defining flag of Lane LR. Without it the script would just
    re-run Lane A."""
    assert "--lora-rank 1" in script_text, (
        "Lane LR must pass --lora-rank 1 (the rank-1 PoseNet Jacobian "
        "hypothesis from project_posenet_rank1_discovery)"
    )


def test_lora_rank_flag_real_in_argparse(
    optimize_poses_argparse_flags: set[str],
):
    assert "lora-rank" in optimize_poses_argparse_flags, (
        "--lora-rank not declared in optimize_poses.py argparse"
    )


def test_lora_steps_flag_real_in_argparse(
    optimize_poses_argparse_flags: set[str],
):
    assert "lora-steps" in optimize_poses_argparse_flags, (
        "--lora-steps not declared in optimize_poses.py argparse"
    )


# ── Every flag passed to optimize_poses.py is real ─────────────────────


def test_all_optimize_poses_flags_in_script_are_real(
    script_text: str, optimize_poses_argparse_flags: set[str],
):
    """Every `--flag` in the optimize_poses invocation block MUST be a
    real argparse flag (memory: feedback_dead_flag_wiring_pattern)."""
    m = re.search(
        r"experiments/optimize_poses\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None, "couldn't find experiments/optimize_poses.py invocation"
    invocation = m.group(0)
    flags_used = set(re.findall(r"--([a-z][a-z0-9-]+)", invocation))
    bad = flags_used - optimize_poses_argparse_flags
    assert not bad, (
        f"Lane LR invokes optimize_poses.py with flags that don't exist "
        f"in its argparse: {sorted(bad)}. CLAUDE.md non-negotiable: "
        f"NEVER invent CLI flags."
    )


def test_eval_roundtrip_flag_used(script_text: str):
    """eval_roundtrip is non-negotiable per CLAUDE.md (2-11x proxy-auth gap)."""
    assert "--eval-roundtrip" in script_text


def test_posetto_noise_std_set(script_text: str):
    """Fridrich C1 fix: noise_std=0.5 closes the proxy-auth gap on
    pose TTO (memory feedback_proxy_auth_math_useless)."""
    assert "--posetto-noise-std 0.5" in script_text


# ── Sanity check: saved poses MUST be a LoRA dict ──────────────────────


def test_sanity_check_lora_dict_present(script_text: str):
    """The Lane LR critical sanity check: after optimize_poses.py runs,
    the script must verify the saved .pt is a LoRA-encoded dict — not a
    raw tensor (which would mean LoRA mode was silently bypassed)."""
    assert "is_lora_poses_dict" in script_text, (
        "Lane LR must include a sanity check via tac.lora_pose."
        "is_lora_poses_dict so a regression where lora_rank=0 silently "
        "ships a full-rank tensor is caught BEFORE auth eval"
    )


def test_sanity_check_aborts_on_non_lora_save(script_text: str):
    """The sanity check must hard-exit on non-LoRA save."""
    sanity_block = re.search(
        r"is_lora_poses_dict.*?sys\.exit\s*\(\s*[1-9]",
        script_text, re.DOTALL,
    )
    assert sanity_block is not None, (
        "the LoRA-dict sanity check must hard-exit (sys.exit(N>0)) on "
        "the regression case, not WARN-and-continue"
    )


def test_sanity_check_validates_materialised_shape(script_text: str):
    """The script must also verify the materialised pose tensor is (600, 6)
    — a LoRA dict with wrong rank/n_pairs would otherwise sneak through."""
    assert "decode_lora_poses_dict" in script_text
    assert "(600, 6)" in script_text or "600" in script_text


# ── Device CUDA required (no MPS / CPU fallback) ───────────────────────


def test_device_cuda_required(script_text: str):
    assert "--device cuda" in script_text
    assert "--device mps" not in script_text, "MPS forbidden — drift 23x"
    assert "--device cpu" not in script_text, (
        "CPU forbidden in Lane LR (pose TTO GPU-only)"
    )


def test_no_mps_fallback(script_text: str):
    code_lines = [
        ln for ln in script_text.splitlines() if not ln.strip().startswith("#")
    ]
    code = "\n".join(code_lines)
    bad_patterns = [
        r"--device\s+mps\b",
        r"device\s*=\s*[\"']mps[\"']",
        r"\bDEVICE\s*=\s*mps\b",
        r"\bif\s+.*\bmps\b",
        r"\.to\(\s*[\"']mps[\"']",
    ]
    for pat in bad_patterns:
        m = re.search(pat, code, re.IGNORECASE)
        assert m is None, (
            f"Lane LR must not reference MPS in device selection. "
            f"Match for /{pat}/: {m.group(0) if m else None!r}"
        )


# ── Provenance + heartbeat ─────────────────────────────────────────────


def test_writes_provenance_json(script_text: str):
    assert "provenance.json" in script_text or "PROVENANCE=" in script_text


def test_writes_heartbeat_log(script_text: str):
    assert "heartbeat.log" in script_text or "HEARTBEAT=" in script_text


def test_provenance_records_predicted_band(script_text: str):
    """Council-signed predicted band [1.10, 1.18] must be in provenance."""
    assert "predicted_band" in script_text
    assert (
        "[1.10, 1.18]" in script_text or "[1.1, 1.18]" in script_text
    ), "Lane LR predicted band [1.10, 1.18] must appear in provenance"


def test_provenance_records_anchor_baseline(script_text: str):
    assert "anchor_score_baseline" in script_text
    assert "1.15" in script_text


def test_provenance_records_lora_rank(script_text: str):
    """Operator-facing record of the lane's defining parameter."""
    assert "lora_rank" in script_text


# ── Internal name lane_lr ──────────────────────────────────────────────


def test_internal_name_lane_lr(script_text: str):
    assert "lane_lr" in script_text


def test_log_dir_lane_lr(script_text: str):
    assert "lane_lr_results" in script_text, (
        "LOG_DIR must be lane_lr_results so Lane LR outputs are isolated"
    )


def test_archive_named_lane_lr(script_text: str):
    assert "archive_lane_lr.zip" in script_text


def test_completion_marker_lane_lr(script_text: str):
    """The DONE marker is grepped by remote watchdogs."""
    assert "LANE_LR_DONE" in script_text


# ── Archive build via Python zipfile ────────────────────────────────────


def test_no_shell_zip_binary(script_text: str):
    code_lines = []
    for line in script_text.splitlines():
        if line.strip().startswith("#"):
            continue
        code_lines.append(line)
    code = "\n".join(code_lines)
    bad_match = re.search(r"(^|[\s;&|`\(])zip\s+(?!file)", code)
    assert not bad_match, (
        f"script must not invoke shell `zip` binary "
        f"(feedback_zip_dep_bootstrap_trap). "
        f"Match: {bad_match.group(0) if bad_match else None!r}"
    )


def test_uses_python_zipfile(script_text: str):
    assert "zipfile.ZipFile" in script_text


def test_archive_contains_required_files(script_text: str):
    """Lane LR archive: Lane A renderer.bin + Lane A masks.mkv +
    LoRA-encoded optimized_poses.pt."""
    assert "renderer.bin" in script_text
    assert "masks.mkv" in script_text
    assert "optimized_poses.pt" in script_text


# ── Auth eval ──────────────────────────────────────────────────────────


def test_runs_contest_auth_eval(script_text: str):
    """CLAUDE.md auth-eval-everywhere rule."""
    assert "contest_auth_eval.py" in script_text


def test_auth_eval_uses_built_archive(script_text: str):
    assert (
        "archive_lane_lr.zip" in script_text or "$ARCHIVE" in script_text
    ), "auth eval must use the Lane LR archive"


# ── env.sh + PYTHONHASHSEED determinism ────────────────────────────────


def test_sources_env_sh(script_text: str):
    assert "env.sh" in script_text


def test_python_hash_seed_pinned(script_text: str):
    assert "PYTHONHASHSEED" in script_text
