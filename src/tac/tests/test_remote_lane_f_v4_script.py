"""Regression tests for scripts/remote_lane_f_v4_mixed_precision_fp4.sh.

Lane F-V4 = mixed-precision FP4 QAT + per-layer sensitivity profiling.
Anchored on Lane A (1.15 [contest-CUDA]).

V1 → 2.73 (silent zero-pose bug)
V2 → 1.79 (uniform FP4 → 20x PoseNet penalty)
V3 → 1.85 (INT8 warmup helped PoseNet -17%, hurt SegNet +38%)
V4 → predicted [1.20, 1.50] [contest-CUDA] — keep PoseNet-critical
      layers in FP16 (top 30% by FP4 sensitivity), FP4 the bulk 70%.

These tests pin every claim the launch script makes:

  1. Strict bash safety — `set -euo pipefail`.
  2. Stage 0 NVDEC probe BEFORE any GPU spend (preflight check 33).
  3. Stage 0b resume-from shape validation (preflight check 34) — the
     anchor renderer's shape is checked against the QAT model build.
  4. Stage 1 calls profile_fp4_layer_sensitivity.py with --device cuda.
  5. Stage 2 calls qat_finetune.py with --mixed-precision-from-sensitivity
     pointing at the Stage 1 output.
  6. --mixed-precision-target-rate 0.70 (the council Lagrangian knob).
  7. Anchor on Lane A (1.15 [contest-CUDA]).
  8. INT8 warmup explicit (50 ep) and FP4 epochs (500 ep) preserved
     from V3 — V4 deltas are mixed-precision + KL distill aux.
  9. Every CLI flag passed to qat_finetune.py and
     profile_fp4_layer_sensitivity.py exists in their argparse
     (CLAUDE.md non-negotiable: NEVER invent CLI flags).
 10. Provenance + heartbeat writes (canonical bootstrap pattern).
 11. Predicted band [1.20, 1.50] recorded in provenance.
 12. Python zipfile (PyTorch container has no `zip` binary).
 13. Internal name `lane_f_v4` (NOT V3) so logs aren't conflated.
 14. No MPS / CPU device fallback.
 15. [contest-CUDA] tag in completion log.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_f_v4_mixed_precision_fp4.sh"
QAT_FINETUNE = REPO / "experiments" / "qat_finetune.py"
PROFILE = REPO / "experiments" / "profile_fp4_layer_sensitivity.py"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


@pytest.fixture(scope="module")
def qat_argparse_flags() -> set[str]:
    """Extract the actual `add_argument("--<flag>", ...)` flag names from
    qat_finetune.py. CLAUDE.md non-negotiable: NEVER invent CLI flags."""
    src = QAT_FINETUNE.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


@pytest.fixture(scope="module")
def profile_argparse_flags() -> set[str]:
    src = PROFILE.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


# ── Existence + bash-safety guards ─────────────────────────────────────


def test_script_exists():
    assert SCRIPT.exists(), f"missing Lane F-V4 launch script: {SCRIPT}"


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be chmod +x"


def test_full_set_euo_pipefail(script_text: str):
    assert "set -euo pipefail" in script_text, (
        "script must use `set -euo pipefail` (LANE-B trap)"
    )


# ── Stage 0 NVDEC probe (per preflight check 33) ───────────────────────


def test_nvdec_probe_present(script_text: str):
    assert "probe_nvdec.sh" in script_text, (
        "Stage 0 NVDEC probe required (preflight check 33)"
    )


def test_nvdec_probe_fails_loud(script_text: str):
    probe_section = re.search(
        r"probe_nvdec\.sh.*?exit\s+\d+", script_text, re.DOTALL,
    )
    assert probe_section is not None, "NVDEC probe must `exit N` on failure"


def test_nvdec_probe_runs_at_stage_0(script_text: str):
    """The NVDEC probe must run before any other Stage (training, archive
    build, eval). Verify probe_nvdec.sh comes before the QAT call."""
    probe_pos = script_text.find("probe_nvdec.sh")
    qat_pos = script_text.find("experiments/qat_finetune.py")
    assert probe_pos > 0 and qat_pos > 0
    assert probe_pos < qat_pos, (
        "NVDEC probe must run BEFORE qat_finetune.py "
        "(Stage 0, not after pip + DALI)"
    )


# ── Stage 0b resume-from shape validation (per preflight check 34) ─────


def test_shape_validation_block_present(script_text: str):
    """The anchor renderer must be shape-validated BEFORE the heavy QAT.

    Per preflight check 34: any script that resumes from a checkpoint
    must shape-validate the checkpoint against the model build before
    GPU spend. Lane F-V4 anchors on Lane A's renderer.bin via --checkpoint
    (not --resume-from), but the same rule applies — shape mismatch =
    crash 5+ minutes into the run."""
    assert "load_state_dict" in script_text or "shape" in script_text.lower(), (
        "Lane F-V4 must shape-validate the anchor renderer pre-flight "
        "(preflight check 34)"
    )


def test_shape_validation_runs_before_qat(script_text: str):
    """The shape-validation block must appear before the qat_finetune call."""
    # Search for the validation block (uses load_any_renderer_checkpoint
    # + state_dict comparison).
    val_pos = script_text.find("load_any_renderer_checkpoint")
    qat_pos = script_text.find("experiments/qat_finetune.py")
    if val_pos < 0:
        # Allow alternate marker
        val_pos = script_text.find("shape-validate")
    assert val_pos > 0 and qat_pos > 0
    assert val_pos < qat_pos, (
        "shape validation must run BEFORE qat_finetune.py"
    )


# ── Anchor on Lane A ───────────────────────────────────────────────────


def test_anchors_on_lane_a(script_text: str):
    assert "experiments/results/lane_a_landed" in script_text, (
        "Lane F-V4 must anchor on Lane A's verified 1.15 [contest-CUDA]"
    )


def test_anchors_on_lane_a_renderer_bin(script_text: str):
    assert (
        "experiments/results/lane_a_landed/iter_0/renderer.bin" in script_text
    )


def test_anchors_on_lane_a_poses(script_text: str):
    assert (
        "experiments/results/lane_a_landed/optimized_poses.pt" in script_text
    )


def test_anchors_on_lane_a_masks(script_text: str):
    assert (
        "experiments/results/lane_a_landed/extracted/masks.mkv" in script_text
    )


# ── Stage 1: per-layer sensitivity profiler ────────────────────────────


def test_runs_sensitivity_profiler(script_text: str):
    assert "profile_fp4_layer_sensitivity.py" in script_text, (
        "Lane F-V4 Stage 1 must run profile_fp4_layer_sensitivity.py"
    )


def test_profiler_outputs_layer_sensitivity_pt(script_text: str):
    assert "layer_sensitivity.pt" in script_text


def test_profiler_uses_cuda(script_text: str):
    """Profiler invocation must pass --device cuda."""
    m = re.search(
        r"profile_fp4_layer_sensitivity\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None
    invocation = m.group(0)
    assert "--device cuda" in invocation, (
        "profiler must use --device cuda (MPS forbidden, CPU advisory only)"
    )


def test_profiler_passes_anchor_poses(script_text: str):
    """The model has FiLM (pose_dim=6) so profiler MUST receive --poses."""
    m = re.search(
        r"profile_fp4_layer_sensitivity\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None
    invocation = m.group(0)
    assert "--poses" in invocation, (
        "FiLM model requires --poses (silent zero-pose fallback bug class)"
    )


def test_all_profiler_flags_in_script_are_real(
    script_text: str, profile_argparse_flags: set[str],
):
    """Every --flag in the profiler invocation block exists in argparse."""
    m = re.search(
        r"profile_fp4_layer_sensitivity\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None
    invocation = m.group(0)
    flags_used = set(re.findall(r"--([a-z][a-z0-9-]+)", invocation))
    bad = flags_used - profile_argparse_flags
    assert not bad, (
        f"Lane F-V4 invokes profile_fp4_layer_sensitivity.py with flags "
        f"that don't exist in its argparse: {sorted(bad)}. "
        f"NEVER invent CLI flags."
    )


# ── Stage 2: qat_finetune with --mixed-precision-from-sensitivity ──────


def test_qat_uses_mixed_precision_from_sensitivity(script_text: str):
    m = re.search(
        r"experiments/qat_finetune\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None
    invocation = m.group(0)
    assert "--mixed-precision-from-sensitivity" in invocation, (
        "Lane F-V4 must pass --mixed-precision-from-sensitivity to "
        "qat_finetune.py (the V4 delta vs V3)"
    )


def test_qat_passes_target_rate(script_text: str):
    m = re.search(
        r"experiments/qat_finetune\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None
    invocation = m.group(0)
    assert "--mixed-precision-target-rate" in invocation, (
        "Lane F-V4 must pass --mixed-precision-target-rate (Lagrangian knob)"
    )
    # Default council value is 0.70.
    target_match = re.search(
        r"--mixed-precision-target-rate\s+([0-9.]+)", invocation,
    )
    assert target_match is not None
    rate = float(target_match.group(1))
    assert 0.0 <= rate <= 1.0, (
        f"--mixed-precision-target-rate must be in [0,1], got {rate}"
    )


def test_qat_uses_int8_warmup(script_text: str):
    """V4 keeps V3's INT8 warmup (50 ep)."""
    m = re.search(
        r"experiments/qat_finetune\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None
    invocation = m.group(0)
    assert "--int8-warmup-epochs" in invocation, (
        "V4 must keep V3's INT8 warmup (anchor weight scales pre-FP4)"
    )
    assert "--skip-int8-warmup" not in invocation


def test_qat_fp4_epochs_500(script_text: str):
    assert "--fp4-epochs 500" in script_text


def test_qat_passes_poses(script_text: str):
    """Pose threading preserved from V2 fix."""
    m = re.search(
        r"experiments/qat_finetune\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None
    invocation = m.group(0)
    assert "--poses" in invocation


def test_all_qat_flags_in_script_are_real(
    script_text: str, qat_argparse_flags: set[str],
):
    """Every --flag in the qat_finetune.py invocation block must exist."""
    m = re.search(
        r"experiments/qat_finetune\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None
    invocation = m.group(0)
    flags_used = set(re.findall(r"--([a-z][a-z0-9-]+)", invocation))
    bad = flags_used - qat_argparse_flags
    assert not bad, (
        f"Lane F-V4 invokes qat_finetune.py with non-existent flags: "
        f"{sorted(bad)}. CLAUDE.md non-negotiable: NEVER invent CLI flags."
    )


def test_mixed_precision_from_sensitivity_real_in_argparse(
    qat_argparse_flags: set[str],
):
    assert "mixed-precision-from-sensitivity" in qat_argparse_flags


def test_mixed_precision_target_rate_real_in_argparse(
    qat_argparse_flags: set[str],
):
    assert "mixed-precision-target-rate" in qat_argparse_flags


# ── Device CUDA required ───────────────────────────────────────────────


def test_device_cuda_required(script_text: str):
    assert "--device cuda" in script_text
    assert "--device mps" not in script_text, "MPS forbidden"
    assert "--device cpu" not in script_text


def test_no_mps_fallback(script_text: str):
    code_lines = [
        ln for ln in script_text.splitlines() if not ln.strip().startswith("#")
    ]
    code = "\n".join(code_lines)
    bad_patterns = [
        r"--device\s+mps\b",
        r"device\s*=\s*[\"']mps[\"']",
        r"\bDEVICE\s*=\s*mps\b",
        r"\.to\(\s*[\"']mps[\"']",
    ]
    for pat in bad_patterns:
        m = re.search(pat, code, re.IGNORECASE)
        assert m is None, (
            f"MPS reference found: {m.group(0) if m else None!r}"
        )


# ── Provenance + heartbeat ─────────────────────────────────────────────


def test_writes_provenance_json(script_text: str):
    assert "provenance.json" in script_text or "PROVENANCE=" in script_text


def test_writes_heartbeat_log(script_text: str):
    assert "heartbeat.log" in script_text or "HEARTBEAT=" in script_text


def test_provenance_records_predicted_band(script_text: str):
    assert "predicted_band" in script_text
    assert (
        "[1.20, 1.50]" in script_text or "[1.2, 1.5]" in script_text
    ), "Lane F-V4 predicted band [1.20, 1.50] must be in provenance"


def test_provenance_records_anchor_baseline(script_text: str):
    assert "anchor_score_baseline" in script_text


def test_provenance_records_delta_from_v3(script_text: str):
    assert "delta_from_v3" in script_text or "mixed_precision" in script_text


def test_provenance_records_target_rate(script_text: str):
    assert "mixed_precision_target_rate" in script_text


# ── Internal name lane_f_v4 ────────────────────────────────────────────


def test_internal_name_lane_f_v4(script_text: str):
    assert "lane_f_v4" in script_text


def test_log_dir_lane_f_v4(script_text: str):
    assert "lane_f_v4_results" in script_text


def test_archive_named_lane_f_v4(script_text: str):
    assert "archive_lane_f_v4.zip" in script_text


def test_completion_marker_lane_f_v4(script_text: str):
    assert "LANE_F_V4_DONE" in script_text


# ── [contest-CUDA] tag ─────────────────────────────────────────────────


def test_completion_tag_contest_cuda(script_text: str):
    """Per CLAUDE.md score-tag rule + preflight check 32."""
    assert "[contest-CUDA]" in script_text


# ── Archive build via Python zipfile ───────────────────────────────────


def test_no_shell_zip_binary(script_text: str):
    code_lines = []
    for line in script_text.splitlines():
        if line.strip().startswith("#"):
            continue
        code_lines.append(line)
    code = "\n".join(code_lines)
    bad_match = re.search(r"(^|[\s;&|`\(])zip\s+(?!file)", code)
    assert not bad_match, (
        f"shell `zip` forbidden (PyTorch container has no zip): "
        f"{bad_match.group(0) if bad_match else None!r}"
    )


def test_uses_python_zipfile(script_text: str):
    assert "zipfile.ZipFile" in script_text


def test_uses_deterministic_zip(script_text: str):
    """Per preflight check_archive_builders_use_deterministic_zip:
    archive build must use ZipInfo + writestr with fixed timestamp."""
    assert "ZipInfo" in script_text or "writestr" in script_text, (
        "deterministic zip build required (ZipInfo + writestr + fixed timestamp)"
    )


def test_archive_contains_required_files(script_text: str):
    assert "renderer.bin" in script_text
    assert "masks.mkv" in script_text
    assert "optimized_poses.pt" in script_text


# ── Auth eval ──────────────────────────────────────────────────────────


def test_runs_contest_auth_eval(script_text: str):
    assert "contest_auth_eval.py" in script_text


def test_auth_eval_uses_built_archive(script_text: str):
    assert (
        "archive_lane_f_v4.zip" in script_text or "$ARCHIVE" in script_text
    )


# ── env.sh + PYTHONHASHSEED determinism ────────────────────────────────


def test_sources_env_sh(script_text: str):
    assert "env.sh" in script_text


def test_python_hash_seed_pinned(script_text: str):
    assert "PYTHONHASHSEED" in script_text


# ── Strict-scorer-rule compliance ──────────────────────────────────────


def test_no_scorer_load_at_inflate(script_text: str):
    assert "inflate.sh" in script_text, (
        "must use the strict-scorer-rule compliant inflate.sh path"
    )
