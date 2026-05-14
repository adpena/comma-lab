# SPDX-License-Identifier: MIT
"""Regression tests for scripts/remote_lane_d_v3_full_engineering.sh.

Lane D-V3 = dilated-h64 half-frame retrain stacking V2's higher LR floor
+ Lane V-V2's annealed mask_half_sim_prob (0.0 → 0.5) + post-bug-fix
KL distill weight (0.002 not 1.0) + Phase-0 mechanism instrumentation.

These tests pin every claim the launch script makes:

  1. Strict bash safety — `set -euo pipefail`.
  2. Stage 0 NVDEC probe BEFORE any GPU spend.
  3. Profile reference — uses --profile dilated_h64_half_frame_v3_annealed_kldistill.
  4. Every CLI flag passed to train_renderer.py / optimize_poses.py /
     build_baseline_archive.py / contest_auth_eval.py MUST exist in
     argparse (CLAUDE.md NEVER-INVENT-CLI-FLAGS).
  5. Provenance + heartbeat (canonical bootstrap pattern).
  6. Predicted band metadata [1.50, 2.50] (pre-registered).
  7. Python zipfile (NOT shell `zip`).
  8. [contest-CUDA] tag in completion log.
  9. Half-frame archive build with --half-frame.
 10. Pose TTO with --posetto-noise-std 0.5 + --eval-roundtrip.
 11. Auth eval on the EXACT archive that would be submitted.
 12. RESULT_JSON validation gate (LANE-B silent-crash guard).
 13. NVDEC probe fails loud (exit, not WARN-and-continue).
 14. No MPS / CPU fallback (CLAUDE.md MPS-CUDA drift 23x rule).
 15. V3-specific: documents annealing + KL fix + V2 LR floor stack.
 16. V3-specific: documents the V1 plateau cause to prevent regression.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_d_v3_full_engineering.sh"
TRAIN_RENDERER = REPO / "src" / "tac" / "experiments" / "train_renderer.py"
OPTIMIZE_POSES = REPO / "experiments" / "optimize_poses.py"
BUILD_ARCHIVE = REPO / "experiments" / "build_baseline_archive.py"
CONTEST_AUTH_EVAL = REPO / "experiments" / "contest_auth_eval.py"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


@pytest.fixture(scope="module")
def train_renderer_argparse_flags() -> set[str]:
    src = TRAIN_RENDERER.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


@pytest.fixture(scope="module")
def optimize_poses_argparse_flags() -> set[str]:
    src = OPTIMIZE_POSES.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


@pytest.fixture(scope="module")
def build_archive_argparse_flags() -> set[str]:
    src = BUILD_ARCHIVE.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


@pytest.fixture(scope="module")
def contest_auth_eval_argparse_flags() -> set[str]:
    src = CONTEST_AUTH_EVAL.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


# ── Existence + executability ──────────────────────────────────────────


def test_script_exists():
    assert SCRIPT.exists(), f"missing Lane D-V3 launch script: {SCRIPT}"


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be executable"


# ── Bash safety ────────────────────────────────────────────────────────


def test_full_set_euo_pipefail(script_text: str):
    """Belt-and-braces: assert the canonical `set -euo pipefail` line."""
    assert "set -euo pipefail" in script_text, (
        "script must use the canonical `set -euo pipefail` "
        "(memory: feedback_zip_dep_bootstrap_trap)"
    )


# ── Stage 0 NVDEC probe ────────────────────────────────────────────────


def test_nvdec_probe_present(script_text: str):
    assert "probe_nvdec.sh" in script_text


def test_nvdec_probe_fails_loud(script_text: str):
    probe_section = re.search(
        r"probe_nvdec\.sh.*?exit\s+\d+", script_text, re.DOTALL,
    )
    assert probe_section is not None


# ── Profile reference ──────────────────────────────────────────────────


def test_uses_v3_profile(script_text: str):
    assert "--profile dilated_h64_half_frame_v3_annealed_kldistill" in script_text


def test_no_other_profile_used(script_text: str):
    matches = re.findall(r"--profile\s+(\w+)", script_text)
    other = [m for m in matches if m != "dilated_h64_half_frame_v3_annealed_kldistill"]
    assert not other, (
        f"script references unexpected profiles {other} alongside V3's "
        f"dilated_h64_half_frame_v3_annealed_kldistill"
    )


# ── CLI flag wiring (CLAUDE.md NEVER-INVENT-CLI-FLAGS) ─────────────────


def test_train_renderer_flags_all_real(
    script_text: str, train_renderer_argparse_flags: set[str],
):
    """Memory: feedback_dead_flag_wiring_pattern."""
    m = re.search(
        r"src/tac/experiments/train_renderer\.py\s*\\(.*?)(?=\n\s*BEST_FP32=|\Z)",
        script_text,
        re.DOTALL,
    )
    assert m is not None, "could not locate train_renderer.py invocation"
    used = set(re.findall(r"\B--([a-z][a-z0-9-]+)", m.group(1)))
    invented = used - train_renderer_argparse_flags
    assert not invented, (
        f"INVENTED FLAGS in train_renderer invocation: {sorted(invented)} "
        f"not in train_renderer.py argparse"
    )


def test_optimize_poses_flags_all_real(
    script_text: str, optimize_poses_argparse_flags: set[str],
):
    m = re.search(
        r"experiments/optimize_poses\.py\s*\\(.*?)(?=\n\s*\[|\Z)",
        script_text,
        re.DOTALL,
    )
    assert m is not None
    used = set(re.findall(r"\B--([a-z][a-z0-9-]+)", m.group(1)))
    invented = used - optimize_poses_argparse_flags
    assert not invented, (
        f"INVENTED FLAGS in optimize_poses invocation: {sorted(invented)}"
    )


def test_build_archive_flags_all_real(
    script_text: str, build_archive_argparse_flags: set[str],
):
    m = re.search(
        r"experiments/build_baseline_archive\.py\s*\\(.*?)(?=\nmkdir|\Z)",
        script_text,
        re.DOTALL,
    )
    assert m is not None
    used = set(re.findall(r"\B--([a-z][a-z0-9-]+)", m.group(1)))
    invented = used - build_archive_argparse_flags
    assert not invented, (
        f"INVENTED FLAGS in build_baseline_archive invocation: "
        f"{sorted(invented)}"
    )


def test_contest_auth_eval_flags_all_real(
    script_text: str, contest_auth_eval_argparse_flags: set[str],
):
    m = re.search(
        r"experiments/contest_auth_eval\.py\s*\\(.*?)(?=\nif !|\Z)",
        script_text,
        re.DOTALL,
    )
    assert m is not None
    used = set(re.findall(r"\B--([a-z][a-z0-9-]+)", m.group(1)))
    invented = used - contest_auth_eval_argparse_flags
    assert not invented, (
        f"INVENTED FLAGS in contest_auth_eval invocation: "
        f"{sorted(invented)}"
    )


def test_uses_no_auth_eval_on_best(script_text: str):
    """Stage 4 builds the real archive separately + runs contest_auth_eval."""
    assert "--no-auth-eval-on-best" in script_text


def test_use_qat_enabled(script_text: str):
    assert "--use-qat" in script_text


def test_device_cuda_required(script_text: str):
    assert "--device cuda" in script_text


def test_no_mps_fallback(script_text: str):
    assert "--device mps" not in script_text, (
        "MPS forbidden — PoseNet drift 23x on MPS vs CUDA"
    )


def test_no_cpu_fallback(script_text: str):
    assert "--device cpu" not in script_text


# ── Half-frame archive build (preflight requirement) ───────────────────


def test_half_frame_archive_build_present(script_text: str):
    assert "--half-frame" in script_text


def test_half_frame_archive_assertion_present(script_text: str):
    assert "halfframe-profile-assertion" in script_text
    assert "mask_half_sim_prob" in script_text


# ── Pose TTO ───────────────────────────────────────────────────────────


def test_posetto_noise_std_05(script_text: str):
    assert "--posetto-noise-std 0.5" in script_text


def test_pose_tto_uses_eval_roundtrip(script_text: str):
    m = re.search(
        r"experiments/optimize_poses\.py(.*?)(?=\n\s*\[|\Z)",
        script_text,
        re.DOTALL,
    )
    assert m is not None
    assert "--eval-roundtrip" in m.group(1)


# ── Provenance + heartbeat (canonical bootstrap pattern) ───────────────


def test_writes_provenance_json(script_text: str):
    assert "provenance.json" in script_text or "PROVENANCE=" in script_text


def test_writes_heartbeat_log(script_text: str):
    assert "heartbeat.log" in script_text or "HEARTBEAT=" in script_text


def test_provenance_records_predicted_band(script_text: str):
    """Council pre-registered the band [1.50, 2.50]."""
    assert "predicted_band" in script_text
    assert "1.50" in script_text and "2.50" in script_text


def test_provenance_records_anchor_baseline(script_text: str):
    """Anchor score baseline (Lane A 1.15) must be recorded."""
    assert "anchor_score_baseline" in script_text


def test_provenance_records_v3_premise(script_text: str):
    """The V3 stack (V2 LR + annealed warp + KL fix) must be encoded."""
    assert "lane_d_v3_premise" in script_text or "delta_from_v2" in script_text


# ── Archive build via Python zipfile ───────────────────────────────────


def test_no_shell_zip_binary(script_text: str):
    """PyTorch container has no `zip` shell binary."""
    code_lines = []
    for line in script_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        code_lines.append(line)
    code = "\n".join(code_lines)
    bad_match = re.search(r"(^|[\s;&|`\(])zip\s+(?!file)", code)
    assert not bad_match, (
        f"script should not invoke shell `zip`; use Python zipfile. "
        f"Match: {bad_match.group(0) if bad_match else None!r}"
    )


def test_uses_python_zipfile(script_text: str):
    assert "zipfile.ZipFile" in script_text


def test_archive_contains_required_files(script_text: str):
    assert "renderer.bin" in script_text
    assert "masks.mkv" in script_text
    assert "optimized_poses.pt" in script_text


def test_archive_includes_zoom_scalars_when_present(script_text: str):
    """V3 trained with use_zoom_flow=True; inflate-side warp expansion needs
    zoom_scalars.pt."""
    assert "zoom_scalars.pt" in script_text


# ── Auth eval on the actual archive ────────────────────────────────────


def test_runs_contest_auth_eval(script_text: str):
    assert "contest_auth_eval.py" in script_text


def test_auth_eval_uses_built_archive(script_text: str):
    assert "archive_lane_d_v3.zip" in script_text or "$ARCHIVE" in script_text


def test_auth_eval_result_validated(script_text: str):
    """LANE-B silent-crash guard."""
    assert "RESULT_JSON" in script_text


# ── Lane tag on the completion log [contest-CUDA] ──────────────────────


def test_completion_log_tags_contest_cuda(script_text: str):
    assert "[contest-CUDA]" in script_text


# ── Predicted band rationale + V3-specific premise documentation ───────


def test_script_documents_v1_plateau_cause(script_text: str):
    """The header must reference V1's plateau (ep ~700) so a future operator
    knows why V3 stacks the LR + annealing + KL fixes."""
    text_lower = script_text.lower()
    assert "plateau" in text_lower
    # And the canonical ep ~700 anchor.
    assert "700" in script_text


def test_script_documents_v2_lr_inheritance(script_text: str):
    """V3 inherits V2's CHOICE B LR fix; the script must document this."""
    text_lower = script_text.lower()
    assert "choice b" in text_lower or "v2" in text_lower
    assert "5e-4" in script_text  # P2 LR (raised from V1's 3e-4)


def test_script_documents_kl_distill_post_fix_math(script_text: str):
    """The header must explain why kl_distill_weight=0.002 (POST-FIX), not
    1.0 like V1/V2 used. Otherwise a future operator may bump it back to 1.0."""
    text_lower = script_text.lower()
    assert (
        "post-bug-fix" in text_lower
        or "post-fix" in text_lower
        or "post-2026-04-27" in text_lower
    ), (
        "script header must document the POST-FIX math justifying "
        "kl_distill_weight=0.002 (vs the pre-fix 1.0 that drowned scorer)"
    )


def test_script_documents_annealing_schedule(script_text: str):
    """The header must reference the annealing schedule shape."""
    text_lower = script_text.lower()
    assert "anneal" in text_lower
    # And the canonical 0.0 → 0.5 endpoint.
    assert "0.0" in script_text and "0.5" in script_text


def test_script_documents_phase_0_instrumentation(script_text: str):
    """The header / log lines must reference Phase-0 instrumentation so a
    future operator knows to look for hf_fires / hf_warp_diff metrics."""
    text_lower = script_text.lower()
    assert "phase-0" in text_lower or "instrumentation" in text_lower or "hf_fires" in text_lower


# ── Pre-flight gates (script's own internal validation) ────────────────


def test_script_runs_profile_preflight(script_text: str):
    assert "preflight_profiles" in script_text


def test_script_runs_dead_flag_scan(script_text: str):
    assert "INVENTED FLAGS" in script_text


# ── Required artifact gates ─────────────────────────────────────────────


def test_required_artifacts_checked(script_text: str):
    assert "segnet.safetensors" in script_text
    assert "posenet.safetensors" in script_text
    assert "0.mkv" in script_text
