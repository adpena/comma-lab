# SPDX-License-Identifier: MIT
"""Regression tests for scripts/remote_lane_omega_v2_lagrangian.sh.

Lane Ω-V2 = Lagrangian dual ascent on per-WEIGHT learnable bit-depth, anchored
on Lane A (1.15 [contest-CUDA]). Predicted band [0.65, 1.05] [contest-CUDA].

Mirrors test_remote_lane_omega_script.py (Ω-V1) with V2-specific deltas:
  * Calls qat_omega_lagrangian.py (NOT just water-fill + export).
  * Hessian profiler is OPTIONAL (LANE_OMEGA_V2_HESSIAN_INIT env-gated).
  * Provenance includes Lagrangian schedule constants.
  * Predicted band [0.65, 1.05] (tighter than Ω-V1's [0.70, 1.05]).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_omega_v2_lagrangian.sh"
QAT = REPO / "experiments" / "qat_omega_lagrangian.py"
PROFILER = REPO / "experiments" / "profile_hessian_per_weight.py"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


@pytest.fixture(scope="module")
def qat_argparse_flags() -> set[str]:
    src = QAT.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


@pytest.fixture(scope="module")
def profiler_argparse_flags() -> set[str]:
    src = PROFILER.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


# ── Existence + bash-safety ──────────────────────────────────────────────


def test_script_exists():
    assert SCRIPT.exists(), f"missing Lane Ω-V2 launch script: {SCRIPT}"


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be chmod +x"


def test_full_set_euo_pipefail(script_text: str):
    """`set -euo pipefail` — the LANE-B 6.5h cascade trap."""
    assert "set -euo pipefail" in script_text


# ── Stage 0 NVDEC probe ──────────────────────────────────────────────────


def test_nvdec_probe_present(script_text: str):
    assert "probe_nvdec.sh" in script_text, (
        "Stage 0 NVDEC probe required (memory: feedback_vastai_nvdec_host_variation)"
    )


def test_nvdec_probe_aborts_on_failure(script_text: str):
    section = re.search(r"probe_nvdec\.sh.*?exit\s+\d+", script_text, re.DOTALL)
    assert section is not None, "NVDEC probe must abort with `exit N` on failure"


# ── Anchor verification ────────────────────────────────────────────────


def test_anchors_on_lane_a(script_text: str):
    """Lane Ω-V2 anchors on Lane A's 1.15 [contest-CUDA]."""
    assert "experiments/results/lane_a_landed" in script_text


def test_anchors_on_lane_a_renderer_bin(script_text: str):
    assert "experiments/results/lane_a_landed/iter_0/renderer.bin" in script_text


def test_anchors_on_lane_a_poses_and_masks(script_text: str):
    assert "experiments/results/lane_a_landed/iter_0/optimized_poses.pt" in script_text
    assert "experiments/results/lane_a_landed/iter_0/masks.mkv" in script_text


# ── QAT CLI flag wiring (CLAUDE.md non-negotiable) ──────────────────────


@pytest.mark.parametrize(
    "flag",
    [
        "checkpoint", "video", "masks-mkv", "poses", "upstream",
        "output-dir", "init-bits", "target-bits", "lambda-start",
        "lambda-end", "lambda-ramp-start-frac", "total-epochs",
        "lr", "bits-lr-scale", "noise-std", "seg-weight",
        "pose-weight", "device", "seed", "log-every",
    ],
)
def test_qat_flag_used_in_script(script_text: str, flag: str):
    """Every QAT flag the launch script needs MUST appear in the script."""
    assert f"--{flag}" in script_text, (
        f"Lane Ω-V2 script must pass --{flag} to qat_omega_lagrangian.py"
    )


@pytest.mark.parametrize(
    "flag",
    [
        "checkpoint", "video", "masks-mkv", "poses", "upstream",
        "output-dir", "init-bits", "target-bits", "lambda-start",
        "lambda-end", "lambda-ramp-start-frac", "total-epochs",
        "lr", "bits-lr-scale", "noise-std", "seg-weight",
        "pose-weight", "hessian-init", "device", "seed", "log-every",
    ],
)
def test_qat_flag_real_in_argparse(qat_argparse_flags: set[str], flag: str):
    """Each QAT flag MUST exist in qat_omega_lagrangian.py's argparse.
    CLAUDE.md non-negotiable: NEVER invent CLI flags."""
    assert flag in qat_argparse_flags, (
        f"--{flag} not declared in qat_omega_lagrangian.py argparse"
    )


def test_dead_flag_scan_present(script_text: str):
    """The script must contain a runtime dead-flag preflight scan."""
    assert "INVENTED FLAGS" in script_text, (
        "script must include a Python preflight that fails-loud on any "
        "invented qat_omega_lagrangian flag"
    )


# ── Profiler is OPTIONAL ─────────────────────────────────────────────────


def test_profile_step_is_env_gated(script_text: str):
    """Lane Ω-V2 doesn't NEED the Hessian profile (Lagrangian dual ascent
    converges from any reasonable init); the warm-start is a 20-30%
    speedup. Must be env-gated so default invocations don't pay 30 min
    of profile time."""
    assert "LANE_OMEGA_V2_HESSIAN_INIT" in script_text, (
        "Hessian profile stage must be env-gated by LANE_OMEGA_V2_HESSIAN_INIT"
    )


def test_profile_flags_real_when_used(script_text: str, profiler_argparse_flags: set[str]):
    """When the optional Hessian profile is invoked, every flag passed
    must exist in profile_hessian_per_weight.py argparse."""
    if "experiments/profile_hessian_per_weight.py" not in script_text:
        pytest.skip("script doesn't invoke profiler")
    m = re.search(
        r"experiments/profile_hessian_per_weight\.py(.*?)(?=\n# Stage|\nlog \"===|\Z)",
        script_text, re.DOTALL,
    )
    assert m, "could not locate profile_hessian_per_weight.py invocation"
    used = set(re.findall(r"\B--([a-z][a-z0-9-]+)", m.group(0)))
    invented = used - profiler_argparse_flags
    assert not invented, (
        f"profile_hessian_per_weight.py invocation invents flags: {sorted(invented)}"
    )


# ── Device / scope ──────────────────────────────────────────────────────


def test_device_cuda_required(script_text: str):
    """Every Python invocation must pass --device cuda. CLAUDE.md MPS-CUDA
    drift = 23x; Lane Ω-V2 compute path is GPU-only."""
    assert "--device cuda" in script_text
    assert "--device mps" not in script_text
    assert "--device cpu" not in script_text


# ── Provenance + heartbeat ────────────────────────────────────────────────


def test_writes_provenance_json(script_text: str):
    assert "provenance.json" in script_text or "PROVENANCE=" in script_text


def test_writes_heartbeat_log(script_text: str):
    assert "heartbeat.log" in script_text or "HEARTBEAT=" in script_text


def test_provenance_records_predicted_band(script_text: str):
    """V2 predicted band [0.65, 1.05] (tighter than V1's [0.70, 1.05])."""
    assert "predicted_band" in script_text
    # Confirm the V2 band is recorded as [0.65, 1.05]
    m = re.search(r"predicted_band[^\n]*\[\s*0\.65\s*,\s*1\.05\s*\]", script_text)
    assert m, "V2 predicted_band must be [0.65, 1.05] (Yousfi council)"


def test_provenance_records_anchor_baseline(script_text: str):
    assert "anchor_score_baseline" in script_text


def test_provenance_records_lambda_schedule(script_text: str):
    """V2 specific: the Lagrangian schedule constants must appear in
    provenance so the dual-ascent dynamics are auditable."""
    for key in ("lambda_start", "lambda_end", "lambda_ramp_start_frac",
                "target_bits_per_weight"):
        assert key in script_text, (
            f"provenance must record Lagrangian schedule key {key}"
        )


def test_provenance_records_target_total_bits(script_text: str):
    """V1 compatibility: still record total bit budget for cross-lane
    comparisons."""
    assert "target_total_bits" in script_text


# ── Archive build via Python zipfile ─────────────────────────────────────


def test_no_shell_zip_binary(script_text: str):
    """PyTorch container has no `zip` shell binary."""
    code = "\n".join(
        ln for ln in script_text.splitlines() if not ln.strip().startswith("#")
    )
    bad = re.search(r"(^|[\s;&|`\(])zip\s+(?!file)", code)
    assert not bad, (
        f"script must not invoke shell `zip` binary; "
        f"match: {bad.group(0) if bad else None!r}"
    )


def test_uses_python_zipfile(script_text: str):
    assert "zipfile.ZipFile" in script_text


def test_archive_contains_required_files(script_text: str):
    assert "renderer.bin" in script_text
    assert "masks.mkv" in script_text
    assert "optimized_poses.pt" in script_text


# ── Auth eval ────────────────────────────────────────────────────────────


def test_runs_contest_auth_eval(script_text: str):
    assert "contest_auth_eval.py" in script_text


def test_auth_eval_uses_built_archive(script_text: str):
    assert "archive_lane_omega_v2.zip" in script_text or "$ARCHIVE" in script_text


def test_auth_eval_result_validated(script_text: str):
    """Detect silent auth_eval crashes via RESULT_JSON grep guard."""
    assert "RESULT_JSON" in script_text


# ── Lane tag on completion log ────────────────────────────────────────────


def test_completion_log_tags_contest_cuda(script_text: str):
    """Every score reported must carry a lane tag."""
    assert "[contest-CUDA]" in script_text


# ── Anti-MPS guard (no fallback) ─────────────────────────────────────────


def test_no_mps_fallback(script_text: str):
    """CLAUDE.md FORBIDDEN: no MPS fallback. PoseNet drift on MPS is 23x."""
    code = "\n".join(
        ln for ln in script_text.splitlines() if not ln.strip().startswith("#")
    )
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
            f"Lane Ω-V2 must not reference MPS in device selection; "
            f"match for /{pat}/: {m.group(0) if m else None!r}"
        )


# ── Lane Ω-V2 specifics ──────────────────────────────────────────────────


def test_calls_qat_omega_lagrangian(script_text: str):
    assert "qat_omega_lagrangian.py" in script_text


def test_uses_lane_v2_naming(script_text: str):
    """The script's internal name + log-tag must say 'omega_v2' (not 'omega'
    or 'omega_v1') so post-hoc analysis can disambiguate."""
    assert "lane_omega_v2" in script_text
    # Output dir prefix is lane_omega_v2_results
    assert "lane_omega_v2_results" in script_text


def test_does_not_just_use_water_fill(script_text: str):
    """Sanity: the V2 script must call the Lagrangian QAT loop, NOT just
    the V1 water-fill + export. The bit_allocator.allocate_bits call from
    V1 is allowed (warm-start path) but the QAT loop MUST be the primary
    driver."""
    # Stage 2 must be the QAT call, not just water-fill
    m = re.search(r"=== Stage 2:.*?(?==== Stage|\Z)", script_text, re.DOTALL)
    assert m, "missing Stage 2 block"
    stage2 = m.group(0)
    assert "qat_omega_lagrangian.py" in stage2, (
        "Stage 2 must invoke the Lagrangian QAT loop"
    )


def test_bits_lr_scale_present(script_text: str):
    """The QAT loop uses a smaller lr for the bits parameter group; the
    script must pass --bits-lr-scale (default 0.1) for explicit control."""
    assert "--bits-lr-scale" in script_text


# ── Stage ordering ───────────────────────────────────────────────────────


def test_stage_ordering(script_text: str):
    stages = [
        "=== Stage 0:",
        "=== Stage 1:",
        "=== Stage 2:",
        "=== Stage 3:",
        "=== Stage 4:",
        "=== Stage 5:",
    ]
    positions = [script_text.find(s) for s in stages]
    for i, p in enumerate(positions):
        assert p > 0, f"missing stage marker {stages[i]!r}"
    for i in range(len(positions) - 1):
        assert positions[i] < positions[i + 1], (
            f"stage ordering wrong: {stages[i]} appears after {stages[i + 1]}"
        )
