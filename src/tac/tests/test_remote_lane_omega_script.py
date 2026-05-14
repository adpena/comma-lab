# SPDX-License-Identifier: MIT
"""Regression tests for scripts/remote_lane_omega_hessian_qat.sh.

Lane Ω = per-weight Hessian-aware bit-budget quantization anchored on Lane A
(1.15 [contest-CUDA]). Predicted band [0.70, 1.05] (sub-Lane-A possible).

Mirrors test_remote_lane_s_script.py's coverage:
  1. Strict bash safety (set -euo pipefail).
  2. Stage 0 NVDEC probe BEFORE GPU spend.
  3. Anchor on Lane A artifacts (NOT phantom 0.9001, NOT baseline 2.29).
  4. Every CLI flag in the profile_hessian_per_weight invocation must exist
     in that script's argparse (CLAUDE.md non-negotiable).
  5. Provenance + heartbeat writes.
  6. Predicted band recorded.
  7. Python zipfile, NOT shell `zip` binary.
  8. [contest-CUDA] tag in completion log.
  9. No MPS / CPU device fallback in the GPU compute path.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_omega_hessian_qat.sh"
PROFILER = REPO / "experiments" / "profile_hessian_per_weight.py"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


@pytest.fixture(scope="module")
def profiler_argparse_flags() -> set[str]:
    src = PROFILER.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


# ── Existence + bash-safety ──────────────────────────────────────────────


def test_script_exists():
    assert SCRIPT.exists(), f"missing Lane Ω launch script: {SCRIPT}"


def test_script_is_executable():
    import os
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be chmod +x"


def test_full_set_euo_pipefail(script_text: str):
    """The canonical safe shell defaults — `set -euo pipefail`. The cascade
    trap that ate LANE-B 6.5h + $2 in 2026-04-26 (set -uo without -e) MUST
    NOT recur."""
    assert "set -euo pipefail" in script_text


# ── Stage 0 NVDEC probe ──────────────────────────────────────────────────


def test_nvdec_probe_present(script_text: str):
    assert "probe_nvdec.sh" in script_text, (
        "Stage 0 NVDEC probe required (memory: feedback_vastai_nvdec_host_variation)"
    )


def test_nvdec_probe_aborts_on_failure(script_text: str):
    """The probe must call exit N on failure, not WARN-and-continue."""
    section = re.search(r"probe_nvdec\.sh.*?exit\s+\d+", script_text, re.DOTALL)
    assert section is not None, "NVDEC probe must abort with `exit N` on failure"


# ── Anchor verification ─────────────────────────────────────────────────


def test_anchors_on_lane_a(script_text: str):
    """Lane Ω anchors on Lane A's 1.15 [contest-CUDA] — NOT phantom 0.9001
    (memory: project_baseline_0_9001_lost_archive_test) and NOT baseline 2.29."""
    assert "experiments/results/lane_a_landed" in script_text


def test_anchors_on_lane_a_renderer_bin(script_text: str):
    assert "experiments/results/lane_a_landed/iter_0/renderer.bin" in script_text


def test_anchors_on_lane_a_poses_and_masks(script_text: str):
    assert "experiments/results/lane_a_landed/iter_0/optimized_poses.pt" in script_text
    assert "experiments/results/lane_a_landed/iter_0/masks.mkv" in script_text


# ── Profiler CLI flag wiring (CLAUDE.md non-negotiable) ──────────────────


@pytest.mark.parametrize(
    "flag",
    [
        "checkpoint", "video", "masks-mkv", "poses", "upstream",
        "output", "top-k", "all-pairs", "device", "pair-batch",
    ],
)
def test_profiler_flag_used_in_script(script_text: str, flag: str):
    """Every profiler flag the Lane Ω script needs must be passed."""
    assert f"--{flag}" in script_text, (
        f"Lane Ω launch script must pass --{flag} to profile_hessian_per_weight.py"
    )


@pytest.mark.parametrize(
    "flag",
    [
        "checkpoint", "video", "masks-mkv", "poses", "upstream",
        "output", "top-k", "all-pairs", "pair-weights", "device", "pair-batch",
    ],
)
def test_profiler_flag_real_in_argparse(profiler_argparse_flags: set[str], flag: str):
    """Each profiler flag MUST exist in profile_hessian_per_weight.py's argparse.
    CLAUDE.md non-negotiable: NEVER invent CLI flags (memory:
    feedback_dead_flag_wiring_pattern)."""
    assert flag in profiler_argparse_flags, (
        f"--{flag} not declared in profile_hessian_per_weight.py argparse"
    )


def test_dead_flag_scan_present(script_text: str):
    """The script itself must contain a runtime dead-flag scan that catches
    any future drift between the script and the profiler argparse."""
    assert "INVENTED FLAGS" in script_text, (
        "script must include a Python preflight that fails-loud on any "
        "invented profiler flag (mirrors Lane S)"
    )


def test_device_cuda_required(script_text: str):
    """Every Python invocation must pass --device cuda. CLAUDE.md MPS-CUDA
    drift = 23x; Lane Ω compute path is GPU-only."""
    # At least the profiler call uses --device cuda
    assert "--device cuda" in script_text
    assert "--device mps" not in script_text
    assert "--device cpu" not in script_text


# ── Provenance + heartbeat ────────────────────────────────────────────────


def test_writes_provenance_json(script_text: str):
    assert "provenance.json" in script_text or "PROVENANCE=" in script_text


def test_writes_heartbeat_log(script_text: str):
    assert "heartbeat.log" in script_text or "HEARTBEAT=" in script_text


def test_provenance_records_predicted_band(script_text: str):
    """Council signed off on a concrete predicted band [0.70, 1.05]; the
    provenance JSON must record it for post-hoc analysis."""
    assert "predicted_band" in script_text


def test_provenance_records_anchor_baseline(script_text: str):
    assert "anchor_score_baseline" in script_text


def test_provenance_records_target_total_bits(script_text: str):
    """Lane Ω-specific: the target water-fill budget (in bits) must be in
    provenance so the bit-budget choice is auditable."""
    assert "target_total_bits" in script_text


# ── Archive build via Python zipfile ─────────────────────────────────────


def test_no_shell_zip_binary(script_text: str):
    """PyTorch container has no `zip` shell binary (memory:
    feedback_zip_dep_bootstrap_trap)."""
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
    assert "archive_lane_omega.zip" in script_text or "$ARCHIVE" in script_text


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
            f"Lane Ω code path must not reference MPS in device selection; "
            f"match for /{pat}/: {m.group(0) if m else None!r}"
        )


# ── Lane Ω–specific: bit allocator / OMG1 export wiring ───────────────────


def test_uses_bit_allocator(script_text: str):
    """Stage 2 must call tac.bit_allocator.allocate_bits."""
    assert "from tac.bit_allocator import" in script_text or \
        "tac.bit_allocator" in script_text, (
            "Stage 2 must use the canonical Lane Ω water-fill allocator"
        )


def test_uses_export_omega_renderer(script_text: str):
    """Stage 4 (or stub Stage 3) must call export_omega_renderer to emit OMG1."""
    assert "export_omega_renderer" in script_text


def test_target_total_bits_present(script_text: str):
    """The bit budget must be a concrete integer constant in the script."""
    assert re.search(r"TARGET_BITS\s*=\s*\d", script_text), (
        "TARGET_BITS constant missing — Lane Ω must commit to a specific bit budget"
    )


def test_calls_profile_hessian_per_weight(script_text: str):
    assert "profile_hessian_per_weight.py" in script_text


# ── Stage ordering ──────────────────────────────────────────────────────


def test_stage_ordering(script_text: str):
    """Stages should appear in the script in canonical order."""
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
