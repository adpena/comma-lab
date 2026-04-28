"""Regression tests for scripts/remote_lane_m_v2_radial_zoom_proper.sh.

Lane M-V2 = radial-zoom pose engineering audit retry. Anchored on Lane A
(1.15 [contest-CUDA]). The V1 saved (N, 1) and zero-padded at inflate;
the V2 fix saves (N, 6) where dim 0 is the optimized scalar and dims
1-5 are FROZEN baseline values from `--gt-poses-path`.

V1 → 2.35 (zero-pad bug — PoseNet auxiliary dims overwritten with zeros)
V2 → predicted [1.10, 1.30] [contest-CUDA] — could BEAT Lane A 1.15 if
      the rank-1 hypothesis (project_posenet_rank1_discovery: 99.8%
      variance in dim 0) holds when dims 1-5 are properly preserved.

These tests pin every claim the launch script makes:

  1. Strict bash safety — `set -euo pipefail`.
  2. Stage 0 NVDEC probe BEFORE any GPU spend.
  3. Anchor on Lane A renderer (NOT baseline_dilated_h64_0_90).
  4. Warm-start from Lane A's optimized_poses.pt (--gt-poses-path).
  5. --pose-mode radial-zoom passed (the V1+V2 differentiator).
  6. Every CLI flag verified against optimize_poses.py argparse.
  7. Provenance + heartbeat writes.
  8. Predicted band [1.10, 1.30] recorded in provenance.
  9. Sanity check: saved tensor MUST be (N, 6) — fail loud if (N, 1).
 10. Python zipfile (PyTorch container has no `zip` binary).
 11. Internal name `lane_m_v2`.
 12. No MPS / CPU device fallback.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_m_v2_radial_zoom_proper.sh"
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
    assert SCRIPT.exists(), f"missing Lane M-V2 launch script: {SCRIPT}"


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be chmod +x"


def test_full_set_euo_pipefail(script_text: str):
    assert "set -euo pipefail" in script_text, (
        "script must use `set -euo pipefail` (LANE-B trap)"
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
    """Lane M-V2 uses Lane A's RENDERER (not baseline_dilated_h64_0_90)
    so the experiment isolates the pose-padding fix."""
    assert "experiments/results/lane_a_landed" in script_text, (
        "Lane M-V2 must anchor on Lane A's verified 1.15 [contest-CUDA] "
        "artifacts at experiments/results/lane_a_landed/"
    )


def test_anchors_on_lane_a_renderer_bin(script_text: str):
    """The renderer is Lane A's, not baseline_dilated_h64_0_90's."""
    assert (
        "experiments/results/lane_a_landed/iter_0/renderer.bin" in script_text
    ), (
        "Lane M-V2 must use Lane A's renderer (NOT baseline_dilated_h64_0_90); "
        "this isolates the pose-padding fix as the single variable vs Lane A"
    )


def test_warm_start_from_lane_a_poses(script_text: str):
    """Lane M-V2 warm-starts the radial-zoom optimizable from Lane A's
    optimized_poses.pt (NOT baseline poses) so the experiment compares
    apples-to-apples with Lane A's pose floor. The same file is also
    the source of frozen baseline values for dims 1-5 (the V2 fix)."""
    assert (
        "experiments/results/lane_a_landed/optimized_poses.pt" in script_text
    ), (
        "Lane M-V2 must use Lane A's optimized_poses.pt as --gt-poses-path "
        "(both warm-start AND frozen-pad source for dims 1-5)"
    )


# ── V2 differentiator: --pose-mode radial-zoom ─────────────────────────


def test_pose_mode_radial_zoom(script_text: str):
    """The defining flag of Lane M (V1 + V2). Without it the script
    would just re-run Lane A."""
    assert "--pose-mode radial-zoom" in script_text, (
        "Lane M-V2 must pass --pose-mode radial-zoom (the 1-DOF "
        "optimizable, the rank-1 hypothesis from "
        "project_posenet_rank1_discovery)"
    )


def test_pose_mode_flag_real_in_argparse(
    optimize_poses_argparse_flags: set[str],
):
    """`--pose-mode` MUST exist in optimize_poses.py argparse.
    CLAUDE.md non-negotiable: NEVER invent CLI flags."""
    assert "pose-mode" in optimize_poses_argparse_flags, (
        "--pose-mode not declared in optimize_poses.py argparse"
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
        f"Lane M-V2 invokes optimize_poses.py with flags that don't exist "
        f"in its argparse: {sorted(bad)}. CLAUDE.md non-negotiable: "
        f"NEVER invent CLI flags."
    )


def test_eval_roundtrip_flag_used(script_text: str):
    """eval_roundtrip is non-negotiable per CLAUDE.md (2-11x proxy-auth gap)."""
    assert "--eval-roundtrip" in script_text, (
        "--eval-roundtrip required (CLAUDE.md non-negotiable: every "
        "training path MUST use eval_roundtrip)"
    )


def test_posetto_noise_std_set(script_text: str):
    """Fridrich C1 fix: noise_std=0.5 closes the proxy-auth gap on
    pose TTO (memory feedback_proxy_auth_math_useless)."""
    assert "--posetto-noise-std 0.5" in script_text, (
        "--posetto-noise-std 0.5 required (Fridrich C1 fix; without "
        "it the proxy-auth gap can be 350x)"
    )


# ── V2 sanity check: saved poses MUST be (N, 6) ────────────────────────


def test_sanity_check_n6_save_present(script_text: str):
    """The V2 critical sanity check: after optimize_poses.py runs, the
    script must verify the saved tensor is (N, 6) — not (N, 1) which
    would mean the V1 zero-pad bug was reintroduced."""
    # Look for a python -c or similar block that loads optimized_poses.pt
    # and asserts shape[1] == 6.
    assert "shape[1] != 6" in script_text or "shape[1] == 6" in script_text, (
        "Lane M-V2 must include a sanity check that the saved "
        "optimized_poses.pt is (N, 6). Without it a regression to the "
        "V1 (N, 1) bug would burn $0.20 of contest_auth_eval before "
        "crashing with a shape mismatch."
    )


def test_sanity_check_aborts_on_n1_regression(script_text: str):
    """The sanity check must hard-exit on (N, 1) — not WARN-and-continue."""
    # Look for sys.exit with a non-zero argument near the shape check.
    sanity_block = re.search(
        r"shape\[1\].*?sys\.exit\s*\(\s*[1-9]",
        script_text, re.DOTALL,
    )
    assert sanity_block is not None, (
        "the shape-1 sanity check must hard-exit (sys.exit(N>0)) on the "
        "(N, 1) regression case, not WARN-and-continue"
    )


# ── Device CUDA required (no MPS / CPU fallback) ───────────────────────


def test_device_cuda_required(script_text: str):
    assert "--device cuda" in script_text
    assert "--device mps" not in script_text, "MPS forbidden — drift 23x"
    assert "--device cpu" not in script_text, (
        "CPU forbidden in Lane M-V2 (pose TTO GPU-only)"
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
            f"Lane M-V2 must not reference MPS in device selection. "
            f"Match for /{pat}/: {m.group(0) if m else None!r}"
        )


# ── Provenance + heartbeat ─────────────────────────────────────────────


def test_writes_provenance_json(script_text: str):
    assert "provenance.json" in script_text or "PROVENANCE=" in script_text


def test_writes_heartbeat_log(script_text: str):
    assert "heartbeat.log" in script_text or "HEARTBEAT=" in script_text


def test_provenance_records_predicted_band(script_text: str):
    """Council-signed predicted band [1.10, 1.30] must be in provenance."""
    assert "predicted_band" in script_text
    assert (
        "[1.10, 1.30]" in script_text or "[1.1, 1.3]" in script_text
    ), "Lane M-V2 predicted band [1.10, 1.30] must appear in provenance"


def test_provenance_records_anchor_baseline(script_text: str):
    assert "anchor_score_baseline" in script_text


def test_provenance_records_delta_from_v1(script_text: str):
    """Operator-facing record of the V1→V2 fix (frozen baseline padding
    for dims 1-5 instead of zero-padding at inflate)."""
    assert (
        "delta_from_v1" in script_text or "frozen_baseline" in script_text
    ), "provenance must record the V2 delta from V1 (the (N, 6) padding fix)"


# ── Internal name lane_m_v2 ────────────────────────────────────────────


def test_internal_name_lane_m_v2(script_text: str):
    assert "lane_m_v2" in script_text


def test_log_dir_lane_m_v2(script_text: str):
    assert "lane_m_v2_results" in script_text, (
        "LOG_DIR must be lane_m_v2_results so V1 + V2 outputs are separate"
    )


def test_archive_named_lane_m_v2(script_text: str):
    assert "archive_lane_m_v2.zip" in script_text


def test_completion_marker_lane_m_v2(script_text: str):
    """The DONE marker is grepped by remote watchdogs."""
    assert "LANE_M_V2_DONE" in script_text


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
        f"script must not invoke shell `zip` binary. "
        f"Match: {bad_match.group(0) if bad_match else None!r}"
    )


def test_uses_python_zipfile(script_text: str):
    assert "zipfile.ZipFile" in script_text


def test_archive_contains_required_files(script_text: str):
    """Lane M-V2 archive: Lane A renderer.bin + Lane A masks.mkv +
    V2-optimized poses.pt (which is now (N, 6) with frozen dims 1-5)."""
    assert "renderer.bin" in script_text
    assert "masks.mkv" in script_text
    assert "optimized_poses.pt" in script_text


# ── Auth eval ──────────────────────────────────────────────────────────


def test_runs_contest_auth_eval(script_text: str):
    """CLAUDE.md auth-eval-everywhere rule."""
    assert "contest_auth_eval.py" in script_text


def test_auth_eval_uses_built_archive(script_text: str):
    assert (
        "archive_lane_m_v2.zip" in script_text or "$ARCHIVE" in script_text
    ), "auth eval must use the Lane M-V2 archive"


# ── env.sh + PYTHONHASHSEED determinism ────────────────────────────────


def test_sources_env_sh(script_text: str):
    assert "env.sh" in script_text


def test_python_hash_seed_pinned(script_text: str):
    assert "PYTHONHASHSEED" in script_text


# ── Strict-scorer-rule compliance ──────────────────────────────────────


def test_no_scorer_load_at_inflate(script_text: str):
    """Auth eval must go through inflate.sh (the canonical compliant path)."""
    assert "inflate.sh" in script_text
