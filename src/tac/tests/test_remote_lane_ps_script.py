# SPDX-License-Identifier: MIT
"""Regression tests for scripts/remote_lane_ps_per_class_segnet.sh.

Lane PS = per-class SegNet weighting on the auxiliary KL distill loss.
Anchored on Lane A (1.15 [contest-CUDA]). The hypothesis is that
Lane A's averaged SegNet distortion (0.0046) hides per-class imbalance,
and biasing the auxiliary KL distill loss toward costly classes
(``--segnet-class-weights "1,5,5,1,1"``) closes the residual gap.

These tests pin every claim the launch script makes:

  1. Strict bash safety — `set -euo pipefail`.
  2. Stage 0 NVDEC probe BEFORE any GPU spend.
  3. Anchor on Lane A renderer + poses + masks (1.15 baseline).
  4. ``--segnet-class-weights "1,5,5,1,1"`` is passed.
  5. ``--kl-distill-weight 1.0`` is passed (mandatory — Lane PS
     fires only on the auxiliary KL path; without it the per-class
     tensor is parsed but never read).
  6. Every ``--flag`` in the optimize_poses invocation is REAL —
     verified against ``optimize_poses.py`` argparse (CLAUDE.md
     non-negotiable: NEVER invent CLI flags).
  7. ``[lane-ps]`` log-banner sanity check is present and aborts on
     missing banner (silent fall-back to uniform = $0.20 wasted).
  8. Provenance + heartbeat writes (CLAUDE.md canonical pipeline).
  9. Predicted band [1.05, 1.20] recorded in provenance.
 10. Python zipfile (PyTorch container has no `zip` binary).
 11. Internal name `lane_ps`.
 12. No MPS / CPU device fallback.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_ps_per_class_segnet.sh"
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
    assert SCRIPT.exists(), f"missing Lane PS launch script: {SCRIPT}"


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be chmod +x"


def test_full_set_euo_pipefail(script_text: str):
    assert "set -euo pipefail" in script_text, (
        "script must use `set -euo pipefail` (LANE-B trap: "
        "feedback_zip_dep_bootstrap_trap)"
    )


# ── Stage 0 NVDEC probe ────────────────────────────────────────────────


def test_nvdec_probe_present(script_text: str):
    assert "probe_nvdec.sh" in script_text


def test_nvdec_probe_fails_loud(script_text: str):
    probe_section = re.search(
        r"probe_nvdec\.sh.*?exit\s+\d+", script_text, re.DOTALL,
    )
    assert probe_section is not None, "NVDEC probe must `exit N` on failure"


# ── Anchor on Lane A ───────────────────────────────────────────────────


def test_anchors_on_lane_a_renderer(script_text: str):
    assert (
        "experiments/results/lane_a_landed/iter_0/renderer.bin"
        in script_text
    ), (
        "Lane PS must anchor on Lane A's renderer (NOT "
        "baseline_dilated_h64_0_90); this isolates the per-class "
        "weighting as the single variable vs Lane A"
    )


def test_anchors_on_lane_a_poses(script_text: str):
    assert (
        "experiments/results/lane_a_landed/optimized_poses.pt"
        in script_text
    ), "Lane PS must use Lane A's optimized_poses.pt as warm-start"


def test_anchors_on_lane_a_masks(script_text: str):
    assert (
        "experiments/results/lane_a_landed/extracted/masks.mkv"
        in script_text
    ), "Lane PS must use Lane A's masks (no rebuild)"


def test_records_anchor_baseline_score(script_text: str):
    assert "1.15" in script_text, (
        "provenance must record Lane A baseline 1.15 [contest-CUDA]"
    )


# ── Lane PS differentiator: --segnet-class-weights ─────────────────────


def test_segnet_class_weights_flag_passed(script_text: str):
    """The defining flag of Lane PS. Without it the script would
    just re-run a Lane A pose TTO with a wasted KL distill term."""
    assert (
        '--segnet-class-weights "1,5,5,1,1"' in script_text
        or "--segnet-class-weights '1,5,5,1,1'" in script_text
    ), (
        "Lane PS must pass --segnet-class-weights '1,5,5,1,1' (the "
        "canonical 'boost lane + boundary classes' setting from the "
        "research survey)"
    )


def test_segnet_class_weights_flag_real_in_argparse(
    optimize_poses_argparse_flags: set[str],
):
    """`--segnet-class-weights` MUST exist in optimize_poses.py argparse.
    CLAUDE.md non-negotiable: NEVER invent CLI flags."""
    assert "segnet-class-weights" in optimize_poses_argparse_flags, (
        "--segnet-class-weights not declared in optimize_poses.py argparse "
        "— the Lane PS flag would silently fail at run time. "
        "Add it before re-running this test."
    )


def test_kl_distill_weight_passed(script_text: str):
    """Lane PS only fires on the AUXILIARY KL distill path. Without
    --kl-distill-weight > 0 the per-class tensor is parsed but never
    read — the run becomes a uniform-weighting Lane A re-run."""
    assert "--kl-distill-weight 1.0" in script_text, (
        "Lane PS requires --kl-distill-weight 1.0 (Quantizr default); "
        "without it the per-class weighting is a NO-OP and the run is "
        "just a Lane A re-run that burns $0.20 with no signal"
    )


def test_kl_distill_temperature_passed(script_text: str):
    assert "--kl-distill-temperature 2.0" in script_text, (
        "Lane PS uses Quantizr/Hinton T=2.0 (kl_distill_temperature)"
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
        f"Lane PS invokes optimize_poses.py with flags that don't exist "
        f"in its argparse: {sorted(bad)}. CLAUDE.md non-negotiable: "
        f"NEVER invent CLI flags."
    )


def test_eval_roundtrip_flag_used(script_text: str):
    """eval_roundtrip is non-negotiable per CLAUDE.md (2-11x proxy-auth gap)."""
    assert "--eval-roundtrip" in script_text, (
        "--eval-roundtrip required (CLAUDE.md non-negotiable)"
    )


def test_posetto_noise_std_set(script_text: str):
    """Fridrich C1 fix: noise_std=0.5 closes the proxy-auth gap on
    pose TTO (memory feedback_proxy_auth_math_useless)."""
    assert "--posetto-noise-std 0.5" in script_text, (
        "--posetto-noise-std 0.5 required (Fridrich C1 fix; without "
        "it the proxy-auth gap can be 350x)"
    )


# ── Lane PS sanity check: [lane-ps] banner must fire ──────────────────


def test_banner_sanity_check_present(script_text: str):
    """The [lane-ps] banner is the only operator-visible signal that
    the per-class tensor was actually parsed and active. The script
    must grep for it post-run and abort on missing banner — without
    this guard a silent fall-back to uniform weighting (e.g., due
    to a parse bug returning None) would burn $0.20 of contest_auth_eval."""
    assert "lane-ps" in script_text and "grep" in script_text, (
        "Lane PS must include a post-run grep for the [lane-ps] banner "
        "to verify per-class weights were actually active"
    )


def test_banner_sanity_check_aborts_on_missing(script_text: str):
    """The banner check must hard-exit when the banner is absent — not
    WARN-and-continue (silent uniform-fallback would otherwise produce
    a uniform-weighting score misnamed as Lane PS)."""
    sanity_block = re.search(
        r"grep[^\n]*lane-ps[^\n]*\|\|.*?exit\s*[1-9]",
        script_text, re.DOTALL,
    )
    assert sanity_block is not None, (
        "the [lane-ps] banner-check must hard-exit (`exit N>0`) on a "
        "missing banner, not silently continue with uniform weighting"
    )


# ── Device CUDA required (no MPS / CPU fallback) ──────────────────────


def test_device_cuda_required(script_text: str):
    assert "--device cuda" in script_text
    assert "--device mps" not in script_text, "MPS forbidden — drift 23x"
    assert "--device cpu" not in script_text, (
        "CPU forbidden in Lane PS (pose TTO GPU-only)"
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
            f"Lane PS must not reference MPS in device selection. "
            f"Match for /{pat}/: {m.group(0) if m else None!r}"
        )


# ── Provenance + heartbeat ─────────────────────────────────────────────


def test_writes_provenance_json(script_text: str):
    assert "provenance.json" in script_text or "PROVENANCE=" in script_text


def test_writes_heartbeat_log(script_text: str):
    assert "heartbeat.log" in script_text or "HEARTBEAT=" in script_text


def test_provenance_records_predicted_band(script_text: str):
    """Council-signed predicted band [1.05, 1.20] must be in provenance."""
    assert "predicted_band" in script_text
    assert (
        "[1.05, 1.20]" in script_text or "[1.05, 1.2]" in script_text
    ), "Lane PS predicted band [1.05, 1.20] must appear in provenance"


def test_provenance_records_segnet_class_weights(script_text: str):
    """Provenance must record the SegNet class weights (operator can
    diff old runs vs new runs without grepping the launch invocation)."""
    assert "segnet_class_weights" in script_text, (
        "provenance must record the segnet_class_weights value used"
    )


def test_provenance_records_anchor_baseline(script_text: str):
    assert "anchor_score_baseline" in script_text


def test_provenance_records_delta_from_lane_a(script_text: str):
    assert (
        "delta_from_lane_a" in script_text
        or "segnet_class_weights_1_5_5_1_1" in script_text
    ), "provenance must record the Lane PS delta from Lane A"


# ── Internal name lane_ps ──────────────────────────────────────────────


def test_internal_name_lane_ps(script_text: str):
    assert "lane_ps" in script_text


def test_log_dir_lane_ps(script_text: str):
    assert "lane_ps_results" in script_text, (
        "LOG_DIR must be lane_ps_results so Lane PS outputs are isolated "
        "from other lanes"
    )


def test_archive_named_lane_ps(script_text: str):
    assert "archive_lane_ps.zip" in script_text


def test_completion_marker_lane_ps(script_text: str):
    """The DONE marker is grepped by remote watchdogs."""
    assert "LANE_PS_DONE" in script_text


# ── Archive build via Python zipfile (PyTorch container has no `zip`) ──


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
        f"(memory: feedback_zip_dep_bootstrap_trap). "
        f"Match: {bad_match.group(0) if bad_match else None!r}"
    )


def test_uses_python_zipfile(script_text: str):
    assert "zipfile.ZipFile" in script_text


def test_archive_contains_required_files(script_text: str):
    """Lane PS archive: Lane A renderer.bin + Lane A masks.mkv +
    Lane PS optimized_poses.pt."""
    for name in ("renderer.bin", "masks.mkv", "optimized_poses.pt"):
        assert name in script_text, f"archive build must include {name}"


# ── Contest auth eval Stage 4 ──────────────────────────────────────────


def test_runs_contest_auth_eval(script_text: str):
    assert "contest_auth_eval.py" in script_text, (
        "Stage 4 must run contest_auth_eval.py on the Lane PS archive "
        "(CLAUDE.md non-negotiable: every chained experiment MUST end "
        "with a CUDA auth eval)"
    )


def test_auth_eval_uses_inflate_sh(script_text: str):
    assert "inflate.sh" in script_text, (
        "auth eval must go through the canonical inflate.sh path"
    )


def test_auth_eval_writes_log(script_text: str):
    assert "auth_eval.log" in script_text, (
        "Stage 4 must tee to auth_eval.log so RESULT_JSON is captured"
    )


# ── --gt-poses-path warm-start (mandatory for pose TTO) ────────────────


def test_gt_poses_path_warm_start(script_text: str):
    """Lane PS warm-starts pose TTO from Lane A's optimized poses, NOT
    from baseline poses. Per memory project_baseline_poses_load_bearing,
    the renderer + poses are a JOINT artifact — initializing with
    baseline poses on Lane A's renderer destroys the score (33% pixel
    shift, 23x PoseNet degrade). Lane PS preserves the Lane A pose
    distribution and only re-shapes the convergence direction via the
    per-class KL term."""
    assert "--gt-poses-path" in script_text, (
        "--gt-poses-path required (warm-start from Lane A's poses; "
        "without it pose TTO degrades 23x per "
        "project_baseline_poses_load_bearing)"
    )
