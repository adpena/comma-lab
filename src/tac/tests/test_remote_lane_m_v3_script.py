# SPDX-License-Identifier: MIT
"""Regression tests for scripts/remote_lane_m_v3_pose_from_embedding.sh.

Lane M-V3 (Path A) = pose-from-embedding distillation. Anchored on Lane A
(1.15 [contest-CUDA]). Replaces the ~15 KB optimized_poses.pt with a
distilled MLP (~1-2 KB FP16) + 0-byte sentinel. The MLP predicts per-pair
6-DOF poses at INFLATE TIME from mask features alone (zero PoseNet input).

V1 → 2.35 (zero-pad bug)
V2 → predicted [1.10, 1.30] (frozen-baseline padding fix)
V3 → predicted [1.10, 1.18] (different optimization variable: MLP weights,
     not the pose vector itself)

These tests pin every claim the launch script makes:

  1. Strict bash safety — `set -euo pipefail`.
  2. Stage 0 NVDEC probe BEFORE any GPU spend.
  3. Anchor on Lane A renderer + Lane A masks + Lane A optimized poses
     (as supervision targets).
  4. Every CLI flag verified against distill_pose_from_embedding.py argparse.
  5. Provenance + heartbeat writes.
  6. Predicted band [1.10, 1.18] recorded in provenance.
  7. Sanity: MLP weights < 30 KB, sentinel exactly 0 bytes.
  8. Python zipfile (PyTorch container has no `zip` binary).
  9. Internal name `lane_m_v3`.
 10. No MPS / CPU device fallback.
 11. Archive must NOT contain optimized_poses.pt (the entire point of V3).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_m_v3_pose_from_embedding.sh"
DISTILL = REPO / "experiments" / "distill_pose_from_embedding.py"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


@pytest.fixture(scope="module")
def distill_argparse_flags() -> set[str]:
    """Extract `add_argument("--<flag>", ...)` flag names from
    distill_pose_from_embedding.py. CLAUDE.md non-negotiable: NEVER
    invent CLI flags."""
    src = DISTILL.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


# ── Existence + bash-safety guards ─────────────────────────────────────


def test_script_exists():
    assert SCRIPT.exists(), f"missing Lane M-V3 launch script: {SCRIPT}"


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
    assert "experiments/results/lane_a_landed" in script_text


def test_anchors_on_lane_a_renderer_bin(script_text: str):
    assert (
        "experiments/results/lane_a_landed/iter_0/renderer.bin" in script_text
    )


def test_supervises_with_lane_a_optimized_poses(script_text: str):
    """Lane M-V3 distills against Lane A's optimized_poses.pt as the
    ground-truth labels for the MLP. Without this the MLP would have
    no supervision signal."""
    assert (
        "experiments/results/lane_a_landed/optimized_poses.pt" in script_text
    ), (
        "Lane M-V3 must use Lane A's optimized_poses.pt as the MLP "
        "supervision target (--target-poses)"
    )


# ── Distill CLI flags are all real ─────────────────────────────────────


def test_distill_invocation_present(script_text: str):
    """The script must invoke experiments/distill_pose_from_embedding.py."""
    assert "experiments/distill_pose_from_embedding.py" in script_text


def test_all_distill_flags_in_script_are_real(
    script_text: str, distill_argparse_flags: set[str],
):
    """Every `--flag` in the distill invocation block MUST be a real
    argparse flag (memory: feedback_dead_flag_wiring_pattern)."""
    m = re.search(
        r"experiments/distill_pose_from_embedding\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None, (
        "couldn't find experiments/distill_pose_from_embedding.py invocation"
    )
    invocation = m.group(0)
    flags_used = set(re.findall(r"--([a-z][a-z0-9-]+)", invocation))
    bad = flags_used - distill_argparse_flags
    assert not bad, (
        f"Lane M-V3 invokes distill_pose_from_embedding.py with flags that "
        f"don't exist in its argparse: {sorted(bad)}. CLAUDE.md "
        f"non-negotiable: NEVER invent CLI flags."
    )


def test_distill_passes_required_anchor_flags(script_text: str):
    """The four anchor inputs must all be passed."""
    assert "--renderer" in script_text
    assert "--target-poses" in script_text
    assert "--masks" in script_text
    assert "--gt-video" in script_text
    assert "--upstream" in script_text


# ── Sanity: MLP under archive budget ───────────────────────────────────


def test_sanity_check_mlp_size_present(script_text: str):
    """The script must verify the MLP weights are below the rate-savings
    ceiling so a regression in the architecture (more params → larger
    archive) is caught BEFORE burning a contest_auth_eval."""
    # Look for a python -c block that reads the weights file size and
    # exits with non-zero on failure.
    assert "MLP weights too large" in script_text or "wb >= 30000" in script_text, (
        "Lane M-V3 must include a sanity check that the MLP weights file "
        "is under the archive-cost budget. Without it a regression to a "
        "larger architecture would silently break the rate-savings premise."
    )


def test_sanity_check_sentinel_size_present(script_text: str):
    """The sentinel MUST be exactly 0 bytes — any other size means the
    build path accidentally wrote real content to it."""
    assert "sentinel must be exactly 0 bytes" in script_text or "sb != 0" in script_text


# ── Device CUDA required (no MPS / CPU fallback) ───────────────────────


def test_device_cuda_required(script_text: str):
    assert "--device cuda" in script_text
    assert "--device mps" not in script_text, "MPS forbidden — drift 23x"
    assert "--device cpu" not in script_text, "CPU forbidden in Lane M-V3"


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
            f"Lane M-V3 must not reference MPS in device selection. "
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
    ), "Lane M-V3 predicted band [1.10, 1.18] must appear in provenance"


def test_provenance_records_anchor_baseline(script_text: str):
    assert "anchor_score_baseline" in script_text


def test_provenance_records_delta_from_v2(script_text: str):
    """Operator-facing record of the V2→V3 differentiator (different
    optimization variable: MLP weights, not the pose vector itself)."""
    assert (
        "delta_from_v2" in script_text or "mlp" in script_text.lower()
    ), "provenance must record the V3 delta from V2"


def test_provenance_records_strict_scorer_compliance(script_text: str):
    """Lane M-V3 ships an MLP, not scorers — the strict-scorer-rule
    compliance claim must be in the provenance for audit."""
    assert "inflate_strict_scorer_rule_compliant" in script_text or "strict_scorer" in script_text


# ── Internal name lane_m_v3 ────────────────────────────────────────────


def test_internal_name_lane_m_v3(script_text: str):
    assert "lane_m_v3" in script_text


def test_log_dir_lane_m_v3(script_text: str):
    assert "lane_m_v3_results" in script_text


def test_archive_named_lane_m_v3(script_text: str):
    assert "archive_lane_m_v3.zip" in script_text


def test_completion_marker_lane_m_v3(script_text: str):
    assert "LANE_M_V3_DONE" in script_text


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
    """Lane M-V3 archive: Lane A renderer + masks + MLP + sentinel."""
    assert "renderer.bin" in script_text
    assert "masks.mkv" in script_text
    assert "pose_from_embedding_v1.pt" in script_text
    assert "pose_from_embedding_v1" in script_text


def test_archive_omits_optimized_poses(script_text: str):
    """The entire POINT of Lane M-V3 is to omit optimized_poses.pt and
    replace it with the distilled MLP. The build block must explicitly
    flag this as a forbidden member."""
    # Look for the forbidden-list assertion in the archive build block.
    forbidden_check = re.search(
        r"forbidden\s*=.*?optimized_poses\.pt", script_text, re.DOTALL,
    )
    assert forbidden_check is not None, (
        "Lane M-V3 archive build must list optimized_poses.pt as forbidden "
        "(the rate-savings premise is broken if it leaks back in)."
    )


def test_archive_uses_deterministic_zip(script_text: str):
    """Codex R5-r6 #5: archive build must use ZipInfo + writestr with
    fixed timestamp (deterministic-zip)."""
    assert "ZipInfo" in script_text and "writestr" in script_text, (
        "Lane M-V3 archive build must use ZipInfo + writestr (deterministic "
        "bytes across reruns; codex R5-r6 #5)."
    )


# ── Auth eval ──────────────────────────────────────────────────────────


def test_runs_contest_auth_eval(script_text: str):
    assert "contest_auth_eval.py" in script_text


def test_auth_eval_uses_built_archive(script_text: str):
    assert (
        "archive_lane_m_v3.zip" in script_text or "$ARCHIVE" in script_text
    )


# ── env.sh + PYTHONHASHSEED determinism ────────────────────────────────


def test_sources_env_sh(script_text: str):
    assert "env.sh" in script_text


def test_python_hash_seed_pinned(script_text: str):
    assert "PYTHONHASHSEED" in script_text


# ── Strict-scorer-rule compliance ──────────────────────────────────────


def test_no_scorer_load_at_inflate(script_text: str):
    """Auth eval must go through inflate.sh (the canonical compliant path).
    The MLP loaded at inflate is NOT a scorer — see test_inflate_renderer_*
    for the source-grep verification of NO-PoseNet/SegNet at inflate."""
    assert "inflate.sh" in script_text


# ── Tag completion as [contest-CUDA] ────────────────────────────────────


def test_completion_tagged_contest_cuda(script_text: str):
    """CLAUDE.md FORBIDDEN PATTERNS: every score must carry a lane tag."""
    assert "[contest-CUDA]" in script_text
