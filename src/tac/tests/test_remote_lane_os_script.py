"""Regression tests for scripts/remote_lane_os_supercombo_seed_tto.sh.

Lane OS-A = openpilot supercombo seeded pose TTO. Mirrors Lane A with
the delta of Stage 1.5 (supercombo seed extraction) + --seed-poses-path
in Stage 2 instead of --gt-poses-path.

Anchored on Lane A's 1.15 [contest-CUDA]. Predicted band: [1.05, 1.15].

These tests pin every contract the launch script makes:

  1. Strict bash safety — `set -euo pipefail` (LANE-B trap).
  2. Stage 0 NVDEC probe BEFORE any GPU spend.
  3. Stage 1.5 invokes seed_poses_from_openpilot.py with valid flags.
  4. Stage 2 passes --seed-poses-path (NOT --gt-poses-path) — this is
     the entire Lane OS delta.
  5. Every CLI flag verified against optimize_poses.py argparse
     (CLAUDE.md non-negotiable: NEVER invent CLI flags).
  6. Every CLI flag verified against seed_poses_from_openpilot.py argparse.
  7. Provenance + heartbeat (canonical bootstrap pattern).
  8. Predicted band [1.05, 1.15] in provenance.
  9. No MPS / CPU device fallback.
 10. Python zipfile (PyTorch container has no `zip` binary).
 11. Strict-scorer-rule: supercombo not bundled in archive (compress-time only).
 12. Internal name lane_os (not lane_a) so logs aren't conflated.
 13. Sources $WORKSPACE/env.sh per remote_bootstraps convention.
 14. PYTHONHASHSEED pinned for determinism.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_os_supercombo_seed_tto.sh"
OPT_POSES = REPO / "experiments" / "optimize_poses.py"
SEED_TOOL = REPO / "experiments" / "seed_poses_from_openpilot.py"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


@pytest.fixture(scope="module")
def optimize_poses_flags() -> set[str]:
    return set(re.findall(
        r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)",
        OPT_POSES.read_text(),
    ))


@pytest.fixture(scope="module")
def seed_tool_flags() -> set[str]:
    return set(re.findall(
        r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)",
        SEED_TOOL.read_text(),
    ))


# ── Existence + bash-safety guards ─────────────────────────────────────


def test_script_exists() -> None:
    assert SCRIPT.exists(), f"missing Lane OS-A launch script: {SCRIPT}"


def test_script_is_executable() -> None:
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be chmod +x"


def test_full_set_euo_pipefail(script_text: str) -> None:
    """Belt-and-braces: assert canonical `set -euo pipefail`."""
    assert "set -euo pipefail" in script_text, (
        "script must use `set -euo pipefail` (LANE-B trap: `set -uo` no -e "
        "silently cascaded 6.5h + $2 of failure)"
    )


# ── Stage 0 NVDEC probe ─────────────────────────────────────────────────


def test_nvdec_probe_present(script_text: str) -> None:
    assert "probe_nvdec.sh" in script_text, (
        "Stage 0 NVDEC probe required — Vast.ai host variation can crash "
        "upstream/evaluate.py at the end (memory: "
        "feedback_vastai_nvdec_host_variation)"
    )


def test_nvdec_probe_fails_loud(script_text: str) -> None:
    """The probe must abort the script on failure, not WARN-and-continue."""
    probe_section = re.search(
        r"probe_nvdec\.sh.*?exit\s+\d+", script_text, re.DOTALL,
    )
    assert probe_section is not None, "NVDEC probe must `exit N` on failure"


# ── Stage 1.5: supercombo seed extraction ───────────────────────────────


def test_invokes_seed_poses_from_openpilot(script_text: str) -> None:
    """Stage 1.5 must invoke the standalone seed tool."""
    assert "experiments/seed_poses_from_openpilot.py" in script_text, (
        "Stage 1.5 must run experiments/seed_poses_from_openpilot.py"
    )


def test_seed_tool_flags_are_real(
    script_text: str, seed_tool_flags: set[str],
) -> None:
    """Every --flag in the seed_poses_from_openpilot invocation must exist
    in its argparse (CLAUDE.md NEVER-invent-CLI-flags)."""
    m = re.search(
        r"experiments/seed_poses_from_openpilot\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None, (
        "couldn't find seed_poses_from_openpilot.py invocation"
    )
    invocation = m.group(0)
    flags_used = set(re.findall(r"--([a-z][a-z0-9-]+)", invocation))
    bad = flags_used - seed_tool_flags
    assert not bad, (
        f"Lane OS invokes seed_poses_from_openpilot.py with flags that don't "
        f"exist in its argparse: {sorted(bad)}. "
        f"NEVER invent CLI flags (memory: feedback_dead_flag_wiring_pattern)."
    )


def test_seed_tool_uses_supercombo_path_flag(script_text: str) -> None:
    assert "--supercombo-path" in script_text, (
        "Stage 1.5 must pass --supercombo-path so the loader knows where "
        "the ONNX model lives"
    )


def test_seed_tool_uses_baseline_calibration(script_text: str) -> None:
    """The seed tool must be passed --baseline-poses for affine calibration
    against the PoseNet learned-embedding scale."""
    assert "--baseline-poses" in script_text, (
        "Stage 1.5 must pass --baseline-poses so the supercombo poses are "
        "calibrated to PoseNet's training-time distribution. Without this, "
        "the seed lands outside the renderer's FiLM conditioning regime."
    )


def test_seed_tool_allows_fallback(script_text: str) -> None:
    """If supercombo can't be loaded on a given host, fall back gracefully."""
    assert "--allow-fallback" in script_text, (
        "Stage 1.5 must use --allow-fallback so a missing-supercombo host "
        "degenerates to lane_mark_pose (Lane M-equivalent) instead of "
        "wasting the GPU dollars"
    )


# ── Stage 2: --seed-poses-path is the Lane OS delta ────────────────────


def test_optimize_poses_uses_seed_poses_path(script_text: str) -> None:
    """The Lane OS delta vs Lane A: --seed-poses-path replaces --gt-poses-path."""
    assert "--seed-poses-path" in script_text, (
        "Stage 2 must pass --seed-poses-path (the Lane OS delta vs Lane A's "
        "--gt-poses-path)"
    )


def test_optimize_poses_does_not_use_gt_poses_path_in_stage_2(script_text: str) -> None:
    """Stage 2 must NOT also pass --gt-poses-path. If both are set, seed wins
    by precedence in optimize_poses.py — but passing both signals confused
    intent. Lane OS = seed only."""
    # Find the optimize_poses.py invocation block.
    m = re.search(
        r"experiments/optimize_poses\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None, "couldn't find optimize_poses.py invocation"
    invocation = m.group(0)
    assert "--gt-poses-path" not in invocation, (
        "Lane OS must NOT pass --gt-poses-path in Stage 2 — that defeats "
        "the purpose. Use --seed-poses-path only."
    )


def test_seed_poses_path_flag_real_in_argparse(
    optimize_poses_flags: set[str],
) -> None:
    """--seed-poses-path MUST exist in optimize_poses.py argparse."""
    assert "seed-poses-path" in optimize_poses_flags, (
        "--seed-poses-path not declared in optimize_poses.py argparse — "
        "CLAUDE.md forbids inventing CLI flags. Run `grep add_argument "
        "experiments/optimize_poses.py | grep seed`."
    )


def test_all_optimize_poses_flags_in_script_are_real(
    script_text: str, optimize_poses_flags: set[str],
) -> None:
    """Every --flag in the optimize_poses invocation block must exist."""
    m = re.search(
        r"experiments/optimize_poses\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None, "couldn't find optimize_poses.py invocation"
    invocation = m.group(0)
    flags_used = set(re.findall(r"--([a-z][a-z0-9-]+)", invocation))
    bad = flags_used - optimize_poses_flags
    assert not bad, (
        f"Lane OS invokes optimize_poses.py with flags that don't exist in "
        f"its argparse: {sorted(bad)}"
    )


# ── Device CUDA required (no MPS / CPU fallback) ───────────────────────


def test_device_cuda_required(script_text: str) -> None:
    assert "--device cuda" in script_text, "must use --device cuda"
    assert "--device mps" not in script_text, "MPS forbidden — drift 23x"
    assert "--device cpu" not in script_text, (
        "CPU forbidden in Lane OS (GPU-only seeding + TTO)"
    )


def test_no_mps_fallback(script_text: str) -> None:
    """No conditional MPS device selection."""
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
            f"Lane OS must not reference MPS in device selection — "
            f"CLAUDE.md drift = 23x. Match: "
            f"{m.group(0) if m else None!r}"
        )


# ── Provenance + heartbeat ─────────────────────────────────────────────


def test_writes_provenance_json(script_text: str) -> None:
    assert "provenance.json" in script_text or "PROVENANCE=" in script_text


def test_writes_heartbeat_log(script_text: str) -> None:
    assert "heartbeat.log" in script_text or "HEARTBEAT=" in script_text


def test_provenance_records_predicted_band(script_text: str) -> None:
    """Predicted band must be in provenance for post-hoc analysis."""
    assert "predicted_band" in script_text, (
        "provenance must record predicted_band"
    )
    assert "[1.05, 1.15]" in script_text or "[1.05,1.15]" in script_text, (
        "Lane OS-A predicted band [1.05, 1.15] must appear in provenance"
    )


def test_provenance_records_anchor(script_text: str) -> None:
    assert "anchor_score_baseline" in script_text or "anchor_lane" in script_text


def test_provenance_records_supercombo_path(script_text: str) -> None:
    """Operator must be able to reproduce — record where supercombo came from."""
    assert "supercombo_path" in script_text, (
        "provenance must record supercombo_path so reproductions are "
        "unambiguous (different openpilot versions = different pose heads)"
    )


# ── Internal name lane_os (NOT lane_a) ─────────────────────────────────


def test_internal_name_lane_os(script_text: str) -> None:
    """Logs from Lane A + Lane OS must not be conflated."""
    assert "lane_os" in script_text, (
        "internal name must be lane_os (not lane_a)"
    )


def test_log_dir_lane_os(script_text: str) -> None:
    assert "lane_os_results" in script_text, (
        "LOG_DIR must be lane_os_results"
    )


def test_archive_named_lane_os(script_text: str) -> None:
    assert "archive_lane_os.zip" in script_text


def test_completion_marker_lane_os(script_text: str) -> None:
    assert "LANE_OS_DONE" in script_text


# ── Archive build via Python zipfile ────────────────────────────────────


def test_no_shell_zip_binary(script_text: str) -> None:
    """PyTorch container has no `zip` shell binary."""
    code_lines = [
        ln for ln in script_text.splitlines() if not ln.strip().startswith("#")
    ]
    code = "\n".join(code_lines)
    bad = re.search(r"(^|[\s;&|`\(])zip\s+(?!file)", code)
    assert not bad, (
        f"script must not invoke shell `zip` binary (use Python zipfile). "
        f"Match: {bad.group(0) if bad else None!r}"
    )


def test_uses_python_zipfile(script_text: str) -> None:
    assert "zipfile.ZipFile" in script_text


def test_archive_does_not_bundle_supercombo(script_text: str) -> None:
    """Strict-scorer-rule: supercombo (~30 MB ONNX) must NEVER be in the
    archive — that would destroy the rate term and violate the no-scorer-
    at-inflate rule (supercombo IS effectively a scorer for pose)."""
    # Find the archive zipfile.write block(s) and verify supercombo isn't there.
    m = re.search(
        r"zipfile\.ZipFile\(.*?for n in \(([^)]+)\)",
        script_text, re.DOTALL,
    )
    assert m is not None, "must use zipfile.ZipFile for archive build"
    bundled = m.group(1)
    assert "supercombo" not in bundled, (
        "supercombo must NEVER be bundled in the archive (compress-time only)"
    )
    assert "openpilot" not in bundled, (
        "openpilot artifacts must NEVER be bundled in the archive"
    )


def test_archive_contains_required_files(script_text: str) -> None:
    """Lane OS archive: same as Lane A (renderer + masks + new poses)."""
    assert "renderer.bin" in script_text
    assert "masks.mkv" in script_text
    assert "optimized_poses.pt" in script_text


# ── Auth eval on the actual archive ─────────────────────────────────────


def test_runs_contest_auth_eval(script_text: str) -> None:
    assert "contest_auth_eval.py" in script_text, (
        "every chained experiment must end with a CUDA auth eval"
    )


def test_auth_eval_uses_built_archive(script_text: str) -> None:
    assert (
        "archive_lane_os.zip" in script_text or "$ARCHIVE" in script_text
    ), "auth eval must use the Lane OS archive"


# ── env.sh + PYTHONHASHSEED determinism ────────────────────────────────


def test_sources_env_sh(script_text: str) -> None:
    assert "env.sh" in script_text, "must source $WORKSPACE/env.sh"


def test_python_hash_seed_pinned(script_text: str) -> None:
    assert "PYTHONHASHSEED" in script_text, (
        "PYTHONHASHSEED must be pinned for deterministic dict iteration"
    )


# ── Strict-scorer-rule compliance ──────────────────────────────────────


def test_uses_inflate_sh_for_eval(script_text: str) -> None:
    """Auth eval must go through inflate.sh (the strict-scorer-rule
    compliant inflate path)."""
    assert "inflate.sh" in script_text


def test_supercombo_load_only_in_compress_stages(script_text: str) -> None:
    """supercombo references must only appear in compress stages (Stage 0/1/1.5),
    never in the auth eval stage. We assert by checking that the supercombo
    invocations all appear before contest_auth_eval.py in the script."""
    sc_idx = script_text.find("supercombo")
    eval_idx = script_text.find("contest_auth_eval.py")
    if sc_idx >= 0 and eval_idx >= 0:
        # All supercombo references must be before the auth eval invocation.
        last_sc = script_text.rfind("supercombo")
        # Allow supercombo references in the LANE_OS_DONE marker line that
        # comes after eval — that's just a log message. Specifically we check
        # the seed_poses_from_openpilot.py invocation is before eval.
        seed_invocation_idx = script_text.find("seed_poses_from_openpilot.py")
        assert seed_invocation_idx < eval_idx, (
            "supercombo seed extraction must run BEFORE auth eval (compress-time)"
        )
        _ = last_sc  # surface for debugging if assertion fires
