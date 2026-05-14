# SPDX-License-Identifier: MIT
"""Regression tests for scripts/remote_lane_lm_zero_cost_poses.sh.

Lane LM-A = zero-archive-cost poses computed at inflate from lane-mark mask
displacement. Anchored on Lane A (1.15 [contest-CUDA]). The archive omits
``optimized_poses.pt`` (15.3 KB) and writes a 0-byte ``zero_cost_poses_v1``
sentinel; the inflate side reads ``INFLATE_ZERO_COST_POSES=1`` and computes
per-pair 6-DOF poses via
``tac.lane_mark_pose.compute_zero_cost_poses_from_masks``.

These tests pin every claim the launch script makes:

  1. Strict bash safety — ``set -euo pipefail`` (LANE-B trap).
  2. Stage 0 NVDEC probe BEFORE any GPU spend (vastai-host-variation).
  3. Anchor on Lane A's verified 1.15 archive (NOT baseline_dilated_h64_0_90).
  4. Builder invocation references build_zero_cost_pose_archive.py with
     real argparse flags only (NEVER invent CLI flags).
  5. INFLATE_ZERO_COST_POSES=1 is set on the contest_auth_eval call.
  6. Output archive contains the sentinel + omits the pose file.
  7. Provenance + heartbeat writes.
  8. Predicted band [1.05, 1.15] recorded in provenance.
  9. Internal name ``lane_lm_a``.
 10. No MPS / CPU device fallback.
 11. Python zipfile (PyTorch container has no `zip` binary).
 12. Strict-scorer-rule compliance flag in provenance.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_lm_zero_cost_poses.sh"
BUILDER = REPO / "experiments" / "build_zero_cost_pose_archive.py"
CONTEST_AUTH_EVAL = REPO / "experiments" / "contest_auth_eval.py"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


@pytest.fixture(scope="module")
def builder_argparse_flags() -> set[str]:
    """Extract `add_argument("--<flag>", ...)` flag names from
    build_zero_cost_pose_archive.py. CLAUDE.md non-negotiable: NEVER
    invent CLI flags."""
    src = BUILDER.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


@pytest.fixture(scope="module")
def contest_auth_eval_argparse_flags() -> set[str]:
    src = CONTEST_AUTH_EVAL.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


# ── Existence + bash-safety guards ─────────────────────────────────────


def test_script_exists():
    assert SCRIPT.exists(), f"missing Lane LM-A launch script: {SCRIPT}"


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be chmod +x"


def test_full_set_euo_pipefail(script_text: str):
    """LANE-B trap: `set -uo pipefail` (no -e) silently cascades failures."""
    assert "set -euo pipefail" in script_text, (
        "Lane LM-A script must use `set -euo pipefail` "
        "(memory feedback_zip_dep_bootstrap_trap)"
    )


def test_no_shell_zip_binary(script_text: str):
    """PyTorch container has no `zip` binary; use python zipfile.

    Memory: feedback_zip_dep_bootstrap_trap. The match must be the actual
    `zip` shell command (`zip ...`), not the literal substring 'zip' which
    appears in `zipfile.ZipFile` and `gzip` references throughout the
    script.
    """
    # Strip shell comments first so docstring references don't false-match.
    code = "\n".join(
        ln for ln in script_text.splitlines() if not ln.strip().startswith("#")
    )
    # Pattern: a `zip` invocation at start-of-line or after `;`/`&&`/`||`
    # followed by a flag or path. Excludes `zipfile`, `gzip`, `bzip2`,
    # `unzip`.
    bad = re.search(r"(?:^|[;&|]\s*)zip\s+(?!file)", code, re.MULTILINE)
    assert bad is None, (
        "Lane LM-A script must NOT call the `zip` shell binary (PyTorch "
        f"container does not have it). Use python zipfile. Match: "
        f"{bad.group(0) if bad else None!r}"
    )


# ── Stage 0 NVDEC probe ─────────────────────────────────────────────────


def test_nvdec_probe_present(script_text: str):
    assert "probe_nvdec.sh" in script_text, (
        "Lane LM-A must invoke scripts/probe_nvdec.sh in Stage 0 "
        "(memory feedback_vastai_nvdec_host_variation)"
    )


def test_nvdec_probe_fails_loud(script_text: str):
    probe_section = re.search(
        r"probe_nvdec\.sh.*?exit\s+\d+", script_text, re.DOTALL,
    )
    assert probe_section is not None, (
        "NVDEC probe must `exit N` on failure (no warn-and-continue)"
    )


# ── Anchor on Lane A archive ─────────────────────────────────────────────


def test_anchors_on_lane_a_archive(script_text: str):
    """Lane LM-A = Lane A's archive minus optimized_poses.pt.

    The script MUST anchor on the Lane A archive at
    experiments/results/lane_a_landed/archive_lane_a.zip — NOT on the
    baseline_dilated_h64_0_90 directory (which still scores 53.60 because
    the masks are 48x64).
    """
    assert "experiments/results/lane_a_landed/archive_lane_a.zip" in script_text, (
        "Lane LM-A must anchor on the verified Lane A archive at "
        "experiments/results/lane_a_landed/archive_lane_a.zip "
        "(score 1.15 [contest-CUDA])"
    )


# ── Builder invocation: every flag must exist in argparse ─────────────────


def test_builder_invocation_present(script_text: str):
    assert "experiments/build_zero_cost_pose_archive.py" in script_text, (
        "Lane LM-A must call experiments/build_zero_cost_pose_archive.py"
    )


def test_all_builder_flags_in_script_are_real(
    script_text: str, builder_argparse_flags: set[str],
):
    """Every `--flag` in the build_zero_cost_pose_archive.py invocation
    block MUST be a real argparse flag. Memory:
    feedback_dead_flag_wiring_pattern."""
    m = re.search(
        r"experiments/build_zero_cost_pose_archive\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None, (
        "couldn't find experiments/build_zero_cost_pose_archive.py invocation "
        "with continuation lines"
    )
    invocation = m.group(0)
    flags_used = set(re.findall(r"--([a-z][a-z0-9-]+)", invocation))
    bad = flags_used - builder_argparse_flags
    assert not bad, (
        f"Lane LM-A invokes build_zero_cost_pose_archive.py with flags "
        f"that don't exist in its argparse: {sorted(bad)}. CLAUDE.md "
        f"non-negotiable: NEVER invent CLI flags."
    )


def test_min_correlation_gate_present(script_text: str):
    """The --min-correlation gate is the cheap pre-eval sanity check
    that prevents Vast.ai $ from being burned on a calibration that is
    obviously broken (analytical estimate decoupled from optimized
    target)."""
    assert "--min-correlation" in script_text, (
        "Lane LM-A must pass --min-correlation to gate against a "
        "decoupled lane-mark calibration BEFORE the eval."
    )


# ── INFLATE_ZERO_COST_POSES env on the contest_auth_eval call ───────────


def test_inflate_zero_cost_poses_env_set(script_text: str):
    """The inflate side requires INFLATE_ZERO_COST_POSES=1 to compute
    poses from the sentinel; without it, the archive falls through to
    unconditioned rendering (catastrophic). The env gate is INTENTIONAL
    (prevents silent activation on stale archives), so the launch script
    MUST set it explicitly on the eval invocation."""
    # Match: "INFLATE_ZERO_COST_POSES=1 ... contest_auth_eval.py"
    m = re.search(
        r"INFLATE_ZERO_COST_POSES=1\s+\"?\$?PYBIN\"?[^\n]*contest_auth_eval\.py",
        script_text,
    )
    assert m is not None, (
        "Lane LM-A must set INFLATE_ZERO_COST_POSES=1 on the "
        "contest_auth_eval.py invocation (otherwise the archive falls "
        "through to unconditioned rendering)."
    )


def test_all_contest_auth_eval_flags_in_script_are_real(
    script_text: str, contest_auth_eval_argparse_flags: set[str],
):
    """Every `--flag` on the contest_auth_eval.py invocation MUST be a
    real argparse flag (memory: feedback_dead_flag_wiring_pattern)."""
    m = re.search(
        r"experiments/contest_auth_eval\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None, (
        "couldn't find experiments/contest_auth_eval.py invocation"
    )
    invocation = m.group(0)
    flags_used = set(re.findall(r"--([a-z][a-z0-9-]+)", invocation))
    bad = flags_used - contest_auth_eval_argparse_flags
    assert not bad, (
        f"Lane LM-A invokes contest_auth_eval.py with flags that don't "
        f"exist in its argparse: {sorted(bad)}."
    )


# ── Archive sanity assertions in the script body ────────────────────────


def test_sanity_check_sentinel_present(script_text: str):
    """After the build, the script must verify the output archive
    actually contains the sentinel — fail loud if missing."""
    assert "zero_cost_poses_v1" in script_text, (
        "Lane LM-A script must reference the zero_cost_poses_v1 "
        "sentinel for the archive sanity check."
    )


def test_sanity_check_no_pose_pt(script_text: str):
    """The whole point of Lane LM-A is to OMIT optimized_poses.pt.
    If the output archive accidentally still contains it, the score
    delta vs Lane A is meaningless. The sanity check must verify
    omission."""
    assert "optimized_poses.pt" in script_text, (
        "Lane LM-A script must explicitly check for absence of "
        "optimized_poses.pt in the output archive (fail-loud sanity)."
    )


# ── Device CUDA required (no MPS / CPU fallback) ───────────────────────


def test_device_cuda_required(script_text: str):
    assert "--device cuda" in script_text, (
        "contest_auth_eval must run on CUDA (CLAUDE.md non-negotiable: "
        "MPS auth eval is NOISE, drift 23x)"
    )
    assert "--device mps" not in script_text, "MPS forbidden — drift 23x"
    assert "--device cpu" not in script_text, (
        "CPU forbidden in Lane LM-A (contest_auth_eval GPU-only)"
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
            f"Lane LM-A must not reference MPS in device selection. "
            f"Match for /{pat}/: {m.group(0) if m else None!r}"
        )


# ── Provenance + heartbeat ─────────────────────────────────────────────


def test_writes_provenance_json(script_text: str):
    assert "provenance.json" in script_text or "PROVENANCE=" in script_text


def test_writes_heartbeat_log(script_text: str):
    assert "heartbeat.log" in script_text or "HEARTBEAT=" in script_text


def test_provenance_records_predicted_band(script_text: str):
    """Council-signed predicted band [1.05, 1.15] must be in provenance."""
    assert "predicted_band" in script_text
    assert (
        "[1.05, 1.15]" in script_text
        or "[1.05,1.15]" in script_text
    ), "Lane LM-A predicted band [1.05, 1.15] must appear in provenance"


def test_provenance_records_anchor_baseline(script_text: str):
    assert "anchor_score_baseline" in script_text


def test_provenance_records_strict_scorer_rule(script_text: str):
    """Strict-scorer-rule compliance is the crux of why Lane LM-A is
    legal: NO scorers loaded at inflate. The provenance must record this
    explicit claim so an auditor can verify it."""
    assert "strict_scorer_rule_compliant" in script_text, (
        "provenance must record strict_scorer_rule_compliant=true"
    )


def test_provenance_records_inflate_env_required(script_text: str):
    """The provenance must document INFLATE_ZERO_COST_POSES=1 as a
    required env var for the inflate side — operators looking at the
    provenance.json should not need to read the script to find this."""
    assert "INFLATE_ZERO_COST_POSES" in script_text, (
        "provenance must record the INFLATE_ZERO_COST_POSES=1 env "
        "requirement"
    )


# ── Internal name lane_lm_a ────────────────────────────────────────────


def test_internal_name_lane_lm_a(script_text: str):
    assert "lane_lm_a" in script_text, (
        "script must reference lane_lm_a internally for log-filtering"
    )


# ── Workspace + python interpreter ─────────────────────────────────────


def test_uses_container_python(script_text: str):
    """Per memory feedback_canonical_remote_bootstraps: use
    /opt/conda/bin/python (NOT a venv) on the PyTorch container."""
    assert "/opt/conda/bin/python" in script_text, (
        "Lane LM-A must use /opt/conda/bin/python (NOT a venv) per "
        "memory feedback_canonical_remote_bootstraps"
    )


def test_sources_env_sh(script_text: str):
    assert "env.sh" in script_text, (
        "Lane LM-A must source env.sh (canonical remote bootstrap pattern)"
    )
