# SPDX-License-Identifier: MIT
"""Regression tests for scripts/remote_lane_g_v3_corrected_kl_weight.sh.

Lane G V3 corrected = pose TTO + KL distill SegNet auxiliary, with the
KL weight RECALIBRATED for the post-bug-fix raw-KL value.

V1 (--kl-distill-weight 1.0):  KL dominated ~14000× over scorer. Killed.
V2 (--kl-distill-weight 0.01): KL dominated ~4000×. Killed.
V3-prev (--kl-distill-weight 5e-6): sized for BUGGY raw KL ~20000;
    after commit f17de3bb fixed `F.kl_div(reduction=...)` math the raw
    KL is ~2.7. 5e-6 × 2.7 = 1.4e-5 — TOO SMALL, no aux signal.
V3 corrected (--kl-distill-weight 0.002): council math (post-fix raw
    KL ~2.7, scorer loss ~0.05) → 0.002 × 2.7 = 5.4e-3 = ~10% of
    scorer loss = canonical Hinton 2015 auxiliary regime.

Predicted band: [1.10, 1.18] (anchor Lane A 1.15).

These tests pin every claim the launch script makes:

  1. Strict bash safety — `set -euo pipefail` (LANE-B trap).
  2. Stage 0 NVDEC probe BEFORE any GPU spend (--ensure-dali).
  3. Anchors on Lane A (experiments/results/lane_a_landed/).
  4. --kl-distill-weight = 0.002 (NOT 5e-6, NOT 0.01, NOT 1.0).
  5. --kl-distill-temperature = 2.0 (Hinton 2015 default, Quantizr-validated).
  6. Every CLI flag verified against optimize_poses.py argparse
     (CLAUDE.md non-negotiable: NEVER invent CLI flags).
  7. Provenance + heartbeat writes (canonical bootstrap pattern).
  8. Predicted band [1.10, 1.18] recorded.
  9. Internal name lane_g_v3 (NOT lane_g, NOT lane_g_v2).
 10. Python zipfile (PyTorch container has no `zip` binary).
 11. No MPS / CPU device fallback.
 12. Archive build + contest_auth_eval at the end.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_g_v3_corrected_kl_weight.sh"
OPTIMIZE_POSES = REPO / "experiments" / "optimize_poses.py"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


@pytest.fixture(scope="module")
def optimize_poses_argparse_flags() -> set[str]:
    """Extract real `add_argument("--<flag>", ...)` flag names from
    optimize_poses.py. CLAUDE.md non-negotiable: NEVER invent CLI flags."""
    src = OPTIMIZE_POSES.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


# ── Existence + bash-safety guards ─────────────────────────────────────


def test_script_exists():
    assert SCRIPT.exists(), f"missing Lane G V3 launch script: {SCRIPT}"


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be chmod +x"


def test_full_set_euo_pipefail(script_text: str):
    assert "set -euo pipefail" in script_text, (
        "script must use `set -euo pipefail` (LANE-B trap)"
    )


# ── Stage 0 NVDEC probe ─────────────────────────────────────────────────


def test_nvdec_probe_present(script_text: str):
    assert "probe_nvdec.sh" in script_text, (
        "Stage 0 NVDEC probe required (memory: "
        "feedback_vastai_nvdec_host_variation)"
    )


def test_nvdec_probe_fails_loud(script_text: str):
    probe_section = re.search(
        r"probe_nvdec\.sh.*?exit\s+\d+", script_text, re.DOTALL,
    )
    assert probe_section is not None, "NVDEC probe must `exit N` on failure"


def test_nvdec_probe_ensure_dali(script_text: str):
    """--ensure-dali so a fresh container that hasn't run remote_setup_full.sh
    installs DALI rather than spuriously failing on import."""
    assert "--ensure-dali" in script_text, (
        "Lane G V3 NVDEC probe must use --ensure-dali (the eval at the end "
        "needs DALI)"
    )


# ── Anchor on Lane A (NOT phantom 0.9001, NOT baseline 2.29) ───────────


def test_anchors_on_lane_a(script_text: str):
    assert "experiments/results/lane_a_landed" in script_text, (
        "Lane G V3 must anchor on Lane A's verified 1.15 [contest-CUDA] "
        "artifacts at experiments/results/lane_a_landed/"
    )


def test_anchors_on_lane_a_renderer_bin(script_text: str):
    assert (
        "experiments/results/lane_a_landed/iter_0/renderer.bin" in script_text
    ), "must reference Lane A's renderer.bin specifically"


def test_anchors_on_lane_a_poses(script_text: str):
    """Lane A's optimized_poses.pt is the warm-start prior for the TTO."""
    assert (
        "experiments/results/lane_a_landed/optimized_poses.pt" in script_text
    ), "must reference Lane A's optimized_poses.pt specifically"


# ── KL weight calibrated to post-bug-fix scale ─────────────────────────


def test_kl_distill_weight_0_002(script_text: str):
    """V3 corrected weight: 0.002 (V3-prev was 5e-6, V2 was 0.01, V1 was 1.0).

    Council math (post-fix): raw KL ≈ 2.7, scorer loss ≈ 0.05. Weight
    0.002 → KL contribution 5.4e-3 ≈ 10% of scorer = canonical Hinton
    auxiliary regime.

    We scope the search to the actual optimize_poses.py invocation block
    so historical weight values mentioned in script header comments
    (`V1 used --kl-distill-weight 1.0`) don't false-positive the parse.
    """
    m = re.search(
        r"experiments/optimize_poses\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None, "couldn't find optimize_poses.py invocation block"
    invocation = m.group(0)
    flag_match = re.search(r"--kl-distill-weight\s+(\S+)", invocation)
    assert flag_match is not None, (
        "Lane G V3 invocation must pass --kl-distill-weight"
    )
    val = float(flag_match.group(1))
    assert val == pytest.approx(0.002, rel=1e-9), (
        f"Lane G V3 corrected --kl-distill-weight must be 0.002. "
        f"Got {val}. (V3-prev's 5e-6 was sized for the BUGGY raw KL ~20000; "
        f"after commit f17de3bb the raw KL is ~2.7 and 5e-6 leaves the KL "
        f"contribution at 1.4e-5 — too small.)"
    )


def test_kl_distill_weight_not_v3_prev(script_text: str):
    """Defensive: ensure the V3-prev (broken) weight isn't lurking elsewhere
    in the invocation block (e.g., via a duplicate flag pass)."""
    m = re.search(
        r"experiments/optimize_poses\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None, "couldn't find optimize_poses.py invocation"
    invocation = m.group(0)
    # 5e-6 should NOT appear as a kl-distill-weight value in the invocation.
    assert "--kl-distill-weight 5e-6" not in invocation, (
        "Lane G V3 corrected must NOT use the V3-prev weight 5e-6 "
        "(too small for post-bug-fix raw KL ~2.7)"
    )
    assert "--kl-distill-weight 0.01" not in invocation, (
        "Lane G V3 corrected must NOT use the V2 weight 0.01"
    )
    assert "--kl-distill-weight 1.0" not in invocation, (
        "Lane G V3 corrected must NOT use the V1 weight 1.0"
    )


def test_kl_distill_temperature_2_0(script_text: str):
    """Hinton 2015 default + Quantizr-validated. Scope to the invocation
    block so header-comment mentions of 'temperature 2.0' don't shadow."""
    m = re.search(
        r"experiments/optimize_poses\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None, "couldn't find optimize_poses.py invocation block"
    invocation = m.group(0)
    flag_match = re.search(r"--kl-distill-temperature\s+(\S+)", invocation)
    assert flag_match is not None, (
        "Lane G V3 invocation must pass --kl-distill-temperature"
    )
    val = float(flag_match.group(1))
    assert val == pytest.approx(2.0, rel=1e-9), (
        f"Lane G V3 --kl-distill-temperature must be 2.0 (Hinton + Quantizr). "
        f"Got {val}."
    )


def test_kl_distill_flags_real_in_argparse(
    optimize_poses_argparse_flags: set[str],
):
    """`--kl-distill-weight` + `--kl-distill-temperature` MUST exist in
    optimize_poses.py argparse (verified at L159-171)."""
    assert "kl-distill-weight" in optimize_poses_argparse_flags, (
        "--kl-distill-weight not declared in optimize_poses.py argparse — "
        "CLAUDE.md forbids inventing CLI flags."
    )
    assert "kl-distill-temperature" in optimize_poses_argparse_flags, (
        "--kl-distill-temperature not declared in optimize_poses.py argparse"
    )


# ── Other CLI flags preserved from Lane A flow ─────────────────────────


def test_eval_roundtrip_present(script_text: str):
    """eval_roundtrip is a CLAUDE.md non-negotiable for every training path."""
    assert "--eval-roundtrip" in script_text, (
        "--eval-roundtrip required (CLAUDE.md non-negotiable: 2-11x "
        "proxy-auth gap without it)"
    )


def test_posetto_noise_std_0_5(script_text: str):
    """Fridrich C1 fix — 0.5 added pose noise during TTO simulates the
    inflate-side roundtrip noise."""
    m = re.search(r"--posetto-noise-std\s+(\S+)", script_text)
    assert m is not None, "Lane G V3 must pass --posetto-noise-std"
    val = float(m.group(1))
    assert val == pytest.approx(0.5, rel=1e-9), (
        f"Lane G V3 --posetto-noise-std must be 0.5 (Fridrich C1). Got {val}."
    )


def test_steps_500(script_text: str):
    """Same TTO budget as Lane A (500 steps × 600 pairs = 300k pair-steps)."""
    assert "--steps 500" in script_text, (
        "Lane G V3 must use --steps 500 (Lane A budget)"
    )


# ── Every flag passed to optimize_poses is real ────────────────────────


def test_all_optimize_poses_flags_in_script_are_real(
    script_text: str, optimize_poses_argparse_flags: set[str],
):
    """Every `--flag` in the optimize_poses.py invocation block MUST
    exist in its argparse (memory: feedback_dead_flag_wiring_pattern)."""
    m = re.search(
        r"experiments/optimize_poses\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None, "couldn't find optimize_poses.py invocation block"
    invocation = m.group(0)
    flags_used = set(re.findall(r"--([a-z][a-z0-9-]+)", invocation))
    bad = flags_used - optimize_poses_argparse_flags
    assert not bad, (
        f"Lane G V3 invokes optimize_poses.py with flags that don't exist "
        f"in its argparse: {sorted(bad)}. CLAUDE.md non-negotiable: "
        f"NEVER invent CLI flags. Run `grep add_argument "
        f"experiments/optimize_poses.py` to see real flags."
    )


# ── Device CUDA required (no MPS / CPU fallback) ───────────────────────


def test_device_cuda_required(script_text: str):
    assert "--device cuda" in script_text, "must use --device cuda"
    assert "--device mps" not in script_text, "MPS forbidden — drift 23x"
    assert "--device cpu" not in script_text, (
        "CPU forbidden in Lane G V3 (pose TTO GPU-only)"
    )


def test_no_mps_fallback(script_text: str):
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
            f"Lane G V3 must not reference MPS in device selection. "
            f"Match for /{pat}/: {m.group(0) if m else None!r}"
        )


# ── Provenance + heartbeat (canonical bootstrap pattern) ───────────────


def test_writes_provenance_json(script_text: str):
    assert "provenance.json" in script_text or "PROVENANCE=" in script_text


def test_writes_heartbeat_log(script_text: str):
    assert "heartbeat.log" in script_text or "HEARTBEAT=" in script_text


def test_provenance_records_predicted_band(script_text: str):
    """Council-signed predicted band [1.10, 1.18] must be in provenance."""
    assert "predicted_band" in script_text, (
        "provenance must record predicted_band"
    )
    assert (
        "[1.10, 1.18]" in script_text or "[1.1, 1.18]" in script_text
        or "[1.10, 1.18]" in script_text
    ), "Lane G V3 predicted band [1.10, 1.18] must appear in provenance"


def test_provenance_records_anchor_baseline(script_text: str):
    """V3 anchor baseline = Lane A's 1.15."""
    assert "anchor_score_baseline" in script_text
    assert "1.15" in script_text, (
        "provenance must record Lane A's 1.15 anchor"
    )


def test_provenance_records_kl_weight(script_text: str):
    """The corrected KL weight (0.002) must be in the provenance metadata
    so post-hoc analysis can verify the actual run config matched intent."""
    assert "kl_distill_weight" in script_text
    assert "0.002" in script_text, (
        "provenance must record kl_distill_weight = 0.002"
    )


def test_provenance_records_delta_from_v3_prev(script_text: str):
    """Operator-facing record of WHY this V3 corrected differs from V3-prev."""
    assert (
        "delta_from_v3_prev" in script_text or "kl_bug_fix_commit" in script_text
    ), (
        "provenance must record the V3 corrected delta from V3-prev "
        "(post-bug-fix KL weight recalibration) so post-hoc analysis "
        "is unambiguous"
    )


# ── Internal name lane_g_v3 (not lane_g, not lane_g_v2) ────────────────


def test_internal_name_lane_g_v3(script_text: str):
    """Logs from V1/V2/V3-prev/V3-corrected must not be conflated."""
    assert "lane_g_v3" in script_text, (
        "internal name must be lane_g_v3"
    )


def test_log_dir_lane_g_v3(script_text: str):
    assert "lane_g_v3_results" in script_text, (
        "LOG_DIR must be lane_g_v3_results"
    )


def test_archive_named_lane_g_v3(script_text: str):
    assert "archive_lane_g_v3.zip" in script_text, (
        "archive must be named archive_lane_g_v3.zip"
    )


def test_completion_marker_lane_g_v3(script_text: str):
    """The DONE marker is grepped by remote watchdogs."""
    assert "LANE_G_V3_DONE" in script_text, (
        "completion marker must be LANE_G_V3_DONE"
    )


# ── Archive build via Python zipfile ────────────────────────────────────


def test_no_shell_zip_binary(script_text: str):
    """PyTorch container has no `zip` shell binary."""
    code_lines = []
    for line in script_text.splitlines():
        if line.strip().startswith("#"):
            continue
        code_lines.append(line)
    code = "\n".join(code_lines)
    bad_match = re.search(r"(^|[\s;&|`\(])zip\s+(?!file)", code)
    assert not bad_match, (
        f"script must not invoke shell `zip` binary. Match: "
        f"{bad_match.group(0) if bad_match else None!r}"
    )


def test_uses_python_zipfile(script_text: str):
    assert "zipfile.ZipFile" in script_text, (
        "archive build must go through zipfile.ZipFile (Python stdlib)"
    )


def test_archive_contains_required_files(script_text: str):
    """Lane G V3 archive: Lane A renderer.bin + Lane A masks.mkv +
    NEW optimized_poses.pt."""
    assert "renderer.bin" in script_text
    assert "masks.mkv" in script_text
    assert "optimized_poses.pt" in script_text


# ── Auth eval on the actual archive ─────────────────────────────────────


def test_runs_contest_auth_eval(script_text: str):
    """CLAUDE.md auth-eval-everywhere rule."""
    assert "contest_auth_eval.py" in script_text, (
        "every chained experiment must end with a CUDA auth eval"
    )


def test_auth_eval_uses_built_archive(script_text: str):
    """Auth eval must use the SAME archive that would be submitted."""
    assert (
        "archive_lane_g_v3.zip" in script_text or "$ARCHIVE" in script_text
    ), "auth eval must use the Lane G V3 archive"


# ── env.sh + PYTHONHASHSEED determinism ────────────────────────────────


def test_sources_env_sh(script_text: str):
    assert "env.sh" in script_text, "must source $WORKSPACE/env.sh"


def test_python_hash_seed_pinned(script_text: str):
    assert "PYTHONHASHSEED" in script_text, (
        "PYTHONHASHSEED must be pinned for deterministic dict iteration"
    )


# ── Strict-scorer-rule compliance ──────────────────────────────────────


def test_no_scorer_load_at_inflate(script_text: str):
    """Auth eval must go through inflate.sh."""
    assert "inflat" in script_text.lower(), (
        "auth eval must go through inflate.sh (the strict-scorer-rule "
        "compliant inflate path)"
    )


def test_pytorch_cuda_alloc_conf(script_text: str):
    """V3-prev finding (inherited): KL distill adds a SECOND SegNet
    forward+backward; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    helps avoid OOM."""
    assert "PYTORCH_CUDA_ALLOC_CONF" in script_text, (
        "Lane G V3 must set PYTORCH_CUDA_ALLOC_CONF (KL distill OOM mitigation)"
    )


def test_batch_pairs_4(script_text: str):
    """V3-prev finding (inherited): --batch-pairs 4 fits in 13GB on 4090
    24GB; --batch-pairs 8 OOMs at 23GB peak with the second SegNet pass."""
    assert "--batch-pairs 4" in script_text, (
        "Lane G V3 must use --batch-pairs 4 (the OOM-safe value with KL distill)"
    )
