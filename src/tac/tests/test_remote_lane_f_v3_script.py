# SPDX-License-Identifier: MIT
"""Regression tests for scripts/remote_lane_f_v3_fp4_qat_int8warmup.sh.

Lane F-V3 = FP4 QAT engineering audit retry (V3): INT8 warmup phase
restored + LR lowered 20x. Anchored on Lane A (1.15 [contest-CUDA]).

V1 → 2.73 (silent zero-pose bug + 50 epoch micro-budget)
V2 → 1.79 (bug fixed, 500 ep, but --skip-int8-warmup + LR 5e-5 caused
            20x PoseNet regression because the FP4 fake-quant fitted
            unstable scales while the optimizer was simultaneously
            moving the FP32 weights)
V3 → predicted [1.30, 1.80] [contest-CUDA] — INT8 warmup (50 ep)
      anchors weight scales BEFORE FP4 fake-quant, then 500 FP4
      epochs at LR 2.5e-6 (20x lower than V2) cosine-anneal from a
      stable point.

These tests pin every claim the launch script makes:

  1. Strict bash safety — `set -euo pipefail` (LANE-B trap).
  2. Stage 0 NVDEC probe BEFORE any GPU spend.
  3. Anchor on Lane A (1.15 [contest-CUDA]).
  4. INT8 warmup explicitly enabled (--int8-warmup-epochs 50, NOT
     --skip-int8-warmup which V2 used).
  5. LR lowered to 2.5e-6 (V2 was 5e-5 — 20x reduction).
  6. --fp4-epochs preserved at 500 (the V2 budget — V1 was 50).
  7. Pose threading preserved (--poses, the V2 fix).
  8. Every CLI flag verified against qat_finetune.py argparse
     (CLAUDE.md non-negotiable: NEVER invent CLI flags).
  9. Provenance + heartbeat writes (canonical bootstrap pattern).
 10. Predicted band metadata recorded.
 11. Python zipfile (PyTorch container has no `zip` binary).
 12. Internal name `lane_f_v3` (NOT V2) so logs aren't conflated.
 13. No MPS / CPU device fallback.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_f_v3_fp4_qat_int8warmup.sh"
QAT_FINETUNE = REPO / "experiments" / "qat_finetune.py"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


@pytest.fixture(scope="module")
def qat_argparse_flags() -> set[str]:
    """Extract the actual `add_argument("--<flag>", ...)` flag names from
    qat_finetune.py. CLAUDE.md non-negotiable: NEVER invent CLI flags —
    always grep the target's argparse first."""
    src = QAT_FINETUNE.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


# ── Existence + bash-safety guards ─────────────────────────────────────


def test_script_exists():
    assert SCRIPT.exists(), f"missing Lane F-V3 launch script: {SCRIPT}"


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be chmod +x"


def test_full_set_euo_pipefail(script_text: str):
    """Belt-and-braces: assert canonical `set -euo pipefail`."""
    assert "set -euo pipefail" in script_text, (
        "script must use `set -euo pipefail` (LANE-B trap: `set -uo` no -e "
        "silently cascaded 6.5h + $2 of failure)"
    )


# ── Stage 0 NVDEC probe ─────────────────────────────────────────────────


def test_nvdec_probe_present(script_text: str):
    assert "probe_nvdec.sh" in script_text, (
        "Stage 0 NVDEC probe required — Vast.ai host variation can crash "
        "upstream/evaluate.py at the end (memory: "
        "feedback_vastai_nvdec_host_variation)"
    )


def test_nvdec_probe_fails_loud(script_text: str):
    """The probe must abort the script on failure, not WARN-and-continue."""
    probe_section = re.search(
        r"probe_nvdec\.sh.*?exit\s+\d+", script_text, re.DOTALL,
    )
    assert probe_section is not None, "NVDEC probe must `exit N` on failure"


# ── Anchor on Lane A (NOT baseline 2.29, NOT phantom 0.9001) ────────────


def test_anchors_on_lane_a(script_text: str):
    assert "experiments/results/lane_a_landed" in script_text, (
        "Lane F-V3 must anchor on Lane A's verified 1.15 [contest-CUDA] "
        "artifacts at experiments/results/lane_a_landed/"
    )


def test_anchors_on_lane_a_renderer_bin(script_text: str):
    assert (
        "experiments/results/lane_a_landed/iter_0/renderer.bin" in script_text
    ), "must reference Lane A's renderer.bin specifically"


def test_anchors_on_lane_a_poses(script_text: str):
    """Lane A's optimized_poses.pt is used both as --poses for QAT (the
    V2 bug-fix) and as the archive's pose payload."""
    assert (
        "experiments/results/lane_a_landed/optimized_poses.pt" in script_text
    ), "must reference Lane A's optimized_poses.pt specifically"


# ── V3 deltas vs V2: INT8 warmup restored ──────────────────────────────


def test_no_skip_int8_warmup_flag(script_text: str):
    """V2 used --skip-int8-warmup; V3 must NOT pass it to qat_finetune.py.
    That was the regression cause: INT8 warmup anchors weight scales
    BEFORE FP4 fake-quant.

    We narrow the check to the qat_finetune.py invocation block — the
    comments + log lines may DOCUMENT the V2 mistake (which is good
    operator context); only the actual flag-emission to the subprocess
    must omit --skip-int8-warmup.
    """
    m = re.search(
        r"experiments/qat_finetune\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None, "couldn't find qat_finetune.py invocation block"
    invocation = m.group(0)
    assert "--skip-int8-warmup" not in invocation, (
        "Lane F-V3 must NOT pass --skip-int8-warmup to qat_finetune.py. "
        "V2 used it and got 1.79 with 20x PoseNet regression because FP4 "
        "fake-quant fitted unstable weight scales. V3 restores INT8 warmup."
    )


def test_int8_warmup_epochs_50(script_text: str):
    """V3 explicitly sets INT8 warmup to 50 epochs (the qat_finetune
    default, made explicit so a future flag-default flip can't silently
    change behavior)."""
    assert "--int8-warmup-epochs 50" in script_text, (
        "Lane F-V3 must pass `--int8-warmup-epochs 50` explicitly. V2 "
        "skipped INT8 entirely; V3 uses 50 ep to anchor weight scales "
        "before FP4 quantization."
    )


def test_int8_warmup_epochs_flag_real_in_argparse(qat_argparse_flags: set[str]):
    """`--int8-warmup-epochs` MUST exist in qat_finetune.py's argparse.
    Verified at experiments/qat_finetune.py:630 manually; this test
    automates that gate (CLAUDE.md non-negotiable: NEVER invent flags)."""
    assert "int8-warmup-epochs" in qat_argparse_flags, (
        "--int8-warmup-epochs not declared in qat_finetune.py argparse — "
        "CLAUDE.md forbids inventing CLI flags. Verify with "
        "`grep add_argument experiments/qat_finetune.py`."
    )


# ── V3 deltas vs V2: LR lowered 20x ─────────────────────────────────────


def test_lr_2_5e_6(script_text: str):
    """V3 LR must be 2.5e-6 (V2 was 5e-5 — 20x reduction). Lower LR
    prevents perturbation of weight scales anchored by INT8 warmup."""
    # Match `--lr 2.5e-6` allowing scientific-notation variants.
    m = re.search(r"--lr\s+(\S+)", script_text)
    assert m is not None, "Lane F-V3 must pass --lr explicitly"
    lr_str = m.group(1)
    lr = float(lr_str)
    assert lr == pytest.approx(2.5e-6, rel=1e-9), (
        f"Lane F-V3 LR must be 2.5e-6 (V2 was 5e-5, 20x higher). "
        f"Got --lr {lr_str}. The cosine schedule from the INT8 warmup "
        f"endpoint must not blow up the auxiliary FiLM weights."
    )


# ── V3 keeps from V2: 500 FP4 epochs + pose threading ──────────────────


def test_fp4_epochs_500_preserved(script_text: str):
    """V2 fixed the V1 micro-budget bug (50→500 ep); V3 keeps that."""
    assert "--fp4-epochs 500" in script_text, (
        "Lane F-V3 must keep V2's --fp4-epochs 500 (V1 was 50 = 5% of "
        "canonical recipe — the original Bug 2)"
    )


def test_poses_flag_threaded(script_text: str):
    """V2 fixed the V1 silent zero-pose bug by threading --poses; V3 keeps it.
    The script passes `--poses "$ANCHOR_POSES"` where $ANCHOR_POSES is
    set to Lane A's optimized_poses.pt — so we must verify both:
      a) `--poses "$ANCHOR_POSES"` (or equivalent variable) appears
      b) ANCHOR_POSES is bound to a path containing optimized_poses.pt
    """
    # (a) the --poses flag appears with a value (variable expansion or
    # literal path).
    assert re.search(r"--poses\s+\S+", script_text), (
        "Lane F-V3 must thread --poses to qat_finetune.py. "
        "V1's silent zero-pose fallback caused +58% PoseNet regression."
    )
    # (b) the variable bound to the --poses value must reference Lane A's
    # optimized_poses.pt path. Look for either a literal path or the
    # ANCHOR_POSES variable assignment.
    has_anchor_poses_var = re.search(
        r'ANCHOR_POSES\s*=\s*"[^"]*optimized_poses\.pt"', script_text,
    )
    has_literal_poses_path = re.search(
        r'--poses\s+"?[^"\s]*optimized_poses\.pt', script_text,
    )
    assert has_anchor_poses_var or has_literal_poses_path, (
        "Lane F-V3 must bind --poses to a path containing "
        "optimized_poses.pt (Lane A's poses). Either pass the path "
        "literally or set an ANCHOR_POSES variable."
    )


def test_poses_flag_real_in_argparse(qat_argparse_flags: set[str]):
    assert "poses" in qat_argparse_flags, (
        "--poses not declared in qat_finetune.py argparse"
    )


# ── Every flag passed to qat_finetune.py is real ───────────────────────


def test_all_qat_flags_in_script_are_real(
    script_text: str, qat_argparse_flags: set[str],
):
    """Every `--flag` that appears in the qat_finetune invocation block
    MUST be a real argparse flag. This is the canonical dead-flag check
    (memory: feedback_dead_flag_wiring_pattern)."""
    # Find the qat_finetune.py invocation block by looking for the line and
    # all backslash-continued continuation lines.
    m = re.search(
        r"experiments/qat_finetune\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None, "couldn't find experiments/qat_finetune.py invocation"
    invocation = m.group(0)
    # Extract every --flag in the invocation block.
    flags_used = set(re.findall(r"--([a-z][a-z0-9-]+)", invocation))
    bad = flags_used - qat_argparse_flags
    assert not bad, (
        f"Lane F-V3 invokes qat_finetune.py with flags that don't exist "
        f"in its argparse: {sorted(bad)}. CLAUDE.md non-negotiable: "
        f"NEVER invent CLI flags. Run `grep add_argument "
        f"experiments/qat_finetune.py` to see real flags."
    )


# ── Device CUDA required (no MPS / CPU fallback) ───────────────────────


def test_device_cuda_required(script_text: str):
    assert "--device cuda" in script_text, "must use --device cuda"
    assert "--device mps" not in script_text, "MPS forbidden — drift 23x"
    assert "--device cpu" not in script_text, (
        "CPU forbidden in Lane F-V3 (FP4 QAT GPU-only)"
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
            f"Lane F-V3 must not reference MPS in device selection — "
            f"CLAUDE.md drift = 23x. Match for /{pat}/: "
            f"{m.group(0) if m else None!r}"
        )


# ── Provenance + heartbeat (canonical bootstrap pattern) ───────────────


def test_writes_provenance_json(script_text: str):
    assert "provenance.json" in script_text or "PROVENANCE=" in script_text


def test_writes_heartbeat_log(script_text: str):
    assert "heartbeat.log" in script_text or "HEARTBEAT=" in script_text


def test_provenance_records_predicted_band(script_text: str):
    """Council-signed predicted band [1.30, 1.80] must be in provenance."""
    assert "predicted_band" in script_text, (
        "provenance must record predicted_band"
    )
    # Specifically the V3 band [1.30, 1.80].
    assert (
        "[1.30, 1.80]" in script_text or "[1.3, 1.8]" in script_text
    ), "Lane F-V3 predicted band [1.30, 1.80] must appear in provenance"


def test_provenance_records_anchor_baseline(script_text: str):
    assert "anchor_score_baseline" in script_text


def test_provenance_records_delta_from_v2(script_text: str):
    """Operator-facing record of WHY this V3 differs from V2."""
    assert "delta_from_v2" in script_text or "int8_warmup_added" in script_text, (
        "provenance must record the V3 delta from V2 "
        "(int8_warmup_added + lr_lowered_20x) so post-hoc analysis is unambiguous"
    )


# ── Internal name lane_f_v3 (NOT V2) ────────────────────────────────────


def test_internal_name_lane_f_v3(script_text: str):
    """Logs from V2 + V3 must not be conflated. The output dir, archive
    name, and completion marker must use lane_f_v3."""
    assert "lane_f_v3" in script_text, (
        "internal name must be lane_f_v3 (not lane_f_v2)"
    )


def test_log_dir_lane_f_v3(script_text: str):
    assert "lane_f_v3_results" in script_text, (
        "LOG_DIR must be lane_f_v3_results so V2 and V3 outputs are separate"
    )


def test_archive_named_lane_f_v3(script_text: str):
    assert "archive_lane_f_v3.zip" in script_text, (
        "archive must be named archive_lane_f_v3.zip"
    )


def test_completion_marker_lane_f_v3(script_text: str):
    """The DONE marker is grepped by remote watchdogs; must be V3-tagged."""
    assert "LANE_F_V3_DONE" in script_text, (
        "completion marker must be LANE_F_V3_DONE"
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
        f"script must not invoke shell `zip` binary "
        f"(use Python zipfile). Match: "
        f"{bad_match.group(0) if bad_match else None!r}"
    )


def test_uses_python_zipfile(script_text: str):
    assert "zipfile.ZipFile" in script_text, (
        "archive build must go through zipfile.ZipFile (Python stdlib)"
    )


def test_archive_contains_required_files(script_text: str):
    """Lane F-V3 archive: FP4 renderer.bin + Lane A masks.mkv +
    Lane A optimized_poses.pt."""
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
        "archive_lane_f_v3.zip" in script_text or "$ARCHIVE" in script_text
    ), "auth eval must use the Lane F-V3 archive"


# ── env.sh + PYTHONHASHSEED determinism ────────────────────────────────


def test_sources_env_sh(script_text: str):
    """Source env.sh for canonical environment (CLAUDE.md remote-bootstraps)."""
    assert "env.sh" in script_text, "must source $WORKSPACE/env.sh"


def test_python_hash_seed_pinned(script_text: str):
    """Determinism: PYTHONHASHSEED must be pinned for reproducibility."""
    assert "PYTHONHASHSEED" in script_text, (
        "PYTHONHASHSEED must be pinned for deterministic dict iteration "
        "(canonical pipeline standard)"
    )


# ── Strict-scorer-rule compliance ──────────────────────────────────────


def test_no_scorer_load_at_inflate(script_text: str):
    """The Lane F-V3 archive's renderer.bin is the FP4 output of QAT —
    no scorers allowed at inflate per strict-scorer-rule. The script
    inflates via inflate.sh which is the canonical compliant path."""
    assert "inflate.sh" in script_text, (
        "auth eval must go through inflate.sh (the strict-scorer-rule "
        "compliant inflate path)"
    )
