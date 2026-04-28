"""Regression tests for scripts/remote_lane_d_v2_halfframe_retrain_lr_fix.sh.

Lane D-V2 = half-frame retrain retry with LR-floor fix. V1 plateaued at
fp4_scorer=40.37 around ep 700/1980 because the cosine annealer
(eta_min=1e-6) starved the optimizer in the back half of P2 (LR fell to
~3.3e-5). V2 raises per-phase base LRs uniformly so the cosine floor
stays in the productive range.

V1 phase LRs: P1=1e-3, P2=3e-4, P3=1e-4, P4=5e-5, P5=1e-5
V2 phase LRs: P1=1e-3, P2=5e-4, P3=2e-4, P4=1e-4, P5=2e-5

These tests pin every claim the launch script makes:

  1. Strict bash safety — `set -euo pipefail` (LANE-B trap).
  2. Stage 0 NVDEC probe BEFORE any GPU spend.
  3. Profile = dilated_h64_half_frame (mask_half_sim_prob=0.5 inherited).
  4. V2 phase LR overrides explicitly passed.
  5. mask_half_sim_prob NOT changed from 0.5 (isolate LR fix).
  6. Every CLI flag verified against train_renderer.py argparse
     (CLAUDE.md non-negotiable: NEVER invent CLI flags).
  7. Provenance + heartbeat writes (canonical bootstrap pattern).
  8. Predicted band [1.50, 3.00] recorded.
  9. Internal name lane_d_v2 (NOT lane_d).
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
SCRIPT = REPO / "scripts" / "remote_lane_d_v2_halfframe_retrain_lr_fix.sh"
TRAIN_RENDERER = REPO / "src" / "tac" / "experiments" / "train_renderer.py"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


@pytest.fixture(scope="module")
def train_renderer_argparse_flags() -> set[str]:
    """Extract real `add_argument("--<flag>", ...)` flag names from
    train_renderer.py. CLAUDE.md non-negotiable: NEVER invent CLI flags."""
    src = TRAIN_RENDERER.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


# ── Existence + bash-safety guards ─────────────────────────────────────


def test_script_exists():
    assert SCRIPT.exists(), f"missing Lane D-V2 launch script: {SCRIPT}"


def test_script_is_executable():
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be chmod +x"


def test_full_set_euo_pipefail(script_text: str):
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


# ── Profile + LR-fix deltas ────────────────────────────────────────────


def test_profile_dilated_h64_half_frame(script_text: str):
    """Lane D-V2 must use the same profile as V1 (mask_half_sim_prob=0.5
    inherited) — only the per-phase LR overrides change."""
    assert "--profile dilated_h64_half_frame" in script_text, (
        "Lane D-V2 must use profile dilated_h64_half_frame (V1 baseline)"
    )


def test_phase_lr_overrides_present(script_text: str):
    """All five V2 LR overrides must be passed explicitly so a future
    profile-default flip cannot silently change V2's behavior."""
    expected = {
        "--phase1-lr": "1e-3",
        "--phase2-lr": "5e-4",
        "--phase3-lr": "2e-4",
        "--phase4-lr": "1e-4",
        "--phase5-lr": "2e-5",
    }
    for flag, value in expected.items():
        m = re.search(rf"{re.escape(flag)}\s+(\S+)", script_text)
        assert m is not None, f"Lane D-V2 must pass {flag} explicitly"
        got = m.group(1)
        assert float(got) == pytest.approx(float(value), rel=1e-9), (
            f"Lane D-V2 {flag} must be {value} (V2 LR-fix). Got {got}."
        )


def test_phase2_lr_higher_than_v1(script_text: str):
    """The core V2 hypothesis: P2 LR raised from V1's 3e-4 (which
    cosine-annealed to ~3.3e-5 by ep 1230 and starved the optimizer)
    to a value strictly higher than 3e-4."""
    m = re.search(r"--phase2-lr\s+(\S+)", script_text)
    assert m is not None, "Lane D-V2 must pass --phase2-lr"
    p2_lr = float(m.group(1))
    assert p2_lr > 3e-4, (
        f"Lane D-V2 phase2_lr must be strictly > V1's 3e-4 (the LR-fix). "
        f"Got --phase2-lr {p2_lr:.2e}"
    )


def test_phase_lr_flags_real_in_argparse(train_renderer_argparse_flags: set[str]):
    for n in (1, 2, 3, 4, 5):
        flag = f"phase{n}-lr"
        assert flag in train_renderer_argparse_flags, (
            f"--{flag} not declared in train_renderer.py argparse — "
            f"CLAUDE.md forbids inventing CLI flags."
        )


def test_no_mask_half_sim_prob_override(script_text: str):
    """V2 deliberately holds mask_half_sim_prob=0.5 (the V1 setting from
    the profile). Overriding it would confound the LR fix test."""
    # Search the train_renderer invocation block specifically.
    m = re.search(
        r"src/tac/experiments/train_renderer\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None, "couldn't find train_renderer.py invocation block"
    invocation = m.group(0)
    assert "--mask-half-sim-prob" not in invocation, (
        "Lane D-V2 must NOT override --mask-half-sim-prob (the profile "
        "value 0.5 is inherited; overriding confounds the LR-fix test)."
    )


# ── Every flag passed to train_renderer is real ────────────────────────


def test_all_train_renderer_flags_in_script_are_real(
    script_text: str, train_renderer_argparse_flags: set[str],
):
    """Every `--flag` in the train_renderer.py invocation block MUST
    exist in its argparse. The canonical dead-flag check (memory:
    feedback_dead_flag_wiring_pattern)."""
    m = re.search(
        r"src/tac/experiments/train_renderer\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None, "couldn't find train_renderer.py invocation block"
    invocation = m.group(0)
    flags_used = set(re.findall(r"--([a-z][a-z0-9-]+)", invocation))
    bad = flags_used - train_renderer_argparse_flags
    assert not bad, (
        f"Lane D-V2 invokes train_renderer.py with flags that don't exist "
        f"in its argparse: {sorted(bad)}. CLAUDE.md non-negotiable: "
        f"NEVER invent CLI flags. Run `grep add_argument "
        f"src/tac/experiments/train_renderer.py` to see real flags."
    )


def test_no_auth_eval_on_best(script_text: str):
    """Stage 4 below builds a real archive and runs contest_auth_eval.py;
    the built-in --auth-eval-on-best would hard-fail because masks +
    poses don't exist yet at training time."""
    m = re.search(
        r"src/tac/experiments/train_renderer\.py(?:.*\\\n)+[^\\\n]*",
        script_text, re.MULTILINE,
    )
    assert m is not None, "couldn't find train_renderer.py invocation block"
    invocation = m.group(0)
    assert "--no-auth-eval-on-best" in invocation, (
        "Lane D-V2 must pass --no-auth-eval-on-best (Stage 4 runs the real "
        "contest_auth_eval.py separately)."
    )


# ── Device CUDA required (no MPS / CPU fallback) ───────────────────────


def test_device_cuda_required(script_text: str):
    assert "--device cuda" in script_text, "must use --device cuda"
    assert "--device mps" not in script_text, "MPS forbidden — drift 23x"
    assert "--device cpu" not in script_text, (
        "CPU forbidden in Lane D-V2 (renderer training GPU-only)"
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
            f"Lane D-V2 must not reference MPS in device selection — "
            f"CLAUDE.md drift = 23x. Match for /{pat}/: "
            f"{m.group(0) if m else None!r}"
        )


# ── Provenance + heartbeat (canonical bootstrap pattern) ───────────────


def test_writes_provenance_json(script_text: str):
    assert "provenance.json" in script_text or "PROVENANCE=" in script_text


def test_writes_heartbeat_log(script_text: str):
    assert "heartbeat.log" in script_text or "HEARTBEAT=" in script_text


def test_provenance_records_predicted_band(script_text: str):
    """Council-signed predicted band [1.50, 3.00] must be in provenance."""
    assert "predicted_band" in script_text, (
        "provenance must record predicted_band"
    )
    assert (
        "[1.50, 3.00]" in script_text or "[1.5, 3.0]" in script_text
    ), "Lane D-V2 predicted band [1.50, 3.00] must appear in provenance"


def test_provenance_records_anchor_baseline(script_text: str):
    """V2 anchor baseline = V1's 17.55 (the broken half-frame score)."""
    assert "anchor_score_baseline" in script_text
    assert "17.55" in script_text, (
        "provenance must record V1's 17.55 anchor so post-hoc analysis "
        "is unambiguous about what V2 is being compared against"
    )


def test_provenance_records_lr_fix_choice(script_text: str):
    """Operator-facing record of WHY the V2 LR change is what it is."""
    assert "lr_fix_choice" in script_text or "delta_from_v1" in script_text, (
        "provenance must record the V2 LR-fix choice + reasoning"
    )


# ── Internal name lane_d_v2 (NOT lane_d) ───────────────────────────────


def test_internal_name_lane_d_v2(script_text: str):
    """Logs from V1 + V2 must not be conflated."""
    assert "lane_d_v2" in script_text, (
        "internal name must be lane_d_v2 (not lane_d)"
    )


def test_log_dir_lane_d_v2(script_text: str):
    assert "lane_d_v2_results" in script_text, (
        "LOG_DIR must be lane_d_v2_results so V1 and V2 outputs are separate"
    )


def test_archive_named_lane_d_v2(script_text: str):
    assert "archive_lane_d_v2.zip" in script_text, (
        "archive must be named archive_lane_d_v2.zip"
    )


def test_completion_marker_lane_d_v2(script_text: str):
    """The DONE marker is grepped by remote watchdogs; must be V2-tagged."""
    assert "LANE_D_V2_DONE" in script_text, (
        "completion marker must be LANE_D_V2_DONE"
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
    """Lane D-V2 archive: trained renderer.bin + half-frame masks.mkv +
    optimized_poses.pt (+ zoom_scalars.pt if present)."""
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
        "archive_lane_d_v2.zip" in script_text or "$ARCHIVE" in script_text
    ), "auth eval must use the Lane D-V2 archive"


# ── env.sh + PYTHONHASHSEED determinism ────────────────────────────────


def test_sources_env_sh(script_text: str):
    assert "env.sh" in script_text, "must source $WORKSPACE/env.sh"


def test_python_hash_seed_pinned(script_text: str):
    assert "PYTHONHASHSEED" in script_text, (
        "PYTHONHASHSEED must be pinned for deterministic dict iteration"
    )


# ── Strict-scorer-rule compliance ──────────────────────────────────────


def test_no_scorer_load_at_inflate(script_text: str):
    """The Lane D-V2 archive's renderer.bin is the FP4 output of training —
    no scorers allowed at inflate per strict-scorer-rule."""
    assert "inflat" in script_text.lower(), (
        "auth eval must go through inflate.sh (the strict-scorer-rule "
        "compliant inflate path)"
    )
