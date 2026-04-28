"""Regression tests for scripts/remote_lane_v_quantizr_replica_88k_halfframe.sh.

Lane V = Quantizr-replica (88K params + DSConv + FiLM + KL distill T=2.0)
trained from epoch 0 with mask_half_sim_prob=1.0 (always-on warp expansion).
Council 2026-04-27 — biggest single swing in strategy. Predicted band
[0.50, 1.10] standalone; [0.30, 0.55] stacked with Lane A+C. Cost $4-5.

These tests pin every claim the launch script makes:

  1. Strict bash safety — `set -euo pipefail` (the cascade trap that ate
     LANE-B 6.5h + $2 in 2026-04-26 must not reappear).
  2. Stage 0 NVDEC probe BEFORE any GPU spend (memory:
     feedback_vastai_nvdec_host_variation).
  3. Profile reference — the script must use --profile
     quantizr_replica_88k_halfframe.
  4. Every CLI flag passed to train_renderer.py MUST exist in
     train_renderer.py's argparse. CLAUDE.md NEVER-INVENT-CLI-FLAGS rule.
  5. Provenance + heartbeat writes (canonical bootstrap pattern).
  6. Predicted band metadata (council pre-registered the band [0.50, 1.10]).
  7. Python zipfile (NOT shell `zip`).
  8. [contest-CUDA] tag in completion log.
  9. Half-frame archive build with --half-frame flag near a profile mention
     (preflight rule: half-frame requires a half-frame-trained profile).
 10. Pose TTO with posetto_noise_std=0.5 matching the profile.
 11. Auth eval on the EXACT archive that would be submitted (CLAUDE.md
     auth-eval-everywhere + auth-eval-on-submission-archive rules).
 12. RESULT_JSON validation gate (LANE-B silent-crash guard).
 13. NVDEC probe fails loud (exit, not WARN-and-continue).
 14. No MPS / CPU fallback (CLAUDE.md MPS-CUDA drift 23x rule).
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_v_quantizr_replica_88k_halfframe.sh"
TRAIN_RENDERER = REPO / "src" / "tac" / "experiments" / "train_renderer.py"
OPTIMIZE_POSES = REPO / "experiments" / "optimize_poses.py"
BUILD_ARCHIVE = REPO / "experiments" / "build_baseline_archive.py"
CONTEST_AUTH_EVAL = REPO / "experiments" / "contest_auth_eval.py"


@pytest.fixture(scope="module")
def script_text() -> str:
    return SCRIPT.read_text()


@pytest.fixture(scope="module")
def train_renderer_argparse_flags() -> set[str]:
    """Extract the actual `add_argument("--<flag>", ...)` flag names from
    train_renderer.py. We use these to verify the launch script never
    invents a flag that doesn't exist (CLAUDE.md non-negotiable rule)."""
    src = TRAIN_RENDERER.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


@pytest.fixture(scope="module")
def optimize_poses_argparse_flags() -> set[str]:
    src = OPTIMIZE_POSES.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


@pytest.fixture(scope="module")
def build_archive_argparse_flags() -> set[str]:
    src = BUILD_ARCHIVE.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


@pytest.fixture(scope="module")
def contest_auth_eval_argparse_flags() -> set[str]:
    src = CONTEST_AUTH_EVAL.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


# ── Existence + executability ──────────────────────────────────────────


def test_script_exists():
    assert SCRIPT.exists(), f"missing Lane V launch script: {SCRIPT}"


def test_script_is_executable():
    """Bash scripts in scripts/ should be chmod +x so the parent agent can
    invoke directly without `bash` prefix."""
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be executable"


# ── Bash safety ────────────────────────────────────────────────────────


def test_set_e_present(script_text: str):
    """The `set -uo pipefail` trap (no -e) silently cascades failures.
    LANE-B 2026-04-26 burned 6.5h + $2 because of this exact pattern."""
    has_set_e = bool(re.search(r"^set -[eu]*e[uo]*\s", script_text, re.MULTILINE))
    assert has_set_e, "script must use `set -e` (or -euo / -eu) to abort on first failure"


def test_full_set_euo_pipefail(script_text: str):
    """Belt-and-braces: assert the canonical `set -euo pipefail` line."""
    assert "set -euo pipefail" in script_text, (
        "script must use the canonical `set -euo pipefail` (CLAUDE.md "
        "non-negotiable; the cascade trap that ate LANE-B 6.5h + $2)"
    )


# ── Stage 0 NVDEC probe ────────────────────────────────────────────────


def test_nvdec_probe_present(script_text: str):
    """Stage 0 MUST run probe_nvdec.sh BEFORE any GPU spend. Memory:
    feedback_vastai_nvdec_host_variation."""
    assert "probe_nvdec.sh" in script_text, (
        "Stage 0 NVDEC probe required — different Vast.ai hosts have "
        "different NVDEC support even with the same 4090 image."
    )


def test_nvdec_probe_fails_loud(script_text: str):
    """The probe must abort the script (not WARN-and-continue) when NVDEC
    is missing. Otherwise we burn 30 min of setup before discovering."""
    probe_section = re.search(
        r"probe_nvdec\.sh.*?exit\s+\d+", script_text, re.DOTALL,
    )
    assert probe_section is not None, (
        "NVDEC probe must abort with `exit N` on failure"
    )


# ── Profile reference ──────────────────────────────────────────────────


def test_uses_lane_v_profile(script_text: str):
    """Lane V trains from scratch with --profile quantizr_replica_88k_halfframe.
    No fallback to other profiles."""
    assert "--profile quantizr_replica_88k_halfframe" in script_text, (
        "must reference --profile quantizr_replica_88k_halfframe"
    )


def test_no_other_profile_used(script_text: str):
    """Defensive: ensure no rogue --profile <other> sneaks in. The grep is
    permissive (only matches `--profile <name>` patterns), so it should
    only find quantizr_replica_88k_halfframe."""
    matches = re.findall(r"--profile\s+(\w+)", script_text)
    other = [m for m in matches if m != "quantizr_replica_88k_halfframe"]
    assert not other, (
        f"script references unexpected profiles {other} alongside Lane V's "
        f"quantizr_replica_88k_halfframe"
    )


# ── CLI flag wiring (CLAUDE.md NEVER-INVENT-CLI-FLAGS) ─────────────────


def test_train_renderer_flags_all_real(
    script_text: str, train_renderer_argparse_flags: set[str],
):
    """Every CLI flag in the train_renderer.py invocation MUST exist in
    its argparse. CLAUDE.md NEVER-INVENT-CLI-FLAGS non-negotiable.
    Memory: feedback_dead_flag_wiring_pattern (the 2026-04-26
    --auth-eval-masks invented-flag burn that took 3 review rounds)."""
    # Extract the train_renderer.py invocation block.
    m = re.search(
        r"src/tac/experiments/train_renderer\.py\s*\\(.*?)(?=\n\s*BEST_FP32=|\Z)",
        script_text,
        re.DOTALL,
    )
    assert m is not None, (
        "could not locate train_renderer.py invocation in script"
    )
    used = set(re.findall(r"\B--([a-z][a-z0-9-]+)", m.group(1)))
    invented = used - train_renderer_argparse_flags
    assert not invented, (
        f"INVENTED FLAGS: {sorted(invented)} not in train_renderer.py "
        f"argparse. CLAUDE.md NEVER-INVENT-CLI-FLAGS rule. Verify with "
        f"`grep add_argument src/tac/experiments/train_renderer.py`."
    )


def test_optimize_poses_flags_all_real(
    script_text: str, optimize_poses_argparse_flags: set[str],
):
    """Every CLI flag in the optimize_poses.py invocation MUST exist in
    its argparse."""
    m = re.search(
        r"experiments/optimize_poses\.py\s*\\(.*?)(?=\n\s*\[|\Z)",
        script_text,
        re.DOTALL,
    )
    assert m is not None, (
        "could not locate optimize_poses.py invocation in script"
    )
    used = set(re.findall(r"\B--([a-z][a-z0-9-]+)", m.group(1)))
    invented = used - optimize_poses_argparse_flags
    assert not invented, (
        f"INVENTED FLAGS in optimize_poses invocation: {sorted(invented)} "
        f"not in optimize_poses.py argparse"
    )


def test_build_archive_flags_all_real(
    script_text: str, build_archive_argparse_flags: set[str],
):
    """Every CLI flag in the build_baseline_archive.py invocation MUST
    exist in its argparse."""
    m = re.search(
        r"experiments/build_baseline_archive\.py\s*\\(.*?)(?=\nmkdir|\Z)",
        script_text,
        re.DOTALL,
    )
    assert m is not None, (
        "could not locate build_baseline_archive.py invocation in script"
    )
    used = set(re.findall(r"\B--([a-z][a-z0-9-]+)", m.group(1)))
    invented = used - build_archive_argparse_flags
    assert not invented, (
        f"INVENTED FLAGS in build_baseline_archive invocation: "
        f"{sorted(invented)} not in build_baseline_archive.py argparse"
    )


def test_contest_auth_eval_flags_all_real(
    script_text: str, contest_auth_eval_argparse_flags: set[str],
):
    """Every CLI flag in the contest_auth_eval.py invocation MUST exist in
    its argparse."""
    m = re.search(
        r"experiments/contest_auth_eval\.py\s*\\(.*?)(?=\nif !|\Z)",
        script_text,
        re.DOTALL,
    )
    assert m is not None, (
        "could not locate contest_auth_eval.py invocation in script"
    )
    used = set(re.findall(r"\B--([a-z][a-z0-9-]+)", m.group(1)))
    invented = used - contest_auth_eval_argparse_flags
    assert not invented, (
        f"INVENTED FLAGS in contest_auth_eval invocation: "
        f"{sorted(invented)} not in contest_auth_eval.py argparse"
    )


def test_uses_no_auth_eval_on_best(script_text: str):
    """Stage 4 builds the real archive separately and runs contest_auth_eval
    against it; train_renderer's built-in auth-eval would need
    --auth-eval-masks AND --auth-eval-poses (which don't exist yet at this
    point in the chain — masks haven't been encoded, poses haven't been
    TTO-optimized). So the script MUST disable the built-in eval."""
    assert "--no-auth-eval-on-best" in script_text, (
        "Lane V Stage 1 must disable train_renderer's built-in auth-eval "
        "(masks/poses don't exist yet — Stage 4 runs the real auth eval)"
    )


def test_use_qat_enabled(script_text: str):
    """Lane V trains with the full 5-stage QAT pipeline (anchor → finetune
    → joint → QAT → final). The QAT phases require --use-qat to enable
    FakeQuantFP4 in train_renderer."""
    assert "--use-qat" in script_text


def test_device_cuda_required(script_text: str):
    """No MPS or CPU fallback — CLAUDE.md MPS-CUDA drift is 23x on PoseNet.
    Score must be measured on CUDA only."""
    assert "--device cuda" in script_text, "Lane V must use --device cuda"


def test_no_mps_fallback(script_text: str):
    """CLAUDE.md FORBIDDEN: no MPS fallback. PoseNet drift on MPS is 23x."""
    assert "--device mps" not in script_text, (
        "MPS forbidden — PoseNet drift 23x on MPS vs CUDA"
    )


def test_no_cpu_fallback(script_text: str):
    """CPU is allowed only for `--device cpu` deterministic-bytes builds; the
    Lane V compute path is GPU-only, so cpu device must not appear."""
    # Allow the literal substring `cpu` only inside comments / strings — but
    # never as `--device cpu` on a command line.
    assert "--device cpu" not in script_text, (
        "Lane V compute path is GPU-only; --device cpu would silently "
        "produce invalid scores"
    )


# ── Half-frame archive build (preflight requirement) ───────────────────


def test_half_frame_archive_build_present(script_text: str):
    """Stage 2 must build a half-frame archive (--half-frame on
    build_baseline_archive.py)."""
    assert "--half-frame" in script_text, (
        "Lane V's whole point is the half-frame archive; --half-frame "
        "must appear on the build_baseline_archive invocation"
    )


def test_half_frame_archive_assertion_present(script_text: str):
    """The bootstrap must include the belt-and-braces assertion that the
    profile being used has mask_half_sim_prob>0 OR use_zoom_flow=True
    (matches the preflight rule). This is a runtime gate inside the
    script — Python `assert` statement on PROFILES."""
    assert "halfframe-profile-assertion" in script_text, (
        "must include the halfframe-profile-assertion before the "
        "--half-frame archive build"
    )
    assert "mask_half_sim_prob" in script_text


# ── Pose TTO with posetto_noise_std matching profile ───────────────────


def test_posetto_noise_std_matches_profile(script_text: str):
    """The optimize_poses Stage 3 invocation must pass
    --posetto-noise-std 0.5 to match the profile's posetto_noise_std=0.5
    field. Mismatching this would mean the profile field is decorative."""
    assert "--posetto-noise-std 0.5" in script_text, (
        "Stage 3 pose TTO must use --posetto-noise-std 0.5 to match the "
        "profile's posetto_noise_std=0.5"
    )


def test_pose_tto_uses_eval_roundtrip(script_text: str):
    """eval_roundtrip is non-negotiable; pose TTO must enable it too."""
    # Find the optimize_poses invocation and check for --eval-roundtrip.
    m = re.search(
        r"experiments/optimize_poses\.py(.*?)(?=\n\s*\[|\Z)",
        script_text,
        re.DOTALL,
    )
    assert m is not None
    assert "--eval-roundtrip" in m.group(1), (
        "Stage 3 pose TTO must use --eval-roundtrip"
    )


# ── Provenance + heartbeat (canonical bootstrap pattern) ───────────────


def test_writes_provenance_json(script_text: str):
    """Every remote run must emit provenance.json (memory:
    feedback_canonical_remote_bootstraps)."""
    assert "provenance.json" in script_text or "PROVENANCE=" in script_text


def test_writes_heartbeat_log(script_text: str):
    """Heartbeat per minute is the watchdog signal (CLAUDE.md remote code
    parity rule: tmux session existence is NOT a heartbeat)."""
    assert "heartbeat.log" in script_text or "HEARTBEAT=" in script_text


def test_provenance_records_predicted_band(script_text: str):
    """The council signed off on a concrete predicted band [0.50, 1.10]
    before launch. The provenance JSON must record it so post-hoc analysis
    can compare predicted vs measured."""
    assert "predicted_band" in script_text, (
        "provenance must record the predicted score band"
    )
    # And the band itself should appear in the script.
    assert "0.50" in script_text and "1.10" in script_text, (
        "the [0.50, 1.10] predicted band must be encoded in the script"
    )


def test_provenance_records_anchor_baseline(script_text: str):
    """Anchor score baseline (Lane A 1.15) must be recorded so post-hoc
    delta is unambiguous."""
    assert "anchor_score_baseline" in script_text


def test_provenance_records_lane_v_premise(script_text: str):
    """The Lane V bet (mask_half_sim_prob=1.0 from epoch 0 vs Lane D's 0.5
    retrofit which failed) must be encoded so a fresh agent can read why
    this lane was launched."""
    assert "lane_v_premise" in script_text


# ── Archive build via Python zipfile ───────────────────────────────────


def test_no_shell_zip_binary(script_text: str):
    """PyTorch container has no `zip` shell binary (memory:
    feedback_zip_dep_bootstrap_trap). Use Python zipfile instead."""
    code_lines = []
    for line in script_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        code_lines.append(line)
    code = "\n".join(code_lines)
    bad_match = re.search(r"(^|[\s;&|`\(])zip\s+(?!file)", code)
    assert not bad_match, (
        f"script should not invoke shell `zip` binary (missing on PyTorch "
        f"container); use Python zipfile instead. "
        f"Match: {bad_match.group(0) if bad_match else None!r}"
    )


def test_uses_python_zipfile(script_text: str):
    """Affirmative: archive build must go through Python's zipfile module."""
    assert "zipfile.ZipFile" in script_text, (
        "Lane V archive must be built via zipfile.ZipFile (Python stdlib)"
    )


def test_archive_contains_required_files(script_text: str):
    """The Lane V archive must contain renderer.bin + masks.mkv +
    optimized_poses.pt — the three canonical artifacts."""
    assert "renderer.bin" in script_text
    assert "masks.mkv" in script_text
    assert "optimized_poses.pt" in script_text


def test_archive_includes_zoom_scalars_when_present(script_text: str):
    """Lane V uses use_zoom_flow=True at training; the inflate-side warp
    expansion needs zoom_scalars.pt. Script must conditionally include it
    in the archive."""
    assert "zoom_scalars.pt" in script_text, (
        "Lane V trained with use_zoom_flow=True; zoom_scalars.pt must be "
        "bundled into the archive for the inflate-side warp expansion"
    )


# ── Auth eval on the actual archive ────────────────────────────────────


def test_runs_contest_auth_eval(script_text: str):
    """Lane V must end with contest_auth_eval.py against the EXACT archive
    that would be submitted (CLAUDE.md auth-eval-everywhere rule)."""
    assert "contest_auth_eval.py" in script_text


def test_auth_eval_uses_built_archive(script_text: str):
    """auth-eval must run on archive_lane_v.zip we just built (CLAUDE.md:
    'the EXACT archive that will be submitted')."""
    assert "archive_lane_v.zip" in script_text or "$ARCHIVE" in script_text


def test_auth_eval_result_validated(script_text: str):
    """Detect auth_eval crashes — guard against silent zero-exit on a
    crashed eval (LANE-B-style cascade pattern)."""
    assert "RESULT_JSON" in script_text, (
        "must validate RESULT_JSON line in auth_eval.log to catch silent "
        "auth-eval crashes (LANE-B 2026-04-26 cascade pattern)"
    )


# ── Lane tag on the completion log [contest-CUDA] ──────────────────────


def test_completion_log_tags_contest_cuda(script_text: str):
    """CLAUDE.md non-negotiable: every score reported must carry a lane tag
    (`[contest-CUDA]`, `[advisory only]`, `[MPS-PROXY]`, ...)."""
    assert "[contest-CUDA]" in script_text, (
        "completion log must tag the Lane V result [contest-CUDA]"
    )


# ── Predicted band rationale ───────────────────────────────────────────


def test_script_documents_lane_d_failure(script_text: str):
    """The header must explicitly explain the Lane V vs Lane D distinction
    (joint vs retrofit) so a future operator can't accidentally argue that
    Lane V was redundant with Lane D."""
    text_lower = script_text.lower()
    assert "lane d" in text_lower or "retrofit" in text_lower, (
        "script header must reference Lane D's failed retrofit so the "
        "Lane V premise is documented"
    )


def test_script_documents_kl_distill_post_fix_math(script_text: str):
    """The header must explain why kl_distill_weight=0.002 (POST-FIX), not
    1.0 like DEN/SHIRAZ/Lane D used. Otherwise a future operator may bump
    it back to 1.0 and resurrect the 5000× amplification bug."""
    assert "POST-FIX" in script_text or "post-fix" in script_text or "post-2026-04-27" in script_text.lower(), (
        "script header must document the POST-FIX math justifying "
        "kl_distill_weight=0.002 (vs the pre-fix 1.0 that ran 5000× over)"
    )


# ── Pre-flight gates (script's own internal validation) ────────────────


def test_script_runs_profile_preflight(script_text: str):
    """The script's pre-flight phase must call preflight_profiles to
    catch profile-resolver drift before training starts."""
    assert "preflight_profiles" in script_text, (
        "script must call preflight_profiles in its pre-flight phase"
    )


def test_script_runs_dead_flag_scan(script_text: str):
    """The script's pre-flight phase must include the argparse dead-flag
    scan (the same regex check this test file applies, but at runtime so
    a stale script catches drift before launch)."""
    assert "INVENTED FLAGS" in script_text, (
        "script must include a runtime argparse dead-flag scan (same "
        "regex contract this test file enforces statically)"
    )


def test_script_runs_param_count_smoke(script_text: str):
    """The script's pre-flight phase must build a model from the profile
    and verify the param count is in the 88K Quantizr-class budget."""
    assert "param count" in script_text or "param_count" in script_text or "param count target" in script_text.lower(), (
        "script must include a param-count smoke check before launch"
    )


# ── Required artifact gates ─────────────────────────────────────────────


def test_required_artifacts_checked(script_text: str):
    """The script must check that scorer weights + GT video are present
    BEFORE starting any GPU work."""
    assert "segnet.safetensors" in script_text
    assert "posenet.safetensors" in script_text
    assert "0.mkv" in script_text
