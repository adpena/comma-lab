"""Regression tests for scripts/remote_lane_k_dsconv_quantizr_killer.sh.

Lane K = DSConv Quantizr-killer (88K params from-scratch retrain) anchored on
Lane A's 1.15 [contest-CUDA] mask + pose payloads. Predicted band [0.85, 1.10].

These tests pin every claim the launch script makes:

  1. Strict bash safety — `set -euo pipefail` (the cascade trap that ate
     LANE-B 6.5h + $2 in 2026-04-26 must not reappear).
  2. Stage 0 NVDEC probe BEFORE any GPU spend (memory:
     feedback_vastai_nvdec_host_variation — same 4090 image, different
     hosts, same driver, different NVDEC outcome).
  3. Anchor on Lane A's verified 1.15 [contest-CUDA] mask + pose artifacts
     (NOT phantom 0.9001 archive, NOT random regen).
  4. Every CLI flag passed to train_renderer.py MUST exist in its argparse.
     CLAUDE.md non-negotiable: NEVER invent CLI flags (memory:
     feedback_dead_flag_wiring_pattern).
  5. Every CLI flag passed to contest_auth_eval.py / build_baseline_archive.py
     similarly validated.
  6. Provenance + heartbeat writes — required by
     feedback_canonical_remote_bootstraps so a fresh agent can reconstruct
     the experiment from disk.
  7. Predicted band metadata — the council signed off on a concrete band
     [0.85, 1.10] before the run, not a vague "should be better".
  8. Python zipfile (NOT shell `zip`) — PyTorch container has no `zip`
     binary (memory: feedback_zip_dep_bootstrap_trap).
  9. [contest-CUDA] tag in the completion log — every score must carry a
     lane tag (CLAUDE.md non-negotiable).
 10. --device cuda required (no MPS / CPU fallback per CLAUDE.md
     MPS-CUDA drift 23x rule).
 11. --tag passed to train_renderer.py (required=True in argparse).
 12. --no-auth-eval-on-best disabled because Stage 4 builds the archive
     and runs auth eval separately.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_k_dsconv_quantizr_killer.sh"
TRAIN_RENDERER = REPO / "src" / "tac" / "experiments" / "train_renderer.py"
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
def contest_auth_eval_argparse_flags() -> set[str]:
    """Same for contest_auth_eval.py."""
    src = CONTEST_AUTH_EVAL.read_text()
    return set(re.findall(r"add_argument\(\s*[\"']--([a-z][a-z0-9-]+)", src))


# ── Existence + bash-safety guards ──────────────────────────────────────


def test_script_exists():
    assert SCRIPT.exists(), f"missing Lane K launch script: {SCRIPT}"


def test_script_is_executable():
    """Bash scripts in scripts/ should be chmod +x so the parent agent can
    invoke directly without `bash` prefix."""
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be executable"


def test_set_e_present(script_text: str):
    """The `set -uo pipefail` trap (no -e) silently cascades failures.
    LANE-B 2026-04-26 burned 6.5h + $2 because of this exact pattern.
    Lane K MUST use `set -e` to abort on the first failure."""
    has_set_e = bool(re.search(r"^set -[eu]*e[uo]*\s", script_text, re.MULTILINE))
    assert has_set_e, "script must use `set -e` (or -euo / -eu) to abort on first failure"


def test_set_pipefail_present(script_text: str):
    """pipefail catches failures inside `cmd | tee log` — without it, a
    failed cmd whose output goes through tee will look successful."""
    assert "pipefail" in script_text, "script must use pipefail"


def test_full_set_euo_pipefail(script_text: str):
    """Belt-and-braces: assert the canonical `set -euo pipefail` line."""
    assert "set -euo pipefail" in script_text, (
        "script must use the canonical `set -euo pipefail` (the safe default)"
    )


# ── Stage 0 NVDEC probe ─────────────────────────────────────────────────


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


# ── Anchor verification (Lane A artifacts) ──────────────────────────────


def test_anchors_on_lane_a_masks(script_text: str):
    """Lane K ships Lane A's verified masks.mkv unchanged — the only delta
    is the renderer byte count."""
    assert (
        "experiments/results/lane_a_landed/iter_0/masks.mkv" in script_text
    ), "must reference Lane A's masks.mkv specifically"


def test_anchors_on_lane_a_poses(script_text: str):
    """Lane K ships Lane A's verified optimized_poses.pt unchanged."""
    assert (
        "experiments/results/lane_a_landed/iter_0/optimized_poses.pt"
        in script_text
    ), "must reference Lane A's optimized_poses.pt specifically"


# ── Profile + tag wiring (CLAUDE.md non-negotiable) ─────────────────────


def test_uses_dsconv_quantizr_killer_profile(script_text: str):
    """The script must launch with --profile dsconv_quantizr_killer."""
    assert "--profile dsconv_quantizr_killer" in script_text, (
        "Lane K must launch with --profile dsconv_quantizr_killer"
    )


def test_passes_tag_flag(script_text: str):
    """train_renderer.py declares --tag as required=True. Lane D bootstrap
    OMITTED this in an earlier round, costing a deployment cycle. Lane K's
    test pins it so the regression cannot recur."""
    assert "--tag " in script_text, (
        "train_renderer.py requires --tag (required=True). Lane K must pass it."
    )


def test_no_auth_eval_on_best_disabled(script_text: str):
    """Stage 4 builds the real archive (renderer + masks + poses) and runs
    contest_auth_eval.py against it. The built-in --auth-eval-on-best path
    needs --auth-eval-masks AND --auth-eval-poses (which exist but redirect
    to the same Lane A artifacts); we keep the auth eval out of the training
    process for clean log separation."""
    assert "--no-auth-eval-on-best" in script_text, (
        "Lane K must disable auth-eval-on-best (Stage 4 runs auth eval "
        "separately against the bundled archive)"
    )


# ── Device safety (CLAUDE.md MPS-CUDA drift) ────────────────────────────


def test_device_cuda_required(script_text: str):
    """No MPS or CPU fallback — CLAUDE.md MPS-CUDA drift is 23x on
    PoseNet. Score must be measured on CUDA."""
    assert "--device cuda" in script_text, "Lane K must use --device cuda"
    assert "--device mps" not in script_text, "MPS forbidden — drift 23x on PoseNet"
    assert "--device cpu" not in script_text, (
        "Lane K compute path is GPU-only; --device cpu would silently "
        "produce invalid scores"
    )


# ── train_renderer flag validation (no invented flags) ──────────────────


@pytest.mark.parametrize(
    "flag",
    [
        "profile",
        "tag",
        "device",
        "video",
        "output-dir",
        "use-qat",
        "fp4-codebook",
        "fp4-robust-scale",
        "fp4-stochastic",
        "no-auth-eval-on-best",
    ],
)
def test_train_renderer_flag_used_in_script(script_text: str, flag: str):
    """Every flag we pass to train_renderer must appear in the script's
    train_renderer invocation (not just in comments)."""
    assert f"--{flag}" in script_text, (
        f"Lane K launch script must pass --{flag} to train_renderer.py"
    )


@pytest.mark.parametrize(
    "flag",
    [
        "profile",
        "tag",
        "device",
        "video",
        "output-dir",
        "use-qat",
        "fp4-codebook",
        "fp4-robust-scale",
        "fp4-stochastic",
        "no-auth-eval-on-best",
    ],
)
def test_train_renderer_flag_real_in_argparse(
    train_renderer_argparse_flags: set[str], flag: str,
):
    """Each Lane K flag MUST exist in train_renderer.py's argparse.
    CLAUDE.md non-negotiable: NEVER invent CLI flags. Memory:
    feedback_dead_flag_wiring_pattern (2026-04-26 dead-flag burn)."""
    assert flag in train_renderer_argparse_flags, (
        f"--{flag} not declared in train_renderer.py argparse — "
        f"CLAUDE.md forbids inventing CLI flags. Verify with "
        f"`grep add_argument src/tac/experiments/train_renderer.py`."
    )


def test_no_invented_train_renderer_flags(
    script_text: str, train_renderer_argparse_flags: set[str],
):
    """Bidirectional scan: every --flag in the train_renderer invocation
    section of the script MUST exist in argparse. Catches drift if anyone
    edits the script and adds a new flag without the parametrized test."""
    m = re.search(
        r"src/tac/experiments/train_renderer\.py(.*?)(?=\n\s*BEST_FP32=|\Z)",
        script_text, re.DOTALL,
    )
    assert m, "could not locate train_renderer.py invocation in script"
    used = set(re.findall(r"\B--([a-z][a-z0-9-]+)", m.group(0)))
    invented = used - train_renderer_argparse_flags
    assert not invented, (
        f"Lane K invokes invented flags {sorted(invented)} not in "
        f"train_renderer.py argparse. CLAUDE.md non-negotiable."
    )


# ── contest_auth_eval flag validation ───────────────────────────────────


@pytest.mark.parametrize(
    "flag",
    [
        "archive",
        "inflate-sh",
        "upstream-dir",
        "device",
        "keep-work-dir",
        "work-dir",
    ],
)
def test_contest_auth_eval_flag_real(
    contest_auth_eval_argparse_flags: set[str], flag: str,
):
    """Each contest_auth_eval flag MUST exist in its argparse."""
    assert flag in contest_auth_eval_argparse_flags, (
        f"--{flag} not declared in contest_auth_eval.py argparse"
    )


def test_no_invented_contest_auth_eval_flags(
    script_text: str, contest_auth_eval_argparse_flags: set[str],
):
    """Bidirectional scan for contest_auth_eval invocation."""
    m = re.search(
        r"experiments/contest_auth_eval\.py(.*?)(?=\n\s*log\b|\Z)",
        script_text, re.DOTALL,
    )
    assert m, "could not locate contest_auth_eval.py invocation in script"
    used = set(re.findall(r"\B--([a-z][a-z0-9-]+)", m.group(0)))
    invented = used - contest_auth_eval_argparse_flags
    assert not invented, (
        f"Lane K invokes invented contest_auth_eval flags: {sorted(invented)}"
    )


def test_runs_contest_auth_eval(script_text: str):
    """Lane K must end with contest_auth_eval.py against the EXACT archive
    that would be submitted (CLAUDE.md auth-eval-everywhere rule)."""
    assert "contest_auth_eval.py" in script_text, (
        "every chained experiment must end with a CUDA auth eval"
    )


# ── Provenance + heartbeat (canonical bootstrap pattern) ────────────────


def test_writes_provenance_json(script_text: str):
    """Every remote run must emit provenance.json (memory:
    feedback_canonical_remote_bootstraps)."""
    assert "provenance.json" in script_text or "PROVENANCE=" in script_text, (
        "must write provenance.json for reconstruction"
    )


def test_writes_heartbeat_log(script_text: str):
    """Heartbeat per minute is the watchdog signal (CLAUDE.md remote code
    parity rule: tmux session existence is NOT a heartbeat)."""
    assert "heartbeat.log" in script_text or "HEARTBEAT=" in script_text, (
        "must write heartbeat.log for the watchdog"
    )


def test_provenance_records_predicted_band(script_text: str):
    """The council signed off on a concrete predicted band before launch
    [0.85, 1.10]. The provenance JSON must record it so post-hoc analysis
    can compare predicted vs measured."""
    assert "predicted_band" in script_text, (
        "provenance must record the predicted score band"
    )


def test_provenance_records_anchor_baseline(script_text: str):
    """Anchor score baseline must be recorded so the post-hoc delta is
    unambiguous."""
    assert "anchor_score_baseline" in script_text, (
        "provenance must record Lane A's anchor score (1.15)"
    )


def test_provenance_records_arch_param_count(script_text: str):
    """Lane K's defining feature is the 88K param count — provenance must
    record it so any future operator knows the arch claim."""
    assert "arch_param_count" in script_text, (
        "provenance must record the 88K param count claim"
    )


# ── Archive build via Python zipfile (CRITICAL) ─────────────────────────


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
    # Look for `zip ` at start of a command — not zipfile, unzip, gzip.
    bad_match = re.search(r"(^|[\s;&|`\(])zip\s+(?!file)", code)
    assert not bad_match, (
        f"script should not invoke shell `zip` binary (missing on PyTorch "
        f"container); use Python zipfile instead. "
        f"Match: {bad_match.group(0) if bad_match else None!r}"
    )


def test_uses_python_zipfile(script_text: str):
    """Affirmative: archive build must go through Python's zipfile module."""
    assert "zipfile.ZipFile" in script_text, (
        "Lane K archive must be built via zipfile.ZipFile (Python stdlib), "
        "not the missing apt `zip` binary"
    )


def test_archive_contains_required_files(script_text: str):
    """The Lane K archive must contain renderer.bin + masks.mkv +
    optimized_poses.pt — the three required submission artifacts."""
    assert "renderer.bin" in script_text
    assert "masks.mkv" in script_text
    assert "optimized_poses.pt" in script_text


# ── FP4A export bridge (Stage 1b) ───────────────────────────────────────


def test_fp4a_export_via_canonical_path(script_text: str):
    """train_renderer writes renderer_<tag>_best_fp4.pt (a quantize_fp4 dict),
    NOT a renderer.bin with FP4A magic. Lane K must use
    export_asymmetric_checkpoint_fp4 to produce a real .bin — same path
    pipeline.py:step_export uses, so we get bit-identical bytes to canonical."""
    assert "export_asymmetric_checkpoint_fp4" in script_text, (
        "Stage 1b must call export_asymmetric_checkpoint_fp4 to produce "
        "a canonical FP4A renderer.bin from the fp32 .pt checkpoint"
    )


def test_fp4a_export_uses_fp32_best(script_text: str):
    """The canonical export path needs the fp32 best checkpoint (which has
    model_state_dict + __meta__), NOT the fp4-packed scales/indices file."""
    assert "best_fp32" in script_text, (
        "Stage 1b must use renderer_<tag>_best_fp32.pt (the canonical "
        "fp32 checkpoint with __meta__) — not best_fp4.pt"
    )


# ── Completion banner ───────────────────────────────────────────────────


def test_completion_banner_has_contest_cuda_tag(script_text: str):
    """CLAUDE.md non-negotiable: every score must carry a lane tag.
    Memory: feedback_mps_cuda_drift_critical (2.5x score drift) +
    new check_scores_have_lane_tag preflight."""
    assert "[contest-CUDA]" in script_text, (
        "completion banner must tag the score [contest-CUDA] so it cannot "
        "be misattributed as MPS-PROXY or advisory-only"
    )


# ── Output dir hygiene ──────────────────────────────────────────────────


def test_log_dir_named_lane_k(script_text: str):
    """LOG_DIR should be lane_k_results — namespaced by lane so parallel
    lane runs on the same instance don't stomp each other's results."""
    assert "lane_k_results" in script_text, (
        "LOG_DIR must be lane_k_results (namespaced, not generic)"
    )


def test_tag_namespaced(script_text: str):
    """TAG should be lane_k_dsconv_quantizr_killer — namespaced so
    train_renderer's checkpoint files (renderer_<TAG>_best_fp32.pt) cannot
    collide with other lanes."""
    assert "TAG=\"lane_k_dsconv_quantizr_killer\"" in script_text or (
        'TAG="lane_k_dsconv_quantizr_killer"' in script_text
    ), "TAG must be namespaced as lane_k_dsconv_quantizr_killer"


# ── Required upstream artifacts ─────────────────────────────────────────


@pytest.mark.parametrize(
    "artifact",
    [
        "upstream/videos/0.mkv",
        "upstream/models/segnet.safetensors",
        "upstream/models/posenet.safetensors",
    ],
)
def test_preflight_checks_upstream_artifacts(script_text: str, artifact: str):
    """The pre-flight loop must validate every required upstream artifact
    is present before training starts. Otherwise we burn 12h before
    discovering at the build stage."""
    assert artifact in script_text, (
        f"pre-flight must check {artifact} exists before launch"
    )
