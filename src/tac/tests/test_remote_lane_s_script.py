"""Regression tests for scripts/remote_lane_s_self_compress.sh.

Lane S = Self-Compressing renderer (Szabolcs 2301.13142) anchored on Lane A
(1.15 [contest-CUDA]). Predicted band [0.85, 1.20].

These tests pin every claim the launch script makes:

  1. Strict bash safety — `set -euo pipefail` (the cascade trap that ate
     LANE-B 6.5h + $2 in 2026-04-26 must not reappear).
  2. Stage 0 NVDEC probe BEFORE any GPU spend (memory:
     feedback_vastai_nvdec_host_variation — same 4090 image, different
     hosts, same driver, different NVDEC outcome).
  3. Anchor on Lane A (the 1.15 [contest-CUDA] frontier — NOT baseline 2.29
     and NOT the phantom 0.9001 archive). This test asserts the script
     references experiments/results/lane_a_landed/.
  4. Lane S self-compress CLI flags — every --self-compress-* flag MUST
     exist in train_renderer.py's argparse. CLAUDE.md forbids inventing
     CLI flags (memory: feedback_dead_flag_wiring_pattern).
  5. Provenance + heartbeat writes — required by
     feedback_canonical_remote_bootstraps so a fresh agent can reconstruct
     the experiment from disk.
  6. Predicted band metadata — the council must have signed off on a
     concrete band before the run, not a vague "should be better".
  7. Python zipfile (NOT shell `zip`) — PyTorch container has no `zip`
     binary (memory: feedback_zip_dep_bootstrap_trap).
  8. [contest-CUDA] tag in the completion log — every score must carry a
     lane tag (CLAUDE.md non-negotiable).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_s_self_compress.sh"
TRAIN_RENDERER = REPO / "src" / "tac" / "experiments" / "train_renderer.py"


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


# ── Existence + bash-safety guards ──────────────────────────────────────


def test_script_exists():
    assert SCRIPT.exists(), f"missing Lane S launch script: {SCRIPT}"


def test_script_is_executable():
    """Bash scripts in scripts/ should be chmod +x so the parent agent can
    invoke directly without `bash` prefix."""
    import os
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be executable"


def test_set_e_present(script_text: str):
    """The `set -uo pipefail` trap (no -e) silently cascades failures.
    LANE-B 2026-04-26 burned 6.5h + $2 because of this exact pattern.
    The Lane S script MUST use `set -e` to abort on the first failure."""
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
    # Look for a fatal-exit pattern after the probe call.
    probe_section = re.search(
        r"probe_nvdec\.sh.*?exit\s+\d+", script_text, re.DOTALL,
    )
    assert probe_section is not None, (
        "NVDEC probe must abort with `exit N` on failure"
    )


# ── Anchor verification ─────────────────────────────────────────────────


def test_anchors_on_lane_a(script_text: str):
    """Lane S is a fine-tune of Lane A's renderer — NOT a from-scratch
    train and NOT anchored on phantom 0.9001 (memory:
    project_baseline_0_9001_lost_archive_test). The script must
    explicitly reference experiments/results/lane_a_landed/."""
    assert "experiments/results/lane_a_landed" in script_text, (
        "Lane S must anchor on Lane A's verified 1.15 [contest-CUDA] "
        "artifacts at experiments/results/lane_a_landed/"
    )


def test_anchors_on_lane_a_renderer_bin(script_text: str):
    """Specifically the renderer.bin (the warm-start weights for SC swap)."""
    assert (
        "experiments/results/lane_a_landed/iter_0/renderer.bin" in script_text
    ), "must reference Lane A's renderer.bin specifically (not a directory glob)"


def test_anchors_on_lane_a_poses_and_masks(script_text: str):
    """Lane A's optimized_poses.pt + masks.mkv go into the Lane S archive
    unchanged (we are only attacking the renderer rate, not the pose or
    mask payload)."""
    assert (
        "experiments/results/lane_a_landed/iter_0/optimized_poses.pt"
        in script_text
    )
    assert "experiments/results/lane_a_landed/iter_0/masks.mkv" in script_text


# ── Self-compress CLI flag wiring (CLAUDE.md non-negotiable) ────────────


@pytest.mark.parametrize(
    "flag",
    [
        "use-self-compress-codec",
        "self-compress-init-bits",
        "self-compress-target-bits",
        "self-compress-lambda-start",
        "self-compress-lambda-end",
        "self-compress-lambda-ramp-start-frac",
    ],
)
def test_sc_flag_used_in_script(script_text: str, flag: str):
    """Every Lane S CLI flag MUST appear in the train_renderer invocation."""
    assert f"--{flag}" in script_text, (
        f"Lane S launch script must pass --{flag} to train_renderer.py"
    )


@pytest.mark.parametrize(
    "flag",
    [
        "use-self-compress-codec",
        "self-compress-init-bits",
        "self-compress-target-bits",
        "self-compress-lambda-start",
        "self-compress-lambda-end",
        "self-compress-lambda-ramp-start-frac",
    ],
)
def test_sc_flag_real_in_argparse(
    train_renderer_argparse_flags: set[str], flag: str,
):
    """Each Lane S flag MUST exist in train_renderer.py's argparse.
    CLAUDE.md non-negotiable: NEVER invent CLI flags. Memory:
    feedback_dead_flag_wiring_pattern (2026-04-26 dead flag burn)."""
    assert flag in train_renderer_argparse_flags, (
        f"--{flag} not declared in train_renderer.py argparse — "
        f"CLAUDE.md forbids inventing CLI flags. Verify with "
        f"`grep add_argument src/tac/experiments/train_renderer.py`."
    )


def test_resume_from_flag_used(script_text: str):
    """Lane S WARM-STARTS from Lane A's weights — not a from-scratch
    train. The --resume-from flag is how train_renderer ingests that."""
    assert "--resume-from" in script_text, (
        "Lane S must --resume-from a SC-init checkpoint warm-started "
        "from Lane A's weights"
    )


def test_no_auth_eval_on_best_disabled(script_text: str):
    """SCv1 export needs a separate Python step (Stage 3). The built-in
    auth_eval_on_best path uses FP4A export which is incompatible with
    SC mode (train_renderer auto-disables it for SC anyway, but we
    document intent by passing the flag explicitly)."""
    assert "--no-auth-eval-on-best" in script_text, (
        "SC training must disable auth-eval-on-best (FP4A path "
        "incompatible with SC weights)"
    )


def test_device_cuda_required(script_text: str):
    """No MPS or CPU fallback — CLAUDE.md MPS-CUDA drift is 23x on
    PoseNet. Score must be measured on CUDA."""
    assert "--device cuda" in script_text, "Lane S must use --device cuda"
    assert "--device mps" not in script_text, "MPS forbidden — drift 23x on PoseNet"
    # CPU is allowed only for `--device cpu` deterministic-bytes builds; the
    # Lane S compute path is GPU-only, so cpu device must not appear.
    assert "--device cpu" not in script_text, (
        "Lane S compute path is GPU-only; --device cpu would silently "
        "produce invalid scores"
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
    [0.85, 1.20]. The provenance JSON must record it so post-hoc analysis
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


# ── Archive build via Python zipfile (PyTorch container has no `zip`) ───


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
    # Look for `zip ` at start of a command — not zipfile, unzip, gzip, --zip.
    bad_match = re.search(r"(^|[\s;&|`\(])zip\s+(?!file)", code)
    assert not bad_match, (
        f"script should not invoke shell `zip` binary (missing on PyTorch "
        f"container); use Python zipfile instead. "
        f"Match: {bad_match.group(0) if bad_match else None!r}"
    )


def test_uses_python_zipfile(script_text: str):
    """Affirmative: archive build must go through Python's zipfile module."""
    assert "zipfile.ZipFile" in script_text, (
        "Lane S archive must be built via zipfile.ZipFile (Python stdlib), "
        "not the missing apt `zip` binary"
    )


def test_archive_contains_required_files(script_text: str):
    """The Lane S archive must contain renderer.bin + masks.mkv +
    optimized_poses.pt — the three Lane A artifacts (renderer swapped
    for the SCv1 binary)."""
    assert "renderer.bin" in script_text
    assert "masks.mkv" in script_text
    assert "optimized_poses.pt" in script_text


# ── Auth eval on the actual archive ─────────────────────────────────────


def test_runs_contest_auth_eval(script_text: str):
    """Lane S must end with contest_auth_eval.py against the EXACT archive
    that would be submitted (CLAUDE.md auth-eval-everywhere rule)."""
    assert "contest_auth_eval.py" in script_text, (
        "every chained experiment must end with a CUDA auth eval"
    )


def test_auth_eval_uses_built_archive(script_text: str):
    """auth-eval must run on the archive_lane_s.zip we just built, not
    a different archive (CLAUDE.md: 'the EXACT archive that will be
    submitted')."""
    assert "archive_lane_s.zip" in script_text or "$ARCHIVE" in script_text, (
        "auth eval must operate on the Lane S archive we constructed, "
        "not on some other archive"
    )


def test_auth_eval_result_validated(script_text: str):
    """Detect auth_eval crashes — guard against silent zero-exit on a
    crashed eval (LANE-B-style cascade). Look for a RESULT_JSON grep gate
    after the eval call."""
    assert "RESULT_JSON" in script_text, (
        "must validate RESULT_JSON line in auth_eval.log to catch silent "
        "auth-eval crashes (LANE-B 2026-04-26 cascade pattern)"
    )


# ── Lane tag on the completion log [contest-CUDA] ───────────────────────


def test_completion_log_tags_contest_cuda(script_text: str):
    """CLAUDE.md non-negotiable: every score reported must carry a lane tag
    (`[contest-CUDA]`, `[advisory only]`, `[MPS-PROXY]`, ...). Lane S goes
    through inflate.sh + upstream/evaluate.py on CUDA, so the completion
    log must tag the result [contest-CUDA]."""
    # Look for the LANE_S_DONE marker with a [contest-CUDA] tag.
    assert "[contest-CUDA]" in script_text, (
        "completion log must tag the Lane S result [contest-CUDA] "
        "(CLAUDE.md non-negotiable: every score needs a lane tag)"
    )


# ── Anti-MPS guard (no fallback) ────────────────────────────────────────


def test_no_mps_fallback(script_text: str):
    """CLAUDE.md FORBIDDEN: no MPS fallback. PoseNet drift on MPS is 23x.
    Lane S is a remote-CUDA-only script.

    We look for `mps` ONLY in device-selection contexts (not as a substring
    of `json.dumps`, `nvidia-smi`, etc.):
      * `--device mps`
      * `device=...mps...`  (e.g., `device='mps'` / `device="mps"`)
      * `DEVICE=mps`        (env-var assignment)
      * `if .*mps`          (conditional fallback selection)
    """
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
            f"Lane S code path must not reference MPS in device selection — "
            f"CLAUDE.md drift = 23x. Match for /{pat}/: {m.group(0) if m else None!r}"
        )


# ── Phase epoch overrides land where we expect ──────────────────────────


def test_500_epoch_short_finetune(script_text: str):
    """Lane S overrides the SC profile's 1980 ep schedule to 500 ep total
    (warm-started from Lane A). The phase override flags must sum to 500."""
    # Extract phaseN-epochs values from the script
    matches = re.findall(r"--phase(\d)-epochs\s+(\d+)", script_text)
    assert matches, "no --phase[1-5]-epochs overrides found"
    by_phase: dict[int, int] = {}
    for phase, val in matches:
        by_phase[int(phase)] = int(val)
    total = sum(by_phase.values())
    assert total == 500, (
        f"phase epoch overrides must sum to 500 (got {total}: {by_phase}). "
        f"Lane S is a fine-tune-from-Lane-A; 1980 ep would be train-from-scratch."
    )


def test_low_lr_preserves_lane_a(script_text: str):
    """Lane A's quality must be preserved during the SC fine-tune. The LR
    must be low (5e-5 across all phases) so we don't undo Lane A's
    distortion-floor work."""
    matches = re.findall(r"--phase(\d)-lr\s+(\S+)", script_text)
    assert matches, "no --phase[1-5]-lr overrides found"
    for phase, lr_str in matches:
        lr = float(lr_str)
        assert lr <= 1e-4, (
            f"phase{phase} LR={lr_str} too high — Lane S fine-tune "
            f"must use LR ≤ 1e-4 to preserve Lane A's distortion floor"
        )
