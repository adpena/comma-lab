# SPDX-License-Identifier: MIT
"""Regression tests for scripts/remote_lane_i_coolchic_masks.sh.

Lane I = Cool-Chic renderer (CCh1 magic) anchored on Lane A. Predicted
band [0.95, 1.30] [contest-CUDA].

These tests pin every claim the launch script makes:

  1. Strict bash safety — `set -euo pipefail` (the cascade trap that ate
     LANE-B 6.5h + $2 in 2026-04-26 must not reappear).
  2. Stage 0 NVDEC probe BEFORE any GPU spend (memory:
     feedback_vastai_nvdec_host_variation).
  3. Anchor on Lane A's masks + poses (the 1.15 [contest-CUDA] frontier).
  4. Cool-Chic CLI flags — every flag MUST exist in train_renderer.py's
     argparse. CLAUDE.md forbids inventing CLI flags (memory:
     feedback_dead_flag_wiring_pattern).
  5. coolchic_renderer_full profile registration in profiles.py.
  6. Provenance + heartbeat writes — required by
     feedback_canonical_remote_bootstraps.
  7. Predicted band metadata.
  8. Python zipfile (NOT shell `zip`) — feedback_zip_dep_bootstrap_trap.
  9. [contest-CUDA] tag in the completion log.
 10. CCh1 export step (export_coolchic_renderer) in Stage 3.
 11. Honest documentation of the operator-framing reinterpretation
     (Lane I is renderer-replacement, not mask-codec).
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_i_coolchic_masks.sh"
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
    assert SCRIPT.exists(), f"missing Lane I launch script: {SCRIPT}"


def test_script_is_executable():
    """Bash scripts in scripts/ should be chmod +x so the parent agent can
    invoke directly without `bash` prefix."""
    import os
    assert os.access(SCRIPT, os.X_OK), f"{SCRIPT} should be executable"


def test_set_e_present(script_text: str):
    """The `set -uo pipefail` trap (no -e) silently cascades failures.
    LANE-B 2026-04-26 burned 6.5h + $2 because of this exact pattern.
    The Lane I script MUST use `set -e` to abort on the first failure."""
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


# ── Anchor verification ─────────────────────────────────────────────────


def test_anchors_on_lane_a(script_text: str):
    """Lane I reuses Lane A's masks + poses. Renderer is the only thing
    being replaced (Cool-Chic for ASYM)."""
    assert "experiments/results/lane_a_landed" in script_text, (
        "Lane I must anchor on Lane A's verified 1.15 [contest-CUDA] "
        "artifacts at experiments/results/lane_a_landed/"
    )


def test_anchors_on_lane_a_renderer_bin(script_text: str):
    """Specifically the renderer.bin (recorded as anchor for documentation
    even though Lane I REPLACES it with CCh1 — the file is referenced for
    the size comparison and provenance)."""
    assert (
        "experiments/results/lane_a_landed/iter_0/renderer.bin" in script_text
    ), "must reference Lane A's renderer.bin specifically"


def test_anchors_on_lane_a_poses_and_masks(script_text: str):
    """Lane A's optimized_poses.pt + masks.mkv go into the Lane I archive
    unchanged."""
    assert (
        "experiments/results/lane_a_landed/iter_0/optimized_poses.pt"
        in script_text
    )
    assert "experiments/results/lane_a_landed/iter_0/masks.mkv" in script_text


# ── Cool-Chic CLI flag wiring (CLAUDE.md non-negotiable) ────────────────


@pytest.mark.parametrize(
    "flag",
    [
        "profile",
        "variant",
        "tag",
        "device",
        "video",
        "output-dir",
        "epochs",
        "no-auth-eval-on-best",
    ],
)
def test_lane_i_flag_used_in_script(script_text: str, flag: str):
    """Every Lane I CLI flag MUST appear in the train_renderer invocation."""
    assert f"--{flag}" in script_text, (
        f"Lane I launch script must pass --{flag} to train_renderer.py"
    )


@pytest.mark.parametrize(
    "flag",
    [
        "profile",
        "variant",
        "tag",
        "device",
        "video",
        "output-dir",
        "epochs",
        "no-auth-eval-on-best",
    ],
)
def test_lane_i_flag_real_in_argparse(
    train_renderer_argparse_flags: set[str], flag: str,
):
    """Each Lane I flag MUST exist in train_renderer.py's argparse.
    CLAUDE.md non-negotiable: NEVER invent CLI flags. Memory:
    feedback_dead_flag_wiring_pattern (2026-04-26 dead flag burn).

    Note: --no-auth-eval-on-best is registered as the dest='auth_eval_on_best'
    inverse — argparse stores the dest, but the flag name in the source is
    'no-auth-eval-on-best' (kebab-case from `--no-auth-eval-on-best`).
    """
    assert flag in train_renderer_argparse_flags, (
        f"--{flag} not declared in train_renderer.py argparse — "
        f"CLAUDE.md forbids inventing CLI flags. Verify with "
        f"`grep add_argument src/tac/experiments/train_renderer.py`."
    )


def test_uses_coolchic_renderer_variant(script_text: str):
    """Lane I trains the coolchic_renderer variant (CCh1 magic) — verify
    the variant name appears in the invocation, not by accident."""
    assert "--variant coolchic_renderer" in script_text, (
        "Lane I must explicitly pass --variant coolchic_renderer to "
        "guarantee the CCh1 dispatch (not the default ASYM/dilated path)."
    )


def test_uses_coolchic_profile(script_text: str):
    """Lane I uses the coolchic_renderer_full profile."""
    assert "--profile coolchic_renderer_full" in script_text


def test_no_auth_eval_on_best_disabled(script_text: str):
    """CCh1 export needs a separate Python step (Stage 3). The built-in
    auth_eval_on_best path uses FP4A export which is incompatible with
    CCh1's CoolChicLatentRenderer state_dict layout."""
    assert "--no-auth-eval-on-best" in script_text, (
        "CCh1 training must disable auth-eval-on-best (FP4A path "
        "incompatible with Cool-Chic state_dict)"
    )


def test_device_cuda_required(script_text: str):
    """No MPS or CPU fallback — CLAUDE.md MPS-CUDA drift is 23x on PoseNet."""
    assert "--device cuda" in script_text, "Lane I must use --device cuda"
    assert "--device mps" not in script_text, "MPS forbidden — drift 23x on PoseNet"
    assert "--device cpu" not in script_text, (
        "Lane I compute path is GPU-only; --device cpu would silently "
        "produce invalid scores"
    )


# ── Profile validation ──────────────────────────────────────────────────


def test_coolchic_profile_registered():
    """coolchic_renderer_full profile MUST be registered in profiles.py.
    The script does its own pre-flight check, but we double-pin it here
    so a profile rename breaks the test before burning GPU time."""
    import sys
    # Resolve import the way the script does
    sys.path.insert(0, str(REPO / "src"))
    try:
        from tac.profiles import PROFILES
        assert "coolchic_renderer_full" in PROFILES
        p = PROFILES["coolchic_renderer_full"]
        assert p["variant"] == "coolchic_renderer"
        for k in ("latent_ch", "latent_shapes", "embed_dim", "epochs", "lr"):
            assert k in p, f"profile missing key: {k}"
    finally:
        sys.path.remove(str(REPO / "src"))


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
    [0.95, 1.30]. The provenance JSON must record it."""
    assert "predicted_band" in script_text, (
        "provenance must record the predicted score band"
    )
    # Pin the actual band values too — guards against silent drift if a
    # future edit tries to bump the band without council sign-off.
    assert "0.95" in script_text and "1.30" in script_text, (
        "predicted_band must be [0.95, 1.30] — council band for Lane I"
    )


def test_provenance_records_anchor_baseline(script_text: str):
    """Anchor score baseline must be recorded so the post-hoc delta is
    unambiguous."""
    assert "anchor_score_baseline" in script_text, (
        "provenance must record Lane A's anchor score (1.15)"
    )
    assert "1.15" in script_text, "anchor baseline must be Lane A's 1.15"


def test_provenance_documents_reinterpretation(script_text: str):
    """The operator's framing was 'Cool-Chic on the MASK SEQUENCE' but
    the existing CCh1 infra is renderer-replacement. The provenance must
    document this reinterpretation so a future reviewer knows."""
    assert "note_re_operator_framing" in script_text or "renderer-replacement" in script_text or "mask codec" in script_text, (
        "provenance must document the renderer-replacement vs mask-codec "
        "reinterpretation so the operator can audit the choice"
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
    bad_match = re.search(r"(^|[\s;&|`\(])zip\s+(?!file)", code)
    assert not bad_match, (
        f"script should not invoke shell `zip` binary (missing on PyTorch "
        f"container); use Python zipfile instead. "
        f"Match: {bad_match.group(0) if bad_match else None!r}"
    )


def test_uses_python_zipfile(script_text: str):
    """Affirmative: archive build must go through Python's zipfile module."""
    assert "zipfile.ZipFile" in script_text, (
        "Lane I archive must be built via zipfile.ZipFile (Python stdlib), "
        "not the missing apt `zip` binary"
    )


def test_archive_contains_required_files(script_text: str):
    """The Lane I archive must contain renderer.bin (CCh1) + masks.mkv +
    optimized_poses.pt — Lane A's three artifacts with renderer swapped
    for the CCh1 binary."""
    assert "renderer.bin" in script_text
    assert "masks.mkv" in script_text
    assert "optimized_poses.pt" in script_text


# ── CCh1 export step ────────────────────────────────────────────────────


def test_calls_export_coolchic_renderer(script_text: str):
    """Stage 3 must run export_coolchic_renderer (the CCh1 serializer).
    Without this, the trained Cool-Chic state_dict cannot be packed into
    a CCh1-magic .bin that submissions/robust_current/inflate_renderer.py
    can dispatch."""
    assert "export_coolchic_renderer" in script_text, (
        "Stage 3 must call export_coolchic_renderer to produce the CCh1 binary"
    )


def test_calls_build_coolchic_renderer(script_text: str):
    """Stage 3 rebuilds the same Cool-Chic arch the training run used."""
    assert "build_coolchic_renderer" in script_text, (
        "Stage 3 must rebuild the model via build_coolchic_renderer"
    )


def test_stage_3_writes_cch1_bin(script_text: str):
    """Stage 3 produces a renderer_coolchic.bin file (the CCh1 binary)."""
    assert "renderer_coolchic.bin" in script_text, (
        "Stage 3 must write renderer_coolchic.bin"
    )


# ── Auth eval on the actual archive ─────────────────────────────────────


def test_runs_contest_auth_eval(script_text: str):
    """Lane I must end with contest_auth_eval.py against the EXACT archive
    that would be submitted (CLAUDE.md auth-eval-everywhere rule)."""
    assert "contest_auth_eval.py" in script_text, (
        "every chained experiment must end with a CUDA auth eval"
    )


def test_auth_eval_uses_built_archive(script_text: str):
    """auth-eval must run on the archive_lane_i.zip we just built."""
    assert "archive_lane_i.zip" in script_text or "$ARCHIVE" in script_text, (
        "auth eval must operate on the Lane I archive we constructed"
    )


def test_auth_eval_result_validated(script_text: str):
    """Detect auth_eval crashes — guard against silent zero-exit on a
    crashed eval (LANE-B-style cascade)."""
    assert "RESULT_JSON" in script_text, (
        "must validate RESULT_JSON line in auth_eval.log to catch silent "
        "auth-eval crashes (LANE-B 2026-04-26 cascade pattern)"
    )


# ── Lane tag on the completion log [contest-CUDA] ───────────────────────


def test_completion_log_tags_contest_cuda(script_text: str):
    """CLAUDE.md non-negotiable: every score reported must carry a lane
    tag. Lane I goes through inflate.sh + upstream/evaluate.py on CUDA,
    so the completion log must tag the result [contest-CUDA]."""
    assert "[contest-CUDA]" in script_text, (
        "completion log must tag the Lane I result [contest-CUDA]"
    )


# ── Anti-MPS guard (no fallback) ────────────────────────────────────────


def test_no_mps_fallback(script_text: str):
    """CLAUDE.md FORBIDDEN: no MPS fallback. PoseNet drift on MPS is 23x.

    We look for `mps` ONLY in device-selection contexts:
      * `--device mps`
      * `device=...mps...`
      * `DEVICE=mps`
      * `if .*mps`
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
            f"Lane I code path must not reference MPS in device selection — "
            f"CLAUDE.md drift = 23x. Match for /{pat}/: {m.group(0) if m else None!r}"
        )


# ── Epoch cap sanity ────────────────────────────────────────────────────


def test_epoch_cap_present(script_text: str):
    """Lane I caps epochs at 1000 (overrides profile default 2500) for
    first lane shake-out. The cap must be explicit so the parent agent
    knows the budget envelope."""
    assert "--epochs 1000" in script_text, (
        "Lane I must explicitly cap --epochs at 1000 for the first run"
    )
