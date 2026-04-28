"""Lane SI: structural tests for scripts/remote_lane_si_saliency_inversion.sh.

The script can't be executed locally (it requires CUDA + a Vast.ai-style
container). But CLAUDE.md mandates several structural properties that
ARE statically checkable. This test file pins those.

Catches:
- Missing `set -euo pipefail` (forbidden silent-skip cascade trap)
- Missing NVDEC probe Stage 0 (feedback_vastai_nvdec_host_variation)
- Missing provenance.json + heartbeat.log (canonical-bootstraps memory)
- Shell `zip` binary instead of python zipfile (zip-dep bootstrap trap)
- `--device cpu` or MPS fallback (mps-cuda-drift-critical)
- Invented CLI flags (dead-flag-wiring trap)
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "scripts" / "remote_lane_si_saliency_inversion.sh"


def test_script_exists() -> None:
    assert SCRIPT.is_file(), f"missing {SCRIPT}"


def test_script_is_executable() -> None:
    import os
    import stat

    mode = os.stat(SCRIPT).st_mode
    assert mode & stat.S_IXUSR, f"{SCRIPT} must be executable (chmod +x)"


def test_set_euo_pipefail_present() -> None:
    """CLAUDE.md FORBIDDEN PATTERN: `set -uo pipefail` (no -e) cascades
    silent failures. zip-dep bootstrap trap cost 6.5h + $2 to learn."""
    src = SCRIPT.read_text()
    assert "set -euo pipefail" in src, (
        "Missing `set -euo pipefail` — silent-skip cascade trap "
        "(see feedback_zip_dep_bootstrap_trap)."
    )


def test_no_set_uo_only() -> None:
    """The exact forbidden pattern is `set -uo pipefail` (no -e). Make
    sure it's not lurking somewhere later in the script."""
    src = SCRIPT.read_text()
    bad_lines = [ln for ln in src.splitlines() if re.search(r"^\s*set\s+-uo\s", ln)]
    assert not bad_lines, f"Forbidden `set -uo`: {bad_lines}"


def test_nvdec_probe_stage_0() -> None:
    """Memory feedback_vastai_nvdec_host_variation: probe NVDEC BEFORE
    spending 30+ min on setup — same 4090 image can lack NVDEC."""
    src = SCRIPT.read_text()
    assert "probe_nvdec.sh" in src, (
        "Missing NVDEC probe — same 4090 image can lack NVDEC and crash "
        "upstream/evaluate.py at the end of a 6h run "
        "(feedback_vastai_nvdec_host_variation)."
    )


def test_provenance_json_written() -> None:
    """canonical-remote-bootstraps memory: every remote run must emit
    provenance.json so a fresh agent can reconstruct the experiment."""
    src = SCRIPT.read_text()
    assert "provenance.json" in src or "PROVENANCE=" in src
    assert "git_hash" in src or "GIT_HASH" in src
    assert "gpu_name" in src.lower() or "GPU_NAME" in src


def test_heartbeat_log_written() -> None:
    src = SCRIPT.read_text()
    assert "heartbeat" in src.lower(), (
        "Missing heartbeat — feedback_canonical_remote_bootstraps requires "
        "a heartbeat.log so the watchdog can detect a dead session."
    )


def test_python_zipfile_not_shell_zip() -> None:
    """CLAUDE.md FORBIDDEN: shell `zip` binary (PyTorch container has no zip).
    Use python zipfile.ZipFile instead."""
    src = SCRIPT.read_text()
    # Allow `unzip` (read-only is fine, we don't ship `unzip`-dependent
    # archive creation). Disallow `zip` (the binary that creates archives).
    bad = re.findall(r"^\s*zip\s+", src, re.MULTILINE)
    assert not bad, (
        f"Forbidden shell `zip` binary: {bad}. PyTorch container has no "
        "zip; use python zipfile.ZipFile (see feedback_zip_dep_bootstrap_trap)."
    )
    # Confirm we DO use python zipfile
    assert "zipfile.ZipFile" in src, (
        "Archive build must use python zipfile.ZipFile, not shell `zip`."
    )


def test_no_cpu_or_mps_device() -> None:
    """CLAUDE.md FORBIDDEN: `--device cpu` or `--device mps` in any
    saliency / auth eval invocation. CUDA required."""
    src = SCRIPT.read_text()
    # Look for `--device cpu` / `--device mps` literals
    assert "--device cpu" not in src, "FORBIDDEN: --device cpu in Lane SI script"
    assert "--device mps" not in src, "FORBIDDEN: --device mps in Lane SI script"
    # Confirm --device cuda IS used somewhere
    assert "--device cuda" in src, "Must explicitly pass --device cuda"


def test_stage_4_runs_contest_auth_eval() -> None:
    """Auth-eval-EVERYWHERE rule: Stage 4 must end with contest_auth_eval.py."""
    src = SCRIPT.read_text()
    assert "contest_auth_eval.py" in src, (
        "Missing contest_auth_eval.py — every chained experiment MUST end "
        "with a CUDA auth eval (CLAUDE.md non-negotiable)."
    )


def test_predicted_band_recorded() -> None:
    """Per the spec, the script's provenance must include
    predicted_band so a fresh agent knows the hypothesis bounds."""
    src = SCRIPT.read_text()
    assert "predicted_band" in src or "predicted band" in src.lower(), (
        "predicted_band must appear in the script (provenance hypothesis)"
    )


def test_invokes_profile_scorer_saliency_with_real_flags() -> None:
    """Mirror of test_profile_scorer_saliency.test_script_remote_bootstrap_uses_only_real_flags
    but checked at the script level too — the dead-flag trap is the most
    expensive bug class in this codebase."""
    src = SCRIPT.read_text()
    assert "profile_scorer_saliency.py" in src
    profile_block = re.search(r"profile_scorer_saliency\.py(.+?)2>&1", src, re.DOTALL)
    assert profile_block is not None
    flags = set(re.findall(r"--[\w-]+", profile_block.group(1)))
    # Lane SI passes these (all must exist in profile_scorer_saliency.py argparse)
    expected = {
        "--checkpoint", "--poses", "--masks-mkv", "--video", "--output",
        "--device", "--upstream-dir", "--n-pairs", "--reduce",
    }
    missing = expected - flags
    assert not missing, f"Lane SI script doesn't pass expected flags: {missing}"


def test_invokes_contest_auth_eval_with_real_flags() -> None:
    """contest_auth_eval.py only accepts the flags listed in its argparse.
    Verify the Lane SI invocation uses only real flags."""
    src = SCRIPT.read_text()
    eval_block = re.search(r"contest_auth_eval\.py(.+?)2>&1", src, re.DOTALL)
    assert eval_block is not None
    flags = set(re.findall(r"--[\w-]+", eval_block.group(1)))
    # contest_auth_eval.py argparse (verified 2026-04-27 via grep):
    real_flags = {
        "--archive", "--inflate-sh", "--upstream-dir", "--video-names-file",
        "--device", "--work-dir", "--inflate-timeout", "--evaluate-timeout",
        "--keep-work-dir",
    }
    invented = flags - real_flags
    assert not invented, (
        f"Lane SI invokes contest_auth_eval.py with invented flags: {invented}. "
        "This is the dead-flag-wiring trap CLAUDE.md forbids."
    )


def test_documents_outstanding_inflate_decoder_todo() -> None:
    """Component 4 (inflate-time saliency-aware decoder) is deferred.
    The script header must say so explicitly, otherwise an operator
    might think the SLI1 payload is load-bearing in the auth eval."""
    src = SCRIPT.read_text()
    has_todo = (
        "OUTSTANDING TODO" in src
        or "Component 4" in src
        or "inflate-time decoder" in src.lower()
    )
    assert has_todo, (
        "Script must document that the SLI1 payload is a research artifact "
        "(no inflate-time decoder yet); without it, operators may think "
        "the saliency-weighted masks are scoring the auth eval."
    )


def test_required_artifact_preflight_present() -> None:
    """Pre-flight: the script must check for the Lane A baseline files +
    upstream scorers BEFORE any GPU work."""
    src = SCRIPT.read_text()
    assert "renderer.bin" in src
    assert "optimized_poses.pt" in src
    assert "posenet.safetensors" in src
    assert "segnet.safetensors" in src
