# SPDX-License-Identifier: MIT
"""Tests for codex round-3 MEDIUM 1 fix — local NVDEC probe is opt-in.

Bug class (codex round-3 MEDIUM 1, 2026-05-09): the
``scripts/remote_lane_avvideodataset_cuda_path_mechanism_discriminator.sh``
runbook unconditionally invoked ``probe_nvdec.sh`` LOCALLY whenever
``SKIP_CUDA=0`` (the default), including under ``DRY_RUN=1``. The
operator workstation / CI host typically has no CUDA / NVDEC / DALI
installed, so the probe exited 2 before the script could reach the
dispatch-decision block.

The fix: guard local NVDEC probing behind explicit
``LOCAL_CUDA_WORKER=1`` env. ``DRY_RUN=1`` always skips the probe.
``LOCAL_CUDA_WORKER!=1`` (the default) skips the probe; the probe
belongs in the remote provider bootstrap where the GPU actually lives.

These tests pin the new guard contract by inspecting the script's source
text (the canonical pattern used by sister tests like
``test_verify_vast_setup_stuck.py`` for parser-flag wiring).

Memory: feedback_codex_round3_findings_fix_landed_20260509.md.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = (
    REPO_ROOT
    / "scripts"
    / "remote_lane_avvideodataset_cuda_path_mechanism_discriminator.sh"
)


# ── Source-text guard contract ───────────────────────────────────────────


def test_script_has_local_cuda_worker_guard_token():
    """The script must reference `LOCAL_CUDA_WORKER` as the explicit
    opt-in for local probing.

    Pre-fix: there was no such guard; the probe always ran when
    `SKIP_CUDA != 1`.
    """
    src = SCRIPT_PATH.read_text()
    assert "LOCAL_CUDA_WORKER" in src, (
        "MEDIUM 1: script must declare a `LOCAL_CUDA_WORKER` env "
        "guard so local NVDEC probing is opt-in. Without it, the probe "
        "fires on the dispatcher host (typically no CUDA/NVDEC/DALI) "
        "and the dispatch never starts."
    )


def test_script_has_local_cuda_worker_default_zero():
    """The default for `LOCAL_CUDA_WORKER` must be 0 (skip-by-default).

    A default of 1 would re-introduce the original bug.
    """
    src = SCRIPT_PATH.read_text()
    assert 'LOCAL_CUDA_WORKER="${LOCAL_CUDA_WORKER:-0}"' in src, (
        "MEDIUM 1: `LOCAL_CUDA_WORKER` must default to 0 (skip-by-"
        "default). Otherwise the dispatch fires `probe_nvdec.sh` on "
        "the dispatcher host."
    )


def test_script_probe_block_checks_dry_run_and_local_cuda_worker():
    """The probe-running branch must require BOTH `DRY_RUN != 1` AND
    `LOCAL_CUDA_WORKER = 1`.

    A bug where DRY_RUN is checked but LOCAL_CUDA_WORKER is not (or
    vice versa) re-opens the failure mode.
    """
    src = SCRIPT_PATH.read_text()
    # Both tokens must appear in the same `if` block guarding the probe.
    # Find the probe invocation site.
    probe_idx = src.find("bash \"${REPO_ROOT}/scripts/probe_nvdec.sh\"")
    assert probe_idx != -1, "probe_nvdec.sh invocation must exist"
    # The 800 chars before it must include both guard tokens.
    preamble = src[max(0, probe_idx - 800):probe_idx]
    assert 'DRY_RUN' in preamble, (
        "MEDIUM 1: the probe-running branch must check DRY_RUN."
    )
    assert 'LOCAL_CUDA_WORKER' in preamble, (
        "MEDIUM 1: the probe-running branch must check LOCAL_CUDA_WORKER."
    )


def test_script_skip_path_emits_explanatory_message():
    """When the probe is skipped, the script must tell the operator WHY
    so they can flip `LOCAL_CUDA_WORKER=1` if they really do want local
    probing.
    """
    src = SCRIPT_PATH.read_text()
    assert "NVDEC probe SKIPPED" in src, (
        "MEDIUM 1: the skip path must print `NVDEC probe SKIPPED` "
        "with the env-flag values so the operator understands why."
    )
    assert "LOCAL_CUDA_WORKER=1" in src, (
        "MEDIUM 1: the skip-path message must hint that "
        "`LOCAL_CUDA_WORKER=1` is the opt-in flag."
    )


# ── Behavioural test: dry-run executes without invoking probe ────────────


@pytest.mark.skipif(
    not SCRIPT_PATH.exists(),
    reason="script must exist for behavioural test",
)
def test_probe_skipped_when_dry_run_set(tmp_path, monkeypatch):
    """End-to-end: DRY_RUN=1 + SKIP_CUDA=0 must NOT execute probe_nvdec.sh.

    We assert this by intercepting the probe via a PATH shim and
    verifying it was never called. The script aborts before the
    dispatch-decision block at this point because no
    `--skip-cpu` work happens for an unbuilt variant — but the
    probe-skip behaviour is observable.
    """
    # Prepare a sandbox where probe_nvdec.sh is intercepted.
    sandbox_repo = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "scripts", sandbox_repo / "scripts")
    # Replace probe_nvdec.sh with a shim that signals if it was called.
    probe_marker = tmp_path / "probe_was_called.marker"
    probe_path = sandbox_repo / "scripts" / "probe_nvdec.sh"
    probe_path.write_text(
        "#!/usr/bin/env bash\n"
        f"touch {probe_marker}\n"
        "echo PROBE_INVOKED\n"
        "exit 0\n"
    )
    probe_path.chmod(0o755)
    # Stand up the minimum REPO_ROOT siblings the script expects.
    (sandbox_repo / "upstream" / "videos").mkdir(parents=True)
    (sandbox_repo / ".venv" / "bin").mkdir(parents=True)
    venv_python = sandbox_repo / ".venv" / "bin" / "python"
    venv_python.write_text("#!/usr/bin/env bash\necho mock-python\n")
    venv_python.chmod(0o755)
    (sandbox_repo / "tools").mkdir(parents=True)
    fake_claim = sandbox_repo / "tools" / "claim_lane_dispatch.py"
    fake_claim.write_text("#!/usr/bin/env python3\nimport sys; sys.exit(0)\n")
    # Fake `git rev-parse` for provenance line.
    env = os.environ.copy()
    env["DRY_RUN"] = "1"
    env["SKIP_CUDA"] = "0"
    env["SKIP_CPU"] = "1"
    env["LOCAL_CUDA_WORKER"] = "0"
    env["DISCRIMINATOR_TIMESTAMP_SUFFIX"] = "20260509T000000Z"
    # Invoke the original script (not the sandbox copy) but with the
    # sandbox script siblings staged. The script computes REPO_ROOT
    # as the parent of itself, so we run a sandbox copy of the
    # discriminator script too.
    discriminator_script = (
        sandbox_repo / "scripts"
        / "remote_lane_avvideodataset_cuda_path_mechanism_discriminator.sh"
    )
    # If somehow the sandbox copy doesn't exist, the assertion is moot.
    if not discriminator_script.exists():
        pytest.skip("sandbox copy of discriminator script missing")
    result = subprocess.run(
        ["bash", str(discriminator_script)],
        env=env,
        cwd=str(sandbox_repo),
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Whether the script eventually exits 0/1/2 is irrelevant — what
    # matters is that probe_nvdec.sh was NOT invoked.
    assert not probe_marker.exists(), (
        "MEDIUM 1: probe_nvdec.sh was invoked under DRY_RUN=1 + "
        f"LOCAL_CUDA_WORKER=0. Script stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.skipif(
    not SCRIPT_PATH.exists(),
    reason="script must exist for behavioural test",
)
def test_probe_skipped_when_local_cuda_worker_unset(tmp_path):
    """LOCAL_CUDA_WORKER unset (default 0) + SKIP_CUDA=0 + DRY_RUN=0
    must STILL skip the probe.

    This is the operator-default state: someone running the script on
    a dispatcher host that doesn't have CUDA/NVDEC. Pre-fix, the probe
    would fire and abort. Post-fix, it skips.
    """
    sandbox_repo = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "scripts", sandbox_repo / "scripts")
    probe_marker = tmp_path / "probe_was_called.marker"
    probe_path = sandbox_repo / "scripts" / "probe_nvdec.sh"
    probe_path.write_text(
        "#!/usr/bin/env bash\n"
        f"touch {probe_marker}\n"
        "echo PROBE_INVOKED\n"
        "exit 0\n"
    )
    probe_path.chmod(0o755)
    (sandbox_repo / "upstream" / "videos").mkdir(parents=True)
    (sandbox_repo / ".venv" / "bin").mkdir(parents=True)
    venv_python = sandbox_repo / ".venv" / "bin" / "python"
    venv_python.write_text("#!/usr/bin/env bash\necho mock-python\n")
    venv_python.chmod(0o755)
    (sandbox_repo / "tools").mkdir(parents=True)
    fake_claim = sandbox_repo / "tools" / "claim_lane_dispatch.py"
    fake_claim.write_text("#!/usr/bin/env python3\nimport sys; sys.exit(0)\n")
    env = os.environ.copy()
    env["DRY_RUN"] = "0"
    env["SKIP_CUDA"] = "0"
    env["SKIP_CPU"] = "1"
    # LOCAL_CUDA_WORKER intentionally NOT set (default 0)
    env.pop("LOCAL_CUDA_WORKER", None)
    env["DISCRIMINATOR_TIMESTAMP_SUFFIX"] = "20260509T000000Z"
    discriminator_script = (
        sandbox_repo / "scripts"
        / "remote_lane_avvideodataset_cuda_path_mechanism_discriminator.sh"
    )
    if not discriminator_script.exists():
        pytest.skip("sandbox copy of discriminator script missing")
    result = subprocess.run(
        ["bash", str(discriminator_script)],
        env=env,
        cwd=str(sandbox_repo),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert not probe_marker.exists(), (
        "MEDIUM 1: probe_nvdec.sh was invoked with LOCAL_CUDA_WORKER "
        f"unset (default skip). Script stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


@pytest.mark.skipif(
    not SCRIPT_PATH.exists(),
    reason="script must exist for behavioural test",
)
def test_probe_runs_when_local_cuda_worker_explicitly_set(tmp_path):
    """LOCAL_CUDA_WORKER=1 + SKIP_CUDA=0 + DRY_RUN=0 → probe MUST run.

    This is the explicit opt-in case for an operator who really does
    have CUDA on the dispatcher and wants the safety net.
    """
    sandbox_repo = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "scripts", sandbox_repo / "scripts")
    probe_marker = tmp_path / "probe_was_called.marker"
    probe_path = sandbox_repo / "scripts" / "probe_nvdec.sh"
    probe_path.write_text(
        "#!/usr/bin/env bash\n"
        f"touch {probe_marker}\n"
        "echo PROBE_INVOKED\n"
        "exit 0\n"
    )
    probe_path.chmod(0o755)
    (sandbox_repo / "upstream" / "videos").mkdir(parents=True)
    (sandbox_repo / ".venv" / "bin").mkdir(parents=True)
    venv_python = sandbox_repo / ".venv" / "bin" / "python"
    venv_python.write_text("#!/usr/bin/env bash\necho mock-python\n")
    venv_python.chmod(0o755)
    (sandbox_repo / "tools").mkdir(parents=True)
    fake_claim = sandbox_repo / "tools" / "claim_lane_dispatch.py"
    fake_claim.write_text("#!/usr/bin/env python3\nimport sys; sys.exit(0)\n")
    env = os.environ.copy()
    env["DRY_RUN"] = "0"
    env["SKIP_CUDA"] = "0"
    env["SKIP_CPU"] = "1"
    env["LOCAL_CUDA_WORKER"] = "1"  # explicit opt-in
    env["DISCRIMINATOR_TIMESTAMP_SUFFIX"] = "20260509T000000Z"
    discriminator_script = (
        sandbox_repo / "scripts"
        / "remote_lane_avvideodataset_cuda_path_mechanism_discriminator.sh"
    )
    if not discriminator_script.exists():
        pytest.skip("sandbox copy of discriminator script missing")
    result = subprocess.run(
        ["bash", str(discriminator_script)],
        env=env,
        cwd=str(sandbox_repo),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert probe_marker.exists(), (
        "MEDIUM 1: probe_nvdec.sh was NOT invoked with "
        f"LOCAL_CUDA_WORKER=1 (explicit opt-in). "
        f"Script stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ── Sister tests: SKIP_CUDA=1 still skips probe (legacy contract) ────────


@pytest.mark.skipif(
    not SCRIPT_PATH.exists(),
    reason="script must exist for behavioural test",
)
def test_probe_skipped_when_skip_cuda_set(tmp_path):
    """SKIP_CUDA=1 (regardless of LOCAL_CUDA_WORKER) → probe MUST be skipped.

    This is the original short-circuit and must not regress.
    """
    sandbox_repo = tmp_path / "repo"
    shutil.copytree(REPO_ROOT / "scripts", sandbox_repo / "scripts")
    probe_marker = tmp_path / "probe_was_called.marker"
    probe_path = sandbox_repo / "scripts" / "probe_nvdec.sh"
    probe_path.write_text(
        "#!/usr/bin/env bash\n"
        f"touch {probe_marker}\n"
        "exit 0\n"
    )
    probe_path.chmod(0o755)
    (sandbox_repo / "upstream" / "videos").mkdir(parents=True)
    (sandbox_repo / ".venv" / "bin").mkdir(parents=True)
    venv_python = sandbox_repo / ".venv" / "bin" / "python"
    venv_python.write_text("#!/usr/bin/env bash\necho mock-python\n")
    venv_python.chmod(0o755)
    (sandbox_repo / "tools").mkdir(parents=True)
    fake_claim = sandbox_repo / "tools" / "claim_lane_dispatch.py"
    fake_claim.write_text("#!/usr/bin/env python3\nimport sys; sys.exit(0)\n")
    env = os.environ.copy()
    env["DRY_RUN"] = "0"
    env["SKIP_CUDA"] = "1"
    env["SKIP_CPU"] = "1"
    env["LOCAL_CUDA_WORKER"] = "1"  # would normally enable probe, but SKIP_CUDA short-circuits
    env["DISCRIMINATOR_TIMESTAMP_SUFFIX"] = "20260509T000000Z"
    discriminator_script = (
        sandbox_repo / "scripts"
        / "remote_lane_avvideodataset_cuda_path_mechanism_discriminator.sh"
    )
    if not discriminator_script.exists():
        pytest.skip("sandbox copy of discriminator script missing")
    result = subprocess.run(
        ["bash", str(discriminator_script)],
        env=env,
        cwd=str(sandbox_repo),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert not probe_marker.exists(), (
        "MEDIUM 1 sister: SKIP_CUDA=1 must short-circuit before the "
        "probe even when LOCAL_CUDA_WORKER=1. "
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
