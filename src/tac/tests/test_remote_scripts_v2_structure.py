# SPDX-License-Identifier: MIT
"""Structural tests for the V2 remote bootstrap scripts.

CLAUDE.md non-negotiables on every remote script:

  * `set -euo pipefail` (NOT `set -uo` — that ate LANE-B 6.5h+$2 in 2026-04-26).
  * `--device cuda` only — no MPS / CPU fallback (memory:
    feedback_default_to_convenience_trap).
  * Python `zipfile` instead of shell `zip` (PyTorch container has no `zip`).
  * `source env.sh` so all `TAC_*` env-vars resolve.
  * Provenance JSON written (memory feedback_canonical_remote_bootstraps).
  * Heartbeat file written.
  * Stage 0 NVDEC probe BEFORE any GPU spend (memory
    feedback_vastai_nvdec_host_variation).
  * Auth eval at the end (CLAUDE.md "Auth eval EVERYWHERE" non-negotiable).
  * RESULT_JSON guard so a silent eval crash exits non-zero.

These tests source-grep each V2 script for the required patterns. They
catch a regression FAST (no GPU spend) and document the contract.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]

V2_SCRIPTS = (
    "scripts/remote_lane_lr_v2_learnable_rank.sh",
    "scripts/remote_lane_lm_v2_endpoint_tracking.sh",
    "scripts/remote_lane_v_v2_annealed_halfframe.sh",
)


def _read(script: str) -> str:
    p = REPO / script
    assert p.exists(), f"{script} does not exist"
    return p.read_text()


# ── Per-script existence + executable ─────────────────────────────────────


@pytest.mark.parametrize("script", V2_SCRIPTS)
def test_script_exists(script: str) -> None:
    p = REPO / script
    assert p.exists(), f"{script} does not exist"
    assert p.stat().st_size > 0


# ── Bash safety ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("script", V2_SCRIPTS)
def test_script_uses_set_e_u_o_pipefail(script: str) -> None:
    """`set -euo pipefail` is mandatory. `set -uo pipefail` (no -e) caused
    the LANE-B cascade silent-failure trap (memory:
    feedback_zip_dep_bootstrap_trap)."""
    src = _read(script)
    assert "set -euo pipefail" in src, (
        f"{script} missing `set -euo pipefail` — silent-cascade trap risk"
    )
    # Catch the partial-set trap explicitly.
    assert "set -uo pipefail" not in src.replace("set -euo pipefail", ""), (
        f"{script} contains a stray `set -uo pipefail` (no -e); the cascade "
        f"trap that ate LANE-B 6.5h + $2 in 2026-04-26 must not reappear"
    )


# ── Device safety ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("script", V2_SCRIPTS)
def test_script_uses_device_cuda(script: str) -> None:
    """All eval / training / pose-TTO invocations must pass --device cuda.
    NEVER MPS, NEVER CPU (CLAUDE.md non-negotiable)."""
    src = _read(script)
    assert "--device cuda" in src, (
        f"{script} missing --device cuda — must be CUDA-only per CLAUDE.md"
    )
    assert "--device mps" not in src, (
        f"{script} contains --device mps — MPS forbidden per CLAUDE.md"
    )
    # --device cpu is only permissible if the script passes a deterministic-
    # bytes acceptable caveat; we do not allow CPU in any V2 script.
    assert "--device cpu" not in src, (
        f"{script} contains --device cpu — V2 scripts must be CUDA-only"
    )


# ── Python zipfile instead of shell zip ───────────────────────────────────


@pytest.mark.parametrize("script", V2_SCRIPTS)
def test_script_uses_python_zipfile(script: str) -> None:
    """Use Python `zipfile.ZipFile`, NOT shell `zip` binary (PyTorch container
    has no `zip` — memory feedback_zip_dep_bootstrap_trap)."""
    src = _read(script)
    # Only flag bare `zip` invocations (e.g. lines starting with `zip ` or
    # `zip -X`). Whitelist patterns that contain `zip` as substring (e.g.
    # `zipfile`, `unzip`, `.zip`) which are legitimate.
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Match `zip ` as a command (start of a shell statement, not part
        # of a path / variable / longer word).
        if (
            stripped.startswith("zip ")
            or stripped.startswith("zip -")
            or " | zip " in stripped
            or "; zip " in stripped
        ):
            pytest.fail(
                f"{script} uses shell `zip` binary on line: {stripped!r}; "
                f"use Python zipfile.ZipFile instead "
                f"(PyTorch container has no `zip`)"
            )


# ── env.sh sourced ────────────────────────────────────────────────────────


@pytest.mark.parametrize("script", V2_SCRIPTS)
def test_script_sources_env_sh(script: str) -> None:
    """`source env.sh` must run early so TAC_* env-vars + PYTHONPATH resolve."""
    src = _read(script)
    assert 'source "$WORKSPACE/env.sh"' in src or 'source $WORKSPACE/env.sh' in src, (
        f"{script} does not source env.sh — TAC_* env-vars + PYTHONPATH may "
        f"be unset, training/eval can fail in surprising ways"
    )


# ── Provenance JSON ───────────────────────────────────────────────────────


@pytest.mark.parametrize("script", V2_SCRIPTS)
def test_script_writes_provenance_json(script: str) -> None:
    """Every remote script must emit provenance.json so a fresh agent can
    reconstruct the experiment without chat memory (memory:
    feedback_canonical_remote_bootstraps)."""
    src = _read(script)
    assert "provenance.json" in src, (
        f"{script} missing provenance.json — fresh-agent reconstruction is "
        f"the canonical remote bootstrap contract"
    )
    assert "git_hash" in src, (
        f"{script} provenance does not capture git_hash — code-version drift "
        f"will not be auditable"
    )


# ── Heartbeat ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("script", V2_SCRIPTS)
def test_script_writes_heartbeat(script: str) -> None:
    """Every remote script must emit a heartbeat so the watchdog can detect
    a dead training loop while the tmux session is still alive
    (CLAUDE.md remote-code-parity rule #3)."""
    src = _read(script)
    assert "heartbeat" in src.lower(), (
        f"{script} does not write a heartbeat file — silent training death "
        f"would not be detected before 30+ min of idle GPU spend"
    )


# ── NVDEC probe before GPU spend ──────────────────────────────────────────


@pytest.mark.parametrize("script", V2_SCRIPTS)
def test_script_runs_nvdec_probe_before_gpu_work(script: str) -> None:
    """Stage 0 NVDEC probe must run BEFORE any GPU work (memory:
    feedback_vastai_nvdec_host_variation — same 4090 image, different host,
    NVDEC silently broken until the auth eval at the end)."""
    src = _read(script)
    assert "probe_nvdec.sh" in src, (
        f"{script} does not invoke probe_nvdec.sh — risk of burning $0.20 "
        f"of setup before discovering NVDEC is broken on the host"
    )


# ── Auth eval at the end ──────────────────────────────────────────────────


@pytest.mark.parametrize("script", V2_SCRIPTS)
def test_script_runs_auth_eval_at_end(script: str) -> None:
    """Every chained experiment MUST end with a CUDA auth eval against its
    archive (CLAUDE.md "Auth eval EVERYWHERE" non-negotiable)."""
    src = _read(script)
    assert "contest_auth_eval.py" in src, (
        f"{script} does not invoke contest_auth_eval.py — CLAUDE.md "
        f"'Auth eval EVERYWHERE' non-negotiable violated"
    )


# ── RESULT_JSON silent-crash guard ────────────────────────────────────────


@pytest.mark.parametrize("script", V2_SCRIPTS)
def test_script_has_result_json_guard(script: str) -> None:
    """The auth_eval log MUST be checked for RESULT_JSON before the script
    declares success (memory: feedback_zip_dep_bootstrap_trap — LANE-B
    cascaded silently because no RESULT_JSON validation existed)."""
    src = _read(script)
    assert "RESULT_JSON" in src, (
        f"{script} does not validate auth_eval log for RESULT_JSON — silent "
        f"eval crash would emit zero-exit and fool the operator"
    )


# ── No invented CLI flags (canonical pattern) ─────────────────────────────


@pytest.mark.parametrize("script", V2_SCRIPTS)
def test_script_has_dead_flag_scan(script: str) -> None:
    """Every script must include a dead-flag-wiring guard (memory:
    feedback_dead_flag_wiring_pattern — the 2026-04-26 incident burned 3
    rounds of council review on auth-eval-on-best wiring with --auth-eval-
    masks flag that auth_eval_renderer.py never had)."""
    src = _read(script)
    assert (
        "INVENTED FLAGS" in src
        or "dead-flag" in src.lower()
    ), (
        f"{script} missing dead-flag-wiring guard — risk of inventing CLI "
        f"flags that the target script's argparse never declared"
    )


# ── V2 lane-specific identifiers ──────────────────────────────────────────


def test_lane_lr_v2_passes_learnable_lora_max_rank() -> None:
    """Lane LR-V2 must pass --learnable-lora-max-rank to optimize_poses.py
    (the V2 oversight fix). Catches a refactor that silently reverts to
    --lora-rank (V1)."""
    src = _read("scripts/remote_lane_lr_v2_learnable_rank.sh")
    assert "--learnable-lora-max-rank" in src, (
        "Lane LR-V2 script must pass --learnable-lora-max-rank — the V2 fix"
    )
    # And must NOT pass the V1 --lora-rank (mutual exclusion at runtime).
    # We grep for the FLAG (not the comment) — flags appear after backslash-
    # continuation indentation.
    assert "    --lora-rank " not in src, (
        "Lane LR-V2 script must NOT pass --lora-rank — that's the V1 path "
        "and the optimize_poses mutual-exclusion gate would SystemExit"
    )


def test_lane_lm_v2_passes_method_endpoint() -> None:
    """Lane LM-V2 must pass --method endpoint (the V2 oversight fix)."""
    src = _read("scripts/remote_lane_lm_v2_endpoint_tracking.sh")
    assert "--method endpoint" in src, (
        "Lane LM-V2 script must pass --method endpoint — the V2 fix"
    )
    # Min correlation gate must be elevated to 0.30 for V2.
    assert "--min-correlation 0.30" in src, (
        "Lane LM-V2 must use the elevated --min-correlation 0.30 gate "
        "(V1's 0.017 correlation could not clear this; V2's endpoint "
        "tracking is expected to land at 0.30+)"
    )


def test_lane_v_v2_uses_annealed_profile() -> None:
    """Lane V-V2 must pass the annealed profile (the V2 oversight fix)."""
    src = _read("scripts/remote_lane_v_v2_annealed_halfframe.sh")
    assert "--profile quantizr_replica_88k_halfframe_annealed" in src, (
        "Lane V-V2 script must pass the annealed profile name — the V2 fix"
    )
    # Must NOT pass the V1 profile.
    assert "--profile quantizr_replica_88k_halfframe " not in src, (
        "Lane V-V2 script must NOT pass the non-annealed V1 profile"
    )
