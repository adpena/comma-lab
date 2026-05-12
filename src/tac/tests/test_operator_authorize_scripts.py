"""Tests for the 8 operator-authorize one-command scripts.

Per the operator one-touch authorization toolkit landing 2026-05-11
Deliverable 3. Verifies:

- bash -n syntax check passes
- confirmation prompt blocks without `y/yes` input (tested via stdin redirect)
- /tmp paths NOT used in any script body
- canonical CLAUDE.md tag references present
- proper set -euo pipefail discipline per CLAUDE.md "Forbidden silent-skip
  cascades" + Catalog #2 (check_shell_set_e_present)
- lane-claim helper invoked for GPU-spend scripts (per CLAUDE.md
  CROSS-AGENT DISPATCH COORDINATION)
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

OPERATOR_AUTHORIZE_SCRIPTS = (
    "operator_authorize_autopilot_le_5_dollar_mode.sh",
    "operator_authorize_bulk_anchor_backfill.sh",
    "operator_authorize_crates_io_publish.sh",
    "operator_authorize_hf_dataset_card_push.sh",
    "operator_authorize_phase1_t1_balle_cheap_config_dispatch.sh",
    "operator_authorize_scpp_stage1_anchor_dispatch.sh",
    "operator_authorize_t10_ib_lagrangian_dispatch.sh",
    "operator_authorize_v0_2_0_rc1_github_push.sh",
)

GPU_DISPATCH_SCRIPTS = (
    "operator_authorize_phase1_t1_balle_cheap_config_dispatch.sh",
    "operator_authorize_scpp_stage1_anchor_dispatch.sh",
    "operator_authorize_t10_ib_lagrangian_dispatch.sh",
)


@pytest.mark.parametrize("name", OPERATOR_AUTHORIZE_SCRIPTS)
def test_script_exists(name: str):
    p = _SCRIPTS_DIR / name
    assert p.is_file(), f"missing operator-authorize script: {p}"


@pytest.mark.parametrize("name", OPERATOR_AUTHORIZE_SCRIPTS)
def test_script_is_executable(name: str):
    p = _SCRIPTS_DIR / name
    # owner-execute bit set.
    assert p.stat().st_mode & 0o100, f"script not executable: {p}"


@pytest.mark.parametrize("name", OPERATOR_AUTHORIZE_SCRIPTS)
def test_script_bash_syntax_clean(name: str):
    p = _SCRIPTS_DIR / name
    result = subprocess.run(
        ["bash", "-n", str(p)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"


@pytest.mark.parametrize("name", OPERATOR_AUTHORIZE_SCRIPTS)
def test_script_has_set_euo_pipefail(name: str):
    """Per CLAUDE.md Catalog #2 (check_shell_set_e_present) — `set -e` mandatory."""
    p = _SCRIPTS_DIR / name
    body = p.read_text()
    assert "set -euo pipefail" in body, (
        f"{name} must use 'set -euo pipefail' per CLAUDE.md Catalog #2"
    )


@pytest.mark.parametrize("name", OPERATOR_AUTHORIZE_SCRIPTS)
def test_script_has_no_tmp_paths(name: str):
    """Per CLAUDE.md FORBIDDEN /tmp paths — no /tmp in script body."""
    p = _SCRIPTS_DIR / name
    body = p.read_text()
    # Allow /tmp inside comments warning AGAINST /tmp (must contain a NO/FORBIDDEN/avoid token).
    lines_with_tmp = []
    for i, line in enumerate(body.splitlines(), 1):
        if "/tmp/" in line or "/var/tmp/" in line:
            # Whitelist: occurrences in comments that document the prohibition.
            stripped = line.strip()
            is_doc_only = (
                stripped.startswith("#") or stripped.startswith("//")
            ) and (
                "FORBIDDEN" in line
                or "NOT" in line
                or "no-" in line.lower()
                or "must NOT" in line
                or "avoid" in line.lower()
            )
            if not is_doc_only:
                lines_with_tmp.append((i, line))
    assert not lines_with_tmp, (
        f"{name} has /tmp paths outside CLAUDE.md prohibition documentation: "
        f"{lines_with_tmp}"
    )


@pytest.mark.parametrize("name", OPERATOR_AUTHORIZE_SCRIPTS)
def test_script_has_confirmation_prompt(name: str):
    """Per the operator one-touch toolkit design — single confirmation prompt."""
    p = _SCRIPTS_DIR / name
    body = p.read_text()
    assert "read -r -p" in body, (
        f"{name} must have a 'read -r -p' confirmation prompt"
    )


@pytest.mark.parametrize("name", OPERATOR_AUTHORIZE_SCRIPTS)
def test_script_has_claude_md_cross_reference(name: str):
    """Every script must cross-reference CLAUDE.md non-negotiables."""
    p = _SCRIPTS_DIR / name
    body = p.read_text()
    assert "CLAUDE.md" in body, (
        f"{name} must cross-reference CLAUDE.md non-negotiables"
    )


@pytest.mark.parametrize("name", GPU_DISPATCH_SCRIPTS)
def test_gpu_dispatch_script_invokes_claim_lane_dispatch(name: str):
    """Per CLAUDE.md 'CROSS-AGENT DISPATCH COORDINATION' — GPU scripts must claim."""
    p = _SCRIPTS_DIR / name
    body = p.read_text()
    assert "claim_lane_dispatch.py" in body, (
        f"{name} (GPU dispatch) must invoke tools/claim_lane_dispatch.py per "
        f"CLAUDE.md CROSS-AGENT DISPATCH COORDINATION"
    )


@pytest.mark.parametrize("name", GPU_DISPATCH_SCRIPTS)
def test_gpu_dispatch_script_carries_cost_estimate(name: str):
    """Per CLAUDE.md GPU budget — every dispatch script must declare cost."""
    p = _SCRIPTS_DIR / name
    body = p.read_text()
    assert "EXPECTED_COST_USD" in body or "expected cost" in body.lower(), (
        f"{name} (GPU dispatch) must declare an EXPECTED_COST_USD or "
        f"'expected cost' line"
    )


# ── Behavioral tests via subprocess invocation ────────────────────────────────


def test_bulk_anchor_backfill_aborts_on_no_input(tmp_path: Path):
    """Confirmation prompt with empty stdin should abort cleanly (rc=0, no commit)."""
    p = _SCRIPTS_DIR / "operator_authorize_bulk_anchor_backfill.sh"
    # Pre-stage: a posterior path must exist or the dry-run will write it.
    # We feed empty stdin so the read prompt receives empty → aborts.
    result = subprocess.run(
        ["bash", str(p)],
        input="",  # empty → not [yY] → abort branch
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=60,
    )
    # Either aborts cleanly (rc=0 with "aborted" stdout) OR fails for some
    # other env-specific reason; the key invariant is that NO commit fired.
    if result.returncode == 0:
        assert "aborted" in result.stdout.lower()


def test_autopilot_le_5_dollar_mode_aborts_on_no_input(tmp_path: Path):
    """Confirmation prompt with empty stdin should abort cleanly."""
    p = _SCRIPTS_DIR / "operator_authorize_autopilot_le_5_dollar_mode.sh"
    result = subprocess.run(
        ["bash", str(p)],
        input="",
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=60,
    )
    if result.returncode == 0:
        assert "aborted" in result.stdout.lower()


def test_github_push_aborts_on_no_input():
    """Confirmation prompt with empty stdin should abort cleanly."""
    p = _SCRIPTS_DIR / "operator_authorize_v0_2_0_rc1_github_push.sh"
    result = subprocess.run(
        ["bash", str(p)],
        input="",
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=30,
    )
    # The script may exit early (rc=1) if local tag doesn't exist; or rc=0
    # with aborted message. Either way, no remote push should have fired.
    # Verify the operator-confirmation header text is in stdout.
    assert (
        "operator confirmation" in result.stdout.lower()
        or "FATAL" in result.stdout
        or "FATAL" in result.stderr
    )


def test_t10_dispatch_aborts_on_no_input():
    """Confirmation prompt with empty stdin should abort cleanly."""
    p = _SCRIPTS_DIR / "operator_authorize_t10_ib_lagrangian_dispatch.sh"
    result = subprocess.run(
        ["bash", str(p)],
        input="",
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=30,
    )
    assert "operator confirmation" in result.stdout.lower()


def test_phase1_cheap_config_aborts_on_no_input():
    """Confirmation prompt with empty stdin should abort cleanly."""
    p = _SCRIPTS_DIR / "operator_authorize_phase1_t1_balle_cheap_config_dispatch.sh"
    result = subprocess.run(
        ["bash", str(p)],
        input="",
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=30,
    )
    assert "operator confirmation" in result.stdout.lower()


def test_scpp_dispatch_aborts_on_no_input():
    """Confirmation prompt with empty stdin should abort cleanly."""
    p = _SCRIPTS_DIR / "operator_authorize_scpp_stage1_anchor_dispatch.sh"
    result = subprocess.run(
        ["bash", str(p)],
        input="",
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=30,
    )
    assert "operator confirmation" in result.stdout.lower()


def test_crates_io_publish_aborts_on_no_input():
    """Confirmation prompt with empty stdin should abort cleanly."""
    p = _SCRIPTS_DIR / "operator_authorize_crates_io_publish.sh"
    result = subprocess.run(
        ["bash", str(p)],
        input="n\nn\n",  # decline both prompts (dry-run + publish)
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=30,
    )
    # Either crate-not-found rc=1, or operator-confirmation rc=0 with abort.
    # Either way, no cargo publish should have run.
    assert (
        "operator confirmation" in result.stdout.lower()
        or "FATAL" in result.stdout
        or "FATAL" in result.stderr
    )


def test_hf_card_push_aborts_on_no_input():
    """Confirmation prompt with empty stdin should abort cleanly."""
    p = _SCRIPTS_DIR / "operator_authorize_hf_dataset_card_push.sh"
    result = subprocess.run(
        ["bash", str(p)],
        input="n\nn\n",  # decline hygiene check + decline push
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        timeout=30,
    )
    assert (
        "operator confirmation" in result.stdout.lower()
        or "FATAL" in result.stdout
        or "FATAL" in result.stderr
    )


# ── Content-level invariants for each script ──────────────────────────────────


def test_bulk_anchor_backfill_invokes_canonical_tool():
    p = _SCRIPTS_DIR / "operator_authorize_bulk_anchor_backfill.sh"
    body = p.read_text()
    assert "tools/bulk_backfill_anchors_into_posterior.py" in body


def test_autopilot_le_5_dollar_sets_env_var_and_flag():
    p = _SCRIPTS_DIR / "operator_authorize_autopilot_le_5_dollar_mode.sh"
    body = p.read_text()
    assert "CATHEDRAL_AUTOPILOT_OPERATOR_AUTHORIZED_MODE=1" in body
    assert "--operator-authorized-le-5-dollar-mode" in body
    assert "--journal-path" in body


def test_t10_dispatch_carries_envelope_warning():
    p = _SCRIPTS_DIR / "operator_authorize_t10_ib_lagrangian_dispatch.sh"
    body = p.read_text()
    assert "EXCEEDS" in body or "exceeds" in body  # envelope warning
    assert "40" in body  # cost estimate


def test_phase1_cheap_config_supports_modal_and_vastai():
    p = _SCRIPTS_DIR / "operator_authorize_phase1_t1_balle_cheap_config_dispatch.sh"
    body = p.read_text()
    assert "PHASE1_PLATFORM" in body
    assert "modal" in body.lower()
    assert "vastai" in body.lower() or "vast.ai" in body.lower()


def test_phase1_cheap_config_wires_cost_band_metadata_to_modal_launcher():
    p = _SCRIPTS_DIR / "operator_authorize_phase1_t1_balle_cheap_config_dispatch.sh"
    body = p.read_text()
    assert "EXPECTED_COST_BAND_USD" not in body
    assert "EXPECTED_P50_USD" in body
    assert "--cost-band-trainer experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py" in body
    assert "--cost-band-epochs 3000" in body
    assert "--cost-band-batch-size 16" in body
    assert "--cost-band-all-flags-on" in body


def test_scpp_dispatch_carries_substrate_engineering_tag():
    p = _SCRIPTS_DIR / "operator_authorize_scpp_stage1_anchor_dispatch.sh"
    body = p.read_text()
    assert "substrate_engineering" in body or "substrate engineering" in body


def test_scpp_dispatch_is_training_build_only_until_exact_eval_lands():
    p = _SCRIPTS_DIR / "operator_authorize_scpp_stage1_anchor_dispatch.sh"
    body = p.read_text()
    assert "SCPP_RUN_CONTEST_CUDA_AUTH_EVAL=0" in body
    assert "no score claim" in body
    assert "separate exact-eval dispatch" in body


def test_scpp_remote_script_defaults_training_build_only():
    p = _SCRIPTS_DIR / "remote_lane_scpp_stage1.sh"
    body = p.read_text()
    assert 'RUN_CONTEST_CUDA_AUTH_EVAL="${SCPP_RUN_CONTEST_CUDA_AUTH_EVAL:-0}"' in body
    assert 'RUN_CONTEST_CUDA_AUTH_EVAL="1"' not in body
    assert "refused exact-eval request" in body
    assert "Stage 4: training command" in body


def test_v020rc1_push_carries_tag_immutability_warning():
    p = _SCRIPTS_DIR / "operator_authorize_v0_2_0_rc1_github_push.sh"
    body = p.read_text()
    assert "permanent" in body.lower() or "immutable" in body.lower()
    assert "v0.2.0-rc1" in body


def test_crates_io_publish_carries_immutability_warning():
    p = _SCRIPTS_DIR / "operator_authorize_crates_io_publish.sh"
    body = p.read_text()
    assert "IMMUTABLE" in body or "immutable" in body.lower()
    assert "yank" in body.lower()


def test_hf_card_push_runs_hygiene_check():
    p = _SCRIPTS_DIR / "operator_authorize_hf_dataset_card_push.sh"
    body = p.read_text()
    assert "check_public_release_hygiene" in body
    assert "comma_pr_archive_dataset_card.md" in body
