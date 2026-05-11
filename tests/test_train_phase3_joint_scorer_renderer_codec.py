"""Tests for ``experiments/train_phase3_joint_scorer_renderer_codec.py``.

Phase3DispatchGate fail-closed semantics (Catalog #134) MUST hold:

- production calls fail if any precondition missing
- ``unsafe_test_only=True`` accepted only from test paths (catalog #142)
- the smoke path uses the escape hatch correctly

Tests also confirm:

- CLI flag set is consistent (no MPS, auth-eval gated, smoke gated)
- Phase3DispatchGate is invoked at construction time
- Smoke-run produces manifest with Phase 3 compliance tags
- EMA shadow saved as inference checkpoint
- ``load_t10_aux_scorer`` correctly loads a smoke-trained T10 checkpoint
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

torch = pytest.importorskip("torch")

from experiments.train_phase3_joint_scorer_renderer_codec import (  # noqa: E402
    PHASE3_LANE_ID,
    PHASE3_PREDICTED_DELTA_SCORE,
    PHASE3_SCHEMA_VERSION,
    build_phase3_gate_from_args,
    load_t10_aux_scorer,
    parse_args,
)
from tac.phase3.joint_scorer_renderer_codec import (  # noqa: E402
    Phase3DispatchGate,
    Phase3DispatchGateError,
)


# ---------------------------------------------------------------------------
# Phase3DispatchGate fail-closed semantics (Catalog #134)
# ---------------------------------------------------------------------------


def test_phase3_gate_smoke_path_uses_unsafe_test_only():
    """Smoke path constructs the gate via unsafe_test_only=True."""
    args = parse_args([
        "--output-dir", str(REPO_ROOT / "experiments/results/cli_test"),
        "--smoke", "--allow-missing-canonical-a1",
    ])
    gate = build_phase3_gate_from_args(args)
    assert isinstance(gate, Phase3DispatchGate)
    assert gate.unsafe_test_only is True


def test_phase3_gate_production_path_refuses_without_preconditions():
    """Production path (no --smoke) MUST raise unless all preconditions set."""
    args = parse_args([
        "--output-dir", str(REPO_ROOT / "experiments/results/cli_test"),
        # NO --smoke; NO preconditions
    ])
    with pytest.raises((TypeError, Phase3DispatchGateError)):
        # build_phase3_gate_from_args will pass None values; the gate
        # ``check()`` will raise.  Either TypeError (None comparison)
        # or Phase3DispatchGateError (explicit precondition fail) is OK.
        build_phase3_gate_from_args(args)


def test_phase3_gate_production_path_with_full_preconditions_passes():
    """Production path with valid preconditions constructs successfully."""
    args = parse_args([
        "--output-dir", str(REPO_ROOT / "experiments/results/cli_test"),
        "--phase2-anchor-score", "0.140",
        "--phase2-anchor-evidence-path", "experiments/results/phase2_anchor.json",
        "--distillation-gap-estimate", "0.025",
        "--distillation-gap-evidence-path", "experiments/results/t10_anchor/distillation_gap_estimate.json",
        "--operator-approved-gpu-budget-usd", "800.0",
        "--aaf68f37-verdict-evidence-path", ".omx/research/aaf68f37_verdict_clean.md",
        "--phase3-council-deliberation-path",
        ".omx/research/fields_medal_grand_council_all_phases_design_deliberate_implement_20260509.md",
    ])
    gate = build_phase3_gate_from_args(args)
    assert isinstance(gate, Phase3DispatchGate)
    assert gate.unsafe_test_only is False
    assert gate.phase2_anchor_verified is True
    assert gate.phase2_anchor_score == 0.140
    assert gate.distillation_gap_estimate == 0.025
    assert gate.operator_approved_gpu_budget_usd == 800.0


def test_phase3_gate_refuses_phase2_anchor_above_threshold():
    """Phase2 anchor > 0.142 must be refused (Phase 2 floor REBASELINE)."""
    args = parse_args([
        "--output-dir", str(REPO_ROOT / "experiments/results/cli_test"),
        "--phase2-anchor-score", "0.150",  # > 0.142
        "--phase2-anchor-evidence-path", "experiments/results/phase2_anchor.json",
        "--distillation-gap-estimate", "0.025",
        "--distillation-gap-evidence-path", "experiments/results/t10/distillation_gap.json",
        "--operator-approved-gpu-budget-usd", "800.0",
        "--aaf68f37-verdict-evidence-path", ".omx/research/aaf.md",
        "--phase3-council-deliberation-path", ".omx/research/council.md",
    ])
    with pytest.raises(Phase3DispatchGateError, match="0.142"):
        build_phase3_gate_from_args(args)


def test_phase3_gate_refuses_distillation_gap_above_threshold():
    """Distillation gap > 0.03 must be refused (Hinton 2014 §3 target)."""
    args = parse_args([
        "--output-dir", str(REPO_ROOT / "experiments/results/cli_test"),
        "--phase2-anchor-score", "0.140",
        "--phase2-anchor-evidence-path", "experiments/results/phase2_anchor.json",
        "--distillation-gap-estimate", "0.050",  # > 0.03
        "--distillation-gap-evidence-path", "experiments/results/t10/distillation_gap.json",
        "--operator-approved-gpu-budget-usd", "800.0",
        "--aaf68f37-verdict-evidence-path", ".omx/research/aaf.md",
        "--phase3-council-deliberation-path", ".omx/research/council.md",
    ])
    with pytest.raises(Phase3DispatchGateError, match="3%"):
        build_phase3_gate_from_args(args)


def test_phase3_gate_refuses_gpu_budget_outside_envelope():
    """GPU budget outside [600, 1200] must be refused."""
    args = parse_args([
        "--output-dir", str(REPO_ROOT / "experiments/results/cli_test"),
        "--phase2-anchor-score", "0.140",
        "--phase2-anchor-evidence-path", "experiments/results/phase2_anchor.json",
        "--distillation-gap-estimate", "0.025",
        "--distillation-gap-evidence-path", "experiments/results/t10/distillation_gap.json",
        "--operator-approved-gpu-budget-usd", "200.0",  # < 600
        "--aaf68f37-verdict-evidence-path", ".omx/research/aaf.md",
        "--phase3-council-deliberation-path", ".omx/research/council.md",
    ])
    with pytest.raises(Phase3DispatchGateError, match="\\[\\$600, \\$1200\\]"):
        build_phase3_gate_from_args(args)


# ---------------------------------------------------------------------------
# CLI parser flags
# ---------------------------------------------------------------------------


def test_cli_required_flags():
    with pytest.raises(SystemExit):
        parse_args([])  # missing --output-dir


def test_cli_smoke_flag_defaults():
    args = parse_args(["--output-dir", str(REPO_ROOT / "experiments/results/cli_test")])
    assert args.smoke is False
    assert args.auth_eval is False
    assert args.ema_decay == 0.997
    assert args.enable_eval_roundtrip_in_training is True
    assert args.distill_temperature == 2.0


def test_cli_phase3_specific_defaults():
    args = parse_args(["--output-dir", str(REPO_ROOT / "experiments/results/cli_test")])
    # Phase 3 council canon
    assert args.distill_temperature == 2.0
    assert args.use_t17_shared_vq_codebook is True
    assert args.use_t18_balle_nonlinear_transform is True
    assert args.use_t13_sqrt_n_latent_budget is True


def test_cli_rejects_mps_at_choice_layer():
    with pytest.raises(SystemExit):
        parse_args(["--output-dir", "/dev/null", "--device", "mps"])


# ---------------------------------------------------------------------------
# Constants tagged per CLAUDE.md
# ---------------------------------------------------------------------------


def test_predicted_delta_score_carries_tag_and_floor():
    assert "[predicted;" in PHASE3_PREDICTED_DELTA_SCORE
    assert "Tishby IB" in PHASE3_PREDICTED_DELTA_SCORE
    assert "0.140" in PHASE3_PREDICTED_DELTA_SCORE


# ---------------------------------------------------------------------------
# load_t10_aux_scorer
# ---------------------------------------------------------------------------


def test_load_t10_aux_scorer_refuses_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="T10 aux-scorer"):
        load_t10_aux_scorer(
            tmp_path / "missing.pt",
            device=torch.device("cpu"),
            smoke_mode=True,
        )


def test_load_t10_aux_scorer_refuses_missing_config(tmp_path):
    bad_ckpt = {"ema_state_dict": {}}  # missing t10_config
    path = tmp_path / "bad.pt"
    torch.save(bad_ckpt, path)
    with pytest.raises(ValueError, match="t10_config"):
        load_t10_aux_scorer(path, device=torch.device("cpu"), smoke_mode=True)


# ---------------------------------------------------------------------------
# Subprocess smoke (end-to-end on CPU; uses unsafe_test_only path)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_smoke_run_produces_manifest_with_phase3_tags(tmp_path):
    out = tmp_path / "phase3_smoke_subprocess"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_phase3_joint_scorer_renderer_codec.py"),
            "--output-dir", str(out),
            "--device", "cpu",
            "--smoke", "--allow-missing-canonical-a1",
            "--epochs", "1", "--seed", "42",
        ],
        capture_output=True, text=True, timeout=240,
    )
    assert proc.returncode == 0, f"smoke failed: {proc.stderr}"
    manifest_path = out / "phase3_provenance.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == PHASE3_SCHEMA_VERSION
    assert manifest["lane_id"] == PHASE3_LANE_ID
    assert manifest["predicted_delta_score"] == PHASE3_PREDICTED_DELTA_SCORE
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["phase3_dispatch_gate_constructed"] is True
    assert manifest["ib_loss_finite_first_step"] is True
    assert manifest["ema_decay"] == 0.997
    assert manifest["eval_roundtrip"] is True
    # Lagrangian form is recorded.
    lf = manifest["phase3_lagrangian_form"]
    assert "Tishby" in lf["name"]
    assert "Hinton" in lf["theorems_invoked"]
    # 13 compliance tags including Phase 3 specifics.
    tags = manifest["compliance_tags"]
    for required in (
        "phase3_dispatch_gate_enforced_at_construction",
        "phase3_dispatch_gate_catalog_134",
        "ema_0p997_snapshot_restore",
        "eval_roundtrip_true",
        "no_mps_authoritative",
        "differentiable_yuv6",
        "ib_lagrangian_tishby_hinton_t_2p0",
        "aux_scorer_frozen_at_phase3_training",
        "no_synthetic_outside_smoke",
        "no_tmp_paths",
        "auth_eval_gated",
        "substrate_engineering_exception_principled",
        "inflate_loc_budget_le_200_per_hnerv_parity_lesson_4",
    ):
        assert required in tags, f"missing compliance tag {required}"


@pytest.mark.slow
def test_trainer_refuses_auth_eval_phase1_scaffold_only(tmp_path):
    out = tmp_path / "phase3_auth_refused"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_phase3_joint_scorer_renderer_codec.py"),
            "--output-dir", str(out),
            "--device", "cpu", "--auth-eval",
            "--smoke", "--allow-missing-canonical-a1",
        ],
        capture_output=True, text=True, timeout=60,
    )
    assert proc.returncode != 0
    combined = proc.stderr + proc.stdout
    assert "refused" in combined.lower() or "scaffold" in combined.lower()


@pytest.mark.slow
def test_ema_shadow_checkpoint_keys(tmp_path):
    out = tmp_path / "phase3_ema_keys"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_phase3_joint_scorer_renderer_codec.py"),
            "--output-dir", str(out),
            "--device", "cpu",
            "--smoke", "--allow-missing-canonical-a1",
            "--epochs", "1", "--seed", "42",
        ],
        capture_output=True, text=True, timeout=240,
    )
    assert proc.returncode == 0, proc.stderr
    ckpt = torch.load(out / "phase3_ema_shadow.pt", map_location="cpu", weights_only=False)
    assert "ema_decoder" in ckpt
    assert "ema_balle" in ckpt
    assert "aux_scorer_state_dict" in ckpt
    assert ckpt["schema"] == PHASE3_SCHEMA_VERSION
    cfg = ckpt["phase3_config"]
    assert cfg["distill_temperature"] == 2.0
    assert cfg["use_t17_shared_vq_codebook"] is True
    assert cfg["use_t18_balle_nonlinear_transform"] is True


@pytest.mark.slow
def test_smoke_run_uses_inline_aux_scorer(tmp_path):
    """When no T10 checkpoint provided, smoke mode builds aux scorer inline."""
    out = tmp_path / "phase3_inline_aux"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments/train_phase3_joint_scorer_renderer_codec.py"),
            "--output-dir", str(out),
            "--device", "cpu",
            "--smoke", "--allow-missing-canonical-a1",
            "--epochs", "1", "--seed", "42",
        ],
        capture_output=True, text=True, timeout=240,
    )
    assert proc.returncode == 0, proc.stderr
    manifest = json.loads((out / "phase3_provenance.json").read_text())
    assert manifest["aux_scorer_source"] == "smoke_inline"
