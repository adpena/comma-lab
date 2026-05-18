# SPDX-License-Identifier: MIT
"""Tests for Z7-Mamba-2 substrate scaffold (TOP-5 #2 SCAFFOLD).

Verifies:
1. Mamba2PredictorConfig defaults match design memo §7.
2. Mamba2Predictor instantiates with reference_torch backend on
   non-CUDA environments (MPS / CPU).
3. Forward-pass on small input produces correct shape.
4. Runtime-configurable ego-source flag works via Mamba2PredictorConfig
   (default ego_motion_dim=8 matches Z6-v1 PoseNet-projection; different
   ego_motion_dim allowed for Z6 4c scorer-logit channel).
5. Per-pair master gradient compatibility (Catalog #810): gradients flow
   through both z_prev and ego_motion inputs.
6. Canonical signatures match Z6 sister
   (FilmConditionedNextFramePredictor + MultiLayerFilmPredictor).
7. NotImplementedError raised from _full_main per Catalog #240.
8. _smoke_main exits 0 with non-None backend_active.

Per parent design memo (.omx/research/z7_mamba2_substrate_design_memo_20260518.md)
+ Z7 parent symposium (.omx/research/council_per_substrate_symposium_z7_lstm_predictive_coding_20260517.md).

Lane: lane_top5_2_z7_mamba2_scaffold_design_20260518.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path

import pytest
import torch

from tac.optimization.mamba2_predictor import (
    MAMBA_SSM_AVAILABLE,
    MAMBA_SSM_BACKEND,
    REFERENCE_TORCH_BACKEND,
    Mamba2Predictor,
    Mamba2PredictorConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Mamba2PredictorConfig tests
# ---------------------------------------------------------------------------


def test_config_defaults_match_design_memo_section_7():
    cfg = Mamba2PredictorConfig()
    # Defaults from design memo §7 architectural specification
    assert cfg.latent_dim == 24, "latent_dim must match Z6-v1 (design memo §7)"
    assert cfg.ego_motion_dim == 8, "ego_motion_dim default must match PoseNet-projection baseline"
    assert cfg.d_model == 64, "d_model default per design memo §7"
    assert cfg.d_state == 16, "d_state default per Mamba-2 canonical for language"
    assert cfg.expand == 2, "expand default per upstream reference"
    assert cfg.d_conv == 4, "d_conv default per upstream reference"
    assert cfg.backend == "auto", "backend default must be 'auto'"
    assert cfg.stateful is True, "stateful default must be True (Wyner-Ziv pattern)"
    assert cfg.identity_predictor is False, "identity_predictor default must be False"


def test_config_d_inner_property():
    cfg = Mamba2PredictorConfig(d_model=64, expand=2)
    assert cfg.d_inner == 128, "d_inner = expand * d_model"


def test_config_predictor_input_dim_property():
    cfg = Mamba2PredictorConfig(latent_dim=24, ego_motion_dim=8)
    assert cfg.predictor_input_dim == 32, "predictor_input_dim = latent_dim + ego_motion_dim"

    # Verify runtime-configurable ego-source (Revision #4): different
    # ego_motion_dim for scorer-logit channel
    cfg_4c = Mamba2PredictorConfig(latent_dim=24, ego_motion_dim=16)
    assert cfg_4c.predictor_input_dim == 40, "ego_motion_dim is runtime-configurable per Z7 symposium Revision #4"


# ---------------------------------------------------------------------------
# Mamba2Predictor instantiation tests
# ---------------------------------------------------------------------------


def test_reference_torch_backend_always_available():
    """reference_torch backend must work on any environment (MPS/CPU/CUDA)."""
    cfg = Mamba2PredictorConfig(backend="reference_torch")
    predictor = Mamba2Predictor(cfg)
    assert predictor.backend_active == REFERENCE_TORCH_BACKEND
    # Has trainable parameters (not identity mode)
    assert predictor.num_parameters() > 0


def test_auto_backend_falls_back_to_reference_when_mamba_ssm_missing():
    """auto backend on MPS/CPU must fall back to reference_torch with warning."""
    cfg = Mamba2PredictorConfig(backend="auto")
    if not MAMBA_SSM_AVAILABLE:
        # On M5 Max / non-CUDA: must warn + fall back
        with pytest.warns(UserWarning, match="mamba_ssm not available"):
            predictor = Mamba2Predictor(cfg)
        assert predictor.backend_active == REFERENCE_TORCH_BACKEND
    else:
        # On CUDA env: auto should pick mamba_ssm
        predictor = Mamba2Predictor(cfg)
        assert predictor.backend_active == MAMBA_SSM_BACKEND


def test_mamba_ssm_backend_raises_when_unavailable():
    """Forcing mamba_ssm on a system without it must raise ImportError."""
    if MAMBA_SSM_AVAILABLE:
        pytest.skip("mamba_ssm is available; cannot test ImportError path")
    cfg = Mamba2PredictorConfig(backend="mamba_ssm")
    with pytest.raises(ImportError, match="mamba_ssm backend requested"):
        Mamba2Predictor(cfg)


def test_unknown_backend_raises_value_error():
    cfg = Mamba2PredictorConfig(backend="bogus_backend")
    with pytest.raises(ValueError, match="Unknown backend"):
        Mamba2Predictor(cfg)


def test_identity_predictor_mode_zero_params_returns_z_prev_unchanged():
    """Catalog #125 hook #6 + Z7 symposium Revision #2 disambiguator probe."""
    cfg = Mamba2PredictorConfig(latent_dim=24, ego_motion_dim=8, identity_predictor=True)
    predictor = Mamba2Predictor(cfg)
    assert predictor.num_parameters() == 0
    assert predictor.backend_active == "identity"

    z_prev = torch.randn(4, 24)
    ego = torch.randn(4, 8)
    out = predictor(z_prev, ego)
    assert torch.allclose(out, z_prev), "identity_predictor must return z_prev unchanged"


# ---------------------------------------------------------------------------
# Forward-pass shape + signature compatibility tests
# ---------------------------------------------------------------------------


def test_forward_pass_shape_small_input():
    """Output shape must match z_prev shape per canonical Z6 signature."""
    cfg = Mamba2PredictorConfig(backend="reference_torch")
    predictor = Mamba2Predictor(cfg)
    z_prev = torch.randn(4, 24)
    ego = torch.randn(4, 8)
    out = predictor(z_prev, ego)
    assert out.shape == z_prev.shape, f"shape mismatch: {out.shape} != {z_prev.shape}"


def test_forward_pass_canonical_signature_matches_z6_sister():
    """Mamba2Predictor.forward signature must match Z6 FilmConditionedNextFramePredictor."""
    from tac.substrates.time_traveler_l5_z6.architecture import (
        FilmConditionedNextFramePredictor,
    )

    # Inspect both signatures
    z6_sig = inspect.signature(FilmConditionedNextFramePredictor.forward)
    z6_params = [p for p in z6_sig.parameters.values() if p.name != "self"]

    mamba_sig = inspect.signature(Mamba2Predictor.forward)
    mamba_params = [p for p in mamba_sig.parameters.values() if p.name != "self"]

    assert [p.name for p in z6_params] == [p.name for p in mamba_params], (
        f"Mamba2Predictor.forward params {[p.name for p in mamba_params]} must "
        f"match Z6 FilmConditionedNextFramePredictor.forward params "
        f"{[p.name for p in z6_params]} per design memo §6 layer #2 ADOPT_CANONICAL"
    )


def test_to_z6_compatible_signature_returns_human_readable_string():
    cfg = Mamba2PredictorConfig(latent_dim=24, ego_motion_dim=8, backend="reference_torch")
    predictor = Mamba2Predictor(cfg)
    sig = predictor.to_z6_compatible_signature()
    assert "Mamba2Predictor canonical signature" in sig
    assert "z_prev" in sig and "ego_motion" in sig
    assert "(B, 24)" in sig and "(B, 8)" in sig
    assert "backend_active=" in sig


def test_num_parameters_matches_z6_sister_signature():
    """Mamba2Predictor.num_parameters must exist matching Z6 sister pattern."""
    from tac.substrates.time_traveler_l5_z6.architecture import (
        FilmConditionedNextFramePredictor,
    )
    # Both must have num_parameters method
    assert hasattr(FilmConditionedNextFramePredictor, "num_parameters")
    assert hasattr(Mamba2Predictor, "num_parameters")


# ---------------------------------------------------------------------------
# Runtime-configurable ego-source tests (Z7 symposium Revision #4)
# ---------------------------------------------------------------------------


def test_runtime_configurable_ego_source_posenet_projection_baseline():
    """ego_motion_dim=8 matches Z6-v1 PoseNet-projection baseline."""
    cfg = Mamba2PredictorConfig(latent_dim=24, ego_motion_dim=8, backend="reference_torch")
    predictor = Mamba2Predictor(cfg)
    z_prev = torch.randn(4, 24)
    ego_8 = torch.randn(4, 8)
    out = predictor(z_prev, ego_8)
    assert out.shape == (4, 24)


def test_runtime_configurable_ego_source_scorer_logit_compressed():
    """Different ego_motion_dim simulates Z6 4c scorer-logit channel."""
    # Z6 4c winning channel may have different dim (e.g., 16-dim compressed
    # logit vector); the predictor must accept it via runtime config
    cfg = Mamba2PredictorConfig(latent_dim=24, ego_motion_dim=16, backend="reference_torch")
    predictor = Mamba2Predictor(cfg)
    z_prev = torch.randn(4, 24)
    ego_16 = torch.randn(4, 16)
    out = predictor(z_prev, ego_16)
    assert out.shape == (4, 24)


def test_ego_motion_dim_mismatch_raises_value_error():
    cfg = Mamba2PredictorConfig(latent_dim=24, ego_motion_dim=8, backend="reference_torch")
    predictor = Mamba2Predictor(cfg)
    z_prev = torch.randn(4, 24)
    ego_wrong = torch.randn(4, 16)  # Mismatch
    with pytest.raises(ValueError, match="ego_motion last dim 16 != ego_motion_dim 8"):
        predictor(z_prev, ego_wrong)


def test_latent_dim_mismatch_raises_value_error():
    cfg = Mamba2PredictorConfig(latent_dim=24, ego_motion_dim=8, backend="reference_torch")
    predictor = Mamba2Predictor(cfg)
    z_prev_wrong = torch.randn(4, 32)  # Mismatch
    ego = torch.randn(4, 8)
    with pytest.raises(ValueError, match="z_prev last dim 32 != latent_dim 24"):
        predictor(z_prev_wrong, ego)


# ---------------------------------------------------------------------------
# Per-pair master gradient compatibility (Catalog #810)
# ---------------------------------------------------------------------------


def test_per_pair_master_gradient_compatibility_z_prev():
    """Catalog #810: gradients must flow through z_prev input."""
    cfg = Mamba2PredictorConfig(backend="reference_torch")
    predictor = Mamba2Predictor(cfg)
    z_prev = torch.randn(4, 24, requires_grad=True)
    ego = torch.randn(4, 8, requires_grad=True)
    out = predictor(z_prev, ego)
    out.sum().backward()
    assert z_prev.grad is not None and z_prev.grad.abs().sum() > 0, (
        "z_prev gradient must flow through Mamba-2 predictor per Catalog #810"
    )


def test_per_pair_master_gradient_compatibility_ego_motion():
    """Catalog #810: gradients must flow through ego_motion input."""
    cfg = Mamba2PredictorConfig(backend="reference_torch")
    predictor = Mamba2Predictor(cfg)
    z_prev = torch.randn(4, 24, requires_grad=True)
    ego = torch.randn(4, 8, requires_grad=True)
    out = predictor(z_prev, ego)
    out.sum().backward()
    assert ego.grad is not None and ego.grad.abs().sum() > 0, (
        "ego_motion gradient must flow through Mamba-2 predictor per Catalog #810"
    )


def test_per_pair_master_gradient_compatibility_module_parameters():
    """Trainable Mamba-2 weights must receive gradients during backward."""
    cfg = Mamba2PredictorConfig(backend="reference_torch")
    predictor = Mamba2Predictor(cfg)
    z_prev = torch.randn(4, 24)
    ego = torch.randn(4, 8)
    out = predictor(z_prev, ego)
    out.sum().backward()
    # At least one parameter must have a non-zero gradient
    grads = [p.grad for p in predictor.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads), (
        "at least one trainable param must receive gradient"
    )


# ---------------------------------------------------------------------------
# Recurrent state evolution tests
# ---------------------------------------------------------------------------


def test_stateful_mode_evolves_hidden_state_across_calls():
    cfg = Mamba2PredictorConfig(backend="reference_torch", stateful=True)
    predictor = Mamba2Predictor(cfg)
    predictor.reset_state(batch_size=1, device="cpu")

    z_seq = [torch.randn(1, 24) for _ in range(20)]
    e_seq = [torch.randn(1, 8) for _ in range(20)]
    outs = []
    for z, e in zip(z_seq, e_seq):
        with torch.no_grad():
            outs.append(predictor(z, e).detach().clone())

    # State should evolve: t=0 output != t=10 output for SAME input would
    # differ, but here we have different inputs so just verify state changed
    assert (outs[0] - outs[10]).abs().sum() > 0


def test_reset_state_zeros_hidden_state():
    cfg = Mamba2PredictorConfig(backend="reference_torch", stateful=True)
    predictor = Mamba2Predictor(cfg)
    predictor.reset_state(batch_size=2, device="cpu")
    assert predictor._h is not None
    assert predictor._h.shape == (2, cfg.d_inner, cfg.d_state)
    assert predictor._h.abs().sum().item() == 0


def test_reset_state_no_op_in_identity_mode():
    cfg = Mamba2PredictorConfig(identity_predictor=True)
    predictor = Mamba2Predictor(cfg)
    predictor.reset_state(batch_size=2, device="cpu")
    # In identity mode, _h is not allocated
    assert predictor._h is None


# ---------------------------------------------------------------------------
# Trainer scaffold integration tests
# ---------------------------------------------------------------------------


def _import_trainer_module():
    """Helper to import the trainer scaffold for testing."""
    trainer_path = REPO_ROOT / "experiments" / "train_substrate_time_traveler_l5_z7_mamba2.py"
    spec = importlib.util.spec_from_file_location(
        "train_substrate_time_traveler_l5_z7_mamba2", trainer_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trainer_module_imports_successfully():
    """Trainer scaffold must import without errors."""
    mod = _import_trainer_module()
    assert hasattr(mod, "_smoke_main")
    assert hasattr(mod, "_full_main")
    assert hasattr(mod, "main")
    assert hasattr(mod, "TIER_1_OPERATOR_REQUIRED_FLAGS")


def test_trainer_full_main_raises_notimplementederror_per_catalog_240():
    """Per Catalog #240: _full_main must raise until Wave N+1 council."""
    mod = _import_trainer_module()
    parser = mod._build_argparser()
    args = parser.parse_args([])  # No --smoke flag

    with pytest.raises(NotImplementedError, match="PRE-BUILD per Catalog #240"):
        mod._full_main(args)


def test_trainer_full_main_error_cites_z7_symposium_revision_6():
    """Error message must reference Z7 symposium Revision #6 (Wave-N+2 pivot)."""
    mod = _import_trainer_module()
    parser = mod._build_argparser()
    args = parser.parse_args([])

    try:
        mod._full_main(args)
    except NotImplementedError as e:
        msg = str(e)
        assert "Z7 parent symposium Revision #6" in msg or "Wave-N+2" in msg
        assert "Catalog #240" in msg
        assert "Catalog #315" in msg
        assert "Catalog #300" in msg, "must reference operator-frontier-override path"


def test_trainer_tier_1_operator_required_flags_present():
    """Catalog #151 manifest must declare all required Z7-Mamba-2 flags."""
    mod = _import_trainer_module()
    flags = mod.TIER_1_OPERATOR_REQUIRED_FLAGS
    required_flags = {
        "--video-path",
        "--output-dir",
        "--epochs",
        "--batch-size",
        "--lr",
        "--mamba2-d-model",
        "--mamba2-d-state",
        "--mamba2-expand",
        "--mamba2-backend",
        "--ego-source",
        "--ego-motion-dim",
        "--identity-predictor",
        "--stateful",
        "--beta-ib",
        "--smoke",
        "--device",
    }
    assert required_flags.issubset(set(flags.keys())), (
        f"Catalog #151 manifest must declare all required flags; missing: "
        f"{required_flags - set(flags.keys())}"
    )


def test_trainer_video_path_flag_is_required_input_file_per_catalog_152():
    mod = _import_trainer_module()
    video_meta = mod.TIER_1_OPERATOR_REQUIRED_FLAGS["--video-path"]
    assert video_meta.get("required_input_file") is True, (
        "Per Catalog #152: --video-path must declare required_input_file=True"
    )


def test_trainer_smoke_mode_completes_successfully(tmp_path):
    """_smoke_main must complete with rc=0 and write stats."""
    mod = _import_trainer_module()
    parser = mod._build_argparser()
    args = parser.parse_args([
        "--smoke",
        "--output-dir", str(tmp_path),
        "--device", "cpu",
    ])
    rc = mod._smoke_main(args)
    assert rc == 0
    stats_path = tmp_path / "z7_mamba2_scaffold_smoke_stats.json"
    assert stats_path.exists(), "smoke must write stats json"
    import json
    stats = json.loads(stats_path.read_text())
    assert stats["substrate_id"] == "time_traveler_l5_z7_mamba2"
    assert stats["score_claim"] is False, "scaffold smoke must NOT claim score"
    assert stats["promotion_eligible"] is False
    assert stats["ready_for_exact_eval_dispatch"] is False
    # Evidence grade must mark this as non-promotable scaffold-only signal.
    grade = stats["evidence_grade"]
    assert "scaffold" in grade and ("NOT_promotable" in grade or "not_promotable" in grade), (
        f"evidence_grade {grade!r} must mark scaffold smoke as non-promotable"
    )


# ---------------------------------------------------------------------------
# Catalog #240 recipe-vs-trainer-state consistency
# ---------------------------------------------------------------------------


def test_recipe_declares_research_only_per_catalog_240():
    """Per Catalog #240: recipe must NOT carry dispatch_enabled: true while
    _full_main raises NotImplementedError.
    """
    recipe_path = (
        REPO_ROOT / ".omx" / "operator_authorize_recipes"
        / "substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml"
    )
    assert recipe_path.exists(), f"recipe must exist at {recipe_path}"
    content = recipe_path.read_text()
    assert "research_only: true" in content, (
        "Per Catalog #240: scaffold recipe must declare research_only: true"
    )
    assert "dispatch_enabled: false" in content, (
        "Per Catalog #240: scaffold recipe must declare dispatch_enabled: false"
    )


def test_recipe_declares_predicted_band_validation_status_pending_per_catalog_324():
    """Per Catalog #324: recipe must declare predicted_band_validation_status."""
    recipe_path = (
        REPO_ROOT / ".omx" / "operator_authorize_recipes"
        / "substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml"
    )
    content = recipe_path.read_text()
    # Must declare validation_status (pending_post_training OR
    # research_prior_prebuild OR phantom_random_init etc.)
    assert "predicted_band_validation_status" in content, (
        "Per Catalog #324: recipe must declare predicted_band_validation_status"
    )


# ---------------------------------------------------------------------------
# Design memo + canonical-vs-unique decision section regression
# ---------------------------------------------------------------------------


def test_design_memo_exists_with_canonical_vs_unique_section_per_catalog_290():
    """Per Catalog #290: substrate design memo MUST have canonical-vs-unique section."""
    memo_path = REPO_ROOT / ".omx" / "research" / "z7_mamba2_substrate_design_memo_20260518.md"
    assert memo_path.exists(), f"design memo must exist at {memo_path}"
    content = memo_path.read_text()
    assert "## 6. Canonical-vs-unique decision per layer" in content or \
           "Canonical-vs-unique decision per layer" in content, (
        "Per Catalog #290: design memo MUST have Canonical-vs-unique decision per layer section"
    )


def test_design_memo_has_9_dim_checklist_section_per_catalog_294():
    """Per Catalog #294: substrate design memo MUST have 9-dimension success checklist evidence section."""
    memo_path = REPO_ROOT / ".omx" / "research" / "z7_mamba2_substrate_design_memo_20260518.md"
    content = memo_path.read_text()
    assert "## 3. 9-dimension success checklist evidence per Catalog #294" in content or \
           "9-dimension success checklist evidence" in content


def test_design_memo_has_cargo_cult_audit_section_per_catalog_303():
    memo_path = REPO_ROOT / ".omx" / "research" / "z7_mamba2_substrate_design_memo_20260518.md"
    content = memo_path.read_text()
    assert "Cargo-cult audit per assumption" in content


def test_design_memo_has_observability_surface_section_per_catalog_305():
    memo_path = REPO_ROOT / ".omx" / "research" / "z7_mamba2_substrate_design_memo_20260518.md"
    content = memo_path.read_text()
    assert "## 4. Observability surface declaration per Catalog #305" in content or \
           "Observability surface declaration" in content


def test_design_memo_has_predicted_band_with_dykstra_feasibility_per_catalog_296():
    """Per Catalog #296: predicted band must cite Dykstra-feasibility OR first-principles."""
    memo_path = REPO_ROOT / ".omx" / "research" / "z7_mamba2_substrate_design_memo_20260518.md"
    content = memo_path.read_text()
    # Must mention Dykstra-feasibility OR first-principles anchor
    assert "Dykstra-feasibility" in content, "Per Catalog #296: must cite Dykstra-feasibility"
    # And Atick-Redlich first-principles
    assert "Atick-Redlich" in content


# ---------------------------------------------------------------------------
# Local M5 Max proxy training pattern (design memo §13) regression
# ---------------------------------------------------------------------------


def test_mps_proxy_training_pattern_works_on_m5_max_if_available():
    """Per design memo §13: M5 Max MPS must be able to instantiate
    Mamba2Predictor + forward-pass for local proxy training (research-signal
    only per CLAUDE.md MPS noise non-negotiable).
    """
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available; can't test M5 Max proxy training pattern")
    cfg = Mamba2PredictorConfig(backend="reference_torch")
    predictor = Mamba2Predictor(cfg).to("mps")
    z_prev = torch.randn(4, 24, device="mps")
    ego = torch.randn(4, 8, device="mps")
    out = predictor(z_prev, ego)
    assert out.device.type == "mps"
    assert out.shape == (4, 24)
