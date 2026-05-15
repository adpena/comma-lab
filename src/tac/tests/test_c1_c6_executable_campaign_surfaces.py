# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_c1_dispatch_surfaces_default_to_identity_no_world_model() -> None:
    trainer = _read("experiments/train_substrate_c1_world_model_foveation.py")
    remote_driver = _read("scripts/remote_lane_substrate_c1_world_model_foveation.sh")
    recipe = _read(
        ".omx/operator_authorize_recipes/"
        "substrate_c1_world_model_foveation_modal_t4_smoke_dispatch.yaml"
    )

    assert "DEFAULT_RECURRENCE_MODE = WorldModelRecurrenceMode.IDENTITY_NO_WORLD_MODEL.value" in trainer
    assert "default=DEFAULT_RECURRENCE_MODE" in trainer
    assert "choices=list(RECURRENCE_MODE_TO_ID.keys())" in trainer
    assert (
        'C1_WORLD_MODEL_FOVEATION_RECURRENCE_MODE="${'
        "C1_WORLD_MODEL_FOVEATION_RECURRENCE_MODE:-identity_no_world_model"
    ) in remote_driver
    assert "C1_WORLD_MODEL_FOVEATION_RECURRENCE_MODE: identity_no_world_model" in recipe

    assert 'C1_WORLD_MODEL_FOVEATION_RECURRENCE_MODE="${C1_WORLD_MODEL_FOVEATION_RECURRENCE_MODE:-gru}"' not in remote_driver
    assert "C1_WORLD_MODEL_FOVEATION_RECURRENCE_MODE: gru" not in recipe
    assert '"default": "gru"' not in trainer


def test_c1_recipe_is_smoke_only_training_artifact_with_axis_label() -> None:
    recipe = _read(
        ".omx/operator_authorize_recipes/"
        "substrate_c1_world_model_foveation_modal_t4_smoke_dispatch.yaml"
    )

    assert "smoke_only: true" in recipe
    assert "smoke_validation_contract: training_artifact_v1" in recipe
    assert "predicted_band: [0.153, 0.173]" in recipe
    assert "predicted_band_axis:" in recipe
    assert "contest-CUDA hypothetical; score_claim=false" in recipe


def test_c6_recipe_min_smoke_gpu_blocks_stale_t4_timeout_surface() -> None:
    recipe = _read(
        ".omx/operator_authorize_recipes/"
        "substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml"
    )
    wrapper = _read("scripts/operator_authorize_substrate_c6_e4_mdl_ibps_modal_t4_dispatch.sh")

    assert 'min_smoke_gpu: "A10G"' in recipe
    assert "--recipe substrate_c6_e4_mdl_ibps_modal_t4_dispatch" in wrapper
    assert "helper upgrades to A10G per recipe min" in wrapper
    assert 'C6_E4_MDL_IBPS_ENABLE_AUTOCAST_FP16: "true"' in recipe


def test_c6_trainer_wraps_score_aware_hot_loop_in_autocast_helper() -> None:
    trainer = _read("experiments/train_substrate_c6_e4_mdl_ibps.py")

    assert "AUTOCAST_FP16_WIRED" in trainer
    assert "autocast_aware_forward as _autocast_aware_forward" in trainer
    assert "with _autocast_aware_forward(" in trainer
    assert 'enabled=bool(getattr(args, "enable_autocast_fp16", False))' in trainer


def test_c6_campaign_ledger_uses_existing_dry_run_surface_only() -> None:
    ledger = _read(".omx/research/campaign_lane_c6_e4_mdl_ibps_substrate_20260514.md")

    assert "tools/run_modal_smoke_before_full.py" in ledger
    assert "--recipe substrate_c6_e4_mdl_ibps_modal_t4_dispatch" in ledger
    assert "--dry-run" in ledger

    stale_references = (
        "experiments/train_substrate_c6_mdl_ibps.py",
        "--recipe .omx/operator_authorize_recipes/substrate_c6_e4_mdl_ibps_modal_t4_dispatch.yaml",
        "substrate_c6_procedural_decoder_modal_t4_dispatch",
        "substrate_c6_ib_regularized_modal_a100_dispatch",
        "substrate_c6_composed_full_modal_a100_dispatch",
        "tools/probe_c6_procedural_vs_ib_dominance.py",
        "--max-cost-usd",
        "--smoke-batch-size",
    )
    for stale in stale_references:
        assert stale not in ledger
