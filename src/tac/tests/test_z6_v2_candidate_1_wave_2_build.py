# SPDX-License-Identifier: MIT
"""Tests for Z6-v2 Candidate 1 Wave 2 BUILD trainer extension + recipe.

Per Phase 3 council §9 binding spec + Path B BUILD design memo §4.1 + the
Z6-v2 Candidate 1 Wave 2 BUILD landing 2026-05-17.

Coverage:
- Trainer argparse flag wiring (6 new flags)
- TIER_1_OPERATOR_REQUIRED_FLAGS extended per Catalog #151 + #168 AnnAssign
- _resolve_predictor_architecture canonical mapping
- Identity-predictor disambiguator at SAME archive bytes (Catalog #229
  paired-control marker; Council Revision #2)
- Recipe schema validation
- Catalog #270 dispatch optimization protocol PASS verdict
- Provenance contract emission
- Sister regression: Z6-v1 still works backward-compat at default depth=1
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
RECIPE_PATH = (
    REPO_ROOT
    / ".omx/operator_authorize_recipes/"
    "substrate_z6_v2_candidate_1_multi_layer_film_modal_t4_smoke_dispatch.yaml"
)
CANDIDATE4C_RECIPE_PATH = (
    REPO_ROOT
    / ".omx/operator_authorize_recipes/"
    "substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.yaml"
)
TRAINER_PATH = REPO_ROOT / "experiments/train_substrate_time_traveler_l5_z6.py"


# ===========================================================================
# Predictor-architecture resolver
# ===========================================================================


def test_resolver_single_layer_film_75k_returns_depth_1() -> None:
    """Default architecture preserves Z6-v1 backward compat."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    depth, hidden = z6._resolve_predictor_architecture("single_layer_film_75k")
    assert depth == 1
    assert hidden == 64


def test_resolver_multi_layer_film_depth_3_300k_returns_depth_3_hidden_96() -> None:
    """Council §9 binding spec: depth=3 hidden_dim=96 → ~300K total params."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    depth, hidden = z6._resolve_predictor_architecture("multi_layer_film_depth_3_300k")
    assert depth == 3
    assert hidden == 96


def test_resolver_unknown_architecture_raises_systemexit() -> None:
    """Unknown architecture rejected with SystemExit (Catalog #151 + #229)."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    with pytest.raises(SystemExit):
        z6._resolve_predictor_architecture("not_a_real_arch")


def test_predictor_width_metadata_uses_effective_hidden_dim_alias() -> None:
    """Sidecars and archive meta must expose the same effective width alias."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6

    args = z6._build_parser().parse_args([
        "--output-dir", "/tmp/_test",
        "--predictor-architecture", "single_layer_film_75k",
        "--predictor-hidden-dim", "72",
        "--predictor-film-mlp-hidden-dim", "32",
    ])

    assert z6._predictor_width_metadata(
        args,
        effective_predictor_hidden_dim=72,
        predictor_depth=1,
    ) == {
        "predictor_hidden_dim": 72,
        "requested_predictor_hidden_dim": 72,
        "effective_predictor_hidden_dim": 72,
        "predictor_film_mlp_hidden_dim": 32,
        "predictor_architecture": "single_layer_film_75k",
        "predictor_depth": 1,
    }


# ===========================================================================
# Argparse flag wiring (Phase 3 council §9 spec)
# ===========================================================================


def test_argparse_accepts_predictor_architecture_flag() -> None:
    """--predictor-architecture multi_layer_film_depth_3_300k parses."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    args = z6._build_parser().parse_args([
        "--output-dir", "/tmp/_test",
        "--predictor-architecture", "multi_layer_film_depth_3_300k",
    ])
    assert args.predictor_architecture == "multi_layer_film_depth_3_300k"


def test_argparse_default_predictor_architecture_is_z6_v1() -> None:
    """Default --predictor-architecture preserves Z6-v1 backward compat."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    args = z6._build_parser().parse_args(["--output-dir", "/tmp/_test"])
    assert args.predictor_architecture == "single_layer_film_75k"


def test_argparse_accepts_disambiguator_flag() -> None:
    """--emit-identity-predictor-disambiguator-archive parses as store_true."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    args = z6._build_parser().parse_args([
        "--output-dir", "/tmp/_test",
        "--emit-identity-predictor-disambiguator-archive",
    ])
    assert args.emit_identity_predictor_disambiguator_archive is True


def test_argparse_disambiguator_decision_criterion_delta_s_default() -> None:
    """Default decision threshold matches Council Revision #3 binding ΔS = 0.005."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    args = z6._build_parser().parse_args(["--output-dir", "/tmp/_test"])
    assert args.paired_control_disambiguator_decision_criterion_delta_s == 0.005


def test_auth_eval_json_path_resolver_supports_identity_disambiguator_suffix(
    tmp_path: Path,
) -> None:
    """Paired exact eval needs distinct durable/local JSON paths for identity."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6

    out_dir = tmp_path / "output"
    durable_root = tmp_path / "durable_auth_eval"
    gate_json, local_json = z6._resolve_auth_eval_json_paths(
        out_dir,
        durable_root=durable_root,
        filename="contest_auth_eval_identity_predictor_disambiguator.json",
    )

    assert gate_json == (
        durable_root
        / out_dir.name
        / "contest_auth_eval_identity_predictor_disambiguator.json"
    )
    assert local_json == (
        out_dir / "contest_auth_eval_identity_predictor_disambiguator.json"
    )


def test_argparse_predictor_param_count_target_default() -> None:
    """Default --predictor-param-count-target=300_000 per Council binding ceiling."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    args = z6._build_parser().parse_args(["--output-dir", "/tmp/_test"])
    assert args.predictor_param_count_target == 300_000


def test_argparse_paired_control_initialization_default() -> None:
    """Default paired-control marker matches Catalog #229 canonical."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    args = z6._build_parser().parse_args(["--output-dir", "/tmp/_test"])
    assert (
        args.enable_paired_control_initialization
        == "shared_modules_seed_order_matched_v2"
    )


def test_argparse_ego_source_default() -> None:
    """Default ego-source preserves Z6-v1 PoseNet projection per Atick Revision #6."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    args = z6._build_parser().parse_args(["--output-dir", "/tmp/_test"])
    assert args.ego_source == "posenet_projection"


def test_argparse_accepts_candidate_4c_scorer_logit_ego_source() -> None:
    """Candidate 4c scorer-logit conditioning is executable, not just prose."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    args = z6._build_parser().parse_args([
        "--output-dir", "/tmp/_test",
        "--ego-source", "scorer_logit",
    ])
    assert args.ego_source == "scorer_logit"


def test_candidate_4c_slot_allocator_preserves_all_signal_groups() -> None:
    """Default 8-dim scorer-logit buffer must not prefix-truncate a raw bank."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6

    assert z6._allocate_scorer_logit_feature_slots(8) == {
        "seg_mean": 2,
        "pose": 2,
        "entropy": 1,
        "margin": 1,
        "seg_std": 2,
    }


def test_candidate_4c_slot_metadata_is_json_safe_and_source_gated() -> None:
    """Artifact metadata should disclose the exact active side-info grammar."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6

    assert z6._scorer_logit_feature_slot_metadata(
        "posenet_projection",
        8,
    ) is None
    assert z6._scorer_logit_feature_slot_metadata("scorer_logit", 8) == {
        "seg_mean": 2,
        "pose": 2,
        "entropy": 1,
        "margin": 1,
        "seg_std": 2,
    }


def test_candidate_4c_pair_capped_param_diagnostic_uses_full_equivalent_count() -> None:
    """Pair-capped probes must not falsely mark full-run capacity underfilled."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6

    diagnostic = z6._param_count_target_diagnostic(
        {"total": 103_762, "residuals": 48},
        target=120_000,
        actual_num_pairs=2,
        latent_dim=24,
    )

    assert diagnostic["actual_total"] == 103_762
    assert diagnostic["full_equivalent_total"] == 118_114
    assert diagnostic["comparison_total"] == 118_114
    assert diagnostic["comparison_basis"] == (
        "full_equivalent_total_from_pair_capped_run"
    )
    assert diagnostic["within_5pct"] is True


def test_candidate_4c_archive_meta_contains_effective_width_fields(tmp_path: Path) -> None:
    """Archive metadata must preserve the width config used for sweep evidence."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    from tac.substrates.time_traveler_l5_z6 import (
        Z6PredictiveCodingConfig,
        Z6PredictiveCodingSubstrate,
        parse_archive,
        pack_archive,
    )

    cfg = Z6PredictiveCodingConfig(
        latent_dim=4,
        decoder_embed_dim=4,
        decoder_channels=(4,),
        decoder_num_upsample_blocks=1,
        num_pairs=2,
        output_height=6,
        output_width=8,
        predictor_hidden_dim=72,
        predictor_film_mlp_hidden_dim=32,
        predictor_ego_motion_dim=8,
    )
    sub = Z6PredictiveCodingSubstrate(cfg)
    slot_meta = z6._scorer_logit_feature_slot_metadata("scorer_logit", 8)
    args = z6._build_parser().parse_args([
        "--output-dir", str(tmp_path),
        "--predictor-architecture", "single_layer_film_75k",
        "--predictor-hidden-dim", "72",
        "--predictor-film-mlp-hidden-dim", "32",
    ])
    width_meta = z6._predictor_width_metadata(
        args,
        effective_predictor_hidden_dim=cfg.predictor_hidden_dim,
        predictor_depth=1,
    )
    blob = pack_archive(
        sub.encoder.state_dict(),
        sub.decoder.state_dict(),
        sub.predictor.state_dict(),
        sub.latent_init.detach().cpu(),
        sub.residuals.detach().cpu(),
        sub.ego_motion_buffer.detach().cpu(),
        {
            "decoder_embed_dim": cfg.decoder_embed_dim,
            "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
            "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
            "decoder_channels": list(cfg.decoder_channels),
            "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
            "output_height": cfg.output_height,
            "output_width": cfg.output_width,
            **width_meta,
            "scorer_logit_feature_slot_allocation": slot_meta,
        },
    )

    archive_path = tmp_path / "0.bin"
    archive_path.write_bytes(blob)
    arc = parse_archive(blob)
    assert arc.meta["predictor_hidden_dim"] == 72
    assert arc.meta["requested_predictor_hidden_dim"] == 72
    assert arc.meta["effective_predictor_hidden_dim"] == 72
    assert arc.meta["predictor_film_mlp_hidden_dim"] == 32
    assert arc.meta["predictor_architecture"] == "single_layer_film_75k"
    assert arc.meta["predictor_depth"] == 1
    assert arc.meta["scorer_logit_feature_slot_allocation"] == slot_meta


def test_candidate_4c_feature_reduction_keeps_pose_entropy_margin() -> None:
    """Regression for scorer-logit concat-then-truncate dropping key signals."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6

    batch = 2
    reduced = z6._reduce_scorer_logit_feature_groups(
        seg_mean=torch.full((batch, 12), 10.0),
        pose_tensor=torch.full((batch, 12), 100.0),
        seg_std=torch.full((batch, 12), 200.0),
        entropy=torch.full((batch, 1), 300.0),
        margin=torch.full((batch, 1), 400.0),
        ego_motion_dim=8,
    )

    assert tuple(reduced.shape) == (batch, 8)
    assert torch.equal(reduced[:, 0:2], torch.full((batch, 2), 10.0))
    assert torch.equal(reduced[:, 2:4], torch.full((batch, 2), 100.0))
    assert torch.equal(reduced[:, 4:6], torch.full((batch, 2), 200.0))
    assert torch.equal(reduced[:, 6:7], torch.full((batch, 1), 300.0))
    assert torch.equal(reduced[:, 7:8], torch.full((batch, 1), 400.0))


class _FakePoseNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.preprocess_calls = 0

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        self.preprocess_calls += 1
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        base = x.mean(dim=(1, 2, 3, 4), keepdim=False).reshape(-1, 1)
        offsets = torch.arange(12, device=x.device, dtype=x.dtype).reshape(1, 12)
        return {"pose": base + offsets}


class _FakeSegNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.preprocess_calls = 0

    def preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        self.preprocess_calls += 1
        return x[:, -1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.shape[0]
        base = x.mean(dim=(1, 2, 3), keepdim=True).reshape(batch, 1, 1, 1)
        class_offsets = torch.arange(5, device=x.device, dtype=x.dtype).reshape(
            1, 5, 1, 1
        )
        spatial = torch.arange(4, device=x.device, dtype=x.dtype).reshape(1, 1, 2, 2)
        return base + class_offsets + spatial


def test_candidate_4c_scorer_logit_ego_motion_uses_both_scorers() -> None:
    """Scorer-logit side-info reduces SegNet logits + PoseNet head to ego buffer."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    posenet = _FakePoseNet()
    segnet = _FakeSegNet()
    pairs = torch.arange(4 * 2 * 3 * 4 * 4, dtype=torch.float32).reshape(
        4, 2, 3, 4, 4
    )

    ego = z6._derive_ego_motion_from_scorer_logits(
        posenet,
        segnet,
        pairs,
        ego_motion_dim=8,
        chunk_size=2,
        device=torch.device("cpu"),
    )

    assert tuple(ego.shape) == (4, 8)
    assert torch.isfinite(ego).all()
    assert float(ego.abs().sum()) > 0.0
    assert posenet.preprocess_calls == 2
    assert segnet.preprocess_calls == 2


# ===========================================================================
# TIER_1_OPERATOR_REQUIRED_FLAGS manifest (Catalog #151 + #168 AnnAssign)
# ===========================================================================


def test_tier1_required_flags_contains_all_6_wave_2_build_flags() -> None:
    """Phase 3 council §9 required_flags surfaced in TIER_1 manifest."""
    import experiments.train_substrate_time_traveler_l5_z6 as z6
    required = {
        "--predictor-architecture",
        "--predictor-param-count-target",
        "--predictor-hidden-dim",
        "--predictor-film-mlp-hidden-dim",
        "--ego-source",
        "--enable-paired-control-initialization",
        "--emit-identity-predictor-disambiguator-archive",
        "--paired-control-disambiguator-decision-criterion-delta-s",
    }
    flags = set(z6.TIER_1_OPERATOR_REQUIRED_FLAGS.keys())
    missing = required - flags
    assert not missing, f"missing TIER_1 manifest entries: {missing}"


def test_tier1_required_flags_uses_annassign_per_catalog_168() -> None:
    """The dict is declared with type annotation (AST AnnAssign per Catalog #168)."""
    import ast
    source = TRAINER_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    found_annassign = False
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign):
            target = node.target
            if (
                isinstance(target, ast.Name)
                and target.id == "TIER_1_OPERATOR_REQUIRED_FLAGS"
            ):
                found_annassign = True
                break
    assert found_annassign, (
        "TIER_1_OPERATOR_REQUIRED_FLAGS must be declared as ast.AnnAssign per "
        "Catalog #168 (bare ast.Assign is silently skipped by the AST walker)"
    )


# ===========================================================================
# Recipe schema validation
# ===========================================================================


def test_recipe_yaml_file_exists() -> None:
    assert RECIPE_PATH.is_file(), f"recipe missing at {RECIPE_PATH}"


def test_recipe_yaml_loads_and_has_required_fields() -> None:
    """Per Phase 3 council §9 spec: recipe declares all required canonical fields."""
    import yaml
    recipe = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))
    # Identity + lifecycle
    assert recipe["name"] == "substrate_z6_v2_candidate_1_multi_layer_film_modal_t4_smoke_dispatch"
    assert recipe["lane_id"].startswith("lane_z6_v2_candidate_1_wave_2_build")
    # The recipe was flipped after operator approval; dispatch is still guarded
    # by operator_authorize/probe-predecessor checks before provider creation.
    assert recipe["research_only"] is False
    assert recipe["dispatch_enabled"] is True
    # Per Catalog #270 dispatch optimization protocol Tier 2/3
    assert recipe["platform"] == "modal"
    assert recipe["min_vram_gb"] >= 14
    assert recipe["min_smoke_gpu"] == "T4"
    assert recipe["video_input_strategy"] == "per_dispatch_local_copy"
    assert recipe["pyav_decode_strategy"] == "cpu_thread_async_upload"
    assert isinstance(recipe["target_modes"], list) and recipe["target_modes"]
    assert recipe["canary_status"] == "post_canary_dependent"
    assert recipe["canary_dependency"] == "time_traveler_l5_z6"
    # Catalog #167 smoke-before-full
    assert recipe["smoke_before_full"] is True
    # Catalog #309 horizon_class
    assert recipe["horizon_class"] == "frontier_pursuit"
    # Catalog #220 + #272 distinguishing-feature integration contract
    assert recipe["distinguishing_feature_name"] == "multi_layer_film_predictor_depth_3"
    assert recipe["distinguishing_bytes_path"] == "predictor_blob"
    # Predicted band matches Path B design memo §4.1 + Phase 3 council §5
    assert recipe["predicted_band"] == [0.13, 0.17]
    # Cost band Wave 2 envelope
    cost_band = recipe["cost_band"]
    assert cost_band["epochs"] == 100
    assert cost_band["predicted_cost_usd"] == 3.0


def test_recipe_yaml_declares_all_6_tier1_required_flags() -> None:
    """Per Catalog #151: operator_required_flags must enumerate every Wave 2 flag."""
    import yaml
    recipe = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))
    required = recipe["operator_required_flags"]["TIER_1_OPERATOR_REQUIRED_FLAGS"]
    expected = {
        "--enable-paired-control-initialization",
        "--emit-identity-predictor-disambiguator-archive",
        "--predictor-architecture",
        "--predictor-param-count-target",
        "--ego-source",
        "--paired-control-disambiguator-decision-criterion-delta-s",
    }
    missing = expected - set(required)
    assert not missing, f"recipe operator_required_flags missing: {missing}"


def test_recipe_yaml_env_overrides_thread_all_tier1_required_flags() -> None:
    """Per Catalog #151 env→CLI ladder + Catalog #152 required-input file."""
    import yaml
    recipe = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))
    env = recipe["env_overrides"]
    for required_env in (
        "Z6_PREDICTOR_ARCHITECTURE",
        "Z6_PREDICTOR_PARAM_COUNT_TARGET",
        "Z6_PREDICTOR_HIDDEN_DIM",
        "Z6_PREDICTOR_FILM_MLP_HIDDEN_DIM",
        "Z6_EGO_SOURCE",
        "Z6_ENABLE_PAIRED_CONTROL_INITIALIZATION",
        "Z6_EMIT_IDENTITY_PREDICTOR_DISAMBIGUATOR_ARCHIVE",
        "Z6_PAIRED_CONTROL_DISAMBIGUATOR_DECISION_CRITERION_DELTA_S",
    ):
        assert required_env in env, f"env_overrides missing {required_env}"
    # The canonical Wave 2 architecture is multi_layer_film_depth_3_300k
    assert env["Z6_PREDICTOR_ARCHITECTURE"] == "multi_layer_film_depth_3_300k"


def test_recipe_yaml_dispatch_blockers_cleared_after_operator_approval() -> None:
    """Wave 2 recipe blockers are cleared; authorize/probe gates still guard spend."""
    import yaml
    recipe = yaml.safe_load(RECIPE_PATH.read_text(encoding="utf-8"))
    assert recipe["dispatch_blockers"] == []
    assert recipe["research_only"] is False
    assert recipe["dispatch_enabled"] is True


def test_candidate_4c_recipe_yaml_loads_as_distinct_diagnostic_lane() -> None:
    """Candidate 4c is a separate scorer-logit diagnostic, not a Candidate 1 edit."""
    import yaml
    assert CANDIDATE4C_RECIPE_PATH.is_file(), (
        f"Candidate 4c recipe missing at {CANDIDATE4C_RECIPE_PATH}"
    )
    recipe = yaml.safe_load(CANDIDATE4C_RECIPE_PATH.read_text(encoding="utf-8"))

    assert (
        recipe["name"]
        == "substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch"
    )
    assert recipe["lane_id"] == (
        "lane_z6_v2_candidate_4c_scorer_logit_conditioning_20260518"
    )
    assert recipe["research_only"] is False
    assert recipe["dispatch_enabled"] is False
    assert recipe["dispatch_blockers"] == [
        "candidate4c_modal_training_recipe_is_diagnostic_only_exact_cuda_handoff_required"
    ]
    assert recipe["smoke_only"] is True
    assert recipe["smoke_validation_contract"] == "training_artifact_v1"
    assert "contest_exact_eval" not in recipe["target_modes"]
    assert recipe["distinguishing_feature_name"] == "scorer_logit_ego_source"
    assert recipe["distinguishing_bytes_path"] == "scorer_logit_ego_motion"
    assert recipe["distinguishing_byte_range"] == "0.bin@192390:16"
    assert recipe["byte_mutation_smoke_passes"].endswith(
        "scorer_logit_ego_motion_byte_mutation_proof.json"
    )
    assert recipe["smoke_before_full"] is True
    assert recipe["horizon_class"] == "asymptotic_pursuit"
    assert recipe["predicted_band"] == [0.11, 0.17]
    assert recipe["predicted_band_validation_status"] == "pending_post_training"
    assert "Catalog #324" in recipe["predicted_band_reactivation_criteria"]
    assert "planning prior" in recipe["predicted_band_reactivation_criteria"]
    assert "full-vs-identity disambiguator" in recipe[
        "predicted_band_reactivation_criteria"
    ]
    assert recipe["smoke_score_band"] == [0.13, 0.25]
    assert "full_minus_identity_score <= -0.005" in recipe["pact_must_prove"]
    assert "identity_minus_full_score >= 0.005" in recipe["pact_must_prove"]
    assert "lower score wins" in recipe["pact_must_prove"]

    cost_band = recipe["cost_band"]
    assert cost_band["epochs"] == 100
    assert cost_band["predicted_cost_usd"] == 1.25
    assert cost_band["hand_calibrated_fallback_p50_usd"] == 1.0
    assert cost_band["future_full_plus_paired_exact_eval_envelope_usd"] == 13.0


def test_candidate_4c_recipe_threads_scorer_logit_env_ladder() -> None:
    """The recipe must dispatch the scorer-logit side-info hypothesis explicitly."""
    import yaml
    recipe = yaml.safe_load(CANDIDATE4C_RECIPE_PATH.read_text(encoding="utf-8"))
    env = recipe["env_overrides"]

    assert env["Z6_LANE_ID"] == recipe["lane_id"]
    assert env["Z6_RECIPE_PATH"] == (
        ".omx/operator_authorize_recipes/"
        "substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch.yaml"
    )
    assert env["Z6_EGO_SOURCE"] == "scorer_logit"
    assert env["Z6_PREDICTOR_ARCHITECTURE"] == "single_layer_film_75k"
    assert env["Z6_PREDICTOR_PARAM_COUNT_TARGET"] == "120000"
    assert env["Z6_PREDICTOR_HIDDEN_DIM"] == "64"
    assert env["Z6_PREDICTOR_FILM_MLP_HIDDEN_DIM"] == "32"
    assert env["Z6_EMIT_IDENTITY_PREDICTOR_DISAMBIGUATOR_ARCHIVE"] == "true"
    assert env["Z6_MAX_PAIRS"] == "64"
    assert env["Z6_SKIP_AUTH_EVAL"] == "1"
    assert env["Z6_TRAINER_MODE"] == "full"
    assert env["SMOKE_ONLY"] == "0"
    assert recipe["modal"]["cost_band_epochs"] == 100


def test_z6_remote_driver_pair_cap_skips_auth_eval() -> None:
    """Pair-capped Candidate 4c diagnostic smokes must not invoke auth-eval."""

    driver = (
        REPO_ROOT / "scripts" / "remote_lane_substrate_time_traveler_l5_z6.sh"
    ).read_text(encoding="utf-8")

    assert 'Z6_MAX_PAIRS="${Z6_MAX_PAIRS:-}"' in driver
    assert 'Z6_SKIP_AUTH_EVAL="${Z6_SKIP_AUTH_EVAL:-}"' in driver
    assert 'MAX_PAIRS_ARGS+=(--max-pairs "$Z6_MAX_PAIRS")' in driver
    assert "stage_4_pair_capped_smoke_skips_auth_eval" in driver
    assert "AUTH_EVAL_ARGS+=(--skip-auth-eval)" in driver
    assert '${MAX_PAIRS_ARGS[@]+"${MAX_PAIRS_ARGS[@]}"}' in driver
    assert '${AUTH_EVAL_ARGS[@]+"${AUTH_EVAL_ARGS[@]}"}' in driver


def test_full_main_runs_paired_identity_auth_eval_when_disambiguator_emits() -> None:
    """Candidate 4c exact arbitration must not require hand-running identity eval."""
    import inspect

    import experiments.train_substrate_time_traveler_l5_z6 as z6

    source = inspect.getsource(z6._full_main)
    assert "contest_auth_eval_identity_predictor_disambiguator.json" in source
    assert "identity_auth_eval_result = _canon_gate_auth_eval_call" in source
    assert "identity_disambiguator_archive_zip_path" in source
    assert "identity_disambiguator_auth_eval_cuda_done_valid_claim" in source


# ===========================================================================
# Catalog #270 dispatch optimization protocol PASS
# ===========================================================================


def test_recipe_passes_catalog_270_dispatch_optimization_protocol() -> None:
    """Per CLAUDE.md "Production-hardened dispatch optimization protocol" non-negotiable:
    Tier 1 + Tier 2 + Tier 3 all PASS; 0 blockers; overall_pass: true."""
    tool = REPO_ROOT / "tools/canonical_dispatch_optimization_protocol.py"
    if not tool.is_file():
        pytest.skip(f"protocol tool missing at {tool}")
    result = subprocess.run(
        [
            sys.executable, str(tool),
            "--trainer", str(TRAINER_PATH),
            "--recipe", "substrate_z6_v2_candidate_1_multi_layer_film_modal_t4_smoke_dispatch",
            "--json",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
    )
    assert result.returncode == 0, f"protocol returned rc={result.returncode}; stderr={result.stderr}"
    verdict = json.loads(result.stdout)
    assert verdict["overall_pass"] is True, (
        f"Catalog #270 protocol FAILED for Z6-v2 Candidate 1 Wave 2 recipe; "
        f"blockers: {verdict.get('blockers', [])}"
    )
    # Per-tier zero-blockers contract
    assert verdict["tier1"]["blockers"] == []
    assert verdict["tier2"]["blockers"] == []
    assert verdict["tier3"]["blockers"] == []


def test_candidate_4c_recipe_passes_dispatch_optimization_protocol() -> None:
    """Candidate 4c must stay launchable through the same Catalog #270 gate."""
    tool = REPO_ROOT / "tools/canonical_dispatch_optimization_protocol.py"
    if not tool.is_file():
        pytest.skip(f"protocol tool missing at {tool}")
    result = subprocess.run(
        [
            sys.executable, str(tool),
            "--trainer", str(TRAINER_PATH),
            "--recipe", (
                "substrate_z6_v2_candidate_4c_scorer_logit_modal_t4_smoke_dispatch"
            ),
            "--json",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
    )
    assert result.returncode == 0, (
        f"protocol returned rc={result.returncode}; stderr={result.stderr}"
    )
    verdict = json.loads(result.stdout)
    assert verdict["overall_pass"] is True, (
        "Catalog #270 protocol FAILED for Z6-v2 Candidate 4c scorer-logit "
        f"recipe; blockers: {verdict.get('blockers', [])}"
    )
    assert verdict["tier1"]["blockers"] == []
    assert verdict["tier2"]["blockers"] == []
    assert verdict["tier3"]["blockers"] == []


# ===========================================================================
# Identity-predictor disambiguator at SAME archive bytes (Council Revision #2)
# ===========================================================================


def test_smoke_disambiguator_emits_two_archives_at_same_bytes(tmp_path: Path) -> None:
    """Per Phase 3 council Revision #2 + Catalog #229 paired-control marker:
    --emit-identity-predictor-disambiguator-archive emits BOTH archives so the
    Wave 2 disambiguator probe can compute ΔS = full_FiLM_score - identity_score
    at the SAME archive bytes (same encoder + decoder + latent_init + residuals
    + ego_motion; ONLY identity_predictor=True changes)."""
    result = subprocess.run(
        [
            sys.executable, str(TRAINER_PATH),
            "--output-dir", str(tmp_path),
            "--smoke",
            "--device", "cpu",
            "--predictor-architecture", "multi_layer_film_depth_3_300k",
            "--emit-identity-predictor-disambiguator-archive",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=120,
        env={"PYTHONPATH": "src:upstream", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, (
        f"smoke rc={result.returncode}; stderr={result.stderr[-500:]}"
    )
    # Both archives present
    full_archive = tmp_path / "0.bin"
    disambiguator_archive = tmp_path / "0_identity_predictor_disambiguator.bin"
    assert full_archive.is_file(), "full-predictor archive 0.bin missing"
    assert disambiguator_archive.is_file(), (
        "identity-predictor disambiguator archive missing"
    )
    # Stats record the disambiguator emission
    stats = json.loads((tmp_path / "stats.json").read_text(encoding="utf-8"))
    assert stats["emit_identity_predictor_disambiguator_archive"] is True
    assert stats["identity_predictor_disambiguator_archive_bytes"] is not None
    assert stats["identity_predictor_disambiguator_archive_bytes"] > 0
    # Council Revision #3 decision criterion ΔS = 0.005
    assert stats["paired_control_disambiguator_decision_criterion_delta_s"] == 0.005
    # Wave 2 BUILD architecture recorded
    assert stats["predictor_architecture"] == "multi_layer_film_depth_3_300k"
    assert stats["predictor_depth"] == 3


def test_full_pair_capped_disambiguator_emits_identity_archive(tmp_path: Path) -> None:
    """Full path must honor the disambiguator flag, not only smoke mode."""
    video_path = REPO_ROOT / "upstream" / "videos" / "0.mkv"
    if not video_path.is_file():
        pytest.skip(f"missing local contest video: {video_path}")

    result = subprocess.run(
        [
            sys.executable, str(TRAINER_PATH),
            "--video-path", str(video_path),
            "--output-dir", str(tmp_path),
            "--epochs", "1",
            "--batch-size", "1",
            "--device", "cpu",
            "--full-cpu",
            "--advisory-cpu-explicitly-waived",
            "--max-pairs", "2",
            "--skip-auth-eval",
            "--ego-source", "scorer_logit",
            "--predictor-architecture", "single_layer_film_75k",
            "--predictor-hidden-dim", "72",
            "--predictor-film-mlp-hidden-dim", "32",
            "--predictor-param-count-target", "120000",
            "--predictor-ego-motion-dim", "8",
            "--emit-identity-predictor-disambiguator-archive",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=180,
        env={"PYTHONPATH": "src:upstream", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, (
        f"full pair-capped rc={result.returncode}; stderr={result.stderr[-500:]}"
    )

    from tac.substrates.time_traveler_l5_z6.archive import parse_archive

    full_archive = tmp_path / "0.bin"
    identity_archive = tmp_path / "0_identity_predictor_disambiguator.bin"
    identity_zip = tmp_path / "archive_identity_predictor_disambiguator.zip"
    assert full_archive.is_file()
    assert identity_archive.is_file()
    assert identity_zip.is_file()

    full_arc = parse_archive(full_archive.read_bytes())
    identity_arc = parse_archive(identity_archive.read_bytes())
    assert (
        full_arc.meta["predictive_coding_world_model_meta"]["identity_predictor"]
        is False
    )
    assert (
        identity_arc.meta["predictive_coding_world_model_meta"]["identity_predictor"]
        is True
    )
    assert identity_arc.meta["identity_predictor_disambiguator"] is True
    assert identity_arc.predictor_state_dict.keys() == full_arc.predictor_state_dict.keys()

    provenance = json.loads((tmp_path / "provenance.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    stats = json.loads((tmp_path / "stats.json").read_text(encoding="utf-8"))
    assert provenance["identity_predictor_disambiguator_archive_bytes"] > 0
    assert provenance["identity_predictor_disambiguator_archive_sha256"]
    assert (
        provenance["identity_predictor_disambiguator_auth_eval_json_path"]
        .endswith("contest_auth_eval_identity_predictor_disambiguator.json")
    )
    assert manifest["identity_predictor_disambiguator_archive_bytes"] > 0
    assert (
        manifest["identity_predictor_disambiguator_auth_eval_json_path"]
        .endswith("contest_auth_eval_identity_predictor_disambiguator.json")
    )
    assert manifest["score_claim"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert stats["stats_schema"] == "time_traveler_l5_z6_full_stats_v1"
    assert stats["paired_identity_auth_eval_required"] is True
    assert stats["primary_auth_eval_score_claim_valid"] is False
    assert stats["identity_predictor_disambiguator_auth_eval_score_claim_valid"] is False
    assert stats["auth_eval_score_claim_valid"] is False


def test_smoke_default_does_not_emit_disambiguator(tmp_path: Path) -> None:
    """Backward compat: omitting --emit-identity-predictor-disambiguator-archive
    preserves Z6-v1 single-archive behavior."""
    result = subprocess.run(
        [
            sys.executable, str(TRAINER_PATH),
            "--output-dir", str(tmp_path),
            "--smoke",
            "--device", "cpu",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
        env={"PYTHONPATH": "src:upstream", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, f"smoke rc={result.returncode}; stderr={result.stderr}"
    full_archive = tmp_path / "0.bin"
    disambiguator_archive = tmp_path / "0_identity_predictor_disambiguator.bin"
    assert full_archive.is_file()
    assert not disambiguator_archive.is_file(), (
        "disambiguator emitted without --emit-identity-predictor-disambiguator-archive"
    )


# ===========================================================================
# Sister regression: Z6-v1 backward compat
# ===========================================================================


def test_z6_v1_smoke_still_works_at_default_depth_1(tmp_path: Path) -> None:
    """Default --predictor-architecture=single_layer_film_75k preserves Z6-v1.

    Per CLAUDE.md "Subagent coherence-by-default": the Wave 2 BUILD extension
    MUST NOT break the Z6-v1 dispatch path (sister `lane_time_traveler_l5_z6_l1_
    scaffold_substrate_build_20260516` still depends on the default behavior).
    """
    result = subprocess.run(
        [
            sys.executable, str(TRAINER_PATH),
            "--output-dir", str(tmp_path),
            "--smoke",
            "--device", "cpu",
        ],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=60,
        env={"PYTHONPATH": "src:upstream", "PATH": "/usr/bin:/bin"},
    )
    assert result.returncode == 0, (
        f"Z6-v1 default smoke regressed; rc={result.returncode}; "
        f"stderr={result.stderr[-500:]}"
    )
    stats = json.loads((tmp_path / "stats.json").read_text(encoding="utf-8"))
    assert stats["predictor_depth"] == 1
    assert stats["predictor_architecture"] == "single_layer_film_75k"
    assert stats["emit_identity_predictor_disambiguator_archive"] is False
