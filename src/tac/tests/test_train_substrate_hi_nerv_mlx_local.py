# SPDX-License-Identifier: MIT
from __future__ import annotations

import ast
import builtins
import inspect
import json
from pathlib import Path

import pytest

from comma_lab.storage_tiers import StorageTierError
from experiments.train_substrate_hi_nerv_mlx_local import (
    DIRECT_TRAINER_CANONICALIZATION_SCHEMA,
    DIRECT_TRAINER_LAUNCH_REFUSAL_SCHEMA,
    HI_NERV_HARD_BYTE_CEILING_CONTROL_SCHEMA,
    HI_NERV_MODELSIZE_CANDIDATE_CONSUMPTION_SCHEMA,
    HI_NERV_SHORT_SCORER_SMOKE_READINESS_SCHEMA,
    HI_NERV_TRAIN_TIME_CONTROL_SCHEMA,
    HI_NERV_TRAIN_TIME_DECODER_MUTATION_IDENTITY_SCHEMA,
    PR95_FULL_CONTROL_CONTRACT_SCHEMA,
    TRAINER_SCHEMA,
    HiNervTrainTimeControlConfig,
    _apply_train_time_decoder_controls,
    _attach_hinerv_short_scorer_smoke_readiness_to_training_artifact,
    _build_hi_nerv_train_time_section_byte_metrics_callback,
    _build_hinerv_hard_byte_ceiling_control,
    _build_hinerv_short_scorer_smoke_readiness_report,
    _build_parser,
    _build_staged_scorer_curriculum,
    _build_train_time_decoder_mutation_identity,
    _checkpoint_retention_keep_last_n_from_args,
    _coder_qat_config_from_args,
    _config_from_args,
    _configure_decoder_fake_quant_forward,
    _curriculum_stages_from_args,
    _decoder_codec_from_args,
    _decoder_weight_waterfill_plan_attachment_metadata,
    _decoder_weight_waterfill_plan_from_args,
    _direct_trainer_canonicalization_contract,
    _full_main,
    _gradient_multiplier_by_name_from_args,
    _hard_byte_ceiling_from_args,
    _hard_byte_ceiling_from_modelsize_candidate,
    _hinerv_short_scorer_smoke_readiness_summary,
    _maybe_write_post_export_receiver_cache_quality,
    _metadata_safe,
    _modelsize_candidate_consumption_metadata,
    _modelsize_candidate_from_args,
    _optimizer_control_metadata_from_args,
    _pair_sampling_weights_from_args,
    _pose_student_input_channels,
    _pr95_full_control_contract,
    _prioritized_pair_indices_from_args,
    _prioritized_pair_training_lineage_metadata,
    _prioritized_pair_training_metadata,
    _receiver_cache_quality_manifest_summary,
    _resolve_output_dir,
    _smoke_forward_statistics,
    _smoke_main,
    _train_time_control_config_from_args,
    _train_time_dual_ascent_config_from_args,
    _validate_shared_harness_train_time_actuator_args,
)
from tac.analysis.nerv_modelsize_budget import analyze_hinerv_modelsize_candidate
from tac.repo_io import sha256_file
from tac.substrates._shared.mlx_score_aware.adapter import (
    DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND,
)


def test_hinerv_mlx_trainer_binds_modelsize_row_and_overrides() -> None:
    args = _build_parser().parse_args(
        [
            "--smoke",
            "--modelsize-row",
            "hi_nerv_local_small",
            "--num-pairs",
            "7",
            "--decoder-channels",
            "9,8,7,6,5,4,3",
            "--latent-dim-coarse",
            "11",
            "--output-height",
            "96",
            "--output-width",
            "128",
            "--seed",
            "37",
        ]
    )

    cfg = _config_from_args(args)

    assert cfg.num_pairs == 7
    assert cfg.latent_dim_coarse == 11
    assert cfg.latent_dim_mid == 15
    assert cfg.latent_dim_fine == 18
    assert cfg.embed_dim == 48
    assert cfg.decoder_channels == (9, 8, 7, 6, 5, 4, 3)
    assert cfg.output_height == 96
    assert cfg.output_width == 128
    assert cfg.init_seed == 37


def test_hinerv_smoke_forward_statistics_records_target_initialized_head_error() -> None:
    import numpy as np

    target0_channel = np.array([0.10, 0.20, 0.30], dtype=np.float32)
    target1_channel = np.array([0.60, 0.70, 0.80], dtype=np.float32)
    target0 = np.broadcast_to(target0_channel, (1, 2, 2, 3)).copy()
    target1 = np.broadcast_to(target1_channel, (1, 2, 2, 3)).copy()
    output = np.zeros((1, 2, 3, 2, 2), dtype=np.float32)
    output[0, 0] = (target0_channel * 255.0).reshape(3, 1, 1)
    output[0, 1] = (target1_channel * 255.0).reshape(3, 1, 1)

    stats = _smoke_forward_statistics(
        output=output,
        target_rgb_0=target0,
        target_rgb_1=target1,
    )

    assert stats["output_std"] > 0.0
    assert stats["target_std_255"] > 0.0
    assert stats["target_mean_abs_error_after_bias_init"] == pytest.approx(0.0)
    assert stats["target_channel_mean_abs_error_after_bias_init_255"][0] == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    assert stats["target_channel_mean_abs_error_after_bias_init_255"][1] == pytest.approx(
        [0.0, 0.0, 0.0]
    )
    assert stats["target_channel_means_255"][0] == pytest.approx([25.5, 51.0, 76.5])
    assert stats["target_channel_means_255"][1] == pytest.approx([153.0, 178.5, 204.0])
    assert stats["output_channel_means_255"][0] == pytest.approx(
        stats["target_channel_means_255"][0]
    )
    assert stats["output_channel_means_255"][1] == pytest.approx(
        stats["target_channel_means_255"][1]
    )
    assert stats["neutral_gray_global_abs_error_255"] > 10.0
    assert max(stats["neutral_gray_channel_abs_error_255"][0]) > 50.0


def test_hinerv_smoke_main_initializes_decoded_target_head_before_forward() -> None:
    source = inspect.getsource(_smoke_main)

    decode_pos = source.index("decode_mlx_targets(")
    init_pos = source.index("_initialize_output_head_target_bias(")
    contrast_pos = source.index("output_head_target_contrast_init =")
    forward_pos = source.index("output = model(idx)")
    stats_pos = source.index("_smoke_forward_statistics(")

    assert decode_pos < init_pos < contrast_pos < forward_pos < stats_pos
    assert "\"output_head_target_contrast_init\"" in source
    assert "hinerv_smoke_missing_output_head_target_contrast_init" in source


def test_hinerv_full_main_short_readiness_consumes_strict_launch_actuators() -> None:
    source = inspect.getsource(_full_main)
    call = source[
        source.index("short_scorer_smoke_readiness =") :
        source.index("_attach_hinerv_short_scorer_smoke_readiness_to_training_artifact")
    ]

    assert "require_section_byte_dual_ascent=modelsize_hard_byte_ceiling is not None" in call
    assert "require_pose_direct_live_distillation=True" in call
    assert "decoder_weight_waterfill_plan_metadata=(" in call
    assert "output_head_target_bias_init_metadata=output_head_target_bias_init" in call


def test_hinerv_mlx_trainer_consumes_modelsize_candidate_for_config_codec_and_byte_cap(
    tmp_path: Path,
) -> None:
    candidate = analyze_hinerv_modelsize_candidate(
        hard_byte_ceiling=500_000,
        num_pairs=7,
        latent_dim=10,
        embed_dim=16,
        decoder_channel=8,
        decoder_codec="int4_mixed",
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
        local_grid_levels=3,
        local_grid_channels=5,
        convnext_mlp_ratio=3,
        convnext_kernel_size=5,
        mid_injection_block_index=1,
        fine_injection_block_index=4,
    ).as_dict()
    assert candidate["nominal_under_ceiling"] is True
    candidate_path = tmp_path / "hinerv_candidate.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    args = _build_parser().parse_args(
        [
            "--smoke",
            "--num-pairs",
            "7",
            "--modelsize-candidate-json",
            candidate_path.as_posix(),
            "--seed",
            "41",
        ]
    )

    loaded = _modelsize_candidate_from_args(args)
    assert loaded is not None
    cfg = _config_from_args(args, modelsize_candidate=loaded)
    assert cfg.init_seed == 41
    consumption = _modelsize_candidate_consumption_metadata(
        args=args,
        candidate=loaded,
    )
    canonicalization = _direct_trainer_canonicalization_contract(
        mode="smoke",
        modelsize_candidate_consumption=consumption,
    )

    assert cfg.num_pairs == 7
    assert cfg.latent_dim_coarse == 5
    assert cfg.latent_dim_mid == 10
    assert cfg.latent_dim_fine == 20
    assert cfg.embed_dim == 16
    assert cfg.decoder_channels == (8, 8, 8, 8, 8, 8, 8)
    assert cfg.use_hierarchical_feature_grid is True
    assert cfg.use_convnext_blocks is True
    assert cfg.local_grid_levels == 3
    assert cfg.local_grid_channels == 5
    assert cfg.convnext_mlp_ratio == 3
    assert cfg.convnext_kernel_size == 5
    assert _decoder_codec_from_args(args, modelsize_candidate=loaded) == "int4_mixed"
    assert _hard_byte_ceiling_from_modelsize_candidate(loaded) == 500_000
    assert consumption["schema"] == HI_NERV_MODELSIZE_CANDIDATE_CONSUMPTION_SCHEMA
    assert consumption["attached"] is True
    assert consumption["consumed_by_trainer_config"] is True
    assert consumption["consumed_by_decoder_codec"] is True
    assert consumption["consumed_by_archive_export_hard_byte_ceiling"] is True
    assert consumption["sha256"] == sha256_file(candidate_path)
    assert consumption["decoder_codec"] == "int4_mixed"
    assert consumption["hard_byte_ceiling"] == 500_000
    assert "control_precedence" in consumption["modelsize_control_contract"]
    assert canonicalization["modelsize_candidate_contract_consumed"] is True
    assert (
        "hinerv_direct_modelsize_row_not_budget_candidate_contract"
        not in canonicalization["blockers"]
    )
    assert "hinerv_direct_trainer_missing_planner_row_id" in canonicalization["blockers"]
    assert canonicalization["score_claim"] is False


def test_hinerv_mlx_trainer_rejects_over_cap_modelsize_candidate(
    tmp_path: Path,
) -> None:
    candidate = analyze_hinerv_modelsize_candidate(
        hard_byte_ceiling=1,
        num_pairs=7,
        latent_dim=10,
        embed_dim=16,
        decoder_channel=8,
        decoder_codec="int4_mixed",
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
    ).as_dict()
    assert candidate["nominal_under_ceiling"] is False
    candidate_path = tmp_path / "hinerv_candidate_over_cap.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    args = _build_parser().parse_args(
        [
            "--smoke",
            "--num-pairs",
            "7",
            "--modelsize-candidate-json",
            candidate_path.as_posix(),
        ]
    )

    with pytest.raises(ValueError, match="nominally_over_hard_byte_ceiling"):
        _modelsize_candidate_from_args(args)


def test_hinerv_mlx_trainer_accepts_explicit_hard_byte_ceiling_without_candidate() -> None:
    args = _build_parser().parse_args(
        [
            "--smoke",
            "--modelsize-row",
            "hi_nerv_local_tiny",
            "--hard-byte-ceiling",
            "178000",
        ]
    )

    assert _hard_byte_ceiling_from_args(args, modelsize_candidate=None) == 178_000
    control = _build_hinerv_hard_byte_ceiling_control(
        candidate=None,
        hard_byte_ceiling=_hard_byte_ceiling_from_args(args, modelsize_candidate=None),
        archive_path="/Volumes/VertigoDataTier/pact/hinerv/archive.zip",
        archive_sha256="b" * 64,
        archive_bytes=177_999,
        archive_export_requested=True,
    )

    assert control["attached"] is True
    assert control["enforced"] is True
    assert control["hard_byte_ceiling"] == 178_000
    assert control["archive_bytes"] == 177_999
    assert control["under_hard_byte_ceiling"] is True
    assert control["blockers"] == []
    assert control["score_claim"] is False


def test_hinerv_mlx_trainer_rejects_conflicting_candidate_and_cli_byte_ceiling(
    tmp_path: Path,
) -> None:
    candidate = analyze_hinerv_modelsize_candidate(
        hard_byte_ceiling=216_000,
        num_pairs=7,
        latent_dim=10,
        embed_dim=16,
        decoder_channel=8,
        decoder_codec="int4_mixed",
        use_hierarchical_feature_grid=True,
        use_convnext_blocks=True,
    ).as_dict()
    candidate_path = tmp_path / "hinerv_candidate.json"
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    args = _build_parser().parse_args(
        [
            "--smoke",
            "--num-pairs",
            "7",
            "--modelsize-candidate-json",
            candidate_path.as_posix(),
            "--hard-byte-ceiling",
            "178000",
        ]
    )
    loaded = _modelsize_candidate_from_args(args)

    with pytest.raises(ValueError, match="conflicts with modelsize candidate"):
        _hard_byte_ceiling_from_args(args, modelsize_candidate=loaded)


def test_hinerv_mlx_trainer_byte_cap_control_records_measured_archive_delta() -> None:
    control = _build_hinerv_hard_byte_ceiling_control(
        candidate={
            "candidate_id": "hi_cap",
            "byte_cap_controller": {"predicted_under_hard_byte_ceiling": True},
        },
        hard_byte_ceiling=178_000,
        archive_path="/tmp/archive.zip",
        archive_sha256="a" * 64,
        archive_bytes=177_500,
        archive_export_requested=True,
    )

    assert control["schema"] == HI_NERV_HARD_BYTE_CEILING_CONTROL_SCHEMA
    assert control["candidate_id"] == "hi_cap"
    assert control["attached"] is True
    assert control["enforced"] is True
    assert control["archive_bytes"] == 177_500
    assert control["under_hard_byte_ceiling"] is True
    assert control["delta_bytes_vs_hard_byte_ceiling"] == -500
    assert control["blockers"] == []
    assert control["score_claim"] is False


def test_hinerv_mlx_trainer_byte_cap_control_blocks_when_export_disabled() -> None:
    control = _build_hinerv_hard_byte_ceiling_control(
        candidate={"candidate_id": "hi_cap"},
        hard_byte_ceiling=178_000,
        archive_path=None,
        archive_sha256=None,
        archive_bytes=None,
        archive_export_requested=False,
    )

    assert control["attached"] is True
    assert control["enforced"] is False
    assert control["under_hard_byte_ceiling"] is None
    assert control["delta_bytes_vs_hard_byte_ceiling"] is None
    assert control["blockers"] == [
        "hinerv_hard_byte_ceiling_not_enforced_archive_export_disabled"
    ]


def test_hinerv_mlx_trainer_coder_qat_config_is_real_and_validated() -> None:
    args = _build_parser().parse_args(
        [
            "--smoke",
            "--coder-qat",
            "--coder-qat-bits",
            "4",
            "--coder-qat-quant-residual-weight",
            "0.25",
            "--coder-qat-magnitude-weight",
            "0.125",
            "--coder-qat-delta-weight",
            "0.0625",
            "--coder-qat-c1a-entropy-weight",
            "0.0003",
            "--coder-qat-c1a-sigma",
            "0.35",
            "--coder-qat-c1a-sample-size",
            "64",
        ]
    )

    cfg = _coder_qat_config_from_args(args)

    assert cfg.enabled is True
    assert cfg.quant_bits == 4
    assert cfg.quant_residual_weight == pytest.approx(0.25)
    assert cfg.magnitude_weight == pytest.approx(0.125)
    assert cfg.delta_weight == pytest.approx(0.0625)
    assert cfg.c1a_entropy_weight == pytest.approx(0.0003)
    assert cfg.c1a_sigma == pytest.approx(0.35)
    assert cfg.c1a_sample_size == 64


def test_hinerv_train_time_control_config_is_explicit_and_false_authority() -> None:
    args = _build_parser().parse_args(
        [
            "--full",
            "--pr95-faithful-curriculum",
            "--pr95-curriculum-total-epochs",
            "29650",
            "--coder-qat",
            "--coder-qat-bits",
            "4",
            "--coder-qat-c1a-entropy-weight",
            "0.0003",
            "--coder-qat-c1a-sigma",
            "0.35",
            "--coder-qat-c1a-sample-size",
            "64",
            "--decoder-fake-quant-forward",
            "--decoder-fake-quant-bits",
            "4",
            "--segnet-direct-live-distillation-weight",
            "0.4",
            "--segnet-direct-live-base-loss-weight",
            "0.25",
            "--segnet-direct-live-class-balanced-ce-weight",
            "0.75",
            "--scorer-input-contrast-floor-weight",
            "0.5",
            "--scorer-input-contrast-floor-segnet-min-std-ratio",
            "0.625",
            "--scorer-input-contrast-floor-posenet-yuv6-min-std-ratio",
            "0.75",
            "--posenet-yuv6-geometry-tether-weight",
            "1.125",
            "--posenet-temporal-signal-floor-weight",
            "1.25",
            "--pose-direct-live-distillation-weight",
            "0.625",
            "--posenet-temporal-signal-min-std-ratio",
            "0.35",
            "--posenet-temporal-signal-min-mean-abs-ratio",
            "0.45",
            "--train-time-decoder-pruning-ratio",
            "0.125",
            "--train-time-decoder-quant-noise-bits",
            "4",
            "--train-time-decoder-quant-noise-scale",
            "0.25",
            "--train-time-decoder-control-start-epoch",
            "2",
            "--train-time-decoder-control-frequency-epochs",
            "3",
            "--export-decoder-pruning-ratio",
            "0.0625",
            "--export-decoder-quant-noise-bits",
            "6",
            "--export-decoder-quant-noise-scale",
            "0.125",
        ]
    )

    cfg = _train_time_control_config_from_args(args)
    metadata = cfg.metadata()
    contract = _pr95_full_control_contract(args, train_time_controls=cfg)

    assert metadata["schema"] == HI_NERV_TRAIN_TIME_CONTROL_SCHEMA
    assert metadata["stage_loss_schedule"] == "pr95_faithful_8stage"
    assert metadata["optimizer_kind"] == DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND
    assert metadata["optimizer_surface"] == "pr95_faithful_stage_descriptors"
    assert metadata["coder_qat_enabled"] is True
    assert metadata["coder_qat_c1a_sigma"] == pytest.approx(0.35)
    assert metadata["segnet_student_live_calibration_weight"] == pytest.approx(1.0)
    direct_live = metadata["segnet_direct_live"]
    assert direct_live["enabled"] is True
    assert direct_live["weight"] == pytest.approx(0.4)
    assert direct_live["base_loss_weight"] == pytest.approx(0.25)
    assert direct_live["class_balanced_ce_weight"] == pytest.approx(0.75)
    assert contract["controls"]["segnet_direct_live_base_loss_weight"] == pytest.approx(
        0.25
    )
    assert contract["controls"]["segnet_direct_live"]["base_loss_weight"] == pytest.approx(
        0.25
    )
    contrast_floor = metadata["scorer_input_contrast_floor"]
    assert contrast_floor["enabled"] is True
    assert contrast_floor["weight"] == pytest.approx(0.5)
    assert contrast_floor["segnet_last_rgb_min_std_ratio"] == pytest.approx(0.625)
    assert contrast_floor["posenet_yuv6_pair_min_std_ratio"] == pytest.approx(0.75)
    assert contrast_floor["human_visual_fidelity_objective"] is False
    geometry_tether = metadata["posenet_yuv6_geometry_tether"]
    assert geometry_tether["enabled"] is True
    assert geometry_tether["weight"] == pytest.approx(1.125)
    assert geometry_tether["target_surface"] == (
        "exact_upstream_posenet_two_frame_yuv6_geometry"
    )
    assert geometry_tether["human_visual_fidelity_objective"] is False
    temporal_floor = metadata["posenet_temporal_signal_floor"]
    assert temporal_floor["enabled"] is True
    assert temporal_floor["weight"] == pytest.approx(1.25)
    assert temporal_floor["min_std_ratio"] == pytest.approx(0.35)
    assert temporal_floor["min_mean_abs_ratio"] == pytest.approx(0.45)
    assert temporal_floor["human_visual_fidelity_objective"] is False
    pose_direct_live = metadata["pose_direct_live_distillation"]
    assert pose_direct_live["enabled"] is True
    assert pose_direct_live["weight"] == pytest.approx(0.625)
    assert pose_direct_live["pair_geometry_objective"] is True
    assert pose_direct_live["human_visual_fidelity_objective"] is False
    assert metadata["decoder_fake_quant_forward_enabled"] is True
    assert metadata["decoder_fake_quant_bits"] == 4
    assert metadata["train_time_decoder_controls_enabled"] is True
    assert metadata["train_time_decoder_pruning_ratio"] == pytest.approx(0.125)
    assert metadata["train_time_decoder_quant_noise_bits"] == 4
    assert metadata["train_time_decoder_control_start_epoch"] == 2
    assert metadata["train_time_decoder_control_frequency_epochs"] == 3
    assert metadata["export_decoder_pruning_ratio"] == pytest.approx(0.0625)
    assert metadata["export_decoder_quant_noise_bits"] == 6
    guard = metadata["scorer_input_distribution_guard"]
    assert "rgb_dynamic_range" in guard["components"]
    assert "segnet_frame1_mse" in guard["components"]
    assert "segnet_frame1_mae" in guard["components"]
    assert "posenet_yuv6_pair_dynamic_range" in guard["components"]
    assert "posenet_yuv6_pair_mse" in guard["components"]
    assert "posenet_yuv6_pair_mae" in guard["components"]
    assert "posenet_yuv6_temporal_delta" in guard["components"]
    assert "posenet_yuv6_temporal_delta_mse" in guard["components"]
    assert "posenet_yuv6_temporal_delta_mae" in guard["components"]
    assert guard["dynamic_range_repair_before_replay"] is True
    output_init = metadata["output_head_target_bias_init"]
    assert output_init["enabled"] is True
    assert output_init["runtime_sidecar_bytes"] == 0
    assert output_init["archive_charged_decoder_tensors"] == [
        "head_rgb_0.bias",
        "head_rgb_1.bias",
    ]
    assert metadata["score_claim"] is False
    assert metadata["ready_for_exact_eval_dispatch"] is False


def test_hinerv_train_time_control_config_rejects_ambiguous_or_fake_controls() -> None:
    with pytest.raises(ValueError, match="exactly one stage-loss authority"):
        _train_time_control_config_from_args(
            _build_parser().parse_args(
                ["--full", "--pr95-faithful-curriculum", "--staged-scorer-curriculum"]
            )
        )

    with pytest.raises(ValueError, match="owns optimizer staging"):
        _train_time_control_config_from_args(
            _build_parser().parse_args(
                ["--full", "--pr95-faithful-curriculum", "--optimizer-kind", "adamw"]
            )
        )

    with pytest.raises(ValueError, match="requires --coder-qat"):
        _train_time_control_config_from_args(
            _build_parser().parse_args(
                ["--full", "--coder-qat-c1a-entropy-weight", "0.001"]
            )
        )

    with pytest.raises(ValueError, match="bits is required"):
        _train_time_control_config_from_args(
            _build_parser().parse_args(
                ["--full", "--train-time-decoder-quant-noise-scale", "0.1"]
            )
        )

    with pytest.raises(ValueError, match="segnet_direct_live_base_loss_weight"):
        _train_time_control_config_from_args(
            _build_parser().parse_args(
                ["--full", "--segnet-direct-live-base-loss-weight", "-0.1"]
            )
        )

    with pytest.raises(ValueError, match="output_head_target_bias_init_epsilon"):
        _train_time_control_config_from_args(
            _build_parser().parse_args(
                ["--full", "--output-head-target-bias-init-epsilon", "0.75"]
            )
        )


def test_hinerv_mlx_trainer_binds_decoder_weight_waterfill_plan(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "waterfill.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": "nerv_decoder_weight_waterfill.v1",
                "family": "hi_nerv",
                "candidate_id": "hi_nerv_local_tiny",
                "rows": [
                    {
                        "group_name": "head_rgb_0.weight",
                        "selected_bits": 4,
                        "selected_action": "int4",
                    }
                ],
                "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
            }
        ),
        encoding="utf-8",
    )
    args = _build_parser().parse_args(
        [
            "--smoke",
            "--decoder-weight-waterfill-plan-json",
            plan_path.as_posix(),
        ]
    )
    controls = _train_time_control_config_from_args(args)
    plan = _decoder_weight_waterfill_plan_from_args(args)
    optimizer_control = _optimizer_control_metadata_from_args(
        args,
        decoder_weight_waterfill_plan=plan,
    )

    class DummyModel:
        def __init__(self) -> None:
            self.configured: dict[str, object] | None = None

        def configure_decoder_fake_quant_forward_from_waterfill_plan(
            self,
            decoder_weight_waterfill_plan: dict[str, object],
            *,
            fallback_quant_bits: int | None = None,
        ) -> dict[str, object]:
            from tac.substrates.hi_nerv.bitstream import (
                build_decoder_waterfill_fake_quant_forward_plan,
            )

            report = build_decoder_waterfill_fake_quant_forward_plan(
                decoder_weight_waterfill_plan
            )
            self.configured = {
                "plan": decoder_weight_waterfill_plan,
                "fallback_quant_bits": fallback_quant_bits,
                "per_tensor_bits": report["per_tensor_bits"],
            }
            return {
                **report,
                "configured": bool(report["per_tensor_bits"]),
                "configured_per_tensor_bits": dict(report["per_tensor_bits"]),
            }

    model = DummyModel()
    binding = _configure_decoder_fake_quant_forward(
        model=model,
        controls=controls,
        decoder_weight_waterfill_plan=plan,
    )
    attachment = _decoder_weight_waterfill_plan_attachment_metadata(
        args=args,
        plan=plan,
        fake_quant_forward=binding,
    )

    assert model.configured is not None
    assert model.configured["per_tensor_bits"] == {"head_rgb_0.weight": 4}
    assert binding["mode"] == "decoder_weight_waterfill_plan"
    assert binding["enabled"] is True
    assert binding["score_claim"] is False
    assert attachment["attached"] is True
    assert attachment["path"] == plan_path.resolve(strict=False).as_posix()
    assert attachment["sha256"] == sha256_file(plan_path)
    assert attachment["row_count"] == 1
    assert attachment["train_time_fake_quant_bound"] is True
    assert attachment["export_bound"] is True
    assert attachment["trainer_launch_validation"]["validated"] is True
    assert attachment["trainer_launch_validation"]["matched_candidate_keys"] == [
        "hi_nerv_local_tiny"
    ]
    assert (
        attachment["trainer_launch_validation"]["fake_quant_per_tensor_bits"]
        == {"head_rgb_0.weight": 4}
    )
    assert attachment["fake_quant_forward"]["targeted_tensor_count"] == 1
    assert optimizer_control["waterfill_gradient_multiplier_bound"] is True
    assert optimizer_control["waterfill_gradient_multiplier_by_name"] == {
        "head_rgb_0.weight": pytest.approx(0.7071067811865476)
    }
    assert optimizer_control["gradient_multiplier_by_name"] == {
        "head_rgb_0.weight": pytest.approx(0.7071067811865476)
    }
    assert optimizer_control["gradient_multiplier_waterfill_count"] == 1

    override_args = _build_parser().parse_args(
        [
            "--smoke",
            "--gradient-multiplier",
            "head_rgb_0.weight=0.25",
        ]
    )
    override_control = _optimizer_control_metadata_from_args(
        override_args,
        decoder_weight_waterfill_plan=plan,
    )
    assert override_control["waterfill_gradient_multiplier_by_name"] == {
        "head_rgb_0.weight": pytest.approx(0.7071067811865476)
    }
    assert override_control["gradient_multiplier_by_name"] == {
        "head_rgb_0.weight": pytest.approx(0.25)
    }
    assert attachment["score_claim"] is False


def test_hinerv_mlx_trainer_rejects_mismatched_decoder_weight_waterfill_plan(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "waterfill_mismatch.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": "nerv_decoder_weight_waterfill.v1",
                "family": "hi_nerv",
                "candidate_id": "hi_nerv_local_small",
                "rows": [
                    {
                        "group_name": "head_rgb_0.weight",
                        "selected_bits": 4,
                        "selected_action": "int4",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = _build_parser().parse_args(
        [
            "--smoke",
            "--modelsize-row",
            "hi_nerv_local_tiny",
            "--decoder-weight-waterfill-plan-json",
            plan_path.as_posix(),
        ]
    )

    with pytest.raises(ValueError, match="candidate_id_mismatch"):
        _decoder_weight_waterfill_plan_from_args(args)


def test_hinerv_mlx_trainer_rejects_bad_decoder_weight_waterfill_bits(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "waterfill_bad_bits.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": "nerv_decoder_weight_waterfill.v1",
                "family": "hi_nerv",
                "candidate_id": "hi_nerv_local_tiny",
                "rows": [
                    {
                        "group_name": "head_rgb_0.weight",
                        "selected_bits": 3,
                        "selected_action": "int3",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = _build_parser().parse_args(
        [
            "--smoke",
            "--decoder-weight-waterfill-plan-json",
            plan_path.as_posix(),
        ]
    )

    with pytest.raises(ValueError, match="fake_quant_bits_invalid"):
        _decoder_weight_waterfill_plan_from_args(args)


def test_hinerv_mlx_trainer_rejects_stale_decoder_weight_waterfill_shape(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "waterfill_stale_shape.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": "nerv_decoder_weight_waterfill.v1",
                "family": "hi_nerv",
                "candidate_id": "hi_nerv_local_tiny",
                "rows": [
                    {
                        "group_name": "head_rgb_0.weight",
                        "shape": [999, 999, 1, 1],
                        "selected_bits": 4,
                        "selected_action": "int4",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = _build_parser().parse_args(
        [
            "--smoke",
            "--decoder-weight-waterfill-plan-json",
            plan_path.as_posix(),
        ]
    )

    with pytest.raises(ValueError, match="shape_mismatch"):
        _decoder_weight_waterfill_plan_from_args(args)


def test_hinerv_train_time_decoder_controls_mutate_mlx_decoder_not_latents() -> None:
    mx = pytest.importorskip("mlx.core")
    import numpy as np

    class TinyMlxModel:
        def __init__(self) -> None:
            self.params = {
                "decoder": {
                    "weight": mx.array(
                        [[0.01, -0.20, 0.50, 2.0], [0.03, -0.04, 1.5, -3.0]],
                        dtype=mx.float32,
                    )
                },
                "latents": mx.array([1.0, 2.0, 3.0], dtype=mx.float32),
            }

        def parameters(self) -> dict[str, object]:
            return self.params

        def update(self, params: dict[str, object]) -> None:
            self.params = params

    model = TinyMlxModel()
    before_decoder = np.asarray(model.params["decoder"]["weight"]).copy()
    before_latents = np.asarray(model.params["latents"]).copy()
    controls = HiNervTrainTimeControlConfig(
        stage_loss_schedule="single_stage_score_aware_full",
        optimizer_kind=DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND,
        pr95_faithful_curriculum_enabled=False,
        pr95_curriculum_total_epochs=None,
        staged_scorer_curriculum_enabled=False,
        coder_qat_enabled=False,
        coder_qat_quant_bits=8,
        coder_qat_c1a_entropy_weight=0.0,
        coder_qat_c1a_sigma=0.2,
        coder_qat_c1a_sample_size=512,
        segnet_student_live_calibration_weight=1.0,
        decoder_fake_quant_forward_enabled=False,
        decoder_fake_quant_bits=8,
        train_time_decoder_pruning_ratio=0.25,
        train_time_decoder_quant_noise_bits=4,
        train_time_decoder_quant_noise_scale=0.5,
        train_time_decoder_quant_noise_seed=7,
    ).validated()

    report = _apply_train_time_decoder_controls(model, controls, epoch=0)

    after_decoder = np.asarray(model.params["decoder"]["weight"])
    after_latents = np.asarray(model.params["latents"])
    assert report["applied"] is True
    assert report["selected_tensor_count"] == 1
    assert report["changed_tensor_count"] == 1
    assert report["pruning"]["pruned_values"] > 0
    assert report["quant_noise"]["changed_tensor_count"] == 1
    assert not np.array_equal(after_decoder, before_decoder)
    assert np.count_nonzero(after_decoder == 0.0) > np.count_nonzero(
        before_decoder == 0.0
    )
    assert np.array_equal(after_latents, before_latents)
    mutation_identity = report["mutation_identity"]
    assert mutation_identity["schema"] == HI_NERV_TRAIN_TIME_DECODER_MUTATION_IDENTITY_SCHEMA
    assert mutation_identity["decoder_only_mutation"] is True
    assert mutation_identity["non_decoder_changed_tensor_names"] == []
    assert mutation_identity["selected_tensor_names"] == ["decoder.weight"]
    assert mutation_identity["changed_tensor_names"] == ["decoder.weight"]
    assert mutation_identity["changed_rows"][0]["selected_by_decoder_control"] is True
    assert mutation_identity["changed_rows"][0]["sha256_before"] != mutation_identity[
        "changed_rows"
    ][0]["sha256_after"]
    assert mutation_identity["selected_state_sha256_before"] != mutation_identity[
        "selected_state_sha256_after"
    ]
    assert mutation_identity["changed_value_count"] > 0
    assert mutation_identity["score_claim"] is False
    assert mutation_identity["ready_for_exact_eval_dispatch"] is False
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_hinerv_train_time_decoder_mutation_identity_flags_non_decoder_delta() -> None:
    import numpy as np

    identity = _build_train_time_decoder_mutation_identity(
        before_parameter_arrays={
            "decoder.weight": np.array([1.0, 2.0], dtype=np.float32),
            "latents": np.array([3.0], dtype=np.float32),
        },
        after_parameter_arrays={
            "decoder.weight": np.array([1.0, 2.0], dtype=np.float32),
            "latents": np.array([4.0], dtype=np.float32),
        },
        selected_tensor_names={"decoder.weight"},
    )

    assert identity["schema"] == HI_NERV_TRAIN_TIME_DECODER_MUTATION_IDENTITY_SCHEMA
    assert identity["decoder_only_mutation"] is False
    assert identity["changed_tensor_names"] == ["latents"]
    assert identity["non_decoder_changed_tensor_names"] == ["latents"]
    assert identity["changed_rows"][0]["selected_by_decoder_control"] is False
    assert identity["score_claim"] is False
    assert identity["ready_for_exact_eval_dispatch"] is False


def test_hinerv_mlx_trainer_pose_student_channels_match_preprocess() -> None:
    assert _pose_student_input_channels("rgb") == 3
    assert _pose_student_input_channels("pr95_yuv6") == 6

    with pytest.raises(ValueError, match="pose_student_input_preprocess"):
        _pose_student_input_channels("not_real")


def test_hinerv_mlx_trainer_builds_staged_scorer_curriculum() -> None:
    stages = _build_staged_scorer_curriculum(
        epochs=100,
        recon_fraction=0.75,
        segnet_fraction=0.15,
        final_recon_weight=0.25,
        segnet_lr_scale=0.3,
        final_lr_scale=0.1,
    )

    assert [stage.name for stage in stages] == [
        "hi_nerv_receiver_fit_recon_scaffold",
        "hi_nerv_segnet_last_frame_admission",
        "hi_nerv_joint_scorer_waterfill_finetune",
    ]
    assert [(stage.start_epoch, stage.end_epoch) for stage in stages] == [
        (0, 75),
        (75, 90),
        (90, 100),
    ]
    assert stages[0].loss_weights == {
        "recon": 1.0,
        "distill": 0.0,
        "pose_distill": 0.0,
        "scorer_input_guard": 1.0,
        "scorer_input_contrast_floor": 1.0,
        "scorer_input_shape_tether": 1.0,
        "posenet_yuv6_geometry_tether": 1.0,
        "posenet_temporal_signal_floor": 1.0,
        "segnet_direct_live_distill": 0.0,
        "segnet_direct_live_base_loss": 1.0,
    }
    assert stages[1].loss_weights == {
        "recon": 1.0,
        "distill": 1.0,
        "pose_distill": 0.0,
        "scorer_input_guard": 1.0,
        "scorer_input_contrast_floor": 1.0,
        "scorer_input_shape_tether": 1.0,
        "posenet_yuv6_geometry_tether": 1.0,
        "posenet_temporal_signal_floor": 1.0,
        "segnet_direct_live_distill": 1.0,
        "segnet_direct_live_base_loss": 1.0,
    }
    assert stages[2].loss_weights == {
        "recon": 0.25,
        "distill": 1.0,
        "pose_distill": 1.0,
        "scorer_input_guard": 1.0,
        "scorer_input_contrast_floor": 1.0,
        "scorer_input_shape_tether": 1.0,
        "posenet_yuv6_geometry_tether": 1.0,
        "posenet_temporal_signal_floor": 1.0,
        "segnet_direct_live_distill": 1.0,
        "segnet_direct_live_base_loss": 1.0,
    }
    assert stages[1].lr_scale == pytest.approx(0.3)
    assert stages[2].lr_scale == pytest.approx(0.1)


def test_hinerv_mlx_trainer_staged_curriculum_from_args_and_validation() -> None:
    args = _build_parser().parse_args(
        [
            "--full",
            "--epochs",
            "80",
            "--staged-scorer-curriculum",
            "--staged-scorer-recon-fraction",
            "0.5",
            "--staged-scorer-segnet-fraction",
            "0.25",
        ]
    )

    stages = _curriculum_stages_from_args(args)

    assert stages is not None
    assert [(stage.start_epoch, stage.end_epoch) for stage in stages] == [
        (0, 40),
        (40, 60),
        (60, 80),
    ]
    with pytest.raises(ValueError, match="epochs >= 3"):
        _build_staged_scorer_curriculum(
            epochs=2,
            recon_fraction=0.5,
            segnet_fraction=0.25,
            final_recon_weight=0.25,
            segnet_lr_scale=0.3,
            final_lr_scale=0.1,
        )


def test_hinerv_mlx_trainer_rejects_local_output_without_opt_in(
    tmp_path: Path,
) -> None:
    args = _build_parser().parse_args(["--smoke", "--output-dir", str(tmp_path / "local")])

    with pytest.raises(StorageTierError, match="local_disk_tier_disabled"):
        _resolve_output_dir(args)


def test_hinerv_mlx_trainer_allows_explicit_local_smoke_output(
    tmp_path: Path,
) -> None:
    args = _build_parser().parse_args(
        [
            "--smoke",
            "--output-dir",
            str(tmp_path / "local"),
            "--allow-local-output-dir",
        ]
    )

    output, storage = _resolve_output_dir(args)

    assert output == (tmp_path / "local").resolve(strict=False)
    assert output.is_dir()
    assert storage["schema"] == "hi_nerv_mlx_trainer_explicit_output_preflight.v1"
    assert storage["score_claim"] is False
    assert storage["ready_for_exact_eval_dispatch"] is False


def test_hinerv_mlx_trainer_parser_requires_mode() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args([])

    assert TRAINER_SCHEMA == "hi_nerv_mlx_score_aware_trainer.v1"


def test_hinerv_direct_trainer_canonicalization_contract_blocks_authority() -> None:
    contract = _direct_trainer_canonicalization_contract(mode="full")

    assert contract["schema"] == DIRECT_TRAINER_CANONICALIZATION_SCHEMA
    assert contract["canonical_runner_entrypoint"] == (
        "tools/run_compact_renderer_mlx_spine_runner.py --execute-family hi_nerv"
    )
    assert contract["direct_trainer_role"] == ("runner_subprocess_or_research_smoke_only")
    assert contract["planner_row_required"] is True
    assert contract["planner_row_id"] is None
    assert contract["source_parity_contract_consumed"] is False
    assert contract["pr95_prelaunch_gate_consumed"] is False
    assert contract["trainer_launch_allowed"] is False
    assert "hinerv_direct_trainer_missing_planner_row_id" in contract["blockers"]
    assert "hinerv_direct_trainer_local_cpu_replay_gate_not_bound" in contract["blockers"]
    assert contract["score_claim"] is False
    assert contract["ready_for_exact_eval_dispatch"] is False


def test_hinerv_direct_full_refuses_before_score_aware_trainer_call(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real_import = builtins.__import__

    def import_tripwire(name: str, *args: object, **kwargs: object) -> object:
        if name == "tac.substrates._shared.mlx_score_aware":
            raise AssertionError("run_mlx_score_aware_full_main must not be reached")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_tripwire)
    args = _build_parser().parse_args(["--full"])

    assert _full_main(args) == 2

    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert payload["schema"] == DIRECT_TRAINER_LAUNCH_REFUSAL_SCHEMA
    assert payload["mode"] == "full"
    assert payload["training_executed"] is False
    assert payload["export_executed"] is False
    assert payload["trainer_launch_allowed"] is False
    assert payload["allowed_direct_research_mode"] == "--smoke"
    assert "hinerv_direct_full_trainer_launch_blocked_by_canonicalization_contract" in payload["blockers"]
    assert "hinerv_full_trainer_launch_blocked_by_pr95_control_contract" in payload["blockers"]
    for blocker in (
        "hinerv_full_missing_segnet_distillation_loss",
        "hinerv_full_missing_eval_roundtrip_ste",
        "hinerv_full_missing_pr95_faithful_curriculum",
        "hinerv_full_pr95_epoch_budget_below_29650",
        "hinerv_full_missing_coder_aware_qat",
        "hinerv_full_missing_c1a_entropy_control",
        "hinerv_full_missing_train_time_hard_byte_ceiling",
        "hinerv_full_missing_train_time_section_byte_metrics",
        "hinerv_full_missing_ema_archive_selection",
        "hinerv_full_missing_archive_parse_back_selection",
        "hinerv_full_missing_scorer_input_distribution_guard",
        "hinerv_full_missing_direct_live_segnet_distillation",
        "hinerv_full_missing_direct_live_class_escape_pressure",
        "hinerv_full_missing_scorer_input_contrast_floor",
        "hinerv_full_missing_scorer_input_shape_tether",
        "hinerv_full_missing_posenet_yuv6_geometry_tether",
        "hinerv_full_missing_posenet_temporal_signal_floor",
        "hinerv_full_missing_scorer_space_step_guard",
        "hinerv_full_missing_strict_checkpoint_selection",
    ):
        assert blocker in payload["blockers"]
    assert (
        "hinerv_full_missing_boundary_argmax_hinge_segnet_objective"
        not in payload["blockers"]
    )
    control = payload["pr95_full_control_contract"]
    assert control["schema"] == PR95_FULL_CONTROL_CONTRACT_SCHEMA
    assert control["production_full_control_ready"] is False
    assert "score_claim" not in control
    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_hinerv_full_control_contract_clears_when_pr95_controls_are_present() -> None:
    args = _build_parser().parse_args(
        [
            "--full",
            "--epochs",
            "29650",
            "--distillation-weight",
            "1.0",
            "--pose-distillation-weight",
            "1.0",
            "--pose-direct-live-distillation-weight",
            "0.25",
            "--eval-roundtrip-ste",
            "--pr95-faithful-curriculum",
            "--pr95-stage-source-weight-amplification",
            "--coder-qat",
            "--coder-qat-c1a-entropy-weight",
            "0.0003",
            "--coder-qat-c1a-sigma",
            "0.35",
            "--coder-qat-c1a-sample-size",
            "64",
            "--hard-byte-ceiling",
            "500000",
            "--ema-archive-selection",
            "--post-export-receiver-cache-quality-gate",
            "--scorer-space-step-guard",
            "--checkpoint-selection-metric-required",
            "--scorer-input-distribution-guard-weight",
            "2.0",
            "--segnet-distillation-objective",
            "boundary_argmax_hinge",
            "--segnet-direct-live-distillation-weight",
            "0.25",
            "--segnet-direct-live-class-histogram-weight",
            "0.25",
            "--segnet-direct-live-class-balanced-hinge-weight",
            "0.5",
            "--segnet-direct-live-class-balanced-ce-weight",
            "0.25",
            "--segnet-direct-live-class-balanced-squared-hinge-weight",
            "0.75",
            "--scorer-input-contrast-floor-weight",
            "0.5",
            "--scorer-input-contrast-floor-segnet-min-std-ratio",
            "0.6",
            "--scorer-input-contrast-floor-posenet-yuv6-min-std-ratio",
            "0.6",
            "--scorer-input-shape-tether-weight",
            "0.75",
            "--posenet-yuv6-geometry-tether-weight",
            "1.125",
            "--posenet-temporal-signal-floor-weight",
            "1.25",
            "--posenet-temporal-signal-min-std-ratio",
            "0.35",
            "--posenet-temporal-signal-min-mean-abs-ratio",
            "0.45",
        ]
    )

    contract = _pr95_full_control_contract(args)

    assert contract["schema"] == PR95_FULL_CONTROL_CONTRACT_SCHEMA
    assert contract["production_full_control_ready"] is True
    assert contract["blockers"] == []
    controls = contract["controls"]
    assert controls["real_segnet_distillation_loss"] is True
    assert controls["real_posenet_distillation_loss"] is True
    assert controls["pose_direct_live_distillation_weight"] == pytest.approx(0.25)
    assert controls["pose_direct_live_distillation"]["enabled"] is True
    assert controls["pose_direct_live_distillation"][
        "human_visual_fidelity_objective"
    ] is False
    assert controls["eval_roundtrip_ste_enabled"] is True
    assert controls["pose_student_input_preprocess"] == "pr95_yuv6"
    assert controls["stage_loss_schedule"] == "pr95_faithful_8stage"
    assert controls["optimizer_kind"] == DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND
    assert controls["optimizer_surface"] == "pr95_faithful_stage_descriptors"
    assert controls["pr95_faithful_curriculum_enabled"] is True
    assert controls["pr95_muon_policy"] == "every_stage"
    assert controls["pr95_stage_source_weight_amplification_enabled"] is True
    assert controls["telemetry_flush_interval_epochs"] == 1
    assert controls["scorer_space_step_guard_enabled"] is True
    assert controls["scorer_space_step_guard"]["enabled"] is True
    assert controls["checkpoint_selection"]["metric_key"] == "total"
    assert controls["checkpoint_selection"]["metric_required"] is True
    assert controls["coder_qat_enabled"] is True
    assert controls["coder_qat_c1a_entropy_weight"] == pytest.approx(0.0003)
    assert controls["coder_qat_c1a_sigma"] == pytest.approx(0.35)
    assert controls["coder_qat_c1a_sample_size"] == 64
    assert controls["train_time_dual_ascent_enabled"] is True
    assert controls["hard_byte_ceiling_attached"] is True
    assert controls["train_time_section_byte_metrics_enabled"] is True
    assert controls["train_time_section_byte_control_measurement_phase"] == (
        "pre_model_requested"
    )
    assert controls["train_time_section_byte_control_required_for_training"] is False
    assert controls["segnet_student_live_calibration_weight"] == pytest.approx(1.0)
    assert controls["segnet_student_live_calibration_active"] is True
    assert controls["segnet_distillation_objective"] == "boundary_argmax_hinge"
    assert controls["segnet_direct_live_distillation_weight"] == pytest.approx(0.25)
    assert controls["segnet_direct_live_base_loss_weight"] == pytest.approx(1.0)
    assert controls["segnet_direct_live_class_escape_weight"] == pytest.approx(0.75)
    assert controls["segnet_direct_live"]["base_loss_weight"] == pytest.approx(1.0)
    assert controls["segnet_direct_live"]["class_histogram_weight"] == pytest.approx(0.25)
    assert controls["segnet_direct_live"]["class_balanced_hinge_weight"] == pytest.approx(0.5)
    assert controls["segnet_direct_live"]["class_balanced_ce_weight"] == pytest.approx(0.25)
    assert controls["segnet_direct_live"][
        "class_balanced_squared_hinge_weight"
    ] == pytest.approx(0.75)
    assert controls["train_time_decoder_controls_enabled"] is False
    assert controls["export_decoder_pruning_ratio"] == pytest.approx(0.0)
    assert controls["ema_archive_selection_enabled"] is True
    assert controls["archive_parse_back_selection_enabled"] is True
    assert controls["scorer_input_distribution_guard_enabled"] is True
    assert controls["scorer_input_distribution_guard_weight"] == pytest.approx(2.0)
    assert "rgb_dynamic_range" in controls[
        "scorer_input_distribution_guard_components"
    ]
    assert "segnet_frame1_mse" in controls[
        "scorer_input_distribution_guard_components"
    ]
    assert "segnet_frame1_mae" in controls[
        "scorer_input_distribution_guard_components"
    ]
    assert "posenet_yuv6_pair_dynamic_range" in controls[
        "scorer_input_distribution_guard_components"
    ]
    assert "posenet_yuv6_pair_mse" in controls[
        "scorer_input_distribution_guard_components"
    ]
    assert "posenet_yuv6_pair_mae" in controls[
        "scorer_input_distribution_guard_components"
    ]
    assert "posenet_yuv6_temporal_delta" in controls[
        "scorer_input_distribution_guard_components"
    ]
    assert "posenet_yuv6_temporal_delta_mse" in controls[
        "scorer_input_distribution_guard_components"
    ]
    assert "posenet_yuv6_temporal_delta_mae" in controls[
        "scorer_input_distribution_guard_components"
    ]
    assert controls["dynamic_range_repair_before_replay"] is True
    assert controls["scorer_input_contrast_floor_enabled"] is True
    assert controls["scorer_input_contrast_floor_weight"] == pytest.approx(0.5)
    assert controls["scorer_input_contrast_floor_segnet_min_std_ratio"] == pytest.approx(0.6)
    assert controls["scorer_input_contrast_floor_posenet_yuv6_min_std_ratio"] == pytest.approx(0.6)
    assert controls["scorer_input_shape_tether_enabled"] is True
    assert controls["scorer_input_shape_tether_weight"] == pytest.approx(0.75)
    assert controls["scorer_input_shape_tether_components"] == [
        "segnet_last_frame_rgb_centered_reference_variance_fit",
        "posenet_yuv6_pair_centered_reference_variance_fit",
        "posenet_yuv6_temporal_delta_centered_reference_variance_fit",
    ]
    assert controls["posenet_yuv6_geometry_tether_enabled"] is True
    assert controls["posenet_yuv6_geometry_tether_weight"] == pytest.approx(1.125)
    assert controls["posenet_yuv6_geometry_tether_components"] == [
        "posenet_yuv6_pair_mean_fit",
        "posenet_yuv6_pair_std_fit",
        "posenet_yuv6_pair_dynamic_range_fit",
        "posenet_yuv6_temporal_delta_fit",
    ]
    assert controls["posenet_temporal_signal_floor_enabled"] is True
    assert controls["posenet_temporal_signal_floor_weight"] == pytest.approx(1.25)
    assert controls["posenet_temporal_signal_min_std_ratio"] == pytest.approx(0.35)
    assert controls["posenet_temporal_signal_min_mean_abs_ratio"] == pytest.approx(
        0.45
    )
    assert (
        controls["posenet_temporal_signal_floor"]["human_visual_fidelity_objective"]
        is False
    )
    assert controls["output_head_target_bias_init_enabled"] is True
    assert controls["output_head_target_bias_init_epsilon"] == pytest.approx(
        1.0 / 1024.0
    )
    assert controls["output_head_target_contrast_init_enabled"] is True
    assert controls["output_head_target_contrast_init_max_pairs"] == 8
    assert controls["output_head_target_contrast_init_min_output_std"] == pytest.approx(
        1.0e-6
    )
    assert controls["output_head_target_contrast_init_max_gain"] == pytest.approx(
        4096.0
    )
    assert contract["score_claim"] is False
    assert contract["promotion_eligible"] is False
    assert contract["ready_for_exact_eval_dispatch"] is False


def test_hinerv_full_control_contract_blocks_disabled_output_head_contrast_init() -> None:
    args = _build_parser().parse_args(
        [
            "--full",
            "--epochs",
            "29650",
            "--distillation-weight",
            "1.0",
            "--pose-distillation-weight",
            "1.0",
            "--pose-direct-live-distillation-weight",
            "0.25",
            "--eval-roundtrip-ste",
            "--pr95-faithful-curriculum",
            "--pr95-stage-source-weight-amplification",
            "--coder-qat",
            "--coder-qat-c1a-entropy-weight",
            "0.0003",
            "--coder-qat-c1a-sigma",
            "0.35",
            "--coder-qat-c1a-sample-size",
            "64",
            "--hard-byte-ceiling",
            "500000",
            "--ema-archive-selection",
            "--post-export-receiver-cache-quality-gate",
            "--scorer-space-step-guard",
            "--checkpoint-selection-metric-required",
            "--scorer-input-distribution-guard-weight",
            "2.0",
            "--segnet-distillation-objective",
            "boundary_argmax_hinge",
            "--segnet-direct-live-distillation-weight",
            "0.25",
            "--segnet-direct-live-class-histogram-weight",
            "0.25",
            "--segnet-direct-live-class-balanced-hinge-weight",
            "0.5",
            "--segnet-direct-live-class-balanced-ce-weight",
            "0.25",
            "--segnet-direct-live-class-balanced-squared-hinge-weight",
            "0.75",
            "--scorer-input-contrast-floor-weight",
            "0.5",
            "--scorer-input-contrast-floor-segnet-min-std-ratio",
            "0.6",
            "--scorer-input-contrast-floor-posenet-yuv6-min-std-ratio",
            "0.6",
            "--scorer-input-shape-tether-weight",
            "0.75",
            "--posenet-yuv6-geometry-tether-weight",
            "1.125",
            "--posenet-temporal-signal-floor-weight",
            "1.25",
            "--posenet-temporal-signal-min-std-ratio",
            "0.35",
            "--posenet-temporal-signal-min-mean-abs-ratio",
            "0.45",
            "--no-output-head-target-contrast-init",
        ]
    )

    contract = _pr95_full_control_contract(args)

    assert contract["production_full_control_ready"] is False
    assert "hinerv_full_missing_output_head_target_contrast_init" in contract[
        "blockers"
    ]
    assert "hinerv_full_missing_output_head_target_bias_init" not in contract[
        "blockers"
    ]
    assert contract["controls"]["output_head_target_contrast_init_enabled"] is False
    assert contract["controls"]["output_head_target_bias_init_enabled"] is True


def test_hinerv_full_control_contract_requires_measured_section_byte_actuation() -> None:
    args = _build_parser().parse_args(
        [
            "--full",
            "--epochs",
            "29650",
            "--distillation-weight",
            "1.0",
            "--segnet-student-live-calibration-weight",
            "1.0",
            "--pose-distillation-weight",
            "1.0",
            "--pose-direct-live-distillation-weight",
            "0.25",
            "--eval-roundtrip-ste",
            "--pose-student-input-preprocess",
            "pr95_yuv6",
            "--pr95-faithful-curriculum",
            "--pr95-stage-source-weight-amplification",
            "--pr95-curriculum-total-epochs",
            "29650",
            "--coder-qat",
            "--coder-qat-c1a-entropy-weight",
            "0.0003",
            "--coder-qat-c1a-sigma",
            "0.35",
            "--coder-qat-c1a-sample-size",
            "64",
            "--hard-byte-ceiling",
            "500000",
            "--ema-archive-selection",
            "--post-export-receiver-cache-quality-gate",
            "--scorer-space-step-guard",
            "--checkpoint-selection-metric-required",
            "--scorer-input-distribution-guard-weight",
            "2.0",
            "--segnet-distillation-objective",
            "boundary_argmax_hinge",
            "--segnet-direct-live-distillation-weight",
            "0.25",
            "--segnet-direct-live-class-balanced-squared-hinge-weight",
            "0.75",
            "--scorer-input-contrast-floor-weight",
            "0.5",
            "--scorer-input-shape-tether-weight",
            "0.75",
            "--posenet-yuv6-geometry-tether-weight",
            "1.125",
            "--posenet-temporal-signal-floor-weight",
            "1.25",
        ]
    )

    blocked = _pr95_full_control_contract(
        args,
        train_time_section_byte_control={
            "schema": "hi_nerv_train_time_section_byte_control.v1",
            "active": False,
            "blockers": ["unit_inactive"],
        },
    )
    ready = _pr95_full_control_contract(
        args,
        train_time_section_byte_control={
            "schema": "hi_nerv_train_time_section_byte_control.v1",
            "active": True,
            "section_byte_budgets": {"decoder_state": 12345},
            "blockers": [],
        },
    )

    assert blocked["production_full_control_ready"] is False
    assert "hinerv_full_train_time_section_byte_control_not_active" in blocked[
        "blockers"
    ]
    assert blocked["controls"]["measured_train_time_section_byte_control_attached"] is True
    assert blocked["controls"]["measured_train_time_section_byte_control_active"] is False
    assert ready["production_full_control_ready"] is True
    assert ready["controls"]["measured_train_time_section_byte_control_active"] is True


def test_hinerv_full_control_contract_blocks_post_model_unactuated_byte_cap() -> None:
    args = _build_parser().parse_args(
        [
            "--full",
            "--epochs",
            "29650",
            "--distillation-weight",
            "1.0",
            "--segnet-student-live-calibration-weight",
            "1.0",
            "--pose-distillation-weight",
            "1.0",
            "--pose-direct-live-distillation-weight",
            "0.25",
            "--eval-roundtrip-ste",
            "--pose-student-input-preprocess",
            "pr95_yuv6",
            "--pr95-faithful-curriculum",
            "--pr95-stage-source-weight-amplification",
            "--pr95-curriculum-total-epochs",
            "29650",
            "--coder-qat",
            "--coder-qat-c1a-entropy-weight",
            "0.0003",
            "--coder-qat-c1a-sigma",
            "0.35",
            "--coder-qat-c1a-sample-size",
            "64",
            "--hard-byte-ceiling",
            "500000",
            "--ema-archive-selection",
            "--post-export-receiver-cache-quality-gate",
            "--scorer-input-distribution-guard-weight",
            "2.0",
            "--segnet-distillation-objective",
            "boundary_argmax_hinge",
            "--segnet-direct-live-distillation-weight",
            "0.25",
            "--segnet-direct-live-class-balanced-squared-hinge-weight",
            "0.75",
            "--scorer-input-contrast-floor-weight",
            "0.5",
            "--scorer-input-shape-tether-weight",
            "0.75",
            "--posenet-yuv6-geometry-tether-weight",
            "1.125",
            "--posenet-temporal-signal-floor-weight",
            "1.25",
        ]
    )

    contract = _pr95_full_control_contract(
        args,
        train_time_section_byte_control={
            "schema": "hi_nerv_train_time_section_byte_control.v1",
            "active": False,
            "controlled_section_count": 0,
            "pending_section_count": 3,
            "blockers": ["hinerv_train_time_section_byte_no_actuated_sections"],
        },
        require_measured_section_byte_control=True,
    )

    assert contract["production_full_control_ready"] is False
    assert "hinerv_full_train_time_section_byte_control_not_active" in contract[
        "blockers"
    ]
    controls = contract["controls"]
    assert controls["train_time_section_byte_control_measurement_phase"] == (
        "post_model_measured"
    )
    assert controls["train_time_section_byte_control_required_for_training"] is True
    assert controls["measured_train_time_section_byte_control_attached"] is True
    assert controls["measured_train_time_section_byte_control_active"] is False
    assert controls["measured_train_time_section_byte_controlled_section_count"] == 0
    assert controls["measured_train_time_section_byte_pending_section_count"] == 3
    assert controls["measured_train_time_section_byte_control_blockers"] == [
        "hinerv_train_time_section_byte_no_actuated_sections"
    ]


def test_hinerv_train_time_dual_ascent_config_prices_section_bytes() -> None:
    args = _build_parser().parse_args(
        [
            "--full",
            "--distillation-weight",
            "1.0",
            "--pose-distillation-weight",
            "1.0",
            "--coder-qat",
            "--coder-qat-c1a-entropy-weight",
            "0.0003",
        ]
    )
    controls = _train_time_control_config_from_args(args)

    cfg = _train_time_dual_ascent_config_from_args(
        args,
        train_time_controls=controls,
        coder_qat_loss_weight_map={"coder_qat_c1a_entropy": 0.0003},
        section_byte_control={
            "hard_byte_ceiling": 500_000,
            "section_byte_budgets": {"decoder_state": 12345},
            "section_byte_loss_weight_key_map": {
                "decoder_state": "coder_qat_c1a_entropy"
            },
            "section_byte_loss_weight_scale_map": {"decoder_state": 1.0},
        },
    )

    constraints = {row["constraint_id"]: row for row in cfg["constraints"]}
    archive = constraints["hi_nerv_archive_total_bytes"]
    decoder = constraints["hi_nerv_decoder_state_section_bytes"]
    assert cfg["enabled"] is True
    assert archive["metric_name"] == "train_time_archive_rate_score"
    assert archive["loss_weight_key"] == "coder_qat_c1a_entropy"
    assert archive["target"] > decoder["target"]
    assert "Global archive-byte pressure" in archive["rationale"]
    assert decoder["metric_name"] == "train_time_section_rate_score__decoder_state"
    assert decoder["loss_weight_key"] == "coder_qat_c1a_entropy"
    assert decoder["target"] > 0.0
    assert "25/uncompressed_total" in decoder["rationale"]


def test_hinerv_train_time_section_byte_metrics_callback_measures_live_payload() -> None:
    pytest.importorskip("mlx.core")

    from tac.substrates._shared.mlx_score_aware.coder_qat import (
        coder_qat_loss_weights,
    )
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    args = _build_parser().parse_args(
        [
            "--full",
            "--num-pairs",
            "1",
            "--output-height",
            "96",
            "--output-width",
            "128",
            "--decoder-channels",
            "4,4,4,4,4,4,4",
            "--latent-dim-coarse",
            "2",
            "--latent-dim-mid",
            "2",
            "--latent-dim-fine",
            "2",
            "--embed-dim",
            "8",
            "--coder-qat",
            "--coder-qat-c1a-entropy-weight",
            "0.0003",
            "--hard-byte-ceiling",
            "500000",
            "--train-time-section-byte-metrics-frequency-steps",
            "2",
        ]
    )
    controls = _train_time_control_config_from_args(args)
    model = HinervSubstrateMLX(_config_from_args(args))
    callback, attachment = _build_hi_nerv_train_time_section_byte_metrics_callback(
        args=args,
        model=model,
        decoder_codec=_decoder_codec_from_args(args, modelsize_candidate=None),
        controls=controls,
        decoder_weight_waterfill_plan=None,
        hard_byte_ceiling=500_000,
        active_loss_weights=coder_qat_loss_weights(_coder_qat_config_from_args(args)),
    )

    assert callback is not None
    assert attachment["enabled"] is True
    assert attachment["active"] is True
    initial = attachment["initial_control"]
    assert initial["section_byte_budgets"]["decoder_state"] > 0
    assert initial["section_byte_loss_weight_key_map"]["decoder_state"] == (
        "coder_qat_c1a_entropy"
    )

    metrics = callback(model, None, {})

    assert metrics["schema"] == "hi_nerv_train_time_section_byte_metrics.v1"
    assert metrics["archive_bytes"] > 0
    assert metrics["section_bytes"]["decoder_state"] > 0
    assert metrics["metadata"]["section_byte_control_active"] is True
    assert metrics["metadata"]["refreshed_this_step"] is True
    assert metrics["metadata"]["section_byte_loss_weight_key_map"][
        "decoder_state"
    ] == "coder_qat_c1a_entropy"
    assert "coder_qat_c1a_entropy" in metrics["metadata"][
        "positive_active_loss_weight_keys"
    ]


def test_hinerv_section_byte_metrics_refresh_uses_effective_dual_weights() -> None:
    pytest.importorskip("mlx.core")

    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX

    args = _build_parser().parse_args(
        [
            "--full",
            "--num-pairs",
            "1",
            "--output-height",
            "96",
            "--output-width",
            "128",
            "--decoder-channels",
            "4,4,4,4,4,4,4",
            "--latent-dim-coarse",
            "2",
            "--latent-dim-mid",
            "2",
            "--latent-dim-fine",
            "2",
            "--embed-dim",
            "8",
            "--coder-qat",
            "--hard-byte-ceiling",
            "500000",
            "--train-time-section-byte-metrics-frequency-steps",
            "1",
        ]
    )
    controls = _train_time_control_config_from_args(args)
    model = HinervSubstrateMLX(_config_from_args(args))
    callback, attachment = _build_hi_nerv_train_time_section_byte_metrics_callback(
        args=args,
        model=model,
        decoder_codec=_decoder_codec_from_args(args, modelsize_candidate=None),
        controls=controls,
        decoder_weight_waterfill_plan=None,
        hard_byte_ceiling=500_000,
        active_loss_weights={},
    )

    assert callback is not None
    assert attachment["enabled"] is True
    assert attachment["active"] is False
    assert "hinerv_train_time_section_byte_no_actuated_sections" in attachment[
        "blockers"
    ]

    metrics = callback(model, None, {"coder_qat_c1a_entropy": 0.25})

    assert metrics["metadata"]["section_byte_control_active"] is True
    assert metrics["metadata"]["section_byte_loss_weight_key_map"][
        "decoder_state"
    ] == "coder_qat_c1a_entropy"
    assert metrics["metadata"]["section_byte_budgets"]["decoder_state"] > 0
    assert metrics["metadata"]["active_loss_weight_keys"] == [
        "coder_qat_c1a_entropy"
    ]
    assert metrics["metadata"]["positive_active_loss_weight_keys"] == [
        "coder_qat_c1a_entropy"
    ]
    assert metrics["metadata"]["score_claim"] is False


def test_hinerv_full_control_contract_blocks_neutral_gray_head_init() -> None:
    args = _build_parser().parse_args(
        [
            "--full",
            "--epochs",
            "29650",
            "--distillation-weight",
            "1.0",
            "--pose-distillation-weight",
            "1.0",
            "--eval-roundtrip-ste",
            "--pr95-faithful-curriculum",
            "--coder-qat",
            "--coder-qat-c1a-entropy-weight",
            "0.0003",
            "--coder-qat-c1a-sigma",
            "0.35",
            "--coder-qat-c1a-sample-size",
            "64",
            "--ema-archive-selection",
            "--post-export-receiver-cache-quality-gate",
            "--scorer-input-distribution-guard-weight",
            "2.0",
            "--no-output-head-target-bias-init",
        ]
    )

    contract = _pr95_full_control_contract(args)

    assert contract["production_full_control_ready"] is False
    assert "hinerv_full_missing_output_head_target_bias_init" in contract["blockers"]
    assert contract["controls"]["output_head_target_bias_init_enabled"] is False


def test_hinerv_full_control_contract_blocks_uncalibrated_segnet_student() -> None:
    args = _build_parser().parse_args(
        [
            "--full",
            "--epochs",
            "29650",
            "--distillation-weight",
            "1.0",
            "--pose-distillation-weight",
            "1.0",
            "--segnet-student-live-calibration-weight",
            "0.0",
            "--eval-roundtrip-ste",
            "--pr95-faithful-curriculum",
            "--coder-qat",
            "--coder-qat-c1a-entropy-weight",
            "0.0003",
            "--coder-qat-c1a-sigma",
            "0.35",
            "--coder-qat-c1a-sample-size",
            "64",
            "--ema-archive-selection",
            "--post-export-receiver-cache-quality-gate",
            "--scorer-input-distribution-guard-weight",
            "2.0",
        ]
    )

    contract = _pr95_full_control_contract(args)

    assert contract["production_full_control_ready"] is False
    assert "hinerv_full_missing_segnet_student_live_calibration" in contract["blockers"]
    assert contract["controls"]["segnet_student_live_calibration_active"] is False


def test_hinerv_mlx_trainer_optimizer_choices_match_adapter() -> None:
    default_args = _build_parser().parse_args(["--full"])
    assert default_args.optimizer_kind == DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND
    assert default_args.segnet_distillation_objective == "boundary_argmax_hinge"

    for optimizer_kind in ("rmsprop", "lion", "adafactor", "muon", "pact_muon_adamw"):
        args = _build_parser().parse_args(["--full", "--optimizer-kind", optimizer_kind])
        assert args.optimizer_kind == optimizer_kind

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--full", "--optimizer-kind", "definitely_not_optimizer"])


def test_hinerv_mlx_trainer_parses_optimizer_actuator_controls() -> None:
    args = _build_parser().parse_args(
        [
            "--full",
            "--pr95-stage-source-weight-amplification",
            "--optimizer-warmup-steps-per-epoch",
            "37",
            "--pair-sampling-default-weight",
            "0.25",
            "--pair-sampling-weight",
            "5=3.5",
            "--pair-sampling-weight",
            "17=0",
            "--gradient-multiplier",
            "decoder.blocks.0.weight=0.125",
            "--gradient-multiplier",
            "head_rgb_1.bias=0",
            "--bias-gradient-multiplier",
            "0.5",
            "--output-head-bias-gradient-multiplier",
            "0.25",
        ]
    )

    optimizer_control = _optimizer_control_metadata_from_args(args)

    assert optimizer_control["schema"] == "hi_nerv_direct_trainer_optimizer_control.v1"
    assert optimizer_control["warmup_steps_per_epoch"] == 37
    assert optimizer_control["pr95_stage_source_weight_amplification_enabled"] is True
    assert optimizer_control["pair_sampling_default_weight"] == pytest.approx(0.25)
    assert _pair_sampling_weights_from_args(args) == {5: 3.5, 17: 0.0}
    assert optimizer_control["pair_sampling_weights"] == {5: 3.5, 17: 0.0}
    assert _gradient_multiplier_by_name_from_args(args) == {
        "decoder.blocks.0.weight": 0.125,
        "head_rgb_1.bias": 0.0,
    }
    assert optimizer_control["gradient_multiplier_by_name"] == {
        "decoder.blocks.0.weight": 0.125,
        "head_rgb_1.bias": 0.0,
    }
    assert optimizer_control["bias_gradient_multiplier"] == pytest.approx(0.5)
    assert optimizer_control["output_head_bias_gradient_multiplier"] == pytest.approx(
        0.25
    )
    assert optimizer_control["score_claim"] is False


def test_hinerv_mlx_trainer_parses_prioritized_pair_controls(
    tmp_path: Path,
) -> None:
    pair_file = tmp_path / "sample_generalization_gate.json"
    pair_file.write_text(
        '{"sample_generalization_gate":{"hard_pair_coverage":{"prioritized_pair_indices":[9,4,9]}}}',
        encoding="utf-8",
    )
    args = _build_parser().parse_args(
        [
            "--full",
            "--prioritized-pair-indices",
            "3,4,3",
            "--prioritized-pair-indices-file",
            str(pair_file),
        ]
    )

    pair_indices = _prioritized_pair_indices_from_args(args)
    metadata = _prioritized_pair_training_metadata(pair_indices)

    assert pair_indices == (3, 4, 9)
    assert metadata["schema"] == "hi_nerv_direct_trainer_prioritized_pair_training.v1"
    assert metadata["enabled"] is True
    assert metadata["pair_indices"] == [3, 4, 9]
    assert metadata["pair_index_domain"] == "decoded_prefix_pair_indices_0_to_num_pairs_minus_1"
    assert metadata["arbitrary_source_pair_hydration"] is False
    assert metadata["target_hydration_pair_indices_consumed"] is False
    assert metadata["requires_num_pairs_covering_pair_ids"] is True
    assert metadata["score_claim"] is False
    assert metadata["promotion_eligible"] is False
    assert metadata["ready_for_exact_eval_dispatch"] is False


def test_hinerv_prioritized_pair_lineage_metadata_has_no_canonical_authority() -> None:
    metadata = _prioritized_pair_training_lineage_metadata((4, 1))

    assert metadata["enabled"] is True
    assert metadata["pair_indices"] == [4, 1]
    assert metadata["pair_index_domain"] == "decoded_prefix_pair_indices_0_to_num_pairs_minus_1"
    assert metadata["arbitrary_source_pair_hydration"] is False
    assert metadata["target_hydration_pair_indices_consumed"] is False
    assert metadata["requires_num_pairs_covering_pair_ids"] is True
    assert metadata["canonical_authority_surface"] == ("TrainingArtifact top-level false-authority fields")
    for forbidden in (
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "rank_or_kill_eligible",
        "score_claim_valid",
    ):
        assert forbidden not in metadata


def test_hinerv_prioritized_pair_metadata_records_consumed_source_hydration() -> None:
    metadata = _prioritized_pair_training_metadata(
        (417, 22),
        target_hydration_pair_indices_consumed=True,
    )
    lineage = _prioritized_pair_training_lineage_metadata(
        (417, 22),
        target_hydration_pair_indices_consumed=True,
    )

    for payload in (metadata, lineage):
        assert payload["enabled"] is True
        assert payload["pair_indices"] == [417, 22]
        assert payload["source_pair_indices"] == [417, 22]
        assert payload["local_pair_indices"] == [0, 1]
        assert payload["pair_index_domain"] == "source_video_pair_indices"
        assert payload["pair_index_alignment_mode"] == ("local_target_rows_to_source_pair_indices")
        assert payload["arbitrary_source_pair_hydration"] is True
        assert payload["target_hydration_pair_indices_consumed"] is True
        assert payload["requires_num_pairs_covering_pair_ids"] is False


def test_hinerv_mlx_trainer_rejects_out_of_range_prioritized_pairs() -> None:
    args = _build_parser().parse_args(
        [
            "--full",
            "--num-pairs",
            "4",
            "--prioritized-pair-indices",
            "3,4",
        ]
    )

    with pytest.raises(ValueError, match="out-of-range"):
        _prioritized_pair_indices_from_args(args)


def test_hinerv_mlx_trainer_forwards_prioritized_pairs_to_harness() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "experiments/train_substrate_hi_nerv_mlx_local.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    run_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_mlx_score_aware_full_main"
    ]

    assert run_calls
    assert any(any(keyword.arg == "prioritized_pair_indices" for keyword in call.keywords) for call in run_calls)
    assert any(
        any(
            keyword.arg == "prioritized_pair_indices"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "local_training_pair_indices"
            for keyword in call.keywords
        )
        for call in run_calls
    )


def test_hinerv_mlx_trainer_forwards_optimizer_actuators_to_harness() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "experiments/train_substrate_hi_nerv_mlx_local.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    run_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_mlx_score_aware_full_main"
    ]
    required = {
        "pr95_stage_source_weight_amplification_enabled",
        "warmup_steps_per_epoch",
        "pair_sampling_weights",
        "pair_sampling_default_weight",
        "gradient_multiplier_by_name",
        "bias_gradient_multiplier",
        "output_head_bias_gradient_multiplier",
    }

    assert run_calls
    forwarded = {
        str(keyword.arg)
        for call in run_calls
        for keyword in call.keywords
        if keyword.arg
    }
    assert required.issubset(forwarded)
    assert any(
        keyword.arg == "warmup_steps_per_epoch"
        and isinstance(keyword.value, ast.Call)
        and isinstance(keyword.value.func, ast.Name)
        and keyword.value.func.id == "int"
        for call in run_calls
        for keyword in call.keywords
    )


def test_hinerv_mlx_trainer_forwards_modelsize_hard_byte_ceiling_to_archive_export() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "experiments/train_substrate_hi_nerv_mlx_local.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    export_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "export_hi_nerv_mlx_archive"
    ]

    assert export_calls
    assert all(
        any(
            keyword.arg == "hard_byte_ceiling"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "modelsize_hard_byte_ceiling"
            for keyword in call.keywords
        )
        for call in export_calls
    )


def test_hinerv_mlx_trainer_forwards_section_byte_dual_controls_to_harness() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "experiments/train_substrate_hi_nerv_mlx_local.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    run_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_mlx_score_aware_full_main"
    ]
    bundle_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RendererBundle"
    ]

    assert run_calls
    assert any(
        any(
            keyword.arg == "train_time_dual_ascent_config"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "train_time_dual_ascent_config"
            for keyword in call.keywords
        )
        for call in run_calls
    )
    assert any(
        any(
            keyword.arg == "train_time_section_byte_metrics"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "train_time_section_byte_metrics"
            for keyword in call.keywords
        )
        for call in bundle_calls
    )


def test_hinerv_direct_trainer_forwards_shared_harness_actuators() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "experiments/train_substrate_hi_nerv_mlx_local.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    run_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_mlx_score_aware_full_main"
    ]
    required = {
        "telemetry_flush_interval_epochs",
        "pr95_muon_policy",
        "scorer_space_step_guard_enabled",
        "scorer_space_step_guard_min_pre_segnet_occupied_class_fraction",
        "scorer_space_step_guard_min_post_segnet_occupied_class_fraction",
        "scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction",
        "scorer_space_step_guard_min_post_segnet_target_class_min_ratio",
        "scorer_space_step_guard_max_post_segnet_target_class_ratio_drop",
        "scorer_space_step_guard_max_post_segnet_contrast_ratio",
        "scorer_space_step_guard_max_post_segnet_distribution_mae",
        "scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae",
        "scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio",
        "scorer_space_step_guard_max_post_segnet_argmax_disagreement",
        "scorer_space_step_guard_max_post_pose_score_term",
        "scorer_space_step_guard_max_post_pose_direct_live_score_term",
        "scorer_space_step_guard_max_pose_score_term_relative_worsening",
        "scorer_space_step_guard_max_pose_score_term_absolute_worsening",
        "scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening",
        "scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening",
        "scorer_space_step_guard_max_direct_nonrate_score_worsening",
        "scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening",
        "scorer_space_step_guard_backtracking_steps",
        "scorer_space_step_guard_backtracking_shrink",
        "checkpoint_retention_keep_last_n",
        "checkpoint_retention_keep_best_n",
        "checkpoint_retention_keep_every_n_epochs",
        "checkpoint_retention_cold_store_roots",
        "checkpoint_dir",
        "resume_from_checkpoint",
        "checkpoint_selection_metric_key",
        "checkpoint_selection_metric_mode",
        "checkpoint_selection_metric_required",
        "checkpoint_selection_tie_break_metric_key",
        "checkpoint_selection_tie_break_metric_mode",
        "checkpoint_selection_tie_break_metric_required",
    }

    assert run_calls
    forwarded = {
        str(keyword.arg)
        for call in run_calls
        for keyword in call.keywords
        if keyword.arg
    }
    assert required.issubset(forwarded)


def test_hinerv_direct_trainer_shared_harness_actuator_defaults_and_validation() -> None:
    args = _build_parser().parse_args(["--full"])

    assert args.telemetry_flush_interval_epochs == 1
    assert args.pr95_muon_policy == "every_stage"
    assert args.scorer_space_step_guard is False
    assert (
        args.scorer_space_step_guard_min_post_segnet_occupied_class_fraction
        == pytest.approx(0.400001)
    )
    assert (
        args.scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction
        == pytest.approx(1.0)
    )
    assert (
        args.scorer_space_step_guard_min_post_segnet_target_class_min_ratio
        == pytest.approx(0.2)
    )
    assert (
        args.scorer_space_step_guard_max_post_segnet_target_class_ratio_drop
        == pytest.approx(0.05)
    )
    assert args.scorer_space_step_guard_max_post_segnet_contrast_ratio == pytest.approx(4.25)
    assert args.scorer_space_step_guard_max_post_segnet_distribution_mae == pytest.approx(0.31)
    assert (
        args.scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae
        == pytest.approx(0.22)
    )
    assert (
        args.scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio
        == pytest.approx(3.75)
    )
    assert args.scorer_space_step_guard_max_post_segnet_argmax_disagreement == pytest.approx(0.5)
    assert (
        args.scorer_space_step_guard_max_pose_score_term_relative_worsening
        == pytest.approx(0.01)
    )
    assert (
        args.scorer_space_step_guard_max_pose_score_term_absolute_worsening
        == pytest.approx(1.0e-4)
    )
    assert (
        args.scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening
        == pytest.approx(0.01)
    )
    assert (
        args.scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening
        == pytest.approx(1.0e-4)
    )
    assert args.scorer_space_step_guard_max_direct_nonrate_score_worsening == pytest.approx(1.0e-3)
    assert (
        args.scorer_space_step_guard_max_bootstrap_direct_nonrate_score_worsening
        == pytest.approx(15.0)
    )
    assert args.checkpoint_selection_metric_key == "total"
    assert args.checkpoint_retention_keep_last_n == 4
    assert _checkpoint_retention_keep_last_n_from_args(args) == 4
    assert args.checkpoint_retention_keep_best_n == 2
    _validate_shared_harness_train_time_actuator_args(args)

    preserve_all = _build_parser().parse_args(
        ["--full", "--checkpoint-retention-keep-last-n", "-1"]
    )
    assert _checkpoint_retention_keep_last_n_from_args(preserve_all) is None
    _validate_shared_harness_train_time_actuator_args(preserve_all)

    disabled_thresholds = _build_parser().parse_args(
        [
            "--full",
            "--scorer-space-step-guard-max-post-segnet-distribution-mae",
            "-1",
            "--scorer-space-step-guard-max-direct-nonrate-score-worsening",
            "-1",
        ]
    )
    _validate_shared_harness_train_time_actuator_args(disabled_thresholds)

    bad = _build_parser().parse_args(
        [
            "--full",
            "--scorer-space-step-guard-backtracking-shrink",
            "1.0",
        ]
    )
    with pytest.raises(ValueError, match="backtracking-shrink"):
        _validate_shared_harness_train_time_actuator_args(bad)

    bad_retention = _build_parser().parse_args(
        ["--full", "--checkpoint-retention-keep-last-n", "-2"]
    )
    with pytest.raises(ValueError, match="keep-last-n"):
        _validate_shared_harness_train_time_actuator_args(bad_retention)


def test_hinerv_mlx_trainer_forwards_contrast_floor_and_direct_live_bundle_kwargs() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "experiments/train_substrate_hi_nerv_mlx_local.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    final_bundle_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RendererBundle"
        and any(keyword.arg == "export_archive_fn" for keyword in node.keywords)
    ]

    assert len(final_bundle_calls) == 1
    bundle_call = final_bundle_calls[0]

    def keyword_reads_train_time_control(keyword_name: str, attr_name: str) -> bool:
        for keyword in bundle_call.keywords:
            if keyword.arg != keyword_name:
                continue
            return any(
                isinstance(node, ast.Attribute)
                and node.attr == attr_name
                and isinstance(node.value, ast.Name)
                and node.value.id == "train_time_controls"
                for node in ast.walk(keyword.value)
            )
        return False

    assert keyword_reads_train_time_control(
        "segnet_direct_live_base_loss_weight",
        "segnet_direct_live_base_loss_weight",
    )
    assert keyword_reads_train_time_control(
        "pose_direct_live_distillation_weight",
        "pose_direct_live_distillation_weight",
    )
    assert keyword_reads_train_time_control(
        "scorer_input_contrast_floor_weight",
        "scorer_input_contrast_floor_weight",
    )
    assert keyword_reads_train_time_control(
        "scorer_input_contrast_floor_segnet_min_std_ratio",
        "scorer_input_contrast_floor_segnet_min_std_ratio",
    )
    assert keyword_reads_train_time_control(
        "scorer_input_contrast_floor_posenet_yuv6_min_std_ratio",
        "scorer_input_contrast_floor_posenet_yuv6_min_std_ratio",
    )


def test_hinerv_mlx_trainer_direct_live_weight_builds_real_segnet_teacher() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "experiments/train_substrate_hi_nerv_mlx_local.py").read_text(
        encoding="utf-8"
    )

    assert "segnet_teacher_needed = bool(" in source
    assert (
        "or float(train_time_controls.segnet_direct_live_distillation_weight) > 0.0"
        in source
    )
    assert "if segnet_teacher_needed:" in source
    assert "scorer_teacher = build_mlx_segnet_pair_teacher(" in source


def test_hinerv_mlx_trainer_pose_direct_live_weight_builds_real_posenet_teacher() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "experiments/train_substrate_hi_nerv_mlx_local.py").read_text(
        encoding="utf-8"
    )

    assert "pose_teacher_needed = bool(" in source
    assert (
        "or float(train_time_controls.pose_direct_live_distillation_weight) > 0.0"
        in source
    )
    assert "if pose_teacher_needed:" in source
    assert "pose_scorer_teacher = build_mlx_posenet_pair_teacher(" in source


def test_hinerv_mlx_trainer_hydrates_targets_from_source_pairs() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    source = (repo_root / "experiments/train_substrate_hi_nerv_mlx_local.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    decode_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "decode_mlx_targets"
    ]
    bundle_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "RendererBundle"
    ]

    assert decode_calls
    assert any(
        any(
            keyword.arg == "pair_indices"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "source_pair_indices"
            for keyword in call.keywords
        )
        for call in decode_calls
    )
    assert any(
        any(
            keyword.arg == "source_pair_indices"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "source_pair_indices"
            for keyword in call.keywords
        )
        for call in bundle_calls
    )


def test_hinerv_mlx_trainer_metadata_safe_drops_nested_authority_keys() -> None:
    payload = {
        "storage": {
            "score_claim": False,
            "ready_for_exact_eval_dispatch": False,
            "selected_workload_root": "/Volumes/VertigoDataTier/pact/x",
            "children": [{"rank_or_kill_eligible": False, "keep": "yes"}],
        },
        "keep_top": True,
    }

    safe = _metadata_safe(payload)

    assert "score_claim" not in safe["storage"]
    assert "ready_for_exact_eval_dispatch" not in safe["storage"]
    assert safe["storage"]["selected_workload_root"].endswith("/x")
    assert safe["storage"]["children"] == [{"keep": "yes"}]
    assert safe["keep_top"] is True


def test_hinerv_mlx_trainer_parses_post_export_receiver_cache_quality_gate() -> None:
    args = _build_parser().parse_args(
        [
            "--full",
            "--post-export-receiver-cache-quality-gate",
            "--receiver-cache-quality-max-pairs",
            "4",
            "--receiver-cache-quality-batch-pairs",
            "2",
            "--receiver-cache-quality-min-segnet-dynamic-range",
            "8",
            "--receiver-cache-quality-reference-cache-dir",
            "/Volumes/VertigoDataTier/pact/ref_cache",
            "--receiver-cache-quality-min-segnet-argmax-occupied-class-fraction-for-fit-gate",
            "0.55",
            "--receiver-cache-quality-mlx-scorer-response-device-type",
            "metal",
            "--receiver-cache-quality-mlx-scorer-response-batch-pairs",
            "3",
            "--receiver-cache-quality-max-mlx-scorer-response-posenet-dist-for-fit-gate",
            "0.004",
            "--receiver-cache-quality-max-mlx-scorer-response-segnet-dist-for-fit-gate",
            "0.125",
        ]
    )

    assert args.post_export_receiver_cache_quality_gate is True
    assert args.receiver_cache_quality_max_pairs == 4
    assert args.receiver_cache_quality_batch_pairs == 2
    assert args.receiver_cache_quality_min_segnet_dynamic_range == pytest.approx(8.0)
    assert args.receiver_cache_quality_reference_cache_dir.as_posix().endswith("/ref_cache")
    assert (
        args.receiver_cache_quality_min_segnet_argmax_occupied_class_fraction_for_fit_gate
        == pytest.approx(0.55)
    )
    assert args.receiver_cache_quality_mlx_scorer_response_probe is True
    assert args.receiver_cache_quality_mlx_scorer_response_device_type == "metal"
    assert args.receiver_cache_quality_mlx_scorer_response_batch_pairs == 3
    assert (
        args.receiver_cache_quality_max_mlx_scorer_response_posenet_dist_for_fit_gate
        == pytest.approx(0.004)
    )
    assert (
        args.receiver_cache_quality_max_mlx_scorer_response_segnet_dist_for_fit_gate
        == pytest.approx(0.125)
    )


def test_hinerv_mlx_trainer_receiver_cache_quality_forwards_mlx_response_probe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"unit archive bytes")
    reference = tmp_path / "reference_cache"
    reference.mkdir()
    captured: dict[str, object] = {}

    from tac.substrates.hi_nerv import receiver_cache_quality

    def fake_write_hi_nerv_receiver_cache_quality_report(**kwargs):
        captured.update(kwargs)
        return {
            "schema": "hi_nerv_receiver_cache_quality_report.v1",
            "report_path": (tmp_path / "report.json").as_posix(),
            "archive_path": Path(kwargs["archive_zip_path"]).as_posix(),
            "archive_sha256": "c" * 64,
            "reference_cache_dir": Path(kwargs["reference_cache_dir"]).as_posix(),
            "quality_gate_passed": True,
            "quality_gate": {"verdict": "CACHE_QUALITY_GATE_PASSED"},
            "mlx_scorer_response_probe_required": True,
            "mlx_scorer_response_probe": {
                "fit_gate_passed": True,
                "avg_posenet_dist": 0.003,
                "avg_segnet_dist": 0.04,
            },
            "blockers": ["hi_nerv_receiver_cache_quality_is_false_authority"],
        }

    monkeypatch.setattr(
        receiver_cache_quality,
        "write_hi_nerv_receiver_cache_quality_report",
        fake_write_hi_nerv_receiver_cache_quality_report,
    )
    args = _build_parser().parse_args(
        [
            "--full",
            "--post-export-receiver-cache-quality-gate",
            "--receiver-cache-quality-reference-cache-dir",
            reference.as_posix(),
            "--receiver-cache-quality-mlx-scorer-response-device-type",
            "gpu",
            "--receiver-cache-quality-mlx-scorer-response-batch-pairs",
            "5",
            "--receiver-cache-quality-max-mlx-scorer-response-posenet-dist-for-fit-gate",
            "0.007",
            "--receiver-cache-quality-max-mlx-scorer-response-segnet-dist-for-fit-gate",
            "0.2",
        ]
    )

    report = _maybe_write_post_export_receiver_cache_quality(
        args=args,
        output_dir=tmp_path / "training",
        archive_path=archive,
    )

    assert report["quality_gate_passed"] is True
    assert captured["require_mlx_scorer_response_probe"] is True
    assert Path(captured["mlx_scorer_response_upstream_dir"]).name == "upstream"
    assert Path(captured["segnet_argmax_probe_upstream_dir"]).name == "upstream"
    assert captured["mlx_scorer_response_device_type"] == "gpu"
    assert captured["mlx_scorer_response_batch_pairs"] == 5
    assert captured[
        "max_mlx_scorer_response_posenet_dist_for_fit_gate"
    ] == pytest.approx(0.007)
    assert captured[
        "max_mlx_scorer_response_segnet_dist_for_fit_gate"
    ] == pytest.approx(0.2)


def test_hinerv_receiver_cache_quality_normalizes_metal_for_mlx_scorer_probe(
    tmp_path: Path,
) -> None:
    from tac.substrates.hi_nerv.receiver_cache_quality import (
        build_hi_nerv_receiver_cache_mlx_scorer_response_probe,
    )

    captured: dict[str, object] = {}

    def fake_response_payload_fn(**kwargs):
        captured.update(kwargs)
        return {
            "avg_posenet_dist": 0.001,
            "avg_segnet_dist": 0.02,
            "canonical_score": 0.03,
            "score_rate_contribution": 0.001,
            "archive_size_bytes": kwargs["archive_size_bytes"],
            "n_samples": kwargs["max_pairs"],
        }

    report = build_hi_nerv_receiver_cache_mlx_scorer_response_probe(
        candidate_cache_dir=tmp_path / "candidate",
        reference_cache_dir=tmp_path / "reference",
        archive_size_bytes=123,
        output_json=tmp_path / "probe.json",
        upstream_dir=Path("upstream"),
        device_type="metal",
        sample_pairs=2,
        response_payload_fn=fake_response_payload_fn,
    )

    assert captured["device_type"] == "gpu"
    assert captured["allow_gpu_research_signal"] is True
    assert report["requested_device_type"] == "metal"
    assert report["device_type"] == "gpu"
    assert report["mlx_device_alias_normalized"] is True
    assert report["fit_gate_passed"] is True


def test_hinerv_receiver_cache_quality_summary_drops_authority_keys() -> None:
    summary = _receiver_cache_quality_manifest_summary(
        {
            "report_path": "/Volumes/VertigoDataTier/pact/run/report.json",
            "archive_path": "/Volumes/VertigoDataTier/pact/run/archive.zip",
            "archive_sha256": "a" * 64,
            "candidate_cache_dir": "/Volumes/VertigoDataTier/pact/run/cache",
            "quality_gate_path": "/Volumes/VertigoDataTier/pact/run/gate.json",
            "segnet_argmax_probe_path": (
                "/Volumes/VertigoDataTier/pact/run/segnet_argmax_probe.json"
            ),
            "mlx_scorer_response_probe_path": (
                "/Volumes/VertigoDataTier/pact/run/mlx_scorer_response_probe.json"
            ),
            "quality_gate_passed": False,
            "blockers": ["hi_nerv_receiver_cache_quality_is_false_authority"],
            "score_claim": False,
            "quality_gate": {
                "verdict": "RENDER_OUTPUT_DYNAMIC_RANGE_TOO_LOW",
                "distance_to_reference": {"segnet_last_rgb_mae": 3.0},
                "stats": {
                    "candidate_segnet_last_rgb": {
                        "dynamic_range": 4.0,
                        "std": 1.5,
                    }
                },
                "score_claim": False,
            },
            "segnet_argmax_probe": {
                "fit_gate_passed": False,
                "segnet_argmax_disagreement_rate": 0.17,
                "candidate_occupied_class_fraction": 0.25,
                "reference_occupied_class_fraction": 0.75,
                "blockers": ["hi_nerv_receiver_cache_segnet_argmax_class_collapse"],
                "score_claim": False,
            },
            "mlx_scorer_response_probe_required": True,
            "mlx_scorer_response_probe": {
                "fit_gate_passed": False,
                "avg_posenet_dist": 0.25,
                "avg_segnet_dist": 0.17,
                "blockers": [
                    "hi_nerv_receiver_cache_posenet_response_too_high"
                ],
                "score_claim": False,
            },
        }
    )

    assert summary is not None
    assert summary["schema"] == "hi_nerv_receiver_cache_quality_summary.v1"
    assert summary["quality_gate_passed"] is False
    assert summary["quality_gate_verdict"] == "RENDER_OUTPUT_DYNAMIC_RANGE_TOO_LOW"
    assert "score_claim" not in summary
    assert summary["candidate_segnet_last_rgb_stats"]["dynamic_range"] == pytest.approx(4.0)
    assert summary["segnet_argmax_probe_passed"] is False
    assert summary["candidate_argmax_occupied_class_fraction"] == pytest.approx(0.25)
    assert summary["reference_argmax_occupied_class_fraction"] == pytest.approx(0.75)
    assert summary["segnet_argmax_disagreement_rate"] == pytest.approx(0.17)
    assert summary["segnet_argmax_probe_blockers"] == [
        "hi_nerv_receiver_cache_segnet_argmax_class_collapse"
    ]
    assert summary["mlx_scorer_response_probe_required"] is True
    assert summary["mlx_scorer_response_probe_passed"] is False
    assert summary["mlx_scorer_response_avg_posenet_dist"] == pytest.approx(0.25)


def _short_scorer_smoke_controls() -> HiNervTrainTimeControlConfig:
    return _train_time_control_config_from_args(
        _build_parser().parse_args(
            [
                "--full",
                "--segnet-direct-live-distillation-weight",
                "0.4",
                "--segnet-direct-live-class-balanced-hinge-weight",
                "0.5",
                "--scorer-input-contrast-floor-weight",
                "0.5",
                "--scorer-input-contrast-floor-segnet-min-std-ratio",
                "0.6",
                "--scorer-input-contrast-floor-posenet-yuv6-min-std-ratio",
                "0.6",
                "--scorer-input-shape-tether-weight",
                "0.25",
                "--posenet-temporal-signal-floor-weight",
                "0.25",
            ]
        )
    )


def _passing_short_scorer_receiver_quality() -> dict[str, object]:
    return {
        "report_path": "/Volumes/VertigoDataTier/pact/run/quality.json",
        "archive_path": "/Volumes/VertigoDataTier/pact/run/archive.zip",
        "archive_sha256": "b" * 64,
        "candidate_cache_dir": "/Volumes/VertigoDataTier/pact/run/cache",
        "quality_gate_path": "/Volumes/VertigoDataTier/pact/run/gate.json",
        "quality_gate_passed": True,
        "quality_gate": {
            "fit_gate_passed": True,
            "verdict": "PASS",
            "stats": {},
        },
        "segnet_argmax_probe_path": (
            "/Volumes/VertigoDataTier/pact/run/segnet_argmax_probe.json"
        ),
        "segnet_argmax_probe": {
            "fit_gate_passed": True,
            "segnet_argmax_disagreement_rate": 0.02,
            "candidate_occupied_class_fraction": 0.8,
            "candidate_target_class_coverage_fraction": 0.8,
            "candidate_target_class_min_ratio": 0.25,
            "candidate_target_material_class_covered_count": 4.0,
            "target_material_class_count": 5.0,
            "reference_occupied_class_fraction": 0.9,
            "blockers": ["hi_nerv_receiver_cache_segnet_argmax_probe_is_false_authority"],
        },
        "scorer_input_distribution_gate_path": (
            "/Volumes/VertigoDataTier/pact/run/scorer_input_distribution_gate.json"
        ),
        "scorer_input_distribution_gate": {
            "fit_gate_passed": True,
            "segnet_last_frame_rgb": {
                "candidate": {"std": 0.25, "dynamic_range": 0.9},
            },
            "posenet_yuv6_pair": {
                "candidate": {"std": 0.22, "dynamic_range": 0.8},
            },
            "posenet_yuv6_temporal_signal": {
                "candidate_delta": {"std": 0.12},
                "candidate_delta_mean_abs": 0.15,
            },
            "blockers": [
                "hi_nerv_receiver_cache_scorer_input_distribution_is_false_authority"
            ],
        },
        "mlx_scorer_response_probe_path": (
            "/Volumes/VertigoDataTier/pact/run/mlx_scorer_response_probe.json"
        ),
        "mlx_scorer_response_probe_required": True,
        "mlx_scorer_response_probe": {
            "fit_gate_passed": True,
            "avg_posenet_dist": 0.0025,
            "avg_segnet_dist": 0.02,
            "blockers": [
                "hi_nerv_receiver_cache_mlx_scorer_response_probe_is_false_authority"
            ],
        },
        "blockers": ["hi_nerv_receiver_cache_quality_is_false_authority"],
    }


def _pose_distill_score_metrics() -> dict[str, float]:
    return {
        "loss_part_pose_score_term": 0.2,
        "loss_part_pose_distill_raw_mse": 0.004,
        "loss_part_pose_score_marginal_wrt_raw_mse": 25.0,
        "loss_part_pose_distill_score_marginal_wrt_raw_mse": 25.0,
    }


def test_hinerv_short_scorer_smoke_readiness_requires_live_telemetry() -> None:
    report = _build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_short_scorer_smoke_controls(),
        final_loss_components={},
        post_export_quality=_passing_short_scorer_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
        min_segnet_occupied_class_fraction_for_fit_gate=0.55,
    )

    assert report["schema"] == HI_NERV_SHORT_SCORER_SMOKE_READINESS_SCHEMA
    assert report["short_scorer_teacher_smoke_ready"] is False
    assert report["ready_for_long_run"] is False
    assert "score_claim" in report and report["score_claim"] is False
    assert "hi_nerv_short_smoke_missing_direct_live_segnet_telemetry" in report[
        "actionable_blockers"
    ]
    assert "hi_nerv_short_smoke_missing_scorer_input_contrast_floor_telemetry" in report[
        "actionable_blockers"
    ]


def test_hinerv_short_scorer_smoke_readiness_accepts_nondegenerate_metrics() -> None:
    report = _build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_short_scorer_smoke_controls(),
        final_loss_components={
            "loss_part_segnet_direct_live_distill": 0.12,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.03,
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_missing_fraction": 0.0,
            "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.25,
            "loss_part_segnet_direct_live_class_balanced_hinge_loss": 0.04,
            "loss_part_scorer_input_contrast_floor": 0.01,
            "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 0.75,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 0.8,
            "loss_part_scorer_input_shape_tether": 0.02,
            "loss_part_scorer_input_shape_tether_segnet_last_rgb": 0.003,
            "loss_part_scorer_input_shape_tether_posenet_yuv6_pair": 0.004,
            "loss_part_scorer_input_shape_tether_posenet_yuv6_temporal_delta": 0.005,
            "loss_part_posenet_temporal_signal_floor": 0.03,
            "loss_part_posenet_temporal_signal_floor_mean_std_ratio": 0.7,
            "loss_part_posenet_temporal_signal_floor_mean_abs_ratio": 0.72,
            **_pose_distill_score_metrics(),
            "dual_ascent_active": 1.0,
            "dual_ascent_constraint_count": 2.0,
            "dual_ascent_metric__hi_nerv_segnet_direct_live_distill": 0.12,
            "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_distill": 0.0,
            "dual_ascent_lambda__hi_nerv_segnet_direct_live_distill": 0.03,
            "dual_ascent_update_count__hi_nerv_segnet_direct_live_distill": 1.0,
            "dual_ascent_weight_applied__hi_nerv_segnet_direct_live_distill": 1.0,
            "dual_ascent_effective_loss_weight__hi_nerv_segnet_direct_live_distill": 0.4,
            "dual_ascent_violation__hi_nerv_segnet_direct_live_distill": 0.1,
            "dual_ascent_metric__hi_nerv_segnet_direct_live_argmax_disagreement": 0.03,
            "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_argmax_disagreement": 0.0,
            "dual_ascent_lambda__hi_nerv_segnet_direct_live_argmax_disagreement": 0.03,
            "dual_ascent_update_count__hi_nerv_segnet_direct_live_argmax_disagreement": 1.0,
            "dual_ascent_weight_applied__hi_nerv_segnet_direct_live_argmax_disagreement": 1.0,
            "dual_ascent_effective_loss_weight__hi_nerv_segnet_direct_live_argmax_disagreement": 0.4,
            "dual_ascent_violation__hi_nerv_segnet_direct_live_argmax_disagreement": 0.1,
            "dual_ascent_metric__hi_nerv_segnet_direct_live_class_balanced_hinge": 0.04,
            "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_class_balanced_hinge": 0.0,
            "dual_ascent_lambda__hi_nerv_segnet_direct_live_class_balanced_hinge": 0.03,
            "dual_ascent_update_count__hi_nerv_segnet_direct_live_class_balanced_hinge": 1.0,
            "dual_ascent_weight_applied__hi_nerv_segnet_direct_live_class_balanced_hinge": 1.0,
            "dual_ascent_effective_loss_weight__hi_nerv_segnet_direct_live_class_balanced_hinge": 0.5,
            "dual_ascent_violation__hi_nerv_segnet_direct_live_class_balanced_hinge": 0.1,
        },
        post_export_quality=_passing_short_scorer_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
        min_segnet_occupied_class_fraction_for_fit_gate=0.55,
    )

    assert report["short_scorer_teacher_smoke_ready"] is True
    assert report["ready_for_long_run"] is True
    assert report["actionable_blockers"] == []
    assert report["blockers"] == ["hi_nerv_short_scorer_smoke_is_false_authority"]
    assert report["direct_live_segnet_gate"]["metrics"][
        "loss_part_segnet_direct_live_candidate_occupied_class_fraction"
    ] == pytest.approx(0.8)
    assert report["scorer_input_contrast_floor_gate"]["metrics"][
        "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio"
    ] == pytest.approx(0.8)
    summary = _hinerv_short_scorer_smoke_readiness_summary(report)
    assert summary is not None
    assert "score_claim" not in summary
    assert summary["short_scorer_teacher_smoke_ready"] is True


def test_hinerv_short_scorer_smoke_readiness_blocks_unactuated_direct_live_dual_with_generic_teacher() -> None:
    report = _build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_short_scorer_smoke_controls(),
        final_loss_components={
            "loss_part_segnet_direct_live_distill": 0.12,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.03,
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_missing_fraction": 0.0,
            "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.25,
            "loss_part_segnet_direct_live_class_balanced_hinge_loss": 0.04,
            "loss_part_scorer_input_contrast_floor": 0.01,
            "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 0.75,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 0.8,
            "loss_part_scorer_input_shape_tether": 0.02,
            "loss_part_scorer_input_shape_tether_segnet_last_rgb": 0.003,
            "loss_part_scorer_input_shape_tether_posenet_yuv6_pair": 0.004,
            "loss_part_scorer_input_shape_tether_posenet_yuv6_temporal_delta": 0.005,
            "loss_part_posenet_temporal_signal_floor": 0.03,
            "loss_part_posenet_temporal_signal_floor_mean_std_ratio": 0.7,
            "loss_part_posenet_temporal_signal_floor_mean_abs_ratio": 0.72,
        },
        post_export_quality=_passing_short_scorer_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
        min_segnet_occupied_class_fraction_for_fit_gate=0.55,
    )

    assert report["ready_for_long_run"] is False
    assert report["direct_live_dual_ascent_gate"]["required"] is True
    assert "hi_nerv_short_smoke_missing_direct_live_dual_ascent_telemetry" in report[
        "actionable_blockers"
    ]


def test_hinerv_short_scorer_smoke_readiness_failure_marks_training_artifact_long_run_blocker(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "training_artifact.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema": TRAINER_SCHEMA,
                "substrate_artifact_metadata": {
                    "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
                },
            }
        ),
        encoding="utf-8",
    )
    report = _build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_short_scorer_smoke_controls(),
        final_loss_components={
            "loss_part_segnet_direct_live_distill": 0.12,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.03,
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_missing_fraction": 0.0,
            "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.25,
            "loss_part_segnet_direct_live_class_balanced_hinge_loss": 0.04,
            "loss_part_scorer_input_contrast_floor": 0.01,
            "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 0.75,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 0.8,
            "loss_part_scorer_input_shape_tether": 0.02,
            "loss_part_scorer_input_shape_tether_segnet_last_rgb": 0.003,
            "loss_part_scorer_input_shape_tether_posenet_yuv6_pair": 0.004,
            "loss_part_scorer_input_shape_tether_posenet_yuv6_temporal_delta": 0.005,
            "loss_part_posenet_temporal_signal_floor": 0.03,
            "loss_part_posenet_temporal_signal_floor_mean_std_ratio": 0.7,
            "loss_part_posenet_temporal_signal_floor_mean_abs_ratio": 0.72,
        },
        post_export_quality=_passing_short_scorer_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
        min_segnet_occupied_class_fraction_for_fit_gate=0.55,
    )
    report["report_path"] = (tmp_path / "hi_nerv_short_scorer_smoke_readiness.json").as_posix()

    _attach_hinerv_short_scorer_smoke_readiness_to_training_artifact(
        output_dir=tmp_path,
        report=report,
    )

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    metadata = artifact["substrate_artifact_metadata"]
    admission = metadata["short_scorer_teacher_smoke_long_run_admission"]
    assert admission["long_run_admission_passed"] is False
    assert admission["short_scorer_teacher_smoke_passed"] is False
    assert admission["report_path"].endswith("hi_nerv_short_scorer_smoke_readiness.json")
    assert "hi_nerv_short_scorer_smoke_not_ready_for_long_run" in admission[
        "admission_blockers"
    ]
    assert "hi_nerv_short_smoke_missing_direct_live_dual_ascent_telemetry" in admission[
        "admission_blockers"
    ]
    assert "contest_cpu_cuda_exact_eval_not_executed" in metadata["blockers"]
    assert "hi_nerv_short_scorer_smoke_not_ready_for_long_run" in metadata["blockers"]
    assert metadata["short_scorer_teacher_smoke_readiness"][
        "short_scorer_teacher_smoke_ready"
    ] is False


def test_hinerv_short_scorer_smoke_readiness_pass_preserves_training_artifact_long_run_admission(
    tmp_path: Path,
) -> None:
    artifact_path = tmp_path / "training_artifact.json"
    artifact_path.write_text(
        json.dumps(
            {
                "schema": TRAINER_SCHEMA,
                "substrate_artifact_metadata": {
                    "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
                },
            }
        ),
        encoding="utf-8",
    )
    report = _build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_short_scorer_smoke_controls(),
        final_loss_components={
            "loss_part_segnet_direct_live_distill": 0.12,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.03,
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_missing_fraction": 0.0,
            "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.25,
            "loss_part_segnet_direct_live_class_balanced_hinge_loss": 0.04,
            "loss_part_scorer_input_contrast_floor": 0.01,
            "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 0.75,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 0.8,
            "loss_part_scorer_input_shape_tether": 0.02,
            "loss_part_scorer_input_shape_tether_segnet_last_rgb": 0.003,
            "loss_part_scorer_input_shape_tether_posenet_yuv6_pair": 0.004,
            "loss_part_scorer_input_shape_tether_posenet_yuv6_temporal_delta": 0.005,
            "loss_part_posenet_temporal_signal_floor": 0.03,
            "loss_part_posenet_temporal_signal_floor_mean_std_ratio": 0.7,
            "loss_part_posenet_temporal_signal_floor_mean_abs_ratio": 0.72,
            **_pose_distill_score_metrics(),
            "dual_ascent_active": 1.0,
            "dual_ascent_constraint_count": 2.0,
            "dual_ascent_metric__hi_nerv_segnet_direct_live_distill": 0.12,
            "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_distill": 0.0,
            "dual_ascent_lambda__hi_nerv_segnet_direct_live_distill": 0.03,
            "dual_ascent_update_count__hi_nerv_segnet_direct_live_distill": 1.0,
            "dual_ascent_weight_applied__hi_nerv_segnet_direct_live_distill": 1.0,
            "dual_ascent_effective_loss_weight__hi_nerv_segnet_direct_live_distill": 0.4,
            "dual_ascent_violation__hi_nerv_segnet_direct_live_distill": 0.1,
            "dual_ascent_metric__hi_nerv_segnet_direct_live_argmax_disagreement": 0.03,
            "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_argmax_disagreement": 0.0,
            "dual_ascent_lambda__hi_nerv_segnet_direct_live_argmax_disagreement": 0.03,
            "dual_ascent_update_count__hi_nerv_segnet_direct_live_argmax_disagreement": 1.0,
            "dual_ascent_weight_applied__hi_nerv_segnet_direct_live_argmax_disagreement": 1.0,
            "dual_ascent_effective_loss_weight__hi_nerv_segnet_direct_live_argmax_disagreement": 0.4,
            "dual_ascent_violation__hi_nerv_segnet_direct_live_argmax_disagreement": 0.1,
            "dual_ascent_metric__hi_nerv_segnet_direct_live_class_balanced_hinge": 0.04,
            "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_class_balanced_hinge": 0.0,
            "dual_ascent_lambda__hi_nerv_segnet_direct_live_class_balanced_hinge": 0.03,
            "dual_ascent_update_count__hi_nerv_segnet_direct_live_class_balanced_hinge": 1.0,
            "dual_ascent_weight_applied__hi_nerv_segnet_direct_live_class_balanced_hinge": 1.0,
            "dual_ascent_effective_loss_weight__hi_nerv_segnet_direct_live_class_balanced_hinge": 0.5,
            "dual_ascent_violation__hi_nerv_segnet_direct_live_class_balanced_hinge": 0.1,
        },
        post_export_quality=_passing_short_scorer_receiver_quality(),
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
        min_segnet_occupied_class_fraction_for_fit_gate=0.55,
    )

    _attach_hinerv_short_scorer_smoke_readiness_to_training_artifact(
        output_dir=tmp_path,
        report=report,
    )

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    metadata = artifact["substrate_artifact_metadata"]
    admission = metadata["short_scorer_teacher_smoke_long_run_admission"]
    assert admission["long_run_admission_passed"] is True
    assert admission["short_scorer_teacher_smoke_passed"] is True
    assert admission["admission_blockers"] == []
    assert metadata["blockers"] == ["contest_cpu_cuda_exact_eval_not_executed"]


def test_hinerv_short_scorer_smoke_readiness_accepts_direct_live_segnet_binding() -> None:
    report = _build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_short_scorer_smoke_controls(),
        final_loss_components={
            "loss_part_segnet_direct_live_distill": 0.12,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.03,
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_missing_fraction": 0.0,
            "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.25,
            "loss_part_segnet_direct_live_class_balanced_hinge_loss": 0.04,
            "dual_ascent_active": 1.0,
            "dual_ascent_constraint_count": 1.0,
            "dual_ascent_metric__hi_nerv_segnet_direct_live_distill": 0.12,
            "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_distill": 0.0,
            "dual_ascent_lambda__hi_nerv_segnet_direct_live_distill": 0.03,
            "dual_ascent_update_count__hi_nerv_segnet_direct_live_distill": 1.0,
            "dual_ascent_weight_applied__hi_nerv_segnet_direct_live_distill": 1.0,
            "dual_ascent_effective_loss_weight__hi_nerv_segnet_direct_live_distill": 0.4,
            "dual_ascent_violation__hi_nerv_segnet_direct_live_distill": 0.1,
            "dual_ascent_metric__hi_nerv_segnet_direct_live_argmax_disagreement": 0.03,
            "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_argmax_disagreement": 0.0,
            "dual_ascent_lambda__hi_nerv_segnet_direct_live_argmax_disagreement": 0.03,
            "dual_ascent_update_count__hi_nerv_segnet_direct_live_argmax_disagreement": 1.0,
            "dual_ascent_weight_applied__hi_nerv_segnet_direct_live_argmax_disagreement": 1.0,
            "dual_ascent_effective_loss_weight__hi_nerv_segnet_direct_live_argmax_disagreement": 0.4,
            "dual_ascent_violation__hi_nerv_segnet_direct_live_argmax_disagreement": 0.1,
            "dual_ascent_metric__hi_nerv_segnet_direct_live_class_balanced_hinge": 0.04,
            "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_class_balanced_hinge": 0.0,
            "dual_ascent_lambda__hi_nerv_segnet_direct_live_class_balanced_hinge": 0.03,
            "dual_ascent_update_count__hi_nerv_segnet_direct_live_class_balanced_hinge": 1.0,
            "dual_ascent_weight_applied__hi_nerv_segnet_direct_live_class_balanced_hinge": 1.0,
            "dual_ascent_effective_loss_weight__hi_nerv_segnet_direct_live_class_balanced_hinge": 0.5,
            "dual_ascent_violation__hi_nerv_segnet_direct_live_class_balanced_hinge": 0.1,
            "loss_part_scorer_input_contrast_floor": 0.01,
            "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 0.75,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 0.8,
            "loss_part_scorer_input_shape_tether": 0.02,
            "loss_part_scorer_input_shape_tether_segnet_last_rgb": 0.003,
            "loss_part_scorer_input_shape_tether_posenet_yuv6_pair": 0.004,
            "loss_part_scorer_input_shape_tether_posenet_yuv6_temporal_delta": 0.005,
            "loss_part_posenet_temporal_signal_floor": 0.03,
            "loss_part_posenet_temporal_signal_floor_mean_std_ratio": 0.7,
            "loss_part_posenet_temporal_signal_floor_mean_abs_ratio": 0.72,
            **_pose_distill_score_metrics(),
        },
        post_export_quality=_passing_short_scorer_receiver_quality(),
        segnet_distillation_weight=0.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
        min_segnet_occupied_class_fraction_for_fit_gate=0.55,
    )

    assert "hi_nerv_short_smoke_real_segnet_teacher_not_requested" not in report[
        "actionable_blockers"
    ]
    assert report["ready_for_long_run"] is True


def test_hinerv_short_scorer_smoke_readiness_accepts_region_subcontrol_only_binding() -> None:
    controls = _train_time_control_config_from_args(
        _build_parser().parse_args(
            [
                "--full",
                "--segnet-direct-live-distillation-weight",
                "0.0",
                "--segnet-direct-live-class-region-recon-weight",
                "0.75",
                "--scorer-input-contrast-floor-weight",
                "0.5",
                "--scorer-input-contrast-floor-segnet-min-std-ratio",
                "0.6",
                "--scorer-input-contrast-floor-posenet-yuv6-min-std-ratio",
                "0.6",
                "--scorer-input-shape-tether-weight",
                "0.25",
                "--posenet-temporal-signal-floor-weight",
                "0.25",
            ]
        )
    )
    report = _build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=controls,
        final_loss_components={
            "loss_part_segnet_direct_live_distill": 0.12,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.03,
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_missing_fraction": 0.0,
            "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.25,
            "loss_part_segnet_direct_live_class_region_recon_loss": 0.07,
            "dual_ascent_active": 1.0,
            "dual_ascent_constraint_count": 2.0,
            "dual_ascent_metric__hi_nerv_segnet_direct_live_class_region_recon": 0.07,
            "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_class_region_recon": 0.0,
            "dual_ascent_lambda__hi_nerv_segnet_direct_live_class_region_recon": 0.03,
            "dual_ascent_update_count__hi_nerv_segnet_direct_live_class_region_recon": 1.0,
            "dual_ascent_weight_applied__hi_nerv_segnet_direct_live_class_region_recon": 1.0,
            "dual_ascent_effective_loss_weight__hi_nerv_segnet_direct_live_class_region_recon": 0.75,
            "dual_ascent_violation__hi_nerv_segnet_direct_live_class_region_recon": 0.1,
            "dual_ascent_metric__hi_nerv_segnet_direct_live_target_min_ratio_region_recon": 0.25,
            "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_target_min_ratio_region_recon": 0.0,
            "dual_ascent_lambda__hi_nerv_segnet_direct_live_target_min_ratio_region_recon": 0.03,
            "dual_ascent_update_count__hi_nerv_segnet_direct_live_target_min_ratio_region_recon": 1.0,
            "dual_ascent_weight_applied__hi_nerv_segnet_direct_live_target_min_ratio_region_recon": 1.0,
            "dual_ascent_effective_loss_weight__hi_nerv_segnet_direct_live_target_min_ratio_region_recon": 0.75,
            "dual_ascent_violation__hi_nerv_segnet_direct_live_target_min_ratio_region_recon": 0.1,
            "loss_part_scorer_input_contrast_floor": 0.01,
            "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 0.75,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 0.8,
            "loss_part_scorer_input_shape_tether": 0.02,
            "loss_part_scorer_input_shape_tether_segnet_last_rgb": 0.003,
            "loss_part_scorer_input_shape_tether_posenet_yuv6_pair": 0.004,
            "loss_part_scorer_input_shape_tether_posenet_yuv6_temporal_delta": 0.005,
            "loss_part_posenet_temporal_signal_floor": 0.03,
            "loss_part_posenet_temporal_signal_floor_mean_std_ratio": 0.7,
            "loss_part_posenet_temporal_signal_floor_mean_abs_ratio": 0.72,
            **_pose_distill_score_metrics(),
        },
        post_export_quality=_passing_short_scorer_receiver_quality(),
        segnet_distillation_weight=0.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
        min_segnet_occupied_class_fraction_for_fit_gate=0.55,
    )

    assert report["ready_for_long_run"] is True
    assert "hi_nerv_short_smoke_direct_live_segnet_distillation_disabled" not in report[
        "actionable_blockers"
    ]
    assert "hi_nerv_short_smoke_real_segnet_teacher_not_requested" not in report[
        "actionable_blockers"
    ]
    gate = report["direct_live_segnet_gate"]
    assert gate["base_distillation_enabled"] is False
    assert gate["subcontrol_enabled"] is True
    assert gate["subcontrol_weights"][
        "segnet_direct_live_class_region_recon_weight"
    ] == pytest.approx(0.75)
    assert gate["metrics"][
        "loss_part_segnet_direct_live_class_region_recon_loss"
    ] == pytest.approx(0.07)


def test_hinerv_short_scorer_smoke_readiness_accepts_target_mass_floor_only_binding() -> None:
    controls = _train_time_control_config_from_args(
        _build_parser().parse_args(
            [
                "--full",
                "--segnet-direct-live-distillation-weight",
                "0.0",
                "--segnet-direct-live-target-mass-floor-weight",
                "0.75",
                "--scorer-input-contrast-floor-weight",
                "0.5",
                "--scorer-input-contrast-floor-segnet-min-std-ratio",
                "0.6",
                "--scorer-input-contrast-floor-posenet-yuv6-min-std-ratio",
                "0.6",
                "--scorer-input-shape-tether-weight",
                "0.25",
                "--posenet-temporal-signal-floor-weight",
                "0.25",
            ]
        )
    )
    report = _build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=controls,
        final_loss_components={
            "loss_part_segnet_direct_live_distill": 0.12,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.03,
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_missing_fraction": 0.0,
            "loss_part_segnet_direct_live_candidate_target_class_min_ratio": 0.25,
            "loss_part_segnet_direct_live_target_mass_floor_loss": 0.07,
            "dual_ascent_active": 1.0,
            "dual_ascent_constraint_count": 2.0,
            "dual_ascent_metric__hi_nerv_segnet_direct_live_target_mass_floor": 0.07,
            "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_target_mass_floor": 0.0,
            "dual_ascent_lambda__hi_nerv_segnet_direct_live_target_mass_floor": 0.03,
            "dual_ascent_update_count__hi_nerv_segnet_direct_live_target_mass_floor": 1.0,
            "dual_ascent_weight_applied__hi_nerv_segnet_direct_live_target_mass_floor": 1.0,
            "dual_ascent_effective_loss_weight__hi_nerv_segnet_direct_live_target_mass_floor": 0.75,
            "dual_ascent_violation__hi_nerv_segnet_direct_live_target_mass_floor": 0.1,
            "dual_ascent_metric__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor": 0.25,
            "dual_ascent_missing_metric__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor": 0.0,
            "dual_ascent_lambda__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor": 0.03,
            "dual_ascent_update_count__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor": 1.0,
            "dual_ascent_weight_applied__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor": 1.0,
            "dual_ascent_effective_loss_weight__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor": 0.75,
            "dual_ascent_violation__hi_nerv_segnet_direct_live_target_min_ratio_mass_floor": 0.1,
            "loss_part_scorer_input_contrast_floor": 0.01,
            "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 0.75,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 0.8,
            "loss_part_scorer_input_shape_tether": 0.02,
            "loss_part_scorer_input_shape_tether_segnet_last_rgb": 0.003,
            "loss_part_scorer_input_shape_tether_posenet_yuv6_pair": 0.004,
            "loss_part_scorer_input_shape_tether_posenet_yuv6_temporal_delta": 0.005,
            "loss_part_posenet_temporal_signal_floor": 0.03,
            "loss_part_posenet_temporal_signal_floor_mean_std_ratio": 0.7,
            "loss_part_posenet_temporal_signal_floor_mean_abs_ratio": 0.72,
            **_pose_distill_score_metrics(),
        },
        post_export_quality=_passing_short_scorer_receiver_quality(),
        segnet_distillation_weight=0.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
        min_segnet_occupied_class_fraction_for_fit_gate=0.55,
    )

    assert controls.segnet_direct_live_target_mass_floor_weight == pytest.approx(0.75)
    assert report["ready_for_long_run"] is True
    assert "hi_nerv_short_smoke_direct_live_segnet_distillation_disabled" not in report[
        "actionable_blockers"
    ]
    assert "hi_nerv_short_smoke_real_segnet_teacher_not_requested" not in report[
        "actionable_blockers"
    ]
    gate = report["direct_live_segnet_gate"]
    assert gate["base_distillation_enabled"] is False
    assert gate["subcontrol_enabled"] is True
    assert gate["subcontrol_weights"][
        "segnet_direct_live_target_mass_floor_weight"
    ] == pytest.approx(0.75)
    assert gate["metrics"][
        "loss_part_segnet_direct_live_target_mass_floor_loss"
    ] == pytest.approx(0.07)


def test_hinerv_short_scorer_smoke_readiness_blocks_missing_region_subcontrol_metric() -> None:
    controls = _train_time_control_config_from_args(
        _build_parser().parse_args(
            [
                "--full",
                "--segnet-direct-live-distillation-weight",
                "0.0",
                "--segnet-direct-live-class-region-recon-weight",
                "0.75",
                "--scorer-input-contrast-floor-weight",
                "0.5",
                "--scorer-input-contrast-floor-segnet-min-std-ratio",
                "0.6",
                "--scorer-input-contrast-floor-posenet-yuv6-min-std-ratio",
                "0.6",
            ]
        )
    )
    report = _build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=controls,
        final_loss_components={
            "loss_part_segnet_direct_live_distill": 0.12,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.03,
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
            "loss_part_segnet_direct_live_class_balanced_hinge_loss": 0.04,
            "loss_part_scorer_input_contrast_floor": 0.01,
            "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 0.75,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 0.8,
        },
        post_export_quality=_passing_short_scorer_receiver_quality(),
        segnet_distillation_weight=0.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
        min_segnet_occupied_class_fraction_for_fit_gate=0.55,
    )

    assert report["ready_for_long_run"] is False
    assert (
        "hi_nerv_short_smoke_missing_direct_live_segnet_subcontrol_telemetry"
        in report["actionable_blockers"]
    )


def test_hinerv_short_scorer_smoke_readiness_blocks_failed_mlx_response() -> None:
    quality = _passing_short_scorer_receiver_quality()
    quality["quality_gate_passed"] = False
    quality["mlx_scorer_response_probe"] = {
        "fit_gate_passed": False,
        "avg_posenet_dist": 0.2,
        "avg_segnet_dist": 0.01,
        "blockers": ["hi_nerv_receiver_cache_posenet_response_too_high"],
    }
    report = _build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_short_scorer_smoke_controls(),
        final_loss_components={
            "loss_part_segnet_direct_live_distill": 0.12,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.03,
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
            "loss_part_segnet_direct_live_class_balanced_hinge_loss": 0.04,
            "loss_part_scorer_input_contrast_floor": 0.01,
            "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 0.75,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 0.8,
        },
        post_export_quality=quality,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
        min_segnet_occupied_class_fraction_for_fit_gate=0.55,
    )

    assert report["short_scorer_teacher_smoke_ready"] is False
    assert "hi_nerv_short_smoke_receiver_cache_quality_failed" in report[
        "actionable_blockers"
    ]
    assert (
        "hi_nerv_short_smoke_receiver_cache_mlx_scorer_response_probe_failed"
        in report["actionable_blockers"]
    )


def test_hinerv_short_scorer_smoke_readiness_blocks_collapsed_receiver_occupancy() -> None:
    quality = _passing_short_scorer_receiver_quality()
    quality["quality_gate_passed"] = False
    quality["segnet_argmax_probe"] = {
        "fit_gate_passed": False,
        "segnet_argmax_disagreement_rate": 0.04,
        "candidate_occupied_class_fraction": 0.2,
        "reference_occupied_class_fraction": 0.9,
        "blockers": ["hi_nerv_receiver_cache_segnet_argmax_class_collapse"],
    }
    report = _build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=_short_scorer_smoke_controls(),
        final_loss_components={
            "loss_part_segnet_direct_live_distill": 0.12,
            "loss_part_segnet_direct_live_argmax_disagreement": 0.03,
            "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
            "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
            "loss_part_scorer_input_contrast_floor": 0.01,
            "loss_part_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 0.75,
            "loss_part_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 0.8,
        },
        post_export_quality=quality,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        allow_mock_scorer_teacher=False,
        min_segnet_occupied_class_fraction_for_fit_gate=0.55,
    )

    assert report["short_scorer_teacher_smoke_ready"] is False
    assert "hi_nerv_short_smoke_receiver_cache_quality_failed" in report[
        "actionable_blockers"
    ]
    assert "hi_nerv_short_smoke_receiver_cache_segnet_argmax_probe_failed" in report[
        "actionable_blockers"
    ]
    assert (
        "hi_nerv_short_smoke_receiver_cache_segnet_argmax_class_occupancy_collapsed"
        in report["actionable_blockers"]
    )
