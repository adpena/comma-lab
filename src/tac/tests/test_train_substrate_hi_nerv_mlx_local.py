# SPDX-License-Identifier: MIT
from __future__ import annotations

import ast
import builtins
import json
from pathlib import Path

import pytest

from comma_lab.storage_tiers import StorageTierError
from experiments.train_substrate_hi_nerv_mlx_local import (
    DIRECT_TRAINER_CANONICALIZATION_SCHEMA,
    DIRECT_TRAINER_LAUNCH_REFUSAL_SCHEMA,
    HI_NERV_HARD_BYTE_CEILING_CONTROL_SCHEMA,
    HI_NERV_MODELSIZE_CANDIDATE_CONSUMPTION_SCHEMA,
    HI_NERV_TRAIN_TIME_CONTROL_SCHEMA,
    HI_NERV_TRAIN_TIME_DECODER_MUTATION_IDENTITY_SCHEMA,
    PR95_FULL_CONTROL_CONTRACT_SCHEMA,
    TRAINER_SCHEMA,
    HiNervTrainTimeControlConfig,
    _apply_train_time_decoder_controls,
    _build_hinerv_hard_byte_ceiling_control,
    _build_parser,
    _build_staged_scorer_curriculum,
    _build_train_time_decoder_mutation_identity,
    _coder_qat_config_from_args,
    _config_from_args,
    _configure_decoder_fake_quant_forward,
    _curriculum_stages_from_args,
    _decoder_codec_from_args,
    _decoder_weight_waterfill_plan_attachment_metadata,
    _decoder_weight_waterfill_plan_from_args,
    _direct_trainer_canonicalization_contract,
    _full_main,
    _hard_byte_ceiling_from_args,
    _hard_byte_ceiling_from_modelsize_candidate,
    _metadata_safe,
    _modelsize_candidate_consumption_metadata,
    _modelsize_candidate_from_args,
    _pose_student_input_channels,
    _pr95_full_control_contract,
    _prioritized_pair_indices_from_args,
    _prioritized_pair_training_lineage_metadata,
    _prioritized_pair_training_metadata,
    _receiver_cache_quality_manifest_summary,
    _resolve_output_dir,
    _train_time_control_config_from_args,
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
        ]
    )

    loaded = _modelsize_candidate_from_args(args)
    assert loaded is not None
    cfg = _config_from_args(args, modelsize_candidate=loaded)
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

    assert metadata["schema"] == HI_NERV_TRAIN_TIME_CONTROL_SCHEMA
    assert metadata["stage_loss_schedule"] == "pr95_faithful_8stage"
    assert metadata["optimizer_kind"] == DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND
    assert metadata["optimizer_surface"] == "pr95_faithful_stage_descriptors"
    assert metadata["coder_qat_enabled"] is True
    assert metadata["coder_qat_c1a_sigma"] == pytest.approx(0.35)
    assert metadata["decoder_fake_quant_forward_enabled"] is True
    assert metadata["decoder_fake_quant_bits"] == 4
    assert metadata["train_time_decoder_controls_enabled"] is True
    assert metadata["train_time_decoder_pruning_ratio"] == pytest.approx(0.125)
    assert metadata["train_time_decoder_quant_noise_bits"] == 4
    assert metadata["train_time_decoder_control_start_epoch"] == 2
    assert metadata["train_time_decoder_control_frequency_epochs"] == 3
    assert metadata["export_decoder_pruning_ratio"] == pytest.approx(0.0625)
    assert metadata["export_decoder_quant_noise_bits"] == 6
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


def test_hinerv_mlx_trainer_binds_decoder_weight_waterfill_plan(
    tmp_path: Path,
) -> None:
    plan_path = tmp_path / "waterfill.json"
    plan_path.write_text(
        json.dumps(
            {
                "schema": "nerv_decoder_weight_waterfill.v1",
                "family": "hi_nerv",
                "candidate_id": "unit",
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
    assert attachment["fake_quant_forward"]["targeted_tensor_count"] == 1
    assert attachment["score_claim"] is False


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
    }
    assert stages[1].loss_weights == {
        "recon": 1.0,
        "distill": 1.0,
        "pose_distill": 0.0,
        "scorer_input_guard": 1.0,
    }
    assert stages[2].loss_weights == {
        "recon": 0.25,
        "distill": 1.0,
        "pose_distill": 1.0,
        "scorer_input_guard": 1.0,
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
        "hinerv_full_missing_ema_archive_selection",
        "hinerv_full_missing_archive_parse_back_selection",
        "hinerv_full_missing_scorer_input_distribution_guard",
    ):
        assert blocker in payload["blockers"]
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

    assert contract["schema"] == PR95_FULL_CONTROL_CONTRACT_SCHEMA
    assert contract["production_full_control_ready"] is True
    assert contract["blockers"] == []
    controls = contract["controls"]
    assert controls["real_segnet_distillation_loss"] is True
    assert controls["real_posenet_distillation_loss"] is True
    assert controls["eval_roundtrip_ste_enabled"] is True
    assert controls["pose_student_input_preprocess"] == "pr95_yuv6"
    assert controls["stage_loss_schedule"] == "pr95_faithful_8stage"
    assert controls["optimizer_kind"] == DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND
    assert controls["optimizer_surface"] == "pr95_faithful_stage_descriptors"
    assert controls["pr95_faithful_curriculum_enabled"] is True
    assert controls["coder_qat_enabled"] is True
    assert controls["coder_qat_c1a_entropy_weight"] == pytest.approx(0.0003)
    assert controls["coder_qat_c1a_sigma"] == pytest.approx(0.35)
    assert controls["coder_qat_c1a_sample_size"] == 64
    assert controls["train_time_decoder_controls_enabled"] is False
    assert controls["export_decoder_pruning_ratio"] == pytest.approx(0.0)
    assert controls["ema_archive_selection_enabled"] is True
    assert controls["archive_parse_back_selection_enabled"] is True
    assert controls["scorer_input_distribution_guard_enabled"] is True
    assert controls["scorer_input_distribution_guard_weight"] == pytest.approx(2.0)
    assert contract["score_claim"] is False
    assert contract["promotion_eligible"] is False
    assert contract["ready_for_exact_eval_dispatch"] is False


def test_hinerv_mlx_trainer_optimizer_choices_match_adapter() -> None:
    default_args = _build_parser().parse_args(["--full"])
    assert default_args.optimizer_kind == DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND

    for optimizer_kind in ("rmsprop", "lion", "adafactor", "muon", "pact_muon_adamw"):
        args = _build_parser().parse_args(["--full", "--optimizer-kind", optimizer_kind])
        assert args.optimizer_kind == optimizer_kind

    with pytest.raises(SystemExit):
        _build_parser().parse_args(["--full", "--optimizer-kind", "definitely_not_optimizer"])


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
        ]
    )

    assert args.post_export_receiver_cache_quality_gate is True
    assert args.receiver_cache_quality_max_pairs == 4
    assert args.receiver_cache_quality_batch_pairs == 2
    assert args.receiver_cache_quality_min_segnet_dynamic_range == pytest.approx(8.0)
    assert args.receiver_cache_quality_reference_cache_dir.as_posix().endswith("/ref_cache")


def test_hinerv_receiver_cache_quality_summary_drops_authority_keys() -> None:
    summary = _receiver_cache_quality_manifest_summary(
        {
            "report_path": "/Volumes/VertigoDataTier/pact/run/report.json",
            "archive_path": "/Volumes/VertigoDataTier/pact/run/archive.zip",
            "archive_sha256": "a" * 64,
            "candidate_cache_dir": "/Volumes/VertigoDataTier/pact/run/cache",
            "quality_gate_path": "/Volumes/VertigoDataTier/pact/run/gate.json",
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
        }
    )

    assert summary is not None
    assert summary["schema"] == "hi_nerv_receiver_cache_quality_summary.v1"
    assert summary["quality_gate_passed"] is False
    assert summary["quality_gate_verdict"] == "RENDER_OUTPUT_DYNAMIC_RANGE_TOO_LOW"
    assert "score_claim" not in summary
    assert summary["candidate_segnet_last_rgb_stats"]["dynamic_range"] == pytest.approx(4.0)
