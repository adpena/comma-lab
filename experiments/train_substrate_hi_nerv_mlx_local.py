#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""HiNeRV MLX-local score-aware trainer.

This is a real MLX harness binding for the current local HiNeRV archive family.
It is still false-authority: MLX/local training artifacts may guide iteration,
but contest CPU/CUDA replay is the only score/rank surface.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.operator_storage_waterfall import (
    operator_storage_policy_payload,
    operator_storage_tier_cli_specs,
)
from comma_lab.storage_tiers import (
    DEFAULT_RESERVE_FREE_GB,
    StorageTierError,
    parse_storage_tier_specs,
    plan_experiment_storage,
    require_selected_storage,
)
from tac.adaptation.hard_pair_indices import (
    HardPairIndicesError,
    load_pair_indices_file,
    merge_pair_indices,
    parse_pair_indices_csv,
    validate_pair_indices_in_range,
)
from tac.analysis.nerv_modelsize_budget import (
    MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS,
    build_hinerv_config_from_modelsize_candidate,
    modelsize_control_precedence_contract,
)
from tac.analysis.nerv_modelsize_ladder import (
    hi_nerv_modelsize_config_rows,
)
from tac.repo_io import sha256_file, write_json
from tac.substrates._shared.mlx_score_aware.adapter import (
    DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND,
    SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS,
)
from tac.substrates._shared.mlx_score_aware.modelsize_budget_plan import (
    FALSE_AUTHORITY,
)
from tac.substrates.hi_nerv.receiver_cache_quality import (
    DEFAULT_MAX_MLX_SCORER_RESPONSE_POSENET_DIST_FOR_FIT_GATE,
    DEFAULT_MAX_MLX_SCORER_RESPONSE_SEGNET_DIST_FOR_FIT_GATE,
)
from tac.substrates.hi_nerv.short_scorer_readiness import (
    HI_NERV_SHORT_SCORER_SMOKE_DEFAULT_MIN_SEGNET_OCCUPIED_CLASS_FRACTION,
    HI_NERV_SHORT_SCORER_SMOKE_READINESS_SCHEMA,
)
from tac.substrates.hi_nerv.short_scorer_readiness import (
    build_hinerv_short_scorer_smoke_readiness_report as _shared_build_readiness,
)
from tac.substrates.hi_nerv.short_scorer_readiness import (
    hinerv_short_scorer_smoke_readiness_summary as _shared_readiness_summary,
)
from tac.substrates.hi_nerv.short_scorer_readiness import (
    receiver_cache_quality_manifest_summary as _shared_receiver_quality_summary,
)

TRAINER_SCHEMA = "hi_nerv_mlx_score_aware_trainer.v1"
TRAINER_AUTHORITY = "false_authority_macos_mlx_training_no_contest_score_claim"
TRAINER_AXIS_TAG = "[macOS-MLX research-signal]"
PR95_FULL_CONTROL_CONTRACT_SCHEMA = "hi_nerv_pr95_full_control_contract.v1"
CANONICAL_PR95_FULL_EPOCHS = 29_650
HI_NERV_TRAIN_TIME_CONTROL_SCHEMA = "hi_nerv_train_time_controls.v1"
HI_NERV_TRAIN_TIME_DECODER_CONTROL_REPORT_SCHEMA = "hi_nerv_train_time_decoder_control_report.v1"
HI_NERV_TRAIN_TIME_DECODER_MUTATION_IDENTITY_SCHEMA = (
    "hi_nerv_train_time_decoder_mutation_identity.v1"
)
HI_NERV_MODELSIZE_CANDIDATE_CONSUMPTION_SCHEMA = (
    "hi_nerv_trainer_modelsize_candidate_consumption.v1"
)
HI_NERV_HARD_BYTE_CEILING_CONTROL_SCHEMA = "hi_nerv_hard_byte_ceiling_control.v1"
DIRECT_TRAINER_CANONICALIZATION_SCHEMA = "hi_nerv_direct_trainer_canonicalization_contract.v1"
DIRECT_TRAINER_LAUNCH_REFUSAL_SCHEMA = "hi_nerv_direct_trainer_launch_refusal.v1"
DIRECT_TRAINER_CANONICAL_RUNNER_ENTRYPOINT = "tools/run_compact_renderer_mlx_spine_runner.py --execute-family hi_nerv"
DIRECT_TRAINER_CANONICALIZATION_BLOCKERS = (
    "direct_hinerv_trainer_launch_not_compact_runner_owned",
    "hinerv_direct_trainer_missing_planner_row_id",
    "hinerv_direct_trainer_source_parity_contract_not_consumed",
    "hinerv_direct_trainer_source_faithfulness_gate_not_consumed",
    "hinerv_direct_trainer_pr95_prelaunch_gate_not_consumed",
    "hinerv_direct_modelsize_row_not_budget_candidate_contract",
    "hinerv_direct_trainer_full_video_prefilter_not_bound",
    "hinerv_direct_trainer_local_cpu_replay_gate_not_bound",
)
DEFAULT_WORKLOAD_SUBDIR = "hinerv_mlx_local_training"
DEFAULT_DECODER_CODEC = "int8_mixed"
MODEL_SIZE_ROWS = tuple(row["row_id"] for row in hi_nerv_modelsize_config_rows(num_pairs=600))
_HI_NERV_DECODER_CONTROL_INCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "latent_embed",
    "blocks",
    "feature_grids",
    "convnext_blocks",
    "injector",
    "head",
    "decoder",
)
_HI_NERV_DECODER_CONTROL_EXCLUDE_SUBSTRINGS: tuple[str, ...] = (
    "latents",
    "codebook",
    "ema",
    "student",
    "teacher",
    "quantizer",
)
_HI_NERV_TRAIN_TIME_QUANT_NOISE_BITS: tuple[int, ...] = (2, 4, 6, 7, 8)


@dataclass(frozen=True)
class HiNervTrainTimeControlConfig:
    """Explicit, validated controls for HiNeRV train-time rate pressure."""

    stage_loss_schedule: str
    optimizer_kind: str
    pr95_faithful_curriculum_enabled: bool
    pr95_curriculum_total_epochs: int | None
    staged_scorer_curriculum_enabled: bool
    coder_qat_enabled: bool
    coder_qat_quant_bits: int
    coder_qat_c1a_entropy_weight: float
    coder_qat_c1a_sigma: float
    coder_qat_c1a_sample_size: int
    segnet_student_live_calibration_weight: float
    decoder_fake_quant_forward_enabled: bool
    decoder_fake_quant_bits: int
    segnet_distillation_objective: str = "kl_t2"
    distillation_temperature: float = 2.0
    segnet_direct_live_distillation_weight: float = 0.0
    segnet_direct_live_base_loss_weight: float = 1.0
    segnet_direct_live_class_histogram_weight: float = 0.0
    segnet_direct_live_class_balanced_hinge_weight: float = 0.0
    segnet_direct_live_class_balanced_ce_weight: float = 0.0
    segnet_direct_live_class_balanced_squared_hinge_weight: float = 0.0
    segnet_tau_boundary: float = 1.0
    segnet_hinge_margin: float = 1.0
    train_time_decoder_pruning_ratio: float = 0.0
    train_time_decoder_quant_noise_bits: int | None = None
    train_time_decoder_quant_noise_scale: float = 0.0
    train_time_decoder_quant_noise_seed: int = 0
    train_time_decoder_control_start_epoch: int = 0
    train_time_decoder_control_frequency_epochs: int = 1
    export_decoder_pruning_ratio: float = 0.0
    export_decoder_quant_noise_bits: int | None = None
    export_decoder_quant_noise_scale: float = 0.0
    export_decoder_quant_noise_seed: int = 0
    scorer_input_distribution_guard_weight: float = 0.0
    scorer_input_distribution_guard_saturation_margin: float = 0.02
    scorer_input_distribution_guard_temperature: float = 0.01
    scorer_input_contrast_floor_weight: float = 0.0
    scorer_input_contrast_floor_segnet_min_std_ratio: float = 0.5
    scorer_input_contrast_floor_posenet_yuv6_min_std_ratio: float = 0.5
    scorer_input_shape_tether_weight: float = 0.0
    posenet_temporal_signal_floor_weight: float = 0.0
    posenet_temporal_signal_min_std_ratio: float = 0.25
    posenet_temporal_signal_min_mean_abs_ratio: float = 0.25
    output_head_target_bias_init_enabled: bool = True
    output_head_target_bias_init_epsilon: float = 1.0 / 1024.0
    output_head_target_contrast_init_enabled: bool = True
    output_head_target_contrast_init_max_pairs: int = 8
    output_head_target_contrast_init_min_output_std: float = 1.0e-6
    output_head_target_contrast_init_max_gain: float = 4096.0

    def validated(self) -> HiNervTrainTimeControlConfig:
        if self.stage_loss_schedule not in {
            "single_stage_score_aware_full",
            "staged_scorer_curriculum",
            "pr95_faithful_8stage",
        }:
            raise ValueError(
                "stage_loss_schedule must be one of "
                "single_stage_score_aware_full, staged_scorer_curriculum, "
                f"pr95_faithful_8stage; got {self.stage_loss_schedule!r}"
            )
        if self.pr95_faithful_curriculum_enabled and self.staged_scorer_curriculum_enabled:
            raise ValueError(
                "HiNeRV train-time controls require exactly one stage-loss "
                "authority; --pr95-faithful-curriculum and "
                "--staged-scorer-curriculum cannot both be set"
            )
        if (
            self.pr95_faithful_curriculum_enabled
            and str(self.optimizer_kind) != DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND
        ):
            raise ValueError(
                "HiNeRV PR95 faithful curriculum owns optimizer staging; leave "
                "--optimizer-kind at the Pact partitioned Muon+AdamW default "
                "or disable --pr95-faithful-curriculum"
            )
        if int(self.coder_qat_quant_bits) < 1 or int(self.coder_qat_quant_bits) > 16:
            raise ValueError("coder_qat_quant_bits must be in [1, 16]")
        if float(self.coder_qat_c1a_entropy_weight) > 0.0 and not self.coder_qat_enabled:
            raise ValueError(
                "coder_qat_c1a_entropy_weight requires --coder-qat so the "
                "C1a sigma control reaches the real loss path"
            )
        _require_finite_nonnegative(
            self.coder_qat_c1a_entropy_weight,
            "coder_qat_c1a_entropy_weight",
        )
        _require_finite_positive(self.coder_qat_c1a_sigma, "coder_qat_c1a_sigma")
        if int(self.coder_qat_c1a_sample_size) <= 0:
            raise ValueError("coder_qat_c1a_sample_size must be positive")
        _require_finite_nonnegative(
            self.segnet_student_live_calibration_weight,
            "segnet_student_live_calibration_weight",
        )
        if self.segnet_distillation_objective not in {
            "kl_t2",
            "argmax_hinge",
            "boundary_tckd",
            "boundary_kl_t2",
            "boundary_argmax_hinge",
        }:
            raise ValueError(
                "segnet_distillation_objective must be a supported shared "
                f"score-aware SegNet objective; got {self.segnet_distillation_objective!r}"
            )
        _require_finite_positive(self.distillation_temperature, "distillation_temperature")
        _require_finite_nonnegative(
            self.segnet_direct_live_distillation_weight,
            "segnet_direct_live_distillation_weight",
        )
        _require_finite_nonnegative(
            self.segnet_direct_live_base_loss_weight,
            "segnet_direct_live_base_loss_weight",
        )
        _require_finite_nonnegative(
            self.segnet_direct_live_class_histogram_weight,
            "segnet_direct_live_class_histogram_weight",
        )
        _require_finite_nonnegative(
            self.segnet_direct_live_class_balanced_hinge_weight,
            "segnet_direct_live_class_balanced_hinge_weight",
        )
        _require_finite_nonnegative(
            self.segnet_direct_live_class_balanced_ce_weight,
            "segnet_direct_live_class_balanced_ce_weight",
        )
        _require_finite_nonnegative(
            self.segnet_direct_live_class_balanced_squared_hinge_weight,
            "segnet_direct_live_class_balanced_squared_hinge_weight",
        )
        _require_finite_positive(self.segnet_tau_boundary, "segnet_tau_boundary")
        _require_finite_positive(self.segnet_hinge_margin, "segnet_hinge_margin")
        if int(self.decoder_fake_quant_bits) < 1 or int(self.decoder_fake_quant_bits) > 16:
            raise ValueError("decoder_fake_quant_bits must be in [1, 16]")
        _validate_ratio(
            self.train_time_decoder_pruning_ratio,
            "train_time_decoder_pruning_ratio",
        )
        _validate_ratio(self.export_decoder_pruning_ratio, "export_decoder_pruning_ratio")
        _validate_quant_noise_controls(
            bits=self.train_time_decoder_quant_noise_bits,
            scale=self.train_time_decoder_quant_noise_scale,
            field_prefix="train_time_decoder_quant_noise",
        )
        _validate_quant_noise_controls(
            bits=self.export_decoder_quant_noise_bits,
            scale=self.export_decoder_quant_noise_scale,
            field_prefix="export_decoder_quant_noise",
        )
        if int(self.train_time_decoder_control_start_epoch) < 0:
            raise ValueError("train_time_decoder_control_start_epoch must be >= 0")
        if int(self.train_time_decoder_control_frequency_epochs) <= 0:
            raise ValueError("train_time_decoder_control_frequency_epochs must be positive")
        _require_finite_nonnegative(
            self.scorer_input_distribution_guard_weight,
            "scorer_input_distribution_guard_weight",
        )
        if not (
            0.0
            < float(self.scorer_input_distribution_guard_saturation_margin)
            < 0.5
        ):
            raise ValueError(
                "scorer_input_distribution_guard_saturation_margin must be in "
                "(0, 0.5)"
            )
        _require_finite_positive(
            self.scorer_input_distribution_guard_temperature,
            "scorer_input_distribution_guard_temperature",
        )
        _require_finite_nonnegative(
            self.scorer_input_contrast_floor_weight,
            "scorer_input_contrast_floor_weight",
        )
        _require_finite_positive(
            self.scorer_input_contrast_floor_segnet_min_std_ratio,
            "scorer_input_contrast_floor_segnet_min_std_ratio",
        )
        _require_finite_positive(
            self.scorer_input_contrast_floor_posenet_yuv6_min_std_ratio,
            "scorer_input_contrast_floor_posenet_yuv6_min_std_ratio",
        )
        _require_finite_nonnegative(
            self.scorer_input_shape_tether_weight,
            "scorer_input_shape_tether_weight",
        )
        _require_finite_nonnegative(
            self.posenet_temporal_signal_floor_weight,
            "posenet_temporal_signal_floor_weight",
        )
        _require_finite_positive(
            self.posenet_temporal_signal_min_std_ratio,
            "posenet_temporal_signal_min_std_ratio",
        )
        _require_finite_positive(
            self.posenet_temporal_signal_min_mean_abs_ratio,
            "posenet_temporal_signal_min_mean_abs_ratio",
        )
        _validate_unit_open_interval(
            self.output_head_target_bias_init_epsilon,
            "output_head_target_bias_init_epsilon",
        )
        if int(self.output_head_target_contrast_init_max_pairs) <= 0:
            raise ValueError("output_head_target_contrast_init_max_pairs must be positive")
        _require_finite_positive(
            self.output_head_target_contrast_init_min_output_std,
            "output_head_target_contrast_init_min_output_std",
        )
        if float(self.output_head_target_contrast_init_max_gain) < 1.0:
            raise ValueError("output_head_target_contrast_init_max_gain must be >= 1")
        _require_finite_positive(
            self.output_head_target_contrast_init_max_gain,
            "output_head_target_contrast_init_max_gain",
        )
        return replace(
            self,
            optimizer_kind=str(self.optimizer_kind),
            pr95_curriculum_total_epochs=(
                None
                if self.pr95_curriculum_total_epochs is None
                else int(self.pr95_curriculum_total_epochs)
            ),
            coder_qat_quant_bits=int(self.coder_qat_quant_bits),
            coder_qat_c1a_entropy_weight=float(self.coder_qat_c1a_entropy_weight),
            coder_qat_c1a_sigma=float(self.coder_qat_c1a_sigma),
            coder_qat_c1a_sample_size=int(self.coder_qat_c1a_sample_size),
            segnet_student_live_calibration_weight=float(
                self.segnet_student_live_calibration_weight
            ),
            segnet_distillation_objective=str(self.segnet_distillation_objective),
            distillation_temperature=float(self.distillation_temperature),
            segnet_direct_live_distillation_weight=float(
                self.segnet_direct_live_distillation_weight
            ),
            segnet_direct_live_base_loss_weight=float(
                self.segnet_direct_live_base_loss_weight
            ),
            segnet_direct_live_class_histogram_weight=float(
                self.segnet_direct_live_class_histogram_weight
            ),
            segnet_direct_live_class_balanced_hinge_weight=float(
                self.segnet_direct_live_class_balanced_hinge_weight
            ),
            segnet_direct_live_class_balanced_ce_weight=float(
                self.segnet_direct_live_class_balanced_ce_weight
            ),
            segnet_direct_live_class_balanced_squared_hinge_weight=float(
                self.segnet_direct_live_class_balanced_squared_hinge_weight
            ),
            segnet_tau_boundary=float(self.segnet_tau_boundary),
            segnet_hinge_margin=float(self.segnet_hinge_margin),
            decoder_fake_quant_bits=int(self.decoder_fake_quant_bits),
            train_time_decoder_pruning_ratio=float(self.train_time_decoder_pruning_ratio),
            train_time_decoder_quant_noise_bits=(
                None
                if self.train_time_decoder_quant_noise_bits is None
                else int(self.train_time_decoder_quant_noise_bits)
            ),
            train_time_decoder_quant_noise_scale=float(
                self.train_time_decoder_quant_noise_scale
            ),
            train_time_decoder_quant_noise_seed=int(self.train_time_decoder_quant_noise_seed),
            train_time_decoder_control_start_epoch=int(
                self.train_time_decoder_control_start_epoch
            ),
            train_time_decoder_control_frequency_epochs=int(
                self.train_time_decoder_control_frequency_epochs
            ),
            export_decoder_pruning_ratio=float(self.export_decoder_pruning_ratio),
            export_decoder_quant_noise_bits=(
                None
                if self.export_decoder_quant_noise_bits is None
                else int(self.export_decoder_quant_noise_bits)
            ),
            export_decoder_quant_noise_scale=float(self.export_decoder_quant_noise_scale),
            export_decoder_quant_noise_seed=int(self.export_decoder_quant_noise_seed),
            scorer_input_distribution_guard_weight=float(
                self.scorer_input_distribution_guard_weight
            ),
            scorer_input_distribution_guard_saturation_margin=float(
                self.scorer_input_distribution_guard_saturation_margin
            ),
            scorer_input_distribution_guard_temperature=float(
                self.scorer_input_distribution_guard_temperature
            ),
            scorer_input_contrast_floor_weight=float(
                self.scorer_input_contrast_floor_weight
            ),
            scorer_input_contrast_floor_segnet_min_std_ratio=float(
                self.scorer_input_contrast_floor_segnet_min_std_ratio
            ),
            scorer_input_contrast_floor_posenet_yuv6_min_std_ratio=float(
                self.scorer_input_contrast_floor_posenet_yuv6_min_std_ratio
            ),
            scorer_input_shape_tether_weight=float(
                self.scorer_input_shape_tether_weight
            ),
            posenet_temporal_signal_floor_weight=float(
                self.posenet_temporal_signal_floor_weight
            ),
            posenet_temporal_signal_min_std_ratio=float(
                self.posenet_temporal_signal_min_std_ratio
            ),
            posenet_temporal_signal_min_mean_abs_ratio=float(
                self.posenet_temporal_signal_min_mean_abs_ratio
            ),
            output_head_target_bias_init_enabled=bool(
                self.output_head_target_bias_init_enabled
            ),
            output_head_target_bias_init_epsilon=float(
                self.output_head_target_bias_init_epsilon
            ),
            output_head_target_contrast_init_enabled=bool(
                self.output_head_target_contrast_init_enabled
            ),
            output_head_target_contrast_init_max_pairs=int(
                self.output_head_target_contrast_init_max_pairs
            ),
            output_head_target_contrast_init_min_output_std=float(
                self.output_head_target_contrast_init_min_output_std
            ),
            output_head_target_contrast_init_max_gain=float(
                self.output_head_target_contrast_init_max_gain
            ),
        )

    @property
    def train_time_decoder_controls_enabled(self) -> bool:
        return bool(
            float(self.train_time_decoder_pruning_ratio) > 0.0
            or (
                self.train_time_decoder_quant_noise_bits is not None
                and float(self.train_time_decoder_quant_noise_scale) > 0.0
            )
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": HI_NERV_TRAIN_TIME_CONTROL_SCHEMA,
            "stage_loss_schedule": self.stage_loss_schedule,
            "optimizer_kind": self.optimizer_kind,
            "optimizer_surface": (
                "pr95_faithful_stage_descriptors"
                if self.pr95_faithful_curriculum_enabled
                else "shared_mlx_score_aware_adapter"
            ),
            "pr95_faithful_curriculum_enabled": bool(
                self.pr95_faithful_curriculum_enabled
            ),
            "pr95_curriculum_total_epochs": self.pr95_curriculum_total_epochs,
            "staged_scorer_curriculum_enabled": bool(
                self.staged_scorer_curriculum_enabled
            ),
            "coder_qat_enabled": bool(self.coder_qat_enabled),
            "coder_qat_quant_bits": int(self.coder_qat_quant_bits),
            "coder_qat_c1a_entropy_weight": float(
                self.coder_qat_c1a_entropy_weight
            ),
            "coder_qat_c1a_sigma": float(self.coder_qat_c1a_sigma),
            "coder_qat_c1a_sample_size": int(self.coder_qat_c1a_sample_size),
            "segnet_student_live_calibration_weight": float(
                self.segnet_student_live_calibration_weight
            ),
            "segnet_direct_live": {
                "schema": "hi_nerv_train_time_segnet_direct_live_control.v1",
                "enabled": float(self.segnet_direct_live_distillation_weight) > 0.0,
                "objective": str(self.segnet_distillation_objective),
                "distillation_temperature": float(self.distillation_temperature),
                "weight": float(self.segnet_direct_live_distillation_weight),
                "base_loss_weight": float(self.segnet_direct_live_base_loss_weight),
                "class_histogram_weight": float(
                    self.segnet_direct_live_class_histogram_weight
                ),
                "class_balanced_hinge_weight": float(
                    self.segnet_direct_live_class_balanced_hinge_weight
                ),
                "class_balanced_ce_weight": float(
                    self.segnet_direct_live_class_balanced_ce_weight
                ),
                "class_balanced_squared_hinge_weight": float(
                    self.segnet_direct_live_class_balanced_squared_hinge_weight
                ),
                "tau_boundary": float(self.segnet_tau_boundary),
                "hinge_margin": float(self.segnet_hinge_margin),
                "target_surface": "upstream_segnet_last_frame_logits",
                "human_visual_fidelity_objective": False,
            },
            "decoder_fake_quant_forward_enabled": bool(
                self.decoder_fake_quant_forward_enabled
            ),
            "decoder_fake_quant_bits": int(self.decoder_fake_quant_bits),
            "train_time_decoder_controls_enabled": (
                self.train_time_decoder_controls_enabled
            ),
            "train_time_decoder_pruning_ratio": float(
                self.train_time_decoder_pruning_ratio
            ),
            "train_time_decoder_quant_noise_bits": (
                None
                if self.train_time_decoder_quant_noise_bits is None
                else int(self.train_time_decoder_quant_noise_bits)
            ),
            "train_time_decoder_quant_noise_scale": float(
                self.train_time_decoder_quant_noise_scale
            ),
            "train_time_decoder_quant_noise_seed": int(
                self.train_time_decoder_quant_noise_seed
            ),
            "train_time_decoder_control_start_epoch": int(
                self.train_time_decoder_control_start_epoch
            ),
            "train_time_decoder_control_frequency_epochs": int(
                self.train_time_decoder_control_frequency_epochs
            ),
            "export_decoder_pruning_ratio": float(self.export_decoder_pruning_ratio),
            "export_decoder_quant_noise_bits": (
                None
                if self.export_decoder_quant_noise_bits is None
                else int(self.export_decoder_quant_noise_bits)
            ),
            "export_decoder_quant_noise_scale": float(
                self.export_decoder_quant_noise_scale
            ),
            "export_decoder_quant_noise_seed": int(self.export_decoder_quant_noise_seed),
            "scorer_input_distribution_guard": {
                "schema": "hi_nerv_train_time_scorer_input_distribution_guard.v1",
                "enabled": float(self.scorer_input_distribution_guard_weight) > 0.0,
                "weight": float(self.scorer_input_distribution_guard_weight),
                "components": [
                    "rgb_mean",
                    "rgb_std",
                    "rgb_dynamic_range",
                    "soft_saturation_mass",
                    "segnet_frame1_mse",
                    "segnet_frame1_mae",
                    "posenet_yuv6_pair_mean",
                    "posenet_yuv6_pair_std",
                    "posenet_yuv6_pair_dynamic_range",
                    "posenet_yuv6_pair_mse",
                    "posenet_yuv6_pair_mae",
                    "posenet_yuv6_temporal_delta",
                    "posenet_yuv6_temporal_delta_mse",
                    "posenet_yuv6_temporal_delta_mae",
                ],
                "dynamic_range_repair_before_replay": True,
                "saturation_margin": float(
                    self.scorer_input_distribution_guard_saturation_margin
                ),
                "temperature": float(
                    self.scorer_input_distribution_guard_temperature
                ),
            },
            "scorer_input_contrast_floor": {
                "schema": "hi_nerv_train_time_scorer_input_contrast_floor.v1",
                "enabled": float(self.scorer_input_contrast_floor_weight) > 0.0,
                "weight": float(self.scorer_input_contrast_floor_weight),
                "segnet_last_rgb_min_std_ratio": float(
                    self.scorer_input_contrast_floor_segnet_min_std_ratio
                ),
                "posenet_yuv6_pair_min_std_ratio": float(
                    self.scorer_input_contrast_floor_posenet_yuv6_min_std_ratio
                ),
                "target_surface": (
                    "segnet_last_frame_rgb_and_posenet_two_frame_yuv6_std_ratio"
                ),
                "human_visual_fidelity_objective": False,
            },
            "scorer_input_shape_tether": {
                "schema": "hi_nerv_train_time_scorer_input_shape_tether.v1",
                "enabled": float(self.scorer_input_shape_tether_weight) > 0.0,
                "weight": float(self.scorer_input_shape_tether_weight),
                "components": [
                    "segnet_last_frame_rgb_centered_reference_variance_fit",
                    "posenet_yuv6_pair_centered_reference_variance_fit",
                    "posenet_yuv6_temporal_delta_centered_reference_variance_fit",
                ],
                "target_surface": (
                    "exact_upstream_scorer_inputs_centered_and_normalized_by_"
                    "reference_channel_variance"
                ),
                "human_visual_fidelity_objective": False,
            },
            "posenet_temporal_signal_floor": {
                "schema": "hi_nerv_train_time_posenet_temporal_signal_floor.v1",
                "enabled": float(self.posenet_temporal_signal_floor_weight) > 0.0,
                "weight": float(self.posenet_temporal_signal_floor_weight),
                "min_std_ratio": float(self.posenet_temporal_signal_min_std_ratio),
                "min_mean_abs_ratio": float(
                    self.posenet_temporal_signal_min_mean_abs_ratio
                ),
                "target_surface": (
                    "exact_upstream_posenet_yuv6_frame1_minus_frame0_temporal_signal"
                ),
                "reason": (
                    "HiNeRV compact smokes showed pairwise YUV6 temporal signal "
                    "collapsing to near zero while archive bytes looked healthy; "
                    "PoseNet cannot recover ego-motion from identical decoded frames."
                ),
                "human_visual_fidelity_objective": False,
            },
            "output_head_target_bias_init": {
                "schema": "hi_nerv_output_head_target_bias_init_control.v1",
                "enabled": bool(self.output_head_target_bias_init_enabled),
                "epsilon": float(self.output_head_target_bias_init_epsilon),
                "closed_form": "bias=logit(clamp(mean(target_rgb_channel),eps,1-eps))",
                "runtime_sidecar_bytes": 0,
                "archive_charged_decoder_tensors": [
                    "head_rgb_0.bias",
                    "head_rgb_1.bias",
                ],
            },
            "output_head_target_contrast_init": {
                "schema": "hi_nerv_output_head_target_contrast_init_control.v1",
                "enabled": bool(self.output_head_target_contrast_init_enabled),
                "max_pairs": int(self.output_head_target_contrast_init_max_pairs),
                "min_output_std": float(
                    self.output_head_target_contrast_init_min_output_std
                ),
                "max_gain": float(self.output_head_target_contrast_init_max_gain),
                "closed_form": (
                    "head_rgb_weight *= clip(std(target_rgb)/std(output_rgb), "
                    "1/max_gain, max_gain), with std(output_rgb) floored by "
                    "min_output_std"
                ),
                "target_surface": "upstream_scorer_resolution_rgb_contrast",
                "runtime_sidecar_bytes": 0,
                "archive_charged_decoder_tensors": [
                    "head_rgb_0.weight",
                    "head_rgb_1.weight",
                ],
                "human_visual_fidelity_objective": False,
            },
            "authority": TRAINER_AUTHORITY,
            **FALSE_AUTHORITY,
        }


def _full_main(args: argparse.Namespace) -> int:
    """Run canonical MLX score-aware training for the current HiNeRV carrier."""

    train_time_controls = _train_time_control_config_from_args(args)
    modelsize_candidate = _modelsize_candidate_from_args(args)
    modelsize_candidate_consumption = _modelsize_candidate_consumption_metadata(
        args=args,
        candidate=modelsize_candidate,
    )
    pr95_full_control_contract = _pr95_full_control_contract(
        args,
        train_time_controls=train_time_controls,
    )
    canonicalization = _direct_trainer_canonicalization_contract(
        mode="full",
        modelsize_candidate_consumption=modelsize_candidate_consumption,
    )
    launch_refusal = _direct_trainer_launch_refusal_payload(
        canonicalization,
        mode="full",
        pr95_full_control_contract=pr95_full_control_contract,
    )
    if launch_refusal is not None:
        print(json.dumps(launch_refusal, sort_keys=True), file=sys.stderr)
        return 2

    from tac.substrates._shared.mlx_score_aware import (
        RendererBundle,
        build_mlx_posenet_pair_teacher,
        build_mlx_segnet_pair_teacher,
        decode_mlx_targets,
        run_mlx_score_aware_full_main,
    )
    from tac.substrates._shared.mlx_score_aware.coder_qat import (
        build_decoder_coder_qat_terms,
        coder_qat_loss_weights,
        coder_qat_metadata,
    )
    from tac.substrates.hi_nerv.archive_candidate import export_hi_nerv_mlx_archive
    from tac.substrates.hi_nerv.mlx_renderer import HinervSubstrateMLX
    from tac.substrates.hinton_distilled_scorer_surrogate import (
        DEFAULT_POSE_DIMS,
        DEFAULT_SEGNET_CLASSES,
        build_learnable_pose_student_head,
        build_learnable_student_head,
    )

    output_dir, storage_payload = _resolve_output_dir(args)
    cfg = _config_from_args(args, modelsize_candidate=modelsize_candidate)
    effective_decoder_codec = _decoder_codec_from_args(
        args,
        modelsize_candidate=modelsize_candidate,
    )
    modelsize_hard_byte_ceiling = _hard_byte_ceiling_from_args(
        args,
        modelsize_candidate=modelsize_candidate,
    )
    prioritized_pair_indices = _prioritized_pair_indices_from_args(args)
    source_pair_indices = prioritized_pair_indices or None
    effective_training_num_pairs = len(source_pair_indices) if source_pair_indices is not None else int(cfg.num_pairs)
    local_training_pair_indices = (
        tuple(range(effective_training_num_pairs)) if source_pair_indices is not None else prioritized_pair_indices
    )
    optimizer_control = _optimizer_control_metadata_from_args(args)
    decoder_weight_waterfill_plan = _decoder_weight_waterfill_plan_from_args(
        args,
        modelsize_candidate=modelsize_candidate,
    )
    model = HinervSubstrateMLX(cfg)
    decoder_fake_quant_forward = _configure_decoder_fake_quant_forward(
        model=model,
        controls=train_time_controls,
        decoder_weight_waterfill_plan=decoder_weight_waterfill_plan,
    )
    projection_hook = _build_train_time_decoder_control_callback(
        model=model,
        controls=train_time_controls,
        output_dir=output_dir,
    )
    if projection_hook is not None:
        model.post_optimizer_projection = projection_hook
    coder_qat_cfg = _coder_qat_config_from_args(args)
    coder_qat_loss_weight_map = coder_qat_loss_weights(coder_qat_cfg)
    extra_loss_terms = None
    if coder_qat_cfg.enabled:

        def _extra_loss_terms(model_obj: Any, _idx: Any) -> dict[str, Any]:
            return dict(build_decoder_coder_qat_terms(model_obj, coder_qat_cfg))

        extra_loss_terms = _extra_loss_terms

    target_rgb_0, target_rgb_1 = decode_mlx_targets(
        args.video_path,
        num_pairs=int(cfg.num_pairs),
        output_height=int(cfg.output_height),
        output_width=int(cfg.output_width),
        pair_indices=source_pair_indices,
    )
    output_head_target_bias_init = _initialize_output_head_target_bias(
        model=model,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        controls=train_time_controls,
        source_pair_indices=source_pair_indices,
    )
    (
        train_time_section_byte_metrics,
        train_time_section_byte_control,
    ) = _build_hi_nerv_train_time_section_byte_metrics_callback(
        args=args,
        model=model,
        decoder_codec=effective_decoder_codec,
        controls=train_time_controls,
        decoder_weight_waterfill_plan=decoder_weight_waterfill_plan,
        hard_byte_ceiling=modelsize_hard_byte_ceiling,
        active_loss_weights=coder_qat_loss_weight_map,
    )
    train_time_dual_ascent_config = _train_time_dual_ascent_config_from_args(
        args,
        train_time_controls=train_time_controls,
        coder_qat_loss_weight_map=coder_qat_loss_weight_map,
        section_byte_control=(
            train_time_section_byte_control.get("initial_control")
            if bool(train_time_section_byte_control.get("enabled"))
            else None
        ),
    )
    pr95_full_control_contract = _pr95_full_control_contract(
        args,
        train_time_controls=train_time_controls,
        train_time_section_byte_control=(
            train_time_section_byte_control.get("initial_control")
            if bool(train_time_section_byte_control.get("enabled"))
            else train_time_section_byte_control
        ),
        require_measured_section_byte_control=True,
    )
    if pr95_full_control_contract.get("production_full_control_ready") is not True:
        refusal = _direct_trainer_launch_refusal_payload(
            {
                **canonicalization,
                "trainer_launch_allowed": True,
                "blockers": [],
            },
            mode="full",
            pr95_full_control_contract=pr95_full_control_contract,
        )
        write_json(
            output_dir / "hi_nerv_mlx_training_launch_preflight.json",
            {
                "schema": TRAINER_SCHEMA,
                "authority": TRAINER_AUTHORITY,
                "output_dir": output_dir.as_posix(),
                "storage_preflight": storage_payload,
                "direct_trainer_canonicalization": canonicalization,
                "pr95_full_control_contract": pr95_full_control_contract,
                "train_time_controls": train_time_controls.metadata(),
                "train_time_section_byte_control": _metadata_safe(
                    train_time_section_byte_control
                ),
                "train_time_dual_ascent_config": _metadata_safe(
                    train_time_dual_ascent_config
                ),
                "modelsize_candidate_consumption": modelsize_candidate_consumption,
                "blockers": list(pr95_full_control_contract.get("blockers") or []),
                "training_executed": False,
                "export_executed": False,
                "command": sys.argv,
                **FALSE_AUTHORITY,
            },
        )
        print(json.dumps(refusal, sort_keys=True), file=sys.stderr)
        return 2

    scorer_teacher = None
    pose_scorer_teacher = None
    learnable_student_head = None
    learnable_pose_student_head = None
    pose_distillation_weight = 0.0
    segnet_teacher_needed = bool(
        (
            float(args.distillation_weight) > 0.0
            and not bool(args.allow_mock_scorer_teacher)
        )
        or float(train_time_controls.segnet_direct_live_distillation_weight) > 0.0
    )
    pose_teacher_needed = bool(float(args.pose_distillation_weight) > 0.0)
    if segnet_teacher_needed or pose_teacher_needed:
        bundle_no_teacher = RendererBundle(
            model=model,
            target_rgb_0=target_rgb_0,
            target_rgb_1=target_rgb_1,
            num_pairs=effective_training_num_pairs,
            forward_convention="call_b2chw_255",
            distillation_weight=0.0,
            pose_distillation_weight=0.0,
            pose_dims=DEFAULT_POSE_DIMS,
            source_pair_indices=source_pair_indices,
        )
    if segnet_teacher_needed:
        scorer_teacher = build_mlx_segnet_pair_teacher(
            bundle_no_teacher,
            upstream_dir=str(args.upstream_dir),
            device="cpu",
        )
        if float(args.distillation_weight) > 0.0:
            learnable_student_head = build_learnable_student_head(
                num_classes=DEFAULT_SEGNET_CLASSES,
                in_channels=3,
                seed=int(args.seed),
            )
    if pose_teacher_needed:
        pose_scorer_teacher = build_mlx_posenet_pair_teacher(
            bundle_no_teacher,
            upstream_dir=str(args.upstream_dir),
            device="cpu",
        )
        learnable_pose_student_head = build_learnable_pose_student_head(
            pose_dims=DEFAULT_POSE_DIMS,
            input_channels=_pose_student_input_channels(str(args.pose_student_input_preprocess)),
            seed=int(args.seed),
        )
        pose_distillation_weight = float(args.pose_distillation_weight)

    bundle = RendererBundle(
        model=model,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        num_pairs=effective_training_num_pairs,
        forward_convention="call_b2chw_255",
        extra_loss_terms=extra_loss_terms,
        extra_loss_weights=coder_qat_loss_weight_map,
        distillation_weight=float(args.distillation_weight),
        scorer_teacher=scorer_teacher,
        learnable_student_head=learnable_student_head,
        pose_distillation_weight=pose_distillation_weight,
        pose_scorer_teacher=pose_scorer_teacher,
        learnable_pose_student_head=learnable_pose_student_head,
        pose_dims=DEFAULT_POSE_DIMS,
        distillation_temperature=float(train_time_controls.distillation_temperature),
        segnet_distillation_objective=str(
            train_time_controls.segnet_distillation_objective
        ),
        segnet_student_live_calibration_weight=(
            float(train_time_controls.segnet_student_live_calibration_weight)
            if scorer_teacher is not None
            else 0.0
        ),
        segnet_direct_live_distillation_weight=(
            float(train_time_controls.segnet_direct_live_distillation_weight)
            if scorer_teacher is not None
            else 0.0
        ),
        segnet_direct_live_base_loss_weight=float(
            train_time_controls.segnet_direct_live_base_loss_weight
        ),
        segnet_direct_live_class_histogram_weight=(
            float(train_time_controls.segnet_direct_live_class_histogram_weight)
            if scorer_teacher is not None
            else 0.0
        ),
        segnet_direct_live_class_balanced_hinge_weight=(
            float(train_time_controls.segnet_direct_live_class_balanced_hinge_weight)
            if scorer_teacher is not None
            else 0.0
        ),
        segnet_direct_live_class_balanced_ce_weight=(
            float(train_time_controls.segnet_direct_live_class_balanced_ce_weight)
            if scorer_teacher is not None
            else 0.0
        ),
        segnet_direct_live_class_balanced_squared_hinge_weight=(
            float(
                train_time_controls.segnet_direct_live_class_balanced_squared_hinge_weight
            )
            if scorer_teacher is not None
            else 0.0
        ),
        segnet_tau_boundary=float(train_time_controls.segnet_tau_boundary),
        segnet_hinge_margin=float(train_time_controls.segnet_hinge_margin),
        allow_mock_scorer_teacher=bool(args.allow_mock_scorer_teacher),
        allow_segnet_only_research=bool(args.allow_segnet_only_research),
        scorer_input_distribution_guard_weight=(
            train_time_controls.scorer_input_distribution_guard_weight
        ),
        scorer_input_distribution_guard_saturation_margin=(
            train_time_controls.scorer_input_distribution_guard_saturation_margin
        ),
        scorer_input_distribution_guard_temperature=(
            train_time_controls.scorer_input_distribution_guard_temperature
        ),
        scorer_input_contrast_floor_weight=(
            train_time_controls.scorer_input_contrast_floor_weight
        ),
        scorer_input_contrast_floor_segnet_min_std_ratio=(
            train_time_controls.scorer_input_contrast_floor_segnet_min_std_ratio
        ),
        scorer_input_contrast_floor_posenet_yuv6_min_std_ratio=(
            train_time_controls.scorer_input_contrast_floor_posenet_yuv6_min_std_ratio
        ),
        scorer_input_shape_tether_weight=(
            train_time_controls.scorer_input_shape_tether_weight
        ),
        posenet_temporal_signal_floor_weight=(
            train_time_controls.posenet_temporal_signal_floor_weight
        ),
        posenet_temporal_signal_min_std_ratio=(
            train_time_controls.posenet_temporal_signal_min_std_ratio
        ),
        posenet_temporal_signal_min_mean_abs_ratio=(
            train_time_controls.posenet_temporal_signal_min_mean_abs_ratio
        ),
        export_archive_fn=lambda model_obj, out_dir: export_hi_nerv_mlx_archive(
            model_obj,
            out_dir,
            repo_root=REPO_ROOT,
            decoder_codec=effective_decoder_codec,
            source_backend="mlx",
            pruning_ratio=float(train_time_controls.export_decoder_pruning_ratio),
            quant_noise_bits=train_time_controls.export_decoder_quant_noise_bits,
            quant_noise_scale=float(train_time_controls.export_decoder_quant_noise_scale),
            quant_noise_seed=int(train_time_controls.export_decoder_quant_noise_seed),
            decoder_weight_waterfill_plan=decoder_weight_waterfill_plan,
            hard_byte_ceiling=modelsize_hard_byte_ceiling,
        ),
        substrate_artifact_metadata={
            "schema": TRAINER_SCHEMA,
            "authority": TRAINER_AUTHORITY,
            "family": "hi_nerv",
            "source_fidelity_status": "local_hi_nerv_fork_not_official_hinerv_parity",
            "modelsize_row": args.modelsize_row,
            "modelsize_candidate_consumption": _metadata_safe(
                modelsize_candidate_consumption
            ),
            "config": _config_snapshot(cfg),
            "training_target_pair_count": int(effective_training_num_pairs),
            "source_pair_indices": (
                [int(value) for value in source_pair_indices] if source_pair_indices is not None else None
            ),
            "local_training_pair_indices": [int(value) for value in local_training_pair_indices],
            "pair_index_alignment_mode": (
                "local_target_rows_to_source_pair_indices"
                if source_pair_indices is not None
                else "identity_local_rows_are_source_pairs"
            ),
            "optimizer_control": _metadata_safe(optimizer_control),
            "decoder_codec": effective_decoder_codec,
            "decoder_fake_quant_forward": _metadata_safe(decoder_fake_quant_forward),
            "decoder_weight_waterfill_plan": _metadata_safe(
                _decoder_weight_waterfill_plan_attachment_metadata(
                    args=args,
                    plan=decoder_weight_waterfill_plan,
                    fake_quant_forward=decoder_fake_quant_forward,
                )
            ),
            "coder_qat": coder_qat_metadata(coder_qat_cfg),
            "train_time_controls": _metadata_safe(train_time_controls.metadata()),
            "train_time_section_byte_control": _metadata_safe(
                train_time_section_byte_control
            ),
            "train_time_dual_ascent_config": _metadata_safe(
                train_time_dual_ascent_config
            ),
            "output_head_target_bias_init": _metadata_safe(
                output_head_target_bias_init
            ),
            "eval_roundtrip_ste_enabled": bool(args.eval_roundtrip_ste),
            "pose_student_input_preprocess": str(args.pose_student_input_preprocess),
            "scorer_input_distribution_guard": _metadata_safe(
                train_time_controls.metadata()["scorer_input_distribution_guard"]
            ),
            "scorer_input_contrast_floor": _metadata_safe(
                train_time_controls.metadata()["scorer_input_contrast_floor"]
            ),
            "posenet_temporal_signal_floor": _metadata_safe(
                train_time_controls.metadata()["posenet_temporal_signal_floor"]
            ),
            "segnet_direct_live": _metadata_safe(
                train_time_controls.metadata()["segnet_direct_live"]
            ),
            "pose_student_input_channels": _pose_student_input_channels(str(args.pose_student_input_preprocess)),
            "prioritized_pair_training": _prioritized_pair_training_lineage_metadata(
                prioritized_pair_indices,
                target_hydration_pair_indices_consumed=source_pair_indices is not None,
            ),
            "storage_preflight": _metadata_safe(storage_payload),
            "direct_trainer_canonicalization": _metadata_safe(canonicalization),
            "pr95_full_control_contract": _metadata_safe(pr95_full_control_contract),
            "blockers": [
                "contest_cpu_cuda_exact_eval_not_executed",
                "official_hinerv_feature_grid_parity_not_proven",
                *canonicalization["blockers"],
                *pr95_full_control_contract["blockers"],
            ],
        },
        eval_roundtrip_ste_enabled=bool(args.eval_roundtrip_ste),
        pose_student_input_preprocess=str(args.pose_student_input_preprocess),
        source_pair_indices=source_pair_indices,
        train_time_section_byte_metrics=train_time_section_byte_metrics,
    )
    write_json(
        output_dir / "hi_nerv_mlx_training_launch_preflight.json",
        {
            "schema": TRAINER_SCHEMA,
            "authority": TRAINER_AUTHORITY,
            "output_dir": output_dir.as_posix(),
            "storage_preflight": storage_payload,
            "direct_trainer_canonicalization": canonicalization,
            "pr95_full_control_contract": pr95_full_control_contract,
            "train_time_controls": train_time_controls.metadata(),
            "train_time_section_byte_control": _metadata_safe(
                train_time_section_byte_control
            ),
            "train_time_dual_ascent_config": _metadata_safe(
                train_time_dual_ascent_config
            ),
            "output_head_target_bias_init": _metadata_safe(
                output_head_target_bias_init
            ),
            "modelsize_candidate_consumption": modelsize_candidate_consumption,
            "prioritized_pair_training": _prioritized_pair_training_metadata(
                prioritized_pair_indices,
                target_hydration_pair_indices_consumed=source_pair_indices is not None,
            ),
            "optimizer_control": _metadata_safe(optimizer_control),
            "blockers": [
                *canonicalization["blockers"],
                *pr95_full_control_contract["blockers"],
            ],
            "command": sys.argv,
            **FALSE_AUTHORITY,
        },
    )
    artifact = run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="hi_nerv_mlx_local",
        lane_id="lane_hi_nerv_mlx_score_aware_local_20260602",
        output_dir=output_dir,
        epochs=int(args.epochs),
        batch_pair_indices_per_step=min(int(args.batch_pairs), int(effective_training_num_pairs)),
        learning_rate=float(args.full_lr),
        seed=int(args.seed),
        checkpoint_interval_epochs=int(args.checkpoint_interval_epochs),
        curriculum_stages=_curriculum_stages_from_args(args),
        pr95_faithful_curriculum_enabled=bool(args.pr95_faithful_curriculum),
        pr95_curriculum_total_epochs=args.pr95_curriculum_total_epochs,
        pr95_stage_source_weight_amplification_enabled=bool(
            optimizer_control["pr95_stage_source_weight_amplification_enabled"]
        ),
        grad_clip_max_norm=args.grad_clip_max_norm,
        warmup_epochs=int(args.warmup_epochs),
        warmup_steps_per_epoch=int(optimizer_control["warmup_steps_per_epoch"]),
        weight_decay=args.weight_decay,
        optimizer_kind=str(args.optimizer_kind),
        cosine_decay_enabled=bool(args.cosine_decay),
        cosine_decay_total_epochs=args.cosine_decay_total_epochs,
        cosine_decay_min_lr_ratio=float(args.cosine_decay_min_lr_ratio),
        ema_archive_selection_enabled=bool(args.ema_archive_selection),
        train_time_dual_ascent_config=train_time_dual_ascent_config,
        prioritized_pair_indices=local_training_pair_indices,
        pair_sampling_weights=optimizer_control["pair_sampling_weights"],
        pair_sampling_default_weight=float(
            optimizer_control["pair_sampling_default_weight"]
        ),
        gradient_multiplier_by_name=optimizer_control["gradient_multiplier_by_name"],
        bias_gradient_multiplier=optimizer_control["bias_gradient_multiplier"],
        output_head_bias_gradient_multiplier=float(
            optimizer_control["output_head_bias_gradient_multiplier"]
        ),
        notes=(
            "HiNeRV MLX-local score-aware training through the canonical "
            "mlx_score_aware harness, with optional real SegNet/PoseNet teacher "
            "binding, PR95 faithful curriculum, coder-aware QAT, eval-roundtrip "
            "STE, and archive export. False-authority until contest CPU/CUDA replay."
        ),
    )
    post_export_quality = _maybe_write_post_export_receiver_cache_quality(
        args=args,
        output_dir=output_dir,
        archive_path=getattr(artifact, "archive_path", None),
    )
    if post_export_quality is not None:
        _attach_post_export_receiver_cache_quality_to_training_artifact(
            output_dir=output_dir,
            report=post_export_quality,
        )
    short_scorer_smoke_readiness = _write_hinerv_short_scorer_smoke_readiness(
        output_dir=output_dir,
        train_time_controls=train_time_controls,
        final_loss_components=_final_loss_components_from_training_artifact(artifact),
        post_export_quality=post_export_quality,
        segnet_distillation_weight=float(args.distillation_weight),
        pose_distillation_weight=float(args.pose_distillation_weight),
        allow_mock_scorer_teacher=bool(args.allow_mock_scorer_teacher),
        min_segnet_occupied_class_fraction_for_fit_gate=float(
            args.receiver_cache_quality_min_segnet_argmax_occupied_class_fraction_for_fit_gate
        ),
    )
    _attach_hinerv_short_scorer_smoke_readiness_to_training_artifact(
        output_dir=output_dir,
        report=short_scorer_smoke_readiness,
    )
    print(
        json.dumps(
            {
                "schema": TRAINER_SCHEMA,
                "output_dir": output_dir.as_posix(),
                "epochs": artifact.total_epochs_completed,
                "archive_bytes": getattr(artifact, "archive_bytes", None),
                "post_export_receiver_cache_quality_report": (
                    post_export_quality.get("report_path") if post_export_quality is not None else None
                ),
                "post_export_receiver_cache_quality_passed": (
                    bool(post_export_quality.get("quality_gate_passed")) if post_export_quality is not None else False
                ),
                "short_scorer_teacher_smoke_readiness_report": (
                    short_scorer_smoke_readiness.get("report_path")
                ),
                "short_scorer_teacher_smoke_ready": bool(
                    short_scorer_smoke_readiness.get("short_scorer_teacher_smoke_ready")
                ),
                "short_scorer_teacher_smoke_actionable_blockers": list(
                    short_scorer_smoke_readiness.get("actionable_blockers") or []
                ),
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _smoke_main(args: argparse.Namespace) -> int:
    """Run a small real MLX forward/export smoke for the HiNeRV binding."""

    try:
        import mlx.core as mx
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: MLX import failed: {exc!r}", file=sys.stderr)
        return 2
    from tac.substrates._shared.mlx_score_aware import decode_mlx_targets
    from tac.substrates.hi_nerv.archive_candidate import export_hi_nerv_mlx_archive
    from tac.substrates.hi_nerv.mlx_renderer import MLX_EVIDENCE_GRADE, HinervSubstrateMLX

    output_dir, storage_payload = _resolve_output_dir(args)
    modelsize_candidate = _modelsize_candidate_from_args(args)
    modelsize_candidate_consumption = _modelsize_candidate_consumption_metadata(
        args=args,
        candidate=modelsize_candidate,
    )
    cfg = _config_from_args(args, modelsize_candidate=modelsize_candidate)
    effective_decoder_codec = _decoder_codec_from_args(
        args,
        modelsize_candidate=modelsize_candidate,
    )
    modelsize_hard_byte_ceiling = _hard_byte_ceiling_from_args(
        args,
        modelsize_candidate=modelsize_candidate,
    )
    canonicalization = _direct_trainer_canonicalization_contract(
        mode="smoke",
        modelsize_candidate_consumption=modelsize_candidate_consumption,
    )
    train_time_controls = _train_time_control_config_from_args(args)
    prioritized_pair_indices = _prioritized_pair_indices_from_args(args)
    decoder_weight_waterfill_plan = _decoder_weight_waterfill_plan_from_args(
        args,
        modelsize_candidate=modelsize_candidate,
    )
    model = HinervSubstrateMLX(cfg)
    decoder_fake_quant_forward = _configure_decoder_fake_quant_forward(
        model=model,
        controls=train_time_controls,
        decoder_weight_waterfill_plan=decoder_weight_waterfill_plan,
    )
    idx = mx.array(list(range(min(2, int(cfg.num_pairs)))), dtype=mx.int32)
    smoke_target_rgb_0, smoke_target_rgb_1 = decode_mlx_targets(
        args.video_path,
        num_pairs=int(idx.shape[0]),
        output_height=int(cfg.output_height),
        output_width=int(cfg.output_width),
    )
    output_head_target_bias_init = _initialize_output_head_target_bias(
        model=model,
        target_rgb_0=smoke_target_rgb_0,
        target_rgb_1=smoke_target_rgb_1,
        controls=train_time_controls,
        source_pair_indices=None,
    )
    output = model(idx)
    mx.eval(output)
    forward_smoke_stats = _smoke_forward_statistics(
        output=output,
        target_rgb_0=smoke_target_rgb_0,
        target_rgb_1=smoke_target_rgb_1,
    )
    archive_path = archive_sha256 = None
    archive_bytes = None
    if args.smoke_export_archive:
        archive_path_obj, archive_sha256, archive_bytes = export_hi_nerv_mlx_archive(
            model,
            output_dir / "smoke_archive_export",
            repo_root=REPO_ROOT,
            decoder_codec=effective_decoder_codec,
            source_backend="mlx",
            pruning_ratio=float(train_time_controls.export_decoder_pruning_ratio),
            quant_noise_bits=train_time_controls.export_decoder_quant_noise_bits,
            quant_noise_scale=float(train_time_controls.export_decoder_quant_noise_scale),
            quant_noise_seed=int(train_time_controls.export_decoder_quant_noise_seed),
            decoder_weight_waterfill_plan=decoder_weight_waterfill_plan,
            hard_byte_ceiling=modelsize_hard_byte_ceiling,
        )
        archive_path = archive_path_obj.as_posix()
    byte_cap_control = _build_hinerv_hard_byte_ceiling_control(
        candidate=modelsize_candidate,
        hard_byte_ceiling=modelsize_hard_byte_ceiling,
        archive_path=archive_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        archive_export_requested=bool(args.smoke_export_archive),
    )
    post_export_quality = _maybe_write_post_export_receiver_cache_quality(
        args=args,
        output_dir=output_dir,
        archive_path=Path(archive_path) if archive_path else None,
    )
    manifest = {
        "schema": "hi_nerv_mlx_trainer_smoke.v1",
        "authority": TRAINER_AUTHORITY,
        "axis_tag": MLX_EVIDENCE_GRADE,
        "family": "hi_nerv",
        "source_fidelity_status": "local_hi_nerv_fork_not_official_hinerv_parity",
        "output_dir": output_dir.as_posix(),
        "storage_preflight": storage_payload,
        "modelsize_row": args.modelsize_row,
        "modelsize_candidate_consumption": modelsize_candidate_consumption,
        "config": _config_snapshot(cfg),
        "num_parameters": int(model.num_parameters()),
        "forward_convention": "call_b2chw_255",
        "forward_smoke": {
            "input_indices": [int(v) for v in idx.tolist()],
            "output_shape": [int(v) for v in output.shape],
            "target_pair_count_for_bias_init": int(idx.shape[0]),
            **forward_smoke_stats,
        },
        "output_head_target_bias_init": _metadata_safe(output_head_target_bias_init),
        "decoder_codec": effective_decoder_codec,
        "decoder_fake_quant_forward": decoder_fake_quant_forward,
        "decoder_weight_waterfill_plan": _decoder_weight_waterfill_plan_attachment_metadata(
            args=args,
            plan=decoder_weight_waterfill_plan,
            fake_quant_forward=decoder_fake_quant_forward,
        ),
        "train_time_controls": train_time_controls.metadata(),
        "prioritized_pair_training": _prioritized_pair_training_metadata(prioritized_pair_indices),
        "archive_path": archive_path,
        "archive_sha256": archive_sha256,
        "archive_bytes": archive_bytes,
        "byte_cap_control": byte_cap_control,
        "post_export_receiver_cache_quality": (
            _receiver_cache_quality_manifest_summary(post_export_quality) if post_export_quality is not None else None
        ),
        "direct_trainer_canonicalization": canonicalization,
        "blockers": [
            "contest_cpu_cuda_exact_eval_not_executed",
            "hi_nerv_smoke_no_training_score",
            "official_hinerv_feature_grid_parity_not_proven",
            *canonicalization["blockers"],
            *byte_cap_control["blockers"],
        ],
        **FALSE_AUTHORITY,
    }
    write_json(output_dir / "smoke_manifest.json", manifest)
    print(
        json.dumps(
            {"smoke_manifest": (output_dir / "smoke_manifest.json").as_posix(), **FALSE_AUTHORITY}, sort_keys=True
        )
    )
    return 0


def _smoke_forward_statistics(
    *,
    output: Any,
    target_rgb_0: Any,
    target_rgb_1: Any,
) -> dict[str, Any]:
    """Build smoke diagnostics proving decoded-target output-head initialization."""

    import numpy as np

    out = np.asarray(output, dtype=np.float32)
    target0 = np.asarray(target_rgb_0, dtype=np.float32) * 255.0
    target1 = np.asarray(target_rgb_1, dtype=np.float32) * 255.0
    if out.ndim != 5 or int(out.shape[1]) != 2 or int(out.shape[2]) != 3:
        raise ValueError(
            "HiNeRV smoke output must be Bx2x3xHxW in 0..255 space; got "
            f"shape={tuple(int(v) for v in out.shape)}"
        )
    if (
        target0.ndim != 4
        or target1.ndim != 4
        or int(target0.shape[-1]) != 3
        or int(target1.shape[-1]) != 3
    ):
        raise ValueError(
            "HiNeRV smoke targets must be NHWC with 3 channels; got "
            f"target0={tuple(int(v) for v in target0.shape)} "
            f"target1={tuple(int(v) for v in target1.shape)}"
        )
    if int(out.shape[0]) != int(target0.shape[0]) or int(out.shape[0]) != int(target1.shape[0]):
        raise ValueError(
            "HiNeRV smoke output/target batch mismatch: "
            f"output={int(out.shape[0])} target0={int(target0.shape[0])} "
            f"target1={int(target1.shape[0])}"
        )
    output_hw = tuple(int(v) for v in out.shape[-2:])
    target0_hw = tuple(int(v) for v in target0.shape[1:3])
    target1_hw = tuple(int(v) for v in target1.shape[1:3])
    if output_hw != target0_hw or output_hw != target1_hw:
        raise ValueError(
            "HiNeRV smoke output/target geometry mismatch: "
            f"output_hw={output_hw} target0_hw={target0_hw} target1_hw={target1_hw}"
        )

    output_nhwc = np.transpose(out, (0, 1, 3, 4, 2))
    targets = np.stack([target0, target1], axis=1)
    output_channel_mean = output_nhwc.mean(axis=(0, 2, 3))
    output_channel_std = output_nhwc.std(axis=(0, 2, 3))
    target_channel_mean = targets.mean(axis=(0, 2, 3))
    target_channel_std = targets.std(axis=(0, 2, 3))
    channel_mean_abs_error = np.abs(output_channel_mean - target_channel_mean)
    neutral_gray_channel_abs_error = np.abs(127.5 - target_channel_mean)

    return {
        "output_min": float(out.min()),
        "output_max": float(out.max()),
        "output_mean": float(out.mean()),
        "output_std": float(out.std()),
        "target_mean_255": float(targets.mean()),
        "target_std_255": float(targets.std()),
        "target_mean_abs_error_after_bias_init": float(
            abs(float(out.mean()) - float(targets.mean()))
        ),
        "target_channel_means_255": [
            [float(v) for v in frame_values]
            for frame_values in target_channel_mean.tolist()
        ],
        "target_channel_stds_255": [
            [float(v) for v in frame_values]
            for frame_values in target_channel_std.tolist()
        ],
        "output_channel_means_255": [
            [float(v) for v in frame_values]
            for frame_values in output_channel_mean.tolist()
        ],
        "output_channel_stds_255": [
            [float(v) for v in frame_values]
            for frame_values in output_channel_std.tolist()
        ],
        "target_channel_mean_abs_error_after_bias_init_255": [
            [float(v) for v in frame_values]
            for frame_values in channel_mean_abs_error.tolist()
        ],
        "neutral_gray_channel_abs_error_255": [
            [float(v) for v in frame_values]
            for frame_values in neutral_gray_channel_abs_error.tolist()
        ],
        "neutral_gray_global_abs_error_255": float(abs(127.5 - float(targets.mean()))),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--full", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--allow-local-output-dir", action="store_true")
    parser.add_argument("--storage-workload-subdir", default=DEFAULT_WORKLOAD_SUBDIR)
    parser.add_argument("--storage-expected-bytes", type=int, default=8 * 1024**3)
    parser.add_argument("--storage-reserve-free-gb", type=float, default=DEFAULT_RESERVE_FREE_GB)
    parser.add_argument("--num-pairs", type=int, default=600)
    parser.add_argument("--modelsize-row", choices=MODEL_SIZE_ROWS, default="hi_nerv_local_tiny")
    parser.add_argument("--latent-dim-coarse", type=int, default=None)
    parser.add_argument("--latent-dim-mid", type=int, default=None)
    parser.add_argument("--latent-dim-fine", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--decoder-channels", default=None)
    parser.add_argument("--output-height", type=int, default=384)
    parser.add_argument("--output-width", type=int, default=512)
    parser.add_argument("--sin-frequency", type=float, default=None)
    parser.add_argument("--video-path", type=Path, default=Path("upstream/videos/0.mkv"))
    parser.add_argument("--upstream-dir", type=Path, default=Path("upstream"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-pairs", type=int, default=8)
    parser.add_argument(
        "--prioritized-pair-indices",
        default="",
        help=(
            "Comma-separated hard-pair/sensitivity pair indices to emphasize "
            "inside MLX training batches. This is false-authority local "
            "sampling only; promotion still requires receiver proof and "
            "contest CPU/CUDA replay."
        ),
    )
    parser.add_argument(
        "--prioritized-pair-indices-file",
        type=Path,
        default=None,
        help=(
            "JSON/list/text artifact containing prioritized_pair_indices, "
            "hard_pair_indices, pair_indices, or sample-generalization "
            "hard-pair coverage."
        ),
    )
    parser.add_argument("--full-lr", type=float, default=1.0e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint-interval-epochs", type=int, default=25)
    parser.add_argument("--decoder-codec", default=DEFAULT_DECODER_CODEC)
    parser.add_argument(
        "--hard-byte-ceiling",
        type=int,
        default=None,
        help=(
            "Hard archive.zip byte ceiling for HiNeRV export. This is a real "
            "export gate, not a nominal modelsize hint; archive export raises "
            "if measured bytes exceed the ceiling. When a modelsize candidate "
            "is attached, the explicit value must match the candidate ceiling."
        ),
    )
    parser.add_argument(
        "--modelsize-candidate-json",
        type=Path,
        default=None,
        help=(
            "HiNeRV hinerv_modelsize_candidate.v1, compact selection, or "
            "compact startup marker whose receiver-visible capacity, decoder "
            "codec, and hard-byte ceiling should drive this trainer/export. "
            "Invalid, over-ceiling, or non-HiNeRV candidates fail before MLX "
            "work starts."
        ),
    )
    parser.add_argument("--decoder-fake-quant-forward", action="store_true")
    parser.add_argument("--decoder-fake-quant-bits", type=int, default=8)
    parser.add_argument(
        "--decoder-weight-waterfill-plan-json",
        type=Path,
        default=None,
        help=(
            "Shared nerv_decoder_weight_waterfill.v1 plan to bind into "
            "HiNeRV train-time named fake quantization and export-side "
            "decoder waterfill preparation."
        ),
    )
    parser.add_argument("--train-time-decoder-pruning-ratio", type=float, default=0.0)
    parser.add_argument("--train-time-decoder-quant-noise-bits", type=int, default=None)
    parser.add_argument("--train-time-decoder-quant-noise-scale", type=float, default=0.0)
    parser.add_argument("--train-time-decoder-quant-noise-seed", type=int, default=0)
    parser.add_argument("--train-time-decoder-control-start-epoch", type=int, default=0)
    parser.add_argument("--train-time-decoder-control-frequency-epochs", type=int, default=1)
    parser.add_argument("--export-decoder-pruning-ratio", type=float, default=0.0)
    parser.add_argument("--export-decoder-quant-noise-bits", type=int, default=None)
    parser.add_argument("--export-decoder-quant-noise-scale", type=float, default=0.0)
    parser.add_argument("--export-decoder-quant-noise-seed", type=int, default=0)
    parser.add_argument("--coder-qat", action="store_true")
    parser.add_argument("--coder-qat-bits", type=int, default=8)
    parser.add_argument("--coder-qat-quant-residual-weight", type=float, default=1.0e-4)
    parser.add_argument("--coder-qat-magnitude-weight", type=float, default=0.0)
    parser.add_argument("--coder-qat-delta-weight", type=float, default=0.0)
    parser.add_argument("--coder-qat-c1a-entropy-weight", type=float, default=0.0)
    parser.add_argument("--coder-qat-c1a-sigma", type=float, default=0.2)
    parser.add_argument("--coder-qat-c1a-sample-size", type=int, default=512)
    parser.add_argument("--disable-train-time-dual-ascent", action="store_true")
    parser.add_argument(
        "--disable-train-time-section-byte-metrics",
        action="store_true",
    )
    parser.add_argument(
        "--train-time-section-byte-metrics-frequency-steps",
        type=int,
        default=25,
    )
    parser.add_argument("--distillation-weight", type=float, default=0.0)
    parser.add_argument("--pose-distillation-weight", type=float, default=1.0)
    parser.add_argument(
        "--segnet-student-live-calibration-weight",
        type=float,
        default=1.0,
        help=(
            "Sibling-head calibration pressure against live real "
            "SegNet(candidate) logits. Direct HiNeRV full runs default it on "
            "when a real SegNet teacher is bound; 0 is a production-control "
            "blocker."
        ),
    )
    parser.add_argument(
        "--segnet-distillation-objective",
        choices=(
            "kl_t2",
            "argmax_hinge",
            "boundary_tckd",
            "boundary_kl_t2",
            "boundary_argmax_hinge",
        ),
        default="kl_t2",
    )
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--segnet-direct-live-distillation-weight", type=float, default=0.0)
    parser.add_argument(
        "--segnet-direct-live-base-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Relative multiplier on RendererBundle's base direct-live SegNet "
            "logit/argmax objective. Keep at 1.0 for normal fit; set lower only "
            "for explicit class-escape research where class-balanced terms own "
            "the early pressure."
        ),
    )
    parser.add_argument("--segnet-direct-live-class-histogram-weight", type=float, default=0.0)
    parser.add_argument("--segnet-direct-live-class-balanced-hinge-weight", type=float, default=0.0)
    parser.add_argument("--segnet-direct-live-class-balanced-ce-weight", type=float, default=0.0)
    parser.add_argument(
        "--segnet-direct-live-class-balanced-squared-hinge-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--segnet-tau-boundary", type=float, default=1.0)
    parser.add_argument("--segnet-hinge-margin", type=float, default=1.0)
    parser.add_argument("--allow-mock-scorer-teacher", action="store_true")
    parser.add_argument("--allow-segnet-only-research", action="store_true")
    parser.add_argument("--eval-roundtrip-ste", action="store_true")
    parser.add_argument("--pose-student-input-preprocess", choices=("rgb", "pr95_yuv6"), default="pr95_yuv6")
    parser.add_argument("--pr95-faithful-curriculum", action="store_true")
    parser.add_argument("--pr95-curriculum-total-epochs", type=int, default=None)
    parser.add_argument(
        "--pr95-stage-source-weight-amplification",
        action="store_true",
        help=(
            "Forward PR95's source-scale SegNet:PoseNet stage weighting into "
            "the shared MLX adapter. This is a real training-loss actuator, not "
            "a metadata tag."
        ),
    )
    parser.add_argument("--staged-scorer-curriculum", action="store_true")
    parser.add_argument("--staged-scorer-recon-fraction", type=float, default=0.75)
    parser.add_argument("--staged-scorer-segnet-fraction", type=float, default=0.15)
    parser.add_argument(
        "--staged-scorer-final-recon-weight",
        type=float,
        default=0.25,
    )
    parser.add_argument("--staged-scorer-segnet-lr-scale", type=float, default=0.3)
    parser.add_argument("--staged-scorer-final-lr-scale", type=float, default=0.1)
    parser.add_argument(
        "--no-output-head-target-bias-init",
        action="store_false",
        dest="output_head_target_bias_init",
        help=(
            "Disable the deterministic compression-time sigmoid-head bias "
            "initialization from target RGB channel means. Production HiNeRV "
            "control contracts treat this as a blocker because zero bias "
            "starts at neutral gray."
        ),
    )
    parser.set_defaults(output_head_target_bias_init=True)
    parser.add_argument(
        "--output-head-target-bias-init-epsilon",
        type=float,
        default=1.0 / 1024.0,
        help=(
            "Clamp epsilon for bias=logit(mean(target_channel)) in the "
            "archive-charged HiNeRV output-head initialization."
        ),
    )
    parser.add_argument(
        "--no-output-head-target-contrast-init",
        action="store_false",
        dest="output_head_target_contrast_init",
        help=(
            "Disable archive-charged sigmoid-head contrast initialization. "
            "Production HiNeRV contracts treat this as a blocker because "
            "recent smokes class-collapsed from low scorer-RGB variance."
        ),
    )
    parser.set_defaults(output_head_target_contrast_init=True)
    parser.add_argument(
        "--output-head-target-contrast-init-max-pairs",
        type=int,
        default=8,
        help=(
            "Maximum target pairs used to estimate the compression-time "
            "output-head contrast scale."
        ),
    )
    parser.add_argument(
        "--output-head-target-contrast-init-min-output-std",
        type=float,
        default=1.0e-6,
        help=(
            "Minimum unit-space output std denominator for output-head "
            "contrast scaling. The default is intentionally low enough to "
            "escape the one-class SegNet collapse seen in flat HiNeRV smokes."
        ),
    )
    parser.add_argument(
        "--output-head-target-contrast-init-max-gain",
        type=float,
        default=4096.0,
        help=(
            "Maximum per-channel output-head weight gain applied by the "
            "closed-form contrast initializer."
        ),
    )
    parser.add_argument(
        "--scorer-input-distribution-guard-weight",
        type=float,
        default=0.0,
        help=(
            "Differentiable train-time guard that matches decoded RGB and "
            "upstream-evaluate scorer inputs: SegNet frame-1 direct fit and "
            "PR95/YUV6 pair mean, std, dynamic range, direct fit, temporal "
            "delta, temporal-delta fit, and soft saturation mass to the contest "
            "video targets before SegNet/PoseNet scorer surrogates consume the "
            "frames."
        ),
    )
    parser.add_argument(
        "--scorer-input-distribution-guard-saturation-margin",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--scorer-input-distribution-guard-temperature",
        type=float,
        default=0.01,
    )
    parser.add_argument("--scorer-input-contrast-floor-weight", type=float, default=0.0)
    parser.add_argument(
        "--scorer-input-contrast-floor-segnet-min-std-ratio",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--scorer-input-contrast-floor-posenet-yuv6-min-std-ratio",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--scorer-input-shape-tether-weight",
        type=float,
        default=0.0,
        help=(
            "Dense train-time scorer-input shape tether on exact evaluate.py "
            "domains: SegNet last-frame RGB plus PoseNet two-frame YUV6 and "
            "YUV6 temporal delta, centered and normalized by reference channel "
            "variance. False-authority MLX pressure only."
        ),
    )
    parser.add_argument(
        "--posenet-temporal-signal-floor-weight",
        type=float,
        default=0.0,
        help=(
            "PoseNet-specific one-sided train-time floor on the exact PR95 "
            "YUV6 temporal delta frame_1-frame_0. This attacks compact "
            "renderer pair-collapse where archive bytes look healthy but "
            "PoseNet sees nearly identical frames."
        ),
    )
    parser.add_argument(
        "--posenet-temporal-signal-min-std-ratio",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--posenet-temporal-signal-min-mean-abs-ratio",
        type=float,
        default=0.25,
    )
    parser.add_argument("--grad-clip-max-norm", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--optimizer-warmup-steps-per-epoch", type=int, default=1)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument(
        "--optimizer-kind",
        choices=SUPPORTED_MLX_SCORE_AWARE_OPTIMIZER_KINDS,
        default=DEFAULT_MLX_SCORE_AWARE_OPTIMIZER_KIND,
    )
    parser.add_argument("--cosine-decay", action="store_true")
    parser.add_argument("--cosine-decay-total-epochs", type=int, default=None)
    parser.add_argument("--cosine-decay-min-lr-ratio", type=float, default=1.0e-2)
    parser.add_argument("--pair-sampling-default-weight", type=float, default=1.0)
    parser.add_argument(
        "--pair-sampling-weight",
        action="append",
        default=[],
        metavar="PAIR=WEIGHT",
        help=(
            "Source-pair sampling mass consumed by the shared adapter. Repeat "
            "for hard-pair/XRay weights; this changes local training batches "
            "only and carries no score authority."
        ),
    )
    parser.add_argument(
        "--gradient-multiplier",
        action="append",
        default=[],
        metavar="PARAM=VALUE",
        help=(
            "Exact named-parameter gradient multiplier consumed inside the "
            "shared MLX adapter before clipping/update. Repeat for decoder "
            "waterfill/ablation experiments."
        ),
    )
    parser.add_argument("--bias-gradient-multiplier", type=float, default=None)
    parser.add_argument(
        "--output-head-bias-gradient-multiplier",
        type=float,
        default=1.0,
    )
    parser.add_argument("--ema-archive-selection", action="store_true")
    parser.add_argument("--smoke-export-archive", action="store_true")
    parser.add_argument("--post-export-receiver-cache-quality-gate", action="store_true")
    parser.add_argument(
        "--receiver-cache-quality-reference-cache-dir",
        type=Path,
        default=Path("experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600"),
    )
    parser.add_argument("--receiver-cache-quality-max-pairs", type=int, default=1)
    parser.add_argument("--receiver-cache-quality-batch-pairs", type=int, default=1)
    parser.add_argument("--receiver-cache-quality-min-segnet-std", type=float, default=1.0)
    parser.add_argument(
        "--receiver-cache-quality-min-segnet-dynamic-range",
        type=float,
        default=16.0,
    )
    parser.add_argument(
        "--receiver-cache-quality-max-segnet-mae-vs-reference-for-fit-gate",
        type=float,
        default=64.0,
    )
    parser.add_argument("--receiver-cache-quality-min-posenet-yuv6-std", type=float, default=1.0)
    parser.add_argument(
        "--receiver-cache-quality-min-posenet-yuv6-dynamic-range",
        type=float,
        default=16.0,
    )
    parser.add_argument(
        "--receiver-cache-quality-max-posenet-yuv6-mae-vs-reference-for-fit-gate",
        type=float,
        default=64.0,
    )
    parser.add_argument(
        "--receiver-cache-quality-min-posenet-yuv6-temporal-signal-std-for-fit-gate",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--receiver-cache-quality-min-posenet-yuv6-temporal-signal-mean-abs-for-fit-gate",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--receiver-cache-quality-min-segnet-argmax-occupied-class-fraction-for-fit-gate",
        type=float,
        default=0.400001,
    )
    parser.add_argument(
        "--skip-receiver-cache-quality-mlx-scorer-response-probe",
        dest="receiver_cache_quality_mlx_scorer_response_probe",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--receiver-cache-quality-mlx-scorer-response-device-type",
        choices=("cpu", "gpu", "metal", "mps"),
        default="cpu",
    )
    parser.add_argument(
        "--receiver-cache-quality-mlx-scorer-response-batch-pairs",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--receiver-cache-quality-max-mlx-scorer-response-posenet-dist-for-fit-gate",
        type=float,
        default=DEFAULT_MAX_MLX_SCORER_RESPONSE_POSENET_DIST_FOR_FIT_GATE,
    )
    parser.add_argument(
        "--receiver-cache-quality-max-mlx-scorer-response-segnet-dist-for-fit-gate",
        type=float,
        default=DEFAULT_MAX_MLX_SCORER_RESPONSE_SEGNET_DIST_FOR_FIT_GATE,
    )
    return parser


def _config_from_args(
    args: argparse.Namespace,
    *,
    modelsize_candidate: Mapping[str, Any] | None = None,
) -> Any:
    candidate = (
        _modelsize_candidate_from_args(args)
        if modelsize_candidate is None
        else dict(modelsize_candidate)
    )
    if candidate:
        _reject_modelsize_candidate_cli_config_overrides(args)
        cfg = build_hinerv_config_from_modelsize_candidate(candidate)
        if int(args.output_height) != int(cfg.output_height) or int(
            args.output_width
        ) != int(cfg.output_width):
            raise ValueError(
                "HiNeRV modelsize candidate fixes receiver output geometry; "
                f"got --output-height/--output-width "
                f"{int(args.output_height)}x{int(args.output_width)} but "
                f"candidate resolves to {int(cfg.output_height)}x{int(cfg.output_width)}"
            )
        return cfg
    rows = {str(row["row_id"]): row["config"] for row in hi_nerv_modelsize_config_rows(num_pairs=int(args.num_pairs))}
    cfg = rows[str(args.modelsize_row)]
    updates: dict[str, Any] = {
        "num_pairs": int(args.num_pairs),
        "output_height": int(args.output_height),
        "output_width": int(args.output_width),
    }
    for attr in (
        "latent_dim_coarse",
        "latent_dim_mid",
        "latent_dim_fine",
        "embed_dim",
        "sin_frequency",
    ):
        value = getattr(args, attr, None)
        if value is not None:
            updates[attr] = value
    if args.decoder_channels:
        updates["decoder_channels"] = tuple(int(part) for part in str(args.decoder_channels).split(",") if part)
    return replace(cfg, **updates)


def _resolve_modelsize_candidate_path(args: argparse.Namespace) -> Path | None:
    path = getattr(args, "modelsize_candidate_json", None)
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve(strict=False)


def _modelsize_candidate_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    path = _resolve_modelsize_candidate_path(args)
    if path is None:
        return None
    if not path.is_file():
        raise ValueError(
            "modelsize_candidate_json must point at an existing file; "
            f"got {path.as_posix()}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("modelsize_candidate_json must contain a JSON object")
    candidate = _extract_hinerv_modelsize_candidate(payload)
    _validate_hinerv_modelsize_candidate(candidate, args=args)
    return candidate


def _extract_hinerv_modelsize_candidate(payload: Mapping[str, Any]) -> dict[str, Any]:
    schema = str(payload.get("schema") or "")
    if schema == "hinerv_modelsize_candidate.v1":
        return dict(payload)
    if schema == "compact_execute_modelsize_candidate_selection.v1":
        candidate = payload.get("candidate")
    elif schema == "compact_carrier_startup_marker.v1":
        candidate = payload.get("modelsize_candidate")
    else:
        raise ValueError(
            "modelsize_candidate_json schema must be hinerv_modelsize_candidate.v1, "
            "compact_execute_modelsize_candidate_selection.v1, or "
            f"compact_carrier_startup_marker.v1; got {schema!r}"
        )
    if not isinstance(candidate, Mapping):
        raise ValueError(
            f"modelsize_candidate_json schema {schema!r} did not contain a "
            "modelsize candidate object"
        )
    return dict(candidate)


def _validate_hinerv_modelsize_candidate(
    candidate: Mapping[str, Any],
    *,
    args: argparse.Namespace,
) -> None:
    blockers: list[str] = []
    if candidate.get("schema") != "hinerv_modelsize_candidate.v1":
        blockers.append("hinerv_modelsize_candidate_schema_mismatch")
    if candidate.get("family") != "hi_nerv":
        blockers.append("hinerv_modelsize_candidate_family_mismatch")
    if not str(candidate.get("candidate_id") or "").strip():
        blockers.append("hinerv_modelsize_candidate_id_missing")
    required_int_fields = (
        "num_pairs",
        "hard_byte_ceiling",
        "latent_dim",
        "embed_dim",
        "decoder_channel",
        "local_grid_levels",
        "local_grid_channels",
        "convnext_mlp_ratio",
        "convnext_kernel_size",
        "mid_injection_block_index",
        "fine_injection_block_index",
        "nominal_total_payload_bytes",
    )
    for field in required_int_fields:
        try:
            value = int(candidate.get(field))
        except (TypeError, ValueError):
            blockers.append(f"hinerv_modelsize_candidate_{field}_missing_or_invalid")
            continue
        if value <= 0 and field not in {
            "mid_injection_block_index",
            "fine_injection_block_index",
        }:
            blockers.append(f"hinerv_modelsize_candidate_{field}_must_be_positive")
        if (
            field in {"mid_injection_block_index", "fine_injection_block_index"}
            and value < 0
        ):
            blockers.append(f"hinerv_modelsize_candidate_{field}_must_be_nonnegative")
    if str(candidate.get("decoder_codec") or "").strip() == "":
        blockers.append("hinerv_modelsize_candidate_decoder_codec_missing")
    try:
        if int(candidate.get("num_pairs")) != int(args.num_pairs):
            blockers.append("hinerv_modelsize_candidate_num_pairs_mismatch")
    except (TypeError, ValueError):
        pass
    controller = candidate.get("byte_cap_controller")
    controller_under = None
    if isinstance(controller, Mapping) and (
        controller.get("predicted_under_hard_byte_ceiling") is False
    ):
        controller_under = False
        blockers.append(
            "hinerv_modelsize_candidate_byte_cap_controller_predicts_over_hard_ceiling"
        )
    elif isinstance(controller, Mapping):
        controller_under = controller.get("predicted_under_hard_byte_ceiling")
    if controller_under is not True and candidate.get("nominal_under_ceiling") is not True:
        blockers.append("hinerv_modelsize_candidate_nominally_over_hard_byte_ceiling")
    contract = candidate.get("modelsize_control_contract")
    if not isinstance(contract, Mapping):
        blockers.append("hinerv_modelsize_candidate_contract_missing")
    else:
        for field in MODELSIZE_CONTROL_CONTRACT_REQUIRED_TRUE_FIELDS:
            if contract.get(field) is not True:
                blockers.append(f"hinerv_modelsize_candidate_contract_missing:{field}")
    if blockers:
        raise ValueError(
            "invalid HiNeRV modelsize candidate: " + ", ".join(dict.fromkeys(blockers))
        )


def _reject_modelsize_candidate_cli_config_overrides(args: argparse.Namespace) -> None:
    overrides = [
        flag
        for flag, attr in (
            ("--latent-dim-coarse", "latent_dim_coarse"),
            ("--latent-dim-mid", "latent_dim_mid"),
            ("--latent-dim-fine", "latent_dim_fine"),
            ("--embed-dim", "embed_dim"),
            ("--decoder-channels", "decoder_channels"),
            ("--sin-frequency", "sin_frequency"),
        )
        if getattr(args, attr, None) is not None
    ]
    if overrides:
        raise ValueError(
            "HiNeRV modelsize candidate owns receiver-visible architecture; "
            "remove conflicting CLI overrides: " + ", ".join(overrides)
        )


def _decoder_codec_from_args(
    args: argparse.Namespace,
    *,
    modelsize_candidate: Mapping[str, Any] | None = None,
) -> str:
    candidate = (
        _modelsize_candidate_from_args(args)
        if modelsize_candidate is None
        else dict(modelsize_candidate)
    )
    requested = str(
        getattr(args, "decoder_codec", DEFAULT_DECODER_CODEC) or DEFAULT_DECODER_CODEC
    )
    if not candidate:
        return requested
    candidate_codec = str(candidate.get("decoder_codec") or "").strip()
    if not candidate_codec:
        raise ValueError("HiNeRV modelsize candidate decoder_codec is missing")
    if requested != DEFAULT_DECODER_CODEC and requested != candidate_codec:
        raise ValueError(
            "HiNeRV modelsize candidate decoder_codec conflicts with explicit "
            f"--decoder-codec: candidate={candidate_codec!r} cli={requested!r}"
        )
    return candidate_codec


def _hard_byte_ceiling_from_modelsize_candidate(
    candidate: Mapping[str, Any] | None,
) -> int | None:
    if not candidate:
        return None
    ceiling = int(candidate.get("hard_byte_ceiling") or 0)
    if ceiling <= 0:
        raise ValueError("HiNeRV modelsize candidate hard_byte_ceiling must be positive")
    return ceiling


def _hard_byte_ceiling_from_args(
    args: argparse.Namespace,
    *,
    modelsize_candidate: Mapping[str, Any] | None = None,
) -> int | None:
    """Resolve the real archive export ceiling from candidate or CLI controls."""

    candidate_ceiling = _hard_byte_ceiling_from_modelsize_candidate(
        modelsize_candidate
    )
    requested = getattr(args, "hard_byte_ceiling", None)
    if requested is None:
        return candidate_ceiling
    requested_ceiling = int(requested)
    if requested_ceiling <= 0:
        raise ValueError("--hard-byte-ceiling must be positive")
    if candidate_ceiling is not None and requested_ceiling != int(candidate_ceiling):
        raise ValueError(
            "HiNeRV --hard-byte-ceiling conflicts with modelsize candidate "
            f"hard_byte_ceiling: cli={requested_ceiling} "
            f"candidate={int(candidate_ceiling)}"
        )
    return requested_ceiling


def _build_hinerv_hard_byte_ceiling_control(
    *,
    candidate: Mapping[str, Any] | None,
    hard_byte_ceiling: int | None,
    archive_path: str | None,
    archive_sha256: str | None,
    archive_bytes: int | None,
    archive_export_requested: bool,
) -> dict[str, Any]:
    candidate_dict = dict(candidate or {})
    controller = candidate_dict.get("byte_cap_controller")
    controller_payload = _metadata_safe(controller) if isinstance(controller, Mapping) else None
    blockers: list[str] = []
    if hard_byte_ceiling is None:
        blockers.append("hinerv_hard_byte_ceiling_not_attached")
        under = None
        delta = None
    elif not archive_export_requested:
        blockers.append("hinerv_hard_byte_ceiling_not_enforced_archive_export_disabled")
        under = None
        delta = None
    elif archive_bytes is None:
        blockers.append("hinerv_hard_byte_ceiling_archive_bytes_missing")
        under = None
        delta = None
    else:
        delta = int(archive_bytes) - int(hard_byte_ceiling)
        under = delta <= 0
        if not under:
            blockers.append("hinerv_archive_exceeds_hard_byte_ceiling")
    return {
        "schema": HI_NERV_HARD_BYTE_CEILING_CONTROL_SCHEMA,
        "family": "hi_nerv",
        "candidate_id": candidate_dict.get("candidate_id"),
        "attached": hard_byte_ceiling is not None,
        "hard_byte_ceiling": hard_byte_ceiling,
        "archive_export_requested": bool(archive_export_requested),
        "enforced": bool(hard_byte_ceiling is not None and archive_export_requested),
        "archive_path": archive_path,
        "archive_sha256": archive_sha256,
        "archive_bytes": None if archive_bytes is None else int(archive_bytes),
        "under_hard_byte_ceiling": under,
        "delta_bytes_vs_hard_byte_ceiling": delta,
        "byte_cap_controller": controller_payload,
        "blockers": blockers,
        "authority": TRAINER_AUTHORITY,
        **FALSE_AUTHORITY,
    }


def _modelsize_candidate_consumption_metadata(
    *,
    args: argparse.Namespace,
    candidate: Mapping[str, Any] | None,
) -> dict[str, Any]:
    path = _resolve_modelsize_candidate_path(args)
    if not candidate:
        return {
            "schema": HI_NERV_MODELSIZE_CANDIDATE_CONSUMPTION_SCHEMA,
            "attached": False,
            "path": path.as_posix() if path is not None else None,
            "consumed_by_trainer_config": False,
            "consumed_by_decoder_codec": False,
            "consumed_by_archive_export_hard_byte_ceiling": False,
            "blockers": ["hinerv_modelsize_candidate_json_not_attached"],
            "authority": TRAINER_AUTHORITY,
            **FALSE_AUTHORITY,
        }
    candidate_dict = dict(candidate)
    contract = dict(candidate_dict.get("modelsize_control_contract") or {})
    contract.setdefault(
        "control_precedence",
        modelsize_control_precedence_contract(candidate_dict),
    )
    return {
        "schema": HI_NERV_MODELSIZE_CANDIDATE_CONSUMPTION_SCHEMA,
        "attached": True,
        "path": path.as_posix() if path is not None else None,
        "sha256": sha256_file(path) if path is not None and path.is_file() else None,
        "bytes": path.stat().st_size if path is not None and path.is_file() else None,
        "candidate_id": candidate_dict.get("candidate_id"),
        "candidate_schema": candidate_dict.get("schema"),
        "capacity_source": candidate_dict.get("capacity_source"),
        "target_modelsize_mparams": candidate_dict.get("target_modelsize_mparams"),
        "modelsize_mparams": candidate_dict.get("modelsize_mparams"),
        "hard_byte_ceiling": _hard_byte_ceiling_from_modelsize_candidate(candidate_dict),
        "nominal_total_payload_bytes": candidate_dict.get("nominal_total_payload_bytes"),
        "nominal_under_ceiling": bool(candidate_dict.get("nominal_under_ceiling")),
        "byte_headroom": candidate_dict.get("byte_headroom"),
        "decoder_codec": _decoder_codec_from_args(
            args,
            modelsize_candidate=candidate_dict,
        ),
        "modelsize_control_contract": contract,
        "byte_cap_controller": _metadata_safe(candidate_dict.get("byte_cap_controller")),
        "consumed_by_trainer_config": True,
        "consumed_by_decoder_codec": True,
        "consumed_by_archive_export_hard_byte_ceiling": True,
        "blockers": ["contest_cpu_cuda_exact_eval_not_executed"],
        "authority": TRAINER_AUTHORITY,
        **FALSE_AUTHORITY,
    }


def _coder_qat_config_from_args(args: argparse.Namespace) -> Any:
    from tac.substrates._shared.mlx_score_aware.coder_qat import CoderAwareQATConfig

    return CoderAwareQATConfig(
        enabled=bool(args.coder_qat),
        quant_bits=int(args.coder_qat_bits),
        quant_residual_weight=float(args.coder_qat_quant_residual_weight),
        magnitude_weight=float(args.coder_qat_magnitude_weight),
        delta_weight=float(args.coder_qat_delta_weight),
        c1a_entropy_weight=float(getattr(args, "coder_qat_c1a_entropy_weight", 0.0)),
        c1a_sigma=float(getattr(args, "coder_qat_c1a_sigma", 0.2)),
        c1a_sample_size=int(getattr(args, "coder_qat_c1a_sample_size", 512)),
    ).validated()


def _build_hi_nerv_train_time_section_byte_control_from_model(
    *,
    model: Any,
    decoder_codec: str,
    controls: HiNervTrainTimeControlConfig,
    decoder_weight_waterfill_plan: Mapping[str, Any] | None,
    hard_byte_ceiling: int | None,
    active_loss_weights: Mapping[str, float],
) -> dict[str, Any]:
    """Measure live HIV1 sections and turn them into train-time byte budgets."""

    from tac.substrates.hi_nerv.archive import build_archive_section_telemetry
    from tac.substrates.hi_nerv.archive_candidate import (
        pack_archive_from_exported_state_dict,
    )
    from tac.substrates.hi_nerv.bitstream import (
        build_hinerv_train_time_section_byte_control,
    )

    bin_bytes, bitstream_report = pack_archive_from_exported_state_dict(
        exported_state_dict=model.export_state_dict(),
        cfg=model.cfg,
        decoder_codec=decoder_codec,
        pruning_ratio=float(controls.export_decoder_pruning_ratio),
        quant_noise_bits=controls.export_decoder_quant_noise_bits,
        quant_noise_scale=float(controls.export_decoder_quant_noise_scale),
        quant_noise_seed=int(controls.export_decoder_quant_noise_seed),
        decoder_weight_waterfill_plan=decoder_weight_waterfill_plan,
        return_bitstream_report=True,
    )
    telemetry = build_archive_section_telemetry(bin_bytes)
    control = build_hinerv_train_time_section_byte_control(
        telemetry,
        active_loss_weights,
        hard_byte_ceiling=hard_byte_ceiling,
    )
    return {
        **control,
        "live_inner_payload_bytes": len(bin_bytes),
        "decoder_codec": decoder_codec,
        "measurement_source": "live_model_export_state_dict_to_hiv1_inner_payload",
        "bitstream_preparation": _metadata_safe(bitstream_report),
        "authority": TRAINER_AUTHORITY,
        **FALSE_AUTHORITY,
    }


def _build_hi_nerv_train_time_section_byte_metrics_callback(
    *,
    args: argparse.Namespace,
    model: Any,
    decoder_codec: str,
    controls: HiNervTrainTimeControlConfig,
    decoder_weight_waterfill_plan: Mapping[str, Any] | None,
    hard_byte_ceiling: int | None,
    active_loss_weights: Mapping[str, float],
) -> tuple[Any | None, dict[str, Any]]:
    """Attach live section-byte telemetry to the shared MLX trainer."""

    enabled = bool(
        not bool(getattr(args, "disable_train_time_section_byte_metrics", False))
        and hard_byte_ceiling is not None
        and bool(getattr(args, "coder_qat", False))
    )
    frequency_steps = int(
        getattr(args, "train_time_section_byte_metrics_frequency_steps", 25)
    )
    if frequency_steps <= 0:
        raise ValueError(
            "train_time_section_byte_metrics_frequency_steps must be positive"
        )
    blockers: list[str] = []
    if hard_byte_ceiling is None:
        blockers.append("hinerv_train_time_section_byte_hard_ceiling_missing")
    if not bool(getattr(args, "coder_qat", False)):
        blockers.append("hinerv_train_time_section_byte_coder_qat_missing")
    if bool(getattr(args, "disable_train_time_section_byte_metrics", False)):
        blockers.append("hinerv_train_time_section_byte_metrics_disabled")
    if not enabled:
        return None, {
            "schema": "hi_nerv_live_train_time_section_byte_metrics_callback.v1",
            "enabled": False,
            "frequency_steps": frequency_steps,
            "initial_control": None,
            "blockers": list(dict.fromkeys(blockers)),
            "authority": TRAINER_AUTHORITY,
            **FALSE_AUTHORITY,
        }

    state: dict[str, Any] = {
        "step": 0,
        "last_control": _build_hi_nerv_train_time_section_byte_control_from_model(
            model=model,
            decoder_codec=decoder_codec,
            controls=controls,
            decoder_weight_waterfill_plan=decoder_weight_waterfill_plan,
            hard_byte_ceiling=hard_byte_ceiling,
            active_loss_weights=active_loss_weights,
        ),
    }

    def _callback(model_obj: Any, _batch: Any, _loss_weights: Mapping[str, float]) -> dict[str, Any]:
        state["step"] = int(state["step"]) + 1
        refreshed = state["step"] == 1 or state["step"] % frequency_steps == 0
        if refreshed:
            state["last_control"] = (
                _build_hi_nerv_train_time_section_byte_control_from_model(
                    model=model_obj,
                    decoder_codec=decoder_codec,
                    controls=controls,
                    decoder_weight_waterfill_plan=decoder_weight_waterfill_plan,
                    hard_byte_ceiling=hard_byte_ceiling,
                    active_loss_weights=active_loss_weights,
                )
            )
        control = dict(state["last_control"])
        metrics = dict(control.get("metrics_payload") or {})
        metrics["metadata"] = {
            "schema": "hi_nerv_live_train_time_section_byte_metrics.v1",
            "measurement_step": int(state["step"]),
            "frequency_steps": frequency_steps,
            "refreshed_this_step": bool(refreshed),
            "section_byte_control_active": bool(control.get("active")),
            "controlled_section_count": int(
                control.get("controlled_section_count") or 0
            ),
            "pending_section_count": int(control.get("pending_section_count") or 0),
            "authority": TRAINER_AUTHORITY,
            **FALSE_AUTHORITY,
        }
        return metrics

    attachment = {
        "schema": "hi_nerv_live_train_time_section_byte_metrics_callback.v1",
        "enabled": True,
        "frequency_steps": frequency_steps,
        "initial_control": _metadata_safe(state["last_control"]),
        "active": bool(state["last_control"].get("active")),
        "blockers": list(dict.fromkeys(
            [str(v) for v in state["last_control"].get("blockers", [])]
        )),
        "authority": TRAINER_AUTHORITY,
        **FALSE_AUTHORITY,
    }
    return _callback, attachment


def _train_time_dual_ascent_config_from_args(
    args: argparse.Namespace,
    *,
    train_time_controls: HiNervTrainTimeControlConfig,
    coder_qat_loss_weight_map: Mapping[str, float],
    section_byte_control: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared train-time dual-ascent config consumed by the harness."""

    from tac.substrates._shared.mlx_score_aware.dual_ascent import (
        build_default_nerv_train_time_dual_ascent_config,
    )

    section_control = dict(section_byte_control or {})
    return build_default_nerv_train_time_dual_ascent_config(
        family="hi_nerv",
        segnet_distillation_weight=float(getattr(args, "distillation_weight", 0.0)),
        segnet_direct_live_distillation_weight=float(
            train_time_controls.segnet_direct_live_distillation_weight
        ),
        segnet_direct_live_class_histogram_weight=float(
            train_time_controls.segnet_direct_live_class_histogram_weight
        ),
        segnet_direct_live_class_balanced_hinge_weight=float(
            train_time_controls.segnet_direct_live_class_balanced_hinge_weight
        ),
        segnet_direct_live_class_balanced_ce_weight=float(
            train_time_controls.segnet_direct_live_class_balanced_ce_weight
        ),
        segnet_direct_live_class_balanced_squared_hinge_weight=float(
            train_time_controls.segnet_direct_live_class_balanced_squared_hinge_weight
        ),
        pose_distillation_weight=float(
            getattr(args, "pose_distillation_weight", 0.0)
        ),
        scorer_input_distribution_guard_weight=float(
            train_time_controls.scorer_input_distribution_guard_weight
        ),
        scorer_input_contrast_floor_weight=float(
            train_time_controls.scorer_input_contrast_floor_weight
        ),
        scorer_input_shape_tether_weight=float(
            train_time_controls.scorer_input_shape_tether_weight
        ),
        posenet_temporal_signal_floor_weight=float(
            train_time_controls.posenet_temporal_signal_floor_weight
        ),
        coder_qat_loss_weight_map=coder_qat_loss_weight_map,
        archive_byte_budget=section_control.get("hard_byte_ceiling"),
        section_byte_budgets=section_control.get("section_byte_budgets"),
        section_byte_loss_weight_key_map=section_control.get(
            "section_byte_loss_weight_key_map"
        ),
        section_byte_loss_weight_scale_map=section_control.get(
            "section_byte_loss_weight_scale_map"
        ),
        enabled=not bool(getattr(args, "disable_train_time_dual_ascent", False)),
    )


def _train_time_control_config_from_args(
    args: argparse.Namespace,
) -> HiNervTrainTimeControlConfig:
    if bool(getattr(args, "pr95_faithful_curriculum", False)):
        stage_loss_schedule = "pr95_faithful_8stage"
    elif bool(getattr(args, "staged_scorer_curriculum", False)):
        stage_loss_schedule = "staged_scorer_curriculum"
    else:
        stage_loss_schedule = "single_stage_score_aware_full"
    return HiNervTrainTimeControlConfig(
        stage_loss_schedule=stage_loss_schedule,
        optimizer_kind=str(getattr(args, "optimizer_kind", "")),
        pr95_faithful_curriculum_enabled=bool(
            getattr(args, "pr95_faithful_curriculum", False)
        ),
        pr95_curriculum_total_epochs=getattr(
            args,
            "pr95_curriculum_total_epochs",
            None,
        ),
        staged_scorer_curriculum_enabled=bool(
            getattr(args, "staged_scorer_curriculum", False)
        ),
        coder_qat_enabled=bool(getattr(args, "coder_qat", False)),
        coder_qat_quant_bits=int(getattr(args, "coder_qat_bits", 8)),
        coder_qat_c1a_entropy_weight=float(
            getattr(args, "coder_qat_c1a_entropy_weight", 0.0)
        ),
        coder_qat_c1a_sigma=float(getattr(args, "coder_qat_c1a_sigma", 0.2)),
        coder_qat_c1a_sample_size=int(
            getattr(args, "coder_qat_c1a_sample_size", 512)
        ),
        segnet_student_live_calibration_weight=float(
            getattr(args, "segnet_student_live_calibration_weight", 1.0)
        ),
        segnet_distillation_objective=str(
            getattr(args, "segnet_distillation_objective", "kl_t2")
        ),
        distillation_temperature=float(getattr(args, "distillation_temperature", 2.0)),
        segnet_direct_live_distillation_weight=float(
            getattr(args, "segnet_direct_live_distillation_weight", 0.0)
        ),
        segnet_direct_live_base_loss_weight=float(
            getattr(args, "segnet_direct_live_base_loss_weight", 1.0)
        ),
        segnet_direct_live_class_histogram_weight=float(
            getattr(args, "segnet_direct_live_class_histogram_weight", 0.0)
        ),
        segnet_direct_live_class_balanced_hinge_weight=float(
            getattr(args, "segnet_direct_live_class_balanced_hinge_weight", 0.0)
        ),
        segnet_direct_live_class_balanced_ce_weight=float(
            getattr(args, "segnet_direct_live_class_balanced_ce_weight", 0.0)
        ),
        segnet_direct_live_class_balanced_squared_hinge_weight=float(
            getattr(
                args,
                "segnet_direct_live_class_balanced_squared_hinge_weight",
                0.0,
            )
        ),
        segnet_tau_boundary=float(getattr(args, "segnet_tau_boundary", 1.0)),
        segnet_hinge_margin=float(getattr(args, "segnet_hinge_margin", 1.0)),
        decoder_fake_quant_forward_enabled=bool(
            getattr(args, "decoder_fake_quant_forward", False)
        ),
        decoder_fake_quant_bits=int(getattr(args, "decoder_fake_quant_bits", 8)),
        train_time_decoder_pruning_ratio=float(
            getattr(args, "train_time_decoder_pruning_ratio", 0.0)
        ),
        train_time_decoder_quant_noise_bits=getattr(
            args,
            "train_time_decoder_quant_noise_bits",
            None,
        ),
        train_time_decoder_quant_noise_scale=float(
            getattr(args, "train_time_decoder_quant_noise_scale", 0.0)
        ),
        train_time_decoder_quant_noise_seed=int(
            getattr(args, "train_time_decoder_quant_noise_seed", 0)
        ),
        train_time_decoder_control_start_epoch=int(
            getattr(args, "train_time_decoder_control_start_epoch", 0)
        ),
        train_time_decoder_control_frequency_epochs=int(
            getattr(args, "train_time_decoder_control_frequency_epochs", 1)
        ),
        export_decoder_pruning_ratio=float(
            getattr(args, "export_decoder_pruning_ratio", 0.0)
        ),
        export_decoder_quant_noise_bits=getattr(
            args,
            "export_decoder_quant_noise_bits",
            None,
        ),
        export_decoder_quant_noise_scale=float(
            getattr(args, "export_decoder_quant_noise_scale", 0.0)
        ),
        export_decoder_quant_noise_seed=int(
            getattr(args, "export_decoder_quant_noise_seed", 0)
        ),
        scorer_input_distribution_guard_weight=float(
            getattr(args, "scorer_input_distribution_guard_weight", 0.0)
        ),
        scorer_input_distribution_guard_saturation_margin=float(
            getattr(
                args,
                "scorer_input_distribution_guard_saturation_margin",
                0.02,
            )
        ),
        scorer_input_distribution_guard_temperature=float(
            getattr(
                args,
                "scorer_input_distribution_guard_temperature",
                0.01,
            )
        ),
        scorer_input_contrast_floor_weight=float(
            getattr(args, "scorer_input_contrast_floor_weight", 0.0)
        ),
        scorer_input_contrast_floor_segnet_min_std_ratio=float(
            getattr(args, "scorer_input_contrast_floor_segnet_min_std_ratio", 0.5)
        ),
        scorer_input_contrast_floor_posenet_yuv6_min_std_ratio=float(
            getattr(
                args,
                "scorer_input_contrast_floor_posenet_yuv6_min_std_ratio",
                0.5,
            )
        ),
        scorer_input_shape_tether_weight=float(
            getattr(args, "scorer_input_shape_tether_weight", 0.0)
        ),
        posenet_temporal_signal_floor_weight=float(
            getattr(args, "posenet_temporal_signal_floor_weight", 0.0)
        ),
        posenet_temporal_signal_min_std_ratio=float(
            getattr(args, "posenet_temporal_signal_min_std_ratio", 0.25)
        ),
        posenet_temporal_signal_min_mean_abs_ratio=float(
            getattr(args, "posenet_temporal_signal_min_mean_abs_ratio", 0.25)
        ),
        output_head_target_bias_init_enabled=bool(
            getattr(args, "output_head_target_bias_init", True)
        ),
        output_head_target_bias_init_epsilon=float(
            getattr(args, "output_head_target_bias_init_epsilon", 1.0 / 1024.0)
        ),
        output_head_target_contrast_init_enabled=bool(
            getattr(args, "output_head_target_contrast_init", True)
        ),
        output_head_target_contrast_init_max_pairs=int(
            getattr(args, "output_head_target_contrast_init_max_pairs", 8)
        ),
        output_head_target_contrast_init_min_output_std=float(
            getattr(args, "output_head_target_contrast_init_min_output_std", 1.0e-6)
        ),
        output_head_target_contrast_init_max_gain=float(
            getattr(args, "output_head_target_contrast_init_max_gain", 4096.0)
        ),
    ).validated()


def _require_finite_nonnegative(value: float, field: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{field} must be finite and non-negative")


def _require_finite_positive(value: float, field: str) -> None:
    if not math.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{field} must be finite and positive")


def _validate_ratio(value: float, field: str) -> None:
    if not math.isfinite(float(value)) or float(value) < 0.0 or float(value) >= 1.0:
        raise ValueError(f"{field} must be finite and in [0, 1)")


def _validate_unit_open_interval(value: float, field: str) -> None:
    if not math.isfinite(float(value)) or not 0.0 < float(value) < 0.5:
        raise ValueError(f"{field} must be finite and in (0, 0.5)")


def _validate_quant_noise_controls(
    *,
    bits: int | None,
    scale: float,
    field_prefix: str,
) -> None:
    _require_finite_nonnegative(float(scale), f"{field_prefix}_scale")
    if bits is None:
        if float(scale) > 0.0:
            raise ValueError(f"{field_prefix}_bits is required when scale > 0")
        return
    if int(bits) not in set(_HI_NERV_TRAIN_TIME_QUANT_NOISE_BITS):
        raise ValueError(
            f"{field_prefix}_bits must be one of "
            f"{list(_HI_NERV_TRAIN_TIME_QUANT_NOISE_BITS)}"
        )


def _resolve_decoder_weight_waterfill_plan_path(args: argparse.Namespace) -> Path | None:
    path = getattr(args, "decoder_weight_waterfill_plan_json", None)
    if path is None:
        return None
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve(strict=False)


def _unique_waterfill_aliases(values: list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


def _decoder_weight_waterfill_candidate_aliases(value: Any) -> tuple[str, ...]:
    text = str(value or "").strip()
    if not text:
        return ()
    aliases = [text]
    has_double_colon = "::" in text
    if has_double_colon:
        parts = [part.strip() for part in text.split("::") if part.strip()]
        if len(parts) >= 3 and parts[0] in {"hi_nerv", "hinerv", "snerv"}:
            aliases.append(parts[1])
        elif parts:
            aliases.append(parts[-1])
    for marker in (
        ":hi_nerv_decoder_weight_waterfill:",
        ":hinerv_decoder_weight_waterfill:",
        ":snerv_decoder_weight_waterfill:",
    ):
        if marker in text:
            aliases.append(text.split(marker, 1)[0])
    if ":" in text and not has_double_colon:
        aliases.append(text.rsplit(":", 1)[-1])
        aliases.append(text.split(":", 1)[0])
    return _unique_waterfill_aliases(aliases)


def _decoder_weight_waterfill_launch_candidate_keys(
    args: argparse.Namespace,
    *,
    modelsize_candidate: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    keys: list[str] = []
    if modelsize_candidate:
        for field in ("candidate_id", "row_id", "planner_row_id", "_modelsize_row_id"):
            keys.extend(
                _decoder_weight_waterfill_candidate_aliases(
                    modelsize_candidate.get(field)
                )
            )
        return _unique_waterfill_aliases(keys)
    keys.extend(
        _decoder_weight_waterfill_candidate_aliases(
            getattr(args, "modelsize_row", None)
        )
    )
    return _unique_waterfill_aliases(keys)


def _shape_tuple_from_waterfill_row(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ValueError("decoder_weight_waterfill row shape must be a sequence")
    out: list[int] = []
    for dim in value:
        try:
            parsed = int(dim)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"decoder_weight_waterfill row shape dim is not an integer: {dim!r}"
            ) from exc
        if parsed < 0:
            raise ValueError(
                f"decoder_weight_waterfill row shape dim must be non-negative: {parsed}"
            )
        out.append(parsed)
    return tuple(out)


def _declared_waterfill_row_shape(row: Mapping[str, Any]) -> tuple[int, ...] | None:
    for field in ("shape", "tensor_shape", "expected_shape", "state_shape"):
        if field in row:
            return _shape_tuple_from_waterfill_row(row.get(field))
    return None


def _declared_waterfill_row_numel(row: Mapping[str, Any]) -> int | None:
    for field in ("numel", "tensor_numel", "group_numel", "expected_numel"):
        if field not in row:
            continue
        value = row.get(field)
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"decoder_weight_waterfill row numel field {field} is not an integer"
            ) from exc
        if parsed < 0:
            raise ValueError(
                f"decoder_weight_waterfill row numel field {field} must be non-negative"
            )
        return parsed
    return None


def _validate_decoder_weight_waterfill_plan_for_launch(
    payload: dict[str, Any],
    *,
    args: argparse.Namespace,
    modelsize_candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    blockers: list[str] = []
    family = payload.get("family")
    if family not in (None, "", "hi_nerv"):
        blockers.append(f"decoder_weight_waterfill_family_mismatch:{family}")

    plan_candidate_id = str(payload.get("candidate_id") or "").strip()
    plan_candidate_keys = _decoder_weight_waterfill_candidate_aliases(
        plan_candidate_id
    )
    launch_candidate_keys = _decoder_weight_waterfill_launch_candidate_keys(
        args,
        modelsize_candidate=modelsize_candidate,
    )
    if not plan_candidate_keys:
        blockers.append("decoder_weight_waterfill_plan_candidate_id_missing")
    if not launch_candidate_keys:
        blockers.append("decoder_weight_waterfill_launch_candidate_missing")
    elif plan_candidate_keys and not set(plan_candidate_keys) & set(launch_candidate_keys):
        blockers.append(
            "decoder_weight_waterfill_candidate_id_mismatch:"
            f"{plan_candidate_id}"
        )

    raw_rows = payload.get("rows")
    if not isinstance(raw_rows, list):
        blockers.append("decoder_weight_waterfill_rows_not_list")
        rows = []
    else:
        rows = raw_rows
    if not rows:
        blockers.append("decoder_weight_waterfill_rows_empty")

    expected_shapes: dict[str, tuple[int, ...]] = {}
    try:
        from tac.substrates.hi_nerv.architecture import expected_decoder_state_shapes

        cfg = _config_from_args(args, modelsize_candidate=modelsize_candidate)
        expected_shapes = expected_decoder_state_shapes(cfg)
    except Exception as exc:  # pragma: no cover - defensive prelaunch metadata path
        blockers.append(
            "decoder_weight_waterfill_expected_shape_resolution_failed:"
            f"{type(exc).__name__}"
        )

    validated_rows: list[dict[str, Any]] = []
    for idx, row_obj in enumerate(rows):
        if not isinstance(row_obj, Mapping):
            blockers.append(f"decoder_weight_waterfill_row_not_mapping:{idx}")
            continue
        group_name = row_obj.get("group_name")
        if not isinstance(group_name, str) or not group_name:
            blockers.append(f"decoder_weight_waterfill_row_missing_group_name:{idx}")
            continue
        expected_shape = expected_shapes.get(group_name)
        if expected_shape is None:
            blockers.append(f"decoder_weight_waterfill_group_missing:{group_name}")
            continue
        try:
            declared_shape = _declared_waterfill_row_shape(row_obj)
            declared_numel = _declared_waterfill_row_numel(row_obj)
        except ValueError as exc:
            blockers.append(
                f"decoder_weight_waterfill_row_metadata_invalid:{group_name}:{exc}"
            )
            continue
        row_validation = {
            "group_name": group_name,
            "expected_shape": [int(dim) for dim in expected_shape],
            "expected_numel": int(math.prod(expected_shape)),
            "shape_checked": declared_shape is not None,
            "numel_checked": declared_numel is not None,
        }
        if declared_shape is not None and declared_shape != expected_shape:
            blockers.append(f"decoder_weight_waterfill_shape_mismatch:{group_name}")
            row_validation["declared_shape"] = [int(dim) for dim in declared_shape]
        expected_numel = int(math.prod(expected_shape))
        if declared_numel is not None and declared_numel != expected_numel:
            blockers.append(f"decoder_weight_waterfill_numel_mismatch:{group_name}")
            row_validation["declared_numel"] = int(declared_numel)
        validated_rows.append(row_validation)

    fake_quant_targeted_tensor_count: int | None = None
    fake_quant_per_tensor_bits: dict[str, int] = {}
    try:
        from tac.substrates.hi_nerv.bitstream import (
            build_decoder_waterfill_fake_quant_forward_plan,
        )

        fake_quant_plan = build_decoder_waterfill_fake_quant_forward_plan(payload)
        fake_quant_targeted_tensor_count = int(
            fake_quant_plan.get("targeted_tensor_count") or 0
        )
        per_tensor_bits = fake_quant_plan.get("per_tensor_bits")
        if isinstance(per_tensor_bits, Mapping):
            fake_quant_per_tensor_bits = {
                str(name): int(bits) for name, bits in per_tensor_bits.items()
            }
        actuation_blockers = [
            str(blocker) for blocker in fake_quant_plan.get("actuation_blockers") or []
        ]
        if actuation_blockers:
            blockers.extend(
                f"decoder_weight_waterfill_fake_quant_actuation_blocker:{blocker}"
                for blocker in actuation_blockers
            )
        if not fake_quant_plan.get("per_tensor_bits"):
            blockers.append("decoder_weight_waterfill_fake_quant_no_targets")
    except Exception as exc:
        blockers.append(
            "decoder_weight_waterfill_fake_quant_bits_invalid:"
            f"{type(exc).__name__}:{exc}"
        )

    if blockers:
        raise ValueError(
            "invalid decoder_weight_waterfill_plan_json for HiNeRV launch: "
            + ", ".join(dict.fromkeys(blockers))
        )
    return {
        "schema": "hi_nerv_trainer_decoder_weight_waterfill_launch_validation.v1",
        "validated": True,
        "family": payload.get("family"),
        "candidate_id": payload.get("candidate_id"),
        "plan_candidate_keys": list(plan_candidate_keys),
        "launch_candidate_keys": list(launch_candidate_keys),
        "matched_candidate_keys": sorted(
            str(key) for key in set(plan_candidate_keys) & set(launch_candidate_keys)
        ),
        "row_count": len(rows),
        "expected_decoder_group_count": len(expected_shapes),
        "validated_row_count": len(validated_rows),
        "validated_rows": validated_rows,
        "fake_quant_targeted_tensor_count": fake_quant_targeted_tensor_count,
        "fake_quant_per_tensor_bits": dict(sorted(fake_quant_per_tensor_bits.items())),
        "blockers": [],
        "authority": TRAINER_AUTHORITY,
        **FALSE_AUTHORITY,
    }


def _decoder_weight_waterfill_plan_from_args(
    args: argparse.Namespace,
    *,
    modelsize_candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    path = _resolve_decoder_weight_waterfill_plan_path(args)
    if path is None:
        return None
    if not path.is_file():
        raise ValueError(
            "decoder_weight_waterfill_plan_json must point at an existing "
            f"file; got {path.as_posix()}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("decoder_weight_waterfill_plan_json must contain a JSON object")
    if payload.get("schema") != "nerv_decoder_weight_waterfill.v1":
        raise ValueError(
            "decoder_weight_waterfill_plan_json schema must be "
            "'nerv_decoder_weight_waterfill.v1'"
        )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("decoder_weight_waterfill_plan_json rows must be a list")
    payload["_trainer_launch_validation"] = _validate_decoder_weight_waterfill_plan_for_launch(
        payload,
        args=args,
        modelsize_candidate=modelsize_candidate,
    )
    return payload


def _configure_decoder_fake_quant_forward(
    *,
    model: Any,
    controls: HiNervTrainTimeControlConfig,
    decoder_weight_waterfill_plan: dict[str, Any] | None,
) -> dict[str, Any]:
    if decoder_weight_waterfill_plan is not None:
        if not hasattr(model, "configure_decoder_fake_quant_forward_from_waterfill_plan"):
            raise RuntimeError(
                "HiNeRV decoder weight waterfill QAT requires an MLX model with "
                "configure_decoder_fake_quant_forward_from_waterfill_plan"
            )
        fallback_quant_bits = (
            int(controls.decoder_fake_quant_bits)
            if bool(controls.decoder_fake_quant_forward_enabled)
            else None
        )
        waterfill_report = model.configure_decoder_fake_quant_forward_from_waterfill_plan(
            decoder_weight_waterfill_plan,
            fallback_quant_bits=fallback_quant_bits,
        )
        return {
            "schema": "hi_nerv_decoder_fake_quant_forward_binding.v1",
            "mode": "decoder_weight_waterfill_plan",
            "enabled": bool(waterfill_report.get("configured")),
            "uniform_fake_quant_fallback_enabled": bool(
                controls.decoder_fake_quant_forward_enabled
            ),
            "fallback_quant_bits": fallback_quant_bits,
            "waterfill_fake_quant_forward": waterfill_report,
            "authority": TRAINER_AUTHORITY,
            **FALSE_AUTHORITY,
        }
    if controls.decoder_fake_quant_forward_enabled:
        model.configure_decoder_fake_quant_forward(
            enabled=True,
            quant_bits=int(controls.decoder_fake_quant_bits),
        )
        return {
            "schema": "hi_nerv_decoder_fake_quant_forward_binding.v1",
            "mode": "uniform_decoder_fake_quant",
            "enabled": True,
            "quant_bits": int(controls.decoder_fake_quant_bits),
            "per_tensor_bits": {},
            "authority": TRAINER_AUTHORITY,
            **FALSE_AUTHORITY,
        }
    return {
        "schema": "hi_nerv_decoder_fake_quant_forward_binding.v1",
        "mode": "disabled",
        "enabled": False,
        "quant_bits": None,
        "per_tensor_bits": {},
        "authority": TRAINER_AUTHORITY,
        **FALSE_AUTHORITY,
    }


def _initialize_output_head_target_bias(
    *,
    model: Any,
    target_rgb_0: Any,
    target_rgb_1: Any,
    controls: HiNervTrainTimeControlConfig,
    source_pair_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    control = controls.validated()
    if not bool(control.output_head_target_bias_init_enabled):
        return {
            "schema": "hi_nerv_output_head_target_bias_init.v1",
            "enabled": False,
            "reason": "disabled_by_cli",
            "runtime_sidecar_bytes": 0,
            "archive_charged_decoder_tensors": [],
            "blockers": ["hinerv_output_head_target_bias_init_disabled"],
            "authority": TRAINER_AUTHORITY,
            **FALSE_AUTHORITY,
        }
    initializer = getattr(model, "initialize_output_head_bias_from_targets", None)
    if not callable(initializer):
        raise RuntimeError(
            "HiNeRV model lacks initialize_output_head_bias_from_targets; "
            "refusing neutral-gray-prone long-run launch"
        )
    payload = dict(
        initializer(
            target_rgb_0,
            target_rgb_1,
            epsilon=float(control.output_head_target_bias_init_epsilon),
        )
    )
    if bool(control.output_head_target_contrast_init_enabled):
        contrast_initializer = getattr(
            model,
            "initialize_output_head_contrast_from_targets",
            None,
        )
        if not callable(contrast_initializer):
            raise RuntimeError(
                "HiNeRV model lacks initialize_output_head_contrast_from_targets; "
                "refusing low-contrast-prone long-run launch"
            )
        import mlx.core as mx

        pair_count = int(target_rgb_0.shape[0])
        contrast_count = min(
            pair_count,
            int(control.output_head_target_contrast_init_max_pairs),
        )
        target0_subset = target_rgb_0[:contrast_count]
        target1_subset = target_rgb_1[:contrast_count]
        if source_pair_indices is None:
            pair_indices = mx.arange(contrast_count, dtype=mx.int32)
        else:
            if len(source_pair_indices) < contrast_count:
                raise ValueError(
                    "source_pair_indices shorter than target contrast init subset: "
                    f"{len(source_pair_indices)} < {contrast_count}"
                )
            pair_indices = mx.array(
                [int(value) for value in source_pair_indices[:contrast_count]],
                dtype=mx.int32,
            )
        contrast_payload = dict(
            contrast_initializer(
                target0_subset,
                target1_subset,
                pair_indices=pair_indices,
                min_output_std=float(
                    control.output_head_target_contrast_init_min_output_std
                ),
                max_gain=float(control.output_head_target_contrast_init_max_gain),
            )
        )
        payload["contrast_init"] = contrast_payload
        payload["archive_charged_decoder_tensors"] = sorted(
            {
                *[str(v) for v in payload.get("archive_charged_decoder_tensors", [])],
                *[
                    str(v)
                    for v in contrast_payload.get(
                        "archive_charged_decoder_tensors",
                        [],
                    )
                ],
            }
        )
    else:
        payload["contrast_init"] = {
            "schema": "hi_nerv_output_head_target_contrast_init.v1",
            "enabled": False,
            "reason": "disabled_by_cli",
            "runtime_sidecar_bytes": 0,
            "archive_charged_decoder_tensors": [],
            "blockers": ["hinerv_output_head_target_contrast_init_disabled"],
            "authority": TRAINER_AUTHORITY,
            **FALSE_AUTHORITY,
        }
    payload.update({
        "authority": TRAINER_AUTHORITY,
        **FALSE_AUTHORITY,
    })
    return payload


def _decoder_weight_waterfill_plan_attachment_metadata(
    *,
    args: argparse.Namespace,
    plan: dict[str, Any] | None,
    fake_quant_forward: dict[str, Any] | None,
) -> dict[str, Any]:
    path = _resolve_decoder_weight_waterfill_plan_path(args)
    if plan is None:
        return {
            "schema": "hi_nerv_decoder_weight_waterfill_plan_attachment.v1",
            "attached": False,
            "path": path.as_posix() if path is not None else None,
            "train_time_fake_quant_bound": False,
            "export_bound": False,
            "blockers": ["hinerv_decoder_weight_waterfill_plan_not_attached"],
            "authority": TRAINER_AUTHORITY,
            **FALSE_AUTHORITY,
        }
    rows = plan.get("rows") if isinstance(plan.get("rows"), list) else []
    waterfill_fake_quant = (
        fake_quant_forward.get("waterfill_fake_quant_forward")
        if isinstance(fake_quant_forward, dict)
        else None
    )
    return {
        "schema": "hi_nerv_decoder_weight_waterfill_plan_attachment.v1",
        "attached": True,
        "path": path.as_posix() if path is not None else None,
        "sha256": sha256_file(path) if path is not None and path.is_file() else None,
        "bytes": path.stat().st_size if path is not None and path.is_file() else None,
        "plan_schema": plan.get("schema"),
        "family": plan.get("family"),
        "candidate_id": plan.get("candidate_id"),
        "trainer_launch_validation": _metadata_safe(
            plan.get("_trainer_launch_validation")
        ),
        "row_count": len(rows),
        "train_time_fake_quant_bound": bool(
            isinstance(fake_quant_forward, dict) and fake_quant_forward.get("enabled")
        ),
        "export_bound": True,
        "fake_quant_forward": _metadata_safe(waterfill_fake_quant),
        "blockers": [
            *[str(blocker) for blocker in plan.get("blockers") or []],
            "contest_cpu_cuda_exact_eval_not_executed",
        ],
        "authority": TRAINER_AUTHORITY,
        **FALSE_AUTHORITY,
    }


def _build_train_time_decoder_control_callback(
    *,
    model: Any,
    controls: HiNervTrainTimeControlConfig,
    output_dir: Path,
) -> Any:
    if not controls.train_time_decoder_controls_enabled:
        return None
    path = output_dir / "hi_nerv_train_time_decoder_controls.jsonl"

    def _callback(*, epoch: int) -> dict[str, Any]:
        report = _apply_train_time_decoder_controls(
            model,
            controls,
            epoch=int(epoch),
        )
        if bool(report.get("applied")):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(report, sort_keys=True) + "\n")
        return report

    return _callback


def _apply_train_time_decoder_controls(
    model: Any,
    controls: HiNervTrainTimeControlConfig,
    *,
    epoch: int,
) -> dict[str, Any]:
    c = controls.validated()
    if not c.train_time_decoder_controls_enabled:
        return _train_time_decoder_control_report(
            controls=c,
            epoch=epoch,
            applied=False,
            reason="disabled",
        )
    if int(epoch) < int(c.train_time_decoder_control_start_epoch):
        return _train_time_decoder_control_report(
            controls=c,
            epoch=epoch,
            applied=False,
            reason="before_start_epoch",
        )
    cadence = int(c.train_time_decoder_control_frequency_epochs)
    if (int(epoch) - int(c.train_time_decoder_control_start_epoch)) % cadence != 0:
        return _train_time_decoder_control_report(
            controls=c,
            epoch=epoch,
            applied=False,
            reason="cadence_skip",
        )

    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten, tree_unflatten
    except Exception as exc:  # pragma: no cover - only on non-MLX hosts.
        raise RuntimeError("HiNeRV train-time decoder controls require MLX") from exc
    import numpy as np

    flat_items = list(tree_flatten(model.parameters()))
    flat = dict(flat_items)
    before_parameter_arrays = _capture_parameter_arrays(flat_items)
    selected = [
        (key, _mlx_tree_key_name(key), value)
        for key, value in flat_items
        if _is_train_time_decoder_control_tensor(_mlx_tree_key_name(key), value)
    ]
    selected_names = {name for _key, name, _value in selected}
    if not selected:
        raise RuntimeError(
            "HiNeRV train-time decoder controls selected no decoder tensors; "
            "adjust include/exclude selectors before launching"
        )

    changed_names: set[str] = set()
    pruned_values = 0
    pruning_threshold: float | None = None

    if float(c.train_time_decoder_pruning_ratio) > 0.0:
        arrays = [
            np.asarray(value, dtype=np.float32)
            for _key, _name, value in selected
            if np.asarray(value).size
        ]
        total_values = int(sum(arr.size for arr in arrays))
        target_pruned = math.floor(float(c.train_time_decoder_pruning_ratio) * total_values)
        if target_pruned > 0 and arrays:
            all_abs = np.concatenate([np.abs(arr).reshape(-1) for arr in arrays])
            pruning_threshold = float(np.partition(all_abs, target_pruned - 1)[target_pruned - 1])
            remaining = target_pruned
            for key, name, value in selected:
                if remaining <= 0:
                    break
                original = np.asarray(flat[key], dtype=np.float32)
                arr = np.array(original, copy=True)
                if arr.size == 0:
                    continue
                abs_arr = np.abs(arr)
                mask = abs_arr < pruning_threshold
                already = int(mask.sum())
                need = max(0, remaining - already)
                if need:
                    equal_flat = (abs_arr == pruning_threshold).reshape(-1)
                    equal_indices = np.flatnonzero(equal_flat)[:need]
                    mask_flat = mask.reshape(-1)
                    mask_flat[equal_indices] = True
                    mask = mask_flat.reshape(mask.shape)
                selected_here = int(mask.sum())
                if selected_here:
                    arr[mask] = 0.0
                    flat[key] = mx.array(arr).astype(value.dtype)
                    pruned_values += selected_here
                    remaining -= selected_here
                    changed_names.add(name)

    quant_noise_changed = 0
    max_abs_quant_delta = 0.0
    if (
        c.train_time_decoder_quant_noise_bits is not None
        and float(c.train_time_decoder_quant_noise_scale) > 0.0
    ):
        bits = int(c.train_time_decoder_quant_noise_bits)
        qmax = (1 << (bits - 1)) - 1
        rng = np.random.default_rng(int(c.train_time_decoder_quant_noise_seed) + int(epoch))
        for key, name, value in selected:
            arr = np.asarray(flat[key], dtype=np.float32)
            if arr.size == 0:
                continue
            abs_max = float(np.max(np.abs(arr))) if arr.size else 0.0
            if abs_max <= 0.0:
                continue
            step = abs_max / float(qmax)
            noise = rng.uniform(-0.5, 0.5, size=arr.shape).astype(np.float32)
            delta = np.where(
                arr != 0.0,
                noise * (step * float(c.train_time_decoder_quant_noise_scale)),
                0.0,
            ).astype(np.float32, copy=False)
            if not bool(np.any(delta != 0.0)):
                continue
            updated = arr + delta
            flat[key] = mx.array(updated).astype(value.dtype)
            quant_noise_changed += 1
            max_abs_quant_delta = max(max_abs_quant_delta, float(np.max(np.abs(delta))))
            changed_names.add(name)

    if not changed_names:
        raise RuntimeError(
            "HiNeRV train-time decoder controls were enabled but changed no "
            "decoder tensors"
        )
    after_parameter_arrays = _capture_parameter_arrays(list(flat.items()))
    mutation_identity = _build_train_time_decoder_mutation_identity(
        before_parameter_arrays=before_parameter_arrays,
        after_parameter_arrays=after_parameter_arrays,
        selected_tensor_names=selected_names,
    )
    non_decoder_changed = mutation_identity["non_decoder_changed_tensor_names"]
    if non_decoder_changed:
        raise RuntimeError(
            "HiNeRV train-time decoder controls changed non-decoder tensors: "
            + ", ".join(str(name) for name in non_decoder_changed)
        )
    changed_names = set(mutation_identity["changed_tensor_names"])

    model.update(tree_unflatten(list(flat.items())))
    mx.eval(model.parameters())
    return _train_time_decoder_control_report(
        controls=c,
        epoch=epoch,
        applied=True,
        reason="applied",
        selected_tensor_count=len(selected),
        changed_tensor_count=len(changed_names),
        changed_tensor_names=sorted(changed_names),
        pruned_values=pruned_values,
        pruning_threshold=pruning_threshold,
        quant_noise_changed_tensor_count=quant_noise_changed,
        quant_noise_max_abs_delta=max_abs_quant_delta,
        mutation_identity=mutation_identity,
    )


def _train_time_decoder_control_report(
    *,
    controls: HiNervTrainTimeControlConfig,
    epoch: int,
    applied: bool,
    reason: str,
    selected_tensor_count: int = 0,
    changed_tensor_count: int = 0,
    changed_tensor_names: list[str] | None = None,
    pruned_values: int = 0,
    pruning_threshold: float | None = None,
    quant_noise_changed_tensor_count: int = 0,
    quant_noise_max_abs_delta: float = 0.0,
    mutation_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "schema": HI_NERV_TRAIN_TIME_DECODER_CONTROL_REPORT_SCHEMA,
        "epoch": int(epoch),
        "applied": bool(applied),
        "reason": str(reason),
        "selected_tensor_count": int(selected_tensor_count),
        "changed_tensor_count": int(changed_tensor_count),
        "changed_tensor_names": list(changed_tensor_names or []),
        "pruning": {
            "ratio": float(controls.train_time_decoder_pruning_ratio),
            "pruned_values": int(pruned_values),
            "threshold": pruning_threshold,
        },
        "quant_noise": {
            "bits": (
                None
                if controls.train_time_decoder_quant_noise_bits is None
                else int(controls.train_time_decoder_quant_noise_bits)
            ),
            "scale": float(controls.train_time_decoder_quant_noise_scale),
            "seed": int(controls.train_time_decoder_quant_noise_seed),
            "changed_tensor_count": int(quant_noise_changed_tensor_count),
            "max_abs_delta": float(quant_noise_max_abs_delta),
        },
        "authority": TRAINER_AUTHORITY,
        **FALSE_AUTHORITY,
    }
    if mutation_identity is not None:
        report["mutation_identity"] = mutation_identity
    return report


def _capture_parameter_arrays(flat_items: list[tuple[Any, Any]]) -> dict[str, Any]:
    import numpy as np

    return {
        _mlx_tree_key_name(key): np.asarray(value, dtype=np.float32).copy()
        for key, value in flat_items
    }


def _numpy_array_sha256(value: Any) -> str:
    import numpy as np

    arr = np.ascontiguousarray(np.asarray(value))
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(b"\0")
    h.update(",".join(str(int(v)) for v in arr.shape).encode("utf-8"))
    h.update(b"\0")
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def _named_array_state_sha256(arrays: dict[str, Any]) -> str:
    h = hashlib.sha256()
    for name in sorted(arrays):
        h.update(str(name).encode("utf-8"))
        h.update(b"\0")
        h.update(_numpy_array_sha256(arrays[name]).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _build_train_time_decoder_mutation_identity(
    *,
    before_parameter_arrays: dict[str, Any],
    after_parameter_arrays: dict[str, Any],
    selected_tensor_names: set[str],
) -> dict[str, Any]:
    import numpy as np

    selected = sorted(str(name) for name in selected_tensor_names)
    changed_rows: list[dict[str, Any]] = []
    total_changed_values = 0
    total_values_in_changed_tensors = 0
    aggregate_abs_delta = 0.0
    max_abs_delta = 0.0
    for name in sorted(set(before_parameter_arrays) | set(after_parameter_arrays)):
        before = before_parameter_arrays.get(name)
        after = after_parameter_arrays.get(name)
        if before is None or after is None:
            raise RuntimeError(
                "HiNeRV train-time decoder controls changed parameter tree shape "
                f"at {name!r}"
            )
        before_arr = np.asarray(before, dtype=np.float32)
        after_arr = np.asarray(after, dtype=np.float32)
        if before_arr.shape != after_arr.shape:
            raise RuntimeError(
                "HiNeRV train-time decoder controls changed tensor shape "
                f"for {name!r}: {before_arr.shape} -> {after_arr.shape}"
            )
        before_sha = _numpy_array_sha256(before_arr)
        after_sha = _numpy_array_sha256(after_arr)
        if before_sha == after_sha:
            continue
        delta = after_arr.astype(np.float64) - before_arr.astype(np.float64)
        abs_delta = np.abs(delta)
        changed_value_count = int(np.count_nonzero(delta != 0.0))
        numel = int(after_arr.size)
        total_changed_values += changed_value_count
        total_values_in_changed_tensors += numel
        aggregate_abs_delta += float(np.sum(abs_delta))
        max_abs_delta = max(max_abs_delta, float(np.max(abs_delta)) if numel else 0.0)
        changed_rows.append(
            {
                "tensor_name": name,
                "shape": [int(v) for v in after_arr.shape],
                "numel": numel,
                "selected_by_decoder_control": name in selected_tensor_names,
                "sha256_before": before_sha,
                "sha256_after": after_sha,
                "changed_value_count": changed_value_count,
                "zero_count_before": int(np.count_nonzero(before_arr == 0.0)),
                "zero_count_after": int(np.count_nonzero(after_arr == 0.0)),
                "max_abs_delta": float(np.max(abs_delta)) if numel else 0.0,
                "mean_abs_delta": float(np.mean(abs_delta)) if numel else 0.0,
            }
        )
    changed_names = [row["tensor_name"] for row in changed_rows]
    non_decoder_changed = [
        str(name)
        for name in changed_names
        if str(name) not in selected_tensor_names
    ]
    before_selected = {
        name: before_parameter_arrays[name]
        for name in selected
        if name in before_parameter_arrays
    }
    after_selected = {
        name: after_parameter_arrays[name]
        for name in selected
        if name in after_parameter_arrays
    }
    return {
        "schema": HI_NERV_TRAIN_TIME_DECODER_MUTATION_IDENTITY_SCHEMA,
        "selector_include_substrings": list(_HI_NERV_DECODER_CONTROL_INCLUDE_SUBSTRINGS),
        "selector_exclude_substrings": list(_HI_NERV_DECODER_CONTROL_EXCLUDE_SUBSTRINGS),
        "selected_tensor_count": len(selected),
        "selected_tensor_names": selected,
        "changed_tensor_count": len(changed_rows),
        "changed_tensor_names": changed_names,
        "non_decoder_changed_tensor_names": non_decoder_changed,
        "decoder_only_mutation": not non_decoder_changed,
        "selected_state_sha256_before": _named_array_state_sha256(before_selected),
        "selected_state_sha256_after": _named_array_state_sha256(after_selected),
        "changed_rows": changed_rows,
        "changed_value_count": int(total_changed_values),
        "changed_tensor_value_count": int(total_values_in_changed_tensors),
        "max_abs_delta": float(max_abs_delta),
        "mean_abs_delta_over_changed_tensors": (
            float(aggregate_abs_delta / total_values_in_changed_tensors)
            if total_values_in_changed_tensors
            else 0.0
        ),
        "authority": TRAINER_AUTHORITY,
        **FALSE_AUTHORITY,
    }


def _mlx_tree_key_name(key: Any) -> str:
    if isinstance(key, (tuple, list)):
        return ".".join(str(part) for part in key if str(part))
    return str(key)


def _is_train_time_decoder_control_tensor(name: str, value: Any) -> bool:
    lowered = str(name).lower()
    if any(token in lowered for token in _HI_NERV_DECODER_CONTROL_EXCLUDE_SUBSTRINGS):
        return False
    if not any(token in lowered for token in _HI_NERV_DECODER_CONTROL_INCLUDE_SUBSTRINGS):
        return False
    shape = getattr(value, "shape", ())
    return bool(shape) and int(math.prod(int(v) for v in shape)) > 0


def _prioritized_pair_indices_from_args(args: argparse.Namespace) -> tuple[int, ...]:
    try:
        merged = merge_pair_indices(
            parse_pair_indices_csv(
                str(getattr(args, "prioritized_pair_indices", "") or ""),
                field="prioritized_pair_indices",
            ),
            load_pair_indices_file(
                getattr(args, "prioritized_pair_indices_file", None),
                base=REPO_ROOT,
                field="prioritized_pair_indices_file",
            ),
        )
        return validate_pair_indices_in_range(
            merged,
            num_pairs=int(args.num_pairs),
            field="prioritized_pair_indices",
        )
    except HardPairIndicesError as exc:
        raise ValueError(f"invalid HiNeRV prioritized pair indices: {exc}") from exc


def _parse_finite_nonnegative_pairs(
    entries: list[str] | tuple[str, ...] | None,
    *,
    field: str,
    integer_keys: bool,
) -> dict[Any, float]:
    out: dict[Any, float] = {}
    for raw_entry in entries or []:
        text = str(raw_entry)
        if "=" not in text:
            raise ValueError(f"{field} entries must be KEY=VALUE; got {text!r}")
        raw_key, raw_value = text.split("=", 1)
        key_text = raw_key.strip()
        if not key_text:
            raise ValueError(f"{field} key must be non-empty")
        if integer_keys:
            try:
                key: Any = int(key_text)
            except ValueError as exc:
                raise ValueError(f"{field} key must be an integer pair index") from exc
            if int(key) < 0:
                raise ValueError(f"{field} pair index must be non-negative; got {key}")
        else:
            key = key_text
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"{field}[{key!r}] must be a finite float") from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"{field}[{key!r}] must be finite and >= 0; got {value!r}")
        out[key] = value
    return out


def _pair_sampling_weights_from_args(args: argparse.Namespace) -> dict[int, float]:
    return {
        int(key): float(value)
        for key, value in _parse_finite_nonnegative_pairs(
            list(getattr(args, "pair_sampling_weight", []) or []),
            field="pair_sampling_weight",
            integer_keys=True,
        ).items()
    }


def _gradient_multiplier_by_name_from_args(
    args: argparse.Namespace,
) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in _parse_finite_nonnegative_pairs(
            list(getattr(args, "gradient_multiplier", []) or []),
            field="gradient_multiplier",
            integer_keys=False,
        ).items()
    }


def _optimizer_control_metadata_from_args(args: argparse.Namespace) -> dict[str, Any]:
    warmup_steps_per_epoch = int(getattr(args, "optimizer_warmup_steps_per_epoch", 1))
    if warmup_steps_per_epoch <= 0:
        raise ValueError("optimizer_warmup_steps_per_epoch must be positive")
    pair_sampling_default_weight = float(
        getattr(args, "pair_sampling_default_weight", 1.0)
    )
    if (
        not math.isfinite(pair_sampling_default_weight)
        or pair_sampling_default_weight < 0.0
    ):
        raise ValueError("pair_sampling_default_weight must be finite and >= 0")
    pair_sampling_weights = _pair_sampling_weights_from_args(args)
    gradient_multiplier_by_name = _gradient_multiplier_by_name_from_args(args)
    bias_gradient_multiplier = getattr(args, "bias_gradient_multiplier", None)
    if bias_gradient_multiplier is not None:
        bias_gradient_multiplier = float(bias_gradient_multiplier)
        if not math.isfinite(bias_gradient_multiplier) or bias_gradient_multiplier < 0.0:
            raise ValueError("bias_gradient_multiplier must be finite and >= 0")
    output_head_bias_gradient_multiplier = float(
        getattr(args, "output_head_bias_gradient_multiplier", 1.0)
    )
    if (
        not math.isfinite(output_head_bias_gradient_multiplier)
        or output_head_bias_gradient_multiplier < 0.0
    ):
        raise ValueError("output_head_bias_gradient_multiplier must be finite and >= 0")
    return {
        "schema": "hi_nerv_direct_trainer_optimizer_control.v1",
        "warmup_steps_per_epoch": int(warmup_steps_per_epoch),
        "pr95_stage_source_weight_amplification_enabled": bool(
            getattr(args, "pr95_stage_source_weight_amplification", False)
        ),
        "pair_sampling_default_weight": float(pair_sampling_default_weight),
        "pair_sampling_weights": {
            int(key): float(value) for key, value in pair_sampling_weights.items()
        },
        "gradient_multiplier_by_name": {
            str(key): float(value)
            for key, value in gradient_multiplier_by_name.items()
        },
        "bias_gradient_multiplier": bias_gradient_multiplier,
        "output_head_bias_gradient_multiplier": float(
            output_head_bias_gradient_multiplier
        ),
        "authority": TRAINER_AUTHORITY,
        **FALSE_AUTHORITY,
    }


def _prioritized_pair_training_metadata(
    pair_indices: tuple[int, ...],
    *,
    target_hydration_pair_indices_consumed: bool = False,
) -> dict[str, Any]:
    consumed = bool(pair_indices) and bool(target_hydration_pair_indices_consumed)
    local_pair_indices = list(range(len(pair_indices))) if consumed else [int(value) for value in pair_indices]
    return {
        "schema": "hi_nerv_direct_trainer_prioritized_pair_training.v1",
        "enabled": bool(pair_indices),
        "pair_indices": [int(value) for value in pair_indices],
        "source_pair_indices": [int(value) for value in pair_indices],
        "local_pair_indices": local_pair_indices,
        "pair_count": len(pair_indices),
        "sampling_scope": (
            "explicit_source_pair_target_hydration" if consumed else "local_mlx_training_batch_emphasis_only"
        ),
        "pair_index_domain": (
            "source_video_pair_indices" if consumed else "decoded_prefix_pair_indices_0_to_num_pairs_minus_1"
        ),
        "pair_index_alignment_mode": (
            "local_target_rows_to_source_pair_indices" if consumed else "identity_local_rows_are_source_pairs"
        ),
        "arbitrary_source_pair_hydration": consumed,
        "target_hydration_pair_indices_consumed": consumed,
        "requires_num_pairs_covering_pair_ids": bool(pair_indices) and not consumed,
        "authority": "macos_mlx_research_signal_false_authority",
        **FALSE_AUTHORITY,
    }


def _prioritized_pair_training_lineage_metadata(
    pair_indices: tuple[int, ...],
    *,
    target_hydration_pair_indices_consumed: bool = False,
) -> dict[str, Any]:
    consumed = bool(pair_indices) and bool(target_hydration_pair_indices_consumed)
    local_pair_indices = list(range(len(pair_indices))) if consumed else [int(value) for value in pair_indices]
    return {
        "schema": "hi_nerv_direct_trainer_prioritized_pair_training.v1",
        "enabled": bool(pair_indices),
        "pair_indices": [int(value) for value in pair_indices],
        "source_pair_indices": [int(value) for value in pair_indices],
        "local_pair_indices": local_pair_indices,
        "pair_count": len(pair_indices),
        "sampling_scope": (
            "explicit_source_pair_target_hydration" if consumed else "local_mlx_training_batch_emphasis_only"
        ),
        "pair_index_domain": (
            "source_video_pair_indices" if consumed else "decoded_prefix_pair_indices_0_to_num_pairs_minus_1"
        ),
        "pair_index_alignment_mode": (
            "local_target_rows_to_source_pair_indices" if consumed else "identity_local_rows_are_source_pairs"
        ),
        "arbitrary_source_pair_hydration": consumed,
        "target_hydration_pair_indices_consumed": consumed,
        "requires_num_pairs_covering_pair_ids": bool(pair_indices) and not consumed,
        "authority": "macos_mlx_research_signal_false_authority",
        "canonical_authority_surface": ("TrainingArtifact top-level false-authority fields"),
    }


def _pose_student_input_channels(preprocess: str) -> int:
    if preprocess == "rgb":
        return 3
    if preprocess == "pr95_yuv6":
        return 6
    raise ValueError(f"unsupported pose_student_input_preprocess {preprocess!r}; expected 'rgb' or 'pr95_yuv6'")


def _curriculum_stages_from_args(args: argparse.Namespace) -> tuple[Any, ...] | None:
    if not bool(args.staged_scorer_curriculum):
        return None
    return _build_staged_scorer_curriculum(
        epochs=int(args.epochs),
        recon_fraction=float(args.staged_scorer_recon_fraction),
        segnet_fraction=float(args.staged_scorer_segnet_fraction),
        final_recon_weight=float(args.staged_scorer_final_recon_weight),
        segnet_lr_scale=float(args.staged_scorer_segnet_lr_scale),
        final_lr_scale=float(args.staged_scorer_final_lr_scale),
    )


def _build_staged_scorer_curriculum(
    *,
    epochs: int,
    recon_fraction: float,
    segnet_fraction: float,
    final_recon_weight: float,
    segnet_lr_scale: float,
    final_lr_scale: float,
) -> tuple[Any, ...]:
    from tac.training.long_training_canonical import CurriculumStage

    if epochs < 3:
        raise ValueError(
            "staged scorer curriculum requires epochs >= 3 so recon, SegNet, "
            f"and joint scorer stages are all non-empty; got {epochs}"
        )
    if not (0.0 < recon_fraction < 1.0):
        raise ValueError(f"staged_scorer_recon_fraction must be in (0, 1); got {recon_fraction}")
    if not (0.0 < segnet_fraction < 1.0):
        raise ValueError(f"staged_scorer_segnet_fraction must be in (0, 1); got {segnet_fraction}")
    if recon_fraction + segnet_fraction >= 1.0:
        raise ValueError(
            "staged scorer recon + SegNet fractions must leave a non-empty "
            "joint scorer stage; got "
            f"recon={recon_fraction} segnet={segnet_fraction}"
        )
    if final_recon_weight < 0.0:
        raise ValueError(f"staged_scorer_final_recon_weight must be non-negative; got {final_recon_weight}")
    if segnet_lr_scale <= 0.0 or final_lr_scale <= 0.0:
        raise ValueError(
            f"staged scorer lr scales must be positive; got segnet={segnet_lr_scale} final={final_lr_scale}"
        )

    recon_end = max(1, min(epochs - 2, round(epochs * recon_fraction)))
    segnet_epochs = max(1, round(epochs * segnet_fraction))
    segnet_end = max(recon_end + 1, min(epochs - 1, recon_end + segnet_epochs))
    if segnet_end >= epochs:
        segnet_end = epochs - 1
    return (
        CurriculumStage(
            name="hi_nerv_receiver_fit_recon_scaffold",
            start_epoch=0,
            end_epoch=recon_end,
            loss_weights={
                "recon": 1.0,
                "distill": 0.0,
                "pose_distill": 0.0,
                "scorer_input_guard": 1.0,
                "scorer_input_contrast_floor": 1.0,
                "scorer_input_shape_tether": 1.0,
                "posenet_temporal_signal_floor": 1.0,
                "segnet_direct_live_distill": 0.0,
                "segnet_direct_live_base_loss": 1.0,
            },
            lr_scale=1.0,
            notes=(
                "Contest-scorer input stabilization stage: fit receiver outputs "
                "without admitting unstable SegNet/PoseNet surrogate gradients."
            ),
        ),
        CurriculumStage(
            name="hi_nerv_segnet_last_frame_admission",
            start_epoch=recon_end,
            end_epoch=segnet_end,
            loss_weights={
                "recon": 1.0,
                "distill": 1.0,
                "pose_distill": 0.0,
                "scorer_input_guard": 1.0,
                "scorer_input_contrast_floor": 1.0,
                "scorer_input_shape_tether": 1.0,
                "posenet_temporal_signal_floor": 1.0,
                "segnet_direct_live_distill": 1.0,
                "segnet_direct_live_base_loss": 1.0,
            },
            lr_scale=float(segnet_lr_scale),
            notes=(
                "Admit SegNet last-frame scorer surrogate after receiver "
                "inputs are nondegenerate; PoseNet remains held out."
            ),
        ),
        CurriculumStage(
            name="hi_nerv_joint_scorer_waterfill_finetune",
            start_epoch=segnet_end,
            end_epoch=epochs,
            loss_weights={
                "recon": float(final_recon_weight),
                "distill": 1.0,
                "pose_distill": 1.0,
                "scorer_input_guard": 1.0,
                "scorer_input_contrast_floor": 1.0,
                "scorer_input_shape_tether": 1.0,
                "posenet_temporal_signal_floor": 1.0,
                "segnet_direct_live_distill": 1.0,
                "segnet_direct_live_base_loss": 1.0,
            },
            lr_scale=float(final_lr_scale),
            notes=(
                "Final contest-scorer finetune: keep only enough reconstruction "
                "anchor to preserve scorer inputs while SegNet/PoseNet terms "
                "drive the renderer."
            ),
        ),
    )


def _resolve_output_dir(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    if args.output_dir is not None:
        output = args.output_dir.expanduser()
        if not output.is_absolute():
            output = REPO_ROOT / output
        output = output.resolve(strict=False)
        if _looks_local(output) and not bool(args.allow_local_output_dir):
            raise StorageTierError("hi_nerv_mlx_trainer_output_storage_preflight_failed: local_disk_tier_disabled")
        output.mkdir(parents=True, exist_ok=True)
        return output, {
            "schema": "hi_nerv_mlx_trainer_explicit_output_preflight.v1",
            "selected_workload_root": output.as_posix(),
            "explicit_output_dir": True,
            "local_output_explicitly_allowed": bool(args.allow_local_output_dir),
            "operator_storage_policy": operator_storage_policy_payload(),
            "blockers": [],
            **FALSE_AUTHORITY,
        }

    tiers = parse_storage_tier_specs(
        operator_storage_tier_cli_specs(()),
        repo_root=REPO_ROOT,
        reserve_free_gb=float(args.storage_reserve_free_gb),
        allow_local_disk=False,
    )
    subdir = f"{str(args.storage_workload_subdir).strip('/')}/{args.modelsize_row!s}_{int(args.num_pairs)}pairs"
    plan = plan_experiment_storage(
        tiers,
        workload_subdir=subdir,
        requested_bytes=int(args.storage_expected_bytes),
        min_free_bytes=0,
        create=True,
        probe_writable=True,
    )
    output = require_selected_storage(plan)
    payload = plan.to_dict()
    payload["operator_storage_policy"] = operator_storage_policy_payload()
    payload["selected_workload_root"] = output.as_posix()
    payload.update(FALSE_AUTHORITY)
    return output, payload


def _direct_trainer_canonicalization_contract(
    *,
    mode: str,
    modelsize_candidate_consumption: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Describe why this script is not the production launch authority.

    The compact runner owns planner rows, campaign locks, source parity,
    full-video prefilter, replay gates, and exact-axis handoff. This trainer
    remains useful as the subprocess implementation behind that runner and for
    explicitly local research smokes, but direct artifacts must never look like
    queue-owned launch authority.
    """

    modelsize_consumed = bool(
        isinstance(modelsize_candidate_consumption, Mapping)
        and modelsize_candidate_consumption.get("attached") is True
        and modelsize_candidate_consumption.get("consumed_by_trainer_config") is True
        and (
            modelsize_candidate_consumption.get(
                "consumed_by_archive_export_hard_byte_ceiling"
            )
            is True
        )
    )
    blockers = [
        blocker
        for blocker in DIRECT_TRAINER_CANONICALIZATION_BLOCKERS
        if not modelsize_consumed
        or blocker != "hinerv_direct_modelsize_row_not_budget_candidate_contract"
    ]
    return {
        "schema": DIRECT_TRAINER_CANONICALIZATION_SCHEMA,
        "canonical_runner_entrypoint": DIRECT_TRAINER_CANONICAL_RUNNER_ENTRYPOINT,
        "direct_trainer_role": "runner_subprocess_or_research_smoke_only",
        "mode": str(mode),
        "planner_row_required": True,
        "planner_row_id": None,
        "source_parity_contract_consumed": False,
        "source_faithfulness_launch_gate_consumed": False,
        "pr95_prelaunch_gate_consumed": False,
        "modelsize_candidate_contract_consumed": modelsize_consumed,
        "modelsize_candidate_consumption": _metadata_safe(
            modelsize_candidate_consumption
        ),
        "compact_runner_startup_marker_present": False,
        "full_video_mlx_prefilter_bound": False,
        "local_cpu_replay_gate_bound": False,
        "trainer_launch_allowed": False,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _pr95_full_control_contract(
    args: argparse.Namespace,
    *,
    train_time_controls: HiNervTrainTimeControlConfig | None = None,
    train_time_section_byte_control: Mapping[str, Any] | None = None,
    require_measured_section_byte_control: bool = False,
) -> dict[str, Any]:
    """Fail-closed production-full control audit for PR95-critical HiNeRV runs."""

    if train_time_controls is None:
        train_time_controls = _train_time_control_config_from_args(args)
    blockers: list[str] = []
    distillation_weight = float(getattr(args, "distillation_weight", 0.0))
    segnet_live_calibration_weight = float(
        train_time_controls.segnet_student_live_calibration_weight
    )
    segnet_direct_live_weight = float(
        train_time_controls.segnet_direct_live_distillation_weight
    )
    segnet_class_escape_weight = max(
        float(train_time_controls.segnet_direct_live_class_histogram_weight),
        float(train_time_controls.segnet_direct_live_class_balanced_hinge_weight),
        float(train_time_controls.segnet_direct_live_class_balanced_ce_weight),
        float(
            train_time_controls.segnet_direct_live_class_balanced_squared_hinge_weight
        ),
    )
    pose_distillation_weight = float(getattr(args, "pose_distillation_weight", 0.0))
    eval_roundtrip_ste = bool(getattr(args, "eval_roundtrip_ste", False))
    pose_preprocess = str(getattr(args, "pose_student_input_preprocess", ""))
    pr95_curriculum = bool(getattr(args, "pr95_faithful_curriculum", False))
    pr95_source_weight_amplification = bool(
        getattr(args, "pr95_stage_source_weight_amplification", False)
    )
    pr95_total_epochs = getattr(args, "pr95_curriculum_total_epochs", None)
    epochs = int(getattr(args, "epochs", 0))
    coder_qat = bool(getattr(args, "coder_qat", False))
    c1a_entropy_weight = float(getattr(args, "coder_qat_c1a_entropy_weight", 0.0))
    c1a_sigma = float(getattr(args, "coder_qat_c1a_sigma", 0.0))
    c1a_sample_size = int(getattr(args, "coder_qat_c1a_sample_size", 0))
    train_time_dual_ascent_enabled = not bool(
        getattr(args, "disable_train_time_dual_ascent", False)
    )
    hard_byte_ceiling_attached = (
        getattr(args, "hard_byte_ceiling", None) is not None
        or getattr(args, "modelsize_candidate_json", None) is not None
    )
    train_time_section_byte_metrics_enabled = bool(
        not bool(getattr(args, "disable_train_time_section_byte_metrics", False))
        and hard_byte_ceiling_attached
        and coder_qat
    )
    measured_section_byte_control_attached = train_time_section_byte_control is not None
    measured_section_byte_control_active = bool(
        isinstance(train_time_section_byte_control, Mapping)
        and train_time_section_byte_control.get("active") is True
    )
    measured_section_byte_control_blockers = [
        str(blocker)
        for blocker in (
            train_time_section_byte_control.get("blockers", [])
            if isinstance(train_time_section_byte_control, Mapping)
            else []
        )
    ]
    measured_section_byte_controlled_count = int(
        train_time_section_byte_control.get("controlled_section_count") or 0
        if isinstance(train_time_section_byte_control, Mapping)
        else 0
    )
    measured_section_pending_count = int(
        train_time_section_byte_control.get("pending_section_count") or 0
        if isinstance(train_time_section_byte_control, Mapping)
        else 0
    )
    section_byte_measurement_phase = (
        "post_model_measured"
        if measured_section_byte_control_attached
        else "pre_model_requested"
    )
    ema_archive_selection = bool(getattr(args, "ema_archive_selection", False))
    archive_parse_back_selection = bool(getattr(args, "post_export_receiver_cache_quality_gate", False))
    scorer_input_guard_weight = float(
        getattr(args, "scorer_input_distribution_guard_weight", 0.0)
    )
    scorer_input_guard_metadata = train_time_controls.metadata()[
        "scorer_input_distribution_guard"
    ]
    scorer_input_contrast_metadata = train_time_controls.metadata()[
        "scorer_input_contrast_floor"
    ]
    scorer_input_shape_metadata = train_time_controls.metadata()[
        "scorer_input_shape_tether"
    ]
    posenet_temporal_signal_metadata = train_time_controls.metadata()[
        "posenet_temporal_signal_floor"
    ]
    output_head_bias_metadata = train_time_controls.metadata()[
        "output_head_target_bias_init"
    ]
    output_head_contrast_metadata = train_time_controls.metadata()[
        "output_head_target_contrast_init"
    ]

    if distillation_weight <= 0.0:
        blockers.append("hinerv_full_missing_segnet_distillation_loss")
    elif segnet_live_calibration_weight <= 0.0:
        blockers.append("hinerv_full_missing_segnet_student_live_calibration")
    if str(train_time_controls.segnet_distillation_objective) != "boundary_argmax_hinge":
        blockers.append("hinerv_full_missing_boundary_argmax_hinge_segnet_objective")
    if segnet_direct_live_weight <= 0.0:
        blockers.append("hinerv_full_missing_direct_live_segnet_distillation")
    if segnet_class_escape_weight <= 0.0:
        blockers.append("hinerv_full_missing_direct_live_class_escape_pressure")
    if pose_distillation_weight <= 0.0:
        blockers.append("hinerv_full_missing_posenet_distillation_loss")
    if bool(getattr(args, "allow_mock_scorer_teacher", False)):
        blockers.append("hinerv_full_real_scorer_teacher_blocked_by_mock_flag")
    if bool(getattr(args, "allow_segnet_only_research", False)):
        blockers.append("hinerv_full_segnet_only_research_not_production")
    if not eval_roundtrip_ste:
        blockers.append("hinerv_full_missing_eval_roundtrip_ste")
    if pose_preprocess != "pr95_yuv6":
        blockers.append("hinerv_full_missing_pr95_yuv6_differentiable_pose_path")
    if not pr95_curriculum:
        blockers.append("hinerv_full_missing_pr95_faithful_curriculum")
    if pr95_curriculum and not pr95_source_weight_amplification:
        blockers.append("hinerv_full_missing_pr95_source_weight_amplification")
    if epochs < CANONICAL_PR95_FULL_EPOCHS:
        blockers.append("hinerv_full_pr95_epoch_budget_below_29650")
    if pr95_total_epochs is not None and int(pr95_total_epochs) != CANONICAL_PR95_FULL_EPOCHS:
        blockers.append("hinerv_full_pr95_curriculum_total_epochs_not_canonical_29650")
    if not coder_qat:
        blockers.append("hinerv_full_missing_coder_aware_qat")
    if c1a_entropy_weight <= 0.0:
        blockers.append("hinerv_full_missing_c1a_entropy_control")
    if c1a_sigma <= 0.0:
        blockers.append("hinerv_full_invalid_c1a_sigma")
    if c1a_sample_size <= 0:
        blockers.append("hinerv_full_invalid_c1a_sample_size")
    if not train_time_dual_ascent_enabled:
        blockers.append("hinerv_full_missing_train_time_dual_ascent")
    if not hard_byte_ceiling_attached:
        blockers.append("hinerv_full_missing_train_time_hard_byte_ceiling")
    if not train_time_section_byte_metrics_enabled:
        blockers.append("hinerv_full_missing_train_time_section_byte_metrics")
    if require_measured_section_byte_control and hard_byte_ceiling_attached:
        if not measured_section_byte_control_attached:
            blockers.append("hinerv_full_missing_measured_train_time_section_byte_control")
        elif not measured_section_byte_control_active:
            blockers.append("hinerv_full_train_time_section_byte_control_not_active")
    if measured_section_byte_control_attached and not measured_section_byte_control_active:
        blockers.append("hinerv_full_train_time_section_byte_control_not_active")
    if not ema_archive_selection:
        blockers.append("hinerv_full_missing_ema_archive_selection")
    if not archive_parse_back_selection:
        blockers.append("hinerv_full_missing_archive_parse_back_selection")
    if scorer_input_guard_weight <= 0.0:
        blockers.append("hinerv_full_missing_scorer_input_distribution_guard")
    if float(train_time_controls.scorer_input_contrast_floor_weight) <= 0.0:
        blockers.append("hinerv_full_missing_scorer_input_contrast_floor")
    if float(train_time_controls.scorer_input_shape_tether_weight) <= 0.0:
        blockers.append("hinerv_full_missing_scorer_input_shape_tether")
    if float(train_time_controls.posenet_temporal_signal_floor_weight) <= 0.0:
        blockers.append("hinerv_full_missing_posenet_temporal_signal_floor")
    if not bool(output_head_bias_metadata["enabled"]):
        blockers.append("hinerv_full_missing_output_head_target_bias_init")
    if not bool(output_head_contrast_metadata["enabled"]):
        blockers.append("hinerv_full_missing_output_head_target_contrast_init")
    blockers = list(dict.fromkeys(blockers))

    return {
        "schema": PR95_FULL_CONTROL_CONTRACT_SCHEMA,
        "family": "hi_nerv",
        "control_surface": "production_full_pr95_critical_controls",
        "production_full_control_ready": not blockers,
        "controls": {
            "stage_loss_schedule": train_time_controls.stage_loss_schedule,
            "optimizer_kind": train_time_controls.optimizer_kind,
            "optimizer_surface": train_time_controls.metadata()["optimizer_surface"],
            "real_segnet_distillation_loss": distillation_weight > 0.0,
            "segnet_student_live_calibration_weight": segnet_live_calibration_weight,
            "segnet_student_live_calibration_active": bool(
                distillation_weight > 0.0 and segnet_live_calibration_weight > 0.0
            ),
            "segnet_distillation_objective": (
                train_time_controls.segnet_distillation_objective
            ),
            "segnet_direct_live_distillation_weight": segnet_direct_live_weight,
            "segnet_direct_live_base_loss_weight": float(
                train_time_controls.segnet_direct_live_base_loss_weight
            ),
            "segnet_direct_live_class_escape_weight": segnet_class_escape_weight,
            "segnet_direct_live": train_time_controls.metadata()["segnet_direct_live"],
            "real_posenet_distillation_loss": pose_distillation_weight > 0.0,
            "mock_scorer_teacher_allowed": bool(getattr(args, "allow_mock_scorer_teacher", False)),
            "segnet_only_research_allowed": bool(getattr(args, "allow_segnet_only_research", False)),
            "eval_roundtrip_ste_enabled": eval_roundtrip_ste,
            "pose_student_input_preprocess": pose_preprocess,
            "pr95_faithful_curriculum_enabled": pr95_curriculum,
            "pr95_stage_source_weight_amplification_enabled": (
                pr95_source_weight_amplification
            ),
            "epochs": epochs,
            "canonical_pr95_full_epochs": CANONICAL_PR95_FULL_EPOCHS,
            "pr95_curriculum_total_epochs": pr95_total_epochs,
            "optimizer_warmup_steps_per_epoch": int(
                getattr(args, "optimizer_warmup_steps_per_epoch", 1)
            ),
            "pair_sampling_weight_count": len(
                list(getattr(args, "pair_sampling_weight", []) or [])
            ),
            "gradient_multiplier_count": len(
                list(getattr(args, "gradient_multiplier", []) or [])
            ),
            "bias_gradient_multiplier": getattr(args, "bias_gradient_multiplier", None),
            "output_head_bias_gradient_multiplier": float(
                getattr(args, "output_head_bias_gradient_multiplier", 1.0)
            ),
            "coder_qat_enabled": coder_qat,
            "coder_qat_c1a_entropy_weight": c1a_entropy_weight,
            "coder_qat_c1a_sigma": c1a_sigma,
            "coder_qat_c1a_sample_size": c1a_sample_size,
            "train_time_dual_ascent_enabled": train_time_dual_ascent_enabled,
            "hard_byte_ceiling_attached": hard_byte_ceiling_attached,
            "train_time_section_byte_metrics_enabled": (
                train_time_section_byte_metrics_enabled
            ),
            "train_time_section_byte_control_measurement_phase": (
                section_byte_measurement_phase
            ),
            "train_time_section_byte_control_required_for_training": bool(
                require_measured_section_byte_control and hard_byte_ceiling_attached
            ),
            "measured_train_time_section_byte_control_attached": (
                measured_section_byte_control_attached
            ),
            "measured_train_time_section_byte_control_active": (
                measured_section_byte_control_active
            ),
            "measured_train_time_section_byte_controlled_section_count": (
                measured_section_byte_controlled_count
            ),
            "measured_train_time_section_byte_pending_section_count": (
                measured_section_pending_count
            ),
            "measured_train_time_section_byte_control_blockers": (
                measured_section_byte_control_blockers
            ),
            "measured_train_time_section_byte_control": _metadata_safe(
                train_time_section_byte_control
            ),
            "decoder_fake_quant_forward_enabled": bool(
                train_time_controls.decoder_fake_quant_forward_enabled
            ),
            "decoder_fake_quant_bits": int(train_time_controls.decoder_fake_quant_bits),
            "train_time_decoder_controls_enabled": (
                train_time_controls.train_time_decoder_controls_enabled
            ),
            "train_time_decoder_pruning_ratio": float(
                train_time_controls.train_time_decoder_pruning_ratio
            ),
            "train_time_decoder_quant_noise_bits": (
                None
                if train_time_controls.train_time_decoder_quant_noise_bits is None
                else int(train_time_controls.train_time_decoder_quant_noise_bits)
            ),
            "train_time_decoder_quant_noise_scale": float(
                train_time_controls.train_time_decoder_quant_noise_scale
            ),
            "export_decoder_pruning_ratio": float(
                train_time_controls.export_decoder_pruning_ratio
            ),
            "export_decoder_quant_noise_bits": (
                None
                if train_time_controls.export_decoder_quant_noise_bits is None
                else int(train_time_controls.export_decoder_quant_noise_bits)
            ),
            "export_decoder_quant_noise_scale": float(
                train_time_controls.export_decoder_quant_noise_scale
            ),
            "ema_archive_selection_enabled": ema_archive_selection,
            "archive_parse_back_selection_enabled": archive_parse_back_selection,
            "scorer_input_distribution_guard_enabled": (
                scorer_input_guard_weight > 0.0
            ),
            "scorer_input_distribution_guard_weight": scorer_input_guard_weight,
            "scorer_input_distribution_guard_components": (
                scorer_input_guard_metadata["components"]
            ),
            "dynamic_range_repair_before_replay": bool(
                scorer_input_guard_metadata["dynamic_range_repair_before_replay"]
            ),
            "scorer_input_distribution_guard_saturation_margin": float(
                getattr(
                    args,
                    "scorer_input_distribution_guard_saturation_margin",
                    0.02,
                )
            ),
            "scorer_input_distribution_guard_temperature": float(
                getattr(args, "scorer_input_distribution_guard_temperature", 0.01)
            ),
            "scorer_input_contrast_floor_enabled": bool(
                scorer_input_contrast_metadata["enabled"]
            ),
            "scorer_input_contrast_floor_weight": float(
                train_time_controls.scorer_input_contrast_floor_weight
            ),
            "scorer_input_contrast_floor_segnet_min_std_ratio": float(
                train_time_controls.scorer_input_contrast_floor_segnet_min_std_ratio
            ),
            "scorer_input_contrast_floor_posenet_yuv6_min_std_ratio": float(
                train_time_controls.scorer_input_contrast_floor_posenet_yuv6_min_std_ratio
            ),
            "scorer_input_shape_tether_enabled": bool(
                scorer_input_shape_metadata["enabled"]
            ),
            "scorer_input_shape_tether_weight": float(
                train_time_controls.scorer_input_shape_tether_weight
            ),
            "scorer_input_shape_tether_components": (
                scorer_input_shape_metadata["components"]
            ),
            "posenet_temporal_signal_floor_enabled": bool(
                posenet_temporal_signal_metadata["enabled"]
            ),
            "posenet_temporal_signal_floor_weight": float(
                train_time_controls.posenet_temporal_signal_floor_weight
            ),
            "posenet_temporal_signal_min_std_ratio": float(
                train_time_controls.posenet_temporal_signal_min_std_ratio
            ),
            "posenet_temporal_signal_min_mean_abs_ratio": float(
                train_time_controls.posenet_temporal_signal_min_mean_abs_ratio
            ),
            "posenet_temporal_signal_floor": (
                posenet_temporal_signal_metadata
            ),
            "output_head_target_bias_init_enabled": bool(
                output_head_bias_metadata["enabled"]
            ),
            "output_head_target_bias_init_epsilon": float(
                output_head_bias_metadata["epsilon"]
            ),
            "output_head_target_contrast_init_enabled": bool(
                output_head_contrast_metadata["enabled"]
            ),
            "output_head_target_contrast_init_max_pairs": int(
                output_head_contrast_metadata["max_pairs"]
            ),
            "output_head_target_contrast_init_min_output_std": float(
                output_head_contrast_metadata["min_output_std"]
            ),
            "output_head_target_contrast_init_max_gain": float(
                output_head_contrast_metadata["max_gain"]
            ),
        },
        "train_time_controls": _metadata_safe(train_time_controls.metadata()),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _direct_trainer_launch_refusal_payload(
    canonicalization: dict[str, Any],
    *,
    mode: str,
    pr95_full_control_contract: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if str(mode) == "smoke":
        return None
    control_ready = True
    if pr95_full_control_contract is not None:
        control_ready = pr95_full_control_contract.get("production_full_control_ready") is True
    if canonicalization.get("trainer_launch_allowed") is True and control_ready:
        return None
    blockers: list[str] = []
    if canonicalization.get("trainer_launch_allowed") is not True:
        blockers.extend(
            [
                "hinerv_direct_full_trainer_launch_blocked_by_canonicalization_contract",
                *[str(blocker) for blocker in canonicalization.get("blockers") or []],
            ]
        )
    if pr95_full_control_contract is not None:
        control_blockers = [str(blocker) for blocker in pr95_full_control_contract.get("blockers") or []]
        if control_blockers:
            blockers.extend(
                [
                    "hinerv_full_trainer_launch_blocked_by_pr95_control_contract",
                    *control_blockers,
                ]
            )
    return {
        "schema": DIRECT_TRAINER_LAUNCH_REFUSAL_SCHEMA,
        "authority": TRAINER_AUTHORITY,
        "mode": str(mode),
        "training_executed": False,
        "export_executed": False,
        "trainer_launch_allowed": False,
        "launch_refusal_reason": (
            "HiNeRV direct --full trainer launch is blocked by its "
            "canonicalization contract; use the compact runner for production "
            "launch custody, or --smoke for explicit unscored research smoke."
        ),
        "canonical_runner_entrypoint": DIRECT_TRAINER_CANONICAL_RUNNER_ENTRYPOINT,
        "allowed_direct_research_mode": "--smoke",
        "direct_trainer_canonicalization": _metadata_safe(canonicalization),
        "pr95_full_control_contract": _metadata_safe(pr95_full_control_contract),
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _config_snapshot(cfg: Any) -> dict[str, Any]:
    return {
        "latent_dim_coarse": int(cfg.latent_dim_coarse),
        "latent_dim_mid": int(cfg.latent_dim_mid),
        "latent_dim_fine": int(cfg.latent_dim_fine),
        "embed_dim": int(cfg.embed_dim),
        "decoder_channels": [int(v) for v in cfg.decoder_channels],
        "sin_frequency": float(cfg.sin_frequency),
        "num_upsample_blocks": int(cfg.num_upsample_blocks),
        "num_pairs": int(cfg.num_pairs),
        "output_height": int(cfg.output_height),
        "output_width": int(cfg.output_width),
    }


def _looks_local(path: Path) -> bool:
    return not path.resolve(strict=False).as_posix().startswith("/Volumes/")


_METADATA_FORBIDDEN_AUTHORITY_KEYS = frozenset(
    {
        "score_claim",
        "promotion_eligible",
        "ready_for_exact_eval_dispatch",
        "rank_or_kill_eligible",
        "promotable",
        "score_claim_valid",
    }
)


def _metadata_safe(value: Any) -> Any:
    """Drop nested authority keys before passing data into RendererBundle metadata."""

    if isinstance(value, dict):
        return {
            str(key): _metadata_safe(child)
            for key, child in value.items()
            if str(key) not in _METADATA_FORBIDDEN_AUTHORITY_KEYS
        }
    if isinstance(value, list):
        return [_metadata_safe(child) for child in value]
    return value


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _finite_mapping_value(mapping: Mapping[str, Any] | None, key: str) -> float | None:
    if not isinstance(mapping, Mapping) or key not in mapping:
        return None
    return _finite_float(mapping.get(key))


def _final_loss_components_from_training_artifact(artifact: Any) -> dict[str, float]:
    per_epoch_metrics = getattr(artifact, "per_epoch_metrics", None)
    if not per_epoch_metrics:
        return {}
    final_metrics = per_epoch_metrics[-1]
    loss_components = getattr(final_metrics, "loss_components", None)
    if not isinstance(loss_components, Mapping):
        return {}
    out: dict[str, float] = {}
    for key, value in loss_components.items():
        finite = _finite_float(value)
        if finite is not None:
            out[str(key)] = finite
    return out


def _build_hinerv_short_scorer_smoke_readiness_report(
    *,
    train_time_controls: HiNervTrainTimeControlConfig,
    final_loss_components: Mapping[str, Any] | None,
    post_export_quality: dict[str, Any] | None,
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
    allow_mock_scorer_teacher: bool,
    min_segnet_occupied_class_fraction_for_fit_gate: float = (
        HI_NERV_SHORT_SCORER_SMOKE_DEFAULT_MIN_SEGNET_OCCUPIED_CLASS_FRACTION
    ),
) -> dict[str, Any]:
    return _shared_build_readiness(
        train_time_controls=train_time_controls,
        final_loss_components=final_loss_components,
        post_export_quality=post_export_quality,
        segnet_distillation_weight=segnet_distillation_weight,
        pose_distillation_weight=pose_distillation_weight,
        allow_mock_scorer_teacher=allow_mock_scorer_teacher,
        min_segnet_occupied_class_fraction_for_fit_gate=(
            min_segnet_occupied_class_fraction_for_fit_gate
        ),
    )


def _write_hinerv_short_scorer_smoke_readiness(
    *,
    output_dir: Path,
    train_time_controls: HiNervTrainTimeControlConfig,
    final_loss_components: Mapping[str, Any] | None,
    post_export_quality: dict[str, Any] | None,
    segnet_distillation_weight: float,
    pose_distillation_weight: float,
    allow_mock_scorer_teacher: bool,
    min_segnet_occupied_class_fraction_for_fit_gate: float,
) -> dict[str, Any]:
    report = _build_hinerv_short_scorer_smoke_readiness_report(
        train_time_controls=train_time_controls,
        final_loss_components=final_loss_components,
        post_export_quality=post_export_quality,
        segnet_distillation_weight=segnet_distillation_weight,
        pose_distillation_weight=pose_distillation_weight,
        allow_mock_scorer_teacher=allow_mock_scorer_teacher,
        min_segnet_occupied_class_fraction_for_fit_gate=(
            min_segnet_occupied_class_fraction_for_fit_gate
        ),
    )
    path = output_dir / "hi_nerv_short_scorer_smoke_readiness.json"
    report["report_path"] = path.as_posix()
    write_json(path, report)
    return report


def _hinerv_short_scorer_smoke_readiness_summary(
    report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    return _shared_readiness_summary(report)


def _attach_hinerv_short_scorer_smoke_readiness_to_training_artifact(
    *,
    output_dir: Path,
    report: dict[str, Any],
) -> None:
    artifact_path = output_dir / "training_artifact.json"
    if not artifact_path.is_file():
        return
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    metadata = dict(artifact.get("substrate_artifact_metadata") or {})
    metadata["short_scorer_teacher_smoke_readiness"] = (
        _hinerv_short_scorer_smoke_readiness_summary(report)
    )
    artifact["substrate_artifact_metadata"] = metadata
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _maybe_write_post_export_receiver_cache_quality(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    archive_path: str | Path | None,
) -> dict[str, Any] | None:
    if not bool(args.post_export_receiver_cache_quality_gate):
        return None
    if archive_path is None:
        return _write_post_export_receiver_cache_quality_refusal(
            output_dir=output_dir,
            blockers=["hi_nerv_archive_export_missing_for_receiver_cache_quality"],
        )
    archive = Path(archive_path).expanduser().resolve(strict=False)
    reference = args.receiver_cache_quality_reference_cache_dir.expanduser()
    if not reference.is_absolute():
        reference = (REPO_ROOT / reference).resolve(strict=False)
    if not archive.is_file():
        return _write_post_export_receiver_cache_quality_refusal(
            output_dir=output_dir,
            blockers=["hi_nerv_archive_export_path_missing_for_receiver_cache_quality"],
            archive_path=archive,
            reference_cache_dir=reference,
        )
    if not reference.is_dir():
        return _write_post_export_receiver_cache_quality_refusal(
            output_dir=output_dir,
            blockers=["hi_nerv_reference_cache_missing_for_receiver_cache_quality"],
            archive_path=archive,
            reference_cache_dir=reference,
        )
    from tac.substrates.hi_nerv.receiver_cache_quality import (
        write_hi_nerv_receiver_cache_quality_report,
    )

    return write_hi_nerv_receiver_cache_quality_report(
        archive_zip_path=archive,
        output_dir=output_dir / "post_export_receiver_cache_quality",
        reference_cache_dir=reference,
        max_pairs=int(args.receiver_cache_quality_max_pairs),
        batch_pairs=int(args.receiver_cache_quality_batch_pairs),
        sample_pairs=int(args.receiver_cache_quality_max_pairs),
        min_segnet_std=float(args.receiver_cache_quality_min_segnet_std),
        min_segnet_dynamic_range=float(args.receiver_cache_quality_min_segnet_dynamic_range),
        max_segnet_mae_vs_reference_for_fit_gate=float(
            args.receiver_cache_quality_max_segnet_mae_vs_reference_for_fit_gate
        ),
        min_posenet_yuv6_std=float(
            args.receiver_cache_quality_min_posenet_yuv6_std
        ),
        min_posenet_yuv6_dynamic_range=float(
            args.receiver_cache_quality_min_posenet_yuv6_dynamic_range
        ),
        max_posenet_yuv6_mae_vs_reference_for_fit_gate=float(
            args.receiver_cache_quality_max_posenet_yuv6_mae_vs_reference_for_fit_gate
        ),
        min_posenet_yuv6_temporal_signal_std_for_fit_gate=float(
            args.receiver_cache_quality_min_posenet_yuv6_temporal_signal_std_for_fit_gate
        ),
        min_posenet_yuv6_temporal_signal_mean_abs_for_fit_gate=float(
            args.receiver_cache_quality_min_posenet_yuv6_temporal_signal_mean_abs_for_fit_gate
        ),
        min_segnet_argmax_occupied_class_fraction_for_fit_gate=float(
            args.receiver_cache_quality_min_segnet_argmax_occupied_class_fraction_for_fit_gate
        ),
        segnet_argmax_probe_upstream_dir=args.upstream_dir,
        require_mlx_scorer_response_probe=bool(
            args.receiver_cache_quality_mlx_scorer_response_probe
        ),
        mlx_scorer_response_upstream_dir=args.upstream_dir,
        mlx_scorer_response_device_type=str(
            args.receiver_cache_quality_mlx_scorer_response_device_type
        ),
        mlx_scorer_response_batch_pairs=int(
            args.receiver_cache_quality_mlx_scorer_response_batch_pairs
        ),
        max_mlx_scorer_response_posenet_dist_for_fit_gate=float(
            args.receiver_cache_quality_max_mlx_scorer_response_posenet_dist_for_fit_gate
        ),
        max_mlx_scorer_response_segnet_dist_for_fit_gate=float(
            args.receiver_cache_quality_max_mlx_scorer_response_segnet_dist_for_fit_gate
        ),
    )


def _write_post_export_receiver_cache_quality_refusal(
    *,
    output_dir: Path,
    blockers: list[str],
    archive_path: Path | None = None,
    reference_cache_dir: Path | None = None,
) -> dict[str, Any]:
    report = {
        "schema": "hi_nerv_receiver_cache_quality_report.v1",
        "output_dir": (output_dir / "post_export_receiver_cache_quality").as_posix(),
        "archive_path": archive_path.as_posix() if archive_path is not None else None,
        "reference_cache_dir": (reference_cache_dir.as_posix() if reference_cache_dir is not None else None),
        "quality_gate": None,
        "quality_gate_passed": False,
        "blockers": [
            "hi_nerv_receiver_cache_quality_is_false_authority",
            *[str(blocker) for blocker in blockers],
        ],
        **FALSE_AUTHORITY,
    }
    out = output_dir / "post_export_receiver_cache_quality"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "hi_nerv_receiver_cache_quality_report.json"
    report["report_path"] = path.as_posix()
    write_json(path, report)
    return report


def _attach_post_export_receiver_cache_quality_to_training_artifact(
    *,
    output_dir: Path,
    report: dict[str, Any],
) -> None:
    artifact_path = output_dir / "training_artifact.json"
    if not artifact_path.is_file():
        return
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    metadata = dict(artifact.get("substrate_artifact_metadata") or {})
    metadata["post_export_receiver_cache_quality"] = (
        _receiver_cache_quality_manifest_summary(report)
    )
    artifact["substrate_artifact_metadata"] = metadata
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _receiver_cache_quality_manifest_summary(
    report: dict[str, Any] | None,
) -> dict[str, Any] | None:
    return _shared_receiver_quality_summary(report)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.full:
        return _full_main(args)
    return _smoke_main(args)


__all__ = [
    "HI_NERV_MODELSIZE_CANDIDATE_CONSUMPTION_SCHEMA",
    "HI_NERV_SHORT_SCORER_SMOKE_DEFAULT_MIN_SEGNET_OCCUPIED_CLASS_FRACTION",
    "HI_NERV_SHORT_SCORER_SMOKE_READINESS_SCHEMA",
    "HI_NERV_TRAIN_TIME_CONTROL_SCHEMA",
    "HI_NERV_TRAIN_TIME_DECODER_CONTROL_REPORT_SCHEMA",
    "PR95_FULL_CONTROL_CONTRACT_SCHEMA",
    "TRAINER_SCHEMA",
    "HiNervTrainTimeControlConfig",
    "_apply_train_time_decoder_controls",
    "_build_hinerv_hard_byte_ceiling_control",
    "_build_parser",
    "_build_train_time_decoder_control_callback",
    "_coder_qat_config_from_args",
    "_config_from_args",
    "_configure_decoder_fake_quant_forward",
    "_decoder_codec_from_args",
    "_decoder_weight_waterfill_plan_attachment_metadata",
    "_decoder_weight_waterfill_plan_from_args",
    "_hard_byte_ceiling_from_args",
    "_hard_byte_ceiling_from_modelsize_candidate",
    "_metadata_safe",
    "_modelsize_candidate_consumption_metadata",
    "_modelsize_candidate_from_args",
    "_pr95_full_control_contract",
    "_prioritized_pair_indices_from_args",
    "_prioritized_pair_training_lineage_metadata",
    "_prioritized_pair_training_metadata",
    "_resolve_output_dir",
    "_train_time_control_config_from_args",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
