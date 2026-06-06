# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the native MLX SNeRV export adapter."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier.archive import (
    SnervArchivePacket,
    decode_official_mfu_hfr_tub_decoder_payload,
    decode_snerv_archive_frames,
    pack_snerv_archive,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
    HfGenerationDecoder,
    SnervCarrierError,
    SnervModelSizeConfig,
    generate_hf_from_lf,
)
from tac.substrates.snerv_inverse_steg_carrier.dwt import (
    WaveletPyramid,
    dwt2_multilevel,
    idwt2_multilevel,
)
from tac.substrates.snerv_inverse_steg_carrier.inflate import _resize_nchw_bilinear
from tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export import (
    DEFAULT_SNERV_SCORER_INPUT_DISTRIBUTION_GUARD_WEIGHT,
    SNERV_DWT_ADJOINT_SALIENCY_WEIGHTED_FIT_MODE,
    SNERV_MLX_NATIVE_REPORT_FILENAME,
    SnervMlxNativeExportError,
    _build_snerv_mlx_native_byte_cap_control,
    _model_size_from_candidate,
    _snerv_official_skip_high_value_domain_gate,
    _snerv_receiver_frame_reconstruction_profile,
    _target_pairs_to_nchw255,
    build_snerv_mlx_native_packet_from_numpy_pairs,
    train_export_snerv_mlx_native,
    write_snerv_mlx_prefilter_profile,
)


def _tiny_pairs(*, pairs: int = 1) -> np.ndarray:
    yy, xx = np.mgrid[0:16, 0:16].astype(np.float32)
    out = np.zeros((pairs, 2, 3, 16, 16), dtype=np.float32)
    for pair_idx in range(pairs):
        for frame_idx in range(2):
            for channel_idx in range(3):
                out[pair_idx, frame_idx, channel_idx] = (
                    80.0
                    + 9.0 * channel_idx
                    + 5.0 * frame_idx
                    + pair_idx
                    + xx * (0.7 + 0.1 * channel_idx)
                    + yy * (0.4 + 0.1 * frame_idx)
                )
    return np.clip(out, 0.0, 255.0)


def _fake_tub_fixture_replay_passed() -> dict[str, object]:
    return {
        "schema": "snerv_official_tub_source_forward_replay.v1",
        "family": "snerv",
        "component_id": "tub",
        "source_forward_replay_executed": True,
        "official_tub_temporal_encoder_output2_source_fixture_replay_passed": True,
        "source_fixture_scope": "deterministic_official_source_fixture_not_trained_checkpoint",
        "closed_blockers": [
            "snerv_official_tub_graph_inputs_only_not_full_source_forward_parity",
            "snerv_official_snerv_t_output2_fusion_source_forward_replay_missing",
            "snerv_official_tub_portable_output2_fusion_receiver_mapping_missing",
        ],
        "preserved_blockers": [
            "snerv_official_trained_checkpoint_state_dict_not_loaded",
            "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
            "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
            "snerv_official_tub_portable_output2_decoder_weight_mapping_missing",
            "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing",
        ],
        "blockers": [
            "snerv_official_trained_checkpoint_state_dict_not_loaded",
            "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
            "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
            "snerv_official_tub_portable_output2_decoder_weight_mapping_missing",
            "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing",
        ],
        "score_claim": False,
        "frontier_score_claim": False,
        "rank_or_kill_eligible": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _minimal_full_official_decoder_state(decoder_len: int = 8) -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {}
    for offset in range(3):
        prefix = f"decoder.{decoder_len + offset}"
        state[f"{prefix}.conv1.weight"] = np.zeros((3, 3, 1, 1), dtype=np.float32)
        state[f"{prefix}.conv1.bias"] = np.zeros((3,), dtype=np.float32)
        state[f"{prefix}.conv2.weight"] = np.zeros((3, 3, 3, 3), dtype=np.float32)
        state[f"{prefix}.conv2.bias"] = np.zeros((3,), dtype=np.float32)
    for offset in (3, 5):
        prefix = f"decoder.{decoder_len + offset}"
        state[f"{prefix}.weight"] = np.zeros((3, 3, 2, 2), dtype=np.float32)
        state[f"{prefix}.bias"] = np.zeros((3,), dtype=np.float32)
    for offset in (4, 6):
        prefix = f"decoder.{decoder_len + offset}"
        state[f"{prefix}.main.0.weight"] = np.zeros((3, 3, 3, 3), dtype=np.float32)
        state[f"{prefix}.main.0.bias"] = np.zeros((3,), dtype=np.float32)
        state[f"{prefix}.main.1.0.conv1.weight"] = np.zeros(
            (3, 3, 3, 3),
            dtype=np.float32,
        )
        state[f"{prefix}.main.1.0.conv1.bias"] = np.zeros((3,), dtype=np.float32)
        state[f"{prefix}.main.1.0.conv2.weight"] = np.zeros(
            (3, 3, 3, 3),
            dtype=np.float32,
        )
        state[f"{prefix}.main.1.0.conv2.bias"] = np.zeros((3,), dtype=np.float32)
    state["encoder.1.weight"] = np.zeros((3, 3, 3), dtype=np.float32)
    state["encoder.2.weight"] = np.zeros((3, 3, 3), dtype=np.float32)
    state[f"decoder.{decoder_len - 1}.weight"] = np.zeros((3, 3, 1, 1), dtype=np.float32)
    return state


def test_pr95_muon_policy_is_bound_to_native_train_export_surfaces() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    public_sig = inspect.signature(train_export_snerv_mlx_native)
    assert "official_trained_checkpoint_state_dict" in public_sig.parameters
    assert "official_trained_checkpoint_state_dict_path" in public_sig.parameters
    assert "official_trained_checkpoint_decoder_len" in public_sig.parameters
    assert "official_trained_checkpoint_state_dict_kind" in public_sig.parameters
    assert "score_aware_long_training_pr95_muon_policy" in public_sig.parameters
    assert (
        public_sig.parameters[
            "score_aware_long_training_pr95_faithful_curriculum"
        ].default
        is True
    )
    assert (
        public_sig.parameters["score_aware_long_training_pr95_muon_policy"].default
        == "faithful_stage8_only"
    )
    assert (
        "score_aware_long_training_pr95_source_weight_amplification"
        in public_sig.parameters
    )
    assert (
        "score_aware_long_training_scorer_input_distribution_guard_weight"
        in public_sig.parameters
    )
    assert (
        "score_aware_long_training_scorer_input_contrast_floor_weight"
        in public_sig.parameters
    )
    assert (
        "score_aware_long_training_scorer_input_shape_tether_weight"
        in public_sig.parameters
    )
    assert (
        "score_aware_long_training_posenet_temporal_signal_floor_weight"
        in public_sig.parameters
    )
    assert (
        "score_aware_long_training_posenet_temporal_signal_min_std_ratio"
        in public_sig.parameters
    )
    assert (
        "score_aware_long_training_posenet_temporal_signal_min_mean_abs_ratio"
        in public_sig.parameters
    )
    assert (
        "score_aware_long_training_scorer_space_step_guard_enabled"
        in public_sig.parameters
    )
    assert (
        "score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_coverage_fraction"
        in public_sig.parameters
    )
    assert (
        "score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_min_ratio"
        in public_sig.parameters
    )
    assert (
        "score_aware_long_training_scorer_space_step_guard_max_post_segnet_target_class_ratio_drop"
        in public_sig.parameters
    )
    assert (
        "score_aware_long_training_scorer_space_step_guard_backtracking_steps"
        in public_sig.parameters
    )
    assert "score_aware_long_training_loss_weights" in public_sig.parameters
    assert (
        "score_aware_long_training_gradient_multiplier_by_name"
        in public_sig.parameters
    )
    assert (
        "score_aware_long_training_bias_gradient_multiplier"
        in public_sig.parameters
    )
    assert (
        "score_aware_long_training_output_head_bias_gradient_multiplier"
        in public_sig.parameters
    )
    assert "score_aware_long_training_pose_warmup_epochs" in public_sig.parameters
    assert (
        "score_aware_long_training_scorer_input_shape_warmup_epochs"
        in public_sig.parameters
    )
    assert (
        "score_aware_long_training_segnet_direct_live_escape_warmup_epochs"
        in public_sig.parameters
    )
    assert (
        "score_aware_long_training_scorer_input_contrast_floor_segnet_min_std_ratio"
        in public_sig.parameters
    )
    assert (
        "score_aware_long_training_scorer_input_contrast_floor_posenet_yuv6_min_std_ratio"
        in public_sig.parameters
    )
    assert (
        public_sig.parameters[
            "score_aware_long_training_pr95_muon_policy"
        ].default
        == "faithful_stage8_only"
    )
    assert (
        public_sig.parameters[
            "score_aware_long_training_scorer_input_distribution_guard_weight"
        ].default
        == DEFAULT_SNERV_SCORER_INPUT_DISTRIBUTION_GUARD_WEIGHT
    )
    assert (
        public_sig.parameters[
            "score_aware_long_training_scorer_space_step_guard_enabled"
        ].default
        is True
    )
    assert (
        public_sig.parameters[
            "score_aware_long_training_eval_roundtrip_ste"
        ].default
        is True
    )
    assert (
        public_sig.parameters["scorer_loop_qat_component_guard_mode"].default
        == "pose_seg_hard"
    )
    assert (
        public_sig.parameters[
            "scorer_loop_qat_pair_guard_min_score_improved_fraction"
        ].default
        == 1.0
    )
    assert (
        public_sig.parameters[
            "scorer_loop_qat_pair_guard_max_pose_worsened_fraction"
        ].default
        == 0.0
    )
    assert public_sig.parameters["scorer_loop_qat_perturb_scale"].default == 0.02
    assert (
        public_sig.parameters["scorer_loop_qat_byte_pressure_multiplier"].default
        == 1.0
    )
    assert (
        public_sig.parameters[
            "scorer_loop_qat_section_value_pressure_multiplier"
        ].default
        == 1.0
    )
    assert public_sig.parameters["scorer_loop_qat_max_archive_byte_growth"].default is None
    assert (
        public_sig.parameters["scorer_loop_qat_byte_growth_admission_mode"].default
        == "hard_cap"
    )
    assert public_sig.parameters["scorer_loop_qat_pose_slack"].default == 0.0
    assert public_sig.parameters["scorer_loop_qat_seg_slack"].default == 0.0
    assert public_sig.parameters["scorer_loop_qat_seed"].default == 1337
    attachment_sig = inspect.signature(mod._run_score_aware_long_training_attachment)
    assert attachment_sig.parameters["eval_roundtrip_ste"].default is inspect.Parameter.empty
    assert "pr95_muon_policy" in attachment_sig.parameters
    assert "scorer_input_distribution_guard_weight" in attachment_sig.parameters
    assert "scorer_input_contrast_floor_weight" in attachment_sig.parameters
    assert "scorer_input_shape_tether_weight" in attachment_sig.parameters
    assert "posenet_temporal_signal_floor_weight" in attachment_sig.parameters
    assert "posenet_temporal_signal_min_std_ratio" in attachment_sig.parameters
    assert (
        "posenet_temporal_signal_min_mean_abs_ratio"
        in attachment_sig.parameters
    )
    assert (
        "scorer_input_contrast_floor_segnet_min_std_ratio"
        in attachment_sig.parameters
    )
    assert (
        "scorer_input_contrast_floor_posenet_yuv6_min_std_ratio"
        in attachment_sig.parameters
    )
    assert "gradient_multiplier_by_name" in attachment_sig.parameters
    assert "bias_gradient_multiplier" in attachment_sig.parameters
    assert "output_head_bias_gradient_multiplier" in attachment_sig.parameters

    source = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    attachment_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_run_score_aware_long_training_attachment"
    ]
    assert attachment_calls
    assert any(
        "pr95_muon_policy" in {kw.arg for kw in call.keywords if kw.arg}
        for call in attachment_calls
    )
    assert any(
        "scorer_input_distribution_guard_weight"
        in {kw.arg for kw in call.keywords if kw.arg}
        for call in attachment_calls
    )
    assert any(
        "scorer_input_contrast_floor_weight"
        in {kw.arg for kw in call.keywords if kw.arg}
        for call in attachment_calls
    )
    assert any(
        "scorer_input_shape_tether_weight"
        in {kw.arg for kw in call.keywords if kw.arg}
        for call in attachment_calls
    )
    assert any(
        "posenet_temporal_signal_floor_weight"
        in {kw.arg for kw in call.keywords if kw.arg}
        for call in attachment_calls
    )
    assert any(
        "gradient_multiplier_by_name" in {kw.arg for kw in call.keywords if kw.arg}
        for call in attachment_calls
    )
    assert any(
        "bias_gradient_multiplier" in {kw.arg for kw in call.keywords if kw.arg}
        for call in attachment_calls
    )
    assert any(
        "output_head_bias_gradient_multiplier"
        in {kw.arg for kw in call.keywords if kw.arg}
        for call in attachment_calls
    )
    harness_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_mlx_score_aware_full_main"
    ]
    assert harness_calls
    assert any(
        "pr95_muon_policy" in {kw.arg for kw in call.keywords if kw.arg}
        for call in harness_calls
    )
    assert any(
        "gradient_multiplier_by_name" in {kw.arg for kw in call.keywords if kw.arg}
        for call in harness_calls
    )
    assert any(
        "bias_gradient_multiplier" in {kw.arg for kw in call.keywords if kw.arg}
        for call in harness_calls
    )
    assert any(
        "output_head_bias_gradient_multiplier"
        in {kw.arg for kw in call.keywords if kw.arg}
        for call in harness_calls
    )


def test_official_checkpoint_npz_ingestion_reaches_train_export_binding(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    state_path = tmp_path / "official_state_dict_slice.npz"
    np.savez(state_path, **_minimal_full_official_decoder_state())
    manifest = mod._official_trained_checkpoint_mapping_manifest_from_inputs(
        state_dict=None,
        state_dict_path=state_path,
        decoder_len=None,
        state_dict_kind="unit_test_npz_official_checkpoint",
    )

    assert manifest["official_trained_checkpoint_loaded"] is True
    assert manifest["official_mfu_hfr_trained_checkpoint_weight_mapping_proven"] is True
    assert manifest["official_tub_temporal_encoder_weight_mapping_proven"] is True
    assert manifest["official_tub_output2_decoder_weight_mapping_proven"] is True
    assert mod._official_checkpoint_full_mapping_verified(manifest) is True

    report = mod._run_score_aware_long_training_attachment(
        requested_epochs=0,
        output_dir=tmp_path / "long_training_attachment",
        pairs_nchw255=_tiny_pairs(pairs=1),
        model_size=SnervModelSizeConfig(
            adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        ),
        levels=1,
        wavelet="haar",
        source_pair_indices=(0,),
        target_bits_per_coeff=8.0,
        step_map_bits_per_coeff=4.0,
        decoder_payload_codec="npz",
        lf_payload_codec="raw",
        recon_pixel_weight=None,
        recon_pixel_weight_metadata=None,
        hf_decoder_saliency_gain=0.0,
        hard_byte_ceiling=None,
        learning_rate=1.0e-3,
        batch_pairs=1,
        section_byte_refresh_every_steps=1,
        optimizer_kind="pact_muon_adamw",
        grad_clip_max_norm=1.0,
        weight_decay=1.0e-4,
        eval_roundtrip_ste=False,
        scorer_input_distribution_guard_weight=0.0,
        scorer_input_distribution_guard_saturation_margin=0.02,
        scorer_input_distribution_guard_temperature=0.01,
        scorer_input_contrast_floor_weight=0.0,
        scorer_input_contrast_floor_segnet_min_std_ratio=0.5,
        scorer_input_contrast_floor_posenet_yuv6_min_std_ratio=0.5,
        checkpoint_retention_keep_last_n=1,
        checkpoint_retention_keep_best_n=1,
        checkpoint_retention_keep_every_n_epochs=None,
        checkpoint_retention_cold_store_roots=(),
        scorer_upstream_dir=tmp_path / "missing_upstream",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        pose_distillation_loss="mse",
        pose_distillation_huber_delta=1.0,
        segnet_distillation_objective="kl_t2",
        distillation_temperature=2.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.0,
        segnet_tau_boundary=1.0,
        segnet_hinge_margin=1.0,
        distillation_device="cpu",
        allow_segnet_only_research=False,
        coder_aware_qat=False,
        coder_qat_quant_bits=8,
        coder_qat_quant_residual_weight=0.0,
        coder_qat_magnitude_weight=0.0,
        coder_qat_delta_weight=0.0,
        coder_qat_c1a_entropy_weight=0.0,
        coder_qat_c1a_sigma=0.2,
        coder_qat_c1a_sample_size=4,
        pr95_faithful_curriculum_enabled=False,
        pr95_muon_policy="every_stage",
        official_trained_checkpoint_mapping_manifest=manifest,
        prioritized_pair_indices=(),
        scorer_error_pair_sampling_weights=None,
        scorer_error_pair_curriculum=None,
        allow_overwrite=True,
    )

    train_export = report["official_mfu_hfr_tub_train_export"]
    assert train_export["official_trained_checkpoint_state_dict_loaded"] is True
    assert (
        train_export["official_trained_checkpoint_state_dict_mapping_verified"]
        is True
    )
    assert train_export["official_trained_checkpoint_source_forward_replay_verified"] is False
    assert train_export["official_trained_checkpoint_mapping_manifest"][
        "state_dict_source"
    ] == state_path.as_posix()
    assert train_export["source_forward_replay_authority"] is False
    assert report["score_claim"] is False


def test_official_checkpoint_full_mapping_requires_tub_output2_decoder(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    decoder_len = 8
    state_without_output2 = {
        key: value
        for key, value in _minimal_full_official_decoder_state(
            decoder_len=decoder_len,
        ).items()
        if not key.startswith(f"decoder.{decoder_len - 1}.")
    }
    state_path = tmp_path / "official_state_dict_without_output2.npz"
    np.savez(state_path, **state_without_output2)

    manifest = mod._official_trained_checkpoint_mapping_manifest_from_inputs(
        state_dict=None,
        state_dict_path=state_path,
        decoder_len=decoder_len,
        state_dict_kind="unit_test_missing_output2_official_checkpoint",
    )

    assert manifest["official_trained_checkpoint_loaded"] is True
    assert manifest["official_mfu_hfr_trained_checkpoint_weight_mapping_proven"] is True
    assert manifest["official_tub_temporal_encoder_weight_mapping_proven"] is True
    assert manifest["official_tub_output2_decoder_weight_mapping_proven"] is False
    assert mod._official_checkpoint_full_mapping_verified(manifest) is False


def test_train_export_threads_official_checkpoint_npz_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    state_path = tmp_path / "official_state_dict_slice.npz"
    np.savez(state_path, **_minimal_full_official_decoder_state())
    source_pairs = _tiny_pairs(pairs=1) / 255.0

    def fake_decode_mlx_targets(*_args: object, **_kwargs: object) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.transpose(source_pairs[:, 0], (0, 2, 3, 1)),
            np.transpose(source_pairs[:, 1], (0, 2, 3, 1)),
        )

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "native_train_export",
        num_pairs=1,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "candidate_id": "official-checkpoint-npz-path",
            "snerv_model_size_adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "decoder_payload_codec": "int8_symmetric",
            "snerv_fc_dim": 9,
            "score_aware_long_training_epochs": 0,
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
        official_trained_checkpoint_state_dict_path=state_path,
    )

    long_training = report["score_aware_long_training"]
    train_export = long_training["official_mfu_hfr_tub_train_export"]
    assert train_export["official_trained_checkpoint_state_dict_loaded"] is True
    assert (
        train_export["official_trained_checkpoint_state_dict_mapping_verified"]
        is True
    )
    assert train_export["official_trained_checkpoint_mapping_manifest"][
        "state_dict_source"
    ] == state_path.as_posix()
    assert train_export["official_trained_checkpoint_source_forward_replay_verified"] is False
    assert report["score_claim"] is False


def test_train_export_refuses_dead_official_checkpoint_control_without_official_adapter(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        SnervMlxNativeExportError,
        match="official trained checkpoint controls require",
    ):
        train_export_snerv_mlx_native(
            output_dir=tmp_path / "native_train_export",
            num_pairs=1,
            source_video_path="unit.mkv",
            modelsize_candidate={
                "candidate_id": "non-official-dead-checkpoint-control",
                "snerv_model_size_adapter": "snerv_fc_dim_emb_size_adapter_v1",
                "levels": 1,
                "wavelet": "haar",
                "bits_per_coeff": 3.0,
                "decoder_payload_codec": "int8_symmetric",
                "score_aware_long_training_epochs": 0,
            },
            scorer_upstream_dir="upstream",
            output_height=16,
            output_width=16,
            run_archive_export=False,
            official_trained_checkpoint_state_dict=(
                _minimal_full_official_decoder_state()
            ),
        )


def test_checkpoint_retention_candidate_null_preserves_safe_default() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    assert (
        mod._candidate_first_non_null(
            {
                "score_aware_long_training_checkpoint_retention_keep_last_n": None,
                "snerv_score_aware_long_training_checkpoint_retention_keep_last_n": None,
            },
            (
                "score_aware_long_training_checkpoint_retention_keep_last_n",
                "snerv_score_aware_long_training_checkpoint_retention_keep_last_n",
            ),
            mod.SNERV_SCORE_AWARE_CHECKPOINT_RETENTION_KEEP_LAST_N_DEFAULT,
        )
        == mod.SNERV_SCORE_AWARE_CHECKPOINT_RETENTION_KEEP_LAST_N_DEFAULT
    )
    assert mod._coerce_checkpoint_keep_last(-1) is None
    assert mod._coerce_checkpoint_keep_last(3) == 3
    with pytest.raises(mod.SnervMlxNativeExportError, match="bool"):
        mod._coerce_checkpoint_keep_last(True)
    with pytest.raises(mod.SnervMlxNativeExportError, match=">= -1"):
        mod._coerce_checkpoint_keep_last(-2)
    with pytest.raises(mod.SnervMlxNativeExportError, match="non-negative"):
        mod._coerce_checkpoint_keep_best(-1)
    with pytest.raises(mod.SnervMlxNativeExportError, match="> 0"):
        mod._coerce_checkpoint_keep_every(0)


def test_snerv_gradient_multiplier_candidate_coercion() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    assert mod._coerce_gradient_multiplier_by_name(
        {"decoder.weight": 0.25}
    ) == {"decoder.weight": 0.25}
    assert mod._coerce_gradient_multiplier_by_name(
        [{"name": "rgb_1.bias", "value": 0.0}]
    ) == {"rgb_1.bias": 0.0}
    assert mod._coerce_gradient_multiplier_by_name(
        [["blocks.0.weight", 2.0]]
    ) == {"blocks.0.weight": 2.0}
    with pytest.raises(mod.SnervMlxNativeExportError, match="finite and >= 0"):
        mod._coerce_gradient_multiplier_by_name({"bad": -1.0})
    with pytest.raises(mod.SnervMlxNativeExportError, match="name/value"):
        mod._coerce_gradient_multiplier_by_name([{"name": "bad"}])


def test_score_aware_checkpoint_selection_policy_fails_closed_on_missing_inputs() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    policy = mod._snerv_score_aware_checkpoint_selection_policy(
        segnet_distillation_weight=0.25,
        segnet_direct_live_distillation_weight=0.5,
        pose_distillation_weight=0.1,
        pose_direct_live_distillation_weight=0.75,
        has_real_segnet_teacher=False,
        has_real_posenet_teacher=False,
        coder_aware_qat_bound=True,
        coder_qat_loss_weight_map={},
        pr95_faithful_curriculum_enabled=True,
    )

    assert policy["uses_score_aware_composite"] is True
    assert policy["selection_metric"] == "score_aware_composite_full_video_surrogate"
    assert policy["fail_closed_on_missing_parts"] is True
    assert "distill" in policy["required_loss_parts"]
    assert "segnet_direct_live_distill" in policy["required_loss_parts"]
    assert "pose_score_term" in policy["required_loss_parts"]
    assert "pose_direct_live_score_term" in policy["required_loss_parts"]
    assert "pr95_stage_scorer_surrogate" in policy["required_loss_parts"]
    assert policy["pose_selection_loss_part"] == "pose_direct_live_score_term"
    assert policy["pose_direct_live_distillation_weight"] == pytest.approx(0.75)
    assert "real_segnet_direct_live_distillation" in policy[
        "active_score_surfaces"
    ]
    assert "real_posenet_direct_live_distillation" in policy[
        "active_score_surfaces"
    ]
    assert "snerv_score_aware_checkpoint_selection_segnet_teacher_missing" in policy[
        "blockers"
    ]
    assert "snerv_score_aware_checkpoint_selection_posenet_teacher_missing" in policy[
        "blockers"
    ]
    assert "snerv_score_aware_checkpoint_selection_coder_qat_terms_missing" in policy[
        "blockers"
    ]
    assert (
        "snerv_score_aware_checkpoint_selection_pr95_stage_selector_missing"
        not in policy["blockers"]
    )


def test_score_aware_checkpoint_selection_policy_preserves_mse_fallback() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    policy = mod._snerv_score_aware_checkpoint_selection_policy(
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        has_real_segnet_teacher=False,
        has_real_posenet_teacher=False,
        coder_aware_qat_bound=False,
        coder_qat_loss_weight_map={},
        pr95_faithful_curriculum_enabled=False,
    )

    assert policy["uses_score_aware_composite"] is False
    assert policy["mse_fallback"] is True
    assert policy["selection_metric"] == "full_reconstruction_mse_nchw255"
    assert policy["selection_metric_value_key"] == "recon_mse_nchw255"
    assert policy["blockers"] == []


def test_score_aware_checkpoint_selection_policy_prices_direct_live_only() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    policy = mod._snerv_score_aware_checkpoint_selection_policy(
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_direct_live_distillation_weight=0.25,
        has_real_segnet_teacher=True,
        has_real_posenet_teacher=False,
        coder_aware_qat_bound=False,
        coder_qat_loss_weight_map={},
        pr95_faithful_curriculum_enabled=False,
    )

    assert policy["uses_score_aware_composite"] is True
    assert policy["required_loss_parts"] == ["recon", "segnet_direct_live_distill"]
    assert policy["blockers"] == []
    assert policy["segnet_direct_live_distillation_weight"] == pytest.approx(0.25)


def test_score_aware_checkpoint_selection_policy_prices_direct_live_subcontrol_only() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    policy = mod._snerv_score_aware_checkpoint_selection_policy(
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_direct_live_distillation_weight=0.0,
        segnet_direct_live_class_region_recon_weight=0.75,
        has_real_segnet_teacher=True,
        has_real_posenet_teacher=False,
        coder_aware_qat_bound=False,
        coder_qat_loss_weight_map={},
        pr95_faithful_curriculum_enabled=False,
    )

    assert policy["uses_score_aware_composite"] is True
    assert policy["mse_fallback"] is False
    assert "real_segnet_direct_live_subcontrols" in policy["active_score_surfaces"]
    assert policy["required_loss_parts"] == [
        "recon",
        "segnet_direct_live_distill",
        "segnet_direct_live_class_region_recon_loss",
    ]
    assert policy["segnet_direct_live_distillation_weight"] == pytest.approx(0.0)
    assert policy["segnet_direct_live_subcontrol_weights"] == {
        "class_region_recon": pytest.approx(0.75)
    }
    assert policy["blockers"] == []


def test_score_aware_checkpoint_selection_policy_prices_contrast_floor() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    policy = mod._snerv_score_aware_checkpoint_selection_policy(
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        scorer_input_contrast_floor_weight=0.875,
        has_real_segnet_teacher=False,
        has_real_posenet_teacher=False,
        coder_aware_qat_bound=False,
        coder_qat_loss_weight_map={},
        pr95_faithful_curriculum_enabled=False,
    )

    assert policy["uses_score_aware_composite"] is True
    assert policy["mse_fallback"] is False
    assert "scorer_input_contrast_floor" in policy["active_score_surfaces"]
    assert "scorer_input_contrast_floor" in policy["required_loss_parts"]
    assert policy["scorer_input_contrast_floor_weight"] == pytest.approx(0.875)
    assert policy["blockers"] == []


def test_score_aware_checkpoint_selection_policy_prices_shape_tether() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    policy = mod._snerv_score_aware_checkpoint_selection_policy(
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        scorer_input_shape_tether_weight=0.625,
        has_real_segnet_teacher=False,
        has_real_posenet_teacher=False,
        coder_aware_qat_bound=False,
        coder_qat_loss_weight_map={},
        pr95_faithful_curriculum_enabled=False,
    )

    assert policy["uses_score_aware_composite"] is True
    assert policy["mse_fallback"] is False
    assert "scorer_input_shape_tether" in policy["active_score_surfaces"]
    assert "scorer_input_shape_tether" in policy["required_loss_parts"]
    assert policy["scorer_input_shape_tether_weight"] == pytest.approx(0.625)
    assert policy["blockers"] == []


def test_score_aware_checkpoint_selection_policy_prices_posenet_temporal_signal_floor() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    policy = mod._snerv_score_aware_checkpoint_selection_policy(
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        posenet_temporal_signal_floor_weight=0.5,
        has_real_segnet_teacher=False,
        has_real_posenet_teacher=False,
        coder_aware_qat_bound=False,
        coder_qat_loss_weight_map={},
        pr95_faithful_curriculum_enabled=False,
    )

    assert policy["uses_score_aware_composite"] is True
    assert policy["mse_fallback"] is False
    assert "posenet_temporal_signal_floor" in policy["active_score_surfaces"]
    assert "posenet_temporal_signal_floor" in policy["required_loss_parts"]
    assert policy["posenet_temporal_signal_floor_weight"] == pytest.approx(0.5)
    assert policy["blockers"] == []


def test_score_aware_checkpoint_selection_policy_prices_posenet_yuv6_geometry_tether() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    policy = mod._snerv_score_aware_checkpoint_selection_policy(
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        posenet_yuv6_geometry_tether_weight=0.5,
        has_real_segnet_teacher=False,
        has_real_posenet_teacher=False,
        coder_aware_qat_bound=False,
        coder_qat_loss_weight_map={},
        pr95_faithful_curriculum_enabled=False,
    )

    assert policy["uses_score_aware_composite"] is True
    assert policy["mse_fallback"] is False
    assert "posenet_yuv6_geometry_tether" in policy["active_score_surfaces"]
    assert "posenet_yuv6_geometry_tether" in policy["required_loss_parts"]
    assert policy["posenet_yuv6_geometry_tether_weight"] == pytest.approx(0.5)
    assert policy["blockers"] == []


def test_score_aware_checkpoint_selection_prefers_segnet_target_support_before_scalar() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    incumbent = {
        "score_aware_composite_loss": 1.0,
        "score_aware_checkpoint_selection_blockers": [],
        "score_aware_composite_parts": {
            "raw_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
            "raw_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
            "raw_segnet_direct_live_candidate_target_class_min_ratio": 0.0,
            "raw_segnet_direct_live_argmax_disagreement": 0.30,
        },
    }
    candidate = {
        "score_aware_composite_loss": 1.25,
        "score_aware_checkpoint_selection_blockers": [],
        "score_aware_composite_parts": {
            "raw_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
            "raw_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
            "raw_segnet_direct_live_candidate_target_class_min_ratio": 0.05,
            "raw_segnet_direct_live_argmax_disagreement": 0.35,
        },
    }

    assert mod._snerv_checkpoint_selection_row_is_better(
        candidate,
        incumbent,
        metric_value_key="score_aware_composite_loss",
    )
    assert not mod._snerv_checkpoint_selection_row_is_better(
        incumbent,
        candidate,
        metric_value_key="score_aware_composite_loss",
    )


def test_score_aware_checkpoint_selection_rejects_blocked_support_row() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    incumbent = {
        "score_aware_composite_loss": 1.0,
        "score_aware_checkpoint_selection_blockers": [],
        "score_aware_composite_parts": {
            "raw_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
            "raw_segnet_direct_live_candidate_target_class_min_ratio": 0.05,
        },
    }
    candidate = {
        "score_aware_composite_loss": 0.5,
        "score_aware_checkpoint_selection_blockers": [
            "snerv_score_aware_checkpoint_selection_required_parts_missing"
        ],
        "score_aware_composite_parts": {
            "raw_segnet_direct_live_candidate_target_class_coverage_fraction": 1.0,
            "raw_segnet_direct_live_candidate_target_class_min_ratio": 0.2,
        },
    }

    assert not mod._snerv_checkpoint_selection_row_is_better(
        candidate,
        incumbent,
        metric_value_key="score_aware_composite_loss",
    )


def test_score_aware_checkpoint_selection_rejects_blocked_support_row_against_scalar_only() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    incumbent = {
        "score_aware_composite_loss": 1.0,
        "score_aware_checkpoint_selection_blockers": [],
    }
    candidate = {
        "score_aware_composite_loss": 0.5,
        "score_aware_checkpoint_selection_blockers": [
            "snerv_score_aware_checkpoint_selection_required_parts_missing"
        ],
        "score_aware_composite_parts": {
            "raw_segnet_direct_live_candidate_target_class_coverage_fraction": 1.0,
            "raw_segnet_direct_live_candidate_target_class_min_ratio": 0.2,
        },
    }

    assert not mod._snerv_checkpoint_selection_row_is_better(
        candidate,
        incumbent,
        metric_value_key="score_aware_composite_loss",
    )


def test_score_aware_long_training_direct_live_only_requires_research_gate(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    report = mod._run_score_aware_long_training_attachment(
        requested_epochs=0,
        output_dir=tmp_path,
        pairs_nchw255=_tiny_pairs(pairs=1),
        model_size=SnervModelSizeConfig(),
        levels=1,
        wavelet="haar",
        source_pair_indices=(0,),
        target_bits_per_coeff=8.0,
        step_map_bits_per_coeff=4.0,
        decoder_payload_codec="npz",
        lf_payload_codec="raw",
        recon_pixel_weight=None,
        recon_pixel_weight_metadata=None,
        hf_decoder_saliency_gain=0.0,
        hard_byte_ceiling=None,
        learning_rate=1.0e-3,
        batch_pairs=1,
        section_byte_refresh_every_steps=1,
        optimizer_kind="pact_muon_adamw",
        grad_clip_max_norm=1.0,
        weight_decay=1.0e-4,
        eval_roundtrip_ste=False,
        scorer_input_distribution_guard_weight=0.0,
        scorer_input_distribution_guard_saturation_margin=0.02,
        scorer_input_distribution_guard_temperature=0.01,
        scorer_input_contrast_floor_weight=0.0,
        scorer_input_contrast_floor_segnet_min_std_ratio=0.5,
        scorer_input_contrast_floor_posenet_yuv6_min_std_ratio=0.5,
        checkpoint_retention_keep_last_n=1,
        checkpoint_retention_keep_best_n=1,
        checkpoint_retention_keep_every_n_epochs=None,
        checkpoint_retention_cold_store_roots=(),
        scorer_upstream_dir=tmp_path / "missing_upstream",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        pose_distillation_loss="mse",
        pose_distillation_huber_delta=1.0,
        segnet_distillation_objective="kl_t2",
        distillation_temperature=2.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.25,
        segnet_tau_boundary=1.0,
        segnet_hinge_margin=1.0,
        distillation_device="cpu",
        allow_segnet_only_research=False,
        coder_aware_qat=False,
        coder_qat_quant_bits=8,
        coder_qat_quant_residual_weight=0.0,
        coder_qat_magnitude_weight=0.0,
        coder_qat_delta_weight=0.0,
        coder_qat_c1a_entropy_weight=0.0,
        coder_qat_c1a_sigma=0.2,
        coder_qat_c1a_sample_size=4,
        pr95_faithful_curriculum_enabled=False,
        pr95_muon_policy="every_stage",
        prioritized_pair_indices=(),
        scorer_error_pair_sampling_weights=None,
        scorer_error_pair_curriculum=None,
        allow_overwrite=True,
    )

    assert (
        "snerv_score_aware_long_training_segnet_requires_posenet_teacher"
        in report["blockers"]
    )


def test_score_aware_long_training_rejects_invalid_contrast_floor_controls(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    report = mod._run_score_aware_long_training_attachment(
        requested_epochs=1,
        output_dir=tmp_path,
        pairs_nchw255=_tiny_pairs(pairs=1),
        model_size=SnervModelSizeConfig(),
        levels=1,
        wavelet="haar",
        source_pair_indices=(0,),
        target_bits_per_coeff=8.0,
        step_map_bits_per_coeff=4.0,
        decoder_payload_codec="npz",
        lf_payload_codec="raw",
        recon_pixel_weight=None,
        recon_pixel_weight_metadata=None,
        hf_decoder_saliency_gain=0.0,
        hard_byte_ceiling=None,
        learning_rate=1.0e-3,
        batch_pairs=1,
        section_byte_refresh_every_steps=1,
        optimizer_kind="pact_muon_adamw",
        grad_clip_max_norm=1.0,
        weight_decay=1.0e-4,
        eval_roundtrip_ste=False,
        scorer_input_distribution_guard_weight=0.0,
        scorer_input_distribution_guard_saturation_margin=0.02,
        scorer_input_distribution_guard_temperature=0.01,
        scorer_input_contrast_floor_weight=-0.1,
        scorer_input_contrast_floor_segnet_min_std_ratio=0.0,
        scorer_input_contrast_floor_posenet_yuv6_min_std_ratio=-0.5,
        checkpoint_retention_keep_last_n=1,
        checkpoint_retention_keep_best_n=1,
        checkpoint_retention_keep_every_n_epochs=None,
        checkpoint_retention_cold_store_roots=(),
        scorer_upstream_dir=tmp_path / "missing_upstream",
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        pose_distillation_loss="mse",
        pose_distillation_huber_delta=1.0,
        segnet_distillation_objective="kl_t2",
        distillation_temperature=2.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.0,
        segnet_tau_boundary=1.0,
        segnet_hinge_margin=1.0,
        distillation_device="cpu",
        allow_segnet_only_research=False,
        coder_aware_qat=False,
        coder_qat_quant_bits=8,
        coder_qat_quant_residual_weight=0.0,
        coder_qat_magnitude_weight=0.0,
        coder_qat_delta_weight=0.0,
        coder_qat_c1a_entropy_weight=0.0,
        coder_qat_c1a_sigma=0.2,
        coder_qat_c1a_sample_size=4,
        pr95_faithful_curriculum_enabled=False,
        pr95_muon_policy="every_stage",
        prioritized_pair_indices=(),
        scorer_error_pair_sampling_weights=None,
        scorer_error_pair_curriculum=None,
        allow_overwrite=True,
    )

    assert report["executed"] is False
    assert (
        "snerv_score_aware_long_training_eval_roundtrip_ste_required_for_nonzero_epochs"
        in report["validation_failures"]
    )
    assert (
        "snerv_score_aware_long_training_scorer_input_contrast_floor_weight_invalid"
        in report["validation_failures"]
    )
    assert (
        "snerv_score_aware_long_training_scorer_input_contrast_floor_segnet_ratio_invalid"
        in report["validation_failures"]
    )
    assert (
        "snerv_score_aware_long_training_scorer_input_contrast_floor_posenet_ratio_invalid"
        in report["validation_failures"]
    )


def test_snerv_scorer_tether_dual_targets_are_strict_before_long_training() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod
    from tac.substrates._shared.mlx_score_aware.dual_ascent import (
        build_default_nerv_train_time_dual_ascent_config,
    )

    base = build_default_nerv_train_time_dual_ascent_config(
        family="snerv",
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
    )
    bound = mod._bind_snerv_scorer_tether_dual_targets(base)
    constraints = {row["constraint_id"]: row for row in bound["constraints"]}

    for constraint_id in (
        "snerv_segnet_last_frame_distill",
        "snerv_posenet_yuv6_pair_distill",
    ):
        assert constraints[constraint_id]["target"] == pytest.approx(0.0)
        assert "target_fraction_of_initial" not in constraints[constraint_id]
        assert (
            constraints[constraint_id]["scorer_tether_launch_gate_target_bound"]
            is True
        )
    assert bound["snerv_scorer_tether_launch_gate_target_policy"][
        "constraint_ids"
    ] == [
        "snerv_posenet_yuv6_pair_distill",
        "snerv_segnet_last_frame_distill",
    ]


def test_score_aware_telemetry_contract_accepts_live_dual_and_section_metrics(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                    "loss_components": {
                        "loss_part_distill": 1.0,
                        "loss_part_pose_distill": 2.0,
                        "loss_part_pose_direct_live_distill": 1.75,
                        "loss_part_pose_direct_live_raw_mse": 0.25,
                        "loss_part_pose_direct_live_score_term": 1.75,
                        "loss_part_pr95_stage_seg_surrogate": 1.0,
                    "loss_part_pr95_stage_pose_surrogate": 2.0,
                    "loss_part_pr95_stage_scorer_input_distribution_guard": 0.25,
                    "loss_part_pr95_stage_scorer_input_contrast_floor": 0.125,
                    "loss_part_pr95_stage_scorer_input_contrast_floor_segnet_last_rgb_mean_std_ratio": 0.7,
                    "loss_part_pr95_stage_scorer_input_contrast_floor_posenet_yuv6_pair_mean_std_ratio": 0.6,
                    "loss_part_pr95_stage_scorer_input_shape_tether": 0.25,
                    "loss_part_pr95_stage_scorer_input_shape_tether_segnet_last_rgb": 0.0625,
                    "loss_part_pr95_stage_scorer_input_shape_tether_posenet_yuv6_pair": 0.09375,
                    "loss_part_pr95_stage_scorer_input_shape_tether_posenet_yuv6_temporal_delta": 0.09375,
                    "loss_part_pr95_stage_posenet_yuv6_geometry_tether": 0.1875,
                    "loss_part_pr95_stage_posenet_yuv6_geometry_tether_pair": 0.09375,
                    "loss_part_pr95_stage_posenet_yuv6_geometry_tether_temporal_delta": 0.09375,
                    "loss_part_pr95_stage_posenet_temporal_signal_floor": 0.03125,
                    "loss_part_pr95_stage_posenet_temporal_signal_floor_mean_std_ratio": 0.7,
                    "loss_part_pr95_stage_posenet_temporal_signal_floor_mean_abs_ratio": 0.6,
                    "loss_part_pr95_stage_segnet_direct_live_distill": 0.0625,
                    "loss_part_pr95_stage_segnet_direct_live_argmax_disagreement": 0.5,
                    "loss_part_pr95_stage_segnet_direct_live_candidate_occupied_class_fraction": 0.6,
                    "loss_part_pr95_stage_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
                    "loss_part_segnet_direct_live_class_region_recon_loss": 0.04,
                    "loss_part_weighted_pr95_stage_segnet_direct_live_distill": 0.03125,
                    "segnet_student_live_calibration_active": 1.0,
                    "loss_part_segnet_student_live_calibration": 0.125,
                    "loss_part_weighted_segnet_student_live_calibration": 0.125,
                    "train_time_archive_rate_score": 0.02,
                    "train_time_section_rate_score__decoder_payload": 0.01,
                    "dual_ascent_missing_metric__snerv_segnet_last_frame_distill": 0.0,
                    "dual_ascent_missing_metric__snerv_posenet_yuv6_pair_distill": 0.0,
                    "dual_ascent_missing_metric__snerv_scorer_input_distribution_guard": 0.0,
                    "dual_ascent_lambda__snerv_segnet_last_frame_distill": 0.25,
                    "dual_ascent_lambda__snerv_posenet_yuv6_pair_distill": 0.5,
                    "dual_ascent_lambda__snerv_archive_total_bytes": 0.375,
                    "dual_ascent_lambda__snerv_decoder_payload_section_bytes": 0.125,
                    "dual_ascent_weight_applied__snerv_archive_total_bytes": 1.0,
                    "dual_ascent_weight_applied__snerv_decoder_payload_section_bytes": 1.0,
                    "gradient_multiplier_requested_control_count": 1.0,
                    "gradient_multiplier_applied_leaf_count": 1.0,
                    "gradient_multiplier_missing_requested_count": 0.0,
                    "gradient_multiplier_requested_but_unapplied": 0.0,
                    "scorer_space_step_guard_enabled": 1.0,
                    "scorer_space_step_guard_eligible": 1.0,
                    "scorer_space_step_guard_rejected": 0.0,
                    "scorer_space_step_guard_effective_optimizer_learning_rate": 1.0e-3,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        pose_direct_live_distillation_weight=0.75,
        segnet_student_live_calibration_weight=1.0,
        segnet_direct_live_distillation_weight=0.5,
        segnet_direct_live_class_region_recon_weight=0.25,
        pr95_faithful_curriculum_enabled=True,
        coder_aware_qat_bound=True,
        train_time_section_byte_control_bound=True,
        scorer_input_distribution_guard_weight=2.0,
        scorer_input_contrast_floor_weight=0.875,
        scorer_input_shape_tether_weight=0.625,
        posenet_yuv6_geometry_tether_weight=0.5,
        posenet_temporal_signal_floor_weight=0.5,
        gradient_multiplier_controls_requested=True,
        scorer_space_step_guard_enabled=True,
    )

    assert contract["passed"] is True
    assert contract["blockers"] == []
    assert contract["expected_posenet_direct_live_distillation"] is True
    assert contract["posenet_direct_live_loss_observed"] is True
    assert contract["posenet_direct_live_raw_mse_observed"] is True
    assert contract["posenet_direct_live_score_term_observed"] is True
    assert contract["segnet_dual_metric_observed"] is True
    assert contract["posenet_dual_metric_observed"] is True
    assert contract["segnet_dual_lambda_active_observed"] is True
    assert contract["posenet_dual_lambda_active_observed"] is True
    assert contract["archive_rate_metric_observed"] is True
    assert contract["archive_byte_dual_lambda_active_observed"] is True
    assert contract["archive_byte_dual_positive_violation_observed"] is False
    assert contract["archive_byte_dual_weight_applied_observed"] is True
    assert contract["section_rate_metric_observed"] is True
    assert contract["section_byte_dual_lambda_active_observed"] is True
    assert contract["section_byte_dual_weight_applied_observed"] is True
    assert contract["section_byte_dual_zero_base_masked_observed"] is False
    assert contract["expected_gradient_multiplier_controls"] is True
    assert contract["gradient_multiplier_requested_observed"] is True
    assert contract["gradient_multiplier_applied_observed"] is True
    assert contract["gradient_multiplier_missing_requested_observed"] is False
    assert contract["gradient_multiplier_noop_observed"] is False
    assert contract["expected_scorer_space_step_guard"] is True
    assert contract["scorer_space_step_guard_config_observed"] is True
    assert contract["scorer_space_step_guard_metric_observed"] is True
    assert contract["scorer_space_step_guard_intervention_observed"] is False
    assert contract["scorer_input_guard_metric_observed"] is True
    assert contract["scorer_input_guard_dual_metric_observed"] is True
    assert contract["expected_scorer_input_contrast_floor_metric"] is True
    assert contract["scorer_input_contrast_floor_metric_observed"] is True
    assert (
        contract["scorer_input_contrast_floor_segnet_ratio_metric_observed"]
        is True
    )
    assert (
        contract["scorer_input_contrast_floor_posenet_ratio_metric_observed"]
        is True
    )
    assert contract["expected_scorer_input_shape_tether_metric"] is True
    assert contract["scorer_input_shape_tether_metric_observed"] is True
    assert contract["scorer_input_shape_tether_segnet_metric_observed"] is True
    assert contract["scorer_input_shape_tether_posenet_pair_metric_observed"] is True
    assert contract["scorer_input_shape_tether_posenet_delta_metric_observed"] is True
    assert contract["expected_posenet_yuv6_geometry_tether_metric"] is True
    assert contract["posenet_yuv6_geometry_tether_metric_observed"] is True
    assert contract["posenet_yuv6_geometry_tether_pair_metric_observed"] is True
    assert contract["posenet_yuv6_geometry_tether_delta_metric_observed"] is True
    assert contract["expected_posenet_temporal_signal_floor_metric"] is True
    assert contract["posenet_temporal_signal_floor_metric_observed"] is True
    assert (
        contract["posenet_temporal_signal_floor_std_ratio_metric_observed"]
        is True
    )
    assert (
        contract["posenet_temporal_signal_floor_mean_abs_ratio_metric_observed"]
        is True
    )
    assert contract["segnet_live_calibration_active_observed"] is True
    assert contract["segnet_live_calibration_loss_observed"] is True
    assert contract["expected_segnet_direct_live_distillation"] is True
    assert contract["expected_segnet_direct_live_class_region_recon"] is True
    assert contract["segnet_direct_live_distillation_loss_observed"] is True
    assert contract["segnet_direct_live_class_region_recon_metric_observed"] is True
    assert contract["segnet_direct_live_argmax_metric_observed"] is True
    assert contract["segnet_direct_live_class_occupancy_metric_observed"] is True
    assert contract[
        "segnet_direct_live_max_candidate_occupied_class_fraction"
    ] == pytest.approx(0.6)
    assert contract["segnet_direct_live_target_class_coverage_metric_observed"] is True
    assert contract[
        "segnet_direct_live_max_candidate_target_class_coverage_fraction"
    ] == pytest.approx(0.8)


def test_score_aware_telemetry_contract_requires_scorer_space_step_guard_metrics(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_segnet_direct_live_distill": 0.125,
                    "loss_part_segnet_direct_live_argmax_disagreement": 0.5,
                    "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
                    "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.25,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
        scorer_space_step_guard_enabled=True,
    )

    assert contract["passed"] is False
    assert contract["expected_scorer_space_step_guard"] is True
    assert contract["scorer_space_step_guard_config_observed"] is False
    assert contract["scorer_space_step_guard_metric_observed"] is False
    assert (
        "snerv_score_aware_long_training_scorer_space_step_guard_config_missing"
        in contract["blockers"]
    )
    assert (
        "snerv_score_aware_long_training_scorer_space_step_guard_metric_missing"
        in contract["blockers"]
    )


def test_score_aware_telemetry_contract_requires_direct_live_region_recon_metric(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_segnet_direct_live_distill": 0.125,
                    "loss_part_segnet_direct_live_argmax_disagreement": 0.5,
                    "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.6,
                    "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.0,
        segnet_direct_live_class_region_recon_weight=0.25,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is False
    assert contract["expected_segnet_direct_live_distillation"] is True
    assert contract["expected_segnet_direct_live_class_region_recon"] is True
    assert contract["segnet_direct_live_distillation_loss_observed"] is True
    assert contract["segnet_direct_live_class_region_recon_metric_observed"] is False
    assert (
        "snerv_score_aware_long_training_direct_live_segnet_class_region_recon_metric_missing"
        in contract["blockers"]
    )


def test_score_aware_telemetry_contract_requires_direct_live_target_floor_metrics(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_segnet_direct_live_distill": 0.125,
                    "loss_part_segnet_direct_live_argmax_disagreement": 0.5,
                    "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
                    "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.25,
        segnet_direct_live_target_mass_floor_weight=0.5,
        segnet_direct_live_target_min_ratio_floor_weight=0.5,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is False
    assert contract["expected_segnet_direct_live_target_mass_floor"] is True
    assert contract["expected_segnet_direct_live_target_min_ratio_floor"] is True
    assert contract["segnet_direct_live_target_mass_floor_metric_observed"] is False
    assert (
        contract["segnet_direct_live_target_min_ratio_floor_metric_observed"]
        is False
    )
    assert (
        "snerv_score_aware_long_training_direct_live_segnet_target_mass_floor_metric_missing"
        in contract["blockers"]
    )
    assert (
        "snerv_score_aware_long_training_direct_live_segnet_target_min_ratio_floor_metric_missing"
        in contract["blockers"]
    )


def test_score_aware_telemetry_contract_requires_direct_live_target_floor_duals(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    loss_components = {
        "loss_part_segnet_direct_live_distill": 0.125,
        "loss_part_segnet_direct_live_argmax_disagreement": 0.5,
        "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
        "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.8,
        "loss_part_segnet_direct_live_target_mass_floor_loss": 0.03125,
        "loss_part_segnet_direct_live_target_min_ratio_floor_loss": 0.015625,
    }
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": loss_components,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.0,
        segnet_direct_live_target_mass_floor_weight=0.5,
        segnet_direct_live_target_min_ratio_floor_weight=0.5,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is False
    assert contract["expected_segnet_direct_live_target_mass_floor"] is True
    assert contract["expected_segnet_direct_live_target_min_ratio_floor"] is True
    assert contract["segnet_direct_live_target_mass_floor_metric_observed"] is True
    assert contract["segnet_direct_live_target_min_ratio_floor_metric_observed"] is True
    assert (
        contract["segnet_direct_live_target_mass_floor_dual_metric_observed"]
        is False
    )
    assert (
        contract[
            "segnet_direct_live_target_mass_floor_dual_lambda_active_observed"
        ]
        is False
    )
    assert (
        contract["segnet_direct_live_target_min_ratio_floor_dual_metric_observed"]
        is False
    )
    assert (
        contract[
            "segnet_direct_live_target_min_ratio_floor_dual_lambda_active_observed"
        ]
        is False
    )
    assert (
        "snerv_score_aware_long_training_direct_live_segnet_target_mass_floor_dual_metric_never_observed"
        in contract["blockers"]
    )
    assert (
        "snerv_score_aware_long_training_direct_live_segnet_target_mass_floor_dual_lambda_never_active"
        in contract["blockers"]
    )
    assert (
        "snerv_score_aware_long_training_direct_live_segnet_target_min_ratio_floor_dual_metric_never_observed"
        in contract["blockers"]
    )
    assert (
        "snerv_score_aware_long_training_direct_live_segnet_target_min_ratio_floor_dual_lambda_never_active"
        in contract["blockers"]
    )

    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    **loss_components,
                    "dual_ascent_missing_metric__snerv_segnet_direct_live_target_mass_floor": 0.0,
                    "dual_ascent_missing_metric__snerv_segnet_direct_live_target_min_ratio_floor": 0.0,
                    "dual_ascent_lambda__snerv_segnet_direct_live_target_mass_floor": 0.25,
                    "dual_ascent_lambda__snerv_segnet_direct_live_target_min_ratio_floor": 0.25,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.0,
        segnet_direct_live_target_mass_floor_weight=0.5,
        segnet_direct_live_target_min_ratio_floor_weight=0.5,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is True
    assert contract["blockers"] == []
    assert contract["segnet_direct_live_target_mass_floor_dual_metric_observed"] is True
    assert (
        contract[
            "segnet_direct_live_target_mass_floor_dual_lambda_active_observed"
        ]
        is True
    )
    assert (
        contract["segnet_direct_live_target_min_ratio_floor_dual_metric_observed"]
        is True
    )
    assert (
        contract[
            "segnet_direct_live_target_min_ratio_floor_dual_lambda_active_observed"
        ]
        is True
    )


def test_score_aware_telemetry_contract_rejects_direct_live_target_class_collapse(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_segnet_direct_live_distill": 0.125,
                    "loss_part_segnet_direct_live_argmax_disagreement": 0.5,
                    "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.8,
                    "loss_part_segnet_direct_live_candidate_target_class_coverage_fraction": 0.6,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.25,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is False
    assert contract["segnet_direct_live_class_occupancy_metric_observed"] is True
    assert contract["segnet_direct_live_target_class_coverage_metric_observed"] is True
    assert contract[
        "segnet_direct_live_max_candidate_target_class_coverage_fraction"
    ] == pytest.approx(0.6)
    assert (
        "snerv_score_aware_long_training_direct_live_segnet_candidate_argmax_collapsed"
        not in contract["blockers"]
    )
    assert (
        "snerv_score_aware_long_training_direct_live_segnet_target_class_coverage_collapsed"
        in contract["blockers"]
    )


def test_score_aware_telemetry_contract_allows_archive_dual_under_byte_target(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_distill": 1.0,
                    "loss_part_pose_distill": 2.0,
                    "train_time_archive_rate_score": 0.002,
                    "train_time_section_rate_score__decoder_payload": 0.01,
                    "dual_ascent_missing_metric__snerv_segnet_last_frame_distill": 0.0,
                    "dual_ascent_missing_metric__snerv_posenet_yuv6_pair_distill": 0.0,
                    "dual_ascent_lambda__snerv_segnet_last_frame_distill": 0.25,
                    "dual_ascent_lambda__snerv_posenet_yuv6_pair_distill": 0.5,
                    "dual_ascent_lambda__snerv_archive_total_bytes": 0.375,
                    "dual_ascent_violation__snerv_archive_total_bytes": -0.004,
                    "dual_ascent_lambda__snerv_decoder_payload_section_bytes": 0.125,
                    "dual_ascent_weight_applied__snerv_decoder_payload_section_bytes": 1.0,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        segnet_student_live_calibration_weight=0.0,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=True,
        train_time_section_byte_control_bound=True,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is True
    assert contract["archive_byte_dual_lambda_active_observed"] is True
    assert contract["archive_byte_dual_positive_violation_observed"] is False
    assert contract["archive_byte_dual_weight_applied_observed"] is False
    assert (
        "snerv_score_aware_long_training_archive_byte_dual_weight_never_applied"
        not in contract["blockers"]
    )


def test_latest_score_aware_training_metrics_extracts_nested_guard_and_deltas(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        "\n".join(
            (
                "{not-json",
                json.dumps(
                    {
                        "epoch": 7,
                        "loss": 12.5,
                        "loss_components": {
                            "scorer_space_step_guard_enabled": 1.0,
                            "scorer_space_step_guard_eligible": 1.0,
                            "scorer_space_step_guard_rejected": 0.0,
                            "scorer_space_step_guard_optimizer_learning_rate_scale": 0.5,
                            "dynamics_pre_update_loss_part_pose_direct_live_score_term": 0.3,
                            "loss_part_pose_direct_live_score_term": 0.2,
                            "dynamics_pre_update_loss_part_segnet_direct_live_argmax_disagreement": 0.05,
                            "loss_part_segnet_direct_live_argmax_disagreement": 0.04,
                            "dynamics_gradient_all_l2": 4.0,
                            "dynamics_param_delta_all_l2": 2.0,
                            "train_time_archive_bytes": 1234.0,
                            "train_time_section_bytes__decoder_payload": 12.0,
                            "train_time_section_bytes__lf_payload": 34.0,
                        },
                    },
                    sort_keys=True,
                ),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    summary = mod._latest_snerv_score_aware_training_metrics(telemetry)

    assert summary["telemetry_exists"] is True
    assert summary["row_count"] == 1
    assert summary["malformed_rows"] == 1
    assert summary["latest_epoch"] == 7
    assert summary["latest_loss"] == pytest.approx(12.5)
    assert summary["train_time_archive_bytes"] == pytest.approx(1234.0)
    assert summary["train_time_section_bytes"] == {
        "decoder_payload": 12.0,
        "lf_payload": 34.0,
    }
    assert summary["scorer_space_step_guard"]["scorer_space_step_guard_enabled"] == 1.0
    assert summary["scorer_space_step_guard"][
        "scorer_space_step_guard_optimizer_learning_rate_scale"
    ] == pytest.approx(0.5)
    pose_delta = summary["scorer_deltas"]["pose_direct_live_score_term"]
    assert pose_delta["pre"] == pytest.approx(0.3)
    assert pose_delta["post"] == pytest.approx(0.2)
    assert pose_delta["delta"] == pytest.approx(-0.1)
    assert pose_delta["improved_or_equal"] is True
    assert summary["scorer_deltas"]["segnet_direct_live_argmax_disagreement"][
        "improved_or_equal"
    ] is True
    assert summary["dynamics"]["dynamics_gradient_all_l2"] == pytest.approx(4.0)
    assert summary["blockers"] == [
        "snerv_score_aware_latest_telemetry_malformed_rows"
    ]


def test_score_aware_telemetry_contract_allows_byte_duals_under_targets(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_distill": 1.0,
                    "loss_part_pose_distill": 2.0,
                    "train_time_archive_rate_score": 0.02,
                    "train_time_section_rate_score__decoder_payload": 0.01,
                    "dual_ascent_missing_metric__snerv_segnet_last_frame_distill": 0.0,
                    "dual_ascent_missing_metric__snerv_posenet_yuv6_pair_distill": 0.0,
                    "dual_ascent_lambda__snerv_segnet_last_frame_distill": 0.25,
                    "dual_ascent_lambda__snerv_posenet_yuv6_pair_distill": 0.5,
                    "dual_ascent_violation__snerv_archive_total_bytes": -0.10,
                    "dual_ascent_update_count__snerv_archive_total_bytes": 1.0,
                    "dual_ascent_violation__snerv_decoder_payload_section_bytes": -0.03,
                    "dual_ascent_update_count__snerv_decoder_payload_section_bytes": 1.0,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        segnet_student_live_calibration_weight=0.0,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=True,
        train_time_section_byte_control_bound=True,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is True
    assert contract["archive_byte_dual_positive_violation_observed"] is False
    assert contract["archive_byte_dual_update_observed"] is True
    assert contract["section_byte_dual_positive_violation_observed"] is False
    assert contract["section_byte_dual_update_observed"] is True
    assert contract["archive_byte_dual_lambda_active_observed"] is False
    assert contract["section_byte_dual_lambda_active_observed"] is False
    assert contract["blockers"] == []


def test_score_aware_telemetry_contract_rejects_section_rate_without_section_dual(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_distill": 1.0,
                    "loss_part_pose_distill": 2.0,
                    "train_time_archive_rate_score": 0.02,
                    "train_time_section_rate_score__decoder_payload": 0.01,
                    "dual_ascent_missing_metric__snerv_segnet_last_frame_distill": 0.0,
                    "dual_ascent_missing_metric__snerv_posenet_yuv6_pair_distill": 0.0,
                    "dual_ascent_lambda__snerv_segnet_last_frame_distill": 0.25,
                    "dual_ascent_lambda__snerv_posenet_yuv6_pair_distill": 0.5,
                    "dual_ascent_lambda__snerv_archive_total_bytes": 0.375,
                    "dual_ascent_violation__snerv_decoder_payload_section_bytes": 0.004,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        segnet_student_live_calibration_weight=0.0,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=True,
        train_time_section_byte_control_bound=True,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is False
    assert contract["section_rate_metric_observed"] is True
    assert contract["section_byte_dual_positive_violation_observed"] is True
    assert contract["section_byte_dual_lambda_active_observed"] is False
    assert (
        "snerv_score_aware_long_training_section_byte_dual_lambda_never_active"
        in contract["blockers"]
    )


def test_score_aware_telemetry_contract_rejects_missing_temporal_floor_ratios(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_posenet_temporal_signal_floor": 0.5,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
        posenet_temporal_signal_floor_weight=0.5,
    )

    assert contract["passed"] is False
    assert contract["posenet_temporal_signal_floor_metric_observed"] is True
    assert contract["posenet_temporal_signal_floor_std_ratio_metric_observed"] is False
    assert (
        contract["posenet_temporal_signal_floor_mean_abs_ratio_metric_observed"]
        is False
    )
    assert (
        "snerv_score_aware_long_training_posenet_temporal_signal_floor_std_ratio_metric_missing"
        in contract["blockers"]
    )
    assert (
        "snerv_score_aware_long_training_posenet_temporal_signal_floor_mean_abs_ratio_metric_missing"
        in contract["blockers"]
    )


def test_score_aware_telemetry_contract_rejects_dual_lambda_without_weight_application(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "train_time_archive_rate_score": 0.02,
                    "train_time_section_rate_score__decoder_payload": 0.01,
                    "dual_ascent_lambda__snerv_archive_total_bytes": 0.375,
                    "dual_ascent_violation__snerv_archive_total_bytes": 0.125,
                    "dual_ascent_lambda__snerv_decoder_payload_section_bytes": 0.125,
                    "dual_ascent_violation__snerv_decoder_payload_section_bytes": 0.125,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=True,
        train_time_section_byte_control_bound=True,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is False
    assert contract["archive_byte_dual_lambda_active_observed"] is True
    assert contract["archive_byte_dual_weight_applied_observed"] is False
    assert contract["section_byte_dual_lambda_active_observed"] is True
    assert contract["section_byte_dual_weight_applied_observed"] is False
    assert (
        "snerv_score_aware_long_training_archive_byte_dual_weight_never_applied"
        in contract["blockers"]
    )
    assert (
        "snerv_score_aware_long_training_section_byte_dual_weight_never_applied"
        in contract["blockers"]
    )


def test_score_aware_telemetry_contract_rejects_stale_gradient_multiplier(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "gradient_multiplier_requested_control_count": 1.0,
                    "gradient_multiplier_applied_leaf_count": 0.0,
                    "gradient_multiplier_missing_requested_count": 1.0,
                    "gradient_multiplier_requested_but_unapplied": 1.0,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
        gradient_multiplier_controls_requested=True,
    )

    assert contract["passed"] is False
    assert contract["gradient_multiplier_requested_observed"] is True
    assert contract["gradient_multiplier_applied_observed"] is False
    assert contract["gradient_multiplier_missing_requested_observed"] is True
    assert contract["gradient_multiplier_noop_observed"] is True
    assert (
        "snerv_score_aware_long_training_gradient_multiplier_never_applied"
        in contract["blockers"]
    )
    assert (
        "snerv_score_aware_long_training_gradient_multiplier_missing_requested_leaf"
        in contract["blockers"]
    )
    assert (
        "snerv_score_aware_long_training_gradient_multiplier_requested_but_unapplied"
        in contract["blockers"]
    )


def test_score_aware_telemetry_contract_rejects_guard_loss_without_guard_dual(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_scorer_input_distribution_guard": 0.25,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=1.0,
    )

    assert contract["passed"] is False
    assert contract["scorer_input_guard_metric_observed"] is True
    assert contract["scorer_input_guard_dual_metric_observed"] is False
    assert (
        "snerv_score_aware_long_training_dual_scorer_input_guard_metric_never_observed"
        in contract["blockers"]
    )


def test_score_aware_telemetry_contract_rejects_missing_direct_live_metrics(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_distill": 1.0,
                    "loss_part_pose_distill": 2.0,
                    "dual_ascent_missing_metric__snerv_segnet_last_frame_distill": 0.0,
                    "dual_ascent_missing_metric__snerv_posenet_yuv6_pair_distill": 0.0,
                    "dual_ascent_lambda__snerv_segnet_last_frame_distill": 0.25,
                    "dual_ascent_lambda__snerv_posenet_yuv6_pair_distill": 0.5,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.25,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is False
    assert (
        "snerv_score_aware_long_training_direct_live_segnet_loss_missing"
        in contract["blockers"]
    )
    assert (
        "snerv_score_aware_long_training_direct_live_segnet_argmax_metric_missing"
        in contract["blockers"]
    )
    assert (
        "snerv_score_aware_long_training_direct_live_segnet_class_occupancy_metric_missing"
        in contract["blockers"]
    )


def test_score_aware_telemetry_contract_rejects_missing_contrast_floor_ratios(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_scorer_input_contrast_floor": 0.25,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.0,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
        scorer_input_contrast_floor_weight=0.875,
    )

    assert contract["passed"] is False
    assert contract["scorer_input_contrast_floor_metric_observed"] is True
    assert (
        "snerv_score_aware_long_training_scorer_input_contrast_floor_segnet_ratio_metric_missing"
        in contract["blockers"]
    )
    assert (
        "snerv_score_aware_long_training_scorer_input_contrast_floor_posenet_ratio_metric_missing"
        in contract["blockers"]
    )


def test_score_aware_telemetry_contract_rejects_missing_shape_tether_submetrics(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_scorer_input_shape_tether": 0.25,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.0,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
        scorer_input_shape_tether_weight=0.625,
    )

    assert contract["passed"] is False
    assert contract["scorer_input_shape_tether_metric_observed"] is True
    assert (
        "snerv_score_aware_long_training_scorer_input_shape_tether_segnet_metric_missing"
        in contract["blockers"]
    )
    assert (
        "snerv_score_aware_long_training_scorer_input_shape_tether_posenet_pair_metric_missing"
        in contract["blockers"]
    )
    assert (
        "snerv_score_aware_long_training_scorer_input_shape_tether_posenet_delta_metric_missing"
        in contract["blockers"]
    )


def test_score_aware_telemetry_contract_rejects_direct_live_class_collapse(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "epoch": 0,
                "loss_components": {
                    "loss_part_segnet_direct_live_distill": 2.0,
                    "loss_part_segnet_direct_live_argmax_disagreement": 0.5,
                    "loss_part_segnet_direct_live_candidate_occupied_class_fraction": 0.4,
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=0.0,
        pose_distillation_weight=0.0,
        segnet_student_live_calibration_weight=0.0,
        segnet_direct_live_distillation_weight=0.25,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is False
    assert contract[
        "segnet_direct_live_max_candidate_occupied_class_fraction"
    ] == pytest.approx(0.4)
    assert (
        "snerv_score_aware_long_training_direct_live_segnet_candidate_argmax_collapsed"
        in contract["blockers"]
    )


def test_score_aware_telemetry_contract_rejects_inactive_scorer_tether_lambdas(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        "\n".join(
            json.dumps(row, sort_keys=True)
            for row in (
                {
                    "epoch": 0,
                    "loss_part_distill": 1.0,
                    "loss_part_pose_distill": 2.0,
                    "dual_ascent_missing_metric__snerv_segnet_last_frame_distill": 0.0,
                    "dual_ascent_missing_metric__snerv_posenet_yuv6_pair_distill": 0.0,
                    "dual_ascent_lambda__snerv_segnet_last_frame_distill": 0.0,
                    "dual_ascent_lambda__snerv_posenet_yuv6_pair_distill": 0.0,
                },
                {
                    "epoch": 1,
                    "loss_part_distill": 1.1,
                    "loss_part_pose_distill": 2.1,
                    "dual_ascent_missing_metric__snerv_segnet_last_frame_distill": 0.0,
                    "dual_ascent_missing_metric__snerv_posenet_yuv6_pair_distill": 0.0,
                    "dual_ascent_lambda__snerv_segnet_last_frame_distill": 0.0,
                    "dual_ascent_lambda__snerv_posenet_yuv6_pair_distill": 0.0,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        segnet_student_live_calibration_weight=0.0,
        pr95_faithful_curriculum_enabled=False,
        coder_aware_qat_bound=False,
        train_time_section_byte_control_bound=False,
        scorer_input_distribution_guard_weight=0.0,
    )

    assert contract["passed"] is False
    assert contract["segnet_dual_metric_observed"] is True
    assert contract["posenet_dual_metric_observed"] is True
    assert contract["segnet_dual_lambda_active_observed"] is False
    assert contract["posenet_dual_lambda_active_observed"] is False
    assert (
        "snerv_score_aware_long_training_dual_segnet_metric_never_observed"
        not in contract["blockers"]
    )
    assert (
        "snerv_score_aware_long_training_dual_posenet_metric_never_observed"
        not in contract["blockers"]
    )
    assert "snerv_score_aware_long_training_dual_segnet_lambda_never_active" in (
        contract["blockers"]
    )
    assert "snerv_score_aware_long_training_dual_posenet_lambda_never_active" in (
        contract["blockers"]
    )


def test_score_aware_telemetry_contract_rejects_stale_pr95_alias_failure(
    tmp_path: Path,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    telemetry = tmp_path / "telemetry.jsonl"
    telemetry.write_text(
        json.dumps(
            {
                "loss_part_pr95_stage_seg_surrogate": 5.0,
                "loss_part_pr95_stage_pose_surrogate": 0.5,
                "dual_ascent_missing_metric__snerv_segnet_last_frame_distill": 1.0,
                "dual_ascent_missing_metric__snerv_posenet_yuv6_pair_distill": 1.0,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    contract = mod._snerv_score_aware_long_training_telemetry_contract(
        telemetry,
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        segnet_student_live_calibration_weight=1.0,
        pr95_faithful_curriculum_enabled=True,
        coder_aware_qat_bound=True,
        train_time_section_byte_control_bound=True,
        scorer_input_distribution_guard_weight=2.0,
    )

    assert contract["passed"] is False
    assert "snerv_score_aware_long_training_pr95_seg_alias_missing" in contract[
        "blockers"
    ]
    assert "snerv_score_aware_long_training_pr95_pose_alias_missing" in contract[
        "blockers"
    ]
    assert (
        "snerv_score_aware_long_training_dual_segnet_metric_never_observed"
        in contract["blockers"]
    )
    assert "snerv_score_aware_long_training_dual_segnet_lambda_never_active" in (
        contract["blockers"]
    )
    assert "snerv_score_aware_long_training_dual_posenet_lambda_never_active" in (
        contract["blockers"]
    )
    assert (
        "snerv_score_aware_long_training_section_rate_metric_missing"
        in contract["blockers"]
    )
    assert (
        "snerv_score_aware_long_training_live_segnet_calibration_never_active"
        in contract["blockers"]
    )
    assert (
        "snerv_score_aware_long_training_live_segnet_calibration_loss_missing"
        in contract["blockers"]
    )


def test_snerv_archive_section_qat_policy_prices_decoder_and_lf_latents() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    policy = mod._build_snerv_pretraining_archive_section_qat_weight_policy(
        pairs_nchw255=_tiny_pairs(pairs=2),
        model_size=SnervModelSizeConfig(fc_dim=4, emb_size=1, patch_radius=1),
        levels=1,
        wavelet="haar",
        source_pair_indices=(3, 7),
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="spatial_delta_zigzag_leb128_lzma",
        recon_pixel_weight=None,
        recon_pixel_weight_metadata=None,
        hf_decoder_saliency_gain=1.0,
        hard_byte_ceiling=10_000,
        base_qat_weights={
            "coder_qat_quant_residual": 1.0e-3,
            "coder_qat_magnitude": 2.0e-4,
        },
    )

    assert policy["schema"] == mod.SNERV_ARCHIVE_SECTION_QAT_WEIGHT_POLICY_SCHEMA
    assert policy["active"] is True
    assert policy["baseline_packet_bytes"] > 0
    assert policy["blockers"] == []
    assert policy["decoder_section_bytes"] > 0
    assert policy["lf_section_bytes"] > 0
    assert policy["extra_loss_weights"]["coder_qat_quant_residual"] >= 1.0e-3
    assert "latent_qat_quant_residual" in policy["extra_loss_weights"]
    assert "latent_qat_magnitude" in policy["extra_loss_weights"]
    assert {
        row["section_name"]: row["operator"]
        for row in policy["applied_section_operators"]
    } == {
        "decoder_payload": "decoder_coder_qat_loss_weight_scaling",
        "lf_payload": "lf_latent_coder_qat_loss_weight_scaling",
    }
    pending = {row["section_name"] for row in policy["pending_section_operators"]}
    assert {"metadata_payload", "step_map_packet"}.issubset(pending)


def test_snerv_train_time_section_byte_control_binds_decoder_and_lf_only() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    policy = mod._build_snerv_pretraining_archive_section_qat_weight_policy(
        pairs_nchw255=_tiny_pairs(pairs=2),
        model_size=SnervModelSizeConfig(fc_dim=4, emb_size=1, patch_radius=1),
        levels=1,
        wavelet="haar",
        source_pair_indices=(0, 1),
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="spatial_delta_zigzag_leb128_lzma",
        recon_pixel_weight=None,
        recon_pixel_weight_metadata=None,
        hf_decoder_saliency_gain=1.0,
        hard_byte_ceiling=128,
        base_qat_weights={
            "coder_qat_quant_residual": 1.0e-3,
            "coder_qat_magnitude": 2.0e-4,
        },
    )

    control = mod._build_snerv_train_time_section_byte_control(
        policy,
        policy["extra_loss_weights"],
        hard_byte_ceiling=128,
    )

    assert control["schema"] == "snerv_train_time_section_byte_control.v1"
    assert control["active"] is True
    assert set(control["section_byte_budgets"]) == {
        "decoder_payload",
        "lf_payload",
    }
    assert control["section_byte_loss_weight_key_map"] == {
        "decoder_payload": "coder_qat_quant_residual",
        "lf_payload": "latent_qat_quant_residual",
    }
    assert control["metrics_payload"]["archive_bytes"] == policy[
        "baseline_packet_bytes"
    ]
    assert control["metrics_payload"]["rate_score_per_byte"] == pytest.approx(
        mod.SNERV_CONTEST_RATE_SCORE_PER_BYTE
    )
    assert control["metrics_payload"]["section_rate_scores"]["decoder_payload"] == (
        pytest.approx(
            control["section_bytes"]["decoder_payload"]
            * mod.SNERV_CONTEST_RATE_SCORE_PER_BYTE
        )
    )
    assert control["metrics_payload"][
        "train_time_section_rate_score__decoder_payload"
    ] == pytest.approx(
        control["section_bytes"]["decoder_payload"]
        * mod.SNERV_CONTEST_RATE_SCORE_PER_BYTE
    )
    budget_rows = {row["section_name"]: row for row in control["budget_rows"]}
    assert budget_rows["decoder_payload"]["rate_score"] == pytest.approx(
        control["section_bytes"]["decoder_payload"]
        * mod.SNERV_CONTEST_RATE_SCORE_PER_BYTE
    )
    assert budget_rows["decoder_payload"]["budget_rate_score"] == pytest.approx(
        budget_rows["decoder_payload"]["budget_bytes"]
        * mod.SNERV_CONTEST_RATE_SCORE_PER_BYTE
    )
    pending = {row["section_name"] for row in control["pending_section_rows"]}
    assert {"metadata_payload", "step_map_packet"}.issubset(pending)
    pending_rows = {row["section_name"]: row for row in control["pending_section_rows"]}
    assert pending_rows["metadata_payload"]["rate_score"] == pytest.approx(
        pending_rows["metadata_payload"]["bytes"]
        * mod.SNERV_CONTEST_RATE_SCORE_PER_BYTE
    )
    assert control["blockers"] == []


def test_snerv_official_section_qat_leaves_dummy_lf_non_actuated() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    official_model_size = SnervModelSizeConfig(
        fc_dim=4,
        emb_size=1,
        patch_radius=1,
        adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        official_skip_high_mode="channel_mean",
    )
    policy = mod._build_snerv_pretraining_archive_section_qat_weight_policy(
        pairs_nchw255=_tiny_pairs(pairs=2),
        model_size=official_model_size,
        levels=1,
        wavelet="haar",
        source_pair_indices=(0, 1),
        target_bits_per_coeff=1.5,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="spatial_delta_zigzag_leb128_lzma",
        recon_pixel_weight=None,
        recon_pixel_weight_metadata=None,
        hf_decoder_saliency_gain=1.0,
        hard_byte_ceiling=128,
        base_qat_weights={
            "coder_qat_quant_residual": 1.0e-3,
            "coder_qat_magnitude": 2.0e-4,
        },
    )

    assert policy["active"] is True
    assert policy["blockers"] == []
    assert policy["decoder_section_bytes"] > 0
    assert policy["lf_section_bytes"] > 0
    assert "coder_qat_quant_residual" in policy["extra_loss_weights"]
    assert "latent_qat_quant_residual" not in policy["extra_loss_weights"]
    assert policy["non_actuated_section_names"] == ["lf_payload"]
    assert policy["non_actuated_section_reasons"] == {
        "lf_payload": "official_payload_frame_decode_uses_decoder_payload_dummy_lf_member"
    }
    pending_policy = {
        str(row["section_name"]): row for row in policy["pending_section_operators"]
    }
    assert pending_policy["lf_payload"]["current_status"] == (
        "not_train_time_actuated"
    )

    control = mod._build_snerv_train_time_section_byte_control(
        policy,
        policy["extra_loss_weights"],
        hard_byte_ceiling=128,
    )

    assert control["active"] is True
    assert control["blockers"] == []
    assert set(control["section_byte_budgets"]) == {"decoder_payload"}
    assert control["section_byte_loss_weight_key_map"] == {
        "decoder_payload": "coder_qat_quant_residual"
    }
    pending_control = {
        str(row["section_name"]): row for row in control["pending_section_rows"]
    }
    assert pending_control["lf_payload"]["current_status"] == (
        "not_train_time_actuated"
    )


def test_snerv_train_time_section_byte_control_prices_pending_without_ceiling() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    policy = mod._build_snerv_pretraining_archive_section_qat_weight_policy(
        pairs_nchw255=_tiny_pairs(pairs=2),
        model_size=SnervModelSizeConfig(fc_dim=4, emb_size=1, patch_radius=1),
        levels=1,
        wavelet="haar",
        source_pair_indices=(0, 1),
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="spatial_delta_zigzag_leb128_lzma",
        recon_pixel_weight=None,
        recon_pixel_weight_metadata=None,
        hf_decoder_saliency_gain=1.0,
        hard_byte_ceiling=None,
        base_qat_weights={"coder_qat_quant_residual": 1.0e-3},
    )

    control = mod._build_snerv_train_time_section_byte_control(
        policy,
        policy["extra_loss_weights"],
        hard_byte_ceiling=None,
    )

    assert control["active"] is False
    assert control["blockers"] == [
        "snerv_train_time_section_byte_hard_ceiling_missing"
    ]
    assert control["metrics_payload"]["rate_score_per_byte"] == pytest.approx(
        mod.SNERV_CONTEST_RATE_SCORE_PER_BYTE
    )
    pending_rows = {row["section_name"]: row for row in control["pending_section_rows"]}
    assert {"decoder_payload", "lf_payload", "metadata_payload"}.issubset(
        pending_rows
    )
    assert pending_rows["decoder_payload"]["rate_score"] == pytest.approx(
        pending_rows["decoder_payload"]["bytes"]
        * mod.SNERV_CONTEST_RATE_SCORE_PER_BYTE
    )
    assert pending_rows["metadata_payload"]["rate_score"] == pytest.approx(
        pending_rows["metadata_payload"]["bytes"]
        * mod.SNERV_CONTEST_RATE_SCORE_PER_BYTE
    )


def test_snerv_live_section_byte_metrics_callback_refreshes_current_submission_packet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    calls: list[np.ndarray] = []

    class FakeModel:
        def render_pairs_nchw255(self, *, batch_size: int) -> np.ndarray:
            value = 10.0 + len(calls)
            rendered = np.full((2, 2, 3, 16, 16), value, dtype=np.float32)
            calls.append(rendered.copy())
            assert batch_size == 2
            return rendered

    def fake_packet_builder(pairs_nchw255: np.ndarray, **kwargs) -> SnervArchivePacket:
        packet_index = len(calls)
        assert np.asarray(pairs_nchw255).shape == (2, 2, 3, 16, 16)
        assert kwargs["native_mlx_decoder_train_steps"] == 0
        packet = f"SNAR-live-{packet_index}".encode("ascii")
        return SnervArchivePacket(
            packet=packet,
            schema="snerv_inverse_steg_archive.v1",
            section_order=(
                "metadata_payload",
                "lf_payload",
                "decoder_payload",
                "step_map_packet",
            ),
            section_bytes={
                "metadata_payload": 11,
                "lf_payload": 100 + packet_index,
                "decoder_payload": 200 + packet_index,
                "step_map_packet": 3,
            },
            section_sha256={},
            section_reports={},
            metadata={"packet_index": packet_index},
            header_bytes=7,
            total_bytes=314 + packet_index,
        )

    monkeypatch.setattr(
        mod,
        "build_snerv_mlx_native_packet_from_numpy_pairs",
        fake_packet_builder,
    )

    def fake_submission_repack(
        packet: bytes,
        *,
        submission_archive_format: str,
    ) -> tuple[bytes, dict[str, object]]:
        assert submission_archive_format == "snar2"
        packet_index = int(packet.decode("ascii").rsplit("-", 1)[1])
        out = b"S" * (1000 + packet_index)
        return out, {
            "schema": "snerv_submission_archive_repack.v1",
            "requested_archive_format": submission_archive_format,
            "input_packet_bytes": len(packet),
            "input_packet_sha256": hashlib.sha256(packet).hexdigest(),
            "output_packet_schema": "snerv_inverse_steg_archive.snar2.v1",
            "output_packet_bytes": len(out),
            "output_packet_sha256": hashlib.sha256(out).hexdigest(),
            "repacked": True,
            "bytes_saved": -1,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(mod, "_snerv_submission_packet_for_export", fake_submission_repack)
    callback, metadata = mod._build_snerv_live_train_time_section_byte_metrics_callback(
        model_size=SnervModelSizeConfig(fc_dim=4, emb_size=1, patch_radius=1),
        levels=1,
        wavelet="haar",
        source_pair_indices=(0, 1),
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="spatial_delta_zigzag_leb128_lzma",
        recon_pixel_weight=None,
        recon_pixel_weight_metadata=None,
        hf_decoder_saliency_gain=1.0,
        train_time_section_byte_control={
            "active": True,
            "section_byte_budgets": {"decoder_payload": 5},
            "section_byte_loss_weight_key_map": {
                "decoder_payload": "coder_qat_quant_residual"
            },
            "metrics_payload": {
                "schema": "snerv_train_time_section_byte_metrics.v1",
                "archive_bytes": 10,
                "section_bytes": {"decoder_payload": 5},
            }
        },
        batch_size=2,
        refresh_every_steps=2,
    )

    assert callback is not None
    first = dict(callback(FakeModel(), None, {}))
    second = dict(callback(FakeModel(), None, {}))
    third = dict(callback(FakeModel(), None, {}))

    assert metadata["submission_archive_format"] == "snar2"
    assert first["archive_bytes"] == 1001
    assert first["byte_basis"] == "current_receiver_submission_packet_sections"
    assert first["submission_archive_format"] == "snar2"
    assert first["live_profile"]["packet_bytes_before_submission_repack"] == 315
    assert first["live_profile"]["submission_packet_bytes"] == 1001
    assert first["live_profile"]["submission_packet_schema"] == (
        "snerv_inverse_steg_archive.snar2.v1"
    )
    assert first["section_bytes"]["decoder_payload"] == 201
    assert first["rate_score_per_byte"] == pytest.approx(
        mod.SNERV_CONTEST_RATE_SCORE_PER_BYTE
    )
    assert first["section_rate_scores"]["decoder_payload"] == pytest.approx(
        201 * mod.SNERV_CONTEST_RATE_SCORE_PER_BYTE
    )
    assert first["train_time_section_rate_score__decoder_payload"] == pytest.approx(
        201 * mod.SNERV_CONTEST_RATE_SCORE_PER_BYTE
    )
    assert second == first
    assert third["archive_bytes"] == 1002
    assert third["live_profile"]["packet_bytes_before_submission_repack"] == 316
    assert third["section_bytes"]["lf_payload"] == 102
    assert third["train_time_section_rate_score__lf_payload"] == pytest.approx(
        102 * mod.SNERV_CONTEST_RATE_SCORE_PER_BYTE
    )
    assert third[
        "train_time_section_rate_score__metadata_payload"
    ] == pytest.approx(11 * mod.SNERV_CONTEST_RATE_SCORE_PER_BYTE)
    assert third["train_time_section_rate_score__step_map_packet"] == pytest.approx(
        3 * mod.SNERV_CONTEST_RATE_SCORE_PER_BYTE
    )
    assert len(calls) == 2
    assert metadata["active"] is True
    assert metadata["refresh_calls"] == 2
    assert metadata["cache_hits"] == 1
    assert metadata["last_section_bytes"]["decoder_payload"] == 202
    assert sorted(
        key
        for key, value in third.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    ) == [
        "archive_bytes",
        "rate_score_per_byte",
        "train_time_section_rate_score__decoder_payload",
        "train_time_section_rate_score__lf_payload",
        "train_time_section_rate_score__metadata_payload",
        "train_time_section_rate_score__step_map_packet",
    ]


def test_snerv_live_section_byte_metrics_callback_uses_official_components(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    captured: dict[str, object] = {}
    official_model_size = SnervModelSizeConfig(
        fc_dim=4,
        adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
    )

    class FakeOfficialModel:
        def export_official_components(self) -> dict[str, np.ndarray]:
            captured["export_official_components_called"] = True
            return {"low": np.zeros((1, 6, 8, 8), dtype=np.float32)}

        def render_pairs_nchw255(self, *, batch_size: int) -> np.ndarray:
            raise AssertionError(
                "official live byte metrics must not refit rendered pairs"
            )

    def fake_official_packet_builder(
        components: Mapping[str, object],
        **kwargs,
    ) -> SnervArchivePacket:
        captured["components"] = components
        captured["source_pair_indices"] = kwargs["source_pair_indices"]
        captured["model_size_adapter"] = kwargs["model_size"].adapter
        return SnervArchivePacket(
            packet=b"SNAR-official-live",
            schema="snerv_inverse_steg_archive.v1",
            section_order=(
                "metadata_payload",
                "lf_payload",
                "decoder_payload",
                "step_map_packet",
            ),
            section_bytes={
                "metadata_payload": 9,
                "decoder_payload": 321,
            },
            section_sha256={},
            section_reports={},
            metadata={"official": True},
            header_bytes=5,
            total_bytes=400,
        )

    monkeypatch.setattr(
        mod,
        "_build_official_mfu_hfr_tub_packet_from_components",
        fake_official_packet_builder,
    )

    def fake_submission_repack(
        packet: bytes,
        *,
        submission_archive_format: str,
    ) -> tuple[bytes, dict[str, object]]:
        assert submission_archive_format == "snar2"
        out = b"O" * 377
        return out, {
            "schema": "snerv_submission_archive_repack.v1",
            "requested_archive_format": submission_archive_format,
            "input_packet_bytes": len(packet),
            "input_packet_sha256": hashlib.sha256(packet).hexdigest(),
            "output_packet_schema": "snerv_inverse_steg_archive.snar2.v1",
            "output_packet_bytes": len(out),
            "output_packet_sha256": hashlib.sha256(out).hexdigest(),
            "repacked": True,
            "bytes_saved": 23,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(mod, "_snerv_submission_packet_for_export", fake_submission_repack)
    callback, metadata = mod._build_snerv_live_train_time_section_byte_metrics_callback(
        model_size=official_model_size,
        levels=1,
        wavelet="haar",
        source_pair_indices=(3,),
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="spatial_delta_zigzag_leb128_lzma",
        recon_pixel_weight=None,
        recon_pixel_weight_metadata=None,
        hf_decoder_saliency_gain=1.0,
        train_time_section_byte_control={
            "metrics_payload": {
                "schema": "snerv_train_time_section_byte_metrics.v1",
                "archive_bytes": 10,
                "section_bytes": {"decoder_payload": 5},
            }
        },
        batch_size=1,
        refresh_every_steps=1,
    )

    assert callback is not None
    payload = dict(callback(FakeOfficialModel(), None, {}))
    assert payload["archive_bytes"] == 377
    assert payload["byte_basis"] == "current_receiver_submission_packet_sections"
    assert payload["live_profile"]["packet_bytes_before_submission_repack"] == 400
    assert payload["live_profile"]["submission_packet_bytes"] == 377
    assert payload["section_bytes"] == {
        "decoder_payload": 321,
        "metadata_payload": 9,
    }
    assert payload["live_profile"]["packet_source"] == (
        "current_official_renderer_components"
    )
    assert metadata["packet_builder_scope"] == (
        "official_mfu_hfr_tub_current_component_packet"
    )
    assert metadata["requires_current_official_components"] is True
    assert mod._snerv_live_section_byte_metrics_blockers(
        metadata,
        train_time_section_byte_control_bound=True,
    ) == []
    assert captured["export_official_components_called"] is True
    assert captured["source_pair_indices"] == (3,)
    assert (
        captured["model_size_adapter"]
        == SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER
    )


def test_snerv_live_section_byte_metrics_callback_blocks_official_static_fallback() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    official_model_size = SnervModelSizeConfig(
        fc_dim=4,
        adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
    )

    class FakePartialOfficialModel:
        def render_pairs_nchw255(self, *, batch_size: int) -> np.ndarray:
            raise AssertionError(
                "official live byte metrics must not refit rendered pairs"
            )

    callback, metadata = mod._build_snerv_live_train_time_section_byte_metrics_callback(
        model_size=official_model_size,
        levels=1,
        wavelet="haar",
        source_pair_indices=(3,),
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="spatial_delta_zigzag_leb128_lzma",
        recon_pixel_weight=None,
        recon_pixel_weight_metadata=None,
        hf_decoder_saliency_gain=1.0,
        train_time_section_byte_control={
            "metrics_payload": {
                "schema": "snerv_train_time_section_byte_metrics.v1",
                "archive_bytes": 10,
                "section_bytes": {"decoder_payload": 5},
            }
        },
        batch_size=1,
        refresh_every_steps=1,
    )

    assert callback is not None
    payload = callback(FakePartialOfficialModel(), None, {})

    assert payload is None
    assert metadata["active"] is False
    assert metadata["fallback_count"] == 1
    assert metadata["last_fallback_reason"] == (
        "snerv_official_live_section_byte_export_components_missing"
    )
    assert (
        mod.SNERV_LIVE_SECTION_BYTE_OFFICIAL_COMPONENTS_MISSING_BLOCKER
        in metadata["blockers"]
    )
    assert (
        mod.SNERV_LIVE_SECTION_BYTE_OFFICIAL_FALLBACK_BLOCKER
        in metadata["blockers"]
    )
    assert (
        "snerv_live_section_byte_active_control_static_fallback_forbidden"
        in metadata["blockers"]
    )

    blockers = mod._snerv_live_section_byte_metrics_blockers(
        metadata,
        train_time_section_byte_control_bound=True,
    )
    assert (
        "snerv_official_live_section_byte_export_components_missing"
        in blockers
    )
    assert (
        mod.SNERV_LIVE_SECTION_BYTE_OFFICIAL_COMPONENTS_MISSING_BLOCKER
        in blockers
    )
    assert mod.SNERV_LIVE_SECTION_BYTE_OFFICIAL_FALLBACK_BLOCKER in blockers
    assert (
        mod.SNERV_LIVE_SECTION_BYTE_OFFICIAL_NEVER_REFRESHED_BLOCKER
        in blockers
    )
    assert (
        "snerv_live_section_byte_active_control_static_fallback_forbidden"
        in blockers
    )
    assert (
        "snerv_live_section_byte_active_control_never_refreshed_current_packet"
        in blockers
    )


def test_snerv_live_section_byte_metrics_callback_allows_inactive_diagnostic_fallback() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    class FakeBrokenNativeModel:
        pass

    callback, metadata = mod._build_snerv_live_train_time_section_byte_metrics_callback(
        model_size=SnervModelSizeConfig(fc_dim=4, emb_size=1, patch_radius=1),
        levels=1,
        wavelet="haar",
        source_pair_indices=(3,),
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="spatial_delta_zigzag_leb128_lzma",
        recon_pixel_weight=None,
        recon_pixel_weight_metadata=None,
        hf_decoder_saliency_gain=1.0,
        train_time_section_byte_control={
            "active": False,
            "metrics_payload": {
                "schema": "snerv_train_time_section_byte_metrics.v1",
                "archive_bytes": 10,
                "section_bytes": {"decoder_payload": 5},
            },
        },
        batch_size=1,
        refresh_every_steps=1,
    )

    assert callback is not None
    payload = dict(callback(FakeBrokenNativeModel(), None, {}))

    assert payload["schema"] == "snerv_train_time_section_byte_metrics_fallback.v1"
    assert payload["archive_bytes"] == 10
    assert metadata["fallback_count"] == 1
    assert (
        "snerv_live_section_byte_active_control_static_fallback_forbidden"
        not in payload["blockers"]
    )


def test_snerv_archive_section_qat_policy_fails_closed_on_empty_weights() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    policy = mod._build_snerv_pretraining_archive_section_qat_weight_policy(
        pairs_nchw255=_tiny_pairs(pairs=1),
        model_size=SnervModelSizeConfig(),
        levels=1,
        wavelet="haar",
        source_pair_indices=(0,),
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="spatial_delta_zigzag_leb128_lzma",
        recon_pixel_weight=None,
        recon_pixel_weight_metadata=None,
        hf_decoder_saliency_gain=1.0,
        hard_byte_ceiling=None,
        base_qat_weights={},
    )

    assert policy["active"] is False
    assert "snerv_archive_section_qat_base_weights_empty" in policy["blockers"]


def test_torch_scorer_device_alias_resolves_gpu_for_direct_snerv_export() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    fake_mps = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    assert (
        mod._resolve_torch_scorer_device_alias(
            "gpu",
            torch_module=fake_mps,
        )
        == "mps"
    )
    fake_cuda = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
    )
    assert (
        mod._resolve_torch_scorer_device_alias(
            "gpu",
            torch_module=fake_cuda,
        )
        == "cuda"
    )
    fake_cpu_only = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)),
    )
    with pytest.raises(
        mod.SnervMlxNativeExportError,
        match=r"neither torch\.cuda nor torch\.backends\.mps",
    ):
        mod._resolve_torch_scorer_device_alias(
            "gpu",
            torch_module=fake_cpu_only,
        )


def test_pr95_every_stage_muon_falls_back_when_snerv_has_no_matrix_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=1)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)
    harness_calls: list[dict[str, object]] = []

    telemetry_path = tmp_path / "zero_muon_every_stage_telemetry.jsonl"

    class FakeArtifact:
        def as_dict(self) -> dict[str, object]:
            return {
                "total_epochs_completed": 1,
                "telemetry_path": telemetry_path.as_posix(),
                "live_checkpoint_path": "",
                "ema_shadow_checkpoint_path": "",
            }

    def fake_run_mlx_score_aware_full_main(**kwargs):
        harness_calls.append(kwargs)
        telemetry_path.write_text(
            json.dumps(
                {
                    "epoch": 0,
                    "scorer_space_step_guard_enabled": 1.0,
                    "scorer_space_step_guard_eligible": 1.0,
                    "scorer_space_step_guard_rejected": 0.0,
                    "scorer_space_step_guard_effective_optimizer_learning_rate": 1.0e-3,
                    "scorer_space_step_guard_optimizer_learning_rate_scale": 1.0,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        on_epoch_end = kwargs.get("on_epoch_end")
        if on_epoch_end is not None:
            on_epoch_end(SimpleNamespace(epoch=0, loss=0.0))
        return FakeArtifact()

    monkeypatch.setattr(
        "tac.substrates._shared.mlx_score_aware.harness.run_mlx_score_aware_full_main",
        fake_run_mlx_score_aware_full_main,
    )

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "zero_muon_every_stage",
        num_pairs=1,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int8_symmetric",
            "hard_byte_ceiling": 10_000,
            "score_aware_long_training_epochs": 1,
            "score_aware_long_training_pr95_faithful_curriculum": True,
            "score_aware_long_training_pr95_muon_policy": "every_stage",
            "score_aware_long_training_scorer_input_distribution_guard_weight": 0.0,
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
    )

    long_training = report["score_aware_long_training"]
    assert report["score_aware_long_training_executed"] is True
    assert long_training["executed"] is True
    assert long_training["blockers"] == []
    assert harness_calls
    assert harness_calls[0]["pr95_muon_policy"] == "faithful_stage8_only"
    assert long_training["pr95_muon_policy_requested"] == "every_stage"
    assert long_training["pr95_muon_policy"] == "faithful_stage8_only"
    assert long_training["pr95_optimizer_coverage"]["muon_tensor_count"] == 0
    assert long_training["pr95_optimizer_coverage"]["adamw_tensor_count"] > 0
    assert (
        long_training["pr95_optimizer_coverage"]["pr95_muon_policy_requested"]
        == "every_stage"
    )
    assert (
        long_training["pr95_optimizer_coverage"]["pr95_muon_policy"]
        == "faithful_stage8_only"
    )
    assert (
        long_training["pr95_optimizer_coverage"]["muon_policy_fallback_applied"]
        is True
    )


def test_score_aware_long_training_stage_weights_reach_snerv_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=1)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)

    monkeypatch.setattr(
        mod,
        "decode_mlx_targets",
        lambda *_args, **_kwargs: (target0, target1),
    )
    harness_calls: list[dict[str, object]] = []

    class FakeArtifact:
        def as_dict(self) -> dict[str, object]:
            return {
                "total_epochs_completed": 4,
                "telemetry_path": "",
                "live_checkpoint_path": "",
                "ema_shadow_checkpoint_path": "",
            }

    def fake_run_mlx_score_aware_full_main(**kwargs):
        harness_calls.append(kwargs)
        on_epoch_end = kwargs.get("on_epoch_end")
        if on_epoch_end is not None:
            on_epoch_end(SimpleNamespace(epoch=3, loss=0.0))
        return FakeArtifact()

    monkeypatch.setattr(
        "tac.substrates._shared.mlx_score_aware.harness.run_mlx_score_aware_full_main",
        fake_run_mlx_score_aware_full_main,
    )

    stage_weights = {
        "recon": 0.5,
        "distill": 1.25,
        "pose_distill": 0.75,
        "scorer_input_guard": 0.25,
        "scorer_input_contrast_floor": 0.375,
        "scorer_input_shape_tether": 0.625,
        "posenet_temporal_signal_floor": 0.875,
        "segnet_direct_live_distill": 0.125,
    }
    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "snerv_stage_weights",
        num_pairs=1,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int8_symmetric",
            "score_aware_long_training_epochs": 4,
            "score_aware_long_training_scorer_input_distribution_guard_weight": 0.0,
            "snerv_score_aware_long_training_stage_loss_weights": stage_weights,
            "snerv_score_aware_long_training_pose_warmup_epochs": 2,
            "snerv_score_aware_long_training_scorer_input_shape_warmup_epochs": 1,
            "snerv_score_aware_long_training_segnet_direct_live_escape_warmup_epochs": 1,
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
    )

    assert harness_calls
    stages = harness_calls[0]["curriculum_stages"]
    assert [stage.start_epoch for stage in stages] == [0, 1, 2]
    assert [stage.end_epoch for stage in stages] == [1, 2, 4]
    assert stages[0].loss_weights["pose_distill"] == 0.0
    assert stages[0].loss_weights["segnet_direct_live_distill"] == 0.0
    assert stages[0].loss_weights["segnet_direct_live_base_loss"] == 0.0
    assert stages[1].loss_weights["pose_distill"] == 0.0
    assert stages[1].loss_weights["segnet_direct_live_distill"] == pytest.approx(
        0.125
    )
    assert stages[2].loss_weights["pose_distill"] == pytest.approx(0.75)
    assert stages[2].loss_weights["segnet_direct_live_distill"] == pytest.approx(
        0.125
    )
    assert stages[2].loss_weights["posenet_temporal_signal_floor"] == pytest.approx(
        0.875
    )
    long_training = report["score_aware_long_training"]
    assert long_training["stage_loss_weights"] == {
        **stage_weights,
        "pose_direct_live_distill": 0.75,
        "segnet_direct_live_class_histogram": 0.125,
        "segnet_direct_live_class_balanced_hinge": 0.125,
        "segnet_direct_live_class_balanced_ce": 0.125,
        "segnet_direct_live_class_balanced_squared_hinge": 0.125,
        "segnet_direct_live_class_region_recon": 0.125,
        "segnet_direct_live_rare_class_logit": 0.125,
        "segnet_direct_live_target_mass_floor": 0.125,
        "segnet_direct_live_target_min_ratio_floor": 0.125,
        "posenet_yuv6_geometry_tether": 0.25,
    }
    assert long_training["curriculum_warmup_epochs"] == {
        "pose_distillation_warmup_epochs": 2,
        "scorer_input_shape_warmup_epochs": 1,
        "segnet_direct_live_escape_warmup_epochs": 1,
        "segnet_direct_live_escape_class_multiplier": 1.0,
    }
    assert long_training["curriculum_stage_count"] == 3


def test_score_aware_step_guard_pose_yuv6_candidate_controls_reach_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=1)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)
    monkeypatch.setattr(
        mod,
        "decode_mlx_targets",
        lambda *_args, **_kwargs: (target0, target1),
    )

    harness_calls: list[dict[str, object]] = []

    class FakeArtifact:
        def as_dict(self) -> dict[str, object]:
            return {
                "total_epochs_completed": 1,
                "telemetry_path": "",
                "live_checkpoint_path": "",
                "ema_shadow_checkpoint_path": "",
            }

    def fake_run_mlx_score_aware_full_main(**kwargs):
        harness_calls.append(kwargs)
        on_epoch_end = kwargs.get("on_epoch_end")
        if on_epoch_end is not None:
            on_epoch_end(SimpleNamespace(epoch=0, loss=0.0))
        return FakeArtifact()

    monkeypatch.setattr(
        "tac.substrates._shared.mlx_score_aware.harness.run_mlx_score_aware_full_main",
        fake_run_mlx_score_aware_full_main,
    )

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "snerv_step_guard_pose_yuv6_candidate_controls",
        num_pairs=1,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int8_symmetric",
            "score_aware_long_training_epochs": 1,
            "score_aware_long_training_scorer_input_distribution_guard_weight": 0.0,
            "score_aware_long_training_scorer_space_step_guard_min_post_segnet_target_class_min_ratio": 0.23,
            "snerv_score_aware_long_training_scorer_space_step_guard_max_post_segnet_target_class_ratio_drop": 0.045,
            "score_aware_long_training_scorer_space_step_guard_max_post_segnet_distribution_mae": 0.11,
            "snerv_score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae": 0.12,
            "score_aware_long_training_scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio": 1.3,
            "snerv_score_aware_long_training_scorer_space_step_guard_max_post_pose_score_term": 2.4,
            "score_aware_long_training_scorer_space_step_guard_max_post_pose_direct_live_score_term": 0.05,
            "snerv_score_aware_long_training_scorer_space_step_guard_max_pose_score_term_relative_worsening": 0.06,
            "score_aware_long_training_scorer_space_step_guard_max_pose_score_term_absolute_worsening": 0.07,
            "snerv_score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening": 0.08,
            "score_aware_long_training_scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening": 0.09,
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
    )

    assert report["score_aware_long_training_executed"] is True
    assert harness_calls
    harness_kwargs = harness_calls[0]
    assert harness_kwargs[
        "scorer_space_step_guard_min_post_segnet_target_class_min_ratio"
    ] == pytest.approx(0.23)
    assert harness_kwargs[
        "scorer_space_step_guard_max_post_segnet_target_class_ratio_drop"
    ] == pytest.approx(0.045)
    assert harness_kwargs["scorer_space_step_guard_max_post_segnet_distribution_mae"] == pytest.approx(
        0.11
    )
    assert harness_kwargs[
        "scorer_space_step_guard_max_post_posenet_yuv6_distribution_mae"
    ] == pytest.approx(0.12)
    assert harness_kwargs[
        "scorer_space_step_guard_max_post_posenet_yuv6_contrast_ratio"
    ] == pytest.approx(1.3)
    assert harness_kwargs["scorer_space_step_guard_max_post_pose_score_term"] == pytest.approx(
        2.4
    )
    assert harness_kwargs[
        "scorer_space_step_guard_max_post_pose_direct_live_score_term"
    ] == pytest.approx(0.05)
    assert harness_kwargs[
        "scorer_space_step_guard_max_pose_score_term_relative_worsening"
    ] == pytest.approx(0.06)
    assert harness_kwargs[
        "scorer_space_step_guard_max_pose_score_term_absolute_worsening"
    ] == pytest.approx(0.07)
    assert harness_kwargs[
        "scorer_space_step_guard_max_pose_direct_live_score_term_relative_worsening"
    ] == pytest.approx(0.08)
    assert harness_kwargs[
        "scorer_space_step_guard_max_pose_direct_live_score_term_absolute_worsening"
    ] == pytest.approx(0.09)


def test_packet_builder_emits_receiver_decodable_snar1() -> None:
    packet = build_snerv_mlx_native_packet_from_numpy_pairs(
        _tiny_pairs(pairs=2),
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="auto",
    )

    decoded = unpack_snerv_archive(packet.packet)
    frames = decode_snerv_archive_frames(packet.packet)

    assert decoded.metadata["wavelet"] == "haar"
    assert decoded.metadata["lf_plane_count"] == 12
    assert decoded.metadata["allocation_mode"] == "uniform_mlx_native_closed_form_export"
    assert decoded.metadata["step_map_packet_schema"] == ("snerv_step_map_coder.adaptive.v1")
    assert decoded.metadata["step_map_coder_mode"] == ("waterfill_mlx_native_uniform_importance_bridge")
    assert decoded.metadata["contest_scorer_distortion_objective"] is False
    assert decoded.metadata["score_aware_hf_decoder_fit_executed"] is False
    assert decoded.metadata["score_aware_long_training_executed"] is False
    assert decoded.metadata["step_map_waterfill_bits_per_coeff"] == pytest.approx(0.5)
    assert decoded.metadata["step_map_coder_groups"]
    assert decoded.metadata["lf_payload_codec_requested"] == "auto"
    assert decoded.metadata["lf_payload_codec"] == decoded.metadata[
        "lf_payload_codec_selected"
    ]
    assert decoded.metadata["lf_payload_codec_selected"] != "auto"
    assert decoded.metadata["lf_payload_codec_selection_report"]["section_bytes"] == (
        packet.section_bytes["lf_payload"]
    )
    assert frames.shape == (2, 2, 3, 16, 16)
    assert np.isfinite(frames).all()
    assert packet.score_claim is False


def test_mlx_target_hydration_selects_arbitrary_pair_indices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.data as data_mod
    import tac.substrates._shared.mlx_score_aware.targets as target_mod

    class FakeFrame:
        def __init__(self, frame_idx: int) -> None:
            self._array = np.full((4, 5, 3), frame_idx, dtype=np.float32)

        def numpy(self) -> np.ndarray:
            return self._array

    seen: dict[str, object] = {}

    def fake_decode_video(*_args, **kwargs):
        seen.update(kwargs)
        return [FakeFrame(idx) for idx in range(int(kwargs["max_frames"]))]

    monkeypatch.setattr(target_mod, "require_mlx_for_harness", lambda: mx)
    monkeypatch.setattr(data_mod, "decode_video", fake_decode_video)

    target0, target1 = target_mod.decode_mlx_targets(
        "unit.mkv",
        num_pairs=2,
        output_height=4,
        output_width=5,
        pair_indices=[3, 1, 3],
    )

    assert seen["max_frames"] == 8
    np.testing.assert_allclose(np.asarray(target0)[:, 0, 0, 0], [6 / 255.0, 2 / 255.0])
    np.testing.assert_allclose(np.asarray(target1)[:, 0, 0, 0], [7 / 255.0, 3 / 255.0])


def test_mlx_target_hydration_rejects_pair_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates._shared.mlx_score_aware.targets as target_mod

    monkeypatch.setattr(target_mod, "require_mlx_for_harness", lambda: mx)

    with pytest.raises(target_mod.MlxScoreAwareHarnessError, match="does not match"):
        target_mod.decode_mlx_targets(
            "unit.mkv",
            num_pairs=3,
            output_height=4,
            output_width=5,
            pair_indices=[3, 1, 3],
        )


def test_packet_builder_preserves_explicit_source_pair_indices() -> None:
    packet = build_snerv_mlx_native_packet_from_numpy_pairs(
        _tiny_pairs(pairs=2),
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="auto",
        source_pair_indices=[7, 2, 7],
    )

    decoded = unpack_snerv_archive(packet.packet)

    assert decoded.metadata["n_pairs"] == 2
    assert decoded.metadata["source_pair_indices"] == [7, 2]
    assert decoded.metadata["source_pair_indices_preserved"] is True
    assert decoded.metadata["pair_index_alignment_mode"] == ("explicit_source_pair_indices")
    rows = decoded.metadata["lf_step_allocation_rows"]
    assert {row["pair_idx"] for row in rows[:6]} == {0}
    assert {row["source_pair_idx"] for row in rows[:6]} == {7}
    assert {row["pair_idx"] for row in rows[6:12]} == {1}
    assert {row["source_pair_idx"] for row in rows[6:12]} == {2}


def test_packet_builder_defaults_to_portfolio_lf_payload_codec() -> None:
    packet = build_snerv_mlx_native_packet_from_numpy_pairs(
        _tiny_pairs(pairs=1),
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
    )

    decoded = unpack_snerv_archive(packet.packet)

    assert decoded.metadata["lf_payload_codec_requested"] == "portfolio_auto"
    assert decoded.metadata["lf_payload_codec"] == decoded.metadata[
        "lf_payload_codec_selected"
    ]
    assert decoded.metadata["lf_payload_codec_selected"].startswith("v2:")
    assert packet.section_bytes["lf_payload"] > 0


def test_packet_builder_consumes_joint_recon_pixel_weight_in_decoder_fit() -> None:
    pairs = _tiny_pairs(pairs=1)
    yy, xx = np.mgrid[0:16, 0:16].astype(np.float32)
    pairs[0, 1, 0] += 18.0 * np.sin(xx * 0.9) * np.cos(yy * 0.7)
    pairs[0, 0, 2] += 12.0 * (((xx.astype(np.int32) + yy.astype(np.int32)) % 3) == 0)
    pairs = np.clip(pairs, 0.0, 255.0)
    weight = np.ones((1, 2, 16, 16, 1), dtype=np.float32)
    weight[:, :, 3:11, 4:12, :] = 64.0

    unweighted = build_snerv_mlx_native_packet_from_numpy_pairs(
        pairs,
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="auto",
    )
    weighted = build_snerv_mlx_native_packet_from_numpy_pairs(
        pairs,
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="auto",
        recon_pixel_weight=weight,
        recon_pixel_weight_metadata={"schema": "unit_joint_weight.v1"},
        hf_decoder_saliency_gain=3.0,
    )

    decoded = unpack_snerv_archive(weighted.packet)
    unweighted_decoded = unpack_snerv_archive(unweighted.packet)
    assert decoded.metadata["recon_pixel_weight_consumed"] is True
    assert decoded.metadata["recon_pixel_weight_verified_gradient_manifest"] is False
    assert decoded.metadata["contest_scorer_distortion_objective"] is False
    assert decoded.metadata["allocation_mode"] == ("joint_p18_p19_lf_waterfill_plus_hf_dwt_adjoint_saliency")
    assert decoded.metadata["lf_step_allocation_mode"] == ("joint_p18_p19_dwt_adjoint_lf_reverse_waterfill")
    assert unweighted_decoded.metadata["lf_step_allocation_mode"] == ("uniform_l2_baseline")
    assert decoded.metadata["step_map_coder_mode"] == ("joint_p18_p19_lf_step_map_waterfill")
    assert decoded.metadata["lf_step_allocation_rows"][0]["mode"] == ("joint_p18_p19_dwt_adjoint_lf_reverse_waterfill")
    assert decoded.metadata["hf_decoder_fit_mode"] == (SNERV_DWT_ADJOINT_SALIENCY_WEIGHTED_FIT_MODE)
    assert decoded.metadata["exact_pixel_weighted_objective"] is False
    assert decoded.metadata["hf_decoder_saliency_gain"] == pytest.approx(3.0)
    assert decoded.metadata["recon_pixel_weight_metadata"]["schema"] == ("unit_joint_weight.v1")
    weighted_steps = decoded.decode_step_maps()
    assert any(float(np.std(step)) > 0.0 for step in weighted_steps)
    assert all(row["mode"] == "uniform_l2_baseline" for row in unweighted_decoded.metadata["lf_step_allocation_rows"])
    assert weighted.packet != unweighted.packet
    assert not np.allclose(
        decode_snerv_archive_frames(weighted.packet),
        decode_snerv_archive_frames(unweighted.packet),
    )


def test_packet_builder_runs_native_mlx_hf_decoder_training() -> None:
    pytest.importorskip("mlx.core")
    pairs = _tiny_pairs(pairs=1)
    pairs[0, 1, 0, 2:13, 3:14] += 9.0
    pairs = np.clip(pairs, 0.0, 255.0)

    packet = build_snerv_mlx_native_packet_from_numpy_pairs(
        pairs,
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="auto",
        native_mlx_decoder_train_steps=2,
        native_mlx_decoder_train_lr=1.0e-5,
        native_mlx_decoder_train_ridge=1.0e-6,
    )

    decoded = unpack_snerv_archive(packet.packet)
    training = decoded.metadata["native_mlx_hf_decoder_training"]
    assert training["schema"] == "snerv_native_mlx_hf_decoder_training.v1"
    assert training["executed"] is True
    assert training["optimizer"] == "pact_guarded_adamw"
    assert training["optimizer_backend"] == "mlx.optimizers+guarded_manual_fallback"
    assert training["steps"] == 2
    assert training["learning_rate"] == pytest.approx(1.0e-5)
    assert training["all_final_losses_finite"] is True
    assert training["accepted"] is True
    assert training["any_loss_worsened"] is False
    assert training["optimizer_used_counts"]
    assert training["level_subband_rows"]
    assert decoded.metadata["native_mlx_training_executed"] is True
    assert decoded.metadata["native_mlx_training_kind"] == (
        "hf_decoder_pact_guarded_adamw"
    )
    assert decoded.metadata["hf_decoder_fit_mode"].startswith(
        "native_mlx_pact_guarded_adamw_from_"
    )
    assert training["score_claim"] is False
    assert training["ready_for_exact_eval_dispatch"] is False
    assert packet.score_claim is False


def test_packet_builder_consumes_opt_in_native_mlx_manual_gradient_descent_optimizer() -> None:
    pytest.importorskip("mlx.core")
    pairs = _tiny_pairs(pairs=1)
    pairs[0, 1, 0, 2:13, 3:14] += 9.0
    pairs = np.clip(pairs, 0.0, 255.0)

    packet = build_snerv_mlx_native_packet_from_numpy_pairs(
        pairs,
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="auto",
        native_mlx_decoder_train_steps=2,
        native_mlx_decoder_train_lr=1.0e-5,
        native_mlx_decoder_train_ridge=1.0e-6,
        native_mlx_decoder_train_optimizer="full_batch_gradient_descent",
    )

    decoded = unpack_snerv_archive(packet.packet)
    training = decoded.metadata["native_mlx_hf_decoder_training"]
    assert training["executed"] is True
    assert training["optimizer"] == "full_batch_gradient_descent"
    assert training["optimizer_backend"] == "manual_mlx"
    assert training["optimizer_used_counts"]
    assert decoded.metadata["native_mlx_training_kind"] == (
        "hf_decoder_full_batch_gradient_descent"
    )
    assert decoded.metadata["hf_decoder_fit_mode"].startswith(
        "native_mlx_full_batch_gradient_descent_from_"
    )


def test_packet_builder_rejects_worsening_native_mlx_hf_decoder_training() -> None:
    pytest.importorskip("mlx.core")
    pairs = _tiny_pairs(pairs=1)
    pairs[0, 1, 0, 2:13, 3:14] += 9.0
    pairs = np.clip(pairs, 0.0, 255.0)

    baseline = build_snerv_mlx_native_packet_from_numpy_pairs(
        pairs,
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="auto",
    )
    divergent = build_snerv_mlx_native_packet_from_numpy_pairs(
        pairs,
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="auto",
        native_mlx_decoder_train_steps=2,
        native_mlx_decoder_train_lr=1.0e6,
        native_mlx_decoder_train_ridge=1.0e-6,
        native_mlx_decoder_train_optimizer="adamw",
    )

    baseline_decoded = unpack_snerv_archive(baseline.packet)
    divergent_decoded = unpack_snerv_archive(divergent.packet)
    training = divergent_decoded.metadata["native_mlx_hf_decoder_training"]
    assert training["attempted"] is True
    assert training["accepted"] is False
    assert training["executed"] is False
    assert training["blockers"]
    assert training["any_loss_worsened"] is True
    guard = divergent_decoded.metadata["native_mlx_training_export_guard"]
    assert guard["export_guard_passed"] is False
    assert "snerv_native_mlx_decoder_loss_worsened_export_blocked" in guard["blockers"]
    assert divergent_decoded.metadata["native_mlx_training_executed"] is False
    assert divergent_decoded.metadata["native_mlx_training_kind"] == "none"
    assert not divergent_decoded.metadata["hf_decoder_fit_mode"].startswith("native_mlx_full_batch_gradient_descent")
    assert divergent_decoded.sections["decoder_payload"] == (baseline_decoded.sections["decoder_payload"])
    assert divergent_decoded.metadata["hf_decoder_fit_mode"] == (baseline_decoded.metadata["hf_decoder_fit_mode"])


def test_train_export_records_blocker_when_native_mlx_training_worsens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=1)
    pairs[0, 1, 0, 2:13, 3:14] += 9.0
    pairs = np.clip(pairs, 0.0, 255.0)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "divergent",
        num_pairs=1,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "decoder_payload_codec": "int8_symmetric",
            "native_mlx_decoder_train_steps": 2,
            "native_mlx_decoder_train_lr": 1.0e6,
            "native_mlx_decoder_train_optimizer": "adamw",
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
    )

    assert Path(report["packet_path"]).is_file()
    assert report["native_mlx_training_executed"] is False
    assert report["native_mlx_training_kind"] == "none"
    training = report["native_mlx_hf_decoder_training"]
    assert training["attempted"] is True
    assert training["accepted"] is False
    assert training["any_loss_worsened"] is True
    guard = report["native_mlx_training_export_guard"]
    assert guard["export_guard_passed"] is False
    assert "snerv_native_mlx_decoder_loss_worsened_export_blocked" in guard["blockers"]
    assert "snerv_native_mlx_decoder_loss_worsened_export_blocked" in report["blockers"]
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_train_export_hydrates_mlx_targets_and_writes_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=1)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path,
        num_pairs=1,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int8_symmetric",
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
    )

    packet_path = Path(report["packet_path"])
    assert packet_path.is_file()
    assert Path(report["report_path"]).name == SNERV_MLX_NATIVE_REPORT_FILENAME
    assert report["bridge_drift"]["allclose"] is True
    assert report["scorer_custody"]["schema"] == "upstream_contest_eval_contract.v1"
    assert report["scorer_custody"]["contract_valid"] is True
    assert any(
        row["relative_path"] == "evaluate.py" and row["sha256"] for row in report["scorer_custody"]["source_custody"]
    )
    assert report["archive_package"] is None
    assert report["archive_path"] is None
    assert report["step_map_bits_per_coeff"] == pytest.approx(0.5)
    assert report["step_map_packet_schema"] == "snerv_step_map_coder.adaptive.v1"
    assert report["step_map_coder_mode"] == ("waterfill_mlx_native_uniform_importance_bridge")
    assert report["step_map_coder_groups"]
    decoded = unpack_snerv_archive(packet_path.read_bytes())
    assert decoded.metadata["lf_payload_codec_requested"] == "portfolio_auto"
    assert decoded.metadata["lf_payload_codec"] == decoded.metadata[
        "lf_payload_codec_selected"
    ]
    assert report["lf_payload_codec"] == decoded.metadata["lf_payload_codec_selected"]
    assert report["receiver_proof_passed"] is False
    assert "snerv_mlx_score_aware_long_training_not_executed" in report["blockers"]
    assert "snerv_real_segnet_posenet_teacher_loop_not_attached" in report["blockers"]
    assert report["scorer_loop_qat"]["requested"] is False
    target_profile = report["receiver_target_reconstruction_profile"]
    export_profile = report["receiver_export_reconstruction_profile"]
    assert target_profile["schema"] == (
        "snerv_receiver_frame_reconstruction_profile.v1"
    )
    assert target_profile["receiver_decoded_selected_packet"] is True
    assert target_profile["reference_kind"] == "source_targets_nchw255"
    assert target_profile["shape_matches"] is True
    assert target_profile["receiver_frames_finite"] is True
    assert target_profile["blockers"] == []
    assert np.isfinite(float(target_profile["mse_nchw255"]))
    assert target_profile["worst_pairs_by_mse"][0]["source_pair_idx"] == 0
    assert export_profile["reference_kind"] == "source_targets_nchw255"
    assert export_profile["mse_nchw255"] == pytest.approx(
        target_profile["mse_nchw255"]
    )
    frames = decode_snerv_archive_frames(packet_path.read_bytes())
    assert frames.shape == (1, 2, 3, 16, 16)


def test_receiver_frame_reconstruction_profile_rejects_wrong_reference_layout() -> None:
    packet = build_snerv_mlx_native_packet_from_numpy_pairs(
        _tiny_pairs(pairs=1),
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        source_pair_indices=(11,),
    )
    bad_reference = np.transpose(_tiny_pairs(pairs=1), (0, 1, 3, 4, 2))

    profile = _snerv_receiver_frame_reconstruction_profile(
        packet.packet,
        reference_pairs_nchw255=bad_reference,
        source_pair_indices=(11,),
        profile_id="unit_bad_reference_layout",
        reference_kind="bad_nhwc_reference",
        packet_source="unit",
    )

    assert profile["shape_matches"] is False
    assert profile["source_pair_indices"] == [11]
    assert profile["blockers"] == [
        "snerv_receiver_frame_reconstruction_reference_not_nchw_pair_tensor"
    ]


def test_receiver_frame_reconstruction_profile_projects_to_scorer_geometry() -> None:
    pairs = _tiny_pairs(pairs=1)
    packet = build_snerv_mlx_native_packet_from_numpy_pairs(
        pairs,
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=8.0,
        step_map_bits_per_coeff=4.0,
        decoder_payload_codec="float32_lzma",
        source_pair_indices=(7,),
    )
    scorer_reference = _resize_nchw_bilinear(
        pairs.reshape(2, 3, 16, 16),
        out_hw=(384, 512),
    ).reshape(1, 2, 3, 384, 512)

    profile = _snerv_receiver_frame_reconstruction_profile(
        packet.packet,
        reference_pairs_nchw255=scorer_reference,
        source_pair_indices=(7,),
        profile_id="unit_receiver_to_scorer_geometry",
        reference_kind="source_targets_nchw255",
        packet_source="unit",
    )

    assert profile["raw_shape_matches"] is False
    assert profile["shape_matches"] is True
    assert profile["scorer_geometry_resize_applied"] is True
    assert profile["comparison_domain"] == "upstream_scorer_geometry_bilinear"
    assert profile["comparison_decoded_shape"] == [1, 2, 3, 384, 512]
    assert "snerv_receiver_frame_reconstruction_shape_mismatch" not in profile[
        "blockers"
    ]
    assert np.isfinite(float(profile["mse_nchw255"]))
    assert np.isfinite(float(profile["segnet_frame1_rgb_mse_nchw255"]))
    assert np.isfinite(float(profile["posenet_yuv6_pair_mse"]))
    assert np.isfinite(float(profile["posenet_yuv6_temporal_delta_mse"]))
    anatomy = profile["scorer_domain_distortion_anatomy"]
    assert anatomy["schema"] == "snerv_scorer_domain_distortion_anatomy.v1"
    assert anatomy["scorer_geometry"]["segnet"].startswith("last_frame_rgb")
    assert anatomy["scorer_geometry"]["posenet"] == (
        "two_frame_upstream_rgb_to_yuv6_pair"
    )
    assert anatomy["human_visual_fidelity_objective"] is False
    assert anatomy["worst_pairs_by_segnet_frame1_mse"][0]["source_pair_idx"] == 7
    assert profile["worst_pairs_by_mse"][0]["source_pair_idx"] == 7


def test_target_pairs_to_nchw255_rejects_byte_scale_targets() -> None:
    target0 = np.full((1, 16, 16, 3), 255.0, dtype=np.float32)
    target1 = np.zeros((1, 16, 16, 3), dtype=np.float32)

    with pytest.raises(SnervMlxNativeExportError, match="normalized RGB"):
        _target_pairs_to_nchw255(target0, target1)


def test_receiver_frame_reconstruction_profile_blocks_constant_receiver_decode() -> None:
    packet = build_snerv_mlx_native_packet_from_numpy_pairs(
        np.zeros((1, 2, 3, 16, 16), dtype=np.float32),
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        source_pair_indices=(0,),
    )

    profile = _snerv_receiver_frame_reconstruction_profile(
        packet.packet,
        reference_pairs_nchw255=_tiny_pairs(pairs=1),
        source_pair_indices=(0,),
        profile_id="unit_constant_receiver_decode",
        reference_kind="source_targets_nchw255",
        packet_source="unit",
    )

    assert profile["shape_matches"] is True
    assert profile["receiver_frames_finite"] is True
    assert "snerv_receiver_frame_reconstruction_decoded_dynamic_range_degenerate" in profile[
        "blockers"
    ]
    assert "snerv_receiver_frame_reconstruction_decoded_std_collapsed" in profile[
        "blockers"
    ]
    assert profile["receiver_value_domain_gate"]["passed"] is False


def test_train_export_executes_real_mlx_hf_decoder_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=1)
    pairs[0, 1, 1, 2:14, 2:14] += 17.0
    pairs = np.clip(pairs, 0.0, 255.0)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)

    baseline = train_export_snerv_mlx_native(
        output_dir=tmp_path / "baseline",
        num_pairs=1,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "decoder_payload_codec": "int8_symmetric",
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
    )
    trained = train_export_snerv_mlx_native(
        output_dir=tmp_path / "trained",
        num_pairs=1,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "decoder_payload_codec": "int8_symmetric",
            "native_mlx_decoder_train_steps": 2,
            "native_mlx_decoder_train_lr": 1.0e-5,
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
    )

    assert baseline["native_mlx_training_executed"] is False
    assert trained["native_mlx_training_executed"] is True
    assert trained["native_mlx_training_kind"] == (
        "hf_decoder_pact_guarded_adamw"
    )
    training = trained["native_mlx_hf_decoder_training"]
    assert training["schema"] == "snerv_native_mlx_hf_decoder_training.v1"
    assert training["executed"] is True
    assert training["steps"] == 2
    assert training["optimizer"] == "pact_guarded_adamw"
    assert training["level_subband_rows"]
    assert training["all_final_losses_finite"] is True
    assert training["accepted"] is True
    assert training["any_loss_worsened"] is False
    assert trained["packet_source"].startswith("native_mlx_pact_guarded_adamw")
    assert Path(trained["packet_path"]).read_bytes() != Path(baseline["packet_path"]).read_bytes()
    decoded = unpack_snerv_archive(Path(trained["packet_path"]).read_bytes())
    assert decoded.metadata["native_mlx_training_executed"] is True
    assert decoded.metadata["native_mlx_hf_decoder_training"]["executed"] is True
    assert decode_snerv_archive_frames(Path(trained["packet_path"]).read_bytes()).shape == (
        1,
        2,
        3,
        16,
        16,
    )


def test_snerv_mlx_haar_renderer_trains_under_shared_pact_muon_harness(
    tmp_path: Path,
) -> None:
    mx = pytest.importorskip("mlx.core")

    from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle
    from tac.substrates._shared.mlx_score_aware.harness import (
        run_mlx_score_aware_full_main,
    )
    from tac.substrates.snerv_inverse_steg_carrier.mlx_renderer import (
        SNERV_MLX_RENDERER_SCHEMA,
        SnervMlxHaarScoreRenderer,
    )

    pairs = _tiny_pairs(pairs=1)
    model = SnervMlxHaarScoreRenderer.from_numpy_pairs(
        pairs,
        levels=1,
        wavelet="haar",
    )
    model.latents_lf_planes = model.latents_lf_planes * 0.25
    before = np.asarray(model.latents_lf_planes, dtype=np.float32).copy()
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)
    bundle = RendererBundle(
        model=model,
        target_rgb_0=target0,
        target_rgb_1=target1,
        num_pairs=2,
        forward_convention="reconstruct_pair_nchw01",
        substrate_artifact_metadata={
            "schema": "unit_snerv_mlx_renderer_bundle.v1",
            "renderer_schema": SNERV_MLX_RENDERER_SCHEMA,
        },
    )

    artifact = run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="snerv_inverse_steg_carrier",
        lane_id="lane_unit_snerv_mlx_renderer_train",
        output_dir=tmp_path / "snerv_renderer_long_train",
        epochs=3,
        batch_pair_indices_per_step=2,
        learning_rate=1.0e-2,
        optimizer_kind="pact_muon_adamw",
        notes=(
            "unit SNeRV renderer shared-harness training proof: pact_muon_adamw "
            "updates LF latents and decoder weights; false-authority smoke"
        ),
    )

    after = np.asarray(model.latents_lf_planes, dtype=np.float32)
    metadata = artifact.as_dict()["substrate_artifact_metadata"]
    assert artifact.total_epochs_completed == 3
    assert not np.allclose(before, after)
    assert metadata["score_aware_training"]["schema"] == (
        "mlx_score_aware_training_objective.v1"
    )
    assert model.metadata()["schema"] == SNERV_MLX_RENDERER_SCHEMA
    assert model.metadata()["trainable_parameter_count"] > before.size


def test_snerv_mlx_haar_renderer_restores_exported_state_dict() -> None:
    pytest.importorskip("mlx.core")

    from tac.substrates.snerv_inverse_steg_carrier.mlx_renderer import (
        SnervMlxHaarScoreRenderer,
    )

    pairs = _tiny_pairs(pairs=1)
    model = SnervMlxHaarScoreRenderer.from_numpy_pairs(
        pairs,
        levels=1,
        wavelet="haar",
    )
    state = model.export_state_dict()
    expected = model.render_pairs_nchw255(batch_size=1)
    model.latents_lf_planes = model.latents_lf_planes * 0.0

    model.import_state_dict(state)

    np.testing.assert_allclose(
        model.render_pairs_nchw255(batch_size=1),
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_snerv_mlx_temporal_context_matches_receiver_feature_algebra() -> None:
    pytest.importorskip("mlx.core")

    from tac.substrates.snerv_inverse_steg_carrier.mlx_renderer import (
        SnervMlxHaarScoreRenderer,
    )

    pairs = _tiny_pairs(pairs=3)
    model_size = SnervModelSizeConfig(
        fc_dim=9,
        emb_size=0,
        temporal_context=1,
        temporal_mode="official_haar_dwt1d_lowpass",
    )
    model = SnervMlxHaarScoreRenderer.from_numpy_pairs(
        pairs,
        levels=1,
        wavelet="haar",
        model_size=model_size,
    )
    selected_pairs = [2, 0]
    mlx_recon = model.render_pairs_nchw255(
        pair_indices=selected_pairs,
        batch_size=2,
    )
    state = model.export_state_dict()
    decoder = HfGenerationDecoder(
        kernels={
            0: {
                subband: state[f"decoder_kernels.0.{subband}"].reshape(
                    model_size.feature_count,
                )
                for subband in ("LH", "HL", "HH")
            }
        },
        levels=1,
        model_size=model_size,
    )
    pyramids = [
        dwt2_multilevel(
            pairs[pair_idx, frame_idx, channel_idx],
            levels=1,
            wavelet="haar",
        )
        for pair_idx in range(3)
        for frame_idx in range(2)
        for channel_idx in range(3)
    ]
    lf_sequence_all = [np.asarray(pyr.lf, dtype=np.float64) for pyr in pyramids]
    expected = np.empty_like(mlx_recon)
    for out_pair_idx, pair_idx in enumerate(selected_pairs):
        for frame_idx in range(2):
            for channel_idx in range(3):
                flat_idx = pair_idx * 6 + frame_idx * 3 + channel_idx
                group = flat_idx % 3
                temporal_sequence = lf_sequence_all[group::3]
                temporal_index = flat_idx // 3
                coeffs = generate_hf_from_lf(
                    state["latents_lf_planes"][pair_idx, frame_idx, channel_idx],
                    decoder,
                    pyramids[flat_idx],
                    lf_sequence=temporal_sequence,
                    sequence_index=temporal_index,
                )
                expected[out_pair_idx, frame_idx, channel_idx] = idwt2_multilevel(
                    WaveletPyramid(
                        coeffs=coeffs,
                        levels=1,
                        wavelet="haar",
                        orig_hw=pyramids[flat_idx].orig_hw,
                    )
                )

    np.testing.assert_allclose(mlx_recon, expected, rtol=2.0e-5, atol=2.0e-3)


def test_train_export_runs_score_aware_long_training_before_packet_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=2)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "score_aware_long_train",
        num_pairs=2,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int8_symmetric",
            "score_aware_long_training_epochs": 2,
            "score_aware_long_training_lr": 1.0e-3,
            "score_aware_long_training_batch_pairs": 2,
            "score_aware_long_training_optimizer": "pact_muon_adamw",
            "score_aware_long_training_checkpoint_retention_keep_last_n": None,
            "snerv_score_aware_long_training_checkpoint_retention_keep_last_n": None,
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
    )

    assert report["score_aware_long_training_executed"] is True
    assert report["score_aware_long_training_kind"] == (
        "snerv_mlx_score_aware_haar_renderer"
    )
    assert report["native_mlx_training_executed"] is True
    assert "snerv_mlx_score_aware_long_training_not_executed" not in report["blockers"]
    long_training = report["score_aware_long_training"]
    assert long_training["executed"] is True
    assert long_training["optimizer_kind"] == "pact_muon_adamw"
    assert long_training["final_recon_mse_nchw255"] <= (
        long_training["initial_recon_mse_nchw255"] + 1.0e-8
    )
    assert long_training["checkpoint_selection_policy"]["mse_fallback"] is False
    assert long_training["scorer_input_distribution_guard_bound"] is True
    assert long_training["checkpoint_selection_policy"][
        "scorer_input_distribution_guard_weight"
    ] == pytest.approx(DEFAULT_SNERV_SCORER_INPUT_DISTRIBUTION_GUARD_WEIGHT)
    assert "scorer_input_distribution_guard" in long_training[
        "checkpoint_selection_policy"
    ]["active_score_surfaces"]
    assert long_training["best_checkpoint_selection"]["selection_metric"] == (
        "score_aware_composite_full_video_surrogate"
    )
    assert long_training["checkpoint_retention"]["keep_last_n"] == (
        mod.SNERV_SCORE_AWARE_CHECKPOINT_RETENTION_KEEP_LAST_N_DEFAULT
    )
    assert long_training["selection_history_tail"]
    best = long_training["best_checkpoint_selection"]
    assert np.isfinite(float(best["segnet_frame1_rgb_mse_nchw255"]))
    assert np.isfinite(float(best["posenet_yuv6_pair_mse"]))
    assert np.isfinite(float(best["posenet_yuv6_temporal_delta_mse"]))
    assert best["scorer_domain_distortion_anatomy"]["schema"] == (
        "snerv_scorer_domain_distortion_anatomy.v1"
    )
    assert best["scorer_domain_distortion_anatomy"]["scorer_geometry"][
        "human_visual_fidelity_objective"
    ] is False
    assert Path(long_training["report_path"]).is_file()
    assert Path(long_training["training_artifact"]["telemetry_path"]).is_file()
    packet = Path(report["packet_path"]).read_bytes()
    decoded = unpack_snerv_archive(packet)
    assert decoded.metadata["score_aware_long_training_executed"] is True
    assert decoded.metadata["score_aware_long_training"]["executed"] is True
    frames = decode_snerv_archive_frames(packet)
    assert frames.shape == (2, 2, 3, 16, 16)
    assert np.isfinite(frames).all()


def test_train_export_does_not_infer_exportable_state_from_executed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=1)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    def fake_long_training(*_args, **_kwargs):
        return {
            "schema": "snerv_mlx_score_aware_long_training_attachment.v1",
            "executed": True,
            "training_completed": True,
            "training_kind": "unit_executed_not_exportable",
            "optimizer_kind": "pact_muon_adamw",
            "blockers": [],
            "_trained_pairs_nchw255": pairs + 33.0,
        }

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)
    monkeypatch.setattr(
        mod,
        "_run_score_aware_long_training_attachment",
        fake_long_training,
    )

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "executed_not_exportable",
        num_pairs=1,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int8_symmetric",
            "score_aware_long_training_epochs": 2,
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
    )

    assert report["score_aware_long_training_executed"] is True
    assert report["score_aware_long_training_trained_state_exportable"] is False
    assert report["native_mlx_training_executed"] is False
    decoded = unpack_snerv_archive(Path(report["packet_path"]).read_bytes())
    assert decoded.metadata["score_aware_long_training_executed"] is True
    assert decoded.metadata["score_aware_long_training_trained_state_exportable"] is False
    assert "trained_state_exportable" not in decoded.metadata[
        "score_aware_long_training"
    ]


def test_train_export_long_training_binds_real_scorer_teachers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates._shared.mlx_score_aware.loss as loss_mod
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=2)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)
    fake_upstream = tmp_path / "upstream"
    (fake_upstream / "models").mkdir(parents=True)
    (fake_upstream / "modules.py").write_text("# fake scorer custody\n", encoding="utf-8")
    (fake_upstream / "models" / "posenet.safetensors").write_bytes(b"pose")
    (fake_upstream / "models" / "segnet.safetensors").write_bytes(b"seg")
    captured: dict[str, object] = {}

    def patterned_segnet_logits(batch: int):
        class_map = np.fromfunction(lambda y, x: (y + x) % 5, (16, 16), dtype=int)
        logits = np.full((int(batch), 16, 16, 5), -3.0, dtype=np.float32)
        for class_index in range(5):
            logits[:, class_map == class_index, class_index] = 3.0
        return mx.array(logits, dtype=mx.float32)

    class FakeSegTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, indices):
            captured["seg_indices_shape"] = tuple(indices.shape)
            return patterned_segnet_logits(int(indices.shape[0]))

        def teacher_logits_for_frames_nhwc01(self, frames):
            captured["seg_live_frames_shape"] = tuple(frames.shape)
            return patterned_segnet_logits(int(frames.shape[0]))

    class FakePoseTeacher:
        pose_dims = 6
        per_dim_scale = mx.ones((6,), dtype=mx.float32)

        def teacher_pose_for_indices(self, indices):
            captured["pose_indices_shape"] = tuple(indices.shape)
            return mx.zeros((int(indices.shape[0]), 6), dtype=mx.float32)

        def teacher_pose_for_yuv6_pair_nhwc(self, yuv6_pair):
            captured["pose_direct_live_yuv6_shape"] = tuple(yuv6_pair.shape)
            return mx.zeros((int(yuv6_pair.shape[0]), 6), dtype=mx.float32)

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    def fake_build_segnet_teacher(bundle, *, upstream_dir, device):
        captured["segnet_upstream_dir"] = Path(upstream_dir)
        captured["segnet_device"] = device
        captured["segnet_bundle_hw"] = tuple(bundle.target_rgb_1.shape[1:3])
        return FakeSegTeacher()

    def fake_build_posenet_teacher(bundle, *, upstream_dir, device):
        captured["posenet_upstream_dir"] = Path(upstream_dir)
        captured["posenet_device"] = device
        captured["posenet_bundle_hw"] = tuple(bundle.target_rgb_0.shape[1:3])
        return FakePoseTeacher()

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)
    monkeypatch.setattr(
        loss_mod,
        "build_mlx_segnet_pair_teacher",
        fake_build_segnet_teacher,
    )
    monkeypatch.setattr(
        loss_mod,
        "build_mlx_posenet_pair_teacher",
        fake_build_posenet_teacher,
    )
    monkeypatch.setattr(
        mod,
        "_resolve_torch_scorer_device_alias",
        lambda value: "mps" if value == "gpu" else value,
    )

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "score_aware_real_teacher_train",
        num_pairs=2,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int8_symmetric",
            "hard_byte_ceiling": 10_000,
            "score_aware_long_training_epochs": 2,
            "score_aware_long_training_lr": 1.0e-3,
            "score_aware_long_training_batch_pairs": 2,
            "score_aware_long_training_optimizer": "pact_muon_adamw",
            "score_aware_long_training_scorer_input_distribution_guard_weight": 0.5,
            "score_aware_long_training_scorer_input_distribution_guard_saturation_margin": 0.03,
            "score_aware_long_training_scorer_input_distribution_guard_temperature": 0.02,
            "score_aware_long_training_scorer_input_contrast_floor_weight": 0.875,
            "score_aware_long_training_scorer_input_contrast_floor_segnet_min_std_ratio": 0.55,
            "score_aware_long_training_scorer_input_contrast_floor_posenet_yuv6_min_std_ratio": 0.45,
            "score_aware_long_training_scorer_input_shape_tether_weight": 0.625,
            "score_aware_long_training_posenet_temporal_signal_floor_weight": 0.5,
            "score_aware_long_training_posenet_temporal_signal_min_std_ratio": 0.4,
            "score_aware_long_training_posenet_temporal_signal_min_mean_abs_ratio": 0.35,
            "score_aware_long_training_pr95_source_weight_amplification": True,
        },
        scorer_upstream_dir=fake_upstream,
        output_height=16,
        output_width=16,
        run_archive_export=False,
        segnet_distillation_weight=0.01,
        pose_distillation_weight=0.001,
        pose_direct_live_distillation_weight=0.75,
        pose_distillation_loss="huber",
        pose_distillation_huber_delta=2.0,
        segnet_distillation_objective="kl_t2",
        segnet_direct_live_distillation_weight=0.25,
        segnet_direct_live_class_histogram_weight=0.125,
        segnet_direct_live_class_balanced_hinge_weight=0.375,
        segnet_direct_live_class_balanced_ce_weight=0.625,
        segnet_direct_live_class_balanced_squared_hinge_weight=0.875,
        distillation_device="gpu",
        coder_aware_qat=True,
        coder_qat_quant_bits=4,
        coder_qat_c1a_entropy_weight=0.0,
        score_aware_long_training_pr95_faithful_curriculum=True,
    )

    assert captured["segnet_upstream_dir"] == fake_upstream.resolve(strict=False)
    assert captured["posenet_upstream_dir"] == fake_upstream.resolve(strict=False)
    assert captured["segnet_device"] == "mps"
    assert captured["posenet_device"] == "mps"
    assert captured["segnet_bundle_hw"] == (16, 16)
    assert captured["posenet_bundle_hw"] == (16, 16)
    assert report["native_mlx_training_executed"] is True
    assert report["score_aware_long_training_executed"] is True
    assert report["score_aware_long_training_real_teachers_bound"] is True
    assert report["score_aware_long_training_has_real_segnet_teacher"] is True
    assert report["score_aware_long_training_has_real_posenet_teacher"] is True
    assert report["score_aware_long_training_coder_qat_bound"] is True
    assert report["score_aware_long_training_pr95_curriculum_bound"] is True
    assert (
        report["score_aware_long_training_pr95_source_weight_amplification_bound"]
        is True
    )
    assert report["score_aware_long_training_muon_adamw_partition_bound"] is True
    assert "snerv_real_segnet_posenet_teacher_loop_not_attached" not in report["blockers"]
    assert "snerv_score_aware_long_training_dual_segnet_lambda_never_active" not in report[
        "blockers"
    ]
    long_training = report["score_aware_long_training"]
    assert long_training["training_completed"] is True
    assert long_training["trained_state_exportable"] is True
    assert long_training["executed"] is True
    assert long_training["training_telemetry_contract"]["passed"] is True
    assert (
        long_training["training_telemetry_contract"][
            "archive_byte_dual_pending_weight_after_short_update"
        ]
        is True
    )
    assert long_training["pr95_stage_source_weight_amplification_enabled"] is True
    assert long_training["has_real_segnet_teacher"] is True
    assert long_training["has_real_posenet_teacher"] is True
    assert long_training["teacher_binding"]["requested_distillation_device"] == "gpu"
    assert long_training["teacher_binding"]["distillation_device"] == "mps"
    # Direct-live PoseNet consumes the upstream YUV6 pair surface, which halves
    # the scorer RGB spatial size after rgb_to_yuv6. For the 16x16 fixture this
    # must be 8x8x12, matching upstream modules.py PoseNet.preprocess_input.
    assert captured["pose_direct_live_yuv6_shape"] == (2, 8, 8, 12)
    assert long_training["teacher_binding"]["distillation_device_resolution"] == {
        "schema": "snerv_native_torch_scorer_device_resolution.v1",
        "requested": "gpu",
        "resolved": "mps",
        "scope": "real_pytorch_segnet_posenet_teacher_cache",
    }
    assert long_training["coder_aware_qat"]["enabled"] is True
    assert long_training["coder_aware_qat"]["quant_bits"] == 4
    contract = long_training["training_telemetry_contract"]
    assert contract["expected_posenet_direct_live_distillation"] is True
    assert contract["posenet_direct_live_loss_observed"] is True
    assert contract["posenet_direct_live_raw_mse_observed"] is True
    assert contract["posenet_direct_live_score_term_observed"] is True
    selection_policy = long_training["checkpoint_selection_policy"]
    assert selection_policy["pose_direct_live_distillation_weight"] == pytest.approx(
        0.75
    )
    assert selection_policy["pose_selection_loss_part"] == "pose_direct_live_score_term"
    assert (
        "real_posenet_direct_live_distillation"
        in selection_policy["active_score_surfaces"]
    )
    assert "pose_direct_live_score_term" in selection_policy["required_loss_parts"]
    assert (
        "weighted_pose_direct_live_score_term"
        in long_training["best_checkpoint_selection"]["score_aware_composite_parts"]
    )
    assert long_training["archive_section_qat_weight_policy_bound"] is True
    section_policy = long_training["archive_section_qat_weight_policy"]
    assert section_policy["schema"] == mod.SNERV_ARCHIVE_SECTION_QAT_WEIGHT_POLICY_SCHEMA
    assert section_policy["active"] is True
    assert section_policy["baseline_packet_bytes"] > 0
    assert section_policy["decoder_section_bytes"] > 0
    assert section_policy["lf_section_bytes"] > 0
    assert "latent_qat_quant_residual" in section_policy["extra_loss_weights"]
    section_control = long_training["train_time_section_byte_control"]
    assert long_training["train_time_section_byte_control_bound"] is True
    assert section_control["active"] is True
    assert set(section_control["section_byte_budgets"]) == {
        "decoder_payload",
        "lf_payload",
    }
    assert section_control["section_byte_loss_weight_key_map"] == {
        "decoder_payload": "coder_qat_quant_residual",
        "lf_payload": "latent_qat_quant_residual",
    }
    live_section_metrics = long_training["live_train_time_section_byte_metrics"]
    assert live_section_metrics["schema"] == (
        "snerv_live_train_time_section_byte_metrics_callback.v1"
    )
    assert live_section_metrics["active"] is True
    assert live_section_metrics["uses_current_renderer_state"] is True
    assert live_section_metrics["last_section_bytes"]["decoder_payload"] > 0
    assert long_training["latent_qat_bound"] is True
    assert (
        report["score_aware_long_training_scorer_input_distribution_guard_bound"]
        is True
    )
    assert long_training["scorer_input_distribution_guard_bound"] is True
    assert long_training["scorer_input_distribution_guard"] == {
        "schema": "snerv_mlx_score_aware_scorer_input_distribution_guard.v1",
        "requested": True,
        "enabled": True,
        "bound_to_renderer_bundle": True,
        "weight": 0.5,
        "saturation_margin": 0.03,
        "temperature": 0.02,
        "target_surface": "decoded_rgb01_vs_target_rgb01_mean_std_soft_saturation",
        "score_authority": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    assert long_training["scorer_input_contrast_floor_bound"] is True
    assert long_training["scorer_input_contrast_floor"] == {
        "schema": "snerv_mlx_score_aware_scorer_input_contrast_floor.v1",
        "requested": True,
        "enabled": True,
        "bound_to_renderer_bundle": True,
        "weight": 0.875,
        "segnet_last_rgb_min_std_ratio": 0.55,
        "posenet_yuv6_pair_min_std_ratio": 0.45,
        "target_surface": (
            "segnet_last_frame_rgb_and_posenet_two_frame_yuv6_std_ratio"
        ),
        "human_visual_fidelity_objective": False,
        "score_authority": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    assert long_training["scorer_input_shape_tether_bound"] is True
    assert long_training["scorer_input_shape_tether"] == {
        "schema": "snerv_mlx_score_aware_scorer_input_shape_tether.v1",
        "requested": True,
        "enabled": True,
        "bound_to_renderer_bundle": True,
        "weight": 0.625,
        "target_surface": (
            "segnet_last_frame_rgb_plus_posenet_yuv6_pair_and_temporal_delta_"
            "centered_reference_variance_normalized_fit"
        ),
        "human_visual_fidelity_objective": False,
        "score_authority": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    assert long_training["posenet_temporal_signal_floor_bound"] is True
    assert long_training["posenet_temporal_signal_floor"] == {
        "schema": "snerv_mlx_score_aware_posenet_temporal_signal_floor.v1",
        "requested": True,
        "enabled": True,
        "bound_to_renderer_bundle": True,
        "weight": 0.5,
        "min_std_ratio": 0.4,
        "min_mean_abs_ratio": 0.35,
        "target_surface": "exact_upstream_posenet_yuv6_frame1_minus_frame0_temporal_signal",
        "human_visual_fidelity_objective": False,
        "score_authority": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    assert long_training["pr95_faithful_curriculum_enabled"] is True
    assert long_training["muon_adamw_partition_bound"] is True
    assert long_training["teacher_binding"]["pose_distillation_loss"] == "huber"
    assert long_training["teacher_binding"]["pose_distillation_huber_delta"] == 2.0
    assert long_training["teacher_binding"]["pose_direct_live_distillation_weight"] == 0.75
    assert long_training["teacher_binding"]["segnet_direct_live_distillation_weight"] == 0.25
    assert long_training["teacher_binding"]["segnet_direct_live_class_histogram_weight"] == 0.125
    assert (
        long_training["teacher_binding"]["segnet_direct_live_class_balanced_hinge_weight"]
        == 0.375
    )
    assert (
        long_training["teacher_binding"]["segnet_direct_live_class_balanced_ce_weight"]
        == 0.625
    )
    assert (
        long_training["teacher_binding"][
            "segnet_direct_live_class_balanced_squared_hinge_weight"
        ]
        == 0.875
    )
    assert long_training["teacher_binding"]["learnable_student_head_bound"] is True
    assert long_training["teacher_binding"]["learnable_pose_student_head_bound"] is True
    assert long_training["checkpoint_selection_policy"]["mse_fallback"] is False
    assert (
        long_training["checkpoint_selection_policy"]["selection_metric"]
        == "score_aware_composite_full_video_surrogate"
    )
    assert long_training["checkpoint_selection_policy"][
        "pose_direct_live_distillation_weight"
    ] == pytest.approx(0.75)
    assert "real_posenet_direct_live_distillation" in long_training[
        "checkpoint_selection_policy"
    ]["active_score_surfaces"]
    assert "pose_direct_live_score_term" in long_training[
        "checkpoint_selection_policy"
    ]["required_loss_parts"]
    assert "real_segnet_teacher_distillation" in long_training[
        "checkpoint_selection_policy"
    ]["active_score_surfaces"]
    assert "real_segnet_direct_live_distillation" in long_training[
        "checkpoint_selection_policy"
    ]["active_score_surfaces"]
    assert "real_posenet_teacher_distillation" in long_training[
        "checkpoint_selection_policy"
    ]["active_score_surfaces"]
    assert "coder_aware_qat" in long_training["checkpoint_selection_policy"][
        "active_score_surfaces"
    ]
    assert "latent_qat_quant_residual" in long_training["checkpoint_selection_policy"][
        "weighted_coder_qat_terms"
    ]
    assert "latent_qat_magnitude" in long_training["checkpoint_selection_policy"][
        "weighted_coder_qat_terms"
    ]
    assert "pr95_faithful_curriculum" in long_training[
        "checkpoint_selection_policy"
    ]["active_score_surfaces"]
    assert "scorer_input_distribution_guard" in long_training[
        "checkpoint_selection_policy"
    ]["active_score_surfaces"]
    assert "scorer_input_contrast_floor" in long_training[
        "checkpoint_selection_policy"
    ]["active_score_surfaces"]
    assert "scorer_input_shape_tether" in long_training[
        "checkpoint_selection_policy"
    ]["active_score_surfaces"]
    assert "posenet_temporal_signal_floor" in long_training[
        "checkpoint_selection_policy"
    ]["active_score_surfaces"]
    assert "scorer_input_distribution_guard" in long_training[
        "checkpoint_selection_policy"
    ]["required_loss_parts"]
    assert "scorer_input_contrast_floor" in long_training[
        "checkpoint_selection_policy"
    ]["required_loss_parts"]
    assert "scorer_input_shape_tether" in long_training[
        "checkpoint_selection_policy"
    ]["required_loss_parts"]
    assert "posenet_temporal_signal_floor" in long_training[
        "checkpoint_selection_policy"
    ]["required_loss_parts"]
    assert "segnet_direct_live_distill" in long_training[
        "checkpoint_selection_policy"
    ]["required_loss_parts"]
    assert "pr95_stage_scorer_surrogate" in long_training[
        "checkpoint_selection_policy"
    ]["required_loss_parts"]
    assert (
        "snerv_score_aware_checkpoint_selection_pr95_stage_selector_missing"
        not in long_training["checkpoint_selection_policy"]["blockers"]
    )
    assert (
        long_training["checkpoint_selection_policy"]["pose_selection_loss_part"]
        == "pose_direct_live_score_term"
    )
    assert (
        long_training["checkpoint_selection_policy"][
            "scorer_input_distribution_guard_weight"
        ]
        == 0.5
    )
    assert (
        long_training["checkpoint_selection_policy"][
            "scorer_input_contrast_floor_weight"
        ]
        == 0.875
    )
    assert (
        long_training["checkpoint_selection_policy"][
            "scorer_input_shape_tether_weight"
        ]
        == 0.625
    )
    assert (
        long_training["checkpoint_selection_policy"][
            "posenet_temporal_signal_floor_weight"
        ]
        == 0.5
    )
    assert long_training["best_checkpoint_selection"]["selection_metric"] == (
        "score_aware_composite_full_video_surrogate"
    )
    assert np.isfinite(
        long_training["best_checkpoint_selection"]["score_aware_composite_loss"]
    )
    assert (
        "snerv_score_aware_checkpoint_selection_pr95_stage_selector_missing"
        not in long_training["best_checkpoint_selection"][
            "score_aware_checkpoint_selection_blockers"
        ]
    )
    assert (
        "snerv_score_aware_checkpoint_selection_pr95_stage_selector_missing"
        not in long_training["selection_failures"]
    )
    assert "weighted_distill" in long_training["best_checkpoint_selection"][
        "score_aware_composite_parts"
    ]
    assert "raw_pr95_stage_scorer_surrogate" in long_training[
        "best_checkpoint_selection"
    ]["score_aware_composite_parts"]
    assert "weighted_pr95_stage_scorer_surrogate" in long_training[
        "best_checkpoint_selection"
    ]["score_aware_composite_parts"]
    assert (
        "weighted_segnet_direct_live_distill"
        in long_training["best_checkpoint_selection"]["score_aware_composite_parts"]
    )
    assert "weighted_pose_score_term" in long_training["best_checkpoint_selection"][
        "score_aware_composite_parts"
    ]
    assert (
        "weighted_scorer_input_distribution_guard"
        in long_training["best_checkpoint_selection"]["score_aware_composite_parts"]
    )
    assert (
        "weighted_scorer_input_shape_tether"
        in long_training["best_checkpoint_selection"]["score_aware_composite_parts"]
    )
    assert (
        "weighted_posenet_temporal_signal_floor"
        in long_training["best_checkpoint_selection"]["score_aware_composite_parts"]
    )
    decoded = unpack_snerv_archive(Path(report["packet_path"]).read_bytes())
    assert (
        decoded.metadata[
            "score_aware_long_training_scorer_input_distribution_guard_bound"
        ]
        is True
    )
    assert decoded.metadata["score_aware_long_training"][
        "scorer_input_distribution_guard_bound"
    ] is True
    assert decoded.metadata["score_aware_long_training"][
        "scorer_input_shape_tether_bound"
    ] is True
    assert decoded.metadata["score_aware_long_training"][
        "posenet_temporal_signal_floor_bound"
    ] is True
    assert decoded.metadata["score_aware_long_training"]["teacher_binding"][
        "has_real_segnet_teacher"
    ] is True
    assert decoded.metadata["score_aware_long_training"]["teacher_binding"][
        "has_real_posenet_teacher"
    ] is True
    assert decoded.metadata["score_aware_long_training"][
        "checkpoint_selection_policy"
    ]["selection_metric"] == "score_aware_composite_full_video_surrogate"


def test_train_export_official_primitives_mode_emits_receiver_bound_surrogate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=2)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)
    captured: dict[str, object] = {}

    def fake_decode_mlx_targets(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = dict(kwargs)
        return target0, target1

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "official_surrogate",
        num_pairs=2,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "candidate_id": "official-primitives-request",
            "snerv_model_size_adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "decoder_payload_codec": "int8_symmetric",
            "snerv_fc_dim": 9,
            "snerv_hfr_gain": 0.125,
            "snerv_temporal_context": 1,
            "snerv_temporal_mode": "official_haar_dwt1d_lowpass",
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
    )

    assert captured["kwargs"]["num_pairs"] == 2
    assert report["executed"] is True
    assert report["snerv_official_mfu_hfr_tub_numeric_primitives_requested"] is True
    assert report["snerv_official_mfu_hfr_tub_export_bound"] is True
    assert report["snerv_official_mfu_hfr_tub_export_bound_semantics"] == (
        "receiver_payload_bound_not_source_forward_parity"
    )
    assert report["snerv_official_mfu_hfr_tub_receiver_payload_bound"] is True
    assert report["snerv_official_mfu_hfr_tub_source_forward_replay_bound"] is False
    assert report["snerv_official_mfu_hfr_tub_source_forward_replay_authority"] is False
    assert report["snerv_official_mfu_hfr_tub_frame_producing_export"] is True
    assert report["snerv_official_mfu_hfr_tub_receiver_bound_surrogate_export"] is False
    assert "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload" not in report["blockers"]
    assert "snerv_official_mfu_hfr_tub_weight_mapping_missing" in report["blockers"]
    assert "snerv_official_mfu_hfr_tub_source_forward_replay_missing" in report["blockers"]
    assert Path(report["packet_path"]).is_file()
    assert report["packet_bytes"] == Path(report["packet_path"]).stat().st_size
    assert report["receiver_proof_passed"] is False
    binding = report["official_primitive_binding"]
    assert binding["schema"] == "snerv_official_mfu_hfr_tub_export_binding.v3"
    assert binding["primitive_modules_available"] is True
    assert binding["export_bound_to_receiver_packet"] is True
    assert binding["receiver_native_export_bound"] is True
    assert binding["official_export_bound"] is False
    assert binding["official_export_bound_semantics"] == (
        "requires_receiver_export_native_mlx_export_and_source_forward_replay"
    )
    assert binding["official_receiver_payload_bound"] is True
    assert binding["official_source_forward_replay_bound"] is False
    assert binding["source_forward_replay_bound_by_export"] is False
    assert binding["surrogate_receiver_payload_contract_emitted"] is False
    assert binding["official_receiver_payload_contract_available"] is True
    assert binding["official_receiver_payload_contract_emitted"] is True
    assert binding["available_official_decoder_payload_schema"] == (
        "snerv_decoder_payload.official_mfu_hfr_tub.v1"
    )
    assert binding["official_receiver_runtime_decode_contract"][
        "receiver_runtime_decode_proven"
    ] is True
    assert binding["current_snar_decoder_payload_schema"] == ("linear_hf_generation_decoder_only")
    assert binding["linear_hf_generation_decoder_compatible_with_official_neural_graph"] is False
    authority = binding["selected_packet_authority"]
    assert authority["schema"] == "snerv_selected_packet_official_payload_authority.v1"
    assert authority["status"] == "frame_producing_official_export"
    assert authority["linear_surrogate_decoder_selected"] is False
    assert authority["official_decoder_payload_selected"] is True
    assert authority["frame_decode_succeeded"] is True
    assert authority["frame_producing_official_export"] is True
    assert authority["blockers"] == []
    tensor_map = binding["official_receiver_tensor_map"]
    assert tensor_map["receiver_tensor_map_verified"] is True
    assert tensor_map["receiver_runtime_decode_contract_proven"] is True
    assert tensor_map["receiver_runtime_decode_authority"] is False
    assert tensor_map["official_decoder_payload_selected"] is True
    assert tensor_map["row_count"] > 0
    assert tensor_map["total_tensor_bytes"] > 0
    assert tensor_map["category_counts"]["official_mfu_weight_payload"] > 0
    assert tensor_map["category_counts"]["official_hfr_weight_payload"] > 0
    assert tensor_map["official_state_dict_mapping_verified"] is False
    assert tensor_map["official_weight_mapping_blocker_closed"] is False
    assert tensor_map["official_weight_mapping_scope"] == (
        "receiver_payload_tensor_hashes_only_not_upstream_state_dict_mapping"
    )
    assert "snerv_official_mfu_hfr_tub_weight_mapping_missing" in tensor_map[
        "official_weight_mapping_blockers"
    ]
    assert len(tensor_map["tensor_manifest_sha256"]) == 64
    assert all(len(row["sha256"]) == 64 for row in tensor_map["rows"])
    assert {
        "official_encoder_embedding_payload",
        "official_mfu_weight_payload",
        "official_hfr_weight_payload",
        "official_tub_weight_payload",
        "official_idwt_or_wavelet_payload",
        "official_decoder_graph_topology_payload",
    }.issubset(set(binding["required_receiver_payload_sections"]))
    assert binding["missing_receiver_payload_sections"] == (binding["required_receiver_payload_sections"])
    assert binding["source_pins"]["official_hfr_source_contract"].startswith("official_snerv_lines_62_64_91_122")
    assert binding["source_pins"]["official_tub_source_contract"].startswith("official_snerv_t_lines_125_136")
    surrogate = binding["receiver_bound_surrogate_export"]
    assert surrogate["kind"] == ("snar1_linear_hf_generation_decoder_not_official_neural_graph")
    assert surrogate["packet_sha256"] == report["packet_sha256"]
    assert surrogate["packet_decoder_payload_schema"] == authority["decoder_payload_schema"]
    assert surrogate["packet_receiver_decode_verified_by_builder"] is True
    assert surrogate["surrogate_receiver_contract_satisfied"] is False
    assert surrogate["score_claim"] is False
    evidence = {row["blocker"]: row for row in binding["blocker_evidence"]}
    assert "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload" not in evidence
    assert evidence["snerv_official_mfu_hfr_tub_weight_mapping_missing"][
        "official_authority"
    ] is False
    assert evidence["snerv_official_mfu_hfr_tub_weight_mapping_missing"][
        "source_forward_authority"
    ] is False
    assert evidence["snerv_official_mfu_hfr_tub_weight_mapping_missing"][
        "receiver_payload_binding_authority"
    ] is False
    assert evidence["snerv_official_mfu_hfr_tub_source_forward_replay_missing"]["official_authority"] is False
    assert binding["export_consumed_official_mfu"] is True
    assert binding["export_consumed_official_hfr"] is True
    assert binding["export_consumed_official_tub"] is True
    assert binding["source_forward_replay_authority"] is False
    assert binding["receiver_runtime_decode_authority"] is True
    assert binding["selected_packet_official_payload_runtime_decode_authority"] is True
    assert binding["selected_packet_frame_producing_official_export"] is True
    decoded = unpack_snerv_archive(Path(report["packet_path"]).read_bytes())
    assert decoded.metadata["snerv_official_mfu_hfr_tub_export_bound"] is True
    assert decoded.metadata["snerv_official_mfu_hfr_tub_export_bound_semantics"] == (
        "receiver_payload_bound_not_source_forward_parity"
    )
    assert decoded.metadata["snerv_official_mfu_hfr_tub_receiver_payload_bound"] is True
    assert decoded.metadata["snerv_official_mfu_hfr_tub_source_forward_replay_bound"] is False
    assert (
        decoded.metadata["snerv_official_mfu_hfr_tub_source_forward_replay_authority"]
        is False
    )
    assert decoded.metadata["snerv_official_mfu_hfr_tub_frame_producing_export"] is True
    assert decoded.metadata["source_faithful_stack"] is False
    assert "snerv_official_bootstrap_stores_haar_ll_as_mfu_skip_high" in decoded.metadata[
        "official_source_parity_blockers"
    ]
    assert (
        "snerv_official_encoder_mfu_skip_hierarchy_source_forward_replay_missing"
        in decoded.metadata["official_source_parity_blockers"]
    )
    official_frames = decode_snerv_archive_frames(Path(report["packet_path"]).read_bytes())
    assert official_frames.shape == (
        2,
        2,
        3,
        16,
        16,
    )
    max_abs_error = float(np.max(np.abs(official_frames - pairs)))
    assert max_abs_error < 3.0e-2
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert Path(report["report_path"]).is_file()


def test_official_primitives_receiver_authority_requires_frame_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    def fake_selected_authority(_packet: bytes) -> dict[str, object]:
        return {
            "schema": "snerv_selected_packet_official_payload_authority.v1",
            "status": "official_payload_selected_not_frame_producing",
            "decoder_payload_schema": "snerv_decoder_payload.official_mfu_hfr_tub.v1",
            "decoder_payload_codec": "int8_symmetric",
            "official_decoder_payload_selected": True,
            "linear_surrogate_decoder_selected": False,
            "frame_decode_attempted": True,
            "frame_decode_succeeded": False,
            "official_payload_runtime_decode_authority": False,
            "frame_producing_official_export": False,
            "blockers": [
                "snerv_official_mfu_hfr_tub_selected_payload_not_frame_producing"
            ],
            "score_claim": False,
            "frontier_score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(
        mod,
        "_selected_packet_official_payload_authority",
        fake_selected_authority,
    )
    monkeypatch.setattr(
        mod,
        "_official_receiver_tensor_map_from_packet",
        lambda _packet: {
            "schema": "snerv_official_mfu_hfr_tub_receiver_tensor_map.v1",
            "receiver_tensor_map_verified": False,
            "receiver_runtime_decode_contract_proven": True,
            "receiver_runtime_decode_authority": False,
        },
    )

    binding = mod._receiver_bound_official_primitives_export_binding(
        {
            "schema": "snerv_official_mfu_hfr_tub_export_binding.v2",
            "official_receiver_runtime_decode_contract": {
                "receiver_runtime_decode_proven": True,
            },
            "blockers": [
                "snerv_official_mfu_hfr_tub_weight_mapping_missing",
            ],
        },
        packet_path=tmp_path / "candidate.snar",
        packet_bytes=16,
        packet_sha256="0" * 64,
        selected_packet=b"official-looking-but-not-frame-decodable",
        selected_archive_metadata={"decoder_payload_codec": "int8_symmetric"},
        package=None,
        receiver_proof={
            "runtime_consumption_proof_passed": True,
            "receiver_contract_satisfied": True,
        },
    )

    assert binding["official_receiver_runtime_decode_contract_proven"] is True
    assert binding["official_receiver_payload_bound"] is True
    assert binding["selected_packet_authority"]["frame_decode_succeeded"] is False
    assert binding["selected_packet_official_payload_runtime_decode_authority"] is False
    assert binding["selected_packet_frame_producing_official_export"] is False
    assert binding["receiver_runtime_decode_authority"] is False
    assert binding["receiver_native_export_bound"] is False
    assert binding["export_consumed_official_mfu"] is False
    assert binding["score_claim"] is False
    assert binding["promotion_eligible"] is False
    assert binding["ready_for_exact_eval_dispatch"] is False


def test_train_export_official_primitives_shared_skip_high_is_receiver_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=2)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "official_shared_skip_high",
        num_pairs=2,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "candidate_id": "official-shared-skip-high",
            "snerv_model_size_adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "snerv_fc_dim": 9,
            "snerv_official_skip_high_mode": "shared_mean",
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
    )

    decoded = unpack_snerv_archive(Path(report["packet_path"]).read_bytes())
    official_payload = decode_official_mfu_hfr_tub_decoder_payload(
        decoded.sections["decoder_payload"]
    )
    frames = decode_snerv_archive_frames(Path(report["packet_path"]).read_bytes())

    assert report["snerv_official_mfu_hfr_tub_export_bound"] is True
    assert report["official_skip_high_mode"] == "shared_mean"
    assert report["official_skip_high_full_shape"] == [4, 3, 8, 8]
    assert "snerv_official_bootstrap_stores_haar_ll_as_mfu_skip_high" in report[
        "official_source_parity_blockers"
    ]
    storage = official_payload.header["skip_high_storage"]
    assert storage["codec"] == "shared_mean_float64"
    assert storage["source_shape"] == [4, 3, 8, 8]
    assert storage["stored_shape"] == [1, 3, 8, 8]
    assert storage["raw_byte_savings"] == 4608
    assert official_payload.tensors["inputs.mfu.skip_high"].shape == (4, 3, 8, 8)
    assert frames.shape == (2, 2, 3, 16, 16)
    assert np.isfinite(frames).all()
    assert official_payload.score_claim is False


def test_official_mfu_hfr_tub_packet_elides_output2_payload_from_components() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=2)
    model_size = SnervModelSizeConfig(
        adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        fc_dim=9,
    )
    components = mod._official_mfu_hfr_tub_bootstrap_components_from_pairs(
        pairs,
        model_size=model_size,
    )
    components["tub_temporal_encoder_concat"] = np.arange(
        np.prod(components["temporal_encoder_output_shape"]),
        dtype=np.float64,
    ).reshape(components["temporal_encoder_output_shape"])
    components["tub_output2_raw"] = (
        np.arange(
            np.prod(components["output2_decoder_output_shape"]),
            dtype=np.float64,
        ).reshape(components["output2_decoder_output_shape"])
        / 19.0
    )

    packet = mod._build_official_mfu_hfr_tub_packet_from_components(
        components,
        source_pair_indices=[4, 5],
        model_size=model_size,
        metadata_extra={
            "allocation_mode": "unit_output2_payload_bound",
            "official_tub_output2_storage": {"stored": False},
            "official_tub_output2_payload_export_bound": False,
            "official_tub_output2_receiver_executed": False,
        },
    )
    decoded = unpack_snerv_archive(packet.packet)
    official_payload = decode_official_mfu_hfr_tub_decoder_payload(
        decoded.sections["decoder_payload"]
    )
    proof = official_payload.execute()

    rich_metadata = packet.metadata
    mfu_storage = official_payload.header["mfu_input_storage"]
    assert decoded.schema == "snerv_inverse_steg_archive.snar2.v1"
    assert "official_mfu_input_storage_mode" not in decoded.metadata
    assert rich_metadata["official_mfu_input_storage_mode"] == "zero_synthetic"
    assert mfu_storage["codec"] == "zero_synthetic_float64"
    assert mfu_storage["stored_raw_bytes"] == 16
    assert mfu_storage["raw_byte_savings"] > 0
    assert official_payload.tensors["inputs.mfu.low"].shape == components["low"].shape
    assert official_payload.tensors["inputs.mfu.skip_mid"].shape == components[
        "skip_mid"
    ].shape
    assert np.count_nonzero(official_payload.tensors["inputs.mfu.low"]) == 0
    assert np.count_nonzero(official_payload.tensors["inputs.mfu.skip_mid"]) == 0
    storage = rich_metadata["official_tub_output2_storage"]
    assert rich_metadata["lf_payload_codec"] == "spatial_delta_zigzag_leb128_lzma"
    assert (
        rich_metadata["lf_payload_codec_requested"]
        == "spatial_delta_zigzag_leb128_lzma"
    )
    assert (
        rich_metadata["lf_payload_codec_selected"]
        == "spatial_delta_zigzag_leb128_lzma"
    )
    assert rich_metadata["lf_payload_receiver_usage"] == (
        "unused_dummy_zero_official_payload_frame_decode_uses_decoder_payload"
    )
    assert (
        rich_metadata["lf_payload_codec_selection_report"]["section_bytes"]
        == packet.section_bytes["lf_payload"]
    )
    assert storage["stored"] is False
    assert storage["source_payload_present"] is True
    assert storage["proof_only_elided_from_selected_runtime_packet"] is True
    assert storage["proof_only_false_authority_metadata"] is True
    assert storage["storage_policy"] == "elide_until_receiver_frame_decode_bound"
    assert storage["receiver_executes_output2_fusion_from_payload"] is False
    assert storage["receiver_frame_decode_consumes_output2"] is False
    assert storage["stored_raw_bytes"] == 0
    assert storage["source_raw_bytes"] > 0
    assert storage["raw_byte_savings"] == storage["source_raw_bytes"]
    assert storage["tensor_names"] == [
        "tub.temporal_encoder_concat",
        "tub.output2_raw",
    ]
    assert rich_metadata["official_tub_output2_receiver_executed"] is False
    assert rich_metadata["official_tub_output2_payload_export_bound"] is False
    assert rich_metadata["official_tub_output2_payload_source_available"] is True
    assert rich_metadata["official_tub_output2_payload_proof_only_elided"] is True
    assert (
        rich_metadata[
            "official_tub_output2_payload_false_authority_metadata_bound"
        ]
        is True
    )
    assert rich_metadata["official_tub_output2_receiver_frame_bound"] is False
    assert rich_metadata["official_tub_output2_payload_loss_coupled"] is False
    assert rich_metadata["official_tub_output2_payload_tensor_names"] == []
    assert rich_metadata["official_tub_output2_payload_tensor_count"] == 0
    assert rich_metadata["official_tub_output2_payload_selected_runtime_bytes"] == 0
    assert rich_metadata["official_tub_output2_payload_source_raw_bytes"] > 0
    manifest = {
        row["name"]: row
        for row in rich_metadata["official_tub_output2_payload_tensor_manifest"]
    }
    assert manifest == {}
    assert (
        rich_metadata["official_tub_output2_payload_tensor_manifest_sha256"]
        is None
    )
    assert proof["executed_components"]["official_tub_output2_fusion"] is False
    rows = {row["name"]: row for row in proof["output_tensors"]}
    assert "tub.output2_decoder_input" not in rows
    assert "tub.output2_fused" not in rows
    assert rich_metadata["official_tub_output2_receiver_output_tensor_names"] == []
    assert rich_metadata["official_tub_output2_receiver_output_tensor_count"] == 0
    frames = decode_snerv_archive_frames(packet.packet, clip_to_uint8_range=False)
    components["tub_output2_raw"] = np.asarray(components["tub_output2_raw"]) + 7.0
    mutated_packet = mod._build_official_mfu_hfr_tub_packet_from_components(
        components,
        source_pair_indices=[4, 5],
        model_size=model_size,
        metadata_extra={"allocation_mode": "unit_output2_payload_mutated"},
    )
    mutated_frames = decode_snerv_archive_frames(
        mutated_packet.packet,
        clip_to_uint8_range=False,
    )
    np.testing.assert_array_equal(frames, mutated_frames)
    assert rich_metadata["source_faithful_stack"] is False
    assert rich_metadata["score_claim"] is False


def test_official_mfu_hfr_tub_packet_binds_output2_to_receiver_frames_when_requested() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=1)
    model_size = SnervModelSizeConfig(
        adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        fc_dim=9,
        official_tub_output2_export_mode="receiver_frame_bound",
    )
    components = mod._official_mfu_hfr_tub_bootstrap_components_from_pairs(
        pairs,
        model_size=model_size,
    )
    components["temporal_encoder_output_shape"] = (1, 6, 8, 8)
    components["output2_decoder_output_shape"] = (2, 12, 8, 8)
    components["tub_temporal_encoder_concat"] = np.linspace(
        0.0,
        1.0,
        np.prod(components["temporal_encoder_output_shape"]),
        dtype=np.float64,
    ).reshape(components["temporal_encoder_output_shape"])
    components["tub_output2_raw"] = np.full(
        components["output2_decoder_output_shape"],
        0.125,
        dtype=np.float64,
    )

    packet = mod._build_official_mfu_hfr_tub_packet_from_components(
        components,
        source_pair_indices=[4],
        model_size=model_size,
        metadata_extra={"allocation_mode": "unit_output2_receiver_frame_bound"},
    )
    decoded = unpack_snerv_archive(packet.packet)
    official_payload = decode_official_mfu_hfr_tub_decoder_payload(
        decoded.sections["decoder_payload"]
    )
    proof = official_payload.execute()
    rich_metadata = packet.metadata
    storage = rich_metadata["official_tub_output2_storage"]

    assert model_size.official_tub_output2_store_for_receiver_proof is True
    assert rich_metadata["official_tub_output2_export_mode"] == "receiver_frame_bound"
    assert rich_metadata["official_tub_output2_receiver_frame_bound_required"] is True
    assert rich_metadata["official_tub_output2_payload_export_bound"] is True
    assert rich_metadata["official_tub_output2_receiver_executed"] is True
    assert rich_metadata["official_tub_output2_receiver_frame_bound"] is True
    assert rich_metadata["official_tub_output2_payload_selected_runtime_bytes"] > 0
    assert storage["stored"] is True
    assert storage["proof_only_false_authority_metadata"] is False
    assert storage["receiver_executes_output2_fusion_from_payload"] is True
    assert storage["receiver_frame_decode_consumes_output2"] is True
    assert storage["receiver_output2_frame_shape_match"] is True
    assert storage["frame_decode_blockers"] == []
    assert storage["score_lagrangian_action"] == (
        "keep_only_for_receiver_proof_until_trained_source_forward_parity"
    )
    assert set(rich_metadata["official_tub_output2_payload_tensor_names"]) == {
        "tub.temporal_encoder_concat",
        "tub.output2_raw",
    }
    rows = {row["name"]: row for row in proof["output_tensors"]}
    assert rows["tub.output2_decoder_input"]["shape"] == [2, 3, 8, 8]
    assert rows["tub.output2_fused"]["shape"] == [2, 3, 16, 16]

    frames = decode_snerv_archive_frames(packet.packet, clip_to_uint8_range=False)
    components["tub_output2_raw"] = np.asarray(components["tub_output2_raw"]) + 0.25
    mutated_packet = mod._build_official_mfu_hfr_tub_packet_from_components(
        components,
        source_pair_indices=[4],
        model_size=model_size,
        metadata_extra={"allocation_mode": "unit_output2_receiver_frame_bound_mutated"},
    )
    mutated_frames = decode_snerv_archive_frames(
        mutated_packet.packet,
        clip_to_uint8_range=False,
    )
    assert not np.allclose(frames, mutated_frames)
    assert rich_metadata["source_faithful_stack"] is False
    assert rich_metadata["score_claim"] is False


def test_official_mfu_hfr_tub_packet_fails_closed_when_receiver_frame_bound_missing() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=1)
    model_size = SnervModelSizeConfig(
        adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        fc_dim=9,
        official_tub_output2_export_mode="receiver_frame_bound",
    )
    components = mod._official_mfu_hfr_tub_bootstrap_components_from_pairs(
        pairs,
        model_size=model_size,
    )
    components["tub_temporal_encoder_concat"] = np.zeros(
        components["temporal_encoder_output_shape"],
        dtype=np.float64,
    )
    components["tub_output2_raw"] = np.zeros(
        components["output2_decoder_output_shape"],
        dtype=np.float64,
    )

    with pytest.raises(
        SnervMlxNativeExportError,
        match="snerv_official_tub_output2_receiver_frame_decode_not_bound",
    ):
        mod._build_official_mfu_hfr_tub_packet_from_components(
            components,
            source_pair_indices=[4],
            model_size=model_size,
            metadata_extra={"allocation_mode": "unit_output2_receiver_frame_bound_bad"},
        )


def test_official_renderer_elides_output2_from_selected_receiver_packet() -> None:
    pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod
    from tac.substrates.snerv_inverse_steg_carrier.mlx_renderer import (
        SnervMlxOfficialMfuHfrTubScoreRenderer,
    )

    pairs = _tiny_pairs(pairs=2)
    model_size = SnervModelSizeConfig(
        adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        fc_dim=9,
    )
    components = mod._official_mfu_hfr_tub_bootstrap_components_from_pairs(
        pairs,
        model_size=model_size,
    )
    temporal = np.arange(
        np.prod(components["temporal_encoder_output_shape"]),
        dtype=np.float64,
    ).reshape(components["temporal_encoder_output_shape"])
    output2_raw = (
        np.arange(
            np.prod(components["output2_decoder_output_shape"]),
            dtype=np.float64,
        ).reshape(components["output2_decoder_output_shape"])
        / 23.0
    )
    model = SnervMlxOfficialMfuHfrTubScoreRenderer(
        mfu=components["mfu"],
        hfr_heads=components["hfr_heads"],
        low=components["low"],
        skip_mid=components["skip_mid"],
        skip_high=components["skip_high"],
        output_hw=(16, 16),
        model_size=model_size,
        tub_current=components["tub_current"],
        tub_previous=components["tub_previous"],
        tub_next_frame=components["tub_next_frame"],
        tub_temporal_encoder_concat=temporal,
        tub_output2_raw=output2_raw,
        tub_output2_fc_hw=components["fc_hw"],
    )

    exported = model.export_official_components()
    metadata = model.metadata()
    packet = mod._build_official_mfu_hfr_tub_packet_from_components(
        exported,
        source_pair_indices=[0, 1],
        model_size=model_size,
        metadata_extra={"allocation_mode": "unit_renderer_output2_payload_bound"},
    )
    decoded = unpack_snerv_archive(packet.packet)
    official_payload = decode_official_mfu_hfr_tub_decoder_payload(
        decoded.sections["decoder_payload"]
    )
    proof = official_payload.execute()
    receiver_frames = decoded.decode_frames(clip_to_uint8_range=False)
    mlx_frames = model.render_pairs_nchw255(pair_indices=[0, 1], batch_size=2)
    rich_metadata = packet.metadata

    assert "tub_temporal_encoder_concat" in exported
    assert "tub_output2_raw" in exported
    assert metadata["official_tub_output2_payload_export_bound"] is True
    assert metadata["official_tub_output2_receiver_frame_bound"] is False
    assert metadata["official_tub_output2_payload_loss_coupled"] is False
    assert decoded.schema == "snerv_inverse_steg_archive.snar2.v1"
    assert "official_tub_output2_storage" not in decoded.metadata
    assert rich_metadata["official_tub_output2_storage"]["stored"] is False
    assert (
        rich_metadata["official_tub_output2_storage"][
            "source_payload_present"
        ]
        is True
    )
    assert (
        rich_metadata["official_tub_output2_storage"][
            "proof_only_elided_from_selected_runtime_packet"
        ]
        is True
    )
    assert (
        rich_metadata["official_tub_output2_storage"][
            "proof_only_false_authority_metadata"
        ]
        is True
    )
    assert (
        rich_metadata["official_tub_output2_storage"][
            "receiver_frame_decode_consumes_output2"
        ]
        is False
    )
    assert (
        rich_metadata["official_tub_output2_storage"]["train_time_loss_coupled"]
        is False
    )
    assert rich_metadata["official_tub_output2_payload_export_bound"] is False
    assert rich_metadata["official_tub_output2_payload_source_available"] is True
    assert rich_metadata["official_tub_output2_payload_proof_only_elided"] is True
    assert (
        rich_metadata[
            "official_tub_output2_payload_false_authority_metadata_bound"
        ]
        is True
    )
    assert rich_metadata["official_tub_output2_payload_tensor_count"] == 0
    assert rich_metadata["official_tub_output2_payload_selected_runtime_bytes"] == 0
    assert rich_metadata["official_tub_output2_payload_source_raw_bytes"] > 0
    assert rich_metadata["official_tub_output2_receiver_executed"] is False
    assert proof["executed_components"]["official_tub_output2_fusion"] is False
    packet_tensor_names = {row["name"] for row in official_payload.header["tensor_manifest"]}
    assert "tub.temporal_encoder_concat" not in packet_tensor_names
    assert "tub.output2_raw" not in packet_tensor_names
    pixel_drift = np.abs(mlx_frames - receiver_frames)
    assert float(pixel_drift.max()) < 0.02
    assert float(pixel_drift.mean()) < 0.005


def test_official_primitives_long_training_exports_trained_official_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=2)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)
    monkeypatch.setattr(
        mod,
        "build_snerv_official_tub_source_forward_replay_artifact",
        lambda: _fake_tub_fixture_replay_passed(),
    )
    original_bootstrap = mod._official_mfu_hfr_tub_bootstrap_components_from_pairs

    def bootstrap_with_output2(pairs_arg, *, model_size):
        components = original_bootstrap(pairs_arg, model_size=model_size)
        components["tub_temporal_encoder_concat"] = np.arange(
            np.prod(components["temporal_encoder_output_shape"]),
            dtype=np.float64,
        ).reshape(components["temporal_encoder_output_shape"])
        components["tub_output2_raw"] = (
            np.arange(
                np.prod(components["output2_decoder_output_shape"]),
                dtype=np.float64,
            ).reshape(components["output2_decoder_output_shape"])
            / 29.0
        )
        return components

    monkeypatch.setattr(
        mod,
        "_official_mfu_hfr_tub_bootstrap_components_from_pairs",
        bootstrap_with_output2,
    )

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "official_long_training_bound",
        num_pairs=2,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "candidate_id": "official-primitives-long-training-request",
            "snerv_model_size_adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "decoder_payload_codec": "int8_symmetric",
            "snerv_fc_dim": 9,
            "score_aware_long_training_epochs": 1,
            "score_aware_long_training_batch_pairs": 2,
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
    )

    old_blocker = (
        "snerv_score_aware_long_training_official_mfu_hfr_tub_renderer_not_bound"
    )
    assert old_blocker not in report["blockers"]
    assert report["score_aware_long_training_executed"] is True
    assert report["score_aware_long_training_kind"] == (
        "snerv_mlx_official_mfu_hfr_tub_score_renderer"
    )
    assert report["native_mlx_training_executed"] is True
    assert report["native_mlx_training_kind"] == (
        "snerv_mlx_official_mfu_hfr_tub_score_renderer"
    )
    assert report["snerv_official_mfu_hfr_tub_export_bound"] is True
    assert report["snerv_official_mfu_hfr_tub_receiver_payload_bound"] is True
    assert report["snerv_official_mfu_hfr_tub_source_forward_replay_bound"] is False
    assert report["snerv_official_tub_source_fixture_replay_bound"] is True
    assert report["snerv_official_tub_source_fixture_replay_passed"] is True
    assert report["snerv_official_tub_source_forward_fixture_bound"] is True
    tub_fixture_binding = report["snerv_official_tub_source_fixture_binding"]
    assert tub_fixture_binding["source_fixture_replay_bound"] is True
    assert tub_fixture_binding[
        "official_tub_temporal_encoder_output2_source_fixture_replay_passed"
    ] is True
    assert tub_fixture_binding["full_tub_source_forward_parity_proven"] is False
    assert (
        "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing"
        in tub_fixture_binding["preserved_source_parity_blockers"]
    )
    assert (
        "snerv_official_tub_batched_temporal_context_source_forward_replay_missing"
        not in report["official_source_parity_blockers"]
    )
    assert (
        "snerv_official_encoder_mfu_skip_hierarchy_source_forward_replay_missing"
        in report["official_source_parity_blockers"]
    )
    assert report["snerv_official_mfu_hfr_tub_frame_producing_export"] is True
    assert (
        "snerv_score_aware_long_training_official_mfu_hfr_tub_differentiable_mlx_renderer_missing"
        not in report["blockers"]
    )
    assert (
        "snerv_official_mfu_hfr_tub_trained_weight_mapping_to_long_training_missing"
        in report["blockers"]
    )
    assert "snerv_official_trained_checkpoint_state_dict_mapping_missing" in report[
        "blockers"
    ]
    assert "snerv_official_trained_checkpoint_source_forward_replay_missing" in report[
        "blockers"
    ]
    stale_blocker = "snerv_official_mfu_hfr_tub_source_forward_replay_missing"
    assert stale_blocker not in report["blockers"]
    assert "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing" in report[
        "blockers"
    ]

    long_training = report["score_aware_long_training"]
    assert long_training["requested_epochs"] == 1
    assert long_training["executed"] is True
    assert long_training["training_kind"] == (
        "snerv_mlx_official_mfu_hfr_tub_score_renderer"
    )
    assert long_training["renderer"]["schema"] == (
        "snerv_mlx_official_mfu_hfr_tub_score_renderer.v1"
    )
    assert long_training["renderer"]["official_tub_output2_payload_export_bound"] is True
    assert "tub.temporal_encoder_concat" in long_training["renderer"][
        "receiver_export_payload_atoms"
    ]
    assert "tub.output2_raw" in long_training["renderer"][
        "receiver_export_payload_atoms"
    ]
    train_export = long_training["official_mfu_hfr_tub_train_export"]
    assert train_export["requested"] is True
    assert train_export["train_renderer_bound"] is True
    assert train_export["trained_receiver_payload_exported"] is True
    assert train_export["trained_receiver_state_bound"] is True
    assert train_export["trained_receiver_state_mapping_scope"] == (
        "mlx_receiver_component_state_not_upstream_official_state_dict"
    )
    assert train_export["trained_weight_mapping_to_long_training_bound"] is False
    assert train_export["official_trained_checkpoint_state_dict_loaded"] is False
    assert (
        train_export["official_trained_checkpoint_state_dict_mapping_verified"]
        is False
    )
    assert (
        train_export["official_trained_checkpoint_source_forward_replay_verified"]
        is False
    )
    train_manifest = train_export["official_trained_checkpoint_mapping_manifest"]
    assert train_manifest["schema"] == (
        "snerv_official_trained_checkpoint_state_dict_mapping_manifest.v1"
    )
    assert train_manifest["official_trained_checkpoint_loaded"] is False
    assert "snerv_official_trained_checkpoint_state_dict_not_loaded" in train_manifest[
        "blockers"
    ]
    assert (
        "snerv_official_mfu_hfr_tub_weight_mapping_missing"
        in train_export["authority_blockers"]
    )
    assert "snerv_official_trained_checkpoint_state_dict_mapping_missing" in train_export[
        "authority_blockers"
    ]
    assert len(train_export["trained_packet_sha256"]) == 64
    assert train_export["official_tub_output2_payload_export_bound"] is False
    assert train_export["official_tub_output2_payload_source_available"] is True
    assert train_export["official_tub_output2_payload_proof_only_elided"] is True
    assert (
        train_export["official_tub_output2_payload_false_authority_metadata_bound"]
        is True
    )
    assert train_export["official_tub_output2_payload_tensor_names"] == []
    assert train_export["official_tub_output2_payload_tensor_count"] == 0
    assert train_export["official_tub_output2_payload_selected_runtime_bytes"] == 0
    assert train_export["official_tub_output2_payload_source_raw_bytes"] > 0
    assert train_export["official_tub_output2_receiver_executed"] is False
    assert train_export["official_tub_output2_receiver_output_tensor_count"] == 0
    assert train_export["snerv_official_tub_source_fixture_replay_bound"] is True
    assert train_export["snerv_official_tub_source_fixture_replay_passed"] is True
    assert (
        "snerv_official_tub_batched_temporal_context_source_forward_replay_missing"
        not in train_export["official_source_parity_blockers"]
    )
    assert train_export["source_forward_replay_authority"] is False
    assert old_blocker not in long_training["blockers"]
    replay = long_training["official_mfu_hfr_tub_source_forward_replay"]
    assert replay["schema"] == (
        "snerv_official_mfu_hfr_tub_source_forward_replay_contract.v1"
    )
    assert Path(replay["artifact_path"]).is_file()
    assert len(replay["artifact_sha256"]) == 64
    assert replay["receiver_official_payload_forward_replay_passed"] is True
    assert replay["source_forward_replay_bound"] is False
    assert replay["source_forward_replay_verified"] is False
    assert replay["score_aware_long_training_renderer_bound"] is True
    assert replay["train_renderer_bound"] is True
    assert replay["trained_receiver_state_bound"] is True
    assert replay["trained_receiver_state_mapping_scope"] == (
        "mlx_receiver_payload_components_bound_to_training_state"
    )
    assert replay["trained_weight_mapping_to_long_training_bound"] is False
    assert replay["official_trained_checkpoint_loaded"] is False
    assert replay["official_trained_checkpoint_state_dict_mapping_verified"] is False
    assert (
        replay["official_trained_checkpoint_source_forward_replay_verified"] is False
    )
    replay_manifest = replay["official_trained_checkpoint_mapping_manifest"]
    assert replay_manifest["official_trained_checkpoint_loaded"] is False
    assert "snerv_official_trained_checkpoint_state_dict_not_loaded" in replay_manifest[
        "blockers"
    ]
    assert replay["official_torch_source_forward_replay_passed"] is False
    assert replay["official_tub_fixture_source_forward_replay_proven"] is True
    assert replay["official_tub_source_forward_fixture_replay"][
        "official_tub_temporal_encoder_output2_source_fixture_replay_passed"
    ] is True
    assert replay["selected_packet_authority"]["status"] == (
        "frame_producing_official_export"
    )
    assert replay["official_receiver_tensor_map"][
        "receiver_tensor_map_verified"
    ] is True
    assert replay["official_receiver_runtime_decode_proof"][
        "receiver_runtime_decode_proven"
    ] is True
    assert replay["max_abs_error_nchw255"] < 5.0e-2
    assert {row["component_id"] for row in replay["component_rows"]} == {
        "mfu",
        "hfr",
        "tub",
    }
    assert all(
        row["receiver_payload_forward_replay_proven"] is True
        and row["official_source_forward_parity_proven"] is False
        and row["score_aware_long_training_renderer_bound"] is True
        and row["trained_receiver_state_bound"] is True
        and row["official_trained_checkpoint_state_dict_mapping_verified"] is False
        for row in replay["component_rows"]
    )
    assert (
        "snerv_score_aware_long_training_official_mfu_hfr_tub_differentiable_mlx_renderer_missing"
        not in replay["blockers"]
    )
    assert (
        "snerv_official_mfu_hfr_tub_trained_weight_mapping_to_long_training_missing"
        in replay["blockers"]
    )
    assert "snerv_official_trained_checkpoint_state_dict_mapping_missing" in replay[
        "blockers"
    ]
    assert stale_blocker not in replay["blockers"]
    assert "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing" in replay[
        "blockers"
    ]
    tub_row = {
        row["component_id"]: row for row in replay["component_rows"]
    }["tub"]
    assert tub_row["official_tub_fixture_source_forward_replay_proven"] is True
    assert "snerv_official_snerv_t_output2_fusion_source_forward_replay_missing" in tub_row[
        "official_tub_fixture_closed_blockers"
    ]
    decoded = unpack_snerv_archive(Path(report["packet_path"]).read_bytes())
    assert decoded.schema == "snerv_inverse_steg_archive.snar2.v1"
    assert report["snerv_official_mfu_hfr_tub_export_bound"] is True
    assert report["snerv_official_mfu_hfr_tub_receiver_payload_bound"] is True
    assert report["snerv_official_mfu_hfr_tub_source_forward_replay_bound"] is False
    assert report["snerv_official_tub_source_fixture_replay_bound"] is True
    assert report["snerv_official_tub_source_fixture_replay_passed"] is True
    assert report["snerv_official_tub_source_forward_fixture_bound"] is True
    decoded_tub_binding = report["snerv_official_tub_source_fixture_binding"]
    assert decoded_tub_binding["source_fixture_replay_bound"] is True
    assert decoded_tub_binding["source_forward_replay_authority"] is False
    assert (
        "snerv_official_tub_batched_temporal_context_source_forward_replay_missing"
        not in report["official_source_parity_blockers"]
    )
    assert (
        "snerv_official_encoder_mfu_skip_hierarchy_source_forward_replay_missing"
        in report["official_source_parity_blockers"]
    )
    assert report["official_tub_output2_payload_export_bound"] is False
    assert report["official_tub_output2_payload_source_available"] is True
    assert report["official_tub_output2_payload_proof_only_elided"] is True
    assert (
        report[
            "official_tub_output2_payload_false_authority_metadata_bound"
        ]
        is True
    )
    assert report["official_tub_output2_payload_tensor_count"] == 0
    assert report["official_tub_output2_payload_selected_runtime_bytes"] == 0
    assert report["official_tub_output2_payload_source_raw_bytes"] > 0
    assert report["official_tub_output2_receiver_executed"] is False
    assert report["official_tub_output2_receiver_output_tensor_count"] == 0
    packet_manifest = {
        row["name"]: row
        for row in report["official_tub_output2_payload_tensor_manifest"]
    }
    assert packet_manifest == {}
    assert report["official_tub_output2_storage"][
        "source_payload_present"
    ] is True
    assert report["official_tub_output2_storage"][
        "proof_only_elided_from_selected_runtime_packet"
    ] is True
    assert report["official_tub_output2_storage"][
        "stored_raw_bytes"
    ] == 0
    official_payload = decode_official_mfu_hfr_tub_decoder_payload(
        decoded.sections["decoder_payload"]
    )
    assert official_payload.execute()["executed_components"][
        "official_tub_output2_fusion"
    ] is False
    assert "tub.temporal_encoder_concat" not in {
        row["name"] for row in official_payload.header["tensor_manifest"]
    }
    assert "tub.output2_raw" not in {
        row["name"] for row in official_payload.header["tensor_manifest"]
    }
    assert decoded.metadata["score_aware_long_training_executed"] is True
    assert decoded.metadata["score_aware_long_training"]["executed"] is True
    assert decoded.metadata["score_aware_long_training"]["official_mfu_hfr_tub_train_export"][
        "trained_receiver_payload_exported"
    ] is True
    frames = decode_snerv_archive_frames(Path(report["packet_path"]).read_bytes())
    assert frames.shape == (2, 2, 3, 16, 16)
    assert np.isfinite(frames).all()
    target_profile = report["receiver_target_reconstruction_profile"]
    export_profile = report["receiver_export_reconstruction_profile"]
    assert target_profile["profile_id"] == "selected_packet_vs_source_targets"
    assert "official_mfu_hfr_tub" in target_profile["packet_source"]
    assert target_profile["shape_matches"] is True
    assert target_profile["blockers"] == []
    assert export_profile["profile_id"] == "selected_packet_vs_export_reference"
    assert export_profile["reference_kind"] == (
        "score_aware_long_training_selected_pairs_nchw255"
    )
    assert export_profile["shape_matches"] is True
    assert export_profile["blockers"] == []
    assert np.isfinite(float(export_profile["mse_nchw255"]))
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_official_primitives_long_training_consumes_checkpoint_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    state_path = tmp_path / "official_state_dict_slice.npz"
    np.savez(state_path, **_minimal_full_official_decoder_state())

    pairs = _tiny_pairs(pairs=2)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)
    monkeypatch.setattr(
        mod,
        "build_snerv_official_tub_source_forward_replay_artifact",
        lambda: _fake_tub_fixture_replay_passed(),
    )

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "official_long_training_checkpoint_bound",
        num_pairs=2,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "candidate_id": "official-primitives-checkpoint-bound",
            "snerv_model_size_adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "decoder_payload_codec": "int8_symmetric",
            "snerv_fc_dim": 9,
            "score_aware_long_training_epochs": 1,
            "score_aware_long_training_batch_pairs": 2,
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
        official_trained_checkpoint_state_dict_path=state_path,
    )

    mapping_blockers = {
        "snerv_official_mfu_hfr_tub_weight_mapping_missing",
        "snerv_official_trained_checkpoint_state_dict_not_loaded",
        "snerv_official_trained_checkpoint_state_dict_mapping_missing",
        "snerv_official_tub_trained_temporal_encoder_decoder_weights_not_loaded",
        "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing",
        "snerv_official_tub_portable_output2_decoder_weight_mapping_missing",
    }
    assert mapping_blockers.isdisjoint(set(report["blockers"]))
    assert "snerv_official_trained_checkpoint_source_forward_replay_missing" in report[
        "blockers"
    ]

    long_training = report["score_aware_long_training"]
    train_export = long_training["official_mfu_hfr_tub_train_export"]
    assert train_export["trained_receiver_payload_exported"] is True
    assert train_export["trained_weight_mapping_to_long_training_bound"] is True
    assert train_export["official_trained_checkpoint_state_dict_loaded"] is True
    assert (
        train_export["official_trained_checkpoint_state_dict_mapping_verified"]
        is True
    )
    assert mapping_blockers.isdisjoint(set(train_export["authority_blockers"]))
    assert train_export["authority_blockers"] == [
        "snerv_official_trained_checkpoint_source_forward_replay_missing"
    ]

    replay = long_training["official_mfu_hfr_tub_source_forward_replay"]
    assert replay["trained_weight_mapping_to_long_training_bound"] is True
    assert replay["official_trained_checkpoint_loaded"] is True
    assert replay["official_trained_checkpoint_state_dict_mapping_verified"] is True
    assert mapping_blockers.isdisjoint(set(replay["blockers"]))
    assert "snerv_official_trained_checkpoint_source_forward_replay_missing" in replay[
        "blockers"
    ]
    rows = {row["component_id"]: row for row in replay["component_rows"]}
    assert rows["mfu"]["trained_weight_mapping_to_long_training_bound"] is True
    assert rows["hfr"]["trained_weight_mapping_to_long_training_bound"] is True
    assert rows["tub"]["trained_weight_mapping_to_long_training_bound"] is True
    assert (
        "snerv_mfu_source_forward_replay_requires_upstream_torch_state_dict_mapping"
        not in rows["mfu"]["blockers"]
    )
    assert (
        "snerv_hfr_source_forward_replay_requires_upstream_torch_state_dict_mapping"
        not in rows["hfr"]["blockers"]
    )

    primitive_binding = report["official_primitive_binding"]
    tensor_map = primitive_binding["official_receiver_tensor_map"]
    assert tensor_map["official_state_dict_mapping_verified"] is True
    assert tensor_map["official_weight_mapping_blocker_closed"] is True
    assert tensor_map["official_weight_mapping_blockers"] == []
    assert mapping_blockers.isdisjoint(set(primitive_binding["blockers"]))


def test_official_long_training_keeps_trained_packet_with_nonrender_blocker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=2)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    def forced_telemetry_blocker(*_args, **_kwargs):
        return {
            "schema": "snerv_score_aware_long_training_telemetry_contract.v1",
            "telemetry_exists": True,
            "row_count": 1,
            "passed": False,
            "blockers": ["unit_nonrender_telemetry_blocker"],
        }

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)
    monkeypatch.setattr(
        mod,
        "build_snerv_official_tub_source_forward_replay_artifact",
        lambda: _fake_tub_fixture_replay_passed(),
    )
    monkeypatch.setattr(
        mod,
        "_snerv_score_aware_long_training_telemetry_contract",
        forced_telemetry_blocker,
    )

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "official_long_training_blocked_but_trained",
        num_pairs=2,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "candidate_id": "official-primitives-trained-state-exportable",
            "snerv_model_size_adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "decoder_payload_codec": "int8_symmetric",
            "snerv_fc_dim": 9,
            "score_aware_long_training_epochs": 1,
            "score_aware_long_training_batch_pairs": 2,
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
    )

    assert report["score_aware_long_training_executed"] is True
    assert report["score_aware_long_training_trained_state_exportable"] is True
    assert "unit_nonrender_telemetry_blocker" in report["blockers"]
    assert report["packet_source"] == "official_mfu_hfr_tub_mlx_trained_payload_atoms"
    long_training = report["score_aware_long_training"]
    assert long_training["executed"] is True
    assert long_training["training_completed"] is True
    assert long_training["blocker_free_execution"] is False
    assert long_training["trained_state_exportable"] is True
    assert long_training["official_mfu_hfr_tub_train_export"][
        "trained_receiver_payload_exported"
    ] is True
    decoded = unpack_snerv_archive(Path(report["packet_path"]).read_bytes())
    assert decoded.schema == "snerv_inverse_steg_archive.snar2.v1"
    assert decoded.metadata["score_aware_long_training_executed"] is True
    assert decoded.metadata["score_aware_long_training"]["executed"] is True
    assert decoded.score_claim is False
    assert decoded.promotion_eligible is False
    assert decoded.ready_for_exact_eval_dispatch is False
    assert report["score_aware_long_training_executed"] is True
    assert report["score_aware_long_training_trained_state_exportable"] is True
    assert report["native_mlx_training_executed"] is True
    assert report["score_aware_long_training"]["blocker_free_execution"] is False
    assert report["score_aware_long_training"]["trained_state_exportable"] is True
    assert report["score_aware_long_training"][
        "official_mfu_hfr_tub_train_export"
    ]["trained_receiver_payload_exported"] is True


@pytest.mark.parametrize(
    ("mode", "codec", "stored_shape"),
    [
        ("shared_mean", "shared_mean_float64", [1, 3, 8, 8]),
        ("channel_mean", "channel_mean_float64", [1, 3, 1, 1]),
        ("scalar_mean", "scalar_mean_float64", [1, 1, 1, 1]),
    ],
)
def test_official_primitives_long_training_compact_skip_high_exports_full_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    codec: str,
    stored_shape: list[int],
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=2)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / f"official_long_training_{mode}",
        num_pairs=2,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "candidate_id": f"official-primitives-long-training-{mode}",
            "snerv_model_size_adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "decoder_payload_codec": "int8_symmetric",
            "snerv_fc_dim": 9,
            "snerv_official_skip_high_mode": mode,
            "score_aware_long_training_epochs": 1,
            "score_aware_long_training_batch_pairs": 2,
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
    )

    assert report["score_aware_long_training_executed"] is True
    assert report["score_aware_long_training"]["executed"] is True
    gate = report["official_skip_high_value_domain_gate"]
    assert gate["official_skip_high_mode"] == mode
    assert gate["compact_skip_high_mode"] is True
    assert gate["passed"] is True
    assert gate["blockers"] == []
    replay = report["score_aware_long_training"][
        "official_mfu_hfr_tub_source_forward_replay"
    ]
    assert replay["receiver_official_payload_decode_replay_passed"] is True
    assert replay["receiver_official_payload_forward_replay_passed"] is True
    assert replay["receiver_official_payload_forward_replay_scope"] == (
        "archive_payload_decode_and_self_consistency_not_target_distortion"
    )
    assert replay["target_reconstruction_within_tolerance"] is False
    assert np.isfinite(float(replay["max_abs_error_nchw255"]))
    assert "snerv_official_mfu_hfr_tub_receiver_payload_replay_failed" not in replay[
        "blockers"
    ]
    assert "snerv_official_mfu_hfr_tub_target_reconstruction_outside_tolerance" in replay[
        "blockers"
    ]
    decoded = unpack_snerv_archive(Path(report["packet_path"]).read_bytes())
    official_payload = decoded.decode_official_mfu_hfr_tub_payload()
    storage = official_payload.header["skip_high_storage"]
    assert storage["codec"] == codec
    assert storage["source_shape"] == [4, 3, 8, 8]
    assert storage["stored_shape"] == stored_shape
    assert storage["encoder_consumed_compact_train_state"] is True
    assert report["official_skip_high_export_storage_shape"] == stored_shape
    assert report["official_skip_high_export_is_compact_train_state"] is True
    assert official_payload.tensors["inputs.mfu.skip_high"].shape == (4, 3, 8, 8)
    frames = decode_snerv_archive_frames(Path(report["packet_path"]).read_bytes())
    assert frames.shape == (2, 2, 3, 16, 16)


@pytest.mark.parametrize("mode", ["shared_mean", "channel_mean", "scalar_mean"])
def test_official_compact_skip_high_gate_blocks_collapsed_value_domain(
    mode: str,
) -> None:
    failed_profile = {
        "receiver_value_domain_gate": {
            "passed": False,
            "blockers": [
                "snerv_receiver_frame_reconstruction_decoded_std_collapsed"
            ],
        }
    }

    gate = _snerv_official_skip_high_value_domain_gate(
        {"official_skip_high_mode": mode},
        receiver_target_profile=failed_profile,
        receiver_export_profile=failed_profile,
    )

    assert gate["compact_skip_high_mode"] is True
    assert gate["passed"] is False
    assert "snerv_official_compact_skip_high_target_value_domain_not_passed" in gate[
        "blockers"
    ]
    assert "snerv_official_compact_skip_high_export_value_domain_not_passed" in gate[
        "blockers"
    ]
    assert f"snerv_official_{mode}_skip_high_collapse_risk" in gate["blockers"]


def test_official_primitives_full_video_long_training_defers_replay_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pair_count = mod.SNERV_OFFICIAL_LONG_TRAINING_REPLAY_MAX_PAIRS + 1
    pairs = _tiny_pairs(pairs=pair_count)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    def fail_if_expensive_replay_is_called(*_args, **_kwargs):
        raise AssertionError("full-video training should defer pretraining replay")

    class FakeArtifact:
        def as_dict(self) -> dict[str, object]:
            return {
                "schema": "mlx_score_aware_training_artifact.v1",
                "substrate_id": "snerv_inverse_steg_carrier",
                "lane_id": "lane_snerv_mlx_score_aware_train_export",
                "total_epochs_completed": 1,
                "total_wall_clock_seconds": 0.1,
                "telemetry_path": "",
                "live_checkpoint_path": "",
                "ema_shadow_checkpoint_path": "",
            }

    harness_calls: list[dict[str, object]] = []

    def fake_run_mlx_score_aware_full_main(**kwargs):
        harness_calls.append(kwargs)
        on_epoch_end = kwargs.get("on_epoch_end")
        if on_epoch_end is not None:
            on_epoch_end(SimpleNamespace(epoch=0, loss=0.0))
        return FakeArtifact()

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)
    monkeypatch.setattr(
        mod,
        "build_snerv_official_tub_source_forward_replay_artifact",
        lambda: _fake_tub_fixture_replay_passed(),
    )
    monkeypatch.setattr(
        mod,
        "_build_official_mfu_hfr_tub_long_training_replay_contract",
        fail_if_expensive_replay_is_called,
    )
    monkeypatch.setattr(
        "tac.substrates._shared.mlx_score_aware.harness.run_mlx_score_aware_full_main",
        fake_run_mlx_score_aware_full_main,
    )

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "official_full_video_training_deferred_replay",
        num_pairs=pair_count,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "candidate_id": "official-primitives-full-video-training-request",
            "snerv_model_size_adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "decoder_payload_codec": "int8_symmetric",
            "snerv_fc_dim": 9,
            "score_aware_long_training_epochs": 1,
            "score_aware_long_training_batch_pairs": pair_count,
            "score_aware_long_training_scorer_input_distribution_guard_weight": 0.0,
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
    )

    assert harness_calls
    assert report["score_aware_long_training_executed"] is True
    assert report["snerv_official_mfu_hfr_tub_export_bound"] is True
    blocker = (
        "snerv_official_mfu_hfr_tub_receiver_payload_replay_deferred_full_video"
    )
    assert blocker in report["blockers"]
    replay = report["score_aware_long_training"][
        "official_mfu_hfr_tub_source_forward_replay"
    ]
    assert Path(replay["artifact_path"]).is_file()
    replay_on_disk = json.loads(Path(replay["artifact_path"]).read_text())
    assert replay["deferred_for_full_video_training_start"] is True
    assert replay_on_disk["deferred_for_full_video_training_start"] is True
    assert replay["defer_threshold_pairs"] == (
        mod.SNERV_OFFICIAL_LONG_TRAINING_REPLAY_MAX_PAIRS
    )
    assert replay["requested_pair_count"] == pair_count
    assert replay["receiver_official_payload_forward_replay_passed"] is False
    assert replay["score_aware_long_training_renderer_bound"] is True
    assert replay["train_renderer_bound"] is True
    assert replay["official_tub_fixture_source_forward_replay_proven"] is True
    assert replay_on_disk["score_aware_long_training_renderer_bound"] is True
    assert replay_on_disk["train_renderer_bound"] is True
    assert blocker in replay["blockers"]
    assert "snerv_official_mfu_hfr_tub_source_forward_replay_missing" not in replay[
        "blockers"
    ]
    assert "snerv_official_snerv_t_trained_full_tub_source_forward_parity_missing" in replay[
        "blockers"
    ]
    assert all(
        row["train_renderer_bound"] is True
        and row["receiver_payload_forward_replay_proven"] is False
        and blocker in row["blockers"]
        for row in replay["component_rows"]
    )
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_official_hfr_bootstrap_least_squares_caps_design_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    ll = np.arange(2 * 3 * 8 * 8, dtype=np.float64).reshape(2, 3, 8, 8)
    detail = ll * 0.25
    seen: dict[str, tuple[int, ...]] = {}

    def fake_lstsq(design, target, rcond=None):
        seen["design_shape"] = tuple(int(v) for v in design.shape)
        seen["target_shape"] = tuple(int(v) for v in target.shape)
        beta = np.zeros((int(design.shape[1]), int(target.shape[1])), dtype=np.float64)
        return beta, np.empty((0,), dtype=np.float64), 0, np.empty((0,), dtype=np.float64)

    monkeypatch.setattr(np.linalg, "lstsq", fake_lstsq)

    head = mod._fit_official_hfr_head_from_ll(ll, detail, max_rows=17)

    assert seen["design_shape"] == (17, 28)
    assert seen["target_shape"] == (17, 3)
    assert head.conv2.weight.shape == (3, 3, 3, 3)
    assert head.conv2.bias.shape == (3,)


def test_official_renderer_coder_qat_selects_hfr_decoder_weights() -> None:
    pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod
    from tac.substrates._shared.mlx_score_aware.coder_qat import (
        CoderAwareQATConfig,
        build_decoder_coder_qat_terms,
        coder_qat_metadata,
    )
    from tac.substrates.snerv_inverse_steg_carrier.mlx_renderer import (
        SnervMlxOfficialMfuHfrTubScoreRenderer,
    )

    pairs = _tiny_pairs(pairs=1)
    model_size = SnervModelSizeConfig(
        adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        fc_dim=9,
    )
    components = mod._official_mfu_hfr_tub_bootstrap_components_from_pairs(
        pairs,
        model_size=model_size,
    )
    model = SnervMlxOfficialMfuHfrTubScoreRenderer(
        mfu=components["mfu"],
        hfr_heads=components["hfr_heads"],
        low=components["low"],
        skip_mid=components["skip_mid"],
        skip_high=components["skip_high"],
        output_hw=(16, 16),
        model_size=model_size,
        tub_current=components["tub_current"],
        tub_previous=components["tub_previous"],
        tub_next_frame=components["tub_next_frame"],
    )
    cfg = CoderAwareQATConfig(
        enabled=True,
        quant_bits=4,
        quant_residual_weight=1.0,
        magnitude_weight=1.0,
        delta_weight=1.0,
        c1a_entropy_weight=1.0,
        c1a_sample_size=8,
    ).validated()

    terms = build_decoder_coder_qat_terms(model, cfg)
    metadata = coder_qat_metadata(cfg)

    assert "hfr_" in metadata["include_substrings"]
    assert "mfu_" in metadata["include_substrings"]
    assert set(terms) == {
        "coder_qat_quant_residual",
        "coder_qat_magnitude",
        "coder_qat_delta",
        "coder_qat_c1a_entropy",
    }


def test_official_renderer_frame_bound_output2_moves_loss_path() -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod
    from tac.substrates.snerv_inverse_steg_carrier.mlx_renderer import (
        SnervMlxOfficialMfuHfrTubScoreRenderer,
    )

    pairs = _tiny_pairs(pairs=1)
    model_size = SnervModelSizeConfig(
        adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        fc_dim=9,
    )
    components = mod._official_mfu_hfr_tub_bootstrap_components_from_pairs(
        pairs,
        model_size=model_size,
    )
    temporal = np.linspace(0.0, 1.0, 1 * 6 * 8 * 8, dtype=np.float32).reshape(
        1,
        6,
        8,
        8,
    )
    output2_raw = np.full((2, 12, 8, 8), 0.5, dtype=np.float32)

    base = SnervMlxOfficialMfuHfrTubScoreRenderer(
        mfu=components["mfu"],
        hfr_heads=components["hfr_heads"],
        low=components["low"],
        skip_mid=components["skip_mid"],
        skip_high=components["skip_high"],
        output_hw=(16, 16),
        model_size=model_size,
        tub_current=components["tub_current"],
        tub_previous=components["tub_previous"],
        tub_next_frame=components["tub_next_frame"],
    )
    with_output2 = SnervMlxOfficialMfuHfrTubScoreRenderer(
        mfu=components["mfu"],
        hfr_heads=components["hfr_heads"],
        low=components["low"],
        skip_mid=components["skip_mid"],
        skip_high=components["skip_high"],
        output_hw=(16, 16),
        model_size=model_size,
        tub_current=components["tub_current"],
        tub_previous=components["tub_previous"],
        tub_next_frame=components["tub_next_frame"],
        tub_temporal_encoder_concat=temporal,
        tub_output2_raw=output2_raw,
        tub_output2_fc_hw=(2, 2),
    )

    indices = mx.array([0], dtype=mx.int32)
    base_render = base(indices)
    output2_render = with_output2(indices)
    mx.eval(base_render, output2_render)
    delta = np.asarray(output2_render - base_render, dtype=np.float32)

    assert float(np.max(delta)) > 0.25
    metadata = with_output2.metadata()
    assert metadata["official_tub_output2_receiver_frame_bound"] is True
    assert metadata["official_tub_output2_payload_loss_coupled"] is True
    assert metadata["official_tub_output2_fused_shape"] == [2, 3, 16, 16]
    exported = with_output2.export_official_components()
    assert exported["fc_hw"] == (2, 2)
    assert exported["temporal_encoder_output_shape"] == (1, 6, 8, 8)
    assert exported["output2_decoder_output_shape"] == (2, 12, 8, 8)


def test_official_renderer_non_frame_bound_output2_is_not_loss_coupled() -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod
    from tac.substrates.snerv_inverse_steg_carrier.mlx_renderer import (
        SnervMlxOfficialMfuHfrTubScoreRenderer,
    )

    pairs = _tiny_pairs(pairs=2)
    model_size = SnervModelSizeConfig(
        adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        fc_dim=9,
    )
    components = mod._official_mfu_hfr_tub_bootstrap_components_from_pairs(
        pairs,
        model_size=model_size,
    )
    temporal = np.linspace(0.0, 1.0, 1 * 6 * 8 * 8, dtype=np.float32).reshape(
        1,
        6,
        8,
        8,
    )
    output2_raw = np.full((2, 12, 8, 8), 0.5, dtype=np.float32)

    base = SnervMlxOfficialMfuHfrTubScoreRenderer(
        mfu=components["mfu"],
        hfr_heads=components["hfr_heads"],
        low=components["low"],
        skip_mid=components["skip_mid"],
        skip_high=components["skip_high"],
        output_hw=(16, 16),
        model_size=model_size,
        tub_current=components["tub_current"],
        tub_previous=components["tub_previous"],
        tub_next_frame=components["tub_next_frame"],
    )
    with_output2 = SnervMlxOfficialMfuHfrTubScoreRenderer(
        mfu=components["mfu"],
        hfr_heads=components["hfr_heads"],
        low=components["low"],
        skip_mid=components["skip_mid"],
        skip_high=components["skip_high"],
        output_hw=(16, 16),
        model_size=model_size,
        tub_current=components["tub_current"],
        tub_previous=components["tub_previous"],
        tub_next_frame=components["tub_next_frame"],
        tub_temporal_encoder_concat=temporal,
        tub_output2_raw=output2_raw,
        tub_output2_fc_hw=(2, 2),
    )

    metadata = with_output2.metadata()
    assert metadata["official_tub_output2_payload_export_bound"] is True
    assert metadata["official_tub_output2_receiver_frame_shape"] == [4, 3, 16, 16]
    assert metadata["official_tub_output2_fused_shape"] == [2, 3, 16, 16]
    assert metadata["official_tub_output2_receiver_frame_bound"] is False
    assert metadata["official_tub_output2_payload_loss_coupled"] is False

    indices = mx.array([0], dtype=mx.int32)
    base_render = base(indices)
    output2_render = with_output2(indices)
    mx.eval(base_render, output2_render)
    assert np.allclose(
        np.asarray(output2_render, dtype=np.float32),
        np.asarray(base_render, dtype=np.float32),
        atol=1.0e-6,
    )


def test_train_export_official_primitives_receiver_proof_stays_surrogate_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=1)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)
    exported: dict[str, object] = {}

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    def fake_export_snerv_mlx_archive(
        *,
        model_or_artifact,
        output_dir,
        repo_root,
        retain_receiver_output=False,
        receiver_proof_timeout_seconds=1800,
        **_kwargs,
    ):
        packet_path = Path(model_or_artifact["packet_path"])
        packet = packet_path.read_bytes()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        archive_path = out / "archive.zip"
        archive_path.write_bytes(b"zip:" + packet[:16])
        proof_path = out / "receiver_proof.json"
        proof_path.write_text('{"runtime_consumption_proof_passed":true}\n')
        exported["packet_sha256"] = hashlib.sha256(packet).hexdigest()
        exported["repo_root"] = Path(repo_root).as_posix()
        exported["retain_receiver_output"] = retain_receiver_output
        exported["timeout"] = receiver_proof_timeout_seconds
        return {
            "schema": "fake_snerv_mlx_archive_package.v1",
            "receiver_proof": {
                "archive_path": archive_path.as_posix(),
                "archive_bytes": archive_path.stat().st_size,
                "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                "proof_path": proof_path.as_posix(),
                "runtime_consumption_proof_passed": True,
                "receiver_contract_satisfied": True,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)
    monkeypatch.setattr(mod, "export_snerv_mlx_archive", fake_export_snerv_mlx_archive)

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "official_archive_bound_surrogate",
        num_pairs=1,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "candidate_id": "official-primitives-request",
            "snerv_model_size_adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "decoder_payload_codec": "int8_symmetric",
            "snerv_fc_dim": 9,
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=True,
        retain_receiver_output=False,
        receiver_proof_timeout_seconds=77,
    )

    assert exported["packet_sha256"] == report["packet_sha256"]
    assert exported["timeout"] == 77
    assert report["receiver_proof_passed"] is True
    assert report["receiver_contract_satisfied"] is True
    assert "snerv_official_receiver_runtime_decode_missing" not in report["blockers"]
    assert "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload" not in report["blockers"]
    assert "snerv_official_mfu_hfr_tub_weight_mapping_missing" in report["blockers"]
    binding = report["official_primitive_binding"]
    assert binding["official_receiver_tensor_map"]["receiver_tensor_map_verified"] is True
    assert (
        binding["official_receiver_tensor_map"]["official_weight_mapping_blocker_closed"]
        is False
    )
    surrogate = binding["receiver_bound_surrogate_export"]
    assert surrogate["archive_sha256"] == report["archive_sha256"]
    assert surrogate["surrogate_receiver_contract_satisfied"] is True
    assert surrogate["surrogate_runtime_consumption_proof_passed"] is True
    assert binding["receiver_runtime_decode_authority"] is True
    assert binding["selected_packet_official_payload_runtime_decode_authority"] is True
    assert binding["selected_packet_frame_producing_official_export"] is True
    assert binding["selected_packet_authority"]["status"] == (
        "frame_producing_official_export"
    )
    assert "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload" not in {
        row["blocker"] for row in binding["blocker_evidence"]
    }
    evidence = {row["blocker"]: row for row in binding["blocker_evidence"]}
    assert evidence["snerv_official_mfu_hfr_tub_weight_mapping_missing"][
        "official_authority"
    ] is False
    assert evidence["snerv_official_mfu_hfr_tub_weight_mapping_missing"][
        "source_forward_authority"
    ] is False
    assert evidence["snerv_official_mfu_hfr_tub_weight_mapping_missing"][
        "receiver_payload_binding_authority"
    ] is False
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_train_export_blocks_over_hard_byte_ceiling_using_measured_archive_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=1)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    def fake_export_snerv_mlx_archive(
        *,
        model_or_artifact,
        output_dir,
        repo_root,
        retain_receiver_output=False,
        receiver_proof_timeout_seconds=1800,
        **_kwargs,
    ):
        del repo_root, retain_receiver_output, receiver_proof_timeout_seconds
        packet = Path(model_or_artifact["packet_path"]).read_bytes()
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        archive_path = out / "archive.zip"
        archive_path.write_bytes(b"charged-archive:" + packet[:64])
        proof_path = out / "receiver_proof.json"
        proof_path.write_text('{"runtime_consumption_proof_passed":true}\n')
        return {
            "schema": "fake_snerv_mlx_archive_package.v1",
            "receiver_proof": {
                "archive_path": archive_path.as_posix(),
                "archive_bytes": archive_path.stat().st_size,
                "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                "proof_path": proof_path.as_posix(),
                "runtime_consumption_proof_passed": True,
                "receiver_contract_satisfied": True,
            },
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)
    monkeypatch.setattr(mod, "export_snerv_mlx_archive", fake_export_snerv_mlx_archive)

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "over_hard_cap",
        num_pairs=1,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "candidate_id": "over-hard-cap",
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int8_symmetric",
            "hard_byte_ceiling": 8,
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=True,
    )

    cap = report["byte_cap_control"]
    assert cap["attached"] is True
    assert cap["enforced"] is True
    assert cap["hard_byte_ceiling"] == 8
    assert cap["archive_bytes"] == report["archive_bytes"]
    assert cap["under_hard_byte_ceiling"] is False
    assert cap["delta_bytes_vs_hard_byte_ceiling"] == report["archive_bytes"] - 8
    assert cap["section_bytes"]["lf_payload"] == cap["lf_payload_bytes"]
    assert cap["lf_payload_bytes"] > 8
    assert cap["lf_payload_exceeds_hard_byte_ceiling"] is True
    assert cap["lf_payload_fraction_of_packet"] > 0.0
    assert "snerv_lf_payload_exceeds_hard_byte_ceiling" in cap["blockers"]
    assert (
        "snerv_lf_payload_recode_or_representation_change_required_for_hard_ceiling"
        in cap["blockers"]
    )
    assert "snerv_mlx_native_archive_exceeds_hard_byte_ceiling" in cap["blockers"]
    assert "snerv_mlx_native_archive_exceeds_hard_byte_ceiling" in report["blockers"]
    assert "snerv_lf_payload_exceeds_hard_byte_ceiling" in report["blockers"]
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_byte_cap_control_reports_lf_pressure_without_under_ceiling_blocker() -> None:
    cap = _build_snerv_mlx_native_byte_cap_control(
        candidate={"candidate_id": "under-cap"},
        hard_byte_ceiling=512,
        packet_source="unit_selected_packet",
        packet_sha256="b" * 64,
        packet_bytes=300,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="v2:signed_int2_bitpack:none",
        section_bytes={
            "metadata_payload": 12,
            "lf_payload": 190,
            "decoder_payload": 64,
            "step_map_packet": 34,
        },
        archive_bytes=360,
        archive_sha256="a" * 64,
        receiver_proof_passed=True,
        receiver_contract_satisfied=True,
        run_archive_export=True,
    )

    assert cap["attached"] is True
    assert cap["enforced"] is True
    assert cap["archive_bytes_authoritative"] is True
    assert cap["authority"] == "measured_receiver_proven_archive_zip_bytes"
    assert cap["packet_source"] == "unit_selected_packet"
    assert cap["packet_sha256"] == "b" * 64
    assert cap["decoder_payload_codec"] == "int8_symmetric"
    assert cap["lf_payload_codec"] == "v2:signed_int2_bitpack:none"
    assert cap["under_hard_byte_ceiling"] is True
    assert cap["delta_bytes_vs_hard_byte_ceiling"] == -152
    assert cap["lf_payload_bytes"] == 190
    assert cap["largest_section_name"] == "lf_payload"
    assert cap["lf_payload_is_largest_section"] is True
    assert cap["lf_payload_exceeds_hard_byte_ceiling"] is False
    assert cap["lf_payload_can_cover_archive_overrun"] is None
    assert cap["blockers"] == []


def test_byte_cap_control_rejects_archive_bytes_without_receiver_proof() -> None:
    cap = _build_snerv_mlx_native_byte_cap_control(
        candidate={"candidate_id": "proof-missing"},
        hard_byte_ceiling=512,
        packet_source="unit_selected_packet",
        packet_sha256="b" * 64,
        packet_bytes=300,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="portfolio_auto",
        section_bytes={
            "metadata_payload": 12,
            "lf_payload": 190,
            "decoder_payload": 64,
            "step_map_packet": 34,
        },
        archive_bytes=360,
        archive_sha256="a" * 64,
        receiver_proof_passed=False,
        receiver_contract_satisfied=False,
        run_archive_export=True,
    )

    assert cap["attached"] is True
    assert cap["enforced"] is False
    assert cap["archive_bytes_authoritative"] is False
    assert cap["under_hard_byte_ceiling"] is None
    assert cap["delta_bytes_vs_hard_byte_ceiling"] is None
    assert (
        cap["authority"]
        == "archive_bytes_not_authoritative_until_receiver_proof_passes"
    )
    assert (
        "snerv_mlx_native_hard_byte_ceiling_receiver_proof_missing_or_failed"
        in cap["blockers"]
    )


def test_byte_cap_control_exposes_official_component_pressure_rows() -> None:
    cap = _build_snerv_mlx_native_byte_cap_control(
        candidate={"candidate_id": "official-pressure"},
        hard_byte_ceiling=700,
        packet_source="official_mfu_hfr_tub_mlx_trained_payload_atoms",
        packet_sha256="b" * 64,
        packet_bytes=800,
        decoder_payload_codec="official_mfu_hfr_tub_payload",
        lf_payload_codec="none",
        section_bytes={
            "metadata_payload": 16,
            "lf_payload": 80,
            "decoder_payload": 500,
            "step_map_packet": 24,
        },
        official_receiver_tensor_map={
            "receiver_tensor_map_verified": True,
            "total_tensor_bytes": 400,
            "category_bytes": {
                "official_mfu_weight_payload": 160,
                "official_hfr_weight_payload": 40,
                "official_tub_output2_payload": 200,
            },
        },
        archive_bytes=1000,
        archive_sha256="b" * 64,
        receiver_proof_passed=True,
        receiver_contract_satisfied=True,
        run_archive_export=True,
    )

    assert cap["official_decoder_payload_component_pressure_bound"] is True
    assert cap["official_decoder_payload_component_bytes"] == {
        "official_hfr_weight_payload": 40,
        "official_mfu_weight_payload": 160,
        "official_tub_output2_payload": 200,
    }
    assert cap["official_decoder_payload_proof_only_component_bytes"] == {
        "official_tub_output2_payload": 200,
    }
    assert cap["official_decoder_payload_proof_only_component_total_bytes"] == 200
    assert cap["official_decoder_payload_non_score_causal_component_bytes"] == {
        "official_tub_output2_payload": 200,
    }
    assert cap["official_decoder_payload_non_score_causal_component_total_bytes"] == 200
    assert cap["official_decoder_payload_non_score_causal_byte_cap_action"] == (
        "elide_or_implement_source_faithful_receiver_frame_decode_before_score_candidate"
    )
    assert cap["largest_pressure_scope"] == "snar_archive_section"
    assert cap["largest_pressure_name"] == "decoder_payload"
    assert cap["largest_pressure_bytes"] == 500
    section_rows = {row["name"]: row for row in cap["section_pressure_rows"]}
    assert section_rows["decoder_payload"]["byte_basis"] == (
        "exact_receiver_packet_section_bytes"
    )
    component_rows = {
        row["name"]: row for row in cap["official_decoder_payload_component_rows"]
    }
    assert component_rows["official_tub_output2_payload"]["byte_basis"] == (
        "receiver_tensor_manifest_raw_float64_bytes_inside_single_lzma_decoder_payload"
    )
    assert component_rows["official_tub_output2_payload"][
        "fraction_of_official_raw_tensor_bytes"
    ] == pytest.approx(0.5)
    assert component_rows["official_tub_output2_payload"][
        "fraction_of_decoder_payload_section"
    ] == pytest.approx(0.4)
    assert (
        component_rows["official_mfu_weight_payload"][
            "receiver_frame_decode_bound"
        ]
        is True
    )
    assert (
        component_rows["official_mfu_weight_payload"]["byte_cap_action"]
        == "protect_quantize_or_waterfill_by_scorer_gradient"
    )
    assert (
        component_rows["official_tub_output2_payload"][
            "receiver_frame_decode_bound"
        ]
        is False
    )
    assert (
        component_rows["official_tub_output2_payload"][
            "waterfill_admission_class"
        ]
        == "receiver_activation_not_frame_decode_bound_rate_liability"
    )
    assert (
        component_rows["official_tub_output2_payload"]["byte_cap_action"]
        == "elide_unless_receiver_frame_decode_bound_or_scored_delta_positive"
    )
    assert (
        component_rows["official_tub_output2_payload"][
            "receiver_activation_payload_bound"
        ]
        is True
    )
    assert (
        component_rows["official_tub_output2_payload"][
            "receiver_activation_payload_score_causal"
        ]
        is False
    )
    assert "snerv_decoder_payload_is_largest_section_on_over_ceiling_export" in cap[
        "blockers"
    ]
    assert (
        "snerv_decoder_payload_component_recode_or_modelsize_change_required_for_hard_ceiling"
        in cap["blockers"]
    )
    assert (
        "snerv_official_mfu_hfr_tub_component_byte_pressure_requires_modelsize_waterfill"
        in cap["blockers"]
    )
    assert (
        "snerv_official_mfu_hfr_tub_proof_only_component_bytes_require_ablation_before_modelsize_growth"
        in cap["blockers"]
    )


def test_native_export_modelsize_candidate_consumes_official_fc_dim_solution() -> None:
    model_size = _model_size_from_candidate(
        {
            "candidate_id": "official-modelsize-only",
            "fc_dim": 3,
            "snerv_fc_dim": 5,
            "modelsize_mparams": 0.05,
            "official_modelsize_solution": {
                "schema": "official_snerv_modelsize_to_fc_dim.v1",
                "modelsize_mparams": 0.05,
                "fc_dim": 11,
            },
        }
    )

    assert model_size.fc_dim == 11
    assert model_size.fc_dim_source == "official_modelsize_solution"
    assert model_size.feature_count == 11


def test_native_export_modelsize_candidate_recomputes_fc_dim_when_formula_inputs_exist() -> None:
    model_size = _model_size_from_candidate(
        {
            "candidate_id": "official-modelsize-formula",
            "fc_dim": 3,
            "snerv_fc_dim": 5,
            "modelsize_mparams": 0.05,
            "full_data_length": 1200,
            "final_size": 384 * 512,
            "enc_strds": [5, 4, 2, 2, 2],
            "dec_strds": [5, 4, 2, 2, 2],
        }
    )

    assert model_size.fc_dim == 11
    assert model_size.fc_dim_source == "official_modelsize_formula"
    assert model_size.feature_count == 11


def test_native_export_modelsize_candidate_rejects_missing_formula_inputs() -> None:
    with pytest.raises(
        SnervCarrierError,
        match="modelsize_mparams requires official_modelsize_solution",
    ):
        _model_size_from_candidate(
            {
                "candidate_id": "official-modelsize-missing-formula-inputs",
                "modelsize_mparams": 0.05,
            }
        )


def test_native_export_without_modelsize_keeps_manual_default_fc_dim_source() -> None:
    model_size = _model_size_from_candidate(
        {
            "candidate_id": "manual-default-no-modelsize",
        }
    )
    assert model_size.fc_dim == 9
    assert model_size.fc_dim_source == "fallback_default_missing_official_modelsize_inputs"
    assert model_size.as_jsonable()["fc_dim_source"] == (
        "fallback_default_missing_official_modelsize_inputs"
    )


def test_native_export_modelsize_auto_elides_tub_output2_for_score_candidate() -> None:
    model_size = _model_size_from_candidate(
        {
            "candidate_id": "score-safe-tub-output2",
            "snerv_model_size_adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            "official_tub_output2_store_for_receiver_proof": True,
        }
    )

    assert model_size.official_tub_output2_store_for_receiver_proof_requested is True
    assert model_size.official_tub_output2_store_for_receiver_proof is False
    assert model_size.official_tub_output2_export_mode == "auto_elide"
    assert model_size.as_jsonable()[
        "official_tub_output2_store_for_receiver_proof_requested"
    ] is True
    assert model_size.as_jsonable()[
        "official_tub_output2_store_for_receiver_proof"
    ] is False


def test_native_export_modelsize_honors_explicit_tub_output2_proof_only_mode() -> None:
    model_size = _model_size_from_candidate(
        {
            "candidate_id": "proof-only-tub-output2",
            "snerv_model_size_adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            "official_tub_output2_store_for_receiver_proof": True,
            "official_tub_output2_export_mode": "proof_only",
        }
    )

    assert model_size.official_tub_output2_store_for_receiver_proof_requested is True
    assert model_size.official_tub_output2_store_for_receiver_proof is True
    assert model_size.official_tub_output2_export_mode == "proof_only"


def test_native_export_modelsize_accepts_tub_output2_receiver_frame_bound_mode() -> None:
    model_size = _model_size_from_candidate(
        {
            "candidate_id": "receiver-bound-tub-output2",
            "snerv_model_size_adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            "official_tub_output2_export_mode": "receiver_frame_bound",
        }
    )

    assert model_size.official_tub_output2_store_for_receiver_proof_requested is True
    assert model_size.official_tub_output2_store_for_receiver_proof is True
    assert model_size.official_tub_output2_export_mode == "receiver_frame_bound"
    assert model_size.as_jsonable()[
        "official_tub_output2_store_for_receiver_proof"
    ] is True


def test_native_export_modelsize_rejects_unknown_tub_output2_export_mode() -> None:
    with pytest.raises(
        SnervCarrierError,
        match="official_tub_output2_export_mode",
    ):
        _model_size_from_candidate(
            {
                "candidate_id": "bad-tub-output2-mode",
                "official_tub_output2_export_mode": "score_candidate_store",
            }
        )


def test_train_export_preserves_explicit_source_pair_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=2)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)
    captured: dict[str, object] = {}

    def fake_decode_mlx_targets(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = dict(kwargs)
        return target0, target1

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path,
        num_pairs=600,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int8_symmetric",
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        pair_indices=[7, 2, 7],
        run_archive_export=False,
    )

    assert captured["kwargs"]["num_pairs"] == 2
    assert captured["kwargs"]["pair_indices"] == (7, 2)
    assert report["num_pairs"] == 2
    assert list(report["source_pair_indices"]) == [7, 2]
    assert report["storage_preflight"]["n_pairs"] == 2
    decoded = unpack_snerv_archive(Path(report["packet_path"]).read_bytes())
    assert decoded.metadata["source_pair_indices"] == [7, 2]
    assert decoded.metadata["pair_index_alignment_mode"] == ("explicit_source_pair_indices")


def test_train_export_consumes_file_backed_recon_pixel_weight_with_custody(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=1)
    pairs[0, 1, 0, 3:12, 4:13] += 24.0
    pairs = np.clip(pairs, 0.0, 255.0)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)
    weight = np.ones((1, 2, 16, 16, 1), dtype=np.float32)
    weight[:, :, 3:12, 4:13, :] = 16.0
    weight_path = tmp_path / "joint_p18_p19_weight.npz"
    np.savez(weight_path, weight=weight)

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "export",
        num_pairs=1,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int8_symmetric",
            "hf_decoder_saliency_gain": 2.5,
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
        recon_pixel_weight_path=weight_path,
        recon_pixel_weight_normalize="mean",
    )

    assert report["packet_source"] == (
        "mlx_target_hydration_numpy_joint_p18_p19_dwt_adjoint_saliency_weighted_decoder_fit"
    )
    recon = report["recon_pixel_weight"]
    assert recon["enabled"] is True
    assert recon["source_kind"] == "file"
    assert recon["path"] == weight_path.as_posix()
    assert recon["sha256"]
    assert recon["npz_key"] == "weight"
    assert recon["normalize"] == "mean"
    assert recon["consumed_shape"] == [1, 2, 16, 16, 1]
    assert recon["score_claim"] is False
    assert recon["promotion_eligible"] is False
    decoded = unpack_snerv_archive(Path(report["packet_path"]).read_bytes())
    assert decoded.metadata["score_aware_hf_decoder_fit_executed"] is True
    assert decoded.metadata["hf_decoder_fit_mode"] == (SNERV_DWT_ADJOINT_SALIENCY_WEIGHTED_FIT_MODE)
    assert decoded.metadata["hf_decoder_weight_domain"] == ("dwt_adjoint_detail_saliency_diagonal")
    assert decoded.metadata["exact_pixel_weighted_objective"] is False
    assert decoded.metadata["contest_scorer_distortion_objective"] is False
    assert decoded.metadata["hf_decoder_saliency_gain"] == pytest.approx(2.5)
    assert decoded.metadata["recon_pixel_weight_consumed"] is True
    assert decoded.metadata["recon_pixel_weight_metadata"]["sha256"] == recon["sha256"]
    assert recon["producer_manifest_verified"] is False
    assert recon["producer_manifest"]["status"] == ("not_found_unverified_manual_or_legacy_weight")
    assert "snerv_recon_pixel_weight_verified_gradient_manifest_not_bound_to_native_export" in report["blockers"]


def test_train_export_certifies_verified_recon_pixel_weight_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=1)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)
    weight = np.ones((1, 2, 16, 16, 1), dtype=np.float32)
    weight_path = tmp_path / "joint_p18_p19_recon_pixel_weight.npz"
    np.savez_compressed(weight_path, weight=weight)
    weight_sha = mod.sha256_file(weight_path)
    manifest_path = tmp_path / "joint_p18_p19_recon_pixel_weight_manifest.json"
    gradient_health = {
        "schema": "joint_recon_pixel_weight_gradient_health.v1",
        "status": "pass_finite",
        "component_count": 2,
        "components_with_nonfinite": 0,
        "total_nonfinite_values": 0,
        "consumption_recommended": True,
    }
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "joint_p18_p19_recon_pixel_weight_manifest.v1",
                "weight_path": weight_path.as_posix(),
                "weight_sha256": weight_sha,
                "config": {
                    "num_pairs": 1,
                    "scorer_hw": [16, 16],
                },
                "metadata": {
                    "schema": "joint_p18_p19_recon_pixel_weight.v1",
                    "blockers": [],
                    "training_consumption_recommended": True,
                    "gradient_health": gradient_health,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "export_verified",
        num_pairs=1,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int8_symmetric",
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
        recon_pixel_weight_path=weight_path,
        recon_pixel_weight_manifest_path=manifest_path,
        recon_pixel_weight_normalize="mean",
    )

    recon = report["recon_pixel_weight"]
    assert recon["producer_manifest_verified"] is True
    assert recon["verification_status"] == "verified_finite_gradient_manifest"
    assert recon["producer_manifest"]["status"] == "verified_finite_gradient_manifest"
    assert recon["producer_manifest"]["consumption_certified"] is True
    assert recon["producer_manifest"]["gradient_health"] == gradient_health
    assert "snerv_recon_pixel_weight_verified_gradient_manifest_not_bound_to_native_export" not in report["blockers"]
    decoded = unpack_snerv_archive(Path(report["packet_path"]).read_bytes())
    assert decoded.metadata["recon_pixel_weight_metadata"]["producer_manifest_verified"] is True
    assert decoded.metadata["recon_pixel_weight_verified_gradient_manifest"] is True
    assert decoded.metadata["contest_scorer_distortion_objective"] is True


def test_train_export_refuses_recon_pixel_weight_manifest_sha_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=1)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)
    weight_path = tmp_path / "joint_p18_p19_recon_pixel_weight.npz"
    np.savez_compressed(
        weight_path,
        weight=np.ones((1, 2, 16, 16, 1), dtype=np.float32),
    )
    manifest_path = tmp_path / "joint_p18_p19_recon_pixel_weight_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "joint_p18_p19_recon_pixel_weight_manifest.v1",
                "weight_path": weight_path.as_posix(),
                "weight_sha256": "0" * 64,
                "config": {
                    "num_pairs": 1,
                    "scorer_hw": [16, 16],
                },
                "metadata": {
                    "schema": "joint_p18_p19_recon_pixel_weight.v1",
                    "blockers": [],
                    "training_consumption_recommended": True,
                    "gradient_health": {
                        "schema": "joint_recon_pixel_weight_gradient_health.v1",
                        "status": "pass_finite",
                        "component_count": 1,
                        "components_with_nonfinite": 0,
                        "total_nonfinite_values": 0,
                        "consumption_recommended": True,
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)

    with pytest.raises(
        mod.SnervMlxNativeExportError,
        match="producer manifest SHA does not match",
    ):
        train_export_snerv_mlx_native(
            output_dir=tmp_path / "export_stale_manifest",
            num_pairs=1,
            source_video_path="unit.mkv",
            modelsize_candidate={
                "levels": 1,
                "wavelet": "haar",
                "bits_per_coeff": 3.0,
                "decoder_payload_codec": "int8_symmetric",
            },
            scorer_upstream_dir="upstream",
            output_height=16,
            output_width=16,
            run_archive_export=False,
            recon_pixel_weight_path=weight_path,
            recon_pixel_weight_manifest_path=manifest_path,
        )


def test_train_export_refuses_bad_recon_pixel_weight_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=1)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)
    bad_weight_path = tmp_path / "bad_weight.npy"
    np.save(bad_weight_path, np.ones((8, 8), dtype=np.float32))

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)

    with pytest.raises(mod.SnervMlxNativeExportError, match="spatial shape"):
        train_export_snerv_mlx_native(
            output_dir=tmp_path / "export_bad",
            num_pairs=1,
            source_video_path="unit.mkv",
            modelsize_candidate={
                "levels": 1,
                "wavelet": "haar",
                "bits_per_coeff": 3.0,
                "decoder_payload_codec": "int8_symmetric",
            },
            scorer_upstream_dir="upstream",
            output_height=16,
            output_width=16,
            run_archive_export=False,
            recon_pixel_weight_path=bad_weight_path,
        )


def test_train_export_reports_actual_active_decoder_payload_codec(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    pairs = _tiny_pairs(pairs=1)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path,
        num_pairs=1,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "decoder_payload_codec": "float32_lzma",
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
        scorer_loop_qat_decoder_payload_codec="int8_symmetric",
    )

    decoded = unpack_snerv_archive(Path(report["packet_path"]).read_bytes())
    assert report["decoder_payload_codec"] == "int8_symmetric"
    assert decoded.metadata["decoder_payload_codec"] == "int8_symmetric"
    assert report["packet_source"] == "mlx_target_hydration_numpy_closed_form_decoder_fit"


def test_train_export_attaches_real_scorer_loop_qat_without_overclaiming(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod
    import tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat as qat_mod

    pairs = _tiny_pairs(pairs=1)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    raw_best_packet = build_snerv_mlx_native_packet_from_numpy_pairs(
        pairs + 1.0,
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="auto",
    ).packet
    raw_best_archive = unpack_snerv_archive(raw_best_packet)
    stripped_best_metadata = {
        key: value
        for key, value in raw_best_archive.metadata.items()
        if not key.startswith("step_map_")
    }
    best_packet = pack_snerv_archive(
        metadata_payload=raw_best_archive.sections["metadata_payload"],
        lf_payload=raw_best_archive.sections["lf_payload"],
        decoder_payload=raw_best_archive.sections["decoder_payload"],
        step_map_packet=raw_best_archive.sections["step_map_packet"],
        metadata=stripped_best_metadata,
    ).packet
    best_packet_metadata = unpack_snerv_archive(best_packet).metadata
    best_packet_sha256 = hashlib.sha256(best_packet).hexdigest()
    assert "step_map_coder_mode" not in best_packet_metadata

    class FakeQatResult:
        def __init__(self) -> None:
            self.best_packet = best_packet

        def as_jsonable(self) -> dict:
            return {
                "schema": "snerv_scorer_loop_decoder_qat_smoke.v1",
                "axis_tag": "[macOS-CPU advisory]",
                "n_pairs": 1,
                "decoder_payload_codec": "int8_symmetric",
                "lf_payload_codec": "v2:signed_int2_bitpack:none",
                "lf_payload_codec_requested": "portfolio_auto",
                "lf_payload_codec_selected": "v2:signed_int2_bitpack:none",
                "lf_payload_codec_selection_report": {
                    "schema": "snerv_lf_quant_payload.v2",
                    "mode_histogram": {"signed_int2_bitpack": 1},
                    "wrapper_histogram": {"none": 1},
                    "section_bytes": 42,
                },
                "perturb_scale": 0.0,
                "byte_pressure_multiplier": 2.0,
                "section_value_pressure_multiplier": 0.0,
                "max_archive_byte_growth": 0,
                "byte_growth_admission_mode": "rate_paid",
                "pose_slack": 0.0,
                "seg_slack": 0.0,
                "scorer_loop_evaluations": 2,
                "accepted_improvement": True,
                "receiver_contract_satisfied": True,
                "ready_for_pose_guard_gate": True,
                "baseline": {
                    "archive_bytes": 111,
                    "archive_sha256": "1" * 64,
                    "score_linf": 3.0,
                },
                "best": {
                    "archive_bytes": len(best_packet),
                    "archive_sha256": best_packet_sha256,
                    "score_linf": 2.5,
                },
                "best_packet_bytes": len(best_packet),
                "best_packet_sha256": best_packet_sha256,
                "component_guard_mode": "pose_seg_hard",
                "pair_robust_admission": {
                    "schema": "snerv_pair_robust_admission.v1",
                    "n_pairs": 1,
                    "min_score_improved_fraction": 1.0,
                    "max_pose_worsened_fraction": 0.0,
                    "pose_slack": 0.0,
                    "score_improved_fraction": 1.0,
                    "pose_worsened_fraction": 0.0,
                    "permissive_guard": False,
                    "passed": True,
                    "blockers": [],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "blockers": [
                    "snerv_scorer_loop_qat_best_packet_not_materialized_into_native_export",
                    "snerv_scorer_loop_qat_auxiliary_warning",
                    "snerv_scorer_loop_qat_auxiliary_warning",
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

    captured: dict[str, object] = {}

    def fake_run_qat(**kwargs):
        captured.update(kwargs)
        return FakeQatResult()

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)
    monkeypatch.setattr(qat_mod, "run_snerv_scorer_loop_decoder_qat_smoke", fake_run_qat)

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path,
        num_pairs=1,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "decoder_payload_codec": "int8_symmetric",
            "snerv_fc_dim": 5,
            "snerv_mfu_scales": (1, 2),
            "snerv_temporal_context": 1,
            "snerv_temporal_mode": "official_haar_dwt1d_lowpass",
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
        run_scorer_loop_qat=True,
        scorer_loop_qat_max_trials=1,
        scorer_loop_qat_search_mode="top_weight_coordinate",
        scorer_loop_qat_qat_bits=4,
        scorer_loop_qat_component_guard_mode="pose_seg_hard",
        scorer_loop_qat_perturb_scale=0.03125,
        scorer_loop_qat_byte_pressure_multiplier=2.0,
        scorer_loop_qat_section_value_pressure_multiplier=1.5,
        scorer_loop_qat_max_archive_byte_growth=9,
        scorer_loop_qat_byte_growth_admission_mode="rate_paid",
        scorer_loop_qat_pose_slack=0.004,
        scorer_loop_qat_seg_slack=0.005,
        scorer_loop_qat_seed=99,
    )

    assert captured["n_pairs"] == 1
    assert captured["max_trials"] == 1
    assert captured["qat_bits"] == 4
    assert captured["decoder_payload_codec"] == "int8_symmetric"
    assert captured["lf_payload_codec"] == "portfolio_auto"
    assert captured["component_guard_mode"] == "pose_seg_hard"
    assert captured["perturb_scale"] == 0.03125
    assert captured["byte_pressure_multiplier"] == 2.0
    assert captured["section_value_pressure_multiplier"] == 1.5
    assert captured["max_archive_byte_growth"] == 9
    assert captured["byte_growth_admission_mode"] == "rate_paid"
    assert captured["pose_slack"] == 0.004
    assert captured["seg_slack"] == 0.005
    assert captured["seed"] == 99
    assert captured["pair_guard_min_score_improved_fraction"] == 1.0
    assert captured["pair_guard_max_pose_worsened_fraction"] == 0.0
    assert captured["snerv_fc_dim"] == 5
    assert captured["snerv_mfu_scales"] == (1, 2)
    assert captured["snerv_temporal_context"] == 1
    assert captured["snerv_temporal_mode"] == "official_haar_dwt1d_lowpass"
    scorer_loop = report["scorer_loop_qat"]
    assert scorer_loop["requested"] is True
    assert scorer_loop["executed"] is True
    assert scorer_loop["component_guard_mode"] == "pose_seg_hard"
    assert scorer_loop["perturb_scale"] == pytest.approx(0.0)
    assert scorer_loop["byte_pressure_multiplier"] == pytest.approx(2.0)
    assert scorer_loop["section_value_pressure_multiplier"] == pytest.approx(0.0)
    assert scorer_loop["max_archive_byte_growth"] == 0
    assert scorer_loop["byte_growth_admission_mode"] == "rate_paid"
    assert scorer_loop["pose_slack"] == pytest.approx(0.0)
    assert scorer_loop["seg_slack"] == pytest.approx(0.0)
    assert scorer_loop["seed"] == 99
    assert scorer_loop["lf_payload_codec"] == best_packet_metadata[
        "lf_payload_codec_selected"
    ]
    assert scorer_loop["receiver_contract_satisfied"] is True
    assert scorer_loop["accepted_improvement"] is True
    assert scorer_loop["pair_robust_admission"]["passed"] is True
    assert scorer_loop["pair_robust_admission"]["permissive_guard"] is False
    assert scorer_loop["best_archive_sha256"] == best_packet_sha256
    assert scorer_loop["best_packet_sha256"] == best_packet_sha256
    assert scorer_loop["best_packet_materialized"] is True
    assert scorer_loop["best_packet_path_sha256"] == best_packet_sha256
    assert scorer_loop["best_packet_schema"] == "snerv_inverse_steg_archive.v1"
    assert scorer_loop["best_packet_wire_format"] == "snar1"
    assert scorer_loop["best_packet_contest_submission_wire_format_ready"] is False
    assert Path(scorer_loop["best_packet_path"]).read_bytes() == best_packet
    assert scorer_loop["emitted_packet_uses_scorer_loop_best_decoder"] is True
    assert scorer_loop["emitted_packet_sha256"] == report["packet_sha256"]
    assert scorer_loop["emitted_packet_sha256"] != best_packet_sha256
    assert scorer_loop["emitted_packet_schema"] == (
        "snerv_inverse_steg_archive.snar2.v1"
    )
    assert scorer_loop["emitted_packet_wire_format"] == "snar2"
    assert scorer_loop["emitted_packet_contest_submission_wire_format_ready"] is True
    continuity = scorer_loop["selected_packet_metadata_continuity"]
    assert continuity["metadata_only_repack"] is True
    assert continuity["container_repacked_to_submission_format"] is True
    assert continuity["output_packet_wire_format"] == "snar2"
    assert continuity["contest_submission_wire_format_ready"] is True
    assert continuity["input_selected_packet_sha256"] == best_packet_sha256
    assert continuity["output_selected_packet_sha256"] == report["packet_sha256"]
    assert {
        row["field"] for row in continuity["inherited_fields"]
    } >= {
        "step_map_packet_schema",
        "step_map_coder_mode",
        "step_map_waterfill_bits_per_coeff",
        "step_map_coder_groups",
    }
    assert scorer_loop["blockers"] == ["snerv_scorer_loop_qat_auxiliary_warning"]
    assert report["packet_source"] == "scorer_loop_qat_best_receiver_packet"
    assert report["packet_schema"] == "snerv_inverse_steg_archive.snar2.v1"
    assert report["packet_wire_format"] == "snar2"
    assert report["packet_contest_submission_wire_format_ready"] is True
    assert report["packet_sha256"] != best_packet_sha256
    assert report["byte_cap_control"]["packet_source"] == report["packet_source"]
    assert report["byte_cap_control"]["packet_sha256"] == report["packet_sha256"]
    assert report["byte_cap_control"]["decoder_payload_codec"] == (
        report["decoder_payload_codec"]
    )
    assert report["byte_cap_control"]["lf_payload_codec"] == report["lf_payload_codec"]
    emitted_packet = Path(report["packet_path"]).read_bytes()
    assert emitted_packet != best_packet
    emitted_archive = unpack_snerv_archive(emitted_packet)
    assert emitted_archive.schema == "snerv_inverse_steg_archive.snar2.v1"
    assert emitted_archive.sections == raw_best_archive.sections
    assert "step_map_coder_mode" not in emitted_archive.metadata
    assert "step_map_packet_schema" not in emitted_archive.metadata
    assert report["receiver_packet_report_metadata_source"] == (
        "snar2_compact_wire_metadata_plus_selected_packet_report_metadata"
    )
    assert report["step_map_coder_mode"] == (
        "waterfill_mlx_native_uniform_importance_bridge"
    )
    assert report["step_map_packet_schema"] == "snerv_step_map_coder.adaptive.v1"
    assert report["step_map_coder_groups"]
    assert "snerv_real_segnet_posenet_teacher_loop_not_attached" not in report["blockers"]
    assert "snerv_scorer_loop_qat_best_packet_not_materialized_into_native_export" not in report["blockers"]
    assert "snerv_scorer_loop_qat_not_full_video" in report["blockers"]
    assert report["score_claim"] is False


def test_official_train_export_rejects_generic_qat_best_packet(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod
    import tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat as qat_mod

    pairs = _tiny_pairs(pairs=1)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)
    generic_best_packet = build_snerv_mlx_native_packet_from_numpy_pairs(
        pairs + 1.0,
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="auto",
    ).packet
    generic_best_sha256 = hashlib.sha256(generic_best_packet).hexdigest()

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    class FakeQatResult:
        def __init__(self) -> None:
            self.best_packet = generic_best_packet

        def as_jsonable(self) -> dict:
            return {
                "schema": "snerv_scorer_loop_decoder_qat_smoke.v1",
                "axis_tag": "[macOS-CPU advisory]",
                "n_pairs": 1,
                "decoder_payload_codec": "int8_symmetric",
                "lf_payload_codec": "v2:signed_int2_bitpack:none",
                "lf_payload_codec_requested": "portfolio_auto",
                "lf_payload_codec_selected": "v2:signed_int2_bitpack:none",
                "scorer_loop_evaluations": 1,
                "accepted_improvement": True,
                "receiver_contract_satisfied": True,
                "ready_for_pose_guard_gate": True,
                "baseline": {
                    "archive_bytes": 111,
                    "archive_sha256": "1" * 64,
                    "score_linf": 3.0,
                },
                "best": {
                    "archive_bytes": len(generic_best_packet),
                    "archive_sha256": generic_best_sha256,
                    "score_linf": 2.5,
                },
                "best_packet_bytes": len(generic_best_packet),
                "best_packet_sha256": generic_best_sha256,
                "component_guard_mode": "pose_seg_hard",
                "pair_robust_admission": {
                    "schema": "snerv_pair_robust_admission.v1",
                    "n_pairs": 1,
                    "min_score_improved_fraction": 1.0,
                    "max_pose_worsened_fraction": 0.0,
                    "pose_slack": 0.0,
                    "score_improved_fraction": 1.0,
                    "pose_worsened_fraction": 0.0,
                    "permissive_guard": False,
                    "passed": True,
                    "blockers": [],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "blockers": [
                    "snerv_scorer_loop_qat_best_packet_not_materialized_into_native_export"
                ],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)
    monkeypatch.setattr(
        qat_mod,
        "run_snerv_scorer_loop_decoder_qat_smoke",
        lambda **_kwargs: FakeQatResult(),
    )

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "official_generic_qat_reject",
        num_pairs=1,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "candidate_id": "official-generic-qat-reject",
            "snerv_model_size_adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "decoder_payload_codec": "int8_symmetric",
            "snerv_fc_dim": 9,
            "snerv_official_skip_high_mode": "channel_mean",
            "snerv_temporal_context": 1,
            "snerv_temporal_mode": "official_haar_dwt1d_lowpass",
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
        run_scorer_loop_qat=True,
        scorer_loop_qat_max_trials=1,
        scorer_loop_qat_component_guard_mode="pose_seg_hard",
    )

    emitted_packet = Path(report["packet_path"]).read_bytes()
    decoded = unpack_snerv_archive(emitted_packet)
    official_payload = decode_official_mfu_hfr_tub_decoder_payload(
        decoded.sections["decoder_payload"]
    )
    scorer_loop = report["scorer_loop_qat"]

    assert emitted_packet != generic_best_packet
    assert report["packet_sha256"] != generic_best_sha256
    assert report["packet_source"] != "scorer_loop_qat_best_receiver_packet"
    assert scorer_loop["best_packet_materialized"] is True
    assert Path(scorer_loop["best_packet_path"]).read_bytes() == generic_best_packet
    assert scorer_loop["official_decoder_payload_binding_required"] is True
    assert scorer_loop["official_decoder_payload_binding_preserved"] is False
    assert scorer_loop["emitted_packet_uses_scorer_loop_best_decoder"] is False
    assert "snerv_scorer_loop_qat_best_packet_rejected_official_payload_mismatch" in scorer_loop["blockers"]
    assert "snerv_scorer_loop_qat_best_packet_rejected_official_payload_mismatch" in report["blockers"]
    assert report["snerv_official_mfu_hfr_tub_export_bound"] is True
    assert report["snerv_official_mfu_hfr_tub_frame_producing_export"] is True
    assert official_payload.header["schema"] == (
        "snerv_decoder_payload.official_mfu_hfr_tub.v1"
    )
    assert official_payload.score_claim is False
    assert report["score_claim"] is False


def test_official_train_export_rejects_qat_packet_with_tub_output2_binding_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod
    import tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat as qat_mod

    pairs = _tiny_pairs(pairs=1)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)
    model_size_elided = SnervModelSizeConfig(
        adapter=SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
        fc_dim=9,
        official_tub_output2_store_for_receiver_proof=False,
    )
    original_bootstrap = mod._official_mfu_hfr_tub_bootstrap_components_from_pairs

    def components_with_output2() -> dict[str, object]:
        components = original_bootstrap(pairs, model_size=model_size_elided)
        components["temporal_encoder_output_shape"] = (1, 6, 8, 8)
        components["output2_decoder_output_shape"] = (2, 12, 8, 8)
        components["tub_temporal_encoder_concat"] = np.arange(
            np.prod(components["temporal_encoder_output_shape"]),
            dtype=np.float64,
        ).reshape(components["temporal_encoder_output_shape"])
        components["tub_output2_raw"] = (
            np.arange(
                np.prod(components["output2_decoder_output_shape"]),
                dtype=np.float64,
            ).reshape(components["output2_decoder_output_shape"])
            / 31.0
        )
        return components

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    def fake_bootstrap(_pairs_arg, *, model_size):
        components = components_with_output2()
        return components

    best_packet = mod._build_official_mfu_hfr_tub_packet_from_components(
        components_with_output2(),
        source_pair_indices=[0],
        model_size=model_size_elided,
        metadata_extra={"allocation_mode": "unit_qat_output2_elided"},
    ).packet
    best_sha256 = hashlib.sha256(best_packet).hexdigest()

    class FakeQatResult:
        def __init__(self) -> None:
            self.best_packet = best_packet

        def as_jsonable(self) -> dict:
            return {
                "schema": "snerv_scorer_loop_decoder_qat_smoke.v1",
                "axis_tag": "[macOS-CPU advisory]",
                "n_pairs": 1,
                "source_pair_indices": [0],
                "decoder_payload_codec": "snerv_decoder_payload.official_mfu_hfr_tub.v1",
                "lf_payload_codec": "spatial_delta_zigzag_leb128_lzma",
                "lf_payload_codec_requested": "spatial_delta_zigzag_leb128_lzma",
                "lf_payload_codec_selected": "spatial_delta_zigzag_leb128_lzma",
                "scorer_loop_evaluations": 1,
                "accepted_improvement": True,
                "receiver_contract_satisfied": True,
                "ready_for_pose_guard_gate": True,
                "baseline": {
                    "archive_bytes": 111,
                    "archive_sha256": "1" * 64,
                    "score_linf": 3.0,
                },
                "best": {
                    "archive_bytes": len(best_packet),
                    "archive_sha256": best_sha256,
                    "score_linf": 2.5,
                },
                "best_packet_bytes": len(best_packet),
                "best_packet_sha256": best_sha256,
                "component_guard_mode": "pose_seg_hard",
                "pair_robust_admission": {
                    "schema": "snerv_pair_robust_admission.v1",
                    "n_pairs": 1,
                    "min_score_improved_fraction": 1.0,
                    "max_pose_worsened_fraction": 0.0,
                    "pose_slack": 0.0,
                    "score_improved_fraction": 1.0,
                    "pose_worsened_fraction": 0.0,
                    "permissive_guard": False,
                    "passed": True,
                    "blockers": [],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "blockers": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)
    monkeypatch.setattr(
        mod,
        "_official_mfu_hfr_tub_bootstrap_components_from_pairs",
        fake_bootstrap,
    )
    monkeypatch.setattr(
        qat_mod,
        "run_snerv_scorer_loop_decoder_qat_smoke",
        lambda **_kwargs: FakeQatResult(),
    )

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "official_qat_output2_mismatch",
        num_pairs=1,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "candidate_id": "official-qat-output2-mismatch",
            "snerv_model_size_adapter": SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "snerv_fc_dim": 9,
            "snerv_official_tub_output2_store_for_receiver_proof": True,
            "snerv_official_tub_output2_export_mode": "proof_only",
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
        run_scorer_loop_qat=True,
        scorer_loop_qat_max_trials=1,
        scorer_loop_qat_component_guard_mode="pose_seg_hard",
    )

    emitted_packet = Path(report["packet_path"]).read_bytes()
    emitted_payload = decode_official_mfu_hfr_tub_decoder_payload(
        unpack_snerv_archive(emitted_packet).sections["decoder_payload"]
    )
    scorer_loop = report["scorer_loop_qat"]
    binding = scorer_loop["official_tub_output2_binding_report"]

    assert emitted_packet != best_packet
    assert report["packet_sha256"] != best_sha256
    assert report["packet_source"] != "scorer_loop_qat_best_receiver_packet"
    assert emitted_payload.header["tub_output2_storage"]["stored"] is True
    assert scorer_loop["best_packet_materialized"] is True
    assert scorer_loop["official_decoder_payload_binding_preserved"] is True
    assert scorer_loop["official_tub_output2_binding_required"] is True
    assert scorer_loop["official_tub_output2_binding_preserved"] is False
    assert binding["preserved"] is False
    assert {
        "stored",
        "proof_only_elided_from_selected_runtime_packet",
        "stored_raw_bytes",
        "payload_tensor_names",
    }.issubset(set(binding["mismatched_fields"]))
    assert "snerv_scorer_loop_qat_best_packet_rejected_official_payload_mismatch" not in scorer_loop[
        "blockers"
    ]
    assert "snerv_scorer_loop_qat_best_packet_rejected_official_tub_output2_binding_mismatch" in scorer_loop[
        "blockers"
    ]
    assert "snerv_scorer_loop_qat_best_packet_rejected_official_tub_output2_binding_mismatch" in report[
        "blockers"
    ]
    assert scorer_loop["emitted_packet_uses_scorer_loop_best_decoder"] is False
    assert report["score_claim"] is False


def test_train_export_rejects_qat_packet_when_pose_guard_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod
    import tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat as qat_mod

    pairs = _tiny_pairs(pairs=2)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)
    best_packet = build_snerv_mlx_native_packet_from_numpy_pairs(
        pairs + 1.0,
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="auto",
    ).packet
    best_packet_sha256 = hashlib.sha256(best_packet).hexdigest()

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    class FakeQatResult:
        def __init__(self) -> None:
            self.best_packet = best_packet

        def as_jsonable(self) -> dict:
            return {
                "schema": "snerv_scorer_loop_decoder_qat_smoke.v1",
                "axis_tag": "[macOS-CPU advisory]",
                "n_pairs": 2,
                "decoder_payload_codec": "int8_symmetric",
                "lf_payload_codec": "v2:signed_int2_bitpack:none",
                "lf_payload_codec_requested": "portfolio_auto",
                "lf_payload_codec_selected": "v2:signed_int2_bitpack:none",
                "lf_payload_codec_selection_report": {
                    "schema": "snerv_lf_quant_payload.v2",
                    "mode_histogram": {"signed_int2_bitpack": 1},
                    "wrapper_histogram": {"none": 1},
                    "section_bytes": 42,
                },
                "scorer_loop_evaluations": 2,
                "accepted_improvement": True,
                "receiver_contract_satisfied": True,
                "ready_for_pose_guard_gate": False,
                "baseline": {
                    "archive_bytes": 111,
                    "archive_sha256": "1" * 64,
                    "score_linf": 3.0,
                },
                "best": {
                    "archive_bytes": len(best_packet),
                    "archive_sha256": best_packet_sha256,
                    "score_linf": 2.5,
                },
                "best_packet_bytes": len(best_packet),
                "best_packet_sha256": best_packet_sha256,
                "component_guard_mode": "pose_seg_hard",
                "pair_robust_admission": {
                    "schema": "snerv_pair_robust_admission.v1",
                    "n_pairs": 2,
                    "min_score_improved_fraction": 1.0,
                    "max_pose_worsened_fraction": 0.0,
                    "pose_slack": 0.0,
                    "score_improved_fraction": 0.5,
                    "pose_worsened_fraction": 0.5,
                    "permissive_guard": False,
                    "passed": False,
                    "blockers": [],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                "blockers": ["pair_robust_pose_guard_not_ready_unit"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)
    monkeypatch.setattr(
        qat_mod,
        "run_snerv_scorer_loop_decoder_qat_smoke",
        lambda **_kwargs: FakeQatResult(),
    )

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "pose_guard_reject",
        num_pairs=2,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "decoder_payload_codec": "int8_symmetric",
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
        run_scorer_loop_qat=True,
        scorer_loop_qat_max_trials=1,
        scorer_loop_qat_component_guard_mode="pose_seg_hard",
    )

    scorer_loop = report["scorer_loop_qat"]
    assert scorer_loop["accepted_improvement"] is True
    assert scorer_loop["receiver_contract_satisfied"] is True
    assert scorer_loop["ready_for_pose_guard_gate"] is False
    assert scorer_loop["pair_guard_min_score_improved_fraction"] == 1.0
    assert scorer_loop["pair_guard_max_pose_worsened_fraction"] == 0.0
    assert scorer_loop["pair_robust_admission"]["passed"] is False
    assert scorer_loop["best_packet_materialized"] is True
    assert scorer_loop["best_packet_path_sha256"] == best_packet_sha256
    assert scorer_loop["emitted_packet_uses_scorer_loop_best_decoder"] is False
    assert "snerv_scorer_loop_qat_best_packet_not_materialized_into_native_export" in report[
        "blockers"
    ]
    assert report["packet_source"] == "mlx_target_hydration_numpy_closed_form_decoder_fit"
    assert report["packet_sha256"] != best_packet_sha256
    assert Path(report["packet_path"]).read_bytes() != best_packet


def test_train_export_rejects_qat_packet_with_mismatched_source_pair_indices(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod
    import tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat as qat_mod

    pairs = _tiny_pairs(pairs=2)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)
    mismatched_packet = build_snerv_mlx_native_packet_from_numpy_pairs(
        pairs + 1.0,
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="auto",
        source_pair_indices=[0, 1],
    ).packet
    mismatched_sha256 = hashlib.sha256(mismatched_packet).hexdigest()

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    class FakeQatResult:
        best_packet = mismatched_packet

        def as_jsonable(self) -> dict:
            return {
                "schema": "snerv_scorer_loop_decoder_qat_smoke.v1",
                "axis_tag": "[macOS-CPU advisory]",
                "n_pairs": 2,
                "source_pair_indices": [7, 2],
                "decoder_payload_codec": "int8_symmetric",
                "lf_payload_codec": "v2:signed_int2_bitpack:none",
                "lf_payload_codec_requested": "portfolio_auto",
                "lf_payload_codec_selected": "v2:signed_int2_bitpack:none",
                "lf_payload_codec_selection_report": {
                    "schema": "snerv_lf_quant_payload.v2",
                    "mode_histogram": {"signed_int2_bitpack": 1},
                    "wrapper_histogram": {"none": 1},
                    "section_bytes": 42,
                },
                "scorer_loop_evaluations": 1,
                "accepted_improvement": True,
                "receiver_contract_satisfied": True,
                "ready_for_pose_guard_gate": True,
                "baseline": {
                    "archive_bytes": 111,
                    "archive_sha256": "1" * 64,
                    "score_linf": 3.0,
                },
                "best": {
                    "archive_bytes": len(mismatched_packet),
                    "archive_sha256": mismatched_sha256,
                    "score_linf": 2.5,
                },
                "best_packet_bytes": len(mismatched_packet),
                "best_packet_sha256": mismatched_sha256,
                "component_guard_mode": "score_primary",
                "blockers": [],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

    captured: dict[str, object] = {}

    def fake_run_qat(**kwargs):
        captured.update(kwargs)
        return FakeQatResult()

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)
    monkeypatch.setattr(qat_mod, "run_snerv_scorer_loop_decoder_qat_smoke", fake_run_qat)

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path,
        num_pairs=600,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "decoder_payload_codec": "int8_symmetric",
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        pair_indices=[7, 2],
        run_archive_export=False,
        run_scorer_loop_qat=True,
        scorer_loop_qat_max_trials=1,
    )

    assert captured["n_pairs"] == 2
    assert captured["pair_indices"] == (7, 2)
    scorer_loop = report["scorer_loop_qat"]
    assert scorer_loop["accepted_improvement"] is True
    assert scorer_loop["source_pair_indices_binding_required"] is True
    assert scorer_loop["source_pair_indices_binding_preserved"] is False
    assert scorer_loop["source_pair_indices_expected"] == [7, 2]
    assert scorer_loop["source_pair_indices_actual"] == [0, 1]
    assert "snerv_scorer_loop_qat_best_packet_rejected_source_pair_indices_mismatch" in scorer_loop["blockers"]
    assert "snerv_scorer_loop_qat_best_packet_rejected_source_pair_indices_mismatch" in report["blockers"]
    assert report["packet_source"] == "mlx_target_hydration_numpy_closed_form_decoder_fit"
    assert report["packet_sha256"] != mismatched_sha256
    decoded = unpack_snerv_archive(Path(report["packet_path"]).read_bytes())
    assert decoded.metadata["source_pair_indices"] == [7, 2]


def test_train_export_rejects_unweighted_qat_packet_when_recon_weight_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = pytest.importorskip("mlx.core")
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod
    import tac.substrates.snerv_inverse_steg_carrier.scorer_loop_decoder_qat as qat_mod

    pairs = _tiny_pairs(pairs=1)
    pairs[0, 1, 0, 3:12, 4:13] += 24.0
    pairs = np.clip(pairs, 0.0, 255.0)
    target0 = mx.array(np.transpose(pairs[:, 0], (0, 2, 3, 1)) / 255.0)
    target1 = mx.array(np.transpose(pairs[:, 1], (0, 2, 3, 1)) / 255.0)
    weight = np.ones((1, 2, 16, 16, 1), dtype=np.float32)
    weight[:, :, 3:12, 4:13, :] = 16.0
    weight_path = tmp_path / "joint_p18_p19_weight.npz"
    np.savez(weight_path, weight=weight)
    unweighted_best_packet = build_snerv_mlx_native_packet_from_numpy_pairs(
        pairs + 1.0,
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="auto",
    ).packet
    unweighted_best_sha256 = hashlib.sha256(unweighted_best_packet).hexdigest()

    def fake_decode_mlx_targets(*_args, **_kwargs):
        return target0, target1

    class FakeQatResult:
        best_packet = unweighted_best_packet

        def as_jsonable(self) -> dict:
            return {
                "schema": "snerv_scorer_loop_decoder_qat_smoke.v1",
                "axis_tag": "[macOS-CPU advisory]",
                "n_pairs": 1,
                "decoder_payload_codec": "int8_symmetric",
                "lf_payload_codec": "v2:signed_int2_bitpack:none",
                "lf_payload_codec_requested": "portfolio_auto",
                "lf_payload_codec_selected": "v2:signed_int2_bitpack:none",
                "lf_payload_codec_selection_report": {
                    "schema": "snerv_lf_quant_payload.v2",
                    "mode_histogram": {"signed_int2_bitpack": 1},
                    "wrapper_histogram": {"none": 1},
                    "section_bytes": 42,
                },
                "scorer_loop_evaluations": 2,
                "accepted_improvement": True,
                "receiver_contract_satisfied": True,
                "ready_for_pose_guard_gate": True,
                "baseline": {
                    "archive_bytes": 111,
                    "archive_sha256": "1" * 64,
                    "score_linf": 3.0,
                },
                "best": {
                    "archive_bytes": len(unweighted_best_packet),
                    "archive_sha256": unweighted_best_sha256,
                    "score_linf": 2.5,
                },
                "best_packet_bytes": len(unweighted_best_packet),
                "best_packet_sha256": unweighted_best_sha256,
                "component_guard_mode": "pose_seg_hard",
                "blockers": ["snerv_scorer_loop_qat_best_packet_not_materialized_into_native_export"],
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }

    monkeypatch.setattr(mod, "decode_mlx_targets", fake_decode_mlx_targets)
    monkeypatch.setattr(
        qat_mod,
        "run_snerv_scorer_loop_decoder_qat_smoke",
        lambda **_kwargs: FakeQatResult(),
    )

    report = train_export_snerv_mlx_native(
        output_dir=tmp_path / "export_qat_reject",
        num_pairs=1,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "decoder_payload_codec": "int8_symmetric",
            "hf_decoder_saliency_gain": 2.5,
        },
        scorer_upstream_dir="upstream",
        output_height=16,
        output_width=16,
        run_archive_export=False,
        run_scorer_loop_qat=True,
        scorer_loop_qat_max_trials=1,
        scorer_loop_qat_component_guard_mode="pose_seg_hard",
        recon_pixel_weight_path=weight_path,
        recon_pixel_weight_normalize="mean",
    )

    scorer_loop = report["scorer_loop_qat"]
    assert scorer_loop["accepted_improvement"] is True
    assert scorer_loop["receiver_contract_satisfied"] is True
    assert scorer_loop["best_packet_materialized"] is True
    assert scorer_loop["best_packet_path_sha256"] == unweighted_best_sha256
    assert Path(scorer_loop["best_packet_path"]).read_bytes() == unweighted_best_packet
    assert scorer_loop["emitted_packet_uses_scorer_loop_best_decoder"] is False
    assert scorer_loop["recon_weight_binding_required"] is True
    assert scorer_loop["recon_weight_binding_preserved"] is False
    assert "snerv_scorer_loop_qat_best_packet_rejected_recon_weight_binding_mismatch" in scorer_loop["blockers"]
    assert "snerv_scorer_loop_qat_best_packet_rejected_recon_weight_binding_mismatch" in report["blockers"]
    assert report["packet_source"] == (
        "mlx_target_hydration_numpy_joint_p18_p19_dwt_adjoint_saliency_weighted_decoder_fit"
    )
    assert report["packet_sha256"] != unweighted_best_sha256
    assert Path(report["packet_path"]).read_bytes() != unweighted_best_packet
    decoded = unpack_snerv_archive(Path(report["packet_path"]).read_bytes())
    assert decoded.metadata["recon_pixel_weight_consumed"] is True


def test_receiver_decoded_mlx_prefilter_uses_selected_packet_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.local_acceleration.mlx_renderer_prefilter_profile as prefilter_mod
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    decoded_pairs = np.arange(2 * 2 * 3 * 4 * 5, dtype=np.float32).reshape(
        2, 2, 3, 4, 5
    )
    target0 = np.linspace(0.0, 1.0, num=2 * 4 * 5 * 3, dtype=np.float32).reshape(
        2, 4, 5, 3
    )
    target1 = target0 + np.float32(0.25)
    captured: dict[str, object] = {}

    mlx_pkg = types.ModuleType("mlx")
    mx_mod = types.ModuleType("mlx.core")
    mx_mod.float32 = np.float32
    mx_mod.array = lambda value, dtype=None: np.asarray(value, dtype=dtype)
    mx_mod.take = lambda value, idx, axis=0: np.take(
        value,
        np.asarray(idx, dtype=np.int64),
        axis=axis,
    )
    mlx_pkg.core = mx_mod
    monkeypatch.setitem(sys.modules, "mlx", mlx_pkg)
    monkeypatch.setitem(sys.modules, "mlx.core", mx_mod)
    def fake_decode_snerv_archive_frames(packet: bytes) -> np.ndarray:
        captured["packet"] = packet
        return decoded_pairs

    monkeypatch.setattr(
        mod,
        "decode_snerv_archive_frames",
        fake_decode_snerv_archive_frames,
    )

    def fake_write_profile(**kwargs):
        bundle = kwargs["bundle"]
        captured["bundle_metadata"] = dict(bundle.substrate_artifact_metadata)
        captured["archive_bytes"] = int(kwargs["archive_bytes"])
        captured["archive_sha256"] = str(kwargs["archive_sha256"])
        captured["scorer_device"] = str(kwargs["scorer_device"])
        captured["scorer_batch_pairs"] = int(kwargs["scorer_batch_pairs"])
        captured["progress_every"] = int(kwargs["progress_every"])
        np.testing.assert_array_equal(bundle.model(np.asarray([1])), decoded_pairs[[1]])
        np.testing.assert_array_equal(np.asarray(bundle.target_rgb_0), target0)
        np.testing.assert_array_equal(np.asarray(bundle.target_rgb_1), target1)
        output_path = Path(kwargs["output_path"])
        output_path.write_text(
            json.dumps(
                {
                    "schema": "hprc_mlx_component_profile.v1",
                    "blockers": [],
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        progress_path = Path(kwargs["progress_jsonl_path"])
        progress_path.write_text(
            json.dumps({"schema": "mlx_renderer_prefilter_progress.v1"}) + "\n",
            encoding="utf-8",
        )
        return {
            "schema": "hprc_mlx_component_profile.v1",
            "blockers": [],
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(
        prefilter_mod,
        "write_mlx_renderer_prefilter_profile",
        fake_write_profile,
    )

    profile = mod._write_snerv_native_receiver_decoded_mlx_prefilter(
        requested=True,
        output_dir=tmp_path / "prefilter",
        selected_packet=b"SNAR1 receiver packet",
        target0_np=target0,
        target1_np=target1,
        archive_bytes=1234,
        archive_sha256="d" * 64,
        source_video_path=tmp_path / "source.mkv",
        scorer_upstream_dir=tmp_path / "upstream",
        scorer_device="gpu",
        scorer_batch_pairs=4,
        progress_every=7,
        allow_overwrite=False,
    )

    assert captured["packet"] == b"SNAR1 receiver packet"
    assert captured["archive_bytes"] == 1234
    assert captured["archive_sha256"] == "d" * 64
    assert captured["scorer_device"] == "gpu"
    assert captured["scorer_batch_pairs"] == 4
    assert captured["progress_every"] == 7
    assert captured["bundle_metadata"]["receiver_decoded_selected_packet"] is True
    assert captured["bundle_metadata"]["contest_scorer_prefilter_only"] is True
    assert profile["written"] is True
    assert profile["blockers"] == []
    assert profile["score_claim"] is False
    assert Path(profile["profile_path"]).is_file()
    assert Path(profile["progress_path"]).is_file()


def test_prefilter_profile_is_false_authority_until_component_scores_exist(
    tmp_path: Path,
) -> None:
    profile = write_snerv_mlx_prefilter_profile(
        artifact={
            "schema": "snerv_mlx_native_train_export.v1",
            "report_path": "/tmp/report.json",
            "packet_path": "/tmp/packet.snar",
            "num_pairs": 2,
        },
        archive_bytes=123,
        archive_sha256="a" * 64,
        output_path=tmp_path / "profile.json",
        upstream_dir="upstream",
    )

    assert profile["prefilter_ready_for_cpu_replay"] is False
    assert "snerv_mlx_prefilter_component_scorers_not_attached" in profile["blockers"]
    assert "snerv_mlx_prefilter_not_full_video" in profile["blockers"]
    assert profile["score_claim"] is False


def test_prefilter_profile_rejects_blocked_full_video_artifact(
    tmp_path: Path,
) -> None:
    profile = write_snerv_mlx_prefilter_profile(
        artifact={
            "schema": "snerv_mlx_native_train_export.v1",
            "report_path": "/tmp/report.json",
            "packet_path": "/tmp/packet.snar",
            "num_pairs": 600,
            "archive_path": "/tmp/archive.zip",
            "archive_bytes": 456,
            "archive_sha256": "b" * 64,
            "bridge_drift": {"allclose": True},
            "receiver_proof_passed": True,
            "receiver_contract_satisfied": True,
            "blockers": ["snerv_mlx_score_aware_long_training_not_executed"],
        },
        archive_bytes=456,
        archive_sha256="b" * 64,
        output_path=tmp_path / "profile_blocked.json",
        upstream_dir="upstream",
        component_profile={"segnet_delta": 0.0, "posenet_delta": 0.0},
    )

    assert profile["prefilter_ready_for_cpu_replay"] is False
    assert "snerv_mlx_prefilter_artifact_has_blockers" in profile["blockers"]
    assert profile["artifact_blockers"] == ["snerv_mlx_score_aware_long_training_not_executed"]


def test_prefilter_profile_accepts_receiver_proven_full_video_artifact(
    tmp_path: Path,
) -> None:
    profile = write_snerv_mlx_prefilter_profile(
        artifact={
            "schema": "snerv_mlx_native_train_export.v1",
            "report_path": "/tmp/report.json",
            "packet_path": "/tmp/packet.snar",
            "num_pairs": 600,
            "archive_path": "/tmp/archive.zip",
            "archive_bytes": 456,
            "archive_sha256": "c" * 64,
            "bridge_drift": {"allclose": True},
            "receiver_proof_passed": True,
            "receiver_contract_satisfied": True,
            "blockers": [],
        },
        archive_bytes=456,
        archive_sha256="c" * 64,
        output_path=tmp_path / "profile_ready.json",
        upstream_dir="upstream",
        component_profile={"segnet_delta": -0.001, "posenet_delta": 0.0},
    )

    assert profile["prefilter_ready_for_cpu_replay"] is True
    assert profile["blockers"] == []
    assert profile["score_claim"] is False


def test_official_receiver_tensor_map_accepts_nbytes_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    def fake_unpack(packet: bytes):
        assert packet == b"packet"
        return type("Decoded", (), {"sections": {"decoder_payload": b"decoder"}})()

    def fake_header(payload: bytes) -> dict[str, object]:
        assert payload == b"decoder"
        header: dict[str, object] = {
            "schema": mod.DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA,
            "codec": "lzma_raw_tensor_payload",
            "mfu_spec": {"num_blocks": 0},
            "tub_output2_storage": {"stored": True},
        }
        required = mod._official_receiver_required_tensor_keys_from_header(
            header,
            present_tensor_names=set(),
        )

        def row(name: str, *, shape: tuple[int, ...] = (1,), dialect: str = "nbytes"):
            nbytes = int(np.prod(shape)) * np.dtype("<f8").itemsize
            out = {
                "name": name,
                "shape": list(shape),
                "dtype": "float64_le",
                "sha256": hashlib.sha256(name.encode("utf-8")).hexdigest(),
            }
            if dialect in {"bytes", "bytes+nbytes"}:
                out["bytes"] = nbytes
            if dialect in {"nbytes", "bytes+nbytes"}:
                out["nbytes"] = nbytes
            return out

        rows = []
        for name in sorted(required):
            shape = (2, 2) if name == "mfu.upsample_mid.weight" else (1,)
            dialect = (
                "bytes+nbytes"
                if name == "hfr.lh.conv1.bias"
                else "nbytes"
            )
            rows.append(row(name, shape=shape, dialect=dialect))
        rows.append(row("tub.temporal_encoder.weight", shape=(5,), dialect="nbytes"))
        header["tensor_manifest"] = rows
        return header

    monkeypatch.setattr(mod, "unpack_snerv_archive", fake_unpack)
    monkeypatch.setattr(mod, "inspect_decoder_payload_header", fake_header)

    tensor_map = mod._official_receiver_tensor_map_from_packet(b"packet")

    assert tensor_map["receiver_tensor_map_verified"] is True
    assert tensor_map["blockers"] == []
    assert tensor_map["missing_required_tensor_keys"] == []
    assert tensor_map["required_tensor_key_count"] > 20
    assert tensor_map["total_tensor_bytes"] == sum(
        row["bytes"] for row in tensor_map["rows"]
    )
    assert tensor_map["category_bytes"]["official_mfu_weight_payload"] > 0
    assert tensor_map["category_bytes"]["official_hfr_weight_payload"] > 0
    assert tensor_map["category_bytes"]["official_tub_input_payload"] > 0
    assert tensor_map["category_bytes"]["official_tub_output2_payload"] == 16
    assert tensor_map["category_bytes"]["official_tub_weight_payload"] == 40
    rows = {row["name"]: row for row in tensor_map["rows"]}
    assert rows["mfu.upsample_mid.weight"]["manifest_byte_key"] == "nbytes"
    assert rows["hfr.lh.conv1.bias"]["manifest_byte_key"] == "bytes+nbytes"
    assert rows["tub.output2_raw"]["category"] == "official_tub_output2_payload"
    assert (
        rows["tub.temporal_encoder.weight"]["category"]
        == "official_tub_weight_payload"
    )


def test_official_receiver_tensor_map_blocks_partial_required_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    monkeypatch.setattr(
        mod,
        "unpack_snerv_archive",
        lambda _packet: type(
            "Decoded", (), {"sections": {"decoder_payload": b"decoder"}}
        )(),
    )
    monkeypatch.setattr(
        mod,
        "inspect_decoder_payload_header",
        lambda _payload: {
            "schema": mod.DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA,
            "codec": "lzma_raw_tensor_payload",
            "mfu_spec": {"num_blocks": 0},
            "tensor_manifest": [
                {
                    "name": "mfu.upsample_mid.weight",
                    "shape": [1],
                    "dtype": "float64_le",
                    "bytes": 8,
                    "sha256": "a" * 64,
                }
            ],
        },
    )

    tensor_map = mod._official_receiver_tensor_map_from_packet(b"packet")

    assert tensor_map["receiver_tensor_map_verified"] is False
    assert tensor_map["official_decoder_payload_selected"] is True
    assert tensor_map["blockers"] == [
        "snerv_official_receiver_tensor_map_missing_required_tensors"
    ]
    assert "mfu.upsample_mid.bias" in tensor_map["missing_required_tensor_keys"]


def test_official_receiver_tensor_map_blocks_mismatched_byte_dialects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    monkeypatch.setattr(
        mod,
        "unpack_snerv_archive",
        lambda _packet: type(
            "Decoded", (), {"sections": {"decoder_payload": b"decoder"}}
        )(),
    )
    monkeypatch.setattr(
        mod,
        "inspect_decoder_payload_header",
        lambda _payload: {
            "schema": mod.DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA,
            "tensor_manifest": [
                {
                    "name": "mfu.blocks.0.weight",
                    "shape": [2, 2],
                    "dtype": "float64_le",
                    "bytes": 32,
                    "nbytes": 24,
                    "sha256": "a" * 64,
                }
            ],
        },
    )

    tensor_map = mod._official_receiver_tensor_map_from_packet(b"packet")

    assert tensor_map["receiver_tensor_map_verified"] is False
    assert tensor_map["official_decoder_payload_selected"] is True
    assert tensor_map["blockers"] == [
        "snerv_official_receiver_tensor_map_invalid_tensor_bytes"
    ]
    assert "mismatched bytes and nbytes" in tensor_map["error"]
