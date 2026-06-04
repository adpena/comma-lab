# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the native MLX SNeRV export adapter."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier.archive import (
    decode_official_mfu_hfr_tub_decoder_payload,
    decode_snerv_archive_frames,
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
from tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export import (
    SNERV_DWT_ADJOINT_SALIENCY_WEIGHTED_FIT_MODE,
    SNERV_MLX_NATIVE_REPORT_FILENAME,
    _build_snerv_mlx_native_byte_cap_control,
    _model_size_from_candidate,
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


def test_pr95_muon_policy_is_bound_to_native_train_export_surfaces() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    public_sig = inspect.signature(train_export_snerv_mlx_native)
    assert "score_aware_long_training_pr95_muon_policy" in public_sig.parameters
    assert (
        "score_aware_long_training_scorer_input_distribution_guard_weight"
        in public_sig.parameters
    )
    assert (
        public_sig.parameters[
            "score_aware_long_training_pr95_muon_policy"
        ].default
        == "faithful_stage8_only"
    )
    attachment_sig = inspect.signature(mod._run_score_aware_long_training_attachment)
    assert "pr95_muon_policy" in attachment_sig.parameters
    assert "scorer_input_distribution_guard_weight" in attachment_sig.parameters

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


def test_score_aware_checkpoint_selection_policy_fails_closed_on_missing_inputs() -> None:
    import tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export as mod

    policy = mod._snerv_score_aware_checkpoint_selection_policy(
        segnet_distillation_weight=0.25,
        pose_distillation_weight=0.1,
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
    assert "pose_distill" in policy["required_loss_parts"]
    assert "snerv_score_aware_checkpoint_selection_segnet_teacher_missing" in policy[
        "blockers"
    ]
    assert "snerv_score_aware_checkpoint_selection_posenet_teacher_missing" in policy[
        "blockers"
    ]
    assert "snerv_score_aware_checkpoint_selection_coder_qat_terms_missing" in policy[
        "blockers"
    ]


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
        output_dir=tmp_path / "zero_muon_every_stage",
        num_pairs=1,
        source_video_path="unit.mkv",
        modelsize_candidate={
            "levels": 1,
            "wavelet": "haar",
            "bits_per_coeff": 3.0,
            "step_map_bits_per_coeff": 0.5,
            "decoder_payload_codec": "int8_symmetric",
            "score_aware_long_training_epochs": 1,
            "score_aware_long_training_pr95_faithful_curriculum": True,
            "score_aware_long_training_pr95_muon_policy": "every_stage",
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
    assert decoded.metadata["lf_payload_codec"] == "auto"
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

    assert decoded.metadata["lf_payload_codec"] == "portfolio_auto"
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
    assert report["lf_payload_codec"] == "portfolio_auto"
    assert report["receiver_proof_passed"] is False
    assert "snerv_mlx_score_aware_long_training_not_executed" in report["blockers"]
    assert "snerv_real_segnet_posenet_teacher_loop_not_attached" in report["blockers"]
    assert report["scorer_loop_qat"]["requested"] is False
    frames = decode_snerv_archive_frames(packet_path.read_bytes())
    assert frames.shape == (1, 2, 3, 16, 16)


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

    pairs = _tiny_pairs(pairs=2)
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
    assert long_training["checkpoint_selection_policy"]["mse_fallback"] is True
    assert (
        long_training["checkpoint_selection_policy"]["mse_fallback_reason"]
        == "no_scorer_teacher_no_coder_qat_no_pr95_curriculum"
    )
    assert long_training["best_checkpoint_selection"]["selection_metric"] == (
        "full_reconstruction_mse_nchw255"
    )
    assert long_training["checkpoint_retention"]["keep_last_n"] == (
        mod.SNERV_SCORE_AWARE_CHECKPOINT_RETENTION_KEEP_LAST_N_DEFAULT
    )
    assert long_training["selection_history_tail"]
    assert Path(long_training["report_path"]).is_file()
    assert Path(long_training["training_artifact"]["telemetry_path"]).is_file()
    packet = Path(report["packet_path"]).read_bytes()
    decoded = unpack_snerv_archive(packet)
    assert decoded.metadata["score_aware_long_training_executed"] is True
    assert decoded.metadata["score_aware_long_training"]["executed"] is True
    frames = decode_snerv_archive_frames(packet)
    assert frames.shape == (2, 2, 3, 16, 16)
    assert np.isfinite(frames).all()


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

    class FakeSegTeacher:
        num_classes = 5

        def teacher_logits_for_indices(self, indices):
            captured["seg_indices_shape"] = tuple(indices.shape)
            return mx.zeros((int(indices.shape[0]), 16, 16, 5), dtype=mx.float32)

    class FakePoseTeacher:
        pose_dims = 6
        per_dim_scale = mx.ones((6,), dtype=mx.float32)

        def teacher_pose_for_indices(self, indices):
            captured["pose_indices_shape"] = tuple(indices.shape)
            return mx.zeros((int(indices.shape[0]), 6), dtype=mx.float32)

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
            "score_aware_long_training_epochs": 1,
            "score_aware_long_training_lr": 1.0e-3,
            "score_aware_long_training_batch_pairs": 2,
            "score_aware_long_training_optimizer": "pact_muon_adamw",
            "score_aware_long_training_scorer_input_distribution_guard_weight": 0.5,
            "score_aware_long_training_scorer_input_distribution_guard_saturation_margin": 0.03,
            "score_aware_long_training_scorer_input_distribution_guard_temperature": 0.02,
        },
        scorer_upstream_dir=fake_upstream,
        output_height=16,
        output_width=16,
        run_archive_export=False,
        segnet_distillation_weight=0.01,
        pose_distillation_weight=0.001,
        pose_distillation_loss="huber",
        pose_distillation_huber_delta=2.0,
        segnet_distillation_objective="kl_t2",
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
    assert report["score_aware_long_training_executed"] is True
    assert report["score_aware_long_training_real_teachers_bound"] is True
    assert report["score_aware_long_training_has_real_segnet_teacher"] is True
    assert report["score_aware_long_training_has_real_posenet_teacher"] is True
    assert report["score_aware_long_training_coder_qat_bound"] is True
    assert report["score_aware_long_training_pr95_curriculum_bound"] is True
    assert report["score_aware_long_training_muon_adamw_partition_bound"] is True
    assert "snerv_real_segnet_posenet_teacher_loop_not_attached" not in report["blockers"]
    long_training = report["score_aware_long_training"]
    assert long_training["has_real_segnet_teacher"] is True
    assert long_training["has_real_posenet_teacher"] is True
    assert long_training["teacher_binding"]["requested_distillation_device"] == "gpu"
    assert long_training["teacher_binding"]["distillation_device"] == "mps"
    assert long_training["teacher_binding"]["distillation_device_resolution"] == {
        "schema": "snerv_native_torch_scorer_device_resolution.v1",
        "requested": "gpu",
        "resolved": "mps",
        "scope": "real_pytorch_segnet_posenet_teacher_cache",
    }
    assert long_training["coder_aware_qat"]["enabled"] is True
    assert long_training["coder_aware_qat"]["quant_bits"] == 4
    assert long_training["archive_section_qat_weight_policy_bound"] is True
    section_policy = long_training["archive_section_qat_weight_policy"]
    assert section_policy["schema"] == mod.SNERV_ARCHIVE_SECTION_QAT_WEIGHT_POLICY_SCHEMA
    assert section_policy["active"] is True
    assert section_policy["baseline_packet_bytes"] > 0
    assert section_policy["decoder_section_bytes"] > 0
    assert section_policy["lf_section_bytes"] > 0
    assert "latent_qat_quant_residual" in section_policy["extra_loss_weights"]
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
    assert long_training["pr95_faithful_curriculum_enabled"] is True
    assert long_training["muon_adamw_partition_bound"] is True
    assert long_training["teacher_binding"]["pose_distillation_loss"] == "huber"
    assert long_training["teacher_binding"]["pose_distillation_huber_delta"] == 2.0
    assert long_training["teacher_binding"]["learnable_student_head_bound"] is True
    assert long_training["teacher_binding"]["learnable_pose_student_head_bound"] is True
    assert long_training["checkpoint_selection_policy"]["mse_fallback"] is False
    assert (
        long_training["checkpoint_selection_policy"]["selection_metric"]
        == "score_aware_composite_full_video_surrogate"
    )
    assert "real_segnet_teacher_distillation" in long_training[
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
    assert "scorer_input_distribution_guard" in long_training[
        "checkpoint_selection_policy"
    ]["required_loss_parts"]
    assert (
        long_training["checkpoint_selection_policy"][
            "scorer_input_distribution_guard_weight"
        ]
        == 0.5
    )
    assert long_training["best_checkpoint_selection"]["selection_metric"] == (
        "score_aware_composite_full_video_surrogate"
    )
    assert np.isfinite(
        long_training["best_checkpoint_selection"]["score_aware_composite_loss"]
    )
    assert "weighted_distill" in long_training["best_checkpoint_selection"][
        "score_aware_composite_parts"
    ]
    assert "weighted_pose_distill" in long_training["best_checkpoint_selection"][
        "score_aware_composite_parts"
    ]
    assert (
        "weighted_scorer_input_distribution_guard"
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
    assert decoded.metadata["official_skip_high_mode"] == "shared_mean"
    assert decoded.metadata["official_skip_high_full_shape"] == [4, 3, 8, 8]
    assert "snerv_official_bootstrap_stores_haar_ll_as_mfu_skip_high" in decoded.metadata[
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


def test_official_mfu_hfr_tub_packet_carries_output2_payload_from_components() -> None:
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
        metadata_extra={"allocation_mode": "unit_output2_payload_bound"},
    )
    decoded = unpack_snerv_archive(packet.packet)
    official_payload = decode_official_mfu_hfr_tub_decoder_payload(
        decoded.sections["decoder_payload"]
    )
    proof = official_payload.execute()

    storage = decoded.metadata["official_tub_output2_storage"]
    assert storage["stored"] is True
    assert storage["receiver_executes_output2_fusion_from_payload"] is True
    assert storage["tensor_names"] == [
        "tub.temporal_encoder_concat",
        "tub.output2_raw",
    ]
    assert decoded.metadata["official_tub_output2_receiver_executed"] is True
    assert proof["executed_components"]["official_tub_output2_fusion"] is True
    rows = {row["name"]: row for row in proof["output_tensors"]}
    assert rows["tub.output2_decoder_input"]["shape"] == [2, 2, 4, 4]
    assert rows["tub.output2_fused"]["shape"] == [2, 2, 8, 8]
    assert decoded.metadata["source_faithful_stack"] is False
    assert decoded.metadata["score_claim"] is False


def test_official_renderer_exports_output2_payload_into_receiver_packet() -> None:
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
    )

    exported = model.export_official_components()
    metadata = model.metadata()
    packet = mod._build_official_mfu_hfr_tub_packet_from_components(
        exported,
        source_pair_indices=[8, 9],
        model_size=model_size,
        metadata_extra={"allocation_mode": "unit_renderer_output2_payload_bound"},
    )
    decoded = unpack_snerv_archive(packet.packet)
    official_payload = decode_official_mfu_hfr_tub_decoder_payload(
        decoded.sections["decoder_payload"]
    )
    proof = official_payload.execute()

    assert "tub_temporal_encoder_concat" in exported
    assert "tub_output2_raw" in exported
    assert metadata["official_tub_output2_payload_export_bound"] is True
    assert metadata["official_tub_output2_payload_loss_coupled"] is False
    assert decoded.metadata["official_tub_output2_storage"]["stored"] is True
    assert decoded.metadata["official_tub_output2_receiver_executed"] is True
    assert proof["executed_components"]["official_tub_output2_fusion"] is True


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
    assert (
        "snerv_official_mfu_hfr_tub_weight_mapping_missing"
        in train_export["authority_blockers"]
    )
    assert "snerv_official_trained_checkpoint_state_dict_mapping_missing" in train_export[
        "authority_blockers"
    ]
    assert len(train_export["trained_packet_sha256"]) == 64
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
    assert decoded.metadata["snerv_official_mfu_hfr_tub_export_bound"] is True
    assert decoded.metadata["snerv_official_mfu_hfr_tub_receiver_payload_bound"] is True
    assert decoded.metadata["snerv_official_mfu_hfr_tub_source_forward_replay_bound"] is False
    assert decoded.metadata["score_aware_long_training_executed"] is True
    assert decoded.metadata["score_aware_long_training"]["executed"] is True
    assert decoded.metadata["score_aware_long_training"]["official_mfu_hfr_tub_train_export"][
        "trained_receiver_payload_exported"
    ] is True
    frames = decode_snerv_archive_frames(Path(report["packet_path"]).read_bytes())
    assert frames.shape == (2, 2, 3, 16, 16)
    assert np.isfinite(frames).all()
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


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
    replay = report["score_aware_long_training"][
        "official_mfu_hfr_tub_source_forward_replay"
    ]
    assert replay["receiver_official_payload_forward_replay_passed"] is False
    assert np.isfinite(float(replay["max_abs_error_nchw255"]))
    assert "snerv_official_mfu_hfr_tub_receiver_payload_replay_failed" in replay[
        "blockers"
    ]
    decoded = unpack_snerv_archive(Path(report["packet_path"]).read_bytes())
    official_payload = decoded.decode_official_mfu_hfr_tub_payload()
    storage = official_payload.header["skip_high_storage"]
    assert storage["codec"] == codec
    assert storage["source_shape"] == [4, 3, 8, 8]
    assert storage["stored_shape"] == stored_shape
    assert storage["encoder_consumed_compact_train_state"] is True
    assert decoded.metadata["official_skip_high_export_storage_shape"] == stored_shape
    assert decoded.metadata["official_skip_high_export_is_compact_train_state"] is True
    assert official_payload.tensors["inputs.mfu.skip_high"].shape == (4, 3, 8, 8)
    frames = decode_snerv_archive_frames(Path(report["packet_path"]).read_bytes())
    assert frames.shape == (2, 2, 3, 16, 16)


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
        packet_bytes=300,
        section_bytes={
            "metadata_payload": 12,
            "lf_payload": 190,
            "decoder_payload": 64,
            "step_map_packet": 34,
        },
        archive_bytes=360,
        archive_sha256="a" * 64,
        run_archive_export=True,
    )

    assert cap["attached"] is True
    assert cap["enforced"] is True
    assert cap["under_hard_byte_ceiling"] is True
    assert cap["delta_bytes_vs_hard_byte_ceiling"] == -152
    assert cap["lf_payload_bytes"] == 190
    assert cap["largest_section_name"] == "lf_payload"
    assert cap["lf_payload_is_largest_section"] is True
    assert cap["lf_payload_exceeds_hard_byte_ceiling"] is False
    assert cap["lf_payload_can_cover_archive_overrun"] is None
    assert cap["blockers"] == []


def test_byte_cap_control_exposes_official_component_pressure_rows() -> None:
    cap = _build_snerv_mlx_native_byte_cap_control(
        candidate={"candidate_id": "official-pressure"},
        hard_byte_ceiling=700,
        packet_bytes=800,
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
        run_archive_export=True,
    )

    assert cap["official_decoder_payload_component_pressure_bound"] is True
    assert cap["official_decoder_payload_component_bytes"] == {
        "official_hfr_weight_payload": 40,
        "official_mfu_weight_payload": 160,
        "official_tub_output2_payload": 200,
    }
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


def test_native_export_modelsize_candidate_consumes_official_fc_dim_solution() -> None:
    model_size = _model_size_from_candidate(
        {
            "candidate_id": "official-modelsize-only",
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

    best_packet = build_snerv_mlx_native_packet_from_numpy_pairs(
        pairs + 1.0,
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="auto",
    ).packet
    best_packet_sha256 = hashlib.sha256(best_packet).hexdigest()

    class FakeQatResult:
        def __init__(self) -> None:
            self.best_packet = best_packet

        def as_jsonable(self) -> dict:
            return {
                "schema": "snerv_scorer_loop_decoder_qat_smoke.v1",
                "axis_tag": "[macOS-CPU advisory]",
                "n_pairs": 1,
                "decoder_payload_codec": "int8_symmetric",
                "lf_payload_codec": "portfolio_auto",
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
    )

    assert captured["n_pairs"] == 1
    assert captured["max_trials"] == 1
    assert captured["qat_bits"] == 4
    assert captured["decoder_payload_codec"] == "int8_symmetric"
    assert captured["lf_payload_codec"] == "portfolio_auto"
    assert captured["component_guard_mode"] == "pose_seg_hard"
    assert captured["snerv_fc_dim"] == 5
    assert captured["snerv_mfu_scales"] == (1, 2)
    assert captured["snerv_temporal_context"] == 1
    assert captured["snerv_temporal_mode"] == "official_haar_dwt1d_lowpass"
    scorer_loop = report["scorer_loop_qat"]
    assert scorer_loop["requested"] is True
    assert scorer_loop["executed"] is True
    assert scorer_loop["component_guard_mode"] == "pose_seg_hard"
    assert scorer_loop["lf_payload_codec"] == "portfolio_auto"
    assert scorer_loop["receiver_contract_satisfied"] is True
    assert scorer_loop["accepted_improvement"] is True
    assert scorer_loop["pair_robust_admission"]["passed"] is True
    assert scorer_loop["pair_robust_admission"]["permissive_guard"] is False
    assert scorer_loop["best_archive_sha256"] == best_packet_sha256
    assert scorer_loop["best_packet_sha256"] == best_packet_sha256
    assert scorer_loop["best_packet_materialized"] is True
    assert scorer_loop["best_packet_path_sha256"] == best_packet_sha256
    assert Path(scorer_loop["best_packet_path"]).read_bytes() == best_packet
    assert scorer_loop["emitted_packet_uses_scorer_loop_best_decoder"] is True
    assert scorer_loop["emitted_packet_sha256"] == best_packet_sha256
    assert scorer_loop["blockers"] == ["snerv_scorer_loop_qat_auxiliary_warning"]
    assert report["packet_source"] == "scorer_loop_qat_best_receiver_packet"
    assert report["packet_sha256"] == best_packet_sha256
    assert Path(report["packet_path"]).read_bytes() == best_packet
    assert "snerv_real_segnet_posenet_teacher_loop_not_attached" not in report["blockers"]
    assert "snerv_scorer_loop_qat_best_packet_not_materialized_into_native_export" not in report["blockers"]
    assert "snerv_scorer_loop_qat_not_full_video" in report["blockers"]
    assert report["score_claim"] is False


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
                "lf_payload_codec": "portfolio_auto",
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
                "lf_payload_codec": "portfolio_auto",
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
        return {
            "schema": mod.DECODER_PAYLOAD_OFFICIAL_MFU_HFR_TUB_SCHEMA,
            "codec": "lzma_raw_tensor_payload",
            "tensor_manifest": [
                {
                    "name": "mfu.blocks.0.weight",
                    "shape": [2, 2],
                    "dtype": "float64_le",
                    "nbytes": 32,
                    "sha256": "a" * 64,
                },
                {
                    "name": "hfr.heads.0.bias",
                    "shape": [4],
                    "dtype": "float64_le",
                    "bytes": 32,
                    "nbytes": 32,
                    "sha256": "b" * 64,
                },
                {
                    "name": "inputs.tub.current",
                    "shape": [2],
                    "dtype": "float64_le",
                    "bytes": 16,
                    "sha256": "c" * 64,
                },
                {
                    "name": "tub.output2_raw",
                    "shape": [8],
                    "dtype": "float64_le",
                    "bytes": 64,
                    "sha256": "d" * 64,
                },
                {
                    "name": "tub.temporal_encoder.weight",
                    "shape": [5],
                    "dtype": "float64_le",
                    "nbytes": 40,
                    "sha256": "e" * 64,
                },
            ],
        }

    monkeypatch.setattr(mod, "unpack_snerv_archive", fake_unpack)
    monkeypatch.setattr(mod, "inspect_decoder_payload_header", fake_header)

    tensor_map = mod._official_receiver_tensor_map_from_packet(b"packet")

    assert tensor_map["receiver_tensor_map_verified"] is True
    assert tensor_map["blockers"] == []
    assert tensor_map["total_tensor_bytes"] == 184
    assert tensor_map["category_bytes"]["official_mfu_weight_payload"] == 32
    assert tensor_map["category_bytes"]["official_hfr_weight_payload"] == 32
    assert tensor_map["category_bytes"]["official_tub_input_payload"] == 16
    assert tensor_map["category_bytes"]["official_tub_output2_payload"] == 64
    assert tensor_map["category_bytes"]["official_tub_weight_payload"] == 40
    rows = {row["name"]: row for row in tensor_map["rows"]}
    assert rows["mfu.blocks.0.weight"]["manifest_byte_key"] == "nbytes"
    assert rows["hfr.heads.0.bias"]["manifest_byte_key"] == "bytes+nbytes"
    assert rows["tub.output2_raw"]["category"] == "official_tub_output2_payload"
    assert (
        rows["tub.temporal_encoder.weight"]["category"]
        == "official_tub_weight_payload"
    )


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
