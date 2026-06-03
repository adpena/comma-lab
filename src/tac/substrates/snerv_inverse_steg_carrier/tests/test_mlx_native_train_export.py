# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the native MLX SNeRV export adapter."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier.archive import (
    decode_snerv_archive_frames,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SNERV_OFFICIAL_MFU_HFR_TUB_PRIMITIVES_ADAPTER,
)
from tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export import (
    SNERV_DWT_ADJOINT_SALIENCY_WEIGHTED_FIT_MODE,
    SNERV_MLX_NATIVE_REPORT_FILENAME,
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
    assert training["optimizer"] == "full_batch_gradient_descent"
    assert training["steps"] == 2
    assert training["learning_rate"] == pytest.approx(1.0e-5)
    assert training["all_final_losses_finite"] is True
    assert training["accepted"] is True
    assert training["any_loss_worsened"] is False
    assert training["level_subband_rows"]
    assert decoded.metadata["native_mlx_training_executed"] is True
    assert decoded.metadata["native_mlx_training_kind"] == ("hf_decoder_full_batch_gradient_descent")
    assert decoded.metadata["hf_decoder_fit_mode"].startswith("native_mlx_full_batch_gradient_descent_from_")
    assert training["score_claim"] is False
    assert training["ready_for_exact_eval_dispatch"] is False
    assert packet.score_claim is False


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
    assert trained["native_mlx_training_kind"] == "hf_decoder_full_batch_gradient_descent"
    training = trained["native_mlx_hf_decoder_training"]
    assert training["schema"] == "snerv_native_mlx_hf_decoder_training.v1"
    assert training["executed"] is True
    assert training["steps"] == 2
    assert training["level_subband_rows"]
    assert training["all_final_losses_finite"] is True
    assert training["accepted"] is True
    assert training["any_loss_worsened"] is False
    assert trained["packet_source"].startswith("native_mlx_full_batch_gradient_descent")
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
    assert report["snerv_official_mfu_hfr_tub_export_bound"] is False
    assert report["snerv_official_mfu_hfr_tub_receiver_bound_surrogate_export"] is True
    assert {
        "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload",
        "snerv_official_mfu_hfr_tub_weight_mapping_missing",
        "snerv_official_mfu_hfr_tub_source_forward_replay_missing",
    }.issubset(set(report["blockers"]))
    assert Path(report["packet_path"]).is_file()
    assert report["packet_bytes"] == Path(report["packet_path"]).stat().st_size
    assert report["receiver_proof_passed"] is False
    binding = report["official_primitive_binding"]
    assert binding["schema"] == "snerv_official_mfu_hfr_tub_export_binding.v3"
    assert binding["primitive_modules_available"] is True
    assert binding["export_bound_to_receiver_packet"] is True
    assert binding["surrogate_receiver_payload_contract_emitted"] is True
    assert binding["official_receiver_payload_contract_available"] is True
    assert binding["official_receiver_payload_contract_emitted"] is False
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
    assert authority["status"] == "surrogate_linear_decoder_frame_producing"
    assert authority["linear_surrogate_decoder_selected"] is True
    assert authority["official_decoder_payload_selected"] is False
    assert authority["frame_decode_succeeded"] is True
    assert authority["frame_producing_official_export"] is False
    assert authority["blockers"] == [
        "snerv_selected_packet_uses_linear_surrogate_decoder_payload"
    ]
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
    assert evidence[
        "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload"
    ]["missing_artifact"].startswith(
        "native MLX train/export selected packet"
    )
    assert evidence[
        "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload"
    ]["selected_packet_status"] == "surrogate_linear_decoder_frame_producing"
    assert evidence[
        "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload"
    ]["selected_packet_frame_producing_official_export"] is False
    assert evidence["snerv_official_mfu_hfr_tub_weight_mapping_missing"]["closure_test"].startswith(
        "map official encoder"
    )
    assert evidence["snerv_official_mfu_hfr_tub_source_forward_replay_missing"]["official_authority"] is False
    assert binding["export_consumed_official_mfu"] is False
    assert binding["export_consumed_official_hfr"] is False
    assert binding["export_consumed_official_tub"] is False
    assert binding["source_forward_replay_authority"] is False
    assert binding["receiver_runtime_decode_authority"] is True
    assert binding["selected_packet_official_payload_runtime_decode_authority"] is False
    assert binding["selected_packet_frame_producing_official_export"] is False
    decoded = unpack_snerv_archive(Path(report["packet_path"]).read_bytes())
    assert decoded.metadata["snerv_official_mfu_hfr_tub_receiver_bound_surrogate_export"] is True
    assert decoded.metadata["snerv_official_mfu_hfr_tub_export_bound"] is False
    assert decoded.metadata["source_faithful_stack"] is False
    assert decode_snerv_archive_frames(Path(report["packet_path"]).read_bytes()).shape == (
        2,
        2,
        3,
        16,
        16,
    )
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert Path(report["report_path"]).is_file()


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
    assert (
        "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload"
        in report["blockers"]
    )
    binding = report["official_primitive_binding"]
    surrogate = binding["receiver_bound_surrogate_export"]
    assert surrogate["archive_sha256"] == report["archive_sha256"]
    assert surrogate["surrogate_receiver_contract_satisfied"] is True
    assert surrogate["surrogate_runtime_consumption_proof_passed"] is True
    assert binding["receiver_runtime_decode_authority"] is True
    assert binding["selected_packet_official_payload_runtime_decode_authority"] is False
    assert binding["selected_packet_frame_producing_official_export"] is False
    assert binding["selected_packet_authority"]["status"] == (
        "surrogate_linear_decoder_frame_producing"
    )
    receiver_decode_row = {row["blocker"]: row for row in binding["blocker_evidence"]}[
        "snerv_official_mfu_hfr_tub_native_mlx_export_not_bound_to_official_payload"
    ]
    assert receiver_decode_row["closed"] is False
    assert receiver_decode_row["selected_packet_official_decoder_payload_selected"] is False
    assert receiver_decode_row["surrogate_receiver_runtime_decode_passed"] is True
    assert receiver_decode_row["official_authority"] is False
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


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
    assert model_size.feature_count == 11


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
