# SPDX-License-Identifier: MIT
"""NO-FAKE tests for the native MLX SNeRV export adapter."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier.archive import (
    decode_snerv_archive_frames,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.mlx_native_train_export import (
    SNERV_MLX_NATIVE_REPORT_FILENAME,
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
    assert decoded.metadata["step_map_packet_schema"] == (
        "snerv_step_map_coder.adaptive.v1"
    )
    assert decoded.metadata["step_map_coder_mode"] == (
        "waterfill_mlx_native_uniform_importance_bridge"
    )
    assert decoded.metadata["step_map_waterfill_bits_per_coeff"] == pytest.approx(0.5)
    assert decoded.metadata["step_map_coder_groups"]
    assert decoded.metadata["lf_payload_codec"] == "auto"
    assert frames.shape == (2, 2, 3, 16, 16)
    assert np.isfinite(frames).all()
    assert packet.score_claim is False


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
    assert decoded.metadata["recon_pixel_weight_consumed"] is True
    assert decoded.metadata["hf_decoder_fit_mode"] == (
        "joint_p18_p19_recon_pixel_weighted_least_squares"
    )
    assert decoded.metadata["hf_decoder_saliency_gain"] == pytest.approx(3.0)
    assert decoded.metadata["recon_pixel_weight_metadata"]["schema"] == (
        "unit_joint_weight.v1"
    )
    assert weighted.packet != unweighted.packet
    assert not np.allclose(
        decode_snerv_archive_frames(weighted.packet),
        decode_snerv_archive_frames(unweighted.packet),
    )


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
        row["relative_path"] == "evaluate.py" and row["sha256"]
        for row in report["scorer_custody"]["source_custody"]
    )
    assert report["archive_package"] is None
    assert report["archive_path"] is None
    assert report["step_map_bits_per_coeff"] == pytest.approx(0.5)
    assert report["step_map_packet_schema"] == "snerv_step_map_coder.adaptive.v1"
    assert report["step_map_coder_mode"] == (
        "waterfill_mlx_native_uniform_importance_bridge"
    )
    assert report["step_map_coder_groups"]
    assert report["lf_payload_codec"] == "portfolio_auto"
    assert report["receiver_proof_passed"] is False
    assert "snerv_mlx_score_aware_long_training_not_executed" in report["blockers"]
    assert "snerv_real_segnet_posenet_teacher_loop_not_attached" in report["blockers"]
    assert report["scorer_loop_qat"]["requested"] is False
    frames = decode_snerv_archive_frames(packet_path.read_bytes())
    assert frames.shape == (1, 2, 3, 16, 16)


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
        "mlx_target_hydration_numpy_joint_p18_p19_weighted_decoder_fit"
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
    assert decoded.metadata["hf_decoder_fit_mode"] == (
        "joint_p18_p19_recon_pixel_weighted_least_squares"
    )
    assert decoded.metadata["hf_decoder_saliency_gain"] == pytest.approx(2.5)
    assert decoded.metadata["recon_pixel_weight_consumed"] is True
    assert decoded.metadata["recon_pixel_weight_metadata"]["sha256"] == recon["sha256"]


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
    scorer_loop = report["scorer_loop_qat"]
    assert scorer_loop["requested"] is True
    assert scorer_loop["executed"] is True
    assert scorer_loop["component_guard_mode"] == "pose_seg_hard"
    assert scorer_loop["lf_payload_codec"] == "portfolio_auto"
    assert scorer_loop["receiver_contract_satisfied"] is True
    assert scorer_loop["accepted_improvement"] is True
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
    assert "snerv_real_segnet_posenet_teacher_loop_not_attached" not in report[
        "blockers"
    ]
    assert "snerv_scorer_loop_qat_best_packet_not_materialized_into_native_export" not in report[
        "blockers"
    ]
    assert "snerv_scorer_loop_qat_not_full_video" in report["blockers"]
    assert report["score_claim"] is False


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
    assert profile["artifact_blockers"] == [
        "snerv_mlx_score_aware_long_training_not_executed"
    ]


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
