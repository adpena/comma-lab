# SPDX-License-Identifier: MIT
"""Tests for direct SNeRV checkpoint archive export."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from tac.substrates._shared.numpy_portable_inflate import pack_state_dict_numpy
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    decode_snerv_archive_frames,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import SnervModelSizeConfig
from tools import export_snerv_checkpoint_archive as export_tool
from tools.export_snerv_checkpoint_archive import (
    build_snerv_checkpoint_packet,
    export_snerv_checkpoint_archive,
)


def test_snerv_checkpoint_packet_uses_state_lf_and_decoder_directly() -> None:
    model_size = SnervModelSizeConfig(fc_dim=9, emb_size=0, temporal_context=0)
    lf = np.zeros((2, 2, 3, 8, 8), dtype=np.float32)
    for pair_idx in range(2):
        for frame_idx in range(2):
            for channel_idx in range(3):
                lf[pair_idx, frame_idx, channel_idx] = (
                    32.0
                    + 3.0 * pair_idx
                    + 5.0 * frame_idx
                    + 7.0 * channel_idx
                )
    state: dict[str, np.ndarray] = {"latents_lf_planes": lf}
    for subband in ("LH", "HL", "HH"):
        state[f"decoder_kernels.0.{subband}"] = np.zeros(
            (model_size.feature_count,),
            dtype=np.float32,
        )

    packet = build_snerv_checkpoint_packet(
        state,
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=3.0,
        step_map_bits_per_coeff=0.5,
        decoder_payload_codec="int8_symmetric",
        lf_payload_codec="portfolio_auto",
        model_size=model_size,
        metadata_extra={"unit_test_marker": "direct_checkpoint_packet"},
    )
    decoded = unpack_snerv_archive(packet.packet)
    frames = decode_snerv_archive_frames(packet.packet)

    assert decoded.metadata.get("checkpoint_export_schema") is None
    assert decoded.metadata["unit_test_marker"] == "direct_checkpoint_packet"
    assert decoded.metadata["hf_decoder_fit_mode"] == "trained_mlx_checkpoint_decoder_kernels"
    assert decoded.metadata["native_mlx_training_executed"] is True
    assert decoded.metadata["score_claim"] is False
    lf_report = packet.section_reports["lf_payload_codec_report"]
    assert lf_report["report_status"] == "receiver_visible_lf_payload_accounting_verified"
    assert lf_report["schema"] == "snerv_lf_quant_payload.v2"
    assert lf_report["section_bytes"] == packet.section_bytes["lf_payload"]
    assert lf_report["mode_histogram"]
    assert lf_report["score_claim"] is False
    assert packet.total_bytes == len(packet.packet)
    assert frames.shape == (2, 2, 3, 16, 16)
    assert np.isfinite(frames).all()


def test_snerv_checkpoint_export_can_write_receiver_decoded_mlx_prefilter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_size = SnervModelSizeConfig(fc_dim=9, emb_size=0, temporal_context=0)
    state: dict[str, np.ndarray] = {
        "latents_lf_planes": np.zeros((2, 2, 3, 8, 8), dtype=np.float32),
    }
    for subband in ("LH", "HL", "HH"):
        state[f"decoder_kernels.0.{subband}"] = np.zeros(
            (model_size.feature_count,),
            dtype=np.float32,
        )
    state_path = tmp_path / "state.npsd"
    state_path.write_bytes(pack_state_dict_numpy(state, dtype="fp32"))
    checkpoint_meta = tmp_path / "checkpoint.meta.json"
    checkpoint_meta.write_text(
        json.dumps(
            {
                "global_epoch": 17,
                "ema_shadow_state_path": state_path.as_posix(),
                "live_state_path": state_path.as_posix(),
            }
        ),
        encoding="utf-8",
    )
    startup = tmp_path / "startup.json"
    source_video = tmp_path / "source.mkv"
    startup.write_text(
        json.dumps(
            {
                "schema": "compact_carrier_startup_marker.v1",
                "source_video_path": source_video.as_posix(),
                "modelsize_candidate": {
                    "candidate_id": "snerv_test_prefilter",
                    "fc_dim": 9,
                    "levels": 1,
                    "wavelet": "haar",
                    "bits_per_coeff": 3.0,
                    "step_map_bits_per_coeff": 0.5,
                    "decoder_payload_codec": "int8_symmetric",
                    "lf_payload_codec": "portfolio_auto",
                    "nominal_total_payload_bytes": 6,
                    "hard_byte_ceiling": 10,
                },
                "hard_byte_ceilings": [12],
                "command_args": {
                    "num_pairs": 2,
                    "scorer_upstream_dir": (tmp_path / "upstream").as_posix(),
                },
            }
        ),
        encoding="utf-8",
    )

    def fake_export_snerv_mlx_archive(*_args, **_kwargs):
        archive = tmp_path / "archive.zip"
        archive.write_bytes(b"archive")
        return {
            "receiver_proof": {
                "archive_path": archive.as_posix(),
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": "a" * 64,
                "proof_path": (tmp_path / "receiver_proof.json").as_posix(),
                "runtime_consumption_proof_passed": True,
                "receiver_contract_satisfied": True,
            }
        }

    calls: dict[str, object] = {}

    def fake_decode_mlx_targets(
        source_video_path,
        *,
        num_pairs,
        output_height,
        output_width,
        pair_indices,
    ):
        calls["source_video_path"] = Path(source_video_path).as_posix()
        calls["num_pairs"] = int(num_pairs)
        calls["output_hw"] = (int(output_height), int(output_width))
        calls["pair_indices"] = tuple(int(value) for value in pair_indices)
        target0 = np.zeros((num_pairs, output_height, output_width, 3), dtype=np.float32)
        target1 = np.ones((num_pairs, output_height, output_width, 3), dtype=np.float32)
        return target0, target1

    def fake_prefilter(**kwargs):
        calls["prefilter_archive_bytes"] = int(kwargs["archive_bytes"])
        calls["prefilter_archive_sha256"] = str(kwargs["archive_sha256"])
        calls["prefilter_scorer_device"] = str(kwargs["scorer_device"])
        calls["prefilter_target0_shape"] = tuple(kwargs["target0_np"].shape)
        calls["prefilter_packet_bytes"] = len(kwargs["selected_packet"])
        out = Path(kwargs["output_dir"])
        return {
            "schema": "snerv_mlx_native_prefilter_profile.v1",
            "requested": True,
            "written": True,
            "profile_path": (out / "local_mlx_prefilter_profile.json").as_posix(),
            "progress_path": (out / "local_mlx_prefilter_progress.jsonl").as_posix(),
            "blockers": ["macos_mlx_research_signal_false_authority"],
        }

    monkeypatch.setattr(export_tool, "export_snerv_mlx_archive", fake_export_snerv_mlx_archive)
    monkeypatch.setattr(export_tool, "decode_mlx_targets", fake_decode_mlx_targets)
    monkeypatch.setattr(
        export_tool,
        "_write_snerv_native_receiver_decoded_mlx_prefilter",
        fake_prefilter,
    )

    report = export_snerv_checkpoint_archive(
        startup_json=startup,
        checkpoint_meta=checkpoint_meta,
        output_dir=tmp_path / "export",
        state_kind="ema",
        emit_receiver_proof=True,
        write_mlx_prefilter_profile=True,
        mlx_prefilter_scorer_device="mps",
        mlx_prefilter_scorer_batch_pairs=4,
        repo_root=tmp_path,
    )

    assert report["receiver_proof_passed"] is True
    assert report["hard_byte_ceiling_requested_by_candidate_or_startup"] == 10
    assert report["hard_byte_ceiling_checked_after_export"] is True
    feedback = report["modelsize_byte_cap_feedback_row"]
    assert feedback["schema"] == "nerv_modelsize_byte_cap_feedback_row.v1"
    assert feedback["family"] == "snerv"
    assert feedback["candidate_id"] == "snerv_test_prefilter"
    assert feedback["hard_byte_ceiling"] == 10
    assert feedback["nominal_total_payload_bytes"] == 6
    assert feedback["measured_archive_bytes"] == len(b"archive")
    assert feedback["calibrated_archive_overrun_bytes"] == 0
    assert feedback["receiver_closed"] is True
    assert report["local_mlx_prefilter_written"] is True
    assert report["local_mlx_prefilter_profile"]["scorer_batch_pairs_requested"] == 4
    assert report["local_mlx_prefilter_profile"]["scorer_batch_pairs_effective"] == 1
    assert (
        report["local_mlx_prefilter_profile"][
            "scorer_batch_pairs_normalized_to_singleton"
        ]
        is True
    )
    lf_summary = report["packet_section_report_summary"]["lf_payload_codec_report"]
    assert lf_summary["report_status"] == "receiver_visible_lf_payload_accounting_verified"
    assert lf_summary["schema"] == "snerv_lf_quant_payload.v2"
    assert lf_summary["section_bytes"] == report["packet_section_bytes"]["lf_payload"]
    assert lf_summary["mode_histogram"]
    lf_report = report["packet_section_reports"]["lf_payload_codec_report"]
    assert lf_report["section_sha256"] == report["packet_section_sha256"]["lf_payload"]
    assert lf_report["promotion_eligible"] is False
    assert "full_video_scorer_replay_not_executed" not in report["blockers"]
    assert "contest_cpu_cuda_exact_eval_not_executed" in report["blockers"]
    assert "macos_mlx_research_signal_false_authority" in report["blockers"]
    assert calls["source_video_path"] == source_video.as_posix()
    assert calls["num_pairs"] == 2
    assert calls["pair_indices"] == (0, 1)
    assert calls["output_hw"] == (16, 16)
    assert calls["prefilter_target0_shape"] == (2, 16, 16, 3)
    assert calls["prefilter_archive_bytes"] == len(b"archive")
    assert calls["prefilter_archive_sha256"] == "a" * 64
    assert calls["prefilter_scorer_device"] == "gpu"
    assert int(calls["prefilter_packet_bytes"]) == report["packet_bytes"]


def test_snerv_checkpoint_export_records_over_cap_measurement_feedback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_size = SnervModelSizeConfig(fc_dim=9, emb_size=0, temporal_context=0)
    state: dict[str, np.ndarray] = {
        "latents_lf_planes": np.zeros((1, 2, 3, 4, 4), dtype=np.float32),
    }
    for subband in ("LH", "HL", "HH"):
        state[f"decoder_kernels.0.{subband}"] = np.zeros(
            (model_size.feature_count,),
            dtype=np.float32,
        )
    state_path = tmp_path / "state.npsd"
    state_path.write_bytes(pack_state_dict_numpy(state, dtype="fp32"))
    checkpoint_meta = tmp_path / "checkpoint.meta.json"
    checkpoint_meta.write_text(
        json.dumps(
            {
                "global_epoch": 19,
                "ema_shadow_state_path": state_path.as_posix(),
                "live_state_path": state_path.as_posix(),
            }
        ),
        encoding="utf-8",
    )
    startup = tmp_path / "startup.json"
    startup.write_text(
        json.dumps(
            {
                "schema": "compact_carrier_startup_marker.v1",
                "modelsize_candidate": {
                    "candidate_id": "snerv_overcap_measurement",
                    "fc_dim": 9,
                    "levels": 1,
                    "wavelet": "haar",
                    "bits_per_coeff": 3.0,
                    "step_map_bits_per_coeff": 0.5,
                    "decoder_payload_codec": "int8_symmetric",
                    "lf_payload_codec": "portfolio_auto",
                    "nominal_total_payload_bytes": 120_000,
                    "hard_byte_ceiling": 178_000,
                },
                "hard_byte_ceilings": [216_000],
                "command_args": {"num_pairs": 1},
            }
        ),
        encoding="utf-8",
    )

    def fake_export_snerv_mlx_archive(*_args, **_kwargs):
        archive = tmp_path / "archive.zip"
        archive.write_bytes(b"x" * 214_187)
        return {
            "receiver_proof": {
                "archive_path": archive.as_posix(),
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": "d" * 64,
                "proof_path": (tmp_path / "receiver_proof.json").as_posix(),
                "runtime_consumption_proof_passed": True,
                "receiver_contract_satisfied": True,
            }
        }

    monkeypatch.setattr(export_tool, "export_snerv_mlx_archive", fake_export_snerv_mlx_archive)

    report = export_snerv_checkpoint_archive(
        startup_json=startup,
        checkpoint_meta=checkpoint_meta,
        output_dir=tmp_path / "export",
        state_kind="ema",
        emit_receiver_proof=True,
        allow_over_hard_byte_ceiling_for_measurement=True,
        repo_root=tmp_path,
    )

    assert report["archive_bytes"] == 214_187
    assert report["hard_byte_ceiling_requested_by_candidate_or_startup"] == 178_000
    assert report["hard_byte_ceiling_checked_after_export"] is True
    assert report["hard_byte_ceiling_measurement_bypass_enabled"] is True
    assert "archive_bytes_exceed_tightest_hard_ceiling" in report["blockers"]
    assert "hard_byte_ceiling_export_bypassed_for_measurement" in report["blockers"]
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    feedback = report["modelsize_byte_cap_feedback_row"]
    assert feedback["hard_byte_ceiling"] == 178_000
    assert feedback["hard_byte_ceiling_measurement_bypass_enabled"] is True
    assert feedback["measured_archive_bytes"] == 214_187
    assert feedback["calibrated_archive_overrun_bytes"] == 36_187
    assert feedback["required_nominal_payload_bytes_max"] == 99_725
    assert feedback["receiver_closed"] is True


def test_snerv_checkpoint_export_prefers_receiver_raw_cache_prefilter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_size = SnervModelSizeConfig(fc_dim=9, emb_size=0, temporal_context=0)
    state: dict[str, np.ndarray] = {
        "latents_lf_planes": np.zeros((2, 2, 3, 8, 8), dtype=np.float32),
    }
    for subband in ("LH", "HL", "HH"):
        state[f"decoder_kernels.0.{subband}"] = np.zeros(
            (model_size.feature_count,),
            dtype=np.float32,
        )
    state_path = tmp_path / "state.npsd"
    state_path.write_bytes(pack_state_dict_numpy(state, dtype="fp32"))
    checkpoint_meta = tmp_path / "checkpoint.meta.json"
    checkpoint_meta.write_text(
        json.dumps(
            {
                "global_epoch": 23,
                "ema_shadow_state_path": state_path.as_posix(),
                "live_state_path": state_path.as_posix(),
            }
        ),
        encoding="utf-8",
    )
    source_video = tmp_path / "source.mkv"
    source_video.write_bytes(b"video")
    startup = tmp_path / "startup.json"
    startup.write_text(
        json.dumps(
            {
                "schema": "compact_carrier_startup_marker.v1",
                "source_video_path": source_video.as_posix(),
                "modelsize_candidate": {
                    "candidate_id": "snerv_test_cache_prefilter",
                    "fc_dim": 9,
                    "levels": 1,
                    "wavelet": "haar",
                    "bits_per_coeff": 3.0,
                    "step_map_bits_per_coeff": 0.5,
                    "decoder_payload_codec": "int8_symmetric",
                    "lf_payload_codec": "portfolio_auto",
                },
                "command_args": {"num_pairs": 2},
            }
        ),
        encoding="utf-8",
    )
    raw = tmp_path / "receiver.raw"
    raw.write_bytes(b"raw")
    proof_path = tmp_path / "receiver_proof.json"
    proof_path.write_text("{}\n", encoding="utf-8")

    def fake_export_snerv_mlx_archive(*_args, **_kwargs):
        archive = tmp_path / "archive.zip"
        archive.write_bytes(b"archive")
        return {
            "receiver_proof": {
                "archive_path": archive.as_posix(),
                "archive_bytes": archive.stat().st_size,
                "archive_sha256": "b" * 64,
                "proof_path": proof_path.as_posix(),
                "receiver_output_path": raw.as_posix(),
                "receiver_output_sha256": "c" * 64,
                "runtime_consumption_proof_passed": True,
                "receiver_contract_satisfied": True,
            }
        }

    calls: dict[str, object] = {}

    def write_fake_cache(output_dir: Path, *, source: str) -> dict[str, object]:
        output_dir.mkdir(parents=True, exist_ok=True)
        artifacts: dict[str, dict[str, object]] = {}
        for name in ("segnet_last_rgb", "posenet_yuv6_pair", "pair_indices"):
            path = output_dir / f"{name}.npy"
            path.write_bytes(f"{source}:{name}".encode())
            artifacts[name] = {
                "path": path.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        manifest = {
            "schema_version": "mlx_scorer_input_cache.v1",
            "pair_count": 2,
            "artifacts": artifacts,
        }
        (output_dir / "manifest.json").write_text(
            json.dumps(manifest, sort_keys=True),
            encoding="utf-8",
        )
        return manifest

    def fake_video_cache(video_path, output_dir, **kwargs):
        calls["video_cache"] = (Path(video_path).as_posix(), Path(output_dir).name, kwargs)
        return write_fake_cache(Path(output_dir), source="reference")

    def fake_raw_cache(raw_path, output_dir, **kwargs):
        calls["raw_cache"] = (Path(raw_path).as_posix(), Path(output_dir).name, kwargs)
        return write_fake_cache(Path(output_dir), source="candidate")

    def fake_response(**kwargs):
        calls["response"] = kwargs
        return {
            "schema": "mlx_scorer_response.v1",
            "schema_version": "mlx_scorer_response.v1",
            "response_family": "snerv",
            "archive_sha256": "b" * 64,
            "archive_size_bytes": len(b"archive"),
            "n_samples": 2,
            "max_pairs": 2,
            "avg_segnet_dist": 0.25,
            "avg_posenet_dist": 0.5,
            "score_recomputed_from_components": 27.0,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }

    monkeypatch.setattr(export_tool, "export_snerv_mlx_archive", fake_export_snerv_mlx_archive)
    monkeypatch.setattr(export_tool, "write_scorer_input_cache_from_video_file", fake_video_cache)
    monkeypatch.setattr(export_tool, "write_scorer_input_cache_from_raw_file", fake_raw_cache)
    monkeypatch.setattr(export_tool, "build_mlx_scorer_response_payload", fake_response)
    monkeypatch.setattr(
        export_tool,
        "_write_snerv_native_receiver_decoded_mlx_prefilter",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("bulk path used")),
    )

    report = export_snerv_checkpoint_archive(
        startup_json=startup,
        checkpoint_meta=checkpoint_meta,
        output_dir=tmp_path / "export",
        state_kind="ema",
        emit_receiver_proof=True,
        write_mlx_prefilter_profile=True,
        mlx_prefilter_scorer_device="metal",
        repo_root=tmp_path,
    )

    assert report["local_mlx_prefilter_written"] is True
    assert report["local_mlx_prefilter_profile"]["cache_backed"] is True
    assert report["local_mlx_prefilter_profile"]["n_samples"] == 2
    cleanup = report["local_mlx_prefilter_profile"]["cache_cleanup"]
    assert cleanup["reference_cache_cleanup"]["blockers"] == []
    assert cleanup["candidate_cache_cleanup"]["blockers"] == []
    assert len(cleanup["reference_cache_cleanup"]["deleted_files"]) == 3
    assert len(cleanup["candidate_cache_cleanup"]["deleted_files"]) == 3
    for row in (
        cleanup["reference_cache_cleanup"]["deleted_files"]
        + cleanup["candidate_cache_cleanup"]["deleted_files"]
    ):
        assert row["delete_certified_rebuildable"] is True
        assert not Path(row["path"]).exists()
    assert calls["video_cache"][0] == source_video.as_posix()
    assert calls["raw_cache"][0] == raw.as_posix()
    response_kwargs = calls["response"]
    assert response_kwargs["device_type"] == "gpu"
    assert response_kwargs["allow_gpu_research_signal"] is True
    assert response_kwargs["allow_unaudited_candidate_cache_debug"] is True
    profile = json.loads(Path(report["local_mlx_prefilter_profile_path"]).read_text())
    assert profile["schema"] == "mlx_scorer_response.v1"
    assert profile["cache_quality_gate"]["schema"] == "mlx_cache_quality_gate.v1"
    assert profile["cache_quality_gate"]["verdict"] == "CACHE_QUALITY_GATE_FAILED"
    assert report["local_mlx_prefilter_profile"]["cache_quality_gate"]["schema"] == (
        "mlx_cache_quality_gate.v1"
    )
    assert profile["snerv_receiver_raw_cache_prefilter"][
        "source_pair_indices_alignment"
    ] == "prefix_source_pair_indices"
