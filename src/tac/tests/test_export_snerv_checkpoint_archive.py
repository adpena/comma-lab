# SPDX-License-Identifier: MIT
"""Tests for direct SNeRV checkpoint archive export."""

from __future__ import annotations

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
                },
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
        repo_root=tmp_path,
    )

    assert report["receiver_proof_passed"] is True
    assert report["local_mlx_prefilter_written"] is True
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
