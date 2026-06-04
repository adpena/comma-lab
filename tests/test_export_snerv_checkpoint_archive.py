from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from tac.substrates._shared.numpy_portable_inflate import pack_state_dict_numpy
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    decode_official_mfu_hfr_tub_decoder_payload,
    unpack_snerv_archive,
)

TOOL_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "export_snerv_checkpoint_archive.py"
)


def _load_tool():
    spec = importlib.util.spec_from_file_location(
        "export_snerv_checkpoint_archive", TOOL_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_checkpoint_packet_resolves_portfolio_decoder_codec() -> None:
    tool = _load_tool()
    model_size = tool.SnervModelSizeConfig(
        fc_dim=1,
        emb_size=0,
        patch_radius=0,
        temporal_context=0,
    )
    state = {
        "latents_lf_planes": np.zeros((1, 2, 3, 1, 1), dtype=np.float32),
        "decoder_kernels.0.LH": np.asarray([0.0125], dtype=np.float32),
        "decoder_kernels.0.HL": np.asarray([-0.0075], dtype=np.float32),
        "decoder_kernels.0.HH": np.asarray([0.0], dtype=np.float32),
    }

    packet = tool.build_snerv_checkpoint_packet(
        state,
        levels=1,
        wavelet="haar",
        target_bits_per_coeff=1.0,
        step_map_bits_per_coeff=0.25,
        decoder_payload_codec="portfolio_auto",
        lf_payload_codec="portfolio_auto",
        model_size=model_size,
    )

    assert packet.metadata["decoder_payload_codec"] == "mixed_magnitude_symmetric"
    assert packet.metadata["decoder_payload_codec_requested"] == "portfolio_auto"
    assert packet.metadata["lf_payload_codec"] == "portfolio_auto"
    assert packet.section_bytes["decoder_payload"] > 0
    assert packet.section_bytes["lf_payload"] > 0


def test_official_checkpoint_packet_uses_official_payload() -> None:
    tool = _load_tool()
    model_size = tool.SnervModelSizeConfig(
        adapter="snerv_official_mfu_hfr_tub_numeric_primitives_v1",
        official_skip_high_mode="scalar_mean",
    )
    state = _official_checkpoint_state()

    packet = tool.build_snerv_official_checkpoint_packet(
        state,
        model_size=model_size,
        metadata_extra={"unit_test": True},
    )

    assert packet.metadata["decoder_payload_codec"] == (
        "snerv_decoder_payload.official_mfu_hfr_tub.v1"
    )
    assert packet.metadata["checkpoint_packetization_mode"] == (
        "official_mfu_hfr_tub_receiver_payload"
    )
    assert packet.metadata["snerv_official_mfu_hfr_tub_export_bound"] is True
    assert packet.metadata["official_skip_high_export_is_compact_train_state"] is True
    assert packet.metadata["official_skip_high_full_shape"] == [2, 3, 8, 8]
    summary = tool._packet_metadata_summary(packet)
    assert summary["checkpoint_packetization_mode"] == (
        "official_mfu_hfr_tub_receiver_payload"
    )
    assert summary["decoder_payload_codec"] == (
        "snerv_decoder_payload.official_mfu_hfr_tub.v1"
    )
    assert summary["snerv_official_mfu_hfr_tub_export_bound"] is True
    assert summary["snerv_official_mfu_hfr_tub_export_bound_semantics"] == (
        "receiver_payload_bound_not_source_forward_parity"
    )
    assert summary["snerv_official_mfu_hfr_tub_receiver_payload_bound"] is True
    assert summary["snerv_official_mfu_hfr_tub_source_forward_replay_bound"] is False
    assert (
        summary["snerv_official_mfu_hfr_tub_source_forward_replay_authority"]
        is False
    )
    assert summary["source_faithful_stack"] is False
    frames = tool.decode_snerv_archive_frames(packet.packet)
    assert frames.shape == (1, 2, 3, 16, 16)


def test_official_checkpoint_export_report_classifies_receiver_payload(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    state_path = tmp_path / "official_state.npsd"
    state_path.write_bytes(pack_state_dict_numpy(_official_checkpoint_state()))
    checkpoint_meta = tmp_path / "checkpoint.meta.json"
    checkpoint_meta.write_text(
        json.dumps(
            {
                "global_epoch": 29,
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
                    "candidate_id": "official_checkpoint_unit",
                    "snerv_model_size_adapter": (
                        "snerv_official_mfu_hfr_tub_numeric_primitives_v1"
                    ),
                    "official_skip_high_mode": "scalar_mean",
                },
                "hard_byte_ceilings": [100_000],
                "command_args": {"num_pairs": 1},
            }
        ),
        encoding="utf-8",
    )

    report = tool.export_snerv_checkpoint_archive(
        startup_json=startup,
        checkpoint_meta=checkpoint_meta,
        output_dir=tmp_path / "export",
        state_kind="ema",
        repo_root=tmp_path,
    )

    binding = report["official_checkpoint_export_binding"]
    assert binding["schema"] == "snerv_official_checkpoint_export_binding.v1"
    assert binding["requested"] is True
    assert binding["native_checkpoint_export_bound_to_official_payload"] is True
    assert binding["official_receiver_payload_bound"] is True
    assert binding["official_receiver_tensor_map_verified"] is True
    assert binding["selected_packet_status"] == "frame_producing_official_export"
    assert binding["selected_packet_decoded_frame_shape"] == [1, 2, 3, 16, 16]
    manifest = binding["official_trained_checkpoint_mapping_manifest"]
    assert manifest["schema"] == (
        "snerv_official_trained_checkpoint_state_dict_mapping_manifest.v1"
    )
    assert manifest["official_trained_checkpoint_loaded"] is False
    assert "snerv_official_trained_checkpoint_state_dict_not_loaded" in manifest[
        "blockers"
    ]
    assert binding["official_trained_checkpoint_state_dict_slice_present"] is False
    assert (
        binding["official_mfu_hfr_trained_checkpoint_weight_mapping_proven"]
        is False
    )
    assert binding["official_tub_temporal_encoder_weight_mapping_proven"] is False
    assert (
        binding["official_trained_checkpoint_state_dict_mapping_verified"] is False
    )
    assert binding["official_export_bound"] is False
    assert binding["source_forward_replay_authority"] is False
    assert "snerv_official_trained_checkpoint_state_dict_mapping_missing" in (
        binding["preserved_blockers"]
    )
    assert "snerv_official_trained_checkpoint_source_forward_replay_missing" in (
        binding["preserved_blockers"]
    )
    assert binding["official_tensor_category_bytes"]
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_official_checkpoint_export_report_consumes_upstream_decoder_keys(
    tmp_path: Path,
) -> None:
    tool = _load_tool()
    state_path = tmp_path / "official_state_with_source_keys.npsd"
    state_path.write_bytes(
        pack_state_dict_numpy(_official_checkpoint_state_with_upstream_mfu_hfr_keys())
    )
    checkpoint_meta = tmp_path / "checkpoint.meta.json"
    checkpoint_meta.write_text(
        json.dumps(
            {
                "global_epoch": 30,
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
                    "candidate_id": "official_checkpoint_source_key_unit",
                    "snerv_model_size_adapter": (
                        "snerv_official_mfu_hfr_tub_numeric_primitives_v1"
                    ),
                    "official_skip_high_mode": "scalar_mean",
                },
                "hard_byte_ceilings": [100_000],
                "command_args": {"num_pairs": 1},
            }
        ),
        encoding="utf-8",
    )

    report = tool.export_snerv_checkpoint_archive(
        startup_json=startup,
        checkpoint_meta=checkpoint_meta,
        output_dir=tmp_path / "export",
        state_kind="ema",
        repo_root=tmp_path,
    )

    binding = report["official_checkpoint_export_binding"]
    manifest = binding["official_trained_checkpoint_mapping_manifest"]
    assert manifest["official_trained_checkpoint_loaded"] is True
    assert manifest["decoder_len"] == 8
    assert manifest["decoder_len_source"] == "inferred_from_decoder_prefixes"
    assert manifest["state_dict_kind"] == (
        "checkpoint_export_upstream_official_state_dict_slice"
    )
    assert manifest["official_mfu_hfr_trained_checkpoint_weight_mapping_proven"] is True
    assert manifest["official_tub_temporal_encoder_weight_mapping_proven"] is False
    rows = {row["component_id"]: row for row in manifest["component_rows"]}
    assert rows["hfr"]["trained_checkpoint_weight_mapping_proven"] is True
    assert rows["mfu"]["trained_checkpoint_weight_mapping_proven"] is True
    assert rows["tub"]["trained_checkpoint_weight_mapping_proven"] is False
    receiver_keys = {row["receiver_key"] for row in manifest["weight_entries"]}
    assert "hfr.lh.conv1.weight" in receiver_keys
    assert "mfu.upsample_mid.weight" in receiver_keys
    assert "mfu.rb_high.residual_blocks.0.conv2.bias" in receiver_keys
    assert binding["official_trained_checkpoint_state_dict_slice_present"] is True
    assert (
        binding["official_mfu_hfr_trained_checkpoint_weight_mapping_proven"]
        is True
    )
    assert binding["official_tub_temporal_encoder_weight_mapping_proven"] is False
    assert (
        binding["official_trained_checkpoint_state_dict_mapping_verified"] is False
    )
    assert "snerv_official_trained_checkpoint_state_dict_mapping_missing" not in (
        binding["preserved_blockers"]
    )
    assert "snerv_official_trained_checkpoint_source_forward_replay_missing" in (
        binding["preserved_blockers"]
    )
    assert "snerv_official_tub_portable_temporal_encoder_weight_mapping_missing" in (
        binding["preserved_blockers"]
    )
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False


def test_official_checkpoint_packet_preserves_tub_output2_payload_from_state() -> None:
    tool = _load_tool()
    model_size = tool.SnervModelSizeConfig(
        adapter="snerv_official_mfu_hfr_tub_numeric_primitives_v1",
        official_skip_high_mode="scalar_mean",
    )
    state = _official_checkpoint_state()
    state["tub.temporal_encoder_concat"] = np.arange(
        1 * 4 * 4 * 4,
        dtype=np.float32,
    ).reshape(1, 4, 4, 4)
    state["tub.output2_raw"] = (
        np.arange(2 * 8 * 4 * 4, dtype=np.float32).reshape(2, 8, 4, 4) / 31.0
    )

    packet = tool.build_snerv_official_checkpoint_packet(
        state,
        model_size=model_size,
        metadata_extra={"unit_test": True},
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


def test_official_checkpoint_packet_fails_closed_on_incomplete_tub_output2_pair() -> None:
    tool = _load_tool()
    model_size = tool.SnervModelSizeConfig(
        adapter="snerv_official_mfu_hfr_tub_numeric_primitives_v1",
        official_skip_high_mode="scalar_mean",
    )
    state = _official_checkpoint_state()
    state["tub.temporal_encoder_concat"] = np.zeros((1, 4, 4, 4), dtype=np.float32)

    try:
        tool.build_snerv_official_checkpoint_packet(
            state,
            model_size=model_size,
            metadata_extra={"unit_test": True},
        )
    except ValueError as exc:
        assert "requires both tub.temporal_encoder_concat and tub.output2_raw" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("official checkpoint output2 export must fail closed")


def test_official_checkpoint_packet_fails_closed_without_official_atoms() -> None:
    tool = _load_tool()
    model_size = tool.SnervModelSizeConfig(
        adapter="snerv_official_mfu_hfr_tub_numeric_primitives_v1",
        official_skip_high_mode="scalar_mean",
    )

    try:
        tool.build_snerv_official_checkpoint_packet(
            {"latents_lf_planes": np.zeros((1, 2, 3, 1, 1), dtype=np.float32)},
            model_size=model_size,
        )
    except ValueError as exc:
        assert "official checkpoint state missing" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("official checkpoint export must fail closed")


def _official_checkpoint_state() -> dict[str, np.ndarray]:
    state: dict[str, np.ndarray] = {
        "low": np.zeros((2, 3, 2, 2), dtype=np.float32),
        "skip_mid": np.zeros((2, 3, 4, 4), dtype=np.float32),
        "skip_high": np.asarray([[[[32.0]]]], dtype=np.float32),
    }
    for name in ("lh", "hl", "hh"):
        state[f"hfr_{name}_conv1_weight"] = np.zeros((3, 3, 1, 1), dtype=np.float32)
        state[f"hfr_{name}_conv1_bias"] = np.zeros((3,), dtype=np.float32)
        state[f"hfr_{name}_conv2_weight"] = np.zeros((3, 3, 3, 3), dtype=np.float32)
        state[f"hfr_{name}_conv2_bias"] = np.zeros((3,), dtype=np.float32)
    return state


def _official_checkpoint_state_with_upstream_mfu_hfr_keys() -> dict[str, np.ndarray]:
    state = _official_checkpoint_state()
    state.update(_minimal_official_mfu_hfr_state(decoder_len=8))
    return state


def _minimal_official_mfu_hfr_state(decoder_len: int) -> dict[str, np.ndarray]:
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
    return state
