from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

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
    frames = tool.decode_snerv_archive_frames(packet.packet)
    assert frames.shape == (1, 2, 3, 16, 16)


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
