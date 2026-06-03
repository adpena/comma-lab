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
