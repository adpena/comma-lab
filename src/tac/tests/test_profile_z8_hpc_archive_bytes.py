# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from tac.substrates.z8_hierarchical_predictive_coding.archive import pack_archive
from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    pack_pair_pyramids_to_wavelet_blob,
)
from tac.substrates.z8_hierarchical_predictive_coding.mallat_dwt_adapter import (
    WaveletDetail2D,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL_PATH = REPO_ROOT / "tools" / "profile_z8_hpc_archive_bytes.py"


def _load_tool_module():
    spec = importlib.util.spec_from_file_location("profile_z8_hpc_archive_bytes", TOOL_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_profile_z8_hpc_archive_bytes_reports_wavelet_rate_surfaces(tmp_path: Path) -> None:
    tool = _load_tool_module()
    archive_bin = tmp_path / "0.bin"
    archive_bin.write_bytes(_build_tiny_z8hpc_archive_bytes())

    report = tool.build_profile(
        archive_bin=archive_bin,
        archive_zip=None,
        headroom_json=None,
        replay_json=None,
        frontier_archive_bytes=128,
        brotli_quality=5,
        measure_solid_pair_brotli=True,
        solid_brotli_quality=1,
    )

    assert report["schema"] == "z8_hpc_archive_byte_profile.v1"
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["archive"]["num_pairs"] == 1

    sections = {row["name"]: row for row in report["z8hpc1_sections"]}
    assert sections["wavelet_blob"]["bytes"] > sections["z8hpc1_header"]["bytes"]
    assert report["wavelet_blob_profile"]["num_pairs"] == 1
    assert report["wavelet_blob_profile"]["detail_coefficients"] > 0
    assert report["wavelet_blob_profile"]["top_ll_raw_payload_bytes"] > 0
    assert report["wavelet_blob_profile"]["detail_codec_method_counts"]
    assert "solid_pair_raw_brotli_bytes" in report["wavelet_blob_profile"]

    opportunity_names = {row["name"] for row in report["opportunities"]}
    assert "wavelet_blob_dominance" in opportunity_names
    assert "top_ll_float_payload" in opportunity_names
    assert "contest_rate_distance" in opportunity_names
    assert "Archive Byte Profile" in tool.render_markdown(report)


def _build_tiny_z8hpc_archive_bytes() -> bytes:
    top0 = np.arange(12, dtype=np.float32).reshape(2, 2, 3) / 255.0
    top1 = top0 + np.float32(1.0 / 255.0)
    detail = WaveletDetail2D(
        lh=np.full((2, 2, 3), 0.025, dtype=np.float32),
        hl=np.full((2, 2, 3), -0.015, dtype=np.float32),
        hh=np.zeros((2, 2, 3), dtype=np.float32),
    )
    wavelet_blob = pack_pair_pyramids_to_wavelet_blob(
        [
            {
                "frame_0_top_ll": top0,
                "frame_1_top_ll": top1,
                "frame_0_details": [detail],
                "frame_1_details": [detail],
            }
        ],
        detail_quantization_step=0.01,
    )
    return pack_archive(
        decoder_state_dict={"decoder.weight": np.ones((1, 1), dtype=np.float32)},
        per_level_category_indices=[
            np.zeros((1, 1), dtype=np.int32),
            np.zeros((1, 1), dtype=np.int32),
            np.zeros((1, 1), dtype=np.int32),
        ],
        wavelet_coeffs_blob=wavelet_blob,
        wyner_ziv_top_blob=b"wz",
        dreamer_state_dict={"dreamer.weight": np.ones((1,), dtype=np.float32)},
        meta={"fixture": "tiny_z8_profile"},
        num_levels=3,
        num_groups_per_level=(1, 1, 1),
        num_categories_per_level=(2, 2, 2),
        num_pairs=1,
        decoder_latent_dim=1,
        base_channels=1,
        wavelet_basis_id=0,
    )
