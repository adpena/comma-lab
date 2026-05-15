from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tac.substrates.pretrained_driving_prior.archive import pack_archive
from tac.substrates.pretrained_driving_prior.architecture import (
    DrivingPriorRenderer,
    DrivingPriorRendererConfig,
)
from tac.substrates.pretrained_driving_prior.codebook import (
    DashcamCodebook,
    LANE_CURVATURE_PCA_SHAPE,
    ROAD_PLANE_BASIS_SHAPE,
    SKY_HORIZON_PROFILE_SHAPE,
    VEHICLE_APPEARANCE_BASIS_SHAPE,
    deterministic_zero_codebook,
)
from tac.substrates.pretrained_driving_prior.inflate import inflate_one_video


def _sky_codebook(value: int) -> DashcamCodebook:
    base = deterministic_zero_codebook()
    return DashcamCodebook(
        road_plane_basis=np.zeros(ROAD_PLANE_BASIS_SHAPE, dtype=np.int8),
        sky_horizon_profile=np.full(
            SKY_HORIZON_PROFILE_SHAPE,
            value,
            dtype=np.int8,
        ),
        lane_curvature_pca=np.zeros(LANE_CURVATURE_PCA_SHAPE, dtype=np.float16),
        vehicle_appearance_basis=np.zeros(
            VEHICLE_APPEARANCE_BASIS_SHAPE,
            dtype=np.int8,
        ),
        metadata={
            **base.metadata,
            "dataset_provenance": f"test_sky_{value}",
        },
    )


def _archive_for_codebook(book: DashcamCodebook) -> bytes:
    torch.manual_seed(123)
    cfg = DrivingPriorRendererConfig(
        hidden_dim=8,
        num_hidden_layers=2,
        output_height=16,
        output_width=16,
    )
    renderer = DrivingPriorRenderer(cfg)
    state_dict = {k: v.detach().cpu() for k, v in renderer.state_dict().items()}
    return pack_archive(
        book,
        state_dict,
        bytes([0] * 8),
        {
            "residual_int8_scale": 64.0,
            "prior_inflate_strength": 1.0,
            "test_fixture": "dp1_codebook_consumption",
        },
        num_pairs=1,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
        per_pair_bytes=8,
    )


def test_dp1_inflate_codebook_bytes_affect_raw_output(tmp_path: Path) -> None:
    dark_archive = _archive_for_codebook(_sky_codebook(0))
    bright_archive = _archive_for_codebook(_sky_codebook(64))

    dark_raw = tmp_path / "dark.raw"
    bright_raw = tmp_path / "bright.raw"
    assert inflate_one_video(dark_archive, dark_raw, device="cpu") == 2
    assert inflate_one_video(bright_archive, bright_raw, device="cpu") == 2

    dark_bytes = dark_raw.read_bytes()
    bright_bytes = bright_raw.read_bytes()
    assert len(dark_bytes) == len(bright_bytes)
    assert dark_bytes != bright_bytes

