# SPDX-License-Identifier: MIT
"""Tests for the Time-Traveler L5 Autonomy inflate runtime."""

from __future__ import annotations

import numpy as np
import torch

from tac.substrates.time_traveler_l5_autonomy.architecture import (
    TimeTravelerConfig,
    TimeTravelerSubstrate,
)
from tac.substrates.time_traveler_l5_autonomy.archive import (
    pack_archive,
)
from tac.substrates.time_traveler_l5_autonomy.inflate import (
    _apply_per_pair_residual,
    _build_substrate_from_archive,
    inflate_one_video,
)


def _build_toy_archive_bytes(num_pairs: int = 4) -> bytes:
    """Build a small TT5L archive from a freshly initialized substrate."""
    cfg = TimeTravelerConfig(
        hidden_dim=16,
        num_hidden_layers=2,
        output_height=64,
        output_width=96,
        num_pairs=num_pairs,
    )
    substrate = TimeTravelerSubstrate(cfg)
    sd = substrate.state_dict()
    side_info = np.zeros((num_pairs, cfg.per_pair_side_info_bytes), dtype=np.int8)
    meta = {
        "int8_scale": 64.0,
        "first_omega": cfg.first_omega,
        "hidden_omega": cfg.hidden_omega,
        "coord_feature_freqs": cfg.coord_feature_freqs,
        "coord_dim": cfg.coord_dim,
        "markov_transition_band": cfg.markov_transition_band,
    }
    return pack_archive(
        world_model_state_dict=sd,
        per_pair_side_info=side_info,
        meta=meta,
        num_pairs=cfg.num_pairs,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
        foveation_grid_h=cfg.foveation_grid_h,
        foveation_grid_w=cfg.foveation_grid_w,
        pose_dim=cfg.pose_dim,
        per_pair_bytes=cfg.per_pair_side_info_bytes,
    )


def test_build_substrate_from_archive_loads_state_dict() -> None:
    """The inflate-time substrate builder reconstructs a valid substrate."""
    archive_bytes = _build_toy_archive_bytes()
    from tac.substrates.time_traveler_l5_autonomy.archive import parse_archive

    arc = parse_archive(archive_bytes)
    substrate = _build_substrate_from_archive(arc, device="cpu")
    assert isinstance(substrate, TimeTravelerSubstrate)
    # Render works after load.
    rgb_0, rgb_1 = substrate.render_pair(0)
    assert rgb_0.shape == (1, 3, 64, 96)


def test_apply_per_pair_residual_returns_unit_range_outputs() -> None:
    """Residual application keeps RGB in [0, 1]."""
    rgb_0 = torch.full((1, 3, 64, 96), 0.5)
    rgb_1 = torch.full((1, 3, 64, 96), 0.5)
    side = torch.randn(45)
    out_0, out_1 = _apply_per_pair_residual(rgb_0, rgb_1, side, int8_scale=64.0)
    assert float(out_0.min()) >= 0.0
    assert float(out_0.max()) <= 1.0
    assert float(out_1.min()) >= 0.0
    assert float(out_1.max()) <= 1.0


def test_apply_per_pair_residual_zero_side_info_is_identity() -> None:
    """Zero side info => output equals input (modulo clamp)."""
    rgb_0 = torch.full((1, 3, 16, 16), 0.5)
    rgb_1 = torch.full((1, 3, 16, 16), 0.5)
    side = torch.zeros(45)
    out_0, out_1 = _apply_per_pair_residual(rgb_0, rgb_1, side, int8_scale=64.0)
    assert torch.allclose(out_0, rgb_0)
    assert torch.allclose(out_1, rgb_1)


def test_inflate_one_video_writes_expected_raw_byte_count(tmp_path) -> None:
    """The .raw output contains ``num_pairs * 2 * 874 * 1164 * 3`` uint8 bytes.

    The inflate runtime always upsamples to the contest's (874, 1164) native
    resolution regardless of the renderer's eval resolution.
    """
    archive_bytes = _build_toy_archive_bytes(num_pairs=2)
    out_path = tmp_path / "0.raw"
    n_frames = inflate_one_video(archive_bytes, out_path, device="cpu")
    assert n_frames == 4  # 2 pairs * 2 frames
    expected_bytes = 4 * 874 * 1164 * 3
    assert out_path.stat().st_size == expected_bytes


def test_inflate_one_video_handles_zero_side_info(tmp_path) -> None:
    """Side info = zeros must still produce a valid .raw output."""
    archive_bytes = _build_toy_archive_bytes(num_pairs=1)
    out_path = tmp_path / "test.raw"
    n_frames = inflate_one_video(archive_bytes, out_path, device="cpu")
    assert n_frames == 2
    assert out_path.is_file()


def test_inflate_main_cli_three_arg_contract(tmp_path) -> None:
    """The CLI honors ``inflate.py <archive_dir> <output_dir> <file_list>``."""
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    archive_bytes = _build_toy_archive_bytes(num_pairs=2)
    (archive_dir / "0.bin").write_bytes(archive_bytes)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    file_list = tmp_path / "files.txt"
    file_list.write_text("video_a.mkv\nvideo_b.mkv\n", encoding="utf-8")

    import sys

    from tac.substrates.time_traveler_l5_autonomy.inflate import main_cli

    saved_argv = sys.argv
    try:
        sys.argv = [
            "inflate.py", str(archive_dir), str(output_dir), str(file_list)
        ]
        rc = main_cli()
    finally:
        sys.argv = saved_argv
    assert rc == 0
    assert (output_dir / "video_a.raw").is_file()
    assert (output_dir / "video_b.raw").is_file()


def test_inflate_main_cli_falls_back_to_x_path(tmp_path) -> None:
    """If 0.bin is absent, CLI tries archive_dir/x (single-zip-member contract)."""
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    archive_bytes = _build_toy_archive_bytes(num_pairs=1)
    (archive_dir / "x").write_bytes(archive_bytes)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    file_list = tmp_path / "files.txt"
    file_list.write_text("v.mkv\n", encoding="utf-8")

    import sys

    from tac.substrates.time_traveler_l5_autonomy.inflate import main_cli

    saved_argv = sys.argv
    try:
        sys.argv = [
            "inflate.py", str(archive_dir), str(output_dir), str(file_list)
        ]
        rc = main_cli()
    finally:
        sys.argv = saved_argv
    assert rc == 0
    assert (output_dir / "v.raw").is_file()


def test_inflate_main_cli_returns_2_when_args_missing() -> None:
    """Missing positional args => CLI returns rc=2 (Catalog #146 contract)."""
    import sys

    from tac.substrates.time_traveler_l5_autonomy.inflate import main_cli

    saved_argv = sys.argv
    try:
        sys.argv = ["inflate.py"]
        rc = main_cli()
    finally:
        sys.argv = saved_argv
    assert rc == 2


def test_end_to_end_archive_pack_parse_inflate_byte_roundtrip(tmp_path) -> None:
    """Full grammar roundtrip: build substrate -> pack -> parse -> inflate."""
    archive_bytes = _build_toy_archive_bytes(num_pairs=3)
    out_path = tmp_path / "endtoend.raw"
    n_frames = inflate_one_video(archive_bytes, out_path, device="cpu")
    assert n_frames == 6
    # Two separate calls produce identical output (deterministic forward).
    out_path2 = tmp_path / "endtoend2.raw"
    n2 = inflate_one_video(archive_bytes, out_path2, device="cpu")
    assert n_frames == n2
    assert out_path.read_bytes() == out_path2.read_bytes()
