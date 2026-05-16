# SPDX-License-Identifier: MIT
"""Tests for the Time-Traveler L5 Autonomy inflate runtime."""

from __future__ import annotations

from collections.abc import Callable

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
    apply_quantized_per_pair_residual_for_training,
    inflate_one_video,
    quantize_per_pair_residual_for_inflate_ste,
)


def _build_toy_archive_bytes(
    num_pairs: int = 4,
    *,
    side_info: np.ndarray | None = None,
    ac_state: bytes = b"",
    state_mutator: Callable[[dict[str, torch.Tensor]], None] | None = None,
) -> bytes:
    """Build a small TT5L archive from a freshly initialized substrate."""
    torch.manual_seed(0)
    cfg = TimeTravelerConfig(
        hidden_dim=16,
        num_hidden_layers=2,
        output_height=64,
        output_width=96,
        num_pairs=num_pairs,
    )
    substrate = TimeTravelerSubstrate(cfg)
    sd = substrate.state_dict()
    if state_mutator is not None:
        sd = {key: value.clone() for key, value in sd.items()}
        state_mutator(sd)
    if side_info is None:
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
        ac_state=ac_state,
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


def test_apply_per_pair_residual_consumes_every_canonical_section() -> None:
    """Each TT5L side-info section must affect decoded RGB output."""
    rgb_0 = torch.full((1, 3, 32, 48), 0.5)
    rgb_1 = torch.full((1, 3, 32, 48), 0.5)
    zero_side = torch.zeros(45)
    zero_0, zero_1 = _apply_per_pair_residual(
        rgb_0,
        rgb_1,
        zero_side,
        int8_scale=64.0,
    )
    sections = {
        "se3_lie": slice(0, 12),
        "seg_boundary": slice(12, 30),
        "hf_residual": slice(30, 36),
        "predict_residual": slice(36, 45),
    }
    for name, span in sections.items():
        side = torch.zeros(45)
        side[span] = 64.0
        out_0, out_1 = _apply_per_pair_residual(
            rgb_0,
            rgb_1,
            side,
            int8_scale=64.0,
        )
        moved = not torch.allclose(out_0, zero_0) or not torch.allclose(out_1, zero_1)
        assert moved, f"TT5L side-info section {name} did not affect output"


def test_training_side_info_transform_matches_inflate_quantized_path() -> None:
    """Training applies the same quantized side-info residual as inflate."""
    rgb_0 = torch.full((1, 3, 16, 16), 0.5)
    rgb_1 = torch.full((1, 3, 16, 16), 0.25)
    side_float = torch.zeros(45, requires_grad=True)
    side_float.data[:] = torch.linspace(-0.5, 0.5, 45)

    train_0, train_1, side_int8_ste = apply_quantized_per_pair_residual_for_training(
        rgb_0,
        rgb_1,
        side_float,
        int8_scale=64.0,
    )
    expected_side_int8 = quantize_per_pair_residual_for_inflate_ste(
        side_float,
        int8_scale=64.0,
    ).detach()
    inflate_0, inflate_1 = _apply_per_pair_residual(
        rgb_0,
        rgb_1,
        expected_side_int8,
        int8_scale=64.0,
    )
    assert torch.equal(side_int8_ste.detach(), expected_side_int8)
    assert torch.allclose(train_0, inflate_0)
    assert torch.allclose(train_1, inflate_1)

    (train_0.mean() + train_1.mean()).backward()
    assert side_float.grad is not None
    for name, span in {
        "se3_lie": slice(0, 12),
        "seg_boundary": slice(12, 30),
        "hf_residual": slice(30, 36),
        "predict_residual": slice(36, 45),
    }.items():
        assert side_float.grad[span].abs().sum().item() > 0.0, name


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


def test_inflate_one_video_side_info_bytes_affect_decoded_frames(tmp_path) -> None:
    """Mutating side-info bytes changes the decoded raw frames."""
    zero_side = np.zeros((1, 45), dtype=np.int8)
    active_side = zero_side.copy()
    active_side[0, -9:] = 64

    zero_archive = _build_toy_archive_bytes(num_pairs=1, side_info=zero_side)
    active_archive = _build_toy_archive_bytes(num_pairs=1, side_info=active_side)
    zero_out = tmp_path / "zero.raw"
    active_out = tmp_path / "active.raw"

    assert inflate_one_video(zero_archive, zero_out, device="cpu") == 2
    assert inflate_one_video(active_archive, active_out, device="cpu") == 2
    assert zero_out.read_bytes() != active_out.read_bytes()


def test_inflate_one_video_ac_state_bytes_affect_decoded_frames(tmp_path) -> None:
    """Non-empty AC state is consumed by the inflate residual path."""
    side = np.zeros((1, 45), dtype=np.int8)
    side[0, 36:45] = 64
    ac_state_a = bytes([0, 64, 128, 192, 255] * 4)
    ac_state_b = bytes([255, 192, 128, 64, 0] * 4)

    archive_a = _build_toy_archive_bytes(
        num_pairs=1,
        side_info=side,
        ac_state=ac_state_a,
    )
    archive_b = _build_toy_archive_bytes(
        num_pairs=1,
        side_info=side,
        ac_state=ac_state_b,
    )
    out_a = tmp_path / "ac_a.raw"
    out_b = tmp_path / "ac_b.raw"

    assert inflate_one_video(archive_a, out_a, device="cpu") == 2
    assert inflate_one_video(archive_b, out_b, device="cpu") == 2
    assert out_a.read_bytes() != out_b.read_bytes()


def test_inflate_one_video_world_model_pose_and_dynamics_bytes_affect_decoded_frames(
    tmp_path,
) -> None:
    """Pose-code and dynamics bytes in WORLD_MODEL_BLOB affect raw output."""

    def mutate_pose_codes(sd: dict[str, torch.Tensor]) -> None:
        sd["pose_codes"][0, 0] += 8.0

    def mutate_dynamics_bias(sd: dict[str, torch.Tensor]) -> None:
        sd["dynamics.bias"][1] += 8.0

    base_archive = _build_toy_archive_bytes(num_pairs=1)
    pose_archive = _build_toy_archive_bytes(
        num_pairs=1,
        state_mutator=mutate_pose_codes,
    )
    dynamics_archive = _build_toy_archive_bytes(
        num_pairs=1,
        state_mutator=mutate_dynamics_bias,
    )
    base_out = tmp_path / "base.raw"
    pose_out = tmp_path / "pose.raw"
    dynamics_out = tmp_path / "dynamics.raw"

    assert inflate_one_video(base_archive, base_out, device="cpu") == 2
    assert inflate_one_video(pose_archive, pose_out, device="cpu") == 2
    assert inflate_one_video(dynamics_archive, dynamics_out, device="cpu") == 2
    base_bytes = base_out.read_bytes()
    assert pose_out.read_bytes() != base_bytes
    assert dynamics_out.read_bytes() != base_bytes


def test_inflate_one_video_consumes_all_side_info_sections(tmp_path) -> None:
    """Archive-level mutation proof for all canonical TT5L side-info sections."""
    zero_side = np.zeros((1, 45), dtype=np.int8)
    zero_archive = _build_toy_archive_bytes(num_pairs=1, side_info=zero_side)
    zero_out = tmp_path / "zero_sections.raw"
    assert inflate_one_video(zero_archive, zero_out, device="cpu") == 2
    zero_bytes = zero_out.read_bytes()

    sections = {
        "se3_lie": slice(0, 12),
        "seg_boundary": slice(12, 30),
        "hf_residual": slice(30, 36),
        "predict_residual": slice(36, 45),
    }
    for name, span in sections.items():
        active_side = zero_side.copy()
        active_side[0, span] = 64
        active_archive = _build_toy_archive_bytes(num_pairs=1, side_info=active_side)
        active_out = tmp_path / f"{name}.raw"
        assert inflate_one_video(active_archive, active_out, device="cpu") == 2
        assert active_out.read_bytes() != zero_bytes, name


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
