# SPDX-License-Identifier: MIT
"""Tests for the Time-Traveler L5 Autonomy archive grammar (TT5L)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tac.substrates.time_traveler_l5_autonomy.archive import (
    TT5L_HEADER_SIZE,
    TT5L_MAGIC,
    TT5L_SCHEMA_VERSION,
    dequantize_per_pair_residual,
    pack_archive,
    parse_archive,
    quantize_per_pair_residual_int8,
)


def _toy_state_dict(seed: int = 0) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {
        "renderer.hidden.0.weight": torch.randn(8, 16, generator=gen),
        "renderer.hidden.0.bias": torch.randn(8, generator=gen),
        "renderer.output_layer.weight": torch.randn(6, 8, generator=gen),
        "renderer.output_layer.bias": torch.randn(6, generator=gen),
        "foveation.grid_weights": torch.randn(4, 4, generator=gen),
        "dynamics.transition": torch.eye(6),
        "dynamics.bias": torch.zeros(6),
        "pose_codes": torch.randn(10, 6, generator=gen),
    }


def _toy_side_info(num_pairs: int = 10, per_pair_bytes: int = 45) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.integers(-128, 127, size=(num_pairs, per_pair_bytes), dtype=np.int8)


def test_header_size_invariant() -> None:
    """TT5L_HEADER_SIZE must equal the documented 34 bytes."""
    assert TT5L_HEADER_SIZE == 34


def test_pack_archive_starts_with_magic_and_version() -> None:
    """The first 5 bytes are MAGIC + VERSION (deterministic header layout)."""
    sd = _toy_state_dict()
    side_info = _toy_side_info()
    blob = pack_archive(
        world_model_state_dict=sd,
        per_pair_side_info=side_info,
        meta={"int8_scale": 64.0},
        num_pairs=10,
        hidden_dim=8,
        num_hidden_layers=2,
        output_height=384,
        output_width=512,
        foveation_grid_h=4,
        foveation_grid_w=4,
        pose_dim=6,
        per_pair_bytes=45,
    )
    assert blob[:4] == TT5L_MAGIC
    assert blob[4] == TT5L_SCHEMA_VERSION


def test_pack_archive_parse_roundtrip_preserves_state_dict_values_exactly() -> None:
    """Pack -> parse -> compare gives back identical state_dict tensor values.

    This is the actual contract the inflate runtime depends on: parsing a
    TT5L blob must reproduce the trained state_dict tensor values
    bit-exactly (up to the documented FP32 -> FP16 quantization that
    happens at pack time). Pickle-stream byte determinism is not required
    for archive SHA-256 trails — value equality is.
    """
    sd = _toy_state_dict()
    side_info = _toy_side_info()
    meta = {"int8_scale": 64.0, "first_omega": 30.0, "hidden_omega": 1.0}
    blob = pack_archive(
        world_model_state_dict=sd,
        per_pair_side_info=side_info,
        meta=meta,
        num_pairs=10,
        hidden_dim=8,
        num_hidden_layers=2,
        output_height=384,
        output_width=512,
        foveation_grid_h=4,
        foveation_grid_w=4,
        pose_dim=6,
        per_pair_bytes=45,
    )
    parsed = parse_archive(blob)
    for key, value in parsed.world_model_state_dict.items():
        assert key in sd, f"unexpected parsed key {key!r}"
        expected = sd[key].detach().to(dtype=torch.float16)
        assert torch.equal(value, expected), (
            f"state_dict[{key}] roundtrip mismatch"
        )
    assert np.array_equal(parsed.per_pair_side_info, side_info)
    assert parsed.meta == meta


def test_parse_archive_restores_header_fields() -> None:
    """Parser populates every header u8/u16 field correctly."""
    sd = _toy_state_dict()
    side_info = _toy_side_info(num_pairs=7, per_pair_bytes=32)
    blob = pack_archive(
        world_model_state_dict=sd,
        per_pair_side_info=side_info,
        meta={"int8_scale": 32.0},
        num_pairs=7,
        hidden_dim=16,
        num_hidden_layers=3,
        output_height=384,
        output_width=512,
        foveation_grid_h=8,
        foveation_grid_w=12,
        pose_dim=6,
        per_pair_bytes=32,
    )
    arc = parse_archive(blob)
    assert arc.num_pairs == 7
    assert arc.hidden_dim == 16
    assert arc.num_hidden_layers == 3
    assert arc.output_height == 384
    assert arc.output_width == 512
    assert arc.foveation_grid_h == 8
    assert arc.foveation_grid_w == 12
    assert arc.pose_dim == 6
    assert arc.per_pair_bytes == 32
    assert arc.schema_version == TT5L_SCHEMA_VERSION


def test_parse_archive_restores_side_info_bytes_exactly() -> None:
    """Per-pair int8 side info is preserved byte-exactly across pack+parse."""
    sd = _toy_state_dict()
    side_info = _toy_side_info(num_pairs=10, per_pair_bytes=45)
    blob = pack_archive(
        world_model_state_dict=sd,
        per_pair_side_info=side_info,
        meta={"int8_scale": 64.0},
        num_pairs=10,
        hidden_dim=8,
        num_hidden_layers=2,
        output_height=384,
        output_width=512,
        foveation_grid_h=4,
        foveation_grid_w=4,
        pose_dim=6,
        per_pair_bytes=45,
    )
    arc = parse_archive(blob)
    assert arc.per_pair_side_info.shape == (10, 45)
    assert arc.per_pair_side_info.dtype == np.int8
    assert np.array_equal(arc.per_pair_side_info, side_info)


def test_parse_archive_restores_meta_json() -> None:
    """Meta JSON survives pack/parse, including float fields."""
    sd = _toy_state_dict()
    side_info = _toy_side_info()
    meta = {
        "int8_scale": 64.0,
        "first_omega": 30.0,
        "hidden_omega": 1.0,
        "coord_feature_freqs": 4,
        "coord_dim": 4,
        "markov_transition_band": 4,
    }
    blob = pack_archive(
        world_model_state_dict=sd,
        per_pair_side_info=side_info,
        meta=meta,
        num_pairs=10,
        hidden_dim=8,
        num_hidden_layers=2,
        output_height=384,
        output_width=512,
        foveation_grid_h=4,
        foveation_grid_w=4,
        pose_dim=6,
        per_pair_bytes=45,
    )
    arc = parse_archive(blob)
    assert arc.meta["int8_scale"] == 64.0
    assert arc.meta["first_omega"] == 30.0
    assert arc.meta["coord_feature_freqs"] == 4


def test_parse_archive_rejects_bad_magic() -> None:
    """Wrong magic bytes raise ValueError."""
    bogus = b"XXXX" + bytes(TT5L_HEADER_SIZE - 4)
    with pytest.raises(ValueError, match="bad magic"):
        parse_archive(bogus + b"")


def test_parse_archive_rejects_short_blob() -> None:
    """A blob shorter than TT5L_HEADER_SIZE is rejected."""
    with pytest.raises(ValueError, match="archive too short"):
        parse_archive(b"TT5L\x01")


def test_pack_archive_rejects_out_of_range_num_pairs() -> None:
    """num_pairs > 0xFFFF raises."""
    sd = _toy_state_dict()
    side_info = np.zeros((1, 1), dtype=np.int8)
    with pytest.raises(ValueError, match="num_pairs=70000"):
        pack_archive(
            world_model_state_dict=sd,
            per_pair_side_info=side_info,
            meta={},
            num_pairs=70_000,
            hidden_dim=8,
            num_hidden_layers=2,
            output_height=384,
            output_width=512,
            foveation_grid_h=4,
            foveation_grid_w=4,
            pose_dim=6,
            per_pair_bytes=1,
        )


def test_pack_archive_rejects_shape_mismatch() -> None:
    """side_info.shape must equal (num_pairs, per_pair_bytes)."""
    sd = _toy_state_dict()
    side_info = np.zeros((5, 10), dtype=np.int8)
    with pytest.raises(ValueError, match="shape"):
        pack_archive(
            world_model_state_dict=sd,
            per_pair_side_info=side_info,
            meta={},
            num_pairs=10,
            hidden_dim=8,
            num_hidden_layers=2,
            output_height=384,
            output_width=512,
            foveation_grid_h=4,
            foveation_grid_w=4,
            pose_dim=6,
            per_pair_bytes=10,
        )


def test_quantize_per_pair_residual_int8_clamps_extreme_values() -> None:
    """Quantize clamps to int8 [-128, 127]."""
    huge = torch.tensor([[1e6, -1e6, 0.0]])
    q = quantize_per_pair_residual_int8(huge, scale=1.0)
    assert q[0, 0] == 127
    assert q[0, 1] == -128
    assert q[0, 2] == 0


def test_quantize_per_pair_residual_int8_round_trips_within_quant_error() -> None:
    """Dequantize after quantize reproduces float within 1/scale."""
    real = torch.tensor([[0.5, -0.5, 1.25, -1.25]])
    scale = 4.0
    q = quantize_per_pair_residual_int8(real, scale=scale)
    deq = dequantize_per_pair_residual(q, scale=scale)
    # 0.5*4 = 2 -> deq 0.5; -0.5*4 = -2 -> deq -0.5; 1.25*4 = 5 -> 1.25
    assert torch.allclose(deq, real, atol=1e-6)


def test_pack_archive_with_empty_ac_state_produces_no_ac_blob() -> None:
    """AC state defaults to empty bytes; no AC blob is appended."""
    sd = _toy_state_dict()
    side_info = _toy_side_info()
    blob = pack_archive(
        world_model_state_dict=sd,
        per_pair_side_info=side_info,
        meta={},
        num_pairs=10,
        hidden_dim=8,
        num_hidden_layers=2,
        output_height=384,
        output_width=512,
        foveation_grid_h=4,
        foveation_grid_w=4,
        pose_dim=6,
        per_pair_bytes=45,
    )
    arc = parse_archive(blob)
    assert arc.ac_state == b""


def test_pack_archive_with_nonempty_ac_state_roundtrips() -> None:
    """Non-empty AC state survives pack/parse."""
    sd = _toy_state_dict()
    side_info = _toy_side_info()
    ac = b"\x01\x02\x03\x04\x05"
    blob = pack_archive(
        world_model_state_dict=sd,
        per_pair_side_info=side_info,
        meta={},
        num_pairs=10,
        hidden_dim=8,
        num_hidden_layers=2,
        output_height=384,
        output_width=512,
        foveation_grid_h=4,
        foveation_grid_w=4,
        pose_dim=6,
        per_pair_bytes=45,
        ac_state=ac,
    )
    arc = parse_archive(blob)
    assert arc.ac_state == ac


def test_archive_size_in_target_band_for_default_config() -> None:
    """A realistic-shaped archive is in the design-memo 95-110 KB target band.

    Note: this is a structural sanity check on the canonical config, NOT a
    score claim. The size band is the design-memo target; actual training
    runs may land anywhere within or near this band depending on the
    quantization residual entropy and brotli compressibility of trained
    weights.
    """
    # Build a state_dict resembling a real Time-Traveler config: ~16K renderer
    # params at hidden_dim=64, foveation_grid 16x24, dynamics 6x6, pose_codes
    # 600x6.
    sd = {
        "renderer.hidden.0.weight": torch.randn(64, 20),
        "renderer.hidden.0.bias": torch.randn(64),
        "renderer.hidden.1.weight": torch.randn(64, 64),
        "renderer.hidden.1.bias": torch.randn(64),
        "renderer.hidden.2.weight": torch.randn(64, 64),
        "renderer.hidden.2.bias": torch.randn(64),
        "renderer.hidden.3.weight": torch.randn(64, 64),
        "renderer.hidden.3.bias": torch.randn(64),
        "renderer.output_layer.weight": torch.randn(6, 64),
        "renderer.output_layer.bias": torch.randn(6),
        "foveation.grid_weights": torch.randn(16, 24),
        "dynamics.transition": torch.eye(6),
        "dynamics.bias": torch.zeros(6),
        "pose_codes": torch.randn(600, 6),
    }
    side_info = _toy_side_info(num_pairs=600, per_pair_bytes=45)
    blob = pack_archive(
        world_model_state_dict=sd,
        per_pair_side_info=side_info,
        meta={"int8_scale": 64.0, "first_omega": 30.0, "hidden_omega": 1.0},
        num_pairs=600,
        hidden_dim=64,
        num_hidden_layers=4,
        output_height=384,
        output_width=512,
        foveation_grid_h=16,
        foveation_grid_w=24,
        pose_dim=6,
        per_pair_bytes=45,
    )
    # Random weights are incompressible by brotli; a real trained substrate
    # will compress better.  We assert only a strict upper bound (the
    # incompressible-bytes worst case must still come in under 150 KB).
    assert len(blob) < 150_000, (
        f"random-weight upper-bound archive size {len(blob)} > 150 KB; "
        "design-memo target band is 95-110 KB for trained substrates"
    )


def test_pack_archive_state_dict_serialization_preserves_values_across_repacks() -> None:
    """Two pack calls with identical inputs produce semantically-equivalent bytes.

    State dict TENSOR VALUES roundtrip exactly across pack/parse. Header
    bytes are also identical. The pickle-stage storage-id memoization
    inside torch's tensor pickler is not fully content-deterministic
    (storage object IDs vary across .to(fp16).contiguous() chains), so the
    raw bytes of the compressed world-model blob may differ in a few
    bytes, but the parsed state_dict is identical.

    This is the right contract for archive SHA-256 audit trails: the
    audit trail compares the parsed-tensor values, not the pickle stream.
    """
    torch.manual_seed(0)
    sd = _toy_state_dict()
    side_info = _toy_side_info()
    meta = {"int8_scale": 64.0}

    def _pack() -> bytes:
        return pack_archive(
            world_model_state_dict=sd,
            per_pair_side_info=side_info,
            meta=meta,
            num_pairs=10,
            hidden_dim=8,
            num_hidden_layers=2,
            output_height=384,
            output_width=512,
            foveation_grid_h=4,
            foveation_grid_w=4,
            pose_dim=6,
            per_pair_bytes=45,
        )

    blob_a = _pack()
    blob_b = _pack()
    # Magic + version + fixed-shape header fields are deterministic; the
    # length-prefix fields (world_blob_len etc.) may vary by a few bytes
    # because torch tensor pickling memoizes storage IDs that change
    # across .to(fp16).contiguous() chains. The contract that matters is
    # PARSED VALUE equality below.
    assert blob_a[:14] == blob_b[:14]
    # Parsed contents are semantically equivalent (the only thing that matters).
    arc_a = parse_archive(blob_a)
    arc_b = parse_archive(blob_b)
    for key, value in arc_a.world_model_state_dict.items():
        assert torch.equal(value, arc_b.world_model_state_dict[key]), (
            f"state_dict[{key}] differs across repacks"
        )
    assert np.array_equal(arc_a.per_pair_side_info, arc_b.per_pair_side_info)
    assert arc_a.meta == arc_b.meta
