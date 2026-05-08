from __future__ import annotations

import numpy as np
import pytest

from tac.codec.frame_conditional_bit_budget import (
    ComplexityComponents,
    allocate_per_frame_bits,
    apply_frame_conditional_q_bits,
    build_frame_conditional_wire_contract,
    pack_frame_conditional_latent_codes,
    pack_frame_conditional_q_bits,
    unpack_frame_conditional_latent_codes,
    unpack_frame_conditional_q_bits,
)


def test_complexity_components_multiplies_three_axes() -> None:
    components = ComplexityComponents(
        edge_density=np.array([1.0, 2.0, 3.0]),
        pixel_variance=np.array([10.0, 20.0, 30.0]),
        frame_difference=np.array([0.5, 0.25, 0.125]),
    )

    np.testing.assert_allclose(components.complexity, np.array([5.0, 10.0, 11.25]))


def test_allocate_per_frame_bits_eta_zero_is_uniform() -> None:
    out = allocate_per_frame_bits(
        np.array([1.0, 10.0, 100.0]),
        total_bit_budget=300.0,
        eta=0.0,
    )

    np.testing.assert_allclose(out, np.array([100.0, 100.0, 100.0]))


def test_allocate_per_frame_bits_preserves_sum_and_respects_floor_cap() -> None:
    out = allocate_per_frame_bits(
        np.array([1.0, 2.0, 100.0, 200.0]),
        total_bit_budget=400.0,
        eta=1.0,
        floor=0.5,
        cap=1.5,
    )

    assert float(out.sum()) == pytest.approx(400.0, abs=1e-6)
    assert float(out.min()) >= 50.0 - 1e-6
    assert float(out.max()) <= 150.0 + 1e-6
    assert out[-1] >= out[-2] >= out[1] >= out[0]


def test_allocate_per_frame_bits_zero_complexity_falls_back_to_uniform() -> None:
    out = allocate_per_frame_bits(
        np.zeros(4, dtype=np.float64),
        total_bit_budget=80.0,
        eta=2.0,
    )

    np.testing.assert_allclose(out, np.array([20.0, 20.0, 20.0, 20.0]))


def test_allocate_per_frame_bits_single_frame_gets_total() -> None:
    out = allocate_per_frame_bits([42.0], total_bit_budget=123.0)

    np.testing.assert_allclose(out, np.array([123.0]))


def test_allocate_per_frame_bits_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        allocate_per_frame_bits([], 10.0)
    with pytest.raises(ValueError, match="1-D"):
        allocate_per_frame_bits(np.zeros((2, 2)), 10.0)
    with pytest.raises(ValueError, match="non-negative"):
        allocate_per_frame_bits([1.0, -1.0], 10.0)
    with pytest.raises(ValueError, match="total_bit_budget"):
        allocate_per_frame_bits([1.0], -1.0)
    with pytest.raises(ValueError, match="floor"):
        allocate_per_frame_bits([1.0, 2.0], 10.0, floor=1.2)
    with pytest.raises(ValueError, match="cap"):
        allocate_per_frame_bits([1.0, 2.0], 10.0, cap=0.9)


def test_pack_frame_conditional_q_bits_roundtrips_pr101_sideinfo_size() -> None:
    q_bits = np.resize(np.arange(1, 9, dtype=np.uint8), 600)

    packed = pack_frame_conditional_q_bits(q_bits)
    decoded = unpack_frame_conditional_q_bits(packed, n_pairs=600)

    assert len(packed) == 225
    np.testing.assert_array_equal(decoded, q_bits)


def test_unpack_frame_conditional_q_bits_fails_closed_on_bad_padding() -> None:
    packed = bytearray(pack_frame_conditional_q_bits([1, 2, 3]))
    packed[-1] |= 1

    with pytest.raises(ValueError, match="non-zero padding"):
        unpack_frame_conditional_q_bits(bytes(packed), n_pairs=3)


def test_frame_conditional_latent_codes_require_sideinfo_for_decode() -> None:
    q = np.array(
        [
            [255, 128, 17, 1],
            [255, 128, 17, 1],
        ],
        dtype=np.uint8,
    )
    q_bits = np.array([4, 8], dtype=np.uint8)

    packed = pack_frame_conditional_latent_codes(q, q_bits)
    decoded = unpack_frame_conditional_latent_codes(packed, q_bits, latent_dim=4)

    np.testing.assert_array_equal(
        decoded[0],
        np.array([240, 128, 16, 0], dtype=np.uint8),
    )
    np.testing.assert_array_equal(decoded[1], q[1])
    np.testing.assert_array_equal(decoded, apply_frame_conditional_q_bits(q, q_bits))
    with pytest.raises(ValueError, match="latent bitstream length"):
        unpack_frame_conditional_latent_codes(
            packed,
            np.array([8, 8], dtype=np.uint8),
            latent_dim=4,
        )


def test_build_frame_conditional_wire_contract_is_no_score_and_fail_closed() -> None:
    q = np.arange(24, dtype=np.uint8).reshape(3, 8)
    contract = build_frame_conditional_wire_contract(
        [2.9, 4.1, 8.0],
        latent_dim=8,
        q_pair_first=q,
    )

    assert contract["score_claim"] is False
    assert contract["ready_for_exact_eval_dispatch"] is False
    assert contract["decoder_helper_consumes_sideinfo_bytes"] is True
    assert contract["q_bits_sideinfo"]["bytes"] == 2
    assert contract["q_bits_roundtrip"]["passed"] is True
    assert contract["latent_decode_roundtrip"]["passed"] is True
    assert (
        "per_pair_bit_width_schema_change_requires_inflate_path_update"
        in contract["cleared_blockers"]
    )
    assert (
        "frame_conditional_packet_runtime_patch_not_built"
        in contract["remaining_blockers"]
    )
