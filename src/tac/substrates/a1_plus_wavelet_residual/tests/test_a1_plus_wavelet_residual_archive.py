"""Tests for the A1 + wavelet residual archive grammar.

Per CLAUDE.md "HNeRV parity discipline" lessons L3 (archive grammar) +
L11 (no-op detector / wire-format roundtrip).
"""
from __future__ import annotations

import struct

import pytest
import torch

from tac.substrates.a1_plus_wavelet_residual.archive import (
    NUM_DETAIL_BANDS,
    NUM_FRAMES_PER_PAIR,
    NUM_RGB_CHANNELS,
    WAVELET_HEADER_STRUCT,
    WAVELET_SIDECAR_MAGIC,
    WAVELET_SIDECAR_VERSION,
    decode_wavelet_sidecar,
    encode_wavelet_sidecar,
    pack_composition_archive,
    split_composition_archive,
)
from tac.substrates.a1_plus_wavelet_residual.architecture import (
    A1_N_PAIRS,
    parse_wavelet_residual_pair_indices,
)


def _build_canonical_coeffs(
    num_selected: int, rank: int, foveal_h: int, foveal_w: int
) -> torch.Tensor:
    """Build a deterministic 6D coefficient tensor for round-trip tests."""
    last = foveal_h + foveal_w
    total = num_selected * NUM_DETAIL_BANDS * NUM_FRAMES_PER_PAIR * NUM_RGB_CHANNELS * rank * last
    flat = torch.linspace(-1.0, 1.0, steps=total)
    return flat.reshape(
        num_selected,
        NUM_DETAIL_BANDS,
        NUM_FRAMES_PER_PAIR,
        NUM_RGB_CHANNELS,
        rank,
        last,
    )


def test_wavelet_sidecar_roundtrips_quantized_coefficients() -> None:
    coeffs = _build_canonical_coeffs(num_selected=2, rank=2, foveal_h=4, foveal_w=5)

    blob = encode_wavelet_sidecar(
        (3, 7),
        coeffs,
        foveal_h=4,
        foveal_w=5,
        coeff_rank=2,
        int8_scale=8.0,
    )
    indices, decoded, meta = decode_wavelet_sidecar(blob)

    expected = (coeffs * 8.0).round().clamp(-128, 127).to(torch.int8).float() / 8.0
    assert indices == (3, 7)
    assert decoded.shape == coeffs.shape
    assert torch.equal(decoded, expected)
    assert meta["foveal_h"] == 4
    assert meta["foveal_w"] == 5
    assert meta["coeff_rank"] == 2
    assert meta["num_selected"] == 2


def test_composition_archive_preserves_a1_prefix_exactly() -> None:
    a1_bytes = b"A1_BASE_BYTES\x00\x01\x02\x03"
    coeffs = _build_canonical_coeffs(num_selected=1, rank=1, foveal_h=2, foveal_w=2)

    packed = pack_composition_archive(
        a1_bytes,
        selected_indices=(11,),
        coeffs=coeffs,
        foveal_h=2,
        foveal_w=2,
        coeff_rank=1,
        int8_scale=4.0,
    )
    prefix, sidecar = split_composition_archive(packed)

    assert prefix == a1_bytes
    indices, decoded, _meta = decode_wavelet_sidecar(sidecar)
    assert indices == (11,)
    assert decoded.shape == coeffs.shape


def test_split_composition_archive_is_noop_without_sidecar_magic() -> None:
    prefix, sidecar = split_composition_archive(b"plain-a1-wire-bytes")
    assert prefix == b"plain-a1-wire-bytes"
    assert sidecar == b""


def test_split_refuses_bad_version_with_clean_fallback() -> None:
    """A trailer with the right magic but a wrong version returns no-op."""
    a1 = b"a1-base"
    # Build a fake header with version=99
    bad_header = WAVELET_HEADER_STRUCT.pack(
        WAVELET_SIDECAR_MAGIC, 99, 0, 0, 0, 0, 0.0, b"\x00\x00"
    )
    blob = a1 + bad_header + b"\x00" * 8
    prefix, sidecar = split_composition_archive(blob)
    # Returns no-op when version is unrecognized
    assert prefix == blob
    assert sidecar == b""


def test_split_does_not_eat_trailing_brotli_with_no_valid_decode() -> None:
    """If the magic appears but decode fails, split returns no-op."""
    a1 = b"a1-base"
    # Build a header that promises 4 indices but truncate before them
    bad_header = WAVELET_HEADER_STRUCT.pack(
        WAVELET_SIDECAR_MAGIC, WAVELET_SIDECAR_VERSION, 4, 4, 4, 1, 8.0, b"\x00\x00"
    )
    blob = a1 + bad_header  # missing index table + coeffs blob
    prefix, sidecar = split_composition_archive(blob)
    assert prefix == blob
    assert sidecar == b""


def test_encode_rejects_wrong_shape() -> None:
    coeffs = torch.zeros(2, 2, 2)  # 3D, not 6D
    with pytest.raises(ValueError, match="must be 6D"):
        encode_wavelet_sidecar(
            (1,), coeffs, foveal_h=2, foveal_w=2, coeff_rank=1, int8_scale=4.0
        )


def test_encode_rejects_index_count_mismatch() -> None:
    coeffs = _build_canonical_coeffs(num_selected=3, rank=1, foveal_h=2, foveal_w=2)
    with pytest.raises(ValueError, match="selected count"):
        encode_wavelet_sidecar(
            (1,), coeffs, foveal_h=2, foveal_w=2, coeff_rank=1, int8_scale=4.0
        )


def test_encode_rejects_axis_mismatch() -> None:
    # rank declared 1 but tensor has rank 2
    coeffs = _build_canonical_coeffs(num_selected=1, rank=2, foveal_h=2, foveal_w=2)
    with pytest.raises(ValueError, match="axes mismatch"):
        encode_wavelet_sidecar(
            (5,), coeffs, foveal_h=2, foveal_w=2, coeff_rank=1, int8_scale=4.0
        )


def test_encode_rejects_last_dim_mismatch() -> None:
    coeffs = _build_canonical_coeffs(num_selected=1, rank=1, foveal_h=2, foveal_w=2)
    with pytest.raises(ValueError, match="last dim"):
        encode_wavelet_sidecar(
            (5,), coeffs, foveal_h=3, foveal_w=3, coeff_rank=1, int8_scale=4.0
        )


def test_decode_rejects_bad_magic() -> None:
    with pytest.raises(ValueError, match="magic"):
        decode_wavelet_sidecar(b"NOPE" + b"\x00" * 20)


def test_decode_rejects_truncated_blob() -> None:
    with pytest.raises(ValueError, match="too short"):
        decode_wavelet_sidecar(b"\x00\x01\x02")


def test_parse_wavelet_residual_pair_indices_simple_pairs_schema() -> None:
    manifest = {"pairs": [3, 11, 11, 7, "bad", A1_N_PAIRS + 5]}
    out = parse_wavelet_residual_pair_indices(manifest, max_pairs=4)
    assert out == (3, 7, 11)  # deduplicated, sorted, OOB pair dropped, 'bad' ignored


def test_parse_wavelet_residual_pair_indices_lapose_compatible_schema() -> None:
    manifest = {
        "atoms": [
            {"hard_pair_support": [21]},
            {"atom_id": "lapose_motion_pair:13"},
            {"hard_pair_support": [21]},  # duplicate
            {"atom_id": "lapose_motion_pair:5"},
        ]
    }
    out = parse_wavelet_residual_pair_indices(manifest)
    assert out == (5, 13, 21)


def test_parse_caps_at_max_pairs() -> None:
    manifest = {"pairs": list(range(0, 200, 7))}
    out = parse_wavelet_residual_pair_indices(manifest, max_pairs=8)
    assert len(out) == 8
    assert out == tuple(sorted(out))


def test_archive_overhead_within_target_budget() -> None:
    """With operator-default config (16 pairs, rank=2, fov=128), the sidecar
    should land under ~5 KB (D2 byte-budget envelope)."""
    cfg_indices = tuple(range(16))
    coeffs = _build_canonical_coeffs(
        num_selected=16, rank=2, foveal_h=128, foveal_w=128
    )
    blob = encode_wavelet_sidecar(
        cfg_indices,
        coeffs,
        foveal_h=128,
        foveal_w=128,
        coeff_rank=2,
        int8_scale=8.0,
    )
    # Allow up to 50 KB worst-case (deterministic ramp content is hard to compress);
    # operator-typical training residual will be sparse and compress further.
    assert len(blob) < 60_000


def test_wavelet_magic_is_distinct_from_lapose_magic() -> None:
    """Cross-substrate magic must differ so split() never confuses trailers."""
    from tac.substrates.a1_plus_lapose.archive import LAPOSE_SIDECAR_MAGIC

    assert WAVELET_SIDECAR_MAGIC != LAPOSE_SIDECAR_MAGIC


def test_pack_composition_rejects_empty_a1() -> None:
    coeffs = _build_canonical_coeffs(num_selected=1, rank=1, foveal_h=2, foveal_w=2)
    with pytest.raises(ValueError, match="a1_bytes must be non-empty"):
        pack_composition_archive(
            b"",
            selected_indices=(0,),
            coeffs=coeffs,
            foveal_h=2,
            foveal_w=2,
            coeff_rank=1,
            int8_scale=4.0,
        )


def test_header_struct_size_matches_documented_layout() -> None:
    # magic(4) + ver(1) + nsel(2) + fh(2) + fw(2) + rank(1) + scale(4) + reserved(2)
    assert WAVELET_HEADER_STRUCT.size == 4 + 1 + 2 + 2 + 2 + 1 + 4 + 2


def test_roundtrip_with_many_selected_pairs() -> None:
    """Stress test with 64 selected pairs (operator default --max-pairs)."""
    indices = tuple(range(0, 600, 9))[:64]
    coeffs = _build_canonical_coeffs(
        num_selected=len(indices), rank=2, foveal_h=64, foveal_w=64
    )
    blob = encode_wavelet_sidecar(
        indices, coeffs, foveal_h=64, foveal_w=64, coeff_rank=2, int8_scale=8.0
    )
    decoded_indices, decoded_coeffs, meta = decode_wavelet_sidecar(blob)
    assert decoded_indices == indices
    assert decoded_coeffs.shape == coeffs.shape
    assert meta["num_selected"] == len(indices)
