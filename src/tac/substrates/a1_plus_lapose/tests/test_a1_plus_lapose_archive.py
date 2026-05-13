from __future__ import annotations

import torch

from tac.substrates.a1_plus_lapose.archive import (
    decode_lapose_sidecar,
    encode_lapose_sidecar,
    pack_composition_archive,
    split_composition_archive,
)
from tac.substrates.a1_plus_lapose.architecture import parse_lapose_atom_indices


def test_lapose_sidecar_roundtrips_quantized_residuals() -> None:
    residuals = torch.linspace(-1.0, 1.0, steps=2 * 2 * 2 * 17).reshape(2, 2, 2, 17)

    blob = encode_lapose_sidecar(
        (3, 7),
        residuals,
        foveal_h=4,
        foveal_w=5,
        residual_rank=2,
        int8_scale=8.0,
    )
    indices, decoded, meta = decode_lapose_sidecar(blob)

    expected = (residuals * 8.0).round().clamp(-128, 127).to(torch.int8).float() / 8.0
    assert indices == (3, 7)
    assert decoded.shape == residuals.shape
    assert torch.equal(decoded, expected)
    assert meta["foveal_h"] == 4
    assert meta["foveal_w"] == 5
    assert meta["residual_rank"] == 2


def test_composition_archive_preserves_a1_prefix_exactly() -> None:
    a1_bytes = b"A1_BASE_BYTES\x00\x01"
    residuals = torch.zeros(1, 2, 1, 8)

    packed = pack_composition_archive(
        a1_bytes,
        selected_indices=(11,),
        residuals=residuals,
        foveal_h=2,
        foveal_w=2,
        residual_rank=1,
        int8_scale=4.0,
    )
    prefix, sidecar = split_composition_archive(packed)

    assert prefix == a1_bytes
    indices, decoded, _meta = decode_lapose_sidecar(sidecar)
    assert indices == (11,)
    assert decoded.shape == residuals.shape


def test_split_composition_archive_is_noop_without_sidecar_magic() -> None:
    prefix, sidecar = split_composition_archive(b"plain-a1-wire")

    assert prefix == b"plain-a1-wire"
    assert sidecar == b""


def test_parse_lapose_atom_indices_deduplicates_and_caps() -> None:
    manifest = {
        "atoms": [
            {"hard_pair_support": [7]},
            {"atom_id": "lapose_motion_pair:3"},
            {"hard_pair_support": [7]},
            {"atom_id": "bad"},
            {"hard_pair_support": [9999]},
        ]
    }

    assert parse_lapose_atom_indices(manifest, max_atoms=2) == (3, 7)
