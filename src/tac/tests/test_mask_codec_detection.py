# SPDX-License-Identifier: MIT
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tac.mask_codec import (
    decode_masks_auto,
    detect_mask_codec,
    encode_masks_auto,
)


def test_detect_mask_codec_recognizes_public_frontier_magics(tmp_path: Path) -> None:
    cases = {
        "masks.qma9": (b"QMA9" + b"\0" * 28, "qma9_range"),
        "masks.hpm1": (b"HPM1" + b"\0" * 44, "hpm1_hpac"),
        "masks.stbm1br": (b"STBM1BR\0" + b"brotli-body", "stbm1br_public"),
    }

    for name, (payload, expected_codec) in cases.items():
        path = tmp_path / name
        path.write_bytes(payload)

        assert detect_mask_codec(path) == expected_codec


def test_qma9_decode_masks_auto_roundtrips_storage_order(tmp_path: Path) -> None:
    masks = torch.tensor(
        [
            [
                [0, 1],
                [2, 3],
                [4, 0],
            ],
            [
                [1, 2],
                [3, 4],
                [0, 1],
            ],
        ],
        dtype=torch.long,
    )
    path = tmp_path / "masks.qma9"

    size = encode_masks_auto(masks, path, codec="qma9")

    assert size == path.stat().st_size
    assert detect_mask_codec(path) == "qma9_range"
    decoded = decode_masks_auto(path, codec=detect_mask_codec(path))
    assert torch.equal(decoded, masks)


def test_detectable_only_public_codecs_fail_closed_in_generic_decoder(tmp_path: Path) -> None:
    hpm1 = tmp_path / "masks.hpm1"
    hpm1.write_bytes(b"HPM1" + b"\0" * 44)
    stbm = tmp_path / "masks.stbm1br"
    stbm.write_bytes(b"STBM1BR\0" + b"body")

    with pytest.raises(ValueError, match="PR91-specific HPAC"):
        decode_masks_auto(hpm1, codec=detect_mask_codec(hpm1))
    with pytest.raises(ValueError, match="Rust bridge"):
        decode_masks_auto(stbm, codec=detect_mask_codec(stbm))
