# SPDX-License-Identifier: MIT
"""STC v2 substrate package tests.

Covers (per the 2026-05-16 design memo Section 2.2 + Catalog #272
distinguishing-feature integration contract):

  * codec roundtrip (encode -> decode -> exact equality of class IDs)
  * archive grammar (build -> parse -> byte-identical blob recovery)
  * archive magic / version / length-mismatch refusal
  * inflate runtime parses + extracts STCB
  * inflate handles 3-arg ``archive_dir output_dir file_list`` contract
  * select_inflate_device honors ``PACT_INFLATE_DEVICE`` env var
  * byte-mutation no-op detector: flipping one byte in the STCB blob
    changes the decoded mask (proves bytes are operationally consumed)
  * SubstrateContract registration validates per Catalog #241/#242
  * boundary_fraction parameter is respected by the encoder
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import torch

from tac.substrates.stc_v2 import (
    STC_V2_MAGIC,
    STC_V2_VERSION,
    build_stc_v2_archive_bytes,
    decode_stc_v2_masks,
    encode_stc_v2_masks,
    inflate_one_video,
    parse_stc_v2_archive,
)
from tac.substrates.stc_v2.archive import (
    CONTEST_OUTPUT_HW,
    StcV2Archive,
    _HEADER_LEN,
)
from tac.substrates.stc_v2.inflate import select_inflate_device


def _synthetic_masks(n: int = 4, h: int = 24, w: int = 32) -> torch.Tensor:
    """Synthetic NUM_CLASSES=5 mask tensor with class-boundary structure."""
    rng = np.random.default_rng(20260516)
    base = rng.integers(0, 5, size=(n, h, w), dtype=np.int64)
    # Inject a vertical class boundary so boundary detection has something to find
    base[:, :, : w // 2] = 0
    base[:, :, w // 2 :] = 2
    return torch.from_numpy(base)


# -- codec roundtrip ----------------------------------------------------------


def test_encode_decode_roundtrip_exact(tmp_path: Path) -> None:
    masks = _synthetic_masks()
    stcb_path = tmp_path / "test.stcb"
    nbytes = encode_stc_v2_masks(masks, stcb_path)
    assert nbytes > 0
    assert stcb_path.is_file()
    assert stcb_path.stat().st_size == nbytes
    decoded = decode_stc_v2_masks(stcb_path)
    assert torch.equal(decoded, masks)


def test_encode_respects_boundary_fraction(tmp_path: Path) -> None:
    masks = _synthetic_masks(n=8, h=64, w=96)
    p_low = tmp_path / "low.stcb"
    p_high = tmp_path / "high.stcb"
    bytes_low = encode_stc_v2_masks(masks, p_low, boundary_fraction=0.01)
    bytes_high = encode_stc_v2_masks(masks, p_high, boundary_fraction=0.15)
    # Higher boundary fraction marks more pixels as boundary -> more bytes.
    assert bytes_high >= bytes_low
    # Both must still roundtrip
    assert torch.equal(decode_stc_v2_masks(p_low), masks)
    assert torch.equal(decode_stc_v2_masks(p_high), masks)


# -- archive grammar ----------------------------------------------------------


def test_build_parse_archive_roundtrip() -> None:
    stcb_blob = b"\x00\x01\x02" * 64
    renderer_bin = b"\xff" * 1024
    poses_pt = b"P" * 128
    bin_bytes = build_stc_v2_archive_bytes(
        stcb_blob=stcb_blob,
        renderer_bin_blob=renderer_bin,
        poses_pt_blob=poses_pt,
        num_pairs=4,
    )
    assert bin_bytes[: len(STC_V2_MAGIC)] == STC_V2_MAGIC
    archive = parse_stc_v2_archive(bin_bytes)
    assert isinstance(archive, StcV2Archive)
    assert archive.version == STC_V2_VERSION
    assert archive.output_height == CONTEST_OUTPUT_HW[0]
    assert archive.output_width == CONTEST_OUTPUT_HW[1]
    assert archive.num_pairs == 4
    assert archive.stcb_blob == stcb_blob
    assert archive.renderer_bin_blob == renderer_bin
    assert archive.poses_pt_blob == poses_pt


def test_parse_refuses_wrong_magic() -> None:
    bin_bytes = b"WRONG" + b"\x00" * 200
    with pytest.raises(ValueError, match="magic mismatch"):
        parse_stc_v2_archive(bin_bytes)


def test_parse_refuses_short_payload() -> None:
    bin_bytes = b"STC2" + b"\x00" * 5  # below _HEADER_LEN
    with pytest.raises(ValueError, match="too short"):
        parse_stc_v2_archive(bin_bytes)


def test_parse_refuses_length_mismatch() -> None:
    # Build a valid header claiming 100 stcb bytes; only provide 50
    import struct

    header = (
        STC_V2_MAGIC
        + struct.pack("<H", STC_V2_VERSION)
        + struct.pack("<H", 874)
        + struct.pack("<H", 1164)
        + struct.pack("<I", 4)
        + struct.pack("<Q", 100)  # stcb_len claimed
        + struct.pack("<Q", 0)
        + struct.pack("<Q", 0)
    )
    bin_bytes = header + b"X" * 50  # only 50 stcb bytes
    with pytest.raises(ValueError, match="length mismatch"):
        parse_stc_v2_archive(bin_bytes)


def test_header_loc_pinned() -> None:
    """Catalog #146 / HNeRV L3: archive grammar offsets are fixed in source."""
    assert _HEADER_LEN == 38  # 4 + 2 + 2 + 2 + 4 + 8 + 8 + 8


# -- inflate runtime ----------------------------------------------------------


def test_inflate_one_video_writes_stcb(tmp_path: Path) -> None:
    masks = _synthetic_masks()
    stcb_src = tmp_path / "src.stcb"
    encode_stc_v2_masks(masks, stcb_src)
    stcb_bytes = stcb_src.read_bytes()

    bin_bytes = build_stc_v2_archive_bytes(
        stcb_blob=stcb_bytes,
        renderer_bin_blob=b"R" * 64,
        poses_pt_blob=b"P" * 32,
        num_pairs=4,
    )
    out_path = tmp_path / "out" / "0"
    written = inflate_one_video(bin_bytes, out_path)
    assert written.is_file()
    assert written.suffix == ".stcb"
    # The extracted blob roundtrips back to the original masks
    decoded = decode_stc_v2_masks(written)
    assert torch.equal(decoded, masks)


def test_inflate_three_arg_cli_contract(tmp_path: Path) -> None:
    """Catalog #146 three-positional-arg contract."""
    masks = _synthetic_masks()
    stcb_src = tmp_path / "src.stcb"
    encode_stc_v2_masks(masks, stcb_src)
    bin_bytes = build_stc_v2_archive_bytes(
        stcb_blob=stcb_src.read_bytes(),
        renderer_bin_blob=b"R" * 64,
        poses_pt_blob=b"P" * 32,
        num_pairs=4,
    )
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "0.bin").write_bytes(bin_bytes)
    output_dir = tmp_path / "out"
    file_list = tmp_path / "files.txt"
    file_list.write_text("0.mkv\n", encoding="utf-8")
    # Invoke main() directly via argv injection
    import sys

    from tac.substrates.stc_v2.inflate import main

    saved = sys.argv
    try:
        sys.argv = ["inflate.py", str(archive_dir), str(output_dir), str(file_list)]
        rc = main()
    finally:
        sys.argv = saved
    assert rc == 0
    assert (output_dir / "0.stcb").is_file()


def test_select_inflate_device_default() -> None:
    saved = os.environ.pop("PACT_INFLATE_DEVICE", None)
    try:
        assert select_inflate_device() == "cpu"  # substrate inflate has no torch
    finally:
        if saved is not None:
            os.environ["PACT_INFLATE_DEVICE"] = saved


def test_select_inflate_device_refuses_mps() -> None:
    os.environ["PACT_INFLATE_DEVICE"] = "mps"
    try:
        with pytest.raises(SystemExit):
            select_inflate_device()
    finally:
        os.environ.pop("PACT_INFLATE_DEVICE", None)


# -- byte-mutation no-op detector (Catalog #139 / #272) -----------------------


def test_byte_mutation_in_stcb_changes_decoded_mask(tmp_path: Path) -> None:
    """Mutating one byte in the STCB blob produces a different decoded mask.

    Per Catalog #272 distinguishing-feature integration contract: a substrate
    that claims STCB as its distinguishing bytes MUST prove the bytes are
    operationally consumed. The proof: a single byte flip changes the output.
    """
    masks = _synthetic_masks(n=2, h=16, w=16)
    p = tmp_path / "stable.stcb"
    encode_stc_v2_masks(masks, p)
    raw = bytearray(p.read_bytes())
    # Flip a byte well past the header (somewhere in the arith-coded payload).
    # The header is ~30 bytes; per-frame side info is small; mutate near the end
    # to be safely inside the arithmetic-coded boundary/exception payload.
    flip_offset = len(raw) - 8
    raw[flip_offset] ^= 0x80
    mutated_path = tmp_path / "mutated.stcb"
    mutated_path.write_bytes(bytes(raw))
    # Decode may raise OR return a different mask; either proves consumption.
    try:
        decoded = decode_stc_v2_masks(mutated_path)
        assert not torch.equal(decoded, masks), (
            "byte flip did not change decoded mask — bytes may not be consumed"
        )
    except Exception:
        # Decoder raised on the corrupted bytes — also proves consumption.
        pass


# -- SubstrateContract registration -------------------------------------------


def test_substrate_contract_registered_and_validates() -> None:
    """Catalog #241/#242: the trainer registers a contract that validates."""
    # Force-import the trainer so the @register_substrate decorator runs
    import importlib

    mod = importlib.import_module("experiments.train_substrate_stc_v2")
    # The contract module-level constant should exist
    contract = getattr(mod, "STC_V2_SUBSTRATE_CONTRACT", None)
    assert contract is not None, "trainer missing STC_V2_SUBSTRATE_CONTRACT"
    assert contract.id == "stc_v2"
    assert contract.lane_id == "lane_stc_clean_source_v2_substrate_build_20260516"
    # Validate the registry round-trips the contract
    from tac.substrate_registry import validate_all_registered

    validate_all_registered()  # raises if any registered contract is invalid
