"""Tests for the retained DDM-SM3 semantic representation packers."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import ddm_sm3_semantic_representation as sm3


def tiny_state() -> OrderedDict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(7)
    return OrderedDict(
        [
            ("token_embed.weight", torch.randn((5, 8), generator=generator)),
            ("frame_embed.weight", torch.randn((12, 4), generator=generator)),
            ("coord_mix.weight", torch.randn((8, 12, 1, 1), generator=generator)),
            ("coord_mix.bias", torch.randn((8,), generator=generator)),
            ("blocks.0.pw.weight", torch.randn((8, 8, 1, 1), generator=generator)),
            ("blocks.1.film.weight", torch.randn((16, 4), generator=generator)),
            ("head.bias", torch.randn((3,), generator=generator)),
        ]
    )


@pytest.mark.parametrize("mode", [sm3.MODE_VECTOR_VQ, sm3.MODE_SCALE_VQ, sm3.MODE_BOTH_VQ])
def test_vq_payload_roundtrip_is_exact_to_packer_state(mode: int, tmp_path: Path) -> None:
    state = tiny_state()
    payload, expected, _ = sm3.pack_vq_candidate(state, mode, 4)
    (tmp_path / "candidate.bin").write_bytes(payload)
    actual = sm3.unpack_candidate(payload, state)
    sm3.assert_state_equal(expected, actual)


def test_lowrank_payload_roundtrip_is_exact_to_packer_state(tmp_path: Path) -> None:
    state = tiny_state()
    selected = {"coord_mix.weight", "blocks.0.pw.weight"}
    original = sm3.LOWRANK_NAMES
    sm3.LOWRANK_NAMES = selected
    try:
        payload, expected, details = sm3.pack_lowrank_candidate(state, 4)
        (tmp_path / "candidate.bin").write_bytes(payload)
        actual = sm3.unpack_lowrank_candidate(payload, state)
    finally:
        sm3.LOWRANK_NAMES = original
    assert details["rank"] == 4
    sm3.assert_state_equal(expected, actual)


def test_prune_payload_roundtrip_is_exact_to_packer_state(tmp_path: Path) -> None:
    state = tiny_state()
    original = sm3.PRUNE_NAMES
    sm3.PRUNE_NAMES = {"blocks.1.film.weight"}
    try:
        payload, expected, details = sm3.pack_prune_candidate(state, 50)
        (tmp_path / "candidate.bin").write_bytes(payload)
        actual = sm3.unpack_prune_candidate(payload, state)
    finally:
        sm3.PRUNE_NAMES = original
    assert details["kept_rows"]["blocks.1.film.weight"] == 8
    sm3.assert_state_equal(expected, actual)


def test_unsigned_bitpack_roundtrip(tmp_path: Path) -> None:
    source = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 0], dtype=np.uint16)
    payload = sm3.pack_unsigned_bits(source, 3)
    (tmp_path / "bits.bin").write_bytes(payload)
    actual, remaining = sm3.unpack_unsigned_bits(memoryview(payload), len(source), 3)
    assert not remaining
    assert np.array_equal(actual, source)


@pytest.mark.skipif(
    not (sm3.DEFAULT_OUT / "final_v3/SM3_RESULT.json").is_file(),
    reason="DDM-SM3 custody is not mounted",
)
def test_real_retained_receipt_has_every_payload() -> None:
    receipt = json.loads((sm3.DEFAULT_OUT / "final_v3/SM3_RESULT.json").read_text())
    assert receipt["selection_status"] == "NO_WINNER_WITHOUT_DSEG_AND_DPOSE_MEASUREMENT"
    assert len(receipt["candidates"]) == 8
    for candidate in receipt["candidates"]:
        assert candidate["checks"]["all_tensor_decode_equal_to_packer_state"]
        for record in candidate["retained"].values():
            path = Path(record["path"])
            assert path.stat().st_size == record["bytes"]
            assert sm3.sha256_file(path) == record["sha256"]
