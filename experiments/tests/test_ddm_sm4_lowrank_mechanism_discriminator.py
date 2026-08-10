from __future__ import annotations

from collections import OrderedDict

import pytest
import torch

from experiments import ddm_sm4_lowrank_mechanism_discriminator as sm4
from experiments.ddm_sm4_runtime import sm4r_receiver


def fake_state() -> OrderedDict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    state: OrderedDict[str, torch.Tensor] = OrderedDict()
    state["vector"] = torch.randn(8, generator=generator)
    state["other.weight"] = torch.randn(8, 8, 1, 1, generator=generator)
    for name in sm4.TARGET_NAMES:
        state[name] = torch.randn(8, 8, 1, 1, generator=generator)
    return state


@pytest.mark.parametrize("bits", sm4.GRID_BITS)
@pytest.mark.parametrize("centered", sm4.GRID_CENTERED)
def test_pack_and_independent_receiver_agree(bits: int, centered: bool) -> None:
    state = fake_state()
    bases = {
        (name, option): sm4.canonical_basis(state[name], option)
        for name in sm4.TARGET_NAMES
        for option in sm4.GRID_CENTERED
    }
    payload, expected, _ = sm4.pack_candidate(
        state,
        bases,
        rank=4,
        bits=bits,
        centered=centered,
    )
    decoded = sm4r_receiver.unpack_sm4r_or_none(payload, state)
    assert decoded is not None
    sm4.assert_state_equal(expected, decoded)


def test_receiver_rejects_truncation_unknown_flags_and_trailing_bytes() -> None:
    state = fake_state()
    bases = {
        (name, option): sm4.canonical_basis(state[name], option)
        for name in sm4.TARGET_NAMES
        for option in sm4.GRID_CENTERED
    }
    payload, _, _ = sm4.pack_candidate(state, bases, rank=4, bits=8, centered=True)
    with pytest.raises(sm4r_receiver.SM4RFormatError, match="truncated"):
        sm4r_receiver.unpack_sm4r_or_none(payload[:-1], state)
    malformed = bytearray(payload)
    malformed[7] |= 0x80
    with pytest.raises(sm4r_receiver.SM4RFormatError, match="unknown SM4R flags"):
        sm4r_receiver.unpack_sm4r_or_none(bytes(malformed), state)
    with pytest.raises(sm4r_receiver.SM4RFormatError, match="trailing bytes"):
        sm4r_receiver.unpack_sm4r_or_none(payload + b"x", state)


def test_magic_absence_is_the_only_none_case() -> None:
    state = fake_state()
    assert sm4r_receiver.unpack_sm4r_or_none(b"legacy", state) is None
    with pytest.raises(sm4r_receiver.SM4RFormatError, match="truncated SM4R header"):
        sm4r_receiver.unpack_sm4r_or_none(b"SM4R", state)


def test_inflate_transformation_adds_sm4_before_legacy_dispatch() -> None:
    transformed = sm4.transformed_inflate_source().decode()
    assert "SM4R_MAGIC" in transformed
    assert "semantic_state = unpack_sm4r_or_none" in transformed
    assert transformed.index("unpack_sm4r_or_none(semantic_blob") < transformed.index(
        "unpack_sm3r_or_none(semantic_blob"
    )
    compile(transformed, "inflate.py", "exec")


def test_equal_byte_law_pair_is_close_on_real_geometry() -> None:
    rows = columns = 128

    def wire(rank: int, bits: int) -> int:
        return (
            (rows * rank * bits + 7) // 8
            + (rank * columns * bits + 7) // 8
            + (rows + rank) * 2
        )

    assert wire(32, 4) == 4_416
    assert wire(16, 8) == 4_384
