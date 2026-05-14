# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.composition.stack_of_stacks import (
    InnerStackSpec,
    MiddleStackSpec,
    OuterStackSpec,
    compose_stack_of_stacks,
)
from tac.composition.stack_of_stacks.inflate import (
    SOS_HEADER_STRUCT,
    SOS_SIDECAR_MAGIC,
    base_arm_passthrough_bytes,
    parse_sos_trailer,
    selected_arm_bytes,
)


def _compose_two_arm_payload(per_pair_arm: tuple[int, ...]) -> bytes:
    composed, _ = compose_stack_of_stacks(
        middle_stack_spec=MiddleStackSpec(
            inner_specs=(
                InnerStackSpec(substrate_id="arm0", base_bytes=b"ARM0_BYTES"),
                InnerStackSpec(substrate_id="arm1", base_bytes=b"ARM1_BYTES"),
            )
        ),
        outer_stack_spec=OuterStackSpec(k=2, per_pair_arm=per_pair_arm),
        n_pairs=len(per_pair_arm),
    )
    return composed


def test_selected_arm_bytes_supports_base_arm_passthrough() -> None:
    parsed = parse_sos_trailer(_compose_two_arm_payload((0, 0, 0)))

    assert base_arm_passthrough_bytes(parsed) == b"ARM0_BYTES"
    assert selected_arm_bytes(parsed) == b"ARM0_BYTES"
    assert selected_arm_bytes(parsed, pair_index=2) == b"ARM0_BYTES"


def test_selected_arm_bytes_uses_per_pair_selector() -> None:
    parsed = parse_sos_trailer(_compose_two_arm_payload((0, 1, 0, 1)))

    assert selected_arm_bytes(parsed, pair_index=0) == b"ARM0_BYTES"
    assert selected_arm_bytes(parsed, pair_index=1) == b"ARM1_BYTES"
    assert selected_arm_bytes(parsed, pair_index=3) == b"ARM1_BYTES"


def test_selected_arm_bytes_refuses_implicit_passthrough_for_non_base_selector() -> None:
    parsed = parse_sos_trailer(_compose_two_arm_payload((0, 1, 0)))

    with pytest.raises(ValueError, match=r"base-arm passthrough.*pair 1 selects arm 1"):
        selected_arm_bytes(parsed)


def test_parse_sos_trailer_fails_closed_on_invalid_selector_byte() -> None:
    payload = bytearray(_compose_two_arm_payload((0, 1, 0)))
    trailer_offset = bytes(payload).rfind(SOS_SIDECAR_MAGIC)
    selector_offset = trailer_offset + SOS_HEADER_STRUCT.size
    payload[selector_offset] = 2

    with pytest.raises(ValueError, match="selector byte 2 for pair 0 out of range"):
        parse_sos_trailer(bytes(payload))


def test_selected_arm_bytes_fails_closed_on_mutated_invalid_selector() -> None:
    parsed = parse_sos_trailer(_compose_two_arm_payload((0, 1)))
    parsed["selector"] = b"\x03\x01"

    with pytest.raises(ValueError, match="selector byte 3 for pair 0 out of range"):
        selected_arm_bytes(parsed, pair_index=0)
