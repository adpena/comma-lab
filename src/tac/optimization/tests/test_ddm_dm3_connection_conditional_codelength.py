from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.ddm_dm1_solved_value_pricing import (
    SolvedValueRecord,
    support_sha256,
)
from tac.optimization.ddm_dm3_connection_conditional_codelength import (
    HISTORY_FAMILIES,
    ConnectionCodelengthError,
    ProgramFit,
    SolvedSupport,
    SupportPopulation,
    decode_history_packet,
    encode_history_packet,
    fit_programs,
    price_history,
)
from tac.optimization.ddm_min_description_contract import LayerHome, StreamType


def _support_record(
    pair_id: int,
    support: np.ndarray,
    *,
    cell: bool = False,
    shift: int = 0,
) -> SolvedSupport:
    winners = bytes((index + shift) % 3 for index in range(len(support)))
    relations = bytes((index + shift + 1) % 3 for index in range(len(support)))
    record = SolvedValueRecord(
        pair_id=pair_id,
        bucket_id=(
            "road_lane__cell__static_in_image"
            if cell
            else "road_lane__boundary__static_in_image"
        ),
        class_left=0,
        class_right=1,
        stream_type=StreamType.FIBER if cell else StreamType.SKELETON,
        layer_home=LayerHome.L4_SCORER_FEATURE,
        support_sha256=support_sha256(support),
        winners=winners,
        margin_relations=relations,
        flat_indices=() if cell else tuple(int(value) for value in support),
    )
    return SolvedSupport(
        record=record,
        support=support,
        raw=record.encode(),
        solved_summary={},
        roundtrip_max_abs=0.0,
    )


@pytest.mark.parametrize(
    ("family", "state"),
    [
        ("identity", b""),
        ("xi_advected", b"\x01\x00\x00\x00"),
        (
            "affine_tracked",
            (
                b"\x00\x00\x01\x00"
                b"\x00\x00\x00\x00"
                b"\x00\x00\x00\x00"
                b"\x00\x00\x00\x00"
                b"\x00\x00\x01\x00"
                b"\x00\x00\x00\x00"
            ),
        ),
    ],
)
@pytest.mark.parametrize("cell", [False, True])
def test_history_packet_exact_roundtrip(
    family: str,
    state: bytes,
    cell: bool,
) -> None:
    previous_support = np.asarray([0, 1, 513, 1026, 196_607], dtype=np.uint32)
    target_support = np.asarray([1, 2, 514, 1027, 196_607], dtype=np.uint32)
    previous = _support_record(10, previous_support, cell=cell)
    target = _support_record(11, target_support, cell=cell, shift=1)
    packet = encode_history_packet(
        target.raw,
        previous.record,
        previous.support,
        target.support,
        family=family,
        state=state,
        codec="zlib9",
    )
    decoded = decode_history_packet(
        packet,
        previous.record,
        previous.support,
        target.support,
    )
    assert decoded["target_raw"] == target.raw
    assert decoded["family"] == family
    corrupt = bytearray(packet)
    corrupt[-1] ^= 1
    with pytest.raises(ConnectionCodelengthError):
        decode_history_packet(
            bytes(corrupt),
            previous.record,
            previous.support,
            target.support,
        )


def test_fit_programs_excludes_the_heldout_transition() -> None:
    supports = {
        0: np.asarray([0, 512, 1024], dtype=np.uint32),
        1: np.asarray([1, 513, 1025], dtype=np.uint32),
        2: np.asarray([2, 514, 1026], dtype=np.uint32),
        3: np.asarray([100, 900, 1700], dtype=np.uint32),
    }
    population = SupportPopulation(
        bucket_id="road_lane__boundary__static_in_image",
        array_key="fixture",
        supports=supports,
        transitions=((0, 1), (1, 2), (2, 3)),
        holdout=(2, 3),
    )
    fits = {fit.family: fit for fit in fit_programs(population)}
    assert fits["identity"].state == b""
    assert fits["xi_advected"].diagnostics["translation_dx"] == 1
    assert fits["xi_advected"].training_transition_count == 2
    assert fits["affine_tracked"].diagnostics["heldout_excluded"] is True


def test_one_transition_leaves_fitted_state_null_but_identity_priceable() -> None:
    left = np.asarray([0, 512, 1024], dtype=np.uint32)
    right = np.asarray([1, 513, 1025], dtype=np.uint32)
    population = SupportPopulation(
        bucket_id="road_lane__boundary__static_in_image",
        array_key="fixture",
        supports={5: left, 6: right},
        transitions=((5, 6),),
        holdout=(5, 6),
    )
    fits = fit_programs(population)
    assert fits[0].state == b""
    assert all(fit.state is None for fit in fits[1:])
    previous = _support_record(5, left)
    target = _support_record(6, right, shift=1)
    prices, winner = price_history(target, previous, fits)
    assert winner == "identity"
    assert prices["identity"]["exact_counted_bytes"] > 0
    assert prices["xi_advected"]["exact_counted_bytes"] is None


def test_price_history_tie_break_stays_in_sealed_family_order() -> None:
    support = np.asarray([0, 512, 1024], dtype=np.uint32)
    previous = _support_record(8, support)
    target = _support_record(9, support)
    fits = tuple(
        ProgramFit(
            family=family,
            state=b"" if family == "identity" else None,
            training_transition_count=0,
            training_correspondence_count=0,
            status="fixture",
            diagnostics={},
        )
        for family in HISTORY_FAMILIES
    )
    _, winner = price_history(target, previous, fits)
    assert winner == "identity"
