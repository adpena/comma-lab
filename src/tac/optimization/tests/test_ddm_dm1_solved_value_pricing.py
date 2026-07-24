from __future__ import annotations

import json

import numpy as np
import pytest

from tac.optimization.ddm_dm1_solved_value_pricing import (
    CODECS,
    SolvedValuePricingError,
    SolvedValueRecord,
    _validate_false_authority,
    decode_codec,
    decode_context_arithmetic,
    decode_joint_raw,
    encode_codec,
    encode_context_arithmetic,
    encode_joint_raw,
    price_raw,
    support_sha256,
    typed_home,
)
from tac.optimization.ddm_min_description_contract import LayerHome, StreamType
from tac.optimization.solve_diff_operator_mining import AXIS, POINTER


def _boundary_record() -> tuple[SolvedValueRecord, np.ndarray]:
    support = np.asarray([0, 7, 512, 513, 196_607], dtype=np.uint32)
    return (
        SolvedValueRecord(
            pair_id=55,
            bucket_id="lane_undrivable__boundary__static_in_image",
            class_left=1,
            class_right=2,
            stream_type=StreamType.SKELETON,
            layer_home=LayerHome.L4_SCORER_FEATURE,
            support_sha256=support_sha256(support),
            winners=bytes([0, 1, 2, 0, 1]),
            margin_relations=bytes([0, 1, 0, 1, 1]),
            flat_indices=tuple(int(value) for value in support),
        ),
        support,
    )


def _cell_record() -> tuple[SolvedValueRecord, np.ndarray]:
    support = np.asarray([2, 4, 8, 16], dtype=np.uint32)
    return (
        SolvedValueRecord(
            pair_id=54,
            bucket_id="road_mycar__cell__static_in_image",
            class_left=0,
            class_right=4,
            stream_type=StreamType.FIBER,
            layer_home=LayerHome.L4_SCORER_FEATURE,
            support_sha256=support_sha256(support),
            winners=bytes([0, 0, 1, 2]),
            margin_relations=bytes([0, 0, 1, 1]),
        ),
        support,
    )


def test_boundary_semantic_record_exact_roundtrip() -> None:
    record, _ = _boundary_record()
    raw = record.encode()
    assert SolvedValueRecord.decode(raw) == record
    assert record.flat_indices[-1] == 196_607


def test_cell_requires_sha_bound_external_support() -> None:
    record, support = _cell_record()
    raw = record.encode()
    with pytest.raises(SolvedValuePricingError, match="requires SHA-verified support"):
        SolvedValueRecord.decode(raw)
    assert SolvedValueRecord.decode(raw, external_cell_support=support) == record
    with pytest.raises(SolvedValuePricingError, match="SHA-256 mismatch"):
        SolvedValueRecord.decode(
            raw,
            external_cell_support=np.asarray([2, 4, 8, 17], dtype=np.uint32),
        )


@pytest.mark.parametrize("codec", CODECS)
def test_each_codec_is_deterministic_and_exact(codec: str) -> None:
    raw = (_boundary_record()[0].encode() + b"deterministic-context") * 4
    first = encode_codec(raw, codec)
    second = encode_codec(raw, codec)
    assert first == second
    assert decode_codec(first) == (codec, raw)
    corrupt = bytearray(first)
    corrupt[-1] ^= 1
    with pytest.raises(SolvedValuePricingError):
        decode_codec(bytes(corrupt))


def test_context_arithmetic_direct_roundtrip() -> None:
    raw = bytes(range(256)) + b"a" * 500 + bytes(range(255, -1, -1))
    encoded = encode_context_arithmetic(raw)
    assert decode_context_arithmetic(encoded, len(raw)) == raw


def test_joint_25_row_shared_context_roundtrip_and_price() -> None:
    raws = tuple(
        json.dumps({"row": index, "bucket": f"same-{index % 3}"}, sort_keys=True).encode()
        for index in range(25)
    )
    joint = encode_joint_raw(raws)
    assert decode_joint_raw(joint) == raws
    prices, winner = price_raw(joint)
    assert winner in CODECS
    assert all(prices[codec]["parseback_exact"] for codec in CODECS)
    with pytest.raises(SolvedValuePricingError, match="exactly 25"):
        encode_joint_raw(raws[:-1])


def test_deepest_rehome_corrects_boundary_and_confirms_cell() -> None:
    boundary = typed_home("boundary", 101)
    cell = typed_home("cell", 17)
    assert (boundary.type, boundary.layer_home, boundary.counted_bytes) == (
        StreamType.SKELETON,
        LayerHome.L4_SCORER_FEATURE,
        101,
    )
    assert (cell.type, cell.layer_home, cell.counted_bytes) == (
        StreamType.FIBER,
        LayerHome.L4_SCORER_FEATURE,
        17,
    )


def test_false_authority_contract_fails_closed() -> None:
    valid = {
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "archive_emitted": False,
        "pointer_moved": False,
        "pointer": POINTER,
        "evidence_axis": AXIS,
    }
    _validate_false_authority(valid)
    for key in (
        "research_only",
        "execution_allowed",
        "score_claim",
        "promotion_eligible",
        "archive_emitted",
        "pointer_moved",
    ):
        invalid = dict(valid)
        invalid[key] = not invalid[key]
        with pytest.raises(SolvedValuePricingError, match=key):
            _validate_false_authority(invalid)
