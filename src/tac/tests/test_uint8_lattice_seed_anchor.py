from __future__ import annotations

import struct
from collections.abc import Sequence

import pytest

from tac.optimization.uint8_lattice_profile import (
    LatticeProfileError,
    PoseFilterDecision,
    ProfileStatus,
    SignedResidualCostModel,
    decode_candidate_stream,
    encode_candidate_stream,
    profile_cache_key,
    profile_integer_block,
)


class _CountingCost:
    identity = "test.counting_cost.u0.v1"

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def cost_bits(self, candidate: tuple[int, ...]) -> int:
        self.calls.append(candidate)
        return 8 * candidate[0]


class _CountingPose:
    identity = "test.counting_pose.v1"

    def __init__(
        self,
        *,
        rejected: Sequence[tuple[int, ...]] = (),
        error_at: tuple[int, ...] | None = None,
    ) -> None:
        self.rejected = frozenset(rejected)
        self.error_at = error_at
        self.calls: list[tuple[int, ...]] = []

    def evaluate(self, candidate: tuple[int, ...]) -> PoseFilterDecision:
        self.calls.append(candidate)
        if candidate == self.error_at:
            raise RuntimeError("synthetic pose failure")
        return PoseFilterDecision(
            feasible=candidate not in self.rejected,
            additional_cost_bits=3,
            diagnostic={"candidate": list(candidate)},
        )


def test_seed_is_a_bounded_certified_anchor_without_upper_double_count() -> None:
    seed = (4, 6)
    pose = _CountingPose()
    cost = _CountingCost()
    result = profile_integer_block(
        (1, 1),
        2,
        10,
        seed_candidate=seed,
        pose_plugin=pose,
        cost_model=cost,
        max_nodes=1,
    )

    assert result.status is ProfileStatus.BOUNDED_NODE_CAP
    assert result.exhaustive is False
    assert result.exact_cardinality is None
    assert result.cardinality_lower_bound == 1
    # There are exactly eleven affine candidates.  A seed that remained in the
    # unresolved subtree without subtraction would incorrectly report twelve.
    assert result.cardinality_upper_bound == 11
    assert result.affine_feasible_seen == 1
    assert result.selected_candidate == seed
    assert result.selected_cost_bits == 8 * seed[0] + 3
    assert result.selection_globally_exact is False
    assert pose.calls == [seed]
    assert cost.calls == [seed]


def test_seeded_and_unseeded_exhaustion_have_identical_global_result() -> None:
    seed = (7, 3)
    unseeded_pose = _CountingPose()
    unseeded_cost = _CountingCost()
    unseeded = profile_integer_block(
        (1, 1),
        2,
        10,
        pose_plugin=unseeded_pose,
        cost_model=unseeded_cost,
        max_nodes=1_000,
    )
    seeded_pose = _CountingPose()
    seeded_cost = _CountingCost()
    seeded = profile_integer_block(
        (1, 1),
        2,
        10,
        seed_candidate=seed,
        pose_plugin=seeded_pose,
        cost_model=seeded_cost,
        max_nodes=1_000,
    )

    assert seeded == unseeded
    assert seeded.status is ProfileStatus.EXACT
    assert seeded.exact_cardinality == 11
    assert seeded.selected_candidate == (0, 10)
    assert seeded.selection_globally_exact is True
    assert len(seeded_pose.calls) == 11
    assert seeded_pose.calls.count(seed) == 1
    assert len(seeded_cost.calls) == 11
    assert seeded_cost.calls.count(seed) == 1


@pytest.mark.parametrize(
    ("coefficients", "target", "seed", "message"),
    [
        ((1, 1), 10, (10,), "arity"),
        ((1, 1), 10, (True, 9), "integer"),
        ((1, 1), 10, (1.0, 9), "integer"),
        ((1, 1), 10, (-1, 11), "at least 0"),
        ((1, 1), 256, (256, 0), "uint8"),
        ((1, 1), 10, (3, 8), "exact integer equation"),
    ],
)
def test_invalid_seed_refuses_before_search(
    coefficients: tuple[int, ...],
    target: int,
    seed: tuple[object, ...],
    message: str,
) -> None:
    with pytest.raises(LatticeProfileError, match=message):
        profile_integer_block(
            coefficients,
            sum(coefficients),
            target,
            seed_candidate=seed,  # type: ignore[arg-type]
            max_nodes=1,
        )


def test_pose_rejected_seed_is_resolved_once_but_does_not_claim_infeasibility() -> None:
    seed = (4, 6)
    pose = _CountingPose(rejected=(seed,))
    cost = _CountingCost()
    result = profile_integer_block(
        (1, 1),
        2,
        10,
        seed_candidate=seed,
        pose_plugin=pose,
        cost_model=cost,
        max_nodes=1,
    )

    assert result.status is ProfileStatus.BOUNDED_NODE_CAP
    assert result.cardinality_lower_bound == 0
    assert result.cardinality_upper_bound == 10
    assert result.proved_infeasible is False
    assert result.pose_rejected_seen == 1
    assert result.selected_candidate is None
    assert pose.calls == [seed]
    assert cost.calls == []


def test_pose_rejected_seed_has_exhaustive_seeded_unseeded_parity() -> None:
    seed = (4, 6)
    unseeded = profile_integer_block(
        (1, 1),
        2,
        10,
        pose_plugin=_CountingPose(rejected=(seed,)),
        cost_model=_CountingCost(),
        max_nodes=1_000,
    )
    seeded_pose = _CountingPose(rejected=(seed,))
    seeded_cost = _CountingCost()
    seeded = profile_integer_block(
        (1, 1),
        2,
        10,
        seed_candidate=seed,
        pose_plugin=seeded_pose,
        cost_model=seeded_cost,
        max_nodes=1_000,
    )

    assert seeded == unseeded
    assert seeded.status is ProfileStatus.EXACT
    assert seeded.exact_cardinality == 10
    assert seeded.pose_rejected_seen == 1
    assert seeded_pose.calls.count(seed) == 1
    assert seed not in seeded_cost.calls


def test_pose_error_on_seed_is_unknown_and_not_retried() -> None:
    seed = (4, 6)
    pose = _CountingPose(error_at=seed)
    cost = _CountingCost()
    result = profile_integer_block(
        (1, 1),
        2,
        10,
        seed_candidate=seed,
        pose_plugin=pose,
        cost_model=cost,
        max_nodes=1_000,
    )

    assert result.status is ProfileStatus.PLUGIN_ERROR_UNKNOWN
    assert result.exhaustive is False
    assert result.exact_cardinality is None
    assert result.cardinality_lower_bound == 0
    assert result.cardinality_upper_bound >= 11
    assert result.nodes_visited == 0
    assert result.selected_candidate is None
    assert result.proved_infeasible is False
    assert result.plugin_error and "synthetic pose failure" in result.plugin_error
    assert pose.calls == [seed]
    assert cost.calls == []


def test_pose_error_after_accepted_seed_preserves_seed_and_sound_upper_bound() -> None:
    seed = (4, 6)
    failing_candidate = (0, 10)
    pose = _CountingPose(error_at=failing_candidate)
    cost = _CountingCost()
    result = profile_integer_block(
        (1, 1),
        2,
        10,
        seed_candidate=seed,
        pose_plugin=pose,
        cost_model=cost,
        max_nodes=1_000,
    )

    assert result.status is ProfileStatus.PLUGIN_ERROR_UNKNOWN
    assert result.cardinality_lower_bound == 1
    assert result.cardinality_upper_bound == 11
    assert result.selected_candidate == seed
    assert result.selected_cost_bits == 8 * seed[0] + 3
    assert result.plugin_error and "synthetic pose failure" in result.plugin_error
    assert pose.calls == [seed, failing_candidate]
    assert cost.calls == [seed]


def test_profile_cache_key_separates_complete_seed_identity() -> None:
    base = {
        "coefficients": (1, 1),
        "denominator": 2,
        "target_integer": 2,
        "selector_identity": "selector.v1",
        "pose_plugin_identity": "pose.v1",
    }
    unseeded = profile_cache_key(**base)
    first = profile_cache_key(**base, seed_candidate=(0, 2))
    second = profile_cache_key(**base, seed_candidate=(1, 1))

    assert len({unseeded, first, second}) == 3
    assert first == profile_cache_key(**base, seed_candidate=[0, 2])
    with pytest.raises(LatticeProfileError, match="exact integer equation"):
        profile_cache_key(**base, seed_candidate=(2, 2))


@pytest.mark.parametrize(
    ("model", "rows"),
    [
        (SignedResidualCostModel(), ()),
        (
            SignedResidualCostModel(),
            ((0,), (0, 255), None, (255, 1, 128), (1, 2, 3, 4)),
        ),
        (
            SignedResidualCostModel(predictor=0),
            ((0, 1), None, (254, 255)),
        ),
        (
            SignedResidualCostModel(predictor=(0, 128, 255, 1)),
            ((0, 128, 255, 1), None, (255, 0, 1, 254)),
        ),
    ],
)
def test_candidate_stream_encode_decode_roundtrip(
    model: SignedResidualCostModel,
    rows: tuple[tuple[int, ...] | None, ...],
) -> None:
    encoded = encode_candidate_stream(rows, cost_model=model)
    assert decode_candidate_stream(encoded, cost_model=model) == rows
    assert decode_candidate_stream(bytearray(encoded), cost_model=model) == rows
    assert decode_candidate_stream(memoryview(encoded), cost_model=model) == rows


def _stream(row_count: int, tail: bytes) -> bytes:
    return b"U8RDS1" + struct.pack("<I", row_count) + tail


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "header is truncated"),
        (b"BADHDR" + struct.pack("<I", 0), "header mismatch"),
        (_stream(1, b""), "row count exceeds"),
        (_stream(0, b"\x00"), "trailing bytes"),
        (_stream(1, b"\x05"), "invalid arity"),
        (_stream(1, b"\x01\x80"), "ULEB128 is unterminated"),
        (_stream(1, b"\x01\x81\x00\x00"), "ULEB128 is noncanonical"),
        (_stream(1, b"\x01\x00"), "payload length is impossible"),
        (_stream(1, b"\x01\x03\x00\x00\x00"), "payload length is impossible"),
        (_stream(1, b"\x01\x02\x00"), "payload is truncated"),
        (_stream(1, b"\x01\x01\x80"), "ULEB128 is unterminated"),
        (_stream(1, b"\x01\x02\x80\x00"), "ULEB128 is noncanonical"),
        (_stream(1, b"\x02\x03\x80\x80\x00"), "ULEB128 exceeds"),
        (_stream(1, b"\x01\x02\x00\x00"), "trailing payload bytes"),
        (_stream(1, b"\x01\x02\x80\x02"), "outside uint8"),
        (encode_candidate_stream(((1, 2),)) + b"\x00", "trailing bytes"),
    ],
)
def test_candidate_stream_decoder_refuses_malformed_payloads(
    payload: bytes,
    message: str,
) -> None:
    with pytest.raises(LatticeProfileError, match=message):
        decode_candidate_stream(payload)


def test_candidate_stream_decoder_refuses_non_bytes_and_predictor_arity_drift() -> None:
    with pytest.raises(LatticeProfileError, match="bytes-like"):
        decode_candidate_stream([0, 1, 2])  # type: ignore[arg-type]

    encoded = encode_candidate_stream(((1, 2),))
    with pytest.raises(LatticeProfileError, match="predictor arity"):
        decode_candidate_stream(
            encoded,
            cost_model=SignedResidualCostModel(predictor=(128,)),
        )
