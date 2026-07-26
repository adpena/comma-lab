# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tac.optimization.s2_partition_seed import PartitionEventSeed, encode_partition_seed
from tac.witness_dsl.predictor_bound_residual import (
    PredictorBoundResidualError,
    apply_predictor_bound_partition_residual,
    build_predictor_bound_partition_residual,
    decode_predictor_bound_partition_residual,
    packet_accounting,
)

CONTRACT = "test-v9-five-class-predictor.v1"
RENDERER_SHA = hashlib.sha256(b"renderer-source").hexdigest()


def _fixture() -> tuple[bytes, np.ndarray, np.ndarray]:
    program = b"counted-v9-program"
    predictor = np.array(
        [
            [[0, 0, 2, 2], [0, 1, 2, 4], [4, 4, 4, 4]],
            [[0, 0, 2, 2], [0, 1, 3, 4], [4, 4, 4, 4]],
        ],
        dtype=np.uint8,
    )
    target = predictor.copy()
    target[0, 0, 1] = 1
    target[1, 1, 2] = 2
    return program, predictor, target


def _packet() -> bytes:
    program, predictor, target = _fixture()
    return build_predictor_bound_partition_residual(
        predictor_program=program,
        predictor_contract_id=CONTRACT,
        predictor_renderer_sha256=RENDERER_SHA,
        predictor_labels=predictor,
        target_labels=target,
    )


def test_packet_is_deterministic_strict_and_recovers_target() -> None:
    program, predictor, target = _fixture()
    first = _packet()
    second = _packet()
    assert first == second
    decoded = decode_predictor_bound_partition_residual(first)
    assert len(decoded.seed.events) == 2
    recovered = apply_predictor_bound_partition_residual(
        first,
        predictor_program=program,
        predictor_contract_id=CONTRACT,
        predictor_renderer_sha256=RENDERER_SHA,
        predictor_labels=predictor,
    )
    assert np.array_equal(recovered, target)
    accounting = packet_accounting(first)
    assert accounting["packet_bytes"] == len(first)
    assert accounting["event_count"] == 2
    assert accounting["target_table_bytes"] == 0
    assert accounting["score_claim"] is False


def test_predictor_program_swap_is_refused() -> None:
    _program, predictor, _target = _fixture()
    with pytest.raises(PredictorBoundResidualError, match="program identity"):
        apply_predictor_bound_partition_residual(
            _packet(),
            predictor_program=b"different-program",
            predictor_contract_id=CONTRACT,
            predictor_renderer_sha256=RENDERER_SHA,
            predictor_labels=predictor,
        )


def test_predictor_semantic_swap_is_refused_even_with_same_program() -> None:
    program, predictor, _target = _fixture()
    predictor = predictor.copy()
    predictor[0, 2, 0] = 3
    with pytest.raises(PredictorBoundResidualError, match="semantic stream"):
        apply_predictor_bound_partition_residual(
            _packet(),
            predictor_program=program,
            predictor_contract_id=CONTRACT,
            predictor_renderer_sha256=RENDERER_SHA,
            predictor_labels=predictor,
        )


@pytest.mark.parametrize(
    ("contract", "renderer", "message"),
    [
        ("other-contract.v1", RENDERER_SHA, "contract"),
        (CONTRACT, hashlib.sha256(b"other-renderer").hexdigest(), "source"),
    ],
)
def test_predictor_interpreter_swap_is_refused(contract: str, renderer: str, message: str) -> None:
    program, predictor, _target = _fixture()
    with pytest.raises(PredictorBoundResidualError, match=message):
        apply_predictor_bound_partition_residual(
            _packet(),
            predictor_program=program,
            predictor_contract_id=contract,
            predictor_renderer_sha256=renderer,
            predictor_labels=predictor,
        )


def test_packet_refuses_corruption_and_trailing_bytes() -> None:
    packet = _packet()
    corrupted = bytearray(packet)
    corrupted[-5] ^= 1
    with pytest.raises(PredictorBoundResidualError, match="CRC"):
        decode_predictor_bound_partition_residual(bytes(corrupted))
    with pytest.raises(PredictorBoundResidualError, match="trailing"):
        decode_predictor_bound_partition_residual(packet + b"x")


def test_unbound_s2_packet_is_not_accepted_as_bound_residual() -> None:
    raw_s2 = encode_partition_seed(
        PartitionEventSeed(
            n_pairs=1,
            height=2,
            width=2,
            semantic_class_ids=(0, 1, 2, 3, 4),
            events=(),
        )
    )
    with pytest.raises(PredictorBoundResidualError, match="magic/version"):
        decode_predictor_bound_partition_residual(raw_s2)


def test_zero_event_identity_residual_is_valid_and_still_bound() -> None:
    program, predictor, _target = _fixture()
    packet = build_predictor_bound_partition_residual(
        predictor_program=program,
        predictor_contract_id=CONTRACT,
        predictor_renderer_sha256=RENDERER_SHA,
        predictor_labels=predictor,
        target_labels=predictor,
    )
    decoded = decode_predictor_bound_partition_residual(packet)
    assert decoded.header["event_count"] == 0
    recovered = apply_predictor_bound_partition_residual(
        packet,
        predictor_program=program,
        predictor_contract_id=CONTRACT,
        predictor_renderer_sha256=RENDERER_SHA,
        predictor_labels=predictor,
    )
    assert np.array_equal(recovered, predictor)


@pytest.mark.parametrize(
    ("program", "contract", "renderer", "message"),
    [
        (b"", CONTRACT, RENDERER_SHA, "non-empty"),
        (b"x", "", RENDERER_SHA, "non-empty"),
        (b"x", CONTRACT, "bad", "SHA-256"),
    ],
)
def test_builder_rejects_unbound_identity(program: bytes, contract: str, renderer: str, message: str) -> None:
    _program, predictor, target = _fixture()
    with pytest.raises(PredictorBoundResidualError, match=message):
        build_predictor_bound_partition_residual(
            predictor_program=program,
            predictor_contract_id=contract,
            predictor_renderer_sha256=renderer,
            predictor_labels=predictor,
            target_labels=target,
        )
