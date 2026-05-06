"""Research-grade custody checks for the PARADIGM-gamma JCSP orchestrator."""
from __future__ import annotations

import numpy as np
import pytest

from tac.balle_hyperprior_codec import (
    BalleHyperpriorCodec,
    HyperDecoder,
    HyperEncoder,
)
from tac.joint_codec_stack_orchestrator import (
    KIND_ARITHMETIC_STATIC,
    KIND_BALLE_HYPERPRIOR,
    StreamSource,
    run_joint_codec_stack,
    run_sequential_codec_stack,
    unpack_jcsp_container,
)


def _balle_codec() -> BalleHyperpriorCodec:
    encoder = HyperEncoder(block_size=64, z_dim=4, hidden_dim=8, seed=2026)
    decoder = HyperDecoder(z_dim=4, hidden_dim=8, seed=2026)
    return BalleHyperpriorCodec(
        block_size=64,
        z_dim=4,
        hyper_encoder=encoder,
        hyper_decoder=decoder,
    )


def test_balle_static_wins_records_actual_arithmetic_codec_kind() -> None:
    rng = np.random.default_rng(42)
    qints = rng.integers(-7, 8, size=128, dtype=np.int8)
    stream = StreamSource(
        name="renderer_qints",
        qints=qints,
        num_symbols=15,
        offset=7,
        codec_kind=KIND_BALLE_HYPERPRIOR,
        balle_codec=_balle_codec(),
        score_per_byte_marginal=1e-6,
    )

    result = run_joint_codec_stack(
        streams=[stream],
        byte_budget=10_000,
        admm_max_iters=2,
    )
    parsed = unpack_jcsp_container(result.container_bytes)

    assert result.streams[0].codec_kind == KIND_ARITHMETIC_STATIC
    assert parsed["streams"][0]["codec_kind"] == KIND_ARITHMETIC_STATIC
    assert parsed["streams"][0]["payload"][:4] == b"AQv1"
    assert result.total_bytes == len(result.container_bytes)


def test_jcsp_budget_is_enforced_on_full_container_bytes() -> None:
    qints = np.zeros(64, dtype=np.int8)
    stream = StreamSource(
        name="pose_qints",
        qints=qints,
        num_symbols=15,
        offset=7,
        codec_kind=KIND_ARITHMETIC_STATIC,
        score_per_byte_marginal=1e-6,
    )
    baseline = run_joint_codec_stack(
        streams=[stream],
        byte_budget=10_000,
        admm_max_iters=2,
    )

    with pytest.raises(ValueError, match="JCSP container bytes"):
        run_joint_codec_stack(
            streams=[stream],
            byte_budget=baseline.total_bytes - 1,
            admm_max_iters=2,
        )


def test_unpack_jcsp_container_rejects_trailing_bytes() -> None:
    qints = np.array([0, 1, -1, 2], dtype=np.int8)
    stream = StreamSource(
        name="tiny",
        qints=qints,
        num_symbols=15,
        offset=7,
        codec_kind=KIND_ARITHMETIC_STATIC,
        score_per_byte_marginal=1e-6,
    )
    result = run_sequential_codec_stack(streams=[stream])

    with pytest.raises(ValueError, match="trailing bytes"):
        unpack_jcsp_container(result.container_bytes + b"x")
