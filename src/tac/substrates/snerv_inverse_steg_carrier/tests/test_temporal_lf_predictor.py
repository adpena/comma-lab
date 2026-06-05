# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier.temporal_lf_predictor import (
    SnervTemporalLfPredictorError,
    build_temporal_lf_predictor_receiver_proof,
    decode_temporal_lf_predictor_payload,
    encode_temporal_lf_predictor_payload,
    inspect_temporal_lf_predictor_payload,
)


def test_temporal_lf_predictor_payload_roundtrips_lf_planes() -> None:
    frames = _frames()

    packet = encode_temporal_lf_predictor_payload(
        frames,
        pair_indices=[2, 5],
        lf_downsample=4,
        correction_quant_step=0.5,
    )
    decoded = decode_temporal_lf_predictor_payload(packet.packet)
    header, _compressed = inspect_temporal_lf_predictor_payload(packet.packet)

    assert header["correction_stream_byte_charged"] is True
    assert header["first_lf_anchor_bytes"] > 0
    assert header["correction_stream_raw_bytes"] > 0
    assert decoded.shape == (2, 2, 3, 4, 6)
    expected = frames.reshape(2, 2, 3, 4, 4, 6, 4).mean(
        axis=(4, 6),
        dtype=np.float32,
    )
    assert float(np.max(np.abs(decoded - expected))) <= 0.25 + 1.0e-4


def test_temporal_lf_predictor_receiver_proof_closes_implementation_blockers() -> None:
    proof, payload = build_temporal_lf_predictor_receiver_proof(
        _frames(),
        pair_indices=[0, 1],
        packet_path="/ssd/source.snar",
        source_packet_sha256="a" * 64,
        payload_path="/ssd/temporal_lf.stlp",
        lf_downsample=4,
        correction_quant_step=1.0,
    )

    assert payload.startswith(b"STLP1")
    assert proof["schema"] == "snerv_temporal_lf_predictor_receiver_proof.v1"
    assert proof["receiver_payload_implemented"] is True
    assert proof["receiver_decode_proven"] is True
    assert proof["numpy_receiver_decode"] is True
    assert proof["correction_stream_byte_charged"] is True
    assert proof["section_native_byte_telemetry_present"] is True
    assert proof["closed_campaign_blockers"] == [
        "snerv_temporal_lf_predictor_gate_not_implemented",
        "snerv_temporal_lf_predictor_correction_stream_not_byte_charged",
    ]
    assert proof["blockers"] == ["snerv_temporal_lf_predictor_payload_false_authority"]
    assert proof["score_claim"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False


def test_temporal_lf_predictor_rejects_bad_magic() -> None:
    with pytest.raises(SnervTemporalLfPredictorError, match="bad temporal LF"):
        decode_temporal_lf_predictor_payload(b"bad")


def _frames() -> np.ndarray:
    base = np.arange(2 * 2 * 3 * 16 * 24, dtype=np.float32).reshape(
        2,
        2,
        3,
        16,
        24,
    )
    return np.asarray((base % 251) + 2.0, dtype=np.float32)
