# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac.substrates.snerv_inverse_steg_carrier.lf_super_resolution_tiny_anchor import (
    SnervLfSuperResolutionTinyAnchorError,
    build_lf_super_resolution_tiny_anchor_receiver_proof,
    decode_lf_super_resolution_tiny_anchor_payload,
    encode_lf_super_resolution_tiny_anchor_payload,
    inspect_lf_super_resolution_tiny_anchor_payload,
)


def test_lf_super_resolution_tiny_anchor_payload_decodes_full_frames() -> None:
    frames = _frames()

    packet = encode_lf_super_resolution_tiny_anchor_payload(
        frames,
        pair_indices=[3, 4],
        anchor_downsample=8,
        anchor_quant_step=1.0,
    )
    decoded = decode_lf_super_resolution_tiny_anchor_payload(packet.packet)
    header, _compressed = inspect_lf_super_resolution_tiny_anchor_payload(
        packet.packet
    )

    assert header["tiny_anchor_component_deltas_present"] is True
    assert header["component_delta_scope"] == "receiver_pixel_domain_not_scorer_component"
    assert header["anchor_raw_bytes"] == 2 * 2 * 3 * 2 * 3 * 2
    assert decoded.shape == frames.shape
    assert float(np.std(decoded)) > 0.0


def test_lf_super_resolution_tiny_anchor_proof_closes_payload_delta_blockers() -> None:
    proof, payload = build_lf_super_resolution_tiny_anchor_receiver_proof(
        _frames(),
        pair_indices=[0, 1],
        packet_path="/ssd/source.snar",
        source_packet_sha256="a" * 64,
        payload_path="/ssd/tiny_anchor.slsr",
        anchor_downsample=8,
        anchor_quant_step=1.0,
    )

    assert payload.startswith(b"SLSR1")
    assert proof["schema"] == "snerv_lf_super_resolution_tiny_anchor_receiver_proof.v1"
    assert proof["receiver_payload_implemented"] is True
    assert proof["receiver_decode_proven"] is True
    assert proof["numpy_receiver_decode"] is True
    assert proof["tiny_anchor_component_deltas_present"] is True
    assert proof["component_delta_scope"] == "receiver_pixel_domain_not_scorer_component"
    assert proof["section_native_byte_telemetry_present"] is True
    assert proof["receiver_component_delta_stats"]["all_frames"]["max_abs"] > 0.0
    assert proof["closed_campaign_blockers"] == [
        "snerv_lf_super_resolution_receiver_payload_not_implemented",
        "snerv_lf_downsampled_anchor_component_deltas_missing",
    ]
    assert proof["blockers"] == [
        "snerv_lf_super_resolution_tiny_anchor_payload_false_authority"
    ]
    assert proof["score_claim"] is False
    assert proof["ready_for_exact_eval_dispatch"] is False


def test_lf_super_resolution_tiny_anchor_rejects_bad_magic() -> None:
    with pytest.raises(SnervLfSuperResolutionTinyAnchorError, match="bad tiny-anchor"):
        decode_lf_super_resolution_tiny_anchor_payload(b"bad")


def _frames() -> np.ndarray:
    base = np.arange(2 * 2 * 3 * 16 * 24, dtype=np.float32).reshape(
        2,
        2,
        3,
        16,
        24,
    )
    return np.asarray((base % 251) + 2.0, dtype=np.float32)
