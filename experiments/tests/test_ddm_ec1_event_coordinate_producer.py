from __future__ import annotations

import numpy as np
import pytest

from experiments import ddm_ec1_event_coordinate_producer as ec1
from tac.payload_retention_gate import check_no_measure_and_discard_payload

REPO = ec1.REPO


def synthetic_planes() -> tuple[np.ndarray, np.ndarray]:
    source = np.zeros((ec1.N, ec1.H, ec1.W), dtype=np.uint8)
    target = source.copy()
    target[0, 10:12, 20:23] = 1
    target[1, 30, 40] = 3
    return source, target


def test_class_stream_roundtrip_and_full_delta_reconstruction(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ec1, "N", 2)
    source = np.zeros((2, ec1.H, ec1.W), dtype=np.uint8)
    target = source.copy()
    target[0, 10:12, 20:23] = 1
    target[1, 30, 40] = 3
    streams = {}
    for class_id in range(len(ec1.CLASSES)):
        payload, records, summaries = ec1.encode_class_stream("base_to_hy1", class_id, source, target)
        mode, decoded_class, decoded = ec1.decode_class_stream(payload)
        assert mode == "base_to_hy1"
        assert decoded_class == class_id
        assert len(records) == len(decoded) == 2
        streams[class_id] = payload
        if class_id in (1, 3):
            assert summaries
    restored = ec1.reconstruct("base_to_hy1", streams, source)
    assert np.array_equal(restored, target)


def test_temporal_stream_frame_zero_is_absolute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ec1, "N", 2)
    target = np.zeros((2, ec1.H, ec1.W), dtype=np.uint8)
    target[0, :, ec1.W // 2 :] = 2
    target[1] = target[0]
    target[1, 7:9, 8:12] = 4
    source = np.empty_like(target)
    source[0].fill(255)
    source[1] = target[0]
    streams = {
        class_id: ec1.encode_class_stream("hy1_temporal", class_id, source, target)[0]
        for class_id in range(len(ec1.CLASSES))
    }
    assert np.array_equal(ec1.reconstruct("hy1_temporal", streams, None), target)


def test_proposal_parseback_and_source_precondition() -> None:
    indices = np.array([11, 17, 99], dtype=np.int64)
    payload = ec1.proposal_payload(7, 0, 1, indices, ec1.EVENT_TYPE["lane_program_delta"])
    frame, source_class, target_class, event_type, restored = ec1.decode_proposal(payload)
    assert (frame, source_class, target_class) == (7, 0, 1)
    assert event_type == ec1.EVENT_TYPE["lane_program_delta"]
    assert np.array_equal(restored, indices)


def test_proposal_parser_rejects_trailing_bytes() -> None:
    payload = ec1.proposal_payload(0, 0, 1, np.array([1], dtype=np.int64), 1)
    with pytest.raises(ec1.EC1Error, match="trailing"):
        ec1.decode_proposal(payload + b"x")


def test_real_coder_roundtrips() -> None:
    records = [b"abc", b"abcabc", b"xyz"]
    raw = b"".join(records)
    for coder in ec1.CODERS:
        payload = ec1.encode_coder(coder, raw, records)
        assert ec1.decode_coder(coder, payload, len(raw)) == raw


def test_full_container_has_one_entry_per_class() -> None:
    entries = {class_id: ("brotli-q11", bytes([class_id]), 10 + class_id) for class_id in range(5)}
    payload = ec1.build_container("base_to_hy1", entries)
    assert payload.startswith(ec1.CONTAINER_MAGIC)
    assert len(payload) == ec1.CONTAINER_HEADER.size + 5 * (ec1.ENTRY_HEADER.size + 1)
    assert ec1.parse_container(payload) == ("base_to_hy1", entries)


def test_sp1_container_parseback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ec1, "N", 2)
    streams = {name: f"payload-{name}".encode() for name in ec1.SP1_STREAM_NAMES}
    payload = ec1.build_sp1_container("hy1_temporal", streams)
    assert ec1.parse_sp1_container(payload) == ("hy1_temporal", streams)
    with pytest.raises(ec1.EC1Error, match="trailing"):
        ec1.parse_sp1_container(payload + b"x")


def test_producer_passes_payload_retention_gate() -> None:
    findings = check_no_measure_and_discard_payload(
        repo_root=REPO,
        strict=False,
        roots=("experiments/ddm_ec1_event_coordinate_producer.py",),
    )
    assert findings == []
