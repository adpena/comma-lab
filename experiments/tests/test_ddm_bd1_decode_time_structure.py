from __future__ import annotations

import numpy as np
import pytest

from experiments import ddm_bd1_decode_time_structure as bd1


def test_constraint_stream_roundtrip_and_mutation_refusal() -> None:
    values = [index % len(bd1.FLOW_CONFIGS) for index in range(bd1.PAIR_COUNT)]
    payload = bd1.pack_constraint_stream(values)
    assert bd1.parse_constraint_stream(payload) == values
    mutated = bytearray(payload)
    mutated[-1] ^= 1
    with pytest.raises(bd1.BD1Error):
        bd1.parse_constraint_stream(bytes(mutated))


def test_candidate_archive_consumes_exact_body_and_constraint() -> None:
    values = [0] * bd1.PAIR_COUNT
    values[17] = 4
    raw = bd1.pack_constraint_stream(values)
    body = b"body-fixture"
    old_sha = bd1.BODY_ARCHIVE_SHA256
    try:
        bd1.BODY_ARCHIVE_SHA256 = bd1.sha256_bytes(body)
        raw = bd1.pack_constraint_stream(values)
        coded = bd1._encode_constraint("zlib9", raw)
        archive = bd1.pack_candidate_archive(body, "zlib9", raw, coded)
        parsed_body, parsed_values, codec = bd1.parse_candidate_archive(archive)
    finally:
        bd1.BODY_ARCHIVE_SHA256 = old_sha
    assert parsed_body == body
    assert parsed_values == values
    assert codec == "zlib9"


def test_identity_flow_is_exact_and_active_flow_changes_real_input() -> None:
    yy, xx = np.indices((bd1.CAMERA_H, bd1.CAMERA_W))
    master = np.stack(
        ((xx % 256), (yy % 256), ((xx + yy) % 256)), axis=-1
    ).astype(np.uint8)
    labels = np.zeros((bd1.EVAL_H, bd1.EVAL_W), dtype=np.uint8)
    labels[:, bd1.EVAL_W // 2 :] = 1
    identity = bd1.solve_token_topology_flow(master, labels, bd1.FLOW_CONFIGS[0])
    active = bd1.solve_token_topology_flow(master, labels, bd1.FLOW_CONFIGS[1])
    assert np.array_equal(identity, master)
    assert active.shape == master.shape
    assert active.dtype == np.uint8
    assert np.count_nonzero(active != master) > 0
