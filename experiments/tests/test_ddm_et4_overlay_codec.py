# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from experiments import ddm_et4_overlay_codec as codec


def test_sparse_patch_roundtrip_and_apply() -> None:
    before = np.zeros((codec.CAMERA_H, codec.CAMERA_W, codec.CHANNELS), dtype=np.uint8)
    after = before.copy()
    after.reshape(-1)[[0, 17, codec.FRAME_VALUES - 1]] = [3, 7, 11]

    record = codec.frame1_delta_record(5, before, after)
    compressed, receipt = codec.encode_patch_records([record], quality=1)
    decoded = codec.decode_patch_records(compressed)
    restored = codec.apply_patch_to_frame1(before, decoded[5])

    assert receipt["record_count"] == 1
    assert receipt["total_nnz"] == 3
    assert np.array_equal(restored, after)


def test_overlay_payload_closes_sections() -> None:
    parent = b"parent-payload"
    patch, _receipt = codec.encode_patch_records([], quality=1)
    payload, build = codec.encode_overlay_payload(
        parent_payload=parent,
        compressed_patch=patch,
        metadata={"schema": "test", "score_claim": False},
    )

    got_parent, got_patch, got_meta = codec.decode_overlay_payload(payload)

    assert got_parent == parent
    assert got_patch == patch
    assert got_meta["schema"] == "test"
    assert build["payload_bytes"] == len(payload)
