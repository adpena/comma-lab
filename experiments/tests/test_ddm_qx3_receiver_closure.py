# SPDX-License-Identifier: MIT
from __future__ import annotations

import struct

import numpy as np

from experiments import ddm_qbflow_packet as qbf1
from experiments import ddm_qx3_receiver_closure as qx3


def _split_model(raw: bytes) -> tuple[bytes, bytes, bytes]:
    view = memoryview(raw)
    count = struct.unpack_from(">H", view, 4)[0]
    offset = 6
    records: list[bytes] = []
    for _ in range(count):
        start = offset
        name_len, _bits, ndim, _reserved, _scale, _count, packed_len = qbf1._TENSOR_HEADER.unpack_from(
            view, offset
        )
        offset += qbf1._TENSOR_HEADER.size
        offset += name_len + 2 * ndim + packed_len
        records.append(bytes(view[start:offset]))
    assert offset == len(raw)
    # Role groups are deliberately not in global tensor-name order.
    grouped = (records[:14] + records[-14:], records[14:24], records[24:28])
    groups = []
    for group_id, group_records in enumerate(grouped, 1):
        groups.append(
            qx3.QXT_HEADER.pack(b"QXT1", group_id, 0, len(group_records))
            + b"".join(group_records)
        )
    return tuple(groups)  # type: ignore[return-value]


def test_qxt_groups_reassemble_qbt_model_exactly() -> None:
    params = {
        name: np.zeros(shape, dtype=np.float32)
        for name, shape in qbf1.expected_param_shapes().items()
    }
    raw = qbf1.encode_model(params)
    groups = _split_model(raw)
    rebuilt, trace = qx3.reassemble_qbt_model(groups, expected_sha256=qx3.sha256_bytes(raw))
    assert rebuilt == raw
    assert trace["model_raw_bytes"] == len(raw)


def test_dense_delta_round_trip_for_all_source_target_classes() -> None:
    source = np.repeat(np.arange(5, dtype=np.uint8), 5).reshape(5, 5)
    target = np.tile(np.arange(5, dtype=np.uint8), 5).reshape(5, 5)
    codes = qx3.delta_codes(source, target)
    assert np.array_equal(qx3.apply_delta_codes(source, codes), target)
    assert np.all(codes[source == target] == 0)
    assert np.all((codes[source != target] >= 1) & (codes[source != target] <= 4))
