# SPDX-License-Identifier: MIT
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
TOOL = REPO / "tools" / "measure_ddm_dr2_scc_outside_view.py"
SPEC = importlib.util.spec_from_file_location("measure_ddm_dr2_scc_outside_view", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_record_race_and_packet_roundtrip(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "EXPECTED_SHAPE", (6, 11, 8))
    tensor = np.zeros(MODULE.EXPECTED_SHAPE, dtype="<i8")
    for row_index in range(tensor.shape[1]):
        tensor[:, row_index, 0] = np.arange(tensor.shape[0]) * (row_index + 1)

    rows = []
    streams = []
    for row_index in range(tensor.shape[1]):
        row, stream = MODULE.measure_record(tensor[:, row_index : row_index + 1, :], row_index)
        rows.append(row)
        streams.append(stream)

    packet = MODULE.build_measurement_packet(rows, streams)
    restored = MODULE.decode_measurement_packet(packet)
    redundancy = MODULE.measure_pairwise_redundancy(rows, streams)
    assert np.array_equal(restored, tensor)
    assert all(row["unique_states"] == tensor.shape[0] for row in rows)
    assert all(row["available_modes"]["static"]["admissible_exact"] is False for row in rows)
    assert all(row["audit_triple"]["three_layer_decomposition"]["status"] == "INCOMPLETE" for row in rows)
    assert redundancy["ordered_pair_count"] == 11 * 10
    assert len(redundancy["ordered_pairs"]) == 11 * 10


def test_packet_rejects_trailing_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(MODULE, "EXPECTED_SHAPE", (2, 11, 8))
    tensor = np.zeros(MODULE.EXPECTED_SHAPE, dtype="<i8")
    rows = []
    streams = []
    for row_index in range(tensor.shape[1]):
        row, stream = MODULE.measure_record(tensor[:, row_index : row_index + 1, :], row_index)
        rows.append(row)
        streams.append(stream)
    packet = MODULE.build_measurement_packet(rows, streams)
    with pytest.raises(MODULE.MeasurementError, match="trailing"):
        MODULE.decode_measurement_packet(packet + b"x")
