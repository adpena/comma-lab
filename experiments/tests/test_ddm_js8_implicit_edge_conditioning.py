from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.ddm_ec1_runtime import js8_edge_state_conditioner as js8


def test_gate_round_trip_and_edge_state() -> None:
    weights = np.zeros((5, 5), dtype=np.float32)
    weights[0, 1] = weights[1, 0] = 1.0
    weights[0, 2] = weights[2, 0] = 0.5
    coded = js8.serialize_gate(weights, adapter_scale=0.75)
    decoded, scale, header = js8.parse_gate(coded)
    assert np.array_equal(decoded, weights)
    assert scale == 0.75
    assert header["schema"] == js8.SCHEMA

    tokens = torch.tensor([[[0, 1, 1], [0, 0, 2], [3, 3, 2]]])
    gate = js8.decoded_edge_state(tokens, torch.from_numpy(decoded))
    assert gate.shape == (1, 1, 3, 3)
    assert gate[0, 0, 0, 0].item() == 1.0
    assert gate[0, 0, 1, 2].item() == 0.5
    assert gate[0, 0, 2, 0].item() == 0.0


def test_gate_rejects_explicit_or_asymmetric_tables() -> None:
    asymmetric = np.zeros((5, 5), dtype=np.float32)
    asymmetric[0, 1] = 1.0
    with pytest.raises(js8.JS8RuntimeError):
        js8.serialize_gate(asymmetric, adapter_scale=1.0)

    diagonal = np.eye(5, dtype=np.float32)
    with pytest.raises(js8.JS8RuntimeError):
        js8.serialize_gate(diagonal, adapter_scale=1.0)


def test_zero_table_derives_exact_zero_gate() -> None:
    tokens = torch.randint(0, 5, (2, 9, 11), generator=torch.Generator().manual_seed(20260814))
    gate = js8.decoded_edge_state(tokens, torch.zeros((5, 5), dtype=torch.float32))
    assert torch.equal(gate, torch.zeros((2, 1, 9, 11), dtype=torch.float32))
