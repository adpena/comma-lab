from __future__ import annotations

import numpy as np
import pytest

from experiments import ddm_bs4x_selected_storage_preflight as selected_gate


def test_selected_storage_floor_uses_full_cube_and_descent_surface() -> None:
    codes = np.zeros((selected_gate.PAIR_COUNT, selected_gate.DIMENSIONS), dtype=np.int16)
    result = selected_gate.selected_storage_floor(codes)

    assert result["minimum_endpoint_margin"] == 2047
    assert result["full_surface_rows"] == selected_gate.PAIR_COUNT
    assert result["minimum_candidate_evaluations_per_pair"] == 177
    assert result["minimum_materialized_payload_bytes"] == 51_859_719_936
    assert result["required_free_bytes"] == 60_449_654_528


def test_selected_storage_floor_refuses_endpoint_ambiguous_rows() -> None:
    codes = np.zeros((selected_gate.PAIR_COUNT, selected_gate.DIMENSIONS), dtype=np.int16)
    codes[3, 7] = 2047 - 34

    with pytest.raises(
        selected_gate.BS4XStorageError,
        match="all-full-row lower-bound proof",
    ):
        selected_gate.selected_storage_floor(codes)
