"""Synthetic-only tests for the cached n600 compander mechanism-ledger builder."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
TOOL = REPO / "tools/build_compander_ground_class_pair_ledger.py"
SPEC = importlib.util.spec_from_file_location("_compander_pair_ledger", TOOL)
assert SPEC is not None and SPEC.loader is not None
ledger_tool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ledger_tool)


def test_accumulator_preserves_direction_and_rows() -> None:
    gt = np.array(
        [
            [[0, 0, 1], [1, 0, 1], [0, 1, 1], [0, 0, 0]],
            [[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]],
        ],
        dtype=np.int64,
    )
    witness = gt.astype(np.int8)
    witness[0, 1, 1] = 1  # Road -> Lane on row 1
    witness[0, 2, 1] = 0  # Lane -> Road on row 2
    witness[1, 3, 0] = 1  # Road -> Lane on row 3
    counts, seen = ledger_tool.accumulate_directed_row_counts(
        gt,
        [(np.array([1, 0]), witness[[1, 0]])],
        expected_shape=(2, 4, 3),
    )
    assert seen == [0, 1]
    assert counts[0, 1].tolist() == [0, 1, 0, 1]
    assert counts[1, 0].tolist() == [0, 0, 1, 0]
    payload = ledger_tool.ledger_from_counts(counts, n_pairs=2, height=4, width=3)
    assert payload["total_flip_count"] == 3
    road_lane = next(
        row
        for row in payload["undirected_pairs"]
        if row["source_class"] == 0 and row["target_class"] == 1
    )
    assert road_lane["strict_planar_ground_pair"] is True
    assert road_lane["flip_count"] == 3


def test_pair_custody_refuses_duplicate_or_gap() -> None:
    gt = np.zeros((2, 2, 2), dtype=np.int64)
    witness = np.zeros((2, 2, 2), dtype=np.int8)
    with pytest.raises(ValueError, match=r"exactly 0\.\.1"):
        ledger_tool.accumulate_directed_row_counts(
            gt,
            [(np.array([0, 0]), witness)],
            expected_shape=(2, 2, 2),
        )


def test_all_twenty_directed_and_ten_undirected_pairs_stay_visible() -> None:
    counts = np.zeros((5, 5, 384), dtype=np.int64)
    counts[0, 1, 193] = 4
    payload = ledger_tool.ledger_from_counts(counts, n_pairs=1, height=384, width=1)
    assert len(payload["directed_pairs"]) == 20
    assert len(payload["undirected_pairs"]) == 10
    assert sum(row["strict_planar_ground_pair"] for row in payload["directed_pairs"]) == 2
    row = next(
        item
        for item in payload["directed_pairs"]
        if item["source_class"] == 0 and item["target_class"] == 1
    )
    assert row["ground_rows_v_gt_174_candidate_metrics"] is not None
    assert len(row["row_counts_0_through_383"]) == 384
