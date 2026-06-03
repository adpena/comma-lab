# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.adaptation.hard_pair_indices import (
    HardPairIndicesError,
    load_pair_indices_file,
    merge_pair_indices,
    normalize_pair_indices,
    pair_indices_from_mapping,
    parse_pair_indices_csv,
    validate_pair_indices_in_range,
)


def test_hard_pair_indices_parse_ordered_unique_csv() -> None:
    assert parse_pair_indices_csv("") == ()
    assert parse_pair_indices_csv("3,1,3,0") == (3, 1, 0)
    assert normalize_pair_indices([4, "2", 4]) == (4, 2)
    assert merge_pair_indices((3, 1), "1,2,3") == (3, 1, 2)


def test_hard_pair_indices_reject_ambiguous_values() -> None:
    with pytest.raises(HardPairIndicesError, match="negative"):
        parse_pair_indices_csv("2,-1")
    with pytest.raises(HardPairIndicesError, match="boolean"):
        normalize_pair_indices([True])
    with pytest.raises(HardPairIndicesError, match="invalid"):
        normalize_pair_indices(["not-an-index"])
    with pytest.raises(HardPairIndicesError, match="invalid"):
        normalize_pair_indices([1.0])
    with pytest.raises(HardPairIndicesError, match="invalid"):
        normalize_pair_indices([1.9])
    with pytest.raises(HardPairIndicesError, match="invalid"):
        normalize_pair_indices(["1.9"])


def test_hard_pair_indices_load_json_and_text_artifacts(tmp_path: Path) -> None:
    json_path = tmp_path / "hard_pairs.json"
    json_path.write_text(json.dumps({"hard_pair_indices": [5, 2, 5]}), encoding="utf-8")
    text_path = tmp_path / "hard_pairs.txt"
    text_path.write_text("7\n3,7\n", encoding="utf-8")

    assert load_pair_indices_file(json_path) == (5, 2)
    assert load_pair_indices_file(text_path) == (7, 3)


def test_hard_pair_indices_extract_nested_feedback_schema() -> None:
    payload = {
        "sample_generalization_gate": {
            "hard_pair_coverage": {
                "prioritized_pair_indices": [22, 4, 22],
            },
        },
        "pair_indices": [22, 4],
    }

    assert pair_indices_from_mapping(payload) == (22, 4)


def test_hard_pair_indices_conflicting_sources_fail_closed() -> None:
    payload = {
        "sample_generalization_gate": {
            "hard_pair_coverage": {
                "prioritized_pair_indices": [22, 4, 22],
            },
            "hard_pair_indices": [99],
        },
        "pair_indices": [1],
    }

    with pytest.raises(HardPairIndicesError, match="conflicting"):
        pair_indices_from_mapping(payload)


def test_hard_pair_indices_validate_range() -> None:
    assert validate_pair_indices_in_range([3, "1", 3], num_pairs=4) == (3, 1)
    with pytest.raises(HardPairIndicesError, match="out-of-range"):
        validate_pair_indices_in_range([3, 4], num_pairs=4)
