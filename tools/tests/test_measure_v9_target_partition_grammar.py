from __future__ import annotations

import copy
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
import pytest

from tools.measure_v9_target_partition_grammar import (
    LABELS,
    CensusError,
    build_receipt,
    canonical_json_bytes,
    make_receipt_envelope,
    measure_partition,
    snapshot_file_custody,
    validate_labels,
    validate_receipt,
    write_once_receipt,
)


def _fixture() -> np.ndarray:
    return np.array(
        [
            [[0, 0, 1, 1, 2], [0, 3, 1, 2, 2], [4, 3, 3, 2, 4]],
            [[0, 1, 1, 2, 2], [0, 3, 4, 2, 2], [4, 3, 3, 4, 4]],
        ],
        dtype=np.uint8,
    )


def _component_sizes(mask: np.ndarray) -> list[int]:
    seen: set[tuple[int, int]] = set()
    sizes: list[int] = []
    for y, x in np.argwhere(mask):
        start = (int(y), int(x))
        if start in seen:
            continue
        queue = deque([start])
        seen.add(start)
        size = 0
        while queue:
            cy, cx = queue.popleft()
            size += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (cy + dy, cx + dx)
                    if (
                        (dy or dx)
                        and 0 <= neighbor[0] < mask.shape[0]
                        and 0 <= neighbor[1] < mask.shape[1]
                        and mask[neighbor]
                        and neighbor not in seen
                    ):
                        seen.add(neighbor)
                        queue.append(neighbor)
        sizes.append(size)
    return sizes


def _receipt() -> dict[str, object]:
    return make_receipt_envelope(
        {
            "schema": "tac.v9_target_partition_grammar_census.v1",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "candidate_lineage_prohibition": "target table is forbidden from archive payloads",
            "authority_scope": "test fixture",
        }
    )


def test_exact_small_census_matches_independent_counts() -> None:
    labels = _fixture()
    result = measure_partition(labels, expected_shape=labels.shape)
    assert result["shape"] == [2, 3, 5]
    assert result["aggregate"]["evidence_rows"] == 2
    assert result["aggregate"]["temporal_transition_rows"] == 1
    assert result["aggregate"]["total_sites"] == 30
    for frame_index, frame in enumerate(labels):
        row = result["frames"][frame_index]
        expected_runs = int(frame.shape[0] + np.count_nonzero(frame[:, 1:] != frame[:, :-1]))
        assert row["row_runs"] == expected_runs
        for label in LABELS:
            class_row = row["classes"][label]
            sizes = _component_sizes(frame == label)
            assert class_row["pixels"] == int(np.count_nonzero(frame == label))
            assert class_row["components_8_connected"] == len(sizes)
            assert class_row["components_ge_16"] == sum(size >= 16 for size in sizes)
            assert sum(class_row["component_size_buckets"].values()) == len(sizes)


def test_diagonal_pixels_are_one_eight_connected_component() -> None:
    labels = np.zeros((1, 3, 5), dtype=np.uint8)
    labels[0, 0, 0] = 1
    labels[0, 1, 1] = 1
    labels[0, 0, 4] = 2
    labels[0, 1, 4] = 3
    labels[0, 2, 4] = 4
    result = measure_partition(labels, expected_shape=labels.shape)
    assert result["frames"][0]["classes"][1]["components_8_connected"] == 1


def test_spatial_adjacency_and_temporal_matrix_close_exactly() -> None:
    labels = _fixture()
    result = measure_partition(labels, expected_shape=labels.shape)
    for frame_index, frame in enumerate(labels):
        row = result["frames"][frame_index]
        assert row["horizontal_boundaries"] == int(np.count_nonzero(frame[:, 1:] != frame[:, :-1]))
        assert row["vertical_boundaries"] == int(np.count_nonzero(frame[1:, :] != frame[:-1, :]))
    temporal = result["successive_pair_end_temporal"][0]
    matrix = np.asarray(temporal["transition_matrix_previous_to_current"])
    assert int(matrix.sum()) == labels.shape[1] * labels.shape[2]
    assert temporal["changed_sites"] == int(np.count_nonzero(labels[0] != labels[1]))
    assert temporal["previous_source_frame_index"] == 1
    assert temporal["current_source_frame_index"] == 3
    assert temporal["source_frame_stride"] == 2


def test_canonical_uint8_stream_hash_detects_one_site_mutation() -> None:
    labels = _fixture()
    first = measure_partition(labels, expected_shape=labels.shape)
    mutated = labels.copy()
    mutated[0, 0, 0] = 4
    second = measure_partition(mutated, expected_shape=mutated.shape)
    assert first["canonical_uint8_label_stream_sha256"] == hashlib.sha256(labels.tobytes()).hexdigest()
    assert first["canonical_uint8_label_stream_sha256"] != second["canonical_uint8_label_stream_sha256"]


@pytest.mark.parametrize(
    ("labels", "shape", "message"),
    [
        (_fixture().astype(np.float32), (2, 3, 5), "integer dtype"),
        (_fixture(), (3, 3, 5), "shape changed"),
        (np.where(_fixture() == 4, 3, _fixture()), (2, 3, 5), "alphabet changed"),
        (np.where(_fixture() == 4, 9, _fixture()), (2, 3, 5), "outside"),
    ],
)
def test_validation_fails_closed(labels: np.ndarray, shape: tuple[int, int, int], message: str) -> None:
    with pytest.raises(CensusError, match=message):
        validate_labels(labels, expected_shape=shape)


def test_body_hash_rejects_mutated_receipt(tmp_path: Path) -> None:
    receipt = _receipt()
    validate_receipt(receipt)
    assert receipt["body_sha256"] == hashlib.sha256(canonical_json_bytes(receipt["body"])).hexdigest()
    mutated = copy.deepcopy(receipt)
    mutated["body"]["authority_scope"] += " changed"
    with pytest.raises(CensusError, match="SHA-256 mismatch"):
        validate_receipt(mutated)


def test_receipt_builder_refuses_small_measurement() -> None:
    measurements = measure_partition(_fixture(), expected_shape=_fixture().shape)
    with pytest.raises(CensusError, match="exact n600 geometry"):
        build_receipt(cache_path=Path("unused.npz"), measurements=measurements)


def test_receipt_validation_rejects_rehashed_authority_mutation() -> None:
    mutated = copy.deepcopy(_receipt())
    mutated["body"]["score_claim"] = True
    mutated["body_sha256"] = hashlib.sha256(canonical_json_bytes(mutated["body"])).hexdigest()
    with pytest.raises(CensusError, match="authority marker changed"):
        validate_receipt(mutated)


def test_write_once_receipt_refuses_same_and_different_rewrites(tmp_path: Path) -> None:
    receipt = _receipt()
    output = tmp_path / "receipt.json"
    write_once_receipt(output, receipt)
    loaded = json.loads(output.read_text(encoding="utf-8"))
    validate_receipt(loaded)
    with pytest.raises(CensusError, match="already exists"):
        write_once_receipt(output, receipt)
    changed = copy.deepcopy(receipt)
    changed["body"]["authority_scope"] += " changed"
    changed["body_sha256"] = hashlib.sha256(canonical_json_bytes(changed["body"])).hexdigest()
    with pytest.raises(CensusError, match="already exists"):
        write_once_receipt(output, changed)


def test_research_only_and_candidate_lineage_prohibition_are_explicit(tmp_path: Path) -> None:
    receipt = _receipt()
    body = receipt["body"]
    assert body["research_only"] is True
    assert body["score_claim"] is False
    assert body["promotion_eligible"] is False
    assert body["pointer_moved"] is False
    assert "forbidden from archive payloads" in body["candidate_lineage_prohibition"]


def test_snapshot_file_custody_binds_exact_bytes(tmp_path: Path) -> None:
    cache = tmp_path / "cache.npz"
    cache.write_bytes(b"frozen-cache")
    custody = snapshot_file_custody(cache)
    assert custody == {
        "path": str(cache.resolve()),
        "bytes": 12,
        "sha256": hashlib.sha256(b"frozen-cache").hexdigest(),
    }
