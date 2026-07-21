# SPDX-License-Identifier: MIT

from __future__ import annotations

import numpy as np
import pytest

from tac.optimization.s2_partition_seed import (
    PartitionEvent,
    PartitionEventSeed,
    PartitionSeedError,
    apply_partition_seed,
    decode_partition_seed,
    detect_partition_semantics,
    encode_partition_seed,
    packet_accounting,
)


def _semantic_fixture() -> tuple[np.ndarray, tuple[int, ...]]:
    # Channel ids are intentionally permuted relative to semantic order:
    # Road=3, Lane=4, Undrivable=1, Movable=0, MyCar=2.
    semantic_ids = (3, 4, 1, 0, 2)
    labels = np.full((4, 12, 16), semantic_ids[2], dtype=np.uint8)
    labels[:, 5:9] = semantic_ids[0]
    labels[:, 9:] = semantic_ids[4]
    labels[:, 6:9, 7] = semantic_ids[1]
    labels[:, 6:8, 11:13] = semantic_ids[3]
    # Make Movable less static without changing its area ordering.
    labels[1, 6:8, 11:13] = semantic_ids[0]
    labels[1, 6:8, 3:5] = semantic_ids[3]
    labels[3, 6:8, 11:13] = semantic_ids[0]
    labels[3, 6:8, 5:7] = semantic_ids[3]
    return labels, semantic_ids


def test_detect_partition_semantics_uses_spatial_static_signature() -> None:
    labels, expected = _semantic_fixture()
    detected = detect_partition_semantics(labels)
    assert detected.semantic_class_ids == expected
    assert detected.to_dict()["luma_consulted"] is False
    assert [row["class_id"] for row in detected.per_class] == list(range(5))


def _seed() -> PartitionEventSeed:
    return PartitionEventSeed(
        n_pairs=3,
        height=4,
        width=5,
        semantic_class_ids=(3, 4, 1, 0, 2),
        events=(
            PartitionEvent(0, 0, 1, 4, 3),
            PartitionEvent(0, 3, 4, 1, 2),
            PartitionEvent(2, 1, 0, 0, 3),
        ),
    )


def test_partition_seed_is_deterministic_strict_and_applied() -> None:
    seed = _seed()
    first = encode_partition_seed(seed)
    second = encode_partition_seed(seed)
    assert first == second
    decoded = decode_partition_seed(first)
    assert decoded == seed

    baseline = np.full((3, 4, 5), 3, dtype=np.uint8)
    baseline[0, 3, 4] = 2
    output = apply_partition_seed(baseline, decoded)
    assert output[0, 0, 1] == 4
    assert output[0, 3, 4] == 1
    assert output[2, 1, 0] == 0
    assert np.count_nonzero(output != baseline) == 3

    accounting = packet_accounting(first)
    assert accounting["packet_bytes"] == len(first)
    assert accounting["event_count"] == 3
    assert accounting["stored_plane_value_bytes"] == 0


def test_partition_seed_rejects_corruption_trailing_and_baseline_drift() -> None:
    payload = encode_partition_seed(_seed())
    broken = bytearray(payload)
    broken[-5] ^= 1
    with pytest.raises(PartitionSeedError, match="CRC"):
        decode_partition_seed(bytes(broken))
    with pytest.raises(PartitionSeedError, match="trailing"):
        decode_partition_seed(payload + b"x")

    baseline = np.full((3, 4, 5), 3, dtype=np.uint8)
    baseline[0, 3, 4] = 2
    baseline[0, 0, 1] = 1
    with pytest.raises(PartitionSeedError, match="baseline class"):
        apply_partition_seed(baseline, _seed())

    baseline[0, 0, 1] = 5
    with pytest.raises(PartitionSeedError, match="class ids"):
        apply_partition_seed(baseline, _seed())


@pytest.mark.parametrize(
    "events,match",
    [
        ((PartitionEvent(0, 0, 1, 3, 3),), "target"),
        (
            (PartitionEvent(0, 0, 1, 4, 3), PartitionEvent(0, 0, 1, 2, 3)),
            "strictly site-sorted",
        ),
        ((PartitionEvent(4, 0, 1, 4, 3),), "pair"),
    ],
)
def test_partition_seed_rejects_invalid_events(
    events: tuple[PartitionEvent, ...], match: str
) -> None:
    with pytest.raises(PartitionSeedError, match=match):
        PartitionEventSeed(
            n_pairs=3,
            height=4,
            width=5,
            semantic_class_ids=(0, 1, 2, 3, 4),
            events=events,
        )
