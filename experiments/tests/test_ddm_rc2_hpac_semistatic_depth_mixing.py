"""Behavioral guards for the ddm_rc2 IHS1 lossless coders."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from experiments import ddm_rc2_hpac_semistatic_depth_mixing as experiment
from experiments import ddm_rc2_hpac_semistatic_mixing_codec as codec


def _pack_depths(depths: list[int]) -> bytes:
    padded = depths + ([0] if len(depths) % 2 else [])
    values = np.asarray(padded, dtype=np.uint8).reshape(-1, 2)
    return (values[:, 0] | (values[:, 1] << 4)).tobytes()


def _fixture(seed: int = 7) -> tuple[bytes, list[int]]:
    rng = np.random.default_rng(seed)
    depths = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    row_counts = [11, 17, 29, 31, 37, 41, 43, 47, 53]
    rows = []
    for depth, count in zip(depths, row_counts, strict=True):
        if depth == 0:
            rows.append(np.zeros(count, dtype=np.int16))
        else:
            low, high = -(1 << (depth - 1)), 1 << (depth - 1)
            values = rng.integers(low, high, size=count, dtype=np.int16)
            values[1:] = np.where(rng.random(count - 1) < 0.65, values[:-1], values[1:])
            rows.append(values)
    body = b"IHS1" + _pack_depths(depths) + codec.pack_rows(rows, np.asarray(depths)) + b"tail-fixture"
    return body, row_counts


@pytest.mark.parametrize("family", sorted(codec.FAMILY_NAMES))
@pytest.mark.parametrize("serializer", sorted(codec.SERIALIZER_NAMES))
def test_every_semistatic_form_roundtrips_and_is_deterministic(family: int, serializer: int) -> None:
    body, row_counts = _fixture()
    first, first_parts = codec.encode_semistatic(body, row_counts, family, serializer)
    second, second_parts = codec.encode_semistatic(body, row_counts, family, serializer)
    assert first == second
    assert first_parts["payload"] == second_parts["payload"]
    assert first_parts["table_blob"]
    assert first_parts["table_entries"] > 0
    assert codec.restore_hpac(first, row_counts) == body


@pytest.mark.parametrize("weight_bytes", [8, 16, 32, 64])
def test_each_counted_logistic_weight_size_roundtrips(weight_bytes: int) -> None:
    body, row_counts = _fixture(13)
    groups = weight_bytes // codec.EXPERT_COUNT
    weights = np.zeros((groups, codec.EXPERT_COUNT), dtype=np.int8)
    weights[:, 0] = codec.WEIGHT_SCALE
    weights[:, 2] = 4
    stream, parts = codec.encode_logistic(body, row_counts, weights)
    assert len(parts["weights_blob"]) == weight_bytes
    assert parts["payload"]
    assert codec.restore_hpac(stream, row_counts) == body
    repeated, _ = codec.encode_logistic(body, row_counts, weights.copy())
    assert repeated == stream


def test_table_and_weights_change_coding_behavior_not_only_metadata() -> None:
    body, row_counts = _fixture(19)
    table_a, parts_a = codec.encode_semistatic(body, row_counts, codec.FAMILY_PREV_ZERO, codec.SERIALIZER_COUNTS_ULEB)
    table_b, parts_b = codec.encode_semistatic(body, row_counts, codec.FAMILY_EXACT_PREV, codec.SERIALIZER_PROB12)
    assert parts_a["table_blob"] != parts_b["table_blob"]
    assert parts_a["payload"] != parts_b["payload"]
    assert table_a != table_b

    one = np.zeros((1, codec.EXPERT_COUNT), dtype=np.int8)
    one[0, 0] = codec.WEIGHT_SCALE
    mixed = one.copy()
    mixed[0, 1] = codec.WEIGHT_SCALE // 2
    stream_one, one_parts = codec.encode_logistic(body, row_counts, one)
    stream_mixed, mixed_parts = codec.encode_logistic(body, row_counts, mixed)
    assert one_parts["weights_blob"] != mixed_parts["weights_blob"]
    assert one_parts["payload"] != mixed_parts["payload"]
    assert stream_one != stream_mixed
    assert codec.restore_hpac(stream_mixed, row_counts) == body


def test_integer_stretch_squash_is_exact_and_monotone() -> None:
    assert all(a < b for a, b in itertools.pairwise(codec.STRETCH))
    for probability in range(1, codec.PROBABILITY_ONE):
        assert codec.squash(codec.stretch(probability)) == probability


def test_malformed_headers_and_lengths_refuse() -> None:
    body, row_counts = _fixture()
    stream, _ = codec.encode_semistatic(body, row_counts, codec.FAMILY_PREV_SIGN, codec.SERIALIZER_PROB8)
    with pytest.raises(codec.Rc2CodecError, match="header"):
        codec.restore_hpac(b"bad", row_counts)
    with pytest.raises(codec.Rc2CodecError, match="header"):
        codec.restore_hpac(b"BAD!" + stream[4:], row_counts)

    fields = list(codec.HEADER.unpack_from(stream))
    fields[5] += len(stream)
    oversized = codec.HEADER.pack(*fields) + stream[codec.HEADER.size :]
    with pytest.raises(codec.Rc2CodecError, match="overrun"):
        codec.restore_hpac(oversized, row_counts)


def test_noncanonical_table_and_bad_weight_count_refuse() -> None:
    with pytest.raises(codec.Rc2CodecError, match="non-canonical"):
        codec.parse_table(b"\x81\x00", codec.SERIALIZER_PROB8)

    body, row_counts = _fixture()
    weights = np.zeros((1, codec.EXPERT_COUNT), dtype=np.int8)
    weights[0, 0] = codec.WEIGHT_SCALE
    stream, _ = codec.encode_logistic(body, row_counts, weights)
    fields = list(codec.HEADER.unpack_from(stream))
    fields[6] = 9
    malformed = codec.HEADER.pack(*fields) + stream[codec.HEADER.size :]
    with pytest.raises(codec.Rc2CodecError):
        codec.restore_hpac(malformed, row_counts)


def test_event_collection_exercises_all_eight_real_experts() -> None:
    body, row_counts = _fixture(23)
    rows, depths = codec.unpack_rows(body, row_counts)
    features, outcomes, event_depths, positions, nodes = codec.collect_logistic_events(rows, depths)
    assert features.shape[1] == codec.EXPERT_COUNT
    assert features.shape[0] == outcomes.size == event_depths.size == positions.size == nodes.size
    assert features.shape[0] > sum(row_counts)
    assert np.any(features[:, 0] != features[:, 1])
    assert set(np.unique(outcomes)) == {0, 1}


def test_container_grid_is_complete_and_selects_only_shippable_rows() -> None:
    grid = experiment.container_grid()
    assert len(grid) == 18
    assert len(set(grid)) == 18
    rows = [
        {
            "object": "candidate",
            "ck2": ck2,
            "quality": quality,
            "lgwin": lgwin,
            "container_bytes": 1000 - quality - lgwin - (10 if ck2 else 100),
            "shippable": ck2,
        }
        for ck2, quality, lgwin in grid
    ]
    selected = experiment.select_shippable(rows, "candidate")
    assert selected["ck2"] is True
    assert (selected["quality"], selected["lgwin"]) == (11, 24)


@pytest.mark.parametrize("groups", [1, 2, 4, 8])
def test_driver_group_ids_match_receiver_grouping(groups: int) -> None:
    depths = np.asarray([1, 2, 4, 8, 8], dtype=np.uint8)
    positions = np.asarray([0, 1, 2, 6, 7], dtype=np.uint8)
    expected = np.asarray(
        [
            codec.weight_group(int(depth), int(position), groups)
            for depth, position in zip(depths, positions, strict=True)
        ]
    )
    assert np.array_equal(experiment.group_ids(depths, positions, groups), expected)


def test_driver_integer_objective_matches_receiver_mixer() -> None:
    probabilities = np.asarray(
        [
            [250, 900, 1500, 2048, 2500, 3100, 3700, 4000],
            [4000, 3700, 3100, 2500, 2048, 1500, 900, 250],
            [1200, 1600, 1900, 2100, 2400, 2800, 3300, 3900],
        ],
        dtype=np.int16,
    )
    features = np.asarray(
        [[codec.stretch(int(value)) for value in row] for row in probabilities],
        dtype=np.int16,
    )
    weights = np.asarray([[19, -7, 12, 4, -2, 8, 3, 1]], dtype=np.int8)
    outcomes = np.asarray([1, 0, 1], dtype=np.uint8)
    gids = np.zeros(3, dtype=np.int64)
    decoder_probabilities = np.asarray([codec.mixed_probability(row, weights[0]) for row in probabilities])
    frequencies = np.where(
        outcomes == 1,
        decoder_probabilities,
        codec.PROBABILITY_ONE - decoder_probabilities,
    )
    expected = float(np.sum(-np.log2(frequencies / codec.PROBABILITY_ONE)))
    assert experiment.ideal_mixer_bits(features, outcomes, gids, weights) == pytest.approx(expected)
