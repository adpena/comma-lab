"""Load-bearing unit tests for the CCS1 receiver-causal schema."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EXPERIMENTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENTS))

import ddm_ccs1_causal_schedule_builder as ccs1


def test_schedule_is_complete_and_causal() -> None:
    schedule = ccs1.build_schedule()
    assert schedule.order.size == ccs1.PLANE
    assert np.unique(schedule.order).size == ccs1.PLANE
    for group, positions in enumerate(schedule.group_positions):
        assert np.all(schedule.groups.reshape(-1)[positions] == group)


def test_context_uses_only_earlier_current_groups() -> None:
    schedule = ccs1.build_schedule()
    previous = np.zeros((ccs1.H, ccs1.W), dtype=np.uint8)
    current_a = np.full((ccs1.H, ccs1.W), ccs1.UNK, dtype=np.uint8)
    current_b = current_a.copy()
    group = 80
    positions = schedule.group_positions[group]
    later = schedule.groups.reshape(-1) >= group
    current_b.reshape(-1)[later] = 4
    key_a, base_a = ccs1.context_keys(previous, current_a, positions, schedule)
    key_b, base_b = ccs1.context_keys(previous, current_b, positions, schedule)
    assert np.array_equal(key_a, key_b)
    assert np.array_equal(base_a, base_b)


def test_model_roundtrip_and_exact_frequency_lattice() -> None:
    rng = np.random.default_rng(7)
    base = ccs1.quantize_rows(rng.integers(0, 1000, size=(ccs1.BASE_ROWS, ccs1.K)))
    keys = np.array([3, 10, 9000], dtype=np.uint32)
    leaves = ccs1.quantize_rows(rng.integers(0, 1000, size=(keys.size, ccs1.K)))
    raw = ccs1.serialize_model(base, keys, leaves)
    model = ccs1.parse_model(raw)
    assert np.array_equal(model.base_freq, base)
    assert np.array_equal(model.leaf_keys, keys)
    assert np.array_equal(model.leaf_freq, leaves)
    probs = ccs1.probabilities(model, keys, np.zeros(keys.size, dtype=np.uint16))
    restored = np.floor(probs.astype(np.float64) * (1 << 31)).astype(np.uint32)
    assert np.array_equal(restored, leaves.astype(np.uint32) << 15)


def test_split_matches_lm1_protocol() -> None:
    train, heldout = ccs1.split_frames()
    assert train.size == 450
    assert heldout.size == 150
    assert np.array_equal(heldout[:30], np.arange(30, 60))
    assert set(train).isdisjoint(set(heldout))
