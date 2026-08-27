"""Real-field QBW1 mechanism tests; these are not synthetic empirical anchors."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_real_pair_receiver_and_mutation_contract() -> None:
    packet = _load("ddm_qbw1_packet", REPO / "experiments/ddm_qbw1_packet.py")
    builder = _load("ddm_qbw1_builder", REPO / "experiments/ddm_qbw1_builder.py")
    field = np.memmap(
        builder.SOURCE_FIELD,
        dtype=np.uint8,
        mode="r",
        shape=(builder.N, builder.H, builder.W),
    )
    source = np.asarray(field[0]).copy()
    obj = builder.extract_object(source)
    model = packet.QBW1Model(dictionary=b"")
    encoded = packet.encode_record(
        0,
        model,
        obj["chains"],
        obj["seed_labels"],
        obj["lane_events"],
    )
    repeat = packet.encode_record(
        0,
        model,
        obj["chains"],
        obj["seed_labels"],
        obj["lane_events"],
    )
    assert encoded == repeat
    receiver = packet.decode_receiver(encoded, model)
    expected = source.copy()
    expected[expected == 1] = 0
    assert np.array_equal(receiver["base_field"], expected)
    original = packet.decode_record(encoded, model)
    for _name, start, end in packet.section_spans(encoded):
        mutated = builder.flip_one_bit(encoded, start, end)
        assert builder.mutation_outcome(mutated, model, original) in {
            "REFUSED",
            "DECLARED_OBJECT_CHANGED",
        }


def test_preregistered_selection_is_seeded_stratified_n32() -> None:
    packet = _load("ddm_qbw1_packet_selection", REPO / "experiments/ddm_qbw1_packet.py")
    sys.modules["ddm_qbw1_packet"] = packet
    builder = _load("ddm_qbw1_builder_selection", REPO / "experiments/ddm_qbw1_builder.py")
    field = np.memmap(
        builder.SOURCE_FIELD,
        dtype=np.uint8,
        mode="r",
        shape=(builder.N, builder.H, builder.W),
    )
    strata, selected = builder.selection_rows(field)
    assert len(strata) == 20
    assert len(selected) == 32
    assert len({row["pair_id"] for row in selected}) == 32
    assert all(row["population_size"] == 30 for row in selected)
    by_id = {row["stratum_id"]: row for row in strata}
    assert all(row["pair_id"] in by_id[row["stratum_id"]]["population_pair_ids"] for row in selected)
    _strata_repeat, selected_repeat = builder.selection_rows(field)
    assert selected_repeat == selected
