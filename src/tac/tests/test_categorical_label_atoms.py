from __future__ import annotations

from tac.categorical_label_atoms import (
    CATEGORICAL_TYPED_LABEL_ATOMS_CONTRACT,
    build_categorical_typed_label_atoms,
    canonical_categorical_label_atom_rows,
)


def test_categorical_typed_label_atoms_are_deterministic_and_ordered() -> None:
    first = build_categorical_typed_label_atoms()
    second = build_categorical_typed_label_atoms()

    assert first == second
    assert first["typed_label_atoms_contract"] == CATEGORICAL_TYPED_LABEL_ATOMS_CONTRACT
    assert first["score_claim"] is False
    assert first["dispatch_attempted"] is False
    assert first["ready_for_exact_eval_dispatch"] is False
    assert [row["name"] for row in first["atoms"]] == [
        "road",
        "lane_markings",
        "undrivable",
        "movable",
        "my_car",
    ]
    assert [row["selfcomp_gray"] for row in first["atoms"]] == [0, 255, 64, 192, 128]


def test_categorical_typed_label_atoms_emit_prior_weights_and_policy() -> None:
    rows = canonical_categorical_label_atom_rows()

    assert [row["atom_id"] for row in rows] == [
        "contest_class_0_road",
        "contest_class_1_lane_markings",
        "contest_class_2_undrivable",
        "contest_class_3_movable",
        "contest_class_4_my_car",
    ]
    assert [row["semantic_priority_weight_ppm"] for row in rows] == [
        266667,
        266667,
        133333,
        200000,
        133333,
    ]
    assert sum(row["semantic_priority_weight_ppm"] for row in rows) == 1_000_000
    assert rows[1]["openpilot_prior_hint"] == "lane_marking_track_prior"
    assert {row["runtime_consumed"] for row in rows} == {False}
    assert {row["usage"] for row in rows} == {"compression_time_atom_ranking_only"}
