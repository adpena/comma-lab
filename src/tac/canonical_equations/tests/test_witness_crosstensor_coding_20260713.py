from __future__ import annotations

from tac.canonical_equations.registry import get_equation_by_id
from tac.canonical_equations.witness_crosstensor_coding_20260713 import (
    EQUATION_ID,
    build_witness_lossless_cross_tensor_storage_law_v1,
    lossless_joint_storage_gain_bytes,
    populate_witness_lossless_cross_tensor_storage_law_v1,
)


def test_lossless_joint_storage_gain_bytes() -> None:
    assert lossless_joint_storage_gain_bytes(63_659, 63_242) == 417


def test_equation_carries_exact_n600_anchor() -> None:
    equation = build_witness_lossless_cross_tensor_storage_law_v1()
    assert equation.equation_id == EQUATION_ID
    anchor = equation.empirical_anchors[0]
    assert anchor.inputs["eval_pairs"] == 600
    assert anchor.empirical_output["decoded_state_exact_equal"] is True
    assert anchor.empirical_output["joint_archive_bytes_saved"] == 417


def test_populate_roundtrips_through_locked_registry(tmp_path) -> None:
    path = tmp_path / "equations.jsonl"
    lock = tmp_path / "equations.lock"
    populate_witness_lossless_cross_tensor_storage_law_v1(
        path=path,
        lock_path=lock,
        agent="test",
        subagent_id="witness_crosstensor_structure",
    )
    got = get_equation_by_id(EQUATION_ID, path=path)
    assert got is not None
    assert got.empirical_anchors[0].empirical_output["joint_archive_bytes_saved"] == 417
