from __future__ import annotations

import json

from tac.canonical_equations.defect_network_rate_code_20260712 import (
    BYTES_SAVED,
    CANDIDATE_BYTES,
    COMPONENT_STREAM_DELTA_BYTES,
    EQUATION_ID,
    INCUMBENT_BYTES,
    build_defect_network_component_delta_rate_v1,
    populate_defect_network_component_delta_rate_v1,
)


def test_equation_keeps_rate_go_separate_from_receiver_verdict() -> None:
    equation = build_defect_network_component_delta_rate_v1()
    anchor = equation.empirical_anchors[0]

    assert equation.equation_id == EQUATION_ID
    assert BYTES_SAVED == 6_382 == INCUMBENT_BYTES - CANDIDATE_BYTES
    assert anchor.empirical_output["rate_code_subverdict"] == "GO"
    assert anchor.empirical_output["defect_mechanism_subverdict"] == (
        "NO-GO_HEADER_DEDUPLICATION_CONFOUND"
    )
    assert COMPONENT_STREAM_DELTA_BYTES == 2_349
    assert anchor.empirical_output["overall_verdict"] == (
        "NEEDS-MORE_RECEIVER_GEOMETRY_AND_CONSUMPTION_UNMEASURED"
    )
    assert "d_seg or d_pose equality claim" in equation.domain_of_validity["excluded"]
    assert anchor.noise_floor == 0.0


def test_populate_uses_append_only_registry(tmp_path) -> None:
    path = tmp_path / "equations.jsonl"
    lock_path = tmp_path / "equations.lock"

    populate_defect_network_component_delta_rate_v1(
        path=path,
        lock_path=lock_path,
        agent="pytest",
        subagent_id="task452-test",
    )
    rows = [json.loads(line) for line in path.read_text().splitlines()]

    assert len(rows) == 1
    assert rows[0]["equation_id"] == EQUATION_ID
    assert rows[0]["event_type"] == "registered"


def test_package_exports_task452_builder_and_populator() -> None:
    from tac.canonical_equations import (
        build_defect_network_component_delta_rate_v1 as exported_builder,
    )
    from tac.canonical_equations import (
        populate_defect_network_component_delta_rate_v1 as exported_populator,
    )

    assert exported_builder is build_defect_network_component_delta_rate_v1
    assert exported_populator is populate_defect_network_component_delta_rate_v1
