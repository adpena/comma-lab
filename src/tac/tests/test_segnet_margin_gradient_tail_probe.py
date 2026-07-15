from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from tac.canonical_equations.segnet_margin_gradient_tail_20260715 import (
    segnet_margin_gradient_tail_block_jacobian,
)
from tac.scorer_surrogate.segnet_margin_gradient_tail_probe import (
    BLOCK_RELATIONS,
    PAIR_COUNT,
    QUERY_KINDS,
    RADII_PX,
    ArtifactBinding,
    MarginGradientTailObservation,
    MarginGradientTailProbePlan,
    append_observation,
    build_terminal_receipt,
    load_observations,
    load_plan,
    next_probe_tasks,
    write_plan,
)


def _artifact(tmp_path: Path, role: str, payload: bytes) -> ArtifactBinding:
    path = tmp_path / f"{role}.bin"
    path.write_bytes(payload)
    return ArtifactBinding.from_path(role=role, path=path)


def _plan(tmp_path: Path) -> MarginGradientTailProbePlan:
    return MarginGradientTailProbePlan(
        scorer=_artifact(tmp_path, "frozen_segnet_scorer", b"scorer"),
        source=_artifact(tmp_path, "n600_source", b"source"),
        cache=_artifact(tmp_path, "scorer_cache", b"cache"),
    )


def _row(
    plan: MarginGradientTailProbePlan,
    pair_index: int = 0,
    query_kind: str = "minimum_margin",
) -> MarginGradientTailObservation:
    digit = f"{pair_index:064x}"[-64:]
    cache_digit = f"{pair_index + 1:064x}"[-64:]
    return MarginGradientTailObservation(
        plan_id=plan.plan_id,
        pair_index=pair_index,
        scored_frame_index=2 * pair_index + 1,
        query_kind=query_kind,
        query_y=12,
        query_x=23,
        top_class=2,
        rival_class=3,
        margin=0.5,
        total_gradient_energy=2.0,
        nonzero_input_fraction=1.0,
        tail_energy_fraction=((64, 0.09), (128, 0.03), (192, 0.01)),
        block_jacobian_energy=(
            ("same_edge", 1.0),
            ("adjacent_edge", 0.2),
            ("remote_edge", 0.1),
        ),
        source_frame_sha256=digit,
        cache_record_sha256=cache_digit,
    )


def test_plan_roundtrip_and_live_hash_closure(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    path = tmp_path / "plan.json"
    write_plan(path, plan)
    assert load_plan(path) == plan
    Path(plan.cache.path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="custody mismatch"):
        plan.verify_live_custody()


@pytest.mark.parametrize("field,value", [("pair_count", 599), ("scorer_batch_size", 8)])
def test_plan_refuses_noncanonical_n600_geometry(
    tmp_path: Path, field: str, value: int
) -> None:
    base = _plan(tmp_path)
    kwargs = {
        "cache": base.cache,
        "pair_count": PAIR_COUNT,
        "scorer": base.scorer,
        "scorer_batch_size": 32,
        "source": base.source,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        MarginGradientTailProbePlan(**kwargs)


def test_observation_refuses_wrong_tail_or_block_geometry(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    base = _row(plan).to_dict()
    base.pop("row_key")
    base.pop("row_sha256")
    base["tail_energy_fraction"] = [[64, 0.01], [128, 0.02], [192, 0.0]]
    with pytest.raises(ValueError, match="non-increasing"):
        MarginGradientTailObservation.from_dict(
            {**base, "row_key": "unused", "row_sha256": "unused"}
        )
    base["tail_energy_fraction"] = [[64, 0.09], [128, 0.03], [192, 0.01]]
    base["block_jacobian_energy"] = [["same_edge", 1.0]]
    with pytest.raises(ValueError, match="block relations"):
        MarginGradientTailObservation.from_dict(
            {**base, "row_key": "unused", "row_sha256": "unused"}
        )


def test_append_is_resume_safe_and_conflict_closed(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    path = tmp_path / "rows.jsonl"
    row = _row(plan)
    assert append_observation(path, plan=plan, row=row)
    assert not append_observation(path, plan=plan, row=row)
    assert load_observations(path) == (row,)
    conflicting = replace(row, margin=0.75)
    with pytest.raises(ValueError, match="immutable observation conflict"):
        append_observation(path, plan=plan, row=conflicting)


def test_resume_task_order_is_pair_then_query(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    tasks = next_probe_tasks(plan, [_row(plan)])
    assert tasks[0] == (0, "high_margin_control")
    assert (0, "minimum_margin") not in tasks
    assert tasks[-1] == (599, "minimum_margin")


def test_terminal_receipt_refuses_partial_rows(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    with pytest.raises(ValueError, match="1199 tasks remain"):
        build_terminal_receipt(plan, [_row(plan)])


def test_terminal_receipt_closes_exact_n600_matrix(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    rows = [
        _row(plan, pair_index=pair_index, query_kind=query_kind)
        for pair_index in range(PAIR_COUNT)
        for query_kind in QUERY_KINDS
    ]
    receipt = build_terminal_receipt(plan, rows)
    assert receipt["completion"] == {
        "pair_count": 600,
        "query_count_per_pair": 2,
        "row_count": 1200,
    }
    assert receipt["factorization_verdict"] == "NO_VERDICT_THRESHOLD_NOT_PREREGISTERED"
    assert receipt["authority"]["score_claim"] is False
    assert receipt["tail_energy_fraction_mean"] == {
        "64": pytest.approx(0.09),
        "128": pytest.approx(0.03),
        "192": pytest.approx(0.01),
    }


def test_plan_json_rejects_authority_mutation(tmp_path: Path) -> None:
    plan = _plan(tmp_path)
    raw = json.loads(json.dumps(plan.to_dict()))
    raw["authority"]["promotion_eligible"] = True
    with pytest.raises(ValueError, match="authority"):
        MarginGradientTailProbePlan.from_dict(raw)


def test_canonical_equation_keeps_locality_verdict_open() -> None:
    law = segnet_margin_gradient_tail_block_jacobian()
    assert law["tail_energy_fraction"]["radii_px"] == list(RADII_PX)
    assert law["block_jacobian"]["relations"] == list(BLOCK_RELATIONS)
    assert law["negative_boundary"]["verdict"] == "NO_VERDICT_THRESHOLD_NOT_PREREGISTERED"
