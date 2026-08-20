from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tac.witness_dsl.taskspace_g8_a3_interaction_feedback import (
    TaskspaceG8A3InteractionFeedbackError,
    build_taskspace_g8_a3_interaction_feedback,
    feedback_receipt_bytes,
)

REPO = Path(__file__).resolve().parents[4]
FROZEN_TEST = REPO / "tools/tests/test_run_taskspace_g8_a3_n2_allocator.py"
SPEC = importlib.util.spec_from_file_location("g18_frozen_g14_synthetic_fixture", FROZEN_TEST)
assert SPEC is not None and SPEC.loader is not None
fixture = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fixture
SPEC.loader.exec_module(fixture)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _partition() -> dict[str, Any]:
    total = 2 * 384 * 512
    counts = {
        "closed_cell_count": total - 18,
        "realization_debt_cell_count": 16,
        "topology_debt_cell_count": 1,
        "fortunate_semantic_mismatch_cell_count": 1,
    }
    return {
        "source_pair_ids": [0, 1],
        "label_shape": [2, 384, 512],
        "current_semantic_labels_sha256": _sha("current-semantic"),
        "target_labels_sha256": _sha("target-labels"),
        "realized_labels_sha256": _sha("realized-labels"),
        "closed_mask_sha256": _sha("closed-mask"),
        "realization_debt_mask_sha256": _sha("realization-mask"),
        "topology_debt_mask_sha256": _sha("topology-mask"),
        "fortunate_semantic_mismatch_mask_sha256": _sha("fortunate-mask"),
        **counts,
        "closed_count_by_target_class": [counts["closed_cell_count"], 0, 0, 0, 0],
        "realization_debt_count_by_target_class": [16, 0, 0, 0, 0],
        "topology_debt_count_by_target_class": [1, 0, 0, 0, 0],
        "fortunate_semantic_mismatch_count_by_target_class": [1, 0, 0, 0, 0],
        "schema": "tac.same_class_realization_cell_partition_telemetry.v1",
        "exact_partition_definition": (
            "closed_z_eq_t_h_eq_t__realization_z_eq_t_h_ne_t__"
            "topology_z_ne_t_h_ne_t__fortunate_z_ne_t_h_eq_t.v1"
        ),
        "dense_masks_serialized": False,
        "counts_by_target_class": True,
    }


def _receipt(tmp_path: Path) -> bytes:
    store = fixture._store(tmp_path)
    experiment = fixture.runner.run_coupled_experiment(
        store=store,
        backend=fixture.FakeBackend(),
        g8_prefix_mode="1",
        a_prefixes=(1,),
    )
    summary = experiment["g12"]["summary"]
    partition = _partition()
    summary["canonical_four_way_z_t_h_partition"] = partition
    for field in (
        "closed_cell_count",
        "realization_debt_cell_count",
        "topology_debt_cell_count",
        "fortunate_semantic_mismatch_cell_count",
    ):
        summary[field] = partition[field]
    summary["dense_labels_or_rgb_serialized"] = False
    return fixture.runner.finalize_receipt(
        store=store,
        experiment=experiment,
        pointer_latest={"pointer_sha256": _sha("latest-pointer"), "target_score": 0.17},
    )


def _mutate(payload: bytes, callback: Callable[[dict[str, Any]], None]) -> bytes:
    value = json.loads(payload)
    callback(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"


def test_feedback_is_deterministic_complete_and_six_hook_closed(tmp_path: Path) -> None:
    payload = _receipt(tmp_path)
    first = build_taskspace_g8_a3_interaction_feedback(payload)
    second = build_taskspace_g8_a3_interaction_feedback(json.loads(payload))
    assert feedback_receipt_bytes(first) == feedback_receipt_bytes(second)
    assert first["source_receipt_sha256"] == hashlib.sha256(payload).hexdigest()
    partition = first["four_way_z_t_h_partition"]
    assert sum(partition[field] for field in (
        "closed_cell_count",
        "realization_debt_cell_count",
        "topology_debt_cell_count",
        "fortunate_semantic_mismatch_cell_count",
    )) == 2 * 384 * 512
    source = json.loads(payload)
    expected_rows = (
        len(source["g0_a_acquisition"]["measurements"])
        + len(source["g8_screen"]["measurements"])
        + 2 * len(source["conditional_a_treatments"])
    )
    assert first["row_inventory"]["occurrence_count"] == expected_rows
    assert set(first["hook_coverage"]) == {"1", "2", "3", "4", "5", "6"}
    assert first["downstream_payloads"]["continual_learning"]["compatible"] is False
    assert first["downstream_payloads"]["probe_outcomes"]["append_performed"] is False
    assert first["downstream_payloads"]["autopilot"]["dispatch_ready"] is False
    bitalloc = first["downstream_payloads"]["bit_allocator"]
    assert bitalloc["additive_independence_assumed"] is False
    assert bitalloc["mutually_exclusive_or_composed_actions_require_bundle_remeasurement"] is True


def test_feedback_preserves_exact_deltas_byte_ceilings_and_interactions(tmp_path: Path) -> None:
    result = build_taskspace_g8_a3_interaction_feedback(_receipt(tmp_path))
    transitions = result["transition_inventory"]["transitions"]
    interactions = result["interaction_inventory"]["interactions"]
    assert transitions and interactions
    assert all(isinstance(row["finite_byte_ceiling_real"], float) for row in transitions)
    assert all(row["improves_score"] == (row["exact_score_delta"] < 0.0) for row in transitions)
    probe_ids = {
        row["probe_id"] for row in result["downstream_payloads"]["probe_outcomes"]["payloads"]
    }
    assert {row["transition_id"] for row in transitions}.issubset(probe_ids)
    assert {row["interaction_id"] for row in interactions}.issubset(probe_ids)
    assert set(result["hook_coverage"]["5"]["interaction_ids"]) == {
        row["interaction_id"] for row in interactions
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (
            lambda value: value["g8_screen"]["measurements"].pop(),
            "one-to-one",
        ),
        (
            lambda value: value["g12"]["summary"].__setitem__("closed_cell_count", 1),
            "four-way count",
        ),
        (
            lambda value: value["pointer_latest"].__setitem__("below_target", True),
            "target-relative",
        ),
        (
            lambda value: value["truth"].__setitem__("score_claim", True),
            "frozen G14",
        ),
        (
            lambda value: value["conditional_a_treatments"][0].__setitem__(
                "interaction_I", value["conditional_a_treatments"][0]["interaction_I"] + 1.0
            ),
            "interaction",
        ),
        (
            lambda value: value["baseline"]["scorer_evidence"]["per_pair_d_seg"].__setitem__(
                0, value["baseline"]["scorer_evidence"]["per_pair_d_seg"][0] + 0.25
            ),
            "frozen G14",
        ),
    ],
)
def test_feedback_fails_closed_on_lost_or_laundered_signal(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    payload = _receipt(tmp_path)
    bad = _mutate(payload, mutation)
    with pytest.raises(TaskspaceG8A3InteractionFeedbackError, match=match):
        build_taskspace_g8_a3_interaction_feedback(bad)


def test_feedback_fails_closed_when_retained_interaction_is_orphaned(tmp_path: Path) -> None:
    payload = _receipt(tmp_path)

    def drop_treatment(value: dict[str, Any]) -> None:
        retained = value["g8_screen"]["retained_for_a_proposal_ids"]
        victim = retained[0]
        value["conditional_a_treatments"] = [
            row for row in value["conditional_a_treatments"] if row["g8_branch"]["proposal_id"] != victim
        ]
        production_rows = [
            *value["g0_a_acquisition"]["measurements"],
            *value["g8_screen"]["measurements"],
        ]
        for row in value["conditional_a_treatments"]:
            production_rows.extend((row["g8_a_measurement"], row["g0_a_measurement"]))
        best = min(production_rows, key=lambda row: (row["derived_component_total"], row["measurement_id"]))
        selected_path = value["selected_research_row"]["selected_archive_path"]
        value["selected_research_row"] = copy.deepcopy(best)
        value["selected_research_row"]["selected_archive_path"] = selected_path

    bad = _mutate(payload, drop_treatment)
    with pytest.raises(TaskspaceG8A3InteractionFeedbackError, match="lack complete"):
        build_taskspace_g8_a3_interaction_feedback(bad)
