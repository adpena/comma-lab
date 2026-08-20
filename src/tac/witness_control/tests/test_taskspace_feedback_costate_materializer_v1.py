from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tac.witness_control.taskspace_feedback_costate_materializer_v1 import (
    TaskspaceFeedbackCostateMaterializerError,
    materialize_taskspace_feedback_costate_v1,
)
from tac.witness_control.taskspace_interaction_costate_bridge_v1 import (
    FeedbackExactBaseBindingV1,
    FullN600ReceiverEqualityClosureV1,
    ReceiverEqualityAdmissionControlV1,
)
from tac.witness_dsl.pair_population_envelope import PairPopulation

REPO = Path(__file__).resolve().parents[4]
G18_TEST = REPO / "src/tac/witness_dsl/tests/test_taskspace_g8_a3_interaction_feedback.py"
G19_TEST = REPO / "src/tac/witness_control/tests/test_taskspace_interaction_costate_bridge_v1.py"
G20_RECEIPT = (
    REPO
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
    / "ep725_lossless_xcodec_recode_20260726/receipt.json"
)
G22_BOUNDED = (
    REPO
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
    / "g22_ep725_xcodec_n600_equality_replay_20260726/bounded_pair0_decode_receipt_v4.json"
)
G22_FULL = (
    REPO
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
    / "g22_ep725_xcodec_n600_equality_replay_20260726/full_n600_decode_receipt.json"
)
G25_RECEIPT = (
    REPO
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
    / "ep725_population_global_recode_v2_20260726_r2/receipt.json"
)
G28_RECEIPT = (
    REPO
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
    / "g28_ep725_xcodec_full_n600_bridge_eval_20260726.json"
)


def _load_fixture(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


g18_fixture = _load_fixture(G18_TEST, "g26_g18_fixture")
g19_fixture = _load_fixture(G19_TEST, "g26_g19_fixture")


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _population() -> bytes:
    pair_ids = tuple(range(600))
    return PairPopulation.derive(
        source_pair_ids=pair_ids,
        v9_local_to_source_pair_ids=pair_ids,
        pbr_local_to_source_pair_ids=pair_ids,
        ir_local_to_source_pair_ids=pair_ids,
        v10_local_to_source_pair_ids=pair_ids,
    ).to_bytes()


def _full_g22() -> bytes:
    value = json.loads(G22_BOUNDED.read_bytes())
    value["decode_receipt"]["pair_ids"] = list(range(600))
    value["decode_receipt"]["pair_population_exact_ordered_0_through_599"] = True
    value["output_witness"].update(
        {
            "pairs_compared": 600,
            "frames_compared": 1200,
            "raw_bytes_each": 600 * 2 * 874 * 1164 * 3,
        }
    )
    value["authority_pointer_status"]["full_n600_receiver_replay_owed"] = False
    argv = value["reproducibility"]["exact_resume_argv"]
    argv[argv.index("--pair-count") + 1] = "600"
    argv.append("--confirm-full-n600")
    return _canonical(value)


def _binding_from_record(record: dict[str, Any]) -> FeedbackExactBaseBindingV1:
    value = dict(record["g19_controller"]["record"]["binding"])
    value["source_pair_ids"] = tuple(value["source_pair_ids"])
    return FeedbackExactBaseBindingV1(**value)


def test_refuses_partial_g14_and_arbitrary_population_digest(tmp_path: Path) -> None:
    population = _population()
    with pytest.raises(TaskspaceFeedbackCostateMaterializerError, match="terminal final_receipt"):
        materialize_taskspace_feedback_costate_v1(
            _canonical({"schema": "tac.taskspace_g8_a3_n2_allocator.v1"}),
            population,
        )

    final_receipt = g18_fixture._receipt(tmp_path)
    with pytest.raises(TaskspaceFeedbackCostateMaterializerError, match="caller population digest"):
        materialize_taskspace_feedback_costate_v1(
            final_receipt,
            population,
            population_sha256_hint=_sha("invented-population"),
        )


def test_exact_terminal_and_reopened_population_emit_g18_g19(tmp_path: Path) -> None:
    population_bytes = _population()
    result = materialize_taskspace_feedback_costate_v1(
        g18_fixture._receipt(tmp_path),
        population_bytes,
    )
    population = PairPopulation.from_bytes(population_bytes)
    assert result["input_custody"]["pair_population"]["population_sha256"] == population.population_sha256
    assert result["g19_controller"]["record"]["binding"]["population_manifest_sha256"] == (population.population_sha256)
    assert result["truth"]["authority_invented"] is False
    assert result["truth"]["live_partial_g14_consumed"] is False


def test_g22_bounded_refused_and_exact_full_n600_accepted(tmp_path: Path) -> None:
    population = _population()
    g20 = G20_RECEIPT.read_bytes()
    with pytest.raises(TaskspaceFeedbackCostateMaterializerError, match="full-n600 G20"):
        materialize_taskspace_feedback_costate_v1(
            g18_fixture._receipt(tmp_path),
            population,
            g20_placement_receipt=g20,
            g22_receiver_equality_receipt=G22_BOUNDED.read_bytes(),
        )

    result = materialize_taskspace_feedback_costate_v1(
        g18_fixture._receipt(tmp_path / "full"),
        population,
        g20_placement_receipt=g20,
        g22_receiver_equality_receipt=_full_g22(),
    )
    placements = result["g19_controller"]["record"]["whole_object_representation_placements"]
    assert placements["receiver_equality_closed_count"] == 1
    assert placements["rows"][0]["receiver_equality_handoff_unblocked"] is True
    handoff = next(
        row
        for row in result["g19_controller"]["record"]["blocked_handoffs"]
        if row["consumer_api"] == "whole_object_representation_placement_costate_ingest"
    )
    assert handoff["status"] == "READY_RECEIVER_EQUALITY_CLOSED_CONTEST_EVAL_SEPARATELY_OWED_NO_WRITE"


def test_g22_exact_custody_mismatch_refused() -> None:
    value = json.loads(_full_g22())
    value["archive_artifact"]["selected"]["member"]["sha256"] = _sha("wrong-member")
    with pytest.raises(Exception, match="selected member SHA"):
        FullN600ReceiverEqualityClosureV1.from_receipts(
            g20_receipt=G20_RECEIPT.read_bytes(),
            g22_receipt=_canonical(value),
            pair_population=_population(),
        )


def test_g25_rate_telemetry_requires_exact_g20_parent_and_keeps_video_blocker(
    tmp_path: Path,
) -> None:
    final_receipt = g18_fixture._receipt(tmp_path)
    population = _population()
    g25 = G25_RECEIPT.read_bytes()
    with pytest.raises(TaskspaceFeedbackCostateMaterializerError, match="exact G20 parent"):
        materialize_taskspace_feedback_costate_v1(
            final_receipt,
            population,
            g25_population_global_recode_receipt=g25,
        )

    result = materialize_taskspace_feedback_costate_v1(
        final_receipt,
        population,
        g20_placement_receipt=G20_RECEIPT.read_bytes(),
        g25_population_global_recode_receipt=g25,
    )
    custody = result["input_custody"]["g25_population_global_recode_receipt"]
    assert custody["sha256"] == hashlib.sha256(g25).hexdigest()
    recodes = result["g19_controller"]["record"]["population_global_representation_recodes"]
    assert recodes["conditional_best_archive_nbytes"] == 80_238
    assert recodes["rows"][0]["telemetry_admission"]["exact_score_endpoint_eligible"] is False


def test_g28_same_object_score_routes_pose_with_exact_g20_g22_parents(tmp_path: Path) -> None:
    final_receipt = g18_fixture._receipt(tmp_path)
    population = _population()
    g28 = G28_RECEIPT.read_bytes()
    with pytest.raises(TaskspaceFeedbackCostateMaterializerError, match="exact G20 placement and G22"):
        materialize_taskspace_feedback_costate_v1(
            final_receipt,
            population,
            g28_same_object_score_receipt=g28,
        )

    result = materialize_taskspace_feedback_costate_v1(
        final_receipt,
        population,
        g20_placement_receipt=G20_RECEIPT.read_bytes(),
        g22_receiver_equality_receipt=G22_FULL.read_bytes(),
        g25_population_global_recode_receipt=G25_RECEIPT.read_bytes(),
        g28_same_object_score_receipt=g28,
    )
    custody = result["input_custody"]["g28_same_object_score_receipt"]
    assert custody["sha256"] == hashlib.sha256(g28).hexdigest()
    controller = result["g19_controller"]["record"]
    rows = controller["same_object_score_measurements"]["rows"]
    assert len(rows) == 1
    row = rows[0]
    assert row["exact_object"]["realized_output_sha256"] == (
        "8565df10cbff8f86f02233fd20ececd74857a0d3806caf278a385a4d5421dcae"
    )
    assert row["measured_action"]["score"] == pytest.approx(36.09269518733842)
    assert row["conditional_population_global_rate_projection"]["archive_nbytes"] == 80_238
    assert row["conditional_population_global_rate_projection"]["score_endpoint_eligible"] is False
    assert row["controller_route"]["dominant_component"] == "pose"
    assert row["controller_route"]["next_phase"] == "MAXIMAL_INVERSE_POSE_PRESERVING_REPAIR"
    assert row["coupled_frontier_geometry"]["required_pose_reduction_fraction_at_zero_seg"] > 0.999
    assert row["coupled_frontier_geometry"]["required_seg_reduction_fraction_at_zero_pose"] > 0.66
    assert controller["truth"]["n600_evaluation"] is True
    handoff = next(
        item
        for item in controller["blocked_handoffs"]
        if item["consumer_api"] == "same_object_bridge_score_costate_ingest"
    )
    assert handoff["status"] == "READY_ADVISORY_COMPONENT_ROUTING_NO_PROMOTION_NO_WRITE"


def test_g28_score_arithmetic_drift_is_refused_even_with_rebound_identity(tmp_path: Path) -> None:
    value = json.loads(G28_RECEIPT.read_bytes())
    value["score_row"]["pose_component"] += 1.0
    value.pop("receipt_identity_sha256")
    value["receipt_identity_sha256"] = hashlib.sha256(_canonical(value).rstrip(b"\n")).hexdigest()
    with pytest.raises(Exception, match="Pose component"):
        materialize_taskspace_feedback_costate_v1(
            g18_fixture._receipt(tmp_path),
            _population(),
            g20_placement_receipt=G20_RECEIPT.read_bytes(),
            g22_receiver_equality_receipt=G22_FULL.read_bytes(),
            g28_same_object_score_receipt=_canonical(value),
        )


def test_pointer_refresh_does_not_invalidate_decode_and_target_change_only_rebases() -> None:
    closure = FullN600ReceiverEqualityClosureV1.from_receipts(
        g20_receipt=G20_RECEIPT.read_bytes(),
        g22_receipt=_full_g22(),
        pair_population=_population(),
    )
    metadata_refresh = ReceiverEqualityAdmissionControlV1(
        pointer_artifact_sha256=_sha("later-pointer-artifact"),
        pointer_artifact_nbytes=20_000,
        semantic_target_sha256=closure.semantic_target_end_sha256,
    )
    status = closure.admission_status(metadata_refresh)
    assert status == {
        "pointer_artifact_changed_since_proof": True,
        "semantic_target_changed_since_proof": False,
        "rebase_required_before_admission": False,
        "decode_equality_invalidated": False,
    }

    target_change = ReceiverEqualityAdmissionControlV1(
        pointer_artifact_sha256=_sha("later-semantic-pointer"),
        pointer_artifact_nbytes=20_100,
        semantic_target_sha256=_sha("new-semantic-target"),
    )
    status = closure.admission_status(target_change)
    assert status["semantic_target_changed_since_proof"] is True
    assert status["rebase_required_before_admission"] is True
    assert status["decode_equality_invalidated"] is False


def test_duplicate_safe_controller_state_through_materializer(tmp_path: Path) -> None:
    final_receipt = g18_fixture._receipt(tmp_path)
    population = _population()
    seed = materialize_taskspace_feedback_costate_v1(final_receipt, population)
    feedback = seed["g18_feedback"]["record"]
    binding = _binding_from_record(seed)
    predictions = g19_fixture._predictions(feedback, binding)
    first = materialize_taskspace_feedback_costate_v1(
        final_receipt,
        population,
        predictions=predictions,
    )
    second = materialize_taskspace_feedback_costate_v1(
        final_receipt,
        population,
        predictions=predictions,
        prior_controller_state=first["g19_controller"]["record"]["controller_state"],
    )
    first_state = first["g19_controller"]["record"]["controller_state"]
    second_state = second["g19_controller"]["record"]["controller_state"]
    assert first_state == second_state
    assert all(row["observation_count"] == 1 for row in second_state["proposals"])
    assert all(row["duplicate_observation"] is True for row in second["g19_controller"]["record"]["calibrations"])
