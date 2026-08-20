# SPDX-License-Identifier: MIT
"""Production-shaped, research-only G14 final-receipt feedback materializer.

The materializer is intentionally pure.  It accepts exact terminal G14 receipt
bytes and an exact serialized C0B ``PairPopulation`` envelope, derives every
identity itself, and emits the frozen G18 feedback plus G19 controller telemetry.
It never reads a live run directory, pointer, scorer, archive, or ledger.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any, Final

from tac.witness_control.taskspace_interaction_costate_bridge_v1 import (
    ActionEffectPredictionV1,
    FeedbackExactBaseBindingV1,
    FullN600ReceiverEqualityClosureV1,
    ReceiverEqualityAdmissionControlV1,
    bridge_receipt_bytes,
    build_taskspace_interaction_costate_bridge_v1,
)
from tac.witness_dsl.pair_population_envelope import PairDomain, PairPopulation
from tac.witness_dsl.taskspace_g8_a3_interaction_feedback import (
    build_taskspace_g8_a3_interaction_feedback,
    feedback_receipt_bytes,
)

SCHEMA: Final = "tac.taskspace_feedback_costate_materialization.v1"
PAIR_COUNT: Final = 600


class TaskspaceFeedbackCostateMaterializerError(ValueError):
    """Raised before output when terminal, population, or authority custody fails."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise TaskspaceFeedbackCostateMaterializerError("materialization value is not finite canonical JSON") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def materialization_receipt_bytes(value: Mapping[str, Any]) -> bytes:
    if type(value) is not dict or value.get("schema") != SCHEMA:
        raise TaskspaceFeedbackCostateMaterializerError("materialization record schema changed")
    return _canonical_json(value) + b"\n"


def _strict_final_receipt(payload: bytes) -> dict[str, Any]:
    if type(payload) is not bytes:
        raise TaskspaceFeedbackCostateMaterializerError("g14_final_receipt must be exact terminal receipt bytes")
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TaskspaceFeedbackCostateMaterializerError("G14 final_receipt is not JSON") from exc
    if type(value) is not dict or _canonical_json(value) + b"\n" != payload:
        raise TaskspaceFeedbackCostateMaterializerError("G14 final_receipt bytes are not canonical terminal bytes")
    try:
        from tools.run_taskspace_g8_a3_n2_allocator import parse_final_receipt

        parsed = parse_final_receipt(payload)
    except Exception as exc:
        raise TaskspaceFeedbackCostateMaterializerError(
            "G14 input is not a parser-accepted terminal final_receipt"
        ) from exc
    if parsed != value:
        raise TaskspaceFeedbackCostateMaterializerError("G14 terminal parser changed final_receipt semantics")
    return value


def _strict_population(payload: bytes, population_sha256_hint: str | None) -> PairPopulation:
    if type(payload) is not bytes:
        raise TaskspaceFeedbackCostateMaterializerError("pair_population must be exact serialized canonical bytes")
    try:
        population = PairPopulation.from_bytes(payload)
    except Exception as exc:
        raise TaskspaceFeedbackCostateMaterializerError(
            "pair_population is not a strict canonical PairPopulation envelope"
        ) from exc
    canonical_ids = tuple(range(PAIR_COUNT))
    if population.source_pair_ids != canonical_ids or any(
        population.domain_index(domain).local_to_source_pair_ids != canonical_ids for domain in PairDomain
    ):
        raise TaskspaceFeedbackCostateMaterializerError("pair_population must be exact ordered 0..599 in V9/PBR/IR/V10")
    if population_sha256_hint is not None and population_sha256_hint != population.population_sha256:
        raise TaskspaceFeedbackCostateMaterializerError(
            "caller population digest differs from reopened PairPopulation identity"
        )
    return population


def _binding(
    feedback: dict[str, Any], feedback_payload: bytes, population: PairPopulation
) -> FeedbackExactBaseBindingV1:
    occurrences = feedback["row_inventory"]["occurrences"]
    baseline = next(row["measurement"] for row in occurrences if row["occurrence_id"] == "g0:0")
    selected = feedback["row_inventory"]["selected_research_row"]
    custody = feedback["pointer_observation_custody"]
    source_pair_ids = tuple(feedback["four_way_z_t_h_partition"]["source_pair_ids"])
    if any(pair_id not in population.source_pair_ids for pair_id in source_pair_ids):
        raise TaskspaceFeedbackCostateMaterializerError("G14 source pairs are absent from reopened PairPopulation")
    return FeedbackExactBaseBindingV1(
        feedback_sha256=_sha256(feedback_payload),
        source_receipt_sha256=feedback["source_receipt_sha256"],
        canonical_receipt_sha256=feedback["canonical_receipt_sha256"],
        source_lane_id=feedback["source_lane_id"],
        axis=feedback["axis"],
        baseline_bundle_sha256=baseline["baseline_bundle_sha256"],
        baseline_archive_sha256=baseline["selected_archive_sha256"],
        baseline_archive_nbytes=baseline["selected_archive_bytes"],
        selected_research_archive_sha256=selected["selected_archive_sha256"],
        frozen_scorer_sha256=baseline["scorer_evidence"]["frozen_scorer_sha256"],
        pointer_manifest_sha256=custody["manifest_sha256"],
        pointer_start_sha256=custody["pointer_start"]["pointer_sha256"],
        pointer_start_target_score=custody["pointer_start"]["target_score"],
        pointer_latest_sha256=custody["pointer_latest"]["pointer_sha256"],
        pointer_latest_target_score=custody["pointer_latest"]["target_score"],
        population_manifest_sha256=population.population_sha256,
        source_pair_ids=source_pair_ids,
    )


def materialize_taskspace_feedback_costate_v1(
    g14_final_receipt: bytes,
    pair_population: bytes,
    *,
    population_sha256_hint: str | None = None,
    predictions: Sequence[ActionEffectPredictionV1] = (),
    prior_controller_state: Mapping[str, Any] | None = None,
    g20_placement_receipt: bytes | None = None,
    g22_receiver_equality_receipt: bytes | None = None,
    g25_population_global_recode_receipt: bytes | None = None,
    g28_same_object_score_receipt: bytes | None = None,
    current_placement_admission_control: ReceiverEqualityAdmissionControlV1 | None = None,
) -> dict[str, Any]:
    """Materialize G18/G19 from a terminal receipt; never inspect a live partial run."""

    _strict_final_receipt(g14_final_receipt)
    population = _strict_population(pair_population, population_sha256_hint)
    feedback = build_taskspace_g8_a3_interaction_feedback(g14_final_receipt)
    feedback_payload = feedback_receipt_bytes(feedback)
    binding = _binding(feedback, feedback_payload, population)

    placements: tuple[bytes, ...] = ()
    equalities: tuple[FullN600ReceiverEqualityClosureV1, ...] = ()
    controls: dict[str, ReceiverEqualityAdmissionControlV1] = {}
    if g22_receiver_equality_receipt is not None and g20_placement_receipt is None:
        raise TaskspaceFeedbackCostateMaterializerError(
            "G22 receiver equality cannot be consumed without exact matching G20 receipt bytes"
        )
    if g25_population_global_recode_receipt is not None and g20_placement_receipt is None:
        raise TaskspaceFeedbackCostateMaterializerError(
            "G25 population-global recode cannot be consumed without its exact G20 parent receipt bytes"
        )
    if g25_population_global_recode_receipt is not None and type(g25_population_global_recode_receipt) is not bytes:
        raise TaskspaceFeedbackCostateMaterializerError("G25 population-global recode receipt must be exact bytes")
    if g28_same_object_score_receipt is not None and (
        g20_placement_receipt is None or g22_receiver_equality_receipt is None
    ):
        raise TaskspaceFeedbackCostateMaterializerError(
            "G28 same-object score cannot be consumed without exact G20 placement and G22 equality parents"
        )
    if g28_same_object_score_receipt is not None and type(g28_same_object_score_receipt) is not bytes:
        raise TaskspaceFeedbackCostateMaterializerError("G28 same-object score receipt must be exact bytes")
    if g20_placement_receipt is not None:
        if type(g20_placement_receipt) is not bytes:
            raise TaskspaceFeedbackCostateMaterializerError("G20 placement receipt must be exact bytes")
        placements = (g20_placement_receipt,)
    if g22_receiver_equality_receipt is not None:
        try:
            closure = FullN600ReceiverEqualityClosureV1.from_receipts(
                g20_receipt=g20_placement_receipt,
                g22_receipt=g22_receiver_equality_receipt,
                pair_population=pair_population,
            )
        except Exception as exc:
            raise TaskspaceFeedbackCostateMaterializerError(
                "G22 receipt does not close exact full-n600 G20 receiver equality"
            ) from exc
        equalities = (closure,)
        if current_placement_admission_control is not None:
            if type(current_placement_admission_control) is not ReceiverEqualityAdmissionControlV1:
                raise TaskspaceFeedbackCostateMaterializerError(
                    "placement admission control must be exact typed control"
                )
            controls[closure.g22_receipt_sha256] = current_placement_admission_control
    elif current_placement_admission_control is not None:
        raise TaskspaceFeedbackCostateMaterializerError(
            "admission control cannot exist without a full-n600 G22 closure"
        )

    controller = build_taskspace_interaction_costate_bridge_v1(
        feedback_payload,
        binding=binding,
        predictions=predictions,
        representation_placement_receipts=placements,
        representation_placement_receiver_equalities=equalities,
        representation_placement_admission_controls=controls,
        population_global_recode_receipts=(
            () if g25_population_global_recode_receipt is None else (g25_population_global_recode_receipt,)
        ),
        same_object_bridge_score_receipts=(
            () if g28_same_object_score_receipt is None else (g28_same_object_score_receipt,)
        ),
        prior_controller_state=prior_controller_state,
    )
    controller_payload = bridge_receipt_bytes(controller)
    output = {
        "schema": SCHEMA,
        "input_custody": {
            "g14_final_receipt": {
                "schema": "tac.taskspace_g8_a3_n2_allocator.v1",
                "sha256": _sha256(g14_final_receipt),
                "bytes": len(g14_final_receipt),
                "terminal_parser_accepted": True,
                "live_partial_run_consumed": False,
            },
            "pair_population": {
                "schema": "tac.pair_population.envelope.v1",
                "envelope_sha256": _sha256(pair_population),
                "envelope_bytes": len(pair_population),
                "population_sha256": population.population_sha256,
                "pair_count": PAIR_COUNT,
                "source_pair_ids_sha256": _sha256(_canonical_json(list(population.source_pair_ids))),
                "identity_derived_from_reopened_bytes": True,
                "caller_digest_used_as_authority": False,
            },
            **(
                {}
                if g20_placement_receipt is None
                else {
                    "g20_placement_receipt": {
                        "schema": "tac.ep725_lossless_xcodec_recode.v1",
                        "sha256": _sha256(g20_placement_receipt),
                        "bytes": len(g20_placement_receipt),
                    }
                }
            ),
            **(
                {}
                if g22_receiver_equality_receipt is None
                else {
                    "g22_receiver_equality_receipt": {
                        "schema": "tac.g22_ep725_xcodec_decode_receipt.v1",
                        "sha256": _sha256(g22_receiver_equality_receipt),
                        "bytes": len(g22_receiver_equality_receipt),
                    }
                }
            ),
            **(
                {}
                if g25_population_global_recode_receipt is None
                else {
                    "g25_population_global_recode_receipt": {
                        "schema": "tac.ep725_population_global_recode.v2",
                        "sha256": _sha256(g25_population_global_recode_receipt),
                        "bytes": len(g25_population_global_recode_receipt),
                    }
                }
            ),
            **(
                {}
                if g28_same_object_score_receipt is None
                else {
                    "g28_same_object_score_receipt": {
                        "schema": "tac.taskspace_ep725_same_object_bridge_eval.v1",
                        "sha256": _sha256(g28_same_object_score_receipt),
                        "bytes": len(g28_same_object_score_receipt),
                    }
                }
            ),
        },
        "g18_feedback": {
            "sha256": _sha256(feedback_payload),
            "bytes": len(feedback_payload),
            "record": feedback,
        },
        "g19_controller": {
            "sha256": _sha256(controller_payload),
            "bytes": len(controller_payload),
            "record": controller,
        },
        "truth": {
            "research_only": True,
            "score_claim": False,
            "candidate_claim": False,
            "promotion_eligible": False,
            "authority_invented": False,
            "population_identity_invented": False,
            "scorer_eval_dispatch_or_pointer_write_performed": False,
            "live_partial_g14_consumed": False,
        },
    }
    round_trip = json.loads(materialization_receipt_bytes(output))
    if round_trip != output:
        raise TaskspaceFeedbackCostateMaterializerError("materialization canonical round trip changed output")
    return output


__all__ = [
    "SCHEMA",
    "TaskspaceFeedbackCostateMaterializerError",
    "materialization_receipt_bytes",
    "materialize_taskspace_feedback_costate_v1",
]
