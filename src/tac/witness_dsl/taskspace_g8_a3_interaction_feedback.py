# SPDX-License-Identifier: MIT
"""Dense-free, fail-closed feedback harvest for the frozen G14 n2 receipt.

This module is deliberately an adapter, not an experiment runner.  It reparses
one final G14 receipt, closes all cross-collection identities and arithmetic,
then emits deterministic payloads for the six canonical consumer hooks.  It
never reads a run directory, archive, scorer, pointer, or ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

from tac.boosting.pareto_front import ParetoFrontTracker
from tac.cathedral.consumer_contract import ConsumerTier, HookNumber
from tac.multi_granularity_sensitivity import per_pair_axis_score_contribution
from tac.witness_sensitivity_bitalloc import score_delta

SCHEMA: Final = "tac.taskspace_g8_a3_interaction_feedback.v1"
AXIS: Final = "[macOS-CPU advisory]"
_TARGET_RELATIVE_FIELDS: Final = frozenset(
    {"below_target", "gap_to_target", "target_sublevel_admission"}
)
_FOUR_WAY_SCHEMA: Final = "tac.same_class_realization_cell_partition_telemetry.v1"
_FOUR_WAY_DEFINITION: Final = (
    "closed_z_eq_t_h_eq_t__realization_z_eq_t_h_ne_t__"
    "topology_z_ne_t_h_ne_t__fortunate_z_ne_t_h_eq_t.v1"
)
_COUNT_FIELDS: Final = (
    "closed_cell_count",
    "realization_debt_cell_count",
    "topology_debt_cell_count",
    "fortunate_semantic_mismatch_cell_count",
)
_CLASS_COUNT_FIELDS: Final = (
    "closed_count_by_target_class",
    "realization_debt_count_by_target_class",
    "topology_debt_count_by_target_class",
    "fortunate_semantic_mismatch_count_by_target_class",
)
_ROW_FIELDS_WITHOUT_PATH: Final = frozenset({"selected_archive_path"})


class TaskspaceG8A3InteractionFeedbackError(ValueError):
    """Raised before output when any G14 signal or custody edge is incomplete."""


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
        raise TaskspaceG8A3InteractionFeedbackError("feedback input is not canonical-JSON-safe") from exc


def feedback_receipt_bytes(value: Mapping[str, Any]) -> bytes:
    """Return deterministic canonical bytes for a completed feedback record."""

    if type(value) is not dict or value.get("schema") != SCHEMA:
        raise TaskspaceG8A3InteractionFeedbackError("feedback record schema changed")
    return _canonical_json(value) + b"\n"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _load_g14(payload: bytes) -> dict[str, Any]:
    # The runner is the frozen schema owner.  Importing it performs no run and
    # keeps G18 from drifting into a second, permissive final-receipt parser.
    try:
        from tools.run_taskspace_g8_a3_n2_allocator import parse_final_receipt

        return parse_final_receipt(payload)
    except Exception as exc:
        raise TaskspaceG8A3InteractionFeedbackError("frozen G14 final-receipt validation failed") from exc


def _receipt_bytes(receipt: bytes | Mapping[str, Any]) -> tuple[bytes, bytes, dict[str, Any]]:
    if type(receipt) is bytes:
        source = receipt
    elif type(receipt) is dict:
        source = _canonical_json(receipt) + b"\n"
    else:
        raise TaskspaceG8A3InteractionFeedbackError("receipt must be exact bytes or a plain mapping")
    value = _load_g14(source)
    canonical = _canonical_json(value) + b"\n"
    # Reparse the canonical representation as a second independent closure.
    canonical_value = _load_g14(canonical)
    if canonical_value != value:
        raise TaskspaceG8A3InteractionFeedbackError("G14 canonical round trip changed the receipt")
    return source, canonical, value


def _forbid_target_relative_fields(value: object, *, path: str = "$") -> None:
    if type(value) is dict:
        forbidden = _TARGET_RELATIVE_FIELDS.intersection(value)
        if forbidden:
            raise TaskspaceG8A3InteractionFeedbackError(
                f"target-relative field forbidden at {path}: {sorted(forbidden)}"
            )
        for key, item in value.items():
            _forbid_target_relative_fields(item, path=f"{path}.{key}")
    elif type(value) is list:
        for index, item in enumerate(value):
            _forbid_target_relative_fields(item, path=f"{path}[{index}]")


def _exact_dict(value: object, fields: set[str], field: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        raise TaskspaceG8A3InteractionFeedbackError(f"{field} schema changed")
    return value


def _validate_four_way(g12: dict[str, Any]) -> dict[str, Any]:
    summary = g12.get("summary")
    if type(summary) is not dict:
        raise TaskspaceG8A3InteractionFeedbackError("G12 summary is not one object")
    partition = summary.get("canonical_four_way_z_t_h_partition")
    expected = {
        "source_pair_ids",
        "label_shape",
        "current_semantic_labels_sha256",
        "target_labels_sha256",
        "realized_labels_sha256",
        "closed_mask_sha256",
        "realization_debt_mask_sha256",
        "topology_debt_mask_sha256",
        "fortunate_semantic_mismatch_mask_sha256",
        *_COUNT_FIELDS,
        *_CLASS_COUNT_FIELDS,
        "schema",
        "exact_partition_definition",
        "dense_masks_serialized",
        "counts_by_target_class",
    }
    partition = _exact_dict(partition, expected, "canonical four-way Z/T/H partition")
    if (
        partition["schema"] != _FOUR_WAY_SCHEMA
        or partition["exact_partition_definition"] != _FOUR_WAY_DEFINITION
        or partition["dense_masks_serialized"] is not False
        or partition["counts_by_target_class"] is not True
    ):
        raise TaskspaceG8A3InteractionFeedbackError("four-way partition truth labels changed")
    pair_ids = partition["source_pair_ids"]
    shape = partition["label_shape"]
    if (
        type(pair_ids) is not list
        or not pair_ids
        or any(type(item) is not int or item < 0 for item in pair_ids)
        or len(set(pair_ids)) != len(pair_ids)
        or type(shape) is not list
        or len(shape) != 3
        or shape != [2, 384, 512]
        or shape[0] != len(pair_ids)
        or any(type(item) is not int or item < 1 for item in shape)
    ):
        raise TaskspaceG8A3InteractionFeedbackError("four-way source-pair/shape custody changed")
    total = math.prod(shape)
    counts: list[int] = []
    for count_field, class_field in zip(_COUNT_FIELDS, _CLASS_COUNT_FIELDS, strict=True):
        count = partition[count_field]
        classes = partition[class_field]
        if (
            type(count) is not int
            or count < 0
            or type(classes) is not list
            or len(classes) != 5
            or any(type(item) is not int or item < 0 for item in classes)
            or sum(classes) != count
            or summary.get(count_field) != count
        ):
            raise TaskspaceG8A3InteractionFeedbackError(f"four-way count does not close: {count_field}")
        counts.append(count)
    if sum(counts) != total:
        raise TaskspaceG8A3InteractionFeedbackError("four-way Z/T/H partition is not exhaustive")
    if summary.get("dense_labels_or_rgb_serialized") is not False:
        raise TaskspaceG8A3InteractionFeedbackError("G12 summary claims dense labels or RGB persisted")
    return partition


@dataclass(frozen=True, slots=True)
class _RowView:
    measurement_id: str
    baseline_bundle_sha256: str
    d_seg: float
    d_pose: float
    selected_archive_bytes: int

    @property
    def score(self) -> float:
        return 100.0 * self.d_seg + math.sqrt(10.0 * self.d_pose) + (
            25.0 * self.selected_archive_bytes / 37_545_489
        )


def _view(row: Mapping[str, Any]) -> _RowView:
    return _RowView(
        measurement_id=row["measurement_id"],
        baseline_bundle_sha256=row["baseline_bundle_sha256"],
        d_seg=row["d_seg"],
        d_pose=row["d_pose"],
        selected_archive_bytes=row["selected_archive_bytes"],
    )


def _transition(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    try:
        from tools.run_taskspace_g8_a3_n2_allocator import PairwiseTransitionV1

        result = PairwiseTransitionV1.between(_view(before), _view(after)).as_dict()
    except Exception as exc:
        raise TaskspaceG8A3InteractionFeedbackError("exact pairwise transition failed") from exc
    independent = score_delta(
        after["d_seg"],
        after["d_pose"],
        after["selected_archive_bytes"],
        before["d_seg"],
        before["d_pose"],
        before["selected_archive_bytes"],
    )
    if independent != result["exact_score_delta"]:
        raise TaskspaceG8A3InteractionFeedbackError("canonical exact-delta implementations disagree")
    return result


def _same_row_except_path(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return {key: value for key, value in left.items() if key not in _ROW_FIELDS_WITHOUT_PATH} == {
        key: value for key, value in right.items() if key not in _ROW_FIELDS_WITHOUT_PATH
    }


def _validate_g0(
    baseline: dict[str, Any], g0: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    programs = g0.get("programs")
    rows = g0.get("measurements")
    if type(programs) is not list or type(rows) is not list or not rows or len(programs) != len(rows):
        raise TaskspaceG8A3InteractionFeedbackError("G0 programs and rows are not complete one-to-one lists")
    if not _same_row_except_path(rows[0], baseline):
        raise TaskspaceG8A3InteractionFeedbackError("G0 first row is not the exact baseline")
    seen_programs: set[tuple[str, str]] = set()
    for index, (program, row) in enumerate(zip(programs, rows, strict=True)):
        program = _exact_dict(
            program,
            {
                "program_id",
                "program_sha256",
                "mode",
                "row_count",
                "acquisition_y1_sha256",
                "ranking_sha256",
            },
            f"G0 A program {index}",
        )
        identity = (program["program_id"], program["program_sha256"])
        if identity in seen_programs:
            raise TaskspaceG8A3InteractionFeedbackError("G0 A program identity repeats")
        seen_programs.add(identity)
        if (
            row["g8_program_sha256"] is not None
            or row["a_program_sha256"] != program["program_sha256"]
            or row["a_mode"] != program["mode"]
            or row["a_row_count"] != program["row_count"]
            or program["acquisition_y1_sha256"] != g0.get("acquisition_y1_sha256")
        ):
            raise TaskspaceG8A3InteractionFeedbackError("G0 A descriptor/measurement custody disagrees")
    if rows[0]["a_mode"] != "PASS_A_V1" or rows[0]["a_row_count"] != 0:
        raise TaskspaceG8A3InteractionFeedbackError("G0 baseline is not versioned PASS-A")
    return programs, rows


def _validate_g8(
    baseline: dict[str, Any], g12: dict[str, Any], g8: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str], set[str]]:
    branches = g8.get("branches")
    rows = g8.get("measurements")
    if type(branches) is not list or type(rows) is not list or not rows or len(branches) != len(rows):
        raise TaskspaceG8A3InteractionFeedbackError("G8 branches and rows are not complete one-to-one lists")
    allowed = g12.get("allowed_prefixes")
    if type(allowed) is not list or any(type(item) is not int or item < 1 for item in allowed):
        raise TaskspaceG8A3InteractionFeedbackError("G12 allowed-prefix universe changed")
    if g12.get("unique_screened_program_count") != len(branches):
        raise TaskspaceG8A3InteractionFeedbackError("G12 unique-screened count disagrees with G8 rows")
    if type(g12.get("proposal_count")) is not int or g12["proposal_count"] < len(branches):
        raise TaskspaceG8A3InteractionFeedbackError("G12 proposal count cannot cover screened rows")
    aliases = g12.get("aliases_by_program_sha256")
    if type(aliases) is not dict:
        raise TaskspaceG8A3InteractionFeedbackError("G12 alias custody changed")
    branch_ids: set[str] = set()
    program_shas: set[str] = set()
    runner_branches: list[Any] = []
    try:
        from tools.run_taskspace_g8_a3_n2_allocator import (
            G8BranchV1,
            nondominated_g8_indices,
            retained_g8_branch_ids,
        )

        for branch, row in zip(branches, rows, strict=True):
            branch = _exact_dict(
                branch,
                {
                    "proposal_id",
                    "program_sha256",
                    "family",
                    "prefix_order",
                    "palette_bound_per_class",
                    "prefix_cell_count",
                },
                "G8 branch",
            )
            if branch["proposal_id"] in branch_ids or branch["program_sha256"] in program_shas:
                raise TaskspaceG8A3InteractionFeedbackError("screened G8 branch/program repeats")
            branch_ids.add(branch["proposal_id"])
            program_shas.add(branch["program_sha256"])
            if (
                branch["prefix_cell_count"] not in allowed
                or row["g8_program_sha256"] != branch["program_sha256"]
                or row["a_mode"] != "PASS_A_V1"
                or row["a_row_count"] != 0
                or row["baseline_bundle_sha256"] != baseline["baseline_bundle_sha256"]
                or branch["proposal_id"] not in aliases.get(branch["program_sha256"], [])
            ):
                raise TaskspaceG8A3InteractionFeedbackError("G8 branch/row/alias custody disagrees")
            runner_branches.append(G8BranchV1(**branch, program=object()))
        row_views = [_view(row) for row in rows]
        expected_nondom = {branches[index]["proposal_id"] for index in nondominated_g8_indices(row_views)}
        expected_retained = set(retained_g8_branch_ids(runner_branches, row_views))
    except TaskspaceG8A3InteractionFeedbackError:
        raise
    except Exception as exc:
        raise TaskspaceG8A3InteractionFeedbackError("G8 Pareto/retention recomputation failed") from exc
    observed_nondom = g8.get("nondominated_proposal_ids")
    observed_retained = g8.get("retained_for_a_proposal_ids")
    if (
        type(observed_nondom) is not list
        or len(observed_nondom) != len(set(observed_nondom))
        or set(observed_nondom) != expected_nondom
        or type(observed_retained) is not list
        or len(observed_retained) != len(set(observed_retained))
        or set(observed_retained) != expected_retained
    ):
        raise TaskspaceG8A3InteractionFeedbackError("G8 nondominated/retained set changed")
    return branches, rows, expected_nondom, expected_retained


def _a_invariant(pass_row: dict[str, Any], variant: dict[str, Any]) -> None:
    equalities = (
        "baseline_bundle_sha256",
        "a_source_binding_sha256",
        "camera_y1_sha256",
        "candidate_seg_labels_sha256",
        "d_seg",
    )
    if variant["g8_program_sha256"] != pass_row["g8_program_sha256"] or any(
        variant[field] != pass_row[field] for field in equalities
    ):
        raise TaskspaceG8A3InteractionFeedbackError("conditional A changed G8 Y1/Seg invariant")
    left = pass_row["scorer_evidence"]
    right = variant["scorer_evidence"]
    for field in ("per_pair_d_seg", "frozen_scorer_sha256", "target_forward_receipt_sha256"):
        if left[field] != right[field]:
            raise TaskspaceG8A3InteractionFeedbackError("conditional A changed bounded Seg/scorer custody")


def _validate_treatments(
    baseline: dict[str, Any],
    branches: list[dict[str, Any]],
    g8_rows: list[dict[str, Any]],
    retained: set[str],
    treatments: object,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if type(treatments) is not list:
        raise TaskspaceG8A3InteractionFeedbackError("conditional treatments changed exact type")
    branch_by_id = {branch["proposal_id"]: branch for branch in branches}
    row_by_id = {branch["proposal_id"]: row for branch, row in zip(branches, g8_rows, strict=True)}
    seen: set[tuple[str, str, str]] = set()
    covered: set[str] = set()
    normalized: list[dict[str, Any]] = []
    interactions: list[dict[str, Any]] = []
    for index, treatment in enumerate(treatments):
        treatment = _exact_dict(
            treatment,
            {
                "g8_branch",
                "a_program",
                "g8_a_measurement",
                "g0_a_measurement",
                "transition_vs_g8_pass",
                "transition_vs_g0_pass",
                "interaction_I",
            },
            f"conditional treatment {index}",
        )
        branch = treatment["g8_branch"]
        if type(branch) is not dict or branch.get("proposal_id") not in branch_by_id:
            raise TaskspaceG8A3InteractionFeedbackError("conditional treatment refers to missing G8 branch")
        branch_id = branch["proposal_id"]
        if branch != branch_by_id[branch_id] or branch_id not in retained:
            raise TaskspaceG8A3InteractionFeedbackError("conditional treatment branch is not exact retained branch")
        program = _exact_dict(
            treatment["a_program"],
            {
                "program_id",
                "program_sha256",
                "mode",
                "row_count",
                "acquisition_y1_sha256",
                "ranking_sha256",
            },
            f"conditional A program {index}",
        )
        identity = (branch_id, program["program_id"], program["program_sha256"])
        if identity in seen:
            raise TaskspaceG8A3InteractionFeedbackError("conditional treatment identity repeats")
        seen.add(identity)
        covered.add(branch_id)
        g8_a = treatment["g8_a_measurement"]
        g0_a = treatment["g0_a_measurement"]
        pass_row = row_by_id[branch_id]
        if (
            program["mode"] == "PASS_A_V1"
            or program["acquisition_y1_sha256"] != pass_row["camera_y1_sha256"]
            or g8_a["g8_program_sha256"] != branch["program_sha256"]
            or g0_a["g8_program_sha256"] is not None
            or g8_a["a_program_sha256"] != program["program_sha256"]
            or g0_a["a_program_sha256"] != program["program_sha256"]
            or g8_a["a_mode"] != program["mode"]
            or g0_a["a_mode"] != program["mode"]
            or g8_a["a_row_count"] != program["row_count"]
            or g0_a["a_row_count"] != program["row_count"]
            or g0_a["a_source_binding_sha256"] != baseline["a_source_binding_sha256"]
            or g0_a["a_source_binding_sha256"] == g8_a["a_source_binding_sha256"]
        ):
            raise TaskspaceG8A3InteractionFeedbackError("conditional treatment program/row custody disagrees")
        _a_invariant(pass_row, g8_a)
        transition_g8 = _transition(pass_row, g8_a)
        transition_g0 = _transition(baseline, g0_a)
        if treatment["transition_vs_g8_pass"] != transition_g8:
            raise TaskspaceG8A3InteractionFeedbackError("stored G8+A transition differs from exact recomputation")
        if treatment["transition_vs_g0_pass"] != transition_g0:
            raise TaskspaceG8A3InteractionFeedbackError("stored matched G0+A transition differs from exact recomputation")
        interaction = g8_a["derived_component_total"] - pass_row["derived_component_total"] - g0_a[
            "derived_component_total"
        ] + baseline["derived_component_total"]
        if not math.isfinite(interaction) or treatment["interaction_I"] != interaction:
            raise TaskspaceG8A3InteractionFeedbackError("G-by-A interaction differs from exact recomputation")
        treatment_id = f"treatment:{index}:{branch_id}:{program['program_id']}"
        normalized.append({**treatment, "treatment_id": treatment_id})
        interactions.append(
            {
                "interaction_id": f"interaction:{index}",
                "treatment_id": treatment_id,
                "g8_proposal_id": branch_id,
                "a_program_id": program["program_id"],
                "a_program_sha256": program["program_sha256"],
                "family": branch["family"],
                "prefix_order": branch["prefix_order"],
                "palette_bound_per_class": branch["palette_bound_per_class"],
                "prefix_cell_count": branch["prefix_cell_count"],
                "interaction_I": interaction,
                "definition": "S_g8_a_minus_S_g8_pass_minus_S_g0_a_plus_S_g0_pass.v1",
            }
        )
    if covered != retained:
        raise TaskspaceG8A3InteractionFeedbackError("retained G8 branches lack complete conditional-A treatment")
    return normalized, interactions


def _row_occurrences(
    g0_rows: Sequence[dict[str, Any]],
    branches: Sequence[dict[str, Any]],
    g8_rows: Sequence[dict[str, Any]],
    treatments: Sequence[dict[str, Any]],
    nondominated: set[str],
    retained: set[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for index, row in enumerate(g0_rows):
        out.append({"occurrence_id": f"g0:{index}", "kind": "g0_a", "measurement": row})
    for index, (branch, row) in enumerate(zip(branches, g8_rows, strict=True)):
        out.append(
            {
                "occurrence_id": f"g8:{index}",
                "kind": "g8_pass_a",
                "g8_branch": branch,
                "nondominated_3axis": branch["proposal_id"] in nondominated,
                "retained_for_a": branch["proposal_id"] in retained,
                "measurement": row,
            }
        )
    for index, treatment in enumerate(treatments):
        coordinates = {
            "g8_branch": treatment["g8_branch"],
            "a_program": treatment["a_program"],
            "treatment_id": treatment["treatment_id"],
        }
        out.append(
            {
                "occurrence_id": f"g8_a:{index}",
                "kind": "g8_a",
                **coordinates,
                "measurement": treatment["g8_a_measurement"],
            }
        )
        out.append(
            {
                "occurrence_id": f"matched_g0_a:{index}",
                "kind": "matched_g0_a",
                **coordinates,
                "measurement": treatment["g0_a_measurement"],
            }
        )
    expected = len(g0_rows) + len(g8_rows) + 2 * len(treatments)
    if len(out) != expected or len({item["occurrence_id"] for item in out}) != expected:
        raise TaskspaceG8A3InteractionFeedbackError("production row occurrence was dropped")
    return out


def _transition_records(
    baseline: dict[str, Any],
    g0_rows: Sequence[dict[str, Any]],
    branches: Sequence[dict[str, Any]],
    g8_rows: Sequence[dict[str, Any]],
    treatments: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def add(kind: str, before: dict[str, Any], after: dict[str, Any], context: dict[str, Any]) -> None:
        exact = _transition(before, after)
        records.append(
            {
                "transition_id": f"transition:{len(records)}",
                "kind": kind,
                **context,
                **exact,
            }
        )

    for index, row in enumerate(g0_rows[1:], start=1):
        add("g0_pass_to_g0_a", baseline, row, {"after_occurrence_id": f"g0:{index}"})
    for index, (branch, row) in enumerate(zip(branches, g8_rows, strict=True)):
        add(
            "g0_pass_to_g8_pass",
            baseline,
            row,
            {"after_occurrence_id": f"g8:{index}", "g8_proposal_id": branch["proposal_id"]},
        )
    for index, treatment in enumerate(treatments):
        branch_id = treatment["g8_branch"]["proposal_id"]
        pass_index = next(i for i, branch in enumerate(branches) if branch["proposal_id"] == branch_id)
        add(
            "g8_pass_to_g8_a",
            g8_rows[pass_index],
            treatment["g8_a_measurement"],
            {
                "after_occurrence_id": f"g8_a:{index}",
                "treatment_id": treatment["treatment_id"],
                "g8_proposal_id": branch_id,
            },
        )
        add(
            "g0_pass_to_matched_g0_a",
            baseline,
            treatment["g0_a_measurement"],
            {
                "after_occurrence_id": f"matched_g0_a:{index}",
                "treatment_id": treatment["treatment_id"],
                "g8_proposal_id": branch_id,
            },
        )
    return records


def _sensitivity_payload(occurrences: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for occurrence in occurrences:
        measurement = occurrence["measurement"]
        evidence = measurement["scorer_evidence"]
        ranked = [
            asdict(item)
            for item in per_pair_axis_score_contribution(
                evidence["per_pair_d_seg"], evidence["per_pair_d_pose"]
            )
        ]
        rows.append(
            {
                "occurrence_id": occurrence["occurrence_id"],
                "measurement_id": measurement["measurement_id"],
                "measurement_receipt_sha256": evidence["measurement_receipt_sha256"],
                "candidate_forward_receipt_sha256": evidence["candidate_forward_receipt_sha256"],
                "candidate_pose6_sha256": evidence["candidate_pose6_sha256"],
                "sample_count": evidence["sample_count"],
                "pair_rank_rows": ranked,
            }
        )
    return {
        "api": "tac.multi_granularity_sensitivity.per_pair_axis_score_contribution",
        "semantics": "n2_pair_ranking_only_pose_magnitudes_are_not_additive.v1",
        "sensitivity_map_api_compatible": False,
        "sensitivity_map_blocker": (
            "tac.sensitivity_map.SensitivityMap requires model-channel gradient/CUDA custody; "
            "G14 supplies n2 macOS whole-object finite differences"
        ),
        "rows": rows,
    }


def _pareto_payload(occurrences: Sequence[dict[str, Any]], g8: dict[str, Any]) -> dict[str, Any]:
    tracker = ParetoFrontTracker(axis=AXIS)
    for occurrence in occurrences:
        row = occurrence["measurement"]
        tracker.track_anchor(
            rate=float(row["selected_archive_bytes"]),
            distortion=100.0 * row["d_seg"] + math.sqrt(10.0 * row["d_pose"]),
            source=occurrence["occurrence_id"],
        )
    pareto = tracker.pareto_optimal_anchors()
    return {
        "api": "tac.boosting.pareto_front.ParetoFrontTracker",
        "axis": AXIS,
        "history": tracker.to_dict(),
        "nondominated_2axis_occurrence_ids": [anchor.source for anchor in pareto],
        "g14_nondominated_3axis_g8_proposal_ids": list(g8["nondominated_proposal_ids"]),
        "g14_retained_for_a_proposal_ids": list(g8["retained_for_a_proposal_ids"]),
    }


def _hook_payloads(
    occurrences: Sequence[dict[str, Any]],
    transitions: Sequence[dict[str, Any]],
    interactions: Sequence[dict[str, Any]],
    sensitivity: dict[str, Any],
    pareto: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    transition_ids = [row["transition_id"] for row in transitions]
    interaction_ids = [row["interaction_id"] for row in interactions]
    bit_allocator = {
        "api": "tac.witness_sensitivity_bitalloc.score_delta",
        "kind": "measured_atomic_whole_object_marginals_not_precision_ladders.v1",
        "marginal_basis": "each_transition_is_bound_to_its_exact_measured_base_archive.v1",
        "allocation_executed": False,
        "no_arbitrary_budget_introduced": True,
        "additive_independence_assumed": False,
        "dynamic_archive_cost_preserved": True,
        "mutually_exclusive_or_composed_actions_require_bundle_remeasurement": True,
        "interaction_policy": (
            "use measured G-by-A interactions for ranking context; never sum atomic marginals "
            "as a substitute for byte-closed bundle acquisition"
        ),
        "transition_ids": transition_ids,
        "interaction_ids": interaction_ids,
        "marginals": [
            {
                "transition_id": row["transition_id"],
                "delta_selected_archive_bytes": row["delta_selected_archive_bytes"],
                "delta_d_seg": row["delta_d_seg"],
                "delta_d_pose": row["delta_d_pose"],
                "distortion_term_delta": row["distortion_term_delta"],
                "exact_score_delta": row["exact_score_delta"],
                "finite_byte_ceiling_real": row["finite_byte_ceiling_real"],
                "greatest_strict_integer_byte_delta": row["greatest_strict_integer_byte_delta"],
            }
            for row in transitions
        ],
    }
    autopilot = {
        "contract": "tac.cathedral.consumer_contract",
        "tier": ConsumerTier.TIER_A_OBSERVABILITY_ONLY.name,
        "axis_tag": AXIS,
        "predicted_delta_adjustment": 0.0,
        "promotable": False,
        "dispatch_ready": False,
        "transition_ids": transition_ids,
        "interaction_ids": interaction_ids,
        "candidates": [
            {
                "candidate_id": row["transition_id"],
                "measured_exact_score_delta": row["exact_score_delta"],
                "measured_improves_common_n2_basis": row["improves_score"],
                "finite_byte_ceiling_real": row["finite_byte_ceiling_real"],
            }
            for row in transitions
        ],
    }
    probes = {
        "api": "tac.probe_outcomes_ledger.register_probe_outcome",
        "append_performed": False,
        "root_review_required": True,
        "payloads": [
            {
                "probe_id": row["transition_id"],
                "substrate": "macos_cpu_n2_advisory",
                "recipe_path": None,
                "probe_kind": "g14_whole_object_transition",
                "verdict": "PARTIAL",
                "metric_name": "exact_score_delta_common_n2_basis",
                "metric_value": row["exact_score_delta"],
                "threshold": 0.0,
                "threshold_token": "whole_object_exact_score_delta_lt_zero",
                "evidence_path": None,
                "next_action": "Use this advisory marginal to compose the next complete n600-capable row; do not promote it.",
                "reactivation_criteria": ["A complete byte-closed n600-capable composition is ready for exact replay."],
                "blocker_status": "advisory",
                "notes": "G18 emits kwargs only; root appends after real G14 receipt review.",
                "axis": AXIS,
                "promotion_eligible": False,
            }
            for row in transitions
        ]
        + [
            {
                "probe_id": row["interaction_id"],
                "substrate": "macos_cpu_n2_advisory",
                "recipe_path": None,
                "probe_kind": "g14_g_by_a_interaction",
                "verdict": "PARTIAL",
                "metric_name": "nonlinear_interaction_I_common_n2_basis",
                "metric_value": row["interaction_I"],
                "threshold": 0.0,
                "threshold_token": "descriptive_zero_interaction_reference_not_admission_gate",
                "evidence_path": None,
                "next_action": "Preserve the measured interaction in composition ranking; do not infer n600 efficacy.",
                "reactivation_criteria": ["The same composed mechanism has complete n600 byte-closed evidence."],
                "blocker_status": "advisory",
                "notes": "Interaction sign is descriptive; no family kill or promotion is authorized.",
                "axis": AXIS,
                "promotion_eligible": False,
            }
            for row in interactions
        ],
    }
    posterior = {
        "api": "tac.continual_learning.posterior_update",
        "compatible": False,
        "update_performed": False,
        "refused_class": "macos_substrate",
        "reason": "G14 is n2 macOS advisory and cannot form an authoritative ContestResult.",
        "preserved_occurrence_ids": [row["occurrence_id"] for row in occurrences],
        "preserved_interaction_ids": interaction_ids,
        "honest_persistence_surface": "deferred probe-outcome payloads pending root review",
    }
    hooks = {
        str(int(HookNumber.SENSITIVITY_MAP)): {
            "hook": HookNumber.SENSITIVITY_MAP.name,
            "status": "pair_ranking_active_gradient_map_incompatible",
            "occurrence_ids": [row["occurrence_id"] for row in occurrences],
            "payload_api": sensitivity["api"],
        },
        str(int(HookNumber.PARETO_CONSTRAINT)): {
            "hook": HookNumber.PARETO_CONSTRAINT.name,
            "status": "active_advisory_axis_only",
            "occurrence_ids": [row["occurrence_id"] for row in occurrences],
            "payload_api": pareto["api"],
        },
        str(int(HookNumber.BIT_ALLOCATOR)): {
            "hook": HookNumber.BIT_ALLOCATOR.name,
            "status": "measured_marginals_only",
            "transition_ids": transition_ids,
            "interaction_ids": interaction_ids,
            "payload_api": bit_allocator["api"],
        },
        str(int(HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH)): {
            "hook": HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH.name,
            "status": "tier_a_observability_only_no_dispatch",
            "transition_ids": transition_ids,
            "interaction_ids": interaction_ids,
            "payload_api": autopilot["contract"],
        },
        str(int(HookNumber.CONTINUAL_LEARNING_POSTERIOR)): {
            "hook": HookNumber.CONTINUAL_LEARNING_POSTERIOR.name,
            "status": "typed_incompatible_no_update",
            "interaction_ids": interaction_ids,
            "payload_api": posterior["api"],
        },
        str(int(HookNumber.PROBE_DISAMBIGUATOR)): {
            "hook": HookNumber.PROBE_DISAMBIGUATOR.name,
            "status": "deferred_kwargs_emitted_no_append",
            "transition_ids": transition_ids,
            "interaction_ids": interaction_ids,
            "payload_api": probes["api"],
        },
    }
    payloads = {
        "sensitivity": sensitivity,
        "pareto": pareto,
        "bit_allocator": bit_allocator,
        "autopilot": autopilot,
        "probe_outcomes": probes,
        "continual_learning": posterior,
    }
    return payloads, hooks


def _validate_hook_closure(
    hooks: dict[str, Any], transitions: Sequence[dict[str, Any]], interactions: Sequence[dict[str, Any]]
) -> None:
    if set(hooks) != {"1", "2", "3", "4", "5", "6"}:
        raise TaskspaceG8A3InteractionFeedbackError("six-hook inventory is incomplete")
    transition_ids = {row["transition_id"] for row in transitions}
    interaction_ids = {row["interaction_id"] for row in interactions}
    for number in ("3", "4", "6"):
        if set(hooks[number]["transition_ids"]) != transition_ids:
            raise TaskspaceG8A3InteractionFeedbackError("transition signal became orphaned from a consumer hook")
    for number in ("3", "4", "5", "6"):
        if set(hooks[number]["interaction_ids"]) != interaction_ids:
            raise TaskspaceG8A3InteractionFeedbackError("interaction signal became orphaned from a consumer hook")


def build_taskspace_g8_a3_interaction_feedback(
    receipt: bytes | Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and harvest every G14 row/interaction into closed hook payloads."""

    source, canonical, value = _receipt_bytes(receipt)
    _forbid_target_relative_fields(value)
    partition = _validate_four_way(value["g12"])
    baseline = value["baseline"]
    _programs, g0_rows = _validate_g0(baseline, value["g0_a_acquisition"])
    branches, g8_rows, nondominated, retained = _validate_g8(
        baseline, value["g12"], value["g8_screen"]
    )
    treatments, interactions = _validate_treatments(
        baseline,
        branches,
        g8_rows,
        retained,
        value["conditional_a_treatments"],
    )
    occurrences = _row_occurrences(g0_rows, branches, g8_rows, treatments, nondominated, retained)
    transitions = _transition_records(baseline, g0_rows, branches, g8_rows, treatments)
    expected_transition_count = max(0, len(g0_rows) - 1) + len(g8_rows) + 2 * len(treatments)
    if len(transitions) != expected_transition_count:
        raise TaskspaceG8A3InteractionFeedbackError("exact transition inventory dropped a row")
    selected = min(
        (item["measurement"] for item in occurrences),
        key=lambda row: (row["derived_component_total"], row["measurement_id"]),
    )
    if not _same_row_except_path(selected, value["selected_research_row"]):
        raise TaskspaceG8A3InteractionFeedbackError("selected research row differs from complete-row minimum")
    sensitivity = _sensitivity_payload(occurrences)
    pareto = _pareto_payload(occurrences, value["g8_screen"])
    downstream, hooks = _hook_payloads(occurrences, transitions, interactions, sensitivity, pareto)
    _validate_hook_closure(hooks, transitions, interactions)
    output = {
        "schema": SCHEMA,
        "source_receipt_sha256": _sha256(source),
        "canonical_receipt_sha256": _sha256(canonical),
        "source_schema": value["schema"],
        "source_lane_id": value["lane_id"],
        "axis": AXIS,
        "truth": value["truth"],
        "latest_pointer_comparison": value["latest_pointer_comparison"],
        "pointer_observation_custody": {
            "pointer_start": value["pointer_start"],
            "pointer_latest": value["pointer_latest"],
            "manifest_sha256": value["manifest_sha256"],
            "pointer_observation_paths": value["pointer_observation_paths"],
            "pointer_mutation_performed": False,
        },
        "four_way_z_t_h_partition": partition,
        "g12_acquisition": value["g12"],
        "diagnostic_exact_semantic_g_control": {
            "measurement": value["exact_semantic_g_control"],
            "selection_exclusion": value["diagnostic_controls"],
        },
        "row_inventory": {
            "occurrence_count": len(occurrences),
            "unique_measurement_id_count": len(
                {item["measurement"]["measurement_id"] for item in occurrences}
            ),
            "occurrences": occurrences,
            "selected_research_row": value["selected_research_row"],
        },
        "transition_inventory": {
            "transition_count": len(transitions),
            "transitions": transitions,
        },
        "interaction_inventory": {
            "interaction_count": len(interactions),
            "interactions": interactions,
        },
        "downstream_payloads": downstream,
        "hook_coverage": hooks,
        "dense_frames_or_masks_serialized": False,
        "live_run_scorer_eval_or_ledger_access_performed": False,
    }
    # Final round-trip is part of construction; it also rejects NaN/Inf.
    round_trip = json.loads(feedback_receipt_bytes(output).decode("ascii"))
    if round_trip != output:
        raise TaskspaceG8A3InteractionFeedbackError("feedback canonical round trip changed the record")
    return output


__all__ = [
    "AXIS",
    "SCHEMA",
    "TaskspaceG8A3InteractionFeedbackError",
    "build_taskspace_g8_a3_interaction_feedback",
    "feedback_receipt_bytes",
]
