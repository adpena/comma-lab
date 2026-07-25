# SPDX-License-Identifier: MIT
"""SHA-bound layer pricing and residual/context audit for DDM #669(b+c).

The compiler consumes already-materialized C1, G4, DM1, DM2, DM4, PF, and
MS7 receipts.  It does not rerun scorers or coder races.  Its purpose is to
separate three accounting states that the original C1 planning table mixed:

* exact receiver-owned bytes that are already measured;
* semantic/L3 prices that are measured but are not an admissible C1 archive
  allocation; and
* planning reserves whose marginal value is still unmeasured.

Generic receiver/context implementations remain free under rule 118.  Only
video-derived parameters and innovations are counted.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import Any

SCHEMA = "ddm_lp1_layer_pricing.v1"
CONFIG_SCHEMA = "ddm_lp1_layer_pricing_config.v1"
CONTEXT_RACE_SCHEMA = "ddm_lp1_context_race.v1"
C1_ROW_SCHEMA = "ddm_lp1_c1_stream_home.v1"
SENSE_ROW_SCHEMA = "ddm_lp1_costate_sense_row.v1"
LANE_ID = "lane_ddm_lp1_layer_pricing_669bc_20260725"
RATE_DUAL_NUMERATOR = 25
RATE_DUAL_DENOMINATOR = 37_545_489
REPO = Path(__file__).resolve().parents[3]


class LayerPricingError(ValueError):
    """A source-custody, layer-home, or accounting invariant failed."""


class LP1Type(StrEnum):
    """The five types sealed by the #669(b+c) authority."""

    GAUGE = "GAUGE"
    FIBER = "FIBER"
    RESIDUAL = "RESIDUAL"
    CONTEXT = "CONTEXT"
    PROGRAM = "PROGRAM"


class LP1Layer(StrEnum):
    """Scorer-recursion layers below the L5 verdict."""

    L1_PROGRAM = "L1_program"
    L2_CHART = "L2_chart_grammar"
    L3_RGB = "L3_RGB_realization"
    L4_SCORER_FEATURE = "L4_scorer_feature"


@dataclass(frozen=True)
class StreamHomePolicy:
    """Sealed interpretation for one C1 budget row."""

    stream_type: LP1Type | None
    layer_home: LP1Layer | None
    stratum: str
    disposition: str
    survival_authority: str
    counted_in_corrected_total: bool


_POLICIES: dict[str, StreamHomePolicy] = {
    "v15_predictor_zip_outer_home": StreamHomePolicy(
        LP1Type.CONTEXT,
        LP1Layer.L2_CHART,
        "all_roles_plus_xi",
        "KEEP_MEASURED_HOME",
        "MEASURED_COMPOSED_N600_FORWARD_CHAIN_AND_EXACT_MEMBER_HOME",
        True,
    ),
    "g1_movable_worldsheet_outer_home": StreamHomePolicy(
        LP1Type.CONTEXT,
        LP1Layer.L2_CHART,
        "movable_worldsheet",
        "KEEP_MEASURED_HOME",
        "MEASURED_COMPOSED_N600_FORWARD_CHAIN_AND_EXACT_MEMBER_HOME",
        True,
    ),
    "receiver_realization_profile": StreamHomePolicy(
        LP1Type.PROGRAM,
        LP1Layer.L1_PROGRAM,
        "receiver_profile",
        "KEEP_COUNTED_VIDEO_DERIVED_PROFILE_GENERIC_RECEIVER_FREE",
        "MEASURED_EXACT_MEMBER_HOME_AND_RECEIVER_PARSEBACK",
        True,
    ),
    "solved_template_outer_home": StreamHomePolicy(
        LP1Type.FIBER,
        LP1Layer.L4_SCORER_FEATURE,
        "shared_scorer_template",
        "KEEP_MEASURED_HOME",
        "MEASURED_L4_TO_L3_TO_EXACT_R_N600_FORWARD_CHAIN",
        True,
    ),
    "manifest": StreamHomePolicy(
        LP1Type.PROGRAM,
        LP1Layer.L1_PROGRAM,
        "container_manifest",
        "KEEP_CONTAINER_RATE_HOME",
        "MEASURED_EXACT_MEMBER_HOME_AND_PARSEBACK_ONLY",
        True,
    ),
    "central_directory_and_eocd": StreamHomePolicy(
        LP1Type.PROGRAM,
        LP1Layer.L1_PROGRAM,
        "container_framing",
        "KEEP_CONTAINER_RATE_HOME",
        "MEASURED_EXACT_CONTAINER_HOME_AND_PARSEBACK_ONLY",
        True,
    ),
    "v15_exact_control_subtotal": StreamHomePolicy(
        None,
        None,
        "accounting",
        "RECOMPUTE_ACCOUNTING_ONLY_DO_NOT_DOUBLE_CHARGE",
        "DERIVED_FROM_SIX_EXACT_MEMBER_HOMES",
        False,
    ),
    "lane_program_seed": StreamHomePolicy(
        LP1Type.CONTEXT,
        LP1Layer.L2_CHART,
        "lane_production",
        "KEEP_MEASURED_HOME",
        "MEASURED_EXACT_RECEIVER_OWNING_DELTA",
        True,
    ),
    "contextual_bounded_collateral_shared_application_stage_reserve": (
        StreamHomePolicy(
            LP1Type.RESIDUAL,
            LP1Layer.L4_SCORER_FEATURE,
            "lane_movable_exception",
            "ZERO_ALLOCATE_UNTIL_RECEIVER_CLOSED_MARGINAL_PAYS_RATE_DUAL",
            "DM1_L4_AND_DM4_L3_PRICES_MEASURED_MS7_R0_NONADMITTED",
            False,
        )
    ),
    "v18b_first_exact_pricing_rung_reserve": StreamHomePolicy(
        LP1Type.RESIDUAL,
        LP1Layer.L4_SCORER_FEATURE,
        "all_role_generated_columns",
        "ZERO_ALLOCATE_COMPUTABLE_NOT_YET_COMPUTED",
        "NO_EXACT_COMPOSED_MARGINAL_IN_SOURCE_C1",
        False,
    ),
    "j3_finish_and_xi_refinement_reserve": StreamHomePolicy(
        LP1Type.RESIDUAL,
        LP1Layer.L4_SCORER_FEATURE,
        "all_role_plus_pose_finish",
        "ZERO_ALLOCATE_COMPUTABLE_NOT_YET_COMPUTED",
        "NO_EXACT_COMPOSED_MARGINAL_IN_SOURCE_C1",
        False,
    ),
    "final_coder_and_container_contingency": StreamHomePolicy(
        LP1Type.PROGRAM,
        LP1Layer.L1_PROGRAM,
        "container_contingency",
        "ZERO_ALLOCATE_ACCOUNTING_RESERVE_CC2_OWNS_CODER_SELECTION",
        "NO_SAME_STREAM_FINAL_CONTAINER_PRICE_IN_SOURCE_C1",
        False,
    ),
    "hard_total": StreamHomePolicy(
        None,
        None,
        "accounting",
        "PRESERVE_CEILING_RECOMPUTE_ALLOCATED_BYTES",
        "DERIVED_PLANNING_CEILING_ONLY",
        False,
    ),
}


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic repository JSON bytes."""

    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_file(path: Path) -> str:
    """Hash one bounded source artifact."""

    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LayerPricingError(f"{label} must be a positive exact integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LayerPricingError(f"{label} must be a nonnegative exact integer")
    return value


def _checked_json(path: Path, expected_sha256: str) -> Mapping[str, Any]:
    if not path.is_file():
        raise LayerPricingError(f"bound source is absent: {path}")
    if sha256_file(path) != expected_sha256:
        raise LayerPricingError(f"bound source SHA-256 differs: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, Mapping):
        raise LayerPricingError(f"bound source is not a JSON object: {path}")
    return value


def _false_authority(value: Mapping[str, Any], label: str) -> None:
    required = {
        "research_only": True,
        "score_claim": False,
        "pointer_moved": False,
    }
    for key, expected in required.items():
        if value.get(key) != expected:
            raise LayerPricingError(f"{label}.{key} must equal {expected!r}")


def _load_sources(
    config: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, Any]]]:
    rows = config.get("sources")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise LayerPricingError("config.sources must be an ordered sequence")
    sources: dict[str, Mapping[str, Any]] = {}
    custody: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != {
            "id",
            "path",
            "schema",
            "sha256",
        }:
            raise LayerPricingError(f"config.sources[{index}] keys differ")
        source_id = row["id"]
        if not isinstance(source_id, str) or not source_id or source_id in sources:
            raise LayerPricingError("source ids must be nonempty and unique")
        relative = Path(row["path"])
        if relative.is_absolute():
            raise LayerPricingError("source paths must be repository-relative")
        path = (REPO / relative).resolve()
        if not path.is_relative_to(REPO):
            raise LayerPricingError("source path escapes the repository")
        value = _checked_json(path, row["sha256"])
        if value.get("schema") != row["schema"]:
            raise LayerPricingError(f"{source_id} schema differs")
        sources[source_id] = value
        custody.append(
            {
                "id": source_id,
                "path": relative.as_posix(),
                "schema": row["schema"],
                "sha256": row["sha256"],
            }
        )
    required = {
        "c1",
        "g4",
        "dm1",
        "dm2",
        "dm4",
        "pf1",
        "pf2",
        "ms5",
        "ms6",
        "ms7",
        "ms7_r0",
    }
    if set(sources) != required:
        raise LayerPricingError("config source ids differ from the sealed LP1 set")
    for source_id in ("c1", "g4", "dm1", "dm2", "dm4", "pf1", "pf2", "ms7"):
        _false_authority(sources[source_id], source_id)
    return sources, custody


def _validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "main_review_required": True,
        "lane_id": LANE_ID,
        "rate_dual_numerator": RATE_DUAL_NUMERATOR,
        "rate_dual_denominator": RATE_DUAL_DENOMINATOR,
        "coder_choice_owner": "ddm_cc2_coder_races",
    }
    if config.get("schema") != CONFIG_SCHEMA:
        raise LayerPricingError("config schema differs")
    for key, expected in required.items():
        if config.get(key) != expected:
            raise LayerPricingError(f"config {key} must equal {expected!r}")


def _context_race(
    *,
    race_id: str,
    stratum: str,
    explicit_bytes: int,
    contextual_bytes: int,
    counted_context_parameter_bytes: int,
    contextual_type: LP1Type,
    contextual_layer: LP1Layer,
    same_object: bool,
    scope: str,
) -> dict[str, Any]:
    explicit = _positive_int(explicit_bytes, f"{race_id}.explicit_bytes")
    contextual = _positive_int(contextual_bytes, f"{race_id}.contextual_bytes")
    parameters = _nonnegative_int(
        counted_context_parameter_bytes,
        f"{race_id}.counted_context_parameter_bytes",
    )
    contextual_total = contextual + parameters
    savings = explicit - contextual_total
    return {
        "schema": CONTEXT_RACE_SCHEMA,
        "race_id": race_id,
        "stratum": stratum,
        "explicit_exception_bytes": explicit,
        "contextual_innovation_bytes": contextual,
        "counted_context_parameter_bytes": parameters,
        "generic_context_decoder_bytes": 0,
        "contextual_total_counted_bytes": contextual_total,
        "savings_bytes": savings,
        "same_semantic_object": same_object,
        "typed_context_home": {
            "type": contextual_type.value,
            "layer_home": contextual_layer.value,
        },
        "disposition": "KEEP_CONTEXT" if same_object and savings > 0 else "DROP_CONTEXT",
        "verdict_scope": scope,
    }


def _build_context_races(
    g4: Mapping[str, Any],
    dm1: Mapping[str, Any],
) -> list[dict[str, Any]]:
    free = g4["summary"]["free_context"]
    real = free["real_coder_measurement"]
    context_free = real["context_free_raster"]
    aggregate = real["aggregate_pixel_time_order"]
    boundary = real["predictor_boundary_distance_context"]
    aggregate_race = _context_race(
        race_id="g4_aggregate_pixel_time_order",
        stratum="all_argmax_innovations",
        explicit_bytes=context_free["selected_bytes"],
        contextual_bytes=aggregate["selected_bytes"],
        counted_context_parameter_bytes=free["aggregate_spatial_pixel_prior"][
            "context_payload_bytes"
        ],
        contextual_type=LP1Type.CONTEXT,
        contextual_layer=LP1Layer.L1_PROGRAM,
        same_object=True,
        scope=(
            "G4 future innovation bitstream only; not subtractable from the "
            "current C1 exact control or any unrelated receiver object"
        ),
    )
    if aggregate_race["savings_bytes"] != aggregate["gain_bytes_vs_context_free"]:
        raise LayerPricingError("G4 aggregate free-context gain arithmetic differs")
    boundary_race = _context_race(
        race_id="g4_predictor_boundary_distance",
        stratum="predictor_boundary_distance_proxy",
        explicit_bytes=context_free["selected_bytes"],
        contextual_bytes=boundary["selected_bytes"],
        counted_context_parameter_bytes=free[
            "predictor_boundary_distance_margin_proxy"
        ]["context_payload_bytes"],
        contextual_type=LP1Type.CONTEXT,
        contextual_layer=LP1Layer.L1_PROGRAM,
        same_object=True,
        scope=(
            "G4 topological boundary-distance proxy on the measured innovation "
            "stream; no receiver-realized RGB or independent physical-BEV claim"
        ),
    )
    if boundary_race["savings_bytes"] != boundary["gain_bytes_vs_context_free"]:
        raise LayerPricingError("G4 boundary-context gain arithmetic differs")
    independent = dm1["independent"]["sum_winning_row_container_bytes"]
    joint = dm1["joint_shared_context"]["exact_counted_bytes"]
    joint_race = _context_race(
        race_id="dm1_joint_shared_semantic_container",
        stratum="25_registered_L4_semantic_rows",
        explicit_bytes=independent,
        contextual_bytes=joint,
        counted_context_parameter_bytes=0,
        contextual_type=LP1Type.CONTEXT,
        contextual_layer=LP1Layer.L4_SCORER_FEATURE,
        same_object=True,
        scope=(
            "DM1 semantic records only; the joint container is not receiver-closed "
            "RGB/archive rate and cannot be charged into the C1 allocation"
        ),
    )
    return [aggregate_race, boundary_race, joint_race]


def _build_c1_rows(c1: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source_rows = c1.get("byte_budget")
    if not isinstance(source_rows, Sequence) or isinstance(source_rows, (str, bytes)):
        raise LayerPricingError("C1 byte_budget must be a sequence")
    names = [row.get("stream") for row in source_rows if isinstance(row, Mapping)]
    if len(names) != len(source_rows) or set(names) != set(_POLICIES):
        raise LayerPricingError("C1 stream inventory differs from the sealed LP1 audit")
    if len(names) != len(set(names)):
        raise LayerPricingError("C1 stream names must be unique")

    output: list[dict[str, Any]] = []
    allocated = 0
    fixed_subtotal = 0
    planning_reserve = 0
    hard_ceiling = 0
    for source in source_rows:
        name = source["stream"]
        source_bytes = _nonnegative_int(source["bytes"], f"C1 {name}.bytes")
        policy = _POLICIES[name]
        if policy.counted_in_corrected_total:
            corrected_bytes: int | None = source_bytes
            allocated += source_bytes
            if name != "lane_program_seed":
                fixed_subtotal += source_bytes
        elif name in {
            "contextual_bounded_collateral_shared_application_stage_reserve",
            "v18b_first_exact_pricing_rung_reserve",
            "j3_finish_and_xi_refinement_reserve",
            "final_coder_and_container_contingency",
        }:
            corrected_bytes = 0
            planning_reserve += source_bytes
        elif name == "v15_exact_control_subtotal":
            corrected_bytes = source_bytes
        elif name == "hard_total":
            corrected_bytes = None
            hard_ceiling = source_bytes
        else:  # pragma: no cover - sealed map makes this unreachable
            raise LayerPricingError(f"unhandled C1 row {name}")
        output.append(
            {
                "schema": C1_ROW_SCHEMA,
                "stream": name,
                "stratum": policy.stratum,
                "source_planning_bytes": source_bytes,
                "source_provenance": source.get("provenance"),
                "typed_home": (
                    None
                    if policy.stream_type is None
                    else {
                        "type": policy.stream_type.value,
                        "layer_home": policy.layer_home.value,
                    }
                ),
                "survival_authority": policy.survival_authority,
                "disposition": policy.disposition,
                "corrected_allocated_bytes": corrected_bytes,
                "counted_in_corrected_total": policy.counted_in_corrected_total,
                "generic_receiver_implementation_bytes": 0,
                "verdict_scope": (
                    "this exact C1 stream/accounting row only; deepest proven home "
                    "is not a global minimum-description or promotion verdict"
                ),
            }
        )

    source_subtotal = next(
        row["bytes"] for row in source_rows if row["stream"] == "v15_exact_control_subtotal"
    )
    if fixed_subtotal != source_subtotal:
        raise LayerPricingError("C1 exact member homes do not reconcile to the subtotal")
    if allocated + planning_reserve != hard_ceiling:
        raise LayerPricingError("C1 seeded allocation plus reserves do not equal hard total")
    for row in output:
        if row["stream"] == "hard_total":
            row["corrected_allocated_bytes"] = allocated
    return output, {
        "source_exact_control_subtotal_bytes": fixed_subtotal,
        "corrected_measured_allocated_bytes": allocated,
        "source_planning_reserve_bytes": planning_reserve,
        "unallocated_headroom_bytes": hard_ceiling - allocated,
        "hard_ceiling_bytes": hard_ceiling,
    }


def _rows_by_index(value: Mapping[str, Any], label: str) -> list[Mapping[str, Any]]:
    rows = value.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)) or len(rows) != 25:
        raise LayerPricingError(f"{label} must contain exact 25 rows")
    ordered = sorted(rows, key=lambda row: row["row_index"])
    if [row["row_index"] for row in ordered] != list(range(25)):
        raise LayerPricingError(f"{label} row identities differ from 0..24")
    return ordered


def _build_sense_rows(
    dm1: Mapping[str, Any],
    dm2: Mapping[str, Any],
    dm4: Mapping[str, Any],
    ms7_r0: Mapping[str, Any],
) -> list[dict[str, Any]]:
    dm1_rows = _rows_by_index(dm1, "DM1")
    dm2_rows = _rows_by_index(dm2, "DM2")
    dm4_rows = _rows_by_index(dm4, "DM4")
    ms7_rows = _rows_by_index(ms7_r0, "MS7 R0")
    output: list[dict[str, Any]] = []
    for index, (semantic, dm2_row, realized, reach) in enumerate(
        zip(dm1_rows, dm2_rows, dm4_rows, ms7_rows, strict=True)
    ):
        identity = (semantic["pair_id"], semantic["bucket_id"])
        for label, row in (("DM2", dm2_row), ("DM4", realized), ("MS7", reach)):
            if (row["pair_id"], row["bucket_id"]) != identity:
                raise LayerPricingError(f"{label} row {index} identity differs from DM1")
        source_type = semantic["adjudicated_typed_home"]["type"]
        if source_type == "FIBER":
            lp1_type = LP1Type.FIBER
        elif source_type == "SKELETON":
            lp1_type = LP1Type.RESIDUAL
        else:
            raise LayerPricingError(f"DM1 row {index} type is outside the materialized set")
        l4_bytes = _positive_int(semantic["exact_counted_bytes"], "DM1 semantic bytes")
        l3_bytes = _positive_int(
            realized["rgb_record"]["exact_counted_bytes"],
            "DM4 exact L3 bytes",
        )
        if realized["realization_status"] != "SUCCESS_EXACT_L4_RECORD_THROUGH_L3_RGB":
            raise LayerPricingError(f"DM4 row {index} lacks exact L4-through-L3 survival")
        if reach["mass_pays_cheapest_measured_guaranteed_reach"] is not False:
            raise LayerPricingError(f"MS7 R0 row {index} unexpectedly pays reach")
        output.append(
            {
                "schema": SENSE_ROW_SCHEMA,
                "row_index": index,
                "pair_id": semantic["pair_id"],
                "bucket_id": semantic["bucket_id"],
                "stratum": semantic["stratum"],
                "typed_home": {
                    "type": lp1_type.value,
                    "layer_home": LP1Layer.L4_SCORER_FEATURE.value,
                    "source_typology_type": source_type,
                },
                "l4_semantic_counted_bytes": l4_bytes,
                "l4_winning_codec": semantic["winning_codec"],
                "l3_receiver_realization_counted_bytes": l3_bytes,
                "l3_winning_codec": realized["rgb_record"]["winning_codec"],
                "l3_over_l4_price_ratio": l3_bytes / l4_bytes,
                "l4_through_l3_survival": "MEASURED_EXACT",
                "g4_same_object_context_price_bytes": None,
                "context_disposition": (
                    "KEEP_EXPLICIT_L4_DESCRIPTION_PENDING_SAME_OBJECT_G4_CONTEXT"
                ),
                "reach_price_bound_bytes": reach[
                    "cheapest_reach_price_bound_bytes"
                ],
                "flip_weighted_S_leverage": reach["flip_weighted_S_leverage"],
                "mass_minus_rate_score_bound": reach["mass_minus_rate_score_bound"],
                "mass_pays_measured_reach": False,
                "waterfill_allocation_bytes": 0,
                "costate_state": "SENSE_HOLD_ZERO_R0",
                "next_action": (
                    "reprice a receiver-derived same-object context or a dynamic "
                    "scorer-recursive reach curve; do not admit the current L3 exception"
                ),
                "verdict_scope": reach["verdict_scope"],
            }
        )
    return output


def build_receipt(config: Mapping[str, Any], *, config_path: Path) -> dict[str, Any]:
    """Build the deterministic advisory LP1 receipt."""

    _validate_config(config)
    sources, custody = _load_sources(config)
    c1_rows, waterfill = _build_c1_rows(sources["c1"])
    context_races = _build_context_races(sources["g4"], sources["dm1"])
    sense_rows = _build_sense_rows(
        sources["dm1"],
        sources["dm2"],
        sources["dm4"],
        sources["ms7_r0"],
    )
    if sources["ms7"]["r0"]["unreachable_and_ignored_row_count"] != 25:
        raise LayerPricingError("MS7 summary does not preserve 25 R0 nonadmissions")
    if sources["ms7"]["r0"]["mass_paying_row_count"] != 0:
        raise LayerPricingError("MS7 summary unexpectedly has an R0 mass-paying row")
    dm4_joint = sources["dm4"]["aggregate"]["joint_score_accounting"]
    if dm4_joint["joint_score_delta"] <= 0:
        raise LayerPricingError("DM4 current full L3 realization unexpectedly pays")
    config_bytes = canonical_json_bytes(config)
    source_pointer = sources["c1"]["pointer"]
    return {
        "schema": SCHEMA,
        "run_id": config["run_id"],
        "lane_id": LANE_ID,
        "tasks": [669],
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "pointer": source_pointer,
        "pointer_moved": False,
        "main_review_required": True,
        "verdict": (
            "CORRECTED_C1_MEASURED_ALLOCATION_IS_SEEDED_CONTROL_ONLY;"
            "G4_FREE_CONTEXT_GAIN_IS_FUTURE_STREAM_SCOPED;"
            "ALL_25_L3_EXCEPTIONS_HOLD_ZERO_AT_MS7_R0"
        ),
        "verdict_scope": (
            "SHA-bound C1 budget x G4 context measurements x materialized 25-row "
            "DM1/DM2/DM4/MS7 instance. No coder-choice supersession, minimum-"
            "description proof, contest score, promotion, or frontier mutation."
        ),
        "config_path": config_path.relative_to(REPO).as_posix(),
        "config_sha256": sha256(config_bytes).hexdigest(),
        "source_custody": custody,
        "free_interpreter_law": {
            "generic_receiver_program_bytes": 0,
            "generic_context_model_bytes": 0,
            "count_only": (
                "irreducible video-derived parameters, innovations, and exact "
                "archive/container bytes"
            ),
            "hide_data_in_code_allowed": False,
        },
        "context_keep_drop_rows": context_races,
        "c1_corrected_waterfill": {
            **waterfill,
            "rate_dual": {
                "numerator": RATE_DUAL_NUMERATOR,
                "denominator": RATE_DUAL_DENOMINATOR,
                "score_units_per_byte": (
                    RATE_DUAL_NUMERATOR / RATE_DUAL_DENOMINATOR
                ),
            },
            "g4_savings_applied_to_current_c1_bytes": 0,
            "dm1_semantic_bytes_applied_to_current_c1_bytes": 0,
            "dm4_l3_realization_bytes_applied_to_current_c1_bytes": 0,
            "allocation_law": (
                "allocate only a same-object receiver-closed measured marginal "
                "with negative joint delta after rate; unmeasured reserves allocate zero"
            ),
            "rows": c1_rows,
        },
        "costate_sense": {
            "schema": "ddm_lp1_costate_sense_table.v1",
            "row_count": 25,
            "boundary_rows": sum(row["stratum"] == "boundary" for row in sense_rows),
            "cell_rows": sum(row["stratum"] == "cell" for row in sense_rows),
            "zero_allocation_rows": sum(
                row["waterfill_allocation_bytes"] == 0 for row in sense_rows
            ),
            "rows": sense_rows,
        },
        "dm4_current_l3_control": {
            "exact_counted_bytes": sources["dm4"]["aggregate"]["realized_rgb_joint"][
                "exact_counted_bytes"
            ],
            "joint_score_delta": dm4_joint["joint_score_delta"],
            "disposition": "DROP_FROM_C1_WATERFILL_CURRENT_INSTANCE",
            "verdict_scope": sources["dm4"]["verdict_scope"],
        },
        "cc2_coordination": {
            "coder_choice_owner": "ddm_cc2_coder_races",
            "lp1_selected_new_codec": False,
            "source_receipt_codec_labels_preserved": True,
            "ms7_pf3_race_used_as_home_evidence_only": True,
            "g4_free_context_remains_unavailable_for_ms7_flat_receiver_object": (
                sources["ms7"]["pf3"]["coder_race"]["rows"][-1]["unavailable_reason"]
            ),
        },
        "scorer_recursive_derivation": {
            "L4": "exact rank-4 SegNet head and frozen scorer-feature semantic records",
            "L3": (
                "DM4 corrected-inner-Jacobian/resize-adjoint/ERF/stem-lattice "
                "realization through uint8, exact R, SegNet, and PoseNet"
            ),
            "L2": "PF/worldsheet chart and grammar coordinates",
            "L1": "rule-118 receiver program and decoder-derived context",
            "generic_spatial_menu_admitted": False,
        },
        "stores_consulted": [
            "CLAUDE.md",
            "AGENTS.md",
            "PROGRAM.md",
            "docs/operating_manual_craft_handoff.md",
            "C1 composed candidate ledger",
            "G4 spatial stationarity receipt",
            "DM1/DM2/DM4 materialized 25-row receipts",
            "PF1/PF2 and MS5/MS6 admission receipts",
            "MS7 receiver-edge and R0 mass/reach receipts",
            "operator inbox directives through 2026-07-24T23:09:25Z",
        ],
        "triality": {
            "dsl_data": (
                ".omx/research/ddm_lp1_layer_pricing_20260725T031654Z/"
                "ddm_lp1_layer_pricing_receipt.json"
            ),
            "dag": ".omx/research/ddm_lp1_layer_pricing_DAG_FEED_20260725.md",
            "equations": (
                "src/tac/canonical_equations/"
                "ddm_lp1_layer_pricing_20260725.py"
            ),
        },
    }


def materialize(config_path: Path, output_dir: Path) -> dict[str, Any]:
    """Materialize one small deterministic receipt atomically."""

    config_path = config_path.resolve()
    if not config_path.is_relative_to(REPO):
        raise LayerPricingError("config path must remain inside the repository")
    config_value = json.loads(config_path.read_bytes())
    if not isinstance(config_value, Mapping):
        raise LayerPricingError("config must contain a JSON object")
    result = build_receipt(config_value, config_path=config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "ddm_lp1_layer_pricing_receipt.json"
    payload = canonical_json_bytes(result)
    if target.exists():
        if target.read_bytes() != payload:
            raise LayerPricingError("existing LP1 receipt differs; refusing overwrite")
        return result
    temporary = output_dir / f".{target.name}.{os.getpid()}.tmp"
    temporary.write_bytes(payload)
    os.replace(temporary, target)
    return result


__all__ = [
    "CONFIG_SCHEMA",
    "LANE_ID",
    "SCHEMA",
    "LP1Layer",
    "LP1Type",
    "LayerPricingError",
    "build_receipt",
    "canonical_json_bytes",
    "materialize",
]
