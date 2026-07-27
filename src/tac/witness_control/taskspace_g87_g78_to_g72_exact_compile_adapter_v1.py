# SPDX-License-Identifier: MIT
"""Exact G78-to-G72 scorer-stage compile-input adapter.

G78 owns the fresh full-n600 batch-16 target/description scorer cache.  G72
owns the role-aware analytic proposal derivation.  This module connects those
two already-real surfaces without copying dense encoder-only fields and
without inventing proposal-selection thresholds.

The adapter is intentionally narrower than a selected-preimage compiler:

* it recursively reopens the exact G78 aggregate through the strict G78 loader;
* it requires the caller-owned aggregate file and self hashes;
* it maps the five chronological G78 stage views onto the exact G72 stage plan;
* it retains all four scorer fields, including the described margins that G72
  v1 does not yet consume; and
* it emits a small sealed receipt proving which two G72 blockers are closed.

It does not select proposals, emit a counted operand, build an archive, invoke
a scorer, or claim a score.  Dense fields remain encoder-only and forbidden
from candidate payloads.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from tac.witness_control.taskspace_batch16_margin_base_scorer_cache_v1 import (
    PRODUCTION_BATCH_PAIRS,
    PRODUCTION_CLASS_COUNT,
    PRODUCTION_PAIR_COUNT,
    PRODUCTION_SCORER_HW,
    PRODUCTION_STAGE_COUNT,
    PRODUCTION_STAGE_PAIRS,
    REMAINING_G72_BLOCKERS,
    Batch16MarginBaseScorerCacheError,
    MarginBaseScorerCacheLoaderV1,
    canonical_json_bytes,
    file_identity,
    require_ssd_output_root,
    sha256_file,
    write_immutable_json,
)
from tac.witness_dsl.taskspace_g72_fresh_n600_g49_analytic_factor_compiler_v1 import (
    FRESH_BATCH16_MARGIN_CUSTODY_OWED,
    FRESH_V15_BASE_SCORER_CACHE_OWED,
    PROPOSAL_LAW,
    G72StagePlanV1,
    derive_v9_boundary_shearlet_stage_proposals,
    g72_stage_plan,
    reopen_stage_checkpoint,
    write_stage_checkpoint,
)

COMPILE_INPUT_SCHEMA: Final = "tac.taskspace_g87_g78_to_g72_exact_compile_input.v1"
COMPILE_INPUT_RECEIPT_SCHEMA: Final = "tac.taskspace_g87_g78_to_g72_exact_compile_input_receipt.v1"
MATERIALIZATION_RECEIPT_SCHEMA: Final = "tac.taskspace_g87_g78_to_g72_complete_proposal_materialization.v1"
COMPLETE_MINIMUM_COMPONENT_SITES: Final = 1
COMPLETE_MAXIMUM_COMPONENTS_PER_PAIR_ROLE: Final = 4096
G72_COMPILER_SOURCE_PATH: Final = (
    Path(__file__).resolve().parents[1] / "witness_dsl" / "taskspace_g72_fresh_n600_g49_analytic_factor_compiler_v1.py"
)

U8 = npt.NDArray[np.uint8]
F32 = npt.NDArray[np.float32]

_EXPECTED_CLOSED_BLOCKERS: Final = (
    FRESH_BATCH16_MARGIN_CUSTODY_OWED,
    FRESH_V15_BASE_SCORER_CACHE_OWED,
)


class G87ExactCompileAdapterError(RuntimeError):
    """G78 custody or G72 stage compatibility failed closed."""


def _require_sha256(value: object, *, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or value != value.lower()
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise G87ExactCompileAdapterError(f"{label} must be a lowercase SHA-256")
    return value


def _payload_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _seal(body: dict[str, Any], *, field: str) -> dict[str, Any]:
    if field in body:
        raise G87ExactCompileAdapterError(f"{field} is already present")
    return {**body, field: _payload_sha256(body)}


def _require_stage_array(
    value: np.ndarray,
    *,
    label: str,
    expected_dtype: np.dtype[Any],
) -> np.ndarray:
    array = np.asarray(value)
    expected_shape = (PRODUCTION_STAGE_PAIRS, *PRODUCTION_SCORER_HW)
    if array.shape != expected_shape:
        raise G87ExactCompileAdapterError(f"{label} shape differs from the exact G72 stage")
    if array.dtype != expected_dtype:
        raise G87ExactCompileAdapterError(f"{label} dtype differs from the exact G78 wire")
    if array.flags.writeable:
        raise G87ExactCompileAdapterError(f"{label} must remain a read-only encoder-side view")
    return array


@dataclass(frozen=True, slots=True)
class G87G72StageCompileInputV1:
    """One exact, read-only G72 proposal-derivation input stage."""

    plan: G72StagePlanV1
    g78_stage_receipt_sha256: str
    target_cells_u8: U8
    target_margins_f32: F32
    described_cells_u8: U8
    described_margins_f32: F32
    input_identities: dict[str, Any]

    def __post_init__(self) -> None:
        if type(self.plan) is not G72StagePlanV1:
            raise G87ExactCompileAdapterError("stage plan is not an exact G72 plan")
        _require_sha256(
            self.g78_stage_receipt_sha256,
            label="g78_stage_receipt_sha256",
        )
        _require_stage_array(
            self.target_cells_u8,
            label="target_cells_u8",
            expected_dtype=np.dtype(np.uint8),
        )
        _require_stage_array(
            self.target_margins_f32,
            label="target_margins_f32",
            expected_dtype=np.dtype("<f4"),
        )
        _require_stage_array(
            self.described_cells_u8,
            label="described_cells_u8",
            expected_dtype=np.dtype(np.uint8),
        )
        _require_stage_array(
            self.described_margins_f32,
            label="described_margins_f32",
            expected_dtype=np.dtype("<f4"),
        )
        if type(self.input_identities) is not dict or set(self.input_identities) != {
            "described_cells_u8",
            "described_margins_f32",
            "target_cells_u8",
            "target_margins_f32",
        }:
            raise G87ExactCompileAdapterError("stage input identities are incomplete")

    def g72_derivation_kwargs(self) -> dict[str, Any]:
        """Return the exact array arguments accepted by G72's real derivation.

        G72 v1 does not consume ``described_margins_f32``.  It remains carried
        in this typed stage rather than being silently discarded.  The caller
        must separately supply a provenance-bound proposal policy; this method
        has no threshold defaults.
        """

        return {
            "stage": self.plan,
            "target_cells": self.target_cells_u8,
            "target_margins": self.target_margins_f32,
            "described_cells": self.described_cells_u8,
        }


@dataclass(frozen=True, slots=True)
class G87G72CompileInputV1:
    """Complete exact full-n600 G72 stage-input bundle."""

    aggregate_receipt_path: Path
    aggregate_file_sha256: str
    aggregate_self_sha256: str
    stages: tuple[G87G72StageCompileInputV1, ...]
    receipt: dict[str, Any]

    def iter_stages(self) -> tuple[G87G72StageCompileInputV1, ...]:
        """Return all five immutable chronological stage views."""

        return self.stages


@dataclass(frozen=True, slots=True)
class G87CompleteProposalStageCensusV1:
    """Exact representable component census for one G72 stage."""

    stage_index: int
    pair_range: tuple[int, int]
    component_count: int
    proposal_count: int
    mismatch_site_count: int
    maximum_components_per_pair_role_observed: int


def _g72_array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(canonical_json_bytes([int(item) for item in array.shape]))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def census_complete_g72_proposal_universe(
    compile_input: G87G72CompileInputV1,
) -> tuple[G87CompleteProposalStageCensusV1, ...]:
    """Count every 8-connected Road/Undrivable mismatch component.

    ``minimum_component_sites=1`` is the complete nonempty component universe,
    not a selection threshold.  The function refuses if G72's representational
    maximum of 4096 components per pair-role would truncate that universe.
    """

    if type(compile_input) is not G87G72CompileInputV1:
        raise G87ExactCompileAdapterError("component census requires an exact G87 compile input")
    structure = np.ones((3, 3), dtype=np.uint8)
    rows: list[G87CompleteProposalStageCensusV1] = []
    for stage in compile_input.stages:
        component_count = 0
        mismatch_site_count = 0
        observed_maximum = 0
        for local_pair_index in range(PRODUCTION_STAGE_PAIRS):
            target = stage.target_cells_u8[local_pair_index]
            described = stage.described_cells_u8[local_pair_index]
            for class_id in (0, 2):
                mismatch = (target == class_id) != (described == class_id)
                count = int(
                    ndimage.label(
                        mismatch,
                        structure=structure,
                    )[1]
                )
                component_count += count
                mismatch_site_count += int(np.count_nonzero(mismatch))
                observed_maximum = max(observed_maximum, count)
        if observed_maximum > COMPLETE_MAXIMUM_COMPONENTS_PER_PAIR_ROLE:
            raise G87ExactCompileAdapterError(
                "G72 representational component maximum truncates the complete G78 mismatch universe"
            )
        rows.append(
            G87CompleteProposalStageCensusV1(
                stage_index=stage.plan.stage_index,
                pair_range=(
                    stage.plan.pair_start,
                    stage.plan.pair_stop_exclusive,
                ),
                component_count=component_count,
                proposal_count=4 * component_count,
                mismatch_site_count=mismatch_site_count,
                maximum_components_per_pair_role_observed=observed_maximum,
            )
        )
    return tuple(rows)


def _proposal_checkpoint_upper_bound_bytes(
    census: tuple[G87CompleteProposalStageCensusV1, ...],
) -> int:
    """Derive a conservative atomic-write disk bound from exact census."""

    maximum_proposal_bytes = len(
        canonical_json_bytes(
            {
                "schema": ("tac.g72_v9_boundary_shearlet_proposal.v1"),
                "candidate_id": ("undrivable_599_4095_sh_d1_a0.5"),
                "fisher_priority": "1.7976931348623157e+308",
                "atom": {
                    "pair_index": 599,
                    "role": "UndrivableBoundary",
                    "center_y": 383,
                    "center_x": 511,
                    "scale_y": 48,
                    "scale_x": 256,
                    "shear_q4": -64,
                    "amplitude_q4": -512,
                },
            }
        )
    )
    # Each checkpoint stores the proposal and its 64-byte fingerprint.  The
    # separators and fixed custody body are bounded explicitly.  Atomic write
    # temporarily needs both the complete prior stages and one extra stage.
    stage_bounds = [65_536 + row.proposal_count * (maximum_proposal_bytes + 68) for row in census]
    return sum(stage_bounds) + max(stage_bounds, default=0)


def _strict_resume_stage(
    *,
    path: Path,
    stage: G87G72StageCompileInputV1,
    semantic_archive_sha256: str,
    semantic_compile_receipt_sha256: str,
    g46_target_receipt_sha256: str,
    g51_operand_receipt_sha256: str,
    previous_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise G87ExactCompileAdapterError("existing G72 stage checkpoint cannot be resumed") from exc
    if type(raw) is not dict:
        raise G87ExactCompileAdapterError("existing G72 stage checkpoint is not an object")
    expected_self = _require_sha256(
        raw.get("checkpoint_sha256"),
        label="checkpoint_sha256",
    )
    reopened = reopen_stage_checkpoint(
        path,
        expected_checkpoint_sha256=expected_self,
    )
    exact_values = {
        "semantic_archive_sha256": semantic_archive_sha256,
        "semantic_compile_receipt_sha256": (semantic_compile_receipt_sha256),
        "g46_target_receipt_sha256": g46_target_receipt_sha256,
        "g51_operand_receipt_sha256": g51_operand_receipt_sha256,
        "previous_checkpoint_sha256": previous_checkpoint_sha256,
    }
    if any(reopened.get(key) != value for key, value in exact_values.items()):
        raise G87ExactCompileAdapterError("existing G72 stage checkpoint names different custody")
    if reopened.get("derivation_config") != {
        "minimum_component_sites": COMPLETE_MINIMUM_COMPONENT_SITES,
        "maximum_components_per_pair_role": (COMPLETE_MAXIMUM_COMPONENTS_PER_PAIR_ROLE),
        "proposal_law": PROPOSAL_LAW,
    }:
        raise G87ExactCompileAdapterError("existing G72 stage checkpoint is not the complete universe")
    expected_inputs = {
        "target_cells": _g72_array_sha256(stage.target_cells_u8),
        "target_margins": _g72_array_sha256(stage.target_margins_f32),
        "described_cells": _g72_array_sha256(stage.described_cells_u8),
    }
    for label, digest in expected_inputs.items():
        if reopened["input_fields"][label]["sha256"] != digest:
            raise G87ExactCompileAdapterError(f"existing G72 stage {label} differs from live G78 input")
    return reopened


def materialize_complete_g72_proposal_stages(
    compile_input: G87G72CompileInputV1,
    *,
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    """Materialize every G72-representable proposal in five resume stages."""

    root = require_ssd_output_root(Path(output_root))
    root.mkdir(parents=True, exist_ok=True)
    census = census_complete_g72_proposal_universe(compile_input)
    required_free_bytes = _proposal_checkpoint_upper_bound_bytes(census)
    observed_free_bytes = shutil.disk_usage(root).free
    if observed_free_bytes < required_free_bytes:
        raise G87ExactCompileAdapterError("SSD free bytes are below the derived atomic checkpoint bound")
    receipt_path = root / "aggregate_receipt.json"
    recorded_observed_free_bytes = observed_free_bytes
    if receipt_path.exists():
        try:
            existing_receipt = json.loads(receipt_path.read_bytes())
        except (OSError, json.JSONDecodeError) as exc:
            raise G87ExactCompileAdapterError("existing materialization receipt cannot be resumed") from exc
        if type(existing_receipt) is not dict:
            raise G87ExactCompileAdapterError("existing materialization receipt is not an object")
        existing_self = _require_sha256(
            existing_receipt.get("materialization_receipt_sha256"),
            label="materialization_receipt_sha256",
        )
        existing_body = {
            key: value for key, value in existing_receipt.items() if key != "materialization_receipt_sha256"
        }
        if _payload_sha256(existing_body) != existing_self:
            raise G87ExactCompileAdapterError("existing materialization receipt self hash differs")
        storage = existing_receipt.get("storage_preflight")
        if (
            type(storage) is not dict
            or storage.get("output_root") != str(root)
            or storage.get("required_free_bytes_derived") != required_free_bytes
            or type(storage.get("observed_free_bytes")) is not int
        ):
            raise G87ExactCompileAdapterError("existing materialization storage custody differs")
        recorded_observed_free_bytes = storage["observed_free_bytes"]

    aggregate = compile_input.receipt
    semantic = aggregate["semantic_custody"]
    target = aggregate["target_custody"]
    g51 = aggregate["g51_y0_y1_custody"]
    semantic_archive_sha256 = semantic["archive"]["sha256"]
    semantic_compile_receipt_sha256 = semantic["compile_receipt"]["sha256"]
    g46_target_receipt_sha256 = target["g46_receipt_sha256"]
    g51_operand_receipt_sha256 = g51["aggregate_receipt_sha256"]

    checkpoint_rows: list[dict[str, Any]] = []
    previous_checkpoint_sha256: str | None = None
    for stage, census_row in zip(
        compile_input.stages,
        census,
        strict=True,
    ):
        checkpoint_path = (
            root
            / "stage_checkpoints"
            / (
                f"stage_{stage.plan.stage_index:02d}_"
                f"{stage.plan.pair_start:04d}_"
                f"{stage.plan.pair_stop_exclusive:04d}.json"
            )
        )
        if checkpoint_path.exists():
            checkpoint = _strict_resume_stage(
                path=checkpoint_path,
                stage=stage,
                semantic_archive_sha256=semantic_archive_sha256,
                semantic_compile_receipt_sha256=(semantic_compile_receipt_sha256),
                g46_target_receipt_sha256=g46_target_receipt_sha256,
                g51_operand_receipt_sha256=(g51_operand_receipt_sha256),
                previous_checkpoint_sha256=(previous_checkpoint_sha256),
            )
        else:
            proposals = derive_v9_boundary_shearlet_stage_proposals(
                **stage.g72_derivation_kwargs(),
                minimum_component_sites=(COMPLETE_MINIMUM_COMPONENT_SITES),
                maximum_components_per_pair_role=(COMPLETE_MAXIMUM_COMPONENTS_PER_PAIR_ROLE),
            )
            if len(proposals) != census_row.proposal_count:
                raise G87ExactCompileAdapterError(
                    "G72 derivation did not enumerate the exact component census four ways"
                )
            checkpoint_path = write_stage_checkpoint(
                output_root=root,
                stage=stage.plan,
                semantic_archive_sha256=semantic_archive_sha256,
                semantic_compile_receipt_sha256=(semantic_compile_receipt_sha256),
                g46_target_receipt_sha256=g46_target_receipt_sha256,
                g51_operand_receipt_sha256=(g51_operand_receipt_sha256),
                target_cells=stage.target_cells_u8,
                target_margins=stage.target_margins_f32,
                described_cells=stage.described_cells_u8,
                minimum_component_sites=(COMPLETE_MINIMUM_COMPONENT_SITES),
                maximum_components_per_pair_role=(COMPLETE_MAXIMUM_COMPONENTS_PER_PAIR_ROLE),
                proposals=proposals,
                previous_checkpoint_sha256=(previous_checkpoint_sha256),
            )
            checkpoint = _strict_resume_stage(
                path=checkpoint_path,
                stage=stage,
                semantic_archive_sha256=semantic_archive_sha256,
                semantic_compile_receipt_sha256=(semantic_compile_receipt_sha256),
                g46_target_receipt_sha256=g46_target_receipt_sha256,
                g51_operand_receipt_sha256=(g51_operand_receipt_sha256),
                previous_checkpoint_sha256=(previous_checkpoint_sha256),
            )
        previous_checkpoint_sha256 = checkpoint["checkpoint_sha256"]
        checkpoint_rows.append(
            {
                **file_identity(checkpoint_path),
                "stage_index": stage.plan.stage_index,
                "pair_range": [
                    stage.plan.pair_start,
                    stage.plan.pair_stop_exclusive,
                ],
                "checkpoint_sha256": (previous_checkpoint_sha256),
                "component_count": census_row.component_count,
                "proposal_count": census_row.proposal_count,
                "mismatch_site_count": (census_row.mismatch_site_count),
                "maximum_components_per_pair_role_observed": (census_row.maximum_components_per_pair_role_observed),
            }
        )

    body = {
        "schema": MATERIALIZATION_RECEIPT_SCHEMA,
        "g87_compile_input_receipt_sha256": aggregate["compile_input_receipt_sha256"],
        "g72_compiler_source": aggregate["g72_compiler_source"],
        "g78_aggregate": aggregate["g78_aggregate"],
        "geometry": aggregate["geometry"],
        "enumeration_policy": {
            "minimum_component_sites": (COMPLETE_MINIMUM_COMPONENT_SITES),
            "maximum_components_per_pair_role": (COMPLETE_MAXIMUM_COMPONENTS_PER_PAIR_ROLE),
            "interpretation": ("complete_nonempty_representable_universe_not_selection"),
            "selection_thresholds_used": False,
            "representational_cap_nonbinding_for_all_600_pairs": True,
        },
        "storage_preflight": {
            "output_root": str(root),
            "required_free_bytes_derived": required_free_bytes,
            "observed_free_bytes": recorded_observed_free_bytes,
            "derivation": ("sum_stage_serialization_upper_bounds_plus_largest_atomic_tmp"),
        },
        "stages": checkpoint_rows,
        "population": {
            "component_count": sum(row.component_count for row in census),
            "proposal_count": sum(row.proposal_count for row in census),
            "mismatch_site_count": sum(row.mismatch_site_count for row in census),
            "maximum_components_per_pair_role_observed": max(
                (row.maximum_components_per_pair_role_observed for row in census),
                default=0,
            ),
        },
        "geometry_invariant": {
            "scope": (f"all_600_pairs_all_{sum(row.component_count for row in census)}_connected_components"),
            "exact_component_membership_drives_geometry": True,
            "bbox_foreign_mismatch_sites_included": False,
            "enforced_by": ("component_labels_equal_component_id_and_site_count_runtime_assertion"),
        },
        "closed_blockers": list(_EXPECTED_CLOSED_BLOCKERS),
        "remaining_g72_blockers": list(REMAINING_G72_BLOCKERS),
        "five_g72_proposal_stages_materialized": True,
        "selected_preimage_operand_emitted": False,
        "selected_preimage_packet_emitted": False,
        "archive_emitted": False,
        "candidate_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "research_only": True,
        "encoder_only": True,
        "dense_fields_candidate_payload_allowed": False,
        "scorer_weights_candidate_payload_allowed": False,
        "checkpoint_policy": ("immutable_atomic_five_stage_resume_preserve_all"),
        "cleanup_policy": ("preserve_no_delete_until_machine_readable_coldstore_certificate"),
    }
    receipt = _seal(body, field="materialization_receipt_sha256")
    write_immutable_json(receipt_path, receipt)
    if json.loads(receipt_path.read_bytes()) != receipt:
        raise G87ExactCompileAdapterError("materialization receipt changed across parse-back")
    return receipt_path, receipt


def _stage_input_identities(
    stage_receipt: dict[str, Any],
) -> dict[str, Any]:
    target = stage_receipt.get("target_cells")
    files = stage_receipt.get("files")
    if type(target) is not dict or type(files) is not dict:
        raise G87ExactCompileAdapterError("G78 stage field identities are absent")
    required_files = {
        "described_cells_u8",
        "described_margins_f32",
        "target_margins_f32",
    }
    if set(files) != required_files:
        raise G87ExactCompileAdapterError("G78 stage dense-field identity set differs")
    identities = {
        "target_cells_u8": dict(target),
        "target_margins_f32": dict(files["target_margins_f32"]),
        "described_cells_u8": dict(files["described_cells_u8"]),
        "described_margins_f32": dict(files["described_margins_f32"]),
    }
    for label, row in identities.items():
        if type(row) is not dict:
            raise G87ExactCompileAdapterError(f"{label} identity is not an object")
        digest = row.get("slice_sha256") if label == "target_cells_u8" else row.get("sha256")
        _require_sha256(digest, label=f"{label}.sha256")
        if row.get("shape") != [
            PRODUCTION_STAGE_PAIRS,
            *PRODUCTION_SCORER_HW,
        ]:
            raise G87ExactCompileAdapterError(f"{label} identity shape differs")
    return identities


def open_g87_g78_to_g72_compile_input(
    aggregate_receipt_path: str | Path,
    *,
    expected_file_sha256: str,
    expected_self_sha256: str,
) -> G87G72CompileInputV1:
    """Strictly reopen G78 and expose its five exact G72 compile stages."""

    expected_file = _require_sha256(
        expected_file_sha256,
        label="expected_file_sha256",
    )
    expected_self = _require_sha256(
        expected_self_sha256,
        label="expected_self_sha256",
    )
    path = Path(aggregate_receipt_path).expanduser().resolve()
    try:
        loader = MarginBaseScorerCacheLoaderV1.open(
            path,
            expected_sha256=expected_file,
        )
    except (Batch16MarginBaseScorerCacheError, OSError) as exc:
        raise G87ExactCompileAdapterError("strict G78 aggregate reopen failed") from exc
    aggregate = loader.receipt
    if aggregate.get("aggregate_receipt_sha256") != expected_self:
        raise G87ExactCompileAdapterError("G78 aggregate self hash differs")
    if sha256_file(path) != expected_file:
        raise G87ExactCompileAdapterError("G78 aggregate changed after strict reopen")
    exact_geometry = {
        "pair_count": PRODUCTION_PAIR_COUNT,
        "stage_pairs": PRODUCTION_STAGE_PAIRS,
        "stage_count": PRODUCTION_STAGE_COUNT,
        "scorer_batch_pairs": PRODUCTION_BATCH_PAIRS,
        "scorer_hw": list(PRODUCTION_SCORER_HW),
        "class_count": PRODUCTION_CLASS_COUNT,
    }
    if any(aggregate.get(key) != value for key, value in exact_geometry.items()):
        raise G87ExactCompileAdapterError("G78 aggregate geometry differs from G72 production geometry")
    if tuple(aggregate.get("closed_blockers", ())) != _EXPECTED_CLOSED_BLOCKERS:
        raise G87ExactCompileAdapterError("G78 aggregate blocker closure differs")
    if tuple(aggregate.get("remaining_g72_blockers_unmodified", ())) != tuple(REMAINING_G72_BLOCKERS):
        raise G87ExactCompileAdapterError("G78 aggregate remaining-blocker custody differs")
    if (
        aggregate.get("candidate_claim") is not False
        or aggregate.get("score_claim") is not False
        or aggregate.get("promotion_eligible") is not False
        or aggregate.get("pointer_moved") is not False
        or aggregate.get("research_only") is not True
        or aggregate.get("encoder_only") is not True
        or aggregate.get("dense_fields_candidate_payload_allowed") is not False
        or aggregate.get("scorer_weights_candidate_payload_allowed") is not False
    ):
        raise G87ExactCompileAdapterError("G78 aggregate false-authority fences differ")
    semantic_custody = aggregate.get("semantic_custody")
    if type(semantic_custody) is not dict or semantic_custody.get("executed_receiver_contract_id") != (
        "tac.optimization.direct_description_carrier_compose.CarrierComposeReceiverV1.render_camera_pairs.v15"
    ):
        raise G87ExactCompileAdapterError("G78 aggregate does not bind the executed V15 camera receiver")

    stage_views = tuple(loader.iter_stages())
    plans = g72_stage_plan()
    bindings = aggregate.get("stages")
    if (
        len(stage_views) != PRODUCTION_STAGE_COUNT
        or len(plans) != PRODUCTION_STAGE_COUNT
        or type(bindings) is not list
        or len(bindings) != PRODUCTION_STAGE_COUNT
    ):
        raise G87ExactCompileAdapterError("G78/G72 stage population is incomplete")
    stages: list[G87G72StageCompileInputV1] = []
    receipt_stage_rows: list[dict[str, Any]] = []
    for plan, view, binding in zip(plans, stage_views, bindings, strict=True):
        if (
            view.stage_index != plan.stage_index
            or view.pair_range != (plan.pair_start, plan.pair_stop_exclusive)
            or type(binding) is not dict
            or binding.get("stage_index") != plan.stage_index
            or binding.get("pair_range") != [plan.pair_start, plan.pair_stop_exclusive]
        ):
            raise G87ExactCompileAdapterError("G78 stage does not map bijectively onto the G72 plan")
        stage_receipt_path = Path(str(binding.get("path")))
        try:
            stage_receipt = json.loads(stage_receipt_path.read_bytes())
        except (OSError, ValueError, TypeError) as exc:
            raise G87ExactCompileAdapterError("G78 stage receipt could not be reopened") from exc
        if type(stage_receipt) is not dict or stage_receipt.get("stage_receipt_sha256") != binding.get(
            "stage_receipt_sha256"
        ):
            raise G87ExactCompileAdapterError("G78 stage self identity differs after strict aggregate reopen")
        identities = _stage_input_identities(stage_receipt)
        stage = G87G72StageCompileInputV1(
            plan=plan,
            g78_stage_receipt_sha256=binding["stage_receipt_sha256"],
            target_cells_u8=view.target_cells_u8,
            target_margins_f32=view.target_margins_f32,
            described_cells_u8=view.described_cells_u8,
            described_margins_f32=view.described_margins_f32,
            input_identities=identities,
        )
        stages.append(stage)
        receipt_stage_rows.append(
            {
                "stage_index": plan.stage_index,
                "pair_range": [
                    plan.pair_start,
                    plan.pair_stop_exclusive,
                ],
                "g78_stage_receipt_file": {key: binding[key] for key in ("bytes", "path", "sha256")},
                "g78_stage_receipt_sha256": binding["stage_receipt_sha256"],
                "input_identities": identities,
                "g72_derivation_inputs_executable": True,
                "described_margins_carried_forward_not_dropped": True,
            }
        )

    body = {
        "schema": COMPILE_INPUT_RECEIPT_SCHEMA,
        "g78_aggregate": {
            "path": str(path),
            "bytes": path.stat().st_size,
            "file_sha256": expected_file,
            "aggregate_receipt_sha256": expected_self,
        },
        "geometry": exact_geometry,
        "g72_compiler_source": file_identity(G72_COMPILER_SOURCE_PATH),
        "coverage": {
            "pair_range": [0, PRODUCTION_PAIR_COUNT],
            "chronological_contiguous": True,
            "global_batch16_custody_preserved": True,
            "five_g72_stages_complete": True,
        },
        "semantic_custody": {
            "archive": semantic_custody["archive"],
            "compile_receipt": semantic_custody["compile_receipt"],
            "executed_receiver_contract_id": semantic_custody["executed_receiver_contract_id"],
        },
        "target_custody": aggregate["target_custody"],
        "g51_y0_y1_custody": aggregate["g51_y0_y1_custody"],
        "stages": receipt_stage_rows,
        "closed_blockers": list(_EXPECTED_CLOSED_BLOCKERS),
        "remaining_g72_blockers": list(REMAINING_G72_BLOCKERS),
        "proposal_enumeration_policy": {
            "minimum_component_sites": (COMPLETE_MINIMUM_COMPONENT_SITES),
            "maximum_components_per_pair_role": (COMPLETE_MAXIMUM_COMPONENTS_PER_PAIR_ROLE),
            "selection_thresholds_used": False,
            "cap_must_be_proven_nonbinding_before_materialization": True,
        },
        "additional_open_blockers": [],
        "next_executable_stage": ("G72_COMPLETE_ROLE_AWARE_ANALYTIC_PROPOSAL_MATERIALIZATION"),
        "g72_stage_inputs_executable": True,
        "selected_preimage_operand_emitted": False,
        "selected_preimage_packet_emitted": False,
        "archive_emitted": False,
        "candidate_claim": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "research_only": True,
        "encoder_only": True,
        "dense_fields_candidate_payload_allowed": False,
        "scorer_weights_candidate_payload_allowed": False,
    }
    receipt = _seal(body, field="compile_input_receipt_sha256")
    return G87G72CompileInputV1(
        aggregate_receipt_path=path,
        aggregate_file_sha256=expected_file,
        aggregate_self_sha256=expected_self,
        stages=tuple(stages),
        receipt=receipt,
    )


__all__ = [
    "COMPILE_INPUT_RECEIPT_SCHEMA",
    "COMPILE_INPUT_SCHEMA",
    "COMPLETE_MAXIMUM_COMPONENTS_PER_PAIR_ROLE",
    "COMPLETE_MINIMUM_COMPONENT_SITES",
    "MATERIALIZATION_RECEIPT_SCHEMA",
    "G87CompleteProposalStageCensusV1",
    "G87ExactCompileAdapterError",
    "G87G72CompileInputV1",
    "G87G72StageCompileInputV1",
    "census_complete_g72_proposal_universe",
    "materialize_complete_g72_proposal_stages",
    "open_g87_g78_to_g72_compile_input",
]
