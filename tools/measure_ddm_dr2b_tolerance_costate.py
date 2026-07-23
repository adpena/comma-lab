#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure E2 coordinate tolerances and assemble the DR2b pricing receipt.

This is a local, advisory-only measurement.  It never dispatches work and it
does not build a candidate archive.  Each E2 edit is scored in its original
canonical batch-16 window and exactly rebased onto the SHA-bound n600 E2
baseline.  SDWL1 tolerance and mode rows fail closed until a coordinate
crosswalk exists; E2 prices are never silently transferred between
formulations.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import brotli
import numpy as np
import torch
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    model_validator,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tac.boundary_math.power_diagram_witness import (  # noqa: E402
    open_stored_npy_memmap,
)
from tac.optimization.ddm_dr2b_tolerance_costate import (  # noqa: E402
    SCHEMA,
    SCORER_BATCH_SIZE,
    DDMDR2BMeasurementError,
    exact_layer_controls,
    exact_n600_rebase,
    frequency_band_admission,
    head_flip_distance,
    ordered_redundancy_matrix,
    rank_costate_rows,
    require_description_crosswalk,
)
from tac.optimization.ddm_runtime_sensitivity import (  # noqa: E402
    DDMRuntimePerturbationV1,
    RuntimeSensitivityError,
    decode_runtime_state,
    realize_perturbation,
    score_realized_perturbation,
)
from tac.optimization.direct_description_coupled_margin import (  # noqa: E402
    BASE_MEMBER,
    CoupledMarginProgramV1,
    decode_coupled_margin_program,
    encode_coupled_margin_program,
    parse_coupled_margin_archive,
)
from tac.optimization.direct_description_measurement_ladder import (  # noqa: E402
    rfc8785_canonicalize,
)
from tools.measure_ddm_v14_realization_fidelity import _load_models  # noqa: E402
from tools.measure_ddm_v15_scorer_solved_templates import (  # noqa: E402
    DDMV15ScorerSolvedTemplateConfigV1,
)

CONFIG_SCHEMA = "DDMDR2BToleranceCostateConfigV1"
RECEIPT_SCHEMA = "ddm_dr2b_tolerance_ladder_and_costate_rows.v1"
AXIS = "[macOS-CPU frozen-scorer advisory]"
FORBIDDEN_LINEAGE_TOKENS = ("hnerv", "pr95", "pr110", "pr128")


class MeasurementError(ValueError):
    """The DR2b measurement or receipt contract failed closed."""


class BoundPathV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: StrictStr
    bytes: StrictInt = Field(gt=0)
    sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")


class PerturbationProbeV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    probe_id: StrictStr = Field(min_length=1)
    stream: Literal[
        "base/chart.anchors",
        "base/chart.gradients",
        "base/chart.residuals",
        "semantic/composed",
    ]
    flat_index: StrictInt = Field(ge=0)
    delta: StrictInt = Field(ge=-255, le=255)
    expected_original_value: StrictInt
    pair_id: StrictInt = Field(ge=0, lt=592)
    purpose: Literal["chart_tolerance", "semantic_boundary"]
    semantic_y: StrictInt | None = Field(default=None, ge=0, lt=384)
    semantic_x: StrictInt | None = Field(default=None, ge=0, lt=512)
    first_rung: Literal[True] = True

    @model_validator(mode="after")
    def _valid(self) -> PerturbationProbeV1:
        if self.delta == 0:
            raise ValueError("probe delta must be nonzero")
        semantic = self.stream == "semantic/composed"
        if semantic != (self.purpose == "semantic_boundary"):
            raise ValueError("semantic stream and purpose disagree")
        if semantic != (self.semantic_y is not None and self.semantic_x is not None):
            raise ValueError("semantic probes require one scorer cell")
        return self


class DDMDR2BToleranceCostateConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_: Literal["DDMDR2BToleranceCostateConfigV1"] = Field(
        default=CONFIG_SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    run_id: Literal["ddm_dr2b_tolerance_ladder_and_costate_rows_20260723"]
    seed: Literal[1234] = 1234
    scorer_threads: Literal[4] = 4
    scorer_batch_size: Literal[16] = 16
    authority_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    delegation_checkpoint_key: Literal["codex_delegate:ddm_dr2b_tolerance_ladder_and_costate_rows:20260723T184128Z"]
    e2_archive: BoundPathV1
    e2_verification_receipt: BoundPathV1
    e2_findings_receipt: BoundPathV1
    dr2_receipt: BoundPathV1
    dr1_receipt: BoundPathV1
    v19b_receipt: BoundPathV1
    dv2_receipt: BoundPathV1
    dv2_fact_inventory: BoundPathV1
    dv2_selected_payload: BoundPathV1
    scorer_config: BoundPathV1
    output_directory: StrictStr
    probes: tuple[PerturbationProbeV1, ...] = Field(
        min_length=1,
        strict=False,
    )
    sdwl1_to_e2_crosswalk: dict[str, Any] | None = None
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    pointer_moved: Literal[False] = False
    main_landing_review_required: Literal[True] = True

    @model_validator(mode="after")
    def _valid(self) -> DDMDR2BToleranceCostateConfigV1:
        if Path(self.output_directory).is_absolute():
            raise ValueError("output_directory must be repository-relative")
        ids = tuple(row.probe_id for row in self.probes)
        if len(set(ids)) != len(ids):
            raise ValueError("probe_id values must be unique")
        rendered = json.dumps(
            self.model_dump(mode="json", by_alias=True),
            sort_keys=True,
        ).lower()
        if any(token in rendered for token in FORBIDDEN_LINEAGE_TOKENS):
            raise ValueError("config references quarantined old-lineage input")
        return self

    def typed_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            total += len(block)
    return total, digest.hexdigest()


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO_ROOT / value


def _bound_bytes(bound: BoundPathV1) -> bytes:
    path = _resolve(bound.path)
    if path.is_symlink() or not path.is_file():
        raise MeasurementError(f"bound input is unavailable: {bound.path}")
    payload = path.read_bytes()
    if (len(payload), _sha256(payload)) != (bound.bytes, bound.sha256):
        raise MeasurementError(f"bound input changed: {bound.path}")
    return payload


def _bound_json(bound: BoundPathV1) -> dict[str, Any]:
    try:
        value = json.loads(_bound_bytes(bound))
    except json.JSONDecodeError as exc:
        raise MeasurementError(f"bound JSON is malformed: {bound.path}") from exc
    if not isinstance(value, dict):
        raise MeasurementError(f"bound JSON is not an object: {bound.path}")
    return value


def _publish_immutable(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise MeasurementError(f"immutable output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _canonical_json(payload: Mapping[str, Any]) -> bytes:
    return rfc8785_canonicalize(dict(payload)) + b"\n"


def _archive_members(payload: bytes) -> dict[str, bytes]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as reader:
            names = tuple(row.filename for row in reader.infolist())
            if names != (
                "manifest.json",
                "base/chart.ddb",
                "semantic/composed.dds",
            ):
                raise MeasurementError("E2 archive member order changed")
            return {name: reader.read(name) for name in names}
    except (OSError, KeyError, zipfile.BadZipFile) as exc:
        raise MeasurementError("E2 archive is malformed") from exc


def _pair_from_flat_index(
    state: Any,
    probe: PerturbationProbeV1,
) -> int:
    tensor = {
        "base/chart.anchors": state.anchors,
        "base/chart.gradients": state.gradients,
        "base/chart.residuals": state.residuals,
        "semantic/composed": state.labels,
    }[probe.stream]
    per_pair = tensor[0].numel()
    if probe.flat_index >= tensor.numel():
        raise MeasurementError(f"{probe.probe_id} index escapes stream")
    return probe.flat_index // per_pair


def _head_metric(
    *,
    segnet: Any,
    camera: np.ndarray,
    local_pair_index: int,
    y: int,
    x: int,
) -> dict[str, Any]:
    tensor = torch.from_numpy(np.ascontiguousarray(camera)).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        logits = segnet(segnet.preprocess_input(tensor))
    point = logits[local_pair_index, :, y, x].detach().cpu()
    order = torch.argsort(point, descending=True)
    top1, top2 = int(order[0]), int(order[1])
    margin = float(point[top1] - point[top2])
    heads = [module for module in segnet.modules() if isinstance(module, torch.nn.Conv2d) and module.out_channels == 5]
    if not heads:
        raise MeasurementError("frozen SegNet five-class head is unavailable")
    weights = heads[-1].weight.detach().cpu().reshape(5, -1)
    normal = float(torch.linalg.vector_norm(weights[top1] - weights[top2]))
    return {
        "metric": "frozen_head_top1_top2_margin_over_weight_normal",
        "top1_class": top1,
        "top2_class": top2,
        "margin": margin,
        "head_normal_norm": normal,
        "flip_distance": head_flip_distance(
            margin=margin,
            head_normal_norm=normal,
        ),
        "epistemic_status": "MEASURED",
    }


def _measure_probe(
    *,
    probe: PerturbationProbeV1,
    state: Any,
    segnet: Any,
    posenet: Any,
    labels: np.ndarray,
    poses: np.ndarray,
    margins: np.ndarray,
    baseline_d_seg: float,
    baseline_d_pose: float,
) -> dict[str, Any]:
    actual_pair = _pair_from_flat_index(state, probe)
    if actual_pair != probe.pair_id:
        raise MeasurementError(f"{probe.probe_id} pair_id disagrees with flat_index")
    start = (probe.pair_id // SCORER_BATCH_SIZE) * SCORER_BATCH_SIZE
    stop = start + SCORER_BATCH_SIZE
    perturbation = DDMRuntimePerturbationV1(
        stream=probe.stream,
        flat_index=probe.flat_index,
        delta=probe.delta,
        expected_original_value=probe.expected_original_value,
        pair_start=start,
        pair_stop=stop,
    )
    realized = realize_perturbation(state, perturbation)
    changed = np.any(
        realized.baseline_camera != realized.perturbed_camera,
        axis=(1, 2, 3, 4),
    )
    changed_pairs = tuple(start + int(index) for index in np.flatnonzero(changed))
    if changed_pairs != (probe.pair_id,):
        raise MeasurementError(f"{probe.probe_id} failed exact one-pair receiver locality")
    score = score_realized_perturbation(
        realized,
        segnet=segnet,
        posenet=posenet,
        target_labels=np.asarray(labels[start:stop], dtype=np.uint8),
        target_poses=np.asarray(poses[start:stop], dtype=np.float64),
    )
    n600 = exact_n600_rebase(
        baseline_d_seg=baseline_d_seg,
        baseline_d_pose=baseline_d_pose,
        window_d_seg_before=float(score["baseline"]["d_seg"]),
        window_d_seg_after=float(score["perturbed"]["d_seg"]),
        window_d_pose_before=float(score["baseline"]["d_pose"]),
        window_d_pose_after=float(score["perturbed"]["d_pose"]),
        window_pair_count=stop - start,
        delta_bytes=int(score["delta"]["bytes"]),
    )
    metric = None
    if probe.purpose == "semantic_boundary":
        assert probe.semantic_y is not None and probe.semantic_x is not None
        metric = _head_metric(
            segnet=segnet,
            camera=realized.baseline_camera,
            local_pair_index=probe.pair_id - start,
            y=probe.semantic_y,
            x=probe.semantic_x,
        )
        metric["target_cache_margin"] = float(
            margins[
                probe.pair_id,
                probe.semantic_y,
                probe.semantic_x,
            ]
        )
        metric["target_cache_margin_role"] = (
            "separate source-target Fisher prior; not substituted for the measured E2 head margin"
        )
    return {
        "schema": SCHEMA,
        "probe_id": probe.probe_id,
        "purpose": probe.purpose,
        "measurement_status": "MEASURED",
        "first_rung": True,
        "epistemic_status": "MEASURED_WINDOW_PLUS_DERIVED_EXACT_N600_REBASE",
        "perturbation": perturbation.model_dump(mode="json", by_alias=True),
        "canonical_window": {
            "pair_start": start,
            "pair_stop": stop,
            "batch_size": SCORER_BATCH_SIZE,
            "changed_pairs": list(changed_pairs),
            "one_pair_receiver_locality_proven": True,
        },
        "receiver_bijection": score["receiver_bijection"],
        "window_measurement": {
            "baseline": score["baseline"],
            "perturbed": score["perturbed"],
            "delta": score["delta"],
        },
        "fisher_margin": metric,
        "n600_rebase": n600,
        "evidence_axis": AXIS,
        "research_only": True,
        "score_claim": False,
        "verdict_scope": (
            "INSTANCE: one E2 counted coordinate on one SHA-bound pair, "
            "measured in its canonical batch-16 window and exactly rebased "
            "over receiver-invariant n600 peers; not an SDWL1 tolerance."
        ),
    }


def _receiver_blocker_row(
    *,
    probe: PerturbationProbeV1,
    error: RuntimeSensitivityError,
) -> dict[str, Any]:
    """Preserve a receiver-invalid tolerance rung without pricing it."""

    cause = error.__cause__
    return {
        "schema": SCHEMA,
        "probe_id": probe.probe_id,
        "purpose": probe.purpose,
        "measurement_status": "BLOCKED_RECEIVER_INVALID",
        "first_rung": True,
        "perturbation": {
            "schema": "DDMRuntimePerturbationV1",
            "stream": probe.stream,
            "flat_index": probe.flat_index,
            "delta": probe.delta,
            "expected_original_value": probe.expected_original_value,
            "pair_start": (probe.pair_id // SCORER_BATCH_SIZE) * SCORER_BATCH_SIZE,
            "pair_stop": ((probe.pair_id // SCORER_BATCH_SIZE) + 1) * SCORER_BATCH_SIZE,
            "scorer_batch_size": SCORER_BATCH_SIZE,
            "research_only": True,
            "score_claim": False,
        },
        "receiver_blocker": {
            "exception_type": type(error).__name__,
            "reason": str(error),
            "cause_type": None if cause is None else type(cause).__name__,
            "cause_reason": None if cause is None else str(cause),
            "scoped_interpretation": (
                "this coordinate step left the E2 receiver-valid uint8 domain; "
                "it is a measured invalid rung, not a family-level negative"
            ),
        },
        "epistemic_status": "MEASURED_RECEIVER_FAILURE",
        "evidence_axis": AXIS,
        "research_only": True,
        "score_claim": False,
        "verdict_scope": (
            "INSTANCE: one E2 counted coordinate and step size; smaller steps and other coordinates remain open."
        ),
    }


def _dr1_stream_matrix(
    *,
    dr1_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    row = next(
        value
        for value in dr1_receipt["deliverable_A_realization_race"]["rows"]
        if value["row_id"] == "post_int8_lattice"
    )
    path = (
        REPO_ROOT
        / ".omx/research/ddm_dr1_realization_race_coding_gain_n600_20260723"
        / "archives/post_int8_lattice.not_a_candidate.zip.receipt-bytes"
    )
    archive = path.read_bytes()
    expected = row["measurement"]
    if (len(archive), _sha256(archive)) != (
        int(expected["archive_bytes"]),
        str(expected["archive_sha256"]),
    ):
        raise MeasurementError("DR1 post-int8 archive custody changed")
    members, _homes = parse_coupled_margin_archive(archive)
    program = decode_coupled_margin_program(members["render/coupled_margin_program.ddcm"])
    stream_payloads = {
        "track_plus_template_bank_conditioned_on_horizon": members[BASE_MEMBER],
        "template_placements_conditioned_on_track": (
            encode_coupled_margin_program(CoupledMarginProgramV1(placements=program.placements))
        ),
        "sparse_compensations_conditioned_on_track_plus_template": (
            encode_coupled_margin_program(CoupledMarginProgramV1(compensations=program.compensations))
        ),
    }
    order = tuple(stream_payloads)
    pairwise = ordered_redundancy_matrix(
        stream_payloads,
        decode_order=order,
    )
    cumulative_rows = []
    prefix = b""
    prefix_coded = 0
    for name in order:
        standalone = len(brotli.compress(stream_payloads[name], quality=11))
        combined = len(brotli.compress(prefix + stream_payloads[name], quality=11))
        conditioned = combined - prefix_coded
        cumulative_rows.append(
            {
                "stream": name,
                "conditioned_on": list(order[: order.index(name)]),
                "raw_bytes": len(stream_payloads[name]),
                "standalone_bytes": standalone,
                "conditioned_bytes": conditioned,
                "redundancy_bytes": standalone - conditioned,
                "first_rung": True,
            }
        )
        prefix += stream_payloads[name]
        prefix_coded = combined
    return {
        "status": "MEASURED_ORDERED_CODER_DIAGNOSTIC",
        "post_int8_archive": {
            "path": str(path.relative_to(REPO_ROOT)),
            "bytes": len(archive),
            "sha256": _sha256(archive),
        },
        "decoded_record_counts": {
            "template_placements": len(program.placements),
            "sparse_compensations": len(program.compensations),
        },
        "cumulative_decode_order": cumulative_rows,
        "ordered_pairwise_redundancy_matrix": pairwise,
        "reserved_stream_schema": {
            "stream": "prosody_amplitudes",
            "status": "RESERVED_UNMEASURED",
            "coordinate_families": [
                "per_stratum_amplitude",
                "per_boundary_contrast",
                "global_per_channel_statistics",
            ],
            "tolerance_axis": "amplitude_not_geometry",
            "dimension_home": ("grammar-token attributes; no new symbolic token"),
            "conditioned_on": [
                "horizon",
                "track",
                "template",
                "sparse",
            ],
            "pairwise_redundancy_entries": None,
            "excluded_from_measured_matrix": True,
            "reason": (
                "No counted prosody stream or receiver Jacobian exists in "
                "the settled DR1 archive; zero bytes must not be inferred."
            ),
            "additional_stream_slots": [
                {
                    "stream": "prosody_spectrum_by_stratum",
                    "status": "RESERVED_UNMEASURED",
                    "coordinate_families": [
                        "per_stratum_frequency_band_amplitude",
                        "per_boundary_frequency_band_contrast",
                    ],
                    "tolerance_axis": "frequency_band_not_geometry",
                    "dimension_home": ("continuous spectral attributes on grammar tokens; no new symbolic token"),
                    "conditioned_on": [
                        "prosody_amplitudes",
                        "horizon",
                        "track",
                        "template",
                        "sparse",
                    ],
                    "pairwise_redundancy_entries": None,
                    "excluded_from_measured_matrix": True,
                }
            ],
        },
        "pruning": {
            "status": "BLOCKED_RECORD_LEVEL_REALIZED_MARGINALS_ABSENT",
            "reason": (
                "v19b measures strictly accepted move-level marginals, but "
                "the emitted 48 placement and 23 sparse records do not carry "
                "record-level n600 Seg/Pose marginals. Removing them from a "
                "move is non-additive and cannot be inferred."
            ),
            "verdict_scope": (
                "FORMULATION: this emitted DR1 post-int8 program only; "
                "correction pruning remains open after record-level replay."
            ),
        },
        "interpretation": (
            "Brotli-Q11 conditional-byte diagnostic over decoded stream "
            "payloads. It closes the ordered-matrix measurement blocker but "
            "does not rewrite the production ZIP or claim byte savings."
        ),
        "first_rung": True,
    }


def _semantic_content_crosswalk(
    *,
    state: Any,
    dv2_receipt: Mapping[str, Any],
    fact_inventory_payload: bytes,
    selected_payload: bytes,
) -> dict[str, Any]:
    """Test MAIN's provisional same-partition premise on exact bytes."""

    try:
        facts = np.load(
            io.BytesIO(fact_inventory_payload),
            allow_pickle=False,
        )
    except (OSError, ValueError) as exc:
        raise MeasurementError("DV2 fact inventory is malformed") from exc
    if facts.shape != (600, 11, 8) or facts.dtype != np.dtype("int64") or int(facts.size) != 52_800:
        raise MeasurementError("DV2 fact inventory geometry changed")
    inventory = dv2_receipt["inventory_stage"]
    if (
        inventory["payload"]["sha256"] != _sha256(fact_inventory_payload)
        or inventory["described_scalar_fact_count"] != 45_600
        or dv2_receipt["selected_base_row"]["outer_deflate_sha256"] != _sha256(selected_payload)
    ):
        raise MeasurementError("DV2 semantic inventory custody changed")

    fact_raw = np.ascontiguousarray(facts, dtype="<i8").tobytes(order="C")
    semantic_array = np.ascontiguousarray(
        state.labels.numpy(),
        dtype=np.uint8,
    )
    semantic_raw = semantic_array.tobytes(order="C")
    fact_coded = brotli.compress(fact_raw, quality=11)
    semantic_coded = brotli.compress(semantic_raw, quality=11)
    joint_coded = brotli.compress(fact_raw + semantic_raw, quality=11)
    conditioned_semantic_bytes = len(joint_coded) - len(fact_coded)
    histogram = np.bincount(
        semantic_array.reshape(-1),
        minlength=len(state.palette),
    )
    return {
        "question": (
            "Do E2 semantic/composed.dds and the 68,464-byte SDWL1 description encode the same partition content?"
        ),
        "verdict": "NO_DIFFERENT_CONTENT_TYPES",
        "same_content": False,
        "first_rung": True,
        "e2_semantic_member": {
            "content": ("117,964,800 per-cell categorical role assignments used by the runtime paint receiver"),
            "shape": list(semantic_array.shape),
            "dtype": str(semantic_array.dtype),
            "value_histogram": histogram.tolist(),
            "raw_bytes": len(semantic_raw),
            "raw_sha256": _sha256(semantic_raw),
            "member_bytes": len(state.semantic_member),
            "member_sha256": _sha256(state.semantic_member),
            "brotli_q11_body_bytes": len(semantic_coded),
        },
        "sdwl1_description": {
            "content": (
                "6,600 aggregate records / 45,600 declared scalar facts: "
                "partition-cell moments and boxes, separatrix summaries, "
                "topology deltas, and pair screw; no per-cell label plane"
            ),
            "tensor_shape": list(facts.shape),
            "tensor_dtype": str(facts.dtype),
            "tensor_storage_scalars_including_padding": int(facts.size),
            "declared_scalar_facts_excluding_padding": int(inventory["described_scalar_fact_count"]),
            "declared_records": int(inventory["described_record_count"]),
            "raw_bytes": len(fact_raw),
            "raw_sha256": _sha256(fact_raw),
            "selected_outer_bytes": len(selected_payload),
            "selected_outer_sha256": _sha256(selected_payload),
            "semantic_sha256_role": ("hash of the aggregate fact tensor, not a hash of the E2 per-cell role plane"),
        },
        "identical_realized_argmax_accounting": {
            "sdwl1_alone_can_replace_e2_member": False,
            "replacement_bytes": None,
            "measured_existing_e2_member_bytes": len(state.semantic_member),
            "measured_existing_e2_zip_home_bytes": 315_153,
            "measured_sdwl1_outer_bytes": len(selected_payload),
            "conditional_completion_coder": (
                "Brotli-Q11 over raw E2 role plane conditioned by decoded SDWL1 aggregate fact tensor"
            ),
            "conditional_completion_bytes": conditioned_semantic_bytes,
            "standalone_e2_role_plane_brotli_q11_bytes": len(semantic_coded),
            "conditional_delta_vs_standalone_bytes": (conditioned_semantic_bytes - len(semantic_coded)),
            "combined_sdwl1_plus_conditional_completion_bytes": (len(selected_payload) + conditioned_semantic_bytes),
            "interpretation": (
                "The conditional row is a measured coder diagnostic, not a "
                "production packet. Identical realized argmax still requires "
                "the per-cell role content or an invertible replacement."
            ),
        },
        "provisional_4_6x_claim": {
            "ratio_member_home_to_sdwl1": 315_153 / len(selected_payload),
            "free_rate_score": None,
            "status": "FALSIFIED_PREMISE_SAME_CONTENT_FALSE",
        },
        "delta_content": [
            "one role code for every pair x scorer-grid cell",
            "exact region interiors and boundaries",
            "role-to-palette assignment consumed by the E2 paint receiver",
        ],
        "verdict_scope": (
            "FORMULATION: E2 semantic role plane versus DV2 SDWL1 aggregate "
            "fact tensor only. A future invertible partition grammar remains "
            "open."
        ),
    }


def _four_clause_audits(
    *,
    sensitivity_rows: Sequence[Mapping[str, Any]],
    dr1_matrix: Mapping[str, Any],
) -> list[dict[str, Any]]:
    by_stream: dict[str, list[Mapping[str, Any]]] = {}
    for row in sensitivity_rows:
        stream = str(row["perturbation"]["stream"])
        by_stream.setdefault(stream, []).append(row)
    return [
        {
            "stream": "base/chart.ddb",
            "first_rung": True,
            "audit_triple": {
                "scorer_visibility": {
                    "status": "MEASURED_COORDINATE_ROWS",
                    "probe_ids": [
                        row["probe_id"]
                        for stream, rows in by_stream.items()
                        if stream.startswith("base/chart.")
                        for row in rows
                    ],
                },
                "sensitivity_priced_tolerance": {
                    "status": "MEASURED_SPARSE_COORDINATE_SAMPLE_ONLY",
                    "metric": "realized n600 reduced cost at batch16",
                    "coordinate_axes": {
                        "geometry": "sampled chart coordinate steps measured",
                        "amplitude_prosody": ("RESERVED_UNMEASURED; price separately"),
                        "frequency_band_prosody": (
                            "RESERVED_UNMEASURED; exact-R-null bands cost zero, "
                            "all scorer-visible bands require their own flip-distance"
                        ),
                    },
                },
                "three_layer_decomposition": {
                    "descriptive_form": ("anchors, gradients, residuals in one E2 chart member"),
                    "compact_dof": ("per-coordinate int16; gauge quotient not measured"),
                    "coder": "real Brotli-Q11 member bytes after parse-back",
                },
            },
            "nonredundancy": {
                "single_owner": "base/chart.ddb",
                "dimension_home": "pair x frame x chart coordinate",
                "conditioned_on": [],
                "corrections_are_deltas": True,
                "redundancy_entries": ("consume E2 two-stream matrix from the SHA-bound receipt"),
                "reserved_redundancy_streams": [
                    "prosody_amplitudes: RESERVED_UNMEASURED",
                    "prosody_spectrum_by_stratum: RESERVED_UNMEASURED",
                ],
            },
            "candidate_admissible": False,
            "verdict_scope": ("FORMULATION: sampled E2 chart coordinates only, not a complete tolerance field."),
        },
        {
            "stream": "semantic/composed.dds",
            "first_rung": True,
            "audit_triple": {
                "scorer_visibility": {
                    "status": "MEASURED_BOUNDARY_ROWS",
                    "probe_ids": [row["probe_id"] for row in by_stream.get("semantic/composed", [])],
                },
                "sensitivity_priced_tolerance": {
                    "status": "MEASURED_SPARSE_BOUNDARY_SAMPLE_ONLY",
                    "metric": ("|top1-top2 margin| / ||head weight normal|| plus realized n600 reduced cost"),
                    "coordinate_axes": {
                        "geometry": "sampled categorical boundary edit",
                        "boundary_contrast_prosody": ("RESERVED_UNMEASURED; not a new label symbol"),
                        "frequency_band_prosody": ("RESERVED_UNMEASURED; separate continuous spectral coordinate"),
                    },
                },
                "three_layer_decomposition": {
                    "descriptive_form": "categorical frame1 role field",
                    "compact_dof": ("one label per scorer cell; region quotient unmeasured"),
                    "coder": "real Brotli-Q11 member bytes after parse-back",
                },
            },
            "nonredundancy": {
                "single_owner": "semantic/composed.dds",
                "dimension_home": "pair x frame1 x scorer cell",
                "conditioned_on": ["base/chart.ddb"],
                "corrections_are_deltas": True,
                "redundancy_entries": ("consume E2 two-stream matrix from the SHA-bound receipt"),
                "reserved_redundancy_streams": [
                    "prosody_amplitudes: RESERVED_UNMEASURED",
                    "prosody_spectrum_by_stratum: RESERVED_UNMEASURED",
                ],
            },
            "candidate_admissible": False,
            "verdict_scope": (
                "FORMULATION: sampled E2 semantic boundaries only, not a complete boundary tolerance field."
            ),
        },
        {
            "stream": "dr1/post_int8_corrections",
            "first_rung": True,
            "audit_triple": {
                "scorer_visibility": {
                    "status": "MEASURED_N600_SETTLED_INPUT",
                    "joint_delta": -0.12083155237745034,
                    "delta_bytes": 235,
                },
                "sensitivity_priced_tolerance": {
                    "status": (dr1_matrix["pruning"]["status"]),
                },
                "three_layer_decomposition": {
                    "descriptive_form": ("horizon plus track/template/sparse corrections"),
                    "compact_dof": ("48 placement plus 23 sparse records in emitted row"),
                    "coder": ("exact archive settled; new conditional matrix is diagnostic only"),
                },
            },
            "nonredundancy": {
                "single_owner": ("horizon owns base; later streams own named deltas"),
                "dimension_home": ("clip base then pair-local correction records"),
                "conditioned_on": [
                    "horizon",
                    "track",
                    "template",
                ],
                "corrections_are_deltas": True,
                "redundancy_entries": dr1_matrix["ordered_pairwise_redundancy_matrix"],
            },
            "candidate_admissible": False,
            "verdict_scope": dr1_matrix["pruning"]["verdict_scope"],
        },
    ]


def run(
    config: DDMDR2BToleranceCostateConfigV1,
    *,
    config_path: Path,
) -> Path:
    root = (REPO_ROOT / config.output_directory).resolve()
    try:
        root.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise MeasurementError("output_directory escaped repository") from exc
    receipt_path = root / "receipt.json"
    config_hash = config.typed_hash()
    if receipt_path.exists():
        value = json.loads(receipt_path.read_bytes())
        if value.get("typed_config_sha256") != config_hash:
            raise MeasurementError("completed receipt config hash differs")
        print(
            json.dumps(
                {
                    "resumed": True,
                    "complete": True,
                    "receipt": str(receipt_path.relative_to(REPO_ROOT)),
                }
            )
        )
        return receipt_path

    e2_verification = _bound_json(config.e2_verification_receipt)
    e2_findings = _bound_json(config.e2_findings_receipt)
    dr2_receipt = _bound_json(config.dr2_receipt)
    dr1_receipt = _bound_json(config.dr1_receipt)
    v19b_receipt = _bound_json(config.v19b_receipt)
    dv2_receipt = _bound_json(config.dv2_receipt)
    dv2_fact_inventory = _bound_bytes(config.dv2_fact_inventory)
    dv2_selected_payload = _bound_bytes(config.dv2_selected_payload)
    scorer_config_payload = _bound_bytes(config.scorer_config)
    scorer_config = DDMV15ScorerSolvedTemplateConfigV1.model_validate_json(scorer_config_payload)
    if (
        scorer_config.scorer_batch_size != config.scorer_batch_size
        or scorer_config.scorer_threads != config.scorer_threads
        or scorer_config.seed != config.seed
    ):
        raise MeasurementError("scorer config differs from DR2b custody")

    archive = _bound_bytes(config.e2_archive)
    members = _archive_members(archive)
    state = decode_runtime_state(members)
    if state.manifest_sha256 != e2_verification["runtime"]["member_consumption"][0]["sha256"]:
        raise MeasurementError("E2 manifest SHA differs from verification")
    baseline_d_seg = float(e2_verification["score"]["d_seg"])
    baseline_d_pose = float(e2_verification["score"]["d_pose"])

    cache_path = Path(scorer_config.target_cache_path)
    cache_identity = _sha256_file(cache_path)
    if cache_identity != (
        scorer_config.target_cache_bytes,
        scorer_config.target_cache_sha256,
    ):
        raise MeasurementError("target-cache custody changed")
    labels = open_stored_npy_memmap(cache_path, "lstars")
    poses = open_stored_npy_memmap(cache_path, "gt_poses")
    margins = open_stored_npy_memmap(cache_path, "margins")
    segnet, posenet, scorer_custody = _load_models(scorer_config)

    rows = []
    checkpoints = root / "stage_checkpoints"
    for index, probe in enumerate(config.probes):
        checkpoint = checkpoints / f"{index:02d}_{probe.probe_id}.json"
        if checkpoint.exists():
            row = json.loads(checkpoint.read_bytes())
            if row.get("typed_config_sha256") != config_hash or row.get("probe_id") != probe.probe_id:
                raise MeasurementError(f"probe checkpoint differs: {probe.probe_id}")
            row.setdefault("measurement_status", "MEASURED")
        else:
            try:
                row = _measure_probe(
                    probe=probe,
                    state=state,
                    segnet=segnet,
                    posenet=posenet,
                    labels=labels,
                    poses=poses,
                    margins=margins,
                    baseline_d_seg=baseline_d_seg,
                    baseline_d_pose=baseline_d_pose,
                )
            except RuntimeSensitivityError as exc:
                row = _receiver_blocker_row(probe=probe, error=exc)
            row["typed_config_sha256"] = config_hash
            _publish_immutable(checkpoint, _canonical_json(row))
        rows.append(row)

    measured_rows = [row for row in rows if row.get("measurement_status", "MEASURED") == "MEASURED"]
    blocked_rows = [row for row in rows if row.get("measurement_status", "MEASURED") != "MEASURED"]
    ranked = rank_costate_rows(measured_rows)
    dr1_matrix = _dr1_stream_matrix(dr1_receipt=dr1_receipt)
    semantic_content = _semantic_content_crosswalk(
        state=state,
        dv2_receipt=dv2_receipt,
        fact_inventory_payload=dv2_fact_inventory,
        selected_payload=dv2_selected_payload,
    )
    exact_controls = exact_layer_controls()
    crosswalk_status: dict[str, Any]
    try:
        crosswalk = require_description_crosswalk(config.sdwl1_to_e2_crosswalk)
    except DDMDR2BMeasurementError as exc:
        crosswalk = None
        crosswalk_status = {
            "status": "BLOCKED_MISSING_COORDINATE_CROSSWALK",
            "reason": str(exc),
            "verdict_scope": (
                "FORMULATION bridge from SDWL1 fact coordinates to E2 runtime coordinates; DDM families remain open."
            ),
        }
    else:
        crosswalk_status = {
            "status": "AVAILABLE",
            "schema": crosswalk["schema"],
        }

    chart_rows = [row for row in rows if row["purpose"] == "chart_tolerance"]
    semantic_rows = [row for row in rows if row["purpose"] == "semantic_boundary"]
    pose_visible = [
        row
        for row in ranked
        if row["purpose"] == "chart_tolerance" and abs(float(row["n600_rebase"]["delta_d_pose"])) > 0.0
    ]
    positive_rows = [row for row in ranked if float(row["n600_rebase"]["joint_delta"]) < 0.0]
    if any(row.get("first_rung") is not True for row in positive_rows):
        raise MeasurementError("positive row lacks FIRST-RUNG")

    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": config.run_id,
        "authority": {
            "sha256": config.authority_sha256,
            "delegation_checkpoint_key": config.delegation_checkpoint_key,
        },
        "typed_config_path": str(config_path.relative_to(REPO_ROOT)),
        "typed_config_sha256": config_hash,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "pointer_moved": False,
        "main_landing_review_required": True,
        "evidence_axis": AXIS,
        "input_custody": {
            name: value.model_dump(mode="json")
            for name, value in (
                ("e2_archive", config.e2_archive),
                (
                    "e2_verification_receipt",
                    config.e2_verification_receipt,
                ),
                ("e2_findings_receipt", config.e2_findings_receipt),
                ("dr2_receipt", config.dr2_receipt),
                ("dr1_receipt", config.dr1_receipt),
                ("v19b_receipt", config.v19b_receipt),
                ("dv2_receipt", config.dv2_receipt),
                ("dv2_fact_inventory", config.dv2_fact_inventory),
                ("dv2_selected_payload", config.dv2_selected_payload),
                ("scorer_config", config.scorer_config),
            )
        },
        "scorer_custody": {
            **scorer_custody,
            "target_cache": {
                "path": str(cache_path),
                "bytes": cache_identity[0],
                "sha256": cache_identity[1],
            },
        },
        "u1_lossy_tolerance_ladder": {
            "exact_control": exact_controls,
            "e2_chart_coordinate_samples": chart_rows,
            "e2_semantic_boundary_samples": semantic_rows,
            "sdwl1_crosswalk": crosswalk_status,
            "semantic_content_crosswalk": semantic_content,
            "prosody_tolerance_axis": {
                "status": "RESERVED_UNMEASURED",
                "coordinate_families": dr1_matrix["reserved_stream_schema"]["coordinate_families"],
                "separate_from_geometry": True,
                "first_rung": (
                    "measure one counted amplitude attribute through the receiver before assigning any tolerance"
                ),
            },
            "frequency_band_tolerance_axis": {
                "status": "RESERVED_BLOCKED_EXACT_R_TRANSFER_AND_RECEIVER_JACOBIAN_ABSENT",
                "stream_slot": dr1_matrix["reserved_stream_schema"]["additional_stream_slots"][0],
                "separate_from_geometry_and_scalar_amplitude": True,
                "per_stratum": True,
                "exact_r_null_band_admission_law": {
                    "guard": "tac.optimization.ddm_dr2b_tolerance_costate.frequency_band_admission",
                    "predicate": "exact receiver transfer for the band is identically zero",
                    "description_bytes": 0,
                    "distortion_delta": 0,
                    "action": "truncate; never emit a description coordinate",
                    "guard_receipt": frequency_band_admission(
                        exact_r_transfer_zero=True,
                        emitted_description_bytes=0,
                    ),
                },
                "scorer_visible_band_admission_law": {
                    "predicate": "exact receiver transfer for the band is nonzero",
                    "required_measurement": ("per-band realized-through-R top1/top2 flip-distance and Pose delta"),
                    "rate_stop": "marginal distortion per byte reaches 25/37545489",
                },
                "measured_exact_r_null_bands": [],
                "reason": (
                    "E2 and DR1 expose no counted spectrum-parameterized "
                    "prosody stream or exact bandwise R-transfer certificate."
                ),
                "verdict_scope": (
                    "FORMULATION: current E2 and DR1 packets expose no "
                    "spectral receiver coordinate; spectrum-parameterized "
                    "description streams remain open."
                ),
                "first_rung": (
                    "export one SHA-bound band projector and prove exact R-nullness "
                    "or measure its realized scorer visibility"
                ),
            },
            "priced_sdwl1_rungs": [],
            "first_fit_rung": None,
            "status": (
                "BLOCKED_NO_LAWFUL_SDWL1_TO_E2_TOLERANCE_TRANSFER"
                if crosswalk is None
                else "CROSSWALK_PRESENT_BUT_GLOBAL_QUANTIZER_NOT_IMPLEMENTED"
            ),
            "fit_accounting": {
                "description_exact_bytes": (exact_controls["sdwl1_exact_description_bytes"]),
                "cap_bytes": (exact_controls["strict_sub015_cap_bytes_pose_held"]),
                "base_corrections_pose_bytes": None,
                "reason": (
                    "DR1 and E2 are separate packet formulations; their "
                    "bytes cannot be added to SDWL1 without one receiver "
                    "manifest and single-owner byte homes."
                ),
            },
            "verdict_scope": crosswalk_status["verdict_scope"],
        },
        "mode_at_tolerance_rerace": {
            "exact_layer": exact_controls,
            "lossy_rows": [],
            "status": "BLOCKED_SAME_COORDINATE_CROSSWALK",
            "verdict_scope": ("FORMULATION: exact SDWL1 static/track/re-key envelope; lossy mode family remains open."),
        },
        "dr1_post_int8_margin_pruning": {
            "settled_positive": {
                "joint_delta": -0.12083155237745034,
                "delta_bytes": 235,
                "first_rung": True,
                "epistemic_status": "MEASURED_SETTLED_INPUT",
            },
            "ordered_redundancy": dr1_matrix,
            "status": dr1_matrix["pruning"]["status"],
            "verdict_scope": dr1_matrix["pruning"]["verdict_scope"],
        },
        "g2_costate_rows": {
            "ranker": ("realized n600 reduced cost; Fisher/margin annotation where defined; stop at 25/37545489"),
            "rows": ranked,
            "receiver_invalid_rows": blocked_rows,
            "positive_rows": len(positive_rows),
            "first_rung_complete": len(rows) == len(config.probes),
            "status": "MEASURED_SPARSE_E2_COORDINATE_ROWS",
            "verdict_scope": (
                "INSTANCE rows only; not a complete stream tolerance field or prospective candidate ranking."
            ),
        },
        "xi_advection_pose_visibility": {
            "chart_pose_visible_probe_ids": [row["probe_id"] for row in pose_visible],
            "chart_pose_visible": bool(pose_visible),
            "xi_direction_bound": False,
            "status": (
                "PARTIAL_CHART_POSE_VISIBLE_XI_TO_CHART_JACOBIAN_ABSENT"
                if pose_visible
                else "NO_CHART_POSE_SIGNAL_IN_SAMPLED_COORDINATES"
            ),
            "reason": (
                "E2 exports no counted xi/pose member and its manifest has "
                "no xi-to-chart Jacobian. A generic chart perturbation cannot "
                "be relabeled as an xi-advected direction."
            ),
            "verdict_scope": (
                "FORMULATION: E2 compact runtime packet only; pose-legible "
                "chart and explicit xi inverse families remain open."
            ),
        },
        "four_clause_stream_audits": _four_clause_audits(
            sensitivity_rows=ranked,
            dr1_matrix=dr1_matrix,
        ),
        "settled_law_checks": {
            "dr2_exact_bytes_match": (
                dr2_receipt["mode_race"]["baseline_outer_bytes"] == exact_controls["sdwl1_exact_description_bytes"]
            ),
            "dr2_mode_delta_match": (
                dr2_receipt["mode_race"]["byte_delta_vs_baseline"] == exact_controls["mode_race_delta_bytes"]
            ),
            "dr1_positive_match": any(
                row["row_id"] == "post_int8_lattice"
                and row["joint_delta_vs_horizon_control"]["joint_delta"] == -0.12083155237745034
                for row in dr1_receipt["deliverable_A_realization_race"]["rows"]
            ),
            "v19b_move_level_rows": len(v19b_receipt["greedy_screen"]["per_move_joint_table"]),
            "e2_sensitivity_api_schema": e2_findings["typed_sensitivity_api"]["schema"],
            "semantic_same_content_refuted": (semantic_content["same_content"] is False),
        },
        "stores_consulted": [
            config.e2_archive.path,
            config.e2_verification_receipt.path,
            config.e2_findings_receipt.path,
            config.dr2_receipt.path,
            config.dr1_receipt.path,
            config.v19b_receipt.path,
            config.dv2_receipt.path,
            config.dv2_fact_inventory.path,
            config.dv2_selected_payload.path,
            config.scorer_config.path,
            scorer_config.target_cache_path,
        ],
        "directives_consumed": [
            {
                "utc": "2026-07-19T19:42:07Z",
                "application": ("rank realized reduced cost and stop at exact rate dual"),
            },
            {
                "utc": "2026-07-19T19:48:01Z",
                "application": (
                    "Fisher/top1-top2 margin metric; no Fourier residual claim; no unbound xi factorization"
                ),
            },
            {
                "utc": "2026-07-23T18:56:09Z",
                "application": (
                    "exactly diff E2 per-cell roles against DV2 aggregate "
                    "facts; falsify the unverified 4.6x same-content claim"
                ),
            },
            {
                "utc": "2026-07-23T19:00:52Z",
                "application": ("reserve prosody amplitudes as a separate tolerance and redundancy axis"),
            },
            {
                "utc": "2026-07-23T19:11:51Z",
                "application": (
                    "reserve a per-stratum frequency-band prosody axis; admit "
                    "zero-byte truncation only after an exact R-null certificate"
                ),
            },
        ],
        "resume": {
            "per_probe_immutable_checkpoints": True,
            "checkpoint_count": len(rows),
            "completed_probe_ids": [row["probe_id"] for row in rows],
            "measured_probe_ids": [row["probe_id"] for row in measured_rows],
            "receiver_invalid_probe_ids": [row["probe_id"] for row in blocked_rows],
        },
        "verdict": (
            "MEASURED_E2_SPARSE_COSTATE_ROWS; BLOCKED_SDWL1_LADDER_MODE_RERACE_DR1_RECORD_PRUNING_AND_XI_DIRECTION"
        ),
        "verdict_scope": (
            "E2 coordinate instances and the explicit SDWL1/E2/DR1 "
            "formulation bridges named in each negative. No family or "
            "paradigm verdict, score claim, promotion, or pointer move."
        ),
    }
    if not all(receipt["settled_law_checks"].values()):
        raise MeasurementError("settled law cross-check failed")
    _publish_immutable(receipt_path, _canonical_json(receipt))
    print(
        json.dumps(
            {
                "resumed": False,
                "complete": True,
                "receipt": str(receipt_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(receipt_path.read_bytes()),
                "verdict": receipt["verdict"],
            },
            sort_keys=True,
        )
    )
    return receipt_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    config_path = args.config.resolve()
    try:
        config_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise SystemExit("config must live inside the worktree") from exc
    config = DDMDR2BToleranceCostateConfigV1.model_validate_json(config_path.read_bytes())
    run(config, config_path=config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
