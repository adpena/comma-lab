#!/usr/bin/env python3
"""Build and advisory-measure one receiver-closed DDM V9 carrier archive."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import HEAD_PAIR_NORMS  # noqa: E402
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    CLASS_ORDER,
    EVIDENCE_AXIS,
    RESULT_SCHEMA,
    RESULT_SCHEMA_V2,
    RESULT_SCHEMA_V3,
    BoundaryCoefficientDelta,
    BoundaryShearletAtomV1,
    DirectDescriptionV9CarrierComposeConfigV1,
    DirectDescriptionV10FisherEventSearchConfigV1,
    DirectDescriptionV11ObligationSearchConfigV1,
    IslandShapeAtomV1,
    TopologyEventV1,
    compile_carrier_compose_archive,
    prove_carrier_archive_fail_closed,
    receive_carrier_compose_archive,
    recursive_carrier_byte_rows,
)
from tac.optimization.direct_description_entropy_priced_member import (  # noqa: E402
    _load_posenet_oracle,
    _measure_evaluator_bridge,
    _storage_preflight,
)
from tac.optimization.direct_description_measurement_ladder import load_target_receipt  # noqa: E402
from tac.optimization.direct_description_minimizer import (  # noqa: E402
    POINTER_SCORE_TEXT,
    SOURCE_BYTES,
    DirectDescriptionError,
    _publish_new_bytes,
    _read_regular_file_once,
    _sha256,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_polytope_membership import _load_segnet_oracle  # noqa: E402
from tac.optimization.predictor_upgrade_xi_chart import LaneCoefficientDelta  # noqa: E402


def _bound_bytes(path: Path, digest: str, name: str) -> bytes:
    payload = _read_regular_file_once(path)
    if _sha256(payload) != digest:
        raise DirectDescriptionError(f"{name} SHA-256 mismatch")
    return payload


def _bound_json(path: Path, digest: str, name: str) -> dict[str, Any]:
    try:
        value = json.loads(_bound_bytes(path, digest, name))
    except json.JSONDecodeError as exc:
        raise DirectDescriptionError(f"{name} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise DirectDescriptionError(f"{name} must contain one JSON object")
    return value


def _atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    encoded = rfc8785_canonicalize(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_regular_file_once(path) != encoded:
            raise DirectDescriptionError(f"checkpoint exists with different bytes: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    os.replace(temporary, path)


def _publish_identical_or_new(path: Path, payload: bytes) -> Path:
    """Resume-safe immutable publication: an existing path must be byte-identical."""

    if path.exists():
        if _read_regular_file_once(path) != payload:
            raise DirectDescriptionError(f"immutable output exists with different bytes: {path}")
        return path
    return _publish_new_bytes(path, payload)


def _objective(archive_bytes: int, d_seg: str, d_pose: str) -> str:
    value = 100.0 * float(d_seg) + math.sqrt(10.0 * float(d_pose)) + 25.0 * archive_bytes / SOURCE_BYTES
    return f"{value:.12f}"


def _joint_objective_delta(
    *,
    current_errors: int,
    proposed_errors: int,
    sites: int,
    current_dpose: float,
    proposed_dpose: float,
    marginal_bytes: int,
) -> tuple[float, float, float, float]:
    seg_delta = 100.0 * (proposed_errors - current_errors) / sites
    pose_delta = math.sqrt(10.0 * proposed_dpose) - math.sqrt(10.0 * current_dpose)
    rate_delta = 25.0 * marginal_bytes / SOURCE_BYTES
    return seg_delta, pose_delta, rate_delta, seg_delta + pose_delta + rate_delta


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _candidate_row(v6_receipt: dict[str, Any], config: Any) -> dict[str, Any]:
    for row in v6_receipt.get("candidates", ()):
        archive = row.get("archive", {})
        if (
            row.get("mode") == "fixed_ar1_hold24"
            and archive.get("path") == config.predictor_archive_path
            and archive.get("sha256") == config.predictor_archive_sha256
        ):
            return row
    raise DirectDescriptionError("typed predictor is not the bound v6 fixed_ar1_hold24 candidate")


def run(config: DirectDescriptionV9CarrierComposeConfigV1, output_directory: Path, semantic_argv: list[str]) -> Path:
    root = output_directory
    storage = _storage_preflight(root.resolve())
    output_tier = Path(str(storage.get("output_tier", root)))
    storage["output_tier"] = _portable_path(output_tier)
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / f"ddm_v9_carrier_compose_n{config.pair_count}_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(_read_regular_file_once(receipt_path))
        if receipt.get("typed_config_sha256") != config.typed_config_hash():
            raise DirectDescriptionError("completed receipt typed-config hash differs")
        archive = _bound_bytes(Path(receipt["archive"]["path"]), receipt["archive"]["sha256"], "completed v9 archive")
        receive_carrier_compose_archive(archive)
        print(json.dumps({"resumed": True, "receipt": str(receipt_path), "verdict": receipt["verdict"]}))
        return receipt_path

    v6_receipt = _bound_json(Path(config.v6_receipt_path), config.v6_receipt_sha256, "v6 receipt")
    if v6_receipt.get("schema") != "direct_description_dseg_bridge_amortize.v1":
        raise DirectDescriptionError("input is not the governed v6 receipt")
    v6_typed = v6_receipt.get("typed_config", {})
    if (v6_typed.get("pair_start"), v6_typed.get("pair_count")) != (config.pair_start, config.pair_count):
        raise DirectDescriptionError("v9 window differs from bound v6 window")
    _candidate_row(v6_receipt, config)
    predictor_archive = _bound_bytes(
        Path(config.predictor_archive_path), config.predictor_archive_sha256, "v6 predictor archive"
    )

    started = time.perf_counter()
    archive, homes = compile_carrier_compose_archive(predictor_archive, config.symbols())
    receiver = receive_carrier_compose_archive(archive)
    fail_closed = prove_carrier_archive_fail_closed(archive)
    build_seconds = time.perf_counter() - started
    archive_path = root / f"ddm_v9_carrier_compose_n{config.pair_count}.not_a_candidate.zip.receipt-bytes"
    _publish_new_bytes(archive_path, archive)
    _atomic_checkpoint(
        root / "stage_checkpoints" / "01_receiver_closed_build.json",
        {
            "schema": "ddm_v9_carrier_compose_stage_checkpoint.v1",
            "stage": "receiver_closed_build",
            "typed_config_sha256": config.typed_config_hash(),
            "archive": {"path": _portable_path(archive_path), "bytes": len(archive), "sha256": _sha256(archive)},
            "receiver_custody": dict(receiver.custody),
        },
    )

    v5_receipt = _bound_json(Path(v6_typed["v5_receipt_path"]), v6_typed["v5_receipt_sha256"], "v6-bound v5 receipt")
    v5_typed = v5_receipt.get("typed_config", {})
    target_receipt = load_target_receipt(Path(v5_typed["target_receipt_path"]), v5_typed["target_receipt_sha256"])
    cache_path = Path(target_receipt.source_cache.path)
    if not cache_path.is_file() or cache_path.stat().st_size != target_receipt.source_cache.bytes:
        raise DirectDescriptionError("frozen scorer cache is unavailable")
    cached_lstars = open_stored_npy_memmap(cache_path, "lstars")
    cached_margins = open_stored_npy_memmap(cache_path, "margins")
    cached_poses = open_stored_npy_memmap(cache_path, "gt_poses")
    segnet_oracle, segnet_custody = _load_segnet_oracle(Path(config.upstream_root), threads=config.scorer_threads)
    posenet_oracle, posenet_custody = _load_posenet_oracle(Path(config.upstream_root), threads=config.scorer_threads)
    measured_at = time.perf_counter()
    bridge = _measure_evaluator_bridge(
        receiver,  # duck-typed: z + render_pairs are the governed bridge surface.
        pair_start=config.pair_start,
        cached_lstars=cached_lstars,
        cached_margins=cached_margins,
        cached_poses=cached_poses,
        segnet_oracle=segnet_oracle,
        posenet_oracle=posenet_oracle,
        batch_size=config.scorer_batch_size,
    )
    measure_seconds = time.perf_counter() - measured_at
    d_seg = bridge["segmentation"]["d_seg"]
    d_pose = bridge["pose"]["d_pose"]
    class_rows = bridge["segmentation"]["strata"]["target_class"]
    byte_rows = recursive_carrier_byte_rows(archive)
    for row in byte_rows:
        stratum = row["stratum"]
        if stratum in class_rows:
            row["d_seg"] = class_rows[stratum]["d_seg"]
        elif stratum == "xi/Pose6":
            row["d_seg"] = d_seg
        else:
            row["d_seg"] = None
        row["d_pose"] = d_pose if stratum != "chart_symbol_refinement" or row["nested_unique_home_bytes"] else None
        row["causal_attribution_scope"] = (
            "target-class conditional composite error; d_pose shared-composite, not leave-one-out"
        )
    objective = _objective(len(archive), d_seg, d_pose)
    under_box = len(archive) <= 154_600 and float(d_seg) <= 0.00116
    verdict = (
        "ADVISORY_INSTANCE_MEETS_SUB015_BOX_NOT_PROMOTABLE"
        if under_box
        else "ADVISORY_INSTANCE_FAILS_SUB015_BOX_FORMULATION_OPEN"
    )
    receipt: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "lane_id": "ddm_v9_carrier_compose_byteclose",
        "tasks": [603, 613],
        "run_id": config.run_id,
        "seed": config.seed,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "semantic_argv": semantic_argv,
        "archive": {
            "path": _portable_path(archive_path),
            "bytes": len(archive),
            "sha256": _sha256(archive),
            "member_homes": list(homes),
            "parse_reencode_identical": True,
            "receiver_closed": True,
            "all_bytes_have_one_home": sum(row["zip_home_bytes"] for row in homes) == len(archive),
        },
        "per_stratum": byte_rows,
        "bridge": bridge,
        "objective_advisory": {
            "score": objective,
            "d_seg": d_seg,
            "d_pose": d_pose,
            "archive_bytes": len(archive),
            "formula": "100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489",
        },
        "boxes": {
            "sub_0p15": {"max_bytes": 154600, "max_d_seg": "0.001160000000", "met": under_box},
            "pointer_knee_bytes": 216300,
        },
        "correction": {
            "symbols": len(config.symbols()),
            "policy": config.correction_policy,
            "pixel_residual_present": False,
            "admission": "empty unless hard-oracle improvement is preselected by typed config",
            "fisher_margin_curvature_ranker": "required upstream for nonempty symbols; no blanket fixes",
        },
        "receiver_custody": dict(receiver.custody),
        "fail_closed_mutation_proof": fail_closed,
        "scorer_custody": {"segnet": segnet_custody, "posenet": posenet_custody},
        "target_custody": {
            "receipt_path": v5_typed["target_receipt_path"],
            "receipt_sha256": v5_typed["target_receipt_sha256"],
            "cache_path": str(cache_path),
            "cache_bytes": target_receipt.source_cache.bytes,
            "cache_sha256": target_receipt.source_cache.sha256,
        },
        "wallclock": {
            "build_seconds": f"{build_seconds:.6f}",
            "measure_seconds": f"{measure_seconds:.6f}",
            "n600_projection_seconds": f"{(build_seconds + measure_seconds) * 600 / config.pair_count:.6f}",
            "n600_status": "WALLCLOCK_PROJECTION_ONLY_NOT_RUN",
        },
        "storage_preflight": storage,
        "resume": {
            "policy": config.checkpoint_policy,
            "stage_checkpoint": _portable_path(root / "stage_checkpoints" / "01_receiver_closed_build.json"),
            "all_preserved": True,
        },
        "verdict": verdict,
        "verdict_scope": (
            "This exact v6-fixed predictor plus five settled carrier payloads and empty-or-explicit G2CS1 "
            "chart refinement on the stated bridge window only. Failure does not close joint multicoefficient "
            "chart solves, xi-transported birth/death events, corrected inner-Jacobian realization, or the V9 family."
        ),
        "blocker_delta": (
            "Receiver closure and per-stratum byte/Seg/Pose accounting are discharged. Remaining primary DOF is "
            "a Fisher-margin/curvature-ranked joint chart-symbol plus birth/death-event solve with pose-tube "
            "admission; current inherited Pose6 stream is sole-owner but not yet triple-used as carrier transport."
        ),
        "stores_consulted": [
            config.v6_receipt_path,
            config.predictor_archive_path,
            v6_typed["v5_receipt_path"],
            v5_typed["target_receipt_path"],
            str(cache_path),
            "reports/latest.md",
            ".omx/state/lane_registry.json",
            ".omx/state/canonical_task_status.jsonl",
        ],
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "d_seg_claim": False,
        "d_pose_claim": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
    }
    _atomic_checkpoint(
        root / "stage_checkpoints" / "02_frozen_scorer_measurement.json",
        {
            "schema": "ddm_v9_carrier_compose_stage_checkpoint.v1",
            "stage": "frozen_scorer_measurement",
            "typed_config_sha256": config.typed_config_hash(),
            "archive_sha256": _sha256(archive),
            "d_seg": d_seg,
            "d_pose": d_pose,
            "evidence_axis": EVIDENCE_AXIS,
            "score_claim": False,
        },
    )
    _publish_new_bytes(receipt_path, rfc8785_canonicalize(receipt))
    print(json.dumps({"resumed": False, "receipt": str(receipt_path), "verdict": verdict}))
    return receipt_path


@dataclass(frozen=True, slots=True)
class _SearchCandidate:
    candidate_id: str
    mechanism: str
    fisher_priority: float
    source_pair_ids: tuple[int, ...]
    lane_symbols: tuple[LaneCoefficientDelta, ...] = ()
    boundary_symbols: tuple[BoundaryCoefficientDelta, ...] = ()
    topology_events: tuple[TopologyEventV1, ...] = ()
    boundary_shearlets: tuple[BoundaryShearletAtomV1, ...] = ()
    island_shapes: tuple[IslandShapeAtomV1, ...] = ()

    def row(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "mechanism": self.mechanism,
            "fisher_priority": f"{self.fisher_priority:.12e}",
            "source_pair_ids": list(self.source_pair_ids),
            "lane_symbols": [asdict(value) for value in self.lane_symbols],
            "boundary_symbols": [asdict(value) for value in self.boundary_symbols],
            "topology_events": [asdict(value) for value in self.topology_events],
            "boundary_shearlets": [asdict(value) for value in self.boundary_shearlets],
            "island_shapes": [asdict(value) for value in self.island_shapes],
        }

    def fingerprint(self) -> str:
        return _sha256(rfc8785_canonicalize(self.row()))

    def conflict_keys(self) -> frozenset[tuple[Any, ...]]:
        keys: set[tuple[Any, ...]] = {
            ("lane", row.pair_index, row.line_index, row.coefficient_index) for row in self.lane_symbols
        }
        keys.update(
            ("boundary", row.pair_index, row.role, row.coefficient_index) for row in self.boundary_symbols
        )
        keys.update(
            (
                "event",
                row.pair_index,
                row.role,
                row.action,
                row.shape,
                row.y0,
                row.x0,
                row.y1,
                row.x1,
            )
            for row in self.topology_events
        )
        keys.update(
            ("shearlet", row.pair_index, row.role, row.center_y, row.center_x)
            for row in self.boundary_shearlets
        )
        keys.update(
            ("island", row.pair_index, row.action, row.center_y, row.center_x)
            for row in self.island_shapes
        )
        return frozenset(keys)


@dataclass(slots=True)
class _BatchScore:
    cells: np.ndarray
    poses: np.ndarray
    errors: int
    pose_squared_error: float


def _candidate_symbols(
    candidates: list[_SearchCandidate],
) -> tuple[tuple[LaneCoefficientDelta, ...], tuple[BoundaryCoefficientDelta, ...], tuple[TopologyEventV1, ...]]:
    lane = tuple(sorted(row for candidate in candidates for row in candidate.lane_symbols))
    boundary = tuple(sorted(row for candidate in candidates for row in candidate.boundary_symbols))
    events = tuple(sorted(row for candidate in candidates for row in candidate.topology_events))
    return lane, boundary, events


def _compile_candidates(
    predictor_archive: bytes, candidates: list[_SearchCandidate]
) -> tuple[bytes, Any]:
    lane, boundary, events = _candidate_symbols(candidates)
    archive, _homes = compile_carrier_compose_archive(
        predictor_archive,
        lane,
        boundary_symbols=boundary,
        topology_events=events,
    )
    return archive, receive_carrier_compose_archive(archive)


def _obligation_symbols(
    candidates: list[_SearchCandidate],
) -> tuple[
    tuple[LaneCoefficientDelta, ...],
    tuple[BoundaryShearletAtomV1, ...],
    tuple[IslandShapeAtomV1, ...],
]:
    lane = tuple(sorted(row for candidate in candidates for row in candidate.lane_symbols))
    shearlets = tuple(sorted(row for candidate in candidates for row in candidate.boundary_shearlets))
    islands = tuple(sorted(row for candidate in candidates for row in candidate.island_shapes))
    return lane, shearlets, islands


def _compile_obligation_candidates(
    predictor_archive: bytes,
    candidates: list[_SearchCandidate],
) -> tuple[bytes, Any]:
    lane, shearlets, islands = _obligation_symbols(candidates)
    archive, _homes = compile_carrier_compose_archive(
        predictor_archive,
        lane,
        boundary_shearlets=shearlets,
        island_shapes=islands,
        obligation_vocabulary=True,
    )
    return archive, receive_carrier_compose_archive(archive)


def _pair_norm(target: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    result = np.ones(target.shape, dtype=np.float64)
    for left in range(len(CLASS_ORDER)):
        for right in range(left + 1, len(CLASS_ORDER)):
            key = f"{CLASS_ORDER[left]}-{CLASS_ORDER[right]}"
            mask = ((target == left) & (predicted == right)) | ((target == right) & (predicted == left))
            result[mask] = HEAD_PAIR_NORMS[key]
    return result


def _fisher_priority_map(target: np.ndarray, predicted: np.ndarray, margin: np.ndarray) -> np.ndarray:
    """Rank error sites by exact head flip distance, margin band, and curvature.

    The exact rank-4 law supplies ``flip_distance = margin / ||w_c-w_c'||``.
    Lower distance is more actionable, so the acquisition value uses its
    reciprocal, multiplied by logistic Fisher curvature and a preregistered
    near-boundary margin-band weight.  This proposes candidates only; the real
    scorer replay below is the admission authority.
    """

    absolute_margin = np.abs(np.asarray(margin, dtype=np.float64))
    distance = absolute_margin / _pair_norm(target, predicted)
    curvature = 0.5 / np.square(np.cosh(np.minimum(absolute_margin, 20.0) / 2.0))
    band = np.select(
        [absolute_margin < 0.1, absolute_margin < 0.5, absolute_margin < 1.0],
        [4.0, 2.0, 1.0],
        default=0.25,
    )
    value = curvature * band / np.maximum(distance, 1.0e-3)
    value[target == predicted] = 0.0
    return value


def _component_boxes(mask: np.ndarray, minimum_sites: int, maximum: int) -> list[tuple[int, int, int, int, int]]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    rows: list[tuple[int, int, int, int, int]] = []
    for component, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        sites = int(np.count_nonzero(labels[slices] == component))
        if sites < minimum_sites:
            continue
        y_slice, x_slice = slices
        rows.append((sites, y_slice.start, x_slice.start, y_slice.stop, x_slice.stop))
    rows.sort(reverse=True)
    return rows[:maximum]


def _transported_event(
    *,
    pair_index: int,
    role: str,
    action: str,
    bbox: tuple[int, int, int, int],
    current_mask: np.ndarray,
    next_mask: np.ndarray | None,
    pose6_codes: np.ndarray,
    local_pair_id: int,
) -> TopologyEventV1:
    y0, x0, y1, x1 = bbox
    if next_mask is None or local_pair_id + 1 >= len(pose6_codes):
        return TopologyEventV1(pair_index, role, action, "ellipse", 1, y0, x0, y1, x1)
    expanded = np.zeros_like(next_mask, dtype=bool)
    expanded[max(0, y0 - 24) : min(384, y1 + 24), max(0, x0 - 24) : min(512, x1 + 24)] = True
    next_sites = np.argwhere(next_mask & expanded)
    current_sites = np.argwhere(current_mask[y0:y1, x0:x1])
    if next_sites.size == 0 or current_sites.size == 0:
        return TopologyEventV1(pair_index, role, action, "ellipse", 1, y0, x0, y1, x1)
    current_center = current_sites.mean(axis=0) + np.asarray((y0, x0))
    next_center = next_sites.mean(axis=0)
    delta_y, delta_x = next_center - current_center
    pose_delta = pose6_codes[local_pair_id + 1].astype(np.int16) - pose6_codes[local_pair_id].astype(np.int16)
    gains = []
    for displacement, code_delta in ((delta_x, int(pose_delta[0])), (delta_y, int(pose_delta[1]))):
        value = 0 if code_delta == 0 else int(np.rint(float(displacement) * 16.0 / code_delta))
        gains.append(max(-128, min(127, value)))
    if gains == [0, 0]:
        return TopologyEventV1(pair_index, role, action, "ellipse", 1, y0, x0, y1, x1)
    return TopologyEventV1(pair_index, role, action, "ellipse", 2, y0, x0, y1, x1, gains[0], gains[1])


def _select_diverse_candidates(
    candidates: list[_SearchCandidate], *, maximum: int, minimum_per_family: int
) -> list[_SearchCandidate]:
    """Preserve Fisher order while preventing a global cutoff from erasing a mechanism family."""

    family = {
        "Road/cubic_boundary_coefficients": "road_boundary",
        "Lane/G2CS1_centerline_c3": "lane_chart",
    }
    buckets: dict[str, list[_SearchCandidate]] = {
        "road_boundary": [],
        "lane_chart": [],
        "lane_event": [],
        "movable_event": [],
    }
    ordered = sorted(candidates, key=lambda row: (-row.fisher_priority, row.candidate_id))
    for candidate in ordered:
        key = family.get(candidate.mechanism)
        if key is None:
            key = "lane_event" if candidate.mechanism.startswith("Lane/") else "movable_event"
        buckets[key].append(candidate)
    selected: list[_SearchCandidate] = []
    selected_ids: set[str] = set()
    for key in ("road_boundary", "lane_chart", "lane_event", "movable_event"):
        for candidate in buckets[key][:minimum_per_family]:
            selected.append(candidate)
            selected_ids.add(candidate.candidate_id)
    for candidate in ordered:
        if len(selected) >= maximum:
            break
        if candidate.candidate_id not in selected_ids:
            selected.append(candidate)
            selected_ids.add(candidate.candidate_id)
    selected.sort(key=lambda row: (-row.fisher_priority, row.candidate_id))
    return selected


def _derive_candidates(
    receiver: Any,
    *,
    target_cells: np.ndarray,
    target_margins: np.ndarray,
    described_cells: np.ndarray,
    config: DirectDescriptionV10FisherEventSearchConfigV1,
) -> list[_SearchCandidate]:
    candidates: list[_SearchCandidate] = []
    lane_layer = next(layer for layer in receiver.layers if layer.role == "Lane")
    if lane_layer.lane_lines is None or lane_layer.lane_header is None:
        raise DirectDescriptionError("v10 search requires decoded Lane lines")
    camera = receiver.predictor.camera
    for local_pair_id in range(config.pair_count):
        source_pair_id = config.pair_start + local_pair_id
        target = np.asarray(target_cells[local_pair_id])
        predicted = np.asarray(described_cells[local_pair_id])
        priority = _fisher_priority_map(target, predicted, target_margins[local_pair_id])

        lane_miss = (target == 1) & (predicted != 1)
        if int(np.count_nonzero(lane_miss)) >= config.min_component_sites:
            sites = np.argwhere(lane_miss)
            y_center, x_center = sites.mean(axis=0)
            forward = float(camera["height_m"]) * float(camera["fy_scorer"]) / max(
                float(y_center) - float(lane_layer.lane_header["v_h"]), 1.0
            )
            centers: list[float] = []
            for vector in lane_layer.lane_lines[source_pair_id]:
                centers.append(
                    float(lane_layer.lane_header.get("cx") or 256.0)
                    - float(np.polyval(vector[:4], forward)) * float(camera["fx_scorer"]) / forward
                )
            if centers:
                line_index = int(np.argmin(np.abs(np.asarray(centers) - x_center)))
                lateral_delta = -(float(x_center) - centers[line_index]) * forward / float(camera["fx_scorer"])
                for scale in (0.5, 1.0):
                    delta = float(np.float32(lateral_delta * scale))
                    if delta != 0.0 and np.isfinite(delta):
                        candidates.append(
                            _SearchCandidate(
                                f"lane_{source_pair_id}_{line_index}_c3_a{scale:g}",
                                "Lane/G2CS1_centerline_c3",
                                float(priority[lane_miss].sum()) * scale,
                                (source_pair_id,),
                                lane_symbols=(LaneCoefficientDelta(source_pair_id, line_index, 3, delta),),
                            )
                        )

        road_miss = (target == 0) & (predicted != 0)
        if int(np.count_nonzero(road_miss)) >= config.min_component_sites:
            xs: list[int] = []
            shifts: list[float] = []
            for x in range(512):
                target_rows = np.flatnonzero(target[:, x] == 0)
                predicted_rows = np.flatnonzero(predicted[:, x] == 0)
                if target_rows.size and predicted_rows.size:
                    xs.append(x)
                    shifts.append(float(target_rows[0] - predicted_rows[0]))
            if len(xs) >= 16:
                coefficients = np.polynomial.polynomial.polyfit(
                    np.asarray(xs, dtype=np.float64) / 255.5 - 1.0,
                    np.clip(shifts, -32.0, 32.0),
                    3,
                )
                for scale in (0.5, 1.0):
                    rows = tuple(
                        BoundaryCoefficientDelta(source_pair_id, "Road", index, float(np.float32(value * scale)))
                        for index, value in enumerate(coefficients)
                        if abs(float(np.float32(value * scale))) >= 0.25
                    )
                    if rows:
                        candidates.append(
                            _SearchCandidate(
                                f"road_{source_pair_id}_cubic_a{scale:g}",
                                "Road/cubic_boundary_coefficients",
                                float(priority[road_miss].sum()) * scale,
                                (source_pair_id,),
                                boundary_symbols=rows,
                            )
                        )

        for role, class_id in (("Lane", 1), ("Movable", 3)):
            for action, mask in (
                ("birth", (target == class_id) & (predicted != class_id)),
                ("death", (predicted == class_id) & (target != class_id)),
            ):
                boxes = _component_boxes(mask, config.min_component_sites, config.max_components_per_pair_role)
                next_mask = None
                if local_pair_id + 1 < config.pair_count:
                    next_target = np.asarray(target_cells[local_pair_id + 1])
                    next_predicted = np.asarray(described_cells[local_pair_id + 1])
                    next_mask = (
                        (next_target == class_id) & (next_predicted != class_id)
                        if action == "birth"
                        else (next_predicted == class_id) & (next_target != class_id)
                    )
                for ordinal, (_sites, y0, x0, y1, x1) in enumerate(boxes):
                    event = _transported_event(
                        pair_index=source_pair_id,
                        role=role,
                        action=action,
                        bbox=(y0, x0, y1, x1),
                        current_mask=mask,
                        next_mask=next_mask,
                        pose6_codes=receiver.pose6_codes,
                        local_pair_id=local_pair_id,
                    )
                    impacted = tuple(range(source_pair_id, source_pair_id + event.lifetime))
                    candidates.append(
                        _SearchCandidate(
                            f"event_{role}_{action}_{source_pair_id}_{ordinal}",
                            f"{role}/{action}_bbox_{event.shape}_xi_transport",
                            float(priority[y0:y1, x0:x1][mask[y0:y1, x0:x1]].sum()),
                            impacted,
                            topology_events=(event,),
                        )
                    )
    return _select_diverse_candidates(
        candidates,
        maximum=config.max_candidates,
        minimum_per_family=config.minimum_candidates_per_family,
    )


_OBLIGATION_FAMILY_ORDER = (
    "lane_center",
    "lane_width",
    "lane_phase",
    "road_boundary",
    "undrivable_boundary",
    "movable_shape",
)


def _obligation_family(candidate: _SearchCandidate) -> str:
    mechanism = candidate.mechanism
    if mechanism.startswith("Lane/center"):
        return "lane_center"
    if mechanism.startswith("Lane/width"):
        return "lane_width"
    if mechanism.startswith("Lane/dash_phase"):
        return "lane_phase"
    if mechanism.startswith("Road/"):
        return "road_boundary"
    if mechanism.startswith("Undrivable/"):
        return "undrivable_boundary"
    if mechanism.startswith("Movable/"):
        return "movable_shape"
    raise DirectDescriptionError(f"v11 obligation candidate has unknown family: {mechanism}")


def _select_obligation_candidates(
    generated: list[_SearchCandidate],
    *,
    pair_start: int,
    maximum: int,
    minimum_per_family: int,
) -> list[_SearchCandidate]:
    ordered = sorted(generated, key=lambda row: (-row.fisher_priority, row.candidate_id))
    buckets = {name: [] for name in _OBLIGATION_FAMILY_ORDER}
    for candidate in ordered:
        buckets[_obligation_family(candidate)].append(candidate)
    selected: list[_SearchCandidate] = []
    selected_ids: set[str] = set()
    for family in _OBLIGATION_FAMILY_ORDER:
        for candidate in buckets[family][:minimum_per_family]:
            selected.append(candidate)
            selected_ids.add(candidate.candidate_id)
    for candidate in ordered:
        if len(selected) >= maximum:
            break
        if candidate.candidate_id not in selected_ids:
            selected.append(candidate)
            selected_ids.add(candidate.candidate_id)
    selected.sort(key=lambda row: (-row.fisher_priority, row.candidate_id))
    return selected[:maximum]


def _xi_transport_gains(
    *,
    current_center: np.ndarray,
    next_center: np.ndarray | None,
    pose6_codes: np.ndarray,
    local_pair_id: int,
) -> tuple[int, int, int]:
    if next_center is None or local_pair_id + 1 >= len(pose6_codes):
        return 1, 0, 0
    delta_y, delta_x = next_center - current_center
    pose_delta = pose6_codes[local_pair_id + 1].astype(np.int16) - pose6_codes[local_pair_id].astype(np.int16)
    gains: list[int] = []
    for displacement, code_delta in ((delta_x, int(pose_delta[0])), (delta_y, int(pose_delta[1]))):
        value = 0 if code_delta == 0 else int(np.rint(float(displacement) * 16.0 / code_delta))
        gains.append(max(-128, min(127, value)))
    return (2, gains[0], gains[1]) if gains != [0, 0] else (1, 0, 0)


def _bundle_obligation_candidates(
    generated: list[_SearchCandidate],
    *,
    pair_start: int,
    batch_size: int,
    maximum_bundles: int,
    maximum_atoms: int,
    minimum_per_family: int,
) -> list[_SearchCandidate]:
    """Group atomic obligations by scorer batch and obligation family.

    A mixed whole-batch bundle aliases a useful local atom with unrelated harmful
    strata.  Family-specific bundles preserve exact batch reuse while keeping the
    measured admission question surgical enough for reverse water-filling.
    """

    by_batch: dict[int, list[_SearchCandidate]] = {}
    for candidate in generated:
        first = min(candidate.source_pair_ids)
        batch_start = ((first - pair_start) // batch_size) * batch_size
        if any(
            ((source_id - pair_start) // batch_size) * batch_size != batch_start
            for source_id in candidate.source_pair_ids
        ):
            continue
        by_batch.setdefault(batch_start, []).append(candidate)
    family_mechanisms = {
        "lane_center": "Lane/center_canonical_batch_obligation_bundle",
        "lane_width": "Lane/width_canonical_batch_obligation_bundle",
        "lane_phase": "Lane/dash_phase_canonical_batch_obligation_bundle",
        "road_boundary": "Road/canonical_batch_shearlet_obligation_bundle",
        "undrivable_boundary": "Undrivable/canonical_batch_shearlet_obligation_bundle",
        "movable_shape": "Movable/canonical_batch_shape_obligation_bundle",
    }
    bundles: list[_SearchCandidate] = []
    for batch_start, rows in by_batch.items():
        ordered = sorted(rows, key=lambda row: (-row.fisher_priority, row.candidate_id))
        buckets = {name: [] for name in _OBLIGATION_FAMILY_ORDER}
        for row in ordered:
            buckets[_obligation_family(row)].append(row)
        for family in _OBLIGATION_FAMILY_ORDER:
            chosen: list[_SearchCandidate] = []
            occupied: set[tuple[Any, ...]] = set()
            for row in buckets[family]:
                if row.conflict_keys() & occupied:
                    continue
                chosen.append(row)
                occupied.update(row.conflict_keys())
                if len(chosen) >= maximum_atoms:
                    break
            if not chosen:
                continue
            lane = tuple(sorted(value for row in chosen for value in row.lane_symbols))
            shearlets = tuple(sorted(value for row in chosen for value in row.boundary_shearlets))
            islands = tuple(sorted(value for row in chosen for value in row.island_shapes))
            source_ids = tuple(sorted({value for row in chosen for value in row.source_pair_ids}))
            bundles.append(
                _SearchCandidate(
                    f"obligation_bundle_batch_{batch_start:04d}_{family}",
                    family_mechanisms[family],
                    sum(row.fisher_priority for row in chosen),
                    source_ids,
                    lane_symbols=lane,
                    boundary_shearlets=shearlets,
                    island_shapes=islands,
                )
            )
    return _select_obligation_candidates(
        bundles,
        pair_start=pair_start,
        maximum=maximum_bundles,
        minimum_per_family=minimum_per_family,
    )


def _derive_obligation_candidates(
    receiver: Any,
    *,
    target_cells: np.ndarray,
    target_margins: np.ndarray,
    described_cells: np.ndarray,
    config: DirectDescriptionV11ObligationSearchConfigV1,
) -> tuple[list[_SearchCandidate], list[_SearchCandidate], int]:
    """Derive chart atoms from measured scorer-visible error obligations."""

    generated: list[_SearchCandidate] = []
    lane_layer = next(layer for layer in receiver.layers if layer.role == "Lane")
    if lane_layer.lane_lines is None or lane_layer.lane_header is None:
        raise DirectDescriptionError("v11 obligation search requires decoded Lane lines")
    camera = receiver.predictor.camera
    for local_pair_id in range(config.pair_count):
        source_pair_id = config.pair_start + local_pair_id
        target = np.asarray(target_cells[local_pair_id])
        predicted = np.asarray(described_cells[local_pair_id])
        priority = _fisher_priority_map(target, predicted, target_margins[local_pair_id])

        lane_error = (target == 1) != (predicted == 1)
        if int(np.count_nonzero(lane_error)) >= config.min_component_sites:
            sites = np.argwhere(lane_error)
            y_center, x_center = sites.mean(axis=0)
            forward = float(camera["height_m"]) * float(camera["fy_scorer"]) / max(
                float(y_center) - float(lane_layer.lane_header["v_h"]), 1.0
            )
            centers: list[float] = []
            for vector in lane_layer.lane_lines[source_pair_id]:
                centers.append(
                    float(lane_layer.lane_header.get("cx") or 256.0)
                    - float(np.polyval(vector[:4], forward)) * float(camera["fx_scorer"]) / forward
                )
            if centers:
                line_index = int(np.argmin(np.abs(np.asarray(centers) - x_center)))
                vector = lane_layer.lane_lines[source_pair_id][line_index]
                pixel_shift = float(x_center) - centers[line_index]
                lane_priority = float(priority[lane_error].sum())
                for coefficient_index in range(4):
                    sensitivity = float(camera["fx_scorer"]) * max(forward ** (2 - coefficient_index), 1.0e-6)
                    base_delta = float(np.clip(-pixel_shift / sensitivity, -64.0, 64.0))
                    for scale in (0.5, 1.0):
                        delta = float(np.float32(base_delta * scale))
                        if delta != 0.0 and np.isfinite(delta):
                            generated.append(
                                _SearchCandidate(
                                    f"lane_{source_pair_id}_{line_index}_c{coefficient_index}_a{scale:g}",
                                    f"Lane/center_c{coefficient_index}_rank4_obligation",
                                    lane_priority * scale,
                                    (source_pair_id,),
                                    lane_symbols=(
                                        LaneCoefficientDelta(
                                            source_pair_id, line_index, coefficient_index, delta
                                        ),
                                    ),
                                )
                            )
                missing = int(np.count_nonzero((target == 1) & (predicted != 1)))
                excess = int(np.count_nonzero((predicted == 1) & (target != 1)))
                width_sign = 1.0 if missing >= excess else -1.0
                width_pixels = width_sign * float(np.clip(np.std(sites[:, 1]) / 4.0, 0.5, 8.0))
                for coefficient_index, base_delta in (
                    (4, width_pixels / max(float(y_center), 1.0)),
                    (5, width_pixels),
                ):
                    for scale in (0.5, 1.0):
                        delta = float(np.float32(base_delta * scale))
                        if delta != 0.0:
                            generated.append(
                                _SearchCandidate(
                                    f"lane_{source_pair_id}_{line_index}_width{coefficient_index}_a{scale:g}",
                                    f"Lane/width_c{coefficient_index}_obligation",
                                    lane_priority * scale,
                                    (source_pair_id,),
                                    lane_symbols=(
                                        LaneCoefficientDelta(
                                            source_pair_id, line_index, coefficient_index, delta
                                        ),
                                    ),
                                )
                            )
                period = float(vector[6])
                if period > 0.0:
                    xi_key = int(receiver.pose6_codes[local_pair_id, 0]) - int(
                        receiver.pose6_codes[local_pair_id, 1]
                    )
                    desired_phase = forward - 0.5 * float(vector[8]) * period + xi_key * period / 4096.0
                    phase_delta = (desired_phase - float(vector[7]) + period / 2.0) % period - period / 2.0
                    for scale in (0.5, 1.0):
                        delta = float(np.float32(phase_delta * scale))
                        if delta != 0.0:
                            generated.append(
                                _SearchCandidate(
                                    f"lane_{source_pair_id}_{line_index}_phase_a{scale:g}",
                                    "Lane/dash_phase_xi_keyed_obligation",
                                    lane_priority * scale,
                                    (source_pair_id,),
                                    lane_symbols=(LaneCoefficientDelta(source_pair_id, line_index, 7, delta),),
                                )
                            )

        for role, class_id in (("Road", 0), ("UndrivableBoundary", 2)):
            mismatch = (target == class_id) != (predicted == class_id)
            boxes = _component_boxes(mismatch, config.min_component_sites, config.max_components_per_pair_role)
            for ordinal, (site_count, y0, x0, y1, x1) in enumerate(boxes):
                component_sites = np.argwhere(mismatch[y0:y1, x0:x1]) + np.asarray((y0, x0))
                if component_sites.size == 0:
                    continue
                center_y, center_x = np.rint(component_sites.mean(axis=0)).astype(np.int64)
                centered = component_sites - component_sites.mean(axis=0)
                var_x = float(np.square(centered[:, 1]).mean()) + 1.0e-6
                shear = float((centered[:, 0] * centered[:, 1]).mean() / var_x)
                shear_q4 = int(np.clip(np.rint(shear * 16.0), -64, 64))
                scale_y = int(np.clip(max(2, (y1 - y0) // 2), 2, 48))
                scale_x = int(np.clip(max(2 * scale_y, (x1 - x0) * 2, 8), 4, 256))
                missing = int(np.count_nonzero((target[y0:y1, x0:x1] == class_id) & mismatch[y0:y1, x0:x1]))
                excess = site_count - missing
                inferred_sign = 1 if missing >= excess else -1
                amplitude = float(np.clip(max(1.0, (y1 - y0) / 2.0), 1.0, 24.0))
                family_name = "Road" if role == "Road" else "Undrivable"
                local_priority = float(priority[y0:y1, x0:x1][mismatch[y0:y1, x0:x1]].sum())
                for direction_rank, direction in enumerate((inferred_sign, -inferred_sign)):
                    for scale in (0.5, 1.0):
                        amplitude_q4 = int(np.clip(np.rint(direction * amplitude * scale * 16.0), -512, 512))
                        if amplitude_q4 == 0:
                            continue
                        atom = BoundaryShearletAtomV1(
                            source_pair_id,
                            role,
                            int(center_y),
                            int(center_x),
                            scale_y,
                            scale_x,
                            shear_q4,
                            amplitude_q4,
                        )
                        generated.append(
                            _SearchCandidate(
                                f"{family_name.lower()}_{source_pair_id}_{ordinal}_sh_d{direction_rank}_a{scale:g}",
                                f"{family_name}/boundary_compact_parabolic_shearlet",
                                local_priority * scale / (1.0 + direction_rank),
                                (source_pair_id,),
                                boundary_shearlets=(atom,),
                            )
                        )

        for action, mismatch in (
            ("birth", (target == 3) & (predicted != 3)),
            ("death", (predicted == 3) & (target != 3)),
        ):
            boxes = _component_boxes(mismatch, config.min_component_sites, config.max_components_per_pair_role)
            next_mask = None
            if local_pair_id + 1 < config.pair_count:
                next_target = np.asarray(target_cells[local_pair_id + 1])
                next_predicted = np.asarray(described_cells[local_pair_id + 1])
                next_mask = (
                    (next_target == 3) & (next_predicted != 3)
                    if action == "birth"
                    else (next_predicted == 3) & (next_target != 3)
                )
            for ordinal, (_site_count, y0, x0, y1, x1) in enumerate(boxes):
                points = np.argwhere(mismatch[y0:y1, x0:x1]) + np.asarray((y0, x0))
                if len(points) < config.min_component_sites:
                    continue
                center = points.mean(axis=0)
                centered = points - center
                covariance = np.cov(centered.T) if len(points) > 1 else np.eye(2)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                major = eigenvectors[:, int(np.argmax(eigenvalues))]
                angle = float(np.arctan2(major[0], major[1]) % np.pi)
                angle_u8 = int(np.rint(angle * 256.0 / np.pi)) % 256
                radius_y = max(1.0, (y1 - y0) / 2.0)
                radius_x = max(1.0, (x1 - x0) / 2.0)
                normalized_y = centered[:, 0] / radius_y
                normalized_x = centered[:, 1] / radius_x
                skew_q6 = int(np.clip(np.rint(np.mean(normalized_x**3) * 64.0), -96, 96))
                taper_q6 = int(np.clip(np.rint(np.mean(normalized_x**2 * normalized_y) * 64.0), -96, 96))
                curvelet_q6 = int(np.clip(np.rint(np.mean(normalized_x * normalized_y**2) * 64.0), -96, 96))
                next_center = None
                if next_mask is not None:
                    expanded = np.zeros_like(next_mask, dtype=bool)
                    expanded[max(0, y0 - 24) : min(384, y1 + 24), max(0, x0 - 24) : min(512, x1 + 24)] = True
                    next_points = np.argwhere(next_mask & expanded)
                    if next_points.size:
                        next_center = next_points.mean(axis=0)
                lifetime, gain_x, gain_y = _xi_transport_gains(
                    current_center=center,
                    next_center=next_center,
                    pose6_codes=receiver.pose6_codes,
                    local_pair_id=local_pair_id,
                )
                local_priority = float(priority[y0:y1, x0:x1][mismatch[y0:y1, x0:x1]].sum())
                for scale in (0.75, 1.0, 1.25):
                    atom = IslandShapeAtomV1(
                        source_pair_id,
                        action,
                        lifetime,
                        int(np.clip(np.rint(center[0]), 0, 383)),
                        int(np.clip(np.rint(center[1]), 0, 511)),
                        int(np.clip(np.rint(radius_y * scale), 1, 191)),
                        int(np.clip(np.rint(radius_x * scale), 1, 255)),
                        angle_u8,
                        skew_q6,
                        taper_q6,
                        curvelet_q6,
                        gain_x,
                        gain_y,
                    )
                    generated.append(
                        _SearchCandidate(
                            f"movable_{action}_{source_pair_id}_{ordinal}_shape_a{scale:g}",
                            f"Movable/{action}_moments_curvelet_xi_transport",
                            local_priority * scale,
                            tuple(range(source_pair_id, source_pair_id + lifetime)),
                            island_shapes=(atom,),
                        )
                    )

    raw_count = len(generated)
    generated = sorted(generated, key=lambda row: (-row.fisher_priority, row.candidate_id))[
        : config.max_generated_candidates
    ]
    measured = _bundle_obligation_candidates(
        generated,
        pair_start=config.pair_start,
        batch_size=config.scorer_batch_size,
        maximum_bundles=config.max_measured_candidates,
        maximum_atoms=config.max_atoms_per_measured_bundle,
        minimum_per_family=config.minimum_candidates_per_family,
    )
    return generated, measured, raw_count


def _score_batches(
    receiver: Any,
    *,
    batch_starts: list[int],
    pair_start: int,
    pair_count: int,
    batch_size: int,
    target_cells: np.ndarray,
    target_poses: np.ndarray,
    segnet_oracle: Any,
    posenet_oracle: Any,
) -> dict[int, _BatchScore]:
    result: dict[int, _BatchScore] = {}
    for start in batch_starts:
        pair_ids = tuple(range(start, min(pair_count, start + batch_size)))
        source_ids = pair_start + np.asarray(pair_ids, dtype=np.int64)
        described = receiver.render_pairs(pair_ids)
        cells, unexpected_margin = segnet_oracle(described, False)
        if unexpected_margin is not None:
            raise DirectDescriptionError("v10 exact scorer unexpectedly returned margins")
        poses = posenet_oracle(described)
        result[start] = _BatchScore(
            cells=np.ascontiguousarray(cells),
            poses=np.ascontiguousarray(poses),
            errors=int(np.count_nonzero(cells != target_cells[source_ids - pair_start])),
            pose_squared_error=float(np.square(poses - target_poses[source_ids - pair_start]).sum(dtype=np.float64)),
        )
    return result


def _score_totals(rows: dict[int, _BatchScore]) -> tuple[int, float]:
    return sum(row.errors for row in rows.values()), sum(row.pose_squared_error for row in rows.values())


def _fraction_text(numerator: int, denominator: int) -> str:
    return f"{numerator / denominator:.12f}" if denominator else "0.000000000000"


def _compact_bridge_from_batches(
    rows: dict[int, _BatchScore],
    *,
    target_cells: np.ndarray,
    target_margins: np.ndarray,
    target_poses: np.ndarray,
) -> dict[str, Any]:
    """Aggregate exact chunked scorer outputs without retaining RGB batches."""

    target_class = {name: {"errors": 0, "sites": 0} for name in CLASS_ORDER}
    topology = {name: {"errors": 0, "sites": 0} for name in ("boundary_codim1", "cell_interior")}
    bands = {
        name: {"errors": 0, "sites": 0}
        for name in ("[0,0.1)", "[0.1,0.5)", "[0.5,1)", "[1,inf)")
    }
    total_errors = 0
    total_sites = 0
    pose_squared_error = 0.0
    for start, row in sorted(rows.items()):
        stop = start + len(row.cells)
        target = np.asarray(target_cells[start:stop])
        margin = np.asarray(target_margins[start:stop])
        errors = row.cells != target
        boundary = ndimage.maximum_filter(target, size=(1, 3, 3), mode="nearest") != ndimage.minimum_filter(
            target, size=(1, 3, 3), mode="nearest"
        )
        total_errors += int(np.count_nonzero(errors))
        total_sites += int(errors.size)
        pose_squared_error += float(row.pose_squared_error)
        for class_id, name in enumerate(CLASS_ORDER):
            mask = target == class_id
            target_class[name]["errors"] += int(np.count_nonzero(errors & mask))
            target_class[name]["sites"] += int(np.count_nonzero(mask))
        for name, mask in (("boundary_codim1", boundary), ("cell_interior", ~boundary)):
            topology[name]["errors"] += int(np.count_nonzero(errors & mask))
            topology[name]["sites"] += int(np.count_nonzero(mask))
        for low, high, name in (
            (0.0, 0.1, "[0,0.1)"),
            (0.1, 0.5, "[0.1,0.5)"),
            (0.5, 1.0, "[0.5,1)"),
            (1.0, float("inf"), "[1,inf)"),
        ):
            mask = (margin >= low) & (margin < high)
            bands[name]["errors"] += int(np.count_nonzero(errors & mask))
            bands[name]["sites"] += int(np.count_nonzero(mask))
    coordinates = len(target_poses) * 6
    return {
        "segmentation": {
            "definition": "official frozen SegNet last-frame argmax disagreement against gt_n600.lstars",
            "d_seg": _fraction_text(total_errors, total_sites),
            "errors": total_errors,
            "sites": total_sites,
            "strata": {
                "target_class": {
                    name: {**value, "d_seg": _fraction_text(value["errors"], value["sites"])}
                    for name, value in target_class.items()
                },
                "topology": {
                    name: {**value, "d_seg": _fraction_text(value["errors"], value["sites"])}
                    for name, value in topology.items()
                },
                "target_margin": {
                    name: {**value, "d_seg": _fraction_text(value["errors"], value["sites"])}
                    for name, value in bands.items()
                },
            },
            "d_seg_measured": True,
            "d_seg_claim": False,
        },
        "pose": {
            "definition": "official frozen PoseNet YUV6 first-six-output MSE against gt_n600.gt_poses",
            "d_pose": f"{pose_squared_error / coordinates:.12f}",
            "squared_error_sum": f"{pose_squared_error:.12f}",
            "coordinates": coordinates,
            "d_pose_measured": True,
            "d_pose_claim": False,
        },
        "scorer_batch_size": 16,
        "max_rgb_batches_resident": 1,
        "argmax_cells_retained_for_joint_waterfill": True,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "promotion_eligible": False,
    }


def _stratified_byte_rows(archive: bytes, bridge: dict[str, Any]) -> list[dict[str, Any]]:
    d_seg = bridge["segmentation"]["d_seg"]
    d_pose = bridge["pose"]["d_pose"]
    class_rows = bridge["segmentation"]["strata"]["target_class"]
    rows = recursive_carrier_byte_rows(archive)
    for row in rows:
        stratum = row["stratum"]
        row["d_seg"] = class_rows[stratum]["d_seg"] if stratum in class_rows else d_seg
        row["d_pose"] = d_pose
        row["causal_attribution_scope"] = "shared-composite exact replay; not leave-one-out attribution"
    return rows


def run_v10_search(
    config: DirectDescriptionV10FisherEventSearchConfigV1,
    output_directory: Path,
    semantic_argv: list[str],
) -> Path:
    root = output_directory
    storage = _storage_preflight(root.resolve())
    output_tier = Path(str(storage.get("output_tier", root)))
    storage["output_tier"] = _portable_path(output_tier)
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / f"ddm_v10_fisher_event_search_n{config.pair_count}_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(_read_regular_file_once(receipt_path))
        if receipt.get("typed_config_sha256") != config.typed_config_hash():
            raise DirectDescriptionError("completed v10 receipt typed-config hash differs")
        for rung in receipt["ladder"]:
            _bound_bytes(Path(rung["archive"]["path"]), rung["archive"]["sha256"], "completed v10 rung")
        print(json.dumps({"resumed": True, "receipt": str(receipt_path), "verdict": receipt["verdict"]}))
        return receipt_path

    v6_receipt = _bound_json(Path(config.v6_receipt_path), config.v6_receipt_sha256, "v6 receipt")
    if v6_receipt.get("schema") != "direct_description_dseg_bridge_amortize.v1":
        raise DirectDescriptionError("input is not the governed v6 receipt")
    v6_typed = v6_receipt.get("typed_config", {})
    if (v6_typed.get("pair_start"), v6_typed.get("pair_count")) != (config.pair_start, config.pair_count):
        raise DirectDescriptionError("v10 window differs from bound v6 window")
    _candidate_row(v6_receipt, config)  # Both typed configs bind the same settled v6 predictor fields.
    predictor_archive = _bound_bytes(
        Path(config.predictor_archive_path), config.predictor_archive_sha256, "v6 predictor archive"
    )
    base_archive, base_receiver = _compile_candidates(predictor_archive, [])
    fail_closed = prove_carrier_archive_fail_closed(base_archive)

    v5_receipt = _bound_json(Path(v6_typed["v5_receipt_path"]), v6_typed["v5_receipt_sha256"], "v6-bound v5 receipt")
    v5_typed = v5_receipt.get("typed_config", {})
    target_receipt = load_target_receipt(Path(v5_typed["target_receipt_path"]), v5_typed["target_receipt_sha256"])
    cache_path = Path(target_receipt.source_cache.path)
    if not cache_path.is_file() or cache_path.stat().st_size != target_receipt.source_cache.bytes:
        raise DirectDescriptionError("frozen scorer cache is unavailable")
    cached_lstars = open_stored_npy_memmap(cache_path, "lstars")
    cached_margins = open_stored_npy_memmap(cache_path, "margins")
    cached_poses = open_stored_npy_memmap(cache_path, "gt_poses")
    source_slice = slice(config.pair_start, config.pair_start + config.pair_count)
    target_cells = np.asarray(cached_lstars[source_slice])
    target_margins = np.asarray(cached_margins[source_slice])
    target_poses = np.asarray(cached_poses[source_slice])
    segnet_oracle, segnet_custody = _load_segnet_oracle(Path(config.upstream_root), threads=config.scorer_threads)
    posenet_oracle, posenet_custody = _load_posenet_oracle(Path(config.upstream_root), threads=config.scorer_threads)
    all_batch_starts = list(range(0, config.pair_count, config.scorer_batch_size))
    search_started = time.perf_counter()
    baseline_batches = _score_batches(
        base_receiver,
        batch_starts=all_batch_starts,
        pair_start=config.pair_start,
        pair_count=config.pair_count,
        batch_size=config.scorer_batch_size,
        target_cells=target_cells,
        target_poses=target_poses,
        segnet_oracle=segnet_oracle,
        posenet_oracle=posenet_oracle,
    )
    baseline_errors, baseline_pose_squared_error = _score_totals(baseline_batches)
    described_cells = np.concatenate([baseline_batches[start].cells for start in all_batch_starts], axis=0)
    candidates = _derive_candidates(
        base_receiver,
        target_cells=target_cells,
        target_margins=target_margins,
        described_cells=described_cells,
        config=config,
    )
    inventory_sha256 = _sha256(rfc8785_canonicalize([candidate.row() for candidate in candidates]))
    _atomic_checkpoint(
        root / "stage_checkpoints" / "01_candidate_inventory.json",
        {
            "schema": "ddm_v10_fisher_event_search_checkpoint.v1",
            "stage": "candidate_inventory",
            "typed_config_sha256": config.typed_config_hash(),
            "candidate_count": len(candidates),
            "candidate_inventory_sha256": inventory_sha256,
            "candidate_families": sorted({candidate.mechanism for candidate in candidates}),
        },
    )

    accepted: list[_SearchCandidate] = []
    processed = 0
    candidate_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        checkpoint_path = root / "stage_checkpoints" / "candidates" / f"{index:04d}.json"
        if not checkpoint_path.exists():
            break
        row = json.loads(_read_regular_file_once(checkpoint_path))
        if (
            row.get("typed_config_sha256") != config.typed_config_hash()
            or row.get("candidate_inventory_sha256") != inventory_sha256
            or row.get("candidate_fingerprint") != candidate.fingerprint()
        ):
            raise DirectDescriptionError("candidate checkpoint differs from deterministic inventory")
        candidate_rows.append(row)
        if row["admitted"]:
            accepted.append(candidate)
        processed += 1

    current_archive, current_receiver = _compile_candidates(predictor_archive, accepted)
    if not accepted:
        current_batches = dict(baseline_batches)
        current_errors, current_pose_squared_error = baseline_errors, baseline_pose_squared_error
    else:
        current_batches = _score_batches(
            current_receiver,
            batch_starts=all_batch_starts,
            pair_start=config.pair_start,
            pair_count=config.pair_count,
            batch_size=config.scorer_batch_size,
            target_cells=target_cells,
            target_poses=target_poses,
            segnet_oracle=segnet_oracle,
            posenet_oracle=posenet_oracle,
        )
        current_errors, current_pose_squared_error = _score_totals(current_batches)
    occupied = set().union(*(candidate.conflict_keys() for candidate in accepted)) if accepted else set()
    sites = config.pair_count * 384 * 512
    pose_coordinates = config.pair_count * 6
    baseline_dpose = baseline_pose_squared_error / pose_coordinates
    for index in range(processed, len(candidates)):
        candidate = candidates[index]
        row: dict[str, Any] = {
            "schema": "ddm_v10_fisher_event_search_checkpoint.v1",
            "stage": "measured_candidate_admission",
            "typed_config_sha256": config.typed_config_hash(),
            "candidate_inventory_sha256": inventory_sha256,
            "candidate_index": index,
            "candidate_fingerprint": candidate.fingerprint(),
            "candidate": candidate.row(),
        }
        conflicts = sorted(candidate.conflict_keys() & occupied)
        if conflicts:
            admitted = False
            reason = "address_conflict_with_earlier_measured_admission"
        else:
            proposed = [*accepted, candidate]
            try:
                proposed_archive, proposed_receiver = _compile_candidates(predictor_archive, proposed)
            except DirectDescriptionError as exc:
                row["admitted"] = False
                row["reason"] = "strict_receiver_rejected_candidate"
                row["strict_receiver_error"] = str(exc)
                row["accepted_candidate_count_after"] = len(accepted)
                row["exact_added_bytes_after"] = len(current_archive) - len(base_archive)
                _atomic_checkpoint(root / "stage_checkpoints" / "candidates" / f"{index:04d}.json", row)
                candidate_rows.append(row)
                continue
            added_bytes = len(proposed_archive) - len(base_archive)
            if added_bytes > config.added_budget_bytes[-1]:
                row["admitted"] = False
                row["reason"] = "exact_added_byte_budget_exceeded"
                row["accepted_candidate_count_after"] = len(accepted)
                row["exact_added_bytes_after"] = len(current_archive) - len(base_archive)
                _atomic_checkpoint(root / "stage_checkpoints" / "candidates" / f"{index:04d}.json", row)
                candidate_rows.append(row)
                continue
            impacted_starts = sorted(
                {
                    ((source_pair_id - config.pair_start) // config.scorer_batch_size) * config.scorer_batch_size
                    for source_pair_id in candidate.source_pair_ids
                }
            )
            proposed_batches = _score_batches(
                proposed_receiver,
                batch_starts=impacted_starts,
                pair_start=config.pair_start,
                pair_count=config.pair_count,
                batch_size=config.scorer_batch_size,
                target_cells=target_cells,
                target_poses=target_poses,
                segnet_oracle=segnet_oracle,
                posenet_oracle=posenet_oracle,
            )
            proposed_errors = current_errors
            proposed_pose_squared_error = current_pose_squared_error
            for start, batch in proposed_batches.items():
                proposed_errors += batch.errors - current_batches[start].errors
                proposed_pose_squared_error += batch.pose_squared_error - current_batches[start].pose_squared_error
            current_dpose = current_pose_squared_error / pose_coordinates
            proposed_dpose = proposed_pose_squared_error / pose_coordinates
            distortion_gain = (
                100.0 * (current_errors - proposed_errors) / sites
                + math.sqrt(10.0 * current_dpose)
                - math.sqrt(10.0 * proposed_dpose)
            )
            marginal_bytes = len(proposed_archive) - len(current_archive)
            rate_cost = 25.0 * marginal_bytes / SOURCE_BYTES
            admitted = (
                proposed_errors < current_errors
                and proposed_dpose <= baseline_dpose + config.pose_dpose_increase_limit
                and distortion_gain > rate_cost
                and marginal_bytes > 0
            )
            reason = "measured_distortion_gain_exceeds_exact_rate_cost_pose_contained" if admitted else (
                "measured_candidate_failed_gain_rate_or_pose_containment"
            )
            row["measurement"] = {
                "errors_before": current_errors,
                "errors_after": proposed_errors,
                "delta_errors": current_errors - proposed_errors,
                "d_pose_before": f"{current_dpose:.12f}",
                "d_pose_after": f"{proposed_dpose:.12f}",
                "distortion_gain_score_units": f"{distortion_gain:.12e}",
                "marginal_archive_bytes": marginal_bytes,
                "rate_cost_score_units": f"{rate_cost:.12e}",
                "gain_per_byte": f"{distortion_gain / max(marginal_bytes, 1):.12e}",
                "canonical_batch_starts_replayed": impacted_starts,
            }
            if admitted:
                accepted.append(candidate)
                occupied.update(candidate.conflict_keys())
                current_archive = proposed_archive
                current_receiver = proposed_receiver
                current_errors = proposed_errors
                current_pose_squared_error = proposed_pose_squared_error
                current_batches.update(proposed_batches)
        row["admitted"] = admitted
        row["reason"] = reason
        row["accepted_candidate_count_after"] = len(accepted)
        row["exact_added_bytes_after"] = len(current_archive) - len(base_archive)
        _atomic_checkpoint(root / "stage_checkpoints" / "candidates" / f"{index:04d}.json", row)
        candidate_rows.append(row)

    search_seconds = time.perf_counter() - search_started
    _atomic_checkpoint(
        root / "stage_checkpoints" / "02_candidate_search_complete.json",
        {
            "schema": "ddm_v10_fisher_event_search_checkpoint.v1",
            "stage": "candidate_search_complete",
            "typed_config_sha256": config.typed_config_hash(),
            "candidate_inventory_sha256": inventory_sha256,
            "evaluated": len(candidates),
            "admitted": len(accepted),
            "exact_added_bytes": len(current_archive) - len(base_archive),
        },
    )

    ladder: list[dict[str, Any]] = []
    measured_bridges: dict[str, tuple[int, dict[str, Any]]] = {}
    measurement_started = time.perf_counter()
    for requested_budget in config.added_budget_bytes:
        selected: list[_SearchCandidate] = []
        selected_archive = base_archive
        selected_receiver = base_receiver
        for candidate in accepted:
            trial_archive, trial_receiver = _compile_candidates(predictor_archive, [*selected, candidate])
            if len(trial_archive) - len(base_archive) <= requested_budget:
                selected.append(candidate)
                selected_archive, selected_receiver = trial_archive, trial_receiver
        archive_path = root / (
            f"ddm_v10_fisher_event_n{config.pair_count}_add{requested_budget}.not_a_candidate.zip.receipt-bytes"
        )
        _publish_identical_or_new(archive_path, selected_archive)
        archive_sha256 = _sha256(selected_archive)
        reused_from_budget: int | None = None
        if archive_sha256 in measured_bridges:
            reused_from_budget, bridge = measured_bridges[archive_sha256]
        else:
            bridge = _measure_evaluator_bridge(
                selected_receiver,
                pair_start=config.pair_start,
                cached_lstars=cached_lstars,
                cached_margins=cached_margins,
                cached_poses=cached_poses,
                segnet_oracle=segnet_oracle,
                posenet_oracle=posenet_oracle,
                batch_size=config.scorer_batch_size,
            )
            measured_bridges[archive_sha256] = (requested_budget, bridge)
        lane, boundary, events = _candidate_symbols(selected)
        rung = {
            "requested_added_budget_bytes": requested_budget,
            "realized_added_bytes": len(selected_archive) - len(base_archive),
            "unspent_budget_bytes": requested_budget - (len(selected_archive) - len(base_archive)),
            "archive": {
                "path": _portable_path(archive_path),
                "bytes": len(selected_archive),
                "sha256": archive_sha256,
                "parse_reencode_identical": True,
                "receiver_closed": True,
            },
            "selected_candidate_count": len(selected),
            "measurement_reused_from_identical_archive_budget": reused_from_budget,
            "mechanism_counts": {
                "lane_g2cs1_symbols": len(lane),
                "road_boundary_coefficients": len(boundary),
                "topology_events": len(events),
                "xi_transported_events": sum(row.lifetime > 1 for row in events),
            },
            "bridge": bridge,
            "per_stratum": _stratified_byte_rows(selected_archive, bridge),
            "objective_advisory": {
                "score": _objective(
                    len(selected_archive), bridge["segmentation"]["d_seg"], bridge["pose"]["d_pose"]
                ),
                "formula": "100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489",
            },
        }
        ladder.append(rung)
        _atomic_checkpoint(
            root / "stage_checkpoints" / "budgets" / f"add_{requested_budget:06d}.json",
            {
                "schema": "ddm_v10_fisher_event_search_checkpoint.v1",
                "stage": "exact_budget_ladder_rung",
                "typed_config_sha256": config.typed_config_hash(),
                "requested_added_budget_bytes": requested_budget,
                "realized_added_bytes": rung["realized_added_bytes"],
                "archive": rung["archive"],
                "d_seg": bridge["segmentation"]["d_seg"],
                "d_pose": bridge["pose"]["d_pose"],
            },
        )
    measurement_seconds = time.perf_counter() - measurement_started

    final = ladder[-1]
    final_dseg = float(final["bridge"]["segmentation"]["d_seg"])
    max_total_bytes = int(final["archive"]["bytes"])
    near_200kb_measured = max_total_bytes >= 180_000
    plateau_falsifier = near_200kb_measured and final_dseg > 0.00116
    vocabulary_exhausted = final["unspent_budget_bytes"] > 0
    if final_dseg <= 0.00116 and max_total_bytes <= 154_600:
        verdict = "ADVISORY_INSTANCE_MEETS_SUB015_BOX_NOT_PROMOTABLE"
    elif plateau_falsifier:
        verdict = "ADVISORY_INSTANCE_FALSIFIED_BY_PLATEAU_NEAR_200KB_FAMILY_OPEN"
    elif vocabulary_exhausted:
        verdict = "ADVISORY_INSTANCE_VOCABULARY_EXPRESSIVENESS_BOUND_BEFORE_REQUESTED_BUDGET"
    else:
        verdict = "ADVISORY_INSTANCE_FAILS_SUB015_BOX_FORMULATION_OPEN"
    dseg_drops = [
        float(ladder[index - 1]["bridge"]["segmentation"]["d_seg"])
        - float(ladder[index]["bridge"]["segmentation"]["d_seg"])
        for index in range(1, len(ladder))
    ]
    byte_steps = [
        ladder[index]["archive"]["bytes"] - ladder[index - 1]["archive"]["bytes"]
        for index in range(1, len(ladder))
    ]
    marginal = [drop / step if step > 0 else 0.0 for drop, step in zip(dseg_drops, byte_steps, strict=True)]
    knee_index = int(np.argmax(marginal)) + 1 if marginal and max(marginal) > 0 else 0
    receipt: dict[str, Any] = {
        "schema": RESULT_SCHEMA_V2,
        "lane_id": "lane_ddm_v10_fisher_g2cs1_event_solve_20260722",
        "tasks": [603, 613, 578],
        "run_id": config.run_id,
        "seed": config.seed,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "semantic_argv": semantic_argv,
        "base_archive": {"bytes": len(base_archive), "sha256": _sha256(base_archive)},
        "candidate_search": {
            "honesty": "Fisher geometry proposes; exact receiver replay in full canonical scorer batches admits",
            "not_closed_form": True,
            "inventory_count": len(candidates),
            "inventory_sha256": inventory_sha256,
            "evaluated_count": len(candidate_rows),
            "admitted_count": len(accepted),
            "admission_rows": candidate_rows,
            "waterfill_stop_law": "admit iff measured distortion gain > 25*marginal_bytes/37545489 and pose is contained",
            "pixel_stream_present": False,
            "scorer_or_gt_present_in_archive": False,
        },
        "ladder": ladder,
        "knee": {
            "rung_index": knee_index,
            "requested_added_budget_bytes": ladder[knee_index]["requested_added_budget_bytes"],
            "mechanism": "highest measured d_seg drop per exact incremental archive byte",
            "marginal_dseg_per_byte": f"{(marginal[knee_index - 1] if knee_index else 0.0):.12e}",
        },
        "falsifier": {
            "condition": "d_seg remains above 0.00116 after a measured plateau near total 200KB",
            "near_200kb_measured": near_200kb_measured,
            "triggered": plateau_falsifier,
            "final_total_bytes": max_total_bytes,
            "final_d_seg": f"{final_dseg:.12f}",
            "instance_vocabulary_scope": (
                "G2CS1 Lane centerline c3, cubic Road boundary displacement, and parametric Lane/Movable "
                "birth/death bboxes on the bound v6 fixed_ar1_hold24 realization only"
            ),
            "family_closed": False,
        },
        "receiver_custody": dict(current_receiver.custody),
        "fail_closed_mutation_proof": fail_closed,
        "scorer_custody": {"segnet": segnet_custody, "posenet": posenet_custody},
        "target_custody": {
            "receipt_path": v5_typed["target_receipt_path"],
            "receipt_sha256": v5_typed["target_receipt_sha256"],
            "cache_path": str(cache_path),
            "cache_bytes": target_receipt.source_cache.bytes,
            "cache_sha256": target_receipt.source_cache.sha256,
        },
        "wallclock": {
            "candidate_search_seconds": f"{search_seconds:.6f}",
            "ladder_measurement_seconds": f"{measurement_seconds:.6f}",
            "total_seconds": f"{search_seconds + measurement_seconds:.6f}",
            "n600_status": "NOT_RUN_NO_BOUND_N600_V6_PREDICTOR_ARCHIVE",
        },
        "storage_preflight": storage,
        "resume": {
            "policy": config.checkpoint_policy,
            "candidate_checkpoints": len(candidate_rows),
            "budget_checkpoints": len(ladder),
            "all_preserved": True,
        },
        "verdict": verdict,
        "verdict_scope": (
            "This exact candidate inventory and sequential measured greedy ordering on the stated window. "
            "A negative closes only this INSTANCE vocabulary, not transported-event or structured-carrier families."
        ),
        "blocker_delta": (
            "#603 receiver grammar now executes jointly searched Lane G2CS1, Road boundary coefficients, and "
            "Pose6-transported topology events. Remaining blocker is measured vocabulary expressiveness and/or "
            "realization containment at the exact ladder, not missing receiver wiring."
        ),
        "stores_consulted": [
            config.v6_receipt_path,
            config.predictor_archive_path,
            v6_typed["v5_receipt_path"],
            v5_typed["target_receipt_path"],
            str(cache_path),
            "reports/latest.md",
            ".omx/state/lane_registry.json",
            ".omx/state/canonical_task_status.jsonl",
        ],
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "d_seg_claim": False,
        "d_pose_claim": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
    }
    _publish_new_bytes(receipt_path, rfc8785_canonicalize(receipt))
    print(json.dumps({"resumed": False, "receipt": str(receipt_path), "verdict": verdict}))
    return receipt_path


def _candidate_atom_count(candidate: _SearchCandidate) -> int:
    return len(candidate.lane_symbols) + len(candidate.boundary_shearlets) + len(candidate.island_shapes)


def run_v11_search(
    config: DirectDescriptionV11ObligationSearchConfigV1,
    output_directory: Path,
    semantic_argv: list[str],
) -> Path:
    """Run the bounded scorer-obligation solve with exact joint-objective admission."""

    root = output_directory
    storage = _storage_preflight(root.resolve())
    storage["output_tier"] = _portable_path(Path(str(storage.get("output_tier", root))))
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / f"ddm_v11_obligation_search_n{config.pair_count}_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(_read_regular_file_once(receipt_path))
        if receipt.get("typed_config_sha256") != config.typed_config_hash():
            raise DirectDescriptionError("completed v11 receipt typed-config hash differs")
        for rung in receipt["ladder"]:
            _bound_bytes(Path(rung["archive"]["path"]), rung["archive"]["sha256"], "completed v11 rung")
        print(json.dumps({"resumed": True, "receipt": str(receipt_path), "verdict": receipt["verdict"]}))
        return receipt_path

    v6_receipt = _bound_json(Path(config.v6_receipt_path), config.v6_receipt_sha256, "v6 receipt")
    if v6_receipt.get("schema") != "direct_description_dseg_bridge_amortize.v1":
        raise DirectDescriptionError("v11 input is not the governed v6 receipt")
    v6_typed = v6_receipt.get("typed_config", {})
    if (v6_typed.get("pair_start"), v6_typed.get("pair_count")) != (config.pair_start, config.pair_count):
        raise DirectDescriptionError("v11 window differs from bound v6 window")
    _candidate_row(v6_receipt, config)
    predictor_archive = _bound_bytes(
        Path(config.predictor_archive_path), config.predictor_archive_sha256, "v6 predictor archive"
    )
    base_archive, base_receiver = _compile_obligation_candidates(predictor_archive, [])
    fail_closed = prove_carrier_archive_fail_closed(base_archive)

    v5_receipt = _bound_json(Path(v6_typed["v5_receipt_path"]), v6_typed["v5_receipt_sha256"], "v6-bound v5 receipt")
    v5_typed = v5_receipt.get("typed_config", {})
    target_receipt = load_target_receipt(Path(v5_typed["target_receipt_path"]), v5_typed["target_receipt_sha256"])
    cache_path = Path(target_receipt.source_cache.path)
    if not cache_path.is_file() or cache_path.stat().st_size != target_receipt.source_cache.bytes:
        raise DirectDescriptionError("frozen scorer cache is unavailable")
    cached_lstars = open_stored_npy_memmap(cache_path, "lstars")
    cached_margins = open_stored_npy_memmap(cache_path, "margins")
    cached_poses = open_stored_npy_memmap(cache_path, "gt_poses")
    source_slice = slice(config.pair_start, config.pair_start + config.pair_count)
    target_cells = cached_lstars[source_slice]
    target_margins = cached_margins[source_slice]
    target_poses = cached_poses[source_slice]
    segnet_oracle, segnet_custody = _load_segnet_oracle(Path(config.upstream_root), threads=config.scorer_threads)
    posenet_oracle, posenet_custody = _load_posenet_oracle(Path(config.upstream_root), threads=config.scorer_threads)
    all_batch_starts = list(range(0, config.pair_count, config.scorer_batch_size))
    search_started = time.perf_counter()
    baseline_batches = _score_batches(
        base_receiver,
        batch_starts=all_batch_starts,
        pair_start=config.pair_start,
        pair_count=config.pair_count,
        batch_size=config.scorer_batch_size,
        target_cells=target_cells,
        target_poses=target_poses,
        segnet_oracle=segnet_oracle,
        posenet_oracle=posenet_oracle,
    )
    baseline_errors, baseline_pose_squared_error = _score_totals(baseline_batches)
    described_cells = np.concatenate([baseline_batches[start].cells for start in all_batch_starts], axis=0)
    generated, candidates, raw_generated_count = _derive_obligation_candidates(
        base_receiver,
        target_cells=target_cells,
        target_margins=target_margins,
        described_cells=described_cells,
        config=config,
    )
    generated_family_counts = dict.fromkeys(_OBLIGATION_FAMILY_ORDER, 0)
    for candidate in generated:
        generated_family_counts[_obligation_family(candidate)] += 1
    inventory_payload = {
        "generated_atomic_candidates": [candidate.row() for candidate in generated],
        "measured_bundles": [candidate.row() for candidate in candidates],
    }
    inventory_sha256 = _sha256(rfc8785_canonicalize(inventory_payload))
    _atomic_checkpoint(
        root / "stage_checkpoints" / "01_obligation_inventory.json",
        {
            "schema": "ddm_v11_obligation_search_checkpoint.v1",
            "stage": "obligation_inventory",
            "typed_config_sha256": config.typed_config_hash(),
            "raw_generated_count": raw_generated_count,
            "bounded_generated_count": len(generated),
            "measured_bundle_count": len(candidates),
            "generated_family_counts": generated_family_counts,
            "atoms_in_measured_bundles": sum(_candidate_atom_count(row) for row in candidates),
            "candidate_inventory_sha256": inventory_sha256,
        },
    )

    accepted: list[_SearchCandidate] = []
    processed = 0
    candidate_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        checkpoint_path = root / "stage_checkpoints" / "candidates" / f"{index:04d}.json"
        if not checkpoint_path.exists():
            break
        row = json.loads(_read_regular_file_once(checkpoint_path))
        if (
            row.get("typed_config_sha256") != config.typed_config_hash()
            or row.get("candidate_inventory_sha256") != inventory_sha256
            or row.get("candidate_fingerprint") != candidate.fingerprint()
        ):
            raise DirectDescriptionError("v11 candidate checkpoint differs from deterministic inventory")
        if row.get("admitted") and "bridge_after" not in row:
            raise DirectDescriptionError("admitted v11 checkpoint lacks its exact bridge snapshot")
        candidate_rows.append(row)
        if row["admitted"]:
            accepted.append(candidate)
        processed += 1

    current_archive, current_receiver = _compile_obligation_candidates(predictor_archive, accepted)
    if accepted:
        current_batches = _score_batches(
            current_receiver,
            batch_starts=all_batch_starts,
            pair_start=config.pair_start,
            pair_count=config.pair_count,
            batch_size=config.scorer_batch_size,
            target_cells=target_cells,
            target_poses=target_poses,
            segnet_oracle=segnet_oracle,
            posenet_oracle=posenet_oracle,
        )
        current_errors, current_pose_squared_error = _score_totals(current_batches)
    else:
        current_batches = dict(baseline_batches)
        current_errors, current_pose_squared_error = baseline_errors, baseline_pose_squared_error
    occupied = set().union(*(candidate.conflict_keys() for candidate in accepted)) if accepted else set()
    sites = config.pair_count * 384 * 512
    pose_coordinates = config.pair_count * 6
    baseline_dpose = baseline_pose_squared_error / pose_coordinates
    pose_tube_ceiling = baseline_dpose + config.pose_tube_dpose_radius
    for index in range(processed, len(candidates)):
        candidate = candidates[index]
        row: dict[str, Any] = {
            "schema": "ddm_v11_obligation_search_checkpoint.v1",
            "stage": "measured_joint_objective_admission",
            "typed_config_sha256": config.typed_config_hash(),
            "candidate_inventory_sha256": inventory_sha256,
            "candidate_index": index,
            "candidate_fingerprint": candidate.fingerprint(),
            "candidate": candidate.row(),
            "atomic_obligation_count": _candidate_atom_count(candidate),
        }
        conflicts = sorted(candidate.conflict_keys() & occupied)
        admitted = False
        if conflicts:
            reason = "address_conflict_with_earlier_measured_admission"
        else:
            try:
                proposed_archive, proposed_receiver = _compile_obligation_candidates(
                    predictor_archive, [*accepted, candidate]
                )
            except DirectDescriptionError as exc:
                row["strict_receiver_error"] = str(exc)
                reason = "strict_receiver_rejected_candidate_bundle"
            else:
                marginal_bytes = len(proposed_archive) - len(current_archive)
                if len(proposed_archive) > config.total_archive_ceiling_bytes:
                    reason = "preregistered_total_archive_ceiling_exceeded"
                elif len(proposed_archive) - len(base_archive) > config.added_budget_bytes[-1]:
                    reason = "preregistered_added_byte_ceiling_exceeded"
                else:
                    impacted_starts = sorted(
                        {
                            ((source_pair_id - config.pair_start) // config.scorer_batch_size)
                            * config.scorer_batch_size
                            for source_pair_id in candidate.source_pair_ids
                        }
                    )
                    proposed_batches = _score_batches(
                        proposed_receiver,
                        batch_starts=impacted_starts,
                        pair_start=config.pair_start,
                        pair_count=config.pair_count,
                        batch_size=config.scorer_batch_size,
                        target_cells=target_cells,
                        target_poses=target_poses,
                        segnet_oracle=segnet_oracle,
                        posenet_oracle=posenet_oracle,
                    )
                    proposed_errors = current_errors
                    proposed_pose_squared_error = current_pose_squared_error
                    for start, batch in proposed_batches.items():
                        proposed_errors += batch.errors - current_batches[start].errors
                        proposed_pose_squared_error += batch.pose_squared_error - current_batches[start].pose_squared_error
                    current_dpose = current_pose_squared_error / pose_coordinates
                    proposed_dpose = proposed_pose_squared_error / pose_coordinates
                    seg_delta, pose_delta, rate_delta, joint_delta = _joint_objective_delta(
                        current_errors=current_errors,
                        proposed_errors=proposed_errors,
                        sites=sites,
                        current_dpose=current_dpose,
                        proposed_dpose=proposed_dpose,
                        marginal_bytes=marginal_bytes,
                    )
                    admitted = marginal_bytes > 0 and joint_delta < 0.0 and proposed_dpose <= pose_tube_ceiling
                    reason = (
                        "measured_joint_objective_delta_negative_inside_pose_safety_tube"
                        if admitted
                        else "measured_joint_objective_nonnegative_or_pose_safety_tube_exceeded"
                    )
                    row["measurement"] = {
                        "errors_before": current_errors,
                        "errors_after": proposed_errors,
                        "delta_errors": current_errors - proposed_errors,
                        "d_pose_before": f"{current_dpose:.12f}",
                        "d_pose_after": f"{proposed_dpose:.12f}",
                        "pose_tube_ceiling_d_pose": f"{pose_tube_ceiling:.12f}",
                        "seg_objective_delta": f"{seg_delta:.12e}",
                        "pose_objective_delta": f"{pose_delta:.12e}",
                        "rate_objective_delta": f"{rate_delta:.12e}",
                        "joint_objective_delta": f"{joint_delta:.12e}",
                        "marginal_archive_bytes": marginal_bytes,
                        "canonical_batch_starts_replayed": impacted_starts,
                    }
                    if admitted:
                        accepted.append(candidate)
                        occupied.update(candidate.conflict_keys())
                        current_archive = proposed_archive
                        current_receiver = proposed_receiver
                        current_errors = proposed_errors
                        current_pose_squared_error = proposed_pose_squared_error
                        current_batches.update(proposed_batches)
                        row["bridge_after"] = _compact_bridge_from_batches(
                            current_batches,
                            target_cells=target_cells,
                            target_margins=target_margins,
                            target_poses=target_poses,
                        )
        row["admitted"] = admitted
        row["reason"] = reason
        row["accepted_candidate_count_after"] = len(accepted)
        row["exact_added_bytes_after"] = len(current_archive) - len(base_archive)
        _atomic_checkpoint(root / "stage_checkpoints" / "candidates" / f"{index:04d}.json", row)
        candidate_rows.append(row)

    baseline_bridge = _compact_bridge_from_batches(
        baseline_batches,
        target_cells=target_cells,
        target_margins=target_margins,
        target_poses=target_poses,
    )
    states = [
        {
            "accepted_count": 0,
            "exact_added_bytes": 0,
            "archive_bytes": len(base_archive),
            "bridge": baseline_bridge,
        }
    ]
    for row in candidate_rows:
        if row["admitted"]:
            states.append(
                {
                    "accepted_count": row["accepted_candidate_count_after"],
                    "exact_added_bytes": row["exact_added_bytes_after"],
                    "archive_bytes": len(base_archive) + row["exact_added_bytes_after"],
                    "bridge": row["bridge_after"],
                }
            )
    _atomic_checkpoint(
        root / "stage_checkpoints" / "02_candidate_search_complete.json",
        {
            "schema": "ddm_v11_obligation_search_checkpoint.v1",
            "stage": "candidate_search_complete",
            "typed_config_sha256": config.typed_config_hash(),
            "candidate_inventory_sha256": inventory_sha256,
            "generated_atomic_candidates": len(generated),
            "evaluated_bundles": len(candidates),
            "admitted_bundles": len(accepted),
            "admitted_atoms": sum(_candidate_atom_count(row) for row in accepted),
            "exact_added_bytes": len(current_archive) - len(base_archive),
        },
    )

    ladder: list[dict[str, Any]] = []
    maximum_allowed_added = max(0, config.total_archive_ceiling_bytes - len(base_archive))
    for requested_budget in config.added_budget_bytes:
        effective_budget = min(requested_budget, maximum_allowed_added)
        state = max(
            (row for row in states if row["exact_added_bytes"] <= effective_budget),
            key=lambda row: (row["exact_added_bytes"], row["accepted_count"]),
        )
        selected = accepted[: int(state["accepted_count"])]
        selected_archive, _selected_receiver = _compile_obligation_candidates(predictor_archive, selected)
        if len(selected_archive) != state["archive_bytes"]:
            raise DirectDescriptionError("v11 ladder state differs from its accepted-prefix archive bytes")
        archive_path = root / (
            f"ddm_v11_obligation_n{config.pair_count}_add{requested_budget}.not_a_candidate.zip.receipt-bytes"
        )
        _publish_identical_or_new(archive_path, selected_archive)
        lane, shearlets, islands = _obligation_symbols(selected)
        bridge = state["bridge"]
        rung = {
            "requested_added_budget_bytes": requested_budget,
            "effective_added_budget_bytes": effective_budget,
            "realized_added_bytes": state["exact_added_bytes"],
            "unspent_effective_budget_bytes": effective_budget - state["exact_added_bytes"],
            "archive": {
                "path": _portable_path(archive_path),
                "bytes": len(selected_archive),
                "sha256": _sha256(selected_archive),
                "parse_reencode_identical": True,
                "receiver_closed": True,
            },
            "selected_bundle_count": state["accepted_count"],
            "selected_atom_count": len(lane) + len(shearlets) + len(islands),
            "mechanism_counts": {
                "lane_full_coefficient_symbols": len(lane),
                "boundary_shearlet_atoms": len(shearlets),
                "movable_shape_atoms": len(islands),
                "xi_transported_islands": sum(row.lifetime > 1 for row in islands),
            },
            "bridge": bridge,
            "per_stratum": _stratified_byte_rows(selected_archive, bridge),
            "objective_advisory": {
                "score": _objective(
                    len(selected_archive), bridge["segmentation"]["d_seg"], bridge["pose"]["d_pose"]
                ),
                "formula": "100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489",
            },
        }
        ladder.append(rung)
        _atomic_checkpoint(
            root / "stage_checkpoints" / "budgets" / f"add_{requested_budget:06d}.json",
            {
                "schema": "ddm_v11_obligation_search_checkpoint.v1",
                "stage": "exact_budget_ladder_rung",
                "typed_config_sha256": config.typed_config_hash(),
                "requested_added_budget_bytes": requested_budget,
                "effective_added_budget_bytes": effective_budget,
                "realized_added_bytes": state["exact_added_bytes"],
                "archive": rung["archive"],
                "d_seg": bridge["segmentation"]["d_seg"],
                "d_pose": bridge["pose"]["d_pose"],
                "per_stratum": bridge["segmentation"]["strata"],
            },
        )

    objective_values = [float(row["objective_advisory"]["score"]) for row in ladder]
    byte_steps = [ladder[i]["archive"]["bytes"] - ladder[i - 1]["archive"]["bytes"] for i in range(1, len(ladder))]
    objective_drops = [objective_values[i - 1] - objective_values[i] for i in range(1, len(ladder))]
    marginal = [drop / step if step > 0 else 0.0 for drop, step in zip(objective_drops, byte_steps, strict=True)]
    knee_index = int(np.argmax(marginal)) + 1 if marginal and max(marginal) > 0 else 0
    final = ladder[-1]
    final_dseg = float(final["bridge"]["segmentation"]["d_seg"])
    final_bytes = int(final["archive"]["bytes"])
    atoms_in_measured_bundles = sum(_candidate_atom_count(row) for row in candidates)
    unmeasured_bounded_atomic_count = max(0, len(generated) - atoms_in_measured_bundles)
    measurement_inventory_exhausted = unmeasured_bounded_atomic_count == 0
    ceiling_probe_total_bytes = max(
        len(base_archive) + int(row["effective_added_budget_bytes"]) for row in ladder
    )
    near_200kb_budget_ceiling_probed = ceiling_probe_total_bytes >= (
        config.total_archive_ceiling_bytes - 1024
    )
    last_admission = max((index for index, row in enumerate(candidate_rows) if row["admitted"]), default=-1)
    flattened = last_admission < len(candidate_rows) - 4
    plateau_falsifier = (
        config.pair_count == 600
        and final_dseg > 0.00116
        and near_200kb_budget_ceiling_probed
        and flattened
        and measurement_inventory_exhausted
    )
    if final_dseg <= 0.00116 and final_bytes <= 154_600:
        verdict = "ADVISORY_INSTANCE_MEETS_SUB015_BOX_NOT_PROMOTABLE"
    elif plateau_falsifier:
        verdict = "ADVISORY_FORMULATION_PLATEAU_NEAR_200KB_V6_SUCCESSOR_NAMED"
    elif len(generated) > sum(_candidate_atom_count(row) for row in candidates):
        verdict = "ADVISORY_BOUNDED_WATERFILL_ABOVE_TARGET_UNMEASURED_OBLIGATIONS_REMAIN"
    else:
        verdict = "ADVISORY_FORMULATION_ABOVE_TARGET_AFTER_OBLIGATION_INVENTORY"
    search_seconds = time.perf_counter() - search_started
    receipt: dict[str, Any] = {
        "schema": RESULT_SCHEMA_V3,
        "lane_id": "lane_ddm_v11_obligation_vocabulary_solve_20260722",
        "tasks": [603, 613, 578],
        "run_id": config.run_id,
        "seed": config.seed,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "semantic_argv": semantic_argv,
        "base_archive": {"bytes": len(base_archive), "sha256": _sha256(base_archive)},
        "candidate_search": {
            "honesty": "atomic scorer obligations are bundled by canonical scorer batch; exact receiver replay admits bundles",
            "not_closed_form": True,
            "raw_generated_atomic_count": raw_generated_count,
            "bounded_generated_atomic_count": len(generated),
            "generated_family_counts": generated_family_counts,
            "measured_bundle_count": len(candidates),
            "atoms_in_measured_bundles": atoms_in_measured_bundles,
            "unmeasured_bounded_atomic_count": unmeasured_bounded_atomic_count,
            "measurement_inventory_exhausted": measurement_inventory_exhausted,
            "evaluated_bundle_count": len(candidate_rows),
            "admitted_bundle_count": len(accepted),
            "admitted_atom_count": sum(_candidate_atom_count(row) for row in accepted),
            "inventory_sha256": inventory_sha256,
            "admission_rows": candidate_rows,
            "waterfill_stop_law": "admit iff measured delta of 100*d_seg+sqrt(10*d_pose)+25*bytes/37545489 is negative; pose tube is safety ceiling only",
            "pose_tube": {
                "baseline_d_pose": f"{baseline_dpose:.12f}",
                "radius": f"{config.pose_tube_dpose_radius:.12f}",
                "ceiling": f"{pose_tube_ceiling:.12f}",
                "role": "safety rail only; no zero-increase veto",
            },
            "pixel_stream_present": False,
            "scorer_or_gt_present_in_archive": False,
        },
        "ladder": ladder,
        "knee": {
            "rung_index": knee_index,
            "requested_added_budget_bytes": ladder[knee_index]["requested_added_budget_bytes"],
            "mechanism": "largest measured joint-objective drop per exact incremental archive byte",
            "marginal_objective_drop_per_byte": f"{(marginal[knee_index - 1] if knee_index else 0.0):.12e}",
        },
        "falsifier": {
            "condition": "n600 joint-objective obligation curve flattens above d_seg 0.00116 after the measured obligation inventory is exhausted with the near-200KB budget ceiling probed",
            "near_200kb_budget_ceiling_probed": near_200kb_budget_ceiling_probed,
            "ceiling_probe_total_bytes": ceiling_probe_total_bytes,
            "measurement_inventory_exhausted": measurement_inventory_exhausted,
            "unmeasured_bounded_atomic_count": unmeasured_bounded_atomic_count,
            "flattened": flattened,
            "triggered": plateau_falsifier,
            "not_triggered_reason": (
                None
                if plateau_falsifier
                else (
                    "bounded measurement cap left scorer-obligation atoms unmeasured; no formulation-level wrong-worldsheet inference"
                    if not measurement_inventory_exhausted
                    else "other preregistered falsifier conditions were not all met"
                )
            ),
            "final_total_bytes": final_bytes,
            "final_d_seg": f"{final_dseg:.12f}",
            "binding_decomposition": final["bridge"]["segmentation"]["strata"],
            "verdict_scope": "declared v11 obligation grammar over the bound v6 fixed_ar1_hold24 realization",
            "family_closed": False,
            "next_vehicle_if_triggered": "v6-successor predictor with lower wrong-worldsheet baseline before post-solve correction",
        },
        "receiver_custody": dict(current_receiver.custody),
        "fail_closed_mutation_proof": fail_closed,
        "scorer_custody": {"segnet": segnet_custody, "posenet": posenet_custody},
        "target_custody": {
            "receipt_path": v5_typed["target_receipt_path"],
            "receipt_sha256": v5_typed["target_receipt_sha256"],
            "cache_path": str(cache_path),
            "cache_bytes": target_receipt.source_cache.bytes,
            "cache_sha256": target_receipt.source_cache.sha256,
        },
        "wallclock": {
            "total_seconds": f"{search_seconds:.6f}",
            "n600_status": "MEASURED_FULL_0_600_CHUNKED" if config.pair_count == 600 else "WINDOW_LADDER_RUNG",
            "oom_law": "batch16 RGB is released after each scorer call; one source chunk and one RGB scorer batch resident; retained argmax cells only",
            "bounded_rederive_under_600_seconds": search_seconds <= 600.0,
        },
        "storage_preflight": storage,
        "resume": {
            "policy": config.checkpoint_policy,
            "candidate_checkpoints": len(candidate_rows),
            "budget_checkpoints": len(ladder),
            "all_preserved": True,
        },
        "verdict": verdict,
        "verdict_scope": (
            "This exact scorer-obligation grammar, generated inventory cap, measured canonical-batch bundle ordering, "
            "and bound v6 fixed_ar1_hold24 realization. Any negative is formulation-scoped, never family-level."
        ),
        "blocker_delta": (
            "#603 now has full-window fixed_ar1_hold24 binding, full Lane coefficient/width/xi-phase obligations, "
            "Road/Undrivable shearlet boundary atoms, Movable moment/curvelet shapes, and joint contest-objective admission. "
            "Remaining debt is whatever the measured ladder and unmeasured-obligation count state; no hard zero-pose veto remains."
        ),
        "stores_consulted": [
            config.v6_receipt_path,
            config.predictor_archive_path,
            v6_typed["v5_receipt_path"],
            v5_typed["target_receipt_path"],
            str(cache_path),
            "docs/operating_manual_craft_handoff.md",
            "reports/latest.md",
            ".omx/state/lane_registry.json",
            ".omx/state/canonical_task_status.jsonl",
        ],
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "d_seg_claim": False,
        "d_pose_claim": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
    }
    _publish_new_bytes(receipt_path, rfc8785_canonicalize(receipt))
    print(json.dumps({"resumed": False, "receipt": str(receipt_path), "verdict": verdict}))
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    config_bytes = _read_regular_file_once(args.config)
    config_payload = json.loads(config_bytes)
    semantic_argv = [
        _portable_path(Path(__file__)),
        "--config",
        str(args.config),
        "--output-directory",
        str(args.output_directory),
    ]
    if config_payload.get("schema") == "DirectDescriptionV11ObligationSearchConfigV1":
        config = DirectDescriptionV11ObligationSearchConfigV1.model_validate_json(config_bytes)
        run_v11_search(config, args.output_directory, semantic_argv)
    elif config_payload.get("schema") == "DirectDescriptionV10FisherEventSearchConfigV1":
        config = DirectDescriptionV10FisherEventSearchConfigV1.model_validate_json(config_bytes)
        run_v10_search(config, args.output_directory, semantic_argv)
    else:
        config = DirectDescriptionV9CarrierComposeConfigV1.model_validate_json(config_bytes)
        run(config, args.output_directory, semantic_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
