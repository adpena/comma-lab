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
    BoundaryCoefficientDelta,
    DirectDescriptionV9CarrierComposeConfigV1,
    DirectDescriptionV10FisherEventSearchConfigV1,
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


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _candidate_row(v6_receipt: dict[str, Any], config: DirectDescriptionV9CarrierComposeConfigV1) -> dict[str, Any]:
    for row in v6_receipt.get("candidates", ()):  # fixed AR(1)/hold24 is candidate index 1 by sealed v6 order.
        archive = row.get("archive", {})
        if (
            row.get("candidate_index") == 1
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

    def row(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "mechanism": self.mechanism,
            "fisher_priority": f"{self.fisher_priority:.12e}",
            "source_pair_ids": list(self.source_pair_ids),
            "lane_symbols": [asdict(value) for value in self.lane_symbols],
            "boundary_symbols": [asdict(value) for value in self.boundary_symbols],
            "topology_events": [asdict(value) for value in self.topology_events],
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
    if config_payload.get("schema") == "DirectDescriptionV10FisherEventSearchConfigV1":
        config = DirectDescriptionV10FisherEventSearchConfigV1.model_validate_json(config_bytes)
        run_v10_search(config, args.output_directory, semantic_argv)
    else:
        config = DirectDescriptionV9CarrierComposeConfigV1.model_validate_json(config_bytes)
        run(config, args.output_directory, semantic_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
