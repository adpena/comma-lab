# SPDX-License-Identifier: MIT
"""Typed full-video score atlas over frozen-scorer checkpoint surfaces.

This module deliberately does not rank pixel energy.  Its only primary pair
ordering is additive contest-objective debt reconstructed from exact SegNet
argmax disagreements and frozen PoseNet outputs.  Margin, rank-four flip
distance, and the older evaluator-response cone are attached as geometry for
costate consumers; L2 fields remain diagnostic-only and are rejected as rank
keys.

Evidence produced here is ``[macOS-CPU frozen-scorer advisory]``.  The scorer
outputs are reused from SHA-bound v12 canonical-batch caches rather than
recomputed.  This is an index/aggregation pass, not a score claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import tempfile
import zlib
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import HEAD_PAIR_NORMS

SCHEMA = "ddm_g3_score_atlas.v1"
PAIR_SCHEMA = "ddm_g3_score_atlas_pair.v1"
HARD_PAIR_SCHEMA = "ddm_g3_hard_pair_registry.v1"
ADMISSION_SCHEMA = "ddm_g3_admission_efficiency.v1"
AXIS = "[macOS-CPU frozen-scorer advisory]"
POINTER = "0.1910828242 [contest-CPU]"
CLASS_ORDER = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
MARGIN_BANDS = ((0.0, 0.1, "[0,0.1)"), (0.1, 0.5, "[0.1,0.5)"), (0.5, 1.0, "[0.5,1)"), (1.0, math.inf, "[1,inf)"))
FLIP_DISTANCE_EDGES = (0.0, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, math.inf)
ROW_BINS = 12
SOURCE_BYTES = 37_545_489
RATE_PER_BYTE = 25.0 / SOURCE_BYTES
PRIMARY_RANK_KEY = "derived_exact_flip_pose_score_mass"


class ScoreAtlasError(ValueError):
    """Raised when custody, completeness, or score-currency checks fail."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)


class AdmissionReceiptSourceV1(_StrictModel):
    version: Literal["v10", "v11", "v12"]
    pair_count: StrictInt = Field(ge=1, le=600)
    path: StrictStr
    sha256: StrictStr

    @model_validator(mode="after")
    def _validate_sha(self) -> AdmissionReceiptSourceV1:
        if len(self.sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.sha256):
            raise ValueError("admission receipt sha256 must be lowercase SHA-256")
        return self


class DdmG3ScoreAtlasConfigV1(_StrictModel):
    schema_: Literal["DdmG3ScoreAtlasConfigV1"] = Field(
        default="DdmG3ScoreAtlasConfigV1", alias="schema", serialization_alias="schema"
    )
    run_id: StrictStr
    evaluator_atlas_path: StrictStr
    evaluator_atlas_sha256: StrictStr
    v12_receipt_path: StrictStr
    v12_receipt_sha256: StrictStr
    admission_receipts: list[AdmissionReceiptSourceV1]
    v13_pointer_ledger_path: StrictStr
    v13_pointer_ledger_sha256: StrictStr
    v13_pointer_p0_id: StrictStr
    output_directory: StrictStr
    compact_receipt_directory: StrictStr
    n_pairs: Literal[600] = 600
    cache_chunk_size: StrictInt = Field(default=16, ge=1, le=16)
    row_bins: Literal[12] = 12
    seed: Literal[1234] = 1234
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    d_seg_claim: Literal[False] = False
    d_pose_claim: Literal[False] = False

    @model_validator(mode="after")
    def _validate_contract(self) -> DdmG3ScoreAtlasConfigV1:
        for name in ("evaluator_atlas_sha256", "v12_receipt_sha256", "v13_pointer_ledger_sha256"):
            value = getattr(self, name)
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"{name} must be lowercase SHA-256")
        if [item.version for item in self.admission_receipts] != ["v10", "v11", "v12"]:
            raise ValueError("admission_receipts must contain exactly v10, v11, v12 in order")
        v12 = self.admission_receipts[-1]
        if v12.path != self.v12_receipt_path or v12.sha256 != self.v12_receipt_sha256:
            raise ValueError("v12 admission source must match the current-state v12 receipt")
        return self

    def typed_hash(self) -> str:
        payload = self.model_dump(mode="json", by_alias=True)
        return hashlib.sha256(_canonical_json(payload)).hexdigest()


class ScoreMassV1(_StrictModel):
    seg_score_mass: float = Field(ge=0.0)
    pose_score_mass: float = Field(ge=0.0)
    distortion_score_mass: float = Field(ge=0.0)
    rate_score_mass_diagnostic: float = Field(ge=0.0)
    rank_key: Literal["derived_exact_flip_pose_score_mass"] = PRIMARY_RANK_KEY
    epistemic_status: Literal["DERIVED"] = "DERIVED"

    @model_validator(mode="after")
    def _sum_matches(self) -> ScoreMassV1:
        if not math.isclose(
            self.distortion_score_mass,
            self.seg_score_mass + self.pose_score_mass,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("distortion_score_mass must equal seg_score_mass + pose_score_mass")
        if "l2" in self.rank_key.lower() or "energy" in self.rank_key.lower():
            raise ValueError("L2/energy rank keys are forbidden")
        return self


class DdmG3CostatePairSignalV1(_StrictModel):
    schema_: Literal["ddm_g3_costate_pair_signal.v1"] = Field(
        default="ddm_g3_costate_pair_signal.v1", alias="schema", serialization_alias="schema"
    )
    pair_index: StrictInt = Field(ge=0, le=599)
    lambda_proxy_score_debt: float = Field(ge=0.0)
    seg_flip_count: StrictInt = Field(ge=0)
    median_rank4_flip_distance: float | None = Field(default=None, ge=0.0)
    pose_squared_error_sum: float = Field(ge=0.0)
    pose_binds_fraction: float = Field(ge=0.0, le=1.0)
    allocated_bytes: float = Field(ge=0.0)
    ranking_currency: Literal["exact_flip_plus_pose_objective_mass"]
    energy_rank_forbidden: Literal[True] = True
    score_claim: Literal[False] = False


class DdmG3ScoreAtlasPairV1(_StrictModel):
    schema_: Literal["ddm_g3_score_atlas_pair.v1"] = Field(
        default=PAIR_SCHEMA, alias="schema", serialization_alias="schema"
    )
    pair_index: StrictInt = Field(ge=0, le=599)
    frame_indices: tuple[StrictInt, StrictInt]
    scored_seg_frame_index: StrictInt
    score_rank: StrictInt = Field(ge=1, le=600)
    score_mass: ScoreMassV1
    segmentation: dict[str, Any]
    pose: dict[str, Any]
    allocated_bytes: dict[str, Any]
    scene_covariates: dict[str, Any]
    evaluator_response_geometry: dict[str, Any]
    costate_signal: DdmG3CostatePairSignalV1
    source_custody: dict[str, Any]
    evidence_axis: Literal["[macOS-CPU frozen-scorer advisory]"] = AXIS
    score_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False


@dataclass(slots=True)
class BatchScore:
    start: int
    cells: np.ndarray
    poses: np.ndarray
    errors: int
    pose_squared_error: float
    path: str
    sha256: str


@dataclass(slots=True)
class ReconstructedState:
    baseline_batches: dict[int, BatchScore]
    final_batches: dict[int, BatchScore]
    admission_rows: list[dict[str, Any]]
    final_cells: np.ndarray
    final_poses: np.ndarray
    final_errors: int
    final_pose_squared_error: float
    archive_bytes: int
    archive_sha256: str
    archive_path: Path


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_file(path: Path, expected_sha256: str, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ScoreAtlasError(f"{label} missing or unsafe: {path}")
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected_sha256:
        raise ScoreAtlasError(f"{label} SHA mismatch: expected {expected_sha256}, got {actual}")
    return payload


def _resolve(repo: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else repo / path


def _read_bound_json(repo: Path, value: str | Path, sha256: str, label: str) -> dict[str, Any]:
    path = _resolve(repo, value)
    payload = _bound_file(path, sha256, label)
    try:
        result = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ScoreAtlasError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(result, dict):
        raise ScoreAtlasError(f"{label} must be a JSON object")
    return result


def _load_batch_cache(repo: Path, manifest: Mapping[str, Any]) -> BatchScore:
    path = _resolve(repo, str(manifest["path"]))
    payload = _bound_file(path, str(manifest["sha256"]), "frozen scorer batch cache")
    try:
        header_bytes, compressed = payload.split(b"\n", 1)
        header = json.loads(header_bytes)
        body = zlib.decompress(compressed)
    except (ValueError, json.JSONDecodeError, zlib.error) as exc:
        raise ScoreAtlasError(f"invalid frozen scorer batch cache: {path}") from exc
    start = int(manifest["start"])
    if header.get("schema") != "ddm_canonical_batch_score_cache.v1" or int(header.get("start", -1)) != start:
        raise ScoreAtlasError(f"batch cache schema/start mismatch: {path}")
    cells_bytes = int(header["cells_bytes"])
    poses_bytes = int(header["poses_bytes"])
    if len(body) != cells_bytes + poses_bytes:
        raise ScoreAtlasError(f"batch cache decompressed byte count mismatch: {path}")
    cells = np.frombuffer(body[:cells_bytes], dtype=np.dtype(header["cells_dtype"])).reshape(
        tuple(int(v) for v in header["cells_shape"])
    )
    poses = np.frombuffer(body[cells_bytes:], dtype=np.dtype(header["poses_dtype"])).reshape(
        tuple(int(v) for v in header["poses_shape"])
    )
    if cells.ndim != 3 or cells.shape[1:] != (384, 512) or poses.shape != (len(cells), 6):
        raise ScoreAtlasError(f"batch cache tensor shape mismatch: {path}")
    if cells.dtype != np.uint8 or int(cells.min()) < 0 or int(cells.max()) >= len(CLASS_ORDER):
        raise ScoreAtlasError(f"batch cache argmax cell range mismatch: {path}")
    errors = int(header["errors"])
    pose_squared_error = float(header["pose_squared_error"])
    if not np.isfinite(poses).all() or not math.isfinite(pose_squared_error):
        raise ScoreAtlasError(f"batch cache contains nonfinite pose values: {path}")
    return BatchScore(
        start=start,
        cells=np.array(cells, copy=True),
        poses=np.array(poses, copy=True),
        errors=errors,
        pose_squared_error=pose_squared_error,
        path=str(path),
        sha256=str(manifest["sha256"]),
    )


def load_evaluator_response_rows(
    path: Path,
    expected_sha256: str,
    *,
    n_pairs: int = 600,
) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    payload = _bound_file(path, expected_sha256, "evaluator response atlas")
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if len(rows) != n_pairs + 1:
        raise ScoreAtlasError(f"evaluator atlas must contain header + {n_pairs} rows; got {len(rows)}")
    header = rows[0]
    by_pair = {int(row["pair_index"]): row for row in rows[1:]}
    if header.get("schema") != "evaluator_response_atlas.v1" or sorted(by_pair) != list(range(n_pairs)):
        raise ScoreAtlasError("evaluator atlas schema/completeness mismatch")
    if header.get("provenance", {}).get("score_claim") is not False:
        raise ScoreAtlasError("evaluator atlas false-authority marker is missing")
    return header, by_pair


def audit_evaluator_response_cone_maps(rows: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    """Re-hash every #36 cone map so consumability is current-byte evidence."""

    total_bytes = 0
    for pair_index, row in sorted(rows.items()):
        refs = row.get("sensitivity_refs", {})
        path = Path(str(refs.get("cone_map_path", "")))
        expected = str(refs.get("cone_map_sha256", ""))
        if not path.is_file() or path.is_symlink():
            raise ScoreAtlasError(f"#36 cone map missing or unsafe for pair {pair_index}: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ScoreAtlasError(f"#36 cone map SHA mismatch for pair {pair_index}: expected {expected}, got {actual}")
        total_bytes += path.stat().st_size
    return {
        "schema": "ddm_g3_evaluator_response_atlas_audit.v1",
        "map_count": len(rows),
        "map_bytes": total_bytes,
        "all_embedded_sha256_verified": True,
        "consumability_verdict": "INTACT_FOR_GEOMETRY_ONLY",
        "score_currency_verdict": "ROTTED_AS_RANK_KEY; cone/free-budget fields are diagnostic only",
        "epistemic_status": "MEASURED_CURRENT_BYTES",
    }


def reconstruct_v12_state(repo: Path, receipt: Mapping[str, Any], *, n_pairs: int = 600) -> ReconstructedState:
    if receipt.get("schema") != "direct_description_v12_obligation_drain_receipt.v1":
        raise ScoreAtlasError("input receipt is not the v12 obligation-drain schema")
    if receipt.get("score_claim") is not False or receipt.get("pointer_moved") is not False:
        raise ScoreAtlasError("v12 receipt authority markers changed")
    manifests = receipt.get("resume", {}).get("base_batch_cache", [])
    if len(manifests) != math.ceil(n_pairs / 16):
        raise ScoreAtlasError("v12 receipt does not bind every canonical base batch")
    baseline = {int(row["start"]): _load_batch_cache(repo, row) for row in manifests}
    expected_starts = list(range(0, n_pairs, 16))
    if sorted(baseline) != expected_starts:
        raise ScoreAtlasError("v12 base batch cache start coverage is incomplete")
    current = dict(baseline)
    admission_rows = list(receipt.get("candidate_search", {}).get("admission_rows", []))
    for index, row in enumerate(admission_rows):
        measurement = row.get("measurement")
        caches = row.get("batch_score_cache")
        if not isinstance(measurement, dict) or not isinstance(caches, list) or not caches:
            if bool(row.get("admitted")):
                raise ScoreAtlasError(f"admitted v12 row {index} lacks measured cache custody")
            if row.get("reason") not in {
                "strict_receiver_rejected_candidate_bundle",
                "address_conflict_with_earlier_measured_admission",
            }:
                raise ScoreAtlasError(f"v12 unmeasured row {index} has an unknown disposition")
            continue
        proposals = {int(item["start"]): _load_batch_cache(repo, item) for item in caches}
        current_errors = sum(batch.errors for batch in current.values())
        if current_errors != int(measurement["errors_before"]):
            raise ScoreAtlasError(f"v12 admission replay diverged before row {index}")
        proposal_errors = (
            current_errors
            - sum(current[start].errors for start in proposals)
            + sum(batch.errors for batch in proposals.values())
        )
        if proposal_errors != int(measurement["errors_after"]):
            raise ScoreAtlasError(f"v12 proposal cache diverged at row {index}")
        if bool(row.get("admitted")):
            current.update(proposals)
    final_cells = np.concatenate([current[start].cells for start in expected_starts], axis=0)
    final_poses = np.concatenate([current[start].poses for start in expected_starts], axis=0)
    if len(final_cells) != n_pairs or len(final_poses) != n_pairs:
        raise ScoreAtlasError("v12 reconstructed final tensor coverage is incomplete")
    final_errors = sum(row.errors for row in current.values())
    final_pose_squared_error = sum(row.pose_squared_error for row in current.values())
    final_rung = receipt["ladder"][-1]
    if final_errors != int(final_rung["bridge"]["segmentation"]["errors"]):
        raise ScoreAtlasError("v12 reconstructed final SegNet total does not match receipt")
    claimed_pose_sum = float(final_rung["bridge"]["pose"]["squared_error_sum"])
    if not math.isclose(final_pose_squared_error, claimed_pose_sum, rel_tol=0.0, abs_tol=1e-6):
        raise ScoreAtlasError("v12 reconstructed final PoseNet total does not match receipt")
    archive = final_rung["archive"]
    archive_path = _resolve(repo, str(archive["path"]))
    _bound_file(archive_path, str(archive["sha256"]), "v12 final receiver archive")
    if archive_path.stat().st_size != int(archive["bytes"]):
        raise ScoreAtlasError("v12 final receiver archive byte count mismatch")
    return ReconstructedState(
        baseline_batches=baseline,
        final_batches=current,
        admission_rows=admission_rows,
        final_cells=final_cells,
        final_poses=final_poses,
        final_errors=final_errors,
        final_pose_squared_error=final_pose_squared_error,
        archive_bytes=int(archive["bytes"]),
        archive_sha256=str(archive["sha256"]),
        archive_path=archive_path,
    )


def _boundary_mask(labels: np.ndarray) -> np.ndarray:
    boundary = np.zeros_like(labels, dtype=bool)
    vertical = labels[:, 1:, :] != labels[:, :-1, :]
    horizontal = labels[:, :, 1:] != labels[:, :, :-1]
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    return boundary


def _histogram(values: np.ndarray, edges: Sequence[float]) -> dict[str, Any]:
    counts: list[int] = []
    labels: list[str] = []
    for low, high in pairwise(edges):
        mask = (values >= low) & (values < high)
        counts.append(int(np.count_nonzero(mask)))
        labels.append(f"[{low:g},{'inf' if math.isinf(high) else f'{high:g}'})")
    return {"edges": list(edges), "labels": labels, "counts": counts}


def _rank4_flip_distances(target: np.ndarray, predicted: np.ndarray, margins: np.ndarray) -> np.ndarray:
    errors = target != predicted
    if not errors.any():
        return np.empty(0, dtype=np.float64)
    target_ids = target[errors].astype(np.int64)
    predicted_ids = predicted[errors].astype(np.int64)
    norms = np.empty(len(target_ids), dtype=np.float64)
    for index, (left, right) in enumerate(zip(target_ids, predicted_ids, strict=True)):
        names = sorted((CLASS_ORDER[int(left)], CLASS_ORDER[int(right)]), key=CLASS_ORDER.index)
        norms[index] = HEAD_PAIR_NORMS[f"{names[0]}-{names[1]}"]
    return np.abs(np.asarray(margins[errors], dtype=np.float64)) / norms


def _seg_cube(
    target: np.ndarray,
    errors: np.ndarray,
    margins: np.ndarray,
    boundary: np.ndarray,
    *,
    total_video_sites: int,
) -> dict[str, Any]:
    cube: dict[str, Any] = {}
    for class_id, class_name in enumerate(CLASS_ORDER):
        by_margin: dict[str, Any] = {}
        class_mask = target == class_id
        for low, high, band_name in MARGIN_BANDS:
            margin_mask = class_mask & (margins >= low) & (margins < high)
            by_topology: dict[str, Any] = {}
            for topology_name, topology_mask in (("boundary_codim1", boundary), ("cell_interior", ~boundary)):
                sites = int(np.count_nonzero(margin_mask & topology_mask))
                count = int(np.count_nonzero(errors & margin_mask & topology_mask))
                by_topology[topology_name] = {
                    "sites": sites,
                    "flip_count": count,
                    "conditional_d_seg": count / sites if sites else 0.0,
                    "global_d_seg_mass": count / total_video_sites,
                    "seg_score_mass": 100.0 * count / total_video_sites,
                    "epistemic_status": "DERIVED_FROM_MEASURED_ARGMAX_FLIPS",
                }
            by_margin[band_name] = by_topology
        cube[class_name] = by_margin
    return cube


def _row_error_distribution(errors: np.ndarray, bins: int = ROW_BINS) -> dict[str, Any]:
    edges = np.linspace(0, errors.shape[0], bins + 1, dtype=np.int64)
    counts = [int(np.count_nonzero(errors[edges[i] : edges[i + 1]])) for i in range(bins)]
    total = sum(counts)
    return {
        "row_edges": edges.tolist(),
        "flip_counts": counts,
        "flip_fractions": [value / total if total else 0.0 for value in counts],
        "epistemic_status": "DERIVED_FROM_MEASURED_CELLS",
    }


def _movable_components(target: np.ndarray) -> int:
    from scipy import ndimage

    labels, count = ndimage.label(target == CLASS_ORDER.index("Movable"), structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0
    sizes = np.bincount(labels.ravel())[1:]
    return int(np.count_nonzero(sizes >= 8))


def _load_receiver_covariates(archive_path: Path, *, n_pairs: int) -> tuple[np.ndarray, list[float | None]]:
    from tac.optimization.direct_description_carrier_compose import receive_carrier_compose_archive

    receiver = receive_carrier_compose_archive(archive_path.read_bytes())
    if receiver.z.n_pairs != n_pairs or receiver.pose6_codes.shape != (n_pairs, 6):
        raise ScoreAtlasError("v12 receiver covariate coverage mismatch")
    lane_layer = next((layer for layer in receiver.layers if layer.role == "Lane"), None)
    if lane_layer is None or lane_layer.lane_lines is None:
        raise ScoreAtlasError("v12 receiver has no decoded Lane dash-phase layer")
    phases: list[float | None] = []
    for lines in lane_layer.lane_lines:
        values = []
        for vector in lines:
            period = float(vector[6])
            if period > 0:
                values.append(float(np.mod(float(vector[7]), period) / period))
        phases.append(float(np.mean(values)) if values else None)
    return np.asarray(receiver.pose6_codes, dtype=np.float64), phases


def _accepted_bytes_by_pair(admission_rows: Sequence[Mapping[str, Any]], n_pairs: int) -> np.ndarray:
    allocated = np.zeros(n_pairs, dtype=np.float64)
    for row in admission_rows:
        if not bool(row.get("admitted")):
            continue
        sources = sorted({int(value) for value in row["candidate"]["source_pair_ids"]})
        marginal = int(row["measurement"]["marginal_archive_bytes"])
        if not sources or any(value < 0 or value >= n_pairs for value in sources) or marginal < 0:
            raise ScoreAtlasError("accepted admission has invalid source-pair byte attribution")
        share = marginal / len(sources)
        for pair_index in sources:
            allocated[pair_index] += share
    return allocated


def _scene_event_proxies(rows: Sequence[dict[str, Any]]) -> dict[int, list[str]]:
    """Return deterministic named proxies; these are not human scene labels."""

    if not rows:
        return {}
    intersection = max(
        rows,
        key=lambda row: (
            row["segmentation"]["boundary_flip_count"]
            + row["segmentation"]["class_flip_counts"]["Road"]
            + row["segmentation"]["class_flip_counts"]["Lane"]
        ),
    )["pair_index"]
    lead_car = max(
        rows,
        key=lambda row: (
            row["segmentation"]["class_flip_counts"]["Movable"] * (1 + row["scene_covariates"]["movable_track_count"])
        ),
    )["pair_index"]
    lane_change = max(
        rows,
        key=lambda row: (
            row["scene_covariates"]["stored_turn_code_l2"] + 4.0 * abs(row["scene_covariates"]["dash_phase_delta"])
        ),
    )["pair_index"]
    result: dict[int, list[str]] = defaultdict(list)
    result[int(intersection)].append("intersection_proxy")
    result[int(lead_car)].append("lead_car_pass_proxy")
    result[int(lane_change)].append("lane_change_proxy")
    return dict(result)


def build_pair_rows(
    *,
    target_cells: np.ndarray,
    target_margins: np.ndarray,
    target_poses: np.ndarray,
    state: ReconstructedState,
    evaluator_rows: Mapping[int, Mapping[str, Any]],
    source_custody: Mapping[str, Any],
) -> list[dict[str, Any]]:
    n_pairs = len(state.final_cells)
    if target_cells.shape != (n_pairs, 384, 512) or target_margins.shape != target_cells.shape:
        raise ScoreAtlasError("target cell/margin shape mismatch")
    if target_poses.shape != (n_pairs, 6):
        raise ScoreAtlasError("target pose shape mismatch")
    boundaries = _boundary_mask(target_cells)
    pose6_codes, dash_phases = _load_receiver_covariates(state.archive_path, n_pairs=n_pairs)
    accepted_bytes = _accepted_bytes_by_pair(state.admission_rows, n_pairs)
    accepted_total = float(accepted_bytes.sum())
    base_bytes = state.archive_bytes - accepted_total
    if base_bytes <= 0 or not math.isclose(accepted_total, 4001.0, rel_tol=0.0, abs_tol=1e-9):
        raise ScoreAtlasError("v12 exact accepted/base byte decomposition changed")
    base_per_pair = base_bytes / n_pairs
    total_pixels = n_pairs * 384 * 512
    pair_pose_sq = np.square(state.final_poses - target_poses).sum(axis=1, dtype=np.float64)
    global_pose_score = math.sqrt(10.0 * state.final_pose_squared_error / (n_pairs * 6))
    rows: list[dict[str, Any]] = []
    for pair_index in range(n_pairs):
        target = np.asarray(target_cells[pair_index])
        predicted = np.asarray(state.final_cells[pair_index])
        margins = np.asarray(target_margins[pair_index])
        errors = predicted != target
        boundary = boundaries[pair_index]
        flip_count = int(np.count_nonzero(errors))
        class_counts = {
            name: int(np.count_nonzero(errors & (target == class_id))) for class_id, name in enumerate(CLASS_ORDER)
        }
        flip_distances = _rank4_flip_distances(target, predicted, margins)
        atlas_row = evaluator_rows[pair_index]
        pose_sq = float(pair_pose_sq[pair_index])
        seg_mass = 100.0 * flip_count / total_pixels
        pose_mass = global_pose_score * pose_sq / state.final_pose_squared_error
        allocated = base_per_pair + float(accepted_bytes[pair_index])
        pose_code = pose6_codes[pair_index]
        previous_pose_code = pose6_codes[pair_index - 1] if pair_index else pose_code
        phase = dash_phases[pair_index]
        previous_phase = dash_phases[pair_index - 1] if pair_index else phase
        phase_delta = 0.0 if phase is None or previous_phase is None else float(phase - previous_phase)
        response_geometry = {
            "seg_margin_stats": atlas_row["seg_margin_field_stats"],
            "pose_jacobian_stats": atlas_row["pose_jacobian_norm_stats"],
            "joint_cone_summary_diagnostic_only": atlas_row["joint_cone_summary"],
            "cone_map_path": atlas_row["sensitivity_refs"]["cone_map_path"],
            "cone_map_sha256": atlas_row["sensitivity_refs"]["cone_map_sha256"],
            "l2_fields_rank_eligible": False,
            "epistemic_status": "MEASURED_REUSED",
        }
        row = {
            "schema": PAIR_SCHEMA,
            "pair_index": pair_index,
            "frame_indices": (2 * pair_index, 2 * pair_index + 1),
            "scored_seg_frame_index": 2 * pair_index + 1,
            "score_rank": 0,
            "score_mass": {
                "seg_score_mass": seg_mass,
                "pose_score_mass": pose_mass,
                "distortion_score_mass": seg_mass + pose_mass,
                "rate_score_mass_diagnostic": RATE_PER_BYTE * allocated,
                "rank_key": PRIMARY_RANK_KEY,
                "epistemic_status": "DERIVED",
            },
            "segmentation": {
                "flip_count": flip_count,
                "d_seg_pair": flip_count / (384 * 512),
                "class_flip_counts": class_counts,
                "boundary_flip_count": int(np.count_nonzero(errors & boundary)),
                "interior_flip_count": int(np.count_nonzero(errors & ~boundary)),
                "stratum_margin_topology_cube": _seg_cube(
                    target,
                    errors,
                    margins,
                    boundary,
                    total_video_sites=n_pairs * 384 * 512,
                ),
                "rank4_flip_distance_histogram": _histogram(flip_distances, FLIP_DISTANCE_EDGES),
                "rank4_flip_distance_quantiles": {
                    name: (float(np.quantile(flip_distances, q)) if len(flip_distances) else None)
                    for name, q in (("p10", 0.1), ("median", 0.5), ("p90", 0.9))
                },
                "rank4_semantics": "DERIVED cheapest-hyperplane lower bound abs(gt_top1_top2_margin)/head_pair_norm",
                "image_row_error_distribution": _row_error_distribution(errors),
                "epistemic_status": "MEASURED_CELLS_PLUS_DERIVED_REDUCTIONS",
            },
            "pose": {
                "squared_error_sum": pose_sq,
                "d_pose_pair": pose_sq / 6.0,
                "pose_sensitivity_l2_diagnostic": atlas_row["pose_jacobian_norm_stats"]["l2_norm"],
                "pose_binds_fraction": atlas_row["joint_cone_summary"]["pose_binds_fraction"],
                "epistemic_status": "MEASURED_OUTPUT_PLUS_REUSED_SENSITIVITY",
            },
            "allocated_bytes": {
                "base_archive_uniform_amortized_bytes": base_per_pair,
                "accepted_admission_causal_bytes": float(accepted_bytes[pair_index]),
                "allocated_bytes": allocated,
                "allocation_method": "DERIVED exact-base-uniform plus exact-admission-marginal/source-pair split",
                "global_exact_archive_bytes": state.archive_bytes,
                "global_exact_archive_sha256": state.archive_sha256,
                "epistemic_status": "DERIVED_FROM_BYTE_CLOSE_LEDGER",
            },
            "scene_covariates": {
                "stored_pose6_code_l2": float(np.linalg.norm(pose_code)),
                "stored_translation_code_speed_l2": float(np.linalg.norm(pose_code[:3])),
                "stored_turn_code_l2": float(np.linalg.norm(pose_code[3:])),
                "stored_pose6_code_delta_l2": float(np.linalg.norm(pose_code - previous_pose_code)),
                "movable_track_count": _movable_components(target),
                "movable_track_definition": "DERIVED 8-connected GT-cell components with at least 8 sites",
                "dash_phase_mean": phase,
                "dash_phase_delta": phase_delta,
                "scene_event_labels": [],
                "scene_event_label_authority": ("DERIVED_COVARIATE_SPIKE_PROXY; not human semantic ground truth"),
                "epistemic_status": "DERIVED_FROM_STORED_RECEIVER_AND_FROZEN_TARGET_CELLS",
            },
            "evaluator_response_geometry": response_geometry,
            "costate_signal": {
                "schema": "ddm_g3_costate_pair_signal.v1",
                "pair_index": pair_index,
                "lambda_proxy_score_debt": seg_mass + pose_mass,
                "seg_flip_count": flip_count,
                "median_rank4_flip_distance": (float(np.median(flip_distances)) if len(flip_distances) else None),
                "pose_squared_error_sum": pose_sq,
                "pose_binds_fraction": atlas_row["joint_cone_summary"]["pose_binds_fraction"],
                "allocated_bytes": allocated,
                "ranking_currency": "exact_flip_plus_pose_objective_mass",
                "energy_rank_forbidden": True,
                "score_claim": False,
            },
            "source_custody": dict(source_custody),
            "evidence_axis": AXIS,
            "score_claim": False,
            "promotion_eligible": False,
        }
        rows.append(row)
    ranked = sorted(rows, key=lambda row: (-row["score_mass"]["distortion_score_mass"], row["pair_index"]))
    for rank, row in enumerate(ranked, start=1):
        row["score_rank"] = rank
    labels = _scene_event_proxies(rows)
    for row in rows:
        row["scene_covariates"]["scene_event_labels"] = labels.get(row["pair_index"], [])
    validated = [DdmG3ScoreAtlasPairV1.model_validate(row).model_dump(mode="json", by_alias=True) for row in rows]
    if sorted(row["pair_index"] for row in validated) != list(range(n_pairs)):
        raise ScoreAtlasError("typed atlas row coverage is incomplete")
    return validated


def select_stratified_control(rows: Sequence[Mapping[str, Any]], *, excluded: set[int], k: int = 24) -> list[int]:
    """Select temporal x score-quartile controls without using L2/energy."""

    if k != 24 or len(rows) != 600:
        raise ScoreAtlasError("the governed stratified control is 24 of 600")
    score = {int(row["pair_index"]): float(row["score_mass"]["distortion_score_mass"]) for row in rows}
    selected: list[int] = []
    for time_bin in range(6):
        pool = [index for index in range(time_bin * 100, (time_bin + 1) * 100) if index not in excluded]
        ordered = sorted(pool, key=lambda index: (score[index], index))
        for quantile in (0.125, 0.375, 0.625, 0.875):
            candidate = ordered[min(len(ordered) - 1, int(quantile * len(ordered)))]
            if candidate not in selected:
                selected.append(candidate)
    if len(selected) != k:
        raise ScoreAtlasError("stratified control selection did not produce 24 unique pairs")
    return selected


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def measure_subset_correlations(
    repo: Path,
    state: ReconstructedState,
    target_cells: np.ndarray,
    target_poses: np.ndarray,
    subsets: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Replay v12 measured proposals and correlate subset vs full score deltas."""

    current = dict(state.baseline_batches)
    current_pose_total = sum(batch.pose_squared_error for batch in current.values())
    total_pixels = len(target_cells) * 384 * 512
    pose_coordinates = len(target_poses) * 6
    full_gains: list[float] = []
    subset_gains: dict[str, list[float]] = {name: [] for name in subsets}
    subset_distortion_touched: dict[str, int] = dict.fromkeys(subsets, 0)
    validation_residuals: list[float] = []
    for row in state.admission_rows:
        if not isinstance(row.get("measurement"), dict) or not row.get("batch_score_cache"):
            continue
        proposals = {int(item["start"]): _load_batch_cache(repo, item) for item in row["batch_score_cache"]}
        measurement = row["measurement"]
        rate_cost = float(measurement["rate_objective_delta"])
        reconstructed_error_gain = 0
        reconstructed_pose_delta = 0.0
        per_pair: dict[int, tuple[int, float]] = {}
        for start, proposal in proposals.items():
            before = current[start]
            stop = start + len(proposal.cells)
            target_batch = target_cells[start:stop]
            pose_batch = target_poses[start:stop]
            before_errors = np.count_nonzero(before.cells != target_batch, axis=(1, 2)).astype(np.int64)
            after_errors = np.count_nonzero(proposal.cells != target_batch, axis=(1, 2)).astype(np.int64)
            before_pose = np.square(before.poses - pose_batch).sum(axis=1, dtype=np.float64)
            after_pose = np.square(proposal.poses - pose_batch).sum(axis=1, dtype=np.float64)
            for offset in range(len(proposal.cells)):
                pair_index = start + offset
                per_pair[pair_index] = (
                    int(before_errors[offset] - after_errors[offset]),
                    float(after_pose[offset] - before_pose[offset]),
                )
            reconstructed_error_gain += int(before_errors.sum() - after_errors.sum())
            reconstructed_pose_delta += float(after_pose.sum() - before_pose.sum())
        full_gain = (
            100.0 * reconstructed_error_gain / total_pixels
            + math.sqrt(10.0 * current_pose_total / pose_coordinates)
            - math.sqrt(10.0 * (current_pose_total + reconstructed_pose_delta) / pose_coordinates)
            - rate_cost
        )
        receipt_gain = float(measurement["measured_objective_gain"])
        validation_residuals.append(full_gain - receipt_gain)
        full_gains.append(receipt_gain)
        for name, pair_indices in subsets.items():
            chosen = {int(value) for value in pair_indices}
            error_gain = sum(value[0] for pair, value in per_pair.items() if pair in chosen)
            pose_delta = sum(value[1] for pair, value in per_pair.items() if pair in chosen)
            if error_gain != 0 or not math.isclose(pose_delta, 0.0, rel_tol=0.0, abs_tol=1e-15):
                subset_distortion_touched[name] += 1
            gain = (
                100.0 * error_gain / total_pixels
                + math.sqrt(10.0 * current_pose_total / pose_coordinates)
                - math.sqrt(10.0 * (current_pose_total + pose_delta) / pose_coordinates)
                - rate_cost
            )
            subset_gains[name].append(gain)
        if bool(row.get("admitted")):
            current.update(proposals)
            current_pose_total += reconstructed_pose_delta
    if max(abs(value) for value in validation_residuals) > 2e-9:
        raise ScoreAtlasError("v12 admission objective replay does not match measured receipt")
    return {
        "schema": "ddm_g3_hard_subset_correlation.v1",
        "measurement_source": "v12 exact frozen-scorer measured admission proposals",
        "n_measured_proposals": len(full_gains),
        "full_delta_definition": "measured full-n600 joint objective gain including exact marginal rate",
        "subset_delta_definition": "same objective with only subset pair distortion deltas plus exact marginal rate",
        "max_full_replay_residual": max(abs(value) for value in validation_residuals),
        "correlations": {
            name: {
                "pearson_r": _pearson(values, full_gains),
                "nonzero_subset_estimate_rows_including_rate": int(np.count_nonzero(np.asarray(values))),
                "distortion_touched_rows": subset_distortion_touched[name],
                "pair_count": len(subsets[name]),
            }
            for name, values in subset_gains.items()
        },
        "evidence_axis": AXIS,
        "score_claim": False,
        "epistemic_status": "MEASURED_REPLAY",
    }


def build_admission_efficiency_rows(
    receipt: Mapping[str, Any],
    *,
    source_version: Literal["v10", "v11", "v12"],
    n_pairs: int,
) -> list[dict[str, Any]]:
    """Normalize the three measured-admission sign conventions into gain currency."""

    total_pixels = n_pairs * 384 * 512
    rows = []
    for index, row in enumerate(receipt["candidate_search"]["admission_rows"]):
        measurement = row.get("measurement")
        if not isinstance(measurement, dict):
            continue
        marginal_bytes = int(measurement["marginal_archive_bytes"])
        improvement = int(measurement["errors_before"]) - int(measurement["errors_after"])
        if source_version == "v10":
            joint_gain = float(measurement["distortion_gain_score_units"]) - float(measurement["rate_cost_score_units"])
            gain_definition = "distortion_gain_score_units - rate_cost_score_units"
        elif source_version == "v11":
            joint_gain = -float(measurement["joint_objective_delta"])
            gain_definition = "-joint_objective_delta"
        else:
            joint_gain = float(measurement["measured_objective_gain"])
            gain_definition = "measured_objective_gain"
        rows.append(
            {
                "schema": ADMISSION_SCHEMA,
                "source_version": source_version,
                "source_pair_count": n_pairs,
                "candidate_index": index,
                "candidate_id": row["candidate"]["candidate_id"],
                "source_pair_ids": row["candidate"]["source_pair_ids"],
                "admitted": bool(row["admitted"]),
                "marginal_archive_bytes": marginal_bytes,
                "seg_flip_reduction": improvement,
                "delta_d_seg": improvement / total_pixels,
                "delta_d_seg_per_byte": improvement / (total_pixels * marginal_bytes) if marginal_bytes else None,
                "joint_objective_gain": joint_gain,
                "joint_objective_gain_per_byte": joint_gain / marginal_bytes if marginal_bytes else None,
                "joint_objective_gain_definition": gain_definition,
                "rank_seed": f"MEASURED {source_version} admission; exact flips and archive bytes",
                "evidence_axis": AXIS,
                "score_claim": False,
            }
        )
    return rows


def build_summary(rows: Sequence[Mapping[str, Any]], state: ReconstructedState) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: int(row["score_rank"]))
    mass_fields = {
        "joint_distortion": "distortion_score_mass",
        "seg": "seg_score_mass",
        "pose": "pose_score_mass",
    }
    concentration: dict[str, dict[str, float]] = {}
    for component, field in mass_fields.items():
        component_ranked = sorted(rows, key=lambda row: (-float(row["score_mass"][field]), row["pair_index"]))
        masses = np.asarray([float(row["score_mass"][field]) for row in component_ranked])
        total = float(masses.sum())
        concentration[component] = {f"top{k}": float(masses[:k].sum() / total) if total else 0.0 for k in (10, 50, 100)}
    return {
        "schema": SCHEMA,
        "n_pairs": len(rows),
        "primary_rank_key": PRIMARY_RANK_KEY,
        "rank_formula": (
            "100*pair_flip_count/(600*384*512) + sqrt(10*global_d_pose)*"
            "pair_pose_squared_error/global_pose_squared_error"
        ),
        "energy_or_l2_rank_keys_allowed": False,
        "heavy_tail_concentration": concentration,
        "heavy_tail_interpretation": (
            "fractions are concentration diagnostics; the measured joint distribution is broad when top-k is low"
        ),
        "top10_pairs": [
            {
                "pair_index": int(row["pair_index"]),
                "score_mass": float(row["score_mass"]["distortion_score_mass"]),
                "flip_count": int(row["segmentation"]["flip_count"]),
                "d_pose_pair": float(row["pose"]["d_pose_pair"]),
                "scene_event_labels": row["scene_covariates"]["scene_event_labels"],
            }
            for row in ranked[:10]
        ],
        "global_reconstruction": {
            "d_seg": state.final_errors / (len(rows) * 384 * 512),
            "d_pose": state.final_pose_squared_error / (len(rows) * 6),
            "archive_bytes": state.archive_bytes,
            "archive_sha256": state.archive_sha256,
            "advisory_objective": (
                100.0 * state.final_errors / (len(rows) * 384 * 512)
                + math.sqrt(10.0 * state.final_pose_squared_error / (len(rows) * 6))
                + RATE_PER_BYTE * state.archive_bytes
            ),
        },
        "pointer": POINTER,
        "pointer_moved": False,
        "evidence_axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
    }


def build_hard_pair_registry(
    rows: Sequence[Mapping[str, Any]],
    correlations: Mapping[str, Any],
    *,
    atlas_jsonl_path: Path,
    atlas_jsonl_sha256: str,
    source_custody: Mapping[str, Any],
) -> dict[str, Any]:
    ranked = sorted(rows, key=lambda row: int(row["score_rank"]))
    top24 = [int(row["pair_index"]) for row in ranked[:24]]
    top64 = [int(row["pair_index"]) for row in ranked[:64]]
    control = select_stratified_control(rows, excluded=set(top64), k=24)
    return {
        "schema": HARD_PAIR_SCHEMA,
        "selection_rank_key": PRIMARY_RANK_KEY,
        "selection_rank_formula": "exact flip score mass plus proportional nonlinear Pose score mass",
        "top24": top24,
        "top64": top64,
        "stratified_control24": control,
        "correlation_receipt": correlations,
        "measure_first_contract": {
            "order": ["top24", "top64", "stratified_control24", "full_n600"],
            "rule": (
                "measure exact frozen-scorer delta on top24 first; continue to top64/control; "
                "a subset result cannot stand in for n600 unless a contemporaneous full-n600 delta "
                "updates the measured r"
            ),
            "rank_or_kill_from_subset_alone": False,
            "score_claim": False,
        },
        "atlas_jsonl": {
            "path": str(atlas_jsonl_path),
            "sha256": atlas_jsonl_sha256,
            "bytes": atlas_jsonl_path.stat().st_size,
        },
        "source_custody": dict(source_custody),
        "pointer": POINTER,
        "pointer_moved": False,
        "evidence_axis": AXIS,
        "score_claim": False,
        "promotion_eligible": False,
    }


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(_canonical_json(row) + b"\n" for row in rows)
    atomic_write(path, payload)


def write_charts(
    output_directory: Path,
    rows: Sequence[Mapping[str, Any]],
    admissions: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chart_directory = output_directory / "charts"
    chart_directory.mkdir(parents=True, exist_ok=True)
    ranked = sorted(rows, key=lambda row: int(row["score_rank"]))
    timeline = sorted(rows, key=lambda row: int(row["pair_index"]))
    created: list[Path] = []
    with tempfile.TemporaryDirectory(prefix="chart_stage_", dir=output_directory) as tmp:
        stage = Path(tmp)
        masses = np.asarray([row["score_mass"]["distortion_score_mass"] for row in ranked], dtype=np.float64)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
        axes[0].bar(np.arange(100), masses[:100], width=1.0, color="#d95f02")
        axes[0].set(title="Top 100 pair score-debt mass", xlabel="score rank", ylabel="score mass")
        axes[1].plot(np.arange(1, 601), np.cumsum(masses) / masses.sum(), color="#1b9e77")
        axes[1].axvline(10, color="black", alpha=0.3)
        axes[1].axvline(50, color="black", alpha=0.3)
        axes[1].axvline(100, color="black", alpha=0.3)
        axes[1].set(title="Cumulative score-debt mass", xlabel="top k pairs", ylabel="fraction")
        fig.tight_layout()
        fig.savefig(stage / "score_mass_rank_and_cumulative.png")
        plt.close(fig)

        x = np.arange(600)
        y = np.asarray([row["score_mass"]["distortion_score_mass"] for row in timeline])
        fig, ax = plt.subplots(figsize=(15, 5), dpi=150)
        ax.plot(x, y, linewidth=0.8, color="#4575b4")
        for row in timeline:
            for label in row["scene_covariates"]["scene_event_labels"]:
                pair_index = int(row["pair_index"])
                ax.scatter([pair_index], [y[pair_index]], color="#d73027", zorder=3)
                ax.annotate(f"{label} p{pair_index}", (pair_index, y[pair_index]), fontsize=8)
        ax.set(
            title="n600 score-debt timeline (named events are DERIVED proxies)",
            xlabel="pair index",
            ylabel="score mass",
        )
        fig.tight_layout()
        fig.savefig(stage / "score_mass_timeline.png")
        plt.close(fig)

        versions = sorted({str(row["source_version"]) for row in admissions})
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=150)
        offset = 0
        for version in versions:
            version_rows = [row for row in admissions if row["source_version"] == version]
            joint_efficiencies = np.asarray(
                [row["joint_objective_gain_per_byte"] for row in version_rows], dtype=np.float64
            )
            seg_efficiencies = np.asarray([row["delta_d_seg_per_byte"] for row in version_rows], dtype=np.float64)
            admitted = np.asarray([row["admitted"] for row in version_rows], dtype=bool)
            x_values = np.arange(offset, offset + len(version_rows))
            axes[0].scatter(
                x_values[~admitted], seg_efficiencies[~admitted], s=10, alpha=0.5, label=f"{version} rejected"
            )
            axes[0].scatter(x_values[admitted], seg_efficiencies[admitted], s=20, label=f"{version} admitted")
            axes[1].scatter(
                x_values[~admitted],
                joint_efficiencies[~admitted],
                s=10,
                alpha=0.5,
                label=f"{version} rejected",
            )
            axes[1].scatter(x_values[admitted], joint_efficiencies[admitted], s=20, label=f"{version} admitted")
            offset += len(version_rows)
        axes[0].axhline(0.0, color="black", linewidth=0.8)
        axes[0].set(
            title="Measured v10-v12 marginal Seg efficiency",
            xlabel="measured proposal index",
            ylabel="delta d_seg / byte",
        )
        axes[1].axhline(0.0, color="black", linestyle="--", label="net-gain threshold")
        axes[1].set(
            title="Measured marginal joint-objective efficiency",
            xlabel="measured proposal index",
            ylabel="joint objective gain / byte",
        )
        axes[0].legend(fontsize=7)
        axes[1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(stage / "measured_admission_efficiency.png")
        plt.close(fig)

        html = [
            "<!doctype html><meta charset='utf-8'><title>DDM G3 score atlas</title>",
            "<h1>DDM G3 n600 score atlas</h1>",
            f"<p>Axis: {AXIS}; score_claim=false; pointer {POINTER} unchanged.</p>",
            "<p>Primary rank: exact flip + Pose objective mass. L2/energy rank is forbidden.</p>",
            "<ul>",
            *[
                f"<li>{component} {key}: {value:.6f}</li>"
                for component, values in summary["heavy_tail_concentration"].items()
                for key, value in values.items()
            ],
            "</ul>",
            "<img src='score_mass_rank_and_cumulative.png' width='100%'>",
            "<img src='score_mass_timeline.png' width='100%'>",
            "<img src='measured_admission_efficiency.png' width='100%'>",
            "<h2>Top 100 pairs</h2><table><tr><th>rank</th><th>pair</th><th>score mass</th>"
            "<th>flips</th><th>d_pose</th><th>scene proxy</th></tr>",
        ]
        for row in ranked[:100]:
            html.append(
                "<tr>"
                f"<td>{row['score_rank']}</td><td>{row['pair_index']}</td>"
                f"<td>{row['score_mass']['distortion_score_mass']:.9g}</td>"
                f"<td>{row['segmentation']['flip_count']}</td>"
                f"<td>{row['pose']['d_pose_pair']:.9g}</td>"
                f"<td>{', '.join(row['scene_covariates']['scene_event_labels'])}</td>"
                "</tr>"
            )
        html.append("</table>")
        (stage / "score_atlas.html").write_text("\n".join(html))
        for source in sorted(stage.iterdir()):
            destination = chart_directory / source.name
            os.replace(source, destination)
            created.append(destination)
    return created


def storage_preflight(path: Path, *, required_free_bytes: int = 64 * 1024 * 1024) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    if usage.free < required_free_bytes:
        raise ScoreAtlasError(f"storage preflight failed at {path}: {usage.free} free < {required_free_bytes} required")
    return {
        "path": str(path),
        "observed_free_bytes": usage.free,
        "required_free_bytes": required_free_bytes,
        "status": "PASS",
    }


__all__ = [
    "ADMISSION_SCHEMA",
    "AXIS",
    "HARD_PAIR_SCHEMA",
    "PAIR_SCHEMA",
    "POINTER",
    "PRIMARY_RANK_KEY",
    "SCHEMA",
    "DdmG3CostatePairSignalV1",
    "DdmG3ScoreAtlasConfigV1",
    "DdmG3ScoreAtlasPairV1",
    "ScoreAtlasError",
    "atomic_write",
    "audit_evaluator_response_cone_maps",
    "build_admission_efficiency_rows",
    "build_hard_pair_registry",
    "build_pair_rows",
    "build_summary",
    "load_evaluator_response_rows",
    "measure_subset_correlations",
    "reconstruct_v12_state",
    "select_stratified_control",
    "sha256_file",
    "storage_preflight",
    "write_charts",
    "write_json",
    "write_jsonl",
]
