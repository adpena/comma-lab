# SPDX-License-Identifier: MIT
"""SHA-bound real-target apparatus rung for the Task #603 DDM.

This module reuses the existing V2 counted-description receiver but replaces
its seeded fixture target with exact, previously solved C1 scorer-plane bytes.
The bounded n64 result is research-only apparatus evidence: its integer plane
and Pose6 debts are not SegNet/PoseNet distortions and are never a score.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from tac.contest_compliance import compute_upstream_snapshot_sha256
from tac.optimization.direct_description_minimizer import (
    _V2_BODY_BYTES,
    _V2_STREAM_ORDER,
    POINTER_SCORE_TEXT,
    SEED,
    CountedDescriptionStreamV2,
    DirectDescriptionError,
    DirectDescriptionReceiverResultV2,
    DirectDescriptionSearchStageV1,
    DirectDescriptionZV2,
    _publish_new_bytes,
    _read_regular_file_once,
    _require_exact_nonnegative_int,
    _require_sha256,
    _seeded_stream_bytes,
    _sha256,
    _stage_coordinates,
    compile_direct_description_archive_v2,
    parse_direct_description_archive_v2,
    receive_direct_description_archive_v2,
    rfc8785_canonicalize,
)

TARGET_SCHEMA: Final = "direct_description_full_precision_target_planes.v1"
RUNG_SCHEMA: Final = "direct_description_real_target_pose_rung0.v1"
CHECKPOINT_SCHEMA: Final = "DirectDescriptionRealTargetCheckpointV1"
PAIR_COUNT: Final = 64
FULL_PAIR_COUNT: Final = 600
SCORER_HW: Final = (384, 512)
TARGET_OUTPUT_SHAPE: Final = (PAIR_COUNT, 2, 8, 8, 3)
EVIDENCE_AXIS: Final = "[macOS-CPU real-target subset n64 apparatus]"


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DirectDescriptionError(f"target source is unreadable: {path}") from exc
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_regular_file_once(Path(path))

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DirectDescriptionError(f"duplicate {label} key: {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(payload, object_pairs_hook=hook)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise DirectDescriptionError(f"{label} must be a JSON object")
    return value, payload


class TargetSourceFileV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    path: StrictStr
    bytes: StrictInt = Field(ge=1)
    sha256: StrictStr

    @model_validator(mode="after")
    def _valid(self) -> TargetSourceFileV1:
        _require_sha256(self.sha256, "source sha256")
        if not Path(self.path).is_absolute():
            raise ValueError("target source paths must be absolute")
        return self


class TargetChunkV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    chunk_index: StrictInt = Field(ge=0)
    pair_ids: tuple[StrictInt, ...]
    manifest: TargetSourceFileV1
    y0: TargetSourceFileV1
    y1: TargetSourceFileV1

    @model_validator(mode="after")
    def _valid(self) -> TargetChunkV1:
        expected = tuple(range(self.chunk_index * 12, self.chunk_index * 12 + 12))
        if self.pair_ids != expected:
            raise ValueError("target chunk pair coverage is not canonical")
        return self


class DirectDescriptionTargetPlaneReceiptV1(BaseModel):
    """Typed content address for the already-solved C1 n600 target."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["direct_description_full_precision_target_planes.v1"] = Field(
        default=TARGET_SCHEMA, alias="schema", serialization_alias="schema"
    )
    task: Literal[603] = 603
    source_role: Literal["existing_c1_solved_pair_scorer_planes"] = "existing_c1_solved_pair_scorer_planes"
    reconstruction_only_no_resolve: Literal[True] = True
    producer_path: StrictStr
    producer_git_sha: StrictStr
    producer_source_sha256: StrictStr
    materializer_git_sha: StrictStr
    upstream_repo_root: StrictStr
    upstream_snapshot_sha256: StrictStr
    upstream_evaluate_sha256: StrictStr
    prepare_receipt: TargetSourceFileV1
    contest_cpu_receipt: TargetSourceFileV1
    contest_cpu_provenance: TargetSourceFileV1
    archive: TargetSourceFileV1
    source_cache: TargetSourceFileV1
    pairs: Literal[600] = FULL_PAIR_COUNT
    scorer_hw: tuple[Literal[384], Literal[512]] = SCORER_HW
    plane_dtype: Literal["uint8"] = "uint8"
    y0_bytes: Literal[353_894_400] = 353_894_400
    y0_sha256: StrictStr
    y1_bytes: Literal[353_894_400] = 353_894_400
    y1_sha256: StrictStr
    chunk_tree_sha256: StrictStr
    chunks: tuple[TargetChunkV1, ...]
    subset_pair_ids: tuple[StrictInt, ...]
    subset_y0_sha256: StrictStr
    subset_y1_sha256: StrictStr
    subset_projection_recipe: Literal["integer_block_mean_48x64_to_8x8_half_up.v1"] = (
        "integer_block_mean_48x64_to_8x8_half_up.v1"
    )
    subset_projection_sha256: StrictStr
    pose6_source_key: Literal["gt_poses"] = "gt_poses"
    pose6_source_dtype: Literal["float64"] = "float64"
    pose6_source_shape: tuple[Literal[600], Literal[6]] = (600, 6)
    pose6_source_sha256: StrictStr
    pose6_target_recipe: Literal["per_coordinate_n600_ordinal_uint8_pair_tiebreak.v1"] = (
        "per_coordinate_n600_ordinal_uint8_pair_tiebreak.v1"
    )
    subset_pose6_source_sha256: StrictStr
    subset_pose6_target_codes_sha256: StrictStr
    evidence_axis: Literal["[macOS-CPU real-target subset n64 apparatus]"] = EVIDENCE_AXIS
    research_only: Literal[True] = True
    score_claim: Literal[False] = False
    pointer_moved: Literal[False] = False
    pointer: Literal["0.1910828242 [contest-CPU]"] = "0.1910828242 [contest-CPU]"
    verdict_scope: Literal[
        "exact solved-plane custody plus deterministic n64 apparatus projection; no scorer or score"
    ] = "exact solved-plane custody plus deterministic n64 apparatus projection; no scorer or score"

    @model_validator(mode="after")
    def _valid(self) -> DirectDescriptionTargetPlaneReceiptV1:
        for field in (
            "producer_source_sha256",
            "upstream_snapshot_sha256",
            "upstream_evaluate_sha256",
            "y0_sha256",
            "y1_sha256",
            "chunk_tree_sha256",
            "subset_y0_sha256",
            "subset_y1_sha256",
            "subset_projection_sha256",
            "pose6_source_sha256",
            "subset_pose6_source_sha256",
            "subset_pose6_target_codes_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if len(self.producer_git_sha) != 40 or len(self.materializer_git_sha) != 40:
            raise ValueError("target receipt git SHAs must be full length")
        if len(self.chunks) != 50 or tuple(row.chunk_index for row in self.chunks) != tuple(range(50)):
            raise ValueError("target receipt must cover all 50 canonical n600 chunks")
        if self.subset_pair_ids != tuple(range(PAIR_COUNT)):
            raise ValueError("rung-zero target subset must be the canonical first 64 pairs")
        return self


def _source_file(path: Path, expected_sha256: str | None = None) -> TargetSourceFileV1:
    path = Path(path).resolve()
    if not path.is_file():
        raise DirectDescriptionError(f"target source is not a regular file: {path}")
    observed = _stream_sha256(path)
    if expected_sha256 is not None and observed != _require_sha256(expected_sha256, "expected source sha256"):
        raise DirectDescriptionError(f"target source SHA-256 mismatch: {path}")
    return TargetSourceFileV1(path=str(path), bytes=path.stat().st_size, sha256=observed)


def _git_output(args: Sequence[str], repo_root: Path) -> bytes:
    completed = subprocess.run(["git", *args], cwd=repo_root, check=False, capture_output=True)
    if completed.returncode != 0 or not completed.stdout:
        raise DirectDescriptionError(f"git provenance lookup failed: {' '.join(args)}")
    return bytes(completed.stdout)


def _committed_source_custody(relative_path: str) -> dict[str, str]:
    """Bind a runtime source to the last commit that contains its exact bytes."""

    repo_root = Path(__file__).resolve().parents[3]
    source_path = repo_root / relative_path
    source_sha256 = _stream_sha256(source_path)
    git_sha = _git_output(["log", "-n", "1", "--format=%H", "--", relative_path], repo_root).decode().strip()
    if len(git_sha) != 40:
        raise DirectDescriptionError(f"source git custody is not full length: {relative_path}")
    committed_source = _git_output(["show", f"{git_sha}:{relative_path}"], repo_root)
    if _sha256(committed_source) != source_sha256:
        raise DirectDescriptionError(f"runtime source is not committed at its claimed git SHA: {relative_path}")
    return {"path": relative_path, "sha256": source_sha256, "git_sha": git_sha}


def _block_mean_projection(planes: np.ndarray) -> np.ndarray:
    value = np.asarray(planes)
    if value.dtype != np.uint8 or value.ndim != 4 or value.shape[1:] != (384, 512, 3):
        raise DirectDescriptionError("real target projection requires uint8 [N,384,512,3] planes")
    reshaped = value.astype(np.uint64).reshape(value.shape[0], 8, 48, 8, 64, 3)
    sums = reshaped.sum(axis=(2, 4), dtype=np.uint64)
    return np.ascontiguousarray(((sums + 1536) // 3072).astype(np.uint8))


def _pose6_ordinal_codes(poses: np.ndarray) -> np.ndarray:
    value = np.asarray(poses)
    if value.dtype != np.float64 or value.shape != (FULL_PAIR_COUNT, 6) or not np.isfinite(value).all():
        raise DirectDescriptionError("real target Pose6 source must be finite float64 [600,6]")
    codes = np.empty((FULL_PAIR_COUNT, 6), dtype=np.uint8)
    pair_ids = np.arange(FULL_PAIR_COUNT, dtype=np.int64)
    for coordinate in range(6):
        order = np.lexsort((pair_ids, value[:, coordinate]))
        rank = np.empty(FULL_PAIR_COUNT, dtype=np.int64)
        rank[order] = pair_ids
        codes[:, coordinate] = ((rank * 255 + 299) // 599).astype(np.uint8)
    return codes


def _load_pose_source(cache_path: Path) -> np.ndarray:
    try:
        with np.load(cache_path, allow_pickle=False) as cache:
            poses = np.ascontiguousarray(cache["gt_poses"])
    except (OSError, KeyError, ValueError) as exc:
        raise DirectDescriptionError("target cache lacks readable gt_poses custody") from exc
    if poses.dtype != np.float64 or poses.shape != (600, 6):
        raise DirectDescriptionError("target cache gt_poses geometry/dtype mismatch")
    return poses


def build_target_plane_receipt(
    *,
    prepare_receipt_path: Path,
    contest_cpu_receipt_path: Path,
    contest_cpu_provenance_path: Path,
    repo_root: Path,
    upstream_repo_root: Path,
) -> DirectDescriptionTargetPlaneReceiptV1:
    """Re-receipt existing C1 bytes without solving or materializing new planes."""

    prepare, prepare_payload = _read_json(prepare_receipt_path, "C1 prepare receipt")
    if (
        prepare.get("schema") != "v10_two_plane_receiver_prepare.v1"
        or prepare.get("completed") is not True
        or prepare.get("pair_count") != 600
        or prepare.get("chunk_count") != 50
        or prepare.get("chunk_pairs") != 12
        or prepare.get("scorer_hw") != [384, 512]
    ):
        raise DirectDescriptionError("C1 prepare receipt does not bind complete n600 solved planes")
    contest, contest_payload = _read_json(contest_cpu_receipt_path, "contest CPU receipt")
    provenance, provenance_payload = _read_json(contest_cpu_provenance_path, "contest CPU provenance")
    archive_path = Path(str(prepare.get("archive_path")))
    archive = _source_file(archive_path, str(prepare.get("archive_sha256")))
    if (
        archive.bytes != 409_526_925
        or contest.get("n_samples") != 600
        or contest.get("archive_size_bytes") != archive.bytes
        or provenance.get("archive_sha256") != archive.sha256
    ):
        raise DirectDescriptionError("C1 contest receipt/archive lineage mismatch")
    runtime = provenance.get("inflate_runtime_manifest")
    if not isinstance(runtime, Mapping) or not isinstance(runtime.get("upstream_evaluate_py"), Mapping):
        raise DirectDescriptionError("C1 provenance lacks upstream evaluate.py custody")
    evaluate_sha = _require_sha256(runtime["upstream_evaluate_py"].get("sha256"), "upstream evaluate sha")
    source_cache_row = prepare.get("source_cache")
    if not isinstance(source_cache_row, Mapping):
        raise DirectDescriptionError("C1 prepare receipt lacks source-cache custody")
    source_cache = _source_file(Path(str(source_cache_row.get("path"))), str(source_cache_row.get("sha256")))
    if source_cache.bytes != source_cache_row.get("bytes"):
        raise DirectDescriptionError("C1 source-cache byte count mismatch")
    producer_git_sha = str(prepare.get("git_sha"))
    producer_path = "tools/measure_v10_two_plane_receiver_timing.py"
    producer_source = _git_output(("show", f"{producer_git_sha}:{producer_path}"), Path(repo_root))
    materializer_git_sha = _git_output(("rev-parse", "HEAD"), Path(repo_root)).decode().strip()
    upstream_sha = compute_upstream_snapshot_sha256(Path(upstream_repo_root))
    if upstream_sha is None:
        raise DirectDescriptionError("upstream snapshot is unavailable")

    chunks_root = Path(prepare_receipt_path).parent / "prepare_chunks"
    manifest_hashes = prepare.get("prepare_chunk_manifest_sha256")
    if not isinstance(manifest_hashes, list) or len(manifest_hashes) != 50:
        raise DirectDescriptionError("C1 prepare receipt chunk manifest inventory mismatch")
    chunks: list[TargetChunkV1] = []
    aggregate_y0 = hashlib.sha256()
    aggregate_y1 = hashlib.sha256()
    subset_y0_parts: list[np.ndarray] = []
    subset_y1_parts: list[np.ndarray] = []
    for chunk_index in range(50):
        stem = chunks_root / f"chunk-{chunk_index:04d}"
        manifest_path = Path(f"{stem}.manifest.json")
        manifest_file = _source_file(manifest_path, str(manifest_hashes[chunk_index]))
        manifest, _payload = _read_json(manifest_path, "C1 chunk manifest")
        pair_ids = tuple(range(chunk_index * 12, chunk_index * 12 + 12))
        if (
            manifest.get("schema") != "v10_two_plane_receiver_prepare_chunk.v1"
            or manifest.get("complete") is not True
            or tuple(manifest.get("pair_ids", ())) != pair_ids
            or manifest.get("source_cache_sha256") != source_cache.sha256
        ):
            raise DirectDescriptionError(f"C1 chunk {chunk_index} manifest custody mismatch")
        y0 = _source_file(Path(f"{stem}.y0.bin"), str(manifest.get("y0_sha256")))
        y1 = _source_file(Path(f"{stem}.y1.bin"), str(manifest.get("y1_sha256")))
        expected_bytes = 12 * 384 * 512 * 3
        if y0.bytes != expected_bytes or y1.bytes != expected_bytes:
            raise DirectDescriptionError(f"C1 chunk {chunk_index} plane byte count mismatch")
        y0_payload = _read_regular_file_once(Path(y0.path))
        y1_payload = _read_regular_file_once(Path(y1.path))
        aggregate_y0.update(y0_payload)
        aggregate_y1.update(y1_payload)
        if chunk_index < 6:
            subset_y0_parts.append(np.frombuffer(y0_payload, dtype=np.uint8).reshape(12, 384, 512, 3))
            subset_y1_parts.append(np.frombuffer(y1_payload, dtype=np.uint8).reshape(12, 384, 512, 3))
        chunks.append(
            TargetChunkV1(
                chunk_index=chunk_index,
                pair_ids=pair_ids,
                manifest=manifest_file,
                y0=y0,
                y1=y1,
            )
        )
    if aggregate_y0.hexdigest() != prepare.get("y0_sha256") or aggregate_y1.hexdigest() != prepare.get("y1_sha256"):
        raise DirectDescriptionError("C1 aggregate solved-plane SHA-256 mismatch")
    subset_y0 = np.ascontiguousarray(np.concatenate(subset_y0_parts, axis=0)[:PAIR_COUNT])
    subset_y1 = np.ascontiguousarray(np.concatenate(subset_y1_parts, axis=0)[:PAIR_COUNT])
    projection = np.stack((_block_mean_projection(subset_y0), _block_mean_projection(subset_y1)), axis=1)
    poses = _load_pose_source(Path(source_cache.path))
    pose_codes = _pose6_ordinal_codes(poses)
    return DirectDescriptionTargetPlaneReceiptV1(
        producer_path=producer_path,
        producer_git_sha=producer_git_sha,
        producer_source_sha256=_sha256(producer_source),
        materializer_git_sha=materializer_git_sha,
        upstream_repo_root=str(Path(upstream_repo_root).resolve()),
        upstream_snapshot_sha256=upstream_sha,
        upstream_evaluate_sha256=evaluate_sha,
        prepare_receipt=TargetSourceFileV1(
            path=str(Path(prepare_receipt_path).resolve()), bytes=len(prepare_payload), sha256=_sha256(prepare_payload)
        ),
        contest_cpu_receipt=TargetSourceFileV1(
            path=str(Path(contest_cpu_receipt_path).resolve()),
            bytes=len(contest_payload),
            sha256=_sha256(contest_payload),
        ),
        contest_cpu_provenance=TargetSourceFileV1(
            path=str(Path(contest_cpu_provenance_path).resolve()),
            bytes=len(provenance_payload),
            sha256=_sha256(provenance_payload),
        ),
        archive=archive,
        source_cache=source_cache,
        y0_sha256=aggregate_y0.hexdigest(),
        y1_sha256=aggregate_y1.hexdigest(),
        chunk_tree_sha256=_require_sha256(prepare.get("prepare_chunk_tree_sha256"), "chunk tree sha"),
        chunks=tuple(chunks),
        subset_pair_ids=tuple(range(PAIR_COUNT)),
        subset_y0_sha256=_sha256(subset_y0.tobytes(order="C")),
        subset_y1_sha256=_sha256(subset_y1.tobytes(order="C")),
        subset_projection_sha256=_sha256(projection.tobytes(order="C")),
        pose6_source_sha256=_sha256(poses.tobytes(order="C")),
        subset_pose6_source_sha256=_sha256(poses[:PAIR_COUNT].tobytes(order="C")),
        subset_pose6_target_codes_sha256=_sha256(pose_codes[:PAIR_COUNT].tobytes(order="C")),
    )


def write_or_verify_target_receipt(path: Path, receipt: DirectDescriptionTargetPlaneReceiptV1) -> Path:
    payload = rfc8785_canonicalize(receipt.model_dump(mode="json", by_alias=True)) + b"\n"
    target = Path(path)
    if target.exists():
        if _read_regular_file_once(target) != payload:
            raise DirectDescriptionError("existing target receipt differs from deterministic reconstruction")
        return target
    return _publish_new_bytes(target, payload)


@dataclass(frozen=True, slots=True)
class RealTargetSubsetV1:
    receipt: DirectDescriptionTargetPlaneReceiptV1
    receipt_path: Path
    receipt_sha256: str
    projection: np.ndarray
    pose6_codes: np.ndarray


def _load_receipt_file(path: Path, expected_sha256: str) -> DirectDescriptionTargetPlaneReceiptV1:
    payload = _read_regular_file_once(Path(path))
    if _sha256(payload) != _require_sha256(expected_sha256, "target receipt sha256"):
        raise DirectDescriptionError("real-target receipt SHA-256 mismatch")
    if not payload.endswith(b"\n"):
        raise DirectDescriptionError("real-target receipt lacks exactly one canonical LF")
    try:
        receipt = DirectDescriptionTargetPlaneReceiptV1.model_validate_json(payload)
    except ValueError as exc:
        raise DirectDescriptionError("real-target receipt schema is invalid") from exc
    if rfc8785_canonicalize(receipt.model_dump(mode="json", by_alias=True)) + b"\n" != payload:
        raise DirectDescriptionError("real-target receipt is not canonical JCS plus one LF")
    return receipt


def load_real_target_subset(path: Path, expected_sha256: str) -> RealTargetSubsetV1:
    """Fail closed on every source hash before exposing the rung target."""

    receipt = _load_receipt_file(path, expected_sha256)
    for source in (
        receipt.prepare_receipt,
        receipt.contest_cpu_receipt,
        receipt.contest_cpu_provenance,
        receipt.archive,
        receipt.source_cache,
    ):
        observed = _source_file(Path(source.path), source.sha256)
        if observed.bytes != source.bytes:
            raise DirectDescriptionError("real-target source byte count mismatch")
    if compute_upstream_snapshot_sha256(receipt.upstream_repo_root) != receipt.upstream_snapshot_sha256:
        raise DirectDescriptionError("real-target upstream snapshot custody mismatch")
    y0_parts: list[np.ndarray] = []
    y1_parts: list[np.ndarray] = []
    aggregate_y0 = hashlib.sha256()
    aggregate_y1 = hashlib.sha256()
    for row in receipt.chunks:
        for source in (row.manifest, row.y0, row.y1):
            observed = _source_file(Path(source.path), source.sha256)
            if observed.bytes != source.bytes:
                raise DirectDescriptionError("real-target chunk source byte count mismatch")
        y0_payload = _read_regular_file_once(Path(row.y0.path))
        y1_payload = _read_regular_file_once(Path(row.y1.path))
        aggregate_y0.update(y0_payload)
        aggregate_y1.update(y1_payload)
        if row.chunk_index < 6:
            y0_parts.append(np.frombuffer(y0_payload, dtype=np.uint8).reshape(12, 384, 512, 3))
            y1_parts.append(np.frombuffer(y1_payload, dtype=np.uint8).reshape(12, 384, 512, 3))
    if aggregate_y0.hexdigest() != receipt.y0_sha256 or aggregate_y1.hexdigest() != receipt.y1_sha256:
        raise DirectDescriptionError("real-target aggregate plane custody mismatch")
    y0 = np.ascontiguousarray(np.concatenate(y0_parts, axis=0)[:PAIR_COUNT])
    y1 = np.ascontiguousarray(np.concatenate(y1_parts, axis=0)[:PAIR_COUNT])
    if (
        _sha256(y0.tobytes(order="C")) != receipt.subset_y0_sha256
        or _sha256(y1.tobytes(order="C")) != receipt.subset_y1_sha256
    ):
        raise DirectDescriptionError("real-target subset plane custody mismatch")
    projection = np.stack((_block_mean_projection(y0), _block_mean_projection(y1)), axis=1)
    if (
        projection.shape != TARGET_OUTPUT_SHAPE
        or _sha256(projection.tobytes(order="C")) != receipt.subset_projection_sha256
    ):
        raise DirectDescriptionError("real-target projection custody mismatch")
    poses = _load_pose_source(Path(receipt.source_cache.path))
    if _sha256(poses.tobytes(order="C")) != receipt.pose6_source_sha256:
        raise DirectDescriptionError("real-target Pose6 source custody mismatch")
    codes = np.ascontiguousarray(_pose6_ordinal_codes(poses)[:PAIR_COUNT])
    if (
        _sha256(poses[:PAIR_COUNT].tobytes(order="C")) != receipt.subset_pose6_source_sha256
        or _sha256(codes.tobytes(order="C")) != receipt.subset_pose6_target_codes_sha256
    ):
        raise DirectDescriptionError("real-target Pose6 target-code custody mismatch")
    return RealTargetSubsetV1(receipt, Path(path), expected_sha256, projection, codes)


class DirectDescriptionRealTargetRung0ConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionRealTargetRung0ConfigV1"] = Field(
        default="DirectDescriptionRealTargetRung0ConfigV1", alias="schema", serialization_alias="schema"
    )
    run_id: Literal["ddm_real_target_pose_rung0_n64_seed1234"] = "ddm_real_target_pose_rung0_n64_seed1234"
    seed: Literal[1234] = SEED
    n_pairs: Literal[64] = PAIR_COUNT
    receiver: Literal["numpy_integer_uint8_reference.v2"] = "numpy_integer_uint8_reference.v2"
    objective: Literal["real_target_integer_plane_plus_pose6_rank_debt.v3_family"] = (
        "real_target_integer_plane_plus_pose6_rank_debt.v3_family"
    )
    target_receipt_path: StrictStr
    target_receipt_sha256: StrictStr
    evidence_axis: Literal["[macOS-CPU real-target subset n64 apparatus]"] = EVIDENCE_AXIS
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    checkpoint_policy: Literal["atomic_preserve_every_stage"] = "atomic_preserve_every_stage"
    stages: tuple[DirectDescriptionSearchStageV1, ...] = (
        DirectDescriptionSearchStageV1(
            name="real_cells_rung0",
            objective_order="cells_first",
            stream_names=("static_ground_coefficients", "sparse_events", "entropy_state", "exceptions"),
            max_coordinate_steps=12,
        ),
        DirectDescriptionSearchStageV1(
            name="real_pose6_rung0",
            objective_order="pose_first",
            stream_names=("pose6_dxi_residuals",),
            max_coordinate_steps=12,
        ),
        DirectDescriptionSearchStageV1(
            name="real_xi_joint_rung0",
            objective_order="joint_integer_debt",
            stream_names=("xi_curve_knots",),
            max_coordinate_steps=12,
        ),
    )

    @model_validator(mode="after")
    def _valid(self) -> DirectDescriptionRealTargetRung0ConfigV1:
        _require_sha256(self.target_receipt_sha256, "target_receipt_sha256")
        if {name for stage in self.stages for name in stage.stream_names} != set(_V2_STREAM_ORDER):
            raise ValueError("real-target stage plan must cover every counted stream")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))

    def dsl_compile_hash(self) -> str:
        return _sha256(
            rfc8785_canonicalize(
                {
                    "compile_target": "direct_description_real_target_pose_rung0.v1",
                    "typed_config": self.model_dump(mode="json", by_alias=True),
                }
            )
        )


class DirectDescriptionRealTargetProgramV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    config_path: StrictStr
    output_directory: StrictStr

    def compile_consumer_argv(self) -> tuple[str, ...]:
        return (
            "/usr/bin/env",
            "python3",
            "tools/run_direct_description_real_target_rung0.py",
            "--config",
            self.config_path,
            "--output-dir",
            self.output_directory,
            "--execution-allowed",
            "false",
        )


def _initial_description(config: DirectDescriptionRealTargetRung0ConfigV1) -> DirectDescriptionZV2:
    values = {
        name: CountedDescriptionStreamV2(
            payload=_seeded_stream_bytes(config.seed, f"real-target-rung0:{name}", _V2_BODY_BYTES[name])
        )
        for name in _V2_STREAM_ORDER
    }
    return DirectDescriptionZV2(**values)


def _objective(
    receiver: DirectDescriptionReceiverResultV2,
    target: RealTargetSubsetV1,
) -> dict[str, int]:
    if receiver.output.shape != TARGET_OUTPUT_SHAPE or receiver.output.dtype != np.uint8:
        raise DirectDescriptionError("real-target objective receiver geometry mismatch")
    plane_debt = int(np.abs(receiver.output.astype(np.int16) - target.projection.astype(np.int16)).sum(dtype=np.int64))
    pose_values = np.frombuffer(receiver.z.pose6_dxi_residuals.payload, dtype=np.uint8).reshape(PAIR_COUNT, 6)
    pose_debt = int(np.abs(pose_values.astype(np.int16) - target.pose6_codes.astype(np.int16)).sum(dtype=np.int64))
    return {
        "plane_integer_l1_debt": plane_debt,
        "pose6_integer_l1_debt": pose_debt,
        "joint_integer_debt": plane_debt + pose_debt,
        "archive_bytes": len(receiver.archive),
    }


def _objective_key(value: Mapping[str, int], order: str) -> tuple[int, ...]:
    plane = _require_exact_nonnegative_int(value.get("plane_integer_l1_debt"), "plane_integer_l1_debt")
    pose = _require_exact_nonnegative_int(value.get("pose6_integer_l1_debt"), "pose6_integer_l1_debt")
    rate = _require_exact_nonnegative_int(value.get("archive_bytes"), "archive_bytes")
    if order == "cells_first":
        return (plane, pose, rate)
    if order == "pose_first":
        return (pose, plane, rate)
    if order == "joint_integer_debt":
        return (plane + pose, plane, pose, rate)
    raise DirectDescriptionError(f"unknown real-target objective order {order!r}")


class DirectDescriptionRealTargetCheckpointV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DirectDescriptionRealTargetCheckpointV1"] = Field(
        default=CHECKPOINT_SCHEMA, alias="schema", serialization_alias="schema"
    )
    config: dict[str, Any]
    config_sha256: StrictStr
    dsl_compile_hash: StrictStr
    semantic_argv: tuple[StrictStr, ...]
    semantic_argv_sha256: StrictStr
    target_receipt_path: StrictStr
    target_receipt_sha256: StrictStr
    target_projection_sha256: StrictStr
    target_pose6_codes_sha256: StrictStr
    completed_stage_index: StrictInt = Field(ge=0)
    completed_stage_name: StrictStr
    next_stage_index: StrictInt = Field(ge=0)
    global_step: StrictInt = Field(ge=0)
    current_archive_b64: StrictStr
    current_archive_sha256: StrictStr
    current_archive_bytes: StrictInt = Field(ge=1)
    current_output_sha256: StrictStr
    objective: dict[str, StrictInt]
    optimizer_state: dict[str, Any]
    stage_history: tuple[dict[str, Any], ...]
    evidence_axis: Literal["[macOS-CPU real-target subset n64 apparatus]"] = EVIDENCE_AXIS
    research_only: Literal[True] = True
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DirectDescriptionRealTargetCheckpointV1:
        for field in (
            "config_sha256",
            "dsl_compile_hash",
            "semantic_argv_sha256",
            "target_receipt_sha256",
            "target_projection_sha256",
            "target_pose6_codes_sha256",
            "current_archive_sha256",
            "current_output_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        config = DirectDescriptionRealTargetRung0ConfigV1.model_validate_json(rfc8785_canonicalize(self.config))
        if config.typed_config_hash() != self.config_sha256 or config.dsl_compile_hash() != self.dsl_compile_hash:
            raise ValueError("real-target checkpoint config/compile identity mismatch")
        if self.next_stage_index != self.completed_stage_index + 1 or self.next_stage_index > len(config.stages):
            raise ValueError("real-target checkpoint continuation cursor mismatch")
        if config.stages[self.completed_stage_index].name != self.completed_stage_name:
            raise ValueError("real-target checkpoint stage name mismatch")
        if _sha256("\0".join(self.semantic_argv).encode()) != self.semantic_argv_sha256:
            raise ValueError("real-target checkpoint argv hash mismatch")
        try:
            archive = base64.b64decode(self.current_archive_b64, validate=True)
        except (TypeError, ValueError) as exc:
            raise ValueError("real-target checkpoint archive base64 malformed") from exc
        if (
            base64.b64encode(archive).decode() != self.current_archive_b64
            or len(archive) != self.current_archive_bytes
            or _sha256(archive) != self.current_archive_sha256
        ):
            raise ValueError("real-target checkpoint archive custody mismatch")
        receiver = receive_direct_description_archive_v2(archive)
        if receiver.output_sha256 != self.current_output_sha256:
            raise ValueError("real-target checkpoint receiver hash mismatch")
        rfc8785_canonicalize(self.objective)
        rfc8785_canonicalize(self.optimizer_state)
        rfc8785_canonicalize(list(self.stage_history))
        return self

    def to_bytes(self) -> bytes:
        body = self.model_dump(mode="json", by_alias=True)
        return rfc8785_canonicalize({"body": body, "body_sha256": _sha256(rfc8785_canonicalize(body))})

    @classmethod
    def from_bytes(cls, payload: bytes) -> DirectDescriptionRealTargetCheckpointV1:
        value = json.loads(payload)
        if (
            not isinstance(value, dict)
            or set(value) != {"body", "body_sha256"}
            or rfc8785_canonicalize(value) != payload
        ):
            raise DirectDescriptionError("real-target checkpoint envelope/canonical bytes mismatch")
        if _sha256(rfc8785_canonicalize(value["body"])) != _require_sha256(value["body_sha256"], "body_sha256"):
            raise DirectDescriptionError("real-target checkpoint body hash mismatch")
        return cls.model_validate_json(rfc8785_canonicalize(value["body"]))

    def filename(self) -> str:
        return (
            f"ddm_real_target_pose_rung0__stage{self.completed_stage_index:03d}_"
            f"{self.completed_stage_name}_step{self.global_step:012d}.json"
        )

    def write_new(self, directory: Path) -> Path:
        return _publish_new_bytes(Path(directory) / self.filename(), self.to_bytes())


def load_real_target_checkpoint(
    path: Path,
    *,
    config: DirectDescriptionRealTargetRung0ConfigV1,
    semantic_argv: Sequence[str],
    target: RealTargetSubsetV1,
) -> DirectDescriptionRealTargetCheckpointV1:
    checkpoint = DirectDescriptionRealTargetCheckpointV1.from_bytes(_read_regular_file_once(path))
    if (
        checkpoint.config_sha256 != config.typed_config_hash()
        or checkpoint.dsl_compile_hash != config.dsl_compile_hash()
        or checkpoint.semantic_argv != tuple(semantic_argv)
        or checkpoint.target_receipt_sha256 != target.receipt_sha256
        or checkpoint.target_projection_sha256 != target.receipt.subset_projection_sha256
        or checkpoint.target_pose6_codes_sha256 != target.receipt.subset_pose6_target_codes_sha256
    ):
        raise DirectDescriptionError("real-target resume identity differs from the governed run")
    receiver = receive_direct_description_archive_v2(base64.b64decode(checkpoint.current_archive_b64, validate=True))
    if _objective(receiver, target) != dict(checkpoint.objective):
        raise DirectDescriptionError("real-target checkpoint objective does not rederive")
    return checkpoint


@dataclass(frozen=True, slots=True)
class RealTargetRunResultV1:
    final_receiver: DirectDescriptionReceiverResultV2
    objective: Mapping[str, int]
    stage_history: tuple[Mapping[str, Any], ...]
    checkpoint_paths: tuple[Path, ...]
    complete: bool


def run_real_target_optimizer(
    config: DirectDescriptionRealTargetRung0ConfigV1,
    *,
    checkpoint_directory: Path,
    semantic_argv: Sequence[str],
    resume_from: Path | None = None,
    stop_after_stage_index: int | None = None,
    loaded_target: RealTargetSubsetV1 | None = None,
) -> RealTargetRunResultV1:
    target = loaded_target or load_real_target_subset(Path(config.target_receipt_path), config.target_receipt_sha256)
    if target.receipt_path != Path(config.target_receipt_path) or target.receipt_sha256 != config.target_receipt_sha256:
        raise DirectDescriptionError("loaded real target differs from the typed config")
    argv = tuple(semantic_argv)
    if not argv:
        raise DirectDescriptionError("real-target optimizer requires typed semantic argv")
    history: list[dict[str, Any]] = []
    checkpoint_paths: list[Path] = []
    global_step = 0
    start_stage = 0
    if resume_from is None:
        receiver = receive_direct_description_archive_v2(
            compile_direct_description_archive_v2(_initial_description(config)).archive
        )
    else:
        checkpoint = load_real_target_checkpoint(resume_from, config=config, semantic_argv=argv, target=target)
        receiver = receive_direct_description_archive_v2(
            base64.b64decode(checkpoint.current_archive_b64, validate=True)
        )
        history = [dict(row) for row in checkpoint.stage_history]
        global_step = checkpoint.global_step
        start_stage = checkpoint.next_stage_index
    current_z = receiver.z
    objective = _objective(receiver, target)
    for stage_index in range(start_stage, len(config.stages)):
        stage = config.stages[stage_index]
        before = dict(objective)
        plane_ceiling = before["plane_integer_l1_debt"]
        accepted = 0
        rejected = 0
        coordinates = _stage_coordinates(stage)
        for stream_name, coordinate in coordinates:
            value = getattr(current_z, stream_name).payload[coordinate]
            proposals = tuple(candidate for candidate in (value - 1, value + 1) if 0 <= candidate <= 255)
            best = (current_z, receiver, objective, _objective_key(objective, stage.objective_order))
            for candidate_value in proposals:
                proposal_z = current_z.replace_stream_byte(stream_name, coordinate, candidate_value)
                proposal_receiver = receive_direct_description_archive_v2(
                    compile_direct_description_archive_v2(proposal_z).archive
                )
                proposal_objective = _objective(proposal_receiver, target)
                global_step += 1
                if (
                    stage.objective_order == "pose_first"
                    and proposal_objective["plane_integer_l1_debt"] > plane_ceiling
                ):
                    rejected += 1
                    continue
                proposal_key = _objective_key(proposal_objective, stage.objective_order)
                if proposal_key < best[3]:
                    best = (proposal_z, proposal_receiver, proposal_objective, proposal_key)
                else:
                    rejected += 1
            if best[0] is not current_z:
                current_z, receiver, objective = best[:3]
                accepted += 1
        after = dict(objective)
        row = {
            "stage_index": stage_index,
            "stage_name": stage.name,
            "stage_role": stage.role,
            "objective_order": stage.objective_order,
            "coordinates_searched": len(coordinates),
            "accepted_coordinate_updates": accepted,
            "rejected_candidate_proposals": rejected,
            "objective_before": before,
            "objective_after": after,
            "strict_descent": _objective_key(after, stage.objective_order)
            < _objective_key(before, stage.objective_order),
            "pose_stage_plane_ceiling_preserved": stage.objective_order != "pose_first"
            or after["plane_integer_l1_debt"] <= plane_ceiling,
        }
        history.append(row)
        checkpoint = DirectDescriptionRealTargetCheckpointV1(
            config=config.model_dump(mode="json", by_alias=True),
            config_sha256=config.typed_config_hash(),
            dsl_compile_hash=config.dsl_compile_hash(),
            semantic_argv=argv,
            semantic_argv_sha256=_sha256("\0".join(argv).encode()),
            target_receipt_path=config.target_receipt_path,
            target_receipt_sha256=config.target_receipt_sha256,
            target_projection_sha256=target.receipt.subset_projection_sha256,
            target_pose6_codes_sha256=target.receipt.subset_pose6_target_codes_sha256,
            completed_stage_index=stage_index,
            completed_stage_name=stage.name,
            next_stage_index=stage_index + 1,
            global_step=global_step,
            current_archive_b64=base64.b64encode(receiver.archive).decode(),
            current_archive_sha256=_sha256(receiver.archive),
            current_archive_bytes=len(receiver.archive),
            current_output_sha256=receiver.output_sha256,
            objective=after,
            optimizer_state={
                "algorithm": "deterministic_plus_minus_one_coordinate_descent",
                "target_mode": "sha_bound_existing_c1_solved_planes",
                "pose6_stream_term_active": True,
                "candidate_evaluations": global_step,
                "next_stage_index": stage_index + 1,
            },
            stage_history=tuple(history),
        )
        checkpoint_paths.append(checkpoint.write_new(checkpoint_directory))
        if stop_after_stage_index is not None and stage_index >= stop_after_stage_index:
            break
    return RealTargetRunResultV1(
        final_receiver=receiver,
        objective=dict(objective),
        stage_history=tuple(history),
        checkpoint_paths=tuple(checkpoint_paths),
        complete=len(history) == len(config.stages),
    )


def _checkpoint_hashes(paths: Sequence[Path]) -> list[str]:
    return [_sha256(_read_regular_file_once(path)) for path in paths]


def run_real_target_pose_rung_zero(
    config: DirectDescriptionRealTargetRung0ConfigV1,
    *,
    output_directory: Path,
    semantic_argv: Sequence[str],
) -> tuple[dict[str, Any], Path]:
    root = Path(output_directory)
    root.mkdir(parents=True, exist_ok=True)
    producer = {
        "module": _committed_source_custody("src/tac/optimization/direct_description_real_target_rung0.py"),
        "cli": _committed_source_custody("tools/run_direct_description_real_target_rung0.py"),
    }
    target = load_real_target_subset(Path(config.target_receipt_path), config.target_receipt_sha256)
    primary = run_real_target_optimizer(
        config,
        checkpoint_directory=root / "primary" / "checkpoints",
        semantic_argv=semantic_argv,
        loaded_target=target,
    )
    partial = run_real_target_optimizer(
        config,
        checkpoint_directory=root / "resume" / "checkpoints",
        semantic_argv=semantic_argv,
        stop_after_stage_index=0,
        loaded_target=target,
    )
    if partial.complete or len(partial.checkpoint_paths) != 1:
        raise DirectDescriptionError("real-target resume control did not stop after rung stage zero")
    resumed = run_real_target_optimizer(
        config,
        checkpoint_directory=root / "resume" / "checkpoints",
        semantic_argv=semantic_argv,
        resume_from=partial.checkpoint_paths[-1],
        loaded_target=target,
    )
    if not primary.complete or not resumed.complete:
        raise DirectDescriptionError("real-target rung did not preserve every stage")
    if (
        primary.final_receiver.archive != resumed.final_receiver.archive
        or primary.final_receiver.output_sha256 != resumed.final_receiver.output_sha256
        or dict(primary.objective) != dict(resumed.objective)
    ):
        raise DirectDescriptionError("real-target resumed result is not bit-identical")
    parsed = parse_direct_description_archive_v2(primary.final_receiver.archive)
    if compile_direct_description_archive_v2(parsed.z).archive != primary.final_receiver.archive:
        raise DirectDescriptionError("real-target terminal archive parse/re-encode mismatch")
    final_archive = _publish_new_bytes(
        root / "ddm_real_target_pose_rung0_final.not_a_candidate.zip", primary.final_receiver.archive
    )
    receipt = {
        "schema": RUNG_SCHEMA,
        "task": 603,
        "run_id": config.run_id,
        "seed": config.seed,
        "n_pairs": PAIR_COUNT,
        "subset_pair_ids": list(range(PAIR_COUNT)),
        "evidence_axis": EVIDENCE_AXIS,
        "verdict_scope": "bounded n64 real-target apparatus rung only; not n600 evidence and not a finding",
        "research_only": True,
        "execution_allowed": False,
        "candidate_archive": False,
        "score_claim": False,
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "dsl_compile_hash": config.dsl_compile_hash(),
        "semantic_argv": list(semantic_argv),
        "producer": producer,
        "target_receipt_path": str(Path(config.target_receipt_path)),
        "target_receipt_sha256": config.target_receipt_sha256,
        "target_source": {
            "source_role": target.receipt.source_role,
            "full_pairs": target.receipt.pairs,
            "subset_pairs": PAIR_COUNT,
            "y0_sha256": target.receipt.y0_sha256,
            "y1_sha256": target.receipt.y1_sha256,
            "subset_projection_sha256": target.receipt.subset_projection_sha256,
            "subset_projection_recipe": target.receipt.subset_projection_recipe,
            "pose6_source_sha256": target.receipt.pose6_source_sha256,
            "pose6_target_codes_sha256": target.receipt.subset_pose6_target_codes_sha256,
            "pose6_target_recipe": target.receipt.pose6_target_recipe,
        },
        "optimizer": {
            "stage_labels": [row["stage_role"] for row in primary.stage_history],
            "pose6_stream_term_active": True,
            "target_mode": "sha_bound_existing_c1_solved_planes",
            "objective_initial": primary.stage_history[0]["objective_before"],
            "objective_final": dict(primary.objective),
            "trajectory": [dict(row) for row in primary.stage_history],
        },
        "resume": {
            "resumed_from_stage": 0,
            "terminal_archive_bit_identical": True,
            "terminal_receiver_bit_identical": True,
            "terminal_objective_identical": True,
            "all_stage_checkpoints_preserved": True,
            "primary_checkpoint_sha256": _checkpoint_hashes(primary.checkpoint_paths),
            "resume_checkpoint_sha256": _checkpoint_hashes((*partial.checkpoint_paths, *resumed.checkpoint_paths)),
        },
        "archive": {
            "path": str(final_archive),
            "bytes": len(primary.final_receiver.archive),
            "sha256": _sha256(primary.final_receiver.archive),
            "candidate_role": "not_a_candidate",
            "parse_reencode_identical": True,
        },
        "cleanup": {
            "bulk_artifacts_created": False,
            "target_bulk_remains_read_only_on_ssd": True,
            "scratch_policy": "small immutable checkpoints only",
        },
    }
    receipt_payload = rfc8785_canonicalize(receipt) + b"\n"
    receipt_path = _publish_new_bytes(root / "ddm_real_target_pose_rung0_receipt.json", receipt_payload)
    return receipt, receipt_path


__all__ = [
    "DirectDescriptionRealTargetProgramV1",
    "DirectDescriptionRealTargetRung0ConfigV1",
    "DirectDescriptionTargetPlaneReceiptV1",
    "build_target_plane_receipt",
    "load_real_target_subset",
    "run_real_target_optimizer",
    "run_real_target_pose_rung_zero",
    "write_or_verify_target_receipt",
]
