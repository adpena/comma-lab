#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare, time, and compose the V10 C1 two-plane receiver control.

The decoder stays scorer-free.  This encode-side tool owns source-plane
materialization, immutable chunk custody, local timing orchestration, the
native-f32 spot check, and false-authority research-signal reports.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.storage_tiers import (  # noqa: E402
    StorageTierError,
    bytes_from_gib,
    default_storage_tiers,
    plan_experiment_storage,
)
from tac.codec.v10_predictor_residual import (  # noqa: E402
    CONTENT_CODEC_TAG,
    SPATIAL_SMOOTH_121_ID,
    decode_predictor_residual,
    encode_predictor_residual,
)
from tac.codec.v10_predictor_residual import (  # noqa: E402
    MAGIC as PREDICTOR_MAGIC,
)
from tac.codec.v10_predictor_residual import (  # noqa: E402
    PREFIX as PREDICTOR_PREFIX,
)
from tac.codec.v10_predictor_residual import (  # noqa: E402
    VERSION as PREDICTOR_VERSION,
)
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator  # noqa: E402
from tac.witness_dsl.v10_production_receiver import (  # noqa: E402
    ARITHMETIC_ID,
    DESCRIPTION_FRAME0_POLICY_ID,
    MEMBER_NAME,
    PACKET_SCHEMA,
    PREDICTOR_RESIDUAL_Y_CODEC_ID,
    RECEIVER_CONTRACT_ID,
    SECTION_LENGTH,
    TIE_POLICY_ID,
    ProductionSection,
    parse_packet,
)
from tac.witness_dsl.v10_production_receiver import (  # noqa: E402
    MAGIC as PRODUCTION_MAGIC,
)
from tac.witness_dsl.v10_production_receiver import (  # noqa: E402
    PREFIX as PRODUCTION_PREFIX,
)
from tac.witness_dsl.v10_production_receiver import (  # noqa: E402
    VERSION as PRODUCTION_VERSION,
)
from tac.witness_dsl.v10_two_plane_timing_receiver import (  # noqa: E402
    FULL_CAMERA_HW,
    FULL_NUMERATOR_VALUES,
    FULL_PAIR_COUNT,
    FULL_RAW_BYTES,
    FULL_SCORER_HW,
    TIMING_RECEIPT_SCHEMA,
    TwoPlaneTimingReceiverError,
    mlx_runtime_status,
    parity_check_mlx_two_plane,
    timed_inflate_two_plane_archive,
)
from tools.measure_uint8_lattice_feasibility import stored_npy_memmap  # noqa: E402

PREPARE_SCHEMA = "v10_two_plane_receiver_prepare.v1"
PREPARE_CHUNK_SCHEMA = "v10_two_plane_receiver_prepare_chunk.v1"
COMPOSE_SCHEMA = "v10_two_plane_receiver_composed_receipt.v1"
CALIBRATION_SCHEMA = "v10_two_plane_timing_calibration_anchor.v1"
MLX_TOOL_SCHEMA = "v10_two_plane_receiver_mlx_tool_receipt.v1"
MODAL_TICKET_SCHEMA = "v10_two_plane_full_evaluate_modal_ticket.v1"
TOOL_TIMING_SCHEMA = "v10_two_plane_receiver_tool_timing.v1"

CANONICAL_CACHE = Path("/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
CANONICAL_SACRED_ROOT = Path("/Users/adpena/Projects/pact/experiments/results/levelset_n600_witness_20260717T113932Z")
CANONICAL_WORK_ROOT = Path("/Volumes/VertigoDataTier/pact/evidence/c1_two_plane_receiver_20260719")
CANONICAL_WORKLOAD_SUBDIR = "evidence/c1_two_plane_receiver_20260719"
MINIMUM_STORAGE_BYTES = bytes_from_gib(24)
CHUNK_PAIRS = 12
HARD_ORACLE_PAIR_IDS = (90, 175, 277, 381, 424, 573)
EXPECTED_CACHE_SHA256 = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
EXPECTED_Y0_SHA256 = "5e86e419cdd5bd41c9482cabc78cf27cec22281098b64c715d91f1f067d11566"
EXPECTED_Y1_SHA256 = "6a731946e3d9de82089c90de9784c5a5bc72c607c963fb6f79dac16f00ac89bc"
EXPECTED_SEGNET_SHA256 = "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"
EXPECTED_POSENET_SHA256 = "0f3a0874c5c387f990d7b88bd1d7e1f6de35d98b45f2a289989db2c77b9b6576"
F32_LAW_ID = "f32_receiver_arithmetic_exactness_admissibility_v1"
LOCAL_CPU_AXIS = "[macOS-CPU local timing] NON-PROMOTABLE"
MLX_AXIS = "[macOS-MLX research-signal] NON-PROMOTABLE"
TIMING_COMPONENTS = (
    "parse_seconds",
    "expansion_seconds",
    "solve0_seconds",
    "solve1_seconds",
    "assembly_io_seconds",
    "verification_seconds",
)
CUDA_OUTPUT_BYTES = 3_662_409_600
CUDA_NUMERATOR_VALUES = 707_788_800
CUDA_UNIFORM_TAP_PRODUCTS = 2_831_155_200
CALLER_WALL_FIELD = "caller_wall_seconds_through_receiver_receipt_persistence"
HARD_ORACLE_INPUT_CONTRACT = {
    "complete_decoded_camera_frames_per_pair": 2,
    "frame_order": ["frame0", "frame1"],
    "segnet_path": "official frozen SegNet last-frame RGB path; frame1 is evaluated",
    "posenet_path": (
        "official frozen PoseNet two-frame path: complete frame0/frame1 RGB, bilinear resize, "
        "YUV6 per frame, concatenated 12-channel input"
    ),
}


class C1MeasurementError(RuntimeError):
    """Fail-closed preparation, composition, calibration, or custody error."""


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise C1MeasurementError("value cannot be represented as canonical JSON") from exc


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def _exact_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise C1MeasurementError(f"{label} must be an exact integer >= {minimum}")
    return value


def _exact_hw(value: Any, label: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise C1MeasurementError(f"{label} must contain exactly two integers")
    return (
        _exact_int(value[0], f"{label}[0]", minimum=1),
        _exact_int(value[1], f"{label}[1]", minimum=1),
    )


def _strict_descendant(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    resolved_root = root.expanduser().resolve(strict=False)
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise C1MeasurementError(f"{label} must resolve under the prepared SSD root") from exc
    if relative == Path("."):
        raise C1MeasurementError(f"{label} must be a strict descendant of the prepared SSD root")
    return resolved


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise C1MeasurementError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise C1MeasurementError(f"{label} must be positive and finite")
    return result


def _atomic_write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise C1MeasurementError(f"write-once path already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _write_once_or_equal(path: Path, payload: bytes) -> None:
    if path.exists():
        if not path.is_file() or path.read_bytes() != payload:
            raise C1MeasurementError(f"preserved bytes drifted: {path}")
        return
    try:
        _atomic_write_once(path, payload)
    except C1MeasurementError:
        if not path.is_file() or path.read_bytes() != payload:
            raise


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise C1MeasurementError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict) or _canonical_json(value) != payload:
        raise C1MeasurementError(f"{label} is not canonical JSON: {path}")
    return value


def _git_sha() -> str | None:
    run = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = run.stdout.strip()
    return value if run.returncode == 0 and len(value) == 40 else None


def _runtime_source_custody() -> Mapping[str, Any]:
    paths = {
        "measurement_tool": Path(__file__).resolve(),
        "timed_receiver": REPO_ROOT / "src/tac/witness_dsl/v10_two_plane_timing_receiver.py",
        "production_receiver": REPO_ROOT / "src/tac/witness_dsl/v10_production_receiver.py",
        "integer_solver": REPO_ROOT / "src/tac/optimization/uint8_lattice_feasibility.py",
    }
    rows: dict[str, Any] = {}
    git_sha = _git_sha()
    for label, path in paths.items():
        resolved = path.resolve()
        if not resolved.is_file():
            raise C1MeasurementError(f"runtime source custody is absent: {label}")
        try:
            relative = resolved.relative_to(REPO_ROOT.resolve()).as_posix()
        except ValueError as exc:
            raise C1MeasurementError(f"runtime source custody escaped the repository: {label}") from exc
        worktree_sha = _sha256_file(resolved)
        head = subprocess.run(
            ["git", "show", f"{git_sha}:{relative}"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
        )
        head_sha = _sha256_bytes(head.stdout) if head.returncode == 0 else None
        rows[label] = {
            "path": relative,
            "bytes": resolved.stat().st_size,
            "sha256": worktree_sha,
            "head_blob_sha256": head_sha,
            "head_blob_matches_worktree": head_sha == worktree_sha,
        }
    checkout_reproduces = git_sha is not None and all(row["head_blob_matches_worktree"] for row in rows.values())
    return {
        "git_sha": git_sha,
        "required_remote_checkout_git_sha": git_sha,
        "remote_checkout_reproduces_sources": checkout_reproduces,
        "remote_hash_revalidation_required": True,
        "sources": rows,
    }


def _tree_snapshot(root: Path) -> Mapping[str, Any]:
    digest = hashlib.sha256()
    entries = 0
    if not root.is_dir():
        return {"exists": False, "entries": 0, "metadata_sha256": None}
    for current, directories, files in os.walk(root, followlinks=False):
        directories.sort()
        files.sort()
        base = Path(current)
        for name in (*directories, *files):
            path = base / name
            stat = path.lstat()
            digest.update(
                f"{path.relative_to(root).as_posix()}\0{stat.st_mode}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode()
            )
            entries += 1
    return {"exists": True, "entries": entries, "metadata_sha256": digest.hexdigest()}


def _content_tree_sha256(root: Path) -> str:
    """Recompute the receiver's path/size/content tree digest."""

    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        size = path.stat().st_size
        digest.update(struct.pack(">I", len(relative)))
        digest.update(relative)
        digest.update(struct.pack(">Q", size))
        with path.open("rb") as handle:
            while block := handle.read(8 << 20):
                digest.update(block)
    return digest.hexdigest()


def _storage_gate(
    work_root: Path,
    *,
    requested_bytes: int,
    test_only_small_fixture: bool,
) -> Mapping[str, Any]:
    requested = _exact_int(requested_bytes, "requested storage bytes", minimum=1)
    resolved = work_root.expanduser().resolve(strict=False)
    if test_only_small_fixture:
        parent = resolved
        while not parent.exists() and parent.parent != parent:
            parent = parent.parent
        free = int(os.statvfs(parent).f_bavail * os.statvfs(parent).f_frsize)
        if free < requested:
            raise C1MeasurementError("test-only storage preflight lacks requested capacity")
        return {
            "schema": "v10_two_plane_test_only_storage_preflight.v1",
            "selected_workload_root": str(resolved),
            "selected_workload_root_matches_expected": True,
            "requested_bytes": requested,
            "free_bytes_at_check": free,
            "test_only_noncanonical_storage": True,
            "passed": True,
            "score_claim": False,
            "promotion_eligible": False,
        }
    if requested < MINIMUM_STORAGE_BYTES:
        raise C1MeasurementError("full C1 prepare requires at least 24 GiB requested")
    if resolved != CANONICAL_WORK_ROOT.resolve(strict=False):
        raise C1MeasurementError("full C1 prepare requires the exact canonical SSD evidence root")
    tiers = default_storage_tiers(repo_root=REPO_ROOT, reserve_free_gb=0.0, allow_local_disk=False)
    plan = plan_experiment_storage(
        tiers,
        workload_subdir=CANONICAL_WORKLOAD_SUBDIR,
        requested_bytes=requested,
        min_free_bytes=0,
        create=True,
    )
    payload = plan.to_dict()
    selected = payload.get("selected_workload_root")
    matches = isinstance(selected, str) and Path(selected).resolve(strict=False) == resolved
    payload["selected_workload_root_matches_expected"] = matches
    payload["expected_workload_root"] = str(resolved)
    payload["passed"] = bool(matches)
    if not matches:
        blockers = [
            f"{row['name']}:{','.join(row.get('blockers') or ['not_eligible'])}"
            for row in payload.get("tiers", [])
            if isinstance(row, dict)
        ]
        raise C1MeasurementError("canonical SSD storage preflight refused: " + "; ".join(blockers))
    return payload


def _validated_prepared_root(
    prepared: Mapping[str, Any],
    *,
    test_only_small_fixture: bool,
) -> Path:
    """Bind downstream work to the prepare mode and its selected storage root."""

    if type(test_only_small_fixture) is not bool:
        raise C1MeasurementError("test-only invocation flag must be boolean")
    prepared_test_only = prepared.get("test_only_small_fixture")
    if type(prepared_test_only) is not bool or prepared_test_only is not test_only_small_fixture:
        raise C1MeasurementError("prepare receipt test-only mode differs from this invocation")
    archive_dir = prepared.get("archive_dir")
    if not isinstance(archive_dir, str) or not archive_dir:
        raise C1MeasurementError("prepare receipt archive root is missing")
    root = Path(archive_dir).expanduser().resolve(strict=False)
    archive_path = prepared.get("archive_path")
    names_path = prepared.get("video_names_file")
    if (
        not isinstance(archive_path, str)
        or Path(archive_path).expanduser().resolve(strict=False) != root / "archive.zip"
    ):
        raise C1MeasurementError("prepare receipt archive path escaped its selected root")
    if (
        not isinstance(names_path, str)
        or Path(names_path).expanduser().resolve(strict=False) != root / "video_names.txt"
    ):
        raise C1MeasurementError("prepare receipt video-name path escaped its selected root")
    contest_dir = prepared.get("contest_archive_dir")
    contest_packet = prepared.get("contest_packet_path")
    adapter = prepared.get("contest_adapter_path")
    if (
        not isinstance(contest_dir, str)
        or Path(contest_dir).expanduser().resolve(strict=False) != root / "archive_input"
        or not isinstance(contest_packet, str)
        or Path(contest_packet).expanduser().resolve(strict=False) != root / "archive_input" / MEMBER_NAME
        or not isinstance(adapter, str)
        or Path(adapter).expanduser().resolve(strict=False) != root / "inflate.sh"
        or prepared.get("contest_adapter_bound") is not True
    ):
        raise C1MeasurementError("prepare receipt lacks its official three-argument adapter binding")
    packet_path = Path(contest_packet)
    adapter_path = Path(adapter)
    if (
        not packet_path.is_file()
        or packet_path.stat().st_size
        != _exact_int(prepared.get("contest_packet_bytes"), "prepared contest packet bytes", minimum=1)
        or _sha256_file(packet_path) != prepared.get("contest_packet_sha256")
        or prepared.get("contest_packet_sha256") != prepared.get("packet_sha256")
        or not adapter_path.is_file()
        or adapter_path.stat().st_size
        != _exact_int(prepared.get("contest_adapter_bytes"), "prepared contest adapter bytes", minimum=1)
        or _sha256_file(adapter_path) != prepared.get("contest_adapter_sha256")
        or prepared.get("contest_adapter_mode") != "0755"
        or adapter_path.stat().st_mode & 0o777 != 0o755
    ):
        raise C1MeasurementError("prepared official archive input or adapter custody drifted")
    adapter_workers = _exact_int(prepared.get("contest_adapter_workers"), "prepared contest adapter workers", minimum=1)
    if adapter_workers != (1 if test_only_small_fixture else 4):
        raise C1MeasurementError("prepared contest adapter worker count differs from its authority mode")
    storage = prepared.get("storage_preflight")
    if not isinstance(storage, Mapping) or storage.get("passed") is not True:
        raise C1MeasurementError("prepare receipt lacks a passed storage preflight")
    selected = storage.get("selected_workload_root")
    if not isinstance(selected, str) or Path(selected).expanduser().resolve(strict=False) != root:
        raise C1MeasurementError("prepare receipt storage root custody drifted")
    if test_only_small_fixture:
        if storage.get("test_only_noncanonical_storage") is not True:
            raise C1MeasurementError("test-only prepare receipt lacks its non-authority storage marker")
        return root

    if root != CANONICAL_WORK_ROOT.resolve(strict=False):
        raise C1MeasurementError("full prepare receipt is not bound to the exact canonical SSD root")
    if storage.get("selected_workload_root_matches_expected") is not True:
        raise C1MeasurementError("full prepare storage preflight did not select the canonical SSD root")
    if (
        _exact_int(storage.get("requested_bytes"), "prepared requested storage bytes", minimum=1)
        < MINIMUM_STORAGE_BYTES
    ):
        raise C1MeasurementError("full prepare receipt storage request is below 24 GiB")
    source_cache = prepared.get("source_cache")
    if (
        not isinstance(source_cache, Mapping)
        or source_cache.get("sha256") != EXPECTED_CACHE_SHA256
        or prepared.get("y0_sha256") != EXPECTED_Y0_SHA256
        or prepared.get("y1_sha256") != EXPECTED_Y1_SHA256
    ):
        raise C1MeasurementError("full prepare receipt differs from frozen cache/Y custody")
    sacred = prepared.get("sacred_donor_root")
    if not isinstance(sacred, str) or Path(sacred).expanduser().resolve(strict=False) != CANONICAL_SACRED_ROOT.resolve(
        strict=False
    ):
        raise C1MeasurementError("full prepare receipt sacred-donor path custody drifted")
    before = prepared.get("sacred_donor_snapshot_before")
    after = prepared.get("sacred_donor_snapshot_after")
    if not isinstance(before, Mapping) or before.get("exists") is not True or before != after:
        raise C1MeasurementError("full prepare receipt sacred-donor snapshots are not stable")
    return root


def _revalidate_sacred_donor(prepared: Mapping[str, Any], *, test_only_small_fixture: bool) -> Mapping[str, Any]:
    if test_only_small_fixture:
        return {"test_only_not_consulted": True}
    sacred_root = Path(str(prepared["sacred_donor_root"])).expanduser().resolve()
    current = _tree_snapshot(sacred_root)
    if current != prepared.get("sacred_donor_snapshot_before") or current != prepared.get(
        "sacred_donor_snapshot_after"
    ):
        raise C1MeasurementError("sacred donor metadata changed after prepare or during measurement")
    return current


def exact_operator_round_u8(operator: DisjointResizeOperator, frame: np.ndarray) -> np.ndarray:
    """Derive the exact integer resize plane and round nonnegative ties up."""

    numerators, denominator = operator.apply_numerators(frame)
    if denominator <= 0 or np.any(numerators < 0):
        raise C1MeasurementError("resize numerator/denominator escaped the uint8 domain")
    rounded = (numerators.astype(np.int64) + denominator // 2) // denominator
    if np.any(rounded > 255):
        raise C1MeasurementError("rounded scorer plane exceeds uint8")
    return np.ascontiguousarray(rounded.astype(np.uint8))


def _load_cache(
    cache_path: Path,
    *,
    pair_count: int,
    camera_hw: tuple[int, int],
    scorer_hw: tuple[int, int],
    expected_sha256: str | None,
) -> tuple[dict[str, np.memmap], str]:
    if not cache_path.is_file():
        raise C1MeasurementError(f"source cache is absent: {cache_path}")
    cache_sha = _sha256_file(cache_path)
    if expected_sha256 is not None and cache_sha != expected_sha256:
        raise C1MeasurementError("source cache SHA-256 differs from frozen custody")
    try:
        fields = {
            key: stored_npy_memmap(cache_path, key) for key in ("n_pairs", "gt_f0", "gt_f1", "lstars", "gt_poses")
        }
    except (KeyError, OSError, ValueError, zipfile.BadZipFile) as exc:
        raise C1MeasurementError("source cache must contain mmap-compatible stored NPY fields") from exc
    if int(np.asarray(fields["n_pairs"]).reshape(())) != pair_count:
        raise C1MeasurementError("source cache pair count drifted")
    expected_shapes = {
        "gt_f0": (pair_count, *camera_hw, 3),
        "gt_f1": (pair_count, *camera_hw, 3),
        "lstars": (pair_count, *scorer_hw),
        "gt_poses": (pair_count, 6),
    }
    for key, shape in expected_shapes.items():
        if fields[key].shape != shape:
            raise C1MeasurementError(f"source cache {key} shape drifted")
    if fields["gt_f0"].dtype != np.uint8 or fields["gt_f1"].dtype != np.uint8:
        raise C1MeasurementError("source camera frames must remain uint8")
    return fields, cache_sha


def _chunk_paths(work_root: Path, chunk_index: int) -> tuple[Path, Path, Path, Path]:
    base = work_root / "prepare_chunks" / f"chunk-{chunk_index:04d}"
    return (
        base.with_suffix(".y0.bin"),
        base.with_suffix(".y1.bin"),
        base.with_suffix(".predictor.bin"),
        base.with_suffix(".manifest.json"),
    )


def _chunk_manifest(
    *,
    cache_sha256: str,
    chunk_index: int,
    pair_ids: Sequence[int],
    camera_hw: tuple[int, int],
    scorer_hw: tuple[int, int],
    y0_bytes: bytes,
    y1_bytes: bytes,
    predictor_payload: bytes,
) -> Mapping[str, Any]:
    return {
        "schema": PREPARE_CHUNK_SCHEMA,
        "complete": True,
        "chunk_index": chunk_index,
        "pair_ids": list(pair_ids),
        "pair_count": len(pair_ids),
        "camera_hw": list(camera_hw),
        "scorer_hw": list(scorer_hw),
        "source_cache_sha256": cache_sha256,
        "y_codec_id": PREDICTOR_RESIDUAL_Y_CODEC_ID,
        "predictor_mode_id": SPATIAL_SMOOTH_121_ID,
        "frame0_policy_id": DESCRIPTION_FRAME0_POLICY_ID,
        "y0_bytes": len(y0_bytes),
        "y1_bytes": len(y1_bytes),
        "predictor_bytes": len(predictor_payload),
        "y0_sha256": _sha256_bytes(y0_bytes),
        "y1_sha256": _sha256_bytes(y1_bytes),
        "predictor_sha256": _sha256_bytes(predictor_payload),
        "score_claim": False,
        "promotion_eligible": False,
    }


def _validate_predictor_chunk(
    payload: bytes,
    *,
    pair_ids: Sequence[int],
    scorer_hw: tuple[int, int],
    expected_y0: bytes,
    expected_y1: bytes,
) -> None:
    try:
        decoded = decode_predictor_residual(payload)
    except Exception as exc:
        raise C1MeasurementError("predictor chunk parse-back refused") from exc
    shape = (len(pair_ids), *scorer_hw, 3)
    if decoded.pair_ids != tuple(pair_ids) or decoded.frame0.shape != shape or decoded.frame1.shape != shape:
        raise C1MeasurementError("predictor chunk IDs or geometry drifted")
    if decoded.modes != (SPATIAL_SMOOTH_121_ID,) * len(pair_ids):
        raise C1MeasurementError("predictor chunk mode drifted")
    if decoded.frame0.tobytes(order="C") != expected_y0 or decoded.frame1.tobytes(order="C") != expected_y1:
        raise C1MeasurementError("predictor chunk parse-back differs from preserved planes")


def _materialize_chunk(
    *,
    work_root: Path,
    chunk_index: int,
    pair_ids: Sequence[int],
    fields: Mapping[str, np.ndarray],
    operator: DisjointResizeOperator,
    cache_sha256: str,
    camera_hw: tuple[int, int],
    scorer_hw: tuple[int, int],
    resume: bool,
) -> Mapping[str, Any]:
    y0_rows = [exact_operator_round_u8(operator, np.asarray(fields["gt_f0"][pair_id])) for pair_id in pair_ids]
    y1_rows = [exact_operator_round_u8(operator, np.asarray(fields["gt_f1"][pair_id])) for pair_id in pair_ids]
    y0 = np.stack(y0_rows)
    y1 = np.stack(y1_rows)
    if np.shares_memory(y0, y1) or any(np.array_equal(left, right) for left, right in zip(y0, y1, strict=True)):
        raise C1MeasurementError("source-derived two-plane chunk aliases or copies a pair")
    y0_bytes = y0.tobytes(order="C")
    y1_bytes = y1.tobytes(order="C")
    y0_path, y1_path, predictor_path, manifest_path = _chunk_paths(work_root, chunk_index)
    if manifest_path.exists():
        if not resume:
            raise C1MeasurementError("fresh prepare refuses an existing completed chunk")
        manifest = _read_json(manifest_path, "prepare chunk manifest")
        for path in (y0_path, y1_path, predictor_path):
            if not path.is_file():
                raise C1MeasurementError("completed prepare chunk is missing a bound file")
        predictor_payload = predictor_path.read_bytes()
        expected = _chunk_manifest(
            cache_sha256=cache_sha256,
            chunk_index=chunk_index,
            pair_ids=pair_ids,
            camera_hw=camera_hw,
            scorer_hw=scorer_hw,
            y0_bytes=y0_path.read_bytes(),
            y1_bytes=y1_path.read_bytes(),
            predictor_payload=predictor_payload,
        )
        if manifest != expected or y0_path.read_bytes() != y0_bytes or y1_path.read_bytes() != y1_bytes:
            raise C1MeasurementError("completed prepare chunk scientific custody drifted")
        _validate_predictor_chunk(
            predictor_payload,
            pair_ids=pair_ids,
            scorer_hw=scorer_hw,
            expected_y0=y0_bytes,
            expected_y1=y1_bytes,
        )
        return manifest

    predictor_payload = encode_predictor_residual(
        y0,
        y1,
        modes=SPATIAL_SMOOTH_121_ID,
        pair_ids=tuple(pair_ids),
    )
    _validate_predictor_chunk(
        predictor_payload,
        pair_ids=pair_ids,
        scorer_hw=scorer_hw,
        expected_y0=y0_bytes,
        expected_y1=y1_bytes,
    )
    manifest = _chunk_manifest(
        cache_sha256=cache_sha256,
        chunk_index=chunk_index,
        pair_ids=pair_ids,
        camera_hw=camera_hw,
        scorer_hw=scorer_hw,
        y0_bytes=y0_bytes,
        y1_bytes=y1_bytes,
        predictor_payload=predictor_payload,
    )
    # The timing-free manifest is the completion marker and is always written last.
    _write_once_or_equal(y0_path, y0_bytes)
    _write_once_or_equal(y1_path, y1_bytes)
    _write_once_or_equal(predictor_path, predictor_payload)
    _atomic_write_once(manifest_path, _canonical_json(manifest))
    return manifest


def _combine_predictor_chunks(
    work_root: Path,
    manifests: Sequence[Mapping[str, Any]],
    *,
    pair_count: int,
    scorer_hw: tuple[int, int],
) -> bytes:
    bodies: list[bytes] = []
    expected_next = 0
    for manifest in manifests:
        chunk_index = _exact_int(manifest.get("chunk_index"), "chunk index")
        _y0_path, _y1_path, predictor_path, _manifest_path = _chunk_paths(work_root, chunk_index)
        payload = predictor_path.read_bytes()
        if len(payload) < PREDICTOR_PREFIX.size:
            raise C1MeasurementError("predictor chunk is truncated")
        magic, version, content_tag, count, height, width, channels = PREDICTOR_PREFIX.unpack_from(payload)
        if (
            magic != PREDICTOR_MAGIC
            or version != PREDICTOR_VERSION
            or content_tag != CONTENT_CODEC_TAG
            or count != manifest.get("pair_count")
            or (height, width, channels) != (*scorer_hw, 3)
        ):
            raise C1MeasurementError("predictor chunk prefix drifted")
        pair_ids = manifest.get("pair_ids")
        if pair_ids != list(range(expected_next, expected_next + count)):
            raise C1MeasurementError("predictor chunk sequence is not a complete ordered partition")
        expected_next += count
        bodies.append(payload[PREDICTOR_PREFIX.size :])
    if expected_next != pair_count:
        raise C1MeasurementError("predictor chunks do not cover every pair exactly once")
    combined = PREDICTOR_PREFIX.pack(
        PREDICTOR_MAGIC,
        PREDICTOR_VERSION,
        CONTENT_CODEC_TAG,
        pair_count,
        scorer_hw[0],
        scorer_hw[1],
        3,
    ) + b"".join(bodies)
    decoded = decode_predictor_residual(combined)
    if decoded.pair_ids != tuple(range(pair_count)) or decoded.modes != (SPATIAL_SMOOTH_121_ID,) * pair_count:
        raise C1MeasurementError("combined predictor payload parse-back drifted")
    return combined


def _build_production_packet(
    predictor_payload: bytes,
    *,
    pair_count: int,
    camera_hw: tuple[int, int],
    scorer_hw: tuple[int, int],
    decoded_y_sha256: str,
) -> bytes:
    decoded_y_bytes = pair_count * scorer_hw[0] * scorer_hw[1] * 3 * 2
    sections = (
        ProductionSection(
            "y_description",
            PREDICTOR_RESIDUAL_Y_CODEC_ID,
            predictor_payload,
            decoded_y_bytes,
            decoded_y_sha256,
            True,
        ),
        ProductionSection(
            "frame0_policy",
            DESCRIPTION_FRAME0_POLICY_ID,
            b"",
            0,
            _sha256_bytes(b""),
            False,
        ),
    )
    payload_bytes = sum(len(section.payload) for section in sections)
    header: dict[str, Any] = {
        "schema": PACKET_SCHEMA,
        "version": PRODUCTION_VERSION,
        "geometry": {
            "camera_height": camera_hw[0],
            "camera_width": camera_hw[1],
            "scorer_height": scorer_hw[0],
            "scorer_width": scorer_hw[1],
            "channels": 3,
        },
        "pair_count": pair_count,
        "section_count": 2,
        "sections": [section.header_row() for section in sections],
        "counted_section_payload_bytes": payload_bytes,
        "video_derived_payload_bytes": len(predictor_payload),
        "section_framing_bytes": SECTION_LENGTH.size * 2,
        "packet_bytes": 0,
        "receiver_contract_id": RECEIVER_CONTRACT_ID,
        "tie_policy_id": TIE_POLICY_ID,
        "arithmetic_id": ARITHMETIC_ID,
        "frame0_policy_id": DESCRIPTION_FRAME0_POLICY_ID,
        "y_codec_id": PREDICTOR_RESIDUAL_Y_CODEC_ID,
        "residual_codec_id": None,
        "launch_ready": False,
        "score_claim": False,
        "promotion_eligible": False,
    }
    for _ in range(8):
        header_bytes = _canonical_json(header)
        packet_size = PRODUCTION_PREFIX.size + len(header_bytes) + SECTION_LENGTH.size * 2 + payload_bytes
        if header["packet_bytes"] == packet_size:
            break
        header["packet_bytes"] = packet_size
    else:  # pragma: no cover - decimal digit count converges immediately
        raise C1MeasurementError("production packet byte-count fixed point did not converge")
    header_bytes = _canonical_json(header)
    packet = bytearray(PRODUCTION_PREFIX.pack(PRODUCTION_MAGIC, PRODUCTION_VERSION, len(header_bytes)))
    packet.extend(header_bytes)
    for section in sections:
        packet.extend(SECTION_LENGTH.pack(len(section.payload)))
        packet.extend(section.payload)
    result = bytes(packet)
    parsed = parse_packet(result)
    if parsed.packet_bytes != result:
        raise C1MeasurementError("production packet strict parse-back drifted")
    return result


def _canonical_archive_bytes(packet: bytes) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", allowZip64=True) as archive:
        info = zipfile.ZipInfo(MEMBER_NAME, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, packet)
    return stream.getvalue()


def _contest_adapter_bytes(
    *,
    archive_sha256: str,
    packet_sha256: str,
    pair_count: int,
    camera_hw: tuple[int, int],
    scorer_hw: tuple[int, int],
    y0_sha256: str,
    y1_sha256: str,
    workers: int,
    test_only_small_fixture: bool,
) -> bytes:
    """Build the data-free official three-argument shell adapter."""

    if any(len(value) != 64 for value in (archive_sha256, packet_sha256, y0_sha256, y1_sha256)):
        raise C1MeasurementError("contest adapter requires exact archive and packet digests")
    count = _exact_int(pair_count, "contest adapter pair_count", minimum=1)
    camera = _exact_hw(camera_hw, "contest adapter camera_hw")
    scorer = _exact_hw(scorer_hw, "contest adapter scorer_hw")
    worker_count = _exact_int(workers, "contest adapter workers", minimum=1)
    if type(test_only_small_fixture) is not bool:
        raise C1MeasurementError("contest adapter test-only flag must be boolean")
    if not test_only_small_fixture and (
        count != FULL_PAIR_COUNT
        or camera != FULL_CAMERA_HW
        or scorer != FULL_SCORER_HW
        or y0_sha256 != EXPECTED_Y0_SHA256
        or y1_sha256 != EXPECTED_Y1_SHA256
        or worker_count != 4
    ):
        raise C1MeasurementError("production contest adapter differs from frozen full-n600 custody")
    return (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'if [[ "$#" -ne 3 ]]; then\n'
        '  echo "Usage: inflate.sh <archive_dir> <output_dir> <video_names_file>" >&2\n'
        "  exit 2\n"
        "fi\n"
        'PACT_REPO_ROOT="${PACT_REPO_ROOT:-/workspace/pact}"\n'
        'PACT_PYTHON="${PYTHON:-python3}"\n'
        f"export C1_EXPECTED_ARCHIVE_SHA256={archive_sha256}\n"
        f"export C1_EXPECTED_PACKET_SHA256={packet_sha256}\n"
        f"export C1_EXPECTED_Y0_SHA256={y0_sha256}\n"
        f"export C1_EXPECTED_Y1_SHA256={y1_sha256}\n"
        f"export C1_CONTEST_PAIR_COUNT={count}\n"
        f"export C1_CONTEST_CAMERA_HEIGHT={camera[0]}\n"
        f"export C1_CONTEST_CAMERA_WIDTH={camera[1]}\n"
        f"export C1_CONTEST_SCORER_HEIGHT={scorer[0]}\n"
        f"export C1_CONTEST_SCORER_WIDTH={scorer[1]}\n"
        f"export C1_CONTEST_WORKERS={worker_count}\n"
        f"export C1_CONTEST_TEST_ONLY_SMALL_FIXTURE={int(test_only_small_fixture)}\n"
        'exec "${PACT_PYTHON}" "${PACT_REPO_ROOT}/tools/measure_v10_two_plane_receiver_timing.py" '
        'contest-inflate "$1" "$2" "$3"\n'
    ).encode()


def _chunk_manifest_tree_sha256(work_root: Path, count: int) -> str:
    digest = hashlib.sha256()
    for chunk_index in range(count):
        path = _chunk_paths(work_root, chunk_index)[3]
        payload = path.read_bytes()
        relative = path.relative_to(work_root).as_posix().encode("utf-8")
        digest.update(struct.pack(">I", len(relative)))
        digest.update(relative)
        digest.update(struct.pack(">Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def prepare_two_plane_archive(
    *,
    cache_path: Path,
    work_root: Path,
    receipt_path: Path | None = None,
    pair_count: int = FULL_PAIR_COUNT,
    camera_hw: tuple[int, int] = FULL_CAMERA_HW,
    scorer_hw: tuple[int, int] = FULL_SCORER_HW,
    expected_cache_sha256: str | None = EXPECTED_CACHE_SHA256,
    expected_y0_sha256: str | None = EXPECTED_Y0_SHA256,
    expected_y1_sha256: str | None = EXPECTED_Y1_SHA256,
    sacred_root: Path = CANONICAL_SACRED_ROOT,
    resume: bool = False,
    stop_after_chunks: int | None = None,
    requested_storage_bytes: int = MINIMUM_STORAGE_BYTES,
    test_only_small_fixture: bool = False,
) -> Mapping[str, Any]:
    """Materialize canonical chunks and combine their encoded bodies once."""

    count = _exact_int(pair_count, "pair_count", minimum=1)
    camera_hw = _exact_hw(camera_hw, "camera_hw")
    scorer_hw = _exact_hw(scorer_hw, "scorer_hw")
    if type(resume) is not bool or type(test_only_small_fixture) is not bool:
        raise C1MeasurementError("resume/test-only flags must be boolean")
    if not test_only_small_fixture and (
        count != FULL_PAIR_COUNT or camera_hw != FULL_CAMERA_HW or scorer_hw != FULL_SCORER_HW
    ):
        raise C1MeasurementError("non-test prepare is fixed to full n600 C1 geometry")
    root = work_root.expanduser().resolve(strict=False)
    receipt_target = (receipt_path or root / "prepare_receipt.json").expanduser().resolve(strict=False)
    storage = _storage_gate(
        root,
        requested_bytes=requested_storage_bytes,
        test_only_small_fixture=test_only_small_fixture,
    )
    if not resume and root.exists() and any(root.iterdir()):
        raise C1MeasurementError("fresh prepare requires an empty canonical work root")
    root.mkdir(parents=True, exist_ok=True)
    resolved_cache = cache_path.expanduser().resolve()
    resolved_sacred = sacred_root.expanduser().resolve()
    pre_snapshot = {"test_only_not_consulted": True} if test_only_small_fixture else _tree_snapshot(resolved_sacred)
    if not test_only_small_fixture and pre_snapshot.get("exists") is not True:
        raise C1MeasurementError("sacred donor root is absent")
    fields, cache_sha = _load_cache(
        resolved_cache,
        pair_count=count,
        camera_hw=camera_hw,
        scorer_hw=scorer_hw,
        expected_sha256=expected_cache_sha256,
    )
    operator = DisjointResizeOperator.build(
        camera_h=camera_hw[0],
        camera_w=camera_hw[1],
        scorer_h=scorer_hw[0],
        scorer_w=scorer_hw[1],
    )
    total_chunks = (count + CHUNK_PAIRS - 1) // CHUNK_PAIRS
    limit = (
        total_chunks
        if stop_after_chunks is None
        else min(_exact_int(stop_after_chunks, "stop_after_chunks", minimum=1), total_chunks)
    )
    manifests: list[Mapping[str, Any]] = []
    for chunk_index in range(limit):
        start = chunk_index * CHUNK_PAIRS
        stop = min(start + CHUNK_PAIRS, count)
        manifests.append(
            _materialize_chunk(
                work_root=root,
                chunk_index=chunk_index,
                pair_ids=tuple(range(start, stop)),
                fields=fields,
                operator=operator,
                cache_sha256=cache_sha,
                camera_hw=camera_hw,
                scorer_hw=scorer_hw,
                resume=resume,
            )
        )
    if limit < total_chunks:
        if (
            receipt_target.exists()
            or (root / "archive.zip").exists()
            or (root / "archive_input").exists()
            or (root / "inflate.sh").exists()
        ):
            raise C1MeasurementError("partial prepare cannot coexist with final authority artifacts")
        return {
            "schema": PREPARE_SCHEMA,
            "completed": False,
            "completed_chunks": limit,
            "total_chunks": total_chunks,
            "final_receipt_written": False,
            "archive_written": False,
            "score_claim": False,
            "promotion_eligible": False,
        }
    # A resumed run may have skipped completed chunks above; all are now reopened
    # and source-rederived before any final byte is admitted.
    predictor_payload = _combine_predictor_chunks(
        root,
        manifests,
        pair_count=count,
        scorer_hw=scorer_hw,
    )
    y0_digest = hashlib.sha256()
    y1_digest = hashlib.sha256()
    decoded_digest = hashlib.sha256()
    for chunk_index in range(total_chunks):
        y0_path, _y1_path, _predictor_path, _manifest_path = _chunk_paths(root, chunk_index)
        y0_payload = y0_path.read_bytes()
        y0_digest.update(y0_payload)
        decoded_digest.update(y0_payload)
    for chunk_index in range(total_chunks):
        _y0_path, y1_path, _predictor_path, _manifest_path = _chunk_paths(root, chunk_index)
        y1_payload = y1_path.read_bytes()
        y1_digest.update(y1_payload)
        decoded_digest.update(y1_payload)
    y0_sha = y0_digest.hexdigest()
    y1_sha = y1_digest.hexdigest()
    if expected_y0_sha256 is not None and y0_sha != expected_y0_sha256:
        raise C1MeasurementError("aggregate Y0 digest differs from frozen custody")
    if expected_y1_sha256 is not None and y1_sha != expected_y1_sha256:
        raise C1MeasurementError("aggregate Y1 digest differs from frozen custody")
    packet = _build_production_packet(
        predictor_payload,
        pair_count=count,
        camera_hw=camera_hw,
        scorer_hw=scorer_hw,
        decoded_y_sha256=decoded_digest.hexdigest(),
    )
    packet_sha = _sha256_bytes(packet)
    archive_payload = _canonical_archive_bytes(packet)
    archive_sha = _sha256_bytes(archive_payload)
    archive_path = root / "archive.zip"
    _write_once_or_equal(archive_path, archive_payload)
    with zipfile.ZipFile(io.BytesIO(archive_path.read_bytes()), "r") as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != MEMBER_NAME or infos[0].compress_type != zipfile.ZIP_STORED:
            raise C1MeasurementError("prepared archive member grammar drifted")
        reopened_packet = archive.read(MEMBER_NAME)
    if reopened_packet != packet or parse_packet(reopened_packet).packet_bytes != packet:
        raise C1MeasurementError("prepared archive strict parse-back differs")
    contest_archive_dir = root / "archive_input"
    contest_packet_path = contest_archive_dir / MEMBER_NAME
    _write_once_or_equal(contest_packet_path, packet)
    if parse_packet(contest_packet_path.read_bytes()).packet_bytes != packet:
        raise C1MeasurementError("staged official archive input differs on parse-back")
    adapter_path = root / "inflate.sh"
    adapter_payload = _contest_adapter_bytes(
        archive_sha256=archive_sha,
        packet_sha256=packet_sha,
        pair_count=count,
        camera_hw=camera_hw,
        scorer_hw=scorer_hw,
        y0_sha256=y0_sha,
        y1_sha256=y1_sha,
        workers=1 if test_only_small_fixture else 4,
        test_only_small_fixture=test_only_small_fixture,
    )
    _write_once_or_equal(adapter_path, adapter_payload)
    os.chmod(adapter_path, 0o755)
    if adapter_path.stat().st_mode & 0o777 != 0o755:
        raise C1MeasurementError("contest adapter executable mode differs from 0755")
    names_path = root / "video_names.txt"
    _write_once_or_equal(names_path, b"0.mkv\n")
    post_snapshot = pre_snapshot if test_only_small_fixture else _tree_snapshot(resolved_sacred)
    if post_snapshot != pre_snapshot:
        raise C1MeasurementError("sacred donor metadata changed during prepare")
    receipt: dict[str, Any] = {
        "schema": PREPARE_SCHEMA,
        "completed": True,
        "written_at_utc": datetime.now(UTC).isoformat(),
        "axis": LOCAL_CPU_AXIS,
        "pair_count": count,
        "chunk_pairs": CHUNK_PAIRS,
        "chunk_count": total_chunks,
        "camera_hw": list(camera_hw),
        "scorer_hw": list(scorer_hw),
        "source_cache": {
            "path": str(resolved_cache),
            "bytes": resolved_cache.stat().st_size,
            "sha256": cache_sha,
        },
        "sacred_donor_root": None if test_only_small_fixture else str(resolved_sacred),
        "sacred_donor_snapshot_before": pre_snapshot,
        "sacred_donor_snapshot_after": post_snapshot,
        "storage_preflight": storage,
        "y0_sha256": y0_sha,
        "y1_sha256": y1_sha,
        "y0_bytes": count * scorer_hw[0] * scorer_hw[1] * 3,
        "y1_bytes": count * scorer_hw[0] * scorer_hw[1] * 3,
        "predictor_payload_sha256": _sha256_bytes(predictor_payload),
        "predictor_payload_bytes": len(predictor_payload),
        "predictor_chunk_sha256": [row["predictor_sha256"] for row in manifests],
        "prepare_chunk_manifest_sha256": [_sha256_file(_chunk_paths(root, index)[3]) for index in range(total_chunks)],
        "prepare_chunk_tree_sha256": _chunk_manifest_tree_sha256(root, total_chunks),
        "packet_sha256": packet_sha,
        "packet_bytes": len(packet),
        "archive_path": str(archive_path),
        "archive_dir": str(root),
        "archive_sha256": archive_sha,
        "archive_bytes": len(archive_payload),
        "contest_archive_dir": str(contest_archive_dir),
        "contest_packet_path": str(contest_packet_path),
        "contest_packet_bytes": contest_packet_path.stat().st_size,
        "contest_packet_sha256": _sha256_file(contest_packet_path),
        "contest_adapter_path": str(adapter_path),
        "contest_adapter_bytes": adapter_path.stat().st_size,
        "contest_adapter_sha256": _sha256_file(adapter_path),
        "contest_adapter_mode": "0755",
        "contest_adapter_workers": 1 if test_only_small_fixture else 4,
        "contest_adapter_signature": "inflate.sh <archive_dir> <output_dir> <video_names_file>",
        "contest_adapter_bound": True,
        "video_names_file": str(names_path),
        "strict_parseback_identical": True,
        "combined_without_recompression": True,
        "predictor_mode_id": SPATIAL_SMOOTH_121_ID,
        "frame0_policy_id": DESCRIPTION_FRAME0_POLICY_ID,
        "y_codec_id": PREDICTOR_RESIDUAL_Y_CODEC_ID,
        "git_sha": _git_sha(),
        "argv": list(sys.argv),
        "test_only_small_fixture": test_only_small_fixture,
        "score_claim": False,
        "contest_budget_authority": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    if receipt_target.exists():
        if not resume:
            raise C1MeasurementError("fresh prepare refuses an existing final receipt")
        existing = _read_json(receipt_target, "prepare receipt")
        stable_fields = tuple(key for key in receipt if key not in {"written_at_utc", "storage_preflight", "argv"})
        if any(existing.get(key) != receipt.get(key) for key in stable_fields):
            raise C1MeasurementError("resumed final prepare receipt custody drifted")
        return existing
    _atomic_write_once(receipt_target, _canonical_json(receipt))
    return receipt


def _persist_tool_timing_wrapper(
    *,
    result: Any,
    receiver_receipt_path: Path,
    wrapper_path: Path,
    caller_wall: float,
    expected_archive_sha256: str,
    expected_packet_sha256: str,
    expected_archive_input_kind: str,
) -> Mapping[str, Any]:
    _positive_float(caller_wall, CALLER_WALL_FIELD)
    inner = _read_json(receiver_receipt_path, "receiver timing receipt")
    if inner != result.receipt:
        raise C1MeasurementError("returned receiver row differs from its persisted receipt")
    if (
        inner.get("archive_sha256") != expected_archive_sha256
        or inner.get("packet_sha256") != expected_packet_sha256
        or inner.get("archive_input_kind") != expected_archive_input_kind
        or (
            expected_archive_input_kind == "extracted_0_bin"
            and inner.get("canonical_archive_reconstructed") is not True
        )
    ):
        raise C1MeasurementError("receiver input custody differs from its bound archive/packet")
    receiver_total = _positive_float(
        inner.get("timing", {}).get("total_seconds") if isinstance(inner.get("timing"), Mapping) else None,
        "receiver total_seconds",
    )
    if caller_wall + 1e-9 < receiver_total:
        raise C1MeasurementError("caller wall is shorter than the nested receiver timing boundary")
    runtime_custody = _runtime_source_custody()
    source_rows = runtime_custody["sources"]
    expected_receiver_sources = {
        "timed_receiver_sha256": source_rows["timed_receiver"]["sha256"],
        "production_receiver_sha256": source_rows["production_receiver"]["sha256"],
        "integer_solver_sha256": source_rows["integer_solver"]["sha256"],
    }
    if inner.get("source_hashes") != expected_receiver_sources:
        raise C1MeasurementError("receiver source hashes drifted before tool-wrapper persistence")
    wrapper: dict[str, Any] = {
        **inner,
        "schema": TOOL_TIMING_SCHEMA,
        "receiver_receipt_schema": inner.get("schema"),
        "receiver_receipt": {
            "path": str(receiver_receipt_path),
            "bytes": receiver_receipt_path.stat().st_size,
            "sha256": _sha256_file(receiver_receipt_path),
        },
        "tool_runtime_custody": runtime_custody,
        CALLER_WALL_FIELD: caller_wall,
        "caller_wall_boundary": "immediately_before_receiver_call_through_receiver_receipt_persistence_and_reopen",
        "caller_wall_includes_receiver_receipt_persistence": True,
        "caller_wall_includes_tool_wrapper_persistence": False,
    }
    _atomic_write_once(wrapper_path, _canonical_json(wrapper))
    return wrapper


def run_one_inflate(
    *,
    prepare_receipt_path: Path,
    output_dir: Path,
    timing_receipt_path: Path,
    workers: int,
    resume: bool = False,
    stop_after_pairs: int | None = None,
    test_only_small_fixture: bool = False,
) -> Mapping[str, Any]:
    """Call the receiver once and persist a non-self-referential caller-wall row."""

    prepared = _read_json(prepare_receipt_path, "prepare receipt")
    if prepared.get("schema") != PREPARE_SCHEMA or prepared.get("completed") is not True:
        raise C1MeasurementError("inflate requires a complete prepare receipt")
    prepared_root = _validated_prepared_root(
        prepared,
        test_only_small_fixture=test_only_small_fixture,
    )
    pair_count = _exact_int(prepared.get("pair_count"), "prepared pair count", minimum=1)
    camera_hw = _exact_hw(prepared.get("camera_hw"), "prepared camera_hw")
    scorer_hw = _exact_hw(prepared.get("scorer_hw"), "prepared scorer_hw")
    if not test_only_small_fixture and (
        pair_count != FULL_PAIR_COUNT or camera_hw != FULL_CAMERA_HW or scorer_hw != FULL_SCORER_HW
    ):
        raise C1MeasurementError("non-test inflate is fixed to full n600 C1 geometry")
    output_root = output_dir.expanduser().resolve(strict=False)
    if not test_only_small_fixture:
        output_root = _strict_descendant(output_root, prepared_root, "full receiver output_dir")
    wrapper_path = timing_receipt_path.expanduser().resolve(strict=False)
    receiver_receipt_path = Path(f"{wrapper_path}.receiver.json")
    for candidate in (wrapper_path, receiver_receipt_path):
        try:
            candidate.relative_to(output_root)
        except ValueError:
            pass
        else:
            raise C1MeasurementError("timing receipt paths must remain outside the deterministic output tree")
        if candidate.exists():
            raise C1MeasurementError(f"timing wrapper/receiver receipt is write-once: {candidate}")
    archive_path = Path(str(prepared.get("archive_path", "")))
    if (
        not archive_path.is_file()
        or archive_path.stat().st_size != prepared.get("archive_bytes")
        or _sha256_file(archive_path) != prepared.get("archive_sha256")
    ):
        raise C1MeasurementError("prepared archive bytes drifted before inflate")
    caller_start = time.monotonic_ns()
    try:
        result = timed_inflate_two_plane_archive(
            Path(str(prepared["contest_archive_dir"])),
            output_root,
            Path(str(prepared["video_names_file"])),
            timing_receipt_path=receiver_receipt_path,
            resume=resume,
            stop_after_pairs=stop_after_pairs,
            workers=workers,
            expected_pair_count=pair_count,
            expected_camera_hw=camera_hw,
            expected_scorer_hw=scorer_hw,
            expected_y0_sha256=str(prepared["y0_sha256"]),
            expected_y1_sha256=str(prepared["y1_sha256"]),
        )
    except TwoPlaneTimingReceiverError as exc:
        raise C1MeasurementError(f"timed receiver invocation refused: {exc}") from exc
    caller_wall = (time.monotonic_ns() - caller_start) / 1_000_000_000.0
    return _persist_tool_timing_wrapper(
        result=result,
        receiver_receipt_path=receiver_receipt_path,
        wrapper_path=wrapper_path,
        caller_wall=caller_wall,
        expected_archive_sha256=str(prepared["archive_sha256"]),
        expected_packet_sha256=str(prepared["packet_sha256"]),
        expected_archive_input_kind="extracted_0_bin",
    )


def contest_inflate(
    archive_dir: Path,
    output_dir: Path,
    video_names_file: Path,
) -> Mapping[str, Any]:
    """Run the official three-positional-argument adapter with frozen C1 custody."""

    def required_sha(name: str) -> str:
        value = os.environ.get(name)
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise C1MeasurementError(f"contest adapter lacks exact {name} binding")
        return value

    def required_int(name: str) -> int:
        value = os.environ.get(name)
        if not isinstance(value, str) or not value.isdecimal():
            raise C1MeasurementError(f"contest adapter lacks exact {name} binding")
        return _exact_int(int(value), name, minimum=1)

    expected_archive_sha = required_sha("C1_EXPECTED_ARCHIVE_SHA256")
    expected_packet_sha = required_sha("C1_EXPECTED_PACKET_SHA256")
    expected_y0_sha = required_sha("C1_EXPECTED_Y0_SHA256")
    expected_y1_sha = required_sha("C1_EXPECTED_Y1_SHA256")
    pair_count = required_int("C1_CONTEST_PAIR_COUNT")
    camera_hw = (
        required_int("C1_CONTEST_CAMERA_HEIGHT"),
        required_int("C1_CONTEST_CAMERA_WIDTH"),
    )
    scorer_hw = (
        required_int("C1_CONTEST_SCORER_HEIGHT"),
        required_int("C1_CONTEST_SCORER_WIDTH"),
    )
    workers = required_int("C1_CONTEST_WORKERS")
    test_marker = os.environ.get("C1_CONTEST_TEST_ONLY_SMALL_FIXTURE")
    if test_marker not in {"0", "1"}:
        raise C1MeasurementError("contest adapter test-only environment binding is malformed")
    test_only_small_fixture = test_marker == "1"
    if not test_only_small_fixture and (
        pair_count != FULL_PAIR_COUNT
        or camera_hw != FULL_CAMERA_HW
        or scorer_hw != FULL_SCORER_HW
        or expected_y0_sha != EXPECTED_Y0_SHA256
        or expected_y1_sha != EXPECTED_Y1_SHA256
        or workers != 4
    ):
        raise C1MeasurementError("production contest adapter environment differs from frozen full-n600 custody")
    archive_root = archive_dir.expanduser().resolve()
    output_root = output_dir.expanduser().resolve(strict=False)
    packet_path = archive_root / MEMBER_NAME
    if not packet_path.is_file() or (archive_root / "archive.zip").exists():
        raise C1MeasurementError("contest adapter requires the official unzipped archive_dir/0.bin surface")
    packet = packet_path.read_bytes()
    scored_archive_path = archive_root.parent / "archive.zip"
    if not scored_archive_path.is_file():
        raise C1MeasurementError("contest adapter cannot reopen the scored sibling archive.zip")
    scored_archive = scored_archive_path.read_bytes()
    if (
        _sha256_bytes(packet) != expected_packet_sha
        or _sha256_bytes(scored_archive) != expected_archive_sha
        or scored_archive != _canonical_archive_bytes(packet)
    ):
        raise C1MeasurementError("contest adapter archive/packet bytes differ from the prepared binding")
    receipt_key = _sha256_bytes(f"{expected_archive_sha}\0{output_root}".encode())[:20]
    wrapper_path = output_root.parent / f"c1_two_plane_contest_timing_{receipt_key}.json"
    receiver_receipt_path = Path(f"{wrapper_path}.receiver.json")
    try:
        wrapper_path.relative_to(output_root)
    except ValueError:
        pass
    else:
        raise C1MeasurementError("contest timing receipts must remain outside the deterministic output tree")
    if wrapper_path.exists() or receiver_receipt_path.exists():
        raise C1MeasurementError("contest adapter timing receipts are write-once")
    caller_start = time.monotonic_ns()
    try:
        result = timed_inflate_two_plane_archive(
            archive_root,
            output_root,
            video_names_file.expanduser().resolve(),
            timing_receipt_path=receiver_receipt_path,
            resume=False,
            workers=workers,
            expected_pair_count=pair_count,
            expected_camera_hw=camera_hw,
            expected_scorer_hw=scorer_hw,
            expected_y0_sha256=expected_y0_sha,
            expected_y1_sha256=expected_y1_sha,
        )
    except TwoPlaneTimingReceiverError as exc:
        raise C1MeasurementError(f"contest receiver invocation refused: {exc}") from exc
    caller_wall = (time.monotonic_ns() - caller_start) / 1_000_000_000.0
    wrapper = dict(
        _persist_tool_timing_wrapper(
            result=result,
            receiver_receipt_path=receiver_receipt_path,
            wrapper_path=wrapper_path,
            caller_wall=caller_wall,
            expected_archive_sha256=expected_archive_sha,
            expected_packet_sha256=expected_packet_sha,
            expected_archive_input_kind="extracted_0_bin",
        )
    )
    return {
        "schema": "v10_two_plane_contest_inflate_adapter_result.v1",
        "completed": wrapper.get("completed") is True,
        "raw_sha256": wrapper.get("raw_sha256"),
        "raw_bytes": wrapper.get("raw_bytes"),
        "timing_wrapper_path": str(wrapper_path),
        "timing_wrapper_sha256": _sha256_file(wrapper_path),
        "workers": workers,
        "test_only_small_fixture": test_only_small_fixture,
        "archive_input_kind": wrapper.get("archive_input_kind"),
        "score_claim": False,
        "promotion_eligible": False,
    }


def _validate_timing_receipt(
    path: Path,
    *,
    role: str,
    pair_count: int,
    raw_bytes: int,
    numerator_values: int,
) -> Mapping[str, Any]:
    row = _read_json(path, f"{role} timing receipt")
    if row.get("schema") != TOOL_TIMING_SCHEMA:
        raise C1MeasurementError(f"{role} receipt is not a tool-owned caller-wall row")
    receiver_schema = row.get("receiver_receipt_schema")
    receiver_custody = row.get("receiver_receipt")
    if receiver_schema != TIMING_RECEIPT_SCHEMA or not isinstance(receiver_custody, Mapping):
        raise C1MeasurementError(f"{role} nested receiver receipt custody is missing")
    receiver_path_value = receiver_custody.get("path")
    if not isinstance(receiver_path_value, str):
        raise C1MeasurementError(f"{role} nested receiver receipt path is missing")
    receiver_path = Path(receiver_path_value)
    if (
        not receiver_path.is_file()
        or receiver_path.stat().st_size != receiver_custody.get("bytes")
        or _sha256_file(receiver_path) != receiver_custody.get("sha256")
    ):
        raise C1MeasurementError(f"{role} nested receiver receipt custody drifted")
    receiver_row = _read_json(receiver_path, f"{role} nested receiver receipt")
    reconstructed = dict(row)
    for key in (
        "receiver_receipt_schema",
        "receiver_receipt",
        "tool_runtime_custody",
        CALLER_WALL_FIELD,
        "caller_wall_boundary",
        "caller_wall_includes_receiver_receipt_persistence",
        "caller_wall_includes_tool_wrapper_persistence",
    ):
        reconstructed.pop(key, None)
    reconstructed["schema"] = receiver_schema
    if reconstructed != receiver_row:
        raise C1MeasurementError(f"{role} wrapper fields differ from the nested receiver receipt")
    runtime_custody = row.get("tool_runtime_custody")
    if not isinstance(runtime_custody, Mapping) or runtime_custody != _runtime_source_custody():
        raise C1MeasurementError(f"{role} runtime source custody drifted")
    source_hashes = row.get("source_hashes")
    expected_source_keys = {
        "timed_receiver_sha256",
        "production_receiver_sha256",
        "integer_solver_sha256",
    }
    if (
        not isinstance(source_hashes, Mapping)
        or set(source_hashes) != expected_source_keys
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in source_hashes.values()
        )
    ):
        raise C1MeasurementError(f"{role} receiver source hash custody is malformed")
    runtime_sources = runtime_custody["sources"]
    expected_receiver_hashes = {
        "timed_receiver_sha256": runtime_sources["timed_receiver"]["sha256"],
        "production_receiver_sha256": runtime_sources["production_receiver"]["sha256"],
        "integer_solver_sha256": runtime_sources["integer_solver"]["sha256"],
    }
    if source_hashes != expected_receiver_hashes:
        raise C1MeasurementError(f"{role} receiver source hash custody drifted")
    host = row.get("host")
    if (
        not isinstance(host, Mapping)
        or any(not isinstance(host.get(key), str) or not host.get(key) for key in ("platform", "machine", "python"))
        or _exact_int(host.get("pid"), f"{role} host pid", minimum=1) < 1
    ):
        raise C1MeasurementError(f"{role} host custody is malformed")
    thread_environment = row.get("thread_environment")
    expected_thread_keys = {
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }
    if (
        not isinstance(thread_environment, Mapping)
        or set(thread_environment) != expected_thread_keys
        or any(value is not None and not isinstance(value, str) for value in thread_environment.values())
    ):
        raise C1MeasurementError(f"{role} thread-environment custody is malformed")
    invocation_argv = row.get("argv")
    if (
        not isinstance(invocation_argv, list)
        or not invocation_argv
        or any(not isinstance(value, str) for value in invocation_argv)
    ):
        raise C1MeasurementError(f"{role} invocation argv custody is malformed")
    if row.get("completed") is not True:
        raise C1MeasurementError(f"{role} receipt is not a completed timed inflate")
    if (
        row.get("fresh") is not True
        or row.get("resume_requested") is not False
        or type(row.get("resumed_pairs")) is not int
        or row.get("resumed_pairs") != 0
    ):
        raise C1MeasurementError(f"{role} receipt must be a fresh non-resumed invocation")
    if (
        _exact_int(row.get("pair_count"), f"{role} pair_count", minimum=1) != pair_count
        or _exact_int(row.get("raw_bytes"), f"{role} raw_bytes", minimum=1) != raw_bytes
    ):
        raise C1MeasurementError(f"{role} receipt full geometry/count custody drifted")
    if (
        _exact_int(row.get("numerator_values_verified"), f"{role} verified numerators", minimum=1) != numerator_values
        or _exact_int(row.get("numerator_values_expected"), f"{role} expected numerators", minimum=1)
        != numerator_values
        or row.get("both_planes_exact") is not True
    ):
        raise C1MeasurementError(f"{role} receipt exact numerator total drifted")
    execution = row.get("execution")
    if not isinstance(execution, dict):
        raise C1MeasurementError(f"{role} receipt execution row is missing")
    workers = _exact_int(execution.get("workers"), f"{role} workers", minimum=1)
    if role == "serial" and (workers != 1 or execution.get("mode") != "serial"):
        raise C1MeasurementError("serial receipt must be the one-worker baseline")
    if role != "serial" and (workers < 4 or execution.get("mode") != "process_pool"):
        raise C1MeasurementError("parallel receipts require a >=4-worker process pool")
    timing = row.get("timing")
    if not isinstance(timing, dict):
        raise C1MeasurementError(f"{role} receipt timing row is missing")
    components = [_positive_float(timing.get(name), f"{role}.{name}") for name in TIMING_COMPONENTS]
    component_sum = _positive_float(timing.get("component_sum_seconds"), f"{role}.component_sum_seconds")
    total = _positive_float(timing.get("total_seconds"), f"{role}.total_seconds")
    caller_wall = _positive_float(row.get(CALLER_WALL_FIELD), f"{role}.{CALLER_WALL_FIELD}")
    if (
        timing.get("total_boundary") != "entry_through_pre_receipt_evidence_collection"
        or timing.get("receipt_serialization_and_persistence_included") is not False
        or row.get("caller_wall_includes_receiver_receipt_persistence") is not True
        or row.get("caller_wall_includes_tool_wrapper_persistence") is not False
    ):
        raise C1MeasurementError(f"{role} timing boundaries are not explicit or non-self-referential")
    overhead = timing.get("unclassified_overhead_seconds")
    if (
        isinstance(overhead, bool)
        or not isinstance(overhead, (int, float))
        or not math.isfinite(float(overhead))
        or overhead < 0
    ):
        raise C1MeasurementError(f"{role} unclassified timing is invalid")
    if not math.isclose(component_sum, sum(components), rel_tol=1e-9, abs_tol=1e-9):
        raise C1MeasurementError(f"{role} component timing sum is inconsistent")
    if total + 1e-9 < component_sum or caller_wall + 1e-9 < total:
        raise C1MeasurementError(f"{role} receiver/caller timing accounting is inconsistent")
    per_pair = timing.get("per_pair")
    if not isinstance(per_pair, list) or len(per_pair) != pair_count:
        raise C1MeasurementError(f"{role} per-pair timing coverage is incomplete")
    for expected_index, pair in enumerate(per_pair):
        if (
            not isinstance(pair, dict)
            or type(pair.get("pair_index")) is not int
            or pair.get("pair_index") != expected_index
            or pair.get("resumed") is not False
        ):
            raise C1MeasurementError(f"{role} per-pair timing order/resume status drifted")
        _positive_float(pair.get("solve0_seconds"), f"{role}.per_pair.solve0_seconds")
        _positive_float(pair.get("solve1_seconds"), f"{role}.per_pair.solve1_seconds")
    if (
        row.get("contest_budget_verdict") is not None
        or "timing_verdict" in row
        or "local_lt_1800" in row
        or row.get("contest_budget_authority") is not False
    ):
        raise C1MeasurementError(f"{role} local timing receipt claimed contest-budget authority")
    for key in ("score_claim", "promotion_eligible", "pointer_moved"):
        if row.get(key) is not False:
            raise C1MeasurementError(f"{role} receipt authority field {key} must remain false")
    for key in (
        "archive_sha256",
        "packet_sha256",
        "y0_sha256",
        "y1_sha256",
        "raw_sha256",
        "stage_tree_sha256",
        "plane0_tree_sha256",
        "chunk_tree_sha256",
        "output_tree_sha256",
    ):
        value = row.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise C1MeasurementError(f"{role} receipt digest {key} is missing")
    output_root_value = row.get("output_root")
    state_root_value = row.get("state_root")
    raw_relative_value = row.get("raw_relative_path")
    if (
        not isinstance(output_root_value, str)
        or not isinstance(state_root_value, str)
        or not isinstance(raw_relative_value, str)
    ):
        raise C1MeasurementError(f"{role} output path custody is missing")
    output_root = Path(output_root_value).expanduser().resolve()
    state_root = _strict_descendant(Path(state_root_value), output_root, f"{role} state_root")
    raw_relative = Path(raw_relative_value)
    if raw_relative.is_absolute() or ".." in raw_relative.parts or not raw_relative.parts:
        raise C1MeasurementError(f"{role} raw relative path is unsafe")
    raw_path = output_root / raw_relative
    if not raw_path.is_file() or raw_path.stat().st_size != raw_bytes or _sha256_file(raw_path) != row["raw_sha256"]:
        raise C1MeasurementError(f"{role} reopened raw output custody drifted")
    reopened_trees = {
        "stage_tree_sha256": _content_tree_sha256(state_root / "pairs"),
        "plane0_tree_sha256": _content_tree_sha256(state_root / "plane0"),
        "pair_manifest_tree_sha256": _content_tree_sha256(state_root / "pair_manifests"),
        "plane0_manifest_tree_sha256": _content_tree_sha256(state_root / "plane0_manifests"),
        "chunk_tree_sha256": _content_tree_sha256(state_root / "chunk_manifests"),
        "output_tree_sha256": _content_tree_sha256(output_root),
    }
    if any(row.get(key) != value for key, value in reopened_trees.items()):
        raise C1MeasurementError(f"{role} reopened output/stage tree custody drifted")
    return row


def _validate_output_identity(rows: Sequence[Mapping[str, Any]]) -> None:
    keys = (
        "archive_sha256",
        "packet_sha256",
        "y0_sha256",
        "y1_sha256",
        "raw_sha256",
        "raw_bytes",
        "stage_tree_sha256",
        "plane0_tree_sha256",
        "pair_manifest_tree_sha256",
        "plane0_manifest_tree_sha256",
        "chunk_tree_sha256",
        "output_tree_sha256",
        "source_hashes",
        "tool_runtime_custody",
        "archive_input_kind",
        "canonical_archive_reconstructed",
    )
    first = rows[0]
    for index, row in enumerate(rows[1:], start=1):
        if any(row.get(key) != first.get(key) for key in keys):
            raise C1MeasurementError(f"timing output {index} differs from deterministic serial/parallel bytes")


def _load_calibration_anchors(paths: Sequence[Path]) -> list[Mapping[str, Any]]:
    anchors: list[Mapping[str, Any]] = []
    for path in paths:
        row = _read_json(path, "paired timing calibration anchor")
        if row.get("schema") != CALIBRATION_SCHEMA or row.get("measured") is not True or row.get("paired") is not True:
            raise C1MeasurementError("calibration anchor must be an explicitly measured paired row")
        if any("margin" in str(key).lower() for key in row):
            raise C1MeasurementError("calibration anchor contains an invented timing margin")
        classification = row.get("classification")
        if classification not in {"inflate_only", "full_official_evaluation"}:
            raise C1MeasurementError("calibration anchor classification is invalid")
        local = _positive_float(row.get("local_inflate_seconds"), "calibration local_inflate_seconds")
        contest = _positive_float(row.get("contest_seconds"), "calibration contest_seconds")
        if (
            not isinstance(row.get("local_host"), str)
            or not row["local_host"]
            or not isinstance(row.get("contest_instance_class"), str)
            or not row["contest_instance_class"]
        ):
            raise C1MeasurementError("calibration anchor lacks measured host/instance custody")
        archive_sha = row.get("archive_sha256")
        if not isinstance(archive_sha, str) or len(archive_sha) != 64:
            raise C1MeasurementError("calibration anchor lacks exact archive custody")
        for evidence_name in ("local_evidence", "contest_evidence"):
            evidence = row.get(evidence_name)
            if not isinstance(evidence, Mapping):
                raise C1MeasurementError("calibration anchor lacks reopenable paired evidence custody")
            evidence_path_value = evidence.get("path")
            if not isinstance(evidence_path_value, str) or not evidence_path_value:
                raise C1MeasurementError("calibration evidence path is missing")
            evidence_path = Path(evidence_path_value).expanduser().resolve()
            if (
                not evidence_path.is_file()
                or evidence_path.stat().st_size
                != _exact_int(evidence.get("bytes"), f"{evidence_name} bytes", minimum=1)
                or _sha256_file(evidence_path) != evidence.get("sha256")
                or evidence.get("archive_sha256") != archive_sha
            ):
                raise C1MeasurementError("calibration paired evidence custody drifted")
        if row["contest_evidence"].get("classification") != classification:
            raise C1MeasurementError("contest calibration evidence scope differs from its classification")
        anchors.append({**row, "ratio_contest_to_local": contest / local})
    return anchors


def derive_timing_verdict(
    exploited_local_seconds: float,
    calibration_anchors: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Apply only an empirical paired local-to-contest spread."""

    local = _positive_float(exploited_local_seconds, "exploited local receiver seconds")
    full = [row for row in calibration_anchors if row.get("classification") == "full_official_evaluation"]
    if not full:
        return {
            "verdict": "CLOSE -> MODAL_MEASUREMENT_OWED",
            "reason": "no measured paired local-inflate to full-official-evaluation calibration anchor",
            "budget_seconds": 1800,
            "verdict_scope": "exploited local receiver timing mapped to full contest evaluation; calibration absent",
            "calibration_ratio_range": None,
            "predicted_full_evaluation_seconds_range": None,
        }
    ratios = [_positive_float(row.get("ratio_contest_to_local"), "calibration ratio") for row in full]
    low_ratio, high_ratio = min(ratios), max(ratios)
    predicted = (local * low_ratio, local * high_ratio)
    if predicted[1] < 1800.0:
        verdict = "CLEARLY_UNDER"
        reason = "upper edge of the measured paired calibration spread remains below the full-evaluation budget"
    elif predicted[0] >= 1800.0:
        verdict = "CLEARLY_OVER"
        reason = "lower edge of the measured paired calibration spread reaches or exceeds the full-evaluation budget"
    else:
        verdict = "CLOSE -> MODAL_MEASUREMENT_OWED"
        reason = "measured paired calibration spread crosses the full-evaluation budget"
    return {
        "verdict": verdict,
        "reason": reason,
        "budget_seconds": 1800,
        "verdict_scope": "exploited >=4-worker local receiver timing mapped to full official contest evaluation",
        "calibration_ratio_range": [low_ratio, high_ratio],
        "predicted_full_evaluation_seconds_range": list(predicted),
    }


def _modal_ticket(
    *,
    archive_path: Path,
    archive_sha256: str,
    archive_bytes: int,
    packet_sha256: str,
    contest_adapter_path: Path,
    contest_adapter_sha256: str,
    contest_adapter_bytes: int,
    contest_adapter_mode: str,
    contest_adapter_workers: int,
    test_only_small_fixture: bool,
) -> Mapping[str, Any]:
    if (
        not archive_path.is_file()
        or archive_path.stat().st_size != archive_bytes
        or _sha256_file(archive_path) != archive_sha256
    ):
        raise C1MeasurementError("Modal ticket archive custody drifted immediately before readiness")
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            infos = archive.infolist()
            if len(infos) != 1 or infos[0].filename != MEMBER_NAME or infos[0].compress_type != zipfile.ZIP_STORED:
                raise C1MeasurementError("Modal ticket archive member grammar drifted")
            packet = archive.read(MEMBER_NAME)
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        raise C1MeasurementError("Modal ticket archive cannot be reopened") from exc
    if _sha256_bytes(packet) != packet_sha256 or _canonical_archive_bytes(packet) != archive_path.read_bytes():
        raise C1MeasurementError("Modal ticket packet/canonical ZIP custody drifted")
    if (
        not contest_adapter_path.is_file()
        or contest_adapter_path.stat().st_size != contest_adapter_bytes
        or _sha256_file(contest_adapter_path) != contest_adapter_sha256
    ):
        raise C1MeasurementError("Modal ticket adapter custody drifted immediately before readiness")
    single_flight_key = f"c1-two-plane-full-evaluate-{archive_sha256[:20]}"
    lane_id = "lane_c1_two_plane_receiver_timing_20260719"
    claims_ledger = ".omx/state/active_lane_dispatch_claims.md"
    call_id_ledger = ".omx/state/modal_call_id_ledger.jsonl"
    submission_dir = "submissions/c1_two_plane_receiver_timing_20260719"
    workers = _exact_int(contest_adapter_workers, "ticket contest adapter workers", minimum=1)
    if contest_adapter_mode != "0755" or contest_adapter_path.stat().st_mode & 0o777 != 0o755:
        raise C1MeasurementError("Modal ticket requires executable 0755 adapter custody")
    if type(test_only_small_fixture) is not bool:
        raise C1MeasurementError("ticket test-only flag must be boolean")
    if not test_only_small_fixture and workers != 4:
        raise C1MeasurementError("production Modal ticket requires the fixed four-worker adapter")
    runtime_custody = _runtime_source_custody()
    checkout_ready = runtime_custody["remote_checkout_reproduces_sources"] is True
    dispatch_ready = not test_only_small_fixture and checkout_ready
    dispatch_blockers: list[str] = []
    if test_only_small_fixture:
        dispatch_blockers.append("test-only fixture is never dispatch-authoritative")
    if not checkout_ready:
        dispatch_blockers.append(
            "required remote checkout SHA does not reproduce every bound runtime source byte; MAIN landing owed"
        )
    claim_command = [
        "python3",
        "tools/claim_lane_dispatch.py",
        "claim",
        "--claims-path",
        claims_ledger,
        "--lane-id",
        lane_id,
        "--platform",
        "modal",
        "--instance-job-id",
        single_flight_key,
        "--agent",
        "codex:gpt-5.6-sol",
        "--status",
        "eval",
        "--notes",
        "C1 two-plane CPU full evaluate; max_cost_usd=20; exact archive and adapter bound",
    ]
    return {
        "schema": MODAL_TICKET_SCHEMA,
        "status": "TEST_ONLY_UNFIRED" if test_only_small_fixture else "UNFIRED",
        "dispatch_attempted": False,
        "dispatch_authorized": False,
        "test_only_small_fixture": test_only_small_fixture,
        "structural_ticket_ready": dispatch_ready,
        "ready_for_operator_authorized_dispatch": dispatch_ready,
        "dispatch_blockers": dispatch_blockers,
        "evaluation_scope": "full evaluate.sh: unzip + inflate + scoring in one contest-hardware run",
        "budget_seconds": 1800,
        "max_cost_usd": 20,
        "archive": {
            "path": str(archive_path.resolve()),
            "bytes": archive_bytes,
            "sha256": archive_sha256,
            "packet_sha256": packet_sha256,
        },
        "inflate_entrypoint": {
            "tool": "tools/measure_v10_two_plane_receiver_timing.py",
            "subcommand": "contest-inflate",
            "workers": workers,
            "contest_adapter_signature": "inflate.sh <archive_dir> <output_dir> <video_names_file>",
            "contest_adapter_path": str(contest_adapter_path.resolve()),
            "contest_adapter_bytes": contest_adapter_bytes,
            "contest_adapter_sha256": contest_adapter_sha256,
            "contest_adapter_mode": contest_adapter_mode,
            "contest_adapter_bound": True,
            "contest_cli": "contest-inflate ARCHIVE_DIR OUTPUT_DIR VIDEO_NAMES_FILE",
            "current_receiver_archive_input": "<archive_dir>/0.bin",
            "official_evaluate_archive_input": "<archive_dir>/0.bin after unzip",
        },
        "remote_submission_materialization": [
            {
                "source_path": str(archive_path.resolve()),
                "destination_path": f"{submission_dir}/archive.zip",
                "sha256": archive_sha256,
                "bytes": archive_bytes,
            },
            {
                "source_path": str(contest_adapter_path.resolve()),
                "destination_path": f"{submission_dir}/inflate.sh",
                "sha256": contest_adapter_sha256,
                "bytes": contest_adapter_bytes,
                "mode": "0755",
                "post_copy_chmod_argv": ["chmod", "0755", f"{submission_dir}/inflate.sh"],
            },
        ],
        "remote_workdir": "/workspace/pact",
        "runtime_custody": runtime_custody,
        "required_remote_checkout_git_sha": runtime_custody["required_remote_checkout_git_sha"],
        "remote_source_hash_revalidation_required": True,
        "planned_submission_dir": submission_dir,
        "full_evaluate_command": [
            "bash",
            "upstream/evaluate.sh",
            "--submission-dir",
            submission_dir,
            "--video-names-file",
            "upstream/public_test_video_names.txt",
            "--device",
            "cpu",
        ],
        "instance_classes": [
            {
                "axis": "contest-CPU",
                "cpu_cores": 4,
                "memory_gib": 16,
                "resource_axis_ready": True,
                "dispatch_ready": dispatch_ready,
                "blocker": None if dispatch_ready else "test-only fixture",
            },
            {
                "axis": "contest-CUDA",
                "gpu": "T4",
                "host_memory_gib": 26,
                "vram_gib": 16,
                "resource_axis_ready": False,
                "dispatch_ready": False,
                "blocker": "deterministic CUDA receiver implementation and parity custody are absent",
                "seconds": None,
                "timing_verdict": None,
            },
        ],
        "resource_envelope": {
            "catalog": 381,
            "canonical_output_dir_safety_required": True,
            "full_evaluation_timing_breakdown_required": ["inflate_seconds", "scoring_seconds", "total_seconds"],
        },
        "lane_id": lane_id,
        "single_flight_key": single_flight_key,
        "single_flight_guard": "tac.deploy.modal.single_flight.assert_modal_single_flight",
        "pre_dispatch_claim_command": claim_command if dispatch_ready else None,
        "dispatch_claim_ledger": claims_ledger,
        "call_id_ledger": call_id_ledger,
        "call_id_append_required_after_dispatch": True,
        "governed_launcher_required": True,
        "score_claim": False,
        "promotion_eligible": False,
    }


def cuda_workload_envelope() -> Mapping[str, Any]:
    """Return deterministic work counts only; never manufacture CUDA seconds."""

    return {
        "status": "DERIVED_UNMEASURED_CUDA_WORKLOAD",
        "output_bytes": CUDA_OUTPUT_BYTES,
        "numerator_values": CUDA_NUMERATOR_VALUES,
        "uniform_tap_products": CUDA_UNIFORM_TAP_PRODUCTS,
        "seconds": None,
        "timing_verdict": None,
        "contest_budget_authority": False,
        "score_claim": False,
        "promotion_eligible": False,
    }


def _extract_oracle_subset(
    raw_path: Path,
    pair_ids: Sequence[int],
    *,
    pair_count: int,
    camera_hw: tuple[int, int],
    output_root: Path,
) -> Path:
    frame_bytes = camera_hw[0] * camera_hw[1] * 3
    expected_full = pair_count * frame_bytes * 2
    if not raw_path.is_file() or raw_path.stat().st_size != expected_full:
        raise C1MeasurementError("hard-oracle source raw size drifted")
    payload = bytearray()
    with raw_path.open("rb") as handle:
        for pair_id in pair_ids:
            handle.seek(pair_id * frame_bytes * 2)
            block = handle.read(frame_bytes * 2)
            if len(block) != frame_bytes * 2:
                raise C1MeasurementError("hard-oracle pair extraction truncated")
            payload.extend(block)
    digest = _sha256_bytes(bytes(payload))
    target = output_root / f"hard_oracle_pairs_{'-'.join(map(str, pair_ids))}_{digest[:16]}.raw"
    _write_once_or_equal(target, bytes(payload))
    return target


def _default_hard_oracle(
    raw_path: Path,
    pair_ids: Sequence[int],
    *,
    pair_count: int,
    camera_hw: tuple[int, int],
    cache_path: Path,
    upstream: Path,
    cpu_threads: int,
    output_root: Path,
) -> Mapping[str, Any]:
    from tools import measure_v10_free_predictor_floor as predictor_floor

    if camera_hw != FULL_CAMERA_HW:
        raise C1MeasurementError("native-f32 hard oracle admits only real camera geometry")
    subset = _extract_oracle_subset(
        raw_path,
        pair_ids,
        pair_count=pair_count,
        camera_hw=camera_hw,
        output_root=output_root,
    )
    bundle = predictor_floor._load_scorers(upstream.expanduser().resolve(), cpu_threads)
    modules = sys.modules.get("modules")
    if modules is None:
        raise C1MeasurementError("frozen scorer module custody is absent after load")
    weight_rows = {}
    for label, attribute, expected in (
        ("segnet", "segnet_sd_path", EXPECTED_SEGNET_SHA256),
        ("posenet", "posenet_sd_path", EXPECTED_POSENET_SHA256),
    ):
        path = Path(getattr(modules, attribute, "")).expanduser().resolve()
        if not path.is_file() or _sha256_file(path) != expected:
            raise C1MeasurementError(f"frozen {label} weights differ from C1 custody")
        weight_rows[label] = {"path": str(path), "sha256": expected, "bytes": path.stat().st_size}
    result = predictor_floor.score_inflated_raw(
        subset,
        pair_ids=pair_ids,
        cache_path=cache_path,
        upstream=upstream,
        cpu_threads=cpu_threads,
        require_canonical_hash=True,
        scorer_bundle=bundle,
    )
    return {
        **result,
        "input_contract": HARD_ORACLE_INPUT_CONTRACT,
        "subset_raw_path": str(subset),
        "subset_raw_bytes": subset.stat().st_size,
        "subset_raw_sha256": _sha256_file(subset),
        "weights": weight_rows,
        "law_id": F32_LAW_ID,
        "measurement_label": "MEASURED",
    }


def _validate_hard_oracle(result: Mapping[str, Any], pair_ids: Sequence[int]) -> None:
    if result.get("input_contract") != HARD_ORACLE_INPUT_CONTRACT:
        raise C1MeasurementError("hard oracle did not bind both complete frames and official scorer paths")
    rows = result.get("pairs")
    if not isinstance(rows, list) or len(rows) < 6:
        raise C1MeasurementError("hard oracle requires at least six actual pair rows")
    observed_ids = [row.get("pair_id") for row in rows if isinstance(row, dict)]
    if any(type(value) is not int for value in observed_ids) or observed_ids != list(pair_ids):
        raise C1MeasurementError("hard-oracle pair IDs differ from the frozen subset")
    for row in rows:
        if not isinstance(row, dict):
            raise C1MeasurementError("hard-oracle pair row is malformed")
        for key in ("d_seg", "d_pose"):
            value = row.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise C1MeasurementError(f"hard-oracle {key} is invalid")
        if type(row.get("seg_mismatched_pixels")) is not int or row["seg_mismatched_pixels"] < 0:
            raise C1MeasurementError("hard-oracle Seg mismatch count is invalid")
        pose6 = row.get("pose6")
        if (
            not isinstance(pose6, list)
            or len(pose6) != 6
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
                for value in pose6
            )
        ):
            raise C1MeasurementError("hard-oracle Pose6 row is invalid")


HardOracleRunner = Callable[..., Mapping[str, Any]]


def compose_timing_receipt(
    *,
    prepare_receipt_path: Path,
    serial_receipt_path: Path,
    parallel_receipt_paths: Sequence[Path],
    output_path: Path,
    calibration_anchor_paths: Sequence[Path] = (),
    hard_oracle_pair_ids: Sequence[int] = HARD_ORACLE_PAIR_IDS,
    cache_path: Path = CANONICAL_CACHE,
    upstream: Path = Path("/Users/adpena/Projects/pact/upstream"),
    cpu_threads: int = 4,
    hard_oracle_runner: HardOracleRunner | None = None,
    expected_pair_count: int = FULL_PAIR_COUNT,
    expected_raw_bytes: int = FULL_RAW_BYTES,
    expected_numerator_values: int = FULL_NUMERATOR_VALUES,
    test_only_small_fixture: bool = False,
) -> Mapping[str, Any]:
    """Compose one compact receipt from three independent process rows."""

    if len(parallel_receipt_paths) != 2:
        raise C1MeasurementError("compose requires exactly two fresh parallel timing receipts")
    pair_count = _exact_int(expected_pair_count, "expected_pair_count", minimum=1)
    raw_bytes = _exact_int(expected_raw_bytes, "expected_raw_bytes", minimum=1)
    numerator_values = _exact_int(expected_numerator_values, "expected_numerator_values", minimum=1)
    if not test_only_small_fixture and (
        pair_count != FULL_PAIR_COUNT or raw_bytes != FULL_RAW_BYTES or numerator_values != FULL_NUMERATOR_VALUES
    ):
        raise C1MeasurementError("non-test compose is fixed to full n600 exact counts")
    if not test_only_small_fixture and calibration_anchor_paths:
        raise C1MeasurementError(
            "production calibration anchors are blocked until canonical inner full-evaluate receipt validation exists"
        )
    if not test_only_small_fixture and hard_oracle_runner is not None:
        raise C1MeasurementError("production compose forbids injected hard-oracle runners")
    prepared = _read_json(prepare_receipt_path, "prepare receipt")
    if prepared.get("schema") != PREPARE_SCHEMA or prepared.get("completed") is not True:
        raise C1MeasurementError("compose requires a complete prepare receipt")
    prepared_root = _validated_prepared_root(
        prepared,
        test_only_small_fixture=test_only_small_fixture,
    )
    if _exact_int(prepared.get("pair_count"), "prepared pair count", minimum=1) != pair_count:
        raise C1MeasurementError("prepare/timing pair counts disagree")
    archive_path = Path(str(prepared.get("archive_path", "")))
    if (
        not archive_path.is_file()
        or archive_path.stat().st_size != _exact_int(prepared.get("archive_bytes"), "prepared archive bytes", minimum=1)
        or _sha256_file(archive_path) != prepared.get("archive_sha256")
    ):
        raise C1MeasurementError("prepared archive bytes drifted before composition")
    serial = _validate_timing_receipt(
        serial_receipt_path,
        role="serial",
        pair_count=pair_count,
        raw_bytes=raw_bytes,
        numerator_values=numerator_values,
    )
    parallel = [
        _validate_timing_receipt(
            path,
            role=f"parallel-{index + 1}",
            pair_count=pair_count,
            raw_bytes=raw_bytes,
            numerator_values=numerator_values,
        )
        for index, path in enumerate(parallel_receipt_paths)
    ]
    _validate_output_identity((serial, *parallel))
    if serial.get("archive_sha256") != prepared.get("archive_sha256") or serial.get("packet_sha256") != prepared.get(
        "packet_sha256"
    ):
        raise C1MeasurementError("timing receipts do not bind the prepared archive/packet")
    if serial.get("y0_sha256") != prepared.get("y0_sha256") or serial.get("y1_sha256") != prepared.get("y1_sha256"):
        raise C1MeasurementError("timing receipts do not bind the prepared source planes")
    ids = tuple(_exact_int(value, f"hard_oracle_pair_ids[{index}]") for index, value in enumerate(hard_oracle_pair_ids))
    if len(ids) < 6 or len(set(ids)) != len(ids) or any(value < 0 or value >= pair_count for value in ids):
        raise C1MeasurementError("compose requires at least six unique in-range hard-oracle pairs")
    raw_path = Path(str(parallel[0]["output_root"])) / str(parallel[0]["raw_relative_path"])
    camera_hw = _exact_hw(prepared.get("camera_hw"), "prepared camera_hw")
    runner = hard_oracle_runner or _default_hard_oracle
    oracle_root = prepared_root / "hard_oracle"
    oracle_root.mkdir(parents=True, exist_ok=True)
    oracle = runner(
        raw_path,
        ids,
        pair_count=pair_count,
        camera_hw=camera_hw,
        cache_path=cache_path,
        upstream=upstream,
        cpu_threads=cpu_threads,
        output_root=oracle_root,
    )
    if not isinstance(oracle, Mapping):
        raise C1MeasurementError("hard-oracle runner did not return a mapping")
    _validate_hard_oracle(oracle, ids)
    if not test_only_small_fixture:
        weights = oracle.get("weights")
        segnet_weight = weights.get("segnet") if isinstance(weights, Mapping) else None
        posenet_weight = weights.get("posenet") if isinstance(weights, Mapping) else None
        if (
            oracle.get("law_id") != F32_LAW_ID
            or oracle.get("measurement_label") != "MEASURED"
            or oracle.get("cache_sha256") != EXPECTED_CACHE_SHA256
            or oracle.get("raw_sha256") != oracle.get("subset_raw_sha256")
            or not isinstance(segnet_weight, Mapping)
            or segnet_weight.get("sha256") != EXPECTED_SEGNET_SHA256
            or not isinstance(posenet_weight, Mapping)
            or posenet_weight.get("sha256") != EXPECTED_POSENET_SHA256
        ):
            raise C1MeasurementError("production hard-oracle scorer/cache/raw custody is incomplete")
    anchors = _load_calibration_anchors(calibration_anchor_paths)
    exploited_seconds = max(float(row[CALLER_WALL_FIELD]) for row in parallel)
    timing_verdict = derive_timing_verdict(exploited_seconds, anchors)
    sacred_after_measurement = _revalidate_sacred_donor(
        prepared,
        test_only_small_fixture=test_only_small_fixture,
    )
    modal_ticket = (
        _modal_ticket(
            archive_path=archive_path,
            archive_sha256=str(prepared["archive_sha256"]),
            archive_bytes=_exact_int(prepared["archive_bytes"], "prepared archive bytes", minimum=1),
            packet_sha256=str(prepared["packet_sha256"]),
            contest_adapter_path=Path(str(prepared["contest_adapter_path"])),
            contest_adapter_sha256=str(prepared["contest_adapter_sha256"]),
            contest_adapter_bytes=_exact_int(
                prepared["contest_adapter_bytes"],
                "prepared contest adapter bytes",
                minimum=1,
            ),
            contest_adapter_mode=str(prepared["contest_adapter_mode"]),
            contest_adapter_workers=_exact_int(
                prepared["contest_adapter_workers"],
                "prepared contest adapter workers",
                minimum=1,
            ),
            test_only_small_fixture=test_only_small_fixture,
        )
        if timing_verdict["verdict"] == "CLOSE -> MODAL_MEASUREMENT_OWED"
        else None
    )
    source_receipts = [prepare_receipt_path, serial_receipt_path, *parallel_receipt_paths]
    receipt: dict[str, Any] = {
        "schema": COMPOSE_SCHEMA,
        "written_at_utc": datetime.now(UTC).isoformat(),
        "axis": LOCAL_CPU_AXIS,
        "archive": {
            "path": str(archive_path.resolve()),
            "bytes": prepared["archive_bytes"],
            "sha256": prepared["archive_sha256"],
            "packet_sha256": prepared["packet_sha256"],
        },
        "planes": {
            "y0_sha256": prepared["y0_sha256"],
            "y1_sha256": prepared["y1_sha256"],
            "distinct": True,
        },
        "pair_count": pair_count,
        "raw_bytes": raw_bytes,
        "numerator_values_verified": numerator_values,
        "deterministic_output": {
            "serial_parallel_byte_identical": True,
            "parallel_double_decode_byte_identical": True,
            "raw_sha256": serial["raw_sha256"],
            "stage_tree_sha256": serial["stage_tree_sha256"],
            "plane0_tree_sha256": serial["plane0_tree_sha256"],
            "chunk_tree_sha256": serial["chunk_tree_sha256"],
            "output_tree_sha256": serial["output_tree_sha256"],
        },
        "local_timings": {
            "serial": serial["timing"],
            "parallel": [row["timing"] for row in parallel],
            "serial_caller_wall_seconds_through_receiver_receipt_persistence": serial[CALLER_WALL_FIELD],
            "parallel_caller_wall_seconds_through_receiver_receipt_persistence": [
                row[CALLER_WALL_FIELD] for row in parallel
            ],
            "parallel_workers": [row["execution"]["workers"] for row in parallel],
            "verdict_input_seconds": exploited_seconds,
            "verdict_input_rule": (
                "maximum of two fresh >=4-worker caller walls through nested receiver receipt persistence"
            ),
        },
        "execution_custody": {
            "runtime_sources": serial["tool_runtime_custody"],
            "invocations": [
                {
                    "role": role,
                    "host": row["host"],
                    "thread_environment": row["thread_environment"],
                    "argv": row["argv"],
                    "receiver_source_hashes": row["source_hashes"],
                    "receiver_receipt": row["receiver_receipt"],
                }
                for role, row in zip(("serial", "parallel-1", "parallel-2"), (serial, *parallel), strict=True)
            ],
        },
        "hard_oracle": dict(oracle),
        "hard_oracle_pair_ids": list(ids),
        "f32_law_id": F32_LAW_ID,
        "sacred_donor_snapshot_after_measurement": sacred_after_measurement,
        "calibration_anchors": anchors,
        "calibration_authority_status": (
            "TEST_ONLY_PAIRED_EVIDENCE"
            if test_only_small_fixture and anchors
            else "BLOCKED_CANONICAL_FULL_EVALUATE_RECEIPT_VALIDATOR_OWED"
        ),
        "timing_verdict": timing_verdict,
        "modal_measurement_ticket": modal_ticket,
        "cuda_workload": cuda_workload_envelope(),
        "source_receipts": [
            {
                "role": role,
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for role, path in zip(("prepare", "serial", "parallel-1", "parallel-2"), source_receipts, strict=True)
        ],
        "labels": {
            "MEASURED": [
                "three local receiver invocations",
                "three tool caller walls through nested receiver receipt persistence",
                "raw/stage/chunk digests",
                "both-plane exact numerator totals",
                "native-f32 hard-oracle subset",
            ],
            "DERIVED": [
                "timing calibration interval from measured paired anchors",
                "CUDA integer workload counts",
            ],
            "SPECULATIVE": [],
        },
        "verdict_scope": "C1 two-independent-plane receiver viability only; no score, rate, promotion, or pointer authority",
        "full_official_evaluation_measured": False,
        "contest_budget_authority": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "git_sha": _git_sha(),
    }
    content_basis = _canonical_json(receipt)
    receipt["content_address"] = {
        "algorithm": "sha256",
        "scope": "canonical JSON with content_address field absent",
        "sha256": _sha256_bytes(content_basis),
    }
    _atomic_write_once(output_path.expanduser().resolve(), _canonical_json(receipt))
    return receipt


def _load_prepared_planes(
    prepared: Mapping[str, Any], pair_ids: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, DisjointResizeOperator]:
    prepared_mode = prepared.get("test_only_small_fixture")
    if type(prepared_mode) is not bool:
        raise C1MeasurementError("prepare receipt test-only mode is malformed")
    root = _validated_prepared_root(prepared, test_only_small_fixture=prepared_mode)
    pair_count = _exact_int(prepared.get("pair_count"), "prepared pair count", minimum=1)
    scorer_hw = _exact_hw(prepared.get("scorer_hw"), "prepared scorer_hw")
    camera_hw = _exact_hw(prepared.get("camera_hw"), "prepared camera_hw")
    selected0: list[np.ndarray] = []
    selected1: list[np.ndarray] = []
    plane_bytes = scorer_hw[0] * scorer_hw[1] * 3
    cache: dict[int, tuple[bytes, bytes, Mapping[str, Any]]] = {}
    for pair_id in pair_ids:
        if type(pair_id) is not int or pair_id < 0 or pair_id >= pair_count:
            raise C1MeasurementError("MLX pair id is outside prepared range")
        chunk_index = pair_id // CHUNK_PAIRS
        if chunk_index not in cache:
            y0_path, y1_path, _predictor_path, manifest_path = _chunk_paths(root, chunk_index)
            cache[chunk_index] = (
                y0_path.read_bytes(),
                y1_path.read_bytes(),
                _read_json(manifest_path, "MLX prepare chunk manifest"),
            )
        y0_payload, y1_payload, manifest = cache[chunk_index]
        ids = manifest.get("pair_ids")
        if not isinstance(ids, list) or pair_id not in ids:
            raise C1MeasurementError("MLX pair id is absent from its prepared chunk")
        offset = ids.index(pair_id) * plane_bytes
        selected0.append(
            np.frombuffer(y0_payload[offset : offset + plane_bytes], dtype=np.uint8).reshape(*scorer_hw, 3).copy()
        )
        selected1.append(
            np.frombuffer(y1_payload[offset : offset + plane_bytes], dtype=np.uint8).reshape(*scorer_hw, 3).copy()
        )
    operator = DisjointResizeOperator.build(
        camera_h=camera_hw[0], camera_w=camera_hw[1], scorer_h=scorer_hw[0], scorer_w=scorer_hw[1]
    )
    return np.stack(selected0), np.stack(selected1), operator


def run_mlx_parity(
    *,
    prepare_receipt_path: Path,
    output_path: Path,
    pair_ids: Sequence[int] = HARD_ORACLE_PAIR_IDS,
    mlx_module: Any | None = None,
    runtime_status_fn: Callable[..., Mapping[str, Any]] = mlx_runtime_status,
    parity_fn: Callable[..., Mapping[str, Any]] = parity_check_mlx_two_plane,
) -> Mapping[str, Any]:
    """Emit an honest Metal parity row or an explicit host-custody refusal."""

    prepared = _read_json(prepare_receipt_path, "prepare receipt")
    if prepared.get("schema") != PREPARE_SCHEMA or prepared.get("completed") is not True:
        raise C1MeasurementError("MLX parity requires a complete prepare receipt")
    prepared_mode = prepared.get("test_only_small_fixture")
    if type(prepared_mode) is not bool:
        raise C1MeasurementError("prepare receipt test-only mode is malformed")
    _validated_prepared_root(prepared, test_only_small_fixture=prepared_mode)
    ids = tuple(_exact_int(value, f"pair_ids[{index}]") for index, value in enumerate(pair_ids))
    if len(ids) < 6 or len(ids) != len(set(ids)):
        raise C1MeasurementError("MLX parity requires at least six unique pair ids")
    runtime = runtime_status_fn(mlx_module=mlx_module)
    receipt: dict[str, Any] = {
        "schema": MLX_TOOL_SCHEMA,
        "written_at_utc": datetime.now(UTC).isoformat(),
        "axis": MLX_AXIS,
        "pair_ids": list(ids),
        "pair_count": len(ids),
        "runtime_custody": dict(runtime),
        "parity_measured": False,
        "parity_passed": False,
        "contest_verdict_input": False,
        "contest_budget_authority": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    if runtime.get("runtime_installed") is not True or runtime.get("metal_usable") is not True:
        receipt["status"] = "HOST_CUSTODY_REFUSAL"
        receipt["host_custody_refusal"] = runtime.get("host_custody_refusal")
    else:
        y0, y1, operator = _load_prepared_planes(prepared, ids)
        try:
            parity = parity_fn(
                operator,
                y0,
                y1,
                pair_ids=ids,
                mlx_module=mlx_module,
            )
        except TwoPlaneTimingReceiverError as exc:
            raise C1MeasurementError(f"MLX parity execution refused: {exc}") from exc
        if (
            parity.get("score_claim") is not False
            or parity.get("promotion_eligible") is not False
            or parity.get("contest_timing_verdict_eligible") is not False
        ):
            raise C1MeasurementError("MLX parity result attempted to claim contest authority")
        receipt.update(
            {
                "status": "PARITY_PASS" if parity.get("parity_passed") is True else "INTEGER_OP_DIVERGENCE",
                "parity_measured": True,
                "parity_passed": parity.get("parity_passed") is True,
                "parity": dict(parity),
                "divergences": parity.get("divergences", []),
            }
        )
    _atomic_write_once(output_path.expanduser().resolve(), _canonical_json(receipt))
    return receipt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="materialize immutable source/predictor chunks")
    prepare.add_argument("--cache", type=Path, default=CANONICAL_CACHE)
    prepare.add_argument("--work-root", type=Path, default=CANONICAL_WORK_ROOT)
    prepare.add_argument("--receipt", type=Path)
    prepare.add_argument("--resume", action="store_true")
    prepare.add_argument("--stop-after-chunks", type=int)
    prepare.add_argument("--requested-storage-bytes", type=int, default=MINIMUM_STORAGE_BYTES)
    prepare.add_argument("--test-only-small-fixture", action="store_true", help=argparse.SUPPRESS)
    prepare.add_argument("--pair-count", type=int, default=FULL_PAIR_COUNT, help=argparse.SUPPRESS)
    prepare.add_argument("--camera-height", type=int, default=FULL_CAMERA_HW[0], help=argparse.SUPPRESS)
    prepare.add_argument("--camera-width", type=int, default=FULL_CAMERA_HW[1], help=argparse.SUPPRESS)
    prepare.add_argument("--scorer-height", type=int, default=FULL_SCORER_HW[0], help=argparse.SUPPRESS)
    prepare.add_argument("--scorer-width", type=int, default=FULL_SCORER_HW[1], help=argparse.SUPPRESS)

    inflate = subparsers.add_parser("inflate", help="run exactly one timed receiver invocation")
    inflate.add_argument("--prepare-receipt", type=Path, required=True)
    inflate.add_argument("--output-dir", type=Path, required=True)
    inflate.add_argument("--receipt", type=Path, required=True)
    inflate.add_argument("--workers", type=int, required=True)
    inflate.add_argument("--resume", action="store_true")
    inflate.add_argument("--stop-after-pairs", type=int)
    inflate.add_argument("--test-only-small-fixture", action="store_true", help=argparse.SUPPRESS)

    contest = subparsers.add_parser(
        "contest-inflate",
        help="official fixed C1 inflate.sh adapter target",
    )
    contest.add_argument("archive_dir", type=Path)
    contest.add_argument("output_dir", type=Path)
    contest.add_argument("video_names_file", type=Path)

    compose = subparsers.add_parser("compose", help="compose serial + duplicate parallel timing custody")
    compose.add_argument("--prepare-receipt", type=Path, required=True)
    compose.add_argument("--serial-receipt", type=Path, required=True)
    compose.add_argument("--parallel-receipt", type=Path, action="append", required=True)
    compose.add_argument("--calibration-anchor", type=Path, action="append", default=[])
    compose.add_argument("--output", type=Path, required=True)
    compose.add_argument("--cache", type=Path, default=CANONICAL_CACHE)
    compose.add_argument("--upstream", type=Path, default=Path("/Users/adpena/Projects/pact/upstream"))
    compose.add_argument("--cpu-threads", type=int, default=4)

    mlx = subparsers.add_parser("mlx-parity", help="run false-authority Metal integer parity")
    mlx.add_argument("--prepare-receipt", type=Path, required=True)
    mlx.add_argument("--output", type=Path, required=True)
    mlx.add_argument("--pair-id", type=int, action="append", default=[])
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            test_only = bool(args.test_only_small_fixture)
            result = prepare_two_plane_archive(
                cache_path=args.cache,
                work_root=args.work_root,
                receipt_path=args.receipt,
                pair_count=args.pair_count,
                camera_hw=(args.camera_height, args.camera_width),
                scorer_hw=(args.scorer_height, args.scorer_width),
                expected_cache_sha256=None if test_only else EXPECTED_CACHE_SHA256,
                expected_y0_sha256=None if test_only else EXPECTED_Y0_SHA256,
                expected_y1_sha256=None if test_only else EXPECTED_Y1_SHA256,
                resume=args.resume,
                stop_after_chunks=args.stop_after_chunks,
                requested_storage_bytes=args.requested_storage_bytes,
                test_only_small_fixture=test_only,
            )
        elif args.command == "inflate":
            result = run_one_inflate(
                prepare_receipt_path=args.prepare_receipt,
                output_dir=args.output_dir,
                timing_receipt_path=args.receipt,
                workers=args.workers,
                resume=args.resume,
                stop_after_pairs=args.stop_after_pairs,
                test_only_small_fixture=args.test_only_small_fixture,
            )
        elif args.command == "contest-inflate":
            result = contest_inflate(
                args.archive_dir,
                args.output_dir,
                args.video_names_file,
            )
        elif args.command == "compose":
            result = compose_timing_receipt(
                prepare_receipt_path=args.prepare_receipt,
                serial_receipt_path=args.serial_receipt,
                parallel_receipt_paths=args.parallel_receipt,
                output_path=args.output,
                calibration_anchor_paths=args.calibration_anchor,
                cache_path=args.cache,
                upstream=args.upstream,
                cpu_threads=args.cpu_threads,
            )
        elif args.command == "mlx-parity":
            result = run_mlx_parity(
                prepare_receipt_path=args.prepare_receipt,
                output_path=args.output,
                pair_ids=tuple(args.pair_id) if args.pair_id else HARD_ORACLE_PAIR_IDS,
            )
        else:  # pragma: no cover - argparse owns this branch
            raise C1MeasurementError("unknown command")
    except (C1MeasurementError, StorageTierError, OSError, ValueError) as exc:
        raise SystemExit(f"C1 two-plane measurement refused: {exc}") from exc
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
