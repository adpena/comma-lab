#!/usr/bin/env python3
"""Build/measure contract for the pool x channel Jacobian R-D experiment.

``plan`` and ``self-test`` deliberately import only the Python standard library
at module import time. ``measure`` wires the real rounded bank -> canonical
render/R -> frozen CPU Torch SegNet full-path surface, then fails closed before
cell findings until exact skip/deep and range/kernel interventions exist. It
never substitutes a local head Gram or a synthetic scorer.
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import zipfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "pool_channel_jacobian_rd.v1"
ROW_SCHEMA = "pool_channel_jacobian_rd.cell.v1"
CHECKPOINT_SCHEMA = "pool_channel_jacobian_rd.checkpoint.v1"
UNMEASURED = "UNMEASURED_AWAITING_GOVERNED_N600"
LIVENESS_ONLY = "LIVENESS_ONLY_N_LT_600"
POOLS = (
    "A_road_lane_edge_near",
    "B_saddle",
    "C_remainder",
)
SETTLED_SINGULAR_VALUES = (3.128, 2.154, 2.025, 1.796)
PATHS = ("skip", "deep")
RESIZE_COMPONENTS = ("range_A", "ker_A")
STAGES = (
    "custody",
    "baseline_byte_close",
    "geometry",
    "coherent_corrections",
    "archive_rd",
    "complete",
)
CPU_AXIS = "frozen_cpu_torch_segnet"
LANE_ID = "pool_channel_rd_harness_20260718"
QUEUE_PREDECESSOR_RUN_ID = "levelset_n600_witness_20260717T113932Z"
SEGNET_WEIGHTS_SHA256 = "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"

DEFAULT_INPUTS = {
    "bank": (
        "experiments/results/banks/v9c2_defensive_bank_20260718/levelset_witness_ema_BEST.npz",
        460_448,
        "b0a431e9259cd3c54ae53b677076823f36e096b27eb0d9ba74ed7c54c9113cef",
    ),
    "gt_n600": (
        "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
        5_078_017_610,
        "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6",
    ),
    "gt_n96": (
        "experiments/results/mlx_fleet_gt_cache/gt_n96.npz",
        812_484_058,
        "6aad6600d93a5c25e94207ee411d3b4daf93136b8ea4235b6f7b9d96f04ab104",
    ),
}


class HarnessRefusal(RuntimeError):
    """A fail-closed contract refusal, never a scientific negative verdict."""


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def canonical_payload_sha256(payload: Mapping[str, Any]) -> str:
    """Hash a checkpoint payload independently of its mutable record envelope."""
    return hashlib.sha256(canonical_json_bytes(dict(payload))).hexdigest()


def atomic_write_json(path: Path, payload: Any) -> None:
    """Durably replace ``path`` with deterministic JSON on the same filesystem."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_canonical_root(explicit: str | None = None) -> Path:
    """Resolve the canonical data root without hard-coding a user or host path."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("PACT_CANONICAL_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    marker = f"{os.sep}.omx{os.sep}tmp{os.sep}codex_worktrees{os.sep}"
    raw = str(repo)
    if marker in raw:
        return Path(raw.split(marker, 1)[0])
    return repo


def validate_file_custody(path: Path, expected_size: int, expected_sha256: str) -> dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise HarnessRefusal(f"BLOCKED_INPUT_MISSING:{path}")
    size = path.stat().st_size
    if size != expected_size:
        raise HarnessRefusal(f"BLOCKED_INPUT_SIZE_MISMATCH:{path}:expected={expected_size}:actual={size}")
    actual_sha = sha256_file(path)
    if actual_sha != expected_sha256:
        raise HarnessRefusal(f"BLOCKED_INPUT_SHA256_MISMATCH:{path}:expected={expected_sha256}:actual={actual_sha}")
    return {"path": str(path), "bytes": size, "sha256": actual_sha, "validated": True}


def read_dynamic_pointer_custody(path: Path) -> dict[str, Any]:
    """Read/hash the live pointer atomically enough to detect concurrent drift.

    No score, byte size, or digest is compiled into this harness: every plan derives
    the current authority from the canonical file it actually read.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise HarnessRefusal(f"BLOCKED_CANONICAL_POINTER_MISSING:{path}")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise HarnessRefusal("BLOCKED_CANONICAL_POINTER_DRIFT_DURING_READ")
    try:
        payload = json.loads(raw)
        cpu = payload["our_local_frontier_contest_cpu"]
        score = float(cpu["score"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HarnessRefusal("BLOCKED_CANONICAL_POINTER_SCHEMA") from exc
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "current_contest_cpu_score": score,
        "axis": cpu.get("axis"),
        "archive_sha256": cpu.get("archive_sha256"),
        "last_refreshed_utc": payload.get("last_refreshed_utc"),
        "validated": True,
    }


def _read_npy_header(stream: Any) -> tuple[dict[str, Any], int]:
    if stream.read(6) != b"\x93NUMPY":
        raise HarnessRefusal("BLOCKED_CACHE_MEMBER_NOT_NPY")
    major, minor = struct.unpack("BB", stream.read(2))
    if major == 1:
        header_len = struct.unpack("<H", stream.read(2))[0]
    elif major in (2, 3):
        header_len = struct.unpack("<I", stream.read(4))[0]
    else:
        raise HarnessRefusal(f"BLOCKED_UNSUPPORTED_NPY_VERSION:{major}.{minor}")
    raw = stream.read(header_len)
    try:
        header = ast.literal_eval(raw.decode("latin1").strip())
    except (SyntaxError, ValueError) as exc:
        raise HarnessRefusal("BLOCKED_INVALID_NPY_HEADER") from exc
    return header, stream.tell()


def _dtype_itemsize(descr: str) -> int:
    digits = "".join(ch for ch in descr if ch.isdigit())
    if not digits:
        raise HarnessRefusal(f"BLOCKED_UNSUPPORTED_NPY_DTYPE:{descr}")
    return int(digits)


def npz_member_prefix(path: Path, member: str, first_axis_count: int) -> dict[str, Any]:
    """Hash a C-contiguous first-axis prefix without importing NumPy or loading a cache."""
    with zipfile.ZipFile(path) as archive:
        if member not in archive.namelist():
            raise HarnessRefusal(f"BLOCKED_CACHE_MEMBER_MISSING:{path}:{member}")
        with archive.open(member) as stream:
            header, _ = _read_npy_header(stream)
            shape = tuple(int(x) for x in header["shape"])
            if header.get("fortran_order") or not shape:
                raise HarnessRefusal(f"BLOCKED_CACHE_LAYOUT:{path}:{member}")
            if first_axis_count > shape[0]:
                raise HarnessRefusal(f"BLOCKED_CACHE_PAIR_COUNT:{path}:{member}")
            row_bytes = _dtype_itemsize(str(header["descr"]))
            for dim in shape[1:]:
                row_bytes *= dim
            remaining = first_axis_count * row_bytes
            digest = hashlib.sha256()
            while remaining:
                chunk = stream.read(min(8 << 20, remaining))
                if not chunk:
                    raise HarnessRefusal(f"BLOCKED_CACHE_TRUNCATED:{path}:{member}")
                digest.update(chunk)
                remaining -= len(chunk)
    return {
        "member": member,
        "shape": list(shape),
        "dtype": str(header["descr"]),
        "first_axis_count_hashed": first_axis_count,
        "prefix_sha256": digest.hexdigest(),
    }


def read_stored_npz_member_rows(path: Path, member: str, indices: Sequence[int]) -> Any:
    """Read selected C-order rows from a ZIP_STORED NPY member without inflating n600.

    The pinned GT cache intentionally stores ``lstars.npy`` uncompressed.  Requiring that
    layout keeps memory bounded and makes each seek correspond to an exact cached row.
    """
    import numpy as np

    requested = [int(index) for index in indices]
    if any(type(index) is not int or index < 0 for index in indices):
        raise HarnessRefusal("BLOCKED_CACHE_ROW_INDEX_SCHEMA")
    with zipfile.ZipFile(path) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError as exc:
            raise HarnessRefusal(f"BLOCKED_CACHE_MEMBER_MISSING:{path}:{member}") from exc
        if info.compress_type != zipfile.ZIP_STORED:
            raise HarnessRefusal(f"BLOCKED_CACHE_MEMBER_NOT_SEEKABLE_STORED:{path}:{member}")
        with archive.open(info) as stream:
            header, data_offset = _read_npy_header(stream)
            shape = tuple(int(value) for value in header["shape"])
            if header.get("fortran_order") or len(shape) < 1:
                raise HarnessRefusal(f"BLOCKED_CACHE_LAYOUT:{path}:{member}")
            dtype = np.dtype(str(header["descr"]))
            if dtype.hasobject:
                raise HarnessRefusal(f"BLOCKED_CACHE_OBJECT_DTYPE:{path}:{member}")
            if any(index >= shape[0] for index in requested):
                raise HarnessRefusal(f"BLOCKED_CACHE_ROW_OUT_OF_RANGE:{path}:{member}")
            row_bytes = int(dtype.itemsize * math.prod(shape[1:]))
            rows = []
            for index in requested:
                stream.seek(data_offset + index * row_bytes)
                raw = stream.read(row_bytes)
                if len(raw) != row_bytes:
                    raise HarnessRefusal(f"BLOCKED_CACHE_TRUNCATED_ROW:{path}:{member}:{index}")
                rows.append(np.frombuffer(raw, dtype=dtype).reshape(shape[1:]).copy())
    if not rows:
        return np.empty((0, *shape[1:]), dtype=dtype)
    return np.stack(rows)


def validate_cache_identity(gt_n600: Path, gt_n96: Path) -> dict[str, Any]:
    """Prove n96 has the same first 96 frozen-argmax pair identities as n600."""
    large = npz_member_prefix(gt_n600, "lstars.npy", 96)
    small = npz_member_prefix(gt_n96, "lstars.npy", 96)
    if large["shape"][0] != 600 or small["shape"][0] != 96:
        raise HarnessRefusal(f"BLOCKED_CACHE_IDENTITY_PAIR_COUNTS:n600={large['shape'][0]}:n96={small['shape'][0]}")
    if large["shape"][1:] != small["shape"][1:] or large["dtype"] != small["dtype"]:
        raise HarnessRefusal("BLOCKED_CACHE_IDENTITY_LAYOUT_MISMATCH")
    if large["prefix_sha256"] != small["prefix_sha256"]:
        raise HarnessRefusal("BLOCKED_CACHE_IDENTITY_LSTARS_PREFIX_MISMATCH")
    return {
        "validated": True,
        "identity_surface": "exact raw lstars.npy bytes for pair indices 0..95",
        "gt_n600": large,
        "gt_n96": small,
    }


def make_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pool in POOLS:
        for index, singular in enumerate(SETTLED_SINGULAR_VALUES, start=1):
            direction = f"head_sv{index}"
            for path in PATHS:
                for resize in RESIZE_COMPONENTS:
                    rows.append(
                        {
                            "schema": ROW_SCHEMA,
                            "row_id": f"{pool}__{direction}__{path}__{resize}",
                            "pool": pool,
                            "head_direction": direction,
                            "settled_singular_value": singular,
                            "path": path,
                            "resize_component": resize,
                            "status": UNMEASURED,
                            "finding_eligible": False,
                            "intrinsic_floor": None,
                            "extrinsic_ceiling": None,
                            "collateral_cost": None,
                            "rd_curve_points": [],
                            "g_act": None,
                            "cross_location_off_diagonal_energy": None,
                            "path_attribution": None,
                        }
                    )
    validate_rows(rows)
    return rows


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    if len(rows) != 48:
        raise HarnessRefusal(f"BLOCKED_ROW_CARDINALITY:expected=48:actual={len(rows)}")
    ids = [str(row.get("row_id")) for row in rows]
    if len(set(ids)) != 48:
        raise HarnessRefusal("BLOCKED_DUPLICATE_ROW_ID")
    expected = {(p, f"head_sv{i}", q, r) for p in POOLS for i in range(1, 5) for q in PATHS for r in RESIZE_COMPONENTS}
    actual = {(str(x["pool"]), str(x["head_direction"]), str(x["path"]), str(x["resize_component"])) for x in rows}
    if actual != expected:
        raise HarnessRefusal("BLOCKED_ROW_CARTESIAN_PRODUCT_MISMATCH")
    for row in rows:
        if row["intrinsic_floor"] is not None or row["extrinsic_ceiling"] is not None:
            raise HarnessRefusal("BLOCKED_PREMATURE_SCIENTIFIC_VALUE")
        if row["collateral_cost"] is not None or row["rd_curve_points"] != []:
            raise HarnessRefusal("BLOCKED_PREMATURE_RD_VALUE")


def assign_exclusive_pools(flip: Any, saddle: Any, edge_or_near: Any, road_lane_nearest: Any) -> dict[str, Any]:
    """Saddle-first, exactly-one attribution for supplied boolean arrays."""
    import numpy as np

    f, s, e, rl = np.broadcast_arrays(
        np.asarray(flip, dtype=bool),
        np.asarray(saddle, dtype=bool),
        np.asarray(edge_or_near, dtype=bool),
        np.asarray(road_lane_nearest, dtype=bool),
    )
    pool_b = f & s
    pool_a = f & ~s & e & rl
    pool_c = f & ~(pool_a | pool_b)
    total = pool_a.astype(np.uint8) + pool_b.astype(np.uint8) + pool_c.astype(np.uint8)
    if np.any(total[f] != 1) or np.any(total[~f] != 0):
        raise HarnessRefusal("BLOCKED_POOL_DISJOINTNESS_OR_EXHAUSTIVENESS")
    return {POOLS[0]: pool_a, POOLS[1]: pool_b, POOLS[2]: pool_c}


def canonical_pool_masks_from_labels(
    truth: Any,
    predicted: Any,
    strata_masks: Callable[..., Any],
    nearest_boundary_indices: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    """Derive the exact saddle-first pool partition from GT and baseline argmax labels."""
    import numpy as np

    if nearest_boundary_indices is None:
        try:
            from scipy.ndimage import distance_transform_edt
        except ImportError as exc:
            raise HarnessRefusal("BLOCKED_REAL_POOL_SCIPY_DEPENDENCY") from exc

        def nearest_boundary_indices(boundary_mask: Any) -> Any:
            return distance_transform_edt(
                ~np.asarray(boundary_mask, dtype=bool),
                return_distances=False,
                return_indices=True,
            )

    labels = np.asarray(truth)
    baseline = np.asarray(predicted)
    if labels.shape != baseline.shape or labels.ndim != 2:
        raise HarnessRefusal("BLOCKED_REAL_POOL_LABEL_SHAPE")
    flip = baseline != labels
    saddle, boundary, pair_id, near = strata_masks(labels)
    # Attribute every near-edge location to the class pair at its nearest *boundary*.
    # EDT against only Road/Lane edges would make the nearest-pair predicate tautological.
    if np.any(boundary):
        nearest = np.asarray(nearest_boundary_indices(boundary))
        if nearest.shape != (2, *boundary.shape):
            raise HarnessRefusal("BLOCKED_REAL_POOL_NEAREST_INDEX_SHAPE")
        nearest_pair = pair_id[nearest[0], nearest[1]]
        road_lane_nearest = nearest_pair == 1  # min(Road=0,Lane=1)*5+max(...)=1
    else:
        road_lane_nearest = np.zeros_like(flip)
    return assign_exclusive_pools(flip, saddle, near, road_lane_nearest)


def canonical_pool_mask_digest(pool: str, shape: Sequence[int], mask_chunks: Iterable[Any]) -> str:
    """Hash a pool-labelled bool mask independently of its ``.npy`` container."""
    import numpy as np

    if pool not in POOLS:
        raise HarnessRefusal(f"BLOCKED_UNKNOWN_POOL_FOR_MASK_DIGEST:{pool}")
    digest = hashlib.sha256()
    digest.update(
        canonical_json_bytes(
            {
                "schema": "pool_channel_jacobian_rd.canonical_pool_mask.v1",
                "pool": pool,
                "shape": [int(value) for value in shape],
                "encoding": "uint8-C-order; false=0; true=1",
            }
        )
    )
    for chunk in mask_chunks:
        digest.update(np.ascontiguousarray(np.asarray(chunk, dtype=np.uint8)).tobytes())
    return digest.hexdigest()


def deterministic_svd_signs(u: Any, vh: Any) -> tuple[Any, Any]:
    """Fix each right singular vector's largest-absolute coordinate positive."""
    import numpy as np

    fixed_u = np.array(u, dtype=np.float64, copy=True)
    fixed_vh = np.array(vh, dtype=np.float64, copy=True)
    if fixed_u.ndim != 2 or fixed_vh.ndim != 2 or fixed_u.shape[1] != fixed_vh.shape[0]:
        raise ValueError("incompatible U/Vh shapes")
    for k, row in enumerate(fixed_vh):
        pivot = int(np.argmax(np.abs(row)))
        if row[pivot] < 0:
            fixed_vh[k] *= -1.0
            fixed_u[:, k] *= -1.0
    return fixed_u, fixed_vh


def verify_settled_singular_values(values: Sequence[float], atol: float = 5e-4) -> None:
    if len(values) < 5 or abs(float(values[4])) > 1e-6:
        raise HarnessRefusal("BLOCKED_HEAD_NOT_EXACT_RANK4")
    for actual, settled in zip(values[:4], SETTLED_SINGULAR_VALUES, strict=True):
        if abs(float(actual) - settled) > atol:
            raise HarnessRefusal(f"BLOCKED_SETTLED_SINGULAR_DRIFT:expected={settled}:actual={float(actual)}")


def materialize_real_head_svd_custody(centered_head_weights: Any, pinned_weights_sha256: str) -> dict[str, Any]:
    """Materialize/hash the settled rank-4 vectors from already-custodied real head weights.

    This helper is for governed ``measure`` wiring only.  ``plan`` and ``self-test`` never
    call it and therefore never claim a fresh scientific SVD measurement.
    """
    import numpy as np

    weights = np.ascontiguousarray(np.asarray(centered_head_weights, dtype=np.float32))
    if weights.ndim < 2 or weights.shape[0] != 5:
        raise HarnessRefusal(f"BLOCKED_REAL_HEAD_WEIGHT_SHAPE:{weights.shape}")
    matrix = weights.reshape(5, -1).astype(np.float64)
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    u, vh = deterministic_svd_signs(u, vh)
    verify_settled_singular_values(singular)
    reconstructed = (u[:, :4] * singular[:4]) @ vh[:4]
    vector_bytes = np.ascontiguousarray(vh[:4].astype("<f8")).tobytes()
    left_bytes = np.ascontiguousarray(u[:, :4].astype("<f8")).tobytes()
    return {
        "schema": "pool_channel_real_head_svd_custody.v1",
        "source": "pinned frozen SegNet centered segmentation head weights",
        "pinned_segnet_weights_sha256": pinned_weights_sha256,
        "centered_head_weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        "right_singular_vectors_sha256": hashlib.sha256(vector_bytes).hexdigest(),
        "left_singular_vectors_u4_sha256": hashlib.sha256(left_bytes).hexdigest(),
        "projection_law": "raw U4.T @ centered logits; singular values are not divided out",
        "settled_singular_values": list(SETTLED_SINGULAR_VALUES),
        "rank4_reconstruction_max_abs_residual": float(np.max(np.abs(reconstructed - matrix))),
        "sign_convention": "largest-absolute right-vector coordinate is positive; first index breaks ties",
    }


def orthogonal_range_kernel_projectors(matrix: Any, tolerance: float | None = None) -> tuple[Any, Any]:
    """Return projectors onto range(A^T) and ker(A) for a small explicit matrix A."""
    import numpy as np

    a = np.asarray(matrix, dtype=np.float64)
    _, singular, vh = np.linalg.svd(a, full_matrices=True)
    tol = tolerance
    if tol is None:
        tol = max(a.shape, default=1) * np.finfo(np.float64).eps * (singular[0] if singular.size else 0.0)
    rank = int(np.count_nonzero(singular > tol))
    basis = vh[:rank].T
    p_range = basis @ basis.T
    p_kernel = np.eye(a.shape[1], dtype=np.float64) - p_range
    return p_range, p_kernel


def actuator_gram(jacobian: Any, coordinate_metric: Any) -> Any:
    """Compute ``J H^-1 J^T`` after a strict symmetric-positive-definite check."""
    import numpy as np

    j = np.asarray(jacobian, dtype=np.float64)
    h = np.asarray(coordinate_metric, dtype=np.float64)
    if j.ndim != 2 or h.shape != (j.shape[1], j.shape[1]):
        raise ValueError("J/H dimensions are incompatible")
    if not np.allclose(h, h.T, rtol=0.0, atol=1e-12):
        raise HarnessRefusal("BLOCKED_COORDINATE_METRIC_NOT_SYMMETRIC")
    try:
        np.linalg.cholesky(h)
        solved = np.linalg.solve(h, j.T)
    except np.linalg.LinAlgError as exc:
        raise HarnessRefusal("BLOCKED_COORDINATE_METRIC_NOT_POSITIVE_DEFINITE") from exc
    return j @ solved


@dataclass
class StackedJacobianStats:
    """Streaming sufficient statistics for rows of a real stacked spatial Jacobian."""

    deployable_coordinate_count: int
    provenance: str
    coordinate_metric: Any = None
    _k: Any = None
    _row_norm_fourth_sum: float = 0.0
    _n_rows: int = 0
    _whitener: Any = None
    _metric_sha256: str | None = None

    def add(self, jacobian_rows: Any) -> None:
        import numpy as np

        rows = np.asarray(jacobian_rows, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows[None]
        if rows.ndim != 2 or rows.shape[1] != self.deployable_coordinate_count:
            raise ValueError("Jacobian rows have the wrong deployable-coordinate dimension")
        if self._whitener is None:
            metric = (
                np.eye(self.deployable_coordinate_count, dtype=np.float64)
                if self.coordinate_metric is None
                else np.asarray(self.coordinate_metric, dtype=np.float64)
            )
            if metric.shape != (self.deployable_coordinate_count, self.deployable_coordinate_count):
                raise HarnessRefusal("BLOCKED_COORDINATE_METRIC_SHAPE")
            if not np.allclose(metric, metric.T, rtol=0.0, atol=1e-12):
                raise HarnessRefusal("BLOCKED_COORDINATE_METRIC_NOT_SYMMETRIC")
            try:
                chol = np.linalg.cholesky(metric)
            except np.linalg.LinAlgError as exc:
                raise HarnessRefusal("BLOCKED_COORDINATE_METRIC_NOT_POSITIVE_DEFINITE") from exc
            # H = L L^T, so rows must be right-multiplied by L^-T:
            # (J L^-T)(J L^-T)^T = J H^-1 J^T.
            self._whitener = np.linalg.inv(chol.T)
            self._metric_sha256 = hashlib.sha256(np.ascontiguousarray(metric.astype("<f8")).tobytes()).hexdigest()
        rows = rows @ self._whitener
        if self._k is None:
            self._k = np.zeros((self.deployable_coordinate_count, self.deployable_coordinate_count), dtype=np.float64)
        self._k += rows.T @ rows
        row_norm_sq = np.einsum("ij,ij->i", rows, rows)
        self._row_norm_fourth_sum += float(row_norm_sq @ row_norm_sq)
        self._n_rows += int(rows.shape[0])

    def merge_summary(self, summary: Mapping[str, Any]) -> None:
        import numpy as np

        if summary.get("schema") != "stacked_spatial_jacobian_sufficient_stats.v1":
            raise HarnessRefusal("BLOCKED_STACKED_STATS_SCHEMA")
        if int(summary.get("deployable_coordinate_count", -1)) != self.deployable_coordinate_count:
            raise HarnessRefusal("BLOCKED_STACKED_STATS_COORDINATE_DRIFT")
        if self._metric_sha256 is None:
            metric = (
                np.eye(self.deployable_coordinate_count, dtype=np.float64)
                if self.coordinate_metric is None
                else np.asarray(self.coordinate_metric, dtype=np.float64)
            )
            self._metric_sha256 = hashlib.sha256(np.ascontiguousarray(metric.astype("<f8")).tobytes()).hexdigest()
        if summary.get("coordinate_metric_sha256") != self._metric_sha256:
            raise HarnessRefusal("BLOCKED_STACKED_STATS_METRIC_DRIFT")
        k = np.asarray(summary.get("K"), dtype=np.float64)
        if k.shape != (self.deployable_coordinate_count, self.deployable_coordinate_count):
            raise HarnessRefusal("BLOCKED_STACKED_STATS_K_SHAPE")
        if self._k is None:
            self._k = np.zeros_like(k)
        self._k += k
        self._row_norm_fourth_sum += float(summary["sum_row_norm_fourth"])
        self._n_rows += int(summary["n_spatial_rows"])

    def summary(self) -> dict[str, Any]:
        import numpy as np

        if self._k is None:
            raise HarnessRefusal("BLOCKED_EMPTY_STACKED_JACOBIAN")
        trace_k2 = float(np.trace(self._k @ self._k))
        cross = trace_k2 - self._row_norm_fourth_sum
        scale = max(1.0, abs(trace_k2), abs(self._row_norm_fourth_sum))
        if cross < -1e-11 * scale:
            raise HarnessRefusal("BLOCKED_STACKED_GRAM_NUMERIC_INCONSISTENCY")
        cross = max(0.0, cross)
        return {
            "schema": "stacked_spatial_jacobian_sufficient_stats.v1",
            "provenance": self.provenance,
            "n_spatial_rows": self._n_rows,
            "deployable_coordinate_count": self.deployable_coordinate_count,
            "coordinate_metric": "H_u; rows prewhitened so Gram equals J H_u^-1 J^T",
            "coordinate_metric_sha256": self._metric_sha256,
            "K": self._k.tolist(),
            "trace_K_squared": trace_k2,
            "sum_row_norm_fourth": self._row_norm_fourth_sum,
            "cross_location_off_diagonal_energy": cross,
            "authority_requires": (
                "rows from rounded central finite differences through canonical render/R/frozen CPU Torch SegNet"
            ),
        }


def select_dyadic_finite_difference_step(
    base_coordinates: Any,
    direction: Any,
    render_uint8: Callable[[Any], Any],
    dyadic_steps: Iterable[float],
    lower: float,
    upper: float,
) -> tuple[float, Any, Any]:
    """Select the smallest feasible dyadic h changing uint8 output by at most one LSB."""
    import numpy as np

    base = np.asarray(base_coordinates, dtype=np.float64)
    delta = np.asarray(direction, dtype=np.float64)

    def checked_render(coordinates: Any) -> Any:
        rendered = np.asarray(render_uint8(coordinates))
        if rendered.dtype != np.uint8:
            raise HarnessRefusal(f"BLOCKED_RENDER_NOT_UINT8:{rendered.dtype}")
        return rendered.astype(np.int16)

    baseline = checked_render(base)
    for h in sorted({float(x) for x in dyadic_steps if float(x) > 0.0}):
        plus_coords, minus_coords = base + h * delta, base - h * delta
        if np.any(plus_coords < lower) or np.any(plus_coords > upper):
            continue
        if np.any(minus_coords < lower) or np.any(minus_coords > upper):
            continue
        plus = checked_render(plus_coords)
        minus = checked_render(minus_coords)
        if plus.shape != baseline.shape or minus.shape != baseline.shape:
            raise HarnessRefusal("BLOCKED_RENDER_SHAPE_DRIFT_DURING_FINITE_DIFFERENCE")
        plus_change = int(np.max(np.abs(plus - baseline), initial=0))
        minus_change = int(np.max(np.abs(minus - baseline), initial=0))
        changed = bool(np.any(plus != baseline) or np.any(minus != baseline))
        if changed and plus_change <= 1 and minus_change <= 1:
            return h, plus, minus
    raise HarnessRefusal("BLOCKED_NO_DYADIC_STEP_WITHIN_ONE_UINT8_LSB")


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _load_custodied_json(
    candidate: Mapping[str, Any], path_field: str, sha_field: str, blocker: str
) -> tuple[Path, dict[str, Any], str]:
    expected_sha = candidate.get(sha_field)
    if not _is_sha256(expected_sha):
        raise HarnessRefusal(f"{blocker}_SHA256_SCHEMA")
    path = Path(str(candidate.get(path_field, ""))).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise HarnessRefusal(f"{blocker}_CUSTODY")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessRefusal(f"{blocker}_SCHEMA") from exc
    if not isinstance(payload, dict):
        raise HarnessRefusal(f"{blocker}_SCHEMA")
    return path, payload, expected_sha


def _validate_exact_cell_attribution(
    candidate: Mapping[str, Any],
    row: Mapping[str, Any],
    canonical_mask_sha256: str,
    canonical_mask_count: int,
) -> dict[str, Any]:
    """Bind an R-D point to completed geometry for its exact 48-table cell."""

    _, run_wrapper, run_file_sha = _load_custodied_json(
        candidate,
        "geometry_run_contract_path",
        "geometry_run_contract_sha256",
        "BLOCKED_RD_GEOMETRY_RUN_CONTRACT",
    )
    contract = run_wrapper.get("contract")
    if run_wrapper.get("schema") != SCHEMA or not isinstance(contract, dict):
        raise HarnessRefusal("BLOCKED_RD_GEOMETRY_RUN_CONTRACT_SCHEMA")
    contract_sha = hashlib.sha256(canonical_json_bytes(contract)).hexdigest()
    if run_wrapper.get("contract_sha256") != contract_sha:
        raise HarnessRefusal("BLOCKED_RD_GEOMETRY_RUN_CONTRACT_CONTENT_HASH")
    expected_contract = {
        "schema": SCHEMA,
        "mode": "MEASURE_REAL_FULL_PATH_CHAIN",
        "pair_indices": list(range(600)),
        "segnet_batch_size": 32,
        "liveness_only": False,
        "requested_finding_mode": True,
        "harness_sha256": sha256_file(Path(__file__).resolve()),
    }
    if any(contract.get(key) != value for key, value in expected_contract.items()):
        raise HarnessRefusal("BLOCKED_RD_GEOMETRY_RUN_CONTRACT_NOT_N600_FINDING_AUTHORITY")
    input_sha = contract.get("input_sha256")
    if not isinstance(input_sha, Mapping) or any(
        input_sha.get(key) != DEFAULT_INPUTS[key][2] for key in ("bank", "gt_n600")
    ):
        raise HarnessRefusal("BLOCKED_RD_GEOMETRY_RUN_INPUT_CUSTODY")

    _, stage, stage_file_sha = _load_custodied_json(
        candidate,
        "geometry_stage_path",
        "geometry_stage_sha256",
        "BLOCKED_RD_GEOMETRY_STAGE",
    )
    expected_stage = {
        "schema": CHECKPOINT_SCHEMA,
        "stage": "geometry",
        "stage_index": STAGES.index("geometry"),
        "contract_sha256": contract_sha,
        "complete": True,
    }
    if any(stage.get(key) != value for key, value in expected_stage.items()):
        raise HarnessRefusal("BLOCKED_RD_GEOMETRY_STAGE_NOT_CONTRACT_BOUND")
    stage_payload = stage.get("payload")
    if not isinstance(stage_payload, dict) or stage.get("payload_sha256") != canonical_payload_sha256(stage_payload):
        raise HarnessRefusal("BLOCKED_RD_GEOMETRY_STAGE_PAYLOAD_CUSTODY")
    if (
        stage_payload.get("expected_pair_indices") != list(range(600))
        or stage_payload.get("finding_eligible") is not True
    ):
        raise HarnessRefusal("BLOCKED_RD_GEOMETRY_STAGE_NOT_N600_FINDING_ELIGIBLE")
    cells = stage_payload.get("cells")
    if not isinstance(cells, Mapping) or not isinstance(cells.get(row["row_id"]), Mapping):
        raise HarnessRefusal("BLOCKED_RD_GEOMETRY_STAGE_CELL_MISSING")
    cell = dict(cells[row["row_id"]])
    expected_cell = {
        "schema": "pool_channel_jacobian_rd.exact_cell_geometry.v1",
        "row_id": row["row_id"],
        "pool": row["pool"],
        "head_direction": row["head_direction"],
        "settled_singular_value": row["settled_singular_value"],
        "path": row["path"],
        "resize_component": row["resize_component"],
        "axis": CPU_AXIS,
        "pair_count": 600,
        "segnet_batch_size": 32,
        "finding_eligible": True,
        "same_operating_point": True,
        "path_intervention_exact": True,
        "path_recomposition_pass": True,
        "resize_intervention_exact": True,
        "range_kernel_recomposition_pass": True,
        "canonical_pool_mask_sha256": canonical_mask_sha256,
        "canonical_pool_mask_count": canonical_mask_count,
        "target_mask_sha256": candidate["target_mask_sha256"],
        "g_act_measured": True,
    }
    if any(cell.get(key) != value for key, value in expected_cell.items()):
        raise HarnessRefusal("BLOCKED_RD_GEOMETRY_CELL_SEMANTIC_BINDING")
    head = cell.get("head_direction_custody")
    expected_head = {
        "schema": "pool_channel_jacobian_rd.head_direction_custody.v1",
        "head_direction": row["head_direction"],
        "settled_singular_value": row["settled_singular_value"],
        "segnet_weights_sha256": SEGNET_WEIGHTS_SHA256,
        "rank4_verified": True,
        "sign_fixed": True,
    }
    if (
        not isinstance(head, Mapping)
        or any(head.get(key) != value for key, value in expected_head.items())
        or not _is_sha256(head.get("head_direction_sha256"))
    ):
        raise HarnessRefusal("BLOCKED_RD_HEAD_DIRECTION_CUSTODY")
    g_act = cell.get("g_act")
    if not isinstance(g_act, Mapping) or not g_act:
        raise HarnessRefusal("BLOCKED_RD_G_ACT_SCHEMA")
    try:
        trace_squared = float(g_act["trace_G_act_squared"])
        n_rows = int(g_act["n_spatial_rows"])
        cross_energy = float(cell["cross_location_off_diagonal_energy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HarnessRefusal("BLOCKED_RD_G_ACT_SCHEMA") from exc
    if (
        not math.isfinite(trace_squared)
        or trace_squared < 0.0
        or n_rows <= 0
        or not _is_sha256(g_act.get("coordinate_metric_sha256"))
        or not math.isfinite(cross_energy)
        or cross_energy < 0.0
        or cell.get("g_act_receipt_sha256") != hashlib.sha256(canonical_json_bytes(dict(g_act))).hexdigest()
    ):
        raise HarnessRefusal("BLOCKED_RD_G_ACT_CUSTODY")
    path_attribution = cell.get("path_attribution")
    expected_path = {
        "schema": "pool_channel_jacobian_rd.exact_path_attribution.v1",
        "path": row["path"],
        "exact_intervention": True,
        "same_operating_point": True,
        "recomposition_pass": True,
    }
    if not isinstance(path_attribution, Mapping) or any(
        path_attribution.get(key) != value for key, value in expected_path.items()
    ):
        raise HarnessRefusal("BLOCKED_RD_PATH_ATTRIBUTION_NOT_EXACT")
    resize_projection = cell.get("resize_projection")
    expected_resize = {
        "schema": "pool_channel_jacobian_rd.exact_resize_projection.v1",
        "resize_component": row["resize_component"],
        "exact_intervention": True,
        "same_operating_point": True,
        "orthogonal_decomposition_pass": True,
        "recomposition_pass": True,
    }
    if not isinstance(resize_projection, Mapping) or any(
        resize_projection.get(key) != value for key, value in expected_resize.items()
    ):
        raise HarnessRefusal("BLOCKED_RD_RESIZE_PROJECTION_NOT_EXACT")

    cell_sha = hashlib.sha256(canonical_json_bytes(cell)).hexdigest()
    _, intervention, intervention_file_sha = _load_custodied_json(
        candidate,
        "cell_intervention_receipt_path",
        "cell_intervention_receipt_sha256",
        "BLOCKED_RD_CELL_INTERVENTION_RECEIPT",
    )
    expected_intervention = {
        "schema": "pool_channel_jacobian_rd.exact_cell_intervention.v1",
        "row_id": row["row_id"],
        "pool": row["pool"],
        "head_direction": row["head_direction"],
        "settled_singular_value": row["settled_singular_value"],
        "path": row["path"],
        "resize_component": row["resize_component"],
        "geometry_run_contract_sha256": run_file_sha,
        "geometry_stage_sha256": stage_file_sha,
        "cell_geometry_sha256": cell_sha,
        "canonical_pool_mask_sha256": canonical_mask_sha256,
        "canonical_pool_mask_count": canonical_mask_count,
        "target_mask_sha256": candidate["target_mask_sha256"],
        "baseline_archive_sha256": candidate["baseline_archive_sha256"],
        "archive_sha256": candidate["archive_sha256"],
        "baseline_argmax_sha256": candidate["baseline_argmax_sha256"],
        "candidate_argmax_sha256": candidate["candidate_argmax_sha256"],
        "gt_argmax_sha256": candidate["gt_argmax_sha256"],
        "axis": CPU_AXIS,
        "pair_count": 600,
        "segnet_batch_size": 32,
        "receiver_closed": True,
        "exact_byte_close": True,
        "same_operating_point": True,
        "path_intervention_exact": True,
        "path_recomposition_pass": True,
        "resize_intervention_exact": True,
        "range_kernel_recomposition_pass": True,
        "score_claim": False,
        "promotable": False,
    }
    if any(intervention.get(key) != value for key, value in expected_intervention.items()):
        raise HarnessRefusal("BLOCKED_RD_CELL_INTERVENTION_NOT_CONTENT_BOUND")
    return {
        "geometry_run_contract": run_wrapper,
        "geometry_run_contract_sha256": run_file_sha,
        "geometry_stage": stage,
        "geometry_stage_sha256": stage_file_sha,
        "cell_geometry": cell,
        "cell_geometry_sha256": cell_sha,
        "cell_intervention_receipt": intervention,
        "cell_intervention_receipt_sha256": intervention_file_sha,
    }


def admit_rd_point(candidate: Mapping[str, Any]) -> dict[str, Any]:
    """Admit only a receiver-closed n600 point re-derived from custodied argmax arrays.

    The receipt is not allowed to declare its own scientific values.  This gate hashes and
    loads the frozen-GT, baseline, candidate, and target-mask arrays, re-counts every claimed
    transition, and then requires the receiver/scorer receipt to bind those derived values to
    the exact archive member bytes.
    """
    required = {
        "row_id",
        "archive_path",
        "archive_bytes",
        "archive_sha256",
        "payload_bytes",
        "payload_sha256",
        "baseline_archive_path",
        "baseline_archive_bytes",
        "baseline_archive_sha256",
        "parseback_receipt_path",
        "parseback_receipt_sha256",
        "baseline_argmax_path",
        "baseline_argmax_sha256",
        "candidate_argmax_path",
        "candidate_argmax_sha256",
        "gt_argmax_path",
        "gt_argmax_sha256",
        "target_mask_path",
        "target_mask_sha256",
        "geometry_run_contract_path",
        "geometry_run_contract_sha256",
        "geometry_stage_path",
        "geometry_stage_sha256",
        "cell_intervention_receipt_path",
        "cell_intervention_receipt_sha256",
        "scorer_weights_sha256",
        "axis",
        "pair_count",
        "segnet_batch_size",
        "baseline_d_seg",
        "d_seg",
        "rate_term",
        "baseline_flip_count",
        "candidate_flip_count",
        "target_fixes",
        "new_bad",
        "non_target_transitions",
    }
    missing = sorted(required - candidate.keys())
    if missing:
        raise HarnessRefusal(f"BLOCKED_RD_CUSTODY_MISSING:{','.join(missing)}")
    row = next((item for item in make_rows() if item["row_id"] == candidate["row_id"]), None)
    if row is None:
        raise HarnessRefusal("BLOCKED_RD_UNKNOWN_CELL")
    if candidate.get("score_claim") not in (None, False) or candidate.get("promotable") not in (None, False):
        raise HarnessRefusal("BLOCKED_RD_COMPONENT_POINT_CANNOT_CLAIM_SCORE_OR_PROMOTION")
    integer_fields = (
        "archive_bytes",
        "payload_bytes",
        "baseline_archive_bytes",
        "pair_count",
        "segnet_batch_size",
        "baseline_flip_count",
        "candidate_flip_count",
        "target_fixes",
        "new_bad",
        "non_target_transitions",
    )
    for field in integer_fields:
        if type(candidate[field]) is not int or candidate[field] < 0:
            raise HarnessRefusal(f"BLOCKED_RD_INVALID_INTEGER:{field}")
    archive = Path(str(candidate["archive_path"]))
    if not archive.is_file():
        raise HarnessRefusal("BLOCKED_RD_ARCHIVE_MISSING")
    if candidate["archive_bytes"] <= 0 or archive.stat().st_size != candidate["archive_bytes"]:
        raise HarnessRefusal("BLOCKED_RD_ARCHIVE_SIZE_MISMATCH")
    archive_sha = sha256_file(archive)
    if archive_sha != str(candidate["archive_sha256"]):
        raise HarnessRefusal("BLOCKED_RD_ARCHIVE_SHA256_MISMATCH")
    try:
        with zipfile.ZipFile(archive) as zf:
            members = zf.infolist()
            if len(members) != 1 or members[0].filename != "0.bin" or members[0].is_dir():
                raise HarnessRefusal("BLOCKED_RD_ARCHIVE_GRAMMAR")
            blob = zf.read("0.bin")
            blob_sha = hashlib.sha256(blob).hexdigest()
    except (OSError, zipfile.BadZipFile) as exc:
        raise HarnessRefusal("BLOCKED_RD_ARCHIVE_NOT_VALID_ZIP") from exc
    if len(blob) != candidate["payload_bytes"] or blob_sha != candidate["payload_sha256"]:
        raise HarnessRefusal("BLOCKED_RD_PAYLOAD_CUSTODY")
    baseline_archive = Path(str(candidate["baseline_archive_path"]))
    if (
        not baseline_archive.is_file()
        or baseline_archive.stat().st_size != candidate["baseline_archive_bytes"]
        or sha256_file(baseline_archive) != candidate["baseline_archive_sha256"]
    ):
        raise HarnessRefusal("BLOCKED_RD_BASELINE_ARCHIVE_CUSTODY")
    try:
        with zipfile.ZipFile(baseline_archive) as zf:
            baseline_members = zf.infolist()
            if len(baseline_members) != 1 or baseline_members[0].filename != "0.bin" or baseline_members[0].is_dir():
                raise HarnessRefusal("BLOCKED_RD_BASELINE_ARCHIVE_GRAMMAR")
            baseline_blob_sha = hashlib.sha256(zf.read("0.bin")).hexdigest()
    except (OSError, zipfile.BadZipFile) as exc:
        raise HarnessRefusal("BLOCKED_RD_BASELINE_ARCHIVE_NOT_VALID_ZIP") from exc
    if candidate["axis"] != CPU_AXIS:
        raise HarnessRefusal("BLOCKED_RD_NONAUTHORITY_AXIS")
    if candidate["pair_count"] != 600 or candidate["segnet_batch_size"] != 32:
        raise HarnessRefusal("BLOCKED_RD_NOT_N600_BATCH32")
    if candidate["scorer_weights_sha256"] != SEGNET_WEIGHTS_SHA256:
        raise HarnessRefusal("BLOCKED_RD_SCORER_WEIGHTS_DRIFT")

    import numpy as np

    expected_shape = (600, 384, 512)

    def custodied_array(path_field: str, sha_field: str, *, mask: bool = False) -> Any:
        path = Path(str(candidate[path_field]))
        if not path.is_file() or sha256_file(path) != candidate[sha_field]:
            raise HarnessRefusal(f"BLOCKED_RD_ARRAY_CUSTODY:{path_field}")
        try:
            array = np.load(path, mmap_mode="r", allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise HarnessRefusal(f"BLOCKED_RD_ARRAY_SCHEMA:{path_field}") from exc
        if tuple(array.shape) != expected_shape:
            raise HarnessRefusal(f"BLOCKED_RD_ARRAY_SHAPE:{path_field}")
        if mask:
            if array.dtype != np.bool_:
                raise HarnessRefusal(f"BLOCKED_RD_ARRAY_DTYPE:{path_field}")
        elif array.dtype != np.uint8:
            raise HarnessRefusal(f"BLOCKED_RD_ARRAY_DTYPE:{path_field}")
        return array

    baseline = custodied_array("baseline_argmax_path", "baseline_argmax_sha256")
    predicted = custodied_array("candidate_argmax_path", "candidate_argmax_sha256")
    truth = custodied_array("gt_argmax_path", "gt_argmax_sha256")
    target_mask = custodied_array("target_mask_path", "target_mask_sha256", mask=True)
    tools_dir = Path(__file__).resolve().parent
    if str(tools_dir) not in sys.path:
        sys.path.insert(0, str(tools_dir))
    try:
        from necessity_dseg_calibration import _strata_masks
    except (ImportError, OSError) as exc:
        raise HarnessRefusal(f"BLOCKED_RD_CANONICAL_POOL_DEPENDENCY:{type(exc).__name__}:{exc}") from exc

    derived_counts = dict.fromkeys(integer_fields[-5:], 0)
    canonical_mask_count = 0

    def validated_canonical_mask_chunks() -> Iterable[Any]:
        nonlocal canonical_mask_count
        for start in range(0, expected_shape[0], 32):
            stop = min(expected_shape[0], start + 32)
            baseline_chunk = np.asarray(baseline[start:stop])
            predicted_chunk = np.asarray(predicted[start:stop])
            truth_chunk = np.asarray(truth[start:stop])
            target_chunk = np.asarray(target_mask[start:stop])
            if any(int(np.max(chunk, initial=0)) > 4 for chunk in (baseline_chunk, predicted_chunk, truth_chunk)):
                raise HarnessRefusal("BLOCKED_RD_ARGMAX_CLASS_DOMAIN")
            expected_masks = []
            for frame in range(stop - start):
                pools = canonical_pool_masks_from_labels(truth_chunk[frame], baseline_chunk[frame], _strata_masks)
                expected_masks.append(pools[row["pool"]])
            expected_chunk = np.stack(expected_masks)
            if not np.array_equal(target_chunk, expected_chunk):
                raise HarnessRefusal("BLOCKED_RD_TARGET_MASK_NOT_CANONICAL_ROW_POOL")
            canonical_mask_count += int(np.count_nonzero(expected_chunk))
            baseline_bad = baseline_chunk != truth_chunk
            candidate_bad = predicted_chunk != truth_chunk
            changed = predicted_chunk != baseline_chunk
            derived_counts["baseline_flip_count"] += int(np.count_nonzero(baseline_bad))
            derived_counts["candidate_flip_count"] += int(np.count_nonzero(candidate_bad))
            derived_counts["target_fixes"] += int(np.count_nonzero(baseline_bad & ~candidate_bad & expected_chunk))
            derived_counts["new_bad"] += int(np.count_nonzero(~baseline_bad & candidate_bad))
            derived_counts["non_target_transitions"] += int(np.count_nonzero(changed & ~expected_chunk))
            yield expected_chunk

    canonical_mask_sha256 = canonical_pool_mask_digest(row["pool"], expected_shape, validated_canonical_mask_chunks())
    if canonical_mask_count == 0:
        raise HarnessRefusal("BLOCKED_RD_EMPTY_TARGET_MASK")
    if any(candidate[field] != value for field, value in derived_counts.items()):
        raise HarnessRefusal("BLOCKED_RD_COUNTS_NOT_DERIVED_FROM_CUSTODIED_ARGMAX")
    total_pixels = math.prod(expected_shape)
    derived_baseline_d_seg = derived_counts["baseline_flip_count"] / total_pixels
    derived_d_seg = derived_counts["candidate_flip_count"] / total_pixels
    try:
        baseline_d_seg = float(candidate["baseline_d_seg"])
        d_seg = float(candidate["d_seg"])
        rate_term = float(candidate["rate_term"])
    except (TypeError, ValueError) as exc:
        raise HarnessRefusal("BLOCKED_RD_INVALID_DISTORTION_OR_RATE") from exc
    if not math.isclose(baseline_d_seg, derived_baseline_d_seg, rel_tol=0.0, abs_tol=1e-15):
        raise HarnessRefusal("BLOCKED_RD_BASELINE_DSEG_NOT_DERIVED")
    if not math.isclose(d_seg, derived_d_seg, rel_tol=0.0, abs_tol=1e-15):
        raise HarnessRefusal("BLOCKED_RD_DSEG_NOT_DERIVED")
    expected_rate = 25.0 * candidate["archive_bytes"] / 37_545_489
    baseline_rate = 25.0 * candidate["baseline_archive_bytes"] / 37_545_489
    delta_archive_bytes = candidate["archive_bytes"] - candidate["baseline_archive_bytes"]
    delta_d_seg = derived_d_seg - derived_baseline_d_seg
    seg_score_units_bought = 100.0 * (derived_baseline_d_seg - derived_d_seg)
    if not math.isclose(rate_term, expected_rate, rel_tol=0.0, abs_tol=1e-12):
        raise HarnessRefusal("BLOCKED_RD_RATE_NOT_DERIVED_FROM_EXACT_BYTES")

    attribution = _validate_exact_cell_attribution(
        candidate,
        row,
        canonical_mask_sha256,
        canonical_mask_count,
    )

    receipt_path = Path(str(candidate["parseback_receipt_path"]))
    if not receipt_path.is_file() or sha256_file(receipt_path) != str(candidate["parseback_receipt_sha256"]):
        raise HarnessRefusal("BLOCKED_RD_PARSEBACK_RECEIPT_CUSTODY")
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessRefusal("BLOCKED_RD_PARSEBACK_RECEIPT_SCHEMA") from exc
    bound = {
        "schema": "pool_channel_jacobian_rd.receiver_scorer_receipt.v1",
        "parseback_pass": True,
        "receiver_closed": True,
        "scorer_replay_pass": True,
        "component_scope": "d_seg_only",
        "score_claim": False,
        "promotable": False,
        "row_id": candidate["row_id"],
        "archive_sha256": archive_sha,
        "archive_bytes": candidate["archive_bytes"],
        "archive_member": "0.bin",
        "archive_member_sha256": blob_sha,
        "archive_member_bytes": len(blob),
        "baseline_archive_sha256": candidate["baseline_archive_sha256"],
        "baseline_archive_bytes": candidate["baseline_archive_bytes"],
        "baseline_archive_member_sha256": baseline_blob_sha,
        "axis": CPU_AXIS,
        "pair_count": 600,
        "segnet_batch_size": 32,
        "scorer_weights_sha256": SEGNET_WEIGHTS_SHA256,
        "baseline_argmax_sha256": candidate["baseline_argmax_sha256"],
        "candidate_argmax_sha256": candidate["candidate_argmax_sha256"],
        "gt_argmax_sha256": candidate["gt_argmax_sha256"],
        "target_mask_sha256": candidate["target_mask_sha256"],
        "canonical_pool_mask_sha256": canonical_mask_sha256,
        "canonical_pool_mask_count": canonical_mask_count,
        "geometry_run_contract_sha256": attribution["geometry_run_contract_sha256"],
        "geometry_stage_sha256": attribution["geometry_stage_sha256"],
        "cell_geometry_sha256": attribution["cell_geometry_sha256"],
        "cell_intervention_receipt_sha256": attribution["cell_intervention_receipt_sha256"],
        "baseline_d_seg": derived_baseline_d_seg,
        "d_seg": derived_d_seg,
        "rate_term": expected_rate,
        "baseline_rate_term": baseline_rate,
        "delta_archive_bytes": delta_archive_bytes,
        "delta_d_seg": delta_d_seg,
        "seg_score_units_bought": seg_score_units_bought,
        **derived_counts,
    }
    if any(receipt.get(key) != value for key, value in bound.items()):
        raise HarnessRefusal("BLOCKED_RD_PARSEBACK_RECEIPT_NOT_CONTENT_BOUND")
    admitted = dict(candidate)
    admitted["parseback_receipt"] = receipt
    admitted["archive_member_sha256"] = blob_sha
    admitted["baseline_archive_member_sha256"] = baseline_blob_sha
    admitted["canonical_pool_mask_sha256"] = canonical_mask_sha256
    admitted["canonical_pool_mask_count"] = canonical_mask_count
    admitted["exact_cell_attribution"] = attribution
    admitted["derived_counts"] = derived_counts
    admitted["baseline_rate_term"] = baseline_rate
    admitted["delta_archive_bytes"] = delta_archive_bytes
    admitted["delta_d_seg"] = delta_d_seg
    admitted["seg_score_units_bought"] = seg_score_units_bought
    admitted["seg_score_units_bought_per_added_byte"] = (
        seg_score_units_bought / delta_archive_bytes if delta_archive_bytes > 0 else None
    )
    admitted["authority"] = "component d_seg R-D only; not an overall contest-score verdict"
    admitted["score_claim"] = False
    admitted["promotable"] = False
    return admitted


class StageStore:
    """Atomic, content-bound, preserved stage checkpoints for governed measurement."""

    def __init__(self, output_dir: Path, contract: Mapping[str, Any], resume: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.contract_sha256 = hashlib.sha256(canonical_json_bytes(contract)).hexdigest()
        self.resume = resume
        contract_path = self.output_dir / "run_contract.json"
        payload = {"schema": SCHEMA, "contract_sha256": self.contract_sha256, "contract": dict(contract)}
        if contract_path.exists():
            existing = json.loads(contract_path.read_text())
            if existing != payload:
                raise HarnessRefusal("BLOCKED_INCOMPATIBLE_RESUME_CONTRACT")
            if not resume:
                raise HarnessRefusal("BLOCKED_EXISTING_RUN_REQUIRES_EXPLICIT_RESUME")
        else:
            atomic_write_json(contract_path, payload)

    def _validated_stage(self, stage: str) -> dict[str, Any]:
        index = STAGES.index(stage)
        path = self.output_dir / f"stage_{index:02d}_{stage}.json"
        if not path.is_file():
            raise HarnessRefusal(f"BLOCKED_STAGE_PREDECESSOR_MISSING:{stage}")
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessRefusal(f"BLOCKED_STAGE_RECORD_INVALID:{stage}") from exc
        required = {
            "schema": CHECKPOINT_SCHEMA,
            "stage": stage,
            "stage_index": index,
            "contract_sha256": self.contract_sha256,
            "complete": True,
        }
        if any(record.get(key) != value for key, value in required.items()):
            raise HarnessRefusal(f"BLOCKED_STAGE_RECORD_CUSTODY:{stage}")
        if not isinstance(record.get("payload"), dict):
            raise HarnessRefusal(f"BLOCKED_STAGE_PAYLOAD_SCHEMA:{stage}")
        if record.get("payload_sha256") != canonical_payload_sha256(record["payload"]):
            raise HarnessRefusal(f"BLOCKED_STAGE_PAYLOAD_SHA256:{stage}")
        return record

    def _validated_pair(self, stage: str, pair_index: int) -> dict[str, Any]:
        path = self.output_dir / "pairs" / stage / f"pair_{pair_index:04d}.json"
        if not path.is_file():
            raise HarnessRefusal(f"BLOCKED_PAIR_CHECKPOINT_MISSING:{stage}:{pair_index}")
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessRefusal(f"BLOCKED_PAIR_CHECKPOINT_INVALID:{stage}:{pair_index}") from exc
        required = {
            "schema": "pool_channel_jacobian_rd.pair_checkpoint.v1",
            "contract_sha256": self.contract_sha256,
            "stage": stage,
            "pair_index": pair_index,
            "complete": True,
        }
        if any(record.get(key) != value for key, value in required.items()):
            raise HarnessRefusal(f"BLOCKED_PAIR_CHECKPOINT_CUSTODY:{stage}:{pair_index}")
        if not isinstance(record.get("payload"), dict):
            raise HarnessRefusal(f"BLOCKED_PAIR_PAYLOAD_SCHEMA:{stage}:{pair_index}")
        if record.get("payload_sha256") != canonical_payload_sha256(record["payload"]):
            raise HarnessRefusal(f"BLOCKED_PAIR_PAYLOAD_SHA256:{stage}:{pair_index}")
        return record

    def complete_stage(self, stage: str, payload: Mapping[str, Any]) -> Path:
        if stage not in STAGES:
            raise ValueError(f"unknown stage {stage}")
        index = STAGES.index(stage)
        if index:
            self._validated_stage(STAGES[index - 1])
        expected_pairs = payload.get("expected_pair_indices")
        if expected_pairs is not None:
            if not isinstance(expected_pairs, list) or any(type(x) is not int or x < 0 for x in expected_pairs):
                raise HarnessRefusal(f"BLOCKED_STAGE_PAIR_MEMBERSHIP_SCHEMA:{stage}")
            if len(expected_pairs) != len(set(expected_pairs)):
                raise HarnessRefusal(f"BLOCKED_STAGE_PAIR_MEMBERSHIP_DUPLICATE:{stage}")
            for pair_index in expected_pairs:
                self._validated_pair(stage, pair_index)
        path = self.output_dir / f"stage_{index:02d}_{stage}.json"
        preserved_payload = dict(payload)
        record = {
            "schema": CHECKPOINT_SCHEMA,
            "stage": stage,
            "stage_index": index,
            "contract_sha256": self.contract_sha256,
            "complete": True,
            "payload": preserved_payload,
            "payload_sha256": canonical_payload_sha256(preserved_payload),
        }
        if path.exists():
            existing = json.loads(path.read_text())
            if existing != record:
                raise HarnessRefusal(f"BLOCKED_PRESERVED_STAGE_CONFLICT:{stage}")
            if not self.resume:
                raise HarnessRefusal(f"BLOCKED_STAGE_ALREADY_COMPLETE:{stage}")
            return path
        atomic_write_json(path, record)
        manifest = {
            "schema": "pool_channel_jacobian_rd.stage_manifest.v1",
            "contract_sha256": self.contract_sha256,
            "stages": [
                {
                    "stage": name,
                    "complete": (self.output_dir / f"stage_{i:02d}_{name}.json").exists(),
                    "path": f"stage_{i:02d}_{name}.json",
                    "sha256": (
                        sha256_file(self.output_dir / f"stage_{i:02d}_{name}.json")
                        if (self.output_dir / f"stage_{i:02d}_{name}.json").is_file()
                        else None
                    ),
                }
                for i, name in enumerate(STAGES)
            ],
        }
        atomic_write_json(self.output_dir / "checkpoint_manifest.json", manifest)
        return path

    def write_pair_checkpoint(self, stage: str, pair_index: int, payload: Mapping[str, Any]) -> Path:
        """Preserve one atomic pair shard; compatible resume may verify but never replace it."""
        if stage not in STAGES[:-1]:
            raise ValueError(f"pair checkpoints are not valid for stage {stage}")
        if type(pair_index) is not int or pair_index < 0:
            raise ValueError("pair_index must be a non-negative integer")
        stage_index = STAGES.index(stage)
        if stage_index:
            self._validated_stage(STAGES[stage_index - 1])
        path = self.output_dir / "pairs" / stage / f"pair_{pair_index:04d}.json"
        preserved_payload = dict(payload)
        record = {
            "schema": "pool_channel_jacobian_rd.pair_checkpoint.v1",
            "contract_sha256": self.contract_sha256,
            "stage": stage,
            "pair_index": pair_index,
            "complete": True,
            "payload": preserved_payload,
            "payload_sha256": canonical_payload_sha256(preserved_payload),
        }
        if path.exists():
            existing = json.loads(path.read_text())
            if existing != record:
                raise HarnessRefusal(f"BLOCKED_PRESERVED_PAIR_CONFLICT:{stage}:{pair_index}")
            if not self.resume:
                raise HarnessRefusal(f"BLOCKED_PAIR_ALREADY_COMPLETE:{stage}:{pair_index}")
            return path
        atomic_write_json(path, record)
        return path


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    root = resolve_canonical_root(args.canonical_root)
    pointer_path = root / ".omx/state/canonical_frontier_pointer.json"
    pointer_before = read_dynamic_pointer_custody(pointer_path)
    custody: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for key, (relative, expected_size, expected_sha) in DEFAULT_INPUTS.items():
        override = getattr(args, key)
        path = Path(override).expanduser() if override else root / relative
        paths[key] = path
        custody[key] = validate_file_custody(path, expected_size, expected_sha)
    cache_identity = validate_cache_identity(paths["gt_n600"], paths["gt_n96"])
    pointer_after = read_dynamic_pointer_custody(pointer_path)
    if pointer_before != pointer_after:
        raise HarnessRefusal("BLOCKED_CANONICAL_POINTER_DRIFT_DURING_PLAN")
    return {
        "schema": SCHEMA,
        "mode": "BUILD_PLAN",
        "status": UNMEASURED,
        "authority": "BUILD_ONLY_NO_SCIENTIFIC_MEASUREMENT",
        "inputs": custody,
        "cache_identity": cache_identity,
        "canonical_pointer": {
            "pre_plan_custody": pointer_before,
            "post_plan_custody": pointer_after,
            "current_contest_cpu_score": pointer_after["current_contest_cpu_score"],
            "authority": "DYNAMIC_CANONICAL_POINTER_REDERIVED_THIS_PLAN",
            "pointer_delta": 0,
            "mutated": False,
        },
        "rows": make_rows(),
        "row_count": 48,
        "resize_facts": {
            "axis_aligned_zero_weight_camera_support_fraction": 0.226969,
            "full_nullspace_dimension_fraction": 0.80674,
            "prior_energy_fraction_note": (
                "approximately 52 percent was sample-specific measured energy and is not a harness constant"
            ),
        },
        "intrinsic_definition": "RATE-DOMINATED within {ker(A), sub-uint8-LSB, GT-flicker-band}",
        "extrinsic_definition": "coherent uint8-realizable rank-4 control in range(A)",
        "negative_language": "No component is called unreachable.",
        "score_claim": False,
        "promotable": False,
        "pointer_delta": 0,
        "launch_performed": False,
        "cost_usd": 0,
    }


def run_self_test() -> dict[str, Any]:
    import numpy as np

    rows = make_rows()
    flip = np.array([1, 1, 1, 0, 1], dtype=bool)
    saddle = np.array([1, 0, 0, 1, 0], dtype=bool)
    edge = np.array([1, 1, 0, 1, 1], dtype=bool)
    road_lane = np.array([1, 1, 0, 1, 0], dtype=bool)
    pools = assign_exclusive_pools(flip, saddle, edge, road_lane)
    assert [int(pools[p].sum()) for p in POOLS] == [1, 1, 2]

    labels = np.zeros((3, 3), dtype=np.uint8)
    predicted_labels = labels.copy()
    predicted_labels[np.arange(3), np.arange(3)] = 1

    def constructed_strata(_: Any) -> tuple[Any, Any, Any, Any]:
        local_saddle = np.zeros((3, 3), dtype=bool)
        local_saddle[1, 1] = True
        local_boundary = np.zeros((3, 3), dtype=bool)
        local_boundary[0, 0] = local_boundary[2, 2] = True
        local_pair = np.full((3, 3), 255, dtype=np.uint8)
        local_pair[0, 0], local_pair[2, 2] = 1, 2
        return local_saddle, local_boundary, local_pair, np.ones((3, 3), dtype=bool)

    def constructed_nearest_indices(boundary: Any) -> Any:
        coordinates = np.argwhere(np.asarray(boundary, dtype=bool))
        grid = np.moveaxis(np.indices(labels.shape), 0, -1)
        distance_squared = np.sum(
            (grid[:, :, None, :] - coordinates[None, None, :, :]) ** 2,
            axis=-1,
        )
        chosen = np.argmin(distance_squared, axis=-1)
        return np.moveaxis(coordinates[chosen], -1, 0)

    canonical_pools = canonical_pool_masks_from_labels(
        labels,
        predicted_labels,
        constructed_strata,
        constructed_nearest_indices,
    )
    assert [int(canonical_pools[name].sum()) for name in POOLS] == [1, 1, 1]
    pool_a_digest = canonical_pool_mask_digest(POOLS[0], labels.shape, [canonical_pools[POOLS[0]]])
    pool_c_digest = canonical_pool_mask_digest(POOLS[2], labels.shape, [canonical_pools[POOLS[2]]])
    assert pool_a_digest != pool_c_digest

    matrix = np.diag([4.0, 3.0, 2.0]) @ np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
    fixed_u, fixed_vh = deterministic_svd_signs(u, vh)
    assert np.allclose(fixed_u @ np.diag(singular) @ fixed_vh, matrix)
    assert all(row[int(np.argmax(np.abs(row)))] >= 0 for row in fixed_vh)

    resize = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])
    p_range, p_kernel = orthogonal_range_kernel_projectors(resize)
    assert np.allclose(p_range @ p_range, p_range)
    assert np.allclose(p_kernel @ p_kernel, p_kernel)
    assert np.allclose(p_range @ p_kernel, 0.0)
    assert np.allclose(resize @ p_kernel, 0.0)

    j_small = np.array([[1.0, 2.0], [0.5, -1.0]])
    h_small = np.diag([2.0, 4.0])
    assert np.allclose(actuator_gram(j_small, h_small), j_small @ np.linalg.inv(h_small) @ j_small.T)

    jacobian = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 0.25]])
    stats = StackedJacobianStats(2, "SELF_TEST_CONSTRUCTED_NON_SCIENTIFIC")
    stats.add(jacobian[:2])
    stats.add(jacobian[2:])
    summary = stats.summary()
    explicit = jacobian @ jacobian.T
    explicit_cross = float(np.sum(explicit * explicit) - np.sum(np.diag(explicit) ** 2))
    assert np.isclose(summary["cross_location_off_diagonal_energy"], explicit_cross)

    non_diagonal_metric = np.array([[2.0, 0.6], [0.6, 1.5]])
    metric_stats = StackedJacobianStats(
        2, "SELF_TEST_CONSTRUCTED_NON_SCIENTIFIC", coordinate_metric=non_diagonal_metric
    )
    metric_stats.add(jacobian[:1])
    metric_stats.add(jacobian[1:])
    metric_summary = metric_stats.summary()
    explicit_metric_gram = actuator_gram(jacobian, non_diagonal_metric)
    explicit_metric_cross = float(
        np.sum(explicit_metric_gram * explicit_metric_gram) - np.sum(np.diag(explicit_metric_gram) ** 2)
    )
    assert np.isclose(metric_summary["trace_K_squared"], np.trace(explicit_metric_gram @ explicit_metric_gram))
    assert np.isclose(metric_summary["cross_location_off_diagonal_energy"], explicit_metric_cross)

    try:
        admit_rd_point({})
    except HarnessRefusal as exc:
        assert str(exc).startswith("BLOCKED_RD_CUSTODY_MISSING")
    else:
        raise AssertionError("R-D admission accepted missing custody")

    try:
        _validate_execution_authority({}, Path(tempfile.gettempdir()))
    except HarnessRefusal as exc:
        assert str(exc) == "BLOCKED_EXECUTION_AUTHORITY_WRAPPER"
    else:
        raise AssertionError("execution accepted without durable authority")

    with tempfile.TemporaryDirectory(prefix="pool_channel_rd_selftest_") as temp:
        contract = {"mode": "SELF_TEST_NON_SCIENTIFIC", "pair_count": 0}
        store = StageStore(Path(temp), contract)
        store.complete_stage("custody", {"status": "STRUCTURAL_ONLY"})
        store.write_pair_checkpoint("baseline_byte_close", 0, {"status": "STRUCTURAL_ONLY"})
        resumed = StageStore(Path(temp), contract, resume=True)
        resumed.complete_stage("custody", {"status": "STRUCTURAL_ONLY"})
        resumed.write_pair_checkpoint("baseline_byte_close", 0, {"status": "STRUCTURAL_ONLY"})
        resumed.complete_stage("baseline_byte_close", {"expected_pair_indices": [0]})

    with tempfile.TemporaryDirectory(prefix="pool_channel_rd_tamper_selftest_") as temp:
        contract = {"mode": "SELF_TEST_TAMPER", "pair_count": 0}
        store = StageStore(Path(temp), contract)
        stage = store.complete_stage("custody", {"status": "STRUCTURAL_ONLY"})
        tampered = json.loads(stage.read_text())
        tampered["contract_sha256"] = "0" * 64
        atomic_write_json(stage, tampered)
        try:
            store.write_pair_checkpoint("baseline_byte_close", 0, {})
        except HarnessRefusal as exc:
            assert str(exc) == "BLOCKED_STAGE_RECORD_CUSTODY:custody"
        else:
            raise AssertionError("tampered predecessor was accepted")

    with tempfile.TemporaryDirectory(prefix="pool_channel_rd_payload_tamper_selftest_") as temp:
        contract = {"mode": "SELF_TEST_PAYLOAD_TAMPER", "pair_count": 0}
        store = StageStore(Path(temp), contract)
        stage = store.complete_stage("custody", {"status": "STRUCTURAL_ONLY"})
        tampered_stage = json.loads(stage.read_text())
        tampered_stage["payload"]["status"] = "TAMPERED"
        atomic_write_json(stage, tampered_stage)
        try:
            store._validated_stage("custody")
        except HarnessRefusal as exc:
            assert str(exc) == "BLOCKED_STAGE_PAYLOAD_SHA256:custody"
        else:
            raise AssertionError("stage payload tampering was accepted")

    with tempfile.TemporaryDirectory(prefix="pool_channel_rd_pair_tamper_selftest_") as temp:
        contract = {"mode": "SELF_TEST_PAIR_PAYLOAD_TAMPER", "pair_count": 0}
        store = StageStore(Path(temp), contract)
        store.complete_stage("custody", {"status": "STRUCTURAL_ONLY"})
        pair = store.write_pair_checkpoint("baseline_byte_close", 0, {"status": "STRUCTURAL_ONLY"})
        tampered_pair = json.loads(pair.read_text())
        tampered_pair["payload"]["status"] = "TAMPERED"
        atomic_write_json(pair, tampered_pair)
        try:
            store._validated_pair("baseline_byte_close", 0)
        except HarnessRefusal as exc:
            assert str(exc) == "BLOCKED_PAIR_PAYLOAD_SHA256:baseline_byte_close:0"
        else:
            raise AssertionError("pair payload tampering was accepted")

    with tempfile.TemporaryDirectory(prefix="pool_channel_rd_fake_archive_selftest_") as temp:
        archive_path = Path(temp) / "fake.zip"
        fake_blob = b"not-a-receiver-closed-payload"
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("0.bin", fake_blob)
        fake_candidate = {
            "row_id": make_rows()[0]["row_id"],
            "archive_path": str(archive_path),
            "archive_bytes": archive_path.stat().st_size,
            "archive_sha256": sha256_file(archive_path),
            "payload_bytes": len(fake_blob),
            "payload_sha256": hashlib.sha256(fake_blob).hexdigest(),
            "baseline_archive_path": str(archive_path),
            "baseline_archive_bytes": archive_path.stat().st_size,
            "baseline_archive_sha256": sha256_file(archive_path),
            "parseback_receipt_path": str(Path(temp) / "missing_receipt.json"),
            "parseback_receipt_sha256": "0" * 64,
            "baseline_argmax_path": str(Path(temp) / "missing_baseline.npy"),
            "baseline_argmax_sha256": "0" * 64,
            "candidate_argmax_path": str(Path(temp) / "missing_candidate.npy"),
            "candidate_argmax_sha256": "0" * 64,
            "gt_argmax_path": str(Path(temp) / "missing_gt.npy"),
            "gt_argmax_sha256": "0" * 64,
            "target_mask_path": str(Path(temp) / "missing_mask.npy"),
            "target_mask_sha256": "0" * 64,
            "geometry_run_contract_path": str(Path(temp) / "missing_contract.json"),
            "geometry_run_contract_sha256": "0" * 64,
            "geometry_stage_path": str(Path(temp) / "missing_geometry.json"),
            "geometry_stage_sha256": "0" * 64,
            "cell_intervention_receipt_path": str(Path(temp) / "missing_intervention.json"),
            "cell_intervention_receipt_sha256": "0" * 64,
            "scorer_weights_sha256": SEGNET_WEIGHTS_SHA256,
            "axis": CPU_AXIS,
            "pair_count": 600,
            "segnet_batch_size": 32,
            "baseline_d_seg": 0.0,
            "d_seg": 0.0,
            "rate_term": 25.0 * archive_path.stat().st_size / 37_545_489,
            "baseline_flip_count": 0,
            "candidate_flip_count": 0,
            "target_fixes": 0,
            "new_bad": 0,
            "non_target_transitions": 0,
        }
        try:
            admit_rd_point(fake_candidate)
        except HarnessRefusal as exc:
            assert str(exc) == "BLOCKED_RD_ARRAY_CUSTODY:baseline_argmax_path"
        else:
            raise AssertionError("a grammar-only fake archive was admitted")

    return {
        "schema": SCHEMA,
        "mode": "SELF_TEST_STRUCTURAL_ONLY",
        "scientific_measurement": False,
        "synthetic_scorer_instantiated": False,
        "row_count": len(rows),
        "checks": [
            "48-row Cartesian schema",
            "saddle-first canonical pool derivation, disjointness, and content hashing",
            "deterministic SVD sign convention",
            "orthogonal range/kernel projector algebra",
            "SPD actuator Gram J H^-1 J^T",
            "stacked-Gram cross-location sufficient statistics under identity and non-diagonal SPD metrics",
            "strict row-attributed content-derived R-D custody and grammar-only fake refusal",
            "durable execution-authority refusal",
            "atomic preserved stage and per-pair resume",
            "tampered envelope, payload, and pair-membership refusal",
        ],
        "status": "PASS",
        "score_claim": False,
        "promotable": False,
        "pointer_delta": 0,
    }


def _lazy_real_chain(repo_root: Path) -> dict[str, Any]:
    """Import only the canonical real-chain surfaces, never a scorer substitute."""
    source = Path(__file__).resolve().parents[1] / "src"
    tools_dir = Path(__file__).resolve().parents[1] / "tools"
    for path in (source, tools_dir, Path(__file__).resolve().parents[1]):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        from necessity_dseg_calibration import _strata_masks

        from tac.local_acceleration.torch_levelset_inflate import decode_levelset_torch
        from tac.witness_control.factorized_features import (
            decode_pairs_camera_frames,
            load_frozen_segnet_cpu,
            load_witness_ema,
            segnet_logits_for_frames,
        )
        from tools import levelset_byte_close_and_eval as byte_close
    except (ImportError, OSError) as exc:
        raise HarnessRefusal(f"BLOCKED_REAL_CHAIN_DEPENDENCY:{type(exc).__name__}:{exc}") from exc
    return {
        "decode_levelset_torch": decode_levelset_torch,
        "decode_pairs_camera_frames": decode_pairs_camera_frames,
        "load_frozen_segnet_cpu": load_frozen_segnet_cpu,
        "load_witness_ema": load_witness_ema,
        "segnet_logits_for_frames": segnet_logits_for_frames,
        "byte_close": byte_close,
        "strata_masks": _strata_masks,
        "upstream": repo_root / "upstream",
    }


def _byte_close_bank_code(
    manifest: Mapping[str, Any], params: Mapping[str, Any], code: Any, byte_close: Any
) -> tuple[dict[str, Any], dict[str, Any], Any, dict[str, Any]]:
    """Build/parse the baseline blob and return only receiver-consumed parameters/code."""
    import numpy as np

    packed_params = dict(params)
    packed_params["code"] = np.asarray(code, dtype=np.float32)
    cfg_keys = (
        "n_pairs",
        "n_classes",
        "hidden_dim",
        "n_hidden",
        "mod_dim",
        "activation",
        "softmax_temp",
        "chroma",
        "wire_w0",
        "wire_s0",
        "hosc_beta",
        "hosc_omega",
        "bank_n_scales",
        "bank_n_orient0",
        "bank_f0",
        "bank_base",
        "bank_n_iso",
        "max_bank_freq",
        "render_h",
        "render_w",
    )
    cfg = {key: manifest[key] for key in cfg_keys}
    cfg["in_feat"] = int(np.asarray(params["in_proj.weight"]).shape[1])
    so = {
        "self_orient": bool(manifest["self_orient"]),
        "n_dir_freqs": int(manifest.get("n_dir_freqs", 0)),
        "freq_across": float(manifest.get("so_freq_across", 0.0)),
        "freq_along": float(manifest.get("so_freq_along", 0.0)),
        "tau": float(manifest.get("so_tau", 4.0)),
        "iters": int(manifest.get("so_iters", 0)),
    }
    blob, breakdown = byte_close.build_levelset_blob(packed_params, cfg, so, None)
    parsed_manifest, base_block, code_block, pose_block, lane, pose_carrier = byte_close._read_blob_bytes(blob)
    if lane is not None or pose_carrier is not None:
        raise HarnessRefusal("BLOCKED_BASELINE_UNEXPECTED_OPTIONAL_SECTION")
    parsed_params = byte_close._decode_base_params(parsed_manifest, base_block)
    parsed_code = np.asarray(byte_close._decode_code(parsed_manifest, code_block), dtype=np.float32)
    shape = tuple(int(x) for x in parsed_manifest["code_shape"])
    scale = float(parsed_manifest["code_scale"])
    if parsed_code.shape != shape:
        raise HarnessRefusal("BLOCKED_BASELINE_PARSED_CODE_SHAPE")
    quantized_i16 = np.rint(parsed_code / np.float32(scale)).astype(np.int16)
    if np.any(quantized_i16 < -127) or np.any(quantized_i16 > 127):
        raise HarnessRefusal("BLOCKED_BASELINE_PARSED_CODE_RANGE")
    reconstructed = quantized_i16.astype(np.float32) * np.float32(scale)
    if not np.array_equal(parsed_code, reconstructed):
        raise HarnessRefusal("BLOCKED_BASELINE_PARSED_CODE_NOT_LATTICE_EXACT")
    quantized = quantized_i16.astype(np.int8)
    parameter_custody = {
        name: {
            "shape": list(np.asarray(value).shape),
            "dtype": str(np.asarray(value).dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
        for name, value in sorted(parsed_params.items())
    }
    return (
        dict(parsed_manifest),
        parsed_params,
        parsed_code,
        {
            "schema": "pool_channel_baseline_receiver_parseback.v1",
            "blob_bytes": len(blob),
            "blob_sha256": hashlib.sha256(blob).hexdigest(),
            "manifest_sha256": hashlib.sha256(canonical_json_bytes(parsed_manifest)).hexdigest(),
            "base_block_sha256": hashlib.sha256(base_block).hexdigest(),
            "parsed_base_parameter_custody": parameter_custody,
            "parsed_base_parameters_sha256": hashlib.sha256(canonical_json_bytes(parameter_custody)).hexdigest(),
            "pose_block_sha256": hashlib.sha256(pose_block).hexdigest(),
            "code_shape": list(shape),
            "code_scale": scale,
            "quantized_code_sha256": hashlib.sha256(quantized.tobytes()).hexdigest(),
            "parseback_pass": True,
            "receiver_parameters_are_parsed_int8_dequantized": True,
            "receiver_code_is_parsed_int8_dequantized": True,
            "accounting_matches_canonical": bool(breakdown["accounting_matches_canonical"]),
        },
    )


def _real_pool_masks(
    gt: Any, baseline_logits: Any, strata_masks: Callable[..., Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    labels = np.asarray(gt)
    logits = np.asarray(baseline_logits)
    predicted = logits.argmax(axis=0)
    flip = predicted != labels
    pools = canonical_pool_masks_from_labels(labels, predicted, strata_masks)
    counts = {
        "total_pixels": int(flip.size),
        "flip_count": int(flip.sum()),
        "d_seg": float(flip.mean()),
        "pool_counts": {name: int(mask.sum()) for name, mask in pools.items()},
    }
    if sum(counts["pool_counts"].values()) != counts["flip_count"]:
        raise HarnessRefusal("BLOCKED_REAL_POOL_COUNT_RECONCILIATION")
    return pools, counts


def _decode_pair_variants(
    decode_levelset_torch: Callable[..., Any],
    manifest: Mapping[str, Any],
    params: Mapping[str, Any],
    baseline_code: Any,
    pair_index: int,
    frame1_variants: Sequence[Any],
    max_batch_size: int,
) -> list[Any]:
    import numpy as np

    frame0 = np.asarray(baseline_code[2 * pair_index], dtype=np.float32)
    decoded: list[Any] = []
    batch_size = max(1, int(max_batch_size))
    for start in range(0, len(frame1_variants), batch_size):
        chunk = frame1_variants[start : start + batch_size]
        rows: list[Any] = []
        for frame1 in chunk:
            rows.extend((frame0, np.asarray(frame1, dtype=np.float32)))
        local_manifest = dict(manifest)
        local_manifest["n_pairs"] = len(chunk)
        out = decode_levelset_torch(
            local_manifest,
            dict(params),
            np.stack(rows).astype(np.float32),
            device="cpu",
            return_frames=True,
        )
        decoded.extend(pair[1] for pair in out["frames"])
    return decoded


def _measure_pair_full_path(
    *,
    pair_index: int,
    gt: Any,
    baseline_frame: Any,
    baseline_logits: Any,
    manifest: Mapping[str, Any],
    params: Mapping[str, Any],
    baseline_code: Any,
    code_scale: float,
    head_u: Any,
    real: Mapping[str, Any],
    segnet: Any,
    segnet_batch_size: int,
    continuous_dyadic_multipliers: Sequence[float],
) -> dict[str, Any]:
    import numpy as np

    pools, pool_counts = _real_pool_masks(gt, baseline_logits, real["strata_masks"])
    code = np.asarray(baseline_code, dtype=np.float32)
    q_code = np.rint(code / np.float32(code_scale)).astype(np.int16)
    if np.any(q_code < -127) or np.any(q_code > 127):
        raise HarnessRefusal("BLOCKED_PARSED_BASELINE_CODE_OUTSIDE_SYMMETRIC_INT8")
    q_frame1 = np.rint(code[2 * pair_index + 1] / np.float32(code_scale)).astype(np.int16)
    base_frame1_code = code[2 * pair_index + 1].astype(np.float64)
    selected: dict[int, tuple[float, Any, Any]] = {}
    baseline_i16 = np.asarray(baseline_frame, dtype=np.int16)
    # Decode one h-ladder rung at a time and retain only the first qualifying +/- pair
    # for each coordinate.  This bounds camera-frame memory independently of ladder length.
    for multiplier in sorted(set(continuous_dyadic_multipliers)):
        candidates: list[tuple[int, int]] = []
        variant_codes: list[Any] = []
        for coordinate in range(q_frame1.size):
            if coordinate in selected:
                continue
            step = float(code_scale) * multiplier
            if (
                base_frame1_code[coordinate] - step < -127.0 * code_scale
                or base_frame1_code[coordinate] + step > 127.0 * code_scale
            ):
                continue
            for sign in (1, -1):
                variant = base_frame1_code.copy()
                variant[coordinate] += sign * step
                variant_codes.append(variant.astype(np.float32))
                candidates.append((coordinate, sign))
        frames = _decode_pair_variants(
            real["decode_levelset_torch"],
            manifest,
            params,
            code,
            pair_index,
            variant_codes,
            segnet_batch_size,
        )
        by_candidate = dict(zip(candidates, frames, strict=True))
        for coordinate in range(q_frame1.size):
            plus = by_candidate.get((coordinate, 1))
            minus = by_candidate.get((coordinate, -1))
            if plus is None or minus is None or coordinate in selected:
                continue
            plus_i16, minus_i16 = np.asarray(plus, dtype=np.int16), np.asarray(minus, dtype=np.int16)
            max_plus = int(np.max(np.abs(plus_i16 - baseline_i16), initial=0))
            max_minus = int(np.max(np.abs(minus_i16 - baseline_i16), initial=0))
            changed = bool(np.any(plus_i16 != baseline_i16) or np.any(minus_i16 != baseline_i16))
            if changed and max_plus <= 1 and max_minus <= 1:
                selected[coordinate] = (float(code_scale) * multiplier, plus, minus)
    selected_frames: list[Any] = []
    selected_order: list[tuple[int, float]] = []
    for coordinate, (step, plus, minus) in sorted(selected.items()):
        selected_frames.extend((plus, minus))
        selected_order.append((coordinate, step))
    if not selected_frames:
        raise HarnessRefusal(f"BLOCKED_NO_DEPLOYABLE_ONE_LSB_CODE_STEP:pair={pair_index}")
    varied_logits = real["segnet_logits_for_frames"](segnet, selected_frames, batch=segnet_batch_size)
    selected_continuous_h = {str(key): float(value[0]) for key, value in selected.items()}
    selected_continuous_h_bins = {str(key): float(value[0]) / float(code_scale) for key, value in selected.items()}
    selected_coordinates = frozenset(selected)
    del selected_frames
    del selected
    n_coordinates = q_frame1.size
    matrices = {
        (pool, direction): np.zeros((int(mask.sum()), n_coordinates), dtype=np.float64)
        for pool, mask in pools.items()
        for direction in range(4)
    }
    for slot, (coordinate, step) in enumerate(selected_order):
        step_bins = step / float(code_scale)
        derivative = (varied_logits[2 * slot] - varied_logits[2 * slot + 1]) / (2.0 * step_bins)
        derivative = derivative - derivative.mean(axis=0, keepdims=True)
        for direction in range(4):
            projected = np.tensordot(head_u[:, direction], derivative, axes=(0, 0))
            for pool, mask in pools.items():
                matrices[(pool, direction)][:, coordinate] = projected[mask]
    del varied_logits
    lattice_codes: list[Any] = []
    lattice_keys: list[tuple[int, int]] = []
    lattice_roundtrip_rows: list[dict[str, Any]] = []
    for coordinate in range(q_frame1.size):
        if q_frame1[coordinate] <= -127 or q_frame1[coordinate] >= 127:
            continue
        for sign in (1, -1):
            variant_q_code = q_code.copy()
            variant_q_code[2 * pair_index + 1, coordinate] += sign
            compressed = real["byte_close"]._encode_code_brotli(variant_q_code.astype(np.int8), dict(manifest))
            parsed_full_code = np.asarray(real["byte_close"]._decode_code(dict(manifest), compressed), dtype=np.float32)
            parsed_q_code = np.rint(parsed_full_code / np.float32(code_scale)).astype(np.int16)
            if not np.array_equal(parsed_q_code, variant_q_code):
                raise HarnessRefusal("BLOCKED_LATTICE_CODE_SECTION_PARSEBACK")
            lattice_codes.append(parsed_full_code[2 * pair_index + 1].copy())
            lattice_keys.append((coordinate, sign))
            lattice_roundtrip_rows.append(
                {
                    "coordinate": coordinate,
                    "sign": sign,
                    "full_quantized_code_sha256": hashlib.sha256(
                        np.ascontiguousarray(variant_q_code.astype(np.int8)).tobytes()
                    ).hexdigest(),
                    "canonical_code_block_bytes": len(compressed),
                    "canonical_code_block_sha256": hashlib.sha256(compressed).hexdigest(),
                }
            )
    lattice_frames = _decode_pair_variants(
        real["decode_levelset_torch"],
        manifest,
        params,
        code,
        pair_index,
        lattice_codes,
        segnet_batch_size,
    )
    lattice_by_key = dict(zip(lattice_keys, lattice_frames, strict=True))
    lattice_realizability: dict[str, Any] = {}
    for coordinate in range(q_frame1.size):
        plus = lattice_by_key.get((coordinate, 1))
        minus = lattice_by_key.get((coordinate, -1))
        if plus is None or minus is None:
            lattice_realizability[str(coordinate)] = {"status": "INT8_BOUNDARY_BLOCKED"}
            continue
        plus_i16, minus_i16 = np.asarray(plus, dtype=np.int16), np.asarray(minus, dtype=np.int16)
        lattice_realizability[str(coordinate)] = {
            "status": "PARSEBACK_RENDERED",
            "plus_max_abs_lsb": int(np.max(np.abs(plus_i16 - baseline_i16), initial=0)),
            "minus_max_abs_lsb": int(np.max(np.abs(minus_i16 - baseline_i16), initial=0)),
            "plus_changed_pixels": int(np.count_nonzero(plus_i16 != baseline_i16)),
            "minus_changed_pixels": int(np.count_nonzero(minus_i16 != baseline_i16)),
        }
    del lattice_frames
    del lattice_by_key
    stats: dict[str, Any] = {}
    identity_metric = np.eye(n_coordinates, dtype=np.float64)
    for (pool, direction), matrix in matrices.items():
        accumulator = StackedJacobianStats(
            n_coordinates,
            "REAL continuous parsed-bank-code central FD in int8-bin units -> rounded canonical render/R -> frozen CPU Torch SegNet",
            coordinate_metric=identity_metric,
        )
        accumulator.add(matrix)
        stats[f"{pool}__head_sv{direction + 1}"] = accumulator.summary()
    blocked_coordinates = sorted(set(range(n_coordinates)) - selected_coordinates)
    return {
        "schema": "pool_channel_full_path_pair_geometry.v1",
        "pair_index": pair_index,
        "axis": CPU_AXIS,
        "pool_attribution": pool_counts,
        "pool_row_location_custody": {
            name: {
                "count": int(mask.sum()),
                "yx_int32_sha256": hashlib.sha256(
                    np.ascontiguousarray(np.argwhere(mask).astype("<i4")).tobytes()
                ).hexdigest(),
            }
            for name, mask in pools.items()
        },
        "selected_continuous_dyadic_h": selected_continuous_h,
        "selected_continuous_dyadic_h_int8_bins": selected_continuous_h_bins,
        "continuous_h_unit": "dequantized code value; h=code_scale*power_of_two multiplier",
        "lattice_int8_bin_secant": lattice_realizability,
        "lattice_parseback_surface": "full canonical transformed+Brotli code section, not a pair-row surrogate",
        "lattice_code_section_roundtrip_sha256": hashlib.sha256(
            canonical_json_bytes(lattice_roundtrip_rows)
        ).hexdigest(),
        "complete_coordinate_count": len(selected_coordinates),
        "deployable_coordinate_count": n_coordinates,
        "blocked_coordinates": blocked_coordinates,
        "coordinate_complete": not blocked_coordinates,
        "full_path_stats": stats,
        "path_split_status": "BLOCKED_EXACT_SKIP_DEEP_HOOK",
        "resize_split_status": "BLOCKED_FULL_ORTHOGONAL_RANGE_KERNEL_INTERVENTION",
        "finding_eligible": False,
        "coordinate_metric": "identity on parsed int8 stored-code coordinates",
        "coordinate_metric_sha256": hashlib.sha256(
            np.ascontiguousarray(identity_metric.astype("<f8")).tobytes()
        ).hexdigest(),
    }


def _aggregate_pair_local_blocks(pair_payloads: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate pair-local 32-D blocks without inventing cross-pair coupling."""
    import numpy as np

    cells: dict[str, dict[str, Any]] = {}
    cross_pool = {
        f"head_sv{direction}__{a}__{b}": 0.0
        for direction in range(1, 5)
        for a, b in ((POOLS[0], POOLS[1]), (POOLS[0], POOLS[2]), (POOLS[1], POOLS[2]))
    }
    for payload in pair_payloads:
        stats = payload["full_path_stats"]
        for key, row in stats.items():
            out = cells.setdefault(
                key,
                {
                    "n_spatial_rows": 0,
                    "trace_G_act_squared_pair_block_sum": 0.0,
                    "sum_row_norm_fourth_pair_block_sum": 0.0,
                    "cross_location_off_diagonal_energy_pair_block_sum": 0.0,
                    "pair_block_k_sha256": [],
                },
            )
            out["n_spatial_rows"] += int(row["n_spatial_rows"])
            out["trace_G_act_squared_pair_block_sum"] += float(row["trace_K_squared"])
            out["sum_row_norm_fourth_pair_block_sum"] += float(row["sum_row_norm_fourth"])
            out["cross_location_off_diagonal_energy_pair_block_sum"] += float(row["cross_location_off_diagonal_energy"])
            out["pair_block_k_sha256"].append(hashlib.sha256(canonical_json_bytes(row["K"])).hexdigest())
        for direction in range(1, 5):
            matrices = {pool: np.asarray(stats[f"{pool}__head_sv{direction}"]["K"], dtype=np.float64) for pool in POOLS}
            for a, b in ((POOLS[0], POOLS[1]), (POOLS[0], POOLS[2]), (POOLS[1], POOLS[2])):
                cross_pool[f"head_sv{direction}__{a}__{b}"] += float(np.trace(matrices[a] @ matrices[b]))
    return {
        "schema": "pool_channel_pair_block_diagonal_aggregate.v1",
        "pair_count": len(pair_payloads),
        "global_coordinate_geometry": "direct sum of independent pair-local stored-code blocks",
        "forbidden_operation": "never sum K_i across pairs before squaring",
        "cells": cells,
        "within_pair_cross_pool_trace_sum": cross_pool,
    }


def _validate_output_location(output_path: Path, canonical_root: Path) -> Path:
    """Refuse broad/sacred targets for every write surface, including ``plan``."""
    output = output_path.expanduser().resolve()
    broad_targets = {Path("/").resolve(), Path.home().resolve(), canonical_root.resolve()}
    if output in broad_targets:
        raise HarnessRefusal(f"BLOCKED_BROAD_OUTPUT_PATH:{output}")
    protected = (
        canonical_root / "experiments/results/banks",
        canonical_root / "experiments/results/mlx_fleet_gt_cache",
        canonical_root / "experiments/results/levelset_n600_witness_20260717T113932Z",
        canonical_root / "upstream",
        canonical_root / ".omx/state",
    )
    for base in protected:
        try:
            output.relative_to(base.resolve())
        except ValueError:
            continue
        raise HarnessRefusal(f"BLOCKED_PROTECTED_OUTPUT_PATH:{output}")
    return output


def _validate_measure_output(output_dir: Path, canonical_root: Path) -> Path:
    return _validate_output_location(output_dir, canonical_root)


def _validate_plan_output(output_path: Path, canonical_root: Path) -> Path:
    output = _validate_output_location(output_path, canonical_root)
    if output.exists():
        raise HarnessRefusal(f"BLOCKED_PLAN_OUTPUT_ALREADY_EXISTS:{output}")
    if output.parent.exists() and not output.parent.is_dir():
        raise HarnessRefusal(f"BLOCKED_PLAN_OUTPUT_PARENT_NOT_DIRECTORY:{output.parent}")
    return output


def _validate_execution_authority(payload: Mapping[str, Any], canonical_root: Path) -> dict[str, Any]:
    """Verify durable operator-GO, c2 terminal custody, and the live canonical lane row."""

    expected_wrapper = {
        "schema": "pool_channel_jacobian_rd.execution_authority.v1",
        "lane_id": LANE_ID,
        "execution_allowed": True,
        "operator_go": True,
        "queue_predecessor_run_id": QUEUE_PREDECESSOR_RUN_ID,
        "queue_predecessor_terminal": True,
        "cost_usd": 0,
    }
    if any(payload.get(key) != value for key, value in expected_wrapper.items()):
        raise HarnessRefusal("BLOCKED_EXECUTION_AUTHORITY_WRAPPER")

    def resolve_canonical_path(value: Any) -> Path:
        path = Path(str(value)).expanduser()
        return (path if path.is_absolute() else canonical_root / path).resolve()

    def require_under(path: Path, base: Path, blocker: str) -> None:
        try:
            path.relative_to(base.resolve())
        except ValueError as exc:
            raise HarnessRefusal(blocker) from exc

    claim = payload.get("active_lane_claim")
    if not isinstance(claim, Mapping):
        raise HarnessRefusal("BLOCKED_ACTIVE_LANE_CLAIM_SCHEMA")
    expected_claim = {
        "schema": "tac_active_lane_claim_json_v1",
        "active": True,
        "lane_id": LANE_ID,
        "platform": "local_cpu",
        "blockers": [],
        "claimed_with": ".venv/bin/python tools/claim_lane_dispatch.py claim",
        "claim_source": "canonical_claim_file",
    }
    if any(claim.get(key) != value for key, value in expected_claim.items()):
        raise HarnessRefusal("BLOCKED_ACTIVE_LANE_CLAIM_NOT_AUTHORIZING")
    instance_job_id = claim.get("instance_job_id")
    claim_status = claim.get("claim_status")
    if not isinstance(instance_job_id, str) or not instance_job_id:
        raise HarnessRefusal("BLOCKED_ACTIVE_LANE_CLAIM_JOB_ID")
    terminal_prefixes = ("completed_", "failed_", "preempted", "cancelled", "refused_", "stale_", "stopped_")
    if not isinstance(claim_status, str) or not claim_status or claim_status.startswith(terminal_prefixes):
        raise HarnessRefusal("BLOCKED_ACTIVE_LANE_CLAIM_TERMINAL")
    try:
        timestamp = dt.datetime.fromisoformat(str(claim["timestamp_utc"]).replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=dt.UTC)
        ttl_hours = float(claim["ttl_hours"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HarnessRefusal("BLOCKED_ACTIVE_LANE_CLAIM_TIME_SCHEMA") from exc
    age = dt.datetime.now(tz=dt.UTC) - timestamp.astimezone(dt.UTC)
    if not (0.0 < ttl_hours <= 24.0) or age < dt.timedelta(minutes=-5) or age > dt.timedelta(hours=ttl_hours):
        raise HarnessRefusal("BLOCKED_ACTIVE_LANE_CLAIM_STALE")
    claims_path = resolve_canonical_path(claim.get("claims_path"))
    canonical_claims_path = (canonical_root / ".omx/state/active_lane_dispatch_claims.md").resolve()
    if claims_path != canonical_claims_path or not claims_path.is_file():
        raise HarnessRefusal("BLOCKED_ACTIVE_LANE_CLAIM_CANONICAL_PATH")
    if sha256_file(claims_path) != claim.get("claims_file_sha256"):
        raise HarnessRefusal("BLOCKED_ACTIVE_LANE_CLAIM_FILE_DRIFT")
    claim_row_sha = claim.get("claim_row_sha256")
    matching_rows = [
        line
        for line in claims_path.read_text(encoding="utf-8").splitlines()
        if hashlib.sha256(line.encode()).hexdigest() == claim_row_sha
    ]
    if len(matching_rows) != 1:
        raise HarnessRefusal("BLOCKED_ACTIVE_LANE_CLAIM_ROW_CUSTODY")
    cells = [cell.strip().replace("\\|", "|") for cell in matching_rows[0].strip("|").split("|")]
    if len(cells) < 8 or cells[2] != LANE_ID or cells[4] != instance_job_id or cells[6] != claim_status:
        raise HarnessRefusal("BLOCKED_ACTIVE_LANE_CLAIM_ROW_BINDING")

    authorization_ref = payload.get("operator_authorization")
    if not isinstance(authorization_ref, Mapping):
        raise HarnessRefusal("BLOCKED_OPERATOR_AUTHORIZATION_REFERENCE")
    authorization_path = resolve_canonical_path(authorization_ref.get("path"))
    require_under(
        authorization_path,
        canonical_root / ".omx/research/operator_authorizations",
        "BLOCKED_OPERATOR_AUTHORIZATION_PATH",
    )
    if not authorization_path.is_file() or sha256_file(authorization_path) != authorization_ref.get("sha256"):
        raise HarnessRefusal("BLOCKED_OPERATOR_AUTHORIZATION_CUSTODY")
    try:
        authorization = json.loads(authorization_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessRefusal("BLOCKED_OPERATOR_AUTHORIZATION_SCHEMA") from exc
    expected_authorization = {
        "schema": "pool_channel_jacobian_rd.operator_go.v1",
        "lane_id": LANE_ID,
        "instance_job_id": instance_job_id,
        "operator_go": True,
        "pair_count": 600,
        "segnet_batch_size": 32,
        "queue_predecessor_run_id": QUEUE_PREDECESSOR_RUN_ID,
        "bank_sha256": DEFAULT_INPUTS["bank"][2],
        "gt_n600_sha256": DEFAULT_INPUTS["gt_n600"][2],
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "cost_usd": 0,
    }
    if any(authorization.get(key) != value for key, value in expected_authorization.items()):
        raise HarnessRefusal("BLOCKED_OPERATOR_AUTHORIZATION_NOT_FOR_THIS_RUN")
    if not isinstance(authorization.get("operator_quote"), str) or not authorization["operator_quote"].strip():
        raise HarnessRefusal("BLOCKED_OPERATOR_AUTHORIZATION_QUOTE_MISSING")
    try:
        issued = dt.datetime.fromisoformat(str(authorization["issued_utc"]).replace("Z", "+00:00"))
        expires = dt.datetime.fromisoformat(str(authorization["expires_utc"]).replace("Z", "+00:00"))
        if issued.tzinfo is None:
            issued = issued.replace(tzinfo=dt.UTC)
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=dt.UTC)
    except (KeyError, TypeError, ValueError) as exc:
        raise HarnessRefusal("BLOCKED_OPERATOR_AUTHORIZATION_TIME_SCHEMA") from exc
    now = dt.datetime.now(tz=dt.UTC)
    if not (issued.astimezone(dt.UTC) <= now <= expires.astimezone(dt.UTC)):
        raise HarnessRefusal("BLOCKED_OPERATOR_AUTHORIZATION_EXPIRED_OR_FUTURE")
    if expires - issued > dt.timedelta(hours=24) or expires <= issued:
        raise HarnessRefusal("BLOCKED_OPERATOR_AUTHORIZATION_WINDOW")

    predecessor_ref = payload.get("predecessor_terminal_receipt")
    if not isinstance(predecessor_ref, Mapping):
        raise HarnessRefusal("BLOCKED_QUEUE_PREDECESSOR_REFERENCE")
    predecessor_path = resolve_canonical_path(predecessor_ref.get("path"))
    allowed_predecessor_roots = (
        canonical_root / "experiments/results" / QUEUE_PREDECESSOR_RUN_ID,
        canonical_root / ".omx/state/governed_run_receipts",
    )
    if not any(predecessor_path.is_relative_to(base.resolve()) for base in allowed_predecessor_roots):
        raise HarnessRefusal("BLOCKED_QUEUE_PREDECESSOR_RECEIPT_PATH")
    if not predecessor_path.is_file() or sha256_file(predecessor_path) != predecessor_ref.get("sha256"):
        raise HarnessRefusal("BLOCKED_QUEUE_PREDECESSOR_RECEIPT_CUSTODY")
    try:
        predecessor = json.loads(predecessor_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessRefusal("BLOCKED_QUEUE_PREDECESSOR_RECEIPT_SCHEMA") from exc
    if (
        predecessor.get("schema") != "pool_channel_jacobian_rd.queue_predecessor_terminal.v1"
        or predecessor.get("run_id") != QUEUE_PREDECESSOR_RUN_ID
        or predecessor.get("terminal") is not True
        or predecessor.get("c2_complete") is not True
        or predecessor.get("pid_alive") is not False
        or predecessor.get("state") not in {"completed", "stopped_governed"}
    ):
        raise HarnessRefusal("BLOCKED_QUEUE_PREDECESSOR_NOT_TERMINAL")
    try:
        observed = dt.datetime.fromisoformat(str(predecessor["observed_utc"]).replace("Z", "+00:00"))
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=dt.UTC)
    except (KeyError, TypeError, ValueError) as exc:
        raise HarnessRefusal("BLOCKED_QUEUE_PREDECESSOR_TIME_SCHEMA") from exc
    predecessor_age = dt.datetime.now(tz=dt.UTC) - observed.astimezone(dt.UTC)
    if predecessor_age < dt.timedelta(minutes=-5) or predecessor_age > dt.timedelta(hours=1):
        raise HarnessRefusal("BLOCKED_QUEUE_PREDECESSOR_RECEIPT_STALE")

    return {
        "schema": expected_wrapper["schema"],
        "lane_id": LANE_ID,
        "active_lane_claim_sha256": hashlib.sha256(canonical_json_bytes(dict(claim))).hexdigest(),
        "active_lane_claim_row_sha256": claim_row_sha,
        "operator_authorization_path": str(authorization_path),
        "operator_authorization_sha256": authorization_ref["sha256"],
        "predecessor_terminal_receipt_path": str(predecessor_path),
        "predecessor_terminal_receipt_sha256": predecessor_ref["sha256"],
        "validated": True,
    }


def run_measure(args: argparse.Namespace) -> None:
    import numpy as np

    canonical_root = resolve_canonical_root(args.canonical_root)
    if not args.operator_go:
        raise HarnessRefusal("BLOCKED_OPERATOR_GO_REQUIRED")
    if not args.claim_receipt or not args.claim_receipt_sha256:
        raise HarnessRefusal("BLOCKED_GOVERNED_CLAIM_RECEIPT_REQUIRED")
    claim = Path(args.claim_receipt).expanduser().resolve()
    if not claim.is_file() or sha256_file(claim) != args.claim_receipt_sha256:
        raise HarnessRefusal("BLOCKED_GOVERNED_CLAIM_RECEIPT_CUSTODY")
    try:
        claim_payload = json.loads(claim.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise HarnessRefusal("BLOCKED_GOVERNED_CLAIM_RECEIPT_SCHEMA") from exc
    execution_authority = _validate_execution_authority(claim_payload, canonical_root)
    if args.pair_start < 0 or args.segnet_batch_size <= 0:
        raise HarnessRefusal("BLOCKED_PAIR_START_OR_BATCH_SCHEMA")
    if args.pair_count == 600:
        if args.segnet_batch_size != 32 or args.liveness_only or args.pair_start != 0:
            raise HarnessRefusal("BLOCKED_N600_REQUIRES_FINDING_MODE_AND_SEGNET_BATCH32")
        requested_finding_mode = True
        status = "N600_FULL_PATH_REAL_CHAIN_PATH_SPLIT_PENDING"
    elif 0 < args.pair_count < 600 and args.liveness_only:
        requested_finding_mode = False
        status = LIVENESS_ONLY
    else:
        raise HarnessRefusal("BLOCKED_PAIR_COUNT_MODE_CONTRACT")
    plan_args = argparse.Namespace(
        canonical_root=args.canonical_root,
        bank=args.bank,
        gt_n600=args.gt_n600,
        gt_n96=args.gt_n96,
    )
    plan = build_plan(plan_args)
    output_dir = _validate_measure_output(Path(args.output_dir), canonical_root)
    pair_indices = list(range(args.pair_start, args.pair_start + args.pair_count))
    if pair_indices[-1] >= 600:
        raise HarnessRefusal("BLOCKED_PAIR_INDEX_OUT_OF_N600_RANGE")
    try:
        dyadic_multipliers = sorted({float(x) for x in args.dyadic_steps.split(",") if x.strip()})
    except ValueError as exc:
        raise HarnessRefusal("BLOCKED_DYADIC_STEP_SCHEMA") from exc
    if not dyadic_multipliers or any(
        not math.isfinite(x) or x <= 0 or not math.isclose(math.log2(x), round(math.log2(x)), abs_tol=1e-12)
        for x in dyadic_multipliers
    ):
        raise HarnessRefusal("BLOCKED_DYADIC_MULTIPLIERS_MUST_BE_POSITIVE_POWERS_OF_TWO")
    contract = {
        "schema": SCHEMA,
        "mode": "MEASURE_REAL_FULL_PATH_CHAIN",
        "harness_sha256": sha256_file(Path(__file__).resolve()),
        "pair_indices": pair_indices,
        "segnet_batch_size": args.segnet_batch_size,
        "continuous_dyadic_multipliers_of_code_scale": dyadic_multipliers,
        "lattice_secant_step_int8_bins": 1,
        "liveness_only": args.liveness_only,
        "requested_finding_mode": requested_finding_mode,
        "claim_receipt": str(claim),
        "claim_receipt_sha256": args.claim_receipt_sha256,
        "input_sha256": {key: value["sha256"] for key, value in plan["inputs"].items()},
        "canonical_pointer_sha256_at_start": plan["canonical_pointer"]["post_plan_custody"]["sha256"],
    }
    store = StageStore(output_dir, contract, resume=args.resume)
    custody_payload = {
        "status": status,
        "finding_eligible": False,
        "inputs": plan["inputs"],
        "cache_identity": plan["cache_identity"],
        "canonical_pointer": plan["canonical_pointer"],
        "execution_authority_custody": execution_authority,
    }
    if args.resume and (output_dir / "stage_00_custody.json").is_file():
        store._validated_stage("custody")
    else:
        store.complete_stage("custody", custody_payload)
    real = _lazy_real_chain(canonical_root)
    bank_path = Path(plan["inputs"]["bank"]["path"])
    source_manifest, source_params, float_code = real["load_witness_ema"](bank_path)
    manifest, params, baseline_code, parseback = _byte_close_bank_code(
        source_manifest, source_params, float_code, real["byte_close"]
    )
    if not parseback["accounting_matches_canonical"]:
        raise HarnessRefusal("BLOCKED_BASELINE_BYTE_CLOSE_ACCOUNTING")
    segnet = real["load_frozen_segnet_cpu"](real["upstream"])
    head_weight = segnet.segmentation_head[0].weight.detach().cpu().numpy()
    centered = head_weight - head_weight.mean(axis=0, keepdims=True)
    head_matrix = centered.reshape(5, -1)
    head_u, head_s, head_vh = np.linalg.svd(head_matrix.astype(np.float64), full_matrices=False)
    head_u, head_vh = deterministic_svd_signs(head_u, head_vh)
    verify_settled_singular_values(head_s)
    segnet_weights = real["upstream"] / "models/segnet.safetensors"
    head_custody = materialize_real_head_svd_custody(centered, sha256_file(segnet_weights))
    runtime_custody = {
        "python": sys.version,
        "torch": __import__("torch").__version__,
        "segnet_weights_path": str(segnet_weights),
        "segnet_weights_sha256": sha256_file(segnet_weights),
        "head": head_custody,
    }
    baseline_payload = {
        "status": "REAL_CANONICAL_BANK_BLOB_PARSED_BACK",
        "parseback": parseback,
        "runtime_custody": runtime_custody,
        "pair_indices": pair_indices,
    }
    if args.resume and (output_dir / "stage_01_baseline_byte_close.json").is_file():
        existing_baseline = store._validated_stage("baseline_byte_close")["payload"]
        if existing_baseline != baseline_payload:
            raise HarnessRefusal("BLOCKED_RESUME_BASELINE_CUSTODY_DRIFT")
    else:
        store.complete_stage("baseline_byte_close", baseline_payload)
    gt_path = Path(plan["inputs"]["gt_n600"]["path"])
    coordinate_count = int(baseline_code.shape[1])
    identity_metric = np.eye(coordinate_count, dtype=np.float64)
    pair_payloads: list[dict[str, Any]] = []
    pending = []
    for pair_index in pair_indices:
        pair_path = output_dir / "pairs" / "geometry" / f"pair_{pair_index:04d}.json"
        if pair_path.is_file():
            record = store._validated_pair("geometry", pair_index)
            pair_payload = record["payload"]
            pair_payloads.append(pair_payload)
        else:
            pending.append(pair_index)
    for start in range(0, len(pending), args.segnet_batch_size):
        chunk = pending[start : start + args.segnet_batch_size]
        gt_rows = read_stored_npz_member_rows(gt_path, "lstars.npy", chunk)
        decoded = real["decode_pairs_camera_frames"](manifest, params, baseline_code, chunk, device="cpu")
        baseline_frames = [pair[1] for pair in decoded]
        baseline_logits = real["segnet_logits_for_frames"](segnet, baseline_frames, batch=args.segnet_batch_size)
        for local, pair_index in enumerate(chunk):
            pair_payload = _measure_pair_full_path(
                pair_index=pair_index,
                gt=gt_rows[local],
                baseline_frame=baseline_frames[local],
                baseline_logits=baseline_logits[local],
                manifest=manifest,
                params=params,
                baseline_code=baseline_code,
                code_scale=float(parseback["code_scale"]),
                head_u=head_u,
                real=real,
                segnet=segnet,
                segnet_batch_size=args.segnet_batch_size,
                continuous_dyadic_multipliers=dyadic_multipliers,
            )
            store.write_pair_checkpoint("geometry", pair_index, pair_payload)
            if requested_finding_mode and not pair_payload["coordinate_complete"]:
                raise HarnessRefusal(
                    f"BLOCKED_INCOMPLETE_DEPLOYABLE_CODE_JACOBIAN:pair={pair_index}:"
                    f"coordinates={pair_payload['blocked_coordinates']}"
                )
            pair_payloads.append(pair_payload)
    pair_payloads.sort(key=lambda row: int(row["pair_index"]))
    if [int(row["pair_index"]) for row in pair_payloads] != pair_indices:
        raise HarnessRefusal("BLOCKED_PAIR_PAYLOAD_MEMBERSHIP_DRIFT")
    aggregate_payload = {
        "schema": "pool_channel_full_path_geometry.v1",
        "pair_indices": pair_indices,
        "pair_count": len(pair_indices),
        "axis": CPU_AXIS,
        "segnet_batch_size": args.segnet_batch_size,
        "baseline_parseback": parseback,
        "runtime_custody": runtime_custody,
        "coordinate_metric": "identity on parsed int8 stored-code coordinates",
        "coordinate_metric_sha256": hashlib.sha256(
            np.ascontiguousarray(identity_metric.astype("<f8")).tobytes()
        ).hexdigest(),
        "pair_block_diagonal_geometry": _aggregate_pair_local_blocks(pair_payloads),
        "path_split_status": "BLOCKED_EXACT_SKIP_DEEP_HOOK",
        "resize_split_status": "BLOCKED_FULL_ORTHOGONAL_RANGE_KERNEL_INTERVENTION",
        "finding_eligible": False,
        "score_claim": False,
    }
    atomic_write_json(output_dir / "full_path_geometry.json", aggregate_payload)
    store.complete_stage(
        "geometry",
        {
            "expected_pair_indices": pair_indices,
            "full_path_geometry_path": "full_path_geometry.json",
            "full_path_geometry_sha256": sha256_file(output_dir / "full_path_geometry.json"),
            "finding_eligible": False,
        },
    )
    raise HarnessRefusal(
        "BLOCKED_EXACT_SKIP_DEEP_HOOK_AND_RANGE_KERNEL_INTERVENTION:"
        "real full-path rounded deployable-code Jacobian completed and preserved;"
        "48 path/component cells remain null until same-point interventions are exact"
    )


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--canonical-root", help="canonical Pact data root; defaults via worktree ancestry")
    parser.add_argument("--bank", help="override pinned read-only witness EMA path")
    parser.add_argument("--gt-n600", dest="gt_n600", help="override pinned read-only n600 GT cache")
    parser.add_argument("--gt-n96", dest="gt_n96", help="override pinned read-only n96 GT cache")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan", help="validate custody and emit 48 unmeasured build rows")
    add_input_arguments(plan)
    plan.add_argument("--output", required=True, help="deterministic JSON plan receipt")
    sub.add_parser("self-test", help="run dependency-light structural, non-scientific tests")
    admit = sub.add_parser("admit-rd", help="content-derive one receiver-closed n600 R-D point")
    admit.add_argument("--canonical-root", help="canonical Pact root for write-path protection")
    admit.add_argument("--candidate", required=True, help="candidate custody JSON")
    admit.add_argument("--candidate-sha256", required=True)
    admit.add_argument("--output", required=True, help="new deterministic admitted-point JSON")
    measure = sub.add_parser("measure", help="governed real full-path measurement with strict split blockers")
    add_input_arguments(measure)
    measure.add_argument("--operator-go", action="store_true")
    measure.add_argument("--claim-receipt")
    measure.add_argument("--claim-receipt-sha256")
    measure.add_argument("--output-dir", required=True)
    measure.add_argument("--pair-count", type=int, default=600)
    measure.add_argument("--pair-start", type=int, default=0)
    measure.add_argument("--segnet-batch-size", type=int, default=32)
    measure.add_argument(
        "--dyadic-steps",
        default="0.25,0.5,1,2",
        help="pre-registered power-of-two multipliers of parsed code_scale for continuous local h search",
    )
    measure.add_argument("--liveness-only", action="store_true")
    measure.add_argument("--resume", action="store_true")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "self-test":
            print(json.dumps(run_self_test(), sort_keys=True, indent=2))
        elif args.command == "plan":
            output = _validate_plan_output(Path(args.output), resolve_canonical_root(args.canonical_root))
            payload = build_plan(args)
            atomic_write_json(output, payload)
            print(json.dumps({"output": str(output), "row_count": 48, "status": UNMEASURED}))
        elif args.command == "admit-rd":
            candidate_path = Path(args.candidate).expanduser().resolve()
            if not candidate_path.is_file() or sha256_file(candidate_path) != args.candidate_sha256:
                raise HarnessRefusal("BLOCKED_RD_CANDIDATE_RECEIPT_CUSTODY")
            try:
                candidate = json.loads(candidate_path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                raise HarnessRefusal("BLOCKED_RD_CANDIDATE_RECEIPT_SCHEMA") from exc
            if not isinstance(candidate, dict):
                raise HarnessRefusal("BLOCKED_RD_CANDIDATE_RECEIPT_SCHEMA")
            output = _validate_plan_output(Path(args.output), resolve_canonical_root(args.canonical_root))
            admitted = admit_rd_point(candidate)
            atomic_write_json(output, admitted)
            print(json.dumps({"output": str(output), "row_id": admitted["row_id"], "status": "ADMITTED"}))
        elif args.command == "measure":
            run_measure(args)
        else:  # pragma: no cover - argparse makes this unreachable
            raise AssertionError(args.command)
        return 0
    except HarnessRefusal as exc:
        print(json.dumps({"schema": SCHEMA, "status": "REFUSED", "blocker": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
