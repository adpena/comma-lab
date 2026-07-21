#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure oriented-boundary BEV staticity and ruled-worldsheet residuals.

This is a research-only, deterministic NumPy measurement.  It consumes the
hash-pinned n600 frozen-scorer cache and an already-solved PPCS trajectory; it
never estimates motion from the boundaries it measures.  Per-frame stages are
atomic and resumable.  The n64 prefix is directional and the n600 prefix is the
only load-bearing scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
import warnings
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

from tac.boundary_math.ego_xi_trajectory import bspline_fit_error_curve, fit_se3_bspline_controls
from tac.boundary_math.lane_sdf_component import image_to_ground
from tac.clip_profile import camera_for_resolution, detect_class_order
from tac.optimization.predict_project_receiver import counted_planar_xi_series
from tac.optimization.predict_project_schema import parse_constraint_seed

SCHEMA: Final = "bev_staticity_developability_probe.v1"
STAGE_SCHEMA: Final = "bev_staticity_developability_frame.v1"
PAIR_COUNT: Final = 600
SCORER_HW: Final = (384, 512)
GT_CACHE_SHA256: Final = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
SOLVED_SEED_SHA256: Final = "a21dde38128bed7ff62860ef005b994b74202e0bd00a37d1df8824ee325e856b"
V_HORIZON: Final = 174.0
CAMERA_HEIGHT_M: Final = 1.22
FORWARD_RANGE_M: Final = (2.5, 55.0)
PIXEL_FLOOR: Final = 1.0
GRID_BINS: Final = 384
BRANCH_QUANTILES: Final = (0.0, 0.5, 1.0)
CLASS_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
GROUND_CLASSES: Final = frozenset(("Road", "Lane", "Movable"))
POINTER: Final = "0.1910828242 [contest-CPU] UNMOVED"
SSD_ROOTS: Final = (Path("/Volumes/VertigoDataTier/pact"), Path("/Volumes/APDataStore/pact"))


class ProbeError(ValueError):
    """A custody, geometry, resume, or measurement contract violation."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(json.dumps(_json_safe(value), sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _jsonable_array(value: np.ndarray) -> list[Any]:
    """Encode missing signature cells as JSON null, never non-standard NaN."""

    array = np.asarray(value, dtype=np.float64)
    return np.where(np.isfinite(array), array, None).tolist()


def stored_npy_memmap(npz_path: Path, key: str) -> np.memmap:
    """Open a ZIP_STORED NPY member without inflating the 5-GB cache."""

    member = f"{key}.npy"
    with zipfile.ZipFile(npz_path, "r") as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ProbeError(f"{member} is not ZIP_STORED")
        archive_offset = int(info.header_offset)
    with npz_path.open("rb") as handle:
        handle.seek(archive_offset)
        local = handle.read(30)
        if len(local) != 30 or local[:4] != b"PK\x03\x04":
            raise ProbeError(f"invalid local ZIP header for {member}")
        name_len, extra_len = struct.unpack_from("<HH", local, 26)
        handle.seek(name_len + extra_len, os.SEEK_CUR)
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ProbeError(f"unsupported NPY version {version} for {member}")
        data_offset = handle.tell()
    return np.memmap(
        npz_path,
        mode="r",
        dtype=dtype,
        shape=shape,
        offset=data_offset,
        order="F" if fortran else "C",
    )


def oriented_shallow_boundary_points(
    labels: np.ndarray, margins: np.ndarray, class_index: Mapping[str, int]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, int]]]:
    """Return subpixel boundary points only on each edge's shallow-margin side.

    For unlike neighboring labels p,q, the registered #275 localizer places the
    zero at t=M_p/(M_p+M_q).  The endpoint with smaller margin is recorded as the
    oriented shallow (flip-prone) side.  Deep-side counts remain custody, but do
    not enter D1-D3 or byte estimates.
    """

    lab = np.asarray(labels)
    mar = np.asarray(margins, dtype=np.float64)
    if lab.shape != SCORER_HW or mar.shape != SCORER_HW:
        raise ProbeError("boundary inputs must have scorer shape 384x512")
    if not np.all(np.isfinite(mar)) or np.any(mar < 0.0):
        raise ProbeError("margin field must be finite and non-negative")
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in CLASS_NAMES}
    all_pieces: dict[str, list[np.ndarray]] = {name: [] for name in CLASS_NAMES}
    counts = {name: {"shallow": 0, "deep": 0, "ties": 0} for name in CLASS_NAMES}
    id_to_name = {int(index): name for name, index in class_index.items()}

    def consume(a: np.ndarray, b: np.ndarray, ma: np.ndarray, mb: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
        denom = ma + mb
        t = np.where(denom > 0.0, ma / denom, 0.5)
        xy = np.stack((u(t), v(t)), axis=1).astype(np.float64)
        for side, other, own_margin, other_margin in ((a, b, ma, mb), (b, a, mb, ma)):
            del other
            for class_id in np.unique(side):
                name = id_to_name.get(int(class_id))
                if name is None:
                    continue
                owned = side == class_id
                shallow = owned & (own_margin <= other_margin)
                deep = owned & (own_margin > other_margin)
                tie = owned & (own_margin == other_margin)
                if np.any(owned):
                    all_pieces[name].append(xy[owned])
                if np.any(shallow):
                    pieces[name].append(xy[shallow])
                counts[name]["shallow"] += int(np.count_nonzero(shallow))
                counts[name]["deep"] += int(np.count_nonzero(deep))
                counts[name]["ties"] += int(np.count_nonzero(tie))

    hmask = lab[:, :-1] != lab[:, 1:]
    hy, hx = np.nonzero(hmask)
    if hy.size:
        a, b = lab[hy, hx], lab[hy, hx + 1]
        ma, mb = mar[hy, hx], mar[hy, hx + 1]
        consume(a, b, ma, mb, lambda t: hx.astype(np.float64) + t, lambda t: hy.astype(np.float64))
    vmask = lab[:-1, :] != lab[1:, :]
    vy, vx = np.nonzero(vmask)
    if vy.size:
        a, b = lab[vy, vx], lab[vy + 1, vx]
        ma, mb = mar[vy, vx], mar[vy + 1, vx]
        consume(a, b, ma, mb, lambda t: vx.astype(np.float64), lambda t: vy.astype(np.float64) + t)
    shallow_points = {
        name: np.concatenate(rows, axis=0) if rows else np.empty((0, 2), dtype=np.float64)
        for name, rows in pieces.items()
    }
    all_points = {
        name: np.concatenate(rows, axis=0) if rows else np.empty((0, 2), dtype=np.float64)
        for name, rows in all_pieces.items()
    }
    return shallow_points, all_points, counts


def cumulative_camera_poses(dense_xi: np.ndarray) -> np.ndarray:
    """Integrate frozen relative translation-first twists with tac.lie."""

    from tac.lie import _se3_numpy as se3

    xi = np.asarray(dense_xi, dtype=np.float64)
    if xi.shape != (PAIR_COUNT, 6) or not np.all(np.isfinite(xi)):
        raise ProbeError("solved xi must have shape (600,6) and be finite")
    poses = np.empty((PAIR_COUNT, 4, 4), dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    poses[0] = pose
    for frame in range(1, PAIR_COUNT):
        pose = se3.compose(pose, se3.exp_se3(xi[frame]))
        poses[frame] = pose
    return poses


def _quantile_signature(
    independent: np.ndarray,
    dependent: np.ndarray,
    local_forward: np.ndarray,
    edges: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.full((len(BRANCH_QUANTILES), GRID_BINS), np.nan, dtype=np.float64)
    forwards = np.full_like(values, np.nan)
    counts = np.zeros(GRID_BINS, dtype=np.int64)
    bins = np.searchsorted(edges, independent, side="right") - 1
    bins[independent == edges[-1]] = GRID_BINS - 1
    valid = (bins >= 0) & (bins < GRID_BINS) & np.isfinite(dependent) & np.isfinite(local_forward)
    for bin_id in np.unique(bins[valid]):
        selected = valid & (bins == bin_id)
        dep = dependent[selected]
        counts[int(bin_id)] = dep.size
        values[:, int(bin_id)] = np.quantile(dep, BRANCH_QUANTILES)
        forwards[:, int(bin_id)] = float(np.median(local_forward[selected]))
    return values, forwards, counts


def frame_signatures(
    points: Mapping[str, np.ndarray],
    pose: np.ndarray,
    z_edges: np.ndarray,
    camera: Any,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name in CLASS_NAMES:
        xy = np.asarray(points[name], dtype=np.float64)
        if name in GROUND_CLASSES:
            valid = xy[:, 1] > V_HORIZON + 0.5
            fwd, lat = image_to_ground(
                xy[:, 0],
                xy[:, 1],
                cam_h=CAMERA_HEIGHT_M,
                fx=float(camera.fx_scorer),
                fy=float(camera.fy_scorer),
                cx=float(camera.cx_scorer),
                v_h=V_HORIZON,
            )
            valid &= np.isfinite(fwd) & np.isfinite(lat)
            valid &= (fwd >= FORWARD_RANGE_M[0]) & (fwd <= FORWARD_RANGE_M[1])
            local = np.stack((lat[valid], np.zeros(np.count_nonzero(valid)), fwd[valid], np.ones(np.count_nonzero(valid))), axis=1)
            # Accelerate/vecLib emits spurious fp-status warnings for even identity
            # matmul on this host; explicit einsum is the deterministic NumPy path.
            world = (
                np.einsum("ij,nj->ni", pose, local, optimize=False)
                if local.size
                else np.empty((0, 4), dtype=np.float64)
            )
            values, forwards, counts = _quantile_signature(world[:, 2], world[:, 0], fwd[valid], z_edges)
            chart = "ground_bev_world"
        else:
            image_edges = np.linspace(0.0, float(SCORER_HW[0]), GRID_BINS + 1)
            values, forwards, counts = _quantile_signature(
                xy[:, 1], xy[:, 0], np.ones(xy.shape[0], dtype=np.float64), image_edges
            )
            chart = "ego_identity_image" if name == "MyCar" else "rotonly_image"
        result[name] = {
            "chart": chart,
            "values": _jsonable_array(values),
            "local_forward": _jsonable_array(forwards),
            "bin_counts": counts.tolist(),
            "oriented_point_count": int(xy.shape[0]),
        }
    return result


def robust_event_frames(values: np.ndarray, forwards: np.ndarray, *, ground: bool, fx: float) -> dict[str, Any]:
    deltas = np.full(values.shape[0], np.nan, dtype=np.float64)
    for frame in range(1, values.shape[0]):
        diff = np.abs(values[frame] - values[frame - 1])
        if ground:
            scale = fx / np.maximum(forwards[frame], 1e-9)
            diff = diff * scale
        finite = np.isfinite(diff)
        if np.any(finite):
            deltas[frame] = float(np.median(diff[finite]))
    finite_values = deltas[np.isfinite(deltas)]
    if finite_values.size == 0:
        return {"frames": [], "threshold_px": None, "delta_px": deltas.tolist(), "runs": []}
    center = float(np.median(finite_values))
    mad = float(np.median(np.abs(finite_values - center)))
    threshold = max(PIXEL_FLOOR, center + 3.0 * 1.4826 * mad)
    frames = np.flatnonzero(np.isfinite(deltas) & (deltas > threshold)).astype(int).tolist()
    runs: list[list[int]] = []
    for frame in frames:
        if not runs or frame > runs[-1][-1] + 1:
            runs.append([frame])
        else:
            runs[-1].append(frame)
    return {
        "frames": frames,
        "threshold_px": threshold,
        "delta_px": deltas.tolist(),
        "runs": [[run[0], run[-1]] for run in runs],
        "law": "DERIVED robust three-MAD gate, lower-bounded by the measured 1px scorer floor",
    }


def static_segments(n_frames: int, event_frames: Sequence[int]) -> list[tuple[int, int]]:
    excluded = {int(value) for value in event_frames}
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for frame in range(n_frames):
        if frame in excluded:
            if start is not None and frame - start >= 2:
                segments.append((start, frame))
            start = None
        elif start is None:
            start = frame
    if start is not None and n_frames - start >= 2:
        segments.append((start, n_frames))
    return segments


def _finite_quantiles(values: np.ndarray) -> dict[str, float | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"p50": None, "p90": None, "p95": None, "p99": None}
    return {
        "p50": float(np.quantile(finite, 0.50)),
        "p90": float(np.quantile(finite, 0.90)),
        "p95": float(np.quantile(finite, 0.95)),
        "p99": float(np.quantile(finite, 0.99)),
    }


def _fit_segment_polynomials(
    values: np.ndarray,
    forwards: np.ndarray,
    centers: np.ndarray,
    start: int,
    end: int,
    *,
    ground: bool,
    fx: float,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    fits: list[dict[str, Any]] = []
    reconstructed = np.full_like(values[start:end], np.nan)
    for branch in range(values.shape[1]):
        observed = values[start:end, branch]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            template = np.nanmedian(observed, axis=0)
        available = np.isfinite(template)
        selected: dict[str, Any] | None = None
        if np.count_nonzero(available) >= 2:
            z = centers[available]
            z_mid = float((z.min() + z.max()) * 0.5)
            z_scale = float(max((z.max() - z.min()) * 0.5, 1e-9))
            zn = (z - z_mid) / z_scale
            for order in (1, 2, 3):
                if z.size < order + 1:
                    continue
                coeff = np.polyfit(zn, template[available], order)
                coeff16 = coeff.astype(np.float16)
                pred_template = np.polyval(coeff16.astype(np.float64), (centers - z_mid) / z_scale)
                pred = np.broadcast_to(pred_template, observed.shape)
                residual = np.abs(observed - pred)
                if ground:
                    residual = residual * fx / np.maximum(forwards[start:end, branch], 1e-9)
                median = float(np.nanmedian(residual)) if np.any(np.isfinite(residual)) else math.inf
                candidate = {
                    "branch": branch,
                    "order": order,
                    "coefficients_normalized_fp16": [float(value) for value in coeff16],
                    "domain_mid": z_mid,
                    "domain_scale": z_scale,
                    "reconstruction_median_px": median,
                }
                if selected is None or median < float(selected["reconstruction_median_px"]):
                    selected = candidate
                if median <= PIXEL_FLOOR:
                    selected = candidate
                    break
            if selected is not None:
                coeff = np.asarray(selected["coefficients_normalized_fp16"], dtype=np.float64)
                pred = np.polyval(coeff, (centers - float(selected["domain_mid"])) / float(selected["domain_scale"]))
                reconstructed[:, branch, :] = np.broadcast_to(pred, observed.shape)
                selected["within_matched_1px_distortion"] = bool(selected["reconstruction_median_px"] <= PIXEL_FLOOR)
                fits.append(selected)
    return fits, reconstructed


def summarize_stratum(
    values: np.ndarray,
    forwards: np.ndarray,
    centers: np.ndarray,
    *,
    ground: bool,
    fx: float,
) -> dict[str, Any]:
    events = robust_event_frames(values, forwards, ground=ground, fx=fx)
    segments = static_segments(values.shape[0], events["frames"])
    residual_chunks: list[np.ndarray] = []
    poly_rows: list[dict[str, Any]] = []
    observed_count = 0
    for start, end in segments:
        segment = values[start:end]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            template = np.nanmedian(segment, axis=0)
        residual = np.abs(segment - template)
        if ground:
            residual = residual * fx / np.maximum(forwards[start:end], 1e-9)
        residual_chunks.append(residual[np.isfinite(residual)])
        observed_count += int(np.count_nonzero(np.isfinite(residual)))
        fits, _ = _fit_segment_polynomials(values, forwards, centers, start, end, ground=ground, fx=fx)
        poly_rows.append({"frame_range": [start, end], "fits": fits})
    residuals = np.concatenate(residual_chunks) if residual_chunks else np.empty(0, dtype=np.float64)
    static_fraction = float(np.mean(residuals <= PIXEL_FLOOR)) if residuals.size else None
    residual_q = _finite_quantiles(residuals)
    fitted_orders = [int(fit["order"]) for row in poly_rows for fit in row["fits"]]
    matched = [bool(fit["within_matched_1px_distortion"]) for row in poly_rows for fit in row["fits"]]
    near_static = bool(
        residuals.size
        and residual_q["p50"] is not None
        and float(residual_q["p50"]) <= PIXEL_FLOOR
        and static_fraction is not None
        and static_fraction >= 0.5
    )
    return {
        "event_detection": events,
        "static_segments": [{"frame_range": [a, b], "frames": b - a} for a, b in segments],
        "observed_oriented_boundary_samples": observed_count,
        "subpixel_noise_floor_px": PIXEL_FLOOR,
        "ruling_reconstruction_residual_px": residual_q,
        "static_fraction_at_1px_floor": static_fraction,
        "residual_dynamics_fraction": None if static_fraction is None else 1.0 - static_fraction,
        "near_static": near_static,
        "developable_fraction_at_noise_floor": static_fraction,
        "raw_gaussian_K": {
            "status": "NOT_NUMERICALLY_REPORTED_C3_DISCRETE_SECOND_DERIVATIVE_CONFOUND",
            "replacement": "directrix_plus_frozen_xi_ruling_reconstruction_residual",
        },
        "polynomial_order_needed": max(fitted_orders) if fitted_orders and all(matched) else ">3_or_unmatched",
        "polynomial_segments": poly_rows,
    }


def _estimate_bytes(
    summary: Mapping[str, Any], values: np.ndarray, forwards: np.ndarray, *, ground: bool, fx: float
) -> dict[str, Any]:
    packed = bytearray()
    finite = np.isfinite(values)
    packed.extend(np.packbits(finite.reshape(-1)).tobytes())
    pixel_values = values * fx / np.maximum(forwards, 1e-9) if ground else values
    quantized = np.clip(np.rint(pixel_values[finite]), -32768, 32767).astype("<i2")
    packed.extend(quantized.tobytes())
    baseline_bytes = len(brotli.compress(bytes(packed), quality=11))
    static_packet = bytearray()
    matched = True
    for row in summary["polynomial_segments"]:
        start, end = row["frame_range"]
        for fit in row["fits"]:
            matched &= bool(fit["within_matched_1px_distortion"])
            coeff = np.asarray(fit["coefficients_normalized_fp16"], dtype="<f2")
            static_packet.extend(struct.pack("<HHBB", int(start), int(end), int(fit["branch"]), int(fit["order"])))
            static_packet.extend(np.asarray([fit["domain_mid"], fit["domain_scale"]], dtype="<f2").tobytes())
            static_packet.extend(coeff.tobytes())
    for frame in summary["event_detection"]["frames"]:
        static_packet.extend(struct.pack("<H", int(frame)))
    return {
        "per_frame_plane_signature_brotli11_bytes": baseline_bytes,
        "static_coefficients_plus_event_packet_brotli11_bytes": len(brotli.compress(bytes(static_packet), quality=11)),
        "same_stratum": True,
        "matched_distortion": matched,
        "distortion_definition": "median oriented-boundary reconstruction <= registered 1px scorer floor",
        "receiver_closed": False,
    }


def _load_stages(stage_dir: Path, config_sha256: str, prefix: int) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for path in sorted(stage_dir.glob("frame_*.json")) if stage_dir.exists() else []:
        row = json.loads(path.read_text())
        frame = row.get("frame")
        if row.get("schema") != STAGE_SCHEMA or row.get("config_sha256") != config_sha256:
            raise ProbeError(f"stale or incompatible stage {path}")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame in rows:
            raise ProbeError(f"invalid duplicate stage {path}")
        if frame < prefix:
            rows[frame] = row
    return rows


def run_probe(gt_cache: Path, seed_path: Path, output_root: Path, *, prefix: int, chunk_size: int) -> dict[str, Any]:
    if prefix not in (64, 600) or chunk_size < 1:
        raise ProbeError("prefix must be n64 or n600 and chunk-size must be positive")
    resolved_output = output_root.expanduser().resolve()
    if not any(resolved_output == root or resolved_output.is_relative_to(root) for root in SSD_ROOTS):
        raise ProbeError("output-root must use the governed SSD waterfall")
    output_root = resolved_output
    output_root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output_root).free < 1 << 30:
        raise ProbeError("storage preflight requires at least 1 GiB free")
    if _sha256_file(gt_cache) != GT_CACHE_SHA256:
        raise ProbeError("gt_n600 cache SHA-256 mismatch")
    labels = stored_npy_memmap(gt_cache, "lstars")
    margins = stored_npy_memmap(gt_cache, "margins")
    if labels.shape != (PAIR_COUNT, *SCORER_HW) or margins.shape != labels.shape:
        raise ProbeError("gt_n600 labels/margins geometry mismatch")
    class_order, class_index = detect_class_order(np.asarray(labels[:64]))
    if class_order != CLASS_NAMES or set(class_index) != set(CLASS_NAMES):
        raise ProbeError(f"self-detected class map is not canonical: {class_order}, {class_index}")
    camera = camera_for_resolution(1164, 874)
    seed_bytes = seed_path.read_bytes()
    if hashlib.sha256(seed_bytes).hexdigest() != SOLVED_SEED_SHA256:
        raise ProbeError("solved PPCS seed SHA-256 mismatch")
    seed = parse_constraint_seed(seed_bytes)
    if seed["receiver"]["seed"] != 1234:
        raise ProbeError("solved PPCS seed must declare receiver seed 1234")
    xi, xi_custody = counted_planar_xi_series(seed)
    poses = cumulative_camera_poses(xi)
    z_edges = np.linspace(
        float(np.min(poses[:, 2, 3])) + FORWARD_RANGE_M[0],
        float(np.max(poses[:, 2, 3])) + FORWARD_RANGE_M[1],
        GRID_BINS + 1,
    )
    implementation = Path(__file__).resolve()
    config = {
        "schema": SCHEMA,
        "seed": 1234,
        "gt_cache": {"path": str(gt_cache), "sha256": GT_CACHE_SHA256},
        "solved_xi": {"path": str(seed_path), "sha256": hashlib.sha256(seed_bytes).hexdigest(), "custody": xi_custody},
        "class_self_detection_scale": "n64 canonical helper; map reused unchanged at n600",
        "class_order": list(class_order),
        "class_index": class_index,
        "geometry": {
            "v_horizon": V_HORIZON,
            "camera_height_m": CAMERA_HEIGHT_M,
            "intrinsics": {"fx": camera.fx_scorer, "fy": camera.fy_scorer, "cx": camera.cx_scorer, "cy": camera.cy_scorer},
            "forward_range_m": list(FORWARD_RANGE_M),
            "z_edges": z_edges.tolist(),
        },
        "implementation_sha256": _sha256_file(implementation),
        "lawrefs": {
            "v_horizon": "#327 n600 swept-optimal v_h=174",
            "camera_height": "tac.clip_profile.OPENPILOT_DEVICE_HEIGHT_M=1.22",
            "subpixel": "separatrix_asymmetry_t_subpixel_boundary_localizer_v1",
            "fisher_margin": "frozen_scorer_fisher_curvature_margin_colocation_v1",
            "xi": "tac.optimization.predict_project_receiver.counted_planar_xi_series",
            "ground_chart": "#325/#327 openpilot IPM plus tac.lie translation-first SE(3)",
        },
    }
    config_sha256 = hashlib.sha256(_canonical_bytes(config)).hexdigest()
    n64_receipt = output_root / "receipt_n64.json"
    if prefix == 600 and (
        not n64_receipt.is_file()
        or json.loads(n64_receipt.read_text()).get("config_sha256") != config_sha256
    ):
        raise ProbeError("n600 requires the exact-source n64 directional receipt")
    stage_dir = output_root / "stages"
    rows = _load_stages(stage_dir, config_sha256, prefix)
    for begin in range(0, prefix, chunk_size):
        end = min(prefix, begin + chunk_size)
        for frame in range(begin, end):
            if frame in rows:
                continue
            points, all_points, orientation = oriented_shallow_boundary_points(
                np.asarray(labels[frame]), np.asarray(margins[frame]), class_index
            )
            hood_points = {
                name: all_points[name] if name == "MyCar" else np.empty((0, 2), dtype=np.float64)
                for name in CLASS_NAMES
            }
            row = {
                "schema": STAGE_SCHEMA,
                "config_sha256": config_sha256,
                "frame": frame,
                "orientation_counts": orientation,
                "signatures": frame_signatures(points, poses[frame], z_edges, camera),
                "hood_full_silhouette_signature": frame_signatures(
                    hood_points, poses[frame], z_edges, camera
                )["MyCar"],
            }
            _atomic_json(stage_dir / f"frame_{frame:04d}.json", row)
            rows[frame] = row
        _atomic_json(
            output_root / "checkpoints" / f"prefix_{prefix}_chunk_{begin:04d}_{end:04d}.json",
            {"schema": "bev_staticity_checkpoint.v1", "config_sha256": config_sha256, "completed_through": end},
        )
    ordered = [rows[frame] for frame in range(prefix)]
    centers_ground = (z_edges[:-1] + z_edges[1:]) * 0.5
    centers_image = (np.linspace(0.0, float(SCORER_HW[0]), GRID_BINS + 1)[:-1] + np.linspace(0.0, float(SCORER_HW[0]), GRID_BINS + 1)[1:]) * 0.5
    summaries: dict[str, Any] = {}
    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in CLASS_NAMES:
        values = np.asarray([row["signatures"][name]["values"] for row in ordered], dtype=np.float64)
        forwards = np.asarray([row["signatures"][name]["local_forward"] for row in ordered], dtype=np.float64)
        matrices[name] = (values, forwards)
        summaries[name] = summarize_stratum(
            values,
            forwards,
            centers_ground if name in GROUND_CLASSES else centers_image,
            ground=name in GROUND_CLASSES,
            fx=float(camera.fx_scorer),
        )
        shallow = sum(int(row["orientation_counts"][name]["shallow"]) for row in ordered)
        deep = sum(int(row["orientation_counts"][name]["deep"]) for row in ordered)
        summaries[name]["orientation"] = {
            "shallow_side_samples": shallow,
            "deep_side_samples_excluded": deep,
            "shallow_fraction": shallow / (shallow + deep) if shallow + deep else None,
        }
    hood_values = np.asarray(
        [row["hood_full_silhouette_signature"]["values"] for row in ordered], dtype=np.float64
    )
    hood_forwards = np.asarray(
        [row["hood_full_silhouette_signature"]["local_forward"] for row in ordered], dtype=np.float64
    )
    hood_control = summarize_stratum(
        hood_values,
        hood_forwards,
        centers_image,
        ground=False,
        fx=float(camera.fx_scorer),
    )
    hood_pass = bool(hood_control["near_static"])
    summaries["MyCar"]["full_silhouette_control"] = hood_control
    summaries["MyCar"]["transform_positive_control"] = "PASS" if hood_pass else "FAIL"
    for name in ("Road", "Lane"):
        summaries[name]["interpretation_authorized_by_hood_control"] = hood_pass
        summaries[name]["D1_D2_hold"] = bool(hood_pass and summaries[name]["near_static"])
    summaries["Movable"]["expected_nonstatic_control"] = {
        "expected": True,
        "observed_nonstatic": not bool(summaries["Movable"]["near_static"]),
    }
    d3: dict[str, Any] = {
        "status": "ESTIMATE_ONLY_NOT_BYTE_CLOSED_ADMISSION",
        "future_gate": "g2g2-style through-real-homography decode plus frozen-scorer admission",
        "strata": {},
    }
    if any(bool(summaries[name]["D1_D2_hold"]) for name in ("Road", "Lane")):
        curve = bspline_fit_error_curve(xi, [4, 8, 16, 32, 64, 128, 256, 600])
        ground_pixel_m = CAMERA_HEIGHT_M / max(float(camera.fy_scorer), 1e-9)
        acceptable = [row for row in curve if float(row["fwd_rms_m"]) <= ground_pixel_m]
        selected = acceptable[0] if acceptable else curve[-1]
        controls = fit_se3_bspline_controls(xi, int(selected["M"]))
        from tac.lie import _se3_numpy as se3

        xi_payload = np.stack([se3.log_se3(control) for control in controls]).astype("<f2").tobytes()
        xi_bytes = len(brotli.compress(xi_payload, quality=11))
        d3["xi_bspline"] = {"curve": curve, "selected": selected, "brotli11_bytes": xi_bytes}
        for name in ("Road", "Lane"):
            if not summaries[name]["D1_D2_hold"]:
                d3["strata"][name] = {"status": "BLOCKED_D1_D2_OR_HOOD_CONTROL"}
                continue
            values, forwards = matrices[name]
            estimate = _estimate_bytes(summaries[name], values, forwards, ground=True, fx=float(camera.fx_scorer))
            if not estimate["matched_distortion"]:
                estimate["status"] = "BLOCKED_MATCHED_DISTORTION"
            else:
                static = int(estimate["static_coefficients_plus_event_packet_brotli11_bytes"])
                baseline = int(estimate["per_frame_plane_signature_brotli11_bytes"])
                gross = static + xi_bytes
                estimate.update(
                    {
                        "status": "MEASURED_ESTIMATE_MATCHED_ORIENTED_BOUNDARY_DISTORTION",
                        "gross_with_xi_bytes": gross,
                        "amortized_xi_static_plus_events_bytes": static,
                        "gross_collapse_ratio": baseline / gross if gross else None,
                        "amortized_xi_collapse_ratio": baseline / static if static else None,
                    }
                )
            d3["strata"][name] = estimate
    else:
        d3["blocker"] = "NO_ROAD_OR_LANE_STRATUM_PASSED_HOOD_GATED_D1_D2"
    receipt = {
        "schema": SCHEMA,
        "config": config,
        "config_sha256": config_sha256,
        "scale": f"n{prefix}",
        "scale_authority": "DIRECTIONAL_ONLY" if prefix == 64 else "LOAD_BEARING_N600",
        "D1_BEV_staticity": summaries,
        "D2_worldsheet_developability": {
            "estimator": "directrix plus frozen-xi ruling reconstruction residual",
            "raw_K_refused": True,
            "per_stratum": {name: {"residual_px": summaries[name]["ruling_reconstruction_residual_px"], "developable_fraction": summaries[name]["developable_fraction_at_noise_floor"]} for name in CLASS_NAMES},
        },
        "D3_describe_line_collapse": d3,
        "routing": {
            "einstein_kolmogorov_ultra_U1_U2": prefix == 600 and any(bool(summaries[name]["D1_D2_hold"]) for name in ("Road", "Lane")),
            "ops_grammar_U5": prefix == 600 and any(bool(summaries[name]["D1_D2_hold"]) for name in ("Road", "Lane")),
            "P0_register": prefix == 600 and any(bool(summaries[name]["D1_D2_hold"]) for name in ("Road", "Lane")),
            "dispatch_authority": False,
        },
        "richness_caveat": "static fraction is only the collapsible boundary facet; residual dynamics retain curvature changes, forks, dash rhythm, per-pair asymmetry, and independent movable motion",
        "authority": {"axis": "[macOS-CPU advisory]", "seed": 1234, "score_claim": False, "promotion_eligible": False, "pointer": POINTER, "pointer_moved": False},
        "storage": {"root": str(output_root), "resumable_stages": True, "automatic_disk_hygiene": "ZIP_STORED mmap plus small atomic JSON stages; no bulk scratch retained"},
        "verdict_scope": "oriented shallow-side frozen SegNet boundary at n64/n600; ground IPM only for Road/Lane/Movable; Undrivable and MyCar retain their canonical rotonly/identity charts; D3 is signature-space estimate, not receiver/scorer admission",
        "main_landing_review_required": True,
    }
    receipt["receipt_sha256"] = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    receipt_path = output_root / f"receipt_n{prefix}.json"
    receipt = _json_safe(receipt)
    _atomic_json(receipt_path, receipt)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-cache", type=Path, required=True)
    parser.add_argument("--solved-seed", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prefix", type=int, choices=(64, 600), required=True)
    parser.add_argument("--chunk-size", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run_probe(args.gt_cache, args.solved_seed, args.output_root, prefix=args.prefix, chunk_size=args.chunk_size)
    print(json.dumps({"receipt": str(args.output_root / f"receipt_n{args.prefix}.json"), "sha256": receipt["receipt_sha256"], "scale": receipt["scale"], "scale_authority": receipt["scale_authority"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
