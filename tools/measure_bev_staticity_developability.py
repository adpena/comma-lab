#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure absolute-trajectory BEV staticity and ruled-worldsheet residuals.

This is a research-only, deterministic NumPy measurement.  It consumes the
hash-pinned n600 frozen-scorer cache, reconstructs the missing cross-pair
PoseNet transitions, and composes an absolute frame-1 trajectory.  It never
estimates motion from the boundaries it measures.  Per-frame stages are atomic
and resumable.  The n64 prefix is a hood-only positive control; D1--D3 are
authorized only for n600 after that exact-source control passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
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
from tac.optimization.predictor_upgrade_xi_chart import load_g1_worldsheet_motion

REPO: Final = Path(__file__).resolve().parents[1]
SCHEMA: Final = "bev_staticity_developability_probe.v2"
STAGE_SCHEMA: Final = "bev_staticity_developability_frame.v2"
LABEL_STAGE_SCHEMA: Final = "bev_staticity_singleton_label_frame.v1"
PAIR_COUNT: Final = 600
SCORER_HW: Final = (384, 512)
CAMERA_HW: Final = (874, 1164)
GT_CACHE_SHA256: Final = "cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6"
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
D0_STATIC_FRACTION_FLOOR: Final = 0.5
D0_CLOSURE_FLOOR_M: Final = 1e-9


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


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def singleton_label_custody(
    gt_f0: np.ndarray,
    gt_f1: np.ndarray,
    cached_f1_labels: np.ndarray,
    score_frame: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Score f0 and f1 as two independent singleton calls.

    ``score_frame`` is injected so custody semantics can be tested without
    importing or loading the frozen scorer.  Production passes the exact
    ``segnet_argmax_and_margin`` frame path used by ``precompute_gt``.
    """

    f0 = np.asarray(gt_f0)
    f1 = np.asarray(gt_f1)
    cached = np.asarray(cached_f1_labels)
    if f0.shape != (*CAMERA_HW, 3) or f1.shape != f0.shape:
        raise ProbeError("singleton label inputs must have native 874x1164 RGB geometry")
    if cached.shape != SCORER_HW:
        raise ProbeError("cached f1 labels must have scorer geometry 384x512")
    label0 = np.asarray(score_frame(f0), dtype=np.uint8)
    label1 = np.asarray(score_frame(f1), dtype=np.uint8)
    if label0.shape != SCORER_HW or label1.shape != SCORER_HW:
        raise ProbeError("singleton scorer returned noncanonical label geometry")
    mismatch_count = int(np.count_nonzero(label1 != cached))
    return label0, {
        "scorer_call_geometry": "one_native_frame_per_call",
        "singleton_call_count": 2,
        "f0_source_sha256": _array_sha256(f0),
        "f1_source_sha256": _array_sha256(f1),
        "cached_f1_label_sha256": _array_sha256(cached),
        "scored_f1_label_sha256": _array_sha256(label1),
        "f1_cache_label_mismatches": mismatch_count,
        "f1_cache_binding_status": "EXACT" if mismatch_count == 0 else "FAIL_CLOSED_MISMATCH",
    }


def build_canonical_f0_label_sidecar(
    gt_f0: np.ndarray,
    gt_f1: np.ndarray,
    cached_f1_labels: np.ndarray,
    *,
    output: Path,
    stage_dir: Path,
    prefix: int,
    score_frame: Any,
    source_binding: Mapping[str, Any],
) -> tuple[np.memmap, dict[str, Any]]:
    """Build/resume a source-bound f0-label sidecar one frame at a time."""

    if prefix not in (64, 600):
        raise ProbeError("canonical label sidecar prefix must be n64 or n600")
    if gt_f0.shape != (PAIR_COUNT, *CAMERA_HW, 3) or gt_f1.shape != gt_f0.shape:
        raise ProbeError("cached RGB frame geometry mismatch")
    if cached_f1_labels.shape != (PAIR_COUNT, *SCORER_HW):
        raise ProbeError("cached f1 label geometry mismatch")
    binding = {
        "schema": LABEL_STAGE_SCHEMA,
        "gt_cache_sha256": GT_CACHE_SHA256,
        "source_binding": _json_safe(source_binding),
        "scorer_call_geometry": "singleton",
    }
    binding_sha256 = hashlib.sha256(_canonical_bytes(binding)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    stage_dir.mkdir(parents=True, exist_ok=True)
    if output.exists():
        labels0 = np.lib.format.open_memmap(output, mode="r+")
        if labels0.shape != (PAIR_COUNT, *SCORER_HW) or labels0.dtype != np.uint8:
            raise ProbeError("canonical f0 sidecar geometry or dtype drifted")
    else:
        labels0 = np.lib.format.open_memmap(
            output, mode="w+", dtype=np.uint8, shape=(PAIR_COUNT, *SCORER_HW)
        )
    rows: list[dict[str, Any]] = []
    for frame in range(prefix):
        stage_path = stage_dir / f"frame_{frame:04d}.json"
        if stage_path.is_file():
            row = json.loads(stage_path.read_text(encoding="utf-8"))
            if (
                row.get("schema") != LABEL_STAGE_SCHEMA
                or row.get("frame") != frame
                or row.get("binding_sha256") != binding_sha256
            ):
                raise ProbeError(f"stale or incompatible singleton label stage {stage_path}")
            if _array_sha256(np.asarray(labels0[frame])) != row.get("f0_label_sha256"):
                raise ProbeError(f"singleton label sidecar bytes drifted at frame {frame}")
        else:
            label0, custody = singleton_label_custody(
                np.asarray(gt_f0[frame]),
                np.asarray(gt_f1[frame]),
                np.asarray(cached_f1_labels[frame]),
                score_frame,
            )
            labels0[frame] = label0
            labels0.flush()
            row = {
                "schema": LABEL_STAGE_SCHEMA,
                "stage_identity": f"D0.1:singleton_labels:frame_{frame:04d}",
                "binding_sha256": binding_sha256,
                "frame": frame,
                "f0_label_sha256": _array_sha256(label0),
                **custody,
            }
            _atomic_json(stage_path, row)
        rows.append(row)
    mismatch_count = sum(int(row["f1_cache_label_mismatches"]) for row in rows)
    manifest = {
        "schema": "bev_staticity_singleton_label_sidecar.v1",
        "path": str(output),
        "binding": binding,
        "binding_sha256": binding_sha256,
        "processed_pairs": prefix,
        "stage_count": len(rows),
        "f1_cache_label_mismatches": mismatch_count,
        "f1_cache_binding_status": "EXACT" if mismatch_count == 0 else "FAIL_CLOSED_MISMATCH",
        "rebuildable": True,
    }
    _atomic_json(stage_dir / f"manifest_n{prefix}.json", manifest)
    return labels0, manifest


def bottom_connected_component(labels: np.ndarray, class_id: int) -> np.ndarray:
    """Return the largest 4-connected class component touching the bottom edge."""

    plane = np.asarray(labels)
    if plane.ndim != 2:
        raise ProbeError("bottom-connected component input must be a 2-D label plane")
    mask = plane == int(class_id)
    visited = np.zeros(mask.shape, dtype=np.bool_)
    components: list[np.ndarray] = []
    for bottom_x in np.flatnonzero(mask[-1]):
        seed = (mask.shape[0] - 1, int(bottom_x))
        if visited[seed]:
            continue
        connected = np.zeros(mask.shape, dtype=np.bool_)
        stack = [seed]
        while stack:
            y, x = stack.pop()
            if visited[y, x] or not mask[y, x]:
                continue
            visited[y, x] = True
            connected[y, x] = True
            if y:
                stack.append((y - 1, x))
            if y + 1 < mask.shape[0]:
                stack.append((y + 1, x))
            if x:
                stack.append((y, x - 1))
            if x + 1 < mask.shape[1]:
                stack.append((y, x + 1))
        components.append(connected)
    if not components:
        return np.zeros(mask.shape, dtype=np.bool_)
    return max(components, key=np.count_nonzero)


def component_boundary_points(component: np.ndarray) -> np.ndarray:
    """Return scorer-grid pixel centers on a binary component's inner boundary."""

    mask = np.asarray(component, dtype=np.bool_)
    if mask.ndim != 2:
        raise ProbeError("component boundary input must be 2-D")
    interior = mask.copy()
    interior[1:] &= mask[:-1]
    interior[:-1] &= mask[1:]
    interior[:, 1:] &= mask[:, :-1]
    interior[:, :-1] &= mask[:, 1:]
    interior[0] = False
    interior[-1] = False
    interior[:, 0] = False
    interior[:, -1] = False
    y, x = np.nonzero(mask & ~interior)
    return np.column_stack((x, y)).astype(np.float64)


def _validate_transform(transform: np.ndarray, *, name: str) -> None:
    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.all(np.isfinite(value)):
        raise ProbeError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), atol=1e-12, rtol=0.0):
        raise ProbeError(f"{name} has an invalid homogeneous last row")


def absolute_frame_trajectories(
    raw_within: np.ndarray,
    raw_cross: np.ndarray,
    *,
    s_t: float,
    s_r: float,
    pitch_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Compose phase-consistent absolute f0 and f1 trajectories."""

    from tac.boundary_math.warp_real_luma_frame0 import xi_from_pose_calibration
    from tac.lie import _se3_numpy as se3

    within = np.asarray(raw_within, dtype=np.float64)
    cross = np.asarray(raw_cross, dtype=np.float64)
    if within.ndim != 2 or within.shape[1] != 6 or cross.shape != within.shape:
        raise ProbeError("raw within/cross PoseNet targets must have identical (N,6) shape")
    if len(within) < 1 or not np.all(np.isfinite(within)) or not np.all(np.isfinite(cross)):
        raise ProbeError("raw PoseNet targets must be nonempty and finite")
    xi_within = np.stack(
        [xi_from_pose_calibration(row, s_t=s_t, s_r=s_r, pitch=pitch_rad) for row in within]
    )
    xi_cross = np.zeros_like(xi_within)
    for frame in range(1, len(cross)):
        xi_cross[frame] = xi_from_pose_calibration(
            cross[frame], s_t=s_t, s_r=s_r, pitch=pitch_rad
        )
    absolute_f0 = np.empty((len(within), 4, 4), dtype=np.float64)
    absolute_f1 = np.empty_like(absolute_f0)
    absolute_f1[0] = np.eye(4, dtype=np.float64)
    within_step0 = se3.exp_se3(xi_within[0])
    absolute_f0[0] = se3.compose(absolute_f1[0], se3.inverse(within_step0))
    max_inverse_closure = 0.0
    max_within_phase_closure = 0.0
    max_cross_phase_closure = 0.0
    for frame in range(len(within)):
        within_step = se3.exp_se3(xi_within[frame])
        _validate_transform(within_step, name=f"within[{frame}]")
        if frame:
            cross_step = se3.exp_se3(xi_cross[frame])
            absolute_f0[frame] = se3.compose(absolute_f1[frame - 1], cross_step)
            absolute_f1[frame] = se3.compose(absolute_f0[frame], within_step)
            cross_closure = se3.compose(absolute_f1[frame - 1], cross_step)
            max_cross_phase_closure = max(
                max_cross_phase_closure,
                float(np.max(np.abs(cross_closure - absolute_f0[frame]))),
            )
        _validate_transform(absolute_f0[frame], name=f"A_f0[{frame}]")
        _validate_transform(absolute_f1[frame], name=f"A_f1[{frame}]")
        within_closure = se3.compose(absolute_f0[frame], within_step)
        max_within_phase_closure = max(
            max_within_phase_closure,
            float(np.max(np.abs(within_closure - absolute_f1[frame]))),
        )
        for transform in (absolute_f0[frame], absolute_f1[frame]):
            closure = se3.compose(transform, se3.inverse(transform))
            max_inverse_closure = max(
                max_inverse_closure,
                float(np.max(np.abs(closure - np.eye(4)))),
            )
    return absolute_f0, absolute_f1, xi_cross, xi_within, {
        "finite_se3": True,
        "homogeneous_last_rows_valid": True,
        "inverse_compose_closure_max_abs": max_inverse_closure,
        "within_phase_closure_max_abs": max_within_phase_closure,
        "cross_phase_closure_max_abs": max_cross_phase_closure,
        "f0_composition": (
            "A_f0[0]=A_f1[0]*inverse(exp(xi_within[0])); "
            "A_f0[t]=A_f1[t-1]*exp(xi_cross[t]) for t>=1"
        ),
        "f1_composition": "A_f1[t]=A_f0[t]*exp(xi_within[t])",
        "composition_order": "A_f1[t-1] * exp(xi_cross[t]) * exp(xi_within[t])",
        "anchor": "A_f1[0]=I; A_f0[0]=A_f1[0]*inverse(exp(xi_within[0]))",
        "already_relative_targets_redifferenced": False,
    }


def bev_staticity_v2_motion_custody(motion_custody: Mapping[str, Any]) -> dict[str, Any]:
    """Retain G1 calibration authority while superseding its proxy transition policy."""

    required = (
        "g1_receipt_path",
        "g1_receipt_sha256",
        "lawref_equation_ids",
        "lawref_resolutions",
        "pitch_custody",
    )
    missing = [key for key in required if key not in motion_custody]
    if missing:
        raise ProbeError(f"G1 calibration custody is missing required fields: {missing}")
    return {
        "schema": "bev_staticity_absolute_trajectory_motion_custody.v2",
        "calibration_authority": {
            "g1_receipt_path": motion_custody["g1_receipt_path"],
            "g1_receipt_sha256": motion_custody["g1_receipt_sha256"],
            "lawref_equation_ids": list(motion_custody["lawref_equation_ids"]),
            "lawref_resolutions": _json_safe(motion_custody["lawref_resolutions"]),
            "pitch_custody": _json_safe(motion_custody["pitch_custody"]),
        },
        "transition_authority": {
            "within": "gt_poses[t] exact cached raw PoseNet target f0[t]->f1[t]",
            "cross": "exact frozen CPU PoseNet singleton target gt_f1[t-1]->gt_f0[t]",
            "cross_target_is_nearest_pair_proxy": False,
            "absolute_charts": "phase-consistent A_f0[t] and A_f1[t] in one fixed world frame",
            "composition_order": "A_f1[t-1]*exp(xi_cross[t])*exp(xi_within[t])",
        },
        "supersession": {
            "scope": "this BEV staticity v2 tool only",
            "supersedes_g1_nearest_target_proxy_limitation": True,
            "reason": "the missing cross transition is now directly scored on cached adjacent frames",
        },
    }


def hood_world_to_ego_closure(
    hood_boundary_xy: np.ndarray, pose: np.ndarray, camera: Any
) -> dict[str, Any]:
    """Lift hood boundary to ground proxy, world-transform, and invert explicitly."""

    from tac.lie import _se3_numpy as se3

    _validate_transform(pose, name="hood pose")
    xy = np.asarray(hood_boundary_xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ProbeError("hood boundary must have shape (N,2)")
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
    if not np.any(valid):
        raise ProbeError("bottom-connected hood has no finite ground-proxy boundary")
    ego = np.stack(
        (lat[valid], np.zeros(np.count_nonzero(valid)), fwd[valid], np.ones(np.count_nonzero(valid))),
        axis=1,
    )
    world = np.einsum("ij,nj->ni", pose, ego, optimize=False)
    recovered = np.einsum("ij,nj->ni", se3.inverse(pose), world, optimize=False)
    error = np.linalg.norm(recovered[:, :3] - ego[:, :3], axis=1)
    return {
        "lifted_point_count": len(error),
        "max_error_m": float(np.max(error)),
        "p50_error_m": float(np.median(error)),
        "threshold_m": D0_CLOSURE_FLOOR_M,
        "passed": bool(np.max(error) < D0_CLOSURE_FLOOR_M),
    }


def d0_gate_decision(
    *,
    label_mismatches: int,
    trajectory: Mapping[str, Any],
    hood_closure_max_m: float,
    hood_summary: Mapping[str, Any],
) -> dict[str, Any]:
    """Return ordered fail-closed D0 semantics and downstream authorization."""

    residual = hood_summary.get("ruling_reconstruction_residual_px", {})
    p50 = residual.get("p50") if isinstance(residual, Mapping) else None
    fraction = hood_summary.get("static_fraction_at_1px_floor")
    checks = [
        ("D0.1_SINGLETON_LABEL_CUSTODY", label_mismatches == 0),
        (
            "D0.2_ABSOLUTE_TRAJECTORY",
            trajectory.get("finite_se3") is True
            and trajectory.get("homogeneous_last_rows_valid") is True
            and float(trajectory.get("inverse_compose_closure_max_abs", math.inf))
            < D0_CLOSURE_FLOOR_M
            and float(trajectory.get("within_phase_closure_max_abs", math.inf))
            < D0_CLOSURE_FLOOR_M
            and float(trajectory.get("cross_phase_closure_max_abs", math.inf))
            < D0_CLOSURE_FLOOR_M,
        ),
        ("D0.3_HOOD_WORLD_TO_EGO_CLOSURE", math.isfinite(hood_closure_max_m) and hood_closure_max_m < D0_CLOSURE_FLOOR_M),
        ("D0.3_HOOD_P50_RESIDUAL", p50 is not None and float(p50) <= PIXEL_FLOOR),
        ("D0.3_HOOD_STATIC_FRACTION", fraction is not None and float(fraction) >= D0_STATIC_FRACTION_FLOOR),
    ]
    first_failed = next((name for name, passed in checks if not passed), None)
    passed = first_failed is None
    return {
        "passed": passed,
        "first_failed_stage": first_failed,
        "checks": dict(checks),
        "thresholds": {
            "p50_residual_px_max": PIXEL_FLOOR,
            "static_fraction_at_1px_min": D0_STATIC_FRACTION_FLOOR,
            "world_to_ego_closure_m_strict_max": D0_CLOSURE_FLOOR_M,
        },
        "D1_D3_authorized": passed,
    }


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


def _load_frozen_scorers(upstream: Path) -> tuple[Any, Any, dict[str, Any]]:
    """Load the same deterministic singleton CPU paths used by precompute_gt."""

    resolved = upstream.expanduser().resolve()
    if not resolved.is_dir():
        raise ProbeError(f"missing frozen upstream scorer tree: {resolved}")
    if str(resolved) not in sys.path:
        sys.path.insert(0, str(resolved))
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    from tac.boundary_math.seg_core import load_real_segnet, segnet_argmax_and_margin

    torch.manual_seed(1234)
    torch.use_deterministic_algorithms(True)
    segnet = load_real_segnet("cpu").eval()
    distortion = DistortionNet().eval()
    distortion.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    posenet = distortion.posenet.to("cpu").eval()
    for model in (segnet, posenet):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    def score_label(frame: np.ndarray) -> np.ndarray:
        label, _margin = segnet_argmax_and_margin(segnet, np.asarray(frame))
        return np.asarray(label, dtype=np.uint8)

    def score_pose(frame0: np.ndarray, frame1: np.ndarray) -> np.ndarray:
        pair = torch.from_numpy(np.stack((frame0, frame1), axis=0)[None].copy()).float()
        pair = pair.permute(0, 1, 4, 2, 3).contiguous()
        with torch.inference_mode():
            result = posenet(posenet.preprocess_input(pair))
            pose = result["pose"] if isinstance(result, dict) else result
            half = next(
                (head.out // 2 for head in posenet.hydra.heads if head.name == "pose"),
                pose.shape[-1] // 2,
            )
        raw = pose[0, :half].cpu().numpy().astype(np.float64)
        if raw.shape != (6,) or not np.all(np.isfinite(raw)):
            raise ProbeError("frozen PoseNet singleton path returned a malformed raw target")
        return raw

    modules_path = resolved / "modules.py"
    custody = {
        "axis": "frozen CPU-torch deterministic singleton",
        "seed": 1234,
        "torch_deterministic_algorithms": True,
        "modules_py": {"path": str(modules_path), "sha256": _sha256_file(modules_path)},
        "segnet_weights": {"path": str(segnet_sd_path), "sha256": _sha256_file(Path(segnet_sd_path))},
        "posenet_weights": {"path": str(posenet_sd_path), "sha256": _sha256_file(Path(posenet_sd_path))},
    }
    return score_label, score_pose, custody


def build_absolute_pose_stages(
    gt_f0: np.ndarray,
    gt_f1: np.ndarray,
    raw_within: np.ndarray,
    *,
    prefix: int,
    stage_dir: Path,
    config_sha256: str,
    score_pose: Any,
    s_t: float,
    s_r: float,
    pitch_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Resume raw cross-pair scoring, then compose and persist absolute poses."""

    stage_dir.mkdir(parents=True, exist_ok=True)
    raw_cross = np.zeros((prefix, 6), dtype=np.float64)
    raw = np.asarray(raw_within[:prefix], dtype=np.float64)
    for frame in range(prefix):
        path = stage_dir / f"frame_{frame:04d}.json"
        if path.is_file():
            row = json.loads(path.read_text(encoding="utf-8"))
            if (
                row.get("schema") != "bev_staticity_absolute_pose_frame.v2"
                or row.get("config_sha256") != config_sha256
                or row.get("frame") != frame
            ):
                raise ProbeError(f"stale or incompatible absolute-pose stage {path}")
            raw_cross[frame] = np.asarray(row["raw_cross_target"], dtype=np.float64)
        else:
            if frame:
                raw_cross[frame] = score_pose(
                    np.asarray(gt_f1[frame - 1]), np.asarray(gt_f0[frame])
                )
            row = {
                "schema": "bev_staticity_absolute_pose_frame.v2",
                "stage_identity": f"D0.2:absolute_f0_f1_charts:frame_{frame:04d}",
                "config_sha256": config_sha256,
                "frame": frame,
                "raw_cross_target": raw_cross[frame].tolist(),
                "raw_within_target": raw[frame].tolist(),
                "cross_source": (
                    "A_f1[0]=I_no_predecessor" if frame == 0 else "PoseNet(gt_f1[t-1],gt_f0[t])"
                ),
                "source_hashes": {
                    "gt_f1_previous": None if frame == 0 else _array_sha256(np.asarray(gt_f1[frame - 1])),
                    "gt_f0_current": _array_sha256(np.asarray(gt_f0[frame])),
                    "raw_within": _array_sha256(raw[frame]),
                },
            }
            _atomic_json(path, row)
    absolute_f0, absolute_f1, xi_cross, xi_within, validation = absolute_frame_trajectories(
        raw,
        raw_cross,
        s_t=s_t,
        s_r=s_r,
        pitch_rad=pitch_rad,
    )
    for frame in range(prefix):
        path = stage_dir / f"frame_{frame:04d}.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "calibrated_cross_xi": xi_cross[frame].tolist(),
            "calibrated_within_xi": xi_within[frame].tolist(),
            "absolute_f0_pose": absolute_f0[frame].tolist(),
            "absolute_f1_pose": absolute_f1[frame].tolist(),
        }
        if all(key in row for key in expected):
            if any(not np.allclose(np.asarray(row[key]), value, atol=0.0, rtol=0.0) for key, value in expected.items()):
                raise ProbeError(f"absolute-pose derived stage drifted at frame {frame}")
        else:
            row.update(expected)
            _atomic_json(path, row)
    return raw_cross, xi_cross, xi_within, absolute_f0, absolute_f1, validation


def run_probe(
    gt_cache: Path,
    output_root: Path,
    upstream: Path,
    *,
    prefix: int,
    chunk_size: int,
) -> dict[str, Any]:
    if prefix not in (64, 600) or chunk_size < 1:
        raise ProbeError("prefix must be n64 or n600 and chunk-size must be positive")
    gt_cache = gt_cache.expanduser().resolve()
    upstream = upstream.expanduser().resolve()
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
    gt_f0 = stored_npy_memmap(gt_cache, "gt_f0")
    gt_f1 = stored_npy_memmap(gt_cache, "gt_f1")
    raw_within = stored_npy_memmap(gt_cache, "gt_poses")
    if labels.shape != (PAIR_COUNT, *SCORER_HW) or margins.shape != labels.shape:
        raise ProbeError("gt_n600 labels/margins geometry mismatch")
    if gt_f0.shape != (PAIR_COUNT, *CAMERA_HW, 3) or gt_f1.shape != gt_f0.shape:
        raise ProbeError("gt_n600 native RGB geometry mismatch")
    if raw_within.shape != (PAIR_COUNT, 6):
        raise ProbeError("gt_n600 raw within-pair pose geometry mismatch")
    class_order, class_index = detect_class_order(np.asarray(labels[:64]))
    if class_order != CLASS_NAMES or set(class_index) != set(CLASS_NAMES):
        raise ProbeError(f"self-detected class map is not canonical: {class_order}, {class_index}")
    camera = camera_for_resolution(1164, 874)
    score_label, score_pose, scorer_custody = _load_frozen_scorers(upstream)
    try:
        s_t, s_r, pitch_rad, motion_custody = load_g1_worldsheet_motion(REPO)
    except ValueError as error:
        raise ProbeError(f"G1 LawRef calibration resolution failed: {error}") from error
    v2_motion_custody = bev_staticity_v2_motion_custody(motion_custody)
    implementation = Path(__file__).resolve()
    source_config = {
        "schema": SCHEMA,
        "seed": 1234,
        "gt_cache": {"path": str(gt_cache), "sha256": GT_CACHE_SHA256},
        "scorers": scorer_custody,
        "motion": v2_motion_custody,
        "class_self_detection_scale": "n64 canonical helper; map reused unchanged at n600",
        "class_order": list(class_order),
        "class_index": class_index,
        "implementation_sha256": _sha256_file(implementation),
    }
    source_config_sha256 = hashlib.sha256(_canonical_bytes(source_config)).hexdigest()
    n64_receipt = output_root / "receipt_n64.json"
    n64_gate: Mapping[str, Any] | None = None
    if prefix == 600:
        if not n64_receipt.is_file():
            raise ProbeError("n600 refused: exact-source n64 D0 receipt is missing")
        prior = json.loads(n64_receipt.read_text(encoding="utf-8"))
        n64_gate = prior.get("D0_hood_positive_control", {}).get("gate")
        if prior.get("source_config_sha256") != source_config_sha256:
            raise ProbeError("n600 refused: n64 D0 receipt source/config custody differs")
        if not isinstance(n64_gate, Mapping) or n64_gate.get("passed") is not True:
            failed = n64_gate.get("first_failed_stage") if isinstance(n64_gate, Mapping) else "MALFORMED_D0_RECEIPT"
            raise ProbeError(f"n600 refused: n64 D0 did not pass; first_failed_stage={failed}")
    labels0, label_manifest = build_canonical_f0_label_sidecar(
        gt_f0,
        gt_f1,
        labels,
        output=output_root / "canonical_f0_labels_singleton.npy",
        stage_dir=output_root / "singleton_label_stages",
        prefix=prefix,
        score_frame=score_label,
        source_binding=scorer_custody,
    )
    if prefix == 600 and int(label_manifest["f1_cache_label_mismatches"]) != 0:
        raise ProbeError(
            "n600 refused before D1-D3: singleton f1 scorer labels differ from the frozen cache"
        )
    raw_cross, xi_cross, xi_within, poses_f0, poses_f1, trajectory_validation = build_absolute_pose_stages(
        gt_f0,
        gt_f1,
        raw_within,
        prefix=prefix,
        stage_dir=output_root / "absolute_pose_stages",
        config_sha256=source_config_sha256,
        score_pose=score_pose,
        s_t=s_t,
        s_r=s_r,
        pitch_rad=pitch_rad,
    )
    z_edges = np.linspace(
        float(np.min(poses_f1[:, 2, 3])) + FORWARD_RANGE_M[0],
        float(np.max(poses_f1[:, 2, 3])) + FORWARD_RANGE_M[1],
        GRID_BINS + 1,
    )
    config = {
        **source_config,
        "geometry": {
            "v_horizon": V_HORIZON,
            "camera_height_m": CAMERA_HEIGHT_M,
            "intrinsics": {"fx": camera.fx_scorer, "fy": camera.fy_scorer, "cx": camera.cx_scorer, "cy": camera.cy_scorer},
            "forward_range_m": list(FORWARD_RANGE_M),
            "z_edges": z_edges.tolist(),
        },
        "lawrefs": {
            "v_horizon": "#327 n600 swept-optimal v_h=174",
            "camera_height": "tac.clip_profile.OPENPILOT_DEVICE_HEIGHT_M=1.22",
            "subpixel": "separatrix_asymmetry_t_subpixel_boundary_localizer_v1",
            "fisher_margin": "frozen_scorer_fisher_curvature_margin_colocation_v1",
            "xi": v2_motion_custody["calibration_authority"]["lawref_equation_ids"],
            "ground_chart": "#325/#327 openpilot IPM plus absolute tac.lie translation-first SE(3)",
        },
    }
    config_sha256 = hashlib.sha256(_canonical_bytes(config)).hexdigest()
    stage_dir = output_root / f"measurement_stages_n{prefix}"
    rows = _load_stages(stage_dir, config_sha256, prefix)
    for begin in range(0, prefix, chunk_size):
        end = min(prefix, begin + chunk_size)
        for frame in range(begin, end):
            if frame in rows:
                continue
            hood_component = bottom_connected_component(
                np.asarray(labels0[frame]), class_index["MyCar"]
            )
            hood_boundary = component_boundary_points(hood_component)
            hood_points = {
                name: hood_boundary if name == "MyCar" else np.empty((0, 2), dtype=np.float64)
                for name in CLASS_NAMES
            }
            try:
                hood_closure = hood_world_to_ego_closure(
                    hood_boundary, poses_f0[frame], camera
                )
            except ProbeError as error:
                hood_closure = {
                    "lifted_point_count": 0,
                    "max_error_m": None,
                    "p50_error_m": None,
                    "threshold_m": D0_CLOSURE_FLOOR_M,
                    "passed": False,
                    "blocker": str(error),
                }
            row = {
                "schema": STAGE_SCHEMA,
                "stage_identity": f"D0.3:hood_control:frame_{frame:04d}",
                "config_sha256": config_sha256,
                "frame": frame,
                "source_hashes": {
                    "f0_label": _array_sha256(np.asarray(labels0[frame])),
                    "f1_label": _array_sha256(np.asarray(labels[frame])),
                    "raw_cross_target": _array_sha256(raw_cross[frame]),
                    "raw_within_target": _array_sha256(np.asarray(raw_within[frame])),
                },
                "raw_cross_target": raw_cross[frame].tolist(),
                "calibrated_cross_xi": xi_cross[frame].tolist(),
                "calibrated_within_xi": xi_within[frame].tolist(),
                "absolute_f0_pose": poses_f0[frame].tolist(),
                "absolute_f1_pose": poses_f1[frame].tolist(),
                "hood_bottom_connected_pixels": int(np.count_nonzero(hood_component)),
                "hood_boundary_points": len(hood_boundary),
                "hood_world_to_ego_closure": hood_closure,
                "hood_bottom_connected_signature": frame_signatures(
                    hood_points, poses_f0[frame], z_edges, camera
                )["MyCar"],
            }
            if prefix == 600:
                points, _all_points, orientation = oriented_shallow_boundary_points(
                    np.asarray(labels[frame]), np.asarray(margins[frame]), class_index
                )
                row["orientation_counts"] = orientation
                row["signatures"] = frame_signatures(
                    points, poses_f1[frame], z_edges, camera
                )
            _atomic_json(stage_dir / f"frame_{frame:04d}.json", row)
            rows[frame] = row
        _atomic_json(
            output_root / "checkpoints" / f"prefix_{prefix}_chunk_{begin:04d}_{end:04d}.json",
            {"schema": "bev_staticity_checkpoint.v1", "config_sha256": config_sha256, "completed_through": end},
        )
    ordered = [rows[frame] for frame in range(prefix)]
    centers_ground = (z_edges[:-1] + z_edges[1:]) * 0.5
    centers_image = (np.linspace(0.0, float(SCORER_HW[0]), GRID_BINS + 1)[:-1] + np.linspace(0.0, float(SCORER_HW[0]), GRID_BINS + 1)[1:]) * 0.5
    hood_values = np.asarray(
        [row["hood_bottom_connected_signature"]["values"] for row in ordered], dtype=np.float64
    )
    hood_forwards = np.asarray(
        [row["hood_bottom_connected_signature"]["local_forward"] for row in ordered], dtype=np.float64
    )
    hood_control = summarize_stratum(
        hood_values,
        hood_forwards,
        centers_image,
        ground=False,
        fx=float(camera.fx_scorer),
    )
    closure_values = [
        row["hood_world_to_ego_closure"]["max_error_m"]
        for row in ordered
        if row["hood_world_to_ego_closure"]["max_error_m"] is not None
    ]
    closure_max = max(closure_values) if len(closure_values) == len(ordered) else math.inf
    d0_gate = d0_gate_decision(
        label_mismatches=int(label_manifest["f1_cache_label_mismatches"]),
        trajectory=trajectory_validation,
        hood_closure_max_m=closure_max,
        hood_summary=hood_control,
    )
    d0 = {
        "status": "PASS" if d0_gate["passed"] else "FAIL_CLOSED",
        "singleton_label_sidecar": label_manifest,
        "absolute_trajectory": {
            "raw_within_source": "gt_poses[t] cached within-pair PoseNet target; never redifferenced",
            "raw_cross_source": "frozen PoseNet singleton on (gt_f1[t-1],gt_f0[t])",
            "calibration": {"s_t": s_t, "s_r": s_r, "pitch_rad": pitch_rad},
            "custody": v2_motion_custody,
            "phase_chart_uses": {
                "A_f0": "canonical singleton labels0=gt_f0[t]; hood transform and closure only",
                "A_f1": "cached gt_f1[t] labels; Road/Lane D1-D3 world-frame transport",
            },
            "validation": trajectory_validation,
        },
        "hood": {
            "component": "MyCar 4-connected component touching bottom edge only",
            "temporal_boundary_summary": hood_control,
            "world_to_ego_closure_max_m": None if not math.isfinite(closure_max) else closure_max,
        },
        "gate": d0_gate,
    }
    summaries: dict[str, Any] = {}
    matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if prefix == 600:
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
        summaries["MyCar"]["bottom_connected_hood_control"] = hood_control
        summaries["MyCar"]["transform_positive_control"] = "PASS"
    for name in ("Road", "Lane"):
        if prefix == 600:
            summaries[name]["interpretation_authorized_by_n64_D0"] = True
            summaries[name]["D1_D2_hold"] = bool(summaries[name]["near_static"])
    if prefix == 600:
        summaries["Movable"]["expected_nonstatic_control"] = {
            "expected": True,
            "observed_nonstatic": not bool(summaries["Movable"]["near_static"]),
        }
    d3: dict[str, Any] = {
        "status": "NOT_RUN_N64_D0_ONLY" if prefix == 64 else "ESTIMATE_ONLY_NOT_BYTE_CLOSED_ADMISSION",
        "future_gate": "g2g2-style through-real-homography decode plus frozen-scorer admission",
        "strata": {},
    }
    if prefix == 600 and any(bool(summaries[name]["D1_D2_hold"]) for name in ("Road", "Lane")):
        from tac.lie import _se3_numpy as se3

        absolute_xi = np.stack([se3.log_se3(pose) for pose in poses_f1])
        curve = bspline_fit_error_curve(absolute_xi, [4, 8, 16, 32, 64, 128, 256, 600])
        ground_pixel_m = CAMERA_HEIGHT_M / max(float(camera.fy_scorer), 1e-9)
        acceptable = [row for row in curve if float(row["fwd_rms_m"]) <= ground_pixel_m]
        selected = acceptable[0] if acceptable else curve[-1]
        controls = fit_se3_bspline_controls(absolute_xi, int(selected["M"]))

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
    elif prefix == 600:
        d3["blocker"] = "NO_ROAD_OR_LANE_STRATUM_PASSED_HOOD_GATED_D1_D2"
    receipt = {
        "schema": SCHEMA,
        "config": config,
        "config_sha256": config_sha256,
        "source_config_sha256": source_config_sha256,
        "scale": f"n{prefix}",
        "scale_authority": "D0_HOOD_CONTROL_ONLY" if prefix == 64 else "LOAD_BEARING_N600",
        "D0_hood_positive_control": d0,
        "D1_BEV_staticity": summaries if prefix == 600 else {"status": "NOT_RUN_N64_D0_ONLY", "authorized_for_n600": d0_gate["passed"]},
        "D2_worldsheet_developability": ({
            "estimator": "directrix plus absolute-xi ruling reconstruction residual",
            "raw_K_refused": True,
            "per_stratum": {name: {"residual_px": summaries[name]["ruling_reconstruction_residual_px"], "developable_fraction": summaries[name]["developable_fraction_at_noise_floor"]} for name in CLASS_NAMES},
        } if prefix == 600 else {"status": "NOT_RUN_N64_D0_ONLY", "authorized_for_n600": d0_gate["passed"]}),
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
        "verdict_scope": "n64 is only the singleton-label/absolute-SE3/bottom-connected-hood positive control; n600 D1-D3 use absolute f1 poses and oriented shallow-side frozen SegNet boundaries; D3 is a receiver-open signature-space estimate",
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
    parser.add_argument("--upstream", type=Path, default=REPO / "upstream")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prefix", type=int, choices=(64, 600), required=True)
    parser.add_argument("--chunk-size", type=int, default=16)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    receipt = run_probe(
        args.gt_cache,
        args.output_root,
        args.upstream,
        prefix=args.prefix,
        chunk_size=args.chunk_size,
    )
    print(json.dumps({"receipt": str(args.output_root / f"receipt_n{args.prefix}.json"), "sha256": receipt["receipt_sha256"], "scale": receipt["scale"], "scale_authority": receipt["scale_authority"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
