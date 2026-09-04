#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure Task #532 uint8 resize-lattice feasibility on an honest real subset.

This tool is deliberately *not* an evaluator or a contest-score surface.  It
recomputes a temporal/fragility-stratified subset from the frozen n600 cache,
forms ``y = A(gt_f1)`` and then passes only ``(y, its exact integer numerators,
B(y))`` to the lattice solver.
The decoded uint8 candidate is scored with the frozen CPU Torch SegNet through
its real ``preprocess_input`` path at batch size one.

The full n600 scorer forward is outside this tool's authority.  The maximum
subset is twelve pairs, and every receipt is labeled
``[macOS-CPU advisory subset]`` and non-promotable.  Candidate frames are
checkpointed after each pair as durable U8LF payloads; a final aggregate sidecar
is atomically assembled and parsed back before the receipt is admitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _path in (REPO, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.canonical_equations.margin_band_satisficing_threshold_20260712 import (  # noqa: E402
    resolve_margin_band_threshold,
)
from tac.optimization.uint8_lattice_feasibility import (  # noqa: E402
    BlockSolveStatus,
    DisjointResizeOperator,
    HardOracleEvaluation,
    RepairStatus,
    parse_uint8_frame,
    repair_with_hard_oracle,
    serialize_uint8_frame,
)
from tac.subset_selection import quantile_stratified_indices  # noqa: E402

SCHEMA = "v10_uint8_lattice_feasibility_receipt.v1"
STATE_SCHEMA = "v10_uint8_lattice_feasibility_state.v1"
SIDECAR_SCHEMA = "v10_uint8_lattice_feasibility_sidecar.v1"
SIDECAR_MAGIC = b"U8LFS1"
AXIS = "[macOS-CPU advisory subset]"
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)
N_CLASSES = 5
SEED = 20260718
# m_safe is DERIVED by the canonical law, never an independent literal (ddm_ql3, 2026-09-04).
# It was hardcoded here as 0.039180326461791926 — the value derived from the n96 CONTIGUOUS-PREFIX
# delta_R. ddm_dr1 MEASURED delta_R at n600 = 0.021881818771362305 (the n96 prefix read
# 0.019590163230895963, 11.70% LOW) and every sister surface — the law module, the DSL, the hg1
# ring-0 levers, tac.subset_selection — moved to the DERIVED n600 m_safe = 0.04376363754272461.
# These two harnesses did not, because the literal carried NO provenance comment: no grep for
# "n96" could find it. Direction matters and it is the unsafe one — m_safe is a satisficing
# TARGET, so a value 11.70% too low declares pixels/candidates R-SAFE that the real uint8 noise
# can still flip (law annulus_restricted_prefix_bias_detector_v1 + margin_band_satisficing_threshold_v1).
# Resolving through the law makes staleness structurally impossible; the law falls back to the
# same MEASURED n600 constant when the artifact is absent, so this never fails open.
DEFAULT_M_SAFE = resolve_margin_band_threshold().m_safe
KNOWN_N6 = (90, 175, 277, 381, 424, 573)
MAX_SUBSET = 12
SACRED_RESULT_ROOT = Path(
    "/Users/adpena/Projects/pact/experiments/results/levelset_n600_witness_20260717T113932Z"
)
DEFAULT_CACHE = Path(
    "/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
)
DEFAULT_UPSTREAM = Path("/Users/adpena/Projects/pact/upstream")
DEFAULT_OUTPUT = REPO / ".omx/research/v10_uint8_lattice_feasibility_receipt_20260718.json"
DEFAULT_SIDECAR = Path(
    "/Volumes/VertigoDataTier/pact/evidence/v10_uint8_lattice_20260718/candidate_n6.u8lfs"
)
SSD_ROOTS = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)


class MeasurementError(RuntimeError):
    """Fail-closed custody, geometry, scorer, or output error."""


def _require_module_path(
    module_name: str,
    expected_path: Path,
    *,
    allow_absent: bool,
) -> str | None:
    """Bind an imported module to the exact source path whose bytes we hash."""

    module = sys.modules.get(module_name)
    if module is None:
        if allow_absent:
            return None
        raise MeasurementError(f"required scorer module {module_name!r} was not imported")
    loaded_file = getattr(module, "__file__", None)
    if loaded_file is None or Path(loaded_file).resolve() != expected_path.resolve():
        raise MeasurementError(
            f"{module_name} imported from wrong source: {loaded_file}; "
            f"expected {expected_path.resolve()}"
        )
    return str(Path(loaded_file).resolve())


def _prepend_exact_import_root(root: Path) -> None:
    """Put one resolved root first even when an equivalent entry exists later."""

    resolved = root.resolve()
    retained: list[str] = []
    for entry in sys.path:
        try:
            entry_resolved = Path(entry or ".").resolve()
        except (OSError, RuntimeError):
            retained.append(entry)
            continue
        if entry_resolved != resolved:
            retained.append(entry)
    sys.path[:] = [str(resolved), *retained]


def _resume_revalidation_count(
    loaded_state: object,
    *,
    config_sha256: str,
    pair_ids: Sequence[int],
) -> int:
    """Validate only checkpoint topology; stored scientific rows are discarded.

    Returning a count instead of the rows is deliberate: no stored metric,
    diagnostic, runtime, or hash field can flow into a final receipt.  The run
    re-derives every completed pair from frozen inputs and preserved stage bytes.
    """

    if not isinstance(loaded_state, dict):
        raise MeasurementError("resume state must be a JSON object")
    if (
        loaded_state.get("schema") != STATE_SCHEMA
        or loaded_state.get("config_sha256") != config_sha256
    ):
        raise MeasurementError("resume state schema/config/input custody mismatch")
    stored_rows = loaded_state.get("pair_rows")
    if not isinstance(stored_rows, list) or any(
        not isinstance(row, dict) for row in stored_rows
    ):
        raise MeasurementError("resume state pair_rows must be a list of objects")
    stored_pair_ids = [row.get("pair_id") for row in stored_rows]
    if stored_pair_ids != list(pair_ids[: len(stored_rows)]):
        raise MeasurementError("resume state completed pair prefix is noncontiguous")
    return len(stored_rows)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n")


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _refuse_unsafe_output(path: Path, field: str) -> Path:
    resolved = path.expanduser().resolve()
    for root in (Path("/tmp"), Path("/private/tmp"), Path("/var/tmp")):
        if _is_relative_to(resolved, root):
            raise MeasurementError(f"{field} must be durable, not under {root}: {resolved}")
    if _is_relative_to(resolved, SACRED_RESULT_ROOT.resolve()):
        raise MeasurementError(f"{field} may not mutate sacred result root: {resolved}")
    return resolved


def _require_ssd_sidecar(path: Path, *, allow_local: bool) -> None:
    if allow_local:
        return
    if not any(root.exists() and _is_relative_to(path, root.resolve()) for root in SSD_ROOTS):
        raise MeasurementError(
            "sidecar/stages must use the SSD waterfall; pass --allow-local-sidecar only for "
            "an explicit local-disk opt-in"
        )


def _storage_preflight(
    path: Path,
    pair_count: int,
    *,
    directory_target: bool = False,
) -> dict[str, Any]:
    existing = path if directory_target else path.parent
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    raw_frame_bytes = CAMERA_HW[0] * CAMERA_HW[1] * 3
    # Preserved per-pair stages plus the aggregate sidecar, zlib overhead, and
    # a fixed atomic-write/runtime margin.  This is deliberately conservative.
    required = 2 * pair_count * (raw_frame_bytes + (1 << 16)) + (64 << 20)
    if usage.free < required:
        raise MeasurementError(
            f"storage preflight refused: free={usage.free} < required={required} at {existing}"
        )
    return {
        "filesystem_anchor": str(existing),
        "filesystem_device": int(existing.stat().st_dev),
        "free_bytes_before": int(usage.free),
        "required_free_bytes": int(required),
        "raw_frame_bytes": int(raw_frame_bytes),
        "waterfall_order": [str(root) for root in SSD_ROOTS],
        "PASS": True,
    }


def _storage_preflights(
    sidecar: Path,
    stage_dir: Path,
    pair_count: int,
) -> dict[str, Any]:
    """Preflight every filesystem that may receive preserved frame payloads."""

    sidecar_check = _storage_preflight(sidecar, pair_count)
    stage_check = _storage_preflight(
        stage_dir,
        pair_count,
        directory_target=True,
    )
    return {
        "sidecar": sidecar_check,
        "stage_dir": stage_check,
        "same_filesystem_device": (
            sidecar_check["filesystem_device"]
            == stage_check["filesystem_device"]
        ),
        "PASS": bool(sidecar_check["PASS"] and stage_check["PASS"]),
    }


def _frontier_snapshot() -> dict[str, Any]:
    paths = (
        Path("/Users/adpena/Projects/pact/.omx/state/canonical_frontier_pointer.json"),
        Path("/Users/adpena/Projects/pact/reports/latest.md"),
    )
    return {
        str(path): {
            "exists": path.is_file(),
            "bytes": path.stat().st_size if path.is_file() else None,
            "sha256": _sha256_file(path) if path.is_file() else None,
        }
        for path in paths
    }


def _stat_tree_snapshot(root: Path) -> dict[str, Any]:
    """Read-only metadata manifest proving this tool did not address the sacred tree."""

    if not root.exists():
        return {"exists": False, "entries": 0, "metadata_sha256": None}
    digest = hashlib.sha256()
    entries = 0
    for current, directories, files in os.walk(root, followlinks=False):
        directories.sort()
        files.sort()
        current_path = Path(current)
        for name in [*directories, *files]:
            path = current_path / name
            stat = path.lstat()
            relative = path.relative_to(root).as_posix()
            row = f"{relative}\0{stat.st_mode}\0{stat.st_size}\0{stat.st_mtime_ns}\n"
            digest.update(row.encode())
            entries += 1
    return {"exists": True, "entries": entries, "metadata_sha256": digest.hexdigest()}


def stored_npy_memmap(npz_path: Path, key: str) -> np.memmap:
    """Map one unencrypted ZIP_STORED NPY member without loading the 5 GB NPZ."""

    member = key if key.endswith(".npy") else f"{key}.npy"
    with zipfile.ZipFile(npz_path) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError as exc:
            raise MeasurementError(f"cache lacks required member {member!r}") from exc
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1:
            raise MeasurementError(f"{npz_path}:{member} must be unencrypted ZIP_STORED")
        local_header = int(info.header_offset)
    with npz_path.open("rb") as handle:
        handle.seek(local_header)
        header = handle.read(30)
        if len(header) != 30:
            raise MeasurementError(f"truncated local ZIP header for {member}")
        fields = struct.unpack("<IHHHHHIIIHH", header)
        if fields[0] != 0x04034B50:
            raise MeasurementError(f"bad local ZIP signature for {member}")
        handle.seek(local_header + 30 + int(fields[-2]) + int(fields[-1]))
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise MeasurementError(f"unsupported NPY version {version} for {member}")
        data_offset = handle.tell()
    return np.memmap(
        npz_path,
        mode="r",
        dtype=dtype,
        shape=shape,
        offset=data_offset,
        order="F" if fortran else "C",
    )


def _load_cache(cache: Path, m_safe: float) -> tuple[np.memmap, np.memmap, np.memmap, np.ndarray]:
    gt_f1 = stored_npy_memmap(cache, "gt_f1")
    lstars = stored_npy_memmap(cache, "lstars")
    margins = stored_npy_memmap(cache, "margins")
    n_pairs = stored_npy_memmap(cache, "n_pairs")
    if int(np.asarray(n_pairs).reshape(())) != 600:
        raise MeasurementError("measurement requires the real n600 cache")
    if gt_f1.shape != (600, *CAMERA_HW, 3) or gt_f1.dtype != np.uint8:
        raise MeasurementError(f"gt_f1 shape/dtype mismatch: {gt_f1.shape}/{gt_f1.dtype}")
    if lstars.shape != (600, *SCORER_HW) or not np.issubdtype(lstars.dtype, np.integer):
        raise MeasurementError(f"lstars shape/dtype mismatch: {lstars.shape}/{lstars.dtype}")
    if margins.shape != (600, *SCORER_HW) or not np.issubdtype(margins.dtype, np.floating):
        raise MeasurementError(f"margins shape/dtype mismatch: {margins.shape}/{margins.dtype}")
    fragility = np.empty(600, dtype=np.float64)
    label_min, label_max = N_CLASSES, -1
    for pair_id in range(600):
        labels = np.asarray(lstars[pair_id])
        margin = np.asarray(margins[pair_id])
        if not np.isfinite(margin).all():
            raise MeasurementError(f"non-finite cached margin at pair {pair_id}")
        label_min = min(label_min, int(labels.min()))
        label_max = max(label_max, int(labels.max()))
        fragility[pair_id] = float(np.mean(margin < m_safe))
    if (label_min, label_max) != (0, N_CLASSES - 1):
        raise MeasurementError(f"cached target label range is [{label_min},{label_max}], expected [0,4]")
    return gt_f1, lstars, margins, fragility


def _select_pairs(
    lstars: np.memmap,
    fragility: np.ndarray,
    sample_pairs: int,
    explicit: Sequence[int] | None,
) -> tuple[list[int], list[dict[str, Any]], str]:
    if explicit:
        pair_ids = [int(value) for value in explicit]
        if len(set(pair_ids)) != len(pair_ids) or any(value < 0 or value >= 600 for value in pair_ids):
            raise MeasurementError("--pair-indices must be unique integers in [0,600)")
        policy = "explicit override; not the default stratified evidence selection"
    else:
        # Lifted to tac.subset_selection (ddm_ss1, 2026-08-03). This selector and
        # its verbatim twin in tools/constructive_inverse_solve_harness.py were the
        # repo's ONLY stratified pair selection, and being private + duplicated they
        # could not be reused -- part of why 110 other sites reached for [:n].
        # Equivalence to the previous inline code is MEASURED, not assumed: identical
        # output at every sample_pairs in 1..600 on the real fragility, plus 2,160
        # random trials (heavy-tie / uniform / exponential). KNOWN_N6 below is
        # unchanged and still guards this call.
        pair_ids = list(quantile_stratified_indices(sample_pairs, 600, fragility))
        policy = (
            "all-600 equal temporal strata; fragility=mean(cached margin<m_safe); alternating "
            "within-stratum 0.25/0.75 quantile; deterministic pair-index tie break; no candidate "
            "outcome peeking"
        )
        if sample_pairs == 6 and tuple(pair_ids) != KNOWN_N6:
            raise MeasurementError(f"n6 selection drifted: recomputed {pair_ids}, expected {list(KNOWN_N6)}")
    edges = np.linspace(0, 600, len(pair_ids) + 1, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    for position, pair_id in enumerate(pair_ids):
        labels = np.asarray(lstars[pair_id], dtype=np.int64)
        counts = np.bincount(labels.reshape(-1), minlength=N_CLASSES)
        rows.append(
            {
                "pair_id": pair_id,
                "temporal_stratum": None
                if explicit
                else [int(edges[position]), int(edges[position + 1])],
                "fragility_fraction_margin_below_m_safe": float(fragility[pair_id]),
                "class_histogram": counts.tolist(),
            }
        )
    return pair_ids, rows, policy


def _load_segnet(
    upstream: Path,
    *,
    cpu_threads: int,
) -> tuple[Any, Any, dict[str, str]]:
    import torch
    from safetensors.torch import load_file

    modules_path = upstream / "modules.py"
    weights_path = upstream / "models/segnet.safetensors"
    if not modules_path.is_file() or not weights_path.is_file():
        raise MeasurementError(f"upstream modules/SegNet weights missing under {upstream}")
    frame_utils_path = upstream / "frame_utils.py"
    _require_module_path("modules", modules_path, allow_absent=True)
    _require_module_path("frame_utils", frame_utils_path, allow_absent=True)
    _prepend_exact_import_root(upstream)
    from modules import SegNet

    executed_modules_path = _require_module_path(
        "modules", modules_path, allow_absent=False
    )
    executed_frame_utils_path = _require_module_path(
        "frame_utils", frame_utils_path, allow_absent=False
    )
    assert executed_modules_path is not None
    assert executed_frame_utils_path is not None
    executed_modules = {
        "modules": executed_modules_path,
        "frame_utils": executed_frame_utils_path,
    }

    torch.set_num_threads(cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    model = SegNet().eval().to("cpu")
    model.load_state_dict(load_file(str(weights_path), device="cpu"))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, torch, executed_modules


def _score_frame(segnet: Any, torch: Any, frame_hwc: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame_hwc)
    if frame.shape != (*CAMERA_HW, 3) or not np.isfinite(frame).all():
        raise MeasurementError(f"scorer frame shape/value mismatch: {frame.shape}")
    pair = torch.from_numpy(np.stack((frame, frame), axis=0)[None]).float()
    nchw = pair.permute(0, 1, 4, 2, 3).contiguous()
    if tuple(nchw.shape) != (1, 2, 3, *CAMERA_HW):
        raise MeasurementError(f"SegNet batch geometry is not batch-one canonical: {tuple(nchw.shape)}")
    with torch.inference_mode():
        scorer_input = segnet.preprocess_input(nchw)
        if tuple(scorer_input.shape) != (1, 3, *SCORER_HW):
            raise MeasurementError(f"SegNet preprocess_input geometry mismatch: {tuple(scorer_input.shape)}")
        prediction = segnet(scorer_input).argmax(dim=1)[0]
    return prediction.detach().cpu().numpy().astype(np.int64)


def _dseg_metrics(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    mismatch = np.asarray(predicted) != np.asarray(target)
    result: dict[str, Any] = {
        "mismatched_pixels": int(np.count_nonzero(mismatch)),
        "total_pixels": int(mismatch.size),
        "d_seg": float(np.mean(mismatch)),
        "per_class": {},
    }
    for class_id in range(N_CLASSES):
        mask = target == class_id
        denominator = int(np.count_nonzero(mask))
        errors = int(np.count_nonzero(mismatch & mask))
        result["per_class"][str(class_id)] = {
            "target_pixels": denominator,
            "mismatched_pixels": errors,
            "d_seg": float(errors / denominator) if denominator else None,
        }
    return result


def _projection_metrics(operator: DisjointResizeOperator, frame: np.ndarray, target: np.ndarray) -> dict[str, float]:
    residual = np.abs(operator.apply(frame) - target)
    return {
        "max_abs_A_residual": float(residual.max(initial=0.0)),
        "mean_abs_A_residual": float(residual.mean()),
    }


def _integer_projection_metrics(
    operator: DisjointResizeOperator,
    frame: np.ndarray,
    target_numerators: np.ndarray,
    target_denominator: int,
) -> dict[str, Any]:
    numerators, denominator = operator.apply_numerators(frame)
    if denominator != target_denominator:
        raise MeasurementError("integer A denominator changed within one operator")
    delta = numerators.astype(np.int64) - np.asarray(target_numerators, dtype=np.int64)
    return {
        "common_denominator": int(denominator),
        "max_abs_numerator_residual": int(np.max(np.abs(delta), initial=0)),
        "nonzero_numerator_residual_cells": int(np.count_nonzero(delta)),
        "exact_numerator_equality": bool(np.all(delta == 0)),
    }


def _solve_target_only(
    operator: DisjointResizeOperator,
    target: np.ndarray,
    target_numerators: np.ndarray,
    real_preimage: np.ndarray,
    *,
    max_nodes_per_block: int,
) -> Any:
    """Source-copy barrier: this callable has no source-frame parameter."""

    return operator.solve_uint8(
        target,
        target_numerators=target_numerators,
        reference=real_preimage,
        max_nodes_per_block=max_nodes_per_block,
    )


def _aggregate_metrics(pair_rows: Sequence[dict[str, Any]], arm: str) -> dict[str, Any]:
    total_errors = sum(int(row["arms"][arm]["seg"]["mismatched_pixels"]) for row in pair_rows)
    total_pixels = sum(int(row["arms"][arm]["seg"]["total_pixels"]) for row in pair_rows)
    classes: dict[str, Any] = {}
    for class_id in range(N_CLASSES):
        key = str(class_id)
        errors = sum(int(row["arms"][arm]["seg"]["per_class"][key]["mismatched_pixels"]) for row in pair_rows)
        pixels = sum(int(row["arms"][arm]["seg"]["per_class"][key]["target_pixels"]) for row in pair_rows)
        classes[key] = {
            "target_pixels": pixels,
            "mismatched_pixels": errors,
            "d_seg": float(errors / pixels) if pixels else None,
        }
    return {
        "mismatched_pixels": total_errors,
        "total_pixels": total_pixels,
        "d_seg": float(total_errors / total_pixels),
        "per_class": classes,
        "max_abs_A_residual": max(float(row["arms"][arm]["projection"]["max_abs_A_residual"]) for row in pair_rows),
        "mean_abs_A_residual_pair_mean": float(
            np.mean([row["arms"][arm]["projection"]["mean_abs_A_residual"] for row in pair_rows])
        ),
    }


def _write_aggregate_sidecar(
    path: Path,
    pair_ids: Sequence[int],
    stage_paths: Sequence[Path],
    *,
    config_sha256: str,
) -> None:
    header = _canonical_json(
        {
            "schema": SIDECAR_SCHEMA,
            "pair_ids": [int(value) for value in pair_ids],
            "frame_encoding": "tac.uint8_lattice_feasibility.U8LF1",
            "config_sha256": config_sha256,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            handle.write(SIDECAR_MAGIC)
            handle.write(struct.pack(">I", len(header)))
            handle.write(header)
            for pair_id, stage_path in zip(pair_ids, stage_paths, strict=True):
                payload = stage_path.read_bytes()
                handle.write(struct.pack(">IQ", int(pair_id), len(payload)))
                handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _read_exact(handle: BinaryIO, length: int, label: str) -> bytes:
    value = handle.read(length)
    if len(value) != length:
        raise MeasurementError(f"truncated aggregate sidecar {label}")
    return value


def _parse_aggregate_sidecar(path: Path) -> tuple[dict[str, Any], list[tuple[int, np.ndarray, str]]]:
    frames: list[tuple[int, np.ndarray, str]] = []
    with path.open("rb") as handle:
        if _read_exact(handle, len(SIDECAR_MAGIC), "magic") != SIDECAR_MAGIC:
            raise MeasurementError("bad aggregate sidecar magic")
        header_len = struct.unpack(">I", _read_exact(handle, 4, "header length"))[0]
        if header_len > 1 << 20:
            raise MeasurementError("aggregate sidecar header exceeds 1 MiB bound")
        header_raw = _read_exact(handle, header_len, "header")
        header = json.loads(header_raw)
        if _canonical_json(header) != header_raw or header.get("schema") != SIDECAR_SCHEMA:
            raise MeasurementError("aggregate sidecar header is noncanonical or wrong schema")
        header_pair_ids = header.get("pair_ids")
        if (
            not isinstance(header_pair_ids, list)
            or not 1 <= len(header_pair_ids) <= MAX_SUBSET
            or len(set(header_pair_ids)) != len(header_pair_ids)
            or any(not isinstance(value, int) or not 0 <= value < 600 for value in header_pair_ids)
        ):
            raise MeasurementError("aggregate sidecar pair IDs violate bounded subset contract")
        for expected_pair in header_pair_ids:
            pair_id, payload_len = struct.unpack(">IQ", _read_exact(handle, 12, "frame prefix"))
            if pair_id != expected_pair:
                raise MeasurementError("aggregate sidecar pair order/custody mismatch")
            if payload_len > CAMERA_HW[0] * CAMERA_HW[1] * 3 + (1 << 20):
                raise MeasurementError("aggregate sidecar frame payload exceeds bounded uint8 size")
            payload = _read_exact(handle, payload_len, "frame payload")
            frame = parse_uint8_frame(payload)
            frames.append((pair_id, frame, hashlib.sha256(payload).hexdigest()))
        if handle.read(1):
            raise MeasurementError("aggregate sidecar has trailing bytes")
    return header, frames


def _validate_aggregate_custody(
    header: dict[str, Any],
    parsed_frames: Sequence[tuple[int, np.ndarray, str]],
    pair_rows: Sequence[dict[str, Any]],
    *,
    config_sha256: str,
) -> None:
    """Bind aggregate payload bytes, decoded frames, and config to stages."""

    aggregate_hashes = {
        pair_id: _sha256_array(frame) for pair_id, frame, _ in parsed_frames
    }
    aggregate_payload_hashes = {
        pair_id: payload_hash for pair_id, _, payload_hash in parsed_frames
    }
    expected_hashes = {
        int(row["pair_id"]): row["candidate_stage"]["decoded_frame_sha256"]
        for row in pair_rows
    }
    expected_payload_hashes = {
        int(row["pair_id"]): row["candidate_stage"]["payload_sha256"]
        for row in pair_rows
    }
    if (
        aggregate_hashes != expected_hashes
        or aggregate_payload_hashes != expected_payload_hashes
        or header.get("config_sha256") != config_sha256
    ):
        raise MeasurementError(
            "final aggregate sidecar parse-back frame/payload/config custody mismatch"
        )


def _confound_negative_control() -> dict[str, Any]:
    operator = DisjointResizeOperator.build(camera_h=4, camera_w=4, scorer_h=2, scorer_w=2)
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def impossible_oracle(_frame: np.ndarray) -> HardOracleEvaluation:
        return HardOracleEvaluation(
            satisfied=np.zeros((2, 2), dtype=bool),
            margins=-np.ones((2, 2), dtype=np.float64),
            proposals=(),
        )

    result = repair_with_hard_oracle(frame, operator, impossible_oracle, max_iterations=1)
    if result.status != RepairStatus.STALLED_UNKNOWN or np.any(result.evaluation.satisfied):
        raise MeasurementError("known-impossible no-proposal oracle was falsely reported feasible")
    return {
        "control": "synthetic always-false hard oracle with no proposals",
        "expected": RepairStatus.STALLED_UNKNOWN.value,
        "observed": result.status.value,
        "PASS": True,
        "scope": "hard-oracle admission plumbing only; not a frozen-SegNet infeasibility claim",
    }


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.strip()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR)
    parser.add_argument("--state", type=Path, default=None, help="default: <output>.state.json")
    parser.add_argument("--stage-dir", type=Path, default=None, help="default: <sidecar>.stages/")
    parser.add_argument("--sample-pairs", type=int, default=6)
    parser.add_argument("--pair-indices", type=int, nargs="*")
    parser.add_argument("--m-safe", type=float, default=DEFAULT_M_SAFE)
    parser.add_argument("--max-nodes-per-block", type=int, default=4096)
    parser.add_argument("--cpu-threads", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-local-sidecar", action="store_true")
    return parser


def run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    np.random.seed(SEED)
    cache = args.gt_cache.expanduser().resolve()
    upstream = args.upstream_root.expanduser().resolve()
    output = _refuse_unsafe_output(args.output, "output")
    sidecar = _refuse_unsafe_output(args.sidecar, "sidecar")
    state = _refuse_unsafe_output(
        args.state if args.state is not None else output.with_suffix(output.suffix + ".state.json"),
        "state",
    )
    stage_dir = _refuse_unsafe_output(
        args.stage_dir if args.stage_dir is not None else sidecar.with_suffix(sidecar.suffix + ".stages"),
        "stage-dir",
    )
    _require_ssd_sidecar(sidecar, allow_local=args.allow_local_sidecar)
    _require_ssd_sidecar(stage_dir, allow_local=args.allow_local_sidecar)
    if len({output, sidecar, state, stage_dir}) != 4:
        raise MeasurementError("output, sidecar, state, and stage-dir paths must be distinct")
    if not 1 <= int(args.sample_pairs) <= MAX_SUBSET:
        raise MeasurementError(f"--sample-pairs must be in [1,{MAX_SUBSET}]; full n600 is governor-only")
    if args.pair_indices is not None and not 1 <= len(args.pair_indices) <= MAX_SUBSET:
        raise MeasurementError(f"explicit subset must contain [1,{MAX_SUBSET}] pairs")
    if (
        args.cpu_threads < 1
        or args.max_nodes_per_block < 1
        or not np.isfinite(args.m_safe)
        or args.m_safe <= 0.0
    ):
        raise MeasurementError("cpu threads, max nodes, and finite m-safe must be positive")
    for path, label in (
        (cache, "gt cache"),
        (upstream / "modules.py", "upstream modules"),
        (upstream / "frame_utils.py", "upstream frame utilities"),
        (upstream / "models/segnet.safetensors", "SegNet weights"),
    ):
        if not path.is_file():
            raise MeasurementError(f"{label} missing: {path}")
    protected = {
        cache,
        (upstream / "modules.py").resolve(),
        (upstream / "frame_utils.py").resolve(),
        (upstream / "models/segnet.safetensors").resolve(),
    }
    if any(path in protected for path in (output, sidecar, state)):
        raise MeasurementError("output/state/sidecar may not overwrite any frozen input")
    if output.exists():
        raise MeasurementError(f"receipt already exists; preserved evidence is never overwritten: {output}")
    if not args.resume:
        if state.exists():
            raise MeasurementError(f"state already exists; use --resume or choose a new --state: {state}")
        if sidecar.exists():
            raise MeasurementError(
                f"sidecar already exists; preserved evidence is never overwritten: {sidecar}"
            )
        if stage_dir.exists() and any(stage_dir.iterdir()):
            raise MeasurementError(
                f"stage directory contains preserved evidence; use --resume or choose a new path: {stage_dir}"
            )

    pair_count = len(args.pair_indices) if args.pair_indices is not None else int(args.sample_pairs)
    storage_preflight = _storage_preflights(sidecar, stage_dir, pair_count)
    frontier_before = _frontier_snapshot()
    sacred_before = _stat_tree_snapshot(SACRED_RESULT_ROOT)

    input_hashes = {
        "gt_n600_npz_sha256": _sha256_file(cache),
        "upstream_modules_py_sha256": _sha256_file(upstream / "modules.py"),
        "upstream_frame_utils_py_sha256": _sha256_file(upstream / "frame_utils.py"),
        "segnet_safetensors_sha256": _sha256_file(upstream / "models/segnet.safetensors"),
        "solver_module_sha256": _sha256_file(SRC / "tac/optimization/uint8_lattice_feasibility.py"),
        "measurement_tool_sha256": _sha256_file(Path(__file__).resolve()),
    }
    gt_f1, lstars, _margins, fragility = _load_cache(cache, args.m_safe)
    pair_ids, selection_rows, selection_policy = _select_pairs(
        lstars, fragility, args.sample_pairs, args.pair_indices
    )
    config = {
        "schema": STATE_SCHEMA,
        "seed": SEED,
        "pair_ids": pair_ids,
        "selection_policy": selection_policy,
        "m_safe": float(args.m_safe),
        "max_nodes_per_block": int(args.max_nodes_per_block),
        "cpu_threads": int(args.cpu_threads),
        "camera_hw": list(CAMERA_HW),
        "scorer_hw": list(SCORER_HW),
        "output_path": str(output),
        "sidecar_path": str(sidecar),
        "state_path": str(state),
        "stage_dir": str(stage_dir),
        "input_hashes": input_hashes,
    }
    config_sha256 = hashlib.sha256(_canonical_json(config)).hexdigest()
    pair_rows: list[dict[str, Any]] = []
    resume_revalidation_count = 0
    if args.resume:
        if not state.is_file():
            raise MeasurementError(f"--resume requested but state is missing: {state}")
        loaded_state = json.loads(state.read_text())
        resume_revalidation_count = _resume_revalidation_count(
            loaded_state,
            config_sha256=config_sha256,
            pair_ids=pair_ids,
        )
    else:
        # Establish an empty durable checkpoint before any pair work.  A crash
        # during pair zero can therefore resume and reconcile an orphan stage.
        _atomic_json(
            state,
            {
                "schema": STATE_SCHEMA,
                "config_sha256": config_sha256,
                "config": config,
                "pair_rows": [],
                "completed_pairs": [],
                "next_pair": pair_ids[0],
            },
        )
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_HW[0], camera_w=CAMERA_HW[1], scorer_h=SCORER_HW[0], scorer_w=SCORER_HW[1]
    )
    segnet, torch, executed_scorer_modules = _load_segnet(
        upstream, cpu_threads=args.cpu_threads
    )
    stage_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        for pair_id in pair_ids[:resume_revalidation_count]:
            stage_path = stage_dir / f"pair_{pair_id:04d}.u8lf"
            if not stage_path.is_file():
                raise MeasurementError(f"resume stage is missing for pair {pair_id}: {stage_path}")
            payload = stage_path.read_bytes()
            parse_uint8_frame(payload)

    # Resume re-derives every previously completed row.  Stored rows establish
    # only the prefix length above; no metric/diagnostic/hash field is trusted.
    for pair_id in pair_ids:
        pair_started = time.monotonic()
        source = np.asarray(gt_f1[pair_id], dtype=np.uint8).copy()
        source_argmax = _score_frame(segnet, torch, source)
        cached_target = np.asarray(lstars[pair_id], dtype=np.int64)
        cache_disagreement = int(np.count_nonzero(source_argmax != cached_target))
        source_payload = serialize_uint8_frame(source)
        source_parseback = parse_uint8_frame(source_payload)
        if not np.array_equal(source_parseback, source):
            raise MeasurementError("positive-control source U8LF parse-back changed frame bytes")
        source_parseback_argmax = _score_frame(segnet, torch, source_parseback)
        if not np.array_equal(source_parseback_argmax, source_argmax):
            raise MeasurementError("positive-control source U8LF parse-back changed hard argmax")
        source_hash = _sha256_array(source)
        source_payload_sha256 = hashlib.sha256(source_payload).hexdigest()

        float_target = operator.apply(source)
        target_numerators, target_denominator = operator.apply_numerators(source)
        target = target_numerators.astype(np.float64) / target_denominator
        target_parity = float(np.max(np.abs(float_target - target)))
        if target_parity > 5e-10:
            raise MeasurementError(
                f"float/exact-rational A parity failed at pair {pair_id}: {target_parity:.3e}"
            )
        real = operator.minimum_norm_real_preimage(target)
        clip_round = np.clip(np.rint(real), 0.0, 255.0).astype(np.uint8)
        target_hash = _sha256_array(target)
        real_hash = _sha256_array(real)
        # Delete all camera-space source bytes before crossing the target-only
        # solver boundary.  Only y, its exact numerators, and B(y) remain.
        del source, source_payload, source_parseback, source_parseback_argmax
        solve = _solve_target_only(
            operator,
            target,
            target_numerators,
            real,
            max_nodes_per_block=args.max_nodes_per_block,
        )
        if solve.frame.dtype != np.uint8 or solve.frame.shape != (*CAMERA_HW, 3):
            raise MeasurementError("lattice solver returned non-uint8 or wrong-geometry candidate")
        if not solve.certified_exact:
            raise MeasurementError(
                f"pair {pair_id} solver result is not frame-level certified exact"
            )
        if solve.aggregate_status is not BlockSolveStatus.FEASIBLE_EXACT:
            raise MeasurementError(
                f"pair {pair_id} aggregate solver status is {solve.aggregate_status}, "
                "not FEASIBLE_EXACT"
            )

        stage_payload = serialize_uint8_frame(solve.frame)
        stage_path = stage_dir / f"pair_{pair_id:04d}.u8lf"
        if stage_path.exists():
            durable_payload = stage_path.read_bytes()
            if durable_payload != stage_payload:
                raise MeasurementError(
                    f"preserved orphan stage differs from deterministic recomputation at pair {pair_id}"
                )
        else:
            _atomic_bytes(stage_path, stage_payload)
            durable_payload = stage_path.read_bytes()
        decoded = parse_uint8_frame(durable_payload)
        if not np.array_equal(decoded, solve.frame):
            raise MeasurementError("durable stage payload parse-back differs from solver candidate")

        real_argmax = _score_frame(segnet, torch, real)
        clip_argmax = _score_frame(segnet, torch, clip_round)
        candidate_argmax = _score_frame(segnet, torch, decoded)
        clip_failed = clip_argmax != source_argmax
        failed_count = int(np.count_nonzero(clip_failed))
        held_count = int(np.count_nonzero((candidate_argmax == source_argmax) & clip_failed))
        regression_count = int(
            np.count_nonzero((candidate_argmax != source_argmax) & ~clip_failed)
        )

        row = {
            "pair_id": pair_id,
            "source_frame_sha256": source_hash,
            "target_scorer_plane_sha256": target_hash,
            "target_integer_numerators_sha256": _sha256_array(target_numerators),
            "target_common_denominator": int(target_denominator),
            "float_vs_exact_rational_A_max_abs": target_parity,
            "minimum_norm_real_preimage_sha256": real_hash,
            "solver_input_contract": {
                "inputs": [
                    "target scorer plane y",
                    "exact integer numerator plane for y",
                    "minimum-norm real preimage B(y)",
                ],
                "source_frame_passed_to_solver": False,
                "target_was_formed_before_target-only_solver_boundary": True,
            },
            "source_cache_selfcheck_disagree_pixels": cache_disagreement,
            "positive_source_parseback_control": {
                "payload_sha256": source_payload_sha256,
                "frame_bytes_equal": True,
                "hard_argmax_equal": True,
                "PASS": True,
            },
            "arms": {
                "real_minimum_norm_preimage": {
                    "realization": "non-uint8 float diagnostic; not shippable",
                    "projection": _projection_metrics(operator, real, target),
                    "seg": _dseg_metrics(real_argmax, source_argmax),
                },
                "clip_round_minimum_norm": {
                    "realization": "clip(round(B(y))) uint8 baseline",
                    "projection": _projection_metrics(operator, clip_round, target),
                    "integer_projection": _integer_projection_metrics(
                        operator, clip_round, target_numerators, target_denominator
                    ),
                    "seg": _dseg_metrics(clip_argmax, source_argmax),
                },
                "exact_uint8_lattice_candidate": {
                    "realization": "durable U8LF stage payload parse-back uint8",
                    "projection": _projection_metrics(operator, decoded, target),
                    "integer_projection": _integer_projection_metrics(
                        operator, decoded, target_numerators, target_denominator
                    ),
                    "seg": _dseg_metrics(candidate_argmax, source_argmax),
                },
            },
            "clip_failed_cell_holds": {
                "clip_failed_cells": failed_count,
                "candidate_holds": held_count,
                "hold_fraction": float(held_count / failed_count) if failed_count else None,
                "candidate_regressions_on_clip_held_cells": regression_count,
            },
            "lattice_diagnostics": asdict(solve.diagnostics),
            "candidate_stage": {
                "path": str(stage_path),
                "payload_bytes": len(durable_payload),
                "payload_sha256": hashlib.sha256(durable_payload).hexdigest(),
                "decoded_frame_sha256": _sha256_array(decoded),
                "decoded_equals_hidden_source": bool(_sha256_array(decoded) == source_hash),
            },
            "pair_runtime_seconds": float(time.monotonic() - pair_started),
        }
        pair_rows.append(row)
        # Keep the old complete prefix durable until it has been fully
        # revalidated.  A crash during validation therefore loses no prior
        # checkpoint rows; the next resume repeats the bounded validation.
        if len(pair_rows) >= resume_revalidation_count:
            _atomic_json(
                state,
                {
                    "schema": STATE_SCHEMA,
                    "config_sha256": config_sha256,
                    "config": config,
                    "pair_rows": pair_rows,
                    "completed_pairs": [item["pair_id"] for item in pair_rows],
                    "next_pair": pair_ids[len(pair_rows)] if len(pair_rows) < len(pair_ids) else None,
                },
            )

    stage_paths = [stage_dir / f"pair_{pair_id:04d}.u8lf" for pair_id in pair_ids]
    for stage_path in stage_paths:
        if not stage_path.is_file():
            raise MeasurementError(f"completed state lacks preserved stage payload: {stage_path}")
    if not sidecar.exists():
        _write_aggregate_sidecar(sidecar, pair_ids, stage_paths, config_sha256=config_sha256)
    sidecar_header, parsed_frames = _parse_aggregate_sidecar(sidecar)
    _validate_aggregate_custody(
        sidecar_header,
        parsed_frames,
        pair_rows,
        config_sha256=config_sha256,
    )

    clip_fail_total = sum(int(row["clip_failed_cell_holds"]["clip_failed_cells"]) for row in pair_rows)
    candidate_hold_total = sum(int(row["clip_failed_cell_holds"]["candidate_holds"]) for row in pair_rows)
    candidate_regression_total = sum(
        int(row["clip_failed_cell_holds"]["candidate_regressions_on_clip_held_cells"])
        for row in pair_rows
    )
    diagnostics = [row["lattice_diagnostics"] for row in pair_rows]
    frontier_after = _frontier_snapshot()
    if frontier_after != frontier_before:
        raise MeasurementError("canonical frontier surfaces changed during advisory measurement")
    sacred_after = _stat_tree_snapshot(SACRED_RESULT_ROOT)
    receipt = {
        "schema": SCHEMA,
        "written_at_utc": datetime.now(UTC).isoformat(),
        "axis": AXIS,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "promotable": False,
            "subset_non_promotable": True,
            "pointer_moved": False,
            "pointer": "0.1910828242 [contest-CPU Linux x86_64] UNMOVED",
            "verdict_scope": "selected real n600-cache pairs, frozen CPU SegNet, uint8 frame1/A factor only; no PoseNet, receiver archive, contest CPU/CUDA, or full-n600 claim",
        },
        "labels": {
            "MEASURED": [
                "same-run frozen-SegNet source/candidate argmax",
                "decoded-uint8 d_seg and per-class d_seg",
                "A residuals, exact-search counters/nodes, payload bytes/hashes, runtimes",
            ],
            "DERIVED": ["disjoint block separation and exact rational bounded-Diophantine certificates"],
            "INFERRED": ["subset behavior may indicate factor-2 feasibility but cannot establish n600/archive adoption"],
        },
        "configuration": config,
        "selection": {
            "policy": selection_policy,
            "rows": selection_rows,
            "known_n6_recomputed_without_candidate_outcome_peeking": bool(
                args.pair_indices is None and args.sample_pairs == 6 and tuple(pair_ids) == KNOWN_N6
            ),
        },
        "pair_rows": pair_rows,
        "aggregate": {
            "arms": {
                arm: _aggregate_metrics(pair_rows, arm)
                for arm in (
                    "real_minimum_norm_preimage",
                    "clip_round_minimum_norm",
                    "exact_uint8_lattice_candidate",
                )
            },
            "clip_failed_cell_holds": {
                "clip_failed_cells": clip_fail_total,
                "candidate_holds": candidate_hold_total,
                "hold_fraction": float(candidate_hold_total / clip_fail_total) if clip_fail_total else None,
                "candidate_regressions_on_clip_held_cells": candidate_regression_total,
            },
            "exact_search": {
                "exact_blocks": sum(int(row["exact_blocks"]) for row in diagnostics),
                "exact_candidate_blocks": sum(
                    int(row["exact_candidate_blocks"]) for row in diagnostics
                ),
                "heuristic_blocks": sum(int(row["heuristic_blocks"]) for row in diagnostics),
                "budget_blocks": sum(int(row["budget_blocks"]) for row in diagnostics),
                "proven_affine_infeasible_blocks": sum(
                    int(row["proven_affine_infeasible_blocks"]) for row in diagnostics
                ),
                "nodes_visited": sum(int(row["nodes_visited"]) for row in diagnostics),
                "certified_exact_frames": sum(
                    int(row["certified_exact"]) for row in diagnostics
                ),
                "aggregate_statuses": sorted(
                    {str(row["aggregate_status"]) for row in diagnostics}
                ),
                "decoded_frames_with_exact_numerator_equality": sum(
                    int(
                        row["arms"]["exact_uint8_lattice_candidate"]["integer_projection"][
                            "exact_numerator_equality"
                        ]
                    )
                    for row in pair_rows
                ),
                "max_abs_decoded_numerator_residual": max(
                    int(
                        row["arms"]["exact_uint8_lattice_candidate"]["integer_projection"][
                            "max_abs_numerator_residual"
                        ]
                    )
                    for row in pair_rows
                ),
                "nonzero_decoded_numerator_residual_cells": sum(
                    int(
                        row["arms"]["exact_uint8_lattice_candidate"]["integer_projection"][
                            "nonzero_numerator_residual_cells"
                        ]
                    )
                    for row in pair_rows
                ),
            },
        },
        "confound_controls": {
            "named_confound": "soft margin improvement masquerading as a real argmax-cell flip after uint8/resize/parse-back",
            "positive_control": {
                "control": "source uint8 serialized/parsed through U8LF then scored in the same frozen-SegNet run",
                "cache_disagree_pixels": sum(int(row["source_cache_selfcheck_disagree_pixels"]) for row in pair_rows),
                "source_parseback_frame_and_argmax_equal_all_pairs": all(
                    bool(row["positive_source_parseback_control"]["PASS"]) for row in pair_rows
                ),
                "PASS": all(
                    bool(row["positive_source_parseback_control"]["PASS"])
                    for row in pair_rows
                ),
            },
            "negative_control": _confound_negative_control(),
            "primary_verdict_surface": "hard target-vs-predicted argmax only; no soft hinge is admitted",
        },
        "sidecar": {
            "honest_name": "incremental uint8 lattice feasibility sidecar; NOT a contest archive",
            "path": str(sidecar),
            "bytes": sidecar.stat().st_size,
            "sha256": _sha256_file(sidecar),
            "frame_count": len(parsed_frames),
            "parse_back_all_frame_hashes_match": True,
            "preserved_stage_dir": str(stage_dir),
            "cleanup_policy": "preserve all per-pair stage payloads; they are measurement evidence, not scratch",
        },
        "resumability": {
            "schema": STATE_SCHEMA,
            "state_path": str(state),
            "state_sha256": _sha256_file(state),
            "config_sha256": config_sha256,
            "prior_completed_pairs_rederived": int(resume_revalidation_count),
            "stored_pair_metric_fields_reused": False,
            "completed_pairs": [int(row["pair_id"]) for row in pair_rows],
            "next_pair": None,
            "all_per_pair_stages_preserved": True,
        },
        "custody": {
            "input_hashes": input_hashes,
            "executed_scorer_module_paths": executed_scorer_modules,
            "executed_scorer_paths_match_hashed_inputs": True,
            "frontier_surfaces_before": frontier_before,
            "frontier_surfaces_after": frontier_after,
            "frontier_surfaces_unchanged": True,
            "sacred_result_root": str(SACRED_RESULT_ROOT),
            "sacred_metadata_before": sacred_before,
            "sacred_metadata_after": sacred_after,
            "sacred_metadata_unchanged_during_measurement": sacred_before == sacred_after,
            "tool_write_paths_are_disjoint_from_sacred_root": True,
            "storage_preflight": storage_preflight,
            "git_head": _git_head(),
            "command": shlex.join(sys.argv),
            "seed": SEED,
        },
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_device": "cpu",
            "torch_threads": args.cpu_threads,
            "determinism_environment": {
                key: os.environ.get(key)
                for key in ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
            },
            "segnet_preprocess_input_batch_geometry": [1, 2, 3, *CAMERA_HW],
        },
        "runtime_seconds": float(time.monotonic() - started),
        "remaining_blockers": [
            "full n600 governed scorer replay",
            "PoseNet/both-frame interaction",
            "complete receiver archive and inflate custody",
            "contest-CPU and contest-CUDA exact evaluation",
            "independent MAIN landing review",
        ],
    }
    _atomic_json(output, receipt)
    print(json.dumps({"output": str(output), "sidecar": str(sidecar), "pairs": pair_ids, "axis": AXIS}, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(_build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
