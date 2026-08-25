#!/usr/bin/env python3
"""Fresh per-candidate Schur solve and receiver-close package for fx5/rc2.

This module consumes a *realized* n600 candidate frame-1 camera field.  For
every pair it re-solves the shipped 12-dimensional frame-0 carrier against the
candidate frame-1 object with the official PoseNet first-six output, rebuilds
the carrier and counted residual sections, races the real Brotli/ZIP container,
and verifies the resulting single-``p`` archive through a freshly copied
receiver tree.  A solve receipt is content-bound to the candidate semantic
payload and camera field, so stale QS4-style compensation cannot compile.

The module does not launch a scorer job and makes no score claim.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import brotli
import numpy as np
import torch
from safetensors.torch import load_file

from experiments import ddm_jo2_residual_runtime as residual_runtime
from experiments import ddm_rx1_rate_representation_attack as rx1
from tac.pr130_lift.pose.lifted.carrier_codec import encode_compact_carrier
from tac.process_group_kill import run_in_process_group

REPO: Final = Path(__file__).resolve().parents[1]
FX5_RUNTIME: Final = Path("/Volumes/APDataStore/pact/ddm_fx5/candidate_runtime_fx5")
FX5_ARCHIVE: Final = FX5_RUNTIME / "archive.zip"
FX5_ARCHIVE_BYTES: Final = 180_386
FX5_ARCHIVE_SHA256: Final = (
    "4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841"
)
N: Final = 600
D: Final = 12
POSE_DIMS: Final = 6
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
RAW_BYTES: Final = N * 2 * CAMERA_H * CAMERA_W * 3
BASIS_H: Final = 24
BASIS_W: Final = 32
GN_DAMPING: Final = 0.01
MAX_CODE_STEP: Final = 32.0
NEIGHBOUR_DIMS: Final = 3
NEIGHBOUR_RADIUS: Final = 2
POSE_BATCH: Final = 32
AXIS: Final = "[receiver-close component; no score authority]"


class JO2ReceiverCloseError(RuntimeError):
    """A source, fresh-solve, coder, or receiver invariant failed closed."""


@dataclass(frozen=True)
class RuntimeModules:
    residual_archive: Any
    carrier_repack: Any
    carrier_codec: Any
    coefficient_codec: Any
    coefficient_predictor: Any
    frame0_selector: Any
    renderer: Any


@dataclass(frozen=True)
class CarrierSurface:
    parts: Any
    basis_scales: np.ndarray
    basis_codes: np.ndarray
    coefficient_scales: np.ndarray
    codes: np.ndarray
    normalized_basis: torch.Tensor
    selector: bytes
    selector_modes: Sequence[Any]
    selector_indices: np.ndarray
    outer: dict[str, Any]


def sha256_file(path: Path) -> str:
    if not path.is_file():
        raise JO2ReceiverCloseError(f"required file is absent: {path}")
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_record(record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"])).resolve()
    if (
        not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or sha256_file(path) != str(record["sha256"])
    ):
        raise JO2ReceiverCloseError(f"retained artifact drifted: {path}")
    return path


def atomic_bytes(path: Path, payload: bytes, *, executable: bool = False) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if executable:
            temporary.chmod(0o755)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return file_record(path)


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    return atomic_bytes(path, payload)


def atomic_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(value), allow_pickle=False)
    return atomic_bytes(path, buffer.getvalue())


def _runtime_modules(runtime_root: Path) -> RuntimeModules:
    root = runtime_root.resolve()
    if not (root / "runtime/residual_archive.py").is_file():
        raise JO2ReceiverCloseError(f"runtime tree is incomplete: {root}")
    for name in tuple(sys.modules):
        if name == "runtime" or name.startswith("runtime.") or name in {
            "carrier_codec",
            "inflate",
        }:
            sys.modules.pop(name, None)
    # Insert the tree first and cpr1 second so the renderer's ``inflate.py``
    # wins the top-level module name over the outer CLI wrapper.
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "cpr1"))
    try:
        return RuntimeModules(
            residual_archive=importlib.import_module("runtime.residual_archive"),
            carrier_repack=importlib.import_module("runtime.carrier_repack"),
            carrier_codec=importlib.import_module("carrier_codec"),
            coefficient_codec=importlib.import_module(
                "runtime.entropy.coefficient_ar1_codec"
            ),
            coefficient_predictor=importlib.import_module(
                "runtime.entropy.coefficient_predictor"
            ),
            frame0_selector=importlib.import_module("runtime.frame0_selector"),
            renderer=importlib.import_module("inflate"),
        )
    finally:
        sys.path.pop(0)
        sys.path.pop(0)


def _single_member(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if (
            len(infos) != 1
            or infos[0].filename != "p"
            or infos[0].compress_type != zipfile.ZIP_STORED
        ):
            raise JO2ReceiverCloseError("archive must contain exactly one stored member p")
        member = archive.read("p")
        if archive.testzip() is not None:
            raise JO2ReceiverCloseError("archive CRC validation failed")
    return member


def load_surface(
    archive: Path = FX5_ARCHIVE,
    runtime_root: Path = FX5_RUNTIME,
) -> tuple[CarrierSurface, RuntimeModules]:
    if archive.stat().st_size != FX5_ARCHIVE_BYTES or sha256_file(archive) != FX5_ARCHIVE_SHA256:
        raise JO2ReceiverCloseError("fx5 authority archive pin differs")
    modules = _runtime_modules(runtime_root)
    parts = modules.residual_archive.read_residual_archive(archive)
    if parts.compensation_blob is not None:
        raise JO2ReceiverCloseError("fx5 base unexpectedly carries prior compensation")
    carrier, selector = modules.carrier_repack.split_frame0_selector_carrier(
        parts.carrier_blob
    )
    if selector is None:
        raise JO2ReceiverCloseError("fx5 frame-0 selector is absent")
    canonical = modules.carrier_repack.materialize_cpr1(
        carrier, SimpleNamespace(N=N, CARRIER_DIM=D)
    )
    basis_scales, basis_codes, coefficient_scales, encoded = (
        modules.carrier_codec.decode_compact_carrier(
            canonical,
            basis_count=D * 3 * BASIS_H * BASIS_W,
            frames=N,
            dimensions=D,
        )
    )
    delta = (encoded.astype(np.int64) >> 1) ^ -(encoded.astype(np.int64) & 1)
    codes = np.cumsum(delta, axis=0) & 0xFFF
    codes = np.where(codes >= 0x800, codes - 0x1000, codes).astype(np.int32)
    rebuilt = encode_compact_carrier(
        basis_scales,
        basis_codes,
        coefficient_scales,
        delta_zigzag_from_signed_codes(codes),
    )
    if rebuilt != canonical:
        raise JO2ReceiverCloseError("fx5 CPR1 decode/re-encode differs")
    raw_basis = torch.from_numpy(
        basis_codes.reshape(D, 3, BASIS_H, BASIS_W).astype(np.float32)
    ) * torch.from_numpy(np.asarray(basis_scales, dtype=np.float32))[:, None, None, None]
    basis = torch.nn.functional.interpolate(
        raw_basis, size=(384, 512), mode="bicubic", align_corners=False
    )
    basis = basis - basis.mean(dim=(1, 2, 3), keepdim=True)
    basis = basis / basis.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-5)
    modes, indices = modules.frame0_selector.decode_selector(selector)

    member = _single_member(archive)
    header = rx1.RX1_HEADER
    magic, version, codec, table_mode, reserved, hpac_size, semantic_size, carrier_size = (
        header.unpack_from(member)
    )
    if (
        magic != rx1.RX1_MAGIC
        or version != rx1.RX1_VERSION
        or reserved & ~0x0F
    ):
        raise JO2ReceiverCloseError("fx5 RX1 header differs")
    offset = header.size
    hpac_stream = member[offset : offset + hpac_size]
    offset += hpac_size
    semantic_stream = member[offset : offset + semantic_size]
    offset += semantic_size
    carrier_stream = member[offset : offset + carrier_size]
    offset += carrier_size
    return (
        CarrierSurface(
            parts=parts,
            basis_scales=np.asarray(basis_scales, dtype=np.float32),
            basis_codes=np.asarray(basis_codes, dtype=np.int8),
            coefficient_scales=np.asarray(coefficient_scales, dtype=np.float32),
            codes=codes,
            normalized_basis=basis,
            selector=selector,
            selector_modes=modes,
            selector_indices=np.asarray(indices),
            outer={
                "codec": codec,
                "table_mode": table_mode,
                "source_reserved": reserved,
                "hpac_stream": hpac_stream,
                "semantic_stream": semantic_stream,
                "carrier_stream": carrier_stream,
                "tail": member[offset:],
            },
        ),
        modules,
    )


def delta_zigzag_from_signed_codes(codes: np.ndarray) -> np.ndarray:
    value = np.asarray(codes, dtype=np.int64)
    if value.shape != (N, D) or np.any(value < -2048) or np.any(value > 2047):
        raise JO2ReceiverCloseError("signed carrier lattice differs")
    unsigned = value & 0xFFF
    previous = np.zeros_like(unsigned)
    previous[1:] = unsigned[:-1]
    delta_unsigned = (unsigned - previous) & 0xFFF
    delta = np.where(delta_unsigned >= 0x800, delta_unsigned - 0x1000, delta_unsigned)
    return (((delta << 1) ^ (delta >> 63)) & 0xFFF).astype(np.int32)


def load_posenet(upstream_root: Path) -> Any:
    root = upstream_root.resolve()
    sys.modules.pop("modules", None)
    sys.path.insert(0, str(root))
    try:
        modules = importlib.import_module("modules")
    finally:
        sys.path.pop(0)
    network = modules.PoseNet().eval().cpu()
    network.load_state_dict(load_file(modules.posenet_sd_path, device="cpu"))
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


def render_frame0(
    surface: CarrierSurface,
    modules: RuntimeModules,
    codes: np.ndarray,
    pair: int,
) -> np.ndarray:
    values = np.asarray(codes, dtype=np.int32)
    if values.ndim != 2 or values.shape[1] != D or np.any(values < -2048) or np.any(values > 2047):
        raise JO2ReceiverCloseError("frame-0 code candidate differs")
    coefficients = torch.from_numpy(
        values.astype(np.float32) * surface.coefficient_scales[None]
    )
    with torch.inference_mode():
        carrier = torch.einsum("bk,kchw->bchw", coefficients, surface.normalized_basis)
        slave = (127.5 + 64.0 * carrier / math.sqrt(D)).clamp(0.0, 255.0).round()
        slave = torch.nn.functional.interpolate(
            slave, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
        ).clamp(0.0, 255.0).round()
        result = slave.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    mode_index = int(surface.selector_indices[pair])
    return np.asarray(
        modules.frame0_selector.apply_pixel_mode(
            result, surface.selector_modes[mode_index]
        ),
        dtype=np.uint8,
    )


def pose_vectors(posenet: Any, pairs: np.ndarray) -> np.ndarray:
    value = np.asarray(pairs)
    if value.ndim != 5 or value.shape[1:] != (2, CAMERA_H, CAMERA_W, 3):
        raise JO2ReceiverCloseError(f"PoseNet input geometry differs: {value.shape}")
    with torch.inference_mode():
        tensor = torch.from_numpy(np.ascontiguousarray(value)).permute(0, 1, 4, 2, 3).float()
        output = posenet(posenet.preprocess_input(tensor))["pose"][..., :POSE_DIMS]
    return output.cpu().numpy().astype(np.float32, copy=False)


def jacobian_probe_offsets(coordinate: int) -> tuple[tuple[int, int], float, str]:
    """Return two in-domain finite-difference probes for one int12 coordinate.

    Interior coordinates keep the second-order central difference at unit
    step. An int12 endpoint cannot support that stencil, so it uses the
    matched unit-step inward one-sided difference. That endpoint stencil is
    first-order accurate (O(h)) rather than the central stencil's O(h**2).
    Both returned probe coordinates are asserted in-domain before use.
    """
    value = int(coordinate)
    if not -2048 <= value <= 2047:
        raise JO2ReceiverCloseError(f"carrier coefficient is outside int12: {value}")
    if value == -2048:
        offsets, denominator, mode = (0, 1), 1.0, "forward_one_sided_first_order"
    elif value == 2047:
        offsets, denominator, mode = (-1, 0), 1.0, "backward_one_sided_first_order"
    else:
        offsets, denominator, mode = (-1, 1), 2.0, "central_second_order"
    probes = tuple(value + offset for offset in offsets)
    if len(probes) != 2 or not all(-2048 <= probe <= 2047 for probe in probes):
        raise JO2ReceiverCloseError(
            f"finite-difference probes leave int12 domain: value={value},probes={probes}"
        )
    if probes[0] == probes[1]:
        raise JO2ReceiverCloseError("finite-difference probes are not distinct")
    return (int(offsets[0]), int(offsets[1])), denominator, mode


def evaluate_codes(
    *,
    surface: CarrierSurface,
    modules: RuntimeModules,
    posenet: Any,
    codes: Sequence[np.ndarray],
    master: np.ndarray,
    pair: int,
    stage_root: Path,
    retention: Any | None = None,
) -> np.ndarray:
    code_array = np.stack([np.asarray(value, dtype=np.int32) for value in codes])
    vectors: list[np.ndarray] = []
    for first in range(0, len(code_array), POSE_BATCH):
        last = min(first + POSE_BATCH, len(code_array))
        root = stage_root / f"batch_{first:04d}_{last:04d}"
        result_path = root / "RESULT.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if retention is not None:
                retention.verify_explored_result(result)
            vectors.append(np.load(verify_record(result["pose_vectors"]), allow_pickle=False))
            continue
        batch_codes = code_array[first:last]
        slaves = render_frame0(surface, modules, batch_codes, pair)
        masters = np.repeat(np.asarray(master, dtype=np.uint8)[None], len(batch_codes), axis=0)
        inputs = np.stack((slaves, masters), axis=1)
        output = pose_vectors(posenet, inputs)
        if retention is None:
            records = {
                "codes": atomic_npy(root / "codes.int32.npy", batch_codes),
                "slave_camera": atomic_npy(root / "slave_camera.uint8.npy", slaves),
                "pose_input": atomic_npy(root / "pose_input.uint8.npy", inputs),
                "pose_vectors": atomic_npy(root / "pose_vectors.float32.npy", output),
                "retention_mode": "FULL_BYTES",
            }
        else:
            records = retention.retain_explored(
                root=root,
                pair=pair,
                base_codes=np.asarray(surface.codes[pair], dtype=np.int32),
                codes=batch_codes,
                slave_camera=slaves,
                pose_input=inputs,
                pose_vectors=output,
            )
        atomic_json(
            result_path,
            {
                "schema": "ddm_jo2_pose_batch.v1",
                "pair": pair,
                "candidate_first": first,
                "candidate_last_exclusive": last,
                **records,
                "axis": AXIS,
                "score_claim": False,
            },
        )
        vectors.append(output)
    joined = np.concatenate(vectors, axis=0)
    if joined.shape != (len(code_array), POSE_DIMS):
        raise JO2ReceiverCloseError("PoseNet output census differs")
    atomic_npy(stage_root / "ALL_CODES.int32.npy", code_array)
    atomic_npy(stage_root / "ALL_POSE_VECTORS.float32.npy", joined)
    return joined


def _damped_solve(jacobian: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(jacobian, dtype=np.float64)
    error = np.asarray(residual, dtype=np.float64)
    singular = np.linalg.svd(matrix, compute_uv=False)
    largest = float(singular[0]) if singular.size else 0.0
    tolerance = np.finfo(np.float64).eps * max(matrix.shape) * largest
    rank = int(np.count_nonzero(singular > tolerance))
    smallest = float(singular[rank - 1]) if rank else 0.0
    condition = math.inf if not rank or smallest == 0.0 else largest / smallest
    ridge = float((GN_DAMPING * largest) ** 2)
    if ridge:
        normal = matrix.T @ matrix + ridge * np.eye(D)
        update = np.linalg.solve(normal, matrix.T @ error)
    else:
        update = np.linalg.lstsq(matrix, error, rcond=None)[0]
    norm = float(np.linalg.norm(update))
    if norm > MAX_CODE_STEP:
        update *= MAX_CODE_STEP / norm
    return update, {"rank": rank, "condition": condition, "ridge_lambda": ridge}


def _nearby_candidates(
    base: np.ndarray, centre: np.ndarray, jacobian: np.ndarray, update: np.ndarray
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...]]:
    scores = np.linalg.norm(jacobian, axis=0) * (1.0 + np.abs(update))
    active = tuple(sorted(np.argsort(-scores, kind="stable")[:NEIGHBOUR_DIMS].tolist()))
    candidates: list[np.ndarray] = []
    for offsets in np.ndindex(*([2 * NEIGHBOUR_RADIUS + 1] * len(active))):
        candidate = centre.copy()
        for index, dimension in enumerate(active):
            candidate[dimension] += int(offsets[index]) - NEIGHBOUR_RADIUS
        if np.all((candidate >= -2048) & (candidate <= 2047)):
            candidates.append(candidate)
    if not candidates:
        candidates.append(base.copy())
    return tuple(candidates), active


def candidate_object_fingerprint(
    *,
    pair: int,
    semantic_object_sha256: str,
    candidate_master: Mapping[str, Any],
    base_pose6: Mapping[str, Any],
) -> str:
    if (
        not 0 <= pair < N
        or len(semantic_object_sha256) != 64
        or any(value not in "0123456789abcdef" for value in semantic_object_sha256)
    ):
        raise JO2ReceiverCloseError("candidate fingerprint inputs differ")
    body = {
        "pair": pair,
        "semantic_object_sha256": semantic_object_sha256,
        "candidate_master": {
            "bytes": int(candidate_master["bytes"]),
            "sha256": str(candidate_master["sha256"]),
        },
        "base_pose6": {
            "bytes": int(base_pose6["bytes"]),
            "sha256": str(base_pose6["sha256"]),
        },
        "base_archive_sha256": FX5_ARCHIVE_SHA256,
    }
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def materialize_candidate_frame0(
    *,
    surface: CarrierSurface,
    modules: RuntimeModules,
    codes: np.ndarray,
    output: Path,
) -> dict[str, Any]:
    """Persist the intended final frame-0 n600 field with per-pair resume."""
    final = output / "candidate_frame0_n600.uint8.npy"
    progress = output / "candidate_frame0_progress.json"
    temporary = final.with_name(f".{final.name}.partial")
    if final.is_file():
        value = np.load(final, mmap_mode="r", allow_pickle=False)
        if value.shape != (N, CAMERA_H, CAMERA_W, 3) or value.dtype != np.uint8:
            raise JO2ReceiverCloseError("retained candidate frame-0 field differs")
        return file_record(final)
    output.mkdir(parents=True, exist_ok=True)
    cursor = 0
    mode = "w+"
    if temporary.is_file():
        if not progress.is_file():
            raise JO2ReceiverCloseError("frame-0 partial exists without resume cursor")
        receipt = json.loads(progress.read_text(encoding="utf-8"))
        cursor = int(receipt["next_pair"])
        if not 0 <= cursor <= N:
            raise JO2ReceiverCloseError("frame-0 resume cursor differs")
        mode = "r+"
    field = np.lib.format.open_memmap(
        temporary,
        mode=mode,
        dtype=np.uint8,
        shape=(N, CAMERA_H, CAMERA_W, 3),
    )
    for pair in range(cursor, N):
        field[pair] = render_frame0(
            surface,
            modules,
            np.asarray(codes[pair : pair + 1], dtype=np.int32),
            pair,
        )[0]
        field.flush()
        atomic_json(
            progress,
            {
                "schema": "ddm_jo2_frame0_materialization_cursor.v1",
                "next_pair": pair + 1,
                "pair_denominator": N,
                "partial_path": str(temporary.resolve()),
                "payload_preserved_on_failure": True,
            },
        )
    del field
    os.replace(temporary, final)
    atomic_json(
        progress,
        {
            "schema": "ddm_jo2_frame0_materialization_cursor.v1",
            "next_pair": N,
            "pair_denominator": N,
            "final_path": str(final.resolve()),
            "complete": True,
        },
    )
    return file_record(final)


def solve_fresh_compensation(
    *,
    candidate_master: Mapping[str, Any],
    base_pose6: Mapping[str, Any],
    semantic_object_sha256: str,
    output: Path,
    posenet: Any,
    archive: Path = FX5_ARCHIVE,
    runtime_root: Path = FX5_RUNTIME,
    retention: Any | None = None,
) -> dict[str, Any]:
    """Solve every pair; a subset solve is deliberately not a final package."""
    master_path = verify_record(candidate_master)
    pose_path = verify_record(base_pose6)
    masters = np.load(master_path, mmap_mode="r", allow_pickle=False)
    baseline = np.load(pose_path, mmap_mode="r", allow_pickle=False)
    if masters.shape != (N, CAMERA_H, CAMERA_W, 3) or masters.dtype != np.uint8:
        raise JO2ReceiverCloseError("candidate master field must be n600 camera uint8")
    if baseline.shape != (N, POSE_DIMS) or baseline.dtype not in (np.float32, np.float64):
        raise JO2ReceiverCloseError("base Pose6 table must be n600x6 float")
    surface, modules = load_surface(archive, runtime_root)
    solved_codes = surface.codes.copy()
    rows: list[dict[str, Any]] = []
    for pair in range(N):
        root = output / f"pairs/pair_{pair:04d}"
        result_path = root / "RESULT.json"
        fingerprint = candidate_object_fingerprint(
            pair=pair,
            semantic_object_sha256=semantic_object_sha256,
            candidate_master=candidate_master,
            base_pose6=base_pose6,
        )
        if result_path.is_file():
            row = json.loads(result_path.read_text(encoding="utf-8"))
            if row.get("candidate_object_fingerprint_sha256") != fingerprint:
                raise JO2ReceiverCloseError(f"stale compensation at pair {pair}")
            resumed_codes = np.asarray(row["final_codes"], dtype=np.int32)
            if (
                resumed_codes.shape != (D,)
                or np.any(resumed_codes < -2048)
                or np.any(resumed_codes > 2047)
            ):
                raise JO2ReceiverCloseError(f"resumed carrier codes differ at pair {pair}")
            if retention is not None:
                retention.verify_winner(row.get("winner_retention"))
            solved_codes[pair] = resumed_codes
            rows.append(row)
            continue
        base_codes = surface.codes[pair].copy()
        master = np.asarray(masters[pair])
        event = evaluate_codes(
            surface=surface,
            modules=modules,
            posenet=posenet,
            codes=(base_codes,),
            master=master,
            pair=pair,
            stage_root=root / "stage_10_event",
            retention=retention,
        )[0]
        leak = event.astype(np.float64) - baseline[pair].astype(np.float64)
        jacobian_codes = [base_codes.copy()]
        jacobian_probe_modes: list[str] = []
        for dimension in range(D):
            offsets, _denominator, mode = jacobian_probe_offsets(int(base_codes[dimension]))
            jacobian_probe_modes.append(mode)
            for delta in offsets:
                candidate = base_codes.copy()
                candidate[dimension] += delta
                if not -2048 <= candidate[dimension] <= 2047:
                    raise JO2ReceiverCloseError(
                        "endpoint-safe finite-difference probe left the int12 domain"
                    )
                jacobian_codes.append(candidate)
        jacobian_vectors = evaluate_codes(
            surface=surface,
            modules=modules,
            posenet=posenet,
            codes=tuple(jacobian_codes),
            master=master,
            pair=pair,
            stage_root=root / "stage_20_jacobian",
            retention=retention,
        )
        jacobian = np.empty((POSE_DIMS, D), dtype=np.float64)
        for dimension in range(D):
            _offsets, denominator, _mode = jacobian_probe_offsets(
                int(base_codes[dimension])
            )
            jacobian[:, dimension] = (
                jacobian_vectors[2 + 2 * dimension].astype(np.float64)
                - jacobian_vectors[1 + 2 * dimension].astype(np.float64)
            ) / denominator
        atomic_npy(root / "stage_20_jacobian/J_POSE0.float64.npy", jacobian)
        update, diagnostics = _damped_solve(jacobian, -leak)
        centre = np.rint(base_codes.astype(np.float64) + update).astype(np.int32)
        centre = np.clip(centre, -2048, 2047)
        neighbourhood, active = _nearby_candidates(base_codes, centre, jacobian, update)
        vectors = evaluate_codes(
            surface=surface,
            modules=modules,
            posenet=posenet,
            codes=neighbourhood,
            master=master,
            pair=pair,
            stage_root=root / "stage_30_integer_cube",
            retention=retention,
        )
        objectives = np.mean(
            np.square(vectors.astype(np.float64) - baseline[pair][None]), axis=1
        )
        best = min(range(len(neighbourhood)), key=lambda index: (float(objectives[index]), index))
        current = np.asarray(neighbourhood[best], dtype=np.int32)
        objective = float(objectives[best])
        final_vector = np.asarray(vectors[best], dtype=np.float32)
        passes = 0
        while True:
            candidates = [current.copy()]
            for dimension in range(D):
                for delta in (-1, 1):
                    candidate = current.copy()
                    candidate[dimension] += delta
                    if -2048 <= candidate[dimension] <= 2047:
                        candidates.append(candidate)
            vectors = evaluate_codes(
                surface=surface,
                modules=modules,
                posenet=posenet,
                codes=tuple(candidates),
                master=master,
                pair=pair,
                stage_root=root / f"stage_40_descent/pass_{passes:04d}",
                retention=retention,
            )
            objectives = np.mean(
                np.square(vectors.astype(np.float64) - baseline[pair][None]), axis=1
            )
            best = min(range(len(candidates)), key=lambda index: (float(objectives[index]), index))
            value = float(objectives[best])
            final_vector = np.asarray(vectors[best], dtype=np.float32)
            passes += 1
            if not value < objective:
                break
            current = candidates[best]
            objective = value
        winner_retention = None
        if retention is not None:
            if not np.array_equal(np.asarray(candidates[best], dtype=np.int32), current):
                raise JO2ReceiverCloseError(
                    "converged winner code differs from the final explored Pose6 row"
                )
            winner_repeat = retention.recompute_selected_winner(
                root=root / "stage_50_winner_repeat_batch",
                pair=pair,
                base_codes=base_codes,
                candidate_codes=candidates,
                selected_index=best,
                master=master,
                surface=surface,
                modules=modules,
                posenet=posenet,
            )
            winner_pose = np.asarray(winner_repeat["pose_vector"], dtype=np.float32)
            if not np.array_equal(winner_pose, final_vector):
                raise JO2ReceiverCloseError(
                    f"winner deterministic repeat differs from explored Pose6 at pair {pair}"
                )
            winner_retention = retention.retain_winner(
                root=root / "stage_50_winner_full",
                pair=pair,
                base_codes=base_codes,
                codes=np.asarray(winner_repeat["codes"], dtype=np.int32),
                slave_camera=np.asarray(winner_repeat["slave_camera"], dtype=np.uint8),
                pose_input=np.asarray(winner_repeat["pose_input"], dtype=np.uint8),
                pose_vector=winner_pose,
            )
        solved_codes[pair] = current
        row = {
            "schema": "ddm_jo2_fresh_schur_pair.v1",
            "pair": pair,
            "candidate_object_fingerprint_sha256": fingerprint,
            "semantic_object_sha256": semantic_object_sha256,
            "candidate_master": dict(candidate_master),
            "base_codes": base_codes.tolist(),
            "final_codes": current.tolist(),
            "final_code_delta": (current - base_codes).tolist(),
            "float_update": update.tolist(),
            "active_dimensions": list(active),
            "integer_cube_candidates": len(neighbourhood),
            "coordinate_descent_full_passes": passes,
            "final_objective_mse_to_base_pose6": objective,
            "final_pose6": final_vector.tolist(),
            "jacobian": diagnostics,
            "jacobian_probe_modes": jacobian_probe_modes,
            "winner_retention": winner_retention,
            "all_materialized_payloads_retained": True,
            "score_claim": False,
        }
        atomic_npy(root / "FINAL_CODES.int32.npy", current)
        atomic_npy(root / "FINAL_POSE6.float32.npy", final_vector)
        atomic_json(result_path, row)
        rows.append(row)
    codes_record = atomic_npy(output / "candidate_codes.int32.npy", solved_codes)
    frame0_record = materialize_candidate_frame0(
        surface=surface,
        modules=modules,
        codes=solved_codes,
        output=output,
    )
    retention_inventory = retention.finalize() if retention is not None else None
    result = {
        "schema": "ddm_jo2_fresh_schur_n600.v1",
        "status": "COMPLETE",
        "pair_denominator": N,
        "semantic_object_sha256": semantic_object_sha256,
        "candidate_master": dict(candidate_master),
        "base_pose6": dict(base_pose6),
        "candidate_codes": codes_record,
        "candidate_frame0": frame0_record,
        "changed_pairs": int(np.count_nonzero(np.any(solved_codes != surface.codes, axis=1))),
        "changed_coordinates": int(np.count_nonzero(solved_codes != surface.codes)),
        "fresh_per_candidate_asserted_in_code": True,
        "retention_inventory": retention_inventory,
        "pair_results": rows,
        "score_claim": False,
    }
    atomic_json(output / "FRESH_SCHUR_RESULT.json", result)
    return result


def _pack_cap1_metadata(canonical: bytes, modules: RuntimeModules) -> bytes:
    """Exact inverse of the fx5 parser's packed CAP1 metadata restoration."""
    # Delegate to the already-reviewed compiler helper and then independently
    # require the current fx5 parser to restore it byte-for-byte.
    from experiments.ddm_sa2_compile_candidate import pack_cap1_metadata

    packed = pack_cap1_metadata(canonical)
    restored = modules.residual_archive._restore_packed_cap1_metadata(packed)
    if restored != canonical:
        raise JO2ReceiverCloseError("packed CAP1 metadata round-trip differs")
    return packed


def encode_carrier_body(
    codes: np.ndarray,
    surface: CarrierSurface,
    modules: RuntimeModules,
) -> bytes:
    cpr1 = encode_compact_carrier(
        surface.basis_scales,
        surface.basis_codes,
        surface.coefficient_scales,
        delta_zigzag_from_signed_codes(codes),
    )
    cap1 = _encode_cap1(cpr1, np.asarray(codes, dtype=np.int32), modules)
    if modules.coefficient_codec.decode_cap1(cap1, frames=N, dimensions=D) != cpr1:
        raise JO2ReceiverCloseError("CAP1 round-trip differs")
    if not cap1.startswith(modules.residual_archive.CAP1_PREFIX):
        raise JO2ReceiverCloseError("CAP1 prefix differs")
    stripped = cap1[len(modules.residual_archive.CAP1_PREFIX) :]
    body_bytes = modules.residual_archive._cap1_body_bytes(stripped)
    bit_counts, predictor = stripped[:6], stripped[6:42]
    scales, lengths = stripped[42:138], stripped[138:170]
    ks, rest = stripped[170:182], stripped[182:body_bytes]
    canonical = bit_counts + scales + predictor + lengths + ks + rest + surface.selector[5:]
    return _pack_cap1_metadata(canonical, modules)


def _rice_bit_cost(values: np.ndarray) -> tuple[int, int]:
    encoded = np.asarray(values, dtype=np.uint64)
    return min(
        (int((encoded >> candidate).sum()) + encoded.size * (candidate + 1), candidate)
        for candidate in range(12)
    )


def _fit_ar1_dimension(column: np.ndarray, predictor: Any) -> tuple[int, int]:
    previous = np.asarray(column[:-1], dtype=np.int64)
    target = np.asarray(column[1:], dtype=np.int64)
    denominator = int(np.square(previous).sum())
    numerator = 256 * int((previous * target).sum())
    estimate = (
        0
        if denominator == 0
        else (1 if numerator >= 0 else -1)
        * ((abs(numerator) + denominator // 2) // denominator)
    )
    centre = min(512, max(-512, estimate))
    best: tuple[int, int, int, int] | None = None
    for factor in range(max(-512, centre - 96), min(512, centre + 96) + 1):
        baseline = predictor.round_q8(
            previous, np.full(previous.shape, factor, dtype=np.int16)
        ).astype(np.int64)
        central_bias = min(16, max(-16, int(np.median(target - baseline))))
        for bias in range(max(-16, central_bias - 4), min(16, central_bias + 4) + 1):
            residual = predictor.signed_mod(target - predictor.signed_mod(baseline + bias))
            unsigned = ((residual.astype(np.int64) << 1) ^ (residual.astype(np.int64) >> 63)) & 0xFFF
            bits, _ = _rice_bit_cost(unsigned)
            candidate = (bits, abs(factor - 256), factor, bias)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise JO2ReceiverCloseError("AR1 fit produced no candidate")
    return best[2], best[3]


def _encode_cap1(
    cpr1: bytes,
    codes: np.ndarray,
    modules: RuntimeModules,
) -> bytes:
    """Port the shipped exact CAP1 encoder without an external book dependency."""
    if len(cpr1) < 152:
        raise JO2ReceiverCloseError("canonical CPR1 is truncated")
    magic, basis_bits, _coefficient_bits = struct.unpack_from("<4sII", cpr1)
    if magic != b"CPR1" or basis_bits <= 0:
        raise JO2ReceiverCloseError("canonical CPR1 header differs")
    basis_bytes = (basis_bits + 7) // 8
    fixed_end = 12 + 8 * D + 32 + D
    if fixed_end + basis_bytes >= len(cpr1):
        raise JO2ReceiverCloseError("canonical CPR1 field lengths differ")
    factors_biases = [
        _fit_ar1_dimension(codes[:, dimension], modules.coefficient_predictor)
        for dimension in range(D)
    ]
    model = modules.coefficient_predictor.Ar1BiasModel(
        np.asarray([value[0] for value in factors_biases], dtype=np.int16),
        np.asarray([value[1] for value in factors_biases], dtype=np.int8),
    )
    residuals = np.empty_like(codes, dtype=np.int32)
    residuals[0] = codes[0]
    for frame in range(1, N):
        prediction = modules.coefficient_predictor.signed_mod(
            modules.coefficient_predictor.round_q8(codes[frame - 1], model.factors_q8)
            + model.biases
        )
        residuals[frame] = modules.coefficient_predictor.signed_mod(
            codes[frame] - prediction
        )
    zigzag = ((residuals.astype(np.int64) << 1) ^ (residuals.astype(np.int64) >> 63)) & 0xFFF
    ks, rice_payload, residual_bits = modules.carrier_repack._rice_encode(zigzag, 1)
    metadata = (
        np.asarray(model.factors_q8, dtype="<i2").tobytes()
        + np.asarray(model.biases, dtype="i1").tobytes()
    )
    cap1 = (
        b"CAP1"
        + bytes((1, 0, 0, 0))
        + int(basis_bits).to_bytes(3, "little")
        + int(residual_bits).to_bytes(3, "little")
        + metadata
        + cpr1[12 : 12 + 8 * D]
        + cpr1[12 + 8 * D : 12 + 8 * D + 32]
        + ks.reshape(-1).tobytes()
        + cpr1[fixed_end : fixed_end + basis_bytes]
        + rice_payload
    )
    if modules.coefficient_codec.decode_cap1(cap1, frames=N, dimensions=D) != cpr1:
        raise JO2ReceiverCloseError("ported CAP1 encoder did not restore CPR1")
    return cap1


def _replace_once(path: Path, old: str, new: str, label: str) -> None:
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise JO2ReceiverCloseError(f"runtime patch point differs: {label}")
    atomic_bytes(path, source.replace(old, new).encode(), executable=os.access(path, os.X_OK))


def stage_runtime(runtime_root: Path, archive_payload: bytes) -> dict[str, Any]:
    if runtime_root.exists():
        raise JO2ReceiverCloseError(f"candidate runtime already exists: {runtime_root}")
    shutil.copytree(
        FX5_RUNTIME,
        runtime_root,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*", "archive.zip"),
    )
    runtime_source = Path(residual_runtime.__file__).resolve().read_bytes()
    runtime_record = atomic_bytes(
        runtime_root / "runtime/jo2_residual_runtime.py", runtime_source
    )
    _replace_once(
        runtime_root / "runtime/residual_archive.py",
        '    tagged_semantic = semantic_body.startswith((b"SD1M", b"SM3R"))\n',
        '    tagged_semantic = semantic_body.startswith((b"SD1M", b"SM3R", b"J2R1"))\n',
        "RX1 JO2 semantic tag",
    )
    f26 = runtime_root / "runtime/f26_inflate.py"
    _replace_once(
        f26,
        "from .compensation_overlay import apply_compensation_overlay\n",
        "from .compensation_overlay import apply_compensation_overlay\n"
        "from .jo2_residual_runtime import ResidualWrappedRenderer, split_semantic_blob\n",
        "JO2 runtime import",
    )
    _replace_once(
        f26,
        '    if not parts.semantic_blob.startswith((WANS1_MAGIC, b"SD1M", b"SM3R")):\n'
        '        raise InflationError("F26 requires WANS1, SD1M, or SM3R semantic weights")\n',
        "    base_semantic_blob, jo2_residual_payload = split_semantic_blob(parts.semantic_blob)\n"
        '    if not base_semantic_blob.startswith((WANS1_MAGIC, b"SD1M", b"SM3R")):\n'
        '        raise InflationError("F26 requires WANS1, SD1M, or SM3R semantic weights")\n',
        "JO2 semantic unwrap before F26 validation",
    )
    _replace_once(
        f26,
        "    semantic = renderer.SemanticTokenRenderer(96)\n"
        "    tagged_state = renderer.unpack_variant_semantic_or_none(\n"
        "        parts.semantic_blob,\n",
        "    semantic = renderer.SemanticTokenRenderer(96)\n"
        "    tagged_state = renderer.unpack_variant_semantic_or_none(\n"
        "        base_semantic_blob,\n",
        "JO2 base semantic load",
    )
    _replace_once(
        f26,
        "        records = decode_wans1(parts.semantic_blob)\n",
        "        records = decode_wans1(base_semantic_blob)\n",
        "JO2 base WANS decode",
    )
    _replace_once(
        f26,
        "    semantic.load_state_dict(tagged_state, strict=True)\n"
        "    setup_seconds = time.perf_counter() - setup_started\n",
        "    semantic.load_state_dict(tagged_state, strict=True)\n"
        "    if jo2_residual_payload is not None:\n"
        "        semantic = ResidualWrappedRenderer(semantic, jo2_residual_payload)\n"
        "    setup_seconds = time.perf_counter() - setup_started\n",
        "JO2 wrapper construction",
    )
    archive_record = atomic_bytes(runtime_root / "archive.zip", archive_payload)
    inflate = runtime_root / "inflate.py"
    source = inflate.read_text(encoding="utf-8")
    if source.count(FX5_ARCHIVE_SHA256) != 1 or source.count("ARCHIVE_BYTES = 180_386") != 1:
        raise JO2ReceiverCloseError("outer inflate pin surface differs")
    source = source.replace(FX5_ARCHIVE_SHA256, archive_record["sha256"], 1)
    source = source.replace(
        "ARCHIVE_BYTES = 180_386",
        f"ARCHIVE_BYTES = {archive_record['bytes']:_}",
        1,
    )
    inflate_record = atomic_bytes(inflate, source.encode(), executable=True)
    return {
        "archive": archive_record,
        "inflate": inflate_record,
        "jo2_runtime": runtime_record,
        "residual_archive": file_record(runtime_root / "runtime/residual_archive.py"),
        "f26_inflate": file_record(f26),
    }


def validate_staged_receiver(runtime_root: Path, output: Path) -> dict[str, Any]:
    """Decode through the exact staged shipping tree before receiver closure."""
    root = output.resolve()
    if root.exists():
        raise JO2ReceiverCloseError(f"receiver validation already exists: {root}")
    archive_path = runtime_root.resolve() / "archive.zip"
    archive_record = file_record(archive_path)
    archive_dir = root / "archive"
    decoded_dir = root / "output"
    raw_path = decoded_dir / "0.raw"
    names = atomic_bytes(root / "video_names.txt", b"0.mp4\n")
    with zipfile.ZipFile(archive_path) as archive:
        infos = archive.infolist()
        if (
            len(infos) != 1
            or infos[0].filename != "p"
            or infos[0].compress_type != zipfile.ZIP_STORED
        ):
            raise JO2ReceiverCloseError(
                "staged receiver validation requires one stored member p"
            )
        member = archive.read(infos[0])
        if archive.testzip() is not None:
            raise JO2ReceiverCloseError("staged receiver validation archive CRC failed")
    extracted = atomic_bytes(archive_dir / "p", member)
    inflate_sh = runtime_root.resolve() / "inflate.sh"
    if not inflate_sh.is_file() or not os.access(inflate_sh, os.X_OK):
        raise JO2ReceiverCloseError("staged shipping inflate.sh is absent or not executable")
    command = [
        str(inflate_sh),
        str(archive_dir),
        str(decoded_dir),
        str(verify_record(names)),
    ]
    path = os.environ.get("PATH", "")
    host_shims = REPO / "tools/host_shims"
    environment = {
        **os.environ,
        "PATH": f"{host_shims}{os.pathsep}{path}",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    started = time.monotonic()
    try:
        # inflate.sh spawns a python worker; a plain subprocess timeout kills
        # only the shell and orphans the grandchild (Catalog #408) — the
        # group-kill runner escalates SIGTERM/SIGKILL against the whole tree.
        process = run_in_process_group(
            command,
            cwd=runtime_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except subprocess.TimeoutExpired as error:
        stdout = (
            error.stdout.decode(errors="replace")
            if isinstance(error.stdout, bytes)
            else (error.stdout or "")
        )
        stderr = (
            error.stderr.decode(errors="replace")
            if isinstance(error.stderr, bytes)
            else (error.stderr or "")
        )
        log = atomic_bytes(
            root / "inflate.log",
            (stdout + "\n--- STDERR ---\n" + stderr).encode(),
        )
        raise JO2ReceiverCloseError(
            f"staged shipped receiver timed out; log={log['path']}"
        ) from error
    wall_seconds = time.monotonic() - started
    log = atomic_bytes(
        root / "inflate.log",
        (process.stdout + "\n--- STDERR ---\n" + process.stderr).encode(),
    )
    if (
        process.returncode != 0
        or not raw_path.is_file()
        or raw_path.stat().st_size != RAW_BYTES
    ):
        observed = raw_path.stat().st_size if raw_path.is_file() else None
        raise JO2ReceiverCloseError(
            "staged shipped receiver failed before closure: "
            f"rc={process.returncode},raw_bytes={observed},expected={RAW_BYTES},log={log['path']}"
        )
    raw = file_record(raw_path)
    receipt = {
        "schema": "ddm_jo2_staged_receiver_validation.v1",
        "status": "COMPLETE",
        "command": command,
        "returncode": process.returncode,
        "wall_seconds": wall_seconds,
        "archive": archive_record,
        "extracted_member": extracted,
        "raw": raw,
        "expected_raw_bytes": RAW_BYTES,
        "log": log,
        "axis": "[macOS-CPU real staged shipped receiver; no score authority]",
        "all_materialized_payloads_retained": True,
        "score_claim": False,
    }
    receipt_record = atomic_json(root / "RECEIVER_EXECUTION.json", receipt)
    return {**receipt, "receipt": receipt_record}


_PARSEBACK = r'''
import hashlib, json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
root, archive, expected_codes, expected_semantic = (
    Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4])
sys.path.insert(0, str(root)); sys.path.insert(0, str(root / "cpr1"))
from runtime.carrier_repack import materialize_cpr1
from runtime.jo2_residual_runtime import split_semantic_blob, decode_residual_state
from runtime.residual_archive import read_residual_archive
import carrier_codec
parts = read_residual_archive(archive)
base, payload = split_semantic_blob(parts.semantic_blob)
if payload is None: raise RuntimeError("JO2 residual payload absent")
state, hidden, max_delta = decode_residual_state(payload)
canonical = materialize_cpr1(parts.carrier_blob, SimpleNamespace(N=600, CARRIER_DIM=12))
_, _, _, encoded = carrier_codec.decode_compact_carrier(
    canonical, basis_count=12*3*24*32, frames=600, dimensions=12)
delta = (encoded.astype(np.int64) >> 1) ^ -(encoded.astype(np.int64) & 1)
codes = np.cumsum(delta, axis=0) & 0xfff
codes = np.where(codes >= 0x800, codes - 0x1000, codes).astype(np.int32)
expected = np.load(expected_codes, allow_pickle=False)
print(json.dumps({
    "codes_match": bool(np.array_equal(codes, expected)),
    "semantic_sha256": hashlib.sha256(parts.semantic_blob).hexdigest(),
    "expected_semantic_sha256": expected_semantic,
    "residual_payload_sha256": hashlib.sha256(payload).hexdigest(),
    "hidden_channels": hidden,
    "max_rgb_delta": max_delta,
    "state_tensor_count": len(state),
}))
'''


def compile_receiver_closed_stage(
    *,
    residual_payload: Mapping[str, Any],
    solve_result: Mapping[str, Any],
    output: Path,
    candidate_master: Mapping[str, Any],
    archive: Path = FX5_ARCHIVE,
    runtime_root: Path = FX5_RUNTIME,
) -> dict[str, Any]:
    """Build and parse back a fresh-solved, counted, single-p JO2 stage."""
    payload_path = verify_record(residual_payload)
    payload = payload_path.read_bytes()
    residual_runtime.decode_residual_state(payload)
    surface, modules = load_surface(archive, runtime_root)
    semantic_body = residual_runtime.pack_semantic_blob(surface.parts.semantic_blob, payload)
    semantic_sha256 = hashlib.sha256(semantic_body).hexdigest()
    if solve_result.get("status") != "COMPLETE" or int(solve_result.get("pair_denominator", 0)) != N:
        raise JO2ReceiverCloseError("fresh Schur solve is incomplete")
    if solve_result.get("semantic_object_sha256") != semantic_sha256:
        raise JO2ReceiverCloseError("fresh Schur solve is bound to another semantic object")
    if dict(solve_result.get("candidate_master", {})) != dict(candidate_master):
        raise JO2ReceiverCloseError("fresh Schur solve is bound to another camera field")
    verify_record(candidate_master)
    base_pose6 = solve_result.get("base_pose6")
    if not isinstance(base_pose6, Mapping):
        raise JO2ReceiverCloseError("fresh Schur solve base Pose6 binding is absent")
    verify_record(base_pose6)
    pair_rows = solve_result.get("pair_results", [])
    if len(pair_rows) != N or [int(row["pair"]) for row in pair_rows] != list(range(N)):
        raise JO2ReceiverCloseError("fresh Schur pair census differs")
    for row in pair_rows:
        expected = candidate_object_fingerprint(
            pair=int(row["pair"]),
            semantic_object_sha256=semantic_sha256,
            candidate_master=candidate_master,
            base_pose6=base_pose6,
        )
        if row.get("candidate_object_fingerprint_sha256") != expected:
            raise JO2ReceiverCloseError("stale per-pair compensation reached compile")
    codes_path = verify_record(solve_result["candidate_codes"])
    frame0_path = verify_record(solve_result["candidate_frame0"])
    codes = np.load(codes_path, allow_pickle=False)
    frame0 = np.load(frame0_path, mmap_mode="r", allow_pickle=False)
    if codes.shape != (N, D) or codes.dtype != np.int32:
        raise JO2ReceiverCloseError("fresh solved carrier lattice differs")
    if frame0.shape != (N, CAMERA_H, CAMERA_W, 3) or frame0.dtype != np.uint8:
        raise JO2ReceiverCloseError("fresh solved frame-0 field differs")
    carrier_body = encode_carrier_body(codes, surface, modules)
    retained = {
        "residual_payload": atomic_bytes(output / "retained/residual.j2s1", payload),
        "semantic_body": atomic_bytes(output / "retained/semantic.j2r1", semantic_body),
        "carrier_body": atomic_bytes(output / "retained/carrier.raw", carrier_body),
        "candidate_codes": atomic_npy(output / "retained/candidate_codes.int32.npy", codes),
        "candidate_frame0": dict(solve_result["candidate_frame0"]),
    }
    rows = []
    best: tuple[int, int, int, bytes, bytes, bytes] | None = None
    for semantic_quality in range(12):
        semantic_stream = brotli.compress(semantic_body, quality=semantic_quality, lgwin=24)
        if brotli.decompress(semantic_stream) != semantic_body:
            raise JO2ReceiverCloseError("semantic Brotli round-trip differs")
        for carrier_quality in range(12):
            carrier_stream = brotli.compress(carrier_body, quality=carrier_quality, lgwin=24)
            if brotli.decompress(carrier_stream) != carrier_body:
                raise JO2ReceiverCloseError("carrier Brotli round-trip differs")
            # Both rebuilt streams are canonical bodies, so the source's
            # semantic/carrier permutation flags must be cleared.  Retaining
            # them would apply an inverse transform to bytes that were never
            # transformed.
            member = (
                rx1.RX1_HEADER.pack(
                    rx1.RX1_MAGIC,
                    rx1.RX1_VERSION,
                    int(surface.outer["codec"]),
                    int(surface.outer["table_mode"]),
                    0,
                    len(surface.outer["hpac_stream"]),
                    len(semantic_stream),
                    len(carrier_stream),
                )
                + surface.outer["hpac_stream"]
                + semantic_stream
                + carrier_stream
                + surface.outer["tail"]
            )
            archive_payload = rx1.deterministic_zip(member)
            root = output / "retained/rate_race" / f"sq{semantic_quality:02d}_cq{carrier_quality:02d}"
            records = {
                "semantic_stream": atomic_bytes(root / "semantic.br", semantic_stream),
                "carrier_stream": atomic_bytes(root / "carrier.br", carrier_stream),
                "member_p": atomic_bytes(root / "p", member),
                "archive": atomic_bytes(root / "archive.zip", archive_payload),
            }
            row = {
                "semantic_quality": semantic_quality,
                "carrier_quality": carrier_quality,
                "archive_bytes": len(archive_payload),
                "payloads": records,
            }
            atomic_json(root / "RESULT.json", row)
            rows.append(row)
            key = (len(archive_payload), semantic_quality, carrier_quality)
            if best is None or key < best[:3]:
                best = (*key, archive_payload, member, semantic_stream)
    if best is None:
        raise JO2ReceiverCloseError("real-coder race produced no candidate")
    _, semantic_quality, carrier_quality, archive_payload, member, _ = best
    primary = atomic_bytes(output / "archive.zip", archive_payload)
    repeat_payload = rx1.deterministic_zip(member)
    repeat = atomic_bytes(output / "archive.repeat.zip", repeat_payload)
    if primary["sha256"] != repeat["sha256"]:
        raise JO2ReceiverCloseError("candidate archive repeat differs")
    generation = output / "submission"
    runtime_records = stage_runtime(generation, archive_payload)
    parse = subprocess.run(
        [
            sys.executable,
            "-c",
            _PARSEBACK,
            str(generation),
            str(generation / "archive.zip"),
            str(codes_path),
            semantic_sha256,
        ],
        check=False,
        capture_output=True,
        text=True,
        env={"PYTHONDONTWRITEBYTECODE": "1", "PATH": "/usr/bin:/bin"},
    )
    if parse.returncode != 0:
        raise JO2ReceiverCloseError(f"fresh receiver parse-back failed: {parse.stderr[-800:]}")
    parseback = json.loads(parse.stdout.strip().splitlines()[-1])
    if not parseback["codes_match"] or parseback["semantic_sha256"] != semantic_sha256:
        raise JO2ReceiverCloseError("fresh receiver parse-back differs")
    parseback_record = atomic_json(output / "RECEIVER_PARSEBACK.json", parseback)
    receiver_validation = validate_staged_receiver(
        generation, output / "receiver_validation"
    )
    result = {
        "schema": "ddm_jo2_receiver_closed_stage.v1",
        "status": "COMPLETE",
        "axis": AXIS,
        "archive": primary,
        "archive_repeat": repeat,
        "archive_repeat_byte_identical": True,
        "single_p": True,
        "selected_real_coder": {
            "semantic_quality": semantic_quality,
            "carrier_quality": carrier_quality,
            "candidate_denominator": len(rows),
            "selection_mode": "minimum exact archive bytes; lower qualities break ties",
        },
        "semantic_object_sha256": semantic_sha256,
        "fresh_schur_pair_denominator": N,
        "fresh_same_object_compensation": True,
        "receiver_parseback": parseback_record,
        "receiver_parseback_identity": True,
        "staged_receiver_validation": receiver_validation,
        "staged_receiver_decoded_expected_raw_bytes": True,
        "runtime": runtime_records,
        "retained_payloads": retained,
        "candidate_master": dict(candidate_master),
        "candidate_frame0": dict(solve_result["candidate_frame0"]),
        "all_materialized_payloads_retained": True,
        "score_claim": False,
        "frontier_moved": False,
    }
    atomic_json(output / "RECEIVER_CLOSE_RESULT.json", result)
    return result


def verify_decoded_render_identity(
    *,
    receiver_close: Mapping[str, Any],
    candidate_frame0: Mapping[str, Any],
    candidate_master: Mapping[str, Any],
    decoded_raw: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    """Close the last boundary after the shipped runtime has materialized raw."""
    verify_record(receiver_close["archive"])
    frame0_path = verify_record(candidate_frame0)
    master_path = verify_record(candidate_master)
    raw_path = verify_record(decoded_raw)
    frame0 = np.load(frame0_path, mmap_mode="r", allow_pickle=False)
    masters = np.load(master_path, mmap_mode="r", allow_pickle=False)
    if (
        frame0.shape != (N, CAMERA_H, CAMERA_W, 3)
        or frame0.dtype != np.uint8
        or masters.shape != frame0.shape
        or masters.dtype != np.uint8
    ):
        raise JO2ReceiverCloseError("intended decoded camera fields differ")
    raw = np.memmap(
        raw_path,
        mode="r",
        dtype=np.uint8,
        shape=(N * 2, CAMERA_H, CAMERA_W, 3),
    )
    frame0_mismatch = 0
    master_mismatch = 0
    for pair in range(N):
        frame0_mismatch += int(np.count_nonzero(raw[2 * pair] != frame0[pair]))
        master_mismatch += int(np.count_nonzero(raw[2 * pair + 1] != masters[pair]))
    result = {
        "schema": "ddm_jo2_decoded_render_identity.v1",
        "pair_denominator": N,
        "camera_element_denominator_per_parity": N * CAMERA_H * CAMERA_W * 3,
        "frame0_mismatch_elements": frame0_mismatch,
        "master_mismatch_elements": master_mismatch,
        "identity": frame0_mismatch == 0 and master_mismatch == 0,
        "candidate_frame0": dict(candidate_frame0),
        "candidate_master": dict(candidate_master),
        "decoded_raw": dict(decoded_raw),
        "archive": dict(receiver_close["archive"]),
        "axis": AXIS,
        "score_claim": False,
    }
    atomic_json(output / "DECODED_RENDER_IDENTITY.json", result)
    if frame0_mismatch or master_mismatch:
        raise JO2ReceiverCloseError("shipped receiver render differs from solved object")
    return result
