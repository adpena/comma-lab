#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Local constructive constrained-MDL inverse-solve over scorer-grid RGB deltas.

This is a research-only, local CPU instrument.  It renders the selected banked
v9c2 EMA pairs with the canonical NumPy oracle, lifts quantized scorer-grid
deltas through the minimum-norm right inverse of the evaluator resize, and
performs a deterministic satisficing-hinge descent through the frozen SegNet.
It neither trains scorer/model weights nor proves a global MDL optimum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import struct
import subprocess
import sys
import tempfile
import time
import zipfile
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.canonical_equations.margin_band_satisficing_threshold_20260712 import (  # noqa: E402
    resolve_margin_band_threshold,
)
from tac.subset_selection import quantile_stratified_indices  # noqa: E402

SCHEMA = "constructive_inverse_solve.v1"
STATE_SCHEMA = "constructive_inverse_solve.state.v1"
PAYLOAD_MAGIC = b"CIS1"
AXIS = "[macOS-CPU advisory]"
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
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)


class HarnessError(RuntimeError):
    """Fail-closed input, custody, algebra, or resume error."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode())
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_state(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _refuse_tmp(path: Path, field: str) -> None:
    resolved = path.expanduser().resolve()
    bad_roots = {Path("/tmp"), Path("/private/tmp"), Path("/var/tmp"), Path(tempfile.gettempdir()).resolve()}
    for root in bad_roots:
        try:
            resolved.relative_to(root)
        except ValueError:
            continue
        raise HarnessError(f"{field} must be durable, not under temporary root {root}: {resolved}")


def stored_npy_memmap(npz_path: Path, key: str) -> np.memmap:
    """Map one unencrypted ZIP_STORED NPY member without loading the full cache."""

    member = key if key.endswith(".npy") else f"{key}.npy"
    with zipfile.ZipFile(npz_path) as archive:
        try:
            info = archive.getinfo(member)
        except KeyError as exc:
            raise HarnessError(f"cache lacks required member {member!r}") from exc
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1:
            raise HarnessError(f"{npz_path}:{member} must be unencrypted ZIP_STORED")
        header_offset = int(info.header_offset)
    with npz_path.open("rb") as handle:
        handle.seek(header_offset)
        header = handle.read(30)
        if len(header) != 30:
            raise HarnessError(f"truncated local header for {member}")
        fields = struct.unpack("<IHHHHHIIIHH", header)
        if fields[0] != 0x04034B50:
            raise HarnessError(f"bad local ZIP signature for {member}")
        handle.seek(header_offset + 30 + int(fields[-2]) + int(fields[-1]))
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version in {(2, 0), (3, 0)}:
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise HarnessError(f"unsupported NPY version {version} for {member}")
        data_offset = handle.tell()
    return np.memmap(
        npz_path,
        mode="r",
        dtype=dtype,
        shape=shape,
        offset=data_offset,
        order="F" if fortran else "C",
    )


@dataclass(frozen=True)
class SparseAxis:
    n_in: int
    n_out: int
    indices: np.ndarray  # (n_out, 2), repeated index for one-tap rows
    weights: np.ndarray  # A row weights
    inverse_weights: np.ndarray  # A^T (AA^T)^-1 row weights
    taps_per_row: np.ndarray
    offdiag_max: float
    blind_count: int


def _sparse_axis(matrix: np.ndarray, *, tol: float = 1e-14) -> SparseAxis:
    a = np.asarray(matrix, dtype=np.float64)
    if not np.isfinite(a).all():
        raise HarnessError("resize matrix contains non-finite values")
    with np.errstate(all="ignore"):
        gram = a @ a.T
    if not np.isfinite(gram).all():
        raise HarnessError("resize row Gram contains non-finite values")
    off = gram - np.diag(np.diag(gram))
    if not np.isfinite(off).all():
        raise HarnessError("resize off-diagonal Gram contains non-finite values")
    offdiag_max = float(np.max(np.abs(off))) if off.size else 0.0
    if offdiag_max > 1e-12:
        raise HarnessError(f"resize row supports overlap: max off-diagonal Gram={offdiag_max:.3e}")
    idx = np.zeros((a.shape[0], 2), dtype=np.int64)
    weights = np.zeros((a.shape[0], 2), dtype=np.float64)
    taps = np.zeros(a.shape[0], dtype=np.int64)
    used = np.zeros(a.shape[1], dtype=bool)
    for row in range(a.shape[0]):
        nz = np.flatnonzero(np.abs(a[row]) > tol)
        if nz.size not in (1, 2):
            raise HarnessError(f"resize row {row} has {nz.size} taps, expected one or two")
        taps[row] = nz.size
        used[nz] = True
        idx[row, : nz.size] = nz
        weights[row, : nz.size] = a[row, nz]
        if nz.size == 1:
            idx[row, 1] = nz[0]
    norm2 = np.sum(weights * weights, axis=1)
    if not np.isfinite(norm2).all() or np.any(norm2 <= 0):
        raise HarnessError("resize has a non-finite or zero-norm output row")
    inverse = weights / norm2[:, None]
    return SparseAxis(
        n_in=int(a.shape[1]),
        n_out=int(a.shape[0]),
        indices=idx,
        weights=weights,
        inverse_weights=inverse,
        taps_per_row=taps,
        offdiag_max=offdiag_max,
        blind_count=int((~used).sum()),
    )


def _apply_a_numpy(x: np.ndarray, ah: SparseAxis, aw: SparseAxis) -> np.ndarray:
    """Apply A to NCHW camera planes using only the certified one/two-tap rows."""

    value = np.asarray(x, dtype=np.float64)
    if value.shape[-2:] != (ah.n_in, aw.n_in):
        raise HarnessError(f"A input shape {value.shape[-2:]} is not {(ah.n_in, aw.n_in)}")
    tmp = np.zeros((*value.shape[:-1], aw.n_out), dtype=np.float64)
    for tap in range(2):
        tmp += value[..., aw.indices[:, tap]] * aw.weights[:, tap]
    out = np.zeros((*tmp.shape[:-2], ah.n_out, aw.n_out), dtype=np.float64)
    for tap in range(2):
        out += tmp[..., ah.indices[:, tap], :] * ah.weights[:, tap, None]
    return out


def _lift_b_numpy(y: np.ndarray, ah: SparseAxis, aw: SparseAxis) -> np.ndarray:
    value = np.asarray(y, dtype=np.float64)
    if value.shape[-2:] != (ah.n_out, aw.n_out):
        raise HarnessError(f"B input shape {value.shape[-2:]} is not {(ah.n_out, aw.n_out)}")
    tmp = np.zeros((*value.shape[:-1], aw.n_in), dtype=np.float64)
    for tap in range(2):
        np.add.at(tmp, (..., aw.indices[:, tap]), value * aw.inverse_weights[:, tap])
    out = np.zeros((*tmp.shape[:-2], ah.n_in, aw.n_in), dtype=np.float64)
    for tap in range(2):
        np.add.at(out, (..., ah.indices[:, tap], slice(None)), tmp * ah.inverse_weights[:, tap, None])
    return out


def _lift_b_torch(y: Any, ah: SparseAxis, aw: SparseAxis) -> Any:
    import torch

    n, c, sh, sw = y.shape
    if (sh, sw) != (ah.n_out, aw.n_out):
        raise HarnessError("torch B input has wrong scorer geometry")
    tmp = y.new_zeros((n, c, sh, aw.n_in))
    for tap in range(2):
        idx = torch.as_tensor(aw.indices[:, tap], dtype=torch.long, device=y.device)
        coef = torch.as_tensor(aw.inverse_weights[:, tap], dtype=y.dtype, device=y.device)
        tmp.scatter_add_(3, idx.view(1, 1, 1, sw).expand(n, c, sh, sw), y * coef.view(1, 1, 1, sw))
    out = y.new_zeros((n, c, ah.n_in, aw.n_in))
    for tap in range(2):
        idx = torch.as_tensor(ah.indices[:, tap], dtype=torch.long, device=y.device)
        coef = torch.as_tensor(ah.inverse_weights[:, tap], dtype=y.dtype, device=y.device)
        out.scatter_add_(2, idx.view(1, 1, sh, 1).expand(n, c, sh, aw.n_in), tmp * coef.view(1, 1, sh, 1))
    return out


def _resize_algebra(camera_hw: tuple[int, int], scorer_hw: tuple[int, int]) -> tuple[SparseAxis, SparseAxis, dict[str, Any]]:
    from tac.optimization.resize_null_preimage import ResizeProjector

    projector = ResizeProjector.build(
        camera_h=camera_hw[0], camera_w=camera_hw[1], scorer_h=scorer_hw[0], scorer_w=scorer_hw[1]
    )
    ah, aw = _sparse_axis(projector.rh), _sparse_axis(projector.rw)
    camera_dim = camera_hw[0] * camera_hw[1]
    rank = scorer_hw[0] * scorer_hw[1]
    used_h = camera_hw[0] - ah.blind_count
    used_w = camera_hw[1] - aw.blind_count
    axis_blind = camera_dim - used_h * used_w
    return ah, aw, {
        "camera_hw": list(camera_hw),
        "scorer_hw": list(scorer_hw),
        "rank_per_channel": rank,
        "null_dimension_per_channel": camera_dim - rank,
        "full_kernel_fraction": (camera_dim - rank) / camera_dim,
        "height_blind_coordinates": ah.blind_count,
        "width_blind_coordinates": aw.blind_count,
        "axis_blind_pixels": axis_blind,
        "axis_blind_fraction": axis_blind / camera_dim,
        "row_gram_offdiag_max_h": ah.offdiag_max,
        "row_gram_offdiag_max_w": aw.offdiag_max,
        "taps_per_row_h": {str(k): int((ah.taps_per_row == k).sum()) for k in (1, 2)},
        "taps_per_row_w": {str(k): int((aw.taps_per_row == k).sum()) for k in (1, 2)},
        "restriction": "delta is a scorer-grid payload; its float camera lift is in range(A^T); no ker(A) camera DOF is optimized",
    }


def _serialize_payload(q_delta: np.ndarray, poses: np.ndarray, pair_ids: Sequence[int], qstep: float) -> bytes:
    q = np.asarray(q_delta, dtype=np.int8)
    pose = np.asarray(poses, dtype="<f4")
    header = _canonical_json(
        {
            "schema": SCHEMA + ".payload",
            "selected_pair_ids": [int(i) for i in pair_ids],
            "tensor_shape": list(q.shape),
            "dtype": "int8",
            "quantization_step": float(qstep),
            "pose_dtype": "<f4",
            "pose_shape": list(pose.shape),
        }
    )
    delta_stream = zlib.compress(q.astype("i1", copy=False).tobytes(order="C"), level=9)
    pose_bytes = pose.tobytes(order="C")
    return b"".join(
        (
            PAYLOAD_MAGIC,
            struct.pack("<I", len(header)),
            header,
            struct.pack("<I", len(delta_stream)),
            delta_stream,
            struct.pack("<I", len(pose_bytes)),
            pose_bytes,
        )
    )


def _parse_payload(blob: bytes) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    if blob[:4] != PAYLOAD_MAGIC:
        raise HarnessError("bad constructive payload magic")
    off = 4
    hlen = struct.unpack_from("<I", blob, off)[0]
    off += 4
    if off + hlen > len(blob):
        raise HarnessError("truncated constructive payload header")
    header_raw = blob[off : off + hlen]
    header = json.loads(header_raw)
    if _canonical_json(header) != header_raw:
        raise HarnessError("constructive payload header is not canonical JSON")
    off += hlen
    required = {
        "schema",
        "selected_pair_ids",
        "tensor_shape",
        "dtype",
        "quantization_step",
        "pose_dtype",
        "pose_shape",
    }
    if set(header) != required or header["schema"] != SCHEMA + ".payload":
        raise HarnessError("constructive payload header schema/field set mismatch")
    if header["dtype"] != "int8" or header["pose_dtype"] != "<f4":
        raise HarnessError("constructive payload dtype contract mismatch")
    shape = tuple(int(v) for v in header["tensor_shape"])
    pose_shape = tuple(int(v) for v in header["pose_shape"])
    pair_ids = header["selected_pair_ids"]
    if len(shape) != 4 or shape[1] != 3 or any(v < 0 for v in shape):
        raise HarnessError(f"invalid scorer-delta shape in payload: {shape}")
    if pose_shape != (shape[0], 6) or len(pair_ids) != shape[0]:
        raise HarnessError("payload pair mapping/tensor/pose shapes disagree")
    if len(set(pair_ids)) != len(pair_ids) or any(not isinstance(v, int) or v < 0 for v in pair_ids):
        raise HarnessError("payload selected pair IDs are invalid")
    qstep = float(header["quantization_step"])
    if not math.isfinite(qstep) or qstep <= 0:
        raise HarnessError("payload quantization step must be finite and positive")
    if off + 4 > len(blob):
        raise HarnessError("truncated constructive delta length prefix")
    dlen = struct.unpack_from("<I", blob, off)[0]
    off += 4
    if off + dlen > len(blob):
        raise HarnessError("truncated constructive delta zlib stream")
    decompressor = zlib.decompressobj()
    raw = decompressor.decompress(blob[off : off + dlen]) + decompressor.flush()
    if not decompressor.eof or decompressor.unused_data or decompressor.unconsumed_tail:
        raise HarnessError("constructive delta zlib stream is not exactly consumed")
    off += dlen
    if off + 4 > len(blob):
        raise HarnessError("truncated constructive pose length prefix")
    plen = struct.unpack_from("<I", blob, off)[0]
    off += 4
    if off + plen > len(blob):
        raise HarnessError("truncated constructive pose section")
    pose_raw = blob[off : off + plen]
    off += plen
    if off != len(blob):
        raise HarnessError("constructive payload has trailing bytes")
    if len(raw) != math.prod(shape) or plen != 4 * math.prod(pose_shape):
        raise HarnessError("constructive payload raw section length disagrees with declared shapes")
    q = np.frombuffer(raw, dtype=np.int8).reshape(shape)
    pose = np.frombuffer(pose_raw, dtype="<f4").reshape(pose_shape)
    return header, q, pose


def _description(q: np.ndarray, poses: np.ndarray, pair_ids: Sequence[int], qstep: float) -> tuple[bytes, dict[str, int]]:
    blob = _serialize_payload(q, poses, pair_ids, qstep)
    header, parsed_q, parsed_pose = _parse_payload(blob)
    if header["selected_pair_ids"] != list(pair_ids) or not np.array_equal(parsed_q, q):
        raise HarnessError("description grammar failed exact delta parse-back")
    pose_le = np.asarray(poses, dtype="<f4")
    if not np.array_equal(parsed_pose, pose_le):
        raise HarnessError("description grammar failed exact pose parse-back")
    pose_scalar_bytes = int(pose_le.nbytes)
    pose_prefix_bytes = 4
    pose_only = pose_prefix_bytes + pose_scalar_bytes
    payload_only = len(blob) - pose_only
    return blob, {
        "payload_description_bytes": payload_only,
        "pose_bytes": pose_only,
        "pose_length_prefix_bytes": pose_prefix_bytes,
        "pose_scalar_bytes": pose_scalar_bytes,
        "total_description_bytes": len(blob),
        "total_description_bits": 8 * len(blob),
    }


def _quantize(delta: np.ndarray, qstep: float, max_delta: float) -> np.ndarray:
    bound = round(max_delta / qstep)
    if not math.isclose(bound * qstep, max_delta, rel_tol=0.0, abs_tol=1e-9) or bound > 127:
        raise HarnessError("max-delta must be an exact quantization multiple with int8 magnitude <=127")
    return np.clip(np.rint(np.asarray(delta) / qstep), -bound, bound).astype(np.int8)


def _cache(npz_path: Path, m_safe: float) -> tuple[np.memmap, np.memmap, np.memmap, np.ndarray]:
    lstars = stored_npy_memmap(npz_path, "lstars")
    margins = stored_npy_memmap(npz_path, "margins")
    poses = stored_npy_memmap(npz_path, "gt_poses")
    n_pairs = stored_npy_memmap(npz_path, "n_pairs")
    if int(np.asarray(n_pairs).reshape(())) != 600:
        raise HarnessError("evidence mode requires cache n_pairs exactly 600")
    if lstars.shape != (600, 384, 512) or margins.shape != (600, 384, 512) or poses.shape != (600, 6):
        raise HarnessError(
            f"cache shapes must be lstars=(600,384,512), margins=(600,384,512), gt_poses=(600,6); got "
            f"{lstars.shape}, {margins.shape}, {poses.shape}"
        )
    if not np.issubdtype(lstars.dtype, np.integer):
        raise HarnessError(f"lstars dtype must be integer, got {lstars.dtype}")
    if not np.issubdtype(margins.dtype, np.floating) or not np.issubdtype(poses.dtype, np.floating):
        raise HarnessError(f"margins/gt_poses must be floating, got {margins.dtype}/{poses.dtype}")
    fragility = np.empty(600, dtype=np.float64)
    label_min, label_max = 5, -1
    for i in range(600):
        labels_i = np.asarray(lstars[i])
        margins_i = np.asarray(margins[i])
        poses_i = np.asarray(poses[i])
        label_min = min(label_min, int(labels_i.min()))
        label_max = max(label_max, int(labels_i.max()))
        if not np.isfinite(margins_i).all() or not np.isfinite(poses_i).all():
            raise HarnessError(f"cache contains non-finite margins/gt_poses at pair {i}")
        fragility[i] = float(np.mean(margins_i < m_safe))
    if label_min < 0 or label_max > 4:
        raise HarnessError(f"lstars target range must be [0,4], observed [{label_min},{label_max}]")
    return lstars, margins, poses, fragility


def _select_pairs(
    lstars: np.memmap,
    margins: np.memmap,
    sample_pairs: int,
    explicit: Sequence[int] | None,
    m_safe: float,
    fragility: np.ndarray,
) -> tuple[list[int], list[dict[str, Any]], str]:
    if explicit:
        pair_ids = [int(i) for i in explicit]
        if len(set(pair_ids)) != len(pair_ids) or any(i < 0 or i >= 600 for i in pair_ids):
            raise HarnessError("--pair-indices must be unique integers in [0,600)")
        policy = "explicit_override_not_default_stratified_evidence"
    else:
        if not 1 <= sample_pairs <= 600:
            raise HarnessError("sample-pairs must be in [1,600]")
        # Lifted to tac.subset_selection (ddm_ss1, 2026-08-03) -- see the sister
        # note in tools/measure_uint8_lattice_feasibility.py. These two were
        # verbatim copies; they now share one implementation. Equivalence to the
        # previous inline code is MEASURED at every sample_pairs in 1..600 plus
        # 2,160 random trials, so no selection in either tool changes.
        pair_ids = list(quantile_stratified_indices(sample_pairs, 600, fragility))
        policy = (
            "all-600 temporal equal strata; fragility=mean(cached_margin<m_safe); alternating "
            "within-stratum 0.25/0.75 quantile; deterministic pair-index tie break; no candidate outcome peeking"
        )
    rows: list[dict[str, Any]] = []
    edges = np.linspace(0, 600, len(pair_ids) + 1, dtype=np.int64)
    for pos, pair_id in enumerate(pair_ids):
        labels = np.asarray(lstars[pair_id], dtype=np.int64)
        counts = np.bincount(labels.reshape(-1), minlength=5)
        probs = counts / counts.sum()
        entropy = -float(np.sum(probs[probs > 0] * np.log2(probs[probs > 0])))
        rows.append(
            {
                "pair_id": pair_id,
                "temporal_stratum": (
                    [int(edges[pos]), int(edges[pos + 1])] if not explicit else None
                ),
                "stratum_label": "default_equal_temporal_stratum" if not explicit else "explicit_override",
                "fragility_fraction_margin_below_m_safe": float(fragility[pair_id]),
                "class_histogram": counts.tolist(),
                "class_entropy_bits": entropy,
            }
        )
    return pair_ids, rows, policy


def _load_banked_frames(checkpoint: Path, pair_ids: Sequence[int]) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Render arbitrary selected pairs via the canonical NumPy oracle and reindexed code."""

    from tools import levelset_byte_close_and_eval as lbc

    if checkpoint.name != "levelset_witness_ema_BEST.npz":
        raise HarnessError("checkpoint must be the banked levelset_witness_ema_BEST.npz, never live/current EMA")
    params, cfg = lbc._load_levelset_ckpt(checkpoint.parent, checkpoint.name)
    with np.load(checkpoint, allow_pickle=False) as raw:
        required = ("__cfg_freq_across", "__cfg_freq_along")
        if any(key not in raw.files for key in required):
            raise HarnessError("checkpoint lacks persisted self-orient frequency custody")
        freq_across = float(raw["__cfg_freq_across"])
        freq_along = float(raw["__cfg_freq_along"])
    if not math.isclose(freq_across, 32.0) or not math.isclose(freq_along, 8.0):
        raise HarnessError(
            f"v9c2 frequency custody mismatch: expected persisted freq_across=32,freq_along=8; got {freq_across},{freq_along}"
        )
    code = np.concatenate([params["code"][2 * i : 2 * i + 2] for i in pair_ids], axis=0)
    selected_params = dict(params)
    selected_params["code"] = code
    selected_cfg = dict(cfg)
    selected_cfg["n_pairs"] = len(pair_ids)
    so = lbc.detect_self_orient(
        selected_cfg,
        {"freq_across": freq_across, "freq_along": freq_along, "tau": 4.0, "iters": 4},
    )
    blob, breakdown = lbc.build_levelset_blob(selected_params, selected_cfg, so, None)
    manifest, _base_bytes, _code_bytes, _pose_bytes, _lane_bytes, _carrier_bytes = lbc._read_blob_bytes(blob)
    frames, _ = lbc.numpy_oracle_reference_frames(
        selected_params, code, manifest, len(pair_ids), lane_pairs=None, pose_carrier=None
    )
    frame1 = np.stack([frames[2 * i + 1] for i in range(len(pair_ids))], axis=0)
    if frame1.shape != (len(pair_ids), CAMERA_HW[0], CAMERA_HW[1], 3) or frame1.dtype != np.uint8:
        raise HarnessError(f"canonical oracle returned unexpected frame1 shape/dtype {frame1.shape}/{frame1.dtype}")
    return frame1, manifest, breakdown


def _load_segnet(upstream_root: Path) -> Any:
    import torch
    from safetensors.torch import load_file

    modules_path = upstream_root / "modules.py"
    weights = upstream_root / "models" / "segnet.safetensors"
    if not modules_path.is_file() or not weights.is_file():
        raise HarnessError(f"upstream root must contain modules.py and models/segnet.safetensors: {upstream_root}")
    root_s = str(upstream_root.resolve())
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    loaded = sys.modules.get("modules")
    if loaded is not None and Path(loaded.__file__).resolve() != modules_path.resolve():
        raise HarnessError(f"modules already imported from a different upstream root: {loaded.__file__}")
    from modules import SegNet  # type: ignore

    model = SegNet().eval().to("cpu")
    model.load_state_dict(load_file(str(weights), device="cpu"))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    torch.use_deterministic_algorithms(True)
    return model


def _winner_metrics(logits: Any, target: Any, m_safe: float) -> tuple[Any, Any, Any]:
    import torch

    target_logit = logits.gather(1, target[:, None]).squeeze(1)
    gaps = target_logit[:, None] - logits
    mask = torch.nn.functional.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).bool()
    violations = torch.relu(float(m_safe) - gaps).masked_fill(mask, 0.0)
    feasible = (gaps.masked_fill(mask, float("inf")).amin(dim=1) >= float(m_safe))
    argmax_feasible = logits.argmax(dim=1).eq(target)
    return violations, feasible, argmax_feasible


def _camera_from_delta(delta: Any, base_nchw: Any, ah: SparseAxis, aw: SparseAxis, qstep: float, max_delta: float, *, ste: bool) -> Any:
    import torch

    bound = round(max_delta / qstep)
    scaled = delta / qstep
    qhard = torch.clamp(torch.round(scaled), -bound, bound)
    qvalue = scaled + (qhard - scaled).detach() if ste else qhard
    lift = _lift_b_torch(qvalue * qstep, ah, aw)
    camera_float = torch.clamp(base_nchw + lift, 0.0, 255.0)
    hard = torch.round(camera_float)
    return camera_float + (hard - camera_float).detach() if ste else hard


def _hard_camera_from_scorer_delta(scorer_delta: Any, base_nchw: Any, ah: SparseAxis, aw: SparseAxis) -> tuple[Any, Any]:
    import torch

    preclamp = base_nchw + _lift_b_torch(scorer_delta, ah, aw)
    camera = torch.round(torch.clamp(preclamp, 0.0, 255.0))
    return camera, preclamp


def _logits(model: Any, camera_nchw: Any) -> Any:
    import torch

    pair = torch.stack((camera_nchw, camera_nchw), dim=1)
    return model(model.preprocess_input(pair.float()))


def _config(args: argparse.Namespace, pair_ids: Sequence[int], hashes: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "seed": args.seed,
        "pair_ids": [int(i) for i in pair_ids],
        "steps": args.steps,
        "m_safe": args.m_safe,
        "quantization_step": args.quantization_step,
        "max_delta": args.max_delta,
        "learning_rate": args.learning_rate,
        "description_proxy_weight": args.description_proxy_weight,
        "cpu_threads": args.cpu_threads,
        "input_hashes": hashes,
        "update": "stateless gradient step plus deterministic soft-threshold proximal map",
    }


def _config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(config)).hexdigest()


def _prox_update(delta: np.ndarray, grad: np.ndarray, learning_rate: float, l1_weight: float, max_delta: float) -> np.ndarray:
    candidate = np.asarray(delta) - learning_rate * np.asarray(grad)
    threshold = learning_rate * l1_weight
    result = np.sign(candidate) * np.maximum(np.abs(candidate) - threshold, 0.0)
    return np.clip(result, -max_delta, max_delta).astype(np.float32)


def _state_arrays(
    delta: np.ndarray,
    best_delta: np.ndarray,
    best_step: int,
    best_key: tuple[int, int, float] | None,
    step: int,
    history: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, np.ndarray]:
    return {
        "delta": np.asarray(delta, dtype=np.float32),
        "best_delta": np.asarray(best_delta, dtype=np.float32),
        "best_step": np.asarray(best_step, dtype=np.int64),
        "best_key": np.asarray(best_key if best_key is not None else (0, 0, math.inf), dtype=np.float64),
        "step": np.asarray(step, dtype=np.int64),
        "pair_ids": np.asarray(config["pair_ids"], dtype=np.int64),
        "history_json": np.frombuffer(_canonical_json(history), dtype=np.uint8),
        "config_json": np.frombuffer(_canonical_json(config), dtype=np.uint8),
        "config_sha256": np.frombuffer(_config_hash(config).encode(), dtype=np.uint8),
    }


def _load_state(
    path: Path, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, int, tuple[int, int, float], int, list[dict[str, Any]]]:
    with np.load(path, allow_pickle=False) as state:
        stored_config = json.loads(bytes(state["config_json"]).decode())
        stored_hash = bytes(state["config_sha256"]).decode()
        if stored_hash != _config_hash(stored_config) or stored_config != config:
            raise HarnessError("resume state config/input hashes differ from requested solve")
        delta = np.asarray(state["delta"], dtype=np.float32)
        best_delta = np.asarray(state["best_delta"], dtype=np.float32)
        best_step = int(state["best_step"])
        best_key_raw = np.asarray(state["best_key"], dtype=np.float64)
        step = int(state["step"])
        history = json.loads(bytes(state["history_json"]).decode())
    if not np.isfinite(delta).all() or not np.isfinite(best_delta).all():
        raise HarnessError("resume state contains non-finite delta values")
    max_delta = float(config["max_delta"])
    if np.max(np.abs(delta), initial=0.0) > max_delta or np.max(
        np.abs(best_delta), initial=0.0
    ) > max_delta:
        raise HarnessError("resume state delta exceeds configured max_delta")
    if not isinstance(history, list) or not history:
        raise HarnessError("resume history must be a non-empty list")
    if step < 0 or step > int(config["steps"]):
        raise HarnessError(f"resume completed step {step} is outside configured range")
    expected_steps = list(range(step + 1))
    observed_steps = [row.get("step") for row in history]
    if observed_steps != expected_steps:
        raise HarnessError(f"resume history is not contiguous through step {step}: {observed_steps}")
    if best_key_raw.shape != (3,) or not np.isfinite(best_key_raw).all():
        raise HarnessError("resume best_key is missing or non-finite")
    best_key = (int(best_key_raw[0]), int(best_key_raw[1]), float(best_key_raw[2]))
    if best_step < 0 or best_step > step:
        raise HarnessError("resume best_step is outside completed history")
    return delta, best_delta, best_step, best_key, step, history


def _step_row(
    step: int,
    logits: Any,
    target: Any,
    q: np.ndarray,
    poses: np.ndarray,
    pair_ids: Sequence[int],
    qstep: float,
    m_safe: float,
    initial_feasible: np.ndarray,
    hard_camera: Any,
    preclamp_camera: Any,
    elapsed: float,
) -> tuple[dict[str, Any], np.ndarray]:
    import torch

    violations, feasible, argmax_feasible = _winner_metrics(logits, target, m_safe)
    feasible_np = feasible.detach().cpu().numpy()
    blob, desc = _description(q, poses, pair_ids, qstep)
    row: dict[str, Any] = {
        "step": step,
        "feasible_fraction": float(feasible.float().mean()),
        "argmax_feasible_fraction": float(argmax_feasible.float().mean()),
        "violated_pixels": int((~feasible).sum()),
        "newly_feasible_vs_step0": int(np.logical_and(feasible_np, ~initial_feasible).sum()),
        "regressed_vs_step0": int(np.logical_and(~feasible_np, initial_feasible).sum()),
        "hinge_sum": float(violations.sum()),
        "hinge_mean": float(violations.sum() / (target.numel() * (logits.shape[1] - 1))),
        **desc,
        "nonzero_quantized_symbols": int(np.count_nonzero(q)),
        "clip_fraction": float(
            torch.logical_or(preclamp_camera < 0, preclamp_camera > 255).float().mean()
        ),
        "hard_saturation_fraction": float(
            torch.logical_or(hard_camera <= 0, hard_camera >= 255).float().mean()
        ),
        "elapsed_seconds": float(elapsed),
        "payload_sha256": hashlib.sha256(blob).hexdigest(),
    }
    return row, feasible_np


def run(args: argparse.Namespace) -> int:
    import torch

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    np.random.seed(args.seed)
    checkpoint = args.checkpoint.expanduser().resolve()
    cache_path = args.gt_cache.expanduser().resolve()
    upstream = args.upstream_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    state_path = args.state.expanduser().resolve()
    _refuse_tmp(output, "output")
    _refuse_tmp(state_path, "state")
    for path, label in ((checkpoint, "checkpoint"), (cache_path, "gt-cache")):
        if not path.is_file():
            raise HarnessError(f"{label} missing: {path}")
    modules_path = upstream / "modules.py"
    weights_path = upstream / "models" / "segnet.safetensors"
    for path, label in ((modules_path, "upstream modules"), (weights_path, "SegNet weights")):
        if not path.is_file():
            raise HarnessError(f"{label} missing: {path}")
    protected_inputs = {checkpoint, cache_path, modules_path.resolve(), weights_path.resolve()}
    if output == state_path:
        raise HarnessError("--output and --state must be distinct paths")
    if output in protected_inputs or state_path in protected_inputs:
        raise HarnessError("output/state may not equal any read-only input")
    sacred_root = checkpoint.parent.resolve()
    for path, label in ((output, "output"), (state_path, "state")):
        try:
            path.relative_to(sacred_root)
        except ValueError:
            pass
        else:
            raise HarnessError(f"{label} may not be written inside sacred checkpoint directory {sacred_root}")

    hashes = {
        "checkpoint_sha256": _sha256(checkpoint),
        "gt_cache_sha256": _sha256(cache_path),
        "upstream_modules_sha256": _sha256(modules_path),
        "segnet_weights_sha256": _sha256(weights_path),
    }
    lstars, margins, pose_cache, fragility = _cache(cache_path, args.m_safe)
    pair_ids, selection_rows, selection_policy = _select_pairs(
        lstars, margins, args.sample_pairs, args.pair_indices, args.m_safe, fragility
    )
    targets_np = np.stack([np.asarray(lstars[i], dtype=np.int64) for i in pair_ids])
    poses = np.stack([np.asarray(pose_cache[i], dtype=np.float32) for i in pair_ids])
    base_nhwc, manifest, bank_breakdown = _load_banked_frames(checkpoint, pair_ids)
    ah, aw, algebra = _resize_algebra(CAMERA_HW, SCORER_HW)

    probe = np.linspace(-1.0, 1.0, len(pair_ids) * 3 * SCORER_HW[0] * SCORER_HW[1], dtype=np.float64)
    probe = probe.reshape(len(pair_ids), 3, *SCORER_HW)
    ab_residual = float(np.max(np.abs(_apply_a_numpy(_lift_b_numpy(probe, ah, aw), ah, aw) - probe)))
    if ab_residual > 1e-10:
        raise HarnessError(f"right-inverse certification failed: max|AB-I|={ab_residual:.3e}")
    algebra["ab_max_abs_residual"] = ab_residual

    contest_x = np.cos(np.arange(CAMERA_HW[0] * CAMERA_HW[1], dtype=np.float64) * 0.017)
    contest_x = contest_x.reshape(1, 1, *CAMERA_HW)
    px = _lift_b_numpy(_apply_a_numpy(contest_x, ah, aw), ah, aw)
    p2x = _lift_b_numpy(_apply_a_numpy(px, ah, aw), ah, aw)
    kernel_x = contest_x - px
    p_idempotence = float(np.max(np.abs(p2x - px)))
    kernel_projection = float(np.max(np.abs(_apply_a_numpy(kernel_x, ah, aw))))
    orth_inner = float(np.sum(px * kernel_x))
    orth_relative = abs(orth_inner) / max(float(np.linalg.norm(px) * np.linalg.norm(kernel_x)), 1e-300)
    if p_idempotence > 1e-10 or kernel_projection > 1e-10 or orth_relative > 1e-10:
        raise HarnessError(
            "contest-geometry projector certification failed: "
            f"P2-P={p_idempotence:.3e}, A(I-P)={kernel_projection:.3e}, orth={orth_relative:.3e}"
        )
    algebra.update(
        {
            "p_idempotence_max_abs": p_idempotence,
            "a_i_minus_p_max_abs": kernel_projection,
            "range_kernel_inner_product_abs": abs(orth_inner),
            "range_kernel_orthogonality_relative": orth_relative,
        }
    )

    torch.manual_seed(args.seed)
    model = _load_segnet(upstream)
    base = torch.from_numpy(base_nhwc).permute(0, 3, 1, 2).float()
    target = torch.from_numpy(targets_np).long()
    config = _config(args, pair_ids, hashes)
    expected_shape = (len(pair_ids), 3, *SCORER_HW)
    if state_path.exists():
        delta_np, best_delta_np, best_step, best_key, completed_step, history = _load_state(
            state_path, config
        )
        if delta_np.shape != expected_shape or best_delta_np.shape != expected_shape:
            raise HarnessError("resume delta shape differs from selected-pair scorer grid")
    else:
        delta_np = np.zeros(expected_shape, dtype=np.float32)
        best_delta_np = delta_np.copy()
        best_step = -1
        best_key = None
        completed_step = -1
        history = []

    zero_delta = torch.zeros(expected_shape, dtype=torch.float32)
    with torch.no_grad():
        zero_camera = _camera_from_delta(
            zero_delta, base, ah, aw, args.quantization_step, args.max_delta, ste=False
        )
        zero_logits = _logits(model, zero_camera)
        _, zero_feasible, _ = _winner_metrics(zero_logits, target, args.m_safe)
    initial_feasible = zero_feasible.cpu().numpy()
    start = time.monotonic()
    for step in range(completed_step + 1, args.steps + 1):
        delta = torch.tensor(delta_np, dtype=torch.float32, requires_grad=True)
        camera_ste = _camera_from_delta(
            delta, base, ah, aw, args.quantization_step, args.max_delta, ste=True
        )
        logits_ste = _logits(model, camera_ste)
        violations, _, _ = _winner_metrics(logits_ste, target, args.m_safe)
        proxy = torch.log1p(delta.abs() / args.quantization_step).mean()
        # The full sum preserves each pair's locally discriminated lr~5 gradient scale;
        # independent batch members are deliberately not divided by sample count.
        objective = violations.sum() + args.description_proxy_weight * proxy

        q = _quantize(delta_np, args.quantization_step, args.max_delta)
        hard_delta = torch.from_numpy(q.astype(np.float32) * args.quantization_step)
        with torch.no_grad():
            camera_hard, camera_preclamp = _hard_camera_from_scorer_delta(hard_delta, base, ah, aw)
            logits_hard = _logits(model, camera_hard)
        row, feasible_np = _step_row(
            step,
            logits_hard,
            target,
            q,
            poses,
            pair_ids,
            args.quantization_step,
            args.m_safe,
            initial_feasible,
            camera_hard,
            camera_preclamp,
            time.monotonic() - start,
        )
        row["m_safe"] = args.m_safe
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        key = (-int(feasible_np.sum()), row["total_description_bytes"], row["hinge_sum"])
        if best_key is None or key < best_key:
            best_key = key
            best_step = step
            best_delta_np = delta_np.copy()
        if step == args.steps:
            _atomic_state(
                state_path,
                _state_arrays(delta_np, best_delta_np, best_step, best_key, step, history, config),
            )
            break
        objective.backward()
        grad = delta.grad.detach().cpu().numpy()
        next_delta_np = _prox_update(
            delta_np,
            grad,
            args.learning_rate,
            args.description_proxy_weight,
            args.max_delta,
        )
        # The state at completed step k carries the already-derived iterate k+1.
        # A resume therefore loses no stateless update between the atomic save and exit.
        _atomic_state(
            state_path,
            _state_arrays(next_delta_np, best_delta_np, best_step, best_key, step, history, config),
        )
        delta_np = next_delta_np

    best_q = _quantize(best_delta_np, args.quantization_step, args.max_delta)
    best_hard_delta = torch.from_numpy(best_q.astype(np.float32) * args.quantization_step)
    with torch.no_grad():
        best_camera, best_preclamp = _hard_camera_from_scorer_delta(best_hard_delta, base, ah, aw)
        best_logits = _logits(model, best_camera)
    best_viol, best_feasible, best_argmax = _winner_metrics(best_logits, target, args.m_safe)
    best_blob, best_desc = _description(best_q, poses, pair_ids, args.quantization_step)
    _best_header, parsed_best_q, parsed_best_poses = _parse_payload(best_blob)
    if not np.array_equal(parsed_best_poses, np.asarray(poses, dtype="<f4")):
        raise HarnessError("best payload pose parse-back drift")
    parsed_best_delta = torch.from_numpy(
        parsed_best_q.astype(np.float32) * args.quantization_step
    )
    with torch.no_grad():
        parsed_camera, _parsed_preclamp = _hard_camera_from_scorer_delta(
            parsed_best_delta, base, ah, aw
        )
        parsed_logits = _logits(model, parsed_camera)
    parse_camera_max = float(torch.max(torch.abs(parsed_camera - best_camera)))
    parse_logits_max = float(torch.max(torch.abs(parsed_logits - best_logits)))
    if parse_camera_max != 0.0 or parse_logits_max != 0.0:
        raise HarnessError(
            f"best serialized parse-back changed hard forward: camera={parse_camera_max}, logits={parse_logits_max}"
        )
    per_pair = []
    for j, pair_id in enumerate(pair_ids):
        per_pair.append(
            {
                "pair_id": pair_id,
                "feasible_fraction": float(best_feasible[j].float().mean()),
                "argmax_feasible_fraction": float(best_argmax[j].float().mean()),
                "violated_pixels": int((~best_feasible[j]).sum()),
                "hinge_sum": float(best_viol[j].sum()),
            }
        )
    float_lift = _lift_b_numpy(best_q.astype(np.float64) * args.quantization_step, ah, aw)
    scorer_target_delta = best_q.astype(np.float64) * args.quantization_step
    float_ab_residual = _apply_a_numpy(float_lift, ah, aw) - scorer_target_delta
    quantized_camera_correction = (
        best_camera.cpu().numpy().astype(np.float64) - base.cpu().numpy().astype(np.float64)
    )
    hard_scorer_drift = _apply_a_numpy(quantized_camera_correction, ah, aw) - scorer_target_delta
    leakage = quantized_camera_correction - _lift_b_numpy(
        _apply_a_numpy(quantized_camera_correction, ah, aw), ah, aw
    )
    ste_delta = torch.tensor(best_delta_np, dtype=torch.float32, requires_grad=True)
    ste_logits = _logits(
        model,
        _camera_from_delta(ste_delta, base, ah, aw, args.quantization_step, args.max_delta, ste=True),
    )
    closure = float(torch.max(torch.abs(ste_logits.detach() - best_logits)))
    state_stat = state_path.stat()
    any_landed = bool(np.logical_and(best_feasible.cpu().numpy(), ~initial_feasible).any())
    receipt = {
        "schema": SCHEMA,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "pointer_moved": False,
        "optimality_claim": "none_local_iterative_constrained_solve",
        "authority_axis": AXIS,
        "verdict_scope": (
            "INSTANCE: selected real v9c2 EMA pairs, fixed scorer-grid-delta formulation, declared config; "
            "not a family/paradigm/global-optimality verdict"
        ),
        "status": "MEASURED_LOCAL_POC_LANDED_PIXEL" if any_landed else "BLOCKED_NO_INITIAL_PIXEL_ENTERED_MARGIN_CELL",
        "input_paths": {
            "checkpoint": str(checkpoint),
            "gt_cache": str(cache_path),
            "upstream_root": str(upstream),
        },
        "input_hashes": hashes,
        "checkpoint_constant_excluded_from_incremental_description": {
            "bytes": checkpoint.stat().st_size,
            "sha256": hashes["checkpoint_sha256"],
        },
        "sample_policy": selection_policy,
        "selected_pairs": selection_rows,
        "config": config,
        "resize_algebra": algebra,
        "bank_manifest": manifest,
        "bank_blob_accounting_for_manifest_build_only": bank_breakdown,
        "history": history,
        "best": {
            **best_desc,
            "step": best_step,
            "lexicographic_key": list(best_key) if best_key is not None else None,
            "payload_sha256": hashlib.sha256(best_blob).hexdigest(),
            "feasible_fraction": float(best_feasible.float().mean()),
            "argmax_feasible_fraction": float(best_argmax.float().mean()),
            "violated_pixels": int((~best_feasible).sum()),
            "hinge_sum": float(best_viol.sum()),
            "nonzero_quantized_symbols": int(np.count_nonzero(best_q)),
            "preclamp_clip_fraction": float(
                torch.logical_or(best_preclamp < 0, best_preclamp > 255).float().mean()
            ),
            "per_pair": per_pair,
        },
        "hard_authority": {
            "ste_vs_hard_logits_max_abs": closure,
            "serialized_parseback_camera_max_abs": parse_camera_max,
            "serialized_parseback_logits_max_abs": parse_logits_max,
            "float_lift_max_abs": float(np.max(np.abs(float_lift))),
            "float_ab_residual_max_abs": float(np.max(np.abs(float_ab_residual))),
            "hard_scorer_drift_vs_quantized_payload_l2": float(np.linalg.norm(hard_scorer_drift.reshape(-1))),
            "hard_scorer_drift_vs_quantized_payload_max_abs": float(np.max(np.abs(hard_scorer_drift))),
            "camera_quantization_row_space_leakage_l2": float(np.linalg.norm(leakage.reshape(-1))),
            "camera_quantization_row_space_leakage_max_abs": float(np.max(np.abs(leakage))),
            "warning": "camera clamp/uint8 quantization means A(Q(BY)) is not claimed equal to Y",
        },
        "state": {
            "path": str(state_path),
            "bytes": state_stat.st_size,
            "sha256": _sha256(state_path),
            "preserved": True,
            "atomic_per_step": True,
        },
        "provenance": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION,
            "platform_system": platform.system(),
            "platform_release": platform.release(),
            "platform_machine": platform.machine(),
            "cwd": str(Path.cwd().resolve()),
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
            ).strip(),
            "harness_sha256": _sha256(Path(__file__).resolve()),
            "numpy_oracle_module_sha256": _sha256(REPO / "tools" / "levelset_byte_close_and_eval.py"),
            "resize_projector_module_sha256": _sha256(
                REPO / "src" / "tac" / "optimization" / "resize_null_preimage.py"
            ),
        },
        "rebuild_command": shlex.join([sys.executable, *sys.argv]),
        "open_debts": [
            "Pose scalars are counted/stored but this PoC does not solve their joint PoseNet realization",
            "incremental grammar is not a full archive byte-close",
            "decode-under-30-min and exact contest CPU/CUDA closure remain owed",
        ],
    }
    _atomic_json(output, receipt)
    return 0 if any_landed else 2


def self_test() -> int:
    from tac.optimization.resize_null_preimage import ResizeProjector

    projector = ResizeProjector.build(camera_h=8, camera_w=10, scorer_h=3, scorer_w=4)
    ah, aw = _sparse_axis(projector.rh), _sparse_axis(projector.rw)
    rng = np.random.default_rng(20260718)
    y = rng.normal(size=(2, 3, 3, 4))
    x = rng.normal(size=(2, 3, 8, 10))
    by = _lift_b_numpy(y, ah, aw)
    if np.max(np.abs(_apply_a_numpy(by, ah, aw) - y)) > 1e-11:
        raise HarnessError("self-test AB failed")
    px = _lift_b_numpy(_apply_a_numpy(x, ah, aw), ah, aw)
    p2x = _lift_b_numpy(_apply_a_numpy(px, ah, aw), ah, aw)
    if np.max(np.abs(p2x - px)) > 1e-11:
        raise HarnessError("self-test P^2=P failed")
    if np.max(np.abs(_apply_a_numpy(x - px, ah, aw))) > 1e-11:
        raise HarnessError("self-test A(I-P)=0 failed")
    rank = ah.n_out * aw.n_out
    null = ah.n_in * aw.n_in - rank
    if rank != 12 or null != 68:
        raise HarnessError("self-test rank/null dimensions failed")
    _real_ah, _real_aw, real = _resize_algebra(CAMERA_HW, SCORER_HW)
    expected_real = {
        "rank_per_channel": 196608,
        "null_dimension_per_channel": 820728,
        "height_blind_coordinates": 106,
        "width_blind_coordinates": 140,
        "axis_blind_pixels": 230904,
    }
    for key, expected in expected_real.items():
        if real[key] != expected:
            raise HarnessError(f"self-test contest {key}={real[key]} != {expected}")
    q = rng.integers(-7, 8, size=(2, 3, 3, 4), dtype=np.int8)
    pose = rng.normal(size=(2, 6)).astype(np.float32)
    blob = _serialize_payload(q, pose, [11, 487], 0.25)
    header, q2, pose2 = _parse_payload(blob)
    if header["selected_pair_ids"] != [11, 487] or not np.array_equal(q, q2) or not np.array_equal(pose, pose2):
        raise HarnessError("self-test serialization parse-back failed")
    config = {
        "schema": STATE_SCHEMA,
        "pair_ids": [11, 487],
        "steps": 2,
        "max_delta": 16.0,
        "test": True,
    }
    with tempfile.TemporaryDirectory() as td:
        state = Path(td) / "state.npz"
        history = [{"step": 0, "total_description_bytes": len(blob)}]
        delta = q.astype(np.float32) * 0.25
        grad0 = rng.normal(size=delta.shape).astype(np.float32)
        grad1 = rng.normal(size=delta.shape).astype(np.float32)
        next_uninterrupted = _prox_update(delta, grad0, 5.0, 1e-5, 16.0)
        key = (-7, len(blob), 1.25)
        _atomic_state(state, _state_arrays(next_uninterrupted, delta, 0, key, 0, history, config))
        d2, b2, bs2, bk2, s2, h2 = _load_state(state, config)
        resumed_second = _prox_update(d2, grad1, 5.0, 1e-5, 16.0)
        uninterrupted_second = _prox_update(next_uninterrupted, grad1, 5.0, 1e-5, 16.0)
        if (
            not np.array_equal(next_uninterrupted, d2)
            or not np.array_equal(delta, b2)
            or bs2 != 0
            or bk2 != key
            or s2 != 0
            or h2 != history
            or not np.array_equal(resumed_second, uninterrupted_second)
        ):
            raise HarnessError("self-test resume identity failed")
    print(
        json.dumps(
            {
                "self_test": "PASS_SYNTHETIC_NOT_EVIDENCE",
                "ab": True,
                "p_idempotent": True,
                "a_i_minus_p_zero": True,
                "rank": rank,
                "null_dimension": null,
                "axis_blind_h": ah.blind_count,
                "axis_blind_w": aw.blind_count,
                "contest_rank": real["rank_per_channel"],
                "contest_null_dimension": real["null_dimension_per_channel"],
                "contest_axis_blind_h": real["height_blind_coordinates"],
                "contest_axis_blind_w": real["width_blind_coordinates"],
                "contest_axis_blind_pixels": real["axis_blind_pixels"],
                "serialization_parseback": True,
                "resume_next_iterate_identity": True,
            },
            sort_keys=True,
        )
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="read-only levelset_witness_ema_BEST.npz")
    parser.add_argument("--gt-cache", type=Path, help="read-only ZIP_STORED gt_n600.npz")
    parser.add_argument("--upstream-root", type=Path, help="upstream tree containing modules.py and model weights")
    parser.add_argument("--output", type=Path, help="durable atomic JSON receipt")
    parser.add_argument("--state", type=Path, help="durable atomic resumable NPZ state")
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument("--sample-pairs", type=int, default=6)
    parser.add_argument("--pair-indices", type=int, nargs="+", help="explicit override; not default stratified evidence")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--m-safe", type=float, default=DEFAULT_M_SAFE)
    parser.add_argument("--quantization-step", type=float, default=0.25)
    parser.add_argument("--max-delta", type=float, default=16.0)
    parser.add_argument("--learning-rate", type=float, default=5.0)
    parser.add_argument("--description-proxy-weight", type=float, default=1e-5)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--self-test", action="store_true", help="small synthetic algebra/grammar/resume test; never evidence")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    missing = [name for name in ("checkpoint", "gt_cache", "upstream_root", "output", "state") if getattr(args, name) is None]
    if missing:
        parser.error("normal evidence mode requires " + ", ".join("--" + name.replace("_", "-") for name in missing))
    if (
        args.steps < 0
        or args.learning_rate <= 0
        or args.quantization_step <= 0
        or args.max_delta <= 0
        or args.m_safe < 0
        or args.description_proxy_weight < 0
    ):
        parser.error(
            "steps/m-safe/description-proxy-weight must be nonnegative and "
            "learning-rate/quantization-step/max-delta must be positive"
        )
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    try:
        return run(args)
    except (HarnessError, FileNotFoundError, ValueError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
