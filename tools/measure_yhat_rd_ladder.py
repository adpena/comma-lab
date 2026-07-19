#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure a compact shared-plane (y-hat) rate-distortion ladder.

The tool has three deliberately separate stages:

``prepare``
    Byte-close the read-only c2 ep725 EMA-best checkpoint, preserve the real
    full-n600 archive, and run the shipped receiver on an arbitrary real-pair
    subset without requantizing the selected code rows.

``measure``
    Measure one at-most-twelve-pair chunk.  Every rung is decoded to an exact
    uint8 lattice realization and scored by the frozen full DistortionNet
    against the source pair.  Per-pair stages and state are atomic/resumable.

``compose``
    Refuse overlapping chunks and emit the machine-readable n24 table.

This is a local advisory measurement, not an evaluator or score surface.
Frame0 is explicitly the source frame for every rung; this isolates the
frame1 shared-plane description and is not a contest-complete two-plane packet.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import struct
import subprocess
import sys
import time
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
for _path in (REPO, SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.optimization.uint8_lattice_feasibility import (  # noqa: E402
    BlockSolveStatus,
    DisjointResizeOperator,
)
from tools.measure_uint8_lattice_feasibility import (  # noqa: E402
    stored_npy_memmap,
)

SCHEMA_PREP = "yhat_rd_ladder_witness_prepare.v1"
SCHEMA_CHUNK = "yhat_rd_ladder_chunk.v1"
SCHEMA_TABLE = "yhat_rd_ladder_table.v1"
STATE_SCHEMA = "yhat_rd_ladder_chunk_state.v1"
AXIS = "[macOS-CPU advisory n24] NON-PROMOTABLE"
POINTER = "0.1910828242 [contest-CPU Linux x86_64] UNMOVED"
CAMERA_HW = (874, 1164)
SCORER_HW = (384, 512)
RATE_DENOM = 37_545_489
SEED = 20260719
MAX_CHUNK = 12
DEFAULT_PAIRS = tuple(range(0, 600, 50)) + tuple(range(10, 600, 50))
DEFAULT_CACHE = Path("/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
DEFAULT_UPSTREAM = Path("/Users/adpena/Projects/pact/upstream")
DEFAULT_SACRED = Path("/Users/adpena/Projects/pact/experiments/results/levelset_n600_witness_20260717T113932Z")
DEFAULT_CKPT = DEFAULT_SACRED / "levelset_witness_ema_BEST.npz"
SSD_ROOTS = (
    Path("/Volumes/VertigoDataTier/pact"),
    Path("/Volumes/APDataStore/pact"),
)

_DESC_MAGIC = b"YHATRD1\0"
_DESC_HEADER = struct.Struct(">8sBBHHHII32s")
_CONTAINER_MAGIC = b"YHATC1\0\0"
_CONTAINER_PREFIX = struct.Struct(">8sI")
_CONTAINER_ITEM = struct.Struct(">II")
_DTYPE_IDS = {"i32_numerator": 1, "u8_plane": 2}
_ID_DTYPES = {value: key for key, value in _DTYPE_IDS.items()}
RUNG_ORDER = (
    "source_exact_i32",
    "witness_byteclosed_i32",
    "witness_q6_u8",
    "witness_q4_u8",
)


class LadderError(RuntimeError):
    """Fail-closed descriptor, custody, scorer, or composition error."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return _sha256_bytes(np.ascontiguousarray(value).tobytes())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, value: bytes, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        if mode is not None:
            tmp.chmod(mode)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n",
    )


def _write_once_or_equal(path: Path, value: bytes, *, mode: int | None = None) -> None:
    if path.exists():
        if path.read_bytes() != value:
            raise LadderError(f"preserved artifact differs from deterministic rebuild: {path}")
        return
    _atomic_bytes(path, value, mode=mode)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _durable_output(path: Path, field: str, *, require_ssd: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    for root in (Path("/tmp"), Path("/private/tmp"), Path("/var/tmp")):
        if _is_relative_to(resolved, root):
            raise LadderError(f"{field} must be durable, not under {root}: {resolved}")
    if _is_relative_to(resolved, DEFAULT_SACRED.resolve()):
        raise LadderError(f"{field} may not mutate sacred run tree: {resolved}")
    if require_ssd and not any(root.exists() and _is_relative_to(resolved, root.resolve()) for root in SSD_ROOTS):
        raise LadderError(f"{field} must use the SSD waterfall: {resolved}")
    return resolved


def _tree_snapshot(root: Path) -> dict[str, Any]:
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


def serialize_plane_description(
    plane: np.ndarray,
    *,
    encoding: str,
    denominator: int,
    quant_levels: int = 0,
) -> bytes:
    """Serialize one complete, independently parseable scorer plane."""

    if encoding not in _DTYPE_IDS:
        raise LadderError(f"unknown plane encoding: {encoding}")
    array = np.asarray(plane)
    if array.shape != (*SCORER_HW, 3):
        raise LadderError(f"plane shape must be {(*SCORER_HW, 3)}, got {array.shape}")
    if encoding == "i32_numerator":
        if not np.issubdtype(array.dtype, np.integer):
            raise LadderError("i32 numerator plane must be integer")
        if np.any(array < np.iinfo(np.int32).min) or np.any(array > np.iinfo(np.int32).max):
            raise LadderError("numerator plane exceeds signed-int32 descriptor contract")
        body = np.asarray(array, dtype=">i4").tobytes(order="C")
    else:
        if array.dtype != np.uint8:
            raise LadderError("u8 plane descriptor requires exact uint8 input")
        body = np.ascontiguousarray(array).tobytes(order="C")
    if not 1 <= int(denominator) <= np.iinfo(np.uint32).max:
        raise LadderError("descriptor denominator must fit positive uint32")
    if not 0 <= int(quant_levels) <= np.iinfo(np.uint32).max:
        raise LadderError("quant_levels must fit uint32")
    header = _DESC_HEADER.pack(
        _DESC_MAGIC,
        1,
        _DTYPE_IDS[encoding],
        SCORER_HW[0],
        SCORER_HW[1],
        3,
        int(denominator),
        int(quant_levels),
        hashlib.sha256(body).digest(),
    )
    return header + body


def parse_plane_description(payload: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    if len(payload) < _DESC_HEADER.size:
        raise LadderError("truncated yhat descriptor")
    magic, version, dtype_id, height, width, channels, denominator, levels, expected = _DESC_HEADER.unpack_from(payload)
    if magic != _DESC_MAGIC or version != 1 or dtype_id not in _ID_DTYPES:
        raise LadderError("invalid yhat descriptor header")
    if (height, width, channels) != (*SCORER_HW, 3):
        raise LadderError("yhat descriptor geometry mismatch")
    encoding = _ID_DTYPES[dtype_id]
    itemsize = 4 if encoding == "i32_numerator" else 1
    expected_bytes = height * width * channels * itemsize
    body = payload[_DESC_HEADER.size :]
    if len(body) != expected_bytes or hashlib.sha256(body).digest() != expected:
        raise LadderError("yhat descriptor body size/hash custody failure")
    dtype = ">i4" if encoding == "i32_numerator" else np.uint8
    array = np.frombuffer(body, dtype=dtype).reshape(height, width, channels)
    array = array.astype(np.int32) if encoding == "i32_numerator" else array.copy()
    return array, {
        "encoding": encoding,
        "denominator": int(denominator),
        "quant_levels": int(levels),
        "payload_bytes": len(payload),
        "payload_sha256": _sha256_bytes(payload),
    }


def serialize_chunk_container(items: Sequence[tuple[int, bytes]]) -> bytes:
    if not items or len({int(pair) for pair, _ in items}) != len(items):
        raise LadderError("chunk descriptor container needs unique nonempty pair rows")
    out = bytearray(_CONTAINER_PREFIX.pack(_CONTAINER_MAGIC, len(items)))
    for pair_id, payload in items:
        if not 0 <= int(pair_id) < 600:
            raise LadderError("container pair id outside [0,600)")
        out.extend(_CONTAINER_ITEM.pack(int(pair_id), len(payload)))
        out.extend(payload)
    return bytes(out)


def parse_chunk_container(payload: bytes) -> list[tuple[int, bytes]]:
    if len(payload) < _CONTAINER_PREFIX.size:
        raise LadderError("truncated chunk container")
    magic, count = _CONTAINER_PREFIX.unpack_from(payload)
    if magic != _CONTAINER_MAGIC or not 1 <= count <= MAX_CHUNK:
        raise LadderError("invalid chunk container header")
    off = _CONTAINER_PREFIX.size
    rows: list[tuple[int, bytes]] = []
    for _ in range(count):
        if off + _CONTAINER_ITEM.size > len(payload):
            raise LadderError("truncated chunk container item")
        pair_id, size = _CONTAINER_ITEM.unpack_from(payload, off)
        off += _CONTAINER_ITEM.size
        end = off + size
        if end > len(payload):
            raise LadderError("truncated chunk descriptor payload")
        descriptor = payload[off:end]
        parse_plane_description(descriptor)
        rows.append((int(pair_id), descriptor))
        off = end
    if off != len(payload) or len({pair for pair, _ in rows}) != len(rows):
        raise LadderError("chunk container trailing bytes or duplicate pair ids")
    return rows


def _zstd19(value: bytes) -> tuple[bytes, str]:
    try:
        import zstandard as zstd
    except ImportError:
        try:
            proc = subprocess.run(
                ["zstd", "-19", "--stdout", "--quiet"],
                input=value,
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as exc:  # pragma: no cover - environment gate
            raise LadderError("zstandard module or zstd CLI is required for exact rate custody") from exc
        if proc.returncode:
            raise LadderError(f"zstd CLI compression failed with rc={proc.returncode}") from None
        return proc.stdout, "zstd-cli-19"
    return zstd.ZstdCompressor(level=19).compress(value), "python-zstandard-19"


def _zstd_decompress(value: bytes, backend: str) -> bytes:
    if backend == "python-zstandard-19":
        import zstandard as zstd

        return zstd.ZstdDecompressor().decompress(value)
    proc = subprocess.run(
        ["zstd", "--decompress", "--stdout", "--quiet"],
        input=value,
        capture_output=True,
        check=False,
    )
    if proc.returncode:
        raise LadderError(f"zstd CLI decompression failed with rc={proc.returncode}")
    return proc.stdout


def compress_payload(value: bytes) -> dict[str, Any]:
    import brotli

    started = time.monotonic()
    brotli_value = brotli.compress(value, quality=11)
    brotli_seconds = time.monotonic() - started
    started = time.monotonic()
    zstd_value, zstd_backend = _zstd19(value)
    zstd_seconds = time.monotonic() - started
    if brotli.decompress(brotli_value) != value:
        raise LadderError("Brotli-Q11 descriptor parse-back changed bytes")
    if _zstd_decompress(zstd_value, zstd_backend) != value:
        raise LadderError("zstd-19 descriptor parse-back changed bytes")
    return {
        "raw_bytes": len(value),
        "raw_sha256": _sha256_bytes(value),
        "brotli_q11_bytes": len(brotli_value),
        "brotli_q11_sha256": _sha256_bytes(brotli_value),
        "brotli_q11_seconds": brotli_seconds,
        "zstd_19_bytes": len(zstd_value),
        "zstd_19_sha256": _sha256_bytes(zstd_value),
        "zstd_19_seconds": zstd_seconds,
        "zstd_backend": zstd_backend,
        "lossless_parseback": True,
    }


def quantize_u8_plane(target: np.ndarray, levels: int) -> np.ndarray:
    if not 2 <= int(levels) <= 256:
        raise LadderError("quantization levels must be in [2,256]")
    value = np.asarray(target, dtype=np.float64)
    if value.shape != (*SCORER_HW, 3) or not np.isfinite(value).all():
        raise LadderError("quantized target must be one finite scorer plane")
    index = np.rint(np.clip(value, 0.0, 255.0) * (levels - 1) / 255.0)
    return np.rint(index * 255.0 / (levels - 1)).astype(np.uint8)


def _archive_zip_bytes(blob: bytes) -> bytes:
    sink = io.BytesIO()
    info = zipfile.ZipInfo(filename="0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    with zipfile.ZipFile(sink, "w") as archive:
        archive.writestr(info, blob)
    return sink.getvalue()


def _materialize_packet(blob: bytes, packet_dir: Path, byteclose: Any) -> dict[str, Any]:
    packet_dir.mkdir(parents=True, exist_ok=True)
    manifest = byteclose._read_blob_bytes(blob)[0]
    archive = _archive_zip_bytes(blob)
    inflate_py = byteclose._inflate_source_for_manifest(manifest).encode()
    inflate_sh = byteclose._INFLATE_SH.encode()
    _write_once_or_equal(packet_dir / "archive.zip", archive)
    _write_once_or_equal(packet_dir / "inflate.py", inflate_py)
    _write_once_or_equal(packet_dir / "inflate.sh", inflate_sh, mode=0o755)
    return {
        "archive_zip_bytes": len(archive),
        "archive_zip_sha256": _sha256_bytes(archive),
        "blob_bytes": len(blob),
        "blob_sha256": _sha256_bytes(blob),
        "inflate_py_sha256": _sha256_bytes(inflate_py),
        "inflate_sh_sha256": _sha256_bytes(inflate_sh),
    }


def _selected_blob(full_blob: bytes, pairs: Sequence[int], byteclose: Any) -> tuple[bytes, dict[str, Any]]:
    import brotli

    manifest, base_b, code_b, pose_b, lane_b, pcar_b = byteclose._read_blob_bytes(full_blob)
    if pose_b or lane_b is not None or pcar_b is not None or manifest.get("xcodec"):
        raise LadderError("selected witness measurement requires base LVLS1 grammar without optional sections/xcodec")
    quantized = byteclose._wxc.decode_code_quantized(
        brotli.decompress(code_b), manifest["code_shape"], byteclose._wxc.CODE_TRANSFORM_RAW
    )
    selected = np.concatenate([quantized[2 * pair : 2 * pair + 2] for pair in pairs], axis=0)
    selected_raw = byteclose._wxc.encode_code_quantized(selected, byteclose._wxc.CODE_TRANSFORM_RAW)
    selected_brotli = brotli.compress(selected_raw, quality=11)
    subset_manifest = dict(manifest)
    subset_manifest["n_pairs"] = len(pairs)
    subset_manifest["code_shape"] = list(selected.shape)
    subset_json = json.dumps(subset_manifest, separators=(",", ":")).encode()
    blob = byteclose._io_pack(subset_json, base_b, selected_brotli, pose_b or None)
    parsed = byteclose._decode_code(subset_manifest, selected_brotli)
    expected = selected.astype(np.float32) * float(subset_manifest["code_scale"])
    if not np.array_equal(parsed, expected):
        raise LadderError("selected code rows changed during receiver parse-back")
    return blob, {
        "pair_ids": [int(pair) for pair in pairs],
        "source_quantized_code_sha256": _sha256_array(selected),
        "selected_dequant_code_sha256": _sha256_array(parsed),
        "code_scale_unchanged": float(subset_manifest["code_scale"]),
        "requantized": False,
        "selection": "exact int8 code-row slice from the full archive stream",
    }


def prepare_witness(args: argparse.Namespace) -> dict[str, Any]:
    from tools import levelset_byte_close_and_eval as byteclose

    output_root = _durable_output(args.output_root, "output-root", require_ssd=not args.allow_local)
    receipt_path = _durable_output(args.receipt, "receipt")
    checkpoint = args.checkpoint.expanduser().resolve()
    sacred = args.sacred.expanduser().resolve()
    pairs = [int(pair) for pair in args.pairs]
    if len(pairs) < 1 or len(set(pairs)) != len(pairs) or any(pair < 0 or pair >= 600 for pair in pairs):
        raise LadderError("prepare pair set must be unique ids in [0,600)")
    if not checkpoint.is_file() or not sacred.is_dir():
        raise LadderError("checkpoint or sacred run root missing")
    if receipt_path.exists() and not args.resume:
        raise LadderError(f"prepare receipt already exists: {receipt_path}")
    if args.resume and receipt_path.is_file():
        existing = json.loads(receipt_path.read_text())
        raw = Path(existing["selected_decode"]["raw_path"])
        if (
            existing.get("schema") == SCHEMA_PREP
            and raw.is_file()
            and raw.stat().st_size == existing["selected_decode"]["raw_bytes"]
            and _sha256_file(raw) == existing["selected_decode"]["raw_sha256"]
        ):
            print(json.dumps({"receipt": str(receipt_path), "resumed": True}, sort_keys=True))
            return existing
        raise LadderError("prepare resume receipt/raw custody mismatch")

    sacred_before = _tree_snapshot(sacred)
    with np.load(checkpoint, allow_pickle=False) as stored:
        epoch = int(stored["__epoch"].item()) if "__epoch" in stored.files else None
        render_hw = np.asarray(stored["__render_hw"]).astype(int).tolist()
        persisted_so = {
            "freq_across": float(stored["__cfg_freq_across"].item()),
            "freq_along": float(stored["__cfg_freq_along"].item()),
            "n_dir_freqs": int(stored["__cfg_n_dir_freqs"].item()),
        }
    if epoch != 725 or render_hw != list(SCORER_HW):
        raise LadderError(f"rung B requires ep725 384x512 checkpoint, got epoch={epoch}, render={render_hw}")
    params, cfg = byteclose._load_levelset_ckpt(checkpoint.parent, checkpoint.name)
    if int(cfg["n_pairs"]) != 600:
        raise LadderError("byte-closed witness must contain all 600 pair codes")
    so_overrides = {
        "freq_across": persisted_so["freq_across"],
        "freq_along": persisted_so["freq_along"],
        "tau": float(args.so_tau),
        "iters": int(args.so_iters),
    }
    so = byteclose.detect_self_orient(cfg, so_overrides)
    full_blob, breakdown = byteclose.build_levelset_blob(
        params,
        cfg,
        so,
        None,
        cross_tensor_codec=False,
    )
    full_packet = output_root / "full_n600_packet"
    full_packet_info = _materialize_packet(full_blob, full_packet, byteclose)
    subset_blob, selection = _selected_blob(full_blob, pairs, byteclose)
    subset_packet = output_root / "selected_n24_packet"
    subset_packet_info = _materialize_packet(subset_blob, subset_packet, byteclose)
    expected_raw_bytes = 2 * len(pairs) * CAMERA_HW[0] * CAMERA_HW[1] * 3
    storage = byteclose._raw_storage_preflight(subset_packet / "inflated", expected_raw_bytes)
    bit_exact = byteclose.bit_exact_roundtrip_gate(
        subset_packet, subset_blob, min(int(args.bit_exact_pairs), len(pairs)), True
    )
    inflate = byteclose.run_inflate(subset_packet, len(pairs), None)
    raw_path = Path(inflate["raw_path"])
    if raw_path.stat().st_size != expected_raw_bytes:
        raise LadderError("selected receiver raw size mismatch")
    sacred_after = _tree_snapshot(sacred)
    if sacred_after != sacred_before:
        raise LadderError("sacred run metadata changed during byte-close preparation")
    receipt = {
        "schema": SCHEMA_PREP,
        "written_at_utc": datetime.now(UTC).isoformat(),
        "axis": AXIS,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": POINTER,
            "pointer_moved": False,
            "verdict_scope": "ep725 byte-closed frame1 yhat preparation on selected real pairs; no contest axis",
        },
        "checkpoint": {
            "path": str(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256_file(checkpoint),
            "epoch": epoch,
            "weights_arm": "EMA BEST ep725 (explicit; live arm not used)",
            "render_hw": render_hw,
            "n_pairs": int(cfg["n_pairs"]),
        },
        "self_orient": {**so, "source": "persisted checkpoint fields plus explicit decoder iteration/tau"},
        "full_archive": {
            **full_packet_info,
            "path": str(full_packet / "archive.zip"),
            "actual_full_n600_archive": True,
            "bytes_per_pair_derived": full_packet_info["archive_zip_bytes"] / 600.0,
            "rate_term_actual": 25.0 * full_packet_info["archive_zip_bytes"] / RATE_DENOM,
            "breakdown": breakdown,
        },
        "selection": selection,
        "selected_packet": {
            **subset_packet_info,
            "path": str(subset_packet / "archive.zip"),
            "measurement_only": True,
            "not_the_counted_rung_B_rate": True,
        },
        "selected_decode": {
            "raw_path": str(raw_path),
            "raw_bytes": raw_path.stat().st_size,
            "raw_sha256": _sha256_file(raw_path),
            "frame_layout": f"({2 * len(pairs)},874,1164,3) uint8, selected-pair order",
            "receiver_runtime": inflate,
            "bit_exact_roundtrip": bit_exact,
            "storage_preflight": storage,
        },
        "sacred": {
            "root": str(sacred),
            "before": sacred_before,
            "after": sacred_after,
            "unchanged": True,
        },
        "no_hidden_sidecar": {
            "PASS": True,
            "witness_target_source": "frame1 bytes emitted by shipped inflate.py from the selected exact code rows",
            "full_archive_is_counted": True,
            "raw_is_measurement_scratch_on_SSD_not_a_counted_payload": True,
        },
    }
    _atomic_json(receipt_path, receipt)
    print(json.dumps({"receipt": str(receipt_path), "raw": str(raw_path)}, sort_keys=True))
    return receipt


def _load_distortion_net(upstream: Path, threads: int) -> tuple[Any, Any, dict[str, str]]:
    upstream = upstream.resolve()
    modules_path = upstream / "modules.py"
    weights = {
        "modules.py": _sha256_file(modules_path),
        "posenet.safetensors": _sha256_file(upstream / "models/posenet.safetensors"),
        "segnet.safetensors": _sha256_file(upstream / "models/segnet.safetensors"),
    }
    retained: list[str] = []
    for entry in sys.path:
        try:
            if Path(entry or ".").resolve() == upstream:
                continue
        except OSError:
            pass
        retained.append(entry)
    sys.path[:] = [str(upstream), *retained]
    import torch
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    loaded = Path(sys.modules["modules"].__file__).resolve()
    if loaded != modules_path:
        raise LadderError(f"frozen modules imported from {loaded}, expected {modules_path}")
    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    model = DistortionNet().eval().to("cpu")
    model.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, torch, weights


def _distortion_outputs(model: Any, torch: Any, f0: np.ndarray, f1: np.ndarray) -> tuple[Any, Any]:
    pair = torch.from_numpy(np.stack((f0, f1), axis=0)[None]).float()
    with torch.inference_mode():
        return model(pair)


def _distortion_from_outputs(model: Any, candidate: tuple[Any, Any], source: tuple[Any, Any]) -> dict[str, float]:
    pose = model.posenet.compute_distortion(candidate[0], source[0])
    seg = model.segnet.compute_distortion(candidate[1], source[1])
    return {"d_pose": float(pose.item()), "d_seg": float(seg.item())}


def _load_cache(path: Path) -> dict[str, np.memmap]:
    values = {key: stored_npy_memmap(path, key) for key in ("n_pairs", "gt_f0", "gt_f1")}
    if int(np.asarray(values["n_pairs"]).reshape(())) != 600:
        raise LadderError("ladder requires the real n600 cache")
    for key in ("gt_f0", "gt_f1"):
        if values[key].shape != (600, *CAMERA_HW, 3) or values[key].dtype != np.uint8:
            raise LadderError(f"{key} cache geometry/dtype mismatch")
    return values


def _target_for_rung(
    rung: str,
    operator: DisjointResizeOperator,
    source1: np.ndarray,
    witness1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, int]:
    if rung == "source_exact_i32":
        numerators, denominator = operator.apply_numerators(source1)
        return numerators / denominator, numerators, "i32_numerator", 0
    witness_numerators, denominator = operator.apply_numerators(witness1)
    witness_y = witness_numerators / denominator
    if rung == "witness_byteclosed_i32":
        return witness_y, witness_numerators, "i32_numerator", 0
    levels = 64 if rung == "witness_q6_u8" else 16 if rung == "witness_q4_u8" else None
    if levels is None:
        raise LadderError(f"unknown rung {rung}")
    target_u8 = quantize_u8_plane(witness_y, levels)
    target_numerators = target_u8.astype(np.int64) * denominator
    return target_u8.astype(np.float64), target_numerators, "u8_plane", levels


def _rung_row(
    rung: str,
    pair_id: int,
    operator: DisjointResizeOperator,
    source1: np.ndarray,
    witness1: np.ndarray,
    source0: np.ndarray,
    source_outputs: tuple[Any, Any],
    model: Any,
    torch: Any,
    max_nodes: int,
) -> tuple[dict[str, Any], bytes]:
    started = time.monotonic()
    target, target_numerators, encoding, levels = _target_for_rung(rung, operator, source1, witness1)
    denominator = operator.apply_numerators(source1)[1]
    description_plane = target_numerators if encoding == "i32_numerator" else target.astype(np.uint8)
    descriptor = serialize_plane_description(
        description_plane,
        encoding=encoding,
        denominator=denominator,
        quant_levels=levels,
    )
    decoded_plane, decoded_header = parse_plane_description(descriptor)
    if encoding == "i32_numerator":
        decoded_numerators = decoded_plane.astype(np.int64)
        decoded_target = decoded_numerators.astype(np.float64) / denominator
    else:
        decoded_target = decoded_plane.astype(np.float64)
        decoded_numerators = decoded_plane.astype(np.int64) * denominator
    solve_started = time.monotonic()
    solved = operator.solve_uint8(
        decoded_target,
        target_numerators=decoded_numerators,
        max_nodes_per_block=int(max_nodes),
    )
    solve_seconds = time.monotonic() - solve_started
    if not solved.certified_exact or solved.aggregate_status is not BlockSolveStatus.FEASIBLE_EXACT:
        raise LadderError(f"rung={rung} pair={pair_id} did not produce an exact lattice realization")
    actual_numerators, actual_denominator = operator.apply_numerators(solved.frame)
    numerator_delta = actual_numerators.astype(np.int64) - decoded_numerators
    score_started = time.monotonic()
    candidate_outputs = _distortion_outputs(model, torch, source0, solved.frame)
    distortion = _distortion_from_outputs(model, candidate_outputs, source_outputs)
    score_seconds = time.monotonic() - score_started
    source_num, _ = operator.apply_numerators(source1)
    source_y = source_num.astype(np.float64) / denominator
    plane_error = decoded_target - source_y
    diagnostics = solved.diagnostics
    row = {
        "rung": rung,
        "pair_id": int(pair_id),
        "description": decoded_header,
        "target_plane_sha256": _sha256_array(decoded_target),
        "target_integer_numerators_sha256": _sha256_array(decoded_numerators),
        "frame0_policy": "source gt_f0; external diagnostic policy; zero counted bytes; not contest-complete",
        "lattice": {
            "certified_exact": bool(diagnostics.certified_exact),
            "aggregate_status": str(diagnostics.aggregate_status),
            "exact_blocks": int(diagnostics.exact_blocks),
            "heuristic_blocks": int(diagnostics.heuristic_blocks),
            "budget_blocks": int(diagnostics.budget_blocks),
            "proven_affine_infeasible_blocks": int(diagnostics.proven_affine_infeasible_blocks),
            "target_repair_cells": 0,
            "target_numerator_repair_l1": 0,
            "max_abs_realized_numerator_error": int(np.max(np.abs(numerator_delta), initial=0)),
            "nonzero_realized_numerator_error_cells": int(np.count_nonzero(numerator_delta)),
        },
        "shared_plane_error_vs_source": {
            "max_abs": float(np.max(np.abs(plane_error), initial=0.0)),
            "mean_abs": float(np.mean(np.abs(plane_error))),
            "rmse": float(np.sqrt(np.mean(np.square(plane_error)))),
        },
        "distortionnet": distortion,
        "candidate_frame1_sha256": _sha256_array(solved.frame),
        "runtime_seconds": {
            "lattice_solve": solve_seconds,
            "full_distortionnet": score_seconds,
            "pair_rung_total": time.monotonic() - started,
        },
    }
    return row, descriptor


def _scientific_stage(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a non-mutating stage projection with environmental timing removed."""

    projected = json.loads(json.dumps(value, allow_nan=False))
    projected.pop("pair_runtime_seconds", None)
    for rung in projected["rungs"].values():
        rung.pop("runtime_seconds", None)
    return projected


def measure_chunk(args: argparse.Namespace) -> dict[str, Any]:
    output = _durable_output(args.output, "output")
    state = _durable_output(args.state, "state")
    stage_dir = _durable_output(args.stage_dir, "stage-dir")
    witness_receipt_path = args.witness_receipt.expanduser().resolve()
    cache_path = args.cache.expanduser().resolve()
    upstream = args.upstream.expanduser().resolve()
    pairs = [int(pair) for pair in args.pairs]
    if not 1 <= len(pairs) <= MAX_CHUNK or len(set(pairs)) != len(pairs):
        raise LadderError(f"measurement chunk needs 1..{MAX_CHUNK} unique pairs")
    if output.exists():
        raise LadderError(f"chunk receipt already exists: {output}")
    witness_receipt = json.loads(witness_receipt_path.read_text())
    if witness_receipt.get("schema") != SCHEMA_PREP:
        raise LadderError("wrong witness preparation schema")
    selected_pairs = [int(pair) for pair in witness_receipt["selection"]["pair_ids"]]
    if any(pair not in selected_pairs for pair in pairs):
        raise LadderError("chunk pair absent from byte-closed selected receiver output")
    raw_path = Path(witness_receipt["selected_decode"]["raw_path"])
    if (
        not raw_path.is_file()
        or raw_path.stat().st_size != witness_receipt["selected_decode"]["raw_bytes"]
        or _sha256_file(raw_path) != witness_receipt["selected_decode"]["raw_sha256"]
    ):
        raise LadderError("byte-closed selected raw custody mismatch")
    sacred_before = _tree_snapshot(DEFAULT_SACRED)
    config = {
        "schema": STATE_SCHEMA,
        "pairs": pairs,
        "rungs": list(RUNG_ORDER),
        "frame0_policy": "source gt_f0",
        "max_nodes_per_block": int(args.max_nodes_per_block),
        "cpu_threads": int(args.cpu_threads),
        "distortionnet_batch_size": 1,
        "distortionnet_pair_geometry": "one [1,2,874,1164,3] uint8 pair per frozen forward",
        "cache_sha256": _sha256_file(cache_path),
        "witness_receipt_sha256": _sha256_file(witness_receipt_path),
        "witness_raw_sha256": witness_receipt["selected_decode"]["raw_sha256"],
    }
    config_sha = _sha256_bytes(_canonical(config))
    rows: list[dict[str, Any]] = []
    if args.resume:
        loaded = json.loads(state.read_text())
        if loaded.get("schema") != STATE_SCHEMA or loaded.get("config_sha256") != config_sha:
            raise LadderError("resume state config/custody mismatch")
        # Stored scientific rows are never reused.  Stages establish only the completed prefix;
        # every row is re-derived below from frozen inputs and checked against the stage hash.
        completed_prefix = [int(pair) for pair in loaded.get("completed_pairs", [])]
        if completed_prefix != pairs[: len(completed_prefix)]:
            raise LadderError("resume completed pair prefix is not contiguous")
    else:
        if state.exists() or (stage_dir.exists() and any(stage_dir.iterdir())):
            raise LadderError("preserved state/stages exist; pass --resume or choose new paths")
        _atomic_json(
            state,
            {
                "schema": STATE_SCHEMA,
                "config_sha256": config_sha,
                "config": config,
                "completed_pairs": [],
                "next_pair": pairs[0],
            },
        )
    stage_dir.mkdir(parents=True, exist_ok=True)
    fields = _load_cache(cache_path)
    raw = np.memmap(
        raw_path,
        mode="r",
        dtype=np.uint8,
        shape=(2 * len(selected_pairs), *CAMERA_HW, 3),
    )
    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_HW[0], camera_w=CAMERA_HW[1], scorer_h=SCORER_HW[0], scorer_w=SCORER_HW[1]
    )
    model, torch, scorer_hashes = _load_distortion_net(upstream, args.cpu_threads)
    descriptors: dict[str, list[tuple[int, bytes]]] = {rung: [] for rung in RUNG_ORDER}
    for pair_id in pairs:
        pair_started = time.monotonic()
        selected_index = selected_pairs.index(pair_id)
        source0 = np.asarray(fields["gt_f0"][pair_id], dtype=np.uint8).copy()
        source1 = np.asarray(fields["gt_f1"][pair_id], dtype=np.uint8).copy()
        witness1 = np.asarray(raw[2 * selected_index + 1], dtype=np.uint8).copy()
        source_outputs = _distortion_outputs(model, torch, source0, source1)
        witness_outputs = _distortion_outputs(model, torch, source0, witness1)
        direct_witness = _distortion_from_outputs(model, witness_outputs, source_outputs)
        rung_rows: dict[str, Any] = {}
        for rung in RUNG_ORDER:
            rung_row, descriptor = _rung_row(
                rung,
                pair_id,
                operator,
                source1,
                witness1,
                source0,
                source_outputs,
                model,
                torch,
                args.max_nodes_per_block,
            )
            rung_rows[rung] = rung_row
            descriptors[rung].append((pair_id, descriptor))
        stage_payload = {
            "schema": "yhat_rd_ladder_pair_stage.v1",
            "config_sha256": config_sha,
            "pair_id": pair_id,
            "source_f0_sha256": _sha256_array(source0),
            "source_f1_sha256": _sha256_array(source1),
            "byteclosed_witness_f1_sha256": _sha256_array(witness1),
            "byteclosed_witness_direct_distortion": direct_witness,
            "rungs": rung_rows,
            "pair_runtime_seconds": time.monotonic() - pair_started,
        }
        stage_path = stage_dir / f"pair_{pair_id:04d}.json"
        stage_bytes = json.dumps(stage_payload, indent=2, sort_keys=True, allow_nan=False).encode() + b"\n"
        if stage_path.exists():
            preserved_stage = json.loads(stage_path.read_text())
            if _scientific_stage(preserved_stage) != _scientific_stage(stage_payload):
                raise LadderError(f"resume re-derivation differs from preserved stage: {stage_path}")
            # Runtime is environmental metadata rather than scientific state. Preserve the
            # first measurement, while re-deriving every scientific field above.
            stage_payload = preserved_stage
        else:
            _atomic_bytes(stage_path, stage_bytes)
        rows.append(
            {
                **stage_payload,
                "stage": {"path": str(stage_path), "sha256": _sha256_file(stage_path)},
            }
        )
        _atomic_json(
            state,
            {
                "schema": STATE_SCHEMA,
                "config_sha256": config_sha,
                "config": config,
                "completed_pairs": [int(row["pair_id"]) for row in rows],
                "next_pair": pairs[len(rows)] if len(rows) < len(pairs) else None,
            },
        )
        print(f"[yhat-rd] pair={pair_id} complete ({len(rows)}/{len(pairs)})", flush=True)

    chunk_rate: dict[str, Any] = {}
    for rung in RUNG_ORDER:
        container = serialize_chunk_container(descriptors[rung])
        if [pair for pair, _ in parse_chunk_container(container)] != pairs:
            raise LadderError("chunk descriptor parse-back pair order changed")
        chunk_rate[rung] = {
            **compress_payload(container),
            "container_schema": "YHATC1",
            "pair_count": len(pairs),
            "contains_complete_plane_description_per_pair": True,
            "no_hidden_sidecar_bytes": True,
        }
    sacred_after = _tree_snapshot(DEFAULT_SACRED)
    if sacred_after != sacred_before:
        raise LadderError("sacred run tree changed during chunk measurement")
    receipt = {
        "schema": SCHEMA_CHUNK,
        "written_at_utc": datetime.now(UTC).isoformat(),
        "axis": AXIS,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "pointer": POINTER,
            "pointer_moved": False,
            "verdict_scope": "selected real n600-cache pairs; frame1 yhat with source-frame0 policy; full frozen CPU DistortionNet; no contest axis",
        },
        "labels": {
            "MEASURED": [
                "actual Brotli-Q11 and zstd-19 bytes of complete parseable plane containers",
                "frozen full-DistortionNet d_seg and d_pose",
                "lattice exact/infeasible/repair counts and runtimes",
                "actual full-n600 ep725 witness archive bytes",
            ],
            "DERIVED": ["bytes per pair and n600 direct-plane extrapolations"],
            "SPECULATIVE": [],
        },
        "config": config,
        "config_sha256": config_sha,
        "witness_prepare": {
            "path": str(witness_receipt_path),
            "sha256": _sha256_file(witness_receipt_path),
            "full_archive": witness_receipt["full_archive"],
        },
        "pairs": rows,
        "chunk_rate": chunk_rate,
        "resumability": {
            "state": str(state),
            "state_sha256": _sha256_file(state),
            "stage_dir": str(stage_dir),
            "all_pair_stages_preserved": True,
            "stored_scientific_rows_reused": False,
        },
        "custody": {
            "cache": {"path": str(cache_path), "sha256": config["cache_sha256"]},
            "scorer_hashes": scorer_hashes,
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "torch_deterministic_algorithms": True,
            "seed": SEED,
            "sacred_before": sacred_before,
            "sacred_after": sacred_after,
            "sacred_unchanged": True,
        },
        "composition_contract": {
            "pair_ids": pairs,
            "must_be_disjoint": True,
            "minimum_composed_pairs": 24,
            "same_witness_prepare_sha256_required": True,
        },
        "remaining_blockers": [
            "frame0 yhat description and two-plane contest closure",
            "full n600 measured direct-plane rate (not rerun here)",
            "contest-CPU and contest-CUDA exact archive replay",
            "independent MAIN landing review",
        ],
    }
    _atomic_json(output, receipt)
    print(json.dumps({"output": str(output), "pairs": pairs}, sort_keys=True))
    return receipt


def _sum_lattice(rows: Sequence[Mapping[str, Any]], rung: str, key: str) -> int:
    return int(sum(int(row["rungs"][rung]["lattice"][key]) for row in rows))


def compose_chunks(args: argparse.Namespace) -> dict[str, Any]:
    output = _durable_output(args.output, "output")
    csv_path = _durable_output(args.csv, "csv")
    if output.exists() or csv_path.exists():
        raise LadderError("composed table/csv is write-once; choose new outputs")
    paths = [path.expanduser().resolve() for path in args.receipts]
    docs = [json.loads(path.read_text()) for path in paths]
    if not docs or any(doc.get("schema") != SCHEMA_CHUNK for doc in docs):
        raise LadderError("compose inputs must all be yhat ladder chunk v1 receipts")
    witness_hashes = {doc["witness_prepare"]["sha256"] for doc in docs}
    if len(witness_hashes) != 1:
        raise LadderError("chunk receipts do not share one byte-closed witness preparation")
    rows = [row for doc in docs for row in doc["pairs"]]
    pair_ids = [int(row["pair_id"]) for row in rows]
    if len(set(pair_ids)) != len(pair_ids):
        raise LadderError("chunk receipts overlap in pair ids")
    if len(pair_ids) < 24:
        raise LadderError("composed ladder requires at least 24 disjoint real pairs")
    full_archive = docs[0]["witness_prepare"]["full_archive"]
    table_rows: list[dict[str, Any]] = []
    for rung in RUNG_ORDER:
        distortions = [row["rungs"][rung]["distortionnet"] for row in rows]
        plane_errors = [row["rungs"][rung]["shared_plane_error_vs_source"] for row in rows]
        brotli_bytes = int(sum(doc["chunk_rate"][rung]["brotli_q11_bytes"] for doc in docs))
        zstd_bytes = int(sum(doc["chunk_rate"][rung]["zstd_19_bytes"] for doc in docs))
        if rung == "witness_byteclosed_i32":
            primary = {
                "basis": "actual full-n600 ep725 byte-closed witness archive.zip",
                "actual_counted_bytes": int(full_archive["archive_zip_bytes"]),
                "bytes_per_pair": float(full_archive["archive_zip_bytes"] / 600.0),
                "n600_bytes": int(full_archive["archive_zip_bytes"]),
                "n600_byte_status": "MEASURED actual archive",
                "rate_term_n600": float(full_archive["rate_term_actual"]),
            }
        else:
            bytes_per_pair = brotli_bytes / len(pair_ids)
            n600 = bytes_per_pair * 600.0
            primary = {
                "basis": "sum of actual independent chunk Brotli-Q11 complete-plane containers",
                "actual_counted_bytes": brotli_bytes,
                "actual_pair_count": len(pair_ids),
                "bytes_per_pair": bytes_per_pair,
                "n600_bytes": n600,
                "n600_byte_status": "DERIVED linear extrapolation from measured disjoint chunks",
                "rate_term_n600": 25.0 * n600 / RATE_DENOM,
            }
        table_rows.append(
            {
                "rung": rung,
                "description_encoding": rows[0]["rungs"][rung]["description"]["encoding"],
                "quant_levels": rows[0]["rungs"][rung]["description"]["quant_levels"],
                "pair_count": len(pair_ids),
                "frame0_policy": "source gt_f0; external diagnostic policy; not contest-complete",
                "primary_rate": primary,
                "secondary_direct_plane_rate": {
                    "actual_brotli_q11_chunk_bytes": brotli_bytes,
                    "actual_zstd_19_chunk_bytes": zstd_bytes,
                },
                "d_seg_mean": float(np.mean([value["d_seg"] for value in distortions])),
                "d_pose_mean": float(np.mean([value["d_pose"] for value in distortions])),
                "plane_mean_abs_vs_source": float(np.mean([value["mean_abs"] for value in plane_errors])),
                "plane_rmse_vs_source": float(np.mean([value["rmse"] for value in plane_errors])),
                "lattice": {
                    "exact_blocks": _sum_lattice(rows, rung, "exact_blocks"),
                    "heuristic_blocks": _sum_lattice(rows, rung, "heuristic_blocks"),
                    "budget_blocks": _sum_lattice(rows, rung, "budget_blocks"),
                    "proven_affine_infeasible_blocks": _sum_lattice(rows, rung, "proven_affine_infeasible_blocks"),
                    "target_repair_cells": _sum_lattice(rows, rung, "target_repair_cells"),
                    "nonzero_realized_numerator_error_cells": _sum_lattice(
                        rows, rung, "nonzero_realized_numerator_error_cells"
                    ),
                },
                "receiver_runtime_seconds_per_pair": float(
                    np.mean([row["rungs"][rung]["runtime_seconds"]["lattice_solve"] for row in rows])
                ),
            }
        )
    direct_witness = [row["byteclosed_witness_direct_distortion"] for row in rows]
    table = {
        "schema": SCHEMA_TABLE,
        "written_at_utc": datetime.now(UTC).isoformat(),
        "axis": AXIS,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": POINTER,
            "pointer_moved": False,
            "verdict_scope": "n24 frame1-yhat R-D ladder with source-frame0 policy; no contest or full two-plane claim",
        },
        "pair_ids": sorted(pair_ids),
        "pair_count": len(pair_ids),
        "source_receipts": [{"path": str(path), "sha256": _sha256_file(path)} for path in paths],
        "witness_prepare_sha256": next(iter(witness_hashes)),
        "byteclosed_witness_direct_baseline": {
            "d_seg_mean": float(np.mean([value["d_seg"] for value in direct_witness])),
            "d_pose_mean": float(np.mean([value["d_pose"] for value in direct_witness])),
            "role": "interaction baseline before alternate exact lattice realization",
        },
        "rows": table_rows,
        "existing_n600_context_not_rerun": {
            "source": ".omx/research/v10_lattice_rate_verdict_and_composition_20260719.md lines 19-35",
            "measured_formulation": "Brotli-Q11 arbitrary/minimum-norm solved camera frames",
            "mean_bytes_per_frame": 1_700_000,
            "derived_n600_bytes": 1_020_000_000,
            "derived_rate_term": 680.0,
            "verdict": "DEAD",
            "verdict_scope": "direct-solved-camera-frame-as-payload formulation only; compact yhat family OPEN",
        },
        "no_hidden_sidecar_review": {
            "PASS": True,
            "source_and_quantized_rungs": "every counted chunk is a complete parseable plane container; codec parse-back exact",
            "witness_rung": "actual full archive.zip counted; selected raw is measurement-only and hash-bound to shipped inflate.py",
            "frame0_caveat": "source frame0 is an explicit external diagnostic policy, not silently counted or claimed contest-complete",
        },
        "remaining_blockers": [
            "compact frame0 yhat description and joint two-plane packet",
            "full n600 measurement for direct plane descriptions",
            "receiver-integrated descriptor decode plus exact contest CPU/CUDA replay",
            "independent MAIN landing review",
        ],
    }
    _atomic_json(output, table)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = csv_path.with_name(f".{csv_path.name}.tmp-{os.getpid()}")
    try:
        with tmp.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=(
                    "rung",
                    "description_encoding",
                    "quant_levels",
                    "pair_count",
                    "bytes_per_pair",
                    "n600_bytes",
                    "n600_byte_status",
                    "rate_term_n600",
                    "d_seg_mean",
                    "d_pose_mean",
                    "plane_mean_abs_vs_source",
                    "plane_rmse_vs_source",
                    "infeasible_blocks",
                    "repair_cells",
                    "receiver_runtime_seconds_per_pair",
                ),
            )
            writer.writeheader()
            for row in table_rows:
                writer.writerow(
                    {
                        "rung": row["rung"],
                        "description_encoding": row["description_encoding"],
                        "quant_levels": row["quant_levels"],
                        "pair_count": row["pair_count"],
                        "bytes_per_pair": row["primary_rate"]["bytes_per_pair"],
                        "n600_bytes": row["primary_rate"]["n600_bytes"],
                        "n600_byte_status": row["primary_rate"]["n600_byte_status"],
                        "rate_term_n600": row["primary_rate"]["rate_term_n600"],
                        "d_seg_mean": row["d_seg_mean"],
                        "d_pose_mean": row["d_pose_mean"],
                        "plane_mean_abs_vs_source": row["plane_mean_abs_vs_source"],
                        "plane_rmse_vs_source": row["plane_rmse_vs_source"],
                        "infeasible_blocks": row["lattice"]["proven_affine_infeasible_blocks"],
                        "repair_cells": row["lattice"]["target_repair_cells"],
                        "receiver_runtime_seconds_per_pair": row["receiver_runtime_seconds_per_pair"],
                    }
                )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, csv_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    print(json.dumps({"output": str(output), "csv": str(csv_path), "pairs": len(pair_ids)}, sort_keys=True))
    return table


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    prep.add_argument("--sacred", type=Path, default=DEFAULT_SACRED)
    prep.add_argument("--output-root", type=Path, required=True)
    prep.add_argument("--receipt", type=Path, required=True)
    prep.add_argument("--pairs", type=int, nargs="+", default=list(DEFAULT_PAIRS))
    prep.add_argument("--so-tau", type=float, default=4.0)
    prep.add_argument("--so-iters", type=int, default=4)
    prep.add_argument("--bit-exact-pairs", type=int, default=2)
    prep.add_argument("--allow-local", action="store_true")
    prep.add_argument("--resume", action="store_true")

    measure = sub.add_parser("measure")
    measure.add_argument("--witness-receipt", type=Path, required=True)
    measure.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    measure.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    measure.add_argument("--pairs", type=int, nargs="+", required=True)
    measure.add_argument("--output", type=Path, required=True)
    measure.add_argument("--state", type=Path, required=True)
    measure.add_argument("--stage-dir", type=Path, required=True)
    measure.add_argument("--max-nodes-per-block", type=int, default=4096)
    measure.add_argument("--cpu-threads", type=int, default=4)
    measure.add_argument("--resume", action="store_true")

    compose = sub.add_parser("compose")
    compose.add_argument("--receipts", type=Path, nargs="+", required=True)
    compose.add_argument("--output", type=Path, required=True)
    compose.add_argument("--csv", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "prepare":
        prepare_witness(args)
    elif args.command == "measure":
        measure_chunk(args)
    else:
        compose_chunks(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
