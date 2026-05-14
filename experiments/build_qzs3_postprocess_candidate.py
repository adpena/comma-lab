#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a QZS3/QP1 archive plus a counted PR65-style qpost sidecar.

This is a build-only transform.  It copies the base top-submission-style
member ``p`` byte-for-byte from a source archive, extracts the public PR65
postprocess streams from its compact ``x`` member, and emits a deterministic
archive containing:

    p
    qpost.bin

The resulting archive has no score claim until exact CUDA auth eval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import zipfile
from pathlib import Path
from typing import Any

import brotli
import numpy as np


FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
QPOST_MAGIC = b"QPS1"
QPOST_STREAM_NAMES = (
    "post",
    "shift",
    "frac",
    "frac2",
    "frac3",
    "bias",
    "region",
    "randmulti",
)
ORIGINAL_VIDEO_BYTES = 37_545_489


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_single_level_members(archive: Path) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    with zipfile.ZipFile(archive, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            path = Path(name)
            if not name or name.startswith("/") or ".." in path.parts or len(path.parts) != 1:
                raise ValueError(f"unsafe archive member path: {name!r}")
            if name in members:
                raise ValueError(f"duplicate archive member: {name!r}")
            members[name] = zf.read(info)
    return members


def extract_pr65_qpost_streams(pr65_archive: Path) -> dict[str, bytes]:
    members = _safe_single_level_members(pr65_archive)
    if "x" not in members:
        raise ValueError(f"{pr65_archive} missing compact PR65 member 'x'")
    raw = members["x"]
    if len(raw) < 30:
        raise ValueError("PR65 compact member too short for v4 header")
    lens = [int.from_bytes(raw[i:i + 3], "little") for i in range(0, 30, 3)]
    l_mask, l_model, l_pose, l_post, l_shift, l_frac, l_frac2, l_frac3, l_bias, l_region = lens
    if not (l_mask > 1000 and l_model > 1000 and l_pose > 100):
        raise ValueError(f"PR65 compact member has implausible core lengths: {lens[:3]}")
    pos = 30 + l_mask + l_model + l_pose
    lengths = [l_post, l_shift, l_frac, l_frac2, l_frac3, l_bias, l_region]
    chunks: dict[str, bytes] = {}
    for name, n in zip(QPOST_STREAM_NAMES[:-1], lengths):
        if n <= 0:
            raise ValueError(f"PR65 stream {name} has invalid length {n}")
        chunks[name] = raw[pos:pos + n]
        pos += n
    chunks["randmulti"] = raw[pos:]
    if not chunks["randmulti"]:
        raise ValueError("PR65 compact member has empty randmulti tail")
    return chunks


def encode_qpost(streams: dict[str, bytes]) -> bytes:
    missing = [name for name in QPOST_STREAM_NAMES if name not in streams]
    if missing:
        raise ValueError(f"missing qpost streams: {missing}")
    lengths = [len(streams[name]) for name in QPOST_STREAM_NAMES]
    return QPOST_MAGIC + struct.pack("<" + "I" * len(QPOST_STREAM_NAMES), *lengths) + b"".join(
        streams[name] for name in QPOST_STREAM_NAMES
    )


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _parse_stream_selection(selection: str | None) -> tuple[str, ...]:
    if selection is None or selection.strip().lower() in {"all", "*"}:
        return QPOST_STREAM_NAMES
    raw_names = [part.strip() for part in selection.split(",") if part.strip()]
    if not raw_names:
        raise ValueError("--include-streams must name at least one stream, or use all")
    unknown = sorted(set(raw_names) - set(QPOST_STREAM_NAMES))
    if unknown:
        raise ValueError(f"unknown qpost stream(s): {unknown}; valid={list(QPOST_STREAM_NAMES)}")
    seen: set[str] = set()
    ordered = []
    for name in raw_names:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)


def _parse_pair_indices(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or not raw.strip():
        return None
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 0 or value >= 600:
            raise ValueError(f"pair index out of 600-pair range: {value}")
        values.append(value)
    if not values:
        raise ValueError("--pair-indices must contain at least one index")
    return tuple(sorted(set(values)))


def _pair_mask(pair_indices: tuple[int, ...]) -> np.ndarray:
    mask = np.zeros(600, dtype=bool)
    mask[list(pair_indices)] = True
    return mask


def _filter_post_stream(blob: bytes, keep: np.ndarray) -> bytes:
    raw = brotli.decompress(blob)
    if raw[:4] == b"PCD1":
        out = bytearray(raw[:5])
        pos = 5
        stage_count = raw[4]
        for _ in range(stage_count):
            stage_id = raw[pos]
            n = struct.unpack_from("<H", raw, pos + 1)[0]
            pos += 3
            choices = bytearray(raw[pos:pos + n])
            pos += n
            if n != 600:
                raise ValueError(f"PCD1 post stage has {n} choices, expected 600")
            for idx, use_pair in enumerate(keep):
                if not use_pair:
                    choices[idx] = 0
            out.extend(bytes([stage_id]))
            out.extend(struct.pack("<H", n))
            out.extend(choices)
        if pos != len(raw):
            raise ValueError("PCD1 post stream has trailing bytes")
        return brotli.compress(bytes(out), quality=11)

    if len(raw) % 600 != 0:
        raise ValueError("headerless post stream length is not a multiple of 600")
    stage_count = len(raw) // 600
    if stage_count not in (3, 4):
        raise ValueError(f"unsupported headerless post stage count: {stage_count}")
    arr = np.frombuffer(raw, dtype=np.uint8).copy().reshape(stage_count, 600)
    arr[:, ~keep] = 0
    return brotli.compress(arr.tobytes(), quality=11)


def _vlq_indices_values(raw: bytes, pos: int, count: int) -> tuple[list[int], np.ndarray, int]:
    idx = -1
    indices = []
    for _ in range(count):
        acc = 0
        shift = 0
        while True:
            by = raw[pos]
            pos += 1
            acc |= (by & 127) << shift
            if by & 128:
                shift += 7
            else:
                break
        idx += acc + 1
        indices.append(idx)
    vals = np.frombuffer(raw, dtype=np.uint8, count=count, offset=pos).astype(np.uint8)
    pos += count
    return indices, vals, pos


def _dense_or_delta_to_array(
    blob: bytes,
    *,
    magic_full: bytes,
    magic_delta: bytes,
    default: int,
    center: int | None = None,
) -> np.ndarray:
    raw = brotli.decompress(blob)
    magic = raw[:3]
    if magic == magic_full:
        arr = np.frombuffer(raw, dtype=np.uint8, offset=3).copy()
    elif magic == magic_delta:
        d = np.frombuffer(raw, dtype=np.uint8, offset=3).astype(np.int64)
        arr = np.where(d == 0, default, d - 1).astype(np.uint8)
    elif center is not None and magic == b"BV1":
        count = int.from_bytes(raw[3:5], "little")
        indices, vals, pos = _vlq_indices_values(raw, 5, count)
        if pos != len(raw):
            raise ValueError("BV1 dense/delta stream has trailing bytes")
        arr = np.full(600, center, dtype=np.uint8)
        for idx, val in zip(indices, vals):
            arr[idx] = np.uint8(int(val) - 1)
    else:
        raise ValueError(f"unsupported dense/delta stream magic {magic!r}")
    if arr.shape != (600,):
        raise ValueError(f"dense/delta stream decoded to shape {arr.shape}, expected (600,)")
    return arr


def _frac_to_array(blob: bytes) -> np.ndarray:
    raw = brotli.decompress(blob)
    magic = raw[:3]
    if magic == b"FH1":
        arr = np.frombuffer(raw, dtype=np.uint8, offset=3).copy()
    elif magic == b"FV1":
        count = int.from_bytes(raw[3:5], "little")
        indices, vals, pos = _vlq_indices_values(raw, 5, count)
        if pos != len(raw):
            raise ValueError("FV1 frac stream has trailing bytes")
        arr = np.full(600, 4, dtype=np.uint8)
        for idx, val in zip(indices, vals):
            arr[idx] = np.uint8(int(val) - 1)
    else:
        raise ValueError(f"unsupported frac stream magic {magic!r}")
    if arr.shape != (600,):
        raise ValueError(f"frac stream decoded to shape {arr.shape}, expected (600,)")
    return arr


def _region_to_array(blob: bytes) -> np.ndarray:
    raw = brotli.decompress(blob)
    magic = raw[:3]
    if magic == b"RH1":
        arr = np.frombuffer(raw, dtype=np.uint8, offset=3).copy()
    elif magic == b"RD1":
        d = np.frombuffer(raw, dtype=np.uint8, offset=3).astype(np.int64)
        arr = np.where(d == 0, 0, d - 1).astype(np.uint8)
    elif magic == b"RV1":
        count = int.from_bytes(raw[3:5], "little")
        indices, vals, pos = _vlq_indices_values(raw, 5, count)
        if pos != len(raw):
            raise ValueError("RV1 region stream has trailing bytes")
        arr = np.zeros(600, dtype=np.uint8)
        for idx, val in zip(indices, vals):
            arr[idx] = np.uint8(int(val) - 1)
    else:
        raise ValueError(f"unsupported region stream magic {magic!r}")
    if arr.shape != (600,):
        raise ValueError(f"region stream decoded to shape {arr.shape}, expected (600,)")
    return arr


def _filter_dense_stream(blob: bytes, keep: np.ndarray, *, name: str) -> bytes:
    if name == "shift":
        arr = _dense_or_delta_to_array(blob, magic_full=b"SH4", magic_delta=b"SD4", default=40)
        arr[~keep] = 40
        raw = b"SH4" + arr.tobytes()
    elif name == "frac":
        arr = _frac_to_array(blob)
        arr[~keep] = 4
        raw = b"FH1" + arr.tobytes()
    elif name == "frac2":
        arr = _dense_or_delta_to_array(blob, magic_full=b"FH2", magic_delta=b"FD2", default=4)
        arr[~keep] = 4
        raw = b"FH2" + arr.tobytes()
    elif name == "frac3":
        arr = _dense_or_delta_to_array(blob, magic_full=b"FH3", magic_delta=b"FD3", default=4)
        arr[~keep] = 4
        raw = b"FH3" + arr.tobytes()
    elif name == "bias":
        arr = _dense_or_delta_to_array(blob, magic_full=b"BH1", magic_delta=b"BD1", default=13, center=13)
        arr[~keep] = 13
        raw = b"BH1" + arr.tobytes()
    elif name == "region":
        arr = _region_to_array(blob)
        arr[~keep] = 0
        raw = b"RH1" + arr.tobytes()
    else:
        raise ValueError(f"pair filtering is not implemented for qpost stream {name!r}")
    return brotli.compress(raw, quality=11)


def filter_qpost_streams_to_pairs(
    streams: dict[str, bytes],
    pair_indices: tuple[int, ...],
    *,
    include_streams: tuple[str, ...],
) -> dict[str, bytes]:
    keep = _pair_mask(pair_indices)
    filtered: dict[str, bytes] = {}
    include_set = set(include_streams)
    for name in QPOST_STREAM_NAMES:
        if name not in include_set:
            filtered[name] = b""
            continue
        if name == "randmulti":
            raise ValueError(
                "pair-filtered randmulti is intentionally unsupported; omit "
                "randmulti or add a reviewed sparse encoder first"
            )
        if name == "post":
            filtered[name] = _filter_post_stream(streams[name], keep)
        else:
            filtered[name] = _filter_dense_stream(streams[name], keep, name=name)
    return filtered


def build_candidate(
    source_archive: Path,
    pr65_archive: Path,
    output_archive: Path,
    *,
    include_streams: tuple[str, ...] = QPOST_STREAM_NAMES,
    pair_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    source_archive = source_archive.resolve()
    pr65_archive = pr65_archive.resolve()
    output_archive = output_archive.resolve()
    source_members = _safe_single_level_members(source_archive)
    if "p" not in source_members:
        raise ValueError(f"{source_archive} missing base top-submission member 'p'")
    qpost_streams_all = extract_pr65_qpost_streams(pr65_archive)
    include_set = set(include_streams)
    if pair_indices is None:
        qpost_streams = {
            name: qpost_streams_all[name] if name in include_set else b""
            for name in QPOST_STREAM_NAMES
        }
        pair_filter = None
    else:
        qpost_streams = filter_qpost_streams_to_pairs(
            qpost_streams_all,
            pair_indices,
            include_streams=include_streams,
        )
        pair_filter = {
            "pair_indices": list(pair_indices),
            "pair_count": len(pair_indices),
            "nonselected_pairs_default_to_identity": True,
            "randmulti_supported": False,
        }
    qpost = encode_qpost(qpost_streams)
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_archive, "w") as zf:
        zf.writestr(_zip_info("p"), source_members["p"])
        zf.writestr(_zip_info("qpost.bin"), qpost)

    output_bytes = output_archive.stat().st_size
    source_bytes = source_archive.stat().st_size
    return {
        "schema_version": 1,
        "tool": "experiments/build_qzs3_postprocess_candidate.py",
        "score_claim": False,
        "evidence_grade": "empirical_byte_only_until_cuda_auth_eval",
        "source_archive": str(source_archive),
        "source_archive_bytes": source_bytes,
        "source_archive_sha256": _sha256_path(source_archive),
        "pr65_archive": str(pr65_archive),
        "pr65_archive_sha256": _sha256_path(pr65_archive),
        "output_archive": str(output_archive),
        "output_archive_bytes": output_bytes,
        "output_archive_sha256": _sha256_path(output_archive),
        "archive_byte_delta": output_bytes - source_bytes,
        "formula_rate_score_delta": 25.0 * float(output_bytes - source_bytes) / ORIGINAL_VIDEO_BYTES,
        "include_streams": list(include_streams),
        "omitted_streams": [name for name in QPOST_STREAM_NAMES if name not in include_set],
        "pair_filter": pair_filter,
        "members": {
            "p": {"bytes": len(source_members["p"]), "sha256": _sha256_bytes(source_members["p"])},
            "qpost.bin": {"bytes": len(qpost), "sha256": _sha256_bytes(qpost)},
        },
        "qpost_streams": {
            name: {
                "active": name in include_set,
                "bytes": len(qpost_streams[name]),
                "original_bytes": len(qpost_streams_all[name]),
                "sha256": _sha256_bytes(qpost_streams[name]),
                "original_sha256": _sha256_bytes(qpost_streams_all[name]),
            }
            for name in QPOST_STREAM_NAMES
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--pr65-archive", type=Path, default=Path("reports/raw/leaderboard_intel_20260501/pr65_archive.zip"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-archive", type=Path, default=None)
    parser.add_argument(
        "--include-streams",
        default="all",
        help="Comma-separated qpost streams to keep; omitted streams are encoded as zero length. Use 'all' for the full PR65 sidecar.",
    )
    parser.add_argument(
        "--pair-indices",
        default=None,
        help="Optional comma-separated 0-based contest pair indices. When set, active qpost streams are re-encoded so non-selected pairs decode to no-op defaults.",
    )
    args = parser.parse_args(argv)
    output_archive = args.output_archive or (args.output_dir / "archive.zip")
    meta = build_candidate(
        args.source_archive,
        args.pr65_archive,
        output_archive,
        include_streams=_parse_stream_selection(args.include_streams),
        pair_indices=_parse_pair_indices(args.pair_indices),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "build_provenance.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
