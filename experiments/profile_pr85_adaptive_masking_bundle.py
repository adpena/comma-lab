#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile the public PR85 adaptive masking bundle without executing inflate.

This is a custody and reverse-engineering helper only.  It parses the single
ZIP member ``x`` used by PR85's public attachment and emits segment byte counts,
SHA-256s, and optional Brotli-decompressed magic.  It does not claim score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from zipfile import ZipFile

try:
    import brotli
except ImportError:  # pragma: no cover - exercised only in minimal envs
    brotli = None


SEGMENT_ORDER = (
    "mask",
    "model",
    "pose",
    "post",
    "shift",
    "frac",
    "frac2",
    "frac3",
    "bias",
    "region",
    "randmulti",
)
FIXED_V5_BIAS_BYTES = 223
FIXED_V5_REGION_BYTES = 273


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _u24le(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 3], "little")


def _load_single_member(archive: Path, member: str = "x") -> tuple[dict[str, object], bytes]:
    with ZipFile(archive) as zf:
        infos = zf.infolist()
        if len(infos) != 1:
            raise ValueError(f"expected one ZIP member, found {len(infos)}")
        info = infos[0]
        if info.filename != member:
            raise ValueError(f"expected ZIP member {member!r}, found {info.filename!r}")
        data = zf.read(info.filename)
    zip_info = {
        "member_name": info.filename,
        "member_file_size": info.file_size,
        "member_compress_size": info.compress_size,
        "member_crc": info.CRC,
        "member_sha256": _sha256(data),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": _sha256(archive.read_bytes()),
    }
    return zip_info, data


def parse_pr85_v5_micro_bundle(raw: bytes) -> dict[str, bytes]:
    """Parse PR85's fixed v5 micro-header bundle.

    The first 24 bytes are eight little-endian 24-bit lengths:
    mask, model, pose, post, shift, frac, frac2, frac3.  Bias and region have
    fixed public-runtime lengths and the remaining tail is randmulti.
    """

    if len(raw) < 24:
        raise ValueError("PR85 bundle is too short for v5 micro header")
    lengths = {
        "mask": _u24le(raw, 0),
        "model": _u24le(raw, 3),
        "pose": _u24le(raw, 6),
        "post": _u24le(raw, 9),
        "shift": _u24le(raw, 12),
        "frac": _u24le(raw, 15),
        "frac2": _u24le(raw, 18),
        "frac3": _u24le(raw, 21),
        "bias": FIXED_V5_BIAS_BYTES,
        "region": FIXED_V5_REGION_BYTES,
    }
    pos = 24
    segments: dict[str, bytes] = {}
    for name in SEGMENT_ORDER[:-1]:
        n = lengths[name]
        if n <= 0:
            raise ValueError(f"invalid nonpositive PR85 segment length for {name}: {n}")
        end = pos + n
        if end > len(raw):
            raise ValueError(f"truncated PR85 segment {name}")
        segments[name] = raw[pos:end]
        pos = end
    if pos >= len(raw):
        raise ValueError("PR85 bundle is missing randmulti tail")
    segments["randmulti"] = raw[pos:]
    return segments


def _brotli_probe(data: bytes, max_magic: int) -> dict[str, object]:
    if brotli is None:
        return {"brotli_available": False}
    try:
        decoded = brotli.decompress(data)
    except brotli.error as exc:
        return {"brotli_available": True, "brotli_ok": False, "brotli_error": str(exc)}
    return {
        "brotli_available": True,
        "brotli_ok": True,
        "decoded_bytes": len(decoded),
        "decoded_sha256": _sha256(decoded),
        "decoded_magic_hex": decoded[:max_magic].hex(),
        "decoded_magic_ascii": decoded[:max_magic].decode("ascii", errors="replace"),
    }


def profile_archive(archive: Path, *, member: str = "x", max_magic: int = 8) -> dict[str, object]:
    zip_info, raw = _load_single_member(archive, member=member)
    segments = parse_pr85_v5_micro_bundle(raw)
    segment_rows = []
    for name in SEGMENT_ORDER:
        data = segments[name]
        row: dict[str, object] = {
            "name": name,
            "bytes": len(data),
            "sha256": _sha256(data),
            "magic_hex": data[:max_magic].hex(),
            "magic_ascii": data[:max_magic].decode("ascii", errors="replace"),
        }
        if name != "mask":
            row.update(_brotli_probe(data, max_magic=max_magic))
        segment_rows.append(row)
    return {
        "schema_version": 1,
        "tool": "experiments/profile_pr85_adaptive_masking_bundle.py",
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "score_claim": False,
        "evidence_grade": "external_static_profile",
        "archive": zip_info,
        "bundle_format": "pr85_v5_micro_24bit_lengths_fixed_bias_region",
        "header_bytes": 24,
        "segments": segment_rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--member", default="x")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--max-magic", type=int, default=8)
    args = parser.parse_args(argv)

    payload = profile_archive(args.archive, member=args.member, max_magic=args.max_magic)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
