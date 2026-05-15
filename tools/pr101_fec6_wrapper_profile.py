#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile the PR101/FEC6 FP11 wrapper and fixed-Huffman selector bytes.

This is a no-network forensic helper.  It does not inflate frames, dispatch
evals, or claim score movement; it only reconstructs deterministic byte
offsets and selector-code facts from an existing archive.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FEC6_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
)
DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/public_pr101_hnerv_ft_microcodec_intake_20260504_codex/archive.zip"
)

SCHEMA = "pr101_fec6_wrapper_profile.v1"
OUTER_MAGIC = b"FP11"
SELECTOR_MAGIC = b"FEC6"
MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
MAX_MEMBER_BYTES = 10 * 1024 * 1024
FEC6_FIXED_K16_MODE_IDS = (
    "none",
    "frame0_blue_chroma_amp_1",
    "frame0_blue_chroma_amp_3",
    "frame0_luma_bias_+1",
    "frame0_luma_bias_-1",
    "frame0_luma_bias_-2",
    "frame0_luma_bias_-4",
    "frame0_rgb_bias_m2_p1_p1",
    "frame0_rgb_bias_m4_p2_p2",
    "frame0_rgb_bias_p0_m1_p1",
    "frame0_rgb_bias_p0_m2_p2",
    "frame0_rgb_bias_p0_p1_m1",
    "frame0_rgb_bias_p0_p2_m2",
    "frame0_rgb_bias_p2_m1_m1",
    "frame0_rgb_bias_p4_m2_m2",
    "frame0_roll_dx+0_dy+1",
)
FEC6_FIXED_K16_CODE_BITS = (
    "00",
    "1100",
    "01",
    "111010",
    "11010",
    "111011",
    "111100",
    "100",
    "111101",
    "11011",
    "1111110",
    "111110",
    "11111110",
    "101",
    "11100",
    "11111111",
)
FEC6_FIXED_K16_DECODE = {
    bits: code for code, bits in enumerate(FEC6_FIXED_K16_CODE_BITS)
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _zip_compression_name(compress_type: int) -> str:
    if compress_type == zipfile.ZIP_STORED:
        return "stored"
    if compress_type == zipfile.ZIP_DEFLATED:
        return "deflated"
    if compress_type == zipfile.ZIP_BZIP2:
        return "bzip2"
    if compress_type == zipfile.ZIP_LZMA:
        return "lzma"
    return f"unknown_{compress_type}"


def _read_single_member_payload(archive: Path) -> tuple[dict[str, Any], bytes]:
    archive_size = archive.stat().st_size
    if archive_size > MAX_ARCHIVE_BYTES:
        raise ValueError(
            f"archive too large for PR101/FEC6 forensic profiler: {archive_size} > {MAX_ARCHIVE_BYTES}"
        )
    with zipfile.ZipFile(archive) as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        if len(infos) != 1:
            raise ValueError(f"expected one archive member in {archive}, found {len(infos)}")
        info = infos[0]
        member_name = info.filename
        if member_name.startswith("/") or ".." in Path(member_name).parts:
            raise ValueError(f"unsafe archive member name: {member_name!r}")
        if info.file_size > MAX_MEMBER_BYTES:
            raise ValueError(
                f"archive member too large for PR101/FEC6 forensic profiler: "
                f"{info.file_size} > {MAX_MEMBER_BYTES}"
            )
        payload = zf.read(member_name)
    meta = {
        "member_name": member_name,
        "member_bytes": len(payload),
        "member_sha256": sha256_bytes(payload),
        "zip_compression": _zip_compression_name(info.compress_type),
        "zip_compress_size": info.compress_size,
        "zip_file_size": info.file_size,
        "zip_crc32_hex": f"{info.CRC:08x}",
        "zip_header_offset": info.header_offset,
    }
    return meta, payload


def _shannon_floor_bytes(counts: Mapping[int, int]) -> int:
    total = sum(counts.values())
    if total <= 0:
        return 0
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    return math.ceil(entropy * total / 8.0)


def decode_fec6_fixed_huffman_codes(payload: bytes, *, n_pairs: int) -> tuple[list[int], int]:
    if n_pairs < 0:
        raise ValueError(f"n_pairs must be non-negative, got {n_pairs}")
    codes: list[int] = []
    prefix = ""
    bit_pos = 0
    max_bits = len(payload) * 8
    while len(codes) < n_pairs:
        if bit_pos >= max_bits:
            raise ValueError("FEC6 compact selector bitstream truncated")
        bit = (payload[bit_pos // 8] >> (7 - (bit_pos % 8))) & 1
        bit_pos += 1
        prefix += "1" if bit else "0"
        code = FEC6_FIXED_K16_DECODE.get(prefix)
        if code is not None:
            codes.append(int(code))
            prefix = ""
            continue
        if len(prefix) > 8:
            raise ValueError("FEC6 compact selector contains invalid prefix code")
    if prefix:
        raise ValueError("FEC6 compact selector ended mid-symbol")
    for trailing in range(bit_pos, max_bits):
        if (payload[trailing // 8] >> (7 - (trailing % 8))) & 1:
            raise ValueError("FEC6 compact selector has non-zero padding bits")
    return codes, bit_pos


def parse_fec6_selector_payload(selector_payload: bytes) -> dict[str, Any]:
    if len(selector_payload) < 6:
        raise ValueError("FEC6 selector payload truncated before header")
    if selector_payload[:4] != SELECTOR_MAGIC:
        raise ValueError(f"expected FEC6 selector payload, got {selector_payload[:4]!r}")
    n_pairs = struct.unpack_from("<H", selector_payload, 4)[0]
    index_payload = selector_payload[6:]
    codes, used_bits = decode_fec6_fixed_huffman_codes(index_payload, n_pairs=n_pairs)
    expected_index_bytes = math.ceil(used_bits / 8)
    if len(index_payload) != expected_index_bytes:
        raise ValueError(
            f"FEC6 compact selector has trailing zero bytes: "
            f"index_bytes={len(index_payload)} expected={expected_index_bytes}"
        )
    counts = Counter(codes)
    code_bits_total = sum(len(FEC6_FIXED_K16_CODE_BITS[code]) for code in codes)
    if code_bits_total != used_bits:
        raise ValueError(
            f"FEC6 decoded bit count mismatch: code_bits={code_bits_total} used_bits={used_bits}"
        )
    return {
        "magic": "FEC6",
        "payload_bytes": len(selector_payload),
        "payload_sha256": sha256_bytes(selector_payload),
        "selector_header_bytes": 6,
        "n_pairs": n_pairs,
        "palette_size": len(FEC6_FIXED_K16_MODE_IDS),
        "palette_mode_ids": list(FEC6_FIXED_K16_MODE_IDS),
        "selector_index_bytes": len(index_payload),
        "selector_code_bits_total": code_bits_total,
        "selector_avg_bits_per_pair": code_bits_total / n_pairs if n_pairs else 0.0,
        "zero_padding_bits": len(index_payload) * 8 - used_bits,
        "entropy_floor_bytes": _shannon_floor_bytes(counts),
        "gap_to_entropy_floor_bytes": len(index_payload) - _shannon_floor_bytes(counts),
        "selector_index_gap_to_entropy_floor_bytes": len(index_payload) - _shannon_floor_bytes(counts),
        "selector_payload_gap_to_entropy_floor_bytes": len(selector_payload) - _shannon_floor_bytes(counts),
        "code_histogram": {str(code): int(count) for code, count in sorted(counts.items())},
        "mode_histogram": {
            FEC6_FIXED_K16_MODE_IDS[code]: int(count) for code, count in sorted(counts.items())
        },
        "code_stream_prefix_bits": "".join(
            FEC6_FIXED_K16_CODE_BITS[code] for code in codes[:16]
        ),
        "first_32_codes": [int(code) for code in codes[:32]],
    }


def parse_fp11_wrapper_payload(wrapper_payload: bytes) -> dict[str, Any]:
    if len(wrapper_payload) < 10:
        raise ValueError("PR101 FEC6 wrapper truncated before header")
    if wrapper_payload[:4] != OUTER_MAGIC:
        raise ValueError(f"PR101 FEC6 wrapper magic mismatch: {wrapper_payload[:4]!r}")
    source_len = struct.unpack_from("<I", wrapper_payload, 4)[0]
    source_start = 8
    source_end = source_start + source_len
    if source_end > len(wrapper_payload):
        raise ValueError("PR101 FEC6 wrapper truncated in source payload")
    selector_len_offset = source_end
    if selector_len_offset + 2 > len(wrapper_payload):
        raise ValueError("PR101 FEC6 wrapper truncated before selector length")
    selector_len = struct.unpack_from("<H", wrapper_payload, selector_len_offset)[0]
    selector_start = selector_len_offset + 2
    selector_end = selector_start + selector_len
    if selector_end > len(wrapper_payload):
        raise ValueError("PR101 FEC6 wrapper truncated in selector payload")
    if selector_end != len(wrapper_payload):
        raise ValueError(
            f"PR101 FEC6 wrapper has trailing bytes: selector_end={selector_end} total={len(wrapper_payload)}"
        )
    source_payload = wrapper_payload[source_start:source_end]
    selector_payload = wrapper_payload[selector_start:selector_end]
    selector_profile = parse_fec6_selector_payload(selector_payload)
    return {
        "outer_magic": "FP11",
        "wrapper_payload_bytes": len(wrapper_payload),
        "wrapper_overhead_bytes": 10,
        "source_payload": {
            "offset": source_start,
            "bytes": len(source_payload),
            "sha256": sha256_bytes(source_payload),
        },
        "selector_length_field": {
            "offset": selector_len_offset,
            "bytes": 2,
            "value": selector_len,
        },
        "selector_payload": {
            "offset": selector_start,
            "end_offset": selector_end,
            **selector_profile,
        },
        "runtime_parser_offsets": {
            "magic": [0, 4],
            "source_len_u32le": [4, 8],
            "source_payload": [source_start, source_end],
            "selector_len_u16le": [selector_len_offset, selector_len_offset + 2],
            "selector_payload": [selector_start, selector_end],
        },
    }


def profile_archive(archive: Path, *, source_archive: Path | None = DEFAULT_SOURCE_ARCHIVE) -> dict[str, Any]:
    archive = Path(archive)
    member_meta, wrapper_payload = _read_single_member_payload(archive)
    wrapper = parse_fp11_wrapper_payload(wrapper_payload)
    profile: dict[str, Any] = {
        "schema": SCHEMA,
        "score_claim": False,
        "score_claim_valid": False,
        "score_axis": None,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "research_only": True,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "archive": {
            "path": _rel(archive),
            "bytes": archive.stat().st_size,
            "sha256": sha256_bytes(archive.read_bytes()),
            **member_meta,
        },
        "wrapper": wrapper,
        "repeatability_gates": {
            "single_zip_member": True,
            "archive_member_stored": member_meta["zip_compression"] == "stored",
            "selector_packed_in_archive": True,
            "source_payload_reference_match": None,
            "full_frame_parity_claim": False,
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
        },
    }
    if source_archive is not None:
        source_archive = Path(source_archive)
        source_meta, source_payload = _read_single_member_payload(source_archive)
        source_match = sha256_bytes(source_payload) == wrapper["source_payload"]["sha256"]
        profile["source_reference"] = {
            "path": _rel(source_archive),
            "archive_bytes": source_archive.stat().st_size,
            "archive_sha256": sha256_bytes(source_archive.read_bytes()),
            **source_meta,
            "source_payload_matches_wrapper": source_match,
            "source_payload_bytes_match_wrapper": len(source_payload)
            == wrapper["source_payload"]["bytes"],
            "member_name_matches_wrapper_archive": source_meta["member_name"]
            == member_meta["member_name"],
        }
        profile["repeatability_gates"]["source_payload_reference_match"] = source_match
    return profile


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_FEC6_ARCHIVE)
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=DEFAULT_SOURCE_ARCHIVE,
        help="Optional source PR101 archive used to verify the FP11 source payload.",
    )
    parser.add_argument("--no-source-archive", action="store_true")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    source_archive = None if args.no_source_archive else args.source_archive
    profile = profile_archive(Path(args.archive), source_archive=source_archive)
    text = json.dumps(profile, indent=2, sort_keys=True) + "\n"
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
