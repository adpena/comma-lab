#!/usr/bin/env python3
"""Build a planning-only PR85 side-channel attribution profile.

This helper is static forensics only. It parses the public PR85 single-member
bundle, records first-level decoded schema facts that are available without
importing or executing the PR85 runtime, and ranks isolated transplant/eval
candidates for later exact-eval planning. It does not run inflate, load scorers,
claim scores, dispatch jobs, or write dispatch state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

try:
    import brotli
except ImportError:  # pragma: no cover - exercised only in minimal envs
    brotli = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.qma9_range_mask_contract import QMA9ContractError, parse_qma9_header, trace_qma9_prefix

TOOL = "experiments/profile_pr85_adaptive_masking_sidechannel_attribution.py"
SCHEMA = "pr85_adaptive_masking_sidechannel_attribution_plan_v1"
EVIDENCE_GRADE = "external/planning_only_static_anatomy"
DEFAULT_INTAKE_DIR = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex"
DEFAULT_ARCHIVE = DEFAULT_INTAKE_DIR / "archive.zip"
DEFAULT_JSON_OUT = DEFAULT_INTAKE_DIR / "pr85_adaptive_masking_sidechannel_attribution_plan.json"
DEFAULT_QMA9_PREFIX_PIXELS = 4096

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
PAIR_COUNT = 600
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_LAMBDA = 25.0 / ORIGINAL_VIDEO_BYTES

# Headerless PR85 randmulti decode schedule copied as data from the public
# replay runtime. Keeping it here avoids importing the Torch/AV runtime.
HEADERLESS_RANDMULTI_SPECS = (
    (24, 32, 1, 12),
    (12, 16, 1, 1),
    (6, 8, 1, 1),
    (3, 4, 1, 1),
    (2, 2, 1, 1),
    (8, 8, 1, 1),
    (4, 4, 1, 1),
    (4, 8, 1, 1),
    (2, 4, 1, 1),
    (2, 8, 1, 1),
    (1, 2, 1, 1),
    (1, 4, 1, 1),
    (2, 1, 1, 1),
    (4, 1, 1, 1),
    (8, 1, 1, 1),
    (1, 8, 1, 1),
    (16, 1, 1, 1),
    (1, 16, 1, 1),
    (32, 1, 1, 1),
    (64, 1, 1, 1),
    (256, 1, 1, 1),
    (1024, 1, 1, 1),
    (2048, 1, 1, 1),
    (4096, 1, 1, 1),
    (8192, 1, 1, 1),
    (8192, 1, 1, 1),
    (16384, 1, 1, 1),
    (32768, 1, 1, 1),
    (65536, 1, 1, 1),
    (131072, 1, 1, 1),
    (262144, 1, 1, 1),
    (524288, 1, 1, 1),
    (1048576, 1, 1, 1),
    (874, 1, 1, 1),
    (874, 1, 1, 1),
    (2097152, 1, 1, 1),
    (875, 1, 1, 1),
    (876, 1, 1, 1),
    (877, 1, 1, 1),
    (1164, 1, 1, 1),
    (878, 1, 1, 1),
    (879, 1, 1, 1),
    (880, 1, 1, 1),
    (881, 1, 1, 1),
    (882, 1, 1, 1),
    (512, 2, 1, 1),
    (256, 2, 1, 1),
    (128, 2, 1, 1),
    (64, 2, 1, 1),
    (32, 2, 1, 1),
    (16, 2, 1, 1),
    (8, 2, 1, 1),
    (4, 2, 1, 1),
    (4, 4, 1, 1),
    (8, 4, 1, 1),
    (16, 4, 1, 1),
    (32, 4, 1, 1),
    (64, 4, 1, 1),
    (128, 4, 1, 1),
    (64, 8, 1, 1),
    (32, 8, 1, 1),
    (222, 222, 4, 1),
    (222, 223, 4, 1),
    (223, 222, 2, 1),
    (223, 223, 4, 1),
    (223, 221, 4, 1),
    (223, 224, 4, 1),
    (223, 221, 4, 1),
    (223, 219, 4, 1),
    (64, 16, 1, 1),
    (223, 218, 4, 1),
    (224, 222, 4, 1),
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _u24le(data: bytes, offset: int) -> int:
    return int.from_bytes(data[offset : offset + 3], "little")


def _magic_ascii(data: bytes, n: int = 8) -> str:
    return data[:n].decode("ascii", errors="replace")


def _read_single_member_archive(path: Path, member: str) -> tuple[dict[str, Any], bytes]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != [member]:
            raise ValueError(f"{path} must contain exactly one non-directory member {member!r}; got {names!r}")
        info = infos[0]
        raw = zf.read(info)
    return (
        {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
            "member_name": info.filename,
            "member_file_size": int(info.file_size),
            "member_compress_size": int(info.compress_size),
            "member_crc32_hex": f"{info.CRC:08x}",
            "member_sha256": _sha256_bytes(raw),
        },
        raw,
    )


def parse_pr85_v5_bundle(raw: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    if len(raw) < 24:
        raise ValueError("PR85 bundle is too short for v5 micro header")
    header_lengths = {
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
    offsets: dict[str, int] = {}
    for name in SEGMENT_ORDER[:-1]:
        size = header_lengths[name]
        if size <= 0:
            raise ValueError(f"invalid PR85 segment length for {name}: {size}")
        end = pos + size
        if end > len(raw):
            raise ValueError(f"truncated PR85 segment {name}")
        offsets[name] = pos
        segments[name] = raw[pos:end]
        pos = end
    if pos >= len(raw):
        raise ValueError("PR85 bundle is missing randmulti tail")
    offsets["randmulti"] = pos
    segments["randmulti"] = raw[pos:]
    return (
        {
            "format": "pr85_v5_micro_24bit_lengths_fixed_bias_region",
            "header_bytes": 24,
            "segment_offsets": offsets,
            "segment_lengths": {name: len(segments[name]) for name in SEGMENT_ORDER},
            "fixed_length_segments": {"bias": FIXED_V5_BIAS_BYTES, "region": FIXED_V5_REGION_BYTES},
        },
        segments,
    )


def _try_brotli_decode(data: bytes) -> dict[str, Any]:
    if brotli is None:
        return {"brotli_available": False, "decoded_ok": False, "decoded_error": "brotli_not_available"}
    try:
        decoded = brotli.decompress(data)
    except brotli.error as exc:
        return {"brotli_available": True, "decoded_ok": False, "decoded_error": str(exc)}
    return {
        "brotli_available": True,
        "decoded_ok": True,
        "decoded_bytes": int(len(decoded)),
        "decoded_sha256": _sha256_bytes(decoded),
        "decoded_magic_hex": decoded[:8].hex(),
        "decoded_magic_ascii": _magic_ascii(decoded),
        "decoded_payload": decoded,
    }


def _byte_stats(values: bytes) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "unique_count": 0,
            "nonzero_count": 0,
            "entropy_bits_per_symbol": 0.0,
            "ideal_entropy_bytes": 0.0,
            "dominant_symbol": None,
            "dominant_symbol_count": 0,
            "dominant_symbol_fraction": None,
        }
    counts = Counter(values)
    total = len(values)
    entropy = -sum((count / total) * math.log2(count / total) for count in counts.values())
    dominant_symbol, dominant_count = max(counts.items(), key=lambda item: (item[1], -item[0]))
    return {
        "count": int(total),
        "min": int(min(values)),
        "max": int(max(values)),
        "unique_count": int(len(counts)),
        "nonzero_count": int(sum(1 for value in values if value)),
        "entropy_bits_per_symbol": round(float(entropy), 6),
        "ideal_entropy_bytes": round(float(entropy * total / 8.0), 3),
        "dominant_symbol": int(dominant_symbol),
        "dominant_symbol_count": int(dominant_count),
        "dominant_symbol_fraction": round(float(dominant_count / total), 6),
    }


def _read_varints(raw: bytes, pos: int, count: int) -> tuple[list[int], int]:
    values = []
    for _ in range(count):
        acc = 0
        shift = 0
        while True:
            if pos >= len(raw):
                raise ValueError("truncated varint stream")
            byte = raw[pos]
            pos += 1
            acc |= (byte & 0x7F) << shift
            if byte & 0x80:
                shift += 7
            else:
                break
        values.append(acc)
    return values, pos


def _parse_qma9_mask(raw: bytes) -> dict[str, Any]:
    if len(raw) < 20 or raw[:4] != b"QMA9":
        return {"kind": "unknown_mask_payload", "recognized": False}
    frame_count = int.from_bytes(raw[4:8], "little")
    height = int.from_bytes(raw[8:12], "little")
    width = int.from_bytes(raw[12:16], "little")
    bitstream_bytes = int.from_bytes(raw[16:20], "little")
    return {
        "kind": "qma9_range_mask",
        "recognized": True,
        "magic": "QMA9",
        "frame_count": frame_count,
        "height": height,
        "width": width,
        "declared_bitstream_bytes": bitstream_bytes,
        "header_bytes": 20,
        "actual_bitstream_bytes": max(0, len(raw) - 20),
        "declared_length_matches_payload": bitstream_bytes == max(0, len(raw) - 20),
        "decoded_mask_bytes": int(frame_count * height * width),
        "class_count_assumption": 5,
    }


def _parse_model(raw: bytes) -> dict[str, Any]:
    magic = raw[:3]
    if magic in (b"QM0", b"QH0"):
        return {
            "kind": "custom_quantized_joint_frame_model",
            "recognized": True,
            "magic": magic.decode("ascii"),
            "hilo_split": magic == b"QH0",
            "static_parse_limit": "tensor table depends on JointFrameGenerator module order in replay runtime",
            "runtime_imported": False,
        }
    return {
        "kind": "torch_serialized_model_or_unknown",
        "recognized": False,
        "magic_ascii": _magic_ascii(raw),
        "runtime_imported": False,
    }


def _parse_pose(raw: bytes) -> dict[str, Any]:
    if raw[:4] == b"P1D1" and len(raw) >= 5:
        pos = 4
        dim_count = raw[pos]
        pos += 1
        dims = []
        lengths = []
        for _ in range(dim_count):
            if pos + 3 > len(raw):
                raise ValueError("truncated P1D1 pose header")
            dims.append(int(raw[pos]))
            pos += 1
            lengths.append(int.from_bytes(raw[pos : pos + 2], "little"))
            pos += 2
        return {
            "kind": "p1d1_delta_pose_stream",
            "recognized": True,
            "magic": "P1D1",
            "frame_count_assumption": PAIR_COUNT,
            "dimension_count": int(dim_count),
            "dimensions": dims,
            "stream_lengths": lengths,
            "encoded_stream_bytes": int(sum(lengths)),
            "header_bytes": int(pos),
            "payload_length_matches_streams": pos + sum(lengths) == len(raw),
        }
    if raw[:4] in (b"PQ12", b"PQB1"):
        return {"kind": "quantized_pose_stream", "recognized": True, "magic": raw[:4].decode("ascii")}
    if raw[:6] == b"\x93NUMPY":
        return {"kind": "numpy_pose_array", "recognized": True, "magic": "NUMPY"}
    return {"kind": "unknown_pose_payload", "recognized": False, "magic_ascii": _magic_ascii(raw)}


def _parse_post(raw: bytes) -> dict[str, Any]:
    if raw[:4] == b"PCD1" and len(raw) >= 5:
        pos = 5
        stage_count = raw[4]
        stages = []
        for _ in range(stage_count):
            if pos + 3 > len(raw):
                raise ValueError("truncated PCD1 post stage")
            stage_id = raw[pos]
            pos += 1
            choices = int.from_bytes(raw[pos : pos + 2], "little")
            pos += 2
            stages.append({"stage_id": int(stage_id), "choice_count": choices})
            pos += choices
        return {
            "kind": "pcd1_post_code_stages",
            "recognized": True,
            "magic": "PCD1",
            "stage_count": int(stage_count),
            "stages": stages,
            "payload_length_matches_stages": pos == len(raw),
        }
    if len(raw) % PAIR_COUNT == 0:
        stage_count = len(raw) // PAIR_COUNT
        stage_stats = [_byte_stats(raw[i * PAIR_COUNT : (i + 1) * PAIR_COUNT]) for i in range(stage_count)]
        return {
            "kind": "headerless_post_code_stages",
            "recognized": True,
            "stage_count": int(stage_count),
            "pairs_per_stage": PAIR_COUNT,
            "choice_stats_by_stage": stage_stats,
            "ideal_entropy_bytes_total": round(float(sum(row["ideal_entropy_bytes"] for row in stage_stats)), 3),
        }
    return {"kind": "unknown_post_payload", "recognized": False, "bytes_mod_600": len(raw) % PAIR_COUNT}


def _parse_simple_choice_payload(raw: bytes, *, default_center: int | None = None) -> dict[str, Any]:
    magic = raw[:3].decode("ascii", errors="replace")
    values = raw[3:]
    facts: dict[str, Any] = {
        "recognized": magic in {"SH4", "SD4", "FH2", "FH3", "FD3", "BH1", "BD1", "RH1", "RD1"},
        "magic": magic,
        "encoded_choice_stats": _byte_stats(values),
    }
    if len(values) == PAIR_COUNT:
        facts["pair_count"] = PAIR_COUNT
    if default_center is not None and magic in {"SD4", "FD3", "BD1", "RD1"}:
        facts["delta_zero_maps_to_choice"] = default_center
    return facts


def _parse_sparse_var_choice_payload(raw: bytes, *, magic: bytes, default_choice: int) -> dict[str, Any]:
    if not raw.startswith(magic) or len(raw) < 5:
        return {"recognized": False, "expected_magic": magic.decode("ascii")}
    count = int.from_bytes(raw[3:5], "little")
    pos = 5
    gaps, pos = _read_varints(raw, pos, count)
    values = raw[pos : pos + count]
    indices = []
    idx = -1
    for gap in gaps:
        idx += gap + 1
        indices.append(idx)
    return {
        "recognized": True,
        "magic": magic.decode("ascii"),
        "pair_count": PAIR_COUNT,
        "sparse_override_count": int(count),
        "sparse_override_fraction": round(float(count / PAIR_COUNT), 6),
        "default_choice": int(default_choice),
        "min_index": min(indices) if indices else None,
        "max_index": max(indices) if indices else None,
        "value_stats": _byte_stats(values),
        "payload_length_matches_sparse_table": pos + count == len(raw),
    }


def _parse_randmulti(raw: bytes) -> dict[str, Any]:
    if raw[:3] == b"NM1" and len(raw) >= 4:
        scount = int(raw[3])
        return {
            "kind": "nm1_randmulti_grid_table",
            "recognized": True,
            "magic": "NM1",
            "group_count": 1,
            "selection_rows": scount,
            "pairs_per_row": PAIR_COUNT,
            "payload_length_matches": len(raw) == 4 + scount * PAIR_COUNT,
        }
    if raw[:3] == b"NM2" and len(raw) >= 4:
        pos = 4
        groups = []
        for _ in range(int(raw[3])):
            if pos + 4 > len(raw):
                raise ValueError("truncated NM2 randmulti group")
            lh, lw, amp, scount = raw[pos], raw[pos + 1], raw[pos + 2], raw[pos + 3]
            pos += 4 + scount * PAIR_COUNT
            groups.append({"lh": int(lh), "lw": int(lw), "amp": int(amp), "selection_rows": int(scount)})
        return {
            "kind": "nm2_randmulti_grid_tables",
            "recognized": True,
            "magic": "NM2",
            "group_count": len(groups),
            "groups": groups,
            "payload_length_matches": pos == len(raw),
        }

    pos = 0
    groups = []
    total_nonzero = 0
    max_count = 0
    for group_index, (lh, lw, amp, scount) in enumerate(HEADERLESS_RANDMULTI_SPECS):
        group_nonzero = 0
        row_counts = []
        for _ in range(scount):
            if pos >= len(raw):
                raise ValueError("truncated headerless randmulti count")
            count = raw[pos]
            pos += 1
            if count == 255:
                if pos + 2 > len(raw):
                    raise ValueError("truncated extended headerless randmulti count")
                count = int.from_bytes(raw[pos : pos + 2], "little")
                pos += 2
            _, pos = _read_varints(raw, pos, count)
            pos += count
            group_nonzero += count
            row_counts.append(count)
            max_count = max(max_count, count)
        total_nonzero += group_nonzero
        possible_entries = PAIR_COUNT * scount
        groups.append(
            {
                "group_index": int(group_index),
                "lh": int(lh),
                "lw": int(lw),
                "amp": int(amp),
                "selection_rows": int(scount),
                "nonzero_entries": int(group_nonzero),
                "possible_pair_rows": int(possible_entries),
                "nonzero_density": round(float(group_nonzero / possible_entries), 6),
                "row_count_stats": _byte_stats(bytes(min(count, 255) for count in row_counts)),
            }
        )
    selection_rows = int(sum(group["selection_rows"] for group in groups))
    possible_pair_rows = PAIR_COUNT * selection_rows
    return {
        "kind": "headerless_randmulti_sparse_tables",
        "recognized": True,
        "group_count": len(groups),
        "selection_rows": selection_rows,
        "pairs_per_row": PAIR_COUNT,
        "nonzero_entries": int(total_nonzero),
        "possible_pair_rows": int(possible_pair_rows),
        "nonzero_density": round(float(total_nonzero / possible_pair_rows), 6),
        "max_nonzero_entries_in_row": int(max_count),
        "payload_length_matches_specs": pos == len(raw),
        "top_groups_by_nonzero": sorted(
            groups,
            key=lambda group: (-int(group["nonzero_entries"]), int(group["group_index"])),
        )[:10],
        "groups": groups,
    }


def decoded_schema_facts(name: str, decoded: bytes | None, raw: bytes) -> dict[str, Any]:
    if name == "mask":
        return _parse_qma9_mask(raw)
    if decoded is None:
        return {"recognized": False, "reason": "decoded_payload_unavailable"}
    if name == "model":
        return _parse_model(decoded)
    if name == "pose":
        return _parse_pose(decoded)
    if name == "post":
        return _parse_post(decoded)
    if name == "shift":
        return _parse_simple_choice_payload(decoded, default_center=40)
    if name == "frac":
        if decoded[:3] == b"FV1":
            return _parse_sparse_var_choice_payload(decoded, magic=b"FV1", default_choice=4)
        return _parse_simple_choice_payload(decoded)
    if name == "frac2":
        return _parse_simple_choice_payload(decoded)
    if name == "frac3":
        return _parse_simple_choice_payload(decoded, default_center=4)
    if name == "bias":
        if decoded[:3] == b"BV1":
            return _parse_sparse_var_choice_payload(decoded, magic=b"BV1", default_choice=13)
        return _parse_simple_choice_payload(decoded, default_center=13)
    if name == "region":
        if decoded[:3] == b"RV1":
            return _parse_sparse_var_choice_payload(decoded, magic=b"RV1", default_choice=0)
        return _parse_simple_choice_payload(decoded, default_center=0)
    if name == "randmulti":
        return _parse_randmulti(decoded)
    return {"recognized": False, "reason": "unknown_segment"}


def segment_profiles(segments: dict[str, bytes], offsets: dict[str, int]) -> list[dict[str, Any]]:
    rows = []
    for name in SEGMENT_ORDER:
        raw = segments[name]
        decode = {"decoded_ok": False, "decoded_payload": None}
        if name != "mask":
            decode = _try_brotli_decode(raw)
        decoded = decode.get("decoded_payload")
        decoded_bytes = int(decode["decoded_bytes"]) if decode.get("decoded_ok") else None
        row = {
            "name": name,
            "offset": int(offsets[name]),
            "raw_bytes": int(len(raw)),
            "raw_sha256": _sha256_bytes(raw),
            "raw_magic_hex": raw[:8].hex(),
            "raw_magic_ascii": _magic_ascii(raw),
            "container_codec": "raw" if name == "mask" else "brotli",
            "decoded_ok": True if name == "mask" else bool(decode.get("decoded_ok")),
            "decoded_bytes": int(len(raw)) if name == "mask" else decoded_bytes,
            "decoded_sha256": _sha256_bytes(raw) if name == "mask" else decode.get("decoded_sha256"),
            "decoded_magic_hex": raw[:8].hex() if name == "mask" else decode.get("decoded_magic_hex"),
            "decoded_magic_ascii": _magic_ascii(raw) if name == "mask" else decode.get("decoded_magic_ascii"),
            "schema_facts": decoded_schema_facts(name, decoded if isinstance(decoded, bytes) else None, raw),
        }
        if "decoded_payload" in row:
            raise AssertionError("internal error: decoded payload leaked into JSON row")
        rows.append(row)
    return rows


def qma9_mask_token_prefix_profile(mask_payload: bytes, *, max_pixels: int) -> dict[str, Any]:
    """Extract a bounded PR85 QMA9 token/predictor trace for entropy planning.

    This is intentionally not a score or decoded-video parity proof. It checks
    only the charged QMA9 mask bytes and emits deterministic prefix statistics
    that can guide a future contest-runtime entropy coder.
    """

    max_pixels = int(max_pixels)
    if max_pixels <= 0:
        return {
            "status": "skipped",
            "reason": "qma9 prefix token extraction disabled",
            "score_claim": False,
            "dispatch_status": "planning_only_no_dispatch",
        }
    try:
        header = parse_qma9_header(mask_payload)
        prefix_pixels = min(max_pixels, int(header.decoded_mask_bytes))
        trace = trace_qma9_prefix(mask_payload, max_pixels=prefix_pixels)
    except (QMA9ContractError, ValueError) as exc:
        return {
            "status": "failed_closed",
            "reason": str(exc),
            "score_claim": False,
            "dispatch_status": "planning_only_no_dispatch",
            "payload_sha256": _sha256_bytes(mask_payload),
        }

    return {
        "status": "passed",
        "tool": "src/tac/qma9_range_mask_contract.py::trace_qma9_prefix",
        "score_claim": False,
        "dispatch_status": "planning_only_no_dispatch",
        "promotable": False,
        "promotable_reason": "bounded mask-token prefix profile only; no archive transform or exact CUDA eval",
        "payload_sha256": _sha256_bytes(mask_payload),
        "qma9_header": {
            "magic": header.magic,
            "frame_count": int(header.frame_count),
            "width": int(header.width),
            "height": int(header.height),
            "bitstream_bytes": int(header.bitstream_bytes),
            "decoded_mask_bytes": int(header.decoded_mask_bytes),
            "payload_sha256": header.payload_sha256,
            "bitstream_sha256": header.bitstream_sha256,
        },
        "prefix_pixels_requested": max_pixels,
        "prefix_pixels_traced": int(trace["decoded_prefix_pixels"]),
        "decoded_prefix_sha256": trace["decoded_prefix_sha256"],
        "estimated_model_bits": trace["estimated_model_bits"],
        "estimated_model_bytes": trace["estimated_model_bytes"],
        "stage_counts": trace["stage_counts"],
        "stage_estimated_bits": trace["stage_estimated_bits"],
        "predictor_counts": trace["predictor_counts"],
        "class_counts": trace["class_counts"],
        "top_contexts": trace["top_contexts"],
        "decoder_state_after_prefix": trace["decoder_state_after_prefix"],
        "prefix_self_roundtrip": trace.get("prefix_self_roundtrip"),
        "failure_modes_guarded": [
            "bad QMA9 magic",
            "declared bitstream overrun",
            "invalid decoded class symbol",
            "checkpoint or prefix outside decoded mask bounds",
        ],
    }


def _segments_by_name(segment_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in segment_rows}


def _candidate(
    *,
    candidate_id: str,
    segments: tuple[str, ...],
    segment_rows: dict[str, dict[str, Any]],
    opportunity: str,
    risk: str,
    risk_penalty: float,
    isolation_scope: str,
    rationale: str,
    next_local_step: str,
) -> dict[str, Any]:
    raw_bytes = sum(int(segment_rows[name]["raw_bytes"]) for name in segments)
    decoded_bytes = sum(int(segment_rows[name].get("decoded_bytes") or 0) for name in segments)
    recognized = sum(1 for name in segments if segment_rows[name]["schema_facts"].get("recognized"))
    schema_bonus = 1.0 + 0.12 * recognized
    rate_budget = raw_bytes * RATE_LAMBDA
    # A deterministic planning heuristic only: byte mass, decoded control
    # surface, and static schema confidence, discounted by transplant risk.
    ev = (math.log2(raw_bytes + 1.0) + 0.35 * math.log2(decoded_bytes + 1.0)) * schema_bonus
    ev -= risk_penalty
    return {
        "candidate_id": candidate_id,
        "segment_names": list(segments),
        "charged_raw_bytes": int(raw_bytes),
        "decoded_bytes": int(decoded_bytes),
        "rate_score_budget_if_components_unchanged": round(rate_budget, 12),
        "expected_ev_score": round(ev, 6),
        "opportunity": opportunity,
        "risk": risk,
        "isolation_scope": isolation_scope,
        "rationale": rationale,
        "next_local_step": next_local_step,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "dispatch_status": "planning_only_no_dispatch",
    }


def attribution_candidates(segment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = _segments_by_name(segment_rows)
    candidates = [
        _candidate(
            candidate_id="randmulti_sparse_table_isolation",
            segments=("randmulti",),
            segment_rows=rows,
            opportunity="largest explicit post-render side-channel with sparse per-pair/grid corrections",
            risk="medium_high",
            risk_penalty=1.2,
            isolation_scope="zero or transplant only randmulti while preserving mask/model/pose/post/shift/frac/bias/region bytes",
            rationale="High decoded control surface and direct frame-1 correction path; attribution can be measured by one isolated ablation before any remote exact eval.",
            next_local_step="build a local replay-only archive variant with randmulti removed or replaced by neutral tables, then compare byte closure and decoded output hashes.",
        ),
        _candidate(
            candidate_id="mask_qma9_range_mask_transplant",
            segments=("mask",),
            segment_rows=rows,
            opportunity="dominant charged semantic-mask representation and likely reusable base for PR85-style adaptive masking",
            risk="high",
            risk_penalty=3.0,
            isolation_scope="mask-only transplant with unchanged joint-frame model and all side-channel payloads",
            rationale="Largest byte mass and first-order semantic representation; high component risk means it needs local decode parity and pose-safety gates before dispatch planning.",
            next_local_step="extract QMA9 header/body facts and compare decoded mask geometry against candidate bases without running scorer.",
        ),
        _candidate(
            candidate_id="model_qh0_joint_frame_model_transplant",
            segments=("model",),
            segment_rows=rows,
            opportunity="compressed JointFrameGenerator weights isolated from mask and side-channel controls",
            risk="high",
            risk_penalty=2.8,
            isolation_scope="model-only transplant with PR85 mask/pose/post side channels preserved",
            rationale="Large byte block with a custom QH0 quantized schema; model-only swaps can attribute renderer capacity but require strict runtime-custody controls.",
            next_local_step="parse tensor-table boundaries with an AST-derived module manifest before any torch import or runtime execution.",
        ),
        _candidate(
            candidate_id="post_code_stage_ablation",
            segments=("post",),
            segment_rows=rows,
            opportunity="compact 4-stage per-pair RGB gain/bias postprocess codebook choices",
            risk="medium",
            risk_penalty=0.4,
            isolation_scope="post-only neutralization while preserving all frame-1 geometric and sparse correction streams",
            rationale="Low charged bytes but complete 600-pair stage coverage; cheap to ablate locally and likely cleanly attributable.",
            next_local_step="emit neutral post choices for all stages and run local replay output-hash comparison only.",
        ),
        _candidate(
            candidate_id="f1_micro_sidechannel_stack",
            segments=("shift", "frac", "frac2", "frac3", "bias", "region"),
            segment_rows=rows,
            opportunity="small frame-1 geometric/RGB/region adjustment stack with strong byte-to-control leverage",
            risk="medium",
            risk_penalty=0.7,
            isolation_scope="neutralize or transplant the six micro side-channel streams as a stack, then bisect by segment",
            rationale="Tiny charged footprint controls every pair through shifts, fractional grids, bias, and region choices; good EV for attribution despite small size.",
            next_local_step="build neutral decoded tables for each stream and plan a deterministic local ablation ladder.",
        ),
        _candidate(
            candidate_id="pose_p1d1_stream_transfer",
            segments=("pose",),
            segment_rows=rows,
            opportunity="very small 6D pose delta stream coupled to the frame-1 conditioned head",
            risk="high",
            risk_penalty=2.0,
            isolation_scope="pose-only transplant or neutralization with PR85 mask/model/post side channels preserved",
            rationale="Excellent byte leverage but PoseNet-sensitive; planning should remain local until exact pose safety and output parity gates exist.",
            next_local_step="decode P1D1 dimensions and stream lengths, then compare pose manifold ranges against candidate archives.",
        ),
    ]
    return [
        {**candidate, "rank": rank}
        for rank, candidate in enumerate(
            sorted(candidates, key=lambda item: (-item["expected_ev_score"], item["candidate_id"])),
            start=1,
        )
    ]


def build_profile(
    archive_path: Path,
    *,
    member: str = "x",
    qma9_prefix_pixels: int = DEFAULT_QMA9_PREFIX_PIXELS,
) -> dict[str, Any]:
    archive, raw = _read_single_member_archive(archive_path, member)
    bundle, segments = parse_pr85_v5_bundle(raw)
    rows = segment_profiles(segments, bundle["segment_offsets"])
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "deterministic": True,
        "score_claim": False,
        "dispatch_performed": False,
        "gpu_required": False,
        "evidence_grade": EVIDENCE_GRADE,
        "source_archive": archive,
        "bundle": bundle,
        "segments": rows,
        "qma9_mask_token_prefix_profile": qma9_mask_token_prefix_profile(
            segments["mask"],
            max_pixels=qma9_prefix_pixels,
        ),
        "recommended_isolated_transplant_eval_candidates": attribution_candidates(rows),
        "planning_constraints": [
            "No score claim is made by this static profile.",
            "No remote job or exact eval dispatch is performed by this tool.",
            "Any future score claim must use archive.zip -> inflate.sh -> upstream/evaluate.py on CUDA.",
            "PR85 runtime is not imported; schema facts are first-level static parses only.",
            "QMA9 token-prefix facts are bounded pure-Python mask-stream forensics, not decoded-video parity.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--member", default="x")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument(
        "--qma9-prefix-pixels",
        type=int,
        default=DEFAULT_QMA9_PREFIX_PIXELS,
        help="bounded QMA9 mask-token prefix pixels to trace; set 0 to skip",
    )
    args = parser.parse_args(argv)

    payload = build_profile(args.archive, member=args.member, qma9_prefix_pixels=args.qma9_prefix_pixels)
    text = _json_text(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
