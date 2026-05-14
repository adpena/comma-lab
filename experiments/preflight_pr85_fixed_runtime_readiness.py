#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Planning-only PR85 fixed-runtime readiness preflight.

This helper answers one narrow question: what still blocks replaying the
public PR85 v5 single-member archive through ``submissions/robust_current``
instead of the public replay runtime?  It parses the exact PR85 bundle with
``tac.pr85_bundle``, performs cheap byte/decompression probes, statically
inspects the robust_current runtime surface, and emits deterministic JSON.

It does not inflate archives, run scorers, claim score evidence, dispatch jobs,
or mutate dispatch state.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import struct
import sys
import zipfile
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

from tac.pr85_bundle import (  # noqa: E402
    FIXED_V5_LENGTHS,
    QPOST_STREAM_NAMES,
    SEGMENT_ORDER,
    Pr85BundleError,
    expand_pr85_bundle_to_runtime_members,
    pack_pr85_bundle,
    parse_pr85_bundle,
    validate_pr85_member_name,
)


TOOL = "experiments/preflight_pr85_fixed_runtime_readiness.py"
SCHEMA = "pr85_fixed_runtime_readiness_preflight_v1"
EVIDENCE_GRADE = "planning_only/no_score_claim"
DEFAULT_INTAKE_DIR = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex"
DEFAULT_ARCHIVE = DEFAULT_INTAKE_DIR / "archive.zip"
DEFAULT_ROBUST_CURRENT = REPO_ROOT / "submissions/robust_current"
DEFAULT_JSON_OUT = DEFAULT_INTAKE_DIR / "pr85_fixed_runtime_readiness_preflight.json"
KNOWN_PUBLIC_PR85 = {
    "archive_bytes": 236_328,
    "archive_sha256": "eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e",
    "member_name": "x",
}
SEGMENT_RUNTIME_SURFACES: dict[str, tuple[str, ...]] = {
    "mask": ("masks.qma9",),
    "model": ("renderer.bin",),
    "pose": ("optimized_poses.bin",),
    "post": ("qpost.bin", "qpost:post"),
    "shift": ("qpost.bin", "qpost:shift"),
    "frac": ("qpost.bin", "qpost:frac"),
    "frac2": ("qpost.bin", "qpost:frac2"),
    "frac3": ("qpost.bin", "qpost:frac3"),
    "bias": ("qpost.bin", "qpost:bias"),
    "region": ("qpost.bin", "qpost:region"),
    "randmulti": ("qpost.bin", "qpost:randmulti", "QRM1_transcode"),
}

EXPECTED_DECODED_MAGICS: dict[str, bytes | None] = {
    "mask": b"QMA9",
    "model": b"QH0",
    "pose": b"P1D1",
    "post": None,
    "shift": b"SD4",
    "frac": b"FV1",
    "frac2": b"FH2",
    "frac3": b"FD3",
    "bias": b"BD1",
    "region": b"RH1",
    "randmulti": None,
}

# PR85 v5 public replay headerless randmulti schedule.  The same sequence is
# already documented in the PR85 static attribution tooling; keeping it local
# here lets the preflight validate decoded byte closure without importing the
# replay runtime or the heavier planning module.
PR85_HEADERLESS_RANDMULTI_SPECS: tuple[tuple[int, int, int, int], ...] = (
    (24, 32, 1, 12), (12, 16, 1, 1), (6, 8, 1, 1), (3, 4, 1, 1),
    (2, 2, 1, 1), (8, 8, 1, 1), (4, 4, 1, 1), (4, 8, 1, 1),
    (2, 4, 1, 1), (2, 8, 1, 1), (1, 2, 1, 1), (1, 4, 1, 1),
    (2, 1, 1, 1), (4, 1, 1, 1), (8, 1, 1, 1), (1, 8, 1, 1),
    (16, 1, 1, 1), (1, 16, 1, 1), (32, 1, 1, 1), (64, 1, 1, 1),
    (256, 1, 1, 1), (1024, 1, 1, 1), (2048, 1, 1, 1), (4096, 1, 1, 1),
    (8192, 1, 1, 1), (8192, 1, 1, 1), (16384, 1, 1, 1), (32768, 1, 1, 1),
    (65536, 1, 1, 1), (131072, 1, 1, 1), (262144, 1, 1, 1),
    (524288, 1, 1, 1), (1048576, 1, 1, 1), (874, 1, 1, 1),
    (874, 1, 1, 1), (2097152, 1, 1, 1), (875, 1, 1, 1),
    (876, 1, 1, 1), (877, 1, 1, 1), (1164, 1, 1, 1),
    (878, 1, 1, 1), (879, 1, 1, 1), (880, 1, 1, 1),
    (881, 1, 1, 1), (882, 1, 1, 1), (512, 2, 1, 1),
    (256, 2, 1, 1), (128, 2, 1, 1), (64, 2, 1, 1),
    (32, 2, 1, 1), (16, 2, 1, 1), (8, 2, 1, 1), (4, 2, 1, 1),
    (4, 4, 1, 1), (8, 4, 1, 1), (16, 4, 1, 1), (32, 4, 1, 1),
    (64, 4, 1, 1), (128, 4, 1, 1), (64, 8, 1, 1),
    (32, 8, 1, 1), (222, 222, 4, 1), (222, 223, 4, 1),
    (223, 222, 2, 1), (223, 223, 4, 1), (223, 221, 4, 1),
    (223, 224, 4, 1), (223, 221, 4, 1), (223, 219, 4, 1),
    (64, 16, 1, 1), (223, 218, 4, 1), (224, 222, 4, 1),
)

RUNTIME_REQUIRED_FILES = (
    "inflate.sh",
    "inflate_renderer.py",
    "unpack_renderer_payload.py",
    "apply_qzs3_postprocess.py",
    "range_mask_codec.cpp",
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


def _repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _ascii_prefix(data: bytes, n: int = 8) -> str:
    return data[:n].decode("ascii", errors="replace")


def _safe_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _function_names(path: Path) -> dict[str, Any]:
    text = _safe_text(path)
    if not text:
        return {"parse_ok": False, "functions": [], "error": "file_missing_or_empty"}
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return {"parse_ok": False, "functions": [], "error": str(exc)}
    functions = sorted(
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    )
    return {"parse_ok": True, "functions": functions}


def _extract_function_source(path: Path, function_name: str) -> str:
    text = _safe_text(path)
    if not text:
        return ""
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return ""
    lines = text.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            start = max(int(getattr(node, "lineno", 1)) - 1, 0)
            end = int(getattr(node, "end_lineno", start + 1))
            return "\n".join(lines[start:end])
    return ""


def _read_pr85_archive(path: Path) -> tuple[dict[str, Any], bytes]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["x"]:
            raise Pr85BundleError(
                f"PR85 archive must contain exactly one non-directory member 'x'; got {names!r}"
            )
        info = infos[0]
        validate_pr85_member_name(info.filename)
        raw = zf.read(info)
    archive_sha = _sha256_file(path)
    return (
        {
            "path": _repo_rel(path),
            "archive_bytes": int(path.stat().st_size),
            "archive_sha256": archive_sha,
            "known_public_pr85_v5_match": {
                "matches": (
                    int(path.stat().st_size) == KNOWN_PUBLIC_PR85["archive_bytes"]
                    and archive_sha == KNOWN_PUBLIC_PR85["archive_sha256"]
                ),
                "expected_archive_bytes": KNOWN_PUBLIC_PR85["archive_bytes"],
                "expected_archive_sha256": KNOWN_PUBLIC_PR85["archive_sha256"],
            },
            "member_name": info.filename,
            "member_file_size": int(info.file_size),
            "member_compress_size": int(info.compress_size),
            "member_crc32_hex": f"{info.CRC:08x}",
            "member_sha256": _sha256_bytes(raw),
            "zip_compress_type": int(info.compress_type),
        },
        raw,
    )


def _try_brotli_decode(data: bytes) -> dict[str, Any]:
    if brotli is None:
        return {
            "attempted": True,
            "brotli_available": False,
            "decoded_ok": False,
            "decoded_error": "brotli_not_available",
        }
    try:
        decoded = brotli.decompress(data)
    except brotli.error as exc:
        return {
            "attempted": True,
            "brotli_available": True,
            "decoded_ok": False,
            "decoded_error": str(exc),
        }
    return {
        "attempted": True,
        "brotli_available": True,
        "decoded_ok": True,
        "decoded_bytes": int(len(decoded)),
        "decoded_sha256": _sha256_bytes(decoded),
        "decoded_magic_ascii": _ascii_prefix(decoded),
        "decoded_magic_hex": decoded[:8].hex(),
        "decoded_payload": decoded,
    }


def _qma9_facts(payload: bytes) -> dict[str, Any]:
    if len(payload) < 20 or payload[:4] != b"QMA9":
        return {"recognized": False, "reason": "not_qma9_or_truncated"}
    frame_count, width, height, bitstream_bytes = struct.unpack_from("<IIII", payload, 4)
    declared_total = 20 + int(bitstream_bytes)
    return {
        "recognized": True,
        "magic": "QMA9",
        "frame_count": int(frame_count),
        "width": int(width),
        "height": int(height),
        "declared_bitstream_bytes": int(bitstream_bytes),
        "actual_bitstream_bytes": max(len(payload) - 20, 0),
        "declared_length_matches_payload": declared_total == len(payload),
    }


def _decoded_segment_facts(name: str, decoded: bytes | None) -> dict[str, Any]:
    if name == "mask":
        return _qma9_facts(decoded or b"")
    if decoded is None:
        return {"recognized": False, "reason": "not_decoded"}
    if name == "post":
        return {
            "recognized": True,
            "kind": "pr85_post_choice_stages",
            "decoded_bytes": int(len(decoded)),
            "pairs_per_stage_assumption": 600,
            "stage_count_if_600_pairs": int(len(decoded) // 600) if len(decoded) % 600 == 0 else None,
            "length_multiple_of_600": len(decoded) % 600 == 0,
        }
    if name == "randmulti":
        expected_rows = sum(spec[3] for spec in PR85_HEADERLESS_RANDMULTI_SPECS)
        cursor = 0
        rows_ok = True
        error = None
        total_nonzero = 0
        max_nonzero_in_row = 0
        for _lh, _lw, _amp, scount in PR85_HEADERLESS_RANDMULTI_SPECS:
            for _ in range(scount):
                if cursor >= len(decoded):
                    rows_ok = False
                    error = "truncated_before_count"
                    break
                count = decoded[cursor]
                cursor += 1
                if count == 255:
                    if cursor + 2 > len(decoded):
                        rows_ok = False
                        error = "truncated_extended_count"
                        break
                    count = int.from_bytes(decoded[cursor : cursor + 2], "little")
                    cursor += 2
                for _item in range(int(count)):
                    shift = 0
                    while True:
                        if cursor >= len(decoded):
                            rows_ok = False
                            error = "truncated_sparse_row_varint"
                            break
                        byte = decoded[cursor]
                        cursor += 1
                        if not (byte & 0x80):
                            break
                        shift += 7
                        if shift > 28:
                            rows_ok = False
                            error = "overlong_sparse_row_varint"
                            break
                    if not rows_ok:
                        break
                if not rows_ok:
                    break
                cursor += int(count)
                if cursor > len(decoded):
                    rows_ok = False
                    error = "truncated_sparse_row_values"
                    break
                total_nonzero += int(count)
                max_nonzero_in_row = max(max_nonzero_in_row, int(count))
            if not rows_ok:
                break
        return {
            "recognized": rows_ok and cursor == len(decoded),
            "kind": "pr85_v5_headerless_sparse_randmulti",
            "decoded_bytes": int(len(decoded)),
            "expected_group_count": len(PR85_HEADERLESS_RANDMULTI_SPECS),
            "expected_sparse_row_count": int(expected_rows),
            "nonzero_entries": int(total_nonzero),
            "max_nonzero_entries_in_row": int(max_nonzero_in_row),
            "payload_consumed_bytes": int(cursor),
            "payload_length_matches_specs": cursor == len(decoded),
            "parse_error": error,
        }
    magic = EXPECTED_DECODED_MAGICS.get(name)
    if magic is None:
        return {"recognized": True, "decoded_bytes": int(len(decoded))}
    return {
        "recognized": decoded.startswith(magic),
        "expected_magic_ascii": magic.decode("ascii", errors="replace"),
        "actual_magic_ascii": _ascii_prefix(decoded, len(magic)),
    }


def _segment_rows(bundle: Any) -> list[dict[str, Any]]:
    rows = []
    for name in SEGMENT_ORDER:
        raw = bytes(bundle.segments[name])
        row: dict[str, Any] = {
            "name": name,
            "offset": int(bundle.segment_offsets[name]),
            "bytes": int(len(raw)),
            "sha256": _sha256_bytes(raw),
            "encoded_magic_ascii": _ascii_prefix(raw),
            "encoded_magic_hex": raw[:8].hex(),
            "fixed_v5_length_expected": FIXED_V5_LENGTHS.get(name),
            "fixed_v5_length_matches": (
                len(raw) == FIXED_V5_LENGTHS[name] if name in FIXED_V5_LENGTHS else None
            ),
        }
        if name == "mask":
            row["decompression"] = {
                "attempted": False,
                "reason": "PR85 mask segment is raw QMA9",
            }
            row["schema_facts"] = _decoded_segment_facts(name, raw)
        else:
            probe = _try_brotli_decode(raw)
            decoded_payload = probe.pop("decoded_payload", None)
            row["decompression"] = probe
            row["schema_facts"] = _decoded_segment_facts(name, decoded_payload)
            expected_magic = EXPECTED_DECODED_MAGICS.get(name)
            if expected_magic is not None:
                row["decoded_magic_matches_expected"] = bool(
                    decoded_payload is not None and decoded_payload.startswith(expected_magic)
                )
        rows.append(row)
    return rows


def _bundle_report(raw: bytes) -> dict[str, Any]:
    bundle = parse_pr85_bundle(raw)
    roundtrip_mode = "v5" if bundle.header_bytes == 24 else "explicit_30"
    roundtrip = pack_pr85_bundle(bundle.segments, header_mode=roundtrip_mode)
    segment_rows = _segment_rows(bundle)
    cheap_probe_ok = all(
        row["schema_facts"].get("recognized", True) is True
        and (
            row["name"] == "mask"
            or row["decompression"].get("decoded_ok") is True
        )
        for row in segment_rows
    )
    return {
        "format": bundle.format,
        "header_bytes": int(bundle.header_bytes),
        "fixed_length_segments": dict(bundle.fixed_length_segments),
        "segment_lengths": bundle.segment_lengths,
        "roundtrip_pack_mode": roundtrip_mode,
        "roundtrip_matches_input": roundtrip == raw,
        "cheap_segment_probe_ok": cheap_probe_ok,
        "segments": segment_rows,
    }


def _runtime_file_rows(runtime_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for name in RUNTIME_REQUIRED_FILES:
        path = runtime_dir / name
        row: dict[str, Any] = {
            "name": name,
            "path": _repo_rel(path),
            "present": path.is_file(),
        }
        if path.is_file():
            row.update({"bytes": int(path.stat().st_size), "sha256": _sha256_file(path)})
        rows.append(row)
    return rows


def _capability(
    capability_id: str,
    passed: bool,
    evidence: list[str],
    blocker: str | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": capability_id,
        "status": "passed" if passed else "failed",
        "required_for_fixed_runtime_pr85_v5": True,
        "evidence": evidence,
    }
    if blocker:
        row["blocker"] = blocker
    return row


def _check(
    check_id: str,
    passed: bool,
    evidence: list[str],
    blocker: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": check_id,
        "status": "passed" if passed else "failed",
        "evidence": evidence,
    }
    if blocker:
        row["blocker"] = blocker
    row.update(extra)
    return row


def _literal_int_tuple_sequence(path: Path, constant_name: str) -> tuple[tuple[int, ...], ...] | None:
    text = _safe_text(path)
    if not text:
        return None
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return None
    for node in tree.body:
        value_node: ast.AST | None = None
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == constant_name for target in node.targets):
                value_node = node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == constant_name:
            value_node = node.value
        if value_node is None:
            continue
        try:
            value = ast.literal_eval(value_node)
        except (ValueError, TypeError):
            return None
        if not isinstance(value, tuple):
            return None
        rows: list[tuple[int, ...]] = []
        for item in value:
            if (
                not isinstance(item, tuple)
                or not item
                or not all(isinstance(part, int) for part in item)
            ):
                return None
            rows.append(tuple(int(part) for part in item))
        return tuple(rows)
    return None


def _consume_sparse_rows(raw: bytes, cursor: int, row_count: int) -> tuple[int, int, int]:
    total_nonzero = 0
    max_nonzero = 0
    for _ in range(row_count):
        if cursor >= len(raw):
            raise Pr85BundleError("QRM1 randmulti row ended before count byte")
        count = int(raw[cursor])
        cursor += 1
        if count == 255:
            if cursor + 2 > len(raw):
                raise Pr85BundleError("QRM1 randmulti extended count is truncated")
            count = int.from_bytes(raw[cursor : cursor + 2], "little")
            cursor += 2
        for _item in range(count):
            shift = 0
            while True:
                if cursor >= len(raw):
                    raise Pr85BundleError("QRM1 randmulti sparse index stream is truncated")
                byte = raw[cursor]
                cursor += 1
                if byte < 128:
                    break
                shift += 7
                if shift > 28:
                    raise Pr85BundleError("QRM1 randmulti sparse index VLQ is overlong")
        cursor += count
        if cursor > len(raw):
            raise Pr85BundleError("QRM1 randmulti value stream is truncated")
        total_nonzero += count
        max_nonzero = max(max_nonzero, count)
    return cursor, total_nonzero, max_nonzero


def _split_qpost_streams(qpost: bytes) -> tuple[dict[str, bytes], dict[str, Any]]:
    header_size = 4 + 4 * len(QPOST_STREAM_NAMES)
    if len(qpost) < header_size or qpost[:4] != b"QPS1":
        return {}, {
            "recognized": False,
            "reason": f"bad_qpost_magic_or_truncated:{qpost[:4]!r}",
        }
    lengths = struct.unpack_from("<" + "I" * len(QPOST_STREAM_NAMES), qpost, 4)
    cursor = header_size
    streams: dict[str, bytes] = {}
    for name, n_bytes in zip(QPOST_STREAM_NAMES, lengths):
        end = cursor + int(n_bytes)
        if end > len(qpost):
            return streams, {
                "recognized": False,
                "reason": f"qpost_stream_overruns_payload:{name}",
                "stream_lengths": {k: len(v) for k, v in streams.items()},
            }
        streams[name] = qpost[cursor:end]
        cursor = end
    if cursor != len(qpost):
        return streams, {
            "recognized": False,
            "reason": f"qpost_trailing_bytes:{len(qpost) - cursor}",
            "stream_lengths": {name: len(streams[name]) for name in QPOST_STREAM_NAMES if name in streams},
        }
    return streams, {
        "recognized": True,
        "stream_lengths": {name: len(streams[name]) for name in QPOST_STREAM_NAMES},
        "stream_sha256": {name: _sha256_bytes(streams[name]) for name in QPOST_STREAM_NAMES},
    }


def _qrm1_stream_facts(
    encoded_randmulti: bytes,
    runtime_specs: tuple[tuple[int, ...], ...] | None,
) -> dict[str, Any]:
    probe = _try_brotli_decode(encoded_randmulti)
    decoded = probe.pop("decoded_payload", None)
    if decoded is None or not probe.get("decoded_ok"):
        return {
            "recognized": False,
            "reason": probe.get("decoded_error", "brotli_decode_failed"),
            "decompression": probe,
        }
    if not decoded.startswith(b"QRM1"):
        return {
            "recognized": False,
            "reason": f"not_qrm1:{decoded[:4]!r}",
            "decompression": probe,
        }
    if len(decoded) < 6:
        return {
            "recognized": False,
            "reason": "qrm1_truncated_header",
            "decompression": probe,
        }
    if runtime_specs is None:
        return {
            "recognized": False,
            "reason": "runtime_qrm1_specs_missing_or_not_literal",
            "decompression": probe,
        }
    cursor = 4
    group_count = int.from_bytes(decoded[cursor : cursor + 2], "little")
    cursor += 2
    group_ids: list[int] = []
    nonzero_entries = 0
    max_nonzero = 0
    try:
        for _ in range(group_count):
            if cursor + 2 > len(decoded):
                raise Pr85BundleError("QRM1 group id is truncated")
            group_id = int.from_bytes(decoded[cursor : cursor + 2], "little")
            cursor += 2
            if group_id in group_ids:
                raise Pr85BundleError(f"QRM1 duplicate group id: {group_id}")
            if group_id >= len(runtime_specs):
                raise Pr85BundleError(f"QRM1 group id outside runtime specs: {group_id}")
            spec = runtime_specs[group_id]
            if len(spec) != 4:
                raise Pr85BundleError(f"QRM1 runtime spec {group_id} is malformed: {spec!r}")
            cursor, row_nonzero, row_max_nonzero = _consume_sparse_rows(
                decoded,
                cursor,
                int(spec[3]),
            )
            group_ids.append(group_id)
            nonzero_entries += row_nonzero
            max_nonzero = max(max_nonzero, row_max_nonzero)
        if cursor != len(decoded):
            raise Pr85BundleError("QRM1 stream has trailing bytes")
    except Pr85BundleError as exc:
        return {
            "recognized": False,
            "reason": str(exc),
            "decompression": probe,
            "decoded_group_count": group_count,
            "group_ids": group_ids,
        }
    return {
        "recognized": True,
        "contract": "QRM1_sparse_group_id_stream",
        "decompression": probe,
        "decoded_group_count": group_count,
        "group_ids": group_ids,
        "expected_group_ids": list(range(len(PR85_HEADERLESS_RANDMULTI_SPECS))),
        "group_ids_match_pr85_schedule": group_ids == list(range(len(PR85_HEADERLESS_RANDMULTI_SPECS))),
        "nonzero_entries": nonzero_entries,
        "max_nonzero_entries_in_row": max_nonzero,
    }


def _bridge_substrate_report(raw: bytes, expansion: Any | None, runtime_dir: Path) -> dict[str, Any]:
    if expansion is None:
        return {
            "schema": "pr85_fixed_runtime_atom_substrate_v1",
            "checks": [
                _check(
                    "bridge_expansion_available",
                    False,
                    ["No PR85 runtime expansion object was produced"],
                    "cannot prove atom substrate without expanded runtime members",
                )
            ],
            "remaining_blockers": [
                {
                    "code": "bridge_expansion_available",
                    "severity": "blocking",
                    "detail": "cannot prove atom substrate without expanded runtime members",
                }
            ],
        }

    source_bundle = parse_pr85_bundle(raw)
    source_segment_sha = {
        name: _sha256_bytes(bytes(source_bundle.segments[name])) for name in SEGMENT_ORDER
    }
    source_segment_bytes = {
        name: len(bytes(source_bundle.segments[name])) for name in SEGMENT_ORDER
    }
    members = {name: bytes(value) for name, value in expansion.members.items()}
    manifest = dict(expansion.manifest)
    runtime_members = manifest.get("runtime_members", {})
    qpost_meta = manifest.get("qpost", {})
    qpost_stream_meta = qpost_meta.get("streams", {})
    qpost_streams, qpost_facts = _split_qpost_streams(members.get("qpost.bin", b""))
    runtime_specs = _literal_int_tuple_sequence(
        runtime_dir / "apply_qzs3_postprocess.py",
        "PR82_QRM1_RANDMULTI_SPECS",
    )
    qrm1_facts = _qrm1_stream_facts(qpost_streams.get("randmulti", b""), runtime_specs)
    qpost_text = _safe_text(runtime_dir / "apply_qzs3_postprocess.py")
    prints_apply_summary = (
        "print(json.dumps" in qpost_text
        and '"records"' in qpost_text
        and '"qpost"' in qpost_text
    )

    source_sha = _sha256_bytes(raw)
    source_payload_ok = (
        manifest.get("source_payload_sha256") == source_sha
        and manifest.get("source_payload_bytes") == len(raw)
    )
    member_manifest_mismatches = []
    for name, data in members.items():
        row = runtime_members.get(name, {})
        if row.get("sha256") != _sha256_bytes(data) or row.get("bytes") != len(data):
            member_manifest_mismatches.append(name)
    qpost_manifest_ok = bool(qpost_facts.get("recognized"))
    qpost_stream_mismatches = []
    for name in QPOST_STREAM_NAMES:
        stream = qpost_streams.get(name)
        meta = qpost_stream_meta.get(name, {})
        if stream is None:
            qpost_stream_mismatches.append(name)
            continue
        if meta.get("sha256") != _sha256_bytes(stream) or meta.get("bytes") != len(stream):
            qpost_stream_mismatches.append(name)
            continue
        if meta.get("source_segment_sha256") != source_segment_sha[name]:
            qpost_stream_mismatches.append(name)
            continue
        if meta.get("source_segment_bytes") != source_segment_bytes[name]:
            qpost_stream_mismatches.append(name)
            continue
        if name != "randmulti" and stream != bytes(source_bundle.segments[name]):
            qpost_stream_mismatches.append(name)
            continue
        if name == "randmulti" and meta.get("transcoded") is not True:
            qpost_stream_mismatches.append(name)
    qpost_member_meta = runtime_members.get("qpost.bin", {})
    qpost_header_manifest_ok = (
        qpost_manifest_ok
        and qpost_member_meta.get("sha256") == _sha256_bytes(members.get("qpost.bin", b""))
        and qpost_member_meta.get("bytes") == len(members.get("qpost.bin", b""))
        and not qpost_stream_mismatches
    )
    runtime_specs_match = runtime_specs == PR85_HEADERLESS_RANDMULTI_SPECS
    qrm1_consumable = (
        bool(qrm1_facts.get("recognized"))
        and bool(qrm1_facts.get("group_ids_match_pr85_schedule"))
        and runtime_specs_match
    )

    checks = [
        _check(
            "source_payload_sha_matches_archive_member",
            source_payload_ok,
            [
                f"manifest source_payload_sha256={manifest.get('source_payload_sha256')}",
                f"actual source_payload_sha256={source_sha}",
            ],
            None
            if source_payload_ok
            else "PR85 expansion manifest does not match the source member bytes",
        ),
        _check(
            "runtime_member_manifest_matches_bytes",
            not member_manifest_mismatches,
            [
                "expanded runtime member bytes match manifest SHA/size records"
                if not member_manifest_mismatches
                else f"member mismatches: {member_manifest_mismatches}",
            ],
            None
            if not member_manifest_mismatches
            else "expanded runtime member manifest does not match actual bytes",
            mismatched_members=member_manifest_mismatches,
        ),
        _check(
            "qpost_member_header_matches_stream_manifest",
            qpost_header_manifest_ok,
            [
                "qpost.bin QPS1 lengths, stream SHAs, and source segment SHAs match"
                if qpost_header_manifest_ok
                else f"qpost facts={qpost_facts}; stream mismatches={qpost_stream_mismatches}",
            ],
            None
            if qpost_header_manifest_ok
            else "qpost.bin does not faithfully expose PR85 side-channel segment bytes",
            qpost_facts=qpost_facts,
            mismatched_streams=qpost_stream_mismatches,
        ),
        _check(
            "qrm1_runtime_schedule_matches_pr85",
            runtime_specs_match,
            [
                "runtime PR82_QRM1_RANDMULTI_SPECS exactly matches the PR85 72-group schedule"
                if runtime_specs_match
                else "runtime PR82_QRM1_RANDMULTI_SPECS is missing or differs from the PR85 schedule",
            ],
            None
            if runtime_specs_match
            else "QRM1 runtime schedule mismatch would make randmulti atom edits non-consumable",
            runtime_spec_count=len(runtime_specs) if runtime_specs is not None else None,
            expected_spec_count=len(PR85_HEADERLESS_RANDMULTI_SPECS),
        ),
        _check(
            "qrm1_stream_consumable_by_runtime_schedule",
            qrm1_consumable,
            [
                "transcoded randmulti QRM1 stream parses under the runtime schedule"
                if qrm1_consumable
                else f"QRM1 facts={qrm1_facts}",
            ],
            None
            if qrm1_consumable
            else "PR85 randmulti stream is not provably consumable by fixed-runtime QRM1",
            qrm1_facts=qrm1_facts,
        ),
        _check(
            "qpost_apply_summary_observable",
            prints_apply_summary,
            [
                "apply_qzs3_postprocess.py emits a JSON qpost records summary"
                if prints_apply_summary
                else "apply_qzs3_postprocess.py does not expose an apply summary in stdout",
            ],
            None
            if prints_apply_summary
            else "qpost atom application has no durable apply summary/log surface",
        ),
    ]
    return {
        "schema": "pr85_fixed_runtime_atom_substrate_v1",
        "checks": checks,
        "remaining_blockers": [
            {
                "code": check["id"],
                "severity": "blocking",
                "detail": check["blocker"],
            }
            for check in checks
            if check["status"] != "passed"
        ],
    }


def _runtime_report(runtime_dir: Path) -> dict[str, Any]:
    inflate_sh = runtime_dir / "inflate.sh"
    inflate_renderer = runtime_dir / "inflate_renderer.py"
    unpacker = runtime_dir / "unpack_renderer_payload.py"
    qpost = runtime_dir / "apply_qzs3_postprocess.py"
    range_codec = runtime_dir / "range_mask_codec.cpp"

    texts = {
        "inflate.sh": _safe_text(inflate_sh),
        "inflate_renderer.py": _safe_text(inflate_renderer),
        "unpack_renderer_payload.py": _safe_text(unpacker),
        "apply_qzs3_postprocess.py": _safe_text(qpost),
    }
    functions = {
        "inflate_renderer.py": _function_names(inflate_renderer),
        "unpack_renderer_payload.py": _function_names(unpacker),
        "apply_qzs3_postprocess.py": _function_names(qpost),
    }

    inflate_renderer_funcs = set(functions["inflate_renderer.py"].get("functions", []))
    unpacker_funcs = set(functions["unpack_renderer_payload.py"].get("functions", []))
    qpost_funcs = set(functions["apply_qzs3_postprocess.py"].get("functions", []))

    lower_runtime = "\n".join(texts.values()).lower()
    inflate_sh_lower = texts["inflate.sh"].lower()
    unpacker_lower = texts["unpack_renderer_payload.py"].lower()
    qpost_randmulti_source = _extract_function_source(qpost, "_decode_randmulti")

    has_pr85_named_support = "pr85" in lower_runtime
    has_single_x_dispatch = (
        has_pr85_named_support
        and (
            "parse_pr85" in lower_runtime
            or "pr85_bundle" in lower_runtime
            or "member named ``x``" in lower_runtime
            or 'member named "x"' in lower_runtime
        )
    )
    has_unpacked_pr85_members = (
        has_pr85_named_support
        and "masks.qma9" in unpacker_lower
        and "qpost.bin" in unpacker_lower
        and "renderer.bin" in unpacker_lower
    )
    qma9_decode_ready = (
        "_load_masks_from_qma9" in inflate_renderer_funcs
        and range_codec.is_file()
        and "masks.qma9" in texts["inflate_renderer.py"]
    )
    qh0_loader_ready = "QH0" in texts["inflate_renderer.py"]
    p1d1_pose_ready = "P1D1" in texts["inflate_renderer.py"] or "P1D1" in texts["unpack_renderer_payload.py"]
    qpost_runtime_ready = (
        {"read_qpost", "apply_qpost_to_raw", "_decode_randmulti"} <= qpost_funcs
        and "qpost.bin" in inflate_sh_lower
    )
    pr85_sidechannels_exposed = (
        has_pr85_named_support
        and "qpost.bin" in lower_runtime
        and all(name in lower_runtime for name in ("post", "shift", "frac", "randmulti"))
    )
    randmulti_direct_ready = (
        "PR85" in qpost_randmulti_source
        or "PR85_HEADERLESS" in qpost_randmulti_source
        or "PR82_QRM1_RANDMULTI_SPECS" in qpost_randmulti_source
        or "(224, 222, 4, 1)" in qpost_randmulti_source
    )
    randmulti_qrm1_ready = "QRM1" in texts["apply_qzs3_postprocess.py"] and "_decode_qrm1_randmulti" in qpost_funcs
    pr85_randmulti_ready = pr85_sidechannels_exposed and (randmulti_direct_ready or randmulti_qrm1_ready)

    capabilities = [
        _capability(
            "pr85_single_member_x_dispatch",
            has_single_x_dispatch,
            [
                "PR85 named adapter found in robust_current source"
                if has_single_x_dispatch
                else "No robust_current source path names tac.pr85_bundle/parse_pr85 or a PR85 x-member adapter",
            ],
            None
            if has_single_x_dispatch
            else "robust_current inflate/unpack currently ignores the public PR85 single ZIP member 'x'",
        ),
        _capability(
            "pr85_bundle_expands_to_runtime_members",
            has_unpacked_pr85_members,
            [
                "PR85 adapter materializes renderer/mask/qpost runtime members"
                if has_unpacked_pr85_members
                else "unpack_renderer_payload.py has fixed-slice public payload handlers, but no PR85 v5 x-bundle expander",
            ],
            None
            if has_unpacked_pr85_members
            else "PR85 segments are never materialized as renderer.bin, masks.qma9, optimized poses, or qpost.bin",
        ),
        _capability(
            "qma9_mask_decode_available",
            qma9_decode_ready,
            [
                "inflate_renderer.py::_load_masks_from_qma9 and range_mask_codec.cpp are present"
                if qma9_decode_ready
                else "QMA9 mask decode support is missing or incomplete",
            ],
            None if qma9_decode_ready else "PR85 mask segment cannot be consumed as masks.qma9",
        ),
        _capability(
            "qh0_model_loader_available",
            qh0_loader_ready,
            [
                "inflate_renderer.py contains QH0 model-loader support"
                if qh0_loader_ready
                else "inflate_renderer.py has no QH0 model-loader token",
            ],
            None if qh0_loader_ready else "PR85 model segment decodes to QH0, but robust_current _load_renderer does not load QH0",
        ),
        _capability(
            "p1d1_pose_loader_available",
            p1d1_pose_ready,
            [
                "P1D1 pose codec support is visible in robust_current runtime"
                if p1d1_pose_ready
                else "No P1D1 token found in inflate_renderer.py or unpack_renderer_payload.py",
            ],
            None if p1d1_pose_ready else "PR85 pose segment decodes to P1D1, but robust_current has no fixed-runtime P1D1 pose path",
        ),
        _capability(
            "qpost_runtime_available",
            qpost_runtime_ready,
            [
                "apply_qzs3_postprocess.py can apply qpost.bin after renderer inflate"
                if qpost_runtime_ready
                else "qpost postprocess runtime functions or inflate.sh qpost dispatch are missing",
            ],
            None if qpost_runtime_ready else "PR85 side-channel streams cannot be applied through qpost runtime",
        ),
        _capability(
            "pr85_sidechannels_exposed_to_qpost",
            pr85_sidechannels_exposed,
            [
                "PR85 adapter exposes side-channel streams to qpost.bin"
                if pr85_sidechannels_exposed
                else "qpost.bin is only applied when already present as a separate archive member",
            ],
            None
            if pr85_sidechannels_exposed
            else "PR85 post/shift/frac/bias/region/randmulti streams remain embedded in x and are not handed to qpost",
        ),
        _capability(
            "pr85_randmulti_v5_consumption",
            pr85_randmulti_ready,
            [
                "PR85 randmulti is either decoded directly or transcoded to a supported QRM1 qpost stream"
                if pr85_randmulti_ready
                else "QRM1 helpers exist, but no PR85 adapter exposes/transcodes the v5 headerless randmulti stream",
            ],
            None
            if pr85_randmulti_ready
            else "PR85 randmulti decoded payload uses the v5 headerless schedule and is not wired into robust_current consumption",
        ),
    ]

    return {
        "runtime_dir": _repo_rel(runtime_dir),
        "files": _runtime_file_rows(runtime_dir),
        "functions": functions,
        "capabilities": capabilities,
    }


def _bridge_capability(
    capability_id: str,
    passed: bool,
    evidence: list[str],
    blocker: str | None = None,
) -> dict[str, Any]:
    row = _capability(capability_id, passed, evidence, blocker)
    row["surface"] = "expanded_fixed_runtime_candidate"
    return row


def _fixed_runtime_bridge_report(
    raw: bytes,
    runtime_dir: Path,
    runtime: dict[str, Any],
) -> dict[str, Any]:
    runtime_caps = {row["id"]: row for row in runtime["capabilities"]}
    qma9_ready = runtime_caps.get("qma9_mask_decode_available", {}).get("status") == "passed"
    qh0_ready = runtime_caps.get("qh0_model_loader_available", {}).get("status") == "passed"
    qpost_ready = runtime_caps.get("qpost_runtime_available", {}).get("status") == "passed"
    qpost_text = _safe_text(runtime_dir / "apply_qzs3_postprocess.py")
    qrm1_runtime_ready = "QRM1" in qpost_text and "_decode_qrm1_randmulti" in qpost_text

    expansion_error = None
    expansion_manifest: dict[str, Any] | None = None
    member_names: set[str] = set()
    renderer_qh0 = False
    pose_raw_fp16 = False
    qpost_qps1 = False
    qpost_sidechannels = False
    randmulti_qrm1 = False
    expansion: Any | None = None
    try:
        expansion = expand_pr85_bundle_to_runtime_members(raw, transcode_randmulti_qrm1=True)
        expansion_manifest = dict(expansion.manifest)
        member_names = set(expansion.members)
        renderer_qh0 = expansion.members["renderer.bin"].startswith(b"QH0")
        pose_raw_fp16 = (
            len(expansion.members["optimized_poses.bin"]) == 600 * 6 * 2
            and expansion_manifest["pose"]["codec"] == "P1D1_to_raw_fp16"
        )
        qpost_qps1 = expansion.members["qpost.bin"].startswith(b"QPS1")
        qpost_sidechannels = set(QPOST_STREAM_NAMES) == set(
            expansion_manifest["qpost"]["stream_order"]
        )
        randmulti_qrm1 = (
            expansion_manifest["qpost"]["randmulti"].get("runtime_contract")
            == "QRM1_sparse_group_id_stream"
        )
    except Exception as exc:  # pragma: no cover - diagnostic surface
        expansion_error = str(exc)

    atom_substrate = _bridge_substrate_report(raw, expansion, runtime_dir)

    required_members = {"masks.qma9", "renderer.bin", "optimized_poses.bin", "qpost.bin"}
    expansion_ok = expansion_error is None
    materializes_members = expansion_ok and required_members <= member_names
    sidechannels_ready = expansion_ok and qpost_qps1 and qpost_sidechannels
    p1d1_materialized = expansion_ok and pose_raw_fp16
    randmulti_ready = sidechannels_ready and randmulti_qrm1 and qrm1_runtime_ready

    capabilities = [
        _bridge_capability(
            "pr85_single_member_x_dispatch",
            materializes_members,
            [
                "Owned PR85 bridge helper materializes source member 'x' into standard runtime members"
                if materializes_members
                else f"PR85 bridge expansion failed before replacing member 'x': {expansion_error}",
            ],
            None
            if materializes_members
            else "PR85 source member 'x' still has no fixed-runtime bridge expansion",
        ),
        _bridge_capability(
            "pr85_bundle_expands_to_runtime_members",
            materializes_members,
            [
                f"Expanded members: {sorted(member_names)}"
                if materializes_members
                else "No expanded renderer/mask/pose/qpost member set was produced",
            ],
            None
            if materializes_members
            else "PR85 segments are not materialized as robust_current runtime members",
        ),
        _bridge_capability(
            "qma9_mask_decode_available",
            qma9_ready and materializes_members,
            [
                "masks.qma9 is emitted and robust_current QMA9 mask decode is present"
                if qma9_ready and materializes_members
                else "QMA9 bridge member or runtime decoder is missing",
            ],
            None
            if qma9_ready and materializes_members
            else "PR85 mask segment cannot be consumed as masks.qma9",
        ),
        _bridge_capability(
            "qh0_model_loader_available",
            qh0_ready and renderer_qh0,
            [
                "renderer.bin is decoded QH0 and robust_current advertises QH0 loader support"
                if qh0_ready and renderer_qh0
                else "renderer.bin decodes to QH0, but robust_current still lacks a QH0 loader",
            ],
            None
            if qh0_ready and renderer_qh0
            else "PR85 model segment decodes to QH0, but robust_current _load_renderer does not load QH0",
        ),
        _bridge_capability(
            "p1d1_pose_loader_available",
            p1d1_materialized,
            [
                "P1D1 pose segment is materialized to raw fp16 optimized_poses.bin before inflate"
                if p1d1_materialized
                else "P1D1 pose segment was not materialized to raw fp16 optimized_poses.bin",
            ],
            None
            if p1d1_materialized
            else "PR85 pose segment remains in P1D1 form with no fixed-runtime pose path",
        ),
        _bridge_capability(
            "qpost_runtime_available",
            qpost_ready and sidechannels_ready,
            [
                "qpost.bin is emitted as QPS1 and robust_current applies qpost.bin after renderer inflate"
                if qpost_ready and sidechannels_ready
                else "qpost.bin bridge member or qpost runtime application is missing",
            ],
            None
            if qpost_ready and sidechannels_ready
            else "PR85 side-channel streams cannot be applied through qpost runtime",
        ),
        _bridge_capability(
            "pr85_sidechannels_exposed_to_qpost",
            sidechannels_ready,
            [
                "post/shift/frac/bias/region/randmulti streams are exposed in qpost.bin"
                if sidechannels_ready
                else "PR85 side-channel streams were not exposed as QPS1 qpost.bin",
            ],
            None
            if sidechannels_ready
            else "PR85 side-channel streams remain embedded in x",
        ),
        _bridge_capability(
            "pr85_randmulti_v5_consumption",
            randmulti_ready,
            [
                "PR85 headerless randmulti is transcoded to runtime-supported QRM1"
                if randmulti_ready
                else "PR85 randmulti was not transcoded to supported QRM1 or QRM1 runtime is missing",
            ],
            None
            if randmulti_ready
            else "PR85 randmulti v5 stream is not consumable by fixed-runtime qpost",
        ),
    ]

    return {
        "schema": "pr85_fixed_runtime_bridge_probe_v1",
        "expansion_available": expansion_ok,
        "expansion_error": expansion_error,
        "expansion_manifest": expansion_manifest,
        "atom_substrate": atom_substrate,
        "capabilities": capabilities,
        "remaining_blockers": [
            {
                "code": cap["id"],
                "severity": "blocking",
                "detail": cap["blocker"],
            }
            for cap in capabilities
            if cap["status"] != "passed"
        ],
    }


def _custody_expectation_report(
    archive_info: dict[str, Any],
    *,
    expected_archive_sha256: str | None,
    expected_member_sha256: str | None,
) -> dict[str, Any]:
    checks = []
    if expected_archive_sha256:
        actual = archive_info.get("archive_sha256")
        checks.append(
            _check(
                "expected_archive_sha256_matches",
                actual == expected_archive_sha256,
                [f"expected={expected_archive_sha256}", f"actual={actual}"],
                None if actual == expected_archive_sha256 else "archive SHA-256 does not match requested readiness target",
            )
        )
    if expected_member_sha256:
        actual = archive_info.get("member_sha256")
        checks.append(
            _check(
                "expected_member_sha256_matches",
                actual == expected_member_sha256,
                [f"expected={expected_member_sha256}", f"actual={actual}"],
                None if actual == expected_member_sha256 else "PR85 member payload SHA-256 does not match requested readiness target",
            )
        )
    return {
        "schema": "pr85_readiness_custody_expectations_v1",
        "required": bool(expected_archive_sha256 or expected_member_sha256),
        "checks": checks,
        "remaining_blockers": [
            {
                "code": check["id"],
                "severity": "blocking",
                "detail": check["blocker"],
            }
            for check in checks
            if check["status"] != "passed"
        ],
    }


def _atom_edit_report(raw: bytes, source_archive: Path | None) -> dict[str, Any]:
    if source_archive is None:
        return {
            "schema": "pr85_atom_edit_non_noop_guard_v1",
            "required": False,
            "status": "not_requested",
            "ready_for_atom_edit_exact_eval": None,
            "changed_segments": [],
            "remaining_blockers": [],
        }
    source_info, source_raw = _read_pr85_archive(source_archive)
    source_bundle = parse_pr85_bundle(source_raw)
    candidate_bundle = parse_pr85_bundle(raw)
    changed_segments = []
    for name in SEGMENT_ORDER:
        source_segment = bytes(source_bundle.segments[name])
        candidate_segment = bytes(candidate_bundle.segments[name])
        if source_segment != candidate_segment:
            changed_segments.append(
                {
                    "segment": name,
                    "runtime_surfaces": list(SEGMENT_RUNTIME_SURFACES[name]),
                    "source_bytes": len(source_segment),
                    "candidate_bytes": len(candidate_segment),
                    "source_sha256": _sha256_bytes(source_segment),
                    "candidate_sha256": _sha256_bytes(candidate_segment),
                }
            )
    no_op = source_raw == raw or not changed_segments
    blockers = []
    if no_op:
        blockers.append(
            {
                "code": "pr85_atom_edit_noop_source_preserving",
                "severity": "blocking",
                "detail": "candidate PR85 payload preserves every source segment; no charged atom edit can be consumed",
            }
        )
    return {
        "schema": "pr85_atom_edit_non_noop_guard_v1",
        "required": True,
        "status": "passed" if not blockers else "failed",
        "ready_for_atom_edit_exact_eval": not blockers,
        "source_archive": source_info,
        "source_payload_sha256": _sha256_bytes(source_raw),
        "candidate_payload_sha256": _sha256_bytes(raw),
        "payload_sha256_changed": _sha256_bytes(source_raw) != _sha256_bytes(raw),
        "changed_segment_count": len(changed_segments),
        "changed_segments": changed_segments,
        "remaining_blockers": blockers,
    }


def build_preflight(
    archive: Path,
    runtime_dir: Path,
    *,
    atom_source_archive: Path | None = None,
    expected_archive_sha256: str | None = None,
    expected_member_sha256: str | None = None,
) -> dict[str, Any]:
    archive_info, raw = _read_pr85_archive(archive)
    bundle = _bundle_report(raw)
    runtime = _runtime_report(runtime_dir)
    bridge = _fixed_runtime_bridge_report(raw, runtime_dir, runtime)
    custody = _custody_expectation_report(
        archive_info,
        expected_archive_sha256=expected_archive_sha256,
        expected_member_sha256=expected_member_sha256,
    )
    atom_edit = _atom_edit_report(raw, atom_source_archive)

    blockers = []
    if not bundle["roundtrip_matches_input"]:
        blockers.append(
            {
                "code": "pr85_bundle_roundtrip_mismatch",
                "severity": "blocking",
                "detail": "tac.pr85_bundle could not losslessly repack the parsed input bytes",
            }
        )
    if not bundle["cheap_segment_probe_ok"]:
        blockers.append(
            {
                "code": "pr85_segment_probe_failed",
                "severity": "blocking",
                "detail": "one or more PR85 segments failed cheap magic/decompression/schema validation",
            }
        )
    for row in runtime["files"]:
        if not row["present"]:
            blockers.append(
                {
                    "code": f"missing_runtime_file:{row['name']}",
                    "severity": "blocking",
                    "detail": f"required robust_current runtime file is missing: {row['path']}",
                }
            )
    for cap in bridge["capabilities"]:
        if cap["status"] != "passed":
            blockers.append(
                {
                    "code": cap["id"],
                    "severity": "blocking",
                    "detail": cap["blocker"],
                }
            )
    for row in bridge["atom_substrate"]["remaining_blockers"]:
        blockers.append(
            {
                "code": f"pr85_atom_substrate:{row['code']}",
                "severity": row["severity"],
                "detail": row["detail"],
            }
        )
    for row in custody["remaining_blockers"]:
        blockers.append(
            {
                "code": f"pr85_custody:{row['code']}",
                "severity": row["severity"],
                "detail": row["detail"],
            }
        )
    for row in atom_edit["remaining_blockers"]:
        blockers.append(row)

    blockers = sorted(blockers, key=lambda row: row["code"])
    ready = not blockers
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "evidence_grade": EVIDENCE_GRADE,
        "planning_only": True,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "gpu_required": False,
        "ready_for_fixed_runtime_exact_eval": ready,
        "readiness_status": "ready" if ready else "blocked",
        "archive": archive_info,
        "custody_expectations": custody,
        "atom_edit_guard": atom_edit,
        "bundle": bundle,
        "robust_current_runtime": runtime,
        "fixed_runtime_bridge": bridge,
        "blockers": blockers,
        "non_claim_notes": [
            "This preflight is static/planning-only and does not run archive.zip -> inflate.sh -> upstream/evaluate.py.",
            "A ready result would permit a later claimed/remote exact-eval dispatch only after the lane is claimed separately.",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--robust-current-dir", type=Path, default=DEFAULT_ROBUST_CURRENT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument(
        "--atom-source-archive",
        type=Path,
        default=None,
        help=(
            "When supplied, require the target PR85 archive to change at least one "
            "charged segment versus this source archive."
        ),
    )
    parser.add_argument(
        "--expected-archive-sha256",
        default=None,
        help="Fail closed if the target archive file SHA-256 differs from this value.",
    )
    parser.add_argument(
        "--expected-member-sha256",
        default=None,
        help="Fail closed if the PR85 x member payload SHA-256 differs from this value.",
    )
    parser.add_argument(
        "--fail-if-not-ready",
        action="store_true",
        help="Return exit code 2 when blockers remain. Default is report-only exit 0.",
    )
    args = parser.parse_args(argv)

    payload = build_preflight(
        args.archive,
        args.robust_current_dir,
        atom_source_archive=args.atom_source_archive,
        expected_archive_sha256=args.expected_archive_sha256,
        expected_member_sha256=args.expected_member_sha256,
    )
    text = _json_text(payload)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text, end="")
    if args.fail_if_not_ready and not payload["ready_for_fixed_runtime_exact_eval"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
