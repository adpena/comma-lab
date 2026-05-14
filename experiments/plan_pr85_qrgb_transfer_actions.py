#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan PR90 QRGB residual controls as PR85 action evidence.

This is a deterministic, local-only planner. It parses PR90's packed QRGB
residual stream and PR85's existing side-channel streams, then emits PR85
stream/value action candidates. It does not build archives, import scorers, run
GPU code, edit dispatch state, or unlock exact eval dispatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import brotli
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.pr85_bundle import (  # noqa: E402
    PR85_HEADERLESS_RANDMULTI_SPECS,
    SEGMENT_ORDER,
    parse_pr85_bundle,
    validate_pr85_member_name,
)


TOOL = "experiments/plan_pr85_qrgb_transfer_actions.py"
SCHEMA = "pr85_qrgb_transfer_action_plan_v1"
PAIR_ACTION_EVIDENCE_SCHEMA = "pr85_pair_action_lowering_evidence_v1"
PAIR_COUNT = 600
ORIGINAL_VIDEO_BYTES = 37_545_489
RATE_SCORE_PER_BYTE = 25.0 / ORIGINAL_VIDEO_BYTES

PR85_ARCHIVE_SHA256 = "eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e"
PR85_ARCHIVE_BYTES = 236_328
PR85_SCORE_CONTEXT = 0.25806611029397786
PR90_ARCHIVE_SHA256 = "608ea0355e60faad97b046c27644205d05120ac85ab3e8a99543a75a4ab2dd2d"
PR90_ARCHIVE_BYTES = 218_080

DEFAULT_PR85_ARCHIVE = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex/archive.zip"
DEFAULT_PR90_ARCHIVE = REPO_ROOT / "experiments/results/public_pr90_intake_20260504_worker/archive.zip"
DEFAULT_PR90_PROBE = REPO_ROOT / "experiments/results/public_pr90_intake_20260504_worker/payload_probe.json"
DEFAULT_PAIR_PLANNING = REPO_ROOT / "experiments/results/pr85_pair_atom_candidates_20260504_orchestrator/planning.json"
DEFAULT_OUT_JSON = REPO_ROOT / "experiments/results/pr85_qrgb_transfer_actions_20260504_worker/planning.json"
DEFAULT_PAIR_ACTION_EVIDENCE_JSON = (
    REPO_ROOT / "experiments/results/pr85_qrgb_transfer_actions_20260504_worker/pair_action_evidence.json"
)
DEFAULT_LEDGER = REPO_ROOT / ".omx/research/pr85_qrgb_transfer_worker_20260504.md"

RGB_COMPACT_PLANE_ORDER = (
    34, 33, 29, 45, 47, 44, 35, 31, 26, 14, 38, 27,
    30, 15, 42, 25, 32, 8, 43, 28, 4, 17, 46, 24,
    1, 41, 21, 12, 36, 39, 13, 9, 10, 7, 16, 20,
    19, 37, 6, 3, 18, 23, 11, 22, 40, 5, 0, 2,
)
VALS_AMP2 = (-2, -1, 1, 2)
VALS_AMP4 = (-4, -3, -2, -1, 1, 2, 3, 4)


class QrgbTransferPlanError(ValueError):
    """Raised for malformed planner inputs or unsupported source artifacts."""


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json_text(payload), encoding="utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_digest(payload: Mapping[str, Any]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key != "stable_plan_digest_sha256"
    }
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise QrgbTransferPlanError(f"{_rel(path)} must contain a JSON object")
    return payload


def _read_single_member_zip(path: Path, *, expected_member: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise QrgbTransferPlanError(f"missing archive: {_rel(path)}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != [expected_member]:
            raise QrgbTransferPlanError(
                f"{_rel(path)} must contain exactly {expected_member!r}, got {names!r}"
            )
        if expected_member == "x":
            validate_pr85_member_name(infos[0].filename)
        raw = zf.read(infos[0])
    return (
        {
            "path": _rel(path),
            "archive_bytes": int(path.stat().st_size),
            "archive_sha256": _sha256_file(path),
            "member_name": expected_member,
            "member_bytes": int(len(raw)),
            "member_sha256": _sha256(raw),
            "zip_stored": infos[0].compress_type == zipfile.ZIP_STORED,
        },
        raw,
    )


def _read_varints(raw: bytes, pos: int, count: int) -> tuple[list[int], int]:
    out: list[int] = []
    for _ in range(count):
        acc = 0
        shift = 0
        while True:
            if pos >= len(raw):
                raise QrgbTransferPlanError("truncated varint stream")
            byte = int(raw[pos])
            pos += 1
            acc |= (byte & 0x7F) << shift
            if byte & 0x80:
                shift += 7
                if shift > 63:
                    raise QrgbTransferPlanError("overlong varint stream")
            else:
                out.append(acc)
                break
    return out, pos


def _decode_qrgb_residual(qrgb_br: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    raw = brotli.decompress(qrgb_br)
    if len(raw) % 2:
        raise QrgbTransferPlanError("PR90 QRGB residual raw stream must be gap/value pairs")
    nonzero = len(raw) // 2
    gaps = np.frombuffer(raw, dtype=np.uint8, count=nonzero, offset=0).astype(np.int32)
    values = np.frombuffer(raw, dtype=np.int8, count=nonzero, offset=nonzero).astype(np.int16)
    positions = np.cumsum(gaps + 1, dtype=np.int32) - 1
    if nonzero and (int(positions[-1]) >= 48 * PAIR_COUNT or int(positions[0]) < 0):
        raise QrgbTransferPlanError("PR90 QRGB sparse residual position outside 48x600 basis")
    flat = np.zeros(48 * PAIR_COUNT, dtype=np.int16)
    flat[positions] = values
    ordered_planes = flat.reshape(48, PAIR_COUNT)
    planes = np.empty_like(ordered_planes)
    for source_plane, semantic_plane in enumerate(RGB_COMPACT_PLANE_ORDER):
        planes[semantic_plane] = ordered_planes[source_plane]

    per_pair = np.count_nonzero(planes, axis=0)
    nonzero_abs = np.abs(values) if nonzero else np.array([], dtype=np.int16)
    stats = {
        "compressed_bytes": int(len(qrgb_br)),
        "compressed_sha256": _sha256(qrgb_br),
        "raw_bytes": int(len(raw)),
        "raw_sha256": _sha256(raw),
        "nonzero_values": int(nonzero),
        "basis_planes": 48,
        "pair_count": PAIR_COUNT,
        "mean_edits_per_pair": float(nonzero / PAIR_COUNT),
        "max_edits_per_pair": int(per_pair.max()) if per_pair.size else 0,
        "value_min": int(values.min()) if nonzero else 0,
        "value_max": int(values.max()) if nonzero else 0,
        "abs_value_mean": float(nonzero_abs.mean()) if nonzero else 0.0,
        "compressed_bytes_per_pair": float(len(qrgb_br) / PAIR_COUNT),
        "compressed_bytes_per_nonzero": float(len(qrgb_br) / nonzero) if nonzero else None,
    }
    return planes, stats


def _load_pr90_qrgb(pr90_archive: Path, pr90_probe_json: Path) -> tuple[dict[str, Any], np.ndarray]:
    archive_meta, payload = _read_single_member_zip(pr90_archive, expected_member="p")
    if archive_meta["archive_sha256"] != PR90_ARCHIVE_SHA256:
        archive_meta["known_pr90_anchor_match"] = {
            "matches": False,
            "expected_archive_sha256": PR90_ARCHIVE_SHA256,
            "expected_archive_bytes": PR90_ARCHIVE_BYTES,
        }
    else:
        archive_meta["known_pr90_anchor_match"] = {
            "matches": archive_meta["archive_bytes"] == PR90_ARCHIVE_BYTES,
            "expected_archive_sha256": PR90_ARCHIVE_SHA256,
            "expected_archive_bytes": PR90_ARCHIVE_BYTES,
        }
    probe = _read_json(pr90_probe_json)
    qslice = ((probe.get("slices") or {}).get("qrgb_residual_br") or {})
    offset = qslice.get("offset")
    length = qslice.get("len")
    if not isinstance(offset, int) or not isinstance(length, int) or offset < 0 or length <= 0:
        raise QrgbTransferPlanError("payload_probe.json lacks a valid qrgb_residual_br slice")
    qrgb_br = payload[offset : offset + length]
    if len(qrgb_br) != length:
        raise QrgbTransferPlanError("PR90 QRGB residual slice is truncated")
    planes, qrgb_stats = _decode_qrgb_residual(qrgb_br)
    return (
        {
            **archive_meta,
            "payload_probe": {
                "path": _rel(pr90_probe_json),
                "sha256": _sha256_file(pr90_probe_json),
            },
            "payload_sha256": probe.get("payload_sha256"),
            "payload_len": probe.get("payload_len"),
            "qrgb_slice": {
                "offset": offset,
                "bytes": length,
            },
            "qrgb_residual": qrgb_stats,
        },
        planes,
    )


def _decode_dense_choice(raw: bytes, *, default_center: int | None = None) -> np.ndarray:
    values = np.frombuffer(raw, dtype=np.uint8, offset=3).astype(np.int64)
    if values.size != PAIR_COUNT:
        raise QrgbTransferPlanError(f"dense choice stream has {values.size} values")
    if raw[:3] in {b"SH4", b"FH1", b"FH2", b"FH3", b"BH1", b"RH1"}:
        return values
    if raw[:3] in {b"SD4", b"FD3", b"BD1", b"RD1"}:
        if default_center is None:
            raise QrgbTransferPlanError(f"default center required for {raw[:3]!r}")
        return np.where(values == 0, default_center, values - 1).astype(np.int64)
    raise QrgbTransferPlanError(f"unsupported dense choice magic {raw[:3]!r}")


def _decode_sparse_choice(raw: bytes, *, magic: bytes, default_choice: int) -> np.ndarray:
    if not raw.startswith(magic) or len(raw) < 5:
        raise QrgbTransferPlanError(f"bad sparse choice magic for {magic!r}")
    count = int.from_bytes(raw[3:5], "little")
    pos = 5
    gaps, pos = _read_varints(raw, pos, count)
    values = np.frombuffer(raw, dtype=np.uint8, count=count, offset=pos).astype(np.int64)
    if pos + count != len(raw):
        raise QrgbTransferPlanError("sparse choice stream has trailing or truncated bytes")
    out = np.full(PAIR_COUNT, default_choice, dtype=np.int64)
    index = -1
    for gap, value in zip(gaps, values, strict=True):
        index += gap + 1
        if not 0 <= index < PAIR_COUNT:
            raise QrgbTransferPlanError(f"sparse choice index outside pair range: {index}")
        out[index] = int(value) - 1
    return out


def _decode_choice_stream(name: str, segment: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    raw = brotli.decompress(segment)
    magic = raw[:3]
    if name == "shift":
        values = _decode_dense_choice(raw, default_center=40)
        default = 40
    elif name == "frac":
        values = _decode_sparse_choice(raw, magic=b"FV1", default_choice=4) if magic == b"FV1" else _decode_dense_choice(raw)
        default = 4
    elif name == "frac2":
        values = _decode_dense_choice(raw)
        default = 4
    elif name == "frac3":
        values = _decode_dense_choice(raw, default_center=4)
        default = 4
    elif name == "bias":
        values = _decode_sparse_choice(raw, magic=b"BV1", default_choice=13) if magic == b"BV1" else _decode_dense_choice(raw, default_center=13)
        default = 13
    elif name == "region":
        values = _decode_sparse_choice(raw, magic=b"RV1", default_choice=0) if magic == b"RV1" else _decode_dense_choice(raw, default_center=0)
        default = 0
    else:
        raise QrgbTransferPlanError(f"unsupported choice stream {name!r}")
    unique, counts = np.unique(values, return_counts=True)
    dominant_index = int(np.argmax(counts)) if counts.size else 0
    stats = {
        "magic": magic.decode("ascii", errors="replace"),
        "default_choice": default,
        "raw_bytes": int(len(segment)),
        "decoded_bytes": int(len(raw)),
        "decoded_sha256": _sha256(raw),
        "unique_count": int(unique.size),
        "dominant_symbol": int(unique[dominant_index]) if unique.size else None,
        "dominant_symbol_count": int(counts[dominant_index]) if counts.size else 0,
        "nondefault_count": int(np.count_nonzero(values != default)),
    }
    return values, stats


def _decode_randmulti(segment: bytes) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = brotli.decompress(segment)
    groups: list[dict[str, Any]] = []
    pos = 0
    for group_index, (lh, lw, amp, selection_rows) in enumerate(PR85_HEADERLESS_RANDMULTI_SPECS):
        rows = np.zeros((selection_rows, PAIR_COUNT), dtype=np.int64)
        for row_index in range(selection_rows):
            if pos >= len(raw):
                raise QrgbTransferPlanError("truncated PR85 randmulti stream")
            count = int(raw[pos])
            pos += 1
            if count == 255:
                if pos + 2 > len(raw):
                    raise QrgbTransferPlanError("truncated PR85 randmulti extended count")
                count = int.from_bytes(raw[pos : pos + 2], "little")
                pos += 2
            gaps, pos = _read_varints(raw, pos, count)
            values = np.frombuffer(raw, dtype=np.uint8, count=count, offset=pos).astype(np.int64)
            pos += count
            index = -1
            for gap, value in zip(gaps, values, strict=True):
                index += gap + 1
                if not 0 <= index < PAIR_COUNT:
                    raise QrgbTransferPlanError("PR85 randmulti sparse index outside pair range")
                rows[row_index, index] = int(value)
        groups.append(
            {
                "group_index": group_index,
                "lh": int(lh),
                "lw": int(lw),
                "amp": int(amp),
                "selection_rows": int(selection_rows),
                "rows": rows,
                "nonzero_entries": int(np.count_nonzero(rows)),
            }
        )
    if pos != len(raw):
        raise QrgbTransferPlanError("PR85 randmulti stream has trailing bytes")
    stats = {
        "raw_bytes": int(len(segment)),
        "decoded_bytes": int(len(raw)),
        "decoded_sha256": _sha256(raw),
        "group_count": len(groups),
        "nonzero_entries": int(sum(group["nonzero_entries"] for group in groups)),
    }
    return groups, stats


def _load_pr85_streams(pr85_archive: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, Any], list[dict[str, Any]]]:
    archive_meta, member = _read_single_member_zip(pr85_archive, expected_member="x")
    archive_meta["known_pr85_anchor_match"] = {
        "matches": (
            archive_meta["archive_sha256"] == PR85_ARCHIVE_SHA256
            and archive_meta["archive_bytes"] == PR85_ARCHIVE_BYTES
        ),
        "expected_archive_sha256": PR85_ARCHIVE_SHA256,
        "expected_archive_bytes": PR85_ARCHIVE_BYTES,
        "exact_t4_score_context": PR85_SCORE_CONTEXT,
    }
    bundle = parse_pr85_bundle(member)
    values: dict[str, np.ndarray] = {}
    segment_stats: dict[str, Any] = {
        "bundle_format": bundle.format,
        "header_bytes": bundle.header_bytes,
        "segments": {},
    }
    for name in SEGMENT_ORDER:
        segment = bundle.segments[name]
        row = {
            "raw_bytes": int(len(segment)),
            "raw_sha256": _sha256(segment),
            "offset": int(bundle.segment_offsets[name]),
        }
        if name in {"shift", "frac", "frac2", "frac3", "bias", "region"}:
            decoded, stats = _decode_choice_stream(name, segment)
            values[name] = decoded
            row.update(stats)
        elif name == "randmulti":
            groups, stats = _decode_randmulti(segment)
            row.update({key: value for key, value in stats.items() if key != "raw_bytes"})
        elif name == "post":
            decoded = brotli.decompress(segment)
            row.update(
                {
                    "decoded_bytes": int(len(decoded)),
                    "decoded_sha256": _sha256(decoded),
                    "stage_count": int(len(decoded) // PAIR_COUNT) if len(decoded) % PAIR_COUNT == 0 else None,
                }
            )
        segment_stats["segments"][name] = row
    return archive_meta, values, segment_stats, groups


def _find_randmulti_group(groups: Sequence[Mapping[str, Any]], spec: tuple[int, int, int, int]) -> Mapping[str, Any] | None:
    lh, lw, amp, selection_rows = spec
    for group in groups:
        if (
            group.get("lh") == lh
            and group.get("lw") == lw
            and group.get("amp") == amp
            and group.get("selection_rows") == selection_rows
        ):
            return group
    return None


def _sign_unit(value: int) -> int:
    return -1 if value < 0 else 1 if value > 0 else 0


def _clamped_signed(value: int, vals: Sequence[int]) -> int:
    if value == 0:
        return 0
    sign = -1 if value < 0 else 1
    magnitude = min(max(abs(int(value)), 1), max(abs(item) for item in vals))
    candidate = sign * magnitude
    if candidate in vals:
        return candidate
    # VALS_AMP2 and VALS_AMP4 skip zero only; this keeps the fallback explicit.
    return sign * min(abs(item) for item in vals if (item < 0) == (sign < 0))


def _value_index(value: int, vals: Sequence[int]) -> int:
    try:
        return list(vals).index(int(value))
    except ValueError as exc:
        raise QrgbTransferPlanError(f"value {value!r} is not encodable in {tuple(vals)!r}") from exc


def _encode_bias_choice(rgb_signs: Sequence[int]) -> int:
    r, g, b = (max(-1, min(1, int(value))) for value in rgb_signs)
    return (r + 1) * 9 + (g + 1) * 3 + (b + 1)


def _encode_region_choice(*, mask_index: int, channel_index: int, value: int) -> int:
    if not 0 <= mask_index <= 5:
        raise QrgbTransferPlanError("region mask index outside PR85 runtime range")
    if not 0 <= channel_index <= 3:
        raise QrgbTransferPlanError("region channel index outside PR85 runtime range")
    return 1 + ((mask_index * 4 + channel_index) * 4 + _value_index(value, VALS_AMP2))


def _encode_randglobal_choice(*, channel_type: int, value: int) -> int:
    if not 0 <= channel_type <= 3:
        raise QrgbTransferPlanError("randglobal channel type outside PR85 runtime range")
    return 1 + _value_index(value, VALS_AMP4) * 4 + channel_type


def _encode_randtile_choice(*, tile_index: int, channel_type: int, value: int) -> int:
    if not 0 <= tile_index <= 3:
        raise QrgbTransferPlanError("randtile tile index outside PR85 runtime range")
    if not 0 <= channel_type <= 2:
        raise QrgbTransferPlanError("randtile channel type outside PR85 runtime range")
    return 1 + tile_index * 24 + _value_index(value, VALS_AMP4) * 3 + channel_type


def _load_pair_planning(path: Path, *, top_n: int) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    payload = _read_json(path)
    pair_file_sha = _sha256_file(path)
    scorer_report = payload.get("scorer_gradient_plan") if isinstance(payload.get("scorer_gradient_plan"), dict) else {}
    scorer_path_raw = scorer_report.get("path")
    scorer_plan: dict[str, Any] | None = None
    scorer_file_sha = scorer_report.get("sha256") if isinstance(scorer_report.get("sha256"), str) else None
    stable = None
    if isinstance(scorer_path_raw, str) and scorer_path_raw:
        scorer_path = Path(scorer_path_raw)
        if not scorer_path.is_absolute():
            scorer_path = REPO_ROOT / scorer_path
        if scorer_path.is_file():
            scorer_plan = _read_json(scorer_path)
            scorer_file_sha = _sha256_file(scorer_path)
            stable = scorer_plan.get("stable_plan_digest_sha256")
    top_atoms_raw = None
    if scorer_plan is not None and isinstance(scorer_plan.get("atom_ranking"), list):
        top_atoms_raw = scorer_plan["atom_ranking"]
    elif isinstance(scorer_report.get("top_atoms"), list):
        top_atoms_raw = scorer_report["top_atoms"]
    else:
        top_atoms_raw = []
    top_atoms = [dict(row) for row in top_atoms_raw[:top_n] if isinstance(row, Mapping)]
    source_sha_for_pair_action = (
        stable if isinstance(stable, str) and stable else scorer_file_sha or pair_file_sha
    )
    meta = {
        "path": _rel(path),
        "sha256": pair_file_sha,
        "schema": payload.get("schema"),
        "score_claim": payload.get("score_claim"),
        "dispatch_unlocked": payload.get("dispatch_unlocked"),
        "candidate_archive_count": payload.get("candidate_archive_count"),
        "scorer_gradient_plan": {
            "path": scorer_path_raw,
            "sha256": scorer_file_sha,
            "stable_plan_digest_sha256": stable,
            "declared_sha256": scorer_report.get("sha256"),
        },
        "source_sha_for_pair_action": source_sha_for_pair_action,
    }
    return meta, top_atoms, source_sha_for_pair_action


def _break_even(atom: Mapping[str, Any]) -> dict[str, Any]:
    byte_break_even = atom.get("byte_break_even") if isinstance(atom.get("byte_break_even"), Mapping) else {}
    def section(name: str) -> float | None:
        row = byte_break_even.get(name) if isinstance(byte_break_even, Mapping) else None
        if isinstance(row, Mapping) and isinstance(row.get("max_charged_bytes_for_zero_net_change"), (int, float)):
            return float(row["max_charged_bytes_for_zero_net_change"])
        value = atom.get("break_even_bytes")
        return float(value) if name == "combined" and isinstance(value, (int, float)) else None

    combined = section("combined")
    pose = section("pose_only")
    seg = section("seg_only")
    if pose is None and seg is None:
        dominant = "unknown"
    elif seg is None or (pose is not None and pose >= seg):
        dominant = "pose"
    else:
        dominant = "seg"
    return {
        "combined_max_charged_bytes_for_zero_net_change": combined,
        "pose_only_max_charged_bytes_for_zero_net_change": pose,
        "seg_only_max_charged_bytes_for_zero_net_change": seg,
        "dominant_single_component": dominant,
        "formula_rate_score_per_byte": RATE_SCORE_PER_BYTE,
    }


def _qrgb_pair_summary(planes: np.ndarray, pair_index: int) -> dict[str, Any]:
    values = planes[:, pair_index]
    nonzero_planes = np.nonzero(values)[0]
    top = sorted(
        (
            {"semantic_plane": int(plane), "value": int(values[plane])}
            for plane in nonzero_planes
        ),
        key=lambda row: (-abs(int(row["value"])), int(row["semantic_plane"])),
    )
    return {
        "pair_index": int(pair_index),
        "nonzero_qrgb_edits": int(nonzero_planes.size),
        "max_abs_qrgb_value": int(max((abs(int(values[idx])) for idx in nonzero_planes), default=0)),
        "top_semantic_planes": top[:8],
    }


def _candidate_base(
    *,
    candidate_id: str,
    design_kind: str,
    atom: Mapping[str, Any],
    source_archive: Mapping[str, Any],
    pr90_meta: Mapping[str, Any],
    qrgb_pair: Mapping[str, Any],
    source_sha_for_pair_action: str,
    charged_bytes: int,
) -> dict[str, Any]:
    pair_index = int(atom["pair_index"])
    break_even = _break_even(atom)
    expected = {
        **break_even,
        "proxy_charged_bytes": charged_bytes,
        "break_even_margin_bytes": (
            break_even["combined_max_charged_bytes_for_zero_net_change"] - charged_bytes
            if isinstance(break_even["combined_max_charged_bytes_for_zero_net_change"], (int, float))
            else None
        ),
        "formula_only_rate_score_delta": charged_bytes * RATE_SCORE_PER_BYTE,
        "evidence_status": "formula_only_break_even_from_pr85_component_trace",
    }
    return {
        "candidate_id": candidate_id,
        "design_kind": design_kind,
        "source_archive_sha256": source_archive.get("archive_sha256"),
        "source_archive_bytes": source_archive.get("archive_bytes"),
        "pr90_source_evidence_sha256": pr90_meta["qrgb_residual"]["compressed_sha256"],
        "pr90_source_archive_sha256": pr90_meta.get("archive_sha256"),
        "pair_index": pair_index,
        "frame_indices": atom.get("frame_indices"),
        "source_atom_id": atom.get("atom_id"),
        "targeted_component": "combined",
        "expected_break_even": expected,
        "charged_byte_proxy": {
            "candidate_action_bytes": charged_bytes,
            "basis": (
                "ceil(PR90 observed QRGB residual compressed bytes per frame pair); "
                "archive delta remains unproven until PR85 local recode"
            ),
            "source_artifact_sha256": source_sha_for_pair_action,
            "pr90_qrgb_compressed_bytes_per_pair": pr90_meta["qrgb_residual"]["compressed_bytes_per_pair"],
            "pr90_qrgb_compressed_bytes": pr90_meta["qrgb_residual"]["compressed_bytes"],
        },
        "qrgb_pair_summary": qrgb_pair,
        "archive_changing": False,
        "archive_changing_path": None,
        "dispatch_unlocked": False,
        "can_feed_pair_action_evidence": True,
        "feed_pair_action_evidence_status": "action_spec_only_no_archive_changing_path",
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "blockers": [
            {
                "blocker_class": "no_archive_changing_path",
                "reason": "planner emitted stream/value action evidence only; no PR85 candidate archive was built",
            },
            {
                "blocker_class": "no_pr85_component_response_for_direction",
                "reason": "PR90 QRGB sign gives a transfer direction, not measured PR85 component response",
            },
        ],
    }


def _design_f1_bias(
    *,
    atom: Mapping[str, Any],
    planes: np.ndarray,
    pr85_values: Mapping[str, np.ndarray],
    source_archive: Mapping[str, Any],
    pr90_meta: Mapping[str, Any],
    source_sha_for_pair_action: str,
    charged_bytes: int,
) -> dict[str, Any] | None:
    pair = int(atom["pair_index"])
    rgb = [int(v) for v in planes[0:3, pair]]
    signs = [_sign_unit(v) for v in rgb]
    if not any(signs):
        return None
    candidate_value = _encode_bias_choice(signs)
    source_value = int(pr85_values["bias"][pair])
    if candidate_value == source_value:
        return None
    candidate = _candidate_base(
        candidate_id=f"pr85_qrgb_f1_bias_pair_{pair:04d}",
        design_kind="qrgb_frame0_global_rgb_to_pr85_bias",
        atom=atom,
        source_archive=source_archive,
        pr90_meta=pr90_meta,
        qrgb_pair=_qrgb_pair_summary(planes, pair),
        source_sha_for_pair_action=source_sha_for_pair_action,
        charged_bytes=charged_bytes,
    )
    candidate.update(
        {
            "basis_action_schema": {
                "source_basis": "PR90 semantic planes 0..2: frame0 global RGB offsets",
                "target_stream": "bias",
                "target_runtime_semantics": "PR85 frame0 RGB bias choice; values encode {-1,0,+1} per channel",
                "source_qrgb_rgb_values": rgb,
                "candidate_rgb_signs": signs,
            },
            "actions": [
                {
                    "pair_index": pair,
                    "stream": "bias",
                    "op": "set",
                    "source_value": source_value,
                    "candidate_value": candidate_value,
                    "source_artifact_sha256": source_sha_for_pair_action,
                    "source_atom_id": atom.get("atom_id"),
                    "rationale": "PR90 QRGB frame0 global RGB residual sign transferred to PR85 frame0 bias choice",
                }
            ],
        }
    )
    return candidate


def _design_f2_randglobal(
    *,
    atom: Mapping[str, Any],
    planes: np.ndarray,
    randglobal_group: Mapping[str, Any],
    source_archive: Mapping[str, Any],
    pr90_meta: Mapping[str, Any],
    source_sha_for_pair_action: str,
    charged_bytes: int,
) -> dict[str, Any] | None:
    pair = int(atom["pair_index"])
    rgb = [int(v) for v in planes[3:6, pair]]
    if not any(rgb):
        return None
    channel = max(range(3), key=lambda idx: (abs(rgb[idx]), -idx))
    value = _clamped_signed(rgb[channel], VALS_AMP4)
    if value == 0:
        return None
    candidate_value = _encode_randglobal_choice(channel_type=channel + 1, value=value)
    source_value = int(randglobal_group["rows"][0, pair])
    if candidate_value == source_value:
        return None
    candidate = _candidate_base(
        candidate_id=f"pr85_qrgb_f2_randglobal_pair_{pair:04d}",
        design_kind="qrgb_frame1_global_rgb_to_pr85_randmulti_global_f2",
        atom=atom,
        source_archive=source_archive,
        pr90_meta=pr90_meta,
        qrgb_pair=_qrgb_pair_summary(planes, pair),
        source_sha_for_pair_action=source_sha_for_pair_action,
        charged_bytes=charged_bytes,
    )
    candidate.update(
        {
            "basis_action_schema": {
                "source_basis": "PR90 semantic planes 3..5: frame1 global RGB offsets",
                "target_stream": "randmulti",
                "target_randmulti_group": {
                    "group_index": randglobal_group["group_index"],
                    "lh": randglobal_group["lh"],
                    "lw": randglobal_group["lw"],
                    "amp": randglobal_group["amp"],
                    "selection_row": 0,
                },
                "target_runtime_semantics": "PR85 frame1/f2 global RGB-bias oracle stage, radius 4",
                "source_qrgb_rgb_values": rgb,
                "selected_channel_index": channel,
                "selected_value": value,
            },
            "actions": [
                {
                    "pair_index": pair,
                    "stream": "randmulti",
                    "op": "set",
                    "source_value": source_value,
                    "candidate_value": candidate_value,
                    "source_artifact_sha256": source_sha_for_pair_action,
                    "source_atom_id": atom.get("atom_id"),
                    "rationale": "PR90 QRGB frame1 global RGB residual transferred to PR85 randmulti global f2 bias choice",
                }
            ],
        }
    )
    return candidate


def _best_region_source(planes: np.ndarray, pair: int) -> dict[str, Any] | None:
    options: list[dict[str, Any]] = []
    # PR90 frame0 vertical band planes: frame0 * 6 + channel * 2 + half.
    for channel in range(3):
        for half in range(2):
            plane = 6 + channel * 2 + half
            value = int(planes[plane, pair])
            if value:
                options.append(
                    {
                        "semantic_plane": plane,
                        "value": value,
                        "mask_index": half,  # 0 top, 1 bottom
                        "channel_index": channel + 1,
                        "source_basis": "PR90 frame0 vertical half-band RGB residual",
                    }
                )
    # PR90 frame0 horizontal band planes: frame0 * 6 + channel * 2 + half.
    for channel in range(3):
        for half in range(2):
            plane = 18 + channel * 2 + half
            value = int(planes[plane, pair])
            if value:
                options.append(
                    {
                        "semantic_plane": plane,
                        "value": value,
                        "mask_index": 2 + half,  # 2 left, 3 right
                        "channel_index": channel + 1,
                        "source_basis": "PR90 frame0 horizontal half-band RGB residual",
                    }
                )
    if not options:
        return None
    return sorted(options, key=lambda row: (-abs(int(row["value"])), int(row["semantic_plane"])))[0]


def _design_f1_region(
    *,
    atom: Mapping[str, Any],
    planes: np.ndarray,
    pr85_values: Mapping[str, np.ndarray],
    source_archive: Mapping[str, Any],
    pr90_meta: Mapping[str, Any],
    source_sha_for_pair_action: str,
    charged_bytes: int,
) -> dict[str, Any] | None:
    pair = int(atom["pair_index"])
    selected = _best_region_source(planes, pair)
    if selected is None:
        return None
    value = _clamped_signed(int(selected["value"]), VALS_AMP2)
    if value == 0:
        return None
    candidate_value = _encode_region_choice(
        mask_index=int(selected["mask_index"]),
        channel_index=int(selected["channel_index"]),
        value=value,
    )
    source_value = int(pr85_values["region"][pair])
    if candidate_value == source_value:
        return None
    candidate = _candidate_base(
        candidate_id=f"pr85_qrgb_f1_region_pair_{pair:04d}",
        design_kind="qrgb_frame0_halfband_to_pr85_region",
        atom=atom,
        source_archive=source_archive,
        pr90_meta=pr90_meta,
        qrgb_pair=_qrgb_pair_summary(planes, pair),
        source_sha_for_pair_action=source_sha_for_pair_action,
        charged_bytes=charged_bytes,
    )
    candidate.update(
        {
            "basis_action_schema": {
                "source_basis": selected["source_basis"],
                "source_semantic_plane": selected["semantic_plane"],
                "target_stream": "region",
                "target_runtime_semantics": "PR85 frame0 half/middle-region RGB or all-channel bias",
                "target_region_mask_index": selected["mask_index"],
                "target_channel_index": selected["channel_index"],
                "source_qrgb_value": selected["value"],
                "selected_value": value,
            },
            "actions": [
                {
                    "pair_index": pair,
                    "stream": "region",
                    "op": "set",
                    "source_value": source_value,
                    "candidate_value": candidate_value,
                    "source_artifact_sha256": source_sha_for_pair_action,
                    "source_atom_id": atom.get("atom_id"),
                    "rationale": "PR90 QRGB frame0 half-band residual transferred to PR85 frame0 region bias choice",
                }
            ],
        }
    )
    return candidate


def _design_f2_tile(
    *,
    atom: Mapping[str, Any],
    planes: np.ndarray,
    randtile_group: Mapping[str, Any],
    source_archive: Mapping[str, Any],
    pr90_meta: Mapping[str, Any],
    source_sha_for_pair_action: str,
    charged_bytes: int,
) -> dict[str, Any] | None:
    pair = int(atom["pair_index"])
    quad = [int(v) for v in planes[33:36, pair]]
    if not any(quad):
        return None
    channel = max(range(3), key=lambda idx: (abs(quad[idx]), -idx))
    value = _clamped_signed(quad[channel], VALS_AMP4)
    if value == 0:
        return None
    tile_index = 0 if value > 0 else 1
    candidate_value = _encode_randtile_choice(tile_index=tile_index, channel_type=channel, value=value)
    source_value = int(randtile_group["rows"][0, pair])
    if candidate_value == source_value:
        return None
    candidate = _candidate_base(
        candidate_id=f"pr85_qrgb_f2_tile_pair_{pair:04d}",
        design_kind="qrgb_frame1_quadrant_to_pr85_randmulti_tile_f2",
        atom=atom,
        source_archive=source_archive,
        pr90_meta=pr90_meta,
        qrgb_pair=_qrgb_pair_summary(planes, pair),
        source_sha_for_pair_action=source_sha_for_pair_action,
        charged_bytes=charged_bytes,
    )
    candidate.update(
        {
            "basis_action_schema": {
                "source_basis": "PR90 semantic planes 33..35: frame1 checker/quadrant RGB residual",
                "target_stream": "randmulti",
                "target_randmulti_group": {
                    "group_index": randtile_group["group_index"],
                    "lh": randtile_group["lh"],
                    "lw": randtile_group["lw"],
                    "amp": randtile_group["amp"],
                    "selection_row": 0,
                },
                "target_runtime_semantics": "PR85 frame1/f2 2x2 tile RGB bias, radius 4",
                "source_qrgb_quad_values": quad,
                "selected_channel_index": channel,
                "selected_tile_index": tile_index,
                "selected_value": value,
                "transfer_confidence": "low_one_tile_projection_of_checker_basis",
            },
            "actions": [
                {
                    "pair_index": pair,
                    "stream": "randmulti",
                    "op": "set",
                    "source_value": source_value,
                    "candidate_value": candidate_value,
                    "source_artifact_sha256": source_sha_for_pair_action,
                    "source_atom_id": atom.get("atom_id"),
                    "rationale": "PR90 QRGB frame1 quadrant residual projected to one PR85 f2 tile-bias choice",
                }
            ],
        }
    )
    return candidate


def _rank_candidates(candidates: Sequence[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    def key(row: Mapping[str, Any]) -> tuple[float, int, str]:
        expected = row.get("expected_break_even") if isinstance(row.get("expected_break_even"), Mapping) else {}
        margin = expected.get("break_even_margin_bytes")
        qrgb = row.get("qrgb_pair_summary") if isinstance(row.get("qrgb_pair_summary"), Mapping) else {}
        max_abs = qrgb.get("max_abs_qrgb_value")
        return (
            -(float(margin) if isinstance(margin, (int, float)) and math.isfinite(float(margin)) else -1.0),
            -int(max_abs) if isinstance(max_abs, int) else 0,
            str(row.get("candidate_id")),
        )

    ranked = [dict(row) for row in sorted(candidates, key=key)[:limit]]
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def _pair_action_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate_id"],
        "targeted_component": row["targeted_component"],
        "header_mode": "explicit_30",
        "charged_bytes_proxy": dict(row["charged_byte_proxy"]),
        "archive_changing_path": None,
        "actions": [
            {
                "pair_index": action["pair_index"],
                "stream": action["stream"],
                "op": action.get("op", "set"),
                "source_value": action["source_value"],
                "candidate_value": action["candidate_value"],
                "source_artifact_sha256": action["source_artifact_sha256"],
                "source_atom_id": action.get("source_atom_id"),
                "rationale": action.get("rationale"),
            }
            for action in row.get("actions", [])
        ],
    }


def _build_pair_action_evidence(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = []
    seen_pairs: set[int] = set()
    for candidate in candidates:
        actions = candidate.get("actions")
        if not isinstance(actions, list) or not actions:
            continue
        pair = actions[0].get("pair_index")
        if not isinstance(pair, int) or pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        selected.append(_pair_action_candidate(candidate))
    return {
        "schema": PAIR_ACTION_EVIDENCE_SCHEMA,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "thresholds": [],
        "candidates": selected,
    }


def _blocked_designs() -> list[dict[str, Any]]:
    return [
        {
            "design_kind": "direct_pr90_qrgb_stream_transplant",
            "dispatch_unlocked": False,
            "can_feed_pair_action_evidence": False,
            "blocker_class": "incompatible_runtime_consumer",
            "reason": "PR90 QRGB is consumed as 48 renderer-bias planes by PR90 inflate; PR85 runtime has no matching QRGB consumer.",
            "exact_next_implementation": "Implement a PR85 local archive builder that lowers selected QRGB-like controls into existing PR85 streams, not a raw PR90 payload transplant.",
        },
        {
            "design_kind": "pr85_post_stage_rgb_bias",
            "dispatch_unlocked": False,
            "can_feed_pair_action_evidence": False,
            "blocker_class": "pair_action_schema_lacks_post_stage_selector",
            "reason": "Current pair-action evidence schema has stream/value but no post-stage field; post choices are stage-indexed in PR85 runtime.",
            "exact_next_implementation": "Extend the local pair-atom archive builder contract for post stage ids, or keep QRGB transfer on bias/region/randmulti streams.",
        },
        {
            "design_kind": "pr85_random_lowfreq_randmulti_projection",
            "dispatch_unlocked": False,
            "can_feed_pair_action_evidence": False,
            "blocker_class": "ungrounded_basis_direction",
            "reason": "Generic PR85 randmulti low-frequency rows use seeded random patterns; PR90 QRGB signs do not identify which random pattern choice improves PR85 components.",
            "exact_next_implementation": "Run a local non-scorer output-delta builder first, then require CUDA component-response evidence before promotion.",
        },
    ]


def build_plan(
    *,
    pr85_archive: Path = DEFAULT_PR85_ARCHIVE,
    pr90_archive: Path = DEFAULT_PR90_ARCHIVE,
    pr90_probe_json: Path = DEFAULT_PR90_PROBE,
    pair_planning_json: Path = DEFAULT_PAIR_PLANNING,
    top_n: int = 8,
    max_candidates: int = 8,
) -> dict[str, Any]:
    pr90_meta, qrgb_planes = _load_pr90_qrgb(pr90_archive, pr90_probe_json)
    pr85_meta, pr85_values, pr85_segment_stats, randmulti_groups = _load_pr85_streams(pr85_archive)
    pair_meta, top_atoms, source_sha_for_pair_action = _load_pair_planning(pair_planning_json, top_n=top_n)
    charged_bytes = int(math.ceil(float(pr90_meta["qrgb_residual"]["compressed_bytes_per_pair"])))

    randglobal = _find_randmulti_group(randmulti_groups, (222, 222, 4, 1))
    randtile = _find_randmulti_group(randmulti_groups, (224, 222, 4, 1))
    if randglobal is None or randtile is None:
        raise QrgbTransferPlanError("PR85 randmulti source lacks required global/tile groups")

    proposals: list[dict[str, Any]] = []
    for atom in top_atoms:
        if not isinstance(atom.get("pair_index"), int):
            continue
        for builder in (
            lambda row: _design_f2_randglobal(
                atom=row,
                planes=qrgb_planes,
                randglobal_group=randglobal,
                source_archive=pr85_meta,
                pr90_meta=pr90_meta,
                source_sha_for_pair_action=source_sha_for_pair_action,
                charged_bytes=charged_bytes,
            ),
            lambda row: _design_f1_bias(
                atom=row,
                planes=qrgb_planes,
                pr85_values=pr85_values,
                source_archive=pr85_meta,
                pr90_meta=pr90_meta,
                source_sha_for_pair_action=source_sha_for_pair_action,
                charged_bytes=charged_bytes,
            ),
            lambda row: _design_f1_region(
                atom=row,
                planes=qrgb_planes,
                pr85_values=pr85_values,
                source_archive=pr85_meta,
                pr90_meta=pr90_meta,
                source_sha_for_pair_action=source_sha_for_pair_action,
                charged_bytes=charged_bytes,
            ),
            lambda row: _design_f2_tile(
                atom=row,
                planes=qrgb_planes,
                randtile_group=randtile,
                source_archive=pr85_meta,
                pr90_meta=pr90_meta,
                source_sha_for_pair_action=source_sha_for_pair_action,
                charged_bytes=charged_bytes,
            ),
        ):
            candidate = builder(atom)
            if candidate is not None:
                proposals.append(candidate)

    ranked = _rank_candidates(proposals, limit=max_candidates)
    pair_action_evidence = _build_pair_action_evidence(ranked)
    plan = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "dispatch_performed": False,
        "remote_jobs_dispatched": False,
        "gpu_required": False,
        "dispatch_unlocked": False,
        "source_archive": pr85_meta,
        "pr90_source": pr90_meta,
        "pair_planning_source": pair_meta,
        "basis_action_schema": {
            "pr90_qrgb_basis": [
                "planes 0..5: frame0/frame1 global RGB offsets",
                "planes 6..17: frame0/frame1 vertical half-band RGB offsets",
                "planes 18..29: frame0/frame1 horizontal half-band RGB offsets",
                "planes 30..35: frame0/frame1 checker/quadrant RGB offsets",
                "planes 36..47: frame0/frame1 vertical detail alternating-band RGB offsets",
            ],
            "pr85_target_streams": [
                "bias: frame0 global RGB choice",
                "region: frame0 half/middle-region RGB choice",
                "randmulti group (222,222,4): frame1/f2 global RGB-bias oracle",
                "randmulti group (224,222,4): frame1/f2 2x2 tile RGB-bias oracle",
            ],
            "pair_action_evidence_schema": PAIR_ACTION_EVIDENCE_SCHEMA,
        },
        "pr85_segment_evidence": pr85_segment_stats,
        "ranked_candidates": ranked,
        "candidate_count": len(ranked),
        "can_feed_pair_action_evidence_count": len(pair_action_evidence["candidates"]),
        "ready_for_exact_eval_after_lane_claim_count": 0,
        "pair_action_evidence": pair_action_evidence,
        "blocked_designs": _blocked_designs(),
        "blocker_class": "no_archive_changing_path" if ranked else "no_feasible_pr85_qrgb_like_actions",
        "blockers": [
            {
                "blocker_class": "no_archive_changing_path",
                "reason": "No byte-closed PR85 archive candidate was built or non-noop-preflighted in this planner.",
            },
            {
                "blocker_class": "no_gpu_dispatch_by_contract",
                "reason": "Worker D task explicitly forbids remote GPU dispatch.",
            },
        ],
        "exact_next_implementation": (
            "Feed pair_action_evidence.json into build_pr85_pair_action_candidates.py, "
            "then build a local PR85 candidate archive with build_pr85_pair_atom_candidates.py "
            "and require fixed-runtime non-noop custody before any lane claim or exact CUDA eval."
        ),
    }
    plan["stable_plan_digest_sha256"] = _stable_digest(plan)
    return plan


def write_ledger(path: Path, plan: Mapping[str, Any], *, out_json: Path, pair_action_json: Path) -> None:
    ranked = plan.get("ranked_candidates", [])
    lines = [
        "# PR85 QRGB Transfer Worker",
        "",
        "## Contract",
        "",
        f"- tool: `{TOOL}`",
        "- score_claim: false",
        "- dispatch_performed: false",
        "- remote_jobs_dispatched: false",
        "- gpu_required: false",
        "- dispatch_unlocked: false",
        "- no exact eval dispatch was attempted",
        "",
        "## Sources",
        "",
        f"- PR85 archive: {plan['source_archive']['archive_bytes']} bytes, sha256 `{plan['source_archive']['archive_sha256']}`",
        f"- PR85 exact T4 score context: {PR85_SCORE_CONTEXT}",
        f"- PR90 qrepro archive: {plan['pr90_source']['archive_bytes']} bytes, sha256 `{plan['pr90_source']['archive_sha256']}`",
        f"- PR90 QRGB residual: {plan['pr90_source']['qrgb_residual']['compressed_bytes']} compressed bytes, {plan['pr90_source']['qrgb_residual']['nonzero_values']} nonzero int8 edits",
        f"- pair planning source: `{plan['pair_planning_source']['path']}`",
        "",
        "## Design Choice",
        "",
        "PR90's raw QRGB stream is not transplanted into PR85. The planner lowers the idea into PR85-native stream/value actions on `bias`, `region`, and selected `randmulti` groups because those are the existing archive-consuming PR85 runtime controls. The best current candidate is still action evidence only, not an archive.",
        "",
        "## Ranked Candidates",
        "",
    ]
    if isinstance(ranked, list) and ranked:
        for row in ranked:
            action = row.get("actions", [{}])[0]
            margin = ((row.get("expected_break_even") or {}).get("break_even_margin_bytes"))
            lines.append(
                f"- rank {row.get('rank')}: `{row.get('candidate_id')}` pair={row.get('pair_index')} "
                f"stream={action.get('stream')} {action.get('source_value')}->{action.get('candidate_value')} "
                f"proxy_bytes={(row.get('charged_byte_proxy') or {}).get('candidate_action_bytes')} "
                f"break_even_margin={margin} dispatch_unlocked=false"
            )
    else:
        lines.append("- no feasible PR85-native QRGB-like actions were found in the selected top pairs")
    lines.extend(
        [
            "",
            "## Blockers",
            "",
            "- no_archive_changing_path: no byte-closed PR85 archive candidate or non-noop custody exists yet",
            "- no_pr85_component_response_for_direction: PR90 QRGB signs are transfer priors, not measured PR85 scorer response",
            "- direct_pr90_qrgb_stream_transplant is blocked because PR85 has no matching QRGB runtime consumer",
            "",
            "## Artifacts",
            "",
            f"- ranking JSON: `{_rel(out_json)}`",
            f"- pair-action evidence JSON: `{_rel(pair_action_json)}`",
            f"- stable plan digest: `{plan.get('stable_plan_digest_sha256')}`",
            "",
            "## Exact Next Implementation",
            "",
            str(plan.get("exact_next_implementation")),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr85-archive", type=Path, default=DEFAULT_PR85_ARCHIVE)
    parser.add_argument("--pr90-archive", type=Path, default=DEFAULT_PR90_ARCHIVE)
    parser.add_argument("--pr90-probe-json", type=Path, default=DEFAULT_PR90_PROBE)
    parser.add_argument("--pair-planning-json", type=Path, default=DEFAULT_PAIR_PLANNING)
    parser.add_argument("--top-n", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--pair-action-evidence-json", type=Path, default=DEFAULT_PAIR_ACTION_EVIDENCE_JSON)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    args = parser.parse_args(argv)

    if args.top_n <= 0 or args.max_candidates <= 0:
        raise QrgbTransferPlanError("--top-n and --max-candidates must be positive")

    plan = build_plan(
        pr85_archive=args.pr85_archive,
        pr90_archive=args.pr90_archive,
        pr90_probe_json=args.pr90_probe_json,
        pair_planning_json=args.pair_planning_json,
        top_n=args.top_n,
        max_candidates=args.max_candidates,
    )
    _write_json(args.out_json, plan)
    _write_json(args.pair_action_evidence_json, plan["pair_action_evidence"])
    write_ledger(args.ledger, plan, out_json=args.out_json, pair_action_json=args.pair_action_evidence_json)
    print(
        _json_text(
            {
                "out_json": _rel(args.out_json),
                "pair_action_evidence_json": _rel(args.pair_action_evidence_json),
                "ledger": _rel(args.ledger),
                "candidate_count": plan["candidate_count"],
                "can_feed_pair_action_evidence_count": plan["can_feed_pair_action_evidence_count"],
                "dispatch_unlocked": plan["dispatch_unlocked"],
                "stable_plan_digest_sha256": plan["stable_plan_digest_sha256"],
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
