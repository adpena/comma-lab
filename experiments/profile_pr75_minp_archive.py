#!/usr/bin/env python3
"""Forensic byte/profile support for the current PR75/minp public archive.

This is a local reverse-engineering tool only. It does not load the scorer,
does not require CUDA, does not dispatch jobs, and never claims score. Its
purpose is to make the current public PR75/minp single-blob grammar explicit so
runtime parity work can be implemented against measured bytes instead of chat
notes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/archive.zip"
)
DEFAULT_COMPARE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip"
)
DEFAULT_OUTPUT_JSON = (
    REPO_ROOT
    / "experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/"
    "pr75_minp_grammar_profile.json"
)
DEFAULT_PR75_SOURCE_ROOT = Path("/tmp/pr75-minp")
DEFAULT_UNPACKER = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"

SCHEMA = "pr75_minp_archive_grammar_profile_v1"
TOOL = "experiments/profile_pr75_minp_archive.py"
EVIDENCE_GRADE = "empirical_reverse_engineering_profile"
MEMBER_NAME = "p"
MASK_BR_LEN = 219_472
SEG_H = 384
SEG_W = 512
SEG_TILE_SIZE = 32
MAX_TILE_ID = (SEG_H // SEG_TILE_SIZE) * (SEG_W // SEG_TILE_SIZE)
PUBLIC_ACTION_COUNT = 9 * 6 * 2


@dataclass(frozen=True)
class FixedSlicePlan:
    """Fixed public single-blob slice contract recovered from PR75 inflate.py."""

    payload_bytes: int
    label: str
    mask_br_bytes: int
    renderer_br_bytes: int
    actions_br_bytes: int

    @property
    def pose_br_bytes(self) -> int:
        return (
            self.payload_bytes
            - self.mask_br_bytes
            - self.renderer_br_bytes
            - self.actions_br_bytes
        )


FIXED_SLICE_PLANS: dict[int, FixedSlicePlan] = {
    276_641: FixedSlicePlan(276_641, "pr75_fixed_qpose14_r55_actions236_model56034", MASK_BR_LEN, 56_034, 236),
    276_520: FixedSlicePlan(276_520, "pr75_fixed_qpose14_r55_actions236_model55914", MASK_BR_LEN, 55_914, 236),
    276_381: FixedSlicePlan(276_381, "pr75_minp_fixed_actions255_model55756", MASK_BR_LEN, 55_756, 255),
    276_379: FixedSlicePlan(276_379, "pr75_minp_fixed_actions253_model55756", MASK_BR_LEN, 55_756, 253),
    276_362: FixedSlicePlan(276_362, "pr75_minp_fixed_actions236_model55756", MASK_BR_LEN, 55_756, 236),
    277_247: FixedSlicePlan(277_247, "pr79_minp_s1_split_actions1121_model55756", MASK_BR_LEN, 55_756, 1_121),
    277_288: FixedSlicePlan(277_288, "pr79_minp_v2_fixed_actions1162_model55756", MASK_BR_LEN, 55_756, 1_162),
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _read_single_payload_zip(path: Path) -> tuple[bytes, dict[str, Any]]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != [MEMBER_NAME]:
            raise ValueError(f"{path} must contain exactly one member {MEMBER_NAME!r}; got {names!r}")
        info = infos[0]
        payload = zf.read(info)
        return payload, {
            "member_name": info.filename,
            "member_compress_type": int(info.compress_type),
            "member_file_size": int(info.file_size),
            "member_compress_size": int(info.compress_size),
            "member_crc": int(info.CRC),
        }


def _safe_prefix(data: bytes, n: int = 8) -> str:
    return data[:n].hex()


def _stream_summary(name: str, charged: bytes, decoded: bytes, codec: str) -> dict[str, Any]:
    return {
        "name": name,
        "charged_bytes": int(len(charged)),
        "charged_sha256": _sha256_bytes(charged),
        "codec": codec,
        "decoded_bytes": int(len(decoded)),
        "decoded_sha256": _sha256_bytes(decoded),
        "decoded_prefix_hex": _safe_prefix(decoded),
    }


def fixed_slice_plan_for_payload(payload: bytes) -> FixedSlicePlan:
    plan = FIXED_SLICE_PLANS.get(len(payload))
    if plan is None:
        known = ", ".join(str(k) for k in sorted(FIXED_SLICE_PLANS))
        raise ValueError(
            f"unsupported PR75/minp fixed payload length {len(payload)}; "
            f"known fixed lengths: {known}"
        )
    if plan.pose_br_bytes <= 0:
        raise ValueError(f"invalid fixed slice plan {plan}")
    return plan


def split_fixed_public_payload(payload: bytes) -> tuple[FixedSlicePlan, dict[str, bytes]]:
    """Split the public current-workflow PR75/minp stored-payload wire bytes."""
    plan = fixed_slice_plan_for_payload(payload)
    cursor = 0
    mask = payload[cursor : cursor + plan.mask_br_bytes]
    cursor += plan.mask_br_bytes
    renderer = payload[cursor : cursor + plan.renderer_br_bytes]
    cursor += plan.renderer_br_bytes
    actions = payload[cursor : cursor + plan.actions_br_bytes]
    cursor += plan.actions_br_bytes
    pose = payload[cursor:]
    if len(pose) != plan.pose_br_bytes:
        raise ValueError(f"pose slice length mismatch: {len(pose)} != {plan.pose_br_bytes}")
    return plan, {
        "masks.mkv.br": mask,
        "renderer.bin.br": renderer,
        "seg_tile_actions.br": actions,
        "optimized_poses.qp1.br": pose,
    }


def _read_uvarint(raw: bytes, cursor: int) -> tuple[int, int]:
    value = 0
    shift = 0
    start = cursor
    while cursor < len(raw):
        byte = raw[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, cursor
        shift += 7
        if shift > 63:
            break
    raise ValueError(f"truncated or overlong uvarint at byte {start}")


def decode_seg_tile_actions_raw(raw: bytes) -> tuple[str, list[tuple[int, int, int]]]:
    """Decode public PR75 action wire forms to runtime (pair, tile, action)."""
    records: list[tuple[int, int, int]] = []
    if raw.startswith(b"TA4"):
        body = raw[3:]
        if len(body) % 4:
            raise ValueError("TA4 action body length is not divisible by 4")
        for offset in range(0, len(body), 4):
            records.append((int.from_bytes(body[offset:offset + 2], "little"), body[offset + 2], body[offset + 3]))
        wire_kind = "TA4_raw_u16pair_u8tile_u8action"
    elif raw.startswith(b"TA5"):
        body = raw[3:]
        if len(body) % 5:
            raise ValueError("TA5 action body length is not divisible by 5")
        for offset in range(0, len(body), 5):
            records.append((int.from_bytes(body[offset:offset + 2], "little"), int.from_bytes(body[offset + 2:offset + 4], "little"), body[offset + 4]))
        wire_kind = "TA5_raw_u16pair_u16tile_u8action"
    elif raw.startswith(b"S1"):
        cursor = 2
        group_count, cursor = _read_uvarint(raw, cursor)
        groups: list[tuple[int, int]] = []
        tile = 0
        for group_index in range(group_count):
            tile_delta, cursor = _read_uvarint(raw, cursor)
            tile = tile_delta if group_index == 0 else tile + tile_delta
            count, cursor = _read_uvarint(raw, cursor)
            groups.append((tile, count))
        pairs: list[tuple[int, int]] = []
        for tile, count in groups:
            frame = 0
            for idx in range(count):
                delta, cursor = _read_uvarint(raw, cursor)
                frame = delta if idx == 0 else frame + delta
                pairs.append((frame, tile))
        if cursor + len(pairs) != len(raw):
            raise ValueError("S1 split action stream length mismatch")
        for frame, tile in pairs:
            action = raw[cursor]
            cursor += 1
            records.append((frame, tile, action))
        wire_kind = "S1_split_tile_delta_count_pair_delta_actions"
    elif raw.startswith(b"SG2") or (len(raw) % 4 != 0 and len(raw) % 5 != 0):
        cursor = 3 if raw.startswith(b"SG2") else 0
        while cursor < len(raw):
            tile, cursor = _read_uvarint(raw, cursor)
            count, cursor = _read_uvarint(raw, cursor)
            frame = 0
            for idx in range(count):
                delta, cursor = _read_uvarint(raw, cursor)
                frame = delta if idx == 0 else frame + delta
                if cursor >= len(raw):
                    raise ValueError("SG2 action stream ended before action byte")
                action = raw[cursor]
                cursor += 1
                records.append((frame, tile, action))
        wire_kind = "SG2_grouped_tile_frame_delta_varint"
    elif len(raw) % 4 == 0 and len(raw) % 5 != 0:
        for offset in range(0, len(raw), 4):
            records.append((int.from_bytes(raw[offset:offset + 2], "little"), raw[offset + 2], raw[offset + 3]))
        wire_kind = "raw4_u16pair_u8tile_u8action"
    elif len(raw) % 5 == 0 and len(raw) % 4 != 0:
        for offset in range(0, len(raw), 5):
            records.append((int.from_bytes(raw[offset:offset + 2], "little"), int.from_bytes(raw[offset + 2:offset + 4], "little"), raw[offset + 4]))
        wire_kind = "raw5_u16pair_u16tile_u8action"
    elif not raw:
        wire_kind = "empty_raw4"
    else:
        raise ValueError(f"ambiguous public action body length without TA4/TA5 header: {len(raw)}")

    for frame, tile, action in records:
        if not 0 <= frame < 600:
            raise ValueError(f"public seg action frame out of range: {frame}")
        if not 0 <= tile < MAX_TILE_ID:
            raise ValueError(f"public seg action tile out of range: {tile}")
        if not 0 <= action < PUBLIC_ACTION_COUNT:
            raise ValueError(f"public seg action id out of range: {action}")
    return wire_kind, records


def encode_runtime_action_records(records: list[tuple[int, int, int]]) -> bytes:
    out = bytearray()
    for frame, tile, action in records:
        out.extend(int(frame).to_bytes(2, "little"))
        out.append(int(tile))
        out.append(int(action))
    return bytes(out)


def _top_counts(counter: Counter[Any], n: int = 12) -> list[dict[str, Any]]:
    return [{"value": key, "count": int(count)} for key, count in counter.most_common(n)]


def summarize_action_records(
    *,
    raw_wire: bytes,
    charged: bytes,
    records: list[tuple[int, int, int]],
    wire_kind: str,
) -> dict[str, Any]:
    runtime_bytes = encode_runtime_action_records(records)
    pair_counts = Counter(frame for frame, _tile, _action in records)
    tile_counts = Counter(tile for _frame, tile, _action in records)
    action_counts = Counter(action for _frame, _tile, action in records)
    pairs = sorted(pair_counts)
    return {
        "wire_kind": wire_kind,
        "charged_brotli_bytes": int(len(charged)),
        "charged_brotli_sha256": _sha256_bytes(charged),
        "wire_raw_bytes": int(len(raw_wire)),
        "wire_raw_sha256": _sha256_bytes(raw_wire),
        "runtime_record_bytes": int(len(runtime_bytes)),
        "runtime_record_sha256": _sha256_bytes(runtime_bytes),
        "record_count": int(len(records)),
        "unique_pair_count": int(len(pair_counts)),
        "pair_min": int(min(pairs)) if pairs else None,
        "pair_max": int(max(pairs)) if pairs else None,
        "unique_tile_count": int(len(tile_counts)),
        "unique_action_count": int(len(action_counts)),
        "top_pairs": _top_counts(pair_counts),
        "top_tiles": _top_counts(tile_counts),
        "top_actions": _top_counts(action_counts),
        "first_records": [
            {"pair": int(frame), "tile": int(tile), "action": int(action)}
            for frame, tile, action in records[:12]
        ],
        "last_records": [
            {"pair": int(frame), "tile": int(tile), "action": int(action)}
            for frame, tile, action in records[-12:]
        ],
    }


def summarize_qp1_pose(raw: bytes) -> dict[str, Any]:
    if not raw.startswith(b"QP1"):
        return {
            "codec": "unknown",
            "raw_bytes": int(len(raw)),
            "raw_sha256": _sha256_bytes(raw),
            "raw_prefix_hex": _safe_prefix(raw),
        }
    if len(raw) < 5:
        raise ValueError("QP1 stream is too short")
    values = [int.from_bytes(raw[3:5], "little")]
    cursor = 5
    while cursor < len(raw):
        acc, cursor = _read_uvarint(raw, cursor)
        delta = (acc >> 1) ^ -(acc & 1)
        values.append(values[-1] + delta)
    return {
        "codec": "QP1_col0_delta_varint",
        "raw_bytes": int(len(raw)),
        "raw_sha256": _sha256_bytes(raw),
        "row_count": int(len(values)),
        "q0_min": int(min(values)),
        "q0_max": int(max(values)),
        "q0_first": int(values[0]),
        "q0_last": int(values[-1]),
        "raw_prefix_hex": _safe_prefix(raw),
    }


def _load_unpacker(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("profile_pr75_minp_unpacker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load unpacker from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def robust_parse_payload(payload: bytes, unpacker_path: Path) -> dict[str, Any]:
    try:
        unpacker = _load_unpacker(unpacker_path)
        header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    except Exception as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "ok": True,
        "payload_format": header.get("payload_format"),
        "members": header.get("members", []),
        "decoded_members": {
            name: {
                "decoded_bytes": int(len(blob)),
                "decoded_sha256": _sha256_bytes(blob),
                "decoded_prefix_hex": _safe_prefix(blob),
            }
            for name, blob in decoded.items()
        },
    }


def profile_public_minp_archive(path: Path, *, unpacker_path: Path) -> dict[str, Any]:
    payload, zip_info = _read_single_payload_zip(path)
    plan, slices = split_fixed_public_payload(payload)
    mask_raw = brotli.decompress(slices["masks.mkv.br"])
    renderer_raw = brotli.decompress(slices["renderer.bin.br"])
    actions_raw = brotli.decompress(slices["seg_tile_actions.br"])
    pose_raw = brotli.decompress(slices["optimized_poses.qp1.br"])
    if not mask_raw.startswith(b"\x12\x00"):
        raise ValueError("decoded mask stream does not look like AV1 OBU")
    if not renderer_raw.startswith(b"QZS3"):
        raise ValueError(f"decoded renderer does not start with QZS3: {renderer_raw[:4]!r}")
    if not pose_raw.startswith(b"QP1"):
        raise ValueError(f"decoded pose stream does not start with QP1: {pose_raw[:3]!r}")
    action_wire_kind, action_records = decode_seg_tile_actions_raw(actions_raw)
    return {
        "archive": {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
            "zip": zip_info,
        },
        "payload": {
            "member": MEMBER_NAME,
            "bytes": int(len(payload)),
            "sha256": _sha256_bytes(payload),
            "prefix_hex": _safe_prefix(payload),
            "fixed_slice_plan": {
                "label": plan.label,
                "payload_bytes": plan.payload_bytes,
                "mask_br_bytes": plan.mask_br_bytes,
                "renderer_br_bytes": plan.renderer_br_bytes,
                "actions_br_bytes": plan.actions_br_bytes,
                "pose_br_bytes": plan.pose_br_bytes,
            },
        },
        "decoded_streams": {
            "masks.mkv": _stream_summary("masks.mkv", slices["masks.mkv.br"], mask_raw, "brotli_av1_obu"),
            "renderer.bin": _stream_summary("renderer.bin", slices["renderer.bin.br"], renderer_raw, "brotli_qzs3"),
            "seg_tile_actions.bin": _stream_summary("seg_tile_actions.bin", slices["seg_tile_actions.br"], actions_raw, "brotli_public_seg_tile_actions"),
            "optimized_poses.qp1": _stream_summary("optimized_poses.qp1", slices["optimized_poses.qp1.br"], pose_raw, "brotli_qp1"),
        },
        "renderer": {
            "magic": renderer_raw[:4].decode("ascii", errors="replace"),
            "qzs3_block_size": int.from_bytes(renderer_raw[4:6], "little") if len(renderer_raw) >= 6 else None,
        },
        "actions": summarize_action_records(
            raw_wire=actions_raw,
            charged=slices["seg_tile_actions.br"],
            records=action_records,
            wire_kind=action_wire_kind,
        ),
        "pose": summarize_qp1_pose(pose_raw),
        "robust_current_parse": robust_parse_payload(payload, unpacker_path),
    }


def profile_compare_archive(path: Path, *, unpacker_path: Path) -> dict[str, Any]:
    payload, zip_info = _read_single_payload_zip(path)
    parsed = robust_parse_payload(payload, unpacker_path)
    out: dict[str, Any] = {
        "archive": {
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": _sha256_file(path),
            "zip": zip_info,
        },
        "payload": {
            "member": MEMBER_NAME,
            "bytes": int(len(payload)),
            "sha256": _sha256_bytes(payload),
            "prefix_hex": _safe_prefix(payload),
        },
        "robust_current_parse": parsed,
    }
    decoded = parsed.get("decoded_members") if parsed.get("ok") else None
    if isinstance(decoded, dict) and "seg_tile_actions.bin" in decoded:
        unpacker = _load_unpacker(unpacker_path)
        _header, decoded_bytes = unpacker._parse_payload(payload)  # noqa: SLF001
        raw_actions = decoded_bytes["seg_tile_actions.bin"]
        records = []
        if len(raw_actions) % 4 == 0:
            for offset in range(0, len(raw_actions), 4):
                records.append((
                    int.from_bytes(raw_actions[offset:offset + 2], "little"),
                    raw_actions[offset + 2],
                    raw_actions[offset + 3],
                ))
            out["actions"] = summarize_action_records(
                raw_wire=raw_actions,
                charged=raw_actions,
                records=records,
                wire_kind="robust_runtime_raw4_after_unpack",
            )
    return out


def compare_public_to_archive(public: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    other_decoded = other.get("robust_current_parse", {}).get("decoded_members", {})
    stream_comparison: dict[str, Any] = {}
    for public_name, other_name in (
        ("masks.mkv", "masks.mkv"),
        ("renderer.bin", "renderer.bin"),
        ("optimized_poses.qp1", "optimized_poses.qp1"),
        ("seg_tile_actions.bin", "seg_tile_actions.bin"),
    ):
        p_stream = public.get("decoded_streams", {}).get(public_name)
        o_stream = other_decoded.get(other_name)
        if not p_stream or not o_stream:
            stream_comparison[public_name] = {
                "available": False,
                "public_available": bool(p_stream),
                "compare_available": bool(o_stream),
            }
            continue
        stream_comparison[public_name] = {
            "available": True,
            "decoded_bytes_delta_public_minus_compare": int(p_stream["decoded_bytes"]) - int(o_stream["decoded_bytes"]),
            "decoded_sha256_equal": p_stream["decoded_sha256"] == o_stream["decoded_sha256"],
            "public_decoded_sha256": p_stream["decoded_sha256"],
            "compare_decoded_sha256": o_stream["decoded_sha256"],
        }
    action_overlap: dict[str, Any] | None = None
    public_actions = public.get("actions")
    other_actions = other.get("actions")
    if public_actions and other_actions:
        # Only use record hashes and counts here; the full tuple overlap is
        # intentionally left to the JSON's first/last/top summaries to avoid
        # bloating the forensic profile.
        action_overlap = {
            "public_record_count": public_actions["record_count"],
            "compare_record_count": other_actions["record_count"],
            "runtime_record_sha256_equal": (
                public_actions["runtime_record_sha256"]
                == other_actions["runtime_record_sha256"]
            ),
            "public_runtime_record_sha256": public_actions["runtime_record_sha256"],
            "compare_runtime_record_sha256": other_actions["runtime_record_sha256"],
        }
    return {
        "public_minus_compare_archive_bytes": int(public["archive"]["bytes"]) - int(other["archive"]["bytes"]),
        "public_minus_compare_payload_bytes": int(public["payload"]["bytes"]) - int(other["payload"]["bytes"]),
        "stream_comparison": stream_comparison,
        "action_record_comparison": action_overlap,
    }


def _git_source_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    info: dict[str, Any] = {"path": str(path), "exists": True}
    if (path / ".git").exists():
        for key, args in (
            ("commit", ["rev-parse", "HEAD"]),
            ("branch", ["rev-parse", "--abbrev-ref", "HEAD"]),
        ):
            try:
                completed = subprocess.run(
                    ["git", "-C", str(path), *args],
                    check=True,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                info[key] = completed.stdout.strip()
            except (OSError, subprocess.CalledProcessError) as exc:
                info[f"{key}_error"] = str(exc)
    inflate = path / "submissions/qpose14_r55_segactions_minp/inflate.py"
    info["inflate_py"] = {
        "path": str(inflate),
        "exists": inflate.exists(),
        "sha256": _sha256_file(inflate) if inflate.exists() else None,
    }
    return info


def build_profile(
    *,
    archive: Path,
    compare_archive: Path | None,
    pr75_source_root: Path,
    unpacker_path: Path,
) -> dict[str, Any]:
    public = profile_public_minp_archive(archive, unpacker_path=unpacker_path)
    compare = None
    comparison = None
    if compare_archive is not None and compare_archive.exists():
        compare = profile_compare_archive(compare_archive, unpacker_path=unpacker_path)
        comparison = compare_public_to_archive(public, compare)
    if public.get("robust_current_parse", {}).get("ok"):
        implications = [
            "current robust_current parses the PR75/minp 276381 fixed-slice payload and converts SG2 actions to runtime raw4 records.",
            "the remaining parity risk is semantic/runtime parity, not byte-slice discovery: run raw-output parity against the public runtime before dispatch.",
            "isolate actions-only, renderer-only, pose-only, and full-stack candidates after local stream/runtime parity is exact.",
        ]
    else:
        implications = [
            "robust_current fixed-slice PR75 parser needs the 276381/276379/276362 table if this public minp grammar is to inflate locally.",
            "robust_current action decoder needs SG2 grouped-by-tile frame-delta varints or a pre-unpack conversion to runtime raw4 records.",
            "after local stream/runtime parity is exact, isolate actions-only, renderer-only, pose-only, and full-stack exact CUDA evals.",
        ]
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "evidence_grade": EVIDENCE_GRADE,
        "notes": [
            "Local byte/runtime grammar profile only; exact CUDA auth eval remains the score truth.",
            "Public current-workflow ZIP permissiveness is studied as forensics, not used as a score claim.",
        ],
        "public_pr75_source": _git_source_info(pr75_source_root),
        "public_archive": public,
        "compare_archive": compare,
        "comparison": comparison,
        "implementation_implications": implications,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--compare-archive", type=Path, default=DEFAULT_COMPARE_ARCHIVE)
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--pr75-source-root", type=Path, default=DEFAULT_PR75_SOURCE_ROOT)
    parser.add_argument("--robust-unpacker", type=Path, default=DEFAULT_UNPACKER)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    compare_archive = None if args.no_compare else args.compare_archive
    profile = build_profile(
        archive=args.archive,
        compare_archive=compare_archive,
        pr75_source_root=args.pr75_source_root,
        unpacker_path=args.robust_unpacker,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_bytes(_json_bytes(profile))
    print(json.dumps(profile, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
