#!/usr/bin/env python3
# pyc-recovery pass2: rehydrated from git blob 2c9bf175e7e5b83048defec249f6d414f6b2afb7 via `git fsck --lost-found`
# original path: experiments/profile_pr94_qpose_intake.py
# This is OUR source, dropped during commit 66c59aae filter-repo cleanup; the .pyc was the only
# orphan left behind. Original blob SHA verified intact.
# Recovered: 2026-05-05 by Sherlock pass2
"""Static PR94 qpose archive intake profiler.

This tool performs local byte and grammar accounting only. It does not run the
contest scorer, does not load scorer models, and does not dispatch GPU work.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = REPO_ROOT / "experiments/results/public_pr94_qpose_intake_20260504_codex/archive.zip"
DEFAULT_PR_JSON = REPO_ROOT / "experiments/results/public_pr94_hpac_contract_probe_20260504_codex/pr94_api.json"
DEFAULT_OUT_DIR = REPO_ROOT / "experiments/results/public_pr94_qpose_intake_20260504_codex"
DEFAULT_JSON_OUT = DEFAULT_OUT_DIR / "profile_pr94_qpose_intake.json"
DEFAULT_MARKDOWN_OUT = DEFAULT_OUT_DIR / "profile_pr94_qpose_intake.md"
CONTEST_ORIGINAL_BYTES = 37_545_489
TOOL = "experiments/profile_pr94_qpose_intake.py"
SCHEMA = "pr94_qpose_static_intake_profile_v1"


@dataclass(frozen=True)
class SegmentLayout:
    payload_format: str
    boundary_authority: str
    header_bytes: int
    mask_len: int
    model_len: int
    actions_len: int
    pose_len: int


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


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _rate_score_delta(byte_delta: int) -> float:
    return 25.0 * byte_delta / CONTEST_ORIGINAL_BYTES


def _read_single_member_zip(path: Path) -> tuple[dict[str, Any], bytes]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != ["p"]:
            raise ValueError(f"PR94 archive must contain exactly one member 'p'; got {names!r}")
        info = infos[0]
        if info.filename.startswith("/") or ".." in Path(info.filename).parts:
            raise ValueError(f"unsafe ZIP member path: {info.filename!r}")
        payload = zf.read(info)
        archive_bytes = path.stat().st_size
        return (
            {
                "path": _rel(path),
                "archive_bytes": archive_bytes,
                "archive_sha256": _sha256_file(path),
                "member_name": info.filename,
                "member_bytes": int(info.file_size),
                "member_compress_size": int(info.compress_size),
                "member_sha256": _sha256_bytes(payload),
                "zip_compress_type": int(info.compress_type),
                "zip_stored": info.compress_type == zipfile.ZIP_STORED,
                "zip_timestamp": list(info.date_time),
                "zip_overhead_bytes": archive_bytes - int(info.compress_size),
                "crc32_hex": f"{info.CRC:08x}",
            },
            payload,
        )


def infer_pr94_layout(payload: bytes) -> SegmentLayout:
    """Mirror the PR94 ``inflate.py`` packed-payload branch without decoding."""
    if payload.startswith(b"P3"):
        if len(payload) < 10:
            raise ValueError("truncated P3 qpose payload")
        mask_len, model_len, actions_len = struct.unpack_from("<IHH", payload, 2)
        header_bytes = 2 + struct.calcsize("<IHH")
        pose_len = len(payload) - header_bytes - mask_len - model_len - actions_len
        if min(mask_len, model_len, actions_len, pose_len) <= 0:
            raise ValueError("invalid P3 qpose segment length")
        return SegmentLayout(
            payload_format="p3_self_describing_qpose_tile_actions",
            boundary_authority="pr94_inflate_p3_header",
            header_bytes=header_bytes,
            mask_len=int(mask_len),
            model_len=int(model_len),
            actions_len=int(actions_len),
            pose_len=int(pose_len),
        )
    if payload.startswith(b"P2"):
        if len(payload) < 8:
            raise ValueError("truncated P2 qpose payload")
        mask_len, model_len = struct.unpack_from("<IH", payload, 2)
        header_bytes = 2 + struct.calcsize("<IH")
        pose_len = len(payload) - header_bytes - mask_len - model_len
        if min(mask_len, model_len, pose_len) <= 0:
            raise ValueError("invalid P2 qpose segment length")
        return SegmentLayout(
            payload_format="p2_self_describing_qpose_no_tile_actions",
            boundary_authority="pr94_inflate_p2_header",
            header_bytes=header_bytes,
            mask_len=int(mask_len),
            model_len=int(model_len),
            actions_len=0,
            pose_len=int(pose_len),
        )
    if len(payload) == 276_641:
        return SegmentLayout(
            payload_format="public_pr75_fixed_qpose_tile_actions",
            boundary_authority="pr94_inflate_exact_len_276641",
            header_bytes=0,
            mask_len=219_472,
            model_len=56_034,
            actions_len=236,
            pose_len=len(payload) - 219_472 - 56_034 - 236,
        )
    if len(payload) in (276_574, 276_749) or 276_900 <= len(payload) <= 278_000:
        mask_len = 219_472
        model_len = 55_756
        pose_len = 898
        actions_len = len(payload) - mask_len - model_len - pose_len
        if actions_len <= 0:
            raise ValueError("PR94 fixed-range payload has no action stream")
        return SegmentLayout(
            payload_format="pr94_fixed_range_qpose_tile_actions",
            boundary_authority="pr94_inflate_len_range_276900_278000",
            header_bytes=0,
            mask_len=mask_len,
            model_len=model_len,
            actions_len=actions_len,
            pose_len=pose_len,
        )
    raise ValueError(f"unsupported PR94 qpose payload length: {len(payload)}")


def _slice_payload(payload: bytes, layout: SegmentLayout) -> dict[str, bytes]:
    cursor = layout.header_bytes
    mask = payload[cursor : cursor + layout.mask_len]
    cursor += layout.mask_len
    model = payload[cursor : cursor + layout.model_len]
    cursor += layout.model_len
    actions = payload[cursor : cursor + layout.actions_len]
    cursor += layout.actions_len
    pose = payload[cursor : cursor + layout.pose_len]
    cursor += layout.pose_len
    if cursor != len(payload):
        raise ValueError(f"payload slice accounting mismatch: cursor={cursor}, len={len(payload)}")
    return {
        "masks.mkv.br": mask,
        "renderer.bin.br": model,
        "seg_tile_actions.br": actions,
        "optimized_poses.qp1.br": pose,
    }


def _decode_uvarint(raw: bytes, cursor: int) -> tuple[int, int]:
    shift = 0
    value = 0
    while cursor < len(raw):
        byte = raw[cursor]
        cursor += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, cursor
        shift += 7
        if shift > 63:
            break
    raise ValueError("truncated or overlong uvarint")


def parse_seg_tile_actions(decoded: bytes) -> dict[str, Any]:
    records: list[tuple[int, int, int]] = []
    if decoded.startswith(b"SG2") or (len(decoded) % 4 != 0 and len(decoded) % 5 != 0):
        cursor = 3 if decoded.startswith(b"SG2") else 0
        while cursor < len(decoded):
            tile, cursor = _decode_uvarint(decoded, cursor)
            count, cursor = _decode_uvarint(decoded, cursor)
            frame = 0
            for idx in range(count):
                delta, cursor = _decode_uvarint(decoded, cursor)
                frame = delta if idx == 0 else frame + delta
                if cursor >= len(decoded):
                    raise ValueError("seg tile action stream ended before action byte")
                action = decoded[cursor]
                cursor += 1
                records.append((frame, tile, action))
        fmt = "sg2_tile_group_varint" if decoded.startswith(b"SG2") else "tile_group_varint"
    elif len(decoded) % 4 == 0:
        for offset in range(0, len(decoded), 4):
            records.append(
                (
                    int.from_bytes(decoded[offset : offset + 2], "little"),
                    decoded[offset + 2],
                    decoded[offset + 3],
                )
            )
        fmt = "fixed4_frame_u16_tile_u8_action_u8"
    elif len(decoded) % 5 == 0:
        for offset in range(0, len(decoded), 5):
            records.append(
                (
                    int.from_bytes(decoded[offset : offset + 2], "little"),
                    int.from_bytes(decoded[offset + 2 : offset + 4], "little"),
                    decoded[offset + 4],
                )
            )
        fmt = "fixed5_frame_u16_tile_u16_action_u8"
    else:
        raise ValueError(f"unsupported seg tile action payload length: {len(decoded)}")

    frames = [record[0] for record in records]
    tiles = [record[1] for record in records]
    actions = [record[2] for record in records]
    return {
        "format": fmt,
        "decoded_bytes": len(decoded),
        "record_count": len(records),
        "unique_frame_count": len(set(frames)),
        "unique_tile_count": len(set(tiles)),
        "unique_action_count": len(set(actions)),
        "frame_min": min(frames) if frames else None,
        "frame_max": max(frames) if frames else None,
        "tile_min": min(tiles) if tiles else None,
        "tile_max": max(tiles) if tiles else None,
        "action_min": min(actions) if actions else None,
        "action_max": max(actions) if actions else None,
        "first_records": [
            {"frame": frame, "tile": tile, "action": action}
            for frame, tile, action in records[:8]
        ],
    }


def parse_qp1_pose(decoded: bytes) -> dict[str, Any]:
    if decoded.startswith(b"QP1"):
        if len(decoded) < 5:
            raise ValueError("truncated QP1 pose stream")
        first = int(np_uint16_le(decoded[3:5]))
        cursor = 5
        values = [first]
        while cursor < len(decoded):
            acc, cursor = _decode_uvarint(decoded, cursor)
            delta = (acc >> 1) ^ -(acc & 1)
            values.append(values[-1] + delta)
        return {
            "format": "QP1_velocity_delta_varint",
            "decoded_bytes": len(decoded),
            "pose_rows": len(values),
            "runtime_pose_dim": 6,
            "encoded_columns": ["velocity_col0"],
            "non_velocity_columns_fixed_zero": True,
            "first_velocity_q": first,
            "velocity_q_min": min(values),
            "velocity_q_max": max(values),
            "velocity_q_sha256": _sha256_bytes(struct.pack("<" + "i" * len(values), *values)),
        }
    if len(decoded) % 12 != 0:
        raise ValueError("qpose uint16 stream is not divisible by 6 columns")
    return {
        "format": "qpose14_uint16_6col",
        "decoded_bytes": len(decoded),
        "pose_rows": len(decoded) // 12,
        "runtime_pose_dim": 6,
        "encoded_columns": ["velocity_col0", "pose_col1", "pose_col2", "pose_col3", "pose_col4", "pose_col5"],
        "non_velocity_columns_fixed_zero": False,
        "raw_sha256": _sha256_bytes(decoded),
    }


def np_uint16_le(data: bytes) -> int:
    return struct.unpack("<H", data)[0]


def _segment_profile(name: str, charged: bytes, decoded: bytes | None, codec: str) -> dict[str, Any]:
    profile = {
        "name": name,
        "codec": codec,
        "charged_bytes": len(charged),
        "charged_sha256": _sha256_bytes(charged),
        "charged_prefix_hex": charged[:8].hex(),
    }
    if decoded is not None:
        profile.update(
            {
                "decoded_bytes": len(decoded),
                "decoded_sha256": _sha256_bytes(decoded),
                "decoded_prefix_hex": decoded[:8].hex(),
            }
        )
    return profile


def profile_payload(payload: bytes, *, layout: SegmentLayout | None = None) -> dict[str, Any]:
    layout = layout or infer_pr94_layout(payload)
    slices = _slice_payload(payload, layout)
    decoded_mask = brotli.decompress(slices["masks.mkv.br"])
    decoded_model = brotli.decompress(slices["renderer.bin.br"])
    decoded_actions = brotli.decompress(slices["seg_tile_actions.br"]) if slices["seg_tile_actions.br"] else b""
    decoded_pose = brotli.decompress(slices["optimized_poses.qp1.br"])

    segments = [
        _segment_profile("masks.mkv", slices["masks.mkv.br"], decoded_mask, "brotli_av1_obu_mask_video"),
        _segment_profile("renderer.bin", slices["renderer.bin.br"], decoded_model, "brotli_qzs_renderer"),
        _segment_profile("seg_tile_actions.bin", slices["seg_tile_actions.br"], decoded_actions, "brotli_seg_tile_actions"),
        _segment_profile("optimized_poses.qp1", slices["optimized_poses.qp1.br"], decoded_pose, "brotli_qpose_qp1"),
    ]
    segment_map = {row["name"]: row for row in segments}
    return {
        "payload": {
            "payload_bytes": len(payload),
            "payload_sha256": _sha256_bytes(payload),
            "payload_prefix_hex": payload[:8].hex(),
            "payload_format": layout.payload_format,
            "boundary_authority": layout.boundary_authority,
            "header_bytes": layout.header_bytes,
        },
        "segments": segments,
        "layout": {
            "mask_bytes": layout.mask_len,
            "model_bytes": layout.model_len,
            "actions_bytes": layout.actions_len,
            "pose_bytes": layout.pose_len,
            "sum_segment_bytes": sum(row["charged_bytes"] for row in segments),
        },
        "classification": {
            "contest_faithfulness": "external_mps_report_only_until_cuda_replay",
            "score_claim": False,
            "renderer_magic": decoded_model[:4].decode("ascii", errors="replace"),
            "qpose": parse_qp1_pose(decoded_pose),
            "tile_actions": parse_seg_tile_actions(decoded_actions) if decoded_actions else None,
            "mask_decoded_sha256": segment_map["masks.mkv"]["decoded_sha256"],
            "renderer_decoded_sha256": segment_map["renderer.bin"]["decoded_sha256"],
        },
    }


def _reported_mps_summary(pr_json: Path | None) -> dict[str, Any] | None:
    if pr_json is None or not pr_json.is_file():
        return None
    payload = json.loads(pr_json.read_text(encoding="utf-8"))
    body = str(payload.get("body", ""))

    def number(pattern: str) -> float | None:
        match = re.search(pattern, body, re.IGNORECASE)
        if not match:
            return None
        return float(match.group(1).replace(",", ""))

    archive_bytes = number(r"Submission file size:\s*([0-9,]+)")
    pose = number(r"Average PoseNet Distortion:\s*([0-9.]+)")
    seg = number(r"Average SegNet Distortion:\s*([0-9.]+)")
    score = (
        100.0 * seg + math.sqrt(10.0 * pose) + 25.0 * archive_bytes / CONTEST_ORIGINAL_BYTES
        if archive_bytes is not None and pose is not None and seg is not None
        else None
    )
    return {
        "source": _rel(pr_json),
        "pr_number": payload.get("number"),
        "title": payload.get("title"),
        "state": payload.get("state"),
        "head_sha": ((payload.get("head") or {}).get("sha")),
        "updated_at": payload.get("updated_at"),
        "device": "mps",
        "archive_bytes": int(archive_bytes) if archive_bytes is not None else None,
        "pose_dist": pose,
        "seg_dist": seg,
        "recomputed_report_score": score,
        "promotion_validity": "invalid_for_promotion_mps_not_cuda",
    }


def stackability_findings(profile: dict[str, Any]) -> list[dict[str, Any]]:
    layout = profile["layout"]
    return [
        {
            "surface": "pose_side_qp1_velocity",
            "charged_bytes": layout["pose_bytes"],
            "static_delta_vs_pr85_stbm_pose_bytes": layout["pose_bytes"] - 1487,
            "rate_score_delta_if_isolated": _rate_score_delta(layout["pose_bytes"] - 1487),
            "verdict": "blocked_not_isolated",
            "reason": "PR94 encodes only velocity col0 and relies on its own qpose runtime path; PR85/STBM uses a different pose contract, so this is not a drop-in pose-side stack without decode/reencode and runtime-output parity.",
        },
        {
            "surface": "renderer_qzs3_model",
            "charged_bytes": layout["model_bytes"],
            "static_delta_vs_pr85_stbm_model_bytes": layout["model_bytes"] - 57074,
            "rate_score_delta_if_isolated": _rate_score_delta(layout["model_bytes"] - 57074),
            "verdict": "blocked_coupled_model_mask_pose",
            "reason": "The smaller QZS3 model is trained for PR94 masks/pose/actions; transplanting it onto PR85_STBM1BR/RMB1 would be a renderer replacement, not an isolated non-mask recode.",
        },
        {
            "surface": "tile_actions_control",
            "charged_bytes": layout["actions_bytes"],
            "static_delta_vs_no_tile_actions": layout["actions_bytes"],
            "rate_score_delta_if_added": _rate_score_delta(layout["actions_bytes"]),
            "verdict": "not_rate_stackable",
            "reason": "Tile actions add charged bytes and only become useful if exact CUDA component gain exceeds their rate cost; PR94 only supplies MPS evidence.",
        },
        {
            "surface": "mask_stream",
            "charged_bytes": layout["mask_bytes"],
            "static_delta_vs_pr85_stbm_mask_bytes": layout["mask_bytes"] - 152439,
            "rate_score_delta_if_replacing_stbm_mask": _rate_score_delta(layout["mask_bytes"] - 152439),
            "verdict": "do_not_stack_onto_stbm",
            "reason": "Replacing PR85_STBM1BR's lossless mask recode with PR94's full mask stream gives back the STBM byte win and changes the scorer-visible mask basin.",
        },
    ]


def build_profile(archive: Path, *, pr_json: Path | None = DEFAULT_PR_JSON) -> dict[str, Any]:
    archive_info, payload = _read_single_member_zip(archive)
    payload_profile = profile_payload(payload)
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "evidence_grade": "empirical_static_archive_intake",
        "notes": [
            "No scorer was run.",
            "No remote GPU job was dispatched.",
            "MPS report text is invalid for promotion; exact CUDA auth eval remains the score truth.",
        ],
        "archive": archive_info,
        **payload_profile,
        "reported_pr_text": _reported_mps_summary(pr_json),
        "stackability": stackability_findings(payload_profile),
    }


def build_markdown(profile: dict[str, Any]) -> str:
    lines = [
        "# PR94 Qpose Static Intake",
        "",
        f"- archive: `{profile['archive']['path']}`",
        f"- archive_bytes: `{profile['archive']['archive_bytes']}`",
        f"- archive_sha256: `{profile['archive']['archive_sha256']}`",
        f"- member: `{profile['archive']['member_name']}` stored={profile['archive']['zip_stored']} bytes={profile['archive']['member_bytes']}",
        f"- payload_format: `{profile['payload']['payload_format']}`",
        f"- evidence_grade: `{profile['evidence_grade']}`",
        f"- score_claim: `{profile['score_claim']}`",
        "",
        "## Segments",
        "",
        "| segment | charged bytes | charged sha256 | decoded bytes | decoded prefix |",
        "| --- | ---: | --- | ---: | --- |",
    ]
    for row in profile["segments"]:
        lines.append(
            f"| `{row['name']}` | {row['charged_bytes']} | `{row['charged_sha256']}` | "
            f"{row.get('decoded_bytes')} | `{row.get('decoded_prefix_hex')}` |"
        )
    lines.extend(["", "## Stackability", ""])
    for row in profile["stackability"]:
        delta = row.get("static_delta_vs_pr85_stbm_pose_bytes")
        delta = row.get("static_delta_vs_pr85_stbm_model_bytes", delta)
        delta = row.get("static_delta_vs_pr85_stbm_mask_bytes", delta)
        delta = row.get("static_delta_vs_no_tile_actions", delta)
        lines.append(f"- `{row['surface']}`: `{row['verdict']}`, byte_delta `{delta}`. {row['reason']}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--pr-json", type=Path, default=DEFAULT_PR_JSON)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    args = parser.parse_args(argv)

    profile = build_profile(args.archive, pr_json=args.pr_json)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(_json_text(profile), encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(build_markdown(profile), encoding="utf-8")
    print(_json_text(profile))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
