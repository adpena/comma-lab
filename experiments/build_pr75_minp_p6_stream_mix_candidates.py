#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build deterministic PR75/minp P6 stream-mix candidates.

This is a local byte-screening tool only.  It mixes encoded PR75/minp public
streams with the current C089/P6 frontier, emits deterministic single-member
archives, and validates each candidate through the robust contest unpacker.
The outputs are not score evidence until exact CUDA auth eval runs on the exact
archive bytes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import brotli


REPO_ROOT = Path(__file__).resolve().parents[1]
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
DEFAULT_C089_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip"
)
DEFAULT_PUBLIC_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/archive.zip"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/pr75_minp_p6_stream_mix_worker_20260503"
TOOL = "experiments/build_pr75_minp_p6_stream_mix_candidates.py"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MEMBER_NAME = "p"
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
BASELINE_SCORE = 0.3154707273953505  # [external: PR-75 contest-CUDA T4 anchor (== PR-65 frontier)]
BASELINE_BYTES = 276_342
CUDA_AUTH_EVAL_REQUIRED = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)


@dataclass(frozen=True)
class EncodedStreams:
    mask_br: bytes
    renderer_br: bytes
    actions_br: bytes
    pose_br: bytes
    action_record_count: int | None


@dataclass(frozen=True)
class ActionChoice:
    source_label: str
    encoded: bytes
    raw_delta_varint: bytes
    record_count: int
    params: dict[str, int] | str
    source_encoded_sha256: str | None = None


@dataclass(frozen=True)
class SourceArchive:
    label: str
    path: Path
    archive_bytes: int
    archive_sha256: str
    payload: bytes
    payload_sha256: str
    payload_format: str
    encoded: EncodedStreams
    decoded: dict[str, bytes]
    runtime_members: dict[str, dict[str, Any]]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _load_unpacker() -> Any:
    spec = importlib.util.spec_from_file_location("pr75_minp_p6_stream_mix_unpacker", UNPACKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load unpacker from {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe_archive_member_name(name: str) -> str:
    path = Path(name)
    hidden = name.startswith(".") or name.startswith("__MACOSX/") or "/." in name
    resource_fork = name.startswith("._") or "/._" in name
    if hidden or resource_fork:
        raise ValueError(f"hidden/system archive member is forbidden: {name!r}")
    if not name or name.startswith("/") or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    return name


def _read_single_payload_member(path: Path) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        seen: set[str] = set()
        names: list[str] = []
        for info in zf.infolist():
            name = _safe_archive_member_name(info.filename)
            if name in seen:
                raise ValueError(f"duplicate ZIP member: {name}")
            seen.add(name)
            if not info.is_dir():
                names.append(name)
        if names != [MEMBER_NAME]:
            raise ValueError(f"{path} must contain exactly member {MEMBER_NAME!r}; got {names!r}")
        return zf.read(MEMBER_NAME)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info(MEMBER_NAME), payload)


def _parse_encoded_streams(payload: bytes) -> EncodedStreams:
    record_count: int | None = None
    if payload.startswith(b"P3"):
        cursor = 2 + struct.calcsize("<IHH")
        mask_len, renderer_len, actions_len = struct.unpack_from("<IHH", payload, 2)
    elif payload.startswith(b"P6"):
        cursor = 2 + struct.calcsize("<IHHH")
        mask_len, renderer_len, actions_len, record_count = struct.unpack_from(
            "<IHHH",
            payload,
            2,
        )
    elif len(payload) == 276_381:
        cursor = 0
        mask_len, renderer_len, actions_len = 219_472, 55_756, 255
    elif len(payload) == 276_379:
        cursor = 0
        mask_len, renderer_len, actions_len = 219_472, 55_756, 253
    elif len(payload) == 276_520:
        cursor = 0
        mask_len, renderer_len, actions_len = 219_472, 55_914, 236
    elif len(payload) == 276_641:
        cursor = 0
        mask_len, renderer_len, actions_len = 219_472, 56_034, 236
    else:
        raise ValueError(
            f"unsupported PR75/minp payload form: prefix={payload[:4]!r} len={len(payload)}"
        )
    if min(mask_len, renderer_len, actions_len) <= 0:
        raise ValueError("empty encoded stream in source payload")
    mask_end = cursor + mask_len
    renderer_end = mask_end + renderer_len
    actions_end = renderer_end + actions_len
    if actions_end >= len(payload):
        raise ValueError("encoded stream lengths leave no pose payload")
    return EncodedStreams(
        mask_br=payload[cursor:mask_end],
        renderer_br=payload[mask_end:renderer_end],
        actions_br=payload[renderer_end:actions_end],
        pose_br=payload[actions_end:],
        action_record_count=record_count,
    )


def _member_summary(header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in header.get("members", []):
        name = str(item["name"])
        out[name] = {
            "bytes": int(item["bytes"]),
            "codec": str(item["codec"]),
            "decoded_bytes": int(item["decoded_bytes"]),
            "decoded_sha256": str(item["decoded_sha256"]),
            "sha256": str(item["sha256"]),
        }
    return out


def _decoded_summary(decoded: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {"bytes": len(data), "sha256": _sha256_bytes(data)}
        for name, data in sorted(decoded.items())
    }


def _load_source(label: str, path: Path, unpacker: Any) -> SourceArchive:
    path = path.resolve()
    payload = _read_single_payload_member(path)
    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    encoded = _parse_encoded_streams(payload)
    return SourceArchive(
        label=label,
        path=path,
        archive_bytes=path.stat().st_size,
        archive_sha256=_sha256_file(path),
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        payload_format=str(header.get("payload_format")),
        encoded=encoded,
        decoded=decoded,
        runtime_members=_member_summary(header),
    )


def _read_action_records(raw_actions: bytes) -> list[tuple[int, int, int]]:
    if len(raw_actions) % 4:
        raise ValueError(f"PR75 action records must be 4 bytes each, got {len(raw_actions)}")
    records: list[tuple[int, int, int]] = []
    for offset in range(0, len(raw_actions), 4):
        pair_index = int.from_bytes(raw_actions[offset : offset + 2], "little")
        records.append((pair_index, raw_actions[offset + 2], raw_actions[offset + 3]))
    return records


def _uleb128(value: int) -> bytes:
    if value < 0:
        raise ValueError(f"cannot varint-encode negative value {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def encode_delta_varint_actions(raw_actions: bytes) -> bytes:
    """Encode runtime action records as P6 pair-delta varints."""
    records = _read_action_records(raw_actions)
    out = bytearray()
    previous_pair = 0
    for index, (pair_index, tile_id, action_id) in enumerate(records):
        delta = pair_index if index == 0 else pair_index - previous_pair
        if delta < 0:
            raise ValueError("P6 delta-varint action encoding requires nondecreasing pairs")
        out.extend(_uleb128(delta))
        out.append(tile_id)
        out.append(action_id)
        previous_pair = pair_index
    return bytes(out)


def default_brotli_param_grid() -> list[tuple[int, int, int, int]]:
    params: list[tuple[int, int, int, int]] = []
    for quality in (11, 10, 9, 8, 6, 4, 2, 0):
        for mode in (0, 1, 2):
            for lgwin in range(10, 25):
                for lgblock in (0, 16, 17, 18, 19, 20, 21, 22, 23, 24):
                    if lgblock and lgblock > lgwin:
                        continue
                    params.append((quality, mode, lgwin, lgblock))
    return params


def _compress(raw: bytes, params: tuple[int, int, int, int]) -> bytes:
    quality, mode, lgwin, lgblock = params
    return brotli.compress(raw, quality=quality, mode=mode, lgwin=lgwin, lgblock=lgblock)


def _best_brotli(
    raw: bytes,
    *,
    source: bytes | None,
    params: Iterable[tuple[int, int, int, int]],
) -> tuple[bytes, dict[str, int] | str]:
    best = source
    best_params: dict[str, int] | str = "source" if source is not None else {}
    for param in params:
        candidate = _compress(raw, param)
        if best is None or len(candidate) < len(best):
            quality, mode, lgwin, lgblock = param
            best = candidate
            best_params = {
                "quality": quality,
                "mode": mode,
                "lgwin": lgwin,
                "lgblock": lgblock,
            }
    if best is None:
        raise ValueError("no Brotli candidate generated")
    if brotli.decompress(best) != raw:
        raise ValueError("selected Brotli stream failed round-trip")
    return best, best_params


def _action_choice(
    source: SourceArchive,
    *,
    params: Iterable[tuple[int, int, int, int]],
) -> ActionChoice:
    raw_actions = source.decoded["seg_tile_actions.bin"]
    delta_raw = encode_delta_varint_actions(raw_actions)
    source_br = source.encoded.actions_br if source.payload.startswith(b"P6") else None
    encoded, choice_params = _best_brotli(delta_raw, source=source_br, params=params)
    return ActionChoice(
        source_label=source.label,
        encoded=encoded,
        raw_delta_varint=delta_raw,
        record_count=len(raw_actions) // 4,
        params=choice_params,
        source_encoded_sha256=_sha256_bytes(source.encoded.actions_br),
    )


def _build_p6_payload(
    *,
    mask_br: bytes,
    renderer_br: bytes,
    actions_br: bytes,
    record_count: int,
    pose_br: bytes,
) -> bytes:
    return (
        b"P6"
        + struct.pack("<IHHH", len(mask_br), len(renderer_br), len(actions_br), record_count)
        + mask_br
        + renderer_br
        + actions_br
        + pose_br
    )


def _validate_candidate(
    *,
    payload: bytes,
    expected_decoded: dict[str, bytes],
    unpacker: Any,
) -> tuple[str, dict[str, dict[str, Any]]]:
    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    payload_format = str(header.get("payload_format"))
    if payload_format != "public_pr75_qzs3_qp1_segactions_p6_delta_varint":
        raise ValueError(f"candidate did not parse as P6: {payload_format!r}")
    missing = sorted(set(expected_decoded) - set(decoded))
    extra = sorted(set(decoded) - set(expected_decoded))
    if missing or extra:
        raise ValueError(f"decoded member mismatch: missing={missing} extra={extra}")
    for name, expected in expected_decoded.items():
        actual = decoded[name]
        if actual != expected:
            raise ValueError(
                f"decoded member {name} mismatch: "
                f"expected={_sha256_bytes(expected)} actual={_sha256_bytes(actual)}"
            )
    return payload_format, _member_summary(header)


def _stream_change_summary(
    *,
    c089: SourceArchive,
    streams: EncodedStreams,
    action_choice: ActionChoice,
    expected_decoded: dict[str, bytes],
) -> dict[str, Any]:
    encoded = {
        "mask": streams.mask_br,
        "renderer": streams.renderer_br,
        "actions": action_choice.encoded,
        "pose": streams.pose_br,
    }
    c089_encoded = {
        "mask": c089.encoded.mask_br,
        "renderer": c089.encoded.renderer_br,
        "actions": c089.encoded.actions_br,
        "pose": c089.encoded.pose_br,
    }
    decoded_name = {
        "mask": "masks.mkv",
        "renderer": "renderer.bin",
        "actions": "seg_tile_actions.bin",
        "pose": "optimized_poses.qp1",
    }
    encoded_changes = {
        name: {
            "changed_vs_c089": encoded[name] != c089_encoded[name],
            "candidate_bytes": len(encoded[name]),
            "candidate_sha256": _sha256_bytes(encoded[name]),
            "c089_bytes": len(c089_encoded[name]),
            "c089_sha256": _sha256_bytes(c089_encoded[name]),
        }
        for name in ("mask", "renderer", "actions", "pose")
    }
    decoded_changes = {
        name: {
            "changed_vs_c089": expected_decoded[decoded_name[name]]
            != c089.decoded[decoded_name[name]],
            "candidate_bytes": len(expected_decoded[decoded_name[name]]),
            "candidate_sha256": _sha256_bytes(expected_decoded[decoded_name[name]]),
            "c089_bytes": len(c089.decoded[decoded_name[name]]),
            "c089_sha256": _sha256_bytes(c089.decoded[decoded_name[name]]),
        }
        for name in ("mask", "renderer", "actions", "pose")
    }
    changed_encoded = [name for name, meta in encoded_changes.items() if meta["changed_vs_c089"]]
    changed_decoded = [name for name, meta in decoded_changes.items() if meta["changed_vs_c089"]]
    if changed_decoded:
        status = "decoded_stream_changed_vs_c089"
    elif changed_encoded:
        status = "encoded_stream_changed_without_decoded_change_vs_c089"
    else:
        status = "byte_identical_noop_vs_c089"
    return {
        "changed_decoded_streams_vs_c089": changed_decoded,
        "changed_encoded_streams_vs_c089": changed_encoded,
        "decoded_streams": decoded_changes,
        "encoded_streams": encoded_changes,
        "is_non_noop_vs_c089": bool(changed_decoded or changed_encoded),
        "status": status,
    }


def _source_summary(source: SourceArchive) -> dict[str, Any]:
    return {
        "archive_bytes": source.archive_bytes,
        "archive_sha256": source.archive_sha256,
        "decoded_members": _decoded_summary(source.decoded),
        "encoded_streams": {
            "actions": {
                "bytes": len(source.encoded.actions_br),
                "record_count": source.encoded.action_record_count,
                "sha256": _sha256_bytes(source.encoded.actions_br),
            },
            "mask": {
                "bytes": len(source.encoded.mask_br),
                "sha256": _sha256_bytes(source.encoded.mask_br),
            },
            "pose": {
                "bytes": len(source.encoded.pose_br),
                "sha256": _sha256_bytes(source.encoded.pose_br),
            },
            "renderer": {
                "bytes": len(source.encoded.renderer_br),
                "sha256": _sha256_bytes(source.encoded.renderer_br),
            },
        },
        "path": str(source.path),
        "payload_bytes": len(source.payload),
        "payload_format": source.payload_format,
        "payload_sha256": source.payload_sha256,
    }


def build_candidates(
    *,
    c089_archive: Path,
    public_archive: Path,
    output_dir: Path,
    force: bool = False,
    params: Iterable[tuple[int, int, int, int]] | None = None,
    unpacker: Any | None = None,
) -> dict[str, Any]:
    if unpacker is None:
        unpacker = _load_unpacker()
    if params is None:
        params = default_brotli_param_grid()
    params = list(params)
    c089 = _load_source("c089_p6_frontier", c089_archive, unpacker)
    public = _load_source("public_pr75_minp", public_archive, unpacker)
    sources = {"c089": c089, "public": public}
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    action_choices: dict[str, ActionChoice] = {}
    skipped: list[dict[str, Any]] = []
    for key, source in sources.items():
        try:
            action_choices[key] = _action_choice(source, params=params)
        except ValueError as exc:
            skipped.append(
                {
                    "candidate_scope": f"{key}_actions",
                    "reason": str(exc),
                    "source": source.label,
                    "status": "skipped_not_p6_delta_varint_encodable",
                }
            )

    plans = [
        ("p6_c089_action_resweep", "c089", "c089", "c089"),
        ("p6_public_renderer_only", "public", "c089", "c089"),
        ("p6_public_pose_only", "c089", "public", "c089"),
        ("p6_public_renderer_pose", "public", "public", "c089"),
        ("p6_public_actions_only", "c089", "c089", "public"),
        ("p6_public_renderer_actions", "public", "c089", "public"),
        ("p6_public_actions_pose", "c089", "public", "public"),
        ("p6_public_all", "public", "public", "public"),
    ]
    rows: list[dict[str, Any]] = []
    for candidate_id, renderer_source, pose_source, action_source in plans:
        if action_source not in action_choices:
            skipped.append(
                {
                    "candidate_id": candidate_id,
                    "reason": f"action source {action_source!r} is not P6 delta-varint encodable",
                    "selected_sources": {
                        "actions": action_source,
                        "mask": "c089",
                        "pose": pose_source,
                        "renderer": renderer_source,
                    },
                    "status": "skipped",
                }
            )
            continue
        action_choice = action_choices[action_source]
        streams = EncodedStreams(
            mask_br=c089.encoded.mask_br,
            renderer_br=sources[renderer_source].encoded.renderer_br,
            actions_br=action_choice.encoded,
            pose_br=sources[pose_source].encoded.pose_br,
            action_record_count=action_choice.record_count,
        )
        expected_decoded = {
            "masks.mkv": c089.decoded["masks.mkv"],
            "optimized_poses.qp1": sources[pose_source].decoded["optimized_poses.qp1"],
            "renderer.bin": sources[renderer_source].decoded["renderer.bin"],
            "seg_tile_actions.bin": sources[action_source].decoded["seg_tile_actions.bin"],
        }
        payload = _build_p6_payload(
            mask_br=streams.mask_br,
            renderer_br=streams.renderer_br,
            actions_br=streams.actions_br,
            record_count=action_choice.record_count,
            pose_br=streams.pose_br,
        )
        payload_format, runtime_members = _validate_candidate(
            payload=payload,
            expected_decoded=expected_decoded,
            unpacker=unpacker,
        )
        candidate_dir = output_dir / candidate_id
        archive_path = candidate_dir / "archive.zip"
        if archive_path.exists() and not force:
            raise FileExistsError(f"{archive_path} exists; pass --force")
        _write_archive(archive_path, payload)
        archive_bytes = archive_path.stat().st_size
        archive_sha256 = _sha256_file(archive_path)
        non_noop = _stream_change_summary(
            c089=c089,
            streams=streams,
            action_choice=action_choice,
            expected_decoded=expected_decoded,
        )
        manifest = {
            "candidate_id": candidate_id,
            "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
            "decoded_stream_closure": {
                "candidate_decoded_members": _decoded_summary(expected_decoded),
                "members_compared": sorted(expected_decoded),
                "runtime_parser_members": runtime_members,
                "status": "passed",
            },
            "delta_vs_c089": {
                "archive_bytes": archive_bytes - c089.archive_bytes,
                "baseline_score_if_components_unchanged": BASELINE_SCORE
                + (archive_bytes - BASELINE_BYTES) * RATE_SCORE_PER_BYTE,
                "formula_only_rate_score": (archive_bytes - c089.archive_bytes)
                * RATE_SCORE_PER_BYTE,
            },
            "evidence_grade": "empirical_byte_screen_only",
            "non_noop_proof": non_noop,
            "noop": not non_noop["is_non_noop_vs_c089"],
            "noop_status": non_noop["status"],
            "output_archive": {
                "bytes": archive_bytes,
                "path": str(archive_path),
                "sha256": archive_sha256,
            },
            "payload": {
                "bytes": len(payload),
                "format": payload_format,
                "member": MEMBER_NAME,
                "sha256": _sha256_bytes(payload),
            },
            "promotion_eligible": False,
            "runtime_parse_validation": {
                "members": runtime_members,
                "parser": str(UNPACKER_PATH),
                "payload_format": payload_format,
            },
            "schema": "pr75_minp_p6_stream_mix_candidate_manifest_v1",
            "score_claim": False,
            "selected_encoded_stream_sha256s": {
                "actions": _sha256_bytes(streams.actions_br),
                "mask": _sha256_bytes(streams.mask_br),
                "pose": _sha256_bytes(streams.pose_br),
                "renderer": _sha256_bytes(streams.renderer_br),
            },
            "selected_sources": {
                "actions": action_source,
                "mask": "c089",
                "pose": pose_source,
                "renderer": renderer_source,
            },
            "stream_bytes": {
                "actions_br": len(streams.actions_br),
                "mask_br": len(streams.mask_br),
                "pose_br": len(streams.pose_br),
                "renderer_br": len(streams.renderer_br),
            },
            "stream_packing": {
                "actions_delta_varint": {
                    "brotli_bytes": len(action_choice.encoded),
                    "brotli_params": action_choice.params,
                    "brotli_sha256": _sha256_bytes(action_choice.encoded),
                    "decoded_runtime_records_sha256": _sha256_bytes(
                        sources[action_source].decoded["seg_tile_actions.bin"]
                    ),
                    "raw_bytes": len(action_choice.raw_delta_varint),
                    "raw_sha256": _sha256_bytes(action_choice.raw_delta_varint),
                    "record_count": action_choice.record_count,
                    "source_encoded_sha256": action_choice.source_encoded_sha256,
                }
            },
            "source_archives": {
                "c089": _source_summary(c089),
                "public": _source_summary(public),
            },
            "tool": TOOL,
        }
        if not math.isfinite(float(manifest["delta_vs_c089"]["formula_only_rate_score"])):
            raise ValueError("non-finite formula-only rate score delta")
        manifest_path = candidate_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        rows.append(
            {
                "archive_bytes": archive_bytes,
                "archive_path": str(archive_path),
                "archive_sha256": archive_sha256,
                "candidate_id": candidate_id,
                "delta_bytes_vs_c089": archive_bytes - c089.archive_bytes,
                "formula_only_rate_score_delta_vs_c089": manifest["delta_vs_c089"][
                    "formula_only_rate_score"
                ],
                "manifest_path": str(manifest_path),
                "noop": manifest["noop"],
                "noop_status": manifest["noop_status"],
                "payload_bytes": len(payload),
                "payload_format": payload_format,
                "score_claim": False,
                "selected_sources": manifest["selected_sources"],
                "stream_bytes": manifest["stream_bytes"],
            }
        )

    summary = {
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "candidates": sorted(rows, key=lambda row: (row["archive_bytes"], row["candidate_id"])),
        "evidence_grade": "empirical_byte_screen_only",
        "promotion_eligible": False,
        "schema": "pr75_minp_p6_stream_mix_candidate_matrix_v1",
        "score_claim": False,
        "skipped_candidates": skipped,
        "source_archives": {
            "c089": _source_summary(c089),
            "public": _source_summary(public),
        },
        "tool": TOOL,
    }
    _write_json(output_dir / "candidate_matrix.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--c089-archive", type=Path, default=DEFAULT_C089_ARCHIVE)
    parser.add_argument("--public-archive", type=Path, default=DEFAULT_PUBLIC_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = build_candidates(
        c089_archive=args.c089_archive,
        public_archive=args.public_archive,
        output_dir=args.output_dir,
        force=bool(args.force),
    )
    print(
        json.dumps(
            {
                "best_by_bytes": summary["candidates"][0] if summary["candidates"] else None,
                "candidate_count": len(summary["candidates"]),
                "output_dir": str(Path(args.output_dir).resolve()),
                "score_claim": False,
                "skipped_count": len(summary["skipped_candidates"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
