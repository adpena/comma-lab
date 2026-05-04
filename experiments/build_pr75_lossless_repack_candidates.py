#!/usr/bin/env python3
"""Build lossless C082/PR75 fixed-slice repack candidates.

This is a local byte-screening tool.  It preserves decoded masks, renderer,
poses, and PR75 tile-action runtime records, then emits deterministic stored-ZIP
archives plus custody manifests.  The outputs remain ``score_claim=false`` until
exact CUDA auth eval runs on the exact bytes.
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
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MEMBER_NAME = "p"
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489
TOOL = "experiments/build_pr75_lossless_repack_candidates.py"
CUDA_AUTH_EVAL_REQUIRED = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)


@dataclass(frozen=True)
class BrotliChoice:
    bytes: bytes
    quality: int
    mode: int
    lgwin: int
    lgblock: int

    @property
    def params(self) -> dict[str, int]:
        return {
            "quality": self.quality,
            "mode": self.mode,
            "lgwin": self.lgwin,
            "lgblock": self.lgblock,
        }


@dataclass(frozen=True)
class P3Slices:
    mask_br: bytes
    model_br: bytes
    actions_br: bytes
    pose_br: bytes


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _load_unpacker() -> Any:
    spec = importlib.util.spec_from_file_location("pr75_lossless_repack_unpacker", UNPACKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load unpacker from {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_single_member_payload(path: Path, *, member_name: str = MEMBER_NAME) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != [member_name]:
            raise ValueError(f"{path} must contain single member {member_name!r}; got {names!r}")
        return zf.read(infos[0])


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


def _parse_p3_payload(payload: bytes) -> P3Slices:
    if not payload.startswith(b"P3"):
        raise ValueError(f"source payload must be P3; got {payload[:2]!r}")
    header_size = 2 + struct.calcsize("<IHH")
    if len(payload) <= header_size:
        raise ValueError("P3 payload too short")
    mask_len, model_len, actions_len = struct.unpack_from("<IHH", payload, 2)
    cursor = header_size
    if min(mask_len, model_len, actions_len) <= 0:
        raise ValueError("P3 payload contains an empty required slice")
    if cursor + mask_len + model_len + actions_len >= len(payload):
        raise ValueError("P3 payload slice lengths leave no pose stream")
    mask_br = payload[cursor : cursor + mask_len]
    cursor += mask_len
    model_br = payload[cursor : cursor + model_len]
    cursor += model_len
    actions_br = payload[cursor : cursor + actions_len]
    cursor += actions_len
    return P3Slices(
        mask_br=mask_br,
        model_br=model_br,
        actions_br=actions_br,
        pose_br=payload[cursor:],
    )


def _compress(raw: bytes, params: tuple[int, int, int, int]) -> bytes:
    quality, mode, lgwin, lgblock = params
    return brotli.compress(
        raw,
        quality=quality,
        mode=mode,
        lgwin=lgwin,
        lgblock=lgblock,
    )


def default_brotli_param_grid() -> list[tuple[int, int, int, int]]:
    """Return a deterministic grid focused on the C082 stream sizes."""
    params: list[tuple[int, int, int, int]] = []
    for quality in (11, 10, 9, 8, 6, 4, 2, 0):
        for mode in (0, 1, 2):
            for lgwin in range(10, 25):
                for lgblock in (0, 16, 17, 18, 19, 20, 21, 22, 23, 24):
                    if lgblock and lgblock > lgwin:
                        continue
                    params.append((quality, mode, lgwin, lgblock))
    return params


def exhaustive_brotli_param_grid() -> list[tuple[int, int, int, int]]:
    params: list[tuple[int, int, int, int]] = []
    for quality in range(11, -1, -1):
        for mode in (0, 1, 2):
            for lgwin in range(10, 25):
                for lgblock in (0, 16, 17, 18, 19, 20, 21, 22, 23, 24):
                    if lgblock and lgblock > lgwin:
                        continue
                    params.append((quality, mode, lgwin, lgblock))
    return params


def best_brotli(raw: bytes, source_br: bytes, params: Iterable[tuple[int, int, int, int]]) -> BrotliChoice:
    best = BrotliChoice(source_br, -1, -1, -1, -1)
    for param in params:
        candidate = _compress(raw, param)
        if len(candidate) < len(best.bytes):
            quality, mode, lgwin, lgblock = param
            best = BrotliChoice(candidate, quality, mode, lgwin, lgblock)
    if brotli.decompress(best.bytes) != raw:
        raise ValueError("selected Brotli stream failed round-trip")
    return best


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
    """Encode sorted P3 action records as pair-delta varints plus tile/action."""
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


def _build_p3_payload(slices: P3Slices) -> bytes:
    return (
        b"P3"
        + struct.pack("<IHH", len(slices.mask_br), len(slices.model_br), len(slices.actions_br))
        + slices.mask_br
        + slices.model_br
        + slices.actions_br
        + slices.pose_br
    )


def _build_p6_payload(
    *,
    mask_br: bytes,
    model_br: bytes,
    actions_delta_varint_br: bytes,
    record_count: int,
    pose_br: bytes,
) -> bytes:
    return (
        b"P6"
        + struct.pack(
            "<IHHH",
            len(mask_br),
            len(model_br),
            len(actions_delta_varint_br),
            record_count,
        )
        + mask_br
        + model_br
        + actions_delta_varint_br
        + pose_br
    )


def _member_summary(header: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in header.get("members", []):
        name = str(item["name"])
        out[name] = {
            "bytes": int(item["bytes"]),
            "sha256": str(item["sha256"]),
            "codec": str(item["codec"]),
            "decoded_bytes": int(item["decoded_bytes"]),
            "decoded_sha256": str(item["decoded_sha256"]),
        }
    return out


def _decoded_member_summary(decoded: dict[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "bytes": len(data),
            "sha256": _sha256_bytes(data),
        }
        for name, data in sorted(decoded.items())
    }


def _candidate_decoded_member_summary(
    runtime_members: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "bytes": int(meta["decoded_bytes"]),
            "sha256": str(meta["decoded_sha256"]),
        }
        for name, meta in sorted(runtime_members.items())
    }


def _validate_payload_parity(
    *,
    payload: bytes,
    source_decoded: dict[str, bytes],
    unpacker: Any,
) -> tuple[str, dict[str, dict[str, Any]]]:
    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    missing = sorted(set(source_decoded) - set(decoded))
    extra = sorted(set(decoded) - set(source_decoded))
    if missing or extra:
        raise ValueError(f"decoded member mismatch: missing={missing} extra={extra}")
    for name, expected in source_decoded.items():
        actual = decoded[name]
        if actual != expected:
            raise ValueError(
                f"decoded member {name} changed: "
                f"expected={_sha256_bytes(expected)} actual={_sha256_bytes(actual)}"
            )
    return str(header.get("payload_format")), _member_summary(header)


def _candidate_manifest(
    *,
    candidate_id: str,
    source_archive: Path,
    source_payload: bytes,
    output_archive: Path,
    payload: bytes,
    payload_format: str,
    runtime_members: dict[str, dict[str, Any]],
    stream_choices: dict[str, BrotliChoice],
    actions_delta_varint_raw: bytes | None,
    actions_delta_varint_br: bytes | None,
    source_decoded: dict[str, bytes],
) -> dict[str, Any]:
    archive_bytes = output_archive.stat().st_size
    source_archive_bytes = source_archive.stat().st_size
    output_archive_sha256 = _sha256_file(output_archive)
    source_archive_sha256 = _sha256_file(source_archive)
    source_payload_sha256 = _sha256_bytes(source_payload)
    payload_sha256 = _sha256_bytes(payload)
    archive_delta = archive_bytes - source_archive_bytes
    source_decoded_summary = _decoded_member_summary(source_decoded)
    candidate_decoded_summary = _candidate_decoded_member_summary(runtime_members)
    payload_byte_identical_to_source = payload == source_payload
    archive_byte_identical_to_source = (
        archive_bytes == source_archive_bytes
        and output_archive_sha256 == source_archive_sha256
    )
    manifest: dict[str, Any] = {
        "candidate_id": candidate_id,
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "decoded_stream_parity": True,
        "decoded_stream_parity_detail": {
            "candidate_decoded_members": candidate_decoded_summary,
            "members_compared": sorted(source_decoded),
            "source_decoded_members": source_decoded_summary,
            "status": "passed",
        },
        "evidence_grade": "empirical_lossless_byte_transform",
        "formula_only_rate_score_delta_vs_source": archive_delta * RATE_SCORE_PER_BYTE,
        "noop": bool(payload_byte_identical_to_source and archive_byte_identical_to_source),
        "noop_status": (
            "byte_identical_to_source_archive"
            if payload_byte_identical_to_source and archive_byte_identical_to_source
            else "not_noop_repacked_payload"
        ),
        "output_archive": {
            "bytes": archive_bytes,
            "path": str(output_archive),
            "sha256": output_archive_sha256,
        },
        "payload": {
            "bytes": len(payload),
            "format": payload_format,
            "member": MEMBER_NAME,
            "sha256": payload_sha256,
        },
        "promotion_eligible": False,
        "runtime_parse_validation": {
            "parser": str(UNPACKER_PATH),
            "payload_format": payload_format,
            "members": runtime_members,
        },
        "schema": "pr75_lossless_repack_candidate_manifest_v1",
        "score_claim": False,
        "source_archive": {
            "bytes": source_archive_bytes,
            "path": str(source_archive),
            "sha256": source_archive_sha256,
        },
        "source_payload": {
            "bytes": len(source_payload),
            "sha256": source_payload_sha256,
        },
        "source_preservation": {
            "archive_byte_identical_to_source": archive_byte_identical_to_source,
            "candidate_archive_sha256": output_archive_sha256,
            "candidate_payload_sha256": payload_sha256,
            "decoded_streams_byte_identical": True,
            "payload_byte_identical_to_source": payload_byte_identical_to_source,
            "source_archive_sha256": source_archive_sha256,
            "source_payload_sha256": source_payload_sha256,
            "status": (
                "byte_identical_noop"
                if payload_byte_identical_to_source and archive_byte_identical_to_source
                else "lossless_decoded_stream_preserving_repack"
            ),
        },
        "source_preserving": True,
        "stream_choices": {
            name: {
                "bytes": len(choice.bytes),
                "params": choice.params if choice.quality >= 0 else "source",
                "sha256": _sha256_bytes(choice.bytes),
            }
            for name, choice in stream_choices.items()
        },
        "decoded_members": source_decoded_summary,
        "archive_delta_bytes_vs_source": archive_delta,
        "tool": TOOL,
    }
    if actions_delta_varint_raw is not None and actions_delta_varint_br is not None:
        manifest["actions_delta_varint"] = {
            "raw_bytes": len(actions_delta_varint_raw),
            "raw_sha256": _sha256_bytes(actions_delta_varint_raw),
            "brotli_bytes": len(actions_delta_varint_br),
            "brotli_sha256": _sha256_bytes(actions_delta_varint_br),
            "record_count": len(source_decoded["seg_tile_actions.bin"]) // 4,
            "decoded_runtime_records_sha256": _sha256_bytes(
                source_decoded["seg_tile_actions.bin"]
            ),
        }
    if not math.isfinite(float(manifest["formula_only_rate_score_delta_vs_source"])):
        raise ValueError("non-finite formula-only rate score delta")
    return manifest


def build_lossless_candidates(
    *,
    source_archive: Path,
    output_dir: Path,
    force: bool = False,
    params: Iterable[tuple[int, int, int, int]] | None = None,
    unpacker: Any | None = None,
) -> dict[str, Any]:
    source_archive = source_archive.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if unpacker is None:
        unpacker = _load_unpacker()
    if params is None:
        params = default_brotli_param_grid()
    params = list(params)

    source_payload = _read_single_member_payload(source_archive)
    source_slices = _parse_p3_payload(source_payload)
    source_header, source_decoded = unpacker._parse_payload(source_payload)  # noqa: SLF001
    source_payload_format = str(source_header.get("payload_format"))
    if source_payload_format != "public_pr75_qzs3_qp1_segactions_p3":
        raise ValueError(
            f"source must parse as public PR75 P3 actions payload, got {source_payload_format!r}"
        )

    raw_streams = {
        "masks.mkv": brotli.decompress(source_slices.mask_br),
        "renderer.bin": brotli.decompress(source_slices.model_br),
        "seg_tile_actions.bin": brotli.decompress(source_slices.actions_br),
        "optimized_poses.bin": brotli.decompress(source_slices.pose_br),
    }
    if raw_streams["seg_tile_actions.bin"] != source_decoded["seg_tile_actions.bin"]:
        raise ValueError("source P3 action stream does not match runtime decoded actions")

    best_mask = best_brotli(raw_streams["masks.mkv"], source_slices.mask_br, params)
    best_model = best_brotli(raw_streams["renderer.bin"], source_slices.model_br, params)
    best_actions = best_brotli(
        raw_streams["seg_tile_actions.bin"],
        source_slices.actions_br,
        params,
    )
    best_pose = best_brotli(raw_streams["optimized_poses.bin"], source_slices.pose_br, params)
    action_delta_raw = encode_delta_varint_actions(raw_streams["seg_tile_actions.bin"])
    action_delta_br = best_brotli(action_delta_raw, brotli.compress(action_delta_raw, quality=11), params)
    record_count = len(raw_streams["seg_tile_actions.bin"]) // 4

    candidates: list[tuple[str, bytes, dict[str, BrotliChoice], bytes | None, bytes | None]] = []
    p3_resweep = _build_p3_payload(
        P3Slices(
            mask_br=best_mask.bytes,
            model_br=best_model.bytes,
            actions_br=best_actions.bytes,
            pose_br=best_pose.bytes,
        )
    )
    candidates.append(
        (
            "c082_p3_stream_resweep",
            p3_resweep,
            {
                "masks.mkv": best_mask,
                "renderer.bin": best_model,
                "seg_tile_actions.bin": best_actions,
                "optimized_poses.bin": best_pose,
            },
            None,
            None,
        )
    )
    p6_actions = _build_p6_payload(
        mask_br=source_slices.mask_br,
        model_br=source_slices.model_br,
        actions_delta_varint_br=action_delta_br.bytes,
        record_count=record_count,
        pose_br=source_slices.pose_br,
    )
    candidates.append(
        (
            "c082_p6_delta_varint_actions",
            p6_actions,
            {
                "masks.mkv": BrotliChoice(source_slices.mask_br, -1, -1, -1, -1),
                "renderer.bin": BrotliChoice(source_slices.model_br, -1, -1, -1, -1),
                "seg_tile_actions.delta_varint": action_delta_br,
                "optimized_poses.bin": BrotliChoice(source_slices.pose_br, -1, -1, -1, -1),
            },
            action_delta_raw,
            action_delta_br.bytes,
        )
    )
    p6_resweep = _build_p6_payload(
        mask_br=best_mask.bytes,
        model_br=best_model.bytes,
        actions_delta_varint_br=action_delta_br.bytes,
        record_count=record_count,
        pose_br=best_pose.bytes,
    )
    candidates.append(
        (
            "c082_p6_delta_varint_actions_stream_resweep",
            p6_resweep,
            {
                "masks.mkv": best_mask,
                "renderer.bin": best_model,
                "seg_tile_actions.delta_varint": action_delta_br,
                "optimized_poses.bin": best_pose,
            },
            action_delta_raw,
            action_delta_br.bytes,
        )
    )

    source_archive_bytes = source_archive.stat().st_size
    rows: list[dict[str, Any]] = []
    for candidate_id, payload, stream_choices, delta_raw, delta_br in candidates:
        candidate_dir = output_dir / candidate_id
        archive_path = candidate_dir / "archive.zip"
        manifest_path = candidate_dir / "manifest.json"
        if archive_path.exists() and not force:
            raise FileExistsError(f"{archive_path} exists; pass --force")
        payload_format, runtime_members = _validate_payload_parity(
            payload=payload,
            source_decoded=source_decoded,
            unpacker=unpacker,
        )
        _write_archive(archive_path, payload)
        manifest = _candidate_manifest(
            candidate_id=candidate_id,
            source_archive=source_archive,
            source_payload=source_payload,
            output_archive=archive_path,
            payload=payload,
            payload_format=payload_format,
            runtime_members=runtime_members,
            stream_choices=stream_choices,
            actions_delta_varint_raw=delta_raw,
            actions_delta_varint_br=delta_br,
            source_decoded=source_decoded,
        )
        _write_json(manifest_path, manifest)
        rows.append(
            {
                "archive_bytes": manifest["output_archive"]["bytes"],
                "archive_delta_bytes_vs_source": manifest["archive_delta_bytes_vs_source"],
                "archive_path": manifest["output_archive"]["path"],
                "archive_sha256": manifest["output_archive"]["sha256"],
                "candidate_id": candidate_id,
                "decoded_stream_parity": True,
                "formula_only_rate_score_delta_vs_source": manifest[
                    "formula_only_rate_score_delta_vs_source"
                ],
                "manifest_path": str(manifest_path),
                "noop": manifest["noop"],
                "noop_status": manifest["noop_status"],
                "payload_bytes": manifest["payload"]["bytes"],
                "payload_format": manifest["payload"]["format"],
                "score_claim": False,
                "source_preservation_status": manifest["source_preservation"]["status"],
                "source_preserving": manifest["source_preserving"],
            }
        )

    summary = {
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "candidates": sorted(rows, key=lambda row: (row["archive_bytes"], row["candidate_id"])),
        "evidence_grade": "empirical_lossless_byte_transform",
        "noop_proof": {
            "source_payload_format": source_payload_format,
            "source_archive_bytes": source_archive_bytes,
            "source_archive_sha256": _sha256_file(source_archive),
            "source_payload_bytes": len(source_payload),
            "source_payload_sha256": _sha256_bytes(source_payload),
            "decoded_members": _decoded_member_summary(source_decoded),
        },
        "promotion_eligible": False,
        "schema": "pr75_lossless_repack_candidate_matrix_v1",
        "score_claim": False,
        "source_preservation_contract": {
            "decoded_streams_byte_identical_required": True,
            "noop_means_archive_and_payload_byte_identical_to_source": True,
            "source_preserving_repack_means_decoded_streams_byte_identical": True,
        },
        "tool": TOOL,
    }
    _write_json(output_dir / "candidate_matrix.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--exhaustive-brotli-grid",
        action="store_true",
        help="Search all Brotli quality levels instead of the focused default grid.",
    )
    args = parser.parse_args(argv)
    summary = build_lossless_candidates(
        source_archive=args.source_archive,
        output_dir=args.output_dir,
        force=bool(args.force),
        params=exhaustive_brotli_param_grid()
        if args.exhaustive_brotli_grid
        else default_brotli_param_grid(),
    )
    print(
        json.dumps(
            {
                "best_by_bytes": summary["candidates"][0],
                "candidate_count": len(summary["candidates"]),
                "output_dir": str(Path(args.output_dir).resolve()),
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
