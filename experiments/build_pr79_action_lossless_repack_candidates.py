#!/usr/bin/env python3
"""Build PR79 lossless seg_tile_actions repack candidates.

This is a local byte-screening tool only.  It preserves PR79 decoded
``seg_tile_actions.bin`` records exactly, keeps mask/renderer/pose streams
byte-identical, and changes only the charged action grammar/compression.
No GPU jobs are submitted and no score claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import brotli


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.archive_byte_profile import profile_archive
from tac.submission_archive import validate_seg_tile_actions_payload


DEFAULT_PR79_ARCHIVE = (
    REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr79/archive.zip"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/pr79_action_lossless_repack_20260503_codex"
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
TOOL = "experiments/build_pr79_action_lossless_repack_candidates.py"
SCHEMA = "pr79_action_lossless_repack_matrix_v1"
MANIFEST_SCHEMA = "pr79_action_lossless_repack_manifest_v1"
MEMBER_NAME = "p"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
SEGMENTS = ("masks.mkv", "renderer.bin", "seg_tile_actions.bin", "optimized_poses.qp1")
NON_ACTION_SEGMENTS = ("masks.mkv", "renderer.bin", "optimized_poses.qp1")
PR79_SHA256 = "01dc02badf851d99108fd92c570271f36f74cc5424c6d2a8f1b499cb4d1c3446"
C102_BYTES = 276_485
C102_SHA256 = "79091f2c3f0c30ef3ca512808f3adc0306010e7f57fed3a09b3664c16fea4ea8"
CUDA_AUTH_EVAL_REQUIRED = (
    "No dispatch from this worker. Before any exact eval, claim a non-conflicting "
    "lane with tools/claim_lane_dispatch.py claim, then run exact T4-equivalent "
    "CUDA auth eval on identical archive bytes through archive.zip -> inflate.sh "
    "-> upstream/evaluate.py via experiments/contest_auth_eval.py --device cuda "
    "with expected archive SHA and size."
)


@dataclass(frozen=True)
class SourceArchive:
    path: Path
    archive_bytes: int
    archive_sha256: str
    payload: bytes
    payload_sha256: str
    payload_format: str
    raw_segments: dict[str, bytes]
    decoded: dict[str, bytes]


@dataclass(frozen=True)
class BrotliChoice:
    data: bytes
    params: dict[str, int]


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


def _repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _load_unpacker() -> Any:
    spec = importlib.util.spec_from_file_location("pr79_lossless_unpacker", UNPACKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load unpacker from {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _safe_member_name(name: str) -> str:
    path = Path(name)
    if not name or name.startswith("/") or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"unsafe ZIP member path: {name!r}")
    if name.startswith(".") or name.startswith("__MACOSX/") or name.startswith("._"):
        raise ValueError(f"hidden/system ZIP member path: {name!r}")
    return name


def _read_single_payload(path: Path) -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        seen: set[str] = set()
        names: list[str] = []
        for info in zf.infolist():
            name = _safe_member_name(info.filename)
            if name in seen:
                raise ValueError(f"duplicate ZIP member: {name}")
            seen.add(name)
            if not info.is_dir():
                names.append(name)
        if names != [MEMBER_NAME]:
            raise ValueError(f"{path} must contain exactly {MEMBER_NAME!r}; got {names!r}")
        return zf.read(MEMBER_NAME)


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_safe_member_name(name), FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info(MEMBER_NAME), payload)


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


def _slice_fixed_payload(
    payload: bytes,
    runtime_members: dict[str, dict[str, Any]],
) -> dict[str, bytes]:
    offset = 0
    raw_segments: dict[str, bytes] = {}
    for name in SEGMENTS:
        size = int(runtime_members[name]["bytes"])
        raw = payload[offset:offset + size]
        if len(raw) != size:
            raise ValueError(f"truncated fixed slice for {name}")
        if _sha256_bytes(raw) != runtime_members[name]["sha256"]:
            raise ValueError(f"raw slice SHA mismatch for {name}")
        raw_segments[name] = raw
        offset += size
    if offset != len(payload):
        raise ValueError(f"fixed slices consume {offset}, payload has {len(payload)}")
    return raw_segments


def load_source_archive(path: Path, *, unpacker: Any | None = None) -> SourceArchive:
    if unpacker is None:
        unpacker = _load_unpacker()
    path = path.resolve()
    payload = _read_single_payload(path)
    header, decoded_raw = unpacker._parse_payload(payload)  # noqa: SLF001
    decoded = {str(name): bytes(data) for name, data in decoded_raw.items()}
    runtime_members = _member_summary(header)
    missing = sorted(set(SEGMENTS) - set(runtime_members))
    if missing:
        raise ValueError(f"source archive missing runtime members: {missing}")
    raw_segments = _slice_fixed_payload(payload, runtime_members)
    validate_seg_tile_actions_payload(
        decoded["seg_tile_actions.bin"],
        source_name="PR79 decoded seg_tile_actions.bin",
    )
    return SourceArchive(
        path=path,
        archive_bytes=path.stat().st_size,
        archive_sha256=_sha256_file(path),
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        payload_format=str(header.get("payload_format")),
        raw_segments=raw_segments,
        decoded=decoded,
    )


def _read_action_records(raw: bytes) -> list[tuple[int, int, int]]:
    if len(raw) % 4:
        raise ValueError(f"decoded actions must be raw4 records, got {len(raw)} bytes")
    return [
        (int.from_bytes(raw[offset:offset + 2], "little"), raw[offset + 2], raw[offset + 3])
        for offset in range(0, len(raw), 4)
    ]


def _uleb128(value: int) -> bytes:
    if value < 0:
        raise ValueError(f"cannot encode negative varint: {value}")
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def encode_s1_split_actions(decoded_actions: bytes) -> bytes:
    """Encode raw4 action records as S1 split grouped actions."""
    records = _read_action_records(decoded_actions)
    groups: list[tuple[int, list[tuple[int, int]]]] = []
    for tile_id in sorted({tile for _pair, tile, _action in records}):
        entries = sorted((pair, action) for pair, tile, action in records if tile == tile_id)
        groups.append((tile_id, entries))

    meta = bytearray(b"S1")
    meta += _uleb128(len(groups))
    deltas = bytearray()
    actions = bytearray()
    previous_tile = 0
    for tile_index, (tile_id, entries) in enumerate(groups):
        tile_delta = tile_id if tile_index == 0 else tile_id - previous_tile
        if tile_delta < 0:
            raise ValueError("S1 tile groups must be sorted")
        meta += _uleb128(tile_delta)
        meta += _uleb128(len(entries))
        previous_pair = 0
        for record_index, (pair_index, action_id) in enumerate(entries):
            delta = pair_index if record_index == 0 else pair_index - previous_pair
            if delta < 0:
                raise ValueError("S1 pair deltas require sorted pairs inside a tile")
            deltas += _uleb128(delta)
            actions.append(action_id)
            previous_pair = pair_index
        previous_tile = tile_id
    return bytes(meta + deltas + actions)


def _brotli_grid() -> list[tuple[int, int, int, int]]:
    params: list[tuple[int, int, int, int]] = []
    for quality in range(11, -1, -1):
        for mode in (0, 1, 2):
            for lgwin in range(10, 25):
                for lgblock in (0, 16, 17, 18, 19, 20, 21, 22, 23, 24):
                    if lgblock and lgblock > lgwin:
                        continue
                    params.append((quality, mode, lgwin, lgblock))
    return params


def best_brotli(raw: bytes, params: Iterable[tuple[int, int, int, int]] | None = None) -> BrotliChoice:
    best_data: bytes | None = None
    best_params: dict[str, int] | None = None
    for quality, mode, lgwin, lgblock in params or _brotli_grid():
        candidate = brotli.compress(
            raw,
            quality=quality,
            mode=mode,
            lgwin=lgwin,
            lgblock=lgblock,
        )
        if best_data is None or len(candidate) < len(best_data):
            best_data = candidate
            best_params = {
                "quality": quality,
                "mode": mode,
                "lgwin": lgwin,
                "lgblock": lgblock,
            }
    if best_data is None or best_params is None:
        raise ValueError("no Brotli candidate generated")
    if brotli.decompress(best_data) != raw:
        raise ValueError("selected Brotli stream failed round-trip")
    return BrotliChoice(data=best_data, params=best_params)


def build_fixed_s1_payload(source: SourceArchive, actions_br: bytes) -> bytes:
    return (
        source.raw_segments["masks.mkv"]
        + source.raw_segments["renderer.bin"]
        + actions_br
        + source.raw_segments["optimized_poses.qp1"]
    )


def build_p3_s1_payload(source: SourceArchive, actions_br: bytes) -> bytes:
    return (
        b"P3"
        + struct.pack(
            "<IHH",
            len(source.raw_segments["masks.mkv"]),
            len(source.raw_segments["renderer.bin"]),
            len(actions_br),
        )
        + source.raw_segments["masks.mkv"]
        + source.raw_segments["renderer.bin"]
        + actions_br
        + source.raw_segments["optimized_poses.qp1"]
    )


def _validate_payload(
    source: SourceArchive,
    payload: bytes,
    *,
    unpacker: Any,
) -> dict[str, Any]:
    header, decoded_raw = unpacker._parse_payload(payload)  # noqa: SLF001
    decoded = {str(name): bytes(data) for name, data in decoded_raw.items()}
    for name in NON_ACTION_SEGMENTS:
        if decoded.get(name) != source.decoded[name]:
            raise ValueError(f"non-action stream changed: {name}")
    if decoded.get("seg_tile_actions.bin") != source.decoded["seg_tile_actions.bin"]:
        raise ValueError("decoded action-record parity failed")
    return {
        "action_record_parity": True,
        "decoded_action_sha256": _sha256_bytes(decoded["seg_tile_actions.bin"]),
        "non_action_streams_preserved": True,
        "payload_format": str(header.get("payload_format")),
        "runtime_members": _member_summary(header),
        "runtime_parser": _repo_rel(UNPACKER_PATH),
        "seg_tile_actions_validation": validate_seg_tile_actions_payload(
            decoded["seg_tile_actions.bin"],
            source_name="candidate decoded seg_tile_actions.bin",
        ),
    }


def _build_one(
    *,
    source: SourceArchive,
    candidate_id: str,
    payload: bytes,
    actions_br: bytes,
    s1_raw: bytes,
    brotli_choice: BrotliChoice,
    output_dir: Path,
    unpacker: Any,
    force: bool,
) -> dict[str, Any]:
    candidate_dir = output_dir / candidate_id
    archive_path = candidate_dir / "archive.zip"
    manifest_path = candidate_dir / "manifest.json"
    if archive_path.exists() and not force:
        raise FileExistsError(f"{archive_path} exists; pass --force")
    _write_archive(archive_path, payload)
    if _read_single_payload(archive_path) != payload:
        raise ValueError(f"{candidate_id}: archive readback mismatch")
    validation = _validate_payload(source, payload, unpacker=unpacker)
    archive_sha = _sha256_file(archive_path)
    payload_sha = _sha256_bytes(payload)
    source_action_br = source.raw_segments["seg_tile_actions.bin"]
    noop_status = (
        "byte_noop"
        if payload == source.payload
        else "decoded_action_semantics_preserved_action_bytes_changed"
    )
    manifest = {
        "archive_byte_profile": profile_archive(archive_path),
        "candidate_id": candidate_id,
        "cuda_auth_eval_required": CUDA_AUTH_EVAL_REQUIRED,
        "dispatch_recommendation": {
            "dispatch_ready_now": False,
            "exact_t4_gate": CUDA_AUTH_EVAL_REQUIRED,
            "lane_claim_required": True,
            "recommended": True,
            "reason": "local runtime-closed lossless byte improvement; exact CUDA T4 gate is still required for score evidence",
        },
        "evidence_grade": "empirical_lossless_byte_screen",
        "no_op_detection": {
            "actions_br_sha_equal_to_source": _sha256_bytes(actions_br) == _sha256_bytes(source_action_br),
            "archive_sha_equal_to_source": archive_sha == source.archive_sha256,
            "decoded_action_sha_equal_to_source": (
                validation["decoded_action_sha256"]
                == _sha256_bytes(source.decoded["seg_tile_actions.bin"])
            ),
            "payload_sha_equal_to_source": payload_sha == source.payload_sha256,
            "status": noop_status,
        },
        "output_archive": {
            "bytes": archive_path.stat().st_size,
            "path": str(archive_path),
            "repo_relative_path": _repo_rel(archive_path),
            "sha256": archive_sha,
        },
        "payload": {
            "bytes": len(payload),
            "member": MEMBER_NAME,
            "sha256": payload_sha,
        },
        "remote_dispatch_performed": False,
        "runtime_parse_validation": validation,
        "schema": MANIFEST_SCHEMA,
        "score_claim": False,
        "source_archive": {
            "bytes": source.archive_bytes,
            "path": _repo_rel(source.path),
            "sha256": source.archive_sha256,
        },
        "stream_delta": {
            "actions_brotli_delta_bytes_vs_source": len(actions_br) - len(source_action_br),
            "actions_brotli_source_bytes": len(source_action_br),
            "actions_brotli_source_sha256": _sha256_bytes(source_action_br),
            "actions_brotli_s1_bytes": len(actions_br),
            "actions_brotli_s1_sha256": _sha256_bytes(actions_br),
            "archive_delta_bytes_vs_c102": archive_path.stat().st_size - C102_BYTES,
            "archive_delta_bytes_vs_pr79": archive_path.stat().st_size - source.archive_bytes,
            "payload_delta_bytes_vs_pr79": len(payload) - len(source.payload),
            "s1_raw_bytes": len(s1_raw),
            "s1_raw_sha256": _sha256_bytes(s1_raw),
            "source_decoded_action_bytes": len(source.decoded["seg_tile_actions.bin"]),
            "source_decoded_action_sha256": _sha256_bytes(source.decoded["seg_tile_actions.bin"]),
        },
        "stream_packing": {
            "action_codec": "S1_split_tile_delta_count_pair_delta_actions_brotli",
            "brotli_params": brotli_choice.params,
            "preserved_segments": list(NON_ACTION_SEGMENTS),
        },
        "tool": TOOL,
    }
    _write_json(manifest_path, manifest)
    return {
        "archive_bytes": manifest["output_archive"]["bytes"],
        "archive_path": manifest["output_archive"]["repo_relative_path"],
        "archive_sha256": archive_sha,
        "candidate_id": candidate_id,
        "delta_bytes_vs_c102": manifest["stream_delta"]["archive_delta_bytes_vs_c102"],
        "delta_bytes_vs_pr79": manifest["stream_delta"]["archive_delta_bytes_vs_pr79"],
        "dispatch_ready_now": False,
        "manifest_path": _repo_rel(manifest_path),
        "no_op_status": noop_status,
        "score_claim": False,
    }


def build_candidates(
    *,
    pr79_archive: Path = DEFAULT_PR79_ARCHIVE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    force: bool = False,
) -> dict[str, Any]:
    unpacker = _load_unpacker()
    source = load_source_archive(pr79_archive, unpacker=unpacker)
    if pr79_archive == DEFAULT_PR79_ARCHIVE and source.archive_sha256 != PR79_SHA256:
        raise ValueError(f"default PR79 archive SHA mismatch: {source.archive_sha256}")

    s1_raw = encode_s1_split_actions(source.decoded["seg_tile_actions.bin"])
    s1_choice = best_brotli(s1_raw)
    if brotli.decompress(s1_choice.data) != s1_raw:
        raise ValueError("S1 Brotli roundtrip failed")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        _build_one(
            source=source,
            candidate_id="pr79_s1_fixed_lossless_actions",
            payload=build_fixed_s1_payload(source, s1_choice.data),
            actions_br=s1_choice.data,
            s1_raw=s1_raw,
            brotli_choice=s1_choice,
            output_dir=output_dir,
            unpacker=unpacker,
            force=force,
        ),
        _build_one(
            source=source,
            candidate_id="pr79_s1_p3_lossless_actions",
            payload=build_p3_s1_payload(source, s1_choice.data),
            actions_br=s1_choice.data,
            s1_raw=s1_raw,
            brotli_choice=s1_choice,
            output_dir=output_dir,
            unpacker=unpacker,
            force=force,
        ),
    ]
    source_action_br = source.raw_segments["seg_tile_actions.bin"]
    matrix = {
        "byte_matrix": [
            {
                "archive_bytes": source.archive_bytes,
                "archive_sha256": source.archive_sha256,
                "candidate_id": "source_pr79_noop_control",
                "delta_bytes_vs_pr79": 0,
                "payload_bytes": len(source.payload),
                "seg_tile_actions_brotli_bytes": len(source_action_br),
                "seg_tile_actions_brotli_sha256": _sha256_bytes(source_action_br),
                "status": "source_reference",
            },
            *candidates,
        ],
        "c102_reference": {"archive_bytes": C102_BYTES, "archive_sha256": C102_SHA256},
        "cuda_auth_eval_required": CUDA_AUTH_EVAL_REQUIRED,
        "dispatch_decision": {
            "best_candidate_id": candidates[0]["candidate_id"],
            "exact_eval_justified": candidates[0]["delta_bytes_vs_pr79"] < 0,
            "no_remote_dispatch_performed": True,
            "required_before_dispatch": [
                "claim lane with tools/claim_lane_dispatch.py claim",
                "run exact T4-equivalent CUDA auth eval on the exact archive SHA/bytes",
                "record contest_auth_eval.json, runtime tree hash, and component gates",
            ],
        },
        "evidence_grade": "empirical_lossless_byte_screen",
        "remote_dispatch_performed": False,
        "schema": SCHEMA,
        "score_claim": False,
        "source_archive": {
            "bytes": source.archive_bytes,
            "path": _repo_rel(source.path),
            "sha256": source.archive_sha256,
        },
        "source_action_stream": {
            "brotli_bytes": len(source_action_br),
            "brotli_sha256": _sha256_bytes(source_action_br),
            "decoded_bytes": len(source.decoded["seg_tile_actions.bin"]),
            "decoded_sha256": _sha256_bytes(source.decoded["seg_tile_actions.bin"]),
            "record_count": len(source.decoded["seg_tile_actions.bin"]) // 4,
        },
        "s1_action_stream": {
            "brotli_bytes": len(s1_choice.data),
            "brotli_params": s1_choice.params,
            "brotli_sha256": _sha256_bytes(s1_choice.data),
            "raw_bytes": len(s1_raw),
            "raw_sha256": _sha256_bytes(s1_raw),
        },
        "tool": TOOL,
    }
    _write_json(output_dir / "candidate_matrix.json", matrix)
    return matrix


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr79-archive", type=Path, default=DEFAULT_PR79_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    matrix = build_candidates(
        pr79_archive=args.pr79_archive,
        output_dir=args.output_dir,
        force=bool(args.force),
    )
    print(json.dumps(matrix["byte_matrix"], indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
