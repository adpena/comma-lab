#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build local PR77 fixed-slice tile-action transplant candidates.

This is a deterministic byte-screening builder only. It transplants PR77's
charged 325-byte fixed-slice SegNet tile-action stream into PR75-family
fixed-slice archives that the robust-current payload parser can validate.
The outputs are not score evidence until exact CUDA auth eval runs on the
identical archive bytes after the dispatch-claim protocol.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.archive_byte_profile import profile_archive


UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
DEFAULT_PR77_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/top_submission_reverse_engineering_20260503_pr77/archive.zip"
)
DEFAULT_TARGETS = (
    (
        "pr75_minp",
        REPO_ROOT
        / "experiments/results/top_submission_reverse_engineering_20260503_pr75_minp/archive.zip",
    ),
    (
        "pr75_public",
        REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr75/archive.zip",
    ),
    (
        "c089_p6_frontier",
        REPO_ROOT
        / "experiments/results/lightning_batch/"
        "exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip",
    ),
    (
        "c089_raw_no_header_probe",
        REPO_ROOT
        / "experiments/results/pr75_lossless_micro_packer_worker_20260503/"
        "c089_raw_no_header_fixedslice_probe/archive.zip",
    ),
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/pr77_tile_action_transplant_stream_mix_20260503_worker"
)
TOOL = "experiments/build_pr77_tile_action_transplant_candidates.py"
SCHEMA = "pr77_tile_action_transplant_candidate_matrix_v1"
MANIFEST_SCHEMA = "pr77_tile_action_transplant_candidate_manifest_v1"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
MEMBER_NAME = "p"
PAYLOAD_FORMAT = "public_pr75_qzs3_qp1_segactions_fixed_slices"
SEGMENT_NAMES = ("masks.mkv", "renderer.bin", "seg_tile_actions.bin", "optimized_poses.qp1")
NON_ACTION_SEGMENTS = ("masks.mkv", "renderer.bin", "optimized_poses.qp1")
CUDA_AUTH_EVAL_REQUIRED = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda after lane claim"
)


@dataclass(frozen=True)
class SourceArchive:
    label: str
    path: Path
    archive_bytes: int
    archive_sha256: str
    payload: bytes
    payload_sha256: str
    payload_format: str
    raw_segments: dict[str, bytes]
    decoded_segments: dict[str, bytes]
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
    spec = importlib.util.spec_from_file_location("pr77_tile_action_transplant_unpacker", UNPACKER_PATH)
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


def _member_summary(header: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    members = header.get("members")
    if not isinstance(members, list):
        raise ValueError("payload parser returned no member table")
    for item in members:
        if not isinstance(item, Mapping):
            raise ValueError("payload parser returned malformed member metadata")
        name = str(item.get("name"))
        out[name] = {
            "bytes": int(item["bytes"]),
            "codec": str(item["codec"]),
            "decoded_bytes": int(item["decoded_bytes"]),
            "decoded_sha256": str(item["decoded_sha256"]),
            "sha256": str(item["sha256"]),
        }
    return out


def _decoded_summary(decoded: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {"bytes": len(data), "sha256": _sha256_bytes(data)}
        for name, data in sorted(decoded.items())
    }


def _slice_fixed_pr75_payload(
    *,
    label: str,
    payload: bytes,
    runtime_members: Mapping[str, Mapping[str, Any]],
) -> dict[str, bytes]:
    missing = sorted(set(SEGMENT_NAMES) - set(runtime_members))
    if missing:
        raise ValueError(f"{label}: missing fixed-slice members {missing}")
    offset = 0
    raw_segments: dict[str, bytes] = {}
    for name in SEGMENT_NAMES:
        size = int(runtime_members[name]["bytes"])
        if size <= 0:
            raise ValueError(f"{label}: non-positive raw segment size for {name}: {size}")
        raw = payload[offset : offset + size]
        offset += size
        expected_sha = str(runtime_members[name].get("sha256", ""))
        actual_sha = _sha256_bytes(raw)
        if expected_sha and actual_sha != expected_sha:
            raise ValueError(
                f"{label}: raw SHA mismatch for {name}: expected {expected_sha}, got {actual_sha}"
            )
        raw_segments[name] = raw
    if offset != len(payload):
        raise ValueError(f"{label}: fixed slices consume {offset}, payload has {len(payload)}")
    return raw_segments


def _load_source(label: str, path: Path, unpacker: Any) -> SourceArchive:
    path = path.resolve()
    payload = _read_single_payload_member(path)
    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    payload_format = str(header.get("payload_format"))
    if payload_format != PAYLOAD_FORMAT:
        raise ValueError(
            f"{label}: unsupported payload_format={payload_format!r}; expected {PAYLOAD_FORMAT!r}"
        )
    runtime_members = _member_summary(header)
    raw_segments = _slice_fixed_pr75_payload(
        label=label,
        payload=payload,
        runtime_members=runtime_members,
    )
    decoded_segments: dict[str, bytes] = {}
    for name in SEGMENT_NAMES:
        segment = decoded.get(name)
        if segment is None:
            raise ValueError(f"{label}: parser did not decode {name}")
        expected_decoded_sha = runtime_members[name].get("decoded_sha256")
        if expected_decoded_sha and _sha256_bytes(segment) != str(expected_decoded_sha):
            raise ValueError(f"{label}: decoded SHA mismatch for {name}")
        decoded_segments[name] = segment
    return SourceArchive(
        label=label,
        path=path,
        archive_bytes=path.stat().st_size,
        archive_sha256=_sha256_file(path),
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        payload_format=payload_format,
        raw_segments=raw_segments,
        decoded_segments=decoded_segments,
        runtime_members=runtime_members,
    )


def _load_source_or_skip(label: str, path: Path, unpacker: Any) -> tuple[SourceArchive | None, dict[str, Any] | None]:
    try:
        return _load_source(label, path, unpacker), None
    except Exception as exc:
        return None, {
            "label": label,
            "path": str(path),
            "reason": str(exc),
            "status": "skipped_not_runtime_compatible_fixedslice_pr75",
        }


def _action_record_stats(raw_actions: bytes) -> dict[str, Any]:
    if len(raw_actions) % 4 != 0:
        raise ValueError(f"PR77 action records must be 4-byte runtime records, got {len(raw_actions)}")
    records: list[tuple[int, int, int]] = []
    for offset in range(0, len(raw_actions), 4):
        pair_index = int.from_bytes(raw_actions[offset : offset + 2], "little")
        tile_id = raw_actions[offset + 2]
        action_id = raw_actions[offset + 3]
        if pair_index >= 10_000:
            raise ValueError(f"PR77 action pair out of bounds: {pair_index}")
        if tile_id >= 192:
            raise ValueError(f"PR77 action tile out of 384x512/32 grid bounds: {tile_id}")
        if action_id >= 108:
            raise ValueError(f"PR77 action id outside default PR75 dictionary: {action_id}")
        records.append((pair_index, tile_id, action_id))
    if not records:
        raise ValueError("PR77 action stream decoded no records")
    pair_tile_counts: dict[tuple[int, int], int] = {}
    for pair_index, tile_id, _action_id in records:
        key = (pair_index, tile_id)
        pair_tile_counts[key] = pair_tile_counts.get(key, 0) + 1
    duplicate_pair_tiles = [
        {"pair": pair, "tile": tile, "count": count}
        for (pair, tile), count in sorted(pair_tile_counts.items())
        if count > 1
    ]
    return {
        "action_max": max(action_id for _pair, _tile, action_id in records),
        "action_min": min(action_id for _pair, _tile, action_id in records),
        "duplicate_pair_tile_count": len(duplicate_pair_tiles),
        "duplicate_pair_tiles_first10": duplicate_pair_tiles[:10],
        "nondecreasing_pair_order": all(
            records[index][0] <= records[index + 1][0]
            for index in range(len(records) - 1)
        ),
        "pair_max": max(pair for pair, _tile, _action in records),
        "pair_min": min(pair for pair, _tile, _action in records),
        "record_count": len(records),
        "records_sha256": _sha256_bytes(raw_actions),
        "tile_max": max(tile for _pair, tile, _action in records),
        "tile_min": min(tile for _pair, tile, _action in records),
        "unique_actions": len({action for _pair, _tile, action in records}),
        "unique_pairs": len({pair for pair, _tile, _action in records}),
        "unique_tiles": len({tile for _pair, tile, _action in records}),
    }


def _source_summary(source: SourceArchive) -> dict[str, Any]:
    return {
        "archive_bytes": source.archive_bytes,
        "archive_path": str(source.path),
        "archive_sha256": source.archive_sha256,
        "decoded_segments": _decoded_summary(source.decoded_segments),
        "payload_bytes": len(source.payload),
        "payload_format": source.payload_format,
        "payload_sha256": source.payload_sha256,
        "raw_segments": {
            name: {
                "bytes": len(raw),
                "sha256": _sha256_bytes(raw),
            }
            for name, raw in sorted(source.raw_segments.items())
        },
    }


def _validate_candidate_payload(
    *,
    candidate_id: str,
    payload: bytes,
    target: SourceArchive,
    pr77: SourceArchive,
    unpacker: Any,
) -> dict[str, Any]:
    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    payload_format = str(header.get("payload_format"))
    if payload_format != PAYLOAD_FORMAT:
        raise ValueError(f"{candidate_id}: runtime parser returned {payload_format!r}")
    runtime_members = _member_summary(header)
    raw_segments = _slice_fixed_pr75_payload(
        label=candidate_id,
        payload=payload,
        runtime_members=runtime_members,
    )
    validation_segments: dict[str, dict[str, Any]] = {}
    for name in SEGMENT_NAMES:
        expected_raw = (
            pr77.raw_segments[name] if name == "seg_tile_actions.bin" else target.raw_segments[name]
        )
        expected_decoded = (
            pr77.decoded_segments[name]
            if name == "seg_tile_actions.bin"
            else target.decoded_segments[name]
        )
        if raw_segments[name] != expected_raw:
            raise ValueError(f"{candidate_id}: raw segment mismatch for {name}")
        decoded_member = decoded.get(name)
        if decoded_member != expected_decoded:
            raise ValueError(f"{candidate_id}: decoded segment mismatch for {name}")
        validation_segments[name] = {
            "decoded_bytes": len(decoded_member),
            "decoded_sha256": _sha256_bytes(decoded_member),
            "raw_bytes": len(raw_segments[name]),
            "raw_sha256": _sha256_bytes(raw_segments[name]),
            "source_label": "pr77" if name == "seg_tile_actions.bin" else target.label,
        }
    return {
        "payload_format": payload_format,
        "runtime_parser": str(UNPACKER_PATH),
        "segments": validation_segments,
    }


def _compatibility_summary(target: SourceArchive, pr77: SourceArchive) -> dict[str, Any]:
    non_action = {}
    mismatches: list[str] = []
    for name in NON_ACTION_SEGMENTS:
        raw_equal = target.raw_segments[name] == pr77.raw_segments[name]
        decoded_equal = target.decoded_segments[name] == pr77.decoded_segments[name]
        if not (raw_equal and decoded_equal):
            mismatches.append(name)
        non_action[name] = {
            "decoded_equal_to_pr77": decoded_equal,
            "decoded_sha256": _sha256_bytes(target.decoded_segments[name]),
            "pr77_decoded_sha256": _sha256_bytes(pr77.decoded_segments[name]),
            "pr77_raw_sha256": _sha256_bytes(pr77.raw_segments[name]),
            "raw_equal_to_pr77": raw_equal,
            "raw_sha256": _sha256_bytes(target.raw_segments[name]),
        }
    if mismatches:
        status = "runtime_parse_only_non_action_mismatch"
        required_preflight = (
            "Do not dispatch on byte closure alone. Parent must claim lane, run exact CUDA "
            "auth eval on identical bytes, and inspect component gates; renderer mismatches "
            "should also get pose/output parity review before promotion."
        )
    else:
        status = "pr77_non_action_streams_identical"
        required_preflight = (
            "Parent must claim lane before any exact CUDA auth eval; this candidate is still "
            "byte/provenance evidence only until CUDA eval."
        )
    return {
        "non_action_segments_vs_pr77": non_action,
        "required_preflight_before_dispatch": required_preflight,
        "semantic_status": status,
    }


def _parse_target_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("target must be LABEL=PATH")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("target label must be non-empty")
    return label, Path(path)


def _default_targets() -> list[tuple[str, Path]]:
    return [(label, path) for label, path in DEFAULT_TARGETS if path.exists()]


def build_candidates(
    *,
    pr77_archive: Path,
    targets: Iterable[tuple[str, Path]],
    output_dir: Path,
    force: bool = False,
    unpacker: Any | None = None,
) -> dict[str, Any]:
    if unpacker is None:
        unpacker = _load_unpacker()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pr77 = _load_source("pr77_qzs3_tile_delta_r147", pr77_archive, unpacker)
    pr77_action_encoded = pr77.raw_segments["seg_tile_actions.bin"]
    if len(pr77_action_encoded) != 325:
        raise ValueError(
            f"expected PR77 encoded action stream to be 325 bytes, got {len(pr77_action_encoded)}"
        )
    pr77_action_stats = _action_record_stats(pr77.decoded_segments["seg_tile_actions.bin"])
    source_pr77 = _source_summary(pr77)

    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for label, target_path in targets:
        target, skip = _load_source_or_skip(label, target_path, unpacker)
        if skip is not None:
            skipped.append(skip)
            continue
        assert target is not None
        candidate_id = f"pr77_actions_on_{target.label}"
        payload = b"".join(
            pr77.raw_segments[name] if name == "seg_tile_actions.bin" else target.raw_segments[name]
            for name in SEGMENT_NAMES
        )
        runtime_parse_validation = _validate_candidate_payload(
            candidate_id=candidate_id,
            payload=payload,
            target=target,
            pr77=pr77,
            unpacker=unpacker,
        )
        candidate_dir = output_dir / candidate_id
        archive_path = candidate_dir / "archive.zip"
        if archive_path.exists() and not force:
            raise FileExistsError(f"{archive_path} exists; pass --force")
        _write_archive(archive_path, payload)
        archive_profile = profile_archive(archive_path)
        archive_bytes = archive_path.stat().st_size
        archive_sha256 = _sha256_file(archive_path)
        compatibility = _compatibility_summary(target, pr77)
        raw_segment_sources = {
            name: {
                "bytes": len(pr77.raw_segments[name] if name == "seg_tile_actions.bin" else target.raw_segments[name]),
                "source_archive_sha256": (
                    pr77.archive_sha256 if name == "seg_tile_actions.bin" else target.archive_sha256
                ),
                "source_label": "pr77" if name == "seg_tile_actions.bin" else target.label,
                "source_segment_sha256": _sha256_bytes(
                    pr77.raw_segments[name] if name == "seg_tile_actions.bin" else target.raw_segments[name]
                ),
            }
            for name in SEGMENT_NAMES
        }
        manifest = {
            "archive_byte_profile": archive_profile,
            "candidate_id": candidate_id,
            "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
            "compatibility": compatibility,
            "delta_vs_pr77_archive": {
                "archive_bytes": archive_bytes - pr77.archive_bytes,
                "payload_bytes": len(payload) - len(pr77.payload),
            },
            "delta_vs_target_archive": {
                "archive_bytes": archive_bytes - target.archive_bytes,
                "payload_bytes": len(payload) - len(target.payload),
            },
            "evidence_grade": "empirical_byte_screen_only",
            "output_archive": {
                "bytes": archive_bytes,
                "path": str(archive_path),
                "sha256": archive_sha256,
            },
            "payload": {
                "bytes": len(payload),
                "format": PAYLOAD_FORMAT,
                "member": MEMBER_NAME,
                "sha256": _sha256_bytes(payload),
            },
            "promotion_eligible": False,
            "raw_segment_sources": raw_segment_sources,
            "runtime_parse_validation": runtime_parse_validation,
            "schema": MANIFEST_SCHEMA,
            "score_claim": False,
            "source_archives": {
                "pr77": source_pr77,
                "target": _source_summary(target),
            },
            "tool": TOOL,
        }
        manifest_path = candidate_dir / "manifest.json"
        _write_json(manifest_path, manifest)
        candidates.append(
            {
                "archive_bytes": archive_bytes,
                "archive_path": str(archive_path),
                "archive_sha256": archive_sha256,
                "candidate_id": candidate_id,
                "delta_bytes_vs_pr77": archive_bytes - pr77.archive_bytes,
                "delta_bytes_vs_target": archive_bytes - target.archive_bytes,
                "manifest_path": str(manifest_path),
                "payload_bytes": len(payload),
                "payload_sha256": _sha256_bytes(payload),
                "runtime_parse_status": "passed",
                "score_claim": False,
                "semantic_status": compatibility["semantic_status"],
                "target_label": target.label,
            }
        )

    summary = {
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "candidates": sorted(candidates, key=lambda row: (row["archive_bytes"], row["candidate_id"])),
        "evidence_grade": "empirical_byte_screen_only",
        "pr77_action_stream": {
            "decoded": pr77_action_stats,
            "encoded_bytes": len(pr77_action_encoded),
            "encoded_sha256": _sha256_bytes(pr77_action_encoded),
        },
        "promotion_eligible": False,
        "schema": SCHEMA,
        "score_claim": False,
        "skipped_targets": skipped,
        "source_archives": {
            "pr77": source_pr77,
        },
        "tool": TOOL,
    }
    _write_json(output_dir / "candidate_matrix.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr77-archive", type=Path, default=DEFAULT_PR77_ARCHIVE)
    parser.add_argument(
        "--target",
        action="append",
        type=_parse_target_spec,
        help="Target fixed-slice archive as LABEL=PATH. Defaults to known local PR75/C089 candidates.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    targets = list(args.target) if args.target else _default_targets()
    summary = build_candidates(
        pr77_archive=args.pr77_archive,
        targets=targets,
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
                "skipped_count": len(summary["skipped_targets"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
