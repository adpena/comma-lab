#!/usr/bin/env python3
"""Build deterministic fixed-slice segment-mix archive candidates.

This tool is build-only. It composes single-member public fixed-slice archives
by choosing the charged mask, renderer, and pose segments independently from
source archives that the robust-current unpacker already understands. The
output archives are not score evidence until exact CUDA auth eval runs on the
identical bytes.
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
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
SCHEMA = "fixedslice_segment_mix_candidates_v1"
TOOL = "experiments/build_fixedslice_segment_mix_candidates.py"
CUDA_AUTH_EVAL_REQUIRED = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
SEGMENT_NAMES = ("masks.mkv", "renderer.bin", "optimized_poses.bin")


@dataclass(frozen=True)
class SourceArchive:
    label: str
    archive_path: Path
    archive_bytes: int
    archive_sha256: str
    payload_bytes: bytes
    payload_sha256: str
    payload_format: str
    segments: dict[str, bytes]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _load_unpacker() -> Any:
    spec = importlib.util.spec_from_file_location("fixedslice_segment_mix_unpacker", UNPACKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load unpacker from {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _read_single_member_payload(path: Path, *, member_name: str = "p") -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != [member_name]:
            raise ValueError(f"{path} must contain single member {member_name!r}; got {names!r}")
        return zf.read(infos[0])


def _load_source(label: str, archive_path: Path, unpacker: Any) -> SourceArchive:
    archive_path = archive_path.resolve()
    payload = _read_single_member_payload(archive_path)
    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    payload_format = str(header.get("payload_format"))
    if payload_format != "public_pr67_qzs3_qp1_fixed_slices":
        raise ValueError(
            f"{label}: unsupported payload_format={payload_format!r}; "
            "expected public_pr67_qzs3_qp1_fixed_slices"
        )
    members = header.get("members")
    if not isinstance(members, list):
        raise ValueError(f"{label}: header.members missing")
    members_by_name: dict[str, Mapping[str, Any]] = {}
    for item in members:
        if not isinstance(item, Mapping):
            raise ValueError(f"{label}: malformed member record")
        name = str(item.get("name"))
        if name not in SEGMENT_NAMES:
            raise ValueError(f"{label}: unexpected segment {name!r}")
        if name in members_by_name:
            raise ValueError(f"{label}: duplicate segment metadata for {name!r}")
        members_by_name[name] = item
    missing = sorted(set(SEGMENT_NAMES) - set(members_by_name))
    if missing:
        raise ValueError(f"{label}: missing segments {missing}")

    # The public PR67-style fixed-slice container stores raw bytes in
    # mask/renderer/pose order even when the runtime metadata reports logical
    # members in another order.  Raw slicing must follow the wire contract.
    position = 0
    segments: dict[str, bytes] = {}
    for name in SEGMENT_NAMES:
        item = members_by_name[name]
        size = int(item.get("bytes"))
        raw = payload[position : position + size]
        position += size
        expected_raw_sha = str(item.get("sha256", ""))
        if expected_raw_sha and _sha256_bytes(raw) != expected_raw_sha:
            raise ValueError(
                f"{label}: raw SHA mismatch for {name}: "
                f"expected {expected_raw_sha}, got {_sha256_bytes(raw)}"
            )
        decoded_member = decoded.get(name)
        if decoded_member is None:
            raise ValueError(f"{label}: unpacker did not decode {name}")
        expected_decoded_sha = item.get("decoded_sha256")
        if expected_decoded_sha is not None and _sha256_bytes(decoded_member) != str(expected_decoded_sha):
            raise ValueError(f"{label}: decoded SHA mismatch for {name}")
        segments[name] = raw
    if position != len(payload):
        raise ValueError(f"{label}: segment sizes consume {position}, payload has {len(payload)}")
    return SourceArchive(
        label=label,
        archive_path=archive_path,
        archive_bytes=archive_path.stat().st_size,
        archive_sha256=_sha256_file(archive_path),
        payload_bytes=payload,
        payload_sha256=_sha256_bytes(payload),
        payload_format=payload_format,
        segments=segments,
    )


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_archive(path: Path, payload: bytes, *, member_name: str = "p") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info(member_name), payload)


def _parse_source_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("source must be LABEL=PATH")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("source label must be non-empty")
    return label, Path(path)


def _parse_mix_spec(raw: str) -> tuple[str, dict[str, str]]:
    if ":" not in raw:
        raise argparse.ArgumentTypeError(
            "mix must be ID:mask=LABEL,renderer=LABEL,pose=LABEL"
        )
    candidate_id, body = raw.split(":", 1)
    candidate_id = candidate_id.strip()
    if not candidate_id:
        raise argparse.ArgumentTypeError("mix candidate id must be non-empty")
    mapping: dict[str, str] = {}
    aliases = {"mask": "masks.mkv", "renderer": "renderer.bin", "pose": "optimized_poses.bin"}
    for part in body.split(","):
        if "=" not in part:
            raise argparse.ArgumentTypeError(f"malformed mix part: {part!r}")
        key, label = [value.strip() for value in part.split("=", 1)]
        if key not in aliases:
            raise argparse.ArgumentTypeError(f"unknown mix key {key!r}")
        if not label:
            raise argparse.ArgumentTypeError(f"empty source label for {key!r}")
        mapping[aliases[key]] = label
    missing = sorted(set(SEGMENT_NAMES) - set(mapping))
    if missing:
        raise argparse.ArgumentTypeError(f"mix {candidate_id!r} missing {missing}")
    return candidate_id, mapping


def _validate_candidate_payload(
    *,
    candidate_id: str,
    payload: bytes,
    mapping: Mapping[str, str],
    sources: Mapping[str, SourceArchive],
    unpacker: Any,
) -> dict[str, Any]:
    """Verify the emitted fixed-slice payload decodes under the runtime parser."""

    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    payload_format = str(header.get("payload_format"))
    if payload_format != "public_pr67_qzs3_qp1_fixed_slices":
        raise ValueError(
            f"{candidate_id}: runtime parser returned payload_format={payload_format!r}"
        )
    members = header.get("members")
    if not isinstance(members, list):
        raise ValueError(f"{candidate_id}: runtime parser returned no member table")
    members_by_name: dict[str, Mapping[str, Any]] = {}
    for item in members:
        if not isinstance(item, Mapping):
            raise ValueError(f"{candidate_id}: malformed runtime member metadata")
        name = str(item.get("name"))
        if name in members_by_name:
            raise ValueError(f"{candidate_id}: duplicate runtime member metadata for {name}")
        members_by_name[name] = item
    missing = sorted(set(SEGMENT_NAMES) - set(members_by_name))
    if missing:
        raise ValueError(f"{candidate_id}: runtime parser missed segments {missing}")

    validation_segments: dict[str, dict[str, Any]] = {}
    for name in SEGMENT_NAMES:
        label = mapping[name]
        expected_raw = sources[label].segments[name]
        item = members_by_name[name]
        runtime_bytes = int(item.get("bytes"))
        runtime_sha = str(item.get("sha256", ""))
        expected_sha = _sha256_bytes(expected_raw)
        if runtime_bytes != len(expected_raw):
            raise ValueError(
                f"{candidate_id}: runtime parse byte mismatch for {name}: "
                f"expected {len(expected_raw)}, got {runtime_bytes}"
            )
        if runtime_sha != expected_sha:
            raise ValueError(
                f"{candidate_id}: runtime parse raw SHA mismatch for {name}: "
                f"expected {expected_sha}, got {runtime_sha}"
            )
        decoded_member = decoded.get(name)
        if decoded_member is None:
            raise ValueError(f"{candidate_id}: runtime parser did not decode {name}")
        decoded_sha = _sha256_bytes(decoded_member)
        expected_decoded_sha = item.get("decoded_sha256")
        if expected_decoded_sha is not None and decoded_sha != str(expected_decoded_sha):
            raise ValueError(
                f"{candidate_id}: runtime parse decoded SHA mismatch for {name}: "
                f"expected {expected_decoded_sha}, got {decoded_sha}"
            )
        validation_segments[name] = {
            "decoded_bytes": len(decoded_member),
            "decoded_sha256": decoded_sha,
            "raw_bytes": len(expected_raw),
            "raw_sha256": expected_sha,
            "source_label": label,
        }
    return {
        "payload_format": payload_format,
        "runtime_parser": str(UNPACKER_PATH),
        "segments": validation_segments,
    }


def build_candidates(
    *,
    sources: Mapping[str, SourceArchive],
    mixes: list[tuple[str, dict[str, str]]],
    output_dir: Path,
    force: bool,
    unpacker: Any | None = None,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if unpacker is None:
        unpacker = _load_unpacker()
    source_summary = {
        label: {
            "archive_bytes": source.archive_bytes,
            "archive_path": str(source.archive_path),
            "archive_sha256": source.archive_sha256,
            "payload_format": source.payload_format,
            "payload_sha256": source.payload_sha256,
            "segments": {
                name: {
                    "bytes": len(raw),
                    "sha256": _sha256_bytes(raw),
                }
                for name, raw in source.segments.items()
            },
        }
        for label, source in sources.items()
    }
    candidates: list[dict[str, Any]] = []
    for candidate_id, mapping in mixes:
        unknown_labels = sorted(set(mapping.values()) - set(sources))
        if unknown_labels:
            raise ValueError(f"{candidate_id}: unknown source labels {unknown_labels}")
        candidate_dir = output_dir / candidate_id
        archive_path = candidate_dir / "archive.zip"
        manifest_path = candidate_dir / "build_manifest.json"
        if archive_path.exists() and not force:
            raise FileExistsError(f"{archive_path} exists; pass --force")
        payload = b"".join(sources[mapping[name]].segments[name] for name in SEGMENT_NAMES)
        runtime_parse_validation = _validate_candidate_payload(
            candidate_id=candidate_id,
            payload=payload,
            mapping=mapping,
            sources=sources,
            unpacker=unpacker,
        )
        _write_archive(archive_path, payload)
        archive_bytes = archive_path.stat().st_size
        archive_sha = _sha256_file(archive_path)
        manifest = {
            "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
            "candidate_id": candidate_id,
            "output_archive": {
                "bytes": archive_bytes,
                "path": str(archive_path),
                "sha256": archive_sha,
            },
            "payload": {
                "bytes": len(payload),
                "format": "public_pr67_qzs3_qp1_fixed_slices",
                "member": "p",
                "sha256": _sha256_bytes(payload),
            },
            "promotion_eligible": False,
            "runtime_parse_validation": runtime_parse_validation,
            "schema": "fixedslice_segment_mix_candidate_manifest_v1",
            "score_claim": False,
            "segment_sources": {
                name: {
                    "bytes": len(sources[label].segments[name]),
                    "source_archive_sha256": sources[label].archive_sha256,
                    "source_label": label,
                    "source_segment_sha256": _sha256_bytes(sources[label].segments[name]),
                }
                for name, label in mapping.items()
            },
            "sources": source_summary,
            "tool": TOOL,
        }
        _write_json(manifest_path, manifest)
        candidates.append(
            {
                "archive_bytes": archive_bytes,
                "archive_path": str(archive_path),
                "archive_sha256": archive_sha,
                "candidate_id": candidate_id,
                "manifest_path": str(manifest_path),
                "score_claim": False,
                "segment_sources": {
                    name: mapping[name]
                    for name in SEGMENT_NAMES
                },
            }
        )
    summary = {
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "candidates": candidates,
        "promotion_eligible": False,
        "schema": SCHEMA,
        "score_claim": False,
        "sources": source_summary,
        "tool": TOOL,
    }
    _write_json(output_dir / "fixedslice_segment_mix_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=_parse_source_spec, required=True)
    parser.add_argument("--mix", action="append", type=_parse_mix_spec, required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    unpacker = _load_unpacker()
    sources = {
        label: _load_source(label, path, unpacker)
        for label, path in args.source
    }
    summary = build_candidates(
        sources=sources,
        mixes=list(args.mix),
        output_dir=args.output_dir,
        force=bool(args.force),
        unpacker=unpacker,
    )
    print(
        json.dumps(
            {
                "best_by_bytes": min(
                    summary["candidates"],
                    key=lambda item: (item["archive_bytes"], item["candidate_id"]),
                ),
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
