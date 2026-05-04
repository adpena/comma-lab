#!/usr/bin/env python3
"""Build C-101 renderer x C091/C101 pose-waterfill stack archives.

This is a local archive builder only. It preserves the charged mask, action,
and pose streams from a pose-waterfill archive, replaces only the charged
renderer stream with a C-101 renderer-self-compression stream, and emits a
deterministic single-member ``p`` ZIP plus a manifest. It does not dispatch
remote work and does not make score claims.
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
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
UNPACKER_PATH = REPO_ROOT / "submissions/robust_current/unpack_renderer_payload.py"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
SCHEMA = "c101_renderer_pose_stack_candidate_v1"
TOOL = "experiments/build_c101_renderer_pose_stack_candidate.py"
POSE_SAFETY_SCHEMA = "renderer_transplant_pose_safety_preflight_v1"
CUDA_AUTH_EVAL_REQUIRED = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)
PAYLOAD_MEMBER = "p"
P3_HEADER_STRUCT = "<IHH"
REQUIRED_DECODED_MEMBERS = (
    "masks.mkv",
    "renderer.bin",
    "seg_tile_actions.bin",
    "optimized_poses.bin",
)


class StackBuildError(ValueError):
    """Raised when a source archive is not safe to stack."""


@dataclass(frozen=True)
class P3Slices:
    mask_br: bytes
    renderer_br: bytes
    actions_br: bytes
    pose_br: bytes
    wire_magic: bytes = b"P3"
    action_record_count: int | None = None


@dataclass(frozen=True)
class SourceArchive:
    label: str
    path: Path
    archive_bytes: int
    archive_sha256: str
    payload: bytes
    payload_sha256: str
    slices: P3Slices
    decoded: dict[str, bytes]
    payload_format: str


@dataclass(frozen=True)
class ParsedPublicFloorPayload:
    path: Path
    archive_bytes: int
    archive_sha256: str
    payload_bytes: int
    payload_sha256: str
    format: str
    mask: bytes
    renderer: bytes
    actions: bytes
    pose: bytes


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _safe_member_name(name: str) -> str:
    path = Path(name)
    if not name or name.startswith("/") or ".." in path.parts or len(path.parts) != 1:
        raise StackBuildError(f"unsafe zip member path: {name!r}")
    return name


def _load_unpacker() -> Any:
    spec = importlib.util.spec_from_file_location("c101_stack_unpacker", UNPACKER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load unpacker from {UNPACKER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_single_payload(path: Path, *, expected_sha256: str | None = None) -> bytes:
    path = path.resolve()
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [_safe_member_name(info.filename) for info in infos]
        if names != [PAYLOAD_MEMBER]:
            raise StackBuildError(f"{path} must contain exactly one member 'p'; got {names!r}")
        payload = zf.read(infos[0])
    if expected_sha256 is not None:
        actual = _sha256_bytes(payload)
        if actual != expected_sha256:
            raise StackBuildError(
                f"{path} payload SHA mismatch: expected {expected_sha256}, got {actual}"
            )
    return payload


def parse_p3_slices(payload: bytes) -> P3Slices:
    if payload.startswith(b"P3"):
        header_struct = P3_HEADER_STRUCT
        header_size = 2 + struct.calcsize(header_struct)
        if len(payload) <= header_size:
            raise StackBuildError("P3 payload is too short")
        mask_len, renderer_len, actions_len = struct.unpack_from(header_struct, payload, 2)
        record_count = None
        wire_magic = b"P3"
    elif payload.startswith(b"P6"):
        header_struct = "<IHHH"
        header_size = 2 + struct.calcsize(header_struct)
        if len(payload) <= header_size:
            raise StackBuildError("P6 payload is too short")
        mask_len, renderer_len, actions_len, record_count = struct.unpack_from(
            header_struct,
            payload,
            2,
        )
        if record_count <= 0:
            raise StackBuildError("P6 payload has no action records")
        wire_magic = b"P6"
    else:
        raise StackBuildError(f"expected P3/P6 payload, got prefix={payload[:4]!r}")
    if len(payload) <= header_size:
        raise StackBuildError(f"{wire_magic.decode('ascii')} payload is too short")
    if min(mask_len, renderer_len, actions_len) <= 0:
        raise StackBuildError(
            f"{wire_magic.decode('ascii')} payload contains an empty required stream"
        )
    cursor = header_size
    mask_end = cursor + int(mask_len)
    renderer_end = mask_end + int(renderer_len)
    actions_end = renderer_end + int(actions_len)
    if actions_end >= len(payload):
        raise StackBuildError("P3 stream lengths leave no pose stream")
    if max(renderer_len, actions_len, len(payload) - actions_end) > 0xFFFF:
        raise StackBuildError("P3 u16 stream length contract exceeded")
    return P3Slices(
        mask_br=payload[cursor:mask_end],
        renderer_br=payload[mask_end:renderer_end],
        actions_br=payload[renderer_end:actions_end],
        pose_br=payload[actions_end:],
        wire_magic=wire_magic,
        action_record_count=record_count,
    )


def parse_public_floor_payload(path: Path) -> ParsedPublicFloorPayload:
    """Parse a raw P3 archive without decoding slices.

    This compatibility helper is intentionally byte-level. The contest-grade
    builder path uses ``load_source`` so the runtime unpacker validates Brotli,
    QZS3, QP1, and action contracts before an archive is emitted.
    """

    path = path.resolve()
    payload = read_single_payload(path)
    slices = parse_p3_slices(payload)
    return ParsedPublicFloorPayload(
        path=path,
        archive_bytes=path.stat().st_size,
        archive_sha256=_sha256_file(path),
        payload_bytes=len(payload),
        payload_sha256=_sha256_bytes(payload),
        format="P3",
        mask=slices.mask_br,
        renderer=slices.renderer_br,
        actions=slices.actions_br,
        pose=slices.pose_br,
    )


def build_payload(
    *,
    pose_source: ParsedPublicFloorPayload,
    renderer_source: ParsedPublicFloorPayload,
) -> bytes:
    """Build a raw P3 payload from parsed byte-level sources."""

    if pose_source.mask != renderer_source.mask:
        raise StackBuildError("mask slices differ between pose and renderer sources")
    if pose_source.actions != renderer_source.actions:
        raise StackBuildError("action slices differ between pose and renderer sources")
    return build_p3_payload(
        P3Slices(
            mask_br=pose_source.mask,
            renderer_br=renderer_source.renderer,
            actions_br=pose_source.actions,
            pose_br=pose_source.pose,
        )
    )


def build_p3_payload(slices: P3Slices) -> bytes:
    if max(len(slices.renderer_br), len(slices.actions_br), len(slices.pose_br)) > 0xFFFF:
        raise StackBuildError("P3/P6 u16 stream length contract exceeded")
    if min(len(slices.mask_br), len(slices.renderer_br), len(slices.actions_br), len(slices.pose_br)) <= 0:
        raise StackBuildError("cannot build P3/P6 payload with an empty stream")
    if slices.wire_magic == b"P3":
        header = b"P3" + struct.pack(
            P3_HEADER_STRUCT,
            len(slices.mask_br),
            len(slices.renderer_br),
            len(slices.actions_br),
        )
    elif slices.wire_magic == b"P6":
        if slices.action_record_count is None or slices.action_record_count <= 0:
            raise StackBuildError("P6 payload requires a positive action_record_count")
        header = b"P6" + struct.pack(
            "<IHHH",
            len(slices.mask_br),
            len(slices.renderer_br),
            len(slices.actions_br),
            int(slices.action_record_count),
        )
    else:
        raise StackBuildError(f"unsupported stack payload magic: {slices.wire_magic!r}")
    return header + slices.mask_br + slices.renderer_br + slices.actions_br + slices.pose_br


def _decode_payload(payload: bytes, unpacker: Any) -> tuple[str, dict[str, bytes]]:
    header, decoded = unpacker._parse_payload(payload)  # noqa: SLF001
    payload_format = str(header.get("payload_format"))
    allowed_formats = {
        "public_pr75_qzs3_qp1_segactions_p3",
        "public_pr75_qzs3_qp1_segactions_p6_delta_varint",
    }
    if payload_format not in allowed_formats:
        raise StackBuildError(f"unsupported runtime payload_format={payload_format!r}")
    out: dict[str, bytes] = {str(name): bytes(blob) for name, blob in decoded.items()}
    if "optimized_poses.bin" not in out and "optimized_poses.qp1" in out:
        out["optimized_poses.bin"] = out["optimized_poses.qp1"]
    missing = [name for name in REQUIRED_DECODED_MEMBERS if name not in out]
    if missing:
        raise StackBuildError(f"runtime unpacker missed required members: {missing}")
    if not out["renderer.bin"].startswith(b"QZS3"):
        raise StackBuildError("decoded renderer.bin is not QZS3")
    if not out["optimized_poses.bin"].startswith(b"QP1"):
        raise StackBuildError("decoded optimized_poses.bin is not QP1")
    return payload_format, out


def load_source(
    label: str,
    path: Path,
    *,
    unpacker: Any,
    expected_archive_sha256: str | None = None,
    expected_payload_sha256: str | None = None,
) -> SourceArchive:
    path = path.resolve()
    if expected_archive_sha256 is not None:
        actual_archive_sha = _sha256_file(path)
        if actual_archive_sha != expected_archive_sha256:
            raise StackBuildError(
                f"{label} archive SHA mismatch: expected {expected_archive_sha256}, "
                f"got {actual_archive_sha}"
            )
    payload = read_single_payload(path, expected_sha256=expected_payload_sha256)
    slices = parse_p3_slices(payload)
    payload_format, decoded = _decode_payload(payload, unpacker)
    return SourceArchive(
        label=label,
        path=path,
        archive_bytes=path.stat().st_size,
        archive_sha256=_sha256_file(path),
        payload=payload,
        payload_sha256=_sha256_bytes(payload),
        slices=slices,
        decoded=decoded,
        payload_format=payload_format,
    )


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(_safe_member_name(name), FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def write_single_member_archive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(_zip_info(PAYLOAD_MEMBER), payload)


def _stream_summary(encoded: bytes, decoded: bytes) -> dict[str, Any]:
    return {
        "encoded_bytes": len(encoded),
        "encoded_sha256": _sha256_bytes(encoded),
        "decoded_bytes": len(decoded),
        "decoded_sha256": _sha256_bytes(decoded),
        "decoded_magic_hex": decoded[:12].hex(),
    }


def _source_summary(source: SourceArchive) -> dict[str, Any]:
    return {
        "label": source.label,
        "path": str(source.path),
        "repo_relative_path": _repo_rel(source.path),
        "archive_bytes": source.archive_bytes,
        "archive_sha256": source.archive_sha256,
        "payload_bytes": len(source.payload),
        "payload_sha256": source.payload_sha256,
        "payload_format": source.payload_format,
        "streams": {
            "masks.mkv": _stream_summary(source.slices.mask_br, source.decoded["masks.mkv"]),
            "renderer.bin": _stream_summary(source.slices.renderer_br, source.decoded["renderer.bin"]),
            "seg_tile_actions.bin": _stream_summary(
                source.slices.actions_br,
                source.decoded["seg_tile_actions.bin"],
            ),
            "optimized_poses.bin": _stream_summary(
                source.slices.pose_br,
                source.decoded["optimized_poses.bin"],
            ),
        },
    }


def _read_pose_safety_reports(paths: Sequence[Path]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in paths:
        resolved = path.resolve()
        payload = json.loads(resolved.read_text())
        if not isinstance(payload, dict):
            raise StackBuildError(f"pose-safety report is not a JSON object: {resolved}")
        payload = dict(payload)
        payload["_pose_safety_json_path"] = _repo_rel(resolved)
        reports.append(payload)
    return reports


def _pose_safety_dispatch_gate(
    *,
    source_archive_sha256: str,
    candidate_archive_sha256: str,
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matching: list[Mapping[str, Any]] = []
    for report in reports:
        source = report.get("source_archive") or {}
        candidate = report.get("candidate_archive") or {}
        if (
            isinstance(source, Mapping)
            and isinstance(candidate, Mapping)
            and source.get("sha256") == source_archive_sha256
            and candidate.get("sha256") == candidate_archive_sha256
        ):
            matching.append(report)
    if not matching:
        return {
            "required": True,
            "status": "missing_pose_safety_report",
            "safe_for_exact_eval_dispatch": False,
            "matching_report_path": None,
            "blockers": [
                "missing renderer transplant pose-safety preflight for exact source/candidate archive SHA pair"
            ],
            "required_tool": "experiments/preflight_renderer_transplant_pose_safety.py",
        }
    report = sorted(matching, key=lambda item: str(item.get("_pose_safety_json_path") or ""))[-1]
    blockers: list[str] = []
    if report.get("schema") != POSE_SAFETY_SCHEMA:
        blockers.append("pose-safety report has unexpected schema")
    if report.get("score_claim") is not False:
        blockers.append("pose-safety report must be no-score evidence")
    if report.get("promotion_eligible") is not False:
        blockers.append("pose-safety report must not claim promotion eligibility")
    if report.get("remote_gpu_dispatch_performed") is not False:
        blockers.append("pose-safety report must be local-only")
    if report.get("safe_for_exact_eval_dispatch") is not True:
        blockers.extend(report.get("fail_closed_reasons") or ["pose-safety report failed closed"])
    safe = not blockers
    return {
        "required": True,
        "status": "pass" if safe else "failed",
        "safe_for_exact_eval_dispatch": safe,
        "matching_report_path": report.get("_pose_safety_json_path"),
        "failure_class": report.get("failure_class"),
        "fail_closed_reasons": report.get("fail_closed_reasons") or [],
        "blockers": sorted(set(str(item) for item in blockers if item)),
        "required_tool": "experiments/preflight_renderer_transplant_pose_safety.py",
    }


def build_stack_candidate(
    *,
    renderer_source: SourceArchive,
    pose_source: SourceArchive,
    output_dir: Path,
    candidate_id: str,
    unpacker: Any,
    pose_safety_reports: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    candidate_slices = P3Slices(
        mask_br=pose_source.slices.mask_br,
        renderer_br=renderer_source.slices.renderer_br,
        actions_br=pose_source.slices.actions_br,
        pose_br=pose_source.slices.pose_br,
        wire_magic=pose_source.slices.wire_magic,
        action_record_count=pose_source.slices.action_record_count,
    )
    payload = build_p3_payload(candidate_slices)
    payload_format, decoded = _decode_payload(payload, unpacker)

    expected_decoded = {
        "masks.mkv": pose_source.decoded["masks.mkv"],
        "renderer.bin": renderer_source.decoded["renderer.bin"],
        "seg_tile_actions.bin": pose_source.decoded["seg_tile_actions.bin"],
        "optimized_poses.bin": pose_source.decoded["optimized_poses.bin"],
    }
    stream_validation: dict[str, dict[str, Any]] = {}
    for name, expected in expected_decoded.items():
        actual = decoded[name]
        stream_validation[name] = {
            "matches_expected_source": actual == expected,
            "actual_decoded_sha256": _sha256_bytes(actual),
            "expected_decoded_sha256": _sha256_bytes(expected),
        }
        if actual != expected:
            raise StackBuildError(f"{candidate_id}: decoded {name} does not match selected source")

    candidate_dir = output_dir / candidate_id
    archive_path = candidate_dir / "archive.zip"
    write_single_member_archive(archive_path, payload)
    roundtrip_payload = read_single_payload(archive_path)
    if roundtrip_payload != payload:
        raise StackBuildError(f"{candidate_id}: archive readback payload mismatch")
    archive_sha = _sha256_file(archive_path)
    dispatch_gate = _pose_safety_dispatch_gate(
        source_archive_sha256=pose_source.archive_sha256,
        candidate_archive_sha256=archive_sha,
        reports=pose_safety_reports,
    )

    changed_vs_pose = {
        "masks.mkv": False,
        "renderer.bin": renderer_source.slices.renderer_br != pose_source.slices.renderer_br,
        "seg_tile_actions.bin": False,
        "optimized_poses.bin": False,
    }
    changed_vs_renderer = {
        "masks.mkv": pose_source.slices.mask_br != renderer_source.slices.mask_br,
        "renderer.bin": False,
        "seg_tile_actions.bin": pose_source.slices.actions_br != renderer_source.slices.actions_br,
        "optimized_poses.bin": pose_source.slices.pose_br != renderer_source.slices.pose_br,
    }
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "tool": TOOL,
        "candidate_id": candidate_id,
        "score_claim": False,
        "remote_dispatch_performed": False,
        "exact_eval_dispatch_gate": dispatch_gate,
        "cuda_auth_eval_required": CUDA_AUTH_EVAL_REQUIRED,
        "sources": {
            "renderer": _source_summary(renderer_source),
            "pose": _source_summary(pose_source),
        },
        "output_archive": {
            "path": str(archive_path.resolve()),
            "repo_relative_path": _repo_rel(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": archive_sha,
            "zip_members": [PAYLOAD_MEMBER],
            "payload_member": PAYLOAD_MEMBER,
            "payload_bytes": len(payload),
            "payload_sha256": _sha256_bytes(payload),
            "payload_format": payload_format,
            "deterministic_zip_timestamp": FIXED_ZIP_TIMESTAMP,
        },
        "encoded_streams": {
            "masks.mkv": _stream_summary(candidate_slices.mask_br, decoded["masks.mkv"]),
            "renderer.bin": _stream_summary(candidate_slices.renderer_br, decoded["renderer.bin"]),
            "seg_tile_actions.bin": _stream_summary(
                candidate_slices.actions_br,
                decoded["seg_tile_actions.bin"],
            ),
            "optimized_poses.bin": _stream_summary(
                candidate_slices.pose_br,
                decoded["optimized_poses.bin"],
            ),
        },
        "stream_selection": {
            "masks.mkv": pose_source.label,
            "renderer.bin": renderer_source.label,
            "seg_tile_actions.bin": pose_source.label,
            "optimized_poses.bin": pose_source.label,
        },
        "changed_streams_vs_pose_source": changed_vs_pose,
        "changed_streams_vs_renderer_source": changed_vs_renderer,
        "validation": {
            "single_member_zip": True,
            "no_sidecars_in_archive": True,
            "runtime_unpack_verified": True,
            "payload_format_is_p3": payload_format == "public_pr75_qzs3_qp1_segactions_p3",
            "payload_format_is_p6": payload_format
            == "public_pr75_qzs3_qp1_segactions_p6_delta_varint",
            "stream_validation": stream_validation,
            "mask_decoded_matches_pose_source": stream_validation["masks.mkv"][
                "matches_expected_source"
            ],
            "actions_decoded_matches_pose_source": stream_validation["seg_tile_actions.bin"][
                "matches_expected_source"
            ],
            "pose_decoded_matches_pose_source": stream_validation["optimized_poses.bin"][
                "matches_expected_source"
            ],
            "renderer_decoded_matches_renderer_source": stream_validation["renderer.bin"][
                "matches_expected_source"
            ],
            "pose_safety_preflight_required_before_dispatch": True,
            "safe_for_exact_eval_dispatch": dispatch_gate["safe_for_exact_eval_dispatch"],
        },
    }
    manifest_path = candidate_dir / "manifest.json"
    manifest_path.write_bytes(_json_bytes(manifest))
    return manifest


def _parse_source_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("pose source must be LABEL=PATH")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("source label must be non-empty")
    return label, Path(path)


def build_many(
    *,
    renderer_archive: Path,
    pose_archives: Sequence[tuple[str, Path]],
    output_dir: Path,
    renderer_label: str = "c101_renderer",
    expected_renderer_archive_sha256: str | None = None,
    pose_safety_json: Sequence[Path] = (),
) -> dict[str, Any]:
    unpacker = _load_unpacker()
    pose_safety_reports = _read_pose_safety_reports(pose_safety_json)
    renderer = load_source(
        renderer_label,
        renderer_archive,
        unpacker=unpacker,
        expected_archive_sha256=expected_renderer_archive_sha256,
    )
    manifests: list[dict[str, Any]] = []
    for pose_label, pose_archive in pose_archives:
        pose = load_source(pose_label, pose_archive, unpacker=unpacker)
        candidate_id = f"{renderer_label}_x_{pose_label}"
        manifests.append(
            build_stack_candidate(
                renderer_source=renderer,
                pose_source=pose,
                output_dir=output_dir,
                candidate_id=candidate_id,
                unpacker=unpacker,
                pose_safety_reports=pose_safety_reports,
            )
        )
    summary = {
        "schema": f"{SCHEMA}_summary",
        "tool": TOOL,
        "score_claim": False,
        "remote_dispatch_performed": False,
        "pose_safety_preflight_required_before_dispatch": True,
        "pose_safety_reports": [_repo_rel(path) for path in pose_safety_json],
        "candidate_count": len(manifests),
        "candidates": [
            {
                "candidate_id": item["candidate_id"],
                "archive": item["output_archive"],
                "exact_eval_dispatch_gate": item["exact_eval_dispatch_gate"],
                "changed_streams_vs_pose_source": item["changed_streams_vs_pose_source"],
                "changed_streams_vs_renderer_source": item["changed_streams_vs_renderer_source"],
            }
            for item in manifests
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_bytes(_json_bytes(summary))
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer-archive", type=Path, required=True)
    parser.add_argument(
        "--renderer-label",
        default="c101_renderer",
        help="Label used in manifests and candidate ids.",
    )
    parser.add_argument(
        "--expected-renderer-archive-sha256",
        help="Optional fail-closed SHA-256 for the renderer source archive.",
    )
    parser.add_argument(
        "--pose-archive",
        action="append",
        type=_parse_source_arg,
        required=True,
        help="Pose source as LABEL=PATH. May be repeated.",
    )
    parser.add_argument(
        "--pose-safety-json",
        action="append",
        type=Path,
        default=[],
        help=(
            "Renderer transplant pose-safety preflight JSON for an exact source/candidate "
            "archive SHA pair. Without a matching pass report, emitted manifests are "
            "fail-closed for exact-eval dispatch."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_many(
        renderer_archive=args.renderer_archive,
        pose_archives=args.pose_archive,
        output_dir=args.output_dir,
        renderer_label=args.renderer_label,
        expected_renderer_archive_sha256=args.expected_renderer_archive_sha256,
        pose_safety_json=args.pose_safety_json,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
