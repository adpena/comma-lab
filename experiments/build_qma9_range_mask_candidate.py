#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build deterministic QMA9 range-mask transfer candidates or manifests.

This is a local construction tool only. It can encode a raw semantic mask with
this repo's pure-Python QMA9 codec and combine it with supplied or template
PR81-style model/pose/router streams. It does not modify inflate runtime files,
does not run the scorer, and does not dispatch GPU work.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.qma9_range_mask_contract import (
    QMA9ContractError,
    encode_qma9_mask,
    parse_qma9_header,
    read_single_member_zip,
    sha256_bytes,
    sha256_file,
    split_qma9_pr81_payload,
    write_stored_single_member_zip,
)


TOOL = "experiments/build_qma9_range_mask_candidate.py"
SCHEMA = "qma9_range_mask_transfer_candidate_manifest_v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments/results/qma9_range_mask_transfer_20260503_codex"
DEFAULT_TEMPLATE_PR81_ARCHIVE = (
    REPO_ROOT / "experiments/results/public_pr81_qzs3_range_mask_intake_20260503_codex/archive.zip"
)

PR81_RANGE_MASK_BYTES = 159_011
PR81_MODEL_BYTES = 55_725
PR81_POSE_BYTES = 899
PR81_ROUTER_BYTES = 225
FRONTIER_T4_A_PLUS_PLUS_SCORE = 0.31453355357318635  # [external: PR-79 S2 contest-CUDA T4 A++ frontier]
ORIGINAL_VIDEO_BYTES = 37_545_489


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


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


def _read_optional(path: Path | None) -> bytes | None:
    if path is None:
        return None
    return path.read_bytes()


def _load_template_segments(path: Path | None) -> dict[str, bytes]:
    if path is None:
        return {}
    payload, _custody = read_single_member_zip(path)
    split = split_qma9_pr81_payload(
        payload,
        range_mask_bytes=PR81_RANGE_MASK_BYTES,
        model_bytes=PR81_MODEL_BYTES,
        pose_bytes=PR81_POSE_BYTES,
        router_bytes=PR81_ROUTER_BYTES,
    )
    return {
        "range_mask": split.range_mask,
        "model": split.model,
        "pose": split.pose,
        "router": split.router,
    }


def _build_qma9_payload(
    *,
    raw_mask_path: Path | None,
    qma9_payload_path: Path | None,
    frame_count: int,
    width: int,
    height: int,
    template_segments: dict[str, bytes],
) -> tuple[bytes, dict[str, Any]]:
    if raw_mask_path is not None and qma9_payload_path is not None:
        raise QMA9ContractError("choose either --raw-mask or --qma9-payload, not both")
    if raw_mask_path is not None:
        raw = raw_mask_path.read_bytes()
        payload = encode_qma9_mask(raw, frame_count=frame_count, width=width, height=height)
        return payload, {
            "source": _repo_rel(raw_mask_path),
            "source_kind": "raw_uint8_class_mask_storage_order",
            "raw_bytes": len(raw),
            "raw_sha256": sha256_bytes(raw),
            "encoded_by": "pure_python_qma9_codec",
        }
    if qma9_payload_path is not None:
        payload = qma9_payload_path.read_bytes()
        return payload, {
            "source": _repo_rel(qma9_payload_path),
            "source_kind": "prebuilt_qma9_payload",
            "encoded_by": "external_or_previous_qma9_payload",
        }
    if "range_mask" in template_segments:
        return template_segments["range_mask"], {
            "source": "template_pr81_archive",
            "source_kind": "template_qma9_payload_reuse_for_profile_or_baseline",
            "encoded_by": "template_bytes_not_reencoded",
        }
    raise QMA9ContractError("provide --raw-mask, --qma9-payload, or --template-pr81-archive")


def build_candidate(
    *,
    output_dir: Path,
    candidate_id: str,
    raw_mask_path: Path | None = None,
    qma9_payload_path: Path | None = None,
    model_payload_path: Path | None = None,
    pose_payload_path: Path | None = None,
    router_payload_path: Path | None = None,
    template_pr81_archive: Path | None = None,
    frame_count: int = 600,
    width: int = 512,
    height: int = 384,
    write_archive: bool = True,
) -> dict[str, Any]:
    template_segments = _load_template_segments(template_pr81_archive)
    qma9_payload, qma9_source = _build_qma9_payload(
        raw_mask_path=raw_mask_path,
        qma9_payload_path=qma9_payload_path,
        frame_count=frame_count,
        width=width,
        height=height,
        template_segments=template_segments,
    )
    qma9 = parse_qma9_header(qma9_payload)
    model = _read_optional(model_payload_path) or template_segments.get("model")
    pose = _read_optional(pose_payload_path) or template_segments.get("pose")
    router = _read_optional(router_payload_path) or template_segments.get("router")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_dir / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    qma9_path = candidate_dir / "range_mask.qma9"
    qma9_path.write_bytes(qma9_payload)

    segments: list[dict[str, Any]] = [
        {
            "name": "range_mask.qma9",
            "bytes": len(qma9_payload),
            "sha256": sha256_bytes(qma9_payload),
            "codec": "qma9_adaptive9_binary_range_mask",
            "source": qma9_source,
        }
    ]
    for name, data, codec, source_path in [
        ("split_model_reordered.br_bundle", model, "brotli_reordered_qzs3_model_bundle", model_payload_path),
        ("optimized_poses.qp1.br", pose, "brotli_qp1_pose_stream", pose_payload_path),
        ("router_actions.3bit", router, "packed_3bit_pair_router_actions", router_payload_path),
    ]:
        if data is None:
            segments.append(
                {
                    "name": name,
                    "available": False,
                    "codec": codec,
                    "source": None,
                    "reason": "missing_stream_payload",
                }
            )
        else:
            segments.append(
                {
                    "name": name,
                    "available": True,
                    "bytes": len(data),
                    "sha256": sha256_bytes(data),
                    "codec": codec,
                    "source": _repo_rel(source_path) if source_path is not None else "template_pr81_archive",
                }
            )

    archive_info: dict[str, Any] | None = None
    archive_path = candidate_dir / "archive.zip"
    all_streams = model is not None and pose is not None and router is not None
    if write_archive and all_streams:
        payload = qma9_payload + model + pose + router
        write_stored_single_member_zip(archive_path, payload)
        archive_info = {
            "path": _repo_rel(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": sha256_file(archive_path),
            "payload_bytes": len(payload),
            "payload_sha256": sha256_bytes(payload),
            "zip_member": "p",
            "zip_storage": "stored",
        }

    manifest = {
        "schema": SCHEMA,
        "tool": TOOL,
        "candidate_id": candidate_id,
        "evidence_grade": "empirical/planning_only",
        "score_claim": False,
        "dispatch_performed": False,
        "frontier_target_score": FRONTIER_T4_A_PLUS_PLUS_SCORE,
        "archive": archive_info,
        "qma9_header": asdict(qma9),
        "segments": segments,
        "determinism": {
            "zip_timestamp": "1980-01-01T00:00:00",
            "zip_compression": "stored",
            "json_sort_keys": True,
        },
        "dispatch_readiness": {
            "safe_for_remote_dispatch": False,
            "reason": (
                "Current task forbids remote dispatch and existing robust runtime files were not edited. "
                "Future exact eval requires a lane claim, runtime integration, payload closure, and CUDA auth eval."
            ),
        },
        "notes": [
            "QMA9 codec implementation is pure Python and independent from public C++ source text.",
            "This manifest is not a component or score claim.",
            f"Rate term remains 25 * archive_bytes / {ORIGINAL_VIDEO_BYTES} for any future exact archive.",
        ],
        "inputs": {
            "raw_mask_path": _repo_rel(raw_mask_path),
            "qma9_payload_path": _repo_rel(qma9_payload_path),
            "model_payload_path": _repo_rel(model_payload_path),
            "pose_payload_path": _repo_rel(pose_payload_path),
            "router_payload_path": _repo_rel(router_payload_path),
            "template_pr81_archive": _repo_rel(template_pr81_archive),
        },
    }
    manifest_path = candidate_dir / "manifest.json"
    _write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--candidate-id", default="qma9_range_mask_transfer_baseline")
    parser.add_argument("--raw-mask", type=Path)
    parser.add_argument("--qma9-payload", type=Path)
    parser.add_argument("--model-payload", type=Path)
    parser.add_argument("--pose-payload", type=Path)
    parser.add_argument("--router-payload", type=Path)
    parser.add_argument("--template-pr81-archive", type=Path, default=DEFAULT_TEMPLATE_PR81_ARCHIVE)
    parser.add_argument("--frame-count", type=int, default=600)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--manifest-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_candidate(
        output_dir=args.output_dir,
        candidate_id=args.candidate_id,
        raw_mask_path=args.raw_mask,
        qma9_payload_path=args.qma9_payload,
        model_payload_path=args.model_payload,
        pose_payload_path=args.pose_payload,
        router_payload_path=args.router_payload,
        template_pr81_archive=args.template_pr81_archive,
        frame_count=args.frame_count,
        width=args.width,
        height=args.height,
        write_archive=not args.manifest_only,
    )
    candidate_dir = args.output_dir / args.candidate_id
    print(f"wrote {candidate_dir / 'manifest.json'}")
    if manifest["archive"] is not None:
        print(f"archive_sha256={manifest['archive']['sha256']} bytes={manifest['archive']['bytes']}")
    else:
        print("archive_not_written=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
