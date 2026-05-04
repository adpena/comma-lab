#!/usr/bin/env python3
"""Build a CMG2 spatial-downsample mask candidate archive.

This is a contest-faithful archive builder, not a score claim. It replaces the
frontier mask stream with a compressed low-resolution class tensor plus a
deterministic nearest-neighbor inflate decoder. The resulting archive still
requires exact CUDA auth eval before ranking or promotion.
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import importlib.util
import json
import struct
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKER_PATH = REPO_ROOT / "experiments" / "build_renderer_packed_payload_archive.py"
UNPACKER_PATH = REPO_ROOT / "submissions" / "robust_current" / "unpack_renderer_payload.py"

SCHEMA = "cmg2_downsample_candidate_v1"
CMG2_MAGIC = b"CMG2"
CMG2_VERSION = 1
CMG2_HEADER_STRUCT = "<4sHI"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _write_source_archive(path: Path, members: list[tuple[str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        for name, data in members:
            info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIMESTAMP)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o644 << 16
            info.extra = b""
            info.comment = b""
            zf.writestr(info, data)


def _read_single_member(path: Path, *, member: str = "p") -> bytes:
    with zipfile.ZipFile(path, "r") as zf:
        names = zf.namelist()
        if names != [member]:
            raise ValueError(f"expected single member {member!r} in {path}, got {names!r}")
        return zf.read(member)


def _extract_frontier_members(frontier_archive: Path) -> dict[str, bytes]:
    unpacker = _load_module(UNPACKER_PATH, "_cmg2_candidate_unpacker")
    payload = _read_single_member(frontier_archive)
    _header, members = unpacker._parse_payload(payload)  # noqa: SLF001 - local contest packer helper
    required = {"renderer.bin", "masks.mkv", "optimized_poses.bin"}
    missing = required - set(members)
    if missing:
        raise ValueError(f"frontier archive did not unpack required members: {sorted(missing)}")
    return {name: members[name] for name in sorted(required)}


def _load_decoded_masks(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 3 or arr.dtype != np.uint8:
        raise ValueError(f"decoded mask array must be uint8 rank-3, got shape={arr.shape} dtype={arr.dtype}")
    if int(arr.min()) < 0 or int(arr.max()) >= 5:
        raise ValueError(f"decoded mask classes must be in [0,5), got [{int(arr.min())},{int(arr.max())}]")
    return np.ascontiguousarray(arr)


def downsample_block_mode(arr: np.ndarray, *, scale_y: int, scale_x: int) -> tuple[np.ndarray, np.ndarray, float]:
    if scale_y <= 0 or scale_x <= 0:
        raise ValueError("scale factors must be positive")
    frames, height, width = arr.shape
    if height % scale_y or width % scale_x:
        raise ValueError(f"shape {arr.shape} is not divisible by scale {(scale_y, scale_x)}")
    blocks = arr.reshape(frames, height // scale_y, scale_y, width // scale_x, scale_x)
    counts = np.stack([(blocks == cls).sum(axis=(2, 4)) for cls in range(5)], axis=0)
    low = counts.argmax(axis=0).astype(np.uint8)
    recon = np.repeat(np.repeat(low, scale_y, axis=1), scale_x, axis=2)
    disagreement = float((recon != arr).mean())
    return low, recon.astype(np.uint8, copy=False), disagreement


def encode_cmg2_payload(
    low: np.ndarray,
    *,
    scale_y: int,
    scale_x: int,
    compressor: str = "bz2",
) -> tuple[bytes, dict[str, Any]]:
    if low.dtype != np.uint8 or low.ndim != 3:
        raise ValueError("CMG2 low tensor must be uint8 rank-3")
    raw = np.ascontiguousarray(low).tobytes(order="C")
    if compressor == "bz2":
        body = bz2.compress(raw, compresslevel=9)
    elif compressor == "raw":
        body = raw
    else:
        raise ValueError(f"unsupported CMG2 compressor: {compressor!r}")
    header = {
        "schema": SCHEMA,
        "mode": "spatial_downsample_block_mode_v1",
        "compressor": compressor,
        "scale": [int(scale_y), int(scale_x)],
        "low_shape": [int(v) for v in low.shape],
        "low_tensor_bytes": len(raw),
        "low_tensor_sha256": _sha256_bytes(raw),
        "body_bytes": len(body),
        "body_sha256": _sha256_bytes(body),
        "score_claim": False,
        "promotion_eligible": False,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = struct.pack(CMG2_HEADER_STRUCT, CMG2_MAGIC, CMG2_VERSION, len(header_bytes)) + header_bytes + body
    return payload, header


def build_candidate(
    *,
    frontier_archive: Path,
    decoded_mask_array: Path,
    output_dir: Path,
    scale_y: int,
    scale_x: int,
    compressor: str = "bz2",
    force: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is non-empty; pass --force to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    members = _extract_frontier_members(frontier_archive.resolve())
    masks = _load_decoded_masks(decoded_mask_array.resolve())
    low, recon, disagreement = downsample_block_mode(masks, scale_y=scale_y, scale_x=scale_x)
    cmg2_payload, cmg2_header = encode_cmg2_payload(
        low,
        scale_y=scale_y,
        scale_x=scale_x,
        compressor=compressor,
    )

    source_archive = output_dir / "cmg2_source_members.zip"
    _write_source_archive(
        source_archive,
        [
            ("renderer.bin", members["renderer.bin"]),
            ("masks.cmg2", cmg2_payload),
            ("optimized_poses.bin", members["optimized_poses.bin"]),
        ],
    )

    packer = _load_module(PACKER_PATH, "_cmg2_candidate_packer")
    archive_path = output_dir / "archive.zip"
    packed_meta = packer.build_packed_archive(
        source_archive,
        archive_path,
        brotli_quality=11,
        pose_codec=packer.POSE_QP1_CODEC,
        payload_member_name=packer.SHORT_PAYLOAD_MEMBER_NAME,
        payload_format=packer.PAYLOAD_FORMAT_RPK1_JSON,
    )

    manifest = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": "empirical_archive_candidate_until_exact_cuda",
        "canonical_score_source_required": (
            "archive.zip -> inflate.sh -> upstream/evaluate.py via "
            "experiments/contest_auth_eval.py --device cuda"
        ),
        "frontier_archive": {
            "path": str(frontier_archive.resolve()),
            "bytes": frontier_archive.stat().st_size,
            "sha256": _sha256_file(frontier_archive.resolve()),
        },
        "decoded_mask_array": {
            "path": str(decoded_mask_array.resolve()),
            "npy_sha256": _sha256_file(decoded_mask_array.resolve()),
            "tensor_sha256": _sha256_bytes(masks.tobytes(order="C")),
            "shape": [int(v) for v in masks.shape],
        },
        "cmg2": {
            **cmg2_header,
            "payload_bytes": len(cmg2_payload),
            "payload_sha256": _sha256_bytes(cmg2_payload),
            "pixel_disagreement_vs_frontier_masks": disagreement,
            "reconstructed_tensor_sha256": _sha256_bytes(recon.tobytes(order="C")),
        },
        "source_archive": {
            "path": str(source_archive),
            "bytes": source_archive.stat().st_size,
            "sha256": _sha256_file(source_archive),
        },
        "output_archive": {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": _sha256_file(archive_path),
            "delta_bytes_vs_frontier": archive_path.stat().st_size - frontier_archive.stat().st_size,
        },
        "packed_payload": packed_meta,
        "required_next_steps": [
            "local unpack/inflate smoke to prove masks.cmg2 routes through runtime",
            "fast H100 exact CUDA diagnostic",
            "T4 promotion only if H100 component gates are near or below C-067",
        ],
    }
    (output_dir / "build_manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-archive", type=Path, required=True)
    parser.add_argument("--decoded-mask-array", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scale-y", type=int, default=2)
    parser.add_argument("--scale-x", type=int, default=2)
    parser.add_argument("--compressor", choices=("bz2", "raw"), default="bz2")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_candidate(
        frontier_archive=args.frontier_archive,
        decoded_mask_array=args.decoded_mask_array,
        output_dir=args.output_dir,
        scale_y=args.scale_y,
        scale_x=args.scale_x,
        compressor=args.compressor,
        force=bool(args.force),
    )
    print(
        json.dumps(
            {
                "archive": manifest["output_archive"],
                "pixel_disagreement": manifest["cmg2"]["pixel_disagreement_vs_frontier_masks"],
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
