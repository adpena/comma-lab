#!/usr/bin/env python3
"""Build a CMG3 non-background row-run mask candidate archive.

This builder defaults the decoded mask canvas to class 0 and stores only the
largest nonzero class runs per row. It is lossy unless ``max_runs_per_row`` is
large enough to preserve every nonzero run. The output archive is contest
faithful but remains non-score evidence until exact CUDA auth eval lands.
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import importlib.util
import json
import lzma
import struct
import sys
import zipfile
import zlib
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKER_PATH = REPO_ROOT / "experiments" / "build_renderer_packed_payload_archive.py"
UNPACKER_PATH = REPO_ROOT / "submissions" / "robust_current" / "unpack_renderer_payload.py"

SCHEMA = "cmg3_nonzero_row_runs_candidate_v1"
CMG3_MAGIC = b"CMG3"
CMG3_VERSION = 1
CMG3_HEADER_STRUCT = "<4sHI"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
HEIGHT = 384
WIDTH = 512
CLASS_COUNT = 5
ORIGINAL_VIDEO_BYTES = 37_545_489


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
    unpacker = _load_module(UNPACKER_PATH, "_cmg3_runs_candidate_unpacker")
    payload = _read_single_member(frontier_archive)
    _header, members = unpacker._parse_payload(payload)  # noqa: SLF001 - local contest packer helper
    required = {"renderer.bin", "masks.mkv", "optimized_poses.bin"}
    missing = required - set(members)
    if missing:
        raise ValueError(f"frontier archive did not unpack required members: {sorted(missing)}")
    return {name: members[name] for name in sorted(required)}


def _load_decoded_masks(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        archive = np.load(path, allow_pickle=False)
        try:
            keys = list(archive.files)
            preferred = [key for key in ("masks", "mask", "decoded_masks", "array") if key in keys]
            if len(preferred) == 1:
                arr = archive[preferred[0]]
            elif len(keys) == 1:
                arr = archive[keys[0]]
            else:
                raise ValueError(
                    f"decoded mask npz must contain one array or a masks/decoded_masks array, got {keys}"
                )
        finally:
            archive.close()
    else:
        arr = np.load(path, allow_pickle=False)
    if arr.ndim != 3 or arr.dtype != np.uint8:
        raise ValueError(f"decoded mask array must be uint8 rank-3, got shape={arr.shape} dtype={arr.dtype}")
    if arr.shape[1:] != (HEIGHT, WIDTH):
        raise ValueError(f"decoded mask array must be {HEIGHT}x{WIDTH}, got {arr.shape[1:]}")
    if int(arr.min()) < 0 or int(arr.max()) >= CLASS_COUNT:
        raise ValueError(f"decoded mask classes must be in [0,{CLASS_COUNT}), got [{int(arr.min())},{int(arr.max())}]")
    return np.ascontiguousarray(arr)


def _row_nonzero_runs(row: np.ndarray) -> list[tuple[int, int, int, int]]:
    runs: list[tuple[int, int, int, int]] = []
    current = int(row[0])
    start = 0
    for x, value in enumerate(row[1:], start=1):
        value = int(value)
        if value != current:
            if current != 0:
                runs.append((start, x - 1, current, x - start))
            current = value
            start = x
    if current != 0:
        runs.append((start, WIDTH - 1, current, WIDTH - start))
    return runs


def encode_run_stream(
    masks: np.ndarray,
    *,
    max_runs_per_row: int,
) -> tuple[bytes, np.ndarray, dict[str, Any]]:
    if not (0 <= max_runs_per_row <= 255):
        raise ValueError(f"max_runs_per_row must be in [0,255], got {max_runs_per_row}")
    out = bytearray()
    recon = np.zeros_like(masks)
    total_runs = 0
    kept_runs = 0
    row_run_counts: list[int] = []
    for frame_index, frame in enumerate(masks):
        for y, row in enumerate(frame):
            runs = _row_nonzero_runs(row)
            total_runs += len(runs)
            selected = sorted(
                sorted(runs, key=lambda item: (-item[3], item[0]))[:max_runs_per_row],
                key=lambda item: item[0],
            )
            kept_runs += len(selected)
            row_run_counts.append(len(selected))
            out.append(len(selected))
            for start, end, class_id, _length in selected:
                out.append(class_id)
                out += int(start).to_bytes(2, "little")
                out += int(end).to_bytes(2, "little")
                recon[frame_index, y, start : end + 1] = np.uint8(class_id)
    diff = recon != masks
    stats = {
        "max_runs_per_row": int(max_runs_per_row),
        "total_nonzero_runs": int(total_runs),
        "kept_runs": int(kept_runs),
        "kept_run_fraction": float(kept_runs / total_runs) if total_runs else 1.0,
        "runs_per_row_mean": float(np.mean(row_run_counts)) if row_run_counts else 0.0,
        "runs_per_row_p95": float(np.percentile(row_run_counts, 95)) if row_run_counts else 0.0,
        "pixel_disagreement": float(diff.mean()),
        "pixel_disagreement_count": int(diff.sum()),
        "reconstructed_tensor_sha256": _sha256_bytes(recon.tobytes(order="C")),
    }
    return bytes(out), recon, stats


def _compress(raw: bytes, compressor: str) -> bytes:
    if compressor == "raw":
        return raw
    if compressor == "bz2":
        return bz2.compress(raw, compresslevel=9)
    if compressor == "zlib":
        return zlib.compress(raw, level=9)
    if compressor == "lzma_xz":
        return lzma.compress(raw, preset=6, format=lzma.FORMAT_XZ)
    raise ValueError(f"unsupported CMG3 compressor: {compressor!r}")


def _choose_compressor(raw: bytes, compressor: str) -> tuple[str, bytes, list[dict[str, Any]]]:
    choices = ("bz2", "lzma_xz", "zlib", "raw") if compressor == "auto" else (compressor,)
    results = [
        {"compressor": name, "body": _compress(raw, name)}
        for name in choices
    ]
    results.sort(key=lambda item: (len(item["body"]), item["compressor"]))
    report = [
        {
            "compressor": str(item["compressor"]),
            "body_bytes": len(item["body"]),
            "body_sha256": _sha256_bytes(item["body"]),
        }
        for item in results
    ]
    winner = results[0]
    return str(winner["compressor"]), bytes(winner["body"]), report


def encode_cmg3_payload(
    run_stream: bytes,
    *,
    frame_count: int,
    max_runs_per_row: int,
    source_mask_sha256: str,
    reconstructed_mask_sha256: str,
    pixel_disagreement: float,
    pixel_disagreement_count: int,
    compressor: str = "auto",
) -> tuple[bytes, dict[str, Any]]:
    selected_compressor, body, compression_report = _choose_compressor(run_stream, compressor)
    header = {
        "schema": SCHEMA,
        "mode": "nonzero_row_runs_topk_v1",
        "compressor": selected_compressor,
        "run_stream_bytes": len(run_stream),
        "run_stream_sha256": _sha256_bytes(run_stream),
        "body_bytes": len(body),
        "body_sha256": _sha256_bytes(body),
        "compression_candidates": compression_report,
        "frame_count": int(frame_count),
        "height": HEIGHT,
        "width": WIDTH,
        "class_count": CLASS_COUNT,
        "default_class": 0,
        "max_runs_per_row": int(max_runs_per_row),
        "record_struct": "u8_count_then_u8_class_u16_start_u16_end_le",
        "source_mask_u8_sha256": source_mask_sha256,
        "reconstructed_mask_u8_sha256": reconstructed_mask_sha256,
        "pixel_disagreement_vs_source": float(pixel_disagreement),
        "pixel_disagreement_count_vs_source": int(pixel_disagreement_count),
        "lossless_under_builder": bool(pixel_disagreement_count == 0),
        "score_claim": False,
        "promotion_eligible": False,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = struct.pack(CMG3_HEADER_STRUCT, CMG3_MAGIC, CMG3_VERSION, len(header_bytes)) + header_bytes + body
    return payload, header


def build_candidate(
    *,
    frontier_archive: Path,
    decoded_mask_array: Path,
    output_dir: Path,
    max_runs_per_row: int,
    compressor: str = "auto",
    force: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is non-empty; pass --force to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    members = _extract_frontier_members(frontier_archive.resolve())
    masks = _load_decoded_masks(decoded_mask_array.resolve())
    run_stream, recon, run_stats = encode_run_stream(masks, max_runs_per_row=max_runs_per_row)
    source_mask_sha = _sha256_bytes(masks.tobytes(order="C"))
    recon_sha = _sha256_bytes(recon.tobytes(order="C"))
    cmg3_payload, cmg3_header = encode_cmg3_payload(
        run_stream,
        frame_count=int(masks.shape[0]),
        max_runs_per_row=max_runs_per_row,
        source_mask_sha256=source_mask_sha,
        reconstructed_mask_sha256=recon_sha,
        pixel_disagreement=float(run_stats["pixel_disagreement"]),
        pixel_disagreement_count=int(run_stats["pixel_disagreement_count"]),
        compressor=compressor,
    )

    source_archive = output_dir / "cmg3_nonzero_runs_source_members.zip"
    _write_source_archive(
        source_archive,
        [
            ("renderer.bin", members["renderer.bin"]),
            ("masks.cmg3", cmg3_payload),
            ("optimized_poses.bin", members["optimized_poses.bin"]),
        ],
    )

    packer = _load_module(PACKER_PATH, "_cmg3_runs_candidate_packer")
    archive_path = output_dir / "archive.zip"
    packed_meta = packer.build_packed_archive(
        source_archive,
        archive_path,
        brotli_quality=11,
        pose_codec=packer.POSE_QP1_CODEC,
        payload_member_name=packer.SHORT_PAYLOAD_MEMBER_NAME,
        payload_format=packer.PAYLOAD_FORMAT_RPK1_JSON,
    )

    archive_size = archive_path.stat().st_size
    frontier_size = frontier_archive.stat().st_size
    delta_bytes = archive_size - frontier_size
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
            "bytes": frontier_size,
            "sha256": _sha256_file(frontier_archive.resolve()),
        },
        "decoded_mask_array": {
            "path": str(decoded_mask_array.resolve()),
            "npy_sha256": _sha256_file(decoded_mask_array.resolve()),
            "tensor_sha256": source_mask_sha,
            "shape": [int(v) for v in masks.shape],
        },
        "cmg3": {
            **cmg3_header,
            "payload_bytes": len(cmg3_payload),
            "payload_sha256": _sha256_bytes(cmg3_payload),
            "run_stats": run_stats,
        },
        "source_archive": {
            "path": str(source_archive),
            "bytes": source_archive.stat().st_size,
            "sha256": _sha256_file(source_archive),
        },
        "output_archive": {
            "path": str(archive_path),
            "bytes": archive_size,
            "sha256": _sha256_file(archive_path),
            "delta_bytes_vs_frontier": delta_bytes,
            "formula_only_rate_delta_vs_frontier": 25.0 * delta_bytes / ORIGINAL_VIDEO_BYTES,
        },
        "packed_payload": packed_meta,
        "required_next_steps": [
            "local unpack/inflate smoke to prove masks.cmg3 routes through runtime",
            "exact CUDA diagnostic on the archive bytes",
            "T4 promotion only if component gates are near or below C-067",
        ],
    }
    (output_dir / "build_manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-archive", type=Path, required=True)
    parser.add_argument("--decoded-mask-array", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-runs-per-row", type=int, default=2)
    parser.add_argument("--compressor", choices=("auto", "bz2", "raw", "zlib", "lzma_xz"), default="auto")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_candidate(
        frontier_archive=args.frontier_archive,
        decoded_mask_array=args.decoded_mask_array,
        output_dir=args.output_dir,
        max_runs_per_row=args.max_runs_per_row,
        compressor=args.compressor,
        force=bool(args.force),
    )
    print(
        json.dumps(
            {
                "archive": manifest["output_archive"],
                "pixel_disagreement": manifest["cmg3"]["run_stats"]["pixel_disagreement"],
                "max_runs_per_row": manifest["cmg3"]["max_runs_per_row"],
                "compressor": manifest["cmg3"]["compressor"],
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
