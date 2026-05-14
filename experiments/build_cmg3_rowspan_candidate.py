#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a CMG3 row-span mask grammar candidate archive.

This is a contest-faithful archive builder, not a score claim. It replaces the
frontier mask stream with compressed per-class horizontal spans sampled every
N rows plus a deterministic runtime fill policy. The archive must receive exact
CUDA auth eval before ranking, promotion, or method conclusions.
"""
from __future__ import annotations

import argparse
import bz2
import hashlib
import itertools
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

SCHEMA = "cmg3_rowspan_candidate_v1"
CMG3_MAGIC = b"CMG3"
CMG3_VERSION = 1
CMG3_HEADER_STRUCT = "<4sHI"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
HEIGHT = 384
WIDTH = 512
CLASS_COUNT = 5
ORIGINAL_VIDEO_BYTES = 37_545_489
ROW_FILL_POLICIES = ("nearest", "forward", "linear")
TOP_POLICY_RECORD_LIMIT = 128


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
    unpacker = _load_module(UNPACKER_PATH, "_cmg3_candidate_unpacker")
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
    if arr.shape[1:] != (HEIGHT, WIDTH):
        raise ValueError(f"decoded mask array must be {HEIGHT}x{WIDTH}, got {arr.shape[1:]}")
    if int(arr.min()) < 0 or int(arr.max()) >= CLASS_COUNT:
        raise ValueError(f"decoded mask classes must be in [0,{CLASS_COUNT}), got [{int(arr.min())},{int(arr.max())}]")
    return np.ascontiguousarray(arr)


def row_spans(arr: np.ndarray, *, row_stride: int) -> np.ndarray:
    if row_stride <= 0 or row_stride > arr.shape[1]:
        raise ValueError(f"row_stride must be in [1,{arr.shape[1]}], got {row_stride}")
    frames, height, width = arr.shape
    rows = np.arange(0, height, row_stride, dtype=np.int32)
    spans = np.full((frames, CLASS_COUNT, len(rows), 2), -1, dtype=np.int16)
    for cls in range(CLASS_COUNT):
        for row_index, y in enumerate(rows):
            present = arr[:, int(y), :] == np.uint8(cls)
            any_present = present.any(axis=1)
            first = present.argmax(axis=1)
            last = width - 1 - present[:, ::-1].argmax(axis=1)
            spans[any_present, cls, row_index, 0] = first[any_present].astype(np.int16)
            spans[any_present, cls, row_index, 1] = last[any_present].astype(np.int16)
    return spans


def reconstruct_row_spans(
    spans: np.ndarray,
    *,
    height: int = HEIGHT,
    width: int = WIDTH,
    row_stride: int,
    default_class: int,
    row_fill: str,
    draw_order: tuple[int, ...],
) -> np.ndarray:
    if spans.dtype != np.int16 or spans.ndim != 4:
        raise ValueError(f"spans must be int16 rank-4, got {spans.shape} {spans.dtype}")
    frames, classes, sampled_rows, endpoints = spans.shape
    if classes != CLASS_COUNT or endpoints != 2:
        raise ValueError(f"invalid CMG3 span shape {spans.shape}")
    if sampled_rows != len(np.arange(0, height, row_stride, dtype=np.int32)):
        raise ValueError("span sampled-row count does not match row_stride")
    if not (0 <= default_class < CLASS_COUNT):
        raise ValueError(f"default_class out of range: {default_class}")
    if sorted(draw_order) != list(range(CLASS_COUNT)):
        raise ValueError(f"draw_order must be a permutation of classes: {draw_order}")

    expanded_spans = expanded_row_spans(spans, height=height, row_stride=row_stride, row_fill=row_fill)
    sampled = np.full((frames, height, width), default_class, dtype=np.uint8)
    for cls in draw_order:
        class_spans = expanded_spans[:, int(cls), :, :]
        valid = (class_spans[..., 0] >= 0) & (class_spans[..., 1] >= class_spans[..., 0])
        for row_index in range(height):
            frame_indices = np.flatnonzero(valid[:, row_index])
            for frame_index in frame_indices:
                start = int(class_spans[int(frame_index), row_index, 0])
                end = int(class_spans[int(frame_index), row_index, 1])
                if start < 0 or end >= width:
                    raise ValueError(f"span out of bounds: frame={frame_index} row={row_index} start={start} end={end}")
                sampled[int(frame_index), row_index, start : end + 1] = np.uint8(cls)
    return np.ascontiguousarray(sampled)


def expanded_row_spans(spans: np.ndarray, *, height: int, row_stride: int, row_fill: str) -> np.ndarray:
    """Expand sampled class spans to one span tensor row per output row."""
    if spans.dtype != np.int16 or spans.ndim != 4 or spans.shape[-1] != 2:
        raise ValueError(f"spans must be int16 rank-4 with endpoint axis, got {spans.shape} {spans.dtype}")
    sampled_rows = spans.shape[2]
    rows = np.arange(height, dtype=np.int32)
    if row_fill == "nearest":
        row_indices = np.minimum((rows + row_stride // 2) // row_stride, sampled_rows - 1)
        return np.ascontiguousarray(spans[:, :, row_indices, :])
    elif row_fill == "forward":
        row_indices = np.minimum(rows // row_stride, sampled_rows - 1)
        return np.ascontiguousarray(spans[:, :, row_indices, :])
    elif row_fill == "linear":
        lower = np.minimum(rows // row_stride, sampled_rows - 1)
        upper = np.minimum(lower + 1, sampled_rows - 1)
        denom = np.maximum((upper - lower) * row_stride, 1).astype(np.float32)
        alpha = ((rows - lower * row_stride).astype(np.float32) / denom).reshape(1, 1, height, 1)
        lo = spans[:, :, lower, :].astype(np.float32, copy=False)
        hi = spans[:, :, upper, :].astype(np.float32, copy=False)
        lo_valid = (lo[..., 0] >= 0) & (lo[..., 1] >= lo[..., 0])
        hi_valid = (hi[..., 0] >= 0) & (hi[..., 1] >= hi[..., 0])
        interpolated = np.rint((1.0 - alpha) * lo + alpha * hi).astype(np.int16)
        out = np.full_like(interpolated, -1, dtype=np.int16)
        both = lo_valid & hi_valid
        only_lo = lo_valid & ~hi_valid
        only_hi = hi_valid & ~lo_valid
        out[both] = interpolated[both]
        out[only_lo] = lo.astype(np.int16)[only_lo]
        out[only_hi] = hi.astype(np.int16)[only_hi]
        inverted = out[..., 1] < out[..., 0]
        out[inverted] = -1
        return np.ascontiguousarray(out)
    else:
        raise ValueError(f"unsupported row_fill policy: {row_fill}")


def _coverage_counts_for_policy(source: np.ndarray, spans: np.ndarray, *, row_stride: int, row_fill: str) -> np.ndarray:
    """Count source classes by the 5-bit set of class spans covering each pixel."""
    frames, height, width = source.shape
    expanded = expanded_row_spans(spans, height=height, row_stride=row_stride, row_fill=row_fill)
    coverage = np.zeros((frames, height, width), dtype=np.uint8)
    x = np.arange(width, dtype=np.int16).reshape(1, 1, width)
    for cls in range(CLASS_COUNT):
        starts = expanded[:, cls, :, 0]
        ends = expanded[:, cls, :, 1]
        valid = (starts >= 0) & (ends >= starts)
        covered = valid[:, :, None] & (x >= starts[:, :, None]) & (x <= ends[:, :, None])
        coverage[covered] |= np.uint8(1 << cls)
    joint = coverage.astype(np.uint16) * CLASS_COUNT + source.astype(np.uint16)
    return np.bincount(joint.reshape(-1), minlength=(1 << CLASS_COUNT) * CLASS_COUNT).reshape(
        1 << CLASS_COUNT, CLASS_COUNT
    )


def _prediction_for_coverage(coverage: int, *, default_class: int, draw_order: tuple[int, ...]) -> int:
    predicted = int(default_class)
    for cls in draw_order:
        if coverage & (1 << int(cls)):
            predicted = int(cls)
    return predicted


def _score_policy_from_coverage_counts(
    counts: np.ndarray,
    *,
    default_class: int,
    draw_order: tuple[int, ...],
) -> tuple[int, np.ndarray]:
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    for coverage in range(1 << CLASS_COUNT):
        pred = _prediction_for_coverage(coverage, default_class=default_class, draw_order=draw_order)
        confusion[:, pred] += counts[coverage]
    matches = int(np.trace(confusion))
    total = int(confusion.sum())
    return total - matches, confusion


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


def choose_rowspan_policy(spans: np.ndarray, source: np.ndarray, *, row_stride: int) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    total_pixels = int(source.size)
    draw_orders = tuple(itertools.permutations(range(CLASS_COUNT)))
    for row_fill in ROW_FILL_POLICIES:
        counts = _coverage_counts_for_policy(source, spans, row_stride=row_stride, row_fill=row_fill)
        for default_class in range(CLASS_COUNT):
            for draw_order in draw_orders:
                disagreement_count, confusion = _score_policy_from_coverage_counts(
                    counts,
                    default_class=default_class,
                    draw_order=tuple(int(v) for v in draw_order),
                )
                candidates.append(
                    {
                        "row_fill": row_fill,
                        "draw_order": [int(v) for v in draw_order],
                        "default_class": int(default_class),
                        "pixel_disagreement": float(disagreement_count / total_pixels),
                        "pixel_disagreement_count": int(disagreement_count),
                        "confusion_matrix": confusion.tolist(),
                    }
                )
    candidates.sort(
        key=lambda item: (
            item["pixel_disagreement_count"],
            item["row_fill"],
            item["default_class"],
            item["draw_order"],
        )
    )
    winner = dict(candidates[0])
    recon = reconstruct_row_spans(
        spans,
        row_stride=row_stride,
        default_class=int(winner["default_class"]),
        row_fill=str(winner["row_fill"]),
        draw_order=tuple(int(v) for v in winner["draw_order"]),
    )
    winner["reconstructed_tensor_sha256"] = _sha256_bytes(recon.tobytes(order="C"))
    winner["searched_policy_count"] = len(candidates)
    winner["searched_policies"] = candidates[:TOP_POLICY_RECORD_LIMIT]
    winner["searched_policies_truncated"] = len(candidates) > TOP_POLICY_RECORD_LIMIT
    winner["policy_search"] = {
        "type": "complete_finite_rowspan_policy_space",
        "row_fill_policies": list(ROW_FILL_POLICIES),
        "draw_order_permutations": len(draw_orders),
        "default_class_count": CLASS_COUNT,
        "coverage_bitsets": 1 << CLASS_COUNT,
        "ranking": "exact pixel disagreement against decoded frontier masks before CUDA scoring",
    }
    return winner


def encode_cmg3_payload(
    spans: np.ndarray,
    *,
    row_stride: int,
    source_mask_sha256: str,
    policy: dict[str, Any],
    compressor: str = "lzma_xz",
) -> tuple[bytes, dict[str, Any]]:
    if spans.dtype != np.int16 or spans.ndim != 4:
        raise ValueError("CMG3 spans must be int16 rank-4")
    raw = np.ascontiguousarray(spans.astype("<i2", copy=False)).tobytes(order="C")
    body = _compress(raw, compressor)
    header = {
        "schema": SCHEMA,
        "mode": "row_span_stride_class_predictor_v1",
        "compressor": compressor,
        "span_shape": [int(v) for v in spans.shape],
        "span_tensor_bytes": len(raw),
        "span_tensor_sha256": _sha256_bytes(raw),
        "body_bytes": len(body),
        "body_sha256": _sha256_bytes(body),
        "frame_count": int(spans.shape[0]),
        "height": HEIGHT,
        "width": WIDTH,
        "class_count": CLASS_COUNT,
        "row_stride": int(row_stride),
        "default_class": int(policy["default_class"]),
        "row_fill": str(policy["row_fill"]),
        "draw_order": [int(v) for v in policy["draw_order"]],
        "source_mask_u8_sha256": source_mask_sha256,
        "lossless_under_builder": False,
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
    row_stride: int = 4,
    compressor: str = "lzma_xz",
    force: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is non-empty; pass --force to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    members = _extract_frontier_members(frontier_archive.resolve())
    masks = _load_decoded_masks(decoded_mask_array.resolve())
    spans = row_spans(masks, row_stride=row_stride)
    source_mask_sha = _sha256_bytes(masks.tobytes(order="C"))
    policy = choose_rowspan_policy(spans, masks, row_stride=row_stride)
    recon = reconstruct_row_spans(
        spans,
        row_stride=row_stride,
        default_class=int(policy["default_class"]),
        row_fill=str(policy["row_fill"]),
        draw_order=tuple(int(v) for v in policy["draw_order"]),
    )
    cmg3_payload, cmg3_header = encode_cmg3_payload(
        spans,
        row_stride=row_stride,
        source_mask_sha256=source_mask_sha,
        policy=policy,
        compressor=compressor,
    )

    source_archive = output_dir / "cmg3_source_members.zip"
    _write_source_archive(
        source_archive,
        [
            ("renderer.bin", members["renderer.bin"]),
            ("masks.cmg3", cmg3_payload),
            ("optimized_poses.bin", members["optimized_poses.bin"]),
        ],
    )

    packer = _load_module(PACKER_PATH, "_cmg3_candidate_packer")
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
            "pixel_disagreement_vs_frontier_masks": float((recon != masks).mean()),
            "pixel_disagreement_count": int((recon != masks).sum()),
            "reconstructed_tensor_sha256": _sha256_bytes(recon.tobytes(order="C")),
            "policy_selection": policy,
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
    parser.add_argument("--row-stride", type=int, default=4)
    parser.add_argument("--compressor", choices=("bz2", "raw", "zlib", "lzma_xz"), default="lzma_xz")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_candidate(
        frontier_archive=args.frontier_archive,
        decoded_mask_array=args.decoded_mask_array,
        output_dir=args.output_dir,
        row_stride=args.row_stride,
        compressor=args.compressor,
        force=bool(args.force),
    )
    print(
        json.dumps(
            {
                "archive": manifest["output_archive"],
                "pixel_disagreement": manifest["cmg3"]["pixel_disagreement_vs_frontier_masks"],
                "policy": {
                    "row_fill": manifest["cmg3"]["row_fill"],
                    "draw_order": manifest["cmg3"]["draw_order"],
                },
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
