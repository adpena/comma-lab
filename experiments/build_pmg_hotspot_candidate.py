#!/usr/bin/env python3
"""Build a PMG-HOTSPOT CMG3 residual mask candidate archive.

This converts a planning-only PMG-HOTSPOT manifest into exact archive bytes by
packing row-span grammar plus compact hotspot residual records into
``masks.cmg3``. It is still not score evidence; exact CUDA auth eval of the
resulting ``archive.zip`` is required before ranking or promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import struct
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CMG3_BUILDER_PATH = REPO_ROOT / "experiments" / "build_cmg3_rowspan_candidate.py"
PMG_PLANNER_PATH = REPO_ROOT / "experiments" / "plan_predictive_mask_hotspot.py"
PACKER_PATH = REPO_ROOT / "experiments" / "build_renderer_packed_payload_archive.py"

SCHEMA = "pmg_hotspot_cmg3_candidate_v1"
MODE = "row_span_stride_class_predictor_hotspot_residual_v1"
CMG3_MAGIC = b"CMG3"
CMG3_VERSION = 1
CMG3_HEADER_STRUCT = struct.Struct("<4sHI")
RESIDUAL_RECORD_STRUCT = struct.Struct("<HHHHB")
RESIDUAL_RECORD_STRUCT_NAME = "u16_frame_u16_y_u16_x0_u16_x1_u8_class_le"
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
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _candidate_from_plan(plan: dict[str, Any], candidate_id: str | None) -> dict[str, Any]:
    candidates = [plan.get("best_candidate")]
    candidates.extend(plan.get("candidate_table") or [])
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if candidate_id is None and candidate is plan.get("best_candidate"):
            return candidate
        if candidate.get("candidate_id") == candidate_id:
            return candidate
    raise ValueError(f"candidate {candidate_id!r} not found in PMG-HOTSPOT plan")


def _load_decoded_masks(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if arr.ndim != 3 or arr.dtype != np.uint8:
        raise ValueError(f"decoded mask array must be uint8 rank-3, got shape={arr.shape} dtype={arr.dtype}")
    if arr.shape[1:] != (384, 512):
        raise ValueError(f"decoded mask array must be 384x512, got {arr.shape[1:]}")
    if int(arr.min()) < 0 or int(arr.max()) >= 5:
        raise ValueError(f"decoded mask classes must be in [0,5), got [{int(arr.min())},{int(arr.max())}]")
    return np.ascontiguousarray(arr)


def residual_records_from_candidate(candidate: dict[str, Any]) -> list[tuple[int, int, int, int, int]]:
    atoms = (((candidate.get("protected_atoms") or {}).get("selected_atoms")) or [])
    records: list[tuple[int, int, int, int, int]] = []
    for atom in atoms:
        identity = atom.get("identity") if isinstance(atom, dict) else None
        if not isinstance(identity, dict):
            raise ValueError("PMG-HOTSPOT atom is missing identity")
        frame = int(identity["frame_index"])
        y = int(identity["y"])
        x0 = int(identity["x0"])
        x1 = int(identity["x1_exclusive"])
        source_class = int(identity["source_class"])
        records.append((frame, y, x0, x1, source_class))
    return validate_residual_records(records)


def residual_records_from_atom_ledger(
    ledger_path: Path,
    *,
    atom_count: int,
    expected_source_mask_sha256: str | None = None,
    expected_source_npy_sha256: str | None = None,
) -> tuple[list[tuple[int, int, int, int, int]], dict[str, Any]]:
    """Select charged PMG row-run repair records from a planning-only atom ledger."""
    if atom_count <= 0:
        raise ValueError(f"residual atom count must be positive, got {atom_count}")
    ledger_path = ledger_path.resolve()
    ledger = _read_json(ledger_path)
    if ledger.get("schema") != "cmg3_pixel_lagrangian_atom_ledger_v1":
        raise ValueError(f"{ledger_path} is not a CMG3 pixel atom ledger")
    if ledger.get("score_claim") is not False:
        raise ValueError(f"{ledger_path} must have score_claim=false")
    if ledger.get("evidence_grade") != "planning_only":
        raise ValueError(f"{ledger_path} must be planning_only evidence")

    source_meta = ((ledger.get("inputs") or {}).get("source_mask_array") or {})
    source_npy_sha = source_meta.get("npy_sha256")
    source_tensor_sha = source_meta.get("tensor_sha256")
    if expected_source_npy_sha256 is not None and source_npy_sha is not None:
        if source_npy_sha != expected_source_npy_sha256:
            raise ValueError(
                f"{ledger_path} source NPY SHA mismatch: "
                f"ledger={source_npy_sha} expected={expected_source_npy_sha256}"
            )
    elif expected_source_mask_sha256 is not None and source_tensor_sha != expected_source_mask_sha256:
        raise ValueError(
            f"{ledger_path} source tensor SHA mismatch: "
            f"ledger={source_tensor_sha} expected={expected_source_mask_sha256}"
        )

    atoms = ledger.get("top_atoms")
    if not isinstance(atoms, list):
        raise ValueError(f"{ledger_path} must contain top_atoms")

    records: list[tuple[int, int, int, int, int]] = []
    selected_atom_ids: list[str] = []
    skipped_non_row_run = 0
    for atom in atoms:
        if len(records) >= atom_count:
            break
        if not isinstance(atom, dict):
            raise ValueError(f"{ledger_path} top_atoms contains a non-object atom")
        if atom.get("atom_family") != "row_run":
            skipped_non_row_run += 1
            continue
        identity = atom.get("identity")
        if not isinstance(identity, dict):
            raise ValueError(f"{ledger_path} row_run atom is missing identity")
        source_class = identity.get("source_class", identity.get("class_id"))
        record = (
            int(identity["frame_index"]),
            int(identity["y"]),
            int(identity["x0"]),
            int(identity["x1_exclusive"]),
            int(source_class),
        )
        records.append(record)
        selected_atom_ids.append(str(atom.get("atom_id", "")))

    if len(records) < atom_count:
        raise ValueError(
            f"{ledger_path} only provided {len(records)} row_run atoms, requested {atom_count}"
        )

    records = validate_residual_records(records)
    selected_atom_id_bytes = _json_bytes(selected_atom_ids)
    selected_raw = encode_residual_records(records)
    report = {
        "schema": "pmg_hotspot_atom_ledger_selection_v1",
        "score_claim": False,
        "ledger_path": str(ledger_path),
        "ledger_sha256": _sha256_file(ledger_path),
        "ledger_schema": ledger.get("schema"),
        "ledger_evidence_grade": ledger.get("evidence_grade"),
        "ledger_atom_count": int(ledger.get("atom_count", len(atoms))),
        "ledger_top_atom_count": len(atoms),
        "ledger_source_npy_sha256": source_npy_sha,
        "ledger_source_tensor_sha256": source_tensor_sha,
        "ledger_candidate_tensor_sha256": (((ledger.get("inputs") or {}).get("candidate") or {}).get("tensor_sha256")),
        "requested_row_run_atom_count": int(atom_count),
        "selected_row_run_atom_count": len(records),
        "skipped_non_row_run_top_atoms": int(skipped_non_row_run),
        "selected_atom_ids_sha256": _sha256_bytes(selected_atom_id_bytes),
        "selected_atom_id_prefix": selected_atom_ids[:16],
        "selected_residual_record_count": len(records),
        "selected_residual_pixels_touched": int(sum(x1 - x0 for _frame, _y, x0, x1, _cls in records)),
        "selected_residual_stream_bytes": len(selected_raw),
        "selected_residual_stream_sha256": _sha256_bytes(selected_raw),
        "selection_note": (
            "Rows come from the planning-only atom ledger ordering. This is "
            "archive byte construction only; exact CUDA auth eval is required "
            "before any score, rank, or promotion claim."
        ),
    }
    return records, report


def validate_residual_records(
    records: list[tuple[int, int, int, int, int]],
) -> list[tuple[int, int, int, int, int]]:
    records = sorted(records)
    previous_by_row: dict[tuple[int, int], int] = {}
    for frame, y, x0, x1, source_class in records:
        if not (0 <= frame <= 65535 and 0 <= y <= 65535 and 0 <= x0 < x1 <= 65535 and 0 <= source_class <= 255):
            raise ValueError(f"PMG-HOTSPOT residual record out of fixed-width range: {(frame, y, x0, x1, source_class)}")
        row = (frame, y)
        previous_end = previous_by_row.get(row, -1)
        if x0 < previous_end:
            raise ValueError(f"PMG-HOTSPOT residual records overlap in row {row}: x0={x0}, previous_end={previous_end}")
        previous_by_row[row] = x1
    return records


def parse_pair_indices(raw: str | None) -> tuple[int, ...]:
    if raw is None or raw.strip() == "":
        return ()
    indices: list[int] = []
    for part in raw.split(","):
        value = part.strip()
        if not value:
            continue
        idx = int(value, 10)
        if idx < 0:
            raise ValueError(f"protected pair index must be non-negative, got {idx}")
        indices.append(idx)
    return tuple(sorted(set(indices)))


def residual_records_for_protected_pairs(
    *,
    source: np.ndarray,
    current: np.ndarray,
    pair_indices: tuple[int, ...],
) -> list[tuple[int, int, int, int, int]]:
    """Emit exact repair records for half-frame PMG frames named by pair index."""
    if source.shape != current.shape:
        raise ValueError(f"source/current mask shape mismatch: {source.shape} != {current.shape}")
    if source.ndim != 3:
        raise ValueError(f"source/current masks must be rank-3, got {source.shape}")
    frames, height, _width = source.shape
    records: list[tuple[int, int, int, int, int]] = []
    for pair_index in pair_indices:
        if pair_index >= frames:
            raise ValueError(
                f"protected pair_index={pair_index} is outside half-frame mask range 0..{frames - 1}"
            )
        frame = pair_index
        diff = current[frame] != source[frame]
        if not bool(diff.any()):
            continue
        for y in range(height):
            xs = np.flatnonzero(diff[y])
            if xs.size == 0:
                continue
            start = int(xs[0])
            previous = start
            source_class = int(source[frame, y, start])
            for raw_x in xs[1:]:
                x = int(raw_x)
                cls = int(source[frame, y, x])
                if x != previous + 1 or cls != source_class:
                    records.append((frame, y, start, previous + 1, source_class))
                    start = x
                    source_class = cls
                previous = x
            records.append((frame, y, start, previous + 1, source_class))
    return validate_residual_records(records)


def encode_residual_records(records: list[tuple[int, int, int, int, int]]) -> bytes:
    return b"".join(RESIDUAL_RECORD_STRUCT.pack(*record) for record in records)


def apply_residual_records(base: np.ndarray, records: list[tuple[int, int, int, int, int]]) -> np.ndarray:
    out = np.ascontiguousarray(base.copy())
    frames, height, width = out.shape
    for frame, y, x0, x1, source_class in records:
        if not (0 <= frame < frames):
            raise ValueError(f"residual frame out of range: {frame}")
        if not (0 <= y < height):
            raise ValueError(f"residual row out of range: {y}")
        if not (0 <= x0 < x1 <= width):
            raise ValueError(f"residual run out of range: x0={x0} x1={x1}")
        if not (0 <= source_class < 5):
            raise ValueError(f"residual class out of range: {source_class}")
        out[frame, y, x0:x1] = np.uint8(source_class)
    return np.ascontiguousarray(out)


def encode_pmg_hotspot_payload(
    *,
    spans: np.ndarray,
    residual_records: list[tuple[int, int, int, int, int]],
    policy: dict[str, Any],
    source_mask_sha256: str,
    base_reconstructed_sha256: str,
    final_reconstructed_sha256: str,
    plan_sha256: str,
    candidate_id: str,
    compressor: str,
) -> tuple[bytes, dict[str, Any]]:
    cmg3 = _load_module(CMG3_BUILDER_PATH, "_pmg_hotspot_cmg3_encoder")
    if spans.dtype != np.int16 or spans.ndim != 4:
        raise ValueError(f"spans must be int16 rank-4, got {spans.shape} {spans.dtype}")
    span_raw = np.ascontiguousarray(spans.astype("<i2", copy=False)).tobytes(order="C")
    residual_raw = encode_residual_records(residual_records)
    body_raw = span_raw + residual_raw
    body = cmg3._compress(body_raw, compressor)  # noqa: SLF001 - shared CMG3 compressor
    header = {
        "schema": SCHEMA,
        "mode": MODE,
        "compressor": compressor,
        "candidate_id": candidate_id,
        "pmg_hotspot_plan_sha256": plan_sha256,
        "span_shape": [int(v) for v in spans.shape],
        "span_tensor_bytes": len(span_raw),
        "span_tensor_sha256": _sha256_bytes(span_raw),
        "body_raw_bytes": len(body_raw),
        "body_raw_sha256": _sha256_bytes(body_raw),
        "body_bytes": len(body),
        "body_sha256": _sha256_bytes(body),
        "frame_count": int(spans.shape[0]),
        "height": 384,
        "width": 512,
        "class_count": 5,
        "row_stride": int(policy["row_stride"]),
        "default_class": int(policy["default_class"]),
        "row_fill": str(policy["row_fill"]),
        "draw_order": [int(v) for v in policy["draw_order"]],
        "residual_record_struct": RESIDUAL_RECORD_STRUCT_NAME,
        "residual_record_count": len(residual_records),
        "residual_record_bytes": len(residual_raw),
        "residual_stream_sha256": _sha256_bytes(residual_raw),
        "source_mask_u8_sha256": source_mask_sha256,
        "base_reconstructed_mask_u8_sha256": base_reconstructed_sha256,
        "reconstructed_mask_u8_sha256": final_reconstructed_sha256,
        "score_claim": False,
        "promotion_eligible": False,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    payload = CMG3_HEADER_STRUCT.pack(CMG3_MAGIC, CMG3_VERSION, len(header_bytes)) + header_bytes + body
    return payload, header


def _write_source_archive(path: Path, members: list[tuple[str, bytes]]) -> None:
    cmg3 = _load_module(CMG3_BUILDER_PATH, "_pmg_hotspot_source_archive")
    cmg3._write_source_archive(path, members)  # noqa: SLF001 - deterministic ZIP writer


def build_candidate(
    *,
    plan_json: Path,
    frontier_archive: Path,
    decoded_mask_array: Path | None,
    output_dir: Path,
    candidate_id: str | None = None,
    compressor: str = "lzma_xz",
    residual_atom_ledger: Path | None = None,
    residual_atom_count: int = 0,
    protect_pair_indices: tuple[int, ...] = (),
    force: bool = False,
) -> dict[str, Any]:
    if residual_atom_ledger is None and residual_atom_count:
        raise ValueError("--residual-atom-count requires --residual-atom-ledger")
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is non-empty; pass --force to overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    cmg3 = _load_module(CMG3_BUILDER_PATH, "_pmg_hotspot_cmg3_builder")
    packer = _load_module(PACKER_PATH, "_pmg_hotspot_packer")
    plan = _read_json(plan_json.resolve())
    candidate = _candidate_from_plan(plan, candidate_id)
    source_info = plan.get("source") or {}
    mask_array_path = (decoded_mask_array or Path(str(source_info["decoded_mask_array_path"]))).resolve()
    masks = _load_decoded_masks(mask_array_path)
    source_mask_sha = _sha256_bytes(masks.tobytes(order="C"))
    expected_source_sha = source_info.get("decoded_mask_tensor_sha256")
    if expected_source_sha is not None and str(expected_source_sha) != source_mask_sha:
        raise ValueError(
            f"decoded mask tensor SHA mismatch: plan={expected_source_sha} actual={source_mask_sha}"
        )

    row_stride = int(candidate["row_stride"])
    policy = {
        "row_stride": row_stride,
        "default_class": int(candidate["default_class"]),
        "row_fill": str(candidate["row_fill"]),
        "draw_order": [int(v) for v in candidate["draw_order"]],
    }
    spans = cmg3.row_spans(masks, row_stride=row_stride)
    base = cmg3.reconstruct_row_spans(
        spans,
        row_stride=row_stride,
        default_class=policy["default_class"],
        row_fill=policy["row_fill"],
        draw_order=tuple(policy["draw_order"]),
    )
    plan_residual_records = residual_records_from_candidate(candidate)
    atom_residual_records: list[tuple[int, int, int, int, int]] = []
    atom_selection_report = None
    if residual_atom_ledger is not None:
        atom_residual_records, atom_selection_report = residual_records_from_atom_ledger(
            residual_atom_ledger,
            atom_count=residual_atom_count,
            expected_source_mask_sha256=source_mask_sha,
            expected_source_npy_sha256=_sha256_file(mask_array_path),
        )
    plan_atom_records = validate_residual_records(plan_residual_records + atom_residual_records)
    plan_final = apply_residual_records(base, plan_atom_records)
    additional_records = residual_records_for_protected_pairs(
        source=masks,
        current=plan_final,
        pair_indices=protect_pair_indices,
    )
    residual_records = validate_residual_records(plan_atom_records + additional_records)
    final = apply_residual_records(base, residual_records)
    payload, header = encode_pmg_hotspot_payload(
        spans=spans,
        residual_records=residual_records,
        policy=policy,
        source_mask_sha256=source_mask_sha,
        base_reconstructed_sha256=_sha256_bytes(base.tobytes(order="C")),
        final_reconstructed_sha256=_sha256_bytes(final.tobytes(order="C")),
        plan_sha256=_sha256_file(plan_json.resolve()),
        candidate_id=str(candidate["candidate_id"]),
        compressor=compressor,
    )

    frontier_members = cmg3._extract_frontier_members(frontier_archive.resolve())  # noqa: SLF001
    source_archive = output_dir / "pmg_hotspot_source_members.zip"
    _write_source_archive(
        source_archive,
        [
            ("renderer.bin", frontier_members["renderer.bin"]),
            ("masks.cmg3", payload),
            ("optimized_poses.bin", frontier_members["optimized_poses.bin"]),
        ],
    )
    archive_path = output_dir / "archive.zip"
    packed_meta = packer.build_packed_archive(
        source_archive,
        archive_path,
        brotli_quality=11,
        pose_codec=packer.POSE_QP1_CODEC,
        payload_member_name=packer.SHORT_PAYLOAD_MEMBER_NAME,
        payload_format=packer.PAYLOAD_FORMAT_RPK1_JSON,
    )

    frontier_size = frontier_archive.stat().st_size
    archive_size = archive_path.stat().st_size
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
        "pmg_hotspot_plan": {
            "path": str(plan_json.resolve()),
            "sha256": _sha256_file(plan_json.resolve()),
            "candidate_id": str(candidate["candidate_id"]),
        },
        "frontier_archive": {
            "path": str(frontier_archive.resolve()),
            "bytes": frontier_size,
            "sha256": _sha256_file(frontier_archive.resolve()),
        },
        "decoded_mask_array": {
            "path": str(mask_array_path),
            "npy_sha256": _sha256_file(mask_array_path),
            "tensor_sha256": source_mask_sha,
            "shape": [int(v) for v in masks.shape],
        },
        "pmg_hotspot_cmg3": {
            **header,
            "payload_bytes": len(payload),
            "payload_sha256": _sha256_bytes(payload),
            "base_pixel_disagreement_vs_source_count": int((base != masks).sum()),
            "base_pixel_disagreement_vs_source_fraction": float((base != masks).mean()),
            "final_pixel_disagreement_vs_source_count": int((final != masks).sum()),
            "final_pixel_disagreement_vs_source_fraction": float((final != masks).mean()),
            "residual_pixels_touched": int(sum(x1 - x0 for _frame, _y, x0, x1, _cls in residual_records)),
            "pair_protection": {
                "schema": "pmg_hotspot_pair_protection_v1",
                "score_claim": False,
                "indexing": "contest_pair_index_maps_to_half_frame_mask_index",
                "protected_pair_indices": [int(v) for v in protect_pair_indices],
                "plan_residual_record_count": len(plan_residual_records),
                "plan_residual_pixels_touched": int(
                    sum(x1 - x0 for _frame, _y, x0, x1, _cls in plan_residual_records)
                ),
                "additional_residual_record_count": len(additional_records),
                "additional_residual_pixels_touched": int(
                    sum(x1 - x0 for _frame, _y, x0, x1, _cls in additional_records)
                ),
                "additional_residual_stream_sha256": _sha256_bytes(
                    encode_residual_records(additional_records)
                ),
            },
            "atom_ledger_selection": atom_selection_report,
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
            "formula_only_rate_delta_vs_frontier": 25.0 * float(delta_bytes) / float(ORIGINAL_VIDEO_BYTES),
            "score_claim": False,
        },
        "packed_payload": packed_meta,
        "required_next_steps": [
            "focused local runtime-loader and packed-payload tests must stay green",
            "run exact CUDA diagnostic on the archive bytes only after dispatch claim",
            "T4 promotion only if diagnostic components survive",
        ],
    }
    (output_dir / "build_manifest.json").write_bytes(_json_bytes(manifest))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-json", type=Path, required=True)
    parser.add_argument("--frontier-archive", type=Path, required=True)
    parser.add_argument("--decoded-mask-array", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-id")
    parser.add_argument("--compressor", choices=("bz2", "raw", "zlib", "lzma_xz"), default="lzma_xz")
    parser.add_argument(
        "--residual-atom-ledger",
        type=Path,
        help=(
            "Planning-only CMG3 pixel atom ledger. The builder selects the top "
            "row-run atoms as charged residual records; this is not score evidence."
        ),
    )
    parser.add_argument(
        "--residual-atom-count",
        type=int,
        default=0,
        help="Number of top row-run atoms to select from --residual-atom-ledger.",
    )
    parser.add_argument(
        "--protect-pair-indices",
        help=(
            "Comma-separated contest pair indices to fully repair in the half-frame "
            "mask grammar. This is a charged byte/protection knob, not score evidence."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build_candidate(
        plan_json=args.plan_json,
        frontier_archive=args.frontier_archive,
        decoded_mask_array=args.decoded_mask_array,
        output_dir=args.output_dir,
        candidate_id=args.candidate_id,
        compressor=args.compressor,
        residual_atom_ledger=args.residual_atom_ledger,
        residual_atom_count=int(args.residual_atom_count),
        protect_pair_indices=parse_pair_indices(args.protect_pair_indices),
        force=bool(args.force),
    )
    print(
        json.dumps(
            {
                "archive": manifest["output_archive"],
                "candidate_id": manifest["pmg_hotspot_plan"]["candidate_id"],
                "final_pixel_disagreement": manifest["pmg_hotspot_cmg3"][
                    "final_pixel_disagreement_vs_source_fraction"
                ],
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
