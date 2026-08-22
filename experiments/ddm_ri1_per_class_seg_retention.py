#!/usr/bin/env python3
"""Retain n600 per-class SegNet outcomes for the integrated RI1 RGB output.

This is a post-score diagnostic.  It reuses upstream's frozen DistortionNet
and its own AV/Tensor datasets, persists both GT and candidate argmax fields,
the per-pair Seg/Pose vectors, and both six-value PoseNet outputs for every
batch before reducing them.  The retained vectors must reproduce the already-
recorded advisory d_seg and d_pose.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
PAIR_COUNT = 600
CLASS_COUNT = len(CLASS_NAMES)
DEFAULT_UPSTREAM = Path("/Volumes/APDataStore/pact/upstream_eval_mirror_20260815")


class RI1PerClassError(RuntimeError):
    """A scorer, retained-payload, or reduction invariant failed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_fact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_fact(expected: dict[str, Any]) -> dict[str, Any]:
    path_text = expected.get("path")
    if not path_text:
        raise RI1PerClassError("retained result contains an incomplete file fact")
    actual = file_fact(Path(path_text))
    if actual["bytes"] != expected.get("bytes") or actual["sha256"] != expected.get("sha256"):
        raise RI1PerClassError(f"retained result payload drifted: {path_text}")
    return actual


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_array(path: Path, array: np.ndarray, dtype: np.dtype[Any]) -> dict[str, Any]:
    contiguous = np.ascontiguousarray(array, dtype=dtype)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        contiguous.tofile(stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return file_fact(path) | {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)}


def chunk_statistics(gt: np.ndarray, candidate: np.ndarray) -> dict[str, Any]:
    if gt.shape != candidate.shape or gt.ndim != 3:
        raise RI1PerClassError("retained SegNet argmax fields have different geometry")
    if np.any(gt >= CLASS_COUNT) or np.any(candidate >= CLASS_COUNT):
        raise RI1PerClassError("retained SegNet argmax field has an invalid class")
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    np.add.at(confusion, (gt.reshape(-1), candidate.reshape(-1)), 1)
    total_pixels = int(gt.size)
    flips = int(total_pixels - np.trace(confusion))
    return {
        "pairs": int(gt.shape[0]),
        "seg_height": int(gt.shape[1]),
        "seg_width": int(gt.shape[2]),
        "pixels": total_pixels,
        "flips": flips,
        "confusion_gt_rows_candidate_columns": confusion.tolist(),
    }


def _load_retained_array(expected: dict[str, Any]) -> np.ndarray:
    path = Path(expected.get("path", ""))
    actual = file_fact(path)
    if actual["bytes"] != expected.get("bytes") or actual["sha256"] != expected.get("sha256"):
        raise RI1PerClassError(f"retained array payload drifted: {path}")
    dtype = np.dtype(expected.get("dtype"))
    shape = tuple(expected.get("shape", ()))
    if not shape or actual["bytes"] != int(np.prod(shape)) * dtype.itemsize:
        raise RI1PerClassError(f"retained array payload has the wrong geometry: {path}")
    return np.fromfile(path, dtype=dtype).reshape(shape)


def validate_retained_chunk(
    receipt: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = tuple(receipt.get("shape", ()))
    if len(shape) != 3:
        raise RI1PerClassError("retained chunk receipt has no three-dimensional shape")
    arrays: list[np.ndarray] = []
    for key in ("gt_argmax", "candidate_argmax"):
        expected = receipt.get(key, {})
        array = _load_retained_array(expected)
        if array.dtype != np.uint8 or array.shape != shape:
            raise RI1PerClassError(f"retained {key} payload has the wrong type or shape")
        arrays.append(array)
    pair_count = shape[0]
    diagnostic_arrays: dict[str, np.ndarray] = {}
    for key, expected_shape in (
        ("per_pair_pose_distortion", (pair_count,)),
        ("per_pair_seg_distortion", (pair_count,)),
        ("gt_pose_vectors", (pair_count, 6)),
        ("candidate_pose_vectors", (pair_count, 6)),
    ):
        array = _load_retained_array(receipt.get(key, {}))
        if array.dtype != np.float32 or array.shape != expected_shape:
            raise RI1PerClassError(f"retained {key} payload has the wrong type or shape")
        diagnostic_arrays[key] = array
    return (
        arrays[0],
        arrays[1],
        diagnostic_arrays["per_pair_pose_distortion"],
        diagnostic_arrays["per_pair_seg_distortion"],
    )


def score_or_resume_chunk(
    *,
    chunk_index: int,
    batch_gt: Any,
    batch_candidate: Any,
    distortion_net: Any,
    out_dir: Path,
) -> dict[str, Any]:
    receipt_path = out_dir / "chunks" / f"chunk_{chunk_index:04d}.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        gt, candidate, pose_values, seg_values = validate_retained_chunk(receipt)
        statistics = chunk_statistics(gt, candidate)
        statistics["pose_distortion_sum"] = float(np.sum(pose_values, dtype=np.float64))
        statistics["seg_distortion_sum"] = float(np.sum(seg_values, dtype=np.float64))
        if statistics != receipt.get("statistics"):
            raise RI1PerClassError(f"retained chunk {chunk_index} statistics drifted")
        return receipt

    import torch

    with torch.inference_mode():
        gt_pose_outputs, gt_seg_logits = distortion_net(batch_gt.to("cpu"))
        candidate_pose_outputs, candidate_seg_logits = distortion_net(batch_candidate.to("cpu"))
        pose_distortion = distortion_net.posenet.compute_distortion(gt_pose_outputs, candidate_pose_outputs)
        gt_labels = gt_seg_logits.argmax(dim=1)
        candidate_labels = candidate_seg_logits.argmax(dim=1)
        seg_distortion = (gt_labels != candidate_labels).float().mean(dim=(1, 2))
        gt = gt_labels.to(dtype=torch.uint8).cpu().numpy()
        candidate = candidate_labels.to(dtype=torch.uint8).cpu().numpy()
        pose_values = pose_distortion.to(dtype=torch.float32).cpu().numpy()
        seg_values = seg_distortion.to(dtype=torch.float32).cpu().numpy()
        gt_pose_values = gt_pose_outputs["pose"][..., :6].to(dtype=torch.float32).cpu().numpy()
        candidate_pose_values = candidate_pose_outputs["pose"][..., :6].to(dtype=torch.float32).cpu().numpy()

    gt_path = out_dir / "chunks" / f"chunk_{chunk_index:04d}.gt_argmax.u8"
    candidate_path = out_dir / "chunks" / f"chunk_{chunk_index:04d}.candidate_argmax.u8"
    # Persist both semantic payloads before reducing them to counts.
    gt_fact = atomic_array(gt_path, gt, np.dtype(np.uint8))
    candidate_fact = atomic_array(candidate_path, candidate, np.dtype(np.uint8))
    pose_fact = atomic_array(
        out_dir / "chunks" / f"chunk_{chunk_index:04d}.per_pair_pose.f32",
        pose_values,
        np.dtype("<f4"),
    )
    seg_fact = atomic_array(
        out_dir / "chunks" / f"chunk_{chunk_index:04d}.per_pair_seg.f32",
        seg_values,
        np.dtype("<f4"),
    )
    gt_pose_fact = atomic_array(
        out_dir / "chunks" / f"chunk_{chunk_index:04d}.gt_pose_vectors.f32",
        gt_pose_values,
        np.dtype("<f4"),
    )
    candidate_pose_fact = atomic_array(
        out_dir / "chunks" / f"chunk_{chunk_index:04d}.candidate_pose_vectors.f32",
        candidate_pose_values,
        np.dtype("<f4"),
    )
    statistics = chunk_statistics(gt, candidate)
    statistics["pose_distortion_sum"] = float(np.sum(pose_values, dtype=np.float64))
    statistics["seg_distortion_sum"] = float(np.sum(seg_values, dtype=np.float64))
    receipt = {
        "schema": "ddm_ri1_per_class_seg_chunk.v1",
        "chunk_index": chunk_index,
        "shape": list(gt.shape),
        "gt_argmax": gt_fact,
        "candidate_argmax": candidate_fact,
        "per_pair_pose_distortion": pose_fact,
        "per_pair_seg_distortion": seg_fact,
        "gt_pose_vectors": gt_pose_fact,
        "candidate_pose_vectors": candidate_pose_fact,
        "statistics": statistics,
        "complete": True,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def aggregate(
    receipts: list[dict[str, Any]],
    reported_d_seg: float,
    reported_d_pose: float,
) -> dict[str, Any]:
    confusion = np.zeros((CLASS_COUNT, CLASS_COUNT), dtype=np.int64)
    pairs = 0
    pixels = 0
    pose_sum = 0.0
    seg_sum = 0.0
    for receipt in receipts:
        stats = receipt["statistics"]
        pairs += int(stats["pairs"])
        pixels += int(stats["pixels"])
        pose_sum += float(stats["pose_distortion_sum"])
        seg_sum += float(stats["seg_distortion_sum"])
        confusion += np.asarray(stats["confusion_gt_rows_candidate_columns"], dtype=np.int64)
    if pairs != PAIR_COUNT or pixels <= 0 or int(confusion.sum()) != pixels:
        raise RI1PerClassError("per-class scorer did not cover the complete n600 field")
    flips = int(pixels - np.trace(confusion))
    d_seg = flips / pixels
    retained_d_seg = seg_sum / pairs
    retained_d_pose = pose_sum / pairs
    class_vs_vector_error = abs(d_seg - retained_d_seg) / (abs(d_seg) or 1.0)
    seg_relative_error = abs(retained_d_seg - reported_d_seg) / (abs(reported_d_seg) or 1.0)
    pose_relative_error = abs(retained_d_pose - reported_d_pose) / (abs(reported_d_pose) or 1.0)
    if max(class_vs_vector_error, seg_relative_error, pose_relative_error) > 1e-6:
        raise RI1PerClassError(
            "retained scorer payloads do not reproduce the reported components: "
            f"class_vs_vector={class_vs_vector_error} seg={seg_relative_error} "
            f"pose={pose_relative_error}"
        )

    rows = []
    for class_id, name in enumerate(CLASS_NAMES):
        gt_pixels = int(confusion[class_id].sum())
        candidate_pixels = int(confusion[:, class_id].sum())
        true_positive = int(confusion[class_id, class_id])
        class_flips = gt_pixels - true_positive
        union = gt_pixels + candidate_pixels - true_positive
        rows.append(
            {
                "class_id": class_id,
                "class_name": name,
                "gt_pixels": gt_pixels,
                "gt_area_fraction": gt_pixels / pixels,
                "flips_from_gt_class": class_flips,
                "conditional_d_seg": class_flips / gt_pixels if gt_pixels else 0.0,
                "contribution_to_total_d_seg": class_flips / pixels,
                "iou": true_positive / union if union else 0.0,
            }
        )
    return {
        "pairs": pairs,
        "pixels": pixels,
        "flips": flips,
        "d_seg_recomputed": d_seg,
        "d_seg_from_retained_per_pair": retained_d_seg,
        "d_seg_reported": reported_d_seg,
        "d_seg_relative_error": seg_relative_error,
        "d_pose_from_retained_per_pair": retained_d_pose,
        "d_pose_reported": reported_d_pose,
        "d_pose_relative_error": pose_relative_error,
        "class_vs_per_pair_seg_relative_error": class_vs_vector_error,
        "reduction_verified": True,
        "confusion_gt_rows_candidate_columns": confusion.tolist(),
        "per_class": rows,
        "lane_class_1": rows[1],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--video-names-file", type=Path, required=True)
    parser.add_argument("--reported-d-seg", type=float, required=True)
    parser.add_argument("--reported-d-pose", type=float, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--batch-pairs", type=int, default=16)
    parser.add_argument("--torch-threads", type=int, default=0)
    args = parser.parse_args()
    if not 1 <= args.batch_pairs <= 120:
        raise RI1PerClassError("--batch-pairs must be in [1, 120]")
    out_dir = (args.resume_from or args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "RESULT.json"
    if result_path.is_file():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("complete") is not True:
            raise RI1PerClassError("existing per-class result is not complete")
        for key in ("video_names_file", "candidate_raw", "segnet_weights", "posenet_weights"):
            validate_fact(result.get(key, {}))
        chunk_facts = result.get("chunk_receipts", [])
        receipts = []
        for expected in chunk_facts:
            receipt_path = Path(validate_fact(expected)["path"])
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            validate_retained_chunk(receipt)
            receipts.append(receipt)
        if aggregate(
            receipts,
            result["summary"]["d_seg_reported"],
            result["summary"]["d_pose_reported"],
        ) != result.get("summary"):
            raise RI1PerClassError("retained per-class aggregate drifted")
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    upstream = args.upstream_dir.resolve()
    submission = args.submission_dir.resolve()
    names_path = args.video_names_file.resolve()
    raw_path = submission / "inflated/0.raw"
    if raw_path.stat().st_size != 3_662_409_600:
        raise RI1PerClassError("retained integrated RGB raw has the wrong byte count")
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))

    import torch
    from frame_utils import AVVideoDataset, TensorVideoDataset
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    torch.manual_seed(1234)
    torch.use_deterministic_algorithms(True)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    distortion_net = DistortionNet().eval().to("cpu")
    distortion_net.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    names = [line.strip() for line in names_path.read_text().splitlines() if line.strip()]
    common = {
        "batch_size": args.batch_pairs,
        "device": torch.device("cpu"),
        "num_threads": 2,
        "seed": 1234,
    }
    gt_dataset = AVVideoDataset(names, data_dir=upstream / "videos", **common)
    candidate_dataset = TensorVideoDataset(names, data_dir=submission / "inflated", **common)
    gt_dataset.prepare_data()
    candidate_dataset.prepare_data()
    gt_loader = torch.utils.data.DataLoader(gt_dataset, batch_size=None, num_workers=0)
    candidate_loader = torch.utils.data.DataLoader(candidate_dataset, batch_size=None, num_workers=0)
    receipts = []
    for chunk_index, ((_, _, batch_gt), (_, _, batch_candidate)) in enumerate(
        zip(gt_loader, candidate_loader, strict=True)
    ):
        receipts.append(
            score_or_resume_chunk(
                chunk_index=chunk_index,
                batch_gt=batch_gt,
                batch_candidate=batch_candidate,
                distortion_net=distortion_net,
                out_dir=out_dir,
            )
        )
    summary = aggregate(receipts, args.reported_d_seg, args.reported_d_pose)
    result = {
        "schema": "ddm_ri1_per_class_seg_retention.v1",
        "complete": True,
        "axis": "[macOS-CPU advisory diagnostic; frozen upstream SegNet]",
        "score_claim": False,
        "promotable": False,
        "gt_decode": "upstream.frame_utils.AVVideoDataset -> yuv420_to_rgb",
        "candidate_decode": "upstream.frame_utils.TensorVideoDataset over retained full-RGB raw",
        "batch_pairs": args.batch_pairs,
        "torch_threads": torch.get_num_threads(),
        "submission_dir": str(submission),
        "upstream_dir": str(upstream),
        "video_names_file": file_fact(names_path),
        "candidate_raw": file_fact(raw_path),
        "segnet_weights": file_fact(Path(segnet_sd_path)),
        "posenet_weights": file_fact(Path(posenet_sd_path)),
        "chunk_receipts": [file_fact(out_dir / "chunks" / f"chunk_{index:04d}.json") for index in range(len(receipts))],
        "summary": summary,
    }
    atomic_json(result_path, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
