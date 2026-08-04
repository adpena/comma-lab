#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""CQ1 comma10k public-model chart-overlap prototype.

This is a scorer-free measurement for the #938 leg-B route.  It runs the
custodied public comma10k segmentation model on qo1 inflated frame_1 images and
measures whether its Road/Lane boundary chart addresses the SE3 Road/Lane band
target flips.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import scipy.ndimage as ndimage
import segmentation_models_pytorch as smp
import torch


REPO = Path(__file__).resolve().parents[1]
MODEL_DIR = Path("/Volumes/VertigoDataTier/pact/public_models/comma10k_segnet")
RAW_PATH = Path("/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit/inflated/0.raw")
ARGMAX_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache")
CX1_ARGMAX = ARGMAX_DIR / "cx1_argmax_n600.npy"
GT_ARGMAX = ARGMAX_DIR / "gt_argmax_n600.npy"
DEFAULT_OUT_DIR = REPO / ".omx/research/ddm_cq1_20260804"
DEFAULT_BULK_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_cq1_20260804")

PAIRS = (
    31,
    43,
    62,
    82,
    94,
    118,
    147,
    165,
    167,
    182,
    185,
    200,
    237,
    241,
    247,
    259,
    272,
    286,
    288,
    292,
    296,
    306,
    327,
    382,
    390,
    419,
    473,
    488,
    525,
    555,
    560,
    581,
)

EXPECTED_SHA256 = {
    MODEL_DIR / "model.safetensors": "8208672861ad1b111dc98f3a7c54196d29875b709c7353e2dd1b7614343fb3a8",
    MODEL_DIR / "config.json": "2b8f16dbad9bd85386609386a9cb5dedc6e0c518253a9af484e0a128d9463c88",
    MODEL_DIR
    / "albumentations_config_eval.json": "d260853fe0a993e23613ff38039fdce59264f5fe31f729c1fa65f8c3e5fde913",
    RAW_PATH: "3ce7d269a7080a4024a576694cd0ddc697099c64cd02fdd2bb879339e4b03f31",
    CX1_ARGMAX: "5e903de650e60ec6a64b34eb455fa1bc911223551d0b31e9ae45cc906e1490be",
    GT_ARGMAX: "b74a14b226a5aceb5824899898bcb06e5413c54b7db2441268da7bc91a10db5d",
}

N_PAIRS = 600
FRAMES_PER_PAIR = 2
CAM_H = 874
CAM_W = 1164
CHANNELS = 3
SEG_H = 384
SEG_W = 512
ROAD = 0
LANE = 1
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
GOOD_OVERLAP_THRESHOLD = 0.80


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def verify_hashes(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        size, digest = sha256_file(path)
        expected = EXPECTED_SHA256.get(path)
        if expected is not None and digest != expected:
            raise SystemExit(f"sha256 mismatch for {path}: got {digest}, expected {expected}")
        out[str(path)] = {
            "bytes": size,
            "sha256": digest,
            "expected_sha256": expected,
            "verified": expected is None or digest == expected,
        }
    return out


def load_eval_config(path: Path) -> dict[str, Any]:
    cfg = json.loads(path.read_text())
    transforms = cfg["transform"]["transforms"]
    resize = transforms[0]
    tensor = transforms[1]
    if resize["__class_fullname__"] != "Resize" or resize["height"] != SEG_H or resize["width"] != SEG_W:
        raise SystemExit(f"unexpected eval resize config: {resize}")
    if resize["interpolation"] != 1:
        raise SystemExit(f"unexpected eval interpolation: {resize['interpolation']}")
    if tensor["__class_fullname__"] != "ToTensorV2":
        raise SystemExit(f"unexpected tensor transform: {tensor}")
    return cfg


def load_raw_frames(raw_path: Path) -> np.memmap:
    expected_bytes = N_PAIRS * FRAMES_PER_PAIR * CAM_H * CAM_W * CHANNELS
    actual_bytes = raw_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise SystemExit(f"raw layout mismatch: got {actual_bytes} bytes, expected {expected_bytes}")
    return np.memmap(raw_path, dtype=np.uint8, mode="r", shape=(N_PAIRS * FRAMES_PER_PAIR, CAM_H, CAM_W, CHANNELS))


def preprocess_frame(frame: np.ndarray) -> torch.Tensor:
    resized = cv2.resize(frame, (SEG_W, SEG_H), interpolation=cv2.INTER_LINEAR)
    chw = np.ascontiguousarray(resized.transpose(2, 0, 1))
    return torch.from_numpy(chw).float()


def run_public_model(
    *,
    model_dir: Path,
    raw_path: Path,
    pairs: tuple[int, ...],
    batch_size: int,
) -> np.ndarray:
    frames = load_raw_frames(raw_path)
    model = smp.from_pretrained(str(model_dir))
    model.to("cpu")
    model.eval()

    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch_pairs = pairs[start : start + batch_size]
            tensors = [preprocess_frame(np.asarray(frames[pair * FRAMES_PER_PAIR + 1])) for pair in batch_pairs]
            x = torch.stack(tensors, dim=0)
            logits = model(x)
            pred = logits.argmax(dim=1).to(torch.uint8).cpu().numpy()
            rows.append(pred)
    return np.concatenate(rows, axis=0)


def band_for(frame_argmax: np.ndarray, radius: int) -> np.ndarray:
    st3 = ndimage.generate_binary_structure(2, 2)
    road = frame_argmax == ROAD
    lane = frame_argmax == LANE
    return ndimage.binary_dilation(road, st3, radius) & ndimage.binary_dilation(lane, st3, radius)


def per_class_iou(pred: np.ndarray, ref: np.ndarray) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cls, name in enumerate(CLASS_NAMES):
        p = pred == cls
        r = ref == cls
        inter = int(np.count_nonzero(p & r))
        union = int(np.count_nonzero(p | r))
        out[name] = {
            "class_id": cls,
            "intersection": inter,
            "union": union,
            "pred_pixels": int(np.count_nonzero(p)),
            "ref_pixels": int(np.count_nonzero(r)),
            "iou": float(inter / union) if union else 1.0,
        }
    return out


def band_stats(pred: np.ndarray, ref: np.ndarray, target: np.ndarray, radius: int) -> dict[str, Any]:
    se3_captured = 0
    micro_captured = 0
    overlap_captured = 0
    se3_band_pixels = 0
    micro_band_pixels = 0
    band_intersection = 0
    band_union = 0
    pair_rows: list[dict[str, Any]] = []

    for n, pair in enumerate(PAIRS):
        se3_band = band_for(ref[n], radius)
        micro_band = band_for(pred[n], radius)
        frame_target = target[n]
        frame_se3_captured = int(np.count_nonzero(frame_target & se3_band))
        frame_micro_captured = int(np.count_nonzero(frame_target & micro_band))
        frame_overlap_captured = int(np.count_nonzero(frame_target & se3_band & micro_band))
        frame_se3_pixels = int(np.count_nonzero(se3_band))
        frame_micro_pixels = int(np.count_nonzero(micro_band))
        frame_intersection = int(np.count_nonzero(se3_band & micro_band))
        frame_union = int(np.count_nonzero(se3_band | micro_band))

        se3_captured += frame_se3_captured
        micro_captured += frame_micro_captured
        overlap_captured += frame_overlap_captured
        se3_band_pixels += frame_se3_pixels
        micro_band_pixels += frame_micro_pixels
        band_intersection += frame_intersection
        band_union += frame_union
        pair_rows.append(
            {
                "pair": pair,
                "radius": radius,
                "target_flips": int(np.count_nonzero(frame_target)),
                "se3_captured_flips": frame_se3_captured,
                "micro_captured_flips": frame_micro_captured,
                "micro_overlap_of_se3_captured_flips": frame_overlap_captured,
                "se3_band_pixels": frame_se3_pixels,
                "micro_band_pixels": frame_micro_pixels,
                "band_intersection_pixels": frame_intersection,
                "band_union_pixels": frame_union,
            }
        )

    return {
        "radius": radius,
        "se3_captured_flips": se3_captured,
        "micro_captured_flips": micro_captured,
        "micro_overlap_of_se3_captured_flips": overlap_captured,
        "micro_over_se3_captured_fraction": float(overlap_captured / se3_captured) if se3_captured else 0.0,
        "micro_capture_fraction_of_total_targets": float(micro_captured / int(target.sum())) if int(target.sum()) else 0.0,
        "se3_capture_fraction_of_total_targets": float(se3_captured / int(target.sum())) if int(target.sum()) else 0.0,
        "se3_band_pixels": se3_band_pixels,
        "micro_band_pixels": micro_band_pixels,
        "band_precision_vs_se3": float(band_intersection / micro_band_pixels) if micro_band_pixels else 0.0,
        "band_recall_vs_se3": float(band_intersection / se3_band_pixels) if se3_band_pixels else 0.0,
        "band_iou_vs_se3": float(band_intersection / band_union) if band_union else 1.0,
        "pair_rows": pair_rows,
    }


def counts_by_pair(argmax: np.ndarray) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n, pair in enumerate(PAIRS):
        counts = np.bincount(argmax[n].reshape(-1), minlength=len(CLASS_NAMES))
        rows.append(
            {
                "pair": pair,
                "class_counts": {name: int(counts[i]) for i, name in enumerate(CLASS_NAMES)},
            }
        )
    return rows


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def measure(args: argparse.Namespace) -> dict[str, Any]:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.bulk_dir.mkdir(parents=True, exist_ok=True)

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)

    eval_config = load_eval_config(args.model_dir / "albumentations_config_eval.json")
    hashes = verify_hashes(
        [
            args.model_dir / "model.safetensors",
            args.model_dir / "config.json",
            args.model_dir / "albumentations_config_eval.json",
            args.raw_path,
            args.cx1_argmax,
            args.gt_argmax,
        ]
    )

    pred = run_public_model(
        model_dir=args.model_dir,
        raw_path=args.raw_path,
        pairs=PAIRS,
        batch_size=args.batch_size,
    )
    pred_path = args.bulk_dir / "comma10k_public_model_argmax_pairs_n32.npy"
    np.save(pred_path, pred)
    pred_bytes, pred_sha = sha256_file(pred_path)

    cx1 = np.load(args.cx1_argmax, mmap_mode="r")[list(PAIRS)]
    gt = np.load(args.gt_argmax, mmap_mode="r")[list(PAIRS)]
    if tuple(pred.shape) != (len(PAIRS), SEG_H, SEG_W):
        raise SystemExit(f"unexpected prediction shape: {pred.shape}")
    if tuple(cx1.shape) != pred.shape or tuple(gt.shape) != pred.shape:
        raise SystemExit(f"unexpected subset shapes: pred={pred.shape}, cx1={cx1.shape}, gt={gt.shape}")

    target = (gt != cx1) & (((gt == ROAD) & (cx1 == LANE)) | ((gt == LANE) & (cx1 == ROAD)))
    total_targets = int(np.count_nonzero(target))
    band_rows = [band_stats(pred, cx1, target, radius) for radius in args.radii]
    decisive = next(row for row in band_rows if row["radius"] == 1)
    decisive_overlap = decisive["micro_over_se3_captured_fraction"]
    verdict = "GOOD-OVERLAP" if decisive_overlap >= GOOD_OVERLAP_THRESHOLD else "POOR-OVERLAP"

    pair_rows_path = args.out_dir / "cq1_pair_rows.jsonl"
    with pair_rows_path.open("w") as handle:
        micro_counts = counts_by_pair(pred)
        cx1_counts = counts_by_pair(np.asarray(cx1))
        gt_counts = counts_by_pair(np.asarray(gt))
        by_radius = {row["radius"]: row["pair_rows"] for row in band_rows}
        for idx, pair in enumerate(PAIRS):
            row = {
                "pair": pair,
                "micro_counts": micro_counts[idx]["class_counts"],
                "cx1_counts": cx1_counts[idx]["class_counts"],
                "gt_counts": gt_counts[idx]["class_counts"],
                "target_flips": int(np.count_nonzero(target[idx])),
                "band_rows": [by_radius[radius][idx] for radius in args.radii],
            }
            handle.write(json.dumps(jsonable(row), sort_keys=True) + "\n")

    result = {
        "schema": "ddm_cq1_comma10k_chart_overlap.v1",
        "axis": "[macOS-CPU advisory / public-model chart-overlap scorer-free]",
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
        "selection_mode": "n32 stratified random non-prefix; seed 20260804; reused SE2 pair list",
        "pairs": list(PAIRS),
        "inputs": {
            "model_dir": str(args.model_dir),
            "raw_path": str(args.raw_path),
            "raw_layout": {
                "frames": N_PAIRS * FRAMES_PER_PAIR,
                "height": CAM_H,
                "width": CAM_W,
                "channels": CHANNELS,
                "dtype": "uint8",
                "frame_selection": "frame_1 of each selected pair => raw frame index pair*2+1",
            },
            "cx1_argmax": str(args.cx1_argmax),
            "gt_argmax": str(args.gt_argmax),
            "hashes": hashes,
            "class_order": {name: cls for cls, name in enumerate(CLASS_NAMES)},
        },
        "public_model": {
            "loader": "segmentation_models_pytorch.from_pretrained(local_dir)",
            "weights_source": "custodied commaai/comma10k-segnet public model; no contest-video finetune/adaptation",
            "eval_mode": True,
            "device": "cpu",
        },
        "preprocessing": {
            "source_config": eval_config,
            "manual_equivalent_reason": "albumentations is not installed in the Pact venv; config contains only Resize and ToTensorV2",
            "resize": {"height": SEG_H, "width": SEG_W, "interpolation": "cv2.INTER_LINEAR"},
            "tensor": "HWC uint8 -> CHW torch.float32, no normalization, preserving the eval config's absence of Normalize",
            "resample_choice_for_chart": "model emits logits directly at 384x512 after eval Resize; argmax uses that scorer grid, so no post-argmax resize is applied",
        },
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "smp": smp.__version__,
            "opencv": cv2.__version__,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "batch_size": args.batch_size,
        },
        "artifacts": {
            "summary_json": str(args.out_dir / "cq1_summary.json"),
            "pair_rows_jsonl": str(pair_rows_path),
            "public_model_argmax_subset": {
                "path": str(pred_path),
                "bytes": pred_bytes,
                "sha256": pred_sha,
                "shape": list(pred.shape),
                "dtype": str(pred.dtype),
            },
        },
        "denominators": {
            "selected_pairs": len(PAIRS),
            "subset_scorer_cells": int(np.prod(pred.shape)),
            "subset_road_lane_target_flips": total_targets,
        },
        "iou_vs_cx1": per_class_iou(pred, np.asarray(cx1)),
        "iou_vs_gt": per_class_iou(pred, np.asarray(gt)),
        "band_overlap_vs_se3_cx1_chart": [{key: value for key, value in row.items() if key != "pair_rows"} for row in band_rows],
        "decisive_metric": {
            "surface": "SE3 r1 Road/Lane band captured flips from cx1 stand-in",
            "threshold": GOOD_OVERLAP_THRESHOLD,
            "measured_micro_overlap_fraction": decisive_overlap,
            "denominator_se3_r1_captured_flips": decisive["se3_captured_flips"],
            "numerator_micro_overlap_of_se3_r1_captured_flips": decisive["micro_overlap_of_se3_captured_flips"],
        },
        "verdict": {
            "status": verdict,
            "verdict_scope": "FORMULATION: public comma10k-segnet receiver-chart proxy on qo1 n32 stratified frame_1 subset; not n600, not contest authority",
            "disposition": (
                "QUEUE comma10k-only tiny-student distillation sizing plus counted-bytes break-even vs 81-101 KB stream prices"
                if verdict == "GOOD-OVERLAP"
                else "FOLD micro-student route for SE3 on this public-model chart source; remaining chart source is a receiver-successor archive carrying a legal class/chart field"
            ),
        },
        "boundaries": [
            "No contest SegNet/PoseNet forward was run.",
            "No upstream/evaluate.py run was performed.",
            "No archive.zip was built.",
            "No public model training, finetuning, distillation, adaptation, or checkpoint selection was performed.",
            "The public model's 38,502,740 B weights are economic/countable if ever shipped; this arm measures only whether a tiny public-data-only student is worth a later unit.",
            "All persisted evidence paths are under .omx/research/ddm_cq1_20260804 or /Volumes/VertigoDataTier/pact/ddm_cq1_20260804; no /tmp evidence.",
        ],
        "next_if_resumed": (
            "If GOOD-OVERLAP, fire CQ2: train/size a comma10k-only tiny student with frozen public-data recipe, then re-run this overlap and count weight bytes against the 81-101 KB SE3 stream prices. "
            "If POOR-OVERLAP, do not train the micro-student for SE3; wait for a receiver-successor archive that legally carries a class/chart field."
        ),
        "own_vehicle_frontier_line": "S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved.",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--raw-path", type=Path, default=RAW_PATH)
    parser.add_argument("--cx1-argmax", type=Path, default=CX1_ARGMAX)
    parser.add_argument("--gt-argmax", type=Path, default=GT_ARGMAX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bulk-dir", type=Path, default=DEFAULT_BULK_DIR)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--radii", type=int, nargs="+", default=[1, 2, 3])
    args = parser.parse_args()
    result = measure(args)
    out = args.out_dir / "cq1_summary.json"
    out.write_text(json.dumps(jsonable(result), indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "out": str(out),
                "verdict": result["verdict"]["status"],
                "decisive_overlap": result["decisive_metric"]["measured_micro_overlap_fraction"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
