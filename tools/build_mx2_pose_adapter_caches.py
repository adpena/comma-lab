#!/usr/bin/env python3
"""Build ddm_mx2 Row-2 pose adapter caches.

This script materializes the exact cache shapes consumed by PR130's
``train_pose_carrier_full.py`` without editing the vendored trainer:

* target cache: ``{"seg": [600,384,512], "pose": [600,6]}``
* master cache: ``{"source_checkpoint": resolved_path, "masters": [600,3,874,1164]}``

The tq1c master cache is an adapter cache.  Its frames come from our inflated
parent RGB raw bytes, while ``source_checkpoint`` is the identity guard required
by the PR130 trainer when ``--reuse-master-cache`` is used.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import lzma
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch


N_TOTAL_PAIRS = 600
CAMERA_H = 874
CAMERA_W = 1164
EVAL_H = 384
EVAL_W = 512

SSD_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_mx2_20260806")
SOURCE_REPO_ROOT = Path("/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo")
DEFAULT_OFFICIAL_CACHE_XZ = SOURCE_REPO_ROOT / "artifacts/caches/gt_cache_600_official_ada.pt.xz"  # GT_LINEAGE_OK: decompressed bytes are registry-classified DALI_NVDEC sha256 382d7dfe38b37c0c
DEFAULT_GT_REFERENCE_NPZ = Path("experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
DEFAULT_TARGET_OUT = SSD_ROOT / "inputs/gt_pose_cache_600.pt"
DEFAULT_PARENT_RAW = (
    Path("/Volumes/VertigoDataTier/pact/ddm_et2_20260806")
    / "parent_tq1c_decode/submission/inflated/0.raw"
)
DEFAULT_MASTER_OUT = SSD_ROOT / "master_cache/OUR_SURFACE_MASTERS.pt"
DEFAULT_MASTER_CHECKPOINT = (
    SOURCE_REPO_ROOT
    / "artifacts/checkpoints/semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_torch_save(payload: Any, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp.{os.getpid()}")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _shape_dtype(tensor: torch.Tensor) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": int(tensor.min().item()) if tensor.numel() and not tensor.is_floating_point() else (
            float(tensor.min().item()) if tensor.numel() else None
        ),
        "max": int(tensor.max().item()) if tensor.numel() and not tensor.is_floating_point() else (
            float(tensor.max().item()) if tensor.numel() else None
        ),
    }


def _load_pr130_official_cache(cache_xz: Path) -> tuple[dict[str, torch.Tensor], bytes]:
    compressed = cache_xz.read_bytes()
    raw = lzma.decompress(compressed)
    cache = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if not isinstance(cache, dict) or "seg" not in cache or "pose" not in cache:
        raise ValueError("official target cache must contain 'seg' and 'pose'")
    seg = cache["seg"]
    pose = cache["pose"]
    if tuple(seg.shape) != (N_TOTAL_PAIRS, EVAL_H, EVAL_W):
        raise ValueError(f"seg cache shape mismatch: {tuple(seg.shape)}")
    if tuple(pose.shape) != (N_TOTAL_PAIRS, 6):
        raise ValueError(f"pose cache shape mismatch: {tuple(pose.shape)}")
    return {"seg": seg.to(torch.uint8).contiguous(), "pose": pose.float().contiguous()}, raw


def _reference_comparison(payload: dict[str, torch.Tensor], reference_npz: Path | None) -> dict[str, Any]:
    if reference_npz is None or not reference_npz.exists():
        return {"reference_npz": str(reference_npz) if reference_npz is not None else None}
    reference = np.load(reference_npz)
    out: dict[str, Any] = {
        "reference_npz": str(reference_npz),
        "reference_keys": list(reference.files),
    }
    if "lstars" in reference.files:
        ref_seg = torch.from_numpy(np.asarray(reference["lstars"])).to(torch.uint8)
        out["reference_seg_disagreement"] = float((payload["seg"] != ref_seg).float().mean())
    if "gt_poses" in reference.files:
        ref_pose = torch.from_numpy(np.asarray(reference["gt_poses"])).float()
        delta = payload["pose"] - ref_pose
        out["reference_pose_mse"] = float(delta.square().mean())
        out["reference_pose_max_abs"] = float(delta.abs().max())
    return out


def build_target_cache(cache_xz: Path, out: Path, receipt: Path, reference_npz: Path | None) -> dict[str, Any]:
    payload, raw = _load_pr130_official_cache(cache_xz)
    atomic_torch_save(payload, out)
    report = {
        "schema": "ddm_mx2_gt_pose_target_cache_receipt.v1",
        "score_claim": False,
        "promotion_eligible": False,
        "cache_role": "PR130 trainer target-cache schema for Row-2 pose adapter",
        "source_cache_xz": str(cache_xz),
        "source_cache_xz_bytes": cache_xz.stat().st_size,
        "source_cache_xz_sha256": sha256_file(cache_xz),
        "source_uncompressed_bytes": len(raw),
        "source_uncompressed_sha256": sha256_bytes(raw),
        "out": str(out),
        "out_bytes": out.stat().st_size,
        "out_sha256": sha256_file(out),
        "seg": _shape_dtype(payload["seg"]),
        "pose": _shape_dtype(payload["pose"]),
        "reference_comparison": _reference_comparison(payload, reference_npz),
    }
    write_json(receipt, report)
    return report


def _validate_rgb_raw_size(raw: Path, pairs: int, camera_h: int, camera_w: int) -> int:
    expected_frames = pairs * 2
    expected = expected_frames * camera_h * camera_w * 3
    actual = raw.stat().st_size
    if actual != expected:
        raise ValueError(f"raw RGB size mismatch: expected {expected}, got {actual}")
    return expected_frames


def build_master_cache(
    raw: Path,
    master_checkpoint: Path,
    out: Path,
    receipt: Path,
    *,
    pairs: int = N_TOTAL_PAIRS,
    camera_h: int = CAMERA_H,
    camera_w: int = CAMERA_W,
    chunk_pairs: int = 8,
) -> dict[str, Any]:
    if pairs < 1:
        raise ValueError("pairs must be positive")
    if chunk_pairs < 1:
        raise ValueError("chunk_pairs must be positive")
    frames = _validate_rgb_raw_size(raw, pairs, camera_h, camera_w)
    if not master_checkpoint.exists():
        raise FileNotFoundError(master_checkpoint)

    mm = np.memmap(raw, dtype=np.uint8, mode="r", shape=(frames, camera_h, camera_w, 3))
    masters = torch.empty((pairs, 3, camera_h, camera_w), dtype=torch.uint8)
    for start in range(0, pairs, chunk_pairs):
        end = min(start + chunk_pairs, pairs)
        second_frames = mm[(2 * start + 1):(2 * end):2]
        chw = np.ascontiguousarray(second_frames.transpose(0, 3, 1, 2))
        masters[start:end].copy_(torch.from_numpy(chw))

    payload = {
        "source_checkpoint": str(master_checkpoint.resolve()),
        "masters": masters,
    }
    atomic_torch_save(payload, out)
    report = {
        "schema": "ddm_mx2_tq1c_master_surface_cache_receipt.v1",
        "score_claim": False,
        "promotion_eligible": False,
        "cache_role": "PR130 trainer master-cache schema for tq1c Row-2 adapter",
        "surface_source": "tq1c parent inflated RGB raw frame_1 per pair",
        "raw": str(raw),
        "raw_bytes": raw.stat().st_size,
        "raw_sha256": sha256_file(raw),
        "pairs": pairs,
        "frames": frames,
        "camera_h": camera_h,
        "camera_w": camera_w,
        "master_checkpoint_guard": str(master_checkpoint.resolve()),
        "master_checkpoint_bytes": master_checkpoint.stat().st_size,
        "master_checkpoint_sha256": sha256_file(master_checkpoint),
        "out": str(out),
        "out_bytes": out.stat().st_size,
        "out_sha256": sha256_file(out),
        "masters": _shape_dtype(masters),
    }
    write_json(receipt, report)
    return report


def validate_caches(target_cache: Path, master_cache: Path, master_checkpoint: Path) -> dict[str, Any]:
    target = torch.load(target_cache, map_location="cpu", weights_only=False)
    master = torch.load(master_cache, map_location="cpu", weights_only=False)
    if tuple(target["seg"].shape) != (N_TOTAL_PAIRS, EVAL_H, EVAL_W):
        raise ValueError("target cache seg shape mismatch")
    if tuple(target["pose"].shape) != (N_TOTAL_PAIRS, 6):
        raise ValueError("target cache pose shape mismatch")
    expected_master = (N_TOTAL_PAIRS, 3, CAMERA_H, CAMERA_W)
    if tuple(master["masters"].shape) != expected_master or master["masters"].dtype != torch.uint8:
        raise ValueError("master cache tensor shape/dtype mismatch")
    if master.get("source_checkpoint") != str(master_checkpoint.resolve()):
        raise ValueError("master cache source_checkpoint guard mismatch")
    return {
        "schema": "ddm_mx2_pose_adapter_cache_validation.v1",
        "target_cache": str(target_cache),
        "target_cache_sha256": sha256_file(target_cache),
        "master_cache": str(master_cache),
        "master_cache_sha256": sha256_file(master_cache),
        "master_checkpoint": str(master_checkpoint.resolve()),
        "status": "PASS",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    target = sub.add_parser("target", help="build gt_pose_cache_600.pt")
    target.add_argument("--official-cache-xz", type=Path, default=DEFAULT_OFFICIAL_CACHE_XZ)
    target.add_argument("--reference-gt-npz", type=Path, default=DEFAULT_GT_REFERENCE_NPZ)
    target.add_argument("--out", type=Path, default=DEFAULT_TARGET_OUT)
    target.add_argument("--receipt", type=Path, default=SSD_ROOT / "inputs/gt_pose_cache_600.receipt.json")

    master = sub.add_parser("master", help="build OUR_SURFACE_MASTERS.pt from inflated raw")
    master.add_argument("--raw", type=Path, default=DEFAULT_PARENT_RAW)
    master.add_argument("--master-checkpoint", type=Path, default=DEFAULT_MASTER_CHECKPOINT)
    master.add_argument("--out", type=Path, default=DEFAULT_MASTER_OUT)
    master.add_argument("--receipt", type=Path, default=SSD_ROOT / "master_cache/OUR_SURFACE_MASTERS.receipt.json")
    master.add_argument("--chunk-pairs", type=int, default=8)

    all_cmd = sub.add_parser("all", help="build both caches and validation receipt")
    all_cmd.add_argument("--official-cache-xz", type=Path, default=DEFAULT_OFFICIAL_CACHE_XZ)
    all_cmd.add_argument("--reference-gt-npz", type=Path, default=DEFAULT_GT_REFERENCE_NPZ)
    all_cmd.add_argument("--target-out", type=Path, default=DEFAULT_TARGET_OUT)
    all_cmd.add_argument("--target-receipt", type=Path, default=SSD_ROOT / "inputs/gt_pose_cache_600.receipt.json")
    all_cmd.add_argument("--raw", type=Path, default=DEFAULT_PARENT_RAW)
    all_cmd.add_argument("--master-checkpoint", type=Path, default=DEFAULT_MASTER_CHECKPOINT)
    all_cmd.add_argument("--master-out", type=Path, default=DEFAULT_MASTER_OUT)
    all_cmd.add_argument("--master-receipt", type=Path, default=SSD_ROOT / "master_cache/OUR_SURFACE_MASTERS.receipt.json")
    all_cmd.add_argument("--validation-receipt", type=Path, default=SSD_ROOT / "inputs/cache_validation.receipt.json")
    all_cmd.add_argument("--chunk-pairs", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cmd == "target":
        print(json.dumps(build_target_cache(
            args.official_cache_xz, args.out, args.receipt, args.reference_gt_npz
        ), indent=2, sort_keys=True))
    elif args.cmd == "master":
        print(json.dumps(build_master_cache(
            args.raw, args.master_checkpoint, args.out, args.receipt,
            chunk_pairs=args.chunk_pairs,
        ), indent=2, sort_keys=True))
    elif args.cmd == "all":
        target_report = build_target_cache(
            args.official_cache_xz, args.target_out, args.target_receipt, args.reference_gt_npz
        )
        master_report = build_master_cache(
            args.raw, args.master_checkpoint, args.master_out, args.master_receipt,
            chunk_pairs=args.chunk_pairs,
        )
        validation = validate_caches(args.target_out, args.master_out, args.master_checkpoint)
        validation["target_receipt"] = str(args.target_receipt)
        validation["master_receipt"] = str(args.master_receipt)
        write_json(args.validation_receipt, validation)
        print(json.dumps({
            "schema": "ddm_mx2_pose_adapter_cache_build_all.v1",
            "target": target_report,
            "master": master_report,
            "validation": validation,
        }, indent=2, sort_keys=True))
    else:
        raise AssertionError(args.cmd)


if __name__ == "__main__":
    main()
