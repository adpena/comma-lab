#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""CQ2 comma10k-only tiny-student sizing curve.

This is a scorer-free public-data arm. It trains small chart students only on
the locally custodied comma10k images and the locally custodied public
comma10k-segnet teacher, selects on comma10k-val teacher-chart metrics, then
runs the single allowed CQ1-style n32 overlap read for the frozen selected
student.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import subprocess
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import brotli
import cv2
import numpy as np
import scipy.ndimage as ndimage
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch import nn

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
import experiments.ddm_cq1_comma10k_chart_overlap as cq1


MODEL_DIR = Path("/Volumes/VertigoDataTier/pact/public_models/comma10k_segnet")
DATASET_DIR = Path("/Volumes/VertigoDataTier/pact/public_datasets/comma10k")
DEFAULT_OUT_DIR = REPO / ".omx/research/ddm_cq2_20260804"
DEFAULT_BULK_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_cq2_20260804")

EXPECTED_TEACHER_SHA256 = {
    "model.safetensors": "8208672861ad1b111dc98f3a7c54196d29875b709c7353e2dd1b7614343fb3a8",
    "config.json": "2b8f16dbad9bd85386609386a9cb5dedc6e0c518253a9af484e0a128d9463c88",
    "albumentations_config_eval.json": "d260853fe0a993e23613ff38039fdce59264f5fe31f729c1fa65f8c3e5fde913",
}
EXPECTED_CLONE_SHA = "6c205fe4c43cc53b2b1befafb1060d0606555027"
EXPECTED_IMG_COUNT = 9888
EXPECTED_MASK_COUNT = 9888

SEG_H = 384
SEG_W = 512
ROAD = 0
LANE = 1
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
CLASS_WEIGHT = (3.0, 6.0, 1.0, 1.0, 1.0)
SIZE_WIDTHS = {"25k": 20, "75k": 40, "150k": 56}
STREAM_BYTES = {
    "side_implied": 81_365,
    "explicit_direction": 100_904,
    "ed1_section_baseline": 169_149,
}
CAPTURED_FLIPS = 161_660
W_BYTES_PER_FLIP = 1.2731082153320312
OWN_FRONTIER_LINE = "S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]; contest pointer borrowed/unmoved."


def sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def atomic_torch_save(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def run_cmd(argv: list[str], cwd: Path | None = None) -> dict[str, Any]:
    proc = subprocess.run(argv, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "argv": argv,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def image_index_for_split(name: str) -> int:
    match = re.match(r"^[A-Za-z]?(\d+)", Path(name).stem)
    if match is None:
        raise ValueError(f"cannot parse comma10k split index from {name!r}")
    return int(match.group(1))


def comma10k_split(dataset_dir: Path) -> tuple[list[Path], list[Path]]:
    imgs = sorted((dataset_dir / "imgs").glob("*.png"))
    train: list[Path] = []
    val: list[Path] = []
    for path in imgs:
        if image_index_for_split(path.name) % 10 == 9:
            val.append(path)
        else:
            train.append(path)
    return train, val


def verify_custody(dataset_dir: Path, model_dir: Path) -> dict[str, Any]:
    manifest_path = dataset_dir / "CLONE_MANIFEST.json"
    if not manifest_path.exists():
        raise SystemExit(f"missing dataset clone manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    imgs = sorted((dataset_dir / "imgs").glob("*.png"))
    masks = sorted((dataset_dir / "masks").glob("*.png"))
    img_names = {p.name for p in imgs}
    mask_names = {p.name for p in masks}
    if len(imgs) != EXPECTED_IMG_COUNT or len(masks) != EXPECTED_MASK_COUNT:
        raise SystemExit(f"unexpected comma10k counts: imgs={len(imgs)} masks={len(masks)}")
    if img_names != mask_names:
        raise SystemExit(
            f"comma10k image/mask mismatch: img_only={len(img_names - mask_names)} mask_only={len(mask_names - img_names)}"
        )
    if manifest.get("sha") != EXPECTED_CLONE_SHA:
        raise SystemExit(f"comma10k manifest sha mismatch: {manifest.get('sha')} != {EXPECTED_CLONE_SHA}")
    if int(manifest.get("imgs", -1)) != EXPECTED_IMG_COUNT or int(manifest.get("masks", -1)) != EXPECTED_MASK_COUNT:
        raise SystemExit(f"comma10k manifest count mismatch: {manifest}")

    train, val = comma10k_split(dataset_dir)
    if len(train) != 8900 or len(val) != 988:
        raise SystemExit(f"unexpected split counts: train={len(train)} val={len(val)}")

    teacher_hashes: dict[str, Any] = {}
    for name, expected in EXPECTED_TEACHER_SHA256.items():
        path = model_dir / name
        size, digest = sha256_file(path)
        if digest != expected:
            raise SystemExit(f"teacher sha mismatch for {path}: got {digest}, expected {expected}")
        teacher_hashes[name] = {"bytes": size, "sha256": digest}

    git_head = run_cmd(["git", "rev-parse", "HEAD"], cwd=dataset_dir)
    git_remote = run_cmd(["git", "remote", "-v"], cwd=dataset_dir)
    lock_files = sorted(str(p.relative_to(dataset_dir)) for p in (dataset_dir / ".git").glob("*.lock"))
    top_dirs = sorted(p.name for p in dataset_dir.iterdir() if p.is_dir())
    return {
        "dataset_dir": str(dataset_dir),
        "clone_manifest": manifest,
        "manifest_path": str(manifest_path),
        "git_head": git_head,
        "git_remote": git_remote,
        "git_lock_files_maxdepth1": lock_files,
        "top_level_dirs": top_dirs,
        "image_count": len(imgs),
        "mask_count": len(masks),
        "matched_image_mask_names": True,
        "split_rule": "README conventional validation: parsed leading numeric id ending in 9; supports numeric, h###, r###, u### names",
        "train_count": len(train),
        "val_count": len(val),
        "teacher_dir": str(model_dir),
        "teacher_hashes": teacher_hashes,
        "class_order": {name: i for i, name in enumerate(CLASS_NAMES)},
    }


class SepConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1, groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.pointwise(self.depthwise(x)))


class TinyStudentUNet(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = int(width)
        w = self.width
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, padding=1), nn.ReLU(inplace=True))
        self.enc1 = SepConv(w, w)
        self.down1 = SepConv(w, 2 * w, stride=2)
        self.enc2 = SepConv(2 * w, 2 * w)
        self.down2 = SepConv(2 * w, 4 * w, stride=2)
        self.bot = SepConv(4 * w, 4 * w)
        self.up1 = SepConv(6 * w, 2 * w)
        self.up0 = SepConv(3 * w, w)
        self.head = nn.Conv2d(w, len(CLASS_NAMES), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(self.down1(e1))
        b = self.bot(self.down2(e2))
        u1 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        u1 = self.up1(torch.cat([u1, e2], dim=1))
        u0 = F.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        u0 = self.up0(torch.cat([u0, e1], dim=1))
        return self.head(u0)


def param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def read_rgb_resized(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"failed to read image {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(rgb, (SEG_W, SEG_H), interpolation=cv2.INTER_LINEAR)


def load_batch(paths: list[Path]) -> torch.Tensor:
    rows = [np.ascontiguousarray(read_rgb_resized(path).transpose(2, 0, 1)) for path in paths]
    return torch.from_numpy(np.stack(rows, axis=0)).float()


def teacher_targets(teacher: nn.Module, x: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        return teacher(x).argmax(dim=1).to(torch.long).clone()


def confusion_update(conf: np.ndarray, pred: np.ndarray, ref: np.ndarray) -> None:
    flat = (ref.reshape(-1).astype(np.int64) * len(CLASS_NAMES)) + pred.reshape(-1).astype(np.int64)
    conf += np.bincount(flat, minlength=len(CLASS_NAMES) * len(CLASS_NAMES)).reshape(len(CLASS_NAMES), len(CLASS_NAMES))


def metrics_from_confusion(conf: np.ndarray) -> dict[str, Any]:
    per_class: dict[str, Any] = {}
    for idx, name in enumerate(CLASS_NAMES):
        inter = int(conf[idx, idx])
        ref_count = int(conf[idx, :].sum())
        pred_count = int(conf[:, idx].sum())
        union = ref_count + pred_count - inter
        per_class[name] = {
            "class_id": idx,
            "intersection": inter,
            "teacher_pixels": ref_count,
            "student_pixels": pred_count,
            "union": union,
            "iou": float(inter / union) if union else 1.0,
        }
    road_iou = per_class["Road"]["iou"]
    lane_iou = per_class["Lane"]["iou"]
    return {
        "per_class": per_class,
        "road_iou": road_iou,
        "lane_iou": lane_iou,
        "road_lane_mean_iou": float((road_iou + lane_iou) / 2.0),
        "pixel_accuracy": float(np.trace(conf) / conf.sum()) if conf.sum() else 0.0,
        "confusion": conf.astype(np.int64).tolist(),
    }


def eval_student(
    student: nn.Module,
    teacher: nn.Module,
    paths: list[Path],
    *,
    batch_size: int,
    torch_threads: int,
) -> dict[str, Any]:
    del torch_threads
    student.eval()
    conf = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    total_loss = 0.0
    total_pixels = 0
    weight = torch.tensor(CLASS_WEIGHT, dtype=torch.float32)
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start : start + batch_size]
            x = load_batch(batch_paths)
            target = teacher_targets(teacher, x)
            logits = student(x)
            loss = F.cross_entropy(logits, target, weight=weight, reduction="sum")
            pred = logits.argmax(dim=1).to(torch.uint8).cpu().numpy()
            ref = target.to(torch.uint8).cpu().numpy()
            confusion_update(conf, pred, ref)
            total_loss += float(loss)
            total_pixels += int(np.prod(ref.shape))
    metrics = metrics_from_confusion(conf)
    metrics["weighted_ce_per_pixel"] = float(total_loss / total_pixels) if total_pixels else 0.0
    metrics["evaluated_images"] = len(paths)
    return metrics


def latest_checkpoint(run_dir: Path, label: str) -> Path | None:
    paths = sorted(run_dir.glob(f"{label}_checkpoint_step_*.pt"))
    return paths[-1] if paths else None


def save_checkpoint(
    path: Path,
    *,
    label: str,
    width: int,
    step: int,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    best_metric: float,
    best_step: int,
    history: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    atomic_torch_save(
        path,
        {
            "schema": "ddm_cq2_tiny_student_checkpoint.v1",
            "label": label,
            "width": width,
            "step": step,
            "model_state": student.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_metric": best_metric,
            "best_step": best_step,
            "history": history,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "batch_size": args.batch_size,
            "class_weight": CLASS_WEIGHT,
        },
    )


def tensor_payload(state_dict: dict[str, torch.Tensor], mode: str) -> bytes:
    entries: list[dict[str, Any]] = []
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(state_dict):
        arr = state_dict[name].detach().cpu().numpy()
        if mode == "int8":
            a = arr.astype(np.float32)
            max_abs = float(np.max(np.abs(a))) if a.size else 0.0
            scale = max_abs / 127.0 if max_abs > 0.0 else 1.0
            q = np.clip(np.round(a / scale), -127, 127).astype(np.int8)
            raw = q.tobytes(order="C")
            entry = {"name": name, "shape": list(a.shape), "dtype": "int8", "scale": scale, "offset": offset, "nbytes": len(raw)}
        elif mode == "fp16":
            q = arr.astype(np.float16)
            raw = q.tobytes(order="C")
            entry = {"name": name, "shape": list(q.shape), "dtype": "fp16", "scale": 1.0, "offset": offset, "nbytes": len(raw)}
        else:
            raise ValueError(mode)
        entries.append(entry)
        chunks.append(raw)
        offset += len(raw)
    header = {
        "schema": "ddm_cq2_tiny_student_payload.v1",
        "mode": mode,
        "entries": entries,
        "payload_nbytes": offset,
    }
    header_raw = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return b"CQ2STU1\n" + len(header_raw).to_bytes(8, "little") + header_raw + b"".join(chunks)


def decode_tensor_payload(raw: bytes) -> dict[str, torch.Tensor]:
    if not raw.startswith(b"CQ2STU1\n"):
        raise ValueError("bad tiny-student payload magic")
    header_len = int.from_bytes(raw[8:16], "little")
    header = json.loads(raw[16 : 16 + header_len].decode("utf-8"))
    payload = raw[16 + header_len :]
    out: dict[str, torch.Tensor] = {}
    for entry in header["entries"]:
        start = int(entry["offset"])
        end = start + int(entry["nbytes"])
        shape = tuple(int(v) for v in entry["shape"])
        if entry["dtype"] == "int8":
            arr = np.frombuffer(payload[start:end], dtype=np.int8).copy().reshape(shape).astype(np.float32)
            arr *= np.float32(entry["scale"])
        elif entry["dtype"] == "fp16":
            arr = np.frombuffer(payload[start:end], dtype=np.float16).copy().reshape(shape).astype(np.float32)
        else:
            raise ValueError(f"unsupported payload dtype {entry['dtype']}")
        out[entry["name"]] = torch.from_numpy(arr)
    return out


def compress_state(state_dict: dict[str, torch.Tensor], out_path: Path) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for mode in ("int8", "fp16"):
        raw = tensor_payload(state_dict, mode)
        compressed = {
            "brotli_q11": brotli.compress(raw, quality=11),
            "zlib_9": zlib.compress(raw, level=9),
        }
        for codec, data in compressed.items():
            candidates.append(
                {
                    "mode": mode,
                    "codec": codec,
                    "raw_bytes": len(raw),
                    "compressed_bytes": len(data),
                    "sha256": sha256_bytes(data),
                    "data": data,
                }
            )
    best = min(candidates, key=lambda item: item["compressed_bytes"])
    atomic_write_bytes(out_path, best["data"])
    return {
        "path": str(out_path),
        "bytes": int(best["compressed_bytes"]),
        "sha256": best["sha256"],
        "mode": best["mode"],
        "codec": best["codec"],
        "raw_bytes": int(best["raw_bytes"]),
        "all_candidates": [
            {k: v for k, v in row.items() if k != "data"}
            for row in sorted(candidates, key=lambda item: (item["mode"], item["codec"]))
        ],
    }


def load_compressed_state(path: Path, codec: str) -> dict[str, torch.Tensor]:
    data = path.read_bytes()
    if codec == "brotli_q11":
        raw = brotli.decompress(data)
    elif codec == "zlib_9":
        raw = zlib.decompress(data)
    else:
        raise ValueError(codec)
    return decode_tensor_payload(raw)


def train_one_size(
    *,
    label: str,
    width: int,
    teacher: nn.Module,
    train_paths: list[Path],
    val_paths: list[Path],
    interim_val_paths: list[Path],
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    print(json.dumps({"event": "size_start", "label": label, "width": width}, sort_keys=True), flush=True)
    student = TinyStudentUNet(width)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_step = 0
    best_metric = -1.0
    best_step = 0
    history: list[dict[str, Any]] = []
    resumed_from: str | None = None

    if args.resume:
        ckpt_path = latest_checkpoint(run_dir, label)
        if ckpt_path is not None:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            student.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            start_step = int(ckpt["step"])
            best_metric = float(ckpt.get("best_metric", -1.0))
            best_step = int(ckpt.get("best_step", 0))
            history = list(ckpt.get("history", []))
            resumed_from = str(ckpt_path)

    weight = torch.tensor(CLASS_WEIGHT, dtype=torch.float32)
    train_loss_rows: list[dict[str, Any]] = []
    no_improve_evals = 0
    stop_reason = "safety_bound_REPORTED_max_steps"
    started = time.time()

    for step in range(start_step + 1, args.max_steps + 1):
        rng = random.Random(args.seed * 1_000_003 + step * 101 + width)
        batch_paths = rng.sample(train_paths, k=min(args.batch_size, len(train_paths)))
        x = load_batch(batch_paths)
        target = teacher_targets(teacher, x)
        student.train()
        optimizer.zero_grad(set_to_none=True)
        logits = student(x)
        loss = F.cross_entropy(logits, target, weight=weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
        optimizer.step()
        loss_value = float(loss.detach())
        train_loss_rows.append({"step": step, "weighted_ce": loss_value})

        if step % args.eval_every == 0 or step == args.max_steps:
            interim = eval_student(student, teacher, interim_val_paths, batch_size=args.batch_size, torch_threads=args.torch_threads)
            metric = float(interim["road_lane_mean_iou"])
            row = {"step": step, "interim_val": interim, "train_weighted_ce": loss_value}
            history.append(row)
            print(
                json.dumps(
                    {
                        "event": "eval",
                        "label": label,
                        "step": step,
                        "road_iou": interim["road_iou"],
                        "lane_iou": interim["lane_iou"],
                        "mean_iou": interim["road_lane_mean_iou"],
                        "weighted_ce": interim["weighted_ce_per_pixel"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if metric > best_metric + args.early_stop_min_delta:
                best_metric = metric
                best_step = step
                no_improve_evals = 0
            else:
                no_improve_evals += 1
            save_checkpoint(
                run_dir / f"{label}_checkpoint_step_{step:06d}.pt",
                label=label,
                width=width,
                step=step,
                student=student,
                optimizer=optimizer,
                best_metric=best_metric,
                best_step=best_step,
                history=history,
                args=args,
            )
            if no_improve_evals >= args.early_stop_patience:
                stop_reason = "convergence_plateau_REPORTED_interim_val_no_improvement"
                break

    final_step = history[-1]["step"] if history else start_step
    save_checkpoint(
        run_dir / f"{label}_checkpoint_final_step_{final_step:06d}.pt",
        label=label,
        width=width,
        step=final_step,
        student=student,
        optimizer=optimizer,
        best_metric=best_metric,
        best_step=best_step,
        history=history,
        args=args,
    )

    compressed_path = run_dir / f"{label}_student_weights.bin"
    compression = compress_state(student.state_dict(), compressed_path)
    quant_student = TinyStudentUNet(width)
    quant_student.load_state_dict(load_compressed_state(compressed_path, compression["codec"]), strict=True)
    quant_student.eval()
    final_val = eval_student(quant_student, teacher, val_paths, batch_size=args.batch_size, torch_threads=args.torch_threads)
    passes_bar = (
        final_val["road_iou"] >= args.bar_road_iou
        and final_val["lane_iou"] >= args.bar_lane_iou
        and final_val["road_lane_mean_iou"] >= args.bar_mean_iou
    )
    return {
        "label": label,
        "width": width,
        "param_count": param_count(student),
        "resumed_from": resumed_from,
        "steps_completed": final_step,
        "stop_reason": stop_reason,
        "wall_seconds": time.time() - started,
        "train_loss_tail": train_loss_rows[-10:],
        "interim_history": history,
        "best_interim_metric": best_metric,
        "best_interim_step": best_step,
        "compression": compression,
        "final_val": final_val,
        "passes_pre_registered_bar": passes_bar,
        "checkpoint_dir": str(run_dir),
        "final_checkpoint": str(run_dir / f"{label}_checkpoint_final_step_{final_step:06d}.pt"),
    }


def select_paths(paths: list[Path], limit: int, seed: int) -> list[Path]:
    if limit <= 0 or limit >= len(paths):
        return list(paths)
    rng = random.Random(seed)
    return sorted(rng.sample(paths, k=limit), key=lambda p: p.name)


def band_for(frame_argmax: np.ndarray, radius: int) -> np.ndarray:
    st3 = ndimage.generate_binary_structure(2, 2)
    road = frame_argmax == ROAD
    lane = frame_argmax == LANE
    return ndimage.binary_dilation(road, st3, radius) & ndimage.binary_dilation(lane, st3, radius)


def run_cq1_overlap_for_student(
    *,
    selected: dict[str, Any],
    bulk_dir: Path,
    out_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    width = int(selected["width"])
    compression = selected["compression"]
    state = load_compressed_state(Path(compression["path"]), compression["codec"])
    student = TinyStudentUNet(width)
    student.load_state_dict(state, strict=True)
    student.eval()

    frames = cq1.load_raw_frames(cq1.RAW_PATH)
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(cq1.PAIRS), batch_size):
            batch_pairs = cq1.PAIRS[start : start + batch_size]
            tensors = []
            for pair in batch_pairs:
                frame = np.asarray(frames[pair * cq1.FRAMES_PER_PAIR + 1])
                resized = cv2.resize(frame, (SEG_W, SEG_H), interpolation=cv2.INTER_LINEAR)
                tensors.append(torch.from_numpy(np.ascontiguousarray(resized.transpose(2, 0, 1))).float())
            logits = student(torch.stack(tensors, dim=0))
            rows.append(logits.argmax(dim=1).to(torch.uint8).cpu().numpy())
    pred = np.concatenate(rows, axis=0)
    pred_path = bulk_dir / "cq2_selected_student_argmax_pairs_n32.npy"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(pred_path, pred)
    pred_bytes, pred_sha = sha256_file(pred_path)

    cx1 = np.load(cq1.CX1_ARGMAX, mmap_mode="r")[list(cq1.PAIRS)]
    gt = np.load(cq1.GT_ARGMAX, mmap_mode="r")[list(cq1.PAIRS)]
    target = (gt != cx1) & (((gt == ROAD) & (cx1 == LANE)) | ((gt == LANE) & (cx1 == ROAD)))
    total_targets = int(np.count_nonzero(target))
    band_rows = [cq1.band_stats(pred, np.asarray(cx1), target, radius) for radius in (1, 2, 3)]
    decisive = next(row for row in band_rows if row["radius"] == 1)
    overlap_fraction = float(decisive["micro_over_se3_captured_fraction"])
    verdict = "GOOD-OVERLAP" if overlap_fraction >= cq1.GOOD_OVERLAP_THRESHOLD else "POOR-OVERLAP"

    pair_rows_path = out_dir / "cq2_selected_overlap_pair_rows.jsonl"
    with pair_rows_path.open("w") as handle:
        by_radius = {row["radius"]: row["pair_rows"] for row in band_rows}
        for idx, pair in enumerate(cq1.PAIRS):
            row = {
                "pair": pair,
                "target_flips": int(np.count_nonzero(target[idx])),
                "band_rows": [by_radius[radius][idx] for radius in (1, 2, 3)],
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    return {
        "schema": "ddm_cq2_selected_student_cq1_overlap.v1",
        "selected_label": selected["label"],
        "selected_width": width,
        "axis": "[macOS-CPU advisory / public-data tiny-student chart-overlap scorer-free]",
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
        "pairs": list(cq1.PAIRS),
        "artifacts": {
            "argmax_subset": {"path": str(pred_path), "bytes": pred_bytes, "sha256": pred_sha, "shape": list(pred.shape)},
            "pair_rows_jsonl": str(pair_rows_path),
        },
        "denominators": {
            "selected_pairs": len(cq1.PAIRS),
            "subset_scorer_cells": int(np.prod(pred.shape)),
            "subset_road_lane_target_flips": total_targets,
        },
        "iou_vs_cx1": cq1.per_class_iou(pred, np.asarray(cx1)),
        "iou_vs_gt": cq1.per_class_iou(pred, np.asarray(gt)),
        "band_overlap_vs_se3_cx1_chart": [{key: value for key, value in row.items() if key != "pair_rows"} for row in band_rows],
        "decisive_metric": {
            "threshold": cq1.GOOD_OVERLAP_THRESHOLD,
            "measured_student_overlap_fraction": overlap_fraction,
            "denominator_se3_r1_captured_flips": decisive["se3_captured_flips"],
            "numerator_student_overlap_of_se3_r1_captured_flips": decisive["micro_overlap_of_se3_captured_flips"],
        },
        "verdict": verdict,
    }


def economics_rows(student_bytes: int) -> list[dict[str, Any]]:
    out = []
    for row, stream_bytes in STREAM_BYTES.items():
        total = int(student_bytes + stream_bytes)
        break_even = float(total / (W_BYTES_PER_FLIP * CAPTURED_FLIPS))
        out.append(
            {
                "row": row,
                "student_bytes": int(student_bytes),
                "stream_bytes": int(stream_bytes),
                "total_bytes": total,
                "break_even_survival": break_even,
                "ed1_baseline_bytes": 169_149,
                "W_bytes_per_flip": W_BYTES_PER_FLIP,
                "captured_flips": CAPTURED_FLIPS,
            }
        )
    return out


def write_receipts(summary: dict[str, Any], out_dir: Path) -> None:
    atomic_write_text(out_dir / "cq2_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    rows = summary["size_results"]
    selected = summary["selection"]["selected_result"]
    overlap = summary.get("overlap")

    size_table = [
        "| size | width | params | counted B | Road IoU | Lane IoU | mean | pass | stop |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        size_table.append(
            "| {label} | {width} | {params:,} | {bytes:,} | {road:.6f} | {lane:.6f} | {mean:.6f} | {passed} | {stop} |".format(
                label=row["label"],
                width=row["width"],
                params=row["param_count"],
                bytes=row["compression"]["bytes"],
                road=row["final_val"]["road_iou"],
                lane=row["final_val"]["lane_iou"],
                mean=row["final_val"]["road_lane_mean_iou"],
                passed=str(row["passes_pre_registered_bar"]),
                stop=row["stop_reason"],
            )
        )

    econ_table = [
        "| stream row | student B | stream B | total B | break-even survival |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary["economics"]:
        econ_table.append(
            f"| {row['row']} | {row['student_bytes']:,} | {row['stream_bytes']:,} | {row['total_bytes']:,} | {row['break_even_survival']:.6f} |"
        )

    overlap_lines = ["Not run."]
    if overlap is not None:
        decisive = overlap["decisive_metric"]
        overlap_lines = [
            f"Selected-student CQ1 overlap verdict: **{overlap['verdict']}**.",
            "",
            "| metric | value |",
            "|---|---:|",
            f"| SE3 r1 captured flips | `{decisive['denominator_se3_r1_captured_flips']}` |",
            f"| selected-student overlap numerator | `{decisive['numerator_student_overlap_of_se3_r1_captured_flips']}` |",
            f"| selected-student overlap fraction | `{decisive['measured_student_overlap_fraction']:.10f}` |",
            f"| GOOD threshold | `{decisive['threshold']}` |",
        ]

    receipt = f"""# CQ2 comma10k-only tiny-student sizing receipt - 2026-08-05

Status: **{summary['selection']['status']}**.

Axis: `[macOS-CPU advisory / public-data tiny-student chart sizing / scorer-free]`.
`score_claim=false`, `promotion_eligible=false`, `n600_scorer_job=false`.

Own-vehicle baseline from hot state: `{OWN_FRONTIER_LINE}`

## Answer First

The dataset blocker from `102d1b4fda` is resolved: comma10k is complete at sha
`{summary['custody']['clone_manifest']['sha']}` with `{summary['custody']['image_count']}` imgs
and `{summary['custody']['mask_count']}` masks. CQ2 trained the requested three public-data-only
student sizes and selected `{selected['label']}` by the pre-registered comma10k-val rule.

## RECALL EVIDENCE

- Charter seeds read: `.omx/research/ddm_cq1_20260804/{{cq1_receipt.md,NEXT_IF_RESUMED.md}}` and `.omx/research/ddm_se3_20260804/se3_receipt.md`.
- Current resume receipt read: `.omx/research/ddm_cq2_20260804/{{cq2_receipt.md,NEXT_IF_RESUMED.md}}`; it changed the plan from blocker-only to dataset re-preflight plus training after the 2026-08-05 manifest.
- Corpus search beyond seeds found `.omx/research/ddm_rf1_20260804/RF1_RECEIPT_20260804.md`: qo1 has no legal receiver class chart, so the student remains the legal chart-source fallback and the SE3 81KB/101KB rows stay assumption-scoped until receiver closure.
- Corpus search beyond seeds found `.omx/research/ddm_nb1_20260804/nb1_receipt.md`: CQ1 GOOD-overlap and SE3 stream prices stand at their written scopes; n32 comparability is allowed but not a population verdict.
- Corpus search beyond seeds found `.omx/research/ddm_bf1_20260805/BF1_RECEIPT_20260805.md` and the 2026-08-05 per-edge directive: BF1 settled a receiver-closed lane-crop price, while cq2's 75KB lane remains relevant only if a low-dim/per-edge description can ride this public student.
- Corpus search beyond seeds found `.omx/research/distillation_smaller_student_20260610T191237Z.md`: older contest-frame student distillation was training-stability-limited and non-monotone; CQ2 therefore records explicit stop reasons and selects only by public comma10k-val metrics.

## Custody

| item | value |
|---|---:|
| dataset path | `{summary['custody']['dataset_dir']}` |
| clone manifest sha | `{summary['custody']['clone_manifest']['sha']}` |
| manifest imgs / masks | `{summary['custody']['clone_manifest']['imgs']} / {summary['custody']['clone_manifest']['masks']}` |
| actual imgs / masks | `{summary['custody']['image_count']} / {summary['custody']['mask_count']}` |
| git HEAD | `{summary['custody']['git_head']['stdout']}` |
| split | `{summary['custody']['train_count']} train / {summary['custody']['val_count']} val` |
| teacher model sha | `{summary['custody']['teacher_hashes']['model.safetensors']['sha256']}` |
| eval config sha | `{summary['custody']['teacher_hashes']['albumentations_config_eval.json']['sha256']}` |

## Pre-Registered Selection Rule

Before any CQ1 overlap read, choose the smallest measured counted-byte student whose public comma10k-val
teacher-chart metrics satisfy: Road IoU >= `{summary['selection']['bar']['road_iou']}`, Lane IoU >=
`{summary['selection']['bar']['lane_iou']}`, and Road/Lane mean IoU >= `{summary['selection']['bar']['mean_iou']}`.
If none pass, choose the best Road/Lane mean IoU candidate as a diagnostic non-passing frozen student;
that overlap read cannot promote the route.

## Size Curve

{chr(10).join(size_table)}

Compression is the smallest measured real payload among int8/fp16 x Brotli q11/zlib9, written as a
decodeable tensor package and reloaded before final validation/overlap.

## Selected Student

| field | value |
|---|---:|
| selected label | `{selected['label']}` |
| counted bytes | `{selected['compression']['bytes']}` |
| payload path | `{selected['compression']['path']}` |
| payload sha256 | `{selected['compression']['sha256']}` |
| selected status | `{summary['selection']['status']}` |

## Final CQ1 Overlap Read

{chr(10).join(overlap_lines)}

## Economics

{chr(10).join(econ_table)}

Live realizer context: se2's paint ceiling remains `0.263-0.407`, which only clears rows whose
break-even survival is below that band; sq2's solved-field eta remains the live realization candidate.
The composition verdict is MAIN's after cq2 and sq2 are both consumed.

## Boundaries

- Training and selection used only comma10k public images plus the public teacher.
- No contest SegNet/PoseNet forward was run.
- No `upstream/evaluate.py` run was performed.
- No `archive.zip` was built.
- The final overlap read used the frozen selected payload after public-val selection; it was not used to choose among candidates.
- All bulk artifacts are under `{summary['bulk_dir']}`; no `/tmp` evidence is cited.

Own-vehicle frontier line: `{OWN_FRONTIER_LINE}`
"""
    atomic_write_text(out_dir / "cq2_receipt.md", receipt)

    next_text = f"""# CQ2 NEXT-IF-RESUMED

CQ2 now has a measured public-data tiny-student curve at comma10k sha
`{summary['custody']['clone_manifest']['sha']}`.

Selected student: `{selected['label']}`, counted bytes `{selected['compression']['bytes']}`,
payload `{selected['compression']['path']}`.

Next fire order:

1. Do not reselect using CQ1 overlap. Selection is already frozen by comma10k-val metrics.
2. If MAIN consumes this row, compare the measured overlap/economics against sq2's solved-field eta
   and the se2 `0.263-0.407` paint ceiling.
3. Receiver/archive work remains queued behind a legal receiver-consumed chart source and scorer-slot
   discipline. No full-n600 scorer job belongs to cq2.

Boundaries: scorer-free, no archive score claim, no `/tmp` evidence.

Own-vehicle frontier line: `{OWN_FRONTIER_LINE}`
"""
    atomic_write_text(out_dir / "NEXT_IF_RESUMED.md", next_text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DATASET_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--bulk-dir", type=Path, default=DEFAULT_BULK_DIR)
    parser.add_argument("--run-id", default="cq2r_20260805")
    parser.add_argument("--sizes", nargs="+", default=["25k", "75k", "150k"], choices=sorted(SIZE_WIDTHS))
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--eval-every", type=int, default=40)
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--train-limit", type=int, default=0)
    parser.add_argument("--val-limit", type=int, default=0)
    parser.add_argument("--interim-val-limit", type=int, default=96)
    parser.add_argument("--bar-road-iou", type=float, default=0.90)
    parser.add_argument("--bar-lane-iou", type=float, default=0.50)
    parser.add_argument("--bar-mean-iou", type=float, default=0.72)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--skip-overlap", action="store_true")
    args = parser.parse_args()

    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.bulk_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.bulk_dir / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    custody = verify_custody(args.dataset_dir, args.model_dir)
    train_paths_all, val_paths_all = comma10k_split(args.dataset_dir)
    train_paths = select_paths(train_paths_all, args.train_limit, args.seed + 11)
    val_paths = select_paths(val_paths_all, args.val_limit, args.seed + 23)
    interim_val_paths = select_paths(val_paths_all, args.interim_val_limit, args.seed + 37)
    if args.preflight_only:
        print(json.dumps({"preflight": "ok", "custody": custody}, indent=2, sort_keys=True))
        return

    teacher = smp.from_pretrained(str(args.model_dir))
    teacher.to("cpu")
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)

    size_results: list[dict[str, Any]] = []
    for label in args.sizes:
        result = train_one_size(
            label=label,
            width=SIZE_WIDTHS[label],
            teacher=teacher,
            train_paths=train_paths,
            val_paths=val_paths,
            interim_val_paths=interim_val_paths,
            run_dir=run_dir,
            args=args,
        )
        size_results.append(result)

    passing = [row for row in size_results if row["passes_pre_registered_bar"]]
    if passing:
        selected = min(passing, key=lambda row: row["compression"]["bytes"])
        selection_status = "PASS_SELECTED_SMALLEST_PUBLIC_VAL_BAR"
    else:
        selected = max(size_results, key=lambda row: row["final_val"]["road_lane_mean_iou"])
        selection_status = "NO_PASS_SELECTED_BEST_PUBLIC_VAL_DIAGNOSTIC"

    overlap = None
    if not args.skip_overlap:
        overlap = run_cq1_overlap_for_student(
            selected=selected,
            bulk_dir=run_dir,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
        )

    summary = {
        "schema": "ddm_cq2_comma10k_tiny_student_sizing.v1",
        "axis": "[macOS-CPU advisory / public-data tiny-student chart sizing / scorer-free]",
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
        "script": str(Path(__file__).relative_to(REPO)),
        "script_sha256": sha256_file(Path(__file__))[1],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "smp": smp.__version__,
            "opencv": cv2.__version__,
            "brotli": getattr(brotli, "__version__", "unknown"),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        },
        "custody": custody,
        "bulk_dir": str(args.bulk_dir),
        "run_dir": str(run_dir),
        "training": {
            "seed": args.seed,
            "train_images_used": len(train_paths),
            "train_images_available": len(train_paths_all),
            "val_images_used": len(val_paths),
            "val_images_available": len(val_paths_all),
            "interim_val_images_used": len(interim_val_paths),
            "batch_size": args.batch_size,
            "max_steps": args.max_steps,
            "eval_every": args.eval_every,
            "class_weight": CLASS_WEIGHT,
            "loss": "weighted hard-label cross entropy against public teacher argmax; Road/Lane weighted",
            "preprocessing": "teacher eval config equivalent: RGB resize to 384x512 linear, HWC uint8 to CHW float32, no normalization for teacher; student divides by 255 internally",
            "candidate_selection_signal": "comma10k-val public teacher chart metrics only",
        },
        "selection": {
            "status": selection_status,
            "bar": {"road_iou": args.bar_road_iou, "lane_iou": args.bar_lane_iou, "mean_iou": args.bar_mean_iou},
            "selected_label": selected["label"],
            "selected_result": selected,
        },
        "size_results": size_results,
        "overlap": overlap,
        "economics": economics_rows(int(selected["compression"]["bytes"])),
        "boundaries": [
            "Training and selection used only comma10k public images plus the public teacher.",
            "No contest SegNet/PoseNet forward was run.",
            "No upstream/evaluate.py run was performed.",
            "No archive.zip was built.",
            "Final overlap, if present, was a single frozen-candidate CQ1-style read after public-val selection.",
            "No /tmp evidence.",
        ],
        "own_vehicle_frontier_line": OWN_FRONTIER_LINE,
    }
    write_receipts(summary, args.out_dir)
    print(
        json.dumps(
            {
                "summary": str(args.out_dir / "cq2_summary.json"),
                "receipt": str(args.out_dir / "cq2_receipt.md"),
                "selection_status": selection_status,
                "selected": selected["label"],
                "selected_bytes": selected["compression"]["bytes"],
                "overlap": None if overlap is None else overlap["decisive_metric"]["measured_student_overlap_fraction"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
