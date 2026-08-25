#!/usr/bin/env python3
"""Retained stratified-random n32 screen for the active JS8 MC36 candidate.

This is a TOY-BRACKET only.  It uses the exact frozen CPU-torch scorers and
the real MC36 semantic receiver state, but the active frame-1 edits are paired
with MC36's unmodified frame 0.  Therefore the PoseNet result is explicitly
pre-compensation and cannot admit or refuse the n600 instance.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from safetensors.torch import load_file

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_js8_implicit_edge_conditioning as build
from experiments.ddm_ec1_runtime import js8_edge_state_conditioner as js8_runtime

OUTPUT: Final = build.BULK_ROOT / "screen_v1"
BUILD_RESULT: Final = build.BULK_ROOT / "build_v1/BUILD_RESULT.json"
ACTIVE_GATE: Final = build.BULK_ROOT / "build_v1/active/retained/js8_edge_gate.br"
PER_PAIR: Final = build.JS1C_ROOT / "decomposition/cp135_base_per_pair.jsonl"
BASE_RAW: Final = Path("/Volumes/VertigoDataTier/pact/ddm_f26p_runtime_cpu_lift_20260814/output/0.raw")
BASE_RAW_SHA256: Final = "e5539653f598a1c31e28900888f450a6de019cb29864674f232ad2f8956b15c9"
UPSTREAM: Final = REPO / "upstream"
VIDEO: Final = UPSTREAM / "videos/0.mkv"
N: Final = 600
CAMERA_H: Final = 874
CAMERA_W: Final = 1164
EVAL_H: Final = 384
EVAL_W: Final = 512
RATE_DENOMINATOR: Final = 37_545_489
SEED: Final = build.SEED


class JS8ScreenError(RuntimeError):
    """A screen input, retention, or scorer invariant failed."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_npy(path: Path, value: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    atomic_bytes(path, buffer.getvalue())


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def select_pairs() -> tuple[np.ndarray, list[dict[str, Any]]]:
    rows = [json.loads(line) for line in PER_PAIR.read_text().splitlines() if line]
    if [int(row["pair"]) for row in rows] != list(range(N)):
        raise JS8ScreenError("per-pair decomposition population differs")
    ordered = np.argsort(np.asarray([int(row["total_flips"]) for row in rows]), kind="stable")
    rng = np.random.default_rng(SEED)
    selected = np.sort(
        np.concatenate([rng.choice(bucket, size=8, replace=False) for bucket in np.array_split(ordered, 4)])
    ).astype(np.int64)
    selection_rows = [rows[int(pair)] for pair in selected]
    return selected, selection_rows


def decode_gt(pair_ids: np.ndarray) -> np.ndarray:
    sys.path.insert(0, str(UPSTREAM))
    try:
        from frame_utils import AVVideoDataset
    finally:
        sys.path.pop(0)
    dataset = AVVideoDataset(
        [VIDEO.name],
        data_dir=VIDEO.parent,
        batch_size=8,
        device=torch.device("cpu"),
        num_threads=2,
        seed=SEED,
        prefetch_queue_depth=2,
    )
    wanted = {int(value) for value in pair_ids}
    found: dict[int, np.ndarray] = {}
    offset = 0
    for _, _, frames in dataset:
        for local in range(len(frames)):
            pair = offset + local
            if pair in wanted:
                found[pair] = frames[local].numpy().astype(np.uint8, copy=True)
        offset += len(frames)
    if offset != N or set(found) != wanted:
        raise JS8ScreenError(f"GT decode population differs: decoded={offset}, found={len(found)}")
    return np.stack([found[int(pair)] for pair in pair_ids])


def render_active(pair_ids: np.ndarray, ec1_blob: bytes, gate_blob: bytes) -> tuple[np.ndarray, np.ndarray]:
    from torch.nn import functional

    semantic = build.load_exact_semantic()
    tokens = np.load(build.TOKENS, mmap_mode="r", allow_pickle=False)
    pre_r = []
    camera = []
    with torch.inference_mode():
        for pair in pair_ids:
            token = torch.from_numpy(np.asarray(tokens[int(pair)]).copy())[None].long()
            value = js8_runtime.conditioned_semantic_forward(
                semantic,
                token,
                torch.tensor([int(pair)], dtype=torch.long),
                ec1_blob,
                gate_blob,
            )
            lifted = (
                functional.interpolate(value, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
                .clamp(0.0, 255.0)
                .round()
                .to(torch.uint8)
            )
            pre_r.append(value[0].cpu().numpy().astype(np.float32, copy=False))
            camera.append(lifted[0].permute(1, 2, 0).cpu().numpy())
    return np.stack(pre_r), np.stack(camera)


def load_scorers() -> tuple[Any, Any]:
    sys.path.insert(0, str(UPSTREAM))
    try:
        from modules import PoseNet, SegNet, posenet_sd_path, segnet_sd_path
    finally:
        sys.path.pop(0)
    segnet = SegNet().eval()
    posenet = PoseNet().eval()
    segnet.load_state_dict(load_file(segnet_sd_path, device="cpu"))
    posenet.load_state_dict(load_file(posenet_sd_path, device="cpu"))
    return segnet, posenet


def score_pairs(frames: np.ndarray, segnet: Any, posenet: Any) -> dict[str, np.ndarray]:
    value = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 1, 4, 2, 3).float()
    with torch.inference_mode():
        seg_input = segnet.preprocess_input(value)
        seg_logits = segnet(seg_input)
        pose_input = posenet.preprocess_input(value)
        pose = posenet(pose_input)["pose"][..., :6]
    return {
        "seg_input": seg_input.numpy().astype(np.float32, copy=False),
        "seg_logits": seg_logits.numpy().astype(np.float32, copy=False),
        "argmax": seg_logits.argmax(dim=1).numpy().astype(np.uint8, copy=False),
        "pose_input": pose_input.numpy().astype(np.float32, copy=False),
        "pose": pose.numpy().astype(np.float32, copy=False),
    }


def confusion(target: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    flat = target.astype(np.int64).ravel() * 5 + predicted.astype(np.int64).ravel()
    return np.bincount(flat, minlength=25).reshape(5, 5)


def main() -> int:
    if OUTPUT.exists():
        raise JS8ScreenError(f"immutable screen output already exists: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    if not BUILD_RESULT.is_file() or not ACTIVE_GATE.is_file():
        raise JS8ScreenError("JS8 admission build is incomplete")
    build_result = json.loads(BUILD_RESULT.read_text())
    if build_result.get("status") != "RECEIVER_CLOSED_ADMISSION_BUILT_SCORER_OWED":
        raise JS8ScreenError("JS8 build status differs")
    if BASE_RAW.stat().st_size != N * 2 * CAMERA_H * CAMERA_W * 3 or sha256_file(BASE_RAW) != BASE_RAW_SHA256:
        raise JS8ScreenError("retained MC36 CPU raw differs")
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)

    pair_ids, source_rows = select_pairs()
    gt_frames = decode_gt(pair_ids)
    raw = np.memmap(BASE_RAW, mode="r", dtype=np.uint8, shape=(N * 2, CAMERA_H, CAMERA_W, 3))
    base_frames = np.stack([np.asarray(raw[2 * int(pair) : 2 * int(pair) + 2]).copy() for pair in pair_ids])
    ec1_blob = build.EC2_MODULE.read_bytes()
    gate_blob = ACTIVE_GATE.read_bytes()
    active_pre_r, active_master = render_active(pair_ids, ec1_blob, gate_blob)
    active_frames = base_frames.copy()
    active_frames[:, 1] = active_master

    segnet, posenet = load_scorers()  # SCORER_LOADER_ORDER_OK: local wrapper (this file :152) returns (segnet, posenet); unpack matches its verified order
    scored = {
        "target": score_pairs(gt_frames, segnet, posenet),
        "base": score_pairs(base_frames, segnet, posenet),
        "active_uncompensated": score_pairs(active_frames, segnet, posenet),
    }
    retained = OUTPUT / "retained"
    arrays = {
        "pair_ids.int64.npy": pair_ids,
        "target_camera.uint8.npy": gt_frames,
        "base_camera.uint8.npy": base_frames,
        "active_pre_r.float32.npy": active_pre_r,
        "active_camera_uncompensated.uint8.npy": active_frames,
    }
    for role, values in scored.items():
        for name, value in values.items():
            arrays[f"{role}_{name}.npy"] = value
    records = {}
    for name, value in arrays.items():
        path = retained / name
        atomic_npy(path, np.asarray(value))
        records[name] = file_record(path)

    target_argmax = scored["target"]["argmax"]
    target_pose = scored["target"]["pose"]
    archive_bytes = int(build_result["payloads"]["active"]["archive"]["bytes"])
    rows = {}
    for role in ("base", "active_uncompensated"):
        predicted = scored[role]["argmax"]
        flips = int(np.count_nonzero(predicted != target_argmax))
        d_seg = flips / predicted.size
        d_pose = float(np.mean((scored[role]["pose"] - target_pose) ** 2))
        bytes_value = build.BASE_ARCHIVE_BYTES if role == "base" else archive_bytes
        rows[role] = {
            "flips": flips,
            "d_seg": d_seg,
            "d_pose": d_pose,
            "archive_bytes": bytes_value,
            "seg_term": 100.0 * d_seg,
            "pose_term": math.sqrt(10.0 * d_pose),
            "rate_term": 25.0 * bytes_value / RATE_DENOMINATOR,
            "sample_S": 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * bytes_value / RATE_DENOMINATOR,
            "confusion_gt_by_rendered": confusion(target_argmax, predicted).tolist(),
        }
    result = {
        "schema": "ddm_js8_stratified_screen.v1",
        "status": "TOY_BRACKET_COMPLETE_NO_VERDICT",
        "axis": "[macOS-CPU frozen-SegNet+PoseNet advisory, seeded stratified-random n32] TOY-BRACKET",
        "selection_mode": "seeded random 8 pairs from each quartile of JS1C cp135 per-pair flip count",
        "seed": SEED,
        "pair_ids": pair_ids.tolist(),
        "selection_source_rows": source_rows,
        "rows": rows,
        "delta_active_minus_base": {
            key: rows["active_uncompensated"][key] - rows["base"][key]
            for key in ("flips", "d_seg", "d_pose", "archive_bytes", "seg_term", "pose_term", "rate_term", "sample_S")
        },
        "payloads": records,
        "boundaries": {
            "full_n600_confirmation": False,
            "candidate_full_receiver_decode": False,
            "pose_compensation": False,
            "score_claim": False,
            "pointer_moved": False,
            "verdict_scope": "TOY-BRACKET only; no instance/formulation/family verdict",
        },
        "next_fire": build_result["next_fire"],
    }
    atomic_json(OUTPUT / "SCREEN_RESULT.json", result)
    atomic_json(build.LOGICAL_ROOT / "SCREEN_POINTER.json", {"result": file_record(OUTPUT / "SCREEN_RESULT.json")})
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
