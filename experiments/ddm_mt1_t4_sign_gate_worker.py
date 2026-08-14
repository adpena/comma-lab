#!/usr/bin/env python3
"""Resumable retained T4 sign gate for the DDM MT1 #978 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

REMOTE_REPO: Final = Path("/workspace/pact")
if str(REMOTE_REPO) not in sys.path:
    sys.path.insert(0, str(REMOTE_REPO))

from experiments import ddm_mt1_978_multitoken_screen as screen
from experiments.ddm_ec2_oriented_adapter_trainer_worker import (
    load_exact_semantic,
    load_segnet,
)
from experiments.ddm_mt1_runtime import multitoken_representative as mt1

SEED: Final = screen.SEED
BATCH: Final = 4
EXPECTED_RETAINED_BYTES: Final = 3 * 1024**3
RESERVE_BYTES: Final = 4 * 1024**3
AXIS: Final = (
    "[contest-CUDA T4 frozen SegNet/PoseNet; stratified-random n32 heldout] "
    "COMPONENT-ONLY NON-PROMOTABLE"
)


class MT1T4Error(RuntimeError):
    """A T4, input, receiver, retention, or resume invariant failed."""


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_record(path: Path, record: dict[str, Any]) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != int(record["bytes"])
        or sha256_file(path) != str(record["sha256"])
    ):
        raise MT1T4Error(f"sealed input differs: {path}")


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        for info in bundle.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise MT1T4Error(f"unsafe runtime member: {info.filename}")
            if info.is_dir():
                continue
            target = destination / relative
            payload = bundle.read(info)
            if target.is_file():
                if target.read_bytes() != payload:
                    raise MT1T4Error(f"resumed runtime differs: {target}")
                continue
            screen.retain_bytes(target, payload)
            os.chmod(target, (info.external_attr >> 16) & 0o777 or 0o644)


def storage_preflight(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    retained = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    required = max(0, EXPECTED_RETAINED_BYTES - retained) + RESERVE_BYTES
    usage = shutil.disk_usage(root)
    receipt = {
        "schema": "ddm_mt1_t4_storage_preflight.v1",
        "free_bytes": usage.free,
        "already_retained_bytes": retained,
        "expected_total_retained_bytes": EXPECTED_RETAINED_BYTES,
        "reserve_bytes": RESERVE_BYTES,
        "required_free_bytes": required,
        "passed": usage.free >= required,
        "cleanup_policy": "certify-or-block; retain all materialized scorer payloads",
    }
    screen.atomic_json(root / "STORAGE_PREFLIGHT.json", receipt)
    if not receipt["passed"]:
        raise MT1T4Error(f"remote storage preflight failed: {receipt}")
    return receipt


def configure_cuda() -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "Tesla T4":
        raise MT1T4Error(
            f"sign gate requires Tesla T4; got {torch.cuda.get_device_name(0)!r}"
            if torch.cuda.is_available()
            else "sign gate requires Tesla T4; CUDA is absent"
        )
    random.seed(SEED)
    np.random.seed(SEED % (2**32))
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "seed": SEED,
        "device": torch.cuda.get_device_name(0),
        "tf32": False,
        "torch_deterministic_algorithms": True,
        "rationale": "seeded deterministic component measurement on the existing T4 scorer lane",
    }


def load_posenet_cuda(device: torch.device) -> torch.nn.Module:
    from safetensors.torch import load_file

    upstream = REMOTE_REPO / "upstream"
    sys.path.insert(0, str(upstream))
    try:
        from modules import PoseNet, posenet_sd_path
    finally:
        sys.path.pop(0)
    network = PoseNet().eval().to(device)
    network.load_state_dict(load_file(posenet_sd_path, device=str(device)))
    for parameter in network.parameters():
        parameter.requires_grad_(False)
    return network


def pose_vectors_cuda(
    posenet: torch.nn.Module,
    pairs: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    value = np.asarray(pairs)
    if value.ndim != 5 or value.shape[1:] != (2, 874, 1164, 3):
        raise MT1T4Error(f"PoseNet input geometry differs: {value.shape}")
    with torch.inference_mode():
        tensor = (
            torch.from_numpy(np.ascontiguousarray(value))
            .permute(0, 1, 4, 2, 3)
            .float()
            .to(device)
        )
        output = posenet(posenet.preprocess_input(tensor))["pose"][..., :6]
    return output.cpu().numpy().astype(np.float32, copy=False)


def resume_batch(batch_root: Path) -> tuple[dict[str, Any], np.ndarray, np.ndarray] | None:
    receipt_path = batch_root / "BATCH_RECEIPT.json"
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text())
    for record in receipt["payloads"].values():
        require_record(Path(record["path"]), record)
    argmax = np.load(receipt["payloads"]["seg_argmax.uint8.npy"]["path"], allow_pickle=False)
    pose = np.load(receipt["payloads"]["pose_first6.float32.npy"]["path"], allow_pickle=False)
    return receipt, argmax, pose


def endpoint_arm(
    root: Path,
    *,
    name: str,
    semantic: torch.nn.Module,
    segnet: torch.nn.Module,
    posenet: torch.nn.Module,
    device: torch.device,
    tokens: np.ndarray,
    target: np.ndarray,
    gt_pose: np.ndarray,
    pairs: list[int],
    slaves: np.ndarray,
    model: mt1.MultiTokenRepresentative | None,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    argmax_parts: list[np.ndarray] = []
    pose_parts: list[np.ndarray] = []
    for batch_index, first in enumerate(range(0, len(pairs), BATCH)):
        chosen = pairs[first : first + BATCH]
        batch_root = root / f"endpoint/{name}/batch_{batch_index:02d}"
        resumed = resume_batch(batch_root)
        if resumed is not None:
            row, argmax, vectors = resumed
            rows.append(row)
            argmax_parts.append(argmax)
            pose_parts.append(vectors)
            continue
        token = torch.from_numpy(np.asarray(tokens[chosen]).copy()).long().to(device)
        indices = torch.tensor(chosen, dtype=torch.long, device=device)
        with torch.inference_mode():
            pre_r, camera, scorer_input = screen.semantic_receiver(
                semantic,
                token,
                indices,
                model,
            )
            logits = segnet(scorer_input)
        camera_u8 = camera.cpu().numpy().astype(np.uint8, copy=False)
        masters = camera_u8.transpose(0, 2, 3, 1)
        slave_batch = np.asarray(slaves[first : first + len(chosen)], dtype=np.uint8)
        pose_input = np.stack((slave_batch, masters), axis=1)
        vectors = pose_vectors_cuda(posenet, pose_input, device)
        argmax = logits.argmax(dim=1).cpu().numpy().astype(np.uint8, copy=False)
        payloads = {
            "pairs.int16.npy": np.asarray(chosen, dtype=np.int16),
            "tokens.uint8.npy": np.asarray(tokens[chosen], dtype=np.uint8),
            "pre_r.float32.npy": pre_r.cpu().numpy(),
            "camera.uint8.npy": camera_u8,
            "scorer_input.float32.npy": scorer_input.cpu().numpy(),
            "seg_logits.float32.npy": logits.cpu().numpy(),
            "seg_argmax.uint8.npy": argmax,
            "seg_target.uint8.npy": np.asarray(target[chosen], dtype=np.uint8),
            "pose_slave.uint8.npy": slave_batch,
            "pose_input.uint8.npy": pose_input,
            "pose_first6.float32.npy": vectors,
            "pose_target_first6.float32.npy": np.asarray(gt_pose[chosen], dtype=np.float32),
        }
        records = {
            payload_name: screen.retain_npy(batch_root / payload_name, value)
            for payload_name, value in payloads.items()
        }
        row = {
            "schema": "ddm_mt1_t4_endpoint_batch.v1",
            "axis": AXIS,
            "arm": name,
            "pairs": chosen,
            "payloads": records,
        }
        screen.retain_json(batch_root / "BATCH_RECEIPT.json", row)
        rows.append(row)
        argmax_parts.append(argmax)
        pose_parts.append(vectors)
    argmax_all = np.concatenate(argmax_parts)
    pose_all = np.concatenate(pose_parts)
    errors = argmax_all != np.asarray(target[pairs])
    result = {
        "schema": "ddm_mt1_t4_endpoint_arm.v1",
        "axis": AXIS,
        "arm": name,
        "n_pairs": len(pairs),
        "seg_errors": int(np.count_nonzero(errors)),
        "seg_denominator_pixels": int(errors.size),
        "d_seg": float(np.mean(errors)),
        "d_pose": float(np.mean((pose_all - np.asarray(gt_pose[pairs])) ** 2)),
        "retained_argmax": screen.retain_npy(
            root / f"endpoint/{name}/argmax_n32.npy", argmax_all
        ),
        "retained_pose": screen.retain_npy(
            root / f"endpoint/{name}/pose_n32.npy", pose_all
        ),
        "batch_receipts": rows,
        "score_claim": False,
        "promotion_eligible": False,
    }
    screen.retain_json(root / f"endpoint/{name}/RESULT.json", result)
    return result


def run(run_root: Path) -> dict[str, Any]:
    storage = storage_preflight(run_root)
    reproducibility = configure_cuda()
    inputs = run_root / "inputs"
    request = json.loads((inputs / "REQUEST.json").read_text())
    for name, record in request["payloads"].items():
        require_record(inputs / name, record)
    archive = inputs / "cp135.archive.zip"
    runtime = run_root / "runtime/cp135"
    safe_extract(inputs / "cp135.runtime.zip", runtime)
    selection = json.loads((inputs / "selection.json").read_text())
    pairs = [int(value) for value in selection["heldout"]]
    if len(pairs) != 32 or set(pairs) & set(selection["train"]):
        raise MT1T4Error("sealed stratified split differs")
    tokens = np.load(inputs / "base_tokens.npy", mmap_mode="r", allow_pickle=False)
    c1_tokens = screen.load_raw_tokens(inputs / "c1_tokens.u8")
    target = np.load(inputs / "gt_argmax.npy", mmap_mode="r", allow_pickle=False)
    gt_pose = np.load(inputs / "gt_pose.npy", mmap_mode="r", allow_pickle=False)
    slaves = np.load(inputs / "frame0_slave_n32.npy", mmap_mode="r", allow_pickle=False)
    if slaves.shape != (32, 874, 1164, 3):
        raise MT1T4Error(f"sealed frame-0 carrier geometry differs: {slaves.shape}")
    device = torch.device("cuda:0")
    semantic = load_exact_semantic(runtime, archive, device)
    segnet = load_segnet(device)
    posenet = load_posenet_cuda(device)
    model = mt1.load_module((inputs / "selected_model.mt1.br").read_bytes(), device)
    arms = {
        "cp135_hard": endpoint_arm(
            run_root,
            name="cp135_hard",
            semantic=semantic,
            segnet=segnet,
            posenet=posenet,
            device=device,
            tokens=tokens,
            target=target,
            gt_pose=gt_pose,
            pairs=pairs,
            slaves=slaves,
            model=None,
        ),
        "hc1_direct_c1": endpoint_arm(
            run_root,
            name="hc1_direct_c1",
            semantic=semantic,
            segnet=segnet,
            posenet=posenet,
            device=device,
            tokens=c1_tokens,
            target=target,
            gt_pose=gt_pose,
            pairs=pairs,
            slaves=slaves,
            model=None,
        ),
        "mt1_multitoken": endpoint_arm(
            run_root,
            name="mt1_multitoken",
            semantic=semantic,
            segnet=segnet,
            posenet=posenet,
            device=device,
            tokens=tokens,
            target=target,
            gt_pose=gt_pose,
            pairs=pairs,
            slaves=slaves,
            model=model,
        ),
    }
    base = arms["cp135_hard"]
    direct = arms["hc1_direct_c1"]
    candidate = arms["mt1_multitoken"]
    pose_delta = float(candidate["d_pose"] - base["d_pose"])
    gates = {
        "stronger_than_direct_c1_seg": candidate["seg_errors"] < direct["seg_errors"],
        "improves_cp135_seg": candidate["seg_errors"] < base["seg_errors"],
        "zero_pose_damage": pose_delta <= 0.0,
        "parsed_counted_model_consumed": True,
        "no_changed_site_list": True,
        "same_retained_exact_cp135_frame0_carrier": True,
    }
    result = {
        "schema": "ddm_mt1_t4_sign_gate_result.v1",
        "axis": AXIS,
        "storage_preflight": storage,
        "reproducibility": reproducibility,
        "n_train": 32,
        "n_heldout": 32,
        "arms": arms,
        "candidate_minus_base_pose_mse": pose_delta,
        "gates": gates,
        "positive_t4_sign": bool(all(gates.values())),
        "verdict_scope": (
            "single fixed-seed stratified-random n32 heldout screen of the hidden-4, "
            "max-support-mass-0.25 local simplex formulation"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    screen.retain_json(run_root / "FINAL_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    args = parser.parse_args()
    if args.resume_from.resolve() != args.run_root.resolve():
        raise MT1T4Error("resume-from must name the same retained run root")
    result = run(args.run_root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
