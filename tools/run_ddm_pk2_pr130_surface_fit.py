#!/usr/bin/env python3
"""ddm_pk2 PR130 CPR1-style pose-carrier fit on the ep854 surface.

This is a scoped n>=120 advisory runner, not a contest submission builder.  It
preserves PR130's neutral-gray 12D low-rank carrier and CPR1 codec shape, but
uses the ep854 decoded frame_1 surface as the fixed PoseNet master.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


REPO = Path(__file__).resolve().parents[1]
PR130_CODE = Path(
    "/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/code"
)
DEFAULT_SSD_OUT = Path("/Volumes/VertigoDataTier/pact/ddm_pk2_20260808")
EP854_RAW = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pfs1_20260729/d1/eval_root/"
    "submissions/v4d_cr2_ep854/inflated/0.raw"
)
GT_POSE6 = Path(
    "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610/"
    "targets_n600/gt_posenet_pose6.npy"
)
INIT_CARRIER = Path(
    "/Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/"
    "artifacts/checkpoints/archive_carrier_int6_coefftail_s4k.pt"
)

N_PAIRS = 600
SEQ_LEN = 2
CAMERA_H = 874
CAMERA_W = 1164
EVAL_H = 384
EVAL_W = 512
CARRIER_DIM = 12
CARRIER_H = 24
CARRIER_W = 32
REFERENCE_BYTES = 37_545_489
EP854_DERIVED_BYTES = 284_248
EP854_BASE_DSEG = 0.003943024
CURRENT_OWN_VEHICLE_S = 0.7534578126155775
PR130_REFERENCE_DPOSE = 2.331e-5
POSE_TUBE = 0.0025


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, default=json_default, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=json_default, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def tensor_sha256(value: torch.Tensor) -> str:
    arr = value.detach().cpu().contiguous().numpy()
    return sha256_bytes(arr.tobytes())


def npy_sha256(value: np.ndarray) -> str:
    return sha256_bytes(np.ascontiguousarray(value).tobytes())


def setup_imports() -> None:
    sys.path.insert(0, str(PR130_CODE))
    sys.path.insert(0, str((REPO / "src").resolve()))
    sys.path.insert(0, str((REPO / "upstream").resolve()))


def pose_target_center_energy(targets: np.ndarray) -> np.ndarray:
    centered = targets.astype(np.float64) - targets.astype(np.float64).mean(axis=0, keepdims=True)
    return np.mean(centered * centered, axis=1)


def make_selection(n: int, seed: int, targets: np.ndarray) -> tuple[list[int], dict[str, Any]]:
    from tac.subset_selection import MODE_STRATIFIED, select

    governing = pose_target_center_energy(targets).tolist()
    selection = select(
        n,
        N_PAIRS,
        mode=MODE_STRATIFIED,
        seed=seed,
        block_count=10,
        governing=governing,
        governing_name="pose_target_center_energy",
    )
    return list(selection.indices), selection.provenance()


def load_ep854_masters(indices: list[int]) -> torch.Tensor:
    expected_bytes = N_PAIRS * SEQ_LEN * CAMERA_H * CAMERA_W * 3
    if EP854_RAW.stat().st_size != expected_bytes:
        raise ValueError(f"ep854 raw size mismatch: {EP854_RAW.stat().st_size} != {expected_bytes}")
    mm = np.memmap(
        EP854_RAW,
        dtype=np.uint8,
        mode="r",
        shape=(N_PAIRS, SEQ_LEN, CAMERA_H, CAMERA_W, 3),
    )
    frame1 = np.array(mm[indices, 1], copy=True)
    return torch.from_numpy(frame1).permute(0, 3, 1, 2).contiguous().float()


def load_gt_pairs(indices: list[int]) -> torch.Tensor:
    import av
    import frame_utils

    wanted = set(indices)
    selected: dict[int, torch.Tensor] = {}
    container = av.open(str(REPO / "upstream" / "videos" / "0.mkv"))
    prev = None
    pair_idx = 0
    for frame in container.decode(container.streams.video[0]):
        rgb = frame_utils.yuv420_to_rgb(frame)
        if prev is None:
            prev = rgb
            continue
        if pair_idx in wanted:
            selected[pair_idx] = torch.stack([prev, rgb])
        prev = None
        pair_idx += 1
        if len(selected) == len(wanted):
            break
    container.close()
    missing = sorted(wanted.difference(selected))
    if missing:
        raise RuntimeError(f"GT video ended before pairs {missing[:8]}")
    return torch.stack([selected[i] for i in indices]).permute(0, 1, 4, 2, 3).contiguous().float()


def load_scorers(device: torch.device):
    import modules

    posenet = modules.PoseNet().eval().to(device)
    segnet = modules.SegNet().eval().to(device)
    posenet.load_state_dict(load_file(str(REPO / "upstream/models/posenet.safetensors"), device=str(device)))
    segnet.load_state_dict(load_file(str(REPO / "upstream/models/segnet.safetensors"), device=str(device)))
    for p in list(posenet.parameters()) + list(segnet.parameters()):
        p.requires_grad_(False)
    return posenet, segnet


def render_slave(master_camera: torch.Tensor, coeff: torch.Tensor, raw_basis: torch.Tensor) -> torch.Tensor:
    from learned_pose_carrier_oracle import render_slave as pr130_render_slave

    return pr130_render_slave(
        master_camera,
        coeff,
        raw_basis,
        amplitude=32.0,
        carrier_base="gray",
    )


def predict_pose(posenet, master_camera: torch.Tensor, coeff: torch.Tensor, raw_basis: torch.Tensor) -> torch.Tensor:
    from learned_pose_carrier_oracle import predict

    return predict(
        posenet,
        master_camera,
        coeff,
        raw_basis,
        amplitude=32.0,
        carrier_base="gray",
        master_carrier_amplitude=0.0,
    )


def quantize_basis_compat(raw: torch.Tensor, bits: int):
    from learned_pose_carrier_oracle import quantize_basis

    return quantize_basis(raw, bits)


def fake_quant_basis_compat(raw: torch.Tensor, bits: int) -> torch.Tensor:
    from learned_pose_carrier_oracle import fake_quant_basis

    return fake_quant_basis(raw, bits)


def quantize_coeff_compat(coeff: torch.Tensor, bits: int):
    from train_pose_carrier_full import quantize_coeff

    return quantize_coeff(coeff, bits)


def fake_quant_coeff_from_full(selected: torch.Tensor, full_coeff: torch.Tensor, bits: int) -> torch.Tensor:
    max_code = (1 << (bits - 1)) - 1
    scales = full_coeff.detach().abs().amax(dim=0).clamp_min(1e-8) / max_code
    normalized = (selected / scales).clamp(-max_code, max_code)
    codes = normalized + (normalized.round() - normalized).detach()
    return codes * scales


def summarize(values: torch.Tensor | np.ndarray, threshold: float = POSE_TUBE) -> dict[str, Any]:
    if isinstance(values, torch.Tensor):
        arr = values.detach().cpu().float().numpy()
    else:
        arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "max": float(arr.max()),
        "min": float(arr.min()),
        "reached": int((arr < threshold).sum()),
        "total": int(arr.size),
    }


def encode_delta_zigzag_int12(codes_signed: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes_signed, dtype=np.int64)
    unsigned = codes & 0xFFF
    prev = np.zeros_like(unsigned)
    prev[1:] = unsigned[:-1]
    delta_u = (unsigned - prev) & 0xFFF
    delta_signed = np.where(delta_u >= 0x800, delta_u - 0x1000, delta_u)
    return ((delta_signed << 1) ^ (delta_signed >> 63)) & 0xFFF


def decode_delta_zigzag_int12(encoded: np.ndarray) -> np.ndarray:
    enc = np.asarray(encoded, dtype=np.int64)
    delta = (enc >> 1) ^ -(enc & 1)
    unsigned = np.cumsum(delta, axis=0) & 0xFFF
    return np.where(unsigned >= 0x800, unsigned - 0x1000, unsigned).astype(np.int32)


def byteclose_carrier(
    basis: torch.Tensor,
    coeff_full: torch.Tensor,
    out_dir: Path,
) -> dict[str, Any]:
    from carrier_codec import decode_compact_carrier, encode_compact_carrier

    basis_q, basis_codes_t, basis_scales_t = quantize_basis_compat(basis, 5)
    coeff_q, coeff_codes_t, coeff_scales_t = quantize_coeff_compat(coeff_full, 12)

    basis_codes = basis_codes_t.detach().cpu().numpy().astype(np.int32).reshape(-1)
    basis_scales = basis_scales_t.detach().cpu().numpy().astype("<f4")
    coeff_codes = coeff_codes_t.detach().cpu().numpy().astype(np.int32)
    coeff_scales = coeff_scales_t.detach().cpu().numpy().astype("<f4")
    encoded_coefficients = encode_delta_zigzag_int12(coeff_codes)
    decoded_codes = decode_delta_zigzag_int12(encoded_coefficients)
    if not np.array_equal(decoded_codes, coeff_codes.astype(np.int32)):
        raise AssertionError("delta-zigzag int12 reconstruction changed coefficient codes")

    blob = encode_compact_carrier(
        basis_scales,
        basis_codes,
        coeff_scales,
        encoded_coefficients.astype(np.int32),
    )
    decoded = decode_compact_carrier(
        blob,
        basis_count=CARRIER_DIM * 3 * CARRIER_H * CARRIER_W,
        frames=N_PAIRS,
        dimensions=CARRIER_DIM,
    )
    expected = (
        basis_scales,
        basis_codes,
        coeff_scales,
        encoded_coefficients.astype(np.int32),
    )
    for actual, want in zip(decoded, expected, strict=True):
        if not np.array_equal(actual, want):
            raise AssertionError("CPR1 parse-back changed a carrier symbol")

    out_dir.mkdir(parents=True, exist_ok=True)
    blob_path = out_dir / "pk2_fitted_cpr1.carrier"
    tmp = blob_path.with_suffix(".carrier.tmp")
    tmp.write_bytes(blob)
    os.replace(tmp, blob_path)

    return {
        "carrier_path": str(blob_path),
        "carrier_bytes": len(blob),
        "carrier_sha256": sha256_file(blob_path),
        "carrier_magic": blob[:4].decode("ascii"),
        "basis_codes_shape": list(basis_codes.shape),
        "encoded_coefficients_shape": list(encoded_coefficients.shape),
        "basis_code_range": [int(basis_codes.min()), int(basis_codes.max())],
        "coefficient_code_range": [int(coeff_codes.min()), int(coeff_codes.max())],
        "basis_scales_sha256": npy_sha256(basis_scales),
        "basis_codes_sha256": npy_sha256(basis_codes),
        "coeff_scales_sha256": npy_sha256(coeff_scales),
        "encoded_coefficients_sha256": npy_sha256(encoded_coefficients.astype(np.int32)),
        "quantized_basis_sha256": tensor_sha256(basis_q),
        "quantized_coeff_sha256": tensor_sha256(coeff_q),
    }


def score_selected(
    posenet,
    segnet,
    masters: torch.Tensor,
    gt_pairs: torch.Tensor,
    target_pose: torch.Tensor,
    coeff_q_selected: torch.Tensor,
    basis_q: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    pose_rows: list[torch.Tensor] = []
    seg_rows: list[torch.Tensor] = []
    frame1_equal = 0
    n = masters.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            master = masters[start:end].to(device)
            coeff = coeff_q_selected[start:end].to(device)
            basis = basis_q.to(device)
            gt = gt_pairs[start:end].to(device)
            slave = render_slave(master, coeff, basis)
            candidate = torch.stack([slave, master], dim=1)
            frame1_equal += int(torch.eq(candidate[:, 1].cpu().to(torch.uint8), masters[start:end].to(torch.uint8)).all(dim=(1, 2, 3)).sum().item())

            pred_pose = posenet(posenet.preprocess_input(candidate))["pose"][:, :6]
            pose_mse = (pred_pose - target_pose[start:end].to(device)).square().mean(dim=1)
            pose_rows.append(pose_mse.detach().cpu())

            cand_seg = segnet(segnet.preprocess_input(candidate)).argmax(dim=1)
            gt_seg = segnet(segnet.preprocess_input(gt)).argmax(dim=1)
            diff = (cand_seg != gt_seg).float().mean(dim=(1, 2))
            seg_rows.append(diff.detach().cpu())

    pose = torch.cat(pose_rows)
    seg = torch.cat(seg_rows)
    return {
        "pose_per_pair": pose.numpy().astype(np.float64).tolist(),
        "seg_per_pair": seg.numpy().astype(np.float64).tolist(),
        "d_pose": float(pose.mean()),
        "d_seg": float(seg.mean()),
        "pose_summary": summarize(pose),
        "seg_summary": summarize(seg, threshold=1.0),
        "frame1_byte_identity_count": int(frame1_equal),
        "frame1_byte_identity_total": int(n),
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    setup_imports()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device("cpu")

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    receipts = out_dir / "receipts.jsonl"
    started = datetime.now(UTC).isoformat()

    targets_np = np.load(GT_POSE6).astype(np.float32)
    if targets_np.shape != (N_PAIRS, 6):
        raise ValueError(f"Pose target cache shape mismatch: {targets_np.shape}")
    indices, selection = make_selection(args.n, args.seed, targets_np)
    target_selected = torch.from_numpy(targets_np[indices]).float()
    masters = load_ep854_masters(indices)

    posenet, segnet = load_scorers(device)
    init = torch.load(INIT_CARRIER, map_location="cpu", weights_only=False)
    coeff_initial_full = init["coeff"].float()
    basis_initial = init["basis"].float()
    if tuple(coeff_initial_full.shape) != (N_PAIRS, CARRIER_DIM):
        raise ValueError(f"init coeff shape mismatch: {tuple(coeff_initial_full.shape)}")
    if tuple(basis_initial.shape) != (CARRIER_DIM, 3, CARRIER_H, CARRIER_W):
        raise ValueError(f"init basis shape mismatch: {tuple(basis_initial.shape)}")

    raw_basis = torch.nn.Parameter(basis_initial.clone())
    coeff_selected = torch.nn.Parameter(coeff_initial_full[indices].clone())
    basis_optimizer = torch.optim.Adam([raw_basis], lr=args.lr_basis)
    coeff_optimizer = torch.optim.Adam([coeff_selected], lr=args.lr_coeff)
    basis_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        basis_optimizer, T_max=args.steps, eta_min=args.lr_basis * 0.01
    )
    coeff_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        coeff_optimizer, T_max=args.steps, eta_min=args.lr_coeff * 0.01
    )

    latest_ckpt = out_dir / "checkpoints" / "pk2_latest.pt"
    start_step = 1
    best: dict[str, Any] = {"mean": float("inf"), "step": None, "basis": None, "coeff_selected": None}
    history: list[dict[str, Any]] = []
    if args.resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        raw_basis.data.copy_(ckpt["basis"])
        coeff_selected.data.copy_(ckpt["coeff_selected"])
        basis_optimizer.load_state_dict(ckpt["basis_optimizer"])
        coeff_optimizer.load_state_dict(ckpt["coeff_optimizer"])
        basis_scheduler.load_state_dict(ckpt["basis_scheduler"])
        coeff_scheduler.load_state_dict(ckpt["coeff_scheduler"])
        start_step = int(ckpt["step"]) + 1
        best = ckpt["best"]
        history = ckpt.get("history", [])

    target_span = targets_np.max(axis=0) - targets_np.min(axis=0)
    target_scale = torch.from_numpy(np.clip(target_span, 1e-4, None)).float()
    order_generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(args.n, generator=order_generator)
    cursor = 0
    freeze_until = int(args.steps * args.basis_freeze_fraction)
    train_until = int(args.steps * args.basis_train_until_fraction)
    qat_start = int(args.steps * args.qat_fraction)
    coeff_qat_start = int(args.steps * args.coeff_qat_fraction)

    append_jsonl(receipts, {
        "event": "start_or_resume",
        "timestamp_utc": started,
        "argv": vars(args),
        "selection": selection,
        "input_hashes": {
            "ep854_raw": sha256_file(EP854_RAW),
            "gt_pose6_npy": sha256_file(GT_POSE6),
            "init_carrier": sha256_file(INIT_CARRIER),
            "pr130_carrier_codec_py": sha256_file(PR130_CODE / "carrier_codec.py"),
            "pr130_learned_pose_carrier_oracle_py": sha256_file(PR130_CODE / "learned_pose_carrier_oracle.py"),
            "pr130_train_pose_carrier_full_py": sha256_file(PR130_CODE / "train_pose_carrier_full.py"),
        },
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": str(device),
        },
    })

    for step in range(start_step, args.stop_after_step + 1):
        if cursor + args.batch_size > args.n:
            order = torch.randperm(args.n, generator=order_generator)
            cursor = 0
        batch_local = order[cursor:cursor + args.batch_size]
        cursor += args.batch_size

        master = masters.index_select(0, batch_local).to(device)
        target = target_selected.index_select(0, batch_local).to(device)
        selected_forward = coeff_selected.index_select(0, batch_local)
        full_for_scales = coeff_initial_full.clone()
        full_for_scales[indices] = coeff_selected

        train_basis = step > freeze_until and step <= train_until
        forward_basis = raw_basis if train_basis else raw_basis.detach()
        if step > qat_start:
            forward_basis = fake_quant_basis_compat(forward_basis, 5)
        if step > coeff_qat_start:
            selected_forward = fake_quant_coeff_from_full(selected_forward, full_for_scales, 12)

        pred = predict_pose(posenet, master, selected_forward, forward_basis)
        residual = pred - target
        normalized = residual / target_scale.to(device)
        loss = normalized.square().mean() + 0.02 * residual.square().mean()

        basis_optimizer.zero_grad(set_to_none=True)
        coeff_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_basis], 10.0)
        torch.nn.utils.clip_grad_norm_([coeff_selected], 10.0)
        basis_optimizer.step()
        coeff_optimizer.step()
        basis_scheduler.step()
        coeff_scheduler.step()

        if step == 1 or step % args.log_every == 0 or step == args.stop_after_step:
            rec = {
                "event": "train_step",
                "step": step,
                "phase": (
                    "full_qat" if step > qat_start and step > coeff_qat_start
                    else "basis_qat" if step > qat_start
                    else "float"
                ),
                "train_basis": train_basis,
                "loss": float(loss.detach()),
                "batch_pose_mse": summarize(residual.detach().square().mean(dim=1)),
                "lr_basis": float(basis_optimizer.param_groups[0]["lr"]),
                "lr_coeff": float(coeff_optimizer.param_groups[0]["lr"]),
            }
            append_jsonl(receipts, rec)
            print(json.dumps(rec), flush=True)

        if step % args.eval_every == 0 or step == args.stop_after_step:
            full_eval_coeff = coeff_initial_full.clone()
            full_eval_coeff[indices] = coeff_selected.detach().cpu()
            basis_q, _, _ = quantize_basis_compat(raw_basis.detach().cpu(), 5)
            coeff_q_full, _, _ = quantize_coeff_compat(full_eval_coeff, 12)
            with torch.no_grad():
                eval_rows = []
                for s in range(0, args.n, args.eval_batch_size):
                    e = min(s + args.eval_batch_size, args.n)
                    pred_eval = predict_pose(
                        posenet,
                        masters[s:e].to(device),
                        coeff_q_full[indices[s:e]].to(device),
                        basis_q.to(device),
                    )
                    eval_rows.append(
                        (pred_eval - target_selected[s:e].to(device)).square().mean(dim=1).cpu()
                    )
            eval_mse = torch.cat(eval_rows)
            summary = summarize(eval_mse)
            eval_rec = {
                "event": "quantized_eval",
                "step": step,
                "summary": summary,
                "per_pair_mse_sha256": npy_sha256(eval_mse.numpy()),
            }
            history.append(eval_rec)
            append_jsonl(receipts, eval_rec)
            print(json.dumps(eval_rec), flush=True)
            if summary["mean"] < best["mean"]:
                best = {
                    "mean": summary["mean"],
                    "step": step,
                    "basis": raw_basis.detach().cpu().clone(),
                    "coeff_selected": coeff_selected.detach().cpu().clone(),
                    "per_pair_mse": eval_mse.numpy().astype(np.float64).tolist(),
                }
                best_path = out_dir / "checkpoints" / f"pk2_best_step{step:05d}.pt"
                best_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"basis": best["basis"], "coeff_selected": best["coeff_selected"], "best": best}, best_path)

            latest_payload = {
                "step": step,
                "basis": raw_basis.detach().cpu(),
                "coeff_selected": coeff_selected.detach().cpu(),
                "basis_optimizer": basis_optimizer.state_dict(),
                "coeff_optimizer": coeff_optimizer.state_dict(),
                "basis_scheduler": basis_scheduler.state_dict(),
                "coeff_scheduler": coeff_scheduler.state_dict(),
                "best": best,
                "history": history,
            }
            latest_ckpt.parent.mkdir(parents=True, exist_ok=True)
            tmp_ckpt = latest_ckpt.with_suffix(".tmp")
            torch.save(latest_payload, tmp_ckpt)
            os.replace(tmp_ckpt, latest_ckpt)

    if best["basis"] is None or best["coeff_selected"] is None:
        raise RuntimeError("training ended without any quantized eval")

    final_coeff_full = coeff_initial_full.clone()
    final_coeff_full[indices] = best["coeff_selected"]
    carrier = byteclose_carrier(best["basis"], final_coeff_full, out_dir)
    basis_q, _, _ = quantize_basis_compat(best["basis"], 5)
    coeff_q_full, _, _ = quantize_coeff_compat(final_coeff_full, 12)
    gt_pairs = load_gt_pairs(indices)
    score = score_selected(
        posenet,
        segnet,
        masters,
        gt_pairs,
        target_selected,
        coeff_q_full[indices],
        basis_q,
        args.eval_batch_size,
        device,
    )

    composed_bytes = EP854_DERIVED_BYTES + int(carrier["carrier_bytes"])
    composed_s = (
        100.0 * score["d_seg"]
        + math.sqrt(10.0 * score["d_pose"])
        + 25.0 * composed_bytes / REFERENCE_BYTES
    )
    ep854_with_pr130_ref_pose_s = (
        100.0 * EP854_BASE_DSEG
        + math.sqrt(10.0 * PR130_REFERENCE_DPOSE)
        + 25.0 * (EP854_DERIVED_BYTES + 23_054) / REFERENCE_BYTES
    )
    result = {
        "task": "ddm_pk2_candidate_A_pr130_cpr1_surface_fit_ep854",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "sample_scope": "seeded stratified n>=120; not contest auth; not n600",
        "selection": selection,
        "indices": indices,
        "training": {
            "best_step": best["step"],
            "best_quantized_train_summary": summarize(np.asarray(best["per_pair_mse"], dtype=np.float64)),
            "history": history,
            "mechanism_delta_vs_pr130_recipe": (
                "scope-reduced CPU fit on ep854 masters; PR130 neutral-gray 12D carrier, "
                "lr/fraction/QAT structure, 5-bit CPR1 basis, 12-bit coefficients retained; "
                "full CUDA raw-video 49-stage E2E not run"
            ),
        },
        "carrier": carrier,
        "scoring": score,
        "report_values": {
            "d_pose": score["d_pose"],
            "d_seg": score["d_seg"],
            "pose_tube": POSE_TUBE,
            "pr130_external_reference_d_pose": PR130_REFERENCE_DPOSE,
            "counted_carrier_bytes": carrier["carrier_bytes"],
            "ep854_derived_bytes_without_carrier": EP854_DERIVED_BYTES,
            "composed_bytes_ep854_plus_carrier": composed_bytes,
            "composed_s_sample_components": {
                "seg_term": 100.0 * score["d_seg"],
                "pose_term": math.sqrt(10.0 * score["d_pose"]),
                "rate_term": 25.0 * composed_bytes / REFERENCE_BYTES,
                "S": composed_s,
            },
            "delta_vs_current_own_vehicle_pointer": composed_s - CURRENT_OWN_VEHICLE_S,
            "ep854_with_pr130_external_reference_pose_s": ep854_with_pr130_ref_pose_s,
        },
        "borrowed_substrate_accounting": {
            "theirs": (
                "PR130 neutral-gray low-rank pose carrier mechanism, 12x3x24x32 basis, "
                "600x12 coefficient shape, 5-bit/12-bit CPR1 Huffman/Rice codec, "
                "and init checkpoint lineage."
            ),
            "ours": (
                "Surface-specific CPU fit of selected coefficient rows and basis on ep854 "
                "frame_1 masters, canonical stratified sample/ratio provenance, CPR1 parse-back "
                "receipt, and CPU-torch PoseNet/SegNet scoring on the same selected rows."
            ),
        },
        "provenance": {
            "cwd": str(REPO),
            "ssd_out_dir": str(out_dir),
            "receipts_jsonl": str(receipts),
            "input_paths": {
                "ep854_raw": str(EP854_RAW),
                "gt_pose6": str(GT_POSE6),
                "init_carrier": str(INIT_CARRIER),
                "pr130_code": str(PR130_CODE),
            },
            "input_hashes": {
                "ep854_raw": sha256_file(EP854_RAW),
                "gt_pose6": sha256_file(GT_POSE6),
                "init_carrier": sha256_file(INIT_CARRIER),
            },
        },
        "verdict": {
            "pose_tube_status": "PASS" if score["d_pose"] <= POSE_TUBE else "FAIL",
            "verdict_scope": (
                "INSTANCE: selected n120 stratified ep854 surface fit under CPU scope-reduced "
                "PR130-style recipe; does not promote or kill full n600/CUDA PR130 family."
            ),
        },
    }
    result_path = out_dir / "pk2_result.json"
    atomic_json(result_path, result)
    append_jsonl(receipts, {
        "event": "final_result",
        "timestamp_utc": result["timestamp_utc"],
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "d_pose": score["d_pose"],
        "d_seg": score["d_seg"],
        "carrier_bytes": carrier["carrier_bytes"],
        "composed_s_sample": composed_s,
    })
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_SSD_OUT)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--steps", type=int, default=750)
    parser.add_argument("--stop-after-step", type=int, default=750)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=12)
    parser.add_argument("--eval-every", type=int, default=75)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--lr-basis", type=float, default=0.003)
    parser.add_argument("--lr-coeff", type=float, default=0.03)
    parser.add_argument("--basis-freeze-fraction", type=float, default=0.30)
    parser.add_argument("--basis-train-until-fraction", type=float, default=1.0)
    parser.add_argument("--qat-fraction", type=float, default=0.65)
    parser.add_argument("--coeff-qat-fraction", type=float, default=0.65)
    parser.add_argument("--torch-threads", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n < 120:
        raise ValueError("--n must be >= 120 for this charter")
    if args.n > N_PAIRS:
        raise ValueError("--n cannot exceed 600")
    if args.stop_after_step < 1 or args.stop_after_step > args.steps:
        raise ValueError("--stop-after-step must be within [1, --steps]")
    t0 = time.time()
    result = train(args)
    print(json.dumps({
        "event": "done",
        "elapsed_s": time.time() - t0,
        "result_path": str(args.out_dir.resolve() / "pk2_result.json"),
        "d_pose": result["scoring"]["d_pose"],
        "d_seg": result["scoring"]["d_seg"],
        "carrier_bytes": result["carrier"]["carrier_bytes"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
