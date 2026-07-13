#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe V9 trunk gradient-role conflict at a read-only real checkpoint.

This is a bounded, $0, macOS-CPU research probe.  It loads the deployed EMA NPZ
without modifying the source run, executes the repository's deterministic Torch
twin of the MLX witness and the real frozen CPU scorers, and compares trunk
gradients from pair-local score losses with the already-typed future temporal
losses.  The receipt is NON-PROMOTABLE and cannot move a score pointer.

The live epoch-275 program has zero temporal weight by schedule.  Consequently
the script reports both (1) the actual live-program result (zero temporal norm;
cosine undefined) and (2) a clearly named counterfactual fully-armed mechanism
probe using the typed eventual weights, w_phase=0.4 and w_screw=0.1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "experiments", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.boundary_math.dseg_aware_fourier_taper import (  # noqa: E402
    apply_dseg_aware_fourier_taper,
    compute_dseg_aware_fourier_taper,
)
from tac.boundary_math.lever_b_levelset_generator import (  # noqa: E402
    CurveletBankConfig,
    build_coords,
    curvelet_directional_B,
    curvelet_feats,
)
from tac.boundary_math.phase_primitives import (  # noqa: E402
    advect_tie_field_numpy,
    cross_scored_frame_xi_interp,
    gt_tie_targets_numpy,
)
from tac.boundary_math.warp_real_luma_frame0 import (  # noqa: E402
    GroundHomographyGeom,
    xi_from_pose_calibration,
)
from tac.cuda_levelset_training import (  # noqa: E402
    CudaLevelSetConfig,
    TorchLevelSetWitness,
    TorchPoseCarrier,
    forward_parity_against_numpy,
    homography_grid_from_xi,
    pose_objective_torch,
    realized_signed_margin,
    warp_field_persist_torch,
    witness_tie_coordinate_torch,
)

import train_levelset_witness_realized_through_R_torch as torch_trainer  # noqa: E402

SCHEMA = "sps_gradient_role_conflict_probe_receipt.v1"
AUTHORITY = "[macOS-CPU local probe; NON-PROMOTABLE]"
TRUNK_PREFIXES = ("in_proj.", "hidden.")
PRED_WEIGHTS = {"seg": 100.0, "pose": 1.0}
TEMP_WEIGHTS = {"phase_advection": 0.4, "temporal_screw": 0.1}
MATERIAL_FRACTION_ASSUMPTION = 0.10
MATERIAL_COSINE_THRESHOLD_ASSUMPTION = -0.05


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def _stored_npz_memmap(path: Path, member: str) -> np.memmap:
    """Memory-map one ZIP_STORED .npy member without inflating the 5 GB cache."""
    name = member if member.endswith(".npy") else member + ".npy"
    with zipfile.ZipFile(path) as zf:
        info = zf.getinfo(name)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{path}:{name} is not ZIP_STORED; bounded mmap refused")
        header_offset = int(info.header_offset)
    with path.open("rb") as f:
        f.seek(header_offset)
        local = f.read(30)
        sig, *_rest, filename_len, extra_len = struct.unpack("<IHHHHHIIIHH", local)
        if sig != 0x04034B50:
            raise ValueError(f"bad local ZIP header for {name}")
        f.seek(header_offset + 30 + filename_len + extra_len)
        version = np.lib.format.read_magic(f)
        shape, fortran, dtype = np.lib.format._read_array_header(f, version)  # type: ignore[attr-defined]
        data_offset = f.tell()
    return np.memmap(
        path,
        mode="r",
        dtype=dtype,
        offset=data_offset,
        shape=shape,
        order="F" if fortran else "C",
    )


def _streaming_taper(margins: np.memmap, feats: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """Reproduce the active auto-scale n600 taper without a multi-GB stack."""
    # float32 and float64 have identical order for these cached fp32 values, so
    # the median is invariant to the source routine's float64 cast.
    width = float(np.median(np.abs(margins)))
    if not np.isfinite(width) or width <= 1e-8:
        width = float(np.mean(np.abs(margins), dtype=np.float64))
    if not np.isfinite(width) or width <= 1e-8:
        width = 1.0
    saliency_sum = np.zeros(margins.shape[1] * margins.shape[2], dtype=np.float64)
    for pair_index in range(int(margins.shape[0])):
        row = np.asarray(margins[pair_index], dtype=np.float64).reshape(-1)
        saliency_sum += np.exp(-np.abs(row) / width)
    saliency = saliency_sum / float(margins.shape[0])
    saliency /= float(saliency.mean())
    taper = compute_dseg_aware_fourier_taper(
        feats, saliency.astype(np.float32), strength=1.0, floor=0.05
    )
    return taper, {
        "auto_scale": width,
        "min": float(taper.min()),
        "max": float(taper.max()),
        "mean": float(taper.mean()),
    }


def _load_checkpoint_model(checkpoint: Path, feats: np.ndarray):
    import torch

    with np.load(checkpoint, allow_pickle=False) as z:
        cfg = CudaLevelSetConfig(
            n_pairs=int(z["code"].shape[0] // 2),
            in_feat=int(z["__cfg_in_feat"]),
            hidden_dim=int(z["__cfg_hidden_dim"]),
            n_hidden=int(z["__cfg_n_hidden"]),
            mod_dim=int(z["code"].shape[1]),
            activation=str(z["__cfg_activation"].item()),
            hosc_beta=float(z["__cfg_hosc_beta"]),
            hosc_omega=float(z["__cfg_hosc_omega"]),
            softmax_temp=float(z["__cfg_softmax_temp"]),
            chroma=bool(int(z["__cfg_chroma"])),
            render_h=int(z["__render_hw"][0]),
            render_w=int(z["__render_hw"][1]),
        )
        if cfg.in_feat != feats.shape[1]:
            raise ValueError(f"feature width {feats.shape[1]} != checkpoint {cfg.in_feat}")
        model = TorchLevelSetWitness.build(cfg, seed=0).cpu()
        named = dict(model.named_parameters())
        with torch.no_grad():
            for name, param in named.items():
                if name not in z.files:
                    raise ValueError(f"checkpoint missing model parameter {name}")
                param.copy_(torch.from_numpy(np.asarray(z[name], np.float32)))
            model.softmax_temp.fill_(float(z["__cfg_softmax_temp"]))
            model.hosc_beta.fill_(float(z["__cfg_hosc_beta"]))

        pose_geom = GroundHomographyGeom.eon(
            native_hw=(cfg.camera_h, cfg.camera_w), pitch=0.0
        )
        carrier = TorchPoseCarrier.build(
            np.asarray(z["pose_carrier.xi_stored"], np.float32),
            pose_geom,
            residual_scale=1.0,
        ).cpu()
        with torch.no_grad():
            carrier.dxi.copy_(
                torch.from_numpy(np.asarray(z["pose_carrier.dxi"], np.float32))
            )
        model.pose_carrier = carrier
        meta = {
            "epoch": int(z["__epoch"]),
            "checkpoint_git_sha": str(z["__cfg_git_sha"].item()),
            "checkpoint_git_dirty": bool(int(z["__cfg_git_dirty"])),
            "softmax_temp": float(z["__cfg_softmax_temp"]),
            "hosc_beta": float(z["__cfg_hosc_beta"]),
            "config": {
                "n_pairs": cfg.n_pairs,
                "in_feat": cfg.in_feat,
                "hidden_dim": cfg.hidden_dim,
                "n_hidden": cfg.n_hidden,
                "mod_dim": cfg.mod_dim,
                "activation": cfg.activation,
                "render_hw": [cfg.render_h, cfg.render_w],
            },
        }
    model.eval()
    return model, cfg, meta


def _phase_row(pair_index: int, lstars, margins, gt_poses, geom):
    _t, direction, _active = gt_tie_targets_numpy(
        np.asarray(lstars[pair_index]), np.asarray(margins[pair_index]), band=2.0
    )
    if pair_index == 0:
        return (
            np.full_like(direction, -1.0, dtype=np.float32),
            direction.astype(np.float32),
            np.zeros_like(direction, dtype=np.float32),
        )
    prev_t, _prev_direction, prev_active = gt_tie_targets_numpy(
        np.asarray(lstars[pair_index - 1]),
        np.asarray(margins[pair_index - 1]),
        band=2.0,
    )
    xi_prev = xi_from_pose_calibration(
        np.asarray(gt_poses[pair_index - 1]), -0.003224707899359239, 0.0, -0.01
    )
    xi_now = xi_from_pose_calibration(
        np.asarray(gt_poses[pair_index]), -0.003224707899359239, 0.0, -0.01
    )
    xi_cross = cross_scored_frame_xi_interp(xi_prev, xi_now)
    ref_warp = advect_tie_field_numpy(
        np.where(prev_t >= 0.0, prev_t, 0.0).astype(np.float32), xi_cross, geom
    )
    active_warp = (
        advect_tie_field_numpy(prev_active.astype(np.float32), xi_cross, geom) >= 0.5
    )
    ref = np.where(active_warp, ref_warp, -1.0).astype(np.float32)
    weight = (
        (np.asarray(margins[pair_index]) < 2.0)
        & np.isin(np.asarray(lstars[pair_index]), [0, 1, 2])
        & active_warp
    ).astype(np.float32)
    return ref, direction.astype(np.float32), weight


def _pose6_differentiable(pose_net, frames_nhwc):
    """Gradient-preserving twin of upstream PoseNet.preprocess_input.

    The frozen evaluator helper ``frame_utils.rgb_to_yuv6`` is decorated with
    ``torch.no_grad`` because it was written for evaluation.  The MLX trainer
    uses a differentiable twin.  This local probe mirrors the same BT.601/YUV6
    arithmetic and validates its forward output against the official helper.
    """
    import torch
    import torch.nn.functional as F

    batch, timesteps = frames_nhwc.shape[:2]
    rgb = frames_nhwc.permute(0, 1, 4, 2, 3).contiguous().reshape(
        batch * timesteps, 3, frames_nhwc.shape[2], frames_nhwc.shape[3]
    )
    rgb = F.interpolate(rgb, size=(384, 512), mode="bilinear")
    h2, w2 = rgb.shape[-2] // 2, rgb.shape[-1] // 2
    rgb = rgb[..., : 2 * h2, : 2 * w2]
    red, green, blue = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = (0.299 * red + 0.587 * green + 0.114 * blue).clamp(0.0, 255.0)
    u = ((blue - y) / 1.772 + 128.0).clamp(0.0, 255.0)
    v = ((red - y) / 1.402 + 128.0).clamp(0.0, 255.0)
    u_sub = (
        u[..., 0::2, 0::2]
        + u[..., 1::2, 0::2]
        + u[..., 0::2, 1::2]
        + u[..., 1::2, 1::2]
    ) * 0.25
    v_sub = (
        v[..., 0::2, 0::2]
        + v[..., 1::2, 0::2]
        + v[..., 0::2, 1::2]
        + v[..., 1::2, 1::2]
    ) * 0.25
    yuv6 = torch.stack(
        (
            y[..., 0::2, 0::2],
            y[..., 1::2, 0::2],
            y[..., 0::2, 1::2],
            y[..., 1::2, 1::2],
            u_sub,
            v_sub,
        ),
        dim=1,
    ).reshape(batch, timesteps * 6, h2, w2)
    output = pose_net(yuv6)
    pose = output["pose"] if isinstance(output, dict) else output
    half = next(
        (head.out // 2 for head in pose_net.hydra.heads if head.name == "pose"),
        pose.shape[-1] // 2,
    )
    return pose[:, :half]


def _losses_for_pair(model, cfg, feats_t, pair_index, data, scorers, phase_geom):
    import torch

    lstars, margins, gt_poses = data
    seg, pose = scorers
    frames, _phi = torch_trainer._generated_pose_pair_dispatch(
        model,
        feats_t,
        [pair_index],
        model.pose_carrier,
        cfg,
    )
    seg_in = seg.preprocess_input(frames.permute(0, 1, 4, 2, 3).contiguous())
    logits1 = seg(seg_in)
    target = torch.as_tensor(
        np.asarray(lstars[pair_index]).copy(), dtype=torch.long
    ).unsqueeze(0)
    gt_margin = torch.as_tensor(
        np.asarray(margins[pair_index]).copy(), dtype=logits1.dtype
    ).unsqueeze(0)
    tau = 0.31
    per_pixel = (
        tau * torch.logsumexp(logits1 / tau, dim=1)
        - logits1.gather(1, target[:, None]).squeeze(1)
    )
    seg_weight = 1.0 + 4.0 * torch.exp(-gt_margin.clamp_min(0.0))
    seg_loss = (per_pixel * seg_weight).mean()

    pose6 = _pose6_differentiable(pose, frames)
    official_pose6 = torch_trainer._pose6(pose, frames.detach())
    pose_forward_delta = float((pose6.detach() - official_pose6.detach()).abs().max())
    pose_target = torch.as_tensor(
        np.asarray(gt_poses[pair_index]).copy(), dtype=pose6.dtype
    ).unsqueeze(0)
    pose_loss = pose_objective_torch(pose6, pose_target)

    raw_logits = logits1.permute(0, 2, 3, 1).contiguous()
    signed = realized_signed_margin(raw_logits, target)
    ref_np, direction_np, weight_np = _phase_row(
        pair_index, lstars, margins, gt_poses, phase_geom
    )
    ref = torch.as_tensor(ref_np, dtype=signed.dtype).unsqueeze(0)
    direction = torch.as_tensor(direction_np, dtype=signed.dtype).unsqueeze(0)
    phase_weight = torch.as_tensor(weight_np, dtype=signed.dtype).unsqueeze(0)
    tie = witness_tie_coordinate_torch(signed, direction)
    phase_num = ((tie - ref).square() * phase_weight).sum(dim=(-2, -1))
    phase_den = phase_weight.sum(dim=(-2, -1)) + 1e-6
    phase_loss = (phase_num / phase_den).mean()

    frame0 = frames[:, 0:1]
    logits0 = seg(
        seg.preprocess_input(frame0.permute(0, 1, 4, 2, 3).contiguous())
    ).permute(0, 2, 3, 1).contiguous()
    classes = [0, 1, 2]
    prob0 = torch.softmax(logits0, dim=-1)[..., classes]
    prob1 = torch.softmax(raw_logits, dim=-1)[..., classes]
    xi = xi_from_pose_calibration(
        np.asarray(gt_poses[pair_index]), -0.003224707899359239, 0.0, -0.01
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        grid, valid = homography_grid_from_xi(
            xi, phase_geom, device=prob0.device, dtype=prob0.dtype
        )
    warped0 = warp_field_persist_torch(prob0, grid, valid)
    annulus = torch.as_tensor(
        np.asarray(margins[pair_index]) < 2.0, dtype=prob0.dtype
    ).unsqueeze(0)
    screw_num = ((prob1 - warped0).square().sum(-1) * annulus).sum(dim=(-2, -1))
    screw_den = annulus.sum(dim=(-2, -1)) + 1e-6
    screw_loss = (screw_num / screw_den).mean()

    prediction_seg = PRED_WEIGHTS["seg"] * seg_loss
    prediction_pose = PRED_WEIGHTS["pose"] * pose_loss
    prediction = prediction_seg + prediction_pose
    temporal = (
        TEMP_WEIGHTS["phase_advection"] * phase_loss
        + TEMP_WEIGHTS["temporal_screw"] * screw_loss
    )
    return prediction_seg, prediction_pose, temporal, {
        "seg_raw": float(seg_loss.detach()),
        "seg_weighted": float((PRED_WEIGHTS["seg"] * seg_loss).detach()),
        "pose_raw_contest_objective": float(pose_loss.detach()),
        "pose_weighted": float((PRED_WEIGHTS["pose"] * pose_loss).detach()),
        "pose_differentiable_vs_official_forward_max_abs": pose_forward_delta,
        "phase_raw": float(phase_loss.detach()),
        "phase_weighted": float((TEMP_WEIGHTS["phase_advection"] * phase_loss).detach()),
        "screw_raw": float(screw_loss.detach()),
        "screw_weighted": float((TEMP_WEIGHTS["temporal_screw"] * screw_loss).detach()),
        "phase_active_pixels": int(np.count_nonzero(weight_np)),
        "screw_annulus_pixels": int(np.count_nonzero(np.asarray(margins[pair_index]) < 2.0)),
        "screw_warp_valid_fraction": float(valid.float().mean()),
    }


def _flat_grads(loss, params, *, retain_graph: bool):
    import torch

    grads = torch.autograd.grad(
        loss, [p for _name, p in params], retain_graph=retain_graph, allow_unused=True
    )
    return {
        name: (
            torch.zeros_like(param).reshape(-1)
            if grad is None
            else grad.detach().float().reshape(-1).clone()
        )
        for (name, param), grad in zip(params, grads, strict=True)
    }


def _cosine(a, b) -> float | None:
    import torch

    na = float(torch.linalg.vector_norm(a))
    nb = float(torch.linalg.vector_norm(b))
    if na == 0.0 or nb == 0.0:
        return None
    return float(torch.dot(a, b) / (na * nb))


def _gradient_stats(pred, temp) -> dict[str, Any]:
    import torch

    names = list(pred)
    gp = torch.cat([pred[n] for n in names])
    gt = torch.cat([temp[n] for n in names])
    np_norm = float(torch.linalg.vector_norm(gp))
    nt_norm = float(torch.linalg.vector_norm(gt))
    scale = max(float(gp.abs().max()), float(gt.abs().max()), 1.0)
    active_eps = 1e-12 * scale
    coactive = (gp.abs() > active_eps) & (gt.abs() > active_eps)
    negative = coactive & ((gp * gt) < 0)
    tensor_rows = []
    neg_tensor_params = 0
    active_tensor_params = 0
    for name in names:
        c = _cosine(pred[name], temp[name])
        n = int(pred[name].numel())
        active = float(torch.linalg.vector_norm(pred[name])) > 0.0 and float(
            torch.linalg.vector_norm(temp[name])
        ) > 0.0
        if active:
            active_tensor_params += n
            if c is not None and c < 0.0:
                neg_tensor_params += n
        tensor_rows.append(
            {
                "name": name,
                "numel": n,
                "cosine": c,
                "prediction_norm": float(torch.linalg.vector_norm(pred[name])),
                "temporal_norm": float(torch.linalg.vector_norm(temp[name])),
            }
        )
    total = int(gp.numel())
    coactive_n = int(coactive.sum())
    negative_n = int(negative.sum())
    global_cosine = _cosine(gp, gt)
    negative_all = negative_n / total if total else 0.0
    negative_coactive = negative_n / coactive_n if coactive_n else 0.0
    negative_tensor_fraction = (
        neg_tensor_params / active_tensor_params if active_tensor_params else 0.0
    )
    conflict_exists = bool(
        global_cosine is not None
        and global_cosine <= MATERIAL_COSINE_THRESHOLD_ASSUMPTION
        and negative_all >= MATERIAL_FRACTION_ASSUMPTION
    )
    return {
        "global_cosine": global_cosine,
        "prediction_norm": np_norm,
        "temporal_norm": nt_norm,
        "trunk_scalar_count": total,
        "coactive_scalar_count": coactive_n,
        "coactive_weight_fraction": coactive_n / total if total else 0.0,
        "negative_product_scalar_count": negative_n,
        "negative_product_weight_fraction_all": negative_all,
        "negative_product_weight_fraction_coactive": negative_coactive,
        "negative_cosine_tensor_weight_fraction": negative_tensor_fraction,
        "material_fraction_threshold_assumed": MATERIAL_FRACTION_ASSUMPTION,
        "material_cosine_threshold_assumed": MATERIAL_COSINE_THRESHOLD_ASSUMPTION,
        "conflict_exists_under_preregistered_rule": conflict_exists,
        "per_tensor": tensor_rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        default="experiments/results/v9_cgauge_432_coherent_arm_20260711/levelset_witness_ema_mlx.npz",
    )
    ap.add_argument(
        "--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
    )
    ap.add_argument("--pairs", default="75,225,375,525")
    ap.add_argument(
        "--out",
        default="experiments/results/sps_gradient_separation_probe_20260713/receipt.json",
    )
    args = ap.parse_args()

    import torch

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    np.random.seed(0)

    checkpoint = (REPO / args.checkpoint).resolve()
    gt_cache = (REPO / args.gt_cache).resolve()
    out = (REPO / args.out).resolve()
    if not checkpoint.is_file() or not gt_cache.is_file():
        raise FileNotFoundError("checkpoint or GT cache missing")
    pair_indices = tuple(int(x) for x in args.pairs.split(",") if x.strip())
    if not pair_indices or len(set(pair_indices)) != len(pair_indices):
        raise ValueError("--pairs must contain unique indices")

    t0 = time.time()
    lstars = _stored_npz_memmap(gt_cache, "lstars")
    margins = _stored_npz_memmap(gt_cache, "margins")
    gt_poses = _stored_npz_memmap(gt_cache, "gt_poses")
    n_pairs = int(lstars.shape[0])
    if any(pi <= 0 or pi >= n_pairs for pi in pair_indices):
        raise ValueError("pair indices must be in [1,n_pairs) so phase has a predecessor")

    bank = CurveletBankConfig()
    basis = curvelet_directional_B(bank, max_freq=64.0)
    coords = build_coords(384, 512)
    flat_feats = curvelet_feats(coords, basis).astype(np.float32)
    taper, taper_row = _streaming_taper(margins, flat_feats)
    feats = apply_dseg_aware_fourier_taper(flat_feats, taper).astype(np.float32)
    model, cfg, checkpoint_meta = _load_checkpoint_model(checkpoint, feats)
    parity = forward_parity_against_numpy(model, feats[:257], code_index=2 * pair_indices[0] + 1)
    if not parity["argmax_equal"] or float(parity["cosine_phi"]) < 0.9997:
        raise RuntimeError(f"Torch/NumPy loaded-checkpoint parity failed: {parity}")

    seg, pose = torch_trainer._load_scorers(torch.device("cpu"))
    feats_t = torch.as_tensor(feats, dtype=torch.float32)
    trunk_params = [
        (name, param)
        for name, param in model.named_parameters()
        if name.startswith(TRUNK_PREFIXES)
    ]
    if not trunk_params:
        raise RuntimeError("no trunk parameters selected")
    phase_geom = GroundHomographyGeom.eon(native_hw=(384, 512), pitch=-0.01)

    aggregate_seg = {name: torch.zeros(param.numel()) for name, param in trunk_params}
    aggregate_pose = {name: torch.zeros(param.numel()) for name, param in trunk_params}
    aggregate_pred = {name: torch.zeros(param.numel()) for name, param in trunk_params}
    aggregate_temp = {name: torch.zeros(param.numel()) for name, param in trunk_params}
    pair_rows = []
    for pair_index in pair_indices:
        model.zero_grad(set_to_none=True)
        prediction_seg, prediction_pose, temporal, losses = _losses_for_pair(
            model,
            cfg,
            feats_t,
            pair_index,
            (lstars, margins, gt_poses),
            (seg, pose),
            phase_geom,
        )
        seg_grad = _flat_grads(prediction_seg, trunk_params, retain_graph=True)
        pose_grad = _flat_grads(prediction_pose, trunk_params, retain_graph=True)
        pred_grad = {
            name: seg_grad[name] + pose_grad[name] for name, _param in trunk_params
        }
        temp_grad = _flat_grads(temporal, trunk_params, retain_graph=False)
        for name, _param in trunk_params:
            aggregate_seg[name] += seg_grad[name]
            aggregate_pose[name] += pose_grad[name]
            aggregate_pred[name] += pred_grad[name]
            aggregate_temp[name] += temp_grad[name]
        pair_rows.append(
            {
                "pair_index": pair_index,
                "losses": losses,
                "gradient_conflict": {
                    "seg_vs_temporal": _gradient_stats(seg_grad, temp_grad),
                    "pose_vs_temporal": _gradient_stats(pose_grad, temp_grad),
                    "fully_armed_seg_plus_pose_vs_temporal": _gradient_stats(
                        pred_grad, temp_grad
                    ),
                },
            }
        )
        del prediction_seg, prediction_pose, temporal, seg_grad, pose_grad, pred_grad, temp_grad

    mechanism_stats = {
        "seg_vs_temporal": _gradient_stats(aggregate_seg, aggregate_temp),
        "pose_vs_temporal": _gradient_stats(aggregate_pose, aggregate_temp),
        "fully_armed_seg_plus_pose_vs_temporal": _gradient_stats(
            aggregate_pred, aggregate_temp
        ),
    }
    primary_stats = mechanism_stats["seg_vs_temporal"]
    live_temporal_weights = {"phase_advection": 0.0, "temporal_screw": 0.0}
    live_stats = {
        "global_cosine": None,
        "prediction_norm": primary_stats["prediction_norm"],
        "temporal_norm": 0.0,
        "reason": "epoch 275 precedes screw start 450 and phase start 726; weighted temporal gradient is exactly zero",
        "conflict_exists": False,
    }
    verdict = (
        "GO_TO_LOCAL_AB_DESIGN"
        if primary_stats["conflict_exists_under_preregistered_rule"]
        else "NO_GO_FOR_SPS_SEPARATION_ON_THIS_PROBE"
    )
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "tool": "tools/probe_sps_gradient_role_conflict.py",
        "authority": AUTHORITY,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "research_only": True,
        "cost_usd": 0.0,
        "heavy_launch": False,
        "seed": 0,
        "torch_threads": 1,
        "backend": "deterministic Torch CPU twin + NumPy-fp32/float64 deploy parity",
        "mlx_backend_attempt": {
            "available": False,
            "reason": "headless sandbox has no Metal device; MLX array evaluation raised metal::load_device No Metal device available",
            "verdict_effect": "fail-closed from MLX authority; Torch result is local parity-probe only",
        },
        "repo_git_sha": _git_sha(),
        "checkpoint": {
            "path": str(checkpoint.relative_to(REPO)),
            "bytes": checkpoint.stat().st_size,
            "sha256": _sha256(checkpoint),
            **checkpoint_meta,
            "source_run_read_only": True,
        },
        "gt_cache": {
            "path": str(gt_cache.relative_to(REPO)),
            "bytes": gt_cache.stat().st_size,
            "n_pairs_available": n_pairs,
            "zip_stored_mmap": True,
        },
        "probe_scope": {
            "pair_indices": list(pair_indices),
            "n_pairs": len(pair_indices),
            "selection": "deterministic four-stratum interior indices, preregistered in argv",
            "n600_owed_for_verdict": True,
            "verdict_scope": "instance-probe at one real checkpoint and four pairs; not family/paradigm",
        },
        "gradient_scope": {
            "parameter_prefixes": list(TRUNK_PREFIXES),
            "parameter_names": [name for name, _param in trunk_params],
            "excludes": ["code", "film", "out_sdf", "out_tex", "palette", "pose_carrier"],
            "prediction_weights": PRED_WEIGHTS,
            "counterfactual_temporal_weights": TEMP_WEIGHTS,
            "live_temporal_weights_at_checkpoint": live_temporal_weights,
        },
        "apparatus_validation": {
            "torch_numpy_loaded_checkpoint_forward_parity": parity,
            "differentiable_pose_yuv6_vs_official_forward_max_abs": max(
                row["losses"]["pose_differentiable_vs_official_forward_max_abs"]
                for row in pair_rows
            ),
            "pose_gradient_note": (
                "upstream frame_utils.rgb_to_yuv6 is evaluation-only @torch.no_grad; "
                "probe uses an op-identical differentiable twin and records forward parity"
            ),
            "active_taper": taper_row,
            "expected_run_log_taper_rounded": {"min": 0.9536, "max": 1.0462, "mean": 1.0},
        },
        "live_program_measurement": live_stats,
        "counterfactual_fully_armed_mechanism_measurement": mechanism_stats,
        "per_pair": pair_rows,
        "verdict": verdict,
        "verdict_scope": "FORMULATION-INSTANCE-PROBE",
        "reformulation_queue": (
            "none unless n600 or a second checkpoint reverses the measured sign"
            if verdict.startswith("NO_GO")
            else "test loss-routing-only PCGrad/scalarization control before architectural stream duplication"
        ),
        "elapsed_seconds": time.time() - t0,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out)
    print(json.dumps({
        "receipt": str(out.relative_to(REPO)),
        "verdict": verdict,
        "global_cosine": primary_stats["global_cosine"],
        "negative_product_weight_fraction_all": primary_stats["negative_product_weight_fraction_all"],
        "negative_cosine_tensor_weight_fraction": primary_stats["negative_cosine_tensor_weight_fraction"],
        "elapsed_seconds": receipt["elapsed_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
