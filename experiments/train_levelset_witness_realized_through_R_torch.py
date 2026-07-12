#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Torch/CUDA training entry point for the typed V9 CGauge #432 program.

The production path always derives configuration from ``spec_v9_cgauge``; arbitrary
hand-written trainer flags are intentionally not accepted.  This is a real frozen-scorer
gradient path (SegNet and PoseNet consume the rendered frames through R), not a cached-label
surrogate.  Local ``--verify-only`` is CPU-light and never instantiates the scorers.

Authority remains unchanged: training rows are ``[contest-CUDA training-advisory]`` and
NON-PROMOTABLE.  Only byte-closed ``upstream/evaluate.py`` results can move the pointer.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (REPO, REPO / "src", REPO / "experiments", REPO / "upstream"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tac.boundary_math.lever_b_levelset_generator import (
    CurveletBankConfig,
    build_coords,
    build_static_core_phi_target,
    curvelet_directional_B,
    curvelet_feats,
)
from tac.cuda_levelset_training import (
    CudaLevelSetConfig,
    DeterministicPairCursor,
    TorchPoseCarrier,
    TorchLevelSetWitness,
    apply_torch_execution_policy,
    area_constraint_torch,
    chroma_boundary_loss,
    compile_identity_probe,
    contest_r,
    eikonal_and_length,
    forward_parity_against_numpy,
    homography_grid_from_xi,
    island_birth_from_signed_torch,
    persistence_topology_loss_torch,
    clip_grad_groups,
    realized_signed_margin,
    round_ste,
    select_torch_execution_policy,
    parameter_groups,
    pose_objective_torch,
    structured_sdf_prefit,
    warp_field_persist_torch,
    weight_entropy_rate_term_torch,
    witness_tie_coordinate_torch,
)
from tac.witness_run_artifacts import (
    TORCH_EMA_PT,
    TORCH_RESUME_PT,
    TORCH_RUN_MANIFEST_JSON,
    TORCH_TRAIN_RESULT_JSON,
    TORCH_TRAJECTORY_JSONL,
)
from tac.witness_training_contract import (
    cuda_v9_port_receipt,
    curriculum_stage,
    loss_terms_row,
)

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _flag_map(argv: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    i = 2  # python + trainer path
    while i < len(argv):
        tok = argv[i]
        if not tok.startswith("--"):
            i += 1
            continue
        if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
            out[tok] = argv[i + 1]
            i += 2
        else:
            out[tok] = True
            i += 1
    return out


def derive_config(args) -> tuple[dict[str, Any], str, tuple[str, ...]]:
    from tac.witness_dsl.spec_v9_cgauge import compile_v9_cgauge_432_launch_config

    compiled = compile_v9_cgauge_432_launch_config(
        args.gt_cache, num_pairs=args.num_pairs, epochs=args.epochs, out_dir=args.out_dir
    )
    argv = tuple(compiled.typed.to_program().compile_trainer_argv())
    flags = _flag_map(argv)
    payload = json.dumps({"argv": argv, "typed_config_hash": compiled.typed.typed_config_hash()}, sort_keys=True)
    return flags, hashlib.sha256(payload.encode()).hexdigest(), argv


def _atomic_torch_save(obj, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _atomic_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _checkpoint_blob(
    model,
    ema,
    optimizer,
    epoch: int,
    config_hash: str,
    argv: tuple[str, ...],
    *,
    pair_cursor: DeterministicPairCursor | None = None,
    controller_state: Mapping[str, Any] | None = None,
) -> dict:
    import torch

    return {
        "schema": "v9_cgauge_torch_resume_v2",
        "epoch": int(epoch),
        "model": model.state_dict(),
        "ema": ema.state_dict(),  # inference/deploy authority is the EMA shadow
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "config_hash": config_hash,
        "dsl_argv": list(argv),
        "pair_cursor": pair_cursor.state_dict() if pair_cursor is not None else None,
        "controller_state": copy.deepcopy(dict(controller_state or {})),
    }


def _restore(
    blob,
    model,
    ema,
    optimizer,
    expected_hash: str,
    *,
    pair_cursor: DeterministicPairCursor | None = None,
    controller_state: dict[str, Any] | None = None,
) -> int:
    import torch

    if blob.get("config_hash") != expected_hash:
        raise ValueError("resume config hash differs from the typed V9 CGauge program; refusing drift")
    model.load_state_dict(blob["model"], strict=True)
    ema.load_state_dict(blob["ema"], strict=True)
    optimizer.load_state_dict(blob["optimizer"])
    torch.set_rng_state(blob["torch_rng"])
    if torch.cuda.is_available() and blob.get("cuda_rng"):
        torch.cuda.set_rng_state_all(blob["cuda_rng"])
    np.random.set_state(blob["numpy_rng"])
    random.setstate(blob["python_rng"])
    if pair_cursor is not None and blob.get("pair_cursor") is not None:
        pair_cursor.load_state_dict(blob["pair_cursor"])
    if controller_state is not None:
        controller_state.clear()
        controller_state.update(copy.deepcopy(blob.get("controller_state", {})))
    return int(blob["epoch"])


def _ema_update(ema, model, decay: float) -> None:
    with __import__("torch").no_grad():
        for dst, src in zip(ema.parameters(), model.parameters(), strict=True):
            dst.mul_(decay).add_(src, alpha=1.0 - decay)
        for dst, src in zip(ema.buffers(), model.buffers(), strict=True):
            dst.copy_(src)


def _load_scorers(device):
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path

    from tac.boundary_math.seg_core import load_real_segnet

    seg = load_real_segnet(str(device)).eval()
    dn = DistortionNet().eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, device)
    pose = dn.posenet.to(device).eval()
    for net in (seg, pose):
        for p in net.parameters():
            p.requires_grad_(False)
    return seg, pose


def _pose6(pose_net, frames_nhwc):
    # frames: (B,2,H,W,3), scorer preprocess is part of the real path.
    x = frames_nhwc.permute(0, 1, 4, 2, 3).contiguous()
    out = pose_net(pose_net.preprocess_input(x))
    pose = out["pose"] if isinstance(out, dict) else out
    half = next((h.out // 2 for h in pose_net.hydra.heads if h.name == "pose"), pose.shape[-1] // 2)
    return pose[:, :half]


def _required_typed_flags(flags: Mapping[str, Any], names: tuple[str, ...]) -> None:
    missing = [name for name in names if name not in flags]
    if missing:
        raise ValueError(
            "active V9 mechanism lacks typed DSL companion values: " + ", ".join(missing)
        )


def _attach_generated_pose_carrier(model, flags, gt_poses, native_hw, device):
    """Attach the typed generated/table pose carrier before EMA and optimizer creation."""
    if not bool(flags.get("--pose-carrier", False)):
        return None, {"active": False}
    needed = (
        "--pose-carrier-source", "--pose-carrier-residual-mode",
        "--pose-carrier-residual-scale", "--pose-carrier-s-t",
        "--pose-carrier-s-r", "--pose-carrier-pitch",
    )
    _required_typed_flags(flags, needed)
    if flags["--pose-carrier-source"] != "generated":
        raise ValueError("Torch V9 carrier currently requires typed source=generated")
    if flags["--pose-carrier-residual-mode"] != "table":
        raise ValueError("Torch V9 carrier currently requires typed residual-mode=table")
    from tac.boundary_math.warp_real_luma_frame0 import (
        GroundHomographyGeom,
        xi_from_pose_calibration,
    )

    native_hw = tuple(int(x) for x in native_hw)
    geom = GroundHomographyGeom.eon(
        native_hw=native_hw, pitch=float(flags["--pose-carrier-pitch"])
    )
    xi_stored = np.stack([
        xi_from_pose_calibration(
            np.asarray(pose),
            float(flags["--pose-carrier-s-t"]),
            float(flags["--pose-carrier-s-r"]),
            float(flags["--pose-carrier-pitch"]),
        )
        for pose in np.asarray(gt_poses)
    ]).astype(np.float32)
    carrier = TorchPoseCarrier.build(
        xi_stored,
        geom,
        residual_scale=float(flags["--pose-carrier-residual-scale"]),
    ).to(device)
    model.pose_carrier = carrier
    return carrier, {
        "active": True,
        "source": "generated",
        "residual_mode": "table",
        "native_hw": list(native_hw),
        "s_t": float(flags["--pose-carrier-s-t"]),
        "s_r": float(flags["--pose-carrier-s-r"]),
        "pitch": float(flags["--pose-carrier-pitch"]),
        "n_pairs": int(len(gt_poses)),
    }


def _run_structured_prefit(model, flags, lstars, feats, *, seed: int, is_resume: bool):
    """Run the active typed structured-core prefit only on a fresh model."""
    if not bool(flags.get("--structured-init", False)):
        return {"active": False, "applied": False}
    needed = (
        "--structured-init-include-lane", "--structured-init-thresh",
        "--structured-init-steps", "--structured-init-lr",
        "--structured-init-subsample", "--structured-init-sdf-clip",
    )
    _required_typed_flags(flags, needed)
    if is_resume:
        return {"active": True, "applied": False, "reason": "resume_preserves_checkpoint"}
    cfg = model.cfg
    if tuple(np.asarray(lstars).shape[1:]) != (cfg.render_h, cfg.render_w):
        raise ValueError("structured-init L* shape differs from the typed render geometry")
    phi_hwk, roles, meta = build_static_core_phi_target(
        np.asarray(lstars),
        n_classes=cfg.n_classes,
        include_lane=bool(flags["--structured-init-include-lane"]),
        static_thresh=float(flags["--structured-init-thresh"]),
    )
    clip = float(flags["--structured-init-sdf-clip"])
    target = np.clip(phi_hwk.reshape(-1, cfg.n_classes), -clip, clip).astype(np.float32)
    import torch

    target_t = torch.as_tensor(target, device=feats.device)
    row = structured_sdf_prefit(
        model,
        feats,
        target_t,
        steps=int(flags["--structured-init-steps"]),
        lr=float(flags["--structured-init-lr"]),
        subsample=int(flags["--structured-init-subsample"]),
        seed=int(seed),
    )
    with torch.no_grad():
        _rgb, pred = model(feats, torch.zeros(1, dtype=torch.long, device=feats.device))
    disagree = float(np.mean(
        pred[0].argmax(-1).detach().cpu().numpy() != target.argmax(-1)
    ))
    return {
        "active": True, "applied": True, **row,
        "direct_argmax_disagree": disagree,
        "roles": roles.as_dict(),
        **{k: v for k, v in meta.items() if k != "roles"},
    }


def _generated_pose_pair_dispatch(model, feats, pair_indices, pose_carrier, cfg):
    """MLX-authority generated/table dispatch: plain f0 up->warp->R-down, f1 witness R.

    ``pair_indices`` indexes pairs, not frame codes. The attached carrier is the only
    consumer of its trainable dxi. Frame1 never reads dxi, preserving the SegNet-free
    frame0 / scorer-visible frame1 separation.
    """
    import torch
    import torch.nn.functional as F

    pair_indices = torch.as_tensor(pair_indices, device=feats.device, dtype=torch.long)
    code0 = 2 * pair_indices
    code1 = code0 + 1
    raw0, _phi0 = model(feats, code0)
    raw1, phi1 = model(feats, code1)
    n = pair_indices.numel()
    raw0 = raw0.reshape(n, cfg.render_h, cfg.render_w, 3)
    raw1 = raw1.reshape(n, cfg.render_h, cfg.render_w, 3)
    native0 = F.interpolate(
        raw0.permute(0, 3, 1, 2),
        size=(cfg.camera_h, cfg.camera_w),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1)
    native0 = torch.clamp(round_ste(native0), 0.0, 255.0)
    warped0 = pose_carrier(native0, pair_indices)
    scored0 = F.interpolate(
        warped0.permute(0, 3, 1, 2),
        size=(cfg.render_h, cfg.render_w),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1).contiguous()
    scored1 = contest_r(raw1, output_hw=(cfg.render_h, cfg.render_w))
    frames = torch.stack((scored0, scored1), dim=1)
    return frames, phi1.reshape(n, cfg.render_h, cfg.render_w, cfg.n_classes)


def _accumulated_pair_step(
    model,
    optimizer,
    pair_indices,
    loss_builder,
    *,
    grad_clip: float,
):
    """Mean a complete pair chunk, then atomically accept or reject one update.

    This mirrors the MLX authority: every pair in the chunk is evaluated exactly
    once, the mean loss is formed before backward, and the finite/spike guard is
    applied to the chunk as a whole.  Telemetry therefore counts accepted chunks,
    not accepted members within a partially retained chunk.
    """
    import torch

    optimizer.zero_grad(set_to_none=True)
    losses = []
    for pair_index in pair_indices:
        loss = loss_builder(int(pair_index))
        if loss is None:
            optimizer.zero_grad(set_to_none=True)
            return {
                "weights_stepped": False, "accepted": 0, "attempted": 1,
                "accepted_frac": 0.0, "loss_mean": None, "group_norms": {},
                "pair_count": len(pair_indices),
            }
        losses.append(loss)
    if not losses:
        optimizer.zero_grad(set_to_none=True)
        return {
            "weights_stepped": False, "accepted": 0, "attempted": 0,
            "accepted_frac": 0.0, "loss_mean": None, "group_norms": {},
            "pair_count": 0,
        }
    differentiable_mean = torch.stack(losses).mean()
    loss_mean = differentiable_mean.detach()
    if not bool(torch.isfinite(loss_mean)):
        optimizer.zero_grad(set_to_none=True)
        return {
            "weights_stepped": False, "accepted": 0, "attempted": 1,
            "accepted_frac": 0.0, "loss_mean": loss_mean, "group_norms": {},
            "pair_count": len(losses),
        }
    differentiable_mean.backward()
    group_norms = clip_grad_groups(parameter_groups(model), float(grad_clip))
    optimizer.step()
    return {
        "weights_stepped": True, "accepted": 1, "attempted": 1,
        "accepted_frac": 1.0, "loss_mean": loss_mean, "group_norms": group_norms,
        "pair_count": len(losses),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--out-dir", default="experiments/results/v9_cgauge_cuda")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume-from")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--compile-probe", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    if any(str(out.resolve()).startswith(x) for x in _FORBIDDEN_TMP):
        raise ValueError("out-dir is a tmp-class path; use the SSD/repo tier")

    flags, cfg_hash, dsl_argv = derive_config(args)
    bank = CurveletBankConfig()
    B = curvelet_directional_B(bank, max_freq=float(flags["--max-bank-freq"]))
    coords = build_coords(int(flags["--render-h"]), int(flags["--render-w"]))
    feats_np = curvelet_feats(coords, B)
    cfg = CudaLevelSetConfig(
        n_pairs=args.num_pairs, in_feat=feats_np.shape[1], hidden_dim=int(flags["--hidden-dim"]),
        n_hidden=int(flags["--n-hidden"]), mod_dim=int(flags["--mod-dim"]),
        activation=str(flags["--activation"]), hosc_beta=float(flags["--hosc-beta"]),
        hosc_omega=float(flags["--hosc-omega"]), softmax_temp=float(flags["--softmax-temp-start"]),
        chroma=bool(flags.get("--chroma", False)), render_h=int(flags["--render-h"]),
        render_w=int(flags["--render-w"]),
    )

    import torch
    if args.device == "cuda" and not torch.cuda.is_available() and not args.verify_only:
        raise RuntimeError("CUDA requested but unavailable (fail closed; use --verify-only for $0 local proof)")
    device = torch.device("cpu" if args.verify_only else args.device)
    execution_policy = select_torch_execution_policy(device)
    if device.type == "cuda":
        apply_torch_execution_policy(execution_policy)
    else:
        torch.use_deterministic_algorithms(True)
    torch.manual_seed(int(flags["--seed"]))
    np.random.seed(int(flags["--seed"]))
    random.seed(int(flags["--seed"]))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(flags["--seed"]))
    model = TorchLevelSetWitness.build(cfg, seed=int(flags["--seed"])).to(device)
    parity = forward_parity_against_numpy(model, feats_np[: min(257, len(feats_np))])
    row = {"stage": "cuda_numpy_forward_parity", "backend": str(device), **parity,
           "measured": True, "promotion_eligible": False}
    print(json.dumps(row), flush=True)
    coverage = cuda_v9_port_receipt()
    print(json.dumps({"stage": "cuda_v9_port_coverage", **coverage}), flush=True)

    compile_probe_result: dict[str, Any] | None = None
    if args.compile_probe:
        f = torch.as_tensor(feats_np[:64], device=device)
        ci = torch.tensor([0], device=device)
        compile_probe_result = compile_identity_probe(
            model, f, ci, lambda rgb, phi: rgb.square().mean() + phi.square().mean()
        )
        print(json.dumps({"stage": "backend_fp_reorder_probe", "backend": str(device),
                          **compile_probe_result}), flush=True)
        if not compile_probe_result.get("adoptable", False):
            print(json.dumps({"stage": "cuda_compile_policy", "compiled_training": False,
                              "reason": "functional argmax/cosine parity gate did not pass"}), flush=True)

    if args.verify_only:
        return 0 if parity["argmax_equal"] and parity["cosine_phi"] >= 0.9997 else 2

    if coverage["status"] != "COMPLETE_1_TO_1":
        raise RuntimeError(
            "NO-FAKE REFUSAL: active V9 CUDA control semantics are not 1:1; "
            f"unclosed surfaces: {coverage['blockers']}"
        )

    z = np.load(args.gt_cache, allow_pickle=False)
    if int(z["n_pairs"]) < args.num_pairs:
        raise ValueError("GT cache contains fewer pairs than requested")
    lstars = z["lstars"][: args.num_pairs]
    margins = z["margins"][: args.num_pairs]
    gt_f1 = z["gt_f1"][: args.num_pairs]
    gt_poses = z["gt_poses"][: args.num_pairs]
    # Generated source never reads/materializes gt_f0. Camera geometry is shared
    # by the already-required gt_f1 array and the canonical receiver dimensions.
    native_hw = tuple(int(x) for x in gt_f1.shape[1:3])
    if native_hw != (cfg.camera_h, cfg.camera_w):
        raise ValueError(
            f"GT camera geometry {native_hw} differs from Torch receiver {(cfg.camera_h, cfg.camera_w)}"
        )
    feats = torch.as_tensor(feats_np, device=device)
    pose_carrier, pose_carrier_row = _attach_generated_pose_carrier(
        model, flags, gt_poses, native_hw, device
    )
    print(json.dumps({"stage": "pose_carrier", **pose_carrier_row}), flush=True)
    resume_path = Path(args.resume_from) if args.resume_from else out / TORCH_RESUME_PT
    if resume_path.is_dir():
        resume_path = resume_path / TORCH_RESUME_PT
    resume_will_load = bool(args.resume_from or resume_path.exists())
    if flags.get("--palette-anchor", False) and not resume_will_load:
        import torch.nn.functional as F

        sums = np.zeros((cfg.n_classes, 3), np.float64)
        cnts = np.zeros(cfg.n_classes, np.float64)
        for pi in range(min(args.num_pairs, 64)):
            frame = torch.from_numpy(np.asarray(gt_f1[pi], np.float32)).permute(2, 0, 1)[None]
            small = F.interpolate(
                frame, size=(cfg.render_h, cfg.render_w), mode="bilinear", align_corners=False
            )[0].permute(1, 2, 0).numpy()
            for cls in range(cfg.n_classes):
                mask = lstars[pi] == cls
                if mask.any():
                    sums[cls] += small[mask].sum(0)
                    cnts[cls] += int(mask.sum())
        mean = np.where(cnts[:, None] > 0, sums / np.maximum(cnts[:, None], 1), 127.0)
        clipped = np.clip(mean / 255.0, 1e-3, 1.0 - 1e-3)
        palette = np.log(clipped / (1.0 - clipped)).astype(np.float32)
        with torch.no_grad():
            model.palette.copy_(torch.as_tensor(palette, device=device))
    structured_row = _run_structured_prefit(
        model,
        flags,
        lstars,
        feats,
        seed=int(flags["--seed"]),
        is_resume=resume_will_load,
    )
    print(json.dumps({"stage": "structured_init", **structured_row}), flush=True)
    counts = np.bincount(lstars.reshape(-1), minlength=cfg.n_classes).astype(np.float64)
    priors = counts / counts.sum()
    la_tau = float(flags.get("--logit-adjust-loss-tau", 0.0))
    la_spec = str(flags.get("--logit-adjust-classes", "all"))
    la_allowed = (
        tuple(range(cfg.n_classes))
        if la_spec.lower() == "all"
        else tuple(int(x) for x in la_spec.split(",") if x.strip())
    )
    logit_offsets_np = np.zeros(cfg.n_classes, np.float32)
    if la_tau != 0.0:
        raw = la_tau * np.log(np.maximum(priors, 1e-8))
        for cls in la_allowed:
            logit_offsets_np[cls] = raw[cls]
    persist_classes = tuple(
        int(x) for x in str(flags.get("--persistence-classes", "3")).split(",")
        if x.strip() and x.strip().lower() != "auto"
    )
    from tac.boundary_math.island_protection import (
        build_island_masks,
        identify_island_classes,
        island_persistence_weight,
    )
    island_detection = identify_island_classes(lstars, n_classes=cfg.n_classes)
    island_weight_cache: dict[int, np.ndarray] = {}
    from tac.canonical_equations.chan_vese_area_constraint_birth_balance_20260708 import (
        area_constraint_lambda,
    )
    area_classes = tuple(
        int(x) for x in str(flags.get("--area-constraint-classes", "1,3")).split(",") if x.strip()
    )
    area_lambdas = {
        cls: area_constraint_lambda(
            float(priors[cls]),
            birth_force=float(flags.get("--area-constraint-birth-force", 1.0)),
            tolerance=float(flags.get("--area-constraint-tolerance", 0.25)),
        )
        for cls in area_classes
    } if flags.get("--area-constraint-birth", False) else {}
    lane_band = None
    if flags.get("--lane-render-band", False):
        from tac.boundary_math.dash_comb import build_combed_lane_band_priors

        lane_priors, lane_comb_fit = build_combed_lane_band_priors(
            lstars,
            gt_poses,
            lane_cls=1,
            softness=float(flags.get("--lane-band-softness", 1.0)),
            dash_forward_max_m=float(flags.get("--lane-band-dash-forward-max-m", 55.0)),
            comb_softness_m=float(flags.get("--lane-band-comb-softness-m", 0.3)),
        )
        lane_band = {
            "priors": lane_priors,
            "tau": float(flags.get("--lane-band-tau", 0.85)),
            "eps": float(flags.get("--lane-band-eps", 0.35)),
            "weight": float(flags.get("--lane-band-weight", 1.0)),
            "start_epoch": int(flags.get("--lane-band-start-epoch", 500)),
            "comb": {
                "period_m": float(lane_comb_fit.period_m),
                "duty": float(lane_comb_fit.duty),
                "ego_scale": float(lane_comb_fit.scale),
            },
        }
    phase_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    phase_weight = float(flags.get("--seg-phase-advect-weight", 0.0))
    phase_start = int(flags.get("--seg-phase-advect-start-epoch", 0))
    if phase_weight > 0.0:
        from tac.boundary_math.phase_primitives import (
            advect_tie_field_numpy,
            cross_scored_frame_xi_interp,
            gt_tie_targets_numpy,
        )
        from tac.boundary_math.warp_real_luma_frame0 import (
            GroundHomographyGeom,
            xi_from_pose_calibration,
        )

        phase_geom = GroundHomographyGeom.eon(
            native_hw=(cfg.render_h, cfg.render_w),
            pitch=float(flags.get("--gfc-pitch", -0.01)),
        )
        phase_xi = [
            xi_from_pose_calibration(
                gt_poses[pi],
                float(flags.get("--gfc-s-t", -0.003224707899359239)),
                float(flags.get("--gfc-s-r", 0.0)),
                float(flags.get("--gfc-pitch", -0.01)),
            )
            for pi in range(args.num_pairs)
        ]
        phase_classes = {
            int(x) for x in str(flags.get("--seg-phase-advect-classes", "0,1,2")).split(",")
            if x.strip()
        }
        phase_band = float(flags.get("--seg-phase-advect-band", 2.0))

        def phase_provider(pair_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if pair_index in phase_cache:
                return phase_cache[pair_index]
            _t, direction, _active = gt_tie_targets_numpy(
                lstars[pair_index], margins[pair_index], band=phase_band
            )
            if pair_index == 0:
                ref = np.full_like(direction, -1.0, dtype=np.float32)
                weight = np.zeros_like(direction, dtype=np.float32)
            else:
                prev_t, _prev_dir, prev_active = gt_tie_targets_numpy(
                    lstars[pair_index - 1], margins[pair_index - 1], band=phase_band
                )
                xi_cross = cross_scored_frame_xi_interp(
                    phase_xi[pair_index - 1], phase_xi[pair_index]
                )
                ref_warp = advect_tie_field_numpy(
                    np.where(prev_t >= 0.0, prev_t, 0.0).astype(np.float32),
                    xi_cross,
                    phase_geom,
                )
                active_warp = advect_tie_field_numpy(
                    prev_active.astype(np.float32), xi_cross, phase_geom
                ) >= 0.5
                ref = np.where(active_warp, ref_warp, -1.0).astype(np.float32)
                weight = (
                    (margins[pair_index] < phase_band)
                    & np.isin(lstars[pair_index], list(phase_classes))
                    & active_warp
                ).astype(np.float32)
            phase_cache[pair_index] = (ref, direction.astype(np.float32), weight)
            return phase_cache[pair_index]
    temporal_weight = float(flags.get("--seg-temporal-screw-weight", 0.0))
    temporal_start = int(flags.get("--seg-temporal-screw-start-epoch", 0))
    temporal_grid_cache: dict[int, tuple[Any, Any]] = {}
    temporal_xi: list[np.ndarray] = []
    temporal_geom = None
    temporal_classes = tuple(
        int(x) for x in str(flags.get("--seg-temporal-screw-classes", "0,1,2")).split(",")
        if x.strip()
    )
    if temporal_weight > 0.0:
        from tac.boundary_math.warp_real_luma_frame0 import (
            GroundHomographyGeom,
            xi_from_pose_calibration,
        )

        temporal_geom = GroundHomographyGeom.eon(
            native_hw=(cfg.render_h, cfg.render_w),
            pitch=float(flags.get("--gfc-pitch", -0.01)),
        )
        temporal_xi = [
            xi_from_pose_calibration(
                gt_poses[pair_index],
                float(flags.get("--gfc-s-t", -0.003224707899359239)),
                float(flags.get("--gfc-s-r", 0.0)),
                float(flags.get("--gfc-pitch", -0.01)),
            )
            for pair_index in range(args.num_pairs)
        ]
    seg, pose = _load_scorers(device)
    ema = copy.deepcopy(model).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(flags["--lr"]),
                                  betas=(0.9, float(flags["--adam-beta2"])),
                                  weight_decay=float(flags["--weight-decay"]))
    pair_cursor = DeterministicPairCursor(args.num_pairs, seed=int(flags["--seed"]))
    controller_state: dict[str, Any] = {}
    start = 0
    if resume_will_load:
        start = _restore(
            torch.load(resume_path, map_location=device, weights_only=False),
            model, ema, optimizer, cfg_hash,
            pair_cursor=pair_cursor, controller_state=controller_state,
        )
    out.mkdir(parents=True, exist_ok=True)
    _atomic_json({"schema": "v9_cgauge_cuda_run_manifest_v1", "dsl_argv": list(dsl_argv),
                  "config_hash": cfg_hash, "device": str(device), "seed": int(flags["--seed"]),
                  "authority": "[contest-CUDA training-advisory] NON-PROMOTABLE",
                  "execution_policy": execution_policy.__dict__,
                  "compile_policy": "auto_adopt_after_functional_argmax_cosine_probe",
                  "fp_reorder_probe": compile_probe_result or {
                      "backend": str(device), "status": "UNMEASURED",
                  },
                  "created_at_utc": _utc()}, out / TORCH_RUN_MANIFEST_JSON)

    ckpt_every = int(flags["--ckpt-every"])
    ema_decay = float(flags["--ema-decay"])
    prev_stage = curriculum_stage(start, flags)
    t0 = time.time()
    for epoch in range(start + 1, args.epochs + 1):
        stage = curriculum_stage(epoch, flags)
        if stage != prev_stage:
            # Seal the COMPLETED prior stage before the first update of the new
            # stage.  This is both crash insurance and an independent A/B byte-
            # close surface; the filename encodes the last completed epoch.
            boundary_blob = _checkpoint_blob(
                model, ema, optimizer, epoch - 1, cfg_hash, dsl_argv,
                pair_cursor=pair_cursor, controller_state=controller_state,
            )
            _atomic_torch_save(
                boundary_blob,
                out / "stage_checkpoints" / f"ep{epoch - 1:05d}_{prev_stage}.pt",
            )
            prev_stage = stage
        model.train()
        progress = epoch / max(args.epochs, 1)
        with torch.no_grad():
            model.softmax_temp.fill_(
                float(flags["--softmax-temp-start"])
                + progress
                * (float(flags["--softmax-temp-end"]) - float(flags["--softmax-temp-start"]))
            )
            model.hosc_beta.fill_(
                float(flags["--hosc-beta"])
                + progress * (float(flags["--hosc-beta-end"]) - float(flags["--hosc-beta"]))
            )
        if pose_carrier is None:
            raise RuntimeError("active V9 generated pose carrier was not attached")
        pair_cursor.begin_epoch(epoch)
        ep_acc = ep_tot = 0
        chunk_index = 0
        while not pair_cursor.epoch_complete():
            chunk = pair_cursor.next_epoch_indices(int(flags["--accum-pairs"]))
            chunk_np = np.asarray(chunk, dtype=np.int64)
            # MAX-throughput steady state: exactly one batched witness/carrier
            # dispatch and one batch through each frozen scorer per accum chunk.
            frames, phi = _generated_pose_pair_dispatch(
                model, feats, chunk, pose_carrier, cfg
            )
            seg_in = seg.preprocess_input(frames.permute(0, 1, 4, 2, 3).contiguous())
            logits = seg(seg_in)
            target = torch.as_tensor(lstars[chunk_np], device=device).long()
            offsets = torch.as_tensor(logit_offsets_np, device=device, dtype=logits.dtype)
            adjusted_logits = logits + offsets[None, :, None, None]
            tau = max(0.31, 0.31 ** (epoch / max(args.epochs, 1)))
            seg_per_pixel = (
                tau * torch.logsumexp(adjusted_logits / tau, dim=1)
                - adjusted_logits.gather(1, target[:, None]).squeeze(1)
            )
            gt_margin = torch.as_tensor(
                margins[chunk_np], device=device, dtype=seg_per_pixel.dtype
            )
            seg_weight = 1.0 + 4.0 * torch.exp(-gt_margin.clamp_min(0.0))
            seg_loss = (seg_per_pixel * seg_weight).mean()
            pose6 = _pose6(pose, frames)
            pose_target = torch.as_tensor(
                gt_poses[chunk_np], device=device, dtype=pose6.dtype
            )
            pose_loss = pose_objective_torch(pose6, pose_target)
            eik, length = eikonal_and_length(phi)
            seg_contrib = float(flags["--w-seg"]) * seg_loss
            pose_contrib = float(flags["--w-pose"]) * pose_loss
            eik_contrib = float(flags["--eikonal-weight"]) * eik
            length_contrib = float(flags["--length-weight"]) * length
            raw_logits_nhwc = logits.permute(0, 2, 3, 1).contiguous()
            signed = realized_signed_margin(raw_logits_nhwc, target)
            for pi in chunk:
                if pi not in island_weight_cache:
                    masks = build_island_masks(
                        lstars[pi], island_detection.lane_cls,
                        island_detection.movable_cls, dilate_px=1,
                    )
                    island_weight_cache[pi] = island_persistence_weight(
                        masks.any_mask,
                        kind=str(flags.get("--amplify-persist", "inverse_thickness")),
                    )
            island_weight = torch.as_tensor(
                np.stack([island_weight_cache[pi] for pi in chunk]),
                device=device, dtype=signed.dtype,
            )
            amplify = float(flags.get("--amplify-weight", 0.0)) * island_birth_from_signed_torch(
                signed, island_weight, float(flags.get("--amplify-margin-target", 1.0)),
                form=str(flags.get("--amplify-form", "hinge")),
            )
            persist_scale = min(
                1.0, epoch / max(1, int(flags.get("--persistence-warmup-epochs", 0)))
            )
            persistence = (
                float(flags.get("--persistence-loss-weight", 0.0)) * persist_scale
                * persistence_topology_loss_torch(raw_logits_nhwc, target, persist_classes)
            )
            area = area_constraint_torch(raw_logits_nhwc, target, area_lambdas)
            _entropy_bits, entropy_rate = weight_entropy_rate_term_torch(
                model, sigma=float(flags.get("--weight-entropy-penalty-sigma", 0.2))
            )
            weight_entropy = (
                float(flags.get("--weight-entropy-penalty-lambda", 0.0)) * entropy_rate
            )
            phase_advect = torch.zeros((), device=device)
            if phase_weight > 0.0 and epoch >= phase_start:
                phase_rows = [phase_provider(pi) for pi in chunk]
                ref = torch.as_tensor(
                    np.stack([r[0] for r in phase_rows]), device=device, dtype=signed.dtype
                )
                direction = torch.as_tensor(
                    np.stack([r[1] for r in phase_rows]), device=device, dtype=signed.dtype
                )
                pw = torch.as_tensor(
                    np.stack([r[2] for r in phase_rows]), device=device, dtype=signed.dtype
                )
                tie = witness_tie_coordinate_torch(signed, direction)
                phase_num = ((tie - ref).square() * pw).sum(dim=(-2, -1))
                phase_den = pw.sum(dim=(-2, -1)) + 1e-6
                phase_advect = phase_weight * (phase_num / phase_den).mean()
            temporal_screw = torch.zeros((), device=device)
            if temporal_weight > 0.0 and epoch >= temporal_start:
                frame0 = frames[:, 0:1]
                seg0_in = seg.preprocess_input(frame0.permute(0, 1, 4, 2, 3).contiguous())
                logits0 = seg(seg0_in).permute(0, 2, 3, 1).contiguous()
                prob0 = torch.softmax(logits0, dim=-1)[..., list(temporal_classes)]
                prob1 = torch.softmax(raw_logits_nhwc, dim=-1)[..., list(temporal_classes)]
                for pi in chunk:
                    if pi not in temporal_grid_cache:
                        assert temporal_geom is not None
                        temporal_grid_cache[pi] = homography_grid_from_xi(
                            temporal_xi[pi], temporal_geom, device=device, dtype=prob0.dtype
                        )
                grid = torch.cat([temporal_grid_cache[pi][0] for pi in chunk], dim=0)
                valid = torch.cat([temporal_grid_cache[pi][1] for pi in chunk], dim=0)
                warped0 = warp_field_persist_torch(prob0, grid, valid)
                annulus = torch.as_tensor(
                    margins[chunk_np] < float(flags.get("--seg-temporal-screw-band", 2.0)),
                    device=device, dtype=prob0.dtype,
                )
                temporal_num = (
                    (prob1 - warped0).square().sum(-1) * annulus
                ).sum(dim=(-2, -1))
                temporal_den = annulus.sum(dim=(-2, -1)) + 1e-6
                temporal_screw = temporal_weight * (temporal_num / temporal_den).mean()
            chroma = torch.zeros((), device=device)
            if epoch >= int(flags["--seg-chroma-boundary-start-epoch"]):
                gt = torch.as_tensor(
                    gt_f1[chunk_np], device=device, dtype=frames.dtype
                ).permute(0, 3, 1, 2)
                gt = torch.nn.functional.interpolate(
                    gt, size=(cfg.render_h, cfg.render_w), mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1)
                ann = torch.as_tensor(
                    margins[chunk_np] < float(flags["--seg-chroma-boundary-margin-band"]),
                    device=device,
                )
                chroma = chroma_boundary_loss(frames[:, 1], gt, ann)
            chroma_contrib = float(flags["--seg-chroma-boundary-weight"]) * chroma
            total = (
                seg_contrib + pose_contrib + eik_contrib + length_contrib
                + amplify + persistence + area + weight_entropy + phase_advect
                + temporal_screw + chroma_contrib
            )
            optimizer.zero_grad(set_to_none=True)
            ep_tot += 1
            weights_stepped = bool(torch.isfinite(total.detach()))
            group_norms = {}
            if weights_stepped:
                total.backward()
                try:
                    group_norms = clip_grad_groups(
                        parameter_groups(model), float(flags["--grad-clip"])
                    )
                except RuntimeError:
                    weights_stepped = False
            if weights_stepped:
                optimizer.step()
                _ema_update(ema, model, ema_decay)
                ep_acc += 1
                pair_cursor.record_accepted(1)
            else:
                optimizer.zero_grad(set_to_none=True)
            finite_norms = [v for v in group_norms.values() if v is not None]
            gnorm = max((float(v) for v in finite_norms), default=0.0)
            terms = {
                "seg": float(seg_contrib.detach()),
                "pose": float(pose_contrib.detach()),
                "eikonal": float(eik_contrib.detach()),
                "length": float(length_contrib.detach()),
                "chroma_boundary": float(chroma_contrib.detach()),
                "island_amplify": float(amplify.detach()),
                "area_constraint": float(area.detach()),
                "persistence": float(persistence.detach()),
                "weight_entropy": float(weight_entropy.detach()),
                "phase_advect": float(phase_advect.detach()),
                "temporal_screw": float(temporal_screw.detach()),
            }
            telemetry = loss_terms_row(
                epoch=epoch, accum_batch=chunk_index, terms=terms,
                total=float(total.detach()), gnorm=round(gnorm, 4),
                accepted_frac=ep_acc / ep_tot, weights_stepped=weights_stepped,
                hosc_beta=round(float(model.hosc_beta), 4),
                softmax_temp=round(float(model.softmax_temp), 4), backend="torch_cuda",
                curriculum_stage=stage, lr=optimizer.param_groups[0]["lr"],
                pairs=chunk, pair_count=len(chunk), vectorized_chunk=True,
                authority="[contest-CUDA training-advisory]", promotion_eligible=False,
                wall_clock_s=time.time() - t0,
            )
            with open(out / TORCH_TRAJECTORY_JSONL, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(telemetry, sort_keys=True) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            print(json.dumps(telemetry), flush=True)
            chunk_index += 1

        blob = None
        if epoch % ckpt_every == 0 or epoch == args.epochs:
            blob = _checkpoint_blob(
                model, ema, optimizer, epoch, cfg_hash, dsl_argv,
                pair_cursor=pair_cursor, controller_state=controller_state,
            )
            _atomic_torch_save(blob, out / TORCH_RESUME_PT)
            _atomic_torch_save({"schema": "v9_cgauge_torch_ema_v1", "epoch": epoch,
                                "ema": ema.state_dict(), "config_hash": cfg_hash,
                                "dsl_argv": list(dsl_argv)}, out / TORCH_EMA_PT)
    final_blob = _checkpoint_blob(
        model, ema, optimizer, args.epochs, cfg_hash, dsl_argv,
        pair_cursor=pair_cursor, controller_state=controller_state,
    )
    _atomic_torch_save(
        final_blob,
        out / "stage_checkpoints" / f"ep{args.epochs:05d}_{prev_stage}.pt",
    )
    _atomic_json(
        {
            "schema": "v9_cgauge_torch_train_result_v1",
            "status": "completed",
            "backend": "torch_cuda",
            "epochs_completed": args.epochs,
            "config_hash": cfg_hash,
            "seed": int(flags["--seed"]),
            "authority": "[contest-CUDA training-advisory] NON-PROMOTABLE",
            "pointer": {"score": 0.19108282, "axis": "contest-CPU", "moved": False},
            "completed_at_utc": _utc(),
        },
        out / TORCH_TRAIN_RESULT_JSON,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
