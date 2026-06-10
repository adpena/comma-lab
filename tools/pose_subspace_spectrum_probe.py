#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure the PoseNet pose-sensitive SUBSPACE spectrum on the real frozen scorer (task #80).

THE HEADLINE MEASUREMENTS this CLI produces (all $0, exact frozen CPU-torch PoseNet, GT via
``frame_utils.yuv420_to_rgb`` ONLY, NEVER MPS, non-promotable ``[local CPU-torch advisory]``):

  1. The pose-sensitive subspace EFFECTIVE DIMENSION per frame slot (the participation ratio of the 6xN
     Jacobian spectrum) -- is it low-dim?
  2. The POSE-NULL energy fraction of GT-realistic frame perturbations (isotropic noise vs the frontier
     comp residual vs a low-res carrier residual) -- how much perturbation energy is invisible to pose?
  3. THE ESCAPE TEST: take a frame perturbation of a fixed pixel-RMSE, project OUT its pose-sensitive
     component (confine it to the pose-null), re-measure the EXACT d_pose. If a large-RMSE null-confined
     perturbation holds the tube where the same-RMSE isotropic one breaks it (#74's RMSE<3) -> the
     pixel-RMSE<3 floor is an artifact of isotropic error, and a Jacobian-aligned representation escapes
     the capacity wall.
  4. The per-frame pose contribution (frame0 vs frame1 each held-GT-other) -- the optimal pose-frame split.

Usage:
    .venv/bin/python tools/pose_subspace_spectrum_probe.py --pairs 4 --out <json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)
for _p in (str(REPO_ROOT / "upstream"), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from tac.boundary_math.posenet_subspace_spectrum import (  # noqa: E402
    SEG_H,
    SEG_W,
    expected_isotropic_null_fraction,
    measure_pose_subspace_spectrum,
    project_onto_pose_null,
)
from tac.differentiable_eval_roundtrip import (  # noqa: E402
    patch_upstream_yuv6_globally,
    unpatch_upstream_yuv6,
)

CAMERA_H, CAMERA_W = 874, 1164


def _load_posenet():
    from modules import PoseNet, posenet_sd_path  # type: ignore
    from safetensors.torch import load_file  # type: ignore

    net = PoseNet().eval()
    net.load_state_dict(load_file(posenet_sd_path, device="cpu"))
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _decode_gt_pairs(n_pairs: int):
    """Decode the FIRST 2*n_pairs GT frames into non-overlapping pairs (matches the eval seq_len=2)."""

    import av  # type: ignore
    from frame_utils import yuv420_to_rgb  # type: ignore

    vid = REPO_ROOT / "upstream" / "videos" / "0.mkv"
    cont = av.open(str(vid))
    frames = []
    need = 2 * n_pairs
    for i, fr in enumerate(cont.decode(video=0)):
        if i >= need:
            break
        frames.append(yuv420_to_rgb(fr).permute(2, 0, 1).float())  # (3,H,W) camera res
    pairs = [(frames[2 * k], frames[2 * k + 1]) for k in range(n_pairs)]
    return pairs


def _pose6(net, f0_chw, f1_chw):
    x = torch.stack([f0_chw, f1_chw]).unsqueeze(0)  # (1,2,3,H,W)
    return net(net.preprocess_input(x))["pose"][..., :6].reshape(-1)


def _feature_pose_subspace(net, gt0, gt1):
    """The 6x512 Jacobian of pose6 w.r.t. the 512-dim SUMMARY feature (the feature-distill subspace).

    Returns (effective_dim, rank, top1_energy_frac, singular_values). This is the dimension of the
    feature directions the 6 scored pose dims read -- the feature-distillation target dimensionality.
    """

    x = torch.stack([gt0, gt1]).unsqueeze(0)
    pin = net.preprocess_input(x)
    vision_out = net.vision((pin - net._mean) / net._std)
    summary = net.summarizer(vision_out).detach().clone().requires_grad_(True)  # (1,512)
    pose6 = net.hydra(summary)["pose"][..., :6].reshape(-1)
    jac = np.zeros((6, summary.numel()), dtype=np.float64)
    for kdim in range(6):
        g = torch.autograd.grad(pose6[kdim], summary, retain_graph=(kdim < 5))[0]
        jac[kdim] = g.detach().reshape(-1).numpy()
    sv = np.linalg.svd(jac, compute_uv=False)
    sv = sv[sv >= 0]
    tot = float((sv**2).sum())
    eff = (float(sv.sum()) ** 2 / tot) if tot > 0 else 0.0
    rank = int(np.sum(sv > 1e-4 * sv[0])) if sv.size else 0
    top1 = float(sv[0] ** 2 / tot) if tot > 0 else 0.0
    return {
        "feature_effective_dim": float(eff),
        "feature_rank": rank,
        "feature_dim_total": int(summary.numel()),
        "feature_top1_energy_frac": top1,
        "feature_singular_values": [float(s) for s in sv[:6]],
    }


def _quant_basis_escape(net, spec, gtw_flat, gt0, gt1, slot):
    """Compare UNIFORM vs JACOBIAN-ALIGNED quantization of frame `slot` at the work resolution.

    The escape question made concrete: does protecting the pose-sensitive coords (lossless) + coarsely
    quantizing the pose-null cost FEWER bytes than uniform quantization for the SAME d_pose? Bytes are the
    zlib-compressed int payload (a coder-agnostic proxy). Returns the two RD curves.
    """

    import zlib

    B = np.asarray(spec.row_basis, dtype=np.float64)  # (r, N)
    other = gt1 if slot == 0 else gt0

    def _enc_bytes(arr_int):
        return len(zlib.compress(arr_int.astype(np.int16).tobytes(), 9))

    def _measure(delta_flat):
        wp = torch.as_tensor(gtw_flat + delta_flat, dtype=torch.float32).reshape(3, SEG_H, SEG_W)
        cam_pert = torch.clamp(_to_camera(wp), 0, 255)
        with torch.no_grad():
            p_gt = _pose6(net, gt0, gt1)
            p = _pose6(net, cam_pert, other) if slot == 0 else _pose6(net, other, cam_pert)
        return float(((p - p_gt) ** 2).mean())

    uniform = []
    for q in (4, 8, 16, 32, 64):
        qd = np.round(gtw_flat / q) * q - gtw_flat
        uniform.append(
            {
                "q": q,
                "work_rmse": float(np.sqrt((qd**2).mean())),
                "d_pose": _measure(qd),
                "bytes": _enc_bytes(np.round(gtw_flat / q)),
            }
        )
    c = B @ gtw_flat
    r = gtw_flat - (c @ B)
    aligned = []
    for qn in (8, 16, 32, 64, 128):
        rq = np.round(r / qn) * qn
        recon = (c @ B) + rq
        delta = recon - gtw_flat
        aligned.append(
            {
                "qn": qn,
                "work_rmse": float(np.sqrt((delta**2).mean())),
                "d_pose": _measure(delta),
                "bytes": int(8 * spec.rank + _enc_bytes(np.round(r / qn))),
            }
        )
    return {"uniform": uniform, "jacobian_aligned": aligned}


def _d_pose(net, f0_chw, f1_chw, gt0, gt1):
    """Exact d_pose = MSE on first 6 dims, comp pair vs GT pair (the contest functional)."""

    with torch.no_grad():
        p_comp = _pose6(net, f0_chw, f1_chw)
        p_gt = _pose6(net, gt0, gt1)
    return float(((p_comp - p_gt) ** 2).mean())


def _work_frame(cam_chw):
    return F.interpolate(cam_chw.unsqueeze(0), size=(SEG_H, SEG_W), mode="bilinear", align_corners=False)[0]


def _to_camera(work_chw):
    return F.interpolate(
        work_chw.unsqueeze(0), size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False
    )[0]


def _rmse(a, b):
    return float(torch.sqrt(((a - b) ** 2).mean()).item())


def _f8_carrier(cam_chw):
    """A downsample-f8 + upsample-back + uint8-round low-res carrier of a camera-res frame."""

    lr = F.interpolate(
        F.interpolate(cam_chw.unsqueeze(0), scale_factor=1 / 8, mode="bilinear", align_corners=False),
        size=(CAMERA_H, CAMERA_W),
        mode="bilinear",
        align_corners=False,
    )[0]
    return torch.clamp(torch.round(lr), 0, 255)


def _apply_work_delta_measure(net, gt_work_flat, delta_flat, base_cam, other, slot, gt0, gt1):
    """Add ``delta_flat`` to the work-res frame, upsample+uint8-round to camera res, return (d_pose, cam_rmse)."""

    work_pert = torch.as_tensor(gt_work_flat + delta_flat, dtype=torch.float32).reshape(3, SEG_H, SEG_W)
    cam_pert = torch.round(torch.clamp(_to_camera(work_pert), 0, 255))
    if slot == 0:
        return _d_pose(net, cam_pert, other, gt0, gt1), _rmse(cam_pert, base_cam)
    return _d_pose(net, other, cam_pert, gt0, gt1), _rmse(cam_pert, base_cam)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--escape-rmse", type=float, default=6.0, help="work-res pixel RMSE for the escape test")
    args = ap.parse_args(argv)

    t0 = time.time()
    net = _load_posenet()
    pairs = _decode_gt_pairs(args.pairs)
    rng = np.random.default_rng(0)

    token = patch_upstream_yuv6_globally()
    results = {"pairs": [], "authority": "local CPU-torch advisory", "promotable": False}
    try:
        for pi, (gt0, gt1) in enumerate(pairs):
            row: dict = {"pair": pi}
            # --- per-frame pose contribution (held-GT-other) ---
            # frame0-only perturb: replace frame0 with a low-res-roundtrip carrier of itself
            for slot in (0, 1):
                spec = measure_pose_subspace_spectrum(net, gt0, gt1, frame_slot=slot)
                summ = spec.to_summary()
                n = spec.n_pixels
                # --- pose-null fraction of three GT-realistic perturbations ---
                # (a) isotropic gaussian at the escape RMSE (the #74-style noise)
                gt_work = _work_frame(gt0 if slot == 0 else gt1).reshape(-1).detach().numpy().astype(np.float64)
                iso = rng.standard_normal(n)
                iso = iso / np.sqrt((iso**2).mean()) * args.escape_rmse  # RMSE == escape_rmse
                iso_dec = project_onto_pose_null(spec, iso)
                # (b) a low-res carrier residual: GT - bilinear(downsample f8(GT)) at work res
                cam = gt0 if slot == 0 else gt1
                lr = F.interpolate(
                    F.interpolate(cam.unsqueeze(0), scale_factor=1 / 8, mode="bilinear", align_corners=False),
                    size=(CAMERA_H, CAMERA_W),
                    mode="bilinear",
                    align_corners=False,
                )[0]
                lr_work = _work_frame(lr).reshape(-1).detach().numpy().astype(np.float64)
                lr_resid = gt_work - lr_work
                lr_dec = project_onto_pose_null(spec, lr_resid)
                # (c) the same-energy null-CONFINED perturbation (escape candidate): take iso, keep ONLY null
                null_only = np.asarray(iso_dec["null"], dtype=np.float64)
                # rescale null_only back to the escape RMSE so the comparison is iso-RMSE vs null-RMSE
                cur_rmse = float(np.sqrt((null_only**2).mean()))
                if cur_rmse > 0:
                    null_only = null_only / cur_rmse * args.escape_rmse

                # --- THE ESCAPE TEST: exact d_pose of iso vs null-confined, same RMSE ---
                base_cam = gt0 if slot == 0 else gt1
                other = gt1 if slot == 0 else gt0
                dpose_iso, rmse_iso = _apply_work_delta_measure(
                    net, gt_work, iso, base_cam, other, slot, gt0, gt1
                )
                dpose_null, rmse_null = _apply_work_delta_measure(
                    net, gt_work, null_only, base_cam, other, slot, gt0, gt1
                )

                row[f"slot{slot}"] = {
                    "effective_dim": summ["effective_dim"],
                    "rank": summ["rank"],
                    "n_pixels": n,
                    "sigma_ratio_max_over_2nd": summ["sigma_ratio_max_over_2nd"],
                    "energy_frac_top1": summ["energy_frac_top1"],
                    "energy_frac_top3": summ["energy_frac_top3"],
                    "expected_isotropic_null_frac": expected_isotropic_null_fraction(n, summ["rank"]),
                    "isotropic_null_frac": iso_dec["null_energy_frac"],
                    "lowres_carrier_resid_null_frac": lr_dec["null_energy_frac"],
                    "lowres_carrier_resid_work_rmse": float(np.sqrt((lr_resid**2).mean())),
                    "escape_test": {
                        "work_rmse_target": args.escape_rmse,
                        "isotropic_camera_rmse": rmse_iso,
                        "isotropic_d_pose": dpose_iso,
                        "null_confined_camera_rmse": rmse_null,
                        "null_confined_d_pose": dpose_null,
                        "d_pose_ratio_iso_over_null": (dpose_iso / dpose_null if dpose_null > 0 else float("inf")),
                    },
                }
            # --- per-frame pose contribution: replace each frame with its own low-res-f8 carrier ---
            lr0 = _f8_carrier(gt0)
            lr1 = _f8_carrier(gt1)
            row["per_frame_pose_contribution"] = {
                "f8carrier0_gt1": _d_pose(net, lr0, gt1, gt0, gt1),
                "gt0_f8carrier1": _d_pose(net, gt0, lr1, gt0, gt1),
                "both_f8carrier": _d_pose(net, lr0, lr1, gt0, gt1),
                "carrier0_camera_rmse": _rmse(lr0, gt0),
                "carrier1_camera_rmse": _rmse(lr1, gt1),
            }
            # --- feature-space pose subspace (the feature-distill target dimensionality) ---
            row["feature_subspace"] = _feature_pose_subspace(net, gt0, gt1)
            # --- the quant-basis escape test (slot 0, the dominant pose carrier) ---
            spec0 = measure_pose_subspace_spectrum(net, gt0, gt1, frame_slot=0)
            gt0_work_flat = _work_frame(gt0).reshape(-1).detach().numpy().astype(np.float64)
            row["quant_basis_escape_slot0"] = _quant_basis_escape(net, spec0, gt0_work_flat, gt0, gt1, 0)
            results["pairs"].append(row)
            print(f"[pair {pi}] done", flush=True)
    finally:
        unpatch_upstream_yuv6(token)

    # --- aggregate ---
    def _agg(getter):
        vals = [getter(r) for r in results["pairs"] if getter(r) is not None]
        return {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))} if vals else None

    results["aggregate"] = {
        "slot0_effective_dim": _agg(lambda r: r["slot0"]["effective_dim"]),
        "slot1_effective_dim": _agg(lambda r: r["slot1"]["effective_dim"]),
        "slot0_isotropic_null_frac": _agg(lambda r: r["slot0"]["isotropic_null_frac"]),
        "slot0_lowres_carrier_resid_null_frac": _agg(lambda r: r["slot0"]["lowres_carrier_resid_null_frac"]),
        "slot1_lowres_carrier_resid_null_frac": _agg(lambda r: r["slot1"]["lowres_carrier_resid_null_frac"]),
        "escape_slot0_iso_dpose": _agg(lambda r: r["slot0"]["escape_test"]["isotropic_d_pose"]),
        "escape_slot0_null_dpose": _agg(lambda r: r["slot0"]["escape_test"]["null_confined_d_pose"]),
        "escape_slot0_ratio": _agg(lambda r: r["slot0"]["escape_test"]["d_pose_ratio_iso_over_null"]),
        "f8carrier0_gt1_dpose": _agg(lambda r: r["per_frame_pose_contribution"]["f8carrier0_gt1"]),
        "gt0_f8carrier1_dpose": _agg(lambda r: r["per_frame_pose_contribution"]["gt0_f8carrier1"]),
    }
    results["elapsed_sec"] = time.time() - t0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results["aggregate"], indent=2))
    print(f"wrote {args.out} in {results['elapsed_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
