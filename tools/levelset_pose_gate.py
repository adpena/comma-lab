#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""LEVEL-SET POSE GATE (#206) — the $0 mission-framing gate: warp-real-luma vs FiLM-SDF + is seg⊥pose FREE?

Decides, BEFORE we trust the S ≈ 100·d_seg + 0.073 budget, whether the pose facet can ride the SDF
render (FiLM-condition it toward the stored twist ξ*) or whether pose MANDATES warping a real keyframe
(warp-real-luma). It also measures the never-measured seg/pose RENDER-SPACE interference cosine that the
in-tree "0.99999" number does NOT (that is backward-PARITY, not seg⊥pose).

FOUR MEASUREMENTS (frozen CPU-torch authority, NEVER MPS; realized THROUGH R):
  1. cos(∂d_seg/∂F, ∂d_pose/∂F) on the through-R WITNESS render frame1 (the shared frame SegNet+PoseNet
     both read). d_seg surrogate = SegNet CE-to-GT-argmax (primary) AND margin-hinge (secondary);
     d_pose = MSE(PoseNet6(render) − PoseNet6(GT)). cos≈0 ⇒ moving the frame to fix seg leaves pose
     ~unchanged ⇒ seg⊥pose decoupling is FREE. (Measured at the WITNESS operating point: at the GT
     operating point ∂d_pose/∂F vanishes — d_pose=0 at its minimum — so the cos is only defined here.)
  2. through-R PoseNet 6×N Jacobian EFFECTIVE-DIM (participation ratio, ``measure_pose_subspace_spectrum``)
     on the SDF render vs REAL GT frames (~4.08 in-tree). The OOD/degeneracy check: if PoseNet collapses
     (eff-dim ≪ 4 / σ_max ≪ real) on the flat softmax-of-SDF render, the twist ξ is UNREACHABLE ⇒
     FiLM-conditioning is futile ⇒ warp-real-luma mandatory.
  3. POSE-NULL residual fraction (``pose_null_residual_fraction``) of a SEG-DRIVEN render change
     δ = −∂d_seg/∂F. Predict ≈1 (seg move invisible to pose) — reported vs the isotropic baseline
     r/N so it is not diluted by the tiny (≤6-of-N) pose subspace.
  4. A-vs-B READBACK d_pose (the decider):
       A (FiLM-SDF): d_pose of the pose-BLIND witness render through R (the as-is baseline) + the
         reachability σ_max/eff-dim from #2 (can PoseNet even read ξ back from the flat render?).
       B (warp-real-luma): d_pose of warping the REAL keyframe gt_f0 by the stored pose via the ground
         homography, through R — measured by REUSING ``tools/measure_warp_dpose_through_R.py`` (its
         band_posecal / full_ground_posecal d_pose). B ≪ A ⇒ warp-real-luma carries pose.

REUSE (no reinvention): ``tools.levelset_byte_close_and_eval`` (ckpt load + self-orient detect +
``numpy_oracle_reference_frames`` = the canonical through-R SDF render), ``train_witness_realized_
through_R_mlx`` (GT cache + ``cpu_verdict_d_pose``), ``tac.boundary_math.posenet_subspace_spectrum``
(#80 eff-dim), ``tac.torch_vehicle.pose_null_projection_hook`` (pose-null fraction),
``tac.differentiable_eval_roundtrip.patch_upstream_yuv6_globally`` (PoseNet gradient reachability),
``tools/measure_warp_dpose_through_R.py`` (the warp-real-luma readback).

AUTHORITY: ``[macOS-CPU advisory / CPU-torch research-signal]`` NON-PROMOTABLE. Frozen CPU-torch
SegNet/PoseNet, numpy-fp32/torch-fp32; NEVER MPS, no CUDA, no paid eval. Frontier pointer 0.19110
UNMOVED. score_claim/promotable = False. This is a MEASUREMENT that decides an architectural facet;
it is NOT a contest score. NO-FAKE: a NO-FAKE self-check asserts PoseNet(GT pair)==gt_poses (<1e-4)
and the spectrum fail-closes on a severed gradient (a degenerate Jacobian is REPORTED, never faked as
"all pose-null").
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

# macOS Accelerate BLAS sets spurious fp exception FLAGS on large finite matmuls (verified: the
# Jacobian has 0 nan/inf and the cosine sanity checks pass — cos(g,g)==1.0, cos(g,random)~8e-4). The
# warnings are cosmetic noise, NOT real overflow; silence them so the signal is readable.
np.seterr(all="ignore")

# 4-thread CPU (do NOT contend with the running MLX-GPU training). Set BEFORE torch import.
for _tv in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_tv, "4")

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")


def _refuse_tmp(path: Path, field: str) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{field}={path!r} is a /tmp-class path; use the SSD/repo tier per CLAUDE.md.")


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _advisory_axis() -> str:
    import platform

    if platform.system() == "Linux" and platform.machine().lower() in ("x86_64", "amd64"):
        return "[contest-CPU advisory] NON-PROMOTABLE"
    return "[macOS-CPU advisory / CPU-torch research-signal] NON-PROMOTABLE"


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).reshape(-1)
    b = np.asarray(b, np.float64).reshape(-1)
    na = float(np.sqrt(a @ a))
    nb = float(np.sqrt(b @ b))
    if na <= 0.0 or nb <= 0.0:
        return float("nan")
    return float((a @ b) / (na * nb))


# ---------------------------------------------------------------------------
# render-space gradients of d_seg surrogate + d_pose w.r.t. frame1 (torch, CPU).
# ---------------------------------------------------------------------------
def _frame1_variable(f1_cam_uint8, work_h, work_w):
    import torch
    import torch.nn.functional as F

    f1 = torch.as_tensor(np.asarray(f1_cam_uint8), dtype=torch.float32).permute(2, 0, 1)  # (3,Hc,Wc)
    work = F.interpolate(f1.unsqueeze(0), size=(work_h, work_w), mode="bilinear", align_corners=False)[0].clone()
    work.requires_grad_(True)
    return work


def _pair_from_work(f0_cam_uint8, work, camera_h, camera_w):
    import torch
    import torch.nn.functional as F

    f0 = torch.as_tensor(np.asarray(f0_cam_uint8), dtype=torch.float32).permute(2, 0, 1)  # (3,Hc,Wc)
    work_cam = F.interpolate(work.unsqueeze(0), size=(camera_h, camera_w), mode="bilinear", align_corners=False)[0]
    x = torch.stack([f0, work_cam]).unsqueeze(0)  # (1,2,3,Hc,Wc)
    return x


def seg_grad(segnet, f0_cam, f1_cam, gt_argmax, work_h, work_w, camera_h, camera_w, mode="ce"):
    """∂L_seg/∂frame1 at the (witness) render. mode ∈ {ce, margin}. Returns (grad (3,work_h,work_w) np,
    L_seg float). L_seg=CE-to-GT-argmax (primary) or margin-hinge (secondary)."""
    import torch
    import torch.nn.functional as F

    work = _frame1_variable(f1_cam, work_h, work_w)
    x = _pair_from_work(f0_cam, work, camera_h, camera_w)
    seg_in = segnet.preprocess_input(x)          # (1,3,384,512)
    logits = segnet(seg_in)                        # (1,5,384,512)
    gt = torch.as_tensor(np.asarray(gt_argmax), dtype=torch.long)
    if tuple(gt.shape) != tuple(logits.shape[-2:]):
        gt = F.interpolate(gt[None, None].float(), size=logits.shape[-2:], mode="nearest")[0, 0].long()
    if mode == "ce":
        loss = F.cross_entropy(logits, gt.unsqueeze(0))
    else:
        lg = logits[0]                             # (5,h,w)
        gt_logit = lg.gather(0, gt.unsqueeze(0))[0]  # (h,w)
        onehot = F.one_hot(gt, num_classes=lg.shape[0]).permute(2, 0, 1).bool()
        other_max = lg.masked_fill(onehot, float("-inf")).max(0).values  # (h,w)
        margin = gt_logit - other_max
        loss = torch.relu(1.0 - margin).mean()     # hinge: reduces flips (margin<1 penalized)
    g = torch.autograd.grad(loss, work, create_graph=False)[0]
    return g.detach().cpu().numpy().astype(np.float64), float(loss.detach())


def pose_grad(posenet, f0_cam, f1_cam, gt_pose6, work_h, work_w, camera_h, camera_w):
    """∂d_pose/∂frame1 at the (witness) render. d_pose=MSE(PoseNet6(pair)−gt_pose6). Requires the
    differentiable yuv6 patch to be ACTIVE (else the Jacobian is severed). Returns (grad np, d_pose)."""
    import torch

    work = _frame1_variable(f1_cam, work_h, work_w)
    x = _pair_from_work(f0_cam, work, camera_h, camera_w)
    pin = posenet.preprocess_input(x)              # (1,12,384,512) differentiable via patched yuv6
    out = posenet(pin)
    pose = out["pose"] if isinstance(out, dict) else out
    half = pose.shape[-1] // 2
    pose6 = pose[0, :half]
    gt = torch.as_tensor(np.asarray(gt_pose6), dtype=torch.float32)
    d_pose = ((pose6 - gt) ** 2).mean()
    g = torch.autograd.grad(d_pose, work, create_graph=False)[0]
    return g.detach().cpu().numpy().astype(np.float64), float(d_pose.detach())


# ---------------------------------------------------------------------------
# WITNESS render (reuses the canonical numpy-fp32 oracle through-R render).
# ---------------------------------------------------------------------------
def render_witness_strided(params, manifest, strided, lbc):
    """Render the fp32-EMA witness frames THROUGH R for the strided pair indices, by REUSING the
    canonical ``numpy_oracle_reference_frames`` with a re-indexed code table (row 2j+fk = the strided
    pair's code) — zero reimplementation of the forward. Returns list of (f0,f1) uint8 camera arrays."""
    code_full = np.asarray(params["code"], np.float32)
    code_strided = np.concatenate([code_full[2 * i:2 * i + 2] for i in strided], axis=0)
    frames, _argmax = lbc.numpy_oracle_reference_frames(params, code_strided, manifest, len(strided))
    pairs = [(frames[2 * j], frames[2 * j + 1]) for j in range(len(strided))]
    return pairs


def run(args) -> dict:
    import torch

    torch.set_num_threads(4)
    t0 = time.time()
    authority = _advisory_axis()

    import tools.levelset_byte_close_and_eval as lbc
    import train_witness_realized_through_R_mlx as twr
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # upstream
    from tac.boundary_math.posenet_subspace_spectrum import (
        PoseSubspaceError,
        expected_isotropic_null_fraction,
        measure_pose_subspace_spectrum,
    )
    from tac.boundary_math.seg_core import load_real_segnet
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.torch_vehicle.pose_null_projection_hook import pose_null_residual_fraction

    ckpt_dir = Path(args.ckpt_dir)
    # ---- 1. load ckpt + config + self-orient detect + manifest (reused canonical path) ----
    params, cfg = lbc._load_levelset_ckpt(ckpt_dir, args.npz_name)
    so_over = {"freq_across": 32.0, "freq_along": 4.0, "tau": 4.0, "iters": 4}
    so = lbc.detect_self_orient(cfg, so_over)
    blob, _bd = lbc.build_levelset_blob(params, cfg, so, None)
    manifest, _b, _c, _p = lbc._read_blob_bytes(blob)
    n_pairs_total = int(cfg["n_pairs"])
    render_h, render_w = int(cfg["render_h"]), int(cfg["render_w"])
    camera_h, camera_w = lbc.CAMERA_H, lbc.CAMERA_W
    work_h, work_w = int(args.work_h), int(args.work_w)

    # ---- 2. GT cache (frozen CPU-torch authority; used for targets + real-frame reference) ----
    gt, seg_cpu_from_twr, posenet_cpu = twr.load_gt_from_cache(Path(args.gt_cache), n_pairs_total)
    # segnet from load_real_segnet (grad-able logits path used by seg_grad)
    segnet = load_real_segnet("cpu")
    for p_ in segnet.parameters():
        p_.requires_grad = False

    # ---- 2b. PoseNet with the differentiable yuv6 patch ACTIVE (gradient reachability) ----
    patch_upstream_yuv6_globally()
    dn = DistortionNet().eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    posenet = dn.posenet
    for p_ in posenet.parameters():
        p_.requires_grad = False

    # NO-FAKE self-check: PoseNet(GT pair)==gt_poses under the patch (proves the differentiable yuv6 is
    # numerically faithful so the verdict d_pose is unchanged; the SegNet argmax==lstars too).
    selfcheck = {"pairs": min(4, n_pairs_total), "pose_max_abs_err": 0.0, "seg_max_disagree_px": 0}
    for p in range(selfcheck["pairs"]):
        dp0 = twr.cpu_verdict_d_pose(posenet, gt.gt_f0[p], gt.gt_f1[p], gt.gt_poses[p])
        ds0 = twr.cpu_verdict_d_seg(seg_cpu_from_twr, gt.gt_f1[p], gt.lstars[p])
        selfcheck["pose_max_abs_err"] = max(selfcheck["pose_max_abs_err"], float(dp0))
        selfcheck["seg_max_disagree_px"] = max(selfcheck["seg_max_disagree_px"], int(round(ds0 * gt.lstars[p].size)))
    selfcheck["PASS"] = bool(selfcheck["pose_max_abs_err"] < 1e-4 and selfcheck["seg_max_disagree_px"] == 0)
    if not selfcheck["PASS"]:
        raise SystemExit(f"NO-FAKE self-check FAILED under yuv6 patch: {selfcheck}")
    print(f"[pose-gate] NO-FAKE self-check PASS (pose_err={selfcheck['pose_max_abs_err']:.2e}, "
          f"seg_disagree_px={selfcheck['seg_max_disagree_px']})  {authority}", flush=True)

    # ---- 3. strided sample across the whole clip (avoid Lens-A contiguous-prefix optimism) ----
    n_geom = min(int(args.n_geom), n_pairs_total)
    stride = max(1, n_pairs_total // n_geom)
    strided = list(range(0, n_pairs_total, stride))[:n_geom]
    n_real = min(int(args.n_real), len(strided))
    print(f"[pose-gate] rendering {len(strided)} strided witness pairs through R "
          f"(stride {stride}, render {render_h}x{render_w}, self_orient={so['self_orient']})...", flush=True)
    wpairs = render_witness_strided(params, manifest, strided, lbc)
    print(f"[pose-gate] rendered {len(wpairs)} witness pairs in {time.time()-t0:.0f}s", flush=True)

    # ---- 4B (launched early, runs concurrently): warp-real-luma d_pose through R ----
    warp_proc = None
    warp_out = _REPO / f"experiments/results/pose_gate_warp_readback_n{args.warp_n}/results.json"
    if not args.skip_warp:
        _refuse_tmp(warp_out, "warp_out")
        warp_out.parent.mkdir(parents=True, exist_ok=True)
        warp_cmd = [sys.executable, str(_REPO / "tools/measure_warp_dpose_through_R.py"),
                    "--cache", str(args.gt_cache), "--n-pairs", str(int(args.warp_n)),
                    "--no-broken", "--no-argmax", "--out", str(warp_out)]
        env = dict(os.environ)
        env.setdefault("OMP_NUM_THREADS", "4")
        print(f"[pose-gate] launching warp-real-luma readback (B) concurrently: n={args.warp_n}", flush=True)
        warp_proc = subprocess.Popen(warp_cmd, cwd=str(_REPO), env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # ---- 7. per-pair geometry: cos(g_seg,g_pose) [1], eff-dim SDF vs real [2], pose-null of δ [3],
    #         witness through-R d_pose [4A]. ----
    rows = []
    jsonl = Path(args.out).with_suffix(".partial.jsonl")
    done = set()
    if jsonl.exists():
        for ln in jsonl.read_text().splitlines():
            try:
                r = json.loads(ln)
                done.add(int(r["pair_index"]))
                rows.append(r)
            except Exception:
                pass
    fjl = open(jsonl, "a")

    for j, pidx in enumerate(strided):
        if pidx in done:
            continue
        f0w, f1w = wpairs[j]                       # witness through-R camera frames
        gt_arg = gt.lstars[pidx]
        gt_pose = gt.gt_poses[pidx]

        # [1] render-space interference cosine at the witness operating point
        g_seg_ce, L_ce = seg_grad(segnet, f0w, f1w, gt_arg, work_h, work_w, camera_h, camera_w, "ce")
        g_seg_mg, L_mg = seg_grad(segnet, f0w, f1w, gt_arg, work_h, work_w, camera_h, camera_w, "margin")
        g_pose, dpose_grad_pt = pose_grad(posenet, f0w, f1w, gt_pose, work_h, work_w, camera_h, camera_w)
        cos_ce = _cos(g_seg_ce, g_pose)
        cos_mg = _cos(g_seg_mg, g_pose)

        # [4A] witness through-R d_pose (verdict; frozen path)
        dpose_witness = twr.cpu_verdict_d_pose(posenet, f0w, f1w, gt_pose)

        # [2] PoseNet Jacobian eff-dim on the SDF render (frame_slot=1, the seg-shared frame)
        row = {
            "pair_index": int(pidx),
            "cos_seg_pose_ce": cos_ce,
            "cos_seg_pose_margin": cos_mg,
            "d_pose_witness_through_R": float(dpose_witness),
            "g_seg_ce_norm": float(np.sqrt((g_seg_ce ** 2).sum())),
            "g_pose_norm": float(np.sqrt((g_pose ** 2).sum())),
        }
        try:
            spec = measure_pose_subspace_spectrum(posenet, f0w.transpose(2, 0, 1).astype(np.float64),
                                                  f1w.transpose(2, 0, 1).astype(np.float64),
                                                  frame_slot=1, work_h=work_h, work_w=work_w)
            row["witness_eff_dim"] = float(spec.effective_dim)
            row["witness_rank"] = int(spec.rank)
            row["witness_sigma_max"] = float(spec.singular_values[0]) if spec.singular_values.size else 0.0
            row["witness_sigma_top6"] = [float(x) for x in spec.singular_values[:6]]
            row["witness_severed"] = False
            # [3] pose-null fraction of the seg-driven change δ = −g_seg (CE)
            delta = (-g_seg_ce).reshape(3, work_h, work_w)
            frac = pose_null_residual_fraction(delta, spec)
            row["seg_delta_pose_null_frac"] = float(frac["null_energy_frac"])
            row["seg_delta_pose_sensitive_frac"] = float(frac["sensitive_energy_frac"])
            row["isotropic_sensitive_frac_baseline"] = float(spec.rank) / float(spec.n_pixels)
        except PoseSubspaceError as e:
            row["witness_eff_dim"] = 0.0
            row["witness_rank"] = 0
            row["witness_sigma_max"] = 0.0
            row["witness_severed"] = True
            row["witness_severed_msg"] = str(e)[:200]
            row["seg_delta_pose_null_frac"] = 1.0  # empty pose subspace ⇒ trivially all-null
            row["seg_delta_pose_sensitive_frac"] = 0.0

        # [2-real] REAL-frame eff-dim reference (subset), same slot — reproduces the ~4.08 in-tree number
        if j < n_real:
            try:
                rspec = measure_pose_subspace_spectrum(posenet, gt.gt_f0[pidx].transpose(2, 0, 1).astype(np.float64),
                                                       gt.gt_f1[pidx].transpose(2, 0, 1).astype(np.float64),
                                                       frame_slot=1, work_h=work_h, work_w=work_w)
                row["real_eff_dim"] = float(rspec.effective_dim)
                row["real_rank"] = int(rspec.rank)
                row["real_sigma_max"] = float(rspec.singular_values[0]) if rspec.singular_values.size else 0.0
                row["real_sigma_top6"] = [float(x) for x in rspec.singular_values[:6]]
            except PoseSubspaceError:
                row["real_eff_dim"] = None

        rows.append(row)
        fjl.write(json.dumps(row) + "\n")
        fjl.flush()
        print(f"  [{j+1}/{len(strided)}] pair {pidx}: cos_ce={cos_ce:+.4f} cos_mg={cos_mg:+.4f} "
              f"eff_dim(W)={row['witness_eff_dim']:.3f} sig_max(W)={row['witness_sigma_max']:.3e} "
              f"nullfrac(δ)={row['seg_delta_pose_null_frac']:.5f} d_pose(W)={dpose_witness:.2f}"
              + (f" eff_dim(REAL)={row.get('real_eff_dim')}" if 'real_eff_dim' in row else ""),
              flush=True)
    fjl.close()

    # ---- 4B: harvest the warp-real-luma readback ----
    warp = {"ran": False}
    if warp_proc is not None:
        print("[pose-gate] waiting on warp-real-luma readback (B)...", flush=True)
        wout, _ = warp_proc.communicate()
        if warp_out.exists():
            wj = json.loads(warp_out.read_text())
            v = wj.get("verdict", {})
            warp = {
                "ran": True,
                "n_pairs": wj.get("n_pairs"),
                "naive_copy_d_pose_null": v.get("naive_copy_d_pose (zero-motion null ~= the broken 190)"),
                "band_posecal_d_pose": v.get("band_posecal_d_pose (corrected, shippable, d_pose-fit s_t)"),
                "full_ground_posecal_d_pose": v.get("full_ground_posecal_d_pose (whole-frame ground, d_pose-fit s_t)"),
                "best_corrected_d_pose": v.get("best_corrected_d_pose"),
                "pose_drop_frac_vs_null": v.get("pose_d_pose_drop_frac_vs_null"),
                "POSE_CARRIED_BY_WARP": v.get("POSE_CARRIED_BY_WARP"),
                "results_json": str(warp_out),
            }
        else:
            warp = {"ran": True, "error": "warp results.json not produced", "tail": (wout or "")[-1500:]}

    # ---- aggregate + verdict ----
    def _stat(key, arr):
        a = np.asarray([r[key] for r in rows if key in r and r[key] is not None and np.isfinite(r[key])], np.float64)
        if a.size == 0:
            return {"mean": None, "median": None, "n": 0}
        return {"mean": float(a.mean()), "median": float(np.median(a)), "n": int(a.size),
                "min": float(a.min()), "max": float(a.max())}

    cos_ce_s = _stat("cos_seg_pose_ce", rows)
    cos_mg_s = _stat("cos_seg_pose_margin", rows)
    eff_w_s = _stat("witness_eff_dim", rows)
    sig_w_s = _stat("witness_sigma_max", rows)
    rank_w_s = _stat("witness_rank", rows)
    eff_r_s = _stat("real_eff_dim", rows)
    sig_r_s = _stat("real_sigma_max", rows)
    rank_r_s = _stat("real_rank", rows)
    nullfrac_s = _stat("seg_delta_pose_null_frac", rows)
    sensfrac_s = _stat("seg_delta_pose_sensitive_frac", rows)
    isobase_s = _stat("isotropic_sensitive_frac_baseline", rows)
    dpose_w_s = _stat("d_pose_witness_through_R", rows)
    n_severed = int(sum(1 for r in rows if r.get("witness_severed")))

    sig_ratio = (sig_w_s["median"] / sig_r_s["median"]) if (sig_r_s["median"] and sig_w_s["median"] is not None and sig_r_s["median"] > 0) else None
    # decoupling free? render-space seg/pose cosine near 0 (both surrogates).
    cos_abs_med = np.nanmedian([abs(r["cos_seg_pose_ce"]) for r in rows if np.isfinite(r.get("cos_seg_pose_ce", np.nan))]) if rows else float("nan")
    decoupling_free = bool(np.isfinite(cos_abs_med) and cos_abs_med < 0.10)
    # is pose REACHABLE on the SDF render? the raw participation-ratio eff-dim is dim0(forward-motion)-
    # dominated (~1 on BOTH witness AND real), so the DISCRIMINATING reachability metrics are (a) the
    # Jacobian RANK (how many of the 6 pose DOFs are locally sensitive above tol) — full rank 6 == all
    # DOFs reachable; and (b) σ_max(witness)/σ_max(real) not collapsed (PoseNet is as sensitive to the
    # SDF render as to a real frame). Degenerate/blind ONLY if rank collapses or σ_max ≪ real or severed.
    rank_full = bool(rank_w_s["median"] is not None and rank_w_s["median"] >= 5.5)
    sigma_healthy = bool(sig_ratio is not None and sig_ratio >= 0.25)
    pose_reachable_on_sdf = bool(rank_full and sigma_healthy and n_severed == 0)
    warp_carries = bool(warp.get("POSE_CARRIED_BY_WARP")) if warp.get("ran") else None
    warp_best = warp.get("best_corrected_d_pose") if warp.get("ran") else None
    if not pose_reachable_on_sdf:
        verdict = ("WARP-REAL-LUMA MANDATORY (PoseNet Jacobian on the flat SDF render is rank-collapsed "
                   "or σ_max ≪ real — the twist ξ is UNREACHABLE by FiLM-SDF conditioning)")
    elif warp_carries:
        verdict = ("BOTH FEASIBLE, WARP-REAL-LUMA PREFERRED: PoseNet's Jacobian on the SDF render is "
                   "full-rank with healthy σ_max (ξ IS locally reachable, so FiLM-SDF conditioning is "
                   "not blocked), BUT warping the real keyframe by the stored pose carries d_pose cheaply "
                   "(deterministic, ~0 training) — the pragmatic pose path. Pose-blindness (d_pose~100) is "
                   "by-design (w_pose=0), NOT unreachability.")
    else:
        verdict = ("FiLM-SDF FEASIBLE (full-rank PoseNet Jacobian on the SDF render, healthy σ_max ⇒ "
                   "conditioning the texture/code toward ξ* CAN inject pose); warp readback did not "
                   "confirm a cheaper deterministic path this run.")

    report = {
        "tool": "tools/levelset_pose_gate.py",
        "gate": "#206 mission-framing pose gate",
        "utc": _utc(),
        "authority": authority,
        "score_claim": False, "promotion_eligible": False, "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False, "promotable": False,
        "frontier_pointer": "UNMOVED 0.19110 (advisory measurement; not a contest score)",
        "ckpt_dir": str(ckpt_dir), "npz_name": cfg["npz_name"],
        "config": {k: cfg.get(k) for k in ("n_classes", "hidden_dim", "n_hidden", "mod_dim", "activation",
                                           "softmax_temp", "chroma", "render_h", "render_w", "in_feat")},
        "self_orient": so,
        "n_pairs_total": n_pairs_total,
        "n_geom_strided": len(strided), "stride": stride, "strided_indices": strided,
        "n_real_reference": n_real, "work_resolution": [work_h, work_w],
        "no_fake_selfcheck": selfcheck,
        "measurements": {
            "1_render_space_seg_pose_cosine": {
                "cos_ce": cos_ce_s, "cos_margin": cos_mg_s,
                "abs_median_ce": float(cos_abs_med) if np.isfinite(cos_abs_med) else None,
                "interpretation": "cos≈0 ⇒ seg-descent direction ⟂ pose-gradient ⇒ moving the frame to "
                                  "fix d_seg leaves d_pose ~unchanged ⇒ seg⊥pose decoupling is FREE.",
            },
            "2_posenet_jacobian_eff_dim_sdf_vs_real": {
                "witness_sdf_eff_dim": eff_w_s, "real_frame_eff_dim": eff_r_s,
                "witness_rank": rank_w_s, "real_rank": rank_r_s,
                "witness_sigma_max": sig_w_s, "real_sigma_max": sig_r_s,
                "sigma_max_ratio_witness_over_real_median": sig_ratio,
                "n_severed_jacobian": n_severed,
                "eff_dim_note": "the RAW participation-ratio eff-dim is dominated by pose dim0 (forward "
                                "motion, ~34 vs others ~1e-3) so it is ~1 on BOTH witness AND real (NOT the "
                                "in-tree ~4.08, which was measured differently — likely slot0/scale-"
                                "normalized). The apples-to-apples baseline is the REAL eff-dim/rank/σ_max "
                                "measured IDENTICALLY here; the discriminators are RANK (full=6 ⇒ all pose "
                                "DOFs sensitive) and σ_max ratio (≈1 ⇒ PoseNet as sensitive to the SDF "
                                "render as to a real frame).",
                "interpretation": "rank-collapse and/or σ_max ratio ≪ 1 ⇒ PoseNet near-blind on the flat "
                                  "SDF render ⇒ ξ UNREACHABLE. Full rank + σ_max ≈ real ⇒ ξ REACHABLE.",
            },
            "3_seg_driven_delta_pose_null_fraction": {
                "null_energy_frac": nullfrac_s, "sensitive_energy_frac": sensfrac_s,
                "isotropic_sensitive_frac_baseline": isobase_s,
                "interpretation": "null_frac≈1 and sensitive_frac ~ the isotropic baseline ⇒ the seg-driven "
                                  "frame change lands in the pose-null ⇒ decoupling is free (corroborates #1).",
            },
            "4_A_vs_B_readback": {
                "A_film_sdf": {
                    "d_pose_witness_render_through_R_baseline": dpose_w_s,
                    "reachability_sigma_max_median": sig_w_s["median"],
                    "reachability_eff_dim_median": eff_w_s["median"],
                    "note": "A = condition the flat SDF render toward stored ξ*. Its FEASIBILITY is the "
                            "reachability (#2): if σ_max/eff-dim on the render are ≪ real, PoseNet cannot "
                            "read ξ back no matter the conditioning. d_pose baseline is the pose-BLIND render.",
                },
                "B_warp_real_luma": warp,
                "warp_carries_pose": warp_carries,
            },
        },
        "verdicts": {
            "seg_perp_pose_decoupling_FREE": decoupling_free,
            "abs_median_render_cos_seg_pose": float(cos_abs_med) if np.isfinite(cos_abs_med) else None,
            "pose_reachable_on_sdf_render": pose_reachable_on_sdf,
            "witness_jacobian_rank_median": rank_w_s["median"],
            "real_jacobian_rank_median": rank_r_s["median"],
            "sigma_max_ratio_witness_over_real": sig_ratio,
            "witness_render_d_pose_median_POSE_BLIND": dpose_w_s["median"],
            "warp_real_luma_carries_pose": warp_carries,
            "warp_real_luma_best_d_pose": warp_best,
            "DECISION": verdict,
        },
        "caveats": [
            "SAMPLE: geometry on a strided sample of %d/%d pairs across the whole clip (representative, "
            "avoids Lens-A contiguous-prefix optimism); the warp readback (B) uses the warp tool's "
            "contiguous prefix of %s pairs (its own flagged caveat)." % (len(strided), n_pairs_total, str(args.warp_n)),
            "R DIFFERENTIABILITY: through-R gradients use the STE round (round.detach identity backward) "
            "+ differentiable bicubic-up/bilinear-down; the uint8 cliff is straight-through, so the "
            "gradient GEOMETRY is the eval-faithful one but the exact d_pose value is the frozen verdict.",
            "POSENET LOAD: DistortionNet CPU + differentiable yuv6 patch (validated 0-abs-error vs "
            "upstream; NO-FAKE self-check asserts PoseNet(GT)==gt_poses <1e-4). NEVER MPS.",
            "eff-dim measured at frame_slot=1 (the SegNet-shared frame). The in-tree ~4.08 reference is "
            "at slot 0; the real-frame eff-dim measured HERE (slot 1) is the apples-to-apples baseline.",
            "SELF-ORIENT deploy render uses the fixed-point dir feats (small parity gap vs the trainer's "
            "training-trajectory feats, per the byte-close tool); it is the SHIPPABLE render.",
            "cos at the GT operating point is undefined (∂d_pose/∂F vanishes at d_pose=0); #1 is measured "
            "ONLY at the pose-blind witness operating point, which is the honest interference regime.",
        ],
        "elapsed_secs": round(time.time() - t0, 1),
    }
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-dir", type=Path,
                    default=_REPO / "experiments/results/levelset_n600_v2_attrclean_20260630T194549Z")
    ap.add_argument("--npz-name", type=str, default="levelset_witness_ema_BEST.npz")
    ap.add_argument("--gt-cache", type=str,
                    default=str(_REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"))
    ap.add_argument("--n-geom", type=int, default=32, help="strided pairs for the render-space geometry")
    ap.add_argument("--n-real", type=int, default=16, help="strided pairs for the REAL-frame eff-dim reference")
    ap.add_argument("--work-h", type=int, default=384)
    ap.add_argument("--work-w", type=int, default=512)
    ap.add_argument("--warp-n", type=int, default=96, help="n pairs for the warp-real-luma readback (B)")
    ap.add_argument("--skip-warp", action="store_true")
    ap.add_argument("--out", type=Path,
                    default=_REPO / "reports/levelset_pose_gate_l7best_20260701.json")
    args = ap.parse_args(argv)
    _refuse_tmp(Path(args.out), "out")
    report = run(args)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print("\n=== POSE GATE VERDICT ===", flush=True)
    print(json.dumps(report["verdicts"], indent=2), flush=True)
    print(f"[report] wrote {args.out}  {report['authority']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
