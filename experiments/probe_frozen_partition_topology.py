#!/usr/bin/env python3
"""Frozen-instance partition TOPOLOGY + boundary ego-deformation probe ($0/CPU).

Tests the decisive frozen-instance topological claim:

    Is the GT seg partition = (near-CONSTANT topology) + (LOW-DIMENSIONAL
    ego-driven boundary deformation)?

If yes, the frozen-instance-optimal d_seg code is TOPOLOGICAL (constant template
+ ego-deformation field + rendered boundary), NOT a pixel decoder -- and it
unifies with d_pose via the shared ego-motion.

AUTHORITY / NO-FAKE:
- The GT partition is the EXACT frozen-SegNet argmax for the 600 scored frames,
  read from the cached argmaps validated by `frozen_instance_horizon_crossframe`
  (cached d_seg = 0.00055989 vs report 0.00055978, dt=1e-7, exact-scorer
  faithful). Source: experiments/results/indep_dseg_bets_20260623_inflated/
  seg_argmaps.npz key 'gt' shape (600,384,512) uint8.
- Ego-motion: (a) horizon-row trajectory v_h(t) from the EXACT GT argmax
  (road<->undrivable boundary, a3061's authority-faithful method), and
  (b) the real frozen PoseNet 6-dim output on GT pairs (the actual d_pose
  TARGET) via upstream.modules.PoseNet + upstream.frame_utils.yuv420_to_rgb
  (NEVER PyAV). CPU-ONLY, NEVER MPS.
- Score/byte math via tac.contest_score (break-even rate slope).

This measures a DIFFERENT object from a3061 (which measured the flip-RESIDUAL,
full-rank 547/600) and from #52 (which measured the static per-frame partition
byte cost). Here we measure the GT TOPOLOGY constancy + the GT BOUNDARY
deformation intrinsic dimension + its ego-explained fraction.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ARGMAPS = (
    REPO
    / "experiments/results/indep_dseg_bets_20260623_inflated/seg_argmaps.npz"
)
N_CLASSES = 5
# Camera intrinsics (openpilot _neo_config, verified comma2k19 RAV4 segment):
# K=[[910,0,582],[0,910,437]]; SegNet operates at 384x512 (sy=384/874, sx=512/1164).
CAM_H, CAM_W = 874, 1164
SEG_H, SEG_W = 384, 512
FY_CAM, CY_CAM = 910.0, 437.0


def _zlib_len(arr: np.ndarray) -> int:
    import zlib

    return len(zlib.compress(np.ascontiguousarray(arr).tobytes(), 9))


# ---------------------------------------------------------------------------
# STAGE 1 -- partition TOPOLOGY constancy
# ---------------------------------------------------------------------------
def topology_signature(label: np.ndarray):
    """Return (adjacency_frozenset, per_class_component_counts, euler_char).

    adjacency: frozenset of {(a,b)} class-pairs that 4-touch (the region
        adjacency graph at the CLASS level -- which of the 5 classes border).
    component_counts: tuple of 4-connected component counts per class (length 5).
    euler: sum over classes of (components - holes) via scipy.ndimage Euler
        number, a topology invariant of the partition's 1-skeleton.
    """
    import scipy.ndimage as ndi

    # class-level adjacency graph (4-connectivity)
    adj = set()
    # horizontal neighbors
    a = label[:, :-1]
    b = label[:, 1:]
    diff = a != b
    for ca, cb in zip(a[diff].ravel(), b[diff].ravel(), strict=False):
        adj.add((int(min(ca, cb)), int(max(ca, cb))))
    a = label[:-1, :]
    b = label[1:, :]
    diff = a != b
    for ca, cb in zip(a[diff].ravel(), b[diff].ravel(), strict=False):
        adj.add((int(min(ca, cb)), int(max(ca, cb))))

    comp_counts = []
    euler_total = 0.0
    struct = ndi.generate_binary_structure(2, 1)  # 4-connectivity
    for c in range(N_CLASSES):
        mask = label == c
        if not mask.any():
            comp_counts.append(0)
            continue
        _, ncomp = ndi.label(mask, structure=struct)
        comp_counts.append(int(ncomp))
        # Euler number (components - holes) for this class binary image
        try:
            from skimage.measure import euler_number

            euler_total += float(euler_number(mask, connectivity=1))
        except Exception:
            # fallback: components only (no hole count) -- still a coarse invariant
            euler_total += float(ncomp)
    return frozenset(adj), tuple(comp_counts), euler_total


def stage1_topology(gt: np.ndarray) -> dict:
    n = gt.shape[0]
    adj_list = []
    comp_list = []
    euler_list = []
    for i in range(n):
        adj, comps, eul = topology_signature(gt[i])
        adj_list.append(adj)
        comp_list.append(comps)
        euler_list.append(eul)

    # adjacency-graph constancy
    adj_counter = Counter(adj_list)
    modal_adj, modal_adj_count = adj_counter.most_common(1)[0]
    # full-class adjacency (all 5 classes mutually touch?) reference
    comp_arr = np.array(comp_list, dtype=float)  # (n, 5)
    euler_arr = np.array(euler_list, dtype=float)

    # component-count variance per class
    comp_mean = comp_arr.mean(axis=0)
    comp_std = comp_arr.std(axis=0)
    # modal component-count vector
    comp_tuples = [tuple(int(x) for x in row) for row in comp_arr]
    comp_counter = Counter(comp_tuples)
    modal_comp, modal_comp_count = comp_counter.most_common(1)[0]

    # total topology signature (adj + comp + rounded euler) constancy
    sig_list = [
        (adj_list[i], comp_tuples[i], round(euler_list[i])) for i in range(n)
    ]
    sig_counter = Counter(sig_list)
    modal_sig, modal_sig_count = sig_counter.most_common(1)[0]

    return {
        "n_frames": n,
        "num_distinct_adjacency_graphs": len(adj_counter),
        "modal_adjacency_frac": modal_adj_count / n,
        "modal_adjacency_graph": sorted([list(p) for p in modal_adj]),
        "num_distinct_component_count_vectors": len(comp_counter),
        "modal_component_count_frac": modal_comp_count / n,
        "modal_component_counts_per_class": list(modal_comp),
        "component_count_mean_per_class": comp_mean.round(3).tolist(),
        "component_count_std_per_class": comp_std.round(3).tolist(),
        "total_components_mean": float(comp_arr.sum(axis=1).mean()),
        "total_components_std": float(comp_arr.sum(axis=1).std()),
        "euler_mean": float(euler_arr.mean()),
        "euler_std": float(euler_arr.std()),
        "num_distinct_full_signatures": len(sig_counter),
        "modal_full_signature_frac": modal_sig_count / n,
    }


# ---------------------------------------------------------------------------
# STAGE 2 -- boundary-deformation intrinsic dimension
# ---------------------------------------------------------------------------
def per_column_class_boundary_rows(label: np.ndarray) -> np.ndarray:
    """For each column, return the row of the FIRST class-transition from the
    top going down (the upper boundary contour of the scene). This is a
    per-column 1D representation of the partition's dominant horizon-type
    boundary -- a low-dim curve representation if the deformation is ego-driven.

    Returns (W,) float row index of first top-down class change (or H if none).
    """
    H, W = label.shape
    rows = np.full(W, H, dtype=np.float64)
    # first row index where label[r,col] != label[0,col]
    top = label[0:1, :]  # (1,W) topmost class per column
    change = label != top  # (H,W)
    # argmax of change along rows gives first True (0 if none -> mask it)
    first = np.argmax(change, axis=0).astype(np.float64)
    has = change.any(axis=0)
    rows[has] = first[has]
    return rows


def boundary_curve_repr(label: np.ndarray) -> np.ndarray:
    """A richer per-column boundary representation: for each column, the row
    of the road<->{sky/undrivable} upper boundary AND the road lower extent.
    Concatenated -> (2W,) curve vector. Captures the ego-driven scene geometry
    (horizon line + road trapezoid) as a low-dim curve if the claim holds.
    Class convention (comma10k SegNet): we use generic 'first transition' and
    'road class' boundaries without hard-coding semantics beyond road=modal-
    ground class detection.
    """
    H, W = label.shape
    upper = per_column_class_boundary_rows(label)
    # road class = the class occupying the most pixels in the lower-middle band
    band = label[H // 2 :, :]
    road_cls = int(np.bincount(band.ravel(), minlength=N_CLASSES).argmax())
    road = label == road_cls
    # topmost road row per column (the road horizon)
    road_top = np.full(W, H, dtype=np.float64)
    has_road = road.any(axis=0)
    rt = np.argmax(road, axis=0).astype(np.float64)
    road_top[has_road] = rt[has_road]
    return np.concatenate([upper, road_top]), road_cls


def stage2_boundary_dim(gt: np.ndarray) -> dict:
    n, H, W = gt.shape
    # build boundary-curve matrix (n, 2W)
    curves = np.zeros((n, 2 * W), dtype=np.float64)
    road_cls_list = []
    for i in range(n):
        cv, rc = boundary_curve_repr(gt[i])
        curves[i] = cv
        road_cls_list.append(rc)
    # clip H-sentinels to H (already H)
    # center
    mean_curve = curves.mean(axis=0)
    centered = curves - mean_curve
    # SVD intrinsic dimension
    # use economy SVD on centered (n x 2W)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    var = S**2
    var_ratio = var / var.sum()
    cum = np.cumsum(var_ratio)
    # participation ratio (effective rank)
    pr = (var.sum() ** 2) / (var**2).sum()
    dims_for = {
        f"k_for_{int(p*100)}pct": int(np.searchsorted(cum, p) + 1)
        for p in (0.50, 0.80, 0.90, 0.95, 0.99)
    }
    # horizon-row trajectory v_h(t): median over columns of the upper road
    # boundary (a3061 method)
    vh = curves[:, W:].copy()  # road_top half
    vh[vh >= H] = np.nan
    vh_traj = np.nanmedian(vh, axis=1)  # (n,)
    return {
        "n_frames": n,
        "curve_dim": int(2 * W),
        "boundary_effective_rank_participation_ratio": float(pr),
        "boundary_singular_values_top10": S[:10].round(2).tolist(),
        "boundary_top1_var_share": float(var_ratio[0]),
        "boundary_top3_var_share": float(var_ratio[:3].sum()),
        "boundary_top6_var_share": float(var_ratio[:6].sum()),
        **dims_for,
        "vh_traj_mean": float(np.nanmean(vh_traj)),
        "vh_traj_std": float(np.nanstd(vh_traj)),
        "road_class_modal": int(Counter(road_cls_list).most_common(1)[0][0]),
        "_curves_centered": centered,  # passed to stage3 (not serialized)
        "_curves": curves,
        "_mean_curve": mean_curve,
        "_vh_traj": vh_traj,
        "_U": U,
        "_S": S,
        "_Vt": Vt,
    }


# ---------------------------------------------------------------------------
# STAGE 3 -- ego-motion unification
# ---------------------------------------------------------------------------
def run_posenet_ego(n_frames: int, subsample: int) -> np.ndarray | None:
    """Run the frozen PoseNet on GT pairs to get the 6-dim d_pose TARGET
    ego-motion. CPU-ONLY, NEVER MPS. Returns (n,6) or None if unavailable."""
    try:
        import einops
        import torch

        sys.path.insert(0, str(REPO / "upstream"))
        import av
        import modules
        from frame_utils import yuv420_to_rgb
        from safetensors.torch import load_file
    except Exception as e:  # pragma: no cover
        print(f"[stage3] PoseNet path unavailable: {e}", file=sys.stderr)
        return None

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    device = torch.device("cpu")  # NEVER MPS
    sd_path = REPO / "upstream/models/posenet.safetensors"
    if not sd_path.exists():
        alts = list((REPO / "upstream").rglob("posenet.safetensors"))
        if alts:
            sd_path = alts[0]
    pose = modules.PoseNet()
    try:
        pose.load_state_dict(load_file(str(sd_path)))
    except Exception as e:  # pragma: no cover
        print(f"[stage3] PoseNet weights load failed: {e}", file=sys.stderr)
        return None
    pose.eval().to(device)

    # decode GT frames; build pairs as upstream does: seq_len=2 NON-OVERLAPPING
    # (frames [2k, 2k+1] = pair k). 600 pairs from 1200 frames.
    vid = REPO / "upstream/videos/0.mkv"
    container = av.open(str(vid))
    frames = []
    max_needed = (n_frames * subsample) * 2 + 2
    for f in container.decode(video=0):
        frames.append(yuv420_to_rgb(f))  # (H,W,3) uint8 torch
        if len(frames) >= max_needed:
            break
    container.close()

    poses = []
    idxs = list(range(0, n_frames * subsample, subsample))
    with torch.inference_mode():
        for pi in idxs:
            a = frames[2 * pi].float()
            b = frames[2 * pi + 1].float()
            pair = torch.stack([a, b]).unsqueeze(0)  # (1,2,H,W,3)
            x = einops.rearrange(pair, "b t h w c -> b t c h w").float()
            x = pose.preprocess_input(x)
            out = pose(x.to(device))  # dict {'pose': (1,12)}
            # d_pose target uses first out//2 = 6 dims (PoseNet.compute_distortion)
            poses.append(out["pose"].squeeze(0).cpu().numpy()[:6])
    return np.array(poses, dtype=np.float64)


def explained_fraction(deform: np.ndarray, ego: np.ndarray) -> dict:
    """Linear regression: what fraction of boundary deformation variance is
    explained by the ego-motion regressors (least squares R^2, multi-output)."""
    # deform: (n, D) centered boundary deformation; ego: (n, k) ego regressors
    n = deform.shape[0]
    deform = np.nan_to_num(deform, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float64
    )
    ego = np.nan_to_num(ego, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)
    # standardize ego regressors to avoid ill-conditioning
    ego_c = ego - ego.mean(axis=0)
    ego_sd = ego_c.std(axis=0)
    ego_sd[ego_sd == 0] = 1.0
    ego_z = ego_c / ego_sd
    X = np.column_stack([np.ones(n), ego_z])  # intercept + standardized ego
    # solve least squares for all D outputs (rcond guards rank-deficiency)
    beta, *_ = np.linalg.lstsq(X, deform, rcond=1e-10)
    pred = X @ beta
    ss_res = float(((deform - pred) ** 2).sum())
    ss_tot = float((deform**2).sum())  # deform already centered
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    # clamp numerical noise
    r2 = max(min(r2, 1.0), -1.0)
    return {"r2_overall": float(r2), "n": int(n), "k_regressors": int(ego.shape[1])}


def stage3_ego(s2: dict, gt: np.ndarray, subsample: int) -> dict:
    centered = s2["_curves_centered"]  # (n, 2W) boundary deformation
    vh = s2["_vh_traj"]  # (n,) horizon-row trajectory (ego pitch proxy)
    n = centered.shape[0]
    out = {}

    # (A) horizon-row proxy ego regressors: v_h(t) + its delta (pitch + pitch rate)
    vh_filled = np.where(np.isnan(vh), np.nanmean(vh), vh)
    dvh = np.gradient(vh_filled)
    ego_proxy = np.column_stack([vh_filled, dvh])
    out["explained_by_horizon_proxy"] = explained_fraction(centered, ego_proxy)

    # (B) real PoseNet 6-dim ego-motion (the d_pose target)
    pose = run_posenet_ego(n, subsample)
    if pose is not None and pose.shape[0] == n:
        out["posenet_available"] = True
        out["posenet_pose_std_per_dim"] = pose.std(axis=0).round(5).tolist()
        out["explained_by_posenet6"] = explained_fraction(centered, pose)
        # cumulative ego: cumulative sum of per-pair pose (trajectory)
        ego_cum = np.cumsum(pose, axis=0)
        out["explained_by_posenet6_cumulative"] = explained_fraction(
            centered, ego_cum
        )
        # unification: does v_h correlate with PoseNet pitch-like dims?
        corrs = []
        for d in range(6):
            c = np.corrcoef(vh_filled, pose[:, d])[0, 1]
            corrs.append(float(c) if np.isfinite(c) else 0.0)
        out["corr_vh_vs_posenet_dims"] = [round(c, 3) for c in corrs]
        out["max_abs_corr_vh_posenet"] = float(max(abs(c) for c in corrs))
    else:
        out["posenet_available"] = False

    return out


# ---------------------------------------------------------------------------
# STAGE 4 -- byte / S projection of a topological code
# ---------------------------------------------------------------------------
def stage4_byte_projection(s1: dict, s2: dict, s3: dict, gt: np.ndarray) -> dict:
    from tac.contest_score import compute_contest_score as contest_score
    from tac.contest_score import rate_term

    n = gt.shape[0]
    centered = s2["_curves_centered"]
    U, S = s2["_U"], s2["_S"]

    # Topological code byte estimate:
    # (1) constant template: the modal partition stored once (#52 anchor ~896 B,
    #     but the template is a SINGLE frame -> ~896 B once, amortized).
    # (2) ego-deformation: store the low-rank coefficients per frame +
    #     the residual after the ego-explained reconstruction.
    template_bytes = 896  # #52 per-frame LZMA contour (single template)

    # low-rank deformation coefficients: keep k components s.t. 95% var
    k95 = s2.get("k_for_95pct", n)
    coeffs = U[:, :k95] * S[:k95]  # (n, k95) projection coefficients
    # quantize coeffs to int16 + zlib (delta along time)
    cq = np.round(coeffs).astype(np.int32)
    cq_delta = np.diff(cq, axis=0, prepend=cq[:1])
    coeff_bytes = _zlib_len(cq_delta.astype(np.int16))

    # residual after k95-rank reconstruction (what the low-rank deform misses)
    Vt = s2["_Vt"]
    k95c = min(k95, Vt.shape[0])
    recon = (U[:, :k95c] * S[:k95c]) @ Vt[:k95c]
    resid = centered - recon
    resid_rms = float(np.sqrt((resid**2).mean()))

    # Also: ego-parameterized version -- store only the ego regressors that
    # explain the deformation (cheapest if ego R^2 high).
    # ego coeffs already cheap (n*6 ints); residual after ego is the cost.

    total_topo_bytes = template_bytes + coeff_bytes

    # The HONEST d_seg of a topological code: a contour/boundary code that
    # reproduces the partition lands d_seg=0 IF it stores the partition exactly
    # (#52). The low-rank deformation is LOSSY -> reintroduces flips. We bound:
    # the residual_rms (rows) maps to boundary flips. We do NOT claim d_seg=0
    # for the lossy low-rank code. Instead report the byte cost of the LOSSLESS
    # template+deformation (which is #52's 524.8 KB extrapolation) vs the
    # low-rank lossy code.
    lossless_600 = 896 * n  # #52 per-frame lossless = no cross-frame reuse

    # Projections via tac.contest_score (rate term only; d_seg/d_pose held at
    # frontier to isolate the rate axis of the topological code).
    L13_witness = 72217
    frontier_bytes = 177169
    # topological code as a SEG carrier (lossless template+deform) -- compare to
    # #52's lossless and to the L13 witness rate.
    def s_proj(bytes_total, dseg, dpose):
        return contest_score(dseg, dpose, bytes_total)

    return {
        "template_bytes_once": template_bytes,
        "low_rank_k95": int(k95),
        "low_rank_coeff_bytes_600": int(coeff_bytes),
        "low_rank_residual_rms_rows": resid_rms,
        "topological_lowrank_total_bytes": int(total_topo_bytes),
        "lossless_template_per_frame_extrapolation_600": int(lossless_600),
        "ref_52_lossless_524KB": 524800,
        "ref_L13_witness_bytes": L13_witness,
        "ref_frontier_bytes": frontier_bytes,
        "break_even_dseg_per_byte": float(rate_term(1) / 100),
        # If the topological LOW-RANK code were lossless (it is not -- residual
        # nonzero), its byte cost vs L13:
        "topo_lowrank_vs_L13_ratio": float(total_topo_bytes / L13_witness),
        "topo_lowrank_vs_frontier_ratio": float(total_topo_bytes / frontier_bytes),
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsample", type=int, default=1,
                    help="frame subsample stride (1 = all 600)")
    ap.add_argument("--limit", type=int, default=0,
                    help="limit frames (0 = all)")
    ap.add_argument("--skip-posenet", action="store_true")
    ap.add_argument("--out", type=str,
                    default="reports/frozen_partition_topology.json")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    t0 = time.time()
    d = np.load(ARGMAPS)
    gt = d["gt"]  # (600,384,512) uint8 -- EXACT frozen-SegNet argmax
    if args.subsample > 1:
        gt = gt[:: args.subsample]
    if args.limit:
        gt = gt[: args.limit]
    print(f"[load] gt {gt.shape} in {time.time()-t0:.1f}s", file=sys.stderr)

    t1 = time.time()
    s1 = stage1_topology(gt)
    print(f"[stage1] topology in {time.time()-t1:.1f}s", file=sys.stderr)

    t2 = time.time()
    s2 = stage2_boundary_dim(gt)
    print(f"[stage2] boundary-dim in {time.time()-t2:.1f}s", file=sys.stderr)

    t3 = time.time()
    if args.skip_posenet:
        s3 = {"posenet_available": False, "skipped": True}
        # still do the horizon proxy
        centered = s2["_curves_centered"]
        vh = s2["_vh_traj"]
        vh_filled = np.where(np.isnan(vh), np.nanmean(vh), vh)
        dvh = np.gradient(vh_filled)
        s3["explained_by_horizon_proxy"] = explained_fraction(
            centered, np.column_stack([vh_filled, dvh])
        )
    else:
        s3 = stage3_ego(s2, gt, args.subsample)
    print(f"[stage3] ego in {time.time()-t3:.1f}s", file=sys.stderr)

    t4 = time.time()
    s4 = stage4_byte_projection(s1, s2, s3, gt)
    print(f"[stage4] byte-proj in {time.time()-t4:.1f}s", file=sys.stderr)

    # strip private arrays
    s2_pub = {k: v for k, v in s2.items() if not k.startswith("_")}
    result = {
        "authority": "contest-CPU advisory (exact frozen-SegNet argmax cache)",
        "n_frames_measured": int(gt.shape[0]),
        "subsample": args.subsample,
        "argmaps_source": str(ARGMAPS.relative_to(REPO)),
        "stage1_topology": s1,
        "stage2_boundary_dim": s2_pub,
        "stage3_ego": s3,
        "stage4_byte_projection": s4,
        "elapsed_sec": time.time() - t0,
    }
    outp = REPO / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(result, indent=2))
    print(f"[done] wrote {outp} in {result['elapsed_sec']:.1f}s", file=sys.stderr)
    print(json.dumps({k: result[k] for k in (
        "stage1_topology", "stage2_boundary_dim", "stage3_ego",
        "stage4_byte_projection")}, indent=2))


if __name__ == "__main__":
    main()
