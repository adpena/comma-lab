#!/usr/bin/env python
"""ddm_pi2 — PoseNet per-dim inversion (the SegNet-head-analog for PoseNet).

Charter (operator 2026-07-30, ledger QA51): "Invert PoseNet the way you inverted
SegNet's head — per-dim null space, response function, cheapest input that moves each dim."

Four deliverables (each with honest EXACT vs PER-PAIR-MEASURED vs CROSS-PAIR-MEASURED labels):
  1. EXACT ALGEBRA         (subcmd `algebra`): final head row-space rank + resize null + f16 floors
  2. PER-DIM RESPONSE ATLAS(subcmd `atlas`) : ∂pose_i/∂input fields — spatial/channel/freq/linearity
  3. PER-DIM NULL + XPAIR   (subcmd `svd`)  : cross-pair shared-basis rank per dim (THE QA47 pre-answer)
  4. CHEAPEST INPUT PER DIM (subcmd `cheapest`): steering atoms, coded-bytes price, realized B/unit

Authority: MPS = legal GRADIENT device only (patch_scorer_for_mps; 104x). Realized checks through
frozen CPU-torch. NEVER an MPS score. All rows [macOS-CPU advisory]; score_claim=false.
Pointer 0.1910828242 [contest-CPU] UNMOVED.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
SSD = Path("/Volumes/VertigoDataTier/pact/ddm_pi2_20260730")
GT_N96 = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n96.npz"
POSENET_SD = REPO / "upstream/models/posenet.safetensors"

# BT.601 luma direction (frame_utils rgb_to_yuv6)
LUMA = np.array([0.299, 0.587, 0.114], dtype=np.float64)
DIM_NAMES = ["p0", "p1", "p2", "p3", "p4", "p5"]  # first 6 scored PoseNet dims


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def _load_posenet(device: str = "cpu"):
    import modules  # upstream on PYTHONPATH
    from safetensors.torch import load_file

    pn = modules.PoseNet().eval()
    sd = load_file(str(POSENET_SD), device="cpu")
    pn.load_state_dict(sd)
    pn = pn.to(device)
    for p in pn.parameters():
        p.requires_grad_(False)
    return pn


# ----------------------------------------------------------------------------- algebra
def cmd_algebra(args: argparse.Namespace) -> None:
    """Deliverable 1 — the EXACT algebra (weight inspection; no forward passes)."""
    out = SSD / "algebra_receipt.json"
    pn = _load_posenet("cpu")

    # --- (a) the final head is EXACTLY linear in the 32-dim post-res_layer feature b ---
    W = pn.hydra.final_layer["pose"].weight.detach().double().numpy()  # (12,32)
    W6 = W[:6, :]                                                       # (6,32) scored rows
    U, S, Vt = np.linalg.svd(W6, full_matrices=False)                  # S: (6,)
    rank_tol = S.max() * max(W6.shape) * np.finfo(np.float64).eps
    rank = int((rank_tol < S).sum())
    # unscored rows 6..11 span span; is scored row-space independent of them?
    Wun = W[6:, :]
    # principal angles between scored row-space and unscored row-space
    Qs = np.linalg.qr(W6.T)[0]      # (32,6) ON basis of scored row-space
    Qu = np.linalg.qr(Wun.T)[0]     # (32,6)
    sv_cross = np.linalg.svd(Qs.T @ Qu, compute_uv=False)
    principal_angles_deg = np.degrees(np.arccos(np.clip(sv_cross, -1, 1))).tolist()

    # --- (b) resize null (EXACT, separable bilinear) ---
    cam_h, cam_w = 874, 1164
    rs_h, rs_w = 384, 512
    cam_dof = cam_h * cam_w
    rs_dof = rs_h * rs_w
    resize_null_dof = cam_dof - rs_dof  # per channel per frame (bilinear downsample is full row rank)
    resize_null_frac = resize_null_dof / cam_dof

    # --- (c) per-dim f16 output-quantization floors, from the real 600 pose targets ---
    # load banked 600 targets if available, else gt_n96 poses
    try:
        import tac.scorer_targets as st
        tgt = st.load_posenet_targets(str(REPO / "experiments/posenet_targets.bin"))
        tgt = np.asarray(tgt, dtype=np.float64)
        if tgt.ndim == 2 and tgt.shape[1] >= 6:
            poses = tgt[:, :6]
            tgt_src = "posenet_targets.bin (600)"
        else:
            raise ValueError("unexpected target shape")
    except Exception:
        d = np.load(GT_N96)
        poses = d["gt_poses"][:, :6].astype(np.float64)
        tgt_src = "gt_n96 gt_poses (96)"

    per_dim = []
    for i in range(6):
        v = poses[:, i]
        amax = float(np.abs(v).max())
        # f16 ulp at the max magnitude: 2^(floor(log2|amax|)) * 2^-10
        if amax > 0:
            exp = np.floor(np.log2(amax))
            f16_ulp_at_max = float(2.0 ** (exp - 10))
        else:
            f16_ulp_at_max = 0.0
        # d_pose contribution of a per-dim error e: e^2/6 summed; S = sqrt(10*d_pose)
        # a single-dim f16 ulp error -> d_pose = ulp^2/6 -> contribution
        dpose_from_ulp = f16_ulp_at_max ** 2 / 6.0
        contrib_from_ulp = float(np.sqrt(10.0 * dpose_from_ulp))
        per_dim.append({
            "dim": DIM_NAMES[i],
            "mean": float(v.mean()), "std": float(v.std()),
            "min": float(v.min()), "max": float(v.max()), "abs_max": amax,
            "f16_ulp_at_abs_max": f16_ulp_at_max,
            "S_contrib_from_single_dim_f16_ulp": contrib_from_ulp,
        })

    receipt = {
        "schema": "ddm_pi2_algebra.v1", "utc": _now(),
        "axis": "[macOS-CPU advisory] EXACT weight algebra; score_claim=false",
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "posenet_sd": str(POSENET_SD),
        "final_head_linear": {
            "weight_shape": list(W.shape), "scored_rows": 6,
            "scored_singular_values": S.tolist(),
            "scored_rank": rank, "rank_tol": float(rank_tol),
            "cond_number": float(S.max() / S[rank_tol < S].min()) if rank > 0 else None,
            "principal_angles_scored_vs_unscored_deg": principal_angles_deg,
            "head_null_dim_in_32d_feature": 32 - rank,
            "note": ("the 6 scored pose scalars are an EXACT linear map W6@b+bias6 of the 32-dim "
                     "post-res_layer ReLU feature b; everything orthogonal to row-space(W6) in that "
                     "32-dim space is EXACTLY head-null for the score (rank analog of SegNet rank-4)."),
        },
        "resize_null_exact": {
            "camera": [cam_h, cam_w], "resized": [rs_h, rs_w],
            "null_dof_per_ch_per_frame": resize_null_dof,
            "null_frac_of_camera_dof": resize_null_frac,
            "note": ("A_pose == A_seg (shared bilinear downsample, modules.py:73==:109); separable "
                     "bilinear downsample is full output-row-rank -> ker dim = cam_dof - resized_dof, "
                     "EXACTLY invisible to BOTH scorers (frozen memo B1). Chroma 2x2 box-avg adds "
                     "further pose-only null above 2px@(384,512)."),
        },
        "per_dim_f16_floor": {"target_source": tgt_src, "dims": per_dim},
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=1))
    print(f"[algebra] scored head singular values = {np.array(S).round(4).tolist()}")
    print(f"[algebra] scored rank = {rank} / 6  (head-null dim in 32d feature = {32-rank})")
    print(f"[algebra] min principal angle scored-vs-unscored row space = {min(principal_angles_deg):.2f} deg")
    print(f"[algebra] resize null (EXACT) = {resize_null_dof} DOF/ch/frame = {resize_null_frac*100:.1f}% of camera DOF")
    for pd in per_dim:
        print(f"[algebra] {pd['dim']}: |val|<= {pd['abs_max']:.3f}  f16_ulp {pd['f16_ulp_at_abs_max']:.4g}  "
              f"-> single-dim-ulp S contrib {pd['S_contrib_from_single_dim_f16_ulp']:.4g}")
    print(f"[algebra] receipt -> {out}")


# --------------------------------------------------------------------- differentiable forward
class PoseForward:
    """resized-RGB (2,3,384,512) float[0,255] -> pose[:6], fully differentiable.

    Replicates PoseNet.preprocess_input (yuv6 + rearrange + norm) + forward, but takes the
    resized RGB as the leaf so the input-Jacobian is post the shared resize A (frozen memo P2).
    Uses tac.differentiable_eval_roundtrip.differentiable_rgb_to_yuv6 (upstream yuv6 is
    @no_grad + in-place clamp_ -> severs grad; PR95/106 monkeypatch precedent).
    """

    def __init__(self, device: str = "mps"):
        import tac.differentiable_eval_roundtrip as de
        from tac.torch_mps_compat import patch_scorer_for_mps

        self.device = device
        if device == "mps":
            patch_scorer_for_mps()  # global BN-contiguous MPS patch (no per-model arg)
        self.pn = _load_posenet(device)
        self.yuv6 = de.differentiable_rgb_to_yuv6

    def __call__(self, rgb_pair: torch.Tensor) -> torch.Tensor:
        # rgb_pair: (2,3,384,512) float [0,255]
        yuv = self.yuv6(rgb_pair)                       # (2,6,192,256)
        x = yuv.reshape(1, 12, 192, 256)                # b (t c) h w  (raw yuv; forward() normalizes)
        out = self.pn(x)["pose"][0, :6]                 # (6,)  PoseNet.forward applies (x-mean)/std
        return out


def _resize_to_model(cam_rgb: torch.Tensor) -> torch.Tensor:
    """(2,3,874,1164) -> (2,3,384,512) bilinear, matching modules.py:73 (align_corners=False)."""
    return torch.nn.functional.interpolate(cam_rgb, size=(384, 512), mode="bilinear")


def _upstream_pose6(pn, cam_rgb_pair: torch.Tensor) -> torch.Tensor:
    """Exact upstream path for equivalence check: (2,3,874,1164)->pose[:6] via preprocess_input."""
    x = cam_rgb_pair.unsqueeze(0)  # (1,2,3,874,1164)
    xin = pn.preprocess_input(x)   # upstream yuv6 path
    with torch.no_grad():
        out = pn(xin)["pose"][0, :6]
    return out


def cmd_smoke(args: argparse.Namespace) -> None:
    """Verify the differentiable forward matches the exact upstream PoseNet on a real pair."""
    import tac.differentiable_eval_roundtrip as de
    d = np.load(GT_N96)
    dev = args.device
    # upstream reference on CPU (authority)
    pn_cpu = _load_posenet("cpu")
    # yuv6 equivalence check (tac helper vs upstream), on a real resized frame
    f0 = torch.from_numpy(d["gt_f0"][0]).permute(2, 0, 1).float()  # (3,874,1164)
    f1 = torch.from_numpy(d["gt_f1"][0]).permute(2, 0, 1).float()
    cam = torch.stack([f0, f1])                                    # (2,3,874,1164)
    rs = _resize_to_model(cam)                                     # (2,3,384,512)
    try:
        res = de.assert_yuv6_forward_equivalence_to_upstream()
        yuv_equiv = f"PASS:{res}"
    except Exception as e:
        yuv_equiv = f"NOTE:{type(e).__name__}:{e}"
    # direct elementwise yuv6 check on this real resized frame (belt-and-suspenders)
    from frame_utils import rgb_to_yuv6 as _up_yuv6
    yuv6_maxabs = float((_up_yuv6(rs) - de.differentiable_rgb_to_yuv6(rs)).abs().max())
    # full-forward equivalence: my differentiable path (cpu) vs upstream (cpu)
    fwd_cpu = PoseForward("cpu")
    my6 = fwd_cpu(rs.to("cpu")).detach().double().numpy()
    up6 = _upstream_pose6(pn_cpu, cam.to("cpu")).double().numpy()
    max_abs = float(np.abs(my6 - up6).max())
    # mps-vs-cpu gradient-device drift (forward only; MPS is grad device, never a score)
    mps_note = None
    if dev == "mps" and torch.backends.mps.is_available():
        fwd_mps = PoseForward("mps")
        my6_mps = fwd_mps(rs.to("mps")).detach().float().cpu().double().numpy()
        mps_note = {"max_abs_mps_vs_cpu_forward": float(np.abs(my6_mps - my6).max())}
    row = {
        "schema": "ddm_pi2_smoke.v1", "utc": _now(),
        "yuv6_equivalence": yuv_equiv, "yuv6_maxabs_this_frame": yuv6_maxabs,
        "fwd_max_abs_diff_diff_vs_upstream_cpu": max_abs,
        "my_pose6_cpu": my6.tolist(), "upstream_pose6_cpu": up6.tolist(),
        "mps_forward_drift": mps_note,
        "note": "differentiable forward vs exact upstream on real pair 0; MPS=grad device only.",
    }
    _append_jsonl(SSD / "smoke_receipt.jsonl", row)
    print(f"[smoke] yuv6 equivalence: {yuv_equiv}")
    print(f"[smoke] full-forward max_abs(diff vs upstream, CPU) = {max_abs:.3e}")
    if mps_note:
        print(f"[smoke] MPS-vs-CPU forward drift = {mps_note['max_abs_mps_vs_cpu_forward']:.3e} (grad device only)")
    print(f"[smoke] pose6 (cpu) = {my6.round(4).tolist()}")


# ------------------------------------------------------------------ atlas (deliverable 2 + fields)
LUMA_HAT = LUMA / np.linalg.norm(LUMA)  # unit luma direction in RGB


def _select_pairs(n: int) -> list[int]:
    """Stratified pair set over the 96 real gt_n96 pairs: hard-cluster + tail + random controls.

    pfs1 §5: the 71-90 turn/dynamic cluster carries the d_pose tail (worst pair 77 @ 4.83);
    median pair solves to ~0.003. We stratify by ||gt_pose|| (motion magnitude proxy) + include
    the known hard cluster, so the Jacobian atlas covers straight/turn/dynamic content.
    """
    d = np.load(GT_N96)
    poses = d["gt_poses"][:, :6].astype(np.float64)
    mag = np.linalg.norm(poses, axis=1)
    order = np.argsort(-mag)  # descending motion magnitude
    hard = list(order[:17])
    tail = list(order[17:17 + 24])
    rng = np.random.default_rng(20260730)
    rest = [i for i in range(len(mag)) if i not in set(hard) | set(tail)]
    ctrl = list(rng.choice(rest, size=min(12, len(rest)), replace=False))
    sel = hard + tail + ctrl
    if n and n < len(sel):
        sel = hard[: max(1, n // 3)] + tail[: max(1, n // 3)] + ctrl[: max(1, n // 3)]
    return [int(i) for i in sel]


def _band_energy(mag_hw: np.ndarray) -> dict:
    """Energy in top/mid/bottom thirds + energy-weighted centroid row (0=top, 383=bottom)."""
    H = mag_hw.shape[0]
    e = mag_hw.sum(axis=1)  # per-row energy (sum over W)
    tot = float(e.sum()) + 1e-30
    t3 = H // 3
    rows = np.arange(H)
    return {
        "far_top_frac": float(e[:t3].sum() / tot),
        "mid_frac": float(e[t3:2 * t3].sum() / tot),
        "near_bottom_frac": float(e[2 * t3:].sum() / tot),
        "centroid_row_frac": float((rows * e).sum() / tot / (H - 1)),
    }


def _radial_spectrum(field_hw: np.ndarray) -> dict:
    """Radial power spectrum of a (H,W) field -> low/mid/high fraction (fraction of max radius)."""
    F = np.fft.fftshift(np.fft.fft2(field_hw - field_hw.mean()))
    P = (F.real ** 2 + F.imag ** 2)
    H, W = P.shape
    cy, cx = H // 2, W // 2
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)  # normalized radius
    tot = float(P.sum()) + 1e-30
    return {
        "lf_frac": float(P[r <= 0.15].sum() / tot),
        "mf_frac": float(P[(r > 0.15) & (r <= 0.5)].sum() / tot),
        "hf_frac": float(P[r > 0.5].sum() / tot),
    }


def cmd_atlas(args: argparse.Namespace) -> None:
    """Deliverable 2 — per-dim input-Jacobian atlas; saves field stacks for svd/cheapest."""
    sel = _select_pairs(args.n)
    d = np.load(GT_N96)
    fwd = PoseForward(args.device)
    dev = args.device
    fielddir = SSD / "fields"
    fielddir.mkdir(parents=True, exist_ok=True)
    summ_path = SSD / "atlas_summary.jsonl"
    # done-set for resume
    done = set()
    if summ_path.exists():
        for ln in summ_path.read_text().splitlines():
            try:
                done.add(json.loads(ln)["pair"])
            except Exception:
                pass
    # field stack accumulators (per dim): saved incrementally to SSD as fp16
    print(f"[atlas] selected {len(sel)} pairs; device={dev}; already-done={len(done)}")
    for pi in sel:
        if pi in done:
            continue
        t0 = time.time()
        f0 = torch.from_numpy(d["gt_f0"][pi]).permute(2, 0, 1).float()
        f1 = torch.from_numpy(d["gt_f1"][pi]).permute(2, 0, 1).float()
        cam = torch.stack([f0, f1])
        rs = _resize_to_model(cam).to(dev)
        leaf = rs.clone().requires_grad_(True)      # (2,3,384,512)
        pose6 = fwd(leaf)                            # (6,)
        grads = np.zeros((6, 2, 3, 384, 512), dtype=np.float32)
        for i in range(6):
            g = torch.autograd.grad(pose6[i], leaf, retain_graph=(i < 5))[0]
            grads[i] = g.detach().float().cpu().numpy()
        # save fp16 field stack for this pair (for svd/cheapest)
        np.save(fielddir / f"pair_{pi:03d}.npy", grads.astype(np.float16))
        # per-dim summaries
        for i in range(6):
            g = grads[i]  # (2,3,384,512)
            row_summ = {"schema": "ddm_pi2_atlas.v1", "pair": pi, "dim": DIM_NAMES[i], "utc": _now()}
            # frame split
            e_f = [float((g[t] ** 2).sum()) for t in range(2)]
            row_summ["frame_energy"] = {"f0": e_f[0], "f1": e_f[1],
                                        "f0_frac": e_f[0] / (sum(e_f) + 1e-30)}
            # per-frame channel (luma vs chroma) + spatial band + spectrum on |grad|
            for t in range(2):
                gt = g[t]  # (3,384,512)
                # luma vs chroma-plane projection at each pixel
                gflat = gt.reshape(3, -1)                       # (3, N)
                luma_comp = (LUMA_HAT[:, None] * gflat).sum(0)  # (N,)
                luma_e = float((luma_comp ** 2).sum())
                tot_e = float((gflat ** 2).sum()) + 1e-30
                chroma_e = max(tot_e - luma_e, 0.0)
                mag = np.sqrt((gt ** 2).sum(0))                 # (384,512) grad magnitude
                row_summ[f"f{t}"] = {
                    "luma_frac": luma_e / tot_e, "chroma_frac": chroma_e / tot_e,
                    "band": _band_energy(mag), "spectrum": _radial_spectrum(mag),
                    "grad_l2": float(np.sqrt(tot_e)),
                }
            _append_jsonl(summ_path, row_summ)
        _append_jsonl(SSD / "atlas_progress.jsonl",
                      {"pair": pi, "utc": _now(), "sec": round(time.time() - t0, 2),
                       "pose6": pose6.detach().float().cpu().numpy().round(4).tolist()})
        print(f"[atlas] pair {pi:3d} done in {time.time()-t0:.1f}s  pose6[0]={float(pose6.detach()[0]):.3f}")
    print(f"[atlas] complete; summaries -> {summ_path}; fields -> {fielddir}")


# --------------------------------------------------------- svd (deliverable 3: cross-pair basis)
def cmd_svd(args: argparse.Namespace) -> None:
    """Deliverable 3 — cross-pair shared-basis rank per dim (THE QA47/PH-1 pre-answer)."""
    fielddir = SSD / "fields"
    files = sorted(fielddir.glob("pair_*.npy"))
    pairs = [int(f.stem.split("_")[1]) for f in files]
    n = len(files)
    print(f"[svd] {n} pairs loaded")
    # load into (n, 6, D) — normalize each field to unit L2 (steering DIRECTION, magnitude removed)
    D = 2 * 3 * 384 * 512
    out = {"schema": "ddm_pi2_svd.v1", "utc": _now(), "n_pairs": n, "pairs": pairs, "dims": {}}
    # raw per-dim per-pair L2 norms (the MAGNITUDE finding: which dims the input actually moves)
    raw = np.zeros((n, 6), dtype=np.float64)
    for k, f in enumerate(files):
        gg = np.load(f).astype(np.float32).reshape(6, D)
        raw[k] = np.linalg.norm(gg, axis=1)
    out["per_dim_raw_Jnorm"] = {
        DIM_NAMES[i]: {"median": float(np.median(raw[:, i])), "max": float(raw[:, i].max()),
                       "min": float(raw[:, i].min())} for i in range(6)}
    # a dim is "signal" if its median raw norm is within 1e-3 of p0's (else input-near-null -> noise)
    p0med = float(np.median(raw[:, 0]))
    for i in range(6):
        M = np.zeros((n, D), dtype=np.float32)
        med = float(np.median(raw[:, i]))
        signal = med > 1e-3 * p0med
        for k, f in enumerate(files):
            g = np.load(f)[i].astype(np.float32).reshape(-1)
            nrm = float(np.linalg.norm(g))
            M[k] = g / nrm if nrm > 0 else 0.0
        M = np.nan_to_num(M)
        G = M @ M.T                              # (n,n) cosine Gram of unit directions
        G = np.nan_to_num(G)
        w = np.linalg.eigvalsh(G)[::-1]
        w = np.clip(w, 0, None)
        wsum = float(w.sum()) + 1e-30
        cum = np.cumsum(w) / wsum
        energy = (w / wsum).tolist()
        k80 = int(np.searchsorted(cum, 0.80) + 1)
        k90 = int(np.searchsorted(cum, 0.90) + 1)
        k95 = int(np.searchsorted(cum, 0.95) + 1)
        cum8 = float(cum[min(7, n - 1)])         # energy captured by a fixed k=8 basis (PH-1's claim)
        off = G[~np.eye(n, dtype=bool)]
        out["dims"][DIM_NAMES[i]] = {
            "signal": bool(signal), "median_raw_Jnorm": med,
            "energy_curve_top8": [round(x, 5) for x in energy[:8]],
            "cum_energy_at_k8": round(cum8, 4),
            "k80": k80, "k90": k90, "k95": k95,
            "top1_energy_frac": round(energy[0], 5),
            "mean_pairwise_cos": round(float(off.mean()), 5),
            "median_abs_pairwise_cos": round(float(np.median(np.abs(off))), 5),
        }
        tag = "" if signal else "  [INPUT-NEAR-NULL: rank is NOISE, not load-bearing]"
        print(f"[svd] {DIM_NAMES[i]}: |J|med={med:.4g} top1={energy[0]:.3f} k8cap={cum8:.2f} "
              f"k80={k80} k90={k90} k95={k95} mean_cos={float(off.mean()):.3f}{tag}")
    # cross-DIM relation: do the 6 per-dim Jacobian fields collapse onto one dominant direction
    # (the image-space analog of the warp-param rank-1 law, pfs1 p_star SVD dim0 96.98%)?
    # Use the per-pair-averaged unit fields' 6x6 Gram.
    meanfields = np.zeros((6, D), dtype=np.float32)
    for f in files:
        gg = np.load(f).astype(np.float32).reshape(6, D)
        for i in range(6):
            gg[i] /= (np.linalg.norm(gg[i]) + 1e-30)
        meanfields += gg
    meanfields /= n
    Gd = meanfields @ meanfields.T  # 6x6
    wd = np.linalg.eigvalsh(Gd)[::-1]
    out["cross_dim"] = {
        "gram_6x6": Gd.round(4).tolist(),
        "eig_energy": (wd / wd.sum()).round(5).tolist(),
        "top1_frac": round(float(wd[0] / wd.sum()), 5),
        "note": ("image-space cross-dim rank of the mean per-dim Jacobian directions; compare to "
                 "pfs1 warp-PARAM p_star SVD dim0=0.9698 (param space) — the honesty-rail relation."),
    }
    print(f"[svd] cross-dim top1 energy = {wd[0]/wd.sum():.3f} (param-space p_star dim0=0.970)")
    (SSD / "svd_receipt.json").write_text(json.dumps(out, indent=1))
    print(f"[svd] receipt -> {SSD/'svd_receipt.json'}")


class PoseForwardCamera:
    """camera-RGB pair (2,3,874,1164) float[0,255] -> pose[:6], differentiable, incl. shared resize.

    For faithful camera-res uint8 realization (the true contest lattice, frozen memo B7).
    """

    def __init__(self, device: str = "mps"):
        import tac.differentiable_eval_roundtrip as de
        from tac.torch_mps_compat import patch_scorer_for_mps
        self.device = device
        if device == "mps":
            patch_scorer_for_mps()
        self.pn = _load_posenet(device)
        self.yuv6 = de.differentiable_rgb_to_yuv6

    def __call__(self, cam_pair: torch.Tensor) -> torch.Tensor:
        rs = torch.nn.functional.interpolate(cam_pair, size=(384, 512), mode="bilinear")
        yuv = self.yuv6(rs)
        x = yuv.reshape(1, 12, 192, 256)
        return self.pn(x)["pose"][0, :6]


def _lowpass_field(field_chw: np.ndarray, keep_r: float = 0.12) -> np.ndarray:
    """Radial low-pass each channel (the 'low-freq far-field parametric' family proxy)."""
    out = np.zeros_like(field_chw)
    C, H, W = field_chw.shape
    yy, xx = np.ogrid[:H, :W]
    cy, cx = H // 2, W // 2
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    mask = (r <= keep_r).astype(np.float32)
    for c in range(C):
        F = np.fft.fftshift(np.fft.fft2(field_chw[c]))
        out[c] = np.real(np.fft.ifft2(np.fft.ifftshift(F * mask)))
    return out


def _chroma_only(field_chw: np.ndarray) -> np.ndarray:
    """Project each pixel's RGB grad onto ker(luma) (remove the luma component)."""
    g = field_chw.reshape(3, -1)
    luma = (LUMA_HAT[:, None] * g).sum(0, keepdims=True)  # (1,N)
    g2 = g - LUMA_HAT[:, None] * luma
    return g2.reshape(field_chw.shape)


def cmd_cheapest(args: argparse.Namespace) -> None:
    """Deliverable 4 — cheapest input per dim: steering atoms, family projections, realized B/unit."""
    d = np.load(GT_N96)
    dev = args.device
    fwd = PoseForwardCamera(dev)
    fwd_cpu = PoseForwardCamera("cpu")  # authority for realized Delta-pose
    # a few realization pairs spanning strata
    sel = _select_pairs(0)
    real_pairs = [sel[0], sel[len(sel) // 2], sel[-1], sel[1], sel[-2]][: args_nreal(args)]
    rows = []
    for pi in real_pairs:
        f0 = torch.from_numpy(d["gt_f0"][pi]).permute(2, 0, 1).float()
        f1 = torch.from_numpy(d["gt_f1"][pi]).permute(2, 0, 1).float()
        cam = torch.stack([f0, f1]).to(dev)
        leaf = cam.clone().requires_grad_(True)
        pose6 = fwd(leaf)
        base6_cpu = fwd_cpu(cam.detach().cpu()).detach().float().double().numpy()  # once per pair
        for i in range(6):
            J = torch.autograd.grad(pose6[i], leaf, retain_graph=(i < 5))[0]
            J = J.detach().float().cpu().numpy()          # (2,3,874,1164) camera-res Jacobian
            Jn = float(np.linalg.norm(J))
            # families (built on frame_0 support unless noted) -> energy fraction
            fam = {}
            J0 = J[0]  # frame_0
            J0e = float((J0 ** 2).sum()) + 1e-30
            fam["frame0_frac"] = float((J0 ** 2).sum() / (Jn ** 2 + 1e-30))
            lp = _lowpass_field(J[0])
            fam["lowfreq_frac"] = float((lp ** 2).sum() / J0e)  # of frame_0 energy
            ch = _chroma_only(J[0])
            fam["chroma_frac"] = float((ch ** 2).sum() / J0e)   # of frame_0 energy
            import zlib
            std_i = float(d["gt_poses"][:, i].std()) or 1.0
            target = 0.3 * std_i if i > 0 else 1.0        # dim0 is large; use unit target
            # input-near-null guard: if the frame_0 Jacobian is negligible vs p0, no cheap carrier
            if Jn < 1e-4:
                rows.append({"pair": pi, "dim": DIM_NAMES[i], "J_l2": Jn,
                             "min_l2_input_per_unit": 1.0 / (Jn + 1e-30), "families": fam,
                             "input_near_null": True,
                             "note": "frame_0 Jacobian negligible; dim barely responds to input"})
                print(f"[cheapest] pair {pi:3d} {DIM_NAMES[i]}: |J|={Jn:.3g}  INPUT-NEAR-NULL")
                continue

            def _realize(direction_c3hw: np.ndarray, alpha: float,
                         _cam=cam, _i=i, _base=base6_cpu) -> float:
                pert = _cam.detach().cpu().clone()
                pert[0] = pert[0] + alpha * torch.from_numpy(direction_c3hw).float()
                with torch.no_grad():
                    n6 = fwd_cpu(pert.round().clamp(0, 255).to("cpu")).float().double().numpy()
                return float(n6[_i] - _base[_i])

            # (A) minimal-L2 steerer = the full frame_0 Jacobian direction (closed form).
            #     tangent slope <J0,J0> ; realize -> secant. Linearity sweep in uint8 quanta.
            J0dot = float((J0 * J0).sum())
            a0 = target / (J0dot + 1e-30)
            sweep = []
            for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
                a = a0 * mult
                meas = _realize(J0, a)
                pred = a * J0dot
                sweep.append({"mult": mult, "alpha": a, "pred": pred, "meas": meas,
                              "gap": meas / (pred + 1e-30)})
            realized_gap_minL2 = sweep[2]["gap"]  # at mult=1

            # (B) cheap coded carrier = low-freq frame_0 atom; price + realize B/unit.
            atom = _lowpass_field(J[0])
            gdot = float((J0 * atom).sum())
            a_lf = target / (gdot + 1e-30)
            meas_lf = _realize(atom, a_lf)
            q = np.round(atom / (np.abs(atom).max() + 1e-30) * 127).astype(np.int8)
            coded_bytes = len(zlib.compress(q.tobytes(), 9))
            b_per_unit = coded_bytes / (abs(meas_lf) + 1e-30)
            rows.append({
                "pair": pi, "dim": DIM_NAMES[i], "J_l2": Jn,
                "min_l2_input_per_unit": 1.0 / (Jn + 1e-30),
                "families": fam,
                "linearity_sweep_minL2": sweep, "realized_gap_minL2_mult1": realized_gap_minL2,
                "lowfreq_carrier": {"alpha": a_lf, "meas_dpose_i": meas_lf,
                                    "coded_bytes": coded_bytes, "B_per_unit_dpose": b_per_unit},
            })
            print(f"[cheapest] pair {pi:3d} {DIM_NAMES[i]}: |J|={Jn:.3g} "
                  f"f0={fam['frame0_frac']:.2f} LF={fam['lowfreq_frac']:.2f} chroma={fam['chroma_frac']:.2f} "
                  f"gap@1={realized_gap_minL2:.2f} LF_B/unit={b_per_unit:.1f}")
    (SSD / "cheapest_receipt.json").write_text(json.dumps(
        {"schema": "ddm_pi2_cheapest.v1", "utc": _now(), "axis": "[macOS-CPU advisory]",
         "pointer": "0.1910828242 [contest-CPU] UNMOVED", "rows": rows}, indent=1))
    print(f"[cheapest] receipt -> {SSD/'cheapest_receipt.json'}")


def args_nreal(args: argparse.Namespace) -> int:
    return getattr(args, "nreal", 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("algebra")
    sp = sub.add_parser("smoke")
    sp.add_argument("--device", default="mps")
    sa = sub.add_parser("atlas")
    sa.add_argument("--device", default="mps")
    sa.add_argument("--n", type=int, default=0, help="0 = full stratified ~53 pairs")
    sub.add_parser("svd")
    sc = sub.add_parser("cheapest")
    sc.add_argument("--device", default="mps")
    sc.add_argument("--nreal", type=int, default=3)
    args = ap.parse_args()
    if args.cmd == "algebra":
        cmd_algebra(args)
    elif args.cmd == "smoke":
        cmd_smoke(args)
    elif args.cmd == "atlas":
        cmd_atlas(args)
    elif args.cmd == "svd":
        cmd_svd(args)
    elif args.cmd == "cheapest":
        cmd_cheapest(args)


if __name__ == "__main__":
    main()
