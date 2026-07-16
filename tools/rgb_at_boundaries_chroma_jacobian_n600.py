# SPDX-License-Identifier: MIT
"""RGB-AT-BOUNDARIES derivation — per-class-pair chroma NECESSITY through the FROZEN scorer (n600, $0).

Operator 2026-07-15: "We might need RGB at boundaries regardless. Or some boundaries. Deep math and
geometry and frozen contest information space should reveal." + coordinator refinement: ground in the
EXACT upstream modules.py/evaluate.py forward, compose with R, check the PoseNet chroma path.

WHAT IT MEASURES (all through the real frozen CPU-torch scorers, fp32, NEVER MPS):

  Per pair (frame1), per class-pair boundary (a,b) of the SegNet argmax L*:
  A1 EXACT desat ablations (forward-only, batch of 3 variants in ONE SegNet forward):
     - desat_full:      rgb -> BT.601 Y replicated (all chroma removed)          [the #276 probe, per-pair, n600]
     - desat_annulus:   chroma removed ONLY inside the margin<1 annulus          [the lever's FAILURE mode]
     - keep_annulus:    chroma kept ONLY inside the annulus, desat elsewhere     [the lever's SUFFICIENCY test]
     Flips vs the cached exact L* are attributed to the NEAREST class-pair boundary.
  A2 chroma-Jacobian split (1 fwd + 1 bwd): g = d(sum of boundary margins)/d(SegNet input);
     per pixel the 3-vector g splits ORTHOGONALLY w.r.t. the BT.601 luma normal k = (0.299,0.587,0.114):
       S_luma = |g . k_hat|   (fastest-Y-change direction), S_chroma = ||g - (g.k_hat)k_hat||  (the Y-level-set plane)
     (upstream rgb_to_yuv6 frame_utils.py:60-63 defines EXACTLY this Y). Also the achromatic
     (1,1,1)/sqrt(3) split for robustness. Linearized distance-to-flip along each subspace = margin/S.
  B  Scene geometry: across-edge contrast delta rgb at the 384x512 grid, chroma/luma split per class-pair
     (which boundaries are chroma-DEFINED in the frozen video).
  C  PoseNet constraint (modules.py L73-74 rgb_to_yuv6 path): d_pose of (f0, f1_variant) vs the cached
     exact GT pose for desat_annulus and keep_annulus (does boundary-chroma carriage/failure move pose?).

R-COMPOSITION (--r-check): SegNet-grid chroma perturbations pushed through the REAL R
(bicubic up to camera -> round/clamp uint8 -> the real segnet.preprocess_input bilinear down):
amplitude transfer gain per LSB level. R is channel-diagonal with identical kernels for R,G,B,
so it COMMUTES with any fixed color-space split — the chroma/luma decomposition is preserved by
R exactly up to the uint8 quantization floor; the check measures that floor.

Resumable: per-frame JSONL rows; re-run skips finished pairs. Cache: the shared exact GT cache
(tools/build_shared_gt_cache_for_mlx_fleet.py output — the EXACT frozen-scorer outputs, no surrogate).

Usage:
  .venv/bin/python tools/rgb_at_boundaries_chroma_jacobian_n600.py --num-pairs 600 \
      --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz
  .venv/bin/python tools/rgb_at_boundaries_chroma_jacobian_n600.py --summarize   # aggregate rows -> summary.json
  .venv/bin/python tools/rgb_at_boundaries_chroma_jacobian_n600.py --r-check     # R survival gain check (8 frames)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OUT_DIR = REPO / "experiments" / "results" / "rgb_at_boundaries_chroma_jacobian_20260715"
ROWS = OUT_DIR / "rows.jsonl"
SUMMARY = OUT_DIR / "summary.json"

CLASS_NAMES = ["Road", "Lane", "Undrivable", "Movable", "MyCar"]
K601 = np.array([0.299, 0.587, 0.114], dtype=np.float64)  # upstream frame_utils.py:60
K601_HAT = K601 / np.linalg.norm(K601)
ACHRO_HAT = np.ones(3) / np.sqrt(3.0)
ANNULUS_BAND = 1.0  # margin<1 fragile annulus (#276/#333 convention)
NEAR_BAND_PX = 8  # near-boundary band for gradient attribution


def _bt601_desat(x_hwc: np.ndarray) -> np.ndarray:
    y = x_hwc[..., 0] * K601[0] + x_hwc[..., 1] * K601[1] + x_hwc[..., 2] * K601[2]
    return np.repeat(y[..., None], 3, axis=-1)


def _boundary_and_pair_maps(lstar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (boundary_mask HW bool, pair_id HW int, -1 off-boundary).

    A boundary pixel differs from >=1 of its 4 neighbors; its pair id is
    a*5+b (a<b) with b the most common differing neighbor class.
    """
    h, w = lstar.shape
    diff_votes = np.zeros((h, w, 5), dtype=np.int16)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nb = np.full_like(lstar, -1)
        ys = slice(max(dy, 0), h + min(dy, 0))
        xs = slice(max(dx, 0), w + min(dx, 0))
        ys_src = slice(max(-dy, 0), h + min(-dy, 0))
        xs_src = slice(max(-dx, 0), w + min(-dx, 0))
        nb[ys, xs] = lstar[ys_src, xs_src]
        m = (nb >= 0) & (nb != lstar)
        for c in range(5):
            diff_votes[..., c] += (m & (nb == c)).astype(np.int16)
    boundary = diff_votes.sum(axis=-1) > 0
    other = diff_votes.argmax(axis=-1)
    a = np.minimum(lstar, other)
    b = np.maximum(lstar, other)
    pair_id = np.where(boundary, a * 5 + b, -1).astype(np.int16)
    return boundary, pair_id


def _nearest_pair_map(boundary: np.ndarray, pair_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-boundary pair id for every pixel + distance (px)."""
    from scipy.ndimage import distance_transform_edt

    dist, (iy, ix) = distance_transform_edt(~boundary, return_indices=True)
    return pair_id[iy, ix], dist


def _seg_forward_batch(segnet, frames_hwc_f32: np.ndarray):
    """frames (B,H,W,3) float 0..255 camera-res -> (argmax (B,384,512), margin (B,384,512))."""
    import torch

    x = torch.from_numpy(frames_hwc_f32).float()  # (B,H,W,3)
    pair = torch.stack([x, x], dim=1).permute(0, 1, 4, 2, 3).contiguous()  # (B,2,3,H,W)
    with torch.inference_mode():
        seg_in = segnet.preprocess_input(pair)  # upstream modules.py:107-109
        logits = segnet(seg_in)
        top2 = torch.topk(logits, k=2, dim=1)
        margin = (top2.values[:, 0] - top2.values[:, 1]).clamp_min(0.0)
        arg = logits.argmax(dim=1)
    return arg.numpy().astype(np.int64), margin.numpy().astype(np.float32)


def _seg_grad(segnet, frame_hwc_f32: np.ndarray, lstar: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    """d(sum of boundary-pixel margins)/d(SegNet 384x512 input) -> (3,384,512) float64."""
    import torch

    x = torch.from_numpy(frame_hwc_f32[None]).float()
    pair = torch.stack([x, x], dim=1).permute(0, 1, 4, 2, 3).contiguous()
    with torch.no_grad():
        seg_in0 = segnet.preprocess_input(pair)
    seg_in = seg_in0.clone().detach().requires_grad_(True)
    logits = segnet(seg_in)  # (1,5,384,512)
    ls = torch.from_numpy(lstar[None])
    top1 = logits.gather(1, ls[:, None]).squeeze(1)  # logit at the exact-argmax class
    masked = logits.clone()
    masked.scatter_(1, ls[:, None], -1e30)
    top2 = masked.max(dim=1).values
    m = top1 - top2
    bsel = torch.from_numpy(boundary[None])
    loss = m[bsel].sum()
    loss.backward()
    return seg_in.grad[0].detach().numpy().astype(np.float64)  # (3,384,512)


def _grad_split(g_chw: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(S_luma601, S_chroma601, S_total) per pixel from gradient (3,H,W)."""
    g = np.moveaxis(g_chw, 0, -1)  # (H,W,3)
    lu = g @ K601_HAT
    ch = g - lu[..., None] * K601_HAT
    s_lu = np.abs(lu)
    s_ch = np.linalg.norm(ch, axis=-1)
    s_tot = np.linalg.norm(g, axis=-1)
    return s_lu, s_ch, s_tot


def _pose_raw(posenet, f0_u8: np.ndarray, f1_hwc_f32: np.ndarray) -> np.ndarray:
    import einops
    import torch

    pair = torch.from_numpy(np.stack([f0_u8.astype(np.float32), f1_hwc_f32.astype(np.float32)], axis=0)[None])
    x = einops.rearrange(pair, "b t h w c -> b t c h w").float()
    with torch.inference_mode():
        pose_in = posenet.preprocess_input(x)  # upstream modules.py:73-74 rgb_to_yuv6 path
        out = posenet(pose_in)
        pose = out["pose"] if isinstance(out, dict) else out
        half = None
        for hh in posenet.hydra.heads:
            if hh.name == "pose":
                half = hh.out // 2
        return pose[0, :half].detach().numpy().astype(np.float64)


def _load_cache(cache: Path, n: int):
    z = np.load(cache, allow_pickle=False)
    P = int(z["n_pairs"])
    if P < n:
        raise ValueError(f"cache has {P} < requested {n}")
    # materialize each member ONCE (NpzFile re-inflates per access)
    gt_f0 = z["gt_f0"][:n]
    gt_f1 = z["gt_f1"][:n]
    lstars = z["lstars"][:n]
    margins = z["margins"][:n]
    gt_poses = z["gt_poses"][:n]
    return gt_f0, gt_f1, lstars, margins, gt_poses


def run(n_pairs: int, cache: Path, threads: int, pose_every: int) -> None:
    import torch

    torch.set_num_threads(threads)
    from tac.boundary_math.seg_core import load_real_segnet

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done: set[int] = set()
    if ROWS.exists():
        for line in ROWS.open():
            try:
                done.add(int(json.loads(line)["pair_idx"]))
            except Exception:
                pass
    print(f"[rgb-boundaries] resume: {len(done)} rows already done", flush=True)

    segnet = load_real_segnet("cpu")
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # upstream

    dn = DistortionNet().eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    posenet = dn.posenet
    for p in posenet.parameters():
        p.requires_grad = False
    for p in segnet.parameters():
        p.requires_grad = False

    gt_f0, gt_f1, lstars, margins, gt_poses = _load_cache(cache, n_pairs)
    cam_h, cam_w = gt_f1.shape[1], gt_f1.shape[2]

    # startup NO-FAKE sanity: recompute frame 0 argmax, must match the cached exact L*
    if 0 not in done:
        arg0, _ = _seg_forward_batch(segnet, gt_f1[0:1].astype(np.float32))
        mismatch = int((arg0[0] != lstars[0]).sum())
        print(f"[rgb-boundaries] baseline sanity frame0: argmax mismatch px = {mismatch}", flush=True)
        if mismatch > 0:
            raise RuntimeError("baseline forward does not reproduce the cached exact L* — abort (NO-FAKE)")

    fout = ROWS.open("a")
    t_start = time.time()
    for i in range(n_pairs):
        if i in done:
            continue
        t0 = time.time()
        f1 = gt_f1[i].astype(np.float64)
        lstar = lstars[i]
        margin = margins[i]
        boundary, pair_id = _boundary_and_pair_maps(lstar)
        near_pair, near_dist = _nearest_pair_map(boundary, pair_id)
        annulus = margin < ANNULUS_BAND  # (384,512)

        # camera-res annulus mask (nearest upsample; scale factors are non-integer -> index map)
        yy = np.clip((np.arange(cam_h) * lstar.shape[0] / cam_h).astype(np.int64), 0, lstar.shape[0] - 1)
        xx = np.clip((np.arange(cam_w) * lstar.shape[1] / cam_w).astype(np.int64), 0, lstar.shape[1] - 1)
        ann_cam = annulus[np.ix_(yy, xx)]  # (H,W) bool

        desat = _bt601_desat(f1)
        v_full = desat
        v_ann = np.where(ann_cam[..., None], desat, f1)
        v_keep = np.where(ann_cam[..., None], f1, desat)
        args, margs = _seg_forward_batch(segnet, np.stack([v_full, v_ann, v_keep]).astype(np.float32))

        # gradient leg
        g = _seg_grad(segnet, f1, lstar, boundary)
        s_lu, s_ch, s_tot = _grad_split(g)
        g_hwc = np.moveaxis(g, 0, -1)
        s_ach = np.abs(g_hwc @ ACHRO_HAT)  # achromatic (1,1,1)/sqrt3 split (robustness column)

        # scene across-edge contrast at the SegNet grid
        import torch as _t

        with _t.inference_mode():
            xseg = (
                _t.nn.functional.interpolate(
                    _t.from_numpy(f1[None]).permute(0, 3, 1, 2).float(),
                    size=lstar.shape, mode="bilinear",
                )[0].permute(1, 2, 0).numpy().astype(np.float64)
            )

        row: dict = {"pair_idx": i, "pairs": {}}
        # global d_seg-equivalent worth of each ablation (argmax disagreement vs exact L*)
        npix = float(lstar.size)
        row["d_seg_equiv"] = {
            "desat_full": float((args[0] != lstar).sum() / npix),
            "desat_annulus": float((args[1] != lstar).sum() / npix),
            "keep_annulus": float((args[2] != lstar).sum() / npix),
        }
        row["annulus_frac"] = float(annulus.mean())

        for pid in np.unique(pair_id[pair_id >= 0]):
            a, b = int(pid) // 5, int(pid) % 5
            bsel = pair_id == pid
            nb = int(bsel.sum())
            if nb < 20:
                continue
            near_sel = (near_pair == pid) & (near_dist <= NEAR_BAND_PX)
            d: dict = {"boundary_px": nb, "near_px": int(near_sel.sum())}
            for vi, vname in enumerate(("desat_full", "desat_annulus", "keep_annulus")):
                flips = args[vi] != lstar
                d[f"flip_at_boundary_{vname}"] = float(flips[bsel].mean())
                d[f"flips_near_{vname}"] = int((flips & near_sel).sum())
            # gradient split over the near-boundary band (energy-weighted)
            e_tot = float((s_tot[near_sel] ** 2).sum())
            d["grad_e_tot"] = e_tot
            d["grad_e_chroma601"] = float((s_ch[near_sel] ** 2).sum())
            d["grad_e_luma601"] = float((s_lu[near_sel] ** 2).sum())
            d["grad_e_achro"] = float((s_ach[near_sel] ** 2).sum())
            # distance-to-flip at the boundary pixels themselves (linearized, 0..255 units)
            mb = margin[bsel].astype(np.float64)
            sc = s_ch[bsel]
            sl = s_lu[bsel]
            eps = 1e-12
            dch = mb / (sc + eps)
            dlu = mb / (sl + eps)
            d["frac_chroma_stronger"] = float((sc > sl).mean())
            d["dtf_chroma_med"] = float(np.median(dch))
            d["dtf_luma_med"] = float(np.median(dlu))
            d["frac_dtf_chroma_le8"] = float((dch <= 8.0).mean())
            d["frac_dtf_chroma_le2"] = float((dch <= 2.0).mean())
            d["frac_dtf_luma_le8"] = float((dlu <= 8.0).mean())
            d["mean_margin_boundary"] = float(mb.mean())
            d["frac_annulus_boundary"] = float(annulus[bsel].mean())
            # scene across-edge contrast (sample p and its differing 4-neighbors via shifted diffs)
            dsum_ch, dsum_lu, ndiff = 0.0, 0.0, 0
            for dy, dx in ((1, 0), (0, 1)):
                pa = lstar[: lstar.shape[0] - dy, : lstar.shape[1] - dx]
                pb = lstar[dy:, dx:]
                m2 = (pa != pb) & ((np.minimum(pa, pb) == a) & (np.maximum(pa, pb) == b))
                if not m2.any():
                    continue
                dv = xseg[: lstar.shape[0] - dy, : lstar.shape[1] - dx] - xseg[dy:, dx:]
                dvm = np.ascontiguousarray(dv[m2], dtype=np.float64)
                # manual dot (Accelerate matmul on masked tiny arrays emits spurious FP-flag warnings)
                lu = dvm[:, 0] * K601_HAT[0] + dvm[:, 1] * K601_HAT[1] + dvm[:, 2] * K601_HAT[2]
                ch = dvm - lu[:, None] * K601_HAT
                dsum_ch += float(np.linalg.norm(ch, axis=1).sum())
                dsum_lu += float(np.abs(lu).sum())
                ndiff += int(m2.sum())
            if ndiff:
                d["edge_chroma_contrast"] = dsum_ch / ndiff
                d["edge_luma_contrast"] = dsum_lu / ndiff
                d["edge_n"] = ndiff
            row["pairs"][f"{a}-{b}"] = d

        # PoseNet constraint (modules.py rgb_to_yuv6 path): every pose_every-th frame
        if pose_every and (i % pose_every == 0):
            gp = gt_poses[i][: 6]
            p_ann = _pose_raw(posenet, gt_f0[i], v_ann)[:6]
            p_keep = _pose_raw(posenet, gt_f0[i], v_keep)[:6]
            row["d_pose_desat_annulus"] = float(np.mean((p_ann - gp) ** 2))
            row["d_pose_keep_annulus"] = float(np.mean((p_keep - gp) ** 2))

        fout.write(json.dumps(row) + "\n")
        fout.flush()
        if i % 10 == 0:
            el = time.time() - t_start
            print(f"[rgb-boundaries] pair {i}/{n_pairs} ({time.time()-t0:.1f}s/frame, {el:.0f}s elapsed)", flush=True)
    fout.close()
    print("[rgb-boundaries] DONE", flush=True)


def summarize() -> dict:
    rows = [json.loads(line) for line in ROWS.open()]
    rows = {r["pair_idx"]: r for r in rows}
    rows = [rows[k] for k in sorted(rows)]
    agg: dict = {"n_pairs": len(rows), "per_pair": {}, "global": {}}
    for key in ("desat_full", "desat_annulus", "keep_annulus"):
        agg["global"][f"d_seg_equiv_{key}"] = float(np.mean([r["d_seg_equiv"][key] for r in rows]))
    agg["global"]["annulus_frac"] = float(np.mean([r["annulus_frac"] for r in rows]))
    dp_ann = [r["d_pose_desat_annulus"] for r in rows if "d_pose_desat_annulus" in r]
    dp_keep = [r["d_pose_keep_annulus"] for r in rows if "d_pose_keep_annulus" in r]
    if dp_ann:
        agg["global"]["d_pose_desat_annulus_mean"] = float(np.mean(dp_ann))
        agg["global"]["d_pose_keep_annulus_mean"] = float(np.mean(dp_keep))
        agg["global"]["d_pose_n"] = len(dp_ann)

    pair_keys = sorted({k for r in rows for k in r["pairs"]})
    for pk in pair_keys:
        rs = [r["pairs"][pk] for r in rows if pk in r["pairs"]]
        tot_b = float(sum(x["boundary_px"] for x in rs))
        w = np.array([x["boundary_px"] for x in rs], dtype=np.float64)
        w /= w.sum()

        def wmean(field, rs=rs, w=w):
            vals = np.array([x.get(field, np.nan) for x in rs], dtype=np.float64)
            m = ~np.isnan(vals)
            return float((vals[m] * (w[m] / max(w[m].sum(), 1e-12))).sum()) if m.any() else None

        e_tot = sum(x["grad_e_tot"] for x in rs)
        a, b = (int(c) for c in pk.split("-"))
        agg["per_pair"][pk] = {
            "names": f"{CLASS_NAMES[a]}|{CLASS_NAMES[b]}",
            "frames_present": len(rs),
            "total_boundary_px": int(tot_b),
            "flip_at_boundary_desat_full": wmean("flip_at_boundary_desat_full"),
            "flip_at_boundary_desat_annulus": wmean("flip_at_boundary_desat_annulus"),
            "flip_at_boundary_keep_annulus": wmean("flip_at_boundary_keep_annulus"),
            "flips_near_desat_full_total": int(sum(x["flips_near_desat_full"] for x in rs)),
            "flips_near_desat_annulus_total": int(sum(x["flips_near_desat_annulus"] for x in rs)),
            "flips_near_keep_annulus_total": int(sum(x["flips_near_keep_annulus"] for x in rs)),
            "grad_chroma601_energy_frac": float(sum(x["grad_e_chroma601"] for x in rs) / max(e_tot, 1e-12)),
            "grad_achro_energy_frac": float(sum(x["grad_e_achro"] for x in rs) / max(e_tot, 1e-12)),
            "frac_chroma_stronger": wmean("frac_chroma_stronger"),
            "dtf_chroma_med": wmean("dtf_chroma_med"),
            "dtf_luma_med": wmean("dtf_luma_med"),
            "frac_dtf_chroma_le8": wmean("frac_dtf_chroma_le8"),
            "frac_dtf_chroma_le2": wmean("frac_dtf_chroma_le2"),
            "frac_dtf_luma_le8": wmean("frac_dtf_luma_le8"),
            "mean_margin_boundary": wmean("mean_margin_boundary"),
            "frac_annulus_boundary": wmean("frac_annulus_boundary"),
            "edge_chroma_contrast": wmean("edge_chroma_contrast"),
            "edge_luma_contrast": wmean("edge_luma_contrast"),
        }
    SUMMARY.write_text(json.dumps(agg, indent=1))
    print(json.dumps(agg["global"], indent=1))
    return agg


def r_check(cache: Path, n_frames: int = 8) -> None:
    """Chroma perturbation SURVIVAL through the real R (bicubic up -> uint8 -> real preprocess bilinear down)."""
    import torch

    from tac.boundary_math.seg_core import load_real_segnet

    segnet = load_real_segnet("cpu")
    gt_f0, gt_f1, lstars, margins, _ = _load_cache(cache, 600)
    idxs = np.linspace(0, 599, n_frames).astype(int)
    cam_h, cam_w = gt_f1.shape[1], gt_f1.shape[2]
    out = []
    # a unit chroma direction (in the k.delta=0 plane): blue-yellow-ish, Y-neutral
    d1 = np.array([1.0, 0.0, 0.0]) - K601_HAT * (np.array([1.0, 0.0, 0.0]) @ K601_HAT)
    d1 /= np.linalg.norm(d1)
    for i in idxs:
        lstar = lstars[i]
        margin = margins[i]
        annulus = (margin < ANNULUS_BAND).astype(np.float64)
        with torch.inference_mode():
            xseg = (
                torch.nn.functional.interpolate(
                    torch.from_numpy(gt_f1[i].astype(np.float64)[None]).permute(0, 3, 1, 2).float(),
                    size=lstar.shape, mode="bilinear",
                )[0].permute(1, 2, 0).numpy().astype(np.float64)
            )
        for lsb in (0.5, 1.0, 2.0, 4.0):
            delta = annulus[..., None] * d1[None, None, :] * lsb
            gains = []
            for sign in (+1.0, -1.0):
                xp = np.clip(xseg + sign * delta, 0, 255)
                # real R: bicubic up to camera -> uint8 -> real preprocess (bilinear down)
                with torch.inference_mode():
                    def _through(v):
                        up = torch.nn.functional.interpolate(
                            torch.from_numpy(v[None]).permute(0, 3, 1, 2).float(),
                            size=(cam_h, cam_w), mode="bicubic", align_corners=False,
                        )
                        up = torch.clamp(torch.round(up), 0.0, 255.0)
                        pair = torch.stack([up, up], dim=1)  # (1,2,3,H,W)
                        return segnet.preprocess_input(pair)[0].permute(1, 2, 0).numpy().astype(np.float64)

                    y0 = _through(xseg)
                    y1 = _through(xp)
                dd = y1 - y0
                lu = dd @ K601_HAT
                ch = dd - lu[..., None] * K601_HAT
                ann_sel = annulus > 0.5
                achieved = float(np.linalg.norm(ch[ann_sel], axis=-1).mean())
                gains.append(achieved / lsb)
            out.append({"frame": int(i), "lsb": lsb, "chroma_gain_through_R": float(np.mean(gains))})
    (OUT_DIR / "r_survival_check.json").write_text(json.dumps(out, indent=1))
    for r in out:
        print(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--gt-cache", type=Path, default=REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--pose-every", type=int, default=1, help="PoseNet variant check every N frames (0=off)")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--r-check", action="store_true")
    args = ap.parse_args()
    if args.summarize:
        summarize()
        return
    if args.r_check:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        r_check(args.gt_cache)
        return
    run(args.num_pairs, args.gt_cache, args.threads, args.pose_every)


if __name__ == "__main__":
    main()
