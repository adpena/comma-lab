#!/usr/bin/env python
"""$0 equal-L2 WARP-vs-NOISE flip-rate probe (Mallat GIS row 5) -- n600, GT frames.

[macOS-CPU advisory] NON-PROMOTABLE. Tests the Group-Invariant-Scattering prediction
(Mallat CPAM 2012: deformation-gradient term load-bearing; SegNet's stride-2 stem + deep
conv cascade as an approximate-scattering HEURISTIC, not a theorem) that smooth
diffeomorphic perturbations flip the frozen SegNet argmax near the separatrix far more
efficiently per unit input energy than additive iid noise. Informs the #268 S_R
reachability-weighting design (deformation-sensitivity vs additive/UNIWARD chart).
Pre-registration (written BEFORE measurement, incl. instrument-validity controls):
.omx/research/warp_vs_noise_flip_probe_20260707.md.

Channel: perturb CAMERA-RES GT frame1 (gt_n600.npz gt_f1, 874x1164x3 uint8; the frame
SegNet scores) in FLOAT -> round/clip -> uint8 -> contest scorer preprocess -> frozen CPU
SegNet -> argmax; flip = pixel argmax differs from the recomputed unperturbed baseline.
Conditions per pair (10 forwards): base; ego-screw ground-plane warps (xi = d*e_z via the
in-tree flat-ground IPM; d in {0.05, 0.15} m); generic smooth warps (Gaussian random
displacement field, sigma=48 px, RMS in {0.5, 1.5} px; seeded per pair); iid noise
L2-MATCHED per pair per warp condition; nz_x4 (4x wego_d15's L2) as the known-worse
dynamic-range control.

Instrument-validity gates (pre-registered): determinism floor (pair-0 double forward
bit-identical); baseline reproduces cached lstars + margins; monotonicity (d15>d05,
a15>a05, nz_x4 > nz_wego_d15); amplitude floor (>=10k pooled flips per condition).
AUTHORITY: one instrument only -- frozen CPU-torch SegNet (load_real_segnet + its
preprocess_input), the same path that produced the cached lstars/margins. NO MPS.

Chunked resumable foreground (atomic tmp+replace state); free-RAM >= 20 GiB gate;
VBATCH <= 4 pairs. Pointer 0.19110 UNMOVED (this is a means).
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage  # noqa: E402

torch.set_num_threads(4)

from tac.boundary_math.seg_core import load_real_segnet  # noqa: E402
from tac.boundary_math.lane_sdf_component import _CAM_H, _FY, _V_HORIZON  # noqa: E402

CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
OUT_DIR = REPO / "experiments/results/warp_vs_noise_flip_probe_20260707"
STATE = OUT_DIR / "probe_state.ckpt.npz"
OUT = OUT_DIR / "warp_vs_noise_flip_n600_20260707.json"

CAM_H_PX, CAM_W_PX = 874, 1164
_SCALE = CAM_H_PX / 384.0
FY_CAM = _FY * _SCALE          # in-tree fy scaled to camera rows
V_H_CAM = _V_HORIZON * _SCALE  # in-tree horizon row scaled to camera rows

EGO_DS = (0.05, 0.15)          # m (5%/15% of a ~1 m real inter-frame ego step)
SMOOTH_AMPS = (0.5, 1.5)       # px RMS
SMOOTH_SIGMA = 48.0            # px
WARP_CONDS = ["wego_d05", "wego_d15", "wsm_a05", "wsm_a15"]
CONDS = WARP_CONDS + [f"nz_{c}" for c in WARP_CONDS] + ["nz_x4"]
MARGIN_EDGES = np.array([0.0, 0.25, 0.5, 1.0, 2.0, 4.0, np.inf])
NBINS = len(MARGIN_EDGES) - 1
VBATCH = int(os.environ.get("WVN_VBATCH", "4"))
SAVE_EVERY = int(os.environ.get("WVN_SAVE_EVERY", "16"))
MIN_FREE_GIB = 20.0


def _free_gib() -> float:
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except Exception:
        return float("inf")


def _peak_rss_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # macOS: bytes


def ego_forward_warp(img_f: np.ndarray, d_m: float) -> np.ndarray:
    """Ground-plane diffeomorphism of the forward-translation twist xi = d*e_z.

    Inverse map per output pixel below the horizon: forward Z_o = cam_h*fy/(v - v_h);
    source row v_s = v_h + cam_h*fy/(Z_o + d); source col u_s - c_x = (u - c_x)*Z_o/(Z_o+d)
    (fx cancels). Identity above the horizon; displacement -> 0 continuously at the
    horizon (Z_o -> inf). Bilinear sampling per channel (order=1, mode='nearest')."""
    H, W = img_f.shape[:2]
    v = np.arange(H, dtype=np.float64)[:, None] * np.ones((1, W))
    u = np.ones((H, 1)) * np.arange(W, dtype=np.float64)[None, :]
    cx = W / 2.0
    below = v > (V_H_CAM + 1.0)
    zo = _CAM_H * FY_CAM / np.maximum(v - V_H_CAM, 1e-3)
    vs = V_H_CAM + _CAM_H * FY_CAM / (zo + d_m)
    us = cx + (u - cx) * zo / (zo + d_m)
    vs = np.where(below, vs, v)
    us = np.where(below, us, u)
    out = np.empty_like(img_f)
    for c in range(img_f.shape[2]):
        out[..., c] = ndimage.map_coordinates(
            img_f[..., c], [vs, us], order=1, mode="nearest")
    return out


def smooth_random_warp(img_f: np.ndarray, rms_px: float, seed: int) -> np.ndarray:
    """Generic smooth warp: Gaussian random displacement field (sigma=SMOOTH_SIGMA px),
    joint-RMS-normalized to rms_px; seeded (deterministic). Bilinear sampling."""
    H, W = img_f.shape[:2]
    rng = np.random.default_rng(seed)
    dx = ndimage.gaussian_filter(rng.standard_normal((H, W)), SMOOTH_SIGMA)
    dy = ndimage.gaussian_filter(rng.standard_normal((H, W)), SMOOTH_SIGMA)
    r = np.sqrt(np.mean(dx * dx + dy * dy))
    s = rms_px / max(r, 1e-12)
    dx, dy = dx * s, dy * s
    v = np.arange(H, dtype=np.float64)[:, None] + dy
    u = np.arange(W, dtype=np.float64)[None, :] + dx
    out = np.empty_like(img_f)
    for c in range(img_f.shape[2]):
        out[..., c] = ndimage.map_coordinates(
            img_f[..., c], [v, u], order=1, mode="nearest")
    return out


def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0.0, 255.0).astype(np.uint8)


def seg_forward_logits(seg, frames_cam_uint8: list[np.ndarray]) -> torch.Tensor:
    arr = np.stack([np.asarray(f)[None] for f in frames_cam_uint8], axis=0)
    xp = torch.from_numpy(arr).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        return seg(seg.preprocess_input(xp))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chunk-seconds", type=float, default=520.0,
                    help="exit cleanly (state saved) after this many seconds; re-invoke to resume.")
    ap.add_argument("--num-pairs", type=int, default=600, help="n600 discipline: keep 600.")
    args = ap.parse_args()

    free = _free_gib()
    if free < MIN_FREE_GIB:
        print(f"REFUSE: free RAM {free:.1f} GiB < {MIN_FREE_GIB} GiB "
              f"(live #205 run protection)", flush=True)
        sys.exit(3)

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seg = load_real_segnet("cpu")
    z = np.load(CACHE, allow_pickle=False)
    gt_f1 = z["gt_f1"]
    lst_all = z["lstars"]
    margins_cached = z["margins"]
    P = min(int(args.num_pairs), int(z["n_pairs"]))
    print(f"[{time.time() - t0:.1f}s] segnet + gt cache loaded; P={P} conds={CONDS}", flush=True)

    st: dict[str, np.ndarray] = {
        "bin_px": np.full((P, NBINS), np.nan),                 # baseline pixels per margin bin
        "base_lstar_mismatch": np.full(P, np.nan),             # validity: vs cached lstars
        "base_margin_maxdiff": np.full(P, np.nan),             # validity: vs cached margins
    }
    for c in CONDS:
        st[f"{c}__flips_bin"] = np.full((P, NBINS), np.nan)
        st[f"{c}__lane_flips"] = np.full(P, np.nan)
        st[f"{c}__l2_float"] = np.full(P, np.nan)
        st[f"{c}__l2_uint8"] = np.full(P, np.nan)
    done = np.zeros(P, bool)
    determinism = {"checked": False, "bit_identical": None}
    if STATE.exists():
        ck = np.load(STATE, allow_pickle=False)
        for k in st:
            if k in ck.files:
                st[k] = ck[k]
        done = ck["done"].astype(bool)
        if "determinism_ok" in ck.files:
            determinism = {"checked": True, "bit_identical": bool(ck["determinism_ok"])}
        print(f"[resume] {int(done.sum())}/{P} done", flush=True)

    def save():
        payload = dict(st)
        payload["done"] = done
        if determinism["checked"]:
            payload["determinism_ok"] = np.asarray(bool(determinism["bit_identical"]))
        tmp = STATE.with_suffix(".tmp.npz")
        np.savez(tmp, **payload)
        tmp.replace(STATE)

    todo = [i for i in range(P) if not done[i]]
    last = int(done.sum())
    for s0 in range(0, len(todo), VBATCH):
        chunk = todo[s0:s0 + VBATCH]
        # --- baseline forward (logits -> argmax + margin) ---
        base_frames = [np.asarray(gt_f1[pi]) for pi in chunk]
        logits = seg_forward_logits(seg, base_frames)
        top2 = torch.topk(logits, 2, dim=1).values
        margin_b = (top2[:, 0] - top2[:, 1]).cpu().numpy()
        base_am = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
        if not determinism["checked"]:
            # run-to-run determinism: re-forward the IDENTICAL batch (a batch-size change
            # legitimately alters reduction order bitwise -- that is not the floor we gate on)
            logits2 = seg_forward_logits(seg, base_frames)
            determinism = {"checked": True,
                           "bit_identical": bool(torch.equal(logits, logits2))}
            print(f"[validity] determinism bit_identical={determinism['bit_identical']}",
                  flush=True)
            del logits2
        del logits, top2

        pert: dict[str, list[np.ndarray]] = {c: [] for c in CONDS}
        for j, pi in enumerate(chunk):
            img_f = np.asarray(gt_f1[pi], np.float64)
            deltas: dict[str, np.ndarray] = {}
            deltas["wego_d05"] = ego_forward_warp(img_f, EGO_DS[0]) - img_f
            deltas["wego_d15"] = ego_forward_warp(img_f, EGO_DS[1]) - img_f
            deltas["wsm_a05"] = smooth_random_warp(img_f, SMOOTH_AMPS[0], 1000 + pi) - img_f
            deltas["wsm_a15"] = smooth_random_warp(img_f, SMOOTH_AMPS[1], 1000 + pi) - img_f
            for ci, c in enumerate(WARP_CONDS):
                d = deltas[c]
                l2 = float(np.sqrt(np.sum(d * d)))
                rng = np.random.default_rng(2000 + 10 * pi + ci)
                nz = rng.standard_normal(img_f.shape)
                nz *= l2 / max(float(np.sqrt(np.sum(nz * nz))), 1e-12)
                deltas[f"nz_{c}"] = nz
            rng = np.random.default_rng(2000 + 10 * pi + 9)
            nz4 = rng.standard_normal(img_f.shape)
            l2_d15 = float(np.sqrt(np.sum(deltas["wego_d15"] ** 2)))
            nz4 *= 4.0 * l2_d15 / max(float(np.sqrt(np.sum(nz4 * nz4))), 1e-12)
            deltas["nz_x4"] = nz4
            base_u8 = np.asarray(gt_f1[pi], np.int64)
            for c in CONDS:
                fu8 = to_uint8(img_f + deltas[c])
                pert[c].append(fu8)
                st[f"{c}__l2_float"][pi] = float(np.sqrt(np.sum(deltas[c] ** 2)))
                du = fu8.astype(np.int64) - base_u8
                st[f"{c}__l2_uint8"][pi] = float(np.sqrt(np.sum(du.astype(np.float64) ** 2)))
            # baseline validity + bin populations
            st["base_lstar_mismatch"][pi] = float(
                np.count_nonzero(base_am[j] != np.asarray(lst_all[pi], np.int64))
            ) / float(base_am[j].size)
            st["base_margin_maxdiff"][pi] = float(
                np.max(np.abs(margin_b[j] - np.asarray(margins_cached[pi], np.float64))))
            for bi in range(NBINS):
                sel = (margin_b[j] >= MARGIN_EDGES[bi]) & (margin_b[j] < MARGIN_EDGES[bi + 1])
                st["bin_px"][pi, bi] = float(sel.sum())

        for c in CONDS:
            am = seg_forward_logits(seg, pert[c]).argmax(dim=1).cpu().numpy().astype(np.int64)
            for j, pi in enumerate(chunk):
                flip = am[j] != base_am[j]
                lane_inv = flip & ((am[j] == 1) | (base_am[j] == 1))
                st[f"{c}__lane_flips"][pi] = float(lane_inv.sum())
                for bi in range(NBINS):
                    sel = (margin_b[j] >= MARGIN_EDGES[bi]) & (margin_b[j] < MARGIN_EDGES[bi + 1])
                    st[f"{c}__flips_bin"][pi, bi] = float((flip & sel).sum())
        for pi in chunk:
            done[pi] = True
        nd = int(done.sum())
        if nd - last >= SAVE_EVERY or nd == P:
            save()
            last = nd

            def pf(c):
                return float(np.nansum(st[f"{c}__flips_bin"]))

            print(f"[{time.time() - t0:.1f}s] {nd}/{P} | flips wego_d15={pf('wego_d15'):.0f} "
                  f"nz_wego_d15={pf('nz_wego_d15'):.0f} wsm_a15={pf('wsm_a15'):.0f} "
                  f"nz_wsm_a15={pf('nz_wsm_a15'):.0f} | rss={_peak_rss_gib():.1f}GiB", flush=True)
        if time.time() - t0 > float(args.chunk_seconds) and nd < P:
            save()
            print(f"[chunk-exit] {nd}/{P} done at {time.time() - t0:.1f}s; re-invoke to "
                  f"resume. peak_rss={_peak_rss_gib():.2f}GiB", flush=True)
            sys.exit(0)
    save()

    def pooled_flips(c, bins=None):
        a = st[f"{c}__flips_bin"]
        if bins is None:
            return float(np.nansum(a))
        return float(np.nansum(a[:, bins]))

    low_bins = [0, 1, 2]  # margin < 1 (pre-registered separatrix-adjacent population)
    ratios = {}
    for c in WARP_CONDS:
        wn = pooled_flips(c, low_bins)
        nn = pooled_flips(f"nz_{c}", low_bins)
        ratios[c] = (wn / nn) if nn > 0 else float("inf")

    # pre-registered validity gates
    gates = {
        "determinism_bit_identical": bool(determinism["bit_identical"]),
        "base_lstar_mismatch_mean": float(np.nanmean(st["base_lstar_mismatch"])),
        "base_margin_maxdiff_max": float(np.nanmax(st["base_margin_maxdiff"])),
        "monotone_ego": pooled_flips("wego_d15") > pooled_flips("wego_d05"),
        "monotone_smooth": pooled_flips("wsm_a15") > pooled_flips("wsm_a05"),
        "monotone_noise_x4": pooled_flips("nz_x4") > pooled_flips("nz_wego_d15"),
        "amplitude_floor_10k": {c: pooled_flips(c) >= 10_000 for c in CONDS},
    }
    energy_flag = {}
    for c in WARP_CONDS:
        eu_w = float(np.nansum(st[f"{c}__l2_uint8"] ** 2))
        eu_n = float(np.nansum(st[f"nz_{c}__l2_uint8"] ** 2))
        energy_flag[c] = {"post_uint8_energy_ratio_warp_over_noise":
                          (eu_w / eu_n) if eu_n > 0 else float("nan")}

    # verdict per pre-registration
    rvals = [ratios[c] for c in WARP_CONDS]
    if all(r >= 2.0 for r in rvals):
        verdict = "DEFORMATION-DOMINATES"
    elif all(r <= 1.2 for r in rvals):
        verdict = "NOISE-COMPARABLE"
    else:
        verdict = "MIXED"

    out = {
        "axis_tag": "[macOS-CPU advisory] NON-PROMOTABLE",
        "task": "equal-L2 warp-vs-noise flip-rate through frozen CPU SegNet on n600 GT "
                "frames (Mallat GIS row 5); informs #268 S_R reachability-weighting chart",
        "pre_registration": ".omx/research/warp_vs_noise_flip_probe_20260707.md",
        "authority": "frozen CPU-torch SegNet forward (load_real_segnet + preprocess_input), "
                     "the same instrument that produced the cached lstars/margins; no "
                     "trainer-verdict or probe-render numbers mixed in",
        "n_pairs": P,
        "conditions": CONDS,
        "ego_ds_m": list(EGO_DS), "smooth_rms_px": list(SMOOTH_AMPS),
        "smooth_sigma_px": SMOOTH_SIGMA,
        "margin_bin_edges": [float(x) for x in MARGIN_EDGES[:-1]] + ["inf"],
        "validity_gates": gates,
        "post_uint8_energy": energy_flag,
        "pooled_flips_total": {c: pooled_flips(c) for c in CONDS},
        "pooled_flips_margin_lt1": {c: pooled_flips(c, low_bins) for c in CONDS},
        "pooled_flips_by_bin": {c: [pooled_flips(c, [bi]) for bi in range(NBINS)]
                                for c in CONDS},
        "bin_px_total": [float(np.nansum(st["bin_px"][:, bi])) for bi in range(NBINS)],
        "lane_involved_flips": {c: float(np.nansum(st[f"{c}__lane_flips"])) for c in CONDS},
        "flip_ratio_warp_over_matched_noise_margin_lt1": ratios,
        "verdict_pre_registered": verdict,
        "l2_float_mean": {c: float(np.nanmean(st[f"{c}__l2_float"])) for c in CONDS},
        "l2_uint8_mean": {c: float(np.nanmean(st[f"{c}__l2_uint8"])) for c in CONDS},
        "peak_rss_gib": _peak_rss_gib(),
        "note": "means not ends: advisory row; pointer 0.19110 moves only via "
                "upstream/evaluate.py on exact archive bytes",
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"[{time.time() - t0:.1f}s] DONE -> {OUT}", flush=True)
    print(f"  verdict={verdict} ratios={ {c: round(r, 3) for c, r in ratios.items()} }",
          flush=True)


if __name__ == "__main__":
    main()
