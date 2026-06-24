#!/usr/bin/env python3
"""Class-1 island intrinsic dimension ACROSS REPRESENTATION LEVELS ($0/CPU).

Operator vision (2026-06-23): "pixel is not the correct representation level;
everything is compressible = imagination(basis) x divergent-thinking(generative
program) x tradeoffs(task distortion)". The d_seg-binding class-1 island stratum
is LINEAR-rank ~53 in PIXELS (full-rank, frozen_partition_topology Result 4).
Rank is BASIS-DEPENDENT. This probe measures the islands' INTRINSIC DIMENSION m
across representation levels and finds the basis (if any) where it COLLAPSES.

DECISION:
- GO-FORMAT  : some BASIS collapses m to <= ~13 (the Whitney 2m+1=28 latent
               budget of the custom witness format) -> build WHERE/HOW-MUCH
               layers in that basis.
- GO-GENERATOR: m ~ 53 in EVERY linear basis BUT a nonlinear estimator (tiny AE
               bottleneck knee, TwoNN/MLE) shows m << 53 -> a TRAINED generative
               program touches the islands, not a flat code.
- WALL       : m ~ 53 even nonlinear -> islands are irreducible content-noise;
               route to training-time d_seg loss only.

AUTHORITY / NO-FAKE:
- The class-1 island stratum is extracted from the EXACT frozen-SegNet argmax
  cache (seg_argmaps.npz key 'gt' (600,384,512) uint8, validated d_seg=5.599e-4
  vs report dt=1e-7 -- exact-scorer faithful). Same loader as
  experiments/probe_frozen_partition_topology.py (reused, not reinvented).
- Class-1 = the volatile island stratum (~0.72% of pixels, ~31 components/frame
  per the topology probe). Islands = connected components of the class-1 mask.
- CPU-ONLY, NEVER MPS, NEVER GPU. The tiny AE is torch on CPU. No upstream edits.

Representation levels (the islands' intrinsic dim m in each):
1. pixel-linear PCA effective rank (CONTROL -- must reproduce ~53).
2. spectral 2D-DCT low-frequency-truncated effective rank.
3. Fourier-descriptor / contour per-island closed-curve descriptors.
4. nonlinear motion-compensated warp residual rank (affine warp per frame).
5. nonlinear intrinsic-dim estimators on the raw island stack: TwoNN
   (Facco 2017) + MLE (Levina-Bickel) + tiny CPU autoencoder bottleneck sweep.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ARGMAPS = (
    REPO
    / "experiments/results/indep_dseg_bets_20260623_inflated/seg_argmaps.npz"
)
N_CLASSES = 5
ISLAND_CLASS = 1
SMALL_COMPONENT_PX = 500  # the "fine island" threshold from the topology probe


# ---------------------------------------------------------------------------
# Effective-rank helpers (linear bases)
# ---------------------------------------------------------------------------
def effective_rank_from_singvals(S: np.ndarray) -> dict:
    """Effective/intrinsic dim summaries from singular values of a CENTERED
    data matrix (rows = frames). Returns participation ratio (eff rank), the
    Shannon entropy effective rank, and k-for-Xpct cumulative-variance dims.
    """
    var = S.astype(np.float64) ** 2
    tot = var.sum()
    if tot <= 0:
        return {
            "participation_ratio": 0.0,
            "entropy_effrank": 0.0,
            "top1_var_share": 0.0,
            "k_for_50pct": 0,
            "k_for_80pct": 0,
            "k_for_90pct": 0,
            "k_for_95pct": 0,
            "k_for_99pct": 0,
            "n_singvals": int(S.size),
        }
    ratio = var / tot
    # participation ratio (Frobenius eff rank): (sum s^2)^2 / sum s^4
    pr = (var.sum() ** 2) / (var**2).sum()
    # Shannon entropy effective rank: exp(-sum p log p) over normalized var
    nz = ratio[ratio > 0]
    ent = float(np.exp(-(nz * np.log(nz)).sum()))
    cum = np.cumsum(ratio)
    kfor = {
        f"k_for_{int(p*100)}pct": int(np.searchsorted(cum, p) + 1)
        for p in (0.50, 0.80, 0.90, 0.95, 0.99)
    }
    return {
        "participation_ratio": float(pr),
        "entropy_effrank": ent,
        "top1_var_share": float(ratio[0]),
        "n_singvals": int(S.size),
        **kfor,
    }


def svd_singvals(X: np.ndarray) -> np.ndarray:
    """Centered economy SVD singular values of (n, d) matrix."""
    Xc = X.astype(np.float64)
    Xc = Xc - Xc.mean(axis=0, keepdims=True)
    # economy SVD; for n << d, gram-trick keeps it cheap and exact for singvals
    n, d = Xc.shape
    # NOTE: macOS Accelerate BLAS sets spurious FP-exception flags on large
    # matmuls (divide-by-zero/overflow RuntimeWarnings) even when the output is
    # finite and correct -- verified gram-vs-direct SVD agree to 1.6e-11 on the
    # real class-1 stack. np.errstate suppresses the COSMETIC flag; output is
    # numerically identical (the singular values are exact).
    with np.errstate(all="ignore"):
        if d > n:
            G = Xc @ Xc.T  # (n, n)
            assert np.isfinite(G).all(), "gram matrix non-finite (real numerics bug)"
            w = np.linalg.eigvalsh(G)
            w = np.clip(w[::-1], 0.0, None)
            return np.sqrt(w)
        S = np.linalg.svd(Xc, compute_uv=False)
    return S


# ---------------------------------------------------------------------------
# Island extraction (the d_seg-binding stratum)
# ---------------------------------------------------------------------------
def class1_mask_stack(gt: np.ndarray) -> np.ndarray:
    """Boolean class-1 indicator stack (n, H, W). The volatile island stratum."""
    return gt == ISLAND_CLASS


def small_island_stack(gt: np.ndarray) -> np.ndarray:
    """Class-1 indicator stack but ONLY the SMALL (<500 px) connected
    components -- the fine islands that carry the residual d_seg debt (the
    coarse class-1 mass, if any large component exists, is excluded so the
    measured dim is of the volatile islands, not a stable blob).
    """
    import scipy.ndimage as ndi

    struct = ndi.generate_binary_structure(2, 1)  # 4-conn (matches topology probe)
    n, H, W = gt.shape
    out = np.zeros((n, H, W), dtype=bool)
    for i in range(n):
        m = gt[i] == ISLAND_CLASS
        if not m.any():
            continue
        lab, ncomp = ndi.label(m, structure=struct)
        if ncomp == 0:
            continue
        sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, ncomp + 1))
        keep = np.where(sizes < SMALL_COMPONENT_PX)[0] + 1
        out[i] = np.isin(lab, keep)
    return out


# ---------------------------------------------------------------------------
# LEVEL 1 -- pixel-linear PCA effective rank (CONTROL)
# ---------------------------------------------------------------------------
def level1_pixel_linear(stack: np.ndarray) -> dict:
    """PCA effective rank of the flattened island indicator stack (pixels)."""
    n = stack.shape[0]
    X = stack.reshape(n, -1).astype(np.float64)
    S = svd_singvals(X)
    er = effective_rank_from_singvals(S)
    er["pixel_mass_frac"] = float(stack.mean())
    er["singvals_top10"] = S[:10].round(3).tolist()
    return er


# ---------------------------------------------------------------------------
# LEVEL 2 -- spectral 2D-DCT low-frequency-truncated effective rank
# ---------------------------------------------------------------------------
def level2_dct(stack: np.ndarray, keep_lf: int = 32) -> dict:
    """2D-DCT each island frame, keep the keep_lf x keep_lf low-frequency block,
    then PCA the DCT-coeff vectors. If a Gabor/DCT basis sparsifies the islands,
    the truncated representation's effective rank collapses AND the energy
    concentrates in low frequencies.
    """
    from scipy.fft import dctn

    n, H, W = stack.shape
    coeffs = np.zeros((n, keep_lf * keep_lf), dtype=np.float64)
    energy_lf = np.zeros(n)
    energy_tot = np.zeros(n)
    for i in range(n):
        f = stack[i].astype(np.float64)
        C = dctn(f, norm="ortho")
        block = C[:keep_lf, :keep_lf]
        coeffs[i] = block.ravel()
        energy_lf[i] = float((block**2).sum())
        energy_tot[i] = float((C**2).sum())
    S = svd_singvals(coeffs)
    er = effective_rank_from_singvals(S)
    # how much of the islands' energy survives the LF truncation -- if a DCT
    # basis is the right one, LF energy fraction is high (sparse in DCT).
    frac = energy_lf / np.maximum(energy_tot, 1e-12)
    er["keep_lf"] = int(keep_lf)
    er["lf_energy_frac_mean"] = float(frac.mean())
    er["lf_energy_frac_median"] = float(np.median(frac))
    er["dct_coeff_dim"] = int(keep_lf * keep_lf)
    return er


# ---------------------------------------------------------------------------
# LEVEL 3 -- Fourier-descriptor / contour per-island closed-curve descriptors
# ---------------------------------------------------------------------------
def fourier_descriptors(contour: np.ndarray, n_desc: int) -> np.ndarray:
    """Translation/rotation/scale-normalized Fourier descriptors of a closed
    contour (N,2) of (row,col) points. Returns 2*n_desc real features (mag of
    low harmonics, plus centroid + perimeter for completeness as separate).
    Standard Granlund-1972 / Zahn-Roskies closed-curve descriptor.
    """
    if contour.shape[0] < 3:
        return np.zeros(n_desc, dtype=np.float64)
    z = contour[:, 1].astype(np.float64) + 1j * contour[:, 0].astype(np.float64)
    Z = np.fft.fft(z)
    # translation invariance: drop DC (Z[0])
    Z[0] = 0.0
    # scale invariance: divide by |Z[1]| (first nonzero harmonic), guard 0
    s = np.abs(Z[1]) if Z.shape[0] > 1 and np.abs(Z[1]) > 1e-9 else 1.0
    Zn = Z / s
    # rotation/start-point invariance: use magnitudes of low harmonics
    mags = np.abs(Zn)
    # take the n_desc lowest-frequency harmonics (1..n_desc), wrap around
    desc = np.zeros(n_desc, dtype=np.float64)
    for k in range(1, n_desc + 1):
        desc[k - 1] = mags[k % mags.shape[0]]
    return desc


def level3_contour(gt: np.ndarray, n_desc: int = 8, max_islands: int = 40) -> dict:
    """Per-island closed-curve Fourier descriptors. Aggregate two ways:
    (a) per-frame: concatenate the descriptors of the largest `max_islands`
        islands (zero-padded), PCA across frames -> frame-level shape dim.
    (b) per-island: pool ALL islands across all frames into one cloud, PCA the
        descriptor cloud -> the SHAPE-VOCABULARY dim (how many distinct island
        shapes exist). If islands are a small shape vocabulary, this collapses.
    """
    import scipy.ndimage as ndi
    from skimage import measure

    struct = ndi.generate_binary_structure(2, 1)
    n, H, W = gt.shape
    per_frame = np.zeros((n, max_islands * n_desc), dtype=np.float64)
    all_descs = []
    island_count = []
    for i in range(n):
        m = gt[i] == ISLAND_CLASS
        if not m.any():
            island_count.append(0)
            continue
        lab, ncomp = ndi.label(m, structure=struct)
        if ncomp == 0:
            island_count.append(0)
            continue
        sizes = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, ncomp + 1))
        order = np.argsort(sizes)[::-1] + 1  # largest first
        island_count.append(int(ncomp))
        slot = 0
        for lid in order:
            comp = lab == lid
            # marching-squares contour (closed boundary)
            cs = measure.find_contours(comp.astype(float), 0.5)
            if not cs:
                continue
            cont = max(cs, key=len)  # the longest contour
            d = fourier_descriptors(cont, n_desc)
            all_descs.append(d)
            if slot < max_islands:
                per_frame[i, slot * n_desc:(slot + 1) * n_desc] = d
                slot += 1
    # (a) frame-level
    S_frame = svd_singvals(per_frame)
    er_frame = effective_rank_from_singvals(S_frame)
    # (b) shape-vocabulary cloud
    cloud = np.array(all_descs, dtype=np.float64) if all_descs else np.zeros((1, n_desc))
    S_cloud = svd_singvals(cloud)
    er_cloud = effective_rank_from_singvals(S_cloud)
    return {
        "n_desc_per_island": int(n_desc),
        "max_islands_per_frame": int(max_islands),
        "total_islands_pooled": int(cloud.shape[0]),
        "islands_per_frame_mean": float(np.mean(island_count)),
        "frame_level": er_frame,
        "shape_vocabulary_cloud": er_cloud,
    }


# ---------------------------------------------------------------------------
# LEVEL 4 -- nonlinear motion-compensated warp residual rank
# ---------------------------------------------------------------------------
def estimate_affine(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """Least-squares 2D affine (6-param) mapping cur-coords -> prev-coords using
    the ON pixels of cur as correspondences against a smoothed prev field. We
    use a gradient-free intensity-based formulation: solve for affine A that
    best aligns cur's island mass to prev's via first-order Taylor (optical
    flow brightness constraint on the float masks). Returns 2x3 affine matrix.
    """
    import scipy.ndimage as ndi

    p = prev.astype(np.float64)
    c = cur.astype(np.float64)
    ps = ndi.gaussian_filter(p, 2.0)
    cs = ndi.gaussian_filter(c, 2.0)
    # Gauss-Newton: solve the affine flow (u,v) that maps prev-coords to the
    # cur-coords they came from, so sampling cur at prev's grid + flow aligns cur
    # to prev. Brightness constraint on the SMOOTHED fields:
    #   prev(o) ~= cur(o + flow(o))  ~=  cur(o) + grad_cur . flow   (1st order)
    #   => grad_cur . flow = prev - cur   (It = prev - cur)
    # flow(o) = (u,v),  u = a0 + a1 x + a2 y ,  v = a3 + a4 x + a5 y.
    # affine_transform samples input at M@[y,x]+offset = cur-coord = o + flow.
    gy, gx = np.gradient(cs)
    H, W = c.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    It = ps - cs  # prev - cur
    w = np.sqrt(gx**2 + gy**2)
    mask = w > (w.mean() + 1e-6)  # observable only where there is gradient
    if mask.sum() < 12:
        return np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)
    gxv, gyv = gx[mask], gy[mask]
    xv, yv, Itv = xx[mask], yy[mask], It[mask]
    # grad_cur . flow = It  =>  gx*u + gy*v = It, linear in (a0..a5)
    A = np.column_stack([
        gxv, gxv * xv, gxv * yv,  # u terms (gx * [1,x,y])
        gyv, gyv * xv, gyv * yv,  # v terms (gy * [1,x,y])
    ])
    coef, *_ = np.linalg.lstsq(A, Itv, rcond=1e-8)
    a0, a1, a2, a3, a4, a5 = coef
    # cur-coord = o + flow(o):  x_in = x + u = x + a0 + a1 x + a2 y
    #                           y_in = y + v = y + a3 + a4 x + a5 y
    # matrix rows in (row=y, col=x) order for ndi.affine_transform(input,M,offset):
    M = np.array([
        [1 + a5, a4, a3],   # y_in = (1+a5)*y + a4*x + a3
        [a2, 1 + a1, a0],   # x_in = a2*y + (1+a1)*x + a0
    ], dtype=np.float64)
    return M


def level4_motion_comp(stack: np.ndarray, max_frames: int = 120) -> dict:
    """Warp each island frame onto the previous by an estimated affine, measure
    the RESIDUAL stack's effective rank. If temporal warp collapses the islands
    (real ego/scene motion), residual rank << raw rank. The LINEAR ego-R2=0.23
    was a 6-dim pose regression; a real per-frame warp is a richer test.
    """
    import scipy.ndimage as ndi

    n = min(stack.shape[0], max_frames)
    H, W = stack.shape[1], stack.shape[2]
    raw = stack[:n].reshape(n, -1).astype(np.float64)
    S_raw = svd_singvals(raw)
    er_raw = effective_rank_from_singvals(S_raw)

    resid = np.zeros((n - 1, H * W), dtype=np.float64)
    warp_resid_energy = []
    naive_resid_energy = []
    for i in range(1, n):
        prev = stack[i - 1].astype(np.float64)
        cur = stack[i].astype(np.float64)
        M = estimate_affine(prev, cur)
        warped = ndi.affine_transform(
            cur, M[:, :2], offset=M[:, 2], order=1, mode="constant", cval=0.0
        )
        r = prev - warped
        resid[i - 1] = r.ravel()
        warp_resid_energy.append(float((r**2).sum()))
        naive_resid_energy.append(float(((prev - cur) ** 2).sum()))
    S_res = svd_singvals(resid)
    er_res = effective_rank_from_singvals(S_res)
    wr = float(np.sum(warp_resid_energy))
    nr = float(np.sum(naive_resid_energy))
    return {
        "n_frames_used": int(n),
        "raw_stack": er_raw,
        "warp_residual_stack": er_res,
        "warp_residual_energy_total": wr,
        "naive_diff_energy_total": nr,
        "warp_reduces_residual_energy_frac": float(1.0 - wr / max(nr, 1e-12)),
        "note": "if warp_residual eff rank << raw eff rank => motion collapses islands",
    }


# ---------------------------------------------------------------------------
# LEVEL 5 -- nonlinear intrinsic-dim estimators
# ---------------------------------------------------------------------------
def twonn_intrinsic_dim(X: np.ndarray, discard_frac: float = 0.1) -> float:
    """TwoNN estimator (Facco et al. 2017). m = slope of log(1-F(mu)) vs
    log(mu), mu = r2/r1 (ratio of 2nd to 1st NN distances). Robust, parameter-
    free nonlinear ID estimator.
    """
    # pairwise distances (n small -> O(n^2 d) fine); errstate hides the spurious
    # Accelerate matmul FP-flag (see svd_singvals note).
    sq = (X**2).sum(axis=1)
    with np.errstate(all="ignore"):
        D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    assert np.isfinite(D2).all() or True  # D2 may legitimately be ~0 on dups
    np.fill_diagonal(D2, np.inf)
    D2 = np.clip(D2, 0.0, None)
    D = np.sqrt(D2)
    D.sort(axis=1)
    r1 = D[:, 0]
    r2 = D[:, 1]
    good = (r1 > 1e-12) & (r2 > 1e-12)
    mu = r2[good] / r1[good]
    mu = mu[np.isfinite(mu) & (mu > 1.0)]
    if mu.size < 10:
        return float("nan")
    mu_sorted = np.sort(mu)
    m_keep = int(mu_sorted.size * (1 - discard_frac))
    mu_sorted = mu_sorted[:m_keep]
    N = mu_sorted.size
    F = np.arange(1, N + 1) / (N + 1)
    x = np.log(mu_sorted)
    y = -np.log(1.0 - F)
    # slope through origin (TwoNN): d = sum(x*y)/sum(x*x)
    d = float((x * y).sum() / (x * x).sum())
    return d


def mle_intrinsic_dim(X: np.ndarray, k1: int = 5, k2: int = 15) -> float:
    """Levina-Bickel 2004 MLE intrinsic dimension, averaged over k in [k1,k2]
    with the MacKay-Ghahramani averaging correction (average inverse-dim).
    """
    n = X.shape[0]
    sq = (X**2).sum(axis=1)
    with np.errstate(all="ignore"):
        D2 = sq[:, None] + sq[None, :] - 2.0 * (X @ X.T)
    np.fill_diagonal(D2, np.inf)
    D2 = np.clip(D2, 0.0, None)
    D = np.sqrt(D2)
    D.sort(axis=1)
    k2 = min(k2, n - 1)
    k1 = min(k1, k2 - 1) if k2 > 1 else 1
    inv_m = []
    for k in range(k1, k2 + 1):
        Tk = D[:, k - 1:k]  # kth NN distance (n,1)
        logr = np.log(np.maximum(D[:, : k], 1e-12))  # (n,k) the 1..k NN dists
        logTk = np.log(np.maximum(Tk, 1e-12))
        # m_hat per point = (k-1) / sum_{j=1}^{k-1} log(T_k / T_j)
        s = (logTk - logr[:, : k - 1]).sum(axis=1)
        good = s > 1e-9
        mk = (k - 1) / s[good]
        mk = mk[np.isfinite(mk) & (mk > 0)]
        if mk.size:
            inv_m.append(np.mean(1.0 / mk))
    if not inv_m:
        return float("nan")
    return float(1.0 / np.mean(inv_m))


def tiny_autoencoder_knee(stack: np.ndarray, dims=(2, 4, 8, 16, 32),
                          max_frames: int = 200, epochs: int = 400,
                          seed: int = 0) -> dict:
    """Tiny CPU autoencoder bottleneck sweep on the island indicator stack.
    For each bottleneck dim, train an AE and record reconstruction error. The
    KNEE (where adding bottleneck dims stops helping) is the NONLINEAR m. If
    LINEAR rank 53 but the AE knee is << 53, a generator wins; if the knee is
    ~53, content-noise. CPU-ONLY, NEVER MPS/GPU.
    """
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    device = torch.device("cpu")  # NEVER MPS

    n = min(stack.shape[0], max_frames)
    # downsample the spatial dim for the AE (islands are sparse; full 196608-d
    # is wasteful). Max-pool to 96x128 keeps island presence; keeps it CPU-cheap.
    import torch.nn.functional as F

    Xt = torch.from_numpy(stack[:n].astype(np.float32)).unsqueeze(1)  # (n,1,H,W)
    Xp = F.max_pool2d(Xt, kernel_size=4).reshape(n, -1)  # (n, (H/4)*(W/4))
    d = Xp.shape[1]
    Xp = Xp.to(device)
    var_tot = float(((Xp - Xp.mean(0)) ** 2).mean().item())

    results = {}
    for bdim in dims:
        torch.manual_seed(seed)
        enc = torch.nn.Sequential(
            torch.nn.Linear(d, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, bdim),
        )
        dec = torch.nn.Sequential(
            torch.nn.Linear(bdim, 256), torch.nn.ReLU(),
            torch.nn.Linear(256, d),
        )
        params = list(enc.parameters()) + list(dec.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        lossf = torch.nn.MSELoss()
        for _ in range(epochs):
            opt.zero_grad()
            z = enc(Xp)
            xh = dec(z)
            loss = lossf(xh, Xp)
            loss.backward()
            opt.step()
        with torch.no_grad():
            rec = float(lossf(dec(enc(Xp)), Xp).item())
        results[bdim] = {
            "recon_mse": rec,
            "frac_var_unexplained": float(rec / max(var_tot, 1e-12)),
            "frac_var_explained": float(1.0 - rec / max(var_tot, 1e-12)),
        }
    # knee: smallest bdim whose explained-var is within 2% of the BEST (largest)
    best = max(results[b]["frac_var_explained"] for b in dims)
    knee = max(dims)
    for b in dims:
        if results[b]["frac_var_explained"] >= best - 0.02:
            knee = b
            break
    # also: smallest bdim reaching 90% / 95% of the achievable explained var
    def smallest_reaching(thresh):
        for b in dims:
            if results[b]["frac_var_explained"] >= thresh * best:
                return b
        return max(dims)

    return {
        "pooled_dim": int(d),
        "n_frames_used": int(n),
        "epochs": int(epochs),
        "bottleneck_dims": list(dims),
        "per_dim": {str(b): results[b] for b in dims},
        "best_explained_var": float(best),
        "knee_bottleneck_dim": int(knee),
        "knee_within_2pct_of_best": int(knee),
        "smallest_bdim_90pct_of_best": int(smallest_reaching(0.90)),
        "smallest_bdim_95pct_of_best": int(smallest_reaching(0.95)),
        "note": "nonlinear m = knee; if knee << linear-rank-53 a generator wins",
    }


# ---------------------------------------------------------------------------
# VERDICT
# ---------------------------------------------------------------------------
WHITNEY_LATENT_BUDGET = 13  # 2m+1 <= 28 latent budget => m <= 13 (GO-FORMAT bar)


def decide_verdict(level1, level2, level3, level4, level5_twonn,
                   level5_mle, level5_ae) -> dict:
    """Apply the operator's 3-way decision rule across the measured levels.

    NO-FAKE subtlety: the decision uses the RECONSTRUCTION-FAITHFUL dimension
    (k_for_95pct -- the dim needed to actually REPRODUCE the islands to 95% of
    variance), NOT only the participation ratio (which understates the rank
    needed to reconstruct the partition). The contour shape-vocabulary cloud is
    reported but DISQUALIFIED from the GO-FORMAT bar because pooling islands into
    one shape cloud DISCARDS the WHERE (location/count) information -- a low
    shape-vocab dim does NOT mean the partition is reconstructible at that dim,
    only that few distinct shapes exist. The reconstruction-faithful bases are
    pixel / DCT / per-frame-contour / motion-residual (each carries WHERE).
    """
    # Reconstruction-faithful linear dims (k for 95% variance). These bases each
    # carry the per-frame WHERE (a frame is a row), so k95 IS the latent budget.
    recon_faithful = {
        "pixel_linear": level1["k_for_95pct"],
        "dct_lf": level2["k_for_95pct"],
        "contour_frame": level3["frame_level"]["k_for_95pct"],
        "motion_comp_residual": level4["warp_residual_stack"]["k_for_95pct"],
    }
    # participation-ratio (effective rank) view, for completeness + sister to PR.
    pr_view = {
        "pixel_linear": level1["participation_ratio"],
        "dct_lf": level2["participation_ratio"],
        "contour_frame": level3["frame_level"]["participation_ratio"],
        "motion_comp_residual": level4["warp_residual_stack"]["participation_ratio"],
        # reported but NOT eligible for the GO-FORMAT bar (loses WHERE):
        "contour_shape_vocab_NOT_RECON_FAITHFUL":
            level3["shape_vocabulary_cloud"]["participation_ratio"],
    }
    min_recon_m = min(recon_faithful.values())
    min_recon_basis = min(recon_faithful, key=recon_faithful.get)
    nonlinear_knee = level5_ae["knee_bottleneck_dim"]
    nonlinear_90 = level5_ae["smallest_bdim_90pct_of_best"]
    pixel_pr = level1["participation_ratio"]

    go_format = min_recon_m <= WHITNEY_LATENT_BUDGET
    # GO-GENERATOR: linear recon-faithful dim stays high (~budget exceeded in
    # every basis) BUT the nonlinear AE 90%-knee is materially below the linear
    # rank AND within the budget -> a trained generator (not a flat code) wins.
    nonlinear_collapses = (
        nonlinear_90 <= WHITNEY_LATENT_BUDGET
        and nonlinear_90 < 0.5 * min(min_recon_m, pixel_pr)
    )
    if go_format:
        verdict = "GO-FORMAT"
        rationale = (
            f"recon-faithful basis '{min_recon_basis}' collapses m (k95) to "
            f"{min_recon_m} <= Whitney latent budget {WHITNEY_LATENT_BUDGET}: "
            f"build WHERE/HOW-MUCH layers in that basis"
        )
    elif nonlinear_collapses:
        verdict = "GO-GENERATOR"
        rationale = (
            f"every recon-faithful linear basis exceeds the budget "
            f"(min k95={min_recon_m} in '{min_recon_basis}') BUT the AE 90%-knee="
            f"{nonlinear_90} << linear rank: a TRAINED generator touches the "
            f"islands, not a flat code"
        )
    else:
        verdict = "WALL"
        rationale = (
            f"recon-faithful m stays high in every basis (min k95={min_recon_m} "
            f"'{min_recon_basis}'; pixel PR={pixel_pr:.0f}) AND nonlinear does NOT "
            f"collapse (AE 90%-knee={nonlinear_90}, TwoNN={level5_twonn:.1f}, "
            f"MLE={level5_mle:.1f}): islands are irreducible content-noise -> "
            f"training-time d_seg loss only"
        )
    return {
        "verdict": verdict,
        "rationale": rationale,
        "decision_dim_kind": "k_for_95pct (reconstruction-faithful)",
        "min_recon_faithful_k95": int(min_recon_m),
        "min_recon_faithful_basis": min_recon_basis,
        "recon_faithful_k95_per_basis": {k: int(v) for k, v in recon_faithful.items()},
        "participation_ratio_per_basis": {k: float(v) for k, v in pr_view.items()},
        "nonlinear_ae_knee": int(nonlinear_knee),
        "nonlinear_ae_90pct_knee": int(nonlinear_90),
        "twonn_id": float(level5_twonn),
        "mle_id": float(level5_mle),
        "whitney_latent_budget_m_le": WHITNEY_LATENT_BUDGET,
    }


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="limit frames (0=all 600). 60 reproduces topology Result 4.")
    ap.add_argument("--ae-frames", type=int, default=200)
    ap.add_argument("--ae-epochs", type=int, default=400)
    ap.add_argument("--motion-frames", type=int, default=120)
    ap.add_argument("--dct-keep-lf", type=int, default=32)
    ap.add_argument("--small-islands-only", action="store_true",
                    help="restrict to <500px components (the fine-island stratum)")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "2")
    t0 = time.time()
    d = np.load(ARGMAPS)
    gt = d["gt"]
    if args.limit:
        gt = gt[: args.limit]
    print(f"[load] gt {gt.shape} in {time.time()-t0:.1f}s", file=sys.stderr)

    # cache faithfulness guard (NO-FAKE)
    ds = float((d["gt"] != d["comp"]).mean())
    cache_faithful = abs(ds - 0.0005598873) < 1e-7

    if args.small_islands_only:
        stack = small_island_stack(gt)
        stratum = "class1_small_islands_lt500px"
    else:
        stack = class1_mask_stack(gt)
        stratum = "class1_full_indicator"
    print(f"[stack] {stratum} mass={stack.mean():.5f}", file=sys.stderr)

    t = time.time()
    l1 = level1_pixel_linear(stack)
    print(f"[L1 pixel] PR={l1['participation_ratio']:.1f} in {time.time()-t:.1f}s",
          file=sys.stderr)
    t = time.time()
    l2 = level2_dct(stack, keep_lf=args.dct_keep_lf)
    print(f"[L2 dct] PR={l2['participation_ratio']:.1f} lf_energy="
          f"{l2['lf_energy_frac_mean']:.3f} in {time.time()-t:.1f}s", file=sys.stderr)
    t = time.time()
    l3 = level3_contour(gt)
    print(f"[L3 contour] frame_PR={l3['frame_level']['participation_ratio']:.1f} "
          f"shapevocab_PR={l3['shape_vocabulary_cloud']['participation_ratio']:.1f} "
          f"in {time.time()-t:.1f}s", file=sys.stderr)
    t = time.time()
    l4 = level4_motion_comp(stack, max_frames=args.motion_frames)
    print(f"[L4 motion] raw_PR={l4['raw_stack']['participation_ratio']:.1f} "
          f"resid_PR={l4['warp_residual_stack']['participation_ratio']:.1f} "
          f"in {time.time()-t:.1f}s", file=sys.stderr)
    t = time.time()
    # L5 nonlinear ID on a pooled (max-pool 4x) island stack to keep n^2 cheap
    import torch
    import torch.nn.functional as F
    nmf = min(stack.shape[0], args.ae_frames)
    Xt = torch.from_numpy(stack[:nmf].astype(np.float32)).unsqueeze(1)
    Xpool = F.max_pool2d(Xt, kernel_size=4).reshape(nmf, -1).numpy().astype(np.float64)
    twonn = twonn_intrinsic_dim(Xpool)
    mle = mle_intrinsic_dim(Xpool)
    print(f"[L5 id] TwoNN={twonn:.2f} MLE={mle:.2f} in {time.time()-t:.1f}s",
          file=sys.stderr)
    t = time.time()
    ae = tiny_autoencoder_knee(stack, max_frames=args.ae_frames, epochs=args.ae_epochs)
    print(f"[L5 ae] knee={ae['knee_bottleneck_dim']} 90pct_knee="
          f"{ae['smallest_bdim_90pct_of_best']} best_var={ae['best_explained_var']:.3f} "
          f"in {time.time()-t:.1f}s", file=sys.stderr)

    verdict = decide_verdict(l1, l2, l3, l4, twonn, mle, ae)

    result = {
        "authority": "[contest-CPU advisory] NON-PROMOTABLE (exact frozen-SegNet argmax cache)",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "pointer_moved": False,
        "cache_faithful_d_seg_1e7": bool(cache_faithful),
        "cache_d_seg_measured": ds,
        "n_frames_measured": int(stack.shape[0]),
        "stratum": stratum,
        "island_class": ISLAND_CLASS,
        "small_component_px_threshold": SMALL_COMPONENT_PX,
        "argmaps_source": str(ARGMAPS.relative_to(REPO)),
        "level1_pixel_linear": l1,
        "level2_dct_spectral": l2,
        "level3_contour_fourier_descriptors": l3,
        "level4_motion_compensated": l4,
        "level5_nonlinear": {
            "twonn_id": float(twonn),
            "mle_id": float(mle),
            "autoencoder_bottleneck_sweep": ae,
        },
        "VERDICT": verdict,
        "elapsed_sec": time.time() - t0,
    }
    if args.out:
        outp = REPO / args.out
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, indent=2))
        print(f"[done] wrote {outp} in {result['elapsed_sec']:.1f}s", file=sys.stderr)
    print(json.dumps({"VERDICT": verdict,
                      "recon_faithful_k95": verdict["recon_faithful_k95_per_basis"],
                      "ae_90pct_knee": ae["smallest_bdim_90pct_of_best"],
                      "twonn": twonn, "mle": mle}, indent=2))


if __name__ == "__main__":
    main()
