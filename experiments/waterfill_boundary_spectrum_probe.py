"""$0 numpy probe: boundary-local tangent-frame spectrum + Eq-4 waterfilling
capacity comparison of an ORIENTED (curvelet-like) vs ISOTROPIC (global-Fourier)
basis on OUR frozen-SegNet argmax boundary annulus.

Grounds task #502 (genuine localized curvelet/shearlet frames). NOT a d_seg row:
this is a LINEAR-BASIS capacity/distortion probe = an UPPER bound on what a trained
curvelet witness would realize. It de-risks #502 by putting a MEASURED number on the
"how much does respecting boundary anisotropy buy?" question.

Theory anchor: Torralba & Weiss arXiv 2607.07470 — isotropic Fourier diagonalizes the
alignment(B)/uniformity(Sigma) pair ONLY under GLOBAL translation-invariant stationarity
(Thm 2.1, Peligrad-Wu). Our codim-1 oriented boundary VIOLATES stationarity, so global
sinusoids are provably sub-optimal. Eq-4 waterfilling greedy (their Alg 1) reallocated
onto tangent/normal-anisotropic bands quantifies the capacity gain.

Method (all MEASURED from existing gt caches; no training, no dispatch):
  1. argmax lstars (H=384,W=512) per pair -> inter-class edge set -> dilated annulus.
  2. margin-field gradient (Sobel) at each boundary pixel -> NORMAL direction; tangent = +90 deg.
  3. Per boundary patch: 2D FFT power; accumulate power into a (radius, angle-relative-to-normal)
     polar histogram (NO pixel rotation -- rotate the angle coordinate only). Averaged over patches.
  4. sigma^2(r, rel_ang) = accumulated tangent-frame power spectrum.
        rel_ang ~ 0deg   => wavevector along NORMAL  (variation across the boundary)
        rel_ang ~ 90deg  => wavevector along TANGENT (variation along the boundary: dashes/curvature)
  5. Gaussian reverse water-filling (the canonical form of the paper's Eq-4 waterfilling)
     on the tangent-frame spectrum sigma^2(r,a), for two allocation regimes:
        ORIENTED  : each (r,a) cell allocates rate freely (a curvelet/steerable wedge can put
                    capacity exactly where the boundary energy is).
        ISOTROPIC : global Fourier is orientation-blind -> all a-cells at a radius share ONE
                    rate P_r; funding radius r costs a_bins (must pay EVERY orientation equally),
                    so capacity is wasted on the empty (diagonal) orientations.
     D(B) = sum_k sigma^2_k * 2^(-2 P_k) at rate budget B (energy-preserving; no free eps knob).
  6. Report D_iso/D_orient at matched budget, capacity ratio B_iso/B_orient to matched
     distortion, and the boundary orientation anisotropy (max/min orientation energy).

Usage:
    .venv/bin/python experiments/waterfill_boundary_spectrum_probe.py --npz <gt_cache.npz>
        [--patch 32 --annulus 2 --patches-per-frame 200 --r-bins 14 --a-bins 12 --max-frames 0 --seed 0]
"""
from __future__ import annotations

import argparse

import numpy as np


def sobel_grad(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (gy, gx) via separable Sobel; m is (H,W) float32."""
    sm = np.array([1.0, 2.0, 1.0])
    # gx = smooth_y * deriv_x
    gx = np.zeros_like(m)
    gy = np.zeros_like(m)
    # deriv in x, smooth in y
    dx = np.zeros_like(m)
    dx[:, 1:-1] = m[:, 2:] - m[:, :-2]
    gx[1:-1, :] = sm[0] * dx[:-2, :] + sm[1] * dx[1:-1, :] + sm[2] * dx[2:, :]
    dy = np.zeros_like(m)
    dy[1:-1, :] = m[2:, :] - m[:-2, :]
    gy[:, 1:-1] = sm[0] * dy[:, :-2] + sm[1] * dy[:, 1:-1] + sm[2] * dy[:, 2:]
    return gy, gx


def edge_annulus(lab: np.ndarray, dil: int) -> np.ndarray:
    """Boolean inter-class edge set dilated by `dil` px (Chebyshev)."""
    e = np.zeros_like(lab, dtype=bool)
    e[:, :-1] |= lab[:, :-1] != lab[:, 1:]
    e[:, 1:] |= lab[:, :-1] != lab[:, 1:]
    e[:-1, :] |= lab[:-1, :] != lab[1:, :]
    e[1:, :] |= lab[:-1, :] != lab[1:, :]
    if dil <= 0:
        return e
    out = e.copy()
    for _ in range(dil):
        nxt = out.copy()
        nxt[:, :-1] |= out[:, 1:]
        nxt[:, 1:] |= out[:, :-1]
        nxt[:-1, :] |= out[1:, :]
        nxt[1:, :] |= out[:-1, :]
        out = nxt
    return out


def accumulate_spectrum(npz_path, patch, dil, ppf, max_frames, seed, r_bins, a_bins):
    """Accumulate TWO tangent-frame-vs-global polar power spectra over boundary patches.

    S_aligned[r,a]: patch spectrum re-oriented so the local boundary NORMAL sits at angle 0
        (what an ORIENTATION-ADAPTIVE / curvelet dictionary effectively sees -> energy
        concentrates in the normal wedge because every edge is aligned the same way).
    S_global[r,a]: patch spectrum in the FIXED image frame (what a GLOBAL Fourier basis
        must cover -> energy SMEARS across orientations because boundaries across the image
        point in every direction).
    Rotation is energy-preserving, so sum(S_aligned)==sum(S_global): the ONLY difference is
    how concentrated the energy is across cells = the curvelet-vs-Fourier N-term question.
    Angle binned over [0,pi) (real signal -> spectrum is point-symmetric)."""
    rng = np.random.default_rng(seed)
    z = np.load(npz_path, allow_pickle=True)
    lstars = z["lstars"]
    margins = z["margins"]
    n = lstars.shape[0] if max_frames <= 0 else min(max_frames, lstars.shape[0])
    H, W = lstars.shape[1], lstars.shape[2]
    half = patch // 2

    ky = np.fft.fftshift(np.fft.fftfreq(patch)) * patch
    kx = np.fft.fftshift(np.fft.fftfreq(patch)) * patch
    KX, KY = np.meshgrid(kx, ky)
    R = np.sqrt(KX**2 + KY**2)
    ANG = np.arctan2(KY, KX)
    r_max = patch / 2.0
    r_edges = np.linspace(0.0, r_max, r_bins + 1)
    ri_flat = np.clip(np.digitize(R.ravel(), r_edges) - 1, 0, r_bins - 1)

    acc_al = np.zeros(r_bins * a_bins, dtype=np.float64)
    cnt_al = np.zeros(r_bins * a_bins, dtype=np.float64)
    acc_gl = np.zeros(r_bins * a_bins, dtype=np.float64)
    cnt_gl = np.zeros(r_bins * a_bins, dtype=np.float64)
    npatch = 0

    wy = np.hanning(patch)
    win = np.outer(wy, wy)
    ang_flat = ANG.ravel()

    for fi in range(n):
        lab = lstars[fi]
        m = margins[fi].astype(np.float32)
        gy, gx = sobel_grad(m)
        ann = edge_annulus(lab, dil)
        ann[:half, :] = False
        ann[H - half:, :] = False
        ann[:, :half] = False
        ann[:, W - half:] = False
        ys, xs = np.nonzero(ann)
        if ys.size == 0:
            continue
        gmag = np.sqrt(gx[ys, xs] ** 2 + gy[ys, xs] ** 2)
        keep = gmag > (0.05 * (gmag.max() + 1e-9))
        ys, xs = ys[keep], xs[keep]
        if ys.size == 0:
            continue
        if ys.size > ppf:
            sel = rng.choice(ys.size, size=ppf, replace=False)
            ys, xs = ys[sel], xs[sel]
        for y, x in zip(ys, xs, strict=False):
            p = m[y - half:y + half, x - half:x + half]
            if p.shape != (patch, patch):
                continue
            p = p - p.mean()
            F = np.fft.fftshift(np.fft.fft2(p * win))
            Pwr = (F.real**2 + F.imag**2).ravel()
            theta_n = np.arctan2(gy[y, x], gx[y, x])
            # aligned: subtract local normal angle
            rel = np.mod(ang_flat - theta_n, np.pi)  # [0,pi)
            ai_al = np.clip((rel / np.pi * a_bins).astype(int), 0, a_bins - 1)
            np.add.at(acc_al, ri_flat * a_bins + ai_al, Pwr)
            np.add.at(cnt_al, ri_flat * a_bins + ai_al, 1.0)
            # global: fixed image frame
            phi = np.mod(ang_flat, np.pi)
            ai_gl = np.clip((phi / np.pi * a_bins).astype(int), 0, a_bins - 1)
            np.add.at(acc_gl, ri_flat * a_bins + ai_gl, Pwr)
            np.add.at(cnt_gl, ri_flat * a_bins + ai_gl, 1.0)
            npatch += 1
    S_al = np.where(cnt_al > 0, acc_al / np.maximum(cnt_al, 1.0), 0.0).reshape(r_bins, a_bins)
    S_gl = np.where(cnt_gl > 0, acc_gl / np.maximum(cnt_gl, 1.0), 0.0).reshape(r_bins, a_bins)
    return S_al, S_gl, npatch, r_edges


def _reverse_waterfill_distortion(sigma2, cost, budget):
    """Gaussian reverse water-filling. sigma2, cost: 1D arrays per band. Allocate rate
    P_k>=0 minimizing D=sum sigma2_k * 2^(-2 P_k) s.t. sum cost_k*P_k = budget.
    Closed form: P_k = max(0, 0.5*log2(sigma2_k/(cost_k*theta))); bisection on theta.
    Returns achieved distortion D."""
    s = sigma2.astype(np.float64)
    c = cost.astype(np.float64)
    active = s > 0
    if not active.any():
        return 0.0

    def rate_at(theta):
        P = np.maximum(0.0, 0.5 * np.log2(s / (c * theta + 1e-300) + 1e-300))
        return np.sum(c * P), P

    lo, hi = 1e-30, (s / np.maximum(c, 1e-300)).max() + 1e-9
    for _ in range(200):
        mid = np.sqrt(lo * hi)
        r, _ = rate_at(mid)
        if r > budget:
            lo = mid
        else:
            hi = mid
    _, P = rate_at(np.sqrt(lo * hi))
    D = np.sum(s * np.power(2.0, -2.0 * P))
    return D


def oriented_vs_isotropic_rd(spec, budgets):
    """spec: (r_bins, a_bins) tangent-frame power density.
    ORIENTED  : every (r,a) cell allocates rate freely (cost 1/cell).
    ISOTROPIC : Fourier is orientation-blind -> all a-cells at a radius share ONE rate P_r;
                funding radius r costs a_bins (must pay every orientation the same).
    Returns (D_orient[b], D_iso[b], V) as fraction of total variance."""
    r_bins, a_bins = spec.shape
    V = spec.sum()
    D_or = np.zeros(len(budgets))
    D_is = np.zeros(len(budgets))
    # oriented: flat cells
    s_cells = spec.ravel()
    cost_cells = np.ones_like(s_cells)
    # isotropic: per-radius bands; a radius band's total variance = sum_a spec(r,a);
    # cost = a_bins (one rate replicated over a_bins orientations). Distortion of that
    # band under shared rate P_r = [sum_a spec(r,a)] * 2^(-2 P_r).
    s_rad = spec.sum(axis=1)
    cost_rad = np.full(r_bins, float(a_bins))
    for i, b in enumerate(budgets):
        D_or[i] = _reverse_waterfill_distortion(s_cells, cost_cells, b) / max(V, 1e-30)
        D_is[i] = _reverse_waterfill_distortion(s_rad, cost_rad, b) / max(V, 1e-30)
    return D_or, D_is, V


def concentration(spec):
    """Participation ratio PR=(sum p)^2/sum p^2 (effective #active cells; lower=concentrated),
    peak-orientation energy fraction, and normal-wedge/tangent-wedge energy ratio
    (aligned frame: a-bin 0 = along-normal wavevector, last a-bin = along-tangent)."""
    p = spec.ravel().astype(np.float64)
    pr = (p.sum() ** 2) / (np.sum(p * p) + 1e-30)
    ori = spec.sum(axis=0)  # energy per orientation bin
    peak_frac = ori.max() / (ori.sum() + 1e-30)
    a_bins = spec.shape[1]
    norm_e = ori[0]
    tan_e = ori[a_bins - 1]
    return pr, peak_frac, norm_e, tan_e, ori


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
    ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--annulus", type=int, default=2)
    ap.add_argument("--patches-per-frame", type=int, default=200)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--r-bins", type=int, default=14)
    ap.add_argument("--a-bins", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    S_al, S_gl, npatch, r_edges = accumulate_spectrum(
        args.npz, args.patch, args.annulus, args.patches_per_frame,
        args.max_frames, args.seed, args.r_bins, args.a_bins,
    )
    print(f"[measured] patches accumulated: {npatch}  spec shape {S_al.shape} "
          f"(r_bins={args.r_bins} a_bins={args.a_bins})  n={args.npz.split('/')[-1]}")

    # Drop DC radius bin (r index 0) -- carries no oriented structure; both bases share it.
    S_al[0, :] = 0.0

    pr, peak, norm_e, tan_e, ori = concentration(S_al)
    a_bins = args.a_bins
    print(f"[measured] tangent-frame spectrum: participation-ratio={pr:.2f} of {S_al.size} cells "
          f"(lower=more anisotropic/concentrated); peak-orientation energy frac={peak:.3f}")
    print(f"[measured] per-orientation energy (bin0=along-normal wavevector ... bin{a_bins-1}=along-tangent):")
    print("           " + "  ".join(f"{e/ori.sum():.3f}" for e in ori))
    aniso = ori.max() / (ori.min() + 1e-30)
    print(f"[measured] orientation anisotropy (max/min orientation energy) = {aniso:.2f}x")

    # ---- Gaussian reverse-waterfilling: free-per-orientation vs orientation-blind ----
    budgets = np.linspace(0.02, 14.0, 200)  # rate budget in bits/cell-equivalents
    D_or, D_is, _ = oriented_vs_isotropic_rd(S_al, budgets)

    def cap_ratio(target_d):
        io = np.where(D_or <= target_d)[0]
        ii = np.where(D_is <= target_d)[0]
        if io.size == 0 or ii.size == 0:
            return float("nan"), float("nan"), float("nan")
        return budgets[ii[0]] / budgets[io[0]], budgets[io[0]], budgets[ii[0]]

    def dist_ratio_at(bud):
        idx = int(np.argmin(np.abs(budgets - bud)))
        return D_is[idx] / max(D_or[idx], 1e-12), D_or[idx], D_is[idx]

    print("\n[measured] reverse-waterfilling distortion (frac of variance) vs rate budget:")
    print(f"{'budget':>7} {'D_orient':>9} {'D_iso':>9} {'D_iso/D_or':>10}")
    for b in (0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 9.0):
        idx = int(np.argmin(np.abs(budgets - b)))
        print(f"{budgets[idx]:7.2f} {D_or[idx]:9.4f} {D_is[idx]:9.4f} {D_is[idx]/max(D_or[idx],1e-9):10.2f}")

    print(f"\n[measured] SUMMARY (patches={npatch}, cache={args.npz.split('/')[-1]})")
    for td in (0.70, 0.60, 0.50, 0.40):
        cr, b_or, b_is = cap_ratio(td)
        print(f"  capacity ratio B_iso/B_orient to reach D<={td:.2f}: {cr:.2f}x  "
              f"(B_or={b_or:.2f} B_is={b_is:.2f})")
    for bud in (1.0, 2.0, 4.0):
        dr, do, di = dist_ratio_at(bud)
        print(f"  distortion ratio D_iso/D_orient at budget={bud:.2f}: {dr:.2f}  "
              f"(D_or={do:.4f} D_is={di:.4f})")
    print(f"  orientation anisotropy (max/min orientation energy): {aniso:.2f}x")
    print(f"  participation ratio (of {S_al.size} cells): {pr:.2f}")


if __name__ == "__main__":
    main()
