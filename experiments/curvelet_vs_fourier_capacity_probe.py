"""$0 probe: N-term reconstruction capacity of the GENUINELY LOCALIZED windowed-
curvelet frame (task #502) vs the witness's current oriented-Fourier basis, on OUR
measured frozen-SegNet boundary annulus.

This is the SECOND, complementary de-risk to the linear reverse-waterfilling probe
(waterfill_boundary_spectrum_probe.py, which measured a 1.7-2.0x capacity gain in the
spectral domain). Here we measure the SPATIAL-domain N-term sparsity directly: how
well each dictionary reconstructs real boundary patches at a MATCHED coefficient
budget (= matched counted archive bytes). Curvelet theory (Candes-Donoho 2004)
predicts localized oriented atoms give optimal N-term approximation for C2 curved
edges, while global plane waves pay a Gibbs/localization tax.

NOT a d_seg row. This is a linear-dictionary UPPER bound on what a trained curvelet
witness would realize; the realized d_seg gain must be MEASURED through-R on exact
bytes (OWED; needs a run). Tagged [macOS advisory / not score authority]. Pointer
UNMOVED.

Method (all MEASURED from existing gt caches; numpy only; no training, no dispatch):
  1. Boundary annulus patches (PxP) sampled from cached SegNet margins (same annulus
     construction as the waterfill probe -- imported, not duplicated). Mean-subtracted,
     unit-L2 normalized.
  2. Two patch-local dictionaries over the PxP grid ([-1,1]^2 coords), COLUMN-NORMALIZED:
       FOURIER   : polar_directional_fourier feats (the current witness basis).
       CURVELET  : windowed_curvelet feats (genuinely localized; task #502 primitive).
     Dictionary column counts matched as closely as the parametric banks allow.
  3. Thresholding-pursuit N-term reconstruction: for coefficient budget K, select the
     top-K atoms by |Phi^T p| then least-squares fit those K columns. IDENTICAL
     procedure for both dictionaries (both handicapped equally). rel_err = ||p-p_hat||^2/||p||^2.
  4. Report mean rel_err vs K, the K-ratio to reach matched error, and the error-ratio
     at matched K. Lower curvelet error / smaller curvelet K = the spatial N-term gain.

Usage:
    .venv/bin/python experiments/curvelet_vs_fourier_capacity_probe.py \
        --npz experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
        [--patch 32 --annulus 2 --n-patches 600 --seed 0]
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from tac.boundary_math.lever_b_levelset_generator import (
    PolarDirectionalFourierBankConfig,
    build_coords,
    polar_directional_fourier_B,
    polar_directional_fourier_feats,
)
from tac.boundary_math.windowed_curvelet_frame import (
    WindowedCurveletConfig,
    n_atoms,
    windowed_curvelet_feats,
)

_ROOT = Path(__file__).resolve().parents[1]


def _load_waterfill():
    spec = importlib.util.spec_from_file_location(
        "wf_probe", _ROOT / "experiments/waterfill_boundary_spectrum_probe.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _colnorm(phi: np.ndarray) -> np.ndarray:
    n = np.sqrt((phi.astype(np.float64) ** 2).sum(axis=0))
    n = np.where(n < 1e-12, 1.0, n)
    return (phi / n).astype(np.float64)


def _sample_patches(npz_path, patch, dil, n_patches, seed):
    wf = _load_waterfill()
    rng = np.random.default_rng(seed)
    z = np.load(npz_path, allow_pickle=True)
    lstars = z["lstars"]
    margins = z["margins"]
    n_frames, H, W = lstars.shape
    half = patch // 2
    patches = []
    order = rng.permutation(n_frames)
    per_frame = max(1, n_patches // max(1, min(n_frames, 64)))
    for fi in order:
        if len(patches) >= n_patches:
            break
        lab = lstars[fi]
        m = margins[fi].astype(np.float32)
        ann = wf.edge_annulus(lab, dil)
        ann[:half, :] = False
        ann[H - half:, :] = False
        ann[:, :half] = False
        ann[:, W - half:] = False
        ys, xs = np.nonzero(ann)
        if ys.size == 0:
            continue
        take = min(per_frame, ys.size)
        sel = rng.choice(ys.size, size=take, replace=False)
        for idx in sel:
            y, x = int(ys[idx]), int(xs[idx])
            p = m[y - half:y + half, x - half:x + half]
            if p.shape != (patch, patch):
                continue
            v = p.ravel().astype(np.float64)
            v = v - v.mean()
            nrm = np.sqrt((v ** 2).sum())
            if nrm < 1e-9:
                continue
            patches.append(v / nrm)
            if len(patches) >= n_patches:
                break
    return np.stack(patches, axis=0) if patches else np.zeros((0, patch * patch))


def _omp_curve(phi: np.ndarray, patches: np.ndarray, kmax: int) -> np.ndarray:
    """Mean relative reconstruction error at every K=1..kmax via Orthogonal Matching Pursuit.

    OMP re-correlates the RESIDUAL with the dictionary at each step (not the raw patch),
    so it does not pick coherent near-duplicate atoms -- the fair N-term greedy for an
    overcomplete frame. IDENTICAL procedure for both dictionaries. Returns rel-err[K-1]
    (||p||^2 == 1 so squared residual is the relative error).
    """
    n_patch, n_pix = patches.shape
    D = phi.shape[1]
    kmax = min(kmax, D)
    acc = np.zeros(kmax)
    with np.errstate(all="ignore"):  # spurious macOS-Accelerate FP-flag warnings only
        for i in range(n_patch):
            p = patches[i]
            r = p.copy()
            chosen: list[int] = []
            for step in range(kmax):
                c = np.abs(phi.T @ r)
                if chosen:
                    c[chosen] = -1.0
                j = int(np.argmax(c))
                chosen.append(j)
                A = phi[:, chosen]
                coef, *_ = np.linalg.lstsq(A, p, rcond=None)
                r = p - A @ coef
                acc[step] += float((r ** 2).sum())
    return acc / max(n_patch, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--annulus", type=int, default=2)
    ap.add_argument("--n-patches", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    coords = build_coords(args.patch, args.patch)

    # Fourier dictionary (the current witness basis) -- rich generic bank.
    f_cfg = PolarDirectionalFourierBankConfig(n_scales=6, n_orient0=8, f0=1.0, base=1.7, n_iso=6)
    phi_f = _colnorm(polar_directional_fourier_feats(coords, polar_directional_fourier_B(f_cfg)))

    # Curvelet dictionary: edge-shaped (aniso>1), low-freq odd atoms that tile the patch
    # (n_trans grid) at a few parabolic scales -- kept COMPARABLE to / SMALLER than Fourier
    # so any win comes from atom SHAPE (localized + oriented), not dictionary count.
    c_cfg = WindowedCurveletConfig(n_scales=3, n_orient0=6, f0=0.6, base=2.0, w0=0.55,
                                   width_ratio=1.7, aniso=3.0, n_trans=2, coord_margin=0.4)
    phi_c = _colnorm(windowed_curvelet_feats(coords, c_cfg))

    print(f"[dict] FOURIER cols={phi_f.shape[1]}  CURVELET cols={phi_c.shape[1]} "
          f"(atoms={n_atoms(c_cfg)})  patch={args.patch}x{args.patch}  "
          f"curvelet_smaller={phi_c.shape[1] <= phi_f.shape[1]}")

    patches = _sample_patches(args.npz, args.patch, args.annulus, args.n_patches, args.seed)
    if patches.shape[0] == 0:
        print("[error] no boundary patches sampled")
        return
    print(f"[measured] boundary patches sampled: {patches.shape[0]}  cache={Path(args.npz).name}")

    kmax = 24
    curve_f = _omp_curve(phi_f, patches, kmax)
    curve_c = _omp_curve(phi_c, patches, kmax)
    ks = [1, 2, 3, 4, 6, 8, 12, 16, 24]
    ks = [k for k in ks if k <= min(len(curve_f), len(curve_c))]
    e_f = np.array([curve_f[k - 1] for k in ks])
    e_c = np.array([curve_c[k - 1] for k in ks])

    print("\n[measured] N-term reconstruction rel-error (frac of patch energy) vs coeff budget K:")
    print(f"{'K':>4} {'FOURIER':>9} {'CURVELET':>9} {'F/C ratio':>10}")
    for i, K in enumerate(ks):
        ratio = e_f[i] / max(e_c[i], 1e-12)
        print(f"{K:>4} {e_f[i]:9.4f} {e_c[i]:9.4f} {ratio:10.2f}")

    # capacity ratio: K_fourier / K_curvelet to reach matched error targets (full curve).
    def k_to_reach(curve, target):
        idx = np.where(curve <= target)[0]
        return int(idx[0] + 1) if idx.size else None

    print("\n[measured] SUMMARY  [macOS advisory / not score authority; pointer UNMOVED]")
    for td in (0.30, 0.20, 0.10, 0.05):
        kf = k_to_reach(curve_f, td)
        kc = k_to_reach(curve_c, td)
        if kf is not None and kc is not None:
            print(f"  coeff budget to reach rel-err<={td:.2f}: FOURIER K={kf}  CURVELET K={kc}  "
                  f"K_fourier/K_curvelet={kf / kc:.2f}x")
        else:
            print(f"  coeff budget to reach rel-err<={td:.2f}: FOURIER K={kf}  CURVELET K={kc}")
    for K in (4, 8, 16):
        if K in ks:
            i = ks.index(K)
            print(f"  error ratio F/C at matched K={K}: {e_f[i] / max(e_c[i], 1e-12):.2f}  "
                  f"(F={e_f[i]:.4f} C={e_c[i]:.4f})")


if __name__ == "__main__":
    main()
