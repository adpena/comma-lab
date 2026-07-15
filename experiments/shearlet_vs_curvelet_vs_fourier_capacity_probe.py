"""$0 probe: N-term reconstruction capacity of SHEARLET vs CURVELET vs FOURIER
directional frames on OUR measured frozen-SegNet boundary annulus (task #502).

This COMPLETES the #502 family selection: the curvelet half already MEASURED
curvelet ~1.09x (n600) / ~1.23x (n96) fewer coefficients than Fourier at matched
bytes (curvelet_vs_fourier_capacity_probe.py). Here we add the shearlet arm and
run the 3-way head-to-head so the family-selection VERDICT (curvelet vs shearlet
for the boundary-annulus frame) is MEASURED, not asserted.

CONFOUND DISCIPLINE (identical to the curvelet probe):
  * OMP (orthogonal matching pursuit), NOT thresholding-pursuit -- the fair
    greedy for an overcomplete frame (re-correlates the residual each step).
  * SHEARLET and CURVELET compared at MATCHED 160 cols (both at finer-orientation
    OPTIMAL FORM), both < FOURIER (236), so the result is atom SHAPE (rotation- vs
    shear-steered), not atom count. Symmetric optimal-form is the NO-FAKE crux: an
    early asymmetric run (finer shearlet vs coarser curvelet) mislabeled shearlet
    ahead; the fair matched-budget comparison reverses it.
  * IDENTICAL patch sampling + column-normalization + OMP procedure for all three
    (reused verbatim from the curvelet probe's harness, not re-implemented).
  * Multiple seeds (default 0,1,2) -> robustness, not a single lucky draw.

NOT a d_seg row. This is a linear-dictionary UPPER bound on what a trained
directional witness would realize; the realized d_seg gain must be MEASURED
through-R on exact bytes (OWED; needs a run). Tagged
[macOS advisory / not score authority]. Pointer UNMOVED.

Usage:
    .venv/bin/python experiments/shearlet_vs_curvelet_vs_fourier_capacity_probe.py \
        --npz experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
        [--patch 32 --annulus 2 --n-patches 600 --seeds 0,1,2]
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import numpy as np

from tac.boundary_math.compact_shearlet_frame import (
    CompactShearletConfig,
    compact_shearlet_feats,
)
from tac.boundary_math.compact_shearlet_frame import (
    n_atoms as shearlet_n_atoms,
)
from tac.boundary_math.lever_b_levelset_generator import (
    PolarDirectionalFourierBankConfig,
    build_coords,
    polar_directional_fourier_B,
    polar_directional_fourier_feats,
)
from tac.boundary_math.windowed_curvelet_frame import (
    WindowedCurveletConfig,
    windowed_curvelet_feats,
)
from tac.boundary_math.windowed_curvelet_frame import (
    n_atoms as curvelet_n_atoms,
)

_ROOT = Path(__file__).resolve().parents[1]


def _load_curvelet_harness():
    """Reuse the curvelet probe's _sample_patches / _omp_curve / _colnorm verbatim.

    Importing by path (not re-implementing) guarantees the shearlet arm is
    handicapped IDENTICALLY -- same OMP, same patch sampling, same normalization.
    """
    spec = importlib.util.spec_from_file_location(
        "cvl_probe", _ROOT / "experiments/curvelet_vs_fourier_capacity_probe.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _k_to_reach(curve, target):
    idx = np.where(curve <= target)[0]
    return int(idx[0] + 1) if idx.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--patch", type=int, default=32)
    ap.add_argument("--annulus", type=int, default=2)
    ap.add_argument("--n-patches", type=int, default=600)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()
    seeds = [int(s) for s in str(args.seeds).split(",") if s.strip() != ""]

    h = _load_curvelet_harness()
    coords = build_coords(args.patch, args.patch)

    # FOURIER (current witness basis) -- rich generic bank (the reference to beat).
    f_cfg = PolarDirectionalFourierBankConfig(n_scales=6, n_orient0=8, f0=1.0, base=1.7, n_iso=6)
    phi_f = h._colnorm(polar_directional_fourier_feats(coords, polar_directional_fourier_B(f_cfg)))

    # CURVELET (rotation-steered) at its OWN OPTIMAL FORM (finer orientation = 10
    # orientations, 2 scales), matched to 160 cols. SYMMETRIC optimal-form discipline
    # (NO-FAKE): an early asymmetric run (finer-orientation shearlet vs the coarser
    # committed curvelet cols192) made shearlet look ahead; giving BOTH the same
    # finer-orientation treatment at MATCHED 160-col budget is the fair family selection.
    c_cfg = WindowedCurveletConfig(n_scales=2, n_orient0=10, f0=0.6, base=2.0, w0=0.55,
                                   width_ratio=1.7, aniso=3.0, n_trans=2, coord_margin=0.4)
    phi_c = h._colnorm(windowed_curvelet_feats(coords, c_cfg))

    # SHEARLET (shear-steered; task #502 shearlet half) at its OWN OPTIMAL FORM
    # (5 shears/cone, 2 scales), matched to 160 cols == CURVELET and < FOURIER (236)
    # so the comparison is atom SHAPE (rotation-steered vs shear-steered), not count.
    s_cfg = CompactShearletConfig(n_scales=2, n_shear=2, two_cones=True, shear_step=0.35,
                                  f0=0.6, base=2.0, w0=0.55, width_ratio=1.7, aniso=3.0,
                                  n_trans=2, coord_margin=0.4)
    phi_s = h._colnorm(compact_shearlet_feats(coords, s_cfg))

    print(f"[dict] FOURIER cols={phi_f.shape[1]}  CURVELET cols={phi_c.shape[1]} "
          f"(atoms={curvelet_n_atoms(c_cfg)})  SHEARLET cols={phi_s.shape[1]} "
          f"(atoms={shearlet_n_atoms(s_cfg)})  patch={args.patch}x{args.patch}")
    print(f"[fair]  shearlet<=fourier={phi_s.shape[1] <= phi_f.shape[1]}  "
          f"shearlet==curvelet={phi_s.shape[1] == phi_c.shape[1]}  "
          f"cache={Path(args.npz).name}  seeds={seeds}")

    kmax = 24
    ks = [1, 2, 3, 4, 6, 8, 12, 16, 24]
    targets = (0.30, 0.20, 0.10, 0.05)

    agg = {"F": [], "C": [], "S": []}  # per-seed full curves for the mean
    for seed in seeds:
        patches = h._sample_patches(args.npz, args.patch, args.annulus, args.n_patches, seed)
        if patches.shape[0] == 0:
            print(f"[error] seed={seed}: no boundary patches sampled")
            continue
        cf = h._omp_curve(phi_f, patches, kmax)
        cc = h._omp_curve(phi_c, patches, kmax)
        cs = h._omp_curve(phi_s, patches, kmax)
        agg["F"].append(cf)
        agg["C"].append(cc)
        agg["S"].append(cs)

        kk = [k for k in ks if k <= min(len(cf), len(cc), len(cs))]
        print(f"\n[measured seed={seed}] N-term rel-error (frac of patch energy) vs coeff budget K "
              f"(patches={patches.shape[0]}):")
        print(f"{'K':>4} {'FOURIER':>9} {'CURVELET':>9} {'SHEARLET':>9} {'best':>9}")
        for K in kk:
            ef, ec, es = cf[K - 1], cc[K - 1], cs[K - 1]
            best = min([("F", ef), ("C", ec), ("S", es)], key=lambda t: t[1])[0]
            print(f"{K:>4} {ef:9.4f} {ec:9.4f} {es:9.4f} {best:>9}")
        print(f"[measured seed={seed}] coeff budget to reach rel-err targets "
              f"[macOS advisory / not score authority]:")
        for td in targets:
            kf, kc, ksh = _k_to_reach(cf, td), _k_to_reach(cc, td), _k_to_reach(cs, td)
            def rat(a, b):
                return f"{a / b:.2f}x" if (a and b) else "n/a"
            print(f"  <= {td:.2f}: FOURIER K={kf} CURVELET K={kc} SHEARLET K={ksh}  "
                  f"| K_F/K_C={rat(kf, kc)} K_F/K_S={rat(kf, ksh)} K_C/K_S={rat(kc, ksh)}")

    if not agg["F"]:
        print("[error] no seeds produced patches")
        return

    mf = np.mean(np.stack(agg["F"]), axis=0)
    mc = np.mean(np.stack(agg["C"]), axis=0)
    ms = np.mean(np.stack(agg["S"]), axis=0)
    print(f"\n[measured MEAN over seeds {seeds}]  [macOS advisory / not score authority; pointer UNMOVED]")
    kk = [k for k in ks if k <= min(len(mf), len(mc), len(ms))]
    print(f"{'K':>4} {'FOURIER':>9} {'CURVELET':>9} {'SHEARLET':>9} {'best':>9}")
    for K in kk:
        ef, ec, es = mf[K - 1], mc[K - 1], ms[K - 1]
        best = min([("F", ef), ("C", ec), ("S", es)], key=lambda t: t[1])[0]
        print(f"{K:>4} {ef:9.4f} {ec:9.4f} {es:9.4f} {best:>9}")
    print("[measured MEAN] coeff budget to reach rel-err targets (family selection):")
    for td in targets:
        kf, kc, ksh = _k_to_reach(mf, td), _k_to_reach(mc, td), _k_to_reach(ms, td)
        def rat(a, b):
            return f"{a / b:.2f}x" if (a and b) else "n/a"
        print(f"  <= {td:.2f}: FOURIER K={kf} CURVELET K={kc} SHEARLET K={ksh}  "
              f"| K_F/K_C={rat(kf, kc)} K_F/K_S={rat(kf, ksh)} K_C/K_S={rat(kc, ksh)}")


if __name__ == "__main__":
    main()
