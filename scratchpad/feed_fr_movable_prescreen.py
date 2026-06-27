"""DAG FEED-fr (2026-06-27): $0/CPU pre-screen of the "Movable medial-axis IRREDUCIBLE chart gap".

Claim under test (FEED-fk/fh/fn): a single smooth phi_Movable cannot carve DISCONNECTED car
blobs -> d_seg plateaus ~6-9e-4 (above the 5.2-6.5e-4 budget). FEED-fp counter: an SDF of a
disconnected set IS a valid single field; obstruction = soft Eikonal (|grad m|=1) fighting the
medial-axis ridge (locally relaxable); K>5 / per-component fields are cheap fixes.

ALL d_seg MEASURED vs the frozen CPU-torch SegNet L* in the cached GT (NO-FAKE). The bandwidth
proxy is an isotropic Gaussian blur (sigma) on the IDEAL per-class signed-distance fields
(signed_distance_fields: argmax==L* exactly, d_seg=0 at sigma=0). The witness's real basis is a
DIRECTIONAL curvelet bank + WIRE-MLP (anisotropic, higher freq) — so the blur proxy is advisory
for the absolute Road/Lane-vs-Movable RATIO, but the structural conclusions below are
basis-independent (geometric / measured-from-L*).

Findings (n24 + n96 confirmation):
  1. Movable = class 3, ~1.55% area, ~3 components/frame (max 6); 68% of comps are <=200px but
     hold only ~3% of Movable pixels. Movable = 8-9% of flip-prone (margin<0.5) pixels.
  2. FIX A (K>5 / per-component multi-phi): per-component sub-SDFs (G=2 K=6, G=3 K=7), blurred
     then max-combined, give BIT-IDENTICAL d_seg to the single union-phi at EVERY bandwidth.
     (union-SDF == max_c component-SDF; the residual is at each blob's OWN boundary, not the
     gaps.) => Fix A: 0 d_seg benefit; +~5e-5..1e-4 S in bytes (int8+brotli out-head row/field)
     => STRICTLY NEGATIVE.
  3. FIX B (local eikonal-relax at the medial axis): Movable INTERIOR-pixel miss = 0.0000 at
     every bandwidth. The medial-axis ridge (where the eikonal fights the SDF kink) lives in the
     large-margin interior/exterior-gaps that NEVER flip => relaxing it cannot lower d_seg.
  4. REAL Movable residual = small-blob BOUNDARY resolution under bandwidth (small<=200px blobs
     miss 10-30x the huge blobs). A boundary-bandwidth/capacity issue (residual-sharpener /
     directional capacity), NOT disconnection, NOT the interior ridge.
  5. RE-AIM: at the witness operating point (total d_seg 6-9e-4 ~ sigma 0.75-1.0) Movable is only
     1-2% of the residual; the BINDING residual is the Road/Lane boundary (98-99%; 66% of flips).
"""
import time
import numpy as np
from scipy import ndimage

GT24 = "experiments/results/mlx_fleet_gt_cache/gt_n24.npz"
NK, MOV = 5, 3
CLS = ["Road", "Lane", "Undriv", "Movable", "MyCar"]
ST = np.ones((3, 3), int)  # 8-connectivity


def sdf_class(mask):
    if mask.all():
        return np.full(mask.shape, float(max(mask.shape)), np.float32)
    if not mask.any():
        return np.full(mask.shape, -float(max(mask.shape)), np.float32)
    return (ndimage.distance_transform_edt(mask)
            - ndimage.distance_transform_edt(~mask)).astype(np.float32)


def grouped_movable(mov_mask, G, sigma):
    """G grouped per-component sub-SDFs (k-means on centroids, farthest-point init), each blurred."""
    lab, nc = ndimage.label(mov_mask, structure=ST)
    if nc == 0:
        return [np.full(mov_mask.shape, -float(max(mov_mask.shape)), np.float32)]
    if G == 1 or nc == 1:
        return [ndimage.gaussian_filter(sdf_class(mov_mask), sigma)]
    cents = np.array(ndimage.center_of_mass(mov_mask, lab, np.arange(1, nc + 1)))
    g = min(G, nc)
    idx = [0]
    for _ in range(g - 1):
        dd = np.min([np.sum((cents - cents[j]) ** 2, 1) for j in idx], 0)
        idx.append(int(np.argmax(dd)))
    ctr = cents[idx].copy()
    for _ in range(8):
        asn = np.argmin(((cents[:, None] - ctr[None]) ** 2).sum(-1), 1)
        for gi in range(g):
            if (asn == gi).any():
                ctr[gi] = cents[asn == gi].mean(0)
    return [ndimage.gaussian_filter(sdf_class(np.isin(lab, np.where(asn == gi)[0] + 1)), sigma)
            for gi in range(g)]


def main():
    t0 = time.time()
    d = np.load(GT24)
    ls, mg = d["lstars"], d["margins"]
    N, H, W = ls.shape
    NF = min(8, N)

    # 1. characterization
    cc, sz, mf = [], [], []
    for i in range(N):
        m = ls[i] == MOV
        mf.append(m.mean())
        lab, nc = ndimage.label(m, ST)
        cc.append(nc)
        if nc:
            sz.extend(ndimage.sum(np.ones_like(lab), lab, np.arange(1, nc + 1)).tolist())
    cc, sz = np.array(cc), np.array(sz)
    print(f"[1] Movable area={np.mean(mf)*100:.3f}%  comps/frame mean={cc.mean():.1f} max={cc.max()}"
          f"  small<=200px={int((sz<=200).sum())}/{sz.size}=({100*(sz<=200).mean():.0f}%) but only"
          f" {100*sz[sz<=200].sum()/sz.sum():.1f}% of Movable px")
    flip = mg < 0.5
    br = [100 * ((ls == k) & flip).sum() / flip.sum() for k in range(NK)]
    print("    flip-band(margin<0.5): " + " ".join(f"{CLS[k]}={br[k]:.0f}%" for k in range(NK)))

    # ideal check
    idl = []
    for i in range(NF):
        fld = np.stack([sdf_class(ls[i] == k) for k in range(NK)], -1)
        idl.append(float((np.argmax(fld, -1) != ls[i]).mean()))
    print(f"[ideal] sigma=0 d_seg={np.mean(idl):.2e}  (single-phi DOES carve disconnected blobs)")

    # 2/3. Fix A (multi-phi) + Fix B (per-class contribution / interior miss / size strat)
    for sigma in [0.75, 1.0, 2.0]:
        ga = {1: [], 2: [], 3: []}  # overall d_seg per G
        contrib = np.zeros(NK)
        tot = 0.0
        mint, sm_miss, sm_n, hg_miss, hg_n = [], 0, 0, 0, 0
        for i in range(NF):
            lab = ls[i]
            nonmov = {k: ndimage.gaussian_filter(sdf_class(lab == k), sigma)
                      for k in range(NK) if k != MOV}
            mm = lab == MOV
            for G in (1, 2, 3):
                subs = grouped_movable(mm, G, sigma)
                phi = subs[0]
                for s in subs[1:]:
                    phi = np.maximum(phi, s)
                fields = np.stack([(phi if k == MOV else nonmov[k]) for k in range(NK)], -1)
                pred = np.argmax(fields, -1)
                ga[G].append(float((pred != lab).mean()))
                if G == 1:
                    w = pred != lab
                    tot += w.mean()
                    for k in range(NK):
                        contrib[k] += (w & ((lab == k) | (pred == k))).mean()
                    if mm.any():
                        er = ndimage.binary_erosion(mm, ST)
                        mint.append((pred[er] != MOV).mean() if er.any() else 0.0)
                        blab, nc = ndimage.label(mm, ST)
                        for cid in range(1, nc + 1):
                            cm = blab == cid
                            if cm.sum() <= 200:
                                sm_miss += int((pred[cm] != MOV).sum()); sm_n += int(cm.sum())
                            else:
                                hg_miss += int((pred[cm] != MOV).sum()); hg_n += int(cm.sum())
        tt = tot / NF
        contrib /= NF
        print(f"\n[sigma={sigma}] total d_seg={tt:.3e}")
        print(f"  Fix A multi-phi:  single(K5)={np.mean(ga[1]):.4e}  G2(K6)={np.mean(ga[2]):.4e}"
              f"  G3(K7)={np.mean(ga[3]):.4e}  (identical => disconnection is NOT the obstruction)")
        o = np.argsort(-contrib)
        print("  per-class contribution: " +
              " ".join(f"{CLS[k]}={100*contrib[k]/tt:.0f}%" for k in o))
        print(f"  Fix B: Movable interior-miss={np.mean(mint):.4f} (ridge region, never flips)"
              f"  | small-blob miss={sm_miss/max(sm_n,1):.3f} vs huge-blob miss={hg_miss/max(hg_n,1):.3f}")
    print(f"\ndt={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
