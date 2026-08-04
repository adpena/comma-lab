"""sg3 part 2: (a) GT SEMANTIC islands (the charitable reading of 'each island'),
(b) the STATIC 2D artifact priced with its label payload, (c) L4 cell / L5 frame rungs.
All DESCRIPTION-ONLY address costs -> lower bounds -> can KILL, never confirm."""
import numpy as np, zlib, lzma, json
from scipy import ndimage
try:
    import brotli; HAVE_B = True
except Exception: HAVE_B = False

CACHE = "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache"
OUT = "/private/tmp/claude-501/-Users-adpena-Projects-pact/b8d27cf3-8307-4e05-856f-9423b281aa38/scratchpad/sg3"
NAMES = ["Road", "Lane", "Undriv", "Movable", "MyCar"]
DEN = 37_545_489


def cbytes(b):
    o = {"zlib": len(zlib.compress(b, 9)), "lzma": len(lzma.compress(b, preset=9 | lzma.PRESET_EXTREME))}
    if HAVE_B: o["brotli"] = len(brotli.compress(b, quality=11))
    k = min(o, key=o.get); return o[k], k


gt = np.ascontiguousarray(np.load(f"{CACHE}/gt_argmax_n600.npy", mmap_mode="r"))
pv = np.ascontiguousarray(np.load(f"{CACHE}/cx1_argmax_n600.npy", mmap_mode="r"))
F, H, W = gt.shape; SLOTS = F * H * W; Wb = 4 * DEN / SLOTS
flip = gt != pv; TOT = int(flip.sum())
res = {}

print("=== (a) GT SEMANTIC islands: connected components of each GT CLASS (a lane dash IS one) ===")
print(f"{'GT class':<10}{'islands':>10}{'/frame':>8}{'med px':>8}{'p90 px':>8}{'flips carried':>14}{'top1% share':>12}")
for c in [1, 3]:  # Lane, Movable -- the thin/small classes
    tot_is = 0; sizes = []; fl = []
    for f in range(F):
        lab, k = ndimage.label(gt[f] == c)
        if k == 0: continue
        tot_is += k
        s = ndimage.sum(np.ones_like(lab), lab, range(1, k + 1))
        ff = ndimage.sum(flip[f], lab, range(1, k + 1))
        sizes.extend(s.tolist()); fl.extend(ff.tolist())
    sizes = np.array(sizes); fl = np.array(fl)
    o = np.argsort(fl)[::-1]; top1 = fl[o][:max(1, len(fl) // 100)].sum() / max(fl.sum(), 1)
    print(f"{NAMES[c]:<10}{tot_is:>10,}{tot_is/F:>8.1f}{np.median(sizes):>8.0f}{np.percentile(sizes,90):>8.0f}"
          f"{fl.sum():>14,.0f}{top1*100:>11.1f}%")
    res[f"gt_islands_{NAMES[c]}"] = dict(n=tot_is, per_frame=tot_is / F, med=float(np.median(sizes)),
                                         p90=float(np.percentile(sizes, 90)), flips=float(fl.sum()),
                                         top1pct_share=float(top1))

print("\n=== (b) THE STATIC 2D ARTIFACT, priced with payload ===")
static = flip.any(axis=0); ever = int(static.sum())
b_map, k_map = cbytes(np.packbits(static.reshape(-1)).tobytes())
# modal GT class per ever-flipping pixel + how concentrated that pixel's GT class is over time
ys, xs = np.nonzero(static)
modal = np.zeros(ever, np.uint8); purity = np.zeros(ever, np.float32)
CH = 4096
for s in range(0, ever, CH):
    e = min(s + CH, ever); yy = ys[s:e]; xx = xs[s:e]
    col = gt[:, yy, xx]                     # (F, n)
    cnt = np.stack([(col == c).sum(0) for c in range(5)], 0)   # (5,n)
    modal[s:e] = cnt.argmax(0); purity[s:e] = cnt.max(0) / F
b_lab, k_lab = cbytes(modal.tobytes())
print(f"static set              : {ever:,} px")
print(f"  map (2D bitmap)       : {b_map:,} B  [{k_map}]")
print(f"  + modal GT label      : {b_lab:,} B  [{k_lab}]   -> TOTAL {b_map+b_lab:,} B")
print(f"  label purity: mean {purity.mean()*100:.1f}%  median {np.median(purity)*100:.1f}%  "
      f">90% pure: {(purity>0.9).mean()*100:.1f}% of pixels")
tot_static = b_map + b_lab
rate_S = tot_static * 25 / DEN
print(f"  rate cost             : {tot_static:,} B = {rate_S:.6f} S")
print(f"  break-even survival   : {rate_S/(100*TOT/SLOTS)*100:.3f}%   (fixes 0.4312 S at survival 1.0)")
res["static_artifact"] = dict(px=ever, map_B=b_map, label_B=b_lab, total_B=tot_static,
                              rate_S=rate_S, purity_mean=float(purity.mean()),
                              breakeven_survival=float(rate_S / (100 * TOT / SLOTS)))

print("\n=== (c) remaining rungs ===")
for cell in [8, 16, 32]:
    ch, cw = H // cell, W // cell
    g = flip.reshape(F, ch, cell, cw, cell).sum(axis=(2, 4))   # (F,ch,cw)
    stat = (g.sum(0) > 0)
    b, k = cbytes(np.packbits(stat.reshape(-1)).tobytes())
    occ = (g > 0)
    bb, kk = cbytes(np.packbits(occ.reshape(-1)).tobytes())
    print(f"  L4 cell {cell:>2}x{cell:<2} ({ch}x{cw}={ch*cw:,} cells): static occupancy {b:,} B "
          f"({stat.mean()*100:.1f}% cells live) | per-frame occupancy {bb:,} B ({bb/TOT:.4f} B/flip, {bb/TOT/Wb:.2f}x W)")
pf = flip.reshape(F, -1).sum(1)
print(f"  L5 frame: Gini {((2*np.sum(np.arange(1,F+1)*np.sort(pf))/(F*pf.sum()))-(F+1)/F):.4f} "
      f"| max/min {pf.max()/pf.min():.2f}x | top-10% frames carry {np.sort(pf)[::-1][:60].sum()/TOT*100:.1f}% (uniform 10%)")

json.dump(res, open(f"{OUT}/static_semantic.json", "w"), indent=1)
print(f"\n# wrote {OUT}/static_semantic.json")
