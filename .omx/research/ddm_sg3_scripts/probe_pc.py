"""sg3 probe P-C -- COMPONENT-INDEX BULK-VERB ADDRESS.

ROLE UNDER TEST (the cg1 reconciliation): this tests the **ADDRESS-role** -- can a per-component
index + a tiny geometric verb ADDRESS enough flip mass to pay? It does NOT test the
**ACTUATOR-role** (cg1's 0-for-11 on whole-class/per-side SCALAR FORCES in the training loop).
Same SHAPE (one aggregate handle per region), different ROLE (decode-time geometric correction
vs training-time scalar force). If this fails too, the 0/11 law EXTENDS to address-role.

LEGALITY: the encoder picks the best verb per component USING GT and SHIPS it as counted bits.
That is ordinary encoding, not an oracle -- the DECODER never sees GT, it just applies the verb.
(An oracle would be the decoder needing GT.) Still DESCRIPTION-ONLY: no realizer is priced.

VERB: dilate component by d in {0,1,2,3} (2 bits). Because components of ALL FIVE classes are
enumerated, dilation alone covers BOTH boundary directions (dilating B into A == eroding A).
"""
import numpy as np, json
from scipy import ndimage

CACHE = "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache"
OUT = "/private/tmp/claude-501/-Users-adpena-Projects-pact/b8d27cf3-8307-4e05-856f-9423b281aa38/scratchpad/sg3"
DEN = 37_545_489
NAMES = ["Road", "Lane", "Undriv", "Movable", "MyCar"]

gt = np.ascontiguousarray(np.load(f"{CACHE}/gt_argmax_n600.npy", mmap_mode="r"))
pv = np.ascontiguousarray(np.load(f"{CACHE}/cx1_argmax_n600.npy", mmap_mode="r"))
F, H, W = gt.shape; SLOTS = F * H * W; Wb = 4 * DEN / SLOTS
TOT = int((gt != pv).sum())
st = ndimage.generate_binary_structure(2, 2)
DELTAS = [1, 2, 3]

tot_net = 0; tot_comp = 0; treated = 0
per_class_net = np.zeros(5, np.int64); per_class_comp = np.zeros(5, np.int64)
best_hist = {0: 0, 1: 0, 2: 0, 3: 0}

for f in range(F):
    G = gt[f]; P = pv[f]
    for c in range(5):
        m = P == c
        if not m.any(): continue
        lab, k = ndimage.label(m, structure=st)
        if k == 0: continue
        objs = ndimage.find_objects(lab)
        for idx, sl in enumerate(objs):
            tot_comp += 1; per_class_comp[c] += 1
            pad = 4
            y0 = max(0, sl[0].start - pad); y1 = min(H, sl[0].stop + pad)
            x0 = max(0, sl[1].start - pad); x1 = min(W, sl[1].stop + pad)
            sub = (lab[y0:y1, x0:x1] == idx + 1)
            Gs = G[y0:y1, x0:x1]; Ps = P[y0:y1, x0:x1]
            best = 0; best_net = 0
            for d in DELTAS:
                ring = ndimage.binary_dilation(sub, st, d) & ~sub
                if not ring.any(): continue
                gr = Gs[ring]; pr = Ps[ring]
                fixed = int(((gr == c) & (pr != c)).sum())     # we were wrong; dilating fixes
                broken = int(((gr != c) & (pr == gr)).sum())   # we were right; dilating breaks
                net = fixed - broken
                if net > best_net: best_net = net; best = d
            best_hist[best] += 1
            if best_net > 0:
                tot_net += best_net; per_class_net[c] += best_net; treated += 1
    if f % 100 == 0: print(f"  ..frame {f}", flush=True)

# bytes: index (log2 comps/frame) + verb (2 bits) for every enumerated component
comps_per_frame = tot_comp / F
idx_bits = np.log2(max(comps_per_frame, 2))
addr_B = tot_comp * (idx_bits + 2) / 8
# cheaper variant: only ship treated components -> index + 2-bit verb, plus a presence bitmap
addr_B_sparse = treated * (idx_bits + 2) / 8 + tot_comp / 8

segS = 100 * tot_net / SLOTS
rateS_all = addr_B * 25 / DEN
rateS_sp = addr_B_sparse * 25 / DEN
print("\n=== PROBE P-C RESULT (ADDRESS-role, DESCRIPTION-ONLY) ===")
print(f"components enumerated      : {tot_comp:,} ({comps_per_frame:.1f}/frame) -> index {idx_bits:.2f} bits")
print(f"components with net>0      : {treated:,} = {treated/tot_comp*100:.1f}%")
print(f"best-delta histogram       : {best_hist}")
print(f"NET flips fixed (ceiling)  : {tot_net:,} = {tot_net/TOT*100:.2f}% of all {TOT:,} flips")
print(f"  seg recovered            : {segS:.5f} S  (of the 0.43118 S seg leg)")
print(f"address bytes (all comps)  : {addr_B:,.0f} B = {rateS_all:.5f} S")
print(f"address bytes (sparse)     : {addr_B_sparse:,.0f} B = {rateS_sp:.5f} S")
print(f"NET S (sparse, FREE realiz): {rateS_sp - segS:+.5f} S  "
      f"{'WIN' if rateS_sp < segS else 'LOSS'}")
print(f"B/flip                     : {addr_B_sparse/max(tot_net,1):.4f} vs W={Wb:.4f} "
      f"= {addr_B_sparse/max(tot_net,1)/Wb:.2f}x")
print("\nper-class net flips fixed:")
for c in range(5):
    print(f"  {NAMES[c]:<10} {per_class_net[c]:>9,}  from {per_class_comp[c]:>8,} components")

json.dump(dict(components=tot_comp, comps_per_frame=comps_per_frame, index_bits=float(idx_bits),
               treated=treated, net_flips=int(tot_net), total_flips=TOT,
               frac_flips=tot_net / TOT, seg_S=segS, addr_B_all=addr_B,
               addr_B_sparse=addr_B_sparse, rate_S_sparse=rateS_sp,
               net_S_sparse=rateS_sp - segS, best_hist={str(k): v for k, v in best_hist.items()},
               per_class_net=per_class_net.tolist()), open(f"{OUT}/probe_pc.json", "w"), indent=1)
print(f"\n# wrote {OUT}/probe_pc.json")
