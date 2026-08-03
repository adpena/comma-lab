"""ddm_dd1 -- per-EDGE x per-FRAME flip concentration on the LIVE cx1 vehicle.
Consumes ddm_pu2's already-materialised per-pair 5x5 directed confusion tensor.
No scorer pass, no decode. [macOS-CPU advisory]; score_claim=false.
Per m91: decompose per EDGE, never per class.  as1 owns asymmetry; hs1 owns cell Gini -- cited, not redone.
"""
import json, numpy as np
SRC = "/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/per_pair_directed.jsonl"
NAMES = {0: "Road", 1: "Lane", 2: "Undriv", 3: "Movable", 4: "MyCar"}
PX = 512 * 384 * 600
S_FLIP = 100.0 / PX
S_BYTE = 25.0 / 37_545_489
W = S_FLIP / S_BYTE
GAP = 0.7910689 - 0.172141

C = np.array([json.loads(l)["C"] for l in open(SRC)], dtype=np.int64)   # (600,5,5)
flips_row = np.array([json.loads(l)["flips"] for l in open(SRC)], dtype=np.int64)
NF = C.shape[0]
offdiag = C.sum(axis=(1, 2)) - np.trace(C, axis1=1, axis2=2)
assert (offdiag == flips_row).all(), "off-diagonal must reproduce the stored flip count"
assert (C.sum(axis=(1, 2)) == 512 * 384).all(), "each frame must sum to the full grid"
TOT = int(flips_row.sum())
print(f"n={NF} frames; total flips {TOT:,}  (charter cx1: 508,639)  ratio {TOT/508639:.6f}")
print(f"implied d_seg = {TOT/PX:.9f}  (charter 0.004311790)")

def gini(x):
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2*np.arange(1, n+1) - n - 1).dot(x) / (n * x.sum())) if x.sum() else 0.0

rows = []
for i in range(5):
    for j in range(i+1, 5):
        per_f = C[:, i, j] + C[:, j, i]
        tot = int(per_f.sum())
        if tot == 0:
            continue
        srt = np.sort(per_f)[::-1]
        cum = np.cumsum(srt) / tot
        rows.append({
            "edge": f"{NAMES[i]}<->{NAMES[j]}", "flips": tot, "share": tot/TOT,
            "gini_across_frames": gini(per_f),
            "top10_frames_capture": float(cum[9]), "top60_frames_capture": float(cum[59]),
            "top120_frames_capture": float(cum[119]),
            "frames_for_50pct": int(np.searchsorted(cum, 0.50) + 1),
            "frames_for_90pct": int(np.searchsorted(cum, 0.90) + 1),
            "max_frame_flips": int(srt[0]), "median_frame_flips": float(np.median(per_f)),
            "directed_ij": int(C[:, i, j].sum()), "directed_ji": int(C[:, j, i].sum()),
        })
rows.sort(key=lambda r: -r["flips"])
print(f"\n{'edge':22s}{'flips':>9s}{'share':>8s}{'gini':>7s}{'f@50%':>7s}{'f@90%':>7s}"
      f"{'top60':>8s}{'maxfr':>7s}{'medfr':>7s}")
for r in rows:
    print(f"{r['edge']:22s}{r['flips']:9,d}{r['share']*100:7.2f}%{r['gini_across_frames']:7.3f}"
          f"{r['frames_for_50pct']:7d}{r['frames_for_90pct']:7d}{r['top60_frames_capture']*100:7.1f}%"
          f"{r['max_frame_flips']:7d}{r['median_frame_flips']:7.0f}")

print("\n--- CROSS-VEHICLE CHECK: pc2 (tb1) shares vs cx1 (LIVE) ---")
PC2 = {"Road<->Lane": .4923, "Road<->Undriv": .1626, "Undriv<->Movable": .1185,
       "Road<->Movable": .1147, "Road<->MyCar": .1089}
for r in rows:
    if r["edge"] in PC2:
        p = PC2[r["edge"]]
        print(f"  {r['edge']:22s} pc2/tb1 {p*100:6.2f}%   cx1/LIVE {r['share']*100:6.2f}%"
              f"   ratio {r['share']/p:5.3f}")

print("\n--- LANE CARRIER RE-PRICED ON THE LIVE VEHICLE (corrects my own section 3.2) ---")
rl = next(r for r in rows if r["edge"] == "Road<->Lane")
for lbl, n in (("pc2/tb1 share 49.23%", int(0.4923*TOT)), (f"cx1 MEASURED {rl['share']*100:.2f}%", rl["flips"])):
    dS = n * S_FLIP
    print(f"  {lbl:26s} {n:8,d} flips -> {dS:.6f} S = {dS/GAP*100:5.2f}% of live gap"
          f" ; 703 B carrier = {703/n:.5f} B/flip = {W/(703/n):6.1f}x W")

json.dump({"arm": "ddm_dd1", "axis": "[macOS-CPU advisory]", "score_claim": False,
           "promotion_eligible": False, "rank_or_kill_eligible": False, "scorer_forwards_run": 0,
           "substrate": "ddm_pu2 per_pair_directed.jsonl (cx1 LIVE vehicle n600); no scorer pass",
           "n_frames": NF, "total_flips": TOT, "implied_d_seg": TOT/PX,
           "per_edge_per_frame": rows,
           "note": "as1 owns asymmetry composition; hs1 owns cell-level Gini. Frame-level "
                   "concentration is the axis measured here."},
          open(".omx/research/ddm_dd1_edge_frame_concentration_n600.json", "w"), indent=1)
print("\nwrote .omx/research/ddm_dd1_edge_frame_concentration_n600.json")
