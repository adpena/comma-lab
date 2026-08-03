"""ddm_dd1 -- contour normal-coherence length on the GT separatrix, n600, scorer-free.
Gates section 3 (kappa*L<<4 gauge validity) and section 5 (segment length L -> byte cost).
No decode, no vehicle, no scorer forwards. [macOS-CPU advisory]; score_claim=false.
"""
import numpy as np, json, time
from scipy.ndimage import uniform_filter

Z = np.load("experiments/results/ot_offset_n600_modal_20260709/gt_n600_lstars_slim.npz")
L = Z["lstars"]                      # (600, 384, 512) int64
NF, H, W = L.shape
print(f"lstars {L.shape} classes={sorted(np.unique(L).tolist())}")

SM = 2                               # box half-width for the normal estimator
RADII = [1,2,3,4,6,8,12,16,24,32]
# unit offsets per radius (8 directions), integer-rounded
OFFS = {r: sorted({(int(round(r*np.sin(a))), int(round(r*np.cos(a))))
                   for a in np.linspace(0, 2*np.pi, 17)[:-1]} - {(0,0)}) for r in RADII}

sum_d = {r: 0.0 for r in RADII}; cnt = {r: 0 for r in RADII}
bnd_total = 0
t0 = time.time()
for f in range(NF):
    lab = L[f]
    # 4-conn boundary: pixel differs from right or down neighbour (either side)
    b = np.zeros((H, W), bool)
    d = lab[:, 1:] != lab[:, :-1]; b[:, 1:] |= d; b[:, :-1] |= d
    d = lab[1:, :] != lab[:-1, :]; b[1:, :] |= d; b[:-1, :] |= d
    bnd_total += int(b.sum())
    # normal from the gradient of the box-smoothed indicator of the pixel's OWN class
    gy = np.zeros((H, W), np.float32); gx = np.zeros((H, W), np.float32)
    for c in range(5):
        m = (lab == c)
        if not m.any(): continue
        s = uniform_filter(m.astype(np.float32), size=2*SM+1, mode="nearest")
        cy, cx = np.gradient(s)
        sel = m & b
        gy[sel] = -cy[sel]; gx[sel] = -cx[sel]   # outward normal
    mag = np.hypot(gy, gx)
    valid = b & (mag > 1e-6)
    th = np.zeros((H, W), np.float32)
    th[valid] = np.mod(np.arctan2(gy[valid], gx[valid]), np.pi)   # ORIENTATION in [0,pi)
    for r in RADII:
        for (dy, dx) in OFFS[r]:
            a = valid[max(0,-dy):H-max(0,dy), max(0,-dx):W-max(0,dx)]
            bsl = valid[max(0,dy):H-max(0,-dy), max(0,dx):W-max(0,-dx)]
            both = a & bsl
            n = int(both.sum())
            if not n: continue
            t1 = th[max(0,-dy):H-max(0,dy), max(0,-dx):W-max(0,dx)][both]
            t2 = th[max(0,dy):H-max(0,-dy), max(0,dx):W-max(0,-dx)][both]
            dd = np.abs(t1 - t2); dd = np.minimum(dd, np.pi - dd)   # fold to [0, pi/2]
            sum_d[r] += float(dd.sum()); cnt[r] += n
    if f % 150 == 0: print(f"  frame {f}/{NF}  t={time.time()-t0:.0f}s")

print(f"\nboundary px total = {bnd_total:,}   (sx1 separatrix_geometry: 2,551,382)"
      f"   ratio {bnd_total/2551382:.4f}")
print("\n r(px)   mean|dtheta|(deg)   pairs        kappa_eff(1/px)   kappa*L")
out = {}
for r in RADII:
    md = np.degrees(sum_d[r]/cnt[r])
    kap = np.radians(md)/r                       # turning per unit arc = effective curvature
    out[r] = {"mean_abs_dtheta_deg": md, "pairs": cnt[r], "kappa_eff_per_px": kap,
              "kappa_times_L": kap*r}
    print(f"{r:5d}   {md:12.3f}      {cnt[r]:12,d}   {kap:.6f}        {kap*r:.4f}")

# coherence length: where mean|dtheta| crosses thresholds (linear interp on r)
rs = np.array(RADII, float); ms = np.array([out[r]["mean_abs_dtheta_deg"] for r in RADII])
print()
Lstar = {}
for thr in (15.0, 22.5, 30.0, 45.0):
    if ms[0] >= thr: Lc = float(rs[0]*thr/ms[0])
    elif ms[-1] <= thr: Lc = float(rs[-1])
    else:
        i = int(np.searchsorted(ms, thr)); 
        Lc = float(rs[i-1] + (thr-ms[i-1])*(rs[i]-rs[i-1])/(ms[i]-ms[i-1]))
    Lstar[thr] = Lc
    print(f"  coherence length at mean|dtheta| = {thr:4.1f} deg :  L* = {Lc:6.2f} px")

# the gauge test: tangential/normal flip ratio = kappa*L/4
print("\n--- SECTION 3 GAUGE VALIDITY: tangential/normal = kappa*L/4 ---")
for r in RADII:
    print(f"  L={r:3d} px : kappa_eff={out[r]['kappa_eff_per_px']:.5f}/px"
          f"  -> tangential/normal = {out[r]['kappa_times_L']/4:.4f}")

json.dump({"arm":"ddm_dd1","axis":"[macOS-CPU advisory]","score_claim":False,
           "promotion_eligible":False,"rank_or_kill_eligible":False,"scorer_forwards_run":0,
           "substrate":"GT lstars n600 (gt_n600_lstars_slim.npz); no decode/vehicle/scorer",
           "n_frames":NF,"grid":[W,H],"smoothing_halfwidth_px":SM,
           "boundary_px_total":bnd_total,"sx1_boundary_px_total":2551382,
           "orientation_coherence":out,"coherence_length_px_by_threshold_deg":Lstar,
           "note":"Euclidean separation approximates arc length; cross-contour pairs ADD "
                  "decoherence, so L* is a LOWER bound (conservative: true usable L is >=)."},
          open(".omx/research/ddm_dd1_contour_coherence_n600.json","w"), indent=1)
print("\nwrote .omx/research/ddm_dd1_contour_coherence_n600.json")
