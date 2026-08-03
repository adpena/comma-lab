"""ddm_dd1 -- is LANE component-decomposable the way Movable is?  n600, scorer-free."""
import numpy as np, json
from scipy.ndimage import label as cclabel
Z=np.load("experiments/results/ot_offset_n600_modal_20260709/gt_n600_lstars_slim.npz")
L=Z["lstars"]; NF,H,W=L.shape
NAMES={0:"Road",1:"Lane",2:"Undrivable",3:"Movable",4:"MyCar"}
S4=np.array([[0,1,0],[1,1,1],[0,1,0]])
acc={c:{"n":0,"n64":0,"area":0,"per":0,"major":[],"minor":[],"area64":0,"per64":0} for c in NAMES}
for f in range(NF):
    lab=L[f]
    for c in NAMES:
        m=lab==c
        if not m.any(): continue
        cc,n=cclabel(m,structure=S4)
        if n==0: continue
        ar=np.bincount(cc.ravel())[1:]
        b=np.zeros((H,W),bool)
        d=m[:,1:]!=m[:,:-1]; b[:,1:]|=d; b[:,:-1]|=d
        d=m[1:,:]!=m[:-1,:]; b[1:,:]|=d; b[:-1,:]|=d
        bm=b&m
        pr=np.bincount(cc[bm].ravel(),minlength=n+1)[1:]
        a=acc[c]; a["n"]+=n; a["area"]+=int(ar.sum()); a["per"]+=int(pr.sum())
        big=ar>=64; a["n64"]+=int(big.sum()); a["area64"]+=int(ar[big].sum()); a["per64"]+=int(pr[big].sum())
        # second moments -> major/minor extent, for the >=64px components only
        idx=np.nonzero(big)[0]+1
        if len(idx):
            ys,xs=np.nonzero(cc>0); lb=cc[ys,xs]
            keep=np.isin(lb,idx)
            ys,xs,lb=ys[keep],xs[keep],lb[keep]
            for li in idx:
                s=lb==li
                if s.sum()<8: continue
                y=ys[s].astype(np.float64); x=xs[s].astype(np.float64)
                y-=y.mean(); x-=x.mean()
                cov=np.array([[ (y*y).mean(),(y*x).mean()],[(y*x).mean(),(x*x).mean()]])
                ev=np.linalg.eigvalsh(cov); ev=np.clip(ev,0,None)
                a["major"].append(2*np.sqrt(3*ev[1])); a["minor"].append(2*np.sqrt(3*ev[0]))
print(f"{'class':11s} {'comp':>7s} {'/frame':>7s} {'>=64':>6s} {'>=64/fr':>8s} {'area':>10s}"
      f" {'perim':>9s} {'medMajor':>9s} {'medMinor':>9s} {'aspect':>7s} {'A/P':>6s}")
out={}
for c,nm in NAMES.items():
    a=acc[c]; mj=np.array(a["major"]); mn=np.array(a["minor"])
    medmj=float(np.median(mj)) if len(mj) else float("nan")
    medmn=float(np.median(mn)) if len(mn) else float("nan")
    ap=a["area64"]/a["per64"] if a["per64"] else float("nan")
    print(f"{nm:11s} {a['n']:7,d} {a['n']/NF:7.2f} {a['n64']:6,d} {a['n64']/NF:8.2f} {a['area']:10,d}"
          f" {a['per']:9,d} {medmj:9.2f} {medmn:9.2f} {medmj/max(medmn,1e-9):7.2f} {ap:6.2f}")
    out[nm]={"components":a["n"],"per_frame":a["n"]/NF,"components_ge64":a["n64"],
             "ge64_per_frame":a["n64"]/NF,"area_px":a["area"],"perimeter_px":a["per"],
             "area_ge64":a["area64"],"perimeter_ge64":a["per64"],
             "median_major_px":medmj,"median_minor_px":medmn,
             "aspect_ratio":medmj/max(medmn,1e-9),"area_over_perimeter":ap}
print("\nA/P is the mean half-width: a THIN object has A/P ~ 0.5*width. Lane vs Movable is the test.")
print("\n--- CARRIER PRICE, 1 NORMAL DOF per component (the section 3 gauge) ---")
Wb=(100.0/(512*384*600))/(25.0/37_545_489)
for nm in ("Lane","Movable"):
    o=out[nm]
    for bits in (2,4):
        cost=o["components_ge64"]*bits/8.0
        print(f"  {nm:8s} {o['components_ge64']:6,d} comps >=64px @ {bits}b(1 DOF) = {cost:8,.0f} B"
              f"   [2 DOF would be {2*cost:8,.0f} B]")
json.dump({"arm":"ddm_dd1","axis":"[macOS-CPU advisory]","score_claim":False,
  "promotion_eligible":False,"rank_or_kill_eligible":False,"scorer_forwards_run":0,
  "substrate":"GT lstars n600; 4-conn components; no decode/vehicle/scorer",
  "n_frames":NF,"connectivity":"4","per_class":out},
  open(".omx/research/ddm_dd1_lane_component_census_n600.json","w"),indent=1)
print("\nwrote .omx/research/ddm_dd1_lane_component_census_n600.json")
