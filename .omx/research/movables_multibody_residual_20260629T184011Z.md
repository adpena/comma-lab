# GAP1 — the MOVABLES (class-3) multi-body residual: measured + economics (F3 arm)

- **UTC** 2026-06-29T18:40:11Z · **authority** frozen-CPU-torch SegNet argmax (ADVISORY;
  NOT a contest score) · **pointer UNMOVED 0.19110** · **$0** (numpy/scipy/CPU SegNet, no heavy GPU)
- **Tool** `tools/measure_movables_multibody_residual.py` (review-gate ×2; self-detects class-3 by signature)
- **Inputs** `gt_n96.npz` (first ~10s, dense lead vehicle) + `gt_strided_n200.npz` (full 60s drive);
  f0 segs recomputed via canonical `tac.boundary_math.seg_core` (f1-recompute is **bit-exact** vs cache).
- **Context** DAG FEED-iw/ja (movables = the 2nd residual term after the ego-warp handles Road/hood/sky/lane);
  reconciles with FEED-jd ("F3 movables last arm running"; "movables ~750 B"; sub-0.15 ⟺ d_seg ≤ 1.23e-3).

## VERDICT (the good news, two roles, opposite answers)

The movables term is **small, per-object-LOW-RANK, and cheap** — but the right action depends on the ROLE:

1. **Multi-body as a WARP-PREDICTOR (reduce d_seg by ego-warping f0 movables + per-object rigid correction): ACCEPT, do NOT build.**
   The per-object-rigid reproduction floor is **V3 = 0.00082 d_seg** (within-pair, contest cadence) ≈ the FEED-ja
   current-witness residual (~0.0008). The whole "multi-body prize" over naive temporal-copy is only
   **V1−V3 = 0.00027 d_seg = 0.027 S MAX** (oracle motion), and the current witness already sits at the floor.
   Below 0.00082 is irreducible PLACEMENT error; per-object SLAM cannot push lower. Dominated by the all-class
   directional-basis lever (−48% of d_seg). No d_seg left to harvest as a predictor.

2. **Multi-body as a cheap STORAGE codec (templates + low-rank trajectories to DIRECTLY place movables): YES — this is the F3 ~750 B line, validated.**
   Storing the class-3 region (not predicting it) drives movables d_seg BELOW the 0.00082 warp-floor toward the
   partition floor (~0). One-template + per-frame pose already reproduces the **median object at 0.00027 d_seg**;
   periodic re-templating + the huge rate slack (FEED-jd: rate term only 0.0021, ~112 KB budget) drives it lower.
   Cost is tiny: **K=50 → 0.9 KB, K=150 → 2.7 KB** (low-rank confirmed), vs a **40 KB** break-even.

**Why role-2 matters at the threshold:** FEED-jd's pass-line is total d_seg ≤ 1.23e-3 for sub-0.15. The warp-floor
0.00082 alone is **67% of that entire budget** — so movables must NOT be warp-predicted (they'd nearly bust the
budget); they must be **stored cheaply via the multi-body decomposition**. The residual is small *because* the
witness's base partition + temporal persistence already collapse the 1.6% movable AREA → ~0.001 d_seg (94% handled);
the remaining bit is the rate-cheap store target, not a SLAM problem.

## MEASUREMENT (within-pair f0→f1, the TRUE contest cadence)

| variant | what | n96 (first 10s) | strided n200 (full drive) |
|---|---|---:|---:|
| V0 | no movable model (predict ∅) = class-3 area | 0.01557 | 0.01238 |
| V1 | temporal persistence (copy f0 mask) | 0.001106 | 0.001089 |
| V2 | per-object translation (oracle dy,dx) | 0.000910 | 0.000926 |
| **V3** | **per-object translate+scale (rigid floor)** | **0.000839** | **0.000824** |
| — | multi-body PRIZE (V1−V3) | **0.000268** | **0.000266** |

- **Remarkably stable across the whole drive** (prize 0.00027 both; floor 0.00082 both) → not a first-10s artifact.
- **Births/deaths negligible**: f1 movable px with no f0 correspondence = **2e-5** (median 0). Objects persist; they
  don't pop into existence → warp-copy works AND tracking-based storage is viable.
- **Objects/frame** mean 3.0 (n96) / 3.5 (n200), max 8. Median object is **tiny (62 px)**; area dominated by 1-2 lead
  vehicles. The d_seg residual is dominated by those few large objects' boundaries.
- **Margin decomposition (the irreducibility proof)**: class-3 **boundary** pixels have **median margin 0.29** (73%
  flip-prone, <0.5), while **interiors** have **median margin 5.99** (1% flip-prone). The residual is a thin BOUNDARY
  annulus where SegNet's OWN decision is a coin-flip — interiors are rock-solid. No motion model fixes a pixel where
  the oracle scorer is itself uncertain.
- **Per-object LOW-RANK (yes)**: tracks persist (max 96 frames = whole sequence; births 2e-5); centroid trajectories
  fit **deg-1 RMS ~3-6 px, deg-2 ~2.5 px** → a handful of polynomial coeffs per object. Consistent with openpilot's own
  compact lead model (1×58) and the multibody-SfM literature (per-object state = location/direction/speed, low-dim).
- **One-template STORE reproduction** (translate+scale, per-frame pose): **median 0.00027** d_seg (well below the
  0.00082 warp-floor), mean 0.00127 (the long morphing lead vehicle drifts → needs periodic re-template). Store is
  tunable toward ~0 with keyframes; warp is hard-floored.

## ECONOMICS (the break-even)

S = 100·d_seg + √(10·d_pose) + 25·bytes/37,545,489 ⇒ **1 S-pt of rate = 1,501,820 bytes**; d_seg weight = 100.

- Removing the full warp-floor (0.00082 d_seg) is worth **0.082 S** → up to **123 KB** of rate budget.
- The warp-only "prize" (0.00027) is worth **0.027 S** → 40 KB break-even — but it's already captured (ACCEPT).
- The **store** path: ~0.9-2.7 KB (multi-body templates + low-rank trajectories) costs ~0.0006-0.0018 S of rate to
  remove up to 0.082 S of movables d_seg. **Overwhelmingly positive given the rate slack** — and it's the only way
  under the 1.23e-3 threshold, since warp can't beat 0.00082.

## OSS / online grounding (part c)

- **Multibody SfM** (Vidal-Ma multibody fundamental matrix; Ozden-Schindler-Van Gool "MFSfM in Practice"): the HARD
  part is motion SEGMENTATION (how many bodies, which features). **We get that FREE** — SegNet already segments class-3.
- **openpilot driving model**: lead-vehicle output is a **compact 1×58** per-lead state (relative pos/vel) — the
  production stack itself models movables as low-dim per-object state. Corroborates "per-object low-rank, cheap."
- **Dynamic-scene NeRF** (D-NeRF / NSFF / dynamic 3D-Gaussians): the field treats dynamic content as the EXPENSIVE
  residual (per-object deformation fields). For a ≤0.027 S warp-prize that is overkill; the cheap low-dim rigid
  store is the right tool, matching the measurement.

Sources: Two-View Multibody SfM (Springer IJCV); Multibody SfM in Practice (ResearchGate); commaai/openpilot
modeld README; comma.ai openpilot-in-2021 blog.

## NEXT (if/when movables become binding)

The all-class directional basis is the d_seg headline; movables are second-order until the directional lever lands.
WHEN it does and movables become 50-67% of the residual budget:
1. Implement movables as a **stored multi-body codec** (K templates + deg-2 per-object trajectories + periodic
   re-template), NOT a warp predictor. ~0.9-2.7 KB counted (rule-118 compliant: video-derived = counted; the
   rasterizer/warp dispatch = free in inflate.py).
2. **Open measurement** (this memo's one gap): the achievable store-d_seg with a fixed keyframe rate (the cheap
   one-template number is median 0.00027 / mean 0.00127; per-frame-pose recovers V3; the keyframe-budget curve is the
   next $0 measurement — extend the tool).

## means ≠ ends

All numbers are ADVISORY structural bounds on the SegNet argmax (pointer 0.19110 unmoved). This unit BOUNDS the 2nd
residual term of the v2 witness and decides its treatment (store-not-predict). The exact row is moved only by a
byte-closed witness eval. Bank it.
