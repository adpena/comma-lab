# v8 Road↔Lane ξ-EGO-COMPENSATED temporal rate — MEASURED (real machinery, $0, read-only)

**Date:** 2026-07-09 · **Scope:** re-measure the Road↔Lane temporal stream with EGO-COMPENSATION
(ξ-transport predictive coding) on top of the just-landed camera-frame baseline
(`v8_roadlane_geometric_rate_20260709.md`, LBND2 temporal-delta, 0.0275 S). `[no-triality-yet]` ·
pointer **0.19110 UNMOVED** · #205 untouched · MPS/GPU untouched (pure numpy on the label cache).

**HEADLINE (honest, MEASURED):** ξ-ego-transport does **NOT** collapse the lane stream — it
**enlarges** it. The honest v8 lane number **stays 0.0275 S** (LBND2 temporal delta). The horizon's
14.6× ego-freeze does **NOT** transfer to lanes, and the measurement explains *why*: the lane coeffs
are **already fit in the ego-ground frame**, so there is no camera-frame ego component left for ξ to
remove — the residual frame-to-frame motion is genuine curvature evolution + multi-instance fit
churn, which a rigid ego-advection model actively **mis**-predicts.

## STORES CONSULTED (recall, not re-derive)
- `.omx/research/v8_roadlane_geometric_rate_20260709.md` — the baseline (LBND2 coherent-slot, 0.0275 S;
  its own honest finding: "lanes are NOT frozen — every coeff moves 55–82%; the 7.4× is PRIMARILY
  PARSIMONY, only modestly helped by temporal delta 2.6×"). **This measurement CONFIRMS and MECHANISM-
  EXPLAINS that finding.**
- DAG **FEED-v8-realmachinery** (the horizon precedent: 14.6× from ego-pitch freezing the image-row
  polynomial's intercept). The KEY DISANALOGY surfaced here: the horizon poly lives in **image rows**
  where inter-frame motion IS pure ego pitch (one intercept); the lane poly lives in the **ground
  frame** (lateral-vs-forward) where the ego component was already removed by the IPM at fit time.
- The pre-built ξ machinery (design `unified_xi_design_and_adversarial_review_20260702.md` §2, the
  P-frame reframe): `tac.boundary_math.ego_xi_trajectory` (`XiEgoTrajectory`, `advect_centerline_coeffs`
  = exact closed-form Taylor-shift + dy + dpsi, `LaneOptimalEgoEstimator`, `PoseTargetEgoEstimator`)
  + `analytic_lane_render_band.{serialize,deserialize}_lane_band_rd3` (LBND3 = ego-predictive P-frame,
  bit-exact, a STRICT generalization of LBND2: ξ=0 ⇒ innovation == LBND2 temporal delta).
- **Real ξ source:** `experiments/results/mlx_fleet_gt_cache/gt_n600.npz['gt_poses']` (600,6) — the
  REAL stored frozen-PoseNet ego readout (the pose section; dual-use, ZERO synthetic proxy). Also the
  **lane-optimal** ξ (per-pair LSQ ds/dy/dψ that best predicts the lanes — the ACHIEVABLE FLOOR).

## MEASUREMENT (all 600 pairs, real `gt_n600.npz['lstars']`, comma10k order Road0/Lane1)

Fits are the SAME `build_lane_band_pairs_from_lstars` the baseline used (2967 lines, 4.95/frame, K=6
sort-slots). Baselines reproduced: LBND2 coherent-slot **41,303 B → S=0.02750** (matches the baseline
memo's headline S=0.0275 exactly), LBND2 sort 41,526 B → 0.02765.

### Arm 1 — ξ-compensated LBND3 (full blob incl. ego stream), brotli q11, bit-exact roundtrip ✓

| coder / ξ | bytes | S | ego-stream ~B | vs LBND2 coherent |
|---|---|---|---|---|
| **LBND2 temporal delta (baseline)** | **41,303** | **0.02750** | — | 1.0× |
| LBND3 ξ = PoseNet-affine (real gt_poses, up-to-affine) | 44,908 | 0.02990 | ~2,092 | **0.92× (WORSE)** |
| LBND3 ξ = PoseNet-geometry (real gt_poses, fixed scale) | 46,178 | 0.03075 | ~2,212 | 0.89× (WORSE) |
| LBND3 ξ = lane-optimal (achievable-floor predictor) | 47,453 | 0.03160 | ~2,744 | 0.87× (WORSE) |

Every ξ arm is LARGER. All roundtrip **bit-exact** (LBND3 dequantized geometry == LBND2 dequantized,
same Q grid — verified).

### Arm 1b — isolate the LANE payload (ego overhead REMOVED — ξ's fairest shot)

To rule out "the ego bytes are the whole regression," compare **innovation-stream-only** brotli vs
**temporal-delta-only** brotli (both zigzag-uint32, ego block excluded entirely):

| stream (lane payload only, no ego) | brotli B | Σ&#124;residual&#124;₁ | vs LBND2 |
|---|---|---|---|
| **LBND2 temporal delta** | **41,085** | **7,584,060** | 1.0× |
| ξ lane-optimal, no-forward (dy+yaw) | 42,195 | 7,727,778 | 0.97× (WORSE) |
| ξ PoseNet-affine | 42,017 | 8,252,100 | 0.98× (WORSE) |
| ξ lane-optimal, smooth-7 | 43,136 | 8,670,182 | 0.95× (WORSE) |
| ξ lane-optimal (fwd+dy+yaw) | 43,964 | 9,983,228 | 0.93× (WORSE) |

**Decisive:** the plain temporal delta already has the smallest L1 AND smallest brotli. Ego-advection
*increases* the residual magnitude — the identity predictor (Q[t]−Q[t−1]) beats every ego-rigid
predictor, including the lane-optimal one fit to minimize innovation and the smoothed variant.

### Per-coeff nonzero-fraction (the apples-to-apples collapse test the operator asked for)

| coeff | LBND2 nz-frac | ξ-lane | ξ-geo | ξ-aff | verdict |
|---|---|---|---|---|---|
| c3 (cubic) | 0.966 | 0.966 | 0.966 | 0.966 | unchanged |
| c2 (quad)  | 0.988 | 0.988 | 0.989 | 0.991 | unchanged / worse |
| c1 (heading) | 0.987 | 0.992 | 0.991 | 0.988 | **worse** |
| c0 (offset) | 0.947 | 0.984 | 0.996 | 0.963 | **worse** |
| hw1/hw0 | 0.987/0.988 | identical | identical | identical | identity-advected (by design) |

No coeff collapses; offset/heading get *denser* under ego-advection (the ego forward-shift over-
predicts the bulk c0 motion the steered ego actually cancels — the exact failure the design's `dy`
DOF was meant to fix, but even with dy fit it does not help because the motion is not ego-rigid).

## Arm 2 — WATERFILLED-LOSSY RD (operator addendum: "loss we know how to optimize and waterfill")

Per-coeff-class quantization sweep on `derive_rd_base_steps` (classes: curvature{c3,c2}, heading{c1},
offset{c0}, halfwidth{hw}), ξ = lane-optimal, bit-exact decode → `rasterize_lane_band` → distortion =
`100 × mean(recon_band XOR gt_lane)` over 600×384×512 (an **upper bound** on the Road↔Lane d_seg
contribution). Joint waterfill trace:

| config | bytes | S_rate | S_dist (UB) | S_total |
|---|---|---|---|---|
| **finest q (lossless)** | 47,453 | 0.0316 | 0.5204 | **0.5520 ← KNEE** |
| all-classes ×2 | 45,534 | 0.0303 | 0.5317 | 0.5621 |
| all-classes ×4 | 40,148 | 0.0267 | 0.5650 | 0.5917 |
| all-classes ×8 | 37,520 | 0.0250 | 0.6447 | 0.6696 |
| all-classes ×16 | 34,954 | 0.0233 | 0.7958 | 0.8191 |
| sensitivity-weighted (curv×8,head×2,off×1,hw×8) | 42,218 | 0.0281 | 0.5352 | 0.5633 |

**Waterfill finding:** the RD knee sits at the **finest (lossless) point** — every quantization step
*raises* S_total. Per-class marginals confirm the KKT ranking (offset is the most distortion-sensitive
class: ×16 offset → S_dist 0.52→0.77; curvature/halfwidth are cheapest but still net-negative). The
distortion is **dominated by a fixed ~0.52 S band-geometry floor** (the filled centerline±halfwidth
band vs the thin painted lane markings — the same 27.5%-coverage / band-width artifact the baseline
flagged), NOT by quantization, so this upper-bound metric gives quantization *no* headroom. Waterfill
does **not** beat lossless. The honest rate axis can be pushed to 0.0233 S (×16) but only by paying
real quantization distortion on top of the floor — a bad trade.

## THE S-NUMBER TABLE (Road↔Lane, temporal axis)

| representation | S | vs bitmap (0.2040) | vs camera-frame baseline (0.0275) |
|---|---|---|---|
| bitmap (de-shared) | 0.2040 | 1.0× | — |
| **LBND2 temporal delta (camera-frame baseline, HELD)** | **0.0275** | **7.4×** | 1.0× |
| LBND3 ξ-ego-compensated (best: PoseNet-affine, incl. ego) | 0.0299 | 6.8× | **0.92× (WORSE)** |
| LBND3 ξ, lane payload only (ego free/dual-use) | ~0.0280 | 7.3× | ~0.99× (still worse) |
| waterfill knee | = lossless | — | = 0.0275 |

## Adversarial self-review (before commit)
1. **Is the ξ real?** YES — `gt_n600.npz['gt_poses']`, the actual frozen-PoseNet ego readout stored
   in the pose section (dual-use). Also tested the lane-OPTIMAL ξ (the achievable floor, fit to the
   lanes themselves) and a smoothed variant — ALL worse. So the negative is not an artifact of a bad
   ξ estimate; the best-possible ego predictor still loses to identity.
2. **Transport exact?** YES — `advect_centerline_coeffs` is the exact closed-form SE(2) coeff transport
   (Pascal Taylor-shift + dy + dψ); LBND3 is the same fp64 advect both encode/decode (bit-identical).
3. **Roundtrip bit-exact?** YES — LBND3 dequantized geometry == LBND2 dequantized (same Q grid),
   verified for all 3 ξ arms. Waterfill configs decode bit-exact (deserialize before rasterize).
4. **No double-count of ξ?** Handled two ways: (a) full LBND3 blob INCLUDING the ~2.1–2.7 KB ego
   stream (conservative — no dual-use credit) is worse; (b) lane-payload-ONLY (ego excluded, the
   dual-use best case) is ALSO worse. The conclusion is robust to the accounting choice.
5. **Coverage caveat unchanged?** YES — same 72.5% dominant-boundary coverage, `residual_sidecar_owed`
   unchanged. This measurement targets the TEMPORAL axis only; it did not improve coverage.
6. **Numbers MEASURED not guessed?** All from real n600 argmax through the real fit + real LBND2/LBND3
   coders + real brotli q11. Verifier: `serialize_lane_band_rd3(pairs, cfg, xi_traj)` with
   `xi_traj = PoseTargetEgoEstimator(calib='affine_to_lane').estimate(gt_poses, ref=lane_optimal)`.

## Consequence for v8 + WHY (the mechanism, the durable insight)
The operator's intuition — "lanes painted on the road are quasi-static in the ground frame; camera-
frame coeff motion is induced by the ego screw ξ; code the delta AFTER ξ-transport and it collapses
toward the horizon's 14.6×" — is **geometrically sound for an IMAGE-frame representation but already
SATISFIED by our ground-frame fit.** The baseline lane coder does NOT store camera-frame coeffs; it
stores **ground-frame** coeffs (`lateral = poly(forward)` via the fixed IPM `image_to_ground`). The
IPM already quotients out the dominant ego-forward-translation component at fit time. What remains in
the temporal delta is the **irreducible** part: real lane-curvature evolution as the road bends, plus
multi-instance slot churn and per-frame fit noise — none of which is a rigid ego transport, so an
ego-advection predictor (however well-estimated) can only *add* prediction error. The horizon won its
14.6× precisely because its polynomial is NOT ground-canonicalized (it lives in image rows), so its
inter-frame motion IS a single removable ego-pitch intercept.

**Net:** the honest v8 Road↔Lane temporal number is the camera-frame-baseline **0.0275 S** (its
7.4× parsimony win over the bitmap stands). The ξ-transport lever is a **MEASURED NO-GO for the lane
rate axis** (implementation-complete: real ξ, exact transport, best predictor, both accounting modes;
paradigm intact — ξ remains decisive for POSE and for the image-frame horizon). For the sibling
roll-up (`v8_increment1_design_draft`): **use 0.0275 S for Road↔Lane; ξ-ego-compensation does not
improve it — do NOT project a horizon-class transfer.**

## FORMALIZATION_PENDING (clean law emerged — the NEGATION of the hoped-for collapse)
Proposed canonical equation `lane_groundframe_xi_transport_no_collapse_v1`:
*For a lane centerline already parametrized in the ego-ground frame (`lateral=poly(forward)` via a
fixed IPM), ego-advected predictive coding (LBND3) yields innovation ≥ the LBND2 identity temporal
delta: `‖innov_ξ‖₁ ≥ ‖ΔQ‖₁` for all ξ (measured: 7.73M–9.98M vs 7.58M @ n600). Corollary: the
horizon's ego-freeze ratio does not transfer to any representation whose coordinate chart has already
absorbed the dominant ego DOF.* Not registered here (measurement subagent scope; would half-wire the
producer/consumer contract) — flagged for a canonical-equations landing with the EmpiricalAnchor
(`experiments/results/mlx_fleet_gt_cache/gt_n600.npz` + this memo's tables).

`[no-triality-yet]` · pointer **0.19110 UNMOVED** · #205 untouched.
