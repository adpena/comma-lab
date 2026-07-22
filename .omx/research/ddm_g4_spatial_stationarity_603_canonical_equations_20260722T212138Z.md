---
task: 603
feeds_task: 613
research_only: true
triality_leg: equations
evidence_axis: "[macOS-CPU frozen-scorer advisory]"
score_claim: false
main_landing_review_required: true
---

# DDM G4 spatial-stationarity canonical equations

Let `N=600`, `H=384`, `W=512`, `P={0,...,H W-1}`, and let
`c_t(p)` and `y_t(p)` be the settled v12 predicted and target SegNet argmax
cells.  The exact transition event and per-pixel frequency are

```text
e_t(p,a,b) = 1[c_t(p)=a, y_t(p)=b, a!=b]
k(p,a,b)   = sum_t e_t(p,a,b)
F(p)       = sum_(t,a,b) e_t(p,a,b)
M          = sum_p F(p) = 4,011,236.
```

For the `q` fraction of image pixels with largest `F(p)`, the concentration
law is

```text
C_q = sum_(p in Top_q(F)) F(p) / M.
```

Measured anchors are `C_0.01=0.212048106868`,
`C_0.05=0.687228076334`, and `C_0.10=0.898731214020`.

## Disjoint stationarity partition

Every flip event is classified hierarchically:

```text
I_t(p,a,b) = e_t(p,a,b) 1[k(p,a,b)>=2]
X_t(p,a,b) = e_t(p,a,b) 1[k(p,a,b)=1] 1[event is in a length>=2 xi-proxy track]
T_t(p,a,b) = e_t(p,a,b) - I_t(p,a,b) - X_t(p,a,b).
```

Thus `I+X+T=e` event-by-event.  Globally the measured masses are
`3,963,354`, `2`, and `47,880`, or `98.8063030946%`,
`0.0000498599%`, and `1.1936470454%`.

`X` is deliberately not named physical BEV.  Its transport is the settled G1
translation-only homography using SHA-bound target-cache metric Pose6.  The
available `pose[t]` is the within-pair `f0[t] -> f1[t]` displacement, not the
cross-pair `f1[t-1] -> f1[t]` displacement, and the metric Pose6 source is not
decoder-free.  Consequently the independently observed physical-BEV fraction
and total receiver byte price are both `null`.

## One-time field opportunity

For a receiver-cell rule `r(p,a)->b`, define the collateral-aware net event
gain

```text
G(r) = sum_(t,p) (1[c_t(p)=a,y_t(p)=b] - 1[c_t(p)=a,y_t(p)=a]) 1[r(p,a)=b].
```

With real selected payload bytes `B(r)` from the deterministic tournament
`{raw,zlib9,brotli11,lzma_raw}`, the cell-space opportunity anchors are

```text
Delta d_seg_cell(r) = G(r)/(N H W)
rho_cell(r) = 100 Delta d_seg_cell(r)/B(r).
```

These are not receiver-realized deltas: `Delta d_seg_receiver=null` until an
RGB construction survives resize, uint8, parse-back, and frozen scoring.  The
rate comparison uses `25/37,545,489 = 6.65858953122e-7` score units/byte but
does not authorize admission.

## Zero-payload context law

For binary flip sequence `z` with `K` flips in `n` sites, the Jeffreys-KT code
length is

```text
L_KT(K,n) = -log2(Beta(K+1/2,n-K+1/2)/Beta(1/2,1/2)).
```

Context-free KT costs `25,254,954.1711` bits.  Resetting KT by image pixel,
whose identity and past decoded events cost zero context bytes, costs
`12,343,747.2289` bits: a `51.1234621719%` ideal reduction.  The corresponding
real generic traversal change reduces the selected stream from `490,794` to
`401,633` bytes (`89,161` bytes, `18.1666850043%`).

A predictor-derived 12-row by 7 boundary-distance context gives a
`41.9133426571%` ideal KT reduction but its measured real stream is
`683,211` bytes, `192,417` bytes worse than context-free.  The old #141
margin-saliency map is vehicle-relative to bc20 and explicitly marked for
recalibration; no current v12 predictor-margin tensor was retained.  Therefore
boundary distance is a scoped topology proxy, not a margin-map claim.

CONSUMED-BY: v13 round-2 / v14 one-time cell-field and context-model design.
Any actuation requires a typed receiver surface, real total bytes, RGB
realization, frozen-scorer measurement, and normal contest authority gates.
Pointer remains `0.1910828242 [contest-CPU]`.
