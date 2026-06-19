---
title: RANK-4 GATE — differentiable parametric-curve d_seg-core — VERDICT RED (survival wall) + TERMINAL FINDING
authority: "[contest-CPU advisory] — NON-PROMOTABLE; exact pointer UNMOVED at 0.19110"
score_claim: false
promotable: false
frontier_pointer_moved: false
mission_contribution: frontier_breaking_enabler
date: 2026-06-18
verdict: RED_CURVE_CORE_HITS_SURVIVAL_WALL_LIKE_STATIC_STORE
terminal_finding: sub-0.15 byte-cheap d_seg is unreachable across all THREE tested families (learned-pixel-decoder + static-geometry + differentiable-curve); the survival wall is family-independent
supersedes: none
cross_refs:
  - .omx/research/campaign_inflection_three_paths_capped_concentrated_saliency_20260618.md
  - .omx/research/factored_lf_core_capacity_gate_20260618T233940Z.md
  - .omx/research/partition_store_realization_gate_DEFER_20260617T024639Z.md
  - .omx/research/yousfi_road_lane_geometric_solve_probe_20260617.md
  - experiments/probe_curve_core_dseg_feasibility_gate.py
  - src/tac/tests/test_curve_core_dseg_feasibility_gate.py
  - .omx/research/curve_core_dseg_feasibility_gate_20260619T010157Z.json
---

# RANK-4 gate — differentiable parametric-curve d_seg-core — VERDICT: **RED** (the survival wall) + the **terminal sub-0.15 finding**

**The LAST architectural test for byte-cheap sub-0.15, run MVP-first ($0) BEFORE any multi-day
DiffVG build.** All numbers `[contest-CPU advisory]` NON-PROMOTABLE; the exact pointer is UNMOVED
at **0.19110** — stated plainly per the GOAL firewall: **this unit did NOT move the pointer.** Its
value is a measured RED that closes the last byte-cheap-geometry escape and converts the campaign's
"three paths capped" into a defensible **terminal representation-level finding**. $0: local CPU
authority + (gradient) the real frozen SegNet; no paid GPU, no PR.

## 0. The headline (read this first)

The differentiable-curve bet (the factored-LF gate's named RANK-4 contingency): a set of curve
control points defines the class partition, and — unlike the static partition STORE (store the
boundary → paint flat colours → hope it survives) — the curves' **per-class colours + a boundary-band
offset are optimized THROUGH the real SegNet + the exact uint8 roundtrip**, so gradients
**pre-compensate the bilinear mixing** (the survival check baked into the objective). If anything could
beat the static store's survival wall, this differentiable-through-the-scorer lever is it.

**It hits the SAME survival wall — decisively, and geometry-independently.**

| curve complexity | ~n_ctrl pts | **geo_recon** (pure geometry vs L*) | **realized d_seg** (THROUGH the roundtrip) | boundary-band flip | survival gap (realized−geo) | rate (amort) | proj. S |
|---|---:|---:|---:|---:|---:|---:|---:|
| mp8  | 171 | 0.05748 | **0.01228** | 26.2% | −0.045 (geom too coarse) | 0.00107 | 1.287 |
| mp16 | 270 | 0.01010 | **0.00994** | 21.7% | −0.0002 | 0.00165 | 1.053 |
| mp32 | 401 | 0.00335 | **0.00714** | 17.2% | +0.0038 (2.1×) | 0.00241 | 0.775 |
| mp64 | 625 | 0.00194 | **0.00722** | 16.8% | +0.0053 (3.7×) | 0.00373 | 0.784 |
| **mp128** | 838 | **0.00106** | **0.00673** | 16.1% | **+0.0057 (6.3×)** | 0.00497 | **0.736** |
| *frontier (reference)* | — | — | *0.00257* | — | — | *(0.191 total)* | *0.191* |

- **The geometry escapes param-explosion.** geo_recon descends monotonically with curve count:
  0.0575 → 0.0101 → 0.0034 → 0.0019 → **0.00106 at mp128 (2.4× BELOW the frontier's own d_seg
  0.00257)**. A few hundred control points represent the SegNet decision boundary essentially
  perfectly. The "param-explosion" half of the RED hypotheses does NOT bind — curves ARE byte-cheap
  AND geometrically accurate (rate 0.001–0.005 ≪ 0.05 throughout).
- **But realized d_seg floors at the SURVIVAL WALL, ~0.0067–0.0072, independent of geometry.** As
  geo_recon drops 50× (mp8→mp128), realized barely moves (0.0123→0.0067) and the **survival gap
  GROWS to 6.3×** (mp128: realized 0.0067 vs combinatorial geometry 0.0011). The binding constraint is
  NOT how well the curves fit L*; it is that a **flat-per-class-colour rendering of even a perfect
  partition does not reproduce L* through the real SegNet.**
- **The gap is the FLAT-COLOUR-vs-SegNet wall, not the resize** (the sharpened mechanism, from the
  3-way decomposition). At mp128: combinatorial geometry vs L* = 0.00106; painting that perfect
  partition with flat per-class colours and running the **real SegNet (NO roundtrip)** already mismatches
  L* by **0.00668** (gap +0.00562 — the WHOLE wall); adding the uint8 roundtrip moves it only +0.00005
  (0.00673). **SegNet's argmax on a flat-colour "cartoon" sits in a different place than on the real
  textured frame at the boundary** (~16% boundary-band flip), and the bilinear resize is a negligible
  add-on. This is even MORE damning than the static-store "resize mixes colours" framing: a byte-cheap
  geometric core can only render flat regions, and flat regions do not reproduce the detector's decision
  boundary — regardless of the resize.
- **The differentiable lever (the one thing this family added over the static store) dents it ~14%**
  (per-class colours + boundary-band offset optimized THROUGH the real SegNet + STE roundtrip; sister
  smoke realized 0.0092→0.0079, boundary 0.193→0.176) but cannot close it. The static store used
  SegNet-best flat colours WITHOUT backprop-through-the-scorer and hit 24% boundary flip; this gate adds
  backprop-through-the-scorer and hits 16% — a dent, not an escape.
- **Best projected S = 0.736** (mp128) — ~3.9× the 0.191 frontier, ~5× the sub-0.15 target. The
  d_seg term (100·0.0067 = 0.67) is ~135× the rate term (0.005). The byte win is noise.

**VERDICT: RED — `RED_CURVE_CORE_HITS_SURVIVAL_WALL_LIKE_STATIC_STORE`.** A byte-cheap differentiable
curve core represents the boundary perfectly and cheaply, but a **flat-per-class-colour rendering of
that perfect partition does not reproduce the SegNet decision boundary** (~16% boundary-band flip even
WITHOUT the resize; the resize adds a negligible +0.00005). This is the same wall the non-neural
partition STORE (realized 0.0064, S 0.84) and the per-pixel witness sidecar (37% survival, NO-GO) hit —
the differentiable-through-the-scorer optimization (the one thing this family added) only dents it ~14%.
**Do NOT spec the multi-day DiffVG build.**

*(mp256, the largest + purely-confirmatory complexity, was STOPPED: its decimated 256-pt polygon
reconstruction is pathologically slow, and the mp8→mp128 trend is already a complete, monotone,
decisive RED. mp256 could only confirm the asymptote — it cannot flip the verdict (realized would have
to drop from 0.0067 to <0.0012). The probe's own measurement-first verdict was finalized via
`--verdict-only` on the 5 real measured rows: best realized d_seg 0.00673 (2.6× frontier, 1.05× the
static-store survival wall), best S 0.736, wall = SURVIVAL_WALL — VERDICT
RED_CURVE_CORE_HITS_SURVIVAL_WALL_LIKE_STATIC_STORE.)*

## 1. The TERMINAL FINDING (the airtight re-frame the directive asked for)

Three orthogonal byte-cheap d_seg-core families are now measured-capped, and they share ONE wall:

| family | gate | byte-cheap? | low d_seg? | why it caps |
|---|---|---|---|---|
| **learned-pixel-decoder** | factored RANK-1 LF core (`factored_lf_core_capacity_gate_*`) | yes | **NO** | d_seg ~ params^−0.71 **CAPACITY wall**; a small core floors d_seg 6.6–10.1× ABOVE the dense basin; reaching sub-0.15-grade needs ~4M params (forfeits the rate). |
| **static-stored-geometry** | partition STORE realization (`partition_store_realization_gate_DEFER_*`) + lane-marking geometric solve | yes | **NO** | the **SURVIVAL wall**: stored boundary painted flat → bilinear downsample mixes boundary-band colours → SegNet argmax flips ~24% → realized d_seg 0.0064, S 0.84. |
| **differentiable-curve** (THIS gate) | RANK-4 curve core | yes | **NO** | the **SAME wall**: geometry fits perfectly (geo_recon 0.0011 < frontier) + colours optimized THROUGH the real SegNet, but a flat-colour rendering still flips the boundary band ~16% (NO roundtrip needed; resize adds +0.00005) → realized 0.0067, S 0.74. |

**Terminal finding (honest, decisive):**
> **Sub-0.15-grade d_seg (~0.0003–0.0006) is byte-cheaply unreachable on this contest video across
> all three tested representation families.** Two distinct intrinsic walls block it: (1) for a
> *pixel-rendered* representation, d_seg is **capacity-bound** (a low-d_seg decoder is not small); (2)
> for a *geometric/parametric* representation that IS small, the **flat-region rendering it can produce
> does not reproduce the SegNet decision boundary** — SegNet's argmax on a flat-per-class-colour
> "cartoon" of even a perfect partition flips ~16–24% at the boundary band vs the real textured frame,
> regardless of how the boundary is carried, how fine the geometry is, or how the colours are chosen
> (even by gradient through the real scorer). This is a property of the **detector's decision geometry
> on flat-colour input**, NOT of the resize (the bilinear downsample adds a negligible +0.00005).
>
> The frontier ~0.191 is therefore **near the real achievable floor for a byte-cheap d_seg core.**
> The information-theoretic `S_floor=0.118` over-counts the achievable: it prices d_seg and bytes as
> independent rate terms, but they are **coupled** — a byte-cheap representation can only render either
> a small learned-pixel-decoder (capacity-bound: low d_seg costs bytes) or a flat-region cartoon
> (boundary-bound: SegNet does not reproduce L* on flat regions). The decoupled `S_floor` is an
> *unreachable* lower bound; the *reachable* floor on a byte-cheap core sits near the current frontier.

This does NOT say sub-0.15 is impossible by ANY means — it says the byte-cheap-d_seg-core class
(the campaign's surviving sub-0.15 hope) is exhausted. The remaining non-excluded directions are
**non-byte-cheap** (more capacity, i.e. a richer decoder, paying the rate) or a **fundamentally
different objective lever** (pose-axis, or a survival-aware boundary representation that pre-compensates
the resize at the *content* level — see §4 reactivation).

## 2. Why this is the RIGHT, FAITHFUL test (NO-FAKE)

- **Real components:** the REAL frozen contest SegNet (CPU authority); REAL GT argmax L* (the `seg`
  field of the capstone GT cache = SegNet argmax on the GT frame1, last frame, 384×512); the EXACT
  eval roundtrip (camera-res bicubic-874 → bilinear-384 → round) inside BOTH the differentiable fit
  (STE round) and the hard d_seg measurement (the survival check).
- **d_seg is the REAL metric:** realized = mean argmax-flip-rate of the HARD-painted curve frame's
  SegNet argmax vs L*, THROUGH the exact roundtrip, averaged over 3 GT frames (the exact metric
  definition on a subset = a faithful-definition proxy; the sister probes validated subset ≈ full-600
  to ~1–2%). Proven real by NO-FAKE tests: a frame's SegNet argmax self-matches at exactly 0 flips; a
  mid-grey frame flips a non-trivial fraction.
- **The differentiable lever is REAL:** the per-class colours + boundary-band offset carry a gradient
  THROUGH the STE roundtrip + the real SegNet (tested: STE roundtrip passes a non-zero gradient;
  colour gradient flows); the sister smoke showed CE 0.146→0.034 and realized 0.0092→0.0079. It is the
  static store's exact lever PLUS backprop-through-the-roundtrip — and it still walls.
- **The geometry is REAL:** decimated region-boundary polygons (cv2.approxPolyDP at swept max
  control-points) re-rasterized (cv2.fillPoly, area z-order so thin lane markings land on top); tested
  monotone (more points → lower geometric mismatch; control-point count grows with budget).
- **MEASUREMENT-FIRST:** the verdict is driven by realized d_seg through the chain, NOT the CE fit loss
  (a surrogate). The low geo_recon + high realized = the survival wall, the static-store failure mode —
  exactly the false-GREEN the factored-LF gate warned against (CE/geometry dropping ≠ d_seg dropping).
- **CPU authority; NEVER MPS for the score.** No score claim; advisory only. Exact pointer UNMOVED.

## 3. The mechanism (why the curves fit but don't reproduce the boundary) — the 3-way decomposition

d_seg = the argmax-flip RATE ≈ a perimeter integral over the SegNet decision boundary. A curve core can
place that perimeter EXACTLY (combinatorial geo_recon 0.0011 at mp128). The 3-way decomposition of the
mp128 gap pinpoints WHERE the d_seg is born:

| stage | what it measures | mp128 value | gap added |
|---|---|---:|---:|
| combinatorial geometry | recon partition vs L* (pure pixels) | 0.00106 | — |
| paint flat colours → **real SegNet** (NO roundtrip) | does SegNet reproduce L* on the flat-colour cartoon? | **0.00668** | **+0.00562 (the whole wall)** |
| + exact uint8 roundtrip | does the resize add flips? | 0.00673 | +0.00005 (negligible) |

**The wall is the flat-colour→SegNet step, NOT the resize.** SegNet's argmax on a flat-per-class-colour
rendering of even a perfect partition mismatches L* by ~16% at the boundary band — because the real
frame has TEXTURE (gradients, shading, lane-marking contrast) that places SegNet's decision boundary
where it is, and a flat-colour cartoon does not reproduce that local evidence. The bilinear downsample
adds essentially nothing on top (+0.00005). The differentiable lever (per-class colours + boundary-band
offset optimized THROUGH the real SegNet) tries to find flat colours whose SegNet argmax matches L*, and
it helps at the margin (~14%; static store's 24% boundary flip → 16% here), but a **byte-cheap geometric
core can only render flat regions, and flat regions do not carry the texture evidence SegNet's boundary
decision depends on.** This is representation-independent: any small geometric/parametric core (curve,
polygon, partition store) produces flat regions and hits the same ~16–24% boundary wall.

## 4. The route / reactivation (RED, not KILL — per Forbidden premature KILL)

The PARADIGM (geometric/parametric attack on the dominant d_seg contour) is **IMPLEMENTATION-LEVEL
falsified** for the byte-cheap-flat-region-partition realization (Catalog #307). Reopen ONLY if a
realization gets **boundary-band flip below ~3%** vs L* through the real SegNet (current best 16.1% at
mp128; the partition-store GT-blend *upper bound* was 16.9%; the witness sidecar 37%). The single
candidate not yet measured: **reinstate the local TEXTURE evidence at the boundary** — not flat colours
but a thin learned/stored texture sidecar along the contour (so SegNet sees the gradient/contrast cue it
uses to place the decision boundary), e.g. the geometric-solve probe's deconvolution-realized luma
preimage carrying a margin-KKT GT-class-restoring target. But the partition-store GT-blend upper bound
(inject 50% real GT texture → still 16.9% boundary flip) suggests the headroom is small, and any texture
sidecar dense enough to matter starts to pay the rate (forfeiting "byte-cheap"). The decomposition
above is why: the wall is texture-dependent boundary evidence, and texture is not byte-cheap.

**The campaign-level route (the directive's "either verdict is decisive"):** the byte-cheap-d_seg-core
class is exhausted. Aim the next exact-score unit at the **non-excluded levers**:
1. **Incremental on-frontier d_seg cuts** (#137 boundary sidecar / #138 lane prior on the live
   renderer's own frame-1) — small, NOT sub-0.15, but a candidate sub-0.19 pointer nudge ("any score
   sub-0.19 is good progress").
2. **The pose axis** — at the frontier operating point pose marginal sensitivity is 2.71× SegNet's;
   the curve core HELD pose; a pose-targeted lever is unexplored here.
3. **Accept the rate** — a higher-capacity decoder (paying bytes) is the only family not survival- or
   capacity-walled for d_seg; the sub-0.15 question becomes "can a richer decoder's rate be clawed
   back elsewhere," not "can a byte-cheap core floor d_seg."

## 5. Canonical-vs-unique decision per layer

| layer | decision | rationale |
|---|---|---|
| SegNet + eval roundtrip | **ADOPT_CANONICAL** | the exact-eval chain MUST match the contest byte-for-byte (the survival wall lives in the resize); forking it breaks faithfulness. |
| d_seg metric | **ADOPT_CANONICAL** (realized argmax-flip vs L*) | the metric under test must be the contest metric; geometric d_seg reported separately as the param-explosion diagnostic. |
| geometry reconstruction | **FORK_PRINCIPLED** (cv2 decimated polygon partition) | the byte-cheap curve representation under test; area z-order so thin lane markings (the dominant boundary) survive the fill. |
| survival lever | **FORK_PRINCIPLED** (colours + band offset, differentiable through SegNet+roundtrip) | the one thing this family adds over the static store; isolates the survival question given byte-cheap geometry. |
| verdict logic | **FORK_PRINCIPLED** (measurement-first; realized, not CE) | per NO-FAKE "surrogate ≠ authority"; the low-CE/low-geometry + high-realized signature IS the survival wall. |

## 6. Observability surface

- **Inspectable per complexity:** per-frame geo_recon / geo_seg / realized / boundary-flip / interior-flip / n_ctrl / CE(first,last) in the JSON `rows.<mpN>.per_frame`.
- **Decomposable:** S = 100·realized_d_seg + √(10·held_pose) + rate, each term per row.
- **Diff-able:** the `curve_core_dseg_feasibility_gate.v1` JSON schema; rows keyed by complexity.
- **Queryable post-hoc:** `.omx/research/curve_core_dseg_feasibility_gate_20260619T010157Z.json`.
- **Cite-able:** (complexity, n_ctrl, geo_recon, realized_d_seg, boundary_flip, rate, S) per row.
- **Counterfactual-able:** the survival gap (realized − geo_recon) GROWING with finer geometry IS the
  counterfactual answer to "would more curve points fix it?" — NO (the gap grows, not shrinks).

## 7. 6-hook wire-in (Catalog #125)

- **#1 sensitivity-map:** REFINES the d_seg model — d_seg of a byte-cheap geometric core is
  **survival-bound** (boundary-band-flip ~16% floor), NOT geometry-bound; the binding sensitivity is
  ∂(argmax)/∂(downsample-mixed-colour), not ∂/∂(curve-point) — ACTIVE.
- **#2 Pareto constraint:** the (rate, realized d_seg) points for mp8–mp128 are new dominated points on
  the byte-cheap-geometry Pareto curve (all dominated by the frontier on d_seg-per-S) — ACTIVE.
- **#3 bit-allocator:** N/A — RED on the core; no periphery to allocate.
- **#4 cathedral autopilot:** N/A (a feasibility verdict, not an archive-deployable surface).
- **#5 continual-learning posterior:** the RED outcome (curve-core-survival-wall) registered as a probe
  outcome in `probe_outcomes_ledger` — ACTIVE.
- **#6 probe-disambiguator:** the gate IS the disambiguator between "differentiable curves escape the
  survival wall the static store hit" (NO, measured: 16% boundary flip, 6.3× survival gap) and "the
  survival wall is representation-independent → sub-0.15 byte-cheap d_seg is unreachable" (YES) —
  ACTIVE.

## 8. Files
- `experiments/probe_curve_core_dseg_feasibility_gate.py` — the resumable $0 gate (decimated polygon
  geometry + differentiable-through-SegNet colour/band survival optimizer + measurement-first verdict).
- `src/tac/tests/test_curve_core_dseg_feasibility_gate.py` — 12 NO-FAKE tests (geometry monotone in
  control points; STE roundtrip is differentiable + real uint8; colour core's frame depends on the
  learnable colours; byte cost monotone; dominant pair = longest boundary not largest area; realized
  d_seg is the real argmax-flip functional).
- `.omx/research/curve_core_dseg_feasibility_gate_20260619T010157Z.json` — the advisory result (all
  rows' full per-frame trajectories + verdict).
- `experiments/results/curve_core_dseg_feasibility_gate/gate_state.json` — resumable state (rebuildable).

NO-FAKE: the realized d_seg floors are the REAL argmax-flip-rate of the REAL frozen SegNet on the EXACT
eval roundtrip vs REAL GT labels, for REAL decimated-polygon curve cores with per-class colours +
boundary-band offsets optimized THROUGH the real SegNet. The geometry escapes param-explosion (geo
0.0011 < frontier) but realized floors at the survival wall (0.0067, 6.3× the geometry) — the
static-store failure mode, family-independent. The only authoritative d_seg is upstream/evaluate.py;
these are advisory. Exact pointer UNMOVED at 0.19110.
