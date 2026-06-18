---
title: EARLY CAPACITY FALSIFICATION GATE — factored RANK-1 LF d_seg-core — VERDICT RED
authority: "[contest-CPU advisory] — NON-PROMOTABLE; exact pointer UNMOVED at 0.19110"
score_claim: false
promotable: false
frontier_pointer_moved: false
mission_contribution: frontier_breaking_enabler
date: 2026-06-18
verdict: RED_LF_CORE_WALLS_ABOVE_CAPACITY_LIMIT
supersedes: none
cross_refs:
  - .omx/research/concentrated_saliency_vehicle_design_20260618T230912Z.md
  - .omx/research/campaign_inflection_three_paths_capped_concentrated_saliency_20260618.md
  - .omx/research/campaign_math_review_dynamics_and_optimization_20260618.md
  - .omx/research/factored_lf_core_capacity_gate_20260618T233913Z.json
  - experiments/probe_factored_lf_core_capacity_gate.py
  - src/tac/tests/test_factored_lf_core_capacity_gate.py
---

# Early capacity falsification gate — factored RANK-1 LF d_seg-core — VERDICT: **RED**

**The MVP-first decisive measurement the design memo §6.5 demanded, run BEFORE the multi-day
build.** All numbers `[contest-CPU advisory]` NON-PROMOTABLE; the exact pointer is UNMOVED at
**0.19110** — stated plainly per the GOAL firewall: **this unit did NOT move the pointer.** Its
value is a measured GREEN/RED that re-routes the sub-0.15 search away from a multi-day build that
would have walled. $0: local MPS fp32 gradient + CPU-authority d_seg; no paid GPU, no PR.

## 0. The headline (read this first)

The RANK-1 factored vehicle's bet (design memo §1, §2): a **SMALL** high-precision LF d_seg-CORE
that renders ONLY the SegNet-decision-band low-frequency structure — spending **ALL** its capacity
on d_seg — can reach **frontier-grade d_seg (~0.0003–0.0006) at FEW bytes**, decoupling the
capacity↔rate tension that capped all three prior paths.

**The gate trained small LF-only cores (narrow-channel HNeRV decoders) DIRECTLY against d_seg
(100% of capacity on the seg objective) and measured the achievable d_seg-vs-byte floor.** This
is STRICTER than the dense bc20 basin (which also spent capacity on recon+pose): if a 100%-d_seg
small core STILL walls, the factorization is dead.

**It walls — decisively, and WORSE than the dense decoder.**

| LF core | params | est. rate | **MEASURED d_seg floor** | × bc20 wall (0.002564) | × frontier (0.0003) | proj. full-vehicle S |
|---|---:|---:|---:|---:|---:|---:|
| bc8 | 20,078 | 0.014 | **0.02584** | 10.1× | 86× | 2.67 |
| bc12 | 36,540 | 0.026 | **0.01689** | 6.6× | 56× | 1.79 |
| *(bc20 dense basin, recon+pose too)* | *83,356* | *0.059* | *0.002564* | *1.0× (the wall)* | *8.5×* | *(basin S 0.378)* |

- Both small LF-only cores floor **6.6–10.1× ABOVE the dense bc20 wall** and **56–86× above
  frontier**. The 1200-epoch bounded trains are fully converged (per-step d_seg drop <1% by ep900).
- The cheap-bytes HALF of the bet IS delivered (bc8 rate 0.014, bc12 0.026 — tiny). But it is
  **irrelevant**: the 100·d_seg term dominates catastrophically (bc12's d_seg term alone = 1.69,
  vs the whole 0.191 frontier). Best projected full-factored-vehicle **S = 1.79** — ~9× WORSE than
  the frontier, ~12× worse than the sub-0.15 target.
- **Capacity extrapolation (power-law over the 2 measured cores, d_seg ~ 29.3·params^−0.71):** to
  reach the bc20 wall (0.0022) a small LF-only core would need **~645K params (7.7× the basin)**;
  sub-0.15-grade d_seg (~0.0006) needs **~4.0M params (48× the basin)**; frontier (0.0003) needs
  **~10.7M params (128× the basin)**. The d_seg-criticality is **capacity-bound**, and a core sized
  to floor d_seg low is NOT small — it FORFEITS the rate win, the exact tension the factorization
  was meant to escape.

**VERDICT: RED — `RED_LF_CORE_WALLS_ABOVE_CAPACITY_LIMIT`.** A small LF-only core does NOT escape
the capacity↔rate tension. It floors d_seg WORSE than the dense decoder per byte, and reaching
frontier-grade d_seg requires MASSIVE capacity (forfeiting the rate). **Do NOT spec the full
multi-day RANK-1 build.** Route to the RANK-4 geometry-anchored parametric d_seg-core contingency
(the design memo's named fallback when RANK-1 walls), OR accept that sub-0.15 needs a
non-learned-decoder boundary representation.

## 1. Why this is the RIGHT, FAITHFUL test (NO-FAKE)

- **Real components:** the canonical exact-eval HNeRV renderer (narrow-channel variants of the
  vendored `HNeRVDecoder`, byte-identical architecture to the bc20 basin); the REAL frozen contest
  SegNet (CPU authority); REAL GT argmax hard labels (`gt_targets_n100.pt`, the SegNet argmax on
  the GT pairs, last-frame); the EXACT eval roundtrip (camera-res bicubic-874 → bilinear-384 →
  round) inside both the training STE path and the d_seg measurement.
- **d_seg is the REAL metric:** mean argmax-flip-rate vs GT over the available GT pairs (the exact
  metric definition, on a subset = a faithful-definition proxy; validated by the sister probe:
  subset bc20 d_seg 0.002564 ≈ full-600 0.0026, ~1% agreement). Proven real by a NO-FAKE test: a
  decoder's d_seg vs ITS OWN SegNet argmax = exactly 0 flips.
- **The core spends 100% on d_seg:** trained on CE on GT argmax through the exact eval path (per
  the design memo: CE not large-margin-hinge, which over-sharpens coarse grids). No recon/pose
  term competes for capacity — the most favorable possible test of the LF core's d_seg ceiling.
- **EMA shadow w/ warmup-decay** (the capstone EMA-shadow-lag fix): d_seg measured on the shadow,
  warmup decay `min(decay,(1+t)/(10+t))` so the shadow tracks early (not frozen near init).
- **MPS = fp32 GRADIENT only; CPU = authority.** Per the MPS train/authority split. No score claim;
  advisory only.
- **Two capacities (bc8, bc12)** + a power-law capacity extrapolation → the verdict is
  capacity-GROUNDED (the descent floor scales with width as a smooth power law), not a
  single-config artifact. Per "Forbidden premature KILL": the closure is scoped (it kills the
  small-LF-core capacity bet; the reactivation criterion is explicit below).

## 2. The mechanism (why CE drops but d_seg walls)

Both cores show the campaign's stretched-exponential / glassy d_seg dynamics: rapid early descent
(0.51 → ~0.05 in ~300 ep), then a sharp bend to a high floor (bc8 ~0.026, bc12 ~0.017). The
diagnostic signal: **CE keeps dropping while d_seg nearly stops.** bc8 ep650→1199: CE stays ~0.10
but d_seg only crawls 0.031 → 0.026. The CE rewards confidence on already-correct pixels; the
residual flips are the **hard boundary pixels** the small core's limited capacity cannot resolve.
This is exactly the campaign's d_seg geometry: d_seg = the argmax-flip RATE ≈ a perimeter integral
over the SegNet decision boundary, concentrated where the detector's top-2 margin is small. A small
core renders the coarse class regions but cannot resolve the thin, high-curvature decision
perimeter — and that perimeter IS d_seg. **More capacity helps** (bc12 < bc8 everywhere; reaches
bc8's final floor 2.6× faster), confirming d_seg-criticality is capacity-bound — but the capacity
needed to floor it LOW is large (the §0 extrapolation), so a *small* core cannot do it.

## 3. The decisive arithmetic (the factorization does not decouple)

The factored bet was: S = 100·d_seg(LF core) + √(10·d_pose) + 25·B_LF/B₀ + 25·B_HF·q/B₀, with
d_seg carried by a small cheap LF core. Measured:
- **bc12 full-vehicle projection: S = 100·(0.0169) + √(10·0.00034) + rate(0.041) = 1.69 + 0.058 +
  0.041 = 1.79.** The d_seg term (1.69) is ~29× the rate term (0.058) — the rate win is noise.
- To make the LF core's d_seg term competitive (say 0.04, i.e. d_seg 0.0004), the §0 extrapolation
  says it needs **~6M params** → B_LF ≈ 6.4MB at high precision → rate ≈ 4.3 (the rate explodes).
  The capacity that floors d_seg IS the bytes; the factorization does not break the coupling — it
  just relocates it (the LF core becomes the dense decoder).

The feasibility probe (sister, `concentrated_saliency_feasibility_*`) already killed the
*regularizer* route (RANK 5: can't concentrate saliency on a shared dense path). This gate kills
the *structural small-core* route (RANK 1): even a physically-distinct, 100%-d_seg, from-scratch
small core walls. **Both the "make the dense path concentrate" and "build a small concentrated
path" hopes are now measured-dead.** The d_seg-criticality is intrinsically capacity-bound and
boundary-bound — it cannot be made both small AND low.

## 4. The verdict + the route (RED)

**RED.** Do NOT spec the full multi-day RANK-1 factored build. The bounded gate ($0, ~15 min of MPS)
saved the multi-day cost of a build that would have capped at S ≈ 1.8 on the d_seg axis.

**Route (the design memo's named contingency, now activated):**
1. **RANK-4 geometry-anchored parametric d_seg-core (the deepest decoupling).** The reason RANK-1
   walls — d_seg lives on a thin, high-curvature decision PERIMETER that a small pixel-rendered core
   cannot resolve — is precisely the argument FOR a parametric boundary: the road↔lane boundary
   (64% of d_seg per the campaign) is low-dimensional (a handful of Bézier/polyline control points),
   so an explicit curve core represents the perimeter EXACTLY at near-zero bytes, sidestepping the
   pixel-rendering capacity wall. Keystones: LSTR / BézierLaneNet (lane = curve params) + DiffVG
   (differentiable rasterization of control points → backprop a margin/d_seg loss into curve
   params). This is a true class-shift (new representation + differentiable rasterizer + from-scratch
   training contract) — HIGH-EV / LOW-feasibility, and now the surviving sub-0.15 direction.
2. **OR the honest representation-level conclusion:** sub-0.15 d_seg (~0.0003–0.0006) is NOT
   reachable by ANY learned-pixel-decoder that is also byte-cheap — the capacity wall is intrinsic
   to pixel-rendering the decision perimeter. The boundary must be carried by a NON-dense,
   parametric/geometric representation (RANK 4) or sub-0.15 is rate-floor-infeasible on this video.

**Before committing to RANK-4 (next $0 gate):** a tiny feasibility probe — fit Bézier/polyline
curves to the GT SegNet road↔lane argmax boundary on a few frames and measure the achievable d_seg
of a differentiably-rasterized curve core (does a low-dim curve set reach d_seg << 0.0022 where the
pixel core could not?). Same MVP-first discipline: $0 gate BEFORE the multi-day DiffVG build.

## 5. Canonical-vs-unique decision per layer (UNIQUE-AND-COMPLETE-PER-METHOD)

| layer | decision | rationale |
|---|---|---|
| LF core renderer | **ADOPT_CANONICAL** (narrow-width vendored `HNeRVDecoder`) | the exact-eval roundtrip + preprocess + pair contract MUST match the contest byte-for-byte; forking the renderer breaks faithfulness. The UNIQUE choice is the WIDTH (the capacity knob under test). |
| d_seg eval path | **ADOPT_CANONICAL** (exact roundtrip + last-frame argmax-flip) | validated faithful (subset ≈ full-600); the metric under test must be the contest metric. |
| train objective | **FORK_PRINCIPLED** (CE on GT argmax, NOT margin-hinge) | the int5-QAT finding: margin-hinge over-sharpens coarse grids; CE recovers d_seg better. Probe-confirmed: CE drives the early descent. |
| EMA | **ADOPT_CANONICAL** (warmup-decay shadow) | the capstone EMA-shadow-lag fix; measured on the shadow. |
| verdict logic | **FORK_PRINCIPLED** (measurement-first, reject degenerate fit) | the stretched-exp projection degenerated to 0.0 (c=0/long-tau grid solution) and would have produced a FALSE GREEN; per NO-FAKE "surrogate ≠ authority", the verdict is driven by the MEASURED floor. |

## 6. Observability surface

- **Inspectable per core:** full d_seg(ep) + CE(ep) trajectory per LF core (25 eval points each),
  in the JSON `cores.<key>.curve`.
- **Decomposable:** S = 100·d_seg + √(10·pose) + rate, each term reported per core in
  `S_projection_rows`.
- **Diff-able:** the `factored_lf_core_capacity_gate.v1` JSON schema; cores keyed by base_channels.
- **Queryable post-hoc:** `.omx/research/factored_lf_core_capacity_gate_20260618T233913Z.json`.
- **Cite-able:** (basin_ckpt sha, base_channels, n_params, measured d_seg floor) tuple per core.
- **Counterfactual-able:** the capacity extrapolation answers "what core size reaches the wall?"
  (7.7× basin) without training it.

## 7. 6-hook wire-in (Catalog #125)

- **#1 sensitivity-map:** the gate REFINES the capacity↔d_seg relation: d_seg-criticality is
  capacity-bound with power-law exponent k≈0.71; a small core cannot floor it — ACTIVE.
- **#2 Pareto constraint:** the measured (d_seg floor, byte) points for bc8/bc12 are new points on
  the LF-only-core Pareto curve (both dominated by the dense basin on d_seg-per-S) — ACTIVE.
- **#3 bit-allocator:** N/A — the verdict is RED on the LF core itself; no periphery-shed to
  allocate (the build is not spec'd) — N/A-with-rationale.
- **#4 cathedral autopilot:** N/A (a feasibility verdict, not an archive-deployable surface).
- **#5 continual-learning posterior:** the RED gate outcome (small-LF-core-walls) is a probe
  outcome; register in `probe_outcomes.jsonl` on the next wiring pass — ACTIVE.
- **#6 probe-disambiguator:** the gate IS the disambiguator between "small LF core escapes the
  capacity wall" (NO, measured) and "d_seg is intrinsically capacity-bound → needs RANK-4 geometry"
  (YES) — ACTIVE.

## 8. Files
- `experiments/probe_factored_lf_core_capacity_gate.py` — the resumable $0 bounded-train gate
  (timing-smoke + 2 LF cores + measurement-first verdict + capacity extrapolation).
- `src/tac/tests/test_factored_lf_core_capacity_gate.py` — 10 NO-FAKE tests (incl. d_seg
  self-match-is-zero; EMA warmup-decay tracks-not-freezes; eval-roundtrip real uint8 path; LF-core
  params shrink with width; stretched-exp fit on a decaying series).
- `.omx/research/factored_lf_core_capacity_gate_20260618T233913Z.json` — the advisory result
  (both cores' full trajectories + verdict + capacity extrapolation).
- `experiments/results/factored_lf_core_capacity_gate/` — gate_state.json + best-shadow ckpts
  (rebuildable).

NO-FAKE: the d_seg floors are the REAL argmax-flip-rate of the REAL frozen SegNet on the EXACT eval
path vs REAL GT labels, on REAL narrow-width HNeRV cores trained 100% on the seg objective; the
probe's auto-verdict was CORRECTED from a degenerate-fit false-GREEN to the measurement-first RED.
The only authoritative d_seg is upstream/evaluate.py; these are advisory. Exact pointer UNMOVED at
0.19110.
