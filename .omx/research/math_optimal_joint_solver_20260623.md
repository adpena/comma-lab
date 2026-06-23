---
title: "Math-optimal joint decoder solver — solving (C,T,Q,E) for min S over the measured surfaces, the achievable-S floor, the training-time Pareto, and the single recommended next config"
authority: "[contest-CPU advisory] / [macOS-CPU advisory] — NON-PROMOTABLE. score_claim=false; promotion_eligible=false; ready_for_exact_eval_dispatch=false. Pointer UNMOVED 0.19109982 [contest-CPU]. $0; CPU-only; NO MPS; NO paid dispatch; NO pinned-upstream edits. This is a MEANS memo (a solver + a config recommendation); it moves no pointer."
date: 2026-06-23
subagent: mathopt-solver-20260623
score_claim: false
promotion_eligible: false
pointer_moved: false
all_score_math_via: tac.contest_score.compute_contest_score
module: src/tac/optimization/math_optimal_joint_solver.py
cli: tools/math_optimal_joint_solve.py
json: reports/math_optimal_joint_solve.json
tests: src/tac/optimization/tests/test_math_optimal_joint_solver.py (39 pass)
cross_refs:
  - .omx/research/dseg_384_achievability_floor_verdict_20260623.md      # FLOOR-384 d_seg 1.875e-4 (S 0.0187), CAPACITY-LIMITED
  - .omx/research/dseg_reducibility_gt_margin_verdict_20260623.md        # OUR flip set IRREDUCIBLE (label-noise), ΔS ceiling 0.012
  - .omx/research/pr95_vs_ours_convergence_gap_and_capacity_rd_deepmath_20260623.md  # capacity-RD S(p), entropy floor, α∈[0.9,1.5]
  - src/tac/capacity_rd_qat.py                                            # the C×Q 1D solve this EXTENDS
  - src/tac/contest_score.py                                             # the compliance bedrock (all score math)
  - .claude .../memory/feedback_terminal_conclusion_needs_existence_proof_crosscheck_20260623.md  # the existence-proof discipline
---

# Math-optimal joint decoder solver — the "solve math-optimal everywhere" artifact

**One-paragraph answer to the operator.** I built the joint solver the directive asked for:
`tac.optimization.math_optimal_joint_solver` minimises the exact contest score `S = 100·d_seg +
sqrt(10·d_pose) + 25·bytes/N` over the four-axis config space **(C capacity, T taper, Q weight-bits, E
training-epochs)**, ingesting every measured response surface (the d_seg-384 achievability floor, the
capacity power law, the int8→int-N byte-shrink ratios, the (epochs, d_seg) convergence anchors, the
frontier + bc20 + PR95 existence-proof anchors). The solve is REUSE, not rebuild — it imports
`tac.contest_score` for ALL score arithmetic and `tac.capacity_rd_qat` for the measured anchors + byte
model + the C×Q 1D solve, and adds the convergence (E), taper (T), pose-convergence, and 384-floor axes.
**The headline is a TWO-LAYER floor**: the surface-MODEL lower bound (the pessimistic power-law-d_seg
optimum) is **S ≈ 0.179** (bc36 + int4), but the **PHYSICAL achievable floor — the real T_floor over the
measured surface, replacing the loose analytic 0.118 — is S ≈ 0.059** (a perfect-384 d_seg decoder + the
converged frontier pose + a small int4 byte budget). The gap between them is the single binding open
question: **capacity-realization** (can a small decoder be TRAINED to approach the 384 d_seg floor at a
small byte budget?). The existence-proof cross-check (mandatory per the 2026-06-23 discipline) fires
exactly here: it flags the model's 0.179 "lower bound" as NOT a physics floor, because a perfect-384
decoder at the bc20 byte budget already reaches S ≈ 0.095. **The single recommended next config to
train: a bc20-class (≈83K-param) decoder, fully converged via the corrected PR95 8-stage curriculum, at
int8 (or int4 IF QAT holds d_seg ≤ ~1.0e-3) — because at the bc20 byte budget with the converged
frontier pose, sub-0.15 needs d_seg < 7.35e-4 (int8) / 1.02e-3 (int4), and PR95 already measures d_seg
5.6e-4.** The model's bc36 optimum is an artifact of the pessimistic 2-point power law; the
existence-proof + break-even arithmetic point to a SMALL decoder at the rate-headroom operating point.

---

## PART A — The formulation (the solve)

Minimise over `(C, T, Q, E)`:

    S(C,T,Q,E) = 100·d_seg(C,T,Q,E) + sqrt(10·d_pose(E)) + 25·bytes(C,Q)/N      (N = 37,545,489)

with the four axes modelled from the measured surfaces:

- **d_seg(C,T,Q,E) = [ dseg_convergence(E → d_seg_inf(C)) · T_mult ] + Q_spill**, CLAMPED at the 384
  floor (1.875e-4). `d_seg_inf(C)` is the capacity power law (bc20↔frontier endpoints) bounded below by
  the 384 achievability floor; `dseg_convergence` is an exponential approach to that asymptote fit to the
  two measured bc20 (epochs, d_seg) anchors; `T_mult` is the byte-neutral boundary-band-taper d_seg
  multiplier (default 1.0, no measured A/B); `Q_spill` is the QAT distortion-hold spill (an ASSUMPTION).
- **d_pose(E) = dpose_convergence(E)** — DECOUPLED from capacity (deepmath), an exponential approach from
  the under-trained basin pose (3.42e-4) to the converged frontier pose (2.93e-5). This is the DOMINANT
  sub-0.15 swing (pose term 0.0585 basin → 0.0171 converged = 0.041 S), so modelling pose as
  convergence-dependent (not a fixed basin constant) is essential — the single most important correction
  the solver makes vs the prior 1D capacity desk model.
- **bytes(C,Q) = native_bytes(C) · mixed_byte_fraction(Q)** — native bytes EXACT from `decoder_param_count`
  (calibrated to the bc20 measured anchor); the mixed byte fraction is the MEASURED int8→int-N ratios
  blended by the low-precision fraction.

The solve: the C-axis convex S(p) optimum is already KKT-solved inside `capacity_rd_qat`; the joint over
T/Q/E is a small structured grid around it (the surfaces are measured-at-anchors + power-law/exponential
models between them, so a grid + the C-axis closed form is the honest "solvable math over arbitrary
sweep" — per CLAUDE.md Meta-Lagrangian/Pareto solver). The achievable-S **lower bound** is the
fully-converged (E→∞), best-T, best-Q config; the **physical floor** is the existence-proof construction
(perfect-384 d_seg + converged pose + smallest viable byte budget).

## PART B — The ingested surfaces (with provenance + the existence-proof cross-checks)

| surface | value | provenance | status |
|---|---|---|---|
| **d_seg 384 floor** | 1.875e-4 (S 0.0187) | `dseg_384_achievability_floor_n600_20260623.json` floors.floor_384.d_seg (MEASURED N=600) | LANDED (read by the solver) |
| d_seg resolution-bottleneck floor | 1.596e-4 (S 0.016) | same JSON floor_384_float | LANDED |
| **capacity power law** d_seg(C) | bc20→0.0026, bc36→0.00056 | `capacity_rd_qat` anchors (bc20 basin + frontier) | MEASURED endpoints; α∈[0.9,1.5] MODELLED between |
| **Q-axis byte shrink** int8→int-N | int4 = 0.52× | `capacity_rd_qat.MEASURED_BYTE_SHRINK_BC20` (`reports/fp_shrink_ptq_bc20_n600.json`) | MEASURED (bc20 post-int8-brotli) |
| **E-axis convergence** (epochs, d_seg) | (120, 0.00376), (2325, 0.00256) | deepmath B.1 bc20 matched-recipe | MEASURED (stage-1 CE-only → conservative) |
| **pose convergence** d_pose(E) | basin 3.42e-4 → frontier 2.93e-5 | `capacity_rd_qat` ANCHOR_BC20.d_pose / ANCHOR_FRONTIER.d_pose | MEASURED endpoints; τ MODELLED |
| frontier S (existence anchor #1) | 0.19110 | `canonical_frontier_pointer.json` | MEASURED [contest-CPU] |
| PR95 own-trained d_seg (anchor #2) | 5.6e-4 | deepmath PART A (author + our recode) | MEASURED |
| bc20 basin d_seg (anchor #3) | 0.0026 | `capacity_rd_qat.ANCHOR_BC20` | MEASURED |

**Sister Q-axis / E-axis agent surfaces** (`.omx/research/qaxis_bitdepth_response_surface_20260623.json`
and `eaxis_training_time_optimization_surface_20260623.json`) are NOT yet published; the solver's
`load_ingested_surfaces` reads them when they land (interface contract: `{"byte_fraction": {...}}` and
`{"epoch_dseg_anchors": [[E, d_seg], ...]}`) and falls back to the deepmath measured anchors until then.
This is scaffolded + tested so the answer sharpens automatically when they arrive.

### Existence-proof cross-checks (every floor the solver emits)

Per the 2026-06-23 binding discipline, every floor/lower-bound is cross-checked against the known
artifacts. The result of the cross-check on the surface-MODEL lower bound (0.179):

> **INVALID floor — perfect-384 d_seg @ bc20 bytes ALREADY achieves S = 0.095 < the claimed 0.179.** The
> 0.179 is a capacity/recipe artifact (the pessimistic 2-point power law), NOT a physics floor.

This is the discipline working as designed: it forces the solver to emit BOTH the model lower bound AND
the physical floor, and to name the gap (capacity-realization) rather than declaring 0.179 a wall.

## PART C — The math-optimal config + the achievable-S frontier

### C.1 The per-capacity converged table (E→∞, frontier pose 2.93e-5)

| C (base_ch) | dec params | d_seg_inf | int8 bytes | **int8 S** | int4 bytes | **int4 S** |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 58,103 | 0.0045 | 66,661 | 0.512 | 34,843 | 0.520 |
| 20 | 83,356 | 0.0026 | 89,136 | 0.337 | 46,590 | 0.338 |
| 24 | 112,901 | 0.00164 | 115,431 | 0.258 | 60,334 | 0.251 |
| 28 | 148,038 | 0.00109 | 146,703 | 0.223 | 76,679 | 0.207 |
| 32 | 186,352 | 0.00077 | 180,802 | 0.214 | 94,502 | 0.187 |
| **36** | **228,958** | **0.00056** | 218,722 | 0.219 | 114,323 | **0.179** ← model optimum |
| 40 | 277,711 | 0.00042 | 262,112 | 0.233 | 137,002 | 0.180 |

### C.2 The two-layer floor (the operator's "math-optimal everywhere" answer)

- **Surface-model lower bound: S ≈ 0.179** at `C=36, T=1.0, Q=int4(frac_low=1.0), E→∞`. Under the
  pessimistic 2-point power law + the 0.0003 int4 d_seg-hold spill, no config in the grid reaches
  sub-0.15. The best is bc36+int4 at 0.179 (just under frontier 0.191).
- **PHYSICAL achievable floor (the real T_floor): S ≈ 0.059** = perfect-384 d_seg (1.875e-4) + converged
  pose (2.93e-5) + bc16 int4 bytes (34,843). At the bc20 int8 budget it is 0.095; at bc20 int4, 0.067.
  **This REPLACES the loose analytic 0.118** (S_floor) with a measured-surface number — and it is
  comfortably sub-0.15.
- **The gap (0.179 → 0.059) is the capacity-realization question.** The power law is recipe + convergence
  confounded (deepmath B: the bc20/bc36 anchors are under-converged / borrowed); the 384 measurement
  proves d_seg 1.875e-4 is physically achievable by a 384-output decoder. Whether TRAINING + byte-closure
  realizes that at a small byte budget is the single binding unknown.

### C.3 The decisive break-even arithmetic (why the recommendation is a SMALL decoder)

At the **bc20 byte budget** with the **converged frontier pose** (2.93e-5), the break-even d_seg for
sub-0.15 is:

- int8 (89,136 B): **d_seg < 7.35e-4**
- int4 (46,590 B): **d_seg < 1.02e-3**

**PR95 measures d_seg = 5.6e-4** — below BOTH break-evens. So a PR95-class own-trained decoder at the
bc20 byte budget with converged pose **crosses sub-0.15**. The model's bc36 optimum is an artifact of the
power law assigning bc20 a d_seg of 0.0026 (its under-converged basin) instead of the ~5.6e-4 a fully
converged small decoder could reach. The existence proof + this break-even are the corrective: **prefer
the SMALL decoder at the rate-headroom operating point, not the big one.**

## PART D — The training-time Pareto (the E-axis the operator asked for)

S vs effective training budget at the optimum (C,T,Q) — both d_seg AND pose improve down the curve:

| effective epochs | d_seg | S |
|---:|---:|---:|
| 500 | 0.0042 | 0.556 |
| 1,000 | 0.00088 | 0.219 |
| 2,325 | 0.00086 | 0.211 |
| 10,000 | 0.00086 | 0.192 |
| 30,000 | 0.00086 | 0.180 |
| ∞ | 0.00086 | 0.179 |

**Min training budget to cross each S threshold** (at the optimum C,T,Q; conservative CE-only fit — the
curriculum's d_seg-finisher stages converge faster): sub-0.19 ≈ 844 epochs; sub-0.17 / sub-0.15
UNREACHABLE at bc36+int4 under the power-law d_seg (this is the model pessimism, not a physics statement —
see the break-even in C.3 for the small-decoder path that IS sub-0.15). The training-time axis tells the
operator the never-fired run's budget: ~1k epochs buys sub-0.19; full convergence (the 8-stage 29,650-epoch
curriculum) is needed to test the sub-0.15 small-decoder hypothesis.

## PART E — What's gated, and on what (the value that flips the optimum)

| gate | current | flips the optimum when… |
|---|---|---|
| **capacity α** (power-law exponent) | MODELLED [0.9, 1.5] | the clean bc36 anchor + its prune-path pins α; α↓ shifts C* down, α↑ up (deepmath C.2: 0.91→bc27, 1.12→bc29, 1.50→bc32). **THE dominant uncertainty.** |
| **QAT d_seg-hold spill** | 0.0003 ASSUMPTION | a measured spill > the int4 byte saving in S-units (~0.07 S at the optimum) flips Q* away from int4 to int8. |
| **taper T multiplier** | 1.0 (no A/B) | a measured `dseg_aware_taper` arm_b/control ratio < 1.0 lowers d_seg at ZERO byte cost → flips C* toward LOWER capacity (`dseg_aware_taper` already exists in the decoder, byte-matched). |
| **384 d_seg floor** | 1.875e-4 HARD | a ≥camera-res decoder dips toward 1.596e-4 but only buys ~0.003 S — not worth a resolution rebuild; capacity-within-384 is the larger lever. |
| **pose convergence τ** | MODELLED 6000 ep | a measured (epochs, d_pose) curve changes WHEN pose reaches the frontier value — the dominant sub-0.15 swing, so this gates the training-time Pareto shape. |

The solve is emitted as a FUNCTION of these gates, so the answer sharpens automatically when each lands.

## PART F — The prune-path capacity-RD tooling (ready for the clean anchor)

`plan_capacity_rd_prune_path()` emits the READY-to-run measurement plan that pins α cleanly: train ONE big
(bc36) decoder to convergence ONCE, then **structured-prune (L2 channel importance) + KD-finetune from the
big teacher** DOWN to each capacity rung (bc16/20/24/28/32), byte-close, and exact-score each. Every rung
shares the SAME teacher + recipe → apples-to-apples (params, d_seg) points, dissolving the contaminated
2-point fit. The per-rung predicted (d_seg, bytes, S) are emitted now; the runner contract (load ckpt →
prune → KD-finetune → byte-close → `upstream/evaluate.py` → fill measured columns → re-solve) executes the
instant the bc36 n600 checkpoint converges (deepmath E.1, the never-fired run). A measured d_seg ABOVE the
prediction at low capacity = capacity wall EARNED; BELOW = more sub-0.15 headroom than the model shows.

## PART G — The single recommended next config to train

**Train a bc20-class (≈83K-param) decoder to FULL convergence via the corrected PR95 8-stage curriculum,
int8, at n600 — the never-fired run (deepmath E.1).** Rationale (the math-optimal call, NOT the model's
naive bc36 argmin):
1. The bc20 rate+pose floor is 0.1178 (sub-0.15); sub-0.15 needs d_seg < 7.35e-4 at int8.
2. PR95 measures d_seg 5.6e-4 < 7.35e-4 → a converged small decoder is the cleanest sub-0.15 shot.
3. It is the cheapest decoder to converge + byte-close + exact-eval (the rate-headroom operating point).
4. If int8-converged d_seg lands in [5.6e-4, 7.35e-4] → sub-0.15 at int8; if it lands higher but the
   prune-path / `dseg_aware_taper` / int4-QAT levers bend it down, the joint solver re-solves to the
   exact next config. The bc36 model optimum is the AMBER fallback (the power-law-safe bet at ~0.18).

**Reactivation/kill:** GREEN if the converged bc20 byte-closed S < 0.15 (goal) or < 0.19110 (frontier
shift); AMBER if d_seg in [5.6e-4, 7.35e-4] but S in [0.15, 0.191] (pivot to int4-QAT / taper); RED
(capacity wall EARNED on solid ground) only if the FULLY-converged bc20 8-stage curriculum caps d_seg ≥
the int8 break-even AND the prune-path shows bc24/bc28 don't help either.

## 6-hook wire-in declaration (per CLAUDE.md "Subagent coherence-by-default")

1. **Sensitivity-map** — ACTIVE: the gating table (PART E) is a per-axis sensitivity surface (which axis
   moves S most per unit; pose-convergence + capacity-α dominate). Not persisted to `tac.sensitivity_map`
   (advisory mechanism, not a score-claim surface).
2. **Pareto constraint** — ACTIVE: the solver IS the joint Pareto solve over (C,T,Q,E); it composes the
   per-axis surfaces into the achievable-S frontier (the two-layer floor) for the campaign planner.
3. **Bit-allocator hook** — ACTIVE: the Q-axis int8→int-N byte fractions + the entropy-floor finding
   (rate needs a retrain not a recode, deepmath C.4) tell the allocator to STOP optimizing the order-0
   coder and pursue int4-QAT (the live lever).
4. **Cathedral autopilot dispatch** — ACTIVE (op-routable): PART G is the named exact-row-feeding dispatch
   (the converged bc20 8-stage run), de-risked by the prune-path.
5. **Continual-learning posterior** — ACTIVE: the solver + JSON are the anchors (the two-layer floor, the
   per-capacity table, the break-even arithmetic, the gating sensitivities) the next campaign inherits;
   the sister Q/E surfaces plug in via the tested ingestion interface.
6. **Probe-disambiguator** — ACTIVE: the existence-proof cross-check IS the disambiguator between
   "0.179 is the achievable floor" (REFUTED — it's a power-law artifact) and "sub-0.15 is permitted, gated
   on capacity-realization" (the physical floor 0.059 + the break-even). `tools/math_optimal_joint_solve.py`
   returns the regime-conditional verdict.

## Observability surface

Every number cites a file:field — the 384-floor JSON, the `capacity_rd_qat` measured anchors, the deepmath
memo's measured table, the frontier pointer, `reports/math_optimal_joint_solve.json` (the full machine-
readable solve), and the 39-test suite. Axis `[contest-CPU advisory] / [macOS-CPU advisory]`,
score_claim=false, pointer UNMOVED 0.19109982.

## Canonical-vs-unique decision per layer

- **Score arithmetic:** ADOPT_CANONICAL `tac.contest_score` (the compliance bedrock; never hand-rolled).
- **Capacity anchors + byte model + C×Q solve:** ADOPT_CANONICAL `tac.capacity_rd_qat` (the existing 1D
  solve; this module EXTENDS it — no fork).
- **The E/T/pose-convergence/384-floor axes + the existence-proof cross-check + the prune-path plan:** NEW
  (UNIQUE) — these axes did not exist in the C×Q desk model; they are the joint-solver's contribution.
- **Decoder param count + taper:** ADOPT_CANONICAL `tac.torch_vehicle.configurable_taper_decoder`.

## NO-FAKE ledger

- **MEASURED:** the 384 d_seg floor (N=600), the bc20/bc36/frontier/PR95 d_seg anchors, the int8→int-N
  byte ratios, the (epochs, d_seg) convergence anchors, the basin/frontier pose — all from cited artifacts.
- **DERIVED (clearly labeled):** the capacity power-law α (2-point fit, under-converged → lower bound),
  the convergence τ (2-point exp fit), the pose τ (sparse), the int4 d_seg-hold spill (ASSUMPTION). The
  S-values are all via `tac.contest_score` on these inputs.
- **NOT claimed:** no score moved; pointer UNMOVED 0.19110; no promotion; no exact row produced; no
  training run launched. This is a MEANS memo + a config recommendation. The OUTPUT is what the clean
  anchor + its prune-path should BECOME.
- **Existence-proof applied:** the solver's own "lower bound" was cross-checked and downgraded from a
  floor to a capacity-realization-limited model artifact — the discipline ran on this very memo's headline.
