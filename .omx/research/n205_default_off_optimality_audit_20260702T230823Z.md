---
title: "#205 PRE-LAUNCH GATE — DEFAULT-OFF OPTIMALITY-COMPLETENESS AUDIT (is the config leaving beneficial levers OFF?)"
date: 2026-07-02
axis: "[macOS-MLX / CPU advisory / design] — NON-PROMOTABLE. means != ends: this audits a launch-ready CONFIG (a MEANS). The ONLY end is a byte-closed n600 exact row < 0.19110 from upstream/evaluate.py (contest-CPU/CUDA, NEVER MPS). Pointer 0.19110 UNMOVED — this audit does NOT move it."
pointer: "0.19110 UNMOVED (contest-CPU recoded-R3)"
scope: "OPTIMALITY-COMPLETENESS lens (INVERSE of the store-nothing orphan): every DEFAULT-OFF / neutral-default lever under the #205 SEALED config, classified A (correctly-off measured-negative) / B (research/not-ready) / C (SHOULD be ON — measured/grounded win orphaned by omission). Sister agent a6eb1078 owns correctness/runnability/honesty. READ-ONLY: no trainer/config/triality edits."
provenance: "git HEAD ccfeccc1b. sealed_205 argv from tools/launch_witness_run.py --config sealed_205 --dry-run (83/83 flags validated). Full 173-action argparse dumped via parse_args intercept. Diff = 90 default-OFF flags. Evidence: measured_lever_inventory_for_synergy_pass_20260701, n205_phase3_recursive_adversarial_review_verdict_20260702 (SEAL), capstone_witness_launch_config_deepmath_optimal_20260702, aa_feasibility_reconciliation_20260702, wave_f_unified_xi_build_measured_20260702, signal_processing_filter_levers_derived_20260701, orphan sweep (tools/audit_orphaned_measured_wins.py), canonical_equations_registry."
---

# #205 default-OFF optimality-completeness audit

**BOTTOM LINE (read first):** the #205 SEALED config **IS optimal-form for the TRAINER-LAUNCH
surface**. Every MEASURED byte-free d_seg win is ALREADY ON (directional/curvelet basis −48%, SDF+hosc
chart, chroma, palette-anchor, structured-init, lane-prior-φ1, persistence, island-amplify, eikonal,
length, self-orient). **Bucket C (a measured/grounded win orphaned by omission at the launch surface)
is EMPTY** — there is NO trainer-launch lever that is one cheap $0-n600 measurement away from being
turned on for the first exact row. The real headroom is entirely POST-GO: (a) the θ* surgical margin
levers as warm-start A/Bs off the #205 per-stage ckpts, and (b) the #202 byte-close RATE levers
(store-nothing keyframe, lane-band coding, L13 format) applied to the TRAINED witness — **all of
which are gated ON THE #205 RUN ITSELF and therefore CANNOT be made ready pre-GO** (not a hold
reason). The config IS optimal-form modulo the 2 deliberate first-row deferrals in §Reverse
(mod-dim 19-vs-32 rate slack; β₂ 0.999-vs-derived-0.9999999) — both shape/optimizer-changing, both
correctly deferred to their own post-run arms (#223 / #222).

**One-line verdict: GO. Nothing measured is orphaned OFF at the launch surface; fire #205, then
harvest the per-stage ckpts for the θ* warm-start A/Bs + do #202 byte-close with the rate levers.**

---

## 0. The lever surface (what was audited)

- **173** total trainer options; **83** ON in the sealed_205 argv; **90** default-OFF / at-neutral.
- Of the 90 default-OFF: **~16 real lever GATES** (master switches: boolean toggles + weight-0 gates)
  carry a score decision → classified A/B/C below. The other **~74** are (i) SUB-PARAMETERS inert when
  their parent lever is off (e.g. `--margin-saliency-target` when `--margin-saliency-weight 0`), (ii)
  θ* tuning knobs (θ* TIER unbuilt), (iii) resume/mode knobs N/A for a from-scratch launch, or (iv)
  score-neutral OOM/perf knobs already at their engaged defaults. Grouped in §4.
- **Determinism note:** MLX-GPU is NOT bit-identical cross-process (MEMORY
  `mlx_gpu_not_bit_identical_crossprocess...`); any "speed" lever that changes the numeric path
  (e.g. `--gpu-reorient`) is a determinism risk, not a free S-neutral win.

---

## 1. THE LEVER-GATE CLASSIFICATION TABLE (the 16 that carry a decision)

| lever (default-OFF) | #205 | MEASURED/PREDICTED evidence + receipt | bucket | recommendation |
|---|---|---|---|---|
| `--margin-saliency-weight` (LEVER-4) | 0 (off) | **PREDICTED −0.0003..−0.0008** through-R n200 θ* A/B, **UNTESTED** (`measured_lever_inventory` A1). Amplify rides the SAME `_signed` mechanism WITHOUT it (deepmath-cfg §"Also OFF"; `_seg_levers_on` L1654 — amplify alone triggers the shared forward, NO no-op). Sub-additive with lane-thin (both hit Lane). | **B** | POST-GO warm-start A/B off #205 ckpts; measure NET over the ON amplify+persistence stack. Not pre-GO. |
| `--lane-thin-weight` (LEVER-B) | 0 (off) | **PREDICTED −0.0004..−0.0010** θ* A/B, **UNTESTED**; targets the MEASURED dominant residual (dashes <5px = 93% missed, birth-death). "#1 predicted per GPU-h" (`measured_lever_inventory` A2). | **B** | POST-GO warm-start A/B (θ* rank-1). Not pre-GO. |
| `--lane-edge-weight` (LEVER-3) | 0 (off) | **PREDICTED −0.0001..−0.0004**, **DOMINATED by all-class A1**, ABLATION-only; assumption-challenge: up-weighting lane may TRADE the 50% Road majority (`measured_lever_inventory` A6). | **B** (near-A) | POST-GO ablation only; likely kill (dominated). Not pre-GO. |
| `--hardness-oversample`/`--hardness-weighted` (LEVER-5) | 0 / off | **PREDICTED −0.0001..−0.0004**, modest (GT-margin per-pair spread only 1.31×) + `+~50% wall-clock tax` (`measured_lever_inventory` A4). | **B** | POST-GO, low priority (modest + wall-clock tax). |
| `--margin-saliency-uniward` (D8/A7) | off | **PREDICTED −0.0001..−0.0003 marginal over A1**, REQUIRES A1>0, late-stage (`measured_lever_inventory` A7). | **B** | POST-GO wave-2 (requires margin-saliency on first). |
| `--film-stiefel` + `--code-spectral-entropy-weight` (DM1/A5) | off / 0 | **MEASURED $0: PR(M) collapses 2.6× WHILE d_seg IMPROVES 1.9× → DEMOTED 2nd-order** (per-pair FiLM can't localize the moving annulus); nonlinear-ID m~9 << 26 → capacity already adequate; primary value is n600 AMORTIZATION not d_seg (`measured_lever_inventory` A5; autoconfig `film_stiefel_dm1`). | **A** | Correctly OFF (measured-demoted). Compose ONLY if FiLM collapse ever becomes binding. |
| `--film-per-layer` / `--film-concat-code` (capacity A1/A2) | off | Capacity ADD (+params, +rate). Nonlinear-ID m~9 << mod-32 → capacity already adequate; D2 capacity-routing MEASURED −70% BUT **+49–63KB rate (+0.033..+0.042 S)** and capacity-alone-on-isotropic HURTS +6% (`measured_lever_inventory` D2). | **B** (near-A) | Correctly OFF: rate-costly, capacity adequate. Likely a non-problem. |
| `--film-rank-floor-weight` (A3) | 0 | **DOMINATED by `--film-stiefel`; NOT recommended** (review FEED-ht/M1). | **A** | Correctly OFF (dominated). |
| `--margin-field-head-weight` + `--head` + `--additive-margin` + `--logit-adjust-per-class` (#218) | 0 / softmax / 0 / off | margin-field WIRED but **net-positivity UNCONFIRMED** (in-flight #218 Laguerre/ETF sweep); `etf` head advertised "byte-free + rate-win" but UNMEASURED on the witness (deepmath-cfg §"Also OFF"). | **B** | POST-GO: the #218 net-positivity sweep decides (`etf` head is the rate-win to test). Not pre-GO. |
| `--code-nuclear-weight` (θ* MUST-2, RATE lever) | 0.0 | low-rank code → rate; θ* TIER-3 STRUCTURAL UNBUILT / DERIVE-only (build-state; `design_refine_thetastar_residual_inr`). | **B** | POST-GO (θ* build + trained arm). A rate lever → belongs at #202 byte-close time. |
| `--seed-islands` (#224/#208) | off | **FAIL-CLOSED `NotImplementedError` L1631** (protected seed-residual param + grad-shield restructure not wired); **`--amplify-weight` is the WIRED substitute** (AMPLIFY_ONLY rides `_signed`) and IS ON (deepmath-cfg §"Fail-closed EXCLUDED" #2). | **A** | Correctly OFF (fail-closed; superseded by the ON amplify path). |
| `--aa-supersample` / `--aa-self-orient-fine-mode` (render-aa supersample) | 1 / refuse | **MEASURED −49% HURT** on the witness (`levelset_render_side_sizing_l7best_n600`: c1 0.00333 → c2_aa 0.00496; re-confirmed phase-3 §1b) + **fp64 decode 41.3min > 30min budget** + neither shipped inflate applies ss (train/decode MISMATCH = a FAKE optimization) (`aa_feasibility_reconciliation`). Config ships `--render-aa none` + analytic `--lane-render-band` instead. | **A** | Correctly OFF (measured-negative on TWO independent grounds; DISQUALIFIED). |
| `--render-aa ipe` (vs `none`) | none | ipe is a WEAK secondary AA ("only SMOOTHS the basis"), O(1) decode-safe, **UNMEASURED to beat `none`+analytic-band on the witness** (`aa_feasibility_reconciliation`; measured_lever_inventory AA table). The config's `none`+analytic-band is the MEASURED-to-help path. | **A/B** | Correctly `none`; `ipe` is the documented alt only if a full-partition AA is ever wanted — NEVER supersample. No measured reason to switch. |
| `--residual-mode` (v2 hybrid) | off | The v2 fixed-bulk (+) INR-residual vehicle; #205 is the from-scratch INR vehicle, NOT v2. | **B** | Correctly OFF for #205 (different vehicle; v2 is a separate track). |
| `--gpu-reorient` (FEED-eo) | off | Throughput lever (reorient argmax on GPU vs numpy-CPU). GPU-fp32 ≠ numpy-fp32 bit-identical → determinism risk; NOT a free S-neutral win. | **A** | Correctly OFF (determinism; speed = lexicographic-secondary, only take if bit-identical, which this is not). |
| `--dm1-telemetry` | off | Observability-only (PR(M) live+shadow row); no score effect. | **A** | Correctly OFF (telemetry; optionally ON for observability, S-neutral). |

---

## 2. BUCKET C — levers that SHOULD be ON for optimal S but are OFF (the orphan-by-omission list)

**EMPTY at the trainer-launch surface.** After auditing all 16 lever-gates + the 74 sub/rate/mode
knobs against the measured ledger, there is **no MEASURED (or well-grounded byte-free) d_seg or rate
win that is a TRAINER-LAUNCH flag and is left OFF and could be flipped ON pre-GO to improve the first
exact row.** The reasons, positively stated:

1. Every MEASURED byte-free d_seg win is **already ON**: directional/curvelet basis (−48% n96 / −31%
   n600, `--self-orient`), SDF+hosc chart (D6), chroma (D9, baked-in), palette-anchor, structured-init
   + lane-prior-φ1 (rule-118 FREE), persistence/clDice (111× erasure-sensitive), island-amplify,
   eikonal/length. All confirmed in the sealed argv.
2. Every remaining d_seg lever that is OFF is **PREDICTED, not MEASURED** (the θ* surgical margin
   family A1/A2/A6/A4/A7) → they cannot be promoted to "turn ON now" without a trained through-R A/B
   (no $0 n600 flips them to a measured win), AND their proper form is a warm-start off the #205
   per-stage ckpts (gated on the run) → bucket B, §3.
3. Every measured RATE win (L13 −59%, lane-band coding −42%, keyframe/store-nothing, QAT −4.4%) is a
   **#202 BYTE-CLOSE lever applied to the TRAINED witness** — NOT a trainer-launch flag → it cannot
   be "on in the argv"; it is applied after the run (§3). The store-nothing keyframe win is ALREADY
   wired as the `store_nothing_205` A/B arm.

This EMPTY-bucket-C is the load-bearing finding: **the config is complete on the launch surface.**
(The only $0 pre-launch de-risk that existed — horizon 174-vs-188 — was already run in phase-3 §1a and
FALSIFIED the proposed change, confirming the shipped 174. There is no analogous open $0 flip.)

---

## 3. BUCKET B — research / not-ready, with MAKE-READY-FOR-#205 EV ANALYSIS (per coordinator)

For each: (i) WHY not ready (exact blocker), (ii) COST to make ready, (iii) MAKE-READY-PRE-GO verdict.
**Ranked by (expected/measured ΔS ÷ make-ready-cost), highest-EV first.** Honest framing up front:
**every bucket-B lever's readiness is gated on artifacts the #205 run PRODUCES** (a trained witness /
per-stage ckpts / a byte-closeable archive). None is one cheap $0-n600 measurement away from bucket-C.
So the "make-ready-pre-GO?" verdict is **NO for all** — but they rank sharply by post-GO harvest EV.

| rank | bucket-B lever | (i) WHY not ready (blocker) | (ii) make-ready COST | (iii) MAKE-READY-PRE-GO? verdict |
|---|---|---|---|---|
| **1** | **store-nothing keyframe rate** (`store_nothing_205` arm: `--pose-carrier-source generated`) | Needs the TRAINED witness frame0 INR + trained dξ residual to measure whether store-nothing reaches the table-carrier's d_pose. **MEASURED byte-close BIT-EXACT: section 697941B(table)→1049B(store-nothing)** — this is the named sub-0.15 rate blocker (phase-3 §4: keyframe payload +0.006/10s, larger full-clip). | **Intrinsically gated on #205** (needs the trained witness). Already WIRED as the A/B arm (autoconfig `derive_store_nothing_205_config`). | **NO — but RUN IT as the A/B arm alongside sealed_205.** Highest post-GO EV: the keyframe payload is the §4-flagged sub-0.15 blocker; store-nothing collapses it ~700×. Not a pre-GO flip; it is a same-launch A/B. |
| **2** | **lane-band RATE coding** (LBND2 smoothing, task-λ) | #202 byte-close lever; net-S is a #205 A/B (applied to the trained band). **MEASURED −42%/−46% (LOSSY, moving-avg 0.01489 near R-D frontier); correspondence=0.5% lossless** (`wave_f_unified_xi_build_measured`; MEMORY corrected: magnitude data-dependent, jitter-not-swaps). | **Intrinsically gated on #205** (byte-close the trained witness band). | **NO — #202 byte-close, net-S #205 A/B.** Not a trainer-launch flag. Harvest post-GO. |
| **3** | **L13 non-RGB witness format −59% rate** | #202 byte-close format; needs the trained witness to encode. **MEASURED lossless-parity 177,169→72,217 B** (`canonical_research_index_rate` R4). | **Intrinsically gated on #205** (byte-close). | **NO — #202 byte-close.** The rate half of sub-0.15; apply post-GO. |
| **4** | **θ* surgical margin: margin-saliency (A1) + lane-thin (A2)** | PREDICTED −0.03..−0.10 ΔS, UNTESTED; sub-additive with the ALREADY-ON amplify+persistence; net-positivity when STACKED is unmeasured. | **Warm-start GPU A/B off #205 per-stage ckpts** (~0.6–1.5 GPU-h each; loss/projection-only, shape-compatible, no new params). Gated on the #205 ckpts existing. | **NO — POST-GO warm-start.** Their PROPER form (deepmath-cfg lever_priors: "re-treat as warm-start, no new params"). Measure NET over the amplify stack; do NOT stack blind pre-GO (confound + Road-trade risk). |
| **5** | **#218 margin-field head / etf / logit-adjust** | WIRED but net-positivity UNCONFIRMED; `etf` head is a claimed byte-free rate-win, UNMEASURED on the witness. | **The in-flight #218 Laguerre/ETF sweep** (a trained GPU arm). | **NO — POST-GO #218 sweep.** "if net-positive else off" (parent's rule); no $0 pre-GO resolution. |
| **6** | **code-nuclear low-rank RATE (θ* MUST-2)** | θ* TIER-3 structural UNBUILT; DERIVE-only, no through-R n600 row. | **Focused local build + a trained arm** (θ* residual-INR stack). | **NO — POST-GO.** A rate lever → #202 byte-close time; θ* build first. |
| **7** | **capacity FiLM (`--film-per-layer`/`--film-concat-code`)** | Adds params + rate; capacity likely already adequate (nonlinear-ID m~9 << mod-32); D2 shows capacity-alone HURTS unless basis-first (which is ON). | **A trained arm** (shape-changing). | **NO — POST-GO, LOW priority.** Likely a non-problem (rate-costly for adequate capacity). |

**EV synthesis:** ranks 1–3 (RATE: store-nothing / lane-band / L13) are MEASURED byte-close wins that
are the **rate half of the sub-0.15 path** and are gated on having a trained witness → the #205 run is
the prerequisite, and store-nothing is a same-launch A/B (fire it). Ranks 4–7 (θ*/#218/capacity) are
PREDICTED d_seg/capacity levers whose proper form is a warm-start/sweep off the #205 ckpts. **No
bucket-B lever is a cheap $0 pre-GO close; the correct action for ALL is: launch #205, then harvest.**

---

## 4. BUCKET A — correctly OFF (measured-negative / neutral / inert / N/A), grouped

- **Measured-negative / disqualified:** `--aa-supersample`/`--aa-self-orient-fine-mode` (−49% HURT +
  decode-over-budget + train/decode mismatch); `--film-stiefel`/`--code-spectral-entropy-weight`
  (PR-collapse-while-d_seg-improves → demoted); `--film-rank-floor-weight` (dominated); lane ξ-coding
  (REFUTED, Pareto-dominated on swap-light clip — phase-3 §6, not a flag).
- **Fail-closed / superseded:** `--seed-islands` (NotImplementedError; superseded by the ON amplify).
- **Determinism / observability / speed-not-free:** `--gpu-reorient` (GPU≠numpy bit-identity),
  `--dm1-telemetry` (telemetry-only).
- **Score-neutral OOM/perf at engaged defaults (correctly not flipped):** `--mlx-cache-clear-accum 1`,
  `--verdict-batch 32`, `--lr-schedule` (default True → ON).
- **SUB-PARAMETERS inert when parent lever OFF (no decision):** `--margin-saliency-{target,tau,
  start-epoch}`, `--lane-thin-{radius,target,class,start-epoch}`, `--lane-edge-{class,target,margin,
  start-epoch}`, `--hardness-{band,power,source}`, `--logit-adjust-tau`, `--additive-margin`,
  `--margin-target-end`, `--hinge-weight`, `--l7-{mult,threshold}` (l7 demoted), `--spike-factor`,
  `--seed-{blend,lr}`, `--containment-*`, `--eikonal-junction-*`, `--code-nuclear-{ns-iters,eps}`,
  `--ema-decay-finisher*`, `--tau-hold-frac`, `--muon-{adamw-lr,weight-decay}`, `--aa-ipe-footprint`,
  `--bank-*` (curvelet bank shape; self-orient uses the proven n-dir-freqs/freq-across/along),
  `--pose-carrier-{pitch,s-r,s-t,fit-pairs,residual-scale}` (pose-carrier IS on; sub-params at
  measured-optimal defaults per #224), `--wire-{s0,w0}`, `--warmup-epochs`, `--pose-eps`,
  `--structured-init-{lr,steps,subsample,thresh,sdf-clip}` (structured-init IS on; tuning at proven
  defaults). These are inert or at proven-defaults; no orphaned signal.
- **Resume / mode knobs N/A for a from-scratch launch:** `--resume-from`,
  `--resume-allow-lever-drift`, `--freeze-decoder-fit-codes`, `--residual-target-npz`,
  `--anneal-epochs`.

---

## 5. REVERSE DIRECTION — is any ON lever set to a value a measurement says is WRONG?

Completeness both directions. The phase-3 SEAL already adjudicated the two material tensions; both are
DELIBERATE first-row deferrals, not defects:

1. **`--mod-dim 32` (ON) vs 19 (the deepmath-optimal doc's rate-saving Whitney floor for measured
   m~9).** The `capstone_witness_launch_config_deepmath_optimal` doc itself computes **19** as
   deep-math-optimal (rate is the binding sub-0.15 term; 19 = Whitney floor for measured m~9). The
   SEALED verdict OVERRODE to **32** (proven-arm; d_seg BINDING; 19's d_seg-NEUTRALITY is UNMEASURED;
   rate has slack 0.055<0.081). **This is the single most material reverse-direction item: 32 may leave
   rate on the table.** But mod-dim is SHAPE-changing → 19's neutrality cannot be measured without its
   own trained arm. Correctly deferred: #223 byte-close sweep folds 32→26→19 ONLY if measured
   d_seg-neutral. NOT a pre-GO fix. (Honest flag for the coordinator: if first-row S lands just above
   0.15 and rate-bound, the 32→19 fold is the first rate lever to pull via #223.)
2. **`--adam-beta2 0.999` (ON) vs the DERIVED optimum 0.9999999** (the deepmath doc's small-n
   `1−β₂*≈1.12e-7`). The SEALED verdict chose **0.999** (== MLX default → byte-identical, no
   bias-correction confound on the first attribution row); 0.9999999 is flagged a MIS-ANCHOR for a
   first row (risks a ~100× step-1 LR blowup via bias-correction gating). **Reverse-direction watch:
   the DERIVED β₂ optimum is deliberately left off the first row for confound-cleanliness.** Correctly
   deferred to the #222 optimizer-vs-representation sweep. NOT a pre-GO fix.

No other ON lever contradicts a measurement. persistence-weight 1.0 / amplify-weight 1.0 / lane-band-*
are ON at ENGAGE (un-tuned T2) values — a calibrate-IN-RUN item (the optimum only exists inside the
descent; closed-loop ROLLBACK is the safety net), not a "should be OFF."

---

## 6. Cross-refs + triality

- Orphan sweep (`tools/audit_orphaned_measured_wins.py`): the high-priority "ORPHAN" memos are either
  ALREADY-WIRED (directional basis, pose-carrier, persistence, amplify) or #202 BYTE-CLOSE rate levers
  (L13, lane-band, keyframe) — confirmed NOT trainer-launch flags. No launch-surface orphan.
- Canonical equations backing the ON/OFF decisions all exist: `l7_linf_sharpening_defect`,
  `analytic_lane_render_band_fp_reduction_v1`, `island_finest_scale_protection_survival_v1`,
  `persistence_topology_cldice...`, `oracle_r_dseg_floor...`.
- Signal-processing "bake into optimal-form run" wins (`signal_processing_filter_levers_derived`):
  L3 NTK band-pass whitening = a SPEED lever (convergence, not d_seg-floor), UNBUILT/not-a-flag →
  bucket B, lexicographic-secondary (only take if free; it needs a build). L4/L5 = margin-saliency/
  uniward (bucket B, §3). L2-phase = inflate-side (#202). None is a measured d_seg-floor flag left off.

**means != ends:** this audits a MEANS. The pointer (0.19110) moves ONLY through the byte-closed n600
`upstream/evaluate.py` exact row. The config is optimal-form for the launch surface; the harvest
(θ* warm-start A/Bs + #202 rate byte-close) is post-GO.
