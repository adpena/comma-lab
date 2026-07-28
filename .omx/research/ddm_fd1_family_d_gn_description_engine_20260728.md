# ddm_fd1 — family-d: Gauss-Newton/CG in DESCRIPTION coordinates (the named build)

**Date:** 2026-07-28 · **Arm:** `ddm_fd1_20260728` · **Charter:** the gc5-adjudicated fork's build arm
**Evidence axis:** `[macOS-CPU frozen-scorer advisory]` (S0) + `[macOS-MLX research-signal]` (S1/S2 proposals);
every accepted S2 point realized through archive parse-back + uint8 + R + frozen CPU scorers.
**`score_claim=false · promotion_eligible=false`. Canonical frontier pointer 0.1910828242 UNMOVED.**

**STORES CONSULTED:** `CLAUDE.md` (NO-FAKE #6 solver-not-search; measured-scored-quantity; SSD-first;
resumability P0; serializer discipline) · `MEMORY.md` current-state
(`box_retired_min_s_target...20260728`, `pose_is_a_terminal_six_equation_solve...`, `objective_is_min_S_over_solution_set`,
`opportunity_pools_non_additive`, `realization_is_quantization_gated`, `verdict_rules_measured_pace_reported_direction`) ·
`.omx/research/ddm_rp1_rangeA_cell_realized_probe_20260728.md` (CELLS HOLD; boxsolve = named next measurement) ·
`.omx/research/council_gc5_schmidhuber_micro_macro_bridge_20260728.md` (family-d naming; capacity fork; Q2 E-cell
reformulation) · ws2/ws3 receipts (`ddm_ws2_warm_start_custody_producer_receipt_20260724.json`,
`ddm_ws3_warm_start_slope_arbitration_receipt_20260724.json` — KEEP_WJOINT) ·
`ddm_ms4d_.../BUNDLE-COMPLETE.json` via the fail-closed loader (COMPLETE, 4/4 components, n600) ·
`ddm_ms2r_r3_box_tolerance_solve_.../04_candidate/candidate.json` (box archive sha e3d0581f…) ·
`r6cal_solved_object_byteclose_eval_20260727.md` (inflate raw custody 32a773a2…) · sched1 (event-continuation) ·
j5/ws3 sealed tickets (`ddm_ws3_w_joint_history_fill_20260724.json`, typed hash 346975b2… = the W_joint checkpoint's).

---

## Verdict first

1. **S0 — THE HOLD SURVIVES AT THE BOX-SOLVE OPERATING POINT (rp1's named next measurement, CLOSED).**
   Receiver-closed inflate of the 277.7 MB box-solve archive reproduced r6cal's raw custody bit-identically
   (raw sha `32a773a2…`). The identical C1 probe (range-carrier zero-ker uint8 lift) on the real box-solve
   frames, n600 through the real frozen SegNet/PoseNet: **the band is UNCHANGED from the GT probe** —
   C1-vs-C0 cell-hold flip rate **3.757e-4** (GT substrate: 3.630e-4), margin-absorption mechanism
   invariant (flips at pre-round margin 0.0337 vs held 5.573 = **165×** gap; rp1 GT: 166×), C1 d_seg vs
   lstars 1.2492e-3 = **1.077× the box baseline** (C0 = 1.1600e-3, reproducing the known box-solve
   baseline exactly), pose collateral +1.1% (1.663e-2 → 1.681e-2). **No re-scope of the build target:**
   the engine's realistic realization-noise band is ~3.6–3.8e-4 cell-hold per range-carrier uint8
   realization, at BOTH margin operating points measured so far.

2. **S1 — THE ENGINE IS BUILT AND IS A REAL SOLVE (NO-FAKE #6).** `FamilyDGaussNewtonEngineV1`
   (`src/tac/optimization/ddm_family_d_gn_description.py`) extends the j2 MLX module (subclass exposing the
   seg-logit feature level; render/loss/acceptance substrate untouched) with matrix-free Gauss-Newton/CG:
   exact GGN `100·JᵀH_CE J/N` through paint → uint8-STE → fused-R → frozen SegNet via `mx.jvp`+`mx.vjp`,
   damped normal equations solved by preconditioned CG. It enumerates nothing; there is no candidate menu.
   **Smoke (block 447–450, K=8 realized secants, 344 active params):** one damped GN/CG proposal
   (3 CG iters, rel. residual 0.50, 6 HVP calls at **0.33 s/HVP**, propose 2.0 s) yields a step whose
   FULL application reduces the exact block objective **27.015 → 23.816 (−11.8%)** — measured through
   the STE forward, not the model. Model reduction 14.53 vs realized 3.20 (ratio 0.22 — the shallow
   Rayleigh curvature 3.85e-4 over-extends the raw step; the multiplier ladder + v19 realized
   acceptance govern, as designed). `mx.jvp` is NOT implemented for the fused-R CustomKernel — the
   engine measures Js·v as a central secant of the smooth surrogate at the parameter-quantum scale
   (ε=0.5, the same linearization scale as j2's own realized ±1-quantum secants), with the exact
   reverse-mode `mx.vjp` transpose. Receipt: `.omx/research/ddm_fd1_gn_engine_smoke_20260728.json`.

3. **S2 — BOUNDED GOVERNED GN WINDOW FROM W_JOINT: RAN TO COMPLETION, ZERO ACCEPTED STEPS — a
   clean instrument reading, not a crash.** 2 GN steps × 3 multipliers = 6 realized candidates,
   every one REJECTED by the unchanged v19 joint gate. The mechanism is crisp and repeatable
   (both steps classified `BLOCK_LOCALITY_OR_REALIZATION_GAP`): the second-order solve is REAL
   on its block (exact block objective −7.0%…−12.7% per proposal through the STE forward) but
   (i) **cross-pair seg transfer is exactly ZERO** — realized n600 d_seg bit-identical to
   baseline (0.0702156745…) in 5/6 candidates — and (ii) **pose collateral of the shared
   template DOF prices every candidate out** (+2.8%…+13.1% implied d_pose) at a warm start
   where the unsolved pose leg is 72.8% of the joint action. Receipt:
   `s2_gn_window/fd1_gn_window_receipt.json` (SSD).

4. **SEAL: TYPED BLOCKER — `BLOCKED_ZERO_ACCEPT_WINDOW_CAPACITY_ROUTED` (fire owner MAIN;
   ticket mechanically valid, custody resealed).** The fire ticket's own `continue_while`
   ("latest window has ≥1 accepted step") is violated by the FIRST measured window — firing the
   long campaign as-is would halt by its own event rules within 1–2 windows. NO-FAKE honesty: a
   READY seal on a config measured to not descend would be a fake readiness claim. The blocker
   routes to the gc5 capacity disambiguator's TWO-RUNG ladder (§Capacity verdict); the ticket
   (`ddm_fd1_gn_campaign_fire_ticket.v1`, execution custody refreshed to the final worktree
   shas, RFC8785 semantic program re-hashed) remains the mechanical fire surface for the
   descendant that clears the blocker.

## S0 — the box-solve C1 probe (n600, measured)

Custody chain: box-solve archive sha `e3d0581f…` (ms2r_r3 candidate receipt) → receiver-closed
`inflate_archive` (v10 production receiver, write-once per-pair stages) → raw sha `32a773a2…` **bit-identical
to r6cal's 07-27 inflate custody** → per-pair stage reads for the probe. Probe = the UNCHANGED rp1 C1
construction (`Y=round(clip(project_range(X)))`, #580 zero-ker min-norm camera preimage) on the real
box-solve frames, with the box-solve C0 forward as the operating-point reference (its argmax = the cells the
lift must hold; its margins = the pre-round margin field).

| quantity (n600, real frozen SegNet/PoseNet) | box-solve substrate | GT substrate (rp1) |
|---|---:|---:|
| C0 d_seg vs lstars (baseline custody) | **1.1600e-3** (= known box baseline) | 0 (defines cells) |
| C1 d_seg vs lstars | 1.2492e-3 (**1.077× C0**) | 3.6296e-4 |
| **C1-vs-C0 cell-hold flip rate** | **3.757e-4** (p90 5.90e-4, max 4.29e-3) | 3.630e-4 (≡ lstars flips) |
| pre-round margin: flipped vs held | 0.0337 vs 5.573 (**165×**) | 0.034 vs 5.61 (166×) |
| d_pose | 1.663e-2 → 1.681e-2 (+1.1%) | 0 → 3.965e-4 |

**Band verdict:** the rp1 verdict_scope caveat ("smaller-margin operating point would flip somewhat more")
is CLOSED — the cell-hold band is invariant across both operating points (3.63e-4 ↔ 3.76e-4) because the
flip population is governed by the near-zero-margin site fraction, which the box solve did NOT expand (it
spent tolerance budget, not margin sign). The +7.7% d_seg-vs-lstars delta is the union of realization noise
with the box-solve's own residual, not a new failure mode (per-class flip mass Road-dominant, same shape).
Artifacts: `/Volumes/VertigoDataTier/pact/ddm_fd1_20260728/chunks/chunk_boxsolve_*.json` (schema
`ddm_rp1_rangeA_cell_probe_chunk.v2`) + `s0_boxsolve_band_receipt.json` + `tools/aggregate_ddm_fd1_s0.py`.

## S1 — engine design + metric-custody honesty

- **The solve.** Seg-leg objective (pose_objective_weight=0 per the #383 terminal-pose law; pose collateral
  + rate priced by the UNCHANGED v19 realized acceptance): `f(θ)=100·(CE+w_m·hinge)(logits(θ))`. GN curvature
  `H = 100·JᵀH_CE J/N` with `H_CE = diag(p)−ppᵀ` frozen at θ₀ (PSD); hinge is curvature-0 a.e. and enters the
  RHS exactly via j2's `loss_and_grad`. Damped normal equations `(H+μ·diag(P))δ = −∇f` solved matrix-free by
  CG; trust region = the launcher's unchanged multiplier ladder (1.0, 0.5, 0.25) + damping adaptation
  (×0.5 on accept, ×4 on saturation, ×0.25 on quantization-gated rejection).
- **Scorer-metric custody (ms3/ms4).** The ms4d bundle loads COMPLETE through the fail-closed loader
  (4/4 components, n600, seg rank-4 head + margin-Fisher + pose ≤6-dim quadratic + composite-R second order)
  and is recorded in every proposal's diagnostics. HONEST DEGRADATION (charter-sanctioned): the bundle's
  atlas dimensions do not index the j5/v15 lift parameters, so the per-parameter preconditioner is NOT read
  from the bundle — it is the **measured Hutchinson Jacobi diagonal of the exact GGN operator itself**. The
  exact GGN through the frozen SegNet IS the scorer-metric pullback (the custodied rank-4 head Gram is
  contained in it by construction); no metric custody is fabricated.
- **Rate (ms1 contract).** The description byte delta is a coder staircase, locally constant under fp32
  perturbations below the re-emit quantum → no smooth term in the CG quadratic; it is priced at REALIZATION
  by v19 (advisory action includes `25·bytes/37 545 489`), which is exactly the 1.273108 B/error water-level
  exchange law in acceptance form.
- **Triality/DSL note:** the fd1 controls are launcher argv (`--fd1-*`), sealed inside the fire ticket's
  RFC8785-hashed semantic program; they are NOT witness-trainer levers (the witness DSL holds trainer levers;
  the j2/fd1 line is config-ticket-governed per the j-series convention). Registered as the ticket, not as
  `Lever` factories — same convention as j1–j10.

## S2 — bounded governed GN window (measured descent curve)

Config: `--fd1-gn-window` from the arbitrated W_joint step-4 checkpoint (ws3, sha-bound), ws3 ticket
typed identity `346975b2…` (run_identity verified MATCH on all 4 hashes), pair block 447–450
(train_batch 4, stage-0 active groups `island_worldsheet`+`shared_template_dof`, 344 active params),
CG 6 iters, damping 1e-3 initial, 4 Hutchinson probes, multipliers (1.0, 0.5, 0.25), v19 chunked
n600 acceptance (batch 32 ≤ the 120-pair chunk law). Memory: projected 15.75 GiB (ws3 measured
values through the live governor, ADMIT at headroom 52 GiB), **measured peak RSS 12.03 GiB**.
Total window 3,139.6 s. Resume-blocker fixed en route: the predecessor's S2 died on
`FrontierPointerCorruptError` (the worktree lacked the gitignored canonical frontier pointer that
the witness verdict module strict-loads at import); copied MAIN's live pointer into the worktree —
deterministic, value-identical fix.

**Measured curve (every point realized through compile → parse-back → uint8 → R → frozen CPU
scorers, n600):**

| gn_step | d_seg | d_pose | bytes | advisory action | accepted | wall s |
|---|---:|---:|---:|---:|---|---:|
| 0 (baseline W_joint) | 0.07021567 | 36.37588 | 138,804 | 26.18645 | — (sha-bound checkpoint verdict) | 0 |
| 1 | 0.07021567 | 36.37588 | 138,804 | 26.18645 | **NO** (3/3 rejected) | 1,543.2 |
| 2 | 0.07021567 | 36.37588 | 138,804 | 26.18645 | **NO** (3/3 rejected) | 1,551.1 |

Measured slope **0.000%/step** vs the first-order references: ws3 W_joint's own measured window
−0.43% over 4 steps (−0.078%/step relative slope, the honest current-vehicle reference embedded in
the receipt) and the charter's −1.26%/ep — which gc5 B8 had already scoped as an ANCESTOR slope
(R1 witness-vehicle pose leg, L18: lessons-never-numbers); cited for completeness, not used.

**Per-candidate anatomy (the mechanism, decomposed from the receipt's attempts):**

| step | mult | block Δ (proxy) | realized Δaction | Δseg-leg | implied Δd_pose |
|---|---|---:|---:|---:|---:|
| 1 | 1.0 | −12.7% | +1.2160 | +0.0033 | +13.1% |
| 1 | 0.5 | −10.2% | +0.9830 | 0 (bit-identical) | +10.6% |
| 1 | 0.25 | −9.0% | +0.6101 | 0 | +6.5% |
| 2 | 1.0 | −12.7% | +0.5300 | 0 | +5.6% |
| 2 | 0.5 | −7.0% | +0.5622 | 0 | +6.0% |
| 2 | 0.25 | −8.2% | +0.2690 | 0 | +2.8% |

Three named observations (instrument reading, not batch job):
1. **The engine solves.** Every proposal reduced the exact block objective through the STE forward
   (−7…−13%); CG converged better on step 2 (rel. residual 0.63 → 0.325 as damping adapted ×4);
   0.33 s/HVP; the GN propose is ~6 s of a ~1,547 s step — **wall-clock is 99.6% realized
   acceptance pricing** (3 chunked n600 CPU verdicts ≈ 514 s each), not the solve. Rung-1 capacity
   growth is therefore nearly free in wall-clock terms.
2. **Cross-pair seg transfer is ZERO at realized precision.** Shared-DOF steps tuned on a 4-pair
   block move the block's smooth logits but leave the realized n600 argmax population bit-unchanged
   (d_seg identical to 10 digits in 5/6 candidates). The one exception (step-1 mult 1.0, step_norm
   649) moved d_seg WRONG by +3.3e-5. The 706-param lift's shared coordinates do not carry
   block-local seg progress to the scored population through the uint8/coder staircase.
3. **The rejection price is pose collateral, not seg failure.** All 6 candidates realized with
   d_pose +2.8…+13.1% (the pose leg is 72.8% of the action at this pose-unsolved warm start);
   ee1's band-lemma watch (character changes near d_seg~1e-2 / ~1e-3) was NOT observABLE — the
   descent never left 0.0702. Honest scope note: with seg frozen and pose regressing, the v19
   joint gate at THIS operating point is effectively a pose-collateral gate; the #383 seg-null
   pose-subspace steering (the terminal-solve law's mid-descent dual) is the named in-family cure
   the next arm can add without touching v19.

## Capacity verdict (gc5 disambiguator input; steers 1–3 + ee1 C10 + pp1 folded)

**Window classifier verdict: `BLOCK_LOCALITY_OR_REALIZATION_GAP` ×2 consecutive → window stop by
rule.** NOT `ENGINE_SCALE_SATURATION` vs the S0 band: baseline d_seg 0.0702 is **56× above** the
box-solve operating point (1.25e-3) and **187×** above the S0 realization-noise band (3.6–3.8e-4)
— the engine never got near the band; the binding constraints are UPSTREAM of raw 706-param lift
capacity. The dimensions that bind, named:

1. **Cross-pair/shared-DOF transfer** (primary, measured): block-local seg progress does not
   realize on the n600 population. The train block (4 pairs, the sealed stage-0 schedule) is
   0.67% of the scored population while the DOF are global.
2. **The pose-collateral channel of the shared template DOF** (co-primary, measured): every
   seg-leg step perturbs the paint in pose-relevant ways; priced out by joint v19 at a
   pose-unsolved warm start.

**Against the pp1/no-correction target (steer 3): can this parametrization plausibly reach native
d_seg ≤ ρ_c = 5.0e-4?** The S0 band (C0 1.16e-3, C1 1.25e-3) sits INSIDE the correction band
(corrections still rational at box-solve margins), and ρ_c would require the box-solve substrate
itself to improve ~2.3×. The measured window gives **no affirmative evidence the 706-param
instance can even descend from 7e-2** under its current block/acceptance discipline — a 140×
reduction to ρ_c has no measured gradient here. Verdict_scope: INSTANCE (this lift + 4-pair block
+ joint-v19-at-pose-unsolved-warm-start), NOT family — the family-d GN solve itself measured
CORRECT (block objective descends second-order, as designed).

**The two-rung ladder (orchestrator steers 1+2; ee1 C10 convergence; routed, not swept — next
arm's charter input):**
- **Rung 1 — grow shared/cross-pair template DOF at fixed discipline:** directly attacks binding
  dimension 1 (train block → population-scale coverage or pair-local DOF), optionally + a
  #383-dual pose-null projector on the seg step for dimension 2. Wall-clock-cheap (observation 1).
- **Rung 2 — re-parametrize as token-grid + small trained partition→pixel renderer (≤64 KB
  counted, scorer-in-loop through R + uint8-STE):** per ee1's C10 convergence theorem this is the
  SAME family in better coordinates; it dissolves binding dimension 1 by construction (the
  renderer trains against ALL pairs simultaneously) and its external existence proof sits BELOW
  the no-correction threshold — PR130 40,252 B int4 → native d_seg 2.97e-4 ≤ ρ_c on the official
  rail (LESSONS-ONLY: never adopt bytes/constants). On measured evidence, rung 2 is the plausible
  ≤5e-4 route; rung 1 is the cheap intermediate test of whether the failure is block-scope rather
  than parametrization. pp1 context: with the partition leg converging at 117–177 KB across three
  parametrizations and the composed explicit route at S≈0.189 (above the 0.172 bar), REALIZATION
  is the campaign's binding constraint — this slot's blocker resolution is the named
  differentiator.

## Wire-in / hooks (Catalog #125)

Sensitivity-map: N/A (no new byte allocation; the GN step reallocates existing description DOF).
Pareto: the window receipt's curve rows are typed (d_seg, d_pose, bytes, advisory action) — planner-consumable.
Bit-allocator: N/A (no per-tensor importance change). Cathedral autopilot: N/A (no paid dispatch).
Continual-learning: this memo + DAG FEED + the fire ticket are the anchors. Probe-disambiguator: the
capacity classifier inside `_fd1_gn_window_locked` IS the disambiguator (DESCENDING /
REALIZATION_QUANTIZATION_GATED / ENGINE_SCALE_SATURATION / BLOCK_LOCALITY_OR_REALIZATION_GAP).

## Artifacts

- Engine: `src/tac/optimization/ddm_family_d_gn_description.py` (extends j2; subclassed feature module).
- Governed mode: `tools/launch_ddm_joint_descent.py` `--fd1-gn-window` (`_fd1_gn_window_locked`).
- S0 tool: `tools/measure_ddm_rp1_rangeA_cell_probe.py` (`--substrate boxsolve` closed; receiver-closed
  inflate custody-bound to r6cal).
- Smoke: `tools/smoke_ddm_fd1_gn_engine.py` + `.omx/research/ddm_fd1_gn_engine_smoke_20260728.json`.
- Fire ticket: `.omx/research/configs/ddm_fd1_family_d_gn_campaign_fire_20260728.json`.
- SSD receipts: `/Volumes/VertigoDataTier/pact/ddm_fd1_20260728/` (chunks/, logs/, boxsolve_inflate/).
