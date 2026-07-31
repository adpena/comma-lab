# ddm_dw1 — QA75 SOLVE-FRAME DISTILL WINDOW: the fork discriminator (2026-07-30)

**Task #790 · gc9 §3 row-5 distill-window arbiter (REPLACES the confounded photometric row-2 as
the capacity-vs-objective discriminator).** Pointer honesty FIRST: submittable
**0.1910828242 [contest-CPU] UNMOVED**. Every number below is **[macOS-CPU/MLX advisory]**,
`score_claim=false`, `research_only`. verdict_scope tags are the narrowest the receipts support.

## STORES-CONSULTED (recall receipts; multi-pass grep, path+sha)
- **pj1 memo** `.omx/research/ddm_pj1_projection_probe_20260730.md` (836a3c9c09): the photometric
  projection probe is CONFOUNDED (cross-vehicle dark-manifold range wall 67.95; verdict_scope
  formulation). Named the QA75 distill-WINDOW (scorer-space, vehicle-agnostic) as the clean arbiter.
- **QA75 ledger row** `.omx/research/ddm_deferral_queue_ledger_20260729.md` (row QA75, unique): the
  preregistered falsifier — distilled NOT clearly better than CE at matched budget ⇒ gap NOT
  distillation-curable (QA24 form fixes lead); distill clearly better ⇒ burn-3 distill-opening GO.
- **b2b teacher field** `/Volumes/VertigoDataTier/pact/ddm_b2b_qa75_field_20260730/` (600 per-pair
  NPZs; manifest v1). Each `distill_logits (5,384,512) fp16` = the precomputed SegNet scorer response
  on the EXACT C1 solve frames (realized d_seg ~1.52e-4). Concatenated → cache
  `distill_field_cache/distill_logits.f16.npy` (1,179,648,128 B, sha **4bb40f01620276b5…**).
- **E2 student** `/Volumes/VertigoDataTier/pact/ddm_bc1_20260731/burn_out/checkpoints/stage_seg_trunk_tau_final.npz`
  (sha e51178c01d3d8062…; meta::epoch 400, stage seg_trunk_tau; E2 realized d_seg 0.0052766 n600).
- **Burn sealed argv** `…/burn_out/launch_receipt.json` (ticket_hash 796283a2…, git 6b546da198) — the
  argv-diff reference (guard 3).
- **gc9 fork** `.omx/research/ddm_gc9_from_here_convocation_20260730.md` §3 row 5 / §4 decision table.
- **Trainer** `experiments/train_tr1_partition_renderer_mlx.py` (#517-twin resume re-anchor at
  6b546da198); **DSL** `spec_tr1_renderer_20260728` (live compile path the launcher consumes).
- **ax1 / gc9 seg-rate product law** (FEED-ax1 #789; `ddm_gc9_seg_rate_product_law_v1`) — the
  stack-arithmetic surface the dw1 seg result re-prices.

## §1 What was built (the real QA75 lever — ph3 stub folded-and-deleted)
The DESIGNED-STUB `Qa75SolveFrameDistill` (ph3 §10.1) is FOLDED-AND-DELETED per its own note; the
live implementation:
- **Trainer-side KD** (`make_loss_fn` additive-hook idiom, default OFF ⇒ **byte-identical**; unit-
  tested `base == off_none == off_w0`): computed THROUGH the deploy path (mask→tie→quantize→render→R
  →uint8-STE) on the SAME `adapter.segnet(f1)` student logits the seg loss already ran — **no second
  scorer forward** (the teacher field IS the precomputed scorer response). THREE raced forms:
  `kd_logits` (Hinton T²·KL), `margin_field` (one-sided hinge to the teacher's feasible top1-top2
  margin — the flip-distance currency), `argmax_ce` (CE to teacher argmax, margin-weighted). Attack
  weighting `exp(-GT_margin/temp)` (QA74 boundary annulus) is a **raced dimension** (guard 2).
- **DSL** `lever_solve_frame_distill` (provenance-manifested: T=2.0 CANONICAL Quantizr/PR95;
  w=100 DERIVED=w_seg S-exact; form/attack RACED; field MEASURED_ANCHOR) + `lever_head_range_relax`;
  matched A/B/C programs in `spec_tr1_dw1_distill_window_20260730`.
- **Window C head-relax** (MAIN charter): a trainable per-channel output residual (init 0 ⇒
  warm-start-EQUIVALENT to sigmoid×255; unit-tested render-identical-at-init) that de-saturates the
  rgb head so gradients reach out-of-chart dark pixels. **ADVISORY-NON-DEPLOYABLE** (breaks the E1
  receiver arch tr1_lotto_combined_ema_v1). Resume-registry hygiene: the new param is ema-backfilled
  on resume (guard 7; unit-tested).
- Field cache builder (sha-verified per pair), mini-race driver, seal+argv-diff verifier, verdict
  analyzer. 17/17 unit tests pass (1 pre-existing tb1 ledger-key test failure is unrelated — the
  SMEVR/rowband keys predate this arm; my diff does not touch `counted_bytes_ledger`).

## §2 The 8 optimal-form guards (MAIN hardening; how each is honored)
1. **Loss-form mini-race** (own-optimum): 3 forms × attack raced from E2 (n600, the real near-floor
   regime — a fresh/n96 race would rank forms in the wrong regime), winner picked at its optimum. §3.
2. **Attack raced, not optional**: attack_temp ∈ {0, 1.0} is a mini-race dimension. §3.
3. **Matched-config discipline**: B = the STRONGEST honest continuation (burn's full endpoint config:
   margin-weighted CE + rate-in-loss + SMEVR + events). A-vs-B differs by EXACTLY the 5 distill flags;
   C-vs-A by EXACTLY --head-range-relax (seal tool ASSERTS this; §4 pastes the 3 diffs).
4. **Under-drive guard** (preregistered): |B slope| < noise floor ⇒ PROVISIONAL-UNDERDRIVEN, next
   step = LR-rewarmup (#518 β₂-derived). ax1 saw QA24 schedule-truncated COUPLED_DESCENT ⇒ a live B
   slope is expected; a dead B is an apparatus signal.
5. **Joint seg+rate**: SMEVR bytes tracked in ALL windows; falsifier at seg+rate (product law); d_seg
   and bytes reported separately (distill could buy d_seg by spending token entropy).
6. **Noise floor preregistered**: B's gate-to-gate residual std stated BEFORE reading the split.
7. **Window C hygiene**: new head param registered in ema (backfill) + first-epoch loss equivalence
   verified before crediting divergence to the chart.
8. **s/ep per window**: reported (matched-cost check; if not matched the budget axis is wall-clock).

Resume invariants VERIFIED on the real E2 resume (mini-race config): `resume_form_reanchor` FIRED
(tau re-anchor, #517-twin), epoch 401, quant_engaged True, distill_field_ready FIRED, ema_backfill
[] for non-C / [head_relax_gain] for C.

## §3 Loss-form mini-race (guards 1+2) — RESULTS (MEASURED)

6 configs (3 forms × attack ∈ {0, 1.0}), each a 12-ep bounded resume from E2 at n600 with the FULL
matched base config (the real near-floor regime), sequential, gate_every=3, EMA-shadow gates.
Receipt: `mini_race/race_summary.json`. Reference: E2 endpoint gate-basis d_seg ≈ 0.00528.

| config | final gate d_seg | attack | A1-refused | rc |
|---|---|---|---|---|
| **kd_logits + attack (WINNER)** | **0.0050507** | 1.0 | no | 0 |
| argmax_ce uniform | 0.0050600 | 0.0 | no | 0 |
| margin_field uniform | 0.0051540 | 0.0 | **YES** (realization-gap ×3) | 0 |
| argmax_ce + attack | 0.0051670 | 1.0 | no | 0 |
| margin_field + attack | 0.0051870 | 1.0 | no | 0 |
| kd_logits uniform | 0.0052514 | 0.0 | **YES** (realization-gap ×3) | 0 |

**Load-bearing structure (MEASURED):** kd_logits WITHOUT attack-weighting A1-REFUSED (smooth loss
fell while realized d_seg did not — uniform KD wastes gradient on the dark Fisher interior exactly
as the race-rationale predicted), WITH attack it won. The attack-set weighting is not a tweak; it is
what makes KD realize. margin_field uniform also refused. Winner picked at its own optimum
(guard 1): **kd_logits, T=2.0, w=100, attack_temp=1.0**.

## §4 Matched windows A/B/C + argv diffs (guard 3) — RESULTS (MEASURED)

Three sealed governed windows (winner form kd_logits+attack), SAME E2 resume, 40 ep matched,
sequential (one scorer job at a time), launched via `tools/launch_tr1_run.py` (G0–G5 all PASS).
Pre-fire argv diffs (seal tool ASSERTED; `window_seal_report.json`): **A-vs-B = exactly the 5
distill flags** · **C-vs-A = exactly --head-range-relax** · B-vs-burn = the 9 intended window
deltas (basin-off, composed-S instrument dropped, derived ema, epochs/gate/wall/outdir/resume).
`resume_form_reanchor` FIRED in all three; C additionally `ema_backfilled_new_params:
['head_relax_gain']` (guard 7).

| window | stop | ep ran | endpoint n600 d_seg (EMA, trainer confirm) | slope/ep (shadow gates) | counted bytes | s/ep |
|---|---|---|---|---|---|---|
| **B control** | epochs_complete | 40 | **0.0051147** | **−6.80e-6** (descending) | 259,407 | 33.2 |
| **A distill** | **a1_realization_gap_refuse @ep430** | 29 | **0.0054967** (zero-step resume confirm, same surface) | **+1.37e-5** (ASCENDING) | 257,951 | 34.8 |
| **C chart-relax** | epochs_complete | 40 | **0.0054394** | +8.41e-6 (ascending) | 261,600 | 34.0 |

Matched gate trajectories (EMA-shadow): B 0.005217→0.004986 monotone-ish COUPLED_DESCENT;
A 0.005175→**0.004995 @ep409 (transient dip, ahead of B)**→0.005124→0.005207→0.005283→0.005463
(refuse) — the smooth KD loss FELL throughout (19.8→8.1) while realized d_seg ROSE = a textbook
**realization gap** (fd2 signature; the trainer's pre-registered guard caught it). C oscillated
with intermittent gap alarms (never consecutive → no refuse) and ended worse than B.
**The 12-ep mini-race horizon ended exactly at A's transient dip** — the 40-ep window reveals
the reversal (a measured lesson about short-window form races).

## §5 Falsifier applied + verdict + verdict_scope

Preregistered numbers first (guard 6): noise floor = **2.99e-5** (B's gate residual std about its
own trend); B total window descent = **2.45e-4** (8.2× noise ⇒ NOT under-driven; guard-4 gate
passes — and NOTE: B improved E2 0.0052766→0.0051147, so plain continued optimization still pays
at this endpoint). Split B−A = **−3.82e-4** (A WORSE by 12.8× noise); slope ratio A/B = **−2.01**
(A ascends at 2× the rate B descends). Joint seg+rate (guard 5): bytes matched within ±0.7%; A's
−1,456 B rate saving = 9.7e-4 S·rate vs its +3.8e-4 d_seg = +3.8e-2 S·seg — seg dominates 40×;
advisory seg+rate S: B 0.6842 < C 0.7181 < A 0.7214. s/ep matched within 5% (guard 8).

**VERDICT — the preregistered QA75 falsifier FIRES, on the WORSE side:** distilled is not merely
≤ CE at matched budget — it is actively worse (realization-gap reversal). **The 25.58× amortization
gap is NOT distillation-curable on this endpoint: optimization/capacity leads; QA24 form fixes
lead; the class-change leg strengthens. Burn-3 distill-opening = NO-GO** (the GO condition "distill
slope clearly better" is inverted: the slope ratio is negative).

**Window C (chart) verdict:** chart relaxation does NOT rescue the distill (C endpoint 0.0054394 >
B 0.0051147; slope still positive). The rgb output chart is NOT the binding constraint for the KD
reversal — the gap lives in the KD objective itself (the teacher's dark-knowledge directions do not
survive the deploy path uint8-STE/R round-trip at this near-floor operating point). C avoided the
refuse threshold only by oscillating; ADVISORY-NON-DEPLOYABLE regardless (E1 receiver break). No
receiver arch rev is motivated by this measurement.

**verdict_scope: FORMULATION** — solve-field distillation as a FINISHING-stage lever on this
converged E2 endpoint (all 3 loss forms × attack raced at their own optima, 6 configs, + the chart
probe; the strongest honest control). NOT covered / still live: distill-from-BIRTH (the ph3
front-loaded opening-stage doctrine — a different regime where the student is far from the teacher),
born-again gen-2, and anneal-to-CE hybrids. But the burn-3 distill-opening GO branch is closed by
the preregistered fork condition, and the from-birth form drops in priority (the finishing-form
reversal + B's live descent both point at optimization/capacity, not target-infeasibility).

## §6 gc9 fork row-2 REPLACEMENT row (for the §4 decision table)

| # | probe | mechanism | falsifier state | fork consequence |
|---|---|---|---|---|
| 2′ | **QA75 distill-WINDOW (ddm_dw1; REPLACES the confounded photometric row-2 AND resolves row-5)** | 3 matched governed 40-ep windows from E2 (control / distill / chart-relax), kd_logits+attack winner of a 6-config own-optimum race, endpoint n600 EMA confirms | **FIRED — distill WORSE (B−A −3.82e-4 = 12.8× noise; slope ratio −2.01; C no rescue)** | **capacity-vs-objective RESOLVED at this endpoint: NOT target-infeasibility — optimization/capacity leads. Burn-3 distill-opening NO-GO; QA24 form fixes + granularity/class-change lead; B's live descent (E2 −3.1% in 40 ep) says the endpoint is NOT converged — cheap continued/rewarmed optimization is real headroom** |

## §7 ax1 stack-arithmetic re-derivation note

ax1's derivation (FEED-ax1 #789) carried QA75-distill as a candidate seg mover inside the composed
stack. dw1 re-prices it: **the QA75 finishing-distill term contributes ZERO (negative) to the
composed stack** — remove it from any stack arithmetic. What dw1 ADDS to the stack instead
(MEASURED): the **plain-continuation dividend** — E2 is not converged; 40 ep of continued training
bought Δd_seg −1.6e-4 (−0.016 S·seg) at zero design cost. Any composed candidate built on the QA24
lineage should assume the (cheap) continued/rewarmed-optimization term, not the distill term. The
seg-rate product-law surface (`ddm_gc9_seg_rate_product_law_v1`) is otherwise unchanged: dw1 moved
neither the rate class (bytes matched ±0.7%) nor the seg floor claim (B's 0.0051147 is the new best
realized n600 on this vehicle lineage — supersedes E2 0.0052766 as the c-side anchor).

## §8 Custody (paths + shas; certify-or-block, no /tmp)
- Distill field cache: `/Volumes/VertigoDataTier/pact/ddm_dw1_20260730/distill_field_cache/`
  (distill_logits.f16.npy 1,179,648,128 B sha 4bb40f01620276b5…; cache_manifest.json; REBUILDABLE via
  `tools/ddm_dw1_build_distill_field_cache.py` over the b2b field — safe to cold-store).
- Mini-race: `…/ddm_dw1_20260730/mini_race/` (race_summary.json + 6 config dirs).
- Windows: `…/ddm_dw1_20260730/{control,distill,distill_head_relax}/` (telemetry + receipt + launch
  receipts + per-stage EMA ckpts); A endpoint confirm `…/distill_endpoint_confirm/`; tickets
  `…/tickets/` (3 sealed, hashes inside); seal report `…/window_seal_report.json`; verdict
  `…/verdict.json`.
- B endpoint ckpt (new best realized n600 anchor 0.0051147):
  `…/control/checkpoints/stage_seg_trunk_tau_final.npz`.
- Code: commits 67ed745266 (core lever+trainer+tests) · 96675879ef (mini-race) · seal tool ·
  dbd00ada76 (verdict fix). burn_out READ-ONLY (untouched).

## §9 LIVE-HYPOTHESES / DEAD-ENDS / NEXT-IF-RESUMED

- **DEAD-END** — verdict_scope: FORMULATION — solve-field KD/margin/argmax distillation as a FINISHING lever on the
  converged QA24 endpoint (6 own-optimum configs + chart probe; realization-gap reversal is the
  mechanism). Do not re-run finishing-distill variants expecting a different verdict at this
  operating point; the teacher's soft directions do not survive the deploy-path round-trip here.
- **LIVE-HYPOTHESIS (strengthened, MEASURED):** the endpoint is NOT converged — B's plain
  continuation still descends (−3.1% in 40 ep, COUPLED_DESCENT, no plateau signature at window
  end). The cheapest real seg mover on this lineage is MORE/rewarmed optimization (LR-rewarmup per
  #518 laws, longer windows) + the QA24 form/granularity fixes — not a new objective.
- **LIVE-HYPOTHESIS (untested, deprioritized):** distill-from-BIRTH (ph3 front-loaded opening
  doctrine) in a fresh burn — a different regime (student far from teacher); the finishing-form
  reversal does not formally cover it, but the fork's GO condition failed, so it needs a NEW
  preregistered case to fire.
- **DEAD-END (advisory):** Window-C head-relax as a distill rescue — no rescue, non-deployable;
  no receiver arch rev motivated.
- **NEXT-IF-RESUMED:** (1) bank B's endpoint (0.0051147, ckpt in custody) as the running best
  realized n600 anchor for the QA24 lineage and the composed-candidate arithmetic; (2) route the
  freed burn-3 slot to the QA24 granularity/form ladder + continued-optimization windows; (3) if
  anyone reopens distill, it must be the from-birth form with a new preregistered falsifier.

Pointer delta: **UNMOVED (0.1910828242 [contest-CPU])**. This unit is means, not end.
