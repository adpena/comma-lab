# Curriculum-candidate POOL — P0 durable inventory + tracked costate class (task #403)

**Operator P0 binding 2026-07-10 (verbatim):** *"Stages or anything in any form ... All of that is p0
update the triality and tasks accordingly ... That is a bad bad bad class of orphaned work necessary
for true final optimal ... Update costate controller accordingly as well."*

A chat-only inventory of built-never-fired curriculum candidates IS the orphan bug (sisters:
`default_off_is_orphaned_signal_activation_ledger_reconciliation_20260706` +
`velocity_driven_orphaning_the_deepest_signal_loss_meta_bug` + memory
`curriculum_candidate_pool_p0_orphan_class_20260710`). This memo is the durable inventory; the TRACKED
surface is DESIGNED to be a new canonical store **`.omx/state/curriculum_candidate_pool.jsonl`** (module
`tac.witness_dsl.curriculum_candidate_pool`, read by `tools/costate_digest.py` §curriculum-pool — the
controller remembers, the operator never has to).

> **⚠ STATUS (2026-07-10, honest — NO-FAKE): the costate WIRING below is DESIGN, NOT YET BUILT.** The
> authoring agent was killed (credit exhaustion) after landing THIS inventory memo but BEFORE creating the
> `.jsonl` store, the `tac.witness_dsl.curriculum_candidate_pool` module, or the `costate_digest.py`
> §curriculum-pool reader. Those three surfaces do NOT exist on disk yet — building them (+ the DSL stubs
> + the DAG FEED) is the remaining task-#403 work. This memo is banked NOW to preserve the (complete,
> re-derived) inventory signal; every "IS read by / the controller remembers" phrasing below is the
> intended design, gated on that build. Do not cite the store/module as existing until #403 lands them.

Discipline per
`docs/operating_manual_craft_handoff.md`: every status below was RE-DERIVED from primary artifacts
(task JSONs, DAG FEEDs, trainer argparse, `lever_registry.lever_factories()`,
`CRUCIBLE_V752_LAUNCH_EXPECTED_LEVERS`), never trusted from the seed list — several seed labels were
CORRECTED (marked ⚠ below). Labels: MEASURED / DERIVED / ESTIMATED per row.

**STORES CONSULTED:** task store `~/.claude/tasks/89ff112f…/{129,195,216,217,218,242,248,301,302,305,
308,309,310,312,319,366,383,403}.json` · DAG `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_
20260611.md` (FEED-stepnative · FEED-collapsefix · FEED-v752-config · FEED-negcure-join ·
FEED-reactivation-397 · FEED-legdisposition-owed16v2 · FEED-400-diagonal-build · 5-LENS verdict L6916 ·
FEED-05r/05s/05v StEik/ViscoReg · FEED-chroma-rung) · `witness_native_schedule_derivation_20260709.md`
(#302) · `src/tac/witness_autoconfig.py` (`CRUCIBLE_V752_LAUNCH_EXPECTED_LEVERS`, v752 delta/removes) ·
`src/tac/witness_dsl/{curriculum_dsl,lever_registry,activation_ledger}.py` + `completeness()` live run
(107 unmapped flags) · trainer argparse `experiments/train_levelset_witness_realized_through_R_mlx.py`
(LEVER-5 hardness-* L11310-11324 · #218 head/AM/logit-adjust L10814-10824 · #224 clDice L11724-11733) ·
`papers_checked_kohonen_som_20260710.md` · `papers_checked_cells2pixels_nca_lppn_20260710.md` ·
`ADVISORY_sdf_pose_inverse_carrier_20260710.md` · `reactivation_campaign_397_20260710.md` (JOIN — this
memo stays scoped to CURRICULUM-form; #397 owns the full reactivation pool) ·
`negative_cure_adjacency_answers_under_our_noses_20260710.md` · memory
`curriculum_candidate_pool_p0_orphan_class_20260710.md` · `tools/costate_digest.py`.

**Scope boundary:** the terminal solvers (D27b/TerminalSolve, mc_finisher click-polish #396/#399/#400)
and the #383 pose conditioning gate are OWNED elsewhere and only appear here where they gate a
curriculum-form candidate. Non-curriculum reactivation items live in #397's pool, not here.

---

## 1. ALREADY FLYING (armed) — documented so nobody re-proposes them

Source of truth for "armed": `CRUCIBLE_V752_LAUNCH_EXPECTED_LEVERS` (witness_autoconfig.py, the
fail-closed expected-active-lever manifest) + the v752 delta block. All MEASURED-in-config.

| candidate | form-class | DSL leg | slot | justification (source) |
|---|---|---|---|---|
| Geometric Γ-derived τ-homotopy (`seg_form_unify_tau`) | loss-geometry | `SegFormUnifyTau` | in-run-stage | DERIVED blinded + built per #302 (`witness_native_schedule_derivation_20260709.md` §1: CE stage IS the τ≈1 arc); composed in v752 launch-1 |
| Event-gated transitions (τ-advance events, nucleus guard) | optimizer-stage | `EventTriggeredCurriculum`/`TauAdvanceEvent` | in-run-stage | DERIVED (#302 §3 events partition a continuous flow, geometric not temporal); event scaffold in trunk |
| TAIL turnpike cycles (`tail_k_warm_restart`) | optimizer-stage | `TailCycles` | in-run-stage | DERIVED (#302 §2 finite-τ turnpike); composed in v752 |
| LADDER island-birth homotopy (`n323_ladder_island_homotopy`) | loss-geometry | `LadderIslandHomotopy` | in-run-stage | MEASURED-vindicated (#302 Phase-2 element-4 STRONG MATCH); composed in v752 |
| Birth-completion ramps (`v75_birth_completion_event` + `v75_area_constraint_birth`) | loss-geometry | `BirthCompletionEvent`/`AreaConstraintBirth` | in-run-stage | composed in v752 (persistence-ordered nucleation, #302 element-4) |
| Polyak finisher arm (`R7_polyak_finisher`) | averaging | `PolyakFinisher` | terminal-band | DERIVED (turnpike orbit → uniform tail mean O(1/√n) beats phase-carrying EMA; extra ckpt candidate, EMA never replaced); composed in v752 |
| Stage-transition re-warmup + moment resets | optimizer-stage | `Beta2WindowRewarmup` | in-run-stage | MEASURED (FEED-bu 0.1×/8ep magnitude; 5-LENS "reheat=continuation-step-control CONFIRM"); trunk |
| Muon conditioning finisher + warm-start (+ lr-final-frac 0.1) | optimizer-stage | `Muon`+`MuonWarmStart` | in-run-stage | MEASURED (#270 restart recipe, L79); #302 element-5 "direction+gentleness+placement MATCH; fixed-726 = un-derived knee-transfer residual" — the residual is a pool row (§2.11) |
| Logit-adjustment (PARTIAL: Menon facet-3 + loss-tau + `--logit-adjust-classes` movable-only) | loss-geometry | `MarginFieldHead`/`LogitAdjust` | in-run-stage | v752 config text "persistence/logit-adjust-classes movable-only UNCHANGED"; facet-1 head geometry NOT yet fired (§2.4) |
| d_seg-aware taper (`--dseg-aware-taper`, v752 delta) | regularizer-schedule | `DsegAwareTaper` | in-run-stage | INSTANCE-grade; owed-15 n600 fresh-arm isolation converts INSTANCE→MEASURED-or-rollback (v752 delta comment) |
| Amber stability preset (grad-clip 0.5 + pose-grad-coeff-max 25) | optimizer-stage | via `--stability-preset` (autoconfig-composed) | in-run-stage | MEASURED law `max_pose_grad_coeff=5/√eps` (FEED-collapsefix; `tac.witness_stability`); on the relaunch |
| GNC graduated non-convexity | optimizer-stage | embodied (no separate lever owed) | in-run-stage | MEASURED-confirmed lens: "ce→tau→l7 IS graduated-non-convexity CONFIRM" + geometric τ/β schedule TRIPLE-triangulated (Hazan + GNC + Fisher-Rao; FEED 5-LENS + FEED-05r §5). Residual build = StEik-normalized (§2.9) |

**Pinned A/B LADDER rungs (armed-as-increments, NOT launch-1; SPEC_v752 §4):** σ_cc′
`--length-sigma-matrix` (rung 1b, fires at the tau boundary BEFORE any other length-touching lever) ·
margin-hinge · chroma rung (FEED-chroma-rung REGISTERED-OFF) · step-native (§2.2). Each is its own
increment A/B, never composed into the first trunk — the rung ORDER is the fire criterion.

---

## 2. BUILT-NEVER-FIRED (the duty-to-measure core — highest readiness)

| # | candidate | form-class | DSL leg | gate / fire criterion | slot | justification | owner |
|---|---|---|---|---|---|---|---|
| 2.1 | Hard-pair emphasis data curriculum (LEVER-5 `--hardness-*`) ⚠ seeded as "new"; actually BUILT in trainer, DSL-orphaned until this landing | data-curriculum | `HardnessOversample` (NEW fold, this landing) | fair A/B = `--hardness-weighted` on/off at fixed oversample (same total steps, different allocation); `source=realized` (trainer-recommended; margin-source spread only MEASURED 1.31×, trainer L11306) | in-run-stage | MEASURED anchors: 44%-of-CE-residual-spikes-are-LANE (#205 CE-floor, L67) + margin-saliency #141; trainer L11303-11324 default-off byte-identical | #403→new |
| 2.2 | Step-native activation arm (#310 FINER++ β-anneal) ⚠ seeded "needs-build"; CORRECTED: BUILT (FEED-stepnative: factory+flags+equation existed 07-07; guard-hardened 07-09) | architecture-growth | `StepNativeActivation`+`FinerBiasInit` | pinned v7.5.2 ladder rung; byte-close A/B annealed-β+FINER vs OFF; surviving flips must move to no-ring survival | in-run-stage | ESTIMATED 31.6% of remaining descent (relative-significance re-audit, FEED-stepnative); duty-to-measure #2; negcure RANK-4 second-exemplar (hosc-death→β-anneal cure) | #310 |
| 2.3 | Focal + boundary-distance loss continuation (#301) | loss-geometry | `SegFocalGamma`+`BoundaryDistance` | PRE-REGISTERED (#301 item 3): ep50→100 witness-alone slope flattens (|Δd_seg|<0.02/25ep with islands >50% of residual) → deploy focal at γ*; steep slope → HOLD. CAVEAT: C17 calibration measured γ*=0 (negcure no-join row) — re-check γ* on the live ckpt before firing focal; BoundaryDistance unaffected | in-run-stage | MEASURED build (task #301 completed: default-OFF byte-identical, tested, READY-not-deployed) | #301 |
| 2.4 | #218 head-geometry facets 1a/1b (ETF head / additive-margin) | loss-geometry | `HeadGeometry` (NEW fold, this landing); sisters `MarginFieldHead`+`PersistenceTopology` already held | fire ETF first (byte-free + rate-win, neural-collapse minority-norm fix); AM-hinge needs `--margin-field-head-weight>0` | in-run-stage | DERIVED (#218 program: rare-class lane margin fix; trainer L10814-10824 built default-off) | #218 |
| 2.5 | Soft-clDice/Betti persistence loss (#218/#224) | loss-geometry | `PersistenceTopology` | warm-up epochs param; fire with the #218 rung; `--persistence-classes auto` self-detects erasure-tail classes | in-run-stage | MEASURED build (#224 wired; dash erasure law `dash_erasure_homogenization_v1` is the target, L65) | #218 |
| 2.6 | msal_uni → exact-S_R reachability A/B (#268) | loss-geometry | `MarginSaliencyReachability` | ZERO build: `gt_n600_sR.npz` READY; A/B on the fragile annulus band | in-run-stage | MEASURED: texture proxy AT CHANCE vs through-R reachability (L76); negcure RANK-2 second-exemplar-grade | #268 |
| 2.7 | Balle rate-in-loss weight-entropy penalty | regularizer-schedule | `WeightEntropyPenaltyMLX` | fire on a rate-attack arm; MEASURED −19.6% bytes (torch ancestor — ancestor-rule: number does NOT transfer, mechanism does) | in-run-stage | FEED-reactivation-397 T-397-1 (4 built-never-fired factories) | #397 |
| 2.8 | Laguerre-OT head-offset solve | discrete-solve-interleave | `HeadOffsetSolver` | fire as a terminal-band solve rung after gradient descent bottoms | terminal-band | FEED-reactivation-397 T-397-1; deep-math #284 (argmax = Laguerre power diagram, L-v8) | #397 |
| 2.9 | StEik nᵀHn NORMALIZED variant (`--eikonal-steik-normalized`) | regularizer-schedule | dsl N/A: eikonal sub-knobs deliberately compiled via `EikonalViscosity`-family config, not a bare-name lever (fold owed if promoted) | gate: #316 FAIR eikonal-viscosity test first (operator-GO flagged; D1 known-tainted era) | in-run-stage | MEASURED: RAW StEik NO-GO n24 (self-amplifying 575×-1431×, verdict_scope: formulation); normalized variant = the named follow-up (FEED-05v) | #316 |
| 2.10 | ViscoReg ε-continuation at n600 (`EikonalViscosity`) | regularizer-schedule | `EikonalViscosity` | #316 fair test (first non-confounded measurement; re-grades D1/D2/D7) | in-run-stage | MEASURED n24: ε=0.3 STABLE + d_seg 2.3× better than control; ε=1.0 explodes (two-sided window; FEED-05v) | #316 |
| 2.11 | Per-param grad-normalize (`--grad-normalize per-param`, Cells2Pixels fold) | optimizer-stage | dsl N/A: stability-preset surface (autoconfig `resolve_stability_config`), not a swept DSL lever yet | OWED trajectory A/B (alters seg-vs-pose gradient scale ratio — documented caveat) | in-run-stage | MEASURED-built FEED-collapsefix (byte-identical default; candidate better PRIMARY than global clip for batch=1) | — |
| 2.12 | Muon event-derived switch (replace fixed ep726) | optimizer-stage | `Muon(start_epoch)` + `--muon-start-event` flag exists (unmapped sub-knob of the event scaffold) | fire when conditioning event (σ_min/curvature) trips AND nucleation done (#302 §3c) | in-run-stage | DERIVED (#302: fixed-726 = un-derived knee-transfer residual; event scaffold present) | #302 residual |
| 2.13 | σ_cc′ length-sigma-matrix rung 1b | loss-geometry | `LengthSigma` | pinned rung 1b: fires at the tau boundary BEFORE any other length-touching lever (v752 W-1 removal comment: reverted to σ≡1 in launch-1 to keep Class-A attribution clean) | in-run-stage | MEASURED build (fitted-20260707 inherited then deliberately reverted; SPEC_v752 §A.8/§1b) | ladder |
| 2.14 | Chroma rung (v7.5.2 REGISTERED-OFF add-back) | loss-geometry | `SegChromaBoundary` | rung fires per FEED-chroma-rung registration (its own increment A/B); SegNet reads RGB → chroma is a genuine d_seg actuator (CLAUDE.md §Chroma: any verdict ignoring chroma is provisional) | in-run-stage | registered-off rung (FEED-chroma-rung) | ladder |
| 2.15 | Margin-hinge rung | loss-geometry | dsl N/A-pending: SPEC_v752 §4 names the rung; VERIFY which held margin lever (`MarginBandSatisficing` vs `HorizonWeightedMargin` vs `MarginSaliency`) realizes it before firing — do not guess the factory | in-run-stage | pinned rung (SPEC_v752 §4) | ladder |

---

## 3. NEEDS-BUILD (queued; design/task exists, no code on THIS vehicle)

| # | candidate | form-class | DSL leg | gate / fire criterion | slot | justification | owner |
|---|---|---|---|---|---|---|---|
| 3.1 | SWA / model-soup across TAIL-cycle ENDPOINTS | averaging | dsl N/A until built (checkpoint-space op, likely a byte-close-side tool + trainer export hook) | design-memo-first; A/B vs PolyakFinisher (the built within-window sibling) + EMA; EMA non-negotiable: soup is an ADDITIONAL candidate, never replaces the shadow | terminal-band | DERIVED: turnpike cycles produce K basin-endpoint iterates; uniform tail averaging already vindicated in-window (PolyakFinisher design); cross-cycle soup is the unbuilt complement. grep MEASURED: zero swa/soup code in tree | NEW (follow-up task) |
| 3.2 | Fisher/GN head natural-gradient mid-run block | preconditioning | dsl N/A until built (in-trainer solve block, not a flag) | JOINS #341: quadratic head chart CONFIRMED (LM ρ 0.847/0.868) but K=8 subset-solve overfits (+5.1% net) → ONLY full-P in-trainer GPU solve admissible (~11min/CG-iter@17×, L77); fire criterion = build the CG solve, run at a stage boundary | in-run-stage | MEASURED foundation: margin=Fisher ρ0.978 (L1) + eq `quadratic_head_chart_subset_solve_gap_v1` | #341-adjacent (follow-up task) |
| 3.3 | Interleaved discrete click/proposal rounds at stage boundaries | discrete-solve-interleave | dsl N/A: mc_finisher is a byte-close-side tool (#400 diagonal BUILD-ONLY, FEED-400-diagonal-build); in-run interleave needs a trainer-side design | promotion gate: #400 diagonal MEASUREMENT first (terminal-band); only then design the in-run interleave | terminal-band → in-run-stage | MEASURED: n8 click row MOVED the pointer −1.7e-5 (FEED-pointer-move-n8click); n600 sweep in-flight | #400 (+ follow-up task for in-run) |
| 3.4 | #309 IGA boundary-tangent NTK preconditioning | preconditioning | dsl N/A until built | HELD-WEAKENED: owed-16 v1 basis ≈0 zero-shot AND owed16v2 REBALANCE measured no-benefit (FEED-legdisposition-owed16v2) → the along-tangent-frequency axis is twice-measured-dead at the input-basis level; re-derive whether the NTK-level mechanism survives before building | in-run-stage | MEASURED 3.2× along-tangent deficit (L65) stands, but two basis-level cures measured ≈0 — negative↔cure join now points AWAY | #309 |
| 3.5 | #242 SAM/flat-minima MDL pre-quantization stage | regularizer-schedule | dsl N/A until built | design-memo-first; adjacency: `WeightEntropyPenaltyMLX` (§2.7) covers the rate-in-loss half — fire that FIRST, build SAM only if the entropy penalty measures insufficient | in-run-stage (pre-quant band) | DERIVED (compression-as-intelligence; task #242 pending) | #242 |
| 3.6 | #308 grids-for-bulk + INR-annulus hybrid | architecture-growth | dsl N/A (vehicle-level, not a flag) | negcure: HELD (no matched MEASURED violated fact); v8 per-class carriers partially embody it — re-scope AGAINST v8 increment-1 before building | next-vehicle | DERIVED (NeurIPS'25 grids-beat-INRs-except-boundary; matched-bytes protocol pre-registered in task) | #308 |
| 3.7 | #217 post-Muon leap-residual/SGLD sub-stage ⚠ seeded "built-never-fired"; CORRECTED: needs-build (grep MEASURED: zero SGLD code in trainer; task pending, gated) | optimizer-stage | dsl N/A until built | GATED on #216 saddle-to-saddle signature test ($0, run first); components (i) Damian reweight ≈ hardness/margin-saliency (§2.1/§2.6 partially cover) | terminal-band | DERIVED (MFLD multi-index leap theory; Muon≈Stiefel so it applies exactly) | #217/#216 |
| 3.8 | #129 KD warm-start actuator (basin teacher → re-tapered student) ⚠ seeded "built-never-fired"; CORRECTED: needs-build, PARTIALLY built (FiLM-v2 trunk-decoupling DONE `867ff3af5`, ∂d_seg/∂pose=0 proven; actuator itself pending) | init-warm-start | dsl N/A until built (production actuator, spans export) | bind-all spec `production_readiness_bind_all_ingredients_20260616.md`; #301 banked it as the 3rd rung (KD-from-#205-teacher on island band) | next-vehicle | DERIVED (Hinton KD; production linchpin) | #129 |
| 3.9 | #319 SimpleTES K>1 candidate emission ⚠ seeded "built-never-fired"; CORRECTED: needs-build (campaign-layer recommendation-SHAPE param; grep: not in witness_control) | discrete-solve-interleave (campaign layer) | dsl N/A (shadow-controller shape, not a trainer flag) | GATED behind #315 + BINDING backtest against v1-v5+#205 logs before adoption; fire when through-R band spans 0 at n=3 | in-run-stage (advisory layer) | DERIVED (SimpleTES DF-1 split-verdict; costate core NOT-RELEVANT refused) | #319/#315 |
| 3.10 | MD-Decoupling A/B (#195) ⚠ seeded "built-never-fired"; CORRECTED: needs-build on THIS vehicle (5-LENS CONTRADICT row: no `--optimizer`/`--md-base` in the levelset trainer = TRAINER-GAP; the residual-INR side owns the original A/B) | optimizer-stage | dsl N/A until the trainer grows the optimizer switch | build the optimizer flag, then LR-transfer A/B per #195 | in-run-stage | DERIVED (MD stable-by-construction claim is UNVERIFIED on this vehicle — that is the point of the A/B) | #195 |
| 3.11 | Pose-inverse-carrier distillation stage | pose-carrier | dsl N/A until built (decoder-native generator + offline distillation; archive-legal per rule-118 boundary) | ADVISORY only (research_only=true): offline PoseNet discovery/distill of frame-0 corrections; archive must decode WITHOUT PoseNet/scorer weights/GT tables; gate = #248 P-B FiLM read-back decisive first | terminal-band / next-vehicle | DERIVED (ADVISORY_sdf_pose_inverse_carrier_20260710: ξ-only carrier too low-rank for a frontier break; unconstrained frame-0 inverse proves evaluator admits accurate witnesses) | #248/#366 |
| 3.12 | SOM-organized codebooks (conscience/frequency-sensitive competition) | state-evolution / data-curriculum | dsl N/A (design-banked; representation-side) | EXACT-GATED A/B ONLY (papers-checked: representation-side chart levers repeatedly MEASURED ≈0 realized through R); pays twice IF measured: click-polish ±1/±2 locality + temporal-delta rate | terminal-band / next-vehicle | DERIVED lens: SOM magnification law (density ∝ p^(d/(d+2))) under-allocates rare regions = the measured lane starvation; the conscience CURE is already embodied by LADDER per-class λ + #218 — the codebook idea is the only NEW piece (`papers_checked_kohonen_som_20260710.md` item 4) | NEW (design-banked) |

---

## 4. REFORMULATION-QUEUE (prior formulation measured NO-GO; named reformulation owed)

| # | candidate | form-class | DSL leg | gate | slot | justification | owner |
|---|---|---|---|---|---|---|---|
| 4.1 | NCA/LPPN evolving-state stage (Cells2Pixels reformulation of AMBER #146) | state-evolution | dsl N/A (vehicle-level) | fires ONLY on AMBER/P10 reactivation; the reformulation = coarse-NCA-grid + our trunk as the LPPN (the split #146's arm lacked) | next-vehicle | MEASURED wall stands (#146 33K-rule generalization gate); `papers_checked_cells2pixels_nca_lppn_20260710.md` names the exact reformulation; state-pool/seed-reinject/random-unroll named as the deep-unroll companions (FEED-collapsefix item 3) | #146/P10 |
| 4.2 | Curvelet-from-scratch trajectory-shaping | init-warm-start / architecture-growth | dsl N/A until re-derived | FIRE-1 diagnostic left it the ONLY uncovered axis of the −48%-direct-vs-≈0-realized gap, "confounded by freq-along starvation (owed16v2 fixing)" — owed16v2 has NOW measured no-benefit ⇒ the confound resolved AGAINST the axis; gate = a fresh derivation must name a mechanism that survives BOTH measured negatives before any build | in-run-stage | MEASURED: naive-palette realized ceiling F=0.0337 ≫ directional-direct 0.0037; trained trunk redundant with basis |Δ|≤1.4% (FEED-reactivation-397 FIRE-1) | #397 |

---

## 5. EXPLICITLY EXCLUDED (retired-with-reason — do not re-propose)

| candidate | reason (MEASURED law) |
|---|---|
| GradNorm / per-step loss balancing | #312 law: per-step balancing FORBIDDEN — naive GradNorm would have DOWN-weighted the eikonal CANARY mid-runaway (canary-muting; FEED-05r §4). Loss weights move at STAGE BOUNDARIES ONLY (SPEC_v75 §8-C, `assert_loss_weights_stage_boundary_only`); the sanctioned form = the #312 stage-boundary gradient-share checkpoint probe (task completed). |
| Mod-dim-as-capacity re-open ("bigger mod-32") | #299 CLOSED (verdict_scope: formulation) — refuted by #300's measured island-gradient-starvation mechanism (FEED-reactivation-397). |
| Raw-gradient StEik | MEASURED NO-GO n24 (self-amplifying 575×–1431×; FEED-05v); the normalized variant is §2.9 — fire THAT, never the raw form. |

---

## 6. TOP-5 by (readiness × derived fit)

1. **Hard-pair emphasis / LEVER-5 hardness oversample (§2.1)** — readiness MAX (built in trainer,
   byte-identical off, DSL lever folded this landing), fit MEASURED (44%-of-spikes-are-lane; realized-d_seg
   source; the fair A/B is pre-specified in the trainer's own comment).
2. **Step-native/FINER++ rung (§2.2)** — built + guard-hardened; ESTIMATED 31.6% of remaining descent;
   already a pinned v7.5.2 rung; negcure second-exemplar.
3. **#268 msal_uni→sR reachability A/B (§2.6)** — zero build, npz ready, sits exactly on the fragile
   annulus band; negcure RANK-2.
4. **#218 head program: ETF head first (§2.4/§2.5)** — byte-free + rate-win + minority-norm fix aimed at
   the lane tail; DSL leg completed this landing; composes with armed Menon facet.
5. **#301 boundary-distance (+ focal pending γ* re-check) (§2.3)** — built READY-not-deployed with a
   pre-registered fire criterion; the ONLY row whose trigger is already written as an inequality.

(6th: `WeightEntropyPenaltyMLX` §2.7 — strongest rate-axis row, fires on a rate-attack arm.)

## 7. Costate integration (deliverable 2 — what was built)

- **Store:** `.omx/state/curriculum_candidate_pool.jsonl` — canonical APPEND-ONLY fcntl-locked JSONL,
  latest-row-wins per candidate; status ∈ {armed · built-never-fired · needs-build ·
  reformulation-queue · measured · retired-with-reason}; every row carries gate + justification +
  source_anchor (NO-FAKE) + exactly-one-of {dsl_lever, dsl_na_reason} (the per-row DSL-leg discipline)
  + optional est_delta_s/axis joining the same `relative_significance` math the lever queue uses.
- **Module:** `tac.witness_dsl.curriculum_candidate_pool` (sibling of `activation_ledger`; the
  activation ledger's event vocabulary stays DSL-lever-scoped — the #400 agent's finding — this store
  holds TOOL/stage/data candidates WITHOUT overloading that schema).
- **Digest:** `tools/costate_digest.py` §curriculum-pool — read-only, score-neutral, defaults ON,
  fail-open; ranks built-never-fired first, then needs-build, then reformulation-queue; surfaces the
  top next-fireable rows beside the lever duty-to-measure line.
- **DSL leg:** two completeness()-gap folds landed (`HardnessOversample`, `HeadGeometry`); every other
  row records dsl_na_reason inline (tool-side / vehicle-level / not-yet-built — folding those as stub
  levers would violate never-half-wire).
- **Equations leg:** N/A-with-rationale — this landing measures nothing and mints no law; the rows CITE
  existing registered equations/anchors (`quadratic_head_chart_subset_solve_gap_v1`,
  `dash_erasure_homogenization_v1`, `step_native_activation_edge_optimality_v1`,
  `curvelet_directional_basis_dseg_reduction_v1`) rather than duplicating them.

## 8. Follow-up tasks for the main agent to create (I cannot TaskCreate)

1. **SWA-tail-soup build** — "Averaging rung: SWA/model-soup across TAIL-cycle endpoints — design memo
   + byte-close-side soup tool + A/B vs PolyakFinisher and EMA shadow (EMA never replaced); gate:
   design-memo-first, fires terminal-band." (pool row §3.1)
2. **GN/Fisher head full-P in-trainer solve** — "#341 follow-through: full-P in-trainer GPU CG solve of
   the quadratic head chart (natural-gradient head block at stage boundaries); subset-solve measured
   NO-GO (+5.1% net), only the full-P solve admissible (~11min/CG-iter@17×)." (pool row §3.2)
3. **In-run click/proposal interleave design memo** — "Post-#400-measurement: if the diagonal n600 sweep
   pays, design the in-run stage-boundary discrete-proposal round (trainer-side; CONTAINMENT: design
   only)." (pool row §3.3)
4. **Hardness-oversample fair A/B** — "Fire LEVER-5: `--dsl-lever HardnessOversample` weighted-on vs
   weighted-off at fixed oversample 0.5, source=realized, n600 governed arm; close the activation-ledger
   loop." (pool row §2.1)
5. **#218 ETF-head rung fire** — "Fire `HeadGeometry(head='etf')` as its own increment A/B (byte-free;
   compose with armed Menon facet-3)." (pool row §2.4)
6. **γ* re-check before focal fire** — "$0: re-run the #301 γ* Shannon-equalization calibration on the
   live v7.5.2 ckpt (C17 measured γ*=0 — decide focal fire/hold; BoundaryDistance unaffected)." (§2.3)

---

**Means, not ends:** pointer **0.19108282 [contest-CPU]** (the 07-10 n8-click row) — NOTHING in this
memo moves it. This landing converts a chat inventory into a tracked, ranked, controller-held queue;
only a byte-closed n600 `upstream/evaluate.py` row moves the pointer.
