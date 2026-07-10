# DEFAULT-OFF COMPREHENSIVE SWEEP — THE DECISION TABLE (task #405) — 2026-07-10

**Operator P0 escalation (2026-07-10, verbatim, x2 same-day recurrence):** *"All default off stuff you
keep forgetting despite me asking for comprehensive sweep and analysis for true final optimal, that is
terrible orphan class as well."*

**The diagnosis this fixes:** the duty-to-measure queue EXISTS (66 owed rows in `tools/costate_digest.py`)
but was never DRAINED — prior sweeps produced INVENTORIES, not per-row DISPOSITIONS that config
finalization is forced to consume. This memo is the missing artifact: one row per default-off surface,
each with a disposition, an owner, and a fire/gate criterion — plus a machine-readable twin
(`.omx/research/default_off_decision_table_20260710.jsonl`, 185 rows + header) and a warn-only
consume-gate (`check_default_off_decision_table_consumed`) so future config finalizations cannot
silently skip it.

**Pointer honesty:** contest-CPU pointer is **0.19108282** (n8 click row, 2026-07-10). This memo is
MEANS, not goal progress — the pointer moves only through a byte-closed `upstream/evaluate.py` n600 row.

## STORES CONSULTED

Machine sources (enumerated, never hand-grep alone):
- `tac.witness_dsl.activation_ledger.duty_to_measure_ranked()` — **70 rows (66 owed)**, ranked by
  %-of-remaining-descent (P8 floor-aware), pointer-anchored 0.19108282 → 0.15.
- `tac.witness_dsl.lever_registry.completeness()` — **261 mapped flags / 107 unmapped**, `stale == []`.
- `.omx/state/lever_activation_ledger.jsonl` (23 rows) + `.omx/state/lever_relative_significance*` via
  `canonicalize_significance_keys` (the ONLY significance-store reader).
- `experiments/results/levelset_v752_pilot_20260710T154100Z/launch.sh` — the sealed v7.5.2 launch
  config (READ-ONLY; live dry-start in flight), 157 distinct flags.
- `.omx/state/deferral_ledger.md` — 35 open D-rows (D4 hygiene pass 2026-07-09 + burn-down 2026-07-10).

Prior sweeps CONSUMED, cited, NOT redone:
- **#225 orphaned-signal sweep** (cited via sweep-C line 20) + **max-signal paranoia sweep, surface C**
  `.omx/research/sweep_C_task_research_orphan_lever_ledger_20260704.md` — the lever-gate inventory.
- **#303 Phase A costate de-orphan inventory** `.omx/research/costate_controller_deorphan_inventory_20260705.md`.
- **#367 optimal-form 3-audit synthesis** `.omx/research/v75_optimal_form_actuation_spec_20260708.md`
  (Audits A/B + the "turn on all optimal via DSL" checklist — realized in the v7.5.2 seal).
- **#377 DSL-completeness build-wave** `.omx/research/buildwave_dsl_completeness_20260709.md` — the
  **complete 107-flag disposition** (68 FOLD-OWED-#332 / 38 BASE-CONFIG / 1 MEASURED-EXCLUDED),
  verified programmatically here (`unmapped − table == {}` and `table − unmapped == {}`).
- **#397 reactivation campaign** `.omx/research/reactivation_campaign_397_20260710.md` — the 28-item
  negcure JOIN feeding the v7.5.3 ranked pool.
- **v7.5.3 ladder + terminal band** `.omx/research/fullstack_fractal_optimal_synthesis_20260710.md`
  (Δ4 rungs 1–10 + operator_go + Δ5 terminal 4a/4a'/4b/4c) — the CONSUMPTION TARGET of this table.
- Coverage-gate sister: `check_significance_keys_canonical` (`src/tac/confound_gates.py`) — the P1
  one-fact-one-store-one-key gate this memo's §6 findings extend to FIRED-event keys.
- CLAUDE.md §"'Off' is a tracked queue, never a forgotten default" + §triality "DSL HOLDS every
  designed lever" + `docs/operating_manual_craft_handoff.md` (§3 risk-ranking, §5 label out loud,
  §8 anti-goldplating) + memory `curriculum_candidate_pool_p0_orphan_class_20260710` (sibling #403).

## EXECUTIVE — TOP-10 BY EV, WITH DISPOSITIONS (feeds the v7.5.3 ladder directly)

| # | Row | EV (% of remaining descent) | Disposition | One-line |
|---|---|---|---|---|
| 1 | **DsegAwareTaper** | 73.0 ESTIMATED (~0.03 S) | measure-cheap | ACTIVE in v7.5.2 (fired-unmeasured); +18% NO-GO RETRACTED as under-converged, converged anchors flip to −8% — RE-VALIDATE by disk A/B on a converged ckpt |
| 2 | **HorizonWeightedMargin** | 43.8 MEASURED ceiling 0.012–0.024 | **fire-now-rung(v7.5.3)** | BUILT+FIREABLE, genuinely never-fired; gate: converged n600 byte-close A/B, surviving flips must shift to HIGHER GT margin |
| 3 | **StepNativeActivation** | 31.6 MEASURED (−4.5% n600 screen) | **fire-now-rung(v7.5.3 rung 6)** | adopt-verdict owed vs sealed hosc β_end 3.177 (with FinerBiasInit as the FINER++ arm) |
| 4 | **latent_table_truncate_d18_k90** | 2.4 ESTIMATED | fire-terminal-band | D18 ARMED; machinery exists (`witness_code_pca_byteclose --ks` + k90 sensor); blocker = NO FINAL CKPT |
| 5 | **mod32_neutrality_19_ab** | 1.2 ESTIMATED | fire-terminal-band | fold into the same stop-time byte-close A/B as D18 |
| 6 | **AACoverageRender** | UNMEASURED (oracle-R lane-recall +0.38 MEASURED) | measure-cheap | ipe proxy already ON in v7.5.2; supersample-authority through-training ΔS = duty-to-ESTIMATE then converged A/B |
| 7 | **MarginSaliencyReachability** | UNMEASURED (rung-1, cache ready) | **fire-now-rung(v7.5.3 rung 1)** | exact through-R S_R; BUILT + `gt_n600_sR.npz` ready + NEVER FIRED; entry: sal=w·sR lowers n600 d_seg |
| 8 | **LanePrior (paint_then_sdf)** | UNMEASURED (FN 3× mask-level MEASURED) | **fire-now-rung(v7.5.3 rung 3)** | #397-r4: paint-mode nucleates FN 0.0058→0.0019; entry: realized d_seg ↓ |
| 9 | **DashComb in-training** | post-hoc +0.0038 OOD | fire-now-rung(v7.5.3 rung 2) | ledger: fired+measured post-hoc; the in-training arm is the ladder rung (render-post-hoc-dead law) |
| 10 | **TextureTrunk 3-arm (+OutTexHidden)** | UNMEASURED (pins d_seg*(T)) | **fire-now(v7.5.3 launch-gating)** | the pre-compose 3-arm A/B in the v7.5.3 launch-gating chain; D26 exact-D home gated on the 3 v753 P0s |

(Rows 2/3/7/8/9 map 1:1 onto ladder rungs; the ladder in `fullstack_fractal_optimal_synthesis_20260710.md`
already consumed the #397 pool — this table confirms it drains the TOP of the duty queue, and adds the
rung-independent owed items: #1 taper re-validation, #4/#5 terminal-band byte cuts, #6 AA estimate.)

## §1 COVERAGE ACCOUNTING (no silent skips)

**185 data rows** across four surfaces; every row has a disposition or sits in §5 BLOCKED:

| Surface | Rows | Source |
|---|---|---|
| S1 duty-queue (activation ledger + significance store) | **70** (66 owed) | `duty_to_measure_ranked()` |
| S2 DSL-unmapped trainer flags | **107** | `completeness().unmapped`, dispositions CONSUMED from buildwave-#377 (verified: missing=[] extra=[]) |
| S3 tools-layer default-off | **8** | mc_finisher #400 · D18 wire · #401/D21 · #402/D22 · #336 bit-alloc · D24 receipt · D26 texture home · deferral-ledger reference |
| S4 trainer-argparse residual | **0 new rows** (claim) | 368 trainer flags = 261 registry-mapped + 107 unmapped; every flag is therefore covered by S1-lever membership or S2. A generic knob with no swept intent is NOT a lever (CLAUDE.md triality); the sealed config supplies its value via the DSL compile. |

Machine-readable twin: `.omx/research/default_off_decision_table_20260710.jsonl` (header + 185 rows;
fields: name · surface · why_off (+label) · ever_fired · active_in_v752_launch · ev · disposition ·
disposition_reason · owner · fire_or_gate_criterion · flags).

**Disposition distribution (S1):** measure-cheap 39 · fire-now-rung(v7.5.3) 13 ·
keep-off-with-DERIVED-reason 11 · fire-terminal-band 3 · retire-with-reason 2 ·
fire-next-vehicle(v8) 1 · BLOCKED 1.

## §2 S1 — THE DUTY-QUEUE DECISION TABLE (70 rows, EV-ranked)

Columns: row · fire-state (ON-v752 = flag+value verified in the sealed launch.sh; else the raw ledger
state) · ever_fired · EV · disposition · justification (MEASURED/DERIVED/ESTIMATED labeled inline) ·
owner · fire/gate criterion.

| Row | fire-state | fired | EV | Disposition | Justification | Owner | Gate |
|---|---|---|---|---|---|---|---|
| DsegAwareTaper | ON-v752 | Y | 73.0% ESTIMATED | measure-cheap($0/n600) | ACTIVE in v7.5.2 (fired-unmeasured). #1 EV 73%: +18% NO-GO RETRACTED (under-converged); converged anchors flip to -8% (~0.03 S). RE-VALIDATE at convergence — cheap disk A | #121 owner + costate queue | converged ckpt exists (>= tau stage); disk A/B |
| HorizonWeightedMargin | ledger:never-fired | n | 43.8% MEASURED | fire-now-rung(v7.5.3) | EV #2 43.8% (MEASURED oracle ceiling 0.012-0.024). BUILT+FIREABLE (DSL factory + trainer L4953 + reference twin + eq horizon_weighted_margin_hinge_v1); genuinely never-fi | v7.5.3 ladder (margin-family, with rung 5) | converged n600 byte-close A/B; surviving flips must shift to HIGHER GT margin, e |
| StepNativeActivation | ledger:never-fired | n | 31.6% MEASURED | fire-now-rung(v7.5.3) | EV #3 31.6% (MEASURED -4.5% n600 screen; -18.7% n100). Ladder rung 6 stepnative_or_finerpp; adopt-verdict owed vs hosc beta_end 3.177. | v7.5.3 ladder rung 6 | n600 adopt-verdict arm vs sealed hosc |
| latent_table_truncate_d18_k90 | ledger:not-registered | n | 2.4% ESTIMATED | fire-terminal-band(D27b) | D18 ARMED: truncate code table to measured k90 at export (machinery EXISTS: witness_code_pca_byteclose --ks + mod_dim_dynamics k90 sensor default-ON). Named blocker: NO F | D18 / #157/#336 waterfill consumer | chosen-chain FINAL ckpt exists |
| mod32_neutrality_19_ab | ledger:not-registered | n | 1.2% ESTIMATED | fire-terminal-band(D27b) | Fold into the same stop-time byte-close A/B alongside D18 (1.2%). Non-blocking != never. | D18 sibling | same stop-time A/B |
| AACoverageRender | ledger:never-fired | n | UNMEASURED | measure-cheap($0/n600) | ipe PROXY mode ACTIVE in v7.5.2 (--render-aa ipe); supersample AUTHORITY mode + through-training d_seg DeltaS = ASSUMED_AWAITING_VERIFICATION (oracle-R lane-recall +0.38  | #220 / costate queue | converged byte-close A/B, supersample vs ipe |
| AdamBeta2 | ledger:never-fired | n | UNMEASURED | keep-off-with-DERIVED-reason | beta2=0.999 SEALED by crucible (guard beta1<sqrt(beta2)). #222 sweep fires only if Beta2WindowRewarmup/stage-transition telemetry implicates the second-moment horizon. | crucible seal | telemetry trigger |
| AmplifyIsland | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--amplify-weight 1.0, hinge form, inverse_thickness persist). Owed: per-lever attribution at governed stop + fired-event ledger canon | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| AnalyticLaneBandTraining | ledger:never-fired | n | UNMEASURED | fire-now-rung(v7.5.3) | v7.5.3 Delta-3 (RANK-1 negcure join): band participates in TRAINED render from ep0 (~10 LOC OWED-BUILD); post-hoc-neutral verdict does not apply (render-post-hoc-dead law | v753 Delta-3 (synthesis) | OWED-BUILD wire + n600 falsifier |
| AnalyticLaneRenderBand | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--lane-render-band + band schedule, event-gated lane_nucleus). Owed: per-lever attribution at governed stop + fired-event ledger cano | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| AreaConstraintBirth | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — counter-force Lever-1 ON (--area-constraint-birth, classes 1,3). Owed: per-lever attribution at governed stop + fired-event ledger canoni | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| Beta2WindowRewarmup | ledger:never-fired | n | UNMEASURED | measure-cheap($0/n600) | R-7 finisher 1: DERIVE rewarmup window from 1/(1-beta2) memory horizon; launch uses fixed 8-epoch window. Cheap stage-transition A/B; value DERIVED not swept. | stop-time arm pool | stage-transition telemetry from v7.5.2 |
| BirthCompletionEvent | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — counter-force Lever-2 ON (--birth-completion-event + ramp). Owed: per-lever attribution at governed stop + fired-event ledger canonicaliz | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| BoundaryDistance | ledger:never-fired | n | UNMEASURED | measure-cheap($0/n600) | #301 loss-geometry (annulus concentration). OVERLAPS DsegAwareTaper + margin family — duty-to-ESTIMATE overlap before any fire (avoid double-counting the same annulus pre | #301 + costate queue | overlap analysis vs taper on converged ckpt |
| CacheGtSkeleton | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--cache-gt-skeleton); bit-identical speed lever by construction. Owed: per-lever attribution at governed stop + fired-event ledger ca | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| ClosedLoopEikonalControl | ledger:never-fired | n | UNMEASURED | keep-off-with-DERIVED-reason | CONTAINMENT-derived: in-run costate ACTUATION requires operator-GO (advisory-only autonomous). The SENSE half already runs (costate shadow observer auto-starts per govern | #247 costate boundary | operator-GO for actuation; shadow rows accumulate meanwhile |
| CodeSpectralEntropy | ledger:never-fired | n | UNMEASURED | measure-cheap($0/n600) | Half-2 of DM1Minimal (capacity penalty keeping code directions live). Same gate. | rung-8 family | same |
| CurriculumReanchorLevers | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--curriculum-reanchor-levers). Owed: per-lever attribution at governed stop + fired-event ledger canonicalization (activation ledger  | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| DM1Minimal | ledger:never-fired | n | UNMEASURED | measure-cheap($0/n600) | DM1 minimal cure = StiefelW + CodeSpectralEntropy (byte-free FiLM rank-collapse root halves; design memo 80/20). ESTIMATE first: PR(cov(code)) recovery on a converged ckp | rung-8 family / per_stage_fractal_optimizer memo | PR(cov(code)) telemetry from v7.5.2 + rung-8 arm |
| DecoupledField | ledger:never-fired | n | UNMEASURED | fire-next-vehicle(v8) | v8 B1 (#398) per-class decoupled-field partition head — IS the v8 architecture (SPEC_v8.1 increment-1a), not a v7.5 bolt-on. | v8 chain / SPEC_v8.1 | v8 increment-1a harness |
| DirectionalBasis | ledger:never-fired | n | UNMEASURED | keep-off-with-DERIVED-reason | DERIVED (owed-16 P9 RESOLVED-REFUTING 2026-07-10): realized-through-R transfer of the -48% proxy MEASURED ~0 (anchor owed16_realized_transfer_measured_zero); v7.5.2 launc | owed16 anchor | re-open only with a NEW realized-through-R mechanism |
| DirectionalBasisRebalance | ledger:never-fired | n | UNMEASURED | keep-off-with-DERIVED-reason | Same owed-16 refutation family: two-regime rebalance rides the self-orient basis that measured ~0 realized transfer. Freq-along deficit law stands as diagnosis, not as th | owed16 anchor | same as DirectionalBasis |
| EikonalViscosity | ledger:never-fired | n | UNMEASURED | fire-now-rung(v7.5.3) | OPERATOR-GO ladder row (operator_go: eikonal_viscosity_316): D1-era NO-GO KNOWN-TAINTED (spike-guard freeze confound RETRACTED, L4/L5); first FAIR n600 test from clean ep | v7.5.3 ladder operator_go | OPERATOR-GO explicit; clean ep100 ckpt |
| EventTriggeredCurriculum | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--curriculum-event-triggered + nucleus-guard). Owed: per-lever attribution at governed stop + fired-event ledger canonicalization (ac | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| FiLMFix | ledger:never-fired | n | UNMEASURED | fire-now-rung(v7.5.3) | Ladder rung 8 component (l235_softcosine_margin_tau_filmv2): attacks MEASURED FiLM participation-ratio collapse 3.34->1.19 (91.8% of per-pair variation in ONE axis). | v7.5.3 ladder rung 8 | entry: holds plateau-break without overfit |
| FinerBiasInit | ledger:never-fired | n | UNMEASURED | fire-now-rung(v7.5.3) | FINER++ variable-periodic first-layer bias init — the other half of rung 6 (beta-anneal 1->8 / FINER++ vs StepNative). | v7.5.3 ladder rung 6 | same rung-6 3-way arm |
| FusedRKernel | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--fused-r-kernel); n600 bit-identity re-check owed (L70; synthesis item 4). Owed: per-lever attribution at governed stop + fired-even | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| GroundFrameChart | ledger:never-fired | n | UNMEASURED | keep-off-with-DERIVED-reason | DERIVED chart-selection law (MEASURED n600 NO-GO): xi/ego-freeze pays only where the chart retains ego DOF — ground-frame lane xi-transport collapsed (memory lane-groundf | chart-law memory + eq | re-open only for a chart retaining ego DOF (horizon-frame family) |
| HeadOffsetSolver | ledger:measured | Y | UNMEASURED | fire-terminal-band(D27b) | #288 decode-time Laguerre per-class head-bias SOLVE (byte-free). MEASURED: OT area-mass-match HURT d_seg -> flip-weighted masses reformulation next (#397-r B1). Terminal- | terminal band 4a-family / #397-r B1 | flip-weighted reformulation + realized d_seg down on final ckpt |
| LadderIslandHomotopy | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--ladder-island-homotopy + full per-class schedule). Owed: per-lever attribution at governed stop + fired-event ledger canonicalizati | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| LanePrior | ledger:never-fired | n | UNMEASURED | fire-now-rung(v7.5.3) | Ladder rung 3 lane_prior_paint_then_sdf (#397-r4, C12/C13): --mode paint + thin-start 0; paint-mode nucleates FN 0.0058->0.0019 (3x, mask-level MEASURED). | v7.5.3 ladder rung 3 | entry: realized d_seg down |
| LengthSigma | ledger:never-fired | n | UNMEASURED | measure-cheap($0/n600) | Consumption path for the MEASURED Young's-law junction fit (junction_sigma_fit.json, commit 3571e5b65) — $0 wire, value DERIVED not swept. Candidate late-rung after margi | v753 candidate rung (post rung-5) | n600 arm consuming the fit JSON |
| LogitAdjust | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--logit-adjust-loss-tau 1.0, classes 3). Owed: per-lever attribution at governed stop + fired-event ledger canonicalization (activati | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| LrAnnealPin | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--lr-anneal-epochs 1000 + --lr-hold-frac 1.0). Owed: per-lever attribution at governed stop + fired-event ledger canonicalization (ac | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| MarginBandSatisficing | ledger:never-fired | n | UNMEASURED | fire-now-rung(v7.5.3) | P0 FORCE 2 (#360, derivation p0_forces_derivation §FORCE 2) — ladder rung 9 remaining_p0_forces. | v7.5.3 ladder rung 9 | rung 9 arm |
| MarginFieldHead | ledger:never-fired | n | UNMEASURED | measure-cheap($0/n600) | FEED-07b lever 3 partial (#218 facets 1b/3): per-class margin-hinge head weight, default 0.0=off byte-identical; composes with LEVER-3/4/B on shared _signed field. ESTIMA | margin family arms | family overlap analysis |
| MarginSaliency | ledger:never-fired | n | UNMEASURED | keep-off-with-DERIVED-reason | KKT-waterfill margin-saliency engaged LATE; composes with rung 1. DERIVED: fire only WITH the S_R weight if rung 1 pays (from-scratch margin starves interior). | rung-1 outcome | rung 1 verdict |
| MarginSaliencyReachability | ledger:never-fired | n | UNMEASURED | fire-now-rung(v7.5.3) | Ladder rung 1 msal_uni_sR: exact through-R S_R fragility-weighted margin-Jacobian; BUILT + gt_n600_sR.npz cache ready + NEVER FIRED (#268/L76; texture proxy measured iner | v7.5.3 ladder rung 1 (#268) | entry gate: sal=w*sR lowers n600 d_seg vs w-alone |
| MicroBatch | ledger:never-fired | n | UNMEASURED | fire-now-rung(v7.5.3) | Ladder rung 10 (#397-r9, D4/D15): 1.56-4x throughput; bit-identity impossible so bounded n600 A/B is the ONLY admission path (SCORE decision, compute facet). | v7.5.3 ladder rung 10 / D15 | bounded n600 A/B after chosen-chain baseline exists |
| Mod32SegOnlyControlBase | ledger:never-fired | n | UNMEASURED | keep-off-with-DERIVED-reason | CONTROL-config expression lever (mod32cap clean baseline as DELTA over proven_base) — an A/B BASE, deliberately not a pointer-mover (T3-designed control; MEMORY L2). | A/B harness | n/a (control) |
| Muon | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON, event-gated (--muon-start-event powerlaw_meat; start-epoch 726 fallback). Owed: per-lever attribution at governed stop + fired-event  | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| MuonWarmStart | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--muon-warm-start-momentum + --muon-lr-final-frac 0.1). Owed: per-lever attribution at governed stop + fired-event ledger canonicaliz | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| OutTexHidden | ledger:never-fired | n | UNMEASURED | fire-now-rung(v7.5.3) | #395 A2 = the matched-bytes MIDDLE arm of the same texture-trunk 3-arm A/B. | v753 launch-gating chain | same 3-arm |
| PersistenceTopology | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--persistence-loss-weight 1.0 + recall 1.0 + warmup 275 + classes 3). Owed: per-lever attribution at governed stop + fired-event ledg | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| PolyakFinisher | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ARMED (--polyak-finisher-arm, start ep2546); exports candidate ckpt beside EMA. Owed: per-lever attribution at governed stop + fired-even | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| PoseDecouple | ledger:never-fired | n | UNMEASURED | keep-off-with-DERIVED-reason | TRADE instrument (w_pose=0 capacity A/B). SUPERSEDED as config: v7.5.2 already gates pose to terminal (pose-blind until conditioning event) — the capacity benefit is real | superseded by TerminalPoseFinish gating | n/a |
| PoseFinishConditioningGate | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--pose-finish-engage-on sigma_min_plateau). Owed: per-lever attribution at governed stop + fired-event ledger canonicalization (activ | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| SafeCompileRegions | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--safe-compile-regions hosc_activation); byte-identical verified 0.0. Owed: per-lever attribution at governed stop + fired-event ledg | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| SeedIslandBirth | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--seed-islands + --witness-alone-island-loss). Owed: per-lever attribution at governed stop + fired-event ledger canonicalization (ac | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| SeedIslandEased | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--seed-island-eased). Owed: per-lever attribution at governed stop + fired-event ledger canonicalization (activation ledger says neve | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| SegChromaBoundary | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON, event-gated (weight 0.1, annulus_plateau event). Owed: per-lever attribution at governed stop + fired-event ledger canonicalization ( | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| SegFocalGamma | ledger:never-fired | n | UNMEASURED | measure-cheap($0/n600) | #301 focal-gamma ($0-calibrated, default 2.0 canonical). Same annulus family as BoundaryDistance — pick ONE via the same n600 arm. | #301 + costate queue | same family arm |
| SegFormUnifyTau | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--seg-form-unify-tau). Owed: per-lever attribution at governed stop + fired-event ledger canonicalization (activation ledger says nev | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| SoftBoundary | ledger:never-fired | n | UNMEASURED | measure-cheap($0/n600) | Tests soft-edge sub-pixel hypothesis; replaces the CONFOUNDED constant-beta~16 beta_steplim arm (review H2). Cheap stop-time arm. | stop-time arm pool | governed-stop arm slot |
| StiefelW | ledger:never-fired | n | UNMEASURED | measure-cheap($0/n600) | Half-1 of DM1Minimal (isometry projection). Same gate as DM1Minimal. | rung-8 family | same |
| StoreNothingPoseCarrier | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--pose-carrier --pose-carrier-source generated = carrier A store-nothing flavor). Owed: per-lever attribution at governed stop + fire | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| TailCycles | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--tail-cycles-max 2 + full tail schedule). Owed: per-lever attribution at governed stop + fired-event ledger canonicalization (activa | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| TauAdvanceEvent | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--tau-advance-mode event). Owed: per-lever attribution at governed stop + fired-event ledger canonicalization (activation ledger says | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| TauFrozen | ledger:never-fired | n | UNMEASURED | keep-off-with-DERIVED-reason | ISOLATION INSTRUMENT (A1b: freeze tau to isolate l7 from anneal) — an A/B control, never an optimum candidate. Fires only inside a specific attribution question. | attribution toolbox | n/a (instrument) |
| TemporalScrewConsistency | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON, event-gated (weight 0.1, ground_gt, annulus_plateau). Owed: per-lever attribution at governed stop + fired-event ledger canonicalizat | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| TerminalPoseFinish | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON, gated (--pose-finish-start-epoch 726 + engage-on sigma_min_plateau; banked R1 dxi fallback). Owed: per-lever attribution at governed  | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| TextureTrunk | ledger:never-fired | n | UNMEASURED | fire-now-rung(v7.5.3) | v7.5.3 launch-gating chain: texture-trunk 3-arm A/B (short warm-start arms; pins d_seg*(T)); D26 exact-D texture home gated on the 3 v753 P0s. #395 P0. | v753 launch-gating chain + D26 | 3-arm A/B before v753 compose |
| TieLocusDisplacement | ledger:never-fired | n | UNMEASURED | fire-now-rung(v7.5.3) | P0 FORCE 3 (#360; d_seg currency IS boundary displacement, FEED-PA; machinery BUILT ~L4559 default-off) — ladder rung 9. | v7.5.3 ladder rung 9 | rung 9 arm |
| UniWARD | ledger:never-fired | n | UNMEASURED | retire-with-reason | FORMULATION-scope retire: UNIWARD texture proxy MEASURED inert vs through-R reachability (Pearson -0.033, top-5% Jaccard 0.024 = chance; L76). Superseded by exact S_R (ru | superseded by #268 | n/a (verdict_scope: formulation) |
| VerdictDevice | ON-v752 | Y | UNMEASURED | keep-off-with-DERIVED-reason | cpu = authority-conservative sealed choice (MPS/GPU never authority). The HYBRID GPU-verdict+CPU-anchor mode is D1/D9-gated (GPU-vs-CPU agreement probe BEFORE promotion). | D1/D9 (deferral ledger) | D1 n600 agreement probe at chosen-chain pre-launch |
| WarpRealLumaFrame0 | ledger:never-fired | n | UNMEASURED | keep-off-with-DERIVED-reason | Pose carrier B (warps STORED real keyframe luma — COUNTED bytes). Seal chose carrier A (generated/store-nothing). DERIVED: B is the terminal-band FALLBACK if A misses the | terminal band 4c fallback chain | 4c rollback guard vs banked R1 0.001610 |
| WeightEntropyPenaltyMLX | ON-v752 | Y | UNMEASURED | measure-cheap($0/n600) | ACTIVE in sealed v7.5.2 launch — ON (--weight-entropy-penalty-lambda 15.0). Owed: per-lever attribution at governed stop + fired-event ledger canonicalization (activation | #403 (ledger schema) + governed-stop attribution | governed stop of v7.5.2 pilot; stage-diff attribution |
| WitnessStability | ledger:never-fired | n | UNMEASURED | BLOCKED | BLOCKED-pending-operator: D25 amber realization/waiver — advisory P0-5 witness_stability_amber preset vs inherited --grad-clip 1.0 + per-group; operator decision on the l | operator (D25) | operator decision |
| seg_chroma_boundary_276 | ledger:not-registered | n | UNMEASURED | retire-with-reason | DUPLICATE legacy significance key for SegChromaBoundary (ACTIVE in v7.5.2). Route: canonicalize alias (#403 request) or in-notes SIGNIFICANCE_KEY_OK waiver; the LEVER row | #403 alias map | n/a (dedup) |
| seg_down_weight_274 | ledger:not-registered | n | UNMEASURED | measure-cheap($0/n600) | BUILT seg down-weight lever, not yet a registered factory; duty-to-ESTIMATE inside the v7.5 optimal-combination set (standing seg play). | #332 fold + costate queue | estimate on converged ckpt; fold as factory if swept |

**The single most important structural finding in this table:** 28 of the 64 "never-fired" ledger rows
are in fact **ACTIVE in the sealed v7.5.2 launch** (flag+value verified in `launch.sh`) — see §6. Their
disposition is measure-cheap = per-lever ATTRIBUTION at the governed stop, not firing.

## §3 S2 — DSL-UNMAPPED FLAGS (107; dispositions CONSUMED from buildwave-#377, NOT re-derived)

Per CLAUDE.md anti-redo discipline, the complete per-flag disposition is
`.omx/research/buildwave_dsl_completeness_20260709.md` §A/§B/§C; this table VERIFIED its accounting
programmatically against today's `completeness().unmapped` (missing=[] extra=[]) and rolls every flag
into the JSONL twin with per-flag rows:

- **§A FOLD-OWED-#332 (68 flags, 17 clusters)** → disposition `keep-off-with-DERIVED-reason`, owner
  **#332**: genuine designed levers whose designed-ON magnitude is not a trainer default; folding with
  an invented magnitude would be a FAKE lever (NO-FAKE #1). Gate: fold as `Lever` factory WITH a cited
  designed value → row then enters the duty queue automatically.
- **§B BASE-CONFIG (38 flags, 8 clusters)** → disposition `retire-with-reason` (not levers): arch dims,
  Muon raw hyperparams (SPEC S5 row 11), optimizer base, curvelet bank, score-neutral runtime
  observability (defaults-ON class), resume/warm-start machinery, spike-guard SAFETY (L5), containment.
- **§C MEASURED-EXCLUDED (1 flag)** → `--mx-compile` retire-with-reason (fp-contraction flips the
  uint8-STE argmax; fused-R + SafeCompileRegions supersede).

Per-flag N/A-or-fold reason: present for ALL 107 rows in the JSONL twin (DSL-leg obligation satisfied
without new `emit_stub_lever` folds — deliberately: sibling #403 owns costate/lever-registry schema
surfaces this window, and §A folds require cited designed values that only their owning units can supply).

## §4 S3 — TOOLS-LAYER DEFAULT-OFF (8 rows)

| Row | Surface | EV | Disposition | Justification | Owner | Gate |
|---|---|---|---|---|---|---|
| mc_finisher_diagonal_400 | tools-layer (tools/mc_finisher — TOOL not Lever) | MEASURED (n8 click row 0.19108282 came through this family) | fire-terminal-band(D27b) | terminal-band 4a' (synthesis Delta-5): exact-metric ratchet fp32 then int8; n600 sweep in flight. | #400 / terminal band | final ckpt + n600 Modal sweep harvest |
| witness_code_pca_byteclose_k90_autofeed | tools-layer (tools/witness_code_pca_byteclose.py) | ESTIMATED | fire-terminal-band(D27b) | D18 ARMED; trivial wire deferred as premature until FINAL ckpt exists (named blocker). | D18 / #157/#336 consumer | chosen-chain FINAL ckpt |
| blind_coordinate_generic_fill_401 | tools-layer (byte-close receiver, tools/levelset_byte_close_ | UNMEASURED (rate-side, small) | fire-terminal-band(D27b) | D21 ARMED: wire into byte-close post-launch (chosen chain). | #401 / D21 | post-launch byte-close of the chosen chain |
| receiver_fail_closed_hardening_402 | tools-layer (byte-close receiver hardening) | N/A (compliance gate, not a score lever) | fire-terminal-band(D27b) | MANDATORY gate BEFORE #399 borrowed-bank dispatch closure AND any v7.5.3/v8 byte-close (short raw = NO-FAKE/compliance failure). | #402 / D22 | any byte-close run |
| perclass_sensitivity_bitalloc_336_sparc | tools-layer (export bit-allocation applier, #336) | UNMEASURED | fire-terminal-band(D27b) | stop-time export A/B alongside D18/#157 waterfill (same final-ckpt gate). | #336 | chosen-chain FINAL ckpt |
| margin_gradient_tail_receipt_D24 | tools-layer (scorer-geometry measurement receipt) | N/A (measurement receipt) | measure-cheap($0/n600) | next measurement window; BEFORE any edge-locality / no-factorization CLAIM (v8 gate input). | v8 / scorer-geometry owner (D24) | next measurement window |
| texture_home_exact_D_v753 | tools-layer + trainer (v7.5.3 exact-D texture home, D26) | UNMEASURED | fire-now-rung(v7.5.3) | v7.5.3 DESIGN/BUILD item gated on the 3 v753 P0s closing (MLX<->NumPy<->inflate one forward; counted bank exclusion; frame1-only home). | v753 build owner / D26 | 3 v753 P0s close |
| deferral_ledger_open_rows | tools-layer (deferral ledger .omx/state/deferral_ledger.md — | N/A (reference row) | keep-off-with-DERIVED-reason | CONSUMED-not-redone: dispositions live in the ledger (D4 hygiene pass re-pointed triggers to the chosen chain 2026-07-09; burn-down pass 2026-07-10).  | per-row owners | per-row named triggers |

Note: `#121 dseg-aware-taper` applier is ON in the sealed launch (S1 row 1); the GPU-verdict hybrid
(D1/D9) is dispositioned under S1 `VerdictDevice`; the D25 amber preset is §5 BLOCKED.

## §5 BLOCKED (1 row — named blocker, no silent skip)

| Row | Named blocker |
|---|---|
| **WitnessStability** (amber deep-unroll stability preset) | **D25 PENDING-OPERATOR**: amber realization/waiver decision on the live pilot's admission semantics (`levelset_v752_pilot_20260710T154100Z` inherited `--grad-clip 1.0` + `--per-group-grad-clip` instead of the `witness_stability_amber` preset). Cannot be dispositioned by an agent: it is an authority/admission-semantics call. Owner: operator + launch executor (deferral ledger D25). |

## §6 FIDELITY FINDINGS + SCHEMA REQUESTS ROUTED TO #403 (read-only here; no schema edits)

The table-building JOIN surfaced three apparatus defects. Per the coordination boundary (#403 owns
costate/lever-registry .py schema), these are REQUESTS, not edits:

1. **FIRED-event keys are not canonicalized (the big one).** `duty_to_measure_ranked` canonicalizes
   SIGNIFICANCE keys (`canonicalize_significance_keys`) but `activation_status(lever)` matches FIRED
   events by EXACT key — the ledger holds `n323_ladder_island_homotopy`, `v75_area_constraint_birth`,
   `seg_form_unify_tau`, `temporal_screw_consistency`, `pose_finish_conditioning_gate`,
   `R7_polyak_finisher`… which never match `LadderIslandHomotopy` etc. Net effect: **28 levers ACTIVE
   in the sealed v7.5.2 launch are reported "never-fired"** and the digest headline over-counts the
   owed queue. Request: apply the SAME alias canonicalization to fired/built/measured event keys
   (sister of `check_significance_keys_canonical`, extended from the significance store to the
   activation ledger), AND/OR auto-emit canonical fired events at DSL compile time from the emitted
   launch flags (the compile knows the factory names — zero-human-memory).
2. **`seg_chroma_boundary_276` legacy significance key** duplicates the held `SegChromaBoundary`
   factory → add to the alias map (or carry the in-notes `# SIGNIFICANCE_KEY_OK:` waiver).
3. **Duty-to-ESTIMATE auto-population:** 39 S1 rows are measure-cheap largely because they carry NO
   significance row (est=None → unranked). Request: when a `Lever` factory lands with a docstring
   citing a measured/derived anchor, auto-seed a significance row (label ESTIMATED, source_anchor =
   the docstring citation) so the ranking sees them — the current NULL-EV rows are exactly the
   re-orphaning vector this sweep exists to kill.

## §7 THE CONSUME-GATE (deliverable 2; warn-only per Strict-flip atomicity)

`check_default_off_decision_table_consumed` (`src/tac/preflight.py`, wired into `preflight_all`
warn-only beside the operating-manual + #362 SPEC-pointer anti-rot gates, whose pattern it copies):

1. this memo + the JSONL twin must exist; the twin must parse, carry the header `_meta` row, and every
   row must have `name`/`surface`/`disposition` with disposition in the enum;
2. any NEW config-finalization artifact (a `SPEC_*.md` under `.omx/research/t5_crucible*/` or a
   `crucible_*_authored_*.md`, with a filename date AFTER this table's date) must reference the
   decision table by name OR carry a `DEFAULT_OFF_TABLE_CONSUMED:` line naming the table version it
   consumed. Waiver: `# DEFAULT_OFF_TABLE_OK:<rationale>` (placeholder rejected, Catalog #287 sister).
Existing artifacts (dated ≤ 2026-07-10) are exempt → live count 0 at landing; the gate is
future-facing, exactly the warn-only pattern.

## §8 TRIALITY LEGS

- **DAG leg:** `### FEED-defaultoff-sweep` appended to
  `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **DSL leg:** zero new folds this unit (deliberate — §3 rationale); per-flag N/A/fold-owed reasons for
  all 107 unmapped flags present in the JSONL twin; §A folds owned by #332 with the cited-value gate.
- **Equations leg: N/A-with-rationale** — this unit is apparatus/disposition class (a decision table +
  a gate), no new measured law; every EV number cited here carries its ORIGINAL anchor
  (source_anchor field) registered by its owning unit.

**Pointer 0.19108282 [contest-CPU] UNMOVED by this unit — apparatus/means.** The exact-score path this
table serves: v7.5.3 ladder rungs (S1 fire-now rows) → governed EVENT launch → byte-closed n600 rows.
