# MEANS audit (4 enumerations) — negatives / open-blockers / derivation-owed / Fourier+cargo-cult — 2026-07-15

**Mode:** `research_only=true`. READ-ONLY on code/ledgers/DSL. $0 — NO training, NO paid dispatch,
NO score claim. Pointer **UNMOVED 0.19108** submittable / **0.18804** borrowed non-submission bank.
Everything below is MEANS. Method: costate_digest + DAG-tail + deferral/harness ledgers read directly;
4 parallel read-only agents over the 07-13..15 corpus (~200 files); load-bearing claims re-verified at
source (cited). Labels: MEASURED (through-R/real-scorer), DERIVED (registry/math), INFERRED (not yet a
row). Authority ladder: only `upstream/evaluate.py` <0.19108 moves the pointer; `[macOS advisory]`/MLX/
MPS/n<600 are MEANS.

**Headline (adversarial, confirmatory):** the corpus is ALREADY honestly re-scoped — the #498/07-14
audits pushed every recent NO-GO to INSTANCE/FORMULATION with a tracked reactivation (D41–D53). **0
PARADIGM-dead, 0 FAMILY-dead** in the window. The value here is (a) one consolidated ranked view, (b)
two QC corrections to load-bearing claims, (c) the highest-EV $0/cheap reopens, (d) the single structural
wall that gates the score: **the default launch path is still Fourier and no clean V9·CGauge C0 baseline
exists**, so the levers that could move d_seg are all BUILD-then-GO blocked behind one owed converged run.

---

## QC CORRECTIONS (load-bearing; verified at source this session)

- **#501 F1 "provenance gates ABSENT from preflight.py" is STALE / OVERTURNED.** VERIFIED at HEAD:
  `check_config_flag_provenance_bijection_complete` (preflight.py:1416 + v9_provenance_gates.py:653),
  `check_v9_fake_claim_guards` (1429 / 789), `check_evidence_authority_claims_are_custodied` (1455 / 926)
  all EXIST and ARE wired into `preflight_all()` **warn-only** at preflight.py:6469-6471 (`strict=False`).
  This matches the CLAUDE.md 2026-07-15 reconciliation banner. Correct statement: the "no bare constants"
  ladder is **gate-DETECTED (warn-only), strict-flip OWED** pending live-count-0 backfill — NOT
  "human-review-only." (Agent-3 meta-finding #4 partially retracted; the fake-audit's own grep missed them.)
- **The live default basis is Fourier and the code says so honestly.** VERIFIED: `--basis` argparse
  default = `legacy_fourier_ab_control` (train_levelset:13124); the deprecated `polar_fourier` token
  normalizes to it (help text). The live feature generator `polar_directional_fourier_B`
  (lever_b_levelset_generator.py:136) docstring states verbatim *"It is not spatially localized and is not
  a curvelet or shearlet frame"* — the historical `curvelet_*` naming is INSPIRED-ONLY. So the ban-gate is
  correctly warn-only and the strict-flip is honest-pending, not a hidden fake.

---

## ENUM 1 — NEGATIVES, verdict-scope re-graded (ranked by score-relevance)

Scope ladder INSTANCE < FORMULATION < FAMILY < PARADIGM. `authority`: AUTHORITY = through-R/real-scorer
n600; SURROGATE = proxy / un-through-R / n<600 / disengaged-regime / apparatus-confound. **Every row is
already tracked (D-row / FEED cited); none is a fresh over-scope leak.**

| # | verdict (numbers) | recorded scope | correct scope | authority | reopenable? | ref |
|---|---|---|---|---|---|---|
| 1 | **flicker floor** — witness at temporal-majority d_seg **0.005318**; sub-0.15 (0.0008–0.0012) = 4.5–7× below; un-warped hood = 32% of flips, AA-irremovable | FORMULATION-closed (warp/label-smooth) | correct | **AUTHORITY** (through-R n600) | **NO for warp vehicle** — family OPEN via appearance-PHASE endgame (spikes deterministic, proof 0.00086, BUILT default-OFF L86) | L85 / rescope-498 hard-wall |
| 2 | **RIPO binary trust-region** `‖Δlogit‖≤√(δ/p₁)` FALSIFIED; Spearman r_bin vs r_dir = **−0.9601** (near-perfect opposite ordering), ratio med 16.3×/worst 1025× | FORMULATION (wrong scalar transfer locus) | correct | AUTHORITY-ish (real SegNet, n96 → advisory) + DERIVED law | corrected law `|t|≤√(8·δ_KL/C_wr)` OPEN; **it is a NO-FAKE guard** (blocks re-adding the wrong law), not a score mover | FEED-ripo-d42 / D42 |
| 3 | **dual-metric no-solve (Bregman)** = SQUARED-Hessian `ΔθᵀH²Δθ` ≠ Fisher-natural `ΔθᵀHΔθ`; differs 600/600 SPD, err ~9e-13 | FORMULATION (identity false; name-preserving fake if built) | correct | AUTHORITY (measured, SYNTHETIC SPD fixture — not real n600) | family OPEN via typed H⁻¹ solve → #500/#501/#504; NO-FAKE guard | rescope-498 hard-wall |
| 4 | **taper +18% NO-GO** (DsegAwareTaper, rank-1 duty 78.9%) | recorded INSTANCE | INSTANCE (under-converged ckpt) | **⚠ SURROGATE** — measured on under-converged ckpt, no converged EMA-matched A/B | **YES** — converged n600 ON/OFF, identical EMA + exact non-treatment custody | D44 |
| 5 | **SPS separation NO-GO** — measured at **ep275 where the temporal-screw term is DISENGAGED (≈0 gradient)** + n=4 counterfactual | over-dispositioned toward design/family NO-GO | **INSTANCE** (disengaged = uninformative) | **⚠ SURROGATE — HIGHEST-risk load-bearing** (zero-gradient regime ≠ conflict authority; n=4≠n600) | **YES** — re-measure screw/phase-ENGAGED n600; scalarize/stratify before PCGrad. Already re-graded DEFER by 07-13 wave | negative_audit_wave N6 / D46 |
| 6 | **D41 margin-adaptive uniform QDQ** `NO_ADMITTED_PRECISION_IN_LADDER` — no bit-width 8..24 holds SegNet argmax n600; binding residual = fp32 argmax TIES (min-margin 0..5e-7); bit-ALLOCATION-INVARIANT (w25→w26: 13→3 flips) | INSTANCE (global uniform fixed-scale QDQ) | **INSTANCE, FORMULATION-DEAD-ON-HEADROOM** | AUTHORITY (n600 cert `d41_margin_waterfill_reopen_certificate`) but `[macOS-CPU 1-thread advisory]` | per-channel×bit + tie-tolerant interval bound — **reopen NOT $0** (headroom moved EV to D51/int64-determinism) | FEED / D41 |
| 7 | **custom sparse-adjoint Metal wall** — whole-net **0.7078× (SLOWDOWN)**, η=0.3205 vs DERIVED 2.2086× ceiling; encoder 114/125 shapes 98.3% support (no sparsity) | INSTANCE/FORMULATION (whole-net M5/MLX) | correct — DOMINATED at whole-net | **AUTHORITY (optimal form)** — real Metal kernel; MEANS advisory | mostly CLOSED; hybrid layer-route (seg-head 1.63×) OPEN but EV LOW, gated on unbuilt oracle-mask predictor (rel-L2 0.514) | D43 |
| 8 | **D37 rate-law refinement** — net **384,637.9 bits** (gross 467,373.9 − 10,342 table), CI [373674,395236]; 5/5 folds pick M-bins=16/ξ=2 → `RESIDUAL_NON_GAUGE_STRUCTURE — M not a sufficient statistic` | FORMULATION (M insufficient) | correct | **AUTHORITY** (V9 EMA-best n600 surface) | reformulation implied (twist=0 → #483 Bousfield-descent owed); DAG D37 line stale at +318,586 | signal_loss_audit / MEMORY L-v8/DAG |
| 9 | **fixedpoint QDQ / YOPO / feature-ball / INSTANT / historical-microbatch / ANE-CoreML / costate-ZOH-K2** — `null`/`no-speedup`/`0.98×`/`0.5889`/`2-4× overturned`/`no joint 10×`/`NOT-ADMITTED` | all INSTANCE/FORMULATION | correct | SURROGATE (n<600 / non-ABBA / projected / no-Metal / disengaged) — MEANS | **YES at optimal form each** (per-channel schedule / sparse-audit cadence / suffix-bound / native primitive / same-SHA ABBA / device-residency proof / transported costate) | D47/D48a/b/c/D49/D50/D52 |
| 10 | **N1/N2/N3 shallow convex support-localizer** — exact ridge best cosine **0.00769**, rel-L2 1.00076; shallow retains **20.17%** mass vs oracle **52.78%** | narrow FAMILY (shallow-cheap-feature convex localizer at fixed replay) | correct | AUTHORITY for frozen-replay; **P9-SUSPECT if exported to live speed/score** (no receiver-closed measurement) | **YES** — change REPRESENTATION (deeper/global/nonlinear/on-policy), NOT another convex head | negative_audit_wave N1-N3 |
| 11 | **N10 raw-matrix MuonH hyperball** — hyperball removes functional radial scale; micro-A/B +15.13% treat / +88.5% control | FORMULATION (current arch) | correct | **⚠ SURROGATE** — 8-step proxy, wrong dynamics/host; **verdict correctly WITHHELD** (no fake committed) | YES — normalized/gain-decoupled sphere; `film_polar_chart_spel_finisher` BUILT-never-fired | negative_audit_wave N10 |
| 12 | **median-freeze "does not converge"** | recorded convergence verdict | **INSTANCE — NO physics verdict exists** | **⚠ SURROGATE — apparatus confound** (accepted-only median can't re-arm; ep_loss=0.0) | YES — liveness-proven clean-ckpt A/B w/ emitted update counts | D52a (CLAUDE.md L5) |
| 13 | **N4/N5 FORE/organ HCM** `NOT_IDENTIFIED` / `COUNT=0`; **LSI τ-anneal** `NO_VERDICT_SOURCE_CUSTODY` | identification/custody refusal | correct | AUTHORITY (honest refusal — a fitted number would be the fake) | OPEN prospectively — need logged (Z,A,R,Z') + propensity + retrieved-hashed paper | negative_audit_wave N4/N5 / codex_premise_falsification_lsi |
| 14 | **megakernel fp-reorder** `#356` — whole-step mx.compile fp32 fusion NEVER bit-identical (grad Δ 2.3e-7…2.3e-5) AND speed marginal (1.12–1.21×) | FORMULATION-closed | correct | **AUTHORITY** (measured) | family OPEN via explicit-order kernels (ON, #432) → D51 exact-integer/explicit-order megakernel | rescope-498 hard-wall |
| 15 | **N7 quant-cost premise** FALSIFIED (positive) — post-hoc int8 IMPROVES exact parsed Seg **−0.0004301** at one ckpt | INSTANCE (one ckpt/format) | correct | AUTHORITY (exact receiver-parsed) | QAT ticket stays CONFIRMATORY | negative_audit_wave N7 |

**⚠ Surrogate-not-authority callouts (load-bearing negatives on surrogate measurement — all already caught):**
row 5 (SPS, disengaged-loss + n=4 → design NO-GO; re-graded DEFER), row 4 (taper +18% on under-converged
ckpt, yet rank-1 duty-to-measure), row 11 (MuonH 8-step proxy — verdict correctly WITHHELD), row 12
(median-freeze apparatus confound), row 10 (N1-N3 exact only on frozen replay, P9-suspect if exported).
row 6 (D41) & row 7 (D43) & row 9 are `[macOS/MLX advisory]` but that is ACCEPTABLE — they are throughput
MEANS, never cited as a score.

---

## ENUM 2 — OPEN items with a NAMED, MEASURED blocker (ranked by score-relevance)

Blocker types: RUN (needs an n600/A-B run) · GO (operator-GO on heavy/governed launch) · ART (needs a
checkpoint/receipt/artifact that doesn't exist) · CUDA (needs contest-CUDA paired eval) · BUILD (needs
code/config/LawRef/consumer built). Compound rows list primary first.

### Tier 1 — could lower exact-eval d_seg (levers + basis A/Bs)
| item | blocker (NAMED+MEASURED) | type | unblock | ref |
|---|---|---|---|---|
| **TAPER-ISO** (DsegAwareTaper 78.9%) | V9 config `v9_cgauge_432_taper_off` does NOT exist; dropping the base taper Lever fails the V9 expected-active-lever/provenance manifest; `--dsl-lever X` shortcut FORBIDDEN (silently emits weight 0.0) | BUILD→GO | build one-delta config (drop taper Lever+LawRefs, keep all else, one-delta compile test) → GO + lane-claim + governed fire | sweep_failures:83 |
| **HORIZON-ISO** (HorizonWeightedMargin 47.3%, never-fired) | 7 scientific LawRefs + measured-weight consumer ABSENT from V9 bijection; would emit default weight 0.0 (forbidden) | BUILD→GO | add `v9_cgauge_432_horizon_iso` + boundary-receipt schema + 7 LawRefs + trainer consumer + refusal test → GO | sweep_failures:97 |
| **STEP-ISO** (StepNativeActivation 34.2%, never-fired) | β_end 8.0 conflicts sealed V9 β 3.177; needs distinct scientific declaration | BUILD→GO | add `v9_cgauge_432_step_iso` matching LawRefs/manifest/consumer + one-activation-delta test → GO | sweep_failures:113 |
| **curvelet_throughR_p0 A/B** (control polar_fourier vs windowed_curvelet) — **IS the owed `curvelet_through_R_dseg_ab` anchor gating the no-Fourier strict-flip** | `PREPARED_NOT_FIRED_OPERATOR_GO_REQUIRED`; `realized_d_seg_row=OWED`; **also curvelet arm REFUSES self_orient/ground-frame/taper/render_aa≠none → can only run a STRIPPED non-shipping config** | GO (+BUILD to compose w/ shipping) | run both dry_run_argv (governor/storage/DSL/readiness pass) → operator-GO fire n600 both arms → `levelset_byte_close_and_eval --run-exact-eval` | curvelet_throughR_p0_launch_ticket / train_levelset:4001-4014 |
| **genuine_frame 3-arm basis A/B** (fourier/curvelet/shearlet) | ordered gate `V9_TYPED_COMPOSITION`=FALSE: scientific-decl sha mismatch (expected 6cfa9845 vs live 5c926130) + reseal owed; standalone verifier bug (`surrogate_vjp_fidelity_policy.py` `binding.get` conflates absent vs explicit-null receipt); `v9_integration_status=PENDING_OWNER` | BUILD | provenance owner reseals + registers 3 configs whose only delta is one BasisLeverSpec + typed affine-Legendre gauge pair; fix binding.get | genuine_frame_fresh_start_3arm_ticket |
| **HORIZON×STEP / AA-SUPER2 / ETF-HEAD / POLAR-FINISH** | held: require BOTH isolated arms measured first / missing V9 typed bindings | RUN/BUILD | fire isolated arms first, then compose | sweep_failures:133 |

### Tier 2 — rate/receiver/scorer-geometry (MET-but-unwired or artifact-blocked)
| item | blocker | type | unblock | ref |
|---|---|---|---|---|
| **D21a blind-coordinate generic fill** (rule-118 FREE rate lever, ~230,904 blind px/frame) | MET/ROUTED: `tac.through_r.blind_coordinate.apply_blind_fill` + proof EXIST but `levelset_byte_close_and_eval.py` does NOT consume them | BUILD | insert apply_blind_fill before raw/archive selection; require n600 bit-identity receipt | D21a / sweep_failures:65 |
| **D18 latent-table TRUNCATE-at-export** (free-rate lever) | NOT-MET: no FINAL V9 ckpt + no `{stage:mod_dim_dynamics}` k90 telemetry; PCA machinery `witness_code_pca_byteclose.py` already exists | ART | produce terminal V9 ckpt + k90 series → `witness_code_pca_byteclose --ks <k90>` through exact receiver | D18 |
| **D24a SegNet margin-gradient-tail receipt** (edge-locality / no-factorization proof) | MET/ROUTED: raw 9.15/3.56/2.21% radii-64/128/192 tail + J(edge←region) block-matrix receipt ABSENT | RUN/ART | run registered n600 radius/block-Jacobian probe binding scorer/source/cache hashes BEFORE any locality claim (`# FORMALIZATION_PENDING`) | D24a |
| **D27b terminal solve-upon-basin stack** (#341 GN/CG + #396 MC + #400 PairLocalDiagonal + HeadOffset) | NOT-MET: no custodied `d27b_ready=true`; solvers BUILT but trigger needs muon-fired AND trailing d_seg rel-slope <5e-3 terminal band | RUN | reach terminal-band converged V9 ckpt → fire ordered solvers on ckpt | D27b |
| **D1 GPU-vs-CPU verdict agreement** / **D9 GPU-verdict promotion** | governor-blocked, no agreement data; D9 gated on D1 | RUN(GO)/CUDA | `safe_run --label d1_gpu_verdict_probe ... d1_gpu_verdict_agreement_probe_n600.py --chunk-seconds 460` | D1/D9 |
| **D38 H² obstruction (global gluing)** | PARTIAL-CLOSED: local strict extension derived; global H_cov gluing NOT-TYPED (required before any global twist-rate claim) | BUILD | rate-law math lane types the global gluing | D38 |

### Tier 3 — telemetry/producer builds that UNBLOCK Tier-1/2 (the real bottleneck)
| item | blocker | type | unblock | ref |
|---|---|---|---|---|
| **next_launch_all_levers ticket** (28 levers) | 4 measured blockers → rc=11: (1) `witness_component_wallclock.v1` 8-field producer MISSING; (2) `sps_gradient_role_conflict_engagement.v1` hook MISSING; (3) memory-waterfill B=2 n600 UNMEASURED (waterfill picks B=1); (4) SSD root `/Volumes/VertigoDataTier/pact/...` mkdir `PermissionError` | BUILD+ART+GO | land D-A/D-B producers; fix SSD root; harvest B1/B2 into waterfill; recompile → require launch_blockers=[] → GO | launch_prego_worklist:124 |
| **GO_PACKET in-loop component timer** (resolves 82%-backward pivot; produces D_A timers) | OPERATOR-GO REQUIRED; instrument BUILT but D-A wiring at trainer:9343 UNCOMMITTED | GO(+BUILD) | operator GO + pick land-D-A vs worktree; run bounded n24/4ep governed profile | GO_PACKET_inloop_component_timer |
| **D39 marked-event telemetry** / **D40 organ causal-OPE identification** | D39 spec landed, producer UNIMPLEMENTED; D40 deterministic walk-forward → off-policy unidentifiable | BUILD(→RUN) | implement marked-event rows (resume-safe) + log ε-greedy/randomized propensity in next schedule-arm launch | D39/D40 |
| **D5/D6/D8/D11/D15/D16/D17/D19** (fp16 cache · async-verdict reclaim · attribution-consolidation · dashboard polish · micro-batch ABBA · Metal #212 kernels · safe-compile GPU re-cert · speed bundle) | ALL NOT-MET for ONE reason: **no clean converged V9·CGauge C0 baseline / first governed stop exists** | RUN | launch + converge ONE clean V9 C0 → this entire cluster fires at its baseline/stop | D-rows 154-168 |
| **D27 reactivation campaign** (13 pinned governed commands) | old v7 argv must NOT fire verbatim; each survivor needs recompile as V9 typed variant + lane-claim + governor admission | BUILD→GO | recompile each as V9 typed variant → claim lane → pass governor → operator-GO | D27 |

### Tier 4 — throughput/apparatus MEANS (low score-relevance) + blocked-on-identity
- D41(→BUILD-RUN, one V9 wire + full receipt) · D42(→RUN RE-CAPTURE, store-nothing deleted logits, NOT $0) ·
  R-D43-hybrid(→BUILD kernel+mask predictor) · D44/45/46/49/52a/b/c(→RUN matched converged n600 A/B) ·
  D47/48a/b/c/50/51(→BUILD native primitive + n600 receipt).
- **GO_PACKET ANE ∥ full-trainer A/B** blocked by `LADDER↔Muon STAGGER VIOLATION` (lane/movable windows
  340/260 not strictly before `--muon-start-epoch 4`) → schedule owner restores typed 4-epoch schedule.
- **GO_PACKET P0 K2 costate reuse** SUPERSEDED → `FIDELITY_BLOCKED_FUTURE_TEMPLATE` (K2 admitted factor 1.0×).
- **3 consolidation apparatus bugs** OWED two-landing (retry inherits pre-CFL sandbox + non-resumable →
  re-launches 4h job from 0 up to 8×; in-flight pre-CFL arms not retrofitted/capped; drain-detector TIMEOUT
  exits 0 = give-up looks like success).
- **2 harness still-open** (zsh_nomatch_glob → monitor-lint refusal; codex_probe_token_limit → chunk below
  ultra token wall + landing-review gate).
- **D53** provenance for transient #495 = BLOCKED-IDENTITY (ART; `repoint_dismissed_intake` must register
  canonical identity before any verdict).

**Cross-cutting (the score-relevant truth):** the dominant blocker is **BUILD of V9 typed configs, not GO
or CUDA.** All three top duty-to-measure levers + both basis A/Bs hit the SAME wall — the V9·CGauge
provenance bijection refuses any arm whose config-id/LawRef/consumer/receipt does not yet exist, and
`--dsl-lever X` is forbidden as a shortcut. AND ~10 rows (D1/D2/D5/D6/D8/D11/D15/D16/D17/D19/D27b) are all
"ARMED, trigger NOT-MET" for the SINGLE reason that **no clean converged V9·CGauge C0 run / governed stop
exists.** Two infra blockers gate every governed launch RIGHT NOW: SSD root `mkdir PermissionError`
(`/Volumes/VertigoDataTier/pact`) and system-governor memory REFUSE (full stack 114.5>100.1 GiB; trimmed
self-orient-OFF admits 67.3<100.2 but B=2 n600 still unmeasured). needs-CUDA is rare (only D1/D9).

---

## ENUM 3 — DERIVATION-OWED (constants/schedules not on DERIVED>CONFIG>ANCHOR>WAIVER)

Operational classifier = `lever_registry.completeness()` (the sweep_arm_B matrix); the `NEEDS_DERIVATION`
bucket = exactly 3 argparse flags (`--muon-lr`, `--l7-mult`, `--l7-threshold`). Ranked by score-relevance.

| # | constant/schedule | current value / provenance | derivation owed (named method) | ref |
|---|---|---|---|---|
| 1 | **curriculum stage clock / fallback epochs** (τ≈300, l7≈800, muon≈726) | INHERITED-PR95 (event-GATING derived, the CLOCK is echo) | witness-native anneal from level-set continuation / Morse-persistence order (#302); event sensors (`birth_completion`/`annulus_plateau`/`powerlaw_meat`) replace clocks | sweep_arm_B:54,93; timer_curriculum:150 |
| 2 | **curriculum stage loss weights** (per-stage w_seg/reg) | INHERITED-PR95 PARTIAL (order justified, weights echo) | witness-native per-stage weight from level-set energy (#302/#430); DSL emits TrainerSupportGap until consumer exists | optimal_metric_training_loss_curriculum:51 |
| 3 | **`--muon-lr`** = 0.1×lr (abs 0.002) | GUESSED / INHERITED-PR95 (NOT in `Muon` factory; auto-derive `0.1*lr` at train_levelset:9346,10422; campaign.py:450) | witness-native Muon base-LR from modular-norm / Manifold-Muon tangent-step theory (#302); B4 folds flag into factory but must NOT hardcode 0.1× | sweep_arm_B:72,102 |
| 4 | **`hosc_beta_end`** — live argv **3.177** vs manifest **10.0** vs derived-candidate **8.0** | **PROVENANCE-DEFECT (3 disagreeing owners; INSTANCE-invalid)** — 3.177 is a dead-launch.sh emit, not derived | derive step-native endpoint from argmax-cell sharpness need (StepNative→8.0 candidate, UNMEASURED); compiler must derive manifest FROM emitted argv + refuse dup owner. **BLOCKS a clean governed launch** | v9_cgauge_truly_optimal:304; v9_cgauge_claim_corrections:35 |
| 5 | **#500/#504 nat-grad preconditioner** — trust-region radius + λ_s damping | metric math DERIVED (categorical-Fisher = Bregman Hessian); **H⁻¹ solve + radius UNBUILT** (no-solve dual is squared-Hessian) | build damped H⁻¹ trust-region solve `u=-η G⁺∇L` + RIPO exact-KL/Fisher radius; IMPLEMENTATION_CUSTODY that live V9 realizes the pullback. **NOT a bare constant — an unbuilt solver** | bregman_all_surfaces_504:75; ripo_fisher_isometric_trust_region_build_spec:117 |
| 6 | **AdamW β₁ / eps / eps-in-vs-out-sqrt / bias-correction** | LLM-tuned/INHERITED (β₂ already DERIVED #222/#223) | per-module norm geometry (Bernstein sign/max-norm); audit MLX-vs-torch AdamW semantics (eps placement matters for small island/lane grads) | muon_dig_directive_adamw_optimality:1 |
| 7 | **weight-decay coeff** | INHERITED, verdict UNRESOLVED (harmful? rate/MDL lever?) | derive from objective: is wd a rate regularizer (flat-minima/MDL #242), what it buys in bytes vs d_seg | muon_dig_directive:20 |
| 8 | **RUN-GATED folded-lever weights** — EikonalStEik.weight · EikonalJunctionRelax · CodeNuclearNorm · SegSpikeReweight · MicroBatch.pairs · BoundaryDistance · DsegAwareTaper · LambdaPreProbe.iters · SpikeGuardRollback.{frac,lr_cut,window,max} | GUESSED — held+composable, weight run-gated (no asserted optimum, correctly per discipline) | per-lever A/B through real n600 verdict → `EmpiricalAnchor` row per lever (operator-GO) | sweep_arm_B:51,60,124 |
| 9 | **per-class birth-weight ABSOLUTE λ_c / δ** | RATIO derived (∝(P/A)_c, Lane 8.9× Movable); ABSOLUTE UNMEASURED | live quasi-static W_birth up/down ramp on EMA-BEST resume → measure λ_c ± hysteresis, δ falls out (operator-GO, NOT $0) | island_birth_saddle_node_hysteresis:92 |
| 10 | **`--l7-mult` / `--l7-threshold`** | INHERITED; **l7 = MEASURED DEFECT** (demote) | moot until #302 witness-native anneal replaces l7 stage | sweep_arm_B:54 |
| 11 | **MarginBandSatisficing `msafe`** — factory 0.06 vs derived 2.0×0.0196=0.0392 | PROVENANCE-DEFECT (2 disagreeing owners; REFUSE until one selected) | reconcile: pick derived 0.0392 (headroom×δR) or re-measure; refuse compile until single owner | v9_cgauge_truly_optimal (DERIVED 7) |
| 12 | ClosedLoopEikonalControl defaults (bump 0.05/ceil 0.2/≤2/3wins) · CodeSpectralEntropy 0.01 · StepNative endpoint 8.0 + FreshFrequencyShift bias · micro-batch pairs 8 | ASSUMED/DSL (INFERRED-favorable, magnitude unmeasured) | measure containment magnitude / rate-vs-rank tradeoff / isolated-basin A/B / full-step functional-parity receipt | v9_cgauge_truly_optimal (DERIVED 8/10/11/15) |

**Meta-finding:** the largest still-owed cluster (rows 1,2,3,10) all route to **#302 witness-native
curriculum + muon-lr** — replacing PR95-echo *schedule/optimizer clocks* with level-set-continuation-derived
sensors. This is **DESIGN-blocked, not run-blocked.** rows 4 & 11 are live PROVENANCE-DEFECTS (multiple
disagreeing owners) that should be resolved before the next governed launch. **Already-derived (do NOT
re-open):** adaptive-ε #318/#320 (floor 0.3/upper 0.7/margin 0.5); β₂-from-n #222/#223; anisotropic σ_cc'
#382 (Road–Lane 0.377); verdict-batch 32 #495; Muon warm-start+anneal #269/#272 (−32% vs AdamW); Bregman/
Legendre identities #504 (math closed, only the H⁻¹-solve BUILD owed); Muon aspect 6.3578=√(768/19);
Beta2WindowRewarmup horizon 14; fused-R #348; megakernel NO-GO #356.

---

## ENUM 4 — FOURIER-REPLACEMENT + CARGO-CULT → RECURSIVE-FRACTAL-OPTIMAL

### PART A — Fourier sites still Fourier AND live-run-relevant
Gate `check_no_fourier_basis_in_witness_representation` (v9_provenance_gates.py:813 / preflight.py:1442,
warn-only 6471). Codex inventory: **BASIS 86 files / 646 line-sites · TOOL 17/144 · REPLACEMENT 11/132;
behavior-changing safe swaps completed = 0** (every owned occurrence changes feature geometry / receiver
bytes / a replay surface — no $0 import swap exists). **The strict-flip blocker is uniform: the ONE owed
operator-GO n600 byte-closed realized-through-R no-d_seg-regression A/B (`curvelet_through_R_dseg_ab`,
UNMEASURED — the OMP 1.09× / spectral 1.7–2.0× are explicit UPPER BOUNDS, `score_claim=false`).**

| fourier_site | file:line | live_path? | replacement_status | blocker |
|---|---|---|---|---|
| `--basis` default = `legacy_fourier_ab_control` | train_levelset:13124 | **DEFAULT-LIVE** | owed-AB | curvelet_through_R_dseg_ab UNMEASURED (GO n600 A/B) |
| default front-end `B=polar_directional_fourier_B(...)` + `curv_feats_np=curvelet_feats(coords,B)` | train_levelset:4022,4027 | **DEFAULT-LIVE** (actual live feature tensor) | owed-AB | same |
| polar-Fourier plane-wave bank `polar_directional_fourier_B` / `curvelet_feats` (`sin/cos(2πX@B)`; docstring self-labels "not a curvelet/shearlet frame") | lever_b_levelset_generator.py:136-184 | **DEFAULT-LIVE** (source of default basis) | owed-AB (behavioral decoder; no $0 swap) | same + receiver/byte-close byte migration |
| self-orient directional-Fourier feats `self_orientation_directional_feats` (concatenated when `self_orient=True`, which proven_base SETS) | lever_b_generator.py (imported train_levelset:87; tag :4161) | **DEFAULT-LIVE** (proven_base self_orient=True) | owed-AB — **AND curvelet arm REFUSES --self-orient (:4003) → no replacement wired for shipping config** | curvelet must first support self_orient composition, THEN n600 A/B [INFERRED] |
| IPE polar-Fourier attenuation render `aa_sdf_observation_render` | aa_sdf_observation_render.py:33,172,193 | **DEFAULT-LIVE** (render_aa="ipe" decode path) | owed-AB / behavioral-migration | curvelet arm refuses render_aa≠none (:4009) |
| d_seg-aware Fourier taper actuator | dseg_aware_fourier_taper.py:131 (gate train_levelset:4071) | opt-in (default False :13161) | owed-AB (own lever) | default OFF; curvelet arm also refuses (:4007) |
| `LEGACY_FOURIER_AB_CONTROL` canonical control | witness_dsl/basis_control.py:14; curriculum_dsl.py:1937 | **control-arm** | **control-kept** (retain until curvelet wins, then delete) | retained BY DESIGN as the A/B baseline |
| torch parity Fourier taper | train_levelset..._torch.py:1102; train_witness..._R.py:106 | opt-in/parity (advisory, not MLX launch path) | owed-AB (parity twin) | migrates with MLX default |
| `windowed_curvelet`+`compact_shearlet_frame` (#502 genuine) | train_levelset:4034; boundary_math/{windowed_curvelet_frame,compact_shearlet_frame}.py | opt-in **REPLACEMENT** (treatment arm) | **replaced, PREPARED_NOT_FIRED** | awaiting same n600 A/B |
| FFT spectral-sensitivity / losses / geometry DFT | scorer_spectral_sensitivity_v2.py; losses/core.py:2164; research/geometry_deliberation.py | **TOOL** (measurement, not decoder) | keep (not basis) | needs `# FFT_TOOL_USE_OK:` waivers before strict flip |

**Load-bearing:** the shipping config (`witness_autoconfig.proven_base`: self_orient=True, render_aa="ipe",
witness_autoconfig.py:2195) is Fourier, and the genuine curvelet arm cannot currently COMPOSE with it (it
refuses those flags), so the owed A/B can only run a **stripped non-shipping** config. Closing the ban
strict-flip needs BOTH the through-R d_seg row AND curvelet↔shipping-config composition [INFERRED from
:4001-4014 + autoconfig:2195]. **FAMILY selection already MEASURED: curvelet WINS over shearlet at matched
budget; both beat Fourier ~1.20× (OMP upper bound only).**

### PART B — Cargo-culted structures owing optimal-per-dimension re-derivation
All `DERIVED_UNMEASURED` (formulation-scope), each gated on a real-n600 through-R byte-closed row. d_seg is
100% of the sub-0.19 gap (L68), so RGB/boundary/loss rows rank highest.

| cargo-culted structure | current form | why cargo-culted | re-derivation owed (named) | ref |
|---|---|---|---|---|
| **RGB appearance representation** | dense 3-ch RGB head / RGB coord-INR before boundary render | RGB-native state stores evaluator-null directions; scorer objective is reachable decision geometry | **DecisionCarrierBundle** (partition φ + winner-rival tie residual + ξ + low-rank PoseNet-YUV6 tangent + sparse chroma decision coeffs; RGB only at render) | train_levelset:1765; amortized_luma_carrier:18 (H1) |
| **boundary basis** | global constant-envelope polar-Fourier plane waves ("curvelet"-named) | provably sub-optimal on 41×-anisotropic boundary spectrum | oriented localized frame (genuine #502, per-orientation waterfilled; rank atoms by generalized-eig / score-unit-per-byte under G_θ) | =PART A #1 (H6) |
| **score-aware recon loss** | uniform/saliency RGB MSE foundation | penalizes evaluator-null RGB dirs; can dominate task term | reachable-decision pullback `argmax_native_vjp_fidelity_v1` + PoseNet-6 tangent trust + exact bytes (RGB MSE diagnostic only) | mlx_score_aware/loss.py:510 (H2) |
| **palette initializer** | mean-GT-RGB per-class centroid | centroid objective not scored; empirical win ≠ task optimality | 15-D palette solve in winner/rival decision geometry + Pose-6 trust + min-byte per scorer cell | train_levelset:1768,4331 (H3) |
| **annulus chroma match** | Euclidean GT BT.601 chroma MSE | ambient-chroma match is a proxy for a winner/rival flip | luma-null chroma plane projected through C·D_xT·D_R; min-norm margin-crossing w/ uint8 survival | chroma_boundary_match.py:66 (H4) |
| **costate/surrogate admission** | ambient RGB cosine + rel-L2 + norm-ratio | ambient space has renderer-null dirs (19-D pullback ↑cosine 12.5×) | vjp_fidelity reachable-decision preconditioner (winner/rival Fisher, on-policy density custody) | segnet_gradient_replacement.py:184 (H5/M1) |
| **pose appearance carrier** | Fourier-coord full-RGB frame generator | PoseNet scored output is 6-D; full RGB synthesis unneeded | ξ/ground-homography + low-rank residual in PoseNet-YUV6 Jacobian coords (RGB at render only) | amortized_luma_carrier:11 (H8) |
| **HiNeRV bootstrap objective** | RGB pair MSE + target-region RGB MSE + YUV6 pixel MSE + task margins | RGB recon spends gradient on irrelevant appearance | keep hard-birth/winner-rival; replace pixel tethers with decision/pose tangents + stage-boundary trust | hi_nerv/mlx_renderer.py:2559 (H7) |
| **curriculum SCHEDULE (epoch)** | PR95-echo CE→τ→l7→Muon | vehicle witness-derived, schedule still PR95 inheritance | witness-native continuation schedule + per-class birth-weight ∝(P/A)_c (#302) | =ENUM3 #1; MEMORY L6 |
| **Muon finishing schedule** | Muon kept (−32%) with un-tuned finish | optimizer derived-kept, finish adopted-not-tuned | anneal-LR + warm-start-momentum finish (#217) | MEMORY L78 |
| **UNIWARD luma-gradient texture prior** | spend margin at low realized-RGB luma gradient | already MEASURED INERT (Pearson −0.033 vs S_R; Jaccard 0.024≈chance) | through-R margin-Jacobian reachability field (+decision-Fisher/byte marginal value) | train_levelset:6253 (M3) |
| **HPRC / NSCS06-v8 / cool-chic / z8 RD acquisition** | rank by decoder-grid RGB MSE-per-KiB (+PSNR); chroma-LUT arbitrated by RGB L2/PSNR | ambient RGB is the policy arbiter; MSE not score-unit | rank by exact/decision-predicted Δ[100·d_seg+√(10·d_pose)] per byte; section-Jacobian → score-units-per-byte (MSE = decoder-parity constraint only) | hprc/learned_receiver.py:968; nscs06_v8/...:21; z8/...:270 (M4/M5/M8) |
| **PIXEL "task-store-is-low-rank" assumption** | assumed pixel/palette low-rank → cheap store | **REFUTED** — pair-0 palette response rank 15/15 in Seg decision space | **none — re-derived AWAY**: pixel is a READOUT (argmax evaluation), NOT a store dimension | codex_findings_recursive_fractal_v9 |

**Sister-status blocker:** the #503 arm returned `V9_INTEGRATION_BLOCKED_OWNER` / `NO_VERDICT_RECEIVER_RATE_CUSTODY`
— DecisionCarrierBundle surfaces are built-default-OFF but NOT live-wired into V9·CGauge and no byte-saving
is proven. **HARD-EARNED, do NOT "fix" (10 surfaces):** final 3-ch RGB decoder (SegNet input contract),
exact R/uint8/resize/color, SegNet-argmax d_seg, PoseNet RGB→YUV6+first-6 MSE, exact archive bytes,
bit-identical parity, chroma-as-DOF, store-nothing ξ warp, tie-coordinate phase residual, exact-S selector.

---

## BOTTOM LINE (ranked, for the next execution window)

1. **The score is gated by ONE structural fact, not by dead ideas:** the default launch path is Fourier
   and NO clean converged V9·CGauge C0 baseline exists. Every d_seg-moving lever (Taper/Horizon/Step),
   both basis A/Bs, and ~10 armed deferrals unblock the moment one clean C0 run lands — but that is
   BUILD-then-GO gated (V9 typed configs + provenance bijection), and two infra blockers (SSD-root
   `mkdir PermissionError`, governor memory REFUSE) currently return rc=11 on every governed launch.
2. **0 PARADIGM-dead, 0 FAMILY-dead** in the 3-day window. The only surrogate-load-bearing mis-scope
   (SPS design NO-GO from a DISENGAGED loss + n=4) was already re-graded DEFER by the 07-13 wave. No
   fresh signal loss.
3. **Highest-EV $0/cheap reopens** (recompute from cached artifacts, no launch): D42 categorical-Fisher
   surrogate metric on cached logits (BUT store-nothing may force RE-CAPTURE → not truly $0); D37 rate-law
   number update (DAG D37 line stale at +318,586 vs measured 384,637.9); D24a is a RUN not $0. D41's $0
   reopen is CLOSED-on-headroom (per-channel×bit is not $0).
4. **Two PROVENANCE-DEFECTS to fix before any governed launch:** `hosc_beta_end` (argv 3.177 vs manifest
   10.0 vs derived 8.0) and `MarginBandSatisficing msafe` (0.06 vs 0.0392) — multiple disagreeing owners.
5. **Two NO-FAKE guards standing (not score movers, keep):** RIPO binary-transfer falsification
   (Spearman −0.9601) and Bregman dual-metric = squared-Hessian — both block re-introducing name-preserving
   fake laws; the families stay OPEN via the categorical-Fisher `|t|≤√(8·δ_KL/C_wr)` law + typed H⁻¹ solve.
6. **QC:** the #501 "provenance gates absent" finding is STALE — gates exist + wired warn-only; strict-flip
   OWED. The largest derivation-owed cluster routes to #302 (witness-native curriculum + muon-lr), which is
   DESIGN-blocked, not run-blocked.

Pointer delta: 0.0000000000. This audit is MEANS; the pointer moves only through `upstream/evaluate.py`
< 0.19108 on a byte-closed archive.
