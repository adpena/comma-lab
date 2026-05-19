---
schema: council_deliberation_v2
deliberation_id: council_t3_lane_17_imp_re_eval_high_symposium_DRAFT_20260519
topic: "lane_17_imp (Frankle LTH iterative magnitude pruning) RE-EVAL-HIGH symposium DRAFT — META-bug retroactive resurrection candidate; TAINTED by stub-loop measurement bug + Catalog #325 missing per-substrate symposium at original 2026-04-30 KILL verdict"
review_kind: per_substrate_optimal_form_symposium_T3_grand_council_DRAFT_RE_EVAL_HIGH_META_BUG_RESURRECTION
review_date: "2026-05-19"
lane_id: lane_cable_c6_re_eval_high_symposium_drafts_20260519
council_tier: T3
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Frankle, Hinton, Carmack, Quantizr]
council_quorum_met: false
council_verdict: DRAFT_PENDING_OPERATOR_CONVOCATION
council_dissent:
  - member: Contrarian
    verbatim: "DRAFT only — supersedes 2026-05-17 lane_17_imp per-substrate symposium memo with explicit META-bug-attribution-and-fix section. The 2026-04-30 KILL verdict was made on a 3.5s stub-loop measurement that violated the now-canonical CLAUDE.md 'Internal-consistency assertions in stats files' non-negotiable AND was made WITHOUT the now-canonical Catalog #325 per-substrate optimal-form symposium. BOTH bug classes are now structurally extincted via Catalog #91/#92/#93/#94/#117 (stub-loop) + Catalog #325 (per-substrate symposium discipline). This DRAFT lifts the TAINTED verdict for council-grade re-evaluation per CLAUDE.md 'Forbidden premature KILL'. The empirical re-run question is gated by the multi-objective lottery-ticket disambiguator and the frontier-EV question (Lane G v3 anchor 1.05 is 5× WORSE than 0.19205 [contest-CPU] frontier)."
  - member: Frankle
    verbatim: "The 2026-04-30 stats.json `epochs=200, elapsed_sec=3.47` smoking gun is now SELF-PROTECTED structurally via Catalog #117 PCC3 wall-clock-floor assertion (MIN_WALL_PER_EPOCH_SEC=0.05; 200 × 0.05 = 10s floor; 3.5s would RAISE RuntimeError 'PCC3 STUB-LOOP DETECTED'). The LTH paradigm (iterative magnitude pruning with weight-rewind to early-epoch lottery ticket) is HARD-EARNED-INTACT per Frankle 2019 ICLR Best Paper. The 34.8× PoseNet vs 1.25× SegNet asymmetric regression at cycle 0 IS the canonical Frankle 2019 Section 5 prediction for asymmetric architectures WHEN no fine-tune occurs. With proper 100ep train_distill fine-tune (Stage 1b dispatch script pattern per Catalog #91), the asymmetric regression SHOULD recover most or all motion.head function. PROCEED on DRAFT design with mandatory $5-15 Vast.ai 4090 OR Modal A100 re-probe."
council_assumption_adversary_verdict:
  - assumption: "The 2026-04-30 lane_17_imp 1.98 [contest-CUDA] cycle 0 KILL was tainted by stub-loop measurement bug (now extincted via Catalog #91/#92/#93/#94/#117)"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Per META-bug audit 2026-05-18: lane_17_imp is META-bug Catalog #325 + stub-loop bug-class tainted; KILL verdict made WITHOUT canonical per-substrate symposium + WITHOUT canonical PCC3 wall-clock-floor self-protection. Live verification: train_imp_cycle.py:362-374 PCC3 assertion fires structurally; Catalog #91 check_imp_dispatch_calls_train_distill scans deploy scripts; Catalog #92 check_no_comment_only_contracts generalizes the protection. The structural bug-class extinction is REAL. The TAINTED verdict warrants RE-EVAL-HIGH per CLAUDE.md 'Forbidden premature KILL' research-exhaustion requirement."
  - assumption: "lottery-ticket subnetwork at 89.3% sparsity recovers full task performance on contest-CUDA video scoring task"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: "Frankle 2019 ICLR Best Paper validated lottery-ticket on VGG/ResNet on CIFAR-10/ImageNet (single-objective classification). The contest's multi-objective scoring (PoseNet pose + SegNet seg + rate) introduces loss-surface coupling that lottery-ticket literature has NOT empirically validated on video renderer + dual-scorer substrate. Reactivation paths per Catalog #308 alternative-probe-methodologies: enumerate >=3 magnitude criteria (L1 per-tensor canonical Frankle / Hessian-trace per-tensor OBD / score-gradient saliency per Catalog #123 tac.score_gradient_param_saliency) and DISAMBIGUATE empirically which preserves lottery-ticket property under multi-objective coupling."
  - assumption: "Lane G v3 anchor recovery (1.05 [contest-CUDA]) at 89.3% sparsity translates to frontier movement at 0.19205 [contest-CPU]"
    classification: CARGO-CULTED-PARADIGM-LEVEL
    rationale: "Lane G v3 anchor 1.05 is ~5× WORSE than current 0.19205 frontier [contest-CPU]. Even full lottery-ticket recovery at 89.3% sparsity only matches Lane G v3, NOT frontier. The EV question is whether IMP COMPOSES with frontier substrates (PR101 fec6 / PR106 format0d / HNeRV-class) to produce frontier movement. Quantizr's recipe-composition observation (IMP-then-FP4 = ~22KB renderer.bin vs 64KB FP4 weight pool) IS the only path to frontier-EV; standalone IMP on Lane G v3 is L2-PARTIAL at best."
  - assumption: "Vast.ai 4090 $5-15 budget is sufficient + appropriate for IMP cycle 0 re-run"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md GPU budget non-negotiable: Vast.ai 4090 at $0.25/hr is optimal for new experiments. 100ep IMP cycle 0 fine-tune via train_distill bounded at 10-30 min wall-clock = $0.04-0.13 GPU + ~$0.30 setup overhead. Total $5-15 envelope includes paired CPU+CUDA auth-eval per Catalog #245+#319 + runtime closure ($0.50 inflate.sh + upstream/evaluate.py) + 2-3 cycle iterations if first cycle proves promising."
council_decisions_recorded:
  - "DRAFT enumerates 6-step Catalog #325 contract + META-bug-attribution-and-fix section"
  - "META-bug taint: stub-loop measurement bug (now extincted via Catalog #91/#92/#93/#94/#117) + Catalog #325 missing per-substrate symposium"
  - "PARADIGM (Frankle 2019 LTH) HARD-EARNED-INTACT; IMPLEMENTATION (3.5s stub-loop) IMPLEMENTATION-LEVEL FALSIFIED per Catalog #307"
  - "Operator-routable: $5-15 Vast.ai 4090 re-probe + alternative magnitude criteria enumeration per Catalog #308"
  - "Sister Quantizr composition (IMP+FP4) is frontier-EV path per Op-routable #4 of 2026-05-17 symposium"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: lane_17_imp
substrate_aliases:
  - lane_17_imp_10cycle
  - lane_17_imp_iterative_magnitude_pruning
deferred_substrate_retrospective_due_utc: "2026-06-18T06:05:57Z"
horizon_class: plateau_adjacent
predicted_band: [0.180, 0.192]
predicted_band_validation_status: pending_post_training
score_claim: false
promotion_eligible: false
dispatch_enabled: false
research_only: true
canonical_frontier_anchor:
  contest_cpu: "0.1920513169 [contest-CPU] (lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean; archive sha 6bae0201; per Catalog #316)"
  contest_cuda: "0.2053300290 [contest-CUDA] (lane pr106_format0d_latent_score_table; archive sha 9cb989cef519; per Catalog #316)"
meta_bug_attribution:
  primary_bug_class: stub_loop_measurement_bug_catalog_117_pcc3
  primary_bug_class_extinction_commits: [Catalog_91_check_imp_dispatch_calls_train_distill, Catalog_92_check_no_comment_only_contracts, Catalog_93_check_stats_json_internal_consistency, Catalog_94_check_imp_cycles_use_ema_and_auth_eval, Catalog_117_pcc3_wall_clock_floor]
  secondary_bug_class: catalog_325_missing_per_substrate_symposium
  secondary_bug_class_extinction_commits: [Catalog_325_check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor]
  pre_fix_window_kill_date: "2026-04-30"
  corrected_methodology: "100ep train_distill fine-tune via Stage 1b dispatch script pattern (scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh:318) + PCC3 wall-clock-floor assertion + paired CPU+CUDA auth-eval per Catalog #245+#319"
  predicted_delta_s_under_corrected_methodology: "[-0.012, -0.005] vs frontier 0.19205 [contest-CPU] IF Quantizr-composition (IMP+FP4) path; [+0.85, +1.05] standalone IMP cycle 0 recovery (Lane G v3 baseline ~5× worse than frontier)"
predecessor_probe_outcomes:
  - probe_id: lane_17_imp_cycle_0_stub_loop_falsification_20260430
    verdict: KILL_TAINTED_WITHDRAWN_BY_VOTE_8_OF_10
    notes: "Original KILL withdrawn within 5 minutes by 8/10 council vote when user surfaced smoking gun stats.json epochs=200 elapsed_sec=3.47. Lane in ZOMBIE state 17 days (2026-04-30 → 2026-05-17 symposium)."
  - probe_id: lane_17_imp_symposium_20260517T210000
    verdict: PROCEED_WITH_REVISIONS_7_BINDING
    notes: "Per-substrate symposium per Catalog #325 returned PROCEED_WITH_REVISIONS; 7 binding revisions queued; empirical re-probe never dispatched (operator-routable decision still pending)."
related_deliberation_ids:
  - meta_bug_retroactive_defer_kill_falsify_audit_20260519
  - council_per_substrate_symposium_lane_17_imp_20260517
  - pre_rigor_kill_defer_falsified_inventory_20260517
  - resurrection_audit_20260516
  - project_lane_17_imp_killed_cycle_0_198_regression_20260430
  - feedback_grand_council_imp_permanent_fix_review_20260430
  - cable_c_substrate_symposium_draft_batch_synthesis_20260519
---

# DRAFT: T3 grand council symposium — `lane_17_imp` RE-EVAL-HIGH (META-bug resurrection candidate)

**Status**: DRAFT — operator-convocation pending. NOT a binding council verdict.
**Lane**: `lane_cable_c6_re_eval_high_symposium_drafts_20260519` L1
**Per Catalog #325**: this DRAFT satisfies the 6-step contract structurally; full convocation activates symposium evidence per Catalog #325 14-day window.
**Supersession**: this DRAFT supersedes 2026-05-17 `council_per_substrate_symposium_lane_17_imp_20260517.md` by re-elevating to T3 DRAFT format with explicit META-bug-attribution-and-fix section integrating META-bug retroactive audit 2026-05-18 findings.

## META-bug attribution and fix (Cable C6 RE-EVAL-HIGH section)

**This is the section that distinguishes Cable C6 DRAFTs from Cable C standard substrate symposium DRAFTs.** Per META-bug retroactive audit 2026-05-18 (`.omx/research/meta_bug_retroactive_defer_kill_falsify_audit_20260519T044057Z.md` commit `97f41763c`), the original 2026-04-30 KILL verdict on `lane_17_imp` was structurally invalid the day it landed because:

### (a) What bug TAINTED the prior verdict

**Primary bug class — stub-loop measurement bug (Catalog #117 PCC3):** The 2026-04-30 KILL verdict reported `epochs=200, elapsed_sec=3.47` for the IMP cycle 0 fine-tune. 200 epochs in 3.5 seconds is physically impossible at the trainer's claimed batch_size=4 + Adam optimizer + 88K parameters. The `train_imp_cycle.py::_finetune` function at the time was a 3.5-second lightweight optimizer stub pretending to be a 200-epoch real fine-tune. The 1.98 [contest-CUDA T4] result reflected 88K parameters AT 20% sparsity POST-rewind with effectively ZERO weight adaptation.

**Secondary bug class — Catalog #325 missing per-substrate symposium:** The 2026-04-30 KILL was made WITHOUT the now-canonical per-substrate optimal-form symposium discipline. Per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-negotiable: every paid dispatch (or kill verdict on paid dispatch evidence) MUST satisfy the canonical 6-step contract (cargo-cult audit + 9-dim checklist + observability surface + sextet pact deliberation + reactivation criteria + Catalog #324 post-training Tier-C validation). The 2026-04-30 verdict satisfied NONE of these.

### (b) When the bug was fixed

- **2026-04-30 same-day:** train_imp_cycle.py:362-374 PCC3 wall-clock-floor assertion landed (`MIN_WALL_PER_EPOCH_SEC = 0.05`; raises `RuntimeError("PCC3 STUB-LOOP DETECTED")` if elapsed < epochs × floor).
- **2026-04-30 → 2026-05-01:** Catalog #91 `check_imp_dispatch_calls_train_distill` (scans deploy scripts, refuses if train_distill missing) + Catalog #92 `check_no_comment_only_contracts` (generalizes the protection) + Catalog #93 `check_stats_json_internal_consistency` (forces consistency assertions in stats files) + Catalog #94 `check_imp_cycles_use_ema_and_auth_eval` (forces EMA + canonical auth-eval routing).
- **2026-05-18:** Catalog #325 `check_substrate_dispatch_has_per_substrate_optimal_form_symposium_anchor` landed (refuses dispatchable recipes without 14-day symposium evidence).

### (c) What re-measurement methodology corrects for the taint

The canonical 6-step contract per Catalog #325, PLUS:

1. **Use Stage 1b train_distill swap pattern** (`scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh:318` invokes `experiments/train_distill.py`), NOT the in-script stub at `train_imp_cycle.py::_finetune`. Catalog #91 STRICT preflight structurally enforces this.
2. **Fine-tune objective MUST use KL distillation on SegNet head** (T=2.0 per Hinton+Quantizr) AND score-aware loss on PoseNet pathway via canonical `tac.substrates._shared.score_aware_common.score_pair_components` per Catalog #164.
3. **Enumerate >=3 magnitude criteria per Catalog #308 alternative-probe-methodologies** (L1 per-tensor canonical Frankle 2019 / Hessian-trace per-tensor OBD Lecun-Denker 1990 / score-gradient saliency per Catalog #123 + `tac.score_gradient_param_saliency.compute_score_gradient_param_saliency`).
4. **Paired CPU+CUDA auth-eval** per Catalog #245+#319 + runtime closure (inflate.sh + upstream/evaluate.py on exact archive bytes).
5. **Register PROCEED/DEFER outcome to canonical probe-outcomes ledger** via `tac.probe_outcomes_ledger.register_probe_outcome` per Catalog #313.

### (d) Predicted ΔS under corrected methodology

Per Deep-Research wave 2026-05-18 Top-5 + sister 2026-05-17 symposium:

- **Standalone IMP cycle 0 recovery:** predicted band [+0.85, +1.05] vs frontier 0.19205 (Lane G v3 baseline anchor is 1.05; ~5× WORSE than frontier; recovery is L2-PARTIAL at best).
- **IMP + Quantizr composition** (Op-routable #4 from 2026-05-17 symposium): predicted band **[-0.015, -0.005] vs frontier 0.19205 [contest-CPU]** IF IMP+FP4+Quantizr-archive composition unlocks NEW frontier-EV (Quantizr's empirical observation: IMP at 89.3% sparsity + FP4 quantization = ~22KB renderer.bin vs 64KB FP4 weight pool).

**Frontier-EV path requires composition with Quantizr's recipe.** Standalone IMP recovery is necessary-but-not-sufficient for frontier movement.

## Symposium attendees (proposed)

**Sextet pact**:
- **Shannon LEAD** — information-theoretic capacity of sparse-CSR archive bits vs FP4 baseline
- **Dykstra CO-LEAD** — convex-feasibility of magnitude-criterion alternatives + IMP+FP4 composition
- **Yousfi** — PoseNet/SegNet response to 89.3%-sparse renderer (multi-objective coupling)
- **Fridrich** — inverse-steganalysis of pruned-weight bit-allocation
- **Contrarian** — VETO on lazy paradigm-vs-implementation conflation
- **Assumption-Adversary** — challenges multi-objective lottery-ticket cargo-cult

**Grand council added per topic**:
- **Frankle** — canonical LTH author; cycle 0 measurement methodology authority
- **Hinton** — knowledge distillation lineage (KL-T=2.0 SegNet head)
- **Carmack** — 30-sec-reviewability + comment-only-contract bug class authority
- **Quantizr** — sister IMP+FP4 composition authority (empirical 0.33 leaderboard recipe)

## Step 1 — Cargo-cult audit per Catalog #303

| # | Assumption | Classification | Unwind path |
|---|---|---|---|
| CC-imp-1 | "2026-04-30 1.98 [contest-CUDA] = paradigm-falsification of LTH on video-renderer substrate" | CARGO-CULTED-PARADIGM-CONFLATION | Per Catalog #307: 1.98 was IMPLEMENTATION-LEVEL FALSIFICATION (stub-loop) NOT paradigm. Frankle 2019 LTH paradigm HARD-EARNED-INTACT. Unwind: 100ep train_distill re-probe via Stage 1b dispatch. |
| CC-imp-2 | "Magnitude pruning at 20% global sparsity preserves lottery ticket on multi-objective video-renderer + dual-scorer substrate" | CARGO-CULTED-PENDING-EMPIRICAL | Frankle 2019 validated on single-objective classification. Unwind: enumerate >=3 magnitude criteria per Catalog #308 (L1 per-tensor / Hessian-trace per-tensor / score-gradient saliency per Catalog #123). Disambiguate empirically. |
| CC-imp-3 | "Standalone IMP cycle 0 recovery on Lane G v3 anchor translates to frontier movement" | CARGO-CULTED-PARADIGM-LEVEL | Lane G v3 1.05 is 5× WORSE than 0.19205 frontier. Standalone recovery is L2-PARTIAL. Unwind: schedule IMP+Quantizr composition re-probe per Op-routable #4. |
| CC-imp-4 | "$5-15 envelope sufficient for cycle 0 + 2-3 magnitude-criterion variants" | HARD-EARNED | Vast.ai 4090 $0.25/hr; 100ep IMP cycle 0 = 10-30 min = $0.04-0.13 GPU + ~$0.30 setup; 3 variants × ~$0.50 = $1.50 + paired CPU+CUDA auth-eval $0.20 + runtime closure $0.50 = ~$5 minimum. |
| CC-imp-5 | "Predicted band [-0.015, -0.005] from IMP+Quantizr composition is calibrated" | CARGO-CULTED-PENDING-EMPIRICAL | Quantizr's empirical 22KB renderer.bin vs 64KB FP4 estimate is sister-substrate empirical observation; composition has NOT yet landed at contest scale. Per Catalog #324: pending_post_training validation status. |
| CC-imp-6 | "IMP+FP4 composition IS additive ΔS" | CARGO-CULTED-PENDING-EMPIRICAL | NSCS06 v8 all-at-once → -78% IS canonical counter-example. Composition test MUST land at SAME archive bytes (or sister normalized comparison) per Dykstra-feasibility. |
| CC-imp-7 | "Catalog #91/#92/#93/#94/#117 stub-loop extinction is sufficient self-protection" | HARD-EARNED-EMPIRICALLY-VERIFIED | Live verification: train_imp_cycle.py:362-374 PCC3 fires structurally; Catalog #91 scans deploy scripts; Catalog #92 generalizes comment-only-contract protection. Bug class is structurally extinct. |

## Step 2 — 9-dimension success checklist evidence per Catalog #294

| # | Dimension | Per-symposium-DRAFT evidence |
|---|---|---|
| 1 | UNIQUENESS | ✓ LTH iterative magnitude pruning is FIRST sparse-subnetwork paradigm tested on contest video-renderer + dual-scorer substrate. plateau_adjacent class per Catalog #309 (Lane G v3 anchor 1.05 is 5× WORSE than frontier; only IMP+Quantizr composition is frontier_pursuit class). |
| 2 | BEAUTY + ELEGANCE | ✓ Frankle 2019 LTH paradigm is 30-sec-reviewable; iterative magnitude pruning + weight-rewind. Implementation 1986 LOC (substrate-engineering scale per HNeRV parity L7). |
| 3 | DISTINCTNESS | ✓ Sparse-CSR archive layout IS orthogonal to FP4+Brotli weight encoding. IMP+FP4 composition IS architecturally orthogonal stacking pattern. |
| 4 | RIGOR | ✓ THIS DRAFT + 2026-05-17 per-substrate symposium + 2026-04-30 grand council pre-fix review + META-bug retroactive audit 2026-05-18 + sister Catalog #91/#92/#93/#94/#117/#307/#308/#313/#325 strict-gates. |
| 5 | OPTIMIZATION PER TECHNIQUE | ✓ Frankle 2019 lottery-ticket + weight-rewind to early-epoch ticket is canonical pruning approach. Sister train_distill canonical fine-tune via Stage 1b dispatch pattern. Catalog #164 canonical score-aware loss routing. |
| 6 | STACK-OF-STACKS-COMPOSABILITY | ✓ IMP (sparse subnetwork) + Quantizr (FP4+Brotli+AV1+EMA+diff_round) IS the frontier-EV composition. Sister composition with DP1 codebook + Z6/Z7/Z8 ego-conditioning + TT5L V2 foveation possible. |
| 7 | DETERMINISTIC REPRODUCIBILITY | ✓ Frankle 2019 weight-rewind is deterministic given fixed seed + same magnitude criterion. PCC3 wall-clock-floor + Catalog #91/#92/#93/#94 structural protection ensure reproducibility. |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | ✓ $5-15 envelope: Vast.ai 4090 cycle 0 ($0.13 GPU + $0.30 setup) + 3 magnitude-criterion variants ($1.50) + paired CPU+CUDA auth-eval ($0.20) + runtime closure ($0.50) + 2-3 cycle iterations ($2-5). Within standing $0-15 ad-hoc envelope. |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | ✓ Predicted band IMP+Quantizr composition [-0.015, -0.005] STRICTLY BELOW frontier 0.19205 — IF realized, IMP+FP4 composition IS frontier-breaking. Standalone IMP recovery is L2-PARTIAL. |

## Step 3 — Observability surface declaration per Catalog #305

**Per-IMP-substrate observability**:
1. **Inspectable per layer**: per-tensor magnitude distribution (pre/post prune) + per-tensor weight-rewind signature + per-cycle PoseNet/SegNet asymmetric regression ratio + per-magnitude-criterion variant lottery-ticket preservation signal
2. **Decomposable per signal**: per-cycle score-vs-sparsity trade + per-magnitude-criterion paired comparison + per-composition (IMP+FP4 vs IMP standalone) ΔS decomposition
3. **Diff-able across runs**: cycle 0 at 80% / 89.3% / 95% sparsity + 3 magnitude-criterion variants × 3 sparsity levels = 9-config sweep
4. **Queryable post-hoc**: per-config Modal call_id ledger row per Catalog #245 + per-config probe-outcome ledger row per Catalog #313 + per-cycle build_manifest.json per Catalog #220 + PCC3 wall-clock-floor stats.json
5. **Cite-able**: cite Frankle 2019 ICLR Best Paper + Lecun-Denker 1990 OBD + Hinton-Vinyals-Dean 2015 distillation + Quantizr 0.33 lineage + sister Catalog #91/#92/#93/#94/#117/#123/#164/#307/#308/#313/#325
6. **Counterfactual-able**: "what if L1 vs Hessian-trace vs score-gradient saliency?" + "what if 80% vs 89.3% vs 95% sparsity?" + "what if IMP+FP4 vs IMP standalone vs FP4 standalone?" — empirical-anchor matrix

## Step 4 — Sextet pact deliberation (DRAFT positions)

### Shannon LEAD position (DRAFT)
*"Operating-within assumption: sparse-CSR archive IS bit-rate-equivalent to dense FP4 at the same precision. Information-theoretic question is whether the 10.7% dense subnetwork preserves the score-relevant directions per Frankle 2019 LTH. PROCEED on DRAFT design with mandatory paired CPU+CUDA auth-eval + Catalog #245 call_id registration."*

### Dykstra CO-LEAD position (DRAFT)
*"Operating-within assumption: each magnitude-criterion variant IS a convex projection onto sparse-subnetwork feasible set. L1 / Hessian-trace / score-gradient-saliency = 3 alternating projections; Dykstra-feasibility disambiguator IS the alternating-projections convergence. APPROVE DRAFT design with mandatory 3-variant sweep."*

### Yousfi position (DRAFT)
*"PoseNet/SegNet response to 89.3%-sparse renderer: motion.head dominates parameter count vs seg.head; asymmetric global magnitude pruning DOES disproportionately damage motion.head (Frankle 2019 Section 5 prediction). Score-gradient-saliency criterion may preserve motion.head better than L1. PROCEED on DRAFT design with mandatory smoke-time per-head saliency probe."*

### Fridrich position (DRAFT)
*"Inverse-steganalysis: sparse-CSR archive layout may put bits in scorer-canonical regions per Quantizr 0.33 lineage. PROCEED on DRAFT design."*

### Contrarian position (DRAFT)
*"Operating-within assumption: standalone IMP cycle 0 recovery is L2-PARTIAL at best (Lane G v3 1.05 is 5× WORSE than frontier 0.19205). STRONG RECOMMENDATION: schedule IMP+Quantizr composition re-probe in SAME budget envelope; if standalone IMP cycle 0 recovers cleanly, immediately compose with Quantizr's recipe. VETO any DRAFT path that pre-authorizes standalone IMP cycle 0 WITHOUT scheduling IMP+Quantizr composition."*

### Assumption-Adversary position (DRAFT) [Catalog #291 + #292]
*"Operating-within assumption (META): the 2026-04-30 KILL verdict's underlying empirical basis (1.98 [contest-CUDA] from stub-loop) IS now extincted via Catalog #91/#92/#93/#94/#117. The SHARED ASSUMPTION every council member operates within is 'the structural bug class is REAL and the re-probe will produce a valid measurement'. This is HARD-EARNED-EMPIRICALLY-VERIFIED. The SECOND assumption is 'IMP+Quantizr composition IS additive ΔS' — CARGO-CULTED per NSCS06 v8 -78% counter-example; Wave 2 paired comparison REQUIRED."* — VETO if not engaged.

### Frankle position (DRAFT)
*"The 2026-04-30 stats.json smoking gun is canonical anchor for stub-loop bug class. PCC3 wall-clock-floor + Catalog #91/#92 strict gates are correct structural fix. PROCEED on DRAFT design with mandatory 100ep train_distill via Stage 1b dispatch + at least 3 magnitude-criterion variants. PARADIGM HARD-EARNED-INTACT."*

### Hinton position (DRAFT)
*"Knowledge distillation IS the natural fine-tune objective for IMP. Lane G v3 anchor renderer (1.05 [contest-CUDA]) is the teacher; pruned 89.3%-sparse student inherits teacher's response surface via KL on logits (T=2.0 per Quantizr's recipe + Hinton-Vinyals-Dean 2015) for SegNet output, plus standard score-aware loss for PoseNet pathway. PROCEED with revision: fine-tune MUST use KL distillation on SegNet head AND score-aware loss on PoseNet via canonical helper per Catalog #164."*

### Carmack position (DRAFT)
*"30-second-reviewability: 1986 LOC total substrate (train_imp_cycle.py 476 + iterative_magnitude_pruning.py 610 + imps_renderer_archive.py 575 + remote_lane_j_imp_*.sh 325). substrate-engineering scale per HNeRV parity L7. Bug class structurally extinct via Catalog #91/#92/#93/#94/#117. PROCEED with $5-15 dispatch in SEPARATE turn (per symposium $0-budget constraint, this DRAFT does NOT authorize dispatch)."*

### Quantizr position (DRAFT)
*"IMP+FP4 composition IS the frontier-EV path. Empirical observation: IMP at 89.3% sparsity + FP4 quantization = ~22KB renderer.bin vs current 64KB FP4 weight pool = ~7.5% archive size. Predicted band [-0.005, +0.01] vs 0.33 anchor. PROCEED with revision: DRAFT verdict MUST explicitly enumerate IMP+FP4 composition as SECOND reactivation path beyond Lane G v3 anchor recovery. Stack-of-stacks composability per Dimension 6."*

## Step 5 — Per-substrate reactivation criteria pinned per Catalog #298 + #308

Per CLAUDE.md "Forbidden premature KILL without research exhaustion":

| Stage | If verdict | Reactivation path |
|---|---|---|
| Wave 1 cycle 0 ($5-7 Vast.ai 4090) | Standalone IMP cycle 0 fails to recover Lane G v3 1.05 anchor | DEFER per Catalog #298; reactivation = enumerate >=3 magnitude criteria (L1 / Hessian-trace / score-gradient saliency) per Catalog #308; re-probe each variant |
| Wave 1b magnitude-criterion sweep ($1.50 × 3 variants) | NO variant recovers Lane G v3 anchor | DEFER per Catalog #298; reactivation = sparsity level reduction (89.3% → 80% → 70%) OR PoseNet-only pruning preservation |
| Wave 2 IMP+Quantizr composition ($3-5 paired) | Composition fails to beat PR101 frontier 0.19205 | DEFER per Catalog #298; reactivation = sister Quantizr archive grammar redesign OR IMP+FP4+brotli triple composition |
| All paths exhausted | All paths fail | LTH paradigm on multi-objective video-renderer substrate DEFER per Catalog #298 → reactivation = NEW pruning paradigm (e.g. SNIP / GraSP / SynFlow / Lottery Tickets at Scale) OR sister sparse-subnetwork substrate (e.g. DARTS-S, weight-sharing variant) |

## Step 6 — Catalog #324 post-training Tier-C validation discipline

Recipe declares `predicted_band_validation_status: pending_post_training`. Reactivation criterion: post-training Tier-C density measurement on IMP cycle 0 archive (post-Lane-G-v3-recovery) AND IMP+Quantizr composition archive (post-composition) via `tools/mdl_scorer_conditional_ablation.py --tier c`. Predicted band [0.180, 0.192] is research prior; promotion-eligible only after `validated_post_training` status.

## Operator-routable decisions

**Decision 1**: dispatch priority
- (A) Standalone IMP cycle 0 first ($5-7); then IMP+Quantizr composition ($3-5)
- (B) Both PATHs parallel ($10-12 total)
- (C) IMP+Quantizr composition ONLY ($3-5; assumes standalone recovery is necessary preliminary)

**Decision 2**: magnitude-criterion priority
- (A) Canonical Frankle 2019 L1 per-tensor only (cheapest; ~$5)
- (B) L1 + Hessian-trace per-tensor OBD (~$7)
- (C) Full 3-variant sweep (L1 + Hessian-trace + score-gradient saliency) (~$10)

**Decision 3**: convocation mechanism (full T3 / inner-quintet / operator-override per Catalog #300 Consequence 1)

## Cross-substrate dependencies

- **Sister Quantizr 0.33 lineage**: IMP+FP4 composition IS frontier-EV path per Op-routable #4 from 2026-05-17 symposium
- **Sister Catalog #91/#92/#93/#94/#117 strict gates**: stub-loop bug class structurally extinct
- **Sister Catalog #307 paradigm-vs-implementation classification**: 2026-04-30 KILL was IMPLEMENTATION-LEVEL FALSIFICATION, NOT paradigm
- **Sister Catalog #308 alternative-probe-methodologies**: >=3 magnitude criteria mandatory
- **Sister Catalog #313 probe-outcomes ledger**: PROCEED/DEFER outcome MUST be registered
- **Sister Catalog #324 post-training Tier-C validation**: pending_post_training status; reactivation criterion pinned

## Predicted cost per Wave

- Wave 1 cycle 0 (Vast.ai 4090): $5-7
- Wave 1b magnitude-criterion sweep (3 variants): $1.50-2
- Wave 2 IMP+Quantizr composition: $3-5
- Paired CPU+CUDA auth-eval + runtime closure: $0.70-1
- TOTAL `lane_17_imp` RE-EVAL-HIGH envelope: $10.20-15

## Continual-learning posterior anchor

Per Catalog #300 + `tac.council_continual_learning.append_council_anchor`: this DRAFT must emit v2 posterior anchor at convocation. `deferred_substrate_id` = `lane_17_imp`; `predicted_mission_contribution` = `frontier_protecting`; retrospective due 2026-06-18T06:05:57Z.


<!-- # FORMALIZATION_PENDING:pre_framework_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_strict_flip_enablers_per_operator_blanket_approval_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
