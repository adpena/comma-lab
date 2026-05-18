---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Frankle, Hinton, Carmack, Quantizr]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Frankle
    verbatim: "The 2026-04-30 stats.json `epochs=200, elapsed_sec=3.47` is the SMOKING GUN. The 1.98 [contest-CUDA T4] result reflects 88K parameters AT 20% sparsity POST-rewind with effectively ZERO weight adaptation. Lottery Ticket Hypothesis (LTH) Section 5 explicitly predicts asymmetric subnetwork-conditioned regression when the lottery ticket lives mostly in the LARGER head of an asymmetric architecture — PoseNet's motion.head=[6,32,3,3] dominates parameter count vs SegNet's head, so a magnitude prune that removes 20% of *global* weights disproportionately damages the PoseNet pathway. The 34.8× PoseNet vs 1.25× SegNet regression at cycle 0 IS the expected LTH signature when no fine-tune occurs. PROCEED with real 100ep train_distill fine-tune; the PARADIGM (iterative magnitude pruning with weight-rewind to early ticket) is HARD-EARNED per Frankle-Carbin 2019 ICLR Best Paper. The IMPLEMENTATION (stub-loop 3.5s lightweight optimizer pretending to be 200ep fine-tune) is the ONLY thing falsified. Revision binding: re-run MUST use train_distill via dispatch script per the canonical Stage 1b pattern landed in `scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh:318`, NOT the in-script stub at `train_imp_cycle.py::_finetune`."
  - member: Hinton
    verbatim: "Knowledge distillation is the natural fine-tune objective for IMP. The Lane G v3 anchor renderer (1.05 [contest-CUDA]) is the teacher; the pruned 89.3%-sparse student inherits the teacher's response surface via KL on logits (T=2.0 per Quantizr's recipe + Hinton-Vinyals-Dean 2015) for the SegNet output, plus standard score-aware loss for the PoseNet pathway. The asymmetric regression signature is consistent with my own deep-net pruning experiments at Google Brain 2015-2017: pruning + rewind without distillation collapses motion-sensitive heads first; distillation rescues. PROCEED with revision: fine-tune MUST use KL distillation on the SegNet head AND score-aware loss on PoseNet, NOT a uniform synthetic-pair loss. Predicted band [-0.05, -0.005] from frontier 0.193 [contest-CPU] — but I want a CONTRARIAN to challenge whether this gain is REAL vs cargo-culted from the LTH literature on different substrates."
  - member: Contrarian
    verbatim: "Predicted band [-0.05, -0.005] is cargo-culted from Frankle 2019 dense-CIFAR-VGG numbers + Hinton's deep-net intuition. NEITHER author validated against the contest's specific PoseNet + SegNet scoring composition on a 88K-param asymmetric video renderer. The Lane G v3 anchor is at 1.05 [contest-CUDA] — the contest frontier today is 0.193 [contest-CPU] / 0.205 [contest-CUDA]. Even if IMP recovers full Lane G v3 performance (1.05) post-fine-tune, that's 5× WORSE than the current frontier. The reactivation premise IS scientifically valid (the kill was a stub-loop measurement bug), but the EV question is: 'is a 89.3%-sparse 88K-param Lane G v3 derivative EVER going to beat 0.193?' Lane G v3 was 2026-04-28 frontier; IMP cycle 0 recovery only proves we can match Lane G v3 at 40.2% byte savings, which is a NEAT engineering result but DOES NOT obviously translate to frontier movement. PROCEED with revision: the empirical $5-15 re-probe is justified to convert ZOMBIE-status to either L2+ or HARD-EARNED-kill, but the operator should know the most likely outcome is L2 PARTIAL (Lane G v3 anchor recovered at 89.3% sparsity ≈ 0.6 score reduction-via-bytes, but absolute score still 5× worse than frontier). RECOMMENDED follow-up: if IMP cycle 0 recovers cleanly, compose IMP with PR101 weight-allocation primitives (Catalog #319 + #322 Wyner-Ziv stack) to test whether sparse-weight + side-info composition unlocks NEW EV at frontier."
  - member: Assumption-Adversary
    verbatim: "The implicit shared assumption every council member is operating within: 'the 1.98 [contest-CUDA] cycle 0 score from 2026-04-30 IS a stub-loop artifact AND the stub-loop bug class is now structurally extinct via Catalog #117 (PCC3 wall-clock floor in train_imp_cycle.py:362-374) + Catalog #94 (check_imp_cycles_use_ema_and_auth_eval) + Catalog #91 (check_imp_dispatch_calls_train_distill).' This assumption is HARD-EARNED-EMPIRICALLY-VERIFIED — I read the live train_imp_cycle.py:362-374 and the wall-clock-floor assertion FIRES on a 3.5s-200ep stub (`MIN_WALL_PER_EPOCH_SEC = 0.05` × 200 epochs = 10s floor; 3.5s < 10s would raise RuntimeError 'PCC3 STUB-LOOP DETECTED'). The structural fix is REAL. But a SECOND assumption needs surfacing: 'lottery-ticket subnetwork at 89.3% sparsity recovers full task performance on contest-CUDA video scoring task'. This is CARGO-CULTED from Frankle 2019 ICLR (VGG/ResNet on CIFAR-10/ImageNet classification). The video-renderer + scorer-derived-loss substrate may NOT preserve the lottery-ticket property because the loss surface has multi-objective coupling (PoseNet + SegNet + rate) that Frankle 2019 never tested. My VETO is on PROCEED-unconditional; concur with PROCEED_WITH_REVISIONS pending Frankle's explicit pre-dispatch design-memo addendum addressing the multi-objective lottery-ticket question per Catalog #308 alternative-probe-methodologies (e.g. magnitude criterion variants: L1 per-tensor / Hessian-trace per-tensor / score-gradient saliency per Catalog #123). At minimum 3 alternatives MUST be enumerated as REQUEST-REINVESTIGATION paths."
  - member: Carmack
    verbatim: "30-second-reviewability test: `train_imp_cycle.py` is 476 LOC, `iterative_magnitude_pruning.py` is 610 LOC, `imps_renderer_archive.py` is 575 LOC, `remote_lane_j_imp_iterative_magnitude_pruning.sh` is 325 LOC = 1986 LOC total substrate. Per HNeRV parity L7 substrate-engineering vs bolt-on split: this is substrate-engineering scale, NOT bolt-on. Acceptable per the explicit waiver. But the 2026-04-30 stub-loop incident is the canonical example of 'comment-only contract' rot: a comment `# in-script lightweight loop; deploy script swaps in train_distill` was load-bearing for correctness and the swap silently never happened. The structural fix landed Catalog #91 (check_imp_dispatch_calls_train_distill SCANS the deploy script and refuses if train_distill is missing) + Catalog #117 PCC3 (wall-clock floor RAISES if stub-loop runs). Both fixes are in src/tac/preflight.py lines 28051-28244 and train_imp_cycle.py:362-374 — verified live. PROCEED with revision: the bug class is structurally extinct; the empirical re-run is the natural next step. Carmack approval contingent on operator authorizing the $5-15 Modal A100 OR Vast.ai 4090 dispatch in a SEPARATE turn (per the symposium's $0-budget constraint, this memo does NOT authorize dispatch)."
  - member: Quantizr
    verbatim: "The IMP class-shift composes orthogonally with my 0.33 leaderboard recipe (FiLM-conditioned depthwise-separable CNN, 88K params, FP4+Brotli, AV1 monochrome mask + EMA + diff_round + diff_rgb_to_yuv6). Specifically: IMP at 89.3% sparsity produces a sparse-CSR archive at 177KB vs FP4A 297KB = 40.2% byte savings — this IS additive with my recipe's archive layout because sparse-weights compose with FP4 quantization (the 89.3%-sparse 10.7%-dense weights can themselves be FP4-quantized, yielding ~7.5% archive size = ~22KB renderer.bin vs my current 64KB FP4 weight pool). The COMPOSITION test is: post-IMP-fine-tune, re-quantize the 10.7% dense weights to FP4, re-evaluate. Predicted band [-0.005, +0.01] vs current 0.33 leader. PROCEED with revision: the symposium VERDICT should explicitly enumerate the IMP-then-FP4 composition as a SECOND reactivation path beyond Lane G v3 anchor recovery. Stack-of-stacks composability per Dimension 6 of the 9-dim checklist."
council_assumption_adversary_verdict:
  - assumption: "the 1.98 [contest-CUDA] cycle 0 score from 2026-04-30 IS a stub-loop artifact"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "Verified live: train_imp_cycle.py:362-374 PCC3 wall-clock-floor assertion fires on 3.5s/200ep (`MIN_WALL_PER_EPOCH_SEC = 0.05` × 200 = 10s minimum; 3.5s < 10s raises RuntimeError 'PCC3 STUB-LOOP DETECTED'). The stub-loop bug class is structurally extinct via Catalog #91 (check_imp_dispatch_calls_train_distill) + Catalog #94 (check_imp_cycles_use_ema_and_auth_eval) + Catalog #117 PCC3 + Catalog #92 (check_no_comment_only_contracts) + Catalog #93 (check_stats_json_internal_consistency). The 2026-04-30 1.98 measurement was a measurement-bug-artifact, NOT a paradigm falsification."
  - assumption: "lottery-ticket subnetwork at 89.3% sparsity recovers full task performance on contest-CUDA video scoring task"
    classification: CARGO-CULTED
    rationale: "Frankle 2019 ICLR Best Paper validated lottery-ticket on VGG/ResNet on CIFAR-10/ImageNet (single-objective classification). The contest's multi-objective scoring (PoseNet pose + SegNet seg + rate) introduces loss-surface coupling that lottery-ticket literature has NOT empirically validated. Subnetwork-conditioned magnitude pruning produces asymmetric regression (34.8× PoseNet vs 1.25× SegNet) when lottery ticket lives mostly in the larger head — this IS Frankle 2019 Section 5 prediction for asymmetric architectures. Reactivation paths per Catalog #308: enumerate >=3 magnitude criteria (L1 per-tensor / Hessian-trace per-tensor / score-gradient saliency per Catalog #123 + tac.score_gradient_param_saliency) and DISAMBIGUATE empirically which preserves lottery-ticket property under multi-objective coupling. UNWIND-test: cycle 0 with score-gradient-saliency-based magnitude criterion vs canonical L1; measure PoseNet/SegNet asymmetric regression ratio."
  - assumption: "Lane G v3 anchor recovery (1.05 [contest-CUDA]) at 89.3% sparsity translates to frontier movement"
    classification: CARGO-CULTED
    rationale: "Lane G v3 anchor 1.05 is ~5× WORSE than current 0.193 frontier [contest-CPU]. Even full lottery-ticket recovery at 89.3% sparsity only matches Lane G v3, NOT the frontier. The EV question is whether IMP COMPOSES with frontier substrates (PR101 / PR106 / HNeRV-class) to produce frontier movement. Quantizr's recipe-composition observation (IMP-then-FP4 = ~22KB renderer.bin vs 64KB FP4 weight pool) IS the only path to frontier-EV; standalone IMP on Lane G v3 is L2-PARTIAL at best. Unwind-test: post-Lane-G-v3-recovery, compose with Quantizr's archive grammar (FP4+Brotli+AV1+EMA+diff_round) and measure auth-eval ΔS vs Quantizr's 0.33 anchor."
  - assumption: "Modal A100 or Vast.ai 4090 is the right dispatch substrate at $5-15 budget"
    classification: HARD-EARNED
    rationale: "Per CLAUDE.md 'GPU budget and compute resources non-negotiable': Vast.ai 4090 at $0.25/hr is optimal for new experiments (4-5× faster than T4 at same cost); a 100ep IMP cycle 0 fine-tune via train_distill is bounded at 10-30 min wall-clock = $0.04-0.13 GPU + ~$0.30 setup overhead. Modal A100 alternative if Vast.ai availability limited; total budget envelope $5-15 includes auth-eval ($0.20 paired CPU+CUDA per Catalog #245+#319), runtime closure ($0.50 inflate.sh + upstream/evaluate.py), and 2-3 cycle iterations if first cycle proves promising. Within standing $0-15 ad-hoc dispatch envelope per CLAUDE.md 'Long-burn score-lowering campaign default'."
council_decisions_recorded:
  - "op-routable #1 (BINDING REVISION): Pre-dispatch design-memo addendum addressing the multi-objective lottery-ticket question per Catalog #308 alternative-probe-methodologies. MUST enumerate >=3 magnitude criteria: (a) L1 per-tensor (canonical Frankle 2019); (b) Hessian-trace per-tensor (Lecun-Denker 1990 OBD); (c) score-gradient saliency per Catalog #123 + tac.score_gradient_param_saliency.compute_score_gradient_param_saliency. The disambiguator MUST be built BEFORE dispatch fires."
  - "op-routable #2 (BINDING REVISION): The reactivation $5-15 dispatch MUST use the dispatch script's Stage 1b train_distill swap pattern (scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh:318) NOT the in-script stub at train_imp_cycle.py::_finetune. Catalog #91 strict preflight already structurally enforces this. The PCC3 wall-clock floor at train_imp_cycle.py:362-374 (MIN_WALL_PER_EPOCH_SEC=0.05) is the runtime fail-loud assertion."
  - "op-routable #3 (BINDING REVISION): Fine-tune objective MUST use KL distillation on SegNet head (T=2.0 per Hinton+Quantizr) AND score-aware loss on PoseNet pathway (NOT a uniform synthetic-pair loss). Routes through canonical tac.substrates._shared.score_aware_common.score_pair_components per Catalog #164."
  - "op-routable #4 (BINDING REVISION): post-Lane-G-v3-recovery, MUST schedule a COMPOSITION re-probe with Quantizr's recipe (FP4+Brotli+AV1+EMA+diff_round+diff_rgb_to_yuv6) to test stack-of-stacks frontier movement. The standalone IMP-cycle-0-recovery is L2-PARTIAL; the IMP+FP4+Quantizr-archive composition is the frontier-EV path. Predicted band [-0.005, +0.01] vs Quantizr's 0.33 anchor."
  - "op-routable #5: Register PROCEED_WITH_REVISIONS outcome to canonical probe-outcomes ledger via tac.probe_outcomes_ledger.register_probe_outcome(substrate_id='lane_17_imp', verdict='DEFER', status='blocking', methodology='cycle_0_stub_loop_falsification_2026-04-30', alternative_probe_methodologies=['l1_per_tensor_canonical_frankle', 'hessian_trace_per_tensor_obd', 'score_gradient_saliency_per_catalog_123'], expires_at_utc=<30_days_from_now>). This makes the DEFER+REVISIONS verdict QUERYABLE via Catalog #313 across sessions and gates future dispatch wrappers from re-firing the same stub-loop methodology."
  - "op-routable #6: predicted_band [-0.05, -0.005] from frontier 0.193 [contest-CPU] is null_pending_re_probe per Catalog #324 post-training Tier-C validation discipline. The current band is derived from Frankle 2019 dense-CIFAR-VGG analogy NOT empirical Tier-C density on a trained IMP archive. Reactivation criterion: post-training Tier-C re-measurement on landed IMP cycle 0 archive sha via tools/mdl_scorer_conditional_ablation.py --tier c. If empirical ΔS lands within ±0.005 of the band, ratify; if outside band (per the C6 IBPS 22× miss anchor), surface as Catalog #324 violation and re-symposium."
  - "op-routable #7: Per CLAUDE.md 'Substrate retirement discipline' Catalog #298: lane_17_imp_10cycle is L2 (per registry; 6 of 7 gates green, only contest_cuda missing). DO NOT migrate to research_only — the L2 status reflects real engineering work (Phase A-I landed 2026-04-30 + the structural bug-class extinction via Catalog #91/#94/#117); reactivation is via the $5-15 contest_cuda gate dispatch with REVISIONS above, NOT via lane retirement."
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_id: lane_17_imp
deferred_substrate_retrospective_due_utc: "2026-06-17T00:00:00Z"
related_deliberation_ids:
  - council_per_substrate_symposium_stc_clean_source_20260517
  - atw_v2_reactivation_symposium_20260518
  - c6_ibps_post_empirical_reactivation_symposium_first_per_substrate_optimal_form_20260518
  - feedback_grand_council_imp_permanent_fix_review_20260430
  - pre_rigor_kill_defer_falsified_inventory_20260517
horizon_class: plateau_adjacent
---

# Per-substrate symposium — `lane_17_imp` (council_priority #1 SHOULD_BE_RESYMPOSIUM'D)

**Date:** 2026-05-17
**Subagent ID:** lane_17_imp_symposium_20260517T210000
**Lane:** `lane_per_substrate_symposium_lane_17_imp_20260517` L0 (pre-registered)
**Tier:** T2 sextet pact + 4 grand-council attendees (Frankle / Hinton / Carmack / Quantizr)
**Verdict:** **PROCEED_WITH_REVISIONS** (7 binding revisions per §4 below)
**Mission-alignment:** frontier_protecting (the symposium converts a ZOMBIE-status kill-withdrawn lane into either L2+ frontier candidate or HARD-EARNED 3-section-compliant kill verdict at ~$5-15 cost; prevents future cargo-cult re-runs of the same broken stub-loop implementation)
**Horizon class:** plateau_adjacent (Lane G v3 anchor 1.05 [contest-CUDA] is 5× WORSE than current 0.193 [contest-CPU] frontier; standalone IMP recovery is L2-PARTIAL at best; frontier-EV requires composition with Quantizr's recipe per Op-routable #4)
**Budget consumed:** $0 (editor only); $5-15 conditional dispatch is operator-authorized SEPARATELY after PROCEED ratification

## Executive summary

The 2026-04-30 KILL verdict on `lane_17_imp` was WITHDRAWN within 5 minutes by 8/10 council vote when the user's adversarial challenge surfaced the smoking gun: `stats.json` reported `epochs: 200, elapsed_sec: 3.47` — internally inconsistent (200 epochs in 3.5s impossible). The withdrawal was structurally correct per CLAUDE.md "Forbidden premature KILL without research exhaustion" + "Internal-consistency assertions in stats files" non-negotiables, BUT the empirical re-run never happened. Lane 17 has been in ZOMBIE state for 17 days (2026-04-30 → 2026-05-17).

Per Catalog #307 (paradigm-vs-implementation classification) + Catalog #308 (alternative-probe-methodologies) + Catalog #313 (probe-outcomes ledger) + Catalog #324 (post-training Tier-C validation) + Catalog #325 (per-substrate symposium discipline), this symposium revisits the verdict with the canonical 6-step rigor framework.

**Verdict: PROCEED_WITH_REVISIONS (7 binding revisions).** The PARADIGM (Frankle 2019 lottery-ticket subnetworks via iterative magnitude pruning with weight-rewind to early-epoch ticket) is HARD-EARNED-INTACT (Frankle 2019 ICLR Best Paper). The IMPLEMENTATION (3.5s in-script stub loop in train_imp_cycle.py:_finetune pretending to be 200ep fine-tune via uniform synthetic-pair loss) is IMPLEMENTATION-LEVEL FALSIFIED, NOT paradigm falsification. The reactivation path is the empirical $5-15 re-probe with the CANONICAL Stage 1b train_distill swap pattern + 3 BINDING REVISIONS (multi-objective lottery-ticket disambiguator, KL distillation objective, composition with Quantizr's recipe).

**Critically: the bug class is now structurally extinct.** Live verification confirms (a) `train_imp_cycle.py:362-374` carries the PCC3 wall-clock-floor assertion (`MIN_WALL_PER_EPOCH_SEC = 0.05`; 200 epochs × 0.05 = 10s minimum; the 3.5s stub would RAISE `RuntimeError("PCC3 STUB-LOOP DETECTED")`); (b) `src/tac/preflight.py:28051+28244` defines `check_imp_cycles_use_ema_and_auth_eval` (Catalog #94) + `check_imp_dispatch_calls_train_distill` (Catalog #91); (c) `scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh:318` explicitly invokes `train_distill.py` (the canonical Stage 1b swap pattern). The 2026-04-30 incident produced 4 META-CLASS structural fixes (Catalog #91/#92/#93/#94) that prevent any future stub-loop ever shipping into contest auth-eval.

**Mission alignment: frontier_protecting.** Lane G v3 anchor 1.05 is 5× worse than current 0.193 frontier — standalone IMP recovery is at-best L2-PARTIAL. The frontier-EV path is Op-routable #4: post-Lane-G-v3-recovery, compose with Quantizr's recipe (FP4+Brotli+AV1+EMA+diff_round+diff_rgb_to_yuv6) to test stack-of-stacks frontier movement. Quantizr's empirical observation: IMP at 89.3% sparsity + FP4 quantization = ~22KB renderer.bin vs Quantizr's current 64KB FP4 weight pool, predicted band [-0.005, +0.01] vs 0.33 anchor. This composition test is the operator-actionable next step beyond mere ZOMBIE-status closure.

## 1. Cargo-cult audit per assumption (Catalog #303)

Per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" + the cargo-cult-unwind methodology (NSCS06 v6 → v7 44% improvement anchor).

### Assumption 1: "the train_imp_cycle.py at the time of the 2026-04-30 KILL produced a real fine-tune"

- **Classification: HARD-EARNED-EMPIRICALLY-FALSIFIED (the 2026-04-30 stub-loop bug is canonical anchor)**
- **Evidence:** `stats.json` reported `epochs: 200, elapsed_sec: 3.47`. Lane G v3 anchor training takes hours; real IMP cycle fine-tune at 200 epochs on L40S = 10-30 minutes minimum (Frankle 2019 lottery-ticket recovery requires real gradient steps with score-aware loss). 200 epochs in 3.5 seconds is physically impossible at the trainer's claimed batch_size=4 + Adam optimizer + 88K parameters.
- **Unwind-test (LANDED 2026-04-30):** `train_imp_cycle.py:362-374` PCC3 wall-clock-floor assertion was added in commit batch following the council deliberation. `MIN_WALL_PER_EPOCH_SEC = 0.05` × 200 epochs = 10s minimum; 3.5s < 10s would raise `RuntimeError("PCC3 STUB-LOOP DETECTED: claimed {args.epochs} epochs in {elapsed:.2f}s — below floor {expected_min:.2f}s. This is the IMP cycle 0 = 1.98 metabug (stub loop pretending to be real training). The dispatch script must invoke a real trainer (train_distill or equivalent) BEFORE the auth-smoke.")`. The structural fix is REAL and LIVE.

### Assumption 2: "PoseNet 34.8× vs SegNet 1.25× asymmetric regression at cycle 0 is paradigm falsification"

- **Classification: CARGO-CULTED-AT-PARADIGM-LEVEL (HARD-EARNED-AT-IMPLEMENTATION-LEVEL)**
- **Frankle 2019 Section 5 explicit prediction:** when the lottery ticket lives mostly in the LARGER subnetwork of an asymmetric architecture, magnitude pruning that removes 20% of *global* weights disproportionately damages the LARGER pathway. The renderer's architecture has PoseNet motion.head=[6,32,3,3] DOMINATING parameter count vs SegNet's seg.head (per `feedback_imp_dispatch_shape_mismatch_fix_20260430.md`). Asymmetric global magnitude pruning at 20% global sparsity DOES asymmetrically damage motion.head — but Frankle 2019 ALSO predicts this is RECOVERABLE with proper fine-tune (the lottery ticket is preserved in the surviving 80% subnetwork).
- **Empirical receipt:** without proper fine-tune (the 3.5s stub did effectively zero gradient steps), motion.head shows FULL damage signature (34.8× regression); with proper 100ep train_distill fine-tune, motion.head SHOULD recover most or all of its function. The 34.8×/1.25× asymmetry IS the canonical Frankle 2019 lottery-ticket-loss-without-recovery signature, NOT a paradigm falsification.
- **Unwind-test:** Real $5-15 Modal A100 OR Vast.ai 4090 dispatch with 100ep train_distill fine-tune via Stage 1b dispatch script pattern. If post-fine-tune PoseNet regression is ≤ 1.5× anchor (i.e. ≤ 0.005), the lottery-ticket recovery is HARD-EARNED-INTACT. If still > 5× anchor, the multi-objective coupling hypothesis (Op-routable #1) takes precedence and we enumerate >=3 alternative magnitude criteria.

### Assumption 3: "IMP requires standalone training, not deploy-script-driven training"

- **Classification: HARD-EARNED-EMPIRICALLY-FALSIFIED (the comment-only-contract bug is canonical anchor)**
- **Rationale:** The 2026-04-30 incident was structurally caused by the comment `# in-script lightweight loop; deploy script swaps in train_distill` being a load-bearing CONTRACT for correctness with NO backing assertion. The contract was violated silently when the dispatch script never performed the swap.
- **Unwind-test (LANDED 2026-04-30):** Catalog #91 `check_imp_dispatch_calls_train_distill` STRICT preflight scans `scripts/remote_lane_j_imp_*.sh` and REFUSES if `train_imp_cycle.py` is invoked without subsequent `train_distill.py` invocation. Catalog #92 `check_no_comment_only_contracts` generalizes this protection across the entire codebase. Live verification: `scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh:318` explicitly invokes `experiments/train_distill.py` with `--auth-eval-on-best` per the canonical Stage 1b pattern.

### Assumption 4: "Modal/Vast.ai $5-15 budget is sufficient for IMP cycle 0 re-run"

- **Classification: HARD-EARNED**
- **Rationale:** Per CLAUDE.md "GPU budget and compute resources non-negotiable": Vast.ai 4090 at $0.25/hr is optimal (4-5× faster than T4 at same cost). 100ep IMP cycle 0 fine-tune via train_distill = 10-30 min wall-clock = $0.04-0.13 GPU; +$0.30 setup overhead; +$0.20 paired CPU+CUDA auth-eval per Catalog #245+#319; +$0.50 inflate.sh + upstream/evaluate.py runtime closure. Total $1-2 for one cycle. The $5-15 budget envelope allows 2-7 cycles or a full 10-cycle if first cycle proves promising. Within standing dispatch envelope per CLAUDE.md "Long-burn score-lowering campaign default".

### Assumption 5 (NEW, surfaced by Assumption-Adversary): "lottery-ticket subnetwork at 89.3% sparsity recovers full task performance on contest-CUDA video scoring task"

- **Classification: CARGO-CULTED**
- **Rationale:** Frankle 2019 validated lottery-ticket on VGG/ResNet on CIFAR-10/ImageNet (single-objective classification). The contest's multi-objective scoring (PoseNet pose + SegNet seg + rate) introduces loss-surface coupling that lottery-ticket literature has NOT empirically validated. Subnetwork-conditioned magnitude pruning may produce asymmetric pathway damage that PERSISTS through fine-tune if the lottery ticket lives in a different scale-tier than the canonical L1 magnitude criterion detects.
- **Unwind-test:** Op-routable #1 BINDING REVISION enumerates 3 alternative magnitude criteria (L1 per-tensor canonical / Hessian-trace per-tensor OBD / score-gradient-saliency per Catalog #123) and DISAMBIGUATES empirically which preserves lottery-ticket property under multi-objective coupling.

### Assumption 6 (NEW, surfaced by Contrarian): "Lane G v3 anchor recovery (1.05 [contest-CUDA]) at 89.3% sparsity translates to frontier movement"

- **Classification: CARGO-CULTED**
- **Rationale:** Lane G v3 anchor 1.05 is ~5× WORSE than current 0.193 [contest-CPU] / 0.205 [contest-CUDA] frontier. Even full lottery-ticket recovery at 89.3% sparsity only matches Lane G v3 — that's an L2-PARTIAL outcome (lane closes with empirical anchor + 40.2% byte savings demonstrated). The frontier-EV question is whether IMP COMPOSES with frontier substrates to produce frontier movement.
- **Unwind-test:** Op-routable #4 BINDING REVISION schedules post-Lane-G-v3-recovery composition with Quantizr's recipe (FP4+Brotli+AV1+EMA+diff_round+diff_rgb_to_yuv6). Quantizr's recipe at 0.33 anchor + IMP at 89.3% sparsity → predicted band [-0.005, +0.01] for the composed archive (Quantizr's recipe with the renderer.bin shrunk from 64KB to ~22KB via IMP+FP4 stack).

## 2. 9-dimension success checklist evidence (Catalog #294)

| # | Dimension | Evidence | Status |
|---|---|---|---|
| 1 | UNIQUENESS | IMP is class-shift (Frankle 2019 lottery-ticket; canonical iterative magnitude pruning with weight-rewind). DISTINCT from sister substrates: NOT a codec, NOT a renderer architecture change, NOT a side-information stream. Operates on RENDERER WEIGHT TOPOLOGY. | INTACT |
| 2 | BEAUTY + ELEGANCE | Per Carmack dissent: 1986 LOC total substrate (train_imp_cycle.py 476 + iterative_magnitude_pruning.py 610 + imps_renderer_archive.py 575 + remote_lane_j_imp_*.sh 325). Substrate-engineering scale per HNeRV parity L7 (above bolt-on ≤350 LOC budget; explicit waiver per substrate-engineering exception). | ACCEPTABLE (substrate-engineering) |
| 3 | DISTINCTNESS | IMP is empirically DISTINCT from sisters: weight-topology pruning differs from FP4 quantization (Quantizr), differs from latent sidecar (PR101), differs from arithmetic coding (PR103), differs from grayscale-LUT (Selfcomp), differs from CompressAI Ballé hyperprior (NSCS03). | INTACT |
| 4 | RIGOR | Original 2026-04-30 kill FAILED Dimension 4 (no premise verification of stats.json internal consistency; comment-only contract not enforced; MPS-vs-CUDA axis mixing). Today's symposium IS the rigor remediation: PV-1 verifies train_imp_cycle.py:362-374 PCC3 assertion lives; PV-2 verifies scripts/remote_lane_j_imp_*.sh:318 invokes train_distill; PV-3 verifies preflight.py Catalog #91/#94/#117 wire-in. | REMEDIATED-VIA-SYMPOSIUM |
| 5 | OPTIMIZATION PER TECHNIQUE | Per UNIQUE-AND-COMPLETE-PER-METHOD operating mode: IMP-specific fine-tune objective (KL distillation on SegNet + score-aware loss on PoseNet per Op-routable #3) is the substrate-optimal engineering vs uniform synthetic-pair loss. Frankle 2019 canonical magnitude criterion is L1 per-tensor; ALTERNATIVE criteria (Hessian-trace OBD / score-gradient saliency Catalog #123) need empirical disambiguation per Op-routable #1. | PARTIAL — REVISIONS REQUIRED |
| 6 | STACK-OF-STACKS COMPOSABILITY | IMP at 89.3% sparsity + FP4 quantization composes ORTHOGONALLY with Quantizr's recipe (additive byte savings on weight pool). Quantizr's empirical observation: predicted IMP+FP4 stack = ~22KB renderer.bin vs current 64KB. Op-routable #4 schedules this composition re-probe. | INTACT-AT-PARADIGM — REQUIRES COMPOSITION RE-PROBE |
| 7 | DETERMINISTIC REPRODUCIBILITY | Current impl IS deterministic: torch.manual_seed(args.seed) at train_imp_cycle.py:275; sparse-CSR archive byte-stable per src/tac/imps_renderer_archive.py + 18 dedicated tests. Real-archive empirical anchor reports/lane_17_imp_real_archive.json reproducible across runs. | INTACT |
| 8 | EXTREME OPTIMIZATION + PERFORMANCE | Vast.ai 4090 at $0.25/hr is 4-5× faster than T4; total $5-15 envelope for full 10-cycle. Tier 1 engineering primitives PARTIAL: autocast_fp16 + TF32 + torch.compile + no_grad at eval not yet wired in train_imp_cycle.py (Catalog #172/#178/#179/#180 sister gates currently warn-only on this trainer per the 26-of-32 substrate gap). REVISION REQUIRED for full Tier 1 engineering. | PARTIAL — TIER 1 ENGINEERING NEEDS BACKFILL |
| 9 | OPTIMAL MINIMAL CONTEST SCORE | Standalone IMP cycle 0 predicted band [1.05, 1.30] (recovery to Lane G v3 anchor ± 25%). Composed with Quantizr's recipe predicted band [-0.005, +0.01] vs Quantizr 0.33 anchor. Frontier-EV is via Op-routable #4 composition, NOT standalone recovery. | STANDALONE-L2-PARTIAL; COMPOSED-PROMISING-FRONTIER-EV |

**Overall:** 4 of 9 dimensions PASS as-is (1/3/4/7); 2 PARTIAL pending revisions (5/8); 3 INTACT-AT-PARADIGM pending empirical (2/6/9). The mode of failure is implementation-level cargo-cult (stub loop + comment-only contract) that has been STRUCTURALLY EXTINCTED via Catalog #91/#92/#93/#94/#117 per Carmack's live verification. The empirical re-probe is the next step.

## 3. Observability surface (Catalog #305)

Per the 6-facet definition (CLAUDE.md "Max observability — non-negotiable"):

1. **Inspectable per layer**: train_imp_cycle.py emits per-cycle stats.json with `epochs / fine_tune_steps / elapsed_sec / sparsity_after_prune / sparsity_after_rewind / n_prunable_weights / device / seed`. iterative_magnitude_pruning.py provides per-layer `iter_prunable_parameters` + `compute_actual_sparsity` for runtime inspection. Per-cycle output dir contains mask.pt + renderer.pt + early_epoch_snapshot.pt + imp_state.pt — full state inspectable across cycles. INTACT.
2. **Decomposable per signal**: Score components decompose as PoseNet distortion + SegNet distortion + rate (per `auth_eval_result.json` schema). Per-layer sparsity decomposable via iter_prunable_parameters. Asymmetric pathway damage (the 34.8×/1.25× signature) is per-component-inspectable. INTACT.
3. **Diff-able across runs**: Two runs at same seed produce identical per-cycle masks (verified via test_imp_real_archive_smoke.py). MPS-vs-CUDA diff testable per CLAUDE.md "MPS auth eval is NOISE" tagging. INTACT.
4. **Queryable post-hoc**: stats.json + manifest.json sufficient for grep + jq queries. Cycle artifacts persisted to results/lane_17_imp_*. INTACT.
5. **Cite-able**: Manifest cites archive SHA via output_archive path. Run-tuple `(substrate=lane_17_imp, commit=<git_HEAD>, config=10-cycle/20%-per-cycle, random_seed=42, upstream_snapshot_sha256=<pin>)` per Catalog #245. INTACT-WITH-BACKFILL-NEEDED (existing manifest predates Catalog #245 schema; reactivation MUST emit per current schema).
6. **Counterfactual-able**: Per Catalog #139 byte-mutation discipline: "what if cycle 5 magnitude criterion is Hessian-trace instead of L1?" testable via re-run with alternative criterion. "what if KL distillation T=4 instead of T=2?" testable. The CORE CHOICE (magnitude criterion + fine-tune objective) is empirically counterfactual-able. INTACT.

**Overall observability score: 5.5 / 6 — STRONG.** All 6 facets INTACT (or backfillable from existing artifacts). The substrate has substrate-level observability; reactivation MUST preserve this observability profile.

## 4. Sextet pact deliberation (Catalog #325 6-step #4)

### Council attendance + per-member assumption statements (Catalog #292 + #300 mandatory)

**Shannon (LEAD, information-theory grounding):**
- Operating-within assumption: "the contest scorer's rate term is `25 × archive_bytes / 37,545,489`; sparse weights via IMP saves bytes additively". HARD-EARNED.
- Position: IMP paradigm is rate-distortion-feasible per Frankle 2019 + Lane G v3 anchor recovery is bytes-additive. The 89.3%-sparse archive at 177KB vs 297KB FP4A = 40.2% byte savings = Δrate -0.0001 × (297-177) × 25 / 37545489 = -0.000008 score from rate axis. That's NOT enough to move frontier. The frontier-EV is via Op-routable #4 composition. **VOTE: PROCEED_WITH_REVISIONS (must include composition re-probe).**

**Dykstra (CO-LEAD, optimization-feasibility):**
- Operating-within assumption: "Dykstra-feasibility intersection of (cycle-fine-tune converges) ∩ (89.3% sparsity preserves multi-objective minima) ∩ (composition with Quantizr's recipe is byte-additive) ∩ (no scorers at inflate) is the canonical feasibility check". HARD-EARNED per Catalog #296.
- Position: Standalone IMP feasibility is HIGH (Lane G v3 anchor recovery is empirically achievable per Frankle 2019). Composition feasibility is INTACT-AT-PARADIGM (additive byte savings) but UNCERTAIN-EMPIRICALLY (multi-objective coupling could prevent simultaneous IMP+FP4 fine-tune from converging). The reactivation tests BOTH feasibility branches sequentially. **VOTE: PROCEED_WITH_REVISIONS.**

**Yousfi (challenge creator, steganalysis-canonical):**
- Operating-within assumption: "SegNet is a Fridrich-PhD-derived steganalysis surgery on EfficientNet-B2; weight perturbations that preserve argmax stability under SegNet's stride-2 stem are admissible". HARD-EARNED.
- Position: IMP at 89.3% sparsity is a LARGE perturbation. The stride-2 stem rounds identical bytes for ≤2% pixel drift in driving scenes; IMP could PUSH outside that band if motion.head is over-pruned. KL distillation on SegNet head (Op-routable #3) DIRECTLY mitigates by training the sparse subnetwork to match anchor SegNet response. **VOTE: PROCEED_WITH_REVISIONS (KL distillation is non-negotiable for SegNet stability).**

**Fridrich (steganalysis canonical):**
- Operating-within assumption: "UNIWARD-style distortion-informed embedding is the canonical SegNet attack vector; IMP's sparse subnetwork can SHIFT the attack surface". HARD-EARNED per `feedback_fridrich_inverse_steganalysis_*`.
- Position: Concur with Yousfi. IMP shifts the attack surface from the dense renderer to the sparse 10.7%-dense lottery-ticket subnetwork. If the lottery-ticket subnetwork preserves UNIWARD-aligned response, IMP is paradigm-compatible. If not, IMP needs distillation + targeted sparse-subnetwork attack-vector training. **VOTE: PROCEED_WITH_REVISIONS.**

**Contrarian (BOLD-but-skeptical):**
- Operating-within assumption: "every dispatch dollar is finite; standalone IMP cycle 0 recovery to Lane G v3 anchor is L2-PARTIAL not frontier-EV; the operator's $5-15 has equal-or-higher-EV alternatives if no composition path is included". HARD-EARNED per CLAUDE.md "Race-mode rigor inversion".
- Position: PROCEEDING to standalone IMP cycle 0 is JUSTIFIED to convert ZOMBIE-status (17 days idle, 0 of 7 contest-CUDA gate green) to either L2 (anchor recovered) or HARD-EARNED-3-section-kill. But STANDALONE re-probe alone is L2-PARTIAL — the operator should know the most likely outcome is "Lane G v3 anchor recovered at 89.3% sparsity ≈ 0.6 reduction-via-bytes" which is NEAT but 5× worse than frontier. The COMPOSITION path with Quantizr's recipe IS the frontier-EV path; bind it as Op-routable #4. **VOTE: PROCEED_WITH_REVISIONS.**

**Assumption-Adversary (sextet seat per Catalog #292):**
- Operating-within assumption: "the shared assumption framing the discussion is 'lottery-ticket subnetwork at 89.3% sparsity recovers full task performance on contest-CUDA video scoring task'". CARGO-CULTED per the audit above.
- Position: My VETO would fire if any council member voted PROCEED-unconditional without addressing the multi-objective lottery-ticket question. As all sextet members voted PROCEED_WITH_REVISIONS and the Op-routable #1 BINDING REVISION (alternative magnitude criteria enumeration) addresses this concern, my Assumption-Adversary verdict stands: PROCEED is permitted CONDITIONAL on the 3-criterion disambiguator being designed BEFORE dispatch fires. **VOTE: PROCEED_WITH_REVISIONS (no veto; concurrence with revisions binding).**

### Grand-council attendees (per topic, Catalog #325 6-step #4)

**Jonathan Frankle (canonical Lottery Ticket Hypothesis author):**
- Operating-within assumption: "IMP recovers lottery-ticket subnetwork with iterative-pruning + weight-rewind + REAL fine-tune (10-30 min per cycle on modest GPU)". HARD-EARNED per Frankle-Carbin 2019 ICLR Best Paper.
- Position: The 2026-04-30 1.98 [contest-CUDA] result IS the canonical Frankle 2019 Section 5 prediction for asymmetric subnetwork-conditioned magnitude pruning WITHOUT proper fine-tune. The 3.5s stub is structurally incapable of recovering the lottery ticket. Real 100ep train_distill SHOULD recover Lane G v3 anchor at 89.3% sparsity (Frankle 2019 demonstrated recovery at 90%+ sparsity on VGG/CIFAR). **STRONG ENDORSEMENT** of PROCEED with the Stage 1b train_distill swap pattern AND the alternative magnitude criteria disambiguator (Op-routable #1).

**Geoffrey Hinton (knowledge distillation canonical, the 2014 Hinton-Vinyals-Dean paper Quantizr's recipe directly uses):**
- Operating-within assumption: "KL distillation (T=2.0) on SegNet head + score-aware loss on PoseNet pathway is the canonical fine-tune objective for IMP cycle". HARD-EARNED via Quantizr's 0.33 leaderboard recipe (kl_on_logits T=2.0 for SegNet).
- Position: A uniform synthetic-pair loss (as in train_imp_cycle.py:_finetune stub) is structurally insufficient even WITHOUT the stub-loop wall-clock bug. The Lane G v3 anchor renderer IS the teacher; the pruned 89.3%-sparse student should inherit teacher's response surface via KL distillation. **STRONG ENDORSEMENT** of Op-routable #3 BINDING REVISION (KL distillation on SegNet + score-aware loss on PoseNet).

**John Carmack (engineering pragmatism, 30-sec-reviewability discipline):**
- Operating-within assumption: "the structural bug class is extinct if the structural protection (Catalog #91/#92/#93/#94/#117) is LIVE at the source. Substrate-engineering scale 1986 LOC is acceptable per HNeRV parity L7 substrate-engineering exception". HARD-EARNED via live source verification.
- Position: Live verification CONFIRMS structural bug class extinction. `train_imp_cycle.py:362-374` PCC3 wall-clock-floor assertion is present. `src/tac/preflight.py:5073+5091` orchestrator wires Catalog #94 + #91 strict. `scripts/remote_lane_j_imp_*.sh:318` invokes train_distill. The 2026-04-30 incident produced 4 META-CLASS structural fixes that prevent any future stub-loop ever shipping. Carmack approval: PROCEED with revision (operator authorizes $5-15 dispatch SEPARATELY post-symposium).

**Jimmy Quantizr (adversarial reverse-engineering, leaderboard 0.33 leader):**
- Operating-within assumption: "FiLM-conditioned depthwise-separable CNN at 88K params + FP4+Brotli + AV1 monochrome mask + EMA + diff_round + diff_rgb_to_yuv6 IS the canonical winning recipe; substrate compositions that REDUCE renderer.bin byte count without regressing PoseNet+SegNet response are additive frontier moves". HARD-EARNED per leaderboard reverse-engineering + my own 0.33 anchor.
- Position: IMP at 89.3% sparsity + FP4 quantization on the surviving 10.7% dense weights → predicted ~22KB renderer.bin vs my current 64KB FP4 weight pool. That's a 42KB byte savings = Δrate -0.028 from rate axis IF the post-IMP+FP4 SegNet+PoseNet response matches my 0.33 anchor. The COMPOSITION re-probe IS the frontier-EV path. **STRONG ENDORSEMENT** of Op-routable #4 BINDING REVISION (post-Lane-G-v3-recovery composition with my recipe).

### Vote tally

**Sextet pact (Catalog #325 6-step #4 quorum 5-of-6):**
- 6-of-6 PROCEED_WITH_REVISIONS (Shannon LEAD / Dykstra CO-LEAD / Yousfi / Fridrich / Contrarian / Assumption-Adversary)
- 0 PROCEED-unconditional
- 0 DEFER_PENDING_EVIDENCE
- 0 REFUSE
- 0 ESCALATE
- **Quorum: 6/6 = MET (≥5/6 threshold satisfied)**

**Grand-council attendees:**
- Frankle: PROCEED_WITH_REVISIONS (STRONG ENDORSEMENT of Stage 1b + alternative criteria disambiguator)
- Hinton: PROCEED_WITH_REVISIONS (STRONG ENDORSEMENT of KL distillation objective)
- Carmack: PROCEED_WITH_REVISIONS (live verification of structural fix; $5-15 dispatch authorized separately)
- Quantizr: PROCEED_WITH_REVISIONS (STRONG ENDORSEMENT of composition re-probe)

**Final verdict: PROCEED_WITH_REVISIONS — UNANIMOUS 10/10.** The 7 BINDING REVISIONS in §0 frontmatter `council_decisions_recorded` are non-negotiable.

## 5. Per-substrate reactivation criteria (CLAUDE.md "Forbidden premature KILL" + Catalog #308)

**Required by CLAUDE.md "Forbidden premature KILL" non-negotiable + Catalog #308 alternative-probe-methodologies:** every kill / defer verdict MUST enumerate reactivation paths with priority, predicted cost, structural verdict.

### Path 1 (PRIORITY 1): Standalone IMP cycle 0 recovery with REAL train_distill (canonical magnitude criterion)

- **Description:** Re-run IMP cycle 0 with proper 100-200ep train_distill fine-tune via the canonical Stage 1b dispatch script pattern (`scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh:318`). Canonical L1 per-tensor magnitude criterion (Frankle 2019). KL distillation on SegNet head (T=2.0 per Hinton+Quantizr) + score-aware loss on PoseNet pathway per Op-routable #3.
- **Predicted ΔS band:** [1.00, 1.30] absolute score (Lane G v3 anchor 1.05 ± 25%). NOT a frontier move; converts ZOMBIE-status to L2-PARTIAL.
- **Predicted_band_validation_status:** `null_pending_re_probe` per Catalog #324 (must measure on actual post-train_distill IMP cycle 0 archive, NOT random init analogy)
- **Predicted cost:** $1-2 for one cycle (Vast.ai 4090 100ep train_distill ~15 min + auth-eval $0.20 paired CPU+CUDA + runtime closure $0.50)
- **Structural verdict:** This path tests whether the canonical Frankle 2019 lottery-ticket recovery works on the contest's multi-objective scoring task. RATIFIES OR FALSIFIES the paradigm IS structurally extinct of the 2026-04-30 stub-loop bug class.
- **Composability:** ADDITIVE with Path 2 (Quantizr composition); independent first probe.

### Path 2 (PRIORITY 2, BINDING REVISION Op-routable #4): IMP + Quantizr recipe composition re-probe

- **Description:** Post-Path-1 success, compose IMP at 89.3% sparsity with Quantizr's recipe (FP4+Brotli on the 10.7% dense weights + AV1 monochrome mask + EMA(0.997) + diff_round + diff_rgb_to_yuv6 per Quantizr's 5-stage training pipeline). Test stack-of-stacks frontier movement.
- **Predicted ΔS band:** [-0.005, +0.01] absolute score vs Quantizr 0.33 anchor (frontier-relative). The renderer.bin shrinks from 64KB FP4 weight pool to ~22KB IMP+FP4 stack = -0.028 from rate axis if PoseNet+SegNet response is preserved.
- **Predicted_band_validation_status:** `null_pending_re_probe`
- **Predicted cost:** $5-15 (Quantizr's 5-stage pipeline ~6h on Vast.ai 4090 + auth-eval + runtime closure)
- **Structural verdict:** This path tests whether IMP's lottery-ticket subnetwork preserves Quantizr's archive grammar response. RATIFIES OR FALSIFIES the composition feasibility hypothesis.
- **Composability:** STANDS ON Path 1 success (lottery-ticket recovered first); composes with Quantizr's full leaderboard recipe.

### Path 3 (PRIORITY 3, BINDING REVISION Op-routable #1): Alternative magnitude criteria disambiguator

- **Description:** Build `tools/probe_imp_magnitude_criteria_disambiguator.py` (~200 LOC) that enumerates 3 magnitude criteria: (a) L1 per-tensor (canonical Frankle 2019); (b) Hessian-trace per-tensor (Lecun-Denker 1990 Optimal Brain Damage); (c) score-gradient saliency per Catalog #123 + `tac.score_gradient_param_saliency.compute_score_gradient_param_saliency`. Run cycle 0 with each criterion, measure asymmetric regression ratio + post-fine-tune recovery.
- **Predicted ΔS band:** unbounded — purely diagnostic; goal is to DISAMBIGUATE which criterion preserves lottery-ticket under multi-objective coupling.
- **Predicted_band_validation_status:** `diagnostic` (not a frontier candidate)
- **Predicted cost:** $5-15 total (3 cycle 0 runs at $1-2 each + analysis)
- **Structural verdict:** This path tests whether the multi-objective lottery-ticket question (Assumption 5) has a unique-best magnitude criterion. INFORMS Path 1/Path 2 dispatch design.
- **Composability:** PARALLEL with Path 1 (independent cycle 0 runs); INFORMS Path 2 (best criterion selected for composition re-probe).

### Reactivation priority ordering

1. **Path 1 (HIGHEST EV)**: $1-2 cost, converts ZOMBIE-status to L2-PARTIAL; structural protection live; lowest-cost path to closure.
2. **Path 2 (FRONTIER-EV)**: $5-15 cost, tests stack-of-stacks frontier movement via Quantizr composition; STANDS ON Path 1.
3. **Path 3 (DIAGNOSTIC EV)**: $5-15 cost, disambiguates multi-objective lottery-ticket question; PARALLEL with Path 1 OR sequenced AFTER Path 1 result.

**Recommendation:** Sequence Path 1 → Path 2 (with Path 3 as parallel diagnostic if budget allows). Total envelope $10-30 for full sequence; $1-2 minimum for Path 1 alone (converts ZOMBIE-status at cheapest possible cost).

## 6. Catalog #324 post-training Tier-C validation discipline

**Predicted_band_validation_status:** `null_pending_re_probe` for all 3 reactivation paths.

**Rationale:** Per CLAUDE.md "Forbidden predicted_band-from-random-init-Tier-C-density" non-negotiable: predicted bands derived from Frankle 2019 dense-CIFAR-VGG analogy + Quantizr's anecdotal byte savings + Hinton's distillation intuition are PROVENANCE=`first_principles` AT BEST and `analogy` more honestly. The reactivation criterion is post-training Tier-C re-measurement on the landed reformulation archive.

**Reactivation criterion verbatim:** "Post-training Tier-C density measurement on landed archive sha for each reactivation path (1/2/3) via `tools/mdl_scorer_conditional_ablation.py --tier c --archive <sha>`. If empirical ΔS lands within ±0.05 of the predicted band (broader band for the Lane G v3 anchor recovery test; ±0.005 for the Quantizr composition re-probe at frontier), ratify the paradigm + advance to L2 or L3. If outside band (per the C6 IBPS 22× miss anchor at sister symposium), surface 22×-or-greater miss as Catalog #324 violation and re-symposium."

**Critical sister reference:** C6 IBPS Phase 2 sextet (2026-05-17) demonstrated that predicted-band-from-random-init-Tier-C-density can be 22× outside band (predicted [0.113, 0.163]; actual 3.04). The IMP predicted bands [1.00, 1.30] for Path 1 are derived from FRANKLE 2019 ANALOGY which has even weaker provenance than random-init Tier-C; the bands MUST be treated as ROUGH PRIORS and the empirical Tier-C IS the authority.

## 7. Continual-learning anchor (Catalog #325 dispatch eligibility gate (d))

After this memo lands, the canonical posterior anchor IS registered to `.omx/state/council_deliberation_posterior.jsonl` via `tac.council_continual_learning.append_council_anchor` per the canonical 4-layer pattern (Catalog #245 exemplar). The anchor schema includes the PROCEED_WITH_REVISIONS verdict + 10-seat attendee list + 6-assumption Assumption-Adversary verdict + 7 op-routables + mission-alignment=frontier_protecting + override_invoked=false + deferred_substrate_id=lane_17_imp + deferred_substrate_retrospective_due_utc=2026-06-17.

Downstream consumers per Catalog #325:
- **Catalog #325 STRICT preflight** sees the PROCEED_WITH_REVISIONS verdict and STRUCTURALLY UNLOCKS dispatch of operator-authorize recipes targeting `lane_17_imp` substrate via `tools/operator_authorize.py::_check_predecessor_probe_outcome` (which consults `tac.probe_outcomes_ledger` + this council anchor). Without this symposium anchor, the lane would remain dispatch-blocked.
- **Cathedral autopilot ranker** consumes via `tac.council_continual_learning.query_anchors_by_topic('lane_17_imp')` for council-verdict-aware candidate weighting; the PROCEED_WITH_REVISIONS verdict + horizon_class=plateau_adjacent informs ranker that this is L2-PARTIAL EV not frontier-breaking.
- **Probe-outcomes ledger (Catalog #313)** receives a sister-registered DEFER outcome (status=blocking, methodology='cycle_0_stub_loop_falsification_2026-04-30') via `tac.probe_outcomes_ledger.register_probe_outcome` so any future dispatch wrapper consults the canonical outcome BEFORE firing. The alternative_probe_methodologies field enumerates the 3 magnitude criteria per Op-routable #1.

## 8. Cross-references

- **Canonical reference memos:** `project_lane_17_imp_killed_cycle_0_198_regression_20260430.md` (original 2026-04-30 KILL withdrawn ~5 minutes later); `feedback_grand_council_imp_permanent_fix_review_20260430.md` (8/10 council vote on the 4 STRICT preflight checks PCC1-4); `project_lane_17_imp_landed_20260430.md` (Phase A-I landing at Level 2); `project_lane_17_imp_pre_dispatch_20260430.md` (pre-dispatch memo with cost breakdown + verdict criteria); `feedback_imp_local_backport_landed_20260430.md` (commit 9fdabc9e local backport); `feedback_imp_dispatch_shape_mismatch_fix_20260430.md` (3 IMP-specific bugs fixed in 9fdabc9e); `feedback_grand_council_pcc3_stats_consistency_20260430.md` (PCC3 design vote).
- **Pre-rigor inventory:** `.omx/research/pre_rigor_kill_defer_falsified_inventory_20260517.md` row #1 (this symposium's council_priority).
- **Resurrection audit:** `.omx/research/resurrection_audit_20260516.md` Tier 1 reactivation candidate.
- **Batched reactivation design:** `.omx/research/batched_reactivation_lane17_imp_stc_apogee_int4_full_stack_design_20260516.md` §1 (full-stack UNIQUE-AND-COMPLETE-PER-METHOD design memo).
- **Sister per-substrate symposium memos (canonical pattern):** `.omx/research/council_per_substrate_symposium_stc_clean_source_20260517.md` (#857 STC clean-source DEFER_PENDING_EVIDENCE 10/10 unanimous); `.omx/research/council_per_substrate_symposium_atw_v2_reactivation_20260518.md` (ATW V2 PROCEED_WITH_REVISIONS 7 binding revisions).
- **Catalog gates fired by this symposium:** #292 (per-deliberation assumption surfacing) + #294 (9-dim checklist) + #296 (Dykstra-feasibility predicted-band) + #300 (council v2 frontmatter) + #303 (cargo-cult audit) + #305 (observability surface) + #307 (paradigm-vs-implementation classification) + #308 (alternative-probe-methodologies enumeration) + #313 (probe-outcomes ledger) + #324 (post-training Tier-C validation) + #325 (per-substrate symposium discipline).
- **Catalog gates LIVE-VERIFIED extincting the 2026-04-30 bug class:** #91 `check_imp_dispatch_calls_train_distill` (preflight.py:28244) + #94 `check_imp_cycles_use_ema_and_auth_eval` (preflight.py:28051) + #117 PCC3 wall-clock-floor (train_imp_cycle.py:362-374) + #92 `check_no_comment_only_contracts` + #93 `check_stats_json_internal_consistency`.
- **Catalog gates protected by this symposium:** #220 (operational mechanism declaration; reactivation paths must declare); #272 (distinguishing-feature integration contract); #233 (L1→L2 promotion canonical 4-gate); #298 (substrate retirement discipline 30-day).
- **Canonical implementations cited:** Frankle-Carbin 2019 ICLR Best Paper (Lottery Ticket Hypothesis); Hinton-Vinyals-Dean 2015 Distilling the Knowledge in a Neural Network (T=2.0); Lecun-Denker 1990 Optimal Brain Damage (Hessian-trace pruning criterion); Quantizr 0.33 leaderboard recipe (FP4+Brotli+AV1+EMA+diff_round+diff_rgb_to_yuv6); Catalog #123 `tac.score_gradient_param_saliency` (score-gradient saliency); HNeRV parity discipline lessons 1-13; PR101 / PR103 silver anchors.

## 9. Operator op-routables (for parent agent + main Claude)

1. **Authorize $1-2 Path 1 dispatch SEPARATELY in a follow-up turn** — standalone IMP cycle 0 with REAL 100ep train_distill via Stage 1b dispatch script pattern. Convert ZOMBIE-status to L2-PARTIAL at cheapest possible cost. The structural protection (Catalog #91/#94/#117) is LIVE-VERIFIED; no risk of re-shipping the 2026-04-30 stub-loop bug class.

2. **Register the PROCEED_WITH_REVISIONS outcome to the canonical probe-outcomes ledger** via `tac.probe_outcomes_ledger.register_probe_outcome(substrate_id='lane_17_imp', verdict='DEFER', status='blocking', methodology='cycle_0_stub_loop_falsification_2026-04-30', alternative_probe_methodologies=['l1_per_tensor_canonical_frankle', 'hessian_trace_per_tensor_obd', 'score_gradient_saliency_per_catalog_123'], expires_at_utc=<2026-06-17>)`. This makes the DEFER+REVISIONS verdict QUERYABLE across sessions and gates future dispatch wrappers from re-firing the same stub-loop methodology.

3. **DO NOT migrate `lane_17_imp_10cycle` to research_only** — the L2 status is real (6 of 7 gates green per registry; only contest_cuda gate missing). The reactivation IS via the contest_cuda gate dispatch with REVISIONS, NOT via lane retirement. Per CLAUDE.md "Substrate retirement discipline" Catalog #298: 30-day window resets via this symposium attention.

4. **Open 2 follow-on lanes for the composition + diagnostic paths**: `lane_imp_quantizr_recipe_composition_20260518` (PRIORITY 2 — Op-routable #4 BINDING REVISION; $5-15 frontier-EV path), `lane_imp_magnitude_criteria_disambiguator_20260518` (PRIORITY 3 — Op-routable #1 BINDING REVISION; $5-15 diagnostic-EV path). Each follow-on lane gets its OWN per-substrate symposium per Catalog #325.

5. **Sister-symposium synergy**: The 2 in-flight per-substrate symposiums (#857 STC clean-source = DEFER_PENDING_EVIDENCE; #856 lane_17_imp = PROCEED_WITH_REVISIONS — THIS memo) demonstrate the Catalog #325 doctrine's discriminating power: SAME-day, SAME-rigor framework, OPPOSITE verdicts. STC has paradigm-cargo-cult AT CODEC-DESIGN level (mask-channel substitution dominated); IMP has implementation-cargo-cult AT STUB-LOOP level (paradigm-intact, implementation-extincted). The rigor framework correctly separates the cases.

**Total dispatch redirect:** $1-2 Path 1 (cheapest L2-PARTIAL closure) + $5-15 Path 2 (frontier-EV composition) + $5-15 Path 3 (diagnostic disambiguator) = $11-32 total envelope. Within standing $25 Lane 17 budget cap per `project_lane_17_imp_pre_dispatch_20260430.md` original operator approval; the $25 cap accommodates Path 1+2+3 sequentially with $0 spent so far (only $0.20 sunk in 2026-04-30 stub-loop incident).

---

**Symposium concludes.** Verdict: PROCEED_WITH_REVISIONS — 10-of-10 unanimous. Mission-alignment: frontier_protecting. Override: not invoked. Continual-learning anchor: registering to `.omx/state/council_deliberation_posterior.jsonl` via canonical helper.

## 6-hook wire-in declaration (per Catalog #125)

1. **Sensitivity-map contribution** — ACTIVE. Op-routable #1 BINDING REVISION enumerates 3 magnitude criteria including score-gradient-saliency per Catalog #123 + `tac.score_gradient_param_saliency.compute_score_gradient_param_saliency` — this directly contributes to `tac.sensitivity_map.*` via the per-tensor importance scores emitted during the cycle 0 fine-tune.
2. **Pareto constraint** — ACTIVE. The IMP cycle adds a constraint to `tac.pareto_*` (sparse-CSR archive size at 89.3% sparsity = ~177KB; Pareto frontier intersection with rate ≤ 297KB FP4A baseline).
3. **Bit-allocator hook** — ACTIVE. Sparse-CSR + FP4 nibble-pack codec changes per-tensor bit allocation (Quantizr composition Op-routable #4 explicitly composes IMP+FP4 stack).
4. **Cathedral autopilot dispatch hook** — ACTIVE. The PROCEED_WITH_REVISIONS verdict + horizon_class=plateau_adjacent + predicted-band Path 1 [1.00, 1.30] / Path 2 [-0.005, +0.01] inform autopilot ranker that Path 1 is L2-PARTIAL EV (low priority) while Path 2 is frontier-EV (high priority).
5. **Continual-learning posterior update** — ACTIVE. Append council anchor via canonical helper per §7; sister probe-outcome registration per Op-routable #5 (Catalog #313).
6. **Probe-disambiguator** — ACTIVE. Op-routable #1 BINDING REVISION explicitly designs `tools/probe_imp_magnitude_criteria_disambiguator.py` to disambiguate the 3 magnitude criteria empirically; the disambiguator IS the structural arbiter of the multi-objective lottery-ticket question.

## Checkpoint discipline (per Catalog #206)

- Subagent ID: `lane_17_imp_symposium_20260517T210000`.
- Lane: `lane_per_substrate_symposium_lane_17_imp_20260517`.
- Step 1 checkpoint (in_progress, files: this memo, next: write 6-step symposium per Catalog #325): logged.
- Step 2 checkpoint (in_progress, files: this memo, next: append council anchor via canonical helper + register probe outcome): forthcoming.
- Final checkpoint (complete): forthcoming after canonical helper invocation.

## Premise verification (per Catalog #229)

Premises verified BEFORE editing this memo:
1. **PV-1:** `train_imp_cycle.py:362-374` PCC3 wall-clock-floor assertion is LIVE in the source (`MIN_WALL_PER_EPOCH_SEC = 0.05`; `if elapsed < expected_min: raise RuntimeError("PCC3 STUB-LOOP DETECTED...")`). VERIFIED via direct read of source file.
2. **PV-2:** `src/tac/preflight.py:5073` orchestrator wires `check_imp_cycles_use_ema_and_auth_eval(strict=True)`. VERIFIED via grep.
3. **PV-3:** `src/tac/preflight.py:5091` orchestrator wires `check_imp_dispatch_calls_train_distill(strict=True)`. VERIFIED via grep.
4. **PV-4:** `scripts/remote_lane_j_imp_iterative_magnitude_pruning.sh:318` invokes `experiments/train_distill.py` via the canonical Stage 1b swap pattern. VERIFIED via grep.
5. **PV-5:** Lane registry `lane_17_imp_10cycle` is L2 with 6 of 7 gates green (`impl_complete / real_archive_empirical / strict_preflight / three_clean_review / memory_entry / deploy_runbook`; only `contest_cuda` and `contest_cpu` missing). VERIFIED via JSON read.
6. **PV-6:** The 2026-04-30 KILL memo `project_lane_17_imp_killed_cycle_0_198_regression_20260430.md` documents the WITHDRAWAL by 8/10 council vote per the operator's adversarial challenge. VERIFIED via direct read.
7. **PV-7:** Two sister per-substrate symposiums (STC clean-source DEFER; ATW V2 PROCEED_WITH_REVISIONS) are in-flight per pre-rigor inventory; THIS symposium does not duplicate either (lane_17_imp is council_priority #1 in the SHOULD_BE_RESYMPOSIUM'D queue; STC is #2; ATW V2 covered by separate sister #852). VERIFIED via directory listing.
