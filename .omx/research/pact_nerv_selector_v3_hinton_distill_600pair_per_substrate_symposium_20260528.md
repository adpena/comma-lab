<!-- SPDX-License-Identifier: MIT -->
---
council_tier: T2
council_attendees:
  - Shannon
  - Dykstra
  - Rudin
  - Daubechies
  - Yousfi
  - Fridrich
  - Contrarian
  - Assumption-Adversary
  - Quantizr
  - Hotz
  - Selfcomp
  - MacKay
  - Balle
  - PR95Author
  - Hinton
  - vdOord
  - Atick
  - Redlich
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "The 3-way 'compounding validated ideas → sub-0.18 candidate' framing inherits a CARGO-CULTED additive composition prior. Per CLAUDE.md 'Meta-Lagrangian/Pareto solver' non-negotiable + Catalog #322 sister discipline: empirical composition_alpha REQUIRES paired-CUDA anchor. The MLX-research-signal scorer-bound finite convergence proves the architecture BINDS; it does NOT prove the composition is super-additive vs additive vs sub-additive on the contest scorer axes. PROCEED only with explicit acknowledgement that the sub-0.18 claim is a HYPOTHESIS NOT a prediction; paired-CUDA test produces the empirical anchor that DISAMBIGUATES."
  - member: Assumption-Adversary
    verbatim: "The shared assumption framing 'SELECTOR-V3 substrate × Hinton-distilled scorer surrogate × 600-pair scale = canonical sub-0.18 candidate' inherits 3 cargo-culted priors: (1) per-pair difficulty-conditioned Rice-Golomb coder generalizes scorer-axis distillation gain from 8-pair smoke to 600-pair; (2) scorer-bound finite convergence at MLX-research-signal axis maps to scorer-bound finite convergence at contest-CUDA axis (the canonical [macOS-MLX vs contest-CUDA] gap per CLAUDE.md 'MPS auth eval is NOISE'); (3) PR95Author's 8-stage curriculum precedent generalizes from PR95-HNeRV substrate-class to PACT-NeRV-SELECTOR-V3 substrate-class. Each is empirically falsifiable; the per-substrate symposium's value-add is naming them HARD-EARNED-vs-CARGO-CULTED so the paired-CUDA empirical anchor disambiguates with full information."
  - member: Yousfi
    verbatim: "Contest scorer is SegNet (smp.Unet tu-efficientnet_b2 5-class) + PoseNet (FastViT-T12 12-channel YUV6). The Hinton-distilled learnable 1x1-conv SegNet student head + learnable pool+linear PoseNet student head are SURROGATES not the real scorer; per CLAUDE.md 'eval_roundtrip — non-negotiable' + Catalog #164 the canonical scorer_loss_terms_btchw routes through the REAL scorer via canonical preprocess_input. The MLX-LOCAL training's combined loss descent IS evidence that gradient signal binds end-to-end through the surrogate; the contest-axis test requires a paired Linux x86_64 + NVIDIA dispatch per Catalog #246 that routes through the REAL SegNet + PoseNet, not the learnable surrogates. The 'scorer-bound finite convergence' verdict is HARD-EARNED at the surrogate-binding axis; it remains an OPEN EMPIRICAL question whether real-scorer-axis convergence shows the same multi-phase signature."
council_assumption_adversary_verdict:
  - assumption: "3-way combination (SELECTOR-V3 substrate × Hinton-distilled scorer surrogate × 600-pair scale) produces sub-0.18 contest-CUDA score on paired Linux x86_64 + NVIDIA hardware"
    classification: CARGO-CULTED-PENDING-EMPIRICAL-ANCHOR
    rationale: "Empirical anchor at this symposium is reconstruction-floor + scorer-bound-finite-convergence verdict at MLX-research-signal axis (loss 107.53 → 3.40 = 31.66× reduction; min reached at last epoch indicating training NOT saturated; combined loss not directly comparable to pure-pixel-MSE). Per CLAUDE.md 'MPS auth eval is NOISE' (MPS PoseNet drift 23×; SegNet 2×; final score 2.5×) + Catalog #1/#127/#192/#317/#341 non-promotable: NEVER promote MLX-research-signal as contest-axis evidence without paired Linux x86_64 + NVIDIA replay per Catalog #246. The sub-0.18 claim is a HYPOTHESIS that the paired-CUDA dispatch DISAMBIGUATES. Unwind path: Path 1 (paired-CUDA dispatch ~$1-3) IS the canonical apples-to-apples disambiguator."
  - assumption: "Hinton KL-T=2.0 distillation generalizes to PACT-NeRV cascade from PR95-HNeRV origin"
    classification: HARD-EARNED
    rationale: "Quantizr 0.33 canonical anchor (PR95-HNeRV substrate class; KL T=2.0 SegNet distillation during training) + sister IA3 integration smoke (commit b551bfd34; 14% loss reduction at 0.42s training wall-clock) + 8 cumulative canonical equation #1 anchors (`hinton_distilled_scorer_surrogate_savings_via_kl_t2_v1`) including 6 PR95-HNeRV sisters + IA3 substrate-class extension + THIS SELECTOR-V3 substrate-class extension. The IA3 smoke EMPIRICALLY proved cross-substrate-class generalization at MLX-research-signal axis. SELECTOR-V3's substrate-distinguishing primitive (per-pair difficulty-conditioned Rice-Golomb at archive-encode time) is ORTHOGONAL to MLX forward path so distillation operates on the same base HNeRV decoder topology as the IA3 sister."
  - assumption: "600-pair pure-pixel-MSE per-pair-difficulty generalization floor 0.00284 maps to Hinton-distilled scorer-axis floor in the same direction (sub-0.18 push REQUIRES scorer-axis distillation at full contest scale)"
    classification: CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION
    rationale: "The 600-pair pure-pixel-MSE landing (`pact_nerv_selector_v3_extended_600pair_long_mlx_landed_20260528.md`) EMPIRICALLY FALSIFIED the cargo-cult prediction that 32-pair saturation generalizes to 600-pair scale (600-pair floor 0.00284 = 1.94× WORSE than 32-pair 0.00146). The current cargo-cult is that the per-pair generalization floor in pure-pixel-MSE space ALSO bounds the scorer-axis distillation gain — but the scorer-axis is a DIFFERENT objective surface (per CLAUDE.md 'SegNet vs PoseNet importance — operating-point dependent'): at frontier operating point pose_avg ~3.4e-5, pose marginal sensitivity is 2.71× SegNet's. The MLX-research-signal evidence (combined-loss multi-phase descent with min-at-last-epoch indicating NOT saturated) is CONSISTENT WITH but does NOT PROVE that the scorer-axis floor is more permissive than pure-pixel-MSE per-pair floor. Unwind path: Path 1 + Path 4 (paired-CUDA + per-axis scorer component sweep) DISAMBIGUATES this empirically."
  - assumption: "scorer-bound finite convergence at MLX-LOCAL ep 1999 STILL-DESCENDING implies cascade unblocking (further training would continue producing gradient signal that moves renderer parameters)"
    classification: HARD-EARNED
    rationale: "Empirical telemetry: 5-phase loss-trajectory profile (Phase 1 fast initial descent ep 0-10 = 1.17×; Phase 2 sharp descent ep 10-100 = 20.3×; Phase 3 SLOW descent ep 100-500 = 1.10×; Phase 4 SLOW slow ep 500-1500 = 1.08×; Phase 5 continued ep 1500-2000 = 1.12× with min at LAST epoch 1999). EMA drift L2 = 2.195 at ep 1999 (vs 0.054 at ep 0) confirms renderer parameters DEMONSTRABLY move under combined scorer-bound gradient. The CARGO-CULT alternative was 'renderer is decoupled from scorer-axis terms because MLX second-order autograd NaN propagates'; the learnable-head surrogate empirically falsifies this alternative. Path 3 (extended-epoch training at ep 5000-10000 MLX-LOCAL) tests the 'further convergence' sub-claim."
  - assumption: "SELECTOR-V3 per-pair difficulty-conditioned arithmetic coder is OPTIMAL vs sister cascade variants (V2 / V4 / VQ) for Hinton-extension at contest-CUDA axis"
    classification: CARGO-CULTED-PENDING-APPLES-TO-APPLES-SISTER-COMPARISON
    rationale: "SELECTOR-V3 achieved the TIGHTEST 32-pair MLX-LOCAL pixel-reconstruction floor (0.00146 vs SELECTOR-V2's 0.00172 = 15.1% LOWER; vs IA3's 0.0024 = 39.2% LOWER per `pact_nerv_selector_v3_l1_long_run_mlx_landed_20260528.md`), but the difference is stochastic seed variance + AdamW noise within the same base HNeRV decoder topology — NOT a substrate-distinguishing effect (per the sister landing's own verdict). The Rice-Golomb coder operates at archive-encode TIME, not in MLX forward path, so MLX-LOCAL signal cannot disambiguate cascade-internal optimality. Path 2 (sister cascade Hinton+600-pair batch V2 / V4 / VQ at $0 MLX-LOCAL each, ~150-200s each = 4 canonical equation #1 anchors at ~10-13 min total) IS the canonical apples-to-apples disambiguator at MLX-research-signal axis BEFORE paired-CUDA spend on any single cascade member."
council_decisions_recorded:
  - "op-routable #1 (TOP-1): paired Linux x86_64 + NVIDIA dispatch on SELECTOR-V3 + Hinton-distilled + 600-pair archive (sha256 ef5a087ff6301dbf...) via canonical operator-routable command sheet (see Phase 5 below); operator-attended per just-discovered structural constraint about subagent-attempted recipe flips; predicted envelope ~$1-3 paired CPU+CUDA on contest-compliant T4; canonical 4-verdict chain prerequisites (Catalog #370 Phase 4+5+6+7) surface as op-routable for next cycle along with recipe authoring per Catalog #240."
  - "op-routable #2 (PARALLEL DISJOINT): sister Slot 2 Hinton + 600-pair cascade batch (V2 + V4 + VQ; 3 more substrate-class anchors at $0 MLX-LOCAL each via IDENTICAL canonical pattern; total ~10-13 min wall-clock). Each anchor compounds canonical equation #1 (anchor count 8 → 11) and produces apples-to-apples sister comparison at MLX-research-signal axis BEFORE paired-CUDA dispatch on any single cascade member. Sister-disjoint per Catalog #314/#340 to THIS symposium subagent's scope."
  - "op-routable #3 (RESEARCH-DEEP): extended-epoch training at ep 5000-10000 MLX-LOCAL on SELECTOR-V3 + Hinton-distilled + 600-pair to test the 'min-at-last-epoch implies further convergence' sub-claim per Path 3 reactivation criterion; ~$0 MLX-LOCAL; ~6-13 min wall-clock per the 159.1s/2000ep linear extrapolation."
  - "op-routable #4 (DIAGNOSTIC): scorer-axis component sweep on SELECTOR-V3 + Hinton-distilled (per-axis seg-only at distillation_weight=1.0 + pose-MSE=0 / pose-only at pose-MSE=1.0 + distillation_weight=0 / combined at canonical defaults) at MLX-LOCAL ~$0 to disambiguate which scorer-axis dominates the combined-loss descent; informs Path 4 reactivation criterion."
  - "op-routable #5 (FOLLOW-UP CATALOG #324): when paired-CUDA empirical anchor lands per op-routable #1, register the post-training Tier-C density measurement on the landed archive sha256 to flip predicted_band_validation_status from `pending_post_training` to `post_training_*ep_*` per Catalog #324; this satisfies the 6th symposium step (Tier-C validation discipline) at canonical-evidence-grade."
  - "REVISION #1 (binding per Contrarian dissent): sub-0.18 claim is HYPOTHESIS NOT prediction; every cross-reference to this symposium must use 'sub-0.18 candidate' (provisional) language NOT 'sub-0.18 substrate' (asserted)."
  - "REVISION #2 (binding per Assumption-Adversary dissent): paired-CUDA dispatch result must be tagged at all 3 cargo-cult-pending axes (3-way combination outcome; pure-pixel-MSE-to-scorer-axis floor mapping; sister cascade optimality) so the empirical anchor disambiguates with full information."
  - "REVISION #3 (binding per Yousfi dissent): MLX-research-signal verdict explicitly tagged 'surrogate-binding axis' to distinguish from 'real-scorer axis'; paired-CUDA dispatch is the real-scorer empirical anchor."
related_deliberation_ids:
  - pact_nerv_selector_v3_hinton_distill_600pair_long_mlx_landed_20260528
  - pact_nerv_selector_v3_extended_600pair_long_mlx_landed_20260528
  - hinton_distilled_scorer_surrogate_mlx_local_pact_nerv_ia3_integration_landed_20260528
  - pact_nerv_selector_v3_l1_long_run_mlx_landed_20260528
canonical_equations_referenced:
  - hinton_distilled_scorer_surrogate_savings_via_kl_t2_v1
related_canonical_artifacts:
  - experiments/results/pact_nerv_selector_v3_hinton_distill_600pair_long_mlx_20260528T080350Z/training_artifact.json
  - experiments/results/pact_nerv_selector_v3_hinton_distill_600pair_long_mlx_20260528T080350Z/telemetry.jsonl
  - experiments/results/pact_nerv_selector_v3_hinton_distill_600pair_long_mlx_20260528T080350Z/archive.zip
  - experiments/train_substrate_pact_nerv_selector_v3_mlx_local.py
  - src/tac/substrates/_shared/mlx_score_aware/
  - src/tac/substrates/hinton_distilled_scorer_surrogate/
  - src/tac/substrates/pact_nerv_selector_v3/
  - .omx/state/canonical_equations_registry.jsonl
canonical_axis: "[macOS-MLX research-signal]"
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
task_id: 1448
lane_id: lane_pact_nerv_selector_v3_hinton_distill_600pair_per_substrate_symposium_20260528
captured_at_utc: "2026-05-28T08:36:00Z"
horizon_class: frontier_pursuit
predicted_band_validation_status: pending_post_training
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_retrospective_due_utc: ""
deferred_substrate_id: pact_nerv_selector_v3_hinton_distill_600pair_mlx_local
schema: council_deliberation_posterior_v1
---

# PACT-NeRV-SELECTOR-V3 + Hinton-distilled scorer surrogate × 600-pair PER-SUBSTRATE SYMPOSIUM 2026-05-28

## 0. Operator mandate (verbatim 2026-05-28)

> *"PER-SUBSTRATE SYMPOSIUM for SELECTOR-V3 + HINTON-DISTILLED + 600-PAIR
> CANDIDATE — task #1448 IN_PROGRESS. $0 MLX-local deliberation (NO paid
> dispatch from this symposium subagent). Per just-landed empirical (commit
> ab650cc78) operator-routable TOP-1: enables operator-attended paired Linux
> x86_64 + NVIDIA dispatch per Catalog #246 in next cycle."*

## 1. Symposium scope per Catalog #325

This is the canonical per-substrate symposium for the 3-way candidate
**SELECTOR-V3 + Hinton-distilled scorer surrogate + 600-pair scale** prior to
any paid Linux x86_64 + NVIDIA dispatch per Catalog #246 + CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA" non-negotiable. The symposium
satisfies the canonical Catalog #325 6-step contract:

1. Cargo-cult audit per Catalog #303 (§3 below)
2. 9-dimension success checklist evidence per Catalog #294 (§4 below)
3. Observability surface declaration per Catalog #305 (§5 below)
4. Sextet+grand-council deliberation per CLAUDE.md "Council conduct"
   amendment + Catalog #292 per-deliberation assumption surfacing +
   Catalog #346 canonical roster (this entire memo + frontmatter)
5. Per-substrate reactivation criteria pinned per CLAUDE.md "Forbidden
   premature KILL" (§6 below)
6. Catalog #324 post-training Tier-C validation discipline (§7 below)

## 2. Canonical roster validation per Catalog #346

```
.venv/bin/python -c "from tac.canonical_council_roster import validate_council_dispatch_roster; ..."
verdict.complete: True
missing_inner_council: ()
missing_co_leads: ()
missing_relevant_grand_council: ()
unknown_attendees: ()
```

Full INNER COUNCIL (14): Shannon LEAD + Dykstra CO-LEAD + Rudin CO-LEAD +
Daubechies CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary
+ Quantizr + Hotz + Selfcomp + MacKay + Balle + PR95Author.

GRAND COUNCIL topical (4): Hinton (memorial seat for KL-distillation
paradigm) + vdOord (VQ-VAE sister substrate) + Atick + Redlich
(cooperative-receiver framing for scorer-binding per Catalog #311).

## 3. Cargo-cult audit per Catalog #303

| # | Assumption | Classification | Unwind path |
|---|---|---|---|
| 1 | 3-way combination produces sub-0.18 contest-CUDA score on paired hardware | **CARGO-CULTED-PENDING-EMPIRICAL-ANCHOR** | Path 1 paired-CUDA dispatch (~$1-3) IS the canonical apples-to-apples disambiguator |
| 2 | Hinton KL T=2.0 distillation generalizes to PACT-NeRV cascade | **HARD-EARNED** | Quantizr 0.33 anchor + 8 canonical equation #1 anchors including IA3 + SELECTOR-V3 substrate-class extensions |
| 3 | 600-pair pure-pixel-MSE floor 0.00284 maps to scorer-axis floor in same direction | **CARGO-CULTED-PENDING-EMPIRICAL-VERIFICATION** | Path 1 + Path 4 (paired-CUDA + per-axis scorer component sweep) DISAMBIGUATE |
| 4 | scorer-bound finite convergence at MLX ep 1999 STILL-DESCENDING implies cascade unblocking | **HARD-EARNED** | EMA drift L2 = 2.195 at ep 1999 (vs 0.054 at ep 0) empirically confirms renderer parameters demonstrably move under combined scorer-bound gradient |
| 5 | SELECTOR-V3 is OPTIMAL vs sister cascade (V2 / V4 / VQ) for Hinton-extension | **CARGO-CULTED-PENDING-APPLES-TO-APPLES-SISTER-COMPARISON** | Path 2 sister cascade batch (V2 + V4 + VQ at $0 MLX-LOCAL each) IS canonical apples-to-apples disambiguator |
| 6 | `distillation_temperature = 2.0` per Hinton 2014 | HARD-EARNED | Sister 8 anchors all converged consistently at T=2.0 (canonical equation #1 anchor sweep) |
| 7 | `distillation_weight = 0.5` (recon-dominant) | HARD-EARNED | IA3 smoke used 1.0; this run uses 0.5 to weight recon more (SELECTOR-V3 has per-pair difficulty-conditioned Rice-Golomb at archive-encode time so recon-axis is more substrate-distinguishing) |
| 8 | `pose_distillation_weight = 1.0` (PoseNet dominant at frontier) | HARD-EARNED | Per CLAUDE.md "SegNet vs PoseNet importance" operating-point dependent: at frontier (pose_avg ~3.4e-5) pose marginal is 2.71× SegNet; pose-MSE 1.0 weight is canonical |
| 9 | `learnable 1x1-conv SegNet student head` + `learnable pool+linear PoseNet student head` | HARD-EARNED | MLX second-order autograd NaN finding per `mlx_score_aware/loss.py:117-128`; learnable-head surrogates give FINITE gradient |
| 10 | `device="cpu"` for teacher cache build | HARD-EARNED | CLAUDE.md "MPS auth eval is NOISE": MPS PoseNet drift 23× — CPU teacher only |
| 11 | The learnable-head SURROGATES generalize to REAL-SegNet + REAL-PoseNet scorer-axis at paired-CUDA | **CARGO-CULTED-AT-REAL-SCORER-AXIS** (per Yousfi dissent) | Path 1 paired-CUDA dispatch routes through canonical scorer_loss_terms_btchw with REAL scorer + canonical preprocess_input per Catalog #164; this IS the canonical real-scorer empirical anchor |

The 3 newly-named CARGO-CULTED assumptions (rows 1, 3, 5, 11) are NOT
unwound at this symposium — they are HYPOTHESES that the canonical Path 1
+ Path 2 + Path 4 reactivation paths empirically test in the next cycle.
The remaining 7 HARD-EARNED assumptions inherit from the sister 8-anchor
canonical equation #1 precedent + the just-landed empirical 5-phase descent
profile.

## 4. 9-dimension success checklist evidence per Catalog #294

1. **UNIQUENESS** — 3-way combination NEVER tested before this candidate.
   SELECTOR-V3 substrate-class (Rice-Golomb per Golomb 1966 + Rice 1971
   over k=16 palette per Step 12 ULTIMATE STAIRCASE; per-pair difficulty-
   conditioned arithmetic coding at archive-encode time) is its OWN per
   the 11th INDIVIDUALLY-FRACTAL standing directive; Hinton-distilled
   scorer surrogate (real SegNet teacher cache + real PoseNet teacher cache
   + learnable student heads at MLX layer; KL T=2.0 + pose-MSE composition)
   is canonical sister-precedent across 8 anchors but FIRST 3-way
   combination on SELECTOR-V3 substrate-class; 600-pair scale IS contest-
   compliant scale per CLAUDE.md "Submission auth eval — BOTH CPU AND
   CUDA" non-negotiable.

2. **BEAUTY + ELEGANCE** — 1 trainer modification (~50 LOC canonical Hinton
   wiring) + 1 CLI flag (`--pose-distillation-weight`); zero forks of
   canonical primitives. Reviewable in 30 seconds per Quantizr's PR101-class
   binding precedent. Canonical Hinton-Vinyals-Dean 2014 KL T=2.0 +
   canonical mlx_score_aware harness + canonical SELECTOR-V3 renderer +
   canonical EMA decay 0.997 + canonical AdamW optimizer + canonical L2
   long_training_canonical harness.

3. **DISTINCTNESS** — explicitly NOT IA3 (different substrate primitive:
   γ-only ego-pose modulation per Liu 2022 §3.2 vs Rice-Golomb per Golomb
   1966); explicitly NOT pure-pixel-MSE 600-pair (different loss
   composition: recon + KL T=2.0 + pose-MSE vs recon-only); explicitly NOT
   IA3 Hinton smoke (different scale 600 vs 8 pairs + different substrate
   SELECTOR-V3 vs IA3); explicitly NOT 32-pair SELECTOR-V3 L1 baseline
   (different scale + different loss composition).

4. **RIGOR** — premise verification per Catalog #229 (cargo-cult audit
   surfacing 11 assumptions across 4 HARD-EARNED + 5 NEWLY-NAMED-CARGO-
   CULTED-PENDING-EMPIRICAL classifications); validation smoke 8-pair
   5-epoch BEFORE 600-pair LONG run (per the sister IA3 integration smoke
   pattern); canonical Provenance per Catalog #323; canonical equation
   registry update per Catalog #344 (anchor 7 → 8); fail-closed gates per
   Catalog #164 + the C6 IBPS lesson verified empirically; per-substrate
   symposium per Catalog #325 (THIS memo).

5. **OPTIMIZATION PER TECHNIQUE** — Hinton-Vinyals-Dean 2014 KL T=2.0 (NOT
   T=4.0 / T=1.0 per the 8-anchor sister sweep) + PoseNet pose-MSE per
   Catalog #164 + canonical EMA decay 0.997 + canonical AdamW optimizer +
   canonical L2 long_training_canonical harness. SELECTOR-V3's per-pair
   difficulty-conditioned Rice-Golomb coder operates at archive-encode time
   (NOT in MLX forward path) so the substrate-distinguishing primitive
   operates ORTHOGONALLY to the scorer-axis distillation surface.

6. **STACK-OF-STACKS COMPOSABILITY** — orthogonal axes confirmed:
   scorer-surrogate distillation (seg+pose axes) composes with SELECTOR-V3's
   per-pair difficulty-conditioned Rice-Golomb at archive-encode time
   (rate-axis); base HNeRV decoder (recon-axis); MLX harness (training
   infrastructure axis); potential composition matrix surface with sister
   substrates (NSCS06 v8 chroma_lut / fec6 / PR101 / PR106) per Catalog
   #322 canonical composition matrix. Cross-references to Slot 2 Hinton +
   600-pair sister cascade batch per op-routable #2 below.

7. **DETERMINISTIC REPRODUCIBILITY** — `--seed=0` pinned for all RNG keys
   (renderer + student head + EMA); canonical EMA decay 0.997 pinned;
   output under `experiments/results/` per Catalog #113 DERIVED_OUTPUT
   (NOT /tmp per CLAUDE.md FORBIDDEN_PATTERNS).

8. **EXTREME OPTIMIZATION + PERFORMANCE** — 159.1s wall-clock for 2000
   epochs on M5 Max MLX-LOCAL (vs 116.3s for 600-pair pure-pixel-MSE =
   +37% overhead for teacher cache build + per-step student-head
   distillation forward pass); canonical mlx_score_aware harness amortizes
   per-step compute via value_and_grad lazy eval; teacher cache build
   (1.04s SegNet + 0.82s PoseNet from IA3 sister precedent) amortized over
   2000 epochs.

9. **OPTIMAL MINIMAL CONTEST SCORE** — non-promotable
   `[macOS-MLX research-signal]` per Catalog #1/#127/#192/#317/#341;
   contest-axis claim DEFERRED to per-substrate symposium per Catalog #325
   (THIS memo) + paired Linux x86_64 + NVIDIA dispatch per Catalog #246
   (op-routable #1). The MLX-research-signal evidence IS the canonical
   PRE-paid-dispatch research signal that unblocks the next pivot toward
   the sub-0.18 candidate empirical test.

## 5. Observability surface declaration per Catalog #305

All 6 observability facets per CLAUDE.md "Max observability — non-negotiable":

| Facet | Coverage |
|---|---|
| **Inspectable per layer** | per-epoch loss + ema_drift_l2 + wall_clock at `experiments/results/.../training_artifact.json` `per_epoch_metrics` field; canonical TrainingArtifact schema |
| **Decomposable per signal** | combined loss decomposes into recon + KL T=2.0 + pose-MSE via canonical `score_aware_loss` `parts_dict`; sister follow-up could surface `loss_components` per Catalog #356 AxisDecomposition for full per-axis Provenance threading |
| **Diff-able across runs** | `--seed=0` produces bit-identical RNG keys (renderer init + student head init + EMA init); inputs hash deterministic via canonical `inputs_sha256` in Provenance per Catalog #323 |
| **Queryable post-hoc** | canonical posterior anchor at `.omx/state/canonical_equations_registry.jsonl` queryable via `tac.canonical_equations.registry.query_equations()`; sister canonical posterior at `.omx/state/council_deliberation_posterior.jsonl` queryable via `tac.council_continual_learning.query_anchors_by_topic` |
| **Cite-able** | every artifact carries canonical Provenance per Catalog #323 with `captured_at_utc` + `source_sha256` + `canonical_helper_invocation` + `axis_tag=[macOS-MLX research-signal]` + `evidence_grade` + `score_claim_valid=False` + `promotable=False` |
| **Counterfactual-able** | per-byte mutation of teacher cache files would re-trigger student head retraining (canonical no-op detector Catalog #105 + #139 applies); per-epoch EMA drift L2 reveals which gradient steps moved renderer most; per-step telemetry at `telemetry.jsonl` (622733 bytes) preserves full forensic surface |

## 6. Per-substrate reactivation criteria per CLAUDE.md "Forbidden premature KILL"

Per the standing directive, this symposium produces NO killed sub-hypothesis
— it produces DEFERRED-pending-research with explicit reactivation paths.
The 4 canonical paths with priority ordering + predicted cost + which
assumption each path tests:

### Path 1 (TOP-1 PRIORITY; ~$1-3 paired-CUDA):
**Paired Linux x86_64 + NVIDIA paired CPU+CUDA dispatch on the
SELECTOR-V3 + Hinton-distilled + 600-pair archive** (sha256
`ef5a087ff6301dbf...`; 137,351 bytes per the just-landed empirical memo).

- **Tests cargo-culted assumption**: row 1 (3-way combination → sub-0.18
  contest-CUDA score) + row 3 (pure-pixel-MSE floor → scorer-axis floor
  mapping) + row 11 (learnable-head surrogate → real-scorer axis generalization)
- **Predicted cost**: ~$1-3 paired T4 + Linux x86_64 CPU per the canonical
  dispatch envelope from sister recent paired anchors
- **Canonical apples-to-apples test** per CLAUDE.md "Apples-to-apples
  evidence discipline"; the empirical result EMPIRICALLY DISAMBIGUATES the
  sub-0.18 candidate hypothesis per CLAUDE.md "Submission auth eval — BOTH
  CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiable
- **Operator-attended per just-discovered structural constraint about
  subagent-attempted recipe flips** (recipe authoring per Catalog #240 +
  4-verdict chain per Catalog #370 prerequisites are operator-routable for
  next cycle)

### Path 2 (HIGH PRIORITY; PARALLEL DISJOINT; $0 MLX-LOCAL):
**Sister cascade Hinton + 600-pair batch** (V2 + V4 + VQ; 3 more
canonical equation #1 anchors at ~150-200s MLX-LOCAL each = ~10-13 min
total wall-clock at $0).

- **Tests cargo-culted assumption**: row 5 (SELECTOR-V3 optimal vs sister
  cascade variants for Hinton-extension)
- **Predicted cost**: $0 MLX-LOCAL on M5 Max; sister-disjoint per Catalog
  #314/#340 to THIS symposium subagent's scope
- **Compounds canonical equation #1** anchor count 8 → 11 (PACT-NeRV
  cascade Hinton-extension substrate-class expansion per the 11th
  INDIVIDUALLY-FRACTAL standing directive)
- **Triggers Catalog #371 auto-recalibration** since ≥3 NEW anchors
  threshold satisfied

### Path 3 (MEDIUM PRIORITY; RESEARCH-DEEP; $0 MLX-LOCAL):
**Extended-epoch training at ep 5000-10000 MLX-LOCAL** on SELECTOR-V3 +
Hinton-distilled + 600-pair to test the "min-at-last-epoch implies further
convergence" sub-claim.

- **Tests cargo-culted assumption**: row 4 sub-claim (still-descending at
  ep 1999 implies further convergence)
- **Predicted cost**: $0 MLX-LOCAL; ~6-13 min wall-clock per the 159.1s/
  2000ep linear extrapolation (likely sub-linear due to teacher cache
  amortization)
- **Disambiguates training-saturation vs further-descent** at MLX-research-
  signal axis BEFORE paired-CUDA dispatch on stale-saturation candidate

### Path 4 (DIAGNOSTIC; $0 MLX-LOCAL):
**Scorer-axis component sweep** on SELECTOR-V3 + Hinton-distilled with
per-axis ablation: seg-only (distillation_weight=1.0 + pose-MSE=0) /
pose-only (pose-MSE=1.0 + distillation_weight=0) / combined (canonical
defaults).

- **Tests cargo-culted assumption**: combined scorer-axis is optimal vs
  per-axis ablation; informs Path 1 result interpretation
- **Predicted cost**: $0 MLX-LOCAL; 3 × 159.1s = ~8 min wall-clock for full
  ablation matrix
- **Disambiguates seg-axis-dominated vs pose-axis-dominated vs combined-
  optimal** scorer-binding pattern at MLX-research-signal axis

## 7. Catalog #324 post-training Tier-C validation discipline

`predicted_band_validation_status: pending_post_training`

The current empirical anchor (159.1s MLX-LOCAL training of SELECTOR-V3 +
Hinton-distilled + 600-pair to combined loss 3.40) is at the MLX-research-
signal axis per Catalog #1/#127/#192/#317/#341 — NOT the canonical
post-training Tier-C measurement required for `validated_post_training`
status per Catalog #324.

**Reactivation criterion for `validated_post_training` status**: paired
Linux x86_64 + NVIDIA dispatch per Path 1 above produces the canonical
contest-axis archive; post-training Tier-C density measurement on the
landed archive sha256 via `tools/mdl_scorer_conditional_ablation.py
--tier c` flips the field from `pending_post_training` to `post_training_
2000ep_paired_cpu_cuda_<utc>` with the landed archive sha256 inline-
referenced.

Per Catalog #324 FORBIDDEN_PATTERN guardrail: the just-landed empirical
memo correctly does NOT declare a `predicted_band: [lo, hi]` field
(avoiding the phantom-random-init Tier-C density bug class anchored by
the 2026-05-17 C6 IBPS 22× miss). The canonical predicted band derivation
is DEFERRED to the post-Path-1 paired-CUDA empirical anchor.

## 8. Predicted ΔS band (Dykstra-feasibility per Catalog #296)

THIS symposium is a PRE-PAID-DISPATCH RESEARCH-SIGNAL SYMPOSIUM, NOT a
contest-axis score prediction. Per CLAUDE.md "Forbidden symposium-band-
prediction-without-Dykstra-feasibility-check" + Catalog #296 sister
discipline: every predicted ΔS band MUST cite the Dykstra-feasibility
intersection check OR first-principles bound OR probe-disambiguator path.

The predicted ΔS band for the sub-0.18 candidate IS the operator's
standing hypothesis (combination of three compounding validated ideas →
sub-0.18); the canonical Dykstra-feasibility derivation requires:

1. Per-axis decomposition at contest-CUDA via Path 1 paired-CUDA dispatch
   (Catalog #356 AxisDecomposition primitive at the consumer surface)
2. Sister cascade apples-to-apples disambiguation via Path 2 (V2 + V4 + VQ
   at MLX-LOCAL)
3. Per-axis ablation matrix via Path 4 (seg-only / pose-only / combined)

Per CLAUDE.md "Meta-Lagrangian/Pareto solver — NON-NEGOTIABLE" + Catalog
#322 sister composition_alpha discipline: the canonical composition_alpha
empirical anchor requires paired-CUDA + the canonical posterior at
`.omx/state/substrate_composition_matrix.json`. Until Path 1 lands, the
predicted ΔS band remains an OPEN EMPIRICAL QUESTION; the symposium
explicitly NAMES this DEFERRED-pending-empirical per CLAUDE.md "Forbidden
premature KILL".

## 9. 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map** = ACTIVE — every loss component (recon + KL
  T=2.0 + pose-MSE) IS a per-axis sensitivity surface per Catalog #356
  AxisDecomposition primitive; downstream consumers route through
  `tac.sensitivity_map.*` via the canonical TrainingArtifact's per-epoch
  metrics surface.
- **Hook #2 Pareto constraint** = ACTIVE — KL T=2.0 + pose-MSE composition
  IS the canonical Pareto polytope axis (seg / pose); MLX-LOCAL evidence
  feeds the polytope consumer at the autopilot ranker surface via canonical
  equation registry per Catalog #344 + auto-discovery per Catalog #335.
- **Hook #3 bit-allocator** = N/A — substrate-class extension is NOT a
  bit-allocator change at this layer (SELECTOR-V3's per-pair Rice-Golomb
  coder operates at archive-encode time; orthogonal to scorer distillation
  at MLX forward path).
- **Hook #4 cathedral autopilot dispatch** = ACTIVE — canonical equation
  #1 consumer at `tac.cathedral_consumers.canonical_equation_lookup_consumer`
  auto-discovers the 8th anchor (the just-landed SELECTOR-V3 substrate-
  class extension) via Catalog #335 + Catalog #344; sister consumers
  consume the per-substrate symposium verdict via the canonical
  `.omx/state/council_deliberation_posterior.jsonl` posterior surface.
- **Hook #5 continual-learning posterior** = ACTIVE PRIMARY — canonical
  posterior anchor appended via `tac.council_continual_learning.append_council_anchor`
  with `deferred_substrate_id=pact_nerv_selector_v3_hinton_distill_600pair_mlx_local`
  (THIS symposium memo IS the canonical posterior anchor for the per-
  substrate symposium surface per Catalog #325 + Catalog #300 v2
  frontmatter).
- **Hook #6 probe-disambiguator** = ACTIVE — the canonical Path 1 + Path 2
  + Path 4 reactivation paths ARE the canonical probe-disambiguators
  between (a) sub-0.18 candidate confirmed at contest-CUDA / (b)
  per-axis-dominated descent pattern / (c) sister cascade optimality.
  Each path produces a falsifiable empirical anchor per CLAUDE.md
  "Forbidden premature KILL without research exhaustion" non-negotiable.

## 10. Mission contribution per Catalog #300

`frontier_breaking_enabler` — this symposium UNBLOCKS the canonical
pre-paid-dispatch research-signal pattern for the next-step Path 1
paired-CUDA dispatch per Catalog #246. The scorer-bound finite convergence
verdict (at MLX-research-signal axis) IS the structural proof that the
canonical Hinton-distilled scorer surrogate × SELECTOR-V3 substrate-class
extension × 600-pair scale combination is empirically valid at the
surrogate-binding axis; the sub-0.18 contest-axis HYPOTHESIS remains OPEN
pending paired-CUDA evidence per CLAUDE.md "MPS auth eval is NOISE" +
"Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
HARDWARE" non-negotiables.

The symposium PROCEED_WITH_REVISIONS verdict (with 3 binding revisions
addressing Contrarian + Assumption-Adversary + Yousfi dissent) authorizes
operator-attended Path 1 dispatch eligibility per Catalog #325 acceptance
cascade (verdict in {PROCEED, PROCEED_WITH_REVISIONS}).

## 11. Sister-disjoint coordination per Catalog #314/#340

This symposium subagent's scope: NEW symposium memo + canonical posterior
anchor only (per parent prompt CONSTRAINTS). Substrate packages NOT
touched; recipe authoring + canonical 4-verdict chain (Catalog #370
Phase 4-7 prerequisites for Path 1 dispatch) are operator-routable for
next cycle.

Sister Slot 2 (Hinton cascade batch V2 + V4 + VQ per op-routable #2) is
disjoint scope and may be dispatched in parallel per the operator's
parallel-dispatch directive + Catalog #340 sister-checkpoint guard.

## 12. AUTOMATED + COMPOUNDING + OPTIMAL discipline (7th META standing directive)

- **AUTOMATED**: per-substrate symposium = 1 canonical memo + 1 canonical
  posterior anchor append; zero manual editing of substrate packages;
  canonical Catalog #325 6-step contract structurally enforced; canonical
  roster validated complete=True via `tac.canonical_council_roster.validate_council_dispatch_roster`
  programmatic check; canonical 4-verdict chain prerequisites (Catalog #370
  Phase 4-7) surfaced as operator-routables.
- **COMPOUNDING**: canonical equation #1 anchor count 8 → 8 (unchanged at
  this symposium; the empirical anchor was added in the just-landed sister
  T1 memo at commit ab650cc78); canonical council deliberation posterior
  anchor count + 1 (THIS symposium); op-routable #2 sister cascade Hinton
  + 600-pair batch compounds canonical equation #1 anchor count 8 → 11
  on next cycle at $0 each.
- **OPTIMAL**: zero forks of canonical primitives; ONE symposium memo; ONE
  canonical posterior anchor append; all sister-disjoint per Catalog
  #314/#340; substrate packages not touched per parent prompt CONSTRAINTS;
  recipe authoring + 4-verdict chain prerequisites operator-routable for
  next cycle.

## 13. Operator-routable TOP-1 command sheet (Path 1; for next cycle)

**Per Catalog #325 dispatch eligibility cascade** (verdict
PROCEED_WITH_REVISIONS in {PROCEED, PROCEED_WITH_REVISIONS}; canonical
roster complete=True; symposium memo within 14-day window): operator-
attended paired Linux x86_64 + NVIDIA paired CPU+CUDA dispatch on the
SELECTOR-V3 + Hinton-distilled + 600-pair archive is dispatch-eligible
per Catalog #325 acceptance cascade.

**Prerequisites for next cycle** (operator-attended per just-discovered
structural constraint about subagent-attempted recipe flips):

1. Recipe authoring per Catalog #240 recipe-vs-trainer-state consistency:
   `.omx/operator_authorize_recipes/substrate_pact_nerv_selector_v3_hinton_distill_modal_t4_paired_dispatch.yaml`
   with canonical 4-verdict chain fields per Catalog #370 (Phase 4 builder
   + Phase 5 linter + Phase 6 compliance + Phase 7 paired_auth_eval).
2. Catalog #324 post-training Tier-C validation: `predicted_band_validation_status`
   declared `pending_post_training` with reactivation criterion = post-
   training Tier-C measurement on landed archive sha256.
3. Catalog #244 canonical NVML env block in remote driver script.
4. Catalog #225 dispatch claim ledger row + Catalog #245 Modal call_id
   ledger registration.

**Canonical Path 1 invocation pattern** (for operator-attended next-cycle
dispatch):

```bash
# Once recipe + 4-verdict-chain prerequisites land per next-cycle operator
# work, the canonical operator-routable command is:
.venv/bin/python tools/operator_authorize.py \
    --recipe substrate_pact_nerv_selector_v3_hinton_distill_modal_t4_paired_dispatch \
    --paired-axis cpu+cuda
```

The dispatch routes through canonical Catalog #243 local pre-deploy harness
+ Catalog #271 codex pre-dispatch review automation + Catalog #313
predecessor probe outcomes ledger + Catalog #339/#360 silent-no-spawn
extinction + Catalog #166 Modal source-parity ledger + Catalog #245
canonical Modal call_id ledger.

## 14. Cross-references

- **Just-landed empirical T1 council memo** (the symposium's foundation):
  `.omx/research/pact_nerv_selector_v3_hinton_distill_600pair_long_mlx_landed_20260528.md`
- **600-pair pure-pixel-MSE generalization-floor evidence**:
  `.omx/research/pact_nerv_selector_v3_extended_600pair_long_mlx_landed_20260528.md`
- **Sister IA3 integration smoke (canonical pattern source)**:
  `.omx/research/hinton_distilled_scorer_surrogate_mlx_local_pact_nerv_ia3_integration_landed_20260528.md`
- **SELECTOR-V3 L1 long-run baseline**:
  `.omx/research/pact_nerv_selector_v3_l1_long_run_mlx_landed_20260528.md`
- **ULTIMATE design memo (Step 12 SELECTOR-V3 spec)**:
  `.omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md`
- **Sister per-substrate symposium pattern** (Z6-v2):
  `.omx/research/z6_v2_cargo_cult_unwind_design_20260527T053000Z.md`
- **Canonical Catalog #325 contract**: CLAUDE.md "PER-SUBSTRATE OPTIMAL
  FORM via adversarial grand council symposium — NON-NEGOTIABLE, HIGHEST
  EMPHASIS" section
- **Canonical Catalog #346 roster helper**:
  `src/tac/canonical_council_roster.py::validate_council_dispatch_roster`
- **Canonical Catalog #300 + #355 posterior**:
  `src/tac/council_continual_learning.py::append_council_anchor`
- **CLAUDE.md non-negotiables honored**:
  - "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" —
    THIS memo IS the canonical per-substrate symposium for the candidate
  - "Council hierarchy: 4-tier protocol" — T2 inner-skunkworks scope per
    in-flight engineering tradeoffs (cascade routing + sister disambiguation)
  - "Council conduct" — full sextet pact (5-of-6 quorum) + Assumption-
    Adversary 7th seat + 4 co-leads (Shannon LEAD + Dykstra/Rudin/
    Daubechies CO-LEAD) per 2026-05-19 amendment
  - "Mission alignment — non-negotiable" — `frontier_breaking_enabler`
    contribution; operator-frontier-override NOT invoked
  - "MLX portable-local-substrate authority" — `[macOS-MLX research-signal]`
    per Catalog #192/#317/#341
  - "Submission auth eval — BOTH CPU AND CUDA" — paired CPU+CUDA per Path 1
  - "Forbidden premature KILL without research exhaustion" — paradigm
    INTACT; 4 reactivation paths pinned
  - "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" — 3-way combination is
    SELECTOR-V3's OWN canonical engineering pass
  - "EMA + eval_roundtrip + MPS auth eval is NOISE" — canonical primitives
    + non-promotable MLX-research-signal axis
  - "META-ASSUMPTION ADVERSARIAL REVIEW" — Assumption-Adversary verbatim
    surfaced 5 NEW cargo-cult-pending classifications + 1 cross-reference
    to row 11 (Yousfi)
  - "Frontier scores are pointer-only" — NO hardcoded score literals
    (canonical frontier pointer per Catalog #343 + #316 remains source of
    truth)
  - "Bugs must be permanently fixed AND self-protected against" —
    structural via Catalog #325 + #300 + #346 + #292 + #305 + #303 + #294
    + #324 + #344 + #371

## 15. Symposium verdict summary

```
council_tier: T2
council_attendees: 18 (14 INNER + 4 GRAND topical)
council_quorum_met: True
canonical_roster_complete: True (per Catalog #346)
council_verdict: PROCEED_WITH_REVISIONS
council_dissent: 3 binding revisions (Contrarian + Assumption-Adversary + Yousfi)
council_assumption_adversary_verdict: 5 assumptions (3 NEWLY-NAMED CARGO-CULTED-PENDING + 2 HARD-EARNED)
council_decisions_recorded: 8 (5 op-routables + 3 binding revisions)
council_predicted_mission_contribution: frontier_breaking_enabler
predicted_band_validation_status: pending_post_training (Catalog #324)
horizon_class: frontier_pursuit
6-hook wire-in: 5 ACTIVE (hooks #1, #2, #4, #5, #6) + 1 N/A (#3 bit-allocator)
```

The symposium AUTHORIZES Catalog #325 dispatch eligibility for Path 1
paired Linux x86_64 + NVIDIA dispatch per the canonical acceptance
cascade (verdict PROCEED_WITH_REVISIONS in {PROCEED, PROCEED_WITH_REVISIONS};
roster complete=True; symposium memo within 14-day window). Operator-
attended next-cycle dispatch is gated on the operator-routable
prerequisites enumerated in §13 (recipe authoring + Catalog #370 4-verdict
chain + Catalog #324 Tier-C validation discipline + Catalog #244 NVML
env block + Catalog #225/#245 ledger registration).
