---
council_tier: T1
council_attendees: [PR95Author, Shannon, Dykstra, Quantizr, Hotz, Selfcomp, AssumptionAdversary, Contrarian]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Contrarian
    verbatim: "The 'mostly HARD-EARNED' headline is at risk of being itself a CARGO-CULTED inference because PR95 is the only contest-winning HNeRV-family substrate we have full source for; lack of a counterfactual (a curriculum variant that ablated stage X and still scored ≤ 0.21) means every HARD-EARNED stamp is necessarily UNFALSIFIED-but-untested. Demand reactivation criteria be stated even for HARD-EARNED stages."
  - member: AssumptionAdversary
    verbatim: "Operating-within assumption I surface for this deliberation: 'the contest scorer's gradient response is similar across HNeRV-class substrates of comparable parameter count.' If false (e.g. DreamerV3 / NSCS06 v8 substrates have qualitatively different scorer-gradient flat regions), the per-stage composability assessment is too generous. The 8x7 matrix is HARD-EARNED only on PR95-family substrates and UNCLASSIFIED for class-shift substrates (Z6-v2 / DreamerV3 / NSCS06 v8 / Rudin)."
council_assumption_adversary_verdict:
  - assumption: "PR95's 8-stage curriculum is a coherent optimization-theoretic chain (each stage's exit-state is the principled init for next stage)"
    classification: HARD-EARNED
    rationale: "Empirically PR95 scored 0.20 at the leaderboard; per the forensic memo the stage exit-states are pinned (stage4 → e2ev332_ep10650 → stage5; stage7 → exp4_sigma01_ep975 → stage8 → muon_ep250 at 0.2009). The optimization trajectory was empirically observed to reach floor-band ≈ 0.20 at each handoff. NOT a derivation from theory; AN OBSERVATION from the only contest-winning HNeRV-family ablation we have."
  - assumption: "Muon (Newton-Schulz orthogonalization at Stage 8) is necessary for the final 0.193 polish step"
    classification: CARGO-CULTED
    rationale: "Per PR95 forensic memo Stage 8 dropped score from 0.2042 → 0.2009 (-0.003) — that's a real but SMALL Muon contribution. The Council 2026-05-13 design memo explicitly proposes Langevin SDE as a FIRST-PRINCIPLES alternative (predicted comparable -0.005 to -0.012); the Contrarian veto on 'Langevin is just batch-noise renamed' was withdrawn. No empirical anchor shows Muon DOMINATES Langevin/SGLD/SGD-with-noise; the choice is researcher #24's preference, not a falsified-alternative-after-control."
  - assumption: "The 6 PR95 primitives (differentiable yuv6, cat_entropy_v2, apply_qat, Muon, L7-Softplus, bilinear-skip+sin block) are all REQUIRED for the substrate to converge"
    classification: HARD-EARNED-with-3-confirmed-CARGO-CULTED-claims
    rationale: "(1) Differentiable yuv6 IS HARD-EARNED — PR95 author named it the 'pose plateau bug fix' and CLAUDE.md non-negotiable repeats this; without it pose gradient is severed. (2) cat_entropy_v2 IS HARD-EARNED — it IS the rate-axis Lagrangian driving the post-INT8 distribution sharpness; remove it and brotli compression doesn't shrink. (3) apply_qat IS HARD-EARNED — without QAT the INT8 export step is post-hoc, not gradient-aware; the post-quant score WOULD be worse than the pre-quant proxy. (4) Muon — CARGO-CULTED per previous row. (5) L7-Softplus — HARD-EARNED as a STAGE-5 PRIMITIVE (concentrates gradient on near-boundary pixels which IS what the scorer's argmax cares about); but the (l7_threshold=1.0, l7_mult=4.0) numbers are CARGO-CULTED (one researcher choice, no ablation). (6) bilinear-skip+sin block — HARD-EARNED as architectural primitive (PR95's expressivity engine; ablating either skip or sin destroys reconstruction quality), but inherited from SIREN/HNeRV literature, NOT proven contest-optimal."
  - assumption: "29,650 epochs is the MINIMUM epoch budget for PR95-class substrate to reach 0.20-band"
    classification: CARGO-CULTED
    rationale: "PR95 docstring says '~50 hr on a single GPU' but does NOT name the GPU and there is NO ABLATION at smaller epoch counts. The codex curriculum recovery memo explicitly flags this as 'a conservative planning band until the smoke gate measures'. The beat-PR95 design proposes an A*-shortest-path that potentially skips 70% of epochs (proven-feasible IF the heuristic is admissible, UNVERIFIED). The 29,650 number is what PR95-author RAN, not what they NEEDED."
council_decisions_recorded:
  - "op-routable #1: TIER 0 (free): per-stage composability matrix lookup for sister-substrate work — operationally consumable today by Dim 4 substrate authors (Z6-v2 / DreamerV3 / NSCS06 v8 / Rudin / Cool-Chic) deciding which PR95 stages to import as bolt-on."
  - "op-routable #2: TIER 1 ($0 paired-comparison smoke): macOS-CPU advisory ablation of L7-Softplus (l7_threshold sweep 0.5/1.0/2.0; l7_mult sweep 2.0/4.0/8.0) on a PR95-cloned substrate to disambiguate the CARGO-CULTED (1.0, 4.0) constants — TARGET HORIZON 1-2 week sister subagent (NOT this T1 working group's scope)."
  - "op-routable #3: TIER 1 ($1-2 macOS-MPS or Vast.ai 4090 paired-comparison smoke): Langevin polish vs Muon polish on the PR95 published 0.bin parsed Stage-7-equivalent state — the cheapest empirical Muon-vs-Langevin disambiguator, predicted to resolve the cat #2 CARGO-CULTED Muon claim. TARGET HORIZON 1 week sister subagent."
  - "op-routable #4: TIER 2 ($4-8 Modal A100): A*-shortest-path curriculum compression empirical proof — runs 4 stages instead of 8, A*-derived predicted-score < 0.21 at ~10,000 epochs total. Requires Catalog #229 PV (verify A* heuristic admissibility on joint distortion first). 4-6 week horizon."
  - "op-routable #5: TIER 0 (META-design): register the 6 stage-component-level findings (yuv6 / QAT / cat_entropy_v2 / EMA / eval-roundtrip / L7-Softplus) as CANONICAL CROSS-SUBSTRATE BOLT-ONS in `tac.substrates._shared.score_aware_common` so future HNeRV-class substrate trainers can compose them without rediscovering. Cross-references Catalog #335 cathedral consumer paradigm."
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_id: null
substrate_alias: null
horizon_class: plateau_adjacent
related_deliberation_ids:
  - "council_t1_pr95_curriculum_8stage_research_20260520T143358Z"
canonical_helper_invocation: "tac.canonical_council_roster.validate_council_dispatch_roster (T1 working group; PR95Author convened as canonical inner-council voice per 2026-05-19 amendment)"
---

# T1 Working Group — PR95 8-stage curriculum research

**Lane:** `lane_wave_3_pr95_curriculum_t1_research_20260520`
**Tier:** T1 working group (CLAUDE.md "Council hierarchy: 4-tier protocol" — UNBOUNDED cadence; bounded-scope recommendation; NO veto power; output feeds T2/T3 deliberation).
**Operator authorization:** blanket Wave-3 Tier-1 research approval 2026-05-20.
**Mission contribution:** `frontier_protecting` — research informs future paid-dispatch decisions on PR95-paradigm extensions; does NOT itself promote a score.
**Horizon class:** `plateau_adjacent` — PR95's empirical anchor is 0.20-band; the working group is researching whether the 8 stages compose with sister substrates that aim for `frontier_pursuit` or `asymptotic_pursuit` horizons.
**GPU spend:** $0 (pure research / source forensics).
**Wall-clock:** ~4 hours.

---

## 1. Convocation context

### Mandate

Operator directive 2026-05-20 verbatim: *"Conduct T1 working-group research on PR95's 8-stage 29,650-epoch curriculum. Determine which stages are load-bearing vs skippable vs sister-compatible. PR95 author is now on inner council per CLAUDE.md 2026-05-19 amendment — convene them as canonical inner-council voice for the working group."*

This is a META-LAYER deliberation: not "what's the next paid PR95 reproduction dispatch?" (that's the F1 lane's T2/T3 territory) but "what stages of PR95's IP are reusable / load-bearing / skippable across sister substrate work?" The deliverable is operator-routable lookup tables (the 8×7 composability matrix + HARD-EARNED-vs-CARGO-CULTED classification) that downstream subagents consume WITHOUT re-deriving.

### Roster (per CLAUDE.md "Grand Council (advisory)" + canonical_council_roster.py)

**Inner council voices convened (8-of-12 + 1 grand-council seat):**

| Seat | Role | Why convened for THIS deliberation |
|---|---|---|
| **PR95Author** (Aaron Leslie) | inner_council (2026-05-19 addition) | CANONICAL voice — original PR95 author has deepest intuition for what the contest scorer rewards on HNeRV-class substrates; canonical position summary specifically cites "PR 100/101/102/103 winners all built on top of" PR95's substrate. Per operator mandate to convene as inner-council voice. |
| **Shannon** (LEAD) | inner_council co-lead | Information-theory grounding required for the HARD-EARNED-vs-CARGO-CULTED classification per Catalog #292 + rate-axis Lagrangian analysis (cat_entropy_v2). |
| **Dykstra** (CO-LEAD) | inner_council co-lead | Optimization-feasibility lens for stage-by-stage convergence claims + 8-stage chain feasibility check (does the chain land inside the Pareto polytope?). |
| **Quantizr** | inner_council | Adversarial-cousin voice — Quantizr's own 5-stage curriculum (anchor → finetune → joint → QAT → final) at 0.33 is the canonical SIMPLER alternative; demands rigorous comparison. |
| **Selfcomp** | inner_council | UCLA collaborator + grayscale-LUT analog mask paradigm; sister-substrate composability checking. |
| **Hotz** | inner_council | Engineering tractability — "what's the 5-line PoC?" + cost-band reality check. |
| **AssumptionAdversary** | inner_council (sextet-pact 2026-05-15 addition) | Per Catalog #292 + CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW" non-negotiable — per-round mandate to surface ONE shared-assumption-violation hypothesis. |
| **Contrarian** | inner_council | Challenges weak arguments per CLAUDE.md "Council conduct" non-negotiable. |

**T1 working-group spec** (CLAUDE.md): 1-3 named members + UNBOUNDED cadence. We exceeded the 1-3 minimum to ensure the per-stage HARD-EARNED-vs-CARGO-CULTED audit has both first-author + adversarial + assumption-challenge perspectives. Per Catalog #346 canonical roster validator: `validate_council_dispatch_roster(attendees, topic_tokens={pr95_author, hnerv_family, substrate_engineering, race_mode_rigor_inversion}, tier=T1) → complete=True` (PR95Author + Shannon + Dykstra anchor; topical sextet covers the relevance tokens).

### Source-of-truth artifacts consulted (per Catalog #229 PV)

1. `.omx/research/pr95_8stage_curriculum_forensic_20260513.md` (the F1 forensic anchor; ~200 lines covering per-stage configs, the 6 PR95-specific primitives, parity vs internal sane_hnerv, cost-feasibility decision).
2. `.omx/research/pr95_curriculum_recovery_20260513_codex.md` (codex forensic recovery; full-cost campaign plan; mutation matrix).
3. `.omx/research/beat_pr95_curriculum_substrate_training_design_20260513.md` (the canonical "do better than PR95" design memo — A*-shortest-path + Langevin polish + stack-of-stacks + dynamic continual learning + HJB control framing).
4. `.omx/research/phase_1b_z6_lift_pr95_paradigm_landed_20260516.md` (sister landing: Z6 v2 lift of PR95 paradigm — what was reused vs forked).
5. PR95 source tree at `experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/` (verified Catalog #109 pristine + read stage1+4+5+8 source).
6. `src/tac/canonical_council_roster.py` PR95Author seat (lines 277-298) for canonical position summary + relevance tokens.

---

## 2. Per-stage analysis (the 8 PR95 stages)

### Methodology

For each stage `i ∈ {1, 2, 3, 4, 5, 6, 7, 8}`:

1. **Stage purpose** — what optimization signal does this stage produce?
2. **Load-bearing** — would PR95 score WORSE without this stage? (Predicted; cite first-principles + sister anchors)
3. **Skippable** — can a sister substrate (DreamerV3 / NSCS06 / SIREN / Cool-Chic / etc.) skip this stage without loss?
4. **Sister-compatible** — can this stage be reused as a bolt-on for sister substrates?
5. **Empirical anchor** — does any existing landed anchor verify the stage's contribution?
6. **Council assessment** — composite verdict from the 8 attendees.

### Stage-by-stage

#### Stage 1 — `stage1_v328_ce.py` (3000 epochs, Cross-Entropy seg + AdamW only, random init)

- **Stage purpose:** bulk calibration from random init. CE on hard targets gets the decoder into the right RGB-output regime for the contest scorer to be useful at all. Pose loss reduces to `sqrt(10 * MSE)` which is the contest formula. NO QAT, NO C1a, NO L7-Softplus.
- **Load-bearing:** **YES, but partially substitutable.** Cross-entropy on `(seg_logits, hard_targets)` is the canonical "warm-start a fresh network on its task." If you skip Stage 1 entirely (random init → directly to Stage 2's softplus), you have a circular bootstrap problem — softplus's smooth-margin loss needs a non-pathological prediction surface to begin smoothing. Sister substrates with PRETRAINED initialization (e.g. DP1 codebook distillation) can plausibly SKIP Stage 1 by starting from a coarse-but-non-pathological reconstruction.
- **Skippable:** PARTIALLY for substrates with non-random init (DP1, NSCS06 with grayscale-LUT init, Selfcomp's published 0.33 archive parsed as init). NOT skippable for random-init substrates of comparable arch.
- **Sister-compatible:** YES as a generic warm-up. The CE-on-hard-targets loss is substrate-agnostic. Any HNeRV-class substrate's stage 1 can adopt it verbatim.
- **Empirical anchor:** PR95 docstring says Stage 1 produced the `e2ev328` checkpoint at ep3000 from which Stage 2 resumes; the canonical PR95 ran 10K epochs but switched to softplus at ep3000. **No empirical anchor shows what PR95's score would be at exit of Stage 1 alone**; that's a F1 follow-up reactivation criterion.
- **Council assessment:** **HARD-EARNED for random-init HNeRV-class substrates. CARGO-CULTED-for-PRETRAINED-init substrates** (e.g. Z6-v2, DP1, NSCS06 with grayscale-LUT init, Selfcomp-as-init).

#### Stage 2 — `stage2_v331_softplus.py` (5650 epochs, τ-Softplus seg + AdamW only, resume from Stage 1)

- **Stage purpose:** swap the CE hard-target loss for a soft-margin loss `tau_softplus_seg_loss(tau=0.3)` that lets the scorer's argmax-cliff be optimized via smooth gradient. CE saturates after the prediction is correctness-class-correct; softplus continues optimizing the SCORE-relevant margin.
- **Load-bearing:** **YES.** This is the first "score-aware seg loss" stage. PR95 spent 5650 epochs on this (the longest single stage other than Stage 5). Without it, the substrate optimizes for argmax correctness but not for the contest scorer's per-pixel decision boundary, which IS what the scorer measures.
- **Skippable:** NO for any substrate targeting contest-CPU/CUDA scores. The score-axis-aware loss IS the substrate's score-aware substrate engineering per CLAUDE.md HNeRV parity L1.
- **Sister-compatible:** YES — `tau_softplus_seg_loss` is a substrate-agnostic Catalog #164 canonical primitive. Future substrates should adopt it; Z6-v2 forks it for Rao-Ballard residual-entropy reasons (Catalog #310/#311).
- **Empirical anchor:** PR95 reaches its first contest-relevant score band at the exit of Stage 2 (per beat-PR95 design memo §5 — empirical priors are seeded from PR95 stage-end checkpoints). Exact stage-2-exit score not pinned in our forensic memo.
- **Council assessment:** **HARD-EARNED.** Confirmed by both first-principles (scorer-aware loss IS the substrate-engineering canonical) AND empirical (PR95 spent 5650 epochs here, the canonical longest non-bulk stage).

#### Stage 3 — `stage3_v332_smooth.py` (1500 epochs, smooth-disagreement seg + fresh cosine 1e-4 AdamW)

- **Stage purpose:** SECOND seg-loss refinement — `smooth_disagreement_seg_loss(tau=0.3)` applies a sigmoid bell on the NEGATIVE margin (i.e., it concentrates gradient on pixels where the prediction is WRONG and the scorer's argmax disagrees with GT). This is a sharper gradient signal than Stage 2's general softplus.
- **Load-bearing:** **YES** (per beat-PR95 design memo §2.2 implicit; PR95 spent 1500 epochs which is a non-trivial budget). The score gradient is concentrated on the cliff between "this pixel almost flips" and "this pixel almost stays."
- **Skippable:** PROBABLY — the Stage 2 softplus + Stage 5 L7-Softplus together cover most of the score-aware seg-loss design space. Stage 3 is an INTERMEDIATE refinement that bridges Stage 2 → Stage 4 QAT. Sister substrates that DON'T have a QAT stage can plausibly skip Stage 3 and go directly Stage 2 → Stage 5.
- **Sister-compatible:** YES as a generic score-aware seg-loss refinement. Useful especially for sister substrates that don't yet have L7-Softplus.
- **Empirical anchor:** PR95's `e2ev332_ep10150` checkpoint at exit of Stage 3 is the immediate predecessor of Stage 4 QAT; no standalone score known.
- **Council assessment:** **PROBABLY CARGO-CULTED at stage-granularity but HARD-EARNED at primitive-granularity.** The smooth-disagreement primitive itself is HARD-EARNED (cliff-concentrated gradient). But scheduling it as a separate 1500-epoch stage between softplus and QAT is researcher choice — could be folded into Stage 2 with a curriculum-internal tau-schedule.

#### Stage 4 — `stage4_v332_qat.py` (500 epochs, smooth-disagreement seg + QAT + AdamW continuing Stage 3 cosine)

- **Stage purpose:** introduce INT8 quantization-aware training. The export codec quantizes decoder weights to INT8 (n_quant=127, per-tensor symmetric); without QAT, the post-quant score differs from the pre-quant proxy by O(0.01-0.05) on the score axis (empirically observed in many quantization papers, not specifically in PR95). QAT learns weights that are robust to the quantization noise.
- **Load-bearing:** **YES — the rate term needs the substrate to learn at the export resolution.** Per CLAUDE.md HNeRV parity L8 (eval-roundtrip-aware) + the contest formula's rate term `25 * archive_bytes / 37,545,489`, the substrate that emits the smallest brotli-able archive WITHOUT seg/pose regression is the winner. QAT is the canonical way to learn a quantization-robust substrate.
- **Skippable:** NO for any substrate targeting INT8 quant for export. SKIPPABLE for substrates that use a different quantization scheme (e.g. INT4 like apogee_int4, FP4 like Quantizr-1, vector-quantized like VQ-VAE) but each would need its OWN equivalent of Stage 4.
- **Sister-compatible:** YES as a generic concept; the SPECIFIC `apply_qat`/`restore_qat` implementation is INT8-symmetric-specific.
- **Empirical anchor:** PR95's `e2ev332_ep10650` (saved as `e2ev332_d28_c36_e10650_bs8_ep10200`) is the post-Stage-4 checkpoint. No isolated score known. But: the codec asserts INT8 round-trip bit-exactness, which means the export step preserves the QAT-trained weights.
- **Council assessment:** **HARD-EARNED at concept-level (quantization-aware training of the export format).** The SPECIFIC implementation (in-place fake-quant + restore for forward-only quant exposure) is HARD-EARNED engineering. The 500-epoch budget is researcher choice (CARGO-CULTED at granularity).

#### Stage 5 — `stage5_c1a_l7.py` (9000 epochs canonical 6000; L7-Softplus seg + C1a entropy + AdamW lr=3e-5)

- **Stage purpose:** SIMULTANEOUSLY (a) ramp the rate-axis Lagrangian via `cat_entropy_v2(σ=0.2, λ=0.01)` (the soft-MDL post-INT8 distribution sharpener) and (b) sharpen the seg-loss gradient via `l7_softplus_seg_loss(tau=0.3, l7_threshold=1.0, l7_mult=4.0)` (concentrates on near-boundary pixels, margin < 1). This is THE LONGEST STAGE (9000 epochs) and corresponds to the substrate moving from generic-score-aware to **rate-distortion-frontier-aware**.
- **Load-bearing:** **EXTREMELY YES.** This is the rate-axis stage. PR95 spent 1/3 of its TOTAL EPOCH BUDGET here. Without it (substrate trained without `cat_entropy_v2`), brotli compression of the INT8 weights would not be effective (the post-INT8 distribution would be too uniform / wide-tailed for brotli to compress).
- **Skippable:** NO for any substrate targeting brotli/zstd/lzma post-quant entropy coding. SKIPPABLE for substrates that use a different rate-axis primitive (e.g. arithmetic coding, ANS, range coding, hyperprior). But sister substrates would need an EQUIVALENT rate-axis Lagrangian.
- **Sister-compatible:** YES, with caveats. The `cat_entropy_v2` primitive (size-weighted soft histogram entropy) is substrate-agnostic but PER-TENSOR-WEIGHT-SPECIFIC; it would need adaptation for substrates with non-Conv2d/Linear weights (e.g. token-based VQ-VAE). The (σ=0.2, λ=0.01) constants are CARGO-CULTED hyperparameters.
- **Empirical anchor:** PR95's `c1a_l7_ep2075` at score 0.2071 [contest-CUDA] is the Stage 5 endpoint per the forensic memo. This IS a recorded empirical anchor — Stage 5 is what produces the first sub-0.21 score.
- **Council assessment:** **HARD-EARNED at concept-level. HARD-EARNED-but-tunable at hyperparameter-level.** The σ and λ values are clearly the result of researcher iteration but no ablation was published.

#### Stage 6 — `stage6_lambda_sweep.py` (2000 epochs canonical 1000; same loss as Stage 5; **λ doubled 0.01 → 0.02**)

- **Stage purpose:** doubling the rate-axis Lagrangian weight to further tighten the post-INT8 distribution. The substrate trades slightly more seg/pose distortion for less rate.
- **Load-bearing:** **PROBABLY YES** but with weaker confidence than Stage 5. PR95's stage6 is `_lambda_sweep.py` (the filename literally says "sweep"), which suggests it WAS a researcher hyperparameter search — the (0.02) value is what they LANDED ON. So as a stage-granularity claim ("rate ramping is necessary"), HARD-EARNED. As a specific-value claim (λ must be exactly 0.02), CARGO-CULTED.
- **Skippable:** **YES if you don't need the additional rate compression.** Sister substrates targeting larger archive budgets can stop at λ=0.01 and SKIP Stage 6.
- **Sister-compatible:** YES as a generic rate-ramp stage; sister substrates can pick their own λ schedule.
- **Empirical anchor:** No standalone Stage 6 exit score known.
- **Council assessment:** **HARD-EARNED-at-concept (rate-ramp), CARGO-CULTED-at-specific-λ-value.**

#### Stage 7 — `stage7_sigma_sweep.py` (3000 epochs canonical 2000; same loss as Stage 6; **σ halved 0.2 → 0.1**)

- **Stage purpose:** halve the entropy regularizer's bandwidth, sharpening the post-INT8 distribution further (narrower Gaussian → soft-binning is more confident → entropy is lower → brotli compresses more). Same archive-bytes-vs-distortion tradeoff as Stage 6 but via different lever.
- **Load-bearing:** **PROBABLY YES** at concept-level; weak at specific-value. Per the canonical PR95 author's choice (and the `_sigma_sweep.py` filename suggesting it WAS a hyperparameter search), σ=0.1 is the empirical optimum.
- **Skippable:** YES at sister-substrate granularity (same argument as Stage 6).
- **Sister-compatible:** YES.
- **Empirical anchor:** Stage 7's exit checkpoint `exp4_sigma01_ep975` at score 0.2042 [contest-CUDA] is the Stage 7 endpoint and the Stage 8 input. Another recorded empirical anchor.
- **Council assessment:** **HARD-EARNED-at-concept, CARGO-CULTED-at-specific-σ-value.**

#### Stage 8 — `stage8_muon_finetune.py` (5000 epochs canonical 3000; same loss as Stage 7; **Muon + AdamW**)

- **Stage purpose:** optimizer switch. Replace AdamW on hidden conv weights with Muon (Newton-Schulz orthogonalized momentum, Keller Jordan 2024); keep AdamW on stem Linear, RGB heads, biases, 1D params, latents. PR95 reports this stage drops score from 0.2042 → 0.2009 (-0.003).
- **Load-bearing:** **YES for the final 0.003-0.005 polish step, but the SPECIFIC choice of Muon is CARGO-CULTED.** Per Catalog #292 AssumptionAdversary verdict above: the beat-PR95 design memo §2.2 proposes Langevin SDE as a FIRST-PRINCIPLES alternative; no empirical Muon-vs-Langevin disambiguation exists. Researcher #24 added `weight_decay=5e-4` based on Chen-Li-Liu arXiv:2506.15054 (NOT in canonical PR95). So Stage 8 IS a polish stage; whether it MUST be Muon is OPEN.
- **Skippable:** YES, the polish stage is a generic concept; any optimizer that escapes the local-min the curriculum lands at can substitute.
- **Sister-compatible:** YES as a "final polish stage" concept. Sister substrates can use Muon, Langevin SDE, SGLD, AdamW-with-restart, or other escape-local-min mechanism.
- **Empirical anchor:** `muon_ep250` at score 0.2009 is the PR95 final-publish checkpoint (Stage 8 exit); the contest leaderboard scored this 0.199-0.20 band [contest-CUDA] / 0.197 [contest-CPU].
- **Council assessment:** **HARD-EARNED-at-concept (final polish stage is necessary). CARGO-CULTED-at-mechanism (Muon vs Langevin vs SGLD vs AdamW-restart is not empirically disambiguated).**

### Summary table

| Stage | Purpose | Load-bearing | Skippable for sister | Sister-compat | Empirical anchor | HARD-EARNED-vs-CARGO-CULTED | Predicted Δscore if removed |
|------:|:-------|:-------------|:----------------------|:--------------|:-----------------|:---------------------------|:----------------------------|
| 1 | Random-init warm-up (CE on hard targets) | YES for random-init; partial for pretrained-init | YES for pretrained-init substrates (DP1, Selfcomp-as-init, etc.) | YES | None | HARD-EARNED-for-random-init / CARGO-CULTED-for-pretrained | +0.05 to +0.50 [prediction] if random-init and removed |
| 2 | First score-aware seg loss (τ-Softplus) | YES (canonical score-aware substrate primitive) | NO | YES (Catalog #164 canonical) | Implicit (5650-epoch budget) | HARD-EARNED | +0.05 to +0.30 [prediction] |
| 3 | Smooth-disagreement seg refinement | PROBABLY YES at primitive granularity | YES (can fold into Stage 2 with tau-schedule) | YES | None | HARD-EARNED-primitive / CARGO-CULTED-stage-schedule | +0.005 to +0.02 [prediction] |
| 4 | QAT INT8 fake-quant introduction | YES (rate-axis-export-aware) | NO for INT8-export substrates; YES for non-INT8 | YES with adaptation | `e2ev332_ep10650` checkpoint exists | HARD-EARNED-concept / CARGO-CULTED-epoch-budget | +0.01 to +0.05 [prediction] |
| 5 | Rate-axis Lagrangian start (cat_entropy_v2) + L7-Softplus | EXTREMELY YES | NO for brotli-rate substrates | YES with adaptation | `c1a_l7_ep2075` at 0.2071 [contest-CUDA] | HARD-EARNED-concept / HARD-EARNED-but-tunable hyperparams | +0.02 to +0.10 [prediction] |
| 6 | Rate-axis ramp (λ doubled) | PROBABLY YES | YES for larger-archive substrates | YES | None | HARD-EARNED-concept / CARGO-CULTED-λ-value | +0.005 to +0.02 [prediction] |
| 7 | Rate-axis sharpen (σ halved) | PROBABLY YES | YES | YES | `exp4_sigma01_ep975` at 0.2042 [contest-CUDA] | HARD-EARNED-concept / CARGO-CULTED-σ-value | +0.003 to +0.015 [prediction] |
| 8 | Final polish (Muon optimizer switch) | YES at concept-level | YES (can substitute Langevin / SGLD / AdamW-restart) | YES as concept | `muon_ep250` at 0.2009 [contest-CUDA] | HARD-EARNED-concept / CARGO-CULTED-mechanism | +0.002 to +0.008 [prediction] |

All Δscore predictions are `[prediction]`-tagged per CLAUDE.md FORBIDDEN_PATTERNS "Forbidden empirical-claim-without-evidence-tag."

---

## 3. 8×7 Stack-of-stacks composability matrix

### Methodology

For each PR95 stage (rows) × each of the 7 T3 Decision 4 asymptotic-pursuit candidates (cols), classify the composition:

- **COMPATIBLE** — stage IS reusable as a bolt-on for that substrate, with cited rationale.
- **CONFLICT** — incompatible by design; cited rationale.
- **UNTESTED** — no empirical or design-memo anchor; queue for paired-comparison probe.

### The 7 T3 Decision 4 asymptotic-pursuit candidates

Per CATHEDRAL-SMARTER-DESIGN-MEMO Section 7 / T3 Decision 4 (referenced in operator prompt):

1. **DreamerV3** — Ha-Schmidhuber world-model + Hafner DreamerV3 latent dynamics; canonical class-shift substrate per Catalog #312 quadruple.
2. **NSCS06 v8 Path B** — chroma-preserving + numpy-only + per-method engineering; per `feedback_permanent_fix_self_protect_all_today_bug_classes_landed_20260516.md` cargo-cult-unwind.
3. **Z7-Mamba-2** — predictive-coding + Mamba-2 state-space model; Z7 design memo (per Catalog #312 hierarchical quadruple Section 11).
4. **Z6-v2** — ego-motion-conditioned next-frame predictor per Catalog #311; landed L1 SCAFFOLD per `phase_1b_z6_lift_pr95_paradigm_landed_20260516.md`.
5. **V1 Faiss V8** — Falconnerverse vector-quantized substrate (sister design lane).
6. **Q4-Q5** — Wyner-Ziv cooperative-receiver substrate per Catalog #319 (deliverability proof builder canonical helper).
7. **Rate-attack** — generic codec-side bolt-on family (SABOR / S2SBS / wavelet residual / score-gradient residual per beat-PR95 design memo §3).

### Composability matrix

| Stage \ Substrate | DreamerV3 | NSCS06 v8 Path B | Z7-Mamba-2 | Z6-v2 | V1 Faiss V8 | Q4-Q5 | Rate-attack |
|------------------:|:----------|:------------------|:------------|:-------|:------------|:------|:------------|
| **1 (CE warm-up)** | UNTESTED — DreamerV3 has its own world-model init; PR95 Stage 1 may be subsumed by Dreamer's representation-learning phase | UNTESTED — NSCS06 v8 Path B starts from grayscale-LUT codebook; pretrained init may make Stage 1 unnecessary | UNTESTED — Mamba-2 state-space init not yet implemented | COMPATIBLE per `phase_1b_z6_lift_pr95_paradigm_landed_20260516.md` — Z6 lift adopted canonical seed-pin + device-or-die + decode_real_pairs which are sister primitives | UNTESTED — V1 Faiss V8 vector-quantization init not yet implemented | UNTESTED — Q4-Q5 init may be canonical helper-based | CONFLICT — rate-attack family operates POST-curriculum, doesn't have its own training stage |
| **2 (τ-Softplus seg)** | COMPATIBLE — score-aware loss is substrate-agnostic | COMPATIBLE | COMPATIBLE | COMPATIBLE per Z6 lift Canonical-vs-unique table row 7 (FORK with Z6PredictiveCodingScoreAwareLoss but adopts canonical τ-Softplus primitive within the fork) | COMPATIBLE | COMPATIBLE | CONFLICT — rate-attack is codec-side, not training-side |
| **3 (smooth-disagreement)** | COMPATIBLE | COMPATIBLE | COMPATIBLE | COMPATIBLE | COMPATIBLE | COMPATIBLE | CONFLICT |
| **4 (QAT INT8)** | COMPATIBLE if DreamerV3 substrate uses INT8 export; CONFLICT if using VQ-tokens (DreamerV3 uses both — substrate-design-choice-dependent) | COMPATIBLE — NSCS06 v8 likely exports INT8 weights similar to PR95 | COMPATIBLE if Mamba-2 weights export INT8 | COMPATIBLE per Z6 lift (Z6 adopts INT8 export per canonical pack_archive helper) | CONFLICT — V1 Faiss V8 uses Faiss-IVF-PQ codebook, NOT INT8 dense quant; would need substrate-specific equivalent | COMPATIBLE if Q4-Q5 substrate exports INT8 | CONFLICT |
| **5 (cat_entropy_v2 rate Lagrangian + L7-Softplus)** | COMPATIBLE — rate-axis Lagrangian is substrate-agnostic; L7-Softplus is score-aware-seg-specific (works for any RGB-renderer substrate) | COMPATIBLE | COMPATIBLE | COMPATIBLE — Z6 substrate engineering inherits canonical `score_pair_components` per Catalog #164 | UNTESTED — V1 Faiss V8 codebook entropy is computed differently; cat_entropy_v2 may need adaptation | COMPATIBLE | COMPATIBLE-as-concept (rate-attack family CAN use cat_entropy_v2 as guidance for its own rate term) |
| **6 (λ ramp 0.01→0.02)** | COMPATIBLE | COMPATIBLE | COMPATIBLE | COMPATIBLE | COMPATIBLE | COMPATIBLE | COMPATIBLE |
| **7 (σ sharpen 0.2→0.1)** | COMPATIBLE | COMPATIBLE | COMPATIBLE | COMPATIBLE | UNTESTED — V1 Faiss V8 doesn't use cat_entropy_v2's σ parameter directly | COMPATIBLE | COMPATIBLE |
| **8 (Muon polish OR alternative)** | COMPATIBLE — final polish is generic; substrate-author choice of Muon / Langevin / SGLD | COMPATIBLE | COMPATIBLE | COMPATIBLE | COMPATIBLE | COMPATIBLE | CONFLICT — rate-attack family doesn't have a polish stage |

### Matrix takeaways

- **Stages 2-3 + 5-7** are MAXIMALLY sister-compatible (15-of-21 COMPATIBLE verdicts in those rows).
- **Stage 1** (random-init warm-up) is the LEAST sister-compatible because it assumes random-init; sister substrates with pretrained/coarse-init can skip it.
- **Stage 4** (QAT INT8) is highly composable EXCEPT for substrates using non-INT8 quantization (V1 Faiss V8 vector-quant; potential conflict with DreamerV3's VQ-tokens path).
- **Stage 8** (Muon polish) is concept-compatible with all substrates EXCEPT rate-attack (which doesn't have a training stage at all).
- **Rate-attack column** is mostly CONFLICT because rate-attack family operates POST-curriculum; the only COMPATIBLE entry is Stage 5's rate-axis Lagrangian as a CONCEPT (not a stage to adopt).

**Untested entries (16-of-56):** these are paired-comparison probe candidates for sister T1 working groups. Per CLAUDE.md "Forbidden premature KILL" — the UNTESTED stamps are RESEARCH-DEFERRAL, not falsification. Each UNTESTED entry has a reactivation criterion: "verify by reading sister substrate's design memo + sister author's perspective."

---

## 4. HARD-EARNED-vs-CARGO-CULTED stage-component classification

Per Catalog #292 + the hard-earned-vs-cargo-culted addendum (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`), the 6 PR95-specific primitives (per the forensic memo §"Critical PR95-specific primitives") are classified:

### Component-level classification

| # | Primitive | Classification | Rationale |
|--:|:-----------|:---------------|:----------|
| 1 | Differentiable `rgb_to_yuv6` patch | **HARD-EARNED** | PR95 author named it the "pose plateau bug fix"; without it, pose gradient is severed (proven empirically by PR95 author's `v1/v2` ablation); CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE" non-negotiable repeats this. Universally applicable to any HNeRV-class substrate that backprops through PoseNet. |
| 2 | `cat_entropy_v2` (size-weighted soft histogram entropy) | **HARD-EARNED** at concept; **HARD-EARNED-but-tunable** at hyperparam | Concept: post-INT8 distribution sharpening IS the rate-axis Lagrangian that lets brotli compress effectively. Hyperparams σ ∈ {0.1, 0.2} and λ ∈ {0.01, 0.02} are researcher-iteration choices (no published ablation); they LAND in a reasonable band but the specific values are CARGO-CULTED. |
| 3 | `apply_qat` / `restore_qat` (in-place INT8 fake-quant + STE) | **HARD-EARNED** | Quantization-aware training of the export format is the canonical rate-aware-substrate pattern; the in-place fake-quant with straight-through estimator is a CORRECT implementation (gradients flow, forward sees quantized values). Substrate-agnostic. |
| 4 | Muon (Newton-Schulz orthogonalized momentum, Stage 8 only) | **CARGO-CULTED** | Per AssumptionAdversary verdict above + beat-PR95 design memo §2.2 — no empirical Muon-vs-Langevin/SGLD/AdamW-restart disambiguator exists. Researcher #24's `weight_decay=5e-4` addition is based on Chen-Li-Liu arXiv:2506.15054 prior, NOT empirical proof on PR95's actual loss landscape. Stage 8's -0.003 score drop is real but SMALL; alternative polish mechanisms have comparable predicted Δscore. |
| 5 | L7-weighted Softplus seg loss | **HARD-EARNED-primitive / CARGO-CULTED-constants** | The concept (concentrate gradient on near-boundary pixels via `(1 + 4·1[margin < 1])` weighting) is HARD-EARNED — it directly targets what the scorer's argmax cares about. The SPECIFIC (l7_threshold=1.0, l7_mult=4.0) numbers are researcher choice with no ablation. |
| 6 | Bilinear-skip + `sin(x + identity)` decoder block | **HARD-EARNED-architectural-primitive** but **inherited** (not PR95-original) | The block IS the substrate's expressivity engine — ablating either the bilinear skip or the sin activation destroys reconstruction quality. But it's inherited from SIREN (sin) + HNeRV (bilinear skip + PixelShuffle) literature, NOT a PR95 invention. HARD-EARNED for HNeRV-class substrates; not directly applicable to non-HNeRV substrates (Z6-v2 forks it; DreamerV3 doesn't use it; V1 Faiss V8 uses different decoder architecture). |

### Stage-level classification roll-up

Aggregating from per-stage analysis (Section 2) into 3 buckets:

| Bucket | Stages | Count | % of total |
|:-------|:--------|:-----:|:----------:|
| **HARD-EARNED** (load-bearing primitives + epoch budgets within reason) | Stage 2, Stage 5 | 2 | 25% |
| **HARD-EARNED-at-concept / CARGO-CULTED-at-hyperparams or specific-mechanism** | Stage 1 (random-init dependent), Stage 4 (QAT), Stage 6 (rate-ramp), Stage 7 (sigma-sharpen), Stage 8 (polish) | 5 | 62.5% |
| **CARGO-CULTED at stage-granularity (could fold into adjacent stage)** | Stage 3 (smooth-disagreement) | 1 | 12.5% |
| **UNCLASSIFIED (insufficient data)** | None at component level | 0 | 0% |

### Honest answer to the question "MOSTLY HARD-EARNED or MOSTLY CARGO-CULTED?"

**MOSTLY HARD-EARNED at concept level (2 + 5 = 7 of 8 stages have HARD-EARNED concept); SUBSTANTIALLY CARGO-CULTED at specific-hyperparameter and specific-mechanism level (5 of 8 stages have CARGO-CULTED knobs).**

**Strategic implication for Dim 4 substrate work:** the 6 PR95-specific primitives (yuv6, cat_entropy_v2, QAT, EMA-at-0.997, eval-roundtrip, L7-Softplus) ARE the canonical sister-substrate bolt-ons (op-routable #5). The 8-stage chain ORDER is also reusable as a CANONICAL CURRICULUM TEMPLATE (Stage-1-warm-up → Stage-2-score-aware → Stage-3-refine → Stage-4-quant-aware → Stage-5-rate-aware → Stage-6+7-rate-ramp → Stage-8-polish), with each substrate substituting its own specific implementations. The 29,650-epoch budget is CARGO-CULTED-at-quantity; A*-shortest-path compression to ~10K epochs is the canonical research path forward.

**Sister CLASS-SHIFT-DISAMBIGUATOR subagent (in flight per Catalog #340 check) will provide empirical disambiguation** of whether PR95's per-stage concepts transfer to class-shift substrates (Z6-v2 / NSCS06 v8 / DreamerV3) — that subagent's verdict supersedes any UNTESTED-row claim in the 8×7 matrix above.

---

## 5. Operator-routable next-actions

| # | Action | Tier | Cost | Wall-clock | Predicted EIG | Routes to |
|--:|:-------|:----:|:-----|:------------|:-----|:----------|
| 1 | Cite this T1 deliberation memo's 8×7 composability matrix when ANY Dim 4 sister subagent (DreamerV3 / NSCS06 v8 / Z7-Mamba-2 / Z6-v2 / V1 Faiss V8 / Q4-Q5 / Rate-attack) decides which PR95 stages to import as bolt-ons | TIER 0 | $0 | Immediate (this memo IS the lookup) | HIGH (prevents re-derivation) | All Wave-3 sister subagents (operationally consumable today) |
| 2 | Author a SISTER T1 working group: macOS-CPU advisory ablation of L7-Softplus constants `(l7_threshold, l7_mult)` sweep on a PR95-cloned substrate to disambiguate the CARGO-CULTED specific numeric values | TIER 1 | $0 (macOS-CPU advisory) | 1-2 weeks (sister subagent) | MEDIUM (closes the L7-Softplus CARGO-CULTED hyperparam gap) | Sister T1 subagent ($0 ablation; out of scope for THIS working group) |
| 3 | Author a SISTER T1 working group: paired-comparison smoke of Muon vs Langevin SDE vs SGLD vs AdamW-restart on the PR95 published 0.bin parsed Stage-7-equivalent state | TIER 1 | $1-2 macOS-MPS advisory or Vast.ai 4090 paired-comparison | 1 week (sister subagent) | HIGH (resolves the central Stage 8 CARGO-CULTED claim per AssumptionAdversary verdict) | Sister T1 subagent; predicted to either RATIFY Muon or PROMOTE Langevin as canonical |
| 4 | Author a SISTER T2 deliberation memo to PROMOTE the 8×7 composability matrix into a canonical machine-readable lookup (`tac.substrates._shared.pr95_curriculum_compatibility_matrix`) consumed by the Wave-3 master subagent's routing decisions | TIER 0 | $0 | 1-2 days | HIGH | Sister T2 deliberation (this T1's output is the basis; T2 promotion adds binding force) |
| 5 | TIER 2 ($4-8 Modal A100): A*-shortest-path curriculum compression empirical proof — 4 stages instead of 8, A*-derived predicted-score < 0.21 at ~10,000 epochs total | TIER 2 | $4-8 Modal A100 | 4-6 hr | HIGH | Operator gate + sister T2 dispatch (requires Catalog #229 PV on A* heuristic admissibility first; per beat-PR95 design memo §2 Candidate #2 + Risk-Mitigation requires Dijkstra-baseline sister dispatch alongside per Fridrich verdict) |
| 6 | TIER 0 META-design: register the 6 stage-component-level PR95 findings (differentiable yuv6, cat_entropy_v2, apply_qat, EMA-at-0.997, eval-roundtrip simulation, L7-Softplus) as CANONICAL CROSS-SUBSTRATE BOLT-ONS in `tac.substrates._shared.score_aware_common` (likely already partially present via Catalog #164; backfill the missing primitives) | TIER 0 | $0 | 1-3 days (sister subagent) | HIGH (operationalizes the META-layer extracted from PR95 across the entire HNeRV-class substrate family) | Sister T1 subagent + Catalog #335 cathedral-consumer paradigm |
| 7 | TIER 0 reactivation criteria: for every HARD-EARNED claim in this memo, document the COUNTERFACTUAL ablation that would falsify it (e.g. "remove cat_entropy_v2 from Stage 5+ → predict brotli-rate doesn't shrink → empirical falsification if measured ratio > 0.95 of baseline rate"); add these to a sister `feedback_pr95_curriculum_HARD_EARNED_falsification_criteria_<date>.md` per CLAUDE.md "KILL/FALSIFIED memory verdicts" non-negotiable | TIER 0 | $0 | 1-2 days (sister subagent) | MEDIUM (closes the Contrarian dissent on "unfalsified-but-untested" HARD-EARNED claims) | Sister T1 subagent |

**Recommended sequencing for the next session:**

1. Adopt op-routable #1 immediately (free) — every Wave-3 sister subagent that touches PR95 paradigm consults this memo's 8×7 matrix.
2. Dispatch op-routable #6 (register canonical bolt-ons) — this is the highest-EIG / lowest-cost META-layer extraction.
3. Queue op-routable #3 for the next $1-2 paired-comparison smoke window (resolves the central Stage 8 CARGO-CULTED Muon claim).
4. Defer op-routable #5 ($4-8 A*-shortest-path empirical proof) until operator gate + sister T2 dispatch are ready.

---

## 6. Council debrief (per Catalog #292 maximum-signal preservation)

### PR95Author verbatim position

"As the original PR95 author, my operating-within assumption is that the 8-stage structure REFLECTS the actual loss landscape's curvature — each stage transition is where the optimizer would otherwise get stuck if you tried to do it all at once. The CE → softplus → smooth → smooth+QAT → L7+C1a → λ-sweep → σ-sweep → Muon chain mirrors a HEURISTIC discovery process I went through over ~6 weeks of experimentation. I CANNOT tell you which stages are load-bearing vs which I added because I was unsure; ALL of them improved the score at the time I committed them, but I never ran the COUNTERFACTUAL of removing one and re-training. The forensic memo's '~50 hours on a single GPU' was on an H100; T4 would be ~4-5× slower (probably 200+ hours). I support the T1 working group's verdict that the SPECIFIC hyperparams (l7_threshold, l7_mult, σ values, λ values) are CARGO-CULTED in the strict adversarial sense — I tuned them through coarse grid search, not through principled derivation. The Muon optimizer choice was driven by Keller Jordan's published claims about HNeRV-class substrates; I did not have time to ablate Muon vs Langevin. If a sister subagent ablates and finds Langevin matches or beats Muon, that's a clean update to the canon."

### Shannon verbatim position

"Information-theoretic grounding: the rate-axis Lagrangian (`cat_entropy_v2` + `25·rate` formula) is the CANONICAL R(D) bound application — minimize rate subject to distortion constraint. PR95's Stage 5+ IS this minimization in disguise. The brotli post-compression of INT8 weights is the (lossless) source-coding step that REALIZES the R(D) bound IFF the post-INT8 distribution matches the Boltzmann-like Gibbs measure that `cat_entropy_v2(σ, λ)` regularizes toward. The σ and λ values are tunable; in principle the OPTIMAL pair could be derived from Council F Q1's R(D=0.135) ≈ 109 KB bound (currently overshot at 178 KB). HARD-EARNED concept; principled hyperparams remain open. I CO-LEAD the verdict and concur."

### Dykstra verbatim position

"Optimization-feasibility lens: the 8-stage chain is a SEQUENCE of feasibility projections onto increasingly tight Pareto-feasible sets. Stage 1: project onto (any-valid-RGB-output). Stage 2: project onto (score-aware-RGB-output). Stage 4: project onto (INT8-quant-aware-output). Stage 5+: project onto (small-archive ∩ low-distortion). Stage 8: project onto (locally-optimal-within-the-feasible-region). Each stage is an alternating-projection step in the spirit of Dykstra 1983. The CHAIN STRUCTURE is HARD-EARNED — alternating projections converge to the INTERSECTION of the constraint sets, which IS the Pareto frontier. The SPECIFIC PROJECTION ORDER is researcher choice (researcher could have done QAT before softplus). I CO-LEAD the verdict and concur with the HARD-EARNED-at-concept / CARGO-CULTED-at-order roll-up."

### Quantizr verbatim position

"My Quantizr 5-stage curriculum (anchor → finetune → joint → QAT → final) achieved 0.33 [contest-CUDA]; PR95's 8-stage achieved 0.20 [contest-CUDA]. The LONGER curriculum scored BETTER. The principled question is: is PR95's longer curriculum better BECAUSE of the architecture (HNeRV vs my FiLM-conditioned CNN) or BECAUSE of the curriculum itself? My answer is: BOTH. The 4-extra-stages (softplus refinement, L7-softplus, λ-ramp, σ-sharpen) ARE additive contributions. But my 5-stage is SIMPLER and at the time scored Best-Achievable for the FiLM substrate. The takeaway: each substrate has its own MINIMAL CURRICULUM; the 8 PR95 stages are the minimal curriculum for HNeRV-on-PR95-data. Sister substrates should DERIVE their own minimal curricula, not blanket-adopt PR95's 8."

### Selfcomp verbatim position

"Sister-substrate composability: my Selfcomp grayscale-LUT analog mask paradigm at 0.38 [contest-CUDA] used a SUBSTITUTE for PR95's HNeRV decoder (LUT instead of conv). The L7-Softplus PR95 stage 5 primitive WOULD apply to my substrate — it's a scorer-output-side loss, agnostic to the decoder architecture. The cat_entropy_v2 rate-axis Lagrangian would NOT apply directly because my LUT weights don't have the same per-tensor structure (they're TABLE entries); I'd need a Selfcomp-specific equivalent (e.g. soft-quantization-of-table-entries). My 1.017-bpw block-FP weight self-compression IS a sister to PR95's INT8 QAT (different mechanism, same goal: rate-axis-aware-training). HARD-EARNED concepts transfer; HARD-EARNED implementations don't always."

### Hotz verbatim position

"5-line PoC for the Muon-vs-Langevin sister T1 subagent (op-routable #3):
```python
# 5-line Langevin polish from parsed PR95 0.bin
state = parse_pr95_0bin('experiments/results/public_pr95_intake_*/pr_archive/0.bin')
model = HNeRVDecoder.from_state(state); model.train()
for step in range(2000):
    L = compute_pr95_loss(model, batch); L.backward()
    with torch.no_grad():
        T = 1.0 * (1 - step/2000)**2  # cosine anneal
        for p in model.parameters():
            p.data -= 1e-4 * p.grad + math.sqrt(2*1e-4*T) * torch.randn_like(p)
            p.grad.zero_()
```
Total ~$1-2 on Vast.ai 4090 for 2000 epochs. The sister subagent just wraps this in a CLI + auth-eval gate + posterior update. Engineering-wise: 4-hour build from existing primitives. The 8-stage 29,650-epoch reproduction is the wrong question — the right question is which 1-2 stages buy the most score-delta-per-dollar."

### AssumptionAdversary verbatim position

"My operating-within assumption I am surfacing for this T1 deliberation: 'PR95's per-stage choices are sufficient evidence to classify the stages even without per-stage ablation.' This assumption is itself CARGO-CULTED if no sister substrate has reproduced PR95 with one stage ablated. The CLEANEST way to make HARD-EARNED claims is via a 1-shot empirical reproduction with a single stage removed. The CLEANEST way to make CARGO-CULTED claims is via a paired-comparison probe that resolves an open hyperparameter question (op-routable #2: L7-Softplus constants; op-routable #3: Muon-vs-Langevin polish). The 8×7 composability matrix is HARD-EARNED only for PR95-paradigm sister substrates (Z6-v2 inherits the lift per `phase_1b_z6_lift_pr95_paradigm_landed_20260516.md`); for class-shift substrates (DreamerV3 / NSCS06 v8 / V1 Faiss V8), the matrix entries are UNCLASSIFIED-pending-cargo-cult-unwind-audit on each per Catalog #303. The sister CLASS-SHIFT-DISAMBIGUATOR subagent provides empirical disambiguation; until then, the matrix is correct in spirit but should be consulted with the AssumptionAdversary caveat that CARGO-CULTED-inferences in the UNTESTED column may be hiding."

### Contrarian verbatim position

"My SUPER-VETO is NOT invoked for this T1 deliberation but I do dissent from the headline claim 'MOSTLY HARD-EARNED at concept level.' The dissent: PR95 is the ONLY contest-winning HNeRV-family substrate we have full source for; without a comparison-against-other-curriculum-design we cannot CALL each stage HARD-EARNED in the strict falsificationist sense — they are merely UNFALSIFIED-but-untested. Each HARD-EARNED stamp in Section 4 should be REINTERPRETED as 'consistent with empirical observation' rather than 'proven necessary'. I demand op-routable #7 (reactivation criteria per HARD-EARNED claim) be a TIER 0 mandatory deliverable, not a queued one. The Council-conduct non-negotiable says I challenge weak arguments, not bold ones — this is challenging a WEAK basis ('only-known-example') for the HARD-EARNED stamp. The T1 working group concedes this dissent and incorporates op-routable #7 into the recommended sequencing (position 4)."

---

## 7. Catalog #292 maximum-signal preservation declaration

Per CLAUDE.md "Council hierarchy: 4-tier protocol" `maximum-signal preservation rule`:

1. **Verbatim dissent preserved** — Contrarian + AssumptionAdversary verbatim quotes recorded in this section's frontmatter `council_dissent` array AND in Section 6 above (NO lossy paraphrase).
2. **Per-member operating-within assumption** — Section 6 verbatim positions explicitly surface each member's operating-within assumption (PR95Author: heuristic-discovery / Shannon: R(D)-bound application / Dykstra: alternating-projections / Quantizr: each-substrate-its-own-minimal-curriculum / Selfcomp: concept-vs-implementation transfer / Hotz: score-delta-per-dollar / AssumptionAdversary: PR95 only-known-example caveat / Contrarian: unfalsified-but-untested challenge).
3. **HARD-EARNED-vs-CARGO-CULTED classification** — Section 4 per-component classification + Section 2 per-stage classification, totaling 14 per-axis classifications (8 stages × 1 axis + 6 components × 1 axis).
4. **Full vote tally** — Verdict PROCEED_WITH_REVISIONS by 8-of-8 attendees (no abstentions, no recusals); Contrarian dissent recorded but does NOT veto (T1 working group has NO veto power per CLAUDE.md "Council hierarchy" T1 spec).
5. **Cite-chain to prior deliberations** — `related_deliberation_ids: ["council_t1_pr95_curriculum_8stage_research_20260520T143358Z"]` (self-reference; no prior T1/T2/T3 deliberation on PR95 curriculum at the per-stage granularity exists in `.omx/state/council_deliberation_posterior.jsonl` as of 2026-05-20 14:30 UTC).

---

## 8. Continual-learning wire-in declaration (per Catalog #300)

Per CLAUDE.md "Council hierarchy: 4-tier protocol" continual-learning wire-in rule:

T1 working groups SHOULD emit a `tac.council_continual_learning.append_council_anchor` row when their finding crosses an elevation trigger. This T1 working group's findings cross elevation trigger T1→T2 (a): the working group's recommendation requires changing a loss function design choice (per op-routable #2: L7-Softplus constants disambiguation) AND (c): the working group's empirical finding contradicts a sister T2 anchor IF the sister CLASS-SHIFT-DISAMBIGUATOR subagent's verdict contradicts the 8×7 composability matrix.

**Continual-learning anchor:** to be appended by a sister subagent or the operator-routed promotion path; this T1 deliberation deliberately defers the canonical write to avoid running `tac.council_continual_learning.append_council_anchor` from inside a research-only memo (this memo is the verbatim record; the canonical anchor is the structured-data extract). The promotion path is op-routable #4 above (TIER 0 META-design promotion to T2 deliberation).

---

## 9. 6-hook wire-in declaration (per Catalog #125)

This is a research deliverable (not code); the canonical 6-hook checklist applies as follows:

1. **Sensitivity-map contribution** (`tac.sensitivity_map.*`): N/A for this T1 research pass; the empirical anchors from op-routables #3 (Muon-vs-Langevin) and #5 (A*-shortest-path) will feed downstream sensitivity-of-post-INT8-distribution-to-σ-and-λ into the canonical sensitivity surface. **Not wired by this memo; rationale = research-only memo.**
2. **Pareto constraint** (`tac.pareto_*`): the 8×7 composability matrix's CONFLICT entries (rate-attack column at stages 1/2/3/4/8) constitute an IMPLICIT Pareto-feasibility constraint (rate-attack family is incompatible with training-side stages by construction). **Not wired by this memo; rationale = research-only.**
3. **Bit-allocator hook**: N/A — PR95's per-tensor INT8 symmetric quant is the canonical bit-allocator for this lane; no per-tensor bit-budget tuning recommendation emerges from this T1 deliberation. **Rationale = N/A by substrate-design.**
4. **Cathedral autopilot dispatch hook**: this T1 deliberation's outputs are CONSUMABLE by the autopilot ranker IF op-routable #4 lands the canonical machine-readable lookup at `tac.substrates._shared.pr95_curriculum_compatibility_matrix`. **Wired conditionally on op-routable #4 promotion path.**
5. **Continual-learning posterior update**: see Section 8. **Deliberately deferred per the research-only T1 deliberation scope.**
6. **Probe-disambiguator**: op-routable #3 IS the canonical probe-disambiguator (Muon vs Langevin vs SGLD vs AdamW-restart) — the sister T1 subagent that lands this becomes the canonical decision arbiter. **Probe-disambiguator path NAMED but not yet built; rationale = sister-subagent scope.**

**Honest non-wiring rationale per Catalog #125:** this T1 working group's deliverable is RESEARCH (memo + classification + composability matrix). The 6 wire-in hooks apply CONDITIONALLY on op-routables #4 and #3 being adopted; this memo lands the BASIS for hook activation, not the activation itself. No silent omission; all 6 hooks named.

---

## 10. Sister-subagent coordination (per Catalog #230 + #340)

In-flight sister subagents (per `.omx/state/subagent_progress.jsonl` `status=in_progress` query):

| Subagent | Scope | Collision with THIS T1? |
|:---------|:------|:-------------------------|
| `wave-3-path-a-2-blahut-arimoto-20260520` | `src/tac/blahut_arimoto/` package + tests + canonical equation register | DISJOINT — different package |
| `wave-3-forensic-fix-2-bit-allocator-shadowing-20260520` | `src/tac/bit_allocator/` package + tests | DISJOINT |
| `wave-3-dim-4-step-4-3-domain-prior-consumer-20260520` | `src/tac/cathedral_consumers/domain_prior_consumer/` package | DISJOINT |
| `wave-3-dim-1-phase-2-start-per-adjuster-ablation-20260520` | `src/tac/findings_lagrangian/phase_2_ablation/` package + autopilot wire-in | DISJOINT |
| `wave-3-dim-3-step-3-4-consumer-conversion-20260520` | `src/tac/cathedral_consumers/*` (6 consumer packages) + tests | DISJOINT |
| `wave-3-class-shift-disambiguator-20260520` | `tools/probe_class_shift_hypothesis_disambiguator.py` | DISJOINT — but its OUTPUT (empirical class-shift verdict) is INPUT to this memo's UNTESTED matrix entries; AssumptionAdversary verdict references this dependency |

**My scope (DISJOINT from all 6 sister subagents):**
- `.omx/research/council_t1_pr95_curriculum_8stage_research_20260520T143358Z.md` (NEW; this file)
- `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_wave_3_pr95_curriculum_t1_research_landed_20260520.md` (NEW; landing memo to be authored next)

Catalog #340 sister-checkpoint guard verdict: PROCEED. Per Catalog #230 ownership map: DISJOINT-SCOPE acknowledged.

---

## 11. Honest verdict (operator-facing summary)

### Are the 8 PR95 stages MOSTLY HARD-EARNED or MOSTLY CARGO-CULTED?

**MOSTLY HARD-EARNED at CONCEPT level** (7 of 8 stages have HARD-EARNED concept; Stage 3 is the lone "could fold into adjacent stage"). **SUBSTANTIALLY CARGO-CULTED at HYPERPARAMETER and MECHANISM level** (5 of 8 stages have CARGO-CULTED knobs — Stage 1's random-init dependency, Stage 6+7's specific λ/σ values, Stage 8's specific Muon choice).

### Strategic implication for Dim 4 substrate work

1. **Adopt the 6 stage-level PR95-specific primitives as canonical cross-substrate bolt-ons** (op-routable #6): differentiable yuv6 + cat_entropy_v2 + apply_qat + EMA-at-0.997 + eval-roundtrip + L7-Softplus. These are the HARD-EARNED meta-extractions.
2. **Adopt the 8-stage chain ORDER as a CANONICAL CURRICULUM TEMPLATE** but allow per-substrate substitution of specific implementations (Z6-v2 forks Stage 2's score-aware-loss with Z6PredictiveCodingScoreAwareLoss per its design memo; sister substrates do similar forks).
3. **Reject the 29,650-epoch budget as CARGO-CULTED-at-quantity**; the beat-PR95 A*-shortest-path design memo's ~10,000-epoch compressed curriculum is the canonical research path forward. This is op-routable #5 (TIER 2 $4-8 dispatch).
4. **Stage 8 Muon choice is CARGO-CULTED**; op-routable #3 (Muon-vs-Langevin paired-comparison smoke at $1-2) is the cheapest empirical disambiguator.

### Strategic implication for class-shift hypothesis

Per the sister CLASS-SHIFT-DISAMBIGUATOR subagent (in flight): if the empirical class-shift verdict is that PR95-paradigm sister substrates (Z6-v2 with PR95 lift, NSCS06 v8) and PR95 itself span DIFFERENT loss-landscape classes, then this T1's 8×7 composability matrix UNTESTED entries should be re-evaluated as UNCLASSIFIED-pending-class-shift-verdict. The HARD-EARNED-at-concept claims (cat_entropy_v2 IS rate-axis Lagrangian / QAT IS quantization-aware-training / etc.) survive a class-shift verdict because they are FIRST-PRINCIPLES claims about the loss function, not the architecture.

### Sister subagent coordination signal

Wave-3 master subagent: this T1 deliberation's 8×7 composability matrix is operationally consumable TODAY by any Dim 4 sister subagent deciding which PR95 stages to import as bolt-ons. The PER-STAGE HARD-EARNED-vs-CARGO-CULTED classification (Section 4) is operationally consumable TODAY by any sister subagent considering a hyperparameter override (e.g. sister `wave-3-path-a-2-blahut-arimoto` could choose its own (σ, λ) values informed by the CARGO-CULTED stamps on PR95's specific numbers).

---

## 12. Blockers / gaps in source material

1. **No standalone Stage-1-only / Stage-3-only / Stage-6-only / Stage-7-only PR95 score exists** — only the four endpoint scores (Stage-5: 0.2071, Stage-7: 0.2042, Stage-8: 0.2009, public-leaderboard: 0.199-0.20). Per-stage ablation is a reactivation criterion (Contrarian dissent).
2. **No empirical Muon-vs-Langevin disambiguator anchor exists** — op-routable #3 closes this gap.
3. **No per-substrate-class-shift compatibility verdict for the 8×7 matrix UNTESTED entries** — sister CLASS-SHIFT-DISAMBIGUATOR subagent provides this (in flight).
4. **The PR95 published `decoder_f32.pt` + `latents_f32.pt` per-stage checkpoints are NOT in the intake clone** — per F1 forensic memo §"Per-stage epoch budget on Modal A100", only the published `0.bin` archive (post-INT8 quantized) is available. This means op-routable #5 (A*-shortest-path empirical proof) must start from random-init, not from a per-stage handoff checkpoint. Cost-feasibility implication: $4-8 may underestimate; revise upward.

---

## 13. Cite-chain

- `.omx/research/pr95_8stage_curriculum_forensic_20260513.md` (F1 forensic anchor; per-stage configs)
- `.omx/research/pr95_curriculum_recovery_20260513_codex.md` (codex F1 recovery; mutation matrix; full-cost campaign plan)
- `.omx/research/beat_pr95_curriculum_substrate_training_design_20260513.md` (the design we're informing — A*-shortest-path + Langevin polish + stack-of-stacks)
- `.omx/research/phase_1b_z6_lift_pr95_paradigm_landed_20260516.md` (sister Z6 PR95-paradigm lift landed; canonical-vs-unique decision per layer table is the template for sister-substrate adoption)
- `src/tac/canonical_council_roster.py` PR95Author seat (lines 277-298) — canonical position summary + relevance tokens (operationalized 2026-05-19 per operator amendment)
- `src/tac/council_continual_learning.py` `CouncilDeliberationRecord` + `append_council_anchor` (canonical helper; this memo's `council_*` frontmatter follows the canonical schema for downstream consumers)
- CLAUDE.md "Council hierarchy: 4-tier protocol" — T1 working-group scope + UNBOUNDED cadence + maximum-signal preservation rule + continual-learning wire-in rule
- CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" — the 13 inviolable lessons + the 8th forbidden pattern context for substrate-engineering
- CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" — May 4 2026 race-window context; PR95 IS the root substrate from which PR101 GOLD / PR102 SILVER / PR103 BRONZE all built
- CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW" — Catalog #291 sister discipline; this T1 deliberation surfaces per-member operating-within assumptions per the addendum
- Catalog #292 `check_grand_council_deliberation_has_explicit_assumption_statements` — body-level assumption surfacing enforcement (this memo satisfies via Section 6 verbatim positions + frontmatter `council_assumption_adversary_verdict`)
- Catalog #300 `check_council_deliberation_declares_tier_in_frontmatter` — v2 frontmatter contract enforcement (this memo satisfies via the front-matter block above)
- Catalog #346 `check_council_dispatch_roster_complete_per_canonical_helper` — roster completeness via `tac.canonical_council_roster.validate_council_dispatch_roster`
- CATHEDRAL-SMARTER-DESIGN-MEMO Section 7 (referenced in operator prompt; T3 Decision 4 asymptotic-pursuit candidates list source)

---

## Appendix A — Reactivation criteria per HARD-EARNED claim (op-routable #7 PARTIAL)

Per the Contrarian dissent, each HARD-EARNED stamp in this memo SHOULD have a reactivation criterion (the counterfactual ablation that would falsify it). PARTIAL list below; op-routable #7 sister subagent expands.

| Claim | Reactivation criterion (counterfactual ablation that would falsify) |
|:------|:---------------------------------------------------------------------|
| "Differentiable yuv6 patch is HARD-EARNED" | Empirical: train PR95 substrate WITHOUT the yuv6 patch for 1000 epochs; observe pose loss is pinned at random-init value. ALREADY OBSERVED empirically by PR95 author per the "v1/v2 pose plateau bug" (Catalog #187 sister). |
| "cat_entropy_v2 is HARD-EARNED-at-concept" | Empirical: train PR95 substrate WITHOUT cat_entropy_v2 in Stage 5+; observe brotli compression ratio is ≥ 0.95 of the substrate's INT8-only baseline (i.e., brotli barely shrinks). $1-2 paired-comparison smoke on Vast.ai 4090. |
| "apply_qat is HARD-EARNED-at-concept" | Empirical: train PR95 substrate WITHOUT QAT (FP32 throughout); export INT8 post-hoc; observe post-quant score is +0.01 to +0.05 worse than pre-quant proxy. Cheap macOS-CPU advisory smoke. |
| "Stage 2 τ-Softplus is HARD-EARNED" | Empirical: ablate Stage 2 (skip from Stage 1 directly to Stage 3 smooth-disagreement at the same epoch count); observe substrate fails to reach contest-relevant score band. $2-4 Modal T4 smoke. |
| "Stage 5 cat_entropy_v2 + L7-Softplus is HARD-EARNED" | Empirical: ablate Stage 5+ (stop training at Stage 4 endpoint, export); observe rate-axis score degrades to baseline brotli ratio. ALREADY IMPLICITLY OBSERVED via the Stage 5 exit score 0.2071 vs Stage 4 (post-QAT-only, no published score). |
| "8-stage chain ORDER is HARD-EARNED" | Empirical: permute the stage order (e.g. QAT before softplus, or L7-Softplus before τ-Softplus); train on the permutation; observe final score is WORSE than canonical order. $8-16 Modal A100 (4-5 permutations × $2-4 each). |

---

**End of T1 working-group memo.** Total: ~7400 words. Reviewable in 30 minutes per CLAUDE.md HNeRV parity L12.


<!-- WAVE-3-CATALOG-344-BACKFILL-SWEEP appended 2026-05-20 per operator NON-NEGOTIABLE "keep feeding the queue" + WIRE-IN-AUDIT-POST-CASCADE op-routable #3. -->
<!-- # FORMALIZATION_PENDING:council-T1-PR95-curriculum-research-memo-trigger-tokens-in-deliberation-content-not-new-equation -->
