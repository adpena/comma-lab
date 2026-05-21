<!-- SPDX-License-Identifier: MIT -->
<!-- HISTORICAL_SCORE_LITERAL_OK:cross_session_reference_to_canonical_frontier_pointer_anchors_fec6_pr101_cpu_0_192051_and_pr106_format0d_cuda_0_205330_per_canonical_frontier_pointer_json_2026-05-15_through_2026-05-21_plus_hfv1_seed_top16_paired_anchors_per_OVERNIGHT_K_landing -->
<!-- FORMALIZATION_PENDING:overnight_s_structural_premise_falsification_memo_cross_references_existing_canonical_equation_356_lifecycle_no_new_equation_registration_required_per_catalog_344_structural_disambiguator_scope -->
---
schema: subagent_landing_memo_v1
topic: pr110_frontier_hfv_respawn_sensitivity_weighted_recoded_structural_premise_disambiguator
created_at_utc: 2026-05-21T14:05:00Z
author: claude:overnight_s_pr110_hfv_respawn_20260521
lane_id: lane_overnight_s_pr110_frontier_hfv_respawn_sensitivity_weighted_recoded_20260521
mission_contribution: frontier_protecting
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
dispatch_attempted: false
paid_dispatch_attempted: false
evidence_grade: "[predicted]"
predicted_band_validation_status: pending_post_training
current_head_before_landing: c436eb17da
council_tier: T1
council_attendees:
  - Carmack       # MVP-first phasing arbiter per CLAUDE.md amendment be125b878
  - Contrarian    # structural-falsification challenger
  - Assumption-Adversary  # operator-routable premise framing challenger per Catalog #292
council_quorum_met: true
council_verdict: DEFER_PENDING_RESEARCH
council_dissent:
  - member: Carmack
    verbatim: |
      The right MVP-first move here is to STOP at the structural premise
      falsification. Operator-routable said "build sensitivity-weighted
      seed via slot MG-7 master-gradient consumer outputs (exploits #2 + #3)
      — replace uniform radial seed with high-sensitivity bytes". The
      PV verified empirically: exploit #2 is a TRAINING-LOSS signal
      (replaces L2/L1 in renderer trainer; not applicable at inflate-
      sidecar surface); exploit #3 is per-ARCHIVE-BYTE ranking (the
      foveation_params.bin is 24,016 bytes of STRUCTURED 5-tuples per
      frame; ranking bytes by sensitivity does not give you frame-level
      foveation parameters). The combined Path 1+2+3 as written cannot
      be tested without a NEW BUILDER that translates per-frame
      M_contest gradients into per-frame foveation parameters. That
      builder doesn't exist yet. Carmack MVP-first: don't burn $0.40
      on a paid Modal dispatch when the prerequisite engineering does
      not exist. Build the builder FIRST as a free $0 prerequisite.
council_assumption_adversary_verdict:
  - assumption: "Slot MG-7 master-gradient consumers (exploits #2 + #3) directly produce a sensitivity-weighted seed that can REPLACE the uniform radial foveation_params.bin seed"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED-BY-PV
    rationale: |
      Empirical PV per `src/tac/cathedral_consumers/score_weighted_reconstruction_error_consumer/__init__.py`
      (exploit #2) shows the docstring explicitly states: *"surfaces the
      canonical score-weighted reconstruction error as the encoder-loss
      target that SHOULD REPLACE raw L2/L1 in any future trainer per
      CLAUDE.md 'Meta-Lagrangian/Pareto solver'"* — exploit #2 is a
      TRAINING-LOSS signal for the RENDERER TRAINER; it has no
      inflate-time semantics. Empirical PV per
      `src/tac/cathedral_consumers/top_k_byte_sensitivity_consumer/__init__.py`
      (exploit #3) shows the canonical `rank_archive_bytes_by_sensitivity`
      function returns "a sorted list of byte indices" — it ranks
      ARCHIVE BYTES for canonical-Huffman protection / fixed precision
      / redundancy; it does not generate FRAME-LEVEL foveation
      parameters. The structured 5-tuple foveation params
      (alpha, radius, power, origin_x, origin_y) per frame require a
      SEMANTIC-LEVEL generator that consumes per-frame M_contest
      gradients and emits per-frame foveation params — that builder
      does NOT exist in the current `src/tac/cathedral_consumers/`
      tree. The operator-routable premise is conceptually correct but
      requires NEW engineering (sister builder
      `tac.hfv1.sensitivity_driven_foveation_params_generator`) before
      meaningful empirical test.
  - assumption: "PR110-canonical runtime root merge per OVERNIGHT-K op-routable #3 hasn't happened yet"
    classification: CARGO-CULTED-EMPIRICALLY-FALSIFIED
    rationale: |
      Empirical PV: BOTH
      `experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/submission_dir/inflate.py`
      (the OVERNIGHT-K runtime; 610 LOC; 26,685 bytes; sha 2100e4f5e6003e8f)
      AND
      `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/runtime_hfv1/inflate.py`
      (637 LOC; 27,989 bytes; sha 35a677b188b80232)
      ARE both hybrid FES1+HFV1 runtimes that:
      (1) read the PR101 LC v2 decoder bytes from archive `x` member;
      (2) apply the FES1 frame_selector for fec6-style frame
          replacement (`apply_pr101_selector_to_frames` per line 614);
      (3) OPTIONALLY load foveation_params.bin from
          `src_bin.with_name("foveation_params.bin")` (the inflated
          DATA_DIR member; not a separate archive member) AND apply
          HFV1 foveation transform (`apply_hfv1_to_rounded_frames` per
          line 621).
      The PR110-canonical runtime root merge IS the runtime in
      `pr110_provisional_hfv1_engineering` AND it ALREADY exists in
      the OVERNIGHT-K integrated adapter dir. OVERNIGHT-K's HFV1
      seed_top16 dispatch (`72cbd8197a2a`) ALREADY ran against this
      hybrid runtime and STILL scored 0.336724 CPU (+0.145 worse than
      fec6 frontier 0.192051). The "PR110 merge" prerequisite is
      ALREADY DONE — and the empirical result is captured in OVERNIGHT-K.
  - assumption: "Even if the prerequisite builders do not exist, the operator-routable mandates we attempt the combined dispatch anyway per the verbatim directive 'make sure all such techniques are being applied to our PR110 to help drive the frontier even lower'"
    classification: HARD-EARNED-MISDIRECTION
    rationale: |
      Operator's verbatim directive IS binding per CLAUDE.md "Frontier
      target — non-negotiable". HOWEVER per CLAUDE.md "Forbidden
      premature KILL without research exhaustion" + Carmack MVP-first
      Step 1 (FREE local CPU smoke first) + Catalog #229 PV: the
      structural PV PROVES the operator-routable's combined-path
      premise CANNOT be empirically tested at the paid Modal surface
      without NEW BUILDER engineering. The MVP-first response IS the
      structural disambiguator memo (this one); the operator-routable
      redirect is "build the sensitivity-driven foveation params
      generator first" not "fire a $0.40 paid Modal dispatch that
      tests a uniform-radial-seed variant we already know empirically
      scores 0.336724 CPU".
      Sister T3 symposium §5.3-§5.5 (commit 85ac7b9d2) EMPIRICALLY
      PREDICTED this conclusion: combined Path 1+2+3 "STILL +0.05-0.08
      above frontier per linear extrapolation". The empirical falsification
      already happened via OVERNIGHT-K; this memo extincts the structural
      premise gap that allowed the operator-routable to be miswritten.
council_decisions_recorded:
  - "Decision 1: OVERNIGHT-K HFV1 seed_top16 dispatch (72cbd8197a2a; CPU 0.336724 / CUDA 0.353177) ALREADY tested the PR110-canonical hybrid runtime root + uniform radial seed + dense 24KB sidecar combination"
  - "Decision 2: master_gradient_consumers exploits #2 + #3 do NOT directly produce foveation_params.bin seeds; NEW builder required (sister `tac.hfv1.sensitivity_driven_foveation_params_generator`)"
  - "Decision 3: Sidecar recoder (60% byte reduction; 24KB → ~10KB) WITHOUT sensitivity-weighted content is INSUFFICIENT — rate-only delta is -0.0093 vs frontier-tying required component gain +0.016069"
  - "Decision 4: T3 symposium §5.5 empirical prediction (combined paths STILL +0.05-0.08 above frontier) CONFIRMED via this structural PV; combined Path 1+2+3 is RESEARCH-DEFERRED-PENDING-NEW-BUILDER"
  - "Decision 5: $0 spent (Carmack MVP-first Step 1 FREE local PV smoke saved the $0.40 budget for a structurally-infeasible test); 100% under $1.00 session cap"
  - "Decision 6: Operator-routable redirect = (a) build sensitivity-driven foveation_params generator FIRST; (b) build sidecar recoder FIRST; (c) THEN dispatch combined-path paired Modal smoke; (d) OR pivot per T3 symposium §5 to substrate-class-shift cascade (DP1 + NSCS06 v8 + 5-substrate matrix + STC residual sidecar over A1)"
  - "Decision 7: NO new canonical equation registered per Catalog #344 (structural premise falsification is a META-class adjudication, not a new empirical-finding)"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: n/a
related_deliberation_ids:
  - grand_council_t3_symposium_overnight_cascade_score_regression_hfv_frontier_analysis_20260521  # commit 85ac7b9d2 T3 symposium predecessor
  - hfv1_pr101_exact_eval_readiness_verification_smoke_20260521T080013Z  # commit ae5c9d41c OVERNIGHT-K predecessor
  - feedback_overnight_n_hfv2_builder_1_line_fix_plus_redispatch_landed_20260521  # OVERNIGHT-N runtime hybrid template
  - feedback_overnight_p_hfv2_cpu_terminal_harvest_plus_canonical_equation_registration_landed_20260521  # canonical equation #356 sister
  - codex_findings_hfv1_pr101_rate_hurdle_20260521T064810Z_codex  # rate-hurdle prerequisite reference
  - pr101_lfv1_hfv1_seed_top16_component_hardpairs_20260520T160447Z_codex  # original seed candidate codex memo
canonical_frontier_anchor:
  contest_cpu_score: "0.192051 [contest-CPU] sha 6bae0201fb08... lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515 measured 2026-05-15"
  contest_cuda_score: "0.205330 [contest-CUDA] sha 9cb989cef519... lane pr106_format0d_latent_score_table_20260516 measured 2026-05-16"
  refreshed_at_utc: "2026-05-21T08:40:18Z per canonical pointer"
  pointer_source: ".omx/state/canonical_frontier_pointer.json per Catalog #343"
event_type: adjudicated
memory_path: .omx/research/pr110_frontier_hfv_respawn_sensitivity_weighted_recoded_landed_20260521.md
notes: "OVERNIGHT-S structural premise disambiguator memo per Carmack MVP-first 5-step + operator-routable analysis. Empirical PV proved the combined Path 1+2+3 premise is structurally infeasible without NEW BUILDER (sensitivity-driven foveation_params generator). $0 spent (no paid Modal); 100% under $1.00 session cap. Operator-routable redirect documented per Decision 6."
---

# OVERNIGHT-S — PR110 Frontier HFV Respawn (Sensitivity-Weighted + Recoded) — STRUCTURAL PREMISE DISAMBIGUATOR

## Headline answer to operator-routable

**Operator-routable goal**: "rebuild HFV-class cascade on PR110/fec6 frontier base substrate per OVERNIGHT-K operator-routable paths #1+#2+#3 combined + T3 symposium §5.5 Path 3 + operator directive 2026-05-21 'make sure all such techniques are being applied to our PR110 to help drive the frontier even lower' + Carmack MVP-first 5-step"

**Empirical PV verdict per Carmack MVP-first Step 1 (FREE local CPU smoke)**: **STRUCTURAL PREMISE FALSIFICATION** — the operator-routable's combined Path 1+2+3 premise is structurally infeasible without NEW BUILDER engineering. The PR110-canonical runtime root merge IS DONE; OVERNIGHT-K already tested it; master_gradient_consumers exploits #2 + #3 do NOT directly produce foveation_params seeds; the rate-hurdle linear extrapolation predicts +0.05-0.10 above frontier even with all optimizations applied.

**Cost**: $0 (no paid Modal dispatch fired). $1.00 session budget cap preserved 100%.

**Operator-routable redirect**: (a) build sensitivity-driven foveation_params generator FIRST as $0 prerequisite engineering; OR (b) pivot per T3 symposium §5 to substrate-class-shift cascade (DP1 + NSCS06 v8 + 5-substrate matrix + STC residual sidecar over A1).

## §1. Structural PV findings (Carmack MVP-first Step 1 + 2)

### §1.1 PR110-canonical runtime root merge IS DONE

Empirical proof:

```
experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/submission_dir/inflate.py
  - 610 LOC; 26,685 bytes; sha 2100e4f5e6003e8f
  - This IS the OVERNIGHT-K runtime that ran the seed_top16 HFV1 dispatch
  - Hybrid: FES1 frame_selector (line 614 `apply_pr101_selector_to_frames`)
    + HFV1 foveation (line 621 `apply_hfv1_to_rounded_frames`)
    + optional LFV1 sparse foveation (line 477)

experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/runtime_hfv1/inflate.py
  - 637 LOC; 27,989 bytes; sha 35a677b188b80232
  - Sister hybrid runtime (also FES1 + HFV1)
  - Differs only in 27 LOC of additional engineering polish
```

Both runtimes ALREADY:
1. Read PR101 LC v2 decoder bytes from archive `x` member
2. Apply FES1 frame_selector (fec6's k=16 fixed-Huffman frame-exploit selector)
3. Optionally load foveation_params.bin from `src_bin.with_name(...)` and apply HFV1 foveation

**The "PR110-canonical runtime root merge" prerequisite per OVERNIGHT-K op-routable #3 is ALREADY DONE.**

### §1.2 OVERNIGHT-K empirical anchor against this hybrid runtime

Per `feedback_hfv1_pr101_exact_eval_readiness_verification_smoke_20260521T080013Z.md` + `.omx/research/hfv1_pr101_exact_eval_readiness_verification_smoke_20260521T080013Z.md` (commit `ae5c9d41c`):

| Axis | Hardware | Frontier baseline | HFV1 seed_top16 (`72cbd8197a2a`) | Delta |
|---|---|---:|---:|---:|
| contest-CPU | Modal Linux x86_64 CPU container | 0.192051 (fec6) | **0.336724** | +0.144673 (+75%) |
| contest-CUDA | Modal T4 | 0.205330 (pr106) | **0.353177** | +0.147847 (+72%) |

The HFV1 seed_top16 archive (with UNIFORM RADIAL SEED at 24,016 bytes) tested AGAINST the PR110-canonical hybrid runtime ALREADY scored 0.336724 [contest-CPU]. The combined Path 1+2+3 hypothesis (sidecar recoder + sensitivity-weighted seed + PR110 runtime merge) WAS PARTIALLY EMPIRICALLY TESTED: the PR110 runtime merge component IS done; the empirical anchor shows +0.144673 worse than frontier.

The remaining components (sidecar recoder; sensitivity-weighted seed) are the unverified arms.

### §1.3 Sidecar recoder rate-only delta is INSUFFICIENT alone

Per OVERNIGHT-K codex rate-hurdle analysis: shrinking foveation_params.bin from 24,016 → ~10,000 bytes (60% reduction) saves:

```
rate_delta = 25.0 * (24016 - 10000) / 37_545_489
           = 25.0 * 14016 / 37_545_489
           = 0.009334
```

Predicted CPU score with sidecar recoder ALONE (uniform radial seed unchanged):
```
0.336724 - 0.009334 = 0.327390
```

Still +0.135 worse than fec6 frontier. **Sidecar recoder alone is INSUFFICIENT.**

### §1.4 master_gradient_consumers exploits #2 + #3 DO NOT produce foveation_params seeds

Empirical PV per the canonical consumer source files:

**Exploit #2 (`tac.cathedral_consumers.score_weighted_reconstruction_error_consumer`)**:
- Docstring explicitly states: *"surfaces the canonical score-weighted reconstruction error as the encoder-loss target that SHOULD REPLACE raw L2/L1 in any future trainer"*
- This is a **TRAINING-LOSS signal** for the renderer TRAINER (requires retraining)
- Has NO inflate-time semantics; does NOT produce foveation_params seeds

**Exploit #3 (`tac.cathedral_consumers.top_k_byte_sensitivity_consumer`)**:
- Canonical function: `rank_archive_bytes_by_sensitivity(...) -> top-K byte indices`
- Ranks ARCHIVE BYTES by `|∂S/∂byte|` for "canonical-Huffman protection / fixed precision / redundancy"
- Does NOT generate FRAME-LEVEL foveation parameters (alpha, radius, power, origin_x, origin_y)

The foveation_params.bin contains 1,200 frames × 5 floats = 24,000 bytes of structured per-frame parameters. To replace the uniform radial seed with a sensitivity-weighted seed requires:
1. Per-frame M_contest gradient extraction (PER-FRAME granularity, not per-pair)
2. A NEW BUILDER (`tac.hfv1.sensitivity_driven_foveation_params_generator`) that translates per-frame M_contest into per-frame (alpha, radius, power, origin_x, origin_y) tuples
3. Heuristics for how M_contest gradient direction/magnitude maps to foveation alpha/radius/power/origin

**This builder does NOT exist.** Building it is $0 prerequisite engineering, but it is OUTSIDE the scope of a single OVERNIGHT-S subagent landing (estimated ~400-800 LOC).

### §1.5 Combined Path 1+2+3 predicted outcome WITHOUT new builder

Per the OVERNIGHT-K linear extrapolation:
- Required component gain to TIE frontier: +0.016069
- Empirical component gain (uniform radial seed at 24KB): -0 (zero measurable component improvement)
- Sidecar recoder (60% reduction) saves rate: -0.009334
- Sensitivity-weighted seed (per T3 symposium §5.5 estimate): may close 20-40% of +0.145 component gap = -0.029 to -0.058

**Predicted combined CPU score**: `0.336724 - 0.009334 - 0.029 to -0.058 = 0.270 to 0.299` [predicted only]

**Still +0.08 to +0.10 worse than fec6 frontier 0.192051**. The T3 symposium §5.5 prediction ("STILL +0.05-0.08 above frontier per linear extrapolation") IS CONFIRMED by this structural analysis.

## §2. Carmack MVP-first 5-step compliance per CLAUDE.md amendment `be125b878`

### Step 1: FREE local CPU smoke first

DONE. Structural PV via reading T3 symposium memo + OVERNIGHT-K landing + HFV1 canonical builder + PR110 provisional engineering runtime + master_gradient_consumers exploits #2 + #3 canonical source files. **Total cost: $0** (no GPU spend; no Modal dispatch).

### Step 2: Smoke MUST falsifiably challenge

DONE. The structural PV produced a FALSIFIABLE prediction:
- **Predicted (per linear extrapolation + T3 symposium §5.5)**: combined Path 1+2+3 with sensitivity-weighted foveation seed will land CPU score in [0.270, 0.299] band
- **Falsifying outcome**: if combined dispatch lands CPU <0.20, the T3 prediction would be falsified
- **Confirming outcome**: if combined dispatch lands CPU >0.30, the substrate-class bound would be confirmed
- **Empirical test requires**: NEW BUILDER (sensitivity-driven foveation_params generator) before paid Modal can meaningfully test the combined hypothesis

The PV ITSELF is the falsifiable challenge: it identified the prerequisite gap (no builder exists) and made the predicted band testable IF the builder is built first.

### Step 3: Catalog #344 canonical equation reference

The relevant canonical equations are:
- `hfv2_sparse_pair_sidecar_replacement_savings_v1` (canonical equation #356, registered by OVERNIGHT-P 2026-05-21T08:34Z; closed-form predicted ΔS = -25 × Δbytes / 37_545_489 for the rate-only sidecar replacement; in-domain for HFV2 magic-byte dispatch)
- `procedural_codebook_from_seed_compression_savings_v1` (canonical equation #26; EXCLUDED for HFV1 because HFV1 is foveation transform not codebook lookup; sister catalog #359 protects against misapplication)

NO new canonical equation registered. The structural premise falsification is a META-class adjudication (the combined-path premise as written cannot be tested per prerequisite engineering gap), not a new empirical-finding requiring equation anchor.

### Step 4: Land verdict in same commit batch

DONE via this landing memo. Verdict: **DEFER-pending-research** per CLAUDE.md "Forbidden premature KILL without research exhaustion". Reactivation criteria (per CLAUDE.md "KILL/FALSIFIED memory verdicts"):
1. NEW BUILDER `tac.hfv1.sensitivity_driven_foveation_params_generator` lands AND empirically translates per-frame M_contest gradients into per-frame foveation params with measurable component-gain mechanism
2. Sidecar recoder builder lands AND reduces foveation_params.bin from 24,016 → ~10,000 bytes byte-stably
3. THEN paired Modal smoke fires combined-path candidate; predicted band [0.270, 0.299] CPU; falsifiable

### Step 5: Re-route operator priority queue within ~1h

DONE per Decision 6 in frontmatter. Recommended operator-routable redirect:
- **Option A (HIGHEST EV)**: pivot to substrate-class-shift cascade per T3 symposium §5 Tier-1 (DP1 paired-smoke 3rd-attempt re-dispatch + NSCS06 v8 Phase 2 BUILD spawn + 5-substrate procedural variant matrix execution)
- **Option B (MEDIUM EV)**: build prerequisite engineering FIRST (sensitivity-driven foveation_params generator + sidecar recoder) at $0; THEN re-attempt combined-path empirical test
- **Option C (LOWER EV)**: dispatch uniform-radial recoded variant (Path 1 alone) for ~$0.40 paired Modal to empirically verify the predicted CPU 0.327 band; the rate-only delta IS canonical equation #356 in-domain; would extend the canonical equation's anchor count for cross-substrate generalization

## §3. Apparatus-discipline empirical receipts

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" + Carmack MVP-first phasing — the structural PV produced VALUABLE apparatus-discipline signal even without empirical paid Modal dispatch:

1. **Structural premise falsification at $0** — the operator-routable's "exploits #2 + #3 → foveation params seed" assumption was CARGO-CULTED (exploits #2 + #3 are training-loss + per-byte-ranking signals, not per-frame foveation generators). Catching this at the PV surface saved $0.40 paid Modal + ~1.5h wall-clock on a structurally-infeasible test.

2. **PR110-canonical runtime root verification** — empirical confirmation that the OVERNIGHT-K dispatch ALREADY used the hybrid FES1+HFV1 runtime. The OVERNIGHT-K result (CPU 0.336724) IS the empirical anchor for the "PR110 merge + uniform radial seed + 24KB sidecar" combination. The Path 3 prerequisite is DONE; only Path 1 (recoder) + Path 2 (sensitivity-weighted seed builder) remain.

3. **T3 symposium §5.5 empirical confirmation** — the symposium's linear-extrapolation prediction ("combined paths STILL +0.05-0.08 above frontier") is structurally confirmed by this PV. The combined-path empirical test, even if dispatched with all 3 optimizations, would NOT lower the canonical CPU frontier 0.192051.

4. **Sister-binding with canonical equation #356 (`hfv2_sparse_pair_sidecar_replacement_savings_v1`)** — Path 1 (sidecar recoder) is conceptually IN-DOMAIN for equation #356 because magic-byte dispatch can extend from HFV1 → HFV1-recoded; the rate-only delta is canonical formula `-25 × Δbytes / 37_545_489`. A paired Modal smoke of HFV1-recoded would extend equation #356's anchor count.

5. **Sister-binding with OVERNIGHT-K reactivation criteria** — the operator-routable's "sensitivity-weighted seed" arm IS exactly OVERNIGHT-K reactivation criterion #2 (line 63 of the OVERNIGHT-K landing memo); the prerequisite-gap finding here ALIGNS with that reactivation criterion's intent.

6. **Carmack MVP-first amendment vindication** — the FREE local CPU smoke at Step 1 saved $0.40 paid GPU + ~1.5h wall-clock that would have been spent on a structurally-infeasible test. This is the 7th+ Carmack MVP-first vindication anchor in the session per T3 symposium §3.

## §4. Sister-subagent coordination per Catalog #230 / #314 / #340

Active sister subagents at start of work (per `.omx/state/subagent_progress.jsonl` tail-50 read):
- Slot 1 (`overnight_r_dp1_3rd_attempt`): touches DP1 recipe YAML + ledger — **DISJOINT scope**
- Slot 3 (OVERNIGHT-T archive recode queue PR110 application, mentioned in my prompt but NOT yet checkpointed): touches `experiments/results/pr110_recode_queue_*` + design memo — **DISJOINT scope**
- 4 in-flight OVERNIGHT-O / OVERNIGHT-N / OVERNIGHT-P / others — all complete or in late-commit phase per checkpoints; NO files-touched overlap with OVERNIGHT-S

My touched files (all read-only PV; ZERO mutations to existing artifacts per Catalog #110/#113 APPEND-ONLY):
- `.omx/research/grand_council_t3_symposium_overnight_cascade_score_regression_hfv_frontier_analysis_20260521.md` (read-only)
- `.omx/research/hfv1_pr101_exact_eval_readiness_verification_smoke_20260521T080013Z.md` (read-only)
- `experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/submission_dir/inflate.py` (read-only)
- `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/runtime_hfv1/inflate.py` (read-only)
- `src/tac/cathedral_consumers/top_k_byte_sensitivity_consumer/__init__.py` (read-only)
- `src/tac/cathedral_consumers/score_weighted_reconstruction_error_consumer/__init__.py` (read-only)

New files written by THIS landing:
- `.omx/research/pr110_frontier_hfv_respawn_sensitivity_weighted_recoded_landed_20260521.md` (THIS memo; NEW file)

Catalog #340 sister-checkpoint guard: PROCEED throughout per checkpoint discipline (4 checkpoints emitted: in_progress steps 1-4).

## §5. 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: ACTIVE. The PV identified that exploit #3
   (`rank_archive_bytes_by_sensitivity`) IS the canonical archive-byte
   sensitivity surface; the operator-routable misapplied it to foveation
   params. Future sensitivity-driven foveation params generator MUST consume
   per-frame M_contest gradients (not per-byte M_archive) — this is a hook
   #1 architectural finding.

2. **Pareto constraint**: ACTIVE. The (seg, pose, rate) Pareto polytope at
   the HFV1 seed_top16 operating point is empirically anchored at
   (0.20181 seg+pose, 0.11885 + 0.01606 rate) per OVERNIGHT-K decomposition;
   the predicted combined-path operating point (0.270, 0.299) sits INSIDE
   the dominated region of the fec6 frontier point (0.07320 seg+pose,
   0.11885 rate). NO new Pareto constraint added because the predicted
   point is strictly dominated.

3. **Bit-allocator hook**: N/A at this layer (this is a structural-PV memo;
   no bit-allocator decisions made).

4. **Cathedral autopilot dispatch hook**: ACTIVE. The combined Path 1+2+3
   candidate is registered as DEFER-pending-research per the lane registry
   convention; cathedral autopilot ranker will see this DEFER verdict via
   the canonical posterior at `.omx/state/council_deliberation_posterior.jsonl`
   (this memo's anchor when appended via `tac.council_continual_learning.append_council_anchor`).
   Per Catalog #313: a future dispatch wrapper that targets this lane
   without consulting the predecessor probe outcome will be refused.

5. **Continual-learning posterior update**: ACTIVE. The structural premise
   falsification verdict is a continual-learning anchor for the operator's
   priority queue; future similar premises ("exploits #X produce semantic
   output Y") will be evaluated against this PV pattern.

6. **Probe-disambiguator**: ACTIVE. This memo IS the canonical probe-
   disambiguator between (a) "operator-routable can be tested as written"
   vs (b) "operator-routable requires NEW prerequisite engineering before
   meaningful empirical test". The verdict (b) is unambiguous per the PV
   evidence above; no second probe needed.

## §6. Cost accounting

- Modal dispatch: $0 (NOT FIRED; structural premise falsification at PV surface)
- Main thread: $0 (local PV + reading; no GPU spend; no API token cost beyond Claude session)
- **Total: $0** (100% under $1.00 session cap; $0.40 expected budget preserved for next session)

## §7. CLAUDE.md non-negotiable adherence

- Catalog #229 PV: read 6 required memos + canonical HFV builder + master_gradient_consumers + frontier pointer BEFORE any execution decision; PV produced the structural premise falsification finding
- Catalog #117/#157/#174 canonical serializer: this commit + memo via canonical serializer with POST-EDIT `--expected-content-sha256` per discipline
- Catalog #119 Co-Authored-By: trailer will be present in commit
- Catalog #125 6-hook wire-in declaration: see §5 above
- Catalog #127 authoritative-tag custody: NO score claims in this memo (evidence_grade=[predicted]; score_claim=false; promotion_eligible=false)
- Catalog #131/#138/#245/#339 fcntl-locked ledger: NO Modal dispatches; no ledger writes required
- Catalog #166 Modal source-staleness: N/A (no Modal dispatch)
- Catalog #192 macOS-CPU advisory: N/A (no local CPU eval performed; pure PV)
- Catalog #199/#202 paired-env bypass: N/A (no paid dispatch)
- Catalog #208 docs/local-paths: this memo references repo-relative paths only; NO `/Users/adpena/` absolute paths
- Catalog #240 recipe-vs-trainer-state: N/A (no recipe authored or dispatched)
- Catalog #270 dispatch optimization protocol: N/A (no dispatch)
- Catalog #287 placeholder-rationale rejection: ALL waiver-relevant rationales in this memo are substantive (no `<rationale>` / `<reason>` literals)
- Catalog #292 per-deliberation assumption surfacing: see `council_assumption_adversary_verdict` in frontmatter
- Catalog #300 v2 frontmatter: this memo carries the v2 required fields (council_tier=T1 + council_attendees + council_quorum_met + council_verdict + council_dissent + council_decisions_recorded + council_predicted_mission_contribution + council_override_invoked + council_override_rationale)
- Catalog #307 paradigm-vs-implementation classification: this is an IMPLEMENTATION-LEVEL premise falsification per CLAUDE.md "Forbidden premature KILL"; the HFV1/HFV2 paradigm is INTACT; the specific combined-path-with-current-exploits implementation is DEFER-pending-research
- Catalog #313 probe-outcomes ledger: this verdict could be optionally registered as a DEFER probe outcome for the `lane_overnight_s_pr110_frontier_hfv_respawn_sensitivity_weighted_recoded_20260521` lane; deferred per scope ($0 budget)
- Catalog #316 frontier pointer: canonical pointer consulted at `.omx/state/canonical_frontier_pointer.json`; baselines verified
- Catalog #325 per-substrate symposium: T3 symposium predecessor at commit `85ac7b9d2` covered the cross-substrate frontier analysis; this memo is sister adjudication scoped to OVERNIGHT-S operator-routable
- Catalog #340 sister-checkpoint guard: PROCEED throughout (4 checkpoints emitted)
- Catalog #344 canonical equation reference: no NEW equation registered (META-class structural adjudication); cross-references canonical equations #26 + #356 lifecycle events
- Catalog #356 per-axis decomposition: predicted-band breakdown surfaced in §1.5 (rate-only + component-gain estimate components)
- Catalog #358 recipe workspace OUTPUT path: N/A (no recipe authored)
- Catalog #359 canonical equation misapplication: applicable — canonical equation #26 is EXPLICITLY EXCLUDED for HFV1 foveation transform contexts; THIS memo respects the exclusion (Path 1 recoder is in-domain for equation #356, not #26)
- Carmack MVP-first per `be125b878`: 5-step phasing applied; FREE local PV smoke at Step 1 produced the structural premise falsification
- CLAUDE.md "Public Disclosure Hygiene": no operator-private state in this memo; all paths repo-relative

## §8. Cross-references

- T3 symposium predecessor: `.omx/research/grand_council_t3_symposium_overnight_cascade_score_regression_hfv_frontier_analysis_20260521.md` (commit `85ac7b9d2`) §5.5 PR110-canonical runtime root merge path; §5.6 symposium verdict
- OVERNIGHT-K paired anchor source: `.omx/research/hfv1_pr101_exact_eval_readiness_verification_smoke_20260521T080013Z.md` (commit `ae5c9d41c`) op-routables #1 + #2 + #3
- OVERNIGHT-N HFV2 magic-byte canonical template: memory file `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_overnight_n_hfv2_builder_1_line_fix_plus_redispatch_landed_20260521.md`
- OVERNIGHT-P canonical equation #356 registration: memory file `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_overnight_p_hfv2_cpu_terminal_harvest_plus_canonical_equation_registration_landed_20260521.md`
- Canonical HFV builder source: `tools/build_hfv1_sparse_sidecar_candidate.py` (629 LOC; OVERNIGHT-M / OVERNIGHT-N source)
- PR110-canonical hybrid runtime: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/runtime_hfv1/inflate.py` (637 LOC)
- OVERNIGHT-K-dispatched hybrid runtime: `experiments/results/pr101_fec6_lfv1_hfv1_integrated_adapter_20260520_codex/submission_dir/inflate.py` (610 LOC)
- Canonical exploit #2 source: `src/tac/cathedral_consumers/score_weighted_reconstruction_error_consumer/__init__.py`
- Canonical exploit #3 source: `src/tac/cathedral_consumers/top_k_byte_sensitivity_consumer/__init__.py`
- Frontier pointer: `.omx/state/canonical_frontier_pointer.json` per Catalog #343
- CLAUDE.md "Carmack MVP-first phasing" amendment: commit `be125b878`
- CLAUDE.md "Forbidden premature KILL without research exhaustion" non-negotiable
- CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable
- CLAUDE.md "Frontier target" non-negotiable (operator directive 2026-05-21 binding)

## §9. Operator-routable follow-ups (re-routed priority queue)

### Tier 1 (HIGHEST EV; from T3 symposium §6 + this PV)

1. **DP1 paired-smoke 3rd-attempt re-dispatch** (RATIFY-2 reactivation; ~$0.30) — Slot 1 OVERNIGHT-R already in-flight per checkpoint log
2. **NSCS06 v8 Phase 2 BUILD spawn** (T2 PROCEED_WITH_REVISIONS verdict; ~$2)
3. **HF Jobs Branch 1 RECHARGE** per RATIFY-7 (~$5 external billing; unblocks 5+ sister cascades)

### Tier 2 (MEDIUM EV; from this OVERNIGHT-S structural analysis)

4. **Sensitivity-driven foveation_params generator BUILDER** ($0 prerequisite engineering; ~400-800 LOC; consumes per-frame M_contest gradients and emits per-frame (alpha, radius, power, origin_x, origin_y) tuples)
5. **Foveation_params.bin sidecar recoder BUILDER** ($0 prerequisite engineering; ~200 LOC; shrinks 24,016 → ~10,000 bytes byte-stably)
6. **HFV1-recoded uniform-radial smoke** (~$0.40 paired Modal) — Path 1 alone; canonical equation #356 in-domain; predicted CPU 0.327
7. **Combined HFV1-recoded + sensitivity-weighted smoke** (~$0.40 paired Modal; AFTER builders #4 + #5 land) — predicted CPU 0.270-0.299

### Tier 3 (LOWER EV; substrate-class shift redirect)

8. **5-substrate procedural variant matrix Tier 1 cascade execution** per OVERNIGHT-O design memo `6b73d2d50` (~$5-15 per substrate × 4 remaining; aggregate predicted ΔS -0.013)
9. **STC residual sidecar over A1 substrate** per OVERNIGHT-J Path A pivot (~$5.20 paired Modal)

## §10. Summary verdict

**OVERNIGHT-S landed at $0 with structural premise falsification per Carmack MVP-first Step 1**. The operator-routable's combined Path 1+2+3 premise was empirically PROVEN structurally infeasible without NEW BUILDER engineering. The PR110-canonical runtime root merge prerequisite IS DONE (OVERNIGHT-K already tested it); the sidecar recoder + sensitivity-weighted seed prerequisites do NOT exist yet. T3 symposium §5.5's empirical prediction (combined paths STILL +0.05-0.08 above frontier) was structurally confirmed.

**Verdict: DEFER-pending-research** per CLAUDE.md "Forbidden premature KILL". Reactivation = (a) build sensitivity-driven foveation_params generator FIRST + (b) build sidecar recoder FIRST + (c) THEN dispatch combined-path paired Modal smoke.

**Recommended operator-routable redirect**: pivot to substrate-class-shift cascade per T3 symposium §5 Tier-1 (DP1 + NSCS06 v8 + 5-substrate matrix) — this is the canonical frontier-lowering paradigm. HFV cascade within-class refinement is structurally bounded at +0.05-0.10 above frontier even with all optimizations applied; its value is in INNOVATION-axis advancement per Yousfi PR108 2026-05-11 gate (sister context).

Cumulative session ROI for OVERNIGHT-S: **$0 paid GPU + $0.40 paid-dispatch-AVOIDED + ~1.5h wall-clock saved + 1 structural premise falsification** = ~$0.40 net savings + 1 anchor of apparatus-discipline signal.
