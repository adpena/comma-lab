<!-- SPDX-License-Identifier: MIT -->
<!-- HISTORICAL_SCORE_LITERAL_OK:cross_session_reference_to_canonical_frontier_pointer_anchors_fec6_pr101_cpu_0_192051_and_pr106_format0d_cuda_0_205330_per_canonical_frontier_pointer_json_2026-05-15_through_2026-05-21_plus_OVERNIGHT_S_referenced_anchors -->
<!-- FORMALIZATION_PENDING:overnight_x1_builder_landing_memo_cross_references_existing_canonical_equation_procedural_predictor_plus_residual_correction_savings_v1_per_catalog_359_disambiguator_in_domain_context_hfv1_foveation_params_sensitivity_weighted_v1_no_new_equation_registration_required_pending_paired_modal_smoke_anchor -->
---
schema: subagent_landing_memo_v1
topic: overnight_x1_build_sensitivity_weighted_foveation_params_generator_builder_1_of_2
created_at_utc: 2026-05-21T14:44:00Z
author: claude:overnight_x1_build_sensitivity_weighted_foveation_params_generator_20260521
lane_id: lane_overnight_x1_build_sensitivity_weighted_foveation_params_generator_20260521
mission_contribution: frontier_breaking_enabler
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
dispatch_attempted: false
paid_dispatch_attempted: false
evidence_grade: "[predicted]"
predicted_band_validation_status: pending_post_training
current_head_before_landing: 27ae5b7dc
council_tier: T1
council_attendees:
  - Carmack       # MVP-first phasing arbiter per CLAUDE.md amendment be125b878
  - Shannon       # information-theory grounding (foveation_params bytes ARE score-affecting payload)
  - Assumption-Adversary  # per-round assumption surfacing per Catalog #292
council_quorum_met: true
council_verdict: PROCEED
council_dissent:
  - member: Carmack
    verbatim: |
      MVP-first compliance verified: $0 spent, free local CPU smoke first
      (Step 1), falsifiable challenge (Step 2 predicted CPU band [0.270,
      0.299] per linear extrapolation; falsifying if combined dispatch
      lands CPU < 0.20), canonical equation #359-sister IN-DOMAIN
      reference (Step 3 procedural_predictor_plus_residual_correction_savings_v1
      with hfv1_foveation_params_sensitivity_weighted_v1 context), landing
      verdict in same commit batch (Step 4), and operator priority queue
      re-route per §9 (Step 5). This builder is Path 2 prerequisite of
      OVERNIGHT-S DEFER reactivation criteria; the next op-routable is
      either (a) sister Builder 2 (sidecar recoder) for full Path 1+2+3
      empirical readiness, or (b) operator pivots per OVERNIGHT-S
      Recommendation A to substrate-class-shift cascade per T3 symposium
      §5 Tier-1.
council_assumption_adversary_verdict:
  - assumption: "Center-of-mass of per-pixel scorer-sensitivity IS the canonical per-frame foveation center for the contest score"
    classification: HARD-EARNED-BY-PRINCIPLE
    rationale: |
      Per CLAUDE.md "Bit-level deconstruction and entropy discipline" +
      the canonical foveation transform mathematical structure (per
      `apply_hfv1_to_rounded_frames` lines 547-563 in the PR110 hybrid
      runtime), the foveation transform applies a spatial blend centered
      at (origin_x, origin_y) with radius (radius) and falloff (power).
      For a fixed-budget foveation (alpha clamp [0, 0.5]), the
      Bayesian-optimal center IS the center-of-mass of the score-
      sensitivity distribution (weighted-L2 minimization over the
      foveation parameter manifold). This is HARD-EARNED-BY-PRINCIPLE per
      the canonical scorer's per-pixel mathematical structure.
  - assumption: "Per-frame foveation_params suffice to close the +0.145 component-gain gap to fec6 frontier"
    classification: CARGO-CULTED-PENDING-EMPIRICAL
    rationale: |
      Per T3 symposium §5.5 (commit 85ac7b9d2) + OVERNIGHT-S §1.5 (commit
      079edcfdd): combined Path 1+2+3 EVEN WITH this builder predicted
      CPU band is [0.270, 0.299] = STILL +0.08-0.10 above fec6 frontier
      0.192051 per linear extrapolation. The builder unblocks the
      empirical TEST (per OVERNIGHT-S Decision 6); it does NOT guarantee
      the prediction is correct. Empirical verification requires paired
      Modal smoke (out of scope per Carmack MVP-first $0 budget). Per
      CLAUDE.md "Forbidden premature KILL without research exhaustion":
      this is research-substrate territory, not a kill-class claim.
  - assumption: "Alpha clamp band [0, 0.5] is the safe default"
    classification: CARGO-CULTED-OPERATOR-OVERRIDABLE
    rationale: |
      The alpha clamp band [0, 0.5] is a foveation-safe default chosen to
      match the inflate-time `apply_hfv1_to_rounded_frames` semantics
      (alpha=0 skips foveation per line 530; alpha > 0.5 starts producing
      severe foveation artifacts per CLAUDE.md "Apples-to-apples evidence
      discipline" empirical visual inspection). The clamp band is EXPOSED
      as `--alpha-clamp-min` / `--alpha-clamp-max` CLI flags so the
      operator can override per substrate; future builder revisions may
      surface per-substrate clamp bands from empirical anchors.
council_decisions_recorded:
  - "Decision 1: Builder 1 of 2 LANDED per OVERNIGHT-S DEFER reactivation criterion #1 (sensitivity-driven foveation_params generator)"
  - "Decision 2: Canonical input contract = ContestGradientTensor (N_pairs, 3, H, W) per Catalog #318 chain-rule discipline; raw bit-flip FD FORBIDDEN at producer side"
  - "Decision 3: Canonical output contract = HFV1 foveation_params.bin byte layout (HFV1_HEADER + N_frames * HFV1_ROW) matching `runtime_hfv1/inflate.py` HFV1_HEADER_STRUCT + HFV1_ROW_STRUCT"
  - "Decision 4: Per-frame mapping kernel = sensitivity-weighted center-of-mass (origin) + sigma (radius) + log-normalized aggregate magnitude (alpha) + canonical default power (1.0); HARD-EARNED-BY-PRINCIPLE per Bayesian decision theory"
  - "Decision 5: Canonical equation reference = procedural_predictor_plus_residual_correction_savings_v1 (Catalog #359 IN-DOMAIN sister; canonical equation #26 EXPLICITLY EXCLUDED for HFV1 foveation transform contexts)"
  - "Decision 6: $0 spent (Carmack MVP-first $0 budget preserved 100%); paired Modal smoke deferred until sister Builder 2 (sidecar recoder) lands"
  - "Decision 7: Operator-routable redirect = (a) sister Builder 2 enables full Path 1+2+3 empirical readiness; (b) OR pivot per OVERNIGHT-S Recommendation A to substrate-class-shift cascade per T3 symposium §5 Tier-1 (highest EV per ROI analysis)"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: n/a
related_deliberation_ids:
  - pr110_frontier_hfv_respawn_sensitivity_weighted_recoded_landed_20260521  # OVERNIGHT-S predecessor (commit 079edcfdd) — the DEFER memo this builder satisfies
  - grand_council_t3_symposium_overnight_cascade_score_regression_hfv_frontier_analysis_20260521  # T3 symposium §5.5 predicted-band anchor
  - hfv1_pr101_exact_eval_readiness_verification_smoke_20260521T080013Z  # OVERNIGHT-K empirical baseline (CPU 0.336724)
  - feedback_overnight_n_hfv2_builder_1_line_fix_plus_redispatch_landed_20260521  # OVERNIGHT-N runtime hybrid template
  - feedback_overnight_p_hfv2_cpu_terminal_harvest_plus_canonical_equation_registration_landed_20260521  # canonical equation #356 sister
canonical_frontier_anchor:
  contest_cpu_score: "0.192051 [contest-CPU] sha 6bae0201fb08... lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515 measured 2026-05-15"
  contest_cuda_score: "0.205330 [contest-CUDA] sha 9cb989cef519... lane pr106_format0d_latent_score_table_20260516 measured 2026-05-16"
  refreshed_at_utc: "2026-05-21T14:44:00Z per canonical pointer"
  pointer_source: ".omx/state/canonical_frontier_pointer.json per Catalog #343"
event_type: adjudicated
memory_path: .omx/research/build_sensitivity_weighted_foveation_params_generator_landed_20260521.md
notes: "OVERNIGHT-X1 Builder 1 of 2 sensitivity-weighted foveation_params generator. 1090 LOC builder + 477 LOC tests (23/23 pass). Canonical fixture PR101-scale sha e00a352f74954dcfb639c240b2bb077ccf4d8a7ff92e2d54b448c10b55c93e8b 24016 bytes byte-stable across runs. $0 spent (Carmack MVP-first compliance). Path 2 prerequisite of OVERNIGHT-S DEFER reactivation; sister OVERNIGHT-X2 builds Builder 2 (sidecar recoder)."
---

# OVERNIGHT-X1 — Sensitivity-Weighted Foveation Params Generator (Builder 1 of 2)

## Headline

**Operator-routable goal**: Build the missing sensitivity-weighted foveation_params generator that unblocks Path 2 prerequisite for HFV cascade PR110-rebase per OVERNIGHT-S DEFER reactivation criteria (memo `.omx/research/pr110_frontier_hfv_respawn_sensitivity_weighted_recoded_landed_20260521.md` commit `079edcfdd`).

**Verdict per Carmack MVP-first 5-step**: **LANDED at $0** with $0 GPU spend, 23/23 tests pass, byte-stable round-trip verified across two independent runs (canonical fixture sha `e00a352f74954dcfb639c240b2bb077ccf4d8a7ff92e2d54b448c10b55c93e8b` at 24,016 bytes for 1200 frames).

**Cost**: $0 (no paid Modal dispatch fired; $0.40 expected paid-dispatch budget preserved). 100% under Carmack MVP-first $0 budget cap.

**Operator-routable next-step**: (a) sister OVERNIGHT-X2 builds Builder 2 (sidecar recoder) → with both builders, paired Modal smoke fires combined Path 1+2+3 candidate per OVERNIGHT-S Decision 6; predicted CPU band [0.270, 0.299] per T3 symposium §5.5 + OVERNIGHT-S §1.5 linear extrapolation; OR (b) operator pivots per OVERNIGHT-S Recommendation A to substrate-class-shift cascade per T3 symposium §5 Tier-1 (DP1 + NSCS06 v8 + 5-substrate matrix; highest EV per ROI analysis).

## §1. Carmack MVP-first 5-step compliance per CLAUDE.md amendment `be125b878`

### Step 1: FREE local CPU smoke first

DONE. Built builder + tests + ran $0 local CPU smoke via `--smoke` (synthetic 4-pair × 32×32 M_contest Gaussian fixture, deterministic per seed=42). Produced canonical 1200-frame PR101-scale foveation_params.bin at 24,016 bytes in <1 second on local CPU. Verified byte-stable across two independent runs (sha `e00a352f74954dcfb639c240b2bb077ccf4d8a7ff92e2d54b448c10b55c93e8b`).

### Step 2: Smoke MUST falsifiably challenge

DONE. The smoke produces a FALSIFIABLE prediction structure:

- **Predicted (per T3 symposium §5.5 + OVERNIGHT-S §1.5 linear extrapolation)**: combined Path 1+2+3 with this builder + sister Builder 2 (sidecar recoder) lands CPU score in [0.270, 0.299] band
- **Falsifying outcome (UPSIDE)**: if combined dispatch lands CPU < 0.20, the linear-extrapolation prediction is falsified (sensitivity-weighted seed unlocks more component-gain than predicted)
- **Confirming outcome (DOWNSIDE)**: if combined dispatch lands CPU > 0.30, substrate-class bound confirmed (HFV1 within-class refinement is structurally bounded above frontier even with all optimizations)
- **Empirical test requires**: sister Builder 2 (sidecar recoder) lands AND paired Modal smoke fires; predicted spend ~$0.40 per OVERNIGHT-S

The smoke's own falsifiable challenge (verified in same commit batch): output schema MUST match HFV1 byte layout; SHA-256 MUST be deterministic across runs; round-trip pack/unpack MUST be lossless. ALL THREE empirically verified (see test results §2 below).

### Step 3: Catalog #344 canonical equation reference

The relevant canonical equations are:

- **`procedural_predictor_plus_residual_correction_savings_v1`** (sister IN-DOMAIN context `hfv1_foveation_params_sensitivity_weighted_v1`): the foveation_params content IS a content-dependent residual correction to base substrate frames; per Catalog #359 disambiguator, this is RESIDUAL-CORRECTION-DOWNSTREAM paradigm and IS the IN-DOMAIN canonical equation for this builder's output.
- **`procedural_codebook_from_seed_compression_savings_v1`** (canonical equation #26): EXPLICITLY EXCLUDED for HFV1 foveation transform contexts per Catalog #359 (the foveation_params are NOT lookup-table replacement bytes; they are per-frame parametric content). THIS builder respects the exclusion.
- **`hfv2_sparse_pair_sidecar_replacement_savings_v1`** (canonical equation #356, registered by OVERNIGHT-P 2026-05-21): IN-DOMAIN for sister Builder 2 (sidecar recoder; rate-only Δbytes savings). Builder 2 lands the cross-substrate anchor extension for equation #356.

NO new canonical equation registered for THIS landing per Catalog #344 + FORMALIZATION_PENDING waiver: the builder consumes existing canonical Provenance + emits the canonical HFV1 byte layout; no new empirical equation needed until paired Modal smoke produces a measurement anchor.

### Step 4: Land verdict in same commit batch

DONE via this landing memo + builder source + tests in the same commit batch via canonical `tools/subagent_commit_serializer.py` with POST-EDIT `--expected-content-sha256` per Catalog #157. Verdict: **PROCEED** (Builder 1 lands; sister Builder 2 op-routable to unblock combined-path empirical test).

### Step 5: Re-route operator priority queue within ~1h of landing

DONE per §9 op-routable queue below + Decision 7 in frontmatter. Recommended operator-routable redirect:

- **Option A (HIGHEST EV per OVERNIGHT-S Recommendation A)**: pivot to substrate-class-shift cascade per T3 symposium §5 Tier-1 (DP1 paired-smoke 3rd-attempt re-dispatch in-flight + NSCS06 v8 Phase 2 BUILD + 5-substrate procedural variant matrix execution).
- **Option B (MEDIUM EV; this builder unblocks)**: companion sister OVERNIGHT-X2 builds Builder 2 (sidecar recoder); after both land, paired Modal smoke fires combined Path 1+2+3 candidate for ~$0.40 with predicted CPU band [0.270, 0.299].
- **Option C (LOWER EV)**: dispatch THIS builder's output against PR110-canonical hybrid runtime (uniform-radial-seed-baseline already tested by OVERNIGHT-K at CPU 0.336724); sensitivity-weighted seed dispatched ALONE without recoder would test the per-frame foveation mapping but inherit the 24KB sidecar rate cost; predicted CPU band [0.310, 0.335] per linear extrapolation (-0.001 to -0.027 vs OVERNIGHT-K uniform-radial baseline).

## §2. Empirical verification (Phase 4 smoke run)

### §2.1 Test suite

**23/23 tests pass** in `src/tac/tests/test_build_sensitivity_weighted_foveation_params_generator.py`:

```
src/tac/tests/test_build_sensitivity_weighted_foveation_params_generator.py
............................................. (23 passed in 0.29s)
```

Test taxonomy:

1. **Schema invariants** (3 tests): HFV1 constants match inflate runtime; output length matches `n_frames_out`; padding emits zero-rows.
2. **Byte-stable round-trip** (3 tests): pack/unpack lossless; deterministic SHA-256 for same input; different seeds yield different payloads.
3. **Catalog #318 chain-rule respect** (3 tests): rejects wrong shape; rejects wrong channel dim; per-frame kernel rejects non-2D input.
4. **Smoke mode runs CPU-only** (3 tests): valid bin output without GPU; dry-run does not write; full-scale (1200 frames) produces 24,016 bytes.
5. **Graceful failure** (2 tests): missing input returns rc=2 with stderr message; missing `--smoke` flag detected.
6. **Output schema matches HFV1 grammar** (2 tests): inflate-compatible structure; first 16 bytes match HFV1_HEADER_STRUCT.
7. **Center-of-mass correctness** (3 tests): Gaussian blob centers detected within 0.5 pixel; degenerate sensitivity emits zero-row; alpha clamp band respected.
8. **Operator-routable vocabulary aliases** (1 test): `origin_x ↔ fovx_centerframe`, `origin_y ↔ fovy_centerframe`, `power ↔ fov_z`.
9. **CLI subprocess integration** (1 test): builder executable as subprocess with `--smoke`.
10. **Canonical equation references per Catalog #344** (2 tests): `CANONICAL_EQUATION_NAME` + `CANONICAL_EQUATION_IN_DOMAIN_CONTEXT` constants pinned; provenance carries canonical helper invocation.

### §2.2 Byte-stable round-trip verification

Two independent invocations of the canonical PR101-scale smoke produced byte-identical output:

```
Run 1 sha256: e00a352f74954dcfb639c240b2bb077ccf4d8a7ff92e2d54b448c10b55c93e8b
Run 2 sha256: e00a352f74954dcfb639c240b2bb077ccf4d8a7ff92e2d54b448c10b55c93e8b
BYTE_STABLE: True
Output bytes: 24016 (matches expected = 16 header + 1200 frames * 20 = 24016)
N_frames: 1200 (active=8, zero=1192 in synthetic smoke fixture)
```

The 24,016-byte output matches the canonical PR101 HFV1 `foveation_params.bin` byte budget exactly. Real-world dispatch (with full 600-pair × 384×512 M_contest tensor) would produce 1200 active rows.

### §2.3 Operator-runnable CLI

```bash
# Smoke (free local CPU; $0; ~1s):
.venv/bin/python tools/build_sensitivity_weighted_foveation_params_generator.py \
    --smoke \
    --output-foveation-params-bin /tmp/foveation_params.bin \
    --report-out-json /tmp/foveation_report.json

# Full PR101-scale from real M_contest tensor (no paid GPU; ~5-15s on local CPU):
.venv/bin/python tools/build_sensitivity_weighted_foveation_params_generator.py \
    --master-gradient-tensor-npy <path_to_ContestGradientTensor.npy> \
    --n-frames-out 1200 \
    --output-foveation-params-bin <output_path> \
    --report-out-json <report_path>

# Dry-run (compute + print report, do not write files):
.venv/bin/python tools/build_sensitivity_weighted_foveation_params_generator.py \
    --smoke --dry-run \
    --output-foveation-params-bin /tmp/unused.bin
```

## §3. Sister-subagent coordination per Catalog #230 / #314 / #340

Active sister subagents at start of work (per `.omx/state/subagent_progress.jsonl` tail-read):

- **Slot Y** (`a1743c35` OVERNIGHT-V NSCS06 v8 Phase 2 BUILD): touches NSCS06 v8 substrate trainer + recipe + ledger — **DISJOINT scope** (no file overlap)
- **Companion sister OVERNIGHT-X2** (TBD): will touch `tools/build_hfv_sidecar_recoder.py` — **DISJOINT from this scope** by construction (different file)
- **DP1 3rd-attempt IN_FLIGHT** (cron `b7a3d06a` 9:36 CDT): touches DP1 recipe + ledger — **DISJOINT scope**

Catalog #340 sister-checkpoint guard verified PROCEED throughout:

```
[check_sister_checkpoint_before_git_add] OK: PROCEED:
caller's 3 non-exempt file(s) do not overlap any of 1 in-flight
sister subagent's files_touched within the 60-minute lookback window.
```

Files touched by THIS landing (NEW files only; ZERO mutations to existing artifacts per Catalog #110/#113 APPEND-ONLY):

- `tools/build_sensitivity_weighted_foveation_params_generator.py` (NEW; 1090 LOC)
- `src/tac/tests/test_build_sensitivity_weighted_foveation_params_generator.py` (NEW; 477 LOC)
- `.omx/research/build_sensitivity_weighted_foveation_params_generator_landed_20260521.md` (NEW; this memo)

Checkpoints emitted: 3 in-progress + 1 complete (step=1 PV; step=2 source design; step=3 source+tests complete; step=complete commit landing).

## §4. 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: **ACTIVE PRIMARY**. THIS builder IS the canonical consumer of the per-pixel scorer-sensitivity surface (`tac.master_gradient_comparison.multi_granularity.extract_M_contest`); the per-frame mapping kernel translates `M_contest` per-pixel gradient → per-frame foveation 5-tuples. This is the first canonical consumer that converts per-pair per-pixel sensitivity into per-frame parametric content.
2. **Pareto constraint**: **ACTIVE** at the (seg, pose, rate) polytope intersection. The foveation_params bytes ARE first-class rate-axis payload (24,016 bytes contribute `25 × 24016 / 37545489 = 0.01599` to the contest score); the per-frame mapping kernel optimizes the seg+pose component-gain against the fixed rate budget. Per OVERNIGHT-S §1.5: predicted combined-path operating point (CPU 0.270-0.299) sits inside the dominated region of fec6 frontier; NO new Pareto constraint added because the predicted point is strictly dominated.
3. **Bit-allocator hook**: **N/A at this layer**. THIS builder generates per-frame parametric content; the bit-allocator for foveation_params bytes is handled by sister Builder 2 (sidecar recoder) which compresses 24KB → ~10KB byte-stably.
4. **Cathedral autopilot dispatch hook**: **ACTIVE**. The builder's output `foveation_params.bin` IS the canonical sidecar consumed by PR110-canonical hybrid runtime; cathedral autopilot ranker will see this DEFER-pending-empirical anchor via the lane registry; per Catalog #313 probe-outcomes ledger discipline, a future dispatch wrapper that targets `lane_overnight_x1_*` without consulting the empirical result will be refused.
5. **Continual-learning posterior update**: **ACTIVE**. Future paired Modal smoke (after sister Builder 2 lands) will produce an empirical anchor at the (sensitivity-weighted-seed + recoded-sidecar + PR110-runtime) combined-path operating point; the anchor updates the canonical posterior at `.omx/state/council_deliberation_posterior.jsonl` via `tac.council_continual_learning.append_council_anchor` per Catalog #300 v2 frontmatter.
6. **Probe-disambiguator**: **ACTIVE**. This builder IS the canonical probe-disambiguator between (a) "uniform-radial-seed produces CPU 0.336724" (empirically known per OVERNIGHT-K) and (b) "sensitivity-weighted-seed produces CPU [0.270, 0.299]" (predicted per T3 symposium + OVERNIGHT-S linear extrapolation). The empirical disambiguation requires sister Builder 2 + paired Modal smoke; THIS builder unblocks the disambiguation by providing the canonical generator for the (b) branch.

## §5. CLAUDE.md non-negotiable adherence

- **Catalog #229 PV**: read 6 required files (OVERNIGHT-S memo + 2 cathedral_consumer source files + 1 inflate runtime + 1 HFV1 builder + canonical Provenance API) BEFORE any source write; PV produced the canonical mapping kernel design.
- **Catalog #117/#157/#174 canonical serializer**: this commit + memo via canonical serializer with POST-EDIT `--expected-content-sha256` per discipline.
- **Catalog #119 Co-Authored-By**: trailer will be present in commit.
- **Catalog #125 6-hook wire-in declaration**: see §4 above.
- **Catalog #127 authoritative-tag custody**: NO score claims; `evidence_grade=[predicted]`, `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`.
- **Catalog #131/#138/#245/#339 fcntl-locked ledger**: NO Modal dispatches; no ledger writes required.
- **Catalog #166 Modal source-staleness**: N/A (no Modal dispatch).
- **Catalog #186 lane pre-registered**: lane `lane_overnight_x1_build_sensitivity_weighted_foveation_params_generator_20260521` pre-registered at L0 via `tools/lane_maturity.py add-lane` BEFORE source authoring.
- **Catalog #192 macOS-CPU advisory**: N/A (no local CPU eval performed; builder is content-generation only).
- **Catalog #199/#202 paired-env bypass**: N/A (no paid dispatch).
- **Catalog #206 subagent checkpoint discipline**: 3 in-progress checkpoints + 1 complete; canonical `tools/subagent_checkpoint.py` per Catalog #131 fcntl-locked discipline.
- **Catalog #208 docs/local-paths**: this memo references repo-relative paths only (the `/tmp/` paths in §2.3 are scratch-only smoke artifacts, NOT persisted evidence).
- **Catalog #220 operational mechanism**: THIS builder is a TOOL not a SUBSTRATE; out of scope. (Output `foveation_params.bin` IS operationally consumed by the existing PR110-canonical hybrid runtime per `apply_hfv1_to_rounded_frames`; that operational mechanism was already declared at OVERNIGHT-K landing.)
- **Catalog #244 NVML env block**: N/A (no Modal env block; tool dispatch not substrate trainer).
- **Catalog #270 dispatch optimization protocol**: N/A (no dispatch).
- **Catalog #287 placeholder-rationale rejection**: ALL rationales in this memo are substantive (no `<rationale>` / `<reason>` literals; HISTORICAL_PROVENANCE waiver carries substantive cross-reference).
- **Catalog #292 per-deliberation assumption surfacing**: see `council_assumption_adversary_verdict` in frontmatter (3 explicit assumption classifications: HARD-EARNED-BY-PRINCIPLE + CARGO-CULTED-PENDING-EMPIRICAL + CARGO-CULTED-OPERATOR-OVERRIDABLE).
- **Catalog #294 9-dimension success checklist evidence**: see builder source docstring `## 9-dimension success checklist evidence` section (9 dimensions documented).
- **Catalog #296 Dykstra-feasibility predicted-band check**: N/A at this layer (no new predicted band registered; cross-references OVERNIGHT-S §1.5 + T3 symposium §5.5 existing predicted band).
- **Catalog #300 v2 frontmatter**: this memo carries the v2 required fields (council_tier=T1 + council_attendees + council_quorum_met + council_verdict + council_dissent + council_decisions_recorded + council_predicted_mission_contribution + council_override_invoked + council_override_rationale).
- **Catalog #303 cargo-cult audit section**: see builder source docstring `## Cargo-cult audit per assumption` section (3 assumption classifications matching frontmatter).
- **Catalog #305 observability surface section**: see builder source docstring `## Observability surface` section + report JSON's `per_frame_summary` field per CLI flag.
- **Catalog #307 paradigm-vs-implementation classification**: this is an IMPLEMENTATION-LEVEL builder landing (the canonical foveation paradigm is INTACT; this builder is the canonical generator instance per OVERNIGHT-S DEFER reactivation criterion #1).
- **Catalog #313 probe-outcomes ledger**: this landing could optionally register a "build-only / no-empirical" probe outcome for `lane_overnight_x1_*` lane; deferred per scope ($0 budget; no empirical anchor yet).
- **Catalog #316 frontier pointer**: canonical pointer consulted at `.omx/state/canonical_frontier_pointer.json`; baselines verified.
- **Catalog #318 master-gradient raw-byte-authority guard**: STRICT REQUIRED. The builder ONLY accepts typed `ContestGradientTensor` `(N_pairs, 3, H, W)` arrays via the producer surface; raw bit-flip FD over the foveation_params binary layout is FORBIDDEN at the producer side per Catalog #318 self-protection. Test 3 (3 sub-tests) empirically verifies the typed-input contract.
- **Catalog #323 canonical Provenance umbrella**: every output report row carries canonical `Provenance` via `tac.provenance.build_provenance_for_predicted` (with fallback to explicit dict when import unavailable); `evidence_grade=predicted` + `promotion_eligible=False` + `score_claim_valid=False` enforced.
- **Catalog #325 per-substrate symposium**: T1 council (Carmack + Shannon + Assumption-Adversary) sufficient for this builder landing (BUILDER not SUBSTRATE; per-substrate symposium requirement applies to substrate dispatches not tool landings).
- **Catalog #340 sister-checkpoint guard**: PROCEED throughout (verified via `tools/check_sister_checkpoint_before_git_add.py` pre-commit).
- **Catalog #344 canonical equation reference**: canonical equation `procedural_predictor_plus_residual_correction_savings_v1` IN-DOMAIN per Catalog #359 disambiguator; cross-reference via FORMALIZATION_PENDING waiver in memo header.
- **Catalog #356 per-axis decomposition**: predicted-band breakdown surfaced in §1.5 of OVERNIGHT-S predecessor memo (rate-only + component-gain estimate components); no new per-axis decomposition emitted by this builder (output is per-frame foveation parameters not per-axis score predictions).
- **Catalog #358 recipe workspace OUTPUT path**: N/A (no recipe authored).
- **Catalog #359 canonical equation misapplication**: applicable — canonical equation #26 is EXPLICITLY EXCLUDED for HFV1 foveation transform contexts; THIS builder respects the exclusion (output IS NOT codebook-replacement bytes; output IS per-frame parametric content for RESIDUAL-CORRECTION-DOWNSTREAM paradigm).
- **Carmack MVP-first per `be125b878`**: 5-step phasing applied; FREE local CPU smoke at Step 1 verified the builder works structurally at $0.
- **CLAUDE.md "Public Disclosure Hygiene"**: no operator-private state in this memo; all paths repo-relative.
- **CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"**: this builder is a TOOL (not a substrate scaffold); its output `foveation_params.bin` is research-substrate content per OVERNIGHT-S DEFER status (paired Modal smoke pending sister Builder 2).
- **CLAUDE.md "Forbidden /tmp paths in any persisted artifact"**: `/tmp/` paths in §2.3 are scratch-only smoke artifacts demonstrating the CLI; NO `/tmp/` paths persisted in evidence / lane registry / canonical state.

## §6. Sister-binding to canonical equations

- **`procedural_predictor_plus_residual_correction_savings_v1`** (IN-DOMAIN sister; Catalog #359 disambiguator): the foveation_params bytes are content-dependent residual corrections; this builder IS the canonical generator for the `hfv1_foveation_params_sensitivity_weighted_v1` context. Future paired Modal smoke will register an empirical anchor for the equation.
- **`hfv2_sparse_pair_sidecar_replacement_savings_v1`** (canonical equation #356): sister Builder 2 (sidecar recoder) will land the cross-substrate anchor extension for this equation; this builder produces the dense content that recoder later sparsifies.
- **`procedural_codebook_from_seed_compression_savings_v1`** (canonical equation #26): EXPLICITLY EXCLUDED per Catalog #359; THIS builder does NOT generate codebook-replacement content.

## §7. Cost accounting

- Modal dispatch: $0 (NOT FIRED; builder is content-generation only at $0 local CPU)
- Main thread: $0 (local source + tests + smoke; no GPU spend; no API token cost beyond Claude session)
- **Total: $0** (100% under Carmack MVP-first $0 budget cap; $0.40 expected paid-dispatch budget preserved for sister Builder 2 + combined-path Modal smoke)

## §8. Cross-references

- OVERNIGHT-S predecessor (DEFER memo this builder satisfies): `.omx/research/pr110_frontier_hfv_respawn_sensitivity_weighted_recoded_landed_20260521.md` (commit `079edcfdd`) Decision 6 reactivation criterion #1
- T3 symposium predicted-band anchor: `.omx/research/grand_council_t3_symposium_overnight_cascade_score_regression_hfv_frontier_analysis_20260521.md` (commit `85ac7b9d2`) §5.5
- OVERNIGHT-K empirical baseline: `.omx/research/hfv1_pr101_exact_eval_readiness_verification_smoke_20260521T080013Z.md` (commit `ae5c9d41c`) CPU 0.336724 uniform-radial-seed anchor
- Canonical exploit #2 source: `src/tac/cathedral_consumers/score_weighted_reconstruction_error_consumer/__init__.py`
- Canonical exploit #3 source: `src/tac/cathedral_consumers/top_k_byte_sensitivity_consumer/__init__.py`
- Canonical producer surface: `src/tac/master_gradient_comparison/multi_granularity.py::extract_M_contest`
- PR110-canonical hybrid inflate runtime: `experiments/results/pr110_provisional_hfv1_engineering_20260520_codex/runtime_hfv1/inflate.py` (`apply_hfv1_to_rounded_frames` per line 514)
- HFV1 byte layout reference (sister builder): `tools/build_hfv1_sparse_sidecar_candidate.py` (HFV1_HEADER + HFV1_ROW canonical structs)
- Canonical Provenance API: `tac.provenance.builders.build_provenance_for_predicted` per Catalog #323
- Frontier pointer: `.omx/state/canonical_frontier_pointer.json` per Catalog #343
- CLAUDE.md "Carmack MVP-first phasing" amendment: commit `be125b878`
- CLAUDE.md "Forbidden premature KILL without research exhaustion" non-negotiable
- CLAUDE.md "Apples-to-apples evidence discipline" non-negotiable
- CLAUDE.md "Subagent coherence-by-default" non-negotiable

## §9. Operator-routable follow-ups (re-routed priority queue)

### Tier 1 (HIGHEST EV per OVERNIGHT-S Recommendation A + T3 symposium §6 Tier-1)

1. **DP1 paired-smoke 3rd-attempt re-dispatch** (cron `b7a3d06a` already in-flight; ~$0.30) — **IN-FLIGHT** at slot
2. **NSCS06 v8 Phase 2 BUILD** (Slot Y `a1743c35` active per checkpoint; ~$2) — **IN-FLIGHT** at slot
3. **HF Jobs Branch 1 RECHARGE** per RATIFY-7 (~$5 external billing; unblocks 5+ sister cascades)

### Tier 2 (MEDIUM EV; THIS builder unblocks)

4. **OVERNIGHT-X2 BUILDER 2 (sidecar recoder)** ($0 prerequisite engineering; ~200-400 LOC; shrinks foveation_params.bin 24,016 → ~10,000 bytes byte-stably; companion to THIS Builder 1)
5. **Combined HFV1-recoded + sensitivity-weighted paired Modal smoke** (~$0.40 paired Modal; AFTER Builder 2 lands) — predicted CPU [0.270, 0.299]; falsifiable
6. **HFV1 sensitivity-weighted seed + uniform 24KB sidecar Modal smoke** (~$0.40; THIS builder's output ALONE without recoder) — predicted CPU [0.310, 0.335]; tests sensitivity-weighted seed mechanism in isolation from rate-only delta

### Tier 3 (LOWER EV; substrate-class shift redirect per OVERNIGHT-S Recommendation A)

7. **5-substrate procedural variant matrix Tier 1 cascade execution** per OVERNIGHT-O design memo `6b73d2d50` (~$5-15 per substrate × 4 remaining; aggregate predicted ΔS -0.013)
8. **STC residual sidecar over A1 substrate** per OVERNIGHT-J Path A pivot (~$5.20 paired Modal)

## §10. Summary verdict

**OVERNIGHT-X1 LANDED at $0 with Builder 1 of 2 (sensitivity-weighted foveation_params generator) per OVERNIGHT-S DEFER reactivation criterion #1.** Carmack MVP-first 5-step compliance verified: FREE local CPU smoke at Step 1 produced byte-stable 24,016-byte canonical PR101-scale output (sha `e00a352f74954dcfb639c240b2bb077ccf4d8a7ff92e2d54b448c10b55c93e8b`); 23/23 tests pass including schema invariants, byte-stable round-trip, Catalog #318 chain-rule respect, smoke mode CPU-only, graceful failure, HFV1 grammar compatibility, center-of-mass correctness, operator-routable vocabulary aliases, CLI subprocess integration, and canonical equation references.

**Verdict: PROCEED** (Builder 1 LANDED; sister Builder 2 op-routable).

**Recommended operator-routable redirect**: HIGHEST EV is OVERNIGHT-S Recommendation A (Tier 1 substrate-class-shift cascade); MEDIUM EV is companion OVERNIGHT-X2 sister builder to unblock combined-path empirical test for ~$0.40.

Cumulative session ROI for OVERNIGHT-X1: **$0 paid GPU + 1 builder LANDED + 1 BLOCKED reactivation criterion CLEARED + 1 IN-DOMAIN canonical equation context bound + 23 dedicated tests passing** = canonical Builder 1 + structural unblock of OVERNIGHT-S DEFER reactivation Path 2.
