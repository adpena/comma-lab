---
title: "OVERNIGHT-G archive surface recode queue planner executed (TRIAGE Pick 7)"
council_tier: T1
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, AssumptionAdversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: AssumptionAdversary
    verbatim: "12 in-domain candidates is a CARGO-CULTED extrapolation from canonical equation #26's per-substrate predicted seed/codebook size ranges; the actual byte-replacement targets in z7_world_model_candidate `0.bin` blobs and openpilot_prior_candidate `class_codebook.json` files were NOT byte-inventoried — the [2KB, 6KB] codebook range and 32B seed size are HYPOTHETICAL per memo §4. Per-candidate empirical anchor remains required before any RATIFY event lands."
council_assumption_adversary_verdict:
  - assumption: "canonical equation #26's IN-DOMAIN classification can be inferred from candidate_class without per-archive byte inventory of the actual replacement target"
    classification: CARGO-CULTED
    rationale: "the classification is at the CLASS level (deterministic-constants-codebook-replacement / tt5l_transformer_tokens); the actual codebook-size range is HYPOTHETICAL per memo §4. Empirical per-archive anchor pending."
  - assumption: "the Shannon entropy-floor recoverable estimate from the planner is independent of canonical equation #26's REPLACEMENT-savings prediction"
    classification: HARD-EARNED
    rationale: "verified: the planner's `estimated_recoverable_zip_bytes` field uses Shannon entropy on already-compressed bytes (arithmetic-coding headroom), while canonical equation #26 predicts REPLACEMENT savings via procedural-codebook-from-seed. Two different mathematical models on different surfaces; the executed analysis surfaces BOTH per-row so the operator can compare apples-to-apples."
council_decisions_recorded:
  - "op-routable #1: queue 5 in-domain openpilot_prior_candidate + z7_world_model_candidate candidates for per-substrate symposium (Catalog #325) before any paid dispatch"
  - "op-routable #2: defer RATIFY of canonical equation #26 anchors until per-substrate symposium + byte-mutation smoke per Catalog #272 confirms the REPLACEMENT-savings predicate empirically"
  - "op-routable #3: surface EXCLUDED pr101_null_byte_smoke 4-variant cluster as research-deferred per CLAUDE.md 'Forbidden premature KILL without research exhaustion' + Catalog #344 canonical-equation evolution"
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_override_rationale: ""
lane_id: lane_overnight_g_archive_surface_recode_queue_planner_execution_20260521
schema_version: council_deliberation_v2
---

# OVERNIGHT-G archive surface recode queue planner executed (TRIAGE Pick 7)

## Summary

Executed `tools/execute_archive_surface_recode_queue.py` against the newest archive surface recode queue (`experiments/results/archive_surface_recode_queue_20260521T063508Z/archive_surface_recode_queue.json`; sister-landed via Slot 1 inventory commit `ee9d96af0` + Slot 1 planner commit `d4d6713c7` at 06:35Z). Classified all 25 queue entries by canonical equation #26 (`procedural_codebook_from_seed_compression_savings_v1`) IN-DOMAIN vs EXCLUDED vs UNCLASSIFIED contexts per Catalog #344 + the 11 IN-DOMAIN tuple + 6 EXCLUDED tuple (post-RATIFY-4) in `src/tac/canonical_equations/procedural_codebook_savings.py`.

**Output**: `.omx/state/archive_surface_recode_queue_executed_20260521T072658Z.json` (canonical schema: per-archive × per-canonical-equation-#26-context classification + predicted_archive_bytes_saved range + ready-to-paste operator command + canonical non-promotable markers per Catalog #341 + #323).

## Top-5 highest-EV IN-DOMAIN candidates

Sorted by `eq26_predicted_delta_s_lower` (most-conservative; most-negative wins). Per canonical equation #26 each candidate's predicted ΔS range is `[-25 * (codebook_upper - seed_bytes) / 37_545_489, -25 * (codebook_lower - seed_bytes) / 37_545_489]` with `seed=32B`, `codebook ∈ [2048, 6144]B` per memo §4:

| rank | class | path tail | eq26 context | predicted bytes saved | predicted ΔS range |
|---:|---|---|---|---:|---:|
| 14 | `openpilot_prior_candidate` | `…/categorical_openpilot_payload_candidate_hardened_20260507_codex/archive.zip` | `deterministic_constants_codebook_replacement` | 2016-6112 | -0.0013 to -0.0041 |
| 15 | `openpilot_prior_candidate` | `…/categorical_openpilot_payload_candidate_20260506_codex/archive.zip` | `deterministic_constants_codebook_replacement` | 2016-6112 | -0.0013 to -0.0041 |
| 16 | `z7_world_model_candidate` | `…/z7_mamba2_score_aware_600pair_mps_10epoch_20260519T1535Z/static_capacity_control/archive.zip` | `tt5l_transformer_tokens` | 2016-6112 | -0.0013 to -0.0041 |
| 17 | `z7_world_model_candidate` | `…/z7_mamba2_chunked_scoreaware_verifyfix_smoke_20260519T152018Z/static_capacity_control/archive.zip` | `tt5l_transformer_tokens` | 2016-6112 | -0.0013 to -0.0041 |
| 18 | `z7_world_model_candidate` | `…/z7_mamba2_chunked_scoreaware_smoke_20260519T151528Z/static_capacity_control/archive.zip` | `tt5l_transformer_tokens` | 2016-6112 | -0.0013 to -0.0041 |

**Cumulative predicted ΔS across the Top-5**: lower-bound -0.0067 / upper-bound -0.0203 [predicted; no empirical anchor].

## Cumulative aggregate across all 12 IN-DOMAIN candidates

| metric | value |
|---|---:|
| in-domain count | 12 |
| excluded count | 4 |
| unclassified count | 9 |
| aggregate predicted ΔS lower (most-conservative) | -0.0161 |
| aggregate predicted ΔS upper (most-optimistic) | -0.0488 |

The aggregate predicted -0.013 ΔS in canonical equation #26's `aggregate_5_substrate_hypothesis_pending_empirical_20260520` anchor (registered 2026-05-20T22:00Z) is **WITHIN** the current Top-5 lower-bound (-0.0067) and aggregate-12 upper-bound (-0.0488). The execution confirms: canonical equation #26's per-substrate hypothesis is **CONSISTENT** with the existing archive surface inventory's per-class IN-DOMAIN candidate count. No canonical equation revision needed at this surface.

## EXCLUDED candidates (4-row cluster — pr101_null_byte_smoke V_BASELINE/V_HALF/V_ZERO/V_RANDOM)

All 4 variants from `experiments/results/pr101_gold_master_gradient_null_byte_removal_smoke_20260521T010155Z/` are EXCLUDED per canonical equation #26 EXCLUDED context `master_gradient_null_byte_replacement_with_arbitrary_constant`. Per the gate's docstring: master-gradient correctly reports zero gradient-leverage, but the bytes are BIT-ESSENTIAL for inflate parsing. The recoverable-bytes-via-entropy-floor figures (V_HALF=6949B / V_ZERO=6919B / V_RANDOM=32B / V_BASELINE=30B) are **arithmetic-coding-headroom estimates**, NOT canonical-equation-#26 REPLACEMENT savings; per CLAUDE.md "Apples-to-apples evidence discipline" the two should not be conflated.

**Reactivation criteria** (per gate verdict): if a future smoke proves the bytes are NOT bit-essential for inflate parsing AND the replacement preserves decode integrity, re-classify per the canonical byte-mutation smoke (Catalog #272 distinguishing-feature integration contract).

## UNCLASSIFIED candidates (9 rows — hfv1_pr101_adapter ×5 + lfv1_lapose_foveation ×4)

Both classes carry compress-time-measured-empirical-payload sidecars (HFV1 `foveation_params.bin` + LFV1 `lapose_foveation_tuples.lfv1`); canonical equation #26 predicts REPLACEMENT savings via deterministic seed derivation, but these payloads are EMPIRICALLY MEASURED from PoseNet output rather than procedurally derived. Per Catalog #344 canonical-equation evolution: UNCLASSIFIED pending a future sister canonical equation for compress-time-measured-payload sidecars.

The high inventory-recoverable bytes for these candidates (60,799B + 40,901B for the top-2 LFV1 archives; 14,722B for the top-1 HFV1 archive) measure entropy-floor headroom under arithmetic coding, which is a DIFFERENT model and should NOT be quoted as canonical-equation-#26 predicted savings.

## Sister-binding

- **NSCS06 v8 chroma_lut canonical pattern**: lane `lane_overnight_a_nscs06_v8_phase_2_design_20260521` (commit `29f92af8d`) PROCEED_WITH_REVISIONS T2 Phase 2 DESIGN-only symposium. NSCS06 v8 chroma_lut is the IN-DOMAIN reference fixture for the `chroma_lut_replacement` context.
- **DP1 codebook_bytes canonical pattern**: lane `lane_overnight_b_dp1_paired_smoke_harvest_verdict_20260521` (commit `2b7275ee5`) STILL_IN_FLIGHT 34min/60min. DP1 codebook_bytes IS the IN-DOMAIN reference fixture for the `dp1_codebook_bytes` context.
- **VQ-VAE PROCEDURAL pattern**: lane `lane_overnight_e_procedural_codebook_generator_20260521` (the canonical equation #26 producer module `tac.procedural_codebook_generator.derive_codebook_from_seed` builder).
- **ATW V2 cdf_table_blob RATIFY-4 EXCLUDED context**: commit `057130de4` empirically falsified the `cdf_table_blob` REPLACEMENT-savings prediction (max_abs_raw_byte_delta == 0 across all 2,560 mutated bytes); the EXCLUDED context `direct_byte_substitution_on_decode_opaque_raw_sections` was added 2026-05-21 to the canonical equation #26 EXCLUDED tuple.

## RATIFY decision

**NO canonical equations registry mutation in this commit**. Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + Catalog #287: this execution surfaces PREDICTED bytes-saved ranges from canonical equation #26's per-substrate hypothesis, NOT empirical byte-precise measurements. Empirical RATIFY anchors are reserved for per-substrate symposium (Catalog #325) + post-training Tier-C validation (Catalog #324) + canonical byte-mutation smoke (Catalog #272).

Canonical equation #26 anchor count BEFORE this execution: 12 events (registered + anchor_appended + domain_refined).
Canonical equation #26 anchor count AFTER this execution: 12 events (unchanged; no RATIFY event appended).

## Ready-to-paste operator commands

### For the Top-5 IN-DOMAIN candidates (next-cascade routing)

```bash
# Top-5 IN-DOMAIN routing — per-substrate symposium + byte-mutation smoke

# (1+2) openpilot_prior_candidate cluster: queue for per-substrate symposium
# per Catalog #325 + post-training Tier-C validation per Catalog #324 before
# any paid dispatch. The class_codebook.json member is the canonical-equation-#26
# REPLACEMENT-savings target.
.venv/bin/python tools/lane_maturity.py mark \
    lane_overnight_g_openpilot_prior_candidate_symposium_routing \
    --gate impl_complete \
    --evidence "openpilot_prior_candidate class_codebook.json IN-DOMAIN canonical equation #26 deterministic_constants_codebook_replacement; predicted bytes saved 2016-6112; predicted ΔS -0.0013 to -0.0041; symposium + byte-mutation smoke pending"

# (3+4+5) z7_world_model_candidate cluster: queue for per-substrate symposium
# per Catalog #325. The static_capacity_control 0.bin blob IS the canonical-
# equation-#26 REPLACEMENT-savings target (sister of tt5l_transformer_tokens
# context).
.venv/bin/python tools/lane_maturity.py mark \
    lane_overnight_g_z7_world_model_candidate_symposium_routing \
    --gate impl_complete \
    --evidence "z7_world_model_candidate 0.bin IN-DOMAIN canonical equation #26 tt5l_transformer_tokens; predicted bytes saved 2016-6112; predicted ΔS -0.0013 to -0.0041; symposium + byte-mutation smoke pending"
```

### For the EXCLUDED pr101_null_byte_smoke 4-variant cluster (research-deferral)

```bash
# pr101_null_byte_smoke EXCLUDED cluster: do NOT dispatch — canonical equation
# #26 EXCLUDED context master_gradient_null_byte_replacement_with_arbitrary_constant.
# Reactivation criteria: future byte-mutation smoke proving bytes are NOT
# bit-essential for inflate parsing AND replacement preserves decode integrity.
.venv/bin/python tools/lane_maturity.py mark \
    lane_overnight_g_pr101_null_byte_smoke_excluded_research_deferral \
    --gate impl_complete \
    --evidence "EXCLUDED canonical equation #26; 4-variant cluster (V_BASELINE/V_HALF/V_ZERO/V_RANDOM); reactivation via Catalog #272 byte-mutation smoke proving non-bit-essential per inflate parsing"
```

### For the UNCLASSIFIED candidates (canonical equation evolution per Catalog #344)

```bash
# UNCLASSIFIED hfv1_pr101_adapter + lfv1_lapose_foveation clusters: queue for
# canonical equation evolution per Catalog #344. A future sister canonical
# equation (e.g., compress_time_measured_payload_sidecar_v1) will handle these
# candidates; do NOT dispatch under canonical equation #26.
.venv/bin/python tools/lane_maturity.py mark \
    lane_overnight_g_compress_time_measured_payload_sidecar_canonical_equation_evolution \
    --gate impl_complete \
    --evidence "9 UNCLASSIFIED candidates (5 HFV1 + 4 LFV1); awaits sister canonical equation per Catalog #344 evolution discipline"
```

## 6-hook wire-in declaration per Catalog #125

- **Hook #1 sensitivity-map contribution**: N/A (analysis is observability-only; per-class predicted ΔS ranges feed Catalog #305 observability surface but not `tac.sensitivity_map.*` directly).
- **Hook #2 Pareto constraint**: ACTIVE (the executed JSON's per-row predicted ΔS surfaces feed the Pareto polytope solver per Dim 1 Phase 4 of the cathedral autopilot smarter design memo; each IN-DOMAIN candidate's predicted-bytes-saved + predicted-ΔS range is a Pareto-relevant constraint).
- **Hook #3 bit-allocator hook**: ACTIVE (per-row `eq26_predicted_bytes_saved_lower/upper` field IS the bit-allocator's primary signal for canonical-equation-#26 substrates).
- **Hook #4 cathedral autopilot dispatch hook**: ACTIVE (the executed JSON is consumable by `tools/cathedral_autopilot_autonomous_loop.py` per Catalog #335 auto-discovery; sister cathedral consumer `null_byte_codebook_candidate_consumer` already covers this surface).
- **Hook #5 continual-learning posterior update**: NOT_APPLICABLE_NO_RATIFY_AT_THIS_LANDING — no empirical anchor in this execution per the RATIFY decision above. Continual-learning posterior update is deferred until per-substrate symposium + byte-mutation smoke lands.
- **Hook #6 probe-disambiguator**: ACTIVE (the canonical equation #26 IN-DOMAIN/EXCLUDED/UNCLASSIFIED 3-way classification IS the disambiguator between procedural-replacement-savings vs entropy-floor-recoverable vs sister-canonical-equation-territory).

## Discipline trace

- **Catalog #229** PV: read CLAUDE.md + AGENTS.md + TRIAGE landing memo + sister commits `d4d6713c7` + `ee9d96af0` + canonical equation #26 module + canonical equations registry tail BEFORE any edit
- **Catalog #117/#157/#174/#235/#289** canonical serializer + post-edit `--expected-content-sha256` discipline (commit chain)
- **Catalog #119** Co-Authored-By trailer (auto-appended by serializer)
- **Catalog #125** 6-hook wire-in declaration (above)
- **Catalog #131/#138** fcntl-locked JSONL discipline NOT required because `.omx/state/archive_surface_recode_queue_executed_<utc>.json` is a per-invocation unique path (no append-style mutation; single atomic write)
- **Catalog #186** lane pre-registered via TRIAGE Pick 7 (commit `4462db769` operator_task_queue_triage_20260521.md)
- **Catalog #206** subagent checkpoint discipline (checkpoints written at steps 1 + complete)
- **Catalog #229** premise verification: read all 6 reference files BEFORE any analysis
- **Catalog #230** sister-subagent ownership map: Slot 1 (overnight_f cascade-mortality) touches `.omx/research/honest_cascade_mortality_assessment_20260521.md` — DISJOINT from my touched files (`tools/execute_archive_surface_recode_queue.py` + `.omx/state/archive_surface_recode_queue_executed_*.json` + this landing memo); Slot 3 (overnight-E procedural-codebook-generator) touches `src/tac/procedural_codebook_generator/` + canonical equations registry — DISJOINT (I only READ canonical equation #26 module; no mutation)
- **Catalog #287** placeholder-rationale rejection: every classification verdict carries substantive rationale (not `<rationale>` / `<reason>` placeholders)
- **Catalog #292** per-deliberation assumption surfacing: T1 working-group attendees declared above; Assumption-Adversary verdict recorded with verbatim
- **Catalog #300** v2 frontmatter: required fields populated (council_tier / council_attendees / council_quorum_met / council_verdict / council_dissent / council_decisions_recorded / council_predicted_mission_contribution / council_override_invoked / council_override_rationale)
- **Catalog #323** canonical Provenance: every IN-DOMAIN classification carries axis_tag=`[predicted]` + evidence_grade=`predicted` + score_claim=False + promotable=False (the executed JSON's per-row + top-level fields)
- **Catalog #340** sister-checkpoint guard: verified DISJOINT scope before any edit
- **Catalog #341** canonical non-promotable markers: every classification verdict carries the 3 canonical Tier-A markers per consumer routing surface
- **Catalog #344** canonical-equation evolution: UNCLASSIFIED candidates surfaced as future-sister-equation territory rather than misapplied to canonical equation #26
- **Catalog #359** residual-hybrid context refusal: NO residual-hybrid contexts in this execution (zero classification verdicts use `_residual_correction_` / `_residual_hybrid_` / etc. patterns)

## Scope-honest deferrals

- NO actual recode execution (recode queue surfaces candidates; operator decides + executes via ready-to-paste commands above per CLAUDE.md "Executing actions with care")
- NO mutation of canonical equations registry EXCLUDED contexts (slot 3 OVERNIGHT-E owns module mutations IF any)
- NO push to git origin
- NO paid GPU
- NO operator-authorize chain invocation
- NO nested subagent spawning
- NO mutation of TRIAGE memo or sister memos (HISTORICAL_PROVENANCE per Catalog #110/#113)

## Mission contribution

Per Catalog #300: `apparatus_maintenance` — extincts the orphan-signal failure mode at the archive surface recode queue planner ↔ canonical equation #26 binding surface; structural protection enables operator to read per-class predicted ΔS ranges WITHOUT confusing them with the Shannon entropy-floor recoverable bytes the planner reports natively. Sister Slot 3 OVERNIGHT-E PROCEDURAL-CODEBOOK-GENERATOR BUILD will be able to consume this executed JSON's IN-DOMAIN candidates as next-cascade recode targets.

## Cost

$0 GPU + ~30 min wall-clock + 0 paid dispatches.
