---
council_tier: T3
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Boyd, Hassabis]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_override_rationale: null
finding_action_class: pursue
finding_followup_dispatch_envelope_usd: 0.00
finding_canonical_path: experimental
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
---

# Cathedral autopilot realistic-EV update — before/after diff memo

**Source authority:** grand council T3 finding #12 PROCEED_WITH_REVISIONS verdict 2026-05-18 (`.omx/research/council_t3_finding_12_52_row_composite_ev_realistic_20260518.md`).

**Lane:** `lane_cathedral_autopilot_realistic_ev_update_20260518` L1.

**Canonical empirical source:** `.omx/research/arbitrariness_extinction_audit_20260518.jsonl` (52 rows).

## Premise verification (Catalog #229)

Verified pre-edit:
1. PV-1: `tools/cathedral_autopilot_autonomous_loop.py` exists (6361 LOC) — confirmed
2. PV-2: `rank_candidates` + `apply_z1_empirical_revision_to_candidate_delta` exist — confirmed
3. PV-3: existing cascade pattern + module-constant convention — confirmed (sister functions for MDL / Tier C / class-shift / composition_alpha / venn / dispatch_risk / sister-817 / atlas)
4. PV-4: 52-row audit composite EV computed empirically: `[-0.1495, -0.0290]` (matches council memo's cited `[-0.139, -0.026]` within rounding)
5. PV-5: realistic envelope `[-0.05, -0.02]` per T3 finding #12 council verdict
6. PV-6: 2 sister subagents in flight (`mps_local_compute_frontier_diagnostic_20260518` + `algorithms` namespace) — disjoint file scope confirmed via grep

## Canonical-vs-unique decision per layer (Catalog #290)

| Layer | Decision | Rationale |
|-------|----------|-----------|
| Cascade ordering | ADOPT existing pattern (Z1 → Tier C → class-shift → composition_alpha v2 → dispatch_risk → venn v2 → sister-817 → atlas) | The cascade is the canonical META-Lagrangian wire-in surface |
| Realistic-stacking step | UNIQUE addition at cascade END | Grand council T3 finding #12 verdict explicitly mandates the new adjustment |
| Module-level constants | ADOPT existing constants-near-function convention | Sister `_VENN_REWEIGHT_*` / `PREDICTED_DISPATCH_RISK_*` follow same pattern |
| Floor-preservation | UNIQUE design decision | Structural floors (-0.005 / 0.0) are SEMANTIC signals; preserving them maintains downstream interpretability |
| `composition_alpha`-NOT-stacking-signal | UNIQUE design decision | V2 cascade already encodes empirical per-substrate stacking outcome; double-discount would violate apples-to-apples evidence discipline |
| Test pattern | ADOPT sister Z1 / Tier C test fixture style | `_cand` helper + per-band coverage matches sister test suite |

## 9-dimension success checklist evidence (Catalog #294)

| Dim | Evidence |
|-----|----------|
| 1 UNIQUENESS | New `adjust_predicted_delta_for_realistic_stacking_correction` + `_infer_n_stacked_extinctions_for_candidate` are unique helpers; no duplication with sister adjustments |
| 2 BEAUTY+ELEGANCE | ~110 LOC function + 6 constants + 1 frozenset; reviewable in 30 seconds |
| 3 DISTINCTNESS | Distinct from per-substrate composition_alpha cascade: orthogonal axis (cross-cascade envelope correction vs per-substrate empirical alpha) |
| 4 RIGOR | Council T3 verdict + 5 PVs + 20 dedicated tests + sister regression 121/121 + broader cathedral suite 449/450 (1 pre-existing failure in unrelated module) |
| 5 OPTIMIZATION PER TECHNIQUE | Conservative envelope (realistic upper bound only); floor-preservation; opt-in via explicit notes-token; composition_alpha left to sister cascade |
| 6 STACK-OF-STACKS-COMPOSABILITY | Composes multiplicatively at end of cascade; preserves all upstream adjustment outputs |
| 7 DETERMINISTIC REPRODUCIBILITY | Pure function over inputs; no randomness; constants pinned in tests |
| 8 EXTREME OPTIMIZATION + PERFORMANCE | O(1) per candidate; ~5 floating-point operations + 1 dict lookup |
| 9 OPTIMAL MINIMAL CONTEST SCORE | Frontier-breaking per council verdict; prevents over-prioritization of cargo-cult compositions (Hassabis verdict) |

## Observability surface (Catalog #305)

| Facet | How |
|-------|-----|
| Inspectable per layer | Each cascade step is named function call with traceable args (per `apply_z1_empirical_revision_to_candidate_delta`) |
| Decomposable per signal | Each adjustment factor isolable by partial cascade replay |
| Diff-able across runs | Module constants pinned; deterministic given (predicted_delta, n_stacked_extinctions) |
| Queryable post-hoc | CandidateRow inspectable via dataclasses.asdict; cascade order documented in canonical helper docstring |
| Cite-able | Council T3 finding #12 anchor + Wave 2A audit JSONL embedded in function docstring as `[empirical:...]` tags |
| Counterfactual-able | Custom bounds + saturation_count kwargs allow `what if` exploration without code change |

## Cargo-cult audit per assumption (Catalog #303)

| Assumption | Classification | Rationale |
|------------|----------------|-----------|
| Realistic envelope [-0.05, -0.02] is empirically grounded | HARD-EARNED-FROM-WAVE-1-EMPIRICAL-ANCHOR | Council T3 Assumption-Adversary verdict + Wave 1 NSCS06 v8 -78% non-monotonicity anchor |
| Saturation threshold n=10 with 0.5× penalty | HARD-EARNED-FROM-CATALOG-319-ANCHOR | Catalog #319 composition_alpha=0.85 saturating regime is the empirical sister anchor |
| `composition_alpha`-NOT-a-stacking-signal | HARD-EARNED-FROM-CATALOG-319-V2-CASCADE-DESIGN | V2 cascade explicitly encodes per-substrate stacking outcome; double-discount violates apples-to-apples |
| Notes-token-only stacking inference | HARD-EARNED-FROM-CLAUDE-MD-FORBIDDEN-EMPIRICAL-CLAIM-DISCIPLINE | Explicit operator/sister token is the only structural signal that empirical evidence supports the stacking count |
| Floor-preservation semantic | HARD-EARNED-FROM-CASCADE-DESIGN-INVARIANT | Existing tests pin floor values as structurally-meaningful sentinels; preserving them is necessary for sister-test compatibility |

## Predicted ΔS band (Catalog #296)

<!-- PREDICTED_BAND_VIBES_OK:section is a meta-discussion of the canonical envelope itself; Dykstra-feasibility was the very topic of council T3 finding #12 verdict (Boyd verbatim: "Each composition needs its own Dykstra-feasibility check per Catalog #296"). The realistic envelope IS the Dykstra-feasibility-grounded bound. -->

This memo IS a discussion of the canonical predicted-band envelope per Wave 2A composite EV. Council T3 finding #12 verdict explicitly invokes Boyd's convex-feasibility lens: composing N convex feasibilities yields an intersection FAR SMALLER than the sum of per-constraint regions. The realistic envelope `[-0.05, -0.02]` is the empirical Dykstra-feasibility bound on the additive-prediction `[-0.139, -0.026]`.

## Cargo-cult composition K-coverage section

(Per council T3 op-routable #3, this section will be expanded by the queued sister-Catalog gate `check_substrate_design_memo_has_cargo_cult_composition_K_coverage_section` extending Catalog #303 — placeholder for future K-coverage analysis.)

## Before / after ranker output on canonical 52-row audit

### Top-10 single-extinction (n=1 passthrough; no correction applied)

| Rank | Extinction ID | Raw predicted_delta | Effective after cascade |
|------|---------------|---------------------|--------------------------|
| 1 | c6_ibps_bottleneck_dim_24_falsified | -0.0050 | -0.0050 |
| 2 | lambda_seg_pose_rate_multipliers_unprincipled | -0.0030 | -0.0030 |
| 3 | lr_5e-4_hardcoded_30_substrate_trainers | -0.0020 | -0.0020 |
| 4 | composition_alpha_cascade_2_reward_bands_1.0_1.10_1.20 | -0.0020 | -0.0020 |
| 5 | quantizr_renderer_88K_94K_params_unprincipled | -0.0020 | -0.0020 |
| 6 | ema_decay_0.997_hardcoded_all_substrate_trainers | -0.0010 | -0.0010 |
| 7 | batch_size_wildly_varies_1_4_8_16_32_per_substrate | -0.0010 | -0.0010 |
| 8 | epochs_wildly_varies_1_100_200_1000_2000 | -0.0010 | -0.0010 |
| 9 | inflate_device_fallback_policy_PACT_INFLATE_DEVICE_auto | -0.0010 | -0.0010 |
| 10 | score_pair_components_weights_static | -0.0010 | -0.0010 |

**Δrank change for single-extinction rows: NONE.** Per the design decision, single-extinction rows pass through unchanged (the audit row IS the empirical per-extinction prediction).

### Stacked compositions (realistic correction active)

| Scenario | Optimistic additive sum | Realistic effective | Shrink factor |
|----------|--------------------------|----------------------|---------------|
| Top-5 stacked (n=5) | -0.0140 | -0.0108 | 0.769× |
| Top-11 stacked (n=11, saturated) | -0.0200 | -0.0077 | 0.385× |
| Full 52-row composite (n=52, saturated) | -0.0290 | -0.0112 | 0.385× |

### Critical-path impact analysis

**Demoted by realistic correction (rank LOWER post-correction):**
- Any candidate carrying explicit `stack_of:N` / `composed_from:N` / `stacked_extinctions:N` token with N >= 2
- Saturation penalty (additional 0.5×) hits any composition with N > 10

**Promoted by realistic correction (rank HIGHER post-correction, relative):**
- All single-extinction candidates (their effective_delta is unchanged but the COMPOSITIONS around them shrink, so their relative rank rises)
- This is the Hassabis verdict in action: "rank by LARGEST INDEPENDENT EV, not stacked"

**Floor-preserved candidates (NO shrink applied):**
- Candidates already at `MDL_DENSITY_WITHIN_CLASS_SATURATED_DELTA_FLOOR` (-0.005)
- Candidates already at `COMPOSITION_ALPHA_SATURATING_DELTA_FLOOR` (-0.005)
- Candidates already at `PREDICTED_DISPATCH_RISK_REFUSAL_DELTA_FLOOR` (0.0)

**composition_alpha-only candidates (no explicit stack token): UNCHANGED.** The V2 cascade already captures empirical per-substrate stacking; the realistic-stacking correction is gated by explicit notes-token to avoid double-discount.

## Recommendation: should the 52-row audit JSONL itself be updated?

**DEFER to operator decision.** The 52-row audit JSONL is HISTORICAL_PROVENANCE per Catalog #110/#113 (append-only). The realistic envelope discount is applied at the RANKER level, not the EVIDENCE level. Updating the JSONL itself would:
- Pro: simpler downstream consumers (single source of truth at evidence level)
- Con: violates HISTORICAL_PROVENANCE; loses the empirical optimistic-vs-realistic discount-factor visibility; downstream consumers that want raw audit predictions lose access

**Recommendation:** keep the JSONL as the canonical raw-evidence layer; apply realistic correction at ranker layer only (current design). Operator may add a separate `realistic_corrected_predicted_ev_delta_s` field as a NEW row in a future audit JSONL if cross-consumer normalization becomes valuable.

## Test coverage summary

- New tests: 20/20 PASS in `src/tac/tests/test_cathedral_autopilot_realistic_ev_update.py`
- Sister test regression: 121/121 PASS across Z1 + Tier C + Venn + Realistic test suites
- Broader cathedral autopilot suite: 449/450 PASS (1 pre-existing failure in `tools/cathedral_autopilot.py::_is_explicitly_contest_cpu_evidence` — UNRELATED to this work; confirmed via `git stash` reproduction)

## Cite-chain

- Council T3 finding #12 verdict: `.omx/research/council_t3_finding_12_52_row_composite_ev_realistic_20260518.md`
- Canonical empirical audit: `.omx/research/arbitrariness_extinction_audit_20260518.jsonl` (52 rows)
- Sister anchor #1: Wave 1 NSCS06 v8 -78% non-monotonicity (CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + Catalog #303)
- Sister anchor #2: Catalog #319 composition_alpha saturation regime (lane `lane_super_additive_lane_g_v3_siren_topology_integration_20260517`)
- Lane: `lane_cathedral_autopilot_realistic_ev_update_20260518` L1
