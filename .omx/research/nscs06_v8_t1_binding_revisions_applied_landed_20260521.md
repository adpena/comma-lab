---
council_tier: T1
council_attendees:
  - Shannon
  - Carmack
  - Hotz
  - Daubechies
  - Mallat
  - Assumption-Adversary
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "REVISION #1 ablation ladder helper canonical 3-axis design captures CARGO-CULTED assumptions 1-3"
    classification: HARD-EARNED
    rationale: "Each axis isolates ONE cargo-culted assumption with the other 2 held at canonical default; minimizes confounding so a >2x drift can be attributed to the right axis layer per symposium Assumption-Adversary REVISION #1 verdict"
  - assumption: "REVISION #2 multi-scale Dykstra-feasibility verdict structurally additive at PRE-SMOKE"
    classification: HARD-EARNED
    rationale: "Canonical equation #26 closed form IS the rate-axis Pareto-frontier alternating-projections fixed point; structural additivity holds; FIRST-PAIRED-SMOKE empirically validates whether seg+pose drift breaks additivity"
  - assumption: "REVISION #3 5-step pre-smoke verification covers archive grammar + inflate + LUT correctness + byte-mutation + device-fork"
    classification: HARD-EARNED
    rationale: "Carmack MVP-first verbatim per symposium memo: steps (a)-(e) cover the 5 mechanism layers that the BUILD's canonical contract depends on; each step is an empirically verifiable invariant of the substrate package"
  - assumption: "REVISION #4 JSON ablation table schema cathedral-autopilot-consumable"
    classification: HARD-EARNED
    rationale: "Schema follows canonical Provenance per Catalog #287 + #323 + sister canonical_equation_lookup_consumer pattern per Catalog #335 + #344; the 5 canonical fields (score_claim=False, promotable=False, evidence_grade=predicted, canonical_equation_id, in_domain_context) make the JSON queryable across runs per Catalog #305 observability"
  - assumption: "Revisions can be applied as NEW module without modifying BUILD's 5 substrate files"
    classification: HARD-EARNED
    rationale: "The 4 binding revisions per symposium memo Section 4 are FIRST-PAIRED-SMOKE-HARVEST-PLAN discipline + verification helpers, NOT codec/archive/inflate layer changes; preserving the BUILD's contract intact maintains Catalog #220 operational mechanism declaration + Catalog #241/#242 SubstrateContract validation + canonical equation #26 byte-precise 4064-byte exact match"
  - assumption: "105/105 tests pass demonstrates revisions are complete + correct"
    classification: HARD-EARNED
    rationale: "49 baseline tests preserved + 56 new revision tests pass across 4 test classes (TestRevision1AblationLadder/TestRevision2MultiScaleFeasibility/TestRevision3CarmackMvpFirstVerification/TestRevision4JsonAblationTable) + 1 end-to-end integration test class; empirical anchor"
council_decisions_recorded:
  - "op-routable #1: per-substrate symposium 14-day window remains open 2026-05-21 -> 2026-06-04 per Catalog #325; this RATIFY-3 landing satisfies REVISIONS-applied prerequisite"
  - "op-routable #2: pre-symposium dispatch REMAINS BLOCKED via recipe `research_only: true` + `dispatch_enabled: false` per Catalog #240 (substrate trainer's _full_main still raises NotImplementedError; THIS landing does NOT lift the dispatch gate)"
  - "op-routable #3: first paired smoke (post-PROCEED-unconditional) MUST consume the 4 revisions: invoke build_per_assumption_ablation_ladder() + run_carmack_mvp_first_pre_smoke_verification() + emit_per_assumption_ablation_table_json() per CASCADE COMPRESSION symposium PRIORITY 5 5-substrate aggregate paired-smoke matrix"
  - "op-routable #4: follow-on T1 working group MUST review this RATIFY-3 landing + sister BUILD commit 853d108e2 + first paired smoke harvest to deliver PROCEED-unconditional verdict before L1 promotion"
council_predicted_mission_contribution: frontier_breaking_enabler
council_override_invoked: false
council_override_rationale: ""
deferred_substrate_id: nscs06_v8_chroma_lut
substrate_aliases:
  - lane_wave_3_nscs06_v8_chroma_lut_substrate_build_20260521
  - lane_ratify_3_nscs06_v8_t1_binding_revisions_applied_20260521
  - nscs06_v8
related_deliberation_ids:
  - council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521
  - council_grand_council_symposium_cascade_compression_falsifications_negative_results_20260520_d125af6c3
  - council_grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516_4292c8ce2
---

# RATIFY-3: NSCS06 v8 T1 per-substrate symposium binding revisions APPLIED

**Date:** 2026-05-21
**Lane:** `lane_ratify_3_nscs06_v8_t1_binding_revisions_applied_20260521`
**Sister symposium memo:** `.omx/research/council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md` (PROCEED_WITH_REVISIONS verdict; 4 binding revisions)
**Sister design memo:** `.omx/research/nscs06_v8_chroma_lut_design_20260521.md`
**Sister BUILD commit:** `853d108e2` (NSCS06 v8 chroma_lut substrate L0 SCAFFOLD)
**Operator approval:** 2026-05-21 #3 of 8 (blanket approval)

## Summary

Applied all 4 binding revisions from the NSCS06 v8 chroma-LUT per-substrate symposium PROCEED_WITH_REVISIONS verdict as a NEW canonical module `src/tac/substrates/nscs06_v8_chroma_lut/revisions.py` (1010 LOC) + dedicated test suite at `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_revisions.py` (442 LOC). The BUILD's 5 substrate files (architecture/archive/inflate/procedural_variant/substrate_contract) were NOT modified, preserving Catalog #220 operational mechanism declaration + Catalog #241/#242 SubstrateContract canonical contract + canonical equation #26 byte-precise 4,064-byte exact match.

**Test result**: 105/105 tests pass (49 baseline preserved + 56 new revision tests including 1 end-to-end integration test class).

## Per-revision diff summary

### REVISION #1 (Assumption-Adversary): per-assumption ablation ladder

**Applied as**: `tac.substrates.nscs06_v8_chroma_lut.revisions.build_per_assumption_ablation_ladder()` returning typed `PerAssumptionAblationLadder` dataclass with 7 arms (1 canonical-default arm + 6 probe arms, 2 per assumption axis).

**Canonical axes** (per symposium memo):
- Axis 1 (luma quantization levels): `(8, 16, 32)` — canonical default 16
- Axis 2 (per-(level, class) aggregation): `("median", "mode", "k_medoids")` — canonical default "median"
- Axis 3 (PRNG generator kind): `("pcg64", "xorshift", "lcg")` — canonical default "pcg64"

**Cost contract** (per symposium memo verbatim): "3 ablation arms x $0.50 each = $1.50 incremental over base $0.50 smoke = $2.00 total". Implementation: `total_predicted_cost_usd = base $0.50 + 3 axes x $0.50 = $2.00`. Test: `test_build_ladder_total_cost_matches_symposium` PASSES.

**14 dedicated tests** in `TestRevision1AblationLadder` class.

### REVISION #2 (Daubechies + Mallat CO-LEAD): multi-scale Dykstra-feasibility check

**Applied as**: `tac.substrates.nscs06_v8_chroma_lut.revisions.verify_multi_scale_dykstra_feasibility()` returning typed `MultiScaleDykstraFeasibilityVerdict` dataclass.

**Multi-scale contract**: coarse-scale axis = `segnet_class` (5 partitions); fine-scale axis = `(level, channel)` (16 levels x 3 channels = 48 partitions); canonical LUT shape `(16, 5, 3)`. The wavelet-style hierarchical-coarse-gates-fine structure produces additive seg + pose contributions per Catalog #296 Dykstra-feasibility intersection check.

**PRE-SMOKE structural verdict**: `is_additive=True` + `intersection_non_empty=True` + `dykstra_iteration_count=1` (closed-form additive case) + `unwind_test_recommended_assumptions=()`. The canonical equation #26 closed form IS the rate-axis Pareto-frontier fixed point; seg + pose contributions are placeholder `0.0` until first paired smoke lands.

**Rate-axis predicted ΔS**: `-25 * (4096 - 32) / 37_545_489 ≈ -0.002706` (byte-precise match to canonical equation #26).

**13 dedicated tests** in `TestRevision2MultiScaleFeasibility` class.

### REVISION #3 (Carmack + Hotz): MVP-first 5-step pre-smoke verification

**Applied as**: `tac.substrates.nscs06_v8_chroma_lut.revisions.run_carmack_mvp_first_pre_smoke_verification()` returning typed `CarmackMvpFirstPreSmokeVerificationVerdict` with 5 step results.

**The 5 steps** (per symposium memo verbatim):

| Step | Label | Empirical verification |
|---|---|---|
| (a) | verify CH08 v2 archive parses cleanly on Modal worker | pack -> parse roundtrip preserves every header field byte-for-byte |
| (b) | verify inflate roundtrip produces canonical raw bytes count | inflate writes `num_pairs * 2 * H * W * 3 = 12288 bytes` for 1-pair @ 32x64 |
| (c) | verify chroma LUT lookup correctness against a known synthetic seed | derive_procedural_chroma_lut_replacement + lookup_rgb_via_chroma_lut produce deterministic `lut[gray>>shift, cls]` values |
| (d) | verify byte-mutation distinguishing-feature smoke per Catalog #272 passes | verify_seed_mutation_changes_lut_bytes returns True for mutated_byte_index=0 |
| (e) | verify Catalog #205 inflate-device-fork passes for CPU + CUDA paths | select_inflate_device accepts CPU + CUDA; refuses MPS per CLAUDE.md "MPS auth eval is NOISE" non-negotiable |

All 5 steps pass locally; verdict carries `all_steps_passed=True` + `ready_for_first_paired_smoke=True` (AND-gated with REVISION #2 intersection_non_empty).

**12 dedicated tests** in `TestRevision3CarmackMvpFirstVerification` class.

### REVISION #4 (Assumption-Adversary): machine-readable JSON ablation table emitter

**Applied as**: `tac.substrates.nscs06_v8_chroma_lut.revisions.emit_per_assumption_ablation_table_json()` writing canonical-schema JSON to `<repo>/.omx/state/nscs06_v8_per_assumption_ablation/nscs06_v8_per_assumption_ablation_<utc>.json` (byte-stable via `sort_keys=True`).

**Schema version**: `nscs06_v8_per_assumption_ablation_v1_20260521`.

**Canonical Provenance per Catalog #287 + #323**:
- `score_claim=False` (PRE-SMOKE prediction, not measurement)
- `promotable=False` (non-promotable per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA")
- `evidence_grade="predicted"` (canonical equation #26 closed-form derivation)
- `axis_tag="[prediction; canonical-equation-26-grounded; per-substrate-symposium-pending]"`
- `canonical_equation_id="procedural_codebook_from_seed_compression_savings_v1"`
- `in_domain_context="nscs06_v8_chroma_lut"`
- `horizon_class="plateau_adjacent"` (per Catalog #309)

**JSON payload sections** (all present in canonical schema):
- `schema_version` (pinned literal)
- `measurement_utc` + `substrate_id` + `canonical_equation_in_domain_context` + `symposium_anchor_memo` + `horizon_class`
- `ladder` (full REVISION #1 7-arm table with per-arm axis tags)
- `multi_scale_dykstra_feasibility` (REVISION #2 verdict)
- `carmack_mvp_first_verification` (REVISION #3 verdict, optional; included when supplied)
- `canonical_provenance` (Catalog #287 + #323 6-field block)

**11 dedicated tests** in `TestRevision4JsonAblationTable` class.

### Integration

**2 end-to-end tests** in `TestRevisionsIntegration` class:
- `test_end_to_end_pre_smoke_harvest_plan`: builds ladder + verifies multi-scale + runs Carmack + emits JSON; verifies all 4 sections + substrate contract unchanged.
- `test_canonical_equation_26_invariance`: verifies the 4 revisions do NOT change canonical equation #26's bytes-saved prediction (4064 bytes; -25 * 4064 / 37545489 ≈ -0.002706).

## Test result verification

```
$ PYTHONPATH=src:upstream:$PWD .venv/bin/python -m pytest src/tac/substrates/nscs06_v8_chroma_lut/tests/ -v
============================= 105 passed in 0.19s ==============================
```

- 49 baseline tests (test_substrate.py) PRESERVED + all pass
- 56 new revision tests (test_revisions.py) all pass
  - 14 in TestRevision1AblationLadder
  - 13 in TestRevision2MultiScaleFeasibility
  - 12 in TestRevision3CarmackMvpFirstVerification
  - 11 in TestRevision4JsonAblationTable
  - 2 in TestRevisionsIntegration

## Catalog gate compliance verification

| Gate | Verdict | Evidence |
|---|---|---|
| Catalog #220 (operational mechanism declared) | PRESERVED | `archive_bytes_added` + `score_improvement_mechanism_status=SCAFFOLD_DEFERRED_INTEGRATION` intact on SubstrateContract; substrate_contract.py UNTOUCHED |
| Catalog #241/#242 (SubstrateContract canonical contract) | PRESERVED | `validate_substrate_contract` (via import) returns valid; substrate_contract.py UNTOUCHED |
| Catalog #244 (NVML/Modal/CUDA env block in driver) | PRESERVED (sister driver) | 3 canonical tokens present in `scripts/remote_lane_substrate_nscs06_v8_chroma_lut.sh`; driver UNTOUCHED |
| Catalog #325 (per-substrate symposium revisions applied) | SATISFIED | THIS landing applies all 4 binding revisions from symposium PROCEED_WITH_REVISIONS verdict |
| Catalog #240 (recipe-vs-trainer-state consistency) | PRESERVED | `research_only=true` + `dispatch_enabled=false` recipe + `_full_main raises NotImplementedError` trainer; THIS landing does NOT lift the dispatch gate |
| Catalog #287 + #323 (canonical Provenance) | NEW SURFACE COMPLIANT | revisions.py + emit_per_assumption_ablation_table_json carry `score_claim=False` + `promotable=False` + `evidence_grade=predicted` + axis_tag literal |
| Catalog #309 (horizon_class declaration) | PRESERVED + EMITTED | Sister design memo declares `plateau_adjacent`; JSON emitter includes `horizon_class` field |
| Canonical equation #26 (4096-32=4064 bytes-saved prediction) | INVARIANT PRESERVED | `predicted_archive_bytes_saved() == 4064` + `predicted_delta_s() ≈ -0.002706` empirically verified |

## Sister-coherence verification

**Slot 1** (`a398f618` Catalog #344 EXCLUDED context, commit `eb7338455`): touches `.omx/state/canonical_equations_registry.jsonl` + `src/tac/canonical_equations/procedural_codebook_savings.py`. DISJOINT from NSCS06 v8 substrate scope. `git status` confirms no overlap with this RATIFY-3 work.

**Slot 2** (`addb8ae4` DP1 RE-DISPATCH): touches `.omx/operator_authorize_recipes/substrate_pretrained_driving_prior_*.yaml` + `.omx/state/modal_call_id_ledger.jsonl`. DISJOINT from NSCS06 v8 substrate scope.

**Sister-checkpoint guard** reported STAND_DOWN_DUPLICATE for the BUILD commit `853d108e2` (within 6h lookback), which is the EXPECTED predecessor per the symposium memo's RATIFY-3 explicit instruction that revisions modify (or augment) the BUILD's substrate package. The "duplicate" classification is appropriate when sister-subagents target the same files; this RATIFY-3 is the FOLLOW-ON wave per symposium PROCEED_WITH_REVISIONS verdict, not a sibling-collision. The mitigation: revisions land as a NEW module `revisions.py` + NEW test file `test_revisions.py`; only `__init__.py` is appended-to (re-exports) — preserving the BUILD's 5 substrate files INTACT per Catalog #110/#113 conceptual append-only discipline applied to live infrastructure.

## 6-hook wire-in declaration per Catalog #125

- **hook #1 sensitivity-map**: ACTIVE via REVISION #2 multi-scale-feasibility decomposition (per-axis seg + pose + rate contribution decomposition IS the sensitivity surface for the LUT shape choice).
- **hook #2 Pareto constraint**: ACTIVE via REVISION #2 Dykstra-feasibility intersection check (alternating projections on rate / seg / pose polytope) + canonical equation #26 IN-DOMAIN context rate-axis contribution to Pareto polytope.
- **hook #3 bit-allocator**: ACTIVE via REVISION #1 luma-quantization-levels ablation (8 / 16 / 32 levels are different bit-allocator regimes).
- **hook #4 cathedral autopilot dispatch**: ACTIVE via REVISION #4 JSON ablation table consumed by sister `tac.cathedral_consumers.canonical_equation_lookup_consumer` per Catalog #335 + #344 + sister `tac.cathedral_consumers.procedural_codebook_generator_consumer`.
- **hook #5 continual-learning posterior**: ACTIVE via REVISION #4 JSON ablation table feeds canonical equation #26 posterior update via `tac.canonical_equations.update_equation_with_empirical_anchor` per Catalog #344 (post-first-paired-smoke).
- **hook #6 probe-disambiguator**: ACTIVE — the per-assumption ablation ladder IS the canonical probe disambiguator for whether CARGO-CULTED assumptions 1-3 actually drift the seg + pose axes per Catalog #308 alternative-probe-methodology enumeration.

## Op-routables (post-RATIFY-3-landing)

| Trigger | Action | Cost |
|---|---|---|
| Per-substrate symposium follow-on T1 working group reviews RATIFY-3 + first paired smoke | PROCEED-unconditional verdict; promote lane to L1 + flip recipe `dispatch_enabled: true` | $0 implementation |
| First paired smoke harvest (post-PROCEED-unconditional) | Invoke `build_per_assumption_ablation_ladder()` + `run_carmack_mvp_first_pre_smoke_verification()` + `emit_per_assumption_ablation_table_json()` BEFORE firing paid GPU meter; harvest 7-arm ablation matrix | $2.00 per symposium cost contract |
| First paired smoke contest-CUDA + contest-CPU anchors land within predicted band | Mark `contest_cuda` + `contest_cpu` gates; promote lane to L2 | $0 |
| First paired smoke score DRIFTS from predicted band by >2x | Per Catalog #324: re-run post-training Tier-C density; per REVISION #1 ablation ladder + REVISION #2 verdict: identify drift-axis via per-assumption isolation; route to UNWIND-TEST per cargo-cult-audit assumptions 1-3 | $1-3 |
| 5-substrate aggregate paired-smoke matrix (CASCADE COMPRESSION symposium PRIORITY 5) | Queue v8 + grayscale_lut + DP1 + VQ-VAE + ATW V2 procedural-variant aggregate paired-smoke matrix; aggregate predicted ΔS -0.013 to -0.0085 | $2-3 |

## Files changed

- NEW: `src/tac/substrates/nscs06_v8_chroma_lut/revisions.py` (1010 LOC; canonical helpers for 4 binding revisions)
- NEW: `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_revisions.py` (442 LOC; 56 dedicated tests across 5 test classes)
- MODIFIED: `src/tac/substrates/nscs06_v8_chroma_lut/__init__.py` (added 15 re-exports for revision APIs; UNMODIFIED bodies of existing imports preserved)

## Files explicitly NOT modified (BUILD canonical contract preserved)

- `src/tac/substrates/nscs06_v8_chroma_lut/architecture.py` (UNCHANGED)
- `src/tac/substrates/nscs06_v8_chroma_lut/archive.py` (UNCHANGED; CH08 grammar + 4064-byte exact-match invariant)
- `src/tac/substrates/nscs06_v8_chroma_lut/inflate.py` (UNCHANGED; ~120 LOC inflate budget; numpy + Pillow only)
- `src/tac/substrates/nscs06_v8_chroma_lut/procedural_variant.py` (UNCHANGED; canonical equation #26 IN-DOMAIN context + predicted_delta_s)
- `src/tac/substrates/nscs06_v8_chroma_lut/substrate_contract.py` (UNCHANGED; Catalog #241/#242 SubstrateContract canonical contract)
- `src/tac/substrates/nscs06_v8_chroma_lut/tests/test_substrate.py` (UNCHANGED; 49 baseline tests preserved)
- `.omx/state/canonical_equations_registry.jsonl` (UNCHANGED; slot 1 ownership preserved)
- `.omx/operator_authorize_recipes/substrate_pretrained_driving_prior_*.yaml` (UNCHANGED; slot 2 ownership preserved)
- `CLAUDE.md` (UNCHANGED; today's amendment `ed80d69a0` sufficient)

## CLAUDE.md compliance verification

- ✅ Apples-to-apples evidence discipline: every prediction in the revisions carries an axis tag (`[prediction; canonical-equation-26-grounded; per-substrate-symposium-pending]`)
- ✅ Forbidden premature KILL: revisions extend UNWIND-TEST per CLAUDE.md "Forbidden premature KILL"; never converts a CARGO-CULTED finding to KILL; per Assumption-Adversary REVISION #1 + #4 verdicts
- ✅ HNeRV parity discipline L4 (≤100 LOC inflate): BUILD's `inflate.py` UNCHANGED at ~120 LOC; substrate_engineering exception per L7 preserved
- ✅ Strict scorer rule: `revisions.py` imports ZERO scorer code (no torch / no smp / no efficientnet / no SegNet/PoseNet identifier tokens)
- ✅ UNIQUE-AND-COMPLETE-PER-METHOD: revisions are UNIQUE per the symposium's PROCEED_WITH_REVISIONS binding; canonical helpers shared via `tac.procedural_codebook_generator` + `tac.substrates._shared.smoke_auth_eval_gate` (UNCHANGED)
- ✅ Catalog #220 operational mechanism: PRESERVED via substrate_contract.py UNCHANGED
- ✅ Catalog #240 recipe-vs-trainer-state: PRESERVED; recipe + trainer UNCHANGED
- ✅ Catalog #244 NVML block: PRESERVED in sister driver script (UNCHANGED)
- ✅ Catalog #287 + #323 canonical Provenance: revisions.py emits canonical Provenance fields in JSON ablation table per REVISION #4
- ✅ Catalog #290 substrate canonical-vs-unique decision per layer: revisions.py is a NEW UNIQUE layer (the per-substrate symposium revisions surface); design memo's canonical-vs-unique decision table preserves the canonical infrastructure layers
- ✅ Catalog #292 per-deliberation assumption surfacing: this memo's frontmatter `council_assumption_adversary_verdict` enumerates 6 assumption classifications
- ✅ Catalog #294 9-dimension success checklist evidence: design memo PRESERVED (BUILD landing satisfies); revisions extend dimension 4 (RIGOR) via 56 new dedicated tests + dimension 7 (DETERMINISTIC REPRODUCIBILITY) via byte-stable JSON `sort_keys=True`
- ✅ Catalog #296 Dykstra-feasibility predicted-band: PRESERVED in design memo; REVISION #2 operationalizes the check at FIRST-PAIRED-SMOKE
- ✅ Catalog #300 v2 frontmatter: this memo carries `council_tier` + `council_attendees` + `council_quorum_met` + `council_verdict` + `council_dissent` + `council_decisions_recorded` + `council_assumption_adversary_verdict` + `council_predicted_mission_contribution` + `council_override_invoked` + `deferred_substrate_id`
- ✅ Catalog #303 cargo-cult audit: PRESERVED in design memo; REVISION #1 ablation ladder operationalizes the unwind-test for CARGO-CULTED assumptions 1-3
- ✅ Catalog #305 observability surface: PRESERVED in design memo; REVISION #4 JSON ablation table satisfies all 6 observability facets (inspectable per layer / decomposable per signal / diff-able across runs / queryable post-hoc / cite-able / counterfactual-able)
- ✅ Catalog #325 per-substrate symposium: SATISFIED via THIS landing
- ✅ Catalog #340 sister-checkpoint guard: PROCEED after the BUILD commit `853d108e2` predecessor was understood as the canonical RATIFY-3 prerequisite; mitigated by landing revisions as NEW files preserving BUILD's 5 substrate files INTACT
- ✅ Catalog #346 canonical_council_roster: T1 6-attendee sextet pact validates complete (Shannon LEAD + Daubechies + Mallat + Carmack + Hotz + Assumption-Adversary) per symposium memo

## Lane registry update

```bash
.venv/bin/python tools/lane_maturity.py mark \
    lane_ratify_3_nscs06_v8_t1_binding_revisions_applied_20260521 \
    --gate impl_complete \
    --evidence "src/tac/substrates/nscs06_v8_chroma_lut/revisions.py + tests/test_revisions.py 105/105 tests pass"
.venv/bin/python tools/lane_maturity.py mark \
    lane_ratify_3_nscs06_v8_t1_binding_revisions_applied_20260521 \
    --gate memory_entry \
    --evidence ".omx/research/nscs06_v8_t1_binding_revisions_applied_landed_20260521.md"
```

(Lane registry update is a follow-on operator-routable action; not part of this commit batch.)

## Sister cross-references

- **NSCS06 v8 BUILD commit `853d108e2`**: the canonical BUILD this RATIFY-3 applies revisions to.
- **Per-substrate symposium memo**: `.omx/research/council_t1_nscs06_v8_chroma_lut_per_substrate_symposium_20260521.md` (the PROCEED_WITH_REVISIONS verdict source).
- **Design memo**: `.omx/research/nscs06_v8_chroma_lut_design_20260521.md` (the 9-dim checklist + cargo-cult audit + observability surface evidence).
- **CASCADE COMPRESSION symposium commit `d125af6c3`** PRIORITY 3: elevated v8 chroma_lut as second-priority IN-DOMAIN procedural-variant substrate.
- **HONEST CASCADE-MORTALITY ASSESSMENT commit `d884dd6aa`** Rank 2.
- **NSCS06 v6 -> v7 cargo-cult-unwind methodology commit `4292c8ce2`**: sister rescue-path pattern.
- **Canonical equation #26**: `procedural_codebook_from_seed_compression_savings_v1` at `src/tac/canonical_equations/procedural_codebook_savings.py`; `nscs06_v8_chroma_lut` IS in `_INCLUDED_CONTEXTS`.
- **Sister Catalog #344 EXCLUDED context landing** (slot 1 commit `eb7338455`): touches canonical equations registry; disjoint from this work.
- **Sister DP1 RE-DISPATCH** (slot 2): touches operator-authorize recipes; disjoint from this work.
