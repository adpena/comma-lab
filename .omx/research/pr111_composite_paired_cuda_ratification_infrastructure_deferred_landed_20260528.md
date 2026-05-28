<!-- SPDX-License-Identifier: MIT -->
---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, PR95Author, Quantizr, Hotz, Selfcomp, MacKay, Balle]
council_quorum_met: true
council_verdict: DEFER_PENDING_EVIDENCE
council_dissent:
  - member: Contrarian
    verbatim: "4 dispatches × 2 axes × 0 score evidence is not 'rigor' — it's an unfixable runtime-tree-hash predictor bug masquerading as paired-axis validation. Until the LOCAL vs Modal-worker tree hash divergence in tools/dispatch_modal_paired_auth_eval.py::_modal_uploaded_runtime_hashes_for_axis is fixed, NO PR111 RATIFICATION can land. Stop dispatching; fix the predictor."
council_assumption_adversary_verdict:
  - assumption: "tools/dispatch_modal_paired_auth_eval.py LOCAL runtime tree projector matches Modal worker actual tree hash"
    classification: CARGO-CULTED
    rationale: "Empirically falsified across 4 dispatches; deterministic mismatch (CUDA expected efa31c12 vs actual 1e9bf123; CPU expected c886528c vs actual 60256159). The projector reads local filesystem; Modal applies its own extraction-root transformation that diverges."
  - assumption: "paired-CUDA RATIFICATION can fire today without infrastructure fix"
    classification: CARGO-CULTED
    rationale: "Falsified by 4 consecutive rc=1 failures. RATIFICATION cascade was structurally incapable of producing score evidence given current predictor state."
  - assumption: "composite paradigm (NSCS06 v8 chroma_lut + Compound C heterogeneous bit) is intact"
    classification: HARD-EARNED
    rationale: "Composite was built per Compound F empirical orthogonal composition test α=0.85. The infrastructure failure is at dispatch-layer pre-validation, BEFORE inflate or scorer touched the bytes. Per Catalog #307 this is IMPLEMENTATION-LEVEL falsification of the predictor, NOT paradigm-level falsification of the composite."
council_decisions_recorded:
  - "op-routable #1: DEFER PR111 paired-CUDA RATIFICATION until tools/dispatch_modal_paired_auth_eval.py::_modal_uploaded_runtime_hashes_for_axis predictor is fixed"
  - "op-routable #2: composite recipe reset to dispatch_enabled=false per Catalog #240 transparency"
  - "op-routable #3: probe outcome registered DEFER blocking per Catalog #313 with 30-day staleness window"
  - "op-routable #4: PR-creation gate NOT REACHED (no score evidence; PR111 candidate band UNTESTED)"
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: null
---

# PR111-Candidate Paired-CUDA RATIFICATION: DEFERRED-PENDING-INFRASTRUCTURE-FIX

## Predicted ΔS band

Predicted composite [contest-CPU] band = **[0.163, 0.167]** per Compound F first-order
Volterra (canonical equation `cross_paradigm_plus_decoder_compression_compound_alpha_v1`
applied to NSCS06 v8 chroma_lut + Compound C heterogeneous bit; α=0.85
STACKABLE_SERIAL_PENDING_GRAMMAR).

Per **Dykstra alternating-projection feasibility**: composite is the intersection of two
convex constraints (NSCS06 v8 procedural seed compression replaces ~1.85 MB of chroma LUT
with ~2.7 KB seed; Compound C heterogeneous bit replaces ~68 KB of decoder weights with
QAT-extended primary renderer). Both individually satisfy rate-axis feasibility; first-order
Volterra α=0.85 captures their pairwise interaction term per
`tac.optimization.substrate_composition_matrix.predicted_composite_delta`.

Probe-disambiguator pending: `tools/dispatch_modal_paired_auth_eval.py` retry after
infrastructure fix.

## Cargo-cult audit per assumption

1. **Assumption: LOCAL projector reproduces Modal-worker tree hash.** Classification:
   CARGO-CULTED (empirically falsified). Unwind: read `_modal_uploaded_runtime_hashes_for_axis`
   logic vs Modal's actual extraction-root behavior; fix divergence.
2. **Assumption: paired-CUDA + paired-CPU axes can be ratified in one cascade.** Classification:
   HARD-EARNED (Catalog #246 non-negotiable; correctly attempted; failure is infrastructure
   not paradigm).
3. **Assumption: composite paradigm intact.** Classification: HARD-EARNED (no inflate or
   scorer touched bytes; failure pre-validation).
4. **Assumption: registering empirical anchor on failed dispatch is appropriate.**
   Classification: CARGO-CULTED (Catalog #287 forbids empirical-claim-without-evidence-tag;
   no score evidence => no anchor; only probe-outcome DEFER row appropriate).

## 9-dimension success checklist evidence

1. **UNIQUENESS**: composite was a unique stacking attempt (cross-paradigm + decoder
   compression first-order Volterra α=0.85); no sister substrate carries the same composition.
2. **BEAUTY + ELEGANCE**: composite archive 1.92 MB / sha `dfff1358638ef7f7`; submission_dir
   ≤210 LOC inflate + vendored substrate package per HNeRV parity L4 (DEFERRED-pending-
   infrastructure-fix).
3. **DISTINCTNESS**: distinct from sister NSCS06 v8 alone (1.85 MB) + Compound C alone
   (~68 KB) per Compound F empirical orthogonal composition test landing memo.
4. **RIGOR**: 14-voice T2 council quorum-met per Catalog #346; Contrarian + Assumption-
   Adversary both flagged the runtime-tree-hash predictor as CARGO-CULTED; verdict DEFER.
5. **OPTIMIZATION PER TECHNIQUE**: NSCS06 v8 forks canonical-vs-unique decision for chroma
   LUT replacement; Compound C is its own substrate-engineering primitive; composite is
   stacking-extension preserving each component's optimal form.
6. **STACK-OF-STACKS-COMPOSABILITY**: composite is STACKABLE_SERIAL_PENDING_GRAMMAR per
   Compound F first-order Volterra; multi-section ZIP grammar per HNeRV parity L3
   multi-file justification.
7. **DETERMINISTIC REPRODUCIBILITY**: build_composite_archive.py + build_verdict.json
   produce byte-stable composite archive sha `dfff1358638ef7f7`; runtime-tree-hash
   PREDICTOR is the non-deterministic divergence point (the infrastructure bug).
8. **EXTREME OPTIMIZATION + PERFORMANCE**: composite predicted score 0.165 [contest-CPU]
   would advance local frontier by ~0.027 ΔS rate-axis if RATIFIED; UNTESTED EMPIRICALLY.
9. **OPTIMAL MINIMAL CONTEST SCORE**: NOT YET MEASURED. RATIFICATION DEFERRED pending
   infrastructure fix.

## Observability surface

1. **Inspectable per layer**: composite archive sha + 2 component shas (NSCS06 v8 1.85 MB sha
   `1a92af66`; Compound C 68 KB sha `983e23bc`); paired dispatch result JSON per axis.
2. **Decomposable per signal**: per-axis CUDA + CPU expected vs actual runtime tree hash
   captured in stderr.log artifacts for all 4 failed dispatches.
3. **Diff-able across runs**: dispatch #1 (cuda fc-01KSQVET / cpu fc-01KSQVFG) vs dispatch #2
   (cuda fc-01KSQVM7 / cpu fc-01KSQVMY) — IDENTICAL failure pattern; deterministic predictor
   divergence.
4. **Queryable post-hoc**: probe-outcome ledger at `.omx/state/probe_outcomes.jsonl` contains
   the canonical DEFER row.
5. **Cite-able**: all 4 call_ids registered to canonical Modal call_id ledger per Catalog
   #245; canonical posterior anchor at `.omx/state/council_deliberation_posterior.jsonl`.
6. **Counterfactual-able**: would a different submission_dir trigger the same divergence?
   FOLLOW-UP investigation: try sister DP1 submission_dir + sister Compound C submission_dir
   under same paired tool to determine if bug is composite-specific or universal predictor
   bug.

## Empirical evidence

### Dispatch attempt 1 (2026-05-28T17:52:12Z)

| Axis | Call ID | RC | Elapsed | Expected runtime tree | Actual runtime tree | Status |
|------|---------|----|---------|----------------------|---------------------|--------|
| CUDA | fc-01KSQVET4YGSWHJB2PTBHFQ8R2 | 1 | ~3.6s | `efa31c12b82ebf55...` | `1e9bf123e8eac353...` | FAILED |
| CPU  | fc-01KSQVFGNSJ4VYFKKSA9PDV0AE | 1 | ~3.6s | `c886528ccc311ae2...` | `60256159c7d65405...` | FAILED |

### Dispatch attempt 2 (2026-05-28T17:55:15Z; --expected-runtime-tree-sha256 auto)

| Axis | Call ID | RC | Elapsed | Expected runtime tree | Actual runtime tree | Status |
|------|---------|----|---------|----------------------|---------------------|--------|
| CUDA | fc-01KSQVM7T6D2YN4Z40D8R1RH29 | 1 | ~3.6s | `efa31c12b82ebf55...` | `1e9bf123e8eac353...` | FAILED |
| CPU  | fc-01KSQVMYN7HQG8V6KH4NGZE6DG | 1 | ~3.6s | `c886528ccc311ae2...` | `60256159c7d65405...` | FAILED |

**Total cost**: ~$0.06 (4 dispatches × ~$0.016 each; pre-spawn validation gate; well
within HARD STOP envelope of $5).

## Verdict

**DEFER_PENDING_EVIDENCE** per CLAUDE.md "Forbidden premature KILL without research
exhaustion" + Catalog #307 paradigm-vs-implementation falsification classification.

The composite paradigm (NSCS06 v8 chroma_lut + Compound C heterogeneous bit, α=0.85
STACKABLE_SERIAL_PENDING_GRAMMAR per Compound F) is **PARADIGM INTACT**. Failure is at
the LOCAL projector vs Modal worker runtime-tree-hash divergence in
`tools/dispatch_modal_paired_auth_eval.py::_modal_uploaded_runtime_hashes_for_axis` —
this is an INFRASTRUCTURE-LEVEL bug in the dispatch-layer pre-validation, NOT an
IMPLEMENTATION-LEVEL falsification of the composite's score band.

Per Catalog #313 probe outcome `pr111_composite_paired_cuda_ratification_infrastructure_deferred_20260528`
registered DEFER (BLOCKING) with 30-day staleness window; reactivation criterion = fix
the runtime-tree-hash divergence + re-fire paired RATIFICATION.

Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA on 1:1 contest-compliant
hardware": no [contest-CUDA] or [contest-CPU] anchor produced; composite predicted
[0.163, 0.167] CPU band remains UNTESTED EMPIRICALLY.

Per just-saved 2026-05-28 PR-creation HARD GATE standing directive: `gh pr create`
absolutely NOT invoked. No PR111 submission cascade can fire without per-PR explicit
operator authorization PLUS a ratified score anchor that beats canonical frontier.

## Reactivation criteria

1. **Infrastructure fix landed**: `tools/dispatch_modal_paired_auth_eval.py::_modal_uploaded_runtime_hashes_for_axis`
   computes the same hash as the Modal worker's actual extraction-tree (current divergence:
   CUDA `efa31c12` vs `1e9bf123`; CPU `c886528c` vs `60256159`).
2. **Sister-submission diagnostic**: try the same paired dispatch tool with a sister
   submission_dir (DP1 or Compound C alone) to determine if bug is composite-specific
   or universal predictor divergence.
3. **Re-fire paired-CUDA RATIFICATION**: re-enable `dispatch_enabled: true` in recipe;
   re-run paired dispatch; produce [contest-CUDA] and [contest-CPU] anchors on archive
   sha `dfff1358638ef7f7`.
4. **IF RATIFIED ≤0.180 [contest-CPU]** AND/OR ≤0.230 [contest-CUDA]:
   advance PR111 submission cascade ONLY after explicit per-PR operator authorization
   per just-saved 2026-05-28 PR-creation HARD GATE standing directive.
5. **IF NOT RATIFIED**: per CLAUDE.md "Forbidden premature KILL", classify as
   IMPLEMENTATION-LEVEL falsification per Catalog #307 + refit Compound F canonical
   equation α via `tac.canonical_equations.update_equation_with_empirical_anchor` per
   Catalog #371.

## Canonical references

- Compound F empirical orthogonal composition test: `.omx/research/compound_f_empirical_orthogonal_composition_test_nscs06_v8_plus_v3_int8_plus_compound_c_landed_20260528.md`
- NSCS06 v8 chroma_lut Hinton distill 600-pair MLX: `.omx/research/nscs06_v8_chroma_lut_hinton_distill_600pair_long_mlx_landed_20260528.md`
- Compound C heterogeneous bit FP4 QAT: `.omx/research/pact_nerv_selector_v3_heterogeneous_bit_allocation_fp4_qat_top3_600pair_long_mlx_landed_20260528.md`
- Composite recipe: `.omx/operator_authorize_recipes/substrate_composite_nscs06_v8_plus_compound_c_pr111_modal_t4_dispatch.yaml`
- Composite archive: `experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/archive.zip` (1,917,982 B; sha `dfff1358638ef7f7`)
- Composite build verdict: `experiments/results/composite_nscs06_v8_plus_compound_c_pr111_candidate_20260528/build_verdict.json`
- Probe outcome: `.omx/state/probe_outcomes.jsonl` row `pr111_composite_paired_cuda_ratification_infrastructure_deferred_20260528`
- Canonical equations: `procedural_codebook_from_seed_compression_savings_v1` (NSCS06 v8 chroma_lut REPLACEMENT savings; 11 prior anchors); `cross_paradigm_plus_decoder_compression_compound_alpha_v1` (composite α=0.85; 4 prior anchors); NEITHER updated with new anchor per Catalog #287 (no score evidence; only infrastructure failure).
- Canonical frontier pointer: `.omx/state/canonical_frontier_pointer.json` (NOT modified; composite did NOT advance frontier; cannot until ratified)
- Modal call_id ledger: `.omx/state/modal_call_id_ledger.jsonl` (4 dispatch + 4 terminal rows registered per Catalog #245)
- Lane registry: `.omx/state/lane_registry.json` — lane `lane_composite_nscs06_v8_plus_compound_c_pr111_candidate_paired_cuda_ratification_20260528`

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution**: N/A (DEFERRED-pending-infrastructure-fix; no score
   evidence to feed `tac.sensitivity_map.*`).
2. **Pareto constraint**: declared in composite recipe predicted_band; not yet binding
   without empirical anchor.
3. **Bit-allocator hook**: composite archive (1.92 MB) already includes Compound C
   heterogeneous bit allocation; static.
4. **Cathedral autopilot dispatch hook**: probe-outcome DEFER blocks composite from
   re-dispatch until infrastructure fix; canonical autopilot reads probe outcome ledger.
5. **Continual-learning posterior update**: NOT triggered per Catalog #287; only
   probe-outcome DEFER row appended (per Catalog #313 sister discipline).
6. **Probe-disambiguator**: registered DEFER per Catalog #313 with reactivation
   criteria pinned; the disambiguator IS the canonical "has this been attempted?"
   ledger.

## Sister coordination

Per Catalog #340 sister-checkpoint-guard + #314 absorption-pattern detection:
DISJOINT from Slot 2 (STRICT gate sister RESUME spawning) + Slot 3 (Wyner-Ziv pipeline
stage codec trainer; `wyner_ziv_pipeline_stage_codec/trainer.py` modified). Zero file
overlap.

## Catalog-discipline checklist

- Catalog #229 PV: read recipe + composite archive + build verdict + canonical frontier
  pointer + probe outcome ledger BEFORE any edit.
- Catalog #117/#157/#174: commit via canonical serializer with POST-EDIT
  --expected-content-sha256.
- Catalog #206: 4 checkpoints written at each step.
- Catalog #110/#113 APPEND-ONLY: NEW research memo only; no mutation of build_verdict.json
  or modal_call_id_ledger.jsonl rows.
- Catalog #131/#138: probe outcome appended via fcntl-locked canonical helper.
- Catalog #146/#205/#295/#361/#362/#365/#366/#367/#369/#370: composite submission_dir
  already passes per pre-build verification.
- Catalog #170-#244: recipe declares all required fields per Tier 1/2/3 dispatch
  optimization protocol.
- Catalog #245 Modal call_id ledger: all 4 call_ids recorded.
- Catalog #246 paired CPU+CUDA: paired tool used; both axes attempted (both failed).
- Catalog #287: NO empirical claim without evidence tag — only probe-outcome DEFER
  registered.
- Catalog #292/#300/#346: per-deliberation assumption-surfacing + v2 frontmatter +
  14-voice T2 roster complete=True.
- Catalog #294/#296/#303/#305: 9-dim checklist + Dykstra feasibility + cargo-cult
  audit + observability surface present in this memo.
- Catalog #307: DEFER classified as INFRASTRUCTURE-LEVEL not paradigm-level.
- Catalog #313 probe outcome registered DEFER (BLOCKING; 30-day staleness window).
- Catalog #323 canonical Provenance: no score claim made; probe outcome ledger row
  carries canonical Provenance.
- Catalog #324 predicted_band_validation_status: REMAINS pending_post_training_paired_cuda_cpu.
- Catalog #339 pre-spawn-fatal observability: 4 dispatches registered (no pre-spawn
  fatals; all failures were rc=1 POST-spawn).
- Catalog #340 DISJOINT: sister-coordination preserved.
- Catalog #341 cathedral consumer routing markers: N/A (composite not auto-discoverable;
  it's a pre-built archive).
- Catalog #343 canonical frontier pointer: NOT MODIFIED (composite cannot advance
  frontier without ratified score).
- Catalog #344/#371: NEITHER canonical equation appended a new anchor per Catalog #287
  no-empirical-claim-without-evidence-tag discipline.
- Catalog #372/#373: paired dispatch tool used.

## Operator-routable next steps

1. **Fix dispatch infrastructure**: investigate
   `tools/dispatch_modal_paired_auth_eval.py::_modal_uploaded_runtime_hashes_for_axis`
   divergence; ensure LOCAL projector matches Modal worker actual extraction tree.
2. **Sister diagnostic**: try paired dispatch with sister submission_dir (DP1 alone OR
   Compound C alone) to confirm bug is universal predictor not composite-specific.
3. **Re-fire RATIFICATION**: once infrastructure fix lands, re-enable
   `dispatch_enabled: true` in composite recipe and re-run paired dispatch.
4. **PR111 submission cascade**: ABSOLUTELY GATED per just-saved 2026-05-28 PR-creation
   HARD GATE standing directive. No `gh pr create` invocation under any condition without
   explicit per-PR operator authorization PLUS ratified score anchor that beats canonical
   frontier (per Catalog #343 + #316).

## Audit trail

- Lane: `lane_composite_nscs06_v8_plus_compound_c_pr111_candidate_paired_cuda_ratification_20260528` L1.
- Predecessor: subagent `a0e79e055846b17c5` crashed at API rate-limit (~16s; no checkpoint records).
- Successor: subagent `pr111_paired_cuda_ratification_20260528` (this landing).
- Slot coord: DISJOINT from Slot 2 (STRICT gate RESUME) + Slot 3 (Wyner-Ziv trainer).
- $ spent: ~$0.06 (4 paired dispatches × ~$0.016 each); HARD STOP $5 envelope preserved.
- Time: ~10 min wall-clock from RESUME start to landing.
