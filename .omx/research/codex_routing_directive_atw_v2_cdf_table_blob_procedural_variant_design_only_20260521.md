# Codex Routing Directive — ATW V2 cdf_table_blob procedural-variant DESIGN-ONLY (subject to Catalog #325 + D4 + Variant-C scoping gates)

**Date**: 2026-05-21T04:54:00Z (UTC)
**Authority**: Operator blanket approval 2026-05-20 + WAVE-3-REVERSE-CODEX-ROUTING-DIRECTIVES-FANOUT fan-out per CODEX CROSS-POLLINATION audit `aafac7c84` §15.4 REVERSE-DIRECTIVE #5
**For consumption by**: codex CLI subagent (Pattern A detached BG invocation)
**Source draft**: `.omx/research/codex_md_files_cross_pollination_synergy_audit_20260520T041700Z.md` §15.4 REVERSE-DIRECTIVE #5
**Lane**: `lane_codex_atw_v2_cdf_table_blob_procedural_variant_design_only_20260521`

## Operator directive

Design (NOT BUILD) ATW V2 cdf_table_blob procedural-variant targeting the **2,528-byte CDF table** section identified by PARSER-SAFE EXTENSION (main-thread commit `d0bf3ce37`) as parser-safe + score-affecting, **rank #2** in `procedural_replacement_surface_matrix_landed_20260521_codex.md`.

**CRITICAL constraint**: per matrix `DESIGN_READY_DEFERRED` status, ATW V2 paid dispatch is gated by:
1. **D4 predecessor verdict** (the ATW v2 D4 H(latent|scorer_class) probe `INDEPENDENT` verdict registered in `.omx/state/probe_outcomes.jsonl` per Catalog #313)
2. **Variant-C scoping gate** (refer to ATW V2 design lineage for Variant-A/B/C scoping)
3. **Catalog #325 per-substrate symposium** PROCEED-unconditional verdict on the new variant

This directive is **DESIGN-ONLY**; NO paid dispatch, NO L0 scaffold code edits beyond the design memo + symposium memo + probe-outcomes ledger registration. The variant's BUILD step is a SEPARATE operator-approval gate AFTER all 3 prerequisite gates clear.

## Pre-flight (Catalog #229 premise verification)

Read these files in full BEFORE any design verdict:

- `src/tac/substrates/atw_codec_v2/` (canonical implementation; full package)
- `experiments/train_substrate_atw_codec_v2.py` (if exists; else `experiments/train_substrate_atw_codec.py` for V1 lineage)
- `.omx/research/atw_codec_atick_tishby_wyner_v1_design_20260515.md` (V1 design lineage)
- `.omx/research/atw_v2_d4_*_*.md` (D4 predecessor verdict files; ALL files matching the glob)
- `.omx/research/atw_v2_*_design_*.md` (V2 design memos)
- Sister VQ-VAE design memo at `.omx/research/vq_vae_procedural_codebook_variant_design_<UTC>.md` (if WAVE-3 REVERSE-DIRECTIVE #4 has landed; cross-pollinate paradigm classification + observability surface design)
- `.omx/research/parser_safe_methodology_extension_landed_20260520.md`
- `.omx/research/procedural_replacement_surface_matrix_landed_20260521_codex.md`
- Canonical equation #26 INCLUDED contexts: `src/tac/canonical_equations/procedural_codebook_savings.py` + `_domain_refinement.py` + `.omx/state/canonical_equations_registry.jsonl` (filter to `equation_id=procedural_codebook_from_seed_compression_savings_v1`)
- NEW equation `procedural_predictor_plus_residual_correction_savings_v1` (commits `d3e63bbe9` design + `af36cd72` slot 1 + `098d8a31c` ratification)
- D4 probe outcomes: `tools/check_predecessor_probe_outcome.py --substrate atw_codec_v2`
- Variant-C scoping memo (search `.omx/research/atw*variant*` glob)
- Sister codex Cluster B audit: `.omx/research/cable_h1_recursive_review_r11_findings_20260519T060942Z.md` (for ATW V2 cooperative-receiver lessons; cable H1 R11 findings)

## Deliverables

1. **Design memo** at `.omx/research/atw_v2_cdf_table_blob_procedural_variant_design_<UTC>.md` with the canonical substrate design memo contract enforced by:
   - Catalog #290 `## Canonical-vs-unique decision per layer` section
   - Catalog #294 `## 9-dimension success checklist evidence` section
   - Catalog #303 `## Cargo-cult audit per assumption` section (CDF table semantic — WHAT is being replaced procedurally? Is the CDF table a learned distribution or a hand-coded one? Hard-earned vs cargo-culted classification of EVERY assumption.)
   - Catalog #305 `## Observability surface` section
   - Catalog #296 Dykstra-feasibility check on `## Predicted ΔS band` section (per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check")
   - Catalog #309 `horizon-class:` declaration
2. **CRITICAL: paradigm classification** per CODEX CROSS-POLLINATION audit §15.1 Insight 1 (3-paradigm taxonomy):
   - **REPLACEMENT-UPSTREAM** (equation #26 applies; CDF table bytes REMOVED and procedural seed bytes ADDED)
   - **RESIDUAL-CORRECTION-DOWNSTREAM** (NEW equation `procedural_predictor_plus_residual_correction_savings_v1` applies; CDF prediction + residual correction stacked per Catalog #359 guard)
   - **REMOVAL** (refuse: CDF is score-affecting per parser-safe matrix, so removal alone would degrade score)
3. **Catalog #325 per-substrate symposium memo** at `.omx/research/council_grand_council_atw_codec_v2_procedural_cdf_table_variant_<YYYYMMDD>.md` per the canonical 6-step contract:
   - Step 1: Catalog #303 cargo-cult audit per assumption
   - Step 2: Catalog #294 9-dim checklist evidence
   - Step 3: Catalog #305 observability surface declaration
   - Step 4: Sextet pact deliberation (Shannon + Dykstra + Yousfi + Fridrich + Contrarian + Assumption-Adversary) PLUS topical grand council attendees: Atick + Redlich + Tishby memorial + Zaslavsky + Wyner (per CLAUDE.md "Grand Council (advisory)" Z4 deliberation pattern — ATW = Atick-Tishby-Wyner triple)
   - Step 5: Per-substrate reactivation criteria pinned
   - Step 6: Catalog #324 post-training Tier-C validation discipline declared
4. **Catalog #313 probe-outcomes ledger** register design verdict at `.omx/state/probe_outcomes.jsonl`:
   - If symposium returns PROCEED-unconditional: register `verdict=PROCEED, status=advisory, event_type=adjudicated`
   - If symposium returns PROCEED_WITH_REVISIONS: register `verdict=DEFER, status=blocking, event_type=adjudicated, reactivation_criteria="apply revisions + re-symposium"`
   - If symposium returns DEFER_PENDING_EVIDENCE: register `verdict=DEFER, status=blocking, event_type=adjudicated, reactivation_criteria="land evidence per symposium spec"`
5. **Variant-C scoping gate decision**: `PROCEED` / `DEFER` / `ESCALATE_TO_OPERATOR` per the existing ATW V2 Variant-A/B/C scoping framework
6. **Class-prior-table-blob signal-preservation probe plan** (19,168 B; rank #3 in matrix): if D4 + Variant-C gates clear AND the cdf_table variant design lands cleanly, append a probe plan for the class-prior-table-blob as a SEPARATE follow-up directive (do NOT execute in this directive's scope — recommend as next operator-routable)
7. **Catalog #296 Dykstra-feasibility check** on predicted ΔS band: cite Shannon R(D) OR MDL OR Tishby IB OR Atick-Redlich receiver-cooperation OR Wyner-Ziv side-information theorem; bare-prediction without first-principles bound is REFUSED per CLAUDE.md "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check"
8. **Catalog #308 alternative-probe-methodology enumeration**: ≥3 alternative reducers/probes for the cdf_table_blob (e.g., (a) procedural CDF from analytic distribution params; (b) procedural CDF from histogram seed; (c) procedural CDF from PoseNet-class-conditioned analytic form per Atick-Redlich)
9. **Catalog #318 raw-byte-authority guard cross-check**: ATW V2 procedural variant consumer MUST use `CandidateModificationSpec` + `grammar_aware_operator` typed responses; NO `byte_modifications: Mapping[int, float]` raw byte APIs

## Discipline (per CLAUDE.md non-negotiables)

- Catalog #229 PV (read D4 history + Variant-C history + V1 design + sister VQ-VAE design)
- Catalog #318 raw-byte-authority guard
- Catalog #325 per-substrate symposium REQUIRED before paid dispatch (THIS DIRECTIVE IS DESIGN-ONLY; paid dispatch is a SEPARATE operator-approval gate)
- Catalog #307 paradigm-vs-implementation classification at design memo surface
- Catalog #308 alternative-probe-methodology enumeration (≥3 alternatives)
- Catalog #344 canonical equation registry: design memo MUST cite equation #26 OR new equation `procedural_predictor_plus_residual_correction_savings_v1` explicitly
- Catalog #359 misapplication-to-residual-hybrid guard: if RESIDUAL-CORRECTION paradigm chosen, design memo MUST route through the NEW equation, NOT equation #26
- Catalog #117/#157/#174 canonical commit serializer with POST-EDIT `--expected-content-sha256`
- Catalog #206 checkpoint every ~5 tool uses; final `--step complete --status complete` checkpoint
- Catalog #119 Co-Authored-By trailer
- Catalog #234 substantive commit message bodies
- Catalog #248 zero residual conflict markers
- Catalog #287 evidence-tag for every claim
- Catalog #340 sister-checkpoint guard PROCEED required at every commit
- 6-hook wire-in declaration per Catalog #125 (for the variant's runtime artifact AND for the symposium memo)

## 6-hook wire-in declaration per Catalog #125

For this DESIGN-ONLY directive:
- Hook #1 sensitivity-map: **N/A** (design memo, no signal contribution; runtime variant's sensitivity is hook #1 ACTIVE if BUILD lands)
- Hook #2 Pareto constraint: **N/A** at design surface
- Hook #3 bit-allocator: **N/A** at design surface
- Hook #4 cathedral autopilot dispatch: **N/A** at design surface (variant's BUILD step is the autopilot-consumable surface)
- Hook #5 continual-learning posterior: **ACTIVE** (probe-outcomes ledger registration is canonical posterior anchor)
- Hook #6 probe-disambiguator: **ACTIVE** (paradigm classification IS the canonical disambiguator between REPLACEMENT-UPSTREAM vs RESIDUAL-CORRECTION-DOWNSTREAM vs REMOVAL)

## Scope limits

DO NOT:
- Fire paid GPU dispatch (DESIGN-ONLY scope; Catalog #325 + D4 + Variant-C gates)
- BUILD trainer code, lane driver, recipes (separate operator-approval gate AFTER design + symposium + probe-outcomes verdict)
- Mutate canonical equation #26 mathematical predicate or `_INCLUDED_CONTEXTS` set without explicit operator authorization
- Mutate sister memos per Catalog #110/#113 APPEND-ONLY
- Mutate D4 probe outcomes (HISTORICAL_PROVENANCE)
- Mutate ATW V2 V1 design lineage memos per Catalog #110/#113
- Spawn nested subagents
- Modify CLAUDE.md
- Push to origin
- Skip Catalog #296 Dykstra-feasibility check on predicted band (gate fires structurally otherwise per `## Predicted ΔS band` section trigger)
- Skip Catalog #294 9-dim checklist evidence section
- Skip Catalog #325 per-substrate symposium memo (paid dispatch is impossible without it)

## Estimated cost

- $0 GPU (DESIGN-ONLY; NO paid dispatch; NO BUILD)
- ~2.5h wall-clock

## Cross-references

- CODEX CROSS-POLLINATION audit memo: `.omx/research/codex_md_files_cross_pollination_synergy_audit_20260520T041700Z.md` §15.4 REVERSE-DIRECTIVE #5
- PARSER-SAFE EXTENSION landing: `.omx/research/parser_safe_methodology_extension_landed_20260520.md` (commit `d0bf3ce37`)
- Procedural replacement surface matrix: `.omx/research/procedural_replacement_surface_matrix_landed_20260521_codex.md`
- ATW V1 design: `.omx/research/atw_codec_atick_tishby_wyner_v1_design_20260515.md`
- ATW V2 D4 probe outcome verdict (search `.omx/research/atw_v2_d4_*_*.md`)
- Sister WAVE-3 VQ-VAE directive: `.omx/research/codex_routing_directive_vq_vae_indices_blob_procedural_variant_20260521.md`
- Sister WAVE-3 DP1 parity audit directive: `.omx/research/codex_routing_directive_dp1_paired_smoke_parity_audit_20260521.md`
- Equation #26 domain refinement event 8: commit `79f1ba387`
- New equation slot 1 commit: `af36cd72` + ratification `098d8a31c` + design memo `d3e63bbe9`
- Aggregate WAVE-3-FAN-OUT landing: `.omx/research/reverse_codex_routing_directives_fan_out_landed_20260521.md`

## Operator-routable instruction

```bash
codex /goal --skill codex-cli-runtime \
    --input .omx/research/codex_routing_directive_atw_v2_cdf_table_blob_procedural_variant_design_only_20260521.md \
    --goal "ATW V2 cdf_table_blob procedural-variant DESIGN-ONLY memo per WAVE-3 REVERSE-DIRECTIVE #5"
```

OR Pattern A detached invocation per CLAUDE.md "Codex CLI invocation":

```bash
mkdir -p .omx/tmp/codex_runs
nohup bash -c '
  codex exec --skip-git-repo-check --sandbox read-only \
    -m gpt-5.5 -c model_reasoning_effort=xhigh \
    -o .omx/tmp/codex_runs/atw_v2_cdf_table_procedural_design_only.last.txt \
    "$(cat .omx/research/codex_routing_directive_atw_v2_cdf_table_blob_procedural_variant_design_only_20260521.md)" \
    2>&1 | tee .omx/tmp/codex_runs/atw_v2_cdf_table_procedural_design_only.log > /dev/null
' < /dev/null > .omx/tmp/codex_runs/atw_v2_cdf_table_procedural_design_only.outer.log 2>&1 &
disown
```
