# Retroactive Sweep for Catalog #381

Per Catalog #348 `check_new_gate_landing_includes_retroactive_sweep_evidence`:
every NEW gate landing requires a retroactive sweep memo with the 4-field
contract.

## 1. Bug-class symptom signature

Research-pipeline tools writing under `.omx/research/**/*.json` mutate JSON
files in-place across separate invocations (mutated fields include
`generated_at_utc`, `archive_path`, `runtime_consumption_proof_bytes`,
`source_records_sha256`, `hash_manifest_sha256`, `environment_sha256`,
`execution_context_sha256`). Per `.omx/state/artifact_kind_registry.yaml`
the `.omx/research/**/*.json` namespace is classified `HISTORICAL_PROVENANCE`
(immutable per Catalog #110 + #113 canonical 4-kind taxonomy).

## 2. Pre-fix window

2026-05-28T22:41 preflight audit landing memo:

`feedback_bugs_config_anti_pattern_preflight_audit_code_recipe_surface_review_landed_20260528.md`

The audit captured **77 in-place field mutations across 3 dirs**:

- `pr95_mlx_runtime_consumption_queue_20260528T131513Z` — 24 mutations
- `repair_multi_archive_autonomous_live_psv3_fec6_20260528T055303Z` — 50 mutations
- `frontier_final_rate_attack_fp11_brotli_exec3_20260528Tlocal` — 3 mutations

Canonical anti-pattern
`research_pipeline_tool_re_writes_historical_provenance_json_with_mutated_fields_v1`
was registered in `.omx/state/canonical_anti_patterns_registry.jsonl` with the
77-row EmpiricalFalsification at landing.

Canonical equation
`historical_provenance_immutability_predicts_zero_in_place_mutation_v1` was
registered in `.omx/state/canonical_equations_registry.jsonl` with the same
empirical anchor and residual=77.0.

## 3. Historical KILL/DEFER/FALSIFY search

A search of historical landing memos + canonical posterior was performed for
any KILL/DEFER/FALSIFY verdicts whose evidence basis is invalidated by this
gate's structural protection. None of the existing KILL/DEFER/FALSIFY rows in
the posterior depend on the absence of this gate; the bug class is
infrastructure-level (research-pipeline-tool source-text shape) and was not
previously used to falsify any substrate paradigm or score claim.

The 77 pre-fix mutated files remain immutable per Catalog #110/#113 APPEND-ONLY
HISTORICAL_PROVENANCE; future invocations on the same dirs will refuse per
the canonical helper's cascade D. The historical artifacts retain their
forensic value (operator can inspect them; they remain on disk; no deletion).

## 4. Per-finding RE-EVAL priority assignment

No historical findings require RE-EVAL because the bug class is purely
infrastructure-level (source-text protection vs scientific claim
invalidation). The structural extinction landing is itself the canonical
remediation; future invocations are protected; pre-fix artifacts are
APPEND-ONLY immutable per Catalog #110/#113.

## Verification

Post-landing preflight on Catalog #381:
- Live count: 0 (all 3 canonical producers now route through the canonical helper)
- 35/35 canonical helper tests pass
- 16/16 STRICT preflight gate tests pass
- 2 sister DERIVED_OUTPUT files (.omx/state/current_focus.md +
  .omx/state/next_experiments.md) now carry the canonical regen header per
  Catalog #113 DerivedOutputGuard

Sister apparatus mutations:
- canonical anti-pattern updated: 2 EmpiricalFalsifications (1 original 77-row
  identification + 1 EmpiricalExtinction ratification)
- canonical equation updated: 2 EmpiricalAnchors (1 original residual=77.0
  + 1 post-fix residual=0.0)

## Cross-references

- CLAUDE.md "Artifact lifecycle compliance" Catalog #113
- CLAUDE.md "Operator gates must be wired and used" non-negotiable
- CLAUDE.md "Bugs must be permanently fixed AND self-protected against" non-negotiable
- Catalog #110 (`check_recovery_metadata_append_only` canonical reference pattern)
- Catalog #287 (placeholder-rationale rejection sister discipline)
- Catalog #348 (this retroactive sweep contract)
- canonical helper: `tac.research_pipeline_output_dir_safety`
- canonical anti-pattern: `research_pipeline_tool_re_writes_historical_provenance_json_with_mutated_fields_v1`
- canonical equation: `historical_provenance_immutability_predicts_zero_in_place_mutation_v1`
- canonical 2-landing pattern: canonical helper + STRICT preflight gate Catalog #381
