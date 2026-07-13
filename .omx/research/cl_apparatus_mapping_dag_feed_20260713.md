# DAG FEED — continual-learning apparatus mapping

**Date:** 2026-07-13  
**Status:** staged update only; `research_only=true`; uncommitted; shared DAG read-only  
**Source:** `.omx/research/cl_apparatus_mapping_20260713.md`

## FEED-cl-apparatus-mapping-20260713

```yaml
feed_id: FEED-cl-apparatus-mapping-20260713
kind: research_design
authority: apparatus_only
pointer_delta: 0
score_claim: false
source:
  - arXiv:2607.07847
  - .omx/research/cl_apparatus_mapping_20260713.md
consumes:
  - ref:#411
  - ref:#346
  - ref:#436
  - .omx/research/graph_memory_dag_reconstruction_20260710.md
  - .omx/research/lens_retrieval_346_wirein_landed_20260712.md
  - .omx/research/organ_regime_conditional_dispatch_436_20260711.md
  - .omx/research/aniso_perclass_lambda_433_20260711.md
  - .omx/research/fore_occupancy_ratio_dig_20260713.md
  - .omx/research/hcm_causal_attribution_dig_20260713.md
derived_verdicts:
  prompt_retrieval_surface: GAPPED
  distilled_registry_surface: MATCHED_ACCUMULATION_GAPPED_STALE_UPDATE
  organ_rl_surface: GUARDED_PREDICTIVE_ONLY_RL_NOT_IDENTIFIED
  gepa_reconciliation: RESCOPE_NOT_CONFIRM
biggest_gap: >-
  No lineage-aware change-pattern and mechanism-selection loop is evaluated on
  current adaptation, prior-lineage retention, and future-lineage transfer.
proposes:
  - cl_lineage_change_mechanism_selector_v1
  - cl_executable_supersession_reactivation_guard_v1
  - gepa_sequential_lineage_forward_transfer_reprobe_v1
edges:
  - "#411/#346 retrieval -> reversible cold-start prior"
  - "lineage/change classifier -> mechanism eligibility"
  - "mechanism eligibility -> #436 within-run regime dispatch"
  - "#436 outcome -> union-residual + surprise gate"
  - "transition-complete multi-run ledger -> FORE support -> HCM uncertainty -> RL credit"
  - "canonical record -> executable supersession/reactivation guard -> consumers"
guards:
  - real_only
  - no_shared_update
  - no_launch
  - no_equation_minted
  - rl_unreachable_without_fore_hcm_support
  - current_retention_forward_non_regression
triality:
  dsl: N/A; no typed apparatus selector exists and no flag may be invented
  dag: this staged FEED; main-agent flip required
  equations: N/A; no contest law or empirical coefficient measured
staged_candidate_rows: .omx/research/cl_apparatus_mapping_candidate_rows_20260713.jsonl
papers_update_note: .omx/research/cl_apparatus_gepa_papers_checked_update_20260713.md
```

## Main-agent flip instructions

After adversarial review, append an equivalent FEED to the canonical DAG using its existing helper
and register only those candidate rows whose ownership does not collide with a live lane. Do not
reinterpret this staged FEED as evidence that any proposed selector, guard, or probe has been built.
