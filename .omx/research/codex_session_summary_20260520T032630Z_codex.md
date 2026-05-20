# Codex Session Summary

**UTC:** 2026-05-20T03:26:30Z  
**Owner:** codex  
**Lane:** `lane_v8_learned_compression_faiss_scaffold_codex_20260520`

## Landed

- Canonicalized `compute_pq_mi_verdict` into `tac.optimization.faiss_ivf_pq_atw_channel`.
- Updated the ATW/Faiss disambiguator to consume the canonical helper.
- Added V8 learned-compression Faiss trainer scaffold, fail-closed inflate runtime, disabled operator recipe, and tests.
- Produced local smoke manifest:
  `experiments/results/lane_v8_learned_compression_faiss_smoke_codex_20260520T032000Z/v8_smoke_results.json`.
- Appended Catalog #287 exact waiver rows for newly exposed historical/proposal `.omx/research` phantom-helper citations and re-ran strict scan green.
- Wrote landing memo:
  `.omx/research/codex_findings_v8_faiss_premise_fix_scaffold_landed_20260520T032630Z_codex.md`.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q src/tac/tests/test_atw_v2_1_faiss_ivf_pq.py src/tac/tests/test_probe_atw_v2_1_faiss_pq_disambiguator.py src/tac/tests/test_v8_learned_compression_faiss_scaffold.py -p no:cacheprovider` -> `43 passed in 0.50s`.
- V8 smoke command wrote a fail-closed manifest with `score_claim=false`, `promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`.
- `tools/operator_authorize.py --recipe substrate_v8_learned_compression_faiss_modal_a100_smoke --dry-run --target none` refused dispatch on `dispatch_enabled=false` plus explicit blockers.
- Direct Catalog #287 strict scan -> `catalog-287-strict-ok`.
- `tools/lane_maturity.py validate` -> 1037 lanes valid.
- `tools/canonical_task_status.py --validate` -> valid before this session's terminal row.
- `git diff --check` on touched V8 files -> clean.

## Remaining

- V8 full training remains blocked on categorical posterior, scale hyperprior, byte-closed export, and score-aware eval-roundtrip trainer.
- Do not dispatch V8 until the recipe blockers are cleared and a lane claim is created.
- Active PR101/FEC6 compliance sister work remains outside this write set.
