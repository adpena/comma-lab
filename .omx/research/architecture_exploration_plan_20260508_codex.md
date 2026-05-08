# Architecture Exploration Plan - 2026-05-08 Codex

## Scope

Bounded lane: inspect architecture-shrink, self-compression, sparsity, MPS
research-signal tooling, and local Lightning arch-shrink state; add one
zero/low-cost artifact that improves planning without launching GPU jobs or
claiming score.

No GPU jobs were launched. No `.omx/state` dispatch files were modified.

## Inspected Surfaces

- `tools/plan_pr101_arch_shrink_retraining.py`: CPU-only PR101 HNeRV
  architecture-shrink/rate-side planner. Output remains `score_claim=false`
  and `ready_for_exact_eval_dispatch=false`.
- `tools/build_hnerv_arch_shrink_driver.py`: generated-schema checkpoint
  handoff. It produces no archive and blocks dispatch until runtime export,
  local parity, strict compliance, lane claim, and exact CUDA auth eval exist.
- `tools/pr101_sparsity_block_sweep.py`: CPU post-hoc sparsity byte anchor.
  Score impact is explicitly unknown without retraining and contest CUDA.
- `src/tac/optimization/mps_research_signal.py` and
  `tools/build_mps_research_signal_manifest.py`: fail-closed MPS proxy curve
  schema. Rows are candidate-generation priors only.
- `src/tac/self_compress.py`,
  `experiments/plan_full_pipeline_self_compression_nextwave.py`, and local
  self-compress summaries: useful payload/curve signals, not score evidence.
- `.omx/state/lightning_active_jobs.json` and
  `.omx/state/active_lane_dispatch_claims.md`: latest local arch-shrink job is
  `arch-shrink-x0-4-lightning-20260508T024304Z`, submitted
  `2026-05-08T02:43:10Z`, with no terminal status in local active-job state.
  The corresponding claim is active until `2026-05-08T20:43:09Z`.

## Concrete Artifact

Added a reusable normalizer:

- `src/tac/optimization/architecture_sweep_recommendations.py`
- `tools/build_architecture_sweep_recommendations.py`
- `src/tac/tests/test_architecture_sweep_recommendations.py`

Generated local research artifacts:

- `.omx/research/architecture_sweep_recommendations_20260508_codex.json`
- `.omx/research/architecture_sweep_recommendations_20260508_codex.md`

The generated manifest normalizes five local source artifacts:

- `experiments/results/mps_research_signal_smoke_20260507_codex/manifest.json`
- `experiments/results/pr101_arch_shrink_retraining_plan_20260507_worker_b/plan.json`
- `reports/raw/pr101_sparsity_sweep_20260508T002611Z/manifest.json`
- `experiments/results/local_smoke_coolchic_c3_20260425/self_compress/summary.json`
- `experiments/results/lane_e_self_compression/results.json`

Result: 22 rows, 8 curves, all `score_claim=false`,
`promotion_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

## Recommendations

1. Do not duplicate the active arch-shrink Lightning dispatch. Poll or harvest
   only the existing `arch-shrink-x0-4-lightning-20260508T024304Z` lane when it
   is terminal.
2. Use normalized CPU/MPS curves for local build order only:
   architecture-shrink width/precision, post-hoc sparsity, renderer
   self-compression payload bytes, and renderer-bin recompression headroom.
3. Treat the MPS `arch_shrink/width_x0_4` curve and CPU sparsity/self-compress
   rows as research-signal priors. They cannot rank, kill, or promote methods.
4. Exact CUDA promotion gates remain mandatory before score use:
   produce a score-affecting archive, record archive bytes/SHA/runtime custody,
   run strict pre-submission compliance, claim the dispatch lane, run full
   sample contest CUDA auth eval, then adjudicate components/rate/payload
   closure.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_architecture_sweep_recommendations.py`
  passed: 5 tests.
- `.venv/bin/python -m py_compile src/tac/optimization/architecture_sweep_recommendations.py tools/build_architecture_sweep_recommendations.py`
  passed.
- The normalizer CLI was run locally only and wrote the `.omx/research`
  recommendation JSON/Markdown listed above.
