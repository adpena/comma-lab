# Codex Findings: DQS1 Acquisition Observation Filter

- UTC: 2026-05-25T12:59:55Z
- Lane: `codex_dqs1_acquisition_observation_filter_20260525`
- Scope: DQS1 pairset acquisition generation, portfolio/action summary, local-first queue handoff.

## Findings

1. The DQS1 queue can now suppress already observed local-first candidates, but the upstream pairset acquisition plan could still emit and rank those same candidates. That left planner, portfolio, and queue disagreeing until the final queue-selection filter.
2. `tac.optimization.decoder_q_pairset_acquisition` now accepts canonical DQS1 local-first harvest observation rows, rejects truthy authority fields, and suppresses matching `acquisition_id`/`candidate_id` rows by default.
3. `tools/plan_decoder_q_pairset_acquisition.py` now accepts repeated `--dqs1-observation-jsonl` / `--dqs1-observations` inputs and deduplicates cumulative JSONLs by canonical observation identity. `--include-observed-dqs1-candidate` is the explicit replay/debug override.
4. Acquisition plans now record `dqs1_pairset_acquisition_observation_skip.v1` in `selection_policy.observation_skip`, plus observed/suppressed counts and candidate IDs in `summary`.
5. The filtered acquisition plan was fed through cross-family portfolio generation and then into DQS1 queue generation. Suppressed candidate IDs do not reappear in filtered acquisition candidates or top operator actions, and the resulting queue validates.

## Durable Proof

- Acquisition plan:
  `.omx/research/codex_dqs1_acquisition_observation_filter_20260525T125955Z/dqs1_pairset_acquisition_observation_filtered.json`
  - SHA-256: `7a4598908b1baba2346ceed52307ea1b638c96adc5f857c06e293d6e9809d00f`
  - unfiltered candidate count: 180
  - filtered candidate count: 168
  - suppressed observed candidate count: 12
  - suppressed overlap remaining in plan: 0
- Portfolio/action summary:
  `.omx/research/codex_dqs1_acquisition_observation_filter_20260525T125955Z/portfolio.observation_filtered.json`
  - SHA-256: `d4a886f878fafa1eb8e81432ba383a0fdcbd49d63a1c2cb17ee2bdff46e201ef`
  `.omx/research/codex_dqs1_acquisition_observation_filter_20260525T125955Z/action_summary.observation_filtered.json`
  - SHA-256: `b385e1e7c896dd028bce4677ec4d0e6f4fef3163342d96be51584a9d203ae3b2`
  - suppressed overlap in top actions: 0
- Queue:
  `.omx/research/codex_dqs1_acquisition_observation_filter_20260525T125955Z/dqs1_pairset_local_first.filtered_acquisition_queue.json`
  - SHA-256: `c255d821127fa83f75c59af8ccddd52acc5121cc27d9e7c32fd0fbe6e8a0afdd`
  - selected candidates: `pairset_diversity_k002`, `pairset_diversity_k004`, `pairset_diversity_k008`, `pairset_diversity_k012`
  - `experiment_queue.v1`, 4 experiments, 28 steps, validation green
- Bridge:
  `.omx/research/codex_dqs1_acquisition_observation_filter_20260525T125955Z/dqs1_materializer_feedback_bridge.filtered_acquisition_queue.json`
  - SHA-256: `910951c6d2dcf6cd230b7b2246b2a14eea9b0ac3e47367d0391a4788444fc100`
  - `observed_dqs1_candidate_count`: 20

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/decoder_q_pairset_acquisition.py tools/plan_decoder_q_pairset_acquisition.py src/tac/tests/test_decoder_q_pairset_acquisition.py --no-cache`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_decoder_q_pairset_acquisition.py -q`
- `.venv/bin/python tools/plan_decoder_q_pairset_acquisition.py --selector-pareto experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/selector_pareto/dqs1_gap_uleb_selector_pareto.json --dqs1-observations .omx/research/dqs1_local_first_harvest_observations_20260523T133010Z.jsonl --dqs1-observations .omx/research/dqs1_local_first_harvest_observations_20260524T065107Z.jsonl --json-out .omx/research/codex_dqs1_acquisition_observation_filter_20260525T125955Z/dqs1_pairset_acquisition_observation_filtered.json --md-out .omx/research/codex_dqs1_acquisition_observation_filter_20260525T125955Z/dqs1_pairset_acquisition_observation_filtered.md`
- `.venv/bin/python tools/plan_cross_family_candidate_portfolio.py --incumbent-score 0.1921 --pairset-acquisition .omx/research/codex_dqs1_acquisition_observation_filter_20260525T125955Z/dqs1_pairset_acquisition_observation_filtered.json --observation-jsonl .omx/research/dqs1_local_first_harvest_observations_20260523T133010Z.jsonl --observation-jsonl .omx/research/dqs1_local_first_harvest_observations_20260524T065107Z.jsonl --top-k 32 --top-actions 12 --json-out .omx/research/codex_dqs1_acquisition_observation_filter_20260525T125955Z/portfolio.observation_filtered.json --md-out .omx/research/codex_dqs1_acquisition_observation_filter_20260525T125955Z/portfolio.observation_filtered.md --summary-json-out .omx/research/codex_dqs1_acquisition_observation_filter_20260525T125955Z/action_summary.observation_filtered.json`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary .omx/research/codex_dqs1_acquisition_observation_filter_20260525T125955Z/action_summary.observation_filtered.json --candidate-limit 4 --materializer-feedback experiments/results/codex_current_cpu_frontier_rate_attack_receiver_smoke_20260525T120353Z/header_elide/sweep.json --materializer-feedback experiments/results/codex_current_cpu_frontier_rate_attack_receiver_smoke_20260525T120353Z/recompress/sweep.json --dqs1-observations .omx/research/dqs1_local_first_harvest_observations_20260523T133010Z.jsonl --dqs1-observations .omx/research/dqs1_local_first_harvest_observations_20260524T065107Z.jsonl --materializer-feedback-bridge-out .omx/research/codex_dqs1_acquisition_observation_filter_20260525T125955Z/dqs1_materializer_feedback_bridge.filtered_acquisition_queue.json --write --output .omx/research/codex_dqs1_acquisition_observation_filter_20260525T125955Z/dqs1_pairset_local_first.filtered_acquisition_queue.json --queue-id dqs1_pairset_local_first_filtered_acquisition`
- `.venv/bin/python tools/experiment_queue.py --queue .omx/research/codex_dqs1_acquisition_observation_filter_20260525T125955Z/dqs1_pairset_local_first.filtered_acquisition_queue.json validate`
- `rg -n '"(score_claim|promotion_eligible|rank_or_kill_eligible|ready_for_exact_eval_dispatch|dispatch_attempted|gpu_launched)": true' .omx/research/codex_dqs1_acquisition_observation_filter_20260525T125955Z`

## Next

Wire optional pairset-acquisition refresh into the DQS1 tranche runner so long-running local-first campaigns can regenerate filtered acquisition plans automatically before portfolio rebuilds.
