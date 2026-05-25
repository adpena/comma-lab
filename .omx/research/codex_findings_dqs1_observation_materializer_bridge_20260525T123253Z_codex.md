# Codex Findings: DQS1 Observations Into Materializer Bridge

- UTC: 2026-05-25T12:32:53Z
- Updated: 2026-05-25T12:37:22Z
- Lane: `codex_dqs1_observation_materializer_bridge_20260525`
- Scope: DQS1 harvest observations, materializer-feedback bridge, queue builder, unattended tranche rebuild path.

## Findings

1. The prior bridge preserved saturated materializer signal into DQS1 queue metadata, but completed DQS1 local-first harvest observations still stopped at the portfolio planner path.
2. `tac.optimization.dqs1_materializer_feedback_bridge` now accepts canonical DQS1 local-first harvest observation rows and records observed DQS1 candidate rows, best observed local-advisory candidate, outcome counts, source JSONL paths, and ignored non-DQS1 rows.
3. `tools/build_dqs1_local_first_queue.py` now accepts repeated `--dqs1-observation-jsonl` / `--dqs1-observations` inputs. The loader fails closed on missing files, non-JSONL summary/raw JSON, non-local-first DQS1 rows, and deduplicates cumulative snapshots by canonical observation identity.
4. `tools/run_dqs1_local_first_tranche.py` forwards existing observation JSONLs and automatically forwards the current round's generated harvest observation JSONL into the queue rebuild.
5. All emitted bridge/queue records keep score, promotion, rank/kill, dispatch, and exact-eval authority false. This is local replanning signal only.

## Durable Proof

- Strict bridge:
  `.omx/research/codex_dqs1_observation_materializer_bridge_20260525T123722Z/dqs1_materializer_feedback_bridge.strict_observation_feedback.json`
  - SHA-256: `006678af72c3ecb2909191cd9f6ddbad40dce078dffc815c7fbda21405c7de84`
  - `observed_dqs1_candidate_count`: 20
  - `observed_dqs1_observation_count`: 20
  - input used two cumulative JSONLs (16-row and 20-row snapshots), proving dedupe avoids inflated evidence counts.
  - `dqs1_harvest_outcome_counts`: 2 improved, 18 regressed, 0 flat/byte-only
  - best observed local-advisory candidate: `pairset_drop_two_r029_017_p0259_0242`
  - recommended next action: `continue_dqs1_pairset_composition_from_positive_harvest_signal`
- Strict queue:
  `.omx/research/codex_dqs1_observation_materializer_bridge_20260525T123722Z/dqs1_pairset_local_first.strict_observation_feedback.json`
  - SHA-256: `077285bfc0035fb235297cfdaa2be62f720047d49e3a9679624c1f29aebe2871`
  - `queue_id`: `dqs1_pairset_local_first_strict_observation_feedback`
  - `experiment_queue.v1`, 4 experiments, 28 steps, validation green.

## Verification

- `.venv/bin/python -m ruff check src/tac/optimization/dqs1_materializer_feedback_bridge.py src/comma_lab/scheduler/dqs1_local_first_queue.py tools/build_dqs1_local_first_queue.py tools/run_dqs1_local_first_tranche.py src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_dqs1_local_first_tranche.py --no-cache`
- `.venv/bin/python -m pytest -q src/tac/tests/test_dqs1_local_first_queue_builder.py src/tac/tests/test_dqs1_local_first_tranche.py src/tac/tests/test_dqs1_local_first_harvest_observations.py src/tac/tests/test_dynamic_sparse_gate_oracle.py src/tac/tests/test_byte_shaving_materializer_campaign_runner.py --durations=30 --durations-min=0.01`
- `.venv/bin/python tools/build_dqs1_local_first_queue.py --action-summary latest --candidate-limit 4 --materializer-feedback experiments/results/codex_current_cpu_frontier_rate_attack_receiver_smoke_20260525T120353Z/header_elide/sweep.json --materializer-feedback experiments/results/codex_current_cpu_frontier_rate_attack_receiver_smoke_20260525T120353Z/recompress/sweep.json --dqs1-observations .omx/research/dqs1_local_first_harvest_observations_20260523T133010Z.jsonl --dqs1-observations .omx/research/dqs1_local_first_harvest_observations_20260524T065107Z.jsonl --materializer-feedback-bridge-out .omx/research/codex_dqs1_observation_materializer_bridge_20260525T123722Z/dqs1_materializer_feedback_bridge.strict_observation_feedback.json --write --output .omx/research/codex_dqs1_observation_materializer_bridge_20260525T123722Z/dqs1_pairset_local_first.strict_observation_feedback.json`
- `.venv/bin/python tools/experiment_queue.py --queue .omx/research/codex_dqs1_observation_materializer_bridge_20260525T123722Z/dqs1_pairset_local_first.strict_observation_feedback.json validate`
- `rg -n '"(score_claim|promotion_eligible|rank_or_kill_eligible|ready_for_exact_eval_dispatch|dispatch_attempted|gpu_launched)": true' .omx/research/codex_dqs1_observation_materializer_bridge_20260525T123722Z/dqs1_materializer_feedback_bridge.strict_observation_feedback.json .omx/research/codex_dqs1_observation_materializer_bridge_20260525T123722Z/dqs1_pairset_local_first.strict_observation_feedback.json`
- `git diff --check`

## Next

Consume the observed-DQS1 bridge fields directly in the next acquisition layer so it chooses between continuing pairset composition, widening pairset composition, or reopening receiver-positive materializers without relying on memo text.
