# Codex Findings: DQS1 Tranche Refresh + Frontier Bootstrap Wiring

UTC: 2026-05-25T13:14:28Z

## Verdict

The DQS1 local-first tranche runner was still carrying harvest observations only
into queue rebuilds. That avoided repeated queue selection, but it left the
pairset-acquisition and portfolio layers anchored to a stale acquisition JSON.
This turn wires the observation signal one layer earlier: after each harvest
observation build, the runner can regenerate a filtered
`decoder_q_pairset_acquisition.v1` and then use that refreshed plan for the
portfolio and queue rebuild.

The dirty `frontier_rate_attack_bootstrap.py` and companion CLI were also a real
orphan risk: the CLI depended on a scheduler module that was not tracked or
tested. They are now integrated as scheduler helpers, covered by tests, and the
queue-owned bootstrap metadata is stamped onto each generated experiment so
normal queue normalization cannot drop the final-rate attack context.

## Durable Proof

- Refreshed DQS1 acquisition:
  `.omx/research/codex_dqs1_tranche_acquisition_refresh_frontier_bootstrap_20260525T131428Z/dqs1_pairset_acquisition_observation_refreshed.json`
  - SHA-256: `661bae0c051000bc77c66fd118b25f89808be82ae6be00679091201c60388bc0`
  - Unfiltered candidates: `547`
  - Filtered candidates: `527`
  - Suppressed observed DQS1 candidates: `20`
- Refreshed DQS1 portfolio/action queue:
  `.omx/research/codex_dqs1_tranche_acquisition_refresh_frontier_bootstrap_20260525T131428Z/dqs1_pairset_local_first.observation_refreshed_queue.json`
  - SHA-256: `8e6808c4a2847303da7269d6bb46abb06437a0be5b362beef2fa096078c173f9`
  - Queue validation: `4` experiments, `28` steps
  - Selected candidates:
    `pairset_drop_one_rank023_pair0440`,
    `pairset_drop_one_rank024_pair0112`,
    `pairset_drop_one_rank018_pair0588`,
    `pairset_drop_one_rank017_pair0242`
- Frontier final-rate attack bootstrap:
  `.omx/research/codex_dqs1_tranche_acquisition_refresh_frontier_bootstrap_20260525T131428Z/frontier/frontier_rate_attack_bootstrap.json`
  - SHA-256: `c9fabf0949f424ecb4e87cef6b3bc518bb9b4b4aa1d49340390478aff6856c02`
  - Source archive: `submissions/robust_current/archive_correct.zip`
  - Executable local materializer targets:
    `packet_member_zip_header_elide_v1`, `packet_member_recompress_v1`
  - Queue validation: `2` experiments, `2` steps

All generated artifacts keep `score_claim=false`,
`promotion_eligible=false`, `rank_or_kill_eligible=false`, and
`ready_for_exact_eval_dispatch=false`. A recursive proof grep found no truthy
score, promotion, dispatch, GPU-launch, or exact-readiness authority field under
the proof directory.

## Verification

- `.venv/bin/python -m ruff check tools/run_dqs1_local_first_tranche.py src/tac/tests/test_dqs1_local_first_tranche.py src/comma_lab/scheduler/frontier_rate_attack_bootstrap.py tools/build_frontier_final_rate_attack_queue.py src/tac/tests/test_frontier_rate_attack_bootstrap.py --no-cache`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_dqs1_local_first_tranche.py src/tac/tests/test_frontier_rate_attack_bootstrap.py -q`
- `.venv/bin/python tools/experiment_queue.py --queue .omx/research/codex_dqs1_tranche_acquisition_refresh_frontier_bootstrap_20260525T131428Z/dqs1_pairset_local_first.observation_refreshed_queue.json validate`
- `.venv/bin/python tools/experiment_queue.py --queue .omx/research/codex_dqs1_tranche_acquisition_refresh_frontier_bootstrap_20260525T131428Z/frontier/experiment_queue.json validate`
- `rg -n '"(score_claim|promotion_eligible|rank_or_kill_eligible|ready_for_exact_eval_dispatch|dispatch_attempted|gpu_launched)"\s*:\s*true' .omx/research/codex_dqs1_tranche_acquisition_refresh_frontier_bootstrap_20260525T131428Z`

## Next Step

The next high-EV continuation is to let the frontier final-rate attack bootstrap
harvest its completed materializer sweeps back into the same feedback bridge
used by DQS1 local-first queues, so real final-rate materializer observations
become acquisition signal rather than remaining local proof artifacts.
