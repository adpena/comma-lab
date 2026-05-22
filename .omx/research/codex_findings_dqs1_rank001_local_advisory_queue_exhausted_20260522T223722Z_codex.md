# Codex Findings - DQS1 Rank001 Local Advisory + Queue Exhaustion - 2026-05-22

## Scope

Follow-up bounded local-first sweep after commit `392c79dd5` and the automatic
rank015 ledger commit `36e535db0`.

## Rank001 harvest

- Candidate: `pairset_drop_one_rank001_pair0501`
- Archive SHA:
  `dd5f2b190bbbc6de04d373713817f7605ae3889759ef2ee0ef0f33b3e7f533a7`
- Local score `[macOS-CPU advisory]`: `0.19204128295713674`
- Current contest-CPU frontier used for drift projection:
  `0.19202828295713675`
- Projected contest score: `0.19203128295713673`
- Conservative projected contest score: `0.19203428295713673`
- Eureka margin: `-0.000005999999999978245`
- Verdict: `observe_only`; no exact CPU/CUDA spend trigger.

Artifacts:

- `.omx/research/local_cpu_contest_drift_eureka_pairset_drop_one_rank001_pair0501_20260522T223722Z.json`
- `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank001_pair0501/local_cpu_advisory.json`
- `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/pareto_gap_uleb/materialized/drop_rank001_pair0501/locality_controls.json`

## Queue state

The current DQS1 local-first action summary has no remaining safe unobserved
rows after rank001:

- observed: ranks 023, 024, 018, 017, 016, 025, 015, and 001
- `tools/build_dqs1_local_first_queue.py --action-summary latest --write`
  correctly refused to produce a new queue target

## Integration note

While rank001 ran, concurrent hardening edits appeared in the working tree. They
were reviewed and integrated instead of discarded:

- nested authority fields now reject string/numeric truthy values, not only
  literal `True`
- contest auth bridge rankability accepts normalized contest CPU/CUDA axis
  variants
- eureka-gated DQS1 queue completion refuses to advance past a candidate whose
  eureka signal requests exact auth dispatch
- candidate queue identity preserves legacy A1 rollup + M5 merge behavior while
  still separating representation-family rows
