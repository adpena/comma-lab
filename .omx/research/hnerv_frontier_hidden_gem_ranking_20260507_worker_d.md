# HNeRV Frontier Hidden-Gem Ranking - 2026-05-07 Worker D

## Scope

Public-frontier custody and roadmap update only. This pass did not dispatch
remote work, did not queue exact eval, and did not claim any score beyond
existing exact CUDA artifacts.

## Current Exact Frontier Row

- label: `PR106x-lowlevel-brotli`
- score: `0.20935073680571203`
- evidence: `A++`
- archive bytes: `186080`
- archive SHA-256: `b0a12549a39e34a0d7f83ea99e05e55fcd01d795a15db2ffb3d92ccc6267e53f`
- eval artifact:
  `experiments/results/lightning_batch/exact_eval_pr106x_lowlevel_brotli_repack_custody_v2_t4_20260506/contest_auth_eval.adjudicated.json`
- scope: `exact_local_cuda_custody_lossless_repack_control`

This row is the current scorecard frontier because it is the lowest
canonical-frontier-eligible exact CUDA row in the local HNeRV scorecard input.
It remains a lossless low-level repack control, not evidence that Brotli
structural recoding supersedes categorical, range-coded, or scorer-changing
HNeRV families.

## Next Exact-Evaluable Target

The next score-lowering target closest to the frontier is the current frontier
archive's largest charged section:

- target label: `PR106x-lowlevel-brotli`
- target section: `decoder_packed_brotli`
- optimization role: `decoder_weight_stream`
- section bytes: `170127`
- section SHA-256:
  `07725c39ff436195e319f258b1e033290de30e259bc3f103b1b487f21a698c5c`
- payload SHA-256:
  `b6ba493aa37446143b003235eaeb3a49c0748a6d64392fb9a666a6c872629171`
- score gap to current frontier: `0.0`
- estimated rate mass if the entire section were removed:
  `0.11328058611781565`

Required next gate: build a byte-different archive with old/new section
SHA-256 and charged-byte proof, then run exact CUDA auth eval only after the
Level 2 lane-claim protocol. No dispatch was attempted in this pass.

## Scorecard Improvement

`experiments/build_hnerv_frontier_scorecard.py` now emits:

- `current_frontier`
- `next_exact_evaluable_target`
- `hidden_gem_byte_mass_ranking`

The new ranking sorts by exact-frontier proximity first and charged byte mass
second. This prevents a stale near-frontier predecessor such as PR106/PR106x
from outranking the actual current frontier row just because its decoder
section is 151 bytes larger than the already exact-evaluated low-level Brotli
control.

## Verification

Commands run locally:

```bash
.venv/bin/python -m pytest src/tac/tests/test_build_hnerv_frontier_scorecard.py -q
.venv/bin/ruff check experiments/build_hnerv_frontier_scorecard.py src/tac/tests/test_build_hnerv_frontier_scorecard.py
.venv/bin/python -m py_compile experiments/build_hnerv_frontier_scorecard.py
.venv/bin/python tools/audit_hnerv_frontier_scorecard.py --format json
.venv/bin/python experiments/build_hnerv_frontier_scorecard.py \
  --profile-json experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/profiles.json \
  --profile-json experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/xrepack_profiles.json \
  --profile-json experiments/results/public_hnerv_frontier_payload_profiles_20260504_codex/lowlevel_brotli_profiles.json \
  --candidate-manifest experiments/results/hnerv_lowlevel_repack_pr106x_20260506_codex/result.json \
  --json-out /tmp/worker_d_hnerv_scorecard.json \
  --md-out /tmp/worker_d_hnerv_scorecard.md \
  PR105=experiments/results/lightning_batch/exact_eval_public_pr105_kitchen_sink_adapter_t4_20260504T1330Z/contest_auth_eval.adjudicated.json \
  PR105x=experiments/results/lightning_batch/exact_eval_public_pr105_kitchen_sink_xrepack_t4_20260504T1342Z/contest_auth_eval.adjudicated.json \
  PR106=experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_adapter_t4_20260504T1330Z/contest_auth_eval.adjudicated.json \
  PR106x=experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_xrepack_t4_20260504T1342Z/contest_auth_eval.adjudicated.json \
  PR106x-lowlevel-brotli=experiments/results/lightning_batch/exact_eval_pr106x_lowlevel_brotli_repack_custody_v2_t4_20260506/contest_auth_eval.adjudicated.json
```

Results:

- focused pytest: `6 passed`
- ruff: `All checks passed`
- `py_compile`: passed
- `tools/audit_hnerv_frontier_scorecard.py --format json`: `ready_for_hidden_gem_routing=true`,
  `blockers=[]`, `dispatch_attempted=false`, `score_claim=false`
- temp scorecard selected `PR106x-lowlevel-brotli / decoder_packed_brotli`
  as `next_exact_evaluable_target`
