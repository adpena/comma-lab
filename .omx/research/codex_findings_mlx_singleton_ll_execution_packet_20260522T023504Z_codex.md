# Codex Findings: MLX Singleton LL Execution Packet

UTC: 2026-05-22T02:35:04Z

## Verdict

PROCEED for singleton CPU MLX scorer-response rows as non-authoritative LL
surrogate-planner input.

The generated packet is a local research-signal artifact, not a score claim.
It is suitable for expanding the scorer-response dataset toward the planner's
`>=50 MLX rows across stable CPU windows/families` gate.

## Source Caches

- Reference cache:
  `experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600`
- Candidate cache:
  `experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs`
- Candidate archive SHA-256:
  `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- Candidate inflated aggregate SHA-256:
  `dbc67c898ecb158912f86c920f09bf2c68307b77c1cec3c1baa27a845d3850f1`
- Candidate raw SHA-256:
  `d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c`
- Empirical ZIP bytes:
  `178517` from `wc -c experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/archive.zip`

The ZIP member `x` remains `178417` stored bytes; scorer rate uses the
`archive.zip` byte count.

## Singleton Profile And Response

Artifact root:

`experiments/results/mlx_singleton_execution_guard_fec6_20260522T0230Z/`

Profile command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/profile_mlx_scorer_response_cache.py \
  --reference-cache-dir experiments/results/mlx_scorer_input_cache_reference_video_20260521T2304Z_full600 \
  --candidate-cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --archive-size-bytes 178517 \
  --output experiments/results/mlx_singleton_execution_guard_fec6_20260522T0230Z/profile_singleton_pairs16_20.json \
  --repo-root . \
  --batch-pairs 1 \
  --devices cpu \
  --start-pair 16 \
  --max-pairs 4 \
  --repeat 1
```

Profile result:

- `batch_pairs`: 1
- `device`: CPU
- pair window: `[16, 20]`
- pairs/sec: `0.9194546301811687`
- local MLX score: `0.17819642970473418`
- avg PoseNet distortion: `0.000006132257439617206`
- avg SegNet distortion: `0.0005149841381353326`
- `score_claim=false`

Response artifact:

`experiments/results/mlx_singleton_execution_guard_fec6_20260522T0230Z/response_singleton_pairs16_20.json`

Component artifacts:

- `components/posenet_distortion.npy`
- `components/segnet_distortion.npy`

## Contract Fix

`src/tac/local_acceleration/mlx_production_contract.py` now accepts reference
cache identity proven by scorer-input array SHA-256 fields when the reference
is the original video and therefore has no archive/raw custody SHA. Candidate
cache identity still requires archive/raw custody hashes.

This aligns the production contract with the scorer-response dataset normalizer
and avoids rejecting valid original-video reference caches.

## Gates

Profile stability:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/check_mlx_scorer_response_profile_stability.py \
  --profile experiments/results/mlx_singleton_execution_guard_fec6_20260522T0230Z/profile_singleton_pairs16_20.json \
  --output experiments/results/mlx_singleton_execution_guard_fec6_20260522T0230Z/stability_singleton_pairs16_20.json \
  --baseline-device cpu \
  --baseline-batch-pairs 1 \
  --run-id fec6_singleton_cpu_pairs16_20_20260522T0230Z
```

Result: `PASS_MLX_PROFILE_STABILITY`.

Production contract:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/check_mlx_scorer_production_contract.py \
  --response experiments/results/mlx_singleton_execution_guard_fec6_20260522T0230Z/response_singleton_pairs16_20.json \
  --output experiments/results/mlx_singleton_execution_guard_fec6_20260522T0230Z/production_contract_singleton_pairs16_20.json \
  --profile-stability experiments/results/mlx_singleton_execution_guard_fec6_20260522T0230Z/stability_singleton_pairs16_20.json \
  --batch-invariance experiments/results/mlx_batch_invariance_clean_head_52093e425_cpu_fec6_pr101_pair208_210_20260522T021300Z.json \
  --run-id fec6_singleton_cpu_pairs16_20_20260522T0230Z
```

Result: `PASS_MLX_SCORER_PRODUCTION_CONTRACT`.

Focused test command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_production_contract.py \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_mlx_execution_plan.py \
  src/tac/tests/test_mlx_profile_stability.py \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py
```

Result: 37 passed.

## LL Planner Output

Dataset:

`experiments/results/mlx_singleton_execution_guard_fec6_20260522T0230Z/scorer_response_dataset_singleton_pairs16_20.json`

LL plan:

`experiments/results/mlx_singleton_execution_guard_fec6_20260522T0230Z/ll_next_probe_plan_singleton_pairs16_20.json`

The LL planner accepted the attached clean-head singleton CPU parity sweep:

- source verdict: `PASS_MLX_TORCH_SCORER_PARITY_SWEEP`
- covered pair window: `[0, 300]`
- windows passed: 300
- SegNet argmax mismatch pixels total: 0
- MLX rows allowed for planner: true

Top planner probe:

`ll_mlx_cpu_stable_response_harvest`

Acceptance gate:

`>=50 MLX rows across stable CPU windows/families, all score_claim=false, with parity-gated held-out correlation before any exact-eval dispatch selection`

## Authority Contract

All outputs remain local MLX research-signal artifacts:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- no leaderboard, PR, promotion, or rank/kill decision may use these as score
  authority.

## Next Action

Expand from this single singleton CPU row to a windowed harvest of at least 50
rows, using `batch_pairs=1` by default and the clean parity sweep as the gate.
Only after held-out correlation is measured should the output influence spend
filters for exact contest eval.
