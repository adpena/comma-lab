# Codex Findings: MLX Cache Auth-Axis Identity

UTC: 2026-05-21T21:03:43Z

## Verdict

NEEDS CONTEST-LINUX CACHE MATERIALIZATION before MLX scorer-input tensors can be
used for contest-CPU transfer calibration.

The FEC6/PR101 local cache is valid for local macOS advisory tensor ingestion,
but it is not byte-identical to the Modal Linux x86_64 contest-CPU auth-eval
inflated output for the same archive SHA. The archive identity matches; the
inflated bytes do not.

## What Landed

- Added `tac.local_acceleration.mlx_cache_audit`.
- Added `tools/audit_mlx_scorer_input_cache.py`.
- Added focused tests in `src/tac/tests/test_mlx_cache_audit.py`.
- The audit fails closed unless archive SHA, inflated-output aggregate SHA, and
  cache pair count match the target auth-eval JSON.
- All outputs preserve the non-authoritative MLX contract:
  `score_claim=false`, `promotion_eligible=false`,
  `rank_or_kill_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

## Empirical Anchor

Cache manifest:

`experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs/manifest.json`

The directory name is legacy/misleading; the manifest is authoritative:
`pair_count=600`.

Cache identity:

- archive SHA-256:
  `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`
- local inflated-output aggregate SHA-256:
  `dbc67c898ecb158912f86c920f09bf2c68307b77c1cec3c1baa27a845d3850f1`
- local raw SHA-256:
  `d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c`
- SegNet cache shape: `(600, 3, 384, 512)`
- PoseNet cache shape: `(600, 12, 192, 256)`

Cache artifacts:

- `segnet_last_rgb.npy`: 1,415,577,728 bytes;
  SHA-256 `eadd0292df3b1f7876c116c059d900234ea7b73a4d15cf271fcd628d4f5d0cbd`
- `posenet_yuv6_pair.npy`: 1,415,577,728 bytes;
  SHA-256 `a19d395759c57814b857d748c38e43e47a55aa16d74e4f1ed46036f13b1dbc6d`
- `pair_indices.npy`: 9,728 bytes;
  SHA-256 `16941927786fd609201a2954f09cf5e2b50ee2eefb850c51306454fa5910091e`

## Audit Results

Local macOS advisory auth-eval:

`experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/local_macos_cpu_advisory_smoke_20260519T143700Z_workdir/contest_auth_eval.json`

Result: `PASS_CACHE_AUTH_EVAL_IDENTITY`.

- score: `0.19206131688110561` `[macOS-CPU advisory]`
- score axis: `cpu_advisory`
- inflated-output aggregate SHA-256:
  `dbc67c898ecb158912f86c920f09bf2c68307b77c1cec3c1baa27a845d3850f1`
- raw SHA-256:
  `d1afc583b01ff4a7aaa844d4f03ece3ed381d56763a06cb2c5e011526e5f868c`

Modal Linux x86_64 contest-CPU auth-eval:

`experiments/results/modal_auth_eval_cpu/archive_6bae0201fb08/contest_auth_eval.json`

Result: `FAIL_CACHE_AUTH_EVAL_IDENTITY`.

- score: `0.1920513168811056` `[contest-CPU]`
- score axis: `contest_cpu`
- blocker: `inflated_outputs_aggregate_sha256_mismatch_or_missing`
- contest-CPU inflated-output aggregate SHA-256:
  `10c68e4266e79fc3e878fd20136e8aaa56262b3a2ff45eed7b8d5a4b1e1ee66d`
- contest-CPU raw SHA-256:
  `fef02ccd53ad4355f2dbb8e0b9cd4efb847daa243bd35a8411c5260d584fda8b`

## Interpretation

The same archive SHA inflates to different raw bytes across local macOS advisory
and Modal Linux x86_64 contest-CPU surfaces. That makes the local cache useful
for local tensor-ingestion debugging and MLX training plumbing, but not for
contest-CPU transfer calibration or scorer-response trust.

The next byte-closed cache must be produced inside the same Linux auth-eval
environment that produces the contest-CPU inflated-output manifest, or the
auth-eval path must preserve enough scorer-input tensors/hashes during the
Modal run to prove identity without returning multi-GB raw frames through a
function result payload.

## Verification

Commands run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_cache_audit.py \
  src/tac/tests/test_mlx_preprocess.py \
  src/tac/tests/test_mlx_scorer_fidelity.py -q
```

Result: `13 passed in 1.38s`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m compileall -q \
  src/tac/local_acceleration/mlx_cache_audit.py \
  tools/audit_mlx_scorer_input_cache.py \
  src/tac/tests/test_mlx_cache_audit.py
```

Result: pass.

```bash
git diff --check -- \
  src/tac/local_acceleration/mlx_cache_audit.py \
  tools/audit_mlx_scorer_input_cache.py \
  src/tac/tests/test_mlx_cache_audit.py
```

Result: pass.

## Recommended Next Action

Extend the Modal/Linux auth-eval path with an opt-in scorer-input cache or
compact scorer-input hash export. That is the next required bridge from
local MLX throughput to contest-CPU-faithful training signal.
