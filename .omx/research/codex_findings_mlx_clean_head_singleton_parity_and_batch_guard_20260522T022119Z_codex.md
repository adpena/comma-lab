# Codex Findings: MLX Clean-HEAD Singleton Parity + Batch Guard

UTC: 2026-05-22T02:21:19Z

## Verdict

PROCEED for singleton CPU MLX scorer-response use as non-authoritative local
training/candidate-generation signal.

DO NOT use multi-pair CPU batches as production signal unless the exact
device/batch shape has a passing batch-invariance gate. A clean-HEAD
batch-4 sweep found a real SegNet argmax mismatch.

## Clean Snapshot

- HEAD: `52093e425c960d9b148b2473c3da3fea219a1499`
- Snapshot: `/tmp/pact_head_mlx_verify_20260522T020141Z`
- Upstream scorer snapshot copied from `/Users/adpena/Projects/pact/upstream`
- Cache: `experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs`
- Shared checkout dirty WIP was not used for the parity sweeps.

## Positive Anchor: Full-Cache Singleton CPU Sweep

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/pact_head_mlx_verify_20260522T020141Z/src \
/Users/adpena/Projects/pact/.venv/bin/python \
/tmp/pact_head_mlx_verify_20260522T020141Z/tools/audit_mlx_scorer_torch_parity_sweep.py \
  --cache-dir /Users/adpena/Projects/pact/experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --repo-root /tmp/pact_head_mlx_verify_20260522T020141Z \
  --device cpu \
  --start-pair 0 \
  --max-pairs 300 \
  --window-pairs 1 \
  --stride-pairs 1 \
  --progress-every 50 \
  --run-id fec6_pr101_full300pairs_clean_head_52093e425_cpu_singleton_20260522T021600Z \
  --output /Users/adpena/Projects/pact/experiments/results/mlx_torch_parity_sweep_clean_head_52093e425_cpu_fec6_pr101_singleton_full300pairs_20260522T021600Z.json
```

Result:

- Verdict: `PASS_MLX_TORCH_SCORER_PARITY_SWEEP`
- Windows: 300/300 passed
- Covered pair window: `[0, 300]`
- SegNet argmax mismatch pixels total: 0
- SegNet component-diff samples max: 0
- SegNet logit abs max: `0.00011815875768661499`
- PoseNet output abs max: `0.00000762939453125`
- PoseNet component abs max: `9.705782184898926e-12`

## Negative Anchor: Full-Cache Batch-4 CPU Sweep

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/tmp/pact_head_mlx_verify_20260522T020141Z/src \
/Users/adpena/Projects/pact/.venv/bin/python \
/tmp/pact_head_mlx_verify_20260522T020141Z/tools/audit_mlx_scorer_torch_parity_sweep.py \
  --cache-dir /Users/adpena/Projects/pact/experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --repo-root /tmp/pact_head_mlx_verify_20260522T020141Z \
  --device cpu \
  --start-pair 0 \
  --max-pairs 300 \
  --window-pairs 4 \
  --stride-pairs 4 \
  --progress-every 10 \
  --run-id fec6_pr101_full300pairs_clean_head_52093e425_cpu_20260522T020141Z \
  --output /Users/adpena/Projects/pact/experiments/results/mlx_torch_parity_sweep_clean_head_52093e425_cpu_fec6_pr101_full300pairs_20260522T020141Z.json
```

Result:

- Verdict: `FAIL_MLX_TORCH_SCORER_PARITY_SWEEP`
- Windows: 74/75 passed
- Failed window: index 52, pair window `[208, 212]`
- Failure: `segnet_argmax_diff_pixels_exceeds_threshold:1>0`
- Mismatch: one pixel at sample 0, `(y=196, x=170)`, PyTorch class 1 vs MLX class 0
- PyTorch top-2 margin at mismatch: `0.00000667572021484375`
- MLX top-2 margin at mismatch: `0.0`
- Mismatch logit abs delta max: `0.00000762939453125`

Pair `208` as a singleton passed. Batch-invariance probe for `[208, 212]`
failed with one SegNet argmax mismatch; `[208, 210]` batch size 2 and
`[156, 160]` batch size 4 passed. The observed bug class is batch-shape
boundary sensitivity, not a decode/cache identity failure.

## Guardrail Landed

`src/tac/local_acceleration/mlx_production_contract.py` now binds multi-pair
response payloads to a passing batch-invariance manifest with the same MLX
device and same `batch_pairs`. This prevents a response produced with an
unsafe batch shape from being greenlit by an unrelated passing gate.

Unit coverage:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_mlx_production_contract.py
```

Result: 9 passed.

## Authority Contract

All artifacts here are local MLX implementation evidence only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- paired contest CPU/CUDA auth eval remains required for any score claim,
  promotion, or public ranking decision.

## Next Action

Use singleton CPU MLX scorer response as the fidelity-safe local signal for the
next training/search loop. Treat multi-pair CPU batching as an optimization
lane that needs a separate full-cache batch-invariance/parity proof before it
can feed production candidate selection.
