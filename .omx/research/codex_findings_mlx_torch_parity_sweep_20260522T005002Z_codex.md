# Codex Findings: MLX Torch Parity Sweep

utc: 2026-05-22T00:50:02Z
lane_id: lane_mlx_auth_scorer_training_signal_fidelity_20260521
evidence_axis: [macOS-MLX research-signal]
verdict: FAIL_CLOSED_ON_FULL300_STRICT_SEGNET_ARGMAX_PARITY

## What changed

- Added a sweep-level PyTorch-vs-MLX scorer parity manifest:
  - `tac.local_acceleration.mlx_scorer_torch_parity.build_mlx_scorer_torch_parity_sweep_manifest`
  - `tools/audit_mlx_scorer_torch_parity_sweep.py`
- The sweep loads the upstream PyTorch `DistortionNet` and the MLX adapter once, then evaluates fixed cache windows.
- The manifest records per-window deltas plus max/mean/p50/p95/p99 summaries for:
  - PoseNet output max abs delta
  - PoseNet component abs delta
  - SegNet logit max abs delta
  - SegNet argmax-diff pixels
  - SegNet argmax-diff fraction
  - SegNet component-diff samples
- Added JSON progress events so full-cache sweeps do not run silently.
- Hardened `build_mlx_scorer_response_payload`: if reference and candidate scorer-input batches are byte-identical, the candidate scorer output reuses the reference output. That is mathematically exact and prevents a false nonzero MLX response from a redundant second forward pass.

## Full300 FEC6 sweep anchor

Command:

```bash
.venv/bin/python tools/audit_mlx_scorer_torch_parity_sweep.py \
  --cache-dir experiments/results/mlx_scorer_input_cache_fec6_pr101_20260521T210100Z_full300pairs \
  --output experiments/results/mlx_scorer_torch_parity_sweep_fec6_pr101_full300_20260522T004129Z_w4_s4_v2.json \
  --repo-root . \
  --device cpu \
  --start-pair 0 \
  --max-pairs 300 \
  --window-pairs 4 \
  --stride-pairs 4 \
  --progress-every 15 \
  --run-id fec6_pr101_full300_w4_s4_v2
```

Result:

- verdict: `FAIL_MLX_TORCH_SCORER_PARITY_SWEEP`
- window_count: `75`
- passed_windows: `74`
- failed_windows: `1`
- failed window: index `39`, pair_window `[156, 160]`
- failure: `segnet_argmax_diff_pixels_exceeds_threshold:1>0`
- failed-window SegNet argmax diff fraction: `1.2715657552083333e-06`
- failed-window SegNet argmax pixel count: `786432`
- PoseNet output max abs delta max: `7.62939453125e-06`
- PoseNet component abs max: `9.713470479344455e-12`
- SegNet logit max abs delta: `0.0007913112640380859`

Interpretation: MLX is very close to upstream PyTorch on this FEC6 full300 cache, but not 1:1 under strict SegNet argmax parity. Keep it as a high-signal local training/ranking substrate, not an auth-eval scorer replacement.

## Verification

```bash
.venv/bin/python -m ruff check src/tac/local_acceleration/mlx_scorer_response.py src/tac/local_acceleration/mlx_scorer_torch_parity.py tools/audit_mlx_scorer_torch_parity_sweep.py src/tac/tests/test_mlx_scorer_torch_parity.py
.venv/bin/python -m pytest -q src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_mlx_scorer_torch_parity.py
.venv/bin/python -m pytest -q src/tac/tests/test_scorer_response_dataset.py src/tac/tests/test_plan_ll_scorer_response_next_cli.py src/tac/tests/test_mlx_execution_plan.py src/tac/tests/test_mlx_profile_stability.py src/tac/tests/test_profile_mlx_scorer_response_cache.py src/tac/tests/test_mlx_scorer_response.py src/tac/tests/test_mlx_batch_invariance.py src/tac/tests/test_mlx_scorer_fidelity.py src/tac/tests/test_mlx_scorer_torch_parity.py src/tac/tests/test_mlx_production_contract.py src/tac/tests/test_mlx_scorer_adapters.py src/tac/tests/test_mlx_preprocess.py src/tac/tests/test_mlx_cache_audit.py src/tac/tests/test_mlx_scorer_state_map.py src/tac/tests/test_mlx_scorer_port_inventory.py src/tac/tests/test_mlx_to_pytorch_export.py
```

Results:

- Ruff: passed.
- Focused response/parity tests: 16 passed.
- Broad MLX suite: 146 passed.

## Next action

Wire the LL scorer-response planner to require either:

1. a passing strict parity sweep for the exact cache window it consumes, or
2. an explicit research-signal override that records the SegNet argmax mismatch distribution and keeps promotion/exact-eval authority false.
