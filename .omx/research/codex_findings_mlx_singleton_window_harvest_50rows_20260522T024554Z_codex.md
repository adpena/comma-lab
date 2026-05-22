# Codex Findings: MLX Singleton Window Harvest 50 Rows

UTC: 2026-05-22T02:45:54Z

## Verdict

PROCEED_WITH_GUARDS. The singleton CPU MLX scorer-response path now has a reusable window splitter and a 50-row local response dataset for LL planning. The dataset is non-authoritative local signal only: `score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`, and `ready_for_exact_eval_dispatch=false` throughout.

## Code Changes

- Added `tac.local_acceleration.mlx_response_windows` to split one singleton `mlx_scorer_response.v1` parent payload into per-window child payloads while recomputing score from sliced PoseNet/SegNet component arrays.
- Added `tools/split_mlx_scorer_response_windows.py` as the operator CLI.
- Fixed windowed MLX dataset validation so original-video/reference baselines may use scorer-input `array_sha256` identity, while actual candidate rows still require archive/raw custody hashes.

## Empirical Harvest

Artifact root:

`experiments/results/mlx_singleton_window_harvest_fec6_20260522T0240Z/`

Parent singleton CPU MLX responses:

- Candidate FEC6 pairs `[0, 50]`: `canonical_score=0.18374145784727336`, `avg_posenet_dist=1.228106591611322e-05`, `avg_segnet_dist=0.0005379231803817674`, `n_samples=50`.
- Reference-vs-reference baseline pairs `[0, 50]`: `canonical_score=0.11886714273451066`, `avg_posenet_dist=0.0`, `avg_segnet_dist=0.0`, `n_samples=50`.

Window split:

- Candidate windows: `50`
- Baseline windows: `50`
- Window size: `1` scorer pair
- Batch pairs: `1`

Windowed dataset:

- Path: `experiments/results/mlx_singleton_window_harvest_fec6_20260522T0240Z/windowed_scorer_response_dataset_50rows.json`
- Rows: `50`
- Skipped rows: `0`
- Family counts: `{"mlx_scorer_response": 50}`
- Best same-window delta vs reference baseline: `0.046142192597503434` at candidate pair window `[20, 21]`.
- Worst same-window delta vs reference baseline: `0.07835571434269023` at candidate pair window `[1, 2]`.

These deltas are expected to be positive because the reference baseline has zero scorer distortion and the candidate response adds FEC6 reconstruction distortion at the same rate term.

## LL Planner Result

Plan path:

`experiments/results/mlx_singleton_window_harvest_fec6_20260522T0240Z/ll_next_probe_plan_windowed_50rows.json`

Priority-1 probe:

`ll_mlx_cpu_stable_response_harvest`

The attached MLX/Torch parity gate is `strict_pass` from the clean singleton parity sweep:

- Source verdict: `PASS_MLX_TORCH_SCORER_PARITY_SWEEP`
- Passed windows: `300`
- Failed windows: `0`
- SegNet argmax mismatch pixels total: `0`
- PoseNet output abs max: `7.62939453125e-06`
- SegNet logit abs max: `0.00011815875768661499`

The 50-row minimum is now satisfied for a first local LL dataset, but this is still one family only and not enough for any score authority, rank/kill decision, or exact-eval spend filter without held-out correlation and broader family coverage.

## Verification Commands

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_response_windows.py \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_mlx_scorer_response.py
```

Result: `48 passed`.

```bash
git diff --check
```

Result: pass.

## Next Action

Use the splitter for broader singleton-safe harvests across additional windows and candidate families, then run held-out LL correlation before the MLX path is allowed to influence paid exact-eval selection. Keep all MLX outputs non-authoritative until byte-closed CUDA/Linux auth eval validates transfer.
