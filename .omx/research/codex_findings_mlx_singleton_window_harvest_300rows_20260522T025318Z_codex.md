# Codex Findings: MLX Singleton Full-Video Window Harvest

UTC: 2026-05-22T02:53:18Z

## Verdict

PROCEED_WITH_GUARDS. The committed singleton response-window splitter now produced a full 300-window local MLX scorer-response table for the FEC6 PR101 cache. This is still non-authoritative local research signal only; every payload and dataset surface keeps `score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

## Artifact Root

`experiments/results/mlx_singleton_window_harvest_fec6_20260522T0250Z_full300/`

The artifact root is ignored experiment output; this memo is the durable tracked summary.

## Parent Responses

Candidate parent:

- Path: `candidate_parent_0000_0300.json`
- Pair window: `[0, 300]`
- Batch pairs: `1`
- Samples: `300`
- Local MLX canonical score: `0.1922605584791145`
- Avg PoseNet distortion: `0.000039679991221153914`
- Avg SegNet distortion: `0.0005347357859136537`
- Elapsed: `274.61687564849854` seconds

Reference-vs-reference baseline parent:

- Path: `baseline_parent_0000_0300.json`
- Pair window: `[0, 300]`
- Batch pairs: `1`
- Samples: `300`
- Local MLX canonical score: `0.11886714273451066`
- Avg PoseNet distortion: `0.0`
- Avg SegNet distortion: `0.0`
- Elapsed: `145.6073830127716` seconds

## Window Dataset

Dataset path:

`windowed_scorer_response_dataset_300rows.json`

Summary:

- Rows: `300`
- Skipped rows: `0`
- Baseline mode: `per_window_mlx_response`
- Baseline windows: `300`
- Family counts: `{"mlx_scorer_response": 300}`
- Best same-window delta vs reference baseline: `0.04240827263793981` at pair window `[218, 219]`.
- Worst same-window delta vs reference baseline: `0.11924633770300272` at pair window `[107, 108]`.
- Improved scorer term count: `0`
- Improved total score count: `0`

The positive deltas are expected for FEC6-vs-reference windows because the baseline is reference-vs-reference with zero scorer distortion at the same archive-size rate term.

## LL Planner

Plan path:

`ll_next_probe_plan_windowed_300rows.json`

Priority-1 probe remains:

`ll_mlx_cpu_stable_response_harvest`

The attached clean singleton parity gate is still `strict_pass`:

- Source verdict: `PASS_MLX_TORCH_SCORER_PARITY_SWEEP`
- Covered pair window: `[0, 300]`
- Passed windows: `300`
- Failed windows: `0`
- SegNet argmax mismatch pixels total: `0`
- PoseNet output abs max: `7.62939453125e-06`
- SegNet logit abs max: `0.00011815875768661499`

Top planner input rows:

- `[218, 219]`
- `[243, 244]`
- `[240, 241]`
- `[20, 21]`
- `[258, 259]`
- `[24, 25]`
- `[247, 248]`
- `[239, 240]`

The planner still prohibits widening coordinate sparse residual sidecars because observed scorer gains do not pay current payload bytes.

## Interpretation

The 50-row minimum has been exceeded by a full-video singleton table, so the immediate coverage blocker for a first local LL response corpus is cleared for FEC6. Remaining blockers are higher-level: held-out correlation, at least one additional response family, and strict separation between MLX research signal and CUDA/Linux auth-eval authority.

## Next Action

Use the 300-row table as the stable FEC6 baseline response corpus, then add at least one byte-neutral or amortized candidate family so the LL planner is not fitting only absolute FEC6-vs-reference distortion. Do not use the MLX table for rank/kill, promotion, or spend selection until held-out correlation against exact-eval-compatible observations is recorded.
