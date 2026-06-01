# Codex Findings: HiNeRV/SNeRV Execute-Family Planner Gate

UTC: 2026-06-01T21:15:25Z
Agent: Codex
Axis: compact carrier queue execution / MLX-first campaign automation
Status: LANDED as fail-closed execution gate, no score claim

## Verdict

`tools/run_compact_renderer_mlx_spine_runner.py --execute-family hi_nerv` and
`--execute-family snerv` are now accepted by the CLI and consume the
score-aware carrier planner before launch. They do not fake training.

Until native MLX train/export/archive adapters exist, the command writes a
normal `compact_renderer_mlx_spine_runner_report.json` with:

- `trainer_launch_allowed=false`;
- the nested `score_aware_carrier_training_plan`;
- the requested campaign scale (`num_pairs`, `epochs`, byte ceilings);
- the adapter contract required to clear launch;
- explicit missing-adapter/export/receiver/replay/exact-auth blockers;
- canonical false-authority fields.

## Why This Matters

Before this landing, top-priority carriers were present in plan rows but could
not be addressed through the same execute-family operator surface as executable
families. That encouraged manual interpretation and made it too easy to lose
the corrected HiNeRV evidence. Now the command path itself preserves the signal:
cheap-but-unfit HiNeRV routes to score-aware decoder-weight training, while
SNeRV stays blocked until a real MLX adapter and scorer/QAT/export stack exist.

## Next Build

The next non-fake implementation step is to add native adapters:

1. `hi_nerv` MLX renderer bundle with real pair decode.
2. `snerv` MLX spectra/wavelet carrier bundle.
3. byte-closed archive exporters and NumPy inflate runtimes.
4. receiver proof, full-video MLX prefilter, local CPU replay, then CPU/CUDA
   exact auth only for local winners.

No large artifact was produced by this landing.

