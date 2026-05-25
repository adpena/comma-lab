# Codex Findings: PR95 MLX Full Queue Execution

Generated: 2026-05-25T17:30:24Z
Lane: `lane_pr95_hnerv_mlx_reproduction`
Queue: `.omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/pr95_mlx_full_profile_queue.json`
State: `.omx/state/experiment_queue_codex_pr95_stage6_stage7_full_profile_20260525T1714Z.sqlite`

## Verdict

The 8-stage PR95/HNeRV MLX full-profile queue executed locally through `experiment_queue.v1`.
All 8 local MLX steps succeeded, with zero orphaned steps and zero postcondition
failures. This is still `[macOS-MLX research-signal]`; all generated artifacts
preserve `score_claim=false`, `promotion_eligible=false`, and
`ready_for_exact_eval_dispatch=false`.

## Execution Facts

Command:

```bash
.venv/bin/python tools/experiment_queue.py --queue .omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/pr95_mlx_full_profile_queue.json run-worker --execute --max-parallel 1 --max-steps 8
```

Queue status after execution:

- `status_counts={"succeeded": 8}`
- `ready_steps=[]`
- `orphaned_step_count=0`
- artifact root size: `59M`
- queue sqlite size: `192K`
- queue logs size: `256K`

Per-stage execution summary:

| Stage | Module | Optimizer | Muon | sec/step | state bytes | archive bytes | runtime proof | PyTorch parity |
| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | `stage1_v328_ce` | `pr95_stage1_adamw_baseline_mlx` | no | 0.034119 | 915944 | 230341 | yes | yes |
| 2 | `stage2_v331_softplus` | `pr95_stage2_adamw_baseline_mlx` | no | 0.034996 | 915944 | 230341 | yes | yes |
| 3 | `stage3_v332_smooth` | `pr95_stage3_adamw_baseline_mlx` | no | 0.036953 | 915944 | 230350 | yes | yes |
| 4 | `stage4_v332_qat` | `pr95_stage4_adamw_qat_mlx` | no | 0.036310 | 915944 | 230350 | yes | yes |
| 5 | `stage5_c1a_l7` | `pr95_stage5_adamw_baseline_mlx` | no | 0.035658 | 915944 | 230339 | yes | yes |
| 6 | `stage6_lambda_sweep` | `pr95_stage6_adamw_lambda_sweep_mlx` | no | 0.035446 | 915944 | 230339 | yes | yes |
| 7 | `stage7_sigma_sweep` | `pr95_stage7_adamw_sigma_sweep_mlx` | no | 0.037409 | 915944 | 230339 | yes | yes |
| 8 | `stage8_muon_finetune` | `pr95_stage8_muon_adamw_mlx` | yes | 0.037466 | 915944 | 230345 | yes | yes |

Common proof facts:

- source-video target: `upstream/videos/0.mkv`, pair index `0`, output size `384x512`
- training surface: `rgb_yuv6_mse`, `steps=1`, `batch_size=1`
- runtime raw output per stage: `6104016` bytes
- PyTorch state-dict export per stage: `924847` bytes
- MLX vs PyTorch forward parity max abs diff: `3.0517578125e-05` across all stages

## Integration Notes

The run closes the first end-to-end PR95 MLX profile proof across Stage 1-8:
source-video preprocess, local MLX train step, PR95 public archive export,
runtime consumption proof, and PyTorch export parity all execute under the queue.

Remaining blockers are intentional and still fail closed:

- this is not contest CPU/CUDA auth eval;
- this is not full-frame inflate parity against an original PR95 runtime packet;
- this is not a trained PR95 reproduction, only a 1-step local MLX timing/proof smoke;
- binary packets/state-dicts/raw runtime scratch remain local custody and are
  ignored by `.gitignore`; durable signal is the compact JSON proofs plus this memo.

## Next Actions

1. Promote the same queue shape from 1-step smoke to measured Stage 1/5/8 timing
   ladders on real source-video pairs.
2. Add byte-closed archive export from a real trained MLX checkpoint, then compare
   full-frame inflate output against the canonical PR95 PyTorch runtime.
3. Keep Stage 8/Muon as a real PR95 control arm, not a placeholder: measure the
   optimizer overhead and convergence against AdamW on source-video pairs.
4. Feed queue timings into the local substrate cost model so PR95-class HNeRV,
   HNeRV boltons, NeRV-family, and non-NeRV campaigns can choose local MLX vs
   cloud spend from measured wall-clock, not stale estimates.
