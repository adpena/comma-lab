# Codex Findings: PR95 MLX Stage 6/7 Full Curriculum Spine

Generated: 2026-05-25T17:14:00Z
Agent: Codex
Axis: [macOS-MLX research-signal]

## Finding

The PR95 MLX local optimizer spine no longer skips the late C1a sweep stages.
Stage 6 (`stage6_lambda_sweep`) and Stage 7 (`stage7_sigma_sweep`) are now both
registered as executable local-MLX timing/proxy descriptors, and the
`full_pr95_source_video_runtime` control profile emits the complete recovered
PR95 stage sequence:

`[1, 2, 3, 4, 5, 6, 7, 8]`

This is a queue/planner integration result, not a score result. All generated
rows remain `score_claim=false`, `promotion_eligible=false`,
`rank_or_kill_eligible=false`, and `ready_for_exact_eval_dispatch=false`.

## Stage 6 Contract

- Descriptor: `pr95_stage6_adamw_lambda_sweep_mlx`
- Module: `stage6_lambda_sweep`
- Optimizer: AdamW, LR `3e-5`, latent LR multiplier `10.0`
- Loss family: `l7_softplus_seg_loss`
- C1a lambda: `0.02`
- C1a sigma: `0.2`
- QAT: `true`
- Muon: `false`
- Canonical epochs: `2000`

## Stage 7 Contract

- Descriptor: `pr95_stage7_adamw_sigma_sweep_mlx`
- Module: `stage7_sigma_sweep`
- Optimizer: AdamW, LR `3e-5`, latent LR multiplier `10.0`
- Loss family: `l7_softplus_seg_loss`
- C1a lambda: `0.02`
- C1a sigma: `0.1`
- QAT: `true`
- Muon: `false`
- Canonical epochs: `3000`

Stage 7 is intentionally a sigma sweep on top of Stage 6, not a new optimizer
family or score-authority row.

## Proof Artifacts

- Stage 7 smoke:
  `.omx/research/codex_pr95_stage7_sigma_sweep_probe_20260525T1710Z/stage7_sigma_sweep_3step_smoke.json`
- Full queue proof root:
  `.omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/`
- Queue summary:
  `.omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/build_summary.json`
- Queue manifest:
  `.omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/pr95_mlx_full_profile_manifest.json`
- Queue:
  `.omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/pr95_mlx_full_profile_queue.json`
- Queue validation:
  `.omx/research/codex_pr95_stage6_stage7_full_profile_queue_20260525T1714Z/queue_validation.json`

Proof summary:

```json
{
  "plan_count": 8,
  "stage_indices": [1, 2, 3, 4, 5, 6, 7, 8],
  "queue_experiments": 8,
  "valid": true,
  "score_claim": false,
  "promotion_eligible": false,
  "ready_for_exact_eval_dispatch": false,
  "all_false_ready": true
}
```

Stage 7 local smoke recorded `state_bytes=915944`, matching the prior Stage 6
architecture/state-size pattern. The stage distinction is carried in
training-config metadata (`stage_cat_sigma=0.1`) and in the optimizer scheduler
descriptor consumed by the queue.

## Integration

- `src/tac/local_acceleration/pr95_hnerv_mlx.py`
  - Added Stage 6 and Stage 7 module/descriptor dispatch entries.
- `src/tac/optimization/optimizer_scheduler_registry.py`
  - Added Stage 6 lambda-sweep and Stage 7 sigma-sweep descriptors with
    false-authority fields.
- `tools/build_pr95_mlx_optimizer_matrix_queue.py`
  - `full_pr95_source_video_runtime` now emits all eight PR95 stages.
- `.omx/state/lane_registry.json`
  - Registered L0 lanes for Stage 6 and Stage 7 curriculum builds.
- `.omx/state/probe_outcomes.jsonl`
  - Preserved Stage 6 probe row and added Stage 7 probe row.

## Verification

- `ruff` on touched PR95 MLX, optimizer registry, queue builder, and tests:
  passed.
- `pytest src/tac/tests/test_pr95_mlx_stage_2_v331_softplus_curriculum_build.py src/tac/tests/test_pr95_mlx_stage_3_v332_smooth_curriculum_build.py src/tac/tests/test_pr95_mlx_stage_4_v332_qat_curriculum_build.py src/tac/tests/test_pr95_mlx_stage_6_lambda_sweep_curriculum_build.py src/tac/tests/test_pr95_mlx_stage_7_sigma_sweep_curriculum_build.py src/tac/tests/test_pr95_mlx_optimizer_matrix_queue.py src/tac/tests/test_optimizer_scheduler_registry.py -q`
  - `74 passed in 11.01s`
- `tools/lane_maturity.py validate`
  - `1351 lane(s) validated cleanly`
- `tools/experiment_queue.py --queue ... validate`
  - `valid=true`, `experiment_count=8`

## Remaining Gap

This closes the local MLX curriculum-spine coverage gap. It does not close the
PR95 score-authority gap. Promotion still requires source-faithful scorer loss,
stage resume/QAT parity, PyTorch export parity on a source checkpoint,
runtime consumption proof, full-frame inflate parity, and paired exact
CPU/CUDA auth eval.
