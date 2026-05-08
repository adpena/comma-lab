# HNeRV Architecture-Shrink Generated-Schema Driver - 2026-05-07 Codex

## Evidence Boundary

- `score_claim=false`.
- `ready_for_exact_eval_dispatch=false`.
- Evidence grade: `empirical-generated-checkpoint-no-score`.
- This tranche produced generated schema/checkpoint/training-driver artifacts
  only. It did not produce a contest archive, runtime loader, or CUDA auth eval.

## Code Landed

- `src/tac/hnerv_arch_schema.py`
  - Generates the HNeRV state schema from `latent_dim`, `base_channels`,
    channel taper, eval size, and base grid.
  - Verifies the default generated schema exactly matches
    `tac.pr101_split_brotli_codec.FIXED_STATE_SCHEMA`.
  - Selects a smaller base width from an element-retention target.
  - Builds deterministic overlap-initialized shrunk state dicts by copying
    prefix slices and zero-filling missing/new channels.
- `tools/build_hnerv_arch_shrink_driver.py`
  - Emits `generated_schema.json`, `initial_state_dict.pt`, and
    `training_driver_manifest.json`.
  - Marks outputs fail-closed and non-promotable until runtime export,
    inflate parity, strict packet preflight, lane claim, and CUDA auth eval.
- `tools/plan_pr101_arch_shrink_retraining.py`
  - Now includes generated HNeRV base-channel/schema fields in each scenario.
  - Operator commands now route through the generated-schema driver before
    δεζ CPU sanity.

## Materialized PR101 Stage-D Artifact

Command:

```bash
.venv/bin/python tools/build_hnerv_arch_shrink_driver.py --source-state-dict experiments/results/cma_pr101_real_substrate_20260507T222605Z/pr101_decoder_state_dict.pt --element-retention 0.45 --scenario-name stage_d_zeta_width_precision_int4 --output-dir experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex --force
```

Outputs:

- Manifest:
  `experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex/training_driver_manifest.json`
- Generated schema:
  `experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex/generated_schema.json`
- Initial generated-schema checkpoint:
  `experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex/initial_state_dict.pt`

Target config:

- `base_channels=22`
- `channels=[22, 22, 22, 16, 12, 11, 11]`
- `n_state_elements=96,861`
- `schema_fingerprint=4c67e69d5cc4a2324e8429465deaccf13dc4077e490335d1b3756680d76ba220`
- `initial_state_dict_bytes=396,165`
- `initial_state_dict_sha256=631c0165811021a4fbd0b047eaf4971c9da7768d3fa1bd92866c4c047de53bbd`

## CPU Sanity

Command:

```bash
.venv/bin/python tools/run_deltaepszeta_training.py --state-dict experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex/initial_state_dict.pt --n-epochs 1 --steps-per-epoch 1 --log-dir experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex/deltaepszeta_cpu_sanity --run-label stage_d_generated_schema_sanity --seed 0
```

Result:

- `trained 1 steps`
- final lambda: `0.0005`
- JSONL log:
  `experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex/deltaepszeta_cpu_sanity/stage_d_generated_schema_sanity_step_log.jsonl`
- checkpoint:
  `experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex/deltaepszeta_cpu_sanity/final_state_dict.pt`

## Updated Planning Artifact

Command:

```bash
.venv/bin/python tools/plan_pr101_arch_shrink_retraining.py --state-dict-path experiments/results/cma_pr101_real_substrate_20260507T222605Z/pr101_decoder_state_dict.pt --entropy-floor-report reports/pr101_provable_optimal_floor.json --output-dir experiments/results/pr101_arch_shrink_retraining_plan_20260507_codex_generated_schema --skip-compact-encode
```

Best predicted row remains prediction-only:

- `stage_e_high_risk_int4_sparse`
- expected archive bytes: `63,011`
- delta vs 178,144 reference: `-115,133`
- rate score delta: `-0.076662`
- blockers include no trained checkpoint, no runtime decoder schema manifest,
  no inflate parity, no exact CUDA auth eval, unstructured mask overhead
  prediction only, and target-bit export not implemented.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_hnerv_arch_schema.py src/tac/tests/test_build_hnerv_arch_shrink_driver.py -q
```

Result: `6 passed`.

```bash
.venv/bin/python -m ruff check src/tac/hnerv_arch_schema.py tools/build_hnerv_arch_shrink_driver.py tools/plan_pr101_arch_shrink_retraining.py src/tac/tests/test_hnerv_arch_schema.py src/tac/tests/test_build_hnerv_arch_shrink_driver.py
```

Result: `All checks passed`.

## Remaining Blockers

- Runtime loader/export for generated HNeRV schemas is still not implemented.
- The generated checkpoint is overlap-initialized, not trained; it has no
  distortion evidence.
- Target int4/int6 codec export for generated schema is not wired.
- Local inflate parity and strict packet preflight are missing.
- Any GPU run still requires a lane claim before dispatch.
- Exact CUDA auth eval remains the only score truth.

## Next Tranche

1. Implement generated-schema runtime export/load for the HNeRV decoder packet.
2. Add a local inflate parity fixture for a tiny generated schema.
3. Teach the CodecOp pipeline to accept generated schemas instead of only the
   hardcoded PR101 `FIXED_STATE_SCHEMA`.
4. Convert the best generated-schema trained checkpoint into a packet candidate,
   then run strict packet preflight before any lane claim or GPU dispatch.
