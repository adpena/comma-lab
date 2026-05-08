# PR101 Architecture-Shrink Retraining Plan - 2026-05-07 Worker B

## Evidence Boundary

- `score_claim=false`; no archive was produced or evaluated.
- Empirical: PR101 state-dict symbol counts, H0, compact decoder bytes.
- Derivation: Shannon H0 floor and contest rate-score arithmetic.
- Prediction: shrink / sparsity / quantization / entropy-ratio rows.

## Baseline

- State dict: `experiments/results/cma_pr101_real_substrate_20260507T222605Z/pr101_decoder_state_dict.pt`
- SHA-256: `b863362aaba1b9cae9b944f5e5b1a43a53ca824b7899ed7b80a2e2146d66f053`
- Tensors: 28
- Quantized symbols: 228,958
- Compact PR101 decoder payload: 162,050 B
- IID per-tensor floor payload: 159,822 B
- Codec efficiency vs IID floor: 1.0139
- Markov-1 oracle payload: 152,106 B
- Markov-2 oracle payload: 98,013 B

## Scenario Ranking

| rank | scenario | evidence | retention | sparsity | bits | entropy ratio | expected archive | delta bytes | rate delta | dispatch |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `stage_e_high_risk_int4_sparse` | prediction | 0.42 | 0.25 | 4.0 | 0.86 | 63,011 | -115,133 | -0.076662 | false |
| 2 | `stage_d_zeta_width_precision_int4` | prediction | 0.45 | 0.10 | 4.0 | 0.90 | 67,202 | -110,942 | -0.073872 | false |
| 3 | `stage_c_width075_int6_qat_dez` | prediction | 0.56 | 0.00 | 6.0 | 0.94 | 102,934 | -75,210 | -0.050079 | false |
| 4 | `stage_b_width080_int8_dez` | prediction | 0.64 | 0.00 | 8.0 | 0.96 | 116,171 | -61,973 | -0.041265 | false |
| 5 | `stage_a_width090_int8_rate_shape` | prediction | 0.81 | 0.00 | 8.0 | 0.98 | 144,986 | -33,158 | -0.022079 | false |
| 6 | `control_current_pr101_int8` | empirical | 1.00 | 0.00 | 8.0 | 1.00 | 178,144 | +0 | +0.000000 | false |

## Best Predicted Row

`stage_e_high_risk_int4_sparse` is the smallest rate-side estimate, but it is still blocked from dispatch because no trained checkpoint, generated schema, runtime loader, or exact CUDA eval exists for that architecture.

## Integration Points Found

- Public PR101/PR106 `HNeRVDecoder` fixes `latent_dim=28`, `base_channels=36`, and a 28-tensor schema.
- `src/tac/pr101_split_brotli_codec.py::FIXED_STATE_SCHEMA` is hardcoded to the current architecture.
- `tools/run_deltaepszeta_training.py` is a state-dict CPU sanity loop, not a renderer/scorer training driver.
- `src/tac/self_compressing_nn.py` has width x precision accounting that can inform the loss once the HNeRV driver exists.
- `src/tac/codec_pipeline_deltaepszeta_callback.py` can log codec bytes per epoch after a pipeline for the generated schema exists.

## Operator Commands

```bash
.venv/bin/python tools/plan_pr101_arch_shrink_retraining.py --state-dict-path experiments/results/cma_pr101_real_substrate_20260507T222605Z/pr101_decoder_state_dict.pt --entropy-floor-report reports/pr101_provable_optimal_floor.json --output-dir experiments/results/pr101_arch_shrink_retraining_plan_20260507_worker_b
```

```bash
.venv/bin/python tools/build_hnerv_arch_shrink_driver.py --source-state-dict experiments/results/cma_pr101_real_substrate_20260507T222605Z/pr101_decoder_state_dict.pt --element-retention 0.45 --scenario-name stage_d_zeta_width_precision_int4 --output-dir experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex
```

```bash
.venv/bin/python tools/run_deltaepszeta_training.py --state-dict experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex/initial_state_dict.pt --n-epochs 1 --steps-per-epoch 2 --log-dir experiments/results/hnerv_arch_shrink_training_driver_stage_d_20260507_codex/deltaepszeta_cpu_sanity --run-label stage_d_generated_schema_sanity
```

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id pr101_arch_shrink_deltaepszeta_gpu --status planned --notes 'blocked until generated HNeRV schema/runtime and CPU roundtrip land'
```

## GPU Dispatch Blockers

- Parameterize HNeRVDecoder and train_stage so shrunk base_channels/latent_dim produce a generated state schema.
- Implement a target codec/runtime loader for the generated schema; current PR101 FIXED_STATE_SCHEMA is hardcoded.
- Run CPU roundtrip through codec pipeline and local inflate parity before any remote eval.
- Claim lane via tools/claim_lane_dispatch.py before GPU training/eval dispatch.
- Run exact CUDA auth eval on produced archive bytes before setting score_claim=true.
