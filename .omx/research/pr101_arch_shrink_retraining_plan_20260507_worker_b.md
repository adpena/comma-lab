# PR101 Architecture-Shrink Retraining Plan - 2026-05-07 Worker B

## Scope

Worker B inspected the current PR101/PR106 HNeRV training and codec surfaces and
converted the structural-retraining idea into an executable CPU planning tool.
No archive was produced, no GPU work was dispatched, and no contest score is
claimed.

`score_claim=false`

## Artifacts

- `tools/plan_pr101_arch_shrink_retraining.py`
- `src/tac/tests/test_plan_pr101_arch_shrink_retraining.py`
- `experiments/results/pr101_arch_shrink_retraining_plan_20260507_worker_b/plan.json`
- `experiments/results/pr101_arch_shrink_retraining_plan_20260507_worker_b/plan.md`

## Code Inspection Summary

- Public PR101 and PR106 use the same HNeRV decoder shape: `latent_dim=28`,
  `base_channels=36`, six pixel-shuffle blocks, and a hardcoded 28-tensor
  decoder schema.
- PR106 training is an eight-stage from-scratch GPU curriculum in
  `experiments/results/public_pr106_belt_and_suspenders_intake_20260504_codex/source/submissions/belt_and_suspenders/src/train.py`
  and `src/stages/common.py`; `train_stage` instantiates
  `HNeRVDecoder(latent_dim=28, base_channels=36, eval_size=EVAL_SIZE)`
  directly.
- PR106 already contains QAT, Muon fine-tuning, and categorical entropy terms,
  but not architecture shrink or a generated schema/runtime contract.
- `tools/run_deltaepszeta_training.py` is useful as a CPU sanity loop over a
  state dict and H0 rate proxy, but it does not run the HNeRV forward path,
  scorer-aware training, architecture shrink, or archive/inflate parity.
- `src/tac/self_compressing_nn.py` has the right width x precision accounting
  primitives for a future self-compress HNeRV driver.
- `src/tac/codec_pipeline_deltaepszeta_callback.py` can log codec bytes per
  epoch once the generated-schema codec pipeline exists.
- `src/tac/pr101_split_brotli_codec.py::FIXED_STATE_SCHEMA` is hardcoded;
  any shrunk architecture needs a generated schema and matching runtime loader
  before it can produce a contest-compliant archive.

## Evidence Boundary

- Empirical: PR101 state-dict tensor count, quantized symbol count, per-tensor
  H0, compact PR101 decoder bytes.
- Derivation: Shannon H0 payload floor, codec efficiency ratio, and contest
  rate-score delta formula.
- Prediction: element-retention, sparsity, quantization, entropy-ratio,
  side-info, and mask-overhead scenario rows.

The generated plan records:

- State dict:
  `experiments/results/cma_pr101_real_substrate_20260507T222605Z/pr101_decoder_state_dict.pt`
- State dict SHA-256:
  `b863362aaba1b9cae9b944f5e5b1a43a53ca824b7899ed7b80a2e2146d66f053`
- Compact PR101 decoder payload: `162,164` bytes.
- IID per-tensor floor payload: `159,822` bytes.
- Codec efficiency vs IID floor: `1.0147`.
- Markov-1 oracle payload: `152,106` bytes.
- Markov-2 oracle payload: `98,013` bytes.

## Scenario Ranking

| rank | scenario | evidence | expected archive | delta bytes vs reference | rate-score delta |
|---:|---|---|---:|---:|---:|
| 1 | `stage_e_high_risk_int4_sparse` | prediction | 63,037 | -115,107 | -0.076645 |
| 2 | `stage_d_zeta_width_precision_int4` | prediction | 67,235 | -110,909 | -0.073850 |
| 3 | `stage_c_width075_int6_qat_dez` | prediction | 102,994 | -75,150 | -0.050039 |
| 4 | `stage_b_width080_int8_dez` | prediction | 116,241 | -61,903 | -0.041219 |
| 5 | `stage_a_width090_int8_rate_shape` | prediction | 145,077 | -33,067 | -0.022018 |
| 6 | `control_current_pr101_int8` | empirical | 178,258 | +114 | +0.000076 |

The high-risk rows are routing estimates only. They do not include distortion
survival, exact runtime closure, or CUDA auth eval. The best practical first
GPU design target is likely the conservative-to-moderate structured path
(`stage_a` then `stage_b`) because it preserves the HNeRV topology class while
forcing the generated-schema work to become real.

## Driver Design Contract

A dispatchable HNeRV architecture-shrink driver must write a manifest with:

- `score_claim=false` until exact CUDA auth eval lands.
- Source archive/state dict SHA-256s.
- Generated architecture schema: latent dim, base channels, channel schedule,
  tensor names, shapes, quantization bits, sparsity/width masks, and runtime
  loader version.
- Stage configs, seeds, optimizer states, EMA decay, and best epoch.
- Deltaepszeta target source and per-tensor rate weights.
- QAT target bits and fake-quant export parameters.
- Self-compress width/precision config if ζ is enabled.
- Codec pipeline manifest and final decoder payload SHA-256.
- Inflate parity or explicit distortion-delta classification before dispatch.
- Strict compliance report and terminal exact-eval artifact after dispatch.

## GPU Dispatch Blockers

- Parameterize `HNeRVDecoder` and `train_stage` so shrunk
  `base_channels`/`latent_dim` emit a generated state schema.
- Implement a target codec/runtime loader for the generated schema.
- Run CPU roundtrip through the target codec pipeline and local inflate parity.
- Claim the lane via `tools/claim_lane_dispatch.py` before GPU training/eval.
- Run exact CUDA auth eval on the produced archive bytes before setting any
  score claim true.

## Verification

- `uv run ruff check tools/plan_pr101_arch_shrink_retraining.py src/tac/tests/test_plan_pr101_arch_shrink_retraining.py`
- `uv run --with pytest python -m pytest src/tac/tests/test_plan_pr101_arch_shrink_retraining.py -q`

Focused result: 4 tests passed, 1 pytest config warning for unknown `timeout`.
