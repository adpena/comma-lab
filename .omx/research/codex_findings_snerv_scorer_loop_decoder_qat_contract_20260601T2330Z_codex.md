# Codex Findings: SNeRV Scorer-Loop Decoder/QAT Contract

UTC: 2026-06-01T23:30:00Z
Agent: codex:gpt-5
Axis: `[macOS-CPU advisory]`
Authority: false-authority; no score claim; no exact-eval readiness

## What Landed

Added a fail-closed implementation contract for the next SNeRV decoder-fit step:

- `src/tac/analysis/snerv_scorer_loop_decoder_qat_contract.py`
- `tools/build_snerv_scorer_loop_decoder_qat_contract.py`
- `src/tac/tests/test_snerv_scorer_loop_decoder_qat_contract.py`

The contract consumes the existing pose-guarded decoder gate and serializes the
next valid implementation target: scorer-loop decoder-weight training and QAT,
not another closed-form scalar/component HF weighting sweep. It preserves the
least-squares waterfill control as the acceptance baseline and keeps PoseNet as
a hard guard before any SegNet gain can advance.

## Artifact

Command:

```text
/Users/adpena/Projects/pact/.venv/bin/python tools/build_snerv_scorer_loop_decoder_qat_contract.py \
  .omx/research/snerv_pose_guarded_decoder_gate_20260601T2324Z.json \
  --dispatch-hold pr101_cpu_pending_blocks_exact_cuda_dispatch \
  --out .omx/research/snerv_scorer_loop_decoder_qat_contract_20260601T2330Z.json
```

Artifact:

- Path: `.omx/research/snerv_scorer_loop_decoder_qat_contract_20260601T2330Z.json`
- SHA-256: `bc66dc698d9e89823f5d60f28ecc0fb59d1eaacb1e5359795cac55afa2eaa067`
- Source pose-gate SHA-256: `25f577bce214d08499e56bf4d7a39d8117aa57a8a654b91a8dd15d2fc211e0c5`
- Lane: `lane_snerv_scorer_loop_decoder_qat_contract_20260601`

Key fields:

- `ready_for_scorer_loop_trainer_implementation=true`
- `ready_for_local_training_smoke=false`
- `ready_for_exact_eval_dispatch=false`
- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `closed_form_scalar_weighting_no_go=true`
- `dispatch_hold_reason=pr101_cpu_pending_blocks_exact_cuda_dispatch`

The local training smoke is deliberately still blocked because the actual
trainer CLI and receiver export proof do not exist yet. The contract is an
implementation handoff, not a fake training result.

## Accepted Implementation Modes

The contract permits two next-code modes:

1. `decoder_weight_linf_waterfill_qat`: train shared
   `HfGenerationDecoder.kernels` in the scorer loop with mixed precision
   waterfill. High-leverage decoder atoms may be protected at int8/fp16; low
   leverage atoms may be driven toward int4/int2/zero only after receiver replay
   and the pose guard pass.
2. `nonlinear_hf_decoder_qat`: replace scalar weighted least squares with a tiny
   receiver-portable nonlinear HF decoder trained with QAT in the scorer loop.

Both modes route the allocator into decoder weights, not post-hoc per-pair
latent tweaks.

## Verification

```text
/Users/adpena/Projects/pact/.venv/bin/ruff check \
  src/tac/analysis/snerv_scorer_loop_decoder_qat_contract.py \
  src/tac/tests/test_snerv_scorer_loop_decoder_qat_contract.py \
  tools/build_snerv_scorer_loop_decoder_qat_contract.py
All checks passed!

/Users/adpena/Projects/pact/.venv/bin/python -m pytest \
  src/tac/tests/test_snerv_scorer_loop_decoder_qat_contract.py \
  src/tac/tests/test_snerv_pose_guarded_decoder_gate.py \
  src/tac/tests/test_snerv_score_aware_decoder_fit_work_order.py \
  src/tac/tests/test_snerv_rate_adjudication.py \
  src/tac/tests/test_snerv_step_map_coder.py -q
33 passed in 1.00s
```

## Remaining Blockers

- `snerv_scorer_loop_decoder_qat_trainer_cli_missing`
- `segnet_posenet_in_loop_gradient_path_missing`
- `decoder_weight_qat_receiver_export_proof_missing`
- `full_600_pair_receiver_proof_missing`
- `paired_contest_cpu_cuda_pass_missing`
- `pr101_cpu_pending_blocks_exact_cuda_dispatch`

PR101 CPU recovery was re-polled before this landing and remains `pending`, so
no new full-video/exact/CUDA work was launched.

## Next Code Move

Implement `src/tac/substrates/snerv_inverse_steg_carrier/scorer_loop_decoder_qat.py`
and `tools/run_snerv_scorer_loop_decoder_qat_smoke.py` as a bounded local smoke.
The smoke must load real frames, put both SegNet and PoseNet into the objective,
export quantized decoder weights through the SNAR1 receiver contract, and remain
false-authority until it passes the pose-guard gate.
