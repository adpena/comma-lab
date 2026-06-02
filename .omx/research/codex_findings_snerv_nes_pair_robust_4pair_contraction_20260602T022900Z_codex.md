# Codex Findings - SNeRV NES Pair-Robust 4-Pair Contraction

UTC: 2026-06-02T02:29Z
Axis: [macOS-CPU advisory]
Authority: false-authority local continuation only

## Purpose

Follow up the 2-pair `nes_pair_robust` local win with a broader bounded
strided 4-pair confirmation under the same hard pair guard:

- no exact eval
- no full-video launch
- no CUDA launch
- receiver-replay local scorer loop only
- preserve per-pair component deltas

PR101 CPU remained pending during this run, so the no-new-exact/full-video/CUDA
hold stayed active.

## Command

```bash
/Users/adpena/Projects/pact/.venv/bin/python tools/run_snerv_scorer_loop_decoder_qat_smoke.py \
  --n-pairs 4 \
  --start-pair 8 \
  --pair-stride 13 \
  --upstream-dir /Users/adpena/Projects/pact/upstream \
  --video-path /Users/adpena/Projects/pact/upstream/videos/0.mkv \
  --search-mode nes_pair_robust \
  --max-trials 4 \
  --perturb-scale 0.02 \
  --pair-guard-min-score-improved-fraction 0.5 \
  --pair-guard-max-pose-worsened-fraction 0.0 \
  --out .omx/research/snerv_scorer_loop_decoder_qat_nes_pair_robust_4pair_strided_smoke_20260602T0225Z.json

/Users/adpena/Projects/pact/.venv/bin/python tools/build_snerv_pose_guarded_decoder_gate.py \
  .omx/research/snerv_scorer_loop_decoder_qat_nes_pair_robust_4pair_strided_smoke_20260602T0225Z.json \
  --out .omx/research/snerv_scorer_loop_decoder_qat_nes_pair_robust_4pair_strided_pose_gate_20260602T0228Z.json
```

## Artifacts

- `.omx/research/snerv_scorer_loop_decoder_qat_nes_pair_robust_4pair_strided_smoke_20260602T0225Z.json`
  - sha256: `9c15dafcd174de87390c0494f9a944347f6030bf41b25c1ccc71305a9ac805c0`
- `.omx/research/snerv_scorer_loop_decoder_qat_nes_pair_robust_4pair_strided_pose_gate_20260602T0228Z.json`
  - sha256: `2d6aceb0df23d98ff257ad8742a0edc090b12ff1abb85451d5039dfa5a8ab892`

## Result

- Baseline score: `1.280373806196691`
- Best accepted score: `1.280373806196691`
- Accepted improvement: `false`
- Scorer-loop evaluations: `10`
- Gate verdict: `NO_GO_FOR_PROMOTION_OR_EXACT_EVAL`
- Gate blocker: `no_candidate_passes_pose_guarded_local_continuation_gate`
- Gate next action: `implement_scorer_loop_or_nonlinear_qat_decoder_before_more_sweeps`

The synthesized `nes_pair_robust_update` had an aggregate-looking improvement:

- score: `1.280373806196691 -> 1.26656708484745`
- aggregate pose: `0.003283989923147601 -> 0.0028453843460738426`
- aggregate seg: `0.0029767354717478156 -> 0.002964019775390625`

But it failed the hard pair guard:

| pair | score delta | pose delta | seg delta |
| --- | ---: | ---: | ---: |
| 0 | `-0.01988589255187445` | `-0.00036687753163278103` | `-0.00007120775990188122` |
| 1 | `0.0023305726696605245` | `0.000014955076039768755` | `-0.000005086418241262436` |
| 2 | `-0.02417223627578391` | `-0.0015089018270373344` | `0.0000152587890625` |
| 3 | `0.005384436153012873` | `0.00010640197433531284` | `0.000010172603651881218` |

The violation is not aggregate PoseNet; it is pair-local PoseNet and score
damage on pairs 1 and 3. That makes the 2-pair win real but too narrow for
promotion or exact-eval routing.

## Verdict

NO-GO for promotion, exact eval, or full-video launch.

GO only for a stronger local optimizer design:

1. pair-minimax or all-pair objective, not average/fractional pair acceptance;
2. scorer-loop/full-main decoder training, not more random direction sweeps;
3. byte-accounted decoder grammar in parallel, because current fake quant still
   emits an fp32 receiver payload.

## Rate/Bit-Mask Reuse Implication

The rate side should now move independently of further simple fit sweeps. The
SNeRV archive currently serializes HF decoder kernels through
`encode_decoder_payload` as `snerv_decoder_payload.v1` with `dtype=float32_le`
and LZMA compression. The next byte work should adapt existing shared surfaces:

- `src/tac/substrates/_shared/decoder_state_codec.py` for int8/int4/int2/fp16
  mixed decoder-state grammar;
- `src/tac/substrates/_shared/int_stream_codec.py` for bitmask, fixed-width,
  zero-run, and delta/varint streams;
- `src/tac/packet_compiler/int_payload_bit_layouts.py` for bitplane/nibble
  final-rate transforms;
- `src/tac/substrates/_shared/mlx_score_aware/coder_qat.py` for coder-aware
  decoder-weight training pressure.

The right full-stack split is now clear: fit work needs stronger pair-robust
optimization; rate work should attack the receiver decoder payload grammar
without waiting for another local scorer sweep.

## Preserved Blockers

- `no_quantized_decoder_trial_improved_score_under_pose_guard`
- `local_smoke_only_not_full_600_pairs`
- `paired_contest_cpu_cuda_pass_missing`
- `mixed_precision_decoder_payload_grammar_not_byte_optimized`
- PR101 CPU recovery pending; no exact/full-video/CUDA launch authorized.
