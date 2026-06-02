# Codex Findings - SNeRV NES Pair-Robust Scorer Loop And Reuse Map

UTC: 2026-06-02T02:20Z
Axis: [macOS-CPU advisory]
Authority: false-authority local continuation only

## What Changed

Implemented `nes_pair_robust` in the bounded SNeRV scorer-loop decoder/QAT
smoke. This mode evaluates symmetric decoder-weight probes, ranks them with a
pair-robust local objective, synthesizes one normalized update, and only accepts
the update through the existing receiver-replay, SegNet, PoseNet, score, and
pair-local guards.

The goal is to retire simple coordinate/subspace luck as the next promotion
path and provide a bounded nonlinear/scorer-loop continuation surface before
any larger receiver-proof or exact-eval work.

## Command

```bash
/Users/adpena/Projects/pact/.venv/bin/python tools/run_snerv_scorer_loop_decoder_qat_smoke.py \
  --n-pairs 2 \
  --start-pair 16 \
  --pair-stride 8 \
  --upstream-dir /Users/adpena/Projects/pact/upstream \
  --video-path /Users/adpena/Projects/pact/upstream/videos/0.mkv \
  --search-mode nes_pair_robust \
  --max-trials 4 \
  --perturb-scale 0.02 \
  --pair-guard-min-score-improved-fraction 0.5 \
  --pair-guard-max-pose-worsened-fraction 0.0 \
  --out .omx/research/snerv_scorer_loop_decoder_qat_nes_pair_robust_2pair_strided_smoke_20260602T0218Z.json

/Users/adpena/Projects/pact/.venv/bin/python tools/build_snerv_pose_guarded_decoder_gate.py \
  .omx/research/snerv_scorer_loop_decoder_qat_nes_pair_robust_2pair_strided_smoke_20260602T0218Z.json \
  --out .omx/research/snerv_scorer_loop_decoder_qat_nes_pair_robust_2pair_strided_pose_gate_20260602T0219Z.json
```

## Artifacts

- `.omx/research/snerv_scorer_loop_decoder_qat_nes_pair_robust_2pair_strided_smoke_20260602T0218Z.json`
  - sha256: `51bd7d8c65891061d6773594cb51e9af4ae9f77b604817dd581570926eeac1af`
- `.omx/research/snerv_scorer_loop_decoder_qat_nes_pair_robust_2pair_strided_pose_gate_20260602T0219Z.json`
  - sha256: `97f16ea4f2885da171ff756fb77066e588c82615ab1daa5878747e530a8fcaaf`

## Result

- Baseline score: `1.0234573726510832`
- Best score: `1.0044682764928354`
- Score delta: `-0.01898909615824773`
- Pose delta: `-0.0011327750980854034`
- Seg delta: `-0.000030517578125`
- Scorer-loop evaluations: `10`
- Best row: `nes_pair_robust_update`
- Gate verdict: `GO_LOCAL_CONTINUATION_ONLY`
- Gate next action: `run_bounded_non_exact_receiver_replay_on_best_local_candidate`

Per-pair deltas against the least-squares baseline:

| pair | score delta | pose delta | seg delta |
| --- | ---: | ---: | ---: |
| 0 | `-0.010077014791883498` | `-0.0003797132521867752` | `-0.000030517578125` |
| 1 | `-0.02530425960817939` | `-0.0018858369439840317` | `-0.000030517578125` |

## Verdict

GO for local continuation only.

This is a meaningful recovery from the coordinate/simple-subspace negative
sequence because the accepted row is nonlinear, pair-robust on this slice, and
PoseNet-improving on both pairs. It is not promotion evidence. It remains
blocked by full-600 receiver proof, paired contest CPU/CUDA eval, and byte
optimized decoder grammar.

## Reuse Map For Bit Mask, Final Rate Attack, And Shared Optimizers

The codebase already has reusable machinery that should be adapted before any
new SNeRV/HiNeRV-specific codec silo is added:

- `src/tac/substrates/_shared/decoder_state_codec.py`: shared decoder-state
  portfolio with fp16, int8, int4, int2, scale-bundled paths, and measured
  envelope stats. This is the first target for SNeRV/HiNeRV mixed-precision
  decoder grammar.
- `src/tac/substrates/_shared/int_stream_codec.py`: unsigned integer stream
  codec with fixed-width packing, packed bitmask, varint, delta-zigzag, and
  zero-run modes. This is the right reusable home for bit-mask, run-length, and
  sparse zero-only stream handling.
- `src/tac/packet_compiler/int_payload_bit_layouts.py`: byte-level lossless
  layouts for flat, nibble-plane, and bitplane payload transforms. Use this as
  the final-rate attack layer before entropy coding.
- `src/tac/archive_diet_pack.py` and packet compiler modules under
  `src/tac/packet_compiler/`: existing deterministic repack and packet grammar
  surfaces. Use these to keep receiver proof and archive byte accounting
  explicit.
- `src/tac/substrates/_shared/mlx_score_aware/coder_qat.py` and
  `src/tac/optimization/optimizer_scheduler_registry.py`: reusable QAT and
  optimizer recipe surfaces. SNeRV/HiNeRV should bind to these rather than
  inventing separate optimizer/QAT descriptors.

Design implication: the scorer-loop fit and rate attack should meet at decoder
weights. NES/local score fitting chooses useful decoder deltas; the shared
codec portfolio then decides whether those deltas are fp16/int8/int4/int2,
bitmask, zero-run, or omitted. The byte allocator should price this as a
receiver grammar decision, not as post-hoc fake quant.

## Blockers Preserved

- `local_smoke_only_not_full_600_pairs`
- `paired_contest_cpu_cuda_pass_missing`
- `mixed_precision_decoder_payload_grammar_not_byte_optimized`
- PR101 CPU recovery remains pending, so no new full-video/exact/CUDA work is
  authorized from this finding.

## Next Local Step

Run a broader bounded local continuation with the same artifact path:

1. 4-pair stratified NES smoke with the same zero-pose-worsening guard.
2. If contraction remains acceptable, implement decoder-state codec adaptation
   using the shared `decoder_state_codec` plus `int_stream_codec` surfaces.
3. If contraction collapses, move NES objective into MLX scorer-loop/full-main
   training and keep this 2-pair artifact as a positive local signal, not as a
   promotion path.
