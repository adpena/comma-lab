# Lane S — Self-Compressing renderer codec (engineering complete)

**Date:** 2026-04-27. **Status:** ENGINEERING COMPLETE in working tree, not committed yet, ready for Vast.ai dispatch.

## What it is

Per-channel learnable bit-depth (Szabolcs Csefalvay arXiv 2301.13142) applied to the dilated-h64 baseline renderer convs. Lagrangian rate penalty drives avg bits down from 8 (init) toward 2.5 (target) during training. Result: per-channel bit allocation rather than the uniform-4-bits FP4 scheme. New `SCv1` magic file format with LZMA-compressed body.

## Files modified (all uncommitted)

- `src/tac/self_compress.py`: `SelfCompressingConv2d` extended with full `nn.Conv2d` kwargs (stride/groups/padding_mode). New helpers: `swap_renderer_convs_with_self_compress`, `list_self_compress_layers`, `renderer_total_weight_bits`, `renderer_average_bits_per_weight`, `compute_renderer_rate_penalty`, plus `SC_PROTECTED_NAME_PATTERNS`.
- `src/tac/renderer_export.py`: new `export_self_compressed_renderer` / `load_self_compressed_renderer` functions, `SCv1` magic. `detect_checkpoint_type` and `load_any_renderer_checkpoint` recognize it.
- `submissions/robust_current/inflate_renderer.py`: new `b"SCv1"` dispatch branch (CRITICAL tier). Calls `tac.renderer_export.load_self_compressed_renderer`; raises clear RuntimeError if tac is missing.
- `src/tac/profiles.py`: `SELF_COMPRESS_RENDERER_SMOKE` (100ep, target=4.0) and `SELF_COMPRESS_RENDERER_FULL` (1980ep, target=2.5, mirror of `dilated_h64_half_frame`).
- `src/tac/experiments/train_renderer.py`: 5 new CLI flags + resolvers, post-build SC swap, in-loop Lagrangian rate penalty, post-step `bit_depth.bits.clamp_(0, 8)`, **automatic disable of `--auth-eval-on-best` when SC is on** (FP4A export would lose all SC gain — must use SCv1 inflate path instead).
- `src/tac/tests/test_self_compress_renderer.py`: 24 new tests, all passing. Covers SC primitives, swap helper, SCv1 round-trip, byte-stability, FP4-vs-SC byte comparison, inflate dispatch, Lagrangian penalty, profile sanity.

## Architecture: which layers stay FP32

Per Lane F finding (FP4-QAT on dilated-h64 caused +0.144 PoseNet vs floor) + standalone postfilter "FiLM 3rd most scorer-sensitive":

PROTECTED (FP32): `renderer.head`, `motion.head`, `*.fuse_conv`, `*.fuse2_conv`, `film_*.scale`, `film_*.shift`, all `nn.ConvTranspose2d`.

For dilated-h64 (288K params): 16 layers swapped (243K params, 84%), 3 protected, 1 ConvTranspose2d skipped.

## Predicted byte counts (verified by smoke export of same arch)

| Variant | Bytes | bits/param |
|---|---|---|
| FP4-QAT | ~144 KB | 4.00 |
| **SCv1 @ 2.5 mean SC bits** | **~115 KB** | **3.20** |
| SCv1 @ 2.0 mean SC bits + 25% pruning | ~92 KB | 2.55 |

Round-trip diff: 0.0 (byte-exact). Export is byte-deterministic across runs.

## Key engineering bug found and fixed

The naive `named_modules()` walk for export iterates BOTH the SC layer AND its inner `nn.Conv2d`, causing each weight to be stored TWICE (~574KB body). Fix: collect inner Conv2d ids and skip them in the iteration (`sc_inner_module_ids`). After fix: 105KB compressed body. **Caught by smoke test, not in production.**

## Recommended Lane S launch config

- Profile: `self_compress_renderer_full` (1980 epochs, 5-phase Quantizr-style)
- GPU: RTX 4090 Vast.ai $0.25/hr × 5h = **$1.25**
- target_bits=2.5, init_bits=8.0, lambda_start=0.0, lambda_end=1.0, ramp_start_frac=0.30
- Mask format: half-frame (Quantizr trick)
- Pose artifact: reuse Lane A optimized_poses.bin
- Predicted archive: ~255 KB (renderer 115 + masks 125 + poses 15)
- **Predicted contest-CUDA score band: 0.85-1.10** (best beats baseline 0.90 by 0.05; worst +0.20)

## Falsifiability checks (kill criteria)

1. Phase 1 end: pixel L1 < 12 AND avg SC bits == 8.0 (lambda not active yet).
2. Phase 2 mid: avg SC bits monotonically decreasing AND scorer < 8.0.
3. Phase 4 end: avg SC bits ≈ target ± 0.5 AND fp4_scorer < 1.5 AND renderer.bin < 130KB.
4. Auth eval (separate launch): contest-CUDA < 1.20.

If #1-#3 fail → kill, SC machinery has config bug.
If #4 fails → BUGGED not DEAD; audit pose threading + protected-layer set per Lane F-V2 protocol.

## Engineering risks

1. Protected-layer set may be insufficient. If PoseNet > 0.10 vs baseline 0.011, add `*.bottleneck.conv2` and `motion.up_conv` to protections.
2. Lagrangian schedule may need tuning. If avg bits doesn't reach target by phase-4 end, try `lambda_end=2.0`. If crashes too fast, try `lambda_end=0.5`.
3. SC pack/unpack assumes inner Conv2d state. Future arch changes that replace inner conv must jointly update swap + export.
4. `bit_depth.bits` uses same LR as conv weights (single optimizer). Standalone postfilter trainer has separate `lr_bits=1e-2`. For finer control, future work: split into two param groups.

## Concurrency note

Lane S engineering is non-overlapping with the in-flight Lane F council audit (which reads `experiments/qat_finetune.py`, `src/tac/quantization.py`, `scripts/remote_lane_b_fp4_qat.sh` — all read-only here).

## Test summary

- 24 new Lane S tests pass.
- 7 existing Lane I dispatch tests still pass.
- Original `self_compress.py` smoke still passes.
- Original `renderer_export.py` smoke still passes.

## Next steps

1. Codex 2nd-approver review of CRITICAL-tier `inflate_renderer.py` SCv1 branch.
2. Operator launches `self_compress_renderer_full` on Vast.ai 4090.
3. After training: separate Vast.ai dispatch for auth eval using SCv1 inflate path.
4. Stack with Lane A pose TTO for combined score (predicted: 0.55-0.75).
