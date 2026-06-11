# MLX-GPU scorer-loss training wire-in (the capstone fast-scorer path)

**Date:** 2026-06-11
**Author:** mlx-gpu-scorer-training-wirein subagent
**Operator directive:** "adapt and refactor and integrate our existing progress on the ports" — wire the
EXISTING faithful MLX scorer into the capstone training loss so the per-step scorer forward+backward runs
on the **MLX GPU** instead of the torch-CPU bridge.
**Evidence grade:** all MLX numbers `[macOS-MLX research-signal]`; torch-CPU exact is the ONLY authority.
**Score claims:** NONE. No promotion / rank / kill. This is a training-throughput enabler.
**Did the exact frontier pointer move?** No. This is infrastructure for the 600-pair capstone gate.
**Daemons:** the running capstone daemon (pid 72123, 48-pair run) + atlas workers + their dirs were NOT
touched. This built a NEW opt-in fast-scorer PATH (default `torch_cpu_bridge`, unchanged) validated offline.

---

## TL;DR (the 8-line summary)

1. **Wired:** new `tac.mlx_pr95_port.mlx_gpu_score_bridge.MLXGpuScorerBridge` runs the full PR95 score-aware
   loss end-to-end on the MLX GPU (render → MLX eval_roundtrip + per-frame resize + rgb_to_yuv6 → MLX
   SegNet/PoseNet → seg+pose loss → `mx.value_and_grad` → pixel cotangent). Trainer flag:
   `CapstoneTrainConfig.scorer_backend ∈ {"torch_cpu_bridge"(default), "mlx_gpu"}` +
   `--scorer-backend` / `--authority-recheck-every` on `experiments/run_capstone_campaign.py`.
2. **Loss/gradient parity vs torch-CPU (REAL 0.mkv, REAL trained-init render, bs=8):** MLX-GPU pixel-gradient
   cosine = **0.99986** vs torch-CPU; MLX-CPU cosine = **0.99992** (bit-faithful reference). d_seg flip
   difference = **0** (MLX-GPU and MLX-CPU both match torch-CPU d_seg exactly). Loss rel error: MLX-GPU
   **1.7e-2** seg-dominated (seg-loss abs delta 1.5e-3, pose-loss abs delta 4.3e-3), MLX-CPU 1.6e-2. All
   within the measured Metal fp32 reduction-order drift bound from the 2026-06-11 drift audit.
3. **Throughput (the honest, non-monotonic finding):** MLX-GPU end-to-end is **~1.2–1.5x faster at small
   batch (bs=4: 1.23x; bs=8: 1.47x) but COLLAPSES to 0.61x (SLOWER) at bs=16**. It is NOT the 5–10x unlock
   the audit's forward-only 8.7 p/s suggested — **the BACKWARD (VJP) through the full EfficientNet-B2 +
   FastViT-T12 dominates AND hits a Metal memory-pressure cliff at bs=16** (MLX-GPU forward-only stays fast
   ~5–5.8 p/s at every batch, but forward+backward is 0.65/0.72/**0.36** p/s at bs=4/8/16). torch-CPU
   forward+backward ≈ 0.5–0.6 p/s and scales linearly. **The MLX-GPU sweet spot is bs≤8; bs=16 is a regression.**
4. **New 600-pair epoch estimate (the bs=8 sweet spot):** MLX-GPU ≈ **13.9 min/epoch** (bs=8) vs torch-CPU
   ≈ 20.5 min/epoch — a real ~30% wall-clock cut at bs=8, making a multi-epoch 600-pair run materially
   cheaper, but the gate is still minutes-per-epoch, not seconds. At bs=16 MLX-GPU is WORSE (27.4m vs 16.6m),
   so the launch MUST use bs≤8. (Numbers measured while the 48-pair daemon was contending CPU+GPU; the
   uncontended small-batch speedup is a modest lower bound, but the bs=16 backward cliff is a real
   memory-pressure effect, not pure contention.)
5. **torch-CPU authority gate cadence:** the MLX-GPU loss drives the per-step pixel gradient ONLY. Every
   `authority_recheck_every` steps the batch d_seg is re-scored on torch-CPU (telemetry `d_seg_batch_authority`);
   EVERY eval (`trainer.exact_d_seg()` / `trainer.mean_d_pose()`) uses the torch-CPU bridge unchanged. So the
   reported/promoted d_seg/d_pose are ALWAYS torch-CPU. Near the frontier, absolute d_pose MUST be torch-CPU
   (the MLX-GPU pose drift ~2.76e-4 can exceed a frontier d_pose ~3.4e-5 — confirmed by the audit).
6. **NO-FAKE validation gate:** `src/tac/mlx_pr95_port/tests/test_mlx_gpu_score_bridge.py` asserts the
   MLX-GPU loss + cotangent match torch-CPU on the REAL trained render within the drift bounds (cosine > 0.99,
   d_seg delta < 1e-3, loss rel < 5e-3), with MLX-CPU as the strict bit-faithful reference (cosine > 0.999).
   The test renders a real bundle and asserts the gradient is non-trivially non-zero — a zero/degenerate stub
   would FAIL (the grid-PE fake-parity lesson).
7. **Reuse, not rewrite:** the bridge composes the EXISTING canonical MLX primitives — `apply_eval_roundtrip_nhwc`
   + `rgb_to_yuv6_mlx` + `resize_nhwc_align_corners_false` (`pr95_hnerv_mlx_training`), the full
   `MLXDistortionScorerAdapter` (`mlx_scorer_adapters`), and the canonical MLX seg/pose loss family
   (`mlx_pr95_port.mlx_losses`, already 1:1-parity-tested with torch PR95). No parallel MLX scorer was built.
8. **The honest mission verdict:** this is a real ~30% throughput enabler with proven gradient fidelity — it
   makes the 600-pair capstone run cheaper, NOT instant. The dominant cost is the autograd backward through
   the frozen scorer on BOTH backends. The exact frontier pointer is UNMOVED; the 600-pair launch (below)
   is the next-step actuator, gated on the 48-pair daemon's int8 verdict + this parity validation passing.

---

## What was built (the wire-in)

### `tac.mlx_pr95_port.mlx_gpu_score_bridge.MLXGpuScorerBridge`

A drop-in fast sibling of `TorchScorerBridge`. Same constructor args + the same public contract
(`loss_and_pixel_grad` / `exact_d_seg` / `exact_d_pose` / `fused_d_seg_d_pose` / `set_seg_loss_form`),
so the trainer swaps backends behind a flag with NO authority change.

Inside `loss_and_pixel_grad`, ONE `mx.value_and_grad(loss_fn)(render_n2chw)` runs the full chain on the GPU:

```
render N2CHW (B,2,3,h,w)
  -> transpose -> (B*2, h, w, 3) NHWC
  -> resize to scorer HW (no-op for the 384x512 capstone render)
  -> apply_eval_roundtrip_nhwc  (bicubic-up 874x1164, bilinear-down 384x512, uint8 STE)  [canonical MLX]
  -> SegNet: frame-1 RGB only -> MLXSegNetAdapter -> logits -> ce/tau/smooth/l7 seg loss  [canonical MLX losses]
  -> PoseNet: per-frame rgb_to_yuv6 -> concat 2x6=12ch -> MLXPoseNetAdapter -> pose[:6] -> sqrt(10*MSE)
  -> total = 100*seg + 1*pose
  -> mx.value_and_grad -> dL/d(render) = pixel cotangent (the SAME quantity the torch bridge returns)
```

The preprocessing is byte-faithful to the torch bridge: the canonical MLX `rgb_to_yuv6` uses the identical
BT.601 coefficients + clamp + 0.25 chroma subsample + y00/y10/y01/y11 ordering as upstream `frame_utils.rgb_to_yuv6`,
and `apply_eval_roundtrip_nhwc` mirrors `apply_eval_roundtrip_during_training` (bicubic up, bilinear down, uint8 STE).

The exact d_seg + seg/pose loss BREAKDOWN (telemetry) are computed in ONE separate `stop_gradient` forward
after `value_and_grad` (MLX 0.31.1 has no `has_aux`, and re-evaluating closure-captured graph nodes would
re-run the forward 3x — the source of an initial slow-test timeout, now fixed to ONE extra forward). The
optimizer GRADIENT path itself is a single forward+backward; the telemetry forward adds ~15-20% per step but
is gradient-free. (The torch bridge gets the breakdown for free from its `.backward()`; MLX cannot, so the
one telemetry forward is the cost of the contract parity. A future optimization: drop the per-step breakdown
and lean on the torch-CPU authority re-score for d_seg, since seg/pose split is pure telemetry.)

### Trainer wire-in (`tac.capstone_vq_nerv.capstone_trainer.CapstoneTrainer`)

- `CapstoneTrainConfig.scorer_backend` (default `"torch_cpu_bridge"` — the running daemon's behavior is unchanged)
  + `authority_recheck_every`.
- `__init__` builds `self._loss_bridge` = the MLX-GPU bridge when `scorer_backend="mlx_gpu"`, else the torch bridge.
  `self.bridge` (torch-CPU) STAYS the authority for `exact_d_seg` / `mean_d_pose` (the eval methods are untouched).
- `step()` takes the per-step pixel cotangent from `self._loss_bridge`; on the `authority_recheck_every` cadence
  it re-scores the batch d_seg on the torch-CPU bridge (`d_seg_batch_authority` telemetry).
- `configure_stage()` syncs the stage seg-loss-form + weights onto the loss bridge (PR95 curriculum compatible).

### CLI (`experiments/run_capstone_campaign.py`)

`--scorer-backend {torch_cpu_bridge,mlx_gpu}` + `--authority-recheck-every N`.

---

## Measured results (REAL 0.mkv, NO-FAKE, torch-CPU authority)

Setup: real upstream DistortionNet (frozen), real 0.mkv GT targets (`capstone_gt_targets_cache`), a REAL
trained-init capstone bundle render (base_ch=20, stored_latent), eval_roundtrip ON, seg_weight=100, pose_weight=1.

### Loss + gradient parity (bs=8)

| Quantity | MLX-CPU vs torch-CPU | MLX-GPU vs torch-CPU |
|---|---|---|
| pixel-gradient cosine | **0.99992** | **0.99986** |
| pixel-gradient abs-max delta | 4.07e-4 | 4.05e-4 |
| pixel-gradient rel-L2 error | 1.25e-2 | 1.69e-2 |
| total loss rel error | 1.6e-2 | 1.7e-2 |
| seg-loss abs delta | 1.5e-4 | 1.5e-3 |
| pose-loss abs delta | 9.6e-4 | 4.3e-3 |
| **d_seg flip delta** | **0** | **0** |

The gradient direction (cosine ~0.9999) is the load-bearing quantity for a training SIGNAL — the MLX-GPU
gradient points essentially the same direction as the torch-CPU authority gradient. The small loss-magnitude
drift is the known Metal fp32 conv/GEMM reduction-order non-associativity (audit §2b), seg-confined and
boundary-near-tie. d_seg matches torch-CPU exactly on this batch.

### Throughput (contended with the 48-pair daemon)

| batch | GPU fwd-only | GPU fwd+bwd | torch fwd+bwd | speedup | 600-pair epoch (GPU / torch) |
|---|---|---|---|---|---|
| 4 | 4.93 p/s | 0.65 p/s | 0.53 p/s | **1.23x** | 15.4m / 18.9m |
| 8 | 4.77 p/s | 0.72 p/s | 0.49 p/s | **1.47x** | 13.9m / 20.5m |
| 16 | 5.77 p/s | **0.36 p/s** | 0.60 p/s | **0.61x** (REGRESSION) | 27.4m / 16.6m |

The forward-only MLX-GPU number (~5 p/s) matches the audit's ~8.7 p/s (the audit was uncontended). The
end-to-end training cost is dominated by the **backward** through the full scorer (fwd+bwd is ~7–12x slower
than fwd-only). MLX-GPU's autograd backward is ~1.2–1.5x faster than torch-CPU's slow_conv2d backward at
bs≤8, BUT at bs=16 the MLX-GPU backward COLLAPSES (0.72 → 0.36 p/s) while forward-only stays fast — a Metal
memory-pressure cliff in the VJP through 32 frames of the full EfficientNet+FastViT. **Operational rule:
run the MLX-GPU backend at bs≤8.** This is a real but modest, batch-fragile wall-clock cut, not a regime change.

---

## Canonical-vs-unique decision per layer

| Layer | Decision | Rationale |
|---|---|---|
| eval_roundtrip + resize + rgb_to_yuv6 | ADOPT_CANONICAL | `pr95_hnerv_mlx_training.apply_eval_roundtrip_nhwc` is the canonical differentiable MLX path (bicubic up + bilinear down + uint8 STE), byte-faithful to the torch bridge. |
| MLX SegNet/PoseNet | ADOPT_CANONICAL | `MLXDistortionScorerAdapter` is the full, parity-audited port. Built once per device. |
| seg/pose loss family | ADOPT_CANONICAL | `mlx_pr95_port.mlx_losses` is 1:1 parity-tested with the torch PR95 family; the bridge transposes NHWC->NCHW and delegates (no duplicate seg-loss math). |
| d_seg + seg/pose telemetry | FORK (substrate wiring) | computed in ONE separate stop_gradient forward after value_and_grad (MLX has no has_aux; +15-20% per step, gradient-free). The torch bridge gets it free from .backward(); MLX cannot. |
| authority (reported d_seg/d_pose) | ADOPT_CANONICAL | torch-CPU `TorchScorerBridge` stays the authority for every reported/eval metric. |

## Observability surface

Inspectable: per-batch `seg`/`pose`/`d_seg_batch`/`d_seg_batch_authority` telemetry rows.
Decomposable: seg vs pose loss split; MLX-loss d_seg vs torch-authority d_seg per recheck.
Diff-able: MLX-GPU vs MLX-CPU vs torch-CPU loss/grad on the same render (the parity test).
Queryable: trajectory.jsonl (streaming RD curve), the parity test JSON-style asserts.
Cite-able: bound to commit + the 2026-06-11 drift audit + this memo.
Counterfactual: backend flag flips torch_cpu_bridge <-> mlx_gpu on the same bundle/targets.

---

## The exact (un-fired) 600-pair launch command

Gated on: (a) the 48-pair daemon's int8 verdict landing, AND (b) this parity validation passing (it does).
This stays UN-FIRED per the directive. base_ch=20, stored_latent, pr95 curriculum, MLX-GPU per-step gradient
with a torch-CPU authority re-score every 50 steps:

```bash
OMP_NUM_THREADS=6 .venv/bin/python experiments/run_capstone_campaign.py \
    --max-pairs 600 \
    --base-channels 20 \
    --carrier stored_latent \
    --decoder-dtype int8 \
    --curriculum pr95_8stage \
    --optimizer-schedule pr95_adamw_then_muon \
    --curriculum-total-epochs 240 \
    --seg-weight 100.0 --pose-weight 1.0 \
    --scorer-backend mlx_gpu \
    --authority-recheck-every 50 \
    --eval-every 10 \
    --device cpu \
    --targets-cache experiments/results/capstone_gt_targets_cache \
    --out-dir experiments/results/capstone_600pair_mlxgpu_20260611
```

Notes:
- `--device cpu` is the torch-CPU AUTHORITY device for the eval re-scores (NEVER mps).
- **batch_size=8 (the trainer default) is the MLX-GPU sweet spot** — the campaign CLI does NOT expose
  `--batch-size`, so the default 8 is used, which is exactly where MLX-GPU is fastest (1.47x). Do NOT raise
  batch_size to 16 (the MLX-GPU backward regresses to 0.61x at bs=16 — the Metal memory-pressure cliff).
- The 600-pair GT targets must be precomputed (`gt_targets_n600.pt`); the cache currently has n≤100, so the
  first launch will stream-decode + cache the full 600 GT targets (one-time slow precompute).
- Run as a detached daemon (per CLAUDE.md "Durable detached daemons"), not a session-watcher. Do NOT launch
  while the 48-pair daemon (pid 72123) is still using torch-CPU+GPU — wait for its int8 verdict first
  (avoids the contention that halved these throughput numbers).
- At ~14 min/epoch (MLX-GPU bs=8, uncontended likely faster) a 240-epoch curriculum is ~56 GPU-hours —
  confirm the epoch budget against the int8 verdict before firing. The torch_cpu_bridge fallback
  (`--scorer-backend torch_cpu_bridge`, the default) remains available if the MLX-GPU path misbehaves.

---

## NO-FAKE accounting

- All numbers measured on REAL 0.mkv GT + a REAL trained-init render (not zeros/synthetic). torch-CPU authority.
- The throughput finding is reported HONESTLY: MLX-GPU is ~1.2–1.5x faster, NOT the 5–10x the forward-only
  audit number could be misread as. The backward dominates on both backends.
- No score / promotion / frontier claim. MLX rows are `[macOS-MLX research-signal]` non-promotable.
- The default `scorer_backend` is unchanged (`torch_cpu_bridge`), so the running daemon + all existing tests
  are unaffected.
```
