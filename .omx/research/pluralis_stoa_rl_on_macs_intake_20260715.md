# Intake: Pluralis "RL post-training on Macs" (Stoa) — warm-start-from-divergence read

**Source:** https://pluralis.ai/blog/rl-post-training-on-macs · repo github.com/PluralisResearch/stoa
(operator-routed 2026-07-15). Per `PAPER_WARM_START_FROM_DIVERGENCE`: trace the assumption fork,
import what survives OUR premises (V9·CGauge dense single-instance overfit, 95% wall = frozen
EfficientNet-B2 fwd+bwd, drift-OK training / bit-exact decode+verdict authority).

## The assumption fork (why the headline architecture does NOT transfer)

Stoa: LFM2.5-8B-A1B MoE RL post-training — **rollout-heavy** (int8 MLX inference on 14 Macs, custom
Metal paged-attention), **gradient-light + off-policy-tolerant** (single remote B200 Megatron bf16
trainer, 3-version staleness budget, DPPO token gate δ=0.2, Dr.GRPO). Their economics: inference ≫
training compute, and RL tolerates a drastic off-policy gap. OURS is inverted: gradient-heavy dense
descent on one instance; no rollouts; staleness-tolerant split-brain training does not apply to a
single-trajectory optimizer. Route-or-dismiss verdict on the topology: NOT APPLICABLE — which per
doctrine is the START, not the end.

## What SURVIVES the fork (4 imports, each with an owner)

1. **int8-grid update absorption as a RATE lever (the deep one).** Measured: per optimizer step only
   **0.55% of weights cross the int8 grid**; fp32 master updates are absorbed by the grid; their PULSE
   delta-sync compresses a 9 GB checkpoint to **median 82 MB (~100×)** by coding only grid-crossings.
   OUR mapping: counted archive bytes ARE weights. (a) Late-stage/finisher training deltas on a
   quantization grid are extremely sparse → stage-checkpoint banking (#444) can delta-code successive
   stage EMAs against an anchor; (b) train-on-the-grid (QAT-adjacent, #496 M+Adam rate lever) makes
   the SHIPPED weights natively grid-sparse vs post-hoc quantization; (c) sister of flat-minima/MDL
   (#242). Owner: fold as a design input to #496 + #444 (witness-line rate batch #406 when it fires).
2. **Custom Metal kernels for a SPECIFIC architecture = #478 existence proof.** They shipped
   production paged-attention Metal kernels specialized for LFM2.5 inside mlx-lm. Same class as our
   grouped-backward 17× win; strengthens the build case for the #478 conv suite (pointwise-GEMM +
   depthwise) specialized for EfficientNet-B2. Owner: #478 (burn-down line). Repo worth mining for
   kernel patterns: PluralisResearch/stoa native kernels.
3. **Unified-memory paged/prefix cache discipline = mem-for-compute receipts.** Their stock cache
   saturates at 32 sequences; the unified-memory paged/prefix store keeps scaling to 64 (+30%
   episodes/hr, 41% fewer prefill tokens). Confirms the M5 campaign lever #6 (pin/precompute, trade
   memory for compute; 128 GB is the structural advantage). Owner: m5max campaign (batch-3 arm).
4. **Star-topology / object-store sync (no inbound connections) = fleet option pattern.** R2
   write-once objects + version pointer + zero-egress; cold join 124 s; no Mac-to-Mac links. Maps to
   #297 (EXO/JACCL fleet option) as the LOW-infrastructure alternative topology, and to CUDA-hybrid
   checkpoint publishing (CUDA trains → publishes grid-delta → M5 byte-closes + CPU-verdicts
   asynchronously). Owner: #297 (banked) + the CUDA campaign's harvest step.

## Explicitly NOT imported (with reasons)

- DPPO token gate / Dr.GRPO / staleness budget: RL off-policy machinery; our optimizer consumes its
  own gradients — no ratio to gate. (The spike-guard already covers our divergence-drop analog.)
- int8 INFERENCE for the verdict: verdict authority stays CPU-torch fp32 (advisory) + numpy-fp32
  decode (authority) — an int8 shadow-verdict would be a surrogate-of-a-surrogate; only admissible
  later as trend telemetry, never a verdict. Not queued.
- Multi-Mac rollout fan-out: no rollouts in our workload.

**Pointer honesty:** intake/means; pointer 0.19108 UNMOVED. papers-checked row: pluralis_stoa_2026
→ verdict IMPORT-4 (grid-delta rate lever · #478 existence proof · mem-for-compute receipts · star
topology), NOT-APPLICABLE (RL off-policy stack) with divergence traced above.
