# Q-FAITHFUL Five-Stage QAT++ Execution Note

Date: 2026-05-02T02:45Z

Purpose: turn the Quantizr five-stage QAT observation into an executable,
evidence-gated upgrade path for the current Shannon-floor push.

## External Anchor

Quantizr PR #55 reports a rounded `0.33` score with a JointFrameGenerator-like
architecture, odd-frame masks, FiLM-on-pose, depthwise-separable convolutions,
and a training script that mimics a five-stage process. The PR text also states
that the training path up/down-samples, clamps/rounds, and resizes through the
evaluation-like path so gradients can compensate for resampling blur and uint8
rounding noise.

The raw public `compress.py` contains a `PipelineRun` stage abstraction with
`anchor`, `finetune`, and `joint` stages plus QAT start controls, EMA, hard
error boosting, cross-entropy/pose weights, warmup, and gradient clipping.

Evidence grade: `external`.

Sources:

- https://github.com/commaai/comma_video_compression_challenge/pull/55
- https://raw.githubusercontent.com/Quantizr/comma_video_compression_challenge/e0b643b0a7c21f62cc93b5d920bcf3fc0d5a33d9/submissions/quantizr/compress.py

## Current Local State

Q-FAITHFUL on Vast A100 `35986044` is already running a true five-phase
schedule:

```text
P1 anchor:   600 epochs @ 1e-3
P2 finetune: 1500 epochs @ 5e-4
P3 joint:    400 epochs @ 1e-4
P4 QAT:      400 epochs @ 5e-5
P5 final:    100 epochs @ 1e-5
```

As of 2026-05-02T02:39Z it was around P1 epoch 380, using A100 SXM4 80GB,
`eval_roundtrip=True`, and a Q-FAITHFUL JointFrameGenerator with 87,836 params.
This is real training signal, but the ETA is too long for it to be the only
deadline path.

Evidence grade: `empirical` until an archive snapshot receives exact CUDA auth
eval.

## QAT++ Upgrade Over Public Quantizr

We should not merely copy five stages. The upgrade is to close the loop around
exact archive anatomy:

1. Stage-level Lagrangian weights:
   Use component traces and PR67/C-057 pair deltas to weight pairs by marginal
   score opportunity per byte. The training sampler, hard-pair curriculum, and
   QAT loss weights should all share this atom ledger.

2. Learned quantization atoms:
   Treat grouped bit depth, per-tensor scale, residual codebook choice,
   stochastic vs deterministic FP4, and QZS3/FP4 packing layout as learnable or
   searched atoms. QAT should optimize for the packed archive stream, not just
   a fake-quant proxy.

3. EMA through QAT and final:
   The EMA shadow must remain active after quantization starts. Export must
   record whether it packed EMA or live weights.

4. Pose-conditioned consistency:
   Pose streams are load-bearing. QAT must train against the exact deployed
   poses or a recorded candidate pose stream. Zero-pose fallback is a preflight
   failure.

5. Snapshot harvesting:
   Every useful checkpoint should be exportable to deterministic archives with
   QZS3/QP1/QPose variants, then exact-evaluated on fast CUDA. T4 is only for
   promotion of an already promising byte-identical archive.

## Highest-EV Decision

Current fastest score path remains the C-057/PR67 public-floor packer basin:
line-search and active-subspace pose refinement can produce promotable archives
in minutes and has already reached A++ `0.3157562807844823`.

QAT++ remains a parallel large-gain lane, but it should be harvested by
snapshot/export/eval checkpoints rather than waiting 20+ hours for the full
schedule. The moment a Q-FAITHFUL snapshot is exportable, run deterministic
packer variants and exact CUDA diagnostics; only T4-promote if it approaches or
beats the C-057/PR67 basin.

## Next Concrete Actions

1. Let the active H100 C-057/PR67 active-subspace search finish and exact-eval
   its accepted archive.
2. Monitor the A100 Q-FAITHFUL training without mutating its artifact tree.
3. If Q-FAITHFUL emits or can safely copy a snapshot, export a snapshot archive
   in a separate output directory and run fast CUDA exact diagnostics.
4. Feed PR67/C-057 hard-pair deltas into the next QAT sampler/quantizer config
   only as empirical weights; archive eval remains the score truth.
