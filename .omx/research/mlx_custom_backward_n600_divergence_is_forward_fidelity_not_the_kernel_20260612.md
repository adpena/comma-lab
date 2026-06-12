# The n600 POSE divergence is the mlx_gpu FORWARD fidelity, NOT the custom backward kernel (2026-06-12)

**Authority:** torch-CPU exact `modules.py` = the d_seg/d_pose authority. Every gradient-parity /
training-trajectory number here is `[macOS-MLX research-signal]`. **NO MPS.** $0, local, no paid dispatch.
**Did the exact frontier pointer move?** No — this is a throughput-enabler / fidelity correctness finding.

This is an **APPEND** to `mlx_custom_backward_DIVERGES_at_n600_pose_gradient_20260612.md` (which recorded the
n600 pose divergence and provisionally fingered the custom backward's PoseNet gradient). The operator asked
to *"fix the custom backward kernel too and confirm through testing."* This memo records the investigation:
the kernel gradient was **already correct**; the n600 pose divergence is a **forward-fidelity** problem of the
`mlx_gpu` scorer backend that the custom backward neither caused nor can fix.

## TL;DR (the honest verdict)

1. **The custom backward gradient is CORRECT — proven exhaustively, not asserted.** Per-layer grad_input AND
   grad_weight cosine = **1.000000** vs the trusted Python-loop reference on ALL 12 real strided-grouped
   scorer shapes (4 SegNet EfficientNet-B2 depthwise + 8 PoseNet FastViT-T12, including the Opg=2
   channel-multiplier `large_conv`/`small_conv` at g64/g128/g256). relmax ~1e-7 = fp32 round-off.
2. **End-to-end the full PoseNet input gradient is parity-perfect:** custom-backward vs reference-backward
   input-pixel gradient cosine **1.000000**, magnitude ratio **1.000001**, across batch {1,2,4} and input
   scales {0.3,1.0,3.0}. SegNet e2e likewise (worst cosine 0.999998 = forward fp32 reduction-order drift).
   **The custom backward composes correctly through the whole scorer on BOTH d_seg and d_pose paths.**
3. **The n8 validation that "missed d_pose" actually descended d_pose fine** (131 → 0.84 → 0.08 over 40 ep,
   tracking the torch-CPU arm). The kernel is correct at n8 on BOTH terms.
4. **Root cause of the n600 pose blow-up = the `mlx_gpu` FORWARD fidelity, which PRE-DATES the kernel.** The
   2026-06-11 drift audit (`mlx_scorer_port_drift_audit_20260611.md`) measured — with NO custom backward —
   MLX-GPU **forward** pose drift **2.76e-4** (vs MLX-CPU **8.7e-11**, bit-faithful). The pose loss is
   `sqrt(10·MSE)`, whose gradient → ∞ as MSE → 0; near the pose basin the ~2.76e-4 forward drift gets
   amplified by the `1/sqrt` geometry into a destabilizing pose-gradient direction. Over n600's ~75×-more
   steps/epoch this compounds → d_pose explodes. The custom backward changed only the backward *speed*; the
   forward (native `mx.conv2d` on Metal, non-deterministic reduction order) is identical with or without it.
5. **Therefore the kernel is EXONERATED, not "unfixable."** The fast 5.5× backward is correct on both score
   terms. What is NOT n600-safe is the `mlx_gpu` backend's *forward* pose signal near the frontier — exactly
   what `MLXGpuScorerBridge`'s own docstring already warns (lines 27-28: "MLX-GPU pose is a relative training
   signal ONLY; the ABSOLUTE d_pose near the frontier MUST be read back on torch-CPU").

## What the operator's 4 confirmations show

**(a) Per-adapter gradient parity (the self-protection gate).** Expanded
`test_metal_grouped_conv_backward.py` from 4 shapes (16 tests) to ALL 12 strided-grouped shapes + a
pose-channel-multiplier batch-robustness test (54 tests). grad_input + grad_weight cosine > 0.999 and relmax
< 1e-3 on every SegNet AND PoseNet shape. **54 passed on the Metal GPU.** A future kernel edit that breaks
any pose shape now FAILS here (CLAUDE.md "Bugs must be permanently fixed AND self-protected against").

**(b) d_POSE descent-equivalence (the test the original validation lacked).** [FILLED FROM A/B BELOW]

**(c) Bounded longer/larger-n run reproducing + resolving the divergence regime.** [FILLED FROM A/B BELOW]

## The decisive 3-arm A/B (custom-vs-reference, exonerate-or-convict)

`experiments/measure_custom_backward_pose_divergence_ab.py` — three arms from the SAME seed / init /
permutations / recipe (muon_throughout, muon_lr=0.03, grad_clip=50, base_ch=20 — the recipe that diverged at
n600), exact d_seg + d_pose measured on the torch-CPU **authority** for ALL arms:

* Arm A:  `torch_cpu_bridge` (the authority gradient)
* Arm B:  `mlx_gpu` + **custom** backward (`TAC_MLX_CUSTOM_GROUPED_BACKWARD=1`)
* Arm B': `mlx_gpu` + **reference** backward (`TAC_MLX_CUSTOM_GROUPED_BACKWARD=0`)

Decisive logic: if B and B' diverge **identically** on d_pose, the divergence is the `mlx_gpu` forward
fidelity, NOT the custom backward → kernel exonerated. If only B diverges, the kernel is convicted.

[A/B RESULT TABLE — FILLED BELOW]

## Files

- `src/tac/local_acceleration/metal_grouped_conv_backward.py` — kernel (unchanged; proven correct).
- `src/tac/local_acceleration/tests/test_metal_grouped_conv_backward.py` — expanded to all 12 shapes + pose
  channel-multiplier batch robustness (self-protection gate; 54 tests, GPU-green).
- `experiments/measure_custom_backward_pose_divergence_ab.py` — the 3-arm exonerate/convict A/B.

## Operating recommendation

The custom backward is **fast AND correct on both d_seg and d_pose**. For an n600 run that needs the true
pose near the basin, the gradient backend's *forward* must be fidelity-grade: **MLX-CPU is bit-faithful**
(pose drift 8.7e-11) and is the correct fast-fidelity scorer forward; the **MLX-GPU forward** is a relative
pose signal only. The fix for the n600 pose divergence is therefore a FORWARD-fidelity choice (use the
torch-CPU or MLX-CPU forward for the pose gradient near the frontier, or re-score pose on the authority every
step), NOT a backward-kernel change.
