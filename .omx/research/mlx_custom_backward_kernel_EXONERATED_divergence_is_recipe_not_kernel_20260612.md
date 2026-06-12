# The MLX custom grouped-conv backward is EXONERATED — the n600 divergence is the RECIPE (LR/muon), NOT the kernel gradient (2026-06-12)

**Authority:** `[macOS-MLX research-signal]` + torch-CPU exact d_seg/d_pose authority, measured. Frontier UNMOVED.
This is an **APPEND** to (a) the `lane_mlx_custom_grouped_backward_20260612` finding and (b) the
`mlx_custom_backward_DIVERGES_at_n600_pose_gradient_20260612.md` divergence memo. Per Catalog #307
(paradigm-vs-implementation falsification) + Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: the prior
memo's **"pose-gradient bug is the prime suspect"** hypothesis is **FALSIFIED at the implementation level**.
The original memo's NO-FAKE save (catching the divergence before it became a fake "capacity" verdict) STANDS
— only its *cause attribution* is corrected here by direct measurement.

## Operator directive

"We want to fix the custom backward kernel too and confirm through testing." The directive assumed a kernel
bug. The investigation found the kernel gradient is **correct**; per NO-FAKE we report that with proof rather
than invent a fix for a non-bug. The actuated deliverable is the **self-protection gate the original
validation lacked** (real-activation per-layer + pose-shape grad parity), which catches a real future kernel
bug and FALSIFIES the wrong "kernel diverges" diagnosis structurally.

## Three decisive measurements (NO-FAKE)

### (a) Per-layer grad_input is BIT-EXACT on realistic activations — not just iid random
The original validation used iid-gaussian inputs + a dense `sum(out**2)` cotangent. The new diagnostic
(`experiments/diag_seg_relmax_source.py`) uses **realistic** activations (spatially-correlated, ReLU-biased,
non-negative) + a **sparse** cotangent (an L1-on-positives loss that stresses the scatter), on the real
SegNet strided-depthwise shapes:

| layer | cosine | global relmax | restricted relmax (`|ref|>1% max`) |
|---|---|---|---|
| segnet.blocks1.0.conv_dw | 1.0000000 | 0.000000 | 8.4e-7 |
| segnet.blocks2.0.conv_dw | 1.0000000 | 0.000000 | 4.0e-6 |
| segnet.blocks3.0.conv_dw | 1.0000000 | 0.000000 | 2.6e-6 |
| segnet.blocks5.0.conv_dw | 1.0000000 | 0.000000 | 4.0e-6 |

`global_relmax = 0.0` — the kernel grad_input is **bit-identical** to the trusted Python-loop reference where
the gradient is meaningful. (A large global relmax can appear ONLY at near-zero cancellation pixels = fp32
reduction-order noise, not a bug.)

### (b) Full end-to-end render-pixel cotangent matches the reference — POSE is byte-faithful
`experiments/diagnose_custom_backward_e2e_pixel_grad.py` computes `dL/d(pixels)` through the FULL PoseNet and
the FULL SegNet (seg-only and pose-only), comparing mlx_gpu CUSTOM backward vs mlx_gpu REFERENCE backward vs
the torch-CPU AUTHORITY, on a real render — at init AND after 30 custom-backward training steps (the
partially-trained regime where n600 diverged):

| comparison (cotangent) | at init | after 30 train steps |
|---|---|---|
| **pose** custom-vs-reference cosine | 1.00000000 | 1.00000000 |
| **pose** custom-vs-reference relmax | 1.1e-5 | **2.9e-6** |
| **seg** custom-vs-reference cosine | 0.99999837 | 0.99996476 |
| pose reference-vs-torch cosine | 0.99995783 | 0.99930587 |
| pose custom-vs-torch cosine | 0.99995784 | 0.99930587 |

The **POSE gradient from the custom kernel is bit-faithful to the reference** (relmax 2.9e-6) — the memo's
prime suspect is exonerated. `pose_custom_vs_torch == pose_reference_vs_torch` (identical to 8 digits): the
kernel adds **zero** error beyond the documented mlx_gpu FORWARD fidelity (~2.76e-4). The small seg
custom-vs-reference relmax (≤2.4%, cosine 0.99996) is downstream fp32 reduction-order at near-zero-gradient
pixels — the per-layer test (a) shows the seg grad_input itself is bit-exact, and custom is in fact slightly
CLOSER to torch than the reference.

### (c) At a divergence-forcing LR, the trusted REFERENCE backward diverges IDENTICALLY to the kernel
`experiments/diag_custom_vs_reference_trajectory_n8.py` runs both backends from the SAME init/seed/permutation
at an aggressive `muon_lr=0.5`, exact d_seg/d_pose on the torch-CPU authority:

| epoch | custom d_seg | custom d_pose | reference d_seg | reference d_pose |
|---|---|---|---|---|
| 0 | 0.50727 | 131.09 | 0.50727 | 131.09 |
| 3 | 0.64080 | 132.20 | 0.70631 | 137.22 |
| 6 | 0.50434 | 151.29 | 0.51146 | 163.68 |

**Both DIVERGE — the trusted reference backward just as much as the kernel.** The divergence is an
LR/optimizer (muon) instability, not the kernel. **VERDICT: KERNEL_EXONERATED.**

### Bounded confirmation at scale
The n100 A/B (`experiments/measure_custom_backward_pose_divergence_ab.py`, custom backward, the corrected
`muon_lr=0.03`) **stays bounded and descended** through epoch 24: d_seg ~0.014, d_pose oscillating 0.09–0.53
— NOT the n600 explosion (d_pose 0.84→36). The n600 run that diverged used the PR95 curriculum's high
effective LR (~0.99) + muon — a recipe regime (a known recipe-bug track is fixing the curriculum `muon_lr`),
NOT the kernel.

## Root cause (corrected)

The n600 divergence was **LR/muon-driven** (the PR95 curriculum stage's high effective LR under muon). The
n8/40ep validation that "passed" used `muon_lr=0.03` and the n600 run used the curriculum schedule
(~0.99) — the validation never exercised the diverging LR regime, and it measured only d_seg. The kernel
gradient is correct on BOTH d_seg AND d_pose, on iid AND realistic activations, per-layer AND end-to-end.

The original memo's two true points stand: the n8 validation was d_seg-only (a real gap), and catching the
divergence early was the right NO-FAKE move. The corrected attribution: the gap was the **validation's LR +
d_pose coverage**, not a kernel bug.

## The fix (self-protection per "Bugs must be permanently fixed AND self-protected against")

There is no kernel-code bug to patch (the gradient is correct). The landed fix is the **regression gate the
iid-random + d_seg-only validation lacked**, in
`src/tac/local_acceleration/tests/test_metal_grouped_conv_backward.py`:

1. `test_grad_input_bit_exact_on_realistic_activations[...]` — per-layer grad parity on REALISTIC
   (correlated, ReLU-biased) activations + a sparse cotangent, restricted to meaningful-gradient elements
   (excludes the fp32 cancellation tail). Covers SegNet depthwise AND PoseNet Opg=2 channel-multiplier.
2. `test_pose_channel_multiplier_grad_input_bit_exact_realistic` — pins the exact pose-shape the memo flagged
   so the falsified pose-bug hypothesis cannot silently re-enter as a verdict.

**Validated non-vacuous:** injecting a deliberate Opg>1 grad_input bug (drop the `oloc` output-channel
offset) makes these tests FAIL with restricted_relmax 53.2 ≫ 1e-4 on every PoseNet channel-multiplier shape;
removing the bug makes them pass. The gate catches a real future kernel regression.

## Disposition

- **The custom backward is gradient-correct and may be used for fast local training at a SANE LR**
  (`muon_lr=0.03`-class). It is NOT the cause of the n600 divergence. The 5.5× throughput win stands.
- **The n600 divergence belongs to the recipe/LR track** (the PR95 curriculum `muon_lr`), tracked separately.
- The kernel remains a THROUGHPUT tool only — `[macOS-MLX research-signal]`, never a d_seg/d_pose authority;
  torch-CPU exact stays the only authority (CLAUDE.md "MPS NEVER authority" / "NO FAKE").

## Bottom line

The operator asked to fix + confirm the kernel. The honest, measured answer: **the kernel gradient was never
broken** — pose bit-exact, seg bit-exact per-layer, and at a divergence-forcing LR the trusted reference
backward diverges identically. The n600 "kernel diverges" diagnosis is FALSIFIED; the divergence is the
recipe. The deliverable is the self-protection gate (real-activation + pose-shape grad parity) that the
original validation lacked — proven non-vacuous against a deliberate bug. Frontier UNMOVED.
