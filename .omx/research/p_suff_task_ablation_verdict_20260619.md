---
title: "P-SUFF / TASK-ABLATION verdict — how much of the 0.19110 frontier does the frozen scorer IGNORE?"
authority: "[contest-CPU advisory] NON-PROMOTABLE — pointer UNMOVED 0.19110; $0; CPU-only; no PR"
score_claim: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false
pointer_moved: false
date: 2026-06-19
verdict: RED_FRONTIER_NEAR_TASK_RD_FLOOR_LITTLE_INVARIANT_MASS
producer: experiments/probe_p_suff_task_ablation.py
result_json: .omx/research/p_suff_task_ablation_20260619T055915Z.json
cross_refs:
  - .omx/research/vcm_theory_primitive_layer_20260619T033429Z.md   # the dominated-rung / indirect-RD thesis this tests
  - .omx/research/vcm_taskaware_compression_sota_survey_20260619T033854Z.md  # P-SUFF / P-R3
  - .omx/research/frozen_instance_exploit_catch_up_then_surpass_vcm_20260619.md  # §3.4 "the per-video INR may be RD-suboptimal"
  - experiments/results/pr110_payload_entropy_recode_20260610/submission_dir  # the 0.19110 frontier vehicle
---

# P-SUFF / TASK-ABLATION — the scorer-invariance diagnostic on the binding rate term

## TL;DR (the headline number)

**The frozen SegNet+PoseNet IGNORE essentially NONE of the 0.19110 frontier's archive.**
The total **scorer-invariant byte mass is ~0.7%** (1,156 B of 177,169) — and even that
0.7% is NOT free: coarsening every tensor jointly to its individual precision floor pushes
the combined task hit to **ΔS_task = +0.0258** (projected S would *rise* to ~0.209, not fall).
**VERDICT: RED.** This frontier vehicle already codes near the **task-RD sufficient statistic**
on its binding rate term. The "dominated-rung" / indirect-RD upside that #151's theory layer
predicted is, for THIS vehicle, **small** — a real, important closure that bounds the whole
task-space program.

All numbers `[contest-CPU advisory]` NON-PROMOTABLE. Pointer UNMOVED 0.19110. $0, CPU-only,
inference-only (no MPS → no contention with the live train).

## What was measured (NO-FAKE, measurement-first)

Decoded the frontier archive (`pr110_payload_entropy_recode` = 0.19110) into its **28 HNeRV
decoder tensors + (600,28) latents**. The decoder section is the **real-measured 161,104 B =
90.9% of the 177,169 B archive = the binding rate term**. Then, measuring **REALIZED**
Δd_seg/Δd_pose through the **REAL frozen contest SegNet+PoseNet + the EXACT camera-resolution
eval roundtrip** (decoder → bicubic-up 384×512→874×1164 → frontier channel-bias → clamp+round
uint8 → `DistortionNet` → argmax-flip vs cached GT-argmax / pose6-MSE vs cached GT-pose), with
ΔS computed from components using the **real nonlinear √(10·d_pose)** term:

- **Baseline (n=48 pairs):** d_seg=0.000538, d_pose=1.49e-5, S=0.18401 — strong parity with the
  frontier's full-600 report (d_seg=0.00056, d_pose=2.94e-5, S=0.19110). The S=0.184 vs 0.191
  gap is the honest **subset variance** (n=48 of 600; the per-pair-average metric on a subset is
  a faithful-definition estimate, flagged below).
- **Phase 1** — per-decoder-tensor ablation (zero / mean) of all 28 tensors.
- **Phase 2** — per-latent-channel ablation (zero / mean) of all 28 latent dims.
- **Phase 3** — per-tensor bit-precision sweep {8,7,6,5,4,3,2} (the precision floor).
- **Phase 4** — JOINT verification: coarsen ALL tensors to their per-tensor bit-floors at once,
  with REAL re-encoded archive bytes + the REAL combined task hit (per-tensor ΔS do NOT sum).

## The four findings

### 1. NOTHING is deletable. (deletion-domain RED)
**0 of 28 decoder tensors are zeroable** and **0 of 28 latent dims are ignorable** within the
ΔS_task ≤ +0.005 tolerance. The *most* ignorable latent dim (dim19) still costs **ΔS_task=+0.078**
when zeroed; the least (dim17) costs **+4.28**. Zeroing the worst weight tensor (`blocks.2.weight`)
drives d_seg to ~0.51 (total breakdown, ΔS_task=+94.6). Every byte of the renderer is doing
task-relevant work. The frontier is NOT carrying a deletable "reconstruct-RGB-the-scorer-ignores"
mass at the tensor/channel granularity.

### 2. The BINDING mass is at its precision floor. (precision-domain RED)
The **6 largest weight tensors** — `stem.weight`, `blocks.0/1/2/3.weight`, `blocks.5.weight`
≈ **146 KB = ~90% of the decoder** — ALL have **bit-floor = 8**: they cannot lose even ONE bit
below the shipped int8 without ΔS_task > +0.005. The frontier's int8 quantization is **already
at the task-relevant precision floor** on the mass that dominates the rate term. Only the small
tensors (refine/skips/rgb/biases, ~6% of bytes) have sub-int8 floors (b4–b7), and their total
individual saving is ~1.2 KB.

### 3. The joint precision cut is NOT free — and barely exists. (the honest aggregate)
Coarsening every tensor to its individual bit-floor simultaneously saves only **1,156 B (0.65%)**
but the **combined ΔS_task = +0.0258** (the per-tensor <+0.005 floors COMPOUND past tolerance).
Projected S with the byte cut = **0.20908 > baseline 0.18401**. There is no free precision-domain
rate cut to take. (This is exactly the failure the joint-verification was built to catch: a naive
sum of per-tensor floors would have falsely implied a small win; the measured joint hit is RED.)

### 4. Axis-specific structure (feeds the bit-allocator #154, the quotient codec #155)
- **`rgb_0.weight` + `rgb_0.bias` are POSE-ONLY** (Δd_seg ≈ 0, Δd_pose > 0): they render frame-0,
  which **SegNet ignores** (last-frame-only) but **PoseNet reads** (both frames). This is the
  cleanest empirical instance of the frozen-scorer reading-asymmetry — a real per-tensor task-axis
  label. (Symmetrically, `rgb_1.*` carries the SegNet-scored frame-1 and has higher d_seg effect.)
- **Task-effect-per-byte ranking** (the bit-allocator prior): small biases have the highest
  per-byte effect (`skips.4.bias`: 9.7e-2 ΔS/byte), big weights the highest absolute effect but
  lower per-byte (`blocks.2.weight`: 3.8e-3 ΔS/byte). The full per-tensor table is in the JSON.
- **20/28 tensors hold at b7** individually (int8→7-bit, ΔS_task≤+0.005), summing 3,228 B — but
  per finding 3 the JOINT b7-floor compounds past tolerance, so this is not an exploitable cut.

## Why this is high-value despite RED (the bound it sets)

The convergent theory (#151 indirect-RD, #150 P-SUFF) predicted the per-video INR might be
**RD-suboptimal** because it reconstructs RGB the scorers ignore — the "dominated rung". This
diagnostic **measures that gap directly for the actual frontier** and finds it **small**: the
frontier already sits near its task-sufficient-statistic rate on the binding term. That is a
real closure — it bounds the entire task-space program:

- **A "quotient codec" (#155) that re-codes only what the scorers read will NOT beat this
  vehicle on rate by deleting/coarsening ITS representation** — there is no large invariant mass
  inside it to remove. The dominated-rung surplus this frontier carries is ~0.7%, not the 20%+ a
  GREEN would have shown.
- **The sub-0.15 rate headroom is NOT in pruning the existing renderer.** It must come from a
  *structurally different, smaller* task-sufficient representation (a genuinely lower-rate code of
  the d_seg argmax-cell + d_pose 6-dim sufficient statistics — the #155 surpass probe coding the
  statistic from scratch, NOT pruning this 161 KB renderer), OR from the d_seg/d_pose terms (not
  rate). The diagnostic redirects #154/#155 away from "compress the frontier's bytes" toward
  "build the smaller sufficient code" or "attack d_seg".

## Honest caveats (NO-FAKE)

- **n=48 subset.** Δ is the per-pair-average metric over the first 48 of 600 pairs (a
  faithful-definition estimate; baseline parity S=0.184 vs full-600 0.191 is the subset variance,
  driven mostly by the lower subset d_pose 1.49e-5 vs 2.94e-5). The per-tensor *rankings* and the
  precision-floor structure are robust to n; the absolute ΔS magnitudes carry subset noise. A
  larger-n re-run would tighten the absolute joint-ΔS but cannot flip RED→GREEN (the binding
  tensors are at b8 floor regardless of n).
- **Decoder-only ablation scope.** The frontier's FECa selector (per-pair RGB bias) + DQS1
  (one-coeff patch on selected pairs) are NOT decoder tensors and are tiny byte contributors; they
  shift the absolute baseline a hair but do not change which decoder tensor carries task-effect.
  The ablated 90.9% IS the decoder = the binding mass.
- **Per-tensor "int8 bytes" are numel-share estimates;** the bit-sweep's `section_byte_saving` is
  the REAL re-encoded delta and is the authoritative byte number used for the verdict.
- **ΔS_task holds bytes at the full archive** (the invariance check); the byte saving is reported
  separately as the upside. The joint Phase-4 number combines both (real task hit AT reduced bytes).

## Verdict + next step

**RED_FRONTIER_NEAR_TASK_RD_FLOOR_LITTLE_INVARIANT_MASS.** The 0.19110 frontier is NOT on a
dominated rung in a way we can exploit by pruning it — it codes near the task-sufficient statistic
on its binding rate term (~0.7% invariant mass, not free). The sub-0.15 rate path is **not** "shrink
this renderer"; it is (a) a from-scratch smaller code of the d_seg/d_pose sufficient statistics
(#155, coding the statistic — NOT pruning the INR), or (b) the d_seg term. This diagnostic is the
byte-level sensitivity map that feeds #154 (bit-allocator: the binding mass is at b8, do not spend
the budget re-quantizing it) and #155 (quotient codec: target the statistic, not the INR's bytes).

Pointer UNMOVED 0.19110. No exact-eval row produced (this is a $0 diagnostic; the verdict redirects
the byte campaign, it does not move the score).
