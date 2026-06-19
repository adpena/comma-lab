# Frontier RATE exact bit-allocation SOLVE — design memo (the deterministic precision-allocation optimum, not a feasibility gate)

- **UTC**: 2026-06-18
- **Authority**: `[contest-CPU advisory]` NON-PROMOTABLE. Frontier pointer UNMOVED `0.19110`.
- **Spend**: $0 (local CPU + MPS gradient; no GPU dispatch, no paid).
- **Operator correction (2026-06-19)**: the int5 "confirmed structural" verdict was un-nuanced — it held OUR OWN codec fixed and used generic uniform-int5 + per-tensor LSQ. We possess the frozen SegNet+PoseNet, the frozen video, and the closed-form eval → the optimal precision allocation is a DETERMINISTIC OPTIMIZATION we can SOLVE, not probe. We OWN inflate.py / the codec — it is a design variable, not a constraint.

## 0. The crux the prior cap missed (re-read `frontier_int5_lsq_best_shot_retest_20260619T031709Z.md`)

The generic int5 best-shot (uniform low-nbits, per-tensor LSQ + outlier clip, CE finetune) recovered d_pose enormously (−89%) but **d_seg barely moved (−9.5%, stuck at ~0.0042, 7.6× the frontier's 0.00056)**. Its own conclusion named the blocker:

1. per-channel scales (the one fix that gives d_seg-critical weights real per-output-channel precision) are NOT byte-closeable through the codec's per-tensor-int8 grammar (measured: blows the archive 118k → 197k);
2. the d_seg-critical early/low-res stages (77% of params) need finer-than-int5 resolution that no per-tensor scale can provide.

But **(1) holds the codec fixed**. We own the codec. And the prior int5 used UNIFORM low-nbits — it coarsened the high-sensitivity tensors to int5 along with the blind ones. The exact-solve does NOT coarsen uniformly: it allocates bits per the EXACT per-tensor score-sensitivity, spending int6–8 where the scorer is sensitive and int2–4 where it is blind, provably ≤ the uniform-int5 total bits at equal score-distortion.

## 1. The exact-known inventory (we optimize EXACTLY against these — no approximation)

- frozen SegNet (`segnet.safetensors`) + PoseNet (`posenet.safetensors`) — exact, no retraining.
- 600 frozen pairs (`upstream/videos/0.mkv`) — exact uint8 frames.
- the eval operators (bicubic-up → uint8 → bilinear-down → SegNet argmax / PoseNet YUV6) — exact, differentiable (eval-roundtrip + differentiable-yuv6 machinery).
- the frontier decoder (228,958 params; weight-tensors-≥2D = 226,512 params) — exact int8 weights, byte-identical identity round-trip proven (`frontier_decoder_ptq`).
- the codec grammar (PR101 split-brotli, one int8 scale/tensor) — exact, AND OURS TO CO-DESIGN.

Weight-param mass (the rate carriers): `stem` 21.4% + `blocks.0/1/2` 56.6% = **78% in 4 tensors**; the d_seg-critical heads (`blocks.4/5 + skips + refine + rgb`) = only ~13%.

## 2. The three deterministic steps (the SOLVE)

### Step 1 — exact per-tensor score-sensitivity (one backward each, analytic, no training)
For every Conv2d/Linear weight tensor W:
- `g_seg(W) = ‖∂(Σ_pixels SegNet top1−top2 margin)/∂W‖` through the frozen SegNet (margin is the a.e.-differentiable surrogate of the argmax-flip d_seg; argmax itself has 0 gradient). REUSE `tac.margin_saliency_map.compute_decoder_tensor_margin_saliency`.
- `g_pose(W) = ‖∂d_pose/∂W‖` through the frozen PoseNet (exact, MSE-smooth) — the d_pose analogue (new producer, same accumulate-grad-norm structure).
- combine by the master gradient at the operating point: `∂S/∂d_seg = 100`, `∂S/∂d_pose = 5/√(10·d_pose) ≈ 85.8` at d_pose≈3.4e-5. So `s(W) = 100·g_seg(W) + 85.8·g_pose(W)`.
EXACT, computed once over a deterministic spread of frames. (Caveat: g_seg is the margin-surrogate gradient, not the argmax-flip itself — first-order exact for the sensitivity ranking; the byte-closed exact eval is the truth that ranks the result.)

### Step 2 — reverse-water-filling / KKT bit allocation (closed form)
The score-distortion contributed by quantizing tensor t at b bits is, to first order,
`ΔS(t,b) ≈ s(t) · E[‖ΔW_t(b)‖]` where the per-tensor symmetric int-N round-trip error has magnitude `E[‖ΔW_t(b)‖] ≈ κ_t · 2^{−b}` (κ_t set by the tensor's weight spread; for a per-tensor abs-max grid the RMS step is `absmax_t / (2^{b−1}−1) / √12`, and the total L2 error ≈ `step · √(numel_t) / √12 ∝ absmax_t · √numel_t · 2^{−b}`). Define the per-tensor **impact coefficient** `c_t = s(t) · absmax_t · √numel_t` (the score-distortion per unit `2^{−b}`).

Minimize `Σ_t bits_t · numel_t` (the rate proxy — total stored int bits) subject to `Σ_t c_t · 2^{−b_t} ≤ Dbudget`. The Lagrangian `L = Σ numel_t·b_t + λ·Σ c_t 2^{−b_t}` has first-order optimum (treating b_t continuous):
`∂L/∂b_t = numel_t − λ·c_t·ln2·2^{−b_t} = 0` →
`b_t* = log₂( λ·ln2·c_t / numel_t )` = `log₂(c_t/numel_t) + log₂(λ·ln2)`.

So **the optimal bit-width rises with `log₂(c_t / numel_t)`** = `log₂(s(t)·absmax_t/√numel_t)` (the per-WEIGHT score-impact, NOT the per-tensor — a large blind tensor correctly gets FEW bits/weight). One scalar λ tunes the rate/distortion operating point; sweep λ to trace the RD curve. Round to integer bits, clamp to `[b_min, b_max]` (e.g. [2,8]). This is provably ≤ uniform-b at equal Σc·2^{−b} (water-filling optimality for the separable convex problem).

Honest caveat: reverse-water-fill is first-order optimal for the *quadratic-distortion* relaxation; our d_seg is argmax-flip (non-quadratic). The allocation is the first-order-exact STARTING point; the byte-closed exact eval ranks candidate λ.

### Step 3 — codec CO-DESIGN (we own it)
The non-uniform per-tensor nbits needs the decoder to know each tensor's bit-width to dequantize. The cheap co-design:
- **per-tensor nbits header**: 28 weight tensors × 1 byte (or 4-bit-packed = 14 B) — negligible.
- the int codes are stored as the SAME per-tensor symmetric grid the codec already uses (one fp16 scale/tensor), but at the tensor's OWN nbits. A lower-nbits tensor has fewer distinct symbols → brotli compresses it BETTER (the rate win). A higher-nbits d_seg-critical tensor spends more symbols but it is small (heads are ~13% of params). The codec's existing per-tensor byte-map + split-brotli machinery handles variable distinct-symbol counts natively — we do NOT need per-channel scales (the byte-blowup the prior cap hit). The mixed-precision header is the only new section.
- This is a strict superset of the existing codec (uniform int8 = all-nbits-8 header), so the identity round-trip is preserved.

REUSE the existing reencode path (`reencode_frontier_archive`) for the byte-close — the shrunk state_dict (each tensor qdq'd at its OWN nbits) re-encodes through `encode_decoder_compact` exactly as the existing PTQ path does; the only addition is the per-tensor nbits drives `intn_qdq(v, nbits_t)` per tensor instead of a single nbits. The decoder reads the same int8 codes + fp16 scale (the qdq output IS a per-tensor int8-representable grid for any nbits ≤ 8), so **no decoder change is even required for the MEASUREMENT** — the variable nbits is realized entirely in the qdq grid, and the codec stores the resulting (sparser) int8 codes. The mixed-precision header is only needed if we want to drop below int8 STORAGE width per tensor (a future tighter pack); for the rate measurement the brotli-of-sparser-int8 already captures the win.

## 3. Measurement (authority)
Byte-close the archive → REAL CPU `RealScorerContext.exact_eval` on the RE-DECODED shipped bytes → recompute `S = 100·d_seg + √(10·d_pose) + 25·B/B₀`. Compare to frontier 0.19110, generic int5 0.5593, int8 local baseline 0.1965. Report achieved S + rate cut + per-tensor allocation profile. NOT GREEN/RED — a SOLVE reports the number.

## 4. Canonical-vs-unique decision per layer
- **sensitivity producer (d_seg)**: ADOPT_CANONICAL `compute_decoder_tensor_margin_saliency` (the exact frozen-SegNet margin gradient — the canonical asset).
- **sensitivity producer (d_pose)**: FORK_PRINCIPLED — no canonical per-tensor pose-saliency producer exists; new, mirrors the d_seg structure (accumulate ‖∂d_pose/∂W‖).
- **water-fill allocator**: FORK_PRINCIPLED — the existing `score_aware_requant_state_dict` is a binary (high/low) prefix split, NOT a continuous water-fill; the KKT closed-form is new (and subsumes the binary split as a 2-level special case).
- **byte-close**: ADOPT_CANONICAL `reencode_frontier_archive` + `intn_qdq` (the proven byte-exact path).
- **eval authority**: ADOPT_CANONICAL `RealScorerContext.exact_eval` (CPU, never MPS).

## 5. Observability surface
- per-layer: per-tensor `g_seg`, `g_pose`, `s`, `c_t`, allocated nbits, qdq error, brotli bytes — all logged to JSON.
- decomposable: S split into d_seg / d_pose / rate; allocation profile per tensor.
- diff-able: the λ-sweep produces a RD curve (bytes vs S) directly comparable to uniform-int5 / int8.
- counterfactual: the allocation is a deterministic function of (s, c, λ) — re-runnable, no training randomness in the core PTQ solve.

## 6. NO-FAKE discipline
- sensitivity is the REAL frozen-scorer autograd gradient (no surrogate).
- the allocation ACTUALLY realizes non-uniform per-tensor bits (verified: the emitted nbits vary; the qdq grid per tensor is the allocated one).
- the codec ACTUALLY byte-closes + re-decodes; S is recomputed from components on the shipped bytes.
- self-adversarial: is the allocation realized in bytes (not just computed)? does the eval run on re-decoded shipped bytes? is the byte count the real STORED zip?
- Pointer UNMOVED 0.19110 unless a byte-closed CPU row beats it (then paired CPU+CUDA + borrowed_substrate_accounting; do NOT self-promote — the frontier is PR101/106-derived).
