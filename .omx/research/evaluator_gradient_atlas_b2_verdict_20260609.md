# B2 gradient atlas — verdict: the pose carrier is NOT low-rank (resolves the open question)

UTC 2026-06-09 · claude · `tools/hi_nerv_renderer_sanity_ladder.py evaluator-gradient-atlas`.
[macOS-CPU advisory]; autograd through the EXACT torch DistortionNet. Smoke (8 pairs spread stride-12);
full 48-pair-spread confirmation running detached → `.omx/research/evaluator_gradient_atlas_20260609.json`.

## The question (synthesis memo open #1)
"Is the PoseNet-sensitive subspace low-dimensional?" If yes → a compact low-rank pose carrier replaces
the dense neural decoder → floor drops below PR95. If no → the dense decoder is the bulk; the win is
the seg sidecar + Y-focus + entropy coding.

## The answer: NEAR-FULL-RANK (NOT low-rank) — dense decoder necessary for pose
- PoseNet outputs 6 scored dims ⇒ per-pair pose subspace is rank ≤ 6. The GLOBAL question is the rank
  of the 6·N pose-gradient set across pairs.
- **Pose intrinsic dimension (8 pairs SPREAD across the video, 48 gradient vectors): 90%=38, 95%=42,
  99%=47, participation_ratio=35.3** — i.e. ~35-47 effective dims of 48. NEAR-FULL-RANK.
- ⇒ The pose-sensitive directions do NOT recur across driving segments; each segment contributes
  ~independent modes. A compact low-rank pose carrier is NOT viable. **The dense amortized carrier
  (neural decoder) IS necessary for the pose bulk.** Resolves the open question against a cheap carrier.

## STRICT-SCRUTINY note (the rank-4 "breakthrough" was an artifact)
The FIRST smoke used the first 4 ADJACENT pairs (frames 0-7) → intrinsic dim ≈ 4 (participation 3.9) —
a tantalizing low-rank signal. It was an artifact of temporal adjacency (near-identical motion). Fixing
the sampling to SPREAD across the video (stride-12) collapsed the signal: rank jumped 4 → ~38. This is
exactly the "apply strict scrutiny to positive results" discipline catching a false positive before it
became a strategy. Also fixed: float32 overflow in the Gram (PoseNet grads can be >1e154 → float64 +
max-abs pre-scale + sanitize); the near-full-rank result is stable across 5 runs.

## Secondary findings (REAL byte levers, robust)
- **Pose is ~96% LUMINANCE (Y), ~4% chroma.** The pose carrier needs Y/motion fidelity but barely any
  chroma ⇒ a Y-dominant carrier (drop most chroma precision for the pose term) is a real rate saving.
- **Pose uses BOTH frames (~52% frame0 / 48% frame1).** frame0 cannot be dropped for pose (it is
  SegNet-free but pose-critical) — frame0 = a pure pose carrier.
- **SegNet margin VJP:** boundary saliency density ~2.6× interior (5.2% of gradient energy in 2.16% of
  pixels), but spread over the SegNet receptive field ⇒ seg correction atoms are receptive-field-shaped
  around boundaries, not single boundary pixels.

## Design convergence (the whole-arc answer)
The path below 0.192 is NOT a low-rank pose carrier (B2 ruled it out). It is the ATLAS-OPTIMAL
allocation within PR95's structure:
1. **Dense neural decoder, Y-DOMINANT** (pose is 96% luma) — the amortized pose+base carrier (the bulk).
2. **Margin-weighted score-aware training** — focus seg loss on the ~5% fragile; dense Y-focused pose loss.
3. **Sparse ΔS-gated SegNet boundary correction atoms** (receptive-field-shaped around boundaries).
4. **Entropy-coded weights** (PR95 L20-L32) — the binding rate term.
The atlas turns "match PR95" into "beat it by spending exactly where the evaluator looks: Y over chroma,
the 5% fragile over the 95% robust, the decoder bulk where pose is dense."

## NOW-RESOLVED vs still-open
RESOLVED: pose carrier must be dense (no cheap low-rank carrier). Y-dominant. Both frames.
STILL OPEN: (a) does atlas-weighted score-aware training make HiNeRV's d_seg/d_pose DESCEND (corrected
B1 — the actual test)? (b) can a temporally-coded seg correction grammar pay rent after bytes? (c) the
full-48 confirmation of the exact intrinsic-dim number.

## Next
1. Full-48 B2 confirmation (detached) → canonical intrinsic-dim number.
2. **Corrected B1**: atlas-weighted score-aware HiNeRV (RGB-anchor base → margin-weighted seg + dense
   Y-focused pose + eval-roundtrip + EMA + QAT) → exact eval; does it DESCEND vs the flat B1-R2?
3. Seg boundary correction atoms (ΔS-gated) + rate attack (L20-L32).
