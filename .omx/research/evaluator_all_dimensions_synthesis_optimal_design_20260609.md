# All-dimensions evaluator synthesis → the optimal witness design (full analysis)

UTC 2026-06-09 · claude · operator: "analysis + engineering + design after full research vs the FULL
upstream evaluate.py and ALL dimensions (pose, seg, pixels, frames, pairs, all interactions, all
temporal)." This is the synthesis of the measured evaluator atlas into the optimal implementation.
North star: lowest `S = 100·d_seg + √(10·d_pose) + 25·B/N` on 0.mkv (N=37,545,489). All numbers
[macOS-CPU advisory], exact torch scorer.

## PRECISION EDITS (operator 2026-06-09 — these SUPERSEDE the over-claims in the body below)
The body below was written one certainty-level too high. The rigorous statements are:
1. **NOT "neural decoder is necessary."** Say: **"a dense amortized carrier is necessary UNLESS B2
   proves PoseNet has a cheap low-dimensional YUV6/motion carrier."** PoseNet outputs only 6 scored
   dims ⇒ per-pair pose-sensitive subspace is rank ≤ 6; the GLOBAL rank across 600 pairs is unmeasured.
   If low-rank (recurring ego-motion modes) → a compact pose carrier replaces the neural bulk and the
   floor drops below PR95. B2 (the JᵀJ spectrum) decides this. Do NOT treat the decoder as a theorem.
2. **NOT "semantic skeleton loses."** Say: **"naive PER-FRAME SegNet target storage loses (424,722 B,
   rate 0.283); TEMPORALLY-CODED boundary/thin-class grammar remains OPEN and must be priced."** The
   424 KB falsifies per-frame argmax/mask storage, NOT motion-compensated/predictive boundary coding.
3. **NOT "sidecar."** Say **"sparse SegNet correction atoms"** — every row CandidateActionEvaluation-
   gated, measured against the CURRENT base, admitted only if exact ΔS<0. (The old sidecar was harmful
   because it was admitted without paying rent against the right base; the new object is ΔS-gated.)

## NOT YET PROVEN (guard against over-confidence)
- The pose carrier must be neural rather than low-rank YUV6/flow. (B2 JᵀJ spectrum decides.)
- A temporally-coded semantic boundary grammar is too expensive. (needs the temporal codec budget; the
  B0.5 mod-5 bug showed only that per-frame storage loses.)
- HiNeRV specifically is the best dense carrier vs HNeRV/SNeRV/low-rank pose field.
- The sparse SegNet correction atom is rent-positive AFTER bytes (only proven value/byte in principle).
- That B0.5 priced temporal prediction (it priced per-frame geometry only — temporal codec is a TODO).

## One-line answer (read WITH the precision edits above)
The optimal witness is **a score-aware DENSE AMORTIZED CARRIER (pose+base) + SPARSE ΔS-gated SegNet
correction atoms (the seg term) + entropy-coded rate, with training WEIGHTED by the evaluator atlas** —
because seg is sparse/spatial/frame1-only while pose is dense/temporal/both-frames. Whether the dense
carrier is an HNeRV-class neural decoder or a cheaper low-rank YUV6/motion carrier is THE open question
B2 resolves; naive per-frame skeleton storage is ruled out, temporally-coded grammar is open.

## The full evaluator structure (measured, all dimensions)

### Score decomposition at the frontier (0.192)
~62% rate (0.119, 178 KB) + ~38% seg+pose (0.073). RATE-BOUND. Bytes are the dominant lever; seg+pose
must stay small while bytes shrink.

### d_seg — SPARSE, SPATIAL, frame1-only (segnet_margin_field.v2, all 600)
- SegNet scores ONLY frame1 (last of pair), argmax of 5 classes, no normalization, [0,255].
- ~4.8% of pixels fragile (margin m<2 logit); ~95% robust (m 4-8). Per-frame p10/p50/p90 = 4.2/4.8/5.5%.
- Class-structured: class 2 (road, 49.5%) is 98.4% ROBUST; class 1 (0.6%, thin) is 93.5% FRAGILE;
  boundaries (2.16% of px) are 99.99% fragile and = 45% of all fragile px.
- Skeleton works: keep the 5% fragile boundaries sharp + cheapen interior 32× → d_seg 0.368→0.009
  (40× better). The seg INFORMATION is sparse + boundary-concentrated.

### d_pose — DENSE, TEMPORAL, both-frames (evaluator_cell_tolerance + region_cheapen_seg_vs_pose)
- PoseNet scores BOTH frames via RGB→YUV6 + mean/std, first 6 of 12 dims, MSE.
- DEMANDING: even k=2 downsample → d_pose ~0.001 (30× the frontier 3.4e-5); interior cheapening
  destroys pose (k8=0.46, k16=8.2, k32=45) EVEN with the seg boundary skeleton preserved.
- The seg skeleton does NOT help pose (boundary preservation barely moved d_pose). Pose needs DENSE,
  near-full-resolution texture (the optical-flow/motion cues), across BOTH frames.
- The neural decoder achieves d_pose=3.4e-5 — far below any downsampling. The decoder is EXCELLENT at pose.

### Rate — the binding term (segnet_fragile_support_codec_budget, all 600)
- Naive seg-target storage: full argmax brotli = 424,722 B → rate 0.283 — WORSE than the whole frontier.
- Fragile/boundary/thin-class mask geometry = 258-536 KB (value/byte 6-13 in principle, but absolute
  bytes large). Per-frame brotli masks DO NOT beat the decoder's amortization.
- ⇒ The naive evaluator-inverse "store the segmentation/skeleton" loses on rate. Amortization (a learned
  decoder shared across 600 pairs) is the only way to make the dense carrier cheap.

### The renderer diagnosis (closed)
HiNeRV B1 (distillation-only, no RGB anchor) collapsed to 2 fixed frames (d_seg=0.50). One-pair RGB
overfit plateaus at 21 dB — below the SegNet cell threshold AND far below pose fidelity. The renderer
trained the WRONG objective (uniform RGB) on the WRONG manifold. Not a paradigm failure — an objective
+ weighting failure.

## The interactions (all levels, all temporal) — and what each implies
- **seg ⟂ pose (orthogonal structure):** seg wants sparse boundary fidelity on frame1; pose wants dense
  temporal fidelity on both frames. They need DIFFERENT mechanisms → decoder (pose) + sidecar (seg).
  They do not compete for the same bytes.
- **frame0 vs frame1:** frame0 is SegNet-FREE (pose-only); frame1 drives both. ⇒ frame0 = pure pose
  carrier (can drop seg fidelity, must keep pose/motion fidelity); frame1 = pose carrier + seg boundaries.
- **pair × pair (temporal):** consecutive pairs are temporally coherent → the decoder amortizes the
  shared structure; per-pair latents encode the variation (28-d, PR95 L19); the seg boundaries move
  slowly → a TEMPORALLY-CODED sidecar (motion-compensated), not per-frame masks.
- **pixel level:** the margin field (per-pixel seg fragility) + the pose Jacobian JᵀJ (per-pixel pose
  sensitivity — B2, pending) give the per-pixel byte-value field for the waterfiller.

## The DERIVED optimal architecture (matches PR95 — derived, not copied)
1. **Neural decoder = the dense pose+base carrier.** HNeRV-class: per-pair 28-d latents → full-res RGB.
   Produces dense full-res frames (nails pose) cheaply via amortization (one decoder, 600 pairs). This is
   where most bytes go (decoder weights ~150 KB after entropy coding). PR95 L18/L19.
2. **Sparse SegNet boundary sidecar = the seg term.** Per-pair corrections at the ~5% fragile boundary
   pixels where the decoder's frame1 gets the argmax wrong. Sparse, byte-cheap, temporally-coded
   (boundaries move slowly). PR95 L27 (per-pair single-dim correction sidecar, −0.001..−0.003 alone).
3. **Entropy-coded weights = the rate term.** PR95 L20-L32: per-tensor byte-maps, split brotli, raw-LZMA
   latents, temporal-delta uint8, range coding, brotli q11. Compress the decoder (the bulk).
4. **Atlas-WEIGHTED score-aware training:** seg loss FOCUSED on the 5% fragile (margin-weighted, not
   uniform RGB); pose loss DENSE (full-res, both frames, the demanding term); + an RGB-anchor base to
   escape the wrong manifold; + eval-roundtrip + EMA. The atlas margin field + pose Jacobian set the
   per-pixel/per-term loss weights — turning "uniform RGB" (which failed) into "evaluator-weighted."
5. **Waterfiller admits sidecar atoms by exact ΔS** (`tac.optimization.evaluator_action_waterfill`):
   each boundary/thin-class correction atom admitted iff 100·Δd_seg + Δ√pose + 25·Δbytes/N < 0.

## Why this is the answer (the reconciliation)
- The "evaluator-equivalent witness compiler" insight is REAL but applies to the SEG term (sparse → cheap
  sidecar). It does NOT apply to the POSE term (dense → needs the amortizing decoder). The naive skeleton
  (425 KB seg target) loses precisely because it can't amortize and can't carry pose.
- The neural decoder is NECESSARY (the only cheap dense pose carrier). The HiNeRV failure was the
  objective+weighting, not the vehicle. Fix: atlas-weighted score-aware training + RGB-anchor base.
- This is PR95's exact winning structure, now DERIVED from the measured evaluator geometry. The atlas
  tells us the WEIGHTS (where fidelity/bytes go) that PR95 found empirically — so we can match or beat it
  by allocating optimally instead of guessing.

## Engineering plan (ordered; each a typed artifact + exact eval)
1. **B2 gradient atlas (next):** SegNet margin VJP + PoseNet JᵀJ saliency → the per-pixel pose-sensitive
   subspace. KEY open question: is the pose-sensitive subspace LOW-DIM? If yes, the pose carrier can be a
   low-rank dense field (cheaper decoder). If no, the decoder must be near-full-rank (the bulk stays).
2. **Corrected B1 (the decoder):** atlas-weighted score-aware HiNeRV — RGB-anchor base (escape the wrong
   manifold) → margin-weighted seg loss (focus the 5%) + dense pose loss + eval-roundtrip + EMA + QAT.
   Validate d_seg/d_pose DESCEND (vs the flat B1-R2) on exact eval.
3. **Seg boundary sidecar (PR95 L27):** per-pair fragile-boundary corrections; waterfiller-admitted.
4. **Rate attack (PR95 L20-L32):** entropy-code the decoder; this is the binding term.
5. **Compose + dual-axis exact eval + submit iff < frontier.**

## Honest open questions (the atlas must still answer)
- **Is the pose-sensitive subspace low-dim?** (B2 JᵀJ). Determines whether the pose carrier (the bulk) can
  be made much smaller than HNeRV's decoder, or whether HNeRV-class is near-optimal for pose.
- **Can a temporally-coded seg sidecar (motion-compensated boundaries) beat per-frame storage?** (the
  fix to the mod-5 codec; B0.5 showed per-frame masks lose, temporal coherence is the lever).
- **Does atlas-weighted training actually descend** where uniform-RGB B1 was flat? (corrected B1 exact eval).

## Bottom line
The all-dimensions atlas resolves the campaign's central question: the lowest-score witness is NOT a
hand-built skeleton grammar (it loses on rate + can't carry pose) and NOT uniform RGB reconstruction (it
wastes 95% of effort + misses the 5% that matter). It is a **score-aware neural decoder (dense pose
carrier, amortized) + sparse atlas-weighted seg sidecar + entropy-coded rate** — PR95's structure,
derived from the evaluator geometry, with the atlas providing the optimal byte/fidelity allocation that
turns "match PR95" into "beat it by allocating where the evaluator actually looks."
