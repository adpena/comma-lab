# Optimal pose-carrier — deep-math grounding (saliency-confined, Jacobian-rank, round-trip-threaded, Wyner-Ziv) + the #57 defects to fix (2026-06-21)

**Operator (2026-06-21):** "Ensure deep math grounding / And optimal" on the witness pose-closure build (action 2).
This grounds the rebuild. $0 theory + module audit. Authority: `[contest-CPU advisory]`, NON-PROMOTABLE, pointer
UNMOVED 0.19110. NO score claim.

---

## 1. The deep-math the OPTIMAL pose-carrier must obey
The contest d_pose = MSE on the 6 scored PoseNet dims. Let `J = ∂pose6/∂x` be the PoseNet input-Jacobian
(x = the frame the scorer ingests, after resize→YUV6). Two exact facts:

1. **Pose is read through a LOW-RANK, SPARSE channel.** The 6 scored dims depend on x only through `J` (rank ≤ 6
   locally), and the per-pixel saliency `w(x,y) = ‖∂pose6/∂pixel(x,y)‖` is concentrated (high on pose-relevant
   structure, ≈0 on the pose-null — measured in `posenet_jacobian_saliency.py`, #61). **The carrier must paint
   luma where `w` is high and is FREE to be arbitrary where `w≈0` (the pose-null).**
2. **Format/round-trip confinement (operator's round-trip point, threaded).** PoseNet preprocess: resize 384×512
   bilinear → `rgb_to_yuv6` = 4 HALF-res luma + 2 (2×2-box) chroma. So pose reads **half-res LUMA** dominantly;
   chroma is subsampled (carries little). The carrier lives in the **YUV6-luma subspace at half-res**, and its
   luma must SURVIVE the uint8 + resize round-trip to land the pose at the scorer.

**→ The optimal carrier = a saliency-CONFINED, half-res-luma, round-trip-survived appearance code, amortized
across the 600 pairs (shared basis + per-pair low-dim modulation ≈ the 6-dim pose), added Wyner-Ziv on top of
the seg-witness frame (which is the decoder's side-info).** The byte floor = pose entropy (~1.5 KB, 600×6
delta-coded) + the shared saliency-confined basis (amortized, near-free/pair) + survival overhead.

## 2. Why #57 as-is is SUB-OPTIMAL (the dense-anchor ceiling)
`amortized_luma_carrier.py` is a FREE full-frame Fourier-INR conditioned on (pair_idx,x,y) — grep shows NO
jacobian / saliency / confine / rank. It spends capacity reproducing the FULL frame, including the pose-null.
That is the EXACT documented cause of the **d_pose 0.0036 INR ceiling** (`posenet_jacobian_saliency.py`
docstring: "the dense anchor spends capacity reproducing pose-IRRELEVANT luma — the source of the d_pose 0.0036
ceiling"). The fix #61 (PTNC) built — the **PoseNet-Jacobian-saliency-WEIGHTED recon anchor** — exists as a
module but is NOT wired into #57. So the optimal build = **#57's amortized-INR architecture × #61's measured
saliency-weighted anchor** (paint only the pose-relevant support; free elsewhere).

## 3. The #57 PARITY/OVERFLOW DEFECT (must fix in the rebuild — NO-FAKE)
The action-2 probe (killed) surfaced: **#57's numpy forward overflows** (matmul OVERFLOW warnings → NaN/inf;
frame0 sha does NOT reproduce the manifest, though frame0 var=821 so texture IS produced). The manifest's
claimed `all_match=True` was almost certainly **parse-back vs direct-forward — BOTH carrying the same overflow**
→ they match each other but neither equals a CLEAN float64 reference. This is a Catalog #304/#307 parity-proof
defect: the parity measured the wrong invariant. **The rebuild MUST: (a) float64 accumulation (or proper int8
dequant range) so the forward doesn't overflow; (b) a CLEAN parity proof — clean-numpy-forward vs the torch
oracle, not parse-back vs same-overflow-forward.** (Same float32-overflow class seen in the latent-dedup
measurement — these INRs have large intermediates; force float64.)

## 4. The optimal build spec (the rebuild)
Compose, with deep-math grounding:
1. **Saliency map**: compute the measured PoseNet-Jacobian saliency `w(x,y)` per pair (frozen CPU PoseNet, #61,
   the atlas-Jacobian-norm at the GT operating point) — the pose-relevant support.
2. **Carrier**: #57's amortized coordinate-INR luma, but with the loss/anchor WEIGHTED by `w` (paint pose-relevant,
   free pose-null) → confines capacity to the low-rank pose channel → fewer bytes + breaks the dense-anchor ceiling.
3. **Round-trip-threaded**: fit + measure through the YUV6 half-res-luma + uint8 + resize round-trip (the carrier
   luma must survive to land the pose at the scorer).
4. **Wyner-Ziv**: the L13 seg-witness palette frame is the side-info; the carrier adds the pose-relevant luma on top.
5. **Clean float64 forward + clean parity** (fix the #57 overflow); REAL byte-close (Catalog #304 — brotli of the
   actual quantized carrier blob, packed + read by inflate).
**Measure:** d_pose AFTER (vs palette 12.658) + real carrier bytes + advisory S decomposition (N≤48, CPU OMP=2).
Honest expectation unchanged: S not a winner (L13 seg-term 0.68 dominates); the job is the pose-wall closure at
**lower** byte cost than the free INR, via the saliency confinement.

## NO-FAKE ledger
- DERIVED: pose is rank-≤6 + sparse-saliency-confined + half-res-luma + round-trip-survived → the optimal carrier
  confines to `w`-high support; the free INR over-spends + ceilings (per #61's measured anchor result).
- MEASURED (this turn): #57 has NO saliency/jacobian confinement (grep); #57 numpy forward overflows + frame0
  parity does not reproduce clean (action-2 probe before kill).
- NOT claimed: no score moved; pointer UNMOVED 0.19110; the optimal carrier is a SPEC to build+measure, not a result.

## Cross-references
- `CAPSTONE_witness_taskspace_roundtrip_byte_floor_formulation_20260621.md` (§3/§6/§8/§9 — the witness; this is its pose mechanism done right).
- `src/tac/boundary_math/posenet_jacobian_saliency.py` (#61 PTNC, the saliency anchor) + `amortized_luma_carrier.py` (#57, the INR to confine + de-overflow).
- `witness_L13_pose_film_integration_20260621.md` (gap #3 re-route: pose-FiLM HNeRV-bound, amortized-luma is the palette-witness mechanism).
- `dseg_boundary_hessian_conditioning_20260621.md` (the sister conditioning math; same Jacobian-saliency lens on the seg side).
