# OPERATOR-ROUTED DIRECTIVE → snerv_branch_b_rate_attack_20260609 (round-2 redirect)

UTC 2026-06-10 · from main agent, operator verbatim trigger: "aren't there more optimal" (re: the
uniform scalar ladder). Binding for the Branch-B round-2+ plan; the in-flight uniform ladder (R7 +
floor rows) completes as-is — it is the PRICING BASELINE, not the destination.

## The redirect: skip the scalar crawl; jump to the structured coders

`inputs.mfu.skip_high` is 1200 half-res (192×256) lowpass frames — **it IS a video**. Scalar
quantization ignores its two dominant redundancies. Round-2 rungs, in EV order:

1. **S1 temporal-delta**: frame-delta (lat[i]=lat[i-1]+Δ) on the uint8-global plane stack + entropy
   code the residuals. REUSE `src/tac/substrates/snerv_inverse_steg_carrier/temporal_lf_predictor.py`
   (built, unused — orphan inventory). Prediction: dashcam LL temporal redundancy >> spatial;
   expect multiple-× over R5 at ~zero fidelity cost.
2. **S2 spatial transform**: per-plane DWT/DCT + dead-zone + REAL entropy coder (range/ANS with
   context, NOT generic LZMA; the lab has constriction/range-coder surfaces per PR103 L30 + the
   z8 dead-zone work `ad73c2863`). JPEG2000/HEVC-intra recipe.
3. **S3 video-codec-class**: encode the LL sequence as an actual lossy stream (AV1/HEVC-class at
   quality matched to the cone budget; the receiver must decode it inside the 30-min budget —
   numpy-portable or stdlib-decodable constraint applies; if AV1 violates the inflate dependency
   budget, S1+S2 IS the in-house video codec). At ~0.05 bpp → ~370 KB ≈ within 2-3× of competitive
   (vs 62 MB at uint6).
4. **S4 evaluator-conditioned allocation on top**: the cone-masked (spatial) + atlas-clustered
   (temporal) budgets via the #46 waterfiller wiring — bits by H_seg/H_pose + cone radius, never
   pixel variance. Compose with S1-S3, don't replace them.

Guards unchanged: every rung byte-closed + exact re-measure (N>=48, same inflate+scorer path);
waterline test for any fidelity-paying rung; pose is the fragile axis (R6 receipt: pose +0.072 at
uint6 while seg +1.2e-4) so S1-S3 quality knobs tune against the POSE term first.

## S5 — THE FRAME-ROLE AXIS (operator-routed addition, 2026-06-10): per-frame precision within each pair

Operator verbatim: "our frame0 and frame1 and each frame within each pair getting precisely what it
needs within the context of the whole video optimized against the upstream contest auth eval scorer."
The evaluator's frame roles are ASYMMETRIC and the allocation must exploit it:

- **frame0 (even planes, HALF the payload): SegNet-BLIND** (verified exact: d_seg=0 for all frame0
  perturbations, R2 mode-table receipt). Its only constraint is the POSE half-cone → quantize frame0
  LL planes MUCH harder than frame1 (the seg budget is infinite there). This alone should bite on 50%
  of the planes immediately.
- **frame1 (odd planes): the JOINT surface** → the #35 joint cone (seg-margin ∧ pose-Jacobian) sets
  its per-pixel floor; pose binds 73% of pixels.
- **per-pair modulation**: the 600-pair atlas budgets (spend 426-442/577-579 hard; protect 510-522,
  133/177-178) scale BOTH frames' quantizers per pair.
- **per-channel**: pose is luma-dominant + the YUV6 chroma-subsample null space → chroma planes of
  BOTH frames coarsen first; frame0-chroma is the cheapest payload in the entire archive (seg-blind ∧
  chroma-subsampled ∧ pose-weakest).

The unified law per coefficient c at (pair p, frame_role r, channel ch, band k, pixel x):
  bits(c) = waterfill vs the MEASURED response surfaces —
    r=frame0 → H_pose only; r=frame1 → joint cone radius(x);
    × atlas pair-budget(p) × channel sensitivity(ch) × band sensitivity(k);
  admitted iff -ΔS_dist(c) > 25·Δbytes(c)/37,545,489.
Every factor is a MEASURED artifact already on disk (mode table, cone maps, atlas JSONL, spectral
atlas). Compose S5 with S1-S4 (frame-role-aware temporal prediction: frame0 planes may even be
PREDICTED from frame1 neighbors and stored only as pose-correcting residuals). Rung order suggestion:
S5a frame0-hard-quant (cheap, immediate) → S1 temporal → S5b full per-(p,r,ch,k,x) waterfill.

## OPERATOR AMENDMENTS (2026-06-10, binding — supersede the rung-order suggestion above)

**Rung priority order (operator-set):** S5a frame0-hard-quant FIRST (cheap, proven asymmetry: frame0
uint4/uint3/delta vs frame1 uint8-global; d_seg must stay ~unchanged, d_pose prices the exact frame0
floor) → S2 YUV/channel split (Y protected; U/V coarser; frame0-chroma coarsest) → S1 temporal
predictor (SIMPLE first: prev-same-role delta / linear / median / neighbor-frame1; NO heavy motion
estimation until simple deltas underperform) → S3 spatial DCT/DWT + dead-zone per-band → S4 entropy
coder comparison (brotli/zstd/LZMA/RLE+entropy/range-ANS; split streams by section/channel/band) →
S6 video-codec-class LL BENCHMARK (a scale reference, NOT a goal) → S5b full per-(p,r,ch,k,x) waterfill.

**Adversarial correction #1 (frame0 prediction):** PoseNet is a temporal model — predicting frame0
from frame1 neighbors risks destroying the exact motion signal PoseNet reads. The rung is
predict-frame0 + STORE-POSE-CORRECTING-RESIDUAL at varying precision + measure d_pose — NEVER
replace-frame0-with-frame1-like-prediction. frame0 is SegNet-free, NOT pose-free.

**Adversarial correction #2 (the absolute bar):** 0.05bpp/370KB is NOT success — its rate term alone
is 25·370,000/37,545,489 ≈ 0.246 > the 0.19199 frontier before ANY distortion. The reference scale is
PR95's 178,417-byte total archive. Structured LL coding must reach well under ~200KB total OR SNeRV
becomes a base carrier for PR110++ atoms + further pruning — state which, honestly.

**Hard requirements per rung:** (a) report gap_to_0.19199 + live_nonrate + archive_nonrate + rate_term
+ payload_section_bytes; (b) NO new uniform knobs — every rung must vary along a measured axis
(per-role/pair/channel/band/cone), uniform settings are admissible ONLY where a measurement shows the
response surface is uniform along that axis (the theorem: every uniform allocation is an implicit —
and here measured-false — claim of surface uniformity); (c) every rung emits the full V3 row
{schema, base_archive_sha256, payload_section, mutation, bytes_before/after, d_seg, d_pose, score,
delta_score_total, authority_tier, metric_family, first_failed_surface, keep_or_reject} — no
prose-only optimization; (d) atlas-measured values enter ONLY with measurement_scope (no new magic
constants from one sweep).

## S7+ — THE CLEVER TIER (operator 2026-06-10: "aggressively attack rate, get clever in ways we
haven't thought of yet"). Don't just code the blob — QUESTION whether the blob should exist.

- **S7 SYNTHESIZE-don't-store (the mean-field inversion — the big one):** the fleet's "disease"
  (every NeRV renders a blurry low-frequency mean-field at ~21dB) is EXACTLY AN LL-BAND GENERATOR.
  The thing our carriers do for free is the thing we're paying 26-581MB to store. Rung: a tiny
  decoder (the failed carriers' own mean-field, or distilled) synthesizes the LL approximation at
  inflate time; store only sparse pose-correcting residuals where the cone says the synthesis is
  outside budget. The disease becomes the cure.
- **S8 COMPOSE-the-frontier (the 178KB LL generator that already exists):** the 0.19199 archive's
  HNeRV+selector renders the full video evaluator-close for 178,417 bytes TOTAL. At the receiver:
  run it → DWT → take ITS LL as skip_high → SNeRV's MFU/HFR machinery refines on top; store only
  residuals where the frontier render is weak (the cone/atlas knows where). This is vehicle
  composition (#31) arriving early through the rate door — and its bytes bar is "residuals must
  pay rent vs the frontier alone," the honest comparison.
- **S9 EGO-MOTION WARP (the dispersion-plane move):** dashcam LL ≈ smooth gradients + a coherent
  6-dof camera warp. Store sparse LL keyframes (every Nth pair) + the pose trajectory (~600×6
  floats, trivial bytes) + warp; residual-correct the rest. The Maxwell/dispersion basis
  operationalized; also the most PoseNet-aligned representation possible (it literally stores
  what PoseNet measures).
- **S10 PLANE-CODEBOOK (VQ over time):** 1200 LL planes, far fewer distinct scenes. K exemplar
  planes + per-plane index + cheap affine/illumination correction. pact_nerv_vq's genuine VQ
  machinery is the reuse surface.
- **S11 PROGRAM-not-samples (inflate-as-interpreter):** 30 receiver-minutes are an interpreter
  budget. Fit procedural models (sky gradient + road plane + horizon spline over time) per region;
  store coefficients, not samples. The V6 witness-program thesis applied to the LL band.
- **The information-theoretic license for all of S7-S11:** the evaluator never sees skip_high —
  it sees rendered frames through MFU/HFR then the scorers. The required information is bounded by
  the SCORER response entropy (the atlas/cone measure exactly this), not the pixel entropy of the
  planes. Any generator whose output lands inside the same scorer cells is byte-free-equivalent.

Order: S8 first (cheapest to test — the frontier archive is on disk; a receiver-side compose +
N=48 exact re-measure prices it immediately), then S7 (distill-the-mean-field), S9, S10, S11.
Same guards: byte-closed, exact re-measure, V3 rows, runtime inside the 30-min budget.

## The strategic bar (record in the round-1/2 memos)

skip_high is SOURCE-DERIVED state, so compressing it is a recursive instance of the contest itself —
and the 0.19199 frontier proves the full video fits in 178 KB when learned synthesis replaces stored
state. The stored-LF approach must beat-or-approach that bar; the round-2 table should state the
bytes-vs-178KB ratio per rung so V3 can judge stored-state vs learned-synthesis honestly.
