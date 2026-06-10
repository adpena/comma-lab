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

## The strategic bar (record in the round-1/2 memos)

skip_high is SOURCE-DERIVED state, so compressing it is a recursive instance of the contest itself —
and the 0.19199 frontier proves the full video fits in 178 KB when learned synthesis replaces stored
state. The stored-LF approach must beat-or-approach that bar; the round-2 table should state the
bytes-vs-178KB ratio per rung so V3 can judge stored-state vs learned-synthesis honestly.
