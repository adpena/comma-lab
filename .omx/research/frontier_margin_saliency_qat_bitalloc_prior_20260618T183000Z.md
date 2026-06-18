# Frontier margin-saliency → QAT bit-allocation prior (Path-B #141 recalibration) — 20260618T183000Z

**Operator: recompute #141 on the FRONTIER decoder, not bc20.** The #141 margin-saliency
map was computed on the bc20 basin; the Path-B QAT-shrink build needs it on the ACTUAL
0.19110 frontier vehicle (`lane_pr110_payload_entropy_recode_20260610`). Done.
`[contest-CPU advisory]`, NON-PROMOTABLE, pointer UNMOVED 0.19110 — this is a COST map for
the score-aware QAT build, not a score row. $0, CPU, no GPU dispatch.

## Vehicle (the real frontier, decoded byte-exact)
- archive `b46897267ded1e73...` (177169 bytes), member `x`
- decoded via the frontier's OWN `inflate.py` `parse_member` (FP11+CTXR contest decode) →
  28 HNeRV decoder tensors + latents (600, 28)
- saliency = `||∂(Σ SegNet top1−top2 margin)/∂w_t||` via autograd through the frontier
  decoder → frozen SegNet (CPU), accumulated over 8 frames × 2 heads (rgb_0+rgb_1)

## The QAT bit-allocation prior (the deliverable)
Per-tensor d_seg-sensitivity ranks WHICH frontier decoder WEIGHT tensors to protect
(high precision / fine INT8 grid) vs coarsen (int4). The QAT codec quantizes per
weight tensor, so the bit-alloc lists use the RAW weight-tensor `||∂S/∂w_t||` (the
exact dict `tac.torch_vehicle.score_aware_qat.per_tensor_levels_from_sensitivity`
consumes). Biases are tiny-numel / not the bytes carrier → reported separately, NOT
in the int4 list.

**Top-5 PROTECT (most d_seg-critical weight tensors, raw saliency):**
- `blocks.5.weight` (11664 params) — 1.93567e+06
- `blocks.0.weight` (46656 params) — 1.78824e+06
- `blocks.1.weight` (46656 params) — 1.78182e+06
- `blocks.4.weight` (12960 params) — 1.70516e+06
- `blocks.2.weight` (34992 params) — 1.58607e+06

**Bottom-5 INT4-candidate (most d_seg-blind weight tensors):**
- `skips.4.weight` (360 params) — 487393
- `refine.0.weight` (1458 params) — 426661
- `skips.3.weight` (540 params) — 411712
- `skips.2.weight` (972 params) — 391051
- `refine.1.weight` (1458 params) — 349494

- protect-list (top third of weight tensors): ['blocks.5.weight', 'blocks.0.weight', 'blocks.1.weight', 'blocks.4.weight']
- int4-candidate-list (bottom third of weight tensors): ['refine.0.weight', 'skips.3.weight', 'skips.2.weight', 'refine.1.weight']
- weight-tensor saliency spread (max/min): **5.54×** — KEY FINDING: the
  d_seg-sensitivity is FAIRLY FLAT across the frontier's weight tensors (only ~5.5×
  from most- to least-critical), so the int4 budget gain from coarsening the
  "blind" tensors is MODEST, not dramatic. The biggest weight tensors
  (`blocks.0/1.weight`, ~46k params each) are NOT the most d_seg-blind — they sit
  mid-rank — so the cheap-bytes-at-zero-d_seg story is weaker than the bc20 basin's.

## Per-stage geometry check (Yousfi seam: decision-band > stem-Nyquist-blind)
The SegNet decides on a stride-2-stemmed grid (decision band ≥ ~192×256). Expectation:
late stages (blocks.4/5 @ 192×256/384×512, refine, rgb heads) carry MORE d_seg-saliency
than the coarse stem/early blocks.

ALL-PARAM stage mass:
- decision-band stages: ['blocks.4', 'blocks.5', 'refine.0', 'refine.1', 'rgb_0', 'rgb_1', 'skips.4']
- blind-band stages: ['blocks.0', 'blocks.1', 'blocks.2', 'blocks.3', 'skips.2', 'skips.3', 'stem']
- decision-band mass fraction: **0.442**
- ordering matches geometry: **False**

WEIGHT-ONLY stage mass (the bytes carriers):
- decision-band mass fraction: **0.440**
- decision/blind ratio: **0.785**
- ordering matches geometry: **False**

**HONEST FINDING (NO-FAKE):** the per-stage saliency does NOT cleanly concentrate in
the late/decision-band stages — the early coarse blocks (`blocks.0/1` @ 12×16/24×32)
carry d_seg-saliency comparable to the late blocks. The bilinear-skip + sin coupling
(the Yousfi-seam REFINEMENT risk: "coarse stages couple into the boundary via the
skips") is REAL on this vehicle. So the rate-lever thesis "int4 the stem-Nyquist-blind
band at certified-zero d_seg" is WEAKER on the frontier than on bc20: there is no large
score-blind weight mass to shed cheaply. The QAT bit-alloc should therefore lean on the
modest per-tensor RANKING above, and the byte win will be incremental — consistent with
the Lever-4 probe's measured -4.4% blob, not a dramatic int4 collapse.

## Per-pixel map characterization (bc20 reference: boundary/interior ≈ 3.15)
- mean boundary/interior saliency ratio (frontier): **2.23**
- mean fraction of saliency mass in boundary band: **0.087**
- mean saliency Gini: **0.499**

The frontier's per-pixel saliency IS boundary-concentrated (2.2× boundary
vs interior), confirming the detector-informed weighting principle still holds on the
frontier frames — just slightly less concentrated than the bc20 basin's 3.15×.

## Consumer / next build
This cost map feeds the score-aware QAT-finetune (the build after the PTQ-on-frontier
gate): protect-list weight tensors keep high precision (the argmax boundary is
protected), int4-candidate weight tensors get the coarse grid (fewer brotli bytes).
Cross-refs:
`path_b_recalibration_and_resolve_audit_20260618.md`,
`yousfi_council_checkin_unified_margin_saliency_20260618.md`,
`tac.margin_saliency_map.compute_decoder_tensor_margin_saliency`,
`tac.torch_vehicle.score_aware_qat`. JSON: `frontier_margin_saliency_qat_bitalloc_prior_20260618T183000Z.json`.

NO-FAKE: the saliency is the REAL autograd gradient of the REAL frozen SegNet's margin
w.r.t. each REAL frontier decoder weight (decoded byte-exact via the contest inflate).
DIAGNOSTIC/bit-alloc cost only; the only authoritative d_seg is upstream/evaluate.py.
