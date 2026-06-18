# Yousfi council check-in — the unified margin-saliency seam (2026-06-18)

**Operator: "check in with Yousfi."** Channeled Yousfi (LEAD, built the comma10k SegNet d_seg detector) +
Fridrich, pressure-testing the sub-0.15 strategy from the contest-designer's seat. Adversarially audited by
me (one refinement noted). All `[advisory]`; pointer UNMOVED 0.19110. This is a MAJOR strategy input — the
highest-value design output of the session — captured here so it drives the build round, not lost to chat.

## CONFIRMED (do not relitigate)
- The boundary frame is right; d_seg is the irreducible boundary residual; it must move via the DECODER, not
  a per-pixel RGB sidecar (the 36.9% survival wall is real — you can't impose argmax(SegNet), only RGB).
- Generic UNIWARD correctly falsified: the SegNet is a SEMANTIC detector; its blind spots are
  semantically-unambiguous regions, NOT textured regions. Cost ⊥ margin proven. Bank it.
- The renderer is already a detector-matcher (105/255 RGB error at d_seg 0.0026) — no free blind budget to
  reallocate; it's already exploited.
- margin-hinge is the steganalysis-optimal loss family (gradient aligned with the hard-argmax boundary;
  soft_cosine's 1e-22 vanishing is the textbook worst case). Keep the 50k run.

## THE UNIFYING REDIRECT — ONE map drives all three levers
The SegNet decides on its DOWNSAMPLED, stride-2-stem-pooled grid (preprocess → 512×384 → stem 256×192 → …),
NOT our 384×512 render grid. So compute the detector's own **per-pixel `∂margin/∂input` saliency map on its
512×384 preprocessed grid** and use that ONE map for:
1. **d_seg lever:** weight the margin-hinge by `w(p)=exp(−margin_p/τ)` on the SegNet grid (saliency-weighted
   hinge = first-order; bare hinge = zeroth-order). REPLACE the road↔lane class-emphasis (a coarse proxy that
   HURT) with the real margin-gradient weight. = the detector-informed (Yousfi-Fridrich 2022 ASO/adversarial)
   embedding cost — the detector's OWN gradient, sharper than UNIWARD (wrong domain). We already built this as
   the Z8 P18 saliency (`exp(−margin/τ)`) — tested on the wrong vehicle (wavelet); REUSE it on bc20.
2. **rate lever:** the COMPLEMENT (high-margin + stem-Nyquist-blind, sub-256×192) is certified-zero-d_seg →
   allocate FP-shrink QAT quant-error + detail-coeff dead-zone bytes INTO that blind band, away from
   low-margin pixels. d_seg and rate are COUPLED through the stem frequency response (we wrongly treated them
   independently). **REFINEMENT (my audit):** "certified-free" is certified-zero-d_seg; it MUST ALSO be
   checked against d_pose (PoseNet = different FastViT on YUV6, both frames — may see higher freq). Certify
   against BOTH scorers before shedding bytes.
3. **survival certification:** the same map says which boundary repairs survive the downsample (receptive
   field coherent at 256×192) vs decorrelate (the 36.9%-survival camera-res ones) → converts #137 from a
   gamble to a certifiable allocation. Operate the in-cell repair / margin-hinge on the 512×384 grid (where
   gradients flow), NOT the native 384×512 (the camera-grid cost map decorrelates — why cost-weighting HURT).

## THE DECISIVE $0 MEASUREMENT (Yousfi's #1 next step — gates the long-train thesis)
**The label-noise floor (REDIRECT 3).** Residual flips at margin median 0.137 include the SegNet's OWN label
ambiguity (comma10k labelers disagree on the lane-paint edge — 64% road↔lane). UNWINNABLE by the decoder.
Measure on the GT pairs: fraction of current flips at pixels where the SegNet's OWN GT-frame top-2 margin
< 0.137. **The winnable d_seg target = 0.00260 − (own-ambiguous-flip-mass)**; re-check sub-0.15 feasibility
against THAT, not the raw 0.000322. If a large fraction is label-noise, the long-train is chasing noise below
some floor > 0.000322 → the FP-shrink rate lever becomes even more load-bearing. THIS measurement tells us if
the d_seg half of the sub-0.15 path is well-posed. Firing now ($0, CPU, real SegNet).

## Contest-mechanics (under-used)
- d_pose = MSE on FIRST 6 of 12 pose dims (back 6 thrown away) → the 1-DOF radial-zoom pose codec (#140) is an
  even bigger win than banked; pose only needs 6-dim fidelity.
- rate denominator = fixed full uncompressed corpus → every byte shed off the stem-Nyquist-blind band is pure,
  un-amortized rate. The mechanics reward exactly the seam above.

## Disposition / wire-in
- NEW unifying design task: the margin-saliency map (#141) → drives margin-hinge weight + FP-shrink allocation
  (#136) + #137 survival cert. The Z8 P18 saliency code is the reuse target (do NOT rebuild).
- The $0 label-noise-floor + saliency-map measurement is firing now (gates the long-train d_seg thesis).
- Cross-ref: SESSION_SYNTHESIS_SoT_20260617_20260618.md; the Z8 joint-P18/P19 dead-zone lineage; #136/#137/#140.
