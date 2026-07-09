# Papers-checked: STAC (arXiv 2203.14481) + SPARC (arXiv 2606.16253) — task-aware compression prior art

Date: 2026-07-09 · operator-supplied links · anti-re-research ledger entry (sister of
`papers_checked_ttd_functional_tensor_train_20260708`). STORES CONSULTED: MEMORY.md L55
papers-checked line · #157 (KKT reverse-waterfill, completed) · #336 (witness-checkpoint
bit-allocation, pending) · #141 margin-saliency · annulus law (#333, ~97% d_seg in ~4.7% annulus)
· V2 originality memo (L16) · rare-class levers (σ_cc′ #382, focal/logit-adjust, island-birth).

## 1. STAC — "DNN-Driven Compressive Offloading for Edge-Assisted Semantic Video Segmentation"
Xiao, Zhang, Wang, He, Zhang — INFOCOM 2022. Regime: LIVE camera→edge TRANSPORT codec
(bandwidth), JPEG/H.264-family DCT quantization, semantic-seg DNN consumer. ~20.95% bandwidth
saved at iso-accuracy.

**Method (verified from PDF §III):** sensitivity g_x = ∂Q/∂x_i (LOSS gradient w.r.t. pixels),
mapped to DCT space g_s; first-order ΔQ = Σ g_s·Δs. Allocation: min Σ log2|s/q| s.t.
Σ|g_s|·q/2 ≤ B ⇒ optimal at EQUAL marginal loss d_s = B/M ⇒ q_s = 2B/(M|g_s|) — an exact
reverse-waterfill under a loss-increment budget. Offline L quantization-table levels; online
per-region table pick (worst-case ΔQ closest to B/r_max); dense-optical-flow (DIS) propagation
of strategy+seg across frames; adaptive keyframe offload. "Fake Q" trick: gradient computed
against the DNN's OWN output as pseudo-label (no GT needed; better matches compressed-frame
loss slope).

**Verdict — CONFIRMS, not a lever** (verdict_scope: FORMULATION — their transport/DCT regime;
no reformulation owed, nothing of ours killed):
- Equal-marginal waterfill allocation = independent 2022 validation of **#157**'s KKT design
  (ours: exact-sensitivity, frozen-scorer, archive-bytes regime — deeper on both axes).
- Their Fig.3: "most sensitive regions are often NOT the boundaries" — TRUE for a LOSS-gradient
  functional (saturates on confident pixels, peaks on ambiguous interiors) and exactly why our
  functional is MARGIN/flip-distance through argmax (annulus law: ~97% of d_seg in the ~4.7%
  boundary annulus). Sharpens the writeup's margin-vs-loss-gradient distinction; citable contrast.
- Optical-flow strategy propagation ⊂ our se(3) ego-screw transport (6-dim vs dense; chart-
  selection law governs where it pays). Keyframe adaptation ⊂ v8 temporal amortization.
- "Fake Q" self-label gradient ≈ our scorer-as-oracle target (L* = frozen SegNet argmax). Covered.
- Originality: no argmax-partition geometry, no task-space witness, rate not co-optimized against
  a frozen scorer's exact bytes. V2 claim untouched; belongs in VCM related-work lineage.

## 2. SPARC — "Learned Image Compression for Vision-Language-Action Models"
Kim, Ryu, Ha, Lee, Kim, Ahn, Lee — arXiv 2606.16253. Regime: learned codec for VLA robot
control (multi-camera transport); RoboCasa365/VLABench/LIBERO; beats standard + learned codecs
at iso-bitrate on task success.

**Method:** (1) lightweight temporal mask selector allocates bitrate across camera views +
spatial regions by task relevance; (2) **tilted rate loss** — modified entropy objective that
prevents OVER-SUPPRESSION OF RARE-BUT-TASK-CRITICAL visual patterns during R-D training.

**Verdict — CONFIRMS + ONE design grain** (verdict_scope: FORMULATION — VLA transport regime):
- "Rare-but-task-critical patterns over-suppressed by the rate objective" is OUR lane-erasure
  phenomenon (MCF/rate pressure erases low-persistence features; Lane = 0.59% area) named
  independently in the rate domain. Their fix is rate-side (tilted entropy); ours are
  distortion-side (σ_cc′ per-class tension #382, focal/logit-adjust, island-birth homotopy).
- **THE GRAIN → #336 (witness-checkpoint sensitivity bit-allocation):** an AGGREGATE-loss
  equal-marginal allocation starves the weights supporting rare classes by construction (Lane's
  aggregate gradient mass ~0.59%-scale) — the exact mechanism SPARC's tilted loss patches. When
  #336 fires, the sensitivity functional MUST be per-class-weighted / margin-aware (flip-distance
  or per-class d_seg share), NOT aggregate d_seg-gradient. Folded into #336's task description.
- Originality: no partition geometry, learned-codec transport regime. V2 untouched; VCM
  related-work citation (nearest-in-time sibling problem: frozen policy vs frozen scorer).

## Disposition
- Ledger: MEMORY.md L55 papers-checked line updated (STAC+SPARC entry → this memo).
- #336 grain recorded above (task note); no new lever, no equation (no measured row of ours —
  prior-art confirmation only; the annulus/margin contrast is already registered).
- Writeup/related-work: cite both in the VCM lineage section (task-aware allocation ancestry:
  GRACE→STAC transport seg; SPARC VLA; ours = archive-codec against frozen evaluator with exact
  rate term + partition geometry).
- means≠ends: pointer 0.19110 UNMOVED; this is anti-re-research banking only.
