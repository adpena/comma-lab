# UNTAPPED-TECHNIQUE INVENTORY — adversarial breadth-first hunt (2026-06-10)

**Subagent:** `untapped_technique_hunt_20260610` (READ-ONLY ideation + grounding sweep; this memo is the only artifact).
**Evidence grade:** `[macOS-CPU advisory]` / mechanism-only. NO score claims, NO dispatch, `promotable=false`.
**Operator question (2026-06-10):** *"what other tricks and techniques aren't we thinking about?"*
**Method:** read `upstream/{evaluate.py,modules.py,frame_utils.py}` IN FULL + the 2026-06-09/10 corpus, then
grepped the repo for each candidate technique to separate GENUINELY-UNTRIED from already-have / killed-for-a-reason.

**Frontier at audit (pointer, never hardcoded — `tools/refresh_canonical_frontier.py`):**
contest-CPU **0.19198275** (178,495 B) / contest-CUDA **0.20533003** (186,876 B).
Score law (frozen authority): `S = 100·d_seg + √(10·d_pose) + 25·B/N`, N=37,545,489. Byte price ≈ **6.66e-7 score/byte**.
**The public test set is a SINGLE video (`0.mkv`, 1200 frames → 600 non-overlapping pairs).** This is load-bearing
for several gaps below.

---

## 0. THE HONEST FRAMING — WHY THE GAPS ARE SUBTLE

The lab is **exhaustively deep**. The 2026-06-09/10 wave mapped every frozen-byte axis to a Pareto vertex, and the
codebase already contains: the resize null space (certified, `evaluator_invisibility_basis`), the 2×2 chroma null
(`yuv6_chroma_subsampled_perturbation_operator` + `constrained_gen.py:1749`), the argmax-margin map, frame0-seg-free,
the pose-null projection (`scorer_read_surface_atoms`), gradient-directed dithering with Floyd-Steinberg
(`quantization.noise_shaped_round`, WIRED into `trick_stack.py`), the full steganography lineage as composition
operators (UNIWARD/HILL/HUGO/MiPOD/Fridrich/Alaska), inter-frame affine warp (`renderer.py:1096`, `nscs06_v8`),
PR#112's lossless entropy recode (R1/R2/R3 harvest queue), and cross-pair pose fungibility (E3).

So **most "obvious" tricks are already-have or correctly-killed.** The genuinely-untried techniques cluster in three
places the exhaustion map did NOT close: **(A) cross-PAIR redundancy** (the single-video structure — 600 pairs of one
drive — is highly self-similar and the frontier codes each pair independently); **(B) the seg-free frame0 as a
cheap-to-CODE synthetic** (we perturb frame0 for free, but never asked "what is the CHEAPEST frame0 that satisfies
pose?"); **(C) the inflate-as-interpreter rate subsidy** (E1/R4 — recognized, never actioned on the frontier).
Everything below is ranked, line-cited, and carries a falsifiable first test. The exact paired CPU+CUDA eval is the
only authority for any ΔS.

---

## 1. THE RANKED INVENTORY

Ranking key: **value × readiness** = `|predicted ΔS lower bound (derived)| / (build-cost × risk)`, hard-gated on
falsifiability. **R** = ready, **SB** = small-build, **RES** = research/campaign. Every row: mechanism + line cite,
predicted ΔS axis/band + derivation, tried? (verdict-cited if killed), readiness, reuse targets, falsifiable first test.

---

### T1 — CROSS-PAIR LATENT DEDUP / CLUSTERING (the single-video redundancy) ⭐ TOP — RATE — SB
- **What:** The frontier stores **600 independent 28-d per-pair latents** (15,387 B, 8.6% of archive). `0.mkv` is ONE
  drive: long stretches are near-stationary (stopped at a light, straight highway at constant speed). Many of the 600
  latents are therefore near-duplicates. Replace the per-pair LZMA stream with a **clustered codebook + per-pair index
  + small residual**: K representative latents (the dictionary) + 600×⌈log2 K⌉-bit indices + sparse per-pair deltas.
- **Mechanism it exploits:** `frame_utils.py:138` `num_sequences = frames_per_file // seq_len` → 600 pairs of a SINGLE
  contiguous video. The latent section is the only payload whose entropy is dominated by INTER-PAIR redundancy (the
  decoder weights are shared across pairs already; the latents are the per-pair degrees of freedom). PR#112's AR(1) +
  cross-DIM LS (`public_pr112_frontier_beat_intake_20260610.md` §3) exploited per-pair temporal AND cross-dimension
  structure but **NOT cross-PAIR clustering** — AR(1) is a local predictor, not a global dictionary.
- **Predicted ΔS:** RATE only, derivation: latent section 15,387 B at LZMA ≈ 7.0 bits/code (latent verdict). If the 600
  pairs cluster into K≈64–128 distinct latents (plausible for one drive), the index cost is 600×7 bits ≈ 525 B + K×28×1 B
  dictionary ≈ 1,800–3,600 B + residuals. Even a conservative 30% reduction of the latent section = **−4,600 B → −0.0031**;
  an aggressive 60% (high self-similarity) = **−9,200 B → −0.0061**. This is LARGER than PR#112's entire −1,381 B win and
  is LOSSLESS-or-near (residual-coded), so d_seg/d_pose ≈ unchanged. **Compounds additively with R1/R2/R3/S12** (different
  exploited structure — global vs local).
- **Tried?** **NO.** `grep latent_dedup|cluster_latent|shared_latent_across` → EMPTY in `src/tac/`. The latent-axis
  verdict (`frontier_latent_axis_waterfill_verdict_20260610.md`) tested LZMA retune / coder swap / 2nd-order delta
  re-prediction (all FALSIFIED) — NONE is cross-pair clustering. The verdict's "LZMA at 3.4% of iid floor" is the
  per-pair iid floor; it does NOT account for inter-pair mutual information.
- **Readiness: SB.** Build a k-means / agglomerative cluster over the 600 decoded latents + a dictionary-index codec
  (varint index + discrete-Gaussian residual à la PR#112's Q_TABLE) + re-pack via `pr101_split_brotli_codec` grammar +
  byte-close + ONE paired CPU+CUDA replay (~$0.3). ~60–90 LOC. The frontier latents are obtained via
  `decode_latents_compact` (LZMA→raw codes) — the exact inverse PR#112 used.
- **Reuse targets:** `tac.pr101_split_brotli_codec.decode_latents_compact`, `tac.pr103_arithmetic_codec`,
  `tac.shared_pmf_model`, `constriction.RangeEncoder`, the latent-axis verdict's per-dim sensitivity ranking, R2's
  discrete-Gaussian Q_TABLE (build R2 first, share the residual coder).
- **FALSIFIABLE FIRST TEST ($0, local):** decode the 600 frontier latents, compute the pairwise L2 distance matrix +
  run k-means at K∈{32,64,128,256}. Measure `bytes(dictionary + indices + residual-at-tolerance)` vs the current 15,387 B
  at the latent quantization step the frontier uses. **KILL if** the clustered representation is NOT smaller than 15,387 B
  at residual-tolerance ≤ the frontier's own latent quantization grain (i.e. the 600 latents are genuinely high-entropy /
  the drive is not self-similar) → then T1 is FALSIFIED and the per-pair iid floor IS the floor.

---

### T2 — THE CHEAPEST-frame0 SYNTHESIS (frame0 is seg-free; minimize its CODE cost, not just perturb it) ⭐ — RATE — RES→SB
- **What:** SegNet reads only `x[:,-1,...]` = frame1 (`modules.py:108`). **Frame0 carries ZERO d_seg signal** — it only
  must satisfy PoseNet's first-6-dim pose on the (frame0,frame1) pair. We have extensively mapped how to PERTURB frame0
  for free (`scorer_read_surface_atoms` pose-null, the 16-mode FEC6 selector). The UNASKED question: **what is the
  CHEAPEST-TO-CODE frame0 that holds pose?** If frame0 can be a low-entropy synthetic (e.g. frame0 ≈ a warped/blurred
  function of frame1, or a coarse base + sparse pose-critical detail) instead of a full second decoded frame, the carrier
  for 600 frame0s shrinks.
- **Mechanism it exploits:** `modules.py:108` (SegNet last-frame-only) ⊕ `modules.py:84` (pose = first-6-dim MSE on the
  2-frame YUV6 input). PoseNet's pose signal is INTER-frame and low-band (B2 atlas: Y-luma 0.964, w_equiv≈294). So
  frame0 needs only to encode the LOW-BAND LUMA inter-frame delta that moves the 6 pose dims — a tiny fraction of a full
  frame's entropy. The frontier wastes bytes decoding a full-fidelity frame0 the scorer never seg-reads.
- **Predicted ΔS:** RATE. Derivation is INDIRECT (it changes what the decoder must represent, so it's a training/arch
  lever, not a frozen-byte transform on the EXISTING decoder). If frame0 is regenerated as `warp(frame1, pose) + sparse
  pose-residual` instead of a full second-frame decode, the per-pair representable content roughly HALVES on the frame0
  axis. On a retrained carrier this could shift 5–15% of decoder capacity. Band (campaign-level, speculative):
  **−0.01 to −0.03** if it lets a smaller decoder hold the same pose at fewer bytes. On the FROZEN frontier it is NOT
  directly actionable (the decoder already jointly produces both frames).
- **Tried?** **NO as a rate-minimization objective.** `grep frame0.*cheap.*pose|frame0.*synthetic.*pose|frame0.*minimal`
  → EMPTY. We have frame0-PERTURBATION (free) and inter-frame WARP (`renderer.py:1096`, `nscs06_v8` affine warp) but
  never the JOINT "frame0 = cheapest function that satisfies pose, seg-free by construction" as the carrier-design
  objective. The `evaluator_optimal_adaptive_waterfilling…20260609.md:114` line NAMES the asymmetry ("frame0 is a
  pose-only carrier") but stops at perturbation budget, not code-cost minimization.
- **Readiness: RES (frozen) → SB (as a retraining-campaign aiming term).** This is properly an aiming term for RANK 1
  (the AFSR-1 score-aware retraining campaign): add a loss/arch that REGENERATES frame0 from frame1+pose (warp-residual
  head) so the decoder spends ~0 bytes on a redundant second frame. The `nscs06_v8` affine-warp inflate code is the
  reusable warp primitive.
- **Reuse targets:** `tac.renderer` (frame1=warp(frame2,flow)+gated residual, line 1088), `nscs06_v8_chroma_lut.inflate
  ._affine_warp_frame1_from_frame0`, `scorer_read_surface_atoms.pose_null_projection`, B2 atlas (Y-fraction), the AFSR-1
  campaign loss.
- **FALSIFIABLE FIRST TEST ($0, local):** for the 600 frontier pairs, compute pose(frame0,frame1) vs pose(warp(frame1,
  est_pose), frame1) — i.e. how much pose error does a pure-warp frame0 (zero stored frame0 bytes) incur? Measure the
  d_pose if frame0 is replaced by `affine_warp(frame1, pose)`. **KILL if** the warp-only frame0 incurs d_pose ≫ the
  frontier's 2.9e-5 by more than the rate it would save (the pose signal needs genuine independent frame0 content) →
  then frame0 is HARD-EARNED-full-frame and the joint decode is correct.

---

### T3 — INFLATE-AS-INTERPRETER: migrate constant payload sections into rate-free CODE (E1/R4) — RATE — SB+JUDGMENT
- **What:** `evaluate.py:63` counts `archive.zip` ONLY. `inflate.py`/`inflate.sh` bytes are NOT in the rate term.
  Any payload section that is a CONSTANT / TABLE / PROCEDURAL GENERATOR can be moved into inflate.py CODE at ZERO rate
  cost, shipping only a seed/index in archive.zip. Precedent: PR110's mode catalog + Huffman codebook ALREADY live in code
  (maintainer-accepted); the demoscene `.kkrieger` hoist pattern (bake the generator, ship the seed) is recognized in
  `grand_council_symposium_inflate_py_extreme_compression_20260518.md:64`.
- **Mechanism it exploits:** `evaluate.py:63` `compressed_size = (submission_dir/'archive.zip').stat().st_size` — the
  interpreter is free real estate. The frontier's sidecar (607 B), selector codebook, and framing constants are
  candidates; a procedural generator for any structured table is rate-free.
- **Predicted ΔS:** RATE. Bounded by which sections are compliance-defensible-as-code. The BIG sections (decoder weights
  162 KB, latents 15 KB) are genuine HIGH-ENTROPY payload — NOT movable to code (you can't bake the memorized video in
  inflate.py and call it compression; review norms bound this). The defensible candidates are the L27 sidecar (607 B,
  "< 7 B from optimal" so the entropy-coded form is already near-floor — but if its STRUCTURE is procedural, the bytes
  could be regenerated) + framing constants (7–10 B) + any deterministic table. Honest band: **−0.0004 to −0.0010** (the
  small sections) — LOW-EV on the current frontier, but ZERO-cost and a STANDING SUBSIDY for any future witness-program
  (V6) carrier where more of the representation is procedural.
- **Tried?** **PARTIALLY-recognized, NEVER actioned.** `evaluate_py_fresh_eurekas_20260610.md` E1 (committed
  `1264a4405`) + `orphan_harvest_recovery_ledger_20260610.md` R4 flag it as a NAMED follow-up that was never executed
  (`grep migrate.*section.*code|table.*in.inflate` → only design-memo mentions, no frontier action). It is genuinely
  un-built on the current frontier.
- **Readiness: SB + JUDGMENT.** Not a missing primitive — a compliance audit ("largest defensible inflate.py") + a
  small refactor that emits the procedural section from code with a seed in archive.zip. The judgment call (defensibility)
  is the gate, not LOC.
- **Reuse targets:** PR110 inflate.py mode-catalog pattern (precedent), `wyner_ziv_residual_encoder` (Y = shared prior
  computable by decoder from baked-in inflate.py), the AFSR-1 export placement logic.
- **FALSIFIABLE FIRST TEST ($0):** audit the frontier archive's 4 sections; for each, ask "is this section a
  deterministic function of a smaller seed + code?" Build a 1-section proof-of-concept (e.g. regenerate the L27 sidecar
  structure from a generator) and measure the archive byte delta + run inflate to confirm byte-identical decode. **KILL
  the section if** moving it to code is NOT compliance-defensible (it embeds video-derived content) OR the seed+code
  saves < the section's current entropy-coded size.

---

### T4 — SELECTOR AS AN EXPLICIT MARKOV / RLE STREAM (temporally-correlated mode IDs) — RATE — SB
- **What:** The FEC6 selector is 600 per-pair mode IDs (the frame0-perturbation menu choice). On a single contiguous
  drive these mode IDs are TEMPORALLY CORRELATED (consecutive pairs pick similar modes). PR#110 used FIXED Huffman;
  PR#112 used adaptive order-0 16-ary AC (already at order-0 entropy). NEITHER coded the selector as an ORDER-1 MARKOV
  chain / RLE that exploits run structure.
- **Mechanism it exploits:** `frame_utils.py:138` (600 pairs of one video → temporal correlation) ⊕ the selector wire
  (`public_pr112_frontier_beat_intake_20260610.md` §3: selector 248–249 B, "already at entropy" for order-0).
- **Predicted ΔS:** RATE, SMALL. The selector is only 248–249 B (0.14% of archive). PR#112's note "selector already at
  entropy" is for ORDER-0. If the modes have strong runs (RLE-friendly), order-1 could shave 20–40% → **−50 to −100 B →
  −0.00003 to −0.00007**. Honest: BELOW the R3 selector edge already harvested (−22 B) and near contest precision. This
  is a "confirm the bound" move, not a frontier breaker. **LOW-EV but cheap and orthogonal.**
- **Tried?** **PARTIALLY.** R3 (`pr110pp_r3_onhost_selector_verdict_20260610.md`) optimized the selector's per-pair
  CHOICE (which mode) on-host and shaved 22 B on the framing/mode-table. PR#112 did order-0 AC. NEITHER did order-1
  Markov / RLE on the mode-ID SEQUENCE. `grep selector.*markov|selector.*range|temporal.*correlat.*selector` → only
  generic readiness-audit hits, no selector-Markov implementation. **The order-1-on-SELECTOR lever is open** (distinct
  from PR#112's order-1-on-DECODER which they reported LOSES — the decoder weights are memoryless given tensor identity,
  but the selector SEQUENCE is a time series with run structure).
- **Readiness: SB.** ~30 LOC adaptive order-1 / RLE range coder on the 600-symbol mode stream + byte-close + the R3
  on-host mode table already exists.
- **Reuse targets:** R3's on-host mode table + `pr103_arithmetic_codec` + `constriction` + `tac.lossless.range_coder`
  (adaptive AC already in-tree).
- **FALSIFIABLE FIRST TEST ($0):** extract the 600 frontier mode IDs, measure order-0 entropy vs order-1 conditional
  entropy vs RLE-coded length. **KILL if** order-1 conditional entropy ≥ order-0 (modes are temporally iid — no run
  structure) → selector is fully exhausted, route to T1.

---

### T5 — TRAIN THE REPRESENTATION ERROR INTO THE CERTIFIED NULL SPACE (invisibility basis as a TRAINING CONSTRAINT) — RATE+DISTORTION — RES
- **What:** The `evaluator_invisibility_basis` certifies that ~22.7% of every camera channel (resize zero-weight
  pixels) is BIT-IDENTICAL invisible to both scorer heads (residual==0.0, amplitude-unlimited). We use it as a POSTPROCESS
  free-byte budget (S12 preimage). The UNTAPPED move: use it as a **TRAINING CONSTRAINT** — train the AFSR-1 decoder to
  put its REPRESENTATION ERROR into the null space, so the residual the encoder must carry is certified-free instead of
  scorer-visible. "Architecture trained to be cheap-to-encode by construction" (the `stacking_synergy` memo §positive-
  externality #3 names this as "the strongest synergy" but it is NOT built).
- **Mechanism it exploits:** `modules.py:73` (PoseNet) + `:109` (SegNet) share a fixed bilinear resize 874→384; its
  null space is certified (`evaluator_invisibility_basis_landed_20260610.md`: 22.7% zero-weight, 80.67% full null).
  Error placed there is provably free; the entropy coder then carries a lower-entropy visible residual.
- **Predicted ΔS:** RATE + DISTORTION (force-multiplier on every rate move). Hard to derive without the campaign, but the
  mechanism is exact: any error the decoder makes that lands in the null space costs 0 distortion AND can be coded more
  cheaply (lower visible-residual entropy). Band (campaign): **−0.01 to −0.04** as a compounding term on RANK 1, NOT
  standalone. It is the highest-leverage SYNERGY term, not an independent move.
- **Tried?** **NO.** It is DESIGNED (the invisibility basis is landed + the synergy memo names it) but NOT built as a
  training constraint — the basis is consumed only as a postprocess waterfiller action + a PR110++ atom generator.
- **Readiness: RES.** Requires the AFSR-1 score-aware retraining campaign (RANK 1) to exist; this is an aiming TERM in
  its loss, not a standalone lane. Add a regularizer: penalize visible-residual entropy / reward error projected onto
  `tier1_resize_null_space.npz`.
- **Reuse targets:** `evaluator_invisibility_basis` (the null mask + query API), `tac.null_space_exploiter`,
  `tac.differentiable_eval_roundtrip` (the differentiable resize), the AFSR-1 loss, `resize_null_preimage` (the
  postprocess sister this generalizes to a training objective).
- **FALSIFIABLE FIRST TEST ($0, smoke):** in a short AFSR-1 descent smoke (16-pair, ~300ep), add the null-space error
  regularizer and measure: (a) does d_seg/d_pose hold? (b) does the visible-residual entropy (∝ codeable bytes) drop vs
  the un-regularized run? **KILL if** the regularizer forces error into the null space but the VISIBLE residual entropy
  does NOT drop (the decoder's error was already mostly in-null, or the constraint hurts fidelity faster than it saves
  bytes — the knife-edge re-manifests).

---

### T6 — POSENET HALF-PIXEL RESIZE PREIMAGE GAMES (PoseNet has its OWN resize null, distinct from SegNet) — DISTORTION — SB
- **What:** Both heads resize 874→384, but they operate on DIFFERENT inputs: SegNet on frame1 only, PoseNet on the
  6-channel YUV6 of BOTH frames (`modules.py:73-74`). The CERTIFIED invisibility basis is the INTERSECTION (invisible to
  BOTH). But PoseNet's resize has a LARGER null space on the frame0 pixels (frame0 is SegNet-free entirely, so on frame0
  the ONLY constraint is PoseNet's resize null + YUV6 chroma null). The UNTAPPED move: exploit the **frame0-specific
  PoseNet-only null** (bigger than the both-heads intersection) — frame0 perturbations invisible to PoseNet's resize but
  NOT necessarily to SegNet's are STILL FREE on frame0 because SegNet never reads frame0.
- **Mechanism it exploits:** `modules.py:108` (SegNet frame1-only) ⊕ `:73` (PoseNet bilinear, `align_corners` defaults
  False → a specific half-pixel-aligned weight pattern). On frame0, the free DOF = PoseNet-resize-null ∪ YUV6-chroma-null
  — strictly LARGER than the both-heads intersection the invisibility basis certifies. The basis's "frame0 corollary"
  certifies frame0-zero-weight-of-BOTH-heads, but the frame0-PoseNet-only null is bigger and equally free.
- **Predicted ΔS:** This is a FREE-BYTE budget EXPANSION on frame0, not a direct ΔS. It enlarges the certified-free
  surface from "both-heads ∩ frame0" to "PoseNet-only ∩ frame0", which feeds T1/T2/S12. Indirect; magnitude = the extra
  null dimension on frame0 (frame0 is 1/2 the pairs' pixels). Could expand the free-byte budget on the frame0 axis by
  the full PoseNet-resize-null fraction (the basis measured 80.67% full-null for the intersection; frame0-PoseNet-only
  is ≥ that).
- **Tried?** **PARTIALLY.** The invisibility basis certifies the BOTH-heads intersection + a frame0 corollary (frame0
  zero-weight of both). `grep posenet.*resize.null|separate.*pose.*resize` → EMPTY for the frame0-PoseNet-ONLY null as a
  distinct certified surface. The basis explicitly notes "frame0 is SegNet-free but NOT PoseNet-free outside the
  zero-weight set" — i.e. it STOPPED at the intersection; the frame0-PoseNet-only null is the unclaimed extension.
- **Readiness: SB.** Extend `evaluator_invisibility_basis` with a `frame0_posenet_only_null` surface (derive PoseNet's
  resize null separately, intersect with chroma-null, restrict to frame0). ~40 LOC + certification test (in-basis →
  PoseNet input bit-identical AND SegNet doesn't read it).
- **Reuse targets:** `evaluator_invisibility_basis` (extend), `yuv6_chroma_subsampled_perturbation_operator`,
  `tac.xray.bilinear_resize_nullspace`, `scorer_read_surface_atoms.seg_scored_frame_mask`.
- **FALSIFIABLE FIRST TEST ($0):** derive PoseNet's resize-null dimension separately; measure
  `dim(frame0_posenet_null ∪ chroma_null) − dim(both_heads_intersection ∩ frame0)`. **KILL if** the difference is
  negligible (PoseNet and SegNet resize nulls coincide — same kernel) → the intersection already captures it.

---

### T7 — CROSS-PAIR POSE BUDGET ALLOCATION (E3 fungibility as an explicit GLOBAL optimizer) — DISTORTION — SB
- **What:** `evaluate.py:81-92` pools pose as a MEAN over 600 pairs BEFORE the sqrt nonlinearity. So per-pair pose is
  FUNGIBLE: a pose regression on pair A exactly offsets equal improvement on pair B (in d-domain). The UNTAPPED move:
  treat the 600×6 pose target as a GLOBAL budget and explicitly REALLOCATE representation capacity — spend pose fidelity
  where it's cheapest to achieve, sacrifice it where it's saturated/expensive, at exactly 1:1 in d_pose.
- **Mechanism it exploits:** `evaluate.py:90` `posenet_dist = posenet_dists / batch_sizes` (pooled mean) then `:92`
  `sqrt(10 * posenet_dist)`. The sqrt is on the MEAN → d_pose contributions add linearly inside the mean.
- **Predicted ΔS:** DISTORTION, derivation: the pose term is already at √(10·2.9e-5) ≈ 0.017 of the 0.192 total. Pose is
  near-floor; cross-pair reallocation can only help if some pairs are FAR from their per-pair pose floor while others
  overspend. On the frozen frontier (593/600 already argmin per R3) the headroom is tiny (**−1e-5 to +1e-3**, likely
  below contest precision). As a TRAINING aiming term (RANK 1) it is more valuable: weight the pose loss by per-pair
  achievability so capacity flows to cheap pairs. Band (campaign): **−0.005 to −0.015** if pose capacity is currently
  mis-allocated across pairs.
- **Tried?** **RECOGNIZED, not built as an optimizer.** E3 (`evaluate_py_fresh_eurekas_20260610.md`) + the stacking memo
  name cross-pair fungibility as "a synergy multiplier / global budget." But `grep cross.pair.*pose|pose.*fungib` →
  sensitivity-map + test hits only; no explicit global pose-allocation optimizer. The frozen-frontier per-pair selector
  (R3) is per-pair, not cross-pair.
- **Readiness: SB (frozen probe) → aiming term (campaign).** Frozen: measure each pair's distance-from-pose-floor +
  whether a cross-pair trade exists. Campaign: a per-pair pose-loss weight ∝ inverse achievability.
- **Reuse targets:** `sensitivity_map.wyner_ziv_reweight`, the R3 on-host per-pair pose table, the latent-axis per-pair
  sensitivity ranking, the AFSR-1 loss.
- **FALSIFIABLE FIRST TEST ($0):** from the R3 on-host per-pair pose table, compute each pair's pose vs its achievable
  floor (the argmin mode). Is the variance across pairs large enough that reallocation helps? **KILL if** all 600 pairs
  are within ε of their per-pair floor (no slack to reallocate — the frontier already globally-optimal in pose).

---

### T8 — LATENT CODES PROJECTED TO SCORER-NULL BEFORE ENTROPY CODING (regenerate, don't just recode) — RATE — SB
- **What:** PR#112 RECODES the latents they inherited (locked to bit-exact PR101 reproduction). We can REGENERATE them:
  push the latent codes toward the SegNet/PoseNet null space (via the invisibility basis / preimage compiler) BEFORE
  entropy coding, shrinking the residual entropy the coder carries — a lever PR#112 structurally cannot reach.
- **Mechanism it exploits:** the latents drive the decoder which produces frame pixels; the certified resize-null means
  some latent perturbations produce only scorer-invisible pixel changes. Project latent deltas onto the pre-image of the
  null space → lower-entropy latent stream at zero distortion.
- **Predicted ΔS:** RATE. This is MOVE 3 of the PR#112 intake (`public_pr112_frontier_beat_intake_20260610.md` §5,
  "structurally PR#112 cannot reach it"). Band: **−0.001 to −0.005** (it moves the latent-section entropy below the
  per-pair iid floor by exploiting the null-space DOF the iid floor counts as signal). Compounds with T1 (cluster the
  NULL-PROJECTED latents — even more redundant).
- **Tried?** **NO (named as MOVE 3, DEFERRED behind MOVE 1).** The PR#112 intake explicitly defers it as "higher EV,
  higher cost, queue after rate parity." It is genuinely un-built.
- **Readiness: SB.** Requires the resize-null preimage applied in LATENT space (project the latent→frame Jacobian's null
  contribution). ~50 LOC + the R1/R2 recoder to code the smaller residual.
- **Reuse targets:** `resize_null_preimage`, `evaluator_invisibility_basis`, `null_space_exploiter`,
  `pr101_split_brotli_codec.decode_latents_compact`, the R2 range coder.
- **FALSIFIABLE FIRST TEST ($0):** for a handful of frontier latents, compute the latent→frame Jacobian (the decoder is
  differentiable), find the null-projected latent that yields scorer-invisible frame change, measure its code length vs
  the original. **KILL if** the null-projected latent is NOT lower-entropy (the decoder's latent→frame map has no useful
  null direction at the frontier operating point — the latents are already minimal).

---

### T9 — DECODER WEIGHT PERMUTATION / SHARED-MODEL CLUSTERING beyond PR#101's CONV4_STORAGE_PERMS — RATE — SB
- **What:** PR#101 (L22) permutes Conv2d axes per-tensor for entropy-friendly storage; PR#112 shares one adaptive model
  across 4 tiny tensors. The UNTAPPED move: a GLOBAL weight-permutation + cross-tensor shared-model SEARCH that goes
  beyond the 13 hand-picked PR#101 perms — find the permutation of ALL decoder weights that minimizes the joint adaptive
  order-0 code length (the decoder is 90.9% of bytes, so even a 1% entropy reduction = −1,600 B).
- **Mechanism it exploits:** decoder weights are "memoryless given tensor identity" (PR#112 §3.1) — so the only lever is
  the MODEL ASSIGNMENT + the byte-map/permutation that makes each tensor's marginal cheapest. PR#112 stopped at per-tensor
  adaptive order-0 + 4-tensor sharing; the optimal tensor→model clustering + global permutation is a larger search.
- **Predicted ΔS:** RATE. PR#112 already captured the per-tensor order-0 floor (−1,060 B). The residual headroom is the
  CLUSTERING gain (more tensors sharing a model when their distributions match) + permutation gain. PR#112 reported
  order-1/kernel-position/neighbor contexts LOSE — so this is NOT a context-model gain, it's a clustering/permutation
  gain. Honest band: **−100 to −500 B → −0.00007 to −0.00033** (small; PR#112 likely near the order-0 floor already).
  LOW-EV but orthogonal and lossless.
- **Tried?** **PARTIALLY (PR#101 fixed perms + PR#112 4-tensor sharing); the SEARCH is open.** `shared_pmf_model` exists
  ("shared model across tensors chosen by exact cost") — but the GLOBAL clustering + permutation optimization over all
  ~28 tensors is not done.
- **Readiness: SB.** Extend `shared_pmf_model` with a clustering search + reuse PR#101's `CONV4_STORAGE_PERMS` machinery.
  ~40 LOC on top of R1.
- **Reuse targets:** `shared_pmf_model`, `pr101_split_brotli_codec` (the perm machinery, L22), R1's adaptive per-tensor
  model, `tac.neural_weight_codec_sensitivity` (codebook_sizes).
- **FALSIFIABLE FIRST TEST ($0):** after R1's per-tensor adaptive models are built, grid the tensor→model clustering +
  try the PR#101 perms on the full tensor set; measure joint code length. **KILL if** clustering+perm saves < 50 B over
  R1's per-tensor floor (PR#112 already at the order-0 clustering optimum).

---

### T10 — THE DUAL-AXIS (CPU vs CUDA) GT-DECODE EXPLOIT (different ground truth per axis) — DISTORTION — RES
- **What:** `evaluate.py:39-42` + `:58`: CUDA GT = `DaliVideoDataset` (NVDEC decode); CPU GT = `AVVideoDataset`
  (`frame_utils.yuv420_to_rgb`, BT.601 limited-range + bilinear chroma upsampling). **The ground-truth PIXELS differ per
  axis.** R3 discovered this as a BUG (the GT-decode mismatch). The UNTAPPED move: a submission tuned to the CPU GT decode
  specifically (the ranking axis) — the comp frames are scored against the CPU `yuv420_to_rgb` GT, so the optimal comp is
  the one closest to THAT specific decode, not the NVDEC one.
- **Mechanism it exploits:** `evaluate.py:58` (`DefaultDatasetClass` = AVVideoDataset on CPU) ⊕ `frame_utils.py:159-183`
  (`yuv420_to_rgb`, the exact CPU GT). Our flip map / atlas / cone were built vs PyAV GT (CPU-correct per E2), so the CPU
  axis is already aimed — but the COMP RENDER (our inflate output) is also being matched to this specific GT, and any
  systematic offset between our render's color space and the GT's BT.601 limited-range conversion is a free distortion
  source to correct (PR#98's channel biases are exactly this class — frame0 R−1/B−1, frame1 G−1, learned to compensate
  a known scorer bias).
- **Predicted ΔS:** DISTORTION. The PR#98 channel biases already capture the first-order systematic offset (−0.0001 to
  −0.0005 per the L28 lesson). The UNTAPPED residual is a per-channel / per-region affine correction tuned to the EXACT
  CPU GT `yuv420_to_rgb` (limited-range 255/219 luma, 255/224 chroma, bilinear chroma up). Band: **−0.0002 to −0.001**
  (second-order color-space alignment beyond the 3 PR#98 bias constants). The CUDA axis needs its OWN correction (R3's
  CUDA verdict showed pose +0.0232 drift — `cuda_axis_frontier_eval_verdict_20260610.md`).
- **Tried?** **PARTIALLY.** PR#98's 3 channel biases are in the frontier. R3 found the GT-decode bug. But a SYSTEMATIC
  per-channel/region affine fit to the exact CPU `yuv420_to_rgb` GT (beyond the 3 hand-set biases) is not done.
- **Readiness: RES→SB.** Build the comp-vs-GT systematic-residual map on the contest CPU host (the R3 machinery already
  decodes the GT correctly), fit an affine correction, fold into inflate's channel postproc.
- **Reuse targets:** the PR#98 channel-bias inflate code, R3's contest-GT decode (`frame_utils.yuv420_to_rgb`),
  `engineered_corrections`, the flip map.
- **FALSIFIABLE FIRST TEST (~$0.3, on-host):** on the contest CPU host, decode GT via `yuv420_to_rgb`, render comp via
  inflate, compute the per-channel mean residual (comp − GT). **KILL if** the residual is already zero-mean per channel
  (PR#98's biases fully captured it — no systematic offset remains).

---

### T11 — STRUCTURED-DROPOUT / SPARSE DECODER (prune decoder weights the reconstruction doesn't need) — RATE — RES
- **What:** The decoder is 162 KB INT8 + brotli at 98.6% of iid Shannon. Coding is exhausted; coarsening is FALSIFIED
  (×2). But PRUNING (structured sparsity — zeroing whole channels/filters the single-video reconstruction doesn't need,
  then NOT coding them) is distinct from coarsening (reducing precision of ALL weights). A single-video memorizer likely
  has redundant capacity; magnitude/Fisher-pruning + retraining the survivors could shrink the coded weight count.
- **Mechanism it exploits:** the decoder is a single-video overfit → many weights are near-zero or redundant. Pruning
  removes them from the coded stream entirely (a zero-channel codes as a 1-bit flag, not 8 bits × N).
- **Predicted ΔS:** RATE. If 20–40% of decoder channels are prunable at the frontier's d_seg/d_pose (with brief
  survivor-retraining), decoder bytes drop 162 KB → 100–130 KB → **−0.01 to −0.02**. This is the "smaller arch" rate bet
  from RANK 1, but via PRUNING the EXISTING frontier rather than retraining from scratch (cheaper, lower-risk).
- **Tried?** **PARTIALLY — coarsening/quantization killed, PRUNING distinct and not clearly tested.** The decoder-axis
  verdicts killed int4/int6 PTQ/QAT/per-channel/GPTQ/AWQ (`lossy_falsification_scope_audit`) — all are PRECISION
  reduction (coarsening). Structured CHANNEL pruning + survivor retrain is a different operator (sparsity, not precision).
  `grep` shows pruning primitives exist (IMP cycles `train_imp_cycle.py`) but not applied to the frontier decoder as a
  rate move.
- **Readiness: RES.** Requires retraining the survivors (a short campaign), so it's RANK-1-adjacent but smaller scope
  (start from the frontier weights, prune+finetune, not train from scratch).
- **Reuse targets:** `train_imp_cycle` (IMP pruning), the per-tensor sensitivity map (which channels are low-|grad|),
  `tac.differentiable_eval_roundtrip`, the AFSR-1 finetune loss, R1 recoder for the survivors.
- **FALSIFIABLE FIRST TEST ($0, smoke):** magnitude-prune 20% of the frontier decoder's channels, brief survivor
  finetune (16-pair smoke), measure d_seg/d_pose vs frontier. **KILL if** d_seg rises faster than the rate drops (the
  knife-edge — same failure mode as coarsening; the memorizer has NO redundant capacity, confirming the post-exhaustion
  finding that there's "no redundant precision").

---

## 2. THE TOP-5 BY VALUE × READINESS

| Rank | Technique | Axis | Predicted ΔS (derived) | Readiness | First test |
|---|---|---|---:|---|---|
| **1** | **T1 cross-pair latent dedup/clustering** | RATE | **−0.0031 to −0.0061** (30–60% of 15.4 KB latent) | SB (~60–90 LOC + $0.3 replay) | k-means over 600 decoded latents; bytes(dict+idx+resid) vs 15,387 B |
| **2** | **T8 latents projected to scorer-null before coding** | RATE | **−0.001 to −0.005** (below iid floor, PR#112 can't reach) | SB (~50 LOC, compounds w/ T1) | latent→frame Jacobian null-projection code-length vs original |
| **3** | **T5 train error into certified null space** | RATE+DIST | **−0.01 to −0.04** (compounding on RANK 1) | RES (aiming term in AFSR-1) | null-space regularizer in 300ep smoke: residual entropy drop? |
| **4** | **T11 structured channel pruning + survivor finetune** | RATE | **−0.01 to −0.02** (20–40% channel prune) | RES (prune+finetune, not from-scratch) | prune 20% channels + 16-pair finetune; d_seg vs frontier |
| **5** | **T2 cheapest-frame0 synthesis (seg-free → pose-only carrier)** | RATE | **−0.01 to −0.03** (campaign aiming) | RES→SB (warp-residual frame0 head) | d_pose of `affine_warp(frame1,pose)` frame0 (zero stored bytes) |

**Stacking note:** T1 ⊕ T8 ⊕ T9 ⊕ R1 ⊕ R2 ⊕ R3 ⊕ S12 are **all RATE, all on disjoint structure / sections** → stack
ADDITIVELY (proof-by-construction for the lossless ones; T1/T8 need a paired re-measure since they touch latent codes).
The DISTORTION/campaign movers (T5, T11, T2, T7) compound on the smaller-byte base AFTER the rate stack ships (race-mode:
ship the lossless rate stack first, then the campaign).

---

## 3. THE SINGLE MOST SURPRISING GAP

**T1 — cross-PAIR latent redundancy is completely unexploited, and it is the LARGEST single untried lossless lever.**

The surprise: the entire lab — including PR#112's sophisticated latent recode — treats the 600 latents through a
PER-PAIR lens (AR(1) on own deltas, cross-DIMENSION LS, discrete-Gaussian residuals). The latent-axis verdict declared
"LZMA at 3.4% of the iid floor" and DEFERRED the axis. But that is the *per-pair iid* floor. **The public test set is a
SINGLE contiguous drive (`0.mkv`, one video).** A single drive has enormous INTER-PAIR redundancy — the car is stopped,
or going straight, for long stretches, so dozens of the 600 latents are near-duplicates. Nobody clustered them. A
dictionary-of-K-representatives + per-pair index + sparse residual exploits a mutual information the per-pair iid floor
COUNTS AS SIGNAL. This is exactly the "we had the planner signal and never built the materializer" orphan pattern that
let PR#112 cash a win we'd identified — except here the win is potentially **−0.003 to −0.006, several times PR#112's
−0.0009**, and the structure (single-video self-similarity) is staring at us from `frame_utils.py:138`. The frozen-byte
exhaustion map was thorough on the per-pair and per-tensor axes but never asked the cross-pair question because the
single-video framing makes the redundancy "obvious" and therefore invisible.

**Honorable mention (the most surprising STRUCTURAL gap):** T2 — we proved frame0 is seg-free and built an entire
free-PERTURBATION apparatus on it, but never asked the dual question "what is the CHEAPEST frame0 that satisfies pose?"
The frontier spends real bytes decoding a full-fidelity frame0 the scorer never seg-reads; a warp-residual frame0
(`renderer.py:1096` already implements the primitive) could nearly halve the frame0 representation cost. We optimized
the budget ON frame0 without questioning whether frame0 needs to EXIST as a full decoded frame at all.

---

## 4. EXCLUDED — looks-untried-but-KILLED-or-already-have (the guard)

- **Pose dims 7-12 as free archive bytes** — DEFER (`hydra_dim_invariance.py`): dims 7-12 are scorer-INTERNAL outputs
  computed FROM pixels, NOT an archive transport channel. "Information must live in archive bytes/RGB pixels, not in
  scorer-internal post-forward dims." Correctly killed. NOT a gap.
- **Gradient-directed dithering / Floyd-Steinberg error diffusion** — ALREADY HAVE (`quantization.noise_shaped_round`,
  WIRED into `trick_stack.py:992/1458`). NOT a gap.
- **2×2 chroma subsample null space** — ALREADY HAVE (`yuv6_chroma_subsampled_perturbation_operator` +
  `constrained_gen.py:1749` zero-sum-within-block). NOT a gap.
- **Decoder coarsening / int4-6 PTQ / QAT / per-channel / GPTQ / AWQ** — FALSIFIED ×2
  (`frontier_decoder_axis_waterfill_verdict` c1/c2/c3 +0.07/+0.09/+0.16; `lossy_coarsening_T0312_retired` 0.3517 CUDA).
  T11 (channel PRUNING + retrain) is the DISTINCT-variant reactivation, not coarsening.
- **Latent 2nd-order delta re-prediction** — FALSIFIED (`frontier_latent_axis_waterfill_verdict`: 7.52 > 7.03 bits).
  T1 (cross-pair clustering) and PR#112's R2 (1st-order AR + range coder) are DISTINCT.
- **Selector recode (order-0)** — at entropy floor (R3 −22 B harvested; PR#112 order-0 AC). T4 (order-1/RLE on the
  temporal SEQUENCE) is the DISTINCT-variant; LOW-EV but open.
- **Frame-1 seg-repair correction sidecar** — information-theoretically incapable (`frontier_seg_repair_pool_verdict`:
  1.525 B/flip floor > 1.27 B/flip break-even). The fix is a better reconstruction (RANK 1 / T5 / T11), not a sidecar.
- **SNeRV stored-LF representations** — every rung 280–530× the frontier (`snerv_branch_b_round2_verdict`). DEFER.
- **R3 CPU→CUDA promotion** — NO TRANSFER (`cuda_axis_frontier_eval_verdict`: 0.226 CUDA > 0.205 control; pose +0.023
  drift). The CUDA axis needs its own attack (T10 is the CPU-axis variant; the CUDA axis is a separate frozen-pool map).
- **Adding files to the videos dir to shrink the rate denominator** — IMPOSSIBLE: `evaluate.py:64` rglob's the GT
  `uncompressed_dir`, which is the contest-provided videos, not ours. Denominator is fixed at 37,545,489. Confirmed
  unexploitable.
- **DALI partial-batch / odd-frame seam** — MOOT for the public set: `0.mkv` has 1200 frames = exactly 600 pairs, no
  odd-frame drop, no partial-batch padding. (Would matter only for an even/odd-frame-count video.)

---

## 5. 6-HOOK WIRE-IN (Catalog #125) + provenance

- **Hook #1 sensitivity-map:** T1/T8 consume the latent-axis per-pair/per-dim sensitivity map; T11 consumes the
  per-tensor |grad| map; T5/T2 consume the invisibility basis + B2 Y-fraction.
- **Hook #2 Pareto:** T1/T8/T9/T4/T3 move the RATE axis only (orthogonal to the saturated distortion vertex); T5/T11/T2
  are the only moves OFF the vertex (re-synthesis).
- **Hook #3 bit-allocator:** T1's dictionary-index codec + T9's clustering/perm search ARE bit-allocator primitives
  (PR95 L21–L32 family); T5's null-space regularizer is a training-time allocator.
- **Hook #4 cathedral-autopilot:** T1/T8/T9 fold into the `byte_range_entropy_recode_chain` materializer + paired-eval
  dispatch surface (same harvest queue as R1/R2/R3).
- **Hook #5 continual-learning:** T1's k-means result (if latents cluster) + T11's prune-finetune d_seg both reseed the
  V3 judge on whether cross-pair / sparsity is a live axis the per-pair/per-tensor verdicts missed.
- **Hook #6 probe-disambiguator:** every row's FIRST TEST is a $0 local disambiguator (k-means bytes; warp-only d_pose;
  null-projected entropy; order-1 conditional entropy; prune-finetune knife-edge) — each resolves the technique's
  open question before any spend.

**Provenance:** every claim cites `{file, line, observed fact}` from `upstream/{evaluate.py,modules.py,frame_utils.py}`
read in full + the 2026-06-09/10 verdict corpus + repo greps (which separate untried from already-have/killed). No
score claim; `[macOS-CPU advisory]`. Frontier read from pointer. The exact paired CPU+CUDA eval is the only authority
for any predicted ΔS. NO FAKE: each technique names the EXACT scorer mechanism it exploits and a falsifiable kill test.

**Cross-refs:** `MASTER_ROADMAP_post_exhaustion_map_20260610.md` (the DEFER verdicts respected) ·
`orphan_harvest_recovery_ledger_20260610.md` (R1/R2/R3/R4 — T1/T8 extend the recovered lossless class) ·
`public_pr112_frontier_beat_intake_20260610.md` (MOVE 3 = T8; the per-pair-only lens T1 breaks) ·
`stacking_synergy_composition_plan_20260610.md` (T1/T8 are the missing cross-pair axis; T5 = synergy #3) ·
`evaluator_invisibility_basis_landed_20260610.md` (T5/T6/T8 consumers) ·
`composition_algebra_coherence_law_20260610.md` (the additive-stacking proof for the rate moves) ·
`evaluate_py_fresh_eurekas_20260610.md` (E1=T3, E3=T7, E2=T10) ·
`frontier_{decoder,latent,seg_repair}_*_verdict_20260610.md` + `cuda_axis_frontier_eval_verdict_20260610.md` (§4 guard) ·
`src/tac/contest_exploits/hydra_dim_invariance.py` (the dims 7-12 kill) ·
`src/tac/optimization/scorer_read_surface_atoms.py` (the read-surface facts T2/T6 extend).
