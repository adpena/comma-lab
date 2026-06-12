# Layer-1 CARRIER — the canonical first-principles full-stack analysis (2026-06-12)

**Author:** Layer-1-carrier first-principles subagent (`layer1-carrier-first-principles-20260612`).
**Type:** DEEP RESEARCH + DESIGN memo. NO production code, NO GPU, NO dispatch, NO collision with running agents (the MPS basin daemon pid owns `experiments/results/torch_vehicle_full_mps_basin_bc20_n600` + `src/tac/torch_vehicle/**`; the Cool-Chic sister owns `src/tac/substrates/cool_chic/**`; this memo touches none of them).
**Evidence grade:** `[analysis]` — every quantified claim is tagged either **[MEASURED:<artifact>]** (an exact number from a cited prior smoke) or **[PREDICTION:<basis>]** (a derivation). NO score is claimed. The means/ends firewall: this is a MEANS (a carrier analysis) toward the END (a lower exact score); it moves no row.
**Frontier (pointer, NOT hardcoded):** `.omx/state/canonical_frontier_pointer.json` → contest-CPU **0.19109982** (177,169 B, sha `b46897267d…`, `lane_pr110_payload_entropy_recode`); contest-CUDA **0.20533** (186,876 B). **Frontier UNMOVED.** Target ladder: T_3 = sub-0.15 (the aim), T_1 = sub-0.19 (floor of acceptable), T_floor ≈ 0.118 (rate-dominated headroom proof).

> **NO-FAKE headline:** The decisive finding of the prior carrier campaign — `capacity_verdict_smaller_basis_by_rate_REFUTED_pivot_to_waterfiller_20260611.md` — is that **the carrier-as-rate-lever thesis is MEASURED-REFUTED**: you cannot simultaneously shrink the decoder AND hold the d_seg basin in this architecture class, and you cannot recode the bytes below the entropy floor. This memo's central conclusion (§D) is the *theoretical explanation* of that measurement: the rate floor is a **scorer-conditional MDL invariant**, shared by all sufficiently-expressive carriers, so **no carrier swap lowers the floor** — sub-0.15 must come from distortion + side-info, not from a better carrier. The one place a carrier *could* go below HNeRV is by spending **zero** on the scorer-null space (the witness/score-quotient carrier, §E), and even that is bounded by the same invariant — it can only recover the SLACK HNeRV pays, not lower the floor itself.

---

## A. CONTEST-AS-COMPRESSION — the scorer-conditional MDL formalization (the theoretical backbone)

### A.1 The exact object

The contest is: produce `archive.zip` (bytes `B`) whose `inflate.sh` deterministically emits 1200 RGB frames `\hat{X} = (\hat{x}_0,\dots,\hat{x}_{1199})` (600 pairs), minimizing

```
S(\hat{X}, B) = 100·d_seg(\hat{X}) + sqrt(10·d_pose(\hat{X})) + 25·B/N,    N = 37_545_489
```

where, with `f_seg` = frozen EfficientNet-B2 Unet (5-class logits, last-frame-only), `f_pose` = frozen FastViT-T12 (6-dim pose on a two-frame YUV6 stack), and `X` = the GT frames decoded by `frame_utils.yuv420_to_rgb`:

- `d_seg(\hat{X}) = mean_pixels[ argmax f_seg(\hat{x}_{last}) ≠ argmax f_seg(x_{last}) ]` — a **0/1 per-pixel argmax-disagreement RATE** (NOT an L2; verified `upstream/modules.py` + `score_pair_components`).
- `d_pose(\hat{X}) = mean_pairs ‖ f_pose(\hat{x}_pair)_{1:6} − f_pose(x_pair)_{1:6} ‖²` — a **continuous 6-dim MSE**, entered through the concave `sqrt(10··)`.

The eval roundtrip (the carrier's frames are bicubic↑874 → bilinear↓384 → uint8 before scoring) is part of the channel, so the carrier optimizes against the *post-roundtrip* render.

### A.2 The carrier must hit an EQUIVALENCE CLASS, not a point

The decisive structural fact: **the scorer is many-to-one.** Two very different frame-sequences `\hat{X}, \hat{X}'` score *identically* if:

- **SegNet:** their last-frame 5-class **argmax maps agree pixel-for-pixel** — the logits may differ arbitrarily as long as the per-pixel winner is unchanged. `d_seg` is invariant under any perturbation that does not cross a SegNet decision boundary. This is a genuine **discrete cell** structure: the frame space is partitioned into argmax-constant polytopes; `d_seg` only sees which cell you are in.
- **PoseNet:** their 6-dim pose readout agrees. `d_pose` is invariant under any perturbation in the **null space of the local PoseNet Jacobian** `J_pose = ∂f_pose/∂\hat{x}` (≈ 6 constrained directions per pair; the rest of the ~590k-pixel pair is pose-invisible to first order). Pose is NOT a fiber/equivalence-class like seg — its level sets are MSE ellipsoids (corrected in `validation_score_program_compiler_quotient_theory_20260606.md` §CORRECTIONS-1) — but the **local null space is large**, which is what matters for byte-spend.

So the carrier's job is: **describe (to within bytes `B`) a point inside the scorer-target equivalence class** {`\hat{X}` : seg-argmax = GT-argmax AND pose-readout ≈ GT within the score's tolerance}.

### A.3 The rate floor is the SCORER-CONDITIONAL MDL, not the pixel MDL

Define the achievable rate floor as the description length of the *cheapest member* of the target equivalence class:

```
R*_scorer  =  min over carriers   B(carrier)
              s.t.  d_seg(decode(carrier)) ≤ τ_seg   AND   d_pose(decode(carrier)) ≤ τ_pose
           =  K_program( the shortest legal archive whose inflate output lands in the scorer-target cell )
```

This is a **conditional Kolmogorov complexity / MDL** quantity — but conditioned on the SCORER, not on pixel-fidelity:

```
R*_scorer  =  K( equivalence-class-representative | inflate.sh runtime )   ≪   K( exact pixels )
```

Why `≪`: the pixel MDL `K(X)` must describe every textured detail, every chroma gradient, every high-frequency edge the dashcam captured. The scorer-conditional MDL only has to describe **which SegNet polytope + which pose-readout** each frame realizes — a *vastly* coarser quotient. A frame can be wrong in 22.7% of its pixels (the certified resize-null fraction, `S12`/`evaluator_invisibility_basis`) and score identically; it can be wrong in *every* interior-class pixel (only the argmax-boundary pixels matter for `d_seg`); it can be wrong in every pose-null direction.

**This is the theoretical backbone for the entire program** and it is exactly the CLAUDE.md "Evaluator-Equivalent Witness Compiler Paradigm": the winning representation minimizes `25·B/N` for the shortest `B` that lands `inflate.sh` output in the frozen evaluator cells. The corollary that decides everything (§D): **R\*_scorer is a property of the (video, scorer) pair — it does not depend on which carrier family you choose.** Any sufficiently-expressive carrier can reach it; the carrier question is about the SLACK above it.

### A.4 The measured anchor that grounds A.3

[MEASURED: `capacity_verdict_…_20260611.md`] Decompose the frontier `S = 0.19110`:
```
S  =  rate(0.11797)  +  100·d_seg(~5.6e-4 → 0.056)  +  sqrt(10·d_pose)(~0.017)
```
- **rate = 0.11797 = 61.7% of S** — the dominant term, and numerically equal to T_floor (not a coincidence: T_floor is "the rate term at the best byte count we have, with distortion at its architectural floor").
- **distortion residual = 0.07314** — the immediately-reachable headroom (drive `d_seg`, `d_pose` to ~0 *without paying bytes* and `S → ~0.118`, crossing T_3).

This decomposition is the ranking prior for every layer (§F).

---

## B. CANDIDATE ANALYSIS — math / geometry / algebra / rate / distortion-control per carrier

Each carrier is graded on five axes: **(i) GEOMETRY** (the manifold of frame-sequences it spans, vs the single-dashcam-drive manifold = ego-motion + slowly-varying scene + a near-stationary camera); **(ii) ALGEBRA** (how it parametrizes, and the symmetry it exploits); **(iii) RATE** (where the bytes live + (in)compressibility); **(iv) DISTORTION CONTROL** (how cheaply/precisely it hits the SegNet/PoseNet targets); **(v) TRAINABILITY** (can the Layer-2 levers shape it?).

### B.0 The compact verdict table

| Carrier | Geometry fit | Rate locus | Compressibility of rate | Distortion control | Score-aware trainable | Verdict vs sub-0.15 |
|---|---|---|---|---|---|---|
| **HNeRV** (conv+PixelShuffle+sin, trained weights = code, per-pair latents) | **EXCELLENT** — conv-decoder manifold matches smooth dashcam; per-pair latents = temporal coords | decoder weights (~91%) + per-pair latents (~9%) | **near-incompressible BY DESIGN** (converged overfit weights ≈ MDL → brotli can't shrink → THAT'S the floor) | **EXCELLENT** — full renderer, every score DOF reachable; argmax-boundary precisely targetable | **YES** (PR95 8-stage curriculum; all 5 Layer-2 levers plug in) | **PROVEN BANK → ~T1.** Sits AT the scorer-conditional floor. The reference. |
| **Cool-Chic** (entropy-coded latent grids + tiny synth MLP + AR prior) | GOOD — multiscale grids span smooth content; weak on the single-video temporal axis | latents (99.9%); synth+AR negligible | **AR-conditional entropy — but MEASURED saturated** [σ/std≈0.75]; latent rate ~44–48× HNeRV floor | MEDIUM — synth MLP is small; d_seg control weaker at compact capacity (capacity-walled @ d_seg~0.014) | YES (the AR-NLL = coded-bits unifier is built) | **DEFERRED-pending-latent-compressibility.** Real lossless carrier; NOT lower-floor on measured evidence. |
| **VQ-NeRV / discrete-latent** | GOOD — codebook discretizes residual features; U-shape | codebook + per-pair indices + decoder | indices compress well (low-entropy IDs) BUT codebook+decoder still pay the weight floor | MEDIUM — quantization-of-features can shift argmax; needs straight-through | YES (capstone `tac.capstone_vq_nerv` IS this; it's the base_ch=20 substrate) | **= HNeRV floor** (it IS our substrate, different container). No separate floor win. |
| **SIREN / FINER / WIRE** (coordinate MLP, periodic/wavelet activations) | POOR for compression — spans a *continuous-frequency-controlled* manifold, but pays full MLP weights with NO content-prior embedding | MLP weights only (no latents) | weights incompressible; NO per-frame embedding → must memorize ALL frames in weights → bytes BLOW UP for 1200 frames | LOW per-byte — global field, hard to localize an argmax-boundary edit | partial (spectral-bias knob ω₀) but no native score path | **DOMINATED.** No content-adaptive embedding → worse rate than HNeRV at equal fidelity. |
| **Wavelet / Ballé scale-hyperprior** | MEDIUM — multiscale transform coding; great for natural-image RD, built for cross-image generalization | transform coeffs + hyperprior side-info | **excellent for PSNR-RD; but optimizes pixel-MDL, not scorer-MDL** → pays for scorer-invisible detail | LOW for THIS scorer — RD-optimal ≠ argmax-optimal; spends bits on texture the argmax ignores | yes (end-to-end) but wrong objective by default | **DOMINATED by default**; useful only as a *component* (entropy coder / null-projection), not a standalone carrier. |
| **Witness / score-quotient / evaluator-null-space** (store ONLY scorer-sensitive DOF) | **OPTIMAL IN PRINCIPLE** — parametrizes exactly the quotient {seg-argmax} × {pose-readout}, spans nothing else | argmax-boundary map + pose-readout side-info + a cheap base render | **lowest possible BY CONSTRUCTION** — never pays for the null space an INR still encodes | **HIGHEST** — directly addresses the score DOF; argmax-boundary is the native object | YES — this IS the score-domain Lagrangian made primary | **THE class-shift candidate.** Could go BELOW HNeRV *iff* the null space is cheaply identifiable + the witness round-trips. Honest risk in §E. |

### B.1 HNeRV — the reference (full math in §C)

**(i) Geometry.** HNeRV decodes each pair from a content-adaptive 28-d embedding through a conv decoder: `\hat{x} = D_θ(e_t)`, `D_θ` = `[Linear → reshape → (Conv(C→4C) → PixelShuffle×2 → bilinear-skip → sin)×6]` from 6×8 to 384×512 (L18). The manifold `{D_θ(e) : e ∈ R^28}` is a smooth 28-dim-parametrized image manifold *per fixed θ*; across pairs, the latents `{e_t}` trace a 1-D-ish curve (the drive's temporal evolution) inside it. **This matches the single-dashcam manifold almost perfectly**: a single drive is a low-dimensional family (ego-motion + slowly varying scene), and a conv decoder's inductive bias (locality, translation-equivariance, smooth upsampling) is exactly the natural-image prior. [WebSearch: HNeRV (arXiv 2304.02633) — "learnable encoder generates content-adaptive embeddings; HNeRV blocks distribute parameters so higher layers store high-resolution detail."]

**(ii) Algebra.** Parametrization = (decoder weights θ, per-pair latents {e_t}). Symmetry exploited: **temporal redundancy** (per-pair latent deltas are small — L25; 2 frames per latent — L19) + **spatial locality** (conv weight sharing). The sin activation (vs ReLU) avoids dead regions for single-video memorization (L18); PixelShuffle is bandwidth-efficient at constant FLOP.

**(iii) Rate.** ~91% decoder weights (INT8→zigzag→brotli, one blob) + ~9% latents (per-dim delta→uint8→split→brotli). The decoder blob is **near-incompressible** — and §C explains why that is a FEATURE.

**(iv) Distortion control.** Full renderer → every score DOF (seg argmax + pose readout) is a function of the rendered frame, so all are reachable by training θ. The argmax-boundary is targetable (Lever-5 margin weighting).

**(v) Trainability.** [MEASURED basin: `async_authority_eval_basin_20260612.md`] ep40 `score=1.20036 d_seg=0.00911 d_pose=0.00522 bytes=92171`; the full PR95 8-stage curriculum + all 5 Layer-2 levers (`incurriculum_levers_…`) plug directly in. This is the only carrier with a live descending basin.

### B.2 Cool-Chic — a real lossless rate carrier, NOT a lower floor

**(i) Geometry.** Hierarchical latent grids (coarse 4×24×32 + fine 4×48×64 per pair) decoded by a tiny synthesis MLP; an AR prior models `p(z_t | z_{t-1})`. The grids span a multiscale-spatial manifold; the synth MLP is too small to carry a strong content prior, so **the latents carry almost all the information** (geometry fit GOOD on spatial smoothness, WEAK on temporal — the AR prior is the only temporal coupling, and it is shallow conv-3×3). [WebSearch: Cool-Chic (arXiv 2401.02156, 2212.05458) — "overfit a lightweight decoder + latent per image; AR model predicts mean/variance from a local context; 680 mult/pixel, 2 orders less complex than Ballé hyperprior."]

**(ii) Algebra.** Parametrization = (latent grids {z_t}, synth MLP, AR prior net). Symmetry: spatial multiscale + (intended) temporal conditional entropy. The AR prior is the rate model.

**(iii) Rate.** [MEASURED: `cool_chic_fullstack_synergy_20260612T153802Z.md`] 99.92% latents. The entropy coder is real + lossless (57.5% vs raw int16; **24.5% pure-AR-entropy** over fixed-width at the same grid). BUT the absolute floor is **7.83 MB → rate term 5.22 → ~44× HNeRV's 161 KB decoder floor.**

**(iv) Distortion control + (v) Trainability — the decisive NO-GO.** [MEASURED: `cool_chic_ar_prior_training_feasibility_20260612T164116Z.md`] Training the AR prior 130× longer moves the coded rate **≤1.3%**; **σ/std saturates at ~0.75** (the conditional Gaussian is only ~25% tighter than the marginal, FIXED by the conv-3×3 capacity, not by training time). The Layer-1↔2 unifier is built (AR-NLL = coded-bits), but the lever it was supposed to power (prior training) is **exhausted at 6 epochs.** Worse: [MEASURED: `capacity_verdict_…`] a compact Cool-Chic-class decoder **capacity-walls at d_seg ~0.014** (~25× the frontier's 5.6e-4); L3 with 64% more latent bytes got *worse* — the saturation signature. **Verdict: real carrier, but neither lower-rate nor lower-distortion than HNeRV on all measured evidence.** The one untested branch is *latent re-optimization for compressibility* (full-model loop, not prior-training) — DEFERRED, not refuted.

### B.3 VQ-NeRV / discrete-latent — this IS our substrate, not a separate floor

**(i)(ii) Geometry/Algebra.** [WebSearch: VQ-NeRV (arXiv 2403.12401) — "U-shaped arch + codebook discretizes shallow residual + inter-frame residual."] Parametrization = (codebook, per-pair indices, decoder). The codebook quantizes feature space; indices are low-entropy IDs. Our **base_ch=20 substrate is exactly this** (`tac.capstone_vq_nerv`, `build_capstone_archive_bytes` = VQ-index codebook + stored per-pair latent + brotli decoder, per `bolton_inventory_…` §0).

**(iii) Rate.** Indices compress well, BUT the codebook + decoder weights still pay the weight floor — and the decoder weight blob is the same ~91%-of-bytes object as HNeRV. **No separate rate floor:** VQ just re-containers the same scorer-conditional MDL.

**(iv)(v).** Quantizing features can flip a SegNet argmax (the straight-through estimator must protect the boundary — Lever-4 score-aware QAT). Trainable (it's the live substrate). **Verdict: = HNeRV floor, different grammar.** Its value is *containerization* (the capstone materializer, `bolton_inventory_…` §Phase-0), not a lower floor.

### B.4 SIREN / FINER / WIRE — dominated for THIS task

**(i) Geometry.** A coordinate MLP `\hat{x}(u,v,t) = MLP_θ(γ(u,v,t))` spans a continuous field whose **spectral content is controlled by the activation** (SIREN ω₀ / FINER variable-period / WIRE wavelet). [WebSearch: SIREN (spectral-bias knob ω₀), FINER (flexible spectral-bias), WIRE (wavelet activation) — "high-capacity activations with rich spectral control but acutely initialization-sensitive."] The frequency-tunability is real and elegant, but...

**(ii) Algebra — the fatal flaw.** Pure coordinate MLPs have **NO content-adaptive per-frame embedding.** All 1200 frames must be memorized in the *shared weights* `θ`. There is no `e_t` to cheaply index a frame. This is precisely the gap HNeRV was invented to close ("unlike NeRV/E-NeRV that reconstruct from fixed content-AGNOSTIC embeddings, HNeRV uses content-adaptive embeddings" — WebSearch).

**(iii) Rate.** To represent 1200 distinct dashcam frames with no embedding, `θ` must grow large → **bytes blow up** at equal fidelity vs HNeRV's `θ + {e_t}` split. [PREDICTION: information-theory — the per-frame information must live *somewhere*; with no latents it all lands in incompressible weights, strictly dominating HNeRV's split where the cheap part is the latents.]

**(iv)(v).** A global field is hard to edit at a single argmax-boundary pixel (no local handle); score-aware training has no native path (only the ω₀ spectral knob). **Verdict: DOMINATED.** The spectral-bias idea is worth borrowing *as an activation inside a NeRV decoder* (a Layer-2 architecture tweak), not as a standalone carrier.

### B.5 Wavelet / Ballé scale-hyperprior — wrong objective by default

**(i)(ii) Geometry/Algebra.** [WebSearch: Ballé scale-hyperprior (arXiv 1802.01436) — "hyperprior captures how neighboring latent scales vary together; Gaussian-scale-mixture entropy model conditioned on side-info; SOTA *PSNR* RD."] Transform coding: `y = g_a(x)`, quantize, entropy-code with a hyperprior `z`. Wavelets (Mallat) are the fixed-basis analogue. Both are **built for cross-image generalization and PSNR-RD.**

**(iii)(iv) — the mismatch.** The objective is `R + λ·D_pixel` (MSE/PSNR distortion), which **spends bits on every textured detail** to lower pixel error — but the scorer ignores texture inside an argmax cell and ignores everything in the pose null. So a PSNR-optimal allocation is **score-suboptimal**: it pays for scorer-invisible detail. [PREDICTION: the orthonormal-invariance argument in `smaller_learned_basis_deep_math_…` — a fixed orthonormal basis cannot reduce the entropy of an already-near-iid signal; energy compaction requires correlation the converged weights/latents MEASURED-DO-NOT-HAVE; the only video-specific win is a video-specific rotation, which must be charged in-archive (no free lunch).]

**(v) Verdict: DOMINATED as a standalone carrier; VALUABLE as a component.** The scale-hyperprior's *conditional entropy model* is exactly the kind of context model Cool-Chic's saturated AR prior needs (B.2 pivot lever #2), and wavelet/null-projection is a Layer-3 bolt-on (`T8`). Use the parts, not the whole.

### B.6 Witness / score-quotient / evaluator-null-space — the class-shift candidate (full design §E)

**(i) Geometry.** Parametrizes ONLY the quotient `Q = {SegNet-argmax maps} × {PoseNet-readout vectors}` — the coarsest manifold consistent with the score. Spans *nothing* in the scorer-null space.

**(ii) Algebra.** Decompose each frame's DOF into (a) **seg-relevant** = the argmax-boundary set `∂` (pixels with small SegNet top-2 logit margin — `compute_logit_margin_map`); (b) **pose-relevant** = the ~6 pose-Jacobian directions per pair; (c) **null** = everything else (≈ 95%+ of pixels). Store (a)+(b); fill (c) with the cheapest legal values (a coarse base render + maximally-compressible null fill, `S12`).

**(iii) Rate — lowest possible BY CONSTRUCTION.** Never pays for the null space an INR still encodes in its weights. [PREDICTION: scorer-conditional MDL — this is the carrier whose `B` literally equals `R*_scorer` if the null space is perfectly identified.]

**(iv)(v) — highest distortion control, but honest risks (§E).** The argmax-boundary IS the native object; pose readout is stored side-info (Wyner-Ziv). The risks: can the null space be identified *cheaply* (the boundary set is video-dependent → must be transmitted or regenerated), and does the witness *round-trip* through uint8/resize/parse-back (a boundary pixel set at 384×512 must survive the bicubic↑/bilinear↓/uint8 channel). **Verdict: THE potential class-shift; gated on the cheap-null-identification + round-trip risks.**

---

## C. WHY HNeRV IS SUCH A GOOD FIT — the deep reverse-engineering (ESSENTIAL vs incidental)

The question "what makes HNeRV such a good fit" has a precise, somewhat counterintuitive answer rooted in §A.3. Five properties, classified:

### C.1 [ESSENTIAL] Full-renderer dominance (HNeRV-parity L5)

The scorer derives masks AND pose **from the rendered frames** (`f_seg(\hat{x}_last)`, `f_pose(\hat{x}_pair)`). Therefore **representing the frames directly dominates representing the components.** A mask-only or pose-only carrier (Lane-12 NeRV mask codec, palette-frame-1 lever B) is dominated by representing the frames the components are derived from — and empirically fails (lever B: `100·d_seg=0.826` busts T_1 alone, pose-blind palette frame1, `adversarial_review_…` claim #4). HNeRV outputs `(T,3,H,W)` RGB — the right object. **This is the single most important property** and it is *essential*: any sub-0.15 carrier must be a full RGB renderer (or a witness that *is* a render, §E).

### C.2 [ESSENTIAL] Trained-decoder-weights ≈ near-MDL of the single video → "incompressible weights" is a FEATURE

The counterintuitive crux: HNeRV's frontier archive is ~91% decoder weights, and **brotli cannot shrink them** — which sounds like a failure but is the **signature of optimality.** A converged, overfit network's weights are a *near-optimal code* for the video it memorized: the training objective drove `θ` to the minimal-loss configuration, and at that minimum the weights carry maximal information per parameter (high entropy, low redundancy). [PREDICTION: rate-distortion — at the RD-optimal operating point, the description (the weights) is incompressible *because* any remaining compressibility = remaining redundancy = sub-optimality the training would have removed.] This is *exactly* the orthonormal-invariance result [MEASURED-via-derivation: `smaller_learned_basis_deep_math_…`]: the weights are near-iid → no fixed basis compacts them → brotli is already near the floor. **So HNeRV sits NEAR R\*_scorer not by accident but because overfitting-to-convergence IS approximate MDL.** This is *essential* and is the reason a "smaller learned basis" cannot beat it on rate (§D): a smaller decoder that holds the d_seg basin would have to encode the same scorer-conditional information in fewer incompressible bytes — impossible if the bytes are already near-MDL.

### C.3 [ESSENTIAL] Cheap, precise distortion control via the conv-decoder manifold

Because the decoder is a smooth conv renderer with a strong natural-image prior, **hitting the SegNet argmax + pose readout is cheap**: small weight/latent adjustments move the render smoothly toward the target cell, and the argmax-boundary (where `d_seg` lives) is reachable by gradient. [MEASURED basin: ep40 `d_seg=0.00911`, descending.] Contrast Cool-Chic at compact capacity (capacity-walled at d_seg~0.014) and SIREN (no local handle). The conv-manifold's match to the single-drive manifold (low-dim, smooth) is what makes the distortion *control* cheap — *essential*.

### C.4 [ESSENTIAL] Per-pair temporal latents exploit the single-drive structure (L19/L25)

2 frames per 28-d latent; per-pair latent deltas are small (temporal redundancy of a single drive). This is the cheap part of the rate split — the part that *is* compressible (delta-coded) — and it is where the video's genuine temporal information lives. *Essential*: it is the right factorization of (shared appearance prior in weights) × (per-pair temporal coordinate in latents).

### C.5 [INCIDENTAL] The specific activation / upsampler / 8-stage curriculum

sin-vs-ReLU, PixelShuffle-vs-ConvTranspose, the exact 29,650-epoch 8-stage schedule (L14-L18), Muon-final-stage (L15) — these are *real wins* (they are why PR95 converged) but they are **incidental to the carrier question**: they tune *how close to R\*_scorer* HNeRV gets (the slack), not *whether* HNeRV is the right manifold. A FINER activation or a different upsampler would still be "an HNeRV-class carrier." These are Layer-2 levers, not carrier identity.

**Synthesis:** HNeRV fits because it is (a) a full RGB renderer (C.1), (b) whose overfit weights are near-MDL of the single video so it sits near the scorer-conditional floor (C.2), (c) with cheap precise distortion control from a conv manifold matched to the single drive (C.3), (d) factored as appearance-prior × cheap-temporal-latents (C.4). The first two are why no carrier *swap* beats it on rate (§D); the third and fourth are why it is the proven distortion-control bank.

---

## D. THE RATE-FLOOR-INVARIANCE CRUX — the decisive verdict

> **VERDICT: The rate floor is a SCORER-CONDITIONAL INVARIANT, NOT carrier-dependent. No carrier swap lowers it. Sub-0.15 must come from DISTORTION + SIDE-INFO, not from a better carrier — EXCEPT the witness/score-quotient carrier (§E), which does not lower the floor either but can recover the SLACK HNeRV pays above it.**

### D.1 The argument (information theory)

The floor `R*_scorer = K(equivalence-class-representative | inflate.sh)` (§A.3) is defined by the **(video, scorer) pair**, with a `min` over ALL carriers. It is therefore a property of the *problem*, not of any carrier family. Any carrier that reaches the target distortion `(τ_seg, τ_pose)` must, by the source-coding theorem applied to the scorer-conditional source, pay **≥ R\*_scorer bits** — however it partitions them (weights vs latents vs coeffs vs indices). A change of carrier is a change of *parametrization* of the same equivalence class; the **information content of "which cell + which pose-readout" is invariant** under reparametrization, just as differential entropy is invariant under an orthonormal change of basis. [MEASURED-via-derivation: `smaller_learned_basis_deep_math_…` — "Φ orthonormal ⟹ differential entropy invariant; coefficient entropy drops ONLY via energy compaction, which requires correlation the weights MEASURED-DO-NOT-HAVE."]

### D.2 The measured corroboration (this is not just theory)

[MEASURED: `capacity_verdict_smaller_basis_by_rate_REFUTED_pivot_to_waterfiller_20260611.md`] The smaller-basis-by-rate thesis was tested directly and **REFUTED from 5 directions** (#64/#71/#72/#73/#67):
- A smaller (Cool-Chic-class) decoder does NOT hold the d_seg basin (measured wall at ~0.014, ~25× the frontier).
- A frontier-class decoder that DOES hold the basin costs ≈ frontier bytes (177 KB).
- ⟹ **"You cannot simultaneously shrink the decoder AND hold the d_seg basin"** — which is exactly D.1: the scorer-conditional information is fixed, so paying fewer bytes means failing the distortion constraint, and meeting it means paying ≈ R*_scorer.

And [MEASURED: `cool_chic_ar_prior_…`] the Cool-Chic latent floor is 44–48× HNeRV's decoder floor and does NOT move with prior training — a *different* carrier, paying *more* (not less) for the same problem, consistent with HNeRV being already near R*_scorer and Cool-Chic carrying slack.

### D.3 The two honest qualifications (so the verdict is not over-claimed)

1. **"Invariant floor" does NOT mean "all carriers reach it."** R*_scorer is a `min`; a given carrier reaches `R*_scorer + slack`. HNeRV's slack is small (its weights are near-MDL, C.2). Cool-Chic's slack is large (44×). VQ's slack ≈ HNeRV's (same container math). SIREN's slack is huge (no embedding). **So the carrier question collapses to: which carrier has the SMALLEST slack — and HNeRV (and its witness refinement) wins that, not by a lower floor but by less waste.**
2. **The ONE carrier property that could approach the floor from BELOW HNeRV's slack is spending ZERO on the null space.** HNeRV's near-MDL weights still encode scorer-INVISIBLE appearance (an INR renders *all* pixels, including the 95% the argmax ignores and the pose-null). The witness carrier (§E) does not — it stores only (a)+(b) of §B.6 and fills (c) with free/cheap values. **This does not lower R*_scorer; it lowers HNeRV's SLACK above R*_scorer by refusing to pay for the null space.** That is the entire class-shift opportunity, and it is bounded: the witness can at best reach R*_scorer, which §A.4 decomposes as `rate-at-the-true-witness-bytes`, plausibly below 0.11797 but NOT below the genuine scorer-conditional information.

### D.4 The decisive consequence for the program

- **Stop searching for a lower-floor carrier.** [MEASURED] The floor is invariant; the search is closed (`capacity_verdict_…` "STOP sweeping Cool-Chic capacity knobs").
- **sub-0.15 (T_3) is reached by the DISTORTION residual, not the carrier.** §A.4: drive `100·d_seg → 0` (Levers 2,5) + `sqrt(10·d_pose) → quant floor` (Lever 3 pose-FiLM) at constant-or-lower bytes → `S → ~0.118`, **crossing T_3.** This needs NO carrier swap — it needs the Layer-2 levers on the HNeRV bank.
- **sub-0.118 (below T_floor's frontier-byte anchor) is the second, harder phase** and is where the witness carrier's null-space-slack-recovery (§E) + Layer-3 lossless recodes (T1/T8) attack the rate term itself.

---

## E. FIRST-PRINCIPLES DESIGN — the from-scratch optimal carrier for THIS contest

Given the scorer (SegNet argmax + PoseNet pose), the single video, and ALL our tools, the optimal carrier is the **scorer-quotient witness** — designed below — with a **pragmatic hybrid** as the lower-risk path.

### E.1 The scorer-quotient witness carrier (the class-shift design)

**Principle:** parametrize ONLY the scorer-sensitive DOF; spend ZERO on the null space. Decompose every pair's frame into three byte-budgets:

```
B_witness  =  B_base  +  B_seg-boundary  +  B_pose-sideinfo  +  B_null-fill(≈ minimal)
```

**Component 1 — `B_base` (a cheap shared render that is RIGHT inside most argmax cells).**
A *small* HNeRV-class decoder (NOT a big one — its job is only to land most interior pixels in the correct argmax cell, which is easy: interior class regions are robust). This is the "coarse base" the witness refines. [PREDICTION: the interior of a SegNet class polytope is large and robust → a low-capacity render suffices for ~95% of pixels; the d_seg residual concentrates at the boundary set ∂.] Geometry: spans the smooth-content manifold cheaply; it is allowed to be wrong everywhere the scorer can't see.

**Component 2 — `B_seg-boundary` (the argmax-boundary witness — the seg DOF).**
Compute the SegNet top-2 logit margin map `m(p) = top1−top2` (`compute_logit_margin_map` / `segnet_boundary_marginals`). The boundary set `∂ = {p : m(p) < τ}` is where `d_seg` flips actually happen — typically a thin 1-D-ish curve set, ≪ the frame. Store a **sparse correction** that nudges *only* `∂` pixels back across the boundary. The KEY economy (the conditional-position trick, `LeverD`/`margin_conditional_residual`): because the decoder **regenerates the margin field for FREE at inflate time** (it runs the same render), the sidecar only has to address the decoder-KNOWN low-margin set — conditional position cost `log2 C(|∂|, K)` ≪ unconditional. Algebra: a colex-rank-coded (L31) flip-position set + per-flip class id. Geometry: parametrizes the seg quotient directly.

**Component 3 — `B_pose-sideinfo` (stored GT pose, FiLM-injected — the pose DOF).**
The pose readout is 6 scalars/pair. Storing them is `600 × 6 × bytes` with delta+brotli → **~1–3 KB** [MEASURED-projection: `pose_film_cpu_disambiguator_…` + `mlx_pr95_port.pose_film.stored_pose_bytes`]. FiLM-condition the base decoder on the stored pose (Wyner-Ziv side-info): the decoder is *told* the pose and modulates frame1 features so the render's PoseNet readout matches GT — **d_pose collapses to the stored-pose quant floor** instead of the decoder's learning floor. [MEASURED GO: `pose_film_cpu_disambiguator_20260612.md` — "realized d_pose drops by a large factor even at the frozen-decoder LOWER BOUND; the `sqrt(10·d_pose)` reduction beats the stored-pose byte cost projected to n=600."] This is the single most-validated witness component.

**Component 4 — `B_null-fill` (≈ minimal).**
Everything not in ∂ and not pose-constrained is the scorer-null space. Fill it with the **maximally-compressible legal values** (the certified resize-null preimage `S12`: 22.7% of every channel is provably invisible after the bicubic↑/bilinear↓ roundtrip; fill it with values that minimize entropy → fewer brotli bytes for the base render). [MEASURED primitive: `evaluator_invisibility_basis`, −10 to −19.5% of coded frame bytes, certified zero distortion.]

**The objective it minimizes:**
```
min_{θ_base, ∂-corrections, pose-store}   25·B_witness/N
   s.t.  argmax f_seg(render) = argmax f_seg(GT)  pixelwise   (seg cell membership)
         ‖f_pose(render) − f_pose(GT)‖² ≤ τ_pose             (pose ellipsoid)
```
This is the score-domain Lagrangian (Lever 2) made **primary AND structural**: the carrier's *architecture* is the quotient, not just its loss.

**Why it could go BELOW HNeRV/Cool-Chic:** it **never pays for scorer-invisible detail an INR still encodes.** HNeRV's near-MDL weights still render all 590k pixels/frame faithfully (paying for texture the argmax ignores and pose-null pixels). The witness pays only for ∂ (thin) + 6 pose scalars + a coarse base. [PREDICTION: scorer-conditional MDL §A.3 — `B_witness → R*_scorer`, recovering HNeRV's null-space slack.]

**The honest risks (NO-FAKE — these are why it is a research bet, not a slam-dunk):**
1. **Can the null space be identified CHEAPLY?** ∂ is video-dependent. If it must be *transmitted*, its bytes eat the win. The conditional-position trick (decoder regenerates the margin field for free) is the mitigation — but it only works if the base render's margin field *agrees* with GT's boundary location (a chicken-and-egg: the base must be good enough that ∂ is identifiable from the render, not from GT). [RISK: if the base render's argmax is wrong at a boundary, the decoder can't regenerate the right ∂ to correct.]
2. **Does the witness ROUND-TRIP through uint8/resize/parse-back?** A boundary correction at 384×512 must survive bicubic↑874 → bilinear↓384 → uint8. A 1-pixel flip can be erased by the resize blur. [RISK: `LeverD` is currently a DEFER (#51) because on the *frozen frontier* a naive boundary sidecar was net-negative — receptive-field collateral. On a *worse-trained base* (more flips) the break-even is easier, but the round-trip survival must be measured.]
3. **Receptive-field collateral.** Flipping one boundary pixel can shift the SegNet receptive field and flip *neighbors* — the correction can cost more flips than it fixes. [MITIGATION: the waterfill `boundary_math/margin_conditional_residual` is fail-closed on NET value, pricing collateral in.]

### E.2 The pragmatic hybrid (the lower-risk path — recommended FIRST)

**HNeRV-renderer-backbone (the proven bank) ⊕ pose-FiLM (validated GO) ⊕ in-curriculum score-domain training (Levers 2,5) ⊕ Layer-3 lossless recodes (R1/R2/T1).**

This is the witness *spirit* without the full quotient-architecture risk:
- Keep the HNeRV base render (C.1–C.4 — proven distortion control, near-MDL weights).
- Add Component 3 (pose-FiLM) — [MEASURED GO], collapses d_pose at ~1–3 KB.
- Train the base with the **score-domain Lagrangian** (Lever 2) + **margin-weighted seg loss** (Lever 5) so the *decoder itself* concentrates capacity on ∂ — this is Component 2 folded into training (the decoder learns to get boundary pixels right) instead of a fragile post-hoc sidecar, sidestepping risks 1–3.
- Add Component 4 (S12 null-fill) + Layer-3 lossless recodes (R1/R2 shipped, T1 cross-pair dedup unbuilt) for the rate term.

**Contrast:** the pure witness (E.1) is the *class-shift* (potential sub-0.118) but carries the boundary-sidecar round-trip risks; the hybrid (E.2) folds the boundary witness *into training* (Lever 5), getting most of the d_seg win at near-zero added risk, and is the **measurement-first MVP path.**

---

## F. RANKED RECOMMENDATION + FIRST $0 STEP (EV toward sub-0.15)

### F.1 The ranking

| Rank | Carrier / path | EV toward sub-0.15 | Why this rank | First step |
|---|---|---|---|---|
| **1** | **HNeRV basin (the proven bank) + Layer-2 score-domain levers** | **HIGHEST** — reaches ~T1 measured; T_3 reachable via the 0.07314 distortion residual | The floor-invariance verdict (§D) says sub-0.15 is a DISTORTION problem, and this is the only carrier with a live descending basin + all 5 levers wired. The distortion residual → 0 lands `S → ~0.118`. | **Confirm the basin frontier**, then deploy Lever 2 (boundary-STE seg surrogate, lowest-risk routing change) — gated on the basin, no $0 probe needed (it's the live run). |
| **2** | **Pose-FiLM (witness Component 3) on the HNeRV bank** | **HIGH** — [MEASURED GO] collapses `sqrt(10·d_pose)` at ~1–3 KB; high marginal value near d_pose→0 | The single most-validated witness component; the disambiguator already returned GO at the frozen-decoder lower bound. Composes additively with rank 1. | **Already $0-cleared.** First *build* step: land `tac.torch_vehicle.pose_film` (port `_PoseFiLM`, default-OFF byte-identical) + the additive pose codec section; stage the paired CPU/CUDA A/B (the memo's Lever-3 deploy plan). |
| **3** | **The pragmatic hybrid = rank1 ⊕ rank2 ⊕ margin-weighted seg (Lever 5) ⊕ R1/R2/T1** | **HIGH (the integrated sub-0.15 candidate)** | This IS the witness-in-spirit (§E.2): folds the boundary witness into training (no fragile sidecar), pose as side-info, lossless recodes for rate. The composed path to T_3. | After ranks 1–2 land: A/B Lever 5 (margin-weighted seg) + apply the Phase-1 lossless rate batch (R1/R2 shipped; **T1 cross-pair dedup is the biggest unbuilt lever, −0.003 to −0.006**, `bolton_inventory_…`). |
| **4** | **Pure scorer-quotient witness (E.1, the class-shift)** | **MEDIUM-HIGH but UNCERTAIN** — the only path that lowers HNeRV's *slack* (potential sub-0.118) | Bounded by the floor invariance (§D.3); carries the boundary-sidecar round-trip + receptive-field risks (E.1). The class-shift bet, sized AFTER the hybrid measures the distortion knee. | **FIRST $0 PROBE (the cheapest validate-or-falsify):** on the **frozen basin checkpoint**, compute the margin map `m(p)` over 600 pairs (CPU, reuse the `seg_out` the eval already produces), measure (a) `|∂|/|frame|` (how thin is the boundary set?), (b) the **conditional-position byte cost** `log2 C(|∂|, K_flips)` vs the d_seg win `100·(flips_fixed/N_pixels)`, and (c) **round-trip survival**: flip K boundary pixels, push through bicubic↑/bilinear↓/uint8, re-measure d_seg. If `|∂|` is thin AND flips survive AND the conditional cost clears the 1.27 B/flip break-even → the witness is GO; else the hybrid (rank 3) captures the d_seg win in-training instead. **This is the single decisive $0 measurement for the whole class-shift question.** |
| **5** | **Cool-Chic latent re-optimization** (the one untested branch) | **LOW-MEDIUM, DEFERRED** | [MEASURED] prior-training is NO-GO; the latent-compressibility branch is untested but starts 44× above HNeRV's floor — a long climb. | Only if ranks 1–4 stall: a $0 local-MPS full-model probe (latents + scorer loop) measuring whether the joint loss's AR rate term makes the *latents themselves* more compressible (NOT prior training). Sequence behind the basin. |
| — | **SIREN/FINER/WIRE, Ballé/wavelet standalone** | **DOMINATED** | §B.4/B.5 — no content embedding (SIREN) / wrong objective (Ballé). | Borrow the PARTS (FINER activation as a Layer-2 decoder tweak; scale-hyperprior as Cool-Chic's context model; wavelet as T8 null-projection), never as standalone carriers. |

### F.2 The first $0 step (MVP-first, the decisive measurement)

**The single highest-value $0 probe is rank-4's boundary-witness feasibility probe** (above), because it answers the ONE open question the whole carrier analysis reduces to: *is the scorer-null space cheaply identifiable AND does a boundary witness round-trip?* — which decides whether the class-shift (E.1) is real or whether the hybrid (E.2, ranks 1–3) is the ceiling. It reuses the `seg_out` the basin eval already computes (no new forward, no GPU, no basin contention — read the frozen fork-point checkpoint, exactly as the pose-FiLM disambiguator did). It is falsifiable: a thin ∂ + surviving flips + cleared break-even = GO for the witness; otherwise the hybrid captures the same d_seg win in-training.

**Full-stack synergy (how it composes):** the carrier (Layer 1) feeds the Layer-2 levers (the score-domain Lagrangian IS the witness objective; pose-FiLM IS witness Component 3; margin-weighted seg IS Component 2 folded into training) and the Layer-3 bolt-ons (S12 null-fill + R1/R2/T1 lossless recodes attack the rate term the witness minimizes). The three layers are ONE co-designed system whose objective is `25·B_witness/N` subject to the seg-cell + pose-ellipsoid constraints — and the binding constraint is **distortion first** (Phase-2a: Levers 2,5,3 drive the 0.07314 residual → ~0, crossing T_3), **then rate** (Phase-2b: Levers 1,4 + T1/T8 + the witness null-space-slack recovery push toward T_floor).

---

## Wire-in hooks (CLAUDE.md 6-hook per Catalog #125)

1. **Sensitivity-map** — ACTIVE (design): the margin map `m(p)` + pose-Jacobian null IS the per-pixel scorer-sensitivity map the witness carrier is built on; feeds the bit-allocator.
2. **Pareto constraint** — ACTIVE: §D establishes the rate floor is a scorer-conditional invariant → the Pareto frontier is `{distortion residual} × {rate slack}`; the witness minimizes slack, the levers minimize the distortion residual.
3. **Bit-allocator** — ACTIVE (design): the witness's three-budget decomposition (`B_base + B_∂ + B_pose + B_null`) IS a bit-allocator prior; S12 null-fill + T1 are its rate primitives.
4. **Cathedral autopilot** — N/A (analysis; no archive-deployable artifact).
5. **Continual-learning posterior** — DESIGN: the floor-invariance verdict (§D) + the boundary-witness feasibility probe's result (rank-4 first step) are falsifiable anchors that reseed the planner ("carrier-swap EV ≈ 0; distortion + witness-slack are the live axes").
6. **Probe-disambiguator** — ACTIVE: the rank-4 boundary-witness $0 probe IS the disambiguator between "pure witness class-shift (E.1)" and "hybrid ceiling (E.2)".

**Mission contribution:** `frontier_breaking_enabler` (a carrier analysis that REDIRECTS the program off the refuted carrier-swap search and onto the distortion + witness-slack axes; names the one $0 probe that decides the class-shift). **Frontier UNMOVED 0.19109982.** No score asserted. No GPU launched. No paid spend. No collision with running agents.

**Pending input (NO-FAKE):** the sister "joint-latent-compressibility" Cool-Chic probe (`cool_chic_joint_latent_compressibility_*.md`) had **NOT landed** at write time — it is the measured test of rank-5 / B.2's untested branch (latent re-optimization). If it returns GO, rank 5 rises and Cool-Chic's slack may narrow; if NO-GO, §D.2's corroboration strengthens (Cool-Chic carries irreducible slack). Either way it does not change the §D floor-invariance verdict (a measured Cool-Chic outcome is a *slack* measurement, not a floor measurement).

---

## Sources (WebSearch, cited inline)

- HNeRV — [arXiv 2304.02633](https://arxiv.org/abs/2304.02633) (content-adaptive embeddings; HNeRV blocks distribute params for high-res detail).
- Cool-Chic — [arXiv 2401.02156](https://arxiv.org/pdf/2401.02156), [arXiv 2212.05458](https://openaccess.thecvf.com/content/ICCV2023/papers/Ladune_COOL-CHIC_Coordinate-based_Low_Complexity_Hierarchical_Image_Codec_ICCV_2023_paper.pdf), [Cool-chic 5.0 arXiv 2605.02726](https://arxiv.org/html/2605.02726) (overfit latent grids + AR entropy model; 680 mult/pixel).
- VQ-NeRV — [arXiv 2403.12401](https://arxiv.org/abs/2403.12401) (U-shape + codebook discretizes residual features).
- SIREN / FINER / WIRE — [SIREN spectral-bias survey](https://arxiv.org/pdf/2411.03688), [FINER (liuzhen0212.github.io/finer)](https://liuzhen0212.github.io/finer/) (spectral-bias tuning; activation expressiveness; no content embedding).
- Ballé scale-hyperprior — [arXiv 1802.01436](https://arxiv.org/pdf/1802.01436) (Gaussian-scale-mixture conditional entropy; PSNR-RD-optimal).
