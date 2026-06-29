# Related-codec literature sweep for the v2 vehicle (SDS-TSC) — 2026-06-29T20:17:07Z

**Status:** `[research-signal / advisory]` · pointer contest-CPU **0.19110 UNMOVED** · this is a research **MEANS**, not goal progress.
**Authority:** $0 subagent `related-codec-sweep`. NO GPU / NO paid dispatch. Online + OSS + papers authority only.
**NO-FAKE discipline:** every arXiv ID below was checked via web search/fetch this session. The "Verification ledger" (last section) records what was VERIFIED real, what was MISLABELED in the dispatch prompt (corrected here), and what could NOT be located (flagged UNVERIFIED — never cited as real). Where a relevance claim is my inference vs the paper's own claim, I say so.

## What v2 is (so the neighbors are legible)
An **indirect rate-distortion / coding-for-machines** codec for ONE comma2k19 dashcam clip: emit the shortest `archive.zip` whose decoded witness, pushed through R (bicubic↑874→uint8→↓384→argmax), lands in the same cells of a FROZEN SegNet (5-class argmax) + FROZEN PoseNet (6-DOF). Vehicle (FEED-jf/jl/jm, SDS-TSC):
static canonical scene + ego-pose/screw-twist SE(3) stream → deterministic **per-class warp** (Road=ground-homography, sky=rotation, hood=identity) → single **1-Lipschitz SDF lane carrier** → small **conditioned learned residual** (Wyner-Ziv X−E[X|Y], Y=openpilot centerline) for the ragged ±1px lane + per-frame jitter → integer-deterministic decode. **One open wall:** per-frame SegNet-argmax **JITTER** floor (~0.008, FEED-jq) — a warp can't predict per-frame flicker; the leading candidates for the ragged residual GENERATOR are flow-matching / diffusion.

## De-dup vs prior surveys (#150/#151/#152/#155/#59)
This memo deliberately does NOT re-report what we already hold. Already-covered (do not re-survey):
- **#150 VCM/coding-for-machines SOTA**, **#151 VCM theory** (rate-distortion / IB / RDP / indirect-RD / CEO), **#152 entropy+context** (Ballé/Minnen factorized+scale-hyperprior, ELIC, MLIC; CompressAI/constriction; INR weight coding). → I add only the **2025–2026 task-aware deltas** (Family 5) and the **distributed/Wyner-Ziv neural line** (Family 4) that those surveys under-covered.
- **#155 level-set/fiber quotient codec, Dubois lossyless**. → The **gauge/canonicalization/quotient prior-art home** (Family 7) is the equivariant-ML formalization of #155's quotient idea; that is genuinely NEW here.
- **#59 contour/region-MDL, waterfilling/OT, amortized luma/pose carriers, Rust crates**. → The **SDF-as-codec-primitive** line (Family 3) and the **INR motion+residual** line (Family 2) extend #59's carrier work.
- Already in-tree from the DAG FEEDs (do not "discover" as new): **2408.13256** diffusion factorize/compose (FEED-jo), **2512.20043** LieFlow (FEED-jp), **2311.01450** DreamSmooth (FEED-jr). I reference them only where a new paper composes with them.

---

## Family 1 — Generative neural codecs (flow-matching / diffusion) fixing TEMPORAL FLICKER (= our jitter wall)

**Top papers (verified):**
1. **GNVC-VD — "Generative Neural Video Compression via Video Diffusion Prior" (arXiv 2512.05016).** DiT video-diffusion codec; a **flow-matching latent-refinement** module denoises intra+inter latents at sequence level; crucially it does **not** denoise from pure Gaussian — it **initializes from the decoded latent and learns a CORRECTION term**. Stable quality below 0.03 bpp; explicitly targets the **perceptual flickering** of frame-wise generative priors. *Relevance: this is the published instantiation of "generator as a residual correction on a decoded prior" = our conditioned-residual fork, and it fixes flicker = our jitter wall.*
2. **GVCC — "Generation Is Compression: Zero-Shot Video Coding via Stochastic Rectified Flow" (arXiv 2603.26571).** Turns a pretrained video foundation model into the codec with **no retraining**: the bitstream specifies the generative ODE trajectory; converts the **deterministic rectified-flow ODE → an SDE at inference** to open per-step stochastic injection points for codebook-driven coding. <0.002 bpp (T2V). *Relevance: the ODE↔SDE bridge is exactly the determinism knob we need — we want the DETERMINISTIC ODE direction (bit-identical decode), they go the other way for fidelity; same machinery, opposite gauge choice.*
3. **DiV-INR — "Extreme Low-Bitrate Diffusion Video Compression with INR Conditioning" (arXiv 2604.08329).** Replaces intra keyframes with a compact **INR that conditions a pretrained video diffusion model**; jointly trains the INR + parameter-efficient diffusion adapters; <0.05 bpp. *Relevance: this is the literal INR+diffusion hybrid our generator-gauge fork is choosing between — INR for the compact base, diffusion for the ragged detail.*

Also verified and relevant: **CoD-Lite (2604.12525)** — one-step lightweight-conv diffusion, real-time (60fps enc/42fps dec @1080p), finding that **compression-oriented pretraining beats generation-oriented at small scale** and **distillation lets lightweight conv suffice** (NB: this is IMAGE, not video — prompt mislabeled). **NeuralLVC (2604.03353)** — masked diffusion + temporal conditioning (lossless). **Free-GVC (2602.09868)** training-free extreme generative w/ temporal coherence. **GIViC (2503.19604)**, **DiffVC-OSD (2508.07682)**, **2501.13528** (temporal diffusion info reuse), **OT-NFM (2604.06413)** ODE-free one-step flow matching.

**DRAW-FROM (for the GPU run):**
- **Generator = residual CORRECTION on a decoded prior, not generation-from-noise** (GNVC-VD). Our residual generator should be trained to correct the warp+SDF prior, initialized from it — small target, stable, exactly our X−E[X|Y] framing.
- **Few-step DETERMINISTIC flow-matching ODE** (OT-NFM one-step; CoD-Lite distillation; FastFlow accel — see ledger) is the determinism×30-min-budget×bit-identical path. Diffusion's stochasticity is a liability for us; the rectified-flow/flow-matching deterministic ODE is the right cousin (this corroborates FEED-jp's LieFlow lean).
- **Compression-oriented pretraining + distillation at small scale** (CoD-Lite) — our generator is tiny and single-clip; distill, don't scale.

**CONTRIBUTE-BACK / NOVELTY:** every one of these optimizes **human-perceptual** fidelity (LPIPS/FID/flicker-as-seen). Ours optimizes an **indirect machine metric** (argmax-cell membership through a frozen SegNet/PoseNet). None conditions the generator on an **explicit geometric warp + SDF prior**; none targets a **stored-jitter / dither split** of the residual (our DreamSmooth-informed plan). v2 = generative-residual-codec **in task-space, geometry-conditioned**.

---

## Family 2 — INR / coordinate motion-compensated video codecs (the v2 architecture's closest neighbors)

**Top papers (verified):**
1. **"Implicit Neural Video Compression" (arXiv 2112.11312).** THE canonical decomposition: an implicit net **modulates the coordinate inputs to do motion compensation** between frames + a **small residual net codes the P-frame residual**. *Relevance: this IS v2's skeleton (warp-the-coords + residual) stated years ago for pixel fidelity — our novelty is the warp being a PHYSICAL SE(3) screw and the residual being task-space + conditioned.* **Closest single architectural neighbor in the literature.**
2. **CWRNN-INVR — "A Coupled WarpRNN based Implicit Neural Video Representation" (arXiv 2604.06564).** Mixed net + **residual grid**; a **Coupled-WarpRNN multi-scale motion** module explicitly represents "regular and structured" motion, residual grid carries the rest. PSNR 33.73 on UVG @3M. *Relevance: the explicit structured-motion / residual-grid SPLIT mirrors our warp(structured) / learned-residual(ragged) split — a 2026 confirmation that the split is the SOTA shape.*
3. **NIRVANA (arXiv 2212.14593, CVPR'23).** Group-of-frames INR, **autoregressive: each group's net initialized from the previous group's weights**; quantization-during-training (no post-hoc prune). 12× faster encode. *Relevance: warm-start-from-neighbor is our per-stage / temporal warm-start discipline; in-training quantization is the integer-deterministic-decode requirement.*

Also verified: **TINC (2211.06689, CVPR'23)** tree-structured hierarchical MLP with distance-shared params (a multi-scale partition prior, Daubechies-flavored); **Neural Residual Flow Fields (2201.04329)** flow + residual decomposition; **NVRC (2409.07414)**, **PNVC (2409.00953)** practical INR codecs; **HF-enhanced hybrid (2411.06685)**; **SHACIRA (2309.15848)** hash-grid INR compression; **ResFields (2309.03160)** residual neural fields for spatiotemporal signals; **"A Survey of Implicit Neural Representations for Video Compression" (TechRxiv, Aug 2025)** = the anchor survey to read once, end-to-end, to fully de-dup Family 2.

**DRAW-FROM:**
- **Coordinate-modulation = motion comp + small residual net** (2112.11312) — adopt this as the named, citeable skeleton; our screw-warp is the principled replacement for the learned coord-modulation MLP (fewer bytes, physical).
- **Residual-GRID, not residual-MLP** (CWRNN-INVR) for the ragged lane residual — a spatial grid may beat an MLP at the high-frequency ragged edge where INR spectral bias bites; worth a $0/cheap arm in the GPU run.
- **In-training quantization** (NIRVANA) — bake the integer-decode constraint into training, do not post-quantize.

**CONTRIBUTE-BACK / NOVELTY:** all Family-2 codecs minimize **reconstruction PSNR/MS-SSIM**; the residual is RGB. Ours minimizes **argmax-cell agreement through R**, the residual lives in task-space, and the motion field is an **interpretable physical SE(3) screw with per-class semantics** (Road-homography / sky-rotation / hood-identity), not a black-box learned coord-modulation. That per-class-physical warp is not in any Family-2 paper I found.

---

## Family 3 — SDF / level-set as a codec PRIMITIVE

**Top papers (verified):**
1. **"Mesh Compression with Quantized Neural Displacement Fields" (Pentapati et al., arXiv 2504.01027, Computer Graphics Forum 2025).** A small net encodes a **displacement field refining a COARSE base mesh**; **quantized** weights; SOTA 4×–380× geometry compression. *Relevance: this is the geometry-coding analog of v2's "coarse warp prior + small learned residual" and confirms quantized-neural-displacement is a real, strong byte regime; the SDF/level-set is the 2D image-space cousin.*
2. **ResFields (arXiv 2309.03160).** Residual neural fields for spatiotemporal signals — augments an INR with time-varying residual weight layers. *Relevance: a principled way to add a small temporal-residual capacity to a (mostly static) field = our static-canonical + per-frame-residual shape, in field form.*
3. **NSDF / level-set background (general, e.g. emergentmind NSDF topic; SDF-from-points lines 2410.14189 / 2503.20066).** *Relevance: confirms the 1-Lipschitz eikonal SDF is the canonical neural representation for a boundary; our FEED-jk MEASURED a single-SDF lane carrier surviving R at 5.9e-4 — the literature supports SDF as a compact, sub-pixel-accurate boundary carrier.*

**DRAW-FROM:**
- **Coarse base + quantized neural displacement** (Pentapati) — directly transferable byte regime + the "store coarse, learn the refinement" decomposition for the SDF lane carrier.
- **Residual weight layers for the small temporal delta** (ResFields) — a clean place to put the per-frame jitter capacity without a second full field.

**CONTRIBUTE-BACK / NOVELTY:** these compress **geometry for rendering fidelity** (Chamfer/Hausdorff). Using an SDF as a carrier whose only job is to **survive a frozen segmentation argmax through a downpath R** (not be rendered) is, as far as I found, **not in the SDF-compression literature**. The "carrier validated by an indirect frozen-scorer metric" is v2-original.

---

## Family 4 — Wyner-Ziv / distributed / CONDITIONAL coding with decoder side-information (= our openpilot head-start)

**Top papers (verified):**
1. **"Neural Distributed Source Coding" (Whang et al., arXiv 2106.02797, IEEE JSAIT 2024).** Learns encoder/decoder where **correlated side-info is available only at the decoder**; recovers Wyner-Ziv binning; conditional VQ-VAE. *Relevance: this is the canonical neural realization of our exact setting — encoder sends X; decoder has Y (openpilot centerline) for free; send only X−E[X|Y].*
2. **"Neural Distributed Compressor Discovers Binning" (Özyılkan, Ballé, Erkip, arXiv 2310.16961, JSAIT 2024).** Shows a learned distributed compressor **rediscovers Wyner-Ziv binning in source space**; Ballé co-author. *Relevance: validates that a learned conditional codec genuinely captures the WZ rate gain — gives us a tested architecture + the expectation that the openpilot conditioning is realizable, not just theoretical (sharpens FEED-jh/jm's measured ~64% head-start).*
3. **"Importance Matching Lemma for Lossy Compression with Side Information" (arXiv 2401.02609).** A practical lemma + scheme for lossy compression WITH decoder side-info (stochastic/one-shot friendly). *Relevance: a concrete coding tool for the conditioned residual; the one-shot/single-instance flavor fits our single-clip regime better than asymptotic WZ.*

Also verified: **"Robust Distributed Compression with Learned Heegard-Berger Scheme" (2403.08411)** (side-info may be absent — robustness if openpilot Y is unreliable on some frames); **"Learning to Write on Dirty Paper" (2507.17427)** (Gelfand-Pinsker, the encoder-side-info dual); **"Distributed Compression in the Era of Machine Learning: A Review" (2402.07997)** = the survey to anchor Family 4 de-dup. (Our in-tree ATW codec / Wyner-Ziv-PoseNet line — `atw_codec_*`, `wyner_ziv_*`, `z8_m6_*` — already operationalizes the PoseNet-6dim decoder-side-info; this family is the external SOTA backing for that.)

**DRAW-FROM:**
- **Conditional VQ-VAE encoder/decoder with decoder-only Y** (Whang; Özyılkan-Ballé) — the canonical architecture for the conditioned-residual generator; cite + mirror, don't reinvent.
- **Importance-matching / one-shot WZ** (2401.02609) for the single-instance regime — asymptotic binning may not be the right tool for one clip; the one-shot lemma is.
- **Heegard-Berger robustness** (2403.08411) — design the residual so a missing/erroneous openpilot frame degrades gracefully (decoder-side-info-optional).

**CONTRIBUTE-BACK / NOVELTY:** these target **MSE/perceptual** rate-distortion with a generic correlated Y. Ours has a **structured, physically-meaningful Y** (an openpilot world-model centerline) and a **task-space distortion** (argmax cells), and composes the WZ residual on top of a **geometric warp+SDF prior** — a 3-stage prior (warp → SDF → WZ-residual) that the flat distributed-coding papers don't have.

---

## Family 5 — Task-aware / machine-vision codecs since the #150–152 window (2025–2026 deltas)

**Top papers (verified):**
1. **"Image Coding for Machines via Feature-Preserving Rate-Distortion Optimization" (arXiv 2504.02216).** RDO where distortion = **feature distance through the downstream net**, not pixel distance. *Relevance: the published statement of our objective shape (optimize for the frozen network's features, not pixels) — but for a soft feature distance, where ours is the hard argmax cell.*
2. **"Symmetric Entropy-Constrained Video Coding for Machines" (arXiv 2510.15347).** Recent (2025) VCM with explicit entropy constraint. **"SMC++: Masked Learning of Unsupervised Video Semantic Compression" (arXiv 2406.04765)** — compress to preserve semantics without task labels. *Relevance: current VCM SOTA framing for the seg-preserving objective.*
3. **"Progressive Learned Image Compression for Machine Perception" (PICM-Net, arXiv 2512.20070)** — trit-plane progressive code with an **adaptive decode controller that stops at the level needed to keep the downstream prediction's CONFIDENCE**. *Relevance: a rate-allocation idea keyed to downstream-decision margin — rhymes with our margin-weighted byte allocation (spend bytes only where the argmax margin is thin = the annulus).*

Also verified: **Adapt-ICMH** (freeze the base codec, train a Spatial-Frequency-Modulation **adapter** for the machine task) — the "frozen base + small adapter" pattern; **Task-Aware Encoder Control (2404.04848)** (control a frozen codec's encoder per task, decoder unchanged); **Rate-Distortion-Cognition Controllable (2407.11700)**; **"Learning-Based Compression for Machines" (2409.19184)** = recent review for de-dup.

**DRAW-FROM:**
- **Decision-margin-keyed rate allocation** (PICM-Net) — formalizes our "spend bytes on the thin-margin annulus" into a known progressive-coding mechanism.
- **Frozen-base + small adapter** (Adapt-ICMH, Task-Aware Encoder Control) — our scorer is frozen by contest rule; the adapter pattern is the right minimal-bytes lever and a citeable precedent that frozen-downstream coding is a recognized regime.

**CONTRIBUTE-BACK / NOVELTY:** all of Family 5 still emits **a learned-codec RGB/latent bitstream decoded by a learned decoder**; the "task" is a soft feature/proxy loss. Ours emits a **generated geometric witness** (no learned image decoder at inflate — a deterministic program), scored by the **EXACT contest oracle's hard argmax** (not a proxy feature distance), with a **physical scene+pose+SDF generative model** as the codec. That generative-geometric-witness-for-machines combination is not represented in the VCM literature I surveyed.

---

## Family 6 — Closest prior art / ORIGINALITY check (NO-FAKE #7)

**Is anyone doing EXACTLY our thing** — a task-space witness coded as static-scene + pose-warp + SDF + conditional-residual, scored through a FROZEN downstream network (indirect-RD, not pixel fidelity)? **Verdict: NO single paper does the whole thing. The closest neighbors are split across two camps, and v2 is the unoccupied intersection.**

- **Camp A — driving-scene reconstruction/world-models with canonical-scene + warp** (the geometry is there, the objective is wrong): **Gaussian Splatting Lucas-Kanade (ICLR 2025)** explicitly "formulates deformations within a **canonical Gaussian space** + a **forward warp field** mapping canonical→spacetime" = structurally our static-canonical + warp. **WorldSplat (2509.23402)**, **BézierGS (2506.22099)**, **FlexDrive (2502.21093)**, **ReconDreamer (2411.19548)**, **DyST (2310.06020)**. ALL optimize **rendering fidelity / novel-view PSNR**, none optimizes an **indirect frozen-scorer cell metric**, and none is a **byte-minimizing codec**.
- **Camp B — codecs for machines / distributed coding** (the objective is right, the geometry is missing): Families 4 + 5 above. They condition on side-info and target downstream tasks but use **black-box learned image/latent decoders** and **no explicit scene+pose+SDF physical model**.
- **Architectural skeleton precedent:** "Implicit Neural Video Compression" 2112.11312 (warp-coords + residual) — but pixel-fidelity, learned warp, no task-space, no SDF, no WZ.

**Itemized originality (ours-original vs absorb-recode):**
- *Absorb / prior art (cite, don't claim novel):* the warp-coords+residual skeleton (2112.11312); decoder-side-info WZ binning (2106.02797/2310.16961); quantized neural displacement / SDF carriers (2504.01027); flow-matching residual-correction generators (2512.05016); canonical-space+warp scene decomposition (GS-LK ICLR'25); coding-for-machines objective (2504.02216).
- *Genuinely v2-novel (the intersection nobody occupies):* (1) the distortion is the **EXACT frozen contest-oracle's hard argmax cell** through a fixed R, not a proxy/soft feature loss or PSNR; (2) a **physical, per-class-semantic SE(3) screw warp** (Road-homography / sky-rotation / hood-identity) as the motion model — interpretable, near-zero-byte, not learned coord-modulation; (3) an **SDF carrier validated by indirect frozen-scorer survival** rather than by rendering; (4) the **3-stage prior chain (warp → SDF → openpilot-conditioned WZ residual)** with a **stored-jitter / generated-structure split** of the residual; (5) the whole thing as a **deterministic inflate-time program** (no learned image decoder ships), scored by real bytes. The honest claim is "v2 is a NOVEL COMPOSITION of well-known prior art into the unoccupied geometry×task-codec intersection," not "a new primitive." That is defensible originality under NO-FAKE #7 as long as the borrowed-substrate accounting above ships with any originality claim.

---

## Family 7 — GAUGE / CANONICALIZATION / QUOTIENT-REPRESENTATION prior art (the equivariant-ML home of our gauge layer + quotient codec #155)

Our gauge meta-layer (FEED-ji/jl: "pick the cheapest representative of the scorer-equivalence class") + the quotient codec (#155) have a precise named home in **equivariant ML canonicalization**.

**Top papers (verified):**
1. **"Equivariance with Learned Canonicalization Functions" (Kaba et al., arXiv 2211.06489, ICML 2023).** Learn a map sending each input to a **canonical representative** of its group orbit; a non-equivariant backbone then acts on the canonical form. *Relevance: this IS our gauge layer structurally — map to a representative of an equivalence class. **The decisive difference: they choose the representative for downstream ACCURACY; we choose it for CODEC COST (bytes / d_seg-through-R).** Same machine, different selection objective.*
2. **"Equivariant Frames and the Impossibility of Continuous Canonicalization" (Dym et al., arXiv 2402.16077, ICML 2024).** **VERIFIED THEOREM:** for commonly-used groups there is **no efficiently-computable choice of (unweighted) frame that preserves continuity** — unweighted frame-averaging can turn a smooth non-symmetric function into a **discontinuous symmetric** one. **The fix (also proven in-paper):** **weighted / probabilistic frames provably preserve continuity.** *Relevance: this DOES bite a naive deterministic canonical ground-frame + SE(3)-twist choice — a hard `argmin`-style fix_gauge can be discontinuous across the pose manifold, which would make the residual GENERATOR's job discontinuous (it must interpolate across a seam). This **rhymes with FEED-jo's "factorized but DISCONTINUOUS manifold."** Mitigation is known: use a **weighted/probabilistic (soft) gauge** rather than a hard argmin representative, OR confine the discontinuity to a measure-zero seam the generator never needs to interpolate. **This is a real well-posedness caveat for the gauge layer and the GPU run should adopt a soft/weighted gauge or explicitly bound the seam.***
3. **"A Canonicalization Perspective on Invariant and Equivariant Learning" (Ma et al., arXiv 2405.18378).** Proves **frames ↔ canonical forms** are two views of one design space; gives a principled way to design frames; some designs provably optimal. *Relevance: the theory tying our "gauge representative" to "frame" — lets us import the frame-design toolkit for the pose/warp gauge.*

Also verified: **"Frame Averaging" (Puny et al., 2110.03336)** the foundational frame-averaging method; **"Equivariance via Minimal Frame Averaging" (2406.07598)** — **minimal frames = the FEWEST representatives needed** — the closest existing idea to an MDL/cost-minimal gauge (averages over the smallest frame for efficiency, an efficiency criterion adjacent-to-but-not-equal-to our byte criterion); **"Lie Algebra Canonicalization" (2410.02698)** — canonicalization under arbitrary Lie groups (our warp lives in SE(3), so the Lie-group machinery is directly relevant; ties to LieFlow 2512.20043 / the algebra-coords-vs-group-element gauge choice in FEED-jp); **LieAugmenter (2506.03914)** symmetry discovery via learnable augmentations (sister to LieFlow). *(Coordinator also cited "Improved Canonicalization for Model Agnostic Equivariance" 2405.14089 — plausible but **NOT independently verified in this sweep**; flagged.)*

**DRAW-FROM:**
- **Use a SOFT / WEIGHTED gauge, not a hard argmin representative** (Dym 2402.16077) — to keep the gauge map continuous across the pose manifold so the residual generator interpolates a smooth target. This is a concrete, theory-backed design constraint for the GPU run's fix_gauge.
- **Frame-design toolkit ↔ gauge representative** (Ma 2405.18378; Puny 2110.03336) — a principled way to PARAMETERIZE the gauge representative (ground-frame + twist) instead of a hand-built cost table.
- **Minimal-frame thinking** (2406.07598) — the existing concept nearest our "cheapest representative"; adopt its minimality machinery, swap the objective from compute-efficiency to bytes/d_seg.
- **Lie-group canonicalization** (2410.02698) for the SE(3) warp gauge (group-element vs algebra-coord parameterization).

**CONTRIBUTE-BACK / NOVELTY:** the entire canonicalization literature selects the representative to **help a downstream predictor (accuracy/efficiency)**. **Canonicalizing for COMPRESSION — picking the minimum-description-length / minimum-rate representative of a task-equivalence class — is, as far as I found, NOT in this literature.** Our gauge-as-codec-canonicalization (gauge cost = MDL pick = bytes-to-survive-the-frozen-scorer) appears genuinely novel: it reframes canonicalization's selection objective from predictor-utility to **rate**, and the equivalence class is the **scorer's argmax-cell fiber** (the #155 quotient), not a symmetry-group orbit. That MDL-canonicalization-over-a-task-equivalence-class is the cleanest original framing v2 contributes back to equivariant ML. (Caveat to keep honest: must respect the impossibility theorem — the novel "cost-minimal representative" still has to be continuous-enough, i.e. weighted, to be well-posed.)

---

## RANKED top draw-froms for the imminent GPU run
1. **Generator = residual CORRECTION initialized from the warp+SDF prior, NOT generation-from-noise** (GNVC-VD 2512.05016). Smallest, most stable target; directly our X−E[X|Y]. **#1 — de-risks the generator.**
2. **Few-step DETERMINISTIC flow-matching ODE for the residual generator** (OT-NFM 2604.06413 one-step; CoD-Lite 2604.12525 distillation; corroborates LieFlow 2512.20043). Gives determinism × 30-min budget × bit-identical decode — the contest constraints diffusion can't natively meet.
3. **SOFT / WEIGHTED gauge (not hard argmin)** to keep canonicalization continuous (Dym 2402.16077). Cheap design change that prevents a discontinuous-target failure mode in the generator. **Highest value-per-effort — a $0 design constraint, not a compute cost.**
4. **Conditional VQ-VAE with decoder-only side-info Y** for the WZ residual (Whang 2106.02797; Özyılkan-Ballé 2310.16961). Canonical, tested architecture for the openpilot-conditioned residual.
5. **Coarse-base + quantized-neural-displacement / residual-grid, with in-training quantization** (Pentapati 2504.01027; CWRNN-INVR 2604.06564; NIRVANA 2212.14593). For the SDF lane carrier + ragged residual at the high-freq edge where INR spectral bias bites, with integer-decode baked in.

## Techniques that directly attack the JITTER WALL (FEED-jq, ~0.008 per-frame argmax flicker)
- **Train the generator to correct a decoded prior, sequence-level, for TEMPORAL coherence** (GNVC-VD 2512.05016 flow-matching latent refinement explicitly fixes per-frame flicker). The published win is exactly "frame-wise generative → flicker; sequence-level refinement → stable."
- **Smooth the per-frame target, store the irreducible jitter separately** (DreamSmooth 2311.01450, already in-tree FEED-jr): don't train the generator to chase per-frame SegNet noise; learn the temporally-smoothed structure, handle residual jitter as stored dither / generator stochasticity. The flow-matching ODE→SDE knob (GVCC 2603.26571) is the principled place to inject that controlled stochasticity if a deterministic generator under-fits the jitter.
- **Decision-margin-keyed allocation** (PICM-Net 2512.20070): the jitter lives in the thin-margin annulus (argmax coin-flip at margin ~0.29, FEED-jq) — spend the stored-dither bytes only there.

---

## Verification ledger (NO-FAKE)
**VERIFIED real this session (web search/fetch):** 2512.05016, 2603.26571, 2604.08329, 2604.12525, 2604.03353, 2602.09868, 2503.19604, 2508.07682, 2501.13528, 2604.06413, 2112.11312, 2604.06564, 2212.14593, 2211.06689, 2409.07414, 2409.00953, 2411.06685, 2201.04329, 2309.15848, 2309.03160, 2504.01027, 2106.02797, 2310.16961, 2401.02609, 2403.08411, 2507.17427, 2402.07997, 2504.02216, 2406.04765, 2510.15347, 2404.04848, 2407.11700, 2512.20070, 2409.19184, 2211.06489, 2402.16077, 2405.18378, 2110.03336, 2406.07598, 2410.02698, 2506.03914. Plus in-tree (prior FEEDs): 2408.13256, 2512.20043, 2311.01450. Plus driving-scene reconstruction (titles verified): GS-Lucas-Kanade (ICLR 2025), 2509.23402, 2506.22099, 2502.21093, 2411.19548, 2310.06020. Survey anchors: TechRxiv INR-for-video-compression (Aug 2025), 2402.07997, 2409.19184.

**CORRECTED mislabels from the dispatch prompt (NO-FAKE):**
- **FastFlow 2602.11105** — REAL but is **"FastFlow: Accelerating Generative Flow Matching Models with Bandit Inference" (ICLR 2026)** — a flow-matching INFERENCE-ACCELERATION method (2.6× speedup via step-skipping bandit), **NOT a video codec**. Useful as a few-step DRAW-FROM, not as a codec citation. The prompt's "FastFlow few-step [codec]" label is wrong.
- **CoD-Lite 2604.12525** — REAL but **IMAGE** compression (real-time diffusion image codec), not video as the prompt implied.
- **NeuralLVC 2604.03353** — REAL; it is **LOSSLESS** video compression (masked diffusion + temporal conditioning) — a different regime from our lossy task-codec; relevance is the temporal-conditioning mechanism only.

**UNVERIFIED / could NOT locate (do NOT cite as real):**
- **"FINO"** (referenced in FEED-jp as "full-metadata FINO-style") — **no paper named FINO** surfaced in any codec/flow-matching search. Possibly an internal shorthand or a mislabel. **Flagged — needs the operator/coordinator to supply the real ID before it is treated as prior art.**
- **"Improved Canonicalization for Model Agnostic Equivariance" 2405.14089** — coordinator-supplied; **not independently verified in this sweep** (plausible, not confirmed). Flagged.

**Pointer 0.19110 UNMOVED. All rows above are research MEANS; none is an exact-eval row.**
