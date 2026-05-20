# Standout-candidates spotlight — operator-pinned extensions

Companion to [`standout_undersold_candidates_spotlight.md`](standout_undersold_candidates_spotlight.md). That document spotlights ten technically standout candidates the [main inventory memo](asymptotic_floor_candidate_inventory.md) understates; this companion adds four candidates the operator specifically asked to surface plus one honest assessment of where the work actually sits.

Same hygiene as the primary spotlight: every score literal axis-tagged, no extrapolation from one configuration to another, honest about what is empirical vs design.

Anchor: [`commaai/comma_video_compression_challenge#110`](https://github.com/commaai/comma_video_compression_challenge/pull/110).

---

## Contents

- [B.11 Fridrich lineage — the contest is inverse steganalysis](#b11-fridrich-lineage--the-contest-is-inverse-steganalysis)
- [B.12 Wavelets at full prominence — Daubechies and Mallat](#b12-wavelets-at-full-prominence--daubechies-and-mallat)
- [B.13 Telescopic foveation revival](#b13-telescopic-foveation-revival)
- [B.14 Water-bucket filling — convex-optimal rate allocation](#b14-water-bucket-filling--convex-optimal-rate-allocation)
- [D. Where we have not fully iterated — honest assessment](#d-where-we-have-not-fully-iterated--honest-assessment)
- [E. Cross-references](#e-cross-references)

---

## B.11 Fridrich lineage — the contest is inverse steganalysis

**What this is.** Explicit acknowledgment that the comma video compression challenge sits in the inverse-steganalysis research tradition rather than the classical neural-compression tradition. The contest scorer composes a SegNet segmentation network and a PoseNet ego-motion network; the encoder's job is to perturb the source video so the scorer's downstream task accuracy is preserved with as few bytes as possible. That is the steganalysis problem inverted: instead of detecting an embedded payload, you are encoding the payload (the reconstructed frames) so the detector's downstream task signal survives.

**Why this is undersold.** The main inventory memo cites Fridrich and Yousfi's rich-models paper as one entry among many in the codec-primitives section. The lineage is much deeper than one citation. Jessica Fridrich (Binghamton DDE Lab) is the canonical inverse-steganalysis researcher; her PhD students include Yassine Yousfi (contest organizer) and Tomáš Filler (canonical syndrome-trellis coding lineage). The contest scorer's architecture and the contest's whole framing — bytes vs detector-output-fidelity tradeoff — reflect that lineage.

**Canonical references.** Holub, V., Fridrich, J., & Denemark, T. (2014). [Universal distortion function for steganography in an arbitrary domain](https://jis-eurasipjournals.springeropen.com/articles/10.1186/1687-417X-2014-1). *EURASIP J. Information Security*, 2014:1. (UNIWARD distortion.) Filler, T., Judas, J., & Fridrich, J. (2011). [Minimizing additive distortion in steganography using syndrome-trellis codes](https://ieeexplore.ieee.org/document/5740595). *IEEE TIFS*, 6(3), 920–935. (STC.) Fridrich & Kodovský (2012) and Yousfi & Fridrich (2020) are already cited in the main memo's C.8 section.

**Internal surfaces.** `src/tac/uniward_delta.py` (UNIWARD distortion implementation), `src/tac/uniward_texture.py` (texture-adaptive cost variant), `src/tac/stc_boundary_codec.py` and `src/tac/codec/pose_filler_stc_codec.py` (syndrome-trellis coding for boundary and pose payloads).

**Empirical vs design state.** UNIWARD and STC primitives are landed in the codec library; STC-Dasher arithmetic-coding maximalism is scaffolded per the main memo's C.8 entry. No standalone Fridrich-lineage substrate has been empirically anchored at contest scale yet. The lineage's contribution is structural: it frames the optimization problem the rest of the work attacks.

**Calibrated framing.** Treating UNIWARD as just another codec primitive understates the lineage. The shoutout is to acknowledge that the contest is inverse steganalysis and the rest of the inventory is downstream of that framing.

---

## B.12 Wavelets at full prominence — Daubechies and Mallat

**What this is.** Wavelet multi-resolution decomposition as a first-class hierarchical-residual primitive across the substrate stack. Daubechies orthonormal compactly supported wavelets and Mallat multi-resolution analysis underpin three distinct surfaces in the codebase: wavelet residual sidechannels (HNeRV-family bolt-ons), wavelet-multi-scale falling-rule-list preflight ranking, and Daubechies-DeVore-Fornasier-Gunturk compressive-sensing coverage estimation.

**Why this is undersold.** The main inventory memo mentions wavelets in three places — section C.7 (`Wavelet residual` SCAFFOLDED), section C.8 (canonical references), and section C.1 (Z8 Daubechies + Mallat + Rao-Ballard + Wyner-Ziv quadruple). Each mention is correct but local. What is missed is the cross-cutting role: wavelets are the canonical multi-scale partition prior underneath every multi-scale predictive-coding decision in the stack. Ingrid Daubechies sits at inner-council co-lead because that role is structural, not advisory.

**Canonical references.** Daubechies, I. (1988). [Orthonormal bases of compactly supported wavelets](https://onlinelibrary.wiley.com/doi/10.1002/cpa.3160410705). *Communications on Pure and Applied Mathematics*, 41(7), 909–996. Mallat, S. (1989). [A theory for multiresolution signal decomposition: the wavelet representation](https://ieeexplore.ieee.org/document/192463). *IEEE Trans. PAMI*, 11(7), 674–693. Daubechies, I., DeVore, R., Fornasier, M., & Gunturk, C. (2010). [Iteratively reweighted least squares minimization for sparse recovery](https://onlinelibrary.wiley.com/doi/10.1002/cpa.20303). *Communications on Pure and Applied Mathematics*, 63(1), 1–38.

**Internal surfaces.** Wavelet residual: `src/tac/hnerv_wavelet_residual.py`, `src/tac/hnerv_wavelet_apply_transform.py`, `src/tac/hnerv_wavelet_sidechannel.py`, `src/tac/wavelet_mask_codec.py`, `src/tac/wavelet_variance.py`. Compressive-sensing coverage: `src/tac/preflight_rudin_daubechies/compressive_coverage_estimator.py` recovers the full coverage manifest from K=8 representative fixtures via L1 reconstruction with bounded uncertainty O(sqrt(N/K)). Multi-scale preflight ranker: `src/tac/preflight_rudin_daubechies/wavelet_multi_scale_preflight.py`.

**Empirical vs design state.** Wavelet residual lanes are scaffolded but not paid-GPU-anchored. The compressive-sensing coverage estimator is landed and operational. The Z8 hierarchical-predictive-coding quadruple (Daubechies + Mallat + Rao-Ballard + Wyner-Ziv composed into one substrate) is design-only.

---

## B.13 Telescopic foveation revival

**What this is.** Multi-resolution foveation that allocates more bits to the focus-of-expansion region of each frame (where ego-motion concentrates dashcam scene change) and progressively fewer bits to peripheral regions. The revival reframes the foveation primitive specifically around the pose-axis structure of the contest video: the camera moves forward; the optical flow has a structured center; bits matter more where the scorer's output also varies more.

**Why this is undersold.** The main inventory memo lists foveation in C.5 alongside TT5L V2 and the FF foveation lane scaffold. The telescopic-foveation revival reframes it as a paradigm-bridge between pose-aware spatial sparsity and scorer-axis bit allocation, not just another spatial-sparsity primitive. Synthesized from research across multiple computational-vision sources earlier this session.

**Canonical references.** Gibson, J. J. (1950). *The Perception of the Visual World*. Houghton Mifflin. (Ego-motion and focus-of-expansion lineage.) Wandell, B. A. (1995). *Foundations of Vision*, Ch. 3. Sinauer. (Foveal acuity falloff.) Itti, L., & Koch, C. (2001). [Computational modelling of visual attention](https://www.nature.com/articles/35058500). *Nature Reviews Neuroscience*, 2(3), 194–203. (Saliency-driven attention.) The Daubechies wavelet foveation lineage provides the multi-resolution structure for the bit-allocation falloff.

**Internal surfaces.** `src/tac/foveation_field.py` (foveation-field generator), `src/tac/foveation_readiness.py` (readiness audit per the canonical Catalog #305 observability surface contract), telescopic-foveation revival composed with TT5L V2 (foveation + LAPose pose codec, design-only per the main memo's C.5 entry).

**Empirical vs design state.** Revival landed at the design-memo + scaffold level. TT5L V2 substrate is design-only; FF foveation lane is L0 scaffold predicated on TT5L probe outcome. No contest-axis empirical anchor yet.

**Calibrated framing.** Dashcam video has strong ego-motion structure; the focus-of-expansion region is where bits matter most for both pose and segmentation. Telescopic foveation is the natural exploit of that structure, but the prediction has not been validated empirically.

---

## B.14 Water-bucket filling — convex-optimal rate allocation

**What this is.** Water-filling (the Cover-and-Thomas convex-optimal rate allocation for parallel Gaussian channels) operationalized as a per-channel bit allocator for the codec stack. Given per-channel sensitivity from the per-pair master-gradient framework, water-filling solves the KKT conditions of the rate-allocation primal problem and emits the optimal per-channel byte budget under a global archive-size constraint.

**Why this is undersold.** The main inventory memo does not list water-filling as a standalone candidate; the work shows up only indirectly via the joint-ADMM proximal water-filling v2 helper. The water-filling family is a substantial multi-version research thread: Lane Ω-W V1 shipped with three commits and a full test suite, Lane Ω-W V2 derived an additional contest-CUDA score improvement on eligible candidates, V3 is launch-ready with a dispatch wrapper. The progression matches the cargo-cult-unwind methodology pattern.

**Canonical references.** Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*, 2nd ed., Ch. 9 (water-filling for parallel Gaussian channels). Wiley. Boyd, S., & Vandenberghe, L. (2004). [*Convex Optimization*](https://web.stanford.edu/~boyd/cvxbook/), Ch. 5 (water-filling as KKT solution of rate-allocation primal problem). Cambridge University Press.

**Internal surfaces.** `src/tac/water_filling_codec.py` (V1 canonical), `src/tac/water_filling_codec_v2.py` (V2 with bit-spend verification), `src/tac/joint_admm_proximal_water_filling_v2.py` (ADMM-coupled variant). V3 launch-ready with dispatch wrapper queued.

**Empirical vs design state.** Lane Ω-W V1 SHIPPED with empirical receipts on internal benchmarks (16/16 tests pass; per-tensor allocation matches Boyd canonical-form derivation). Lane Ω-W V2 derived improvement applies to eligible candidates per the inflate handler integration. Lane Ω-W V3 launch-ready awaits paid GPU dispatch.

**Calibrated framing.** Water-filling is the standard textbook convex-optimization primitive for rate allocation; the work operationalizes it for the contest's specific multi-channel structure. The improvement applies where the per-pair sensitivity is well-estimated; it does not replace substrate-class shifts but composes with them at the bit-allocation layer.

---

## D. Where we have not fully iterated — honest assessment

This section is the operator's specific ask: an honest acknowledgment that much of the inventoried work has not been fully iterated, implemented, or given experimental rigor. The primary spotlight memo and the main inventory memo both surface promising candidates; this section is the counterweight.

**What has been empirically validated at contest scale.** A small set of lanes carry actual paired contest-axis anchors on 1:1 contest-compliant hardware: the FEC6 fixed-Huffman k=16 stack (this PR, `0.192051 [contest-CPU]` + `0.226210 [contest-CUDA T4]`), the PR101 GOLD clone replay (`0.192845 [contest-CPU]`), the A1 substrate-engineering paired anchor (`0.19285 [contest-CPU]` with paired CUDA), the NSCS06 v6 → v7 cargo-cult-unwind result on the CUDA axis (`58.89 [contest-CUDA T4]` after 44% improvement over v6 `105.15`), the PR106 format0d latent score table (`0.205330 [contest-CUDA T4]`, single axis), and a handful of historical pre-HNeRV-family anchors. These have real empirical receipts on contest-1:1 hardware.

**What is scaffolded but not empirically anchored.** A much larger set: Z6 multi-layer FiLM predictive-coder, Z6-v2 cargo-cult-unwind redesign, Z7 LSTM/GRU temporal predictor, Z7 Mamba-2 state-space variant, Z8 hierarchical predictive coding quadruple, DreamerV3 RSSM paradigm-bridge, DP1 pretrained driving prior compositions, TT5L V2 foveation + LAPose, ATW V2-1 Faiss-IVF-PQ, NSCS06 v8 Path B, Cool-Chic, C3, wavelet residual lanes, VQ-VAE, SIREN coordinate-MLP, hybrid renderer + residual, Riemannian-Newton substrate engineering, the joint-ADMM coordinator, telescopic foveation revival, water-filling V3, and the bulk of the codec-primitive scaffolds. These have design memos and L0 / L1 substrate scaffolds; some have CPU smokes or synthetic-data smokes; few have paid contest-CUDA dispatches with paired contest-CPU anchors.

**What still needs more rigor.** Per the in-repo design-memo discipline gates, most of those scaffolds need: post-training tier-C scorer-conditional MDL density measurement on the actual landed archive (catches the C6 IBPS-class bug where pre-training density estimates miss the actual learned bottleneck behavior — empirical anchor on C6 IBPS: design-time predicted band `[0.113, 0.163]` vs measured `3.04 [contest-CUDA A10G]`, a 22x miss); cargo-cult-unwind methodology applied per the assumption-classification framework (the NSCS06 v6 → v7 44% improvement in one iteration is what this looks like when it works); per-substrate optimal-form symposium before paid dispatch (the structural protection against falsifying specific implementations and mislabeling them as paradigm-level kills); empirical predicted-vs-measured gap measurement on each design memo's predicted score band.

**What the gap actually is.** A falsification audit completed earlier in this development cycle documented 4-of-5 distinguishing-feature dispatch failures over a recent window — paid GPU empirical results often falsify specific implementations even when the underlying paradigm is intact. Solo developer plus cost-constrained dispatch budget means most candidates queue indefinitely. Promising-by-design is a much larger set than empirically-promising-on-contest-hardware. The gap between scaffold-level work and paid-GPU-validated work is substantial and is the honest characterization of the inventory.

**What this means for collaboration.** The scaffolded substrate set is research infrastructure that is structurally extensible: another researcher or team could productively pick up specific candidates (Z6 / Z8 / TT5L / wavelet residual / water-filling V3 / NSCS06 v8 Path B) and push them through the next empirical milestone. The meta-engineering — council discipline, canonical-equations registry, cargo-cult-unwind methodology, the strict-preflight gate catalog, master-gradient extractor, per-pair Lagrangian planner, cathedral autopilot ranker — is real and provides the validation harness; what is missing is per-candidate paid-GPU validation cycles. Collaboration could substantially accelerate that.

**Honest framing.** This is not an excuse and not a roadmap. It is the state of the work. The methodology and tooling and meta-engineering are real; the empirical validation per candidate is uneven; the gap is structural rather than incidental. The PR #110 submission is what one researcher operating against a constrained budget can demonstrate end-to-end; the inventory documents the rest.

---

## E. Cross-references

- **Primary spotlight memo** (sister document, 10 candidates): [`standout_undersold_candidates_spotlight.md`](standout_undersold_candidates_spotlight.md).
- **Main inventory memo** (full class-shift inventory): [`asymptotic_floor_candidate_inventory.md`](asymptotic_floor_candidate_inventory.md).
- **Submission packet**: [`commaai/comma_video_compression_challenge#110`](https://github.com/commaai/comma_video_compression_challenge/pull/110).
- **Sister library**: [`adpena/tac`](https://github.com/adpena/tac).
- **This repo**: [`adpena/comma-lab`](https://github.com/adpena/comma-lab).
