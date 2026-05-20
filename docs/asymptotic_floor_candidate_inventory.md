# Asymptotic-floor candidate inventory

This document is a snapshot inventory of class-shift candidate substrates trained, scaffolded, or designed against the comma video compression challenge while building the FEC6 + fixed-Huffman bolt-ons submitted as PR #110. It is technical context for the work that exists beyond the submission packet — not a roadmap, not a benchmark suite, not a runtime dependency, and not a claim about how close any candidate comes to a theoretical floor.

Anchor: [`commaai/comma_video_compression_challenge#110`](https://github.com/commaai/comma_video_compression_challenge/pull/110).

---

## Contents

- [A. Where the local floor sits](#a-where-the-local-floor-sits)
- [B. Empirically-run anchors](#b-empirically-run-anchors)
- [C. Class-shift candidate inventory](#c-class-shift-candidate-inventory)
  - [C.1 Predictive-coding world models](#c1-predictive-coding-world-models-rao--ballard-1999-hafner-dreamerv3-2023)
  - [C.2 Cooperative-receiver framings](#c2-cooperative-receiver-framings-atick--redlich-1990-1992)
  - [C.3 Information-Bottleneck framings](#c3-information-bottleneck-framings-tishby--zaslavsky-2015)
  - [C.4 Pretrained driving priors](#c4-pretrained-driving-priors)
  - [C.5 Pose-axis, foveation, spatial-sparse](#c5-pose-axis-foveation-spatial-sparse-gibson-1950-lapose)
  - [C.6 NeRV-family beyond HNeRV](#c6-nerv-family-beyond-hnerv-chen-et-al-2023-lineage)
  - [C.7 Non-NeRV substrate architectures](#c7-non-nerv-substrate-architectures)
  - [C.8 Codec primitives and entropy coding](#c8-codec-primitives-and-entropy-coding)
  - [C.9 Self-compression family](#c9-self-compression-family)
  - [C.10 Composition substrates + stacking](#c10-composition-substrates--stacking-carmack-hotz-strip-everything-lineage)
  - [C.11 Higher-order optimization framings](#c11-higher-order-optimization-framings)
- [D. Cost-efficiency and hardware](#d-cost-efficiency-and-hardware)
- [E. What is in this repo beyond the submission packet](#e-what-is-in-this-repo-beyond-the-submission-packet)
- [F. What is stuck](#f-what-is-stuck)
- [G. Caveats](#g-caveats)
- [H. Reproducibility and cross-links](#h-reproducibility-and-cross-links)

---

## A. Where the local floor sits

The HNeRV-family cluster (@AaronLeslie138 PR #95 / @EthanYangTW PR #98 / @BradyMeighan PR #100 / @SajayR PR #101 / @EthanYangTW PR #102 / @rem2 PR #103 / this PR #110) sits within roughly `0.0008` of each other on the CPU axis the leaderboard ranks. Within-HNeRV-family the local floor appears effectively reached; further within-class bolt-ons increasingly trade smaller distortion savings against larger rate costs.

Class-shift to a different substrate paradigm is the visible next direction. Multiple theoretical-floor analyses give different estimates of where the next floor sits — Shannon R(D) bound from the contest's scoring formula, Blahut-Arimoto on conditional entropy of the scorer outputs, Dykstra-feasibility on the intersection of rate / segmentation / pose constraint polytopes, scorer-conditional MDL density via the post-training tier-C method. None agree on a single number, so "where the floor is" is treated as a band rather than a point. The rest of this document is what has been tried to push into that band.

---

## B. Empirically-run anchors

End-to-end measurements actually run on contest-1:1 hardware. Score literals are axis-tagged. CPU evaluations are on `linux_x86_64_cpu` Modal containers matching the GitHub-Actions `ubuntu-latest` runner family; CUDA evaluations are on Modal Tesla T4. Anything tagged `[advisory only]` or `[macOS-CPU advisory]` is local development signal — not used for promotion, ranking, or submission decisions.

| Lane | Class | Score | Status |
|---|---|---|---|
| `hnerv_fec6_fixed_huffman_k16` (this PR) | within-HNeRV-family bolt-on stack | `0.192051 [contest-CPU]`, `0.226210 [contest-CUDA T4]` | EMPIRICAL, paired |
| `hnerv_ft_microcodec` PR #101 reproduction | within-HNeRV-family | `0.192845 [contest-CPU]` recomputed from bot eval | EMPIRICAL, sister anchor |
| `lane_a1_…` paired anchor | A1 substrate engineering | `0.19285 [contest-CPU]` paired CUDA | EMPIRICAL, paired |
| `pr106_format0d_latent_score_table` | within-HNeRV-family CUDA-side | `0.205330 [contest-CUDA T4]` | EMPIRICAL, single-axis |
| Earlier within-HNeRV-family iterations | bolt-on | multiple `[contest-CUDA T4]` anchors `0.95`–`1.45` | EMPIRICAL, retired-config (pre-PR101-substrate) |
| `c6_ibps` 50ep IB-bottleneck smoke | information-bottleneck | `final_score = 3.04 [contest-CUDA A10G]` vs design-time predicted band `[0.113, 0.163]` | FALSIFIED-at-this-latent-dim. 24-dim IB bottleneck collapses segmentation (`d_seg` dominates ~86%). Reactivation: latent-dim sweep `{48, 96, 192}` + post-training tier-C density re-measurement |
| `nscs06` Carmack-Hotz Strip-Everything | composition substrate | v6 `105.15 [contest-CUDA T4]` → v7 `58.89 [contest-CUDA T4]` after one cargo-cult-unwind iteration (44% improvement) | EMPIRICAL design-time validation; not paired contest-CPU yet; reactivation = continue unwind ladder |
| `lane_g_v3` historical | within-class | `1.05 [contest-CUDA T4]` | EMPIRICAL historical (pre-HNeRV-family arrival) |
| `apogee_int4` PTQ smoke | low-bit weight quantization | `1.42866394 [contest-CUDA T4]` | FALSIFIED-at-naive-PTQ; reactivation = QAT / LSQ / per-channel / smaller blocks / outlier handling, not yet attempted on post-HNeRV substrate |

The honest summary: outside the HNeRV-family local cluster, every other paradigm empirically tested either falsifies at a specific implementation config (not at the paradigm class) or has not yet been pushed end-to-end to a paired CPU + CUDA anchor on contest-1:1 hardware.

---

## C. Class-shift candidate inventory

Grouped by paradigm class. Each section cites the canonical reference paper(s), the inspiration / "why" for the class, an inventory table of candidates, and where useful an internal-research link. Predicted-band numbers are deliberately omitted for any candidate that has not been empirically anchored at contest scale — the cost of design-time band claims that fail to validate (the C6 IBPS case in Section B above) is high enough that understatement is preferred.

### C.1 Predictive-coding world models (Rao & Ballard 1999; Hafner DreamerV3 2023)

Canonical references:

- Rao, R. P. N., & Ballard, D. H. (1999). [Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects](https://www.nature.com/articles/nn0199_79). *Nature Neuroscience*, 2(1), 79–87.
- Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2023). [Mastering Diverse Domains through World Models](https://arxiv.org/abs/2301.04104). *arXiv:2301.04104* (DreamerV3).

**Why**: the contest video is 20 seconds of ego-motion dashcam footage. The next-frame prior is dominated by translational + rotational optical flow, which Rao-Ballard hierarchical predictive coding is structurally designed for. A world model that predicts the next frame from ego-motion-conditioned latent dynamics has lower scorer-conditional entropy than per-frame independent encoding.

> "Hierarchical predictive coding offers a unifying account of visual processing in the cortex, in which top-down predictions are subtracted from bottom-up inputs and only the residual error is propagated upward." — Rao & Ballard (1999)

| Candidate | Status | One-line summary |
|---|---|---|
| `Z6 multi-layer FiLM (depth=3, ~300K params)` | SCAFFOLDED end-to-end | Primary predictive-receiver substrate. Wave 2 smoke fired but ran smoke-mode trainer path due to driver mode-routing bug (now closed at gate level). Reactivation: re-fire full-mode 100ep canary. |
| `Z6-v2 cargo-cult-unwind redesign` | DESIGN-ONLY | Response to council critique that original Z6 inherited canonical capacity assumptions without testing them. |
| `Z7 LSTM/GRU temporal predictor` | DESIGN-ONLY | Sister to Z6; recurrent temporal predictor variant. |
| `Z7-as-Mamba-2 (SSM substrate)` | DESIGN-ONLY | Mamba-2 selective state-space model variant. Recent deep-research recommended it as the most direct paradigm-bridge to current SSM results. |
| `Z8 hierarchical predictive coding` | DESIGN-ONLY | Canonical quadruple: Daubechies wavelet hierarchical prior + Mallat multi-resolution + Rao-Ballard hierarchy + Wyner-Ziv side-information composed into one substrate. Untested at contest scale. |
| `DreamerV3 RSSM categorical-posterior paradigm-bridge` | DESIGN-ONLY | Tiered cost ladder. Tightened predicted band `[0.20, 0.40]` is honest inherited uncertainty; post-training tier-C measurement is the canonical validator and has not been run. |

Additional canonical reference for Mamba-2 SSM:

- Dao, T., & Gu, A. (2024). [Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality](https://arxiv.org/abs/2405.21060). *arXiv:2405.21060*.

**Internal research**: per-pair master-gradient `Taylor + Cauchy-Schwarz` bound (canonical equation slot `per_pair_master_gradient_score_impact_taylor_v1`) gives the upper-bound score-impact framing the predictive-coding residual rate must beat per pair to be competitive.

### C.2 Cooperative-receiver framings (Atick & Redlich 1990, 1992)

Canonical references:

- Atick, J. J., & Redlich, A. N. (1990). [Towards a theory of early visual processing](https://www.mitpressjournals.org/doi/10.1162/neco.1990.2.3.308). *Neural Computation*, 2(3), 308–320.
- Atick, J. J., & Redlich, A. N. (1992). [What does the retina know about natural scenes?](https://www.mitpressjournals.org/doi/10.1162/neco.1992.4.2.196). *Neural Computation*, 4(2), 196–210.

**Why**: the contest scorer (SegNet + PoseNet weights) is public and fixed. The decoder shares those weights with the encoder. Per Atick-Redlich's cooperative-receiver theorem, the encoder should optimize against the decoder-and-scorer joint, not just the decoder. Bytes that exploit the scorer as side-information are bytes the marginal-source rate did not need to pay for.

> "The natural choice for a measure of information transmission is the mutual information between the input and the output of the visual system, which the system should maximize subject to its physical constraints." — Atick & Redlich (1990)

| Candidate | Status | One-line summary |
|---|---|---|
| `Z4 cooperative-receiver loss` | SCAFFOLDED | Bolt-on objective term. Full path is council-gated. |
| `ATW V1 (Atick-Tishby-Wyner triple)` | DESIGN-ONLY | Folded into ATW V2. |
| `ATW V2` | SCAFFOLDED L1 | First conditioning probe returned `INDEPENDENT` verdict (`MI = 0.006385 bits/symbol`, two orders below the meaningful-conditioning threshold). DEFERRED-pending-research. Reactivation: trained ATW residual probe or substrate-native scorer-logit sketch instead of an opaque conditioning channel. |
| `ATW V2-1 with Faiss IVF-PQ per-region SegNet softmax channel` | DESIGN-ONLY | Probe budget pre-empted by ATW V2's INDEPENDENT verdict. |

Sister-canonical reference for the Wyner half of "ATW":

- Wyner, A. D., & Ziv, J. (1976). [The rate-distortion function for source coding with side information at the decoder](https://ieeexplore.ieee.org/document/1055508). *IEEE Transactions on Information Theory*, 22(1), 1–10.

**Internal research**: the Wyner-Ziv deliverability proof builder at `tac.codec.wyner_ziv_layer` rejects research-sidecar phantom savings; ATW V2's `INDEPENDENT` probe verdict is recorded in the canonical probe-outcomes ledger per Catalog #313 so the apparatus does not re-fire the same probe inside the 30-day staleness window.

### C.3 Information-Bottleneck framings (Tishby & Zaslavsky 2015)

Canonical references:

- Tishby, N., Pereira, F. C., & Bialek, W. (1999). [The Information Bottleneck Method](https://arxiv.org/abs/physics/0004057). *arXiv:physics/0004057*.
- Tishby, N., & Zaslavsky, N. (2015). [Deep Learning and the Information Bottleneck Principle](https://arxiv.org/abs/1503.02406). *arXiv:1503.02406*.

**Why**: the optimal codec compresses to the minimum sufficient statistic for the scorer's outputs, not for pixel reconstruction. The IB framework gives a principled rate-distortion-information trade via the Lagrangian `L = I(X;T) - beta * I(T;Y)`, where `T` is the compressed representation, `X` the input frame, and `Y` the scorer's relevant outputs. A `beta_ib` sweep enumerates the rate-distortion frontier the contest scorer cares about.

> "The information bottleneck method finds a representation T that is maximally informative about the relevance variable Y while compressing the input X." — Tishby & Zaslavsky (2015)

| Candidate | Status | One-line summary |
|---|---|---|
| `C6 IBPS (canonical Path B quadruple)` | EMPIRICAL anchor falsified | Specific 24-dim latent at 50ep falsified at `3.04 [contest-CUDA A10G]` vs predicted `[0.113, 0.163]`. Paradigm not killed — post-training tier-C re-measurement surfaced the structural reason (segmentation collapse). Reactivation: latent-dim sweep + `beta_ib` calibration before any further paid dispatch. |
| `C6 IBPS β_ib sweep` | DESIGN-ONLY | Queued behind latent-dim sweep. |
| `E4 MDL-IBPS` | DESIGN-ONLY | Queued behind C6 IBPS reactivation. |
| `Tishby IB-pure` | DESIGN-ONLY | Queued behind C6 IBPS reactivation. |

**Internal research**: the C6 IBPS empirical falsification is recorded as canonical Provenance per Catalog #323; the `mps_drift_architecture_class_dependent_v1` canonical-equation slot tracks per-architecture noise-source calibration so an IB-class architecture's MPS-vs-CUDA gap is treated as architecture-dependent rather than universal.

### C.4 Pretrained driving priors

Canonical reference for the underlying dataset:

- Schafer, H., Santana, E., Haden, A., & Biasini, R. (2018). [A Commute in Data: The comma2k19 Dataset](https://arxiv.org/abs/1812.05752). *arXiv:1812.05752*.

**Why**: compress against an out-of-distribution dashcam codebook trained on comma2k19, rather than against the contest video itself. The codebook captures driving-domain priors (lane lines, sky region, vehicles, road texture) that the contest scorer rewards faithful preservation of, without leaking contest-video information into the codebook.

| Candidate | Status | One-line summary |
|---|---|---|
| `DP1 Phase 2` | SCAFFOLDED | Comma2k19 local-cache canonical helper, codebook provenance metadata, deliverability proof builder, OOD-codebook + Wyner-Ziv composition path. Full path pending Phase 2 council ratification. |
| `DP1 + PR101 composition` | DESIGN-ONLY | Pending empirical anchor on DP1 alone. |

**Internal research**: the canonical Comma2k19 local-cache helper enforces SHA-pinned chunk integrity per Catalog #213; the DP1 codebook provenance metadata is required per Catalog #210 so license tags and downstream Wyner-Ziv consumers can audit the codebook's origin.

### C.5 Pose-axis, foveation, spatial-sparse (Gibson 1950; LAPose)

Canonical reference:

- Gibson, J. J. (1950). *[The Perception of the Visual World](https://archive.org/details/perceptionofvisu00gibs)*. Houghton Mifflin. (Ego-motion + optical flow + focus-of-expansion).

**Why**: the pose axis of the score and the spatially-non-uniform information density of the dashcam frame both reward representations that allocate bits non-uniformly. The focus-of-expansion (FOE) under forward motion concentrates pose-relevant information into a narrow image region; bits should be allocated proportionally.

| Candidate | Status | One-line summary |
|---|---|---|
| `TT5L V2 foveation + LAPose redesign` | DESIGN-ONLY | Reformulated after a `REFUSE` verdict on V1's monolithic ~3000-epoch training plan; V2 splits into staged probes. |
| `FF foveation lane scaffold` | L0 scaffold | Predicated on TT5L probe outcomes. |
| `RAFT-derived poses` | SCAFFOLDED | Optical-flow-derived pose channel. |
| `LAPose pose codec` | SCAFFOLDED | Pose-axis dedicated codec. |
| `SAR coherent pose pairs` | DESIGN-ONLY | Sister to LAPose. |

Sister-canonical reference for the optical-flow component:

- Teed, Z., & Deng, J. (2020). [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow](https://arxiv.org/abs/2003.12039). *arXiv:2003.12039*.

**Internal research**: the per-pair master-gradient framework (canonical equation slots `per_pair_master_gradient_score_impact_taylor_v1` + `master_gradient_locality_violation_by_codec_v1`) provides the basis for per-pair bit allocation that pose-axis foveation requires.

### C.6 NeRV-family beyond HNeRV (Chen et al. 2023 lineage)

Canonical references:

- Chen, H., Gwilliam, M., Lim, S. N., & Shrivastava, A. (2023). [HNeRV: A Hybrid Neural Representation for Videos](https://arxiv.org/abs/2304.02633). *arXiv:2304.02633*.
- Chen, H., He, B., Wang, H., Ren, Y., Lim, S. N., & Shrivastava, A. (2021). [NeRV: Neural Representations for Videos](https://arxiv.org/abs/2110.13903). *arXiv:2110.13903*.

**Why**: HNeRV is the substrate the entire medal cluster (PR #95 / #100 / #101 / #102 / #103) builds on. Variants of the implicit-neural-representation family that swap the underlying architecture (temporal, block-decomposed, frequency-domain, feature-grid, deformable) are the most direct paradigm-adjacent candidates.

| Candidate | Status | One-line summary |
|---|---|---|
| `TCNeRV` | SCAFFOLDED | Temporal-convolutional NeRV. |
| `BlockNeRV` | SCAFFOLDED | Block-decomposed NeRV. |
| `FFNeRV` | SCAFFOLDED | Feature-grid NeRV. |
| `DSNeRV` | SCAFFOLDED | Deformable-scene NeRV. |
| `HiNeRV` | SCAFFOLDED | Hierarchical NeRV. |
| `e_nerv` | SCAFFOLDED | Sister NeRV variant. |
| `ego_nerv` | SCAFFOLDED | Ego-motion-conditioned NeRV. |
| `nervdc` | SCAFFOLDED | Sister NeRV variant. |

Several of these hit dispatch-time API crashes (Wave 3) and are tagged `research_only` pending reactivation through the canonical Phase 2 council flow — they are not treated as paradigm-falsified.

### C.7 Non-NeRV substrate architectures

Canonical references:

- Sitzmann, V., Martel, J. N. P., Bergman, A. W., Lindell, D. B., & Wetzstein, G. (2020). [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661). *arXiv:2006.09661* (SIREN).
- van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937). *arXiv:1711.00937* (VQ-VAE).
- Ladune, T., Philippe, P., Henry, F., & Bonnet, V. (2023). [COOL-CHIC: Coordinate-based Low Complexity Hierarchical Image Codec](https://arxiv.org/abs/2212.05458). *arXiv:2212.05458*.
- Kim, H., Bauer, M., Theis, L., Schwarz, J. R., & Dupont, E. (2023). [C3: High-performance and low-complexity neural compression from a single image or video](https://arxiv.org/abs/2312.02753). *arXiv:2312.02753*.

**Why**: not all viable codec substrates live in the NeRV family. Coordinate-based MLPs, periodic-activation networks, discrete-latent codecs, and learned hierarchical codecs each represent different inductive biases over the spatio-temporal signal.

| Candidate | Status | One-line summary |
|---|---|---|
| `Cool-Chic` | DESIGN-ONLY | Open export-contract gate. |
| `C3` | DESIGN-ONLY | Open export-contract gate. |
| `Wavelet residual` | SCAFFOLDED | Daubechies wavelet residual sidechannel. |
| `Hybrid renderer + residual` | SCAFFOLDED | Composition substrate. |
| `SIREN coordinate MLP` | SCAFFOLDED | Periodic-activation INR. |
| `VQ-VAE` | SCAFFOLDED | Discrete-latent codec; van den Oord 2017 precedent. |
| `Grayscale-LUT (Selfcomp PR #56 extension)` | DESIGN-ONLY | Beyond what PR #56 already shipped. |
| `Quantizr-faithful reimplementation` | Historical | `[contest-CUDA T4]` anchors `0.33`–`0.41`. Not the current frontier. |
| `Diffusion renderer` | DESIGN-ONLY | Speculative. |

### C.8 Codec primitives and entropy coding

Canonical references for the steganalysis lineage (relevant because the contest scorer IS an inverse-steganalysis primitive):

- Fridrich, J., & Kodovský, J. (2012). [Rich Models for Steganalysis of Digital Images](https://ieeexplore.ieee.org/document/6197267). *IEEE Transactions on Information Forensics and Security*, 7(3), 868–882.
- Yousfi, Y., & Fridrich, J. (2020). [An Intriguing Struggle of CNNs in JPEG Steganalysis and the OneHot Solution](https://ieeexplore.ieee.org/document/9028352). *IEEE Signal Processing Letters*, 27, 830–834.

Canonical references for the neural compression lineage:

- Ballé, J., Laparra, V., & Simoncelli, E. P. (2017). [End-to-end Optimized Image Compression](https://arxiv.org/abs/1611.01704). *arXiv:1611.01704*.
- Ballé, J., Minnen, D., Singh, S., Hwang, S. J., & Johnston, N. (2018). [Variational Image Compression with a Scale Hyperprior](https://arxiv.org/abs/1802.01436). *arXiv:1802.01436*.

Wavelet multi-resolution canonical references:

- Daubechies, I. (1988). [Orthonormal bases of compactly supported wavelets](https://onlinelibrary.wiley.com/doi/10.1002/cpa.3160410705). *Communications on Pure and Applied Mathematics*, 41(7), 909–996.
- Mallat, S. (1989). [A theory for multiresolution signal decomposition: the wavelet representation](https://ieeexplore.ieee.org/document/192463). *IEEE Trans. Pattern Anal. Mach. Intell.*, 11(7), 674–693.

**Why**: every byte that gets allocated to the archive's entropy-coded streams is a byte the rate term charges for. Codec primitives that exploit known structure in the source (texture-adaptive cost like UNIWARD, side-information-aware coding like Wyner-Ziv, hierarchical priors like Ballé's hyperprior, syndrome-trellis coding from steganography) are the entropy-side counterpart to substrate-side architecture changes.

| Candidate | Status | One-line summary |
|---|---|---|
| `tac.codec.wyner_ziv_layer` (canonical) | LANDED | Paired with deliverability proof builder that distinguishes truly deliverable side-information savings from research-sidecar phantom savings (Catalog #321 / #322). |
| Hierarchical Wyner-Ziv composition | DESIGN-ONLY | Canonical Daubechies + Mallat + Rao-Ballard + Wyner-Ziv quadruple (same as Z8 in C.1). |
| STC-Dasher arithmetic-coding maximalism | SCAFFOLDED | Syndrome-trellis coding pushed to arithmetic-coder limit. |
| Ballé hyperprior (CompressAI primitives registered) | SCAFFOLDED | Canonical inventory entry. |
| Selfcomp block-FP (PR #56 lineage) | SCAFFOLDED | Block-floating-point weight compression. |
| Hessian-block-FP | SCAFFOLDED | Sister to block-FP using Hessian-weighted blocks. |
| UNIWARD texture-aware encoding | SCAFFOLDED | Fridrich-lineage texture-adaptive cost; targets the scorer's blind-spot regions per the inverse-steganalysis framing. |

**Internal research**: the per-byte leverage canonical equation (slot `per_byte_leverage_uniformly_distributed_v1`) empirically established that per-byte optimization saturates quickly for entropy-coded archives (PR101 top-1% byte leverage = 6.4%); substrate-class shifts dominate per-byte edits on entropy-coded archives. The `master_gradient_locality_violation_by_codec_v1` canonical equation enforces that raw-byte master-gradient is invalid for entropy-coded archives; post-decompress grain is the canonical basis.

### C.9 Self-compression family

Canonical reference:

- Hinton, G., Vinyals, O., & Dean, J. (2015). [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). *arXiv:1503.02531*.

**Why**: the SegNet weights distilled into a smaller surrogate would amortize per-pair gradient computation cost across many iterations. Iterative magnitude pruning (IMP) is the classic complement: pruning toward the lottery-ticket subnetwork that retains scorer accuracy.

| Candidate | Status | One-line summary |
|---|---|---|
| `SC++ (SegMap + KL distill)` | SCAFFOLDED | KL-distilled SegNet surrogate. |
| `MDL FP4 TTO` | SCAFFOLDED | MDL-optimal FP4 quantization via test-time optimization. |
| `lane_17_imp` iterative magnitude pruning | L2 SCAFFOLDED | Council symposium deferred dispatch pending cycle-0 empirical regression + score-gradient saliency sidecar. |

### C.10 Composition substrates + stacking (Carmack-Hotz Strip-Everything lineage)

**Why**: composing N independent substrate primitives into a single dispatch can be additive in score reduction if the substrates are orthogonal in their bit-allocation axes. Carmack-Hotz Strip-Everything is the canonical composition example: per-class chroma anchor + grayscale-LUT + arithmetic-coded delta + N=1 sample composition. The composition substrate is where Dykstra-feasibility on the intersection of constraint polytopes determines whether the composition is achievable rather than just predicted.

| Candidate | Status | One-line summary |
|---|---|---|
| `NSCS06 v6 → v7 → v8 (Strip-Everything, per-class chroma anchor)` | EMPIRICAL anchor at v7 | `58.89 [contest-CUDA T4]` (44% improvement over v6 `105.15`) in ONE cargo-cult-unwind iteration. Variant C queued. Paradigm intact; v6 falsification is implementation-level. |
| `NSCS01 nullspace-split renderer` | SCAFFOLDED | Renderer split along nullspace direction. |
| `NSCS02 downsampled renderer` | SCAFFOLDED | Downsampled-input renderer. |
| `NSCS03 Ballé end-to-end joint codec` | SCAFFOLDED | Ballé hyperprior + end-to-end-trained renderer. |
| `stack_of_stacks` | SCAFFOLDED | Recipe-level composition framework. |
| `S2SBS byte-stuffing` | SCAFFOLDED | Sample-to-sample byte-stuffing. |
| `SAR composition` | (see C.5) | Pose-axis composition. |

**Internal research**: the canonical cargo-cult-unwind methodology (NSCS06 v6 → v7) is the canonical example of one-iteration paradigm-rescue per CLAUDE.md "Forbidden premature KILL without research exhaustion" — implementation-level falsification at v6 unwound 4-of-7 cargo-culted assumptions to land v7 at 44% improvement.

### C.11 Higher-order optimization framings

Canonical references:

- Boyd, S., Parikh, N., Chu, E., Peleato, B., & Eckstein, J. (2011). [Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers](https://www.nowpublishers.com/article/Details/MAL-016). *Foundations and Trends in Machine Learning*, 3(1), 1–122. (ADMM).
- Dykstra, R. L. (1983). [An algorithm for restricted least squares regression](https://www.jstor.org/stable/2288033). *J. Amer. Statist. Assoc.*, 78(384), 837–842. (Alternating projections for convex feasibility).

**Why**: substrate engineering is a constrained optimization problem. ADMM, Dykstra alternating projections, and Riemannian manifold-aware optimization are the canonical algorithmic primitives for navigating multi-constraint feasibility (rate ≤ R, segmentation ≤ S, pose ≤ P, archive size ≤ A).

| Candidate | Status | One-line summary |
|---|---|---|
| `Riemannian-Newton substrate engineering` | DESIGN-ONLY | Manifold-aware second-order optimization for substrate weight space. |
| `Tropical d_seg solver` | DESIGN-ONLY | Sister to Riemannian-Newton; tropical-algebra approach to segmentation distortion. |
| `Joint-ADMM coordinator` | SCAFFOLDED | Cross-substrate consensus framework following Boyd's ADMM. |
| `3-set Venn classification` (high-pair-invariant / pair-specific / per-pair) | EMPIRICAL classifier | Drives per-pair master-gradient framework; informs which bits live in which composition axis. |

**Internal research**: per-pair Pareto envelope cathedral consumer ingests the 3-set Venn classification + per-pair master-gradient Cauchy-Schwarz bound (canonical equation slot `per_pair_master_gradient_score_impact_taylor_v1`) to emit per-pair candidate-rank annotations downstream of the cathedral autopilot ranker.

### C.12 Selector + microcodec extensions (orthogonal-overlay class)

Canonical references for the orthogonal-overlay paradigm + per-byte sensitivity grounding:

- Wang, F., & Rudin, C. (2015). [Falling Rule Lists](https://proceedings.mlr.press/v38/wang15a.html). *AISTATS*, 1013–1022. (selector-as-falling-rule-list canonical discipline; sister of Catalog #274).
- Huffman, D. A. (1952). [A method for the construction of minimum-redundancy codes](https://ieeexplore.ieee.org/document/4051119). *Proc. IRE*, 40(9), 1098–1101.

**Why**: per the 2026-05-20 cross-candidate sensitivity comparative analysis ([`docs/per_byte_sensitivity_comparative_analysis_methodology.md`](per_byte_sensitivity_comparative_analysis_methodology.md)) the HNeRV-class backbone is **class-saturated** at the 0.19xxx medal cluster (Pearson seg ρ=0.961 + pose ρ=0.971 between PR101 and fec6 on the shared 178,158-byte backbone). The empirical path forward on this class is **orthogonal selector + microcodec overlays** that add bytes ON TOP of the saturated backbone, not refinement OF the backbone itself. fec6 frontier (+259 bytes; +794 ppm reduction) is the proof-of-concept; the SELECTOR-EXTENSIONS class generalizes the discovery.

| Candidate | Status | One-line summary |
|---|---|---|
| `fec6 fixed-Huffman k=16 frame-exploit selector` (PR110) | EMPIRICALLY VALIDATED | `0.19205 [contest-CPU]` on archive sha `6bae0201` — the canonical anchor. |
| `Per-class Huffman selector with k-sweep` | DESIGN-ONLY | Generalize fec6's k=16 to learned-per-class k via Wang-Rudin falling-rule-list discipline. |
| `Frame-pair-specific selector (vs frame-exploit-only)` | DESIGN-ONLY | Extend selector granularity from frame to pair. |
| `Arithmetic-coded selector (vs fixed-Huffman)` | DESIGN-ONLY | STC-Dasher-style entropy-coded selector overhead. |
| `Two-tier selector + microcodec composition` | DESIGN-ONLY | Selector chooses among k microcodecs each tuned for a different per-pair regime. |
| `Selector-aware bit-allocator` (per-class Pareto routing) | DESIGN-ONLY | Bit-allocator that respects the per-class selector signal. |
| `PR106 score-table as orthogonal-codec stack on HNeRV backbone` | DESIGN-ONLY | Per canonical equation #9 `cross_codec_super_additive_orthogonality_predictor_v1`; PR106 vs HNeRV-family pairs classify SUPER_ADDITIVE with top-K Jaccard 0.000. |

**Internal research**: the canonical equation `hnerv_backbone_sensitivity_saturated_across_medal_cluster_v1` (per Catalog #344 commit `80484241f`) empirically established that the entire +794 ppm fec6 advantage is concentrated in selector overhead, not backbone refinement. The sister equation `cross_codec_super_additive_orthogonality_predictor_v1` predicts that PR106-vs-HNeRV-family compositions are orthogonal-codec (Jaccard 0.000) so PR106's score-table stacking on an HNeRV backbone is a structural SUPER_ADDITIVE candidate. Together these establish the SELECTOR-EXTENSIONS class as a first-class attack surface independent of the within-backbone substrate engineering covered by C.6–C.10. The `tac.cathedral_consumers.auto_trigger_similarity_after_master_gradient_anchor_consumer` (Catalog #335-compliant; per the per-byte sensitivity methodology) auto-pre-classifies new master-gradient anchors against this taxonomy so candidate generation surfaces SELECTOR-EXTENSIONS as a default ranker dimension.

---

## D. Cost-efficiency and hardware

| Hardware | Rate | Use |
|---|---|---|
| Vast.ai RTX 4090 | `~$0.25/hr` | Primary; cheapest 24GB CUDA. |
| Modal Tesla T4 | `~$0.59/hr` | Fallback or T4-only recipes. |
| Modal A100 | `~$1.50/hr` | VRAM > 24GB or `min_smoke_gpu: A100`. |
| Lightning Studio free tier | opportunistic | 22h/month subscription cap. |
| Local Linux x86_64 Modal CPU | contest-1:1 | Only authoritative CPU axis (GHA `ubuntu-latest` match). |
| Local macOS CPU (M-series) | free | Advisory proxy; within `~6e-6` of Linux x86_64 on one prior submission. Tagged `[macOS-CPU advisory]`. Never promoted. |
| Local MPS | free | Development only. PoseNet drift vs CUDA `~23x`. Never used for ranking, promotion, or submission. |

A typical class-shift training run on a substrate the size of the HNeRV cluster costs `$15`–`$300` per attempt. Most spend goes into smokes — that's where implementation-level cargo-cult assumptions get falsified before the full meter starts.

---

## E. What is in this repo beyond the submission packet

The submission's runtime is intentionally small and self-contained in PR #110 under `submissions/hnerv_fec6_fixed_huffman_k16/` (six files: `inflate.sh`, `inflate.py`, and four local `src/` modules). The repo carries the rest of the research/build apparatus.

### E.1 Tooling

| Component | What it does |
|---|---|
| Cathedral autopilot ranker | Ingests candidate substrates → ranked dispatch recommendations. |
| Per-pair master-gradient + Lagrangian-dual per-pair treatment plan | Per-pair recommendations the cathedral autopilot consumes. |
| Canonical equations registry | fcntl-locked append-only JSONL of empirically-calibrated equations (e.g. `brotli_cascade_bounded_per_stream_v1`, `mps_drift_architecture_class_dependent_v1`). |
| Modal call-id ledger + harvester | Closes the spawn-and-lose failure mode on Modal's detached function-call cache. |
| Subagent crash-resume checkpoint | fcntl-locked append-only JSONL for resume-from-disk. |
| Frontier pointer canonical helper | Single source of truth for local-best CPU/CUDA anchors; auto-refreshed on dispatch completion. |
| Per-X optimal codec planner | DuckDB unification for cross-pair-sensitivity queries. |
| Probe-outcomes ledger | Prevents re-firing already-adjudicated dispatches within a 30-day window. |
| Canonical Provenance helper | Attached to every score-claiming row in persisted state. |
| Wyner-Ziv deliverability proof builder | Distinguishes deliverable side-information savings from research-sidecar phantom savings. |
| Pre-dispatch adversarial review automation | External reviewer pass before any paid dispatch above a cost threshold. |
| Master-gradient x-ray + asymptotic-pursuit readiness assessment | Scans registry against dispatch-protocol gates; surfaces top-ranked actually-ready candidate. |
| ~300 STRICT preflight gates | Fail-closed on empirically-encountered bug classes. |

### E.2 Methodology and discipline

Cargo-cult unwind methodology (NSCS06 v6→v7 = 44% improvement in one iteration via hard-earned-vs-cargo-culted assumption classification). Per-deliberation explicit assumption surfacing. Per-substrate adversarial-council symposium discipline before paid dispatch above threshold; recursive adversarial review with three-clean-pass counter; 4-tier council hierarchy with explicit attendees, quorum, tie-break. Bug-class extinction at orthogonal surfaces (design-memo / runtime-effect / per-feature contract / promotion gate / retirement gate / council discipline / iteration discipline / post-training validation). Sister library [`adpena/tac`](https://github.com/adpena/tac) holds task-aware-compression research primitives and reusable tooling used during development; it is not required by the PR #110 submission runtime.

---

## F. What is stuck

Each class-shift candidate is stuck on one or more of:

1. **Substrate-engineering cost.** A new architecture from scratch against the contest scorer is `$50`–`$500` per honest attempt. Cheap smokes triage some failures, but smokes do not always disambiguate implementation falsification from paradigm falsification.
2. **Cargo-cult-vs-hard-earned classification.** Knowing which design choices carry over from a paradigm's canonical reference (DreamerV3's GRU state, Atick-Redlich's retinal receptive fields, Rao-Ballard's predictive coding hierarchy, Tishby-Zaslavsky's bottleneck parameterization) and which are domain-specific to the original problem requires empirical testing. Several candidates above were falsified at the implementation level because a canonical-reference assumption did not transfer cleanly to the dashcam contest scorer.
3. **Score-axis surrogate.** Training against the contest scorer directly is GPU-bound. Distilling the scorer into a smaller surrogate would amortize cost, but the distillation gap needs characterization first; the PoseNet gap at minor numerical perturbations is large enough that an unmeasured surrogate would not be safe.
4. **Implementation versus paradigm falsification.** Recent empirical anchors falsified specific implementations (Z6 driver-mode bug, C6 IBPS 24-dim latent collapse, ATW V2 weak conditioning, NSCS06 v6 7-cargo-cult stack, NSCS01 nullspace, Wunderkind G1 v2 reducer) — not the paradigm class. The recurring failure mode this session has been mislabelling implementation falsification as paradigm kill and losing the paradigm.

---

## G. Caveats

Most of what is described above is prototype-level. The intention is production-hardened OSS and the apparatus is moving in that direction (OSS hardening pass on sister `adpena/tac` against comma.ai / openpilot conventions is recent), but several class-shift research artifacts are buggy, half-finished, or duplicated across sibling lanes. The inventory is honest about which candidates have empirical anchors and which are design-only — it does not claim every research path is at the same polish level.

Theoretical-floor estimates jump around depending on the analysis basis. There is no single canonical floor; "where we could go from here" is honestly uncertain at the `0.001`–`0.01` scale even before considering which class-shift would actually validate.

This document is a snapshot at authoring time. It will be wrong in detail within a few sessions — the substrate registry, empirical anchors, and falsification state move fast. The PR body remains the canonical source for the submission packet itself; this document is context for the work that exists beyond it.

---

## H. Reproducibility and cross-links

- **Submission packet**: [`commaai/comma_video_compression_challenge#110`](https://github.com/commaai/comma_video_compression_challenge/pull/110) + the `submissions/hnerv_fec6_fixed_huffman_k16/` directory it ships under.
- **Sister library**: [`adpena/tac`](https://github.com/adpena/tac) — task-aware-compression research primitives, CI/OSS hygiene, canonical helpers, and reusable tooling used during development. It is not required by the PR #110 submission runtime.
- **This repo**: [`adpena/comma-lab`](https://github.com/adpena/comma-lab) — working repo for the broader inventory. Not the submission packet.
- **Meta engineering vision**: [`docs/meta_engineering_vision.md`](meta_engineering_vision.md) — light hors d'oeuvres on the unifying engineering vision (replace arbitrary constants with learned or discovered optimals; continual-learning mechanism; signal-driven; per-element optimal target). Names xray + atoms + meta-Lagrangian/Pareto + canonical equations + master-gradient + canonical Provenance + cathedral autopilot + continual-learning posterior + wiring discipline + STRICT preflight as the canonical operational surfaces that operationalize the META. Lead-with-META reading for the public introduction.
- **Tight public overview**: [`docs/comma_lab_overview.md`](comma_lab_overview.md) — 1500-word operator-facing introduction. TLDR + empirical anchors table + paradigm-class scaffolded table + tooling/methodology bullets + collaboration framing. Targeted at a reader who has 5 minutes and wants the canonical picture of what comma-lab is, what it has, and what it doesn't.
- **Full-stack source map**: [`docs/full_stack_source_map.md`](full_stack_source_map.md) — paper-oriented map from external research lineages (LA-Pose, RAFT, MAE/VideoMAE, V-JEPA, VGGT, SAM2/SAM3/Falcon/DINOv3, openpilot/commaVQ/comma10k, Cool-Chic/C3/CompressAI/DCVC-FM, SPADE/CLADE, SIREN, NeRV/HNeRV variants, UNIWARD/STC, foveation/wavelets) to local surfaces, current evidence grade, and next validation gate. This is the right place for the broad named-source surface; the PR body stays narrow.
- **Standout candidates spotlight**: [`docs/standout_undersold_candidates_spotlight.md`](standout_undersold_candidates_spotlight.md) — 90-second skim of ten candidates from this inventory that deserve prominent attention (Z7-Mamba-2, RAFT-derived poses + LAPose, CLADE, SPADE, SIREN, UNIWARD, freezing toolkit, master-gradient extractor, cargo-cult-unwind methodology, 4-tier council discipline). Targeted at a reader with time for one thing beyond the submission.
- **Operator-pinned spotlight extensions**: [`docs/standout_spotlight_extensions_operator_pinned_20260520.md`](standout_spotlight_extensions_operator_pinned_20260520.md) — companion to the spotlight memo above. Adds Fridrich lineage (the contest is inverse steganalysis), wavelets at full prominence (Daubechies + Mallat cross-cutting role), telescopic foveation revival, and water-bucket filling (Cover-Thomas convex-optimal rate allocation, Lane Ω-W V1 shipped through V3 launch-ready). Includes an honest meta-observation section on the gap between scaffold-level work and paid-GPU-validated work.
- **AI-assisted inverse-steganalysis + persona-council methodology**: [`docs/ai_assisted_inverse_steganalysis_persona_council.md`](ai_assisted_inverse_steganalysis_persona_council.md) — narrative articulation of the two META concepts that ran underneath the substrate work: (1) applying inverse-steganalysis mining discipline to the contest's entire information space (Yousfi PhD lineage, Fridrich methodology, prior-PR maintainer hints, sibling research culture), and (2) the named-persona 4-tier grand council methodology grounded in Anthropic's canonical persona-vectors research (Chen et al. 2025; [arXiv:2507.21509](https://arxiv.org/abs/2507.21509); [anthropic.com/research/persona-vectors](https://www.anthropic.com/research/persona-vectors)). Pairs the personas with structural rigor (per-deliberation assumption surfacing, Assumption-Adversary classification, 3-clean-pass recursive review, per-substrate optimal-form symposium) so the methodology is engineering pattern rather than decoration.

### Companion methodology + tooling tours

- [`docs/cargo_cult_unwind_methodology.md`](cargo_cult_unwind_methodology.md) — paradigm-rescue discipline; the canonical procedure that produced the NSCS06 v6 → v7 = 44% improvement in one iteration. Reusable across substrate paradigms.
- [`docs/strict_preflight_catalog_summary.md`](strict_preflight_catalog_summary.md) — browseable summary of the ~300 strict-preflight catalog gates that structurally extinct empirically-encountered bug classes (per Section E.1 reference). META-meta gates (#118 / #159 / #176 / #185 / #186 / #299) protect the catalog itself from drift.
- [`docs/canonical_equations_tour.md`](canonical_equations_tour.md) — tour of the 6 initial canonical equations (Brotli cascade bounded per stream, MPS-vs-CUDA drift architecture-class dependent, per-byte leverage uniformly distributed on entropy-coded archives, per-pair master-gradient Taylor + Cauchy-Schwarz bound, master-gradient locality violation by codec, canonical frontier pointer). Codifies empirical anchors as first-class artifacts.
- [`docs/master_gradient_extractor_tour.md`](master_gradient_extractor_tour.md) — tool tour for the per-pair / per-byte / per-tensor master-gradient extractor + the 10 exploits it enables (per-pair difficulty atlas, score-weighted reconstruction error, top/bottom-K byte ranking, per-class chroma allocation, substrate-fit diagnostic, Cramér-Rao floor estimate, bit-level critical bits, per-pair clustering, streaming master-gradient during training).
