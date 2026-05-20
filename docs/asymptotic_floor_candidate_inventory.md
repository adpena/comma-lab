# Asymptotic-floor candidate inventory

This document is a snapshot inventory of class-shift candidate substrates I have either trained, scaffolded, or designed against the comma video compression challenge while building the FEC6 + fixed-Huffman bolt-ons submitted as PR #110. It is not a roadmap, not a benchmark suite, and not a claim about how close any of these come to a theoretical floor. It is the operator-facing context for the work that exists in this repo beyond the submission packet.

Anchor: [`commaai/comma_video_compression_challenge#110`](https://github.com/commaai/comma_video_compression_challenge/pull/110).

## A. Where the local floor sits

The HNeRV-family cluster (@AaronLeslie138 PR #95 / @EthanYangTW PR #98 / @BradyMeighan PR #100 / @SajayR PR #101 / @EthanYangTW PR #102 / @rem2 PR #103 / this PR #110) sits within roughly 0.0008 of each other on the CPU axis the leaderboard ranks. Within-HNeRV-family the local floor appears effectively reached; further within-class bolt-ons increasingly trade smaller distortion savings against larger rate costs.

Class-shift to a different substrate paradigm is the visible next direction. Multiple theoretical-floor analyses I have run (R(D) bound from the contest's scoring formula, Blahut-Arimoto on conditional entropy of the scorer outputs, Dykstra-feasibility on the intersection of rate / segmentation / pose constraint polytopes, scorer-conditional MDL density via the post-training tier-C method) all give different estimates of where the next floor sits. None of them agrees on a single number, so I treat "where the floor is" as a band rather than a point, and the rest of this document is what I have tried to push into that band.

## B. Empirically-run anchors

These are end-to-end measurements I have actually run on contest-1:1 hardware. Score literals are axis-tagged; CPU evaluations are on `linux_x86_64_cpu` Modal containers matching the GitHub-Actions `ubuntu-latest` runner family; CUDA evaluations are on Modal Tesla T4. Anything tagged `[advisory only]` or `[macOS-CPU advisory]` is local development signal that I do not use for promotion, ranking, or submission decisions.

| Lane | Class | Score | Status |
|---|---|---|---|
| `hnerv_fec6_fixed_huffman_k16` (this PR) | within-HNeRV-family bolt-on stack | `0.192051 [contest-CPU]`, `0.226210 [contest-CUDA T4]` | EMPIRICAL, paired |
| `hnerv_ft_microcodec` PR #101 reproduction | within-HNeRV-family | `0.192845 [contest-CPU]` recomputed from bot eval | EMPIRICAL, sister anchor |
| `lane_a1_…` paired anchor | A1 substrate engineering | `0.19285 [contest-CPU]` paired CUDA | EMPIRICAL, paired |
| `pr106_format0d_latent_score_table` | within-HNeRV-family CUDA-side | `0.205330 [contest-CUDA T4]` | EMPIRICAL, single-axis |
| Earlier within-HNeRV-family iterations | bolt-on | multiple `[contest-CUDA T4]` anchors `0.95`–`1.45` (historical; pre-PR101-substrate) | EMPIRICAL, retired-config |
| `c6_ibps` 50ep IB-bottleneck smoke | information-bottleneck | `final_score = 3.04 [contest-CUDA A10G]` vs design-time predicted band `[0.113, 0.163]` | FALSIFIED-at-this-latent-dim. The 24-dim IB bottleneck collapses segmentation (`d_seg` dominates ~86% of the score). Reactivation: latent-dim sweep at `{48, 96, 192}` + post-training tier-C density re-measurement on the next archive |
| `nscs06` Carmack-Hotz Strip-Everything | composition substrate | v6 `105.15 [contest-CUDA T4]` → v7 `58.89 [contest-CUDA T4]` after one cargo-cult unwind iteration (44% improvement) | EMPIRICAL design-time validation; not paired contest-CPU yet; reactivation = continue the unwind ladder before paired re-measurement |
| `lane_g_v3` historical | within-class | `1.05 [contest-CUDA T4]` | EMPIRICAL historical (pre-HNeRV-family arrival) |
| `apogee_int4` PTQ smoke | low-bit weight quantization | `1.42866394 [contest-CUDA T4]` | FALSIFIED-at-naive-PTQ; reactivation = QAT / LSQ / per-channel / smaller blocks / outlier handling, not yet attempted on the post-HNeRV substrate |

The honest summary: outside the HNeRV-family local cluster, every other paradigm I have empirically tested either falsifies at a specific implementation config (not at the paradigm class) or has not yet been pushed end-to-end to a paired CPU + CUDA anchor on contest-1:1 hardware.

## C. Class-shift candidate inventory

Grouped by paradigm class rather than alphabetically. Each entry is one-or-two sentences on what the candidate is plus a status tag. I deliberately do not give predicted-band numbers for any candidate that has not been empirically anchored at contest scale — the cost of design-time band claims that fail to validate (the C6 IBPS case above) is high enough that I would rather understate.

### C.1 Predictive-coding world models (Rao & Ballard 1999; Hafner DreamerV3)

Class hypothesis: ego-motion-conditioned next-frame prediction has lower scorer-conditional entropy than per-frame independent encoding. The dashcam contest video has strong ego-motion coherence; this is the class with the most design effort behind it.

- **Z6 multi-layer FiLM (depth=3, ~300K params)** — primary predictive-receiver substrate; SCAFFOLDED end-to-end. Wave 2 smoke fired but ran the smoke-mode trainer path instead of the full-mode path due to a driver mode-routing bug (now closed at the gate level). Reactivation: re-fire the full-mode 100ep canary.
- **Z6-v2 cargo-cult-unwind redesign** — DESIGN-ONLY response to a council critique that the original Z6 inherited canonical capacity assumptions without testing them.
- **Z7 LSTM/GRU temporal predictor** and **Z7-as-Mamba-2 (SSM substrate)** — DESIGN-ONLY sisters; the deep-research path-forward symposium recommended the Mamba-2 variant as the most direct paradigm-bridge to recent state-space-model results.
- **Z8 hierarchical predictive coding (canonical quadruple)** — DESIGN-ONLY. Composes Daubechies wavelet hierarchical prior + Mallat multi-resolution + Rao-Ballard hierarchy + Wyner-Ziv side-information into one substrate. Untested at contest scale.
- **DreamerV3 RSSM categorical-posterior paradigm-bridge** — DESIGN-ONLY with a tiered cost ladder. Tightened predicted band `[0.20, 0.40]` is honest inherited uncertainty; post-training tier-C measurement is the canonical validator and has not been run.

### C.2 Cooperative-receiver framings (Atick & Redlich 1990)

Class hypothesis: bytes the decoder shares with the scorer (the SegNet/PoseNet weights are public and fixed) should be exploited as side information; the encoder optimizes against the decoder-and-scorer joint, not just the decoder.

- **Z4 cooperative-receiver loss** — SCAFFOLDED as a bolt-on objective term; full path is council-gated.
- **ATW V1 (Atick-Tishby-Wyner triple)** — DESIGN-ONLY v1; folded into ATW V2.
- **ATW V2** — SCAFFOLDED L1; first conditioning probe returned `INDEPENDENT` verdict (`MI = 0.006385 bits/symbol`, two orders below the meaningful-conditioning threshold). DEFERRED-pending-research; reactivation = trained ATW residual probe or substrate-native scorer-logit sketch instead of an opaque conditioning channel.
- **ATW V2-1 with Faiss IVF-PQ per-region SegNet softmax channel** — DESIGN-ONLY; probe budget pre-empted by ATW V2's INDEPENDENT verdict.

### C.3 Information-Bottleneck framings (Tishby & Zaslavsky 2015)

Class hypothesis: the optimal codec compresses to the minimum sufficient statistic for the scorer's outputs, rather than for pixel reconstruction.

- **C6 IBPS (canonical Path B quadruple)** — EMPIRICAL anchor falsified the specific 24-dim latent at 50ep; paradigm not killed. The post-training tier-C re-measurement on the landed archive surfaced the structural reason (segmentation collapse). Reactivation: latent-dim sweep + β_ib calibration before any further paid dispatch.
- **C6 IBPS β_ib sweep, E4 MDL-IBPS, Tishby IB-pure** — DESIGN-ONLY sisters; queued behind the latent-dim sweep.

### C.4 Pretrained driving priors

Compress against an out-of-distribution dashcam codebook trained on Comma2k19, not against the contest video itself. **DP1 Phase 2** is SCAFFOLDED with a Comma2k19 local-cache canonical helper, codebook provenance metadata, deliverability proof builder, and OOD-codebook + Wyner-Ziv composition path; full path pending Phase 2 council ratification. **DP1 + PR101 composition** is DESIGN-ONLY, pending an empirical anchor on DP1 alone.

### C.5 Pose-axis, foveation, spatial-sparse (Gibson 1950; LAPose)

Class hypothesis: the pose axis of the score and the spatially-non-uniform information density of the dashcam frame both reward representations that allocate bits non-uniformly. **TT5L V2 foveation + LAPose redesign** is DESIGN-ONLY, reformulated after a `REFUSE` verdict on V1's monolithic ~3000-epoch training plan; V2 splits into staged probes. **FF foveation lane scaffold** is an L0 scaffold predicated on TT5L probe outcomes. **RAFT-derived poses, LAPose pose codec, SAR coherent pose pairs** are SCAFFOLDED or DESIGN-ONLY; none promoted.

### C.6 NeRV-family beyond HNeRV (Chen et al. 2023 lineage)

Variants of the implicit-neural-representation family the medal cluster lives in. TCNeRV, BlockNeRV, FFNeRV, DSNeRV, HiNeRV, e_nerv, ego_nerv, and nervdc are SCAFFOLDED at varying maturity. Several hit dispatch-time API crashes (Wave 3) and are tagged `research_only` pending reactivation through the canonical Phase 2 council flow rather than treated as paradigm-falsified.

### C.7 Non-NeRV substrate architectures

Cool-Chic and C3 are DESIGN-ONLY with open export-contract gates. Wavelet residual, hybrid renderer + residual, SIREN coordinate MLP, and VQ-VAE (van den Oord 2017; discrete-posterior precedent for some predictive-coding work) are SCAFFOLDED. A grayscale-LUT extension of the Selfcomp PR #56 paradigm is DESIGN-ONLY beyond what PR #56 already shipped. A Quantizr-faithful reimplementation has historical `[contest-CUDA T4]` anchors in the `0.33`–`0.41` band from competitive-intelligence work; not the current frontier. A diffusion renderer is DESIGN-ONLY.

### C.8 Codec primitives and entropy coding

A canonical Wyner-Ziv layer at `tac.codec.wyner_ziv_layer` is paired with a deliverability proof builder that distinguishes truly deliverable side-information savings from research-sidecar phantom savings; several speculative composition rows have been rejected as `NOT_DELIVERABLE`. Hierarchical Wyner-Ziv composition (the canonical Daubechies + Mallat + Rao-Ballard + Wyner-Ziv quadruple) is DESIGN-ONLY. STC-Dasher arithmetic-coding maximalism, a Ballé hyperprior with CompressAI primitives registered in the canonical inventory, the Selfcomp block-FP lineage from PR #56, Hessian-block-FP, and UNIWARD texture-aware encoding from the Fridrich lineage are SCAFFOLDED.

### C.9 Self-compression family

SC++ (SegMap + KL distill) and MDL FP4 TTO are SCAFFOLDED. `lane_17_imp` iterative magnitude pruning is L2 SCAFFOLDED; the council symposium deferred dispatch pending a cycle-0 empirical regression and a score-gradient saliency sidecar.

### C.10 Composition substrates + stacking (Carmack-Hotz Strip-Everything lineage)

- **NSCS06 v6 → v7 → v8 (Strip-Everything, per-class chroma anchor)** — EMPIRICAL anchor at v7 `58.89 [contest-CUDA T4]` (44% improvement over v6 `105.15`) in one cargo-cult-unwind iteration. Variant C is a redesign queued for further unwinds. The paradigm is intact; the v6-falsification is implementation-level, not paradigm-level.
- **NSCS01 nullspace-split renderer** — SCAFFOLDED.
- **NSCS02 downsampled renderer** — SCAFFOLDED.
- **NSCS03 Ballé end-to-end joint codec** — SCAFFOLDED.
- **`stack_of_stacks`** — SCAFFOLDED; recipe-level composition framework.
- **S2SBS byte-stuffing** — SCAFFOLDED.
- **SAR composition** — see C.5.

### C.11 Higher-order optimization framings

**Riemannian-Newton substrate engineering** (manifold-aware second-order optimization for the substrate weight space) and **Tropical d_seg solver** (sister) are DESIGN-ONLY. **Joint-ADMM coordinator** is SCAFFOLDED as a cross-substrate consensus framework following Boyd's ADMM. **3-set Venn classification** (high-pair-invariant / pair-specific / per-pair) is the empirical classifier driving the per-pair master-gradient framework; it informs which bits live in which composition axis.

Section-cumulative note: several candidates above have empirical anchors that falsify a specific implementation (Z6 driver-mode bug, C6 IBPS 24-dim latent collapse, NSCS06 v6 7-cargo-cult stack, ATW V2 D4 weak conditioning, Wunderkind G1 v2 reducer, NSCS01 nullspace, NSCS06 v8 Path B variant). In every case I have tried to record the falsification at the level it actually applies (implementation) rather than the level it does not (paradigm class), because the recurring failure mode this session has been mislabelling implementation falsification as paradigm kill and losing the paradigm.

## D. Cost-efficiency and hardware

Concrete cost classes I budget against:

- Vast.ai RTX 4090 at roughly `$0.25/hr` — primary substrate; cheapest path to 24GB of CUDA VRAM.
- Modal Tesla T4 at roughly `$0.59/hr` — fallback when Vast.ai is rate-limited or when a recipe declares T4-only.
- Modal A100 at roughly `$1.50/hr` — used when VRAM exceeds 24GB or when the recipe declares `min_smoke_gpu: A100`.
- Lightning Studio free tier — opportunistic; 22h/month subscription cap.
- Local Linux x86_64 Modal CPU container — contest-1:1 hardware match for the GitHub-Actions `ubuntu-latest` runner family; the only authoritative CPU axis.
- Local macOS CPU on M-series — used as a free advisory proxy; empirically within roughly `6e-6` of the Linux x86_64 anchor on at least one prior submission, but always tagged `[macOS-CPU advisory]` and never promoted.
- Local MPS — development signal only; PoseNet drift versus the contest CUDA scorer is roughly `23×`, so MPS is never used for ranking, promotion, or submission decisions.

A typical class-shift training run on a substrate the size of the HNeRV cluster costs between `$15` and `$300` per attempt depending on stage count, curriculum length, and how many smokes precede the full dispatch. Most of my session-to-session spend goes into smokes rather than full runs, because the smokes are the place where implementation-level cargo-cult assumptions get falsified before the full meter starts.

## E. What is in this repo beyond the submission packet

### E.1 Tooling

The submission's runtime is intentionally small. The repo that produced it carries the rest of the apparatus:

- A cathedral-style autopilot ranker that ingests candidate substrates and emits ranked dispatch recommendations.
- A per-pair master-gradient extractor plus a per-pair optimal treatment plan via a Lagrangian-dual solver that emits per-pair recommendations the cathedral autopilot consumes.
- A canonical equations registry that stores empirically-calibrated equations as fcntl-locked append-only JSONL rows; this is where claims like "brotli cascade saturates at a per-stream bound" or "MPS drift is architecture-class dependent" live.
- A Modal call-id ledger + harvester that closes the spawn-and-lose failure mode on Modal's detached function-call cache.
- A subagent crash-resume checkpoint protocol writing to fcntl-locked append-only JSONL so a session that dies mid-work can be resumed from disk.
- A frontier pointer canonical helper that is the single source of truth for our local-best CPU and CUDA anchors, refreshed automatically on successful dispatch completion.
- An empirical per-X optimal codec planner with DuckDB unification for cross-pair-sensitivity queries.
- A probe-outcomes ledger that prevents re-firing a dispatch the apparatus has already adjudicated within a 30-day window.
- A canonical Provenance helper attached to every score-claiming row in persisted state.
- A Wyner-Ziv deliverability proof builder that distinguishes truly deliverable side-information savings from research-sidecar phantom savings; several speculative composition rows have been rejected by the builder as `NOT_DELIVERABLE` rather than landed.
- Pre-dispatch adversarial review automation that runs an external reviewer pass before any paid dispatch above a cost threshold.
- A master-gradient x-ray visualization tool, and an asymptotic-pursuit candidate readiness assessment that scans the registry against the dispatch-protocol catalog gates and surfaces the top-ranked candidate that is actually ready to fire.
- Roughly 300 STRICT preflight gates that fail closed on the bug classes the session has empirically encountered.

### E.2 Methodology and discipline

The methodology side is what made the per-class candidate inventory possible without compounding errors faster than I could fix them.

- A cargo-cult unwind methodology, with NSCS06 v6 → v7 as the canonical example (44% empirical improvement in one iteration by enumerating each cargo-culted assumption, classifying it as hard-earned-from-evidence or cargo-culted-from-inheritance, and unwinding the cargo-culted ones).
- A hard-earned-vs-cargo-culted classification framework applied per deliberation; council members surface their operating-within assumption explicitly so the framing the discussion sits within can be challenged.
- Per-substrate adversarial-council symposium discipline before any paid dispatch above a threshold; recursive adversarial review with a three-clean-pass counter; a 4-tier council hierarchy with explicit attendees, quorum, and tie-break rules.
- Bug-class extinction at orthogonal surfaces (design-memo / runtime-effect / per-feature contract / promotion gate / retirement gate / council discipline / iteration discipline / post-training validation), which is where most of the preflight-gate count comes from.
- A sister library `adpena/tac` that holds the task-aware-compression primitives the submission's runtime imports a small slice of.

## F. What is stuck

Each class-shift candidate is stuck on one or more of:

1. **Substrate-engineering cost.** Training a new architecture from scratch against the contest scorer costs roughly `$50`–`$500` per honest attempt. Cheap smokes triage some failures, but the smoke does not always disambiguate between implementation falsification and paradigm falsification.
2. **Cargo-cult-vs-hard-earned classification.** Knowing which design choices carry over from a paradigm's canonical reference (Hafner DreamerV3's GRU-deterministic state, Atick-Redlich's retinal-receptive-field structure, Rao-Ballard's predictive coding hierarchy, Tishby-Zaslavsky's bottleneck parameterization) and which are domain-specific to the reference's original problem requires empirical testing. Several candidates above have been falsified at the implementation level because a canonical-reference assumption did not transfer cleanly to the dashcam contest scorer.
3. **Score-axis surrogate.** Training against the contest scorer directly is GPU-bound. Distilling the scorer into a smaller surrogate would amortize the cost, but the distillation gap needs characterization first; the gap on PoseNet at minor numerical perturbations is large enough that an unmeasured surrogate would not be safe.
4. **Distinguishing-feature versus implementation falsification.** Several recent empirical anchors falsified a specific implementation (Z6 driver-mode bug, C6 IBPS 24-dim latent collapse, ATW V2 weak conditioning, NSCS06 v6 7-cargo-cult stack, NSCS01 nullspace, Wunderkind G1 v2 reducer) — not the paradigm class. The recurring failure mode in this session has been mislabelling implementation falsification as paradigm kill and losing the paradigm.

## G. Caveats

Most of what is described above is prototype-level. The intention is production-hardened OSS, and the apparatus is moving in that direction (the OSS hardening pass on the sister `adpena/tac` repo against comma.ai / openpilot conventions is a recent step), but several class-shift research artifacts in the repo are either buggy, flawed, half-finished, or duplicated across sibling lanes. The inventory above is honest about which candidates have empirical anchors and which are design-only, but it does not claim every research path is at the same level of polish.

The theoretical-floor estimates I run jump around depending on the analysis basis (Shannon R(D), Blahut-Arimoto, Dykstra-feasibility intersection, post-training tier-C scorer-conditional MDL density, others). There is no single canonical floor; "where we could go from here" is honestly uncertain at the 0.001–0.01 scale even before considering which class-shift would actually validate.

This document is a snapshot at the time of authoring. It will be wrong in detail within a few sessions of further work — the substrate registry, the empirical anchors, and the falsification state move fast. The PR body remains the canonical source for the submission packet itself; this document is context for the work that exists beyond it.

## H. Reproducibility and cross-links

- Submission packet: [`commaai/comma_video_compression_challenge#110`](https://github.com/commaai/comma_video_compression_challenge/pull/110) + the `submissions/hnerv_fec6_fixed_huffman_k16/` directory it ships under.
- Sister library: [`adpena/tac`](https://github.com/adpena/tac) (task-aware compression primitives; CI badges, OSS hygiene, the canonical helpers the submission runtime imports from a small surface area).
- This repo: [`adpena/comma-lab`](https://github.com/adpena/comma-lab) (the working repo for the broader inventory; not the submission packet).
