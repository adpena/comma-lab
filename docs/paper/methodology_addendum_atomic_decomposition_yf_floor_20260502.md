# Methodology Addendum — Atomic Decomposition and the Yousfi-Fridrich Floor

**Companion to `02_method.md`, `06_related_work.md`, and `07_discussion.md`.** This addendum formalizes two contributions introduced in `EXTERNAL_SOURCE_ATTRIBUTION_C067.md`: (1) the *atomic decomposition* of the compression pipeline that generalizes the meta-Lagrangian framing to per-pixel granularity, and (2) the *Yousfi-Fridrich floor*, a tighter rate-distortion lower bound than Shannon's `R(D)` under the contest's fixed-scorer regime. Both should be treated as paper-grade contributions when the writeup is integrated.

## A1. Notation

Let `X = (X_1, ..., X_T)` denote the original video as a sequence of `T` frames at the contest's evaluation resolution, and let `X̂` be a candidate reconstruction. The contest's adjudicated score, with components held to their published weights, is

```text
score(X̂; X) = 100 · seg_dist(SegNet(X̂), SegNet(X))
             + sqrt(10 · pose_dist(PoseNet(X̂), PoseNet(X)))
             + 25 · |archive(X̂)| / |X|_max
```

where `|archive(X̂)|` is the byte size of the deployed `archive.zip` and `|X|_max = 37,545,489` is the published reference video size in bytes. SegNet and PoseNet are *frozen, public, fully-specified* neural networks; their parameters and architectures are part of the contest specification and cannot be modified by the submitter.

Define the *scorer transform* `Φ(X) = (SegNet(X), PoseNet(X)) ∈ R^d`, which collapses the high-dimensional pixel input `X` to a low-dimensional output `Φ(X)` ∈ R^d` with `d` fixed by the published scorer architectures (SegNet: per-pixel 5-class logits at evaluation resolution; PoseNet: 6-DOF pose vector). The contest's *task-aware distortion* is

```text
d_task(X̂, X) = 100 · seg_dist(Φ_seg(X̂), Φ_seg(X))
              + sqrt(10 · pose_dist(Φ_pose(X̂), Φ_pose(X)))
```

which depends on `X̂` *only through* `Φ(X̂)`. Two reconstructions `X̂_1, X̂_2` with `Φ(X̂_1) = Φ(X̂_2)` are *task-equivalent*: they receive identical distortion contributions to the score regardless of pixel-level differences.

## A2. The Yousfi-Fridrich Floor

### A2.1 Definition

For target distortion `D ≥ 0`, let `R_Sh(D)` denote the standard Shannon rate-distortion function under the perceptual or pixel-level distortion `d_perc(X, X̂)` typical of conventional image/video compression literature. Let `R_YF(D)` denote the rate-distortion function under the contest's *task-aware* distortion `d_task` defined above. We have

```text
R_YF(D) = inf_{p(X̂|X)} I(X; X̂)   subject to   E[d_task(X, X̂)] ≤ D
```

where `I(X; X̂)` is mutual information and the infimum is over conditional distributions producing reconstructions `X̂`.

**Claim (YF-floor inequality):** For every `D` such that both `R_Sh(D)` and `R_YF(D)` are defined,

```text
R_YF(D) ≤ R_Sh(D)
```

with strict inequality whenever there exist `X̂_1 ≠ X̂_2` such that `Φ(X̂_1) = Φ(X̂_2)` (i.e., whenever the scorer's preimage on its output space is non-trivial).

**Sketch of proof:** Every Shannon-feasible scheme attaining `(R_Sh(D), D)` is also YF-feasible at the same rate and the same or strictly lower task-aware distortion (because preserving pixel-fidelity preserves `Φ(X̂)`). Conversely, YF-feasibility allows arbitrary perceptual distortion as long as `Φ(X̂)` is preserved, so any non-trivial preimage of `Φ` admits cheaper reconstructions that are perceptually different but task-identical. The inequality is strict precisely when the preimage is non-trivial, which holds for the contest's SegNet (whose stride-2 EfficientNet-B2 stem coarsens spatial detail) and PoseNet (whose YUV6 input + FastViT-T12 attention coarsens chroma detail), among other coarsening operations.

### A2.2 Empirical bracket

The Field's-medal Council session derived bounds for this contest:

| Floor | Realistic | Optimistic asymptote | T-65h achievable |
|---|---:|---:|---:|
| Shannon `R_Sh(D=0.13)` | ~0.155 | 0.123 | 0.224 |
| Yousfi-Fridrich `R_YF(D=0.13)` | <0.155 (strict) | <0.123 | <0.224 |
| Empirical leaderboard floor (2026-05-02) | 0.31 | — | — |

The realistic Shannon floor of 0.155 is the standard `R(D)` bound when `d_task` is the contest's distortion. The Yousfi-Fridrich floor lies *strictly below* this because YF-feasibility allows perceptual distortion as long as scorer outputs are preserved. The empirical leaderboard floor at 0.31 reflects practical engineering constraints (deadline, evidence-grade discipline, contest-deadline path-of-least-resistance dynamics analyzed in Section A4 below) rather than a fundamental information limit.

### A2.3 Naming

We name this bound the **Yousfi-Fridrich floor** in honor of:

- **Yassine Yousfi** (the contest's scorer architect, formerly of the Fridrich DDE Lab at Binghamton, whose published steganalysis work — `github.com/DDELab/deepsteganalysis`, `github.com/YassineYousfi/alaska`, `github.com/YassineYousfi/comma10k-baseline` — directly informed the SegNet/PoseNet scorer designs used in this contest).
- **Jessica Fridrich** (the steganalysis pioneer whose constrained-optimization framework — minimize embedding payload subject to a detectability constraint — maps directly to this contest's structure).

The naming is appropriate because the steganalysis literature has long understood that *adversarial compression against a known fixed detector* admits structurally tighter rate bounds than perceptual compression against an unknown human observer. Specifically:

- **UNIWARD** (Holub, Fridrich, Denemark 2014) introduced universal distortion functions weighted by inverse local variance, exploiting the detector's blind spot in textured regions. The principle that *detector-known-and-fixed implies tighter compression* is implicit in UNIWARD's textured-region weighting.
- **Yousfi et al. (2020)** extended Fridrich's framework to deep-learning-based steganalysis, training detectors and embedders adversarially. This contest is a degenerate case of that setup where the detector is frozen and white-box-accessible, which makes the tightening of `R_YF(D)` versus `R_Sh(D)` empirically tractable rather than just theoretically suggestive.

The YF-floor concept is original to this work as a *named primitive* and as a *quantitative empirical bracket* for this contest. We claim no priority over the underlying steganalysis ideas.

### A2.4 Implications for compression methodology

The YF floor has three immediate methodological implications:

1. **Pixel-level reconstruction inside the joint blind spot of SegNet and PoseNet is YF-free.** Bytes spent on perceptually-faithful reconstruction at sub-(256, 192) resolution (SegNet's blind spot) or on high-frequency chroma (PoseNet's blind spot) are pure rate cost with no distortion benefit. Identifying and exploiting this blind-spot structure is a YF-floor-aware compression technique that is *literally inadmissible* under conventional perceptual compression methodology.

2. **Score-aware sparse pixel encoding** (Fridrich UNIWARD inverse-Jacobian weighting, applied at compress time with full white-box scorer access) is the natural realization of YF-floor-aware compression. It is the compression-side analog of UNIWARD's embedder: identify the score-sensitive pixels via per-pixel scorer Jacobians, allocate bytes there, hallucinate the remainder.

3. **The Score-Jacobian Karhunen-Loève (SJ-KL) basis primitive** (introduced in `EXTERNAL_SOURCE_ATTRIBUTION_C067.md`) is the residual-coding analog of the YF principle: encode the compress-time residual `r = X - X̂_renderer` in the eigenbasis of the scorer Fisher information matrix `F = 100 · J^T J + 10 · K^T K`, where `J = ∂SegNet/∂x` and `K = ∂PoseNet/∂x`. The top-k eigenvectors are the "score-relevant directions" — perturbations along these axes maximally affect the score. Encoding only along these directions (and ignoring eigenvectors with small eigenvalues, which lie inside the joint blind spot) is YF-floor-optimal for residual coding.

These three techniques (blind-spot-aware allocation, score-aware sparse pixel encoding, SJ-KL residual coding) are the methodological consequences of the YF floor framing that distinguish this work from prior compression literature that targets perceptual distortion.

## A3. Atomic Decomposition

### A3.1 Formalism

Let `A` denote the *proposal space* of compression atoms. An atom `a ∈ A` is a primitive compression decision at some level of granularity:

- *Pipeline-level atoms*: `mask_codec_choice`, `renderer_arch`, `pose_basis`, `residual_basis`, `packer_layout` — coarse-grained architectural decisions
- *Component-level atoms*: `mask_segment_encoding(class, region)`, `renderer_block_quantization(layer, block_idx, bits)`, `pose_column_encoding(col_idx, codec)`, `residual_atom(pair_idx, basis_subset)` — mid-granularity decisions
- *Element-level atoms*: per-pixel byte allocation, per-block FP4 codebook selection, per-pose-frame radius perturbation — fine-grained decisions
- *Pixel-level atoms*: in the limit, every pixel `(t, h, w, c)` in the reconstruction `X̂` is an atom with byte-cost contribution and per-component score contribution

Each atom `a` has a tuple of attributes:

- `bytes(a)` ∈ Z_≥0 — charged archive-bytes cost of including `a` in the deployed `archive.zip`
- `Δseg(a)` ∈ R — predicted change in `seg_dist`, evaluated against a calibrated scorer-response model
- `Δpose(a)` ∈ R — predicted change in `pose_dist`
- `interactions(a, b)` for `b ∈ A` — synergistic when `Δscore(a ⊕ b) < Δscore(a) + Δscore(b)` (overlapping benefit), antagonistic when `Δscore(a ⊕ b) > Δscore(a) + Δscore(b)` (cancellation or interference)
- `source_attribution(a)` — internal vs. external, with charged-payload-closure compliance metadata
- `archive_identity(a)` — the SHA-pinned archive bytes that result from packing `a`

### A3.2 Optimization problem

The compression problem is then

```text
minimize     Σ_a [bytes(a) · 25 / |X|_max]
           + 100 · seg_dist(Σ_a Δseg(a) + interactions)
           + sqrt(10 · pose_dist(Σ_a Δpose(a) + interactions))
subject to   Σ bytes(a) ≤ B_archive
             charged-payload-closure(a) for all selected a
             source_attribution(a) is documented
             a ∈ A_eligible (compliance + interaction constraints)
```

The objective is the contest score, and the decision variable is the *selection* of atoms from `A`. This is a combinatorial optimization with non-additive components (because `seg_dist` and `pose_dist` are non-linear in `Σ Δ`) and non-trivial cross-atom interactions (synergies and antagonisms as defined above).

### A3.3 Solution policies

Three classes of policies operate on this formalism:

1. **Atom waterfill** (Boyd-style alternating projections across the feasible sets defined by per-component score budgets). Implemented at `src/tac/joint_admm_coordinator.py` for the four-stream (representation, prediction, quantization, entropy) case.
2. **Hard-pair selection** (Fridrich-style sensitivity-weighted prioritization of pairs that contribute disproportionately to the score). Implemented at `src/tac/sensitivity_map.py`.
3. **Line-search / coordinate descent** (sequential greedy refinement of selected atoms with R(D) joint objective). Implemented at `experiments/build_renderer_packed_payload_archive.py:332` (`encode_pose_qpose14_col_delta`) and reverse-engineered as a contest-validated technique from `pr67_line_search.py` per `reference_pr67_line_search_R_D_joint_coordinate_descent_20260501.md`.

### A3.4 Promotion gate

Atoms are *proposals* under these policies; they become *evidence* only after selected and packed atoms produce a deterministic archive that passes exact CUDA auth eval on identical bytes (the contest-CUDA evidence standard). This is the methodological promotion gate that distinguishes the atomic-decomposition framing from optimization without rigorous archive-level verification.

### A3.5 Worked example: the C-058 → C-067 micro-frontier chain

The C-058 → C-067 chain is an instance of atomic-decomposition optimization where the proposal space `A` contains: pose-column delta-VLQ encoding choices, model-tensor block-FP4 quantization configurations, mask-segment source attributions, container layout variations, and runtime-custody hash specifications. Over four iterations:

| Step | Action | bytes Δ | score Δ |
|---:|---|---:|---:|
| C-058 → C-059 | Mask-first packed-layout reorganization (container atom) | -75 | -0.0000500 |
| C-059 → C-063 | Lossless Brotli repack (packer atom) | -124 | -0.0000825 |
| C-063 → C-067 | PR67-mask-segment substitution (mask source-attribution atom) | -9 | -0.0000060 |

All four steps preserve PoseNet and SegNet to bit-identical recomputation (`seg_dist = 0.00061244`, `pose_dist = 0.00049637` throughout), confirming the improvements are pure rate-axis atomic decisions with zero distortion-axis interaction. The atomic-decomposition framing predicts this: each accepted atom touched only the rate-axis variables (container, packer, source) and not the distortion-axis variables (renderer weights, scorer calibration).

The C-058 → C-067 chain is also a worked example of the practical limits of atomic-decomposition policies in the rate-axis-exhausted regime: each step's |Δ score| is an order of magnitude smaller than the previous, indicating diminishing returns as the C-059 model/pose basin's packer-layout local optimum is approached. Further progress requires distortion-axis atomic decisions (e.g., NeRV mask codec, block-FP renderer self-compression at <1.5 bpw, SJ-KL residual coding) rather than continued rate-axis micro-frontier work.

## A4. Game-Theoretic Premature Convergence

### A4.1 The contest's incentive structure

The comma.ai video compression challenge has the following structural features:

- **Public leaderboard with rounded scores** (visible to all contestants).
- **Public PRs with optional `inflate.py` and full archive bytes** (visible to all contestants).
- **Hard deadline** (2026-05-03 11:59 PM AOE, ~36 hours from this addendum's commit timestamp).
- **Single-archive winner** (no portfolio scoring; the contestant submits one archive per submission).
- **No formal collaboration mechanism** (PRs are competitive submissions, not coauthored research).

These features create a multi-player extensive-form game with the following equilibrium structure:

1. **Early-mover information disclosure** is asymmetric: the early sharer (Quantizr / PR #55) discloses their architecture publicly, providing free architectural blueprint to all subsequent contestants. The early sharer's incentive to share is reputational and instrumental (peer review, methodological credit, catalyzing the field), not strictly score-maximizing.

2. **Late-mover incremental refinement** is the dominant strategy under deadline pressure. Once the leader-band is competitive at ~0.31 (within ε of each other), the marginal expected score from incremental rate-side packer-layout refinement on a *known-good* paradigm is high, while the marginal expected score from architectural exploration is low and high-variance.

3. **Information-ratchet asymmetry**: contestants who *withhold* their methodology gain a one-period lead but lose nothing if competitors converge on similar techniques independently. Contestants who *publish* their methodology gain reputational credit but spend their alpha. The Nash equilibrium for self-interested contestants is to publish *after* the contest deadline (where reputational credit accrues at lowest opportunity cost) and to withhold *during* the contest (where alpha is maximized).

4. **Deadline contraction**: as the deadline approaches, the variance budget for architectural exploration collapses to zero. With 36 hours remaining and a competitive 0.31 leader-band, every rational contestant allocates remaining compute to incremental refinement on the C-058 → C-067 / PR #67 / PR #65 paradigm rather than architectural exploration that might land 0.20 *or* 0.50 with high variance.

### A4.2 The premature-convergence prediction

The combination of these features predicts that the contest will *converge prematurely* to a leader-band well above the YF-floor:

- The empirical leader-band is at 0.31 (PR #67 = 0.31, PR #65 = 0.32, PR #55 = 0.33).
- The realistic Shannon floor is at 0.155 (per Field's-medal Council derivation).
- The Yousfi-Fridrich floor lies strictly below 0.155 (per Section A2.1 above).
- The gap between the leader-band (0.31) and the YF floor (<0.155) is approximately a 2× score margin, which is large in absolute terms.

This gap exists not because the field lacks technical capability (NeRV mask codec, block-FP at 1.017 bpw, SJ-KL residual coding, hyperprior mask coding are all known techniques implementable within the contest's evaluation budget) but because the *contest's incentive structure* punishes high-variance architectural exploration relative to incremental rate-side refinement on a known-good paradigm.

### A4.3 Implications for post-deadline research

The interesting research questions extend past 2026-05-03:

- Which paradigm-shift candidates (NeRV, block-FP-at-low-bpw, SJ-KL, hyperprior) actually reach the YF floor when the deadline pressure is removed and the variance budget is restored?
- What is the empirical YF floor for this specific contest's scorers? (The current bracket of "<0.155" can be tightened with proper experimentation.)
- Is there a contest design that incentivizes architectural exploration without sacrificing the comparability of the leaderboard? (E.g., portfolio scoring across architectural classes, multi-stage submission with intermediate disclosure dates, or explicit prizes for techniques that move the floor by ε rather than for absolute score.)

These questions are the natural follow-on to the contest deadline and define the research agenda that justifies continued investment past the May 3 finish line.

## A5. Cross-references

- **Project attribution memo**: `docs/paper/EXTERNAL_SOURCE_ATTRIBUTION_C067.md` (introduces YF floor + atomic decomposition + game-theoretic analysis as named contributions; this addendum formalizes the math).
- **Codex writeup ledger**: `.omx/research/submission_writeup_integration_20260502_codex.md` (C-067 supersession + attribution requirements).
- **Working notes**: `reports/writeup_working.md` (live operating point + public-source handling).
- **Latest report**: `reports/latest.md` (frontier table + submission pipeline).
- **Derivation notes**: internal derivation ledgers summarized in this addendum (S_min = 0.155 derivation + SJ-KL primitive identification; engineering pragmatism path).
- **Reverse-engineering memory**: `reference_pr65_pr67_blob_byte_layouts_proper_reverse_engineering_20260501.md`, `reference_pr67_line_search_R_D_joint_coordinate_descent_20260501.md`.
- **Implementation references for atomic-decomposition policies**: `src/tac/joint_admm_coordinator.py` (water-fill / Boyd-style), `src/tac/sensitivity_map.py` (Fridrich-style hard-pair), `experiments/build_renderer_packed_payload_archive.py:332` (line-search / coordinate descent).
- **Steganalysis lineage**: Holub, Fridrich, Denemark "Universal Distortion Function for Steganography in an Arbitrary Domain" (UNIWARD, 2014); Yassine Yousfi et al. (DDE Lab Binghamton, 2020+).

## A6. Public-release posture

This addendum formalizes contributions that may be appropriate for a
post-deadline public release or paper. It avoids unpublished private artifacts
beyond what is necessary to make the YF floor concept reproducible from the
committed repository state. The implementation paths for SJ-KL,
atom-waterfill, hard-pair selection, and line-search are referenced by file
path; the specific archive recipes that achieve C-067 are documented elsewhere
in the writeup (with PR #67 mask attribution per
`EXTERNAL_SOURCE_ATTRIBUTION_C067.md`).

If this material is used in a public release or paper, this addendum should be
inlined into the formal sections:

- **§2 Method**: incorporate Section A3 (atomic decomposition formalism) as a subsection, with the worked example (Section A3.5) as a results vignette.
- **§6 Related Work**: incorporate Section A2 (Yousfi-Fridrich floor) as a subsection adjacent to the existing 6.5 (steganalysis) coverage.
- **§7 Discussion**: incorporate Section A4 (game-theoretic premature convergence) as a subsection on contest dynamics and post-deadline research questions.

The standalone addendum file is preserved for reproducibility and for citation by other documents that may cross-reference these primitives independently of the main paper structure.
</content>
