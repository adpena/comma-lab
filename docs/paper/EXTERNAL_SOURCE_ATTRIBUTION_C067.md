# External Source Attribution and Original Contributions — C-067 Frontier Archive

**Required reading for any public surfacing of the C-067 score** (`0.31561703078448233` [contest-CUDA T4 A++], archive `276214` bytes, SHA `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`, evidence `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/contest_auth_eval.adjudicated.json`).

This memo carries three responsibilities at once: (1) precise external-source attribution where it is owed, (2) substantive accounting of original engineering and research contributions that this work brought to the contest and the field, and (3) honest assessment of the contest dynamics — natural convergence, signal diffusion, and game-theoretic risks — that make C-067's narrow score margin a small part of the actual research story.

## Project Timeline

This work began **2026-04-06** as a from-scratch engagement with the comma.ai video compression challenge by a generalist new to the contest's specific scoring regime (frozen SegNet/PoseNet scorers, T4 inflate budget, archive-as-bitstream contract). Over **25 days and 1,584 git commits**, the project moved through five distinct paradigm phases:

1. **Postfilter era** (early April): convolutional pre/post-processors around standard codecs. Established the eval-roundtrip discipline, contest-CUDA-only score truth, and the auth_eval canonicalization that prevents proxy-vs-authoritative drift. Hit a ceiling at auth ~1.33.

2. **Renderer era** (mid April): scorer-aware learned renderers (Lane G v3 / PFP16 lineage). Established artifact custody, the EMA non-negotiable across all training paths, the fp32→fp16 pose-cast micro-frontier, the OWv3 sensitivity-weighted byte-plan refinement chain, and the orthogonal-stack composition pattern. Best contest-CUDA: 0.9974 (owv3_0120_orthogonal_stack, our first sub-1.000 score).

3. **Public-floor basin era** (late April → early May): pose-manifold water-fill on top of a Quantizr-derivative model lineage (C-058 → C-059 → C-063 → C-067), with byte-level packer/layout micro-frontier search. C-067 is the active frontier of this era.

4. **Reverse-engineering rigor** (2026-05-01): proper byte-level decoding of leaderboard PR #65 and PR #67 (committed at `reports/raw/leaderboard_intel_20260501/`), surfacing both the QZS3 grouped-variable-bit-depth FP4 packer and the ~6KB of side-channel correction (post/shift/frac/bias/region/randmulti) that PR #65 ships in a 30-byte-header 10-section container — multi-stage residual refinement at the bit level, a paradigm prior leaderboard analyses had missed.

5. **Cross-agent infrastructure** (2026-05-01 → 2026-05-02): Level 1 cross-agent dispatch coordination ledger at `.omx/state/active_lane_dispatch_claims.md` to prevent duplicate compute spend between Claude and codex agents working in parallel; STRICT preflight harness with 90+ checks structurally extincting bug classes; lane-maturity registry with seven gates (impl_complete, real_archive_empirical, contest_cuda, strict_preflight, three_clean_review, memory_entry, deploy_runbook); deterministic ZIP construction; payload-closure validation; runtime-custody hashing.

The pose-manifold water-fill micro-frontier C-058→C-067 is the active result lineage, but the durable contributions are the engineering disciplines that make any score claim defensible: contest-CUDA-only score truth, eval-roundtrip canonicalization, EMA discipline across all training paths, evidence-grade taxonomy (A++/A/A-negative/B/empirical/external/external_quarantine/invalid), runtime-custody hashing, and the cross-agent coordination ledger. Those disciplines were assembled across the 25-day project arc, not borrowed from any leaderboard PR.

## What C-067 is

C-067 is a fixed-slice composite archive whose three concatenated payload segments come from two distinct sources:

| Segment | Bytes | Origin | Authorship |
|---|---:|---|---|
| `mask.obu.br` | 219,472 | PR #67 `qpose14_qzs3_filmq9g_slsb1_r55` deployed `archive.zip` | EthanYangTW (MIN-CHUN Yang), comma.ai video compression challenge PR #67 |
| `model.pt.br` | 55,965 | C-059 (internal) QZS3-grouped variable-bit-depth FP4 packed renderer | This work |
| `pose_q.br` | 677 | C-059 (internal) QP1 (delta+VLQ first-column) pose codec output | This work |

The local exact CUDA T4 score of `0.31561703078448233` is A++ evidence for the **exact archive bytes** as built and adjudicated. The score is not a claim of independent invention of the mask payload, and it is not the most important result of the project.

## Why the mask segment is from PR #67

PR #67 ships a 219,472-byte `mask.obu.br` segment containing 600 odd-frame masks at 384×512 monochrome, encoded by libaom AV1 OBU low-overhead bitstream and Brotli-Q11 wrapped (per their `inflate.py` lines 778-784). C-067 reuses this segment verbatim. The bytes are charged against C-067's archive size — they are not a sidecar, not a script-side payload, and not an attempt to evade contest archive-metering. They are PR #67's published mask bytes deployed inside C-067's `archive.zip` per the contest's "all score-affecting payload must live in `archive.zip`" requirement.

The reverse-engineering custody trail for the PR #67 mask segment is committed at `reports/raw/leaderboard_intel_20260501/pr67_archive.zip` (sha `a5ed8da0d9988943c986b231b4cd33cea0ab878a8e1628134341db5f7f41c765`, the PR #67 deployed archive as downloaded 2026-05-01) and `reports/raw/leaderboard_intel_20260501/pr67_inflate.py` (the PR #67 inflate.py script). The PR #67 deployed `p` blob has structure: bytes `[0:219472]` = `mask.obu.br`, bytes `[219472:219472+model_br_len]` = `model.pt.br` with `model_br_len` from a 7-bucket length-lookup table, bytes `[219472+model_br_len:]` = `pose_q.br`. C-067's mask segment is exactly bytes `[0:219472]` of that blob.

## Acknowledgement to Quantizr (PR #55)

The most consequential external contribution to this project's late-April pivot was Quantizr's PR #55, which posted a 0.33 score with a published `inflate.py` that revealed the `JointFrameGenerator` paradigm: an 88K-parameter FiLM-conditioned depthwise-separable CNN with a single-mask trunk + dual head (one unconditional, one FiLM-on-pose), trained against KL-distill at temperature 2.0 from the SegNet logits, half-frame mask economy (only odd-frame masks needed), and FP4-block packed weights with Brotli wrapping.

For a generalist arriving from outside the steganalysis / scorer-aware-compression literature, this PR functioned as **revelation of unknown unknowns**: it demonstrated that the contest's frozen SegNet/PoseNet scorers admit a tiny, structurally compact rendering paradigm that is wildly more parameter-efficient than the renderer architectures we (and other late-entrants) had been pursuing. The "what looks possible" signal alone is research input — knowing a 0.33 score exists with 88K parameters reframes the search space immediately.

We owe Quantizr / Jimmy (UCLA CSE/Neuro) explicit acknowledgement for: (a) the JointFrameGenerator architectural paradigm, (b) the FiLM-on-pose-only conditioning insight, (c) the half-frame mask economy demonstration, (d) the FP4-block packing template, (e) the KL-T=2.0 distillation recipe, and most importantly, (f) the **public posting of `inflate.py`** that allowed reverse-engineering of all of the above. Quantizr explicitly noted in PR #55 that "sub-0.30 is possible by sweeping conv dims" and stopped optimizing — leaving the field open to follow-on work that extends rather than duplicates. That openness, both technical (publishing decoder code) and personal (publishing a "stopped optimizing here" boundary), is the kind of contestant-as-collaborator behavior that makes contests function as research accelerators rather than zero-sum extraction.

The rigorous reverse-engineering of PR #65 and PR #67 (originally undertaken to inform our C-058 → C-067 work, committed at `reports/raw/leaderboard_intel_20260501/`) similarly stands on the shoulders of those PR authors having shipped working `inflate.py` files and complete `archive.zip` payloads. The PR #65 multi-stage-residual paradigm finding (~6KB of post/shift/frac/bias/region/randmulti side-channel correction) is original analysis on our part, but it required PR #65 author henosis-us (Matt Abrahamson) to ship the working artifacts in the first place.

## Original Contributions of This Work

C-067's specific score is a small part of the project. The substantive original contributions, all reproducible from the committed repository state, are:

### Engineering and infrastructure

- **Contest-CUDA-only score truth and the evidence-grade taxonomy** (A++/A/A-negative/B/empirical/external/external_quarantine/invalid). Every score in this work's claim matrix is tagged with its evidence grade; non-`[contest-CUDA]` rows including `[advisory only]` device classes (MPS, CPU, proxy, byte-only, H100-only-without-T4) cannot rank or promote. This taxonomy is documented in `.omx/research/shannon_floor_claim_matrix_20260430_codex.md` and enforced by 90+ STRICT preflight checks at `src/tac/preflight.py`.

- **Eval-roundtrip canonicalization across all training paths** (`eval_roundtrip=True` default, including the simulate_eval_roundtrip(noise_std=0.5) STE for differentiability through the 384→874→uint8 contest pipeline). This caught and prevented a class of proxy-auth gap bugs that historically inflated optimistic scores by 2-11× on PoseNet-dominant lanes.

- **EMA discipline across all training paths** (decay 0.997 standard, applied at eval-time only with snapshot+restore, with the late-bound module-guard at `tac.training.EMA` to prevent freeze symptoms when EMA is incorrectly applied during training). Wired into 8+ training scripts (renderer, segmap, joint-pair, IMP, LoRA-TTO, postfilter, Szabolcs/Selfcomp clones, codebook-EMA codec layers).

- **Deterministic ZIP construction with hidden-file/resource-fork exclusion, zip-slip rejection, and scorer-load guards.** Submission archives carry their own decoder when needed; sidecar dependencies are forbidden per the contest's payload-closure standard.

- **Cross-agent dispatch coordination ledger** at `.omx/state/active_lane_dispatch_claims.md` (Level 1, this work, 2026-05-01 ~23:50 UTC). Mandatory append-before-dispatch protocol prevents duplicate GPU spend between concurrent agents. Codified in the repository operating rules as non-negotiable.

- **Lane-maturity harness** with seven gates (impl_complete, real_archive_empirical, contest_cuda, strict_preflight, three_clean_review, memory_entry, deploy_runbook), CLI tooling, audit log, and STRICT preflight Check 90 enforcement. Documented at `feedback_production_hardened_standard_definition_20260430.md`.

- **Subagent commit serializer with temp `GIT_INDEX_FILE`** (commit `b860710c`) to eliminate the staging-race that previously shuffled commit messages across commit objects when 2+ subagents committed concurrently.

### Compression / coding contributions

- **Pose-manifold water-fill micro-frontier search** (C-058 → C-059 → C-063 → C-067), an iterative byte-level pose refinement that exposed the C-059 basin's exhaustion frontier and produced four sequential A++ frontier improvements over 36 hours.

- **OWv3 sensitivity-weighted byte-plan refinement** (the renderer-era frontier), which combined per-tensor block bit allocation with score-component sensitivity to produce the orthogonal-stack 0.9974 contest-CUDA score before the basin pivot.

- **Independent reimplementation of the QZS3 grouped variable-bit-depth FP4 packer and QP1 (delta+VLQ first-column) pose codec.** Verified byte-identical to PR #67's `get_grouped_qv_state_dict` decoder and `decode_qp1` reader for matching inputs (15 round-trip tests in `src/tac/tests/test_qzs3_packer.py` and `test_qp1_pose_codec.py`). The container layout (mask + model + pose_q concatenation, single-blob no-zip-directory) is also reimplemented.

- **Ω-W water-filling and Joint-ADMM cross-stream coordinator** (Boyd-style alternating projections across {representation, prediction, quantization, entropy, archive-size} feasible sets) for archive-side rate-distortion optimization with hard coupling between streams.

- **Leaderboard reverse-engineering rigor** as a methodology: rigorous byte-level decoding (parser-source-driven, not hex-dump-guessing) of PR #65 and PR #67 surfaced the multi-stage residual-refinement paradigm in PR #65 that prior contestants had missed and informed our own packer work. The methodology itself, separate from the PR-specific findings, is a publishable contribution in a public-PR contest format.

### Research / framing contributions

- **The meta-Lagrangian atom compiler framing**: typed mask, renderer, pose, residual, packer, and layout atoms with charged byte cost, predicted component effect, interaction risk, source attribution, and archive identity once accepted. Atom-waterfill and hard-pair selection become evidence only after charged-archive CUDA-promotion.

- **The Shannon-floor derivation** with explicit lower bounds per score component (S_min realistic = 0.155, T-65h achievable = 0.224, optimistic asymptote = 0.123). Documented in `project_grand_council_FIELDS_MEDAL_shannon_floor_obsession_20260501.md`.

- **The Score-Jacobian Karhunen-Loève (SJ-KL) basis primitive**: the optimal residual subspace for fixed-downstream-task lossy compression (top-k eigenvectors of the scorer Fisher-information matrix `F = 100·JᵀJ + 10·KᵀK`). Identified by the Fields-medal Council session as a publishable compression primitive alongside DCT, wavelet, VQ-VAE, and NeRV, generalizing beyond the comma contest to any domain with a public scorer. Implementation pre-staged but not yet deployed against C-067 anchor.

- **Contest game-theory analysis** (next section) of the deadline + public-PR + leaderboard structure as a force toward premature convergence on local optima.

## Natural Convergence and Signal Diffusion Across the Leaderboard

The top-five public-leaderboard PRs (#55 / Quantizr 0.33, #67 / EthanYangTW 0.31, #65 / henosis-us 0.32, #56 / Selfcomp 0.38, plus the inferred "unified_brotli" 0.33) all converged on the same `JointFrameGenerator` paradigm: 88K parameters, c1=56, c2=64, hidden=52, cond_dim=48, FiLM-on-pose, half-frame mask economy. This is **not** because the paradigm is the global R(D) optimum for the task — it is because:

1. **Quantizr's PR #55 published the paradigm at 0.33**, demonstrating it is *possible*, which immediately reframed every late-entrant's hypothesis space.
2. **The architecture is at a local optimum** for the JointFrameGenerator hypothesis class against the frozen SegNet/PoseNet scorers — incremental capacity changes do not improve score.
3. **The remaining headroom is rate-side**, so subsequent leaders (PR #67, PR #65) explored different packer / container / side-channel-correction paradigms while keeping the architecture identical.

This is a **healthy scientific diffusion pattern**: openly-published `inflate.py` files allow reverse-engineering, follow-on work explicitly extends rather than duplicates, and the field collectively maps the architectural and packer-layout local-optimum landscape.

It is also a **risk factor for premature convergence**, which is the main concern of the next section.

## Beyond the Shannon Floor: the Yousfi-Fridrich Floor

The standard Shannon rate-distortion bound `R(D)` quantifies the minimum bit-rate to transmit a source `X` such that a reconstruction `X̂` satisfies `E[d(X, X̂)] ≤ D` for some distortion measure `d`. The classical assumption — implicit in nearly all video-compression work that targets human viewers — is that `d` is a perceptual distortion (PSNR, MS-SSIM, LPIPS, or some human-calibrated proxy).

This contest is not human-perceptual compression. The distortion is `d(X, X̂) = 100 · seg_dist(SegNet(X̂), SegNet(X)) + sqrt(10 · pose_dist(PoseNet(X̂), PoseNet(X)))` where SegNet and PoseNet are **frozen, public, fully-specified neural networks**. The relevant lower bound is therefore tighter than Shannon's `R(D)` for perceptual `d`: it is the minimum rate to transmit *whatever subset of `X` the frozen scorers actually consume to produce their outputs*, plus the rate to disambiguate the equivalence class of inputs that produce identical scorer outputs.

We name this tighter bound the **Yousfi-Fridrich floor**, after Yassine Yousfi (the contest's scorer architect, formerly of Fridrich's DDE Lab at Binghamton) and Jessica Fridrich (the steganalysis pioneer whose constrained-optimization framework — minimize embedding payload subject to a detectability constraint — maps directly to this contest's structure). The naming is appropriate because the steganalysis literature has long understood that *adversarial compression against a known fixed detector* admits structurally tighter rate bounds than perceptual compression against an unknown human observer:

- In steganalysis, the embedder knows the detector exactly and exploits its blind spots (UNIWARD's textured-region weighting, local-variance-based embedding cost, square-root law for spread-vs-concentrate distortion). The optimal embedder lives at the boundary where the detector's class probability is maximally uncertain.
- In our contest, the "embedder" is the compression archive and the "detector" is the SegNet/PoseNet scorer pair. The optimal compressor lives at the boundary where the scorer's outputs are minimally perturbed per byte spent.

The Yousfi-Fridrich floor `R_YF(D)` is bounded above by the standard Shannon `R(D)` for the same `d` — every Shannon-feasible scheme is YF-feasible — but typically lies strictly *below* it because YF-feasibility allows arbitrary perceptual distortion as long as scorer outputs are preserved. In practice for this contest:

- **SegNet has a stride-2 EfficientNet-B2 stem** (per the published architecture), which means its first layer halves resolution before any class-discriminative computation. Pixel detail at sub-(256, 192) resolution is *invisible to SegNet*. Bits spent reconstructing that detail are wasted under YF but charged under Shannon.
- **PoseNet input is YUV6 (4 luma + 2 chroma-subsampled planes) at 12-channel layout via FastViT-T12 attention**, which means chroma detail at high spatial frequency is consumed with reduced weight. Bits spent on high-frequency chroma reconstruction are partially wasted under YF.
- **Both scorers are frozen** — they cannot learn to penalize previously-ignored input regions. Any input perturbation that fits inside both scorers' joint blind spot is YF-free.

The implication is that the empirical leaderboard floor (currently ~0.31 across the top contestants, driven by C-058→C-067-style packer-layout micro-frontier search on a shared JointFrameGenerator paradigm) is bounded below by a Yousfi-Fridrich floor that is itself bounded below by the Shannon floor (which the Field's-medal Council session derived as ~0.155 for this task). The leaderboard's convergence at ~0.31 reflects practical engineering and contest-deadline dynamics, not a fundamental information limit. The Yousfi-Fridrich floor is the right theoretical target for the post-deadline paper, and the Shannon floor is the right asymptotic ceiling on what any task-aware compression scheme could achieve given infinite compute.

This framing is original to this work and is published here as a research contribution. We claim no priority over the underlying steganalysis ideas, only on the explicit naming and contest-application of the floor concept and on the empirical observation that the contest's leaderboard band is wide above the YF floor.

## Atomic Decomposition of the Compression Pipeline

The standard compression-systems abstraction treats `mask`, `model`, `pose`, `residual`, `packer`, and `layout` as monolithic components. Our late-April work refactored this into an **atomic decomposition**: every input element — at every pipeline stage, down to individual pixels in individual frames — is treated as an atom with a well-defined charged byte cost and a well-defined contribution to each score component.

Concretely, for each candidate atom `a` in the proposal space `A`:

- `bytes(a)` — the charged archive-bytes cost of including `a` in the deployed `archive.zip`
- `Δseg(a)` — the predicted change in `seg_dist` from including `a`, evaluated against a calibrated scorer-response model
- `Δpose(a)` — the predicted change in `pose_dist` from including `a`
- `interactions(a, b)` for `b ∈ A` — synergistic when `Δscore(a ⊕ b) < Δscore(a) + Δscore(b)` (overlapping benefit), antagonistic when `Δscore(a ⊕ b) > Δscore(a) + Δscore(b)` (cancellation or interference)
- `source_attribution(a)` — internal vs. external, with charged-payload-closure compliance metadata
- `archive_identity(a)` — the SHA-pinned archive bytes that result from packing `a` into the deployed archive

Atoms span all granularities: a packer-bit-budget allocation is an atom, a renderer-tensor block-quantization configuration is an atom, a per-pose-column delta-VLQ encoding is an atom, a per-pair sensitivity-weighted residual is an atom, and (in principle) a single-pixel reconstruction-target adjustment is an atom. The atomic decomposition is the natural extension of the meta-Lagrangian framing to the limit of fine-grained byte-charging.

The compression problem then becomes **combinatorial atom selection** over the proposal space `A` subject to `Σ bytes(a) ≤ B_archive` (charged-archive-size budget) and `Σ Δscore(a) + interactions ≤ S_target` (contest-score budget), with atom-waterfill (Boyd-style alternating projections) and hard-pair selection (Fridrich-style sensitivity-weighted prioritization) as the proposal-machinery policies. Atoms become **evidence** only after selected and packed atoms produce a deterministic archive that passes exact CUDA auth eval on identical bytes.

This framing illuminates several practical observations:

1. **Synergy and antagonism are first-class.** A renderer atom that improves PoseNet by reducing per-pixel pose-projection error can synergize with a pose atom that allocates bytes to the most-pose-sensitive frame indices, while antagonizing a mask atom that adds class-boundary detail at sub-(256,192) resolution that PoseNet cannot see (wasted bytes, no score change, possible interaction-induced regression elsewhere).
2. **Cross-stream interactions dominate at the frontier.** In the C-058→C-067 chain, the headroom is no longer additive single-atom improvements; it is interaction-budget management between the pose, model, and mask streams. The Joint-ADMM coordinator at `src/tac/joint_admm_coordinator.py` is the ADMM extension of this framing to four streams (representation, prediction, quantization, entropy) with explicit dual variables for cross-stream coupling.
3. **Per-pixel atoms are real but currently impractical.** The Score-Jacobian Karhunen-Loève (SJ-KL) basis primitive identified by the Fields-medal Council session is the closest realization of pixel-level atomic decomposition: it computes the top-k eigenvectors of the scorer Fisher-information matrix `F = 100·JᵀJ + 10·KᵀK` per pair, which is exactly the per-pixel sensitivity weighting that atomic decomposition predicts is optimal. It has not yet been deployed against the C-067 anchor because the deadline pressure pushed compute toward packer-layout micro-frontier work, but the implementation is pre-staged.
4. **Atoms with negative interaction with the JF blind-spot structure are pure waste.** If a candidate atom's `Δseg(a)` and `Δpose(a)` are both zero (because the atom only affects pixels inside the joint blind spot of SegNet and PoseNet), then `bytes(a) > 0` is pure rate cost with no distortion benefit. The atomic decomposition makes these waste-atoms detectable as a class, and the YF-floor framing predicts they exist in large numbers (which the empirical Q-FAITHFUL zoom-runtime collapse at score 22.15 partially confirms — a renderer that allocates capacity to JF-invisible regions can collapse the entire score because PoseNet's narrow-band sensitivity dominates).

The atomic decomposition is the theoretical complement to the Yousfi-Fridrich floor: the YF floor tells us *how low rate can go* in principle, and the atomic decomposition tells us *which atoms to select* to approach that floor in practice.

## Game-Theoretic Risk: Contest-Overfit and Premature Floor

The comma.ai contest's structure — public PRs, public leaderboard, May-3 deadline, single-archive winner — creates several incentives that, in combination, push the field toward a floor that is well *above* the information-theoretic Shannon limit:

- **Late-entrants face a path-of-least-resistance temptation**: with a deadline and a known-good 0.33 paradigm published, the rational late-entrant's expected-value-maximizing move is to clone the paradigm and squeeze incremental rate-side bytes (which is what PR #67 → PR #65 → C-067 trajectory looks like).
- **Public PR + leaderboard structure creates a one-way information ratchet**: a contestant who shares first publicly (Quantizr's `inflate.py`) provides ~all subsequent contestants free architectural blueprint. The information-sharing equilibrium is asymmetric: early sharers spend their alpha; late observers benefit. This rewards withholding except when the contest's public norms (PR template, peer discussion) reward sharing instrumentally for credit / reputation.
- **Deadline pressure accelerates convergence on the known-good**: with 48 hours remaining, every rational allocation of compute goes to packer / layout / micro-frontier search on the C-058→C-067 chain rather than to high-variance architectural exploration that might land 0.20 *or* 0.50.
- **The frontier-table reporting incentive** (every claim row needs exact archive bytes + SHA + CUDA + components + recomputed score, which we ourselves enforce as evidence policy) raises the per-experiment cost of speculative architectural moves relative to incremental packer-layout moves. Disciplined evidence reporting is the right policy for paper integrity, but it does compound the path-of-least-resistance temptation.
- **Game-theoretically, the entire contest may be hitting a ~0.31 floor that is far above the Shannon-derived ~0.18-0.22 realistic floor** because no single contestant has incentive to absorb the variance of architectural exploration when the deadline is this close and the leader-band is this competitive. The very competitiveness of the band creates the risk: the closer everyone is to each other, the higher the marginal reward for incremental moves and the lower the marginal reward for exploratory moves.

The Field's-medal Council session at `project_grand_council_FIELDS_MEDAL_shannon_floor_obsession_20260501.md` derived a realistic Shannon floor of 0.155 with paradigm-shift unblockings that no contestant has shipped: NeRV-based mask codec replacing AV1 (predicted -45 to -90 KB on the 220KB mask stream), Selfcomp's block-FP at 1.017 bpw on the 88K renderer (predicted -45 KB on the 65KB model), Score-Jacobian Karhunen-Loève residuals (predicted -0.005 distortion at 500 byte cost), Ballé hyperprior over the 5-class mask symbol stream (predicted -70 to -140 KB). None of these has reached contest-CUDA archive evidence in the public leaderboard or in this work, because the contest deadline + public-PR + leaderboard incentives push toward known-good paradigm refinement instead.

This is not a critique of any individual contestant (least of all Quantizr, whose PR #55 is the catalyst that made the paradigm legible to everyone). It is a structural observation about how multi-week public-PR contests with hard deadlines reach floors. The implication for the post-deadline paper is that the **interesting research questions extend past May 3**: which paradigm-shift candidates actually reach the Shannon floor when the deadline pressure is removed and the variance budget is restored.

## What is original to this work

In summary, the original contributions surveyed above are reproducible from the committed repository, durable past the contest deadline, and useful regardless of whether C-067's 0.31561703 score is competitive on the final-day leaderboard. The score is a small part of the project; the engineering disciplines, the reverse-engineering rigor, the cross-agent coordination protocols, the SJ-KL primitive, the meta-Lagrangian framing, the Shannon-floor derivation, and the contest game-theoretic analysis are larger.

## Compliance posture

Under the contest's published policy (every score-affecting byte must live in `archive.zip`), reusing a published payload segment from another open PR is permitted as long as: (a) the bytes are charged to the user's archive size, (b) attribution exists, and (c) no script-side or sidecar payload movement is involved. C-067 satisfies all three.

This is distinct from the practice quarantined in PR #68 / PR #70 (moving score-affecting bytes from `archive.zip` into `inflate.py` to evade archive-metering). C-067 does the opposite: bytes that PR #67 already paid for in `archive.zip` are paid for again in C-067's `archive.zip`, with full byte-charging.

## Paper / writeup wording

When citing C-067 in any public surface, use this attribution boilerplate:

> The C-067 archive composes three charged payload segments: a 219,472-byte mask segment from PR #67 `qpose14_qzs3_filmq9g_slsb1_r55` (EthanYangTW, comma.ai video compression challenge, used with attribution per the open-PR submission norms), a 55,965-byte model segment from this work's C-059 lineage (independently reimplemented QZS3-equivalent grouped variable-bit-depth FP4 packed renderer trained on the pose-manifold water-fill chain), and a 677-byte pose segment from this work's C-059 lineage (independently reimplemented QP1 codec output). The score `0.31561703078448233` is an exact CUDA T4 evaluation of the assembled archive bytes; reverse-engineering custody for the PR #67 mask source is at `reports/raw/leaderboard_intel_20260501/pr67_archive.zip` (sha `a5ed8da0...`).

Forbidden wording:
- "We achieved 0.316 with our own architecture." (Mask source is external.)
- "Our 219,472-byte mask payload..." (The mask payload is PR #67's.)
- "C-067 supersedes PR #67." (C-067 reuses PR #67's mask segment; it is not a wholly independent submission.)

Allowed wording:
- "C-067, an internal exact A++ frontier composing PR #67's published mask segment with our own model and pose segments, scores 0.316 [contest-CUDA T4]."
- "Our charged-rate improvement over C-063 (`-9 bytes`) and C-059 (`-133 bytes`) is from packer/layout optimization on the model and pose segments only; the mask segment is unchanged from PR #67."
- "The contest's leaderboard convergence on the JointFrameGenerator paradigm reflects natural signal diffusion from Quantizr's published PR #55 inflate.py, with rate-side packer-layout work as the remaining headroom."
- "Our work's durable contributions extend past the C-067 score to the engineering disciplines, reverse-engineering methodology, cross-agent coordination protocols, SJ-KL residual primitive, meta-Lagrangian framing, and Shannon-floor derivation, all reproducible from the committed repository."

## Cross-references

- Codex writeup ledger: `.omx/research/submission_writeup_integration_20260502_codex.md` (C-067 supersession + attribution requirements)
- Working notes: `reports/writeup_working.md` (live operating point + public-source handling)
- Latest report: `reports/latest.md` (frontier table + submission pipeline)
- Reverse-engineering memory: `reference_pr65_pr67_blob_byte_layouts_proper_reverse_engineering_20260501.md`, `reference_pr67_line_search_R_D_joint_coordinate_descent_20260501.md`
- Derivation notes: internal derivation ledgers summarized in this memo (S_min=0.155 derivation + SJ-KL primitive; engineering pragmatism path)
- PR #67 inflate.py reference: `reports/raw/leaderboard_intel_20260501/pr67_inflate.py`
- PR #67 archive bytes: `reports/raw/leaderboard_intel_20260501/pr67_archive.zip` (sha `a5ed8da0d9988943c986b231b4cd33cea0ab878a8e1628134341db5f7f41c765`, 276,564 bytes)
- Lane registry + maturity harness: `tools/lane_maturity.py`, `.omx/state/lane_registry.json`
- Cross-agent dispatch coordination ledger: `.omx/state/active_lane_dispatch_claims.md`

## Public-release posture

The C-067 attribution above is appropriate for the contest submission and the eventual paper. It does not expose unpublished private artifacts, internal training recipes beyond what is necessary to make the C-067 archive bytes reproducible from the committed C-059 lineage, or non-essential operational details. The PR #67 mask source-attribution is a paper-integrity requirement, not a competitive disclosure.

When the paper goes public, the bibliographic citations should include: PR #67 (`https://github.com/commaai/comma_video_compression_challenge/pull/67`, EthanYangTW), PR #55 (`https://github.com/commaai/comma_video_compression_challenge/pull/55`, Quantizr / Jimmy, UCLA CSE/Neuro), PR #65 (`https://github.com/commaai/comma_video_compression_challenge/pull/65`, henosis-us / Matt Abrahamson), and PR #56 (`https://github.com/commaai/comma_video_compression_challenge/pull/56`, szabolcs-cs / Selfcomp), with author display names as listed on each PR. The acknowledgements section should explicitly thank Quantizr for the published `inflate.py` that made the JointFrameGenerator paradigm legible to the field, and the contest organizers (comma.ai, Yassine Yousfi) for the open-PR contest format that enabled this kind of cross-contestant signal diffusion.
</content>
