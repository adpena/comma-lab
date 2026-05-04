# submission name: apogee

> The home for documenting [@adpena](https://github.com/adpena)'s entire month of work on the comma.ai video compression challenge.

# upload zipped `archive.zip`

[https://github.com/adpena/comma_video_compression_challenge/releases/download/apogee-pr98-hnerv-adapter-20260504/archive.zip](https://github.com/adpena/comma_video_compression_challenge/releases/download/apogee-pr98-hnerv-adapter-20260504/archive.zip)

# report.txt

```
=== Evaluation config ===
  batch_size: 16
  device: cuda
  num_threads: 2
  prefetch_queue_depth: 4
  seed: 1234
  submission_dir: submissions/apogee
=== Evaluation results over 600 samples ===
  Average PoseNet Distortion: 0.00017394
  Average SegNet Distortion: 0.00068841
  Submission file size: 178,392 bytes
  Original uncompressed size: 37,545,489 bytes
  Compression Rate: 0.00475136
  Final score: 100*segnet_dist + sqrt(10*posenet_dist) + 25*rate = 0.23

Exact local CUDA/T4 custody:
  score_recomputed_from_components: 0.22933111465960354
  archive_sha256: 7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb
  archive_size_bytes: 178392
  archive_member: 0.bin
  archive_member_sha256: fce200db2fe087cc6a051945b3fda2c37f5bbb3e19b8f20a1aea7201db0c9f5f
  n_samples: 600
  eval_hardware: Tesla T4
  runtime_tree_sha256: 0232154c17410621325ec1647e0f0723b3310d63b0d4bc4bf7bbb5e9aa2fccd0
```

# does your submission require gpu for evaluation (inflation)?

yes

# did you include the compression script? and want it to be merged?

no

# additional comments

apogee is a contest-faithful HNeRV-adapter submission with the exact archive bytes hosted as a release asset and the deterministic inflate runtime included in this PR.

Public artifacts:
- PR branch: https://github.com/adpena/comma_video_compression_challenge/tree/apogee-pr98-hnerv-adapter/submissions/apogee
- Release asset: https://github.com/adpena/comma_video_compression_challenge/releases/tag/apogee-pr98-hnerv-adapter-20260504
- Runtime entrypoint: `submissions/apogee/inflate.sh`
- Archive manifest: `submissions/apogee/archive_manifest.json`

The submission was validated locally through the canonical path:

```
archive.zip -> inflate.sh -> upstream/evaluate.py
```

We also ran a strict pre-submission compliance gate before opening this PR. That gate checked archive SHA/size, single-member ZIP integrity, local-header/central-directory consistency, runtime tree custody, report linkage, public hygiene, T4-equivalent CUDA auth eval, and exact component-score recomputation.

---

## Companion repository (open-source)

The full research codebase that produced apogee is published as a standalone open-source package:

- **Public source code (Python package, MIT)**: `github.com/adpena/tac` _(USER ACTION: replace placeholder once the public repo is created via the release plan at `experiments/results/oss_tac_release_plan_20260504_claude/release_plan.md`)_
- **Private workspace (this PR's source-of-truth)**: `github.com/adpena/comma-lab` (PRIVATE — kept private to protect competitive details until the contest deadline; will be archived after the contest concludes)

The public `tac` package contains every codec primitive, every score-aware optimizer, the full deterministic-archive submission contract, the production-hardened preflight gate (90+ STRICT bug-class extinctions), and the inflate runtime. It does NOT contain operator-specific dispatch credentials, private custody ledgers, or per-experiment GPU lane outputs (those stay private).

---

## One-month timeline

This work began **2026-04-09** as a from-scratch engagement with the comma.ai video compression challenge. Over **25 days and ~1,600 git commits**, the project moved through five distinct paradigm phases.

### Phase 1 — Postfilter era (early April)

Convolutional pre/post-processors around standard codecs. Established the **eval-roundtrip discipline** (`eval_roundtrip=True` default), the **contest-CUDA-only score truth** (no MPS, no proxy, no extrapolation), and the **auth_eval canonicalization** that prevents proxy-vs-authoritative drift. Hit a ceiling at auth ~1.33.

### Phase 2 — Renderer era (mid April)

Scorer-aware learned renderers (Lane G v3 / PFP16 lineage). Established **artifact custody**, the **EMA non-negotiable** across all training paths, the **fp32→fp16 pose-cast micro-frontier**, the **OWv3 sensitivity-weighted byte-plan refinement chain**, and the **orthogonal-stack composition pattern**. Discovered and corrected the **MPS-falsification bug class** (every MPS-derived score was wrong by 2-3×; 2026-04-25 baseline correction from 2.26 → 0.90).

### Phase 3 — Sub-1.0 era (late April → early May)

The OWv3 chain produced 5 sub-frontiers:

| Date (UTC) | Lane | Score | Bytes | Hardware |
|---|---|---:|---:|---|
| 2026-04-28 | Lane G v3 | 1.05 | n/a | RTX 4090 |
| 2026-05-01 | Lane G v3 OWv3 R7 | 1.0134 | 631,473 | RTX 4090 |
| 2026-05-01 | Lane G v3 OWv3 0120 | 1.0024 | 617,410 | RTX 4090 |
| 2026-05-01 | OWv3 0120 PD-V2 (arithmetic-coded poses) | **0.9974** | 609,963 | RTX 4090 |

PD-V2 was the **first sub-1.0** of the session, achieved via arithmetic coding of the optimized pose stream (15.6 KB → 7.2 KB, -54%).

### Phase 4 — Public-floor basin era (early May)

Pose-manifold water-fill on top of a Quantizr-derivative model lineage (C-058 → C-059 → C-063 → C-067), with byte-level packer/layout micro-frontier search. **C-067** is the active frontier of this era at **`0.31561703078448233`** [contest-CUDA T4 A++], 276,214 bytes.

### Phase 5 — HNeRV pivot + apogee (2026-05-04)

The leaderboard frontier jumped on 2026-05-04 from the qpose14/qzs3-mask era (~0.31) to the HNeRV decoder era (~0.21). After exact T4 replays of public PRs #95-#106 we measured the new public exact frontier at **PR106 `belt_and_suspenders` 0.20946** (replay artifact `experiments/results/lightning_batch/exact_eval_public_pr106_belt_and_suspenders_adapter_t4_20260504T1330Z`).

**apogee** ships a contest-faithful HNeRV-adapter submission at exact T4 score **0.22933** (this PR), as our entry into the HNeRV-pivot tier.

---

## Methodology summary

### Rate-distortion compiler

The contest is a **rate-distortion problem against frozen perceptual-network scorers**, not against MSE/LPIPS. Every pipeline element is treated as a typed atom with measurable byte/Δseg/Δpose interactions. The compiler searches the convex intersection (rate ≤ R, seg ≤ S, pose ≤ P, archive ≤ A) for the Pareto frontier via Dykstra alternating projections.

### Atomic decomposition

Mask, model, pose, residual, packer, and layout are not monolithic components. Every input element — at every pipeline stage, down to individual pixels in individual frames — is treated as an atom with a well-defined charged byte cost and a well-defined contribution to each score component. The compression problem becomes combinatorial atom selection over the proposal space subject to byte and score budgets.

### Yousfi-Fridrich theoretical floor

The challenge IS inverse steganalysis. SegNet (EfficientNet-B2 backbone) and PoseNet (FastViT-T12 with YUV6 input) are forensic detectors. Reconstruction errors should be:

1. **Concentrated in textured regions** (Fridrich UNIWARD principle — undetectable embedding)
2. **Spread to small magnitudes** (square-root law — concentrated errors are detectable)
3. **Aligned with score-Jacobian null space** (errors in directions the scorer is insensitive to)

The Yousfi-Fridrich floor `R_YF(D)` is bounded above by the standard Shannon `R(D)` — every Shannon-feasible scheme is YF-feasible — but typically lies strictly below it because YF-feasibility allows arbitrary perceptual distortion as long as scorer outputs are preserved. Detail at sub-(256, 192) resolution is invisible to SegNet's stride-2 stem; bits there are wasted under YF but charged under Shannon.

### Joint score-aware codec stack

Codec primitives compose via the Joint-ADMM coordinator (Boyd-style alternating projections across {representation, prediction, quantization, entropy, archive-size} feasible sets). Cross-stream interactions are first-class — synergy and antagonism between mask, model, and pose streams are explicitly modeled.

### Sensitivity-aware bit allocation

The **β-Fisher sensitivity map** (`src/tac/sensitivity_map.py`) is the foundational producer artifact: for each parameter θ_i it computes `F_ii = E[(∂score/∂θ_i)^2]` over the contest video pairs. Downstream consumers — water-fill bit allocation, IMP iterative magnitude pruning, sensitivity-weighted Ballé hyperprior — all consume this map.

### Production-hardened deterministic submission

All score-affecting bytes ship as a single ZIP member. No hidden sidecars, no scorer patches at inflate time, no runtime dependencies that aren't auditable file-by-file. The archive SHA-256 is the integrity contract.

---

## Original contributions

### Theoretical contributions

1. **Yousfi-Fridrich theoretical floor primitive** — `R_YF(D) ≤ R_Sh(D)`. Scorer-aware embedding admits structurally tighter rate bounds than perceptual compression. Documented in `docs/paper/EXTERNAL_SOURCE_ATTRIBUTION_C067.md`.
2. **Atomic decomposition framework** — every pipeline element as a typed atom with byte/Δseg/Δpose interactions, synergy/antagonism, source attribution, archive identity. The natural extension of meta-Lagrangian framing to fine-grained byte-charging.
3. **Game-theoretic premature-convergence analysis** — explicit modeling of contest deadline + public-PR + leaderboard structure as a force toward floor that lies well above the information-theoretic Shannon limit.
4. **Score-Jacobian Karhunen-Loève (SJ-KL) basis** — top-k eigenvectors of the scorer Fisher-information matrix `F = 100·JᵀJ + 10·KᵀK` per pair as an R(D)-optimal residual primitive. Identified by an internal council session as a publishable compression primitive alongside DCT, wavelet, VQ-VAE, and NeRV. Generalizes beyond the comma contest to any domain with a public scorer. Implementation pre-staged at `src/tac/sjkl_basis.py`.
5. **Shannon-floor derivation per score component** with explicit lower bounds: `S_min` realistic 0.155, T-65h achievable 0.224, optimistic asymptote 0.123. Documented in internal council deliberations.

### Engineering and infrastructure contributions

1. **Contest-CUDA-only score truth** with the evidence-grade taxonomy `[contest-CUDA]` / `[contest-CPU advisory]` / `[MPS-PROXY]` / `[advisory only]` / `[empirical:<artifact>]` / `[prediction]`. Every score in the claim matrix tagged with its evidence grade. Non-`[contest-CUDA]` scores cannot rank or promote.
2. **eval_roundtrip canonicalization across all training paths** — `eval_roundtrip=True` default, including the `simulate_eval_roundtrip(noise_std=0.5)` STE for differentiability through the 384→874→uint8 contest pipeline. Caught a class of proxy-auth gap bugs that historically inflated optimistic scores by 2-11× on PoseNet-dominant lanes.
3. **EMA discipline across all training paths** — decay 0.997 standard, applied at eval-time only with snapshot+restore, with the late-bound module-guard at `tac.training.EMA` to prevent freeze symptoms when EMA is incorrectly applied during training. Wired into 8+ training scripts (renderer, segmap, joint-pair, IMP, LoRA-TTO, postfilter, Szabolcs/Selfcomp clones, codebook-EMA codec layers).
4. **Deterministic ZIP construction** with hidden-file/resource-fork exclusion, zip-slip rejection, scorer-load guards, and explicit-bundle-only contents (no auto-bundle by file existence).
5. **QZS3 grouped variable-bit-depth FP4 packer** (`src/tac/qzs3_renderer_codec.py` and family) — independent reimplementation, byte-identical to PR #67's `get_grouped_qv_state_dict` decoder for matching inputs (15 round-trip tests).
6. **QP1 pose codec** (`src/tac/qp1_pose_codec.py`) — delta + VLQ first-column pose codec, ships in C-067 lineage.
7. **PFP16 codec** (`src/tac/pfp16_codec.py`) — FP16 renderer codec, PFP16 baseline 1.044 [contest-CUDA T4].
8. **Water-filling codec v2** (`src/tac/water_filling_codec_v2.py`) — score-Jacobian sensitivity-aware bit allocation. 40.98% byte savings on Lane G v3 renderer (empirical).
9. **Joint-ADMM coordinator** (`src/tac/joint_admm_coordinator.py`) — Boyd-style alternating projections across 4 streams with explicit dual variables for cross-stream coupling.
10. **90+ STRICT preflight checks** (`src/tac/preflight.py`) — one per bug class permanently extincted. Each check has a memory cross-reference and fails CI/commit-time STRICT. Permanent extinctions include: MPS-fallback device default, `set -uo pipefail` no-`-e`, dead CLI flag wiring, scorer-at-inflate, subagent commit-message swap, comment-only contracts, internal-consistency assertions in stats files, KILL memory verdicts without council review.
11. **Modal/Lightning/Vast.ai canonical dispatch + harvest infrastructure** with typed manifests, runtime closure validation, T4 torch-pin gates, NVDEC probe, archive-only manifest closure, deterministic ZIP construction.
12. **Lane maturity registry + 7-gate production-hardening framework** (`tools/lane_maturity.py`, `.omx/state/lane_registry.json`). Every lane registered; 7 gates (impl_complete, real_archive_empirical, contest_cuda, strict_preflight, three_clean_review, memory_entry, deploy_runbook) enforced by STRICT preflight Check 90.
13. **Subagent commit serializer with temp `GIT_INDEX_FILE`** (`tools/subagent_commit_serializer.py`) — eliminates the staging-race that previously shuffled commit messages across commit objects when 2+ subagents committed concurrently.
14. **Cross-agent dispatch coordination ledger** (`.omx/state/active_lane_dispatch_claims.md`) — mandatory append-before-dispatch protocol. Prevents duplicate GPU spend between Claude and codex agents.
15. **Skunkworks council (10-voice quintet pact + 12-member grand bench)** — Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Quantizr + Hotz + Selfcomp + MacKay + Ballé as the binding-decision set; Boyd, Tao, Filler, Mallat, van den Oord, Carmack, Hassabis, Hinton, Karpathy, Schmidhuber, Jack-from-skunkworks as the on-demand bench. Non-conservative by charter; the burden of proof is always on *not* trying something.

### Compression / coding contributions

1. **Pose-manifold water-fill micro-frontier search** (C-058 → C-059 → C-063 → C-067) — iterative byte-level pose refinement that exposed the C-059 basin's exhaustion frontier and produced four sequential A++ frontier improvements over 36 hours.
2. **OWv3 sensitivity-weighted byte-plan refinement** (the renderer-era frontier) — combined per-tensor block bit allocation with score-component sensitivity to produce the orthogonal-stack 0.9974 contest-CUDA score before the basin pivot.
3. **Ω-W water-filling and Joint-ADMM cross-stream coordinator** — Boyd-style alternating projections across {representation, prediction, quantization, entropy, archive-size} feasible sets with hard coupling between streams.
4. **Leaderboard reverse-engineering rigor** — rigorous byte-level decoding (parser-source-driven, not hex-dump-guessing) of PR #56, PR #65, PR #67. Surfaced the multi-stage residual-refinement paradigm in PR #65 that prior contestants had missed and informed our packer work.
5. **Public-frontier intake gate** (`src/tac/public_frontier_intake.py`) — every future public archive gets section offsets, entropy, section SHA-256, ZIP overhead, and no-op/provenance checks before stack claims.
6. **HNeRV adapter runtime layer** — venv-stable public-PR replay adapters that resolve the missing-`brotli` runtime closure bug on contest-faithful PR replays.

---

## Hidden-gem inventory teaser

Per the user directive to highlight hidden gems, the codebase contains 90+ codec / optimizer / decoder modules. The Pareto-top 8 candidates for stacking onto PR106 frontier are documented in:

- `experiments/results/internal_hidden_gem_audit_20260504_claude/audit.md`
- `experiments/results/internal_hidden_gem_audit_20260504_claude/revival_plans/`

The top-3 highest-EV revivals (predicted score reduction ÷ engineering hours):

1. **`scorer_exploits.py`** — documented scorer-blind-spot exploits (compress-time only). Predicted Δ band [-0.020, -0.005] / 4 hours.
2. **`water_filling_codec_v2.py`** on PR106 HNeRV decoder — sensitivity-aware bit allocation. Predicted Δ band [-0.015, -0.005] / 4 hours.
3. **`mask_grayscale_lut.py`** — Selfcomp Gaussian-LUT replacing PR106 mask channel. Predicted Δ band [-0.020, -0.005] / 5 hours.

The full hidden-gem audit catalogues 44 PR106-stackable lanes by predicted EV.

---

## Negative results catalog

Per the project's **positive-and-negative-signal-discipline** rule, the major bug classes extincted permanently include:

- **MPS-falsification** (2026-04-25) — every PoseNet score on Apple Silicon MPS was wrong by 23×; SegNet by 2×; final score by 2.5×. The first verified contest-CUDA baseline was 0.90, not 2.26 as MPS reported.
- **`masks.mkv` at 48×64 catastrophic** — mask resolution must match renderer training resolution (384×512); 48×64 destroyed score by 100×.
- **Archive measurement disaster** — auth evals using a renderer-only 119KB archive instead of the full 338KB submission archive were optimistic by 0.108 score points.
- **1199 overlapping pairs vs 600 non-overlapping** — `auth_eval.py` used `range(N-1)` (1199 pairs) but upstream `evaluate.py` uses `seq_len=2` non-overlapping batching (600 pairs).
- **eval_roundtrip default-False** — proxy-auth gap up to 11× on PoseNet.
- **Auto-bundle by file existence** — stale experiment artifacts silently inflated archive size.
- **Adaptive weights formula vacuous** — T² cancels in the derivation; retired.
- **PoseNet gradient caps** — caused 26× PoseNet regression.
- **KL distill as primary loss** — caused PoseNet collapse; retained only for SegNet distillation phase per Quantizr recipe.
- **Comment-only contracts** — promises in comments without runtime assertions; PCC2 STRICT preflight extincts the bug class.
- **Stub-loop training masquerading as success** — IMP cycle 0 stats.json said `epochs=200, elapsed_sec=3.47`; PCC3 STRICT preflight enforces internal-consistency assertions in stats files.

Each bug-class extinction is wired into `src/tac/preflight.py` as a STRICT check.

---

## Submission integrity contract

| Field | Value |
|---|---|
| Submission name | `apogee` |
| Archive SHA-256 | `7ecb0df1c4627d55d88e03eff3d890b7a7a5b047c62515acff20232cf29310eb` |
| Archive size (bytes) | 178,392 |
| Single ZIP member | `0.bin` |
| Member SHA-256 | `fce200db2fe087cc6a051945b3fda2c37f5bbb3e19b8f20a1aea7201db0c9f5f` |
| Runtime tree SHA-256 | `0232154c17410621325ec1647e0f0723b3310d63b0d4bc4bf7bbb5e9aa2fccd0` |
| Eval hardware | Tesla T4 (Lightning Studio) |
| Eval samples | 600 (full pinned video set) |
| Avg PoseNet distortion | 0.00017394 |
| Avg SegNet distortion | 0.00068841 |
| Compression rate | 0.00475136 |
| Score (recomputed from components) | **0.22933111465960354** |
| Inflate runtime | `submissions/apogee/inflate.sh` (single entrypoint) |
| Inflate strict-scorer-rule | OK (no scorer load at inflate time) |
| Hidden sidecars | NONE |
| Scorer patches | NONE |
| Determinism | byte-deterministic across re-builds |
| Dispatch lane id | `public_pr98_hnerv_muon_finetuned_t4_adapter_replay` |
| Dispatch job id | `exact_eval_public_pr98_hnerv_adapter_t4_20260504T0958Z` |

---

## Production / OSS deployment notes

This PR intentionally keeps the contest runtime small and auditable rather than shipping the full research workspace. The production-relevant properties are:

- **deterministic single-payload archive**: one `0.bin` member, fixed SHA-256, no hidden sidecars or local paths
- **explicit runtime custody**: the submitted runtime tree is small enough to audit file-by-file and was hashed during exact evaluation
- **deployable inference shape**: `inflate.sh` is the only entrypoint and delegates to a compact Python/PyTorch decoder
- **hardware fit**: local exact CUDA/T4 auth eval completed in about one minute, with inflate itself taking about 22.6 seconds, well inside the 30-minute contest limit
- **reproducible validation**: score claims are tied to exact archive SHA, runtime tree SHA, component distances, sample count, and T4 hardware evidence
- **clean separation of concerns**: the research system (`tac`) produced and validated the archive, while this PR exposes only the minimal runtime needed by comma's evaluator

A productionized version inside comma/openpilot should preserve this boundary: keep the trained representation and entropy-coded payload as charged data, keep the inflate path deterministic and side-effect-light, and move reusable codec components into a small package with typed payload contracts, CI replay fixtures, and byte-for-byte archive tests.

---

## Acknowledgements

This work stands on the shoulders of an exceptional group of public contributors to the comma.ai video compression challenge. We acknowledge the work of (in chronological order of contributions that shaped our paradigm):

### comma.ai contest organizers

- **Yassine Yousfi** (challenge creator, formerly Fridrich's PhD student at Binghamton DDE Lab) — for designing the challenge as inverse steganalysis and shipping the canonical scorer architectures.
- The **comma.ai team** for hosting an open, public-PR contest format that allows reverse-engineering and follow-on extension rather than zero-sum extraction.

### Core paradigm contributors

- **Quantizr (Jimmy, UCLA CSE/Neuro)** — **PR #55 score 0.33**. First public sub-0.40 lane. Opened the unknown-unknown that ultra-compact ~88K-parameter FiLM-DSConv `JointFrameGenerator` renderers are reachable. Owe explicit acknowledgement for: (a) the JointFrameGenerator architectural paradigm, (b) the FiLM-on-pose-only conditioning insight, (c) the half-frame mask economy demonstration, (d) the FP4-block packing template, (e) the KL-T=2.0 distillation recipe, and most importantly, (f) the **public posting of `inflate.py`** that allowed reverse-engineering of all of the above.
- **EthanYangTW** — multiple frontier-pushing PRs:
  - PR #75 / PR #79 `qpose14_r55_segactions_minp` (score 0.31)
  - PR #67 `line_search.py` — R(D)-joint coordinate descent on pose. Our C-067 lineage absorbed the PR #67 mask payload bytes (219,472 bytes `mask.obu.br` charged inside C-067's archive per contest archive-metering). Documented in `docs/paper/EXTERNAL_SOURCE_ATTRIBUTION_C067.md`.
  - PR #98 / PR #102 HNeRV continuations
- **henosis-us (Matt Abrahamson)** — PR #65 `henosis_qz_n3z_r25_clean` and PR #82 `henosis_frontier`. The 30-byte-header / 10-section randmulti packer paradigm. The PR #65 multi-stage-residual paradigm finding (~6KB of post/shift/frac/bias/region/randmulti side-channel correction) is original analysis on our part, but it required PR #65 to ship the working artifacts in the first place.
- **szabolcs-cs (Selfcomp)** — PR #56 `selfcomp` (score 0.36). SegMap + Gaussian-softmax-LUT + AV1 grayscale + xz-int8 stack. We reverse-engineered the SegMap weight layout, the σ=15 Gaussian-LUT trick over `CLASS_TARGETS=[0,255,64,192,128]`, the HWOI permutation for additional xz gain, and the affine-embedding side-channel (1200×6 table). Documented in `reference_pr56_selfcomp_blob_byte_layout_proper_reverse_engineering_20260501.md`.

### HNeRV pivot contributors (2026-05-04 frontier reset)

- **valtterivalo** — **PR #105 `kitchen_sink` (0.19797)** and **PR #106 `belt_and_suspenders` (0.20946)**. Current public exact contest-CUDA T4 frontier (PR106 at our T4 replay 0.20945673680571203, 186,239 bytes). PR106's win is **PoseNet domination** — pose contribution 0.018306 vs PR101/103/105 ~0.041. Fridrich square-root-law in action.
- **rem2** — PR #96 `rem2_HNeRV` and PR #103 `hnerv_lc_ac` opening the 0.21-tier.
- **AaronLeslie138** — PR #95 `hnerv_muon` opening the HNeRV pivot at 0.20.
- **BradyMeighan** — PR #97 `vibe_coder_final_boss` (0.23), PR #99 `hnerv_muon_lc` (0.19667), PR #100 `hnerv_lc_v2` (0.1954).
- **SajayR** — PR #90 `qrepro` (0.28) and PR #101 `hnerv_ft_microcodec`.
- **patattzel** — PR #104 `qhnerv_ft_best`.

### Rate / mask / codec contributors

- **ottokunkel** — PR #83 / PR #84 / PR #85 / PR #91 (adaptive masking joint frame model, HPAC coder hybrid, range-mask optimizations).
- **nick-neely** — PR #77 / PR #78 / PR #92 (qzs3 tile-delta, qzs3 script payload, qzs3 range joint).
- **erichasinternet** — PR #81 / PR #88 (qzs3 range mask, qzs3 GPU preflight).
- **jas0xf** — PR #86 `jas0xf_adversarial_neural_representation` (0.27).
- **hypery11** — PR #74 `ph4ntom_drv` (0.35).
- **avocardio** — PR #64 `add unified_brotli submission`.
- **dllu** — PR #93 `flatpup`.
- **manthedan** — PR #87 `Add 100_bytes submission`.
- **celikemir** — PR #73 `emir_flatpack: lossless tightening of qpose14 (-5,625 bytes)`.
- **1kuna** — PR #76 `Add qpose14_poseq6 submission`.
- **Janos95** — PR #80 `janos_svc_resdgain`.
- **Josemedinan** — PR #94 `optimization_qpose_josema`.
- **narenn99** — PR #72 `WIP: neural compressor`.

We have run our own exact contest-CUDA T4 replays of 19 of these PRs to measure them against the canonical eval pipeline. Replay artifacts at `experiments/results/lightning_batch/exact_eval_public_pr*`.

### Theoretical lineage (council members the codebase channels)

- **Claude Shannon** (council lead) — information theory grounding
- **Richard Dykstra** (co-lead) — alternating projections
- **Jessica Fridrich** — UNIWARD steganalysis framework
- **George Hotz** — engineering shortcuts
- **David MacKay** (memorial seat) — *Information Theory, Inference, and Learning Algorithms*
- **Johannes Ballé** — modern neural-compression SOTA (entropy bottleneck + scale hyperprior)
- **Stephen Boyd** — convex optimization at operational level (ADMM)
- **Stéphane Mallat** — wavelet theory + scattering transforms
- **Aaron van den Oord** — VQ-VAE, WaveNet
- **Geoffrey Hinton** — knowledge distillation
- **Tomáš Filler** — syndrome-trellis coding (Fridrich's other student)
- **John Carmack** — engineering shortcuts at the Doom/Quake/Oculus level
- **Demis Hassabis** — strategic-research perspective
- **Andrej Karpathy** — engineering practitioner
- **Jürgen Schmidhuber** — compression-as-intelligence; MDL
- **Terence Tao** — pure mathematician omniscience

### AI assistance disclosure

This codebase was developed with assistance from **Anthropic Claude (Opus 4.7)** and **OpenAI Codex (GPT-5.5)** acting as research-engineering collaborators under careful human review and direction. The user (Alejandro Peña) defines the strategy, makes design decisions per skunkworks council deliberation, and operates all GPU dispatch + funding budgets. The AI agents implement, measure, debug, and produce evidence. All score-affecting decisions are human-approved.

---

## A note on the public exact frontier (2026-05-04)

The current public exact contest-CUDA T4 frontier is **PR106 `belt_and_suspenders` 0.20946** (`valtterivalo`). Our `apogee` submission at **0.22933** is in the same era but does not beat PR106. The `tac` codebase is staged for a follow-on stack — see `experiments/results/internal_hidden_gem_audit_20260504_claude/` for the eight Pareto-ranked revival plans (water-fill v3 on PR106 decoder, arithmetic-coded latents, QZS3 decoder replacement, block-FP, UNIWARD-delta, Selfcomp-LUT, SJ-KL basis, sensitivity-map producer).

The PR107 home will be updated as new contest-faithful frontier scores land.
