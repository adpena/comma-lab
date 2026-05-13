# Optimal Training Curriculum Across All Frontier Tracks

Date: 2026-05-13
Author: Codex xhigh research subagent
Scope: aggressive exact-score lowering below the current approximately 0.19 frontier
Status: research_only=true, score_claim=false, promotion_eligible=false, dispatch_authorized=false

This memo designs the training and curriculum program for the current all-track
frontier campaign. It treats the author-provided PR95 HNeRV/Muon source present
in this repository as the control arm, not as hearsay, and uses it as the
baseline to beat. The target is a byte-closed, exact-evaluable path toward
sub-0.17 first and sub-0.15 second. All score movements below are forecasts or
experiment-prior ranges, not contest claims.

## Preflight And Evidence Read

Required repository files were read before conclusions: `CLAUDE.md`,
`AGENTS.md`, `.omx/state/lane_registry.json`, and the required current ledgers:

- `.omx/research/pr95_curriculum_recovery_20260513_codex.md`
- `.omx/research/hnerv_nerv_options_and_timeline_20260513_codex.md`
- `.omx/research/instruction_state_anti_conservatism_audit_20260513_codex.md`
- `.omx/research/frontier_long_burn_campaign_reset_20260513_codex.md`
- `.omx/research/sub017_frontier_innovation_roadmap_20260513_codex.md`
- `.omx/research/siren_literature_review_20260513.md`
- `.omx/research/siren_activation_family_contract_20260513_codex.md`

Recent `*directive*` ledgers from the last 24 hours were checked; none were
present. The lane registry has many related L0/L1/L2 lanes already registered,
including PR95/HNeRV, PR101 packet-grammar, PR103 arithmetic coding, NeRV-family
replacement lanes, SIREN-family residuals, Balle/CompressAI sketches,
Cool-Chic/C3 sketches, RAFT/ego-motion, LA-pose/foveation, and PacketIR
compiler surfaces. Dispatch still requires an explicit `tools/claim_lane_dispatch.py`
claim before any provider job.

The repository-root `MEMORY.md` required by the preflight rule was not present
in the checkout; only an older `.omx/auto_memory_snapshot_20260504T230223Z/MEMORY.md`
was found. This is a state-surface gap, not a blocker for this memo.

The author-provided PR95 source was inspected directly at:

`experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/`

PR100 and PR101 source trees were inspected where needed for packet and sidecar
comparison.

## Executive Recommendation: Top 10 Paths

Rank is by expected exact-score movement per wall-clock day after the first
timing smoke, weighted by contest compliance and implementation readiness.
Ranges are planning priors, not score claims.

| Rank | Path | Expected movement | First proof gate | Cost/time prior | Why it is first |
|---:|---|---:|---|---|---|
| 1 | PR95 faithful reproduction plus PR101/PR103 archive export | -0.005 to -0.025 | 10 to 50 epoch PR95 smoke, parsed archive roundtrip, then stage replay | 50 to 200 GPUh full path; smoke under 3 GPUh | PR95 is author-provided source, already near 0.20; PR101 shows packet/export discipline can improve it without changing representation |
| 2 | PR95 curriculum mutations with control-preserving ablations | -0.003 to -0.020 | same seeds, same archive grammar, per-stage best archive manifests | 10 to 40 GPUh per short mutation; only winners go full | The source curriculum is good but not fully searched: sampling, loss weights, QAT timing, EMA, and Muon phase are obvious high-EV knobs |
| 3 | PR100/PR101 sidecar search expanded into typed residual atoms | -0.001 to -0.012 | no-op controls, decoded-frame parity checks, exact component recompute | CPU to low GPU, many small searches | Sidecars changed SegNet/PoseNet cheaply; expand from hand-coded corrections to constrained sparse atoms |
| 4 | PacketIR compiler: schema, stream permutation, ANS/range/arithmetic coding | -0.0005 to -0.008 | byte-diff plus inflate parity, then exact CUDA | Mostly CPU | The PR101/PR103 direction is low risk if typed and byte-closed; rate-only sub-0.17 is hard, but free bytes compound with every model |
| 5 | Sparse hard-region residual INR atoms: SIREN/FINER/WIRE/BACON over HNeRV residuals | -0.002 to -0.020 | hard-pair residual fit, atom packet parser, ablate per atom | 5 to 40 GPUh per family | Full-frame SIREN is likely dominated, but small atoms over scorer-sensitive errors can buy component movement at controlled bytes |
| 6 | NeRV-family replacements: HiNeRV, FFNeRV, DS-NeRV, NeRV++, frequency/ego variants | -0.005 to -0.040 | 200 epoch timing smoke, byte-closed export, HNeRV-control matched eval | 20 to 200 GPUh per promoted architecture | Higher ceiling than PR95 polishing if archive grammar stays under control |
| 7 | LA-pose plus Telescope foveation plus RAFT ego-motion priors | -0.002 to -0.030 | motion/foveation feature manifest, renderer conditioning ablation | 5 to 80 GPUh | Scorer is not uniform over pixels or pose geometry; spend bits where PoseNet and SegNet are most sensitive |
| 8 | Balle/CompressAI/Cheng hyperprior path as full replacement or residual codec | -0.005 to -0.050 | tiny train-on-one-video smoke, deterministic entropy model export, no sidecars | 20 to 300 GPUh | Learned image/video compression has strong priors, but contest runtime and one-video overfit need strict pruning |
| 9 | Cool-Chic/C3 overfitted learned codec path | -0.003 to -0.030 | frame/pair residual smoke, decoder budget manifest, byte accounting | 20 to 150 GPUh | Designed for overfitted compression; likely best as residual/foveated path before full replacement |
| 10 | Score-aware first-principles stack: SABOR, S2SBS, PoseNet inverse-craft, equivalence classes, GEPA/Muon proposal search | -0.001 to -0.050, very high variance | falsifiable probes with promotion-ineligible labels until byte-closed | CPU to high GPU depending probe | Potentially breaks the local basin, but must be fenced by exact-eval, scorer-free inflate, and anti-overfit compliance |

Immediate campaign order:

1. Build and time a canonical PR95 control runner that exactly preserves the
   author source semantics.
2. Add archive-in-loop export variants that can emit PR95, PR100-style sidecar,
   PR101 microcodec, and PR103 arithmetic/range-coded packets from the same
   scored state.
3. Run short control-vs-mutation fanout; only mutate against a fresh control.
4. In parallel, run small timing smokes for replacement substrates and residual
   atom paths so the campaign does not collapse back into local HNeRV polish.

## Exact PR95 Control Curriculum

Primary evidence is the local author-provided PR95 source and its README:
`submissions/hnerv_muon`. The public PR is
[commaai/comma_video_compression_challenge#95](https://github.com/commaai/comma_video_compression_challenge/pull/95).

Author-reported public metadata:

- Title: `hnerv_muon submission (0.20)`
- Reported file size: 178,417 bytes
- Reported pose distance: 0.00003494
- Reported segmentation distance: 0.00061212
- Reported final score: approximately 0.1987 to 0.20, public rounded
- README states 178 KB archive, 229K-parameter HNeRV decoder, 28-dimensional
  per-frame-pair latents, and about 50 hours on a single GPU.

The control architecture:

- Decoder: HNeRV-style frame-pair renderer.
- Latents: one 28-dimensional latent per adjacent frame pair, 600 pairs total.
- Base grid: 6 by 8.
- Base channels: 36.
- Upsampling: repeated Conv2d, PixelShuffle(2), bilinear skip, sine activation.
- Output: two RGB frame heads, rendered at 384 by 512 and bicubic-upsampled to
  camera resolution before round/clamp.
- Parameters: approximately 229K.
- Scorer path: training patches `rgb_to_yuv6` so PoseNet gradients are not
  severed; this is not optional.

The exact control curriculum from the local source:

| Stage | Epochs | Optimizer | Loss | QAT | Key settings |
|---:|---:|---|---|---|---|
| 1 | 3000 | AdamW lr 1e-3 | Seg CE plus pose | no | random init, latent lr multiplier 10 |
| 2 | 5650 | AdamW lr 1e-3 | tau Softplus CE, tau 0.3 | no | inherits stage 1 |
| 3 | 1500 | AdamW lr 1e-4 | smooth disagreement | no | component-shaping stage |
| 4 | 500 | AdamW lr 1e-4 | smooth disagreement | yes | first archive-realistic fake quant |
| 5 | 9000 | AdamW lr 3e-5 | L7 Softplus plus C1a entropy | yes | lambda 0.01, sigma 0.2 |
| 6 | 2000 | AdamW lr 3e-5 | L7 Softplus plus C1a entropy | yes | lambda 0.02, sigma 0.2 |
| 7 | 3000 | AdamW lr 3e-5 | L7 Softplus plus C1a entropy | yes | lambda 0.02, sigma 0.1 |
| 8 | 5000 | AdamW lr 1e-5 plus Muon lr 2e-4 | L7 Softplus plus C1a entropy | yes | Muon weight decay 5e-4 |

Control training mechanics that must be preserved in the first reproduction:

- Pair sampling: uniform shuffle over all frame pairs each epoch, batch size 8.
- EMA: decay 0.999, archive export and best selection use EMA parameters.
- Eval cadence: every 25 epochs, build archive from EMA, parse it back, render
  from parsed objects, score the parsed runtime, and update best archive.
- QAT: per-tensor symmetric int8 fake quant for eligible tensors, restored
  before optimizer step.
- C1a entropy proxy: soft histogram entropy over normalized tensor weights,
  Gaussian bandwidth controlled by sigma.
- Archive grammar: metadata brotli, decoder int8 tensor streams with scales,
  and latent per-dimension uint8 plus temporal delta streams.
- Inflate: scorer-free, deterministic, batches frames, uses CUDA if available
  but must also run on CPU.

This is the control arm. Any mutation that cannot beat a freshly reproduced
control on the same axis is not evidence against PR95; it is just a failed
mutation.

## What To Mutate First

The highest-EV PR95 mutations are those that preserve the control evidence
surface while changing one tractable bottleneck at a time.

### M1: Archive-In-Loop Export Mutations

Add one export adapter that can take the same trained decoder/latent state and
emit:

- PR95 original grammar.
- PR100 schema-driven grammar with optional latent-correction sidecar.
- PR101 fixed microcodec grammar with channel postprocess and compact sidecar.
- PR103 arithmetic/range/ANS-coded streams.

Metric: same parsed-state score, archive bytes, member SHA-256, inflated-output
aggregate SHA, and component deltas. This should be the first mutation because
it can harvest packet gains without retraining and forces all later tracks into
a shared typed byte compiler.

### M2: EMA And Selection Mutations

The control uses one EMA decay and selects by parsed archive score every 25
epochs. Mutate:

- EMA decay: 0.995, 0.997, 0.999, 0.9995.
- Multi-EMA shadow weights: export all decays at eval time, train one model.
- Selection metric: exact score, component-weighted frontier target, and
  Pareto archive of `(seg_dist, pose_dist, bytes)`.
- Eval cadence: 10 epochs in late stages, 25 in early stages.

Expected effect: mostly free, often catches narrow late-stage minima. Risk:
over-selecting noisy proxy estimates if parsed archive score is not used.

### M3: Stage 5 To 8 Entropy And QAT Mutation

The PR95 C1a penalty is global and weight-only. Mutate:

- Lambda schedule: ramp 0.0 to 0.03 instead of hard stage jumps.
- Sigma anneal: 0.25 to 0.08 over stages 5 to 8.
- Per-section entropy multipliers from `src/tac/analysis/hnerv_packet_sections.py`.
- Latent-rate proxy added to C1a, not just decoder weights.
- QAT warmup instead of abrupt enable at stage 4.

Do not change the scorer target and rate target simultaneously. One control
dimension must remain stable.

### M4: Pair Sampling And Hard-Pair Curriculum

The control uses uniform pair shuffles. Mutate:

- Hard-pair replay from PR95/PR101 component residuals.
- SegNet-boundary weighted pairs from `src/tac/analysis/segnet_boundary_marginals.py`.
- Pose-sensitive replay from local PoseNet residuals and RAFT/ego-motion
  features.
- Stratified sampling: 70 percent uniform, 20 percent hard component, 10 percent
  stability/no-op controls.

Gate: hard-pair gains must still improve all-pair exact score. A hard-pair
mutation that only improves the sampled subset is classified as proxy leakage.

### M5: Optimizer Schedule Mutation

The PR95 final Muon stage is strong but late. Mutate:

- Muon from stage 5 onward on matrix weights only.
- Two-cycle AdamW to Muon schedule: AdamW basin, Muon polish, AdamW dequant
  repair, Muon final.
- Lower Muon lr grid: 5e-5, 1e-4, 2e-4, 4e-4.
- Decoupled latent optimizer with lower lr in stage 8 to avoid latent-byte
  inflation.

Sources: Muon implementation and modded nanoGPT work by Keller Jordan:
[Muon](https://github.com/KellerJordan/Muon) and
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt).

### M6: Latent Grammar Mutation

The control quantizes latents per dimension to uint8 with temporal deltas.
Mutate:

- Predictive latent coding: previous pair, low-rank temporal basis, and segment
  resets at high-motion cuts.
- Per-dimension bit allocation: 4 to 8 bits by sensitivity map.
- Learned latent reorder and stream grouping before brotli/range coding.
- Small sparse correction sidecar constrained to the PR100/PR101 style.

Gate: correction sidecar must prove non-no-op payload mutation and preserve
full-frame inflate parity with its parser.

## Control-Vs-Mutation Experiment Matrix

| ID | Track | Control | Mutation | Stage budget | Success gate | Kill condition |
|---|---|---|---|---:|---|---|
| C0 | PR95 | author source exact, one seed | none | 10, 50, then full | parsed archive score trend matches source order of magnitude | cannot run scorer or archive parser |
| C1 | PR95 | C0 | PR101/PR103 export from same state | no retrain | lower bytes with same decoded state or equal/lower exact score | decoded frame mismatch not explained by intended postprocess |
| C2 | PR95 | C0 | multi-EMA export | same training | best EMA beats control at same eval point | no EMA improves across two stages |
| C3 | PR95 | C0 | entropy lambda/sigma anneal | stages 5 to 8 | lower score and no component collapse | byte gain causes larger component loss |
| C4 | PR95 | C0 | hard-pair sampler | stages 3 to 8 | all-pair exact improvement | sampled subset improves, global score regresses |
| C5 | PR95 | C0 | earlier Muon | stages 5 to 8 | late-stage convergence speed or final score improves | archive bytes inflate or oscillation persists |
| C6 | Sidecar | PR101 sidecar | sparse residual atom grammar | CPU/GPU search | component improvement per byte beats PR101 | sidecar is no-op or non-canonical |
| C7 | PacketIR | PR101 fixed grammar | ANS/range/section permutations | CPU | byte reduction with inflate parity | parser fragility or runtime budget breach |
| C8 | INR residual | HNeRV parsed output | SIREN/FINER/WIRE/BACON atoms | 200 to 2000 epochs | targeted hard-region component gain per byte | full-frame model exceeds byte budget |
| C9 | NeRV replacement | PR95 C0 | HiNeRV/FFNeRV/DS-NeRV/NeRV++ | 200 epoch smoke | rate-distortion slope beats PR95 early | no byte-closed export plan after smoke |
| C10 | Motion/foveation | PR95 C0 | LA-pose/RAFT/Telescope conditioning | feature smoke then stage train | same bytes improve pose/seg sensitive zones | proxy-only motion feature not embedded or reproducible |
| C11 | Hyperprior | HNeRV control | Balle/CompressAI/Cheng residual/full | tiny full video smoke | entropy model compresses residual better than PacketIR | runtime dependencies cannot be closed |
| C12 | Cool-Chic/C3 | HNeRV control | overfitted residual codec | 1000 frame-pair smoke | residual gain per byte beats sparse INR | decoder overhead dominates |

Every row must write a manifest with archive bytes, SHA-256, runtime tree SHA,
hardware, command, sample count, component deltas, axis label, and dispatch
claim status. Rows C9 to C12 are not allowed to retire the path from a short
negative; they only retire the measured configuration.

## Training Program By Track

### Track A: PR95 Faithful Reproduction And Mutations

Purpose: establish a fresh, exact, local control and then beat it with archive,
curriculum, and optimizer mutations.

Implementation targets:

- New reusable module: `src/tac/substrates/pr95_hnerv_muon/`
- Trainer CLI: `experiments/train_substrate_pr95_hnerv_muon.py`
- Archive helpers: `src/tac/pr95_hnerv.py` or
  `src/tac/substrates/pr95_hnerv_muon/archive.py`
- Tests: existing PR95 recovery test plus new parser/export golden vectors
  under `src/tac/tests/`
- Packet adapters:
  `src/tac/packet_compiler/pr100_schema_driven_decoder.py`,
  `src/tac/packet_compiler/pr101_sidecar_grammar.py`,
  `src/tac/packet_compiler/pr103_arithmetic_coding.py`,
  `src/tac/packet_compiler/deterministic_compiler.py`

Program:

1. Port source without changing math. Preserve stage definitions, loss names,
   tensor shapes, fake quantization, EMA, eval cadence, and archive parser.
2. Add a `--timing-smoke-epochs` path that runs stages 1 to 8 with scaled epoch
   counts but still exports parsed archives.
3. Add archive export mode that emits PR95 original, PR100-like, PR101-like,
   and PacketIR streams from the same in-memory state.
4. Add control seeds: at least seeds 0, 1, 2 for short smoke; one seed for full
   reproduction first.
5. Add mutation grid only after C0 produces a parsed archive and trend.

Loss schedule:

- Preserve PR95 losses for C0.
- Mutation C3: change only lambda/sigma or only latent-rate proxy in one run.
- Mutation C4: keep loss fixed and change only pair sampling.
- Mutation C5: keep loss/sampler fixed and change only optimizer phase.

Optimizer:

- Control: AdamW then Muon exactly as PR95.
- Mutation: earlier Muon, smaller Muon lr, matrix-only Muon, latent AdamW
  decoupled.

Export and selection:

- Always select by parsed archive score, not raw in-memory model.
- Store best-by-score, best-by-seg, best-by-pose, and best-by-bytes snapshots.
- Keep EMA and non-EMA export for diagnostics; only parsed archive can be a
  promotion candidate.

Expected output after the first day:

- One timing-smoke manifest with seconds per epoch and seconds per eval export.
- One parsed archive from the control path.
- One export comparison table across PR95, PR100-like, PR101-like, and PacketIR.

### Track B: HNeRV/NeRV-Family Replacements

Purpose: avoid the trap of only polishing the PR95 local basin. Replacement
architectures should be judged by early rate-distortion slope, not by full-run
completion alone.

Primary sources:

- NeRV: [arXiv:2110.13903](https://arxiv.org/abs/2110.13903), original
  coordinate-to-video frame representation.
- HNeRV: [arXiv:2304.02633](https://arxiv.org/abs/2304.02633) and
  [GitHub](https://github.com/haochen-rye/HNeRV), hybrid content embeddings.
- HiNeRV: [arXiv:2306.09818](https://arxiv.org/abs/2306.09818) and
  [GitHub](https://github.com/hmkx/HiNeRV), hierarchical encoding.
- FFNeRV: [arXiv:2212.12294](https://arxiv.org/abs/2212.12294), flow-guided
  frame-wise neural representations.
- E-NeRV: [arXiv:2207.08132](https://arxiv.org/abs/2207.08132), efficient
  neural video representation.
- DS-NeRV: [arXiv:2403.15679](https://arxiv.org/abs/2403.15679), distribution
  smoothing for INR video.
- NeRV++: [arXiv:2402.18305](https://arxiv.org/abs/2402.18305), enhanced NeRV
  formulation.

Implementation targets already visible:

- `src/tac/substrates/hi_nerv/`
- `src/tac/substrates/ff_nerv/`
- `src/tac/substrates/ds_nerv/`
- `src/tac/substrates/block_nerv/`
- `src/tac/substrates/tc_nerv/`
- `experiments/train_ffnerv_as_renderer.py`
- `experiments/train_lane_12_v2_nerv_as_renderer.py`

Curriculum:

1. Shared PR95 scorer-aware data loader and eval-roundtrip path.
2. Warm start with CE/pose loss and no QAT for 200 epochs.
3. Add temporal or hierarchical structure after the smoke demonstrates speed.
4. Introduce QAT at the same loss level as PR95 stage 4, not before.
5. Add C1a/latent entropy only after reconstruction is within a component
   trust region.
6. Export through PacketIR, not architecture-specific ad hoc archives.

Architecture-specific hypotheses:

- HiNeRV: try hierarchical latent grids for low-frequency structure plus small
  per-pair residuals. Byte risk is metadata and multi-scale tensor overhead.
- FFNeRV: use flow or motion features to avoid wasting bytes on easy temporal
  persistence. Must embed or deterministically derive any motion feature.
- DS-NeRV: use distribution smoothing as a training regularizer, but gate by
  archive-real QAT behavior.
- NeRV++/E-NeRV: use only if their parameterization gives better bytes per
  component gain than PR95 at 200 to 1000 epochs.
- Frequency-enhanced NeRV: reserve high-frequency channels for hard regions
  and scorer-sensitive boundaries; do not add global high-frequency capacity
  without entropy cost.

Provider plan:

- Smoke: one CUDA GPU, 200 epochs, measure seconds per epoch and export size.
- Promotion: only architectures with a better early score slope than PR95 C0
  receive 2000 epoch runs.
- Full: only one replacement family at a time gets full-run spend after C0
  control is alive.

### Track C: SIREN/FINER/WIRE/BACON Coordinate INR

Purpose: use coordinate networks as sparse hard-region residual atoms, not naive
full-frame replacement. The local literature review already identifies full
replacement SIREN as likely byte-dominated under this contest rate budget.

Primary sources:

- SIREN: [arXiv:2006.09661](https://arxiv.org/abs/2006.09661),
  [project](https://www.computationalimaging.org/publications/siren/),
  [GitHub](https://github.com/vsitzmann/siren).
- FINER: [CVPR 2024 paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_FINER_Flexible_Spectral-bias_Tuning_in_Implicit_Neural_Representation_by_Variable-periodic_CVPR_2024_paper.pdf),
  [GitHub](https://github.com/liuzhen0212/FINER).
- WIRE: [arXiv:2301.05187](https://arxiv.org/abs/2301.05187),
  [project](https://vishwa91.github.io/wire/).
- BACON: [arXiv:2112.04645](https://arxiv.org/abs/2112.04645),
  [GitHub](https://github.com/computational-imaging/bacon).

Implementation targets:

- `src/tac/substrates/siren/`
- `experiments/train_substrate_siren.py`
- `src/tac/analysis/segnet_boundary_marginals.py`
- `src/tac/analysis/lapose_foveation_atoms.py`
- `src/tac/packet_compiler/sparse_packet_ir.py`
- `src/tac/substrates/hybrid_renderer_residual/`

Curriculum:

1. Start from a parsed HNeRV or PR101 output, not raw frames.
2. Compute residual maps in scorer-preprocess space and RGB space.
3. Select hard regions by SegNet boundary margin, PoseNet sensitivity, and
   high residual persistence across adjacent pairs.
4. Fit small atoms with local coordinate normalization: `(x, y, t, pair_id)`
   or patch-local `(x, y, t)` plus atom metadata.
5. Quantize atom weights during training, with fixed metadata grammar.
6. Add atoms greedily by marginal score improvement per byte.
7. Export atoms through `sparse_packet_ir`, then run full-frame inflate parity.

Recommended atom families:

- SIREN for smooth periodic geometry.
- FINER for variable-period high-frequency residuals.
- WIRE for wavelet-like localized texture and edges.
- BACON for band-limited multiscale structure and progressive truncation.

Do not train a monolithic full-frame coordinate MLP unless a 200 epoch smoke
shows it beats HNeRV on score-per-byte slope. The likely winning use is a
packet of 10 to 200 tiny atoms over hard regions.

### Track D: Balle/CompressAI/Cheng Hyperprior

Purpose: test whether learned compression priors can beat hand-rolled HNeRV
packets as either a full residual codec or a full replacement. This is a
higher-upside but higher-runtime-risk path.

Primary sources:

- Balle hyperprior: [arXiv:1802.01436](https://arxiv.org/abs/1802.01436).
- CompressAI: [arXiv:2011.03029](https://arxiv.org/abs/2011.03029),
  [docs](https://interdigitalinc.github.io/CompressAI/),
  [GitHub](https://github.com/InterDigitalInc/CompressAI).
- Cheng-style learned image compression is available in CompressAI model
  families and should be treated as a source of architecture patterns, not as a
  runtime dependency to ship blindly.

Implementation targets:

- `src/tac/substrates/balle_renderer/`
- `src/tac/packet_compiler/balle_hyperprior.py`
- `src/tac/packet_compiler/factorized_prior.py`
- `src/tac/packet_compiler/cheng2020.py`
- `experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py`
- `tools/dispatch_t1_balle_endtoend.py`

Curriculum:

1. Train as residual codec over PR95/PR101 output first.
2. Use scorer-aware distortion, not MSE-only.
3. Use differentiable rate estimates during train, then deterministic entropy
   coding at export.
4. Freeze architecture small enough for contest runtime; no dynamic Python
   dependency maze in inflate.
5. After residual success, test full replacement on a smaller frame subset.

Loss:

`L = 100 * seg_dist_proxy + sqrt(10 * pose_proxy + eps) + beta * estimated_bits`

Mutation axes:

- Residual target: RGB residual, scorer-preprocess residual, or feature-space
  residual.
- Hyperprior capacity: tiny, small, medium.
- Entropy model: factorized prior first, scale hyperprior second, context model
  only if runtime budget permits.
- Pair conditioning: none, pair latent, motion-conditioned latent.

Risk controls:

- CompressAI is an OSS reference and training aid, but the contest runtime must
  be a closed, audited local decoder.
- No external pretrained weights unless fully embedded and compliance-reviewed.
- Exported entropy model must have deterministic bitstream and parser tests.

### Track E: Cool-Chic/C3 Overfitted Learned Codecs

Purpose: use overfitted neural codecs where their inductive bias is aligned
with the one-video contest. Start as residual or foveated codecs before full
replacement.

Primary sources:

- Cool-Chic: [project](https://orange-opensource.github.io/Cool-Chic/),
  [GitHub](https://github.com/Orange-OpenSource/Cool-Chic),
  and [arXiv:2307.12706](https://arxiv.org/abs/2307.12706).
- C3: cite as a learned overfitted codec family from
  [arXiv:2312.02753](https://arxiv.org/abs/2312.02753); use repository code
  only after license and runtime review.

Implementation targets:

- `src/tac/substrates/cool_chic/`
- `src/tac/substrates/hybrid_renderer_residual/`
- `src/tac/substrates/self_compress_nn/`
- `src/tac/optimization/entropy_rate_decomposition.py`

Curriculum:

1. Residual over HNeRV/PR101 output on 100 to 300 hardest frame pairs.
2. Train overfitted grids with strong byte penalty and quantization in loop.
3. Export tiny decoder plus latent grids through PacketIR.
4. Compare against sparse INR atoms on the same residual target.
5. Promote only if marginal score per byte beats SIREN/WIRE atom packets.

The likely role is not replacing PR95 immediately. The likely role is a
second-stage residual stack over scorer-sensitive regions where HNeRV smooths
or misallocates capacity.

### Track F: Score-Aware First-Principles Paths

Purpose: exploit contest scorer geometry while staying compliant: no scorer
loads at inflate time, no hidden sidecars, no network, no runtime mismatch.

Subpaths:

- SABOR: score-aware bit and object representation. Treat as an allocator and
  archive grammar planner over real component sensitivities.
- S2SBS: scorer-to-stream bit scheduling. Make it emit typed PacketIR actions,
  not prose.
- PoseNet inverse-craft: search for low-byte perturbations that improve pose
  without collapsing SegNet.
- LA-pose: latent-action pose/motion priors from
  [arXiv:2604.27448](https://arxiv.org/abs/2604.27448). Use as a conceptual
  motion factorization input, not as an unreviewed runtime dependency.
- Telescope foveation: hyperbolic/telescopic foveation from
  [arXiv:2604.06332](https://arxiv.org/abs/2604.06332). Keep distinct from
  LA-pose. Use for where-to-spend-bits scheduling.
- RAFT ego-motion: [arXiv:2003.12039](https://arxiv.org/abs/2003.12039),
  [GitHub](https://github.com/princeton-vl/RAFT). Use at compress/train time
  to derive motion features; do not require RAFT in inflate.
- GEPA/autoresearch: use reflection/search ideas from
  [GEPA](https://arxiv.org/abs/2507.19457) and its
  [optimize-anything API docs](https://gepa-ai.github.io/gepa/api/optimize_anything/optimize_anything/)
  only as proposal generation, not as score authority.

Implementation targets:

- `src/tac/analysis/lapose_motion_atoms.py`
- `src/tac/analysis/lapose_foveation_atoms.py`
- `src/tac/analysis/component_sensitivity.py` or existing sensitivity-map
  surfaces
- `src/tac/optimization/bit_allocator_end_to_end.py`
- `src/tac/optimization/meta_lagrangian_allocator.py`
- `src/tac/optimization/field_equation_planner.py`
- `src/tac/optimization/optimizer_guided_candidate_generation.py`
- `src/tac/optimization/proxy_candidate_contract.py`
- `experiments/derive_poses_from_raft.py`

Curriculum:

1. Build sensitivity maps from exact HNeRV/PR101 residuals.
2. Derive motion and foveation features offline with manifest hashes.
3. Add conditioning to PR95/NeRV-family trainers only if features are embedded
   or deterministically reproducible in the archive.
4. For inverse-craft perturbations, restrict to typed sparse atoms or sidecar
   corrections with no-op controls.
5. Require exact CUDA eval before any rank/kill or promotion decision.

### Track G: PacketIR, Compiler, Sidecar, Arithmetic, Range, ANS

Purpose: make every model and residual path lower into a common typed archive
compiler so byte wins are composable and reproducible.

Primary source for entropy coding:

- ANS: Jarek Duda, asymmetric numeral systems,
  [arXiv:1311.2540](https://arxiv.org/abs/1311.2540). Range coding and
  arithmetic coding should be implemented as deterministic byte-closed passes
  with golden vectors.

Implementation targets:

- `src/tac/packet_compiler/deterministic_compiler.py`
- `src/tac/packet_compiler/pr101_sidecar_grammar.py`
- `src/tac/packet_compiler/pr103_arithmetic_coding.py`
- `src/tac/packet_compiler/sparse_packet_ir.py`
- `src/tac/packet_compiler/golden_vectors.py`
- `src/tac/analysis/hnerv_packet_sections.py`
- `tools/build_hnerv_arch_shrink_driver.py`
- `tools/plan_hnerv_wavelet_residual.py`

Typed passes:

1. Tensor stream normalize: names, shapes, storage order, scales.
2. Stream reorder: by entropy, temporal delta, tensor section, and decoder use.
3. Transform: zigzag, delta, predictive residual, bitplane split, byte-map.
4. Entropy model: brotli baseline, lzma raw, range, arithmetic, ANS.
5. Sidecar atoms: PR100 corrections, PR101 compact sidecar, sparse residual
   atoms, channel postprocess.
6. Header minimization: fixed schema, member name minimization, metadata
   hardcoding only if contest-compliant.
7. Parser golden vectors: every pass must have decode equality tests.

Promotion rule:

PacketIR wins must prove one of two things:

- Same decoded output with fewer bytes, full-frame inflate parity true.
- Intentional decoded-output change with exact component recomputation and
  no-op controls.

## Cost And Time Plan

Cost assumptions are planning placeholders and must be live-verified before
dispatch. Modal public pricing was checked as a current source:
[Modal pricing](https://modal.com/pricing). Prior local ledgers listed example
planning rates around H100 at $3.9492/hr, A100 at $2.0988/hr, A100 80GB at
$2.4984/hr, L40S at $1.9512/hr, and T4 at $0.5904/hr. These rates may drift.

| Phase | Work | Hardware | Time prior | Live verification point |
|---|---|---|---|---|
| 0 | PR95 source import and CPU parser tests | local CPU | same day | archive roundtrip and golden vector |
| 1 | PR95 10 to 50 epoch timing smoke | one CUDA GPU | under 3 GPUh target | seconds/epoch, eval/export time, memory |
| 2 | PR95 full control | H100/A100/L40S | 50 to 200 GPUh | trend vs author source and parsed archive score |
| 3 | PR95 mutation fanout | H100/A100/L40S | 10 to 40 GPUh each short run | mutation beats same-stage control |
| 4 | PacketIR/sidecar searches | CPU plus small GPU | hours to 2 days | byte parity and exact component recompute |
| 5 | Replacement substrate smokes | H100/A100/L40S | 5 to 20 GPUh per smoke | early rate-distortion slope |
| 6 | Residual atom smokes | one CUDA GPU | 5 to 40 GPUh | marginal score per byte |
| 7 | Full promoted replacement/residual campaign | H100/A100/L40S | 50 to 300 GPUh | byte-closed archive and exact CUDA eval |

Before any paid dispatch:

1. Run `tools/claim_lane_dispatch.py claim ...` for the concrete lane id.
2. Record provider, hardware, expected command, git SHA, dirty status, and
   artifact paths.
3. Confirm no active same-lane conflict in `.omx/state/lane_registry.json` and
   active dispatch claims.
4. Run the appropriate import probe and plan-only/dry-run first.

## Exact-Eval And Compliance Gates

All tracks inherit these gates:

- Axis labels must be explicit near every score word: `[contest-CUDA]`,
  `[contest-CPU]`, `[macOS-CPU advisory]`, or proxy.
- PR/public CPU and CUDA are separate axes. Do not convert one to the other.
- Inflater cannot load scorer, training code, network resources, or hidden
  sidecars.
- Every archive must record bytes, SHA-256, member names, member SHA-256s,
  runtime tree SHA, inflated-output aggregate SHA when available, command,
  hardware, sample count, and logs.
- Parsed archive score is the only promotion-relevant training selection.
- Every byte transform must prove non-no-op payload mutation or full decode
  equality.
- Every dispatch must have a claim and terminal claim row.
- Before saying ready, run visible gates:
  - `tools/all_lanes_preflight.py`
  - `tools/operator_briefing.py` if producing queue/report state
  - `scripts/pre_submission_compliance_check.py --contest-final --strict ...`
    before any judge-facing packet

Provider and optimizer proxy rows must flow through
`src/tac/optimization/proxy_candidate_contract.py` or equivalent with
`score_claim=false`, `promotion_eligible=false`, `rank_or_kill_eligible=false`,
and `ready_for_exact_eval_dispatch=false` until byte-closed exact eval exists.

## Stack-Of-Stacks Compiler Integration

Every track must feed the same solver stack, not create orphan signals.

Required six-hook wire-in for any landing:

1. Sensitivity-map contribution: `tac.sensitivity_map.*` or the current
   component-sensitivity surface.
2. Pareto constraint: `tac.pareto_*` or explicit non-binding rationale.
3. Bit-allocator hook: `src/tac/optimization/bit_allocator_end_to_end.py`.
4. Cathedral/autopilot dispatch hook: queue-ready only after custody gates.
5. Continual-learning posterior update: every empirical anchor updates priors.
6. Probe-disambiguator: if two plausible interpretations exist, ship both modes
   and a probe instead of silently choosing.

Canonical lowering order:

`representation -> prediction -> quantization -> hyperprior -> arithmetic -> pack`

Track outputs should lower into PacketIR:

- PR95/HNeRV: decoder tensor streams, latent streams, sidecar streams.
- NeRV replacements: architecture schema plus tensor/latent streams.
- SIREN/WIRE/BACON atoms: sparse atom packets with local coordinate frames.
- Balle/Cool-Chic/C3: entropy-model metadata, quantized latent grids, decoder
  kernels.
- LA-pose/Telescope/RAFT: offline feature manifests and embedded compact
  conditioning fields.
- SABOR/S2SBS/GEPA: allocator actions and candidate proposals, never score
  authority.

## Concrete Next Implementation Slices

These slices are ordered to maximize frontier velocity while preserving
reproducibility.

1. PR95 control port and smoke:
   - Files: `src/tac/substrates/pr95_hnerv_muon/*`,
     `experiments/train_substrate_pr95_hnerv_muon.py`,
     `src/tac/pr95_hnerv.py`.
   - Output: smoke manifest, parsed archive, seconds/epoch.
   - Constraint: preserve author source math exactly first.

2. PR95 export adapter:
   - Files: `src/tac/packet_compiler/pr100_schema_driven_decoder.py`,
     `src/tac/packet_compiler/pr101_sidecar_grammar.py`,
     `src/tac/packet_compiler/pr103_arithmetic_coding.py`,
     `src/tac/packet_compiler/deterministic_compiler.py`.
   - Output: same-state PR95/PR100/PR101/PacketIR archive comparison.

3. Sidecar atom search:
   - Files: `src/tac/packet_compiler/sparse_packet_ir.py`,
     `src/tac/analysis/segnet_boundary_marginals.py`,
     `src/tac/analysis/hnerv_packet_sections.py`.
   - Output: no-op controls, atom manifests, exact component recompute.

4. PR95 mutation fanout:
   - Files: PR95 trainer configs plus `src/tac/optimization/bit_allocator_end_to_end.py`.
   - Mutations: EMA, lambda/sigma schedule, pair sampler, earlier Muon.
   - Output: control-vs-mutation table with identical eval axis.

5. Replacement smokes:
   - Files: `src/tac/substrates/hi_nerv/`, `ff_nerv/`, `ds_nerv/`,
     `experiments/train_ffnerv_as_renderer.py`,
     `experiments/train_lane_12_v2_nerv_as_renderer.py`.
   - Output: 200 epoch timing and rate-distortion slope.

6. Sparse INR residual atoms:
   - Files: `src/tac/substrates/siren/`,
     `experiments/train_substrate_siren.py`,
     `src/tac/substrates/hybrid_renderer_residual/`.
   - Output: atom family comparison on the same hard-region set.

7. Balle/Cool-Chic residual codec:
   - Files: `src/tac/substrates/balle_renderer/`,
     `src/tac/substrates/cool_chic/`,
     `src/tac/packet_compiler/balle_hyperprior.py`.
   - Output: residual compression smoke with deterministic parser.

8. Motion/foveation feature integration:
   - Files: `experiments/derive_poses_from_raft.py`,
     `src/tac/analysis/lapose_motion_atoms.py`,
     `src/tac/analysis/lapose_foveation_atoms.py`.
   - Output: feature manifest and one conditioning ablation in PR95 or FFNeRV.

## Unknown Unknowns And Adversarial Failure Modes

- PR95 source may reproduce public CPU but not local contest CUDA exactly due to
  runtime, scorer, or dependency differences. Keep axes separated.
- PR101 packet gains may depend on CPU/public behavior or postprocess details
  that do not transfer to T4/CUDA exact eval. Treat as indeterminate until
  exact replay.
- QAT may optimize the fake quantizer while real archive parser degrades. Parsed
  archive eval every selection point is mandatory.
- Hard-pair sampling can overfit local error pockets and worsen global score.
  Always evaluate all pairs.
- Sparse residual atoms can become hidden sidecars if grammar is not explicit.
  PacketIR and compliance checks must see every byte.
- Learned hyperprior paths can win rate-distortion but lose runtime budget or
  dependency closure. Runtime contract must be designed before full spend.
- Motion features can leak non-embedded side information. Offline RAFT/LA-pose
  features are legal only if their compressed form or deterministic derivation
  is in the archive/runtime.
- Scorer-equivalence and inverse-craft probes can accidentally exploit
  implementation quirks. They need exact CUDA, no scorer in inflate, and
  adversarial review before status changes.
- Rate-only arithmetic wins are insufficient for sub-0.17 from approximately
  0.19 unless they save tens of KiB. They are still important because they
  compound with representation and residual wins.
- A negative short smoke is not a lane death. It classifies only that
  architecture, budget, loss, and export grammar.

## Bottom Line

The best immediate plan is not "train more HNeRV" and not "abandon HNeRV."
The best plan is a control-first compiler campaign:

1. Reproduce PR95 exactly from the author-provided source.
2. Lower that same state through PR101/PR103/PacketIR packet compilers.
3. Mutate the PR95 curriculum only after the control is alive.
4. Run replacement and residual smokes in parallel so larger representation
   jumps are not starved.
5. Promote only byte-closed archives through exact CUDA gates with explicit
   axis labels.

Sub-0.17 likely requires representation plus packet plus scorer-aware residual
movement. Sub-0.15 likely requires at least one true representation break:
NeRV-family replacement with better byte/component slope, a learned residual
codec that beats sparse atoms, or a first-principles scorer-aware allocation
that changes where bits are spent rather than only how they are packed.
