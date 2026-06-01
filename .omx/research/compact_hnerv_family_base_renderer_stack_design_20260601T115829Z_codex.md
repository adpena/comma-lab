# Compact HNeRV-Family Base Renderer Stack Design

Date: 2026-06-01T11:58:29Z
Author: Codex Subagent A
Scope: design/research memo only; no code, queue, lane-registry, or artifact mutation.

## Authority And Scope

This memo is a design artifact, not a score claim. Every MLX result below is
`[macOS-MLX research-signal]` unless later promoted through byte-closed archive,
receiver proof, same-runtime replay, and exact contest CPU/CUDA evaluation.

The operator explicitly allowed at most one `.omx/research/` memo and requested
no other file edits. Therefore this pass did not register a lane with
`tools/lane_maturity.py`, did not update `.omx/state/*`, did not enqueue
dispatch, and did not stage artifacts. If implementation proceeds, the first
mutation must be lane registration plus dispatch-claim hygiene before training
or exact eval.

## Online Research Provenance

- HNeRV: arXiv 2304.02633, "HNeRV: A Hybrid Neural Representation for Videos",
  primary paper and code links at https://arxiv.org/abs/2304.02633.
  Key design signal: content-adaptive embeddings plus HNeRV blocks improve
  reconstruction quality and convergence relative to index-only NeRV.
- HiNeRV: arXiv 2306.09818, "HiNeRV: Video Compression with Hierarchical
  Encoding-based Neural Representation", https://arxiv.org/abs/2306.09818, with
  official code at https://github.com/hmkx/HiNeRV. Key design signal:
  hierarchical positional encodings, lightweight/depth-wise layers, and
  pruning/quantization matter after the base topology is score-bound.
- RNeRV: arXiv 2506.24127, "How to Design and Train Your Implicit Neural
  Representation for Video Compression", https://arxiv.org/abs/2506.24127, with
  project/code at https://mgwillia.github.io/vinrb/ and
  https://github.com/mgwillia/vinrb. Key design signal: component-level NeRV
  disentangling and equal-time training efficiency; use as architecture-search
  prior, not as immediate archive authority.
- RT/VQ-NeRV: arXiv 2403.12401, now titled "RT-NeRV: Rethinking Hybrid Neural
  Representations for Video via Residual Tokenization",
  https://arxiv.org/abs/2403.12401. Key design signal: discretize shallow and
  inter-frame residual support into compact tokens/codebooks instead of shipping
  continuous support features.
- Tree-NeRV: arXiv 2504.12899, "Tree-NeRV: A Tree-Structured Neural
  Representation for Efficient Non-Uniform Video Encoding",
  https://arxiv.org/abs/2504.12899, with ICCV 2025 OpenAccess final paper at
  https://openaccess.thecvf.com/content/ICCV2025/papers/Zhao_Tree-NeRV_Efficient_Non-Uniform_Sampling_for_Neural_Video_Representation_via_Tree-Structured_ICCV_2025_paper.pdf.
  Key design signal: non-uniform temporal allocation toward high-variation
  regions is more relevant than uniform per-pair latent spending.
- SRNeRV: arXiv 2603.08227, "SRNeRV: A Scale-wise Recursive Framework for
  Neural Video Representation", https://arxiv.org/abs/2603.08227. Key design
  signal: share most channel-mixing parameters across scales and reserve
  scale-specific capacity for spatial mixing.

## Local Provenance And Current Evidence

- `reports/latest.md` currently separates `[contest-CPU Linux x86_64]` and
  `[contest-CUDA T4]` frontiers. Do not compare MLX, CPU, and CUDA by inferred
  equivalence.
- PR95/HNeRV public packet is the strongest compact learned-renderer anchor in
  this family: about 178 KB with strong SegNet/PoseNet behavior and an existing
  archive grammar/runtime pathway.
- The 98,939-byte PACT-NeRV selector-v3 raw-adapter packet is receiver-proof,
  but its local advisory score is about 90 because pixels are bad. It is a
  byte-floor stress test, not a primary score-lowering carrier.
- PACT section-value evidence says `decoder_qw` and `latents_rc` are pixel-value
  carrying; `selectors_rc` is currently effectively dead in the receiver pixel
  path and must be cut, recoded, or made semantic before spending bytes on it.
- HPRC/Z8 are faithful but too byte-heavy for primary carrier status. Treat them
  as residual knowledge and deconstruction signal only.

## Top Stack Verdict

Top EV stack:

1. PR95-faithful HNeRV/Muon base renderer as the primary carrier.
2. Score-aware 8-stage training using real SegNet/PoseNet teacher surfaces where
   available, with PR95 stage curriculum and QAT/export parity as gates.
3. HPRC/PACT section envelope only as the archive discipline layer: section
   mutation, receiver proof, marginal section value, and hard byte ceilings.
4. RT/VQ-NeRV-style residual tokenization as an optional latent/support-section
   replacement after the PR95 base is byte-closed, not as the first carrier.
5. Tree/Hi/SR/RNeRV ideas enter as allocation priors:
   temporal non-uniform capacity, hierarchical scale sharing, and equal-time
   component search. They should not displace PR95/HNeRV until they have the
   same MLX-to-PyTorch bridge, receiver proof, and exact eval path.

Why this stack wins the next action:

- PR95/HNeRV already has contest-shaped runtime and packet grammar. The missing
  work is scorer-faithful training/export closure, not a greenfield receiver.
- PACT raw 98 KB proves low-byte receiver feasibility but not useful pixels.
  Spending the next training cycle on a 100 KB raw PACT base repeats a known
  failure unless decoder/latent fidelity changes.
- RT/VQ tokenization is the right byte-saving idea only after a strong base
  exists. It should compress shallow residual/latent support; it should not be
  allowed to hide a weak base decoder.

## Byte Ceilings And Section Budgets

All ceilings are `archive.zip` byte ceilings. Budgets include an explicit reserve
so archive tooling can fail closed rather than silently crossing a hard limit.

### 100,000-byte stress ceiling

Purpose: reject/validate ultra-compact low-byte ideas before spend.

- `decoder_qw`: 58,000 to 64,000 bytes.
- `latents_rc` or VQ indices/codebook: 24,000 to 31,000 bytes.
- `selectors_rc`: 0 to 512 bytes unless mutation proof shows pixel value.
- `residual_rc`: 0 to 4,000 bytes only if section-value positive.
- manifest/container/header: <= 1,500 bytes.
- reserve: >= 2,000 bytes.

Gate: no exact dispatch unless full-video MLX advisory is already plausible and
receiver proof is complete. Current 98,939-byte PACT raw-adapter result fails
this gate because nonrate distortion dominates.

### 180,000-byte primary ceiling

Purpose: primary PR95-like compact learned base renderer campaign.

- `decoder_qw`: 132,000 to 148,000 bytes.
- `latents_rc`: 22,000 to 30,000 bytes.
- `codebooks_q` or RT/VQ support tokens: 0 to 6,000 bytes, replacing bytes from
  latents/support rather than adding new dead weight.
- `selectors_rc`: <= 512 bytes unless pixel-driving.
- `residual_rc`: 0 to 8,000 bytes, section-value admitted only.
- manifest/container/header: <= 1,500 bytes.
- reserve: >= 4,000 bytes.

Gate: this is the main target. Any candidate above 180,000 bytes must show that
the additional bytes buy enough nonrate improvement to beat rate penalty.

### 285,000-byte escape ceiling

Purpose: allow a byte-heavier score-aware base plus compact residual when 180 KB
cannot preserve SegNet/PoseNet enough.

- `decoder_qw`: 160,000 to 195,000 bytes.
- `latents_rc`: 30,000 to 45,000 bytes.
- `codebooks_q` or RT/VQ support tokens: 4,000 to 16,000 bytes.
- `residual_rc`: 20,000 to 45,000 bytes, admitted only by marginal score value.
- manifest/container/header: <= 2,000 bytes.
- reserve: >= 8,000 bytes.

Gate: 285 KB is an escape valve, not the default. It must demonstrate lower
score than the 180 KB packet on the same axis, with exact rate term included.

## Score-Aware Loss

Contest objective reminder:

`S = 100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/N`

Training objective should be a differentiable proxy with explicit archive-rate
pressure:

```text
L =
  100 * L_seg_teacher_or_stage
  + sqrt(10 * L_pose_teacher + eps)
  + lambda_rgb * L_rgb_roundtrip
  + lambda_entropy * R_hat(theta, latents, tokens)
  + lambda_qat * L_qat_consistency
  + lambda_vq * L_vq_commit_codebook
  + lambda_receiver * L_receiver_parity
```

The primary run should follow the PR95 8-stage curriculum already represented
locally: CE warmup, tau/softplus, smooth, QAT, L7/C1a, lambda sweep, sigma
sweep, and Muon finetune. The local `mlx_score_aware` harness can bind real
SegNet and PoseNet teacher caches for PACT-NeRV VQ/selector experiments via
`--distillation-weight` and `--pose-distillation-weight`. PR95 long-training
currently remains weaker because its long-training path is still RGB-MSE
oriented; bridging it to the same score-aware loss is the highest-value code
landing if implementation is approved.

## Exact Files And Tools To Touch Next

Read/reuse first:

- `src/tac/local_acceleration/pr95_hnerv_mlx.py`
- `src/tac/local_acceleration/pr95_hnerv_mlx_long_training.py`
- `src/tac/local_acceleration/pr95_hnerv_mlx_stage_losses.py`
- `src/tac/local_acceleration/mlx_to_pytorch_export.py`
- `src/tac/substrates/_shared/mlx_score_aware/loss.py`
- `src/tac/substrates/_shared/mlx_score_aware/pr95_faithful_curriculum.py`
- `tools/run_pr95_mlx_long_training.py`
- `tools/export_pr95_mlx_to_pytorch_state_dict.py`
- `tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py`
- `tools/run_pr95_mlx_full_frame_parity_mode_sweep.py`
- `tools/build_pr95_mlx_optimizer_matrix_queue.py`
- `src/tac/substrates/pact_nerv_selector_v3/archive_candidate.py`
- `src/tac/substrates/pact_nerv_selector_v3/section_value.py`
- `tools/profile_pact_nerv_selector_v3_mlx_section_value.py`
- `src/tac/substrates/pact_nerv_vq/mlx_renderer.py`
- `src/tac/substrates/pact_nerv_vq/archive_candidate.py`
- `experiments/train_substrate_pact_nerv_vq_mlx_local.py`
- `tools/export_pact_nerv_vq_mlx_to_pytorch_state_dict.py`
- `tools/gate_mlx_candidate_contest_equivalence_pact_nerv_vq.py`
- `src/tac/substrates/hprc/pr95_adapter.py`
- `src/tac/substrates/hprc/archive.py`
- `src/tac/substrates/hprc/archive_candidate.py`
- `tools/build_hprc_spine_acquisition_queue.py`
- `tools/build_hprc_spine_bounded_runner.py`
- `tools/execute_hprc_spine_receiver_rows.py`
- `tools/gate_hprc_mlx_prefilter_for_local_replay.py`

Likely implementation patch set if approved:

1. Wire PR95 MLX long-training to the canonical `mlx_score_aware` SegNet/PoseNet
   teacher loss, preserving the PR95 8-stage curriculum.
2. Ensure PR95 MLX checkpoints export a full 600-pair latent table, either in
   the `.pt` bundle or the existing `--latents-npy` packaging contract.
3. Add a compact-HNeRV-family queue builder that enumerates 100k/180k/285k
   hard ceilings and emits section budgets into queue rows.
4. Add a PR95/HPRC section-value bridge so PR95 decoder/latent/residual tokens
   can be neutralized and priced with the same marginal score accounting used
   for PACT selector-v3.

## MLX-First Commands

Use SSD output roots first:

```bash
OUT=/Volumes/VertigoDataTier/pact/compact_hnerv_family_base_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
```

PR95 plan and smoke, source-video hashed:

```bash
.venv/bin/python tools/run_pr95_mlx_long_training.py \
  --output-report "$OUT/pr95_hnerv_smoke_report.json" \
  --checkpoint-root "$OUT/pr95_hnerv_smoke_ckpts" \
  --telemetry-path "$OUT/pr95_hnerv_smoke_telemetry.jsonl" \
  --source-video-path upstream/videos/0.mkv \
  --hash-source-video \
  --smoke-mode \
  --smoke-epochs-per-stage 1 \
  --max-frames 8 \
  --execute-smoke
```

PR95 full local MLX run after smoke and storage waterfall:

```bash
.venv/bin/python tools/run_pr95_mlx_long_training.py \
  --output-report "$OUT/pr95_hnerv_full_report.json" \
  --checkpoint-root "$OUT/pr95_hnerv_full_ckpts" \
  --telemetry-path "$OUT/pr95_hnerv_full_telemetry.jsonl" \
  --source-video-path upstream/videos/0.mkv \
  --hash-source-video \
  --execute
```

Packaging trained PR95 state into a byte-closed contest-shaped archive:

```bash
.venv/bin/python tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py \
  --input-pt "$OUT/pr95_hnerv_full.pt" \
  --source-archive-zip experiments/results/lightning_batch/exact_eval_public_pr95_hnerv_muon_t4_fix2_20260504T0848Z/archive.zip \
  --latents-npy "$OUT/pr95_hnerv_full_latents.npy" \
  --output-submission-dir "$OUT/pr95_hnerv_submission" \
  --report-out "$OUT/pr95_hnerv_package_report.json"
```

Full-frame parity sweep for a PR95/HNeRV archive candidate:

```bash
.venv/bin/python tools/run_pr95_mlx_full_frame_parity_mode_sweep.py \
  --archive-zip "$OUT/pr95_hnerv_submission/archive.zip" \
  --output-dir "$OUT/pr95_hnerv_full_frame_parity" \
  --mlx-device gpu \
  --allow-large-output
```

PACT-NeRV VQ 100 KB stress smoke with real scorers bound:

```bash
.venv/bin/python experiments/train_substrate_pact_nerv_vq_mlx_local.py \
  --full \
  --output-dir "$OUT/pact_nerv_vq_32pair_scoreaware" \
  --epochs 2000 \
  --num-pairs 32 \
  --video-path upstream/videos/0.mkv \
  --distillation-weight 1.0 \
  --pose-distillation-weight 1.0 \
  --upstream-dir upstream
```

PACT-NeRV selector-v3 section-value profile for any low-byte archive:

```bash
.venv/bin/python tools/profile_pact_nerv_selector_v3_mlx_section_value.py \
  --archive "$OUT/candidate/archive.zip" \
  --output-dir "$OUT/section_value" \
  --sections decoder_qw latents_rc selectors_rc residual_rc \
  --max-pairs 600 \
  --device gpu \
  --allow-large-tensor-cache \
  --force
```

HPRC receiver-row execution path after an acquisition report is built:

```bash
.venv/bin/python tools/build_hprc_spine_bounded_runner.py \
  --acquisition-report "$OUT/hprc_acquisition_report.json" \
  --output "$OUT/hprc_bounded_runner.json" \
  --force

.venv/bin/python tools/execute_hprc_spine_receiver_rows.py \
  --runner-plan "$OUT/hprc_bounded_runner.json" \
  --output-dir "$OUT/hprc_receiver_rows" \
  --allow-large-output \
  --force
```

## Receiver Proof Path

Required for every candidate before exact eval:

1. Archive packet contains all charged data: decoder weights, latents or VQ
   codebook/indices, residual/support tokens, and manifest. No hidden sidecars.
2. `inflate.sh archive_dir output_dir file_list` consumes archive bytes and
   writes raw RGB output; no scorer-dependent runtime branch.
3. Runtime proof records archive path, bytes, SHA-256, member names, member
   SHA-256s, runtime tree SHA, command, environment/device, output raw bytes,
   and output aggregate hash when retained.
4. Section mutation proof shows which sections are pixel-driving. Metadata-only
   sections cannot justify byte spend.
5. Full-video receiver replay is required before exact CPU/CUDA dispatch.

## Exact Gate Policy

- G0 storage and ownership: use `/Volumes/VertigoDataTier/pact` first, then
  `/Volumes/APDataStore/pact`, with cleanup/certification. Register lane and
  dispatch claim before any long training/eval mutation.
- G1 byte ceiling: fail closed above the chosen 100k, 180k, or 285k ceiling.
- G2 no hidden sidecars: fail if archive cannot reproduce from charged bytes.
- G3 MLX advisory: use only to triage. It cannot promote, rank, or kill across
  CPU/CUDA axes by itself.
- G4 receiver proof: fail without full-video receiver/runtime-consumption proof.
- G5 parity/same-runtime replay: PR95 bridge needs full-frame parity; native
  HPRC/PACT receivers need same-runtime replay and section consumption proof.
- G6 exact auth axis: only exact `[contest-CPU]` or `[contest-CUDA]` artifacts
  may support frontier/medal/submission language, and those axes stay separate.

## Blockers

- PR95 MLX long-training is not yet fully scorer-faithful; current long-training
  infrastructure is still RGB-MSE/plumbing-heavy relative to the PR95 scorer
  curriculum.
- Prior PR95 MLX packaging attempts exposed the full-latent-table blocker:
  packaging trained latents requires 600 pair latents, not a 4-pair smoke table.
- PR95 MLX optimized full-frame parity against public PR95 is not established;
  direct PyTorch parity exists, but MLX optimized output previously showed small
  full-frame byte drift.
- Current 98,939-byte PACT raw adapter is receiver-proof but score-poor, so it
  blocks promotion and acts only as a low-byte negative control.
- VQ/RT-NeRV has the right byte economics but lacks local full-video, byte-closed
  exact-gated evidence in this repo.
- HiNeRV, Tree-NeRV, SRNeRV, and RNeRV are useful priors but lack current
  MLX-first contest archive/runtime proof here.
- This memo intentionally did not mutate lane/queue state due operator scope.
  Implementation must begin with lane registration and claim hygiene.

## Recommended Next Artifact

Land a narrow PR95 score-aware MLX training bridge that reuses
`src/tac/substrates/_shared/mlx_score_aware/loss.py` inside the PR95
long-training path, exports a full 600-pair latent table, packages under the
180,000-byte ceiling, and runs receiver/full-frame parity before any exact
dispatch.
