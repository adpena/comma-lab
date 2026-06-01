# Codex PR95/HNeRV Faithful Stack Design - Subagent A

- generated_utc: 2026-06-01T12:27:12Z
- agent_role: PR95/HNeRV faithful-implementation design subagent A
- cwd: `/Users/adpena/Projects/pact`
- status: design memo only
- score_claim: false
- promotion_eligible: false
- rank_or_kill_eligible: false
- ready_for_exact_eval_dispatch: false
- online_research_used: false

This memo is the implementation-ready design for the immediate PR95/HNeRV EV
lane: a faithful HNeRV core trained against the contest scorer surface and
packed under the PR95 single-member byte grammar. It is not a result ledger and
does not claim any score movement. The current user instruction limited this
subagent to exactly one memo, so I did not register the lane, checkpoint state,
stage files, run training, stage git changes, or mutate any partner work. The
first implementation turn must perform lane registration and checkpointing
before spending compute or writing bulky artifacts.

## Local Evidence Read

- `AGENTS.md` and `CLAUDE.md`: binding non-negotiables, especially main-only
  truth, fail-closed score authority, SSD-first artifact spill, PR95/HNeRV
  parity discipline, and local-MLX false-authority rules.
- Public PR95 source: `experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/`.
- Recovered/source archive paths: `experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/archive.zip`, `experiments/results/lightning_batch/exact_eval_public_pr95_hnerv_muon_t4_fix2_20260504T0848Z/archive.zip`, `experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/archive.zip`.
- Current PR95 helpers: `src/tac/pr95_hnerv.py`, `src/tac/local_acceleration/pr95_hnerv_mlx_long_training.py`, `src/tac/local_acceleration/pr95_hnerv_mlx_stage_losses.py`, `src/tac/substrates/_shared/mlx_score_aware/`, `tools/run_pr95_mlx_long_training.py`, `tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py`, `tools/prove_pr95_public_archive_runtime_consumption.py`, `tools/prove_pr95_public_archive_full_frame_parity.py`.
- Recent PR95/HNeRV memos: `.omx/research/codex_findings_pr95_compact_carrier_controls_and_latent_guard_20260601T122243Z_codex.md`, `.omx/research/compact_hnerv_family_base_renderer_stack_design_20260601T115829Z_codex.md`, `.omx/research/subagent_b_latent_codebook_implicit_stack_design_20260601T120216Z_codex.md`, `.omx/research/codex_findings_hprc_pr95_archive_runnable_score_movement_20260601T005855Z_codex.md`.
- Current frontier pointer: `.omx/state/canonical_frontier_pointer.json` last refreshed 2026-05-31T19:17:24Z. Current local bests are `[contest-CPU]` score `0.19198533626623068`, bytes `178493`, archive sha `b7106c9bdbb8a2df18af622636ca79a11fa0c771a09c75219474d980b8997c8c`; and `[contest-CUDA]` score `0.20533002902019143`, bytes `186876`, archive sha `9cb989cef519ed1771f6c9dc18c988ee93d01a2925da1913d63f9015d6247cf4`.
- Current worktree is dirty and shared. PR95-adjacent dirty surfaces include many deleted historical PR95/HNeRV artifacts and modified `.omx/research/pr95_mlx_runtime_consumption_queue_20260528T131513Z/*.json`; I saw no current unstaged code diffs in the main PR95 helper files listed above. Implementation must re-check status before editing.
- Active dispatch claims currently do not show a fresh PR95/HNeRV faithful lane conflict, but there are unrelated stale/active Modal rows. Implementation must re-read `.omx/state/active_lane_dispatch_claims.md` immediately before any training/eval dispatch.

## Decision

The immediate EV lane is not a new paradigm and not the latent-codebook stack
from subagent B. It is a faithful PR95/HNeRV scorer-aware continuation under
the PR95 byte grammar, starting from the public PR95 archive/runtime and moving
only after each gate is byte-closed:

1. Decode the public PR95 packet into f32 decoder weights, f32 latents, and
   meta.
2. Re-establish source runtime and archive parser identity locally.
3. Run a short scorer-faithful Stage 8 continuation smoke in PyTorch on the
   real source video/scorer path.
4. If timing and gradients are healthy, run the full Stage 8 continuation from
   the public PR95 checkpoint before any fresh 29,650-epoch run.
5. Package the best EMA checkpoint using trained latents, not source latents.
6. Gate with runtime consumption, full-frame parity, same-runtime exact eval,
   and paired `[contest-CPU]`/`[contest-CUDA]` auth eval before any score,
   rank, promotion, or kill claim.

Rationale: current MLX long-training can run RGB/YUV6 advisory controls, but
the fresh 2026-06-01 memo correctly classifies that as a control basin, not a
faithful scorer lane. The PR95 source trainer already contains the missing
scorer-aware training loop. The fastest frontier-relevant path is therefore to
port that faithful loop into controlled pact modules and run Stage 8 from the
public packet first.

## Exact Archive Byte Grammar

Contest packet surface:

- `archive.zip` is a single-member ZIP with one stored member named `0.bin`.
- Current public PR95 archive: zip bytes `178417`, `0.bin` bytes `178309`,
  archive sha `e976acd5fe565c94fb9a8c62e5200c949919f76150e84599f268d6a58588440a`.
- Public section sizes inside `0.bin`: `meta_brotli=80`, `decoder_brotli=162349`,
  `latents_brotli=15868`, three 4-byte length prefixes, trailing bytes `0`.

Top-level `0.bin` grammar, little-endian:

```text
u32 meta_brotli_len
u8  meta_brotli[meta_brotli_len]
u32 decoder_brotli_len
u8  decoder_brotli[decoder_brotli_len]
u32 latents_brotli_len
u8  latents_brotli[latents_brotli_len]
EOF required, no trailing bytes
```

`meta_brotli` is brotli-q11 JSON with at least:

```json
{"n_pairs":600,"latent_dim":28,"base_channels":36,"eval_size":[384,512]}
```

`decoder_brotli` raw payload:

```text
u32 tensor_count
repeat tensor_count:
  u32 name_len
  u8  name_utf8[name_len]
  u32 rank
  repeat rank: u32 dim
  f32 scale
  u32 q_size
  u8  zigzag_int8_quantized_values[q_size]
```

Each tensor is per-tensor symmetric INT8: `scale=max(abs(tensor))/127`,
`q=round(tensor/scale).clip(-127,127)`, then int8 zigzag maps non-negative
values to even bytes and negative values to odd bytes.

`latents_brotli` raw payload:

```text
u32 n_pairs
u32 latent_dim
f16 mins[latent_dim]
f16 scales[latent_dim]
u8  lo_delta_zigzag[n_pairs * latent_dim]
u8  hi_delta_zigzag[n_pairs * latent_dim]
```

Latents are per-dimension min/max scaled to uint8 `[0,254]`, temporal first
differenced, zigzagged to uint16, then split into low/high byte streams.

Immediate implementation must keep this 3-section grammar. Do not add a `wrp`
section, second ZIP member, sidecar file, hidden runtime dependency, or parser
fallback for the first faithful candidate. If later 216k/285k exploration needs
PR101-style wrapper data, that is a separate v2 parser/runtime proof, not this
lane's initial score-bearing surface.

## Source Decode And Resolution Path

The faithful training path is the recovered PR95 source path, not current MLX
RGB-MSE plumbing:

1. Source video: pinned contest video from the challenge root, resolved by
   `get_default_video_path()` in PR95 source or by explicit `upstream/videos/0.mkv`
   in pact wrappers. Record source path, bytes, sha256, decode library versions,
   and frame count before training.
2. Decode: PyAV frames through challenge `frame_utils.yuv420_to_rgb`, grouped
   into consecutive non-overlapping pairs. The binding shape is 600 pairs,
   1200 frames.
3. Renderer output: HNeRV decoder emits `(B,2,3,384,512)` floats in `[0,255]`.
4. Training roundtrip: native output is bicubic-upsampled to camera
   `(874,1164)`, bilinear-downsampled back to `(384,512)`, clamped, rounded via
   a straight-through estimator, then fed to `DistortionNet.preprocess_input`.
   This is the recovered scorer-domain training path from PR95 `stages/common.py`.
5. Evaluation/runtime: packaged inflate emits full camera raw RGB at
   `(N,874,1164,3)` by bicubic-upsampling native frames, clamping/rounding to
   uint8, and writing flat raw bytes. Exact gates compare this output, not only
   tensors, latents, or parser state.
6. Pose gradient patch: the PR95 source monkey-patches `frame_utils.rgb_to_yuv6`
   and `modules.rgb_to_yuv6` with a differentiable BT.601/YUV6 implementation.
   This is mandatory; the challenge helper's `no_grad`/in-place clamp severs
   pose gradients and leaves pose stuck.

## Scorer-Aware Loss

The faithful objective is the PR95 staged scorer objective:

```text
loss = 100 * seg_loss(stage) + sqrt(10 * pose_mse_first6 + 1e-12)
       + cat_lambda(stage) * C1a_entropy(decoder, sigma=stage_sigma)
```

Stage loss families:

- Stage 1: cross entropy against frozen SegNet hard labels.
- Stage 2: target-vs-best-other margin `tau_softplus` with `tau=0.3`.
- Stage 3: smooth-disagreement sigmoid on negative margin.
- Stage 4: Stage 3 plus INT8 fake-quant QAT in the decoder forward.
- Stage 5: L7 hard-pixel weighted softplus plus QAT and C1a entropy,
  `cat_lambda=0.01`, `cat_sigma=0.2`.
- Stage 6: same as Stage 5 but `cat_lambda=0.02`.
- Stage 7: same as Stage 6 but `cat_sigma=0.1`.
- Stage 8: same as Stage 7, optimizer switches hidden conv/linear tensors to
  Muon and leaves latents, biases, stem/RGB heads, and remaining tensors on
  AdamW.

Checkpoint selection must be archive-score-shaped, not loss-only:

```text
score_proxy = 100 * seg_distortion
              + sqrt(10 * pose_distortion + 1e-12)
              + 25 * archive_bytes / 37545489
```

The proxy is useful for selecting local checkpoints only. It is not a contest
score unless it came from the exact auth-eval path with archive/runtime custody.

## Training Curriculum

### Fastest EV: Stage 8 From Public PR95

Use the public PR95 packet as the starting point:

- Parse public `archive.zip` with `tac.pr95_hnerv.parse_top_blob` /
  `tac.local_acceleration.pr95_hnerv_mlx.parse_pr95_public_archive_zip`.
- Recover decoder f32 weights and f32 latents from `0.bin`.
- Verify re-encoding under canonical PR95 codec preserves parser semantics and
  produces a byte-accounted section manifest.
- Run Stage 8 continuation for 10-25 epochs on a tiny pair subset as timing and
  gradient smoke.
- Run Stage 8 continuation for 250, 1000, then 5000 epochs, with eval every 25
  epochs, if the smoke has finite gradients and no section-size explosion.
- Keep only artifacts on SSD and every best checkpoint must carry source video
  sha256, public archive sha, runtime tree sha, command argv/env, checkpoint
  hashes, decoder/latent hashes, and false-authority fields.

Expected first movement is small but high-EV: reduce decoder section entropy
through QAT/C1a/Muon without perturbing SegNet/PoseNet enough to lose the rate
win. Target the public packet's `decoder_brotli=162349` first; the 178k ceiling
needs roughly 500 bytes if targeting the current 178417 public zip, or roughly
4.5 KB if forcing the stricter 178000 budget.

### Full Reproduction / Beat Public PR95

If Stage 8 continuation is healthy, queue the full source-faithful curriculum:

| stage | epochs | loss | optimizer | key knobs |
|---|---:|---|---|---|
| 1 | 3000 | CE SegNet + pose | AdamW | batch 8, lr 1e-3, random decoder/latents |
| 2 | 5650 | tau-softplus SegNet + pose | AdamW | lr 1e-3 |
| 3 | 1500 | smooth disagreement + pose | AdamW | lr 1e-4 |
| 4 | 500 | smooth disagreement + pose | AdamW | QAT on |
| 5 | 9000 | L7 + pose + C1a | AdamW | lambda 0.01, sigma 0.2, QAT |
| 6 | 2000 | L7 + pose + C1a | AdamW | lambda 0.02, sigma 0.2, QAT |
| 7 | 3000 | L7 + pose + C1a | AdamW | lambda 0.02, sigma 0.1, QAT |
| 8 | 5000 | L7 + pose + C1a | Muon + AdamW | Muon lr 2e-4, AdamW lr 1e-5 |

The public README says this is about 50 hours on one GPU from random init.
That is too expensive to start blindly while the Stage 8 continuation route is
available. Full-run authorization should follow the Stage 8 timing smoke and a
measured seconds-per-epoch estimate.

### Variant Ladder Under Same Grammar

All variants keep the same 3-section top-level grammar:

- `c36_ld28`: exact PR95 architecture, immediate Stage 8 continuation.
- `c32_ld28`: decoder shrink, same latent table, full or partial curriculum.
- `c28_ld24`: pressure path for sub-178k and 100k stress tests.
- `c24_ld20` or `c24_ld16`: 100k research stress only unless it holds
  distortion unexpectedly well.
- `c40_ld28` or `c44_ld32`: 216k/285k upper-fidelity diagnostic, useful to
  test whether distortion can approach the rate floor enough to justify bytes.

Do not compare these variants by local MLX loss. Compare only by byte-closed
same-runtime eval artifacts and then exact auth axes.

## MLX / Long-Training Plan

MLX remains valuable, but the current PR95 MLX runner is not yet faithful. Its
current `training_loss_surface` choices are `rgb_mse` and `rgb_yuv6_mse`; both
are marked advisory. The `pr95_hnerv_mlx_stage_losses.py` primitives implement
stage loss math, and `_shared/mlx_score_aware` has a scorer-teacher harness,
but the real PR95 scorer-network forward is not wired through the MLX
long-training pipeline.

Therefore:

1. First scorer-faithful implementation should be PyTorch, composed from
   recovered PR95 source, because it can actually run `DistortionNet`, SegNet,
   PoseNet, differentiable YUV6, QAT, EMA, Muon, and archive eval.
2. MLX should run timing/control and teacher-surrogate long runs only after
   cache/provenance is explicit:
   - precompute SegNet hard labels, PoseNet first-six targets, optional SegNet
     logits/boundary weights, and source RGB/YUV6 teacher targets on SSD;
   - bind those caches through `src/tac/substrates/_shared/mlx_score_aware/targets.py`
     and `bundle.py`;
   - expose a new loss surface such as `pr95_stage_seg_pose_teacher` in
     `src/tac/local_acceleration/pr95_hnerv_mlx_long_training.py`;
   - carry `evidence_grade="[macOS-MLX research-signal]"`, `score_claim=false`,
     `promotion_eligible=false`, and exact-readiness blockers in every output.
3. MLX promotion path remains: MLX checkpoint -> PyTorch state dict + trained
   full `(600,latent_dim)` latents -> package -> runtime consumption -> full
   frame parity -> paired exact auth eval. MLX never becomes the authority by
   itself.

## QAT And Packing Plan

Initial packer is pure PR95:

- decoder: per-tensor symmetric INT8, zigzag, one brotli-q11 stream;
- latents: per-dim uint8 min/max, temporal delta, lo/hi split, brotli-q11;
- meta: brotli-q11 JSON;
- ZIP: single stored `0.bin`.

Score-lowering pack improvements, in order:

1. Make QAT/C1a train directly against archive bytes by running `build_archive`
   at eval cadence and selecting checkpoints by `score_proxy`, not just loss.
2. Keep per-tensor scale fp32 in the PR95-compatible decoder raw layout for the
   first candidate; only switch to fp16 scales after parser/runtime proof shows
   equal or acceptable full-frame effect.
3. Reuse the existing PR95 latent guard: trained decoder with stale source
   latents must fail closed unless `--allow-source-archive-latents` is explicit
   and non-promotional.
4. Add byte-map experiments only after the faithful source candidate is closed:
   `zig`, `negzig`, `twos`, `off`, Conv4 storage perms, split brotli streams,
   raw LZMA latent coding, and canonical Huffman/ANS internal recodes. These
   should remain internal to the decoder/latent sections or land as a separate
   parser version with full receiver proof.
5. Preserve a no-op detector: any repack that changes archive bytes but not
   decoded full-frame bytes must be classified as rate-only; any repack that
   changes parser semantics or full-frame bytes needs same-runtime scorer proof
   before method conclusions.

## Section Budgets

Rate components use the current challenge denominator `37545489` bytes.

| archive ceiling | rate component | meta+zip+prefix target | decoder section | latent section | sidecar/wrp | intended use |
|---:|---:|---:|---:|---:|---:|---|
| 100000 | 0.066585895 | <= 256 | 58k-68k | 18k-26k | 0 | shrink/distill stress, not immediate promotion |
| 178000 | 0.118522894 | <= 256 | 154k-160k | 14k-17k | 0 | primary frontier-lane budget; public PR95 needs small rate drop |
| 216000 | 0.143825534 | <= 512 | 176k-188k | 22k-30k | 0-6k internal only | fidelity diagnostic and CUDA-axis escape candidate |
| 285000 | 0.189769802 | <= 768 | 210k-235k | 32k-46k | 0-10k internal only | upper-bound distortion probe; CPU frontier only if distortion nearly vanishes |

Public PR95 baseline for calibration:

| field | bytes |
|---|---:|
| archive.zip | 178417 |
| `0.bin` | 178309 |
| meta brotli | 80 |
| decoder brotli | 162349 |
| latents brotli | 15868 |
| length prefixes | 12 |

Budget implications:

- The 178k candidate should not start by widening the model. It should recover
  500-5000 bytes from decoder entropy or latent coding while preserving source
  distortion.
- The 100k candidate requires architecture pressure, likely `base_channels<=28`
  and `latent_dim<=24`, plus scorer-aware distillation from the public PR95
  output. It is not the first exact-eval spend.
- The 216k candidate can afford better distortion but must beat a larger rate
  term. It is useful for CUDA-axis exploration if the CPU/CUDA gap persists.
- The 285k candidate has rate alone near the CPU frontier. Treat it as a
  diagnostic upper bound or a source for distillation, not a likely CPU
  submission.

## Concrete Code Changes

Implement as a narrow patch set after re-checking worktree ownership.

1. Add `src/tac/local_acceleration/pr95_hnerv_faithful_training.py`.
   - Wrap recovered PR95 PyTorch `StageConfig`, target precompute, differentiable
     YUV6 patch, QAT, EMA, Muon partition, archive eval, and byte-accounted
     checkpoint reports.
   - Input modes: `from_public_archive_stage8`, `from_scratch_8stage`.
   - Output: f32 decoder `.pt`, trained latents `.npy` or `.pt` key, per-stage
     reports, best archive section manifests, false-authority JSON.

2. Add `tools/run_pr95_faithful_stage8_training.py`.
   - Thin CLI over the new module.
   - Plan-only default; `--execute-smoke` and `--execute` required for training.
   - Requires lane id, output root, source archive, source submission root,
     source video path, device, epochs, max pairs, eval cadence, checkpoint
     cadence, and storage preflight result path.

3. Extend `src/tac/local_acceleration/pr95_hnerv_mlx_long_training.py` only
   after the PyTorch faithful path lands.
   - Add scorer-teacher cache ingestion and a new `training_loss_surface`.
   - Keep MLX reports non-promotional and exact-readiness-blocked.

4. Extend `tools/run_pr95_mlx_long_training.py`.
   - Expose scorer-teacher cache paths only after the module above exists.
   - Keep existing `rgb_yuv6_mse` control behavior unchanged.

5. Harden `tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py`.
   - It already refuses stale source latents. Add the same trained-latent
     requirement for faithful PyTorch bundles, including shape `(600,28)` or the
     variant's exact meta shape.
   - Emit section byte budgets, public-source archive identity, and source-video
     hash in the package report.

6. Reuse `src/tac/pr95_hnerv.py`.
   - It is the canonical parser for single-member ZIP and three-section top
     blob. Add tests for public section sizes and no trailing bytes if missing.

7. Reuse proof tools.
   - `tools/prove_pr95_public_archive_runtime_consumption.py`
   - `tools/prove_pr95_public_archive_full_frame_parity.py`
   - Add a candidate-vs-candidate same-runtime eval wrapper if no current helper
     ties the package report, runtime tree sha, and exact auth handoff together.

8. Tests to add or extend.
   - `src/tac/tests/test_pr95_hnerv_faithful_training.py`
   - `src/tac/tests/test_pr95_mlx_long_training_infrastructure.py`
   - `src/tac/tests/test_pr95_mlx_pytorch_archive_package.py`
   - `_shared/mlx_score_aware` tests only after MLX teacher loss is wired.

## First Commands

These are not run by this memo-only subagent. They are the launchable sequence
for the implementing agent after re-reading current state and registering the
lane.

Pre-claim and SSD preflight:

```bash
LANE=lane_pr95_hnerv_faithful_stage8_20260601
OUT=/Volumes/VertigoDataTier/pact/pr95_hnerv_faithful_stage8_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
df -h /Volumes/VertigoDataTier/pact /Volumes/APDataStore/pact .
.venv/bin/python tools/lane_maturity.py add-lane "$LANE" \
  --name "PR95 HNeRV faithful Stage-8 scorer-aware continuation" \
  --phase 1 \
  --notes "L0 registration before local/GPU timing smoke; score_claim=false"
.venv/bin/python tools/claim_lane_dispatch.py claim --dry-run \
  --lane-id "$LANE" \
  --platform local \
  --instance-job-id "$LANE:timing-smoke" \
  --agent codex:pr95_hnerv_subagentA_successor \
  --status active_local_timing_smoke \
  --notes "Dry-run claim check before PR95 faithful timing smoke; no score claim"
```

Current executable advisory control smoke, before faithful code exists:

```bash
.venv/bin/python tools/run_pr95_mlx_long_training.py \
  --output-report "$OUT/current_control_rgb_yuv6_smoke_report.json" \
  --checkpoint-root "$OUT/current_control_rgb_yuv6_smoke_ckpts" \
  --telemetry-path "$OUT/current_control_rgb_yuv6_smoke_telemetry.jsonl" \
  --source-video-path upstream/videos/0.mkv \
  --training-loss-surface rgb_yuv6_mse \
  --smoke-mode \
  --smoke-epochs-per-stage 2 \
  --checkpoint-every-epochs 2 \
  --max-frames 32 \
  --hash-source-video \
  --execute-smoke \
  --lane-id "$LANE" \
  --operator-run-label pr95_faithful_pre_landing_control
```

First scorer-faithful timing smoke after adding `tools/run_pr95_faithful_stage8_training.py`:

```bash
.venv/bin/python tools/run_pr95_faithful_stage8_training.py \
  --source-archive-zip experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/archive.zip \
  --source-submission-root experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon \
  --source-video-path upstream/videos/0.mkv \
  --output-root "$OUT/stage8_faithful_smoke" \
  --lane-id "$LANE" \
  --device cuda \
  --epochs 10 \
  --max-pairs 16 \
  --eval-every 5 \
  --checkpoint-every 5 \
  --cat-lambda 0.02 \
  --cat-sigma 0.1 \
  --use-qat \
  --use-muon \
  --execute-smoke
```

First full Stage 8 continuation after the timing smoke passes:

```bash
.venv/bin/python tools/run_pr95_faithful_stage8_training.py \
  --source-archive-zip experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/archive.zip \
  --source-submission-root experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon \
  --source-video-path upstream/videos/0.mkv \
  --output-root "$OUT/stage8_faithful_full" \
  --lane-id "$LANE" \
  --device cuda \
  --epochs 5000 \
  --max-pairs 600 \
  --eval-every 25 \
  --checkpoint-every 25 \
  --cat-lambda 0.02 \
  --cat-sigma 0.1 \
  --use-qat \
  --use-muon \
  --archive-score-selection \
  --execute
```

Packaging the first best checkpoint:

```bash
.venv/bin/python tools/package_pr95_mlx_pytorch_state_dict_to_contest_archive.py \
  --input-pt "$OUT/stage8_faithful_full/best_bundle.pt" \
  --source-archive-zip experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/archive.zip \
  --latents-from-pt \
  --output-submission-dir "$OUT/stage8_faithful_submission" \
  --report-out "$OUT/stage8_faithful_package_report.json"
```

Receiver proof and full-frame output proof:

```bash
.venv/bin/python tools/prove_pr95_public_archive_runtime_consumption.py \
  --archive-zip "$OUT/stage8_faithful_submission/archive.zip" \
  --inflate-sh "$OUT/stage8_faithful_submission/inflate.sh" \
  --output-json "$OUT/stage8_faithful_runtime_consumption.json" \
  --allow-large-output

.venv/bin/python tools/prove_pr95_public_archive_full_frame_parity.py \
  --archive-zip "$OUT/stage8_faithful_submission/archive.zip" \
  --inflate-sh "$OUT/stage8_faithful_submission/inflate.sh" \
  --output-json "$OUT/stage8_faithful_full_frame_parity.json" \
  --mlx-device gpu \
  --allow-large-output
```

The actual CLI must be tested with `--help` before launch; these commands are
the intended implementation contract, not proof that the new tool exists yet.

## Exact Gate Blockers

The candidate remains blocked from score authority until all of these are true:

- Lane registered and active dispatch claim exists before training/eval spend.
- Storage waterfall passes and all bulky outputs land under `/Volumes/VertigoDataTier/pact` or `/Volumes/APDataStore/pact`, not local disk, unless the operator explicitly opts in.
- Source video sha256, public archive sha256, runtime tree sha256, command argv,
  env, seed, and checkpoint hashes are recorded.
- Public PR95 parser roundtrip is proven: single stored `0.bin`, three sections,
  no trailing bytes, section sizes recorded.
- Trained latents are packaged from the checkpoint (`--latents-from-pt` or
  `--latents-npy`), not silently inherited from the public source archive.
- Runtime consumption proof emits the expected `3662409600` raw bytes for 1200
  RGB frames.
- Full-frame inflate output parity is established for any claimed exporter or
  same-runtime comparison. Parser or latent parity is not enough.
- Same-runtime local scorer recomputation exists for source packet and candidate
  packet under the same runtime tree.
- Paired `[contest-CPU]` and `[contest-CUDA]` auth eval exists before score,
  rank, promotion, or kill authority.
- CPU/CUDA/local/MLX axes remain separate in reports. A `[macOS-MLX research-signal]`
  row can select follow-up candidates; it cannot promote or retire them.
- Cleanup is certify-or-block: any moved/deleted bulky artifact needs original
  path, bytes, sha256/tree hash, command/config/env, source/runtime hashes,
  cold-store destination, false-authority flags, and rebuildability reason.

## Adversarial Risks

- RGB/YUV6 control trap: current MLX controls can look healthy while missing
  SegNet/PoseNet gradients. Do not authorize exact eval from these alone.
- Source-latent mismatch: pairing a trained decoder with stale public latents
  can produce plausible parser output and false authority. The packager guard is
  correct and must stay fail-closed.
- MLX/PyTorch drift: recent Kahan global full-frame probe changed only 64 bytes
  versus the optimized baseline drift. Global Kahan is not the missing parity
  mechanism.
- Eval-roundtrip mismatch: training loss must include the recovered up/down
  roundtrip and straight-through rounding. A direct 384x512 pixel loss is not
  the PR95 objective.
- Differentiable YUV6 missing: if `rgb_to_yuv6` stays under `no_grad` or
  in-place clamp, pose gradients do not reach the decoder.
- Byte overfitting: C1a/QAT may shrink decoder bytes while damaging SegNet or
  PoseNet. Checkpoint selection must include archive-score-shaped eval.
- 285k mirage: rate alone is almost the CPU frontier. Large packets are useful
  diagnostics, not obvious submissions.
- CPU/CUDA gap: public HNeRV behavior can differ by inflate/scorer substrate.
  Never infer `[contest-CUDA]` from `[contest-CPU]` or vice versa.
- Dirty shared tree: many historical PR95/HNeRV artifacts are deleted or
  modified in the current checkout. Implementation must stage only its own
  validated slice and ignore unrelated work.
- Runtime dependency closure: source PR95 imports challenge modules, PyAV,
  torch, brotli, and scorer weights. Missing runtime dependencies are
  infrastructure failures, not model negatives.
- New parser temptation: internal recodes, split streams, or wrappers can save
  bytes, but every parser change becomes a receiver-proof problem. The first
  candidate should stay under the exact source grammar.

## Immediate Next Artifact

The next concrete artifact should be a narrow code landing for the PyTorch
faithful Stage 8 runner:

- add `src/tac/local_acceleration/pr95_hnerv_faithful_training.py`;
- add `tools/run_pr95_faithful_stage8_training.py`;
- add parser/section-size and trained-latent packaging tests;
- run a 10-epoch, 16-pair scorer-faithful timing smoke on SSD;
- write a false-authority report with seconds/epoch and exact blockers.

Only after that timing smoke should the lane spend on 250/1000/5000 epoch Stage
8 continuations or a fresh 29,650-epoch full curriculum.
