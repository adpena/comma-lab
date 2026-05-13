# PR95 Curriculum Recovery - Worker F1

date: 2026-05-13
worker: F1 PR95 curriculum recovery
repo: `/Users/adpena/Projects/pact`
branch: `main`
research_only: true for this forensic memo/helper
score_claim: false
promotion_eligible: false
dispatch_attempted: false
gpu_spend_attempted: false
verdict: READY-TO-WIRE, not ready-to-dispatch

## Scope And Guardrails

This pass recovers the concrete PR95 `hnerv_muon` training discipline and turns
it into a sane HNeRV reproduction plan. It does not promote a score, spend GPU,
launch Modal/Vast/Lightning, or modify provider dispatch code.

Preflight performed:

- Read `CLAUDE.md`, `AGENTS.md`, and `PROGRAM.md` enough to honor
  source-of-truth, exact-eval, public-frontier, no-proxy-authority, and dispatch
  custody rules.
- Checked `.omx/state/lane_registry.json` and
  `.omx/state/active_lane_dispatch_claims.md`; no PR95/F1 paid dispatch was
  claimed or launched.
- Checked `.omx/research/*_directive_*` files dated within the last 24 hours;
  none were present.
- Did not run `gh pr checkout` or mutate a public PR tree into the shared
  worktree.
- Internet sources used only for public PR/rate context:
  <https://github.com/commaai/comma_video_compression_challenge/pull/95> and
  <https://modal.com/pricing>. Modal rate assumptions below were read on
  2026-05-13 and must be verified live before dispatch.

## Evidence Inventory

Primary recovered PR95 source:

- `experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/`
- mirror:
  `experiments/results/public_pr_archive_release_view/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/`

Primary PR95 metadata:

- `experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/pr_metadata.json`
  records PR 95, title `hnerv_muon submission (0.20)`, author
  `AaronLeslie138`, head SHA
  `9bdce26f2a4f996828c4e3fa2b87c454a0e8fcc9`, merged
  `2026-05-04T20:06:33Z`, leaderboard name `hnerv_muon`, leaderboard score
  `0.199`.
- `experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/pr_body.md`
  records the archive upload link, reported archive size `178417`, pose
  `0.00003494`, seg `0.00061212`, final score `0.20`, and explicitly describes
  the eight-stage curriculum ending with Muon.
- GitHub public PR page:
  <https://github.com/commaai/comma_video_compression_challenge/pull/95>.

Existing repo forensic surfaces:

- `.omx/research/pr95_8stage_curriculum_forensic_20260513.md` already
  identifies the eight-stage schedule but used the earlier small-budget framing;
  this memo supersedes that budget conclusion.
- `.omx/research/pr95_hnerv_muon_trainer_parity_profile_20260510.json` records
  `total_release_view_epochs=29650`, schedule digest
  `55b95573b34c812b73184a3e0e80734c629bc691d9e60dad0590944e60d34d16`, and
  fail-closed dispatch blockers.
- `experiments/profile_pr95_hnerv_muon_intake.py` is an existing static PR95
  profile generator and parity contract.
- `src/tac/pr95_hnerv.py` contains reusable PR95-family archive wire helpers for
  single-member `0.bin` ZIP packets and the length-prefixed brotli top blob.

New deterministic helper:

- `tools/recover_pr95_training_curriculum.py` reads the local PR95 source tree,
  emits the stage schedule and cost campaign estimate, and intentionally does
  not import torch, load scorers, dispatch providers, or spend GPU.
- `src/tac/tests/test_recover_pr95_training_curriculum.py` locks the recovered
  stage sequence, 29650 epoch total, Stage 4 QAT, Stage 8 Muon, and required GPU
  estimate rows.

## Concrete PR95 Curriculum

The source orchestrator is
`experiments/results/public_pr_intake_full/public_pr95_intake_20260505_auto/source/submissions/hnerv_muon/src/train.py`.
It builds stages 1-8 in order, carries each stage's final decoder and latents
forward, and emits a final codec/archive stage. The source comment describes
this as an eight-stage curriculum from random initialization and states
approximately 50 hours on a single GPU, but the hardware is not specified, so
that value is an anchor, not a launch estimate.

| stage | source | epochs | optimizer | seg surrogate | QAT | C1a | resume |
|---:|---|---:|---|---|---|---|---|
| 1 | `src/stages/stage1_v328_ce.py` | 3000 | AdamW `1e-3` | `ce_seg_loss` | no | lambda `0.0`, sigma `0.2` | random init |
| 2 | `src/stages/stage2_v331_softplus.py` | 5650 | AdamW `1e-3` | `tau_softplus_seg_loss` | no | lambda `0.0`, sigma `0.2` | Stage 1 final |
| 3 | `src/stages/stage3_v332_smooth.py` | 1500 | AdamW `1e-4` | `smooth_disagreement_seg_loss` | no | lambda `0.0`, sigma `0.2` | Stage 2 final |
| 4 | `src/stages/stage4_v332_qat.py` | 500 | AdamW `1e-4` | `smooth_disagreement_seg_loss` | yes | lambda `0.0`, sigma `0.2` | Stage 3 final |
| 5 | `src/stages/stage5_c1a_l7.py` | 9000 | AdamW `3e-5` | `l7_softplus_seg_loss` | yes | lambda `0.01`, sigma `0.2` | Stage 4 final |
| 6 | `src/stages/stage6_lambda_sweep.py` | 2000 | AdamW `3e-5` | `l7_softplus_seg_loss` | yes | lambda `0.02`, sigma `0.2` | Stage 5 final |
| 7 | `src/stages/stage7_sigma_sweep.py` | 3000 | AdamW `3e-5` | `l7_softplus_seg_loss` | yes | lambda `0.02`, sigma `0.1` | Stage 6 final |
| 8 | `src/stages/stage8_muon_finetune.py` | 5000 | AdamW `1e-5` plus Muon `2e-4`, Muon WD `5e-4` | `l7_softplus_seg_loss` | yes | lambda `0.02`, sigma `0.1` | Stage 7 final |

Total recovered release-view epochs: `29650`.

Training-loop evidence:

- Stage config and shared loop:
  `.../hnerv_muon/src/stages/common.py`.
- Model:
  `.../hnerv_muon/src/model.py`. HNeRV decoder uses latent dim `28`, base
  channels `36`, initial grid `6x8`, six PixelShuffle upsample blocks, bilinear
  skip terms, sinusoidal residual blocks, dilated refinement, and separate RGB
  heads for the two frames in each pair.
- Optimizer:
  `.../hnerv_muon/src/optim.py`. Stage 8 partitions hidden 2D+ weights into
  Muon and leaves stem/RGB/bias/1D parameters plus latents on AdamW.
- Losses:
  `.../hnerv_muon/src/losses.py`. Segmentation loss moves from CE to tau
  softplus to smooth disagreement to L7 hard-pixel softplus; pose loss uses the
  scorer-domain `sqrt(10*MSE + eps)` form; aggregate loss applies the contest
  weights `100*seg + pose + cat_lambda*C1a`.
- QAT:
  `.../hnerv_muon/src/losses.py` fake-quantizes trainable decoder weights
  during QAT stages and restores original weights before optimizer step.
- Differentiable YUV6:
  `.../hnerv_muon/src/data.py` patches the upstream `rgb_to_yuv6` path so pose
  gradients reach the decoder instead of being severed by the original
  no-grad/in-place implementation.
- Eval roundtrip:
  `.../hnerv_muon/src/stages/common.py` decodes at `384x512`, bicubic-upsamples
  to `874x1164`, bilinear-downsamples back to `384x512`, then applies
  differentiable round-to-uint8 before scorer preprocessing.
- Eval cadence:
  `.../hnerv_muon/src/stages/common.py` evaluates every 25 epochs by building
  the EMA archive, parsing it back through the archive reader, evaluating the
  parsed model/latents, and writing `best_archive.bin`, `decoder_f32.pt`,
  `latents_f32.pt`, and `best_meta.json`.
- Archive codec:
  `.../hnerv_muon/src/codec.py` stores INT8 decoder tensors and per-dimension
  uint8 latent min/scale/delta streams inside length-prefixed brotli blobs.
- Runtime contract:
  `.../hnerv_muon/inflate.sh` implements the contest
  `inflate.sh DATA_DIR OUTPUT_DIR FILE_LIST` signature and expects
  `DATA_DIR/<base>.bin` for each listed video. `inflate.py` parses that payload,
  renders frames, bicubic-upsamples to camera resolution, rounds to uint8, and
  writes raw RGB.

## PR101 And Current Sane HNeRV Gap Analysis

PR101 evidence:

- PR101 source:
  `experiments/results/public_pr_intake_full/public_pr101_intake_20260505_auto/source/submissions/hnerv_ft_microcodec/`.
- PR101 `README.md` states it is built on PR95/PR98 and lists microcodec,
  entropy-map, latent-tail, correction-sidecar, hard-pixel residual, and ZIP
  repack improvements. It contains codec/runtime files but no recovered
  eight-stage trainer.
- Verdict: PR101 gold appears codec/runtime polishing on top of PR95-family
  HNeRV payloads. It is not sufficient evidence for PR95's training discipline.

Current internal `sane_hnerv` gaps:

- Architecture mismatch:
  `src/tac/substrates/sane_hnerv/architecture.py` defaults to a different grid,
  channel schedule, block count, output scale, and skip/refine structure than
  PR95. It cannot be assumed to be PR95-equivalent.
- Training mismatch:
  `experiments/train_substrate_sane_hnerv.py` is a one-stage AdamW path with
  default `2000` epochs, no eight-stage curriculum, no PR95 stage handoffs, no
  QAT schedule, no C1a lambda/sigma sweep, and no Muon partition.
- Loss mismatch:
  `src/tac/substrates/sane_hnerv/score_aware_loss.py` is a useful scorer-domain
  substrate, but it does not encode PR95's stage-specific segmentation
  surrogates or C1a entropy regularizer.
- Archive mismatch:
  `src/tac/substrates/sane_hnerv/archive.py` uses SHV1 with pickled fp16
  state_dict and int16 latents. PR95 uses INT8 decoder tensors plus uint8
  latent delta streams inside brotli blobs and a different runtime packet.
- A1/PR101 fine-tune mismatch:
  phase-A1 PR101 scripts start from a PR101 packet. They are useful for
  codec-level frontier work but are not a from-random PR95 curriculum replay.

Apples-to-apples blocker: a PR95 reproduction must preserve architecture,
losses, optimizer partition, QAT, archive grammar, inflate contract, scorer
runtime, and CPU/CUDA axis labels. Decoded tensor or latent parity alone is not
full-frame inflate parity or scorer parity.

## Full-Cost Campaign Plan

Budget is no longer a feasibility blocker. The full 29650-epoch reproduction is
allowed in principle if the prerequisites below are wired and the operator
authorizes a paid run. The cost table uses Modal GPU-task rates read from
<https://modal.com/pricing> on 2026-05-13. These are planning assumptions, not
commitments; verify live provider rates immediately before any dispatch.

The GPU-hour bands are conservative planning bands until the smoke gate measures
seconds per epoch in the actual wired trainer. PR95 source claims approximately
50 hours on an unspecified single GPU, so the H100 band treats that claim as
plausible but not authoritative.

| GPU | planning GPU-hours | rate assumption USD/hr | planning cost | note |
|---|---:|---:|---:|---|
| H100 | 45-70 | 3.9492 | 177.71-276.44 | fastest recommended first full-run target after smoke |
| A100 40GB | 90-140 | 2.0988 | 188.89-293.83 | good price/perf fallback |
| A100 80GB | 80-125 | 2.4984 | 199.87-312.30 | more memory than needed unless batching changes |
| L40S | 110-170 | 1.9512 | 214.63-331.70 | viable but slower wall clock |
| T4 | 250-400 | 0.5904 | 147.60-236.16 | cheaper but high wall-clock/timeout risk |

Stop gates and artifacts:

| gate | trigger | required artifact/evidence | stop condition |
|---|---|---|---|
| smoke | import, scorer probe, yuv6 grad reachability, 1-2 batches, archive parse smoke | `smoke_manifest.json`, source tree SHA, dependency/import probe, seconds/epoch estimate, tiny archive parse manifest | any scorer/runtime import failure, yuv6 gradient failure, archive parse failure, or measured sec/epoch exceeding operator cap |
| Stage 1 | 3000 CE epochs complete | `stage1/final_decoder.pt`, `stage1/final_latents.pt`, `stage1/best_meta.json`, `stage1/best_archive.bin`, loss trend | CE/pose proxy fails to decrease over a measured window, NaN, or handoff checkpoint missing |
| Stage 4 QAT | 3000+5650+1500+500 epochs complete | Stage 4 checkpoint pair, QAT weight-delta report, parsed archive roundtrip, proxy component report | QAT causes component collapse or archive/eval roundtrip fails |
| Stage 7 pre-Muon | Stage 7 complete before optimizer swap | Stage 7 checkpoint pair, `best_archive.bin`, C1a entropy trend, section-byte manifest | C1a does not reduce entropy/bytes or regresses components beyond operator threshold |
| Stage 8 Muon | 5000 Muon finetune epochs complete | Stage 8 checkpoint pair, Muon/AdamW partition manifest, final `best_archive.bin`, archive SHA | Muon step unstable, NaN, optimizer partition drift, or worse than Stage 7 under same contract |
| exact CUDA/CPU eval | byte-closed archive/runtime packet exists | claimed CUDA and CPU `contest_auth_eval*.json`, `inflated_outputs_manifest.json`, runtime tree SHA, logs, terminal claim rows | missing claim, axis mismatch, dependency closure failure, missing custody, or score regression |

Full reproduction vs Stage-8-only finetune:

- Full 29650-epoch replay is the only clean way to establish PR95 training
  discipline from random initialization. It produces per-stage handoff
  artifacts, validates the stage curriculum, and is the correct apples-to-apples
  foundation for later exact-eval claims.
- Stage-8-only finetune can be a lower-cost Muon/C1a polish probe if it starts
  from a recovered Stage 7 checkpoint. No such Stage 7 checkpoint is currently
  evidenced in this tree. Starting from the public PR95/PR101 quantized archive
  is not equivalent to Stage 7 f32 handoff because codec quantization and
  payload transforms have already occurred.
- Stage-8-only from a quantized public archive can still be useful as a
  cheap engineering smoke of Muon partition, C1a, QAT, export, and exact-eval
  wiring, but it cannot claim PR95 curriculum reproduction.

## Wiring Required Before Any Paid Run

Minimum exact prerequisites:

1. Add a byte-faithful PR95 trainer substrate or adapter that preserves the
   public source architecture, staged configs, loss family, QAT behavior, C1a,
   differentiable YUV6 patch, eval roundtrip, and Muon/AdamW partition.
2. Add a canonical CLI, likely
   `experiments/train_substrate_pr95_hnerv_muon.py`, with plan-only default,
   explicit `--execute` or equivalent only after operator approval, full/stage8
   modes, deterministic seeds, checkpoint paths, and stage manifest emission.
3. Wire PR95 archive build/parse to `src/tac/pr95_hnerv.py` or a dedicated
   reusable module with tests for parse/build byte closure, no-op controls, and
   runtime inflate contract.
4. Add tests for:
   architecture shape and output range, yuv6 gradient reachability, eval
   roundtrip dimensions, StageConfig digest, Muon parameter partition, QAT
   apply/restore, codec parse/build roundtrip, and CLI plan-mode no-spend.
5. Add a smoke manifest schema and a stage manifest schema recording git SHA,
   dirty status, source tree SHA, dependency closure, hardware, seeds,
   optimizer state, checkpoints, archive SHA, and logs.
6. Wire paid-run surfaces through `tools/claim_lane_dispatch.py`, provider
   dry-run, import probe, artifact harvest, and terminal claim updates. Do not
   bury provider-specific runtime logic inside the lane experiment.
7. Run exact CUDA and CPU eval only after the archive/runtime packet is
   byte-closed and the lane has an active claim. Keep CPU and CUDA axes separate.

Six-hook status for this memo/helper:

- `tac.sensitivity_map.*`: not applicable to this research-only recovery pass;
  future exact PR95 artifacts must feed scorer component deltas into the normal
  sensitivity map.
- Pareto constraint: not applicable until a candidate archive exists; future
  PR95 candidate must carry bytes, seg, pose, and axis-labeled score.
- Bit allocator hook: future archive profiler must expose PR95 decoder/latent
  section bytes through the bit allocator.
- Cathedral autopilot dispatch hook: blocked until the trainer, claim, and
  provider dry-run are wired.
- Continual-learning posterior update: future smoke/stage/exact artifacts must
  enter the dated research ledger with positive or negative evidence class.
- Probe-disambiguator: needed if choosing between full replay, Stage-8-only
  quantized restart, or Stage-8 from a recovered f32 Stage 7 checkpoint.

## Lane Claim And Operator Recipe

Recommended lane ids:

- Full run:
  `lane_f1_pr95_full_curriculum_repro_20260513`
- Stage-8-only engineering probe:
  `lane_f1_pr95_stage8_muon_probe_20260513`

Before any paid dispatch, use the real claim helper surface verified by
`tools/claim_lane_dispatch.py claim --help`. Example dry-run shape:

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id lane_f1_pr95_full_curriculum_repro_20260513 \
  --platform modal \
  --instance-job-id pr95_full_curriculum_plan_<timestamp> \
  --agent codex:f1_pr95_curriculum \
  --predicted-eta-utc <UTC_ISO8601> \
  --status training_planned \
  --notes "operator-authorized PR95 full 29650-epoch reproduction; provider rates verified live" \
  --dry-run
```

Paid dispatch requires replacing `--dry-run` with an actual active claim only
after:

- the operator names the provider/GPU/cost ceiling and authorizes spend;
- live provider rates are verified;
- smoke plan mode is green;
- provider import probe and artifact harvest path are green;
- the parent is told the run is ready and explicitly allows launch.

Terminal rows must be appended through the same helper with a precise terminal
status such as `completed_exact_eval`, `failed_import_probe`,
`failed_training`, or `stopped_operator`.

## Final Verdict

READY-TO-WIRE, not ready-to-dispatch.

PR95's training discipline is now concretely recovered from local source and
the full-budget path is acceptable in principle. The block is no longer budget.
The block is exact wiring: a byte-faithful PR95 trainer/runtime path, tests,
stage manifests, claim lifecycle, provider dry-run/import probe, and exact
CUDA/CPU harvest must exist before paid launch. No score promotion, no dispatch,
and no GPU spend occurred in this pass.
