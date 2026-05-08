# Arch Shrink x0.4 Lightning Live Strategy - Worker L3 - 2026-05-08

generated_at_utc: 2026-05-08T07:03:05Z
lane_id: `arch_shrink_x0.4_lightning`
active_job: `arch-shrink-x0-4-lightning-20260508T024304Z`
scope: read-only strategy review; no job stop; no evidence append; no claim mutation

## Decision

Continue observing the active job. Do not manually stop it.

Reason: the live telemetry is internally healthy for an in-progress training
run: Lightning SDK status is `Running`, heartbeat was current through
`2026-05-08T07:01:37Z`, GPU utilization was 95-100%, CUDA training entered
Phase 1, loss fell from `0.0937` at epoch 0 to `0.0208` at epoch 150, and
half-frame instrumentation is firing exactly as intended:
`hf_fires=150/150 (1.00)`, `hf_warp_diff=0.0267`,
`hf_target_prob=1.000`.

But modify future launches. At the observed `94.5s/ep`, the 3000-epoch schedule
has a remaining ETA near 75h, while the launch script default runtime cap is
18h and the active claim predicts terminal time `2026-05-08T20:43:09Z`.
This T4 "full" launch is therefore very unlikely to reach Stage 5 archive
build or Stage 6 exact CUDA auth eval. Treat it as a useful live training
signal and checkpoint source, not as a likely terminal `[contest-CUDA]` score
producer.

## Rules Applied

- AGENTS evidence rules require A++ score evidence to use exact archive custody,
  the canonical `archive.zip -> inflate.sh -> upstream/evaluate.py` path, CUDA,
  full sample count, and adversarial review (`AGENTS.md:528-552`).
- AGENTS score-truth rules say proxy/loss/local telemetry cannot promote, rank,
  kill, retire, or validate the lane; exact CUDA auth eval on exact archive
  bytes wins (`AGENTS.md:554-572`).
- CLAUDE requires chained experiments to end with CUDA auth eval against the
  best checkpoint and treats training loss as signal only
  (`CLAUDE.md:354-371`).
- Q-FAITHFUL successors must record stage shape and early-stop criteria, keep
  `eval_roundtrip=True`, use the deployed pose stream, and produce deterministic
  archives before any score use (`AGENTS.md:1125-1159`).

## Current Evidence

Local state:

- Branch: `main`.
- Current local HEAD observed: `3e3eb15072d06880a7035435b724221e4a0f4a95`.
- The active job source manifest was generated earlier at
  `2026-05-08T02:43:05Z` from HEAD
  `522c96121d07bf7d30b05ddcbcdf07b121e9f6ab`; do not assume the current dirty
  worktree is byte-identical to the staged job.
- Source manifest:
  `experiments/results/lightning_batch/arch-shrink-x0-4-lightning-20260508T024304Z/source_manifest.json`
  with 2360 files and `artifact_paths` including
  `experiments/results/lane_a_landed/optimized_poses.pt`.
- Manifest/file hash checks that still match the current local copy:
  - `src/tac/experiments/train_renderer.py`
    `b412005ff6aa9824a26d0ec614f5a09d1a672d2b8529b22f29b403ac34342cc2`
  - `src/tac/profiles.py`
    `107c206d6670b8e05763abeb52073f60ea46df9620a609cf09f7889abb36f947`
  - `experiments/results/lane_a_landed/optimized_poses.pt`
    `e0cd8ccb29bb3cd8613285e1633a6466f8a01f5df101d80243ec52d50d2fb85b`
- `experiments/arch_shrink_x0.4_lightning_full.py` no longer matches the
  staged manifest hash, so terminal review should cite the manifest and
  captured command, not the current local script contents.

Dispatch state:

- Active claim row:
  `2026-05-08T02:43:09Z | claude_lab | arch_shrink_x0.4_lightning | lightning | arch-shrink-x0-4-lightning-20260508T024304Z | 2026-05-08T20:43:09Z | active_dispatching`.
- Notes on that claim say it is the round-3 fix for the prior pose-forward and
  forced-pose-artifact staging failure.
- `.omx/state/lightning_active_jobs.json` has an active row for this job with
  `machine=g4dn.2xlarge`, `profile=q_faithful_dilated_88k`,
  `target_elements=88000`, and pending evidence tag `[contest-CUDA]`.
- Lightning SDK read-only query at `2026-05-08T07:03:05Z`: status `Running`.

Remote artifacts observed read-only:

- The live Studio checkout artifact path was absent from this SSH view.
- The persisted job artifact mirror existed and contained Stage 1/2 outputs:
  `archive_masks_seed.zip`, `build_masks.log`, `extracted/masks.mkv`,
  `extracted/optimized_poses.pt`, `extracted/renderer.bin`,
  `provenance.json`, `run.log`, `heartbeat.log`, `train.log`, and
  `train/training_state_arch_shrink_x0_4_arch-shrink-x0-4-lightning-20260508T024304Z.pt`.
- Training state checkpoint size observed remotely:
  `1,648,384` bytes, mtime about `2026-05-08T06:57Z`.
- Terminal score artifacts were absent:
  - no final `archive.zip`
  - no `contest_auth_eval.json`
  - no `auth_eval.log`

Prior related failure:

- `arch-shrink-x0-4-lightning-20260508T020205Z` failed before scoring with
  `RuntimeError: Q-FAITHFUL forward requires an explicit deployed pose tensor`.
- The current active job has passed that startup failure: train log shows the
  deployed pose contract active with pose SHA
  `e0cd8ccb29bb3cd8613285e1633a6466f8a01f5df101d80243ec52d50d2fb85b`.

## Q-FAITHFUL No-Warp String

The train log string is expected:

`JointFrameGenerator built: 87,836 params (target ~88K). NO motion, NO warp, single-mask + FiLM-pose.`

Evidence:

- `src/tac/quantizr_faithful_renderer.py:14-21` documents the intended
  architecture: no motion module, no optical flow, no warp, single odd-mask
  input.
- `src/tac/quantizr_faithful_renderer.py:259-318` implements
  `JointFrameGenerator.forward(mask2, pose6) -> (frame1, frame2)`, with frame1
  FiLM-conditioned on pose and frame2 unconditional.
- `src/tac/experiments/train_renderer.py:2375-2435` routes
  `variant=quantizr_faithful` to a shim that intentionally discards `mask_t`,
  requires explicit `pose=`, and prints the no-warp architecture string.
- `src/tac/profiles.py:3370-3389` says this profile should dispatch to
  `JointFrameGenerator`, not the warp-based default renderer.

The confusing part is profile metadata: `src/tac/profiles.py:3445-3446` still
sets `use_zoom_flow=True` and `mask_half_sim_prob=1.0`. In this Q-FAITHFUL
path, `use_zoom_flow=True` is being used to activate half-frame mask
simulation/ego-flow plumbing in the training loop, while the Q-FAITHFUL shim
swallows `ego_flow` and consumes the deployed pose tensor. That is awkward
metadata reuse, not evidence that the live job is training the wrong
architecture. Future cleanup should split this into an explicit single-mask
half-frame contract instead of overloading `use_zoom_flow`.

## Loss And Telemetry Interpretation

The current telemetry argues for continued observation, not score confidence:

- Good signs:
  - CUDA is active.
  - Half-frame branch fires at target probability 1.0.
  - `hf_warp_diff` is nonzero, so half-frame simulation is not an identity
    no-op.
  - Loss dropped quickly from epoch 0 to epoch 10 and then continued a slow
    decrease through epoch 150.
  - The pose stream is loaded and hash-recorded.

- Weak signs:
  - The log is still Phase 1 anchor loss only; no final archive, no exact CUDA
    JSON, and no score-grade component fields exist.
  - The epoch rate makes the declared full run impossible inside the current
    Lightning 18h cap.
  - Only a `training_state_*.pt` checkpoint was observed so far. If future code
    tries to export from a training-state checkpoint, the current export snippet
    must support the `model` or `ema_shadow` keys, not just `model_state_dict`
    or `state_dict`.

Classification: `in_progress_training_signal`, evidence grade `empirical /
invalid for scoring`.

## Early-Stop Criteria

For this active job: do not manually stop it in this turn. Let it keep producing
loss/heartbeat/checkpoint signal unless the operator separately authorizes a
stop.

For future Q-FAITHFUL/arch-shrink launches, stop early or relaunch modified if
any of these happen:

- Infrastructure failure:
  - Lightning SDK is `Running` but heartbeat is stale for more than 15 minutes
    and two read-only SSH checks cannot read a newer log/checkpoint.
  - CUDA disappears, GPU memory drops to idle for more than 15 minutes during
    training, or logs show CPU/MPS fallback.
  - Train log has `Traceback`, `RuntimeError`, OOM, NaN/inf loss, or a
    checkpoint mtime stale for more than 60 minutes while SDK remains running.

- Contract failure:
  - `qfaithful_training_pose_contract` is missing or not promotable.
  - `hf_fires` is materially below target for two consecutive logged epochs
    after the half-frame schedule should be active.
  - `hf_warp_diff` is zero-ish (`<= 0.001`) or nonfinite for two consecutive
    logged epochs when half-frame simulation is expected.
  - The no-warp Q-FAITHFUL shim is not receiving `pose=` in the forward path.

- Economic/schedule failure:
  - Estimated full training + archive + auth-eval time exceeds job
    `max_runtime` by more than 25% and the run does not have an explicit resume
    or graceful checkpoint/export plan.
  - At Phase 1 end (`epoch 600`) the training loss has not improved materially
    from the epoch 150 band or no half-frame-gated best checkpoint exists.
    Do not kill the method family; retire only the measured launch plan and
    relaunch with a shorter/resumed/faster plan.

## Harvest Cadence

While SDK status is `Running`:

- Use read-only SDK status plus read-only remote tails every 15 minutes during
  Phase 1/P2. This matches the log cadence of 10 epochs at about 15.75 minutes.
- Record the latest heartbeat, latest logged epoch, loss, `hf_fires`,
  `hf_warp_diff`, checkpoint mtime, and whether terminal artifacts exist.
- Do not run the mutating harvester as a mere strategy poll if the only goal is
  read-only status. The current harvester can mutate claim/evidence state once
  a job is terminal.

At terminal SDK status (`Completed`, `Failed`, `Stopped`, `Cancelled`) or if
the operator explicitly asks for harvest:

- Run `experiments/arch_shrink_x0.4_lightning_harvest.py --job-name arch-shrink-x0-4-lightning-20260508T024304Z`.
- Prefer the persisted `/teamspace/jobs/.../artifacts/pact/...` mirror if the
  live Studio checkout path is absent.
- If no exact auth-eval JSON exists, classify narrowly and close the claim with
  a terminal failure/status row. Do not append `[contest-CUDA]` evidence.

## Exact Terminal Review Process

Before any score row, complete this review packet:

1. Harvest the local artifact mirror and list final artifacts.
2. Verify `archive.zip` exists, record bytes and SHA-256, and compare to
   `contest_auth_eval.json` provenance fields.
3. Verify `contest_auth_eval.json` is exact CUDA, full sample count, and T4 or
   contest-equivalent. Use the strict validator in
   `src/tac/deploy/lightning/round3_harvest.py`.
4. Recompute:
   `score = 100 * seg_dist + sqrt(10 * pose_dist) + 25 * archive_bytes / 37545489`.
5. Verify the scored path is exactly `archive.zip -> inflate.sh ->
   upstream/evaluate.py`, no scorer loaded at inflate time, and runtime-tree
   custody exists.
6. Confirm the dispatch claim receives a terminal row matching the outcome.
7. Classify narrowly:
   `legitimate score movement`, `measured-config regression`,
   `runtime/dependency failure`, `max_runtime_before_auth_eval`,
   `archive/runtime bug`, `component collapse`, or `indeterminate`.
8. If the result is bad, retire only the measured config unless independent
   exact evidence and review support a broader conclusion.

## Code/Test Gaps After Active Job Completes

Close these after the active job is terminal or after the operator authorizes
modifying future launches:

- Add a prelaunch runtime feasibility guard for training dispatches:
  estimated epochs * seconds/epoch + archive/eval margin must fit inside
  `--max-runtime-sec`, or the launcher must require a resume plan. The current
  T4 telemetry shows 3000 epochs cannot fit in 18h.
- Add an explicit read-only `--status-only` or `--no-terminal-mutate` mode to
  the arch-shrink harvester, so strategy reviews do not need ad hoc SDK code to
  avoid terminal claim/evidence mutation.
- Split Q-FAITHFUL metadata: replace overloaded `use_zoom_flow=True` with an
  explicit `single_mask_half_frame` or equivalent contract, while preserving
  current behavior until tested.
- Add regression coverage that the Q-FAITHFUL training path passes both
  `pose=` and harmless `ego_flow=` through `forward_kwargs`.
- Add export support or a fail-closed test for `training_state_*.pt` checkpoints
  whose state dict is under `model`/`ema_shadow`; the current remote export
  snippet only handles `model_state_dict`/`state_dict` cleanly.
- Add terminal failure classification for `max_runtime_before_auth_eval` and
  make the terminal claim note preserve latest epoch, loss, checkpoint path,
  checkpoint SHA, and reactivation/resume criteria.
- For the next score-producing launch, prefer one of:
  - resume on faster GPU with enough wall-clock to complete all 3000 epochs;
  - shorten the schedule with an explicit evidence tag and exact auth-eval;
  - add a planned graceful `--wall-clock-timeout` before the provider cap,
    followed by deterministic export and exact CUDA auth eval of the best
    compatible checkpoint.
