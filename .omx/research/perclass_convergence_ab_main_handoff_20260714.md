# Per-class convergence A/B — exact main handoff — 2026-07-14

Pointer unchanged: `0.1910828242 [contest-CPU]` submittable; bank `0.1880443980` remains
non-submission defensive. This is a BUILD + preflight handoff. No 150-epoch Metal run, exact scorer
replay, byte-close, score claim, or pointer mutation has occurred.

## Binding order

Run sequentially: A/CE, then B/MARGIN. Analyze A/B before deciding whether the already-built C/STEP
and D/M+ADAM arms still have enough value to run. Do not add a fifth arm. Every real arm remains
gated by its own green two-pass `--dry-start 2`; a dry-start failure never falls through to training.

## 0. Storage waterfall

The builder measured `827380576256` free bytes on Vertigo, but its sandbox could not create the
workload root. On the M5-Max, update the same content-addressed receipt and fail closed if this command
does not select/create `/Volumes/VertigoDataTier/pact/perclass_convergence_ab_20260714`:

```zsh
.venv/bin/python tools/plan_experiment_storage.py \
  --output .omx/research/perclass_convergence_ab_storage_preflight_20260714.json \
  --workload-subdir perclass_convergence_ab_20260714 \
  --reserve-free-gb 100 \
  --requested-bytes 21474836480 \
  --min-free-bytes 107374182400 \
  --expected-workload-root /Volumes/VertigoDataTier/pact/perclass_convergence_ab_20260714 \
  --policy-id perclass_convergence_ab_storage_v1 \
  --policy-schema perclass_convergence_ab_storage_preflight.v1 \
  --lifecycle-kind retained_stage_checkpoints_and_logs \
  --expected-output-sha256 f0ed6b1fba6916d93cf8666496458880c66dff0f79c32c4d2f1deb1a3c9f173e \
  --create
```

Do not opt into local storage. The trainer's atomic, distinct stage/periodic checkpoint path and
existing certified cleanup apparatus remain binding.

## 1. Dispatch custody is per arm

Each independently spawned trainer has its own job id. Claim immediately before that arm's dry-start,
then append a terminal row with the same lane/job id after either the dry-start failure or completed
training. A terminal row is what frees this single sequential lane for the next arm. Never leave one
conceptual campaign claim covering several independent trainer processes. Every terminal invocation
must retain the opening invocation's exact lane id, platform, instance/job id, agent, and
`--ttl-hours 168`; only status and the evidence-bearing notes change.

## 2. Arm A — CE control

Claim A:

```zsh
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id perclass_convergence_ab \
  --platform local-m5-max-metal \
  --instance-job-id perclass_convergence_ce_20260714 \
  --agent main \
  --status train \
  --ttl-hours 168 \
  --notes 'Task #494 arm A CE; dry-start then real n600/150 only if green; pointer unmoved'
```

Gate; require both `boot_ok=true` and `resume_round_trip_ok=true` in the emitted report:

```zsh
.venv/bin/python tools/launch_witness_run.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 \
  --epochs 150 \
  --config perclass_convergence_ce_20260714 \
  --out-dir /Volumes/VertigoDataTier/pact/perclass_convergence_ab_20260714/ce \
  --label perclass_convergence_ce_20260714 \
  --purpose 'Task #494 matched real-n600 CE convergence-rate control; sequential arm A' \
  --mem-preflight-safe-frac 0.70 \
  --no-dashboard \
  --dry-start 2
```

Only after that gate is green:

```zsh
.venv/bin/python tools/launch_witness_run.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 \
  --epochs 150 \
  --config perclass_convergence_ce_20260714 \
  --out-dir /Volumes/VertigoDataTier/pact/perclass_convergence_ab_20260714/ce \
  --label perclass_convergence_ce_20260714 \
  --purpose 'Task #494 matched real-n600 CE convergence-rate control; sequential arm A' \
  --mem-preflight-safe-frac 0.70
```

Wait for A to finish and preserve all stage/periodic checkpoints before B.

On successful completion, close A:

```zsh
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id perclass_convergence_ab \
  --platform local-m5-max-metal \
  --instance-job-id perclass_convergence_ce_20260714 \
  --agent main \
  --status completed_local_mlx_training \
  --ttl-hours 168 \
  --notes 'Arm A trajectory plus preserved stage/periodic checkpoints landed; score_claim=false; pointer_moved=false'
```

If the dry-start fails, do not run training; close A instead:

```zsh
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id perclass_convergence_ab \
  --platform local-m5-max-metal \
  --instance-job-id perclass_convergence_ce_20260714 \
  --agent main \
  --status failed_dry_start_gate \
  --ttl-hours 168 \
  --notes 'Arm A dry-start failed; see ce/dry_start_report.json; no training; pointer_moved=false'
```

## 3. Arm B — zero-margin winner/rival hinge

Claim B only after A has a terminal row:

```zsh
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id perclass_convergence_ab \
  --platform local-m5-max-metal \
  --instance-job-id perclass_convergence_margin_20260714 \
  --agent main \
  --status train \
  --ttl-hours 168 \
  --notes 'Task #494 arm B zero-margin winner-rival; dry-start then real n600/150 only if green; pointer unmoved'
```

Gate:

```zsh
.venv/bin/python tools/launch_witness_run.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 \
  --epochs 150 \
  --config perclass_convergence_margin_20260714 \
  --out-dir /Volumes/VertigoDataTier/pact/perclass_convergence_ab_20260714/margin \
  --label perclass_convergence_margin_20260714 \
  --purpose 'Task #494 matched real-n600 zero-margin winner-rival convergence-rate arm; sequential arm B' \
  --mem-preflight-safe-frac 0.70 \
  --no-dashboard \
  --dry-start 2
```

Only after that gate is green:

```zsh
.venv/bin/python tools/launch_witness_run.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 \
  --epochs 150 \
  --config perclass_convergence_margin_20260714 \
  --out-dir /Volumes/VertigoDataTier/pact/perclass_convergence_ab_20260714/margin \
  --label perclass_convergence_margin_20260714 \
  --purpose 'Task #494 matched real-n600 zero-margin winner-rival convergence-rate arm; sequential arm B' \
  --mem-preflight-safe-frac 0.70
```

After completion, close B:

```zsh
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id perclass_convergence_ab \
  --platform local-m5-max-metal \
  --instance-job-id perclass_convergence_margin_20260714 \
  --agent main \
  --status completed_local_mlx_training \
  --ttl-hours 168 \
  --notes 'Arm B trajectory plus preserved stage/periodic checkpoints landed; score_claim=false; pointer_moved=false'
```

On a dry-start failure, do not train; retain `--ttl-hours 168` and run that same closure with
`--status failed_dry_start_gate` and
`--notes 'Arm B dry-start failed; see margin/dry_start_report.json; no training; pointer_moved=false'`.

## 4. Analyze the decisive A/B pair

```zsh
.venv/bin/python tools/probe_ordinal_perclass_convergence.py \
  --ce-receipt /Volumes/VertigoDataTier/pact/perclass_convergence_ab_20260714/ce/perclass_convergence_trajectory.json \
  --margin-receipt /Volumes/VertigoDataTier/pact/perclass_convergence_ab_20260714/margin/perclass_convergence_trajectory.json \
  --output .omx/research/perclass_convergence_ab_analysis_20260714.json
```

The analyzer's rare-versus-common gap closure is the preregistered A/B verdict. The update axis is the
matched INSTANCE authority; wall time is the measured cost companion. `max(d_seg-0.005318,0)` is only
an upper bound on curable excess. It is not permission to attribute the whole residual to optimizer,
loss, or basis. A naive/first-cut negative remains INSTANCE-scoped and must leave the best-known
reformulation queued.

## 5. Arm C — step-native HOSC/FINER partition-indicator route

Run only after A/B analysis says the representation-rate question remains worth the wall clock.

Claim C first:

```zsh
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id perclass_convergence_ab \
  --platform local-m5-max-metal \
  --instance-job-id perclass_convergence_stepnative_20260714 \
  --agent main \
  --status train \
  --ttl-hours 168 \
  --notes 'Task #497 arm C step-native; dry-start then real n600/150 only if green; pointer unmoved'
```

```zsh
.venv/bin/python tools/launch_witness_run.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 \
  --epochs 150 \
  --config perclass_convergence_stepnative_20260714 \
  --out-dir /Volumes/VertigoDataTier/pact/perclass_convergence_ab_20260714/stepnative \
  --label perclass_convergence_stepnative_20260714 \
  --purpose 'Task #497 matched real-n600 step-native convergence-rate arm; sequential arm C' \
  --mem-preflight-safe-frac 0.70 \
  --no-dashboard \
  --dry-start 2
```

Only after that gate is green:

```zsh
.venv/bin/python tools/launch_witness_run.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 \
  --epochs 150 \
  --config perclass_convergence_stepnative_20260714 \
  --out-dir /Volumes/VertigoDataTier/pact/perclass_convergence_ab_20260714/stepnative \
  --label perclass_convergence_stepnative_20260714 \
  --purpose 'Task #497 matched real-n600 step-native convergence-rate arm; sequential arm C' \
  --mem-preflight-safe-frac 0.70
```

After successful training, close C:

```zsh
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id perclass_convergence_ab \
  --platform local-m5-max-metal \
  --instance-job-id perclass_convergence_stepnative_20260714 \
  --agent main \
  --status completed_local_mlx_training \
  --ttl-hours 168 \
  --notes 'Arm C trajectory plus preserved stage/periodic checkpoints landed; score_claim=false; pointer_moved=false'
```

On a dry-start failure, do not train; retain `--ttl-hours 168` and run that same closure with
`--status failed_dry_start_gate` and
`--notes 'Arm C dry-start failed; see stepnative/dry_start_report.json; no training; pointer_moved=false'`.

## 6. Arm D — M+Adam

Run only after C completes or is explicitly deferred; do not overlap their projected 24.48-GiB peaks.

Claim D first:

```zsh
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id perclass_convergence_ab \
  --platform local-m5-max-metal \
  --instance-job-id perclass_convergence_mplus_adam_20260714 \
  --agent main \
  --status train \
  --ttl-hours 168 \
  --notes 'Task #496 arm D M+Adam; dry-start then real n600/150 only if green; pointer unmoved'
```

```zsh
.venv/bin/python tools/launch_witness_run.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 \
  --epochs 150 \
  --config perclass_convergence_mplus_adam_20260714 \
  --out-dir /Volumes/VertigoDataTier/pact/perclass_convergence_ab_20260714/mplus_adam \
  --label perclass_convergence_mplus_adam_20260714 \
  --purpose 'Task #496 matched real-n600 M+Adam convergence-rate arm; sequential arm D' \
  --mem-preflight-safe-frac 0.70 \
  --no-dashboard \
  --dry-start 2
```

Only after that gate is green:

```zsh
.venv/bin/python tools/launch_witness_run.py \
  --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
  --num-pairs 600 \
  --epochs 150 \
  --config perclass_convergence_mplus_adam_20260714 \
  --out-dir /Volumes/VertigoDataTier/pact/perclass_convergence_ab_20260714/mplus_adam \
  --label perclass_convergence_mplus_adam_20260714 \
  --purpose 'Task #496 matched real-n600 M+Adam convergence-rate arm; sequential arm D' \
  --mem-preflight-safe-frac 0.70
```

After successful training, close D:

```zsh
.venv/bin/python tools/claim_lane_dispatch.py claim \
  --lane-id perclass_convergence_ab \
  --platform local-m5-max-metal \
  --instance-job-id perclass_convergence_mplus_adam_20260714 \
  --agent main \
  --status completed_local_mlx_training \
  --ttl-hours 168 \
  --notes 'Arm D trajectory plus preserved stage/periodic checkpoints landed; score_claim=false; pointer_moved=false'
```

On a dry-start failure, do not train; retain `--ttl-hours 168` and run that same closure with
`--status failed_dry_start_gate` and
`--notes 'Arm D dry-start failed; see mplus_adam/dry_start_report.json; no training; pointer_moved=false'`.

C/D emit the same validated trajectory schema and can be adjudicated through the existing analyzer's
`derive_rates` API, but they must not be relabeled as the strict CE-versus-margin B treatment. No
promotion or pointer movement is permitted until a winner is exact-byte-closed on an authority axis.
