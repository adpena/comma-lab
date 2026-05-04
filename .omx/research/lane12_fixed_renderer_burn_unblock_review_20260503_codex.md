# Lane 12 / Fixed-Renderer Burn Unblock Review - 2026-05-03

Scope: Lane 12 NeRV and C067 fixed-mask/fixed-pose renderer burn unblock
review only. No GPU dispatch was performed in this pass. No score claim,
promotion claim, or archive ranking is made here.

## Verdict

Dispatch recommendation: **NO new dispatch now**.

Reason:

- Lane 12 NeRV remains structurally blocked by missing L2 clearance.
- Fixed-renderer burn has an audited override path, but active same-family
  H100/A100 claims already exist and must be harvested, closed, or superseded
  before any new paid job is launched.
- No post-burn plug-in renderer artifact exists locally that has passed
  trained-renderer transplant preflight, pose-safety preflight, byte-closed
  archive build, and exact CUDA auth eval.

## Lane 12 NeRV Status

Lane 12 NeRV is not dispatchable.

Live guard evidence:

- `.omx/state/lane12_nerv_l2_clearance.json` is absent
  (`test -f` exit code `1`).
- `scripts/remote_lane_nerv.sh` requires the clearance packet before any NeRV
  retraining. The packet must include:
  `cleared_for_retraining_unblock=true`, `lane12_l2=true`,
  `geometry_gate_passed=true`, `grand_council_clean_passes >= 3`, and cited
  evidence.
- `RUN_AUTH_EVAL=1` additionally requires candidate pose-regeneration
  provenance and Alpha-Geo provenance with `pass_fail.overall_pass=true`.
- The older OWv3/0120 NeRV stack script still runs pose regeneration and exact
  eval directly and does not implement the current Lane 12 L2 clearance gate.
  Treat it as stale/unsafe for new dispatch until it is brought under the same
  claim, L2, pose-provenance, and JSON-adjudication rules.

Known negative anchor:

- The prior `jsonfix40` NeRV archive remains a measured implementation negative
  with PoseNet collapse (`avg_posenet_dist=49.7784996` in the existing Lane 12
  readiness ledger). That does not kill NeRV broadly, but it blocks reusing the
  artifact as a dispatch seed.

## Fixed-Renderer Burn Status

Fixed-renderer burn is a separate C067 renderer-only training path. It is not
Lane 12 NeRV retraining and does not create Lane 12 L2 clearance.

Prepared local packet:

- Manifest:
  `experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503/fixed_c067_renderer_burn_manifest.json`
- Runner:
  `experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503/run_fixed_renderer_burn.sh`
- Source archive:
  `experiments/results/lightning_batch/exact_eval_c063_fixedslice_equiv_t4_20260502T0855Z/archive.zip`
- Source archive SHA-256:
  `226475de42ec00d66287a39f98fe6d2eb0464b90738714e5ef05fa4ee8efb38a`

Fixed runtime members:

| member | bytes | sha256 |
|---|---:|---|
| `renderer.bin` | 59288 | `5657593aec0bf380f0bd614578cc4a76da8589b6ef8ce0331c28b6a4d6658efb` |
| `masks.mkv` | 223385 | `a5c2b89c110d75220cd09b2f27f2e92844626ae7ed0d2c797290dcf43c7068eb` |
| `optimized_poses.bin` | 7200 | `5236cf75cc95f6a9731b35f4e2fb1211aa99bfcc69e5964869edd19a7af3df9f` |

The burn command is parser-closed against the live argparse surfaces:

- `src/tac/experiments/train_renderer.py`: dry-run parse OK.
- `scripts/q_faithful_snapshot_loop.py`: dry-run parse OK.
- Training uses fixed C067 masks and poses, `--device cuda`,
  `--deterministic`, `--no-auth-eval-on-best`, and `--wall-clock-timeout 82800`.
- Snapshot export uses `--renderer-codec qzs3`, `--submission-layout
  pr64_mask_first_single_blob`, `--eval-mode none`, and
  `--dispatch-claim-mode none`.

Current active-claim state:

- Active H100 p5 claim:
  `c067_fixed_renderer_burn_qfaithful_fix2_h100p5` /
  `train_c067_fixed_renderer_burn_qfaithful_fix2_h100p5_20260503T0544Z`
  status `training` in `.omx/state/active_lane_dispatch_claims.md`.
- Active A100 p4d claim:
  `c067_fixed_renderer_burn_qfaithful_fix2_a100p4d` /
  `train_c067_fixed_renderer_burn_qfaithful_fix2_a100p4d_20260503T0544Z`
  status `training` in `.omx/state/active_lane_dispatch_claims.md`.
- Lightning state snapshots at 2026-05-03T06:14:42Z still showed both jobs
  `Pending`, with identity confidence `name_only` and no visible job id. A
  dry-run attempt to claim the H100 p5 lane refused dispatch because the active
  claim exists.

## Blocked Plug-In Artifact Evidence

The already-tested QBF1 candidate family is not usable as a plug-in renderer
artifact:

- Candidate:
  `experiments/results/trained_renderer_blockfp_preflight_20260503_qfaithful_qzs3_2146/trained_qbf1_b0512/archive.zip`
- Archive bytes: `283432`
- Archive SHA-256:
  `6174424516a93aa35852cd4dc22b9132516fc38f824b77562b3f91b3f55f58dc`
- Pose-safety preflight:
  `safe_for_exact_eval_dispatch=false`
- Output parity failure:
  `mean_abs_delta=72.1708679`, `rms_delta=87.6108700`,
  `max_abs_delta=254.8052521`
- Exact CUDA eval:
  `score_recomputed_from_components=17.72267562501643`,
  `final_score=17.72`, `avg_posenet_dist=29.82484055`,
  `avg_segnet_dist=0.0026408`, `n_samples=600`

This is a renderer-induced PoseNet collapse and byte regression, not a
promotable transplant seed.

Current export-unlock scan:

- `experiments/results/trained_renderer_export_unlock_20260503_review/plan.json`
  reports `readiness.verdict="blocked_no_h100_dispatch"` and blocker
  `no non-surrogate trained-renderer archive passed preflight`.

## Fastest Contest-Faithful 24h Burn Plan

Do not launch a new job while the active H100/A100 claims are open.

Fast path if the existing H100/A100 jobs start and produce artifacts:

1. Harvest the remote run directory and job artifact path recorded in the
   batch-job JSON for the exact claim.
2. Verify `logs/training_runner_preflight.json`,
   `logs/lightning_supply_chain_scan_pre.json`, `logs/nvidia_smi_preflight.txt`,
   `logs/train_renderer.log`, and `logs/q_faithful_snapshot_loop.log`.
3. Require at least one exported snapshot archive under the claimed run's
   `snapshots/` directory with no exact-eval side effects.
4. Extract the candidate `renderer.bin`; reject if it is source-identical,
   pickle-only, sidecar-dependent, missing magic, or not consumed by the
   reviewed runtime loader.
5. Run `experiments/preflight_trained_renderer_transplant.py` against the fixed
   source archive and exported renderer.
6. Run `experiments/preflight_renderer_transplant_pose_safety.py` against the
   selected source/candidate archive SHA pair.
7. Rerun trained-renderer transplant preflight with `--pose-safety-json`.
8. Only if the plan reports `h100_ready_after_claim`, claim
   `c067_trained_renderer_self_compression_blockfp` and run exact CUDA auth eval
   through `experiments/contest_auth_eval.py` via the Lightning exact-eval
   submitter.
9. Promotion still requires T4/equivalent A++ confirmation on identical archive
   bytes, component trace, adjudicated JSON, runtime tree hash, and manifest.

Fast path if both active jobs remain pending or fail before useful artifacts:

1. Append terminal rows for the existing H100/A100 lane claims with precise
   statuses such as `failed_provider_pending_no_gpu_spend`,
   `stopped_pending_no_visible_artifacts`, or the measured provider status.
2. Keep the failed provider records and source manifests as negative
   infrastructure evidence.
3. Relaunch only after a fresh claim succeeds for a distinct lane/job id and
   the submitter can prove artifact visibility before training starts.
4. Prefer one H100 p5 24h burn first. Use A100 p4d as a bounded hedge only
   under an explicit paid-compute override and separate claim, because both are
   training-only snapshot producers and cannot make score claims by themselves.

## Verification Run In This Review

Commands run locally:

```text
bash -n scripts/remote_lane_nerv.sh
bash -n scripts/remote_lane_12_owv3_0120_nerv_stack.sh
bash -n experiments/results/c067_fixed_renderer_burn_prep_20260503/c067_qfaithful_fixedmask_fixedpose_seed20260503/run_fixed_renderer_burn.sh
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest src/tac/tests/test_lane12_nerv_dependency_closure.py src/tac/tests/test_preflight_nerv_codec_discipline.py src/tac/tests/test_prepare_c067_fixed_renderer_burn.py src/tac/tests/test_preflight_trained_renderer_transplant.py src/tac/tests/test_preflight_renderer_transplant_pose_safety.py src/tac/tests/test_plan_trained_renderer_export_unlock.py -q
test -f .omx/state/lane12_nerv_l2_clearance.json
.venv/bin/python tools/claim_lane_dispatch.py claim --dry-run --lane-id c067_fixed_renderer_burn_qfaithful_fix2_h100p5 --platform lightning --instance-job-id review_no_dispatch --agent codex:gpt-5.5 --predicted-eta-utc 2026-05-04T06:00Z --status training --notes lane12_review_no_dispatch
```

Results:

- Shell syntax checks passed.
- Focused pytest: `43 passed in 4.93s`.
- L2 clearance packet check failed as expected: file absent.
- Dry-run claim refused dispatch because active same-lane H100 p5 claim exists.

## Dispatch Recommendation

No new Lane 12 NeRV dispatch.

No new fixed-renderer burn dispatch until the active H100/A100 claims are
terminally closed or confirmed as the continuing burn. The active claims can be
allowed to continue only as training/export producers with `score_claim=false`;
they are not score evidence and they do not justify exact-eval dispatch until
the post-burn transplant, pose-safety, byte-closure, and exact CUDA gates pass.
