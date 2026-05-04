# Trained Renderer Transplant Recovery Worker - 2026-05-03

Evidence grade: active recovery / no-score operational evidence. Score claim:
false. Promotion eligible: false. Remote exact eval dispatched: false.

## Scope

Worker-owned outputs live under
`experiments/results/trained_renderer_transplant_recovery_worker_20260503/`.
No partner WIP was reverted or edited. Existing lane result directories were
read for sentinels/metadata only; recovered status and logs were mirrored into
the worker-owned directory.

## Current C089 Anchor

- Archive:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip`
- Eval JSON:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/contest_auth_eval.adjudicated.json`
- Bytes: `276342`
- SHA-256:
  `0ec53e5b871149ed6eea56c0b9bcca3baec998d5bfad4f371979e0c90e62cea8`
- Exact CUDA score: `0.3154707273953505`
- SegNet: `0.00061038`
- PoseNet: `0.00049601`

This is the source archive to use for C089 transplant pose-safety if a Modal
burn returns a renderer export.

## Modal Recovery Status

Poll time: `2026-05-03T11:09:08Z`.

| GPU | call ID | label | status | artifacts |
| --- | --- | --- | --- | --- |
| H100 | `fc-01KQP9K42CAWJH7XEV4KC0V28M` | `c067_fixed_renderer_burn_fix3_modal_h100_20260503T0659Z` | active / still running | none |
| A100 | `fc-01KQP9T1VD14785MG63H7JM5VK` | `c067_fixed_renderer_burn_fix3_modal_a100_20260503T0702Z` | active / still running | none |
| A10G | `fc-01KQP9T19Y7PMDETDN99WDMF2W` | `c067_fixed_renderer_burn_fix3_modal_a10g_20260503T0702Z` | active / still running | none |

Official `experiments/modal_recover_lane.py` output for all three calls reports
`STILL RUNNING`. No terminal Modal result dict was returned, so there are no
checkpoint, snapshot, renderer export, archive, bytes, or SHA records to
triage yet.

## Preserved Worker Artifacts

- Summary:
  `experiments/results/trained_renderer_transplant_recovery_worker_20260503/triage_summary.json`
- Modal API poll:
  `experiments/results/trained_renderer_transplant_recovery_worker_20260503/modal_recovery/poll_summary.json`
- Modal app list:
  `experiments/results/trained_renderer_transplant_recovery_worker_20260503/modal_recovery/modal_app_list.txt`
- H100 official recovery stdout:
  `experiments/results/trained_renderer_transplant_recovery_worker_20260503/modal_recovery/h100/modal_recover_lane_stdout.txt`
- A100 official recovery stdout:
  `experiments/results/trained_renderer_transplant_recovery_worker_20260503/modal_recovery/a100/modal_recover_lane_stdout.txt`
- A10G official recovery stdout:
  `experiments/results/trained_renderer_transplant_recovery_worker_20260503/modal_recovery/a10g/modal_recover_lane_stdout.txt`
- H100 app log tail:
  `experiments/results/trained_renderer_transplant_recovery_worker_20260503/modal_recovery/h100/modal_app_logs_tail.txt`
- A100 app log tail:
  `experiments/results/trained_renderer_transplant_recovery_worker_20260503/modal_recovery/app_ap-mFnRt0CeC0nmT1ZKzCIx2v_logs_tail.txt`
- A10G app log tail:
  `experiments/results/trained_renderer_transplant_recovery_worker_20260503/modal_recovery/app_ap-LsyGen7NErI9urFui6MUIo_logs_tail.txt`

The app logs only show source copy and lane start lines, not terminal export
lines.

## Gate Status

- Terminal export artifact: blocked, none returned.
- Candidate archive bytes/SHA: none.
- C089 transplant/readiness: not run; blocked on renderer export.
- Renderer transplant pose-safety: not run; blocked on candidate archive.
- Dispatch claim: not created; no dispatchable candidate exists.
- Remote exact eval: not performed.

## Safe Post-Export Gate Sequence

After a Modal call returns a terminal export, extract the recovered QZS3
`renderer.bin` and build C089-preserving candidates locally:

```bash
.venv/bin/python experiments/build_renderer_shrink_candidate.py \
  --source-archive experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip \
  --renderer-export <recovered_renderer_qzs3.bin> \
  --output-dir experiments/results/trained_renderer_transplant_recovery_worker_20260503/<candidate_id>/c089_transplant \
  --qzs3-block-sizes 32,48,64,96,128 \
  --force
```

Then run pose-safety against the exact selected source/candidate SHA pair:

```bash
.venv/bin/python experiments/preflight_renderer_transplant_pose_safety.py \
  --source-archive experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip \
  --candidate-archive <selected_candidate_archive.zip> \
  --output-json experiments/results/trained_renderer_transplant_recovery_worker_20260503/<candidate_id>/pose_safety_preflight.json \
  --max-pairs 5
```

Do not dispatch exact eval unless `safe_for_exact_eval_dispatch=true`, the
candidate archive bytes/SHA match the local manifest, and a fresh
`tools/claim_lane_dispatch.py claim ...` succeeds for a unique transplant lane
or a deliberately reused lane with non-conflicting active state.

## Highest-EV Next Action

Poll the H100 call first:

```bash
.venv/bin/python experiments/modal_recover_lane.py \
  --label c067_fixed_renderer_burn_fix3_modal_h100_20260503T0659Z
```

Reason: H100 was spawned first and has the earliest claim ETA. Starting a
duplicate burn or remote exact eval now would add custody ambiguity without a
candidate artifact.

## 2026-05-03 C091 Readiness Supersession

Evidence grade: empirical local readiness planning. Score claim: false.
Promotion eligible: false. Remote GPU dispatch: none. Dispatch claims changed:
none.

The current exact score anchor is now C091 PR75 public replay, superseding the
older C089/C067 source default for this recovery-to-candidate handoff:

- Archive:
  `experiments/results/lightning_batch/exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/archive.zip`
- Eval JSON:
  `experiments/results/lightning_batch/exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/contest_auth_eval.adjudicated.json`
- Bytes: `276481`
- SHA-256:
  `03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746`
- Exact CUDA score: `0.31516575028285976`
- SegNet: `0.00060804`
- PoseNet: `0.00049371`
- Sample count: `600`

Local readiness artifact:

- JSON:
  `experiments/results/trained_renderer_transplant_recovery_worker_20260503/c091_readiness/handoff_manifest.json`
- Markdown:
  `experiments/results/trained_renderer_transplant_recovery_worker_20260503/c091_readiness/handoff_manifest.md`

Result: no terminal renderer export exists locally for
`c067_fixed_renderer_burn_fix3_modal_h100_20260503T0659Z`, so no candidate
archive was built and no pose-safety preflight was run. The readiness manifest
fails closed with `exact_eval_dispatch_ready=false`,
`terminal_exports_exist=false`, and `safe_for_exact_eval_dispatch=false`.

The C091 handoff planner now emits the exact post-export local gate sequence
using the PR75/QP1-safe renderer shrink builder and the mandatory renderer
transplant pose-safety preflight. The first command for main remains the H100
recovery poll:

```bash
.venv/bin/python experiments/modal_recover_lane.py --call-id fc-01KQP9K42CAWJH7XEV4KC0V28M
```

After that command returns terminal artifacts, provide the recovered QZS3
renderer export to the generated local builder command in
`handoff_manifest.json`. Do not create a dispatch claim or submit exact eval
until the selected candidate archive has a matching
`experiments/preflight_renderer_transplant_pose_safety.py` JSON with
`safe_for_exact_eval_dispatch=true` for the exact C091 source archive SHA and
candidate archive SHA pair.
