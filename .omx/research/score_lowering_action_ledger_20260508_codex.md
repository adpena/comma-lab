# Score-Lowering Action Ledger - Codex - 2026-05-08

generated_at_utc: `2026-05-08T12:30:04Z`
scope: local evidence review and next-action routing only
write_set_used: `.omx/research/score_lowering_action_ledger_20260508_codex.md`
remote_jobs_launched: `false`
score_claim: `false`

## Inputs Reviewed

- `reports/latest.md`
- `reports/public_pr100_108_eval_comment_scorecard_20260508.json`
- `reports/pr102_dual_device_auth_eval_plan_20260508.json`
- `reports/pr102_public_pr_cpu_auth_eval_plan_20260508.json`
- `.omx/research/pr102_hardened_exact_replay_result_20260508_codex.json`
- `.omx/research/public_pr_auth_eval_comment_drift_and_dual_axis_protocol_20260508_codex.md`
- `.omx/research/evidence_grade_drift_pr102_pr104_pr106_codex_20260508.md`
- `.omx/research/pr104_exact_replay_dispatch_status_20260508_codex.md`
- `.omx/research/pr106_uniward_lagrangian_exact_cuda_regression_20260508_codex.md`
- `.omx/research/arch_shrink_x0_4_lightning_faithfulness_audit_20260508_codex.md`
- `.omx/research/jacobian_fisher_reactivation_cpu_safety_20260508_codex.md`
- `.omx/research/beta_jacobian_fisher_path_b_cpu_candidate_20260508_codex.md`
- `.omx/state/active_lane_dispatch_claims.md`
- `.omx/state/lightning_batch_jobs.json`
- exact JSONs under `experiments/results/lightning_batch/`

## Current Exact Anchors

| anchor | score | bytes | sha256 | status |
| --- | ---: | ---: | --- | --- |
| PR103-on-PR106 AC repack | `0.20898105277982337` | `185578` | `ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce` | active local HNeRV rate anchor |
| PR106 public adapter | `0.20945673680571203` | `186239` | `3fefbe5dfdd738179a55ca5c995ff8f63ec2755662d60684706f20d313913f58` | predecessor PR106 substrate |
| PR106 `x` repack | `0.20945123680571204` | `186231` | `d25bca80057e8b533197895b4c56370678feb4e05fea0312c405bd12f29bec8e` | predecessor rate-only control |
| PR102 hardened replay | `0.22839372989108092` | `178981` | `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641` | exact CUDA drift evidence, not frontier |
| PR104 hardened replay | `0.23113446620399658` | `178637` | `6564c32a9edeeaf08abd7f0ea673ba2fda23444605ca207eb4ba794cc66797b8` | exact CUDA replay gap closed, not frontier |
| PR106 UNIWARD rms=0.05 | `0.3371617511972341` | `150511` | `0641b8ac8084b362b80e7c5bbe3c122946a8dbf7843fcbcd9f445aee7a56af7b` | A-negative scoped config only |
| PR101 lossy coarsening analytical | `0.351718793322788` | `156404` | `ab8a8a13c70b3d3bbf2ce3d8a81a77691b776a6e0fb1cbe9ce504dc3f59c1b28` | A-negative scoped config only |

Interpretation: the current internal CUDA frontier is not the public CPU medal
band. Rate-only HNeRV work must compare against `185578` bytes unless it is a
scorer-changing packet with exact CUDA component evidence.

## PR Comment CPU/CUDA Drift

The public PR comment scorecard shows PR100, PR101, PR102, PR103, and PR105
have lower CPU comment bands than CUDA comment bands. PR102 is the clean paired
case: local exact CUDA `0.22839372989108092` matches the public CUDA comment
band `0.228390831180`, while the CPU comment band is `0.195376176526`.

Action consequence: CPU replay is a public-leaderboard reproduction axis only.
It must not promote, rank, or kill internal CUDA lanes, but it is high value
for explaining public substrate claims before adapting PR102/PR101/PR103 ideas.

Verified planning surfaces:

```bash
.venv/bin/python tools/public_pr_eval_comment_scorecard.py \
  --pr-range 100 108 \
  --json-out reports/public_pr100_108_eval_comment_scorecard_20260508_refresh.json

.venv/bin/python tools/plan_dual_device_auth_eval.py \
  --public-pr 102 \
  --json-out reports/pr102_dual_device_auth_eval_plan_20260508_refresh.json
```

## Active Harvest State

`arch_shrink_x0.4_lightning` remains the highest-upside architecture path, but
it is already owned by an active claim:

- active claim: `arch-shrink-x0-4-lightning-20260508T024304Z`
- latest local audit: running at `2026-05-08T10:46:34Z`, no final
  `archive.zip`, no `contest_auth_eval.json`, no score artifact
- earlier `020205Z` mirror failed in training with the zero-pose fallback guard,
  so it is not method evidence

Next action is harvest-only, no relaunch and no duplicate dispatch:

```bash
.venv/bin/python experiments/arch_shrink_x0.4_lightning_harvest.py \
  --job-name arch-shrink-x0-4-lightning-20260508T024304Z \
  --teamspace comma-lab \
  --user adpena \
  --ssh-target ${LIGHTNING_SSH_TARGET} \
  --once
```

If terminal artifacts are absent, classify custody/infrastructure precisely.
If exact CUDA JSON lands, review against PR103-on-PR106 before any promotion.

## Highest-EV Next Actions

### 1. PR102/PR106 substrate fork: keep the branches separate

PR102 is a runtime/source atom over PR100-identical archive bytes and currently
explains CPU/public drift, not a CUDA byte win. PR106/PR103-on-PR106 is the
CUDA score substrate. The next executable work is to produce two separate
manifests:

- PR102 branch: compress/decode parity and paired Linux CPU/CUDA plans before
  treating scale/r+1 as a stack atom.
- PR106 branch: parser-proven section candidates only; no member-level pose or
  mask budget claims.

Disjoint write target for future worker: new files under `reports/` or a new
timestamped `experiments/results/public_pr_substrate_branch_*` root. No code
edits are required to start.

### 2. Rebase rate-only packer work onto the `185578`-byte anchor

The old PR106x/PR106 rate-only threshold is superseded. A rate-only candidate
must beat `185578` bytes while preserving PR103-on-PR106 runtime behavior, or
it should not consume exact CUDA. The fastest local step is a byte-closed
archive candidate plus strict manifest, not a remote eval.

Use `tools/pr106_archive_decomposition.py` first if a candidate claims logical
section budgets:

```bash
.venv/bin/python tools/pr106_archive_decomposition.py \
  --summary-text \
  --output-json reports/pr106_pr103_active_anchor_layout_20260508.json
```

### 3. Path-B/Jacobian-Fisher: move from raw K selection to guarded blends

Raw beta/Jacobian-Fisher selected Ks built a `147285` byte archive but raised
CPU fp32 smoke rel_err to `0.0873994010`, so it is rejected by the new guard.
The additive-cap-3 blend is safer: `153378` bytes, `0.0512570250` aggregate
fp32 smoke rel_err, and `-293` bytes versus default no-dead-K.

Next executable local action is a guarded rebuild plus manifest review, not
exact CUDA:

```bash
.venv/bin/python tools/build_admm_x_lossy_coarsening_path_b_step6_no_dead_k.py \
  --selected-Ks-json experiments/results/beta_jacobian_fisher_path_b_cpu_20260508T115156Z/jacobian_fisher_allocation_manifest.json \
  --selected-Ks-rms-target 0.0386 \
  --selected-Ks-additive-baseline-cap 3 \
  --selected-Ks-max-fp32-smoke-rel-err 0.06 \
  --output-root experiments/results/path_b_jacobian_fisher_addcap3_20260508
```

Promotion gate: exact negative PR106 rms=0.05 and PR101 lossy-coarsening
negative mean any new K-coarsening dispatch needs scorer-aware pullback or a
documented trust-region reason. Do not infer safety from byte savings.

### 4. Cross-paradigm ADMM x Op1 finalizer: package before eval

Worker D produced a byte-closed CPU packet at `153513` bytes with
`cuda_eval_worth_testing=true`, but strict compliance still fails because the
release packet lacks required submission-side manifest/report/auth-eval
closure. Highest local value is packet staging and fail-closed compliance,
still without a remote job.

```bash
.venv/bin/python tools/build_cross_paradigm_admm_x_op1_finalizer.py \
  --output-root experiments/results/cross_paradigm_admm_x_op1_followup_20260508

.venv/bin/python tools/claim_lane_dispatch.py claim \
  --dry-run \
  --lane-id cross_paradigm_admm_continuous_k_plus_op1_finalizer \
  --platform lightning \
  --instance-job-id exact_eval_cross_paradigm_admm_x_op1_followup_pending \
  --agent codex \
  --status active_exact_eval \
  --notes dry_run_conflict_check_only_no_dispatch
```

The dry-run claim is useful only to verify lane conflict state. A real claim
and exact eval remain outside this task.

### 5. CPU-vs-CUDA closure for PR101/PR103/PR105 before public-branch claims

PR101 and PR103 public CPU rows are the visible medal-band leaders, but local
structured CUDA custody is incomplete for several rows. Use paired plans to
avoid mixing devices:

```bash
.venv/bin/python tools/plan_dual_device_auth_eval.py \
  --public-pr 101 \
  --json-out reports/pr101_dual_device_auth_eval_plan_20260508.json

.venv/bin/python tools/plan_dual_device_auth_eval.py \
  --public-pr 103 \
  --json-out reports/pr103_dual_device_auth_eval_plan_20260508.json
```

These commands plan only unless `--execute` is explicitly supplied.

## Do Not Spend Exact CUDA On

- PR102 as a CUDA score-lowering claim: exact CUDA is already non-frontier and
  the lower public row is CPU-axis drift.
- PR106 UNIWARD rms=0.05 or uniform RMS retreads: exact CUDA already returned
  `0.3371617511972341`.
- Raw beta/Jacobian-Fisher selected Ks without the additive cap or a stronger
  scorer-aware trust region.
- Any HNeRV rate-only candidate above `185578` bytes unless it changes scorer
  components with charged runtime proof.
- A duplicate arch-shrink launch while the active `024304Z` claim is open.

## Summary

The highest-EV score-lowering path is still architecture/rate-stack work on the
PR106/PR103-on-PR106 CUDA substrate, with arch-shrink harvest first if it lands.
The highest-EV local-only continuation is guarded Path-B/Jacobian-Fisher or
cross-paradigm packet staging, because both can generate byte-closed artifacts
without touching parent-edited code. The PR102/PR101/PR103 public CPU signal is
valuable for explanation and substrate selection, but it is not CUDA promotion
evidence.
