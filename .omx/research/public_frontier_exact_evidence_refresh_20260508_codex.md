# Public frontier exact-evidence refresh - 2026-05-08

Generated UTC: `2026-05-08T21:49:06Z`

Scope: no-dispatch public PR state/comment refresh and exact-evidence language
audit. Evidence grade: `external_github_pr_comment_refresh` plus
`local_state_audit_no_score`. Score claim: false. Dispatch performed: false.

## GitHub refresh

- Upstream repo queried with `gh`: `commaai/comma_video_compression_challenge`.
- PR comment scorecard refreshed for PR100 through PR108 with
  `tools/public_pr_eval_comment_scorecard.py`.
- Fresh scorecard matched
  `reports/public_pr100_108_eval_comment_scorecard_20260508.json`
  byte-for-byte.
- `gh pr list --state all --limit 40` showed PR108 as the newest public PR;
  no PR109+ exists at this refresh.
- PR108 remains open with no host auth-eval comment in the refreshed scorecard.

Conclusion: no new public auth-eval comments and no new public frontier signal
were observed. Existing public-comment rows remain unchanged:

| PRs checked | New signal |
| --- | --- |
| `100,101,102,103,104,105,106,107,108` | none |

## Current exact-evidence queue

Read-only claim summary reported `active=3`, all pending authorization/prestage,
and no stale nonterminal claims:

| lane_id | instance/job_id | platform | status |
| --- | --- | --- | --- |
| `apogee_int6_contest_cuda_anchor` | `PRESTAGE:apogee-int6-cuda-anchor-20260508-PLACEHOLDER` | `lightning` | `pending_authorization` |
| `pr101_admm_step6_no_dead_k` | `PRESTAGE:admm-no-dead-k-20260508-PLACEHOLDER` | `lightning` | `pending_authorization` |
| `pr107_apogee_cpu_auth_eval_linux_x86_64` | `PRESTAGE:pr107-cpu-eval-lightning-20260508-PLACEHOLDER` | `lightning` | `pending_authorization` |

`lightning_active_jobs.json` shows the latest arch-shrink/lossy-coarsening
Lightning rows as terminal or failed-harvest states, including
`lossy-coarsening-cuda-20260508T0312-noproject` completed at
`0.351719` and `arch-shrink-x0-4-lightning-20260508T024304Z` at
`failed_artifact_rsync_rc_255`. No remote status probe, harvest, submission,
or dispatch was attempted in this refresh.

## CPU/CUDA language audit

- `reports/latest.md` currently keeps the axes distinct:
  `[contest-CUDA]`, `[contest-CPU]`, `[macOS-CPU advisory only]`, and
  `[CPU-prep]` are used separately, and CPU-prep rows are marked
  non-promotable.
- `.omx/research/frontier_roadmap_evidence_correction_20260508_worker_a.md`
  already supersedes the stale `0.20454` formula projection and identifies the
  current local HNeRV A++ anchor as PR103-on-PR106 at score
  `0.20898105277982337` / `185578` bytes.
- Patched `.omx/research/pr106_uniward_lagrangian_exact_cuda_regression_20260508_codex.md`
  so the exact PR106 UNIWARD regression compares against the PR106 source
  adapter exact CUDA baseline `0.20945673680571203`, not the stale unanchored
  `0.20454` projection.

Operational status: public frontier unchanged; exact-evidence work remains a
queued/prestage local state issue, not a reason to dispatch from this turn.
