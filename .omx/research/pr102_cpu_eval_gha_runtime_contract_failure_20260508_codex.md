# PR102 GHA CPU Eval Runtime-Contract Failure - 2026-05-08

## Summary

Codex attempted a Linux x86_64 GitHub Actions CPU auth-eval dispatch for the
public PR102 archive after claiming lane `public_pr102_cpu_auth_eval_gha`.

- archive: `experiments/results/public_pr102_hnerv_lc_v2_scale095_rplus1_custody_20260507_codex/public_pr102_intake_20260507_auto/archive.zip`
- archive bytes: `178981`
- archive SHA-256: `afd53348f50303bf0ec6a7ffecc1ac037df2f1c70745244b9c45c72e8eb80641`
- workflow run: `25571165912`
- workflow URL: `https://github.com/adpena/comma_video_compression_challenge/actions/runs/25571165912`
- lane terminal status: `failed_workflow_run_25571165912`
- evidence grade: `run abort`
- score claim: `false`

No adjudicated CPU score was produced.

## Failure Class

The fork workflow downloads only `archive.zip` from `submission_url`, then
executes:

```text
uv run --group "$UV_GROUP" bash evaluate.sh --device "$EVAL_DEVICE" --submission-dir ./submissions/<submission_name>
```

The checkout must already contain `submissions/<submission_name>/inflate.sh`.
The attempted dispatch used a unique non-baseline `submission_name` without
`--pr-number`, so the workflow checked out `master`, downloaded `archive.zip`
into a new directory, and failed at evaluate time with:

```text
ERROR: ./submissions/pr102_hnerv_lc_v2_scale095_rplus1_cpu_20260508T175932Z/inflate.sh not found
```

This is a dispatcher/runtime-contract bug, not a scientific result and not a
negative result for PR102.

## Guard Landed

`tools/dispatch_cpu_eval_via_github_actions.py` now fails fast for non-baseline
dispatches without `--pr-number`, because a fork PR merge ref is required to
provide the matching `submissions/<submission_name>/inflate.sh` runtime tree.

Reactivation path:

1. Create or identify a fork PR whose merge ref contains a unique submission
   runtime directory with `inflate.sh`.
2. Re-dispatch with the same archive SHA and `--pr-number <fork_pr_number>`.
3. Preserve `contest_auth_eval.adjudicated.json`, `report.txt`, workflow URL,
   archive SHA/bytes, and terminal dispatch-claim row.
