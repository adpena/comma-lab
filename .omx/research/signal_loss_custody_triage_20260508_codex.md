# Signal Loss Custody Triage - 2026-05-08

Scope: read-only custody triage for untracked and modified artifacts in the
shared `main` worktree. No files were deleted, cleaned, reverted, staged, or
committed during this triage. No dispatch was attempted and no score claim is
made here.

## Audit Evidence

- `.venv/bin/python tools/audit_research_state_tracking.py --repo-root .`
  reported 4,233 scanned files. Key dispositions: `track_in_git=964`,
  `externalize_with_manifest=1004`, `summarize_to_research_ledger=1649`,
  `canonicalize_to_research_ledger=562`, `keep_private_local=20`,
  `manual_review=4`.
- Filtered research-state review classified `.claude/scheduled_tasks.lock` as
  `private_operator_state` / `keep_private_local`, and classified
  `reports/dual_layer_stc_av1_evidence.jsonl` as `public_lab_doc` /
  `track_in_git`.
- `.venv/bin/python tools/audit_untracked_source_artifacts.py --repo-root . --format json --disposition-manifest .omx/research/untracked_source_dispositions_20260505_codex.json`
  reported `untracked_source_like_count=0`, `undispositioned_count=0`,
  `invalid_disposition_count=0`.
- `.venv/bin/python tools/audit_nested_gitlink_custody.py --repo-root . --format json --local-custody-manifest .omx/research/local_custody_release_manifest_20260505_codex.json`
  reported 9 dirty gitlinks, all documented, with 0 warnings.
- `.venv/bin/python tools/audit_reverse_engineering_tree.py --repo-root . --summary`
  reported `files=693 blockers=0`.
- `.venv/bin/python tools/audit_recovery_custody_snapshots.py --repo-root . --format json`
  reported required recovery snapshots present and no unresolved blocked
  recovery inputs, live diffs, or incomplete manual rehydration paths.
- `.venv/bin/python tools/audit_recovered_remote_lanes.py --repo-root . --format json`
  reported 3 recovered lane scripts present with valid shell syntax and custody
  markers.
- `.venv/bin/python tools/audit_git_signal_loss.py --mode worktree --format summary --max-paths 200`
  reported `missing=224 historical=2689 mode=worktree`; this is a historical
  recovery inventory, not evidence of a new deletion in this triage.

## Custody Classification

Track in git:

- `reports/dual_layer_stc_av1_evidence.jsonl` is 1,714 bytes of structured
  public lab evidence. It records a proxy/CPU-prep Filler-Pevny dual-layer
  STC+AV1 result with `score_claim=false`, `dispatch_attempted=false`, and
  explicit dispatch blockers. It should be preserved as a small report summary.
- This ledger should be tracked as the durable triage record.

Leave local only:

- `.claude/scheduled_tasks.lock` contains only local lock churn
  (`pid`, `procStart`, `acquiredAt`) and should not be promoted.
- `experiments/results/recovered_42_dead/recovery_metadata.json` and
  `experiments/results/recovered_99999_phantom/recovery_metadata.json` changed
  only run timestamps and elapsed seconds while keeping `ssh_reachable=false`
  and no recovered artifacts. Leave as local/generated recovery metadata.
- `experiments/results/frontier_roadmap_status_20260507_codex/status.json` and
  `rebuild_command.txt` are generated roadmap status refresh artifacts. The
  current status has `score_claim=false`, `dispatch_attempted=false`,
  `dirty_path_count=54`, and `dirty_blocked_row_count=2`; do not treat it as a
  promotion artifact without a fresh candidate-specific manifest.

Externalize or keep as raw custody, not direct git:

- Large untracked result trees include `experiments/results/pr107_cpu_eval_20260508`
  at about 3.4 GiB, `experiments/results/lightning_batch/lossy-coarsening-cuda-20260508T024250Z`
  at about 3.4 GiB, `experiments/results/lightning_batch/lossy-coarsening-cuda-20260508T0312-noproject`
  at about 3.4 GiB, the PR102 public source custody tree at about 356 MiB, and
  the raw Kaggle nested checkout at about 3.8 GiB. Preserve locally or
  externalize with a committed manifest; do not commit the raw payloads.
- Ignored exact-eval outputs under `experiments/results/lightning_batch/**`
  remain local custody. PR102 hardened exact replay is already summarized in
  `.omx/research/pr102_hardened_exact_replay_result_20260508_codex.json` and
  reverse-engineering manifests, so the raw Lightning directory should remain
  local/externalized.
- Untracked `experiments/results/lane_*`, `lossy_coarsening_*`,
  `admm_x_lossy_*`, `pr101_pysr_*`, and similar run directories are experiment
  outputs. Track only distilled ledgers or final manifests after owner review;
  otherwise leave local or externalize.
- `outputs/` is generated output and should stay local unless a small,
  curated manifest is promoted separately.

Dirty nested gitlinks:

- Public PR gitlinks under `experiments/results/public_pr*_intake_*` are
  documented by `public_pr_intake_gitlinks_forensic` in
  `.omx/research/local_custody_release_manifest_20260505_codex.json`.
- The raw Kaggle nested checkout is documented by
  `kaggle_raw_ingest_externalized` in the same manifest.
- Do not stage the root gitlink dirtiness casually. Preserve the nested edits
  as forensic/local custody unless the public-intake owner extracts a patch
  manifest or curated reverse-engineering note.

Manual review:

- `audit_research_state_tracking.py` still reports four ignored `.omx/status`
  JSON files as uncategorized manual-review items:
  `kaggle-dilated-h64-long1000.json`, `kaggle-pairaware-smoke.json`,
  `kaggle-segnet-attack-fixed-h32.json`, and `modal-dilated-h64-long1000.json`.
  They are small but live under operator status state; review before tracking
  or summarize to a dated research ledger if they contain durable signal.
