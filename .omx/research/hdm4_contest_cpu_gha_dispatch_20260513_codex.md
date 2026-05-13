# HDM4 contest-CPU GHA dispatch - 2026-05-13

## Scope

Paired [contest-CPU GHA Linux x86_64] closure for the HDM4 release surface.
This is a free GitHub Actions dispatch, not a new model result yet. No score
claim is made until the workflow artifact is harvested and adversarially
reviewed.

## Dispatch

- Workflow: `contest_cpu_eval.yml`
- GitHub run: `25795440430`
- URL: `https://github.com/adpena/comma-lab/actions/runs/25795440430`
- Event: `workflow_dispatch`
- Head SHA: `d01a8cae772123906c89a07c3bdb9826cffbbb08`
- Created at: `2026-05-13T11:11:27Z`
- Initial status: `queued`
- Lane ID: `hnerv_hdm4_q_brotli_split_exact_eval`
- Claim job ID: `hdm4_release_surface_contest_cpu_gha_20260513T111109Z`
- Claim status: `active_gha_contest_cpu_eval`

Command:

```bash
gh workflow run contest_cpu_eval.yml \
  --ref main \
  -f archive_path=experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/exact_cuda_release_review_surface/archive.zip \
  -f lane_id=hnerv_hdm4_q_brotli_split_exact_eval \
  -f inflate_sh_path=experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/exact_cuda_release_review_surface/inflate.sh
```

## Inputs

- Archive path: `experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/exact_cuda_release_review_surface/archive.zip`
- Archive bytes: `186492`
- Archive sha256: `218ae16f3f13b722e9752d698667ed8770151e40d44b5756c0ebbccb7682825f`
- Inflate path: `experiments/results/pr106_r2_lowlevel_hdm4_archive_candidate_20260513_codex/exact_cuda_release_review_surface/inflate.sh`
- Inflate sha256: `bfcceb491c01a97c1f5ee46919abd3ce921040a1e0b7d431eb2fbe8184369fe4`

## Claim Discipline

`tools/claim_lane_dispatch.py` recorded an active lane claim before the workflow
was triggered. On harvest, close the same lane/job with a terminal
`completed_...` or `failed_...` row and preserve the artifact paths in this
ledger or a follow-up dated ledger.

## Harvest Checklist

When the run finishes:

1. Download the workflow artifact.
2. Verify `contest_auth_eval.json`, `eval_work/report.txt`, `eval_work/provenance.json`, and any posterior update.
3. Recompute the contest formula from component fields.
4. Confirm axis labels: `[contest-CPU GHA Linux x86_64]`, not Modal CPU and not macOS CPU.
5. Compare against the existing [contest-CUDA] HDM4 release-surface JSON without converting axes.
6. Close the lane claim with the exact workflow run ID and conclusion.
7. Update `.omx/research` with the harvested JSON path, archive SHA, runtime tree SHA, score, components, and blockers.
