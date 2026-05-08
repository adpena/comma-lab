# PR104 Exact Replay Dispatch Status - 2026-05-08

Status: `QUEUED`, `NO_SCORE_CLAIM`.

This ledger records the Lightning dispatch handoff for PR104 `qhnerv_ft_best`.
It is a public-frontier completeness replay, not a score promotion.

## Dispatch

- Job: `pr104-public-exact-replay-g4dn2-20260508T111530Z`.
- Lane: `pr104_public_exact_replay_t4`.
- Platform: Lightning Studio, teamspace `comma-lab`, user `adpena`, studio
  `lossy-compression-challenge`.
- Machine request: `g4dn.2xlarge`; SDK-reported job machine: `T4`.
- Queued at UTC: `2026-05-08T11:16:11Z`.
- Link:
  `https://lightning.ai/adpena/comma-lab/studios/lossy-compression-challenge/app?app_id=jobs&job_name=pr104-public-exact-replay-g4dn2-20260508t111530z`.

## Custody

- Archive:
  `experiments/results/public_pr_intake_full/public_pr104_intake_20260505_auto/source/submissions/qhnerv_ft_best/archive.zip`.
- Archive bytes: `178637`.
- Archive SHA-256:
  `6564c32a9edeeaf08abd7f0ea673ba2fda23444605ca207eb4ba794cc66797b8`.
- Inflate adapter:
  `experiments/public_runtime_adapters/pr104_qhnerv_ft_best_adapter/inflate.sh`.
- Readiness ledger:
  `.omx/research/pr104_exact_replay_readiness_20260508_codex.md`.
- Staged source manifest:
  `.omx/state/pr104-public-exact-replay-g4dn2-20260508T111530Z_manifest.json`.
- Queue record:
  `.omx/state/pr104-public-exact-replay-g4dn2-20260508T111530Z_queue_record.json`.

## Submission Guards

- Local Lightning supply-chain scan: `OK`, strict, zero violations.
- Remote workspace staging before submit: `REMOTE_MANIFEST_VERIFY: OK`.
- Source manifest SHA-256:
  `9b5b1be738df84f241796f0694fa74a391de3452f5ad76d7f34dec6c839d7991`.
- Manifest file count: `2180`.
- Manifest total bytes: `36609496`.
- Submit plan included `--source-manifest`, `--remote-preflight-ssh-target`,
  `--dispatch-lane-id`, explicit `--teamspace comma-lab`, explicit
  `--user adpena`, and the T4 CUDA wheel pins.

## Current Evidence Boundary

First harvest immediately after submit returned `ARTIFACT_NOT_READY`, which is
expected while the job is pending or running. This is `invalid` evidence for
method conclusions and carries:

- `score_claim=false`
- `promotion_eligible=false`
- `score_source=none:artifact_not_ready`

The next valid action is to retry harvest after the Lightning artifact
directory exists, then record `contest_auth_eval.adjudicated.json` if present.
