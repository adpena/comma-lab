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

## 2026-05-08T11:24Z Custody Addendum

Adversarial review found that the first queued job did not prove custody of the
external PR104 runtime root. The adapter declares:

```text
PACT_RUNTIME_DEPENDENCY_ROOT=experiments/results/public_pr_intake_full/public_pr104_intake_20260505_auto/source/submissions/qhnerv_ft_best
```

The staged source manifest for
`pr104-public-exact-replay-g4dn2-20260508T111530Z` included only the archive,
adapter `inflate.sh`, and adapter README. Remote inspection confirmed that the
Studio copy of the dependency root contained `archive.zip` but not `inflate.py`
or `src/`. Therefore this job is now forensic only and must not produce a
promotion result unless a harvested `contest_auth_eval.adjudicated.json`
independently proves the external dependency-root custody expected by the
readiness ledger.

A stop request was submitted. `Lightning Job.stop()` initially timed out and
the SDK reported a nonterminal status regression from `Running` to `Pending`,
then a refresh at `2026-05-08T11:27:49Z` reported terminal status `Stopped`.
The lane claim was closed as `stopped_runtime_root_not_staged`; a replacement
may now be launched only through the hardened source-manifest path.

Hardening landed locally after this finding:

- `scripts/lightning_exact_eval_repro.py` now stages files under literal
  repo-relative `PACT_RUNTIME_DEPENDENCY_ROOT` directives.
- `scripts/launch_lightning_batch_job.py exact-eval` now independently requires
  those dependency-root files in the source manifest before non-dry-run Studio
  submit.
- Focused tests cover both plan staging and submit-time manifest rejection.

Patch verification generated a hardened PR104 plan with `25` artifacts,
including `inflate.py`, `src/model.py`, `src/codec.py`, and all non-cache files
under the declared runtime root.

## 2026-05-08T11:33Z Hardened Relaunch Addendum

Replacement job:
`pr104-public-exact-replay-rootstaged-g4dn2-20260508T1130Z`.

The relaunch used the hardened source-manifest path after commit `8418b029`.
The stage-only remote manifest check reported `REMOTE_MANIFEST_VERIFY: OK` with:

- Manifest path:
  `.omx/state/pr104-public-exact-replay-rootstaged-g4dn2-20260508T1130Z_manifest.json`.
- Source manifest SHA-256:
  `a284ea6c4977c532b5c912df0f95728ff5f974419515c2f7207514cd98bd7537`.
- Manifest file count: `2202`.
- Manifest total bytes: `36690115`.
- Declared dependency root:
  `experiments/results/public_pr_intake_full/public_pr104_intake_20260505_auto/source/submissions/qhnerv_ft_best`.

The submit plan contains `25` top-level artifacts and the exact-eval command
contains `--source-manifest`,
`--remote-preflight-ssh-target`, `--dispatch-lane-id`,
`--teamspace comma-lab`, and `--user adpena`. Queue metadata records both the
PR104 readiness ledger and source manifest.

Lightning refresh at `2026-05-08T11:33:21Z` reported the replacement job as
`Running` on SDK machine `T4`. Immediate guarded harvest still returned
`ARTIFACT_NOT_READY`, which remains `invalid` evidence and carries no score,
promotion, or method conclusion. The active lane claim is
`active_dispatching` until either an adjudicated JSON is harvested or a
terminal failure is recorded.

## 2026-05-08T11:36Z Snapshot Custody Check

Direct SSH inspection of the Lightning job snapshot confirmed that the
hardened relaunch fixed the first job's dependency-root custody failure. The
snapshot contains:

- `archive.zip`
- `inflate.py`
- `inflate.sh`
- `compress.sh`
- `README.md`
- `report.txt`
- `src/codec.py`
- `src/data.py`
- `src/losses.py`
- `src/model.py`
- `src/optim.py`
- `src/score.py`
- `src/train.py`

The snapshot archive hash and size matched the intended PR104 archive:

```text
6564c32a9edeeaf08abd7f0ea673ba2fda23444605ca207eb4ba794cc66797b8  archive.zip
178637 archive.zip
```

This is a custody guard result only. It proves the replacement job has the
runtime source closure that the stopped job lacked, but it still does not
produce score evidence until `contest_auth_eval.adjudicated.json` is harvested.

## 2026-05-08T11:48Z Harvested Exact CUDA Result

The hardened relaunch completed and was harvested over SSH into:

```text
experiments/results/lightning_batch/pr104-public-exact-replay-rootstaged-g4dn2-20260508T1130Z/
```

Structured result:

- Result JSON:
  `experiments/results/lightning_batch/pr104-public-exact-replay-rootstaged-g4dn2-20260508T1130Z/contest_auth_eval.adjudicated.json`.
- Adjudication provenance:
  `experiments/results/lightning_batch/pr104-public-exact-replay-rootstaged-g4dn2-20260508T1130Z/adjudication_provenance.json`.
- Eval provenance:
  `experiments/results/lightning_batch/pr104-public-exact-replay-rootstaged-g4dn2-20260508T1130Z/eval_provenance.json`.
- Archive SHA-256:
  `6564c32a9edeeaf08abd7f0ea673ba2fda23444605ca207eb4ba794cc66797b8`.
- Archive bytes: `178637`.
- Device: `cuda`.
- GPU: `Tesla T4`.
- Samples: `600`.
- PoseNet distance: `0.00017237`.
- SegNet distance: `0.0007067`.
- Recomputed score: `0.23113446620399658`.
- Score delta vs public PR104 CUDA-comment baseline `0.231145103318`:
  `-0.000010637114003425596`.
- Adjudication lane: `IN_PREDICTED_BAND`.
- Evidence grade: `A++ contest T4`.
- Promotion flag in adjudication provenance: `true`.
- Allowed use in adjudication provenance:
  `promotion_review`, `rank_frontier_candidate`.

Runtime custody:

- Runtime tree SHA-256:
  `40e47f8677abc9885179c848a4f44096aa024ac0fb1a937ba8d99e29b49147b9`.
- `eval_provenance.json` records the external dependency root as existing and
  byte/SHA enumerated under
  `experiments/results/public_pr_intake_full/public_pr104_intake_20260505_auto/source/submissions/qhnerv_ft_best`.
- The recorded root includes `compress.sh`, `inflate.py`, `inflate.sh`,
  `report.txt`, `src/{codec,data,losses,model,optim,score,train}.py`, and the
  staged training files under `src/stages/`.

Custody conclusion:

- The earlier first-job runtime-root staging failure is resolved for the
  rootstaged relaunch.
- PR104 exact replay is now a harvested structured exact CUDA replay, not just
  a public/comment or CPU result.
- This does not beat the current local exact frontier; it closes a public PR
  reconstruction gap and narrows the auth-eval drift investigation.
- Follow-up hardening remains desirable: bind `source_manifest_sha256` and the
  source-manifest byte/SHA closure explicitly into adjudication provenance, not
  only the queue metadata and eval runtime manifest.

The dispatch lane claim was closed as `completed_exact_cuda_adjudicated`.
