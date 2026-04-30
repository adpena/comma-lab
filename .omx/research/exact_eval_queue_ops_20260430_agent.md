# Exact Eval Queue Ops - Lightning / Vast / Modal

Date: 2026-04-30
Agent: Codex
Scope: read-only operations audit plus queue plan for OWV3 byte-feasible
archive SHA `e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec`.

No spendful jobs were submitted. No code was edited. This note is an
operations ledger, not score evidence.

## Non-Negotiable Evidence Rule

The OWV3 byte-feasible archive is not ranked, promoted, or killed by this
audit. It has byte-only evidence until an exact CUDA auth eval exists for the
exact archive bytes through:

```bash
experiments/contest_auth_eval.py \
  --archive <archive.zip> \
  --inflate-sh submissions/robust_current/inflate.sh \
  --upstream-dir <upstream> \
  --device cuda \
  --keep-work-dir \
  --work-dir <evidence-dir>
```

The authoritative output must be `contest_auth_eval.json` with `n_samples=600`,
`provenance.device=cuda`, matching archive SHA/bytes, copied logs/provenance,
and component-gate adjudication.

## Target Archive Custody

Local target:

```text
experiments/results/lane_g_v3_owv3_byte_plan_sweep_worker_b/best_byte_feasible/archive_lane_g_v3_owv3.zip
```

Verified local identity:

```text
sha256=e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec
bytes=686557
```

The archive is the selected byte-feasible candidate from
`experiments/results/lane_g_v3_owv3_byte_plan_sweep_worker_b/byte_plan_summary.json`:

- candidate id: `owv3_0018_bbr0p69_protect0p0014_aggr1em05`
- byte delta vs PFP16 A++: `-78` bytes (`686557` vs `686635`)
- status: `byte_feasible_pending_cuda_auth_eval`
- evidence label: `byte-only-pending-cuda-auth-eval`
- decode verified: `true`
- member manifest:
  - `renderer.bin`, raw `292019`, compressed `259877`, SHA `f255ffc862285d4b4209e5d2b6b0493ff3e6a0475d23c0b5f7934d5b686f0769`
  - `masks.mkv`, raw `421483`, compressed `412169`, SHA `d3eeb82ce28b988476a920265751cca3d9fa2ca1364de4f33a1c7e970b7895e9`
  - `optimized_poses.pt`, raw `15620`, compressed `14183`, SHA `cb8517f7a7e3c9382e952ff278dc3f8de44ba066db07746f16354c1dbe2cbca4`

Commands run:

```bash
rg -n "e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec" .omx experiments reports submissions scripts src
find experiments .omx submissions -name '*.zip' -type f -print0 | xargs -0 shasum -a 256 | rg '^e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec'
jq '{summary: .summary, selected: .selected, best_byte_feasible: .best_byte_feasible, candidates: (.candidates|length? // null), keys: keys}' experiments/results/lane_g_v3_owv3_byte_plan_sweep_worker_b/byte_plan_summary.json
```

## Lightning State

Local Lightning Batch Jobs state is dry-run only. The latest OWV3 entry is:

```text
status=DRY_RUN
recorded_at_utc=2026-04-30T19:10:28Z
job_name=owv3_byte_feasible_exact_cuda_20260430_codex_dryrun
role=exact_cuda_eval
machine=T4
expected_archive_sha256=e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec
expected_archive_size_bytes=686557
command_sha256=e8551610ddb813ae6d0ee4857c3f110a22affa201ce64333624709bbeee15e89
local_artifact_dir=experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex
```

No actual Lightning Batch Job record, job id, artifact path, or harvested local
artifact directory exists for this OWV3 eval. `experiments/results/lightning_batch`
is absent locally.

The dry-run command is structurally right for exact eval: it writes queue
metadata, copies fixed bytes, verifies SHA/bytes, runs
`experiments/contest_auth_eval.py --device cuda`, asserts JSON fields, and runs
`scripts/adjudicate_contest_auth_eval.py` with PFP16 A++ component references.

Important caveat: the latest dry-run record has `spec.studio=null` even though
the command assumes `/teamspace/studios/this_studio/...`. The runbook template
uses `--studio pact`. Before spend, submit a corrected dry-run or submit with
`--studio pact` explicitly unless the operator has confirmed the SDK default
maps to the same Studio filesystem.

Commands run:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py list
jq -r '.[] | {status, recorded_at_utc, dry_run, name:(.spec.name // .queue.job_name), role:(.spec.role // .queue.role), machine:(.spec.machine // .queue.machine), queued_at:(.queue.queued_at_utc // null), expected_archive_sha256:(.spec.expected_archive_sha256 // .queue.expected_archive_sha256 // null), expected_archive_size_bytes:(.spec.expected_archive_size_bytes // .queue.expected_archive_size_bytes // null), command_sha256:(.queue.command_sha256 // null), local_artifact_dir:(.queue.local_artifact_dir // .spec.local_artifact_dir // null), queue_metadata:(.spec.queue_metadata // .queue.queue_metadata // null)}' .omx/state/lightning_batch_jobs.json
```

## Lightning Runtime Checks

Local runtime is not CUDA-visible:

```text
nvidia-smi: command not found
torch_version=2.11.0
cuda_available=false
cuda_device_count=0
```

Read-only SSH to the Lightning Studio target from
`.omx/state/owv3_byte_feasible_repro_20260430_r1_manifest.json` succeeded:

```text
remote=s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai
host=ip-10-192-11-229
repo exists=/teamspace/studios/this_studio/pact
upstream evaluate exists=/teamspace/studios/this_studio/upstream/evaluate.py
upstream video exists=/teamspace/studios/this_studio/upstream/videos/0.mkv
target archive exists on Lightning Studio
pfp16 exact venv exists
nvidia-smi=missing
torch_version=2.11.0+cu130
cuda_available=false
cuda_device_count=0
```

Remote target archive identity on Lightning Studio was verified:

```text
remote_archive=experiments/results/lane_g_v3_owv3_byte_plan_sweep_worker_b/best_byte_feasible/archive_lane_g_v3_owv3.zip
sha256=e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec
bytes=686557
```

Commands run:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=20 \
  s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai '<read-only path/GPU/torch probe>'

ssh -o BatchMode=yes -o ConnectTimeout=20 \
  s_01knw7wnzbe79wfq5mqqbx1mbz@ssh.lightning.ai '<remote supply-chain scan and archive hash probe>'
```

Conclusion: the current interactive Studio shell cannot run exact CUDA eval.
The fastest Lightning route is an official T4 Batch Job, not SSH eval from the
current shell.

## Supply-Chain Checks

Current local strict scan:

```text
status=OK
violation_count=0
python=/Users/adpena/Projects/pact/.venv/bin/python
lightning=null
pytorch-lightning=null
lightning-sdk=2026.4.10
lightning_sdk=2026.4.10
```

Current remote Lightning strict scan:

```text
status=OK
violation_count=0
python=/teamspace/studios/this_studio/pact_pfp16_exact_20260430T1625Z/.venv/bin/python
lightning=null
pytorch-lightning=null
lightning-sdk=null
lightning_sdk=null
```

Prior state scans also report `status=OK`:

```text
.omx/state/lightning_supply_chain_scan_20260430_codex.json
.omx/state/lightning_supply_chain_scan_20260430_codex_r2.json
```

Commands run:

```bash
.venv/bin/python scripts/scan_lightning_supply_chain.py --strict --quiet
ssh ... 'cd /teamspace/studios/this_studio/pact && /teamspace/studios/this_studio/pact_pfp16_exact_20260430T1625Z/.venv/bin/python scripts/scan_lightning_supply_chain.py --strict --quiet'
```

## Vast State

Live Vast inventory currently has four running RTX 4090 instances:

```text
35885106 lane_hm_s_2026-04-30_b_a2              running RTX 4090 ssh8.vast.ai:15106 gpu_util=0%
35899850 lane_19_logit_margin_2026-04-30_b_a4   running RTX 4090 ssh2.vast.ai:19850 gpu_util=55%
35906669 lane_sa_segmap_clone_2026-04-30_codex_a2 running RTX 4090 ssh2.vast.ai:26668 gpu_util=46%
35907873 lane_h_v3_joint_halfframe_2026-04-30_codex_a4 running RTX 4090 ssh5.vast.ai:27872 gpu_util=79%
```

`scripts/reconcile_vast_dispatch_state.py --json --max-items 50` reports:

```text
live_count=4
active_dispatch_count=3
tracker_count=204
active_missing_live=35899435,35899552,35899275
live_missing_active=35885106,35906669,35907873
live_missing_tracker=[]
```

Read-only SSH probes found no lane-local OWV3 exact-eval opportunity and no
new claim-ready evals:

- HM-S `35885106`: GPU idle, no `remote_lane` process, no lane-local archive or
  `contest_auth_eval.json`; lane directory contains `segmap_weights.tar.xz`,
  `train/segmap_inference.pt`, and logs ending at Stage 3 pack. It appears
  stopped or failed before archive/eval.
- Lane 19 `35899850`: still training at about epoch `1100/1980`, no lane-local
  archive/eval JSON.
- SA clone `35906669`: still training at about epoch `468/600`, no lane-local
  archive/eval JSON.
- H-V3 `35907873`: still training at about epoch `1150/1980`, no lane-local
  archive/eval JSON.

Commands run:

```bash
.venv/bin/vastai show instances --raw
.venv/bin/python scripts/reconcile_vast_dispatch_state.py --json --max-items 50
ssh -p <port> root@<vast-host> '<nvidia-smi, process, lane-dir, and artifact probes>'
```

Conclusion: Vast can provide CUDA-visible RTX 4090 eval evidence, but it is not
the fastest safe path for this target right now. Existing live machines are
running other lanes or have a stalled/failed lane directory. Reusing them would
risk interfering with active work and still would not be T4 contest-equivalent
evidence. A new Vast exact-eval-only run would be spendful and lower custody
than the prepared Lightning Batch Job path.

## Modal State

Modal is not a safe promotion path with the current helper scripts:

- `experiments/modal_auth_eval.py` advertises T4 auth eval, but invokes
  upstream `evaluate.py` with `--device cpu`.
- `experiments/modal_train_lane.py` explicitly sets `AUTH_EVAL_DEVICE=cpu`.
- Existing Modal harvests are advisory unless rerun through canonical CUDA
  exact eval with complete archive custody.

Commands inspected:

```bash
sed -n '1,260p' experiments/modal_auth_eval.py
rg -n "AUTH_EVAL_DEVICE|device cpu|--device cpu|--device cuda|gpu=|A10G|T4|modal" \
  experiments/modal_train_lane.py experiments/modal_recover_lane.py experiments/modal_auth_eval.py \
  scripts/modal_check.py scripts/remote_auth_eval_only.sh scripts/lightning_auth_eval.sh \
  scripts/pfp16_a_plus_plus_exact_t4_eval.sh
```

Conclusion: do not use Modal for the OWV3 score claim unless a new wrapper is
made to run `contest_auth_eval.py --device cuda` and preserve JSON/archive
custody. No such spend was submitted.

## Fastest Safe Queue Plan

Fastest safe path: Lightning Batch Job on T4 with the exact existing OWV3
archive and strict adjudication. This is faster and cleaner than waiting for
Vast lanes or writing a new Modal CUDA wrapper.

Recommended pre-spend dry-run correction:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py exact-eval \
  --dry-run \
  --job-name owv3_byte_feasible_exact_cuda_20260430_codex_dryrun2 \
  --archive /teamspace/studios/this_studio/pact/experiments/results/lane_g_v3_owv3_byte_plan_sweep_worker_b/best_byte_feasible/archive_lane_g_v3_owv3.zip \
  --repo-dir /teamspace/studios/this_studio/pact \
  --upstream-dir /teamspace/studios/this_studio/upstream \
  --output-dir /teamspace/jobs/owv3_byte_feasible_exact_cuda_20260430_codex/artifacts \
  --machine T4 \
  --studio pact \
  --python-bin /teamspace/studios/this_studio/pact_pfp16_exact_20260430T1625Z/.venv/bin/python \
  --expected-archive-sha256 e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec \
  --expected-archive-size-bytes 686557 \
  --queue-metadata lane=owv3_byte_feasible \
  --queue-metadata source=codex_20260430 \
  --local-artifact-dir experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex \
  --adjudicate \
  --baseline-score 1.043987524793892 \
  --baseline-archive-bytes 686635 \
  --predicted-band 1.0438 1.0442 \
  --regression-threshold 1.04425 \
  --component-reference-label pfp16_a_plus_plus_t4 \
  --baseline-posenet-dist 0.00346442 \
  --baseline-segnet-dist 0.00400656 \
  --max-posenet-relative 1.002 \
  --max-segnet-relative 1.002
```

Spendful submit command is the same command with `--dry-run` removed and a
non-dry-run job name, after explicit spend authorization:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py exact-eval \
  --job-name owv3_byte_feasible_exact_cuda_20260430_codex_$(date -u +%Y%m%dT%H%M%SZ) \
  --archive /teamspace/studios/this_studio/pact/experiments/results/lane_g_v3_owv3_byte_plan_sweep_worker_b/best_byte_feasible/archive_lane_g_v3_owv3.zip \
  --repo-dir /teamspace/studios/this_studio/pact \
  --upstream-dir /teamspace/studios/this_studio/upstream \
  --output-dir /teamspace/jobs/owv3_byte_feasible_exact_cuda_20260430_codex/artifacts \
  --machine T4 \
  --studio pact \
  --python-bin /teamspace/studios/this_studio/pact_pfp16_exact_20260430T1625Z/.venv/bin/python \
  --expected-archive-sha256 e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec \
  --expected-archive-size-bytes 686557 \
  --queue-metadata lane=owv3_byte_feasible \
  --queue-metadata source=codex_20260430 \
  --local-artifact-dir experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex \
  --adjudicate \
  --baseline-score 1.043987524793892 \
  --baseline-archive-bytes 686635 \
  --predicted-band 1.0438 1.0442 \
  --regression-threshold 1.04425 \
  --component-reference-label pfp16_a_plus_plus_t4 \
  --baseline-posenet-dist 0.00346442 \
  --baseline-segnet-dist 0.00400656 \
  --max-posenet-relative 1.002 \
  --max-segnet-relative 1.002
```

Expected artifacts from a completed job:

```text
/teamspace/jobs/owv3_byte_feasible_exact_cuda_20260430_codex/artifacts/archive.zip
/teamspace/jobs/owv3_byte_feasible_exact_cuda_20260430_codex/artifacts/lightning_queue_metadata.json
/teamspace/jobs/owv3_byte_feasible_exact_cuda_20260430_codex/artifacts/auth_eval.log
/teamspace/jobs/owv3_byte_feasible_exact_cuda_20260430_codex/artifacts/contest_auth_eval.json
/teamspace/jobs/owv3_byte_feasible_exact_cuda_20260430_codex/artifacts/eval_provenance.json
/teamspace/jobs/owv3_byte_feasible_exact_cuda_20260430_codex/artifacts/report.txt
/teamspace/jobs/owv3_byte_feasible_exact_cuda_20260430_codex/artifacts/adjudication.log
/teamspace/jobs/owv3_byte_feasible_exact_cuda_20260430_codex/artifacts/adjudication_provenance.json
/teamspace/jobs/owv3_byte_feasible_exact_cuda_20260430_codex/artifacts/contest_auth_eval.adjudicated.json
```

Harvest validation command:

```bash
.venv/bin/python scripts/launch_lightning_batch_job.py validate-artifacts \
  --artifact-dir experiments/results/lightning_batch/owv3_byte_feasible_exact_cuda_20260430_codex \
  --expected-archive-sha256 e1deda126d8623ef9ab6acb03f708832df845bd7ab00d60c66e113f4948cf0ec \
  --expected-archive-size-bytes 686557 \
  --require-adjudication
```

## Blockers / Watch Items

1. Current interactive Lightning Studio shell has no `nvidia-smi` and no CUDA
   visible to torch. Do not run SSH exact eval from that shell.
2. The existing OWV3 Lightning Batch Jobs entry is dry-run only. There is no
   submitted job id and no artifact directory to harvest.
3. The latest dry-run state omitted `--studio pact`; use the corrected command
   above or confirm the SDK default before spend.
4. The staging manifest is reproducible source/artifact custody, but it used
   `requirements_mode=no-install` and `require_cuda=false`; runtime proof must
   come from the exact eval JSON on the Batch T4 worker.
5. Modal current helpers are CPU/advisory; do not use them for OWV3 promotion.
6. Vast live state is drifting from `active_dispatches.md`; live API and
   lane-local artifacts should win over stale rows. No current Vast lane has a
   new lane-local exact eval JSON.

## Next Safe Action

Do not dispatch Modal. Do not reuse live Vast instances for OWV3 unless an
operator explicitly pauses/repurposes one and accepts RTX 4090 Grade A rather
than T4/A++ evidence.

The next safe action is a corrected Lightning Batch dry-run with `--studio
pact`; after explicit spend authorization, submit the same spec without
`--dry-run`. Treat the result as evidence only after
`contest_auth_eval.json` and adjudication provenance validate the exact SHA,
bytes, CUDA device, full sample count, and component gates.
