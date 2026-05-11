# PR106 latent sidecar dual-axis + radius-2 dispatch record (2026-05-11)

## Context

Current exact-CUDA floor is PR106 latent sidecar:

- archive: `experiments/results/pr106_latent_sidecar_from_kaggle_table_20260511_codex/sidecar_archive.zip`
- archive bytes: `186808`
- archive sha256: `947b85e8a69db295d4dcf80b0b528639c47839f40f289a2c05b70a2064658b48`
- exact-CUDA score: `0.20739428085403283`
- adjudicated artifact: `experiments/results/modal_auth_eval/pr106_latent_sidecar_20260511T150517Z/contest_auth_eval.adjudicated.json`

This supersedes PR101/A1/PR103 as the CUDA routing anchor. A1 remains a
separate CPU-axis/public-frontier comparison point; it is not the CUDA floor.

## Runtime fix

Commit `f3734e26` changed `submissions/pr106_latent_sidecar/inflate.py` from a
CUDA-only inflate runtime to a contest-axis-safe runtime:

- `PACT_INFLATE_DEVICE=auto` selects CUDA when present, otherwise CPU.
- `PACT_INFLATE_DEVICE=cpu` explicitly selects CPU.
- `PACT_INFLATE_DEVICE=cuda` fails closed if CUDA is unavailable.
- `PACT_INFLATE_DEVICE=mps` / `metal` is rejected for auth-eval custody.
- `PACT_INFLATE_BATCH_PAIRS` provides explicit deterministic batch tuning.

Verification before dispatch:

- `.venv/bin/python -m pytest src/tac/tests/test_pr106_latent_sidecar.py -q`:
  `24 passed`
- `.venv/bin/python -m ruff check submissions/pr106_latent_sidecar/inflate.py src/tac/tests/test_pr106_latent_sidecar.py`:
  pass
- `.venv/bin/python tools/all_lanes_preflight.py --timings --timeout-s 30`:
  `ALL 29 PREFLIGHT CHECKS PASSED`, wall about `2.30s`

## CPU-axis retry

Paired Linux x86_64 contest-CPU retry was dispatched after the runtime fix:

- lane id: `lane_pr106_latent_sidecar_contest_cpu`
- job id: `pr106_latent_sidecar_modal_contest_cpu_retry_cpuinflate_20260511T151955Z`
- Modal call id: `fc-01KRBT0389N3CCPN4P9BDTYGWR`
- output dir: `experiments/results/modal_auth_eval_cpu/pr106_latent_sidecar_20260511T151955Z`
- status: recovered successfully
- contest-CPU score: `0.2286802845175232`
- CPU components: `avg_segnet_dist=0.00063766`, `avg_posenet_dist=0.00016424`
- CPU inflate elapsed: `58.1s`
- CPU evaluate elapsed: `174.9s`
- CPU inflated raw aggregate sha256:
  `936d9c568d7adcf9b0da76c25531b668f8ad94bc1d64037ca2f583123318c7aa`

The prior CPU attempt failed before scoring because the inflate runtime
hard-required CUDA. That failure is classified as `runtime_cpu_inflate_blocker`,
not as a CPU score result and not as evidence CPU is worse.

Recovered result: CPU is worse than CUDA for this packet, but the reason is not
monotone "CPU bad". CPU slightly improves the seg term relative to CUDA
(`0.063766` vs about `0.064893`) while the pose term worsens sharply
(`0.0405265` vs about `0.01811`). Device-axis behavior is therefore
submission-specific and component-specific. Do not generalize from A1's CPU
advantage to PR106-derived packets.

## Radius-2 score-table dispatch

Fresh-eyes audit noted that the radius-1 sidecar selected 600/600 corrections
at the `+-1` boundary, indicating clipped search. The next score-lowering
action is a radius-2 latent candidate score table.

Claim:

- lane id: `lane_pr106_latent_sidecar`
- platform: `kaggle`
- job id: `kaggle_pr106_latent_score_table_r2_20260511T151955Z`
- status: `active_dispatching`

Kaggle artifacts:

- source dataset: `adpena/comma-lab-pr106-latent-source`
- source dataset version message:
  `PR106 latent radius-2 source bundle 20260511`
- kernel: `adpena/comma-lab-pr106-latent-score-table`
- kernel version: `2`
- status immediately after push: `KernelWorkerStatus.RUNNING`

The Kaggle run is a CUDA score-table producer only. It is `score_claim=false`
until harvested, reduced into a byte-closed sidecar archive, and adjudicated by
exact contest-CUDA auth eval.

## Immediate follow-up

Supersession note (`2026-05-11T16:40Z`): steps 1-4 below are historical.
The CPU retry and R2 harvest/materialization/exact-T4 adjudication later
completed in this same ledger. The current measured PR106-sidecar
`[contest-CUDA]` frontier / constructive upper bound is
`0.20664588545741508` at `186822` bytes, archive SHA-256
`7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f`.
Continue to treat this as a measured frontier, not a certified lower-bound
floor, until strict compliance and any scoped lower-bound certificate land.

1. Harvest CPU retry when Modal call `fc-01KRBT0389N3CCPN4P9BDTYGWR` completes.
2. Harvest Kaggle kernel outputs after `KernelWorkerStatus.COMPLETE`; ingest via
   `tools/harvest_kaggle_pr106_latent_score_table.py`.
3. Build radius-2 sidecar archive from harvested `score_table.npy`.
4. Dispatch exact CUDA auth eval for the radius-2 sidecar candidate only after
   byte-closed archive/runtime/no-op proof exists.
5. Continue polling T1 Modal call `fc-01KR955JSYQAVTTYZA48VAV7WJ`; do not
   duplicate the T1 job while that claim is active.

## Yshift retry note

The pre-existing Kaggle yshift kernel ERROR was inspected on 2026-05-11. Its
log shows a packaging failure, not a method failure:

```text
FileNotFoundError: required source bundle 'pact_pr106_yshift_source_bundle.tar.gz' not found under ['/kaggle/src', '/kaggle/input']
```

Corrective action:

- lane id: `lane_pr106_yshift_score_table`
- platform: `kaggle`
- job id: `kaggle_pr106_yshift_score_table_retry_20260511T1526Z`
- source dataset: `adpena/comma-lab-pr106-yshift-source`
- source dataset version message:
  `PR106 yshift source bundle retry 20260511`
- kernel: `adpena/comma-lab-pr106-yshift-score-table`
- kernel version: `3`
- status immediately after push: `KernelWorkerStatus.RUNNING`

This remains `score_claim=false`. The yshift score table is only a
compress-time profiler until harvested, compiled into a runtime-consumed
sidechannel archive, and adjudicated by exact contest-CUDA auth eval.

## Yshift retry classification: stale embedded claim snapshot

The retry kernel later returned `KernelWorkerStatus.ERROR`. Harvested log:

- log: `reports/raw/kaggle_logs/pr106_yshift_retry_20260511T1526/comma-lab-pr106-yshift-score-table.log`
- failure:

```text
ValueError: no active lane claim found for lane_id=lane_pr106_yshift_score_table instance_job_id=kaggle_pr106_yshift_score_table_retry_20260511T1526Z
```

Classification: `failed_kaggle_embedded_claim_snapshot_stale_no_score_claim`.
This is a packaging/custody bug, not a yshift method negative. The locally built
source tar contained the correct retry claim, but Kaggle executed against a
stale mounted source dataset snapshot whose embedded
`.omx/state/active_lane_dispatch_claims.md` did not include the retry job row.

Hardening fix:

- `src/tac/deploy/kaggle/pr106_yshift_score_table.py` now inlines the
  deterministic source bundle into the kernel package itself.
- `src/tac/deploy/kaggle/pr106_latent_score_table.py` received the same
  source-bundle inlining so future latent retries do not depend solely on
  Kaggle dataset-version freshness.
- The source dataset remains as a fallback mount, but the launcher searches its
  own kernel directory first, so a pushed kernel and its embedded claim snapshot
  are atomically paired.
- This is still `score_claim=false`; a future successful table must be reduced
  into charged bytes and exact-CUDA adjudicated before any score claim.

Verification:

- `.venv/bin/python -m pytest src/tac/tests/test_kaggle_pr106_yshift_score_table.py src/tac/tests/test_kaggle_pr106_latent_score_table.py src/tac/tests/test_materialize_pr106_latent_score_table_candidate.py -q`:
  `15 passed`
- `.venv/bin/python -m ruff check src/tac/deploy/kaggle/pr106_yshift_score_table.py src/tac/deploy/kaggle/pr106_latent_score_table.py src/tac/tests/test_kaggle_pr106_yshift_score_table.py src/tac/tests/test_kaggle_pr106_latent_score_table.py tools/materialize_pr106_latent_score_table_candidate.py src/tac/tests/test_materialize_pr106_latent_score_table_candidate.py`:
  pass

Retry after hardening:

- terminal old job:
  `failed_kaggle_embedded_claim_snapshot_stale_no_score_claim`
- new job id:
  `kaggle_pr106_yshift_score_table_inline_20260511T154114Z`
- kernel version: `4`
- kernel URL:
  `https://www.kaggle.com/code/adpena/comma-lab-pr106-yshift-score-table`
- source dataset version message:
  `PR106 yshift inline source bundle retry 20260511`
- pushed status: `KernelWorkerStatus.RUNNING`

This retry remains a CUDA table producer only. It is not a score claim and is
not promotion-eligible until it emits a table, the table is materialized into a
charged yshift archive, and exact contest-CUDA auth eval adjudicates that
archive.

## Yshift inline retry classification + dataset-ready retry

The inline retry also failed before scorer work. Log recovered outside the repo
raw tree:

- temp log:
  `/tmp/pr106_yshift_inline_logs/comma-lab-pr106-yshift-score-table.log`
- failure:

```text
ValueError: no active lane claim found for lane_id=lane_pr106_yshift_score_table instance_job_id=kaggle_pr106_yshift_score_table_inline_20260511T154114Z
```

The runtime workspace still contained a stale source dataset snapshot whose
embedded claim ledger ended at
`kaggle_pr106_yshift_score_table_retry_20260511T1526Z`. The extra kernel
directory tarball was not used by the Kaggle script runtime. Classification:
`failed_kaggle_source_dataset_snapshot_stale_no_score_claim`.

Corrective action was to retry only after source-dataset readiness:

- terminal inline job:
  `failed_kaggle_source_dataset_snapshot_stale_no_score_claim`
- new job id:
  `kaggle_pr106_yshift_score_table_datasetready_20260511T154910Z`
- source dataset version message:
  `PR106 yshift dataset-ready source bundle 20260511`
- dataset readiness checked with:
  `uv tool run kaggle datasets status adpena/comma-lab-pr106-yshift-source`
  returning `ready`
- kernel version: `5`
- pushed status: `KernelWorkerStatus.RUNNING`

This is still `score_claim=false` and only a CUDA table producer.

## Materialization tools

Both active PR106 score-table producers now have scorer-free local materializers:

- `tools/materialize_pr106_latent_score_table_candidate.py`
  - consumes completed latent `score_table.npy` +
    `score_table_manifest.json`
  - calls `experiments/build_pr106_latent_sidecar.py --search-mode score_table`
  - emits `sidecar_archive.zip` + `materialization_manifest.json`
- `tools/materialize_pr106_yshift_score_table_candidate.py`
  - consumes completed yshift `score_table.npy` +
    `score_table_manifest.json`
  - calls
    `experiments/build_pr106_yshift_sidechannel.py --search-mode score_table`
  - emits `pr106_yshift_sidechannel_archive.zip` +
    `materialization_manifest.json`

Both tools enforce:

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- builder metadata must validate the score-table manifest
- exact contest-CUDA adjudication remains the promotion gate

Focused verification:

- `.venv/bin/python -m pytest src/tac/tests/test_materialize_pr106_yshift_score_table_candidate.py src/tac/tests/test_materialize_pr106_latent_score_table_candidate.py -q`:
  `8 passed`
- `.venv/bin/python -m ruff check tools/materialize_pr106_yshift_score_table_candidate.py src/tac/tests/test_materialize_pr106_yshift_score_table_candidate.py`:
  pass

## Latent radius-2 Kaggle harvest

The radius-2 latent score-table producer completed and was harvested locally.
This is high-signal but still non-promotable diagnostic evidence until exact
contest-equivalent CUDA adjudication lands.

- completed job:
  `kaggle_pr106_latent_score_table_r2_20260511T151955Z`
- kernel:
  `https://www.kaggle.com/code/adpena/comma-lab-pr106-latent-score-table`
- score table:
  `experiments/results/kaggle_pr106_latent_score_table_r2_20260511T151955Z/pr106_latent_score_table/latent_run/score_table/score_table.npy`
- score-table SHA-256:
  `326038a5e2dbb9daf8498e7e8c07fdfc15d832b443b7ad1758cab4f0517001fe`
- materialized archive:
  `experiments/results/pr106_latent_sidecar_r2_from_kaggle_table_20260511_codex/sidecar_archive.zip`
- materialized archive bytes:
  `186822`
- materialized archive SHA-256:
  `7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f`
- Kaggle P100 diagnostic CUDA canonical score:
  `0.2066238854574151`
- score axis:
  `diagnostic_cuda`
- evidence grade:
  `B`
- promotion:
  `false`

Adversarial reading: this beats the prior PR106 latent-sidecar exact T4 score
numerically, but the run was Kaggle P100 with non-T4 diagnostic metadata and is
not a score claim. The correct next step is Modal T4 exact CUDA adjudication on
the local materialized archive, not leaderboard/paper promotion from the Kaggle
log.

## Latent radius-2 exact T4 CUDA adjudication

Modal T4 exact CUDA adjudication completed for the same byte-closed archive.

- job id:
  `pr106_latent_sidecar_r2_20260511T160358Z`
- Modal call id:
  `fc-01KRBWG7KGBX0A26NZZF9CNJXE`
- artifact directory:
  `experiments/results/modal_auth_eval/pr106_latent_sidecar_r2_20260511T160358Z`
- result JSON:
  `experiments/results/modal_auth_eval/pr106_latent_sidecar_r2_20260511T160358Z/contest_auth_eval.json`
- archive bytes:
  `186822`
- archive SHA-256:
  `7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f`
- device:
  `cuda`
- GPU:
  `Tesla T4`
- samples:
  `600`
- PoseNet distortion:
  `0.00003236`
- SegNet distortion:
  `0.00064260`
- canonical score:
  `0.20664588545741508`
- evidence grade:
  `contest-CUDA`

Delta versus prior exact PR106 latent sidecar:

- prior exact T4 score:
  `0.20739428085403283`
- new exact T4 score:
  `0.20664588545741508`
- score delta:
  `-0.00074839539661775`

Delta versus Kaggle P100 diagnostic:

- Kaggle P100 diagnostic:
  `0.2066238854574151`
- Modal T4 exact:
  `0.20664588545741508`
- T4 minus P100:
  `+0.00002199999999996649`

Classification: legitimate score-lowering exact CUDA result. The P100 signal
transferred to T4, but with a measurable SegNet-axis drift; keep axis labels in
all future sidecar score-table comparisons.

## Supersession note - 2026-05-11 PR106 R2 exact T4

This ledger's final exact T4 section supersedes any earlier sections here or in
sibling 2026-05-11 ledgers that call PR106 latent sidecar
`0.20739428085403283` the current floor, treat radius-2 exact adjudication as
pending, or promote Kaggle P100 `0.2066238854574151` beyond diagnostic status.

Current internal PR106-family measured frontier / constructive upper bound is
the byte-closed PR106 latent radius-2 sidecar exact Modal T4 `[contest-CUDA]`
packet:

- canonical score: `0.20664588545741508`
- archive bytes: `186822`
- archive SHA-256:
  `7f926bc3e213af1c3ea4be0608c63d041d455eb6b988562b64465e81b25f3a3f`
- job id: `pr106_latent_sidecar_r2_20260511T160358Z`
- result JSON:
  `experiments/results/modal_auth_eval/pr106_latent_sidecar_r2_20260511T160358Z/contest_auth_eval.json`

Future floor/frontier, Pareto, and spend-gate language should use this exact
T4 result as the comparison baseline unless a newer `[contest-CUDA]` artifact
supersedes it.
