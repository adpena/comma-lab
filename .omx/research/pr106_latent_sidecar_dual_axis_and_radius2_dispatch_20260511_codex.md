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
- status at dispatch record time: pending

The prior CPU attempt failed before scoring because the inflate runtime
hard-required CUDA. That failure is classified as `runtime_cpu_inflate_blocker`,
not as a CPU score result and not as evidence CPU is worse.

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

1. Harvest CPU retry when Modal call `fc-01KRBT0389N3CCPN4P9BDTYGWR` completes.
2. Harvest Kaggle kernel outputs after `KernelWorkerStatus.COMPLETE`; ingest via
   `tools/harvest_kaggle_pr106_latent_score_table.py`.
3. Build radius-2 sidecar archive from harvested `score_table.npy`.
4. Dispatch exact CUDA auth eval for the radius-2 sidecar candidate only after
   byte-closed archive/runtime/no-op proof exists.
5. Continue polling T1 Modal call `fc-01KR955JSYQAVTTYZA48VAV7WJ`; do not
   duplicate the T1 job while that claim is active.
