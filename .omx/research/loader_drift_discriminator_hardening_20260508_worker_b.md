# Loader Drift Discriminator Hardening - 2026-05-08

owner: Worker B
evidence_grade: diagnostic
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false

## Scope

Hardened `tools/probe_eval_loader_drift.py` as a diagnostic-only DALI/PyAV
2x2 discriminator. This ledger is not a result ledger and makes no
contest-CPU or contest-CUDA score claim.

## Contract Added

The probe now records four intended cells even when the local host cannot run
them:

- `CPU+AV`: PyAV/FFmpeg decode plus CPU PoseNet/SegNet forward shape.
- `CUDA+DALI`: DALI/NVDEC decode plus CUDA PoseNet/SegNet forward shape.
- `CUDA+AV/shared-input`: CUDA forward on PyAV decoded bytes.
- `CPU+DALI`: CPU forward on DALI decoded bytes.

Each cell records `score_claim=false`, `score_claim_valid=false`,
`promotion_eligible=false`, `rank_or_kill_eligible=false`,
`ready_for_exact_eval_dispatch=false`, and false contest CPU/CUDA axis claim
labels. Unsupported cells emit typed prerequisite reasons such as
`cuda_available`, `dali_available`, or `cuda_dali_runtime_available`.

## Diagnostic Use

The default path preserves the existing raw DALI-vs-PyAV RGB tensor comparison
before PoseNet/SegNet. The optional forward-cell path is diagnostic only: it
compares PoseNet/SegNet outputs on shared input tensors to separate
decoder/input-byte drift from forward/kernel drift. It still does not run
`inflate.sh`, `upstream/evaluate.py` scoring, or any promotable contest path.

## Local Verification

- `.venv/bin/python -m py_compile tools/probe_eval_loader_drift.py tests/test_probe_eval_loader_drift_matrix.py`
- `.venv/bin/python -m pytest tests/test_probe_eval_loader_drift_matrix.py src/tac/tests/test_probe_eval_loader_drift.py -q`
- `.venv/bin/python tools/probe_eval_loader_drift.py --video-limit 1 --max-batches 1 --json-out /tmp/probe_eval_loader_drift_smoke.json`

The local smoke intentionally returned exit code `2` on macOS because CUDA and
DALI are unavailable here. The emitted JSON still recorded all four cells and
kept the artifact diagnostic/non-promotable.

## Remaining Blockers

- A CUDA host with DALI/NVDEC is still required to fill the `CUDA+DALI` and
  `CPU+DALI` cells with real measurements.
- The optional forward-cell diagnostic should be run on the same CUDA host to
  populate shared-input PoseNet/SegNet comparisons.
- Any mechanism conclusion remains diagnostic until paired exact eval artifacts
  on identical archive/runtime custody exist separately.
