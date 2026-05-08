# Modal Phase A1 NVDEC Preflight Failure — 2026-05-08

## Evidence

- Lane: `track1_phase_a1_score_gradient`
- Modal call: `fc-01KR4SBZ24K5DP3WXAV8M39H0R`
- Instance/job id: `track1_phase_a1_score_gradient_20260508T215443Z_modal`
- Local harvest summary:
  `experiments/results/track1_phase_a1_score_gradient_20260508T215443Z_modal/harvest_summary.json`
- Input custody:
  - PR101 archive: `178258` B,
    `b83bf3488625dbd73adeddff91712994197ab53098e578e91327a0c6e49efb3e`
  - PR101 source zip snapshot: `19137` B,
    `cf7853a09a08654daa5a6363eba0e36f2b5d2ac9060999f7b799d3d99f8a6a17`
  - Video: `37545489` B,
    `2611f5f3e186f3529777749f97bd4cce3a208d6b3559e137bd45d256980d2fa9`

## Result

The Modal T4 dispatch launched after billing readiness was restored, but failed
closed before training at `cuda_dali_nvdec_preflight`.

Observed preflight facts:

- `torch_cuda_available=true`
- GPU: `Tesla T4`
- driver: `580.95.05`
- DALI import: `nvidia-dali 1.52.0`
- failure: `scripts/probe_nvdec.sh` return code `3`
- classification: `DALI_BUILD`
- DALI error tail: `nvml error (999): A nvml internal driver error occurred`

This is not score evidence and not a candidate result. No archive was trained
or evaluated. The terminal claim row was appended as
`failed_modal_recovered`.

## Engineering disposition

Default exact-CUDA behavior remains fail-closed: if NVDEC/DALI fails, the
dispatcher still refuses to claim a CUDA score.

New opt-in path:

```bash
.venv/bin/python experiments/modal_phase_a1_score_gradient_pr101.py plan \
  --continue-after-nvdec-failure
```

or, for actual Modal dispatch after claiming the lane:

```bash
PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \
  experiments/modal_phase_a1_score_gradient_pr101.py \
  --continue-after-nvdec-failure
```

The opt-in fallback may continue with CUDA training and archive build when T4
compute is healthy but DALI/NVDEC exact-eval preflight fails. It writes
`score_claim=false`, `score_claim_valid=false`, and
`evidence_grade=[cuda-training-build-only]`, then blocks promotion on both
`contest_cuda_eval_pending_due_modal_dali_nvdec_preflight_failure` and
`contest_cpu_eval_pending`.

## Reactivation criteria

1. Re-run on a Modal GPU/container whose DALI video probe passes, or on a
   Lightning/Vast host with working `upstream/evaluate.py --device cuda`.
2. If the training/build fallback produces an archive, run paired exact CUDA
   and CPU evals before promotion or retirement.
3. If Modal T4 keeps producing `nvml error (999)`, isolate whether the failure
   is DALI's in-memory `fn.experimental.inputs.video` fixture path or the
   upstream `DaliVideoDataset` file-reader path before changing scorer policy.
