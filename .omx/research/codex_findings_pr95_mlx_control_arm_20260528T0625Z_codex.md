# Codex Findings - PR95/HNeRV MLX Control Arm - 2026-05-28T06:25Z

## Verdict

The PR95/HNeRV MLX control arm is executable beyond scaffold level. Stage 1,
stage 5, and stage 8 all produced byte-closed PR95 archive exports, public
runtime-consumption proofs, and PyTorch export forward-parity proofs on the
MLX CPU parity axis. The MLX GPU/Metal path remains useful for local training
throughput, but its Conv2d accumulation diverges enough from PyTorch that it
must be tracked as a separate calibration/drift signal, not as the archive
export parity gate.

## Artifacts

- Stage 1 CPU export parity + runtime proof:
  `.omx/research/pr95_mlx_stage1_archive_cpuparity_smoke_20260528Tlocal/run_summary.json`
  - seconds/step: `0.07612340625018987`
  - CPU export forward parity: pass, max_abs `3.0517578125e-05`, mean_abs `3.236281372664962e-06`
  - public runtime consumption: pass, raw bytes `48832128`
- Stage 5 CPU export parity + runtime proof:
  `.omx/research/pr95_mlx_stage5_archive_cpuparity_smoke_20260528Tlocal/run_summary.json`
  - seconds/step: `0.06913557300140383`
  - CPU export forward parity: pass, max_abs `3.0517578125e-05`, mean_abs `3.393351335034822e-06`
  - public runtime consumption: pass, raw bytes `48832128`
- Stage 8 Muon/AdamW CPU export parity + runtime proof:
  `.omx/research/pr95_mlx_stage8_archive_cpuparity_smoke_20260528Tlocal/run_summary.json`
  - seconds/step: `0.0765766664990224`
  - CPU export forward parity: pass, max_abs `3.0517578125e-05`, mean_abs `3.346785206304048e-06`
  - public runtime consumption: pass, raw bytes `48832128`
- Stage 8 combined CPU parity + MLX GPU drift attestation:
  `.omx/research/pr95_mlx_stage8_cpuparity_gpu_drift_combined_20260528Tlocal/run_summary.json`
  - CPU export forward parity: pass, max_abs `3.0517578125e-05`, mean_abs `3.346785206304048e-06`
  - MLX GPU drift attestation: fail/pass flag false by design, max_abs `0.0046539306640625`, mean_abs `0.0005879671080037951`
- Per-op drift probes:
  - CPU: `.omx/research/pr95_mlx_per_op_drift_cpu_20260528Tlocal/report.json`
  - GPU: `.omx/research/pr95_mlx_per_op_drift_gpu_20260528Tlocal/report.json`
  - CPU stays within attested bands; GPU drift localizes to `conv2d_3x3_pad1`
    and downstream `hnerv_decoder_full`.
- Bounded long-training smoke:
  `.omx/research/pr95_mlx_long_training_smoke_20260528Tlocal/report.json`
  - executed smoke: true
  - checkpoint root: `/Volumes/VertigoDataTier/experiments/results/pr95_mlx_long_training_smoke_20260528Tlocal/checkpoints`
  - readiness remains false until scorer-loss/QAT/export/full-frame/exact-auth gates close.

## Code Change

`tools/run_pr95_mlx_timing_smoke.py` now supports
`--write-mlx-gpu-drift-attestation`. This keeps the archive-export proof axis
clean:

- `--write-pytorch-export-parity --pytorch-export-mlx-device cpu` proves the
  exported archive checkpoint is PyTorch-consumable within local parity bands.
- `--write-mlx-gpu-drift-attestation` writes
  `mlx_gpu_forward_drift_attestation.json` as false-authority Metal calibration
  evidence, preserving the GPU Conv2d drift signal without poisoning exact
  readiness blockers.

## Next Required Gates

1. Add scorer-loss or calibrated scorer-response loss to long training.
2. Export an actual long-training checkpoint through the same CPU parity gate.
3. Run full-frame inflate parity against source runtime.
4. Only after those pass, claim and dispatch exact CPU/CUDA auth eval.
