---
title: Loader Drift Xray Supersession And Remaining Guard Gaps
date: 2026-05-08
owner: codex
status: adversarial review follow-up; diagnostic only
score_claim: false
promotion_eligible: false
---

# Loader Drift Xray Supersession And Remaining Guard Gaps

This ledger records the xhigh review follow-up on the CPU/CUDA and DALI/PyAV
diagnostic track. It is not score evidence.

## Corrections Landed

- Superseded the stale FastViT attention-softmax explanation in
  `.omx/research/cuda_cpu_drift_sweep_design_20260508_claude.md`. PoseNet uses
  RepMixer/conv-style FastViT-T12 blocks; the current mechanism hypothesis is
  mixed loader-byte drift plus CPU/CUDA forward-kernel drift.
- Reworded the NVDEC/PyAV 25% contribution in
  `.omx/research/cuda_cpu_pose_drift_mechanism_deep_dive_20260508_claude.md`
  from measured/confirmed attribution to unmeasured prior.
- Corrected the discriminator prescription: do not instantiate
  `AVVideoDataset` on a CUDA device. The rigorous test is CPU/PyAV decoded
  tensors fed into CUDA forward, plus DALI tensors fed through CPU where
  available.
- Reworded `reports/latest.md` so the operator next-action menu points to the
  shared-tensor harness rather than a CUDA `AVVideoDataset` override.

## Remaining Guard Gaps

- `tools/probe_eval_loader_drift.py --run-forward-cells` needs a strict
  `forward_matrix_complete` field. A runtime-error cell should fail the
  diagnostic matrix while preserving `score_claim=false`.
- The loader probe should hash and optionally dump shared input tensors for
  AV and DALI batches, then record per-cell input SHA, frame count, video SHA,
  library versions, GPU/driver metadata, and repeat-run jitter.
- The layer xray currently localizes PoseNet/Hydra only. Either add a SegNet
  tracer or document the narrower scope everywhere it is surfaced.
- `reports/eval_loader_drift_probe_dali_vs_pyav_plan_20260508.json` predates
  the hardened schema and should be regenerated or marked superseded.

## Reactivation Criteria

The drift mechanism can become solver-calibration evidence only after a CUDA
host run fills all four diagnostic cells with tensor custody:

1. CPU forward on PyAV tensors.
2. CUDA forward on DALI tensors.
3. CUDA forward on shared PyAV tensors.
4. CPU forward on shared DALI tensors, or a fail-closed unsupported-cell
   record if DALI tensor extraction cannot be run on CPU.

All rows remain `[diagnostic]`; none may promote, rank, kill, or replace exact
auth eval.
