# Trained Renderer Transplant Dispatch Worker - 2026-05-03

Evidence grade: empirical planning/tooling only. Score claim: false. Remote
dispatch: none.

## Purpose

Harden the local handoff from recovered Modal/H100/A100 trained renderer exports
to contest-faithful archive candidates without spending remote wall clock before
local custody checks pass.

## Implementation

- Added `experiments/prepare_trained_renderer_transplant_dispatch.py` as a
  deterministic dry-run planner.
- The planner refuses missing or empty renderer exports, verifies source archive
  bytes/SHA against the exact eval custody record, computes the byte-only
  break-even for strict sub-0.314, and emits the next local commands.
- Exact-eval dispatch readiness stays false unless a pose-safety JSON matches
  the exact source archive SHA and candidate archive SHA and reports
  `safe_for_exact_eval_dispatch=true`.
- The emitted manifest includes dispatch-claim command text before any
  Lightning exact-eval command shape.

## Frontier Custody

- Source archive:
  `experiments/results/lightning_batch/exact_eval_c067_pr75_qp1_top40_p6_t4_awsfix1_20260503T0630Z/archive.zip`
- Source bytes: `276342`
- Exact score: `0.3154707273953505`
- Strict target: `<0.314`
- Byte-only savings needed at unchanged distortion: `2209`
- Max byte-only crossing archive bytes: `274133`

## Active Modal Recovery Inputs

- H100: `fc-01KQP9K42CAWJH7XEV4KC0V28M`
- A100: `fc-01KQP9T1VD14785MG63H7JM5VK`
- A10G: `fc-01KQP9T19Y7PMDETDN99WDMF2W`

## Non-Dispatch Contract

This worker did not poll Modal, launch Lightning, or submit any remote job.
The generated command text is a handoff surface only.
