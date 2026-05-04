# PR85 QRGB Transfer Archive Candidates - 2026-05-04

- tool: `experiments/build_pr85_qrgb_transfer_archive_candidates.py`
- score_claim: false
- dispatch_performed: false
- remote_jobs_dispatched: false
- dispatch_unlocked: true
- blocker_class: `none`

## Source Anchor

- PR85 source archive bytes: `236328`
- PR85 source archive sha256: `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- known PR85 anchor match: True

## Build Scope

- max candidates: `3`
- selected unique pairs: `3`
- archive candidates built: `3`
- fixed-runtime preflights: `3`
- ready after lane claim: `3`

## Candidate Artifacts

- `pr85_qrgb_f2_randglobal_pair_0192`
  - build_status: `built`
  - dispatch_unlocked: true
  - archive: `experiments/results/pr85_qrgb_transfer_archive_candidates_20260504_worker/pr85_qrgb_f2_randglobal_pair_0192/archive.zip`
  - bytes: `236616`
  - sha256: `228f8dff9e14bc7d3cdd445d6c7d73ed1818c0facecaa21e97ab71a523b2da40`
  - changed_segments: `['randmulti']`
  - fixed_runtime_ready: `True`
- `pr85_qrgb_f1_bias_pair_0060`
  - build_status: `built`
  - dispatch_unlocked: true
  - archive: `experiments/results/pr85_qrgb_transfer_archive_candidates_20260504_worker/pr85_qrgb_f1_bias_pair_0060/archive.zip`
  - bytes: `236336`
  - sha256: `81fb8d715e37966ead2764f21846909f4bd570f2bfdc5469c53a83ded495bc81`
  - changed_segments: `['bias']`
  - fixed_runtime_ready: `True`
- `pr85_qrgb_f1_bias_pair_0164`
  - build_status: `built`
  - dispatch_unlocked: true
  - archive: `experiments/results/pr85_qrgb_transfer_archive_candidates_20260504_worker/pr85_qrgb_f1_bias_pair_0164/archive.zip`
  - bytes: `236335`
  - sha256: `d5e2a7904f3a9f5333220670e9fe7a99a8c665f16a63b2af90f5c366202fde9e`
  - changed_segments: `['bias']`
  - fixed_runtime_ready: `True`

## Blockers

- none

## Compliance Notes

- Local archive builds only; no training, scorer import, GPU dispatch, or score claim.
- Archives remain single-member `x` PR85 bundles and are byte-closed.
- Exact eval is allowed only after a lane claim and only through canonical CUDA auth eval.
