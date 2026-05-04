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

- max candidates: `8`
- selected unique pairs: `4`
- archive candidates built: `4`
- fixed-runtime preflights: `4`
- ready after lane claim: `4`

## Candidate Artifacts

- `pr85_qrgb_f2_randglobal_pair_0192`
  - build_status: `built`
  - dispatch_unlocked: true
  - archive: `experiments/results/pr85_qrgb_transfer_archive_candidates_20260504_codex/pr85_qrgb_f2_randglobal_pair_0192/archive.zip`
  - bytes: `236616`
  - sha256: `228f8dff9e14bc7d3cdd445d6c7d73ed1818c0facecaa21e97ab71a523b2da40`
  - changed_segments: `['randmulti']`
  - fixed_runtime_ready: `True`
- `pr85_qrgb_f1_bias_pair_0060`
  - build_status: `built`
  - dispatch_unlocked: true
  - archive: `experiments/results/pr85_qrgb_transfer_archive_candidates_20260504_codex/pr85_qrgb_f1_bias_pair_0060/archive.zip`
  - bytes: `236336`
  - sha256: `81fb8d715e37966ead2764f21846909f4bd570f2bfdc5469c53a83ded495bc81`
  - changed_segments: `['bias']`
  - fixed_runtime_ready: `True`
- `pr85_qrgb_f1_bias_pair_0164`
  - build_status: `built`
  - dispatch_unlocked: true
  - archive: `experiments/results/pr85_qrgb_transfer_archive_candidates_20260504_codex/pr85_qrgb_f1_bias_pair_0164/archive.zip`
  - bytes: `236335`
  - sha256: `d5e2a7904f3a9f5333220670e9fe7a99a8c665f16a63b2af90f5c366202fde9e`
  - changed_segments: `['bias']`
  - fixed_runtime_ready: `True`
- `pr85_qrgb_f1_region_pair_0197`
  - build_status: `built`
  - dispatch_unlocked: true
  - archive: `experiments/results/pr85_qrgb_transfer_archive_candidates_20260504_codex/pr85_qrgb_f1_region_pair_0197/archive.zip`
  - bytes: `236335`
  - sha256: `236751af46a9c98fa286ecfe613c23a2b96bffbe31784da052304701e02b71c6`
  - changed_segments: `['region']`
  - fixed_runtime_ready: `True`

## Blockers

- none

## Compliance Notes

- Local archive builds only; no training, scorer import, GPU dispatch, or score claim.
- Archives remain single-member `x` PR85 bundles and are byte-closed.
- Exact eval is allowed only after a lane claim and only through canonical CUDA auth eval.

## 2026-05-04T06:01Z Randmulti 0192 Exact Negative

The QRGB transfer atom `pr85_qrgb_f2_randglobal_pair_0192` completed on two
independent T4/equivalent Lightning jobs and both packets reproduced the same
A++ exact CUDA result.

Result:

- T4 score: `0.25826470562795345`
- delta versus PR85: `+0.00019859533397559304`
- bytes: `236616`
- sha256:
  `228f8dff9e14bc7d3cdd445d6c7d73ed1818c0facecaa21e97ab71a523b2da40`
- SegNet: `0.00057187`
- PoseNet: `0.00018944`
- sample count: `600`
- primary artifact:
  `experiments/results/lightning_batch/exact_eval_pr85_qrgb_f2_randglobal_pair_0192_t4_20260504T0536Z/contest_auth_eval.adjudicated.json`
- duplicate artifact:
  `experiments/results/lightning_batch/exact_eval_pr85_qrgb_f2_randglobal_pair_0192_t4_g4dn2x_20260504T0544Z/contest_auth_eval.adjudicated.json`

Decision:

- Retire this measured randmulti singleton as a standalone PR85 atom.
- Do not stack this randmulti atom onto STBM1BR unless a later exact-positive
  interaction study gives a new source-specific reason; current evidence says
  it consumes bytes and worsens the component basin.
- Keep the QRGB transfer machinery as external-PR analysis tooling, but do not
  dispatch the current QRGB singleton/combo family again without a new
  exact-evidenced premise.
