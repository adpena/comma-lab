# PR85 STBM1BR + QRGB Randmulti Stack Candidate - 2026-05-04

- tool: `experiments/build_pr85_stbm1br_qrgb_randmulti_stack_candidate.py`
- score_claim: false
- dispatch_performed: false
- remote_jobs_dispatched: false
- dispatch_unlocked: false

## Source Chain

- PR85 source: `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e` bytes `236328`
- STBM source: `c6f004d444ed32c628611a2f21f567c666af6bcbcceba618cc089ec024a0cda6` bytes `229756`
- STBM transform: mask-only, decoded render-order parity, diff_pixels `0`
- QRGB transform: randmulti pair `0192`, source value `0`, candidate value `20`

## Candidate

- archive: `experiments/results/pr85_stbm1br_qrgb_randmulti_stack_20260504_codex/pr85_stbm1br_plus_qrgb_f2_randglobal_pair_0192/archive.zip`
- bytes: `230038`
- sha256: `bd883ed0216e97ca59bc2bb7609d473ee001b77494c8d02164d7e356eebc0e0a`
- manifest: `experiments/results/pr85_stbm1br_qrgb_randmulti_stack_20260504_codex/pr85_stbm1br_plus_qrgb_f2_randglobal_pair_0192/manifest.json`

## Readiness

- orthogonality_status: `passed`
- fixed_runtime_preflight: `experiments/results/pr85_stbm1br_qrgb_randmulti_stack_20260504_codex/pr85_stbm1br_plus_qrgb_f2_randglobal_pair_0192/fixed_runtime_preflight.json`
- fixed_runtime_readiness: `ready`
- exact_eval_safe_after_standalone_exact_positives: true
- remote dispatch: not performed and not unlocked

## Reactivation Criteria

- STBM standalone exact CUDA positive lands for the exact STBM archive SHA
- QRGB randmulti 0192 standalone exact CUDA positive lands for the exact QRGB archive SHA
- A fresh lane claim is recorded before any exact eval dispatch
