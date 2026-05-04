# PR85 lossless pure-rate local candidate lane - 2026-05-04

## Scope

Disjoint from STBM, PR91/HPM1, and Lightning CLI hardening. Local-only screen over strict ZIP repack, non-mask Brotli recode, and P1D1 pose-order recode.

## Source frontier

- Archive: `experiments/results/public_pr85_intake_20260503_codex/archive.zip`
- Bytes/SHA: `236328` / `eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e`
- Exact PR85 score artifact: `experiments/results/lightning_batch/exact_eval_public_pr85_adaptive_masking_joint_frame_model_t4_20260503T2332Z/contest_auth_eval.json`
- Recomputed score: `0.25806611029397786`

## Result

- Built byte-negative candidate: no
- Reason: `local lossless screen found byte-neutral best case only; strict ZIP overhead is already minimal and all decoded-identical non-mask/P1D1 recodes are no smaller`
- Best screened candidate: `bias_brq10_lg16`
- Best screened byte delta: `0`
- Best screened pure-rate score delta: `0.0`

## Dispatch gate

No remote/GPU dispatch from this lane unless a byte-negative manifest exists, `tools/claim_lane_dispatch.py` has an active non-conflicting claim, local manifest SHA fields still match, PR85 replay/fixed-runtime preflight passes, and exact CUDA auth eval is run through `archive.zip -> inflate.sh -> upstream/evaluate.py`.

