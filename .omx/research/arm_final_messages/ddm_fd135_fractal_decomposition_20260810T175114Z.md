Completed the charter. The durable handoff is [ddm_fd135_fractal_decomposition_20260810.md](/Users/adpena/Projects/pact/.omx/research/ddm_fd135_fractal_decomposition_20260810.md), SHA-256 `d3a7a7a8c39f45622539341fadcc740dfd4d9df4408a3aed8c8c1ed835ba9d1a`.

Key conclusions:

- PR135’s exact `−4,328 B` improvement closes as `−52 RCF1`, `−336 IHS2/fixed schema`, `−4 carrier repack`, `−999 WANS1/CAP1/container`, `−2,122 RC64`, and `−815 CBQ`.
- PR135 moved 5,989/27,648 basis symbols, 7,044/7,200 coefficients across all 600 rows, two semantic codes, and five frame-0 events.
- The charter’s “seg untouched” prior was false: their book tested segmentation and accepted a small FiLM-plus-carrier improvement. Explicit overlays failed; implicit joint conditioning remains open.
- All 231 ExperimentBook files were read. The retained test run yielded `181 passed / 32 skipped / 2 failed / 26 errors`; failures were missing `work/` artifacts or `third_party/challenge/frame_utils`, so the book is not self-contained.
- Every extracted payload was retained. The [evidence manifest](/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr135/retained_fd135/EVIDENCE_MANIFEST.json) covers 77 files and 2,246,556 bytes.
- No scorer or remote job ran in this arm. MAIN’s independent replay established PR135 at `S=0.16226942370411543 @ 186,724 B [contest-CUDA T4, n600]`; its CPU path refused because F26 explicitly requires CUDA.
- The bounded Aug-06→Aug-10 search found no new same-family paper. The nearest result, [GVC-RT](https://arxiv.org/abs/2608.04891), was submitted Aug-05 and uses a different LFQ/generative family; older [NeuroQuant](https://arxiv.org/abs/2502.11729) supports mixed-precision plausibility but provides no F26 evidence.

Main Git landing was blocked by the sandbox’s read-only object database. The exact memo was serializer-committed in a clean fallback clone as `758c30e5686c90156daa5fe79885e70455ae84ad`. The verified [bundle](/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr135/retained_fd135/fd135_fallback_commit_758c30e568.bundle) and [receipt](/Volumes/VertigoDataTier/pact/pr135_intake_20260810/pr135/retained_fd135/FALLBACK_COMMIT_RECEIPT.json) are retained; the main-worktree memo remains untracked and byte-identical.

Own-vehicle frontier remains **lc2 `S=0.16959899569230852 @ 187,226 B [contest-CUDA T4, adjudicated, n600]`**. This unit did not move it or reach sub-0.15.

## NEXT_IF_RESUMED

- `fallback_commit_landing`: disposition=`QUEUED-WITH-A-FIRE-ORDER`; owner=`next Git-writable operator`; consumer store=`main branch`; fire trigger=`Git object writes become available; fetch the retained bundle and fast-forward/cherry-pick commit 758c30e568 after preserving the identical untracked memo`.
- `current_base_joint_solve`: disposition=`QUEUED-WITH-A-FIRE-ORDER`; owner=`MAIN/#995 successor`; consumer store=`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/`; fire trigger=`lane claimed and deterministic resume plus per-stage/per-candidate retention proven`.
- `f26_same_state_ans_race`: disposition=`QUEUED-WITH-A-FIRE-ORDER`; owner=`MAIN/lossless-pack successor`; consumer store=`/Volumes/VertigoDataTier/pact/pr135_f26_same_state_ans_20260810/`; fire trigger=`exact F26 probability export and receiver-equality harness exist with no duplicate coder lane`.
- `cap1_metadata_pack`: disposition=`QUEUED-WITH-A-FIRE-ORDER`; owner=`MAIN/lossless-pack successor`; consumer store=`/Volumes/VertigoDataTier/pact/pr135_cap1_metadata_pack_20260810/`; fire trigger=`strict pack/unpack equality and repeat-identical complete-archive builder are ready`.
- `adaptive_mixed_precision`: disposition=`QUEUED-WITH-A-FIRE-ORDER`; owner=`#869 successor folded into MAIN/#995`; consumer store=`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/mixed_precision/`; fire trigger=`exact F26 sensitivity map exists without importing static-W3 byte credits`.
- `implicit_edge_conditioning`: disposition=`QUEUED-WITH-A-FIRE-ORDER`; owner=`MAIN/#995 successor`; consumer store=`/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/`; fire trigger=`mask-free proposals can be evaluated jointly for pose, seg, and rate on retained complete candidates`.
- `learned_pose_gauge_qat`: disposition=`QUEUED-WITH-A-FIRE-ORDER`; owner=`pz4`; consumer store=`pz4 governed store`; fire trigger=`proposal pre-proves at least 2,000 B savings and MSE below 2.5e-6`.

## LIVE-HYPOTHESES

- Same-state lc2 ANS may preserve part of its 178-byte cross-state advantage because F26 RC64 and lc2 encode the same dominant token surface; only an exact same-state race can price it.
- New global joint int12/basis/FiLM starts may improve pose because PR135’s shipped state exhausted local singleton moves, not the global joint space.
- CAP1 metadata can remove up to 40 raw bytes before outer LZ through exact fixed-field packing.
- Adaptive W3/W4 allocation may work where uniform W3 failed because Pact’s prior cell measurements showed strong state dependence and joint effects.
- Implicit edge-conditioned proposals may improve segmentation without paying the explicit-overlay storage cost that killed the author’s margin sidecar.
- Learned gauge-constrained retraining remains plausible; direct blind/null pixel filling does not.

## DEAD-ENDS

- Direct #401/#580 byte reclaim: CPR1/F26 stores no camera-resolution residual field, so there are zero direct bytes to reclaim.
- Frozen post-hoc PK2 carrier transforms: the exact low-rank-plus-residual form added 4,316 bytes and did not clear its fidelity gate.
- Static uniform W3: saved 852 bytes but catastrophically worsened score; this closes that formulation, not adaptive mixed precision.
- Generic RC64 termination/coder tuning: realized RC64 is only 0.539946 byte above its exact model ideal.
- Explicit stored margin overlays and renderer-only seg polish: their byte or pose cost exceeded their segmentation benefit.
- Post-hoc HPAC clipping, temperature, output-bias, LoRA, motion replacement, and tested repacks did not beat the shipped complete archive.
- A PR135 CPU score: the shipped receiver explicitly refuses non-CUDA inflation.
- New same-family research in the bounded Aug-06→Aug-10 window: none was found in the searched primary-source scope.

