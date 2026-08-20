EC1 is complete as a scorer-free design arm and committed at `fa29eb9ea17d3bfd5138478470600f322050634d`. The pointer did not move.

The selected first training family is oriented decoded-token conditioning:

| Family | AUROC | Expected errors in top 34,970 |
|---|---:|---:|
| Class-only | 0.82525 | 442 |
| Undirected | 0.99096 | 2,992 |
| Oriented | **0.99566** | **8,380** |

The counted oriented adapter is injected before CP135’s nonlinear TokenBlocks—no explicit mask or edge sidecar. Its design-price archive delta is 1,707 B, requiring at least 1,340 realized flips at 0.785 flips/B.

The actual CP135 mechanism probe showed:

- Zero adapter: bit-identical output.
- Nonzero design control: 589,814 pre-R values changed on seeded pair 125.
- No trained candidate, scorer pass, PoseNet result, or exact row was produced.
- No full-n600 scorer slot, Metal, or Modal was used.

Artifacts:

- [Evidence memo](/Users/adpena/Projects/pact/.omx/research/ddm_ec1_implicit_edge_conditioning_20260814.md)
- [Consumer implementation](/Users/adpena/Projects/pact/experiments/ddm_ec1_implicit_edge_conditioning.py)
- [Receiver module](/Users/adpena/Projects/pact/experiments/ddm_ec1_runtime/ec1_latent_conditioner.py)
- [Tests](/Users/adpena/Projects/pact/experiments/tests/test_ddm_ec1_implicit_edge_conditioning.py)
- [Final retained result](/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/ddm_ec1_20260814/FINAL_RESULT.json), SHA `bb0a6582…`
- [MAIN fire order](/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/ddm_ec1_20260814/MAIN_CUDA_FIRE_ORDER.json), SHA `0d403be3…`

Verification: 18 focused EC1/JS3/SA1 tests passed, Ruff passed, strict payload-retention inspection returned no findings, both review passes completed, and the adapted identity-control runtime compiled.

The initial run omitted the derived base-error field and exact raw LUT buffers. Both were deterministically recovered from retained inputs, recorded in retention-repair receipts, and the implementation now persists them before measurement.

Own-vehicle frontier unchanged: **S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory] n600**.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN true-CUDA trainer and exact-row owner; consumer store: `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/edge_conditioned/ddm_ec1_20260814/main_cuda/`; fire trigger: MAIN owns the sole n600 scorer lane, no competing scorer job is active, and a reviewed resumable CUDA trainer implements the oriented pre-TokenBlock adapter with live/EMA stage checkpoints and complete payload retention. Package using the recorded command, then measure through the re1t/js1b worker against the 34,970-flip base.

## LIVE-HYPOTHESES

- Oriented context may clear break-even because its 8,380-error targeting mass is 6.25× the 1,340-flip reference hurdle.
- Pre-TokenBlock injection may survive where SA1 failed because four nonlinear, dilated blocks can amplify a small latent displacement before camera rounding.
- Equal-parameter controls may show whether orientation itself matters or whether the improvement is merely additional target-selection capacity.

## DEAD-ENDS

- Explicit overlays: JS1C produced 55,807 flips versus 34,970 base.
- Frozen-receiver singleton edits: JS8 realized 38 flips versus 4,314 needed.
- Additive edge probability calibration: SR1 saved only 2 bytes.
- Post-render hidden-4 conditioning: SA1 was exactly T4-inert.
- Local CPU scoring as admission authority: closed by the measured local/T4 mismatch.
- Seeded EC1 design modules and identity controls are not trained candidates and must not be scored as such.