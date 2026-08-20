The HR1 charter is executed as a complete design specification: [ddm_hr1_realization_engineering_20260811.md](/Users/adpena/Projects/pact/.omx/research/ddm_hr1_realization_engineering_20260811.md).

- SHA-256: `d15c1e3cad98a27ea1fe4919db0e6cd942d6c82c2ff51d00324f1978f292db76`
- Validation: `git diff --check` passed; 583 lines, 39,983 bytes.
- Defines the direct Stage-0 test and four-arm race: frozen renderer, full fine-tune, counted FiLM/head adapter, and joint token-renderer descent.
- Includes exact receiver ordering, event-based continuation, geometry-derived EMA, memory gates, resumability, retained-payload custody, pose/full-score guards, equal-parameter controls, and typed continuation records.
- Identifies a P0 apparatus defect: the existing helper rounds after downsampling, while HR1 requires bicubic camera lift → camera-grid uint8 STE → bilinear scorer downsample.
- The race cannot be collapsed to verification: retained renderer fidelity is 99.9703606474%, below the charter’s 99.99% collapse threshold.

This remained design-only: no scorer, trainer, paid dispatch, evaluator, or payload-materializing run was launched. No score was measured and neither frontier moved.

The required serializer was attempted but could not write a Git object: `unable to create temporary file: Operation not permitted`. The index remains empty and the memo is an uncommitted worktree artifact.

Effective frontier: `0.16195513827824176 @ 186,252 B`, cp135, contest-CUDA T4 n600.  
Own-vehicle frontier: `0.16959899569230852 @ 187,226 B`, LC2, contest-CUDA T4 adjudicated n600.

## NEXT_IF_RESUMED

- `hr1_spec_landing` — disposition: **QUEUED-WITH-A-FIRE-ORDER**; owner: MAIN repository custodian; consumer store: Pact `main` history and js1 Amendment-2 intake; fire trigger: Git-object writes become available and the serializer verifies the exact memo SHA.
- `hr1_roundtrip_and_dsl_preflight` — disposition: **QUEUED-WITH-A-FIRE-ORDER**; owner: js1 apparatus successor; consumer store: `hr1_preflight/` on VertigoDataTier; fire trigger: the specification lands while work remains scorer-free.
- `hr1_stage0_direct_realization` — disposition: **QUEUED-WITH-A-FIRE-ORDER**; owner: js1/#995 scorer successor; consumer store: `hy1_solved_carriage/stage0_v14/`; fire trigger: ps135 terminal receipts exist, C1 independently decodes, storage/governor checks pass, and the n600 scorer lane is claimed.
- `hr1_four_arm_realization_race` — disposition: **QUEUED-WITH-A-FIRE-ORDER**; owner: js1 realization successor; consumer store: `hy1_solved_carriage/joint_realization/`; fire trigger: direct realization misses the refreshed sub-0.15 requirement and all typed/memory preflights pass.
- `hr1_conditional_coordinate_prior` — disposition: **CONDITIONAL-QUEUED**; owner: HY1 representation successor; consumer store: `hy1_solved_carriage/conditional_levelset/`; fire trigger: retained failures localize enough to define an equal-parameter coordinate-prior control.
- `hr1_exact_promotion` — disposition: **QUEUED-WITH-A-FIRE-ORDER**; owner: MAIN; consumer store: exact-evaluation and canonical-pointer stores; fire trigger: a deterministic complete n600 candidate strictly improves full score and passes custody and compliance.

## LIVE-HYPOTHESES

- Full renderer adaptation can preserve enough C1 gain because the grammar-valid tokens begin from 99.9704% renderer fidelity and optimization targets the remaining receiver-realized errors.
- A counted FiLM/head adapter may outperform full fine-tuning in score per byte because prior semantic movement concentrated in FiLM state.
- Joint token-renderer descent may find a nearby hard-token preimage that the renderer realizes more reliably than fixed C1.
- Camera-grid bicubic training may improve boundary survival relative to the shipping bilinear lift.
- A representation-level coordinate prior may help even though additive edge calibration failed, because it changes capacity allocation rather than retuning the saturated probability lattice.

## DEAD-ENDS

- Finishing-stage KD/margin/argmax distillation is closed at formulation scope: its early gain reversed and ended 12.8 noise floors worse than control.
- Short distillation mini-races are not verdicts; DW1’s early winner later reversed.
- Baseline-only verification is closed because retained fidelity misses the explicit collapse threshold.
- Calling the current round-trip helper exact is closed because uint8 occurs at the wrong stage.
- Float-only, loss-only, scalar-parity, and pre-round-trip optimization are closed.
- Literal EMA constants such as `0.997` are closed; decay must derive from actual update geometry.
- Additive edge or scalar-pose calibration is closed at formulation scope by SR1.
- Explicit transmitted edge masks are closed on this vehicle unless a complete representation earns their bytes.
- Generic dense Brotli is closed for the C1 token-plane instance: 429,383 B versus 114,717 B for HPAC plus RC64.