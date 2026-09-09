Implemented, measured, retained, and landed in commit `d368b06fd2a1b83028d765ea2fdc3a3eba4e3a56`.

The full receipt is [ddm_ane3_residue_fixed_conversion_remeasure_20260908.md](/Users/adpena/Projects/pact/.omx/research/ddm_ane3_residue_fixed_conversion_remeasure_20260908.md).

| Fixed-output graph | n120 rate | n600 rate | n120↔n600 | Measured ANE |
|---|---:|---:|---:|---:|
| all-fp16 | 1.52164e-05 | 1.46145e-05 | +4.12% | 0.0% |
| g13 | 1.49197e-05 | 1.45552e-05 | +2.50% | 0.0% |
| g13stem | 1.35634e-05 | 1.31310e-05 | +3.29% | 0.0% |
| tail-k64 | 8.05325e-06 | 7.56158e-06 | +6.50% | 0.0% |

All graphs beat the `3.3e-05` numerical bar, but `MLComputePlan` measured complete CPU fallback after E5/ANEF failure. The ≥75% ANE predicate therefore failed, the provisional g13stem pass was withdrawn, and no screening backend was registered.

PoseNet item 2 retained activation banks for 32 stratified pairs and 18 one-op rescue outputs over 120 stratified pairs. Debug/base parity is exactly zero. The current CPU surface’s dim-0 error is `0.0497447`, not the pinned ANE interval `0.150–0.154`; best op 3 removes only 38.8%. This is a scoped CPU-fallback negative, not an ANE conclusion. Final report: [pose_activation_diff.json](/Volumes/APDataStore/pact/ddm_ane3_residue/item2/pose_activation_diff.json), SHA-256 `6df378abcd50b4d94d4f9044fd72b4b8fb36525d4b56da72c99e6ae66b922b74`.

Verification passed: Ruff, byte compilation, shell syntax, retained-payload/hash audit, two review passes per Python file, 44 tests, Catalog #344, canonical-equation round trip, and task-ledger dispositions. Unrelated worktree changes were preserved.

Frontier unchanged:

`sj1 S 0.13900437796841966 @ 181,645 B [contest-CUDA T4 n600]`

## NEXT_IF_RESUMED

- `QUEUED-WITH-A-FIRE-ORDER` — Owner: `ddm_ane3` handoff to `MAIN / native-ANE runtime owner`; consumer stores: `/Volumes/APDataStore/pact/ddm_ane3_residue/native_ane/` and `/Volumes/VertigoDataTier/pact/ddm_ane3_residue/native_ane_models/`; fire trigger: an E5-cache-writable one-pair g13stem preflight completes without ANE compiler/shared-event failure and `MLComputePlan` proves ≥75% ANE. Then repeat the retained graph set and register segmentation-only g13stem only if its n600 rate remains ≤`3.3e-05`.

## LIVE-HYPOTHESES

- Native g13stem may still pass: forcing fp32 output removed 54.5% of provisional flips, while the predecessor measured 75.5% ANE. The present blocker is the execution environment, not the numerical bar.
- PoseNet’s ANE drift may be distributed across several stem operations: the CPU-fallback best single rescue explains only 38.8%, while the earlier multi-op head intervention removed substantially more.

## DEAD-ENDS

- Treating a `CPU_AND_NE` request as placement evidence: closed because `MLComputePlan` measured 0% ANE.
- Registering g13stem from CPU-fallback fidelity: closed because it satisfies only one acceptance predicate.
- Naming the largest activation-difference tensor as causal: closed on this instance; op 16 has the largest relative error but its rescue removes 0%.
- Reusing the charter’s rounded `11× d_pose`: closed; source arithmetic gives 9.04×.