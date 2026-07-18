# Standalone DAG FEED — shared-resize Seg/Pose joint solve (#538)

Date: 2026-07-18  
Status: `ADVISORY_BUILD`, `research_only=true`, `NO_LAUNCH`  
Verdict scope: `LOCAL_SHARED-A_COUPLING_INSTANCE`  
Pointer delta: `UNMOVED` (`0.19108`)  
Sacred c2: `READ_ONLY / NOT_MUTATED`

## `FEED-538-SHARED-A-JOINT-COUPLING`

```text
frozen scorer factorization
  -> assert A_seg == A_pose: camera (874,1164) -> scorer (384,512)
  -> real n600-trained EMA candidate + stored n600 GT cache
  -> deterministic labeled n-of-600 pair sample
  -> camera uint8 decode (the actual shared render variable)
  -> differentiable clone of the exact shared A and YUV6 preprocessing
  -> {winner-rival zero-margin Seg hinge, raw first-six Pose MSE}
  -> two camera-render pullbacks {g_seg, g_pose}
  -> smooth pullback Gram G = J J^T
  -> score-derived coefficients {100, 5/sqrt(10*d_pose)}
  -> q_joint = 100*g_seg + lambda_pose*g_pose
  -> fixed-support one-LSB directions {seg, pose, joint}
  -> Q_uint8 -> shared A -> frozen CPU-torch {Seg argmax, Pose first-six}
  -> batch-32 padded finite-response matrix
  -> honest subset receipt + help/harm classification
  -> {full-n600 + byte-close adoption measurement | retain argv-inert advisory only}
```

The off-diagonal reported by the smooth leg is the off-diagonal of the
**pullback Gram**, `G_sp = <g_seg,g_pose>`. A Jacobian with two output rows and a
shared high-dimensional render input does not itself have a meaningful scalar
“off-diagonal.” Exact `d_seg` is discontinuous, so only the finite-response leg
through uint8 and the frozen scorer may be labeled measured Seg/Pose response.

### Blocking edges

- A deterministic n-of-600 sample is not a full-n600 effect estimate.
- The winner-rival hinge row is a local differentiable surrogate; it is not the
  derivative of exact argmax disagreement.
- Candidate-space one-LSB secants measure one checkpoint and preregistered
  support family; they do not close all possible joint parametrizations.
- `[macOS-CPU advisory]` is separate from contest-CPU and contest-CUDA.
- No candidate archive, exact counted bytes, receiver parse-back, resumed run,
  or live trainer consumption is created by this lane.
- A local small/zero Gram overlap would not make the full constrained inverse
  separable: both losses still share `Q_uint8`, `A`, payload capacity, and rate.

## `FEED-538-COMPLETENESS-CERTIFICATE`

```text
SPEC v10 §14.11 + frozen-forward factorization
  -> sealed ten-factor inventory
  -> split factor 3 into {3a camera-A preimage, 3b shared-A coupling}
  -> eleven leaf manifest
  -> per leaf:
       {derivation, build SHA, compiler binding, consumer,
        resume replay, measurement receipt, axis,
        interaction matrix, adoption/scoped exclusion}
  -> reopen every receipt and verify schema/hash/producer-consumer identity
  -> refuse on {missing, stale, wrong-axis, argv-inert-as-live, unscoped fold}
  -> real parser-verified config + receiver-closed payload
  -> COMPLETE_BY_CONSTRUCTION or refusal
```

Current result: `NOT_COMPLETE_BY_CONSTRUCTION`. The sealed leaf matrix is in
`.omx/research/inverse_solve_completeness_matrix_20260718.md`; it does not grant
launch authority.

## Triality

- DAG: this standalone FEED. No shared hot DAG ledger was mutated.
- DSL: schema `shared_resize_joint_coupling_measurement.v1`, deliberately
  `live_trainer_argv=()` and fail-closed on launch/promotion/pointer escalation.
- Equation: candidate `shared_resize_joint_coupling_through_A_v1`, which prices
  the raw loss pullbacks with the contest-derived marginals and distinguishes
  structural authority from subset empirical anchoring.

## Natural-form routing

The shared-`A` term routes to a joint-Jacobian/pullback Gram and KKT dual. Chroma
routes to the BT.601 channel basis, camera resolution to an AA/preimage solve,
uint8 to a bounded lattice projection, and blind coordinates to `ker(A)` plus
generic fill. Fourier is **CARGO-CULTED** here and is neither a factor nor a
replacement for any missing term.

## MAIN-review actions

1. Verify the real input hashes, selected pair IDs, batch geometry, output
   receipt hash, and all MEASURED/DERIVED labels.
2. Review the candidate equation and argv-inert DSL policy before merging; do
   not reinterpret them as a live V10 lever.
3. Preserve the exact eleven-leaf completeness key set when #529/#543 build the
   real compiler/receiver gate.
4. Keep the pointer unchanged. Any live or promotion use requires a separate
   governed authorization and full custody.

