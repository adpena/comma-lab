# rr2 free-decode T4 row REFUSED — device-scoped decode identity (2026-08-17)

**Verdict: REFUSED. S 27.832302540259292 [contest-CUDA T4, n600] vs projected 0.15853325034789675.**
Pointer UNMOVED (hv1 ep0634 S 0.15959729295498598 stands). verdict_scope: INSTANCE
(this candidate runtime + this encode; the rr2 ENCODER family survives — see cure).

## The row (call fc-01M0889RGGV57B50NJG8NH2GE4, 472.3 s, ~$0.16, rc=0)

- archive 181,161 B sha `48d5d469d0d87e72d3465e5a76602ae73e3ce4c331d7491895bde240fcc9eb42`
  (pin verified remotely, `expected_archive_sha256_match: true`)
- avg_segnet_dist **0.08924859** (301× the frontier's 0.00029611)
- avg_posenet_dist **35.29444504** (the NO-pose-carrier magnitude; frontier 6.88e-6)
- S = 8.9249 + 18.7868 + 0.1206 = 27.8323 (recomputed from components, exact)
- Result: `experiments/results/ddm_rr2_freedecode_exact_contest_cuda_20260817_r1/MODAL_REMOTE_RESULT.json`

## Mechanism (isolated from retained hashes — three inflates, same 3,662,409,600 B)

| decode | device | inflate sha256 | score |
|---|---|---|---|
| hv1 frontier, arm's local proof | CPU | `e5539653f598a1c31e28900888f450a6de019c…` | (identity reference) |
| hv1 frontier, its own T4 row (r2) | CUDA | `9a6b75e55268a68ed7e1b59d9ee871f99b89b0…` | seg 0.00029611 / pose 6.88e-6 ✓ |
| rr2 candidate, this row | CUDA | `23dcabdbaf781567b9f4c89952d860f9a4ae73…` | seg 0.0892 / pose 35.29 ✗ |

The base decode is **device-dependent by design** — the F26/HPAC lineage's CUDA-lock
(m05): the entropy stream is arithmetic-coded against the neural AR-prior's
probabilities, and those probabilities differ CPU vs CUDA. The shipped base stream
was ENCODED against CUDA probabilities, so it decodes correctly on T4 (row 2). The
rr2 encoder produced its recoded stream against **CPU** probabilities (its
base-probability digests 562ac652…/dd48843b… matched the frontier's **CPU** decode) —
on T4 the adaptive free-decode conditions on CUDA probabilities that do not match the
encoder's, desynchronizes, and emits garbage tokens (both distortion terms destroyed;
poison class m23 CUDA-DRIFT meeting adaptive coding).

## The law (NEW, binds every future free-decode/context-coded candidate)

**Decode-identity proofs are DEVICE-SCOPED.** A CPU byte-identity proof does NOT
license a CUDA fire when the stream is context-coded against model-produced
probabilities. Requirement, one of:
1. **Encode against the DECODE device's probabilities** (dump the per-symbol
   probability sequence ON the target device, encode against it), or
2. **Device-invariant context** — the corrector/coder conditions only on
   integer-exact state (fixed-point prior), never raw float NN outputs.

Sister of `batch_shape_is_part_of_the_forward_instrument` (et4: the instrument
includes device/threads/batch) and `the-instruments-own-units-level-and-aggregation…`
(the measured object ≠ the named object — here "decode-identical" was CPU-decode-
identical, and the fired axis was CUDA). MAIN's fire-time waiver crossed that
boundary; recorded as MAIN's error, not the arm's — the arm labeled its proof
"frontier's own CPU inflate" correctly.

## Cure chain (the −0.0010640 S win stays LIVE)

1. **CUDA-prob dump** (~$0.16): one T4 job runs the BASE decode and persists the
   per-symbol AR-prior probability sequence (payload law: persist bytes + sha).
   The base T4 decode is proven deterministic (hv1 row repeat-identical).
2. **Re-encode** locally: rr2 encoder (proven exact inverse) against the dumped
   CUDA probabilities → new tokens stream, same free_corrector, same target size
   class (~181.2 KB; size may shift a few bytes — reprice before fire).
3. **Restage + refire** (~$0.16) with the SAME candidate-bound runtime discipline;
   expected components = hv1 T4 row's seg/pose exactly (token field bit-identity
   now holds ON the decode device) + rate 25·B/37,545,489.

Option 2 (integer-exact context) is the durable family fix — route to the rr2
successor charter as the shippability requirement for ALL free-decode candidates.

## Ledger

- Spend: +$0.16 → Modal ≈ $6.5/$20 (#381).
- Lane `lane_ddm_rr2_freedecode_harvest_exact_contest_cuda_20260817` → terminal
  `failed_device_scoped_decode_desync`.
- Overridden-QUEUED-disposition note: the arm's NEXT row 1 said "bundle before any
  T4 fire"; MAIN overrode and fired solo (hv1 precedent, ~$0.16 vs headroom).
  The override did NOT cause the failure (a bundled fire would have failed
  identically), but the device-scope gap would have been caught by the arm's own
  pre-registered caution had the CPU-axis row been bought first. Recorded honestly.

STORES CONSULTED: MODAL_REMOTE_RESULT.json (this row + hv1 r2), inflated_outputs
manifests (all three shas above), ddm_rr2_encoder_byteclose_20260817.md (arm proof
chain), m05 (F26 CUDA-lock), m23 (CUDA-drift poison), et4 batch-shape law,
#1054 (CPU-axis device dependence on frontier bytes).
