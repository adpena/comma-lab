# Task #541 constructive solver and free-predictor floor implementation spec

Date: 2026-07-19
Lane: `constructive_solver_541_20260719`
Authority: isolated local build and $0 advisory measurement only. The
`0.1910828242 [contest-CPU]` pointer remains unmoved. No launch, paid dispatch,
contest score, promotion, or pointer mutation is authorized. MAIN landing
review is required.

## Objective

Build and measure a deterministic two-plane scorer-grid description in which
the second plane is encoded as a residual from a generic decode-side
predictor, then provide the local box/half-space plus optional rank-6 pose
projection that will consume the in-flight VJP custody. Close one zero-band
rung through counted archive bytes, parse-back, factor-2 uint8 realization,
and the native-float32 hard-oracle check outside decode.

## Frozen inputs and boundaries

- Read-only real cache:
  `/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz`,
  SHA-256
  `cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
- Sacred and read-only:
  `experiments/results/levelset_n600_witness_20260717T113932Z/`.
- Do not modify `tools/measure_joint_seg_pose_rate.py` or
  `src/tac/optimization/joint_seg_pose_rate.py`.
- Consume, do not rebuild, the factor-2 lattice, range projection,
  content-priced coder, and production archive components.
- VJP custody is optional. The admitted per-pair schema is
  `vjp_custody_pair.v1`; its native scorer-plane tensors are `seg_q`,
  `seg_local_lipschitz`, and `pose_j_y`. The current `seg_q` is a local
  diagonal/aggregate relaxation, not the full global cell-to-plane Jacobian;
  it can propose but never certify without the hard oracle.
- Decode may not import/load SegNet, PoseNet, Torch, source frames, cached
  labels, or margins. Every video-derived bootstrap, descriptor, and residual
  byte is inside `archive.zip`.

## Predictor-floor representation

For each real pair, form uint8 scorer planes

`y0 = round(A(gt_f0))`, `y1 = round(A(gt_f1))`

using the exact factor-2 integer numerator operator. The quantization and its
scope are explicit: this is the production receiver's uint8-plane family, not
the exact i32-numerator plane family.

The three deterministic predictors consume only decoded `y0` plus fixed code:

1. `previous-plane-copy.v1`: `p = y0`.
2. `affine6-q12.v1`: a six-parameter output-to-input affine warp of `y0`.
   Encode-side least squares selects six signed Q12 pixel-coordinate deltas;
   the descriptor is exactly six little-endian int32 values, 24
   video-derived bytes per pair before outer compression. Decode applies a
   fixed-point bilinear warp with integer rounding.
3. `spatial-smooth-121.v1`: separable edge-replicated `[1,2,1]/4` smoothing of
   `y0`, with integer rounding and no fitted parameter.

The exact residual is `r = int16(y1) - int16(p)`. A canonical, fully consumed
payload carries pair id, mode id, frame-0 bootstrap bytes, descriptor length
and bytes, residual length and bytes, hashes, and no trailing data. The packet
decoder reconstructs both `y0` and `y1` byte-identically. The complete archive
rate includes bootstrap + descriptor + residual + framing. The decisive
predictor-floor table also reports descriptor+residual conditional rate so it
can be compared across predictors without pretending the bootstrap is free.

Measurement uses four disjoint, resumable 12-pair chunks (`0..11`, `12..23`,
`24..35`, `36..47`) and composes exactly 48 real pairs toward n600. Each chunk
records actual Brotli quality 11 and zstd level 19 bytes with decompression and
descriptor parse-back. Per-class rows use cached classes `0..4`. Per-margin
rows use `[0,.1)`, `[.1,.25)`, `[.25,.5)`, `[.5,1)`, `[1,2)`, `[2,inf)`.
Attribution streams carry their packed membership masks plus selected int16
RGB residual values and are independently compressed; no global codec bytes
are pro-rated.

## Additive production grammar

Add `predictor-residual-u8.v1` to the production receiver's closed y-codec
registry and add `description-frame0.v1` to its frame-0 policy registry.
Preserve the existing packet version, section order, ZIP_STORED wrapper,
quotient-residual meaning, integer factor-2 realization, write-once stages,
and legacy raw/Brotli behavior.

For the new codec only, `y_description` decodes to the canonical concatenation
of frame-0 and frame-1 uint8 scorer planes. `decode_y_planes` continues to
return frame-1 for compatibility; a new typed decode helper returns both.
Inflate realizes and verifies both planes separately and emits `(frame0,
frame1)` rather than using the repeat policy. The whole y-description section
remains `video_derived=true`, so existing packet/archive accounting charges
all bootstrap, affine, and residual bytes. Internal accounting is re-derived
from parsed bytes.

## Constructive solve

For target planes `y`, predictors `p`, and candidate displacement
`delta = y_hat - y`, minimize the deterministic quadratic rate proxy

`sum w_c * (delta - (p-y))^2 / 2`

subject to the local relaxation

`lower <= delta <= upper`, `q dot delta >= -margin`.

Per pixel, project the predictor anchor onto the box/half-space intersection.
Use the monotone clipped form

`delta(lambda) = clip(anchor + lambda*q/w, lower, upper)`

and deterministic bisection/breakpoint bounds when the half-space binds.
Zero band has `lower=upper=0`, so it returns the exact target plane.

When `pose_j_y` is present, project the two-plane displacement onto

`||J delta||^2 / 6 <= tau_pose`

through a six-dimensional Gram eigensolve/root and deterministic Dykstra
alternation with the Seg/box set. This is a proposal certificate only. Final
admission requires the caller-supplied native-float32 hard oracle.

The module validates the exact `vjp_custody_pair.v1` NPZ field names,
dtypes/shapes, receiver arithmetic, tensor hashes, and pair id. Chunk execution
uses config-hashed write-once per-pair stages and an atomic resume state;
resume re-derives inputs and refuses custody drift. Lattice realization uses
the landed factor-2 function and exact numerator verifier. The hard-oracle hook
runs encode-side after realization and is absent from archive decode.

## File ownership split

- Receiver/codec owner:
  `src/tac/codec/v10_predictor_residual.py`, additive edits to
  `src/tac/witness_dsl/v10_production_receiver.py`, and
  `src/tac/tests/test_v10_predictor_residual.py` plus receiver integration
  tests.
- Solver owner:
  `src/tac/optimization/v10_constructive_solver.py` and
  `src/tac/tests/test_v10_constructive_solver.py`.
- Measurement owner:
  `tools/measure_v10_free_predictor_floor.py` and focused tool tests.
- Primary agent owns integration, real measurements, receipts, self-review,
  memo, review-tracker marks, and commits.

No owner may edit the two forbidden live-arm files, the sacred tree,
`upstream/`, pointer/submission surfaces, or another owner's files.

## Acceptance

1. Predictor codecs are deterministic and byte-exact; malformed geometry,
   length, hash, dtype, descriptor, mode, and trailing bytes are refused.
2. Existing raw/Brotli production receiver tests remain green. New codec
   archive parse-back counts every video-derived byte and inflates exact
   two-plane scorer targets through the lattice.
3. Synthetic solver tests cover inactive/binding half-spaces, box clipping,
   zero band, malformed `q`, rank-6 pose projection, repeated-byte
   determinism, resume equivalence, and hard-oracle refusal.
4. A real `gt_n600` smoke covers cache geometry and zero-band lattice
   realization without citing synthetic numbers.
5. Four disjoint n12 predictor receipts compose n48 and report actual
   Brotli-Q11/zstd-19 global and class/margin attribution bytes.
6. Rung-E builds one actual archive with the measured best predictor, reparses
   it, inflates both planes, verifies exact integer numerator equality and the
   f32 law via the external hard oracle, and reports actual archive bytes,
   `d_seg`, and `d_pose` on the same 48-pair set.
7. All claims are labeled MEASURED/DERIVED/INFERRED; negatives carry
   formulation/instance scope. Pointer delta remains zero.
8. At most five self-review rounds. MAIN independently reviews the branch
   before any landing, promotion, launch, or score interpretation.
