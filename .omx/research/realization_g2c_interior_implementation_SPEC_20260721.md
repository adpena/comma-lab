# Realization G2c cell-interior implementation spec

Date: 2026-07-21
Task: #578, round 3
Lane: `lane_realization_g2c_interior_578_20260721`
Axis: `[macOS-CPU advisory]`; pointer `0.1910828242 [contest-CPU]` unmoved.

## Objective

Extend `tools/measure_realization_g2_lattice.py` in place with a resumable
n16 -> n64 -> n600 receiver measurement.  The measurement must distinguish
four concrete RGB-plane formulations derived from the seed cell field:

1. `R1_FIXED_MAGNITUDE`: a fixed-radius, five-class generic RGB codebook;
2. `R2_MAX_MARGIN`: frozen-scorer constant-tile max-margin RGB constants,
   transported through the exact #580/#583 factor-2 integer realization;
3. `R3_HOPFIELD_MEMORY_PROX`: a deterministic one-step modern-Hopfield prox
   into class-conditioned generic prototype banks, with local cell context in
   the query;
4. `R4_DYING_WRITE_EXCEPTIONS`: the best preregistered zero-byte base (R2, the
   max-margin rung) plus a parse-backed stream containing only R2's measured
   dying declared writes and their source-derived RGB triplets.

R1-R3 add zero seed bytes and invoke no scorer in the receiver.  R4 is an
encoder-counted control and must report exact payload bytes.  A negative is
scoped to these formulations; it cannot close textured, learned, or more
expressive cell-interior decoder families.

## Measurement contract

- Use the existing seed parser, cell predictor, full-kernel factor-2 projector,
  native CPU-Torch `DistortionNet`, GT cache, and hard pose tubes.
- Generate both camera frames from receiver RGB planes; source frames are
  forbidden for R1-R3.  The current seed has no intra-pair frame0 appearance
  carrier, so both frames use the same decoded cell field and this limitation
  is explicit in the receipt.
- Require exact integer factor-2 parse-back and double-decode identity for every
  pair/rung.
- Record whole-description exact pairs, all-declared-write exact pairs, write
  survival fraction, d_seg, d_pose, and pose-tube count.
- Decompose writes by class, stratum, and measured target-logit margin bucket.
- Preserve an immutable JSON stage per pair/rung plus n16/n64/n600 checkpoints;
  no RGB or camera tensor is persisted.

## Acceptance and failure

`predict_project_realization_admissibility_v1` may become `ADMISSIBLE` only for
a full n600, zero-byte, receiver-derived rung with exact factor-2 transport,
identical double decode, 600/600 whole-description semantic equality, and
600/600 pose-tube acceptance.  Otherwise the equation receives a measured
negative anchor with the failed predicates and exact receipt custody.

## Files and review

- Extend `tools/measure_realization_g2_lattice.py`; do not fork the CLI.
- Extend its focused test file only.
- Update the existing admissibility equation and focused tests after the final
  receipt hash exists.
- Land a receipt, findings memo, DAG FEED, and REUSE MANIFEST.
- Two clean review-tracker passes are required for every changed Python file.
- MAIN must inspect the branch diff and receipt before landing.
