# V10 A2 profiler implementation spec — 2026-07-18

## Objective

Build the reusable, research-only instrumentation requested by delegated lane
`v10_A2_profiler_20260718`:

1. a bounded-RSS, resumable, chunked n600 frozen-SegNet head-feature cache;
2. an exact-or-bounded uint8 lattice feasible-set profiler with honest
   cardinality/DOF custody;
3. a deterministic min-description selection and Seg-side rate/distortion
   measurement surface with actual raw and entropy-coded byte counts; and
4. a pose-feasibility plug-in boundary that can later form the factor-10 joint
   intersection without replacing the Seg profiler.

This is an instrument landing. It is not a score, archive, promotion, or
contest-axis claim.

## Re-derived blocker diagnosis

The prior n600 rank-4 run did not stop because of OOM, cache corruption, or a
missing SVD dimension. It stopped because feature extraction and a different
positive-control obligation were coupled. At canonical frame 195, pixel
`(214,112)`, frozen CPU-Torch and cached `lstars` both select class 0 with a
`4.76837158203125e-07` live-logit margin. Generic float64 power scoring of the
serialized factorization selects class 1, while native float32 power scoring
ties class 0/class 1 and first-max selects class 0. The prior extractor treated
that expected reduction-order/serialization boundary as a feature-extraction
failure. The new cache must not.

## Required implementation surfaces

Implementation may choose nearby canonical names, but the intended ownership
is:

- `src/tac/witness_control/segnet_head_feature_cache.py` — cache schema,
  validation, bitwise parity, resume and bounded chunk helpers;
- `tools/extract_segnet_head_features_n600.py` — governed n600 extractor;
- `src/tac/optimization/uint8_lattice_profile.py` — enumeration/bounds,
  description-cost selection, aggregation, and pose plug-in protocol;
- `tools/profile_v10_uint8_lattice_n600.py` — resumable real-cache profiler and
  optional frozen-SegNet RD scorer;
- focused tests under `src/tac/tests/` and/or `tools/tests/`.

Do not edit hot DSL, trainer, canonical equation registry, upstream snapshot,
or the pre-existing factor-2 solver unless a minimal compatibility addition is
strictly necessary. Prefer the new profiler module to destabilizing the
99-test-hardened solver.

## Feature-cache contract

- Read real `gt_f1` from the frozen n600 NPZ by stored-member memmap; do not
  materialize the 5 GB NPZ.
- Load the pinned frozen CPU-Torch SegNet and bind executed `modules.py`,
  `frame_utils.py`, weights, tool, and module bytes/hashes.
- Batch/chunk size is explicit and bounded; the canonical authority mode is
  deterministic CPU float32, batch one. RSS cap and timeout are external
  governed-launch inputs and are recorded in the receipt.
- Cache the live final-head logits copied directly from the scorer forward as
  `float32`; these are the bit-identical frozen-head projection authority.
- Cache rank-4 quotient features separately as algebraic/factorized features.
  Never claim they replay live logits bitwise at float32 ties. Record the
  frame-195 generic-f64/native-f32 discrepancy as a diagnostic, not an
  extraction blocker.
- Each canonical frame is a preserved stage. After writing a frame slice,
  flush/fsync, record content hashes for both slices, and atomically advance a
  progress file. Resume revalidates immutable input/config identity and every
  committed slice hash before continuing. It may not trust stored scientific
  metrics.
- Preallocate only on the approved SSD waterfall, run a free-space preflight,
  and provide machine-readable certification/cleanup metadata. Never mutate or
  reuse the preserved prior blocked cache in place.
- Completion requires all 600 canonical frames, manifest parse-back, no
  nonfinite values, and bitwise equality between a fresh live-logit positive
  control and its cached slice. Partial runs remain explicitly partial.

## Feasible-set/DOF contract

- Profile each factor-2 channel block under the exact integer equation
  `c dot u = target_integer`, `u in [0,255]^n`, using the same exact half-pixel
  integer geometry as `DisjointResizeOperator`.
- Return an exact cardinality only after exhaustive traversal. A node/time cap
  must return a proved lower bound, a sound finite upper bound, nodes visited,
  and an explicit non-exhaustive status. A budget exit is never zero/infeasible.
- Full n600 traversal must be streaming and compact: aggregate per-class and
  global histograms/sums of `log2(count)` bounds plus selected named strata;
  do not persist hundreds of millions of per-block rows. Every real n600 block
  must contribute either an exact count or an honest bound.
- Permit cached reuse only under a complete key containing integer
  coefficients, denominator, target numerator, selector identity, and pose
  plug-in identity. Never reuse by target alone.
- Report exact-count fraction, bounded fraction, zero lower-bound anomalies,
  min/median/tails of lower/upper `log2(count)`, and per-class summaries using
  the real cached target class. Distinguish scorer pixels from RGB
  channel-blocks in every count.
- Include boundary-annulus/fragile and degenerate strata. A named positive
  control must prove that a naive corner-only or budget-as-infeasible
  implementation would issue a false certificate.

## Min-description and RD contract

- Enumeration accepts a deterministic description-cost model and returns the
  cheapest feasible point, with stable tie-breaking. It must not silently use
  Euclidean/min-norm preference.
- The default cost is receiver-public and source-independent (for example a
  fixed or causal already-decoded predictor plus a precisely specified signed
  residual code). Any heuristic local code length is labeled as such.
- Actual candidate streams report raw bytes plus at least zlib and Brotli
  encoded byte counts, including headers/termination used. Order-0 entropy may
  appear only as a labeled optimistic lower bound.
- RD rows state the exactness-insistence rule, selected block count, actual
  frozen-SegNet mismatch count/d_seg when scored, raw bytes, coded bytes,
  cache/receiver scope, and axis. No head-margin proxy may be reported as
  d_seg. If the governed n600 scorer is not run, rows remain
  `NO_VERDICT_SCORER_CUSTODY`.

## Pose-ready interface

- Define a typed, deterministic pose-feasibility plug-in/protocol whose
  identity is persisted. It can accept/filter a candidate and optionally add
  a description cost or diagnostic. The no-op Seg-only plug-in is explicit.
- Cardinality and min-description selection operate over the intersection
  after plug-in filtering. A plug-in budget/error cannot be converted into
  infeasibility.
- Tests prove no-op parity, a synthetic pose tube that shrinks a feasible set,
  deterministic intersection selection, and fail-closed malformed/nonfinite
  plug-in results. This is interface custody only; do not claim the banked R1
  pose is wired or factor 10 is solved.

## Acceptance tests

1. Focused tests pass under `/Users/adpena/Projects/pact/.venv/bin/python`.
2. A frame-195 fixture proves the new extractor continues when generic float64
   power scoring disagrees but cached live logits are bit-identical.
3. Resume is byte-identical to a fresh tiny-cache run and refuses changed
   source/config/slice bytes.
4. Exhaustive counts match brute force on small lattices; budgeted results
   enclose the brute-force truth and never emit false infeasibility.
5. A positive fixture has multiple feasible points and the min-description
   winner differs from a minimum-norm/nearest point while preserving the exact
   integer equation.
6. Pose plug-in tests above pass.
7. Local smoke uses at most a tiny honest subset. No heavy n600 launch occurs
   inside the implementation dispatch.

## Constraints and do-not-touch list

- No Fourier/DCT/rFFT construction.
- Do not touch
  `experiments/results/levelset_n600_witness_20260717T113932Z`.
- Do not mutate `/Users/adpena/Projects/pact`, another worktree, the previous
  blocked SSD cache, frontier pointers, live-run directories, or upstream.
- Do not launch training, GPU, paid work, full n600 scorer work, or a daemon.
- Do not invent score/rate/DOF numbers. Tests may use clearly synthetic values.
- Do not create bulky artifacts locally. Tests use temporary directories and
  success-only scratch cleanup.
- Preserve MAIN landing review as mandatory.

## Measurement and documentation after implementation

The parent agent, not the implementation dispatch, owns governed measurements,
the Round-1 confound hunt, factor-2/6/10 completeness-matrix three-way update,
DAG FEED, equation-candidate JSONL, review seals, serializer commit, and final
manifest/handoff.
