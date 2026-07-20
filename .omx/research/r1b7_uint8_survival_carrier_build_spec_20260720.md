# R1b7 uint8-survival carrier — build specification

Date: 2026-07-20  
Lane: `r1b7_uint8_survival_carrier`  
Authority: delegated R1b7 prompt SHA-256 `cf2e4a374583a836b41a9bf56cb154e847743df2786031a377ea31cd1a4f3401`  
Execution class: local, research-only, `[macOS-CPU advisory]`

## Objective

Turn the sealed R1b6 n16 negative into a receiver-bound mechanism by tracing all
498 exact-feasible Fisher-ordered sites through the actual stages:

`encoder real target -> emitted uint8 camera bytes -> exact four-tap resize -> SegNet stem -> final winner/rival head margin`.

The mandatory output is a mutually exclusive site histogram with explicit
counts for uint8 loss, resize dilution, head/wrong-rival loss, scheduled-site
survival, and collateral damage. The hard-oracle batch geometry remains seed
1234 / batch 16. The pointer remains `0.1910828242 [contest-CPU] UNMOVED`.

## Premise falsification before reformulation

R1b6's `_signed_rounding_block` and `_source_closest_block` were copied from
the measured R2b implementation at commit `98515407bd`. Therefore “R2b fixed
magnitude” may be byte-identical to the already-measured R1b6 replay rather
than a distinct reformulation. The implementation must compare the signed
target numerators, emitted blocks, replay bytes, and sealed candidate hash.
Equivalence is a measured formulation result, not permission to relabel the
R1b6 negative as a fixed-magnitude family verdict.

If the equivalence check is false, build and hard-score the distinct fixed-
magnitude arm on the same 498 sites. If it is true or the distinct arm is not
positive with margin, escalate only a bounded highest-EV sub-prefix to an
integer-lattice solve that requires a representable camera-byte change, exact
four-tap projection, and positive target-vs-rival head-margin movement before
hard-oracle admission.

## Owned files

- `tools/measure_r1b7_uint8_survival_carrier.py`
- `tools/tests/test_measure_r1b7_uint8_survival_carrier.py`
- `.omx/research/r1b7_uint8_survival_carrier_<UTC>.json`
- `.omx/research/r1b7_uint8_survival_carrier_<UTC>.md`
- `.omx/research/r1b7_uint8_survival_carrier_DAG_FEED_<UTC>.md`
- one append-only refinement row for equation ID
  `realization_breakeven_bytes_v1`, only if a newly measured positive arm
  changes its domain
- the lane-registry/audit rows created only through `tools/lane_maturity.py`

No trainer, DSL, upstream snapshot, scorer weights, live run, pointer, or
submission artifact is in scope.

## Required implementation behavior

1. Verify custody hashes for both sealed R1b6 archives, their decode receipts,
   the R1b6 result receipt, Fisher ordering, target raw, decoder, and scorer.
2. Decode the sealed baseline and candidate twice and require bit-identical raw
   outputs. Use the SSD waterfall and certify all success-only raw scratch
   before deletion.
3. Parse R1K1 replay writes and reconstruct the intended signed rational
   endpoint from the exact resize operator. Compare intended absolute writes
   with emitted uint8 camera bytes before attributing any site to uint8 loss.
4. Capture SegNet input and stem tensors plus final logits with deterministic
   CPU Torch. Use the target label and the current maximum non-target rival for
   margins; record rival identity changes rather than collapsing them.
5. Emit per-site stage records and aggregate histograms. All death buckets must
   be mutually exclusive and sum to 498; collateral counts may be an explicit
   overlay but cannot be silently folded into successful scheduled flips.
6. Measure actual replay/archive bytes and bytes per selected site. Waterfill in
   the existing Fisher/necessity order and stop when measured marginal recovery
   does not pay `25/37_545_489` score-units per byte.
7. Never run n600 unless the n16 hard-oracle recovery is positive with a stated
   margin. Do not optimize `xi0`; pose is observation-only for joint/plane
   proximity in this lane.

## Acceptance tests

- deterministic exact-resize projection agrees with Torch bilinear input to a
  bounded numerical tolerance and never substitutes rounded scorer bytes;
- R2b/R1b6 constructor equivalence is tested against a representative exact
  block and against the sealed replay when custody is present;
- stage-classification buckets are exhaustive and mutually exclusive;
- wrong-rival changes are separately counted;
- candidate replay parser round-trips canonically;
- hard oracle refuses batch size other than 16;
- receipt output is atomic and refuses overwrite;
- success-only scratch cleanup includes path, bytes, SHA-256, deterministic
  rebuild command, and reason before deletion;
- Ruff, `py_compile`, focused tests, JSON parse, and `git diff --check` pass;
- both Python files receive two clean `review_tracker` passes after the final
  diff; serializer commits declare post-edit SHA-256 and no co-author trailer.

## Verdict scope and landing boundary

Any negative is limited to the exact sealed R1b4 n16 receiver, the 498
Fisher-ordered sites, the measured signed-block or bounded lattice formulation,
and the local CPU Torch scorer. It is not an n600, contest-axis, curvelet,
shearlet, boundary-carrier, full-kernel, or family verdict. MAIN must review the
branch diff and custody claims before merge.

## Triality disposition

- DSL: N/A; encoder-side measurement API only, no flag invented.
- DAG: a dated FEED will expose every passed and blocked edge.
- Equations: consume `realization_breakeven_bytes_v1` by ID and append only a
  scoped empirical refinement when newly measured positive recovery exists.

