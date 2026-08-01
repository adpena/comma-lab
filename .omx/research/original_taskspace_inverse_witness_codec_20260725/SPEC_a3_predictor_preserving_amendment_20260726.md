# A3 predictor-preserving amendment

Status: append-only amendment to
`SPEC_a3_conditional_pose_preimage_codec_20260726.md`; `research_only=true`.
This amendment changes the receiver/control ordering after executable P/G
composition exposed a signal-loss premise that the earlier specification did
not model.  It makes no candidate, score, originality, promotion, or frontier
claim.

## Newly established premise

The counted ep725 predictor already decodes exact uint8 camera frames.  The
legacy G/A bridge discarded those frames by painting the complete semantic
grid from five palette colours.  That operation erased decoder-owned RGB and
temporal signal even where neither G nor A owned a correction.  Therefore a
full-frame palette materialization cannot remain the A3 control or the input to
the conditional pose solve.

The exact camera-to-scorer resize has disjoint supports.  A receiver can keep
the predictor camera bytes everywhere outside an explicit owned scorer-cell
set, replace only the four camera taps for each owned cell, and prove both:

1. every unowned camera byte is identical to P; and
2. every unowned scorer numerator is identical to P.

This is the reconstruction-residual coordinate that was missing between the
semantic grammar and the camera witness.

## Amended finite receiver/control ordering

The earlier SE3 and basis modes remain owed A3 actuator families.  Before
them, the receiver must expose two predictor-preserving controls:

1. `P0_PASS_THROUGH`: exact ephemeral P frame 0 is emitted unchanged.  It has
   zero A-owned camera/scorer cells and is the mandatory signal-preserving OFF
   arm.  The counted packet still binds P, G, corrected camera Y1, pair order,
   and its exact own bytes.
2. `SPARSE_EXACT_LATTICE_Y0_RESIDUAL`: counted canonical scorer-cell ownership
   plus target RGB/numerator values are applied over exact P frame 0.  Duplicate,
   overlapping, out-of-window, and no-op aliases are refused.  This stream is
   part of A, rather than terminal T, only when its source binding proves that
   it changes Y0 conditionally on the exact corrected Y1 and leaves every other
   P/G byte and Y1 camera byte unchanged.
3. `SE3_XI_WARP_V1`, in both preregistered warp domains from the parent spec.
4. `SE3_XI_PLUS_BASIS_V1`, only after an exact finite score transition admits
   the residual basis at its recompressed outer-archive price.

The predictor-preserving packet is an executable receiver seam, not by itself
completion of the SE3/basis A3 specification.  Its receipt must keep the
inverse-solve, scorer, n600, and exact-evaluation gates false.

## Semantic ownership is not reconstruction ownership

G currently owns cells where `decoded_g_label != predictor_internal_label`.
That is sufficient for semantic relabels but not for all realization repairs.
A predictor internal label can already equal the desired class while its
decoded RGB, resize, uint8, and frozen-scorer path still lies on the wrong side
of the evaluator boundary.  A complete G packet therefore needs two distinct
coordinates:

- semantic relabel ownership; and
- same-class realization-strengthening ownership.

The second coordinate may reuse the generic exact sparse lattice overlay
mechanism, but it must have a distinct counted G binding and may not be inferred
from label inequality.  Until that stream exists, P/G composition receipts
must say that target through-R debt is open even when the semantic-label overlay
is camera-closed.

## Coupled admission law

Neither A mode nor G repair is admitted by an independent Seg, Pose, or byte
cap.  Every measured proposal supplies one complete after-state
`(d_seg, d_pose, archive_bytes)` to the canonical nonlinear score-transition
audit under a freshly reopened dynamic frontier snapshot.  The returned
conditional boundary coordinates are telemetry on one score surface, never
three stand-alone thresholds.

## Triality and no-orphan wiring

- DSL: distinct P0 pass-through, sparse Y0 residual, semantic-G ownership, and
  same-class realization-G ownership; exact P/G/Y1 foreign keys on every A
  packet.
- DAG: exact P camera pair -> semantic G -> reconstruction G -> exact corrected
  Y1 -> conditional A over exact P0 -> chronological `(Y0,Y1)` -> R -> coupled
  score.
- Equations: disjoint-support numerator conservation, conditional preimage
  `Y0 | exact Y1`, and the full nonlinear Seg/Pose/rate finite transition.

The sparse residual exposes byte and sensitivity rows to the bit allocator;
the SE3 and sparse modes remain separate probe interpretations; empirical
results update the canonical posterior rather than a one-off memo.

## Stores consulted

- `CLAUDE.md`, `AGENTS.md`, and the parent A3/G2/G3 specifications;
- `taskspace_predictor_state_v2`, the ep725 exact ephemeral runtime surface,
  the V2 G/A consumer seam, and the predictor-preserving G overlay;
- the strict monolithic P/G/A receiver and exact outer STORE/DEFLATE codec;
- `tac.score_geometry` coupled sublevel and finite-transition audits.

HISTORICAL_PROVENANCE: new executable evidence supersedes only the lossy A0
receiver premise and the earlier default placement of a causally bound sparse
Y0 residual in terminal T.  The SE3, warp-domain, inverse-solve, basis, n600,
CPU/CUDA, and promotion obligations remain open.  Pointer delta is zero.
