# A3 conditional pose-preimage codec specification

## Objective and correction to the current stack

`A` is not a Pose6 sidecar and it is not an independently acceptable scalar
target.  It is the counted conditional program

`A : exact corrected Y1 -> Y0`

whose receiver emits the chronological pair `(Y0,Y1)` and whose value is the
finite change in the complete contest objective after the exact archive is
recompressed.  The current A2 translation plus RGB-delta grammar is a sealed
structural control, not an expressivity claim.  Treating that one grammar as
the whole pose solution would repeat the arbitrary-coordinate mistake that the
coupled score geometry was built to remove.

This lane remains `research_only=true`.  It builds the encoder/receiver and a
bounded same-object measurement.  It does not claim a candidate, score,
frontier movement, promotion, or proof that training is necessary.

## Recovered evidence that fixes the design

1. The frozen evaluator consumes both frames through the same camera-to-scorer
   resize before PoseNet's YUV6 transform.  The factor-2 lattice receiver has
   already realized arbitrary scorer planes through uint8 with exact integer
   numerators.  On the settled n600 replay it left 114 Seg tie-class pixels,
   not a general realization wall.  Therefore A should describe a compact
   scorer-visible conditional plane or transform; it must never serialize a
   dense camera frame.
2. The quantization-aware free-frame0 inverse solve established an existence
   result: a uint8 frame0 can drive Pose error nearly to zero for a fixed
   frame1.  Its dense/coarse image-space stores are rate-prohibitive at n600.
   The old n3 rate calculation that called a 96x128 carrier cheap was retracted
   because it omitted the factor of 600.  Reuse the solver and its Jacobian as
   an encoder oracle only, never its optimized pixels as payload.
3. The existing store-nothing pose carrier already supplies the right generic
   mechanism: counted quantized SE(3) xi, deterministic derive-H, and a generic
   inverse warp.  Its historical wrapper used a level-set render as the source.
   A3 re-homes that mechanism under A and conditions it on the exact corrected
   Y1.  It does not call xi a PoseNet output and does not duplicate a transport
   owner in P or G.
4. Pose authority is one pooled norm over all 600 pairs, not 600 independent
   hard balls.  Per-pair errors remain telemetry and acquisition coordinates;
   they are not admission vetoes.  The scalar `d_pose=2.5e-4` crossover is a
   derivative-coordinate identity, not a target or feasibility boundary.
5. Existing V9/V10 measurements show that low-dimensional boundary/event
   corrections can improve Seg and Pose jointly, but the old bbox/cubic/c3
   vocabulary saturated.  Their strict parsers, Fisher acquisition,
   transported events, exact lattice realizer, and matched controls are
   reusable mechanisms.  Their archives, fixed Pose6 ownership, and measured
   endpoints are not candidate payloads for this stack.

## Closed A3 mode universe

Preserve the A2 packet and decoder byte-for-byte as the mandatory control.
Introduce a distinct versioned A3 packet rather than silently widening A2.
Its canonical modes are:

1. `INTEGER_SHIFT_RGB_V1`: the existing A2 global translation plus global or
   per-role RGB delta.  This is the OFF/control arm.
2. `SE3_XI_WARP_V1`: one counted XIP2-compatible quantized xi row per source
   pair, entropy-coded jointly across pair order.  The receiver derives H with
   the existing generic arithmetic and warps exact corrected Y1 to Y0.
3. `SE3_XI_PLUS_BASIS_V1`: the same xi warp followed by a counted
   low-dimensional residual in a preregistered generic basis.  The initial
   basis set is the existing DCT/low-rank probe family and the governed
   curvelet/shearlet family.  This mode is admitted only after the pure-xi arm
   has a measured remaining Pose marginal worth more than its exact added
   bytes.

No dense frame, dense scorer plane, target Pose6 table, scorer output, target
label table, per-pixel optimized residual, or hidden pair-indexed decoder
constant is a legal A3 section.  An optional sparse exception stream belongs
to terminal T unless a matched counterfactual proves that it is causally part
of the A conditional map.

## Receiver law

For each pair, execute exactly:

1. strict outer archive parse and exact P/G/A section slicing;
2. P decode and transport-qualified G application;
3. exact corrected scorer-grid Y1 realization to uint8 camera bytes;
4. verify the A packet's source-pair order, P/G foreign keys, exact Y1 hash,
   xi/basis section hashes, and canonical parse/re-encode identity;
5. derive H from counted xi using the bound camera/normalized-coordinate
   contract;
6. warp the exact corrected Y1 source, then apply an admitted basis residual;
7. emit uint8 camera Y0 followed by the unchanged exact camera Y1;
8. run the real R parse-back and bind scorer-plane hashes.

The warp domain is an explicit packet discriminator.  Ship both defensible
interpretations until measurement arbitrates them:

- `CAMERA_THEN_R`: warp the exact camera Y1 and let the evaluator resize Y0;
- `SCORER_THEN_FACTOR2`: conjugate H into scorer coordinates, warp scorer Y1,
  then use the exact factor-2 realization for Y0.

`tools/probe_a3_warp_domain_disambiguator.py` compares these modes on identical
xi bytes.  One must not be chosen from intuition.  Both must preserve Y1
bit-for-bit, and a change to A must change only Y0 and the A section/outer ZIP
bytes.

## Encoder-only inverse solve

The encoder may load the frozen source video, target Pose outputs, scorer
weights, exact C1 planes, and dense free-frame0 solutions.  None may cross the
archive boundary.  It fits A3 controls in this order:

1. initialize xi from the existing calibrated SE(3) mechanism;
2. refine quantized xi directly against the frozen pooled Pose norm through
   the real receiver, uint8, and R operators;
3. reuse `pe_free_solve`, `_jacobian6`, `warp_base_fit`, and
   `pf_generic_compress` only as encoder-side proposal/oracle callables;
4. project the dense oracle difference onto each preregistered generic basis;
5. serialize, recompress the complete monolithic archive, decode twice, and
   measure exact same-object Seg/Pose/rate deltas;
6. admit a finite update only through
   `tac.score_geometry.score_transition_audit` against a freshly reopened
   typed dynamic-frontier snapshot.

The optimizer uses the exact pooled pose finite difference.  Per-pair tail
statistics may alter acquisition order, but no per-pair fixed cap and no
independent Seg/Pose/rate threshold can reject an otherwise lower complete
score.

## Bounded experiment matrix

After the ep725 P adapter and in-band G/A receiver are sealed, run n2 for
mechanism integrity and then a preregistered hard-tail-first n24 for local
advisory discrimination:

| arm | counted A content | purpose |
|---|---|---|
| A0 | A2 shift/RGB | byte-identical structural control |
| A1c | XIP2 xi, `CAMERA_THEN_R` | smallest physical conditional warp |
| A1s | same xi bytes, `SCORER_THEN_FACTOR2` | warp-domain disambiguator |
| A2d | winning xi arm plus DCT/low-rank prefixes | reuse the prior inverse-solve compression probe |
| A2c | winning xi arm plus governed curvelet/shearlet prefixes | localized residual control |

Every row records exact P/G/A section bytes, recompressed outer ZIP bytes,
chronological raw hashes, pooled and per-pair Pose telemetry, Seg collateral,
runtime, and the dynamic target snapshot hash.  Rows remain advisory until the
full n600 and exact CPU/CUDA gates in the G3 specification close.

## Acceptance and stop rules

- A1 is retained only if it changes decoded Y0, preserves Y1 exactly, and has
  a negative exact complete-object score delta against A0.
- A basis prefix is retained only if its finite pooled Pose/Seg benefit exceeds
  its measured outer-archive byte cost.  Remeasure after each accepted prefix;
  do not sum isolated marginals.
- If all deterministic A3 arms have nonnegative marginal value, record the
  exact unrepresented residual coordinates.  Only those coordinates may seed
  training or terminal T.
- Do not launch training merely because the A2 shift grammar fails.
- Do not reuse the historical pose-carrier wrapper, old V9/V10 archives, or any
  public archive bytes.  Reuse only reviewed generic algorithms and original
  in-repository mechanisms with new typed lineage.

## Triality and no-orphan wiring

- DSL: versioned A3 packet, warp-domain discriminator, counted xi, optional
  basis prefix, exact P/G/Y1 foreign keys.
- DAG: encoder-only target -> quantization-aware inverse proposal -> counted A
  -> exact Y0 given exact Y1 -> chronological receiver -> exact coupled score.
- Equations: pooled Pose norm, exact finite score transition, H coordinate
  conjugation, and section/outer-byte conservation.

Hooks:

1. sensitivity rows bind each xi/basis coefficient to pair, A-section, and
   exact before/after scorer hashes;
2. Pareto admission consumes only complete-object finite transitions;
3. the bit allocator uses recompressed outer ZIP bytes and pooled Pose value;
4. cathedral/autopilot dispatches the first missing A3 matrix cell;
5. every measurement updates the continual-learning/probe ledger;
6. the warp-domain and basis-family disambiguators preserve all defensible
   interpretations until exact measurement selects one.

## Stores consulted

- `CLAUDE.md`, `AGENTS.md`, and the G2/G3 executable specifications;
- `tools/pose_frame0_inverse_solve_probe.py` and its corrected 2026-07-03
  harvest ledger;
- `tools/levelset_byte_close_and_eval.py` store-nothing pose-carrier and exact
  NumPy receiver paths;
- `tac.boundary_math.xi_pose_coder`, factor-2 lattice realization, score
  geometry, V9/V10 structured carrier, and coupled-preimage implementations;
- the n600 V10 lattice, V9/V10 carrier-composition, exact C1 debt, target
  partition census, and G1 prior-signal receipts.

HISTORICAL_PROVENANCE: append-only executable specification.  Pointer delta is
zero; no candidate or score is claimed.
