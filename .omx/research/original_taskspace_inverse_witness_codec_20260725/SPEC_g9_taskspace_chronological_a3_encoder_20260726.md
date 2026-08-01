# G9 chronological A3 encoder and XIP2 acquisition seam

Status: L0 research-only executable specification, 2026-07-26.  This landing
produces concrete counted substitute-A packets and whole-object proposal
objects.  It does not invoke a scorer, establish a through-R result, create a
candidate archive, claim an exact score, claim originality, dispatch work, or
move a frontier pointer.

## Competitive state and exact object in hand

The current competitive target is not 0.191.  The reopened n2 control receipt's
official-leaderboard snapshot selects **0.172** as the planning target, while
the project goal remains sub-0.15.  Refreshing an external leaderboard row is
not local progress.

The exact local control available to this seam is the research-only ep725 n2
causal P/G/A object:

- archive: 191,838 bytes, DEFLATE,
  `sha256=2049367274334881a53782beba2c764a8c0896dd4681810085eebc56fbd6581a`;
- P: 84,536 bytes,
  `sha256=f0c3e648f00f52e48c7be98997fb7dd57c2e5a607ed385846931af68f88cc78c`;
- G: 341,316 bytes,
  `sha256=8d77818a154917b0a9581dc8815e8905dc9b358391d8166f3c2d9f107c46e6f7`;
- A control: 62-byte `PASS_P0_V1`,
  `sha256=5c6b8c25a6302575b439094b08745f19f2b9914c91575837a1cc2b2c30c2e79e`.

Its frozen-scorer measurement is only
`[macOS-CPU frozen-scorer advisory]`: `d_seg=0.0202229805290699` and
`d_pose=121.45529174804688` over two pairs.  These values show that semantic
debt closure alone did not preserve enough scorer signal.  They are not an
authority score or an n600 finding.  This landing leaves the pointer unmoved.

## Problem closed

The counted predictor-preserving A3 receiver already had executable packet
modes but no bounded encoder that could turn exact P/G/corrected-Y1 plus an
encoder-only target into finite proposal packets for the G7 whole-object
allocator.  `src/tac/witness_dsl/taskspace_chronological_a3_encoder.py` closes
that acquisition/compile seam.

For each exact scorer cell `c`, the encoder computes integer resize numerators
under the frozen disjoint-support operator and ranks the exact improvement

`gain(c) = ||N(P0,c)-N(T,c)||_2^2 - ||N(A(P0),c)-N(T,c)||_2^2`.

Only strictly positive integer gains enter a prefix.  Ties are deterministic:
gain descending, then pair, scorer row, scorer column.  This is an acquisition
ordering, never an admission rule.  The caller supplies a finite sorted set of
prefix cardinalities; the encoder deliberately does not invent independent
Seg, Pose, or rate thresholds.  G7 admits or rejects only after rebuilding and
measuring the exact whole object under

`100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489`.

## Current executable substitute-A modes

The acquisition exposes the exact 62-byte `PASS_P0_V1` control plus finite
prefixes of two existing receiver-closed modes:

1. `TARGET_CONSTANT_RGB_V1` compiles to
   `SPARSE_CONSTANT_RGB_V1`.  Each selected cell stores six coordinate bytes
   and three RGB bytes.  Its integer RGB triple is the nearest uint8 constant
   to the target numerator for that exact cell.
2. `CORRECTED_Y1_SUPPORT_COPY_V1` compiles to
   `COPY_CORRECTED_Y1_SUPPORT_V1`.  Each selected cell stores only six
   coordinate bytes and copies the already source-bound corrected-Y1 camera
   support into Y0.

Every compiled packet is strictly parsed twice and publicly decoded twice.
The public A3 decode itself retains its deterministic double-replay contract.
The encoder independently verifies that corrected Y1 remains byte-identical,
all nonselected P0 camera values remain byte-identical, and all nonselected P0
scorer numerators remain integer-identical.

These modes are concrete counted A packets.  They are not the full A3 SE3/XIP2
universe, not an inverse optimum, and not a candidate claim.

## Encoder-only evidence firewall

`EncoderOnlyA3TargetV1` holds dense target Y0 only in encoder memory and has no
serializer.  Its custody and content enter the acquisition receipt only through
a binding SHA-256.  The counted A packet and G7 transform capture none of:

- dense target RGB or scorer planes;
- target Pose values;
- scorer weights, logits, gradients, or observations; or
- target-derived lookup tables other than the explicitly counted selected row
  coordinates and, for the constant mode, three selected uint8 values.

The canonical encoder receipt fixes all evidence/authority labels closed:
zero serialized target/scorer/pose evidence bytes, no scorer invocation, no
through-R or n600 verification, no exact score, no rate optimality, no
candidate, no originality, no promotion, and `research_only=true`.  Its strict
parser rejects duplicate keys, noncanonical JSON, schema drift, arithmetic
drift, and truth-label relaxation.

## Exact G7 proposal seam

Every compiled packet carries a `TaskspaceWholeArchiveProposalV1`.  Its partial
transform captures exactly:

- the full `PredictorPreservingA3SourceBindingV1`; and
- the replacement counted A packet bytes.

At application time it checks that the current bundle's exact P and G hashes
match the source binding, strictly parses both the current A and replacement A
against that binding, replaces only A, and proves P, G, and optional T stayed
byte-identical.  Dense target evidence is not captured by the callable.  The
result is directly consumable by G7, which remains responsible for monolithic
rebuild, STORE/DEFLATE recompression, receiver replay, realized measurement,
and nonlinear accept/reject.

## Two XIP2 interpretations retained without a fake wire claim

`build_xip2_guidance_targets(...)` makes both parent-spec interpretations
callable from one identical XIP2 byte string:

1. `CAMERA_THEN_R`: warp exact corrected camera Y1 in native camera geometry,
   then let the exact scorer resize operate on the resulting uint8 camera
   target.
2. `SCORER_THEN_FACTOR2`: resize corrected Y1 to scorer space, warp there,
   round the explicit uint8 guidance control, then use the certified factor-2
   integer preimage to return to camera space.

Quantization and coding reuse `xi_pose_coder`; parsing and dequantization reuse
`SE3XiTransportV2`, including its independent exact-EOF parser and canonical
decoder agreement check.  Both guidance targets bind the same XIP2 payload
hash, source pair order, P/G/Y1 source binding, q-levels, coder, and external
pitch.

The current A3 packet does **not** embed XIP2.  The XIP2 container does not carry
pitch, and there is no current counted A3 discriminator/geometry ABI that would
let the receiver replay either domain from those bytes.  Therefore this helper
only creates encoder-side acquisition targets for the two existing sparse A
modes.  Calling either output `SE3_XI_WARP_V1` would be a fake implementation.

## V9/V10 and existing-substrate harvest

| Reused signal | Exact reuse here | What remains open |
|---|---|---|
| V9/XIP2 trajectory codec | quantize, coder body, exact XIP2 bytes, dequantized xi | counted A3 XIP2 mode and pitch/geometry ABI |
| V9 screw warp | deterministic native homography/warp | target-conditioned xi inverse selection |
| V10 integer realization | exact disjoint resize numerators and certified factor-2 preimage | arbitrary nonlinear scorer-cell preimage is not claimed |
| P/G chronological receiver | exact P0, corrected Y1, full source foreign keys | n600 streaming/composition beyond the bounded up-to-four-pair A3 packet |
| predictor-preserving amendment | PASS control and sparse owned supports | same-class G repair is sibling-owned and not duplicated here |
| G7 whole-object allocator | typed atomic A substitution proposal | empirical receiver/measurement callback and dynamic admission |

No historical/public archive, model weights, target table, or borrowed payload
is reused.  The reusable content is generic in-repository algorithm and typed
ABI only.

## Verification

Focused tests live at
`src/tac/witness_dsl/tests/test_taskspace_chronological_a3_encoder.py`.
They cover:

1. exact 62+9k and 62+6k packet byte arithmetic for concrete prefixes;
2. positive integer numerator-SSE ranking and receipt closure;
3. exact P/G drift refusal and P/G/T preservation by G7 transforms;
4. absence of target evidence from packets and transform captures;
5. two strict parses plus two public decodes for every control/proposal;
6. receipt authority-smuggling refusal and pair/prefix fail-closed behavior;
7. both XIP2 domains from one strict payload and explicit absence from current
   A wire bytes.

Focused result at landing: `6 passed`; Ruff: clean.  These are structural
synthetic proofs, not empirical score evidence.

## Autonomous next path

1. The parent materializer supplies the exact target/custody surface for the
   existing n2 P/G/Y1 object and compiles finite prefixes from both sparse
   interpretations, plus prefixes acquired from each XIP2 guidance target.
2. Feed those exact proposal objects to G7 in preregistered order.  For every
   trial, rebuild the monolith and both outer encodings, strict-receive twice,
   measure the exact selected object, and retain only negative total-score
   transitions.  Never use numerator SSE as score admission.
3. Compare the two XIP2 guidance domains on identical q bytes.  If either has
   finite whole-object value, implement a real counted XIP2 A3 receiver mode
   with a closed warp-domain discriminator and pitch/geometry descriptor,
   then repeat the same-object comparison.  If neither does, scope the negative
   to these two current formulations rather than killing SE3 or A3.
4. Record the remaining unrepresented residual after all positive inverse
   proposals.  Only that residual can justify training or a terminal quotient.
5. Lift the bounded packet path to full n600 composition, realized-through-R
   authority, and contest CPU/CUDA exact evaluation only after the receiver
   wire and same-class G repair are byte-closed.  Until then, pointer delta is
   exactly zero.

## Triality and stores consulted

- DSL: encoder-only target, finite prefix plan, exact ranked row, strict
  receipt, counted A packet, G7 proposal, two-domain XIP2 guidance.
- DAG: exact P/G/corrected-Y1 + encoder-only target -> positive integer row
  ordering -> strict counted A -> exact A-only bundle transform -> G7
  whole-object replay/measurement/admission.
- Equations: integer numerator conservation, conditional Y0 given exact Y1,
  exact section foreign keys, and the full nonlinear score transition.

Consulted: `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`, the v7.5/v8 operating specs,
the A3 parent/amendment, the G7 allocator spec/API, predictor-state V2/XIP2,
predictor-preserving A3/overlay, integer lattice realization, V9/V10 research
and inverse-solve memories, and the exact n2 control/measurement receipts.

HISTORICAL_PROVENANCE: append-only executable specification.  No candidate or
score is claimed; no frontier pointer moved.
