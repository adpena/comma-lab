# G8 counted same-class realization-repair subgrammar

Status: implemented structural receiver coordinate; `research_only=true`.
Lane: `lane_g8_same_class_realization_repair_20260726`.
Checkpoint root: `g8_same_class_realization` in
`.omx/state/subagent_progress.jsonl`.
Base commit at implementation: `08279e376636189a2d93b3fe79092e56409dd2b2` on a
shared dirty `main`; this unit owns only the module, focused test, and this
specification.

## Exact scope and non-claim

The semantic G receiver can produce the desired class label while its corrected
uint8 camera Y1 still realizes the wrong class after camera resize, uint8, and
the frozen evaluator.  Label equality therefore does not eliminate
realization debt.  `G8R1` adds a distinct counted G subgrammar for that case:
horizontal scorer-cell runs with one shared uint8 RGB value.  It is not an A3
row alias because it repairs corrected Y1 while preserving the already-decoded
G semantic labels exactly.

This landing does not invoke a scorer and has no observed through-R after-label.
Accordingly:

- `coordinate_present=true`;
- `through_r_target_realization_debt_closed=false`;
- `scorer_invoked=false`;
- `score_claim=false`, `candidate_claim=false`, `exact_eval_invoked=false`,
  `promotion_eligible=false`, and `originality_claim=false`;
- `research_only=true`.

No score improvement, archive candidacy, n600 verdict, evaluator equivalence,
or frontier movement follows from a structural camera/numerator mutation.

## Versioned counted wire grammar

The packet magic/version is `TACG8R1\0` / `1`.  Its fixed 222-byte big-endian
header directly carries:

1. the exact contiguous source-pair start/count and the hash of their complete
   source-binding record;
2. the SHA-256 of the exact corrected-Y1 uint8 camera bytes;
3. the exact G semantic-binding SHA-256;
4. the SHA-256 of the exact current G semantic-label bytes;
5. the canonical counted P-section identity SHA-256;
6. the canonical counted G-section identity SHA-256;
7. run count, body bytes, packet bytes, and a CRC-32 over the header with zeroed
   CRC plus the exact body.

A counted P/G section identity includes the closed role, section index, codec
ID, payload byte count, and payload SHA-256.  P and G identities must be
distinct.  When G8 is nested in the composite G section described below, the G
foreign key denotes the exact inner semantic-G bytes; the enclosing monolithic
directory separately binds the complete composite-G bytes, avoiding a circular
self-hash.  Decoder reconstruction from the live source object must reproduce
every digest in the packet; stale camera bytes, labels, semantic binding, P
identity, inner-G identity, or pair window fail before mutation.

Each canonical 13-byte run is:

`(source_pair_id, scorer_row, col_start, col_stop, semantic_class, semantic_role, R, G, B)`.

Columns are a nonempty half-open horizontal interval.  Runs are sorted by exact
address, unique, and non-overlapping.  Adjacent runs with the same class, role,
and RGB are rejected as a noncanonical split and must merge.  The semantic role
redundantly determines the five-class ID; a class/role mismatch fails closed.
There is no empty/pass mode: deletion is not a no-op coordinate.

`TACG8S1\0` / version 1 is the concrete counted composite-G envelope for the
existing monolithic G role.  Its 92-byte header binds exact inner semantic-G
and G8R1 lengths and SHA-256 values, total bytes, and CRC-32.  Its body is
exactly `semantic_G_packet || G8R1_packet`; truncation, trailing bytes, hash/CRC
drift, wrong inner magic, and noncanonical re-encoding fail.  This gives the
whole-object allocator one concrete replacement
`generative_correction_packet`, rather than an unpriced sidecar.

## Compile-side admission and payload-lineage boundary

`EncoderOnlyExactTargetLabelCustodyV1` binds the source artifact hash, member
name/hash, exact pair IDs, target-label byte hash/shape, and the immutable
target-label array.  It has no serialization method and fixes
`serialized_target_bytes=0` under lineage policy
`exact_target_labels_encoder_only_never_serialized.v1`.

For every expanded run cell, compile admission requires both:

`current_G_label[p,y,x] == run.semantic_class`

and

`target_label[p,y,x] == current_G_label[p,y,x]`.

Thus G8 cannot smuggle a semantic relabel through a camera-repair row.  Target
labels and their custody binding affect only the compile admission receipt;
the public decode API accepts only packet bytes and the live source surface.
No target-custody field or target-derived payload, frozen scorer output, label
grid, margin field, dense RGB target, or scorer state is serialized into the
packet.  The packet does bind the current G-label hash; when target and current
labels are byte-identical their digests naturally coincide, but the target
identity is not a receiver input or separate wire field.  A focused matched
test changes target labels outside the admitted run and proves that packet
bytes and decoded camera bytes remain identical while the compile-only custody
binding changes.

Selection of useful repair runs remains an encoder-side inverse problem.  The
compiler validates class-preserving eligibility; it does not manufacture a
through-R success claim from that eligibility.

## Receiver and exact preservation proof

The receiver expands each run to canonical `SparseConstantRGBCellV1` cells and
calls the shared
`apply_disjoint_camera_cell_reconstruction_v1` implementation from
`predictor_preserving_coupled_preimage.py`.  There is no second resize/lattice
implementation in G8.  That canonical kernel supplies exact disjoint camera
support ownership and proves:

- every owned scorer numerator equals `shared_rgb * scorer_denominator`;
- every unowned camera byte is byte-identical to corrected Y1;
- every unowned scorer numerator is integer-identical to corrected Y1;
- duplicate/overlapping/foreign/no-op cell rows fail closed.

The receiver copies the current G semantic-label bytes without mutation and
requires every parsed run class to equal the packet-bound live G class before
camera mutation, then requires the output-label hash to equal the input-label
hash.  It verifies the canonical helper's exact constant-RGB proof flag, derives
ownership counts from the returned masks, and closes the exact scorer ownership
mask against the packet program.  It decodes twice and compares camera, labels,
ownership masks, and receipt.  Packet parse/re-encode and canonical receipt
parse/re-emit are exact.  Packet absence, stale CRC/length mutation,
truncation, and trailing bytes fail closed.  Matched valid re-encoded RGB
mutation changes camera output; deleting one row from a multi-run program
changes camera output, while deleting the sole run yields the forbidden empty
program.  These are causality statements, not the false claim that every
semantically valid payload mutation must fail.

## Required falsifier and byte measurement

The focused synthetic falsifier constructs a cell where:

- current G semantic class is Road (`0`);
- exact encoder target class is also Road (`0`);
- a separate synthetic realized-label oracle says Lane (`1`).

One counted G8 run changes exact corrected-Y1 camera bytes and its owned scorer
numerator while decoded semantic-label bytes remain bit-identical.  This proves
the coordinate exists and is causally consumed; because the synthetic oracle is
not rerun through the real scorer after repair, it does not close realization
debt.

The same fixture measures one 64-cell horizontal run:

| representation | header bytes | body bytes | total bytes |
|---|---:|---:|---:|
| G8R1 one shared-RGB run | 222 | 13 | 235 |
| expanded A3 per-cell RGB rows | 62 | 576 (`64 * 9`) | 638 |

The exact run-body saving is 563 bytes and the full-packet saving is 403 bytes
for this structural fixture.  These are counted packet-byte comparisons, not
an outer recompressed archive delta, score delta, or rate-optimality verdict.
Outer monolith pricing remains with the whole-archive allocator after G8 is
wired into the monolithic receiver.

## Concrete P/G/A integration API and remaining seam

The standalone public handoff is:

1. construct `SameClassRealizationRepairSurfaceV1` from the exact corrected Y1,
   current G semantic labels/binding, pair IDs, and counted P/G identities;
2. construct compile-only `EncoderOnlyExactTargetLabelCustodyV1`;
3. express selected cells as canonical `SameClassRealizationRepairRunV1` rows
   inside `SameClassRealizationRepairProgramV1`;
4. call `compile_same_class_realization_repair(...)` to obtain exact packet,
   source binding, decoded result, runtime receipt, and compile admission
   receipt;
5. at receiver time call
   `decode_same_class_realization_repair_packet(packet, surface=surface)` and
   feed `decoded.camera_y1` forward while retaining
   `decoded.semantic_labels` unchanged.

The concrete whole-object proposal handoff is
`compile_same_class_realization_repair_for_pga_sections(...)`.  It accepts exact
P and inner semantic-G section bytes plus a typed
`PredictorCameraPairSurfaceV1`.  The P bytes must equal that surface's exact
predictor program, and the surface must carry upstream strict-decode receipt
custody.  The integration builder strictly parses and applies the inner G
packet against that typed predictor state, derives the predictor-preserving
camera overlay internally, and validates the resulting P/G/overlay custody
through the canonical A3 source-binding constructor.  It does not accept a
caller-supplied corrected Y1, label grid, or semantic-binding hash.  Magic-only
P/G prefixes and malformed-but-prefixed G packets are rejected.  Only after
that derivation does it construct both counted section identities, compile G8,
and return `SameClassRealizationRepairPGAProposalV1`.  Its
`replacement_g_section_payload` is the exact `TACG8S1` composite bytes that a
`TaskspaceSectionBundleV1` proposal substitutes for
`generative_correction_packet`; its `post_g8_corrected_y1` is the exact camera
surface conditional A must consume.

The monolithic receiver integration branch is explicit:

1. `parse_same_class_realization_repair_g_section(...)` recovers exact inner
   semantic-G and counted G8R1 bytes;
2. call `decode_same_class_realization_repair_from_pga_sections(...)` with the
   exact P bytes and typed predictor surface; it strictly parses/applies inner
   G and derives and validates the predictor-preserving overlay internally;
3. decode G8R1 against that exact derived surface and require its counted
   inner-G identity and source binding to close;
4. feed `post_g8_corrected_y1` to conditional A and require A's source binding
   to name that post-G8 camera hash;
5. decode the complete object and let the whole-object allocator recompress and
   price the resulting P/composite-G/A bundle.

The existing sibling-owned monolithic receiver still accepts bare `TACG1C` and
binds A3 to pre-G8 overlay Y1, so the proposal truth pins
`monolithic_receiver_branch_required=true` and
`existing_a3_rebind_to_post_g8_y1_required=true`.  Those are exact integration
blockers, not missing proposal bytes: the composite bytes and receiver transform
now exist.  This isolated unit does not edit the sibling receiver or pretend the
baseline's current score already includes G8.

## Acceptance and freeze evidence

Focused command:

`.venv/bin/pytest -q src/tac/witness_dsl/tests/test_taskspace_same_class_realization_repair.py`

Compatibility command:

`.venv/bin/pytest -q src/tac/witness_dsl/tests/test_taskspace_same_class_realization_repair.py src/tac/witness_dsl/tests/test_predictor_preserving_coupled_preimage.py`

Static checks:

`.venv/bin/ruff check src/tac/witness_dsl/taskspace_same_class_realization_repair.py src/tac/witness_dsl/tests/test_taskspace_same_class_realization_repair.py`

`.venv/bin/ruff format --check src/tac/witness_dsl/taskspace_same_class_realization_repair.py src/tac/witness_dsl/tests/test_taskspace_same_class_realization_repair.py`

Final freeze results after both adversarial reviews, including the integration
review's strict typed-P/parsed-G lineage correction:

- focused: `16 passed in 2.25s`;
- focused plus A3 compatibility: `25 passed in 4.20s`;
- Ruff check: `All checks passed!`;
- Ruff format check: `2 files already formatted`;
- module `compileall`: success;
- module SHA-256:
  `4e892c4aa0fab1dce65df54da96181511ecaff6dc7a84fff9876d8bd26060693`;
- test SHA-256:
  `5febce1712c2c98bc080e63903809b3a8b6dc9a84921e6cc98a5aaa05170e759`.

The deterministic 64-cell fixture freezes packet byte count `235`, packet
SHA-256
`5de4c133208f30425134906476ce99c12e9569e8d47a093483f5ca8208914693`,
source-binding SHA-256
`743878b2e397c50c89307ddabe2396686ad86ae3f0e388d906f6f024fda78c03`,
decode-receipt SHA-256
`19b57d3bf6e5800fa33fce50e571999fda48c965b5ace4bce60f40fcf0ce8672`,
and compile-admission-receipt SHA-256
`1e89b8d613b4d721aa6000aed43853dc77f8c5fd8da2473a57a0587581dfb8a2`.
It changes all `768` owned camera values at scorer denominator `786432` while
keeping the packet-bound G-label SHA-256
`3381de4ca9f3a477f25989dfc8b744e7916046b7aa369f61a9a2f7dc0963ec9e`.

## Triality delta

- DSL: distinct `G8R1` horizontal same-class repair runs with direct
  corrected-Y1/G/P/G foreign keys and compile-only target custody.
- DAG: `P -> semantic G -> corrected Y1 -> counted same-class G8 repair -> A ->
  uint8 pair -> R -> scorer`; the target-label edge terminates at the encoder.
- Equation: for each owned cell `c`, the canonical disjoint lattice enforces
  `N_out(c) = denominator * rgb(run(c))`, while for all unowned cells
  `N_out = N_in` and for all unowned camera taps `Y1_out = Y1_in`; semantic
  labels satisfy `L_out == L_in == L_target` only on admitted repair cells.

## Stores consulted

- `CLAUDE.md` / `AGENTS.md` byte-identical full contract and `PROGRAM.md`;
- craft handoff and vehicle operating-system manuals;
- current top-ten project memory anchors, canonical lane registry, subagent
  progress ledger, and latest dated DDM directive;
- A3 generic disjoint reconstruction implementation/tests, V2 predictor-state
  semantic binding, encoder-only teacher evidence, G3 label-local amendment,
  A3 predictor-preserving amendment, and G2 receiver composition spec.

HISTORICAL_PROVENANCE: first isolated executable implementation of the distinct
counted same-class realization-repair coordinate explicitly left owed by the
A3 amendment.  It neither alters semantic G nor closes the scorer-recursive
realization/evaluation remainder.
