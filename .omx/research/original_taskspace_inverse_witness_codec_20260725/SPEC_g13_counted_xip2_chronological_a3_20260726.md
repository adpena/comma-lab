# G13 counted XIP2 chronological A3 packet and receiver ABI

Status: implemented and locally verified at L0, 2026-07-26.  This is a
`research_only=true` packet/receiver landing.  It does not run a scorer, build
an archive, dispatch work, perform an exact evaluation, claim a score, claim a
candidate, claim promotion, or move the frontier pointer.

## Literal problem

G9 made two XIP2-derived Y0 targets executable only on the encoder side.  Its
current counted A packet still stores sparse RGB/copy rows and explicitly says
that XIP2 is absent from the receiver grammar.  G13 closes exactly that missing
coordinate: a counted, source-bound XIP2 A packet whose generic NumPy receiver
replays either preregistered warp domain against the production PASS-semantic-G
conditional Y1 surface.

G10 owns `taskspace_post_g8_conditional_a.py` and
`taskspace_monolithic_pga_receiver.py`; G13 does not edit either file.  G12 owns
`taskspace_same_class_realization_encoder.py`; G13 does not edit it.  G9's
`taskspace_chronological_a3_encoder.py` is frozen and is imported only for its
encoder-only guidance type.  The new implementation and tests are isolated in
new G13 files.

## Closed causal object and non-aliasing source domain

The sole V1 production receiver input is G10's exact typed
`PassConditionalASurfaceV1`:

`exact P0 + nonempty PASS semantic-G envelope + explicit optional G8 + exact conditional Y1`.

The packet embeds a closed source-domain discriminator plus the SHA-256 of the
complete `PassConditionalASourceBindingV1`.  That canonical record covers the
selected `PassSemanticGEnvelopeMode`, exact P state/semantic/program/renderer/
surface/upstream-decode custody, P labels/P0/P1, PASS envelope and receipt,
inner nonempty PASS packet and receipt, semantic-label hash, pre-repair Y1,
optional G8 packet/receipt, resulting conditional Y1, and pair window.  Any
change makes decode fail closed.  `PASS_NO_G8_V1` requires absent repair hashes
and pre-repair Y1 equal to conditional Y1.  `PASS_THEN_G8_V1` requires both
repair hashes and a non-no-op conditional Y1.

The older exact-semantic-G `PostG8ConditionalASourceBindingV1` universe is a
diagnostic control owned by G10.  G13 V1 does not accept it, translate it, or
encode it.  The exact source type check and `PASS_G_CONDITIONAL_V1` wire
discriminator make the production and diagnostic domains non-aliasing.  A
future diagnostic-control adapter would require a distinct packet version or
source-domain value and separate tests; it cannot silently enter V1.

The active XIP2 packet also embeds G9's encoder-guidance binding SHA-256 as
opaque lineage metadata.  The receiver cannot reconstruct or verify that
encoder-only source universe, so the receipt fixes
`guidance_binding_role=OPAQUE_ENCODER_LINEAGE_ONLY`,
`guidance_binding_receiver_verified=false`, and
`guidance_binding_source_authority=false`.  Only the complete production
`PassConditionalASourceBindingV1` digest carries receiver authority.  Dense P0,
conditional Y1, target RGB, target labels, scorer state, and GT never enter the
packet.

The receiver returns exact bounded `uint8` arrays:

- `Y0`: PASS P0, exact global copy of conditional Y1, or the selected XIP2 realization;
- `Y1`: the supplied PASS-G conditional Y1, byte-for-byte unchanged; and
- chronological frames: exact `stack([Y0,Y1], axis=1)` for one contiguous
  pair window of at most four pairs.

## Packet V1

The new wire domain is `TACX2A3\0`, version 1.  All outer fields are big-endian;
the nested XIP2 payload retains its canonical little-endian XIP2 ABI.

Header fields, in order:

1. eight-byte magic;
2. packet version;
3. closed source-domain discriminator (`PASS_G_CONDITIONAL_V1` only);
4. closed mode discriminator;
5. closed interpretation discriminator;
6. closed geometry-profile discriminator;
7. pair start and pair count;
8. camera height/width and scorer height/width;
9. canonical finite IEEE-754 fp32 pitch;
10. 32-byte PASS-G conditional source-binding digest;
11. 32-byte encoder-guidance binding digest;
12. exact XIP2 body length; and
13. XIP2-body CRC32.

The body is either empty (PASS) or one exact EOF-closed XIP2 payload.  A
four-byte packet CRC32 over `header || body` is the sole footer.  The strict
parser checks exact total length before slicing, both CRCs, every closed enum,
frozen dimensions, bounded contiguous pair IDs, source binding, XIP2 exact EOF,
canonical XIP2 parser agreement, and byte-identical re-encoding.  Missing,
truncated, corrupt, noncanonical, foreign-source, and trailing-byte packets are
all refusals.

### Modes

- `PASS_P0_V1`: canonical empty body, `CAMERA_THEN_R` discriminator, positive
  zero pitch bits, and a fixed absent-guidance digest.  Decode returns P0
  exactly.  These canonical restrictions prevent many byte-distinct aliases
  for one no-op.
- `COPY_CONDITIONAL_Y1_V1`: canonical empty body, `CAMERA_THEN_R`
  discriminator, positive zero pitch bits, and the same fixed absent-guidance
  digest.  Decode performs the global chronological predictor
  `Y0 := exact conditional Y1`, while returning Y1 unchanged.  This is a real
  distinct implementation, not enum padding and not a sparse per-cell copy:
  it pays only the fixed packet protocol bytes and mode discriminator, never
  `384x512` copy coordinates or dense Y1 bytes.
- `XIP2_WARP_V1`: one exact XIP2 `[pair,6]` int16 trajectory with six finite
  positive fp32 scales.  Pair count and predictor identity are checked through
  `SE3XiTransportV2`.

`COPY_CONDITIONAL_Y1_V1` is the canonical zero-motion inter-frame predictor.
The current stage ablation identifies chronology as the dominant coordinate;
this wire mode represents that global predictor directly rather than
catastrophically expanding it into sparse-cell controls.  That rationale is a
design input only; G13 runs no scorer and makes no improvement claim.

### Interpretations

The active mode has exactly two non-padding interpretations:

1. `CAMERA_THEN_R`: warp exact PASS-G conditional Y1 in `874x1164` EON camera geometry,
   round the NumPy reference to uint8, and expose that camera Y0 for the frozen
   downstream R operator.
2. `SCORER_THEN_FACTOR2`: first apply the exact generic disjoint R operator to
   PASS-G conditional Y1, warp in `384x512` scorer geometry, round the NumPy reference to
   uint8, then use the certified disjoint factor-2 integer preimage to realize
   a camera Y0.

The interpretation byte is inside both CRC custody and therefore cannot alias
on wire.  A nonconstant/nonzero fixture must also show different realized Y0
bytes, proving these are different programs rather than renamed padding.

## Geometry and numerical reference

The sole V1 geometry profile is
`EON_GROUND_874X1164_TO_384X512_V1`:

- native camera `(H,W)=(874,1164)`;
- scorer `(H,W)=(384,512)`;
- native intrinsics `fx=fy=910`, `cx=582`, `cy=437`;
- camera height `d=1.22 m`;
- plane normal `n=[0,-cos(pitch),-sin(pitch)]`; and
- translation-first twist with
  `H=K(exp_SE3(xi).R - t*n^T/d)K^-1`.

Pitch is counted explicitly as canonical fp32.  XIP2 scales are canonical
fp32; quantized coordinates are int16.  The portable output reference is NumPy
with fp32 sample buffers and round-to-nearest-even/clamp to uint8; homography
construction retains the existing deterministic fp64 EON geometry oracle.
This mixed precision is named and versioned rather than silently inheriting a
host default.

## Counted/free boundary and raw accounting

COUNTED in this A packet:

- exact XIP2 video-derived sufficient statistics;
- pitch, domain, pair/geometry descriptor, source/lineage digests;
- lengths and CRC custody.

GENERIC decoder code, SE(3) expansion, EON geometry, inverse bilinear warp,
disjoint R, factor-2 realization, hashing, and parsing are free algorithm.
There is no scorer model, scorer output, target/GT table, dense Y0, or dense Y1
at receive.

Every receipt must prove exact raw arithmetic:

`raw_counted_bytes = packet_bytes`

`packet_bytes = header_bytes + xip2_payload_bytes + footer_bytes`

`protocol_overhead_bytes = header_bytes + footer_bytes`.

The receipt separately reports XIP2 bytes, protocol bytes, both CRCs, packet
hash, decoded-q hash, output hashes, and source hashes.  Fixed truth labels are:
strict EOF/CRC/parse-reencode/source closure true; PASS-G conditional Y1 preservation true;
generic decoder and counted-sufficient-stat-only true; dense target/Y0/Y1,
scorer, scorer-output, target-label, and GT serialization false; scorer
invocation false; n600/through-R/exact-score/candidate/originality/promotion
claims false; `research_only=true`.

## Versioning and resumption

Packet magic, version, source domain, geometry profile, numeric reference ID,
mode, and interpretation are closed enums.  Adding a source universe,
numerical rule, or geometry requires a new version/profile; V1 parsers do not
guess.  The exact packet bytes are the
resumption checkpoint.  `resume_*` requires the caller's expected packet
SHA-256, re-runs strict parse/source/CRC/re-encode closure, and deterministically
replays the receiver.  No loop or long job is launched, so no hidden in-memory
state exists to checkpoint.

## Public API freeze target

The isolated module will export:

- closed mode, interpretation, and geometry enums;
- typed program (including canonical global conditional-Y1 copy), parsed
  packet, decoded result, receipt, and compiled result;
- `compile_counted_xip2_chronological_a3(...)`;
- `compile_counted_xip2_chronological_a3_from_guidance(...)`;
- `compile_counted_xip2_chronological_a3_pass(...)`;
- strict parse, re-encode, decode, resume, and receipt-parse functions; and
- packet magic/version/header/footer constants for exact accounting tests.

## G10 monolithic integration seam (identified, not edited)

In G10's production monolithic receiver, the seam is the conditional-A section
dispatch immediately after nonempty `TACPG81` has produced an exact
`DecodedPassSemanticGEnvelopeV1` in either `PASS_NO_G8_V1` or
`PASS_THEN_G8_V1`.  A future integration may recognize `TACX2A3\0` beside
`TACAPG1`, call the G13 decoder with the already available exact
`predictor_surface` and `pass_g`, and place the returned `camera_y0`, unchanged
conditional `camera_y1`, chronology, and nested receipt into the G10 success
receipt.  The legacy exact-semantic diagnostic branches and production
`TACAPG1` branch remain unchanged.  No source binding may be translated or
forged at this seam.

## Acceptance tests

The focused suite must prove:

1. PASS is exact P0 and exact unchanged conditional Y1 for both
   `PASS_NO_G8_V1` and `PASS_THEN_G8_V1`;
2. `COPY_CONDITIONAL_Y1_V1` is an empty-body protocol-only packet with exact
   `Y0 == source conditional Y1`, unchanged output Y1, exact chronology, no
   dense/cell-coordinate serialization, and distinct bytes/behavior from PASS;
3. the same copy packet fails across PASS-G modes and foreign sources;
4. both active interpretations consume identical XIP2 bytes and emit distinct
   packets and distinct nontrivial Y0 bytes;
5. parse -> re-encode is byte-identical and receipts strict-parse/re-emit;
6. exact header/body/footer raw byte arithmetic;
7. deletion, truncation, CRC corruption, header mutation, and trailing bytes
   fail closed;
8. a foreign P/PASS-envelope/optional-G8/conditional-Y1 source fails closed,
   and the diagnostic exact-semantic source type is unrepresentable;
9. XIP2 internal trailing bytes or pair mismatch fail closed;
10. conditional Y1 remains byte-for-byte identical for PASS, global copy, and
    both active warp domains;
11. decode is deterministic and arrays are immutable;
12. resume requires and preserves the exact packet hash; and
13. the decoder API has no scorer, target, target-label, GT, or dense-evidence
    input.

Verification results and frozen hashes are appended only after implementation
and independent review.  Pointer delta remains zero by construction because
scorer/eval/archive work is explicitly out of scope.

## Originality accounting

Original in G13: the counted chronological XIP2 A wire domain, its dual CRC and
strict EOF/source ABI, explicit production-source and two-domain geometry
discriminators, canonical PASS and global zero-motion conditional-Y1 copy
representations, PASS-G conditional chronology, raw-byte receipt, and
resumable packet checkpoint.

Reused generic in-repository substrate: G9 guidance/XIP2 bytes,
`SE3XiTransportV2`, `xi_pose_coder`, EON homography/warp, the disjoint resize
operator/factor-2 realization, and G10's typed
`PassConditionalASourceBindingV1`.  No
historical/public candidate payload, learned weights, scorer data, target
table, or archive bytes are reused.  The executable receipt nevertheless keeps
`originality_claim=false`; originality of a research mechanism is not promoted
into a contest-candidate claim without later archive lineage custody.

## Verification freeze, 2026-07-26

The isolated landing is:

- `src/tac/witness_dsl/taskspace_counted_xip2_chronological_a3.py`, SHA-256
  `eb4e185647118f02c5d9f4125702527260c524a55c72167d0993fd20b72c1f5f`;
- `src/tac/witness_dsl/tests/test_taskspace_counted_xip2_chronological_a3.py`, SHA-256
  `2d9e88c1d925cc3125e994ef121131bf44d7e6420a402875bb4651e199860e05`.

Frozen dependency seams, independently re-hashed after implementation:

- G9 chronological guidance:
  `c59366b97e7facac81809ad213f6646485db8e106439a6845ea6b5e4c838cd9a`;
- G10 PASS semantic G:
  `701bbab6561e2b8bfee1a4469f0aeb000347219b4a20fc9ffd2550d143171b38`;
- G10 PASS conditional A:
  `282f620c3c4712c35fc6b1a75b8f48b5de2ce250c145b06b0fb5b4f551a8050f`;
- G10 monolithic receiver (identified seam only, not edited):
  `aa1cc628764de0311a23db791635ec884f24d27dc126ceff28e03134e8f51ced`;
- G12 same-class realization encoder (ownership boundary, not edited):
  `842035c29647f26aa620a1c19a26943bd8a307702563a283215afffcf234285a`.

Verification commands and outcomes:

- `ruff format --check` and `ruff check` on the isolated module and test: PASS;
- focused suite: `15 passed in 3.60s`;
- focused suite plus the two frozen production-source suites: `32 passed in
  10.25s`;
- `py_compile`: PASS;
- `git diff --check` on the two files and this spec: PASS.

The public API landed exactly as the freeze required: closed source/mode/
interpretation/geometry enums; typed program, parsed packet, receipt, decoded,
and compiled results; direct, G9-guidance, and PASS/COPY compilers; strict
parse/re-encode/decode/resume/receipt-parse functions; and exported packet
magic/version/header/footer constants.  V1 fixes a 101-byte header, four-byte
footer, at most four contiguous pairs, and a one-MiB fail-closed XIP2 body
ceiling.  PASS and global COPY therefore each cost exactly 105 raw bytes and
carry zero XIP2, dense-frame, or per-cell-coordinate bytes.

Frozen blockers and non-claims:

- this section is not yet dispatched through G10's monolithic archive member;
- no archive was built, no scorer/evaluator was loaded, and no n600 or exact
  score row exists;
- raw finite packet price and receiver causality are closed, but score value
  per byte is unmeasured;
- G9's binding is carried only as lineage metadata; production authority is
  exclusively the exact G10 PASS-conditional source binding;
- there is no candidate, originality, through-R target-debt, promotion, or
  pointer-delta claim. Exact frontier pointer delta remains zero.
