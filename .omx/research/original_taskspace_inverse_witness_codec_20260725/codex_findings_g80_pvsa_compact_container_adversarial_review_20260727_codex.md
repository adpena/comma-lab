# G80 adversarial review — PVSA compact receiver container

Date: 2026-07-27  
Lane: `lane_g81_g80_pvsa_compact_container_adversarial_review_20260727`  
Mode: read-only L0 adversarial review; G80 source and tests were not edited  
HEAD observed at close: `baa863756d740591330324779c2998f5db48e131`

## Executive verdict

G80 found a materially better physical container. Its one-member `PVSA1`
archive:

- survives the actual public `/usr/bin/unzip` boundary;
- reconstructs the exact canonical 133,941-byte semantic `P`;
- removes decoder-dead rich compiler/provenance fields from the counted wire;
- preserves the real 52-byte G74 role-aware operand;
- selects a deterministic, strict, one-member DEFLATE archive of 129,392 bytes;
  and
- saves 2,516 exact archive bytes against the current rich G77 representation
  of the same `P + G74` object.

That is a real rate result, not a score result. Distortion was not measured,
there is no public receiver, no complete n600 actuator stream, and the
competitive pointer remains the official-display `0.172` row.

The first structural defect found by this review was repaired in the shared
tree before review close:

1. **Permit zero actuators.** The same grammar must encode the semantic-`P`-only
   baseline, then add or remove the final actuator without changing containers.
   The measured zero-actuator archive is 129,335 bytes. The G74 actuator's exact
   marginal cost is therefore 57 bytes, while the container recode itself saves
   4,606 bytes. The original nonempty grammar entangled those two effects. The
   live implementation now accepts zero actuators, implements strict
   `decode_base_pair()`, and has zero/add/remove tests; all seven focused tests
   pass.

Three structural repairs remain before `PVSA1` can be the whole-object codec
substrate:

1. **Lower the exact rich object, not unbound operands.** A typed
   `TSPPV2 -> PVSA1 + external receipt` operation must verify the exact semantic
   object, source window, target/source/runtime identities, factor and actuator
   bytes before it strips decoder-dead fields.
2. **Replace numeric order with a normative state machine.** Strictly
   increasing unique type IDs serialize a set; they do not prove that an
   actuator consumed the required predecessor state or produced the named next
   state. Conditional `Y0|Y1` needs an explicit versioned transition contract.
3. **Cache and stream the public receiver.** The current
   `decode_g74_pair()` reopens and reverifies `P` and double-decodes on every
   call. A warm-process 600-pair extrapolation is about 40 minutes. Opening the
   decoder once and decoding a bounded batch reduces the measured extrapolation
   to about 12.6 minutes, making a compliant path plausible but not proved.

G7 also cannot yet price this object: its archive strategy remains hard-wired
to the G17 monolithic P/G/A/E member. `PVSA1` needs an additive G7 archive
strategy, beginning with the zero-actuator state and applying exact
add/remove/replace actuator transitions.

## Exact reviewed objects

G80 source:

```text
src/tac/witness_dsl/taskspace_pvsa_compact_container_v1.py
sha256 2b8688c30f6f43618409d6cfa2cfcb0e91bfd1b353796e882055bec0609182ad
```

Focused test:

```text
src/tac/witness_dsl/tests/test_taskspace_pvsa_compact_container_v1.py
sha256 e6d4a019650ccd3308561261f06326d04099735683de7cc086b5b22590df59ea
```

Specification:

```text
.omx/research/original_taskspace_inverse_witness_codec_20260725/
  SPEC_g80_pvsa_compact_container_20260727.md
sha256 faccaf7b2a3d82dae71ac5dc4eba75a785f63ae0cb728da1ae9d38f10c104224
```

Receipt:

```text
.omx/research/original_taskspace_inverse_witness_codec_20260725/
  g80_pvsa_compact_container_receipt_20260727.json
sha256 5eb369ba12513746ecdb79797a5b2f7bea4efe643772647fdc23a7c32a142393
```

Focused verification was rerun against the repaired files without this
reviewer changing either one:

```text
7 passed in 43.64s
```

## Exact byte ledger

All rows below are exact archive or packet bytes, not estimates.

| Object | Exact bytes | SHA-256 |
|---|---:|---|
| canonical five-member semantic `P` | 133,941 | `759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df` |
| G74/Y1 operand | 52 | `632465f836a33fe11458f7b32079df6f33815fbb7064ef70f0d51934e2abe3f6` |
| G80 compact member | 133,363 | `308b05e623ad0c6d16b69da61c592b9c87e9fd934aeeec71a76850eace618f39` |
| G80 outer STORE | 133,471 | `951d8290b0724eeb23fa467aaef4c5a73771e6a33922d3d6051da2c87b356464` |
| G80 outer DEFLATE, selected | **129,392** | `6b6b6dc8fd715241339932fbf2fea37060704b8e389169eac30e45455b67edc6` |
| G79 best five-member `P`-only layout | 129,800 | recorded in G79 |
| G77 rich six-member `P + TSPPV2/G74` | 131,908 | `0c4cbe...` |

The exact same-object G77-to-G80 physical-layout delta is:

```text
131,908 - 129,392 = 2,516 bytes
rate differential = 25 * (-2,516) / 37,545,489
                  = -0.001675301126055383 score units
```

This is the defensible rich-wire-to-compact-wire rate comparison because both
objects carry the same semantic `P` and same 52-byte G74 operand.

G80 is also 408 bytes smaller than G79's best top-level mixed compression
result while adding the 52-byte actuator. That comparison establishes that
one cross-section DEFLATE stream beats the tested six-member layout; it is not
a same-object delta because G79's object had no actuator.

### Zero-actuator counterfactual

A read-only construction using the same header, semantic payloads, ZIP32
reconstructor, and outer codec, but actuator count zero, measured:

| Object | Exact bytes | SHA-256 |
|---|---:|---|
| zero-actuator compact member | 133,306 | `6208ac...` |
| zero-actuator outer STORE | 133,414 | `2bf452...` |
| zero-actuator outer DEFLATE, selected | **129,335** | `fa173ef4f75adbe9194d3cd89b04021dabd2b9e9fd3aa87081148b6b42a26c75` |

Therefore:

```text
container recode versus canonical P = 129,335 - 133,941 = -4,606 bytes
G74 marginal in the same container   = 129,392 - 129,335 =     57 bytes
G74 marginal rate contribution       = 25 * 57 / 37,545,489
                                     = 0.000037953960327963764
net G80 delta versus canonical P      = -4,549 bytes
```

The original G80 implementation refused actuator count zero in its dataclass,
encoder, and parser. That was not merely ergonomic. It prevented G7 and a
joint allocator from separating:

```text
physical container gain
+ actuator's exact marginal bytes
+ actuator's exact distortion change
= whole-object score transition
```

The live repair now:

- accepts `0..MAX_ACTUATORS` in the member encoder and parser;
- permits zero or one G74 payload at the V1 build surface;
- represents the baseline with the empty actuator tuple;
- strictly refuses the G74 decoder on that baseline;
- implements `decode_base_pair()` and requires exactly zero actuators;
- opens the exact reconstructed semantic `P`;
- renders the selected pair twice and refuses dtype, shape, or byte
  nondeterminism; and
- tests exact zero/add/remove member reconstruction and the measured 129,335
  versus 129,392 outer archives.

The focused suite passes all seven tests. This closes the container-state
defect at local receiver scope. It does not close public inflate, full-n600
streaming, exact evaluator output, or G7 integration.

## Public-unzip and canonical ZIP32 review

An independent review-time proof built the current selected G80 archive, ran
the actual `/usr/bin/unzip` into a context-managed temporary directory, and
read only the extracted `0.bin`.

Observed:

```text
extracted member set  ["0.bin"]
0.bin byte-identical  true
parsed canonical P bytes 133941
parsed canonical P sha256
  759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df
canonical P byte-identical true
scratch auto-cleaned true
```

This closes the specific G79 objection that the receiver had depended on
original `ZipInfo` metadata that public unzip discards. G80's receiver owns the
fixed five names and generic ZIP constants and reconstructs the exact logical
`P` from `0.bin`.

The manual ZIP writer correctly fixes:

- the five semantic member names and order;
- STORE for reconstructed `P`;
- ZIP version 2.0;
- Unix creator system;
- mode 0644;
- DOS epoch timestamp;
- no extras, comments, data descriptors, encryption, or ZIP64; and
- deterministic local headers, central directory, and EOCD.

### ZIP32 bound defect

The writer checks each payload against a uint32 bound, but the aggregate
offset is packed into ZIP32 fields before a complete checked-add proof over
payloads, local headers, names, central entries, and EOCD. Near-bound synthetic
inputs can therefore reach `struct.error` or attempt large allocation instead
of failing with `CompactPVSAError`.

This is not exercised by the present 133KB object and does not invalidate the
measured archive. It is nevertheless an ABI sharp edge: either impose the
actual small product ceiling at parse time or perform checked aggregate ZIP32
arithmetic before allocating or packing anything.

## Tamper and strictness review

Existing fail-closed surfaces are meaningful:

- exact outer one-member parser, member name, CRC, method, metadata, EOF;
- compact magic and bounded declared lengths;
- unknown, duplicate, reordered, zero-length, oversized, truncated, and
  trailing actuator refusal;
- strict G74 magic/length grammar and nested G2SH CRC;
- strict canonical semantic source parsing and byte-identical re-encoding at
  build time; and
- exact reconstructed `P` binding before G74 execution.

The compact parser itself reconstructs a canonical ZIP from any five nonempty
semantic payloads; it does not semantically open `P` until
`decode_g74_pair()`. This separation is acceptable only if every public video
path opens and validates `P` before writing any output. A tampered semantic
section must never partially emit a video.

Missing focused attacks:

1. actual `/usr/bin/unzip` followed by extracted-member-only reconstruction;
2. compact magic and actuator-count mutation;
3. semantic length shift, semantic byte mutation, and proof of refusal before
   output;
4. outer corruption and extracted member-set/name/type checks;
5. independent base-output equality against the exact semantic receiver and
   actual outer-archive add/remove/rollback, beyond member-level reconstruction;
6. true multi-actuator prerequisite, gap, swap, and forbidden-combination
   refusal;
7. exact rich-`TSPPV2` lowering and full output equivalence; and
8. aggregate ZIP32 boundary refusal with the domain error type.

## Rich IR stripping: legal boundary and missing custody proof

The rich compiler object and compact receiver wire should be different objects.
The following fields can remain external without being counted when they are
decoder-dead:

| Rich field class | Compact treatment | Required proof |
|---|---|---|
| source/target custody receipts | external evidence only | exact final archive receipt binds their hashes |
| compiler config and search trace | external evidence only | replayable compile receipt |
| score budget and diagnostic summaries | external evidence only | never read by receiver |
| fixed semantic names and ZIP metadata | generic receiver constants | exact `P` reconstruction equality |
| decoder source/runtime identities | external evidence plus shipped generic runtime | source/runtime hash in final receipt |
| semantic payload SHA/length | derived from exact compact bytes | parse-back receipt |
| actuator SHA/length | derived from exact compact bytes | strict typed parser |

No output-affecting video-specific value may be moved into `inflate.py` or an
external receipt. If the receiver reads it to decide or render pixels, it is a
counted operand. The external receipt is evidence, never an input.

G80's current builder accepts an arbitrary canonical V15 `P` and raw G74
operand. It does not consume or verify the completed rich `TSPPV2` object and
does not prove that the two representations have the same:

- semantic `P`;
- target custody;
- source window and population;
- factor/realization family;
- source and decoder identities;
- typed actuator bytes; or
- decoded outputs.

G80 honestly lists this lowering receipt as open. The smallest real next
implementation is:

```text
lower_tsppv2_to_pvsa1(exact_tsppv2, exact_semantic_p)
  -> parse and validate exact rich packet
  -> bind target/source/decoder/window/factor identities
  -> prove exact semantic-P length and SHA
  -> extract the exact typed decoder operand
  -> build and strict-reopen PVSA1
  -> emit external lowering receipt mapping every removed rich field to
     derived, generic, or encoder-only status
  -> prove rich and compact receiver outputs byte-identical
```

The final candidate lowering must additionally require
`source_pair_start = 0` and `n_pairs = 600`. The present builder is generic over
canonical V15 inputs, while `decode_g74_pair()` labels its accepted local index
as n600. Either enforce n600 at the candidate lowering boundary or remove the
misleading n600 label from the generic API.

### Telemetry defect

`rich_ir_bytes_avoided` is not derived from a parsed rich packet. It is an
arbitrary caller-supplied diagnostic. The same exact compact archive can report
any supposed savings:

```text
caller rich_ir_bytes 52      -> reported avoided 0
caller rich_ir_bytes 3591    -> reported avoided 3539
caller rich_ir_bytes 3593    -> reported avoided 3541
caller rich_ir_bytes 100000  -> reported avoided 99948
```

The G80 spec and focused test use 3,591 bytes, while the current exact G77
`TSPPV2` packet is 3,593 bytes: 3,541 bytes of rich framing and the 52-byte
operand. The exact same-object archive saving is 2,516 bytes, not either raw
packet-framing number.

Remove the caller-controlled field or bind it to the exact parsed rich packet.
Report the two distinct values:

```text
decoder-dead rich packet bytes removed = 3,541
exact same-object archive bytes saved   = 2,516
```

Only the second is an exact rate delta.

## Typed actuator ordering is not yet transition semantics

G80 validates:

```text
actuator type IDs == sorted(unique(type IDs))
```

That gives canonical serialization order. It does not establish:

- the predecessor state an actuator consumes;
- the output state it produces;
- whether a prerequisite is present;
- whether an optional stage can be skipped;
- whether two packets of the same family may cover disjoint windows;
- whether a refinement family may repeat;
- whether two types commute; or
- whether the selected sequence is legal for this container version.

The current decoder does not dispatch an actuator sequence. It requires exactly
one G74 actuator. Adding type 2 for conditional `Y0|Y1` would therefore serialize
another byte string without proving composition.

The minimal byte-conscious cure is a decoder-owned, versioned transition
registry:

```text
PVSA1 empty             : semantic P -> base receiver state
type 1 G74/Y1           : base receiver state -> Y1-selected state
type 2 conditional Y0|Y1: Y1-selected state -> joint Y0/Y1 state
```

The compact wire need not repeat large predecessor hashes if the closed
registry makes them derivable. The parser/receiver must nevertheless reject
missing prerequisites, gaps, swaps, duplicate singleton stages, forbidden
combinations, and outputs whose actual state binding differs. If repeated or
windowed actuators are required later, add a compact stage ordinal/window
contract rather than retaining the current global uniqueness rule by accident.

## Receiver timing and memory

Timing probes were macOS-CPU advisory only. They are not contest-CPU or score
authority.

The current `decode_g74_pair()`:

- opens and verifies the semantic archive on every call;
- invokes the G74 deterministic double decode on every call; and
- returns camera frames plus large diagnostic tensors/masks.

Measured:

```text
cold single-pair call                 32.917 s
first call in one warm process        33.178 s
second warm call                       4.033 s
third warm call                        3.994 s
naive warm 600-pair extrapolation     ~40 min
```

Opening `V15RoleAwareOverlayDecoderV1` once and decoding a cached batch:

```text
cold open with member effects         31.612 s
cached one-pair double decode          1.404 s
cached eight-pair double decode       10.083 s
cached per-pair rate                   1.260 s
600-pair extrapolation                12.604 min
```

The cached result for eight pairs occupied 125,975,808 bytes across camera and
diagnostic arrays. The corresponding dense 600-pair materialization would be
about 9.45GB before decoder working memory and video encoding, which is unsafe
on the 16GB public tier.

The public receiver therefore needs:

1. one strict outer parse and one semantic-`P` open;
2. one parsed typed actuator registry;
3. bounded pair chunks;
4. immediate ordered video write and release of camera arrays;
5. diagnostic/proof tensors disabled or released on the production path;
6. atomic/resumable output staging consistent with project policy; and
7. two complete public inflations proving byte-identical video output.

A decode-once production path may be reasonable after offline double-decode
proof binds the exact same implementation and bytes. That optimization must not
weaken deterministic receiver proof. Only a real full-n600 public run can close
the 30-minute and memory budgets.

## G7 integration

G80 reuses the exact deterministic one-member outer codec and its STORE versus
DEFLATE arbitration. This is a good physical archive primitive.

G7 still cannot evaluate it. G7's state evaluator directly builds the G17
monolithic P/G/A/E member and its strict parser expects that grammar inside
`0.bin`. A `PVSA1` member is a different grammar. Wrapping `PVSA1` as a G17 P
section with dummy G/A/E sections would add bytes and falsely describe the
state.

Add an archive-strategy seam:

```text
PVSACompactStrategy.build(logical_state)
PVSACompactStrategy.parse(exact_archive)
PVSACompactStrategy.receiver_request(exact_archive, pair)
```

The first state is zero-actuator `PVSA1`. Exact transitions then add, remove,
or replace typed operands and race the complete rebuilt outer archive.
Container gain is booked once; every actuator is judged by its exact marginal
compressed bytes and realized distortion change. G7's existing strict
parse-back, repeat, rollback, score, and pointer logic can remain.

G7's greedy ordering still is not global authority. Interacting Y1 and
conditional `Y0|Y1` proposals need complete-universe or G33 endpoint
arbitration when one proposal changes another's rate or distortion sign.

## Required next gates

In dependency order:

1. preserve the now-closed zero-actuator baseline while later actuator types
   and G7 archive strategies are added;
2. correct the unbound 3,591-byte telemetry and record exact 3,593/3,541/2,516
   quantities with their distinct meanings;
3. implement strict `TSPPV2 -> PVSA1` lowering with an external custody receipt
   and exact output-equivalence proof;
4. define the versioned actuator state machine and add conditional `Y0|Y1`
   only as a real typed transition after Y1;
5. cache the receiver, stream bounded chunks, and prove full-n600 public timing,
   memory, output naming, frame count, resolution, and double-decode equality;
6. add `PVSACompactStrategy` to G7 and compare exact whole-object states,
   including the zero-actuator baseline;
7. build the receiver-closed public archive, hash exact bytes, run two clean
   public inflations, and recursively close `inflate.sh` imports/runtime; then
8. run `upstream/evaluate.py` on contest hardware and report exact n600
   component distances and score.

## Triality and pointer honesty

### DSL

`PVSA1` is a promising normative compact receiver bitstream. Its V1 grammar
must include the base state and its decoder-owned transition registry must be
normative. Rich `TSPPV2` remains compiler/custody IR.

### DAG

```text
fresh exact semantic P + fresh rich TSPPV2 compile
  -> strict custody-bound lowering
  -> PVSA1(empty) base state
  -> add G74/Y1 typed transition
  -> add conditional Y0|Y1 typed transition
  -> one-member exact outer arbitration
  -> cached bounded public receiver
  -> exact output video, double decode
  -> full-n600 upstream evaluate.py
  -> compare against dynamic effective frontier
```

### Equations

Let `C` be the zero-actuator compact container, `A_j` typed actuators, `Z` the
exact selected outer ZIP transform, and `D` realized evaluator distortion:

```text
B(S) = bytes(Z(C || ordered_transitions(S)))

DeltaScore(j | S) =
  100 * (d_seg(S+j) - d_seg(S))
  + sqrt(10*d_pose(S+j)) - sqrt(10*d_pose(S))
  + 25 * (B(S+j) - B(S)) / 37,545,489
```

No independent segment, pose, or rate threshold is sufficient. Admission is a
joint state-contingent score transition.

The exact frontier pointer did not move in G80 or this review. No candidate
archive, exact component distances, or authoritative score exist yet. The
competitive target observed here remains the official-display `0.172`, and the
project mission remains sub-0.15.
