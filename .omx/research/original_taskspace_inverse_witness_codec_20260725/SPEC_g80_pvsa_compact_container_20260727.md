# G80 PVSA compact receiver container

Date: 2026-07-27  
Lane: `lane_g80_flattened_semantic_product_archive_20260727`  
Authority: exact encoder/archive transport proof; research-only; no candidate,
score, promotion, or pointer authority

## Outcome

G80 lands the missing lowering boundary between the rich selected-preimage
compiler/provenance IR and the counted receiver wire.

The exact fresh n600 V15 semantic base `P` is a canonical five-member STORE ZIP
of 133,941 bytes. Its complete ZIP envelope, member names, metadata, target
custody, source hashes, runtime identities, and verbose JSON are not decoder
operands. G80 stores only:

1. the five semantic member payloads in fixed codec order;
2. a compact typed actuator table;
3. the binary actuator operands in execution order.

The receiver owns the fixed semantic names/ZIP32 metadata and reconstructs
`P` byte-for-byte without consulting the original `ZipInfo`. The resulting
single compact member is then passed through the existing exact
`taskspace_outer_archive_codec`, which races STORE against DEFLATE and strictly
reopens both.

The semantic-only compact baseline is 129,335 bytes, 4,606 bytes below the
original 133,941-byte `P` ZIP. Adding a real, strict 52-byte G74/Y1 two-atom
transport operand changes the same container to 129,392 bytes. The actuator's
actual compressed marginal is therefore only +57 bytes, or
+0.000037953960327963764 rate-score units. The larger -4,606-byte movement is
the container recode, not actuator value. Distortion was not measured in this
landing, so neither differential is a candidate score.

The operand is deliberately a pair-0 transport proof, not a claimed n600
correction stream. Public `inflate.sh`, conditional `Y0|Y1`, full-n600 compile,
and upstream evaluation remain open.

## Why this is the codec composition layer

`TSPPV2` and the G17/G41 compiler objects are appropriate rich IRs: they bind
target custody, source/runtime identities, compile configuration, and typed
semantics. They are not appropriate wire formats. A measured two-atom TSPPV2
packet was 3,593 bytes even though its actual G74 decoder operand was 52 bytes.
G80 makes the separation explicit:

`rich compiler IR + external receipt -> compact typed wire -> free generic receiver`.

This is the same structural distinction made by mature codecs between the
encoder's search graph and the decoder's normative bitstream. Provenance is
not discarded: the build/eval receipt must bind the exact compact archive SHA,
compiler IR SHA, source/target custody, decoder source, and runtime. It simply
does not consume candidate bytes when it is not required to decode.

## Exact wire

The compact member is:

```text
magic "PVSA1\0\0\0"
five uint32 little-endian semantic payload lengths
uint8 actuator count
for each actuator: uint8 type, uint32 little-endian length
five semantic payloads in fixed codec order
actuator payloads in canonical transition order
```

V1 accepts zero actuators as the exact semantic-only baseline, or one actuator
of the following type:

`1 = G74_ROLE_AWARE_PREPAINT`.

The decoder validates an explicit prefix of the normative transition DAG:

`semantic P -> optional G74/Y1 -> future conditional Y0|Y1`.

Numeric enum sorting is not semantic authority. Unknown, duplicate, skipped,
reordered, zero-length, oversized, truncated, and trailing sections fail
closed. The outer ZIP CRC protects the complete compact member; the G74
operand also carries its own strict magic/length/CRC grammar.

The five decoder-owned semantic member constants are:

```text
manifest.json
predictor.zip
predict/movable_polygon_worldsheet.g1s
render/receiver_realization.ddrp
render/scorer_solved_templates.ddst
```

Their canonical reconstructed ZIP is STORE-only, Unix creator system,
ZIP version 2.0, timestamp 1980-01-01 00:00:00, mode 0644, no extras,
comments, encryption, or data descriptors. The manual ZIP32 writer reproduces
the retained exact semantic archive byte-for-byte.

## Exact measured bytes

Input semantic base:

- bytes: `133941`;
- SHA-256:
  `759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df`.

G74/Y1 transport operand:

- bytes: `52`;
- SHA-256:
  `632465f836a33fe11458f7b32079df6f33815fbb7064ef70f0d51934e2abe3f6`.

Semantic-only compact member:

- bytes: `133306`;
- SHA-256:
  `6208ac91c465caa8990f7d643f50c06da28c1e00ca359d4ee55005818cc12352`.

Semantic-only outer DEFLATE, selected:

- bytes: `129335`;
- SHA-256:
  `fa173ef4f75adbe9194d3cd89b04021dabd2b9e9fd3aa87081148b6b42a26c75`;
- savings from original P ZIP: `4606` bytes;
- fixed-distortion rate-score delta: `-0.003066946338080721`.

Compact member with G74:

- bytes: `133363`;
- SHA-256:
  `308b05e623ad0c6d16b69da61c592b9c87e9fd934aeeec71a76850eace618f39`.

Outer STORE:

- bytes: `133471`;
- SHA-256:
  `951d8290b0724eeb23fa467aaef4c5a73771e6a33922d3d6051da2c87b356464`.

Outer DEFLATE, selected:

- bytes: `129392`;
- SHA-256:
  `6b6b6dc8fd715241339932fbf2fea37060704b8e389169eac30e45455b67edc6`.

The G74 same-container marginal is `+57` bytes and
`+0.000037953960327963764` rate-score units before its distortion effect.

G79's best exact top-level mixed-member archive was 129,800 bytes. G80 is
another 408 bytes smaller because a single DEFLATE stream can exploit
cross-section redundancy and pays one outer member envelope.

## Verification

Focused verification:

```bash
uv run ruff format --check \
  src/tac/witness_dsl/taskspace_pvsa_compact_container_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_pvsa_compact_container_v1.py
uv run ruff check \
  src/tac/witness_dsl/taskspace_pvsa_compact_container_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_pvsa_compact_container_v1.py
uv run python -m py_compile \
  src/tac/witness_dsl/taskspace_pvsa_compact_container_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_pvsa_compact_container_v1.py
uv run pytest -q \
  src/tac/witness_dsl/tests/test_taskspace_pvsa_compact_container_v1.py
```

Result: `7 passed`.

The tests prove exact semantic-P reconstruction, outer STORE/DEFLATE
parse-back, compact parse/re-encode equality, typed operand ownership,
unknown/duplicate/order/length/EOF/trailing-byte refusal, and deterministic
double decode of both a semantic-only V15 pair and a real G74-actuated pair.
The zero/add/remove roundtrip proves that proposal rollback stays in one
physical grammar. The cached receiver opens P once and yields bounded
chronological batches of at most 16 pairs, rather than accumulating the full
3.66 GB raw video in memory.

## Triality

### DSL

`PVSA1` is the normative compact typed receiver wire. Rich TSPPV2/G17 objects
remain compiler IR and external custody receipts.

### DAG

`fresh semantic P + rich selected-preimage compile`

`-> verify compiler/custody externally`

`-> lower only five semantic payloads + typed decoder operands`

`-> one-member outer STORE/DEFLATE arbitration`

`-> strict public receiver parse`

`-> V15 realization -> ordered actuator transitions -> video`

`-> upstream evaluate.py`.

### Equations

For semantic payloads `P_i`, actuator operands `A_j`, fixed header `H`, and
outer coding choices `C = {STORE, DEFLATE}`:

`W = H || lengths(P_i) || descriptors(A_j) || P_0 ... P_4 || A_0 ... A_(n-1)`,
where `0 <= n <= MAX_ACTUATORS`.

`archive = argmin_(c in C) bytes(ZIP_c("0.bin", W))`.

The candidate objective remains:

`100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489`.

No actuator is admitted merely because it is compact. Its marginal distortion
change must be measured jointly with its exact compressed byte interaction.

## Honest remaining debt

1. Lower the completed rich TSPPV2 object to this compact wire under one
   external custody receipt.
2. Add the conditional `Y0|Y1` typed actuator and its deterministic receiver
   transition after G74/Y1.
3. Connect the now-bounded cached batch iterator to the exact expected video
   writer and prove its runtime under the public limit.
4. Integrate the compact receiver into public `inflate.py`/`inflate.sh`.
5. Prove public archive parse-back and double decode on the receiver runtime.
6. Run full-n600 upstream `evaluate.py` and report exact component distances,
   archive bytes/SHA, hardware axis, and score.
7. Only then compare against the dynamic upstream pointer and promote.
