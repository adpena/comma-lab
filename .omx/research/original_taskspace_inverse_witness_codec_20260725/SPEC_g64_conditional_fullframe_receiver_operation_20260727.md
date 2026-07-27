# G64 conditional full-frame receiver operation

Date: 2026-07-27  
Lane: `lane_g64_primary_selected_solution_program_v2_20260727`  
Status: receiver operation landed; public/archive admission remains blocked  
Authority: implementation mechanics only; no scorer run, archive candidate,
exact row, promotion, or pointer movement

## Outcome

G64 adds one closed operation to the G17 receiver vocabulary:

`G17_CONDITIONAL_FULLFRAME_Y0_GIVEN_EXACT_DECODED_Y1_V1`

Its callable contract is:

```text
(strict_counted_learned_quotient_payload, source_pair_id,
 exact_decoded_camera_Y1_uint8) -> camera_Y0_uint8
```

`source_pair_id` is mandatory because a legal learned quotient packet can
cover more than one pair and Y1 pixels alone are not the canonical
chronological coordinate. The operation returns one full
`uint8[874,1164,3]` Y0 and never mutates the caller-owned exact-decoded Y1.
There is no built-in image transform, direct-plane branch, dense fallback,
scorer, teacher, target, or training path.

## Canonical ownership and wire reuse

The implementation does not create a second G17 ABI. It consumes:

- `G17CompilerPlacementManifestV1`;
- the canonical `CHRONOLOGICAL_POSE` owner and
  `G17ChronologicalPosePreimageV1`;
- the canonical physical coding-group incidence;
- `G17RuntimeDependencyFileV1` at
  `SUBMISSION_RUNTIME_DEPENDENCY`; and
- G49's exact learned irreducible quotient parser and builder.

The selected operand is the named conditional logical owner's exact
`G17ParameterSpellingIdentityV1`, never the monolithic
`manifest.exact_member_bytes`. The strict G49 parser proves the compact
latent/parameter wire, internal hashes, scalar accounting, active pair range,
direct-plane structural bound, and exact EOF. Arbitrary bytes or a
length-smaller-than-frame heuristic are not admitted.

Unrelated logical owners and physical groups are tolerated. A physical group
may be many-to-many and may share the conditional owner with Y1/common owners.
V1 requires the selected conditional owner to map to one exact physical group.
If its bytes span multiple groups, it refuses with
`G17_CONDITIONAL_OPERAND_TO_PHYSICAL_GROUP_SPAN_LINKER_OWED`; no invented span
mapping is claimed.

## Execution and liveness

The receiver:

1. reopens placement, callable source, runtime dependency, payload contract,
   source identity, and exact pair incidence;
2. passes an immutable private Y1 copy plus the exact `source_pair_id`;
3. executes twice and requires bit-identical full-frame uint8 Y0;
4. rechecks the caller Y1 hash;
5. rebuilds four parse-valid G49 packets by mutating the first/last latent and
   first/last parameter bytes; and
6. requires every sampled counted byte either to change Y0 or be explicitly
   refused.

The resulting `logical_group_operand_liveness_results` is logical-owner
liveness derived from the exact owner spelling. It is deliberately not a raw
ZIP-range mutation claim. A one-group incidence does not prove
logical-operand -> archive-range -> reparse -> receiver liveness, so every
receipt retains
`G17_CONDITIONAL_LOGICAL_OPERAND_TO_ARCHIVE_RANGE_LINK_OWED`. The current
monolithic archive also does not expose a canonical multi-group
operand-to-range linker, so construction refuses when one logical operand
spans multiple coding groups.

## Callable custody and the remaining public blockers

The bound decoder must be one inspectable Python function. Its exact source
file SHA-256 and code-object closure are bound. Closures, defaults, non-callable
global values, unresolved names, dynamic access, scorer/teacher/target/direct
plane names, oversized literal tables, embedded artifact hashes, external I/O,
unapproved module globals, and every function-local import are rejected.

That structural audit does **not** prove that the source is generic rather
than video-fitted. The source is typed as a canonical
`G17RuntimeDependencyFileV1` with
`SUBMISSION_RUNTIME_DEPENDENCY`, but neither this operation nor that file type
links it into a terminal public `inflate.sh`/`upstream/evaluate.py` runtime
graph. Therefore every execution receipt retains all three blockers:

- `G17_CONDITIONAL_DECODER_GENERIC_SOURCE_PLACEMENT_OWED`;
- `G17_CONDITIONAL_DECODER_RUNTIME_GRAPH_LINK_OWED`;
- `G17_CONDITIONAL_LOGICAL_OPERAND_TO_ARCHIVE_RANGE_LINK_OWED`.

This operation is public-*shaped* and registry-addressable, not public-archive
closed. G59/G17 still owe the terminal production envelope and public evaluator
closure. G63 owns residual custody only.

## G51 compatibility constraint

The G51 coordination input supplied by the root lane says the exact
common/differential LZMA layout presently beats a strict Seg-Y1 plus
Pose-Y0-given-Y1 split, and class conditioning supplied only a small lower
bound. G64 therefore does not assume that its conditional logical owner has an
exclusive physical byte group. It accepts a shared group while preserving the
logical conditional operation.

This is a compatibility constraint, not a G64 measurement and not a reason to
promote the conditional split.

## Triality

### DSL

- operation:
  `G17_CONDITIONAL_FULLFRAME_Y0_GIVEN_EXACT_DECODED_Y1_V1`;
- input: counted G49 learned quotient, exact pair ID, immutable exact-decoded
  camera Y1;
- output: exact full-frame camera Y0;
- registry: immutable one-entry operation map; unknown operation strings
  refuse.

### DAG

```text
G17 placement manifest
  -> named chronological-pose parameter spelling
  -> strict G49 learned quotient parse
  -> exact callable/source/runtime-dependency binding
  -> (payload, pair_id, exact Y1) deterministic double decode
  -> parse-valid counted-operand liveness
  -> full-frame uint8 Y0 + fail-closed receipt
  -/-> terminal public runtime graph (owed)
```

### Equations

For named pair coordinate `i`, counted operand `A`, and exact decoded Y1
`Y1_i`,

```text
Y0_i = D(A, i, Y1_i)
D(A, i, Y1_i) == D(A, i, Y1_i)              bit exactly
hash(Y1_i before) == hash(Y1_i after)
shape(Y0_i) = (874, 1164, 3), dtype(Y0_i) = uint8
```

For each parse-valid sampled mutation `A_j`,

```text
D(A_j, i, Y1_i) != D(A, i, Y1_i)  OR  D refuses A_j
```

This is receiver-liveness mechanics, not scorer effectiveness and not an
archive score.

## Verification

Implementation:

- `src/tac/witness_dsl/taskspace_conditional_fullframe_receiver_operation_v1.py`
  - SHA-256
    `02023b376fddd9f73b32d4ef799c8333ade1e10af9f7136a122355d4c749449f`
- `src/tac/witness_dsl/tests/test_taskspace_conditional_fullframe_receiver_operation_v1.py`
  - SHA-256
    `ef52e0cc2d64352f78070973c391ec727ffb616bbe8cfcb8f0553d3c6a8e06ef`

Content-addressed source closure:

- operation module:
  `02023b376fddd9f73b32d4ef799c8333ade1e10af9f7136a122355d4c749449f`;
- G49 selected-preimage module:
  `5fd2938ce96bc592fbeddfe519f36b3f2371789dcf322617691e607490c281f0`;
- canonical G17 compiler:
  `36fb4997fd16e30a965e5924c84efa2971e632fc256c2e31c54957b3c4810c91`;
- closure:
  `b7670360f23f546bdf60d3483852faa10855467c825c4ec0711433d3218c6f6d`.

Checks:

```text
ruff format: clean
ruff check: pass
py_compile: pass
focused G64 mechanics: 12 passed in 0.92s
G49 + canonical G17 + G64 regression: 50 passed in 2.76s
```

The tests use full-frame arrays and real strict packets to verify mechanics,
including shared groups, multi-pair coordinate dependence, valid counted-byte
mutation, immutable Y1, hidden globals, local imports, nondeterminism, wrong
dtype/shape/layout, ambiguous spelling, and missing multi-group span linkage.
They are not n600 empirical evidence.

## Pointer delta and next join

Pointer delta: **zero**. Competitive target remains `0.172`; the local
contest-CPU defensive bank remains `0.1880443980` and is not the global
frontier. No archive or scorer was run.

The next real join is for G59/G17 to place a provenance-admissible generic
decoder in the terminal runtime graph and bind the conditional owner to exact
archive spans. Only then can this registry operation participate in recursive
public inflate closure and an exact same-bytes n600 row.
