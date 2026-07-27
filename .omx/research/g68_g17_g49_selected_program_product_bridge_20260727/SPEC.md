# G68 G17-to-G49 selected-program product bridge

Status: `research_only=true`; no candidate, score, or frontier claim.

## Landed exact mechanics

- `G17G49SelectedProgramPreplacementProductV1` binds the exact 133,941-byte
  fresh CarrierCompose semantic archive to the exact G49 `TSPPV1` packet and
  the concrete `BoundV10Factor2SelectedPreimageDecoderV1`.
- The product streams one pair at a time. Each yielded factor-2 pair is decoded
  and realized twice and must be byte-identical. No dense n600 tensor bank,
  scorer, target bank, or direct plane input is admitted by this bridge.
- The live canonical G17
  `G17G49SelectedPreimageStrictParserV1` owns A parsing. G68 builds a
  population-global G49 A descriptor, twice reopens its exact bytes, and builds
  exact G/A/E sections.
- `G17OwnerSpecificReceiverIncidenceV1` is the missing generic structural
  relation `(physical_group_id, logical_owner_id, receiver_consumer,
  receiver_operation)`. It requires exactly one operation per physical
  group/owner pair and requires a shared physical group to retain only the
  archive-unpack operation. It is structural routing, not execution evidence.
- G49 packet component incidences preserve each component's exact packet
  offset, byte length, SHA-256, byte home, lineage class, factor role/mode, and
  derived scientific/semantic role. They never claim local ranges inside the
  compressed outer ZIP.

## Exact production blocker

The fresh semantic P is itself a CarrierCompose ZIP. The current canonical
monolithic P/G/A/E container deliberately refuses a nested ZIP in every role.
The G68 audit executes that real builder refusal; it does not infer it from
source text.

The current open product blockers are:

1. `G17_SEMANTIC_PROGRAM_LOGICAL_VALUE_TYPE_OWED`
2. `G17_SELECTED_PREIMAGE_PROGRAM_LOGICAL_VALUE_TYPE_OWED`
3. `G17_SEMANTIC_PROGRAM_P_NESTED_ARCHIVE_CONTAINER_ABI_OWED`
4. `G17_OWNER_SPECIFIC_RECEIVER_INCIDENCE_EXECUTION_RECEIPT_OWED`

The first two are explicit because P is not merely topology and `TSPPV1` is not
VM bytecode. No placement manifest is emitted until exact logical types with
their own strict parse/re-emit semantics exist. The fourth is explicit because
receiver names and routing relations are not execution evidence.

## Verification

```text
uv run pytest -q src/tac/witness_dsl/tests/test_taskspace_g17_g49_selected_program_product_bridge_v1.py
.... [100%]
4 passed in 32.31s

uv run pytest -q \
  src/tac/witness_dsl/tests/test_taskspace_g17_g49_active_a_abi.py \
  src/tac/witness_dsl/tests/test_taskspace_g17_g49_selected_program_product_bridge_v1.py
.............. [100%]
14 passed in 32.59s

uv run ruff check \
  src/tac/witness_dsl/taskspace_g17_g49_selected_program_product_bridge_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_g17_g49_selected_program_product_bridge_v1.py
All checks passed!
```

`py_compile` also passed for both files. No heavy evaluation was launched.
The real-product fixture skips explicitly when its retained, intentionally
untracked V15 custody artifacts are absent, so a clean checkout does not fail
solely for lacking local research custody. The exact pointer is unchanged.
