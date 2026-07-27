# G69 — canonical G17 active-A ABI for the G49 selected-preimage program

Date: 2026-07-27  
Lane: `lane_g69_g49_active_a_abi_20260727`  
Authority: structural only; `research_only=true`

## Result and non-claim

The canonical G17 A envelope now has one additive, non-cross-cast carriage for
the existing G49 `TSPPV1` selected-preimage program. No parallel packet schema
was introduced. The nested bytes remain exact G49 bytes and are reopened only
by G49's existing `parse_selected_preimage_program` plus
`encode_selected_preimage_program`.

This landing contains no dense frame/target/scorer planes, evaluator call,
score row, candidate, promotion, or pointer movement. G49 still declares
`standalone_decode_closed=false`; this structural ABI does not change that
truth label.

## Append-only wire allocation

All pre-existing wire values are unchanged.

| G17 field | existing terminal value | new appended value |
|---|---:|---:|
| A family | `G17_GENERAL_CONDITIONAL_XIP2=5` | `G49_SELECTED_PREIMAGE_PROGRAM=6` |
| A mode | `QUANTIZED_XIP2=4` | `SELECTED_PREIMAGE_PROGRAM=5` |
| terminal A family | `MIXED=4` | `G49_SELECTED_PREIMAGE=5` |
| terminal A mode | `MIXED=5` | `SELECTED_PREIMAGE_PROGRAM=6` |

The family/mode pair admits only exact `TSPPV1\x00\x00` magic. It is legal only
as one `GLOBAL` descriptor spanning the complete G17 population window.
`SHARDED` carriage is refused because G49 is one population-global selected
program; multiplying or slicing it into G17 shard-local aliases would invent a
different ownership/program contract.

## Frozen strict-parser adapter

`G17G49SelectedPreimageStrictParserV1` is a frozen callable adapter with four
explicit custody inputs:

1. expected exact `V15SemanticProgramIdentityV1`;
2. expected exact `ScorerTargetCustodyIdentityV1`;
3. expected exact `GenericV10Factor2DecoderIdentityV1`; and
4. a positive caller-owned maximum packet byte ceiling.

Its acceptance equation is:

```text
accept(payload, descriptor) iff
  family == G49_SELECTED_PREIMAGE_PROGRAM
  and mode == SELECTED_PREIMAGE_PROGRAM
  and payload[0:8] == TSPPV1 magic
  and parse_G49(payload, maximum_packet_bytes) = program
  and encode_G49(program) == payload
  and (program.compile.start, program.compile.count)
      == (descriptor.start, descriptor.count)
  and (program.semantic.start, program.semantic.count)
      == (descriptor.start, descriptor.count)
  and program.semantic_identity == expected_semantic_identity
  and program.target_identity == expected_target_identity
  and program.decoder_identity == expected_decoder_identity.
```

The target custody type itself is the existing G49 n600 target-bank identity
and has no descriptor-start field. The adapter therefore verifies that object
by exact equality, while verifying descriptor geometry against the two G49
objects that actually carry the selected program window: compile config and
semantic identity.

## Triality

### DSL

The only new language production is:

```text
G17 A descriptor :=
  family=G49_SELECTED_PREIMAGE_PROGRAM,
  mode=SELECTED_PREIMAGE_PROGRAM,
  layout=GLOBAL,
  payload=<exact existing TSPPV1 packet>
```

### DAG

```text
exact P -> canonical sharded PASS G
        -> one global G49 A descriptor
        -> TSPPV1 strict parse -> exact G49 re-encode
        -> window + semantic/target/decoder identity equality
        -> canonical G17 terminal summary
        -> exact P/G/A/E member and STORE/DEFLATE reopen
```

### Equations

The adapter proves representation identity, not scientific quality:

```text
encode_G49(parse_G49(A.payload)) = A.payload
window_G49 = window_A
identity_G49 = identity_expected
```

No distortion, rate, or score coordinate is populated by this landing.

## Structural proof surface

The focused tests prove:

- every old enum value is unchanged and every new value is appended;
- exact n2 and n600 G49 build, G17 A parse/re-encode, terminal derivation, and
  full canonical archive build/reopen;
- the unique population-global window at both structural sizes;
- sharded G49 refusal;
- explicit strict-parser requirement;
- wrong family, mode, magic, window, semantic identity, target identity,
  decoder identity, and packet byte ceiling refusal;
- nested payload mutation refusal; and
- G17 descriptor and terminal wire mutation refusal.

The unchanged canonical G17 tests pass alongside the focused G69 tests.

## Remaining blockers

This landing removes only the former G17 A-family/mode placement blocker for
exact G49 bytes. It does not prove:

- a real fresh semantic P archive matching a particular G49 identity;
- executable BoundV10 receiver incidence from a concrete production archive;
- public `inflate.sh` / `upstream/evaluate.py` closure;
- exact n600 scorer equality or a score below any target; or
- candidate or frontier status.

Those are downstream whole-object obligations, not facts implied by an ABI
parse/re-encode proof.
