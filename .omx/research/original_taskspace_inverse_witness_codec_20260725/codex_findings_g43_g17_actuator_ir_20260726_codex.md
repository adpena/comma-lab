# G43 findings — G17 actuator IR V1

Date: 2026-07-26  
Lane: `lane_g43_g17_actuator_ir_20260726`  
Mode: bounded real-pair implementation and verification; `research_only=true`; no scorer, exact eval, dispatch, archive promotion, pointer mutation, or commit  
HEAD observed: `0058123af31779d83d1fc10a728389b0ce7823ec`

## Outcome

Implemented the first real selected-solution actuator in
`src/tac/witness_dsl/taskspace_g17_actuator_ir_v1.py`:

`EP725_LABEL_LOCAL_SEMANTIC_G`

The actuator consumes exact counted G packet spans, parses and applies them
through `taskspace_predictor_v2_consumer_seam`, then realizes the changed
semantic cells through the exact `predictor_preserving_taskspace_overlay`
receiver. Its admitted state is exactly:

`TaskspacePredictorStateV2 + NoTransportV2`

It never projects ep725 into `PredictorSemanticStateV1`, invents Pose6, or
routes through the V1 full-n600 generative receiver. That is the corrected G29
compatibility required by G42.

This is a real bounded state transition, not an opcode or receiver-name
attestation. The executor calls the typed donor functions and returns their
typed decoded-G, ownership, and overlay objects together with a strict
execution receipt.

## What landed

### Closed kind and dispatch

- `G17ActuatorKindV1` has one V1 value:
  `EP725_LABEL_LOCAL_SEMANTIC_G`.
- `G17ActuatorReceiverOperationV1` has one audited dispatch value.
- Caller-supplied imports, arbitrary function strings, VM opcodes, and ignored
  unknown kinds are not representable.
- The receipt binds the exact source SHA-256 of both receiver modules and the
  exact operation identities.

### Physical counted operands

`G17ActuatorOperandRefV1` binds:

- operand ID and closed kind;
- physical coding group ID;
- exact member name, byte offset, byte length, and span SHA-256;
- exact packet schema and section;
- pair start/count; and
- exact predictor-slice `TaskspacePredictorStateV2` binding.

Its member-relative group type is
`G17ActuatorPhysicalSpanGroupV1`, deliberately distinct from the canonical
archive-level `G17PhysicalCodingGroupV1`. The bounded execution view refuses
overlapping groups so the same counted bytes cannot be double-owned or
double-charged; archive/member/coder custody remains owned by the canonical
selected-solution compiler.

`G17ActuatorProgramV1` reopens the member span and delegates strict parsing to
`parse_generative_taskspace_correction_v2`. It refuses:

- wrong member or span hashes;
- overlaps, gaps, or trailing unowned bytes inside a physical coding group;
- pair-population overlaps or gaps;
- packet-declared pair windows that differ from the IR;
- packet/state foreign-key mismatch;
- non-NONE transport; and
- a transport-dependent G family before mutation.

The G packet's existing length, section table, CRC32, and canonical re-encode
checks make every counted packet byte live. The mutation test flips each of the
156 bytes in the real-pair packet one at a time, recomputes the outer span hash,
and proves that the inner canonical packet path still refuses every mutation.

### Real receiver operation

For each exact pair page, execution performs:

1. reconstruct the exact V2 predictor slice with `NoTransportV2`;
2. parse G through `parse_generative_taskspace_correction_v2`;
3. apply G twice through `apply_generative_taskspace_correction_v2`;
4. derive exact semantic ownership through
   `derive_g_correction_ownership_v2`;
5. overlay only owned scorer supports through
   `overlay_g_on_predictor_camera_y1` twice;
6. preserve chronological Y0 exactly;
7. preserve every G-unowned Y1 camera value exactly; and
8. return the changed chronology and typed donor receipts.

No palette full-frame repaint is used. No target labels, teacher arrays, GT,
scorer weights, or dense predictor frames are counted as G operands.

### Strict receipts and checkpoints

The module adds canonical, duplicate-key-refusing, typed parse/re-emit paths
for:

- `G17ActuatorProgramReceiptV1`;
- `G17ActuatorExecutionReceiptV1`; and
- `G17ActuatorCheckpointReceiptV1`.

Program reverify reopens the exact member/state/span graph. Execution reverify
actually re-executes the semantic receiver and camera overlay and compares the
complete receipt. A hash-shaped forged output field can pass structural parse
but fails live reverify, so parsing is not mistaken for execution.

Checkpoints retain an exact program/execution foreign key, input and realized
output state hashes, completed chronological pair prefix, previous-checkpoint
hash, and binding requirements for atomic writes, distinct stage filenames,
and disk resumability. This is the bounded receipt substrate for the later
path-backed streaming executor; it does not claim that an n600 job was run.

## Real ep725 n1 mechanism receipt

Source: exact frozen ep725 prefix through
`decode_ep725_prefix_ephemeral_surface(pair_count=1)`. This surface is already
bound to the counted LVLS1 member, shipped runtime, canonical NumPy renderer,
internal decoder `phi.argmax`, and deterministic shipped/NumPy equality.

Measured bounded facts:

| field | value |
|---|---:|
| source pairs | 1 |
| counted G packet bytes | 156 |
| changed semantic cells | 13,970 |
| actually changed camera values | 166,614 |
| Y0 preserved | true |
| unowned Y1 preserved | true |
| deterministic double replay | true |
| scorer invoked | false |
| n600 execution proven | false |
| public archive/output proven | false |

This row proves receiver mechanics only. It is not distortion evidence, a
candidate, a public decode, or a score.

## Verification

Focused actuator tests:

```text
.venv/bin/pytest -q src/tac/witness_dsl/tests/test_taskspace_g17_actuator_ir_v1.py
7 passed in 25.63s
```

Typed donor regressions:

```text
.venv/bin/pytest -q \
  src/tac/witness_dsl/tests/test_taskspace_predictor_v2_consumer_seam.py \
  src/tac/witness_dsl/tests/test_predictor_preserving_taskspace_overlay.py
14 passed in 1.45s
```

Static verification:

```text
.venv/bin/ruff check \
  src/tac/witness_dsl/taskspace_g17_actuator_ir_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_g17_actuator_ir_v1.py
All checks passed!
```

The focused tests cover the real ep725 pair transition, exact Y0/unowned-Y1
preservation, strict receipt parse/reverify, exhaustive counted-packet byte
mutation, span drift, physical gap/overlap/trailing-unowned bytes, false public
or n600 claims, byte-VM impersonation, teacher-payload claims, V9 Pose6
cross-cast refusal, and checkpoint receipt closure.

## Forest-level interpretation

The missing composition layer was not another representation family. G17 had
typed representations and selected-state evidence, while G29 had a real public
predictor. What was missing was an executable **actuator instruction** joining
four identities without laundering any of them:

```text
exact counted solution bytes
    -> exact predictor-slice state contract
    -> audited receiver physics
    -> realized chronological output state
```

That separation should remain the codec architecture. The byte program stores
solution operands; the generic decoder owns physics; the state contract says
what each operand may read and mutate; the execution receipt proves what
actually happened. This prevents selected plans, costates, or byte spans from
being mistaken for changed video.

## Honest remaining blocker and next score-directed work

The exact frontier pointer is unmoved. G43 closes the first bounded actuator,
not the production row. The next direct path is:

1. extend the ep725 runtime compatibly with a pairwise
   `(Y0, Y1, phi_argmax)` state API while preserving the old `_render_pair`
   bytes exactly;
2. compile fresh, originality-clean, per-pair label-local G pages from current
   scorer/target custody; historical teacher payload may inform mechanism only;
3. build the SSD-preflighted, path-backed, checkpointed n600 executor that
   calls this unchanged bounded receiver on n1 slices and streams one raw;
4. package P once plus the ordered counted G-page directory and measure exact
   archive bytes;
5. run the full frozen n600 scorer/evaluator on the exact emitted video and
   compare against the dynamic effective frontier.

Two boundaries remain explicit:

- G41's strict whole-object receipt is a private research state and records
  `public_rgb_bridge_proven=false`; no real non-test G17 whole-state receipt was
  available to foreign-key into this bounded ep725 execution. Do not manufacture
  that parent edge. The production program must add it only after a real selected
  state and public bridge exist.
- G43 has no path-backed n600 output receipt, no full archive closure, no fresh
  n600 G pages, and no score receipt. Those are the next execution unit, not
  properties of this one.

Pointer delta: **none**. Frontier lowering has not yet been achieved.

## Root adversarial harvest — 2026-07-27

The completed landing had remained uncommitted. Root re-ran the real ep725
mechanism proof and fixed one ontology collision before harvesting it:

- renamed the bounded member-span type to
  `G17ActuatorPhysicalSpanGroupV1`; it no longer creates a second class named
  `G17PhysicalCodingGroupV1` beside the canonical archive-level owner;
- made distinct actuator span groups refuse overlap, preventing duplicate
  physical ownership and byte charging;
- added the corresponding regression; and
- formatted and reverified the exact landing.

Current verification is 8 focused actuator tests and 22 tests across the
actuator plus both typed donor suites. Ruff, format check, pycompile, and
`git diff --check` pass. The mechanism measurements and authority limits above
are unchanged.
