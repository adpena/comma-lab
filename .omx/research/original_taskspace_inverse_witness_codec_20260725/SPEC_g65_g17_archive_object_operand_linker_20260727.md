# G65 — G17 archive-object conditional operand linker

Date: 2026-07-27  
Lane: `lane_g65_g17_archive_object_operand_linker_20260727`  
Authority: structural, `research_only=true`

## Intended link

The intended proof was:

1. locate the uniquely spelled G64 conditional operand in one strict active A
   descriptor;
2. mutate/rebuild A, rebuild E, rebuild and reparse the exact canonical P-G-A-E
   ZIP;
3. rebuild the canonical `G17CompilerPlacementManifestV1` against the mutated
   archive; and
4. execute G64 twice and record deterministic `CHANGED` or `REFUSED`.

The exact baseline custody portion is implemented and tested. The receiver link
must fail closed before step 3 because the canonical ownership ontology cannot
represent it honestly.

## Exact blocker

`G17PhysicalCodingGroupV1` has one `receiver_consumer` and one
`receiver_operation` for an entire physical group. For a compressed ZIP, the
honest charged range is the whole archive shared by every counted logical
owner. Assigning that whole group's sole receiver fields to G64 would falsely
state that G64 consumes all co-coded owners. Splitting the compressed ZIP into
logical-owner-local byte ranges would invent ownership that the compressed
stream does not have.

The missing additive type is:

`physical_group_id + logical_owner_id + receiver_consumer + receiver_operation`

That incidence must coexist with the whole-archive shared physical group. G64
must validate its conditional owner-specific edge through that incidence. Only
then may G65 rebuild a mutated archive/manifest and attach `CHANGED` or
deterministic `REFUSED` to the exact archive object.

Blocker ID:
`G17_SHARED_PHYSICAL_GROUP_OWNER_SPECIFIC_RECEIVER_INCIDENCE_TYPE_OWED`

The prior G64 blocker
`G17_CONDITIONAL_LOGICAL_OPERAND_TO_ARCHIVE_RANGE_LINK_OWED` remains open.

## Implemented fail-closed evidence

`taskspace_g17_archive_object_operand_linker_v1.py` verifies:

- exact operation/manifest/archive identity;
- strict double parse of the same ZIP;
- deterministic canonical STORE/DEFLATE rebuild equality;
- exact member name and bytes;
- one conditional chronological-pose operand;
- operand occurrence `(P,G,A,E) == (0,0,1,0)`; and
- exactly one active A descriptor containing the operand.

It records the only honest prospective range shape as
`WHOLE_ARCHIVE_SHARED_MANY_TO_MANY`, records zero local ZIP operand-owner range
claims, sets `archive_object_receiver_liveness_proven=false`, and refuses the
link API with the ontology blocker.

No scorer, teacher, target, dense-plane fallback, candidate, score, or pointer
claim is present. No exact-eval or heavy run was launched.

## Verification

Verified after formatting both Python artifacts with Ruff 0.15.20:

```text
uv run ruff format --check \
  src/tac/witness_dsl/taskspace_g17_archive_object_operand_linker_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_g17_archive_object_operand_linker_v1.py
# 2 files already formatted

uv run ruff check \
  src/tac/witness_dsl/taskspace_g17_archive_object_operand_linker_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_g17_archive_object_operand_linker_v1.py
# All checks passed

.venv/bin/python -m py_compile \
  src/tac/witness_dsl/taskspace_g17_archive_object_operand_linker_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_g17_archive_object_operand_linker_v1.py
# exit 0

.venv/bin/pytest -q \
  src/tac/witness_dsl/tests/test_taskspace_g17_archive_object_operand_linker_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_conditional_fullframe_receiver_operation_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_g17_production_envelope.py \
  src/tac/witness_dsl/tests/test_taskspace_selected_solution_compiler.py
# 43 passed in 0.97s
```

Formatted artifact custody:

- production module: 12,306 bytes,
  SHA-256 `21c42ffcde37dc86f7019f4923a9d7fc01bb493403d0758e7995525062080d1d`;
- test module: 9,485 bytes,
  SHA-256 `a99dd5e6d944ac18a0579eee9756c824f46e2427812b96c124ae9b6c784206ac`;
- regenerated structural audit receipt:
  SHA-256 `c25f29cb54ebc7cbcc4815dd291eefcb9b552f4edc56129462b44fa35b57dca4`.

## Verdict

`BLOCKED`, verdict scope: current canonical placement ontology.

This is not a negative on the conditional-archive mechanism. It is a precise
type-system blocker. Closing it requires the additive owner-specific receiver
incidence and corresponding G64 validation; it does not require inventing ZIP
subranges or changing codec physics.
