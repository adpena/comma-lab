# G98 sparse-atlas cumulative G94 lowering (2026-07-27)

Status: **production-real generic seam implemented and fixture-verified; real
n600 materialization blocked on the absent complete persisted G90 V2
aggregate.**

This landing does not claim a candidate, score, pointer move, public receiver
closure, or empirical aggregate benefit. It does not infer an aggregate from
the isolated G90/G92 rows.

## Purpose

G90 V2 defines exact donor-custodied `BoundaryShearletAtomV1` interventions.
G92 can select canonical intervention IDs, but its
`materialize_prefix_family` must fail because the isolated rows do not establish
additivity. G89 is also not a lawful wrapper for the selected sparse Road/UDB
program: G89 requires all five topology roles, both Road/UDB shearlets, and
nonempty Lane and Movable sections. Inventing those missing sections would be
dead filler and a fake implementation.

G98 therefore adds the smallest separate typed seam:

1. validate the complete sealed G90 V2 atlas, G92 plan, exact base archive, and
   each selected donor ID/atom;
2. serialize every selected ID and exact Y1 atom in the counted operand;
3. reopen public `P` and the incumbent base once;
4. append the selected atoms in optimizer-declared counted order to the actual cumulative
   current state;
5. execute the complete current state for every prefix;
6. construct a G94-preconditional uint8 camera pair with incumbent Y0 and
   cumulative Y1; and
7. apply the actual resize-round-u8 `R` operator to every prefix result.

The implementation is
`src/tac/witness_dsl/taskspace_sparse_atlas_cumulative_lowering_v1.py`.

## Typed wire and custody

`SparseAtlasY1OperandV1` is the hermetic counted wire. It contains:

- semantic `P` SHA-256;
- exact base outer-archive and compact-PVSA-member SHA-256 values;
- G90 aggregate file and self-seal SHA-256 values;
- canonical G92 plan SHA-256;
- every selected canonical intervention ID; and
- each exact donor `RoleAwareBoundaryShearletOperandV1(Y1)` and its SHA-256.

The decoder requires only the exact base PVSA member and this counted sparse
operand. It does **not** read the G90 aggregate, G92 plan, receipts, or donor
paths. The G90/G92 identities are embedded provenance/foreign keys, not
decode-time dependencies. STORE/DEFLATE outer-archive construction is
parse-back checked and chooses the smaller exact encoding.

The lowering side verifies that the selected IDs are known and unique in their
declared order; the G90 aggregate and G92 plan agree; the exact replay atlas is marked
complete; the persisted aggregate still has its sealed file identity; base
archive bytes/SHA and the parsed base member agree; and every selected row's
atom tuple, operand bytes, operand size, and operand SHA agree. Unknown IDs,
wrong base/conditioning state, duplicates, and atom collisions fail closed.

## Cumulative realization semantics

For selected steps `a_1, ..., a_k`, prefix `j` is not an isolated replay and
does not add cached deltas. Its semantic receiver is reconstructed as:

`current_j = P.boundary_shearlets union incumbent_G74_BOTH union a_1 ... union a_j`

The canonical atom encoder validates the complete union. The complete receiver
executes twice for deterministic equality. Incumbent Y0 is copied unchanged;
the realized current Y1 is inserted into the G94-preconditional pair; then the
exact base `R` operator realizes scorer-space uint8 frames. Each prefix records
the previous/current native-Y1 hashes, previous/current exact-R-Y1 hashes,
previous/current state hashes, selected ID prefix, pair IDs, and exact arrays.
The current state hash incorporates the previous state hash, so order and
conditioning are explicit even when a later paint operation shadows an earlier
atom at the pixel surface.

No component delta is summed, no isolated exact score is reused, and no
aggregate score is claimed.

## G94 compatibility boundary

The output ABI is
`G94_PRECONDITIONAL_UINT8_CAMERA_PAIR_PLUS_COMBINED_Y1_SHA256_V1`: incumbent
Y0, exact cumulative Y1, exact-R result, and conditioning/current-state
custody.

This is a standalone sparse preconditional producer. G94 V1 remains unchanged
and continues to parse a G89 class-complete operand. A future explicit
versioned G94 union/product may consume the G98 ABI. This landing does not
silently reinterpret the existing G94 wire and does not overlap G95's
P-once/global-basis work.

## Resumable materializer contract

`materialize_next_prefix_checkpoint` is encoder-only and processes exactly one
prefix/batch unit per call. It:

- validates the durable checkpoint root and receiver custody;
- preserves immutable, stage-encoded JSON checkpoints;
- writes with temp-file plus atomic rename;
- verifies any pre-existing checkpoint byte-for-byte and self-seal;
- advances only through complete earlier prefix/batch units; and
- records `research_only=true`, `candidate_claim=false`, and
  `score_claim=false`.

Checkpoints never enter the counted operand. They are measurement/resume
artifacts, not score receipts.

## Fixture scope and verification

The focused fixture uses the exact current G85 public base archive
`b9c8ab2af8886c5b26bba63e02b7c5fe9951bb42a871c5e8472483977788d9fd`
(129392 bytes), exact compact member
`d50aac6eab8114c2c15156354147d1cbfe007b474a0633d5cdec26e66751de31`,
and semantic `P`
`759e28332ce1ea2d4cabba731e4b7b2b21c191fef1bd2b104fab18805388d6df`.
Its two source-shaped donor rows are typed fixtures only; they are not G90
n600 measurements and establish no score fact.

Focused verification:

```text
uv run pytest -q src/tac/witness_dsl/tests/test_taskspace_sparse_atlas_cumulative_lowering_v1.py -x
........                                                                 [100%]
8 passed in 38.83s
```

Adjacent G92/G94 regression verification:

```text
uv run pytest -q \
  src/tac/witness_dsl/tests/test_taskspace_g92_population_global_program_induction_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_g94_sequential_typed_actuator_product_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_sparse_atlas_cumulative_lowering_v1.py
.............................                                            [100%]
29 passed in 56.58s
```

Formatting and lint:

```text
uv run ruff format --check \
  src/tac/witness_dsl/taskspace_sparse_atlas_cumulative_lowering_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_sparse_atlas_cumulative_lowering_v1.py
2 files already formatted

uv run ruff check \
  src/tac/witness_dsl/taskspace_sparse_atlas_cumulative_lowering_v1.py \
  src/tac/witness_dsl/tests/test_taskspace_sparse_atlas_cumulative_lowering_v1.py
All checks passed!
```

The tests cover:

- every selected ID/atom is counted, with no G89 filler;
- direct exact current-state Y1 reproduction and no Y0 mutation;
- cumulative prior-state/hash dependence despite legitimate paint shadowing;
- counted declared-order custody, uniqueness, and collision rejection;
- unknown ID and wrong state/base rejection;
- exact outer-archive parse-back with no research-path dependency;
- atomic immutable checkpointing and forgery refusal; and
- exact base outer/member identity binding.

## Real-materialization blocker and owed chain

`G98_REQUIRES_COMPLETE_PERSISTED_G90_V2_AGGREGATE_BEFORE_REAL_N600_MATERIALIZATION`

The complete persisted G90 V2 aggregate and authoritative selected intervention
sequence were not present during this landing. Therefore the lawful result is
the seam, typed fixtures, tests, and this blocker—not a fabricated aggregate.

Once those inputs exist, the remaining chain is:

1. load and self-seal the complete G90 V2 exact atlas;
2. load the authoritative G92 V2 plan and optimizer-declared selected ID
   sequence; order is an explicit decision variable, not lexicographic policy;
3. lower the exact donor atoms against the exact current base;
4. resume exact cumulative replay over all 600 pairs, prefix by prefix;
5. inspect whole-state Seg/Pose/rate facts without inferred additivity;
6. lower through an explicit versioned G94 sparse-product union and G95
   conditioning surface;
7. build the counted archive/public receiver closure; and
8. run exact `upstream/evaluate.py` on contest-authority hardware.

## Stores consulted

- `CLAUDE.md`, byte-identical `AGENTS.md`, `PROGRAM.md`
- `docs/operating_manual_craft_handoff.md`
- G89 class-complete compiler/receipt
- G90 V2 projected-costate implementation and receipts
- G92 global-program induction implementation/receipt
- G94 sequential typed-actuator product implementation
- G85 compact PVSA public receiver archive/receipt
- lane registry and subagent progress registry
- Claude project memory top-level index and relevant Codex memory hooks
- last-24-hour `.omx/research` directive scan (none found)

Pointer delta: **none; pointer was not read or modified by G98.**
