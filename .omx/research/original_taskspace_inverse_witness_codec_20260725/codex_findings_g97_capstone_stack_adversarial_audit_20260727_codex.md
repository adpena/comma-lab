# G97 capstone-stack adversarial audit

Date: 2026-07-27  
Lane: `lane_g97_capstone_stack_adversarial_audit_20260727`  
Mode: bounded read-only adversarial review  
`research_only=true` · `candidate_claim=false` · `score_claim=false` ·
`promotion_eligible=false` · `pointer_moved=false`

## Executive verdict

The reviewed stack does not yet construct an n600 candidate. Four type/lifecycle
breaks are blockers, not ordinary follow-ups:

1. G90/G92 discovery is never lowered into a sparse, cumulative, realized G94
   Y1 state.
2. G95's counted "population-shared" packet is actually limited to one
   contiguous batch of at most 16 pairs, while its receiver cannot stream
   subsets of that packet. This makes `P once` impossible in the present wire.
3. G94 always embeds and executes G88, but G95 independently owns and overwrites
   Y0. Appending G95 would retain dead G88 bytes and leave two owners for the
   same field. The two conditional-Y0 modes must be mutually exclusive.
4. G29/G31 public closure compiles only the LVPG2-to-LVLS1 format. It has no
   G94/G95 public decoder dispatch, so private receiver equality cannot become
   an `inflate.sh`/`upstream/evaluate.py` row.

These are honest gaps in otherwise deliberately fail-closed research code. The
problem is not permissive truth labels; it is that the present types stop before
the only object that could move the pointer: one byte-closed, public-decoded,
full-n600 semantic-Y1 plus conditional-Y0 archive.

## Frozen review snapshot

- Initial observed committed base: `80841f7b01af` (G90 exact-all landing).
- Final observed shared-tree HEAD: `f2f8c0400ac0de4833af3ec98f3f48689a6ce49c`
  (G92 exact-all intake landed concurrently).
- G92 source SHA-256:
  `90bcb6e469a069ab9b31a02b2aa03431dd723ee076897f970d39fe487f2f6b7e`.
- G89 source SHA-256:
  `01127ec2237b94cb028a702bdee995669ebe98fbd9bab386df38dc5eb822d29b`.
- G94 source SHA-256:
  `e3d6db514897404af4d9c95768183dbeb81a8e7da223d051adc574fa27925e1e`.
- Active uncommitted G95 source SHA-256:
  `de9004a40ec50d59f7fc20e4e2539a2cd50bac7dacabe2c6c9ef53a3924d584c`.
- Public-closure source SHA-256:
  `f7ad196cd92889b9740ce595743dca667eefc387dfce60ed284173ff35528628`.
- Public-closure runner SHA-256:
  `06ea9972978e7b53dd3f22b741122758e3e91de74784ba5c75d5f4bb4a0950e4`.

The tree was shared and dirty. This audit made no source implementation change
and does not attribute concurrent edits to this lane.

## Ranked findings

### 1. BLOCKER — G92 has no sparse cumulative lowering into realized Y1

G90 exact rows retain only a typed
`RoleAwareBoundaryShearletOperandV1(Y1)` containing Road or
UndrivableBoundary atoms
(`taskspace_g92_population_global_program_induction_v1.py:398-455`). G92 then:

- groups rows only by `(role, direction_rank, amplitude_scale)`
  (`:1366-1376`);
- graph-colors address collisions without admission or prefix optimality
  (`:1410-1427`);
- emits a `PopulationProgramPlanV1` made of family and operand IDs
  (`:1430-1453`); and
- unconditionally refuses `materialize_prefix_family`
  (`:1456-1459`).

Therefore even a complete G90 V2 aggregate yields no counted semantic operand,
no cumulative G94 state, no Y1 SHA, and no full-n600 Seg row.

The obvious wrapper is also ill-typed. G89's only program constructor requires:

- topology applications covering all five roles;
- both Road and UndrivableBoundary shearlets;
- a nonempty Lane program; and
- a nonempty Movable worldsheet track

(`taskspace_g89_class_complete_semantic_compiler_v1.py:355-371`).
Sparse G90-selected Road/UDB blocks cannot lawfully inhabit that type without
unmeasured filler. The pair-0 G89 fixture is not a neutral default: it changes
12,453 Y1 channel values under G94 and was explicitly not costate-selected.

Required seam: a sparse selected-semantic program type, or a compatible G89 V2
whose optional streams are genuinely absent rather than filled with fixture
content. It must preserve exact G90 atom bytes/provenance and prove:

1. each isolated lowered G94 Y1 equals its G90 measured Y1 byte-for-byte;
2. every admitted prefix is replayed cumulatively on the previous realized
   state, never assembled from additive isolated deltas; and
3. the selected full program receives one exact n600 Seg replay and one actual
   outer-ZIP price.

Until that seam exists, G92 is an exact-atlas intake and plan, not program
induction in the candidate-producing sense.

### 2. BLOCKER — G95 leaks the 16-pair execution batch into the counted population type

G95 declares `PAIR_COUNT=600` and `MAX_BATCH_PAIRS=16`, but
`_validate_pair_ids` requires the packet's complete address set to be one
contiguous 1..16 range
(`taskspace_g95_population_pose_preimage_chart_v1.py:35-40,123-130`). The
encoder writes that bounded count into the packet header (`:262-275`), the
parser caps the header count to 16 (`:515-554`), and the receiver refuses any
decode whose requested IDs are not exactly the packet's complete IDs
(`:882-910`).

This is not merely a bounded measurement configuration. It is the production
packet and receiver ABI. A single packet cannot contain the n600 coefficient
population, and a hypothetical n600 packet could not be decoded in evaluator
batches. Encoding 38 bounded packets would duplicate the dense learned basis
38 times, contradicting the population-global `P once` rate premise.

Required seam:

```text
one counted population packet
  = one shared basis
  + one canonical 600-row coefficient table (or sparse indexed rows)
  + one final-G94 conditioning foreign key
  + streaming-verifiable preconditional custody

receiver execution
  = select coefficient rows for requested <=16 pair IDs
  + regenerate/verify that G94 preconditional chunk
  + apply the shared basis once by reference
```

The wire's population count and the receiver's working-set batch count must be
distinct types. Whole-state or per-chunk digest custody may be used, but the
basis must have exactly one counted byte home.

### 3. BLOCKER — G88 and G95 are alternative Y0 owners, not composable members

G94 V1 is a fixed three-section wire:

```text
base PVSA G74 BOTH -> G89 Y1 -> G88 conditional Y0
```

Its encoder requires a G88 section (`taskspace_g94_sequential_typed_actuator_product_v1.py:190-224`),
its parsed type contains exactly that operand (`:228-273`), and its receiver
always executes G88 after producing combined Y1 (`:702-751`).

G95's base law instead constructs Y0 as `exact conditional Y1 + residual` while
preserving Y1 (`taskspace_g95_population_pose_preimage_chart_v1.py:911-946`).
It does not consume G88's final Y0. Therefore attaching G95 after current G94
would discard the G88 result while still paying for the G88 operand and would
leave two typed owners for Y0.

The next G94 wire must use a tagged, mutually exclusive conditional-Y0 union:

```text
conditional_y0_mode :=
    G88_EXISTING_CONDITIONAL_OPERAND
  | G95_POPULATION_PREIMAGE_CHART
```

For the G95 branch, no G88 payload may be present or charged. G95 must be fit
only after the final cumulative semantic Y1 is frozen, because G94's
conditioning hash changes with the G89/semantic-program SHA
(`taskspace_g94_sequential_typed_actuator_product_v1.py:144-168,297-301`).
The active G95 runner is explicitly bound to the non-final fixture constants,
so its result cannot transfer to a future selected G92/G94 state.

### 4. BLOCKER — G29/G31 does not provide public dispatch for G94/G95

The public compiler's archive inspector requires exactly one `0.bin` that
parses as an LVPG2 population-global member
(`taskspace_public_auth_eval_closure.py:1934-1968`). Its sole compiler is
`compile_lvpg2_public_runtime` (`:2028-2147`), producing exactly
`inflate.sh`, `inflate.py`, and `lvls1_runtime.py`. The runner accepts an
`--lvls1-runtime` and calls only that compiler
(`tools/run_taskspace_public_auth_eval_closure.py:330-371,403-408`).

Neither G94's three-section typed product nor G95's chart packet has a public
parser/receiver in that runtime tree. G31 already classified G29 as a
compile/discovery/preflight apparatus without official execution closure.
Reusing its receipt schemas does not make a new payload format executable.

Required seam: a capstone public-runtime adapter that parses the selected
G94-V2/G95 outer member, regenerates P, streams all 600 pair outputs, and proves
byte equality to the private receiver before the existing authority collector
is used. The adapter itself then needs the G31 gates: public recursive execution,
two-run output identity, scorer-input/output ledgers, whole-job timing, and
contest-CPU and/or contest-CUDA authority.

### 5. REQUIRED MACRO GATE — G95 reachability is not score/rate reachability

The active G95 ladder admits on `d_pose <= 0.00047366`, while its receipt leaves
`outer_zip_delta=null` and explicitly records the outer-race blocker
(`tools/measure_taskspace_g95_population_pose_preimage_chart.py:1816-1825,2337-2347`).
That threshold is not sufficient for a sub-0.172 archive.

At the threshold:

```text
pose term = sqrt(10 * 0.00047366) = 0.0688229613
G94 fixture bytes = 129,799
fixture rate term = 25 * 129799 / 37545489 = 0.0864278
```

Even with perfect Seg, sub-0.172 permits at most approximately 154,953 total
archive bytes, only about 25,154 bytes beyond the G94 fixture. With
`d_seg=0.0001`, the remaining payload headroom falls to about 10,136 bytes.

A correctly shaped rank-6 n600 raw packet at 48x64 RGB would contain:

```text
shared int8 basis        55,296 bytes
600 x rank int16 coeffs   7,200 bytes
pair IDs, scales, header  1,548 bytes
total                    64,044 bytes
```

Outer compression may reduce this, so this is not an impossibility verdict.
It is a mandatory byte-closed gate: rank success must be selected by actual
whole-archive score, not by Pose threshold alone. Even rank 6 must compress
below roughly 25.2 KB with perfect Seg, and below roughly 10.1 KB at
`d_seg=0.0001`, unless the semantic/base archive becomes smaller.

The semantic side needs the same macro discipline. G90's exact base has
`d_seg=0.02747120`, a Seg contribution of `2.74712`; the strongest reported
first-batch isolated group improved only `0.0024032593` score units. That does
not kill the family, but it means atlas completion is not evidence of target
reachability. Only cumulative full-n600 realized prefixes can establish the
necessary multi-order reduction.

## State/type ledger

| Stage | Input authority | Output actually present | Candidate-producing debt |
|---|---|---|---|
| G90 V2 | exact current G85 and scorer-separated replay | isolated exact Road/UDB group rows | refinement and cumulative selected state |
| G92/G96 | strict five-stage exact-all intake | family/branch plan | sparse typed lowering into realized G94 Y1 |
| G89/G94 | exact P, mandatory class-complete fixture, incumbent BOTH, G88 | executable pair-0 structural product | selected n600 semantic program and conditional-mode union |
| G95 | one fixed non-final G94 fixture, <=16 pairs | counted bounded packet and private NumPy replay | final-Y1-conditioned one-basis n600 packet plus whole-ZIP race |
| G29/G31 | LVPG2 archive and LVLS1 renderer | public runtime compile/discovery/preflight | G94/G95 dispatch and official recursive authority execution |

## Triality

### DSL

The minimum missing candidate DSL is:

```text
SelectedSemanticY1V2(
  semantic_p_sha,
  incumbent_both_sha,
  ordered_sparse_atom_blocks,
  g90_exact_row_provenance
)

ConditionalY0V2 =
  G88Existing(...)
  | G95Population(
      final_conditioning_state_sha,
      shared_basis_once,
      coefficients_n600,
      streaming_custody
    )
```

No fixture content may stand in for an absent selected stream, and the two Y0
variants cannot coexist in one archive.

### DAG

```text
G90 V2 exact-all aggregate
  -> select/refine collision-free semantic blocks
  -> sparse typed G89/G94 lowering
  -> cumulative full-n600 Y1 replay after each admitted prefix
  -> freeze final semantic Y1 + conditioning hash
  -> fit one-basis n600 G95 conditional Y0
  -> G94-V2 conditional-mode union + STORE/DEFLATE race
  -> capstone public runtime compile
  -> private/public n600 byte equality
  -> upstream/evaluate.py exact authority
  -> pointer arbitration
```

### Equations/invariants

```text
S(A) = 100*d_seg(A) + sqrt(10*d_pose(A)) + 25*bytes(A)/37_545_489

Y1(lower_g94(single_g90_row)) == Y1(exact_g90_row)  byte-for-byte

state(k+1) = receiver(state(k), selected_block(k+1))
not sum(isolated_delta(k))

B_g95 = B(shared_basis_once) + B(coefficients_n600) + B(custody)
not sum_over_batches(B(shared_basis) + B(batch_coefficients))

owner(Y0) = exactly_one_of(G88, G95)
owner(Y1) = selected_semantic_program; immutable under conditional-Y0 decode
```

## First real closure rung

The highest-EV next artifact is not another atlas summary or pair-0 chart. It is
one collision-free selected G90 block lowered through a sparse semantic-program
type into G94, with:

1. byte-identical Y1 versus the G90 exact isolated replay;
2. a second selected block applied cumulatively and fully rescored;
3. an outer STORE/DEFLATE price; and
4. a frozen conditioning SHA handed to a redesigned n600-capable G95 packet.

That rung simultaneously tests the missing lowering, receiver transition,
nonadditivity, and rate premise before spending a full pose fit on a Y1 state
that may change.

## STORES CONSULTED

- `CLAUDE.md` / `AGENTS.md` (byte-identical governing contract)
- `PROGRAM.md`
- `.omx/state/lane_registry.json`
- `.omx/state/active_lane_dispatch_claims.md`
- `.omx/state/subagent_progress.jsonl`
- `.omx/research/original_taskspace_inverse_witness_codec_20260725/SPEC_g90_projected_population_costates_20260727.md`
- `.omx/research/original_taskspace_inverse_witness_codec_20260725/SPEC_g90_v2_exact_all_coarse_costates_20260727.md`
- `.omx/research/original_taskspace_inverse_witness_codec_20260725/SPEC_g92_population_global_program_induction_20260727.md`
- `.omx/research/original_taskspace_inverse_witness_codec_20260725/SPEC_g92_v2_exact_atlas_intake_amendment_20260727.md`
- `.omx/research/original_taskspace_inverse_witness_codec_20260725/SPEC_g89_class_complete_semantic_compiler_20260727.md`
- `.omx/research/original_taskspace_inverse_witness_codec_20260725/SPEC_g94_sequential_typed_actuator_product_20260727.md`
- `.omx/research/original_taskspace_inverse_witness_codec_20260725/SPEC_g95_population_pose_preimage_chart_20260727.md`
- `.omx/research/original_taskspace_inverse_witness_codec_20260725/SPEC_g29_public_decoder_auth_eval_closure_20260726.md`
- `.omx/research/original_taskspace_inverse_witness_codec_20260725/codex_findings_g31_upstream_recursive_auth_audit_20260726_codex.md`
- the corresponding G89/G90/G92/G94/G95/public-closure source and runner files
- `/Users/adpena/.codex/memories/MEMORY.md` (July 26 capstone/public-closure recall)

## Pointer-delta honesty

Observed effective competitive pointer: `0.172`.  
This audit produced no archive, no decoder output, and no scorer execution.
Pointer delta: exactly `0.0`. The mission remains unachieved.

