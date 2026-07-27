# G92 — partial population atlas before same-state G94 lowering

Status: **implemented at L1, research-only; same-state lowering fail-closed**  
Lane: `lane_g92_population_global_g89_program_induction_20260727`  
Goal pointer: effective competitive frontier `0.172`; G92 does not move it.

## Correct authority interpretation

G87 owns typed G72 proposals. G90 V1 evaluates finite actuator steps at the
exact G85 incumbent. G89 supplies class-complete semantic program types, and
G94 now supplies the sequential `BOTH -> Y1 -> Y0|Y1` receiver.

The G90 V1 linear screen and Pareto filter are not a complete or optimal search
certificate. Exact-replayed rows form only a **partial atlas**. Rows discarded
before exact replay remain unresolved; they are not negatives. G92 therefore
does not claim that its exact rows, graph-coloured branches, or canonical
operand-ID ordering span the useful program space.

The old `screening_simulation_valid=true` claim is withdrawn. Isolated G90 rows
are exact for the one state they replayed, but no isolated component delta is
transferred to a cumulative G89/G94 state.

## Implemented compiler contract

`taskspace_g92_population_global_program_induction_v1.py`:

- reopens a sealed n600 G90 aggregate, its five stages, and contiguous
  batch-16-or-smaller checkpoints;
- requires canonical proposal dictionaries, proposal fingerprints, exact
  proposed-operand SHA, and separate incumbent-operand SHA;
- reconstructs every exactly replayed `Y1` operand and proves member bytes/SHA;
- retains exactly replayed non-Pareto rows when the source ABI provides them;
  the Pareto bit is provenance, never an admission or completeness predicate;
- records non-exact projection IDs as unresolved screening-only coordinates;
- groups only physical parameters actually shared by the proposal source:
  semantic role, direction rank, and amplitude scale;
- graph-colours donor-address collisions into partial atlas branches. Branch
  order is canonical storage order only, with no prefix optimality;
- counts each exact replay state once. The common base is not counted once per
  graph-coloured branch;
- preserves Seg, Pose, and unmeasured-rate coordinates separately.

G51 is bound only by exact file bytes/SHA as **opaque provenance**. G92 does not
parse its receipt policy and therefore does not infer “teacher-only,”
full-residual, negative, or payload authority from its filename.

## Materializer contract

`materialize_taskspace_g92_population_program.py` writes only immutable
preflight custody, a partial-atlas plan, and a sealed blocker receipt. It emits
no archive and prices no member bytes as ZIP rate. Its plan explicitly records:

- `screening_completeness_claim=false`;
- `exact_replay_atlas_complete=false`;
- `branch_order_semantics=CANONICAL_STORAGE_ORDER_ONLY_NO_PREFIX_OPTIMALITY_OR_COMPREHENSIVENESS`;
- `partial_enumerated_branch_state_count`, excluding duplicate base states.

The active blocker is:

`G90_V2_EXACT_ALL_COARSE_ATLAS_PLUS_G92_TO_G94_LOWERING_PLUS_SAME_STATE_FULL_N600_ROWS_OWED`

## Required lowering chain

1. G90 V2 exactly replays every coarse typed coordinate, not only a
   linear/Pareto-selected subset.
2. G92 maps each exact typed row into G89 source operands without transferring
   isolated deltas.
3. G94 serializes and decodes each cumulative state as
   `[exact G85 PVSA BOTH, G89 Y1 on the same P, G88 Y0|combined-Y1]`.
4. G95 or its equivalent binds every trained conditional operand to G94's exact
   `conditioning_state_sha256`.
5. Each selected state receives full-n600 same-archive Seg/Pose/rate
   measurement through public `inflate.sh` and recursive
   `upstream/evaluate.py`.

Only those same-state rows can feed G83. G92 is not G83-ready and makes no
candidate, score, public-runtime, or pointer-movement claim.

## Authority verdict

G92 is a custody-safe partial-atlas compiler. G94 has closed the previously
missing receiver type, but has not made V1 Pareto screening complete and has
not produced same-state n600 rows. The frontier remains unmoved.
