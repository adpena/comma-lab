# G92 V2 exact-atlas intake amendment

Status: **compile-closed, research-only; empirical G90 V2 aggregate pending**  
Lane: `lane_g96_g92_v2_exact_atlas_intake_20260727`  
Supersession: additive amendment to
`SPEC_g92_population_global_program_induction_20260727.md`; the historical V1
contract and wire shape remain valid.

## Canonical-versus-unique decision

The canonical G92 compiler remains
`taskspace_g92_population_global_program_induction_v1.py`. A parallel compiler
would duplicate custody and branch logic, so V2 enters through a strict
aggregate-schema dispatch. The V1 loader and V1 materialized program schema are
preserved. A G90 V2 source emits the new
`tac.taskspace_g92_exact_coarse_atlas_program.v2` wire schema and a matching V2
blocker; it cannot masquerade as the historical partial-atlas V1 shape.

The unique V2 work is an intake adapter, not a new optimizer. It reopens the
sealed aggregate, five sealed 120-pair stages, and every contiguous
batch-16-or-smaller checkpoint. Every file identity and every canonical
self-seal is verified before any row is exposed.

## Exact V2 contract

The adapter accepts only
`tac.taskspace_exact_coarse_costate_{aggregate,stage,batch}.v2` with:

- full `[0,600)` stage and batch coverage, exact child counts, and reproduced
  base Seg/Pose totals;
- the exact `ALL_DETERMINISTIC_PHYSICAL_GROUPS` policy;
- variable ordered physical-group IDs and counts, never a fixed-eight
  assumption. The contract fixture contains 12 groups for `[288,304)` and 324
  groups over the complete synthetic n600 hierarchy;
- exact equality among expected group IDs, projection rows, basis groups, and
  replay-state custody, including order and coverage;
- canonical proposal fingerprints, proposed-operand bytes/SHA, separate
  incumbent SHA, exact replay fields, Y0 preservation, and Pose/Seg Y1
  conditioning custody;
- scorer-inference cell authority, with differentiable cell/Pose drift retained
  only as non-authoritative telemetry; and
- no Pareto pruning, no local admission, no member-byte rate inference, no
  measured ZIP delta, and no candidate/score/pointer claim.

The resulting atlas is complete only over the isolated deterministic coarse
interventions that G90 V2 actually replayed. “Complete” does not mean additive,
cumulative, hierarchically refined, optimal, compressible, or archive-priced.
Graph colouring supplies collision-free storage branches only; canonical
branch order is not an intervention order or an optimization result.

## Authority, observability, and NO-FAKE boundary

This landing has compile/test authority only. The G90 V2 r5 implementation is
committed at `80841f7b01`, but its governed stage-0 materialization was still
running while this amendment was sealed. No real aggregate was available to
consume. The n600 hierarchy used by tests is a production-schema contract
fixture, not empirical score evidence.

Therefore:

- `empirical_v2_aggregate_present=false`;
- `archive_emitted=false` and `archive_priced=false`;
- `candidate_claim=false`, `score_claim=false`, and `pointer_moved=false`;
- no isolated Seg/Pose component delta is summed, ranked, or transferred to a
  composed state; and
- the competitive frontier remains `0.172`.

## Downstream gate

Successful V2 intake advances the blocker from “build G90 V2” to:

`G92_TO_G94_EXACT_COMPOSED_STATE_REPLAY_PLUS_SAME_STATE_FULL_N600_ROWS_OWED`

The next score-bearing operation is to select/refine a typed Y1 intervention,
realize its exact cumulative state through G94, solve conditional `Y0|Y1`
against that exact state, serialize one public receiver-closed archive, price
its actual ZIP bytes, and run the full n600 upstream scorer. G92 does not
manufacture that evidence from isolated rows.

## Mandatory six-hook declaration

- Sensitivity-map hook: **N/A for new empirical signal.** G92 preserves G90's
  exact rows and authority drift but derives no sensitivity measurement.
- Pareto hook: **active as a refusal gate.** V2 requires
  `pareto_pruning_performed=false`; Pareto/local admission cannot discard or
  promote coordinates.
- Bit-allocator hook: **N/A.** Operand member bytes are not ZIP deltas and G92
  performs no archive allocation or rate pricing.
- Cathedral/autopilot hook: **active.** The typed output targets the existing
  G89/G94 composed-state path and advances one explicit governed blocker.
- Continual-learning hook: **N/A.** No empirical aggregate or authoritative
  candidate row is produced, so there is no learning anchor to ingest.
- Probe-disambiguator hook: **N/A.** Aggregate schema dispatch is closed and
  unambiguous; unsupported or hybrid schemas fail closed.

`research_only=true`  
`council_predicted_mission_contribution=frontier_protecting`

## Triality and stores consulted

- DSL leg: strict dual-schema intake plus distinct V2 materialized wire shape.
- DAG leg: G90 V2 exact isolated atlas -> G92 typed intake -> G94 exact
  composed-state replay -> actual ZIP -> full-n600 upstream row.
- Equation leg: unchanged score authority
  `100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37_545_489`; G92 supplies no
  new value for any term.

Consulted stores: `AGENTS.md`/`CLAUDE.md`, `PROGRAM.md`, lane registry and
subagent progress, the historical G92 spec/receipt, the G90 V2 spec and source
schemas, and direct consumers of the G92 plan schema.
