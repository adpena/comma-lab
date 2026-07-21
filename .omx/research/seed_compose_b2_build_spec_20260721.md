# Seed-compose B2 real-n600 build specification

`lane_id=lane_seed_compose_b2_20260721` · `research_only=true` ·
`[macOS-CPU advisory]` · pointer `0.1910828242 [contest-CPU] UNMOVED` ·
`MAIN_REVIEW_REQUIRED=true`

## Objective

Compose and measure the first real n600 `predict_project_constraint_seed.v0`
from the solved-object custody already on main.  The landing must reuse the
Task #597 schema/receiver/measurement runner, the #595 finite event packet,
the exact n600 L-star/Pose cache, the M1/M2 receipts, the existing xi and
Hungarian-track implementations, and registered LawRefs.  It must not claim
an archive, score, native Morse-Smale raster, or receiver-closed RGB
realization.

## Premise audit and binding corrections

1. `tools/measure_predict_project_receiver.py` presently derives declared
   corrections but passes the uncorrected predictor field to its hard-oracle
   callback.  B2 therefore cannot certify the represented seed.  Extend the
   callback contract additively with the represented field, keep the original
   predictor for B3, and test the distinction.
2. `MS_ARC_TO_CELL_RASTERIZATION_SEMANTICS_UNMEASURED` remains live.  D1 uses
   the existing deterministic site-raster compatibility path and labels the
   gap.  It does not invent native bulk-cell semantics.
3. The #595 17,926-event packet is a literal cell-transition candidate, not a
   second full-seed schema and not a vineyard lifecycle stream.  Decode it
   through `tac.optimization.s2_partition_seed`, bind each selected target to
   the exact L-star cache, and place predictor violations in the existing
   `constraint_seeds` section.  Track lifecycle events remain the existing
   grammar's event alphabet.
4. A hard-oracle row can prove constraint-description fidelity and banked
   Pose-tube membership.  Until camera RGB is regenerated from the seed and
   scored through R, it cannot prove receiver-closed distortion.  All D2/D5
   language and equation scope must preserve that boundary.

## Implementation ownership

- Extend `tools/measure_predict_project_receiver.py`; do not fork it.
- Add one compression-side composer/oracle module or tool only if it imports
  the existing schema, receiver, #595 interpreter, xi, track, and LawRef
  surfaces.  It must not duplicate serialization, prediction, scoring, or
  coding logic.
- Add focused tests beside the existing Task #597 tests.
- Land dated receipt JSON, findings memo, DAG FEED, and this canonical reuse
  table.

Do not edit upstream scorer code, frontier pointers, live runs, the #595
interpreter, M1/M2 receipts, or unrelated shared surfaces.

## D1 curve construction

The source is the exact cache SHA-256
`cf8d83605d2198ef56786c6be23d3470033ad2763f59559f06a79cedfb7b8cd6`.
Access ZIP_STORED arrays by bounded memmap.  The compatibility chart is a
deterministic temporal per-site mode with first-max ties.  One xi trajectory
uses the cache Pose rows through the existing Lie/spline convention.  Movable
components are extracted and tracked with the existing #234 Hungarian
implementation.  The #595 packet supplies candidate cell sites; target IDs
are re-read from L-star so the one recorded stale target cannot propagate.

Emit a measured curve, not one arbitrary seed.  Operating points are derived
from the ordered candidate inventory and measured residual distributions:

- event/cell stream: nested EV-ranked prefixes, with Fisher/top1-top2 margin,
  edge/necessity, and existing #595 custody recorded;
- xi stream: nested quantization/residual rungs derived from the measured Pose
  residual distribution;
- Pose tube: nested radii derived from the banked-row residual distribution;
- predictor-default encoding: omit every selected record already satisfied by
  the actual generic predictor.

Each point serializes with `serialize_constraint_seed`, parses with
`parse_constraint_seed`, reserializes byte-identically, and reports
`component_byte_accounting` plus exact whole-seed bytes.  Report raw and
deterministically compressed bytes per section, class, stratum, and stream
against 216,222 B and the 222,447.0271 B optimistic context comparator.

Every selected point records the LawRef IDs it consumes.  The allocation
price is resolved from `realization_breakeven_bytes_v1`; no copied timestamped
lambda is allowed.  The Fisher/margin, realization, curvelet/shearlet, xi, and
master-action identities are references, not re-derived constants.

## D2/B2 measurement

Run the existing measurement CLI on the real composed seed at n16, then n64,
then n600, with seed 1234, batch 16, bounded chunks, preserved per-pair stages,
and source/config hashes.  The hard-oracle adapter must receive both:

- `predicted_cells`: the uncorrected receiver output used only for B3; and
- `represented_cells`: the field after applying declared seed constraints,
  used for B2 cell fidelity.

For every pair, prove declared-site target identity, Pose target/tube
membership under the banked rows, and double-decode identity.  Report global
cell-description mismatch separately from declared-site exactness.  If the
live CPU-Torch scorer or a universally proved pixel Pose tightening is absent,
surface the literal blocker and do not relabel cache replay as live Torch or
receiver-closed RGB authority.

Register `predict_project_cell_tube_uint8_projection_v1` only if all of the
predecessor registration gates are genuinely met.  Otherwise leave the
registry untouched and land the measured, narrower successor equation or an
explicit registration blocker, as appropriate.

## D3 decomposition

Aggregate predictor satisfaction by target class and by
`cell_interior`/`boundary_codim1`/`movable_track`/`critical_event`.  For every
low-satisfaction cell, classify the cause as predictor miss, selected
constraint tightness, compatibility-chart limitation, track miss, or event
exception.  Re-emit default-predictor-right seeds and report exact byte deltas
at equal represented constraints.

## D4 factorization

Compare the same represented cells in the single-object seed and 600
per-frame equivalents.  Report raw and compressed bytes per component and the
measured cut.  The 40--50 percent bar is a comparator, not a pass assumption.

## D5 local KKT neighborhood

Across the measured nested points, evaluate exact advisory
`delta fidelity + lambda * delta bytes` under the LawRef-resolved price.
Report the cross-stream knee and its immediate tighter and looser neighbors,
with per-stream/class/stratum decomposition.  The eat-the-flip action absorbs
points past break-even.  The resulting `(seed bytes, cell-description
fidelity, banked Pose-tube fidelity)` triples are #578 inputs only: no archive,
score, exact-eval, promotion, or pointer claim.

## Acceptance and landing

- Real cache and #595 packet hashes verified; SSD storage preflight recorded.
- Deterministic parse-back and repeated serialization for every emitted seed.
- Focused pytest, Ruff, py_compile, JSON parse, and `git diff --check` green.
- Two clean `review_tracker` passes after the final edit of every changed
  Python file.
- Serializer commit with post-edit expected SHA-256 values; no override and no
  co-author trailer.
- Receipt and memo label every row `MEASURED`, `DERIVED`, or `SPECULATIVE`,
  state verdict scope, preserve the native-raster/Pose-realization blockers,
  and require MAIN review.

## Canonical reuse manifest

| Component | Canonical source | Disposition |
|---|---|---|
| seed grammar/serialization | `tac.optimization.predict_project_schema` | reuse and narrowly extend only if required |
| predictor/projection/accounting/cache | `tac.optimization.predict_project_receiver` | reuse as-is except additive represented-field contract if needed |
| B1--B5 runner | `tools/measure_predict_project_receiver.py` | extend in place |
| #595 packet | `tac.optimization.s2_partition_seed` | reuse as-is; decode only |
| n600 target/Pose cache | `experiments/results/mlx_fleet_gt_cache/gt_n600.npz` | read-only memmap |
| xi spline | `tac.lie` | reuse as-is |
| movable correspondence | `tac.boundary_math.movable_site_coder` / #234 | reuse as-is |
| section coding | existing #557 grammar contexts/coder stack | reuse identities and existing serialization; no fork |
| M1/M2 evidence | commit `9bd01e2232` and `m2_live_target_selection_20260720T1548Z.json` | read-only custody |
| equations | `tac.witness_dsl.lawref` plus registered equation builders | resolve by ID; no copied numeric fallback |

`NEW-with-justification`: one Task #597 successor composition/oracle surface is
allowed only because no real-seed producer or constraint-description oracle
exists.  It must call every row above and must not become a parallel receiver,
schema, scorer, or coder source of truth.

## STORES CONSULTED

Delegated authority and both live inboxes; `CLAUDE.md`; `AGENTS.md`;
`PROGRAM.md`; craft handoff; v7.5/v8 operating specs; latest Codex findings and
session summary; Task #597 build spec/receipt/DAG; #595 code/receipt/audit;
M1/M2 receipts; `reports/latest.md`; lane/task/subagent/equation state; exact
n600 cache; operator curve directive through `2026-07-21T07:38:02Z`.
