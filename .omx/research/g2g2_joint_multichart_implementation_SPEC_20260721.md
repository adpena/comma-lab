# G2g2 joint multi-chart solve — implementation specification

`lane_id=lane_g2g2_joint_multichart_solve_20260721` · `task=578` ·
`research_only=true` · `[macOS-CPU advisory]` · `seed=1234` ·
`pointer=0.19108 [contest-CPU] UNMOVED` · `MAIN_REVIEW_REQUIRED=true`

## Objective and verdict boundary

Extend the merged G2g receiver measurement one rung deeper: for each of the
six G2f-selected pairs, jointly choose a sparse set of decoded openpilot
LaneLine centerline-coefficient deltas, emit them as one canonical multi-row
G2CS1 packet, and test each selected-cardinality prefix through the real
receiver, factor-2 uint8 realization, frozen CPU-Torch SegNet/PoseNet oracle,
the #549 full semantic-description equality, and the declared pose tube.

The implementation must distinguish:

- a measured admitted row, which is the first realization-line correction
  only if every hard predicate passes;
- a bounded-search stall, which is a formulation-scoped negative and never a
  proof that the chart family is infeasible;
- chart-manifold underreach, where all 20 available centerline coordinates
  have been selected but hard semantic/tube debt remains. This is evidence
  against this LaneLine realization formulation, not a rate-domination proof
  unless a receiver-closed describe-line byte comparator is actually present.

No archive or auth-axis evaluation is authorized. The contest pointer stays
unchanged. Run n64 only after a six-pair admission. Refuse n600 absent that
n64 admission.

## Existing surfaces and ownership

Extend, do not fork:

- `src/tac/optimization/predictor_upgrade_xi_chart.py`: existing G2CS1
  multi-row packet/parser/application. Add only packet-size or validation
  helpers if the measurement needs them. Preserve the exact 12-byte header +
  8-byte row format, fp32 deltas, canonical address order, CRC, duplicate
  refusal, canonical re-encode, and rule-118 statement.
- `tools/measure_realization_g2_lattice.py`: add one mutually exclusive
  `--joint-chart-symbol-solve` mode and its output-root option. Reuse G2g
  baseline replay, chart raster, factor-2 realization, hard oracle, scorer and
  #549 seed/cache custody, G1 motion, palette, and SSD checkpoint conventions.
- `src/tac/tests/test_predictor_upgrade_xi_chart.py`: prove nonempty multi-row
  packet size is `12 + 8*k`, parse/re-encode identity, application at several
  addresses, and fail-closed ordering/duplicates.
- Add focused pure tests for deterministic projected greedy/swap selection and
  receipt summarization in the existing test modules; do not create a new
  module unless repository search proves the established surfaces cannot own
  the behavior.

Do not edit upstream, the frontier pointer, main, other worktrees, live run
directories, scorer weights, or unrelated vehicle files.

## Coefficient-response manifold

For each selected pair, enumerate all decoded LaneLine centerline addresses
`(line_index, coefficient_index)`. The current LBND2 custody has five lines
with four centerline coefficients each, hence 20 pair-local addresses; derive
and validate this from the decoded packet rather than hardcoding 20.

For every address derive its coefficient-to-native-centerline-pixel gain over
active raster rows using the established `_line_row_params` geometry. Generate
the amplitude alphabet from the already measured G2f LawRef-bound ladder
`[0.5, 1, 2, 4, 8, 16]` native scorer pixels and both signs. Do not invent a
new numeric ladder.

Build a realized low-dimensional response at the smallest signed rung:

1. Emit the one-symbol positive and negative G2CS1 packets through
   `decode_lane_chart_with_symbols`.
2. Rasterize the whole coherent Lane coverage change, realize factor-2 camera
   bytes, and run the frozen scorer with batch geometry one.
3. Form a central secant for all five SegNet logits at every scorer cell and
   all six PoseNet outputs. This is the corrected inner-Jacobian surface; do
   not use a naive source-RGB or pre-R derivative.
4. Keep fixed target-vs-each-rival Fisher/margin constraints for the full #549
   represented semantic field. Pose constraints use the exact quantized tube
   bounds already consumed by the G2g hard oracle.

Do not persist full logits, response tensors, camera frames, or coverage
planes. Persist only hashes, dimensions, aggregate secant custody, and solver
rows. One pair lives in memory at a time.

## Deterministic projected greedy-with-swap solve

Use a bounded deterministic projected coordinate search over the coefficient
amplitude lattice:

1. Start at the empty packet.
2. At cardinality `k`, evaluate every unused address at every signed registered
   amplitude in the response model. Select the candidate with the best strict
   lexicographic constraint key: full-field predicted semantic mismatch count,
   quantized pose-tube outside debt, summed negative Fisher/target-margin debt,
   then canonical address/amplitude tie-break. Do not add a coefficient unless
   the model key strictly improves.
3. Coordinate-polish selected amplitudes on the same lattice; then try one
   selected/unselected swap. Accept only strict key improvement, with bounded
   passes and explicit `STALLED_MODEL`, `EXHAUSTED_ALL_ADDRESSES`, or
   `MODEL_ADMITTED_CANDIDATE` status.
4. Preserve every cardinality prefix, encode it as a sorted multi-row G2CS1
   packet, and charge `len(packet)` as the only correction-byte authority.

This is a bounded projected greedy-with-swap solve, not MIQP/global-optimum
proof. The receipt must state its search completeness and verdict scope.

## Hard receiver/oracle replay

For every selected k-prefix, independently:

- parse and canonical re-encode the actual G2CS1 bytes;
- decode LBND2 + apply all coefficient rows;
- rasterize and realize exact factor-2 uint8 camera frames twice;
- require double-decode equality and receiver-derived RGB custody;
- run frozen CPU-Torch SegNet and PoseNet at batch one;
- record full semantic equality, declared write survival, pose-tube status,
  semantic mismatch count, d_seg, d_pose, pose-tube debt, packet bytes,
  changed pixels, saturation, recovered distortion score, and marginal
  score-units per actual packet byte versus the LawRef-bound
  `realization_breakeven_bytes_v1` rate price.

Admission is the conjunction:

`semantic_exact && pose_tube && factor2_uint8_exact && double_decode && receiver_RGB && counted_bytes && rate_above_lambda`.

The headline is the minimum measured admitted k/bytes, if any. If none, report
the complete measured k-to-bytes/debt curve and the narrow status. Never turn
model prediction into a hard predicate.

## Execution order, resumability, and storage

- Candidate order is the settled G2g order: chart-only `[0,34,37,46]`, then
  overlap `[22,30]`. Revalidate it from the hash-pinned G2f receipt.
- SSD root:
  `/Volumes/VertigoDataTier/pact/evidence/g2g2_joint_multichart_20260721/`.
- Preflight at least 1 GiB free. Use an invocation-specific run directory.
- Atomic immutable pair stages, checkpoint after every two pairs, config and
  implementation-source hash refusal on resume, and final top-level receipt.
- Keep only small packets/JSON. Do not persist camera, logits, coverage, or
  response tensors. Never delete uncertified prior G2g/G2f evidence.
- Re-running a completed root must reproduce the final receipt byte-for-byte
  or fail closed.

## Required receipt and durable artifacts

The SSD receipt and repo summary must contain:

- D1 solver method/status, address count, response custody, each pair's k and
  actual G2CS1 bytes, multi-row parse-back and rule-118 predicates;
- D2 full predicate table for every hard-replayed k-prefix and the admitted
  count/minimum admitted k;
- D3 deltas/marginals/routes only if admitted; otherwise the measured k curve,
  no-route status, and scoped realization-vs-describe conclusion;
- exact input/source/config hashes, hardware/axis, seed, command, timings,
  storage hygiene, checkpoint/resume state, pointer honesty, and MAIN review;
- a REUSE MANIFEST covering G2CS1, G2g oracle/replay/ranker, #549, LBND2,
  #557 byte discipline, #580, #547, LawRefs, and failed-search justification;
- a DAG FEED that routes sensitivity, Pareto/rate constraint, allocator,
  autopilot, posterior, and the next disambiguator without granting launch or
  promotion authority.

Durable repo files after measurement:

- `.omx/research/g2g2_joint_multichart_solve_<UTC>.md`
- `.omx/research/g2g2_joint_multichart_solve_<UTC>.json`
- `.omx/research/g2g2_joint_multichart_solve_DAG_FEED_<UTC>.md`
- `.omx/research/g2g2_joint_multichart_solve_REUSE_MANIFEST_<UTC>.md`

## Acceptance checks

Before measurement:

1. Focused packet/solver tests, Python compile, Ruff check and format check.
2. A synthetic objective proves deterministic add, polish, one-swap,
   duplicate-address refusal, exhaustion, and minimum admitted-prefix summary.
3. Multi-row G2CS1 test covers at least three addresses and exact `12+8*k`
   bytes.

After implementation, the parent reviewer—not the nested implementer—will:

1. inspect every diff and the no-fake boundary;
2. run the six-pair SSD measurement and immediate resume check;
3. write measured durable memo/receipt/DAG/reuse files if the runner does not;
4. obtain two clean `review_tracker` passes for each changed `.py` file;
5. commit through `tools/subagent_commit_serializer.py` with post-edit SHA-256
   values, verify `git show --stat`, and require MAIN branch-diff review.

## STORES CONSULTED

Delegated authority and both live inboxes; `CLAUDE.md`; `AGENTS.md`;
`PROGRAM.md`; craft handoff; v7.5/v8 specifications; `reports/latest.md`;
lane/task/subagent state; merged G2g memo, receipt, DAG and reuse manifest;
G2f chart response receipt/memo; G2e secant findings; G2CS1 source/tests;
#549 seed/cache/tube and hard oracle; openpilot LBND2/static/palette custody;
#557 coefficient-codec discipline; #580 full resize kernel; #547 integer
lattice; registered Fisher/margin, corrected inner-Jacobian, and reverse-
waterfill directives.
