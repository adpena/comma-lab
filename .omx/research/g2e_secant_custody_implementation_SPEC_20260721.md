# Task #578 G2e realized-secant custody — implementation specification

`lane_id=lane_g2e_secant_custody_578_20260721` · `BUILD+MEASURE` ·
`research_only=true` · `[macOS-CPU advisory]` · `score_claim=false` ·
`pointer=0.1910828242 [contest-CPU] UNMOVED` · `MAIN_REVIEW_REQUIRED=true`

## Objective

Close, or narrow with measured D-rows, the exact predecessor blocker
`R1B2_RANK4_FIRST_ORDER_REALIZED_SECANT_CUSTODY_ABSENT`.  Starting from the
measured compact contextual/openpilot RGB base, measure candidate-arrangement
rank-4 first-order response, full-receiver finite secants, and deterministic
per-pair low-rank QP correction through the frozen path
`RGB -> uint8 -> factor-2 resize -> SegNet trunk -> 144D penultimate patch ->
rank-4 head margin`.  The correction is admitted only when the actual hard
oracle realizes positive declared-write margins after receiver round-trip.

This is a realization/secant/QP/measurement lane.  It does not own the sibling
generator race or archive composer, and it may not turn pair-0 evidence into an
n600 claim.

## Owned implementation boundary

- Add a reusable pure module under `src/tac/optimization/` for typed secant
  observations, per-class/per-margin-bucket trust regions, deterministic
  minimal-norm inequality solving in a rank-at-most-4 correction chart, uint8
  box constraints, and strict receipt validation.
- Extend `tools/measure_realization_g2_lattice.py` only where needed to expose a
  `--realized-secant-custody` runner that reuses the existing receiver, hard
  oracle, openpilot/contextual base construction, #557 codec, and atomic
  checkpoint helpers.  Do not fork those implementations.
- Extend focused tests under `src/tac/tests/` and/or `tools/tests/` for the new
  module and runner helpers.
- Do not edit upstream, scorer weights, frontier pointers, live run dirs,
  sibling generator/composer files, shared trainer/config surfaces, or another
  worktree.  Do not commit; the supervising Codex owns review and serializer
  custody.

## D1 — realized-secant custody

For every measured pair and every admitted correction column:

1. Capture the candidate-state frozen SegNet logits and the exact input to
   `segmentation_head[0]` (the 16-channel penultimate feature map).  Gather the
   3x3/144D patch and winner/target pair normal at each declared write.  The
   source-arrangement VJP may be used only as a provenance cross-check; fresh
   candidate-state response is required.
2. Apply deterministic signed finite correction amplitudes through the real
   receiver.  Record applied scorer-plane RGB L2/Linf, uint8 saturation count,
   144D feature displacement, predicted first-order margin delta, realized
   margin delta, and the secant ratio
   `Delta realized margin / Delta applied correction`.
3. Aggregate trust regions separately by target class and declared pre-step
   margin bucket.  A region is usable only when every row is finite, has the
   expected sign, and the relative first-order/secant residual is within a
   declared tolerance.  Never pool across class/bucket to hide a failed row.
4. Preserve one independent row per pair per column.  For n600, a global
   column therefore has exactly 600 finite-or-explicitly-refused observations,
   never a single averaged pseudo-measurement.

## D2 — receiver-closed per-pair solve

- Solve `min 0.5 ||alpha||_2^2` in the measured rank-at-most-4 chart subject
  to declared target-vs-current margin inequalities, the trusted secant model,
  and uint8 RGB box limits.  Use a deterministic active-set/projection method
  with explicit feasibility and KKT residuals; no opaque random optimizer.
- Pull the chart back through candidate-state scorer response and the exact
  factor-2 receiver.  Apply only the rounded receiver correction, then rerun
  the hard oracle.  Positive realized margin is the authority, not QP status.
- Decode the correction packet twice and require bit-identical frames.  If a
  residual is shipped, serialize it through the existing #557 codec, parse it
  back, re-encode byte-identically, and count every added byte.
- Infeasible, saturation-limited, trust-region-refused, and negative-realized
  writes remain explicit rows.  They are not silently dropped.

## D3 — semantic/rate ladder

- Run/resume cumulative n16, n64, then n600 using atomic per-pair stages and
  preserved per-chunk/prefix checkpoints under
  `/Volumes/VertigoDataTier/pact/evidence/g2e_secant_20260721/`.
- At every prefix report whole-description exact pairs and declared-write
  survival separately, decomposed by class, stratum, and pre-step margin
  bucket.  Invoke `predict_project_realization_admissibility_v1` unmodified.
- Report base bytes, correction bytes, total bytes, headroom against the
  216,222-byte target box, and marginal score-units per correction byte.  Stop
  allocation at the registered `25/37,545,489` rate break-even.  A zero-byte
  correction is admissible only if no video-derived payload is carried.

## D4 — Pose scope

Measure frozen PoseNet on the exact corrected frame pairs and report realized
`d_pose`, declared-tube debt, and tube-contained pair count.  Decompose any
change from the semantic correction.  Do not force a pose improvement and do
not call the existing nearest-target cross-pair proxy exact.  If the semantic
arm lacks the advected-motion/xi base needed for a valid Pose conclusion,
record that precise blocker and keep the pose-factorized child open.

## Determinism, storage, and custody

- Single recorded seed; sorted stable iteration; numpy-fp32/CPU-Torch only;
  no MPS authority.  Same config and preserved inputs must yield identical
  packet bytes, frames, row hashes, and receipt hash.
- Memory-map/chunk the real cache and VJP/probe arrays.  Keep at most the
  current pair/column scorer tensors live.  Run storage preflight and write
  atomic stage/checkpoint files.  No durable evidence under `/tmp`.
- Source-close the seed, GT cache, scorer weights/source, rank-4 receipt,
  candidate base receipt, receiver/lattice code, #557 codec, tool, and new
  module.  The receiver must invoke no scorer.

## Acceptance

- Focused tests, Ruff, format check, `py_compile`, JSON parse, and `git diff
  --check` pass.
- Synthetic tests prove: signed secant ratios; class/bucket isolation; active
  inequality and uint8 bound handling; infeasible refusal; deterministic
  double solve/decode; KKT residual checks; malformed receipt rejection.
- At least an n16 real smoke produces preserved D1-D4 rows before any n64/n600
  continuation.  A full n600 run is required for an n600 verdict; otherwise
  the durable verdict is explicitly prefix-scoped and the family remains open.
- No score/promotion/GO claim, no pointer move, and MAIN review is required.

## No-touch list

`upstream/**`; `submissions/**`; frontier pointer/state; live run directories;
provider/dispatch surfaces; sibling predictor generator and archive-composer
owned files; `CLAUDE.md`; `AGENTS.md`; any branch or worktree other than this
isolated lane.

## STORES CONSULTED

Delegated authority SHA
`d90b237d9807ced72ecfb792fb7345b4a590ec903095b721cbcca4207b503a95`;
`CLAUDE.md`; `AGENTS.md`; program/craft/vehicle manuals; v7.5/v8 SPECs; current
operator Fisher/margin, corrected first-order+secant+QP, curvelet/shearlet,
xi-factorization, and reverse-waterfill directives; predecessor G2d code,
tests, receipts, findings, DAG and reuse manifest; rank-4 prototype receipt;
terminal n600 VJP/M1 manifest custody; both delegated inboxes.
