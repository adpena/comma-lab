# Codex Findings: DDM J9 Geometry-Escape Cure And Refire

Date: 2026-07-25
Author: Codex
Lane: `lane_ddm_j9_geometry_escape_cure_and_refire_20260725`
Scope: #366 D1-D3 repair only
Research-only: `true`
Evidence axis: `[macOS-CPU frozen-scorer advisory]`
Score claim: `false`
Pointer: `0.1910828242 [contest-CPU]` — unmoved
Landing: MAIN review required; FIRE stays with MAIN

## Verdict

`READY_TO_FIRE_UNDER_STANDING_GO`

The bounded governed smoke traversed a forced G1 Movable geometry escape,
recorded the RG1 cure, rejected the cured-but-score-worse proposal without
terminating, admitted a later proposal with strict joint `delta S < 0`, and
wrote a complete checkpoint after that accepted step. The bounded process then
exited non-promoting with
`BLOCKED_POSE_FINISH_CONDITIONING_HISTORY_INSUFFICIENT`, as expected after only
one accepted step. That bounded-history verdict is not a geometry, custody,
descent-family, or campaign blocker.

No 450-step campaign was launched. MAIN must review the branch diff, ticket,
fresh memory receipt, and smoke receipts before landing or firing.

## D1 — Proposal-Local Geometry Cure

The realized-proposal path now applies the already-landed RG1
`project_polygon_center` cure on the exact `(512,384)` scorer plane before
proxy or exact scoring. Projection failure raises the typed
`ProposalGeometryInfeasibleError`; only that type is handled as
shrink/reject/next-source. Generic archive construction, SHA, and parse-back
custody failures remain process-fatal.

The forced smoke requested track-0 translation `(8192,0)`. RG1 projected it to
`(234,0)` and emitted:

- event: `proposal_infeasible_geometry`
- status: `cured`
- verdict scope: `INSTANCE proposal geometry only`
- event SHA-256:
  `0c4fd549a7770026366cf509f29c35f850e3a8f675c125e4157a51d2e32cd15d`

The cured `worldsheet_joint_active_x_+1` proposal was then scored through the
full n600 path and rejected with joint delta
`+0.015133196578369557`, proving cure-not-crash followed by normal admission
logic. The next `x_-1` proposal was also rejected at
`+0.16061675207795353`; neither rejection terminated the process.

## D2 — Full Resume Checkpoint After Every Accepted Step

Every accepted step now atomically writes a distinct keep-all checkpoint named
`<stage>_accepted_globalNNNNNN.npz` after its complete cursor and exact-verdict
state is known. The existing every-37 and stage/intra cadence remains additive.

The bounded smoke admitted `worldsheet_joint_active_y_-1` at step 1 with:

- joint delta: `-0.05689051019463004`
- Seg term: `-0.021033393012152846`
- Pose term: `-0.03585911475933656`
- Rate term: `+0.000001997576859366514`
- proposal verdict SHA-256:
  `6ed3d5466fb9a85a7dc7115eaddf42eeae8979eddb7b0f7a5858e207fc42b2a6`

The immediate accepted-step checkpoint is:

`/Volumes/VertigoDataTier/pact/experiments/results/ddm_j9_366_forced_geometry_resmoke_20260725T0534Z/checkpoints/01_residual_bucket_realized_acceptance_accepted_global000001.npz`

Its SHA-256 is
`51722d320b9ce7bbfab389e310ab2cade80ef9dc53765278fdfcfa9dd07a0f4d`.
Fresh parse-back recovered step/cursor `1`, finite theta/EMA/first/second
moments, and the realized archive
`4487754bf1517946eb7b604817f99c5623ec0320aad3287edc67b436bae793f5`
at 138,804 bytes. All four state arrays and the step counter were bit-identical
to the additive intra-step checkpoint.

## D3 — Measured Schedule

The J9 reseal uses only attempt-4 measured evidence:

- full n600 proposal seconds: low `410.85283304192126`, central
  `436.65467699989676`, high `452.71083845803514`
- main-loop step seconds: `272.22402341710404`, `273.1033891250845`,
  `272.74639812507667`, `272.4739739999641`, `329.62735133292153`,
  `329.45078045804985`
- typed central main-loop cadence: `312.0` seconds = 5.2 minutes
- 450-step wall clock: low `34.370380288006274` hours, central
  `39.363878897499916` hours, high `41.580677948663556` hours

The central derivation is
`(3 * 436.65467699989676 + 450 * 312.0) / 3600`.
The ticket explicitly states the run exceeds 24 hours. The sealed verdict
interval remains 50 steps and checkpoint interval remains 37 steps; no cadence
semantics were invented.

## Replay, Reseal, And Fresh Memory

Attempt-4 telemetry rows 1-7 were inspected for deterministic replay. They
preserve seed, proposal source, and archive hashes, but not theta, EMA, first or
second moments, run cursor, or candidate archive bytes. No byte comparison was
fabricated. The typed decision is:

`RESTART_FROM_W_JOINT_INSUFFICIENT_SEED_CUSTODY`

The resealed ticket is
`.omx/research/configs/ddm_j9_366_geometry_escape_cure_20260725.json`,
SHA-256
`b4446573eeff9d3beacedb970275fccef0cb574fdf4c28f18a357d453ecd2370`.
It preserves J7 authority SHA-256
`8dac31beda848b94b8bd42f43ffd7008cd024fcf916c0a14149307f68085907e`,
binds source commit `3e8ab50020f990dbb069640b7b7b287cc5946ffc`,
and seals semantic-program SHA-256
`96ca852b61168cf86a6e6d9166a27aa73d955a00b5d06ed940210d79f92f34d7`.

Because the consumer changed, the old geometry receipt was not reused. The
fresh worst-geometry receipt is:

`/Volumes/VertigoDataTier/pact/experiments/results/ddm_j9_366_fresh_memory_bootstrap_20260725T0505Z/worst_geometry_memory_preflight.json`

Receipt SHA-256:
`d957c817a997db3a77b50b20950e9c6d0d5cfc36435bed24fd4ab0e8c4e6017c`.
It measured the exact 52-secant all-groups geometry in 1,596.003 seconds at
16.902 GiB peak RSS, projected 21.282 GiB, below the 116 GiB ceiling. Fused-R
forward and gradient were bit-identical to the NumPy-fp32 oracle, and its
memory-only checkpoint parsed back bit-exactly.

The bounded smoke receipt is:

`/Volumes/VertigoDataTier/pact/experiments/results/ddm_j9_366_forced_geometry_resmoke_20260725T0534Z/full_run_receipt.json`

Receipt SHA-256:
`c364466c2cc1d643993fe0eb856c81993f3482b62202dde083f965a320588a01`.
It is explicitly research-only, non-promoting, `score_claim=false`, and
`pointer_moved=false`.

## Verification

- Focused three-file suite: `49 passed in 106.06s`
- Fresh adversarial subset: `8 passed, 41 deselected in 48.50s`
- Ruff across all six touched Python files: clean
- `git diff --check`: clean
- Two clean review-tracker passes per touched `.py`
- Synthetic 8192-pixel escape fixture takes the RG1 cure path
- Kill/resume fixture restores theta, EMA, both Adam moments, step, and cursor
  bit-faithfully

## Triality And Stores Consulted

- DSL: the typed J9 semantic program seals projection policy, every-accepted
  checkpoint policy, measured schedule, and replay/restart decision.
- DAG: attempt-4 custody -> D1-D3 repair -> reseal -> fresh worst-geometry
  bootstrap -> bounded forced-cure smoke -> MAIN review -> MAIN-owned FIRE.
- Equations: scorer-plane feasibility uses RG1 integer center projection;
  admission remains the exact priced joint action; runtime is derived from the
  measured proposal and main-step terms above.
- Stores consulted: #366 attempt-4 run/log/telemetry/verdicts, J7 authority,
  sealed W_joint ticket and archive custody, RG1 implementation/tests,
  canonical research/equation/DAG/task/document surfaces, lane registry, and
  live delegation inbox.

## MAIN Landing Review

MAIN should review:

1. Only `ProposalGeometryInfeasibleError` is proposal-local; SHA and parse-back
   failures remain fatal.
2. Projection uses RG1 unchanged and emits immutable `cured|rejected` events.
3. The accepted checkpoint is written after complete cursor/verdict mutation
   on every admission, with every-37 and stage checkpoints preserved.
4. The ticket source SHAs, J7 authority SHA, semantic SHA, memory SHA, replay
   decision, and >24-hour schedule all match the receipts above.
5. FIRE remains a separate MAIN action after merge review and normal dispatch
   claiming; this branch did not launch the campaign.
