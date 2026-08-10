# ddm_ps135 — LC2 PoseNet re-solve checkpoint

**Checkpoint date:** 2026-08-10  
**Status:** fail-closed apparatus landed and scorer-free closure complete; Leg A
is blocked by candidate-store headroom, Stage C is blocked by exact-runtime
parity plus an unfrozen scorer seam, and no new n600 score has been measured

**Authority boundary:** every future local scorer result in this arm is
`[macOS-CPU advisory]`; MAIN alone may fire one exact contest-CUDA row

## Outcome first

The load-bearing mechanism is available in retained source, but the charter's
history needed two corrections before execution.

First, PR135/F26 did not stop while accepting singleton moves.  Its exact
all-12 signed-int12 `+/-1` pass history is `412, 187, 72, 39, 15, 9, 2, 0`
accepted rows, and its runner declares the last pass converged for that tested
one-step Pose-MSE neighborhood.  Repeating the same pass on PR135 is therefore
a duplicate, not an uncap.  Leg B remains open only as a broader global joint
int12 x basis x FiLM search, but is locally blocked because the shipped F26
runtime refuses CPU and no exact PR135 raw/master-frame bank is retained.

Second, LC2's shipped `inflate.py` already supports counted `SD1M` mixed
semantic precision.  Stage C is not blocked by a missing selector.  The
retained SD1 cumulative q3 rung order was composed into the LC2 CX2/Brotli
container and reproduced as five byte-exact, parse-backed, scorer-free
archives.  The landed driver refuses master-bank creation until an exact q4
parity receipt exists and the sequential scorer seam is explicitly frozen.
That scorer seam is not implemented, so no uncompensated mixed row is reported
as a candidate.

Leg A remains the first scorer action once storage admits it: start from LC2's
own decoded PR130 carrier state, run full-n600 signed-int12 finite-difference/GN passes with
exact lattice and complete-archive acceptance, require at least eight passes
and three dry passes, then run a distinct exact JRD-style terminal ladder.
Every coefficient state and every materialized archive is retained.  The
public PR133 complete carrier is a separately attributed composition control;
its coefficient values are never copied into LC2 without their basis/scales.

## Custody pins

| Object | Bytes | SHA-256 | Use |
|---|---:|---|---|
| LC2 archive | 187,226 | `f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45` | Leg-A base and hard byte ceiling |
| LC2 retained raw | 3,662,409,600 | `a18eb42a8da9399bcc03e795e17597bfbd459412dbb37990117665f48c4c0353` | exact PR130/LC2 master frames |
| LC2 raw carrier | 23,054 | `a05d0985ca5a8d5110bd5bf5be39f238c6f89640b8a8bb888a3e1269bdf636e4` | native warm start |
| public PR133 archive | 190,212 | `051baf408f57fae3b343d6ee218ab963d070b3935ceb0b2f412c93a53cf3fab0` | complete-carrier control only |
| PR135 archive | 186,724 | `12cf5d71a94065184f097c3e40dfe9f1db8402a1a76a80efc76a6956fe1e4004` | borrowed/current-base Leg-B input |
| ExperimentBook joint primitives | retained source | `5d7424f1e523105766ac1f45d7c9219899534394dae348cd7f29b3304fd4f286` | damped relative-ridge GN and int12 lattice helpers |

The LC2 candidate builder has already reproduced the exact base archive bytes
and SHA in a scorer-free smoke, and the standalone carrier renderer reproduced
retained LC2 slave frame 0 byte-for-byte.  These are apparatus checks, not a
new score.

## RECALL EVIDENCE

### Sources and literal searches

The recall covered the full `.omx/research/` corpus and arm receipts, all
`CANONICAL_RESEARCH_INDEX*` files, `sub015_DAG_*` FEED blocks, design/SPEC
files, queue/hot-state surfaces, canonical task status, the Aug-03 harness
bridge, canonical equations, PR135's complete retained ExperimentBook, PR135
source custody, PR133 custody, LC2 runtime/container custody, and retained
PR130/JRD CPU search code.

Content queries included:

```text
pose re-solve|pose_resolve|int12|joint_pose_solve|PR130|PR133|PR135
ddm_lc2|ddm_pi136|ddm_fd135|#740|#850|#935|#974
#453|#460|JRD|last-safe-plane|safe-plane|harvested priors|step sizes
SD1M|semantic_allocation|mixed precision|selected_mixed_n600
```

The canonical equation registry was enumerated with:

```text
.venv/bin/python tools/list_canonical_equations.py --json
```

and filtered for pose, quantization, lattice, GN, rate, convergence, and
termination.  Exact archive SHAs and byte counts were searched across current
receipts rather than inferred from names.

### Findings beyond the charter seeds and resulting plan changes

1. The actual reusable solver machinery exists in
   `experiment_book/src/cpr1_sub4/joint_pose_solve.py` and the retained F14,
   F17, F18, F25, and F26 scripts.  The plan changed from reconstructing GN/CG
   from prose to adapting those exact primitives and runners.
2. F26 ends in a zero-accept singleton pass.  Leg B changed from “more identical
   passes” to a broader realized multistart/joint search, with the identical
   pass recorded as closed.
3. F26's zero pass is narrow: it optimizes independent row-local Pose MSE, not
   global S, coordinated coder interactions, radius>1 moves, basis/scales, or
   FiLM.  This preserves a global/rate-aware hypothesis without falsifying the
   measured singleton result.
4. `ddm_fd135_fractal_decomposition_20260810.md` shows PR130->PR135 changed
   7,044/7,200 int12 symbols, every row and scale, atoms 2/5/9, and two FiLM
   codes.  It also shows F26 moved segmentation slightly.  The plan therefore
   keeps joint basis/FiLM work for the current-base successor and does not call
   PR135 a pose-only local continuation.
5. Existing F17/F14 runners discard losing charged archives and overwrite
   state non-atomically.  They cannot be launched under P0 retention.  The new
   runner uses immutable per-candidate/per-pass payload paths, SHA/byte records,
   atomic resume state, and a hard LC2 byte ceiling.
6. LC2's raw is exactly the PR130 lineage and its retained native decoded
   codes/scales are valid.  F17 itself refuses direct code transfer when scales
   differ.  This changed the warm start to LC2-native state plus a separate
   complete-PR133-carrier control.
7. The retained PR130 CPU analogue searches exact steps
   `1,2,4,8,16,32`, but #460's bias/weight plane ranges are only dormant n=1
   ordering data.  They cannot be relabelled as LC2 int12 step measurements.
   The terminal exact ladder is retained as an attributed CPU/JRD-style
   finisher; a separate receipt must record that #460 precision actuation stays
   refused without n600 confirmation.
8. LC2's shipped runtime contains the `SD1M` counted mixed-allocation parser,
   and SD1 already retained cumulative q3 semantic rungs.  This invalidated the
   provisional “no LC2 selector” Stage-C blocker and created a concrete
   compensation driver rather than a queued proxy map.
9. A batch-shape seam was measured elsewhere: CPU batch 1 and 16 can disagree.
   The apparatus pins threads and batch 16, keeps candidate screening geometry
   fixed, and uses the exact global `37x16 + final8` population refresh for
   acceptance and final scoring.
10. The current `CANONICAL_RESEARCH_INDEX*` files have no direct PR133/PR135,
    joint-pose, LC2, pi136, or fd135 entry.  That is a bounded freshness gap;
    current memos and primary retained source govern this intake.
11. Task #850 was found as a completed stop-taxonomy lesson.  In the scoped
    canonical task and bridge stores, #935 and #974 were not found as rows;
    #974 maps by content to the true receiver-realized lattice doctrine.  #740
    maps to a different MLX SegNet description-space GN engine and is reusable
    only for matrix-free/trust-region ideas, not as the pose runner.

## Borrowed-substrate accounting

- Mechanism: public PR133/codexblack exact PoseNet-guided int12 re-solve.
- Proposal math and closest retained runners: PR135 ExperimentBook.
- CPU traversal ancestry: retained PR130 Fesal-Fayed lineage; terminal JRD-style
  exact lattice ladder is separately attributed.
- Base/receiver/container: our LC2 artifact, itself PR130-lineage.
- Implementation contributed here: LC2 mutation/parse-back, exact CPU scorer
  geometry, GN/rate-aware orchestration, atomic resume, per-candidate retention,
  storage and fleet locks, global refresh, convergence, and exact final replay.
- Mixed-precision ranking/apparatus: retained SD1 and #869/wr1 mechanism only;
  no old byte credit is transferred.

No PR133, PR135, JRD, or SD1 learned value is called ours-original.  Any result
remains a borrowed-vehicle successor, not the original task-space witness.

## Execution gates and current boundaries

The stale SD2 scorer claim was reconciled with a terminal failure row after
its retained PID was absent and its safe-run log showed receiver-preflight exit
1 before any scorer chunk.  The live claim summary then contained zero active
rows.  The final runner landed in commits `d6ea363904` and `452ce890b5`.
Committed-source preflight found zero active claims and no suspicious Python
open files, but the candidate-store reserve failed, so no ps135 claim was
written.

Candidate archives/coefficients stay under
`/Volumes/VertigoDataTier/pact/ddm_ps135_20260810/`; bulk target caches, raw
decode, and evaluator outputs route to
`/Volumes/APDataStore/pact/ddm_ps135_20260810/`.  Dynamic per-pass storage
checks fail closed as Vertigo headroom falls.  One fleet scorer lock and the
dispatch claim guard the full-n600 slot.

Measured so far: exact custody hashes/bytes, complete source behavior, LC2
base archive reproduction, LC2 frame-0 carrier parity, five scorer-free Stage-C
archives, and the live storage/liveness gates.  Not measured: a new Leg-A
coefficient state, exact q4 semantic-master parity, any new d_pose/d_seg/S, any
compensated mixed-precision rung, any contest-CUDA row, or any PR135 global
candidate.

## Executed receipts and verification

### Landed apparatus

- Commit `d6ea363904` landed the fail-closed Leg-A solver, Stage-C preflight,
  and focused tests.
- Commit `452ce890b5` made the sandboxed process-table preflight reach its
  bounded `lsof` fallback instead of crashing on `PermissionError`.
- Two review-tracker passes were recorded with distinct `council` and `codex`
  approvers.  `py_compile` and Ruff passed on all four files.
- Governed non-parity tests: **54 passed, one exact-runtime parity test
  deliberately deselected**, exit 0 in 43.66 s.  Safe-run receipt:
  `845ec9ee93ddb858eca50670c9e9ee941f05ea30e6a4e8245d0f2b3e110cceea`;
  JUnit receipt:
  `a96026e12298d25b62d4d1bfc0e9a26a9dc924345f09f98211d0716acdc4c772`.
- The full developer preflight was also run.  Seventeen of twenty-five gates
  passed; eight repository-wide gates were already red (state-writer strict
  load, custody-tag validation, codebase drift, dispatch helper, old landing
  wire-ins, lane registration, scorer-contract census, and pose-default
  census).  No exemption was widened and this result is not called green.

### Leg-A scorer-free admission

`/Volumes/VertigoDataTier/pact/ddm_ps135_20260810/preflight.json` is complete,
scorer-free, 4,869 B, SHA
`6fa86be063797deaf82a6272854cd56ebae8f72f197259e4233d1a0c8e405095`.
It measured:

- active dispatch claims: `0`;
- canonical `/bin/ps`: sandbox-blocked with `PermissionError`;
- bounded Python `lsof` fallback: return 0, four Python PIDs, no suspicious
  scorer open files;
- AP bulk store: `1,042,733,596,672` B free versus `8,000,000,000` required;
- Vertigo candidate store: `1,241,710,592` B free versus
  `3,000,000,000` required.

Therefore `admission_ready_except_ps135_claim=false` solely on storage.  The
safe-run receipt is 1,065 B, SHA
`fe007b6eda7143f326ca1b369b9c1d29b33bb14e6bf8eb0baa5572a75c144542`.
No lane was claimed and no scorer was loaded.

### Stage-C scorer-free archive closure

The committed-source preflight at
`/Volumes/VertigoDataTier/pact/ddm_ps135_20260810/stage_c/freeze_452ce890b5/`
completed in 57.83 s.  `PREFLIGHT.json` is 6,406 B, SHA
`fabfa7cf7178147f63c91f585f21ec813b6bd68d18256724ce8790bb4fbbe100`;
the safe-run receipt is 1,397 B, SHA
`19f7747021a86167e23d7cc00375be2fdb1af6f970d5ad313aa35290b9ef8edb`.
Every archive, repeat, wire section, and receipt is retained.

| Rung | Archive bytes | SHA-256 | Scorer status |
|---|---:|---|---|
| q4 LC2 identity | 187,226 | `f154f0abb76980a30715282cf330d611cac7ebce3379c5f8093830dc273e1a45` | not run |
| blocks.3 FiLM q3 | 187,141 | `8def4a214959ac5c42f02a1a04c3d7dcd789d62e7e7acb918daba759a440f548` | not run |
| + blocks.2 FiLM q3 | 186,889 | `059e32fba2b9dea290085dbbd9e57c00a5b66932b9e8c1ad128aaa7a2380a482` | not run |
| + frame embedding q3 | 186,595 | `3543232b40569c19f7fc9a21fcb4f32f479d8e125d3be1be624bf342f59784db` | not run |
| + blocks.1 FiLM q3 | 186,393 | `e89063130bb5f25dee18d9a1079cdd76bae2e5a6bad72a0f358a53be46df58db` | not run |

These are real byte savings, not score improvements.  Compensation and exact
n600 scoring have not run.

### Exact parity and future-launch blockers

The only completed non-authority control used Python 3.11/Torch 2.10/thread 1
and differed from retained q4 frame 1199 at 12 bytes, each by one level.  It is
not parity authority.  The pinned Python 3.13.12/NumPy 1.26.4/Torch 2.12.1
attempt produced no frame before a 126.76 s import timeout; typed blocker SHA
`78b1c98e3155e18c27046c168cba8b6212d00a095d1eb69d64052ea393aa1032`.
A separately governed isolated attempt also timed out at 120.46 s with receipt
SHA `73d9e6a93840bc894572c8b2f93bb6eff4a36cc85fd48fe2f2d4c334ef13e62f`.

Master-bank and scorer paths are therefore hard-blocked.  Before changing the
scorer-seam constant to `FROZEN`, the v3 master validator must also make both
stream checkpoints mandatory and cross-bind their canonical path, candidate,
attempt kind, range, render/driver identity, and payload record.  At the landed
hash this future-launch P0 is contained behind `QUEUED_NOT_FROZEN`; there is no
runnable bank bypass.

## Checkpoint disposition

- Leg A full-n600 re-solve: **BLOCKED-STORAGE**; owner
  `ddm_ps135_pose_resolve`; consumer store
  `/Volumes/VertigoDataTier/pact/ddm_ps135_20260810/`; fire trigger Vertigo has
  at least `3,000,000,000` free bytes, then rerun committed-source preflight,
  obtain an empty process/claim gate, and claim the sole lane.
- Leg B global PR135 joint solve: **QUEUED-WITH-A-FIRE-ORDER**; owner
  `MAIN/#995 current-base joint solver`; consumer store
  `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/`; fire trigger exact
  PR135 master frames or a proven CPU-equivalent receiver are retained, then
  run global int12 x basis x FiLM starts rather than singleton clicks.
- Stage C adaptive mixed precision x compensation: **SCORER-FREE PREFLIGHT
  COMPLETE; SCORER/BANK BLOCKED**; owner `ddm_ps135 Stage-C consumer`;
  consumer store `/Volumes/VertigoDataTier/pact/ddm_ps135_20260810/stage_c/`;
  fire trigger completed Leg A bindings, implemented sequential rung scorer,
  mandatory master-checkpoint cross-binding, frozen driver, passing exact q4
  parity, then the scorer lane re-claim.
- One exact Modal row: **QUEUED-WITH-A-FIRE-ORDER**; owner `MAIN` only;
  consumer store MAIN's exact-eval receipt store; fire trigger the final local
  candidate is byte-closed, <=187,226 B, and strictly improves full advisory S.

## LIVE-HYPOTHESES

- LC2's never-re-solved native coefficients can recover most of the PR133 pose
  gain because the base raw/renderer lineage is exactly PR130 and the unknown
  is search/container coupling, not vehicle legibility.
- Radius>1 GN-centered and wrong/global starts can escape F26's narrow zero
  singleton neighborhood because the full PR130->PR135 displacement was much
  broader than the final local F24->F26 polish.
- Coordinated rate-aware row moves can improve S even when independent Pose MSE
  does not because CAP1's delta/Rice coding couples adjacent stored symbols.
- The four cumulative SD1 q3 rungs can remain favorable after pose
  compensation because they already save real bytes and have small measured
  segmentation cost, while PoseNet drift is the open quantity compensation
  directly targets.

## DEAD-ENDS

- More identical PR135/F26 `+/-1` singleton passes are closed by the measured
  zero-accept eighth pass for that exact neighborhood.
- Copying PR133 integer codes alone into LC2 is closed because basis/scales are
  part of the realized object and may differ.
- Using #740 directly as the CPR1 pose solver is closed because it targets a
  different SegNet lifted-description vehicle.
- Treating #460's n=1 bias/weight plane ranges as measured LC2 precision is
  closed by its own dormant activation gate.
- Treating uncompensated uniform or mixed precision as the Stage-C candidate is
  closed because the retained controls show compensation is the mechanism.
- Treating the four smaller SD1M archives as score improvements is closed:
  they are scorer-free rate artifacts until sequential compensation and exact
  n600 evaluation complete.
- Materializing Stage-C master banks at the landed hash is closed: exact q4
  parity is blocked, the scorer seam is unfrozen, and later driver edits would
  invalidate whole-driver-bound banks.
- Treating local CPU rows as contest authority is closed; only exact
  contest-CPU/CUDA replay on the exact archive can move the pointer.

This checkpoint has not moved the exact pointer or reached sub-0.15.  The
current own-vehicle frontier remains **LC2 `S = 0.16959899569230852 @ 187,226 B`
`[contest-CUDA T4, adjudicated, n600]`**.
