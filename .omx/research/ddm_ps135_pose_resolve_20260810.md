# ddm_ps135 — LC2 PoseNet re-solve checkpoint

**Checkpoint date:** 2026-08-10  
**Status:** apparatus hardening and scorer-free closure in progress; no new n600
score has been measured yet  
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
retained SD1 cumulative q3 rung order can be composed into the LC2 CX2/Brotli
container and each rung must then receive a full receiver-realized int12 pose
compensation.  A separate sequential Stage-C driver is being built against
`.omx/research/ddm_ps135_stage_c_implementation_spec.md`; no uncompensated
mixed row will be reported as a candidate.

Leg A remains the first scorer action: start from LC2's own decoded PR130
carrier state, run full-n600 signed-int12 finite-difference/GN passes with
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
rows.  A fresh ps135 claim will be written only after the final runner review,
serializer commit, storage preflight, and host-equivalent process check.

Candidate archives/coefficients stay under
`/Volumes/VertigoDataTier/pact/ddm_ps135_20260810/`; bulk target caches, raw
decode, and evaluator outputs route to
`/Volumes/APDataStore/pact/ddm_ps135_20260810/`.  Dynamic per-pass storage
checks fail closed as Vertigo headroom falls.  One fleet scorer lock and the
dispatch claim guard the full-n600 slot.

Measured so far: exact custody hashes/bytes, complete source behavior, LC2
base archive reproduction, LC2 frame-0 carrier parity, and scorer-free
container/runtime properties.  Not measured yet: a new Leg-A coefficient
state, any new d_pose/d_seg/S, any compensated mixed-precision rung, any
contest-CUDA row, or any PR135 global candidate.

## Checkpoint disposition

- Leg A full-n600 re-solve: **FIRED only after final code review/commit and the
  sole lane claim**; owner `ddm_ps135_pose_resolve`; consumer store
  `/Volumes/VertigoDataTier/pact/ddm_ps135_20260810/`; fire trigger all runner,
  storage, process, and claim gates pass.
- Leg B global PR135 joint solve: **QUEUED-WITH-A-FIRE-ORDER**; owner
  `MAIN/#995 current-base joint solver`; consumer store
  `/Volumes/VertigoDataTier/pact/pr135_joint_solve_20260810/`; fire trigger exact
  PR135 master frames or a proven CPU-equivalent receiver are retained, then
  run global int12 x basis x FiLM starts rather than singleton clicks.
- Stage C adaptive mixed precision x compensation: **implementation in
  progress, then serially fired after Leg A**; owner `ddm_ps135 Stage-C
  consumer`; consumer store
  `/Volumes/VertigoDataTier/pact/ddm_ps135_20260810/stage_c/`; fire trigger Leg A
  convergence plus the scorer lane re-claim, with q4 identity and four LC2
  mixed archives already byte-closed in scorer-free preflight.
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
- Treating local CPU rows as contest authority is closed; only exact
  contest-CPU/CUDA replay on the exact archive can move the pointer.

This checkpoint has not moved the exact pointer or reached sub-0.15.  The
current own-vehicle frontier remains **LC2 `S = 0.16959899569230852 @ 187,226 B`
`[contest-CUDA T4, adjudicated, n600]`**.
