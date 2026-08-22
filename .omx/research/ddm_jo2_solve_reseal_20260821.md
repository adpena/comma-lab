# DDM JO2 solve and reseal receipt — 2026-08-21

## Outcome

`ddm_jo2_solve_reseal` is **BLOCKED**, not
`READY_TO_FIRE_UNDER_STANDING_GO`.  Seal r5 is honest and names four remaining
blockers:

1. `JO2_REMOTE_TRAINER_ENTRYPOINT_NOT_IMPLEMENTED`
2. `RC2_BASE_ARGMAX_FIELD_MISSING`
3. `FX5_BASE_POSE6_MISSING`
4. `MEMORY_PREFLIGHT_BLOCKED:memory preflight receipt is absent`

The old receiver-close blocker, the missing DALI source-Pose6 target, and the
AP storage blocker were resolved.  No heavy job, scorer job, memory-preflight
job, training job, or exact evaluation was launched by this arm.  The exact
frontier pointer did not move.

Seal r5 lives at
`.omx/research/ddm_jo2_solve_reseal_20260821/seal_r5/`.  Its compiled file SHA-256
is `35487b801d35916de3f9ad252fbae81801d71993815a3914ea2fe3da1b29620f`;
its workload identity is
`1d43eee38c1ee9858c67463840644a7abea1e3708a28118e97ad947ffc38ba9b`.
`READINESS.json` SHA-256 is
`f25c2f4a9cf6ca58bbe674eada53b5f17567b9d5be5deea114ef535a7d0dd75e`.

## What was implemented

`experiments/ddm_jo2_residual_runtime.py` defines the counted JO2 residual
state and the generic shipped receiver.  The learned state is float16,
length-delimited, SHA-bound, strict about trailing bytes, and stored inside the
existing semantic body.  The runtime applies the residual after the final
semantic TokenBlock and before the renderer round trip.

`experiments/ddm_jo2_receiver_close.py` implements the real fx5 receiver-close
path:

- decodes the actual 600 by 12 signed-int12 frame-0 carrier;
- evaluates the official PoseNet first six outputs on realized camera pairs;
- solves a fresh central-difference Schur/Gauss-Newton step per pair, searches
  the quantized neighborhood, and finishes with exact coordinate descent;
- binds every resumed pair receipt to the candidate semantic body, candidate
  frame-1 field, fx5 baseline Pose6 table, and fx5 archive;
- rebuilds the carrier directly rather than stacking a sparse overlay;
- retains every PoseNet input/output and every real Brotli/ZIP race candidate;
- builds one stored ZIP member named `p`, repeats it deterministically, and
  verifies semantic and 600 by 12 carrier parse-back in a fresh process;
- materializes the intended n600 frame-0 field resumably and provides a final
  decoded frame-0/frame-1 identity verifier.

This is a real receiver and solve implementation, but it is not yet invoked by
the Modal training loop.  The existing memory and train entrypoints still
refuse instead of pretending that standalone primitives constitute training.
That is why the first blocker remains.

The JO1 configuration now distinguishes:

- `source_pose6_targets`: DALI source-video targets used by joint descent; and
- `fx5_base_pose6`: the exact fx5 body output that fresh carrier compensation
  preserves for a changed frame-1 candidate.

Substituting one for the other is forbidden.  The verified DALI target is
`/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/full_n600_eval/retained/pose_vectors/gt_first6_dali_n600.npy`,
14,528 bytes, SHA-256
`8d5cfa83df55b89493ba43b1e5386d792c836c32791666192499a089068e7eff`,
shape `(600, 6)`, dtype `float32`.

Local source triples are refreshed as one operation before resealing; SHA-256,
source-object SHA-256, and bytes cannot be partially updated.  The r5 source
pins are:

| Source | Bytes | SHA-256 |
|---|---:|---|
| `ddm_jo1_joint_objective_design.py` | 44,579 | `9f41216e3ed0df33586dba101bda5ca7f408c4dd4b6670bb0b220536fa6e9f1d` |
| `ddm_jo1_joint_objective_worker.py` | 16,232 | `8c906c8483a597784c75904fbd30cfc226dc26dadc31b13b6bc03aae7601615d` |
| `ddm_jo1_modal_joint_objective.py` | 32,503 | `6e440b12a0bb1595f65178e77dc5c99c23dbbf9b7f4ceee5afa28f6a2ec752fb` |
| `ddm_jo2_residual_runtime.py` | 8,625 | `455b1b2ddce6ad1e9f3c5134f6d3bdb5e6f9c4d7cad582e76acce18b1ab32dea` |
| `ddm_jo2_receiver_close.py` | 48,864 | `f391b71963f6cd69611edac10df44408a49aa824942c31c3305d7971386edf5a` |

## Receiver control evidence

A component-only control used the real fx5 archive and runtime.  It did not use
a toy renderer or proxy coder.  The durable receipt is
`.omx/research/ddm_jo2_solve_reseal_20260821/receiver_control_r2/CONTROL_RESULT.json`.

Measured on `[macOS-CPU receiver-control; no scorer authority]`:

- the zero-state JO2 wrapper exactly matched the real fx5 semantic renderer for
  pair 0 before R; maximum absolute difference was `0.0`;
- the JO2 semantic state parsed back byte-identically;
- all 600 by 12 carrier codes parsed back identically;
- the archive contained one stored member `p` and the deterministic repeat was
  byte-identical;
- every materialized payload was retained with bytes and SHA-256;
- the control archive was 181,131 bytes, SHA-256
  `2652a3e474dfd9cd4bad6ddb80c63c325af0f30fcbff02504b411752cf9afef4`.

The 181,131-byte control is not a candidate or an economics result.  It carries
a zero residual using a fixed q11 control encoding.  No SegNet/PoseNet result,
n600 decoded-render result, distortion component, score, or contest authority
was measured for it.

## Payload and storage custody

The charter's claimed local `payloads_r8/` custody could not be verified.  A
bounded search of the repository, APDataStore, and VertigoDataTier did not find
either expected file by name or exact byte size.  The materializer receipt
proves that the bytes were produced remotely, but a remote path in a JSON
receipt is not local custody.

The two missing objects remain on Modal volume
`comma-auth-eval-cache-artifacts` under
`ddm_jo1u_fx5_e1_n600_r4/` according to the COMPLETE materializer receipt:

- `retained/fields/fx5_e1_argmax_n600.npy`, 117,964,928 bytes, SHA-256
  `e89e1ac083e5964975a1b4121cd1bc8bd91236256d6922f66c650246d7783c34`;
- `retained/pose_vectors/fx5_e1_first6_n600.npy`, 14,528 bytes, SHA-256
  `71f7d2639eb624f4d0eb89e40ac5956a74b1f72951dc7f07424468769af8350f`.

A fresh Modal control-plane attempt failed with `Could not connect to the Modal
server.`  The receipt is
`.omx/research/ddm_jo2_solve_reseal_20260821/MODAL_CONNECTIVITY_BLOCKER.json`,
553 bytes, SHA-256
`fa7f8d26b077555b9549ebbaf176dab13cf2a0e591b91d71fe86fe7fa98f3591`.
No new scorer fire is warranted; r5 `FIRE_ORDER.json` contains exact targeted
downloads of the two already-materialized objects.

The 47,244,640,256-byte scratch requirement was re-rooted from constrained
APDataStore to local APFS at
`experiments/.scratch/ddm_jo2_joint_objective_solve`.  R5 measured
603,043,409,920 free bytes there.  Nothing was deleted or moved.

## RECALL EVIDENCE

The recall sweep covered the full `.omx/research/` corpus, arm final messages,
canonical research index/DAG surfaces, `.omx/state/main_hot_state.md`, the task
and lane ledgers, the actual fx5 runtime, and the canonical equation registry.
Content queries included `fresh Schur`, `same-object compensation`,
`candidate_object`, `joint solve`, `Pose6`, `fx5_e1`, `RC2`, `qs4`, `qs5`,
`JG1`, `BU1`, `DALI`, `carrier`, and the two expected payload SHA-256 values.
The canonical equation command was
`.venv/bin/python tools/list_canonical_equations.py --json`; the relevant row
was `score_marginal_lagrange_multipliers_v1`.

Findings beyond the charter's seeds changed the implementation:

- `.omx/research/ddm_bu1_bank_union_compile_20260817.md` records a fresh joint
  Schur solve beating naive addition by 3.705 times and explains that
  compensation objects are non-additive.  JO2 therefore binds and re-solves
  the cumulative candidate object rather than stacking correction payloads.
- `.omx/research/ddm_jg1_joint_solve_20260819.md` reports 98.7 to 100 percent
  recovery through the carrier's 12 int12 codes.  JO2 therefore solves and
  directly recompiles those shipped coordinates rather than introducing the
  linear overlay family already closed by PK4.
- The DALI source target and fx5 baseline Pose6 have different identities and
  roles.  This added a separate `fx5_base_pose6` binding and later caused the
  second review to strengthen every freshness fingerprint with its hash.
- The local-payload assertion in the live dispatch ledger was not byte-backed.
  This changed step 1 from “repoint paths” to “recover and verify the exact
  existing remote bytes.”
- The live board superseded the common contract's old frontier line with
  `fx5_e1` at 0.14823186109359 and 180,386 bytes.

No contrary current-vehicle evidence was found that makes stale cross-object
compensation, linear frame-0 overlays, or the WD4 warm-slice route admissible.

## Verification

- `ruff` passed on all six changed Python files.
- Focused tests: 23 passed; the two warnings are the existing Pydantic `schema`
  field-shadow warnings.
- `py_compile` passed on all six changed Python files.
- The bounded payload-retention scan examined all six changed Python files and
  found 0 measure-and-discard sites.
- Two genuine review passes were recorded for every changed Python file.  The
  second pass found and fixed the missing baseline-Pose6 freshness binding.
- `git diff --check` passed.
- The global developer preflight was not green: 8 of 25 declared gates were red
  in the shared dirty worktree.  A scoped inspection showed, for example, all
  five score-aware-scorer-contract findings were in unrelated
  `src/tac/substrates/**` files and all 14 lane-registration findings were in
  pre-existing unrelated paths.  The JO2-focused lint, tests, compile,
  retention scan, and reviews above are the bounded landing evidence; this memo
  does not claim a repository-wide preflight pass.

## Boundaries

Measured: real fx5 receiver parsing, zero-residual pair-0 forward identity,
full carrier-lattice parse-back identity, deterministic single-`p` control
archive identity, local storage availability, source/payload hashes, and the
r5 readiness blockers.

Not measured: a nonzero learned residual, fresh n600 Schur solve, n600 decoded
render identity, realized SegNet or PoseNet deltas, stage archive economics,
real T4 memory, wall-clock for the JO2 trainer, complete score, contest-CPU, or
contest-CUDA.  Because memory preflight did not run, no derived training
wall-clock is claimed.

Own-vehicle frontier: **fx5_e1 S 0.14823186109359 @ 180,386 B
[contest-CUDA T4 n600]**, archive
`4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841`;
**UNMOVED**.

## LIVE-HYPOTHESES

- Fresh joint carrier recompilation can preserve the fx5 Pose6 floor after a
  useful frame-1 residual, because JG1 recovered 98.7 to 100 percent of tested
  edit-induced damage through the same 12-code coordinate family.
- The post-TokenBlock residual can turn some of fx5's 23,757 remaining Seg
  errors without EC2's four-block collateral spread, because its actuation is
  later and directly conditioned on the retained token boundary field.
- Refitting the exact CAP1 predictor after fresh code changes may keep carrier
  cost small, because JO2 recompiles the native carrier rather than adding a
  separately framed overlay.
- The two missing base payloads are recoverable without a new scorer job when
  Modal connectivity returns, because the COMPLETE remote receipt gives exact
  volume paths, byte counts, and SHA-256 values.

## DEAD-ENDS

- Carrying QS4 compensation across candidate objects: closed because it caused
  the measured refusal and JO2 now makes it unrepresentable with four-way
  object binding.
- Treating a JSON remote path as local payload custody: closed because neither
  expected byte object exists in the searched local/SSD scope.
- Using the DALI source Pose6 table as the fx5 baseline Pose6 table: closed
  because they are different scorer objects with different roles and hashes.
- Re-materializing the fx5 base fields with a new scorer job: closed because a
  COMPLETE retained remote run already owns the exact bytes.
- Keeping the 47.2 GB solve scratch on APDataStore: closed at this instance
  because APDataStore lacks the required free space and local APFS passes.
- Reopening PK4 linear per-pair overlays: formulation-closed at 43 to 997 bytes;
  JO2 uses nonlinear joint descent and native-carrier recompilation instead.
- Treating QS5's below-base Pose6 as sufficient admission: closed because its
  R2 refusal was decided by Seg/rate economics, which JO2 has not measured.
- Reopening WD4 slice warm-start: instance-closed by its 1,792-times gate
  failure.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — Owner: JO2 implementation owner; consumer
  store: `experiments/ddm_jo1_joint_objective_worker.py` and
  `experiments/ddm_jo1_modal_joint_objective.py`; fire trigger: wire the three
  real n600 stages to `solve_fresh_compensation`, receiver-close packaging,
  exact field admission, and distinct stage checkpoints, then pass two reviews.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN; consumer store:
  `experiments/.scratch/ddm_jo2_joint_objective_solve/retained/materializer/`;
  fire trigger: Modal control plane becomes reachable, then execute r5
  `FIRE_ORDER.json` ordinals 1A and 1B and verify their exact hashes and bytes.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner: JO2 seal owner; consumer store:
  `.omx/research/ddm_jo2_solve_reseal_20260821/seal_r6/`; fire trigger: both
  recovered base payloads are locally verified and triple-bound, then reseal
  so the memory-preflight command carries the new workload SHA.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN; consumer store:
  `experiments/.scratch/ddm_jo2_joint_objective_solve/retained/memory_preflight/`;
  fire trigger: reviewed remote trainer, payload-complete r6 seal, unique lane
  claim, and no conflicting governed job; fire the exact r6 memory command and
  harvest its terminal receipt.
- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN; consumer store:
  `experiments/.scratch/ddm_jo2_joint_objective_solve/stages/`; fire trigger:
  fresh matching T4 memory receipt and a new seal whose blockers are empty and
  status is `READY_TO_FIRE_UNDER_STANDING_GO`; fire the emitted training command.
