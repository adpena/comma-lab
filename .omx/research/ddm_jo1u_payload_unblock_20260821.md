# DDM JO1U payload unblock — retained materializer build receipt

- Date: 2026-08-21
- Disposition: **BUILT-AND-REVIEWED; READY fire-order, NOT FIRED**
- Owner of any paid fire: **MAIN only**
- Consumer store: `/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/`
- Authority boundary: component-payload build and custody only; no candidate archive, score row,
  training result, or pointer movement was produced here.

## RESULT

JO1's scorer-payload producer is no longer coupled to the fresh-Schur training implementation or
the 44 GiB training storage gate. The new worker performs the work named by the entrypoint: it
decodes the exact live `fx5_e1` receiver twice, requires raw-byte repeat identity, runs the frozen
T4 SegNet and PoseNet over all 600 pairs in batches of 16, and retains both raws, every scorer input,
every full scorer output, the base argmax field, base Pose6, all batch receipts/cursors, scorer and
runtime identities, immutable stage checkpoints, and a final scorer tuple. Finalization reopens and
hash-verifies every retained per-batch payload before it may write `COMPLETE`.

The two other entrypoints were inspected for the same coupling class. `memory_preflight` and
`train` still route through the named fresh-Schur implementation blocker, and the 44 GiB minimum is
now a validated training-config invariant. That coupling is appropriate because those entrypoints
exercise or qualify the training backend; neither was weakened or made fireable by this build.

## TYPED DELIVERY ROWS

| row | disposition | evidence and boundary |
|---|---|---|
| defect fix | **BUILT-AND-REVIEWED** | `experiments/ddm_jo1_modal_joint_objective.py` now sends only `materialize_scorer_payloads` through the real retained producer. `memory_preflight` and `train` remain fail-closed. The dispatcher reuses the canonical auth-eval image, claim, fleet-wide single-flight, durable call-id, volume, and terminal-recovery apparatus. No dispatch occurred. |
| retained worker | **BUILT-AND-REVIEWED** | `experiments/ddm_jo1_payload_materializer_worker.py`; T4-only, seed 1234, deterministic algorithms on, exact receiver repeat, full-population n600, batch 16 (therefore every cursor increment is `<=120`), atomic immutable checkpoints, resumable from the workload digest, every input/intermediate retained on the mounted auth-cache volume. Failure cleanup is forbidden; partial bytes remain in custody. |
| seal | **READY_TO_FIRE** | `/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/seal_r1/compiled_config.json`, 15,394 B, SHA `a217a5273536a98766379d89f5834026aedbd5ccdb4b63a3e5f9165ba7a0dc40`; workload SHA `83eff3e7eb01919e78925b3a36ff21e0ae8db007d4f431f4b03750815cd25090`. Source-code pins were recomputed after the last edit. |
| storage preflight | **PASS for producer; BLOCKED for training** | Seal-time AP free = 33,372,110,848 B / 31.080200 GiB `[MEASURED local APDataStore]`. Expected retained producer payload = 16,880,011,200 B / 15.720735 GiB; success reserve = 4,294,967,296 B; required free after already-retained seal bytes = 21,174,944,490 B / 19.720704 GiB. The 47,244,640,256 B / 44 GiB training requirement is recorded and explicitly not applied to materialization; it continues to block training on current free space. |
| fire-order | **QUEUED-WITH-A-FIRE-ORDER** | `/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/seal_r1/FIRE_ORDER.json`, SHA `d64c7356f80fb639fccffb914bded4e09d6a586176ce1f90fea9352c562afcf8`; owner MAIN; lane `ddm_jo1_payload_unblock`; fire trigger is producer storage PASS, no active n600 scorer job, and a unique MAIN lane claim. The Modal CLI parsed the sealed combined `file.py::entrypoint` form. |

## BOUND ARTIFACTS

| artifact | typed custody |
|---|---|
| source Pose6 | `/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/direct_v6/full_n600_eval/retained/pose_vectors/gt_first6_dali_n600.npy`; 14,528 B; SHA `8d5cfa83df55b89493ba43b1e5386d792c836c32791666192499a089068e7eff`; shape `(600, 6)` float32; `[contest-CUDA T4 DALI_NVDEC Pose6 source target n600]`; lineage registry says exact match. |
| GT argmax | `/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy`; 117,964,928 B; SHA `91d3ff11a904c476b56a8be8af2225fb4a390d02fac9d3b09ef4704ad6e77248`; shape `(600, 384, 512)` uint8; `[contest-CUDA T4 DALI GT n600]`. |
| live base archive | `/Volumes/APDataStore/pact/ddm_fx5/candidate_runtime_fx5/archive.zip`; 180,386 B; SHA `4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841`; `[contest-CUDA T4 fx5_e1 live-base exact archive n600]`. No rc2 fallback was used. |
| live base runtime | `/Volumes/APDataStore/pact/ddm_fx5/candidate_runtime_fx5`; declared-tree artifact SHA `f910e29a6fa5007a92cf27808647e3b34fe9b63c6ea77d5b4fbea87805b0c5b9`; canonical Modal-projected runtime-tree SHA `8eff613ecec2c371a6fa4cc580b8af9df131f45dc33f5d3c9b829faac1a513a5`; content-tree SHA `70ab25d5980b1e743d65382e8d0efbb9f5564f84bd716c03df58fb5d7fb5407a`. |

The existing rc2 decoded semantic-token payload remains bound in the umbrella config. It is not a
producer dependency for the live-base scorer materializer and was not copied or recomputed.

## RECALL EVIDENCE

The full-corpus recall used content queries, not only charter filenames:

- `.omx/research/`, design/SPEC/task-ledger surfaces, canonical research indexes, and the
  `sub015_DAG_*` FEED graph were searched for `source Pose6|gt_first6_dali|DALI_NVDEC|gt_argmax_n600|91d3ff11`,
  `materialize_scorer_payloads|TRAINING_IMPLEMENTATION_BLOCKER|memory_preflight|44 GiB`, and
  `fx5_e1|4b54fccc|runtime_tree`.
- Source and receipts were searched for
  `decode_exact_receiver|score_argmax_field|score_pose_vectors|retained_batch_receipts|cursor`.
- `.venv/bin/python tools/list_canonical_equations.py --json` was run; no new equation changed this
  plumbing plan.

Beyond the charter seeds, recall found the reusable, real producer substrate in
`experiments/ddm_js1b_cuda_argmax_field_materializer_worker.py` and its full-eval consumer pattern in
`experiments/ddm_pz4r_full_n600_eval.py`. That changed the plan from writing new scorer/decode logic
to a narrow JO1 wrapper around already-proven exact-receiver, per-batch retention, cursor, and resume
primitives. The canonical auth-eval runtime manifest also exposed and fixed an initially invalid
split Modal entrypoint token before sealing.

The bounded search did not find a retained `fx5_e1` base argmax or base Pose6 tensor in the JO1,
fx5, APDataStore, or Vertigo custody scopes. The fx5 authority receipt says scorer tensor caching was
not requested. Therefore those two base tensors remain genuine products of the queued T4 pass, not
stale-custody bindings.

## PRIOR-LAW COUNT

Counting the charter-known repair, one stale binding landed: source Pose6 already existed. One
genuine producer obligation landed: the current-base scorer tensors. The stronger M2 prediction of
an additional, previously unlisted stale-custody payload did **not** land in the bounded recall
scope. GT argmax and rc2 semantic tokens were already named and bound before this split, so neither
is counted as a new discovery.

## VERIFICATION

- Two clean review-tracker passes were completed on each touched Python file after the final edits.
- `ruff`, `py_compile`, `git diff --check`: PASS.
- `pytest -q experiments/tests/test_ddm_jo1_joint_objective.py`: **16 passed**; the two Pydantic
  `schema` shadow warnings pre-existed this split and are not test failures.
- Always-keep-the-payload AST gate on the three producer/dispatcher/design sources: PASS, zero
  findings.
- Production seal reload, all artifact-pin verification, current runtime-tree recomputation, and
  fire-order CLI parse: PASS.
- Repo-wide developer preflight: **17/25 PASS, 8/25 RED**. Every reported path is outside the JO1U
  touched set (legacy launchers, state writers, AGENTS wording, old substrate losses, and old lane
  references). This is a bounded baseline disclosure, not a claim that the repository-wide gate is
  green and not authorization to change unrelated user work.

## MEASURED / NOT MEASURED

**MEASURED now:** exact source/GT/archive/runtime bytes and hashes; source Pose6 shape/dtype and
DALI lineage; AP free bytes; producer storage arithmetic; source review/test/gate results; sealed
readiness. These are local custody/build measurements with the axes stated above.

**NOT MEASURED:** no Modal fire, T4 decode, new base argmax, new base Pose6, scorer tuple, training,
candidate archive, `d_seg`, `d_pose`, score, contest-CPU replay, contest-CUDA score replay, cost, or
wall time occurred in this arm. The materializer estimate is capacity planning, not a produced
payload measurement. `READY_TO_FIRE` means the named producer can now be fired under its trigger; it
does not mean its future outputs exist.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/`; fire trigger: the sealed materializer storage preflight still passes, no full-n600 scorer job is active anywhere in the fleet, and MAIN holds the unique `ddm_jo1_payload_unblock` claim; action: fire ordinal 1 only, then stop for terminal harvest.
- **QUEUED-WITH-A-FIRE-ORDER** — owner: MAIN scorer-lane dispatcher; consumer store: `/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock/harvest/`; fire trigger: ordinal 1 is terminal with `FINAL_RESULT.json` complete and every retained batch record verifies; action: harvest the volume, bind the base argmax/Pose6 into a new seal, and do not continue into memory preflight or training in the same step.
- **BLOCKED** — owner: `ddm_jo1_joint_objective_design` plus MAIN; consumer store: `/Volumes/APDataStore/pact/ddm_jo1_joint_objective_design/`; fire trigger: harvested current-base tensors are resealed, a real T4 peak-memory receipt exists, AP free is at least 44 GiB, and the fresh same-object Schur receiver-close backend is implemented and reviewed; action: only then expose the training fire-order.

## LIVE-HYPOTHESES

- The retained current-base fields will dissolve JO1's remaining data-input blocker because the
  worker uses the exact live archive, receiver, DALI scorer path, and full-population n600 census.
  This is plausible by direct reuse of the JS1B/PZ4 producer rather than by a proxy renderer.
- Materialization will fit the present AP tier because the sealed worst-case retained tensor model
  plus 4 GiB reserve requires 19.720704 GiB while 31.080200 GiB was free. This remains conditional
  on the fire-time storage preflight and creates no training capacity claim.

## DEAD-ENDS

- Recomputing source Pose6 is closed as duplicate work: the exact DALI payload exists, hash-verifies,
  and is registry-bound.
- Falling back to rc2 is closed for this seal: the live fx5 archive and matching runtime are present
  and canonical runtime-tree verification passes. The schema refuses rc2 without a written reason.
- Letting materialization inherit the fresh-Schur or 44 GiB training blockers is closed as the
  wrong dependency graph: the producer performs exact decode and frozen scoring without training.
- Scalar-only scorer receipts are closed: both exact raws, all scorer inputs/full outputs, cursors,
  final fields, and scorer tuple must hash-verify before `COMPLETE`.
- Firing memory preflight or training from this seal is closed: their command rows are null and the
  implementation/storage blockers remain intact.

Own-vehicle frontier: **fx5_e1 — S 0.14823186109359 @ 180,386 B [contest-CUDA T4, n600]**, archive SHA `4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841`; **UNMOVED by JO1U**.
