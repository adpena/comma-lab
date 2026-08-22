r8 is **READY_TO_FIRE_UNDER_STANDING_GO** with zero blockers. The determinism cure landed in `4359aa5e85`; the seal and migration evidence landed in `96c38ecdcb`.

Mechanism: PoseNet’s CPU floating reduction changes with batch shape. Camera and PoseNet-input bytes were identical, but the old singleton winner repeat produced different Pose6 values. One-thread execution did not cure it.

Cure: replay the complete final exploration batch in its original order and select the same winner row. The exact equality gate remains intact.

Proof and custody:

- 3 real pairs × 3 independent processes: 9/9 exact camera, input, and Pose6 repeat matches.
- Old singleton path still diverged in 9/9, making the proof non-vacuous.
- Two clean reviews; all 135 touched Python entities reviewed.
- 35 tests passed; Ruff, compilation, and payload-retention gate passed.
- All six checkpoint payloads migrated byte-identically from r7.
- Restore smoke recovered `target_birth`, step 600, without restarting training.
- Projected r8 wall time: 21.69–36.22 hours; projected RSS 5.36 GB under the 16 GiB cap.

Artifacts: [full memo](/Users/adpena/Projects/pact/.omx/research/ddm_jo5_determinism_cure_reseal_20260821.md), [determinism proof](/Users/adpena/Projects/pact/.omx/research/ddm_jo5_determinism_cure_reseal_20260821/DETERMINISM_REPROOF.json), [reseal receipt](/Users/adpena/Projects/pact/.omx/research/ddm_jo5_determinism_cure_reseal_20260821/RESEAL_RECEIPT.json), [FIRE_ORDER](/Users/adpena/Projects/pact/.omx/research/ddm_jo5_determinism_cure_reseal_20260821/seal_r8/FIRE_ORDER.json).

Exact MAIN fire command:

```bash
.venv/bin/python tools/spawn_durable_daemon.py --log /Users/adpena/Projects/pact/experiments/.scratch/ddm_jo2_joint_objective_solve/ddm_jo5_determinism_cure_reseal_20260821_r8_final/train.log --label ddm_jo2_joint_objective_fx5 --projected-gb 48 --min-free-gb 44 --rss-cap-mb 16384 --walltime-cap-s 259200 --projected-peak-gib 16.0 -- env TAC_GOVERNED_ADMISSION=1 .venv/bin/python -m experiments.ddm_jo3_joint_objective_entrypoint train --compiled-config /Users/adpena/Projects/pact/.omx/research/ddm_jo5_determinism_cure_reseal_20260821/seal_r8/compiled_config.json --expected-config-sha256 38d2f96dc755fd118eaccdac5985adaf6cff8e8beaea401669c8676600731b90 --resume-from /Users/adpena/Projects/pact/experiments/.scratch/ddm_jo2_joint_objective_solve/ddm_jo5_determinism_cure_reseal_20260821_r8_final/checkpoints --main-owned-dispatch-authorization
```

No heavy n600 solve or contest evaluation was launched. Frontier unchanged: `fx5_e1 S 0.14823186109359 @ 180,386 B [contest-CUDA T4 n600]`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN; consumer store: `experiments/.scratch/ddm_jo2_joint_objective_solve/ddm_jo5_determinism_cure_reseal_20260821_r8_final/`; fire trigger: MAIN holds the unique local lane, r8 remains ready with zero blockers, all source/input bindings still match, and MAIN executes ordinal 3 from `seal_r8/FIRE_ORDER.json`.

## LIVE-HYPOTHESES

- Exact exploration-batch shape and order will reproduce Pose6 across all 600 pairs, including shortened endpoint batches. Nine independent proofs and the prior ET4 batch-instrument evidence support this; the full r8 solve is the population test.
- The migrated cursor will proceed directly into retained `target_birth` materialization. All checkpoint bytes and the restore smoke agree, but ordinal 3 is the first complete execution.
- r8 will remain within its 72-hour cap. The conservative measured projection is 36.22 hours, though the complete receiver workload remains unexecuted.

## DEAD-ENDS

- Single-thread pinning is closed: batch-versus-singleton Pose6 still diverged and the gap increased to `1.1444091796875e-05`.
- Fixed thread count plus deterministic-algorithm flags is closed: batch shape remained decisive.
- RNG/cache leakage is closed in the bounded proof: Python, NumPy, and Torch RNG hashes were unchanged.
- Carrier rendering and uint8 realization are closed as the entry point: camera and PoseNet-input bytes matched exactly.
- Tolerance or skip-on-mismatch is rejected because it would weaken the custody gate.
- Restarting from scratch is closed because the retained step-600 checkpoint is migrated and restore-validated.