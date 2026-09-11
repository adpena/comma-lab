# ddm_obx2 — retention certification

<!-- FORMALIZATION_PENDING: a custody record over measured artifacts; it registers no equation and claims no score. -->

Date: 2026-09-11
Custody root: `/Volumes/VertigoDataTier/pact/ddm_obx2_edge_local_implicit_correction/`
Lane: `ddm_obx2_edge_local_implicit_correction_20260911`
Score claim: false · Promotion eligible: false
Volume at certification: **1,439 files, 4.9 GB across 19 Stage-2a rungs**; Vertigo free 29Gi

**Nothing was deleted, moved, or reduced to scalar-only evidence.** Every `/Volumes/...` source outside
this root was opened read-only. No live run's files were landed into the repository.

## 1. Pinned inputs — verbatim, byte-identical to their sources

| file | bytes | SHA-256 (first 16) |
|---|---:|---|
| `inputs/base.packet.qbf` | 106,606 | `607abebda2708f00` |
| `inputs/base.packet.repeat.qbf` | 106,606 | `607abebda2708f00` |
| `inputs/base.archive.zip` | 106,714 | `b26371e50696bdcd` |
| `inputs/base.archive.repeat.zip` | 106,714 | `b26371e50696bdcd` |

These match the QBT2B r10 pins in the burn spec exactly, and each was re-verified at every stage entry
(`stage0`). The move-44 decoded raw, the shipped `subset6.u8` token plane, the DALI GT argmax and the
DALI PoseNet targets were read from their own custody roots and re-hashed at use; they are **not**
copied here, because copying 3.66 GB that already has custody elsewhere would duplicate, not preserve.

## 2. Terminal checkpoints of the two stopped stages

Both training arms were stopped by rule, not by failure, and their checkpoints are preserved under
distinct stage-encoded names. **A re-aimed stage never wrote over the stage it resumed.**

| arm | checkpoint (newest) | bytes | SHA-256 (first 16) | state |
|---|---|---:|---|---|
| base_only | `base_only/checkpoints/obx2_joint_w2_epoch_00040.pt` | 1,736,957 | `e6bb7140dbf99837` | stopped: closed by md1-md4 at formulation scope |
| lattice | `lattice/checkpoints/obx2_joint_w2_epoch_00040.pt` | 1,876,365 | `af845aebbac1791b` | stopped: MAIN rate rule fired |
| lattice | `lattice/checkpoints/obx2_joint_w2_epoch_00030.pt` | 1,875,213 | `3c56b4a06f2372b8` | the object every n600 row in the closure was measured on |

Earlier stage-1 checkpoints (`obx2_joint_epoch_00010/00020/00030.pt` in each arm) are retained
alongside. Every checkpoint carries model state, the EMA shadow, optimizer state, both RNG states,
the full config including its pose-weight derivation, and the history — so each is resumable and each
is independently byte-closeable.

## 3. Candidate objects — the bytes every row was scored on

| file | bytes | SHA-256 (first 16) |
|---|---:|---|
| `lattice/candidates/lattice_obx2_joint_w2_epoch_00030.archive.zip` | 122,778 | `4bd22a4599bb993d` |
| `lattice/candidates/lattice_obx2_joint_w2_epoch_00030.packet` | 122,668 | `a03dcaeb8a511e60` |
| `lattice/candidates/…_00030_zerolattice.archive.zip` | 122,203 | `b93709b5853eeea3` |
| `lattice/candidates/…_00030_zerolattice.packet` | 122,093 | `ea96a1326d53bb4f` |
| `base_only/candidates/base_only_obx2_joint_epoch_00020.archive.zip` | 109,104 | `4b622dc8c1287c38` |
| `base_only/candidates/base_only_obx2_joint_epoch_00020.packet` | 108,994 | `287dd71fa970fbd7` |

None of these is a contest candidate: each fails the byte gate, the distortion gate, or both, and no
promotion or score claim attaches to any of them.

## 4. Result receipts

| file | bytes | SHA-256 (first 16) | what it holds |
|---|---:|---|---|
| `STAGE_2A_RESULT.json` | 76,463 | `f78b26a448b2b59a` | the Stage-2a ladder's per-rung components and per-chunk facts |
| `lattice/CHECKPOINT_SCORE_…_00030.json` | 5,903 | `2fd16af30a1ee582` | the with-lattice n600 row |
| `lattice/CHECKPOINT_SCORE_…_00030_zerolattice.json` | 5,906 | `04b7bf2926390270` | the zeroed ablation row |
| `lattice/SEG_DECOMPOSITION_…_00030.json` | 2,952 | `bfbdae0d7a036270` | the partition / render-floor split |
| `SEG_SLOPE_FALSIFIER_first.json` | 1,445 | `a06d2e26a83110d3` | the rule's first firing, before amendment A1 |
| `SEG_SLOPE_FALSIFIER_ep6.json` | 3,800 | `6b3063fc2e20c91a` | the six-epoch verdict as it fired |
| `SEG_SLOPE_FALSIFIER_a1.json` | 4,166 | `1ddd834197e73aa9` | the same data under amendment A1 |
| `SEG_SLOPE_lattice.json` | 2,533 | `c7c49e5a10c60aed` | the surviving arm's re-fit |

 covers the first ladder; the pose ladder and camera ladder wrote their rungs
into the same per-rung directories and their run logs under , all retained.

`STAGE_2A_RESULT.json` covers the first ladder; the pose ladder and the camera ladder wrote their rungs into the same per-rung directories, and all three ladders' run logs, manifests and done receipts are under `runs/`.

The two pre-amendment falsifier receipts are kept deliberately: the first verdict is reported as it
fired and was never retracted, so its receipt must survive alongside the amended one.

## 5. Bulk payloads and what is rebuildable from what

`stage_2a/` holds **570 files across 19 rungs** — the Stage-2a ladder's per-rung, per-20-pair chunks.
Each chunk carries the scorer's argmax, the PoseNet six-vector, and the targets; the decisive rungs
also carry their camera payloads and logits. `scored/` and `lattice/` hold the per-chunk argmax, pose
and targets of every checkpoint score, so no scorer output anywhere in this arm was materialized and
discarded.

**Certified rebuildable, with the command that rebuilds it.** Camera payloads for rungs outside
`RETAIN_CAMERA_RUNGS` were not stored verbatim; each chunk checkpoint records the rung's exact
deterministic recipe, the master seed `20260911`, and the **SHA-256 of the materialized camera bytes**,
so a consumer can rebuild them from the retained teacher and prove byte-identity:

```bash
.venv/bin/python experiments/ddm_obx2_edge_local_implicit_correction.py stage2a \
  --launch-authorized --rungs <rung>
```

Every render in this arm is likewise rebuildable from a checkpoint plus one command, because the
packet is the object:

```bash
# byte-close + score any retained checkpoint (writes its own argmax/pose/target chunks)
.venv/bin/python experiments/ddm_obx2_trainer.py score --checkpoint <checkpoint.pt> \
  --validate-pairs 600 --output <arm root>
# the partition / render-floor split on the same checkpoint
.venv/bin/python experiments/ddm_obx2_trainer.py decompose --checkpoint <checkpoint.pt> \
  --validate-pairs 600 --output <arm root>
```

`runs/` holds every governed launch's manifest, log, resource status and done receipt, including the
two the admission gate **refused** and the stages that were stopped — refusals and stops are evidence
and were not tidied away.

## 6. What is NOT here, and why

* The move-44 decoded raw (3.66 GB), the shipped token plane, and the GT arrays live in their own
  custody roots and are referenced by pinned SHA-256 rather than copied.
* No contest candidate archive, no Modal artifact, no paid-dispatch output: this arm made none.

## 7. Statement

Every measured payload this arm produced is retained or is certified rebuildable from a retained
checkpoint by a named command, with a recorded hash to prove byte-identity. No payload was measured
and discarded. No file was deleted.

The frontier is unchanged: **composition S 0.1372449041713402 @ 180,406 B [contest-CUDA T4 n600]
(move 44)**.
