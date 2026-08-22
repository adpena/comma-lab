# ddm_jo5 determinism cure and r8 reseal

## Outcome

Seal r8 is **READY_TO_FIRE_UNDER_STANDING_GO** with zero blockers. The retained
r7 `target_birth` checkpoint was migrated into r8 without restarting training:
all six model/EMA/optimizer/RNG/dual/cursor payloads are byte-identical, the
source checkpoint remains untouched, and a real restore smoke recovered cursor
`target_birth / step 600 / field 0 / package 0`.

The cure and its exact-source proof landed in commit `4359aa5e85`. This arm did
not launch the heavy n600 solve and did not measure a contest score.

## Mechanism and cure

The failure was not camera regeneration, uint8 realization, RNG state, or
multi-thread scheduling by itself. For pair 0, the final exploration batch and
the old singleton winner repeat produced byte-identical camera bytes
(`22c691a9...`) and byte-identical PoseNet-input bytes (`e70a6a6c...`), but the
Pose6 output differed by `3.814697265625e-06` max-abs. A one-thread control with
OMP/MKL/OpenBLAS/VECLIB/NUMEXPR pins and deterministic algorithms still differed
by `1.1444091796875e-05` max-abs. Python, NumPy, and Torch CPU RNG states were
unchanged across both forwards.

The mechanism is oneDNN/PoseNet floating accumulation choosing a different
reduction instrument for a batch than for a singleton. The cure repeats the
winner inside the complete final exploration candidate batch, preserving its
exact shape and order, then selects the same row. The deterministic gate remains
an exact byte comparison; no tolerance or skip was introduced. Every recomputed
batch receives the same certified-rebuild custody as exploration, and the
selected winner's camera, PoseNet input, codes, and Pose6 are retained in full.

## Executed determinism proof

Receipt: `DETERMINISM_REPROOF.json`, SHA-256
`8fd86de4504a3d592a13944878f404c561c3795eeb9e6cb70bf94db12e017076`.

The proof used three real pairs and three independent processes per pair. All
nine safe runs exited zero, retained every materialized payload, reproduced the
saved exploration Pose6 exactly, and reproduced the cure's camera, PoseNet input,
and Pose6 exactly. Each pair had exactly one SHA per surface across all three
repeats:

| Pair | Camera SHA-256 | PoseNet-input SHA-256 | Pose6 SHA-256 | Old singleton gap |
|---:|---|---|---|---:|
| 0 | `22c691a9924a...` | `e70a6a6c180c...` | `04bb0110cb40...` | `3.814697265625e-06` |
| 1 | `22c691a9924a...` | `21f65b259702...` | `e54354d08f6a...` | `1.601874828338623e-07` |
| 2 | `22c691a9924a...` | `a261b3bb4c4c...` | `990da2d5ccd1...` | `3.814697265625e-06` |

The old singleton repeat remained different in all nine runs, so the proof is
non-vacuous and discriminates the repaired mechanism from the failed one.

## Review and retention gates

- Two post-fix review passes marked all 135 entities across the four changed
  Python files reviewed: 100% in each file.
- Focused suite: 35 passed; the two warnings are pre-existing Pydantic
  `schema`-shadow warnings.
- Ruff, `py_compile`, and `git diff --check`: pass.
- Bounded P0 payload-discard census: 4 files, 0 findings.
- Committed entrypoint SHA-256:
  `766d3494751b27343df8904db2b74fd21e3d7804274a7e3931316ae11736bcdd`.
- Committed receiver-close SHA-256:
  `0ceba3210fdabe504d975d80ad4f480a7f4e53392cda9db07854713763870931`.

## Wall-clock and storage re-derivation

The cure keeps the four-thread training regime. Its measured same-batch repeat
cost is `1.547001948968197 s` mean and `1.6158145419321954 s` measured max per
pair-stage across nine calibration processes. Across 600 pairs and three stages,
that adds `2,784.603508 s` mean (`0.7735 h`) or `2,908.466175 s` at the measured
max (`0.8079 h`).

The final r8 real-config preflight remeasured one real step at
`1.4132779170759022 s`, measured peak RSS `2,855,944,192 B`, and projected n600
peak RSS `5,360,564,040 B` under the 16 GiB cap. The updated schedule is
`78,072.404010 s` lower (`21.69 h`) and `130,396.266677 s` upper (`36.22 h`).
The preflight retained all real tensors and passed the full receiver retention
projection with `582,749,302,784 B` free.

## r8 seal and checkpoint custody

- Compiled config SHA-256:
  `38d2f96dc755fd118eaccdac5985adaf6cff8e8beaea401669c8676600731b90`.
- Workload identity SHA-256:
  `b8986b44602ce7e0d6f18d74522aff89bb087990dafb414e45228cf66adc83e4`.
- Readiness: `READY_TO_FIRE_UNDER_STANDING_GO`; blockers: 0.
- Memory-preflight SHA-256:
  `b6be6406dc0b381e1362264c43b69618806eef470e8986d574871b523b7e25cb`.
- Migration source manifest SHA-256:
  `fe7c0e947bc65ced1cab0d209ce5c48c36f7b2d0da8f3d5a7f6b0886d13344e2`.
- Migration destination manifest SHA-256:
  `947495f1c28747f63ddb24f3ea5900ee2d585a2237c71a09923cbfd3ab389270`.
- Reseal receipt SHA-256:
  `2adf07fe2ffcb884cc8b9726488a58d8b0ad2f00033bb2708e5e03cdd0266570`.

The exact MAIN fire command is:

```bash
.venv/bin/python tools/spawn_durable_daemon.py --log /Users/adpena/Projects/pact/experiments/.scratch/ddm_jo2_joint_objective_solve/ddm_jo5_determinism_cure_reseal_20260821_r8_final/train.log --label ddm_jo2_joint_objective_fx5 --projected-gb 48 --min-free-gb 44 --rss-cap-mb 16384 --walltime-cap-s 259200 --projected-peak-gib 16.0 -- env TAC_GOVERNED_ADMISSION=1 .venv/bin/python -m experiments.ddm_jo3_joint_objective_entrypoint train --compiled-config /Users/adpena/Projects/pact/.omx/research/ddm_jo5_determinism_cure_reseal_20260821/seal_r8/compiled_config.json --expected-config-sha256 38d2f96dc755fd118eaccdac5985adaf6cff8e8beaea401669c8676600731b90 --resume-from /Users/adpena/Projects/pact/experiments/.scratch/ddm_jo2_joint_objective_solve/ddm_jo5_determinism_cure_reseal_20260821_r8_final/checkpoints --main-owned-dispatch-authorization
```

## RECALL EVIDENCE

The recall census searched `.omx/research/` memos and arm receipts,
`.omx/state/main_hot_state.md`, task/ledger surfaces, `CANONICAL_RESEARCH_INDEX*`,
the `sub015_DAG_*` FEED surfaces, design/SPEC documents, the canonical-equations
registry, and the live JO1-JO4 code. Queries included `Pose6`, `winner
deterministic repeat`, `same batch`, `batch shape`, `torch.set_num_threads`,
`deterministic_algorithms`, `oneDNN`, `reduction order`, and `JO1|JO2|JO3|JO4`.

Beyond the charter seeds, `ddm_et4_pair17_c2_batch_seam_diagnosis_20260806.md`
and `experiments/ddm_et4_rebuild_parent_argmax_cache.py` already measured that
oneDNN batch-1 and batch-16 forwards can differ at fixed threads and with
deterministic algorithms enabled. That evidence changed the plan from a presumed
single-thread pin to an explicit same-instrument batch-shape control, which the
JO5 probes then confirmed on PoseNet. The equations registry contained no
JO5-specific cure to reuse, and no existing same-batch JO5 implementation was
found in the searched scope.

## Boundaries

This is a `[macOS-CPU bounded determinism proof]` and a local real-config
memory/wall-clock seal. It is not a contest-CPU or contest-CUDA score. No n600
solver, upstream evaluator, Modal job, or exact contest replay was launched.
`upstream/` was not modified. The current frontier remains:

`fx5_e1 S 0.14823186109359 @ 180,386 B [contest-CUDA T4 n600]`.

## NEXT_IF_RESUMED

- **QUEUED-WITH-A-FIRE-ORDER** — Owner: MAIN; consumer store: `experiments/.scratch/ddm_jo2_joint_objective_solve/ddm_jo5_determinism_cure_reseal_20260821_r8_final/`; fire trigger: MAIN holds the unique local lane, r8 `READINESS.json` still reports `READY_TO_FIRE_UNDER_STANDING_GO` with zero blockers, the source/input triples still match, and MAIN executes only ordinal 3 from `seal_r8/FIRE_ORDER.json`.

## LIVE-HYPOTHESES

- Exact exploration-batch shape and order are sufficient to reproduce Pose6 for
  all 600 pairs, including endpoint-shortened candidate batches. This is
  plausible because the cure derives the repeat batch from the actual final
  candidate list, ET4 independently established batch shape as part of the
  oneDNN instrument, and all 9 JO5 processes passed; the full r8 solve remains
  the untested population proof.
- The migrated cursor will continue directly into retained `target_birth`
  materialization rather than repeating 600 training steps. This is plausible
  because all six state payloads are byte-identical and the real restore smoke
  recovered the exact step-600 cursor, but ordinal 3 is the first full control-
  flow execution on the migrated bundle.
- The r8 solve will remain inside its 72-hour daemon cap. This is plausible
  because the remeasured conservative upper projection is 36.22 hours with the
  same-batch repeat cost included; the complete n600 receiver workload remains
  unexecuted under r8.

## DEAD-ENDS

- Pinning Pose6 forwards to one CPU thread is closed as a cure for this instance:
  it preserved camera/input bytes but the batch-versus-singleton Pose6 gap
  remained and grew to `1.1444091796875e-05`; it would also change the measured
  training regime.
- Matching only thread count and deterministic-algorithm flags is closed: fixed
  four-thread and fixed one-thread controls both diverged when batch shape
  changed.
- RNG/cache leakage is closed in the bounded three-pair instance: Python, NumPy,
  and Torch CPU RNG hashes were unchanged across every final proof process.
- Carrier rendering and uint8 realization are closed as the divergence entry
  point: the batch and singleton camera and PoseNet-input bytes were identical;
  the first differing retained surface was Pose6.
- Weakening the equality gate with a tolerance or skip is rejected: it would
  certify a regeneration tuple that does not reproduce the retained Pose6 and
  would violate the charter's no-fake boundary.
- Restarting r8 from scratch is closed: the r7 step-600 checkpoint has been
  migrated and restore-validated byte-for-byte, so discarding it would violate
  the resumability contract without adding evidence.
