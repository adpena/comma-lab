# ddm_sa1 shipping-axis Seg actuator — READY_TO_FIRE, pointer unmoved

**Verdict:** `READY_TO_FIRE`, not a score and not a frontier move. SA1 built a real counted
conditioner, inserted it into the CP135 receiver, retained all locally materialized payloads, and
prepared the candidate-only T4 sign gate. The arm did not dispatch Modal, as required by the
charter; MAIN owns that exact checkpoint.

## Measured result

Stage A read the retained full-population T4 fields (`n=600`, 117,964,800 pixels) and reproduced
CP135's **34,970** T4 errors. All three pre-registered mechanism forms fit the 2,048-byte
worst-case box before training:

| form | raw plus ZIP upper bound | Brotli q11 plus ZIP upper bound | disposition |
|---|---:|---:|---|
| per-edge threshold | 502 B | 357 B | FOLDED into design evidence |
| g4 token bias | 1,655 B | 421 B | FOLDED into design evidence |
| hidden-4 context convolution | 1,353 B | 958 B | FIRED locally |

The selected byte-closed Stage-B cell is `stage_01_step_000001_ema`:

| quantity | result |
|---|---:|
| counted conditioner | 814 B, SHA `eae93d3d…57537` |
| candidate archive | 187,178 B, SHA `0724f690…e632` |
| exact archive delta vs CP135 | +926 B |
| local projected Seg change | −37 flips / 600 |
| robust-margin projected change | 0 flips / 600 |
| local pose delta | +1.159633947e−6 |
| local linearized joint delta S | +0.001284479372 |

All values in the last four rows are **[macOS-CPU ordering on contest-CUDA T4 target fields;
non-promotable]**. They are not evidence of shipping-axis improvement. Selection first required
actual local Seg improvement and pose-guard survival; this prevented the numerically lower-joint
step-8 EMA cell from being misnamed a Seg actuator when it worsened Seg by 57 projected flips.

The candidate archive keeps CP135 member `p` byte-identical and adds only
`sa1_conditioner.br`. Archive and deterministic repeat are byte-identical. The free receiver code
parses the counted module and changes 3,655 camera values in the one-pair CPU surface probe. That
probe is deliberately not called an exact receiver run: separate CPU processes differed by at most
`4.425e-4` before R and by one uint8 level on 113 camera values. Exact CUDA receiver consumption is
part of the queued T4 gate.

## T4 fire order

**Disposition:** QUEUED-WITH-A-FIRE-ORDER. **Owner:** MAIN. **Consumer store:**
`/Volumes/VertigoDataTier/pact/ddm_sa1_20260813/t4_sign_gate_v3/`. **Fire trigger:** MAIN
reconciles the live claim ledger and Modal single-flight state, then runs the exact command in
`READY_TO_FIRE.json` with the retained candidate archive/runtime inputs.

The gate is one exact receiver plus one full T4 SegNet field pass. Its timing projection is
805.405 s including a 300 s reserve, leaving 994.595 s inside the 30-minute limit. Every candidate
raw frame, SegNet input, logit tensor, and argmax field is retained on the existing JS1B Modal
volume with per-stage resume checkpoints.

At the selected local pose/rate price, the T4 candidate must remove at least **1,553** flips:
candidate flips must be at most **33,417** versus the 34,970 base, and mixed-axis admission delta
S must be negative. This is a checkpoint admission rule, not an exact score. If admitted, MAIN
still owes the full exact evaluator row before any pointer claim.

## Falsifiers and boundaries

- **F1 did not fire:** all three forms fit the byte box. This does not prove any form works.
- **F2 is not eligible yet:** no T4 candidate field exists. If the queued exact T4 field is above
  the 33,417 break-even ceiling or otherwise has nonnegative gate delta S, close the
  train-local/gate-T4 hybrid at **FORMULATION** scope and route only to true CUDA-in-loop training.
- **Measured:** full-n600 T4 target/base field structure; all Stage-A payload sizes; stratified-random
  n32 local ordering and pose guard; real module coder; deterministic archive bytes; local CPU
  receiver-surface consumption.
- **Not measured:** candidate T4 argmax field, T4 PoseNet, full evaluator score, contest-CPU,
  full-n600 local scorer, or frontier movement.
- The first silent local process handle was mistaken for completion and a second resume briefly
  overlapped it. No external job fired and no payload was deleted. Both failed-attempt trees remain
  under `winner/adapted_runtime`, `winner/.adapted_runtime.13244.partial`, and
  `winner/receiver_probe`; the production candidate is versioned under
  `winner/stage_01_step_000001_ema/`. A fail-closed exclusive run lock now prevents this class.

## RECALL EVIDENCE

The search scope was broader than the charter seeds:

- Canonical equations: `tools/list_canonical_equations.py --json`, queried for `argmax`, `shipping`,
  `edge`, `condition`, `round trip`, and `cuda`.
- Research index, DAG, docs, live board, and full `.omx/research` content were queried for
  `implicit conditioning`, `learned conditioner`, `shipping-axis`, `sign-flip`, `argmax field`,
  and `edge conditioning`.
- The retained JS3 store was inspected directly, including
  `/Volumes/VertigoDataTier/pact/ddm_js3_20260812/main_burn/` and `pose_guard_control/`.

Beyond the charter seeds, recall found a real hidden-4 learned conditioner and completed retained
JS3 burns: the 300-step cell had robust local movement but unacceptable pose harm, while the
8-step pose-control EMA was pose-safe but Seg-weak. This changed the plan from inventing a new
micro-network to reusing the real JS3 mechanism, rebinding its resume configuration to the exact T4
field hashes, replacing its private camera transform with HR2's canonical apparatus, and selecting
only cells that jointly pass the Seg and pose gates. The canonical-equation search also reinforced
that cross-axis sign is not transferable; therefore every local number remains ordering-only.

## Evidence and verification

- Main result: `/Volumes/VertigoDataTier/pact/ddm_sa1_20260813/FINAL_RESULT.json`
- Fire ticket: `/Volumes/VertigoDataTier/pact/ddm_sa1_20260813/READY_TO_FIRE.json`
- Prepared retained fire inputs:
  `/Volumes/VertigoDataTier/pact/ddm_sa1_20260813/t4_sign_gate_v3/fire_inputs/`
- Tests: SA1 plus JS3 focused suite passed; Python compile, Ruff, payload-retention gate, archive
  determinism, runtime/training module equality, run-lock, and T4 gate arithmetic are covered.
- A wider JS1B regression invocation had two failures caused by pre-existing JS1B test/implementation
  drift around the later C1 custody-SHA addendum; SA1 files do not change either failing file.

Exact pointer: **UNMOVED**. Current custodial frontier remains CP135 composed
`S = 0.16195513827824176 @ 186,252 B [contest-CUDA T4, n600]`. Current own-vehicle row remains
LC2 `S = 0.16959899569230852 @ 187,226 B [contest-CUDA T4, n600]`.
