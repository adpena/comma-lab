# ddm_wd3_scorer_aware_width_distillation — sealed build-and-measure charter

Status: **QUEUED-WITH-A-FIRE-ORDER; NO LAUNCH AUTHORIZED.** This is a successor charter, not a
claim that the scorer-aware trainer exists. The declarative design is
`.omx/research/ddm_wd3_scorer_aware_width_distillation_20260815.json`. Do not turn its fields into
invented command-line flags. First build and review the real surfaces named below.

## Objective

Determine whether the retained WD2 student family can exchange its measured rate saving for a
small, score-admissible distortion increase when trained on what the contest actually measures.
The WD2 instance optimized camera decode MSE and produced a retained 165,387 B archive, but its
n600 advisory render had `d_seg=0.00117677` and `d_pose=0.09198625`. WD3 changes the mechanism:
train through the receiver's resize/clamp/uint8/scorer chain and preserve the teacher's realized
SegNet decision geometry and PoseNet first six outputs.

This is an **INSTANCE reactivation**, not a family success claim. Its only success condition is a
receiver-closed archive that passes the exact score gate against a same-instrument hv1 base.

## Hard gates before any launch

1. **G0, base identity — PASS:** consumed
   `/Volumes/APDataStore/pact/ddm_hv1_base_advisory_n600_cpu/contest_auth_eval.json`, 23,416 B, SHA
   `cfdac1fd0965095152ffd88c878d9c4b8f38c644d755e594ad028a798daf3a7f`. It binds archive SHA
   `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`, 182,759 B, n600,
   post-sweep mirror SHA `fa7c4bf51d47a6140ec0f95275ebf86b0e6c3c1dc00caff03a417ee989645799`, and the same local
   environment/instrument used for the student. Local base: `d_seg=0.00042714`,
   `d_pose=0.00014747`, recomputed `S=0.20280753928705508`, `[env-mismatch advisory]`. These values
   are now stamped in the design JSON; refuse cross-instrument admission arithmetic.
2. **G1, ownership:** MAIN owns the build and fire. AV2 has no scorer or Metal slot. Fire only after
   r5 PID 63183 exits and both the global Metal lane and single n600 scorer lane are explicitly
   claimed.
3. **G2, real code:** land a scorer-aware extension to the existing WD2 trainer, a teacher-scorer
   cache builder, receiver parse-back, typed config compiler, resume-registry entries, and retention
   hooks. Two review passes are required for every changed Python file. Do not use
   `REVIEW_GATE_OVERRIDE`.
4. **G3, storage:** pass the storage waterfall. Root new bulk at
   `/Volumes/APDataStore/pact/ddm_wd3_scorer_aware_width_distillation/`. Certify-or-block cleanup;
   never delete or move an unmanifested payload.
5. **G4, reproducibility:** one recorded seed controls Python, NumPy, Torch, and sampling. Preserve
   model, optimizer, EMA, scaler, scheduler, subset, cursor, config, and every stage checkpoint via
   atomic write-and-rename. A determinism repeat is mandatory before candidate promotion.
6. **G5, dry-run:** the typed compiler must reject missing base receipt, missing teacher/scorer
   cache, a prefix subset, non-retaining output, absent resume state, invented flags, an unclaimed
   lane, or chunk size above 120 before a heavy process starts.

## Retained inputs

| Object | Custody |
|---|---|
| Teacher camera renders | `/Volumes/APDataStore/pact/ddm_wd2_width_distillation/teacher_cache_e480b/retained/teacher/teacher_master_camera.rgb.u8`, 1,831,204,800 B, SHA `695023d4ca56e14f53f1e90b56134821c3c0a0c66f9b07f6aa6bd6ffdf9f4ebd` |
| WD2 ep60 checkpoint | `/Volumes/APDataStore/pact/ddm_wd2_width_distillation/primary_flattened_d4_w64/checkpoints/flattened_d4_w64/distill_qat_stage_end_epoch_0060.pt`, 583,929 B, SHA `046ee7d0171e04c3d468edd747a82bc81eb91642e5e85f17316b4419fe615071` |
| WD2 train receipt | `/Volumes/APDataStore/pact/ddm_wd2_width_distillation/primary_flattened_d4_w64/TRAIN_RESULT.json`, 16,234 B, SHA `c4260cf03eb4cb19f1788150592bf84f5468ebc5052dc0ab8a0ee123c3577918` |
| WD2 student archive | 165,387 B, SHA `e9c4a9ed5e6bef89d228ca877a9f9e37345e3c79dc07ba20087c218ff89fcf87` |
| WD2 advisory receipt | `/Volumes/APDataStore/pact/ddm_wd2_width_distillation/primary_flattened_d4_w64/retained/candidates/flattened_d4_w64_epoch_0060/attempt_0000/advisory_n600_cpu/contest_auth_eval.json` |

All inputs are read-only. Every newly materialized cache, candidate render, encoded packet, and
archive is retained with bytes and SHA-256. A scalar-only measurement artifact is forbidden.

## Stage 0 — build the teacher scorer cache once

Run the retained teacher camera stream through the exact `R` chain and frozen scorers in chunks of
at most 120. Retain, for all 600 pairs:

- SegNet five-class logits in a declared stable dtype, argmax, and top1-runner-up margin;
- PoseNet first six outputs;
- the original-video SegNet argmax and PoseNet first-six targets used by contest distortion;
- source archive, mirror snapshot, scorer-weight hashes, command, environment, shapes, dtypes,
  per-file SHA-256, aggregate SHA-256, and a determinism repeat.

Cache teacher outputs once; all architecture arms consume the same bytes. Do not recompute the
teacher to hide a changed environment. If the repeat is not byte-identical, stop and classify the
numerical seam before training.

## Stage 1 — scorer-aware objective through the real chain

Let `R` be the receiver render followed by resize to camera geometry, clamp, uint8 STE, and frozen
scorer preprocessing. The hard selection score is always

`S = 100*d_seg + sqrt(10*d_pose) + (25/37,545,489)*archive_bytes`.

The train objective has three roles:

1. **Seg decision geometry:** the primary differentiable term is original-target soft disagreement
   after `R`. At each stage boundary, derive and freeze its conversion into `d_seg` units as
   `measured hard d_seg / mean soft disagreement` on the same fixed controller subset; never change
   that scale per step. The score coefficient on the calibrated quantity is exactly 100. An
   impostor-complete margin constraint separately makes the teacher's winning class beat every
   student competitor. A T=2 soft-logit KL constraint supplies dense teacher structure, but it is
   auxiliary. Prior evidence says pure KL diverged and KL-dominant KD+CE was worse; do not transfer
   the old vehicle's `kd_w=0.3` as a WD3 constant.
2. **Pose:** match the teacher's first six PoseNet outputs after `R`, while computing the actual
   differentiable MSE to original-video PoseNet targets. Price pose with its exact nonlinear
   `sqrt(10*d_pose)` term, not a guessed scalar. The local diagnostic exchange rate is
   `5/sqrt(10*d_pose_base) = 130.20215255102787` on the consumed local base. Admission still uses
   the exact nonlinear Pose term, not this local linearization.
3. **Decode anchor:** keep `decode_mse_uint8 <= 50.6728233448345`, the retained ep60 endpoint. Treat
   it as a trust-region constraint, not the primary objective.

The Seg score exchange coefficient is exactly 100 after the stage-frozen calibration above.
Teacher-margin preservation, decode anchoring, and teacher-KL preservation use nonnegative adaptive
duals initialized at zero and updated only on constraint violation; they do not receive hand-chosen
fixed weights. Admission always uses hard realized scorer components and exact archive bytes. A
falling smooth loss is not a verdict.

## Stage 2 — validation in the loop

- At every preserved stage checkpoint, and at least every five epochs, render a **fixed evenly
  strided n60** subset through the real receiver, `R`, uint8, SegNet, and PoseNet. It is an early
  warning/control surface only.
- Never use a contiguous prefix. Prefix pose difficulty is measured 2.54–4.21x harder than the
  population while Seg prefixes are 3–5% easier.
- No negative may be banked from n60. Confirm any prospective negative on a fixed seeded,
  stratified-random n120 set at matched state and compute before stopping an architecture.
- A candidate must then parse back and pass a retained n600 same-instrument evaluation under the
  one-scorer-slot rule. Chunk size remains at most 120.

## Architecture and reset order

Run arms sequentially; do not spend the wider rung before the cheaper mechanism test.

1. **W0: flattened d4/w64 ep60 warm continuation.** Preserve model, optimizer moments, EMA, and
   cursor, then engage the new objective as a stage boundary with re-treatment. This isolates the
   changed loss at the lowest cost.
2. **W0-reset control:** same weights with fresh optimizer only under a magnitude-matched ramp. The
   ramp must remove the known 3.16x–6.57x zero-moment Adam step excursion. It is not a free reset.
3. **D56: dense d4/w56.** This has the smallest measured raw packet and changes the inherited
   computation least.
4. **F64: factorized d4/w64/r19.** This tests the factorized form at nearly the primary rate.
5. **W96 conditional:** enter only if the smaller arms preserve scorer cells but show capacity
   pressure. Start with factorized or flattened w96. Dense w96 is projected larger than hv1 and is
   priced out unless a real coder disproves the projection.
6. **Fresh model birth:** allowed only if both matched W0 continuations fail the seeded n120
   mechanism gate at matched compute. Fresh-versus-warm is not a binary choice; reset state and
   reset magnitude are separate actuators per #816.

## Rate-prize erosion and admission

`25/37,545,489 = 6.658589531221714e-7 S/B`. The 165,387 B flattened ep60 archive is measured. The
d4/w56, d4/w64/r19, and flattened d4/w64 raw packet counts are exact structural accounting from
`build_v4/DESIGN_RECEIPT.json`. The w96 raw counts are derived from a read-only
`wd2_receiver.serialized_bytes_for_spec` call; no w96 payload exists. Other full-archive sizes below
are **projections**, using fixed remainder 148,739 B plus
`round(exact_uncompressed_packet_bytes * 16,648/19,465)`. They are routing estimates, not byte claims.

| Arm | Exact/derived uncompressed packet | Full archive | Full-archive status | Bytes saved vs hv1 | Rate prize | Pose-held max `Delta d_seg` after `-3.5e-6` | Effective cap |
|---|---:|---:|---|---:|---:|---:|---:|
| dense d4/w56 | 18,905 | 164,908 | PROJECTED | 17,851 | 0.01188625 | 1.18827e-4 | **1.07e-4** |
| factorized d4/w64/r19 | 19,513 | 165,428 | PROJECTED | 17,331 | 0.01154000 | 1.15365e-4 | **1.07e-4** |
| flattened d4/w64 ep60 | 19,465 | 165,387 | **MEASURED** | 17,372 | 0.01156730 | 1.15638e-4 | **1.07e-4** |
| factorized d4/w96/r20 | 29,833 | 174,255 | PROJECTED | 8,504 | 0.00566246 | 5.65896e-5 | 5.65896e-5 |
| flattened d4/w96 | 35,657 | 179,236 | PROJECTED | 3,523 | 0.00234582 | 2.34232e-5 | 2.34232e-5 |
| dense d4/w96 | 40,265 | 183,177 | PROJECTED | -418 | -0.00027833 | none | priced out by projection |

For every candidate, recompute

`Delta S = 100*(d_seg_candidate-d_seg_base) + sqrt(10*d_pose_candidate) - sqrt(10*d_pose_base) + (25/37,545,489)*(candidate_bytes-base_bytes)`.

Admit only if `Delta S < -3.5e-6`, `Delta d_seg <= 1.07e-4`, receiver parse-back is exact, all
payloads are retained, and the same-instrument base is bound. The nonlinear pose term may consume
some or all of the table's pose-held allowance; there is no separate free pose budget.

## Checkpoints and payloads

Save atomically at most every five epochs and at every stage boundary. Preserve distinct filenames
for every stage. A checkpoint is incomplete unless it reloads model, optimizer, EMA, scheduler,
RNG, subset, cursor, loss duals, and exact config. Retain per arm:

- every student model payload and checkpoint;
- every periodic rendered candidate and scorer-output bundle;
- every exact encoded student packet, not just its byte length;
- every `archive.zip`, `archive.repeat.zip`, runtime, parse-back receipt, and n600 receipt;
- a manifest with path, bytes, SHA-256, source/runtime hashes, command/config/environment, axis,
  and cleanup disposition.

## Stop and promotion rules

- Stop an arm only after its seeded n120 confirmation or an objective fail-closed build/runtime
  blocker. Scope every negative to the instance and architecture.
- Stop the family only if all preregistered forms reach charter-time optimal form and fail the
  same-instrument admission equation. W0 failure alone does not kill distillation.
- Do not submit, promote, or move the canonical pointer from a macOS/MPS result. A contest-CPU or
  contest-CUDA n600 exact evaluation on the exact retained archive is required.

## Fire order and ownership

Disposition: **QUEUED-WITH-A-FIRE-ORDER**. Owner: **MAIN**. Consumer store:
`/Volumes/APDataStore/pact/ddm_wd3_scorer_aware_width_distillation/`. Fire trigger: G0–G5 all pass,
r5 PID 63183 has exited, scorer and Metal lanes are claimed, and the WD3 code/config dry-run is
reviewed. Sequence: build/cache -> W0 preserved-state -> W0 magnitude-matched reset -> D56 -> F64 ->
conditional W96 -> same-instrument n600 -> exact contest axis only for a passing candidate.

Vehicle frontier UNCHANGED: **S 0.15959729295498598 @ 182,759 B [contest-CUDA T4, n600]**, archive
SHA `80d9c8c6fdc72caaa3e180a8abb2a859e7f316a484b38f33fe90d5701420178e`.
