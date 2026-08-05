# OD1 sealed launch tickets - 2026-08-05

Status: `SEALED_FOR_MAIN_REVIEW / NOT FIRED`.

Axis: `[macOS-CPU advisory / scorer-free launch design]`.
`score_claim=false`, `promotion_eligible=false`, `n600_scorer_job=false`.
`READY_TO_FIRE=false`; MAIN owns dispatch.

## Ticket OD1-T0 - Current Batch Join

Disposition: `QUEUED-WITH-FIRE-ORDER`.

Purpose: wait for MAIN's pe2 three-candidate scorer batch to land before OD1 consumes carrier survival information.

Inputs:

- `.omx/research/ddm_pe2_20260805/PE2_RECEIPT_20260805.md` sha `06cbf9dc8492210a4683bf81d8872afcddfaa492ceadf186edca59c08df4fef7`.
- `.omx/research/ddm_pe2_20260805/PE2_QUEUE_NOTE.md` sha `90c50588332d815fe71b08ea117dcb0110ac2a265e9b9cda9acf655c5de964cb`.
- `.omx/research/ddm_pe2_20260805/pe2_three_candidate_scorer_manifest.json` sha `633e52f94bd6fdecaaae6d5bee10c7ead52f288eb114660f556031667a216b3d`.

Fire condition:

- MAIN confirms the pe2 staged batch has terminal receipts or explicitly releases the slot.

Expected outputs:

- Three recomputed S rows for PE1 full, PE1 surgical, and BF1, each with d_seg, d_pose, archive bytes, axis, and archive sha.

Blocker if missing:

- `OD1_BLOCKER_PE2_SURVIVAL_ROWS_PENDING`.

## Ticket OD1-T1 - Seg Base And Frame0 Pose Prototype

Disposition: `QUEUED-WITH-FIRE-ORDER`.

Purpose: build the first OD1 final-composition prototype without using the scorer slot during build.

Route:

1. Materialize a regional/sq2-derived seg base on frame_1.
2. Apply frame_0 C-PRIME/k=4 pose carriage or a measured better frame_0 basis.
3. Preserve all video-derived payload in `archive.zip`; free code may contain only generic deterministic algorithms.
4. Emit parse-back, absent-identity where applicable, and runtime changed-pixel proof.

Inputs:

- SQ2 memo sha `4a0bf7f7e7a104068f29790be840318a1b9412335ca85e51cd093f9b46f1da6c`.
- TJ1 summary sha `5c41bed2b8b9305cde023571993939f4904fbcab23fc9d8bb2e6a7bca155eefe`.
- JS1 staging memo sha `e71476863f4f7259194b2cdc10094f12e6d8045d3e6f55f3872dfb130de8f5b9`.
- ET1 memo sha `141e21797d27ec0dbe60dc863b80e5c83dfe4386750d7702df9c949290791a90`.
- CW1 memo sha `60960a16e62fc041f3ba5dabaf1bd091c0bd082a6bb041f46f105db07ac4ce0b`.

Gates:

- `G1_SOURCE`: all source archives and payloads have durable paths, sha256, and no `/tmp` dependency.
- `G2_RESUME`: any descent/training writes per-stage checkpoints and `--resume-from` metadata.
- `G3_TERMINAL`: stop decision records converged/cap-best/pre-plateau/failed denominators.
- `G4_RECEIVER`: archive parse-back equals the emitted payload, and the receiver changes output pixels when the section is present.

Fail-closed conditions:

- n=8-only bank, prefix-only bank, no terminal census, no pose repair payload, hidden data in receiver code, or missing per-stage checkpoint.

## Ticket OD1-T2 - Final Composition n>=32 Gate

Disposition: `QUEUED-WITH-FIRE-ORDER`.

Purpose: measure the final OD1 composition on a stratified/random n>=32 gate before any full n600.

Fire condition:

- Ticket OD1-T1 emits a byte-closed final composition.
- MAIN confirms no full-n600 job is active and claims the lane if this ticket uses scorer forwards.

Measurement contract:

- Denominator: n>=32, stratified/random; prefix-only is not bankable.
- Selection manifest: exact pair ids and selection mode.
- Axis: label as `[macOS-CPU advisory]` unless run on authority hardware.
- Rate: use full archive bytes and denominator `37,545,489`, even for subset distortion.
- Score: recompute from components; do not rely on rounded printed score.

Pass:

- Retained seg gain survives pose recovery.
- Pose term increase versus the live bank is <= `0.005`, or final S improves versus qo1 and the pose delta is explained by recomputed components.
- No component result depends on an intermediate transient.

Fail closed:

- Any missing d_seg, d_pose, byte, denominator, or selection-mode field.

## Ticket OD1-T3 - Full n600 And Authority Replay

Disposition: `QUEUED-WITH-FIRE-ORDER / MAIN_ONLY`.

Purpose: promote a passed final composition to full n600, then authority replay if it actually beats a target.

Fire condition:

- OD1-T2 passes.
- MAIN confirms scorer-slot availability and appends a lane claim.
- The archive is exact-byte frozen.

Local advisory command shape:

```bash
.venv/bin/python experiments/ddm_fz2_byteclose_and_eval.py \
  --sub-dir <exact_od1_submission_dir> \
  --out <durable_ssd_out>/od1_final_n600_cpu.json \
  --inflate-out <durable_ssd_out>/od1_final_inflate_cpu \
  --device cpu \
  --batch-size <recorded> \
  --num-threads <recorded>
```

The concrete command must be filled with real paths from OD1-T1. Do not run this package as-is.

Authority replay:

- Only after a full n600 row warrants it.
- Must use the same exact archive bytes and include archive sha256, runtime digest, hardware, command, and recomputed S.

## Ticket OD1-T4 - Fallback If Pose Recovery Fails

Disposition: `QUEUED-WITH-FIRE-ORDER`.

Purpose: avoid re-running the same wrong vehicle if Stage 2 pose recovery fails.

Pivot order:

1. Re-test frame_0 pose carriage with n>=32 stratified/random rows and a higher-k basis only if byte/S arithmetic remains positive.
2. If frame_0 fails, use AC-only frame_1 inside-cell correction; constants/DC paint are prohibited without null proof.
3. If both fail, return to per-edge PE carrier survival results from MAIN and choose the best rate/survival candidate as a separate row.
4. If no candidate improves final S, record scoped negative and do not launch n600.

Own-vehicle frontier line: `S = 0.7539807296911207 @ 357,836 B [macOS-CPU advisory]`; contest pointer remains borrowed/unmoved.
