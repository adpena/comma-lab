# ddm_mx2 MAIN Launch Ticket

## Status

Do not dispatch yet. This is a blocked MAIN ticket, not a launch receipt.

Blocking gates:

- et4 owns the full n600 scorer slot.
- Local MLX execution is blocked by Metal device access.
- The vendored PR130 pose trainer has `latest`/`best` checkpoint writes, but no true `--resume-from` load path. The no-nonresumable-launch rule requires a resume wrapper or a proven resume patch before firing a long run.
- The source trainer's `--master-cache` is guarded by `source_checkpoint == --master-checkpoint`; fitting against our tq1c or mx1-rendered surface needs an adapter that writes/accepts our master surface without falsely labeling it as the PR130 semantic checkpoint.

## Intended Vehicle

Target: row-2 pose carrier fit on our current/mx1 surface, not an external PR130 score claim.

Fit surfaces, in order:

1. Current own-vehicle tq1c frame-1 surface from `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes`
2. mx1 row-1 renderer surface when MAIN produces it

The PR130 source semantic checkpoint may be used only as a source-shape reference or initialization context; it is not our own vehicle.

## Source CLI Shape

The vendored full trainer exposes this real CLI shape:

```bash
PYTHONPATH=src/tac/pr130_lift/pose/lifted .venv/bin/python src/tac/pr130_lift/pose/lifted/train_pose_carrier_full.py \
  --challenge-root upstream \
  --target-cache /Volumes/VertigoDataTier/pact/ddm_mx2_20260806/inputs/gt_pose_cache_600.pt \
  --master-checkpoint /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_fixedtau05_tail6k_lr2e7.pt \
  --init-carrier /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints/archive_carrier_int6_stable_s8k.pt \
  --master-cache /Volumes/VertigoDataTier/pact/ddm_mx2_20260806/master_cache/OUR_SURFACE_MASTERS.pt \
  --reuse-master-cache \
  --steps 20000 \
  --batch-size 12 \
  --eval-batch-size 12 \
  --render-batch-size 4 \
  --eval-every 1000 \
  --lr-basis 0.003 \
  --lr-coeff 0.03 \
  --basis-freeze-fraction 0.30 \
  --basis-train-until-fraction 1.0 \
  --qat-fraction 0.65 \
  --hard-mining-power 0.0 \
  --hard-mining-max 8.0 \
  --basis-bits 8 \
  --coeff-bits 12 \
  --amplitude 64.0 \
  --carrier-base gray \
  --seed 20260715 \
  --device cuda \
  --out /Volumes/VertigoDataTier/pact/ddm_mx2_20260806/main_full_pose_carrier/result.json \
  --save /Volumes/VertigoDataTier/pact/ddm_mx2_20260806/main_full_pose_carrier/archive_carrier_mx2.pt
```

This command is not fireable as written until the four blocking gates above are cleared. The `gt_pose_cache_600.pt` and `OUR_SURFACE_MASTERS.pt` names are the required durable target locations for the adapter outputs, not verified existing inputs.

## Required Adapter Before Fire

Add a small mx2-owned adapter that:

- materializes the target cache with keys `seg` and `pose`, shape `(600, ...)` and `(600, 6)`, from the authoritative scorer target source;
- materializes our selected master surface as uint8 tensor shape `(600, 3, 874, 1164)`;
- records source archive/runtime/component hashes for that master surface;
- either patches the trainer to load `--resume-from` or wraps it in stage slices that preserve restartable optimizer/model/RNG state, not just result tensors;
- keeps per-stage checkpoints under distinct filenames in `/Volumes/VertigoDataTier/pact/ddm_mx2_20260806/main_full_pose_carrier/`.

## Verdict Protocol

- n32/n120 selection: stratified random only; never prefix.
- tiny-n PoseNet forwards: allowed only for parity/smoke, not a verdict.
- full n600: only after et4 releases the scorer slot or explicitly assigns this lane.
- authority: frozen CPU-torch scorer through exact R for component distances; `upstream/evaluate.py` on exact archive bytes for score.
- target need: pose term should move toward PR130-scale `d_pose ~2e-5` while preserving our row surface; no claim from projection alone.

## CPR1 Pack Gate

After a real mx2 carrier fit exists, run:

```bash
PYTHONPATH=src/tac/pr130_lift/pose/lifted .venv/bin/python src/tac/pr130_lift/pose/lifted/pack_semantic_pose.py \
  --semantic /Volumes/VertigoDataTier/pact/ddm_mx2_20260806/main_full_pose_carrier/OUR_SEMANTIC_PAYLOAD.pt \
  --carrier /Volumes/VertigoDataTier/pact/ddm_mx2_20260806/main_full_pose_carrier/archive_carrier_mx2.pt \
  --carrier-bits 8 \
  --coeff-bits 12 \
  --out /Volumes/VertigoDataTier/pact/ddm_mx2_20260806/main_full_pose_carrier/semantic_pose_payload.bin \
  --report /Volumes/VertigoDataTier/pact/ddm_mx2_20260806/main_full_pose_carrier/pack_report.json
```

Then run CPR1 only if the produced carrier section is exactly PR130 legacy-carrier-shaped or the pack report proves the PR130 carrier layout. Otherwise classify as not applicable, as in `REPACK_RACE.md`.
