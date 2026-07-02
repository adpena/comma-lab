# $0 PROBE: L28 zero-byte channel offset on the WITNESS — NEGATIVE (drift-fix 2026-07-01)

**Task (drift-fix part 3):** re-measure the L28 / PR98 canonical zero-byte
decode-side channel offset (`(f0,R,−1), (f0,B,−1), (f1,G,−1)`, clamp[0,255])
on the level-set WITNESS `.raw` output at n600 through R — the constants are
ANCESTOR-tuned (HNeRV renderer bias), so RE-MEASURE per the ancestor rule; keep
ONLY if realized d_seg/d_pose drops on the witness, else record the negative.

## Method (NO-FAKE, n600, frozen CPU-torch authority)

- Witness: `experiments/results/levelset_packet_20260701T200034Z/inflated/0.raw`
  (1200 frames = 600 pairs, 874×1164×3).
- Scorer: the REAL upstream `DistortionNet` on CPU (SegNet EfficientNet-B2 argmax
  + PoseNet FastViT-T12), the d_seg/d_pose authority; NEVER MPS.
- GT: `gt_n600.npz` cached frozen argmax (`lstars`) + poses (`gt_poses`).
- **NO-FAKE self-check (pair 0): my comp-side harness reproduces the cached GT
  EXACTLY — seg argmax disagree 0.0, pose MSE 0.0** (proves harness == evaluate.py).
- Full result: `experiments/results/l28_witness_probe_n600/result.json`.

## Result (n600) — [macOS-CPU advisory], NOT a byte-closed score

| term | baseline | L28 | delta |
|---|---|---|---|
| d_seg | 0.00404832 | 0.00404838 | **+5.93e-8 (RAISES — no drop)** |
| d_pose | 98.527 | 97.913 | −0.614 (POSE-BLIND regime) |

## Verdict: **DROP** (ancestor-tuned L28 does not transfer to the witness)

1. **d_seg (the witness's binding controllable job): L28 does NOT lower it** — it
   RAISES d_seg by +5.9e-8. The only d_seg-relevant offset is the `(f1,G,−1)`
   term (SegNet reads the last frame only); the ancestor-tuned green nudge flips
   a few pixels the WRONG way. Exactly the ancestor-rule prediction: HNeRV-renderer-
   bias constants do not transfer to the level-set task-space witness.
2. **The −0.614 d_pose "improvement" is NOT a valid keep basis.** This witness is
   **pose-blind by design (w_pose=0)** → d_pose ≈ 98.5 (vs the solved-pose target
   ~3.4e-5). At d_pose≈98 the √(10·d_pose) term is ~31, so the naive distortion-S
   "KEEP" the script prints is dominated by a MEANINGLESS pose term. Witness pose
   rides the separate stored-twist FiLM sidecar (`pose-solved-screw-twist-dual-use`),
   NOT decode-side channel offsets; the L28 pose constants target the abandoned
   HNeRV pose-through-luma regime.

**Decision: DROP L28 for the witness** (do not bolt it onto the level-set witness
archive). Reactivation: re-measure on a POSE-TRAINED witness (w_pose>0, d_pose near
3.4e-5) where the pose regime is meaningful — but even then the d_seg result already
shows the ancestor green offset does not lower d_seg.

## System-intelligence wire-in

Registered as canonical equation
`l28_channel_offset_does_not_transfer_to_levelset_witness_v1` (measured NEGATIVE
anchor; consumers = `tac.codec.pr98_channel_balance_zero_byte_bolt_on` + the
levelset trainer) so the non-transfer is not naively re-tried. Sister to the
ancestor rule (`ancestor-vehicle-findings-are-lessons-not-transferable`).
