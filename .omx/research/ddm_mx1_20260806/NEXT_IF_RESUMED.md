# ddm_mx1 next if resumed

## Current State

The code path is implemented, but local MLX is unavailable:

```text
[metal::load_device] No Metal device available. This typically occurs in headless, sandboxed, or virtualized macOS sessions where the GPU is not accessible.
```

Do not treat MLX parity or MLX smoke as measured in this sandbox. Torch CPU smoke passed only as a lifted-reference sanity check.

## Immediate MAIN Sequence

1. Run parity on a usable MLX/Metal host:

```bash
.venv/bin/python experiments/ddm_mx1_pr130_semantic_renderer.py --mode mlx-parity --device gpu --pairs 2 --steps 2 --input-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/tq1c_seg_cache.pt --target-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt --init /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_12k.pt --run-dir /Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch/parity_metal --out /Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch/parity_metal/result.json
```

Required parity fields before training claim: `raw_frame_max_abs`, `seg_argmax_equal`, `seg_argmax_diff_count`, `loss_abs_delta`, `token_batch_shape`, `scorer_batch_shape`, adapter identity, and matching loss phase. Because #855 found systematic default-adapter argmax flips, `seg_argmax_diff_count` must be reported even if it is nonzero.

Gradient parity is not implemented in this mode. If the resumed run wants to make a training-mechanism parity claim rather than a forward/loss-only research-signal claim, add one real-input torch-CPU vs MLX gradient-parity check with per-tensor max relative differences before banking the trainer.

2. If parity is admissible, run n32:

```bash
.venv/bin/python experiments/ddm_mx1_pr130_semantic_renderer.py --mode mlx-train --device gpu --pairs 32 --steps 6000 --lr 2e-07 --ce-fraction 0.0 --softplus-fraction -999.0 --bits 4 --seed 20260806 --checkpoint-every 250 --eval-every 250 --input-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/tq1c_seg_cache.pt --target-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt --init /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_12k.pt --run-dir /Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch/n32_metal --out /Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch/n32_metal/result.json
```

3. Resume n32 if interrupted:

```bash
.venv/bin/python experiments/ddm_mx1_pr130_semantic_renderer.py --mode mlx-train --device gpu --pairs 32 --steps 6000 --lr 2e-07 --ce-fraction 0.0 --softplus-fraction -999.0 --bits 4 --seed 20260806 --checkpoint-every 250 --eval-every 250 --input-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/tq1c_seg_cache.pt --target-cache /Volumes/VertigoDataTier/pact/ddm_hb1_20260806/inputs/gt_seg_cache.pt --init /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo/artifacts/checkpoints/semantic_renderer_w96_b4_qat4_12k.pt --run-dir /Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch/n32_metal --resume-from /Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch/n32_metal/mlx.latest.npz --out /Volumes/VertigoDataTier/pact/ddm_mx1_20260806/launch/n32_metal/result.json
```

4. Only after n32 has non-degenerate telemetry, fire n120 using `LAUNCH_TICKET.md`.

## Harvest Requirements

- Record result JSON bytes and sha256.
- Record every stage checkpoint bytes and sha256, especially `mlx.latest.npz`.
- Keep axis labels explicit: `[macOS-MLX research-signal]` for trainer telemetry, frozen CPU-torch SegNet through exact R for d_seg checks.
- Do not bank a prefix result. Do not run n600 from this arm unless et4 hands over the slot.
- If a new byte-closed archive is eventually produced elsewhere from this receiver, evaluate with `upstream/evaluate.py` and recompute S from components; do not use rounded final-score text.

## Follow-on Disposition

- MLX parity: QUEUED-WITH-A-FIRE-ORDER above.
- n32 Row-1: QUEUED-WITH-A-FIRE-ORDER above.
- n120 Row-1: QUEUED-WITH-A-FIRE-ORDER in `LAUNCH_TICKET.md` after n32 telemetry.
- Full n600 scorer: HELD, owner et4.

Own-vehicle frontier remains `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`; no mx1 exact row exists yet.
