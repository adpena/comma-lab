# Public HNeRV Frontier Deconstruction - PR100/101/103/PR106 Path

date: 2026-05-15
research_only: true
score_claim: false
dispatch_attempted: false
promotion_eligible: false
ready_for_exact_eval_dispatch: false

## Source Signals

Live public refresh on 2026-05-15 found no leaderboard movement below PR101:

- PR101 `hnerv_ft_microcodec`: displayed public score `0.193`.
- PR103 `hnerv_lc_ac`: displayed public score `0.195`.
- PR102 `hnerv_lc_v2_scale095_rplus1`: displayed public score `0.195`.
- PR107 remains closed with public CUDA eval rounded `0.23`; no new technical comments after the 2026-05-05 maintainer note.
- PR108 was closed on 2026-05-11 under the new post-deadline guideline: future submissions must be competitive with #1 or innovative beyond established leaderboard tricks.

Post-deadline comments preserve two load-bearing lessons:

1. PR100's per-pair latent-correction sidecar was acknowledged by the maintainer as a novel technique after it propagated into PR102/PR103-style descendants.
2. CPU and CUDA scoring are not interchangeable. PR101/PR103/HNeRV variants improve materially on CPU relative to T4 CUDA, while PR106 PacketIR/format0C is CUDA-favorable on PoseNet. Any frontier claim needs paired CPU/CUDA custody.

## Archive Mechanism Map

- PR100/PR102: same archive bytes (`178981` bytes, one `0.bin` member). Runtime behavior differs: PR102 changes constants (`DELTA_SCALE=0.0095` plus frame-0 red `+1.0`). Grammar is length-prefixed decoder brotli, fp16 scales, latent brotli, latent-correction brotli.
- PR101: compact microcodec baseline. One `x` member; fixed slices roughly `decoder_blob=162164`, `latent_blob=15387`, `sidecar_blob=607`. CPU-frontier relevant, but paired CUDA is not close.
- PR103: lossless repack / arithmetic coding primitive. It is useful after a component-moving packet, but measured retunes are byte-scale only and cannot alone close the component gap.
- PR106/format0C: not CPU-frontier, but its paired eval shows CUDA PoseNet improvement while SegNet barely worsens. This is the orthogonal repair signal to mine.

## Measured Constraint

PR101/FEC6 paired XRay:

- `[contest-CPU]`: `0.192051316881`.
- `[contest-CUDA]`: `0.226210021693`.
- CUDA-minus-CPU delta: `0.034158704812`.
- Dominant delta: PoseNet.
- Byte-equivalent CUDA gap to `<0.192`: about `51378` bytes.

Therefore rate-only PR101/PR103 polishing is not frontier work until a component-moving repair exists.

## Actionable Build Direction

Next plausible `<0.192` hook:

```bash
.venv/bin/python tools/build_pr106_cuda_latent_correction_probe.py \
  --pair-hitlist experiments/results/.../hardpair_hitlist.json \
  --paired-axis-artifact experiments/results/.../paired_axis_delta.md \
  --source-archive experiments/results/.../archive.zip \
  --output-dir experiments/results/pr106_cuda_latent_correction_probe
```

The first implementation may be a strict planner if archive mutation is too risky. It must remain false-authority-only until it emits a byte-closed packet and passes paired exact CPU/CUDA auth eval.

## Immediate Code Landed From This Finding

- `tools/xray_hardpair_hitlist.py` converts pair-component XRay rows plus paired CPU/CUDA axis deltas into deterministic hard-pair repair priorities.
- `tools/operator_briefing.py` and `tools/all_lanes_preflight.py` now surface and enforce the hitlist tool as a visible false-authority XRay primitive.
- A follow-on worker is building the PR106 CUDA latent-correction probe planner with disjoint file ownership.
