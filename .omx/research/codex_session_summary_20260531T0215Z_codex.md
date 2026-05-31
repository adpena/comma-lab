# Codex Session Summary

Date: 2026-05-31T02:15Z
Agent: Codex

## Landed Work

- Extended the P18/P19/P11/P15 queue surface so it can run local CPU component checks, local MLX scorer responses, scorer-response dataset harvest, exact-ready bridge regeneration, and retention as one queue-owned loop.
- Ran the strict scorer-region cascade queue against the current CPU frontier closure. The receiver patch changes inflated output, but the full local CPU advisory is slightly worse, so no exact-auth dispatch is justified.
- Preserved the positive MLX slice as scorer-response dataset signal for acquisition rather than promotion authority.
- Fixed stale receiver proof copying, MLX response postcondition drift, missing MLX retention, canonical MLX cache retention detection, and retention output overwrite ordering.
- Added `--latents-npy` to the PR95 MLX package path to keep MLX long-training checkpoints portable through NumPy custody.
- Executed queue-owned artifact retention and reclaimed rebuildable raw/cache artifacts under `/Volumes/VertigoDataTier`.

## Current Frontier

- `[contest-CPU Linux x86_64]`: `0.19198533626623068`, archive `b7106c9bdbb8...`
- `[contest-CUDA T4]`: `0.20533002902019143`, archive `9cb989cef519...`
- No frontier change in this session.

## Durable Blockers

- The current 12-pair frame1 region waterfill receiver patch is not exact-ready because local full CPU advisory is worse and exact auth has not been claimed or run.
- MLX remains useful for fast acquisition but is not authority for rank, kill, promotion, or dispatch.
- The PR95 32-frame checkpoint is not a byte-closed full-archive candidate; it is a timing/telemetry artifact.

## Next Work

Run the next acquisition wave as grouped, queue-owned candidates: expand P19 PoseNet-null and P18 SegNet-region waterfill beyond the 12-pair leaf; add grouped RGB/YUV delta and region-mode choices; compose with P11 selector codec and P15 repack; use MLX for broad acquisition; require full local CPU advisory before exact auth. In parallel, move PR95 MLX from smokes toward full 600-latent bounded training with `.npy` export and byte-closed archive smoke.
