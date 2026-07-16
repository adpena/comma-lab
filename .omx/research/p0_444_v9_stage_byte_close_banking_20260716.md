# p0_444 — v9/CGauge stage byte-close banking rows (advisory)

**Date:** 2026-07-16 · **Arm:** #515 $0 P0 burn-down closer · **Pointer 0.19108 UNMOVED** (means).

## What this is

The $0 byte-close of the landed **v9_cgauge_432_coherent_arm_20260711** level-set
witness checkpoints. Checkpoints were **copied OUT** of the run dir into scratchpad
(the live run dir was never touched; the live dry-start is `c1_optimal_form_20260715`
under pid 31576, a different arm). Byte-close via
`tools/levelset_byte_close_and_eval.py --skip-parity` (the deterministic archive-bytes
half; realized parity notes below).

All rows are **`[macOS-CPU advisory]` NON-PROMOTABLE** — the exact-eval (`upstream/evaluate.py`,
contest CPU/CUDA) stays operator-GO and is NOT part of this $0 close. `params=83975`,
`n_pairs=600`, `self_orient=False` for every checkpoint. `pose=off` in the byte-close: the
level-set witness carries pose in per-(pair,frame) codes/texture, so these bytes are the
**seg-carrier archive only** (a stored pose sidecar would add COUNTED bytes the scorer never
reads → does not lower realized d_pose; pose rides the joint-descent codes/texture per the
banked R1 dxi result).

## Measured banking rows (deterministic; byte-close)

| checkpoint | 0.bin (B) | archive.zip (B) | rate | rate_term |
|---|---|---|---|---|
| `levelset_witness_ema_mlx.npz` (final EMA) | 66514 | 65797 | 0.001752 | 0.0438 |
| `levelset_witness_ema_BEST.npz` (best EMA) | 64376 | 63659 | 0.001696 | 0.0424 |
| `levelset_ckpt_stageOctave1_ep251.npz` (stage ckpt) | 65172 | 64461 | 0.001717 | 0.0429 |

`bank=FREE(rule118)` for all. Archive.zip ≈ 64–66 KB → rate_term ≈ 0.043 (the seg-carrier
archive is tiny; the full witness score adds d_seg + the pose contribution + any additional
carrier sections). BEST < final on bytes (63.7KB vs 65.8KB) — the EMA-BEST snapshot is the
leanest archive of the three.

## Realized d_seg/d_pose parity — compute-pending (NOT launch-blocked)

The advisory realized-through-R d_seg/d_pose parity requires the full n600 CPU path:
inflate all 1200 frames (→ 3.66 GB raw) then a frozen SegNet+PoseNet forward over 600 pairs.
That forward is a >10-min single-thread CPU job. It was attempted 3× (fg 2-min cap; background
SIGURG at ~5 min; `nohup`+`disown` detach — `setsid` is absent on macOS — also did not survive
the turn boundary). The inflate stage completed each time (`full_output_ok=True`,
raw_bytes=3662409600); the parity forward is the stage that exceeds the harness-sustainable
compute window. This is a **wall-clock/harness constraint, not a launch dependency and not a $0
blocker on the bytes** — the banking bytes above are complete and deterministic. The realized
parity + the Modal exact-eval are the operator-GO / longer-compute sub-parts.

## Provenance

- Source run dir: `experiments/results/v9_cgauge_432_coherent_arm_20260711/` (copied out; untouched).
- Tool: `tools/levelset_byte_close_and_eval.py` (`--skip-parity`, gt cache `mlx_fleet_gt_cache`).
- Bytes are reproducible from the checkpoint alone (independent of gt).
