# C1-WITNESS-CLEAN-STAGE-EMA-20260715

**Purpose:** unblock C4 by producing one eligible, byte-closeable C1/V9 witness checkpoint.  
**Status:** `PENDING`  
**Cost authority:** this ticket does not authorize a launch or paid dispatch. It harvests the existing
C1 run when available, or returns to the operator/main launch owner if no run exists.  
**Consumer:** `lane_c4_mod19_rate_byteclose_20260715`.

## Acceptance contract

The producer must hand C4 a content-addressed run directory containing:

1. The actual compiled `launch.sh`/DSL manifest and the named run identity.
2. Either a clean `levelset_best.json` pointing at the exact preserved EMA shadow that produced its
   verdict, or the most-recent complete stage-EMA checkpoint if the run has no clean best.
3. Checkpoint SHA-256, byte size, epoch/stage, git SHA, upstream snapshot SHA, seed, full config, and
   the n600/real-input custody needed by `tools/levelset_byte_close_and_eval.py`.
4. A reason the artifact is clean and complete. A smoke, synthetic fixture, partial write, live-only
   tensor dump, or checkpoint with `git_sha=unknown`/`upstream_snapshot_sha256=unknown` is ineligible.
5. A stable durable path on the SSD tier or the isolated handoff surface; `/tmp` is forbidden.

Once this contract is satisfied, C4 byte-closes the same eligible weights at mod-32 and mod-19 and
records exact archive sizes plus `[macOS-CPU advisory]` distortion. This ticket itself makes no score,
rate, or `d_seg` claim.
