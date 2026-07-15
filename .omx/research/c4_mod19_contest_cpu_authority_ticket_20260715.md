# AUTH-C4-MOD19-LINUX-X86_64-20260715

**Purpose:** provide the authority row for C4 after the local paired archive receipt exists.  
**Status:** `BLOCKED_BY_C1_AND_C4_LOCAL_BYTECLOSE`  
**Authority axis:** `[contest-CPU]` Linux x86_64 only.  
**Current authority result:** none. No paid dispatch was performed by C4.

## Inputs required

- Eligible checkpoint identity from `C1-WITNESS-CLEAN-STAGE-EMA-20260715`.
- Baseline mod-32 and treatment mod-19 `archive.zip` files produced from the same eligible weights.
- Exact archive SHA-256 and `|archive.zip|` for both.
- C4 local receipt with `[macOS-CPU advisory]` `d_seg`, `d_pose`, rate, and net delta.
- Receiver/inflate hashes, upstream snapshot SHA, source git SHA, seed/config, and full command custody.

## Authority execution contract

Run the pinned `upstream/evaluate.py` on Linux x86_64 against each exact archive, preserving the two
archive hashes and reporting `d_seg`, `d_pose`, byte term, total score, wall-clock, hardware, and runtime
environment independently. Recompute the paired delta from components. Do not infer Linux results from
macOS, and do not promote either row unless the exact submitted archive bytes and receiver close.

This ticket is a named follow-on only; it does not authorize Modal or any other paid provider dispatch.
