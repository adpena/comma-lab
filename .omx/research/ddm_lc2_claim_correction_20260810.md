# Correction: a false claim row I wrote on `lane_ddm_lc2_cpu_20260810_contest_cuda`

**Date:** 2026-08-10 · **Author:** MAIN · **Status:** correction pending append to
`.omx/state/active_lane_dispatch_claims.md` (blocked only by single-flight while the
CPU axis is live; append at CPU terminalization).

## The false statement

I appended a claim row with `status=stale_superseded_never_spawned` whose notes read:

> "Proof it never spawned: no modal_auth_eval_spawn.json in
> experiments/results/modal_auth_eval/ddm_lc2_cpu_20260810_paired_modal_auth_20260810T160324Z_cuda/
> (only the pre-spawn local_request json); zero provider spend."

**That is false.** The job spawned. Evidence now on disk in that exact directory:

- `modal_auth_eval_spawn.json`
- `modal_call_id.txt` → `fc-01KZP6X99F259EKV0XRBVG6963`

The check was real but run TOO EARLY — before the wrapper wrote its spawn record.
I reported an absence as a proof of non-existence.

## Root cause

`tools/launch_detached_process.py` returned **pid 21915**, which is the *wrapper*.
The dispatcher itself ran as **grandchild pid 21916**. `start_new_session` detachment
re-parents that grandchild to PID 1 *by design*, so `kill 21915` did not stop it. The
dispatcher survived, completed the CUDA image build, spawned CUDA, and then spawned CPU.

I then reasoned from "I killed it" to "it never spawned" and wrote that chain into a
custody record as proof.

This is the operator kill rule in CLAUDE.md, learned the expensive way: *kill the child
pid from the child_pidfile* — the pid the launcher hands back is not always the pid that
must be signalled.

## What is actually true

| axis | call_id | note |
|---|---|---|
| contest_cuda | `fc-01KZP6X99F259EKV0XRBVG6963` | REAL, third redundant CUDA run on sha `f154f0ab`, real Modal spend |
| contest_cpu | `fc-01KZP70MKR3Z5B0XZG5BZK77GM` | REAL, **this is the axis the session was trying to buy** |

The CUDA run is redundant with two prior measurements of the same bytes —
`fc-01KZP3R7QWCTJQWGYHGBNJM4GQ` and `fc-01KZP50CVABM8P9VGXMF8TK1PS`, both
`S = 0.16959899569230852`, bit-identical. Its cost is waste attributable to my
ineffective kill, not to the dispatcher.

## The guard was right; I was wrong

Modal single-flight refused my two manual CPU retry launches (rc=5). I labelled the
blocking rows "phantom." **They were live.** The guard was reading true state and
correctly preventing a concurrent Modal job. Both retries were redundant — the CPU axis
was already spawning from the paired dispatcher I had wrongly believed dead.

## Owed

1. Append the corrected claims row when the CPU lane terminalizes (no `--override`:
   a record correction does not justify self-authorizing a single-flight bypass).
2. Attribute the redundant CUDA spend against the #381 Modal envelope.
3. Harvest `fc-01KZP70MKR3Z5B0XZG5BZK77GM` — the contest-CPU row that makes the lc2
   exact row promotion-eligible and settles #998 (device as a score lever on identical
   bytes).
