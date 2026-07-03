# Crash signal-loss recovery sweep — 2026-07-03

**Operator directive:** "ensure no signal loss and request and retask anything that was running and
died such that we got no signal — this is a huge forgetfulness risk."

**The crash:** memory-pressure HANG ~2026-07-02 20:57 PDT (≈03:57 UTC Jul 3). No JetsamEvent or
panic report exists for 07-02 (the FS daemons apfsd/fileproviderd were flagged for CPU-resource at
20:57 = compressor/swap thrash → unresponsive → reboot). Cause: per-process-blind memory guard let
concurrent jobs sum past 128 GB (now fixed by the system governor + admission gate, enforcing as of
`56147e797`). Recurring history: jetsam cascades on Jun 26 (10+ in 20 min), Jun 28, Jun 30, Jul 1.

**The concurrent set live in the crash window (reconstructed from macOS reports + file mtimes; NO
black-box existed pre-crash — that gap is now closed):** R1 `descent_ev1` training (~67 GB) + the
#238 byte-close (`levelset_packet`) + a `bsdtar` archive op (20:40) + the earlier R1 `pose_ft`.

## Recovery classification (every Jul-2 run)

| run / job | last-active | signal status | action |
|---|---|---|---|
| R1 `descent_ev1` (store-nothing pose descent) | 19:49 | crash-killed, no ckpt | **RELAUNCHED** (task #245, pid ~19940) |
| R1 `pose_ft` (17 log lines, no ckpt) | 19:03 | **SUPERSEDED** — descent_ev1 re-obtains it | none |
| **#238 byte-close pose measurement** (`levelset_packet_20260703T005151Z`) | 19:52 | **LOST** — built archive.zip + inflated `0.raw` but DIED before writing the eval result JSON | **RE-TASK #238** (queued behind R1 — the decode-pose should be measured on the co-adapted witness anyway; run single-file / governor-gated, never re-create the concurrent crash pattern) |
| `witness_20260702T210653Z` (15 log lines) | 16:17 | KNOWN #205 OOM — signal = the verdict-batch OOM diagnosis (banked; fix built `240`) | none |
| `keyframe_ego_residual_coding_n600` / `keyframe_pose_sufficiency_ladder_n48_smoke` / `keyframe_pose_sufficiency_ladder_n600` (`ladder_rows.jsonl`+g1–g5) / `keyframe_vcm_rate_384x512` / `n205_oom_probe` | 16:31–17:48 | **RECOVERED** — result/ladder JSONs on disk | none |
| `witness_{230321Z,230325Z,230818Z,233208Z,233216Z}`, `witness_capstone_204138Z`, post-crash `witness_{020855Z,021031Z}` | 18:03–21:10 | **REFUSED / dry-run** — `launch.sh` only, never spawned (OOM saga / memory-preflight refusal). The refusal IS the signal. | none |
| research / triality / review / governor work | through crash window | **COMMITTED to git** (compression v2, #205 risk register, triality reconcile, store-nothing wiring, #205 review, OOM fix, governor) | none |

## Verdict

**Exactly one genuine lost-signal casualty: #238's byte-close pose measurement.** Re-tasked. Everything
else recovered / superseded / refused / committed. The git-commit-early + result-file disciplines held.

## Anti-forgetfulness structural note

- The **memory black-box now exists** (`.omx/state/memory_blackbox.jsonl`) → any FUTURE crash leaves a
  recoverable per-2s trajectory readable via `tools/memory_blackbox.py --last-crash`. This manual sweep
  was the one-time recovery for the **pre-black-box** crash.
- The **admission gate (enforcing)** structurally prevents the concurrent-sum crash that caused this.
- Possible follow-up (low priority, governor should prevent recurrence): a reusable
  `crash_recovery_sweep` that auto-classifies run dirs (has-result / superseded / refused / LOST) after
  any detected reboot.
