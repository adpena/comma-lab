# ddm_gov3 routing directive — two owed governor mechanisms (MAIN, 2026-09-05 00:55Z; source: gov2 f6f3f03f8 + de7b4229c, gs3 addendum 29)

Tokens: `[no-triality] [p0-ledger-ok]`. Owner: MAIN routes; build owner = the next governor arm (codex when quota returns Sep 7 06:39, else Opus).
Context: the memory guard's SIGSTOP actuator was RETIRED on measurement (paused ng4 ×3 and mc1 ×1, helped 0×; a stopped process becomes the
swap victim, so the clear condition became unreachable). The live watchdog runs report-only. These two items are the permanent replacement.

## ITEM 1 — cooperative pause protocol (the actuator that frees resident bytes without losing work)
The watchdog sends SIGUSR1 to the selected governed job (growth-ranked target, never the oldest cell); the governed trainer (`experiments/ddm_qbr1_born_fairform_burn_prep.py`
run-config path, `fpc3` chunked trainer, and any long-lived governed job) installs a handler that finishes the current step, writes its per-stage
checkpoint + a `cooperative_pause.v1` receipt (config sha, step, checkpoint path, resume argv), and EXITS rc=75; `tools/cell_queue_driver.py` re-admits
the job through `cell_admission` when pressure clears and resumes from the receipt. Tests: handler under a fake SIGUSR1 at a step boundary; receipt
schema; driver re-admission; an end-to-end drill on a bounded smoke. Acceptance: a paused cell resumes bit-identically (same milestone shas as an
unpaused run) — the deterministic-resume non-negotiable already requires this.

## ITEM 2 — launcher refuses (or restarts) a governed instance older than the tool it runs
`tools/launch_detached_process.py` / `tools/cell_queue_driver.py`: for long-lived governed processes (watchdog, dashboard, waiters, pollers) record the
tool file's mtime/sha at launch in the manifest; a `reconcile-stale` subcommand (and the digest's live-cells SENSE) flags instances whose tool file
changed after launch; an opt-in `--restart-on-stale` restarts them through the same governed path. The watchdog's self-retire (de7b4229c) is the
per-tool instance of this; this item generalizes it. Tests: stale detection from a touched file; the restart path; the digest line.
Equations leg (`tac.canonical_equations`): none — apparatus.
