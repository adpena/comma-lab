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

## ITEM 3 — Metal occupancy is not modeled: a governed job that is not a `run-config` cell can hold the GPU unseen (MEASURED 2026-09-05 14:46Z)
`tools/cell_admission.py cells` reported "8 live governed jobs (0 training cells)" while cl2's HPAC trainer (`tools/train_ddm_cl1_hpac_capacity.py`, a Metal/MPS
training run launched through the launcher, declared peak 1.76 GiB RSS, measured system-availability delta 15.56 GiB) was mid-epoch on the GPU. `is_cell` recognizes
only the `run-config` argv shape, so the two-Metal-cells refusal (gov2 item 6) would NOT have fired had md3's 49.6 GiB cell been admitted at that moment — md3
queued behind cl2 by its own discipline (`queued_behind_cl2_ladder` claim row), not by the governor's refusal. Cure: classify Metal OCCUPANCY independently of the
cell shape — a governed job whose measured-peak ledger row (or a launch flag `--metal`) records a system-availability delta ≫ tree RSS is a Metal occupant; the
throughput/concurrency leg counts occupants, not cells; the digest's live-cells section names them. Tests: cl2's trainer shape + a `run-config` cell must both
count as occupants; a CPU pricer must not. Owner: the next governor arm (Opus).

## ITEM 4 — fire waiters and queue placeholders (MEASURED 2026-09-05 15:34–15:41Z, two refusals in a row on md3's cell)
(a) md3's fire waiter polled only for "Metal free 3/3" and EXITED rc=2 on the driver's admission REFUSE (ane2's five idle launches reserved 42 GiB of declared
peaks at ~1.4 GiB real RSS). A fire waiter must poll the driver's DRY-RUN readiness (admission + storage + seal), not a single device predicate, and must
RETRY on refusal until a wall cap — MAIN's `wait_admission_then_fire_md3.sh` is the pattern; fold it into `tools/cell_queue_driver.py run --wait-until-ready`.
(b) On the retry the driver refused itself: `REFUSING_DISPATCH: active claim(s) already exist for lane_id=…` — md3's own QUEUED placeholder claim
(`queued_behind_cl2_ladder`, same job id) blocked the driver's live claim at fire time. The driver must ADOPT a queued placeholder for the same lane+job id
(append the terminal `queued→active` transition itself) instead of refusing; a placeholder is the queue's own state, not a foreign dispatch. (c) The launcher
keeps a finished launch's DECLARED peak reserved while its supervisor lingers; release the reservation (or mark the manifest terminal) when the child exits.
Tests: waiter retries through a simulated refusal; driver adopts a same-job placeholder; supervisor exit releases the reservation. Owner: next governor arm.

**ITEM 4(d) — the driver refuses its OWN claim (MEASURED 15:42:19Z, attempts 4–5):** `cell_queue_driver run` files `active_eval` rows on BOTH lanes (agent = the queue spec's `--agent`), then a later stage re-runs the single-flight check and refuses on "active claim(s) already exist for lane_id=<scorer lane>" — the claim it just filed. Every retry files two more orphaned active rows; MAIN killed the loop and closed 6 rows by hand. Cure: the driver must recognize its own claim (same lane+job+agent within the run) as its lease, not a conflict; add a test that a full `run` on a fresh queue files exactly one active row per lane and fires.

### ITEM 3 — SECOND INSTANCE (MEASURED 2026-09-05 16:12:53Z): a "2.4 GiB" Metal trainer beside a live 49.6 GiB Metal cell
cl3's λ=2.0 HPAC-prior trainer (`device mps`, declared peak 2.4 GiB from cl2's RSS-based measured-peak ledger) was ADMITTED by the guard
("admission OK (governed)") beside md3's live Metal cell. Within 8 minutes the watchdog logged a CRITICAL `memory_pressure` alarm: compressor
42.37 GiB (33.1%), growing 5.20 GiB/s; the trainer died 20 s later (rc=143; actor under attribution — the watchdog is report-only and did not
act). Root cause is ITEM 3 verbatim: Metal working sets are not RSS, so an RSS-declared peak of 2.4 GiB says nothing about a Metal trainer's
GPU-side footprint, and two Metal trainers on one 128 GB machine reproduce the 2026-09-04 near-OOM shape. Cure (binding until ITEM 3 lands):
the admission guard must treat ANY `device mps`/Metal trainer as a Metal occupant — one Metal occupant at a time unless a measured N≥3 window
proves two fit — and the measured-peak ledger must record a Metal footprint column (system-availability delta, as gov2 measured for the cell:
49.572 GiB at 1.75 GiB RSS) rather than RSS alone. MAIN's cl3 charter sentence "a 2.4 GiB trainer is admitted beside it" was the same
RSS-for-Metal error and is corrected in the arm's instructions.
**ATTRIBUTION CORRECTION (16:30Z, same day, lm1 law — ask the actor):** the arm's own tree shows the λ=2.0 rung was ABORTED by the arm for a
filesystem reason (`aborted/lambda_2p0_exfat_partial_20260905T1612Z` — APDataStore is ExFAT) and relaunched on Vertigo; the rc=143 was the arm's
SIGTERM, not memory. The CRITICAL compressor alarm (42.4 GiB, 5.2 GiB/s at 16:12:53Z) is COINCIDENT and its cause is NOT attributed (the watchdog's
top grower was md3's cell itself). MEASURED afterwards (16:29Z, 45 s window) with cl3's relaunched Metal trainer LIVE beside md3's cell: compressor
flat 1.8 GiB, free 17.4→28.1 GiB, md3 at 29 steps/min (serial baseline 28) — the first N=1 sample that a small Metal trainer and one cell fit.
What STANDS from this instance: the guard admits Metal trainers on RSS alone (the blind spot is real and unmeasured); the watchdog is the only
tripwire. Governance until ITEM 3 lands: a Metal trainer may run beside one cell ONLY with the watchdog live; any memory_pressure WARN while it
runs → stop the trainer (epoch-checkpointed), resume after the cell's receipt; each such window is a sample for the N≥3 admission table.
