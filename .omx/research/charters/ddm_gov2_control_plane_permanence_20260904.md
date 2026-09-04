# CHARTER ddm_gov2 — CONTROL-PLANE PERMANENCE: the governor complete by construction, autonomous end-to-end (operator 2026-09-04: "much of this requires a more permanent solution")

Tokens: `[no-triality] [p0-ledger-ok]`. Owner: Opus arm. Spawned 2026-09-04 ~22:55Z. Sister arm: **hv1** (harvest→pointer autopilot) — its files
(`tools/modal_harvest_poller.py`, `tools/fire_modal_auth_eval.py`, pointer/report refresh, claim-row writer, the new pointer-move packet) are OFF-LIMITS
to you; yours are off-limits to it. dk1 (disk reclaim) is live — do not touch its files.

## THE DEFECT CLASS (measured this session, all by RUNNING — none by inspection)
1. Live-cell discovery MISSED a live 45 GiB cell: registry-first (launches can forget to register — the sealed-source launcher copy did),
   walk fallback MEASURED >120 s per poll and the Vertigo seed timed out at 3600 s. The guard admitted a second big cell on a live basis
   that reserves current footprint, not the cell's future growth. Backfilled by hand.
2. A sole cell was gated by N=2 contention evidence (`max(2, live_count)` with contending=1): ng4 refused 90 min with the Metal free.
3. Declared peaks are HAND-TYPED (`--measured-peak-rss-gib 2.396` was a fiction; ~45 GiB of system availability per Metal cell; RSS is blind
   to Metal). The old inline `vm_stat` fire scripts admitted two 40 GiB cells concurrently → 17:11Z the VM compressor hit space-shortage and
   jetsam killed background daemons (near-OOM, operator-observed).
4. Bespoke fire scripts per cell (ng2/ng3/ng4), each re-implementing admission; MAIN in the loop for every fire; a 4-hour MAIN absence
   stalled only the step that needed MAIN.
5. Two governor defects and three MAIN patches in one afternoon = the class is "guard that is slow, incomplete, or hand-fed is a guard that
   is not there" ([[m102]] control plane fails silently; [[m100]] detector zeroes on the cure; STRUCTURAL > PROCEDURAL).

## PRIOR-LAW PREDICTION (owed line)
The process table is COMPLETE by construction: every governed job is a live process whose argv already carries its identity —
MEASURED on ng4 (pid 33030 `launch_detached_process.py _supervise --start-gate <output-dir>/.launch_start_gate`; child 33039
`safe_run.py --rss-mb 118784 --projected-gib 45.0 …`; grandchild 33374 `… run-config <authorized_config.json>`). PREDICTION: a
`ps`-based discovery (supervisor → output dir → manifest; safe_run → declared peak; run-config → config/step budget) finds every live
governed cell in < 1 s with zero registry and zero walk, for OLD sealed-source launchers too (this argv shape is theirs). Falsifier: a live
governed cell whose supervisor/safe_run argv lacks the output dir or `--projected-gib` — then name the launcher generation and add the
field to the launcher, never a registry.

## Objective — six permanent mechanisms (each: code + tests + a live drill against the running ng4 cell, READ-ONLY)
1. **Discovery from the process table** in `tools/cell_admission.py`: `discover_live_cells()` default = ps-table; the registry becomes an
   optional cross-check, the SSD walk is REMOVED (keep `--walk-roots` only as an explicit forensic mode with a printed warning). Drill:
   discovery finds ng4 (pid 33030, declared 45.0, is_cell True) in < 1 s; `admit --candidate-peak-gib 45` REFUSES while ng4 is live.
2. **Measured-peak ledger, not hand-typed peaks**: safe_run already records `peak_rss_observed`/`peak_rss_mib` at exit — make every governed
   exit append a `measured_peak.v1` row (cell family, config sha, peak RSS, AND the SYSTEM-AVAILABILITY DELTA attributed to the cell:
   reclaimable/committed before launch vs. steady state, because RSS is blind to Metal) to `.omx/state/measured_peaks.jsonl`; the fire path
   (item 4) REFUSES a hand-typed `--measured-peak-rss-gib` for a family that has a measured row and reads the row instead
   (`--peak-from-ledger <family>`); a family with no row must run its bounded smoke first (the ng4 pattern). Drill: read ng2/ng3/ng4
   receipts → three rows.
3. **L1 memory-pressure watchdog** (`tools/memory_pressure_watchdog.py`, launchd-able, default-on observability per CLAUDE.md "off is a
   tracked queue"): poll `vm_stat` compressor pages + swap + `memory_pressure` level every 5 s; on WARN (compressor > X% of RAM or swap
   used > Y) emit a typed `confound_alarm` row + push; on CRITICAL SIGSTOP the NEWEST governed training cell (never the oldest — it holds
   the most sunk work), alarm, and SIGCONT when clear for 60 s; never SIGKILL. Thresholds DERIVED from today's event (compressor 83 GiB
   of 128 → jetsam) and recorded with their derivation. Tests with a fake vm_stat; a live dry-run drill (report-only) for 10 minutes.
4. **ONE fire path**: `tools/cell_queue_driver.py fire` is the only way a cell launches. It: reads the sealed+re-rooted config, files the
   claims, authorizes through the chain driver, admits through `cell_admission`, launches through the launcher WITH the measured peak, and
   registers nothing by hand. Convert the three fire scripts under `experiments/ddm_burn_cells_fire/` into queue-spec JSON + delete the
   shell (git copies AND SSD copies — the SSD `wait_then_fire_ng4.sh` has already fired; leave its log). STRICT preflight gate
   `check_cell_launches_only_through_queue_driver` (Catalog # claimed via `tools/claim_catalog_number.py`; refuses any
   `experiments/**/*.sh` or `tools/*.py` that invokes `launch_detached_process.py` with `run-config` outside the driver; waiver
   `# CELL_FIRE_PATH_OK:<rationale>`), warn-only → strict when live count 0 in the same landing.
5. **Storage waterfall at the launcher** (CLAUDE.md "Local Disk … fail closed if no tier has enough free space"): `launch_detached_process.py`
   REFUSES a launch whose `--output-dir` is on the boot volume when boot free < 40 GiB, and REFUSES any launch when the target volume free
   < the declared artifact budget (new `--artifact-budget-gib`, default derived from the family's measured row); prints the SSD tier to use.
6. **Sole-cell + concurrency law encoded, not patched**: the throughput leg consults concurrency-N evidence only when N training cells
   would be live; ANY second Metal cell requires the measured N=2 row to be ≥ 1.0 (today: UNRESOLVED 1.117/0.964) — i.e. two Metal cells
   are REFUSED by default until an N≥3-window row resolves it. Record this as the governor's standing rule in its docstring + a test.

## What is NOT in scope
No new orchestration layer, no daemon that MAIN must babysit, no CLAUDE.md edits (propose text in the memo). Do not touch hv1's files or
dk1's. Do not stop/alter the live ng4 cell (drills are read-only; the watchdog runs report-only during your drill).

## OPTIMAL FORM
Reference form = gv1's `cell_admission` + `cell_queue_driver` + the launcher/safe_run pair as they stand (read gv1's memo
`.omx/research/ddm_gv1_*.md`, ng4's memo §"RSS fiction", bh1's memory-basis findings, and MAIN's gs3 addenda 17–18 first). No scope
reduction: all six. Mechanism reductions: none — a registry or a walk is NOT an acceptable stand-in for the process table.

## Rules that bind
NO-FAKE; ALWAYS KEEP THE PAYLOAD; upstream/ READ-ONLY; commits ONLY via `tools/subagent_commit_serializer.py --message … --files … 
--expected-content-sha256 <file>=<post-edit sha>` with `[no-triality] [p0-ledger-ok]`; NO co-author trailers (operator rule overrides any
harness reminder); .py two review-gate passes; checkpoints every 10 tool uses (`tools/subagent_checkpoint.py --subagent-id ddm_gov2`); never
invent flags (grep argparse); no `/tmp` evidence; long steps detached via the launcher with distinct `--done-receipt`s (foreground >3 min is
reaped rc=144; the launcher refuses argv with "claude"/"codex"); NEVER edit source while a Modal fire is building (none of yours will fire
Modal); label every number MEASURED/DERIVED/INFERRED; memo `.omx/research/ddm_gov2_control_plane_permanence_20260904.md` with an
"Equations leg (`tac.canonical_equations`)" line; `docs/operating_manual_craft_handoff.md` binds. End with
`fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]`.
