# ddm_gov2 — control-plane permanence: the governor complete by construction

**Arm:** ddm_gov2 (Opus) · **Charter:** `.omx/research/charters/ddm_gov2_control_plane_permanence_20260904.md`
**Operator source:** 2026-09-04, verbatim *"much of this requires a more permanent solution."*
**Cost:** $0 · 0 Metal · 0 Modal · **Pointer: UNMOVED** (this is apparatus; it cannot move it)
**Constraints honored:** the ng4 cell (pid 33030) was LIVE throughout and was never stopped, signalled, or altered.
Every drill was read-only; the watchdog drill ran `--report-only`. Sister arms hv1 and dk1: none of their files touched.

---

## The headline

MAIN patched the governor three times by hand in one afternoon. Each patch was correct and none of them was the
cure, because all three defects share one root: **the guard was being fed by things that can forget** — a registry a
launcher generation never wrote, a filesystem walk too slow to run, and a memory number a person typed.

Six mechanisms replace all three feeds with things that cannot forget. The two that matter most:

**1. Discovery now reads the PROCESS TABLE, which is complete by construction.** A governed job is a live process
and its argv already carries its whole identity. MEASURED on the live ng4 cell: **0.045 s**, versus the **>120 s**
SSD walk it replaces — a **>2,600×** speedup, and it works for the sealed-source launcher copies that never wrote a
registry row.

**2. The 2.396 GiB peak was not just wrong, it was wrong by 20.69×.** Harvesting the `safe_run` status receipts that
were already on disk gives the family's real cost: **49.572 GiB** of system availability. ng2 ran a full Metal burn
declared at **2.3959503173828125 GiB**. That is the number that admitted a second ~40 GiB cell beside it, and the
machine went to a VM-compressor collapse an hour later.

**And a correction to my own charter, from the machine's own record.** The charter says the compressor hit
"83 GiB of 128" at 17:11Z. MEASURED from `.omx/state/memory_blackbox.jsonl`: the peak was **76.978 GiB** (17:10:49Z),
the peak swap was **72.0 GiB** (17:11:05Z), and **it happened TWICE** — the first collapse was at **16:45:56Z**, 25
minutes earlier. The second event is the one that got noticed.

---

## 1. The event, second by second (MEASURED, and it changes the design)

Every row verbatim from `.omx/state/memory_blackbox.jsonl`, the sampler this repo already runs:

| time (UTC) | compressor | swap | available | free | pressure |
|---|---:|---:|---:|---:|---|
| 16:45:36.831 | 0.66 GiB | 2.30 | 18.83 | 0.03 | normal |
| 16:45:39.981 | **19.68** | 2.30 | 33.72 | 0.05 | normal |
| 16:45:42.714 | **46.42** | 2.29 | 29.16 | 0.04 | normal |
| 16:45:45.105 | 63.92 | 2.29 | 23.28 | 0.02 | **warn** |
| 16:45:56.044 | 73.99 | 22.03 | 19.30 | 0.07 | **CRITICAL** — jetsam |
| 17:10:49.046 | **76.978** (peak) | 64.08 | 18.22 | 0.016 | CRITICAL |
| 17:11:05.610 | 70.17 | **72.00** (peak) | 21.13 | — | CRITICAL |

Three things fall out that I would not have guessed, and each one decided a design choice:

**(a) The whole collapse takes 16 seconds.** From the first sample above 16 GiB of compressor to CRITICAL is
**16.06 s**; the ramp rates are **+6.03 GiB/s** then **+9.79 GiB/s**. A 5 s poll buys three samples. A debounce
("act only after two consecutive criticals") would spend two thirds of the runway, so the watchdog acts on the
**first** critical sample, deliberately.

**(b) Availability is BLIND here.** `available_gib` never fell below **18.222 GiB** during the entire collapse, and
the canonical reclaimable basis read 19–33 GiB while jetsam was killing daemons. A guard watching free memory would
have seen a healthy machine. This is worth stating plainly because the admission guard *is* a free-memory guard: it
is correct for the question it answers (can this cell's declared footprint fit?) and it cannot see this failure at
all. The two guards are complements, not a duplicate.

**(c) `free` looks like the leading indicator and is not usable.** It sat at 0.02–0.3 GiB for ~40 s before the
collapse — but over the day's **13,235** samples, `free < 1 GiB` was true **28.99%** of the time. Rejected on its
base rate, and recorded so it is not re-proposed. The kept triggers are rare: compressor ≥ 16 GiB **4.193%**,
compressor ≥ 48 GiB **3.189%**, swap ≥ 4 GiB **4.594%**, swap ≥ 16 GiB **2.199%**, pressure ≥ warn **3.408%**,
pressure critical **0.748%**.

---

## 2. Mechanism 1 — discovery from the process table

`tools/cell_admission.py`: `live_cells_from_process_table()` is the default; the registry is an optional
cross-check; the SSD walk survives only as `--walk-roots`, with a printed FORENSIC-MODE warning, and it is now
**unioned** with the process table rather than substituted for it (an admission decision must never lose a live job
because the caller asked a narrower question).

The argv shape is the OLD launcher's, which is why it works for sealed trees — MEASURED on live ng4 at 22:57Z:

```
33030 ppid 1     … launch_detached_process.py _supervise --manifest <out>/launch_manifest.json … --
33039 ppid 33030 … safe_run.py --rss-mb 118784 --projected-gib 45.0 --status-receipt <out>/… --
33374 ppid 33039 … ddm_qbr1_born_fairform_burn_prep.py run-config <authorized_config.json>
```

supervisor → output dir + manifest · safe_run → the declared peak · trainer → the sealed config and step budget.

**DRILL (read-only, against the live cell).** `live_cells_from_process_table()` in **0.045 s**; the CLI end-to-end in
**0.261 s**. It found ng4 as `pid=33030, is_cell=True, declared_peak=45.0 GiB, safe_run_pid=33039,
trainer_pid=33374, 1918/5000 steps`, plus the dashboard server as a bare `safe_run` with `ppid 1` and no supervisor.
`admit --candidate-peak-gib 45` returned **rc=2 REFUSE**. Registry cross-check: **agrees**, with the registry
holding **1** live row against the process table's **2** — the registry is a strict subset, exactly as predicted.

Two bugs of my own, both found by running it rather than reading it:

* **The supervisor's argv CONTAINS the safe_run argv** after its `--`, so `any(part.endswith("safe_run.py"))`
  matched the supervisor and reported `safe_run_pid=33030` where the real one is 33039. Pinned by a test.
* **A supervisor with no visible safe_run child** lost the `--projected-gib` reading. Now read from the supervisor's
  tail as a fallback. Also pinned.

Declared peak is **fail-closed: the larger** of the manifest budget and the live `--projected-gib`. An UNDECLARED job
is charged its measured tree RSS so it can never read as free.

---

## 3. Mechanism 2 — the measured-peak ledger

`tools/measured_peaks.py` → `.omx/state/measured_peaks.jsonl` (`measured_peak.v1`, append-only under `LOCK_EX`).
Two numbers per row, because **RSS alone is a lie on Apple Silicon** and the availability delta alone is a lie on a
CPU-bound one:

* `peak_rss_gib` — from `safe_run`'s own status receipt, which every launcher generation already writes.
* `system_availability_delta_gib` — availability just before the launch minus the minimum while it ran, read from
  the memory blackbox rather than from a new instrument.

`governed_peak_gib = max(the two)`. Every row carries an attribution grade
(`SOLE_CELL_INFERRED_FROM_LEDGER` / `CONFOUNDED_OVERLAPPING_CELL` / `UNAVAILABLE_NO_PRE_LAUNCH_READING` /
`UNAVAILABLE_NO_LIVE_SAMPLES`) so a confounded number can never be read as a clean one.

**DRILL — 9 real status receipts harvested; the ledger is gitignored LIVE_STATE, so the rows are verbatim here:**

| cell | declared | peak RSS | availability Δ | governed | grade |
|---|---:|---:|---:|---:|---|
| seed_20260902_control_native100 | 2.396 | 1.631 | — | 1.63 | NO_PRE_LAUNCH |
| seed_20260902_area_cap_control_native100 (4.16 s failed launch) | 2.396 | 0.246 | — | 0.25 | NO_PRE_LAUNCH |
| **seed_20260902_area_cap_control_native100** | **2.396** | 1.661 | — | 1.66 | NO_PRE_LAUNCH |
| seed_20260902_tau_band_control_native100 | 41.500 | 1.630 | — | 1.63 | NO_PRE_LAUNCH |
| ng3 bounded-smoke arm | 42.000 | 40.903 | — | 40.90 | NO_PRE_LAUNCH |
| ng3 bounded-smoke arm | 42.000 | 40.920 | — | 40.92 | NO_PRE_LAUNCH |
| **seed_20260902_continuous_objective_control_native100 (ng4, live)** | 45.000 | **1.746** | **49.572** | **49.57** | SOLE_CELL |
| ng4 bounded-smoke arm | 46.480 | 40.371 | 24.97 | 40.37 | SOLE_CELL |
| ng4 bounded-smoke arm | 42.000 | 40.420 | 22.42 | 40.42 | SOLE_CELL |

Family verdicts: `ddm_qbr1_born_fairform_burn_prep` **49.572 GiB** (10 rows, artifact 1.1762 GiB) ·
`ddm_ng3_tau_band_cell` 40.920 GiB (4 rows) · `ddm_ng4_continuous_objective_cell` 40.420 GiB (4 rows).

**What the table shows that neither instrument shows alone.** On ng4 the availability delta is **28.4×** the peak
RSS (49.572 vs 1.746) — the Metal allocator is invisible to `ps`. On the CPU-side bounded smokes the ordering
INVERTS: RSS 40.9 GiB against an availability delta of 22.4 GiB. `max()` picks the right instrument in both regimes,
and it did so on real data without being told which regime it was in.

**ng4's memo said "on the order of 45 GiB, INFERRED from a system-total delta."** That inference is now a direct
measurement, and it is **49.572** — MAIN's hand estimate of 45.0 was **9.2% low**. The number that mattered is
ng2's: **2.3959503173828125 GiB declared against 49.572 measured = 20.69×**.

The disk side is measured the same way rather than guessed: each row carries `artifact_gib` (a bounded `du` of the
run dir, truncation reported), which is what the launcher's storage waterfall defaults its budget from.

---

## 4. Mechanism 3 — the L1 memory-pressure watchdog

`tools/memory_pressure_watchdog.py`. Polls `vm_stat` (compressor) + `sysctl vm.swapusage` + `memory_pressure` every
5 s. WARN → a typed `confound_alarm` row in `.omx/state/memory_pressure_alarms.jsonl` + stderr. CRITICAL → **SIGSTOP
the NEWEST governed training cell**, never SIGKILL, never the oldest (the oldest holds the most sunk compute; the
newest is the one whose admission was the mistake), and SIGCONT after 60 s clear. `_signal_tree` **raises** on any
signal other than STOP/CONT — a watchdog that can kill a run is a bigger hazard than the pressure it watches. A
stopped cell is always resumed on exit.

Thresholds, all DERIVED from §1 with their lead times: **WARN** compressor ≥ 16 GiB (16.06 s of runway) / swap ≥ 4
GiB (9.4 s) / pressure ≥ warn (10.9 s). **CRITICAL** pressure critical / compressor ≥ 48 GiB (10.9 s) / swap ≥ 16
GiB (2.6 s) / compressor growth ≥ 4.0 GiB/s (the measured ramp was 6.03–9.79).

**DRILL (read-only, 10 minutes, `--report-only`, launched detached).** 602.0 s, **121 polls**, **0 alarms**, rc 0,
nothing signalled. And a two-instrument agreement check against the independently-written blackbox daemon at the
same instant: compressor **0.000 GiB** apart, swap **0.000 GiB** apart.

One parser trap worth naming: `vm_stat` reports both *"Pages stored in compressor"* (393,054) and *"Pages occupied
by compressor"* (86,670) — a **4.5×** difference at the same instant. Reading the wrong line inflates every alarm by
that factor. Pinned by a test.

---

## 5. Mechanism 4 — ONE fire path, and the measured-peak law

`tools/cell_queue_driver.py fire` is now the only way a cell launches. A queue-spec cell may set
`"measured_peak_rss_gib": "from_ledger"`, and a NUMBER below the family's measured row is **REFUSED**. The launcher
argv is rewritten with the resolved peak at fire time — the spec owns every scientific value; the memory declaration
is the one value the governor owns, because it is the value the machine's safety depends on and the one that was
wrong. A family with no measured row may still declare (the honest bootstrap: run the bounded smoke first, as ng4
did; the row it produces then governs every later launch).

The three shell fire scripts are converted to `experiments/ddm_burn_cells_fire/burn_cells_queue.json` and **deleted**
from git AND from the SSD (`wait_then_fire.log` left in place). Git and SSD copies were byte-identical — verified by
sha before deletion (`1bfe987e…`, `d9bd8297…`, `d82708b2…`, `4f56e3c1…` on both sides) — and every launcher argv is
preserved verbatim in the spec, so no information is lost.

**DRILL.** All three cells resolve **49.572 GiB `FROM_LEDGER`**, replacing 2.396 / 41.5 / 45.0. All three are
correctly refused: admission REFUSE (ng4 is live), and ng2 additionally on `PIN_PATHS_NOT_REROOTED` — independently
re-finding the defect that killed its first launch. Feeding back ng2's exact literal:

```json
{"reason": "HAND_TYPED_PEAK_BELOW_MEASURED", "declared_peak_gib": 2.3959503173828125,
 "measured_peak_gib": 49.572, "under_declaration_factor": 20.69,
 "cure": "set \"measured_peak_rss_gib\": \"from_ledger\" in the queue spec"}
```

**STRICT gate, Catalog #413** — `check_cell_launches_only_through_queue_driver` in `src/tac/confound_gates.py`,
wired into `preflight_all` **STRICT in this same landing** (no warn-only purgatory). **Live count 0** over a
non-vacuous denominator: **12,456 files considered, 2 carrying both tokens**. A live positive control (a planted
`experiments/planted_fire_cell.sh` reproducing the ng2 shape) fires, verified through
`check_refusal_gates_have_live_positive_control`.

The detector keys on **execution**, not on mention: a shell command line invoking the launcher, or a Python
`subprocess`/`os` call whose literals name it, in a file that also carries `run-config`. That distinction is
load-bearing — `experiments/ddm_qbr1_born_fairform_burn_prep.py` legitimately *composes* a launcher argv into a
fire-order JSON without ever running it, and a naive scan flags it. The queue driver is exempt by construction, not
by name: it passes `cell.launcher_argv` from the spec, so the launcher path is never a literal in its source.

---

## 6. Mechanism 5 — the storage waterfall at the launcher

`launch_detached_process.py` refuses (rc=11) a boot-volume launch when the boot data volume has < 40 GiB free, and
any launch whose target volume has less free than the artifact budget (`--artifact-budget-gib`, defaulting to **2× the
family's MEASURED `artifact_gib`**, else a 2.0 GiB floor). The record rides in the manifest.

dk1's two facts are honored: free space is measured on `/System/Volumes/Data`, and the refusal **reports the APFS
local-snapshot census** so the operator knows a thin may be the cure rather than a delete (dk1 MEASURED: a certified
32.97 GiB delete moved free space +1 GiB; thinning then released +65 GiB). The launcher **never thins** — that is
destructive and operator-level — it refuses with the reason and names dk1's tools and runbook.

One honest amendment to dk1's warning: `df /` does read the sealed system volume, but **MEASURED here, Python's
`shutil.disk_usage` already reports the data volume for both paths** (211.1 GiB free from either). The hazard is a
`df` hazard. I resolve to `/System/Volumes/Data` explicitly anyway so the intent survives a refactor, and pinned the
agreement in a test rather than the coincidence.

**DRILL.** PASS on VertigoDataTier (88.16 GiB free vs a 2.0 GiB budget). REFUSE rc=11 with the 2026-09-04 boot state
simulated (344 MiB free), snapshot census in the detail. REFUSE rc=11 on the budget leg. Live tier state MEASURED:
**APDataStore 17.04 GiB · VertigoDataTier 88.16 GiB · boot data 211.06 GiB · 3 local snapshots** (dk1 thinned 21 → 3).

---

## 7. Mechanism 6 — the concurrency law, encoded

`throughput_verdict` now carries the standing rule in its docstring and its code:

1. A candidate that would run **ALONE** is never gated on contention (`SOLE_CELL_NO_CONTENTION`) — MAIN's first
   patch, kept.
2. A **second or later** Metal cell is **REFUSED by default**, admitted only when the evidence at that concurrency
   is **RESOLVED and pays**. Two independent ways it fails to resolve, both measured by gv1: the rows **straddle**
   the baseline (1.117 then 0.964), or the **spread exceeds the effect** (4.286 > 3.285 steps/min).

This **reverses gv1's admit-on-no-evidence default, deliberately**, and the reversal is pinned by a test that says
so. gv1's reason was sound — a governor that refuses everything can never collect data — but the measured cost of
that default was the near-OOM. The answer is not to admit blind; it is to make the measurement path explicit:
`decide_admission(..., concurrency_measurement_override="<rationale>")` admits and stamps the row
`MEASUREMENT_OVERRIDE`. "Off" stays a tracked queue with a named way out, not a silent yes. The override cannot
overturn a MEASURED-and-COSTS negative, and it never touches the memory leg — that one has no override at all.

**DRILL.** Live: `UNRESOLVED_AT_CONCURRENCY — a 2-cell Metal configuration is REFUSED by the standing concurrency
law: 2 rows STRADDLE the baseline (ratios 0.964..1.117)`.

---

## What landed

| surface | file | tests |
|---|---|---|
| process-table discovery + concurrency law | `tools/cell_admission.py` | `test_cell_admission.py` 87 (20 new) |
| measured-peak ledger | `tools/measured_peaks.py` | `test_measured_peaks.py` 31 |
| L1 memory-pressure watchdog | `tools/memory_pressure_watchdog.py` | `test_memory_pressure_watchdog.py` 37 |
| the one fire path + measured-peak law | `tools/cell_queue_driver.py` | `test_cell_queue_driver.py` 78 (15 new) |
| storage waterfall | `tools/launch_detached_process.py` | `test_launcher_storage_waterfall.py` 18 |
| STRICT gate Catalog #413 | `src/tac/confound_gates.py`, `src/tac/preflight.py` | positive control + live count 0 |
| shell → spec | `experiments/ddm_burn_cells_fire/burn_cells_queue.json` (4 `.sh` deleted) | — |

**121 new tests. ruff clean on every file.**

Durable artifacts (SSD tier, never `/tmp`): `/Volumes/VertigoDataTier/pact/ddm_gov2_control_plane/`
(`watchdog_drill/`, `wf_pass/`, `handtyped_spec.json`). `.omx/state/measured_peaks.jsonl` and
`.omx/state/memory_pressure_alarms.jsonl` are gitignored LIVE_STATE — the durable record is this memo's verbatim
numbers, and the ledger rebuilds from `measured_peaks.py harvest`.

## What did NOT land, and why

* **No cell was fired.** ng4 is live and the box is legitimately full, so every admission refused. The `fire` path
  is built, drilled to the point of refusal, and structurally tested — but it has **never fired a real cell**. That
  is an honest untested-in-anger path, and the first real fire should be watched.
* **The watchdog has never fired in anger either.** The 10-minute drill saw a quiet machine, so its CRITICAL branch
  is exercised only by tests and by the replayed 2026-09-04 ramp. The threshold derivation is measured; the
  intervention is not.
* **`safe_run` does not append to the ledger at exit.** I chose the harvester instead: `safe_run` runs from SEALED
  SOURCE TREES, where `_REPO` resolves inside the sealed tree, so an in-process append would write the row to the
  wrong `.omx/state`. `harvest` reads the receipts those runs already write, which covers sealed and future
  launchers identically. Naming the alternative rather than silently skipping it.
* **The per-cell `authorize_*.py` scripts remain.** The charter named the shell; the queue driver's `authorize()`
  supersedes them functionally, but deleting them was outside the charter's explicit list.
* **No N≥3 contention row**, so the concurrency law's own blocking condition is still the unresolved N=2 evidence.
* **No CLAUDE.md edit** (out of scope). Proposed text is below.

## Proposed CLAUDE.md text (operator decision; NOT landed)

> **A governed cell launches through ONE path.** `tools/cell_queue_driver.py fire` is the only surface that may
> launch a training cell. It runs the seal law, the duplicate-receipt check, the storage waterfall, the measured-peak
> law, and memory + concurrency admission before spending a second of Metal. A bespoke fire script re-implements
> admission privately; on 2026-09-04 three of them each had their own memory rule, one declared 2.396 GiB for a
> 49.572 GiB family, and the machine reached a VM-compressor collapse that jetsam resolved by killing daemons.
> Enforced by Catalog #413 (`check_cell_launches_only_through_queue_driver`, STRICT).
>
> **A memory declaration is MEASURED, never typed.** `tools/measured_peaks.py` holds each family's measured cost as
> `max(peak RSS, system-availability delta)` — both, because RSS cannot see Metal and the availability delta cannot
> see a busy CPU. A family with a measured row may not be launched under a smaller number.

## Equations leg (`tac.canonical_equations`)

**No new equation registered, and that is the honest call.** The registry holds the campaign's *scientific* laws
(rate/distortion/exchange), and every number this arm produced is an *apparatus* measurement of one machine on one
day — a 128 GiB box's compressor behavior and one trainer family's footprint. Registering them would put
host-specific operational constants in the surface that CLAUDE.md's constants-are-poison rule exists to keep clean,
and a future host would inherit them as law. The consumable form is the ledger
(`.omx/state/measured_peaks.jsonl`, rebuildable by `harvest`) plus the thresholds' recorded derivation inside
`tools/memory_pressure_watchdog.py`, both queryable and both re-measurable.

The one row that TOUCHES the registry is gv1's `metal_concurrency_speedup_gv1_v2` (`verdict: UNRESOLVED_AT_N2`,
`noise_floor_steps_per_min: 4.286`): mechanism 6 is that verdict compiled into the governor's refusal path. The
equation is unchanged; this arm made it binding.

## Cross-references

`[[m102]]` the control plane fails silently — make silence loud at launch · `[[m100]]` the detector zeroes on the
cure; STRUCTURAL > PROCEDURAL · `[[m123]]` two validators disagree ⇒ the disagreement is the finding (used in the
good direction here: the watchdog and the blackbox agree to 0.000 GiB) · `[[m50]]` VACUITY==PASS (the gate reports
its denominator) · `[[m78]]` reclaimable-aware admission · `[[m79]]` ceiling 116 GiB ·
`.omx/research/ddm_gv1_governor_memory_guard_controller_polish_20260904.md` (the surface this extends, and whose
admit-on-no-evidence default this deliberately reverses) ·
`.omx/research/ddm_bh1_fresh_eyes_bug_hunt_20260904.md` §6 (the 1.204× shell over-trust) ·
`.omx/research/ddm_ng4_continuous_objective_cell_20260904.md` §"RSS fiction" (the inference this measured) ·
`.omx/research/ddm_dk1_local_disk_certify_and_move_reclaim_20260904.md` (snapshot pinning; `df /` vs
`/System/Volumes/Data`) · `.omx/research/ddm_gs3_gestalt_after_submission_20260903.md` addenda 17–18 ·
CLAUDE.md "Confound self-protection" (this is L1 + L2 + L3 for one confound class), "Bugs must be permanently fixed
AND self-protected against", "Local Disk, SSD Spill, Auto-Cleanup, And Provenance", "'Off' is a tracked queue".

---

fs2 S 0.14784474152757654 @ 180,023 B [contest-CUDA T4 n600]

---

## ADDENDUM (2026-09-05 ~00:0xZ) — the watchdog's own first alarms falsified its targeting rule

**The rule I shipped was wrong, and its own anger-data proved it inside an hour.** Eleven WARN rows
landed between **23:47Z and 23:55Z** in three bursts — compressor **20.00 → 39.83 GiB**, swap flat at
~1.01 GiB, OS pressure never left `normal`, **no CRITICAL**. So the thresholds behaved: they fired
early, they did not escalate on a machine that was coping, and the compressor drained each time
inside ~15 s (38.47 → 22.33 GiB in one 5 s step).

**But the cause was not the Metal cell, and my rule would have paused the Metal cell.** MEASURED at
23:58Z:

| job | RSS | governed | `is_cell` |
|---|---:|---|---|
| `ceil_planar` pid 81442 | **10.32 GiB** | yes | **False** |
| `ceil_shift` pid 80930 | **10.14 GiB** | yes | **False** |
| `ceil_zoom` pid 81089 | **10.33 GiB** | yes | **False** |
| ng4 cell pid 33030 | **0.49 GiB** | yes | True |

Three `ddm_mc1 --stage ceiling` jobs — launched through the canonical launcher, so governed, but
carrying no `run-config`, so not cells — held **30.79 GiB**, **62.8×** the training cell. "SIGSTOP
the newest governed TRAINING cell" would have stopped **two hours of sunk work holding 1.6% of the
resident footprint** and left every actual allocator running. A guard that pauses the wrong process
is worse than one that pauses nothing: it destroys work *and* does not help.

**The corrected rule** (`select_pressure_target`): rank **all** governed jobs — cell or not — by RSS
**growth** over a 6-poll / 30 s window; ties break on **current RSS**, then non-cell, then newest;
the **oldest live cell is excluded outright**. When the oldest cell is all there is, there is no
target and the row says so. WARN rows now carry `top_rss_growers` (top 3 with pid, `is_cell`, RSS
and delta) so the cause is legible **before** anything is critical, and the CRITICAL action records
the chosen pid, its delta, the selection rule, and the excluded cell.

**A second correction, caught by running the coordinator's proposed rule rather than reading it.**
Pure growth-ranking degenerates on a quiet machine: across a 4 s window every job's delta was within
**0.0013 GiB** of zero, and the ranking picked the **0.17 GiB** dashboard server over a **7.08 GiB**
ceiling job. Pausing 0.17 GiB during a CRITICAL achieves nothing. Hence `GROWTH_TIE_BAND_GIB = 0.5`
— ~380× above the measured idle drift, ~20× below the 10.3 GiB allocators — so a real grower still
wins outright and flat jobs fall through to size. Re-drilled live: the rule now selects
`ceil_shift` (10.02 GiB, non-cell) and excludes ng4 (33030).

Per-poll cost of the new growth signal, MEASURED: **0.168 s for 8 jobs** (0.118 s discovery +
0.050 s of tree walks) = **3.4%** of a 5 s poll.

Nothing was stopped: mc1's jobs are legitimate and finishing, and ng4 was untouched throughout.
`newest_training_cell()` survives as a diagnostic query, and a test pins that the watch loop no
longer calls it. **+20 tests (37 → 57).**

The genus is worth naming: **an L1 alarm that fires and names only the SYMPTOM sends a human to
`ps`.** The first eleven rows said "compressor 38.47 GiB" — true, and useless for acting. The cure
is not a better threshold; it is that the alarm carries the actor.

---

## ADDENDUM 2 (2026-09-05 ~00:4xZ) — the watchdog stranded a live cell; a pause with no guaranteed release is a silent kill

**This one cost work, and the defect is mine.** The pre-`2a24996da` instance fired CRITICAL on
compressor GROWTH RATE three times and SIGSTOPped ng4's trainer (pid 33374) each time. The ledger
holds **three SIGSTOPs and only two SIGCONTs**:

| SIGSTOP | machine went WARN-clear | SIGCONT | paused |
|---|---|---|---:|
| 00:16:45Z | 00:18:22Z (97 s) | 00:20:49Z | 244 s |
| 00:27:45Z | 00:28:07Z (22 s) | 00:29:29Z | 104 s |
| **00:34:20Z** | **never** | **none — MAIN rescued it by hand at 00:36:35Z** | **∞** |

**THE MECHANISM, and it is a deadlock I built.** Stopping a process does not free its memory. A
stopped process is an *ideal* eviction victim — resident, dirty, and not running — so macOS swapped
it out. MEASURED: **swap crossed the 4 GiB WARN threshold at 00:34:32Z, twelve seconds AFTER the
SIGSTOP**, and stayed above it for **126 samples (4m19s, peaking at 11.29 GiB)** with the compressor
parked at ~41 GiB. The clear hold could never be met **because the pause itself sustained the
condition it was waiting on.** The old code could only resume inside the OK branch, so there was no
exit at all. ~10 minutes of a 2.7 h run were lost and the whole run was at risk.

Three cures, all landed and tested:

1. **BOUNDED MAX-PAUSE — `MAX_PAUSE_S = 180.0`**, checked on **every poll before the level branches**
   (the old resume lived only in the OK branch — that was the bug). DERIVED: the worst pause that
   *could* clear went WARN-clear at 97 s, plus the 60 s hold = **157 s**; 180 s leaves 23 s of
   headroom and would have released the third pause at 00:37:20Z instead of never. It is also
   shorter than pause 1's 244 s, which bought nothing (the compressor was at 2.50 GiB by then).
   The forced resume writes `SIGCONT_MAX_PAUSE` with `hold_met: False` and the elapsed pause.
2. **ON-RESTART RECONCILIATION** — `reconcile_orphaned_pauses()` runs at every `watch()` startup and
   as `memory_pressure_watchdog reconcile`: it reads the alarm ledger for SIGSTOPs with no later
   SIGCONT, checks each pid's state, and SIGCONTs the ones still in **T**, logging an
   `orphaned_pause_reconciled` row. A watchdog process is mortal — TERMed on a landing, crashed,
   rebooted — and every one of those exits can strand a pause. The ledger already knew; nothing
   read it.
3. **THE RATE RULE NOW REQUIRES WEIGHT** — a growth-rate CRITICAL also needs the compressor at or
   above the WARN level. A 6 GiB/s ramp from 1 GiB to 5 GiB on a 128 GiB box is a program starting
   up, not a collapse. (All three real triggers — 29.63 / 36.85 / 39.18 GiB — still fire.)

**The reconciler's first run found a live orphan, and it was not hypothetical.** At 00:40Z the
`ceil_block` tree (pids 57580 → 57582 → 57653) had been in state **T since 00:39:14Z**, stopped by
the `2a24996da` instance — which has the corrected *targeting* but not the bounded pause. The
machine sat at WARN held by **swap 5.86 GiB alone**, with the compressor at a healthy 5.42 GiB:
the identical deadlock, on a sister arm's job, forming again while I wrote the fix. Reconciled for
real (`SIGCONT` delivered to all three pids); mc1's job is running again. Nothing was stopped.

**THE FIX IN GIT IS NOT THE FIX IN RAM.** The targeting fix landed as `2a24996da` at 00:05Z, and the
*old build kept running* until MAIN TERMed it — three SIGSTOPs of the wrong process happened after
the fix existed. A landing does not change a running process. Structural cure landed here: the
watchdog compares its own source mtime each poll and **retires itself when the file changes**,
resuming anything it holds on the way out, so a relaunch is the only action a landing needs. The
sister half — a launcher that refuses to keep an instance older than the tool it runs — is named
here and NOT built; it belongs to the launcher's own supervision surface.

Honest note on what remains: the bounded pause makes the *release* guaranteed, but it does not make
the *intervention* effective. A SIGSTOP relieves CPU and future allocation, not resident bytes —
that is why every measured pause was followed by swap growth rather than relief. Whether pausing is
the right actuator at all is now an open question this arm did not settle; the alternative (alarm
loudly, never signal) is a one-line default change if the next anger-data says so. **+20 tests
(57 → 77).**

---

## ADDENDUM 3 (2026-09-05 ~00:5xZ) — the actuator is demoted; the guard alarms and names the actor

**Operator decision, on this tool's own numbers: SIGSTOP hurt twice today and helped zero times.**
ng4's trainer was paused three times with only two releases; mc1's `ceil_block` stranded in state T.
Every measured pause was followed by swap GROWTH, never relief, and the reason is structural, not
incidental: **a stopped process is the ideal eviction victim** — resident, dirty, and not running —
so pausing converts a job's memory into swap rather than freeing it. A SIGSTOP relieves CPU and
future allocation; it does not return resident bytes. Under **never weaker state**, a guard that
destroys work is worse than no guard.

**LANDED: `report_only` now defaults to True.** The watchdog alarms, records `top_rss_growers` on
every row, and signals nothing. `--act` is the explicit opt-in and keeps the full safety apparatus
(growth-based targeting, the bounded `MAX_PAUSE_S`, resume-on-exit). Flag resolution is fail-safe:
`--report-only` wins over `--act` if both appear, and the old `--report-only` argv the live instance
was launched with keeps working unchanged — no combination can turn signalling on by accident.

**One deliberate exception: the reconciler runs in BOTH modes.** Report-only means "never STOP
anything", not "never RESCUE anything". SIGCONT of a process a dead instance stranded can only
restore work, and gating it behind `--act` would have left the very orphan this arm found frozen.
Pinned by a test.

This closes the question ADDENDUM 2 left open honestly rather than by preference: the alternative
was named there as a one-line default change if the next anger-data said so, and it did, twice, in
one night.

### OWED, named and NOT built tonight

1. **A COOPERATIVE PAUSE PROTOCOL — the actuator that would actually work.** The guard sends
   `SIGUSR1`; the governed trainer catches it, **checkpoints at the next step boundary and exits
   with a resumable receipt**; the queue driver re-admits it when pressure clears. That frees
   resident bytes (the process is gone, not parked) and loses no work (the checkpoint is the
   contract this repo already requires of every launch). It is strictly better than SIGSTOP on both
   axes on which SIGSTOP failed. It needs a trainer-side handler, a receipt schema, and a
   re-admission path in `cell_queue_driver` — a real unit, not a bolt-on.
2. **A launcher that refuses to keep an instance older than the tool it runs.** The watchdog now
   retires *itself* on a source change (ADDENDUM 2), which covers this one tool; the general
   "fix-in-git is not fix-in-RAM" guard for every long-lived governed process belongs to the
   launcher's supervision surface and is unowned.

**+10 tests (77 → 87.)** MAIN's live instance (`memory_watchdog_r3`, launched `--report-only` on
`de7b4229c`) already runs the landed behaviour, so this flip needs no relaunch.
