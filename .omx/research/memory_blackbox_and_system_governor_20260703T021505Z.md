# Memory black-box recorder + system-aware dynamic memory governor — landing ledger

- **UTC:** 2026-07-03T02:15:05Z
- **Task:** operator P0 machine-protection (after a full machine crash 2026-07-02).
- **Pointer:** 0.19110 **UNMOVED** — this is apparatus / machine-protection (score-neutral), NOT a d_seg/d_pose/rate lever. Per the means/ends firewall it is honestly labelled MEANS.
- **Verdict:** LANDED. Black-box recorder + adaptive-ceiling + admission HARD-gate (shipping ADVISORY pending independent review) + reversible dynamic throttle. 78 tests pass. `ruff --select F821` clean. Canonical equation `adaptive_ceiling_admission_control_v1` registered (triality EQUATIONS leg).

## 1. Crash root-cause (what we built to fix)

The 2026-07-02 crash was **system-wide memory exhaustion**, not a single-process runaway. Our OOM protection was structurally BLIND to the system total:

- `tools/safe_run.py --rss-mb 90000` caps **each process** at 90 GB — but two processes each < 90 GB sum to > 128 GB.
- `tools/witness_memory_preflight.py` projects a **single run's** peak vs 70 % of RAM — blind to what else is running.
- `tools/memory_guard.py --watch` sheds the largest *training arm* when free < 30 GB — but it (a) only ran if launched, (b) only knew *training* arms, (c) reacted after the cliff rather than preventing admission.

The crash was the SUM of heterogeneous jobs: R1 trainer (~67 GB) + a byte-close/bsdtar/inflate job (several GB) + the ev1 descent-probe + concurrent build/measurement agents → macOS jetsam cascade → hang. And **no memory trajectory was logged** before the runs died — no black box. (JetsamEvent reports on 06-30 / 07-01 corroborate repeated memory-pressure kills.)

## 2. Deliverable 1 — memory black-box recorder (`tools/memory_blackbox.py`)

Always-on lightweight sampler (default 2 s; 0.5 s when pressure is elevated) → one JSON line per sample to the **fcntl-locked, rotated** `.omx/state/memory_blackbox.jsonl` (rotate at 20 MB → `.omx/state/archive/memory_blackbox_<utc>.jsonl`; NEVER `/tmp`). Each sample: wall-clock ts + iso + monotonic + `kern.boottime` (reboot detection); TOTAL / used / available / free / wired / compressor / swap; macOS pressure level (1/2/4); 1/5/15 load avg; the adaptive ceiling + budget + baseline; per-tracked-job RSS (label/pid/group-RSS/priority/paused); **and the accounting-trust fields** (closure_ok / cross_validated / fail_safe / discrepancy).

- `--last-crash [--minutes N]`: reads live + all archives, finds the MOST RECENT gap (ts jump > 30 s **or** a `kern.boottime` change), classifies it REBOOT vs SAMPLER-DEATH, and prints the trajectory of the N minutes leading INTO it (+ a peak-used / min-available / max-pressure summary). This is the black box: after any crash we read exactly what led to it.
- `--tail [N]`, `--sample-once`, `--status`.
- **Singleton:** `fcntl LOCK_NB` on `.omx/state/.memory_blackbox.singleton.lock` held for the daemon's life (a 2nd instance exits 0).
- **Auto-start:** `ensure_blackbox_running()` is invoked by `spawn_durable_daemon._do_start` on the first (non-infra) launch (idempotent) so the black box is recording before any heavy job runs. Recursion-guarded (never auto-starts itself) + skipped under `PYTEST_CURRENT_TEST` (hermetic tests).

## 3. Deliverable 2 — system-aware dynamic memory MANAGER

New `tools/system_memory_governor.py` (pure policy + live readers + CLI) + upgrades to `witness_memory_preflight.py`, `launch_witness_run.py`, `spawn_durable_daemon.py`, `memory_guard.py`.

### Adaptive ceiling (max out 128 GB safely)
```
safety_margin(T)    = max(8 GiB, 0.08·T)                 # 10.24 GiB on 128 GiB
adaptive_ceiling(T) = T − safety_margin(T)               # 117.76 GiB
baseline            = system_used − Σ current RSS of our tracked jobs   # OS + control-plane
training_budget     = adaptive_ceiling − baseline        # ~101.76 GiB at baseline 16
```
On this box the training budget floats to ~100–112 GiB — **much higher** than the old blind 90 GB per-process cap. We MAX OUT the box; concurrent jobs share the SAME system budget.

### Admission control — the P0 HARD PREVENT gate (job-type-BLIND)
```
projected_system_used = system_used + Σ_active max(0, peak_i − current_i) + projected_new_peak
ADMIT  iff  projected_system_used ≤ adaptive_ceiling
```
`system_used` is the **REAL vm_stat used** (counts OS + control-plane + training + byte-close + inflate + bsdtar + probes) → job-type-blind. No double count (current RSS is in `used` once; we add only each active job's REMAINING growth + the new peak). This REFUSES the 2nd/3rd concurrent job whose peak would push the SUM over the ceiling — the exact overflow that crashed us. Wired into BOTH `launch_witness_run.py` and `spawn_durable_daemon.py` (defense in depth); the registry now records each job's `projected_peak_gib` so the NEXT launch's gate can sum it.

**Enforcement rollout (trust req #4):** the gate is a HARD gate *by design* but ships **ADVISORY** (logs "WOULD-REFUSE", proceeds) until an independent adversarial review (reviewer ≠ author) signs off on the accounting + control-plane allowlist + fail-safe paths. Flip with `TAC_ADMISSION_ENFORCE=1` (a one-line, reviewable strict-flip). The only per-launch bypass is an operator-quoted `--admission-override-rationale` (placeholder/empty rejected).

### Dynamic throttle (reversible; never the control plane)
The black-box daemon doubles as governor: each sample it classifies pressure (WARN = available < 15 GiB or pressure≥2; CRITICAL = available < 8 GiB or pressure≥4) and, when sustained (debounced), **SIGSTOP-pauses the lowest-priority throttle-eligible job** (tie-break: largest RSS), **SIGCONT-resuming** (highest-priority first) when pressure clears. Pause halts memory GROWTH and is fully reversible. Priority ranks pointer-movers (#205 / sealed / witness) above probes/sweeps/byte-close. The throttle can pause a **byte-close/inflate/probe** job too (broad our-jobs allowlist), NEVER the control plane: every candidate must pass `memory_guard`'s vendored control-plane exclusions (not a control-plane app, no control-plane lineage, not ssh/tmux/shell/guard/blackbox, pgid not protected) AND be its own process-group leader. If `memory_guard` is unavailable → throttles NOTHING (fail-safe). Killing (last-resort SIGTERM) stays with `memory_guard --watch`; the governor only pauses. `memory_blackbox.py` + `system_memory_governor.py` were added to `memory_guard`'s protected denylist (a governor that pauses itself cannot recover the machine).

## 4. Why trust this over eyeballing `vm_stat`/`ps` by hand (trust reqs #1–#3)

The answer is STRUCTURAL, not "it reads vm_stat":

1. **Accounting is PURE + UNIT-TESTED against FIXED captured kernel counters to EXACT GiB.** `reconcile_memory_accounting(...)` takes raw page counts + page size (read from the kernel via `sysctl hw.pagesize` = 16384, never hardcoded blindly) + `hw.memsize`. The test feeds the 2026-07-02 captured snapshot (free 6572823 / active 728620 / inactive 467151 / speculative 279768 / wired 253178 pages) and asserts available_primary 107.42, used 20.58, wired 3.86, closure 1.33 GiB — so a parse bug (wrong page size, wrong field, CPU-vs-memory confusion) FAILS a test, never ships (a wrong page-size parse blows closure out by ~96 GiB → fail-safe, verified by test).
2. **Runtime CROSS-VALIDATION against a 2nd independent source.** Each check measures the SAME quantity two ways: (a) `vm_stat` free pages ≈ `sysctl vm.page_free_count`; (b) `page_size × memory_pressure page-total` ≈ `hw.memsize`; (c) one-sided: our CONSERVATIVE available must not EXCEED psutil's GENEROUS available (a parse overcount would). Disagreement beyond tolerance → **FAIL SAFE** (available reduced by the discrepancy → used inflated → admission refuses) + logged to `.omx/state/memory_governor.log`. A missing optional source (psutil/memory_pressure) is skipped, not a failure.
3. **Closure check.** `(free+active+inactive+speculative+throttled+wired+compressor)·ps ≈ total` within 4 GiB (the legit unaccounted gap is ~1.3 GiB on this box; a real parse bug is ~100 GiB). A large gap → fail-safe.
4. **Fail-safe forces REFUSE.** `admission_decision(..., fail_safe=True)` refuses even when the arithmetic would fit — we never admit on a reading we could not validate.

## 5. Tests + verification

- `src/tac/tests/test_system_memory_governor.py` (33 tests): exact-GiB accounting; wrong-page-size / free-mismatch / total-mismatch / overcount all FAIL SAFE; psutil-more-generous is NOT a failure; adaptive ceiling / budget math; the SUM-over-RAM admission REFUSE (the crash) + ADMIT + fail-safe-forces-refuse; priority ranking; throttle target (lowest priority, tie-break largest RSS, skips ineligible/paused, none-when-nothing-eligible); resume selection; pressure classification; governor action (warn-pauses-after-debounce, warn-debounce-alert, critical-pauses-fast, critical-escalates, normal-resumes); enforce-mode env default advisory.
- `src/tac/tests/test_memory_blackbox.py` (12 tests): sample shape (all trust + ceiling fields); append + 20 MB rotation; gap detection (sampler-death, reboot-via-boottime, continuous-none, most-recent-of-many); last-crash window trajectory; window summary; singleton exclusivity + daemon-exits-when-held.
- `src/tac/tests/test_witness_memory_preflight.py` (+2): system-aware admission refuses the 2nd concurrent run, admits the lone run.
- Regression: `test_memory_guard.py` + `test_spawn_durable_daemon_{lifecycle,memguard}.py` still pass (the killpg-cascade test passes with the getattr fix + PYTEST auto-start guard). **78 passed.**
- Live end-to-end: recorder writes real samples with validated accounting (closure_ok/cross_validated true, fail_safe false); `--last-crash` reports `continuous_no_gap` on healthy samples; governor `--snapshot`/`--ceiling`/`--admit` CLIs work; `launch_witness_run --dry-run` previews `system-admission: ADMIT` advisorily.

## 6. How we now MAX OUT 128 GB safely (the envelope)

- **Ceiling** = 128 − max(8, 10.24) = **117.76 GiB** system-used tolerated.
- **Budget** = ceiling − baseline(OS+control-plane, ~16 GiB) ≈ **~100–112 GiB** for all our jobs combined (vs the old blind 90 GB PER PROCESS).
- **Admission** lets a lone job use the full budget, but REFUSES a concurrent job whose peak would push the SUM over 117.76.
- **Throttle** pauses the lowest-priority job (reversibly) if a run grows unexpectedly under pressure, before jetsam.
- **Black box** records the whole trajectory so the next crash yields signal.

## 7. Explicit gating note (per trust req #4)

The admission gate is currently **ADVISORY** (logs would-refuse, does not block). It flips to ENFORCE only after an independent adversarial review (reviewer ≠ author) of: the accounting reconciliation + cross-validation tolerances, the control-plane throttle allowlist, and the fail-safe paths. Until then, `safe_run --rss-mb` (per-arm) + `memory_guard --watch` (whole-machine shed) + `witness_memory_preflight` (per-run projection) remain the active backstops, now JOINED by the black-box trajectory + the advisory system ceiling. Do NOT set `TAC_ADMISSION_ENFORCE=1` before that review.

## 8. Triality — EQUATIONS leg

Canonical equation `adaptive_ceiling_admission_control_v1` registered into `.omx/state/canonical_equations_registry.jsonl` (producer `system_memory_governor.py`; consumers launch_witness_run / spawn_durable_daemon / witness_memory_preflight / memory_blackbox; score-neutral; residual 0 — deterministic arithmetic identity; empirical anchor = the 2026-07-02 crash the law would have prevented). Module `src/tac/canonical_equations/adaptive_ceiling_admission_control_20260703.py`; registrar `tools/register_adaptive_ceiling_admission_control_equation_20260703.py`.

## 9. NOT done (held per operator)

R1 / #205 are NOT relaunched (held until this lands + the operator's GO). This landing does not touch the trainer, the archive, or any score path.
