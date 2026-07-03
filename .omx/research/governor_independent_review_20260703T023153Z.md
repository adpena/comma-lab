# Independent adversarial review — system memory governor: SAFE to flip ADVISORY → ENFORCE?

- **UTC:** 2026-07-03T02:31:53Z
- **Reviewer:** independent review agent (reviewer ≠ author of `df3e5d70c` / `21398095a`).
- **Scope:** `tools/system_memory_governor.py`, `tools/memory_blackbox.py`, `tools/witness_memory_preflight.py`,
  `tools/launch_witness_run.py`, `tools/spawn_durable_daemon.py`, `tools/memory_guard.py` + their tests.
- **Method:** ran the named suites (55 pass) + regressions (72 pass) = **127 pass**; `ruff --select F821` clean on all 6 files;
  read all 6 files adversarially; ran the LIVE read-only governor CLIs (`--snapshot`/`--ceiling`/`--select-throttle-dry-run`)
  alongside the live R1 run; cross-checked accounting against raw `vm_stat`/`sysctl`; constructed synthetic control-plane
  process tables to try to make a control-plane pid throttle-eligible. Did NOT touch R1 (pid 19940), the blackbox daemon
  (pid 19895), or the enforce flag. pointer 0.19110 UNMOVED (protection apparatus; MEANS).

## ⭐ TOP-LINE VERDICT: **SAFE-TO-ENFORCE**

Invariant 1 (control plane is NEVER throttled/killed) is **structurally airtight** and — critically — **already in force
under advisory** (the SIGSTOP throttle runs in the blackbox daemon independent of `TAC_ADMISSION_ENFORCE`; the flag only
gates whether a *launch* is blocked). There are **no confirmed admission or fail-safe defects**: the accounting is
unit-tested to exact GiB and fails safe on doubt, the sum-over-ceiling math is correct with no double-count of current RSS,
and — the decisive point — **enforce mode computes the identical decision as advisory and merely ACTS on REFUSE, so
flipping the flag can only ADD refusals, never admit anything advisory would not.** Its failure mode is a *false refusal*
of a legit launch (the SAFE direction for a machine-protection gate; the operator retains `--admission-override-rationale`
+ `--skip-admission-gate`). The five caveats below are characterized NON-BLOCKING limitations / follow-up hardenings; none
makes enforce worse than advisory.

Recommend flipping `TAC_ADMISSION_ENFORCE=1`. File the five follow-ups; none gates the flip.

---

## Per-invariant findings

### Invariant 1 — CONTROL PLANE NEVER THROTTLED/KILLED — **PASS (airtight)**

- **Structural argument.** Throttle eligibility (`_throttle_eligible`) is a pure **conjunction** of exclusions applied to
  the LIVE `ps` sample; a process is eligible ONLY if it passes ALL of: not self/guard-ancestor, not a control-plane app
  (`is_host_control_plane_process`), no control-plane lineage (`has_external_host_control_plane_lineage`), not on the
  extra-protected shell/ssh/tmux/guard/blackbox/governor denylist (`_matches_extra_protected`), pgid not in
  `protected_process_group_ids`, AND own-group-leader. Because it is a conjunction, **adding candidates can never expand
  the eligible set** — the widened candidate discovery (`OUR_JOBS_PATTERN ∪ registry`) is safe by construction.
- **Candidate discovery cannot reach the control plane on its own.** Claude (`node .../cli.js`), codex, shells, sshd,
  tmux, and the blackbox/governor do not match the training/byte-close/inflate/evaluate `OUR_JOBS_PATTERN` and are not in
  the durable-daemon registry, so they are not even candidates.
- **Live evidence (this box, R1 running):** `--select-throttle-dry-run` returns the **single** candidate
  `r1_storenothing_descent_relaunch` (a training arm — correctly, reversibly pausable). Blackbox (19895), shells, ssh,
  codex, claude are all excluded. `throttle_eligible=false` for the blackbox is **structural** (it is in
  `EXTRA_PROTECTED_TOKENS` AND does not match the pattern AND is self).
- **Adversarial synthetic probes (I tried to break it):** fed process tables with control-plane processes that are their
  OWN group leader, carry 90–95 GB RSS, live under a claude/codex ancestor, and whose command literally contains pattern
  tokens (`byte_close`, `inflate.py`). Result: **every** control-plane / CP-lineage / ssh / tmux / blackbox process →
  `throttle_eligible=False`; only a genuine detached trainer with no CP lineage was selected. Zero violations.
  - CP-app match caught even with a pattern token in argv (codex "…byte_close…", claude "…inflate.py" → excluded).
  - CP-lineage excludes even a detached own-group-leader (`evaluate.py` child of a live claude → excluded).
- **SIGSTOP scope is safe:** `pause_job` only `killpg`s when own-group-leader (`pgid==pid`), so the stopped group is rooted
  at the job itself; the control plane (an ancestor in its own group) is never in a training job's group.
- **Independent of the flip:** the throttle is live regardless of `TAC_ADMISSION_ENFORCE`, so this invariant is already
  exercised under advisory — flipping the flag changes nothing here.

### Invariant 2 — ADMISSION REFUSES THE SUM-OVER-CEILING — **PASS (with characterized caveats)**

- **Core math correct + tested.** `test_admission_refuses_sum_over_ceiling_the_crash` asserts `not d.admit` for the
  R1-plus-second-job case (projected 130 > ceiling 117.76); `test_admission_admits_when_it_fits` asserts admit;
  `test_project_system_used_no_double_count` pins the no-double-count identity. `system_used` is the REAL vm_stat used
  (counts every consumer at current RSS once); only each active job's REMAINING growth-to-peak is added → no double count
  of current RSS. The admit inequality uses `adaptive_ceiling = total − margin` **directly** (NOT `baseline`/`budget`), so
  the baseline computation cannot double-count into the decision (verified in `admission_decision`).
- **Error direction (per the prompt's question):** a **dead** registry row is skipped (`samples.get(pid) is None →
  continue`) → correctly ignored (fails toward more headroom, but the dead job's memory is already freed from
  `system_used`). A **live registered** job (R1) carries `projected_peak_gib` → its future growth IS counted. The one
  under-count is a live job that is *unregistered AND* only pattern-matched: its `projected_peak` falls back to current RSS
  → 0 growth headroom → its FUTURE growth is unaccounted (its current RSS is still in `system_used`). Bounded, and closed
  in practice by routing heavy launches through `spawn_durable_daemon --projected-peak-gib` (F4 below).
- **Caveat A (non-blocking): `used` counts inactive-anonymous as available → optimistic at LOW pressure.**
  `available = free + inactive`; on macOS `inactive` mixes reclaimable clean pages with dirty-anonymous pages that need
  compression/swap. This is macOS's conventional "used" (psutil agrees, which is why cross-validation passes), so it is a
  reasonable anchor, AND it **self-corrects**: as real pressure builds, `inactive` collapses toward 0 and `used → total −
  free → truth` — i.e. the optimism vanishes exactly when the gate must be accurate. Live box: `used 51.1`, `inactive
  ~36.7` — abundant headroom, safe to be optimistic now.
- **Caveat B (non-blocking): RSS-vs-vm_stat unit mix.** Growth terms are RSS deltas (`peak_rss − current_rss`) added to a
  vm_stat `used`. RSS overcounts shared/mapped pages (live: R1 RSS 56 GB > whole-system used 51 GB, MLX unified-memory
  effect), but the overcount is roughly constant so it largely cancels in the delta; the mix is bounded and mostly
  conservative.
- **Enforce ⊆ advisory:** enforce and advisory compute the same `AdmissionDecision`; enforce only returns rc=5 on REFUSE.
  So enforce can never admit more than advisory — the flip is monotone toward safety.

### Invariant 3 — ACCOUNTING CORRECT + FAILS SAFE ON DOUBT — **PASS**

- **Unit-tested to exact GiB against a captured kernel snapshot.** `test_reconcile_exact_captured_snapshot` asserts
  total 128.0 / available_primary 107.42 / used 20.58 / free 100.29 / wired 3.86 / closure 1.33 to tight tolerance. Page
  size is read from `sysctl hw.pagesize` (not hardcoded); I confirmed the closure identity by hand on the LIVE raw
  `vm_stat` (free+active+inactive+speculative+throttled+wired+compressor = 126.68 GiB vs 128 → closure 1.32 GiB, matching
  the CLI's `closure_gib=1.324`).
- **Every parse-bug class fails safe (asserts, not smoke):** wrong page-size (4096) → `closure_gib > 50` → `fail_safe`;
  free-page mismatch → `fail_safe`; total (memory_pressure) mismatch → `fail_safe`; conservative-exceeds-psutil overcount
  → `fail_safe`. The one-sided psutil check correctly does NOT trip when psutil is legitimately more generous.
- **Fail-safe forces refuse:** `fail_safe` reduces `available` by the discrepancy → inflates `used` → and
  `admission_decision` returns REFUSE regardless of arithmetic (`test_admission_fail_safe_forces_refuse_even_when_it_would
  _fit`). Loud log to `.omx/state/memory_governor.log`.
- **Live health while R1 runs:** `closure_ok=true`, `cross_validated=true`, `fail_safe=false`, `discrepancy≈0.001` — so
  enforce will NOT spuriously fail-safe-refuse on this box under current conditions.

### Invariant 4 — REGISTRY RACE SAFETY — **PASS for the write; ONE low-severity TOCTOU (non-blocking)**

- **Registry write is solid:** `_registry_lock` (fcntl `LOCK_EX`) + re-entrancy depth counter + `_save_registry_atomic`
  (unique tmp + fsync + `os.replace`) + a runtime guard that REFUSES a save outside the lock. Concurrent
  launchers/stoppers serialize correctly; readers see a stable snapshot via atomic replace.
- **F5 (LOW, non-blocking) — admission↔registration TOCTOU.** In `_do_start`, the admission gate reads live system +
  `list_tracked_jobs()` and decides BEFORE `Popen` and BEFORE `_register_daemon`. Two *truly simultaneous* launchers can
  both pass admission before either registers, then both start → their peaks not accounted against each other. Severity is
  LOW: the operator serializes launches (R1/#205 are hand-gated), advisory has the identical window, and the crash was
  heterogeneous jobs launched at different TIMES (a sequential 2nd launch DOES see the 1st's registered projected_peak).
  Hardening (follow-up): hold the registry lock across admission+register, or write a "pending reservation" row before
  `Popen`.

### Invariant 5 — DEGRADATION — **PASS (acceptable)**

- **Governor import/read raises → fail-OPEN to the PRE-EXISTING backstops**, not to nothing: both `launch_witness_run`
  and `spawn_durable_daemon._system_admission_gate` print a WARNING and proceed, but the 30 GB free-floor preflight
  (`_mem_preflight`) and the per-arm `safe_run --rss-cap-mb` still run. This is fall-back-to-status-quo (the layers that
  existed before this landing), which I judge acceptable — hard-blocking every launch on governor-unavailability would be
  worse and the ledger's "fall back to per-process cap" claim is accurate.
- **In-band accounting doubt → fail-CLOSED** (refuse). The two are correctly distinguished.
- **Blackbox daemon death** stops the throttle + trajectory logging but leaves the admission gate + free-floor +
  per-arm cap intact; the blackbox/governor are in `memory_guard`'s protected denylist so they are never shed/paused
  (a governor that pauses itself cannot recover the machine — verified in `EXTRA_PROTECTED_TOKENS`).

---

## Confirmed issues (most-severe first) — ALL NON-BLOCKING for the flip

1. **F1 — Throttle efficacy gap on the live R1 tree (MEDIUM; throttle, not admission; not a CP-safety issue).**
   Live tree: `safe_run(19937, grp 19937)` → `bash launch.sh(19938, **new** grp 19938)` → `python trainer(19940, grp
   19938, 56 GB)`. The registered custody daemon is the safe_run wrapper (own leader 19937), so a throttle pause would
   `killpg(19937, SIGSTOP)` — group 19937 contains ONLY the 21 MB wrapper; the 56 GB trainer lives in group 19938 and is
   NOT halted. Worse, SIGSTOP-ing safe_run suspends its per-arm RSS monitor while paused. Net: the throttle may fire yet
   not relieve pressure. This does NOT threaten the control plane (invariant 1 holds) and does NOT regress vs the
   pre-throttle status quo (it just fails to help); under sustained pressure the governor escalates to `escalate_alert` →
   `memory_guard --watch`. Fix: pause the job's actual memory-bearing descendant group, or make the launch tree a single
   process group (no intermediate `bash` regrouping). *Also causes the tracked_current double-count below.*

2. **F2 — `tracked_sum_gib` telemetry double-counts (LOW; observability).** Because both the registered wrapper (19937,
   `group_rss` includes descendants = 56 GB) and the pattern-matched trainer child (19940, own group_rss = 56 GB) appear
   as tracked jobs, `tracked_current_gib≈112` for a ~56 GB run. This is **benign for admission** (the admit inequality uses
   `total − margin`, not `baseline`; growth headroom is counted once via 19937's projected_peak; `system_used` is the
   independent vm_stat truth), but it makes the blackbox `tracked_sum_gib`/`baseline` telemetry inaccurate. Per the
   "telemetry accuracy vital" discipline: dedupe by descendant-set (skip a candidate whose pid is a descendant of another
   candidate) so a run is counted once.

3. **F3 — inactive-anonymous optimism in `used`** (see Caveat A) — LOW; self-correcting; document.

4. **F4 — unregistered heavy jobs contribute 0 growth headroom** (see Invariant 2 error-direction) — LOW; close by routing
   ALL heavy launches through `spawn_durable_daemon --projected-peak-gib`.

5. **F5 — admission↔registration TOCTOU** (see Invariant 4) — LOW; serialized-launch operational context mitigates.

## Observations (not defects)

- Ledger §5 says "78 passed"; the current suite is **127** (55 named + 72 regression). Stale count in the ledger; the
  author's later "127 pass" claim is correct and independently confirmed here.
- `admission_enforcing()` default is FALSE; the override rationale rejects placeholders (`_rationale_is_real` / len≥8).
  Infra (`memory_blackbox.py`/`memory_guard.py`/`system_memory_governor.py`) is auto-exempt from the admission gate and the
  black-box auto-start passes `--skip-admission-gate`, so enforce cannot deadlock the protection layer itself. Verified.

## Bottom line

Flip `TAC_ADMISSION_ENFORCE=1`. Invariant 1 is airtight and already live; enforce is monotone toward safety (adds
refusals, never admits more; false-refusal is the safe direction with operator overrides available); accounting is
unit-tested to exact GiB and fails safe; the live box is healthy (closure 1.32, no fail-safe). Track F1–F5 as follow-ups
(F1 the throttle-efficacy gap is the highest-value one, but it is a THROTTLE issue orthogonal to the admission flip).
