# Task #173 — FLEET-SCALING HARD-BLOCKER clearance (concurrent 2nd arm GO/NO-GO)

- **UTC:** 2026-07-10
- **Task:** #173, operator-GO 2026-07-10 ("Clear 173 blockers regardless"). Gate on the council-recommended
  concurrent A/B (measure HorizonWeightedMargin + StepNativeActivation in parallel).
- **Pointer:** 0.19108282 [contest-CPU] **UNMOVED** — this is apparatus / machine-protection (score-neutral).
  Honestly labelled **MEANS** per the means/ends firewall.
- **Triality:** guard/apparatus, no config lever — `[no-triality-lever]` (DSL/equations N/A; the one relevant
  equation `adaptive_ceiling_admission_control_v1` was already registered at the 2026-07-03 governor landing).
- **VERDICT: #173 CLEARABLE. All 6 blockers RESOLVED by intervening work** (5 fully; H2 resolved-for-the-
  sanctioned-path with a documented LOW control-plane-safe residual). Concurrent 2-arm A/B is **GO** — proven by
  a live ENFORCE-mode concurrent-admission smoke (admits 2 real-sized arms, refuses the 3rd that sums over).

## 0. The memory blocker (the council's primary concern) — DISSOLVED

Risk-register D9 assumed 2×67.6 GiB (self-orient ON) two arms exceed 128 GiB → forced SEQUENTIAL. **Self-orient
is now OFF** (owed-16 RESOLVED-REFUTING, realized-through-R transfer measured ≈0, 47 GiB RAM tax removed). Live
preflight on the sealed v7.5.2 config (`experiments/results/__v752_drystart_final__/launch.sh`, `--no-self-orient`
implicit in the sealed config):

```
[witness-mem-preflight] SAFE: projected peak 24.5 GiB <= safe ceiling 89.6 GiB (70% of 128)
  breakdown: fixed=15.0 + cf_mx_cache=0.07 + gt=3.41 + verdict=6.0 = peak 24.48
  config: num_pairs=600 render=384x512 self_orient=False verdict_batch=32 render_aa=ipe
```

**Single arm = 24.5 GiB → 2×24.5 = 49 GiB < 89.6 (preflight 70% ceiling) < 117.76 (governor adaptive ceiling).**
Memory blocker dissolved — CONFIRMED with the real preflight number.

## 1. Enumerated blocker table

Canonical source: **DAG FEED 2026-06-26as** (`sub015_DAG…:947`) + council memo
(`council_v752_relaunch_shape_concurrent_vs_single_20260710.md`: "#173's 2 HIGH + 4 MED blockers"). The 2 HIGH
are DAG-named verbatim (**FOUND**). No separate doc enumerates a crisp M1–M4; the 4 MED are **RECONSTRUCTED**
(consolidated) from the DAG residual line ("+ 4 MED + accurate-SHED-metric + sub-floor `--min-free-gb 18` + no
`--rss-cap-mb`") cross-referenced to the independent review's F1–F5
(`governor_independent_review_20260703T023153Z.md`). Stated plainly per-row.

| # | Sev | Blocker | Found/Recon | Resolved-by (receipt) | Status |
|---|-----|---------|-------------|-----------------------|--------|
| **H1** | HIGH | `check_launch_ok` has **no committed-reservation ledger** → N arms each read the same live-free → all admitted → sum ≫ headroom → OOM (literal 2026-06-25 root cause; repro 10/10 admitted @110 GB) | FOUND (DAG) | **SUM-over-RAM admission gate** (governor, 2026-07-03): registry records each job's `projected_peak_gib`; `project_system_used = system_used + Σ active(peak−current) + new_peak ≤ adaptive_ceiling`. **+ fcntl PENDING admission-reservation** (commit `4ed373e96`, 07-06) closes the launch race. **ENFORCE armed** (`.omx/state/admission_enforce.flag`, after `governor_independent_review` → SAFE-TO-ENFORCE, reviewer≠author). | **RESOLVED + ENFORCED + PROVEN** (smoke §2) |
| **H2** | HIGH | `_kill_pgrp` SIGTERM-only (no SIGKILL escalation) → a **BARE** (non-wrapped) custody arm that ignores SIGTERM is un-sheddable | FOUND (DAG) | **safe_run** installs a SIGTERM→SIGKILL cascade to the inner trainer group (`tools/safe_run.py:137-139,363-376`); **spawn_durable_daemon --stop escalates** SIGTERM→SIGKILL (`:929-954`); **fleet launcher MANDATES** every arm is wrapped `--rss-cap-mb` (`launch_mlx_witness_fleet.py:20-28,162`) → no bare arm through the sanctioned path (a WRAPPED arm cascades — proven). | **RESOLVED for sanctioned path** (LOW residual §3) |
| **M1** | MED | accurate-SHED-metric / **throttle efficacy** — SIGSTOP hit the wrapper group, not the memory-bearing trainer group (= review **F1**) | Recon (DAG "accurate-SHED-metric" + F1) | **#246 fix**: governor SIGSTOP-pauses the job's **FULL process tree** + GB-F1 true-GiB units fix (`system_memory_governor.py:34,150-151`). | RESOLVED |
| **M2** | MED | fleet-launcher **sub-floor `--min-free-gb 18`** default (below the 30 GB floor) | FOUND (DAG) | `4ed373e96`: `--min-free-gb` is now **ADVISORY-only**; the daemon `_system_admission_gate` (authoritative governor) is the single gate (`launch_mlx_witness_fleet.py:135,170-176`). | RESOLVED |
| **M3** | MED | fleet-launcher has **no `--rss-cap-mb`** (arms unbounded) | FOUND (DAG) | `4ed373e96`: fleet launcher derives per-arm `--rss-cap-mb = projected_peak × 1.3` and passes it to every spawn (`:20-28,162`). | RESOLVED |
| **M4** | MED | **admission↔registration TOCTOU** — two simultaneous launchers both pass admission before either registers → the exact **2nd-arm concurrency race** (= review **F5**) | Recon (F5) | `4ed373e96`: **fcntl PENDING admission-reservation** — admission decision + reservation write under `LOCK_EX`; a 2nd launcher blocks then sees the 1st's reservation (`spawn_durable_daemon.py:218-264`; 229-line `test_admission_reservation_toctou.py`). | RESOLVED |

**Also-closed follow-ups** (independent review): F2 `tracked_sum` double-count → #246 descendant-dedup; F4
unregistered-heavy 0-growth-headroom → conservative unknown-peak growth default (`4ed373e96`); coverage-gate
substrate-family visibility (`4ed373e96`).

**Test receipt:** `test_admission_reservation_toctou.py` + `test_system_memory_governor.py` +
`test_witness_memory_preflight.py` + `test_admission_coverage_gate.py` = **102 passed** (2.05s).

## 2. DECISIVE PROOF — concurrent-arm admission smoke (ENFORCE mode)

Two governed sentinel launches at the **real** per-arm peak (24.5 GiB, the actual A/B ask) + a 3rd sized to push
the SYSTEM sum over the ceiling. `spawn_durable_daemon.py --skip-readiness-gate --skip-blackbox-autostart`;
CONTAINMENT-safe (`sleep` sentinels, no GPU); cleaned up (`--stop` + `--reconcile`).

```
SNAPSHOT: system_used=27.0GiB available=101.0GiB ceiling=117.76GiB(adaptive)
ARM 1 (peak 24.5): ADMISSION OK — projected 51.5 <= ceiling 108.1 (headroom 56.6)      rc=0
ARM 2 (peak 24.5): ADMISSION OK — projected 75.9 <= ceiling 94.7  (headroom 18.8)      rc=0   ← 2×24.5=49 real / 75.9 projected, CONCURRENT
ARM 3 (peak 45):   REFUSED [ENFORCE] "Another concurrent job would push the SYSTEM over the ceiling"  rc=5
VERDICT: rc 0 / 0 / 5  (want 0 0 5)  ✓   teardown: both arms stopped, no orphan, registry reconciled
```

The guard **ADMITS the 2-arm concurrent A/B and REFUSES the 3rd that sums over, in ENFORCE mode.** Note the
ceiling is **dynamical** (108.1 → 94.7 as commitment grows — the #298 tier-scaled continuous-band floor), i.e.
*more* conservative than the flat 117.76 as arms are added. This is the decisive test that concurrency is safe.

## 3. Residual risk the operator should know (before firing concurrent arms)

1. **H2 defense-in-depth residual (LOW, non-blocking):** the whole-machine watchdog `memory_guard._kill_pgrp`
   is itself still SIGTERM-only. It is NOT a blocker because the sanctioned path never produces a bare arm (the
   fleet launcher mandates `--rss-cap-mb`; safe_run cascades SIGTERM→SIGKILL). I deliberately did **not** patch
   the vendored control-plane guard: any edit there requires the 3-clean-pass gauntlet
   (`memory_guard_adversarial_review_spec_20260626.md`) and risks re-introducing the molt control-plane-kill
   class — a far worse outcome than a residual that the sanctioned path already covers. **Operational rule:
   always launch arms via `launch_mlx_witness_fleet.py` or `spawn_durable_daemon --rss-cap-mb`; never a bare
   trainer.** (A future SIGKILL-escalation for `memory_guard._kill_pgrp`, mirroring spawn_durable's proven
   pattern, is a queued LOW hardening for the next guard-review cycle.)
2. **Dynamical ceiling headroom:** at 2×24.5 real arms the gate admits with ~18.8 GiB headroom (comfortable, not
   vast). A 3rd real arm would still admit (99.5 < ceiling); the operator's plan is a 2-arm A/B, so fine.
3. **NOT a #173 fleet-safety item, but gating the pointer run** (Contrarian revision, council memo): bit-identity-
   smoke the micro-batch twin at n600 before the concurrent A/B rides a pointer run. Config-correctness gate,
   orthogonal to fleet safety.

## STORES CONSULTED
- `.omx/research/council_v752_relaunch_shape_concurrent_vs_single_20260710.md` (the deferral this clears)
- `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` FEED 2026-06-26as (H1/H2 canonical) + as/at/au (memory root-cause)
- `.omx/research/memory_blackbox_and_system_governor_20260703T021505Z.md` (H1 admission gate landing)
- `.omx/research/governor_independent_review_20260703T023153Z.md` (SAFE-TO-ENFORCE; F1–F5)
- `.omx/research/n205_full_run_risk_register_watchlist_20260702.md` (D9 memory concern)
- `.omx/research/memory_guard_adversarial_review_spec_20260626.md` (#172 guard seal spec)
- commits `4ed373e96` (07-06 memory-safety fix-all: TOCTOU + fleet routing + caps), `#246`/`#298` (throttle + tier-scaled floor)
- code: `tools/{system_memory_governor,spawn_durable_daemon,safe_run,memory_guard,witness_memory_preflight}.py`, `experiments/launch_mlx_witness_fleet.py`
