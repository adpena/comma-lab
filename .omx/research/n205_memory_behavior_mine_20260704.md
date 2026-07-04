# #205 memory-behavior mine — 28.3 h of real telemetry, config→envelope row + projection residual (#295 phase I)

**Date:** 2026-07-04 · **Task:** #295 phase I (operator 2026-07-04: "use 205 telemetry... learn about
the memory usage behaviors under different configs... useful for comma ai in showing for production
deployment on different machines") · **$0, READ-ONLY on the run** (pid 29129 untouched; log reads +
`ps` sampling only) · **Axis:** `[macOS-M5Max-128GB measurement]`, production_generalized METHOD —
no score claim, pointer 0.19110 UNMOVED.

**Run under measurement:** `experiments/results/levelset_n600_witness_20260703T120444Z`
(launch 2026-07-03T12:04:44Z; git `f41ece7f3`, seed 0; mod-32 / bank-4 / in_feat-88 / n600 /
accum-pairs 8 / async verdict-batch 32 / self-orient fp32 cf cache / MLX-GPU grouped-backward;
wrapped in `safe_run.py --rss-mb 90000`). At memo time: epoch ~505, tau_softplus stage, alive,
zero governor interventions.

---

## 1. Telemetry inventory — what EXISTS and what it EMITS

| Surface | Status in THIS run | Schema / cadence |
|---|---|---|
| **trainer `mem_probe`** (`train_levelset_witness_realized_through_R_mlx.py:3591,3789,3795,4476`) | **SILENT — env-gated OFF.** `TAC_MEM_PROBE` (default `"0"`, line 4001) not set in `launch.sh` → **zero mem_probe rows in run.log** (verified: grep of all 100 lines). `mx.reset_peak_memory()` per-epoch (line 4026) is also inside the gate. | When on: `{stage:"mem_probe", phase:(after_cf_mx_cache_build \| before_v0_verdict \| after_v0_verdict), rss_gib, mlx_active_gib, mlx_cache_gib}` + per-accum-batch rows for first `TAC_MEM_PROBE_EPOCHS=3` epochs only. |
| **`safe_run.py` wrapper** (pid 29126) | Polls group-RSS via `ps -g pgid` every **0.2 s**, tracks peak **in-memory**; emits `peak_rss_mib` ONLY at exit (line 244-257). Nothing readable mid-run. | exit line: `status= exit= peak_rss=MiB elapsed= limit_rss=90000MiB` + JSON marker. |
| **`memory_blackbox.py` daemon** (pid 19895, started 07-03T02:19Z) | **THE dataset.** 2 s cadence, fcntl-locked JSONL, 20 MB rotation. Live file + 4 archives (`.omx/state/archive/memory_blackbox_20260703T120710Z…20260704T150245Z.jsonl`) give **UNBROKEN coverage of the entire run**: 46,618 samples carrying pid 29129, span 07-03T12:04:53Z → 07-04T16:22:25Z = 28.29 h. | Per row: system (`total/used/available/free/wired/compressor/swap_used` GiB, `pressure_level`, loads, `adaptive_ceiling_gib 117.76`, `safety_margin_gib 10.24`) + `tracked[]` per job: `{label, pid, pgid, priority, projected_peak_gib, current_rss_gib, paused, throttle_eligible}`. Tracks BOTH the safe_run group (pid 29126, label `levelset_witness_…`, projected_peak_gib **67.61**) and pid 29129 itself. |
| **governor/guard logs** | `memory_blackbox_actions.log`: start line only → **zero pause/throttle actions** on this run. `memory_guard.log`: last WARN 2026-07-01 (pre-launch) → zero interventions. `memory_governor.log`: 2 accounting FAIL-SAFE rows (07-04T05:58, 06:00 — vm_stat vs sysctl free-page cross-check 2.2–2.5 GiB > 2.0 tolerance; available conservatively reduced; no action taken). | — |

**⚠ UNITS FINDING (measured, load-bearing):** `memory_guard.group_rss_gb` (tools/memory_guard.py:565-577)
returns `Σ rss_kb / 1e6` — units of 10⁶ KiB = 1.024×10⁹ B ≈ **0.9537 true GiB per unit** — but the
governor/blackbox field is NAMED `current_rss_gib`. Cross-validated live: ps 16:21:17Z → 65,217,360 KiB
= 62.19 GiB true; blackbox 16:21:31Z → "65.27" units = 62.25 GiB true (Δ 0.06 during a rising phase). The
blackbox SYSTEM fields (`total_gib: 128.0`) ARE true GiB (2³⁰) — **mixed units in one row**. All tracked
values below are converted ×0.95367 to true GiB. (Fix owned by #294 — see §7.)

## 2. RSS trajectory (all numbers MEASURED from the blackbox series unless tagged ps)

**Startup ramp:** 2.6 GiB at 12:04:55Z → crossed 10/20/30/40/50 GiB at t+0.8/1.1/1.2/1.2/1.2 min →
~52.5 GiB true by **t+1.3 min**. The n600 fp32 self-orient cf_mx_cache + GT cache materialize in ~80 s.

**Steady-state structure — a saw-tooth locked to the async-verdict cycle, NOT noise:**
- **Between verdict windows (post-warmup):** baseline **54.0–55.3 GiB true** (CE stage ≈ 55.2, tau
  stage ≈ 54.0); mean 52.5 over all between-samples (pulled down by the §3 dip); p95 55.3.
- **Inside verdict windows:** mean 60.6, p95 66.3, max **67.68 GiB true = the global peak**, at
  2026-07-04T03:49:37Z — the tail of the ep300 window, the LONGEST verdict (2568 s).
- Whole-series (46,618 samples): p50 55.2 · p95 63.7 · max 67.68 true GiB.
- The operator check-in spot-values (54.0 → 61.2 → 56.6 → 60.7 → 54.0 "GB") reproduce exactly as
  blackbox-unit readings of this saw-tooth (54.0/56.6 = between-baselines, 60–61 = in-window).

**Verdict-window signature (19 windows, every 25 epochs; window length 1932–2568 s, mean 2189 s ≈
36.5 min; duty cycle ≈ 47% of wall-clock):** per-window table, true GiB, baseline = 5-min pre-window mean:

| ep | base | peak | spike | step@start | | ep | base | peak | spike | step@start |
|---|---|---|---|---|---|---|---|---|---|---|
| 25 | 54.6 | 67.5 | +12.9 | +5.1 | | 275 | 55.3 | 67.4 | +12.1 | +4.5 |
| 50 | 55.2 | 67.1 | +11.9 | +5.4 | | 300 | 55.3 | **67.7** | +12.4 | +5.4 |
| 75 | 55.2 | 67.3 | +12.1 | +4.5 | | 325 | **33.2** | 45.3 | +12.1 | +4.5 |
| 100 | 55.2 | 67.1 | +11.9 | +5.3 | | 350 | 33.3 | 66.0 | (+32.7 = dip exit) | +5.4 |
| 125–250 | 54.6–55.3 | 67.1–67.6 | +11.9…+12.9 | +4.5…+5.4 | | 375–475 | 54.0 | 66.0–66.3 | +12.1…+12.3 | +4.5…+5.4 |

**The spike is additive and stable: +11.9 to +12.9 GiB over baseline in EVERY window, even from the
33.2 GiB dip baseline (ep325: +12.1).** Shape: an immediate **+4.5–5.4 GiB step** at window start
(verdict thread's GT batch + frozen-scorer buffers at verdict-batch 32) followed by a **monotone ~+7 GiB
climb** across the ~36 min window (buffer-pool growth while training epochs run concurrently with the
verdict thread), then release to baseline within ~2 min of `verdict_async_done`. The preflight's spike
model ("chunked → +5.6 GiB floor") matches the STEP; the intra-window climb under concurrency is the
un-modeled component (absorbed today by the projection's other margins — see §4).

**Checkpoint transients:** none observable at 2 s / 0.1-unit resolution (checkpoints are 0.47–1.8 MB npz;
atomic tmp+rename) — checkpoint writes are memory-free at this scale.

**Leak assessment (the multi-day question):** hourly p10 baselines are FLAT within each stage —
55.2 true GiB for the 14 CE hours (57.9 units, constant to the 0.1-unit quantum), 54.0 for the 9 post-dip
tau hours. **Within-stage leak slope ≈ 0.0 GiB/day.** The naive whole-series fit is −3.3 GiB/day, an
artifact of the §3 dip + a **−1.2 GiB step-DOWN at the CE→tau transition**. No monotone growth over
28 h ⇒ no evidence the multi-day completion is leak-threatened. Swap stayed ≤ 0.006 GiB and
`pressure_level` ≤ 1 (normal) for the entire run.

## 3. The transition dip — a real, reproducible-looking pool-trim event

07-04T04:09:24Z (≈15 min after the ep300/CE→tau verdict window ended; transition itself logged
03:54:08 with AdamW moment reset + spike-guard re-treat): RSS fell 57.98u → ~35u within ~1 min and
sat at **33.2–33.3 GiB true for 2.45 h** (dip p50 33.3, min 33.25; excursions to ~42–50 = the ep325
verdict riding the reduced baseline). During ep350's verdict window (06:27–07:03) the pool re-grew and
**stayed** at the new 54.0 GiB tau baseline. Mechanism candidates (NOT resolvable from outside the
process — exactly what phase-II's `mlx_active/cache` split disambiguates): (a) `mx.clear_cache()`
paths active around the stage transition returning the MLX buffer pool to the OS, (b) macOS reclaiming
cold pool pages post-moment-reset. Two production-relevant implications: **(i)** the working set is
~20 GiB smaller than steady-state RSS suggests — steady RSS includes a re-growable pool; **(ii)** a
deployment sizing from "observed steady RSS" alone would over-provision; sizing from peak (§4) is the
correct envelope.

## 4. The config→envelope ROW (the #294 ledger's first calibration point)

`.omx/state/memory_projection_ledger.jsonl` did NOT exist at memo time → row carried here for
reconciliation (schema mirrors the governor's tracked-row + preflight vocabulary):

```json
{"kind": "measured_actual", "run": "levelset_n600_witness_20260703T120444Z",
 "config": {"mod_dim": 32, "in_feat": 88, "bank": 4, "n_pairs": 600, "accum_pairs": 8,
            "verdict_batch": 32, "async_verdict": true, "self_orient_cache": "fp32",
            "device": "mlx-gpu", "grouped_backward": true, "micro_batch": "serial"},
 "measured": {"window_h": 28.29, "epochs_covered": "0-505 (CE 0-299, tau 300-505)",
              "steady_between_rss_gib": 54.0, "steady_between_band_gib": [54.0, 55.3],
              "in_verdict_mean_gib": 60.6, "peak_rss_gib": 67.68,
              "peak_at": "2026-07-04T03:49:37Z (ep300 verdict tail)",
              "verdict_spike_step_gib": 5.0, "verdict_spike_total_gib": 12.3,
              "spike_sources": "verdict-thread step (+4.5-5.4) + concurrent-training pool climb (~+7)",
              "phase_step_ce_to_tau_gib": -1.2, "transition_dip_min_gib": 33.25,
              "leak_slope_within_stage_gib_per_day": 0.0, "swap_max_gib": 0.006,
              "pressure_level_max": 1, "governor_interventions": 0,
              "startup_ramp": "2.6->52.5 GiB in 80 s",
              "safe_run_peak_vs_cap": "69,336 MiB of 90,000 MiB cap = 77%"},
 "projection": {"preflight_projected_peak_gib": 67.61, "source": "governor tracked-row + witness_memory_preflight"},
 "residual": {"measured_minus_projected_gib": 0.07, "pct": 0.10,
              "status": "INTERIM — Muon (ep726) + l7 (ep1000) stages pending; safe_run exit peak (0.2s poll) is the final authority"},
 "units_note": "governor tracked current_rss_gib is KiB/1e6 (x0.9537 true GiB); all values above converted to true GiB",
 "provenance": {"blackbox_files": ["archive/memory_blackbox_20260703T120710Z.jsonl", "…T211302Z", "…T054845Z", "…T150245Z", "memory_blackbox.jsonl"],
                "n_samples": 46618, "ps_crosscheck": "62.19/62.40/62.64 GiB @16:21-16:26Z vs blackbox 62.25", "git_sha": "f41ece7f3"}}
```

**Projection residual: measured true peak 67.68 vs projected 67.61 = +0.07 GiB (+0.10%)** — an
unusually tight first calibration point, with the honest decomposition that the model's verdict-spike
FLOOR (+5.6) underestimates the realized +12.3 total spike while its baseline terms overestimate by a
compensating amount; the NET is accurate at this config but the components should not be trusted
independently (phase-II split telemetry fixes this). INTERIM: two stage transitions remain.

## 5. Production-deployment framing (the comma.ai asset; production_generalized)

The METHOD demonstrated here is clip-agnostic and contest-free: **(1)** an always-on 2 s system
blackbox (rotated JSONL, crash-forensics-capable, per-job tracked rows with projected-vs-current) +
**(2)** a pre-launch analytic projection (`witness_memory_preflight`: per-component GiB model +
safe-frac refuse line) + **(3)** a runtime governor (admission gate / throttle; zero interventions
needed here) + **(4)** post-hoc mining (this memo) that closes the loop with a measured
**config → memory-envelope row** and a projection residual. For deployment sizing on heterogeneous
machines (edge tiers, comma-device class, workstations): each candidate config contributes one row
{config knobs → steady band, peak, spike signature, leak slope}; the family of rows IS the sizing
curve (RAM tier ↔ admissible {n_pairs, verdict_batch, cache precision, accum} envelope), and the
residual column continuously calibrates the analytic model so NEW configs can be sized without
running them. Portable findings from this row: peak is verdict-(inference-)driven, not
training-driven (+12 GiB, 47% duty cycle — schedule inference bursts against RAM tier); steady RSS
overstates the working set by ~20 GiB of re-growable pool (size to peak, reclaim is real); zero swap /
zero pressure escalation at 53% median RAM utilization on a 128 GB box. **Honest scope: ONE config
measured = ONE point, not a curve.** The curve family accrues from the campaign's subsequent runs
(and phase-II per-knob telemetry); no cross-config interpolation is claimed from this memo.

## 6. Phase-II trainer-telemetry spec (paste-in when #293 frees the file; NO edit now)

The code already exists env-gated; phase II = make it default-on, verdict-cadenced, and split-aware:

- **Row schema:** `{"stage":"mem_probe", "epoch":E, "phase":<seg_form>, "rss_gib":<true GiB, psutil>,
  "mlx_active_gib":…, "mlx_peak_gib":…, "mlx_cache_gib":…, "in_verdict":<bool>, "ts":ISO}`
  (adds `mlx_peak_gib` + `phase` + `in_verdict` to the existing 3-field probe; `_mlx_mem_gib` at
  line ~171 already wraps active/cache/peak).
- **Cadence:** (a) once per `--eval-every` boundary IMMEDIATELY BEFORE spawning the async verdict and
  once at `verdict_async_done` (brackets the spike); (b) once per stage transition, before AND after
  the moment-reset/clear_cache block (resolves the §3 dip mechanism definitively); (c) keep the
  existing after_cf_mx_cache_build + v0 probes; (d) keep `mx.reset_peak_memory()` per epoch so
  `mlx_peak_gib` is epoch-scoped. Cost: ~6 rows/hour — negligible.
- **Wiring:** flip the gate default (`TAC_MEM_PROBE` default "1", or set it in
  `tools/launch_witness_run.py`'s emitted launch.sh alongside `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1`) so
  the next launch carries telemetry without operator memory (this run's silence was exactly a
  default-off gate being forgotten).
- **What it adds over the blackbox:** the RSS-vs-MLX-pool split (active/cache/peak) that no external
  sampler can see — turning §3's "candidate mechanisms" and §4's compensating-component caveat into
  measured attributions, and giving per-knob curves (verdict_batch, cache precision, accum) for the
  §5 sizing family.

## 7. Flags for #294 (tool surface — NOT edited here)

1. **Units:** `memory_guard.group_rss_gb` returns KiB/1e6 in a field named `_gib` (§1) — rename or
   divide by 1024² so governor rows are true GiB; until then downstream consumers must apply ×0.9537.
2. **Double-count:** blackbox `tracked_sum_gib` reached 130.55 while the box holds 128 — the safe_run
   GROUP row (pid 29126) and the pid29129 row are the same memory counted twice once both are tracked.
3. **Ledger:** `memory_projection_ledger.jsonl` absent — §4's row is the seed; schema proposed there.

## 8. Honest gaps

- run.log carries NO memory rows (gated off) — the blackbox + ps ARE the dataset; that dataset is
  2 s-resolution, so sub-2 s transients (MLX lazy-graph spikes) may exceed the observed 67.68 peak;
  safe_run's 0.2 s-poll `peak_rss` at process exit is the finer-grained authority — reconcile then.
- Residual is INTERIM (Muon ep726 / l7 ep1000 pending; Muon adds optimizer state).
- Blackbox coverage starts 9 s after launch (12:04:53 vs 12:04:44) — the ramp's first seconds and
  any pre-blackbox transient are unobserved (nothing suggests one).
- Dip mechanism is attributed to candidates, not proven (phase-II resolves).

Pointer 0.19110 UNMOVED — this memo is MEANS (apparatus/measurement), no score claim.
