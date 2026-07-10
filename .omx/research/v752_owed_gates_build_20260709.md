# v7.5.2 owed-gates BUILD — task #384 — 2026-07-09

**Agent:** #384 BUILD (operator GO 2026-07-09 "any necessary follow-up work... new building and wiring and
integrating as well necessary for optimal v7.5.2"). **Surface:** `crucible2_v752`. `[macOS-MLX/CPU advisory]`
NON-PROMOTABLE MEANS — pointer contest-CPU **0.19110 UNMOVED**. Only a byte-closed `upstream/evaluate.py`
n600 row < 0.19110 moves it. `$0`, no paid GPU, no real multi-hour launch, run dirs READ-ONLY.

**STORES CONSULTED:** `.omx/research/t5_crucible2/SYNTHESIS_v3_v752_20260709.md` (§C owed-before-launch list +
§C-header legacy owed-N reconciliation map + §A.4/§A.5/§B) · `docs/operating_manual_craft_handoff.md` ·
`tools/launch_witness_run.py` (the governed launcher gate chain) · `tools/witness_memory_preflight.py` ·
`tools/safe_run.py` (admission gate + override) · `src/tac/witness_control/{sigma_min_plateau,verdict_trend_alarm,
jacobian_basin}.py` (the sibling-landed observer stack) · `src/tac/witness_autoconfig.py`
(`compile_crucible_v7_config`) · the #205-lineage stopped run
`experiments/results/levelset_n600_witness_20260709T105312Z` (READ-ONLY replay fixture) · MEMORY L70/L45/L84.

## Scope (4 gates; owed-1 pose-gate is a SIBLING agent's job — NOT touched)

| owed | gate | status |
|---|---|---|
| **owed-2 / item 2** | Full-config DRY-START | **BUILT + measured (peak RSS)**; sec/ep measurement long-running |
| **owed-4 / item 3** | Speed-stack composition + wall-clock budget | **BUILT + GREEN** |
| **owed-14 / item 14** | Governed telemetry replay (observer stack) | **BUILT + GREEN (all 4 legs)** |
| **owed-15 / item 15** | Class-A fresh-arm isolation LADDER (configs only) | **BUILT + validated (3 arms), NOT trained** |

## Gate 1 — Full-config DRY-START (`tools/launch_witness_run.py --dry-start N`)

New launcher mode `--dry-start N` (N ≤ 3). Runs the ENTIRE gate chain on the REAL n600 config
(flag-validate → launch.sh → perf-env guard → constants → schedule-provenance → DSL-config gate →
memory-preflight → safe-compile freshness → system-admission → throughput), then — INSTEAD of the unbounded
durable spawn — executes a **wall-clock-bounded governed run of the EXACT REAL launch.sh** (unmodified real
schedule/caps/levers) via `safe_run`, proving BOOT + model build + STEP + a written resume ckpt, then a
**RESUME round-trip** (`--resume-from` the written dir), then EXITS cleanly (NEVER the real launch). Report →
`dry_start_report.json` (peak RSS + sec/ep gross+marginal).

**Design decision (measured, load-bearing):** crucible_v7 pins an ATOMIC 3000-epoch schedule whose interlocking
validators (`1 ≤ *-start-epoch ≤ epochs`; the LADDER↔Muon stage-STAGGER: `max(arm windows) < muon_start`)
a shrunk-epochs smoke CANNOT satisfy — a 3-epoch config is REFUSED by the trainer (measured: `--muon-start-epoch
726 must be in [1, --epochs 3]`, then `LADDER↔Muon STAGGER VIOLATION lane window 340 ≥ muon_start 3`). So the
bound is **wall-clock, not epochs**: run the real config, `safe_run --timeout` SIGTERMs it at the budget — which
DOUBLES as a crash simulation, making PASS-2's resume round-trip a genuine crash-resumability test. (A schedule
monotone-clamp was prototyped and REJECTED — the interlocking stagger constraints cascade; schedule surgery is a
rabbit hole and would no longer be "the real config".)

**MEASURED (first attempt, PASS-1):** **peak RSS = 70.94 GiB** — matches the memory-preflight projection
**71.54 GiB** (fixed 15 + cf_mx_cache 47.13 + gt 3.41 + verdict 6.0) within ~0.8%; the memory dimension of
start-ability is VALIDATED at the real n600 config. system-admission ADMITTED on the re-run (used ~31 / available
~97 / ceiling ~106 GiB). The first n600 epoch's one-time compile (safe-compile hosc_activation + fused-R + the
full accum loop) exceeds ~9.5 min, so a longer-budget re-run (`--dry-start-boot-budget-s 900
--dry-start-per-ep-budget-s 120`, ~21 min/pass) is in flight to capture steady-state sec/ep + the resume
round-trip. **The gate MECHANICS are proven** (full chain runs, governed bounded run executes, peak measured,
FAILED reported honestly when no epoch completes in-window).

**Governed-launcher discipline honored:** routed through `tools/launch_witness_run.py`; the inner `safe_run`
carries the launcher's `--admission-override-rationale` so its own admission gate honors the same operator
decision; `--rss-cap-mb 80000` hard-kill backstop. The one governed override used earlier was a marginal
fail-safe-reserve conservatism (1.2–6.2 GiB over ceiling) with ~94 GiB genuinely free and only `memory_blackbox`
(score-neutral) as the "active job" — the re-run self-ADMITTED without override.

**Join point (precise, verified 2026-07-09):** P7 HAS landed the typed `crucible_v752`
(`witness_autoconfig.derive_crucible_v752_config` / `compile_crucible_v752_config`), but it returns a bare
`(typed, argv, dsl_manifest)` tuple and DELIBERATELY carries **NO launcher adapter** ("P8-wall item: the
dual-chain wall means NO launch fires when crucible-2 seals"). The launcher-facing `CrucibleV752LaunchConfig`
adapter (mirroring `CrucibleV7LaunchConfig`: emit surface + the 3 provenance manifests + schedule-governance-dict
+ `.to_launch_config()`) **+ a `crucible_v752` branch in `derive_named_config`** is the exact P8 item. All four
tools call `derive_named_config(config, …)`, so `--config crucible_v752 --dry-start 3` (and the speed-audit /
isolation-arm base) consume the real v7.5.2 config with ZERO tool change the moment that adapter+branch land.
I did NOT build the adapter (P7 scoped it out for the dual-chain sequencing; building it here would risk a
P8/sibling collision). Until then the gates ride crucible_v7 (the current launchable v7.5.2 substrate).

## Gate 3 — Speed-stack + wall-clock (`tools/witness_speed_stack_audit.py`)

Audits the emitted config against SYNTHESIS §B `speed:`. **All 5 levers composed OK:** fused-R (bit-exact
fixed-order VJP, L70) ON · grouped-backward (env `TAC_MLX_CUSTOM_GROUPED_BACKWARD=1`, ~17×, L45) ON ·
safe-compile-regions=`hosc_activation` (per-chip fingerprint-certified) ON · async-verdict (neutral-by-
construction, advisory-only) ON · **micro-batch-pairs EXCLUDED-WITH-REASON** (#313 batch-dependence 2.26e-2
drift/11 argmax flips → bit-identity-at-speedup impossible). Each row carries its neutrality receipt.

**Wall-clock budget (42 s/ep anchor; re-computes from the dry-start's MEASURED sec/ep via
`--dry-start-report`):** 3 main stages × clamp[150,400] + ~11 min GPU head-solve + 50–150-ep pose →
**lo/nom/hi = 6.02 / 10.1 / 15.93 h** + head-solve. Matches SYNTHESIS "~6–16 h". This is the v7.5.2 half of the
#385 dual-chain comparison brief (§D).

## Gate 4 — owed-14 governed telemetry replay (`tools/witness_observer_replay.py`)

Replays the observer stack against the #205 stopped run's REAL telemetry (473 run.log rows / 31 σ_min points /
5 verdict rows / 42 costate rows), READ-ONLY. **ALL 4 LEGS PASS:**
- **pose-gate negative control** (the load-bearing leg, SYNTHESIS §A.4 v3 AMENDMENT-2): the sibling's
  `evaluate_plateau` detector returns **DEGENERATE_GUARD_TRIPPED, fired=False** on the rising/oscillating σ_min
  (0.0844→0.1061) — it correctly does NOT fire and identifies the CV-0.21 oscillation as degenerate (→ ship
  banked R1), exactly matching P4-M1. The $0 negative control is satisfied.
- **verdict-trend**: fires **TRAIN_VERDICT_DECOUPLING** — matching WHY #205 was stopped (d_seg plateaued ~0.033
  while d_pose blew up 7.8→24.5).
- **costate-shadow**: parses 42 rows, exposes `[lambda_d_seg, lambda_bytes, lambda_d_pose, dS_rollback_to_best]`.
- **disengaged-alarm**: builds a valid `pose_finish_disengaged_shipped_banked_r1` confound_alarm row (never
  silent). `--json` machine-readable.

## Gate 5 — owed-15 Class-A isolation ladder (`tools/build_v752_isolation_arms.py` + manifest + runbook)

Emits + argparse-validates (never-invent-a-flag) THREE fresh incremental-TRAINING arm configs, **NOT trained**
(multi-hour arms await the operator's which-to-run GO):
- **arm1_basis_only** — self-orient ON, taper OFF, `--render-aa none` (incremental baseline)
- **arm2_plus_taper** — + #121 d_seg-aware taper (isolates the taper delta vs arm1)
- **arm3_plus_aa_ipe** — + `--render-aa ipe` (isolates the AA-ipe delta vs arm2 = the full trunk)

**R8 law honored:** each is a FRESH incremental-training arm, NOT an inference-toggle of a trained-WITH render
lever (#121 taper + `--render-aa` reshape the render path → the weights adapt → a one-ckpt toggle mismatches the
render = a toy isolation). ROLLBACK sign-test: keep a lever IFF its isolated n600 through-R d_seg IMPROVES over
the arm below; else drop from the trunk. Manifest: `experiments/results/v752_isolation_arms/manifest.json`;
runbook: `.omx/research/v752_owed15_isolation_runbook_20260709.md`. Base = crucible_v7 (P7 typed crucible_v752 =
the DSL join point).

## What remains
- Dry-start **sec/ep** measurement (longer-budget re-run in flight); fold into `dry_start_report.json` +
  re-run `witness_speed_stack_audit.py --dry-start-report <dir>` to refresh the wall-clock table with the
  MEASURED (not anchor) sec/ep.
- owed-1 pose-gate BUILD is the SIBLING agent's (the detector `sigma_min_plateau.py` is landed and the replay
  harness already exercises it as the negative control).
- The 3 isolation arms + the real v7.5.2 launch await the operator which-to-run GO (dual-chain wall, #385).
- P7 typed `crucible_v752` config: all four tools consume it at the documented join points once it lands.

## Triality
- **DSL:** launcher levers unchanged (dry-start is apparatus, not a trainer lever); the isolation arms ride
  crucible_v7's existing DSL flags with documented deltas → typed WitnessProgram variants when P7 lands.
- **DAG:** FEED-owed-gates appended to `.omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.
- **equations:** N/A — apparatus/gates, no new law (the speed neutrality receipts + the σ* advisory law cite the
  existing `sigma_min_plateau` / MEMORY anchors).
