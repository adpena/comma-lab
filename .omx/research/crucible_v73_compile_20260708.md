# CRUCIBLE v7.3 COMPILE — the launch-candidate config for seal round 2  [landed]

- **UTC:** 20260708 · **Authority:** `[macOS advisory]` $0, NO launch, live run + pid 63069 UNTOUCHED.
  Pointer contest-CPU **0.19110 UNMOVED** — this compile is APPARATUS/MEANS. The END is the byte-closed
  n600 exact row < 0.19110 from `upstream/evaluate.py` AFTER the run.
- **Source:** T3 inclusion-symposium synthesis `.omx/research/t5_crucible/SYNTHESIS_INCL_symposium_20260708.md`
  (§FINAL CLASSES + CRUX-ENGINEERING ADDENDUM) + ORCHESTRATION_LEDGER crux outcomes (item 4 ELEVATED by
  GPU cert · item 3 elevation REVOKED by measured falsification · item 10 stays REGISTERED).

## STORES CONSULTED
- Synthesis class table + crux addendum (the spec) · ORCHESTRATION_LEDGER last 150 (crux verdicts).
- `src/tac/witness_autoconfig.py` (`_build_crucible_v7` / `derive_/compile_crucible_v7_config` /
  `CrucibleV7LaunchConfig`).
- `src/tac/canonical_equations/safe_compile_device_bitidentity_20260708.py` (the ADMIT law
  `safe_compile_hosc_device_bitidentity_v1`: GPU max|Δ|=0, CPU 5.96e-8 REFUSE) + the live manifest
  `.omx/state/mlx_safe_compile_manifest.json` (fingerprint MATCHES this host — verified).
- `.omx/research/r7_finishers_20260708.md` (PolyakFinisher + start_epoch=0 residual) +
  `src/tac/witness_control/polyak_finisher.py` (`polyak_finisher_window_provenance`) +
  `src/tac/witness_control/tail_cycles.py` (TAIL turnpike dwell).
- `.omx/research/d16_metal_kernels_20260708.md` (the `TAC_MLX_CUSTOM_PERSISTENCE_POOL` dispatch flag +
  bit-identity parity evidence) + `src/tac/local_acceleration/scorer_throughput_gate.py` (the min/ep anchor).
- Trainer `experiments/train_levelset_witness_realized_through_R_mlx.py` (the closed-loop sync-branch
  site; the `--safe-compile-regions` / `--polyak-finisher-*` argparse).

## PER-DELTA EVIDENCE TABLE

| # | Delta | Status | Evidence / value |
|---|---|---|---|
| 1 | D16 persistence-pool dispatch ON | DONE | `TAC_MLX_CUSTOM_PERSISTENCE_POOL=1` added to `typed_config.PERF_ENV_PREFIX` (after grouped-backward). Bit-identical max\|Δ\|=0 incl. full-loss flag-on-vs-off on REAL n600 GT, N=5 (d16 memo). `_parse_perf_env` auto-derives it into `REQUIRED_PERF_ENV` ⇒ the launcher perf-env class guard REQUIRES it structurally. |
| 2 | PolyakFinisher composed, start_epoch SIZED | DONE | `PolyakFinisher(start_epoch=2545)` appended to v7 levers. DERIVED-AT-CONFIG via `crucible_v7_polyak_start_provenance` (see arithmetic below). Rides the constants_manifest as a LawRef ⇒ gate classifies it DERIVED, not NAKED. Degenerate-epochs safe (calibration). |
| 3 | safe-compile hosc flip ARMED | DONE | `base["--safe-compile-regions"]="hosc_activation"`. Law `safe_compile_hosc_device_bitidentity_v1` GPU-ADMITs at real coverage. `mlx_device="gpu"` (CPU would REFUSE). Launcher b2 = runtime authority (fingerprint fail-closed); MEASURED `fingerprint_ok=True` on this host. |
| 4 | wall_clock_budget_days re-derived from LIVE cadence | DONE | `RUN1_MEASURED_MIN_PER_EP` 3.1 → **3.62** (LIVE incl-startup, S5-H2 UPHELD), provenance flipped to config-conditional MEASURED (module constant, NOT a LawRef). Budget 3000 ep: **8.673 d** (was 7.427). rc=8 admission bench = final arbiter. |
| 5 | Items 3/6/7/10/11 default-OFF verbatim + named triggers | DONE | `crucible_v7_registered_off_levers()` (duty-to-measure surface). Item 3 trigger UPDATED per the falsification: **bounded n600 d_seg A/B ~day-1, NOT bit-identity engineering** (crux elevation REVOKED by `frozen_scorer_forward_batch_dependence_v1`). |
| 6 | 2 stale closed-loop assertions | DONE | `experiments/test_closed_loop_control.py`: `v = realized_verdict()` → `v = realized_verdict(ep=int(ep))` (a refactor added the `ep` arg; the sync-else-branch boundary token moved). Tests re-pointed, NOT deleted — the M2-wall join/decide/schedule ordering guard is still valid. 24/24 pass. |
| 7 | Verify NEW-1 / 4→5 levers / resume-registry / D7 | DONE | See VERIFY below. |

## POLYAK start_epoch DERIVATION (delta 2)
`crucible_v7_polyak_start_provenance(epochs=3000)`, law `muon_finisher_schedule_warmstart_and_lr_anneal_v1`
(finisher tail window ~0.1–0.3× the finishing stage; frac=0.2):
- finishing-stage window = `epochs − muon_cap` = `3000 − 726` = **2274** ep (post-Muon constant-τ* turnpike).
- Polyak tail window = `round(0.2 × 2274)` = **455** ep.
- relative tail start = `2274 − 455` = **1819**.
- **ABSOLUTE start_epoch = muon_cap + relative = 726 + 1819 = 2545.**

Sized off the muon CAP (the LATEST possible Muon entry — a fail-safe backstop) ⇒ the MOST CONSERVATIVE
start: always post-Muon, always inside the turnpike dwell even if the Muon EVENT fires earlier (a shorter
actual tail fraction, never a pre-Muon average). Guards `epochs ≤ muon_cap` (calibration/smoke) by
degenerating to an inert averager (`start_epoch = epochs` ⇒ never observes) so v7 stays buildable at
`--calibrate-epochs 3`.

## BUDGET RE-DERIVATION (delta 4)
`derive_wall_clock_budget_days(3000) = project_wall_clock_days(3.62, 3000) × 1.15`
= `(3.62 × 3000 / 1440) × 1.15` = `7.5417 × 1.15` = **8.673 d** (round 3).
Prior (3.1 optimistic steady-state): `6.4583 × 1.15` = 7.427 d. The live incl-startup 3.62 (ORCHESTRATION
_LEDGER H2) is the honest floor a fresh, checkpointed, resumable run pays on every epoch; the 3.1 was a
lower bound. Slack 1.15 unchanged (now a PURE thermal/jitter headroom — no longer double-counting startup).

## DRY-RUN GATE CHAIN (acceptance) — GREEN, rc=0
`python tools/launch_witness_run.py --config crucible_v7 --num-pairs 600 --dry-run --skip-mem-preflight`
(NO `--epochs`):
- `# epochs: 3000 (config-sealed default for 'crucible_v7')` — **NEW-1 resolves the sealed default.**
- `# perf-env guard: launch.sh carries ['TAC_MLX_CUSTOM_GROUPED_BACKWARD','TAC_MLX_CUSTOM_PERSISTENCE_POOL']`
- `# schedule-provenance gate: … (0 NAKED)` — **b0.5 clean.**
- `# dsl-config gate: OK — DSL-authored ('crucible_v7', 137 flags, typed-validated)` — **b0.6 VERIFIED.**
- `# safe-compile: spec='hosc_activation' … fingerprint_ok=True — fingerprint matches host` — **b2 OK.**
- launch.sh carries `--safe-compile-regions hosc_activation`, `--polyak-finisher-arm`,
  `--polyak-finisher-start-epoch 2545`, `--epochs 3000`, `TAC_MLX_CUSTOM_PERSISTENCE_POOL=1`.
- system-admission ADVISORY REFUSE (live run occupies 75.8 GiB / 5 active jobs) — dry-run does NOT enforce
  the live-memory gate ⇒ **rc=0**. rc=8 wall-clock budget DERIVED (8.673) & positive.

## VERIFY (delta 7)
- **NEW-1**: `--epochs` omitted ⇒ launcher reads `cfg.epochs` (3000). New test
  `test_dry_run_resolves_sealed_epochs_when_omitted`.
- **4→5 levers, FEED_07a present**: `dsl_levers == (seg_form_unify_tau, tail_k_warm_restart,
  n323_ladder_island_homotopy, FEED_07a_directional_basis_rebalance, R7_polyak_finisher)`. Pins updated in
  both test files.
- **Resume-registry (#358 fold)**: `test_resume_registry` green (legacy + v7-shaped sidecar round-trip;
  the 4 non-gate controllers registered). Polyak scalar rides `__pta_` under the registry (R-7 landing);
  no new trainer edit here.
- **D7 pose block VERBATIM**: `test_pose_block_verbatim_vs_v6` — `--w-pose 1.0`, store-nothing carrier
  `--pose-carrier-source generated` + `--pose-carrier-residual-mode table` all identical to v6.

## TRIALITY LEGS TOUCHED
- **DSL**: `src/tac/witness_dsl/typed_config.py` (PERF_ENV_PREFIX) + the v7 config composes the existing
  `PolyakFinisher` Lever factory (no new factory — the DSL already HELD it). Not a drift.
- **equations**: CONSUMED existing laws (`muon_finisher_schedule_warmstart_and_lr_anneal_v1`,
  `safe_compile_hosc_device_bitidentity_v1`). The budget anchor is a MODULE CONSTANT (per the task: "if a
  module constant, flip its provenance tag") — provenance flipped in-comment, NO new canonical equation.
- **DAG**: this memo is the trajectory leg.

## TESTS / FILES
- 8 files: `witness_autoconfig.py` · `witness_dsl/typed_config.py` ·
  `local_acceleration/scorer_throughput_gate.py` · `experiments/test_closed_loop_control.py` +
  4 test files (`test_crucible_v7_config.py`, `test_launch_witness_crucible_v7_resolution.py`,
  `test_v7_compute_exploitation.py`, `test_wallclock_default_on_perfenv_guard.py`).
- Full changed-test suite: 133 pass (config 46 · resolution 15 · compute-exploit 25 · wallclock/perfenv 24
  · closed-loop 24 + throughput). ruff F clean. Own hostile round-1 review caught + fixed the
  `--calibrate-epochs 3` degenerate-epochs regression (Polyak sizing now clamps, does not raise).

## NAMED RESIDUALS
- The safe-compile dry-run tests are host-conditional (skipif on manifest fingerprint) — b2 fail-closes
  off-fingerprint BY DESIGN (device-conditional cert); on this M5 Max fingerprint they run + pass.
- `crucible_v7_registered_off_levers()` is a queryable duty-to-measure surface (like
  `crucible_v7_wiring_gaps()`); the #247 costate SENSE layer is its intended consumer.
- Pointer 0.19110 UNMOVED. This is the SEAL-round-2 candidate; the run + byte-close are the next units.
