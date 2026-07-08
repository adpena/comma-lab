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

## MODE DECISION RECORD (SEAL v7.3 round-2 A4 — operator EVENT-mode override, verbatim)

The v7 launch candidate emits `--tau-advance-mode event`, DIVERGING from the round-1 3×-convergent
CLOCK-for-run-1 recommendation (event couples 3 schedules to a never-run sensor → confounds the
unify-L_τ attribution). This is a RECORDED operator override, NOT a silent choice:

> **Operator, 2026-07-08 08:45 (ORCHESTRATION_LEDGER, verbatim):** *"We want to transition to event
> based now and accept the risk, this is a new baseline, not clean but we are choosing to make a leap
> forward and accept the related uncertainty"* + *"Your rec regarding the basis is approved"*.

**Risk framing (binding):** v7 is a NEW BASELINE, not an A/B arm — the operator KNOWINGLY trades
clean single-variable attribution for the leap. No reader may grade the v7 trajectory as an isolated
unify-L_τ measurement; v7-vs-run-1 differences are the COMPOSED stack, attributed ONLY via per-stage
checkpoints + would-fire (`cap_fired_before_event`) telemetry. The deep-math BLOCKER that hid under
the mode question (`hosc_beta_end` freezing β≈10 in event mode) is FIXED (A1) → event mode is now
coherent. **Clock-revert recipe (two-token, deep-math MAJOR-2):** reverting to clock is NOT one token
— it needs BOTH the mode AND the shape: `--tau-advance-mode clock` AND `--tau-anneal-shape cosine_hold`
(geometric-clock reaches τ=0.31 only at the denominator end → no pre-Muon turnpike; the incumbent v6
turnpike is `cosine_hold`-specific). The VERIFIED geometric-hold alternative — confirm the trainer's
geometric path honors `--tau-hold-frac` before offering geometric-clock as "byte-identical." The full
launch package is `.omx/research/t5_crucible/LAUNCH_PACKAGE_v7_20260708.md`.

## PER-DELTA EVIDENCE TABLE

| # | Delta | Status | Evidence / value |
|---|---|---|---|
| 1 | D16 persistence-pool dispatch ON | DONE | `TAC_MLX_CUSTOM_PERSISTENCE_POOL=1` added to `typed_config.PERF_ENV_PREFIX` (after grouped-backward). Bit-identical max\|Δ\|=0 incl. full-loss flag-on-vs-off on REAL n600 GT, N=5 (d16 memo). `_parse_perf_env` auto-derives it into `REQUIRED_PERF_ENV` ⇒ the launcher perf-env class guard REQUIRES it structurally. |
| 2 | PolyakFinisher composed, start_epoch SIZED | DONE | `PolyakFinisher(start_epoch=2546)` appended to v7 levers (A3 off-by-one fix: 2545→2546; averages EXACTLY 455 ep). DERIVED-AT-CONFIG via `crucible_v7_polyak_start_provenance`. Rides the constants_manifest as a LawRef ⇒ gate classifies it DERIVED, not NAKED. Degenerate-epochs GENUINELY inert (A3: start=epochs+1). |
| 3 | safe-compile hosc flip ARMED | DONE | `base["--safe-compile-regions"]="hosc_activation"`. Law `safe_compile_hosc_device_bitidentity_v1` GPU-ADMITs at real coverage. `mlx_device="gpu"` (CPU would REFUSE). Launcher b2 = runtime authority (fingerprint fail-closed); MEASURED `fingerprint_ok=True` on this host. |
| 4 | wall_clock_budget_days re-anchored to AMORTIZED cadence | DONE (A2) | `RUN1_MEASURED_MIN_PER_EP` 3.62 → **3.39** (startup-amortized 3000-ep cadence; re-derived from run-1's log — steady r_ss 3.37 measured, NOT the 3.1 lower bound the deepmath's 3.12 used). Budget 3000 ep: **8.122 d** (was 8.673). Refuse is now a TRUE ~15% gate. rc=8 admission bench = final arbiter. |
| 5 | Items 3/6/7/10/11 default-OFF + named triggers (+A5/A6 arms) | DONE | `crucible_v7_registered_off_levers()`. Item 3 trigger = bounded n600 d_seg A/B (crux REVOKED). **A5 counter-arm** `lane_carried_basis_regime` (freq_along≈26 + restore lane recall; trigger = Road↔Lane jitter) + **A6 fallback** `road_boundary_fallback` (Road-first term / Menon-offset audit; trigger = Road flip >0.30 @ep200) registered duty-to-measure. |
| 6 | 2 stale closed-loop assertions | DONE | `experiments/test_closed_loop_control.py`: `v = realized_verdict()` → `v = realized_verdict(ep=int(ep))`. Tests re-pointed, NOT deleted. 24/24 pass. |
| 7 | Verify NEW-1 / 4→5 levers / resume-registry / D7 | DONE | See VERIFY below. |
| **B1** | **event-mode `hosc_beta_end` (round-2 BLOCKER, A1)** | **DONE** | `base["--hosc-beta-end"]` 10.0 → **3.177** = the control's frozen β(726). The clock-endpoint 10.0 FROZE β≈10 under the EVENT octave-fraction driver (forbidden tanh-saturation regime; invalidates every β≈3.18-measured anchor). Event-mode frozen β = β_end, so β_end IS the frozen value. ≤4.0 divergence-bound; RE-DERIVE trigger honored (provenance flip clock→event). |
| **M1** | **lane-regime coherence (round-2, A5)** | **DONE** | `base["--persistence-classes"]` 'auto' → **'3'** (movable only) DERIVED from the basis regime via `persistence_classes_for_basis_regime('lane_offloaded')`. lane rides the FREE analytic band; the frequency-starved (freq_along=6) learned render no longer chases the unsatisfiable ~25-cyc lane recall. LADDER amplify already per-class-λ self-gates. |
| **R3** | **per-group grad-clip ON (round-2, A7)** | **DONE** | `base["--per-group-grad-clip"] = True` — bounds the ep1 gnorm_hijack (island_amplify ~20% of ep1 loss) so it can't starve the seg gradient during the Road-forming window. Requires `--grad-clip>0` (base 1.0). Test asserts present in argv. |
| **A8** | **tail-stop s*=ν·forfeit reactivation extended** | **DONE** | `tail_stop_forfeit_floor_20260708.py`: reactivation_criteria + config_conditional now flag that the −48% DirectionalBasisRebalance changes the d_seg descent rate, so ν(tau_softplus)=0.012653 (fit on the OLD starved basis) is a STALE floor input → re-measure ν on the rebalanced basis. |

## POLYAK start_epoch DERIVATION (delta 2)
`crucible_v7_polyak_start_provenance(epochs=3000)`, law `muon_finisher_schedule_warmstart_and_lr_anneal_v1`
(finisher tail window ~0.1–0.3× the finishing stage; frac=0.2):
- finishing-stage window = `epochs − muon_cap` = `3000 − 726` = **2274** ep (post-Muon constant-τ* turnpike).
- Polyak tail window = `round(0.2 × 2274)` = **455** ep.
- relative tail start = `2274 − 455` = **1819**.
- **ABSOLUTE start_epoch = muon_cap + relative = 726 + 1819 = 2545.**

Sized off the muon CAP (the LATEST possible Muon entry — a fail-safe backstop) ⇒ always post-Muon, always
inside the turnpike dwell. **v7.3 round-2 MINOR-2 off-by-one fix (A3):** the ABSOLUTE start is
`epochs − window + 1` = `3000 − 455 + 1` = **2546** (was 2545), so the inclusive trainer loop
`[start, epochs]` averages EXACTLY `window` = 455 epochs (the prior 2545 observed 456 — inclusive-final
fencepost). Guards `epochs ≤ muon_cap` (calibration/smoke) by degenerating to a GENUINELY INERT averager (verdict_scope: instance — the calibration/smoke guard path only; INERT here is the DESIGNED no-op, not a lever verdict)
(**A3 MINOR-1 fix:** `start_epoch = epochs+1`, strictly beyond the final loop epoch ⇒ observe never fires
⇒ count 0; the prior `start_epoch=epochs` observed ONCE at the final epoch, NOT inert) so v7 stays
buildable + byte-identical at `--calibrate-epochs 3`.

## BUDGET RE-DERIVATION (delta 4 — SEAL v7.3 round-2 A2 amortized re-anchor)
`derive_wall_clock_budget_days(3000) = project_wall_clock_days(3.39, 3000) × 1.15`
= `(3.39 × 3000 / 1440) × 1.15` = `7.0625 × 1.15` = **8.122 d** (round 3).
The anchor `RUN1_MEASURED_MIN_PER_EP` is re-anchored 3.62 → **3.39** = the STARTUP-AMORTIZED 3000-ep
cadence (was 3.62 = the ep~46-115 incl-startup rate, which OVER-counts the one-time startup S for a
3000-ep run: `cadence(n) = r_ss + S/n` falls as n grows). RE-DERIVED DIRECTLY from run-1's log (this
session, verify-by-re-deriving): launch 09:57:30Z; startup(launch→ep0)=24.4 min; ramp-inclusive
S(two-point fit @ep100)=59.5 min; steady r_ss(ep75→100 slope)=(396.62−312.33)/25=**3.37** min/ep (the
MEASURED steady rate, NOT the memo's optimistic r_ss=3.1 lower bound the deepmath's 3.12 rested on);
amortized(3000)=3.37+59.5/3000=**3.39**. Result: the refuse is now a TRUE ~15% gate (`3.39·1.15=3.90`
= 15% above the honest cadence) instead of the ~23% gate the un-amortized 3.62 gave. Reconciles bugs
REVISE-4 (the 3.65-vs-3.62 figures were BOTH early-epoch incl-startup rates; superseded by 3.39). Slack
1.15 unchanged (now a genuinely PURE thermal/jitter headroom — startup is amortized IN the anchor).
REASONED DEVIATION from the synthesis's 3.12: run-1's own measured steady slope contradicts the 3.1
lower bound; the value-provenance ladder forbids anchoring on a bound.

## DRY-RUN GATE CHAIN (acceptance) — GREEN, rc=0
`python tools/launch_witness_run.py --config crucible_v7 --num-pairs 600 --dry-run --skip-mem-preflight`
(NO `--epochs`):
- `# epochs: 3000 (config-sealed default for 'crucible_v7')` — **NEW-1 resolves the sealed default.**
- `# perf-env guard: launch.sh carries ['TAC_MLX_CUSTOM_GROUPED_BACKWARD','TAC_MLX_CUSTOM_PERSISTENCE_POOL']`
- `# schedule-provenance gate: … (0 NAKED)` — **b0.5 clean.**
- `# dsl-config gate: OK — DSL-authored ('crucible_v7', 138 flags, typed-validated)` — **b0.6 VERIFIED** (138 = 137 + `--per-group-grad-clip` A7).
- `# safe-compile: spec='hosc_activation' … fingerprint_ok=True — fingerprint matches host` — **b2 OK.**
- launch.sh carries `--safe-compile-regions hosc_activation`, `--polyak-finisher-arm`,
  `--polyak-finisher-start-epoch 2546`, `--epochs 3000`, `--hosc-beta-end 3.177`,
  `--persistence-classes 3`, `--per-group-grad-clip`, `TAC_MLX_CUSTOM_PERSISTENCE_POOL=1`.
- system-admission ADVISORY REFUSE (live run occupies 75.8 GiB / 5 active jobs) — dry-run does NOT enforce
  the live-memory gate ⇒ **rc=0**. rc=8 wall-clock budget DERIVED (8.122, A2 amortized) & positive.

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
- Round-1 files: `witness_autoconfig.py` · `witness_dsl/typed_config.py` ·
  `local_acceleration/scorer_throughput_gate.py` · `experiments/test_closed_loop_control.py` + 4 test files.
- **Round-2 (SEAL A1-A8) additionally touched:** `witness_dsl/curriculum_dsl.py` (persistence_classes_for_
  basis_regime + DirectionalBasisRebalance coupling doc) · `canonical_equations/tail_stop_forfeit_floor_
  20260708.py` (A8 reactivation) · `t5_crucible/LAUNCH_PACKAGE_v7_20260708.md` (new; A4/A6).
- Full changed-test suite (round-2): 450 pass across config/resolution/compute-exploit/wallclock/perfenv/
  throughput/typed-config/canonical-equations. ruff F clean. Dry-run gate chain GREEN rc=0. (PRE-EXISTING,
  NOT from this wave: `test_check_344_canonical_equation_referenced` live-repo guard flags 501 legacy
  memos — strict-flip purgatory; my two docs are NOT in the flagged set and both cite canonical equations.)

## NAMED RESIDUALS
- The safe-compile dry-run tests are host-conditional (skipif on manifest fingerprint) — b2 fail-closes
  off-fingerprint BY DESIGN (device-conditional cert); on this M5 Max fingerprint they run + pass.
- `crucible_v7_registered_off_levers()` is a queryable duty-to-measure surface (like
  `crucible_v7_wiring_gaps()`); the #247 costate SENSE layer is its intended consumer.
- Pointer 0.19110 UNMOVED. This is the SEAL-round-2 candidate; the run + byte-close are the next units.
