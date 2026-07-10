# owed-1 REPAIRED POSE-ENGAGEMENT GATE — build landing (task #383, 2026-07-09)

**Agent:** owed-1 BUILD (operator GO "Proceed with repaired pose gate build"). **Surface:** `crucible2_v752`.
**$0 · no GPU · no training · run dirs READ-ONLY · #205 UNTOUCHED.** Pointer contest-CPU **0.19110 UNMOVED** —
this is a MEANS (apparatus/lever build); only a byte-closed `upstream/evaluate.py` n600 row < 0.19110 moves it.

## Answer-first
The pose-finish **conditioning gate** is BUILT, wired, DSL-held, resume-registered, and canary-validated at $0.
Pose engages on a d_seg-CONDITIONING EVENT (SEALED rolling-slope σ_min plateau), never an epoch — realizing the
operator binding. Default is byte-identical incumbent (`--pose-finish-engage-on muon`); the σ_min gate is an
opt-in DEFAULT-OFF lever. 25 tests pass. **What remains before the gate is TRUSTED on a real run: owed-14** (a
governed telemetry-ON replay to backtest the detector on REAL σ_min) — explicitly NON-blocking (ships banked R1).

## STORES CONSULTED
- `docs/operating_manual_craft_handoff.md` (the craft bar).
- `.omx/research/t5_crucible2/SYNTHESIS_v3_v752_20260709.md` **§A.4** (the requirements doc — transcribed
  faithfully: PRIMARY rolling-slope on de-noised σ_min; σ* DEMOTED to advisory; hysteresis 3; $0 canaries;
  never-block ship-banked-R1) + §A.5 (W_settle=4.6·τ_EMA=2.6→⌈⌉=3) + §B pose block (engage_trigger, sigma_min_denoise,
  noise_fit_quality_guard, sigma_star_advisory_only, hysteresis, positive_control_canary, canary_fail_branch).
- `.omx/research/t5_crucible2/ORCHESTRATION_LEDGER.md` §205-211 (operator binding verbatim) + §287-313.
- Siblings: `src/tac/witness_control/{jacobian_basin,verdict_trend_alarm,resume_registry}.py` (σ_min source,
  detector style, Resumable protocol) + `costate_estimator` (DEFAULT_STALL_REL_EPS / slope_with_stderr / MIN_ROWS_FOR_SLOPE).
- Trainer `experiments/train_levelset_witness_realized_through_R_mlx.py` (pose-finish engage machinery L8676;
  jacobian_basin T1 emit; argparse) + `curriculum_dsl` / `lever_registry` / `activation_ledger` (DSL leg) +
  `tools/costate_digest.py` (digest surface) + `test_feed07_dsl_wirein.py` (the DSL-test pattern).

## What was built
1. **`src/tac/witness_control/sigma_min_plateau.py`** (detector core, PURE math + Resumable):
   - PRIMARY: `evaluate_plateau` — rolling-mean-slope ≈ 0 over a settle window on the EMA-DE-NOISED σ_min series;
     fires when the smoothed series STOPS TRENDING for `hysteresis` consecutive flat+NON-RISING windows.
   - NOISE/FIT-QUALITY GUARD: the latest window's fitted rel-stderr must resolve to within the flat band, else
     DEGENERATE → `should_ship_banked_r1()` (never fire on a signal we can't distinguish from oscillation).
   - σ* = √(C/(δ_seg·λ_min(F))) as `sigma_star_advisory` — sideband ONLY, never a firing conjunct (A-1).
   - `SigmaMinPlateauDetector` (Resumable): observe (idempotent on non-increasing epochs) / verdict / monotone
     `latch_if_fired` / `state_arrays`+`restore_from_cfg` (persists σ_min series + fire latch).
   - $0 canaries: `synthetic_rising_series` (NEGATIVE, reproduces P4-M1 CV-0.21 rising signature),
     `synthetic_plateau_series` (POSITIVE, relaxation→flat), `canary_suite` (pass iff neg-not-fire ∧ pos-fire),
     `load_sigma_min_series_from_jsonl` (READ-ONLY real backtest helper).
   - Telemetry/alarms: `gate_observer_row` (default-ON observability), `disengaged_alarm_row` (LOUD confound_alarm),
     `format_gate_line` + `scan_run_for_pose_gate` (digest surface).
2. **Trainer wiring** (`experiments/train_levelset_witness_realized_through_R_mlx.py`):
   - argparse `--pose-finish-engage-on {muon(default)|sigma_min_plateau}`.
   - Detector setup after the jacobian_basin block: OBSERVER whenever σ_min telemetry on; ACTUATE + resume-register
     only when `sigma_min_plateau` + armed (`--pose-finish-start-epoch>0`); $0 canary at setup (untrusted ⇒ LOUD +
     disengaged); no-σ_min-source LOUD alarm if telemetry off.
   - T1 emit: `observe(ep, agg["median_sigma_min"])` + observer row.
   - Engage block: `sigma_min_plateau` branch = `detector.fired() (monotone) OR backstop`; `muon` branch =
     unchanged incumbent. DISENGAGED alarm at the final epoch if never engaged.
3. **DSL leg** (`curriculum_dsl.PoseFinishConditioningGate`): composable zero-required-arg Lever factory holding
   `--pose-finish-engage-on sigma_min_plateau` (+ optional backstop/w-pose with guards). Closes the completeness()
   gap; surfaces in the activation-ledger duty-to-measure queue.
4. **Resume discipline** (`resume_registry.DIRECT_CONTROLLER_NAMES`): added `pose_finish_conditioning_gate` (the
   `test_every_trainer_direct_registered_controller_is_canonical` gate correctly forced this — apparatus respected).
5. **Digest surface** (`tools/costate_digest.py`): `section_pose_conditioning_gate` renders the DISENGAGED alarm
   (or latest observer row) — the disengaged alarm DOES reach the costate digest (verified).

## Canary results (MEASURED, $0)
`canary_suite().passed = True`: negative (rising σ_min) classification NOT_PLATEAUED/DEGENERATE → does NOT fire;
synthetic positive (clean plateau) → PLATEAU_FIRED. The detector is trustworthy on the two synthetic controls.

## Round-1 adversarial self-review (own the fixes)
- **De-noiser lags the plateau?** YES, by ~1–2 settle windows (EMA α=0.5) — CONSERVATIVE (fires LATE = more
  conditioning, never premature); honors "sufficiently conditioned." Interacts with the DENSIFY amendment: needs
  ≥ `min_points`=5 σ_min points in the terminal band (owed-14/DENSIFY cadence). Documented (open-item).
- **Hysteresis × spike-guard?** Independent surfaces (σ_min telemetry vs training-loss recent_losses); engage does
  the same `recent_losses.clear()` re-treat the muon path did. No adverse interaction.
- **Disengaged alarm reaches the digest?** VERIFIED (synthetic run.log → `section_pose_conditioning_gate` line).
- **Incumbent byte-identity?** `muon` default takes the unchanged else-branch (same `_pose_finish_on` computation);
  detector runs as OBSERVER only (score-neutral telemetry, like jacobian_basin) and is NOT resume-registered → the
  TRAINED ARTIFACT + checkpoint sidecar are byte-identical. Only run.log gains telemetry rows (established pattern).
- **NO-FAKE:** the detector really computes EMA + OLS rolling slope + stderr guard; canaries are real synthetic
  series reproducing measured signatures; σ* really computed. Not a marker-returning stub.

## Open items (choices flagged where §A.4 was silent)
- **Equations leg = N/A-now** (stated in the FEED): the gate is a CONTROL LAW with DERIVED constants but its
  real-σ_min FIRE BEHAVIOR is UNMEASURED (owed-14). REACTIVATION: register `sigma_min_rolling_slope_plateau_gate_v1`
  when the governed replay lands the empirical anchor. (Chose not to register a low-blast-radius advisory equation
  with no anchor over building the load-bearing legs.)
- **Observer-in-muon-mode:** the detector runs as a score-neutral OBSERVER even in the default `muon` path
  (would-have-fired rows) per the anti-orphan default-ON rule — a deliberate choice (byte-identical artifact).
- **Backstop semantics:** an untrusted/degenerate/never-fired σ_min gate engages ONLY via the
  `--pose-finish-start-epoch` fail-safe backstop (NOT muon — the operator chose the σ_min gate), else ships banked R1.
- **DENSIFY (§A.4 Repair 5 / AMENDMENT-3):** the terminal-window σ_min probe cadence config (a config knob, owed-14
  territory) is NOT set here — this build makes the detector robust to sparse points (min_points guard) but the
  cadence tuning is a launch-config item.
- **σ* λ_min(F):** passed None (annulus probe off launch path) → σ* advisory reports unavailable unless λ_min supplied.

## Launch-readiness statement
The repaired pose-gate DETECTOR + trainer wiring + DSL lever + resume registration + $0 canaries are COMPLETE and
green. Before the gate is TRUSTED to actuate on a real run, ONE governed item remains (NON-blocking): **owed-14** —
a telemetry-ON R1-equivalent replay that LOGS real σ_min(ep), backtested via `load_sigma_min_series_from_jsonl` +
`run_detector_on_series`, asserting the detector fires on a known-conditioned basin. Until then, launch-1 ships
**banked-R1** (0.001610/7.2KB) with the LOUD disengaged alarm — pose-DISENGAGED is a valid terminal state, never a
launch dependency. Pointer 0.19110 UNMOVED (means).

## Canonical equations (Catalog #344)
# FORMALIZATION_PENDING: pose-gate detector build memo — the rolling-slope conditioning-detector law registers on its first real-run firing (run-1 observer rows were INSUFFICIENT_DATA/DEGENERATE_GUARD; no measured law yet).
