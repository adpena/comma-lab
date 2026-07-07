# §15 $0 solver pack — (A) junction σ_ij Young's-law fit + (B) power-law plateau/exit detectors

**BUILD-WAVE agent D, 2026-07-07. Bridges of viscosity_theory_alignment_hunt_20260707.md §7
(EUREKA candidate #2) and §4 (weak KAM, "fit owed"). Everything below is MEANS
([macOS-CPU advisory], NON-PROMOTABLE); pointer contest-CPU 0.19110 UNMOVED.**

Durable artifacts:
- `experiments/results/solver_pack_20260707/junction_sigma/junction_sigma_fit.json` (machine-readable 5×5 + counts + CIs)
- `experiments/results/solver_pack_20260707/powerlaw_detector/powerlaw_fits.json`
- tools: `tools/fit_junction_sigma_youngs_law.py` · `tools/fit_powerlaw_plateau_detector.py`
- library: `src/tac/witness_control/powerlaw_exit.py` (+ producer signal in `producer_bridge.py`)
- equations: `tac.canonical_equations.junction_young_sigma_and_powerlaw_exit_20260707`
  (`junction_young_angle_sigma_fit_v1` + `weak_kam_powerlaw_tail_exit_v1`, REGISTERED)
- tests: `src/tac/tests/test_powerlaw_exit.py` (10) + `src/tac/tests/test_producer_bridge.py` (11, extended)

---

## (A) Junction σ_ij — MEASURED (n600 cached GT argmax, 1.3 s, peak RSS 2.8 GiB)

Method: 2×2-plaquette triple-junction detector (exactly 3 distinct classes) → clean-junction
filter (exactly 3 circular label transitions on a radius-4 circle, 120 samples; 3-class circle
set) → per-triple angle statistics → Young's law (σ_jk/sin θ_i = σ_ik/sin θ_j = σ_ij/sin θ_k)
inverted by weighted log-least-squares across triples, gauge geometric-mean(σ)=1 (all-ones = the
null). Sub-pixel junction refinement SKIPPED (stated): ±0.5 px corner noise averages over
thousands of junctions and is absorbed by the bootstrap (200 resamples, seed 0). Canonical class
order Road0/Lane1/Undrivable2/Movable3/MyCar4 taken from the cache, never re-derived.

**Measured angle distributions vs Herring 120°:**
- 3256 clean junctions / 6702 plaquette candidates (1 quad excluded); mean |angle − 120°| = **36.2°**.
- **19.3%** of clean junctions have an arc ≥ 180° — positive-tension equilibrium is impossible
  there (a Herring violation all by itself; concentrated at Lane slivers: 86/126 of R-L-U).
- Per-triple means (deg): Road-Undrivable-Movable (n=1815, the bulk triple): **[120.0, 123.7,
  116.3] — near-perfectly Herring** (all-ones ≈ CORRECT for bulk). Road-Lane-MyCar (n=760):
  [98.3, 93.9, **167.8**]. Road-Lane-Undrivable (n=40 fit + 86 dropped): [161.1, **28.4**, 170.5].
  Lane-Movable (n=5) and Road-Movable-MyCar (n=6) below the 30-junction fit floor.

**Fitted σ_ij (geometric-mean-1 gauge; bootstrap 95% CI; 4 of 7 fitted pairs exclude all-ones):**

| pair | σ | ci95 | excludes 1.0 |
|---|---|---|---|
| **Road-Lane** | **0.377** | [0.317, 0.441] | **YES** |
| Lane-Undrivable | 0.738 | [0.568, 0.922] | YES |
| Road-Undrivable | 1.085 | [0.994, 1.206] | no |
| Road-Movable | 1.006 | [0.918, 1.125] | no |
| Undrivable-Movable | 1.048 | [0.955, 1.185] | no |
| Lane-MyCar | 1.764 | [1.434, 2.119] | YES |
| Road-MyCar | 1.779 | [1.459, 2.150] | YES |

Unobserved pairs (NaN in the matrix): Lane-Movable, Movable-MyCar, Undrivable-MyCar.

**Reading:** the uniform `--length-weight` over-penalizes Lane boundary length **~2.7×**
(σ_RL = 0.377) relative to the frozen scorer's own junction geometry, and under-penalizes the
hood boundary ~1.8× — a named, fitted mechanism feeding lane erasure (converges with the
homogenization/pinning story, hunt §3). Honest limit: with only 3 usable triples the log-LS
system is exactly solvable (residuals 0 by construction) — cross-triple consistency has NO
leverage yet; the bootstrap CI covers sampling noise only, not model misfit.

**Consumption path (TrainerSupportGap — the flag does NOT exist; not invented):**
- proposed flag: `--length-sigma-matrix <path.json | 15 comma floats upper-tri>` consumed by the
  Chan-Vese length term per class-PAIR (default all-ones = byte-identical OFF).
- **future DSL holder: the `tac.witness_dsl.curriculum_dsl` `Regularizer("--length-weight", ...)`
  factory extended with a `sigma_matrix` argument** — lands as a `Lever` factory per the triality
  DSL-holds-every-lever rule, never a hand flag. Council draft §15 treatment arm, NOT the clean
  baseline (hunt §7 verdict). Owed anchor (ii): the σ-weighted vs uniform A/B with
  junction-local d_seg attribution (registered as the equation's reactivation criterion).

## (B) Power-law plateau/exit detectors — MEASURED retro-fit (long900 + live mod32cap)

Models: d_seg(t) = a + b·t^(−α) vs a + b·exp(−t/τ_e); deterministic profile-LS + golden refine;
a clamped ≥ 0; comparison by **AIC** (n·ln(RSS/n) + 2k — stated; no holdout, the series are too
short to split); bootstrap-200 α CI (seed 0). Windows: long900 ep1-900 (19 pts, single-stage KD
run) · mod32cap CE ep1-299 (11 pts) / tau 300-725 (18 pts) / Muon 726+ (4 pts).

| window | n | preferred | ΔAIC (exp−pow) | α | α ci95 | meat to +300 ep |
|---|---|---|---|---|---|---|
| long900 ep1-900 | 19 | **power_law** | **+45.4** | 0.510 | [0.053, 0.554] | 9.4e-5 |
| mod32cap CE 1-299 | 11 | exponential | −20.4 | (0.12) | [0.050, 0.122] | 3.0e-5 |
| mod32cap tau 300-725 | 18 | exponential | −46.9 | (0.056) | [0.050, 0.059] | 5.5e-6 |
| mod32cap Muon 726+ | 4 | exponential | −5.2 | — | nan (n=4) | 1.7e-4 (LOW-CONF) |

**The quantified §4 confirmation:** on long900 the exponential fit's asymptote is **0.002464 —
broken by the measured trajectory** (0.002334@ep500, 0.002017@ep900 = 22% below the exp floor,
gap 4.5e-4 = 100·4.5e-4 ≈ 0.045 S-units of d_seg the exp model wrote off). An exponential-window
plateau detector calibrated on this run would have declared exhaustion by ~ep200 — the measured
"meat left on the bone". Early stage windows are exponential-preferred, matching §4's regime
split (exponential off the obstruction set, power-law on it at long budget). α CI touches the
grid floor (0.05) on resamples — reported, not hidden; n=19 is thin for a 3-param fit.

**Pre-registered α_lane < α_road check: UNRESOLVED — APPARATUS GAP.** No run logs per-class
d_seg trajectories (verdict rows are total-only; the annulus sidecar holds exactly 2 per-class
snapshots). Duty-to-measure recorded in the equation + producer signal: per-class d_seg in the
verdict row (5 floats/verdict, score-neutral read-only telemetry ⇒ defaults ON per the
default-off-is-orphaned-signal rule). The library + producer signal consume per-class
trajectories the moment they exist.

**Exit rule (the control-surface callable):**
`tac.witness_control.powerlaw_exit:powerlaw_meat_exit(trajectory, *, horizon_epochs, meat_floor,
...)` — per-class dict (or bare series = "total") in → `{exhausted: bool,
remaining_meat_estimate, alpha, ci, binding_class, per_class, reason}` out. Fail-safe direction:
insufficient/unfittable data ⇒ NOT exhausted (confound-L3: never declare exhaustion on a bad
measurement). meat_floor/horizon are config defaults (1e-4 / 300 ep), not calibrated constants —
the fits are the finding, the floor is a knob.

**Costate wiring (REQUIRED per the mid-wave directive — landed):**
`producer_bridge._powerlaw_exit_signal(run_log_path)` mirrors `_harness_failure_signal`:
additive, fail-open (no path / short log / any exception ⇒ honest `available=False` + reason),
appended as the 5th producer in `read_producer_signals(..., run_log_path=None)`. It fits the
trailing half of the run's verdict rows (inside one stage at every observed cadence) and
surfaces `{exhausted, remaining_meat_estimate, alpha, ci}` to the costate digest/DECIDE queue.

**ExitEvent binding note (curriculum_dsl NOT edited — two siblings own it this wave):** a future
`ExitEvent(criterion="powerlaw_meat", floor=<meat_floor>, cap_epoch=<int>)` binds as a GAP-kind
criterion exactly like `marginal_dseg_floor`: `validate()` requires an explicit `floor`;
`flags()` returns `{}` (conservative compile = the fixed stage boundary via `cap_epoch`);
`support_gaps()` emits a TrainerSupportGap with
`flag_proposal="--stage-exit-powerlaw-meat-floor <float> + --stage-exit-powerlaw-horizon <int>
(trainer build; calls tac.witness_control.powerlaw_exit:powerlaw_meat_exit on the stage's
verdict window)"`. Adding `"powerlaw_meat"` to `_EXIT_EVENT_CRITERIA` + the two branch lines in
`validate`/`support_gaps` is the entire DSL touch, owed to the curriculum_dsl owners.

## Triality
- **DAG:** FEED-07x appended (sub015 DAG).
- **DSL:** no flags invented; two named TrainerSupportGaps (σ-matrix Regularizer arg ·
  powerlaw_meat ExitEvent criterion) + one telemetry gap (per-class d_seg verdict rows).
- **equations:** both registered with MEASURED anchors; owed anchors (σ A/B · per-class α) are
  the reactivation criteria, not silent gaps.
