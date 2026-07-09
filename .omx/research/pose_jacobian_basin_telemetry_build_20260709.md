# The ξ→PoseNet Jacobian BASIN telemetry — BUILD + VERIFICATION (crash-resume completion, 2026-07-09)

**Axis:** `[build / verification — n8 MLX advisory smokes, $0 paid, no launch]` · Pointer contest-CPU
**0.19110 UNMOVED — MEANS.** This memo is the BUILD-CONTRACT verification record for the score-neutral
OBSERVER telemetry specified in `pose_jacobian_conditioning_basin_trigger_formalization_20260709.md`
(git `9fd6f9184`). A PREDECESSOR build agent crashed after 94 tool-uses having COMMITTED NOTHING; this
session ASSESSED the uncommitted work, COMPLETED-IN-PLACE (it was coherent + near-complete — no rebuild),
rigorously VERIFIED every binding contract gate on the REAL sealed v7.5 trainer, and committed only after
all gates passed. Every number below is MEASURED (cite the smoke) or DERIVED (cite the spec).

## 0. ASSESS verdict — COMPLETED-IN-PLACE (not rebuilt)

The predecessor's uncommitted work — `src/tac/witness_control/jacobian_basin.py` (the σ_min core),
`src/tac/tests/test_jacobian_basin.py`, the +222-line trainer hook, the +53-line `curriculum_dsl.py`
`TelemetryCadence` sensor + `TerminalPoseFinish(start_event=...)` run-2 actuator stub — was COHERENT,
faithful to the spec (B1–B5), and complete. It cleanly separated the MLX core (`jacobian_xi` via
`mx.vjp`) from the pure-numpy conditioning/aggregation/stratification (unit-testable at $0). Decision:
**complete-in-place** — no revert, no rebuild. The only NEW work this session added is the equation leg
(`pose_jacobian_basin_conditioning_v1`, which did NOT exist despite the task's belief), this memo, and
the DAG feed. No trainer/DSL logic was changed by this session (the hot-path code is exactly the
predecessor's, now verified).

## 1. THE BINDING CONTRACT — every gate MEASURED

| Gate | Result | Evidence |
|---|---|---|
| **T1 finite-diff parity** | **PASS** | `test_jacobian_basin.py::test_jacobian_xi_nonlinear_finite_diff_parity` — analytic `mx.vjp` J_ξ matches central finite-difference to `atol=2e-3`; linear-exact case to `atol=1e-5`. 11/11 unit tests pass. The σ_min the sensor reports comes from a Jacobian proven RIGHT. |
| **B4 launch-path** | **PASS** | The REAL sealed crucible_v7 config (minimal-render variant) STARTS on the governed trainer, renders, reaches the verdict path with `--jacobian-basin-telemetry` on, emits `{stage:jacobian_basin_setup}` + T0 `{stage:jacobian_basin_t0}` + T1 `{stage:jacobian_basin}` rows with `actuated:false`, 0 fail-open skips on the happy path. Not just a config/unit test — the GATE_KEY_PREFIXES lesson honored: the live launch path is exercised. |
| **B1 byte-identity** | **PASS** | See §2. On the DETERMINISTIC CPU substrate telemetry ON vs OFF is BIT-IDENTICAL (EMA 56 arrays + full resume-state 152 arrays incl live weights/optimizer/losses, max_abs **0.0**). The GPU ~1e-4 divergence is proven to be baseline MLX-GPU cross-process nondeterminism, NOT the telemetry. |
| **B2 fail-open** | **PASS** | See §3. Injected `jacobian_xi` failure → 3 `{stage:jacobian_basin_skip}` rows → `{stage:jacobian_basin_disabled}` at epoch 3 (self-disable after 3 consecutive) → training SURVIVED to completion. No exception ever escaped into the train loop. |
| **B5 observer-only** | **PASS** | Every T1 row carries `actuated:false`; the pose-finish stays TERMINAL (`f_basin=1.0` default = the sealed policy). `TerminalPoseFinish(start_event='jacobian_basin')` fail-LOUD `raise NotImplementedError` (the run-2 actuator is default-off / duty-to-measure, NOT built on the trainer). |
| **Cadence overhead** | **MEASURED** | See §4 — negligible at the k=32/every-4 default. |

## 2. B1 — SCORE-NEUTRALITY, rigorously (the confound this dodged)

**The confound:** the sealed v7.5 run uses `--mlx-device gpu`, and MLX-GPU is NOT cross-process
bit-identical by design (commit `5c5840be4` D6: "MLX-GPU crossproc non-bit-identity (byte-close
CPU-locked)"; the #348 dup-index-atomic-scatter finding). So an ON-vs-OFF GPU checkpoint compare is
CONFOUNDED — it measures telemetry PLUS baseline GPU nondeterminism. A naive "ON≠OFF on GPU ⇒ telemetry
corrupts training" would have been a FALSE positive.

**The controls (n8, epoch-8 EMA checkpoint, GPU):**

| pair | telemetry | arrays differ | max_abs |
|---|---|---|---|
| OFF vs OFF2 | **both OFF** (pure baseline) | 19 | 1.02e-4 |
| ON  vs OFF  | ON vs OFF | 19 | 1.19e-4 |
| ON  vs OFF2 | ON vs OFF | 19 | 1.08e-4 |

The telemetry-ON run diverges from OFF by the SAME structure (14 identical / 19 differ) and SAME
magnitude (~1e-4) as two OFF runs diverge from each OTHER. The telemetry adds nothing above baseline
GPU nondeterminism.

**The deterministic CLINCHER (CPU substrate, `--mlx-device cpu`, no Metal fused-R → deterministic):**

| checkpoint | verdict | detail |
|---|---|---|
| `levelset_witness_ema_mlx.npz` | **BIT-IDENTICAL** | 56/56 arrays, max_abs **0.0** |
| `levelset_resume_state.npz` | **BIT-IDENTICAL** | 152/152 arrays (live weights + optimizer moments + `__recent_losses`), max_abs **0.0** |

On the substrate where MLX IS deterministic, telemetry ON == OFF EXACTLY. **B1 is PROVEN: the telemetry
is score-neutral by construction** (it reads the render + FROZEN PoseNet, returns a dict, mutates
nothing; the trainer's hot loop draws no per-step `mx.random` — only `mx.random.seed` at startup; the
sensor's stratification is numpy-deterministic; `mx.vjp` differentiates only ξ). This is what lets the
observer enter the SEALED run without a re-seal.

## 3. B2 — fail-open (injection MEASURED)

Harness: monkeypatch `tac.witness_control.jacobian_basin.jacobian_xi` to RAISE before the trainer's lazy
import resolves it (same cached module object), run the sealed config with `--no-jacobian-basin-t0`
(isolate T1 so the shared consecutive-fails counter is driven purely by the injected T1 failures).
Result: `jacobian_basin_skip` rows with `consecutive_fails` 1→2→3, then `jacobian_basin_disabled` at
epoch 3, and the run CONTINUED to epoch 4 completion (`__B2_MAIN_RETURNED`). No telemetry error ever
became a training error. A multi-day run survives any sensor fault, degrading to zero cost after 3
consecutive failures.

**Honest nuance (noted, not blocking):** the consecutive-fails counter is SHARED between T0 and T1, and
a T0 SUCCESS resets it (`fails=0`). So if T0 is ON and healthy, it can mask T1's self-disable latch
(T1 failures are still individually caught + logged — fail-open safety HOLDS regardless; only the
"disable after 3" latch is affected). This is a robustness quirk, not a correctness bug: fail-open
(never crash) is unconditional; the self-disable is a spam-reduction convenience. A future refinement
could split the T0/T1 counters. The B2 test isolated T1 (`--no-jacobian-basin-t0`) to exercise the
latch directly.

## 4. Cadence overhead (MEASURED)

DERIVED (spec B3): k=32 stratified × 6 VJPs × ~2 fwd-equiv ≈ 384 fwd-equiv ≈ 0.64× a verdict eval per
T1 fire, at every-4-verdict ⇒ ~0.16× amortized. MEASURED on GPU (n8): with T1 at k=8/every-verdict the
run showed NO material wall-clock delta vs OFF (both ~164s/verdict-interval over a 658s span — the
verdict cadence is dominated by the `--verdict-batch 32` async CPU eval, and the T1 sensor is small
relative to it). **Direct sequential-timed A/B at the DEFAULT k=32 (T0 off, T1 every verdict, 3 calls):
ON32 = 249.5s vs OFF = 248.2s ⇒ delta 1.3s / 3 calls ≈ 0.4s per T1 call at k=32** — << 1% of the ~248s
3-epoch window. At the sealed default (k=32, every-4, T0 every verdict) the amortized overhead is
negligible against the multi-day run; if contested at launch, k=16 or every-8 still resolves the
σ_min(epoch) CURVE.

## 5. GLOBAL verification

- **crucible_v7 / DSL:** `TelemetryCadence.flags()` emits exactly the 8 `--jacobian-basin-*` trainer
  flags; all 8 match the trainer argparse (never-invent-flags). `lever_registry.completeness()` shows
  **ZERO** jacobian-related unmapped flags (the DSL HOLDS every designed lever); 120 pre-existing
  unmapped, none jacobian. 135/135 DSL + lever_registry + campaign tests pass. `TerminalPoseFinish`
  default path + guards validated (`start_event='jacobian_basin'` → NotImplementedError;
  invalid start_event → ValueError; f_basin∉(0,1] → ValueError).
- **telemetry-OFF trainer path is byte-identical to the pre-build sealed trainer:** the +222 trainer
  lines are all inside `if _jbasin_on:` / the `--jacobian-basin-telemetry` guard + the fail-open
  wrappers; `--no-jacobian-basin-telemetry` executes no sensor code ⇒ the sealed config's training is
  provably unchanged (and the CPU B1 clincher confirms even ON is bit-identical).
- **ruff F clean** on all changed .py (module, test, trainer, DSL, equation module).
- **Resumability additive:** the sensor holds only mutable observer state (`fails/disabled/plateau/
  t1_count`) that is NOT persisted; nothing in the resume/checkpoint path changed (confirmed by the
  bit-identical resume_state.npz).

## 6. Light re-verify vs re-seal — argued from B1

**No re-seal is required.** B1 proves byte-identity by construction (CPU bit-identical; GPU within
baseline nondeterminism). The telemetry is the SAME class as the existing `annulus`/`confound`
read-only observers, which entered the sealed run without a re-seal. The default-ON observer changes
NO trained byte; `--no-jacobian-basin-telemetry` is the pure-A/B fail-safe. A LIGHT re-verify (the B4
launch-path gate on the full sealed config at launch) is the only owed step before the multi-day run —
the standard launch-path smoke, not a curriculum re-seal.

## 7. Triality legs

- **DSL:** `TelemetryCadence` `JacobianBasin` sensor (8 flags, both tiers OBSERVERS, default-ON per the
  "off is a tracked queue" law) + `TerminalPoseFinish(start_event='jacobian_basin', f_basin=…)` run-2
  actuator stub (default-off / duty-to-measure; fail-loud NotImplementedError). `curriculum_dsl.py`.
- **DAG:** FEED-posejacbuild (this build + the verification matrix) appended to
  `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` same-commit.
- **Equations:** `pose_jacobian_basin_conditioning_v1` REGISTERED (it did NOT pre-exist) —
  `tac.canonical_equations.pose_jacobian_basin_conditioning_20260709`: the coherence↔conditioning law
  (σ_min→0 on a flat render via the ∇source-multiplicative aperture) + the reachability floor
  σ*=√(6·d_pose_init)/ρ_budget + the EXACT ξ-channel seg⊥pose kernel; advisory anchor = the ce-baseline
  σ_min readout (median 0.063, cond ~8000, r_eff ~1.27 — CONFIRMS the DERIVED near-rank-deficient
  basin-empty prediction) + the B1 byte-identity receipt. Anchors OWED: the σ_min(epoch) curve to
  convergence + the run-2 f_basin A/B; n600 + exact-eval before any promotable pose number.

## 8. FINAL STATE

$0 paid, no launch, no re-seal, pose-finish TERMINAL-unchanged. All binding gates GREEN (T1-parity,
B1 byte-identity, B2 fail-open, B4 launch-path, B5 observer-only, cadence). The telemetry is a
score-neutral OBSERVER ready to ride the v7.5 launch and MEASURE the σ_min(epoch) basin curve + the
would-have-fired epoch against the actual terminal outcome — the trust artifact for the run-2
resume-A/B on f_basin. The basin-trigger OPTIMALITY remains the HYPOTHESIS that A/B decides; σ_min is
the INSTRUMENT that makes it measurable, not a proof that earlier engage wins. Pointer **0.19110
UNMOVED — MEANS.**
