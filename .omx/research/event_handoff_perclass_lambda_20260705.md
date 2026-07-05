---
task: 315
tier: T2
axis: "[macOS-CPU advisory] NON-PROMOTABLE"
pointer: "0.19110 UNMOVED"
status: LANDED (default-OFF; run-3 target; net d_seg is a run-3 A/B, operator-GO gated)
equations: curriculum_handoff_critical_nucleus_v1 (anchor-appended, not re-registered)
dsl: CurriculumGauge.HANDOFF_NUCLEUS (gauge.py APPEND)
supersedes: none (completes the BUILDs the #302 symposium §C.ii items 1b + 2 gated on)
---

# Event-triggered CE→tau hand-off: per-class critical-nucleus guard + boundary re-anchor + per-class λ

Task #315 (the derived-schedule flagship). Completes the two BUILDs the T3 curriculum-derivation
symposium (#302, `.omx/research/council_grand_symposium_curriculum_derivation_20260705.md`) gated
the event trigger on: the **per-class critical-nucleus guard** (§C.ii item 1b) and the **boundary
re-anchor** (§C.ii item 2 / review M1). Plus the **per-class λ classifier upgrade** (deliverable 4,
built by a sibling fork, commit `55b1f5052`). ALL default-OFF → the #205 path is byte-identical.

**Everything here is MEANS. Pointer contest-CPU 0.19110 UNMOVED. No live run is training.**

## 1. The boundary re-anchor (M1) — what it is + how built

The symposium's headline: the schedule's continuous laws are DERIVED; what remains inherited is the
**CLOCK** — the wall-clock levers (persistence-warmup completion, seed-anneal withdrawal, analytic-
band engage) were calibrated so their key events land at the FIXED tau boundary (ep300). Under
event-triggering, tau fires at a DIFFERENT epoch, so those levers **de-sync** (review M1, class-
level). The re-anchor makes them **boundary-relative**: it shifts a lever's epoch into the schedule
frame it was calibrated in, so its event tracks the FIRED tau.

- **Mechanism** (`_evt_reanchor_epoch(ep, boundary_fired, boundary_hardcoded)`): feed the lever the
  shifted epoch `ep + (boundary_hardcoded − boundary_fired)` — a **shift, not a rescale** (schedule
  shape + length preserved). Worked: tau calibrated @300, fires @200 → shift +100 → a lever
  completing at virtual 300 completes at real 200; the analytic band at virtual 350 (=tau+50)
  engages at real 250 (=fired+50). This mirrors the ALREADY-shipped eikonal re-anchor
  (`_scheduled_eikonal_weight(..., step_epoch=_eik_step_ep)`) — the eikonal was the first re-anchored
  lever; this completes the tau-relative set.
- **Re-anchored (3 tau-relative levers):** persistence-warmup, seed-anneal, analytic-band — wired
  via the `_lever_epoch(ep)` closure, gated on `--curriculum-reanchor-levers AND event-triggering`.
- **NOT re-anchored (honest adaptation):** **hosc-β**. The symposium §C.ii item 2 lists β among the
  four, but β's `β=4` freeze is anchored to the **MUON** boundary (ep726), not tau. Muon stays a
  fixed cap until the Muon-event-trigger build (§C.ii item 5, out of this task's scope); re-anchoring
  β to tau would **mis-place the freeze point**. Documented in the flag help + the `_evt_reanchor_epoch`
  docstring. This is the OPTIMAL-FORM-honest subset: re-anchor the 3 that are tau-relative + safe,
  defer β to the build it actually depends on.
- **Byte-identity:** unfired / fired-at-cap / re-anchor-OFF → `_evt_reanchor_epoch` is the identity →
  every lever bit-for-bit as the #205 path (tested).

## 2. Per-class critical-nucleus guard + readiness telemetry

The hand-off law (`curriculum_handoff_critical_nucleus_v1`): the tau stage is sharp-limit MCF;
Allen-Cahn's critical-nucleus theorem ⟹ any scored class below its critical size is **ERASED, never
grown** (MEASURED: #205 seeded a lane at part_frac 0 → d_seg CREPT 0.004752@ep300 → 0.006568@ep400).
So a loss plateau is **necessary but NOT sufficient**; CE→tau is admissible only when EVERY scored
class is above its nucleus.

- **Nucleus stats** (`_evt_nucleus_counts` → `_evt_nucleus_stats`, VERBATIM the arithmetic of
  `tools/witness_per_stage_annulus_attribution.stage_stats` — reused, not reinvented):
  `part_frac[c]` = predicted partition area (BORN: `>0` ⟺ the class exists at all — the MCF-cannot-
  nucleate gate); `within_flip[c]` = per-class disagreement (FORMED: `≤ thresh`). Computed from the
  SegNet argmax the d_seg verdict ALREADY produces (`cpu_verdict_d_seg_argmax_batch` — one forward,
  **zero extra SegNet cost**), chunk-accumulated at n600 (`_evt_counts_add`).
- **Nucleus satisfied** (`_evt_nucleus_satisfied`): class `c` above nucleus ⟺ `part_frac > min_part_frac
  ∧ within_flip ≤ within_flip_thresh`; a class with `gt_px==0` in the batch is VACUOUSLY satisfied
  (never blocks). `all_ok` = ∀ scored class.
- **Readiness telemetry** (`_evt_readiness_row` → `handoff_readiness` JSON row per verdict):
  per-class part_frac + within_flip + nucleus_ok + `plateau_ok` + `ready = plateau ∧ nucleus`.
  **OBSERVABILITY-FIRST:** it runs even with the trigger OFF (`--handoff-readiness-telemetry`), so the
  NEXT normal run passively yields the per-class validation data the law needs — and the empirical
  within-flip threshold calibration (the theoretical knee is π₁=w/σ~5; the flag default 0.5 is the
  operational proxy the telemetry will confirm/correct). NEVER read into training/parity.

## 3. Event-trigger completion

- **Nucleus gate** wired into `_evt_resolve_seg_form._fire` (tau boundary only): plateau + nucleus-
  NOT-ready ⟹ HOLD in CE. The `nucleus_ready` state is the MEASURED half of the trigger (updated at
  **verdict cadence** per the litsweep guard — NO per-step adaptive), persisted in the resume sidecar
  (`__evt_nucleus_ready`) for bit-faithful ON-resume.
- **Recalibrated eps:** `--curriculum-plateau-rel-eps` default **1e-3 → 1e-4** (C1: 1e-3 fires ep151
  MID-DESCENT on #205 = 15% CE-floor loss). Byte-identical for default runs (consumed only on the
  event-trigger / readiness paths, both default OFF).
- **Ceiling fallback (never hangs):** the hardcoded cap still fires **unconditionally** even if
  nucleus never satisfies — an event run's fired epoch is never LATER than the OFF schedule (tested).
- **Guard OFF:** `nucleus_ready` stays True ⟹ the trigger is the pure-loss #292 build-2 exactly
  (byte-identical; existing event/closed-loop tests 39/39 green).

## 4. Per-class λ + classifier upgrade (deliverable 4; sibling fork `55b1f5052`)

The witness_control shadow controller classified v5's frozen-descending-S deadlock as "fine." The
upgrade (`costate_estimator.binding_term_stall` + `shadow_controller` overlay): the BINDING term is
the score term with the largest `|λ·slope|` (d_seg, λ=100). It flags **BINDING_TERM_STALL** when the
binding term's within-stage slope ≈ 0 while overall implied_S still descends (the deadlock signature
— S dropping via non-binding terms on stale weights). Per-class within-flip costates degrade to
UNIDENTIFIABLE when per-class data is absent (never fabricated from scalar).

**Backtest (real logs, honest caught/missed):** on `v2_attrclean_20260630` the upgrade CAUGHT **3**
windows (ep 675/700/725) where the scalar classifier called a false-green ("converging"/"plateau" →
advance/early-stop) while the binding d_seg term was frozen ~0.00407 as ep_loss descended (l7
surrogate↔verdict decoupling). On the CE-stall / tau-creep runs both classifiers AGREE_ALARM; on the
healthy seed-fix run no false positive. **Honest negative (stated, not hidden):** the pure
frozen-d_seg + strongly-descending-implied_S v5 ep110-172 trace is NOT in the committed run set (the
seed-fix run descends healthily; the pose-blind runs have S dominated by pose noise), so the
S-descending arm is validated by a **synthetic fixture**; the ep_loss-descending arm is validated on
**real logs** (the 3 l7 catches). Calibration was corrected by MEASUREMENT (`stall_rel_eps` 2e-3 →
3e-4 after the backtest false-fired on genuine −1.5e-3/ep CE descent). 46 tests green.

## 5. Byte-identity + tests

- **Default-OFF byte-identity:** all new flags default OFF/inert; the verdict uses the unchanged
  `_verdict_dseg_dpose_chunked` unless `_nucleus_on`; `nucleus_ready` defaults True; `_lever_epoch`
  is the identity unless re-anchor + event-triggering. Existing event-triggered + closed-loop
  regression suites **39/39 green** unchanged; both trainer files py_compile + ruff-F821 clean.
- **Tests:** 21 new trainer-side (`experiments/test_curriculum_nucleus_guard.py`:
  counts/stats/satisfied/readiness/reanchor/trigger-gate/cap-never-hangs/determinism) + 16 sibling
  witness_control (`test_witness_control_perclass_lambda.py`) = **37 new; all green** (plus the 24
  curriculum-law + 44 gauge tests updated + green). ≥15 requirement met with margin.

## 6. Triality

- **equations:** `curriculum_handoff_critical_nucleus_v1` — appended a `handoff_nucleus_guard_build_
  landed_20260705` EmpiricalAnchor (the forall-class clause is now BUILT; registered via
  `tools/register_lever_laws_curriculum_20260705.py`). NO new law registered (register-nothing-
  unanchored honored). Sibling anchored `margin_saliency_reachability...` — unrelated.
- **DSL:** `CurriculumGauge.HANDOFF_NUCLEUS` (APPEND) emits the completed run-3 hand-off argv (event
  trigger + eps 1e-4/windows 25/min-stage 250 + nucleus guard + reanchor + readiness telemetry) —
  all flags grep-verified REAL in the trainer.
- **DAG:** FEED-05zz appended.

## 7. What is NOT done (honest scope)

- **Muon engage-on-trigger** (§C.ii item 5) — deferred; Muon stays a fixed cap; hence hosc-β
  re-anchor deferred with it.
- **geometric hosc-β anneal** (§C.ii item 4) — not built here.
- **The net-S verdict is a run-3 A/B** — this build is the INSTRUMENT; the within-flip threshold, the
  actual fired-epoch delta, and the d_seg effect are all measurement-owed on the next run
  (operator-GO gated). No score claim. Pointer 0.19110 UNMOVED.
