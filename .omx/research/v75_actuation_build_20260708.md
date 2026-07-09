# v7.5 OPTIMAL-FORM ACTUATION — BUILD LOG (2026-07-08)

Actuation of `v75_optimal_form_actuation_spec_20260708.md` (b685a4eba) into `crucible_v7`
(`tac.witness_autoconfig._build_crucible_v7`) as DSL WitnessProgram deltas. Operator directives:
FEED-v75actuate (94b2e540f) + the mid-build elevation steer (Item A triad → priority-1; dash-comb
defer OVERRIDDEN; item-3 = a MEASUREMENT, register owed, NO concurrent GPU probe).

**CONFIG actuation, NOT a launch.** Heavy/paid launch stays operator-GO. Live #205 run (pid 63069)
untouched. **Pointer 0.19110 UNMOVED** — only a byte-closed n600 exact row from `upstream/evaluate.py`
(contest-CPU/CUDA, NEVER MPS) moves it. Every Δd_seg here is [macOS-MLX research-signal] NON-PROMOTABLE.

## ⚠️ RE-SEAL REQUIRED
This config change (paint + drop-include-lane + dash-comb, and any B/C/D follow-ups) MUST pass a fresh
v7.5 SEAL adversarial review BEFORE any launch. I did NOT re-seal (out of scope; not my authority).

## ITEM A — d_seg surgical fixes — LANDED (commit 1)
Surface: `_build_crucible_v7` `base` dict + `levers` tuple + `_CRUCIBLE_V7_DSL_LEVERS`; test fixtures in
`test_crucible_v7_config.py`. All via the DSL WitnessProgram (the emitter stays the DSL).

| # | item | DSL leg | status | measured anchor |
|---|------|---------|--------|-----------------|
| A.1 | paint-then-SDF + drop dead include-lane | `base["--lane-prior-phi1-mode"]="paint"` + `base.pop("--structured-init-include-lane")` | LANDED | lane FN 0.00713→0.00211 (~3×), #291 |
| A.2 | along-tangent dash comb (#287) | `DashComb()` Lever → crucible_v7 `levers` (8 now) | LANDED, DEFAULT-ON | oracle 0.00695; eq `dash_erasure_homogenization_20260707` |
| A.3 | −48% directional verification | — (a MEASUREMENT, not config) | REGISTERED OWED | −48% circular-probe; production `--self-orient` realization UNVERIFIED |

- **A.1** — prereq `--seed-islands` already emitted by crucible_v7 (verified). `replace` was the #291
  MEASURED NO-OP; `paint` = paint-then-SDF nucleation (ep0 admission gate = MEASURED part_frac[lane]>0).
  Dropping `--structured-init-include-lane` (lane_px=0, inert) makes the argv honest (no orphaned no-op).
- **A.2** — the `DashComb` factory already existed in the DSL; this build COMPOSES it into crucible_v7.
  Under the committed `lane_offloaded` basis regime, lane rides the FREE rule-118 analytic band at
  freq_along≈6 (cartoon scale) which provably cannot represent the ~25-cyc dash comb; DashComb supplies
  that sub-δ dash structure analytically at build time, rule-118 FREE at decode. No schedule-WHEN epoch
  flag ⇒ zero schedule-gate interaction; composes with `--lane-render-band` (F-3 coherence gate already
  asserts the band). Default-ON per the operator override of the spec's defensible-defer.
- **A.3** — operator: the −48% was measured on a circular-probe vehicle; whether production `--self-orient`
  (already ON) delivers it is UNVERIFIED. Do NOT fire a concurrent n600 GPU probe (2nd heavy GPU workload
  trips the machine-crash P0 hard gate while #205 holds the box). OWED: run as the v7.5 run's OWN
  directional A/B (self-orient ON vs OFF, same seed, first ~300 ep) AFTER launch or gated behind the live
  run. Pre-registered: production self-orient must move Road/all-class boundary d_seg toward −48%; a null
  reclassifies the *realization* (verdict-scope: the self-orient realization, not the directional paradigm).

**Triality:** DSL = DashComb composed in the crucible_v7 WitnessProgram + `_CRUCIBLE_V7_DSL_LEVERS`;
DAG = FEED-v75Aactuated (`sub015_DAG_*`); equations = `dash_erasure_homogenization_20260707` (A.2) + #291
measured anchor (A.1). All three consistent (no new EmpiricalAnchor: activation ≠ a new n600 measurement).

**Verification:** `test_crucible_v7_config.py` 51/51 + `test_witness_autoconfig.py` + `test_v75_birth_counterforce.py`
= 149 passed; `test_lever_registry.py`+`test_feed07b_build_levers.py`+`test_p0_forces_phase2_build.py` = 52
passed. `lever_registry.completeness().unmapped` UNCHANGED at 120 (no new orphan). ruff F clean.
crucible_v7 still compiles + validates 0 WitnessProgram.validate violations + schedule-provenance gate 0
NAKED. (Pre-existing UNRELATED collection error in `tools/run_compact_renderer_mlx_spine_runner.py:885`
— a stray `<<<<<<< Updated upstream` merge marker — is NOT from this build.)

## ITEM B — HORIZON FIX — PARTIAL (owed as commit 2)
- **B.4 temporal-screw-consistency (#360)** — the `TemporalScrewConsistency` Lever factory EXISTS
  (`curriculum_dsl.py:2778`; trainer flag `--seg-temporal-screw-weight` default 0.0). WIREABLE. NOT landed
  in this turn: it introduces `--seg-temporal-screw-start-epoch` (the factory default 0 is schedule-gate-
  EXEMPT as always-on, but the factory's own semantics say `start >= l7` — needs a formed partition to
  warp; under `--seg-form-unify-tau` l7 is dissolved, so the correct start-epoch/governance is a real
  design decision deserving its own reviewed commit, not a rushed same-turn add). OWED as commit 2.
- **B.5 horizon-weighted margin (#169) + sky=rotation-only** — **GAP / NO BUILT TRAINER FLAG.** grep of
  the levelset trainer argparse finds NO `--*horizon*` margin flag; sky=rotation-only exists ONLY as a
  warp-mode dict entry (`witness_autoconfig.py:427`), not a trainer flag or Lever factory. Wiring B.5 as a
  DSL lever would require INVENTING flags (forbidden). Item B's chroma boundary IS already on
  (`--seg-chroma-boundary-weight 0.1`). RECOMMENDATION: B.5 needs a trainer-side build (the 0-byte
  horizon-weighted margin term + a sky rotation-only stratification flag) BEFORE it can be a DSL lever —
  a genuine owed build, not an actuation.

## ITEM C — dynamic curriculum residual — OWED (commit 3)
- **C.6 plateau_ok telemetry legibility** — trainer-side observability (stamp `in_stage_epochs`/`stage_start`
  + dense-plateau slope on readiness rows). Score-neutral. Owed (trainer edit).
- **C.7 DERIVE curriculum_min_stage_epochs** — currently the bare literal 250 in crucible_v7's base. Deriving
  off the value-provenance ladder requires a named source (the note at `schedule_governance` shows it is an
  INPUT to `derive_octave_max_dwell` from `--anneal-epochs`); a real derivation, owed.
- **C.8 VERIFY Muon event nucleation positive-control** — RESULT: `--muon-start-event powerlaw_meat` IS
  emitted by crucible_v7 (verified) AND its governance declares it FIRES gated on the S2 REV-B
  nucleation-complete positive control (all LADDER arms past birth+hold+anneal). The `--muon-start-epoch 726`
  is the fail-safe BACKSTOP (LOUD `cap_fired_before_event` if the sensor did not fire by 726). So the Muon
  event nucleation IS wired + satisfiable via the event-wiring; it does NOT silently fall to 726 unless the
  nucleation sensor genuinely never completes (which is the correct fail-safe, LOUD not silent).

## ITEM D — POSE SEQUENCING — OWED (commit 4)
- **D.9 terminal POSE-FINISH stage** — the StoreNothingPoseCarrier + `--pose-carrier-xi-from-ckpt` /
  `--pose-carrier-dxi-scale` connectors EXIST (FEED-238resolved: R1 dxi BANKED, d_pose 0.001610 → 0.127,
  7.2KB). The one STRUCTURAL change: add a terminal pose-finish curriculum stage to the crucible_v7
  WitnessProgram (converge d_seg under the full curriculum → w_pose-emphasized ~100-Muon-ep pose-finish,
  R1 recipe) SUPERSEDING co-train-pose-from-ep0. This is a curriculum-stage build (a real TypedStage /
  Lever), owed as its own reviewed commit.

## NET
Item A (the operator-elevated priority-1 d_seg triad) is LANDED + reviewed + triality-recorded + tests
green. A.3 registered OWED (measurement, no concurrent probe). B.4/C/D are wireable-or-buildable owed
follow-ups; B.5 is a genuine trainer-build GAP (no flag to wire). RE-SEAL required before launch.
Pointer 0.19110 UNMOVED.
