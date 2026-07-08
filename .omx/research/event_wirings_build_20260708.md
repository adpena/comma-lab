# Event-wirings BUILD — the three v7 sensor→start transitions (2026-07-08) [no-triality]

status: BUILT + tested (57 new/updated tests green; 206 across affected suites). Pointer 0.19110
[contest-CPU] UNMOVED — this is a MEANS (a launch-config wiring); only a byte-closed n600
`upstream/evaluate.py` row < 0.19110 moves it. NO launch performed (live run-1 pid 63069 untouched;
run dirs read-only; the trainer edits are default-OFF / byte-identical for the v7 RELAUNCH).

## THE CHARTER (operator override, verbatim)
> "Build the wirings dumbass that's the whole point."

The operator OVERRODE the T3 v7 council's launch-with-caps consensus (S2/S3/S4 all voted
PROCEED_WITH_REVISIONS / launch-with-caps; S4 R1 asked only for a role discriminator). A cap-only
launch re-ships epoch scripting — the exact PR95-skeleton regression the schedule-provenance gate
exists to kill. So the three v7 transitions now FIRE ON SENSORS, with the epoch caps demoted to
fail-safe BACKSTOPS. The wiring IS the deliverable.

## STORES CONSULTED
`.omx/research/t5_crucible/crucible_v7_authored_20260708.md` (the wiring_gap list = the spec) ·
`position_V7_S2_dykstra_continuation_20260708.md` (REV-B positive control) ·
`position_V7_S3_daubechies_multiscale_20260708.md` (R-S3 would_fire calibration) ·
`position_V7_S4_rudin_contracts_20260708.md` (R1 role discriminator) · trainer sensors
(`_evt_resolve_seg_form` / `_evt_nucleus_*` / `_annulus_metrics_from_maps` / the muon switch /
lane-band + seg-chroma engage gates) · `tac.witness_control.powerlaw_exit.powerlaw_meat_exit` ·
`tac.witness_curriculum.ladder_homotopy.LadderArmSpec` (arm windows) ·
`tools/schedule_provenance_gate.py` (the 0-naked classifier) · `tac.witness_dsl.typed_config`
(ScheduleGovernance) · CLAUDE.md (serializer post-edit shas · review gate · byte-identity discipline).

## WHAT LANDED
- **`src/tac/witness_control/event_wirings.py`** (NEW, pure + unit-tested — the heart, where every
  falsification-relevant decision lives): `EventBackstopGate` (the event-vs-fixed-cap primitive) +
  `muon_meat_event` / `lane_nucleus_event` / `annulus_plateau_event` / `ladder_arms_complete` /
  `lane_would_fire_row`. The OFF branch (sensor absent) reduces to the EXACT incumbent `ep >= cap`
  comparison with NO telemetry — the binding byte-identity contract.
- **trainer** (`experiments/train_levelset_witness_realized_through_R_mlx.py`): 3 new
  `--<x>-start-event` flags + annulus-plateau detector params + `--lane-band-would-fire-telemetry`;
  3 `EventBackstopGate` instances wired at the muon switch, lane-band engage, and seg-chroma engage;
  sensor-state capture at verdict cadence (lane nucleus in `_emit_handoff_readiness`, annulus series
  in `_emit_verdict_row`). DEFAULT-OFF ⇒ every gate is event-mode OFF ⇒ byte-identical (verified:
  OFF-mode `start_reached == ep>=cap` for all 3; trainer imports clean).
- **`src/tac/witness_dsl/typed_config.py`** (S4 R1): `GovernanceRole` enum (fires|backstops) +
  `ScheduleGovernance.role` (optional, defaults from class; an explicit role must AGREE with the
  class). A CAP's `sensor` is now un-misreadable as a firing claim.
- **`tools/schedule_provenance_gate.py`** (S4 R1): `event_start_flags` registry + role validation +
  `classify_launch(event_registry=...)` classifies the co-emitted `--*-start-event` wirings as
  EVENT_TRIGGERED alongside their FAIL_SAFE_CAP backstops. Added the 3 start-event flags to
  `RECOGNISED_EVENT_SENSORS`.
- **`src/tac/witness_autoconfig.py`** (`derive_crucible_v7_config`): the 3 governance entries FLIPPED
  from cap-only to EVENT+BACKSTOP pairs (6 entries: 3 role=fires events + 3 role=backstops caps); the
  3 start-event flags co-emitted in `base`; `crucible_v7_wiring_gaps()` reframed as the BUILT wiring
  status. Gate outcome: **0 NAKED**, 3 EVENT_TRIGGERED + 3 FAIL_SAFE_CAP.
- **`tools/launch_witness_run.py`**: passes `event_registry` so the launcher's gate surfaces the wired
  events.

## PER-WIRING SENSOR SEMANTICS + THRESHOLDS (provenance)
1. **muon ← powerlaw_meat** (`--muon-start-event powerlaw_meat`): fires when
   `powerlaw_meat_exit(d_seg-history)["exhausted"]` (the weak-KAM tau-descent tail is below
   `meat_floor`, fail-safe on too-few points) **AND** S2 **REV-B nucleation-complete** positive
   control (`ladder_arms_complete(ep, [lane_window, movable_window])`, each window =
   birth+hold+anneal). The positive control HOLDS Muon while any LADDER arm still anneals so an
   island-birth transient d_seg dip cannot be misread as first-order exhaustion → premature Muon
   before the lane nucleates. Backstop cap: `--muon-start-epoch 726`.
2. **lane-band ← lane_nucleus** (`--lane-band-start-event lane_nucleus`): fires when the LANE class
   is BORN (`part_frac > min_part_frac`) AND FORMED (`within_flip <= --curriculum-nucleus-within-flip`,
   default 0.5; the π₁=w/σ≈5 knee proxy) — the #315/#302 per-class predicate applied to the lane
   class. Reads the per-class nucleus stats already computed at verdict cadence (setting the event
   implies nucleus telemetry ON). **S3 R-S3**: a `lane_band_would_fire` row is emitted EVERY verdict
   epoch when `--lane-band-would-fire-telemetry` OR the event is set (regardless of event/cap mode) so
   the dash-birth-timing calibration accrues even under cap operation. Backstop cap:
   `--lane-band-start-epoch 500`.
3. **seg-chroma ← annulus_plateau** (`--seg-chroma-boundary-start-event annulus_plateau`): fires when
   the within-annulus flip fraction (`threshold.annulus_flip_frac`, #333 telemetry promoted to a
   trigger) PLATEAUS — `|LS slope / mean| <= rel_eps` over the trailing `dwell_windows` verdict points
   AND span `>= min_epochs` (the boundary must DWELL, not momentarily flatten). Detector params carry
   req-T TAGGED provenance (module constants `ANNULUS_PLATEAU_{REL_EPS 1e-4, DWELL_WINDOWS 4,
   MIN_EPOCHS 150}` = sisters of the curriculum-plateau params; argparse-overridable). Backstop cap:
   `--seg-chroma-boundary-start-epoch 450`.

## THE LOUD BACKSTOP (S5)
Each gate emits `{"stage": "cap_fired_before_event", ...}` iff the fixed-epoch backstop fires while
the sensor never triggered by the cap — a firing cap is falsification-relevant (the wired sensor is
mis-calibrated; the next run must SEE it, not silently accept the cap). Event fires emit
`{"stage": "start_event_fired", ...}`. OFF mode emits NOTHING (byte-identity).

## COUNCIL-REVISION STATUS
- **S2 REV-B** (nucleation-complete positive control on the muon meat-exit): BUILT + tested
  (`test_muon_rev_b_positive_control_holds_on_incomplete_nucleation`, `test_ladder_arms_complete_*`).
- **S3 R-S3** (would_fire telemetry regardless of event mode): BUILT + tested
  (`test_lane_would_fire_row_emits_regardless_of_event_mode`); default-OFF for byte-identity, implied
  ON with the event.
- **S4 R1** (role discriminator; cap.role=backstops, event.role=fires): BUILT + tested (typed_config
  validation + gate role-agreement + config governance).

## V7 GOVERNANCE FLIP RESULT
`compile_crucible_v7_config` → schedule-provenance gate: **0 NAKED**. Classification:
`--muon-start-event / --lane-band-start-event / --seg-chroma-boundary-start-event` = EVENT_TRIGGERED;
`--muon-start-epoch 726 / --lane-band-start-epoch 500 / --seg-chroma-boundary-start-epoch 450` =
FAIL_SAFE_CAP. Diff-vs-v6 added set = the 3 lever families + the 3 wiring flags + the DSL
VerdictCadence emitter delta (sibling gpu-verdict work).

## TESTS (57 new/updated; 206 across affected suites)
`test_event_wirings.py` (30): OFF byte-identity · cap-None-never-fires · event fire+latch · event
beats cap · LOUD cap_fired_before_event · muon meat + REV-B hold/fire · ladder arms · lane nucleus
born/formed · would_fire both modes · annulus plateau flat+dwell/descending/short-dwell/too-few ·
detector-params-tagged · role default/mismatch · event_start_flags parse · gate role-agreement ·
classify_launch event surfacing. `test_crucible_v7_config.py` (27): +event-classification, +role,
+wired-status; diff/counts updated.

## BYTE-IDENTITY (the binding contract)
All three `--<x>-start-event` flags absent ⇒ every gate is event-mode OFF ⇒ `start_reached == ep>=cap`
(the lane-band gate on the SAME `_lever_epoch(ep)` the incumbent used; the chroma gate under the
existing `weight>0` block) ⇒ NO new telemetry ⇒ the trained weights / verdict scalars / emitted argv
are unchanged. Verified: OFF-mode reduction for all 3 + trainer imports clean + would-fire/nucleus
telemetry gated behind the event/opt-in flag.

## NOT DONE (honest)
- No n600 launch (means ≠ ends; the relaunch is operator-GO). Event-mode RESUME determinism (re-deriving
  the fired epoch from a replayed history) is a documented v7.1 concern; the OFF (byte-identity) path is
  resume-safe (fired-at-cap reproduces exactly). The equation leg is untouched (this is a wiring build,
  not a measured finding — no canonical_equations row is owed).
