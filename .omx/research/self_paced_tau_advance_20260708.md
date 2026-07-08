# Self-paced τ-advance — event-driven octave ladder BUILD (2026-07-08) [no-triality]

status: BUILT + tested (23 new tau_advance tests + 27 crucible_v7 + 16 lever_registry green; the full
schedule/dsl/autoconfig sweep passes bar 2 PRE-EXISTING L5-measurement-schedule failures unrelated to
this build — no reference to any τ-advance module). Pointer 0.19110 [contest-CPU] / 0.20533 [CUDA]
UNMOVED — this is a MEANS (a launch-config control surface); only a byte-closed n600 `upstream/evaluate.py`
row < 0.19110 moves it. NO launch performed; run dirs read-only; live run-1 untouched; default-OFF
byte-identical.

## THE CHARTER (operator 2026-07-08, verbatim)
> "Why is there a fixed number of epochs if our schedule and curriculum are no longer supposed to be
> hardcoded like pr95"

The τ-anneal `--anneal-epochs` denominator that clocks τ(t) (and the LR pin's `t/1000`) is the LAST
clock-hardcoding in crucible_v7. This build converts it to the S6-R4 event-driven form (the blind
derivation's element 5: *"τ advances on per-scale RELAXATION, self-triggered, one param at a time"* —
Ch.6 §3 critical slowing: a clock cannot slow itself when a specific scale is still relaxing).

## STORES CONSULTED
`.omx/research/t5_crucible/position_V7_S6_structure_blind_20260708.md` (R4 + the §PHASE-1 element 5) ·
`.omx/research/witness_native_schedule_derivation_20260709.md` (§3 self-triggered advance, §2 turnpike
tail, the geometric-shape derivation) · `.omx/research/event_wirings_build_20260708.md` (the
`EventBackstopGate` pattern extended here — event primary, cap LOUD backstop, byte-identity via the
OFF branch) · `tac.witness_control.{event_wirings, powerlaw_exit}` (the sensor engine reused) ·
`tac.witness_dsl.{curriculum_dsl, typed_config, lever_registry}` + `tools/schedule_provenance_gate.py`
(the governance/provenance surfaces) · `src/tac/witness_autoconfig.py` (`derive_crucible_v7_config`) ·
CLAUDE.md (serializer + review gate + byte-identity + resumability non-negotiables).

## WHAT LANDED
- **`src/tac/witness_control/tau_advance.py`** (NEW, pure + unit-tested — the heart):
  `TauAdvanceController` (clock|event) + `tau_octave_ladder` (geometric τ_k) + `derive_n_octaves` /
  `derive_octave_max_dwell` (DERIVED from existing flags, no bare literals) + `tau_advance_state_arrays`
  / `tau_advance_restore_from_cfg` (resume). The event sensor reuses `powerlaw_meat_exit`
  WITHIN the current octave (dwell-gated, thin-data fail-safe); the per-octave MAX-DWELL is a LOUD
  `cap_fired_before_event` backstop (S5). Byte-identity: clock mode returns the incumbent clock fn
  UNCHANGED and never advances/emits.
- **trainer** (`experiments/train_levelset_witness_realized_through_R_mlx.py`, landed via a concurrent
  sibling commit — see §CONCURRENCY): `--tau-advance-mode {clock,event}` (+ `--tau-octaves` /
  `--tau-octave-min-dwell` / `--tau-octave-max-dwell`, all default None => DERIVED); the pure
  `_lr_scheduled_event_for_epoch` + `validate_tau_advance_config` + `_build_tau_advance_controller`
  helpers; controller instantiation before the resume block (+ restore); τ/β/LR routing through the
  controller in event mode (the OFF branch is the incumbent calls, textually unchanged => byte-
  identical); ingest+maybe_advance+per-octave stage-ckpt in the loop; the Muon-switch FREEZE; the TAIL
  no-double-driver assert; the resume-sidecar persistence (`__ta_*` keys).
- **DSL** (`src/tac/witness_dsl/curriculum_dsl.py`, landed via sibling commit): `TauAdvanceEvent()`
  `Lever` factory (the "a lever is not built until it is a `Lever` factory" discipline). Auto-discovered
  by `lever_registry` (AST) — the 4 flags all map (`completeness().mapped`), not `.unmapped`.
- **governance** (`src/tac/witness_autoconfig.py` `_crucible_v7_schedule_governance`): `--tau-advance-mode`
  = class=event/role=fires (sensor=per-band relaxation); `--tau-octave-max-dwell` = class=cap/role=backstops.
  `derive_crucible_v7_config` co-emits `--tau-advance-mode event` (see §LAUNCH RECOMMENDATION).
- **provenance gate** (`tools/schedule_provenance_gate.py`): `--tau-advance-mode` added to
  `RECOGNISED_EVENT_SENSORS` so the governance declaration validates.

## THE COUPLING DECISIONS (+ reasoning)
1. **Octave ladder = geometric clock VALUES.** `τ_k = start·(end/start)^(k/N)` — EXACTLY the geometric
   clock (`_softmax_temp_for_epoch` geometric) sampled at prog=k/N. Event mode reuses the clock ladder
   VALUES; only the per-rung DWELL is event-driven. (Verified by a test: `ladder[k] ==
   _softmax_temp_for_epoch(e_k)` at the octave-boundary epochs.) N defaults to
   `round(anneal_epochs / --curriculum-min-stage-epochs)` (6 at 1500/250) so each octave gets ~one
   min-stage-epochs of clock-equivalent dwell — the one-continuation-param window.
2. **β co-anneal → octave fraction.** β↑ rides τ↓ as ONE Γ-limit (Ch.4 Deriv-3). In event mode β
   interpolates on the octave fraction k/N (the exact incumbent interp form, prog := k/N), so it stays
   coupled to τ regardless of wall-clock and freezes when the fraction stops (at Muon).
3. **LR pin → octave fraction (derivation-consistent).** S6/DE pin LR ∝ the τ-control's OWN denominator.
   In event mode that denominator IS the octave ladder, NOT a fixed 1000-epoch clock. So the LR cosine's
   anneal PROGRESS = the octave fraction (the SHAPE is unchanged; only the prog SOURCE re-clocks from
   epoch to octave). Warmup stays REAL-epoch (initial optimizer stabilization is genuinely time-based,
   not scale-based). Chosen over "keep LR on clock" because keeping a fixed-epoch LR clock while τ is
   event-driven would re-introduce exactly the epoch-hardcoding the operator objected to, on the LR axis.
4. **unify-τ loss-τ = render-τ** follows automatically: the trainer couples the unified-L_τ loss
   temperature to `model.softmax_temp`, which event mode sets from the ladder (no extra wiring).
5. **Muon / TAIL — NO double-driver of τ.** The ladder FREEZES at the Muon switch (`freeze(ep)`); post-
   Muon τ is the finisher freeze + the TAIL τ_k cycles ONLY. Enforced THREE ways: (a) `maybe_advance` is
   guarded on `not muon_switched`; (b) `maybe_advance` ASSERTS `not frozen` (structural); (c) the TAIL
   block ASSERTS the controller is frozen before it drives τ. This mirrors the clock freeze (τ held at
   `_softmax_temp_for_epoch(muon_start_epoch)`); "floored before Muon" is NOT required (clock also freezes
   mid-descent at ~0.55 at ep726) — the ladder simply freezes at whatever rung it reached, and a LOUD
   `tau_advance_frozen_at_muon` row reports whether it was floored.
6. **Transition events unchanged**: the muon/lane-band/chroma `EventBackstopGate`s are untouched.

## RESUME DETERMINISM (launch-critical)
The controller OWNS its current-octave d_seg history (fed via `ingest`, decide-on-previous, single-
threaded in the main loop) so it does NOT depend on the trainer's non-persisted `history` list. Every
checkpoint persists rung / octave-start / last-seen-epoch / frozen / current-octave (epoch, d_seg)
history / fire log via `__ta_*` sidecar keys (mirroring `_cl_state_arrays`). A crash-resume restores all
and re-ingests only genuinely-new points (epoch > last_seen), so the subsequent τ trajectory is
bit-faithful. **Tested** (`test_resume_mid_octave_reproduces_identical_subsequent_tau_sequence`): a
continuous 40-epoch run and a run stopped mid-octave at ep17 + serialized + restored produce IDENTICAL
τ sequences. This mechanism GENERALIZES the wirings-memo v7.1 resume concern; extending it to the three
transition-event gates' fired-state is a cheap same-store follow-up (their `EventBackstopGate._fired_*`
would round-trip through the same `__ta_`-style keys), documented here as the SPLIT (not landed this
build — the gates' OFF/at-cap path is already resume-safe per the wirings memo; only their event-mode
fired-state is the residual, and no event-mode gate run is launch-imminent).

## schedule_governance CLASSIFICATION
τ-advance: class=event, sensor=per-band relaxation (powerlaw_meat). per-octave max-dwell: role=backstops.
`derive_crucible_v7_config` classification unchanged at 0 NAKED (the τ flags are not `--*-start-epoch`
triggers, so they are out of the naked-primary-epoch classifier's scope; they are DECLARED in the
governance surface + HELD by the DSL lever).

## CLOCK-vs-EVENT LAUNCH RECOMMENDATION (both implemented honestly)
**The config emits `--tau-advance-mode event`** (honoring the operator's explicit conversion directive —
that IS the point of this build). **BUT I recommend the council/seal run the FIRST unified-L_τ run in
CLOCK mode**, then flip to EVENT for run-2. Reasoning (honesty over compliance, per the charter's
invitation to say so plainly if event is risky for the first unified run):
- **One continuation parameter at a time — applied to the CAMPAIGN.** Run-1's PRIMARY, load-bearing
  change is the unify-L_τ dissolution of the CE→tau_softplus PR95 skeleton (the STRUCTURAL divergence,
  element 1). Event-driven τ (element 5) is MED-LOW priority in the derivation itself (S6-R4: *"consider
  a relaxation-plateau gate ... OR explicitly accept geometric as the derivation-consistent open-loop
  approximation"*). Turning BOTH on simultaneously CONFOUNDS attribution — a result could not be assigned
  to unify vs event-advance. This is the SAME "one continuation param at a time" principle event mode
  itself implements at the τ level, lifted to the campaign level.
- **Event mode couples THREE schedules (τ, β, LR) to an UNPROVEN sensor** (powerlaw_meat within a short
  octave window). The first unified run should isolate the sensor's first real measurement, not ride it.
- Event mode is FULLY built + governance-tracked + tested; flipping the config to `clock` is a
  ONE-TOKEN change (byte-identical to the incumbent anneal). So run-1-clock costs nothing and run-2-event
  is a trivial flip. The council/seal makes the final launch-mode decision.

## NO canonical-equations row owed
Per the event_wirings-memo precedent: this is a CONTROL/wiring build, not a measured finding. No
`canonical_equations` row is owed until an event-mode run produces a byte-closed d_seg trajectory that
confirms/refutes the self-paced advance. [no-triality]: the DSL leg (TauAdvanceEvent) landed; the DAG
+ equations legs move when a measured event-mode row lands.

## CONCURRENCY note (honest)
Two sibling builders (revisions-B: witness_autoconfig/tail_cycles/ladder_homotopy; compute-audit:
launcher/typed_config) were editing the SAME working tree concurrently. Their commits absorbed my
already-in-flight trainer + curriculum_dsl hunks into their commit objects (the shared-working-tree
absorption pattern — the WORK is preserved + coherent in HEAD, but mis-attributed to sibling messages).
My remaining files (tau_advance.py + tests + witness_autoconfig governance + schedule_provenance_gate)
are committed cleanly here. All my τ-advance edits are present + tested in HEAD.
