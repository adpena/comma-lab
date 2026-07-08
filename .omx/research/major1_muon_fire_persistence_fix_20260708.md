# MAJOR-1 fix — event-muon crash-resume determinism (muon-fire-epoch persistence) [no-triality]

Closes SEAL v7 R1 LENS-3 MAJOR-1 (`.omx/research/t5_crucible/seal_v7_r1_confound_20260708.md`).
Pointer 0.19110 UNMOVED — this hardens a MEANS (a crash-resume determinism gate on an event-muon
launch). Only a byte-closed n600 `upstream/evaluate.py` row < 0.19110 moves the pointer.

## STORES CONSULTED
- `.omx/research/t5_crucible/seal_v7_r1_confound_20260708.md` — the MAJOR-1 finding + fix recipe.
- `src/tac/witness_control/tau_advance.py` — the `__ta_*` persistence pattern to copy EXACTLY
  (`tau_advance_state_arrays` returns `{}` when clock mode ⇒ byte-identical; `tau_advance_restore_from_cfg`
  loader-tolerant; MINOR-1 `dwell_at_cap` seam in `maybe_advance`).
- `src/tac/witness_control/event_wirings.py` — `EventBackstopGate` (`_fired_epoch`/`_fired_by` hold the
  ACTUAL sensor-fire epoch; `event_mode == sensor is not None`; the OFF branch is the byte-identity contract).
- `experiments/train_levelset_witness_realized_through_R_mlx.py` — the seams:
  `_muon_gate` construction (~L5158), `_build_resume_state_arrays` (~L573) + its `_do_checkpoint` call
  site (~L5941), `_resume_into_finisher` (~L6154), `muon_switched = bool(_resume_into_finisher)` (~L6633),
  the muon switch + `_tau_ctrl.freeze` (~L7191/7199), the tau ingest/advance guard `not muon_switched`
  (~L7370), the TAIL `_tau_ctrl.frozen` assert (~L7494), the softmax freeze `_anneal_ep` (~L6382).
- `src/tac/tests/{test_event_wirings,test_tau_advance_self_paced}.py` — the pure-decider test harnesses.
- CLAUDE.md — resumability + per-stage-checkpoint + deterministic-reproducibility + confound-self-protection
  non-negotiables.

## THE FAILURE ANATOMY (unclosed pre-fix)
In event-muon (`--muon-start-event powerlaw_meat`) the switch fires on its SENSOR at an epoch < the
backstop cap `--muon-start-epoch` (e.g. fire@650, cap@726). At fire: `muon_switched=True`, opt rebuilt as
the Muon MultiOptimizer, `_tau_ctrl.freeze(650)` ⇒ ckpt carries `__ta_frozen=1` + Muon opt state. The
fire epoch (650) was NOT persisted. On a crash between fire and cap (crash@700, resume start_epoch=701):
`_resume_into_finisher = start_epoch > args.muon_start_epoch = 701 > 726 = **False**` (cap, not fire). Then:
- (a) opt NOT rebuilt as Muon ⇒ a fresh AdamW is restored against a Muon checkpoint ⇒ optimizer-key
  mismatch (best-effort restore drops the momentum);
- (b) `muon_switched=False` ⇒ the loop re-enters the switch ⇒ Muon re-switches, LOSING 650→700 momentum
  ⇒ non-bit-identical continuation;
- (c) with event-τ on, `__ta_frozen=1` is restored while `muon_switched=False` ⇒ the guarded
  `_tau_ctrl.maybe_advance(ep)` runs on a FROZEN controller ⇒ its `not frozen` assert ⇒ HARD CRASH.

## THE FIX (mechanism)
Persist the muon gate's ACTUAL sensor-fire epoch (`__mg_fired_epoch` + `__mg_fired_by`) via
`event_wirings.muon_gate_state_arrays` — a copy of the `__ta_*` pattern: returns `{}` when the gate is
`None` or in EVENT-muon OFF (`sensor is None`) ⇒ ZERO new sidecar keys for clock/cap muon ⇒ byte-identical.
Persisted from `_build_resume_state_arrays(..., muon_gate=_muon_gate)` alongside the τ controller. On
resume, `muon_gate_restore_from_cfg(_muon_gate, resume_cfg)` restores the gate's fired state and the
finisher decision is reconstructed from the FIRE epoch:
`_resume_into_finisher = start_epoch > _muon_gate.fired_epoch` (when the gate fired on its sensor). Then
`muon_switched = bool(_resume_into_finisher)` is `True` ⇒ (a) the Muon MultiOptimizer is rebuilt BEFORE
the state restore (keys match, momentum continuous), (b) the loop's muon switch is skipped (no re-switch;
the restored gate is also latched), and (c) the tau ingest/advance block (guarded on `not muon_switched`)
is never reached ⇒ the frozen assert cannot trip.

## FALLBACK SEMANTICS (byte-identity)
`muon_gate_restore_from_cfg` returns `False` — leaving the gate fresh and routing to the incumbent cap
comparison `start_epoch > args.muon_start_epoch` — for EVERY non-firing case: a pre-fix sidecar (no
`__mg_*` keys), clock/cap muon (event OFF ⇒ no keys written), or an event-muon gate that had NOT fired at
the crash (`-1` sentinel ⇒ the sensor re-arms; below-cap resume ⇒ no finisher entry). In clock/cap muon
fire == cap, so the fire-epoch and cap rules agree by construction ⇒ the change is byte-identical with
event-muon OFF (asserted in tests). MINOR-1: `dwell_at_cap` bool added to the `tau_advance` EVENT
telemetry row (True when the sensor fires at/after the max-dwell boundary — the S5-suspicious
event∧cap coincidence — additive, never read into training).

## TESTS (10 new; 63 total green across the two harnesses)
event_wirings: off/None-empty byte-identity · not-fired sentinel restore-fresh · fire round-trip
(fired_epoch+by through the `_load_resume_state` `a.item()` parse) · MAJOR-1 reconstruct-from-fire-not-cap
(the core: cap-701>726 False vs fire-701>650 True) · fire-epoch boundary (651>650) · crash-before-fire
resumes-with-cap-no-finisher · frozen maybe_advance IS the documented AssertionError + the muon_switched
guard skips it · pre-fix + clock/cap fallback byte-identical · no-muon-configured never enters finisher.
tau_advance: `dwell_at_cap` False on an early event, True at/past the cap.

## Provenance
git-tracked at commit below; `--patch-file` serializer (shared trainer) + `git show` post-commit verify;
review_tracker 2 clean passes on the 4 `.py` files. NO launches; run dirs READ-ONLY. [no-triality]
