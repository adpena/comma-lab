# SEAL v7 R1 — LENS-3 confound hunt + apparatus validity (2026-07-08) [no-triality]

reviewer: SEAL ROUND-1 LENS-3 (Opus, hostile). target: v7 stack @ HEAD (697dad238) through the
CLAUDE.md §Confound self-protection lens (DEFAULT-HARMFUL × SILENT × MEASUREMENT-CORRUPTING) +
the owed `--tau-advance-mode` clock-vs-event arbitration for the FIRST v7 run. NO launches.
Pointer 0.19110 UNMOVED — this is MEANS (a control surface); only a byte-closed n600 row moves it.

## STORES CONSULTED
`self_paced_tau_advance_20260708.md` (the BUILD memo + its own clock-first recommendation) ·
`DRAFT_v7_restart_config_synthesis_20260708.md` (§1 makes muon/lane/chroma event-triggered for run-1) ·
`src/tac/witness_control/tau_advance.py` (the controller) · `event_wirings.py` (the 3 gates) ·
`experiments/train_levelset_witness_realized_through_R_mlx.py` (loop wiring: ingest/advance 7294-7331,
LR 7389, muon switch 7093-7134, gate updates 7062/7181/7214, resume-into-finisher 6121-6157 + 6565,
resume-tau 6468-6483, sidecar 578-683) · `test_tau_advance_self_paced.py` (23 green) ·
`test_event_wirings.py` (no fired-state resume test — the documented split) · CLAUDE.md
(resumability + confound-self-protection + deterministic-repro non-negotiables).

## FINDINGS

### MAJOR-1 — EVENT-MUON crash-resume determinism break (unclosed; blocks an event-muon launch)
`_resume_into_finisher` (6127) and `muon_switched` (6565) are reconstructed from the CAP epoch
`args.muon_start_epoch`, NOT from the actual muon-fire epoch. The muon-fire epoch is NOT persisted
(no `__muon_*` sidecar; the 3 gates' `_fired_epoch` is the memo's "documented split"). In CLOCK/cap
muon these are identical (switch == cap) so resume is correct. In **event-muon** (`--muon-start-event
powerlaw_meat`, DRAFT §1) the sensor fires < cap (e.g. 650 < 726). A crash between fire and cap →
resume at 700 → `_resume_into_finisher = 700 > 726 = False` →
  (a) fresh AdamW is restored against a Muon-MultiOptimizer checkpoint → key mismatch;
  (b) `muon_switched=False` → the loop re-enters the switch → muon re-switches, LOSING 650→700
      momentum → non-bit-identical continuation;
  (c) if tau=event too, `__ta_frozen=True` is restored while `muon_switched=False` → line 7294
      calls `maybe_advance`, which ASSERTS `not frozen` (tau_advance.py:303) → HARD CRASH.
This violates the crash-resumability non-negotiable for the DRAFT's event-muon run. It is the SAME
root as the memo's acknowledged gate-fired-state split, but it is NOT benign here because
`muon_switched` (and the optimizer identity) are coupled to it. FIX: persist the muon-fire epoch;
reconstruct `_resume_into_finisher`/`muon_switched` from `start_epoch > fired_epoch` (fall back to
cap when unfired); round-trip the 3 gates' `_fired_epoch` via the same `__ta_`-style store. Note:
event-TAU alone (with clock/cap muon) is resume-SAFE (`__ta_*` persisted + tested); the break is
entirely the event-MUON path.

### MINOR-1 — same-epoch event∧cap: cap_fired_before_event row suppressed
maybe_advance (312-318): event wins when both fire the same epoch, so no LOUD cap row — CORRECT
attribution (the sensor DID fire), and dwell==cap is visible in the event row. But a sensor firing
exactly at the dwell==cap boundary is borderline-suspicious and unflagged. Additive `dwell_at_cap`
bool on the event telemetry would close the S5 observability gap. Non-blocking; consistent with
EventBackstopGate priority (event_wirings 160-168).

### MINOR-2 — relaxation sensor grades the EMA shadow (lag), not live weights
The verdict d_seg the sensor ingests is EMA-shadow-graded (deploy authority, correct). The shadow
LAGS live weights → an octave is declared relaxed slightly LATE (conservative, deterministic, not
corrupting). Benign; noted for completeness of the EMA-vs-live axis (d).

### MINOR-3 — octave dwell is verdict-cadence-coupled
Dwell is loop-epoch; history is verdict-epoch (async ~30 min, decide-on-previous). Deterministic AT
a fixed cadence and resume-safe (re-ingest from `_last_seen`), but a resume with a DIFFERENT
`--verdict-every` would not reproduce the τ trajectory. `--verdict-every` is part of the τ-advance
determinism contract — document it as a resume-invariant.

### CLEAN axes
(a) self-confirming/deadlock: τ is HELD fixed within an octave → the powerlaw_meat relaxation is a
genuine fixed-target signal; advancing CLEARS `_octave_hist` (no self-feed); the max-dwell cap ALWAYS
backstops with a LOUD `cap_fired_before_event` row; each advance re-arms (fresh octave). NO
spike-guard-style accepted-only freeze — `_octave_hist` grows on every verdict, `_last_seen`
monotone, a stalled sensor cannot silently freeze the ladder (cap fires loud). (c) verdict-cadence:
decide-on-previous, deterministic on values, tested. (e) tau-advance resume: `__ta_*` full state +
`test_resume_mid_octave_reproduces_identical_subsequent_tau_sequence` green.

## MODE RECOMMENDATION (operator decides)
**`--tau-advance-mode clock` for the FIRST v7 run.** Reasoning chain: (1) campaign one-continuation-
param — run-1's load-bearing change is the unify-L_τ dissolution of the PR95 CE→tau skeleton; adding
event-τ (S6-R4 rates it MED-LOW) confounds attribution of that primary result. (2) Event-τ couples
τ, β, LR to an UNPROVEN within-octave sensor on its FIRST real measurement — isolate the sensor,
don't ride it. (3) The flip to event for run-2 is one token and loses little: lane/chroma would-fire
calibration accrues regardless (S3 design); only the tau would-advance signal is forgone. (4) The
MAJOR-1 event-muon resume break shows the event-family fired-state hardening is incomplete — a clean
run-1 should not depend on it. This HONORS the operator's event directive as SEQUENCING (event is
BUILT + correct for steady state), not refusal. Coupled recommendation: pair clock-τ with **clock/cap
muon** for run-1 (avoids MAJOR-1 entirely and keeps a single-param first measurement); OR, if the
operator wants events live in run-1, close MAJOR-1 (persist muon-fire epoch) FIRST — event-τ + event-
muon on an unresumable optimizer switch is the one combination to not launch.

## VERDICT: NOT_CLEAN
One MAJOR (event-muon crash-resume break) must be dispositioned before an event-muon launch. The
RECOMMENDED run-1 config (clock-τ + clock/cap-muon) sidesteps it and is resume-clean; MINOR-1..3 are
non-blocking hardening items. [no-triality]
