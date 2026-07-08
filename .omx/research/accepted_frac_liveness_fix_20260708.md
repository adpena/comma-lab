# C6 liveness confound: mid-epoch `accepted_frac` reads 0.0 on a LIVE, stepping run

**Date:** 2026-07-08 · **Class:** confound (liveness sentinel misreporting) · **Surface:** telemetry-only, score-neutral · **[no-triality]** (pure bug-fix; no lever/finding)

## Signature (the operator-caught confound)
`experiments/results/levelset_n600_crucible_v6_run1_20260708T095730Z/run.log` (READ-ONLY evidence):
every `{"stage":"loss_terms"}` row (one per epoch, `accum_batch:0`, ep1–31) emits
`"accepted_frac": 0.0` while `"weights_stepped": true`, `spike_skipped:false`, and `ep_loss`/loss
demonstrably DESCEND (seg 20.3→9.4, total 40.7→12.7 over ep1–3). A pessimistically-lying liveness
field — the L1 confound-alarm layer's OWN signal — caused a council seat to declare a live run dead.
This is exactly the direction the "Confound self-protection" non-negotiable warns about: the alarm
must be trustworthy in BOTH directions (a live run must NOT read as frozen).

## Stores consulted
- CLAUDE.md §"Confound self-protection" (L1/L2/L3 immune system; the C6 liveness counters).
- MEMORY.md L5 (spike-guard median-freeze + `ep_loss:0.0` = ALERT signature — the *opposite*, genuinely-frozen case).
- The run's `run.log` + `launch.sh` (`--eval-every 25 --accum-pairs 8` → 75 accum-batches/epoch; `_lt_stride` emits only `accum_batch:0`).
- `src/tac/confound_gates.py` (#402 telemetry-liveness schema gate), `tools/witness_run_introspect.py`, `tools/witness_annulus_live_monitor.py`, `tools/dashboard_server.py`, `src/tac/witness_control/*` (consumer audit).

## Root cause (off-by-one in the numerator at emission time)
Per accum-batch the loop does, in order:
1. `_live["ep_tot"] += 1` (L~7146) — counts the in-flight batch;
2. emit the `loss_terms` row (L~7171) with `accepted_frac=_live_running_frac()` — **HERE**;
3. `opt.update(...)` then `_live["ep_acc"] += 1` (L~7214) — records acceptance.

`_live_running_frac()` returned `ep_acc / max(ep_tot,1)`. At step 2 the current batch is already in the
denominator (`ep_tot`) but never in the numerator (`ep_acc` increments at step 3). On the FIRST batch of
every epoch that is `0/1 == 0.0` even though the batch steps. Because `_lt_stride` emits only
`accum_batch:0` in this config, EVERY row reads 0.0. (Even non-batch-0 rows are undercounted by exactly
one in-flight batch.)

## Fix (telemetry-only; score-neutral by construction)
`_live_running_frac(pending_accept)` now folds the current batch's already-decided accept/skip state
(`not skip`, known at the call site) into the numerator. Arithmetic extracted to a module-level,
unit-testable `_running_accepted_frac(ep_acc, ep_tot, pending_accept)`; the closure is a thin wrapper.
Call site: `accepted_frac=_live_running_frac(not skip)`. Result: 1.0 on a clean epoch, `<1.0` only on
real skips, and never spuriously 0.0. **Assertion of score-neutrality:** the diff touches ONLY the row's
numerator arithmetic + one call argument — it reads `_live["ep_acc"]`/`_live["ep_tot"]`/`skip`, never
writes them, never touches `opt.update`, gradients, EMA, weights, or archive bytes. Byte-identity of the
trained artifact is preserved by construction.

## Downstream consumers audit
- **AUTOMATED liveness readers are SAFE (unaffected by bug or fix).** `tools/witness_run_introspect.py::read_liveness_row` and `tools/witness_annulus_live_monitor.py` read `accepted_frac` ONLY from the `{"stage":"verdict"}` row, which uses the epoch-END snapshot `_live["frac"]` (set L~7277 = full-epoch `ep_acc/ep_tot`, correct). `tools/dashboard_server.py`'s liveness strip consumes `read_liveness_row` (verdict row) too. `src/tac/confound_gates.py` #402 is a schema/presence check on the field, not a value threshold.
- **The costate shadow controller does NOT consume `accepted_frac` at all** — grep of `src/tac/witness_control/*` (shadow_controller / producer_bridge / costate_estimator) finds zero reads of `accepted_frac`/`frozen_epoch`/`weights_stepped`. **No shadow-controller frozen-run misread risk.**
- **The corrupted surface was confined to the human/agent-eyeballed `loss_terms` row.** That is precisely what misled the council seat: the automated readers were immune (verdict-row snapshot), but a reader eyeballing raw `loss_terms` rows saw `0.0` and inferred death. Note `witness_annulus_live_monitor.py:241-242` already *rationalized* the 0.0 as "legitimately 0.0 at emission time" — the confound was hiding in plain sight as an accepted quirk.

## The live run carries the artifact (documented)
The live run (`levelset_n600_crucible_v6_run1_...`) imports the OLD module; this fix lands for **v7 /
resumes**. Run-1's `loss_terms` rows BEFORE this fix carry the artifact (`accepted_frac:0.0` on stepping
epochs). Its VERDICT rows are unaffected (epoch-end snapshot). No re-run needed for the fix's sake.

## Tests
`src/tac/tests/test_accepted_frac_liveness.py` — 9 tests over the actual shipped `_running_accepted_frac`:
first-batch-clean→1.0 · first-batch-skipped→0.0 · mid-epoch-all-accepted→1.0 · mid-epoch-real-skip<1 ·
current-accepts-after-prior-skip · no-batch-ran edge (ep_tot=0 → alive default, no div-by-zero) ·
pending=None raw-fraction back-compat · fold-strictly-increases-vs-prefix · closure-wired-module-level.

## Coordination note
A sibling agent was concurrently editing the trainer's `--verdict-device`/`gpu_verdict` path in the same
working tree. This landing was hunk-isolated (HEAD-restore → re-apply the 3 liveness hunks → commit → sibling
WIP restored) so it commits ONLY the liveness counters, NO sibling absorption.
