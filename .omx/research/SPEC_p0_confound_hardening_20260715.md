# SPEC — P0 confound-pass follow-on hardening

Date: 2026-07-15
Catalog claim: NOT CLAIMED — the canonical counter offered #405, which is already occupied in this tree.
Lane: `p0_confound_hardening_20260715`
Scope: BUILD + local verify only; score-neutral observability/alarm additions; no GPU, no dispatch, no score claim.

## Hard boundaries

- Do not read, write, enumerate, or otherwise touch `experiments/results/levelset_n600_witness_20260715T095030Z`.
- No training launch and no provider/GPU action.
- The additions may read already-available telemetry/model snapshots and emit rows or alarms. They must never write model parameters, optimizer state, EMA state, gradients, controller state, archive payloads, or training decisions.
- Preserve explicit `VerdictLiveGap(every=K)` DSL semantics: a positive cadence continues to observe every Kth verdict for the full run.

## Required landing

### A. L1 partial-freeze alarm

Add a typed `confound_alarm` with alarm name `partial_freeze`, level `WARN`, and the exact open band `0.02 < accepted_frac < 0.5` to both the base and level-set witness trainer alarm layers. The alarm emits epoch, accepted/skipped counts where available, and a note that progress is slow/partially frozen. It never mutates training state.

### B. L3 global d_seg-descent canary

Extend the existing verdict-trend canary apparatus with an explicit known-effect d_seg-descent positive control. A deterministic synthetic descending d_seg trace must be positively registered, not merely described as an alarm-negative sample. At every level-set run setup:

- execute the $0 pure-Python canary;
- emit a typed `dseg_descent_canary_setup` row;
- emit a loud `confound_alarm` if the canary or its L3 clearance fails;
- stamp every async and sync verdict row with the canary pass, descent-positive registration, and L3 clearance fields.

The L3 verdict-clearance helper is pure and fail-closed as a classification, but the trainer only alerts: it must not suppress verdict emission or change any controller/training decision. This preserves score/byte identity even if the canary fails.

### C. verdict-live-gap default-on during EMA warmup

Change the parser default from `0` (off) to `-1` (automatic warmup mode):

- `-1`: run live-gap inference on each scheduled verdict only while accepted EMA updates are below `ceil(2 / (1 - ema_decay))`;
- `0`: explicit operator opt-out;
- `K > 0`: existing explicit DSL behavior, every Kth verdict for the full run.

Track accepted EMA updates only for this read-only cadence predicate; do not persist it or feed it into training. Increment it alongside successful `ema.update(model)`. A resume may conservatively repeat warmup observations; that is safe because the state is observational only. Update the existing `VerdictLiveGap` DSL documentation to describe explicit all-run semantics and the trainer's auto-warmup default. Do not add a new CLI flag.

### D. STRICT preflight protection

In `src/tac/confound_gates.py`, add three static gates with focused synthetic-fixture tests:

1. `check_witness_trainers_emit_partial_freeze_alarm`
2. `check_witness_verdict_rows_carry_dseg_descent_canary`
3. `check_verdict_live_gap_defaults_on_during_ema_warmup`

Wire all three into `CONFOUND_GATES` and add them to `_CONFOUND_STRICT` in `src/tac/preflight.py`. Live-count is zero in this atomic fix+gate landing, so strict-flip now. Gates must detect removal/miswiring rather than accept comments alone. Update registry-count/bound tests.

### E. Triality and proof

- Add a canonical-equations note/module that derives `U_warm = ceil(2/(1-beta_ema))`, states the partial-freeze interval, and records the d_seg known-descent predicate.
- Add a dated DAG FEED receipt linking DSL lever, trainer consumers, L1/L3 telemetry, gates, tests, and byte-neutrality.
- Add focused tests for canary positive/negative behavior, L3 clearance, gate counterexamples, parser/default semantics or source-contract predicate, and real-repo strict-clean state.
- Run compile/focused tests and the relevant strict preflight surface. Mark every changed Python file with `tools/review_tracker.py`; commit via `tools/subagent_commit_serializer.py` using the post-edit expected content SHA.

## Acceptance

All three anti-pattern gates have live-count zero and run STRICT; all focused tests pass; git diff contains no write to the named live C0 run directory; the byte-neutral proof demonstrates that no added value reaches loss/backward/optimizer/EMA/model/archive or controller decisions.
