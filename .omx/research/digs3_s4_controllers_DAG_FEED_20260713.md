# DAG FEED-digs3-s4 (2026-07-13) — finite-bank acquisition refusal + transactional-option fixed-schedule gate

`lane_id=lane_digs3_s4_20260713`; `checkpoint_id=digs3_s4`; `$0`; `[macOS-CPU advisory]`;
`actuation=NONE`; `pointer_moved=false`; source memo
`.omx/research/digs3_s4_controllers_20260713.md`.

## S3 edge

```text
crosswalk charter epoch 31bb1e3
  -> 73 exact DSL factories
  -> canonical fired+measured discharge rule
  -> 72 owed finite-bank rows
  -> strict exact-descriptor + structured-n600-outcome + exact-cost custody join
  -> 0 complete rows / 0 chronological folds
  -> {VIME, posterior, pseudo-count, Double-Q, P8, cheapest, round-robin, random}
       all NOT_IDENTIFIED
  -> primary falsifier FIRED: uncertainty uncalibrated
  -> stratified cheapest-first fallback; P8 floor/axis veto preserved; RND/ICM rejected
```

Double-Q is a future cross-fit guard only: selection and evaluation use disjoint past-data splits.
It cannot create offline support or response outcomes.

## S4 edge

```text
5 real read-only run logs
  -> 82 verdict + 130 checkpoint rows
  -> existing loss/topology/powerlaw sensors replayed
  -> raw-loss plateau can fire while d_seg continues descending
  -> annulus guard 0 fires (4 x 25-epoch samples span 75 < configured 150-epoch dwell)
  -> 0 explicit common-checkpoint common-horizon {stay,advance} pairs
  -> no sensor establishes lower counterfactual loss than existing gates
  -> primary falsifier FIRED: no affordable common-horizon counterfactual
  -> PRESERVE_FIXED_SCHEDULE; organ advisory; no live actuation
```

Conditional future edge, not a closed measured law:

```text
existing eligibility AND dwell/hysteresis
AND UCB(L_advance(H) + C_switch) < LCB(L_stay(H))
AND rollback-loadable pre-boundary checkpoint
  -> ADVANCE
else -> {STAY | ROLLBACK | INSUFFICIENT_INFORMATION}
```

## Receipts and reactivation

- `experiments/results/digs3_s4_20260713/s3_finite_bank_backtest.json`
- `experiments/results/digs3_s4_20260713/s4_option_trace_backtest.json`
- `experiments/results/digs3_s4_20260713/receipt_manifest.json`
- Reactivate S3 learning only after at least two chronological custody-complete rows create one real
  test fold; promotion needs calibrated held-out uncertainty and a win against P8/cost/round-robin.
- Reactivate S4 option selection only after both continuations start from the identical preserved
  checkpoint and run the identical preregistered horizon with full-facet receiver scoring.
- Equations leg remains `FORMALIZATION_PENDING`: neither empirical law closes under zero folds/pairs.
