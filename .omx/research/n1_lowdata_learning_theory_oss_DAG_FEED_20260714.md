# FEED — n=1 learning theory + OSS for the costate organ (2026-07-14)

**FEED id:** `FEED-n1-lowdata-costate-20260714`

**lane:** `lane_n1_lowdata_learning_theory_oss_20260714`

**status:** `research_only=true`; pointer unchanged; no launch, implementation, or adoption

**source:** `codex_findings_n1_lowdata_learning_theory_oss_20260714_codex.md`

## Decision delta

The next $0 candidate is not another free neural architecture and not a duplicate GP. The organ already
contains `T_gp_costate_posterior`, while current B/C/D learned arms lose on the one real trajectory. Add
only after implementation review:

```text
existing fixed prior modes {Q_priormean_iso, P_priormean_aniso}
  -> subtract physics-prior mean from interval-rate target
  -> U_hierarchical_physics_residual
       [conjugate block shrinkage; past-only evidence; posterior covariance;
        P/Q prior-mode disambiguator on identical folds]
  -> existing real-only LOO diagnostic + primary walk-forward gate
       comparators: persistence, A, P, T, prototypes, dispatcher
  -> INSTANCE result
       win: provisional until >=3 independent trajectory records
       loss: current formulation only; queue structured-GP residual / #434 / meta-prior
```

## Coordination edges

- `FEED-433`: comma10k/openpilot/scorer geometry may supply fixed prior directions or frozen features;
  it is not an independent costate-labeled meta-set.
- `FEED-434`: source-faithful transient-rich simulation may pretrain a residual or sequence encoder;
  synthetic evidence never replaces the real-only gate.
- `FEED-426/436`: preserve the existing GP forecast and regime dispatcher as comparators; U addresses
  the missing per-lever response residual, not total-forcing forecast.
- Olmo/Gated-DeltaNet: sequence-arm reformulation queue only. Full target training waits for #434 or
  independent trajectories; frozen encoder + U-style readout is the n=1-compatible form.

## Triality

- **DSL:** candidate enum only after implementation and tests; no invented flag in this feed.
- **DAG:** this standalone feed avoids collision with the shared hot DAG.
- **Equation:** conjugate residual posterior recorded in the findings memo. Registry registration waits
  for a real producer/consumer and backtest receipt (`FORMALIZATION_PENDING`).

## Gates

1. deterministic NumPy reference and MLX parity surface;
2. no hyperparameter target leakage—each fold selects from a preregistered grid using its past prefix;
3. class-weighted walk-forward MAE primary; per-class/sign/calibration/regime stress secondary;
4. identical folds against persistence/A/P/T/prototypes/dispatcher;
5. no adoption on one mean-only win; `GRADUATION_MIN_RECORDS=3` remains binding;
6. all negatives `INSTANCE × FORMULATION`, with optimal-form reformulation queue.

**Pointer delta:** none. **Run/launch/dispatch delta:** none. **Score authority:** none.
