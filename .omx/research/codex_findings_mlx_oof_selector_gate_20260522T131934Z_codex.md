# Codex Findings: MLX OOF Selector Gate

Timestamp: 2026-05-22T13:19:34Z

## Scope

Adversarial hardening pass for the MLX full-600 scorer-response dataset and
OOF response predictor. This pass asks whether the OOF predictor is strong
enough to select decoder-q spend candidates, and installs fail-closed selector
surfaces so MLX can accelerate candidate generation without becoming score,
rank, promotion, kill, or dispatch authority.

## Code Landed

- `tac.optimization.scorer_response_prediction` now supports:
  - `model_family=expanded`, a nested out-of-fold ridge selector over linear,
    harmonic, RBF, and hybrid pair-local bases.
  - `fold_strategy=group_hash`, which groups sibling rows by
    `source_start_pair` so same-window rows do not cross outer train/test folds.
  - per-family utility metrics: per-family Pearson, negative prediction count,
    top-k overlap, and `spend_triage_usable`.
- `tac.optimization.scorer_response_dataset` now exposes
  `prediction_spend_triage_usable` and emits
  `do_not_use_oof_predictions_for_spend_triage_selection` when overall OOF
  correlation passes but no candidate family passes selector-utility metrics.
- `tools/fit_scorer_response_oof_predictions.py` exposes the expanded model,
  ridge grid, and grouped-fold options.
- `tac.optimization.mlx_effective_spend_triage_selection` and
  `tools/select_mlx_effective_spend_triage_candidates.py` select strict-gated
  observed MLX windows for candidate generation only. They require the composed
  strict effective MLX gate, preserve false authority, and require byte-closed
  archive materialization before any exact auth-eval dispatch.

## Empirical Result

Real full-600 expanded grouped OOF artifact:

- Dataset:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/candidate_same_axis_window_response_dataset_full600_oof_expanded_grouped.json`
- Validation:
  `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/candidate_same_axis_window_response_dataset_full600_oof_expanded_grouped_validation_gate.json`
- Overall validation status: `passed`
- Overall held-out Pearson: `0.32517898715374266`
- Prediction spend-triage usable: `False`
- Decoder-q family Pearson: `-0.0756861363816352`
- Decoder-q negative prediction count: `0`
- Decoder-q observed improvement count: `170`
- Decoder-q top-8 predicted/observed overlap: `0/8`

Conclusion: the upgraded OOF predictor is a dataset sanity signal, not a
decoder-q selector. It is more honest than the previous linear fit because it
records candidate-family failure directly instead of letting parent/control rows
mask the sign problem.

## Candidate-Generation Selector

Observed strict-gated selector artifact:

- `experiments/results/mlx_decoderq_parent_contract_closure_20260522T1132Z/mlx_effective_spend_triage_observed_window_selection_top32.json`
- Eligible decoder-q rows above the calibrated MLX decision gap: `148`
- Selected rows: `32`
- Prediction agrees with observed gain: `0/32`
- Best selected window: `[501, 502]`
- Best observed MLX gain: `0.0020326847010743165 [macOS-MLX research-signal]`
- Best byte-budget margin: `3052.725643386161`
- Required next step:
  `materialize_byte_closed_archive_before_claim_or_exact_eval_dispatch`

The prediction-negative selector was intentionally run as an adversarial check
and failed closed with `no rows survived strict MLX spend-triage selection`.

## Authority

All artifacts in this memo are `[macOS-MLX research-signal]` only:

- `score_claim=false`
- `promotion_eligible=false`
- `rank_or_kill_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `promotable=false`

The strict MLX effective gate may accelerate local candidate generation and
exact-eval spend triage for observed, contract-covered windows. OOF predictions
must not select spend candidates until candidate-family metrics become usable.
Every selected row still requires byte-closed archive materialization,
official inflate/raw-output custody, and claimed contest CPU/CUDA auth eval
before any score, rank, promotion, kill, or submission decision.

## Verification

```bash
ruff check src/tac/optimization/scorer_response_prediction.py \
  src/tac/optimization/scorer_response_dataset.py \
  src/tac/optimization/mlx_effective_spend_triage_selection.py \
  tools/fit_scorer_response_oof_predictions.py \
  tools/select_mlx_effective_spend_triage_candidates.py \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_mlx_effective_spend_triage_selection.py
```

Result: `All checks passed!`

```bash
.venv/bin/python -m pytest src/tac/tests/test_mlx_effective_spend_triage_selection.py -q
```

Result: `5 passed`.

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_scorer_response_dataset.py \
  -k "out_of_fold or structural_features or validation_gate or expanded_oof or effective_mlx" -q
```

Result: `10 passed, 83 deselected`.

```bash
.venv/bin/python -m pytest src/tac/tests/test_plan_ll_scorer_response_next_cli.py -q
```

Result: `10 passed`.

## Next Engineering Step

Build the missing byte-closed selector bridge:

1. Map selected MLX singleton windows into a selector/packet runtime contract
   without touching the dirty HFV sidecar WIP.
2. Materialize archives for the top observed windows and small adjacent runs.
3. Run official inflate/raw-output controls and local advisory checks.
4. Dispatch only byte-closed winners through claimed contest CPU/CUDA auth eval.
