# MPS research-signal protocol - 2026-05-07

## Verdict

Use the local Apple MPS device aggressively for free discovery sweeps, but keep
the outputs out of score evidence, promotion, kill, retirement, and paper
empirical claims.

The implementation surface is:

- `src/tac/optimization/mps_research_signal.py`
- `tools/build_mps_research_signal_manifest.py`
- `src/tac/tests/test_mps_research_signal.py`

## Contract

Every MPS sweep row promoted into planning must be serialized with:

- `evidence_grade="MPS-research-signal"`
- `evidence_semantics="mps_proxy_curve_shape_only"`
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `dispatchable=false`

The meta-Lagrangian adapter now treats evidence grades containing `mps`, `cpu`,
or `advisory` as proxy rows. Even if a row has attractive proxy deltas, it is
not score evidence and is not planning-priority rankable.

## Smoke Artifact

Smoke artifact:

`experiments/results/mps_research_signal_smoke_20260507_codex/`

Files:

- `observations.json`
- `manifest.json`
- `atom_ledger.json`

The smoke manifest confirms a three-point MPS proxy curve can be converted into
candidate-generation priors while all exported atoms remain proxy-only and exact
CUDA blocked.

## Next Use

Run overnight MPS sweeps for arch-shrink, sparse retraining, pose/TTO proxy
curves, and foveation geometry. Convert each sweep with the canonical adapter,
then let the autopilot pick CUDA dispatch candidates from the resulting priors.
Exact CUDA auth eval remains the only score truth.
