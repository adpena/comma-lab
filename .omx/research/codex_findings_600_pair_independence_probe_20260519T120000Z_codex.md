# Codex Findings: 600-Pair Independence Probe

Timestamp: 2026-05-19T12:00:00Z  
Owner: codex  
Scope: Cluster F.2 local diagnostic probe  
Result: `SERIAL_DEPENDENCE_DETECTED`

## What Changed

Added `tools/test_600_pair_independence.py`, a dependency-light local diagnostic
for existing JSON artifacts with 600-pair numeric vectors. The tool does not run
the scorer, dispatch jobs, or create score evidence. It emits:

- lag autocorrelation through configurable `--max-lag`
- Ljung-Box Q with Wilson-Hilferty chi-square survival approximation
- median-split runs test
- adjacent Pearson/Spearman correlation
- positive-lag effective sample size estimate
- absolute-contribution concentration, including top-1/top-10/top-50 share
- cross-series Pearson/Spearman dependence summary
- explicit false-authority fields:
  `score_claim=false`, `score_claim_valid=false`,
  `promotion_eligible=false`, `rank_or_kill_eligible=false`,
  `ready_for_exact_eval_dispatch=false`, `dispatchable=false`,
  `auth_eval_skipped=true`, `scorer_invoked=false`, and
  `provider_invoked=false`

Focused tests landed in `src/tac/tests/test_600_pair_independence_tool.py`.

## Real Artifact Run

Command:

```bash
.venv/bin/python tools/test_600_pair_independence.py \
  --input-json experiments/results/lane_local_hardware_maximization_sweep_20260513_20260513T210232Z/per_pair_sensitivity_a1_baseline_600pair.json \
  --input-json experiments/results/d1_pair_component_xray_a1_baseline_cpu_20260515_codex/pair_component_xray.json \
  --input-json experiments/results/pr101_frame_conditional_runtime_packet_20260508_codex/per_pair_score_marginals.advisory.json \
  --max-lag 32 \
  --output-json experiments/results/600_pair_independence_test_20260519T120000Z_codex/report.json \
  --summary
```

Ignored local report:

- path: `experiments/results/600_pair_independence_test_20260519T120000Z_codex/report.json`
- sha256: `c56355b3472eb99e2d3368a21ea4304c36c6856b9d0099fa245538689080117d`
- schema: `pair_independence_diagnostic_v1`
- evidence grade: `local_independence_diagnostic_only`

## Findings

The 600-pair iid independence assumption is not defensible for the tested
vectors. The probe analyzed 16 numeric per-pair series and returned aggregate
`independence_assumption_blocked`. Cross-series dependence also returned
`cross_vector_dependence_blocked`.

High-signal examples:

- `raw_per_pair.pose_per_pair`: max abs autocorr `0.4854`, serial ESS `40.22`,
  top-50 absolute share `0.3112`
- `raw_per_pair.seg_per_pair`: max abs autocorr `0.6032`, serial ESS `21.25`,
  top-50 absolute share `0.1162`
- `rows[*].component_score_no_rate`: max abs autocorr `0.5465`, serial ESS
  `23.61`, top-50 absolute share `0.1141`
- `rows[*].pose_dist`: max abs autocorr `0.4854`, serial ESS `40.22`, top-50
  absolute share `0.3112`
- `rows[*].seg_dist`: max abs autocorr `0.6032`, serial ESS `21.25`, top-50
  absolute share `0.1162`
- `per_pair_score_marginals`: max abs autocorr `0.2567`, serial ESS `85.63`,
  top-50 absolute share `0.2007`; blocked by low serial effective-N, not a
  clean iid pass

## Consequence

Downstream solvers, pair selectors, bootstrap estimates, and Cathedral
autopilot priors should not assume iid 600-pair rows. The safer local model is
block-aware or temporally correlated pair structure. This does not rank, kill,
or promote any lane; it only tightens the assumption surface.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_600_pair_independence_tool.py -q
# 4 passed
```
