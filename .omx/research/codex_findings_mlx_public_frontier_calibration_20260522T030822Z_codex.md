# Codex Findings: MLX Public-Frontier Score Calibration

UTC: 2026-05-22T03:08:22Z

## Verdict

PROCEED_WITH_GUARDS. MLX scorer-response is not a CUDA-score replacement, but it is now measured as a high-signal local training and spend-triage substrate for the current HNeRV/FEC6 frontier family. On the four leading public/frontier archives measured here, MLX preserves the CPU and CUDA ordering exactly and tracks local macOS CPU within micro-score tolerance where paired local CPU anchors exist.

## Authority

- Score claim: `false`
- Promotion eligible: `false`
- Rank-or-kill eligible: `false`
- Ready for exact-eval dispatch: `false`
- Evidence axis: `[macOS-MLX research-signal]`
- Exact contest CPU/CUDA auth eval remains required before any leaderboard, promotion, or rank/kill use.

## Scope

Measured full 600-pair MLX scorer-response against the canonical reference cache for:

- PR110 FEC6 fixed-Huffman k16
- PR101 hnerv_ft_microcodec
- PR103 hnerv_lc_ac
- PR102 hnerv_lc_v2_scale095_rplus1

PR110 was re-checked against GitHub during this pass: open, head `ec6cc7f98c16b6ad2db8bc7cde65757bb7993004`, archive SHA `6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf`, archive size `178517`.

## Calibration Artifact

Reusable manifest:

`experiments/results/mlx_public_frontier_calibration_20260522T023053Z/calibration_manifest.json`

Input rows:

`experiments/results/mlx_public_frontier_calibration_20260522T023053Z/calibration_rows.json`

Both are ignored under `experiments/results/*`; `.gitignore` already covers these generated artifacts and no ignore expansion was needed.

## Results

| PR | MLX | CPU anchor | MLX - CPU | local CPU | MLX - local CPU | CUDA T4 | CUDA - MLX | Rank |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 110 | 0.19206194316409206 | 0.1920513168811056 | +0.000010626282986453406 | n/a | n/a | 0.22621002169349796 | +0.034148078529405906 | 1 |
| 101 | 0.19286261624666795 | 0.1928450127024255 | +0.000017603544242461577 | 0.1928610127024255 | +0.000001603544242445576 | 0.22635331443973267 | +0.033490698193064716 | 2 |
| 103 | 0.19486454641280093 | 0.19488070288878895 | -0.000016156475988016172 | 0.19486470288878893 | -0.00000015647598800017093 | 0.2277649714224471 | +0.032900425009646156 | 3 |
| 102 | 0.19537689298408853 | 0.195376176526 | +0.0000007164580885232752 | 0.19537548831559584 | +0.0000014046684926882769 | 0.22839372989108092 | +0.03301683690699239 | 4 |

Summary:

- `mlx_cpu_rank_inversions`: `0 / 6`
- `mlx_cuda_rank_inversions`: `0 / 6`
- `mlx_minus_cpu_max_abs`: `1.7603544242461577e-05`
- `mlx_minus_local_cpu_max_abs`: `1.603544242445576e-06`
- `cuda_minus_mlx_mean`: `0.03338900965977729`
- `cuda_minus_mlx_min`: `0.032900425009646156`
- `cuda_minus_mlx_max`: `0.034148078529405906`

## Interpretation

MLX is not consistently worse than CPU in this sample. Against public CPU anchors it oscillates by about `+-1.8e-5`, and against local macOS CPU anchors it is within `1.7e-6`. PR103 is slightly better under MLX than its local CPU anchor, so the residual is not a one-sided pessimism bias.

The dominant gap is axis, not local MLX drift: CUDA T4 is consistently higher/worse by about `0.033-0.034` on these archives while preserving the same order. That makes MLX useful for local scorer-response learning, candidate generation, and cloud-spend filtering when expected deltas are comfortably above the measured uncertainty band. It is not sufficient for close calls, CUDA claims, or promotion.

The probable mechanism is ordinary heterogeneous floating-point execution plus scorer-backbone axis differences. Official CUDA documentation emphasizes that changing operation order, FMA use, precision, and parallel reductions can change floating-point results while remaining IEEE-754-compliant; PyTorch's CUDA determinism controls also do not make every operation universally deterministic or identical across devices. The engineering response is therefore calibration and canonicalization, not pretending there is a single device-independent scalar.

## Code Changes

- Added `tac.local_acceleration.mlx_score_calibration`.
- Added `tools/calibrate_mlx_scorer_response_scores.py`.
- Added calibration tests covering false-authority enforcement, rank-inversion accounting, and CLI output.
- Propagated the explicit MLX batch-shape research allowance through `tools/plan_ll_scorer_response_next.py`.
- Kept non-singleton batch selection blocked unless the caller opts into batch-shape research signal.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check \
  src/tac/local_acceleration/mlx_score_calibration.py \
  src/tac/local_acceleration/mlx_execution_plan.py \
  tools/calibrate_mlx_scorer_response_scores.py \
  tools/plan_ll_scorer_response_next.py \
  src/tac/tests/test_mlx_score_calibration.py \
  src/tac/tests/test_mlx_execution_plan.py \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_mlx_profile_stability.py \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py
```

Result: `All checks passed!`

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_mlx_score_calibration.py \
  src/tac/tests/test_mlx_execution_plan.py \
  src/tac/tests/test_mlx_scorer_response.py \
  src/tac/tests/test_mlx_profile_stability.py \
  src/tac/tests/test_profile_mlx_scorer_response_cache.py \
  src/tac/tests/test_mlx_production_contract.py
```

Result: `42 passed`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest $(rg --files src/tac/tests | rg 'mlx')
```

Result: `149 passed`.

```bash
git check-ignore -v \
  experiments/results/mlx_public_frontier_calibration_20260522T023053Z/calibration_rows.json \
  experiments/results/mlx_public_frontier_calibration_20260522T023053Z/calibration_manifest.json
```

Result: both covered by `.gitignore:55:experiments/results/*`.

## Next Action

Use MLX as a calibrated local response substrate with a decision threshold: safe for training signal and spend triage when expected deltas exceed the measured uncertainty band by a wide margin; exact CPU/CUDA auth eval for close deltas, CUDA transfer claims, and all promotion decisions. The next engineering target is full weight-portability/canonicalization so the same scorer state can be verified through NumPy/torch/MLX loaders before local training loops depend on it.
