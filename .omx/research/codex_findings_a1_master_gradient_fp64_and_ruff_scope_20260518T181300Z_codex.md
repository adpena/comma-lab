# Codex Findings: A1 Master-Gradient fp64 Diagnostic + Ruff Scope

Date: 2026-05-18T18:13:00Z

Actor: Codex

Task: `codex_routing_directive_inflate_py_plus_wyner_ziv_compliance_plus_master_gradient_extension_20260518::ITEM_3`

## Verdict

`tools/extract_master_gradient.py` now has end-to-end evidence for the A1 headered PR101-family grammar path: the tool can detect `a1_finetuned`, decode the headered layout, run the differentiable scorer path, and materialize aggregate plus per-pair fp64 gradients for the charged archive.

This is diagnostic CPU subset evidence only. It is not `[contest-CPU]` or `[contest-CUDA]` authority because it used `--axis '[diagnostic-CPU]'`, `--n-pairs-used 8`, local macOS CPU hardware, and `--no-anchor-write`.

## Archive

- Archive: `experiments/results/a1_cuda_cpu_drift_discriminator_v_baseline_20260509T110211Z/submission_dir/archive.zip`
- Inflate path: `experiments/results/a1_cuda_cpu_drift_discriminator_v_baseline_20260509T110211Z/submission_dir/inflate.py`
- Charged archive bytes: `178262`
- Charged archive SHA-256: `87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5`
- Gradient subject domain: `zip_inner_member_payload`
- Gradient subject bytes: `178162`
- Gradient subject SHA-256: `8e664385af0a25ec98bd02d97b697fbf0d2bb3c2d954f5aa5c95b5131330a243`
- Detected grammar: `a1_finetuned`

## Empirical Runs

Grammar detection:

```bash
.venv/bin/python tools/extract_master_gradient.py \
  --archive experiments/results/a1_cuda_cpu_drift_discriminator_v_baseline_20260509T110211Z/submission_dir/archive.zip \
  --detect-grammar-only
```

One-pair fp64 diagnostic:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/extract_master_gradient.py \
  --archive experiments/results/a1_cuda_cpu_drift_discriminator_v_baseline_20260509T110211Z/submission_dir/archive.zip \
  --inflate-py experiments/results/a1_cuda_cpu_drift_discriminator_v_baseline_20260509T110211Z/submission_dir/inflate.py \
  --upstream-dir upstream \
  --axis '[diagnostic-CPU]' \
  --output-npy .omx/state/master_gradient_a1_headered_diagnostic_1pair_20260518.npy \
  --per-pair-output-npy .omx/state/master_gradient_a1_headered_diagnostic_1pair_per_pair_20260518.npy \
  --device cpu \
  --n-pairs-used 1 \
  --n-pairs-total 600 \
  --preserve-per-pair \
  --compute-dtype float64 \
  --storage-dtype float64 \
  --no-anchor-write \
  --verbose
```

Eight-pair fp64 diagnostic:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/extract_master_gradient.py \
  --archive experiments/results/a1_cuda_cpu_drift_discriminator_v_baseline_20260509T110211Z/submission_dir/archive.zip \
  --inflate-py experiments/results/a1_cuda_cpu_drift_discriminator_v_baseline_20260509T110211Z/submission_dir/inflate.py \
  --upstream-dir upstream \
  --axis '[diagnostic-CPU]' \
  --output-npy .omx/state/master_gradient_a1_headered_diagnostic_8pair_20260518.npy \
  --per-pair-output-npy .omx/state/master_gradient_a1_headered_diagnostic_8pair_per_pair_20260518.npy \
  --device cpu \
  --n-pairs-used 8 \
  --n-pairs-total 600 \
  --preserve-per-pair \
  --compute-dtype float64 \
  --storage-dtype float64 \
  --no-anchor-write \
  --verbose
```

## Tensor Artifacts

These `.npy` sidecars are ignored state artifacts and are intentionally not committed.

| Path | Shape | Dtype | SHA-256 | Notes |
|---|---:|---|---|---|
| `.omx/state/master_gradient_a1_headered_diagnostic_1pair_20260518.npy` | `(178162, 3)` | `float64` | `06feb21fb6f34e262dfbea96fe703941ea9d3dd40afca265132c583d9aa31c71` | finite; absmax `3.52625283995e-05`; nonzero `323904` |
| `.omx/state/master_gradient_a1_headered_diagnostic_1pair_per_pair_20260518.npy` | `(178162, 1, 3)` | `float64` | `0a418165febbcc8b2a4c88d226035acd8d14e7842e7f79b5070ce20e89e450e3` | finite; absmax `3.52625283995e-05`; nonzero `323904` |
| `.omx/state/master_gradient_a1_headered_diagnostic_8pair_20260518.npy` | `(178162, 3)` | `float64` | `471c5a402770e31787a56bf1e478bdb442b9fc7265459ee6d9fb4fc4cd0394b1` | finite; absmax `3.60958110832e-05`; nonzero `323904` |
| `.omx/state/master_gradient_a1_headered_diagnostic_8pair_per_pair_20260518.npy` | `(178162, 8, 3)` | `float64` | `dcd6513778d360561c5f4b80b4169fff59aee59f22a5b102716645c3a8fcdc7b` | finite; absmax `6.1124075728e-05`; nonzero `2591232` |

## Operating Point From 8-Pair Diagnostic

- `d_seg=0.0009`
- `d_pose=0.00175394`
- `rate=0.004748`
- `score=0.3393`
- `dS/d_seg=100.0`
- `dS/d_pose=37.75`
- `dS/d_byte=6.659e-07`
- Forward plus 16 backward passes completed in `58.70s`

## Ruff Scope Finding

The operator hypothesis "perhaps the ruff needs to be reconfigured" was partly correct. Ruff was not misconfigured for `tools/extract_master_gradient.py` locally: focused Ruff passed for the extractor and tests.

The actual weakness was CI routing:

- `.github/workflows/ci.yml` blocked F821 over `src/ experiments/ submissions/robust_current/ scripts/`, but omitted `tools/`.
- The same CI command over `experiments/` pulled in generated custody trees under `experiments/results/`, which are ignored artifact state and already contain stale public-intake code.

Fix landed in this batch:

- `tools/` added to the blocking F821 CI command.
- `--force-exclude` added to Ruff CI calls.
- `experiments/results` added to Ruff `extend-exclude`.
- `src/tac/tests/test_ci_ruff_scope.py` added to prevent drift.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q -p no:cacheprovider \
  src/tac/tests/test_ci_ruff_scope.py src/tac/tests/test_extract_master_gradient.py
# 36 passed

.venv/bin/ruff check pyproject.toml src/tac/tests/test_ci_ruff_scope.py \
  tools/extract_master_gradient.py src/tac/tests/test_extract_master_gradient.py
# All checks passed

.venv/bin/ruff check --force-exclude --select F821 \
  src/ experiments/ submissions/robust_current/ scripts/ tools/
# All checks passed; one pre-existing invalid-noqa warning in submissions/robust_current/inflate_renderer.py:5348
```

## Next Action

ITEM_3 still needs full registry coverage before closure:

- PR101_lc_v2 materialization at the same 8-pair fp64 diagnostic scale.
- Unsupported grammar projectors or explicit fail-closed status for PR106 format0d, PR106 packed HNeRV, HNeRV length-prefixed, and PR107 Apogee.
- A full 600-pair authoritative run should only be claimed on a contest-authoritative hardware axis with `--no-anchor-write` removed and custody fields populated.
