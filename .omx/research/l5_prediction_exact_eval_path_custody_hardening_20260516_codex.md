# L5 prediction/exact-eval path custody hardening (2026-05-16)

## Problem

Prediction-band anchor custody had been hardened to require artifact/log file
existence, but two custody gaps remained:

- baseline artifacts still only required a non-empty path string;
- shared exact-eval custody accepted existing absolute paths outside the repo
  and transient paths such as `/tmp/...`.

Those are false-authority risks for L5-v2 and Cathedral: a stale, private, or
ephemeral file could let a planning band look evidence-backed.

## Fix

- `validate_exact_eval_evidence()` now rejects transient artifact/log paths and
  absolute or relative paths that resolve outside the supplied artifact base
  directory.
- `validate_prediction_band()` now validates the baseline artifact path under
  the same explicit artifact base directory used for landed empirical anchors.

New blocker surfaces include:

- `log_path_transient`
- `log_path_outside_base_dir`
- `artifact_path_transient`
- `artifact_path_outside_base_dir`
- `prediction_band_baseline_artifact_base_dir_missing`
- `prediction_band_baseline_artifact_transient`
- `prediction_band_baseline_artifact_outside_repo`

## Verification

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_exact_eval_custody.py \
  src/tac/tests/test_prediction_band.py \
  src/tac/tests/test_l5_staircase_v2.py \
  src/tac/tests/test_l5_v2_probe_disambiguator.py -q
```

Result: `74 passed`.

## Authority

This is custody hardening only. It creates no score claim, promotion claim, or
dispatch claim. It makes repo-local durable evidence mandatory before
prediction-band anchors can influence rank authority.
