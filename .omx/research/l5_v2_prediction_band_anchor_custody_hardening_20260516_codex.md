# L5-v2 prediction-band anchor custody hardening (2026-05-16)

## Problem

Read-only adversarial review found that landed empirical anchors inside
`tac.optimization.prediction_band` could satisfy rank-reward custody from
non-empty `artifact_path` and `log_path` strings alone. The shared exact-eval
custody validator only checks path existence when callers provide an explicit
artifact base directory.

That made a stale or missing exact-eval JSON/log path a false-authority risk:
the band could become `valid_for_rank_reward=true` without the scored artifact
and log being present in repo custody.

## Fix

`validate_prediction_band()` and `validate_optional_prediction_band()` now
accept `artifact_base_dir`. Landed anchors are rank-blocked unless that base
directory is supplied, and each anchor forwards it to
`validate_exact_eval_evidence()` so `artifact_path` and `log_path` must resolve
to real files.

New public blocker:

- `prediction_band_empirical_anchor_artifact_base_dir_missing`

Existing file-custody blockers are preserved:

- `prediction_band_empirical_anchor_artifact_missing`
- `prediction_band_empirical_anchor_log_missing`

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_prediction_band.py -q
```

Result: `16 passed`.

## Authority

This is a rank-authority hardening only. It creates no score claim, promotion
claim, or dispatch claim. Prediction bands remain planning priors until exact
CPU/CUDA evidence, archive/runtime custody, and contest formula closure are
present under explicit repo-root artifact custody.
