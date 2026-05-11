# Xray CUDA-score input hardening (2026-05-11)

Status: landed; diagnostic only; no score claim.

## Bug class

`tools/xray_cpu_cuda_drift_per_arch_class.py` accepted a bare
`--cuda-score <float>`. That made it too easy to feed a rounded, stale, CPU,
MPS, proxy, or chat-derived number into the CPU/CUDA drift predictor and then
treat the prediction as if it was tied to a real contest-CUDA artifact.

## Fix

The normal path is now:

```bash
.venv/bin/python tools/xray_cpu_cuda_drift_per_arch_class.py \
  --archive experiments/results/.../archive.zip \
  --cuda-auth-eval-json experiments/results/.../contest_auth_eval.json
```

The tool parses the JSON with `tools.auth_eval_records.parse_auth_eval_payload`
and refuses inputs unless the record is `score_axis=contest_cuda`, `n_samples=600`,
and `gpu_t4_match=true`.

Manual numeric `--cuda-score` remains only as an explicitly tagged diagnostic
escape hatch and now requires `--manual-cuda-score-justification`. The artifact
records `cuda_score_source=manual_cuda_score_diagnostic` so it cannot be
mistaken for artifact-backed CUDA evidence.

The tool also now fails closed when classification resolves to
`unknown_uncalibrated` unless the operator passes
`--allow-unknown-architecture-class`. This prevents wrong-shape metadata, such
as a device-axis matrix analysis JSON, from silently producing an HNeRV-style
CPU prediction under the wrong architecture class.

## Score-lowering relevance

This protects the score-lowering queue from apples-to-oranges CPU/CUDA drift
decisions. The predictor can still triage whether a CUDA candidate deserves a
CPU dispatch, but its default input now has custody back to the scored archive
and evaluator output instead of a free-floating number.
