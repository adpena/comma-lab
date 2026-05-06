# Paradigm-Alpha Readiness Audit

Use this audit before choosing the next mask-payload overhaul patch. It checks
the four alpha candidates against live repo paths, hidden-gem registry coverage,
runtime source hooks, and current empirical JSON. It does not launch jobs or
make score claims.

```bash
python tools/audit_paradigm_alpha_readiness.py --format markdown
python tools/audit_paradigm_alpha_readiness.py --format json \
  --output .omx/research/paradigm_alpha_readiness_$(date -u +%Y%m%d)_codex.json
python tools/audit_paradigm_alpha_readiness.py --fail-if-missing-core
```

The audit keeps `ready_for_exact_eval_dispatch=false` for every row. Exact CUDA
dispatch still requires the normal archive custody, dispatch claim, preflight,
and `experiments/contest_auth_eval.py --device cuda` evidence path.
