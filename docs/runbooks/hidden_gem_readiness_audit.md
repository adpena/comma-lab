# Hidden-Gem Readiness Audit

Use this audit after recovery intake or before choosing the next hidden-gem
patch. It checks the static hidden-gem registry against live repo files and
reports missing evidence paths or integration targets without reading provider
state, launching jobs, or making score claims.

```bash
python tools/audit_hidden_gem_readiness.py --format markdown
python tools/audit_hidden_gem_readiness.py --format json --status ready_for_patch
python tools/audit_hidden_gem_readiness.py --format json --fail-if-missing-targets
```

The audit can mark a row `ready_for_local_patch` when all referenced evidence
and integration targets exist and the registry status is `ready_for_patch`.
It always keeps `ready_for_exact_eval_dispatch=false`; exact CUDA dispatch still
requires the normal archive custody, manifest preflight, lane claim, and
contest auth-eval workflow.
