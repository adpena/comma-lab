# Provider-Agnostic Deploy Contracts

This runbook is the deterministic deploy boundary for Modal, Kaggle, AWS,
Azure, and GCP surfaces. Lane experiments may bind lane labels and parameters;
provider lifecycle, setup blockers, custody requirements, and score-truth
rules live in `tac.deploy`.

Canonical registry:

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
from tac.deploy.provider_contracts import provider_contracts, validate_provider_contracts
for name, contract in provider_contracts().items():
    print(name, contract.status, contract.module, contract.execution_flag)
violations = validate_provider_contracts()
raise SystemExit(1 if violations else 0)
PY
```

Non-negotiable invariants:

- Real remote dispatch is plan-only by default; spend requires an explicit provider execution flag.
- Any training, eval, or remote-GPU dispatch must claim the lane with `tools/claim_lane_dispatch.py` before provider job creation.
- Every terminal outcome must append a terminal claim row for the same lane/job.
- Provider artifacts require a custody manifest with git state, command, hardware, archive bytes/SHA when applicable, logs, and harvest path.
- Kaggle and any other proxy/free substrate cannot promote, rank, kill, or claim score truth.
- MPS auth eval is never score truth.
- Modal/Azure are the currently implemented CUDA exact-eval provider surfaces.
  AWS/GCP are scaffolds only until lifecycle, quota, budget, and harvest
  contracts land. Any provider can host CUDA exact-eval only after lane claim,
  runtime closure, and adjudication over byte-closed artifacts.

Readiness command:

```bash
.venv/bin/python tools/cloud_provider_readiness.py \
  --output experiments/results/cloud_provider_readiness_latest.json \
  --markdown-output experiments/results/cloud_provider_readiness_$(date -u +%Y%m%dT%H%M%SZ).md \
  --timeout-s 8
```

Interpretation:

- `score_claim=false` and `ready_for_exact_eval_dispatch=false` mean the file
  is inventory only.
- `exact_cuda_evidence_allowed=false` means the provider is not yet a score
  truth surface, even if the CLI is installed.
- Kaggle remains proxy-only even when kernels are runnable; winning configs
  must move to a claimed CUDA provider before score use.
- Modal CLI readiness is not enough for dispatch. Billing/credits and the CUDA
  scorer import probe must both pass first.

Current setup blockers:

- Modal: `modal login`, billing/credits context, CUDA scorer import probe,
  active-call harvest before refire.
- Kaggle: API credentials, uploaded `tac-*.whl` dataset, GPU session quota.
- AWS: scaffold only; `boto3`, credentials, region DLAMI resolution, SSH
  key/security group, budget/quota, and lifecycle/harvest implementation.
- Azure: `az login`, quota/spot availability, SSH public key, lane tarball wiring.
- GCP: scaffold only; `gcloud` auth, billing, GPU quota, project/zone
  selection, GCS harvest bucket, and lifecycle/harvest implementation.

Landing integration note: `research_only=true` for this runbook/check landing.
It changes dispatch safety infrastructure only. Sensitivity-map, Pareto,
bit-allocator, autopilot, continual-learning, and probe-disambiguator hooks are
not binding until a scored provider artifact or lane-specific actuator consumes
the contract.
