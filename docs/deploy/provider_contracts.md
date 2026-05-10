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
- Modal/AWS/Azure/GCP can host CUDA exact-eval only after lane claim, runtime closure, and adjudication over byte-closed artifacts.

Current setup blockers:

- Modal: `modal login`, billing context, CUDA image import probe.
- Kaggle: API credentials, uploaded `tac-*.whl` dataset, GPU session quota.
- AWS: `boto3`, credentials, region DLAMI resolution, SSH key/security group.
- Azure: `az login`, quota/spot availability, SSH public key, lane tarball wiring.
- GCP: `gcloud` auth, GPU quota, project/zone selection, GCS harvest bucket.

Landing integration note: `research_only=true` for this runbook/check landing.
It changes dispatch safety infrastructure only. Sensitivity-map, Pareto,
bit-allocator, autopilot, continual-learning, and probe-disambiguator hooks are
not binding until a scored provider artifact or lane-specific actuator consumes
the contract.
