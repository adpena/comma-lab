# Dual-device auth eval JSON-out custody hardening (2026-05-11)

## Scope

Operator goal: keep CPU/CUDA HNeRV analysis apples-to-apples and avoid another
false conclusion from partial artifacts. The prior PR103-on-PR106 CPU raw
manifest run proved that retained work dirs carry the needed
`inflated_outputs_manifest.json`, but the dual-device planner did not make the
durable per-axis JSON path explicit.

## Change

`tools/plan_dual_device_auth_eval.py` now includes:

- `evals.<axis>.json_out`
- `--json-out <work_dir>/contest_auth_eval.json` in every planned command
- `--keep-work-dir` remains present so raw-output manifests survive

This makes each planned axis self-contained:

```text
experiments/results/dual_device_auth_eval/<run_id>/<axis>/contest_auth_eval.json
experiments/results/dual_device_auth_eval/<run_id>/<axis>/inflated_outputs_manifest.json
```

The first file is the structured score/custody artifact. The second is the
mechanism artifact that separates render-device drift from scorer/loader drift.

## Classification

This is not a score claim and does not launch a job. It is a custody fix for
the next PR103-on-PR106 CUDA raw-output rerun after the active T1 Modal claim
clears.

## Verification

Focused test:

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_plan_dual_device_auth_eval.py
```

Expected invariant: planned CPU and CUDA commands both include `--json-out`
inside the same per-axis work directory and both retain `--keep-work-dir`.
