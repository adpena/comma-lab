# PR84/PR82 Lightning Empty-Artifact Hardening - 2026-05-03

## Infrastructure Guard

- Context: PR84+PR82 packed exact-eval jobs were running on Lightning T4 while prior expanded-wave jobs produced empty artifact directories. Those empty dirs are provider/infrastructure evidence, not PR84/PR82 method evidence, because no `contest_auth_eval.json`, `archive.zip`, adjudication provenance, or exact archive-byte custody exists.
- Change: `src/tac/deploy/lightning/batch_jobs.py` now probes the resolved SSH artifact directory before exact-eval harvest. If the remote directory exists but contains no top-level files, harvest records `ARTIFACT_INFRA_FAILURE` with `terminal_class=empty_lightning_artifact_dir_infra`, expected archive identity, remote dir, mirror dir, and `score_claim=false` / `method_evidence=false` / `promotion_eligible=false`.
- Guardrail: empty artifact dirs are classified before `scp`, so they cannot be converted into `HARVESTED` state or adjudicated as score/method evidence.

## Verification

- `.venv/bin/python -m py_compile src/tac/deploy/lightning/batch_jobs.py scripts/launch_lightning_batch_job.py src/tac/tests/test_lightning_batch_jobs.py`
- `.venv/bin/python -m pytest src/tac/tests/test_lightning_batch_jobs.py -q` -> `108 passed`

## Reroute Packet

If any current PR84/PR82 Lightning T4 job resolves to `terminal_class=empty_lightning_artifact_dir_infra`, treat it as provider/infrastructure failure. Preserve the state record and `lightning_artifact_infra_failure.json`, then have the orchestrator claim a fresh lane/job before retrying on a different Lightning route or provider. Do not rank, retire, or kill PR84/PR82 from the empty-dir result.
