# PR85 STBM1BR + PR92 RMB1 Randmulti Recode - 2026-05-04

- tool: `experiments/build_pr85_stbm1br_pr92_rmb1_randmulti_candidate.py`
- score_claim: false
- dispatch_performed: false
- remote_jobs_dispatched: false

## Candidate

- archive: `experiments/results/pr85_stbm1br_pr92_rmb1_randmulti_20260504_worker/pr85_stbm1br_plus_pr92_rmb1_randmulti_recode/archive.zip`
- bytes: `229480`
- sha256: `f8d2dff12004fe15bdedefcd3f9574fab97f22c302fa1417a265c325468ad774`
- manifest: `experiments/results/pr85_stbm1br_pr92_rmb1_randmulti_20260504_worker/pr85_stbm1br_plus_pr92_rmb1_randmulti_recode/manifest.json`
- archive_delta_bytes_vs_stbm: `-276`
- randmulti decoded rows SHA: `87bcc720c1e80afb9adad5ee01477423ced526f31c54d461d69dbf26e08eecc9`

## Readiness

- strict_zip_valid: `True`
- exact_t4_dispatch_justified_after_claim: `True`
- exact CUDA eval is required before any score claim.

## Exact Next Claim Command

```bash
.venv/bin/python tools/claim_lane_dispatch.py claim --lane-id pr85_stbm1br_pr92_rmb1_randmulti --platform lightning --instance-job-id exact_eval_pr85_stbm1br_pr92_rmb1_t4_20260504T0829Z --agent codex:gpt-5.5 --predicted-eta-utc 2026-05-04T10:29Z --status eval --notes "T4 exact eval for pr85_stbm1br_plus_pr92_rmb1_randmulti_recode; archive_sha256=f8d2dff12004fe15bdedefcd3f9574fab97f22c302fa1417a265c325468ad774"
```

## 2026-05-04 Claim-Aware Preflight Hardening

- local readiness preflight:
  `experiments/results/pr85_stbm1br_pr92_rmb1_randmulti_20260504_worker/pr85_stbm1br_plus_pr92_rmb1_randmulti_recode/dispatch_readiness_preflight_local.json`
- claim-aware preflight:
  `experiments/results/pr85_stbm1br_pr92_rmb1_randmulti_20260504_worker/pr85_stbm1br_plus_pr92_rmb1_randmulti_recode/dispatch_readiness_preflight_active_claim.json`
- local readiness result: `ready_for_exact_eval_dispatch=true` with no blockers
- claim-aware result: `ready_for_exact_eval_dispatch=false` because an active
  same-lane claim already exists:
  `exact_eval_pr85_stbm1br_pr92_rmb1_t4_20260504T082220Z`
- active claim status observed by preflight: `eval`
- active claim predicted ETA: `2026-05-04T09:37:20Z`
- dispatch decision from this turn: no duplicate dispatch; wait for the active
  T4 eval or terminalize it before any relaunch.
- tool hardening: `experiments/preflight_candidate_manifest_dispatch_readiness.py`
  now accepts `--claims-path` and fail-closes on active same-lane claims.
- manifest hardening: candidate manifest now records
  `exact_eval_runtime_contract.ready_for_exact_eval_runtime=true` with the
  replay runtime tree SHA.
