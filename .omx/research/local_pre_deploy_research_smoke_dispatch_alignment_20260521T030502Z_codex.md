# Local Pre-Deploy Research-Smoke Dispatch Alignment

## Finding

The first real DP1 baseline launch stopped before any provider spend because `tools/local_pre_deploy_check.py` treated `research_only: true` as a dispatch refusal even when `dispatch_enabled: true` and no blockers were present.

This was stricter than `tools/operator_authorize.py`: the actuator refuses `dispatch_enabled: false`, `dispatch_blockers`, `pre_promotion_blockers`, and direct full dispatch of `smoke_only: true`, but it does not refuse `research_only: true` by itself. In this repo, `research_only: true` is false-authority metadata for non-promotional research artifacts, not necessarily a provider-dispatch blocker.

## Fix

- Aligned `check_recipe_status_consistent_with_trainer_state()` with the actuator refusal surface.
- Kept protection for stub trainers: `research_only: true` alone no longer counts as a real non-dispatchable flag when `_full_main` raises `NotImplementedError`; such recipes still need `dispatch_enabled: false` or explicit blockers.
- Added regression tests for dispatchable research-smoke recipes and for the `research_only: true`-alone stub-trainer hazard.

## Verification

Commands run:

```bash
.venv/bin/python -m pytest src/tac/tests/test_local_pre_deploy_check.py src/tac/tests/test_dp1_procedural_paired_harvest_plan.py src/tac/tests/test_dispatch_modal_paired_auth_eval.py src/tac/tests/test_modal_paired_dispatch_contract.py
.venv/bin/python -m py_compile tools/local_pre_deploy_check.py src/tac/tests/test_local_pre_deploy_check.py
.venv/bin/python tools/local_pre_deploy_check.py --trainer experiments/train_substrate_pretrained_driving_prior.py --recipe substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch --strict
.venv/bin/python tools/local_pre_deploy_check.py --trainer experiments/train_substrate_pretrained_driving_prior.py --recipe substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch --strict
.venv/bin/python tools/operator_authorize.py --recipe substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch --dry-run
.venv/bin/python tools/operator_authorize.py --recipe substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch --dry-run
.venv/bin/python tools/lane_maturity.py validate
git diff --check
```

Results:

- `39 passed`.
- `py_compile` passed.
- Baseline and procedural local pre-deploy: all 9 checks passed.
- Baseline and procedural operator dry-runs: no dispatch refusal.
- `1106 lane(s) validated cleanly`.
- `git diff --check` clean.
