# DP1 Paired Smoke Operator Activation

## Verdict

Operator continuation directive accepted for the required first-anchor pair:

- baseline recipe: `substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch`
- procedural recipe: `substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch`

The optional null-control recipe remains gated until the procedural-vs-baseline residual is ambiguous.

## Changes

- Flipped `dispatch_enabled: true` only for the baseline and procedural DP1 paired-smoke recipes.
- Added a `dispatch_activation` provenance block with timestamp `20260521T025702Z`.
- Kept `score_claim: false`, `promotion_eligible: false`, `rank_or_kill_eligible: false`, and `ready_for_exact_eval_dispatch: false`.
- Fixed `tac.optimization.dp1_procedural_paired_harvest_plan` so post-activation recipes do not block later paired-harvest planning merely because they are dispatch-enabled.

## Verification

Commands run:

```bash
.venv/bin/python -m pytest src/tac/tests/test_dp1_procedural_paired_harvest_plan.py src/tac/tests/test_dispatch_modal_paired_auth_eval.py src/tac/tests/test_modal_paired_dispatch_contract.py
.venv/bin/python -m py_compile src/tac/optimization/dp1_procedural_paired_harvest_plan.py tools/plan_dp1_procedural_paired_harvest.py tools/operator_authorize.py
.venv/bin/python tools/lane_maturity.py validate
git diff --check
.venv/bin/python tools/operator_authorize.py --recipe substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch --dry-run
.venv/bin/python tools/operator_authorize.py --recipe substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch --dry-run
.venv/bin/python tools/operator_authorize.py --recipe substrate_pretrained_driving_prior_null_exploit_codebook_modal_t4_paired_dispatch --dry-run
.venv/bin/python tools/plan_dp1_procedural_paired_harvest.py --json-out /tmp/dp1_plan.json --md-out /tmp/dp1_plan.md
```

Results:

- `20 passed`.
- `py_compile` passed.
- `1106 lane(s) validated cleanly`.
- `git diff --check` clean.
- Baseline dry-run: no dispatch refusal; no dispatch executed.
- Procedural dry-run: no dispatch refusal; no dispatch executed.
- Null-control dry-run: still refused with `dispatch_enabled=false`, as intended.
- Current harvest plan reports baseline/procedural `recipe_dispatch_enabled: true` and blocks only on missing candidate output dirs, which are expected before Modal training/export runs complete.

## Next Action

Commit this activation so the repo is clean, then launch the baseline and procedural Modal T4 paired-smoke training recipes via `tools/operator_authorize.py --yes`. After outputs are harvested, run `tools/plan_dp1_procedural_paired_harvest.py` with both output dirs and dispatch paired CPU/CUDA auth eval through `tools/dispatch_modal_paired_auth_eval.py`.
