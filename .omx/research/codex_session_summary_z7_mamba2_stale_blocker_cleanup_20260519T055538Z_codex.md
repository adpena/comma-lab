# Codex Session Summary - Z7-Mamba-2 Stale Blocker Cleanup

session_id: 019de465
actor: codex
date_utc: 2026-05-19T05:55:38Z
source_directive: .omx/research/codex_routing_directive_session_20260519_max_score_lowering_batch_BCEF_20260519T051028Z.md
task_id: codex_routing_directive_session_20260519_max_score_lowering_batch_BCEF_20260519T051028Z::CLUSTER_E1
commit: 739b11c9f
score_claim: false
promotion_eligible: false

## What Landed

Cleared the stale Z7-Mamba-2 readiness blockers from
`.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml`.
The recipe now points at the built L1 research-only lane
`lane_z7_as_mamba_2_full_landing_20260518`, removes the obsolete Catalog #240
full-main/module-absent blockers, and keeps dispatch fail-closed behind the six
remaining evidence gates.

The trainer and tests now describe the actual state: `_full_main` is built, but
non-promotable until Wave N+1 council, identity-disambiguator, Modal A100
`mamba_ssm` preflight, and paired exact-eval custody land.

## Bug-Class Hardening

While running the full readiness tests, two adjacent false-authority bugs surfaced
and were fixed in `tools/asymptotic_pursuit_candidate_readiness_assessment.py`:

- ATW predecessor-probe lookup no longer lets `atw_codec_v2_1_*` blocking probes
  bleed into the older `atw_codec_v2` candidate through prefix substring match.
- Recipes with `predicted_score_target: null` and no recipe score band no longer
  inherit loose design-memo delta-S text as horizon-class score-band authority.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:tools .venv/bin/python -m pytest -q src/tac/tests/test_asymptotic_pursuit_candidate_readiness.py src/tac/tests/test_z7_mamba2_scaffold.py src/tac/tests/test_check_240_substrate_contest_cuda_chain.py`
  - result: 132 passed, 1 expected local `mamba_ssm` fallback warning
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/operator_authorize.py --recipe substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch --dry-run`
  - result: refused dispatch exactly because `dispatch_enabled=false` and the six remaining blockers
- `git diff --check` scoped to touched files
  - result: clean

## Remaining Authority State

No score claim was made. No dispatch was fired. The recipe is still
`research_only: true` and `dispatch_enabled: false`. Reactivation remains
conditional on the remaining Wave N+1 / disambiguator / exact-eval gates.
