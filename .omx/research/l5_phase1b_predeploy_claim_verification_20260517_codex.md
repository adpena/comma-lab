---
title: "L5 Phase 1b pre-deploy verification and false-authority correction"
date: 2026-05-17T02:31:24Z
updated: 2026-05-17T03:18:00Z
author: codex
repo_base_commit: f2f81383f83853da84641ae3793ad62fd28c5a0b
evidence_grade: "pre-dispatch-engineering-proof"
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
axis: "none"
lanes:
  - lane_phase_1b_rudin_lift_20260516
  - lane_phase_1b_z6_lift_20260516
---

# L5 Phase 1b pre-deploy verification and false-authority correction

## Scope

This ledger corrects the original 2026-05-17 pre-deploy claim verification
for the Phase 1b Rudin and Z6 lift packets.

The original replay correctly showed Rudin as dispatchable, but it also exposed
a false-authority bug: `tools/local_pre_deploy_check.py` treated an implemented
`_full_main` as sufficient dispatch proof even when the recipe was still
`research_only: true`, `dispatch_enabled: false`, or carried explicit dispatch
blockers. That made Z6 look like a "9/9 safe to dispatch" packet while
`tools/operator_authorize.py` correctly refused the same recipe.

The corrected evidence is:

- Rudin: local pre-deploy strict passes and operator-authorize dry-run does not
  refuse. This is dispatch authority only, not score or promotion authority.
- Z6: local pre-deploy strict now fails closed at
  `recipe_status_consistent_with_trainer_state`, matching operator-authorize.
  Z6 remains implemented but intentionally non-dispatchable until the declared
  Phase 2 and smoke-before-full blockers clear.

This is **not** score evidence. It does not classify either lane as better,
worse, promoted, killed, or submission-ready.

## Harness fix

`tools/local_pre_deploy_check.py` now treats recipe dispatchability as a first
class pre-deploy condition. For an implemented trainer, all of the following
must be clear before the recipe can be used as dispatch proof:

- `research_only: true`
- `dispatch_enabled: false`
- non-empty `dispatch_blockers`
- non-empty `pre_promotion_blockers`

The focused regression coverage is in
`src/tac/tests/test_local_pre_deploy_check.py` and verifies all three relevant
states: implemented+disabled fails, implemented+dispatchable passes, and
NotImplemented+non-dispatchable remains a transparent research packet.

## Commands run

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_local_pre_deploy_check.py
```

Verdict: `PASS`, 17 tests.

```bash
.venv/bin/python tools/local_pre_deploy_check.py \
  --trainer experiments/train_substrate_rudin_floor_interpretable_ml.py \
  --recipe substrate_rudin_floor_interpretable_ml_modal_t4_dispatch \
  --strict
```

Verdict: `PASS`, all 9 checks. The key dispatchability check reports:

```text
recipe-vs-trainer-state consistent (trainer `_full_main` implemented; recipe is contest-CUDA dispatchable)
[local-pre-deploy] ALL 9 CHECKS PASSED. Safe to dispatch.
```

```bash
.venv/bin/python tools/local_pre_deploy_check.py \
  --trainer experiments/train_substrate_time_traveler_l5_z6.py \
  --recipe substrate_time_traveler_l5_z6_modal_t4_dispatch \
  --strict
```

Verdict: `FAIL-CLOSED`, 8 checks pass and 1 check fails. The failing check is
the intended correction:

```text
recipe_status_consistent_with_trainer_state FAIL: trainer `_full_main` is implemented but recipe is still non-dispatchable (research_only=true, dispatch_enabled=false, dispatch_blockers). operator_authorize.py would refuse before claim/provider setup; clear these recipe blockers or do not use local_pre_deploy_check as dispatch proof.
```

```bash
.venv/bin/python tools/operator_authorize.py \
  --recipe substrate_rudin_floor_interpretable_ml_modal_t4_dispatch \
  --dry-run --agent codex:l5-predeploy-audit
```

Verdict: `PASS-DRY-RUN`; no refusal and no dispatch.

```bash
.venv/bin/python tools/operator_authorize.py \
  --recipe substrate_time_traveler_l5_z6_modal_t4_dispatch \
  --dry-run --agent codex:l5-predeploy-audit
```

Verdict: `REFUSED-DRY-RUN`; no dispatch:

```text
dispatch_enabled=false; dispatch_blockers=phase_2_sextet_pact_council_consensus_required_per_design_memo_section_19_criterion_4, catalog_167_smoke_before_full_pattern_required_per_design_memo_section_19_criterion_5, paired_cpu_cuda_empirical_anchor_required_before_promotion_eligible_per_catalog_220
```

## Interpretation

The prior claim that Z6 was "9/9 safe to dispatch" is revoked. The correct
state is:

- Z6 `_full_main` implementation: live.
- Z6 Catalog #270 dispatch optimization protocol: still green.
- Z6 recipe dispatchability: false.
- Z6 local pre-deploy strict: intentionally fails until recipe blockers clear.
- Z6 operator-authorize: correctly refuses before claim/provider setup.

This closes a repeated local-minimum failure class: implementation readiness
must not be collapsed into dispatch readiness. The recipe remains the
operator-facing dispatch authority, and local pre-deploy must match it.

## Remaining blockers

These receipts do not remove the normal dispatch and promotion constraints:

- A remote run still needs a lane dispatch claim before provider job creation.
- Any returned artifact still needs harvest custody, archive/runtime SHA-256s,
  component recomputation, axis label, and result-review classification.
- `[contest-CPU]` and `[contest-CUDA]` remain separate axes.
- Score promotion still requires byte-closed archive/runtime evidence, not this
  pre-dispatch structural proof.
- Z6 specifically needs its Phase 2 council and smoke-before-full blockers
  cleared before any provider dispatch.

## Next concrete action

For the L5 staircase:

1. Rudin can proceed to a claimed timing smoke if operator budget and lane
   claims are clear.
2. Z6 should not dispatch until the recipe blockers are intentionally cleared,
   then `local_pre_deploy_check.py --strict` and `operator_authorize.py
   --dry-run` must both be replayed.
