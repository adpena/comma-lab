---
title: "L5 Phase 1b pre-deploy claim verification"
date: 2026-05-17T02:31:24Z
author: codex
repo_commit: 6cae1cf6d090eb4f4f005bbc5b46b74e1a97c60d
evidence_grade: "pre-dispatch-engineering-proof"
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
axis: "none"
lanes:
  - lane_phase_1b_rudin_lift_20260516
  - lane_phase_1b_z6_lift_20260516
---

# L5 Phase 1b pre-deploy claim verification

## Scope

This ledger verifies the live pre-deploy claims recorded by the Phase 1b Rudin
and Z6 lift packets after `main` was pushed to `origin/main`.

It is **not** score evidence. It does not classify either lane as better,
worse, promoted, killed, or submission-ready. It only verifies that the
repository's canonical pre-deploy harness agrees that each trainer/recipe pair
is structurally dispatchable before any paid provider run.

## Commands run

```bash
.venv/bin/python tools/local_pre_deploy_check.py \
  --trainer experiments/train_substrate_rudin_floor_interpretable_ml.py \
  --recipe substrate_rudin_floor_interpretable_ml_modal_t4_dispatch \
  --strict
```

Verdict: `PASS`, all 9 checks.

Checks observed:

- `py_compile`
- `trainer_importable`
- `full_main_implemented`
- `archive_grammar`
- `auth_eval_reachability`
- `canonical_inflate_device`
- `deterministic_zip`
- `recipe_status_consistent_with_trainer_state`
- `dispatch_optimization_protocol`

The Rudin harness reported:

```text
[local-pre-deploy] ALL 9 CHECKS PASSED. Safe to dispatch.
```

```bash
.venv/bin/python tools/local_pre_deploy_check.py \
  --trainer experiments/train_substrate_time_traveler_l5_z6.py \
  --recipe substrate_time_traveler_l5_z6_modal_t4_dispatch \
  --strict
```

Verdict: `PASS`, all 9 checks.

Checks observed:

- `py_compile`
- `trainer_importable`
- `full_main_implemented`
- `archive_grammar`
- `auth_eval_reachability`
- `canonical_inflate_device`
- `deterministic_zip`
- `recipe_status_consistent_with_trainer_state`
- `dispatch_optimization_protocol`

The Z6 harness reported:

```text
[local-pre-deploy] ALL 9 CHECKS PASSED. Safe to dispatch.
```

## Interpretation

The two Phase 1b lift claims are live-verified against the canonical pre-deploy
harness at commit `6cae1cf6d090eb4f4f005bbc5b46b74e1a97c60d`.

This closes the immediate false-authority risk where the `.omx` landing memos
and recipes claimed `local_pre_deploy_check.py --strict` success without a
current-turn replay receipt.

## Remaining blockers

These receipts do not remove the normal dispatch and promotion constraints:

- A remote run still needs a lane dispatch claim before provider job creation.
- Any returned artifact still needs harvest custody, archive/runtime SHA-256s,
  component recomputation, axis label, and result-review classification.
- `[contest-CPU]` and `[contest-CUDA]` remain separate axes.
- Score promotion still requires byte-closed archive/runtime evidence, not this
  pre-dispatch structural proof.

## Next concrete action

For L5 staircase work, the highest-EV next action is to use these verified
pre-deploy receipts to either:

1. fire a claimed Rudin or Z6 timing smoke through the canonical operator
   authorization path, or
2. if spend is temporarily deferred, run an adversarial dry-run review of the
   exact operator-authorize command path and record the blocker that prevents
   spend.

