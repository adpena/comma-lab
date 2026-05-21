---
schema: codex_findings_v1
topic: dp1_procedural_paired_harvest_planner
created_at_utc: 2026-05-21T02:45:14Z
author: codex
lane_id: lane_dp1_procedural_paired_harvest_plan_20260521
mission_contribution: frontier_breaking_enabler
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
dispatch_attempted: false
paired_dispatch_attempted: false
current_head_before_landing: b93c15afd5fa24ad888200721853b74e2a8f7275
---

# DP1 Procedural Paired-Harvest Planner Landed

## Summary

Landed a planning-only bridge from the three DP1 procedural-codebook training
recipes to paired CPU/CUDA exact-eval custody:

- `src/tac/optimization/dp1_procedural_paired_harvest_plan.py`
- `tools/plan_dp1_procedural_paired_harvest.py`
- `src/tac/tests/test_dp1_procedural_paired_harvest_plan.py`

This does **not** launch spend and does **not** claim a score. It refuses to
emit runnable paired-dispatch commands until each candidate arm has:

- `archive.zip` with a `0.bin` or `x` member;
- `submission/inflate.sh` executable and `submission/inflate.py`;
- `manifest.json`;
- `provenance.json`;
- no true `score_claim`, `score_claim_valid`, `promotion_eligible`,
  `rank_or_kill_eligible`, or `ready_for_exact_eval_dispatch` flags;
- no numeric score field in trainer/proxy manifests;
- recipe-level `DPP_SKIP_AUTH_EVAL: "1"` so trainer-inline one-axis score
  authority stays impossible.

For procedural and null-control arms it additionally requires
`procedural_variant_provenance.json` and checks the null-control marker.

## Adversarial Review

Read-only explorer `019e4866-cd4d-76e3-8ea8-bd3cad76a45e` independently
identified the same missing surface: a DP1-specific paired-eval handoff/harvest
planner, not another recipe. Its required dispatcher shape matches this
landing:

```bash
.venv/bin/python tools/dispatch_modal_paired_auth_eval.py \
  --archive <dp1_output_dir>/archive.zip \
  --expected-archive-sha256 <sha256> \
  --submission-dir <dp1_output_dir>/submission \
  --inflate-sh inflate.sh \
  --label <dp1_variant_label> \
  --run-id <dp1_variant_run_id> \
  --pair-group-id <shared_pair_group_id> \
  --lane-id-base <dp1_variant_lane_base> \
  --output-root experiments/results/dp1_procedural_paired_auth_eval \
  --modal-bin .venv/bin/modal \
  --gpu T4 \
  --claim-agent codex:dp1_procedural_paired_harvest \
  --claim-notes "<variant>; archive_sha=<sha>; score_claim=false" \
  --expected-runtime-tree-sha256 auto \
  --skip-axis-if-promotable-anchor-exists \
  --json-out <plan.json>
```

The execute form appends `--execute` only after plan review.

## Current-State Probe

Command:

```bash
.venv/bin/python tools/plan_dp1_procedural_paired_harvest.py \
  --include-null-control \
  --json-out experiments/results/dp1_procedural_paired_harvest/plan_current_state.json \
  --md-out experiments/results/dp1_procedural_paired_harvest/plan_current_state.md
```

Observed current status:

| arm | status | blockers |
|---|---|---|
| baseline | blocked | `candidate_output_dir_not_supplied` |
| procedural | blocked | `candidate_output_dir_not_supplied` |
| null_control | blocked | `candidate_output_dir_not_supplied` |

This is expected: commit `b93c15afd` authored the disabled training recipes and
runtime/trainer procedural support; no harvested DP1 recipe output directories
exist yet in the local tree for this planner to bind.

## Verification

Commands run:

```bash
.venv/bin/python -m py_compile \
  tools/plan_dp1_procedural_paired_harvest.py \
  src/tac/optimization/dp1_procedural_paired_harvest_plan.py \
  src/tac/tests/test_dp1_procedural_paired_harvest_plan.py

.venv/bin/python -m pytest -q \
  src/tac/tests/test_dp1_procedural_paired_harvest_plan.py

.venv/bin/python -m pytest -q \
  src/tac/tests/test_dp1_procedural_paired_harvest_plan.py \
  src/tac/tests/test_dispatch_modal_paired_auth_eval.py \
  src/tac/tests/test_modal_paired_dispatch_contract.py
```

Result: `6 passed` focused; `19 passed` with paired-dispatch regressions.

Lane registration:

- `lane_dp1_procedural_paired_harvest_plan_20260521`
- L1 with `impl_complete` and `strict_preflight` gates marked.

## Next Action

After the operator intentionally flips and runs the baseline + procedural DP1
recipes, run this planner with:

```bash
.venv/bin/python tools/plan_dp1_procedural_paired_harvest.py \
  --baseline-output-dir <harvested-baseline-output> \
  --procedural-output-dir <harvested-procedural-output> \
  --json-out <review-path>/dp1_paired_harvest_plan.json \
  --md-out <review-path>/dp1_paired_harvest_plan.md
```

If both required arms are ready, run the emitted `--execute` commands and then
recover each axis through the emitted harvest commands. The null-control arm
should remain optional and fire only if the procedural residual is ambiguous.
