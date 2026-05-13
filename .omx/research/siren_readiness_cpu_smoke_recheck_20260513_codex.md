# SIREN readiness + CPU smoke recheck (2026-05-13)

generated_at_utc: `2026-05-13T11:33:00Z`  
agent: `codex`  
lane_id: `lane_substrate_siren_20260512`  
score_claim: `false`  
promotion_eligible: `false`  
ready_for_exact_eval_dispatch: `false`

## Summary

SIREN local substrate surfaces are present and the cheap CPU smoke path runs.
This is not a score claim and not a dispatch authorization. It is a local
integration check that reduces risk before any Modal/GPU first-anchor.

## Readiness Recheck

Command:

```bash
.venv/bin/python tools/audit_siren_substrate_readiness.py \
  --json \
  --output experiments/results/siren_readiness_recheck_20260513_codex.json
```

Result:

- `local_contract_ready=true`
- `ready_for_first_anchor_training=true`
- `ready_for_remote_dispatch=false`
- `ready_for_exact_eval_dispatch=false`
- `local_blockers=[]`
- `dispatch_blockers=["operator_authorization_required", "active_lane_dispatch_claim_required_before_gpu_spend", "no_gpu_spend_in_readiness_gate"]`
- manifest hash: `sha256:b057aec1ed781d0cd095d90ab56bc4aa20c56bb6bdbd7bfe1d70f087ecea7f8a`
- artifact SHA-256: `ffe96043c1b94016493160ca84c072e13481a5380dbcfeb41cfbd92fd728a809`

Key wired surfaces:

- trainer: `experiments/train_substrate_siren.py`
- recipe: `.omx/operator_authorize_recipes/substrate_siren_modal_a100_dispatch.yaml`
- archive grammar: `src/tac/substrates/siren/archive.py`
- inflate runtime: `src/tac/substrates/siren/inflate.py`
- score-aware loss: `src/tac/substrates/siren/score_aware_loss.py`

## CPU Smoke

Command:

```bash
.venv/bin/python experiments/train_substrate_siren.py \
  --video-path upstream/videos/0.mkv \
  --output-dir experiments/results/siren_smoke_20260513_codex \
  --epochs 3 \
  --device cpu \
  --smoke \
  --skip-archive-build \
  --skip-auth-eval
```

Output:

- params: `2438`
- step losses: `1.0080`, `1.0039`, `0.9999`
- checkpoint: `experiments/results/siren_smoke_20260513_codex/smoke_checkpoint.pt`
- checkpoint SHA-256: `914a51f474b80dd40f98cb6888a70edcd3351585be1784581dc61768f91671aa`

## Classification

This closes a local integration question only: SIREN is locally smokeable and
first-anchor-trainable, but still requires the canonical operator-authorize
path, an active lane claim, and a real remote training/eval lifecycle before
any exact score authority exists.

Next safe action is not fan-out. It is one canary through the canonical
operator-authorize path after provider/account state is confirmed:

```bash
scripts/operator_authorize_substrate_siren_modal_a100_dispatch.sh --dry-run
```

Then, only if the dry-run and lane claim are clean, run the real canary through
`tools/operator_authorize.py` so cost-band, claim, required-input validation,
Modal runtime closure, and harvest instructions stay centralized.

## Follow-up Actuator Bug Found And Fixed

The first wrapper dry-run found a real dispatch-actuator bug before any spend:

```text
scripts/operator_authorize_substrate_siren_modal_a100_dispatch.sh: line 32:
SMOKE_ARGS[@]: unbound variable
```

Root cause: macOS bash 3.2 raises under `set -u` when an empty declared array is
expanded as `"${SMOKE_ARGS[@]}"`. The fix rewrites all 13 substrate
smoke-before-full wrappers to use the guarded expansion
`${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"}` and adds `--dry-run` support to
`tools/run_modal_smoke_before_full.py`, so dispatch planning can be verified
without spawning Modal work.

Self-protection: Catalog #189
`check_shell_empty_arrays_guarded_under_set_u` is wired into `preflight_all()`
for substrate smoke-before-full wrappers, with focused regression tests and
live wrapper dry-run coverage.

Post-fix SIREN dry-run:

```bash
scripts/operator_authorize_substrate_siren_modal_a100_dispatch.sh --dry-run
```

Result:

- no Modal dispatch
- recipe resolved:
  `.omx/operator_authorize_recipes/substrate_siren_modal_a100_dispatch.yaml`
- epoch env var resolved: `SIREN_EPOCHS`
- planned smoke: `smoke_epochs=100`, `smoke_gpu=T4`, `timeout_hours=1.0`
- planned full: dispatch only after smoke-green
