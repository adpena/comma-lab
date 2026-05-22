# Codex Findings: NSCS06 Full-Mode Device Guard

utc: 2026-05-22T10:59:04Z
lane: lane_codex_nscs06_full_mode_device_guard_20260522
status: LANDED
evidence_grade: pre_dispatch_guard
score_claim: false
score_claim_valid: false
promotion_eligible: false
rank_or_kill_eligible: false
ready_for_exact_eval_dispatch: false
promotable: false

## Finding

The prior NSCS06 v8 rc=22 mode-refusal guard was already covered, but the
follow-on rc=1 class still needed a structural guard: a recipe can request
full mode while the driver resolves the trainer device to `cpu`, reaching
`device_or_die` only after provider setup.

This landing adds that sister bug class to two pre-dispatch surfaces:

- `tools/audit_substrate_driver_mode_hardcode.py` now classifies
  `FULL_MODE_DEVICE_CPU_BUG_CLASS` when a full-mode recipe resolves the
  driver `--device` env var to `cpu`.
- `src/tac/deploy/dispatch_protocol.py` now blocks the same class in Tier 2
  hardware correctness, which `tools/operator_authorize.py` already invokes
  before native dispatch and lane-claim creation.

The live NSCS06 v8 recipe remains healthy: it declares
`NSCS06_V8_TRAINER_MODE=full` and `NSCS06_V8_DEVICE=cuda`, and the current
dispatch-protocol check passes.

During recursive live-audit replay, the new guard found one same-class
offender outside NSCS06: `substrate_rudin_floor_interpretable_ml_modal_t4_dispatch`
was dispatch-enabled/full-mode while resolving `RUDIN_FLOOR_DEVICE=cpu`.
That was fixed in the same landing by making the Rudin Modal driver default
and recipe override resolve to `cuda`; explicit CPU advisory/smoke runs still
require the existing trainer-side waiver path.

## Verification

- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m ruff check ...` passed for
  the changed guard and test files.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest
  src/tac/tests/test_check_326_substrate_driver_consumes_trainer_mode_env_var.py
  src/tac/tests/test_dispatch_protocol_tool_scope.py` passed:
  `63 passed in 3.08s`.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/audit_substrate_driver_mode_hardcode.py --format json`
  initially found `FULL_MODE_DEVICE_CPU_BUG_CLASS` in Rudin floor; after the
  Rudin recipe/driver fix it reported `bug_class_count=0` on the live repo.
- `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/check_dispatch_protocol_complete.py
  --recipe .omx/operator_authorize_recipes/substrate_nscs06_v8_chroma_lut_modal_t4_dispatch.yaml`
  reported `dispatch_protocol_complete=true`.

## Next Actions

1. Keep the current NSCS06 recipe/device pair as the canonical full-mode
   launch surface.
2. Use the same dispatch-protocol blocker for future full-mode substrate
   recipes; CPU full-mode must either become smoke/research-only or resolve
   to `cuda` before paid provider spend.
3. Proceed to MLX auth tensor-cache ingestion so local MLX can consume the
   already-passing Modal/Linux auth-side tensor cache without re-inflating
   locally.
