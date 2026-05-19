# Codex Findings - Catalog #204 stack-of-stacks recovery pre-dispatch guard

**Task:** `codex_routing_directive_session_20260519_paid_dispatch_batch_C6_plus_204_followon_20260519T071500Z::ITEM_4`  
**Directive:** `.omx/research/codex_routing_directive_session_20260519_paid_dispatch_batch_C6_plus_204_followon_20260519T071500Z.md`  
**Scope:** harden the A1 passthrough recovery path before spending Modal T4 time.

## Finding

The recovery route was not dispatch-safe as written.

1. The stack-of-stacks driver had the Catalog #204 `/modal_results` fallback, but the operator-authorize recipe unconditionally exported `STACK_OF_STACKS_OUTPUT_DIR=/workspace/pact/...`. `experiments/modal_train_lane.py` maps `/workspace/pact/...` to `/tmp/pact/...`, so the recipe bypassed the durable branch and could repeat the same temp-evidence refusal.
2. The recipe exposed `STACK_OF_STACKS_LANGEVIN_T_INIT_CAP`, but the driver did not pass it to `experiments/train_substrate_stack_of_stacks.py`; provenance would keep the default `langevin_t_init=0.3` even when the recipe claimed cap `1.0`.
3. The Modal training wrapper defaults `MODAL_AUTH_EVAL_ADVISORY_ONLY=1`. That is safe for generic training lanes, but this recovery specifically calls `scripts/remote_archive_only_eval.sh`, which runs `contest_auth_eval.py --device cuda`; leaving the advisory flag set would downgrade an otherwise valid full CUDA auth eval to diagnostic-only.

## Fix

- Recipe now leaves `STACK_OF_STACKS_OUTPUT_DIR` unset by default via `${STACK_OF_STACKS_OUTPUT_DIR:-}` so Modal runtime uses `/modal_results/${INSTANCE_JOB_ID}/output`.
- Driver now passes `--langevin-t-init`, `--langevin-polish-epochs`, and `--lane-id` to the trainer.
- Recipe now explicitly sets `STACK_OF_STACKS_AUTH_EVAL_REQUIRE_CONTEST_CUDA=1`; driver unsets `MODAL_AUTH_EVAL_ADVISORY_ONLY` only under that opt-in before the archive-only CUDA eval.
- Added focused regression tests in `src/tac/tests/test_stack_of_stacks_catalog204_recovery_contract.py`.

## Verification

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest \
  src/tac/tests/test_stack_of_stacks_catalog204_recovery_contract.py \
  src/tac/tests/test_check_204_cross_driver_expansion.py -q -p no:cacheprovider
# 24 passed

bash -n scripts/remote_lane_substrate_stack_of_stacks.sh

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python tools/operator_authorize.py \
  --recipe substrate_stack_of_stacks_sgld_convergence_diagnostic_modal_t4_dispatch \
  --target modal --dry-run
# dry-run completed; no dispatch
```

## Authority

No score claim. This memo is a pre-dispatch guard. The actual ITEM_4 evidence still requires a fresh Modal run and recovered `contest_auth_eval.json` before the task can complete.
