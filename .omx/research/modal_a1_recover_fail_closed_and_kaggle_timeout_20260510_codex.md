# Modal A1 recover fail-closed + Kaggle timeout hardening (2026-05-10)

Generated: 2026-05-10T08:03:21Z

## Why this exists

A read-only red-team pass found three dispatch-facing risks while score-lowering
work was in flight:

1. The active A1 Modal run is still live and must be harvested, not duplicated.
2. `experiments/modal_phase_a1_score_gradient_pr101.py recover` printed
   `[contest-CUDA]` whenever any `eval_data` existed, even if strict evidence
   blockers were present.
3. `scripts/kaggle_check.py` had a bounded log fetch but an unbounded status
   subprocess, making it unsuitable for hot-loop DX or preflight-adjacent use.

## Changes landed

- `scripts/kaggle_check.py` now applies a per-kernel status timeout via
  `--status-timeout-s` and returns a visible `STATUS_TIMEOUT_AFTER_<N>S`
  marker instead of hanging.
- `experiments/modal_phase_a1_score_gradient_pr101.py recover` now routes
  harvested eval JSON through `tac.auth_eval_schema` and only prints
  `[contest-CUDA]` when strict contest-CUDA blockers are empty.
- Modal A1 harvest summaries now preserve `score_claim`,
  `score_claim_valid`, `exact_cuda_eval_complete`, `promotion_eligible`,
  `rank_or_kill_eligible`, `auth_eval_blockers`, `claim_blockers`,
  `dispatch_blockers`, `canonical_score_source`, `eval_archive_size_bytes`,
  `n_samples`, `score_axis`, `evidence_semantics`, and `lane_tag`.
- Modal A1 dispatch stdout now labels the predicted band as a planning forecast,
  not as evidence.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_modal_phase_a1_score_gradient_pr101.py tests/test_kaggle_check.py tests/test_build_kaggle_proxy_sweep_kernel.py src/tac/tests/test_modal_t1_balle_endtoend.py src/tac/tests/test_auth_eval_schema.py -q`
  - Result: 40 passed.
- `.venv/bin/python -m py_compile experiments/modal_phase_a1_score_gradient_pr101.py experiments/modal_t1_balle_endtoend.py scripts/kaggle_check.py tools/build_kaggle_proxy_sweep_kernel.py tools/cloud_provider_readiness.py`
  - Result: passed.
- `git diff --check -- experiments/modal_phase_a1_score_gradient_pr101.py scripts/kaggle_check.py tests/test_kaggle_check.py src/tac/tests/test_modal_phase_a1_score_gradient_pr101.py experiments/modal_t1_balle_endtoend.py tools/build_kaggle_proxy_sweep_kernel.py tests/test_build_kaggle_proxy_sweep_kernel.py tools/cloud_provider_readiness.py tools/operator_briefing.py`
  - Result: passed.
- `.venv/bin/python tools/check_dispatch_cli_shell_hazards.py --strict`
  - Result: passed.

## Dispatch state

- Active claim remains:
  `track1_phase_a1_score_gradient_modal_20260510T0738Z_codex` on Modal,
  call id `fc-01KR8D302GXGKGT49ETYMA0BZC`, predicted ETA
  `2026-05-10T11:37:03Z`.
- Recover poll at 2026-05-10T08:03Z returned `NOT READY`; no duplicate A1
  dispatch is permitted while this claim remains active.

## Evidence boundary

Kaggle and MPS remain proxy/config-search substrates only. Modal, GCP, AWS, or
Azure can carry exact CUDA evidence only after an active lane claim, byte-closed
archive/runtime custody, strict auth-eval schema, and terminal claim closure.

## Red-team follow-up fixes

Fresh read-only review found that future A1 dispatches could still under-budget
the exact CUDA eval window and that non-claimable `rc=0` harvests could look
successful to automation watching only exit status.

Fixes:

- `DEFAULT_TIMEOUT_HOURS` for future A1 Modal dispatches is now 6h instead of
  4h.
- A1 plan and dispatch paths now validate the user-visible `--timeout-hours`
  against the Modal function timeout and reject stage timeout budgets that do
  not leave a 10-minute safety margin.
- A1 recover now closes `rc=0` but non-claimable harvests as
  `failed_modal_recovered_no_score_claim` and returns nonzero unless strict
  `score_claim_valid=true`.
- Kaggle status timeouts are now non-clean (`is_error_status(...) == true`) so
  readiness tooling cannot treat an unknown kernel status as OK.

Verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_modal_phase_a1_score_gradient_pr101.py src/tac/tests/test_modal_t1_balle_endtoend.py tests/test_kaggle_check.py src/tac/tests/test_auth_eval_schema.py -q
.venv/bin/python -m py_compile experiments/modal_phase_a1_score_gradient_pr101.py experiments/modal_t1_balle_endtoend.py scripts/kaggle_check.py
```

Result: `40 passed`.
