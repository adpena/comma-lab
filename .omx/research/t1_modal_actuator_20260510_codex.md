# T1 Modal actuator landing ledger (2026-05-10)

## Summary

Worker C implemented a dedicated Modal provider actuator for
`t1_balle_128k_endtoend`:

- code: `experiments/modal_t1_balle_endtoend.py`
- tests: `src/tac/tests/test_modal_t1_balle_endtoend.py`
- lane id: `t1_balle_128k_endtoend`
- provider: Modal T4
- default mode: plan-only / dry-run
- real dispatch gate: `modal run experiments/modal_t1_balle_endtoend.py --execute ...`

The actuator converts the prior remote-script command plan into an actual
claimed Modal `.spawn()` path while preserving the existing T1 remote script as
the CUDA worker:

```text
local plan -> lane claim -> Modal .spawn() -> scripts/remote_lane_t1_balle_endtoend.sh
  -> packet compiler -> contest_auth_eval.py --device cuda -> auth_eval_schema adjudication
```

## Custody and score semantics

- Plan mode opens no claim, creates no Modal app, starts no GPU work, and writes
  `score_claim=false`.
- Non-dry-run dispatch requires `--execute`.
- Non-dry-run dispatch opens a lane claim with `tools/claim_lane_dispatch.py`
  before `.spawn()`.
- The local claim ledger is copied into the Modal worker so
  `scripts/remote_lane_t1_balle_endtoend.sh` can verify the active matching
  claim before score-domain training.
- Recovery is the only local score boundary. It accepts score evidence only
  when:
  - `contest_auth_eval.py --device cuda` ran through the remote script,
  - `n_samples=600`,
  - `tac.auth_eval_schema.required_contest_cuda_evidence_blockers(...)` returns
    zero blockers,
  - remote T1 adjudication reports `score_claim=true`.
- Training-only output remains `score_claim=false` and closes as a no-score
  Modal recovery, not as readiness.

## Six-hook wire-in declaration

1. Sensitivity-map contribution: N/A for this actuator. It does not introduce a
   new scoring functional; it launches the existing T1 score-domain trainer.
2. Pareto constraint: inherited from the Phase 1 packet compiler and remote T1
   script; no new Pareto axis.
3. Bit-allocator hook: inherited from T13 sqrt(n) and T19 adaptive-rho trainer
   flags already wired by the remote T1 script.
4. Cathedral autopilot dispatch hook: this actuator is the Modal dispatch hook
   for `t1_balle_128k_endtoend`.
5. Continual-learning posterior update: deferred to harvest/adjudication after
   a valid contest-CUDA empirical anchor; no posterior update from plan or
   training-only output.
6. Probe-disambiguator: N/A. The relevant surrogate decision is already fixed
   by the T8-alone probe verdict consumed by the remote T1 script.

`research_only=false`; `lane_class=provider_actuator`; `target_modes=["contest_exact_eval"]`.

## Non-regression guarantees

- No A1 Modal files were changed.
- Existing T1 remote script remains the authority for packet compile and exact
  CUDA auth eval.
- The actuator mounts and runs the existing remote script rather than creating a
  second training/eval implementation.
- Result-cache recovery is explicit; claims are closed terminally on recovered
  success/failure/expired/exception paths.

## Verification

Focused verification at landing:

```bash
.venv/bin/python -m py_compile experiments/modal_t1_balle_endtoend.py src/tac/tests/test_modal_t1_balle_endtoend.py
.venv/bin/python -m pytest src/tac/tests/test_modal_t1_balle_endtoend.py -q
.venv/bin/python experiments/modal_t1_balle_endtoend.py plan --label t1-modal-worker-c-plan --epochs 3 --batch-size 2 --timeout-hours 1 --json-out experiments/results/t1_modal_worker_c_plan_20260510.json
.venv/bin/python -m pytest src/tac/tests/test_modal_t1_balle_endtoend.py tests/test_dispatch_t1_balle_endtoend.py src/tac/tests/test_auth_eval_schema.py -q
bash -n scripts/remote_lane_t1_balle_endtoend.sh
.venv/bin/python tools/check_dispatch_cli_shell_hazards.py --strict
.venv/bin/python -m tac.preflight --scope dev --timeout-s 30 --timings-json experiments/results/preflight_dev_timing_after_t1_modal_actuator_20260510.json
```

Results:

```text
7 passed
32 passed
PREFLIGHT PASSED
```

## Codex follow-up hardening

After a fresh adversarial pass, local T1 recover was tightened to validate
against the remote adjudication packet fields instead of only the eval JSON's
self-reported archive size.

- `expected_archive_bytes` now comes from
  `auth_eval_adjudication.packet_archive_size_bytes` when present.
- Recover adds explicit blockers for adjudication-vs-eval archive SHA or size
  mismatches.
- Recover adds an explicit blocker if remote adjudication has nonempty
  blockers even when a malformed result also sets `score_claim=true`.

Verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_modal_t1_balle_endtoend.py src/tac/tests/test_auth_eval_schema.py -q
.venv/bin/python -m py_compile experiments/modal_t1_balle_endtoend.py
```

Result: `18 passed`.

## Red-team follow-up hardening

Fresh read-only review found that plan mode could understate T1 spend by using
the requested `--timeout-hours` while the Modal function timeout remained fixed
at 24h. This blocked real dispatch.

Fixes:

- T1 plan estimated cost is now computed from the actual Modal function timeout
  (`DEFAULT_TIMEOUT_HOURS=24.0`), not the user-requested display timeout.
- T1 plan rejects any `--timeout-hours` value that does not match the Modal
  function timeout.
- T1 plan rejects train timeouts that leave less than a 1h auth-eval/recovery
  buffer inside the Modal function timeout.
- Local T1 recover now requires adjudicated packet archive size and SHA fields
  before accepting contest-CUDA score evidence.

Verification:

```bash
.venv/bin/python -m pytest src/tac/tests/test_modal_phase_a1_score_gradient_pr101.py src/tac/tests/test_modal_t1_balle_endtoend.py tests/test_kaggle_check.py src/tac/tests/test_auth_eval_schema.py -q
```

Result: `40 passed`.
