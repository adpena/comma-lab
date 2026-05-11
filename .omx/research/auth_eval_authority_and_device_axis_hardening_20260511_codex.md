# Auth-eval authority and device-axis hardening (2026-05-11)

score_claim: `false`
promotion_eligible: `false`
rank_or_kill_eligible: `false`

## Scope

This ledger records a custody/authority hardening pass triggered by recursive
adversarial review of the PR103-on-PR106 CPU/CUDA device-axis work.

## Fixed bug classes

1. `tools/auth_eval_records.py` no longer infers `promotion_eligible`,
   `score_claim_valid`, or `rank_or_kill_eligible` from full-sample T4 CUDA
   shape alone. Raw `[contest-CUDA]` evidence may be a score axis, but rank/kill
   authority requires explicit boolean fields.
2. Top-level forged hardware fields no longer manufacture contest-CUDA when a
   provenance object is present without `gpu_t4_match`.
3. `experiments/modal_auth_eval.py` now rejects the ambiguous remote-only state
   `scorer_device=cpu` with `inflate_device_policy=auto`; GPU-host CPU-scorer
   diagnostics must specify `cpu` or `cuda` inflate policy, and pure CPU replay
   belongs in `modal_auth_eval_cpu.py`.
4. `tools/plan_dual_device_auth_eval.py` keeps paired CPU/CUDA completion
   separate from adjudicated rank/kill authority.
5. `tools/lane_maturity.py` adds `contest_cpu` as a first-class L3 gate. Older
   local registries are normalized fail-closed with `contest_cpu=false`.

## Evidence

- Focused tests: `171 passed` for auth-eval parser, score dashboard, Modal
  auth-eval, dual-device planner, lane maturity, representation-lane registry,
  and device-axis matrix tests.
- Lane registry validation: `OK — 283 lane(s) validated cleanly`; current local
  audit shows zero L3 lanes under the 8-gate standard until contest-CPU evidence
  is recorded.

## Device-axis rule

Do not infer CPU or CUDA is globally better from one archive/runtime. Each
submission must preserve separate `[contest-CUDA]`, `[contest-CPU]`, diagnostic
CUDA, diagnostic CPU, macOS CPU advisory, and MPS/proxy labels. MPS remains
useful for sweeps and curve-finding only, never for auth-eval authority.
