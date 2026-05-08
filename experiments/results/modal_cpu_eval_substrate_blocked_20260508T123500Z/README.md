# Modal CPU eval substrate — execution-blocked forensics (2026-05-08)

This directory captures the smoke-test attempts that surfaced the Modal
billing-cap blocker for the PR #107 [contest-CPU] auth eval.

- `modal_cpu_smoke_pr102_attempt1.log` — first attempt at 12:32 UTC. Modal
  began image build; aborted with `ExecutionError('/Users/adpena/Projects/pact/src/tac/preflight.py
  was modified during build process.')` from a concurrent agent's edit to
  preflight.py during Modal's `add_local_dir("src", ...)` upload.
- `modal_cpu_smoke_pr102_attempt2_billing_cap.log` — retry at 12:34 UTC
  after working tree stabilized. Modal accepted the function but refused
  app creation: `App creation failed: workspace billing cycle spend limit
  reached`.

The substrate code itself is committed at 856bda80
(`experiments/modal_auth_eval_cpu.py`) and is correct/ready. See
`feedback_pr107_cpu_eval_substrate_built_modal_blocked_20260508.md` /
`project_pr107_cpu_eval_substrate_built_modal_blocked_20260508.md` in
auto-memory for the full triage + unblock options.
