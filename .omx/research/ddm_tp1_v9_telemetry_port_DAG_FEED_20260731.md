---
schema: dag_feed.v1
feed_id: FEED-tp1-v9-telemetry-port
date_utc: 2026-07-31
arm: ddm_tp1
task: 804
pointer: "0.1910828242 [contest-CPU] UNMOVED"
score_claim: false
tokens: [no-triality, p0-ledger-ok]
commit: 15aad5a28b
---

# FEED-tp1 — v9-line telemetry PORT to TR1: DONE (burn-4 §3.1 prereq 1 CLEARED)

- **The owed port landed (vh1 row 7 / burn-4 §3 prereq 1 — "the single hardest prerequisite").**
  `--telemetry-v9-port {off,on}` (DEFAULT off) on the TR1 trainer emits, when on: per-term
  `loss_terms` (#304, keys {seg,rate,delta_sparsity} = the exact `batch_loss` addends,
  sum_terms/sum_minus_total self-check) · `term_domination` + `term_inert` alarms (#321) ·
  accepted-frac + weights_stepped liveness (#402) · a #404 positive-control sentinel · Q7
  `lever_engage` companions. **Burn-4's F1–F4 halt rules now have their SIGNALS.**

- **PORT not reinvention.** Reused the shipped v9 producers: `term_inert_rows`, `lever_engage_row`,
  `deterministic_strata`, `ProducerResumeState` (`tac.witness_control.telemetry_producers`) +
  `canary_suite` (`tac.witness_control.verdict_trend_alarm`). Only the two TR1-specific pure builders
  (`tr1_loss_terms_row` #304, `tr1_term_domination_alarms` #321) are new (TR1's loss addends differ
  from the witness LOSS_TERM_KEYS).

- **BYTE-IDENTITY MEASURED (the sealed-r1c-lineage guarantee).** CPU off-vs-ON: 112 param/ema/opt
  arrays across 4 checkpoints BIT-IDENTICAL; only the pre-existing wall-clock `gate_wall_seconds`
  differs. CPU off-vs-off control shows the IDENTICAL difference profile ⇒ the flag is fully inert
  (the only difference is flag-independent wall-clock nondeterminism, not my code). GPU off-vs-off
  control: param max|Δ|≈7e-6 = MLX GPU float nondeterminism (flag-independent). ON emits the new
  rows; OFF emits zero. Flag via `args` not cfg (config_hash flag-invariant); new rows via `tlog`
  only, never `telemetry_tail` (checkpoint-baked) ⇒ checkpoints flag-invariant.

- **DSL Lever.** `lever_telemetry_v9_port(state)` in `spec_tr1_renderer_20260728.py`
  (never-invent-flags validated). Observability-as-Lever: default-off for byte-identity, a
  controller/ticket turns it on on-demand (default-off-is-orphan reconciliation SECOND case, recorded).

- **Tests / gates.** 15 tests (schema · off-identity · alarm-fire · DSL validate) pass; ruff F clean;
  #402 liveness gate 0 tr1 violations. 2 review passes/​.py recorded (no REVIEW_GATE_OVERRIDE on .py).

- **Dispositions (recorded, not enum-padded).** Q3 verdict-live-gap NOT-ported (2nd-scorer-pass cost;
  DSL-default-off in v9; outside the charter's named-4) · Q1 grad_clip N/A (TR1 does no clipping;
  gnorm already emitted) · Q4 tail_cycle / Q5 would-fire N/A (no tail cycles / different levers) · Q6
  `birth_completion` is P2-owned (vh1 row 14) and rides this surface.

- **Artifacts.** commit 15aad5a28b · memo `.omx/research/ddm_tp1_v9_telemetry_port_20260731.md` ·
  burn-4 skeleton §3 prereq-1 slot FILLED (`ddm_burn4_charter_skeleton_20260731.md`).

- **Pointer 0.1910828242 [contest-CPU] UNMOVED.** APPARATUS (score-neutral); nothing here moves S.
  Pre-existing UNRELATED test skew noted in the memo (`test_counted_ledger_keys_*` fails on HEAD too;
  a sister ledger-key landing, not this arm).
