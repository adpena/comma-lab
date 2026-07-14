# Round-1 review — LSI tau anneal / task-#500 synthesis

**Reviewer:** Codex self-adversarial round 1  
**Scope:** the two new research-only executable surfaces and their focused tests; no specialist
RIPO, basis, Bregman, DSL, canonical-equation, trainer, or evaluator ownership.  
**Verdict:** `CLEAN_AFTER_FIXES` for local research-only use; live V9 activation remains refused.

## Re-derived invariants

1. The LSI law changes the schedule constraint, not an optimizer update:
   `e_dot <= -rho e + beta_tau |tau_dot|` and inward flow at `e=r` give
   `|tau_dot| <= rho*r/beta_tau`. With `rho=kappa*lambda_gap`, the coefficient is
   `c=kappa*r/beta_tau`; no universal `1/2` is embedded.
2. The #318 object remains exactly a static CFL safe set. It cannot be compared numerically with
   the LSI speed cap until a separately derived `tau_per_epoch` receipt exists.
3. The #500 object is one metric: a basis is a pullback `Psi_B^T G_dec Psi_B`; the full-K KL
   trust region constrains that same reachable decision geometry; tau is a product coordinate.

## Findings fixed in round 1

1. **HIGH — null telemetry identity could compare equal.** The first implementation compared
   dictionaries with `.get`, so two absent/null `state_sha256`, `epoch`, or `tau` values could pass
   equality. Fixed with explicit completeness admission before equality; regression added.
2. **HIGH — a DE rate mapping could omit the rate.** A mapping with units/derivation but no
   `max_abs_tau_rate` did not emit a blocker. Fixed fail-closed; regression added.
3. **MEDIUM — basis metric identity was nullable.** A basis receipt with no `metric_id` was admitted.
   Fixed to require exact `argmax_native_vjp_fidelity_v1`; regression added.
4. **MEDIUM — Fisher/margin evidence was an unowned literal.** The synthesis embedded `0.978`.
   Fixed to consume both the value and source SHA-256 from the canonical-metric component; missing
   custody now blocks activation.

## Verification

- Focused tests: `15 passed`.
- Ruff: clean after import ordering correction.
- Receipt regeneration check: executable cross-check and synthesis outputs match the persisted
  machine receipts, with their deliberately added provenance envelopes.
- Verdict scope: current LSI/DE result remains `NO_VERDICT_SOURCE_CUSTODY`; no formulation family
  is closed and no launch or pointer movement is authorized.

