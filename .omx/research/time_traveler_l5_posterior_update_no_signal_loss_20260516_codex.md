# Time-Traveler L5 Posterior Update No-Signal-Loss Fix - 2026-05-16

## Summary

Adversarial review found that the TT5L full trainer attempted to update the
continual-learning posterior after auth eval, but the helper call did not pass
the required `architecture_class`. The exception path was warning-only, so an
exact auth-eval anchor could be produced while the posterior update silently
failed to preserve the signal.

## Fix

- Added stable `POSTERIOR_ARCHITECTURE_CLASS =
  "time_traveler_l5_autonomy"`.
- Passed that class into
  `posterior_update_locked_from_auth_eval_json(...)`.
- Recorded structured `posterior_update_status` in the full-run provenance for
  accepted, refused, error, and not-run cases.
- Stage logs now use explicit accepted/refused/error markers.
- All posterior update statuses carry `score_claim_promoted=false`.

## Verification

- `src/tac/tests/test_train_time_traveler_full_cpu_mode.py` covers accepted,
  refused, and TypeError/error update outcomes and verifies architecture-class
  forwarding.

## Reactivation Criteria

Reopen if a TT5L exact-eval path can complete without recording whether the
continual-learning posterior accepted, refused, skipped, or errored on the
anchor.
