---
review_kind: adversarial_guard_repair
review_id: catalog326_driver_mode_false_safety_repair_20260518_codex
review_date: "2026-05-18"
catalog_id: 326
evidence_axis: pre_dispatch_guard
score_claim: false
promotion_eligible: false
provider_spend: false
---

# Catalog #326 Driver-Mode False-Safety Repair

## Summary

An adversarial pass over the new Catalog #326 substrate driver
smoke/full-mode audit found two false-safety cases before the gate becomes
strict:

1. A hardcoded-`--smoke` driver was accepted when *any* matching recipe opted
   out of full-mode, even if another matching recipe was dispatchable. That
   could hide a real full-mode recipe behind a smoke-only sibling.
2. Recipe mode-env detection scanned the whole YAML file, so a note/comment
   mentioning `SMOKE_ONLY: "0"` could be mistaken for an actual
   `env_overrides` dispatch setting.

Both are pre-dispatch custody bugs. They do not change archive bytes and do
not make a score claim.

## Repair

- `tools/audit_substrate_driver_mode_hardcode.py` now requires *all* matching
  recipes to opt out before classifying a hardcoded-`--smoke` driver as safe.
- `_recipe_sets_env_var()` now reads only the YAML `env_overrides` block. It
  ignores comments, notes, and other top-level fields.
- `src/tac/tests/test_check_326_substrate_driver_consumes_trainer_mode_env_var.py`
  adds regressions for mixed opt-out/dispatchable siblings and note-only env
  mentions, plus a positive control for real `env_overrides`.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_check_326_substrate_driver_consumes_trainer_mode_env_var.py -q`
  - `33 passed`
- `.venv/bin/python tools/audit_substrate_driver_mode_hardcode.py --format summary`
  - `Drivers scanned: 47`
  - `Bug class count: 0`
- `git diff --check -- tools/audit_substrate_driver_mode_hardcode.py src/tac/tests/test_check_326_substrate_driver_consumes_trainer_mode_env_var.py`
  - clean

## Six-Hook Wire-In

1. Sensitivity-map contribution: N/A. This is a guard repair, not an empirical
   score anchor.
2. Pareto constraint: protects Pareto/evidence comparisons from smoke/full
   mode contamination in future dispatches.
3. Bit-allocator hook: N/A. No archive or byte allocation changed.
4. Cathedral autopilot dispatch hook: active. The operator-facing helper and
   `preflight_all()` gate now share the stricter classifier.
5. Continual-learning posterior update: records the false-safety class so
   future driver-mode bugs are classified as custody hazards, not lane method
   failures.
6. Probe-disambiguator: N/A. There are not two defensible interpretations of
   the env provenance rule; only `env_overrides` may set dispatch env vars.

## Reactivation Criteria

If a future driver intentionally hardcodes `--smoke`, every matching recipe
must be research/smoke-only or dispatch-disabled, or the driver must carry a
substantive waiver. If a dispatchable sibling exists, the driver must expose a
mode env var and the dispatch recipe must set it explicitly.
