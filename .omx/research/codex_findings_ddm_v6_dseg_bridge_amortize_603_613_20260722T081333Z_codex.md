---
schema: codex_findings.v1
task: 603
feeds_task: 613
review_round: 1
reviewer: codex:gpt-5.6-sol
main_landing_review_required: true
---

# Round-1 findings — DDM v6 evaluator bridge and amortization

## Disposition

`PASS_AFTER_FIX`, research-only.  Ruff is clean and the focused suite is 39/39.  No contest score,
n600 result, or promotion is claimed.

## Finding 1 — amortized membership reporting gap

- Severity: medium, evidence-language bug.
- Observed: first-run candidate receipts contained actual `d_seg` and `d_pose`, but no new
  same-C1 membership field even though the delegated output contract asks for `(bytes, membership,
  advisory-d_seg)` points.
- Risk: a consumer might silently substitute `1-d_seg` and mislabel it as measured C1 membership.
- Fix: `0e7be4bbb6` adds an explicit triangle-inequality interval using the settled
  C1-to-GT match `0.999873638153`.  Exact v5 controls preserve their previously MEASURED membership;
  amortized rows are `DERIVED_BOUND` only.
- Test: `test_v6_membership_proxy_is_an_explicit_triangle_bound` proves the measured control lies
  inside the bound and that `score_claim=false` is preserved.

## Finding 2 — AR naming audit

- Severity: low, documentation precision.
- Observed: `fixed_ar1_hold24` and `xi_pose6_ar1_hold24` are not fitted general-phi AR models.
- Disposition: naming is retained only with the exact equation `x_t=1*x_{t-1}+0` between keys,
  i.e. a unit-root zero-innovation AR(1) hold.  The equations note makes this explicit.

## Remaining debt

- The absolute SegNet failure is real and formulation-scoped: best measured `d_seg` is
  `0.038300534089`; the family is not killed.
- n600 was not run within this bounded arm.
- contest-CPU/CUDA evaluation was neither authorized nor run.
- Canonical #603 register count remains 8/19 until MAIN reviews/registers the draft row.
