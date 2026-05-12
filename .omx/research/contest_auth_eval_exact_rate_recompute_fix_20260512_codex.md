# Contest auth eval exact-rate recompute fix (2026-05-12)

## Summary

`experiments/contest_auth_eval.py` previously recomputed
`score_recomputed_from_components` from the rounded `Compression Rate:` text
printed by `upstream/evaluate.py`. The upstream scorer computes the actual rate
as:

```text
archive.zip bytes / original uncompressed bytes
```

but prints the rate with 8 decimal places. That created a tiny but real custody
drift in exact-score JSONs.

## Fix

The parser now:

- parses `Submission file size:` and `Original uncompressed size:`;
- refuses a report whose submission size does not match the observed archive
  size supplied by the wrapper;
- recomputes `rate_unscaled` from exact integer bytes;
- preserves the printed rounded rate as `rate_unscaled_reported_rounded`.

Focused guard:

- `test_parse_report_uses_exact_byte_rate_not_rounded_printed_rate`

## Impact

For PR106/R2 PR101-grammar exact CUDA artifact:

- archive bytes: `186780`
- exact rate term: `25 * 186780 / 37545489 = 0.12436913526415916`
- rounded-report rate term: `25 * 0.00497477 = 0.12436925`
- score drift avoided going forward: `~1.15e-7`

This does **not** change any rank-scale conclusion, but it closes an
apples-to-apples evidence bug before more byte-level candidates are compared.
Existing harvested JSONs remain historical artifacts; future auth-eval runs
will carry exact byte-derived rate components.
