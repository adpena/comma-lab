# Adjudicated JSON promotion-stamp hardening - 2026-05-08

## Finding

Adversarial review found that `scripts/adjudicate_contest_auth_eval.py` wrote
`--result-copy` before regression, component, sane-score, hardware, and
distillation gates were applied. A copied `contest_auth_eval.adjudicated.json`
could therefore retain raw top-level fields such as:

- `evidence_grade: "A++"`
- `promotion_eligible: true`

even when `adjudication_provenance.json` correctly demoted the result to
`A-negative scoped forensic`.

The concrete reproducer was the PR106 UNIWARD rms=0.05 exact CUDA regression:
the provenance was non-promotable, but the copied auth-eval JSON still looked
promotable to downstream compliance checks.

## Patch

`scripts/adjudicate_contest_auth_eval.py` now writes `--result-copy` only after
final adjudication fields are computed. The copied JSON receives:

- `adjudication` summary object
- final `promotion_eligible`
- final `score_claim_valid`
- final `score_claim`
- final `evidence_grade`
- `paper_claim_grade`
- `allowed_use`
- `lane_status`
- gate booleans and regression/component/sane-score details

For non-promotable results, the copied JSON is explicitly demoted to
`A-negative scoped forensic` and carries `rank_or_kill_eligible: false`.

## Local artifact repair

The local PR106 UNIWARD artifact was regenerated with the patched adjudicator.
Its copied JSON now agrees with provenance:

```text
promotion_eligible=false
score_claim_valid=false
score_claim=false
evidence_grade=A-negative scoped forensic
lane_status=REGRESSION_REVIEW_REQUIRED
```

The artifact directory is local/generated custody state and is not committed.

## Verification

```bash
.venv/bin/python -m pytest -q src/tac/tests/test_remote_auth_eval_hardening.py -k 'adjudicator'
.venv/bin/python -m py_compile scripts/adjudicate_contest_auth_eval.py src/tac/tests/test_remote_auth_eval_hardening.py
.venv/bin/ruff check scripts/adjudicate_contest_auth_eval.py src/tac/tests/test_remote_auth_eval_hardening.py
```

The regression coverage asserts that a PR106-style regression cannot emit a
copied adjudicated JSON with promotable top-level fields.
