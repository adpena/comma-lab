# Catalog #323 Score-Key Synonym And Waiver Hardening - 2026-05-17

## Status

Adversarial review of the fresh canonical Provenance umbrella found two
false-authority holes:

1. The Catalog #323 preflight gate and operator audit tool caught
   `recomputed_score` / `auth_eval_score`, but missed common live Pact score
   spellings such as `score_recomputed`, `score_recomputed_from_components`,
   `canonical_score`, `canonical_score_recomputed`, `score_contest_cuda`,
   `score_contest_cpu`, `auth_eval_recomputed_score`, and related harvest
   variants.
2. JSON string values containing `# PROVENANCE_CANONICAL_WAIVED:<rationale>`
   could be mistaken for a file-level waiver because the file-header scan
   searched the raw first lines, not comment-style header lines only. Placeholder
   row waivers could therefore suppress the missing-provenance violation.

Both defects are score-authority bugs: they let persisted exact-eval or harvest
score rows evade the canonical Provenance contract.

## Fix

- Expanded the score-claim key vocabulary in:
  - `src/tac/preflight.py`
  - `src/tac/provenance/validator.py`
  - `tools/audit_provenance_compliance.py`
- Added Catalog #323 waiver-rationale validation and restricted file-level
  waivers to comment-style header lines. JSON row waivers remain allowed only
  when the rationale is explicit, non-placeholder, and at least 3 characters.
- Added regression coverage in:
  - `src/tac/tests/test_check_323_canonical_provenance.py`
  - `src/tac/tests/test_provenance_validator.py`

## Verification

Focused:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_check_323_canonical_provenance.py \
  src/tac/tests/test_provenance_validator.py -q
```

Result: `79 passed in 8.00s`.

Broader provenance/Catalog 322/323 cluster:

```bash
.venv/bin/python -m pytest \
  src/tac/tests/test_provenance_contract.py \
  src/tac/tests/test_provenance_builders.py \
  src/tac/tests/test_provenance_validator.py \
  src/tac/tests/test_provenance_adapters.py \
  src/tac/tests/test_check_322_phantom_provenance_composition_alpha.py \
  src/tac/tests/test_check_323_canonical_provenance.py -q
```

Result: `186 passed in 7.73s`.

Operator-facing audit:

```bash
.venv/bin/python tools/audit_provenance_compliance.py --summary
```

Result at `2026-05-17T23:06:25Z`:

- total artifacts scanned: `1928`
- clean: `1730`
- warn: `2`
- violation: `196`
- classifier counts: `MISSING_PROVENANCE=135`, `INVALID_PROVENANCE_SHAPE=61`

Catalog #323 preflight count after synonym expansion:

```bash
.venv/bin/python - <<'PY'
from tac.preflight import check_no_score_claim_without_canonical_provenance
violations = check_no_score_claim_without_canonical_provenance(strict=False, verbose=False)
print(len(violations))
for item in violations[:10]:
    print(item)
PY
```

Result: `544` warn-only violations, with the first live expanded-key hit at
`.omx/state/lightning_active_jobs.json` for `score_contest_cuda`.

## Evidence Discipline

This is a custody/audit hardening result, not a score claim. It does not change
candidate archive bytes, does not promote any artifact, and does not convert
legacy rows into canonical Provenance. It improves the bug-hunting surface by
preventing common score-key spellings and placeholder waivers from escaping
Catalog #323.
