# Cathedral Evidence Source-Scope Preservation

Date: 2026-05-16
Operator directive: continue L5/Cathedral bug hunting, adversarial review,
paper fidelity, OSS hardening, and no-signal-loss integration.

## Finding

Catalog #293 guarded literature-anchor scope fields in the canonical substrate
composition matrix and serialized Pareto rows, but `tools/cathedral_autopilot.py`
did not model those fields on `TechniqueEvidence`. Evidence rows could therefore
carry `literature_anchor`, `source_supports`, `paper_claim_scope`,
`pact_must_prove`, and `decode_complexity_evidence` in JSON/JSONL and then lose
that source-boundary metadata before validation-queue review.

Impact: a literature-cited candidate could keep a persuasive anchor while
dropping the boundary between what the paper supports and what Pact still must
prove on byte-closed contest archives. That is a no-signal-loss and
false-authority risk for Cathedral/L5 planning.

## Patch

- Added `tac.optimization.literature_source_scope` as the shared source-scope
  helper surface.
- Refactored Catalog #293 preflight helpers to use the shared helper instead of
  duplicating placeholder logic in `src/tac/preflight.py`.
- Extended `TechniqueEvidence` and `_load_evidence()` to preserve source-scope
  fields.
- Added fail-closed blockers of the form
  `literature_anchor_source_scope_missing:<field>` when an evidence row carries
  a literature anchor but misses required scope fields.
- Routed source-scope blockers through the promotability check itself, so direct
  `TechniqueEvidence` construction cannot bypass the literature-anchor contract.
- Preserved source-scope metadata for unknown literature-seeded candidates in
  the Cathedral validation queue.

## Verification

```
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_cathedral_autopilot.py \
  src/tac/tests/test_check_293_cathedral_literature_scope.py
# 39 passed

.venv/bin/python -m ruff check \
  tools/cathedral_autopilot.py \
  src/tac/optimization/literature_source_scope.py \
  src/tac/tests/test_cathedral_autopilot.py \
  src/tac/tests/test_check_293_cathedral_literature_scope.py
# All checks passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  tools/cathedral_autopilot.py \
  src/tac/optimization/literature_source_scope.py \
  src/tac/tests/test_cathedral_autopilot.py

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
from tac.preflight import check_cathedral_literature_anchors_have_source_scope
print(check_cathedral_literature_anchors_have_source_scope(strict=False))
PY
# []
```

## Follow-Up

The next L5/Cathedral hardening pass should check whether ranked catalog rows
that originate from literature-scoped substrate inventory should surface the
same `source_scope` bundle directly in `recommended_top_3`, not just validation
queue and evidence paths.
