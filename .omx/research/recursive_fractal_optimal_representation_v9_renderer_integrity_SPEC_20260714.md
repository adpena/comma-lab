# Task503 renderer implementation custody fix

## Objective

Extinguish the spoofable-renderer bug in `DecisionCarrierBundle`: a renderer object must not gain
authority by self-declaring the `reference_id`, SHA-256, and length already present in the bundle.
Before any RGB render, the bundle renderer reference must be resolved to actual implementation
bytes, those bytes must match the declared SHA-256 and length, and a caller-supplied trusted loader
must construct the executable renderer from those verified bytes.

## Constraints

- Ownership is limited to `src/tac/boundary_math/decision_carrier_bundle.py` and its focused test.
- Preserve the canonical `DCB1` byte grammar and all existing section parse-back behavior.
- Do not touch trainer, autoconfig, DSL provenance, checkpoint, or sibling files.
- Missing resolver, missing loader, wrong bytes, a loader failure, or identity mismatch must fail
  closed before rendering.
- `audit_repeat` remains a determinism check; it must not masquerade as code-identity custody.
- No score, selected-M, archive-compression, or receiver-rate claim.

## Acceptance

```sh
.venv/bin/python -m pytest -q src/tac/boundary_math/tests/test_decision_carrier_bundle.py
.venv/bin/python -m ruff check src/tac/boundary_math/decision_carrier_bundle.py src/tac/boundary_math/tests/test_decision_carrier_bundle.py
.venv/bin/python -m ruff format --check src/tac/boundary_math/decision_carrier_bundle.py src/tac/boundary_math/tests/test_decision_carrier_bundle.py
python3 -m py_compile src/tac/boundary_math/decision_carrier_bundle.py src/tac/boundary_math/tests/test_decision_carrier_bundle.py
```

Tests must include a renderer with spoofed identity attributes that is refused unless it was
created through the verified-bytes loader path.
