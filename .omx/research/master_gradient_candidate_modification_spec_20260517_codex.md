# Master-Gradient CandidateModificationSpec - 2026-05-17

## Context

Catalog #318 now blocks authority-bearing raw archive-byte master-gradient
surfaces. The remaining engineering gap was the positive replacement
interface: code and WIP notes referred to `CandidateModificationSpec`, but the
committed operator-plan module did not yet define it.

This matters for the active L5 / Rule #6 / FEC6 route because a valid
score-response experiment must mutate packet grammar, not raw bytes. The next
object needs to be machine-readable and precise enough for autopilot, packet
builders, and paired CPU/CUDA review.

## Patch

Added `CandidateModificationSpec` to
`src/tac/master_gradient_operator_plan.py`.

The spec is intentionally narrow:

- `mutation_grain="grammar_aware_operator"`
- `coordinate_system="grammar_aware_operator_response"`
- `raw_archive_byte_coordinates_allowed=false`
- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_provider_dispatch=false`
- `ready_for_exact_eval_dispatch=false`

`build_master_gradient_operator_plan(...)` now emits
`candidate_modification_specs` alongside the lower-level operator rows, and the
batch payload carries `candidate_modification_spec_count`.

This is not a score claim. It is the packet-valid bridge needed before any
master-gradient-like signal can influence candidate ranking or dispatch.

## Adversarial Review

Rejected path: raw `{byte_idx: delta}` maps and `(N_archive_bytes, 3)` sidecars.
Those remain blocked by Catalog #318 and by
`tac.master_gradient_feasibility`.

Accepted path: a candidate declares the exact logical packet section and
mutation operator, then a downstream builder must prove:

1. rebuilt archive bytes;
2. ZIP local/central headers refreshed;
3. CRC refreshed;
4. inflate success;
5. byte-consumption/no-op detector proof;
6. axis-labelled result review before promotion.

The spec deliberately does not contain byte indices or byte-value deltas. A
future FEC6 or Rule #6 packet builder can consume the spec and materialize real
archive bytes, but the spec itself stays planning/probe-only.

## Verification

Focused tests:

```text
.venv/bin/python -m pytest \
  src/tac/tests/test_master_gradient_operator_plan.py \
  src/tac/tests/test_master_gradient_feasibility.py \
  src/tac/tests/test_check_318_master_gradient_raw_authority.py
```

Result: `18 passed`.

Lint:

```text
.venv/bin/ruff check \
  src/tac/master_gradient_operator_plan.py \
  src/tac/tests/test_master_gradient_operator_plan.py
```

Result: `All checks passed`.

## Next Route

Use this spec as the bridge for the next score-lowering artifact:

- FEC6 selector / FE6E / format0D-extra grammar mutation;
- Rule #6 A1 entropy-stack or Ballé-sideinfo bolt-on;
- L5/TT5L packet-valid side-info cells.

Each next builder should accept or emit `CandidateModificationSpec` and produce
a byte-closed archive plus inflate/consumption proof before any paired exact
dispatch.
