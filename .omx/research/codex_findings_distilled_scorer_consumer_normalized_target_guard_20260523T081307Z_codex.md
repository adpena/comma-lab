# Codex Findings - Distilled Scorer Consumer Normalized Target Guard

timestamp_utc: 2026-05-23T08:13:07Z
agent: codex
lane_id: lane_codex_distilled_scorer_consumer_normalized_target_guard_20260523
score_claim: false
promotion_eligible: false
rank_or_kill_eligible: false
score_claim_valid: false

## Scope

Follow-up to the OOF scorer-response normalized-objective guard. The cathedral
consumer for distilled scorer surrogate evidence still selected best rows and
improvement counts directly from `delta_vs_baseline_score` /
`scorer_delta_vs_baseline`. That reopened the exact bug class the OOF guard
closed: MLX-derived scorer-response rows could look strongly improving at the
window-local/raw target while their normalized full-video planning objective was
non-improving.

While reviewing adjacent false-authority surfaces, the PR101/FEC6 PacketIR
candidate queue producer/consumer/matrix path also omitted `score_claim_valid`
from its non-authority contract. The queue was already observability-only, but
that missing canonical field could let downstream readers miss the same
authority boundary used by MLX/local/proxy score rows.

## Landed Fix

- `distilled_scorer_surrogate_canonical_equation_consumer` now consumes
  scorer-response targets through
  `scorer_response_planning_value_for_target(...)`.
- The consumer blocks MLX-sourced distilled rows missing a valid normalized
  full-video objective instead of emitting a routing signal.
- Best total/scorer rows and improved-row counts now reflect normalized
  full-video planning values for MLX rows.
- Routing payloads expose `planning_target_accessor` so downstream autopilot and
  review surfaces can verify that the canonical target path was used.
- Distilled scorer test fixtures now carry explicit `score_claim_valid=false`.
- `pr101_fec6_candidate_queue` now emits `score_claim_valid=false` at top-level,
  byte-accounting, and candidate rows, and its markdown renders the flag.
- `packetir_candidate_queue_consumer` rejects `score_claim_valid=true` on queue
  and candidate payloads and always returns `score_claim_valid=false`.
- `pr101_frontier_packetir_matrix` now treats `score_claim_valid` as a
  false-authority field and echoes `score_claim_valid=false` on candidate-queue
  rows.

## Regression Guards

- A distilled scorer row with `source_schema="mlx_scorer_response.v1"` and raw
  `delta_vs_baseline_score=-10` but normalized projected full-video delta
  `+0.001` produces zero improvement counts and a positive best-row planning
  value.
- The same MLX-sourced row without normalized objective fields returns a
  fail-closed `distilled_scorer_surrogate_authority_blocked` payload with
  `score_claim_valid=false`.
- PacketIR queue construction asserts `score_claim_valid=false` across queue,
  byte-accounting, and candidates.
- PacketIR consumer and frontier matrix tests now fail if
  `score_claim_valid=true` slips into queue/candidate artifacts.

## Verification

```bash
.venv/bin/python -m pytest -q \
  src/tac/cathedral_consumers/distilled_scorer_surrogate_canonical_equation_consumer/tests/test_consumer.py \
  src/tac/tests/test_cathedral_consumer_contract.py \
  src/tac/tests/test_cathedral_autopilot_auto_discovery.py \
  src/tac/tests/test_scorer_response_dataset.py
```

Result: `151 passed in 1.16s`.

```bash
.venv/bin/python -m pytest -q \
  src/tac/tests/test_pr101_fec6_candidate_queue.py \
  src/tac/tests/test_pr101_frontier_packetir_matrix.py \
  src/tac/tests/test_pr101_fec6_runtime_consumption.py
```

Result: `31 passed in 0.82s`.

Combined regression:

```bash
.venv/bin/python -m pytest -q \
  src/tac/cathedral_consumers/distilled_scorer_surrogate_canonical_equation_consumer/tests/test_consumer.py \
  src/tac/tests/test_cathedral_consumer_contract.py \
  src/tac/tests/test_cathedral_autopilot_auto_discovery.py \
  src/tac/tests/test_scorer_response_dataset.py \
  src/tac/tests/test_pr101_fec6_candidate_queue.py \
  src/tac/tests/test_pr101_frontier_packetir_matrix.py \
  src/tac/tests/test_pr101_fec6_runtime_consumption.py
```

Result: `182 passed in 1.87s`.

```bash
.venv/bin/ruff check \
  src/tac/cathedral_consumers/distilled_scorer_surrogate_canonical_equation_consumer/__init__.py \
  src/tac/cathedral_consumers/distilled_scorer_surrogate_canonical_equation_consumer/tests/test_consumer.py \
  src/tac/cathedral_consumers/packetir_candidate_queue_consumer/__init__.py \
  src/tac/packet_compiler/pr101_fec6_candidate_queue.py \
  src/tac/packet_compiler/pr101_frontier_packetir_matrix.py \
  src/tac/tests/test_pr101_fec6_candidate_queue.py \
  src/tac/tests/test_pr101_frontier_packetir_matrix.py
git diff --check
```

Result: `ruff` passed after import normalization and `git diff --check` was
clean.

Review tracker:

- `policy-check` clean for the modified tracked Python files.
- The two cathedral consumer `__init__.py` files currently have zero tracked
  review entities in the AST tracker, so their policy check reports `0 entities
  compliant, 0 violations`.

## Remaining Work

1. Apply the same normalized target / false-authority audit to PACT-NeRV
   duplicate-validation surfaces if they ingest scorer-response rows directly.
2. Route byte-shaving scorer-response refs through either this canonical target
   accessor or a typed scorer-response ref schema that preserves the normalized
   objective scope.
