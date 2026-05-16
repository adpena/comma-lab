# Cathedral Dispatch Source-Scope No-Signal-Loss Hardening

Date: 2026-05-16
Operator directive: continue L5/Cathedral bug hunting, adversarial review,
production OSS rigor, paper fidelity, and no signal loss.

## Finding

Read-only adversarial audit found that source-scope fields preserved in
`ParetoRow` could be dropped when rows became `RankedDispatchCandidate`,
`CandidateRow`, `HaltEvent`, or authorized journal rows. This was especially
important for L5/L5-v2 and other literature-seeded class shifts, where a
`literature_anchor` without `source_supports`, `paper_claim_scope`,
`pact_must_prove`, and `decode_complexity_evidence` can become false
authority.

The broader test sweep also exposed an adjacent planning-loader bug:
substrate-composition rankings can legitimately contain `$0` planning-cost
rows, but the loader used the strict positive-cost guard intended for actual
authorization. That made real ranking artifacts fail to load even though paid
dispatch authorization still separately requires a positive cost.

## Patch

- Extended `RankedDispatchCandidate` with source-scope fields and
  `source_fidelity_metadata`.
- Populated singleton and orthogonal-pair dispatch candidates from `ParetoRow`
  source-scope fields.
- Propagated source-scope fields through `as_candidate_row_kwargs()`.
- Added source-scope fields to `HaltEvent` and authorized journal JSONL rows.
- Split planning-source cost parsing from dispatch authorization cost parsing:
  rank/probe loaders allow finite nonnegative planning costs, while actual
  authorization still uses finite positive costs.

## Verification

```
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q \
  src/tac/tests/test_autopilot_dispatch_ranking.py \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py \
  src/tac/tests/test_cathedral_autopilot_substrate_composition_wire.py \
  src/tac/tests/test_build_composition_ranking_json.py
# 189 passed

.venv/bin/python -m ruff check \
  src/tac/optimization/autopilot_dispatch_ranking.py \
  tools/cathedral_autopilot_autonomous_loop.py \
  src/tac/tests/test_autopilot_dispatch_ranking.py \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py
# All checks passed

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  src/tac/optimization/autopilot_dispatch_ranking.py \
  tools/cathedral_autopilot_autonomous_loop.py \
  src/tac/tests/test_autopilot_dispatch_ranking.py \
  src/tac/tests/test_cathedral_autopilot_autonomous_loop.py
```

## Follow-Up

The next audit item is prediction-band validation: landed anchors must be
axis-matched and artifact/custody complete before they can influence rank
reward.
