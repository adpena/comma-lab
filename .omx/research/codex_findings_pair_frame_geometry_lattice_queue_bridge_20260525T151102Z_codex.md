# Pair-Frame Geometry Lattice Queue Bridge

- timestamp_utc: 2026-05-25T15:11:02Z
- agent: codex
- scope: DQS1 pairset acquisition, scorer geometry, local materializer queue
- authority: local planning and queue starts only

## What Changed

The pair/frame scorer-geometry bridge is now a reusable TAC surface instead of
an advisory note. `pair_frame_scorer_geometry_lattice.v1` fuses DQS1 rank order,
SegNet-last-frame/PoseNet-pair topology, frame-pair curriculum rows, and
pair-component X-ray rows into low-impact pair-drop priorities.

The lattice emits only queue-executable DQS1 pairset drop requests. Masked,
feathered, and inverse-scorer null-direction variants remain explicit blocked
families until receiver/materializer semantics exist. This keeps the broad
rate/distortion idea moving without pretending parser-local or advisory signal
is score authority.

`decoder_q_pairset_acquisition.v1` now accepts the lattice and turns those
requests into normal acquisition rows with `selector_kind=
pair_frame_geometry_low_impact_drop_many`. The existing cross-family portfolio
and DQS1 local-first queue can consume them directly.

`cross_family_candidate_portfolio` now preserves each pairset row's
`distortion_repair_budget_from_rate_savings` inside `source_metadata`, so saved
bytes remain visible to queue consumers as PoseNet/SegNet repair budget.

`tools/operator_briefing.py` now counts selected geometry candidates in frontier
feedback refreshes so these starts are visible through the normal briefing path.

## Artifacts

- lattice:
  `.omx/research/codex_pair_frame_geometry_lattice_20260525T151102Z/pair_frame_scorer_geometry_lattice.json`
- geometry-enriched acquisition:
  `.omx/research/codex_pair_frame_geometry_lattice_20260525T151102Z/dqs1_pairset_acquisition_geometry_lattice.json`
- portfolio:
  `.omx/research/codex_pair_frame_geometry_lattice_20260525T151102Z/cross_family_portfolio.json`
- action summary:
  `.omx/research/codex_pair_frame_geometry_lattice_20260525T151102Z/action_summary.json`
- follow-up queue:
  `.omx/research/codex_pair_frame_geometry_lattice_20260525T151102Z/frontier_refresh/dqs1_followup_queue.json`

## Observed Artifact Summary

- lattice rows: 32
- geometry coverage over the current best DQS1 selector pairs: 1.0
- queue-executable lattice requests: 6
- geometry-enriched acquisition candidates: 587
- geometry acquisition candidates: 6
- follow-up queue: 24 experiments, 168 steps
- selected geometry starts in queue: 2
- example selected geometry start:
  `pairset_geometry_lowimpact_k006_h2866d55ee6`
- example saved-rate repair budget:
  6 bytes saved, score budget `3.995153718733028e-06`,
  SegNet distortion budget at fixed PoseNet `3.995153718733028e-08`

All score, promotion, rank/kill, GPU, and dispatch authority fields remain
false.

## Verification

- `ruff` on touched lattice, acquisition, portfolio, briefing, CLI, and tests:
  passed
- `pytest src/tac/tests/test_pair_frame_scorer_geometry_lattice.py src/tac/tests/test_decoder_q_pairset_acquisition.py src/tac/tests/test_cross_family_candidate_portfolio.py -q`:
  31 passed
- `tools/experiment_queue.py --queue .omx/research/codex_pair_frame_geometry_lattice_20260525T151102Z/frontier_refresh/dqs1_followup_queue.json validate`:
  valid, 24 experiments, 168 steps

## Residual Gap

The bridge makes global low-impact DQS1 pair drops queue-executable, but it
does not yet implement non-pair-drop receiver semantics. The next high-EV step
is a receiver/materializer contract for within-selected-set masked/feathered
variants, followed by binding inverse-scorer action cells into the same lattice
request schema.
