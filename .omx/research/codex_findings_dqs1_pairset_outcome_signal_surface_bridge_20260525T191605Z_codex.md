# DQS1 pairset outcome signal surface bridge

## Verdict

The DQS1 pairset/drop-many work is now less leaf-shaped: acquisition candidates
and harvested K>1 local advisory outcomes flow into the byte-shaving signal
surface, campaign plan, PacketIR operation sets, and inverse-steganalysis action
functional without gaining score, promotion, rank/kill, or dispatch authority.

## Landed integration

- Added `--dqs1-observation-jsonl` to `tools/build_byte_shaving_signal_surface.py`.
- Added typed `dqs1_pair_frame_geometry_outcome_anchor.v1` refs from
  `mlx_dynamic_sweep_observation.v1` / `dqs1_local_first_harvest.v1` rows.
- Promoted byte-saving DQS1 harvest rows into planning units only when runtime
  evidence is present: archive SHA, runtime SHA, inflated/cache SHA, advisory
  artifact SHA, planner artifact SHA, baseline artifact SHA, byte delta, score
  delta, selected pairs, and dropped pairs.
- Preserved empirical anchor metadata through campaign ranking and inverse
  action provenance via `dqs1_outcome_signal`, `inverse_scorer_signal`, and
  `bit_allocator_signal`.
- Hardened pair-index parsing so malformed selected/dropped pair IDs fail
  closed as `ByteShavingCampaignError` instead of raising raw conversion errors
  or silently emitting partial pair lists.
- Kept all rows fail-closed: macOS-CPU advisory outcomes remain planning
  anchors only and cannot claim score or authorize dispatch.

## Real artifacts

Output directory:
`.omx/research/codex_pairset_acquisition_signal_surface_20260525T191605Z/`

- `byte_shaving_signal_surface.pairset_acquisition_outcomes.json`
  - 1,216 units total.
  - 2 pairset acquisition refs.
  - 6 DQS1 outcome refs.
  - 90 empirical outcome units.
  - 78 K>1 empirical outcome rows.
- `byte_shaving_campaign_plan.pairset_acquisition_outcomes.json`
  - 1,216 ranked units.
  - 32 operation sets.
  - 32 PacketIR operation sets.
  - 0 queue-consumable materialization-bridge operation sets yet; the bridge is
    correctly blocked on the missing receiver/materializer execution contract.
  - Top operation set: 32 units, 251 predicted saved bytes,
    0.000167130597233665 expected score gain.
- `inverse_action_materialization_bridge.pairset_acquisition_outcomes.json`
- `inverse_steganalysis_action_functional.pairset_acquisition_outcomes.json`
  - 32 cells emitted.
  - 0 water-bucket selections under current conservative residual rule.
  - 0 blocked cells.
  - 0.004671932758686403 expected gain sum recorded as planning signal.

## Verification

- `ruff` clean on touched Python surfaces.
- `pytest src/tac/tests/test_byte_shaving_signal_surface_builder.py -q`
  - 17 passed.
- `pytest src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_inverse_steganalysis_acquisition.py -q`
  - 83 passed.
- `pytest src/tac/tests/test_byte_shaving_campaign_queue.py -q`
  - 81 passed.
- `tools/lane_maturity.py validate`
  - 1358 lanes validated cleanly.

## Remaining gap

The masked/feathered receiver/materializer contract is still correctly blocked.
The next high-EV engineering step is to make that runtime contract executable
against the same outcome-anchor bridge, rather than adding another advisory
variant that cannot be consumed by the solver stack.
