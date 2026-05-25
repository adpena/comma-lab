# DQS1 Pairset Queue-Consumable Bridge Hardening

- timestamp_utc: 2026-05-25T19:40:26Z
- agent: codex
- scope: DQS1 eureka/drop-many signal surface, inverse-action materialization bridge, byte-shaving materializer queue
- authority: planning/local materialization only; no score, promotion, rank/kill, dispatch, or GPU authority

## What Changed

The inverse-action materialization bridge no longer requires a water-bucket
portfolio row before PacketIR can become visible to queue planning. Direct
DQS1 pairset PacketIR operation sets now emit
`packet_ir_operation_set_materialization_bridge_link.v1` rows and are counted
as queue-consumable when they have:

- target kind `dqs1_pairset_drop_pair`;
- materializer `dqs1_pairset_drop_pair_adapter`;
- receiver contract kind `archive_charged_pairset_runtime_selector`;
- either a legacy executable marker or the safer
  `materializer_adapter_registered=true` /
  `registered_adapter_blocked_until_context_and_receiver_proof` status.

This preserves the partner hardening that demotes acquisition-derived DQS1
operations from "executable" to "adapter registered, blocked until context and
receiver proof", while still letting the queue see bounded local materializer
work. `executable_work_ready` remains false until DQS1 base-pair context,
locality controls, receiver runtime consumption proof, same-runtime parity, and
exact auth eval are present.

The queue compiler now also recognizes ranked DQS1 pairset selector units as
atomic local materializer starts. That is the important non-leaf fix: the eureka
rows are full selectors such as drop-many/masked geometry candidates, not
single-pair operations. Aggregate operation-set bundles remain blocked when
they combine alternative selectors, while direct ranked selector rows can
compile into local-first DQS1 materialization actions.

The inverse-steganalysis action functional now preserves the empirical DQS1
outcome anchors even when the top PacketIR operation sets are dominated by
pairset-acquisition priors. Operation-set atoms still expose their selected
operation provenance, while supplemental ranked-unit cells carry the harvested
`dqs1_outcome_signal` rows into the same action surface. This prevents the 90
measured DQS1 outcome units from being hidden by the operation-set ladder.

## Verification Snapshot

In-memory rebuild from
`.omx/research/codex_pairset_acquisition_signal_surface_20260525T191605Z/byte_shaving_signal_surface.pairset_acquisition_outcomes.json`:

- ranked units: `1216`
- harvested DQS1 outcome units: `90`
- operation-set rows: `32`
- PacketIR operation sets: `32`
- queue-consumable PacketIR operation sets: `32`
- executable-work-ready PacketIR operation sets: `0`
- queue context requirements: `dqs1_base_pair_indices`
- first PacketIR status accepted:
  `registered_adapter_blocked_until_context_and_receiver_proof`
- inverse-action functional cells: `122`
- inverse-action cells with direct DQS1 outcome signal: `90`

Queue compile probe with the first selector's 32-pair base set and explicit
partial-materialization rationale:

- executable rows: `4`
- blocked rows: `131`
- queueable rows: `4`
- top executable selection kinds: all `ranked_unit`
- top dropped-pair counts: `16`, `12`, `8`, `8`
- materialization blockers on top executable rows: none
- score/promotion/rank/dispatch authority: all false

## Eureka JSON Result Context

The latest beyond-drop-two acquisition artifact has `581` candidates:

- `495` drop-two candidates;
- `34` drop-many candidates;
- `31` drop-one candidates;
- `12` prefix candidates;
- `9` diversity candidates.

The top acquisition rows are still planning-only. The highest-ranked rows are
`pairset_diversity_k002` and `pairset_prefix_k002`; the first drop-many row is
rank 11, keeps 26 of the 32 base pairs, drops 6 pairs, and saves 6 descriptor
bytes versus the source selector. Those saved bytes are now represented as a
machine-readable SegNet/PoseNet repair budget, not a chat-only intuition.

## Residual Gaps

The masked/feathered within-set family, global no-impact frame/pair drops,
frame-vs-pair SegNet/PoseNet geometry, and master-gradient/inverse-scorer
binding remain the next higher-level optimizer work. This patch makes their
DQS1 selector descendants queue-visible without granting exact-score authority.

The partner MLX parity audit is orthogonal but relevant to the scorer substrate:
Kahan/compensated summation, FP64 Conv2d accumulation, and MLX-side
deterministic reduction enforcement remain genuinely unexplored. Those belong
to the MLX scorer/inverse-scorer substrate hardening lane, not this DQS1 queue
patch, but they should be treated as real drift blockers if MLX parity starts
driving spend or optimizer decisions.

## Verification

- `ruff` on touched DQS1 bridge/queue/builder/test files: passed
- `PYTHONPATH=. pytest src/tac/tests/test_byte_shaving_campaign.py src/tac/tests/test_byte_shaving_signal_surface_builder.py src/tac/tests/test_inverse_steganalysis_acquisition.py src/tac/tests/test_byte_shaving_campaign_queue.py -q`: `185 passed`
