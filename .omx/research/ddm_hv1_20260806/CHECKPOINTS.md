# ddm_hv1 Checkpoints

## Checkpoint 1 - Governing Context

Read the charter, common contract, `CLAUDE.md`, `AGENTS.md`, `PROGRAM.md`,
`docs/operating_manual_craft_handoff.md`, and `.omx/state/main_hot_state.md`.
Boundary: scorer-free, no protected edits, no index manipulation, serializer commit required.

## Checkpoint 2 - Queue Stores

Consumed NP1 queue surfaces, FO1/QJ1 follow-on ledgers, and probe outcomes. Key counts:
NP1 current files 56 NEXT rows / 79 final-message rows; FO1 460 queued rows with owners and
0 unowned queued rows; QJ1 390 queued rows all with owners; probe outcomes 662 rows.

## Checkpoint 3 - Content Routing

Routed the charter row-groups into `.omx/research/ddm_hv1_20260806/HV1_DISPOSITION_LEDGER.jsonl`.
Scoped absence recorded for standalone VW1 and recent BP1 receipt paths; US2 residue verified
already committed at `bb9a69bbb4`.

Own-vehicle frontier line: `S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`;
contest pointer `0.1910828242` borrowed/unmoved.
