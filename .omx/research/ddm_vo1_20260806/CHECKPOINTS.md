# DDM VO1 - Checkpoints

## Contract Checkpoints

- Charter read: `.omx/tmp/codex_runs/vo1_prompt.md`.
- Common contract read: `.omx/tmp/codex_runs/_common_contract.md`.
- Governing files read before routing: `CLAUDE.md`, `AGENTS.md`,
  `PROGRAM.md`, `docs/operating_manual_craft_handoff.md`,
  `.omx/state/main_hot_state.md`, and `canonical_frontier_pointer.json`.
- Scorer-free: no scorer, no `upstream/evaluate.py`, no archive mutation, no
  training launch, no GPU/remote dispatch.
- Protected files not edited:
  `.omx/research/ddm_cr1_composition_row_827_20260801.md`,
  `.omx/research/ddm_pu2_pose_tail_floor_probe_20260803.md`, and
  `src/tac/optimization/direct_description_carrier_compose.py`.
- `/tmp` not used as evidence.
- Staged index check before editing showed no staged paths.

## Evidence Checkpoints

The following prior receipts or ledgers were read and used as evidence:

- SW1 receipt and seam ledger: solve-within eta versus project-after eta.
- DK1 receipt: naive rounding, Dykstra, and CVP/Babai realization ladder.
- CA1 receipt: cap-default and silent-cap census.
- NA6/OA1 receipts: prior negative audit and fresh-eyes overread corrections.
- LC1 receipt: PE3 aggregate target-label result and scope.
- ET1 priced-band memo: phase-field eta, pose block, and Q3 projection cue.
- FD1/FD2 receipts: GN/CG zero-accept and realized uint8 seg-gap diagnosis.
- V19C correction saturation memo and DAG feed.
- RL1 road-lane interface price memo.
- M5R and V12 receipts for restricted-master/top24 and vocabulary-floor scope.
- PF3B durable merge/harvest feeds for real joint-improving but
  rate-dominated edge.
- NG1 negative verdict ledger, AU1 correction index, and probe-outcomes ledger
  as recall/triage surfaces only.

## Artifact Checkpoints

This directory contains:

- `RECEIPT.md`
- `INSTRUMENT_FANOUT.jsonl`
- `REOPEN_LEDGER.jsonl`
- `NEXT_IF_RESUMED.md`
- `CHECKPOINTS.md`

Validation required after file creation:

1. Parse both JSONL files.
2. Confirm no protected file diff was introduced by this arm.
3. Compute post-edit SHA-256 for each VO1 file.
4. Attempt serializer commit with repeated `--expected-content-sha256` entries
   and tags `[no-triality] [p0-ledger-ok]`.

## Boundary Checkpoint

VO1 is not a score-moving unit. It is a verdict-instrument routing audit. The
own-vehicle frontier remains unchanged at the current hot-state value:
`S = 0.7534578126155775 @ 357,837 B [macOS-CPU advisory]`.
