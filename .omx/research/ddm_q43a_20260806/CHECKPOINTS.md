# ddm_q43a checkpoints

## Checkpoint 0 - Intake

- Read charter `.omx/tmp/codex_runs/q43a_prompt.md`.
- Read common contract `.omx/tmp/codex_runs/_common_contract.md`.
- Read governing context: `PROGRAM.md`, `CLAUDE.md`, `AGENTS.md`,
  `docs/operating_manual_craft_handoff.md`, `.omx/state/main_hot_state.md`.
- Confirmed scorer slot ownership in charter, but no scorer job was launched.

## Checkpoint 1 - Freshness

- Correct parent receipt bytes:
  `/Volumes/VertigoDataTier/pact/ddm_tq1_20260805/phase_b_realized_tq1c/candidate_archives/move_0023_snap_r00_c12_L13.zip.receipt-bytes`
- SHA-256:
  `b35e756829306a85ec2ad51634bde74523d89df9046c682253176b393bd59c06`
- Size: `357837` bytes.
- Preserved aggregate exists with 38 batches and expected components:
  `d_seg=0.004305419922`, `d_pose=0.000716508925`, `S=0.7534578126155775`.
- This arm did not rerun the scorer.

## Checkpoint 2 - Invocation Recovery

- Recovered the su2 ready command from
  `.omx/research/ddm_su2_pose_endgame_program_20260730.md`.
- Confirmed CLI from argparse in `experiments/ddm_su2_qa43_tail_solver.py`.
- Confirmed validation/solve are subcommands and production top-k is fixed to
  `56,112,200`.
- Confirmed the only concrete factory is `create_v4d_warp_adapter`.

## Checkpoint 3 - Adapter Validation

Validation command used the correct tq1c receipt-bytes file and the su2 v4d
adapter. It failed scorer-free:

```text
QA43 REFUSED: v4d-warp parent member order/shape differs
```

Archive member evidence:

- tq1c: single stored member `0.bin`.
- v4d expected parent: `manifest.json`, `state/tokens.dr7t`,
  `state/renderer.sec`, `state/selector.sec`, `state/pose_stub.sec`,
  `state/pose_warp.stp`.

No solver state dir, pair checkpoints, stage archives, or candidate archives
were created.

## Checkpoint 4 - Receipt

Persisted:

- `.omx/research/ddm_q43a_20260806/RECEIPT.md`
- `.omx/research/ddm_q43a_20260806/NEXT_IF_RESUMED.md`
- `.omx/research/ddm_q43a_20260806/CHECKPOINTS.md`

No bulk artifacts were created. No cleanup was required.
