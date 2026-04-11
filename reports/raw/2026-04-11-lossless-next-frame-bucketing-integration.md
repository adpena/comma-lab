# 2026-04-11 lossless next-frame and bucketing integration

## landed

- `tac lossless next-frame-sample` now respects profile-driven `context_frames` when `--context-frames` is omitted.
- the grouped next-frame sample path remains explicitly local-only and exactness-aware.
- a deterministic bucketing core now exists for reversible corpus-order experiments:
  - token-only segment features
  - deterministic bucket assignment
  - exact reorder/restore plan

## verification

- `python3 -m unittest experiments.test_tac_lossless_next_frame_coder experiments.test_tac_lossless_bucketing experiments.test_tac_cli -v`
- `python3 -m src.comma_lab.cli lossless-review doctor --repo-root /Users/adpena/Projects/pact --json`

## current interpretation

- GPT arithmetic remains the strongest measured byte-level lane.
- grouped next-frame prediction is now a usable sample-only implementation lane rather than a scaffold.
- bucketing is now ready for token-derived corpus-order experiments and later segmentation-aware labels without changing decode semantics.
