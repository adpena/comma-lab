# Codex Findings - Z8 Argmax Hinge Archive-Path Confirmation

UTC: 2026-06-01T16:56:06Z

## Landing

The Z8 argmax-hinge confirmation harness now measures trained arms through the faithful top-LL-clamp-fixed Z8HPC1 archive receiver path by default:

- trained EMA arm render -> Z8HPC1 bytes -> `projected_pair_pyramids_from_archive_bytes` -> `reconstruct_pair_rgb_from_pyramid` -> scorer-grid resize;
- the harness pins the known faithful reference result and refuses to run if that reference is not unlocked;
- output defaults to the SSD waterfall and fails closed without explicit local-disk opt-in;
- retention manifests preserve output tree size, argv, reference hash, and false-authority markers;
- the training entrypoint can disable the joint variational driver cleanly instead of emitting nested readiness metadata that the canonical MLX harness rejects.

## Authority

This remains `[macOS-MLX research-signal]` only. It is a confirmation harness for Z8 SegNet objective selection, not a promotion path or exact-score claim.

## Verification

- `.venv/bin/ruff check experiments/train_substrate_z8_hierarchical_predictive_coding_mlx.py tools/z8_argmax_hinge_faithful_render_seg_confirm.py src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_z8_argmax_hinge_faithful_render_seg_confirm.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_z8_argmax_hinge_faithful_render_seg_confirm.py -q`

## Score-Lowering Relevance

This prevents a false positive or false negative from a collapsed/direct render path. Z8 rate work is only useful if the trained objective survives the same archive receiver path that will be byte-charged; this harness makes that distinction explicit before any exact-gate spend.
