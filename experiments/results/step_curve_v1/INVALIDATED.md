# INVALIDATED -- Wrong Checkpoint

These results were produced using a 5-epoch smoke-test model (MD5: a9aee326),
NOT the auth=0.87 renderer (MD5: cff8dca4). All absolute numbers are meaningless.

Qualitative findings (phase transition at ~100 steps for PoseNet)
likely hold but need re-validation with the correct checkpoint.

Re-run needed on Vast.ai with the correct checkpoint.

## Discovery

2026-04-15: Discovered that `experiments/results/fridrich_renderer/renderer_best.pt`
had been overwritten with a 5-epoch smoke model at some point. All Vast.ai
experiments that uploaded this checkpoint ran with the wrong model.

## Correct checkpoint

- Path: `experiments/results/v5_lagrangian_renderer/renderer_best.pt`
- MD5 prefix: `cff8dca4`
- Auth score: 0.87
