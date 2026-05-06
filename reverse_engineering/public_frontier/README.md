# Public Frontier Reverse Engineering

This tree holds curated, source-sized reference material recovered from public
submission deconstruction. It is not an active experiment output directory and
it is not a score ledger.

Use this material to inspect public runtime grammars, archive layouts, and
codec ideas, then promote reusable implementation into `src/tac/` or thin
operator wrappers into `experiments/`. Keep raw archives, cloned repos,
provider logs, checkpoints, videos, and generated result directories outside
this tree.

Evidence boundary:

- Code here is `external` or forensic reference evidence.
- Public replay through a contestant runtime is not our score truth.
- Any score-bearing claim still requires our exact archive bytes through
  `archive.zip -> inflate.sh -> upstream/evaluate.py` on CUDA.

## Recovered Runtime Sources

`recovered_runtime/experiments_results_20260505_pyc_recovery/` contains small
Python source and `.recovery_spec.json` files from the 2026-05-05 orphan-pyc
recovery pass. The original path context is preserved below that directory so
PR/runtime provenance remains visible:

- `leaderboard_intel_20260504_codex/`: PR96/PR98/PR99 HNeRV-class runtime
  fragments.
- `public_pr95_intake_20260504_codex/`: PR95 HNeRV reference runtime pieces.
- `public_pr90_intake_20260504_worker/`: qrepro codec/range/sparse reference
  scripts.
- `public_leaderboard_inflate_adapters_20260502T0630Z/`: older public replay
  adapters.
- `public_pr85_intake_20260503_codex/`, `public_pr86_intake_20260504_codex/`,
  `public_pr91_intake_20260504_codex/`, and
  `pr85_stbm1br_mask_recode_20260504_worker/`: mask/replay runtime fragments.
- `renderer_selfcompression_nextwave_worker_20260503/`: local recovered
  self-compression search artifact retained as historical hidden-gem context.
- `top_submission_reverse_engineering_20260503_deep_codex/`: PR75-head
  reference runtime fragment.

Do not edit these reference files to make them "clean" unless promoting them
out of this forensic tree. If a bug or useful primitive is found here, add a
small canonical implementation or test elsewhere and cite the source path in
the relevant `.omx/research/` ledger entry.
