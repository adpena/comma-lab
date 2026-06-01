# Codex Findings - PACT-NeRV Selector-V4 Compact Runner Execution

UTC: 2026-06-01T16:52:42Z

## Landing

Selector-V4 is now a first-class executable compact-base family in the MLX spine runner:

- plan rows route `pact_nerv_selector_v4` to MLX training, archive export, HPRC spine projection, receiver proof, and bounded-runner planning;
- the runner exposes selector palette controls and scorer-aware SegNet/PoseNet distillation knobs;
- the archive exporter writes archive-bound source bytes/hash, pair/frame coverage, selector codec metadata, and runtime-adapter `num_pairs`;
- execution remains false-authority until full-video MLX replay and contest CPU/CUDA exact gates exist.

## Smoke Artifact

One tiny real-video MLX execute/export smoke was run on the SSD tier:

- artifact root: `/Volumes/VertigoDataTier/pact/compact_selector_v4_1pair_1epoch_codex_20260601T1652`
- mode: `executed_pact_nerv_selector_v4_mlx_smoke_and_exported`
- archive bytes: `11015`
- archive sha256: `284a20b499de3995f62250996f7f6b12147ce34db7052e8e8e4503fa4636cd05`
- receiver proof: passed
- promotion blockers: partial one-pair coverage, full-video MLX replay missing, contest CPU/CUDA exact eval missing

This smoke proves the runner path materializes bytes and receiver consumption. It is not a score claim.

## Verification

- `.venv/bin/ruff check tools/run_compact_renderer_mlx_spine_runner.py src/tac/tests/test_compact_renderer_mlx_spine_runner.py src/tac/substrates/pact_nerv_selector_v4/archive_candidate.py src/tac/substrates/pact_nerv_selector_v4/tests/test_pact_nerv_selector_v4.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/tests/test_compact_renderer_mlx_spine_runner.py src/tac/substrates/pact_nerv_selector_v4/tests/test_pact_nerv_selector_v4.py -q`
- `PYTHONPATH=. .venv/bin/python tools/run_compact_renderer_mlx_spine_runner.py --execute-family pact_nerv_selector_v4 --output-dir /Volumes/VertigoDataTier/pact/compact_selector_v4_1pair_1epoch_codex_20260601T1652 --source-video-path upstream/videos/0.mkv --num-pairs 1 --epochs 1 --batch-pairs 1 --learning-rate 0.0001 --compact-latent-dim 4 --compact-embed-dim 4 --compact-selector-palette-size 4 --compact-decoder-channel 4 --compact-ema-decay 0.9 --overwrite`

## Next

Scale this from one-pair custody proof to full-coverage compact-base sweeps under `178k/216k/285k`, then attach full-video MLX scorer replay and section-level value-per-byte pricing before exact-gate spend.
