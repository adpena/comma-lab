# Codex Session Summary - LF/HF Replacement Queue Integrated - 2026-06-05T13:09:00Z

## Landed

- Added `tac.analysis.snerv_lf_hf_replacement_queue`.
- Added `tools/build_snerv_lf_hf_replacement_queue.py`.
- Wired `build_nerv_long_training_campaign_plan(...)` to emit
  `snerv_lf_hf_replacement_queue` beside the existing LF over-ceiling reroute
  queue.
- Added `--output-snerv-lf-hf-replacement-queue` and overwrite SHA support to
  `tools/build_nerv_long_training_campaign_plan.py`.

## Expanded Solution Space

The queue now emits seven learned LF/HF replacement families:

- `official_tub_lf_hf_decoder_replacement`
- `lf_conditioned_hf_residual_generator`
- `joint_lf_hf_factorized_codebook`
- `temporal_lf_predictor_gate`
- `lf_super_resolution_from_tiny_anchor`
- `score_tethered_spectral_band_allocator`
- `entropy_modeled_lf_latent_hyperprior`

Each family is separated into its own queue row with blockers and target
consumers. Rows only receive a command when prerequisite implementation,
source-forward, receiver, renderer, and value-domain blockers are absent.

## Current SSD Handoff

- JSON: `/Volumes/VertigoDataTier/pact/snerv_lf_hf_replacement_queue_integrated_20260605Tcodex/snerv_lf_hf_replacement_queue.json`
- Markdown: `/Volumes/VertigoDataTier/pact/snerv_lf_hf_replacement_queue_integrated_20260605Tcodex/snerv_lf_hf_replacement_queue.md`
- JSON SHA-256: `05ec5fb47314f1f0c522628ce10d4766e8b97c343ba00273e0228a3678f5b151`
- Markdown SHA-256: `ed8034e898b2bde8540ec193147529c1fe37fc6f00d22dce524641fc392bf3f3`
- Queue rows: `21`
- Runnable local rows: `0`
- Selected LF evidence: `666556` bytes from measured LF payload reports.

## Verdict

Current blocker remains correct and intentional:

- `snerv_lf_hf_current_snar2_queue_has_no_lf_over_ceiling_rows`

The freshest SNAR2 terminal-tether LF reroute queue has `0` rows, so measured
historical LF dominance is acquisition signal, not launch authority. The next
byte-lowering move must first clear source-forward official MFU/HFR/TUB,
receiver payload, scorer-input distribution, and non-collapse value-domain
blockers before any bounded learned LF/HF replacement smoke is runnable.

## Verification

- `uv run pytest src/tac/tests/test_snerv_lf_hf_replacement_queue.py src/tac/tests/test_nerv_long_training_campaign_plan.py::test_long_training_campaign_plan_builds_optimizer_matrix src/tac/tests/test_nerv_long_training_campaign_plan.py::test_build_long_training_campaign_plan_cli_writes_outputs -q` -> 5 passed.
- `uv run ruff check src/tac/analysis/snerv_lf_hf_replacement_queue.py src/tac/analysis/nerv_long_training_campaign_plan.py tools/build_snerv_lf_hf_replacement_queue.py tools/build_nerv_long_training_campaign_plan.py src/tac/tests/test_snerv_lf_hf_replacement_queue.py src/tac/tests/test_nerv_long_training_campaign_plan.py` -> passed.
