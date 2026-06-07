# Codex Findings: HiNeRV Tile-Brotli Action Interpreter

UTC: 2026-06-07T14:54:50Z

## Landed

- Added `tile_brotli_v1` to the charged HiNeRV target-region action interpreter.
- The codec stores support as Brotli-compressed 16x16 tile bitmaps and RGB as a
  separate Brotli stream.
- Decode reconstructs support in canonical row-major order, so the existing
  action/support identity is preserved when the input action is canonical.
- Noncanonical or duplicate support falls back to older receiver-decoded codecs;
  the interpreter never silently permutes RGB values.
- `build_hi_nerv_target_region_action_parseback_survival` now emits
  `target_region_action_program_sha256` directly, which makes standalone
  repack receipts source-bound without depending on runner wrapper glue.

## Real Artifact

Path:
`/Volumes/VertigoDataTier/pact/experiments/results/hinerv_witness_readiness_short_smoke_current_main_20260607Tcodex_v3_export_wall_normal_support/hi_nerv_mlx_training/target_region_action_tile_brotli_repack`

- action: `df6f7301995ee1ac60f84637beed9b390826b32c62521f1a5446680b8785c7a2`
- support: `d56a75511244d1cb71bfaca9ddff67513ab80c452e41e088b0a204897a2ced0c`
- archive: `763f8af4b23c021f20e69eff5da9e3958274f155b1f6f16903b21915a2d19cbd`
- archive bytes: `319575`
- payload codec: `tile_brotli_v1`
- payload bytes: `94633`
- payload delta vs split-brotli: `-28526`
- archive delta vs split-brotli: `-13273`
- parseback survived: `true`
- inflate survived: `true`
- survival blockers: `[]`
- lowering verdict: `best_lowering=byte_priced_sidecar`, `first_failing_surface=none`

This remains non-promotable and has no score claim. It is a byte-closed
interpreter-side improvement to the same proven sidecar mechanism; long-run
approval still waits on backend realization, full-video replay, and exact replay.

## Validation

- `PYTHONDONTWRITEBYTECODE=1 uv run pytest src/tac/substrates/hi_nerv/tests/test_hi_nerv_roundtrip.py -k "target_region_action_payload" -q` -> 4 passed
- `PYTHONDONTWRITEBYTECODE=1 uv run pytest src/tac/substrates/hi_nerv/tests/test_receiver_cache_quality.py -k "target_region_action_parseback_survival or tile_brotli" -q` -> 3 passed
- `uv run ruff check src/tac/substrates/hi_nerv/target_region_actions.py src/tac/substrates/hi_nerv/archive_candidate.py src/tac/substrates/hi_nerv/tests/test_hi_nerv_roundtrip.py src/tac/substrates/hi_nerv/tests/test_receiver_cache_quality.py` -> clean
