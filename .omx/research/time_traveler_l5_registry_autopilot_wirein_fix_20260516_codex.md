# Time-Traveler L5 Registry And Autopilot Wire-In Fix - 2026-05-16

## Scope

Adversarial review found that Time-Traveler L5 was implemented as a trainer and
archive grammar, but was not visible through the importable substrate package,
the canonical composition inventory, or the Cathedral dispatch ranker. This is
an integration bug, not a score result.

## Fix

- Added package-level
  `src/tac/substrates/time_traveler_l5_autonomy/registered_substrate.py`.
- Exported `TIME_TRAVELER_L5_AUTONOMY_SUBSTRATE_CONTRACT` from the package
  `__init__`, so importing `tac.substrates.time_traveler_l5_autonomy` registers
  the contract for auto-wire consumers.
- Replaced the stale trainer-local contract with the package contract.
- Corrected contract grammar from stale `TTL5V1` prose to the actual `TT5L`
  archive magic and parser sections:
  `tt5l_header`, `world_model_blob`, `per_pair_side_info_blob`,
  `ac_state_blob`, and `meta_blob`.
- Added `time_traveler_l5_autonomy` to
  `tac.optimization.substrate_composition_matrix` with explicit blockers for
  byte-closed temporal side-info proof, paired CPU/CUDA axis custody, and the
  missing C1/Z5/TT5L probe-disambiguator.

## Evidence

Focused checks:

- `.venv/bin/python -m pytest src/tac/substrates/time_traveler_l5_autonomy/tests/test_registered_substrate.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_substrate_registry.py -q`
- `.venv/bin/python -m py_compile src/tac/substrates/time_traveler_l5_autonomy/registered_substrate.py experiments/train_substrate_time_traveler_l5_autonomy.py src/tac/optimization/substrate_composition_matrix.py`

Integrated verification pass:

- `.venv/bin/python -m pytest src/tac/tests/test_build_pr101_finetuned_archive_codec_dir.py src/tac/tests/test_modal_call_id_ledger.py src/tac/tests/test_modal_training_harvest_summary.py src/tac/substrates/time_traveler_l5_autonomy/tests/test_registered_substrate.py src/tac/tests/test_substrate_registry.py -q`
- Result: `77 passed`.

## Status

TT5L is now visible to registry, inventory, and ranker planning surfaces. It is
still not promotion-ready. Required next evidence:

1. byte-closed temporal side-info consumption proof through `inflate.sh`;
2. paired contest CPU/CUDA exact-eval axis metadata on the same archive/runtime;
3. callable C1/Z5/TT5L probe-disambiguator before architecture lock;
4. cost-band and posterior/correction anchor propagation after any valid exact
   eval.
