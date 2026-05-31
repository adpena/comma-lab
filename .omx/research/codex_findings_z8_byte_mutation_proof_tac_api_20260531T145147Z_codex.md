# Codex Findings: Z8 Byte-Mutation Proof As TAC API

UTC: 2026-05-31T14:51:47Z

## Verdict

The Z8 predictive-stack archive export now classifies section consumption with an executable byte-mutation proof by default. Mamba/Dreamer/Wyner-Ziv remain honest custody-only surfaces until their bytes are proven pixel-consuming.

## What changed

- Moved the Z8HPC1 byte-mutation proof from a standalone tool into `tac.substrates.z8_hierarchical_predictive_coding.byte_mutation_proof`.
- Kept `tools/probe_z8_archive_distinguishing_feature_byte_mutation.py` as a thin CLI wrapper.
- `export_z8hpc1_archive_bytes(...)` now emits `z8_hpc1_byte_mutation_proof.json` by default and fails closed if `wavelet_blob` is not pixel-consuming.
- Archive-bound runtime metadata now includes `byte_mutation_consumption_proof` with pixel-consumed sections, custody-only sections, proof path, and the explicit `mamba_dreamer_wyner_ziv_pixel_consumption_proven=false` boundary.
- Candidate row input artifacts include the byte-mutation proof so the receiver/runtime package carries the section-consumption evidence with the archive.

## Verification

- `.venv/bin/ruff check src/tac/substrates/z8_hierarchical_predictive_coding/byte_mutation_proof.py tools/probe_z8_archive_distinguishing_feature_byte_mutation.py src/tac/substrates/z8_hierarchical_predictive_coding/archive_candidate.py src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_archive_candidate_bridge.py`
- `.venv/bin/python -m pytest src/tac/substrates/z8_hierarchical_predictive_coding/tests/test_archive_candidate_bridge.py src/tac/tests/test_train_substrate_z8_canonical_quadruple_binding.py -q`
