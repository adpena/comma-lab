# Magic Codec Pair #4 Procedural Seed Orthogonality Smoke Landed

**Author**: codex  
**UTC**: 2026-05-21T00:40:54Z  
**Lane**: `lane_wave_3_magic_codec_pair_4_procedural_seed_orthogonality_smoke_20260521`  
**Source memo**: `.omx/research/magic_codec_x_todays_cascade_stacking_analysis_20260520.md`  
**Artifact**: `experiments/results/magic_codec_pair_4_procedural_seed_orthogonality_smoke_20260521T010000Z/smoke_result.json`

## Verdict

`PAIR_4_BOUNDARY_VALIDATED_RAW_SEED_DOMINATES`

Pair #4's null hypothesis is validated: procedural-codebook seed bytes should
stay raw. Magic-codec remains a candidate for residual streams, but wrapping the
seed itself is a rate regression.

This is not a score claim and is not promotion evidence. It is a local
byte-budget boundary check for cascade routing.

## Empirical result

The smoke tested canonical high-entropy seed lengths `16`, `32`, `64`, `128`,
and `256` bytes across six reversible orderings:

- `identity`
- `reverse`
- `even_then_odd`
- `odd_then_even`
- `adjacent_pair_swap`
- `rotate_left_half`

All `30 / 30` canonical reversible seed/order rows selected `raw_seed` as the
smallest compliant representation.

The best non-raw wrapper was always `brotli_q11_seed_bytes`, and the minimum
best-nonraw delta was `+4` bytes versus raw. This is exactly the expected
small-stream compression envelope overhead for high-entropy PRNG-state bytes.

## Ordering dimension

The smoke includes the ordering dimension explicitly. Reversible, decoder-free
permutations are verdict-eligible. Value-dependent sorted orders are emitted as
non-free controls and excluded from the verdict because an inverse permutation
or altered seed semantic would need to be stored.

The non-free controls behave as expected: sorted `256`-byte high-entropy seed
bytes become compressible under brotli, but that apparent saving is not a free
contest representation.

## Other dimensions covered

Codec candidates:

- raw seed bytes
- brotli q11
- lzma preset 9 extreme
- `magic_codec_classic` over `latent_sidecar`, `residual_basis`, `categorical`,
  `weight_tensor`, and `mask` hints
- `magic_codec_dense_streams` with `smallest_byte_count`, `brotli_only`,
  `lzma_only`, and `magic_classic_only`

Structured low-entropy controls (`all_zero`, `alternating_00_ff`,
`ascending_mod_256`) were included to prove the smoke detects compressible
non-seed data. They are not pair #4 evidence because they are not canonical
procedural-codebook seed priors.

## Routing consequence

Pair #4 is closed as a paid-eval candidate. The correct integration rule is:

> store procedural-codebook seed bytes raw; route magic-codec only to residual
> streams where there is an empirical byte stream to compress.

After pair #1 and pair #2 falsified residual rescue paths on their tested
surfaces, and pair #4 validates the raw-seed boundary, the next useful
frontier-moving work is DP1 procedural paired-smoke wiring, LL scorer-surrogate
frame/pair planner wiring, or a new residual-hybrid equation/tooling surface
whose predicate matches predictor-plus-residual byte accounting.

## Verification

Commands run:

```bash
.venv/bin/python tools/run_magic_codec_pair_4_procedural_seed_orthogonality_smoke.py --output-dir experiments/results/magic_codec_pair_4_procedural_seed_orthogonality_smoke_20260521T010000Z
.venv/bin/python -m pytest -q src/tac/tests/test_magic_codec_pair_4_procedural_seed_orthogonality_smoke.py src/tac/tests/test_packet_compiler_magic_codec.py src/tac/tests/test_packet_compiler_magic_codec_dense_streams.py
```

Result:

- smoke verdict: `PAIR_4_BOUNDARY_VALIDATED_RAW_SEED_DOMINATES`
- canonical reversible seed/order rows: `30`
- rows where raw seed dominates: `30`
- minimum best-nonraw delta versus raw: `+4` bytes
- tests: `97 passed`

