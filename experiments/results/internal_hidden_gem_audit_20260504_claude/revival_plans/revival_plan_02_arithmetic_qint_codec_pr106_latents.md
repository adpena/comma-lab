# Revival plan: Lane PD-V3: arithmetic coder for PR106 latent + sidecar streams

**Gem**: `src/tac/arithmetic_qint_codec.py`
**ID**: `02_arithmetic_qint_codec_pr106_latents`

## Current state

Level-3 (measured in PD-V2 0.9974 sub-1.0 lineage). Shipped in `apogee` lineage already. PR106 latents currently brotli-coded at entropy 7.985/8 — near saturated.

## Files touched

- experiments/extract_pr106_latents.py (new): pull `latents_and_sidecar_brotli` (15849 bytes)
- src/tac/arithmetic_qint_codec.py (no changes)
- experiments/repack_pr106_with_arithmetic_latents.py (new)

## Integration sketch

1. Read PR106 0.bin → unpack via PR106's own inflate path.
2. Identify latent + sidecar segments (15849 bytes combined).
3. Build context model from first-order conditional histogram over quint values.
4. Re-encode via arithmetic_qint_codec.encode().
5. Repack 0.bin with same decoder bytes + new latent bytes.
6. Run inflate via PR106 adapter; verify byte-identical decoder output.

## Test plan

- Round-trip: arithmetic.encode(L) → arithmetic.decode(...) == L.
- Compression assert: encoded ≤ 15849 - 200 bytes (arbitrary win threshold for the marginal stream).
- End-to-end: archive.zip → inflate → eval on T4.

## Predicted score basis

Latents at entropy ~7.985/8 = brotli is near-Shannon. Arithmetic with stronger context model could shave 50-300 bytes. Score Δ ≈ -200 × 6.66e-7 = -1.3e-4 (basically nil unless context model is 1+ bit better per symbol).

## What would change my mind

If empirical compression < 1% improvement, arithmetic coder gain on latents is exhausted by brotli — focus on decoder bytes instead.

## Blockers resolved in plan

- (none)

## Skunkworks council deliberation

MacKay/Shannon: 'latent streams near 8 bits/symbol entropy means context model must capture >1 bit gain to matter'. Hotz: 'try it anyway — 3 hours, deterministic, can ship same-day'.

**Verdict**: VOTE 6/10 GO; 4 dissents (Shannon/MacKay/Ballé/Selfcomp say marginal). Recommended sequencing: AFTER water-fill v3 lands.
