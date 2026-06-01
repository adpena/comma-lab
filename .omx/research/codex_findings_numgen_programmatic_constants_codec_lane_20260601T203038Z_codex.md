# Numgen / Programmatic Constants Codec Lane - Codex Findings

Timestamp: 2026-06-01T20:30:38Z

## Verdict

Numgen is a real candidate compression family, but only as charged archive
bytes plus deterministic receiver code. It is not a loophole. A string, integer,
fraction table, seed, generator program, symbolic expression, or codegen kernel
can replace tensor bytes only when:

- the generated tensor/constants are fully determined by bytes inside
  `archive.zip` and the vendored runtime;
- the receiver remains decode-only and imports no scorer/network authority;
- the generated object has lower description length than the tensor stream it
  replaces;
- full-video SegNet/PoseNet replay proves the generated approximation is worth
  its charged bytes under `delta_nonrate + 25*delta_archive_bytes/N < 0`.

## Where It Fits

The immediate codec portfolio now covers explicit numeric streams:

- decoder weights: `fp16_brotli_legacy`, `int8_mixed`, `int4_mixed`,
  `int2_mixed`;
- selector/index streams: raw u32/u16, varints, delta varints, zero-run
  varints, fixed bitpacking, packed bitmasks.

Numgen should sit beside these as a third portfolio class:

- exact constants as rational strings or numerator/denominator integer streams;
- generated sinusoidal / polynomial / spline / Chebyshev basis constants;
- seeded low-rank tensor generators with charged seed, shape, scale, basis
  family, and residual;
- small symbolic programs for deterministic road/sky/lane or receiver priors;
- codebook generators when table bytes exceed program plus correction bytes.

## Non-Negotiable Guardrails

- No hidden dictionaries, no dependency fetch, no uncharged pretrained weights.
- No receiver-side scorer inspection or adaptation.
- No score claim from MLX/proxy rows.
- No use for dense learned weights unless the learned tensor is actually
  low-description-length under the generator family.
- String-as-number tricks are only byte-layout tricks unless they expose real
  structure; they must be compared against normal entropy coding.

## Minimal Prototype

Add a future shared codec:

- `src/tac/substrates/_shared/numgen_codec.py`
- `NumgenRecord(kind, shape, dtype, params_blob, residual_codec)`
- deterministic `materialize()` returning numpy/torch tensors at inflate time
- section stats with generator bytes, residual bytes, and false-authority flags
- tests proving byte-stable materialization and old-packet compatibility

First scorer-relevant target: generated receiver constants and small codebooks,
not full decoder weights. Full decoder-weight generation should only enter after
a byte-value profile shows a tensor family has low-rank/smooth/symbolic
structure that int8/int4 plus entropy coding does not already capture.

## Next Action

Let the numgen research subagent rank SOTA methods and OSS implementations. The
mainline implementation should keep using scorer-priced water-filling to choose
among explicit tensor codecs, integer-stream codecs, zero/drop modes, and future
numgen records per section/atom.
