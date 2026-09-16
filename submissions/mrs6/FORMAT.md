# Archive format

`archive.zip` holds one member, `p`, of 179,186 bytes. Nothing is stored anywhere else. Sizes below are this
archive's measured sizes; `code font` names are the names in the source.

## Header — 14 bytes, `PAYLOAD_HEADER = '<4sBBBBHHH'`

| field | bytes | here | meaning |
|---|---|---|---|
| magic | 0-3 | `RX1M` | format tag |
| version | 4 | 1 | format version |
| codec | 5 | 2 | 1 selects xz for the prior stream; any other value selects Brotli |
| table_mode | 6 | 0 | reserved; the decoder does not read it |
| flags | 7 | 250 | the flag byte below |
| prior_bytes | 8-9 | 11,629 | stored length of the prior stream |
| renderer_bytes | 10-11 | 29,862 | stored length of the renderer stream |
| carrier_bytes | 12-13 | 18,483 | stored length of the carrier stream |

### The flag byte — `HeaderFlags`, all five bits set here

`RENDERER_PLANES` (2) and `PRIOR_ADAPTIVE` (64): that inflated stream is two byte planes and `join_byte_planes`
interleaves them. `ARITHMETIC_BASIS` (8) and `ARITHMETIC_COEFFICIENTS` (16): the carrier is arithmetic-coded;
the decoder refuses if either is clear. `GEOMETRY_MIXER` (128): a geometry-mixer block sits in front of the
token stream. Bits 1, 4 and 32 are not read; bit 32 is set here and has no effect.

## The four sections, in order

1. **Prior stream**, 11,629 bytes stored, 16,249 after Brotli. A context-mixing bit coder
   (`restore_prior_stream`) rewrites it, then `load_prior_weights` fills the prior: a small integer convolutional
   network over 64x64 patches of the token plane, stored as int8 weight codes with one fp16 scale per row and a
   per-channel power-of-two exponent.
2. **Renderer stream**, 29,862 bytes stored, 31,451 after Brotli. `restore_renderer_stream`
   rewrites it and `unpack_renderer_weights` loads the token renderer: quantized rows at a per-row bit depth, fp16
   scales, and three FiLM weights stored row-pruned (`ROW_PRUNE_NAMES`).
3. **Carrier stream**, 18,483 bytes stored, 18,522 after Brotli: 142 bytes of metadata (fp32
   basis scales, fp32 trajectory scales, AR(1) factors and biases, Rice parameters), 12,046 bytes of
   arithmetic-coded basis codes, 6,305 bytes of CABAC-coded trajectory, and a 29-byte selector body whose 5-byte
   header the decoder supplies (`SELECTOR_PREFIX`). The basis is 12 x 3 x 24 x 32 int8 codes over a 32-symbol
   alphabet; the trajectory is 600 x 12 signed 12-bit values, each predicted from the previous pair and corrected
   by the coded residual. The selector names, by combination rank, which pairs get one of the eight first-frame
   pixel edits in `PIXEL_MODES`.
4. **Token section**, 119,198 bytes: a 96-byte boundary table, a 64-byte geometry-mixer block,
   and 119,038 bytes of arithmetic-coded tokens.

## Boundary table — 96 bytes

One fp16 scale and 25 x 5 signed 6-bit codes. The state index is `boundary * 5 + predicted`, pairing the pixel's
distance from a class boundary in the previous plane, capped at 4, with the prior's own argmax. That state's
five values are added to the prior's logits before coding.

## Geometry-mixer block — 64 bytes

Bytes 0-3 are not read. The low nibble of byte 4 is the variant. Bytes 5-44 are 40 int8 weights: 35 for the
context mixer and 5 for the geometry term. Bytes 45-63 are the geometry configuration, `GEOMETRY_FORMAT =
'<BHH8B6B'`: the tracked class, the row band, then a row window, a current-frame weight, a minimum count, a
residual limit, a slope limit, a radius floor, a radius scale, a width limit, and six increasing distance edges.
`geometry.c` fits a line to that class inside 64-pixel slots with exact two-limb integer arithmetic and returns
one of nine bins: 0-6 by distance, 7 when no fit is valid, 8 outside the row band.

## Token stream

The plane is 384 x 512 over a 5-symbol alphabet, decoded in 190 groups. Group `g` holds the pixels
with `column % 64 + 2 * (row % 64) == g`, one from each 64x64 patch. Causality is a flag, not an ordering: `corrector.c`
keeps a `known` bit per pixel and reads a neighbour only when it is set, so a pixel across a patch edge is
skipped until its own group arrives. Per group the decoder takes the prior's logits, adds the boundary table,
turns them into float32 probabilities, corrects them with `corrector.c`, mixes in the geometry term, and decodes
the group with `range_decoder.c`: 63-bit low, high and code registers, a 2^31 frequency total, five frequencies
per pixel each at least 1, the largest taking the rounding slack.

`corrector.c` holds 23 context families (`FAMILY_RULE`, `FAMILY_SIZE`, `FAMILY_COUNT_LIMIT`): one joint family
over class, surprise, two temporal agreements, run length and boundary; two count-limited copies of it; and
twenty smaller families over subsets of those plus 4-neighbour and 8-neighbour spatial agreement, local
homogeneity, group bin, 192-patch and 48-tile identity. A fixed-point logistic mixer with 4,000 weight sets
blends them, and a miss model over the three causal neighbours and the previous plane rescales the losing
classes.

## `0.raw`

1,200 frames of 874 x 1,164 RGB uint8 in pair order — pair 0 frame 0, pair 0 frame 1, pair 1 frame 0, and so on.
3,662,409,600 bytes.
