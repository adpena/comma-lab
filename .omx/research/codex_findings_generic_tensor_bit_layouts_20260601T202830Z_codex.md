# Generic Tensor Int Payload Bit Layouts Landed

Date: 2026-06-01T20:28:30Z
Author: Codex

## Verdict

The generic `tensor_payload_grammar_optimizer.v1` now treats lossless integer
payload layout as a first-class pre-entropy-coder choice. This is the safe
layer for bit-shift / packing ideas: it changes how already-quantized integer
bytes are presented to Brotli/Huffman/range-style coders, but it does not change
the represented tensor values and it never embeds hidden length authority.

Implemented layouts:

- `flat`: existing byte order.
- `nibble_planes`: high nibbles and low nibbles packed into separate streams.
- `bitplanes_lsb`: eight little-endian bitplanes packed with `np.packbits`.

The decoder side requires the original raw byte length from the tensor or
section manifest. That is intentional: payload layout is an archive grammar
decision, not a self-describing side channel.

## Live PR101 Proof

Source:
`/Volumes/VertigoDataTier/pact/pr101_real_grouped_runtime_campaign_20260601T172357Z/source_decoder_state_dict.pt`

Artifacts:

- `/Volumes/VertigoDataTier/pact/tensor_payload_bit_layouts_20260601T202205Z/pr101_tensor_payload_bit_layout_report.json`
  - SHA-256: `c83a2f11870f88becc2be53963dea10b357ce474736f4dddf34428cbed212997`
- `/Volumes/VertigoDataTier/pact/tensor_payload_bit_layouts_20260601T202205Z/pr101_tensor_payload_bit_layout_queue.json`
  - SHA-256: `a3dd46813ce3088ce4a7a8d0cf95a34582fe032ca1f7bf5fc239a051dd0a2a05`
- `/Volumes/VertigoDataTier/pact/tensor_payload_bit_layouts_20260601T202205Z/pr101_tensor_payload_bit_layout_report_consumer_result.json`
  - SHA-256: `c60510dce6b02a079d32327f47c430c61dc185b8cc9e87425a37422cdcdcbf49`

Measured result:

- tensors: 28
- selected isolated tensor bytes: 162,223
- baseline isolated tensor bytes: 162,273
- selected savings versus baseline: 50 bytes
- selected over empirical floor: 1.0146715678015736
- saturation status: `entropy_saturated`
- selected layouts: `flat` for all 28 tensors
- `score_claim`: false
- `ready_for_exact_eval_dispatch`: false

Consumer verdict:

- `planner_action`: `record_tensor_payload_saturation_and_demote_format_churn`
- `receiver_work_justified`: false
- `demotion_recommended`: true

This is a useful negative for PR101/fec6-class decoder weights: the best
lossless bit/nibble/plane presentation is still the existing flat integer
stream. That does not invalidate the mechanism. It says the current competitive
decoder-weight payload is already near its empirical entropy floor, so repeated
same-substrate lossless layout churn should be demoted automatically.

## Representation Implications

There are more optimal numeric formats, but they are data-regime dependent:

- Native low-bit integers (`int2`, `int4`, `int6`, `int8`) with per-block or
  per-channel scales are lossy unless the substrate is trained or projected onto
  that grid. They belong in scorer-aware QAT / hard-projection / replay loops,
  not this lossless layout layer.
- Nonuniform codebooks (`NF4`, Lloyd-Max, learned VQ, power-of-two/log quant)
  are promising when the value distribution is clustered or can be trained to
  cluster. They need receiver-bound codebook manifests and full replay.
- Sparse/procedural forms (`zero/RLE`, top-k residuals, low-rank factors,
  generated filters) are promising when the tensor or coefficient field has a
  real low-description-length structure. Z8 wavelet detail coefficients are the
  canonical current example; PR101 decoder weights are not.
- Split representations (`sign/magnitude`, exponent/mantissa, bitplanes,
  delta/predictor residuals) should remain available as pre-coder transforms,
  but queue promotion should require measured positive savings over the already
  selected isolated or grouped grammar.
- Rational/string/expression constants are viable for constants, tables,
  generators, codebook seeds, coordinate functions, and procedural receivers.
  They are usually worse for dense learned weights unless training constrains
  values to a small algebraic dictionary.

The important distinction: strings can represent numbers, and fractions can
represent an order of operations, but the winning object is the shortest
deterministic program that reconstructs the required values under the receiver
contract. Decimal strings for dense high-entropy weights are usually longer
than integer/codebook payloads. An expression like `sin(pi*x/16)`, `3/255`,
`sqrt(2)/4`, or a small polynomial/codebook seed can win only when many values
are generated from that expression or when training explicitly forces weights
onto that algebraic/codebook manifold.

## System Wiring

The optimizer report now records `int_payload_layouts`, every candidate row
records `int_payload_layout`, and planner operation hints include the layout.
`build_tensor_payload_optimizer_queue(...)` carries the selected layout into
the candidate id and operation params. The grouped Brotli diagnostic uses the
selected transformed payload including the selected bit layout, so grouped
window-order learning cannot silently ignore this dimension.

Tests cover:

- exact layout encode/decode round-trip across empty, odd, even, and non-byte
  multiple lengths;
- wrong-length and unknown-layout refusal;
- optimizer candidate enumeration over all layouts;
- CLI `--payload-layouts` parsing;
- planner feedback carrying `int_payload_layout`.

## Consequence

For the current PR101/fec6 frontier, lossless representation layout is
saturated. The next score-lowering representation work must alter the value
distribution itself:

1. scorer-aware low-bit/codebook QAT (`int2/int4/NF4/VQ`) with full-video
   P18/P19 water-fill and hard replay;
2. substrate-specific procedural/rational expression grammars for constants,
   tables, generated filters, and compact receivers;
3. wavelet/detail/residual lanes where sparsity or low-rank structure is
   already present and the entropy gap is measured, not inferred.

This landing keeps the attractive bit-packing branch alive as an automatic
diagnostic while demoting it for the current saturated decoder-weight payload.
