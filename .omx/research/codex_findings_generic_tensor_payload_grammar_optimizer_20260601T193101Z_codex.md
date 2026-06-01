# Generic tensor payload grammar optimizer landed

## Context

The PR101/fec6 grammar campaign proved the tuned frontier packet is already
near the section/tensor entropy floor, but that machinery was still mostly
PR101 fixed-schema specific. The missing reusable surface for future substrates
was a generic tensor payload gate: given any MLX/PyTorch/NumPy-exported tensor
set, quantize to an explicit integer payload, price byte-map/storage/coder
choices, emit entropy-gap diagnostics, and feed the byte-shaving planner without
claiming archive or score authority.

## Landed surfaces

- `src/tac/packet_compiler/tensor_payload_grammar_optimizer.py`
  - `quantize_tensor_symmetric_int8(...)`
  - `measure_tensor_payload_candidates(...)`
  - `solve_tensor_payload_grammar(...)`
  - `build_tensor_payload_optimizer_queue(...)`
- `tools/tensor_payload_grammar_optimizer.py`
  - `--npz`
  - `--torch-state-dict`
  - `--storage-perm-mode {identity,identity-plus-exhaustive4}`
  - `--coders`
  - `--queue-output`
- `src/tac/tests/test_tensor_payload_grammar_optimizer.py`
  - finite-value quantization guard
  - transform/coder roundtrip proof
  - queue/signal-surface consumption proof
  - CLI NPZ proof

The implementation reuses the existing PR101 byte-map codec and payload-coder
portfolio. It deliberately rewrites runtime status to
`generic_tensor_payload_receiver_required` so generic tensors cannot inherit
PR101 stock-runtime authority by accident.

## Real artifacts

Artifact root:

`/Volumes/VertigoDataTier/pact/tensor_payload_grammar_real_artifacts_20260601T192954Z`

PR101 source decoder state dict:

- input:
  `/Volumes/VertigoDataTier/pact/pr101_real_grouped_runtime_campaign_20260601T172357Z/source_decoder_state_dict.pt`
- report:
  `pr101_generic_tensor_payload_report.json`
- queue:
  `pr101_generic_tensor_payload_queue.json`
- tensor count: `28`
- selected isolated tensor bytes: `162234`
- baseline isolated tensor bytes: `162273`
- selected saved bytes vs baseline: `39`
- selected over Shannon floor ratio: `1.0147403705437608`
- saturation: `entropy_saturated`
- selected coders: all `brotli`

PACT-NeRV selector export, identity storage:

- input:
  `experiments/results/pact_nerv_selector_v4_mlx_full_layoutfix_20260528Tlocal/pytorch_state_dict.pt`
- report:
  `pact_nerv_generic_tensor_payload_report.json`
- queue:
  `pact_nerv_generic_tensor_payload_queue.json`
- tensor count: `12` prefix smoke
- selected isolated tensor bytes: `23006`
- baseline isolated tensor bytes: `23050`
- selected saved bytes vs baseline: `44`
- selected over Shannon floor ratio: `1.0153455614305593`
- saturation: `entropy_saturated`
- selected coders: all `brotli`

PACT-NeRV selector export, exhaustive 4D storage-permutation pass:

- report:
  `pact_nerv_generic_tensor_payload_exhaustive4_report.json`
- queue:
  `pact_nerv_generic_tensor_payload_exhaustive4_queue.json`
- tensor count: `12` prefix smoke
- selected isolated tensor bytes: `23000`
- baseline isolated tensor bytes: `23050`
- selected saved bytes vs baseline: `50`
- selected over Shannon floor ratio: `1.0150989305149727`
- saturation: `entropy_saturated`
- non-identity winners used `0,1,3,2` on several 4D tensors

## Verdict

This closes the reusable per-tensor grammar gap for future substrate exports:
byte-map, storage permutation, scale-tail dtype, and coder choice now have a
generic, fail-closed optimizer surface that emits queue-consumable planning
signal. The current tested PR101/PACT-NeRV tensor payloads remain effectively
saturated, so this is not a direct score-lowering claim. Its value is that new
MLX-trained HPRC/HNeRV/NeRV/non-NeRV candidates can be rejected or routed by an
automated rate gate before local replay or exact auth.

The result is consistent with the prior section-level finding: grammar payoff is
substrate-conditional. Tuned HNeRV-style integer payloads show tens of isolated
bytes, not frontier movement. The unsaturated class remains substrates that are
still transmitting structurally bad payloads, such as raw float wavelet/detail
streams or unoptimized sidecars.

## Verification

- `uv run ruff check src/tac/packet_compiler/tensor_payload_grammar_optimizer.py tools/tensor_payload_grammar_optimizer.py src/tac/tests/test_tensor_payload_grammar_optimizer.py src/tac/packet_compiler/section_payload_grammar_optimizer.py tools/section_payload_grammar_optimizer.py src/tac/tests/test_section_payload_grammar_optimizer.py src/tac/packet_compiler/pr101_per_tensor_grammar_solver.py tools/pr101_per_tensor_grammar_solver.py src/tac/tests/test_pr101_per_tensor_grammar_solver.py`
- `uv run pytest src/tac/tests/test_tensor_payload_grammar_optimizer.py src/tac/tests/test_section_payload_grammar_optimizer.py src/tac/tests/test_pr101_per_tensor_grammar_solver.py -q`

