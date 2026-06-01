# Selector-V4 Section Neutralization - Codex Findings 2026-06-01

## Verdict

Landed parse-preserving PSV4 section neutralization so the selector-v4 compact
base can move from baseline-only byte attribution toward real section
value-per-byte profiles.

## What Changed

- Added `tac.substrates.pact_nerv_selector_v4.section_value`.
- Exposed PSV4 logical section layout for `decoder_qw`, `latents_rc`,
  `selectors_rc`, and `receiver_state`.
- Added semantic neutralization for:
  - `decoder_qw`: zero decoder tensors through the PSV4 archive grammar.
  - `latents_rc`: zero the latent matrix through the PSV4 quantized-latent path.
  - `selectors_rc`: remove the charged selector stream.
- Receiver-state neutralization fails closed because meta fields define runtime
  decode shape and should not be treated as expendable candidate bytes.
- Exported the helper from the selector-v4 package surface.

## Verification

- `.venv/bin/ruff check --fix src/tac/substrates/pact_nerv_selector_v4/section_value.py src/tac/substrates/pact_nerv_selector_v4/__init__.py src/tac/substrates/pact_nerv_selector_v4/tests/test_section_value.py`
- `.venv/bin/ruff check src/tac/substrates/pact_nerv_selector_v4/section_value.py src/tac/substrates/pact_nerv_selector_v4/__init__.py src/tac/substrates/pact_nerv_selector_v4/tests/test_section_value.py`
- `PYTHONPATH=. .venv/bin/pytest src/tac/substrates/pact_nerv_selector_v4/tests/test_section_value.py src/tac/substrates/pact_nerv_selector_v4/tests/test_pact_nerv_selector_v4.py -q`

All passed.

## Next Integration

The remaining gap is a selector-v4 section-value profiler that materializes
baseline plus neutralized archive variants, runs MLX scorer replay, emits
`hprc_mlx_component_neutralization_profile.v1`, and feeds the compact bounded
runner through `--mlx-profile`.
