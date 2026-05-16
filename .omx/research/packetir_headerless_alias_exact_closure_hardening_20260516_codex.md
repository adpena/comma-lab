# PacketIR Headerless Alias Exact-Closure Hardening - 2026-05-16

## Summary

Codex adversarial review found that PR106 PacketIR exact closure accepted
headerless/magicless runtime aliases from runtime-side SHA fields alone. That
proved the runtime reconstructed the original inner PR106 payload, but did not
bind the claimed `candidate_headerless_section_sha256` to the candidate
PacketIR parser's own consumed-byte section row.

## Failure Class

- A runtime proof could claim a syntactically valid headerless section digest,
  offset, and length.
- The exact-closure helper accepted that identity when source/runtime inner
  PR106 hashes matched.
- The closure did not independently require the candidate parser proof's
  consumed section row to carry the same digest, offset, length, and
  `score_affecting=true`.

This is an exact-custody bug, not a score result. No score claim or lane
promotion changes from this memo.

## Fix

`src/tac/packetir_exact_closure.py` now passes the candidate
`packet_ir_consumed_byte_proof` into runtime section matching. Headerless alias
acceptance for formats `0x08`, `0x09`, `0x0A`, `0x0B`, and `0x0C` requires the
runtime proof's claimed candidate section SHA/offset/length to match the
candidate parser proof's `sections[]` row for the expected headerless section.

The returned evidence now exposes the bound parser row and whether
`candidate_section_bound_to_consumed_byte_proof` passed.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_packetir_exact_closure.py -q`
- `.venv/bin/python -m pytest src/tac/tests/test_pr106_hdm9_decoder_recode.py::test_packetir_closure_accepts_hdm9_runtime_section_alias src/tac/tests/test_pr106_hdm9_decoder_recode.py::test_packetir_closure_accepts_hdm9_hlm3_runtime_section_alias src/tac/tests/test_pr106_hdm9_decoder_recode.py::test_packetir_closure_accepts_hdm9_hlm3_magicless_runtime_section_alias -q`
- `.venv/bin/python -m py_compile src/tac/packetir_exact_closure.py src/tac/tests/test_packetir_exact_closure.py src/tac/tests/test_pr106_hdm9_decoder_recode.py`
- `ruff check src/tac/packetir_exact_closure.py src/tac/tests/test_packetir_exact_closure.py src/tac/tests/test_pr106_hdm9_decoder_recode.py`
- `git diff --check`

## Reactivation Criteria

Reopen this finding only if a future PR106/PacketIR closure path introduces a
new headerless, magicless, or reconstructed-payload alias that can pass without
binding runtime-side identity to parser-consumed candidate bytes.
