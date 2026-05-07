# HNeRV HDC2 Mixed Context Recode - 2026-05-07

## Scope

Bounded local HDC2 follow-up after `ad4adb8b`. This pass did not dispatch
remote or GPU work, did not claim a lane, did not create a candidate archive,
and did not make a score claim.

## Variant

Added an `HDM2` planning fixture in `tac.hnerv_decoder_recode`:

- fixed-schema decoder records use `PACKED_STATE_SCHEMA` order instead of
  writing record names and value counts;
- global previous-symbol contexts are preserved from HDC2;
- each context deterministically chooses range coding or raw literal bytes
  when raw storage is smaller than the local static range model plus table.

The existing HDC2 fixture remains unchanged. `HDM2` is a raw-equivalent local
planning artifact only until runtime/archive support, strict compliance, lane
claiming, and exact CUDA auth eval exist.

## Byte Accounting

Measured read-only from
`experiments/results/hnerv_entropy_packet_discovery_20260506_codex/hdc2_stream_work_product/candidate_hdc2_global_prev_symbol_stream.bin`.

- HDC2 bytes: `221381`
- HDC2 SHA-256: `41816a2834eb9ca3e9ff78e138b30a1c22b0ec4d580cea8de55042dcb624cc5f`
- HDC2 header bytes: `40840`
- HDC2 range payload bytes: `180429`
- HDM2 bytes: `208821`
- HDM2 SHA-256: `4a1db82f170db62038ad683d6024920b1da59cd9e54e8fda8e3109605fc08bc5`
- HDM2 header bytes: `19329`
- HDM2 mixed payload bytes: `189380`
- HDM2 range payload bytes: `147466`
- HDM2 raw payload bytes: `41914`
- HDM2 range contexts: `60`
- HDM2 raw contexts: `187`
- fixed-schema metadata elided versus HDC2: `444`
- byte reduction versus HDC2: `12560`
- header reduction versus HDC2: `21511`
- payload increase versus HDC2: `8951`
- current frontier decoder section: `170127`
- HDM2 delta versus frontier decoder section: `+38694`
- remaining reduction needed to beat frontier section: `38695`

## Interpretation

`HDM2` closes a deterministic part of HDC2 static-context overhead, but pays
some of it back as literal payload. It is a useful bounded step toward the
combined target, not an archive-ready or score-lowering packet. The next local
step should target the remaining payload gap without losing the header savings.

## Verification

- `.venv/bin/python -m pytest src/tac/tests/test_hnerv_decoder_recode.py src/tac/tests/test_hnerv_hdc2_combined_entropy.py -q`
  - `9 passed`
- `.venv/bin/python -m py_compile src/tac/hnerv_decoder_recode.py src/tac/hnerv_hdc2_combined_entropy.py`
  - passed
- `.venv/bin/ruff check src/tac/hnerv_decoder_recode.py src/tac/hnerv_hdc2_combined_entropy.py src/tac/tests/test_hnerv_decoder_recode.py src/tac/tests/test_hnerv_hdc2_combined_entropy.py`
  - passed
