# Low-Level Archive Acceleration - Worker C - 2026-05-08

Scope: DX/Rust/Zig/low-level acceleration only. This ledger makes no dispatch,
score, frontier, or candidate-archive claim.

## Current Surface

- `runtime-rs/crates/zipwire` is the safest first acceleration surface. It
  already parses EOCD, central directory entries, local headers, duplicate
  names, unsafe names, methods, encrypted members, local/central parity, and a
  strict JSON inspect record.
- `src/tac/submission_archive.py::validate_archive` remains too semantic for a
  direct replacement in this tranche because it also validates manifests, typed
  sidechannel contracts, and payload-level archive invariants.
- `scripts/pre_submission_compliance_check.py::inspect_archive` is a better
  future consumer, but it currently reads payloads to compute member SHA-256s.
  Header acceleration should therefore enter as an auxiliary record first, not
  as the sole archive oracle.

## Implemented This Tranche

- Added `payload_offset` to the Rust `zipwire` member inspection JSON so Python
  tools can attribute byte ranges without extracting payloads.
- Added strict Rust/Python blocker coverage for ZIP data-descriptor members via
  `data_descriptor_member_not_supported`.
- Added `src/tac/zipwire_archive.py`, an opt-in Python bridge:
  - uses `runtime-rs/target/{release,debug}/zipwire` or `TAC_ZIPWIRE_BIN` when
    present;
  - accepts Rust return codes `0` and `1` because blocked archives still emit
    inspect JSON;
  - falls back to a header-only Python parser when Rust is absent or unusable;
  - does not extract or decompress member payloads.
- Added focused Python parity tests for fallback fields, duplicate names,
  local/central name mismatch, data-descriptor blocking, missing-binary fallback,
  and Rust/Python core-record parity when the built binary is present.

## Safest Next Consumer

Wire `inspect_zip_headers()` into `scripts/pre_submission_compliance_check.py`
as an auxiliary check beside the existing `zipfile` path:

1. Call the helper after `archive_exists`.
2. Record the header-only JSON under `record["zip_header_inspect"]`.
3. Convert helper `blockers` into error checks only when a `--strict-zipwire`
   or equivalent internal flag is set.
4. In default mode, fail closed only on a Rust/Python disagreement when both
   records are available and a parity comparison is explicitly requested.

This preserves Python as the validator oracle while making native header scans
observable on live archives.

## Blockers Before Authoritative Replacement

- Add parity fixtures for central/local extra fields, archive comments, CP437
  names, truncated EOCD, malformed central directory lengths, ZIP64 markers,
  and unsupported compression methods.
- Decide whether member payload SHA-256 belongs in the Rust header tool. It
  would require range hashing compressed payloads or optional decompression,
  which is outside this tranche's header-only contract.
- Add an operator-visible gate only after a live archive corpus shows identical
  blocker classification between Rust and Python.
