# Codex Findings: Packet Compiler Native Status Correction

Date: 2026-05-24T16:38:41Z
Agent: Codex

## Finding

The `runtime-rs/crates/tac-packet-compiler` top-level README and Cargo
metadata still described `v0.2.0-rc1` as "COMPLETE NATIVE PARITY" and
publish-eligible even though the Rust crate explicitly retains scaffolded
selector/search/decode meta surfaces:

- `magic_codec_v1` remains `try_load_only`.
- `adaptive_brotli_param_search` remains scaffold-only.
- `decode_ranked_no_op_sidecar` remains scaffold-only.

The strong true claim is narrower: the 19 committed golden-vector primitives
are byte-for-byte parity GREEN against the Python oracle. That is not complete
crate parity and not a complete native replacement for the Python oracle.

## Fix Landed

- Reworded the runtime README status banner, roadmap, publish eligibility, and
  verification sections to distinguish primitive-set parity from crate parity.
- Reworded Cargo metadata so package description and release notes do not
  advertise complete native parity.
- Reworded the active lane-registry row, generated lane-maturity report, OSS
  release notes, and golden-vector test comment so operator-facing surfaces
  do not repeat the stale publish-ready complete-parity claim.
- Preserved historical `.omx/research` ledgers as provenance instead of
  overwriting past dated records.

## Guardrail

Future native PacketIR/runtime-rs work should avoid static "complete" status
claims unless every public surface is either implemented with golden-vector
parity or explicitly removed/excluded from the public crate contract. Remaining
scaffolds must stay fail-closed with `NotImplemented` or `try_load_only`
coverage until promoted.

## Verification

- `cargo test -p tac-packet-compiler`: 130 unit tests, 41 golden-vector
  integration tests, doctest ignored as expected.
- `cargo doc -p tac-packet-compiler --no-deps`: green.
- Active docs/code search no longer finds active `COMPLETE NATIVE PARITY` or
  publish-ready overclaims in `runtime-rs/crates/tac-packet-compiler`.
