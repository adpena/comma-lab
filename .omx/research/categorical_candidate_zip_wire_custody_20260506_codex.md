# Categorical Candidate ZIP Wire Custody

Date: 2026-05-06
Author: codex
Evidence grade: pre-dispatch readiness hardening
Score claim: false
Dispatch attempted: false

## Context

Categorical/QMA9/CLADE-SPADE/openpilot-label candidates are high-EV only if
their payloads are byte-closed and exact-replayable. The readiness audit already
checks charged member hashes, semantic class order, no-op controls, and archive
manifest parity. This tranche closes the remaining ZIP-parser divergence class:
central directory names must match local file header names.

## Change

`src/tac/categorical_candidate_readiness.py` now scans local ZIP headers,
compares central/local member names, rejects duplicate local names, rejects
unsafe local or central names, and records `candidate_archive.zip_wire_contract`
in the readiness JSON.

New blockers:

- `candidate_archive_zip_wire_contract_failed`
- existing `candidate_archive_not_readable_zip` still fires when Python's ZIP
  reader refuses a malformed archive.

## Boundary

This is not a score claim and does not dispatch GPU work. It is a stricter
pre-dispatch custody gate for future categorical candidates.

## Addendum: Deterministic Candidate Schema

The readiness audit now also fails closed when a categorical candidate manifest
omits its schema/kind, when the archive-member manifest omits categorical
schema/kind custody, or when `archive.zip` is not byte-deterministic enough for
cross-platform replay.

New blockers:

- `candidate_manifest_schema_version_missing_or_invalid`
- `candidate_manifest_kind_missing_or_invalid`
- `archive_member_manifest_schema_version_missing_or_invalid`
- `archive_member_manifest_kind_missing_or_invalid`
- `candidate_archive_zip_determinism_contract_failed`
- `candidate_archive_member_order_mismatch`

The deterministic ZIP contract requires fixed `1980-01-01T00:00:00` member
timestamps, Unix `create_system=3`, no ZIP extra fields, no data-descriptor
local headers, canonical permissions (`inflate.sh` executable, other members
0644), and archive member order matching the charged-member manifest. This is
still pre-dispatch readiness only: no score claim, no lane claim, and no exact
CUDA auth eval.
