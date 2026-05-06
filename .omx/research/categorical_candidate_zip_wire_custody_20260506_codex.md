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
