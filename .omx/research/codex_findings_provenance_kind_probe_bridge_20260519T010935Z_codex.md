# Codex Findings - Provenance Kind Probe Bridge

Date: 2026-05-19T01:09:35Z
Actor: codex
Task: `codex_routing_directive_inflate_py_plus_wyner_ziv_compliance_plus_master_gradient_extension_20260518::ITEM_2`
Scope: provenance-kind authority verification and Catalog #313 bridge hardening

## Authority Standard

Authority is established by a closed custody chain:

1. Durable rule source names the invariant.
2. Canonical implementation encodes the invariant.
3. Public helper path gives callers one correct way to express it.
4. Validator/preflight surface fails closed for unsafe claims.
5. Tests prove both allowed and forbidden paths.
6. Canonical status/commit state records what is complete and what remains separate.

Anything less is advisory signal, not authority.

## Finding

The ITEM_2 source surface was mostly present before this pass:

- `ProvenanceKind` already included `PROCEDURAL_GENERATION_FROM_ARCHIVE_SEED`, `WEIGHT_DERIVED_CODEBOOK`, and `FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD`.
- Builders existed for all three kinds.
- `audit_score_claim_dict` already refused score-claiming `FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD` records.
- Catalog #329 was already the claimed catalog row for this provenance-kind extension.

The remaining authority gap was the Catalog #313 bridge: a forbidden out-of-archive payload verdict needed to become a shared blocking probe outcome with 365-day expiry, instead of living only as local provenance shape.

## Change

Added `register_forbidden_out_of_archive_payload_probe_outcome(...)` in `tac.provenance.builders` and exported it through `tac.provenance`.

The helper:

- accepts only `ProvenanceKind.FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD`;
- requires non-empty evidence path and rejection rationale;
- lazily delegates to `tac.probe_outcomes_ledger.register_probe_outcome`;
- records `probe_kind="forbidden_out_of_archive_payload_provenance"`;
- records `blocker_status="blocking"` and `staleness_window_days=365`;
- carries payload path/SHA and provenance-kind metadata into the ledger row.

Also added a focused ruff config exception for `src/tac/provenance/__init__.py` `RUF022`, matching existing public-API exceptions where semantic export ordering is intentional. Provenance exports are grouped by authority surface: contract, builders, validator, adapters.

## Adversarial Review

Hilbert reviewed the same surface and converged on the close condition:

- The current WIP must be serialized before ITEM_2 can close.
- The Catalog #313 bridge is the missing closure piece.
- A stronger invariant remains separate: archive-seed and weight-derived provenance currently blocks promotion but does not prove archive membership, inflate-time algorithm presence, byte stability, or Catalog #272 mutation proof.

That stronger invariant is not silently folded into ITEM_2. It should be claimed as a separate hardening task if pursued.

## Verification

Commands:

```bash
.venv/bin/python -m pytest src/tac/tests/test_provenance_contract.py src/tac/tests/test_provenance_builders.py src/tac/tests/test_provenance_validator.py -q
.venv/bin/ruff check src/tac/provenance/builders.py src/tac/provenance/__init__.py src/tac/tests/test_provenance_builders.py
.venv/bin/python -m pytest src/tac/tests/test_provenance_contract.py src/tac/tests/test_provenance_builders.py src/tac/tests/test_provenance_validator.py src/tac/tests/test_check_323_canonical_provenance.py src/tac/tests/test_ci_ruff_scope.py -q
git diff --check -- pyproject.toml src/tac/provenance/builders.py src/tac/provenance/__init__.py src/tac/tests/test_provenance_builders.py .omx/state/canonical_task_status.jsonl
```

Results:

- `113 passed in 3.84s`
- touched-file ruff: `All checks passed!`
- `161 passed in 8.14s`
- `git diff --check`: clean

## Status

ITEM_2 is ready to close after serializer commit and canonical task-status update.

