# Codex Findings: Wyner-Ziv Deliverability Authority

Timestamp: 2026-05-19T00:58:50Z
Actor: codex
Task: `codex_routing_directive_inflate_py_plus_wyner_ziv_compliance_plus_master_gradient_extension_20260518::ITEM_1`

## Authority Answer

Authority is established by an evidence chain, not by a field existing in a
dataclass:

1. A persisted proof must carry explicit contest-compliance rationale text.
2. The proof must cite a contest-compliant route in a structured citation chain.
3. The loader must refuse stale persisted JSON that omits those fields instead
   of silently backfilling dataclass defaults.
4. The operator audit must classify missing proof authority with a dedicated
   fail-closed label.
5. Any downstream Cathedral/autopilot reward remains planning-only unless the
   proof survives provenance, axis, runtime, and exact-eval custody gates.

This landing closes the stale-artifact authority gap for ITEM_1. It does not
claim score movement.

## What Landed

- `load_deliverability_proof_for_archive(...)` now refuses persisted
  DeliverabilityProof JSON that lacks `contest_compliance_rationale` or
  `contest_compliance_citation_chain`.
- `tools/audit_provenance_compliance.py` now recognizes WZ deliverability proof
  shape and emits `MISSING_CONTEST_COMPLIANCE_RATIONALE` for missing/blank
  rationale or citation-chain authority.
- Focused tests cover:
  - legacy persisted proof loader refusal;
  - missing rationale audit classification;
  - blank citation-chain audit classification.

## Adversarial Review

Read-only xhigh review confirmed the directive path points at
`src/tac/wyner_ziv_deliverability/contract.py`, but the live dataclass is
`src/tac/wyner_ziv_deliverability/proof_builder.py::DeliverabilityProof`.
It also found the core false-authority risk: default dataclass fields make
in-memory proof construction easy, but persisted artifacts without explicit
compliance evidence can otherwise look authoritative after load.

This landing addresses that at the persisted-artifact and audit boundary.
Route-specific citation specialization remains a follow-up risk: the builder
still emits the broad default citation chain, so future work should narrow
that chain by actual deliverability route before promoting any proof beyond
planning authority.

## Verification

Commands run:

```text
.venv/bin/python -m pytest src/tac/tests/test_wyner_ziv_deliverability_proof_builder.py src/tac/tests/test_provenance_validator.py -q
.venv/bin/ruff check tools/audit_provenance_compliance.py src/tac/wyner_ziv_deliverability/proof_builder.py src/tac/tests/test_wyner_ziv_deliverability_proof_builder.py src/tac/tests/test_provenance_validator.py
.venv/bin/python tools/audit_provenance_compliance.py --scan-root . --summary
.venv/bin/python tools/canonical_task_status.py --validate
```

Results:

- Pytest: 73 passed.
- Ruff: all checks passed on touched files.
- Provenance audit: 2480 artifacts scanned; existing baseline violations
  remain, with no score promotion implied.
- Canonical task status: valid with 99 rows after ITEM_1 claim.

## Residual Risk

This is an authority hardening, not a frontier claim. It does not solve:

- route-specific citation minimization;
- direct constructor defaults for local tests/probes;
- ITEM_2 canonical task-status mismatch, even though source-level provenance
  enum/builder plumbing appears present;
- exact CUDA/CPU promotion custody for any WZ-derived rate saving.
