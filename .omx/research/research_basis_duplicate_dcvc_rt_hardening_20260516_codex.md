# research-basis duplicate DCVC-RT hardening - 2026-05-16

## Trigger

The source-fidelity subagent reported a possible duplicate `dcvc_rt_2025`
research-basis entry. Local verification confirmed two literal entries in
`RESEARCH_SOURCES`; Python kept only the later one, silently discarding the
earlier richer CVPR/source-fidelity contract.

## Landing

The duplicate later `dcvc_rt_2025` literal was removed, preserving the richer
first entry that includes the L5-v2 local paradigms, temporal-model variables,
runtime-cost contract, and paired CPU/CUDA blocker.

A static source-literal test now scans `research_basis.py` for duplicate
top-level `RESEARCH_SOURCES` keys so future duplicate citations fail in CI
instead of silently overriding each other.

## Tests

- `test_research_basis_source_literal_has_no_duplicate_basis_ids`

## Boundary

This is citation/source-fidelity hardening only. Research-basis rows remain
planning-only and cannot authorize dispatch or score claims.
