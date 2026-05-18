---
schema: codex_findings_v1
finding_id: meta_audit_addendum_14th_instance_self_caught_phantom_provenance_api_in_g1_routing_directive_20260518
review_date: "2026-05-18"
author: codex
related_files:
  - .omx/research/codex_routing_directive_g1_authority_upgrade_metadata_bucket_to_genuine_family_via_canonical_provenance_20260518.md
  - src/tac/provenance.py
score_claim: false
promotion_eligible: false
research_only: true
---

# META-audit addendum #14: phantom provenance API in G1 routing directive

## Finding

The G1 authority-upgrade routing directive originally cited provenance helper
names by analogy rather than by the actual exported `tac.provenance` API. That
is another instance of `CONFLATE_DECLARATIVE_WITH_PHYSICAL`: the design
described the right authority discipline but named non-existent helper symbols.

## Corrected authority surface

Use the verified module-level API:

- `build_provenance_for_archive_member(...)`
- `build_provenance_for_predicted(...)`
- `build_provenance_for_macos_cpu_advisory(...)`
- `build_provenance_for_mps_proxy(...)`
- `build_provenance_for_research_sidecar(...)`
- `build_provenance_aggregate(...)`
- `provenance_to_dict(prov)`
- `audit_score_claim_dict(payload, expected_axis=None)`

The corrected directive records these symbols and notes that
`build_provenance_for_archive_member` fails closed when the cited archive path
does not exist.

## Protocol Update

Routing prose must grep or inspect the canonical implementation before naming
helpers, flags, or serializers. This applies the existing "never invent CLI
flags" rule to API names in research directives.

## Status

`score_claim=false`; this is a prose/API authority correction only. No score,
promotion, or candidate readiness changes follow from this addendum.
