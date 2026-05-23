# Codex Findings: Serialized Archive Economics Guard

utc: 2026-05-23T21:49:50Z
agent: codex
topic: serialized archive byte economics and planner-byte authority
research_only: false

## Scope

Follow-up hardening after materializer completion contracts. Planner byte
savings, inverse-action water-fill costs, and nested diagnostic hashes were
still easy to confuse with realized submission-archive evidence. This pass
makes that boundary explicit and fail-closed.

## Landed

- Added `tac.optimization.serialized_archive_economics` as the canonical
  serialized archive delta contract.
- Marked byte-shaving `predicted_saved_bytes` as planner-model signal only and
  injects serialized-delta blockers until source/candidate archive byte records
  prove realized savings.
- Byte-range entropy recode chain manifests now record source archive bytes,
  candidate archive bytes, and a serialized archive delta contract; the queue
  postcondition requires `candidate_archive_bytes < source_archive_bytes` and
  `serialized_archive_delta.status == realized_saving`.
- Inverse-scorer cell chains record archive economics and modeled cell cost
  without requiring serialized savings, preserving the distinction between
  scorer-surface acquisition cost and archive-rate proof.
- Inverse action water-fill costs now carry
  `planner_budget_cost_not_serialized_savings` semantics through acquisition,
  campaign, and materializer surfaces.
- Exact-readiness archive manifest parsing now ignores generic nested
  diagnostic hashes/byte counts; only archive-positioned mappings can satisfy
  archive SHA/size authority.
- Planning-only materializers are no longer resolvable as executable candidate
  archive emitters.

## Verification

- `160 passed, 1 warning`: serialized archive economics, materializer queue,
  candidate queue, byte-range recode materializer, inverse-scorer materializer,
  byte-shaving campaign, and optimizer exact-readiness focused bundle.
- `42 passed`: experiment queue focused bundle.
- Focused `ruff check` passed on touched production/test files.
- `git diff --check` passed.

## Authority Boundary

This tranche does not create score authority, promotion eligibility, rank/kill
eligibility, or exact-eval readiness. It only prevents planner/model byte
signals from satisfying archive-rate evidence until a byte-closed candidate
archive is smaller than its source archive and the serialized delta contract
records the realized saving.
