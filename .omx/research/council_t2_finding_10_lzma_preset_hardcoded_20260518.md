---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: DEFER_PENDING_EVIDENCE
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "lzma preset=9 hardcoded vs zstd-22 vs brotli-11 is meaningful arbitrariness"
    classification: HARD-EARNED
    rationale: "Wave 2A audit row #3 documents the comparison gap; per-payload empirical sweep in Wave 2C will resolve."
council_decisions_recorded:
  - "op-routable #1: ratify Wave 2C empirical sweep result when landed (per-payload encoder + preset table)"
  - "op-routable #2: emit canonical `tac.encoding.optimal_encoder_for_payload(payload_bytes, sample_kind)` helper"
council_predicted_mission_contribution: rigor_overhead
council_override_invoked: false
council_override_rationale: null
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: "2026-06-17T00:00:00Z"
predicted_mission_contribution: rigor_overhead
finding_action_class: defer
finding_followup_dispatch_envelope_usd: 0.00
finding_canonical_path: experimental
---

# Finding 10: lzma preset=9 hardcoded vs zstd-22 vs brotli-11

## What happened

Wave 2A audit row #3: lzma preset=9 hardcoded across multiple sites; comparison against zstd-22 + brotli-11 missing. Per-payload empirical sweep in flight via Wave 2C.

## Council deliberation (T2 sextet, brief)

Unanimous: DEFER per Finding #9 pattern. Editor-only follow-up; no paid GPU.

## Verdict + rationale

**DEFER_PENDING_EVIDENCE**: Wave 2C empirical sweep in flight; ratify when landed.

## Action class + next-step dispatch

**defer** — T2 sextet ratify after Wave 2C lands.

## No-signal-loss persistence

- Atom emitted: `build_council_deliberation_atom(atom_id="council_t2_finding_10_lzma_preset_hardcoded_20260518", ...)`
- Posterior anchor via `append_council_anchor(...)` with retrospective due 2026-06-17
- Cross-references: standing-directive finding #10; Wave 2A audit row #3; Wave 2C in flight

## Reactivation criteria

- Wave 2C empirical results land → ratify per-payload table
- 30 days without results → retrospective
