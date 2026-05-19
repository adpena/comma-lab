---
council_tier: T2
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: DEFER_PENDING_EVIDENCE
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "brotli quality 10 vs 11 across 6 files is meaningful inconsistency"
    classification: HARD-EARNED
    rationale: "Wave 2A audit row #2 documented the inconsistency; Wave 2C empirical sweep in flight will determine per-payload optimal."
  - assumption: "Wave 2C will land empirical sweep results that resolve the inconsistency"
    classification: HARD-EARNED-PENDING-WAVE-2C-LANDING
    rationale: "Wave 2C subagent abea3863 was in flight; sister checkpoint indicates Wave 2C completed (`d142b6ad9` formula_extinctions landed; experimental likely also)."
council_decisions_recorded:
  - "op-routable #1: ratify Wave 2C empirical sweep verdict when it lands (per-payload brotli-quality table)"
  - "op-routable #2: emit canonical `tac.encoding.brotli.optimal_quality_for_payload(payload_bytes, sample_kind)` helper"
  - "op-routable #3: per-callsite wire-in (6 files) to canonical helper"
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

# Finding 9: brotli quality 10 vs 11 inconsistency across 6 files

## What happened

Wave 2A audit row #2: 6 files inconsistently use brotli quality 10 vs 11. Wave 2C empirical sweep was in flight at time of standing directive (sister subagent `abea3863`).

## Council deliberation (T2 sextet, brief)

### Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + Assumption-Adversary
Unanimous: DEFER until Wave 2C results land; then ratify per-payload optimal table. Editor-only follow-up; no paid GPU. Sister-coordination check completed (sister appears to have landed).

## Verdict + rationale

**DEFER_PENDING_EVIDENCE**: Wave 2C empirical sweep in flight; ratify when landed.

## Action class + next-step dispatch

**defer** (T2 sextet ratify after Wave 2C lands; editor-only). Reactivation: when Wave 2C empirical sweep result available.

## No-signal-loss persistence

- Atom emitted: `build_council_deliberation_atom(atom_id="council_t2_finding_9_brotli_quality_inconsistency_20260518", ...)`
- Posterior anchor via `append_council_anchor(...)` with `deferred_substrate_id="wave2c_brotli_quality_sweep"`, `deferred_substrate_retrospective_due_utc="2026-06-17T00:00:00Z"` per Catalog #300 mission-alignment retrospective
- Probe outcome: not yet (Wave 2C will register)
- Cross-references: standing-directive memory file finding #9; Wave 2A audit row #2; Wave 2C in flight (sister `abea3863`)

## Reactivation criteria

- Wave 2C empirical results land → T2 sextet ratify per-payload table
- 30 days elapse without Wave 2C results → retrospective per Catalog #300
