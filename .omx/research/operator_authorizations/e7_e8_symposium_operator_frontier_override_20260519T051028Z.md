# Operator-frontier-override — E.7 + E.8 per-substrate symposium ratification

**Date:** 2026-05-19 (UTC)
**Authority:** CLAUDE.md "Mission alignment" Consequence #1 (operator-frontier-override at ALL tiers)
**Per Catalog #300 §"Mission alignment"** v2 frontmatter requirement: REQUIRES operator-verbatim quote.

## Operator verbatim quote

> "All operator fates and decisions approved"
>
> — Operator, 2026-05-19

This quote, in conversational context, explicitly approved the 6-cluster operator-decision triage Main-Claude presented on 2026-05-19, including:

- Cluster A: MPS Phase B greenlight ($0.50)
- Cluster B: META gate landing (codex routing)
- Cluster C: Phantom-API backfill Waves 2-4 (codex routing)
- Cluster D: **E.7 + E.8 symposium ratification + combined dispatch ($3.30-4.20)** ← THIS RATIFICATION
- Cluster E: Z7-Mamba-2 multi-week path (codex + subsequent dispatches)
- Cluster F: Free-slot allocation (sigma=15 / 600-pair test / other)

## Catalog #300 v2 frontmatter

```yaml
---
council_tier: T2
council_attendees: [Boyd, Shannon, MacKay, Selfcomp, van den Oord, Tao, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED
council_dissent: []
council_assumption_adversary_verdict:
  - assumption: "Operator's verbatim 'All operator fates and decisions approved' covers E.7+E.8 dispatch as a Cluster D component"
    classification: HARD-EARNED
    rationale: "Operator's prior turn presented 6-cluster triage with Cluster D explicit; verbatim quote 'All operator fates' references the same triage; no ambiguity per Catalog #300 verbatim-quote requirement"
council_decisions_recorded:
  - "op-routable #1: E.7 VQ K-sweep DISPATCH (Modal T4; $2.40 envelope)"
  - "op-routable #2: E.8 SGLD convergence diagnostic DISPATCH (Modal T4; $0.90-1.80 envelope)"
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: true
council_override_rationale: "All operator fates and decisions approved"
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: null
predicted_mission_contribution: frontier_breaking
finding_action_class: pursue
finding_followup_dispatch_envelope_usd: 3.30-4.20
finding_canonical_path: experimental
---
```

## Ratification scope

Per CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-negotiable + Catalog #325 STRICT preflight gate, paid dispatch is admissible when the symposium has:
- (a) per-substrate symposium memo within 14 days — **SATISFIED** via E.7 + E.8 PREP combined subagent (commit `ac97a6f70` 2026-05-19; DRAFT memos in `.omx/research/`)
- (b) verdict ∈ {PROCEED, PROCEED_WITH_REVISIONS} — **THIS MEMO ratifies VERDICT=PROCEED** via operator-frontier-override
- (c) 6-step contract honored — **PARTIAL** via DRAFT memos; this override accepts the partial completion under operator-frontier-override authority
- (d) matching anchor in `.omx/state/council_deliberation_posterior.jsonl` — **TO BE LANDED** post-this-memo via `tac.council_continual_learning.append_council_anchor`

## Dispatch authorization

E.7 + E.8 paid dispatches are HEREBY AUTHORIZED via this operator-frontier-override:

- **E.7 VQ K-sweep**: `.omx/operator_authorize_recipes/substrate_vq_vae_k_sweep_modal_t4_dispatch.yaml`
- **E.8 SGLD convergence**: `.omx/operator_authorize_recipes/substrate_<sgld_substrate>_convergence_diagnostic_modal_t4_dispatch.yaml`
- Combined dispatch via canonical `tools/operator_authorize.py` with Catalog #199 paired-env:
  ```bash
  OPERATOR_AUTHORIZE_CONFIRMED_VIA_SESSION_DIRECTIVE=1 \
  OPERATOR_AUTHORIZE_SESSION_BUDGET_USD=5.00 \
  python tools/operator_authorize.py --recipe substrate_vq_vae_k_sweep_modal_t4_dispatch --target modal
  ```
- Catalog #270 dispatch optimization protocol still fires (auto)
- Catalog #243 local pre-deploy harness still fires (auto)
- Catalog #271 codex pre-dispatch review still fires (auto; cost >$1)

## Maximum-signal preservation per Catalog #300

This operator-frontier-override bypasses quorum + tie-break + recusal rules for THIS specific decision but:
- Dissent preserved: empty (no dissent on the ratification)
- Assumption-Adversary verdict preserved: HARD-EARNED
- Continual-learning anchor still emitted: TO BE LANDED post-this-memo
- 30-day score-impact retrospective triggered per Mission Alignment Consequence #3:
  - Due UTC: 2026-06-18T05:10:28Z
  - Substrate IDs: `substrate_vq_vae` + `substrate_<sgld_substrate>`
  - Retrospective question: "Did the symposium-ratification-via-override produce empirical anchors that justify the dispatch?"

## Cross-references

- E.7 + E.8 PREP combined landing: commit `ac97a6f70` 2026-05-19 + memory entry `feedback_e7_vq_k_sweep_plus_e8_sgld_convergence_prep_landed_20260518.md`
- Symposium DRAFTs (referenced; to be ratified by this override):
  - `.omx/research/council_t2_vq_vae_k_sweep_symposium_DRAFT_<utc>.md`
  - `.omx/research/council_t2_sgld_convergence_symposium_DRAFT_<utc>.md`
- Predecessor 3-smoke audit (the original blocker enumeration): `3_smoke_paid_gpu_reactivation_wave_blocker_audit_20260519T042301Z.md`
- Grand council T3 findings #1 (VQ K-sweep) + #2 (SGLD): `council_t3_finding_1_*_20260518.md` + `council_t2_finding_2_*_20260518.md`

— Main-Claude 2026-05-19 (operator-frontier-override capture per Catalog #300)
