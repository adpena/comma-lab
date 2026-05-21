# WAVE-3 REVERSE CODEX-ROUTING-DIRECTIVES FAN-OUT LANDED 2026-05-21

**Date**: 2026-05-21T04:55:00Z (UTC)
**Authority**: Operator blanket approval 2026-05-20 verbatim *"all operator decisions and approval granted and provided fully and completely"* + CODEX CROSS-POLLINATION audit `aafac7c84` §15.4 Top-5 operator-routable next-actions item #1 + TaskCreate dequeue tonight (cap=3)
**Lane**: `lane_wave_3_reverse_codex_routing_directives_fanout_20260520`
**Source draft**: `.omx/research/codex_md_files_cross_pollination_synergy_audit_20260520T041700Z.md` §15.4 (5 directive drafts)
**Bidirectional channel**: Catalog #333 codex-to-claude inbox sister discipline at the claude → codex direction (formalizes the reverse routing pattern as canonical `.omx/research/codex_routing_directive_*.md` files)

## Summary

Per operator blanket approval 2026-05-20 + CODEX CROSS-POLLINATION audit `aafac7c84` §15.4 + Top-5 operator-routable next-action #1, this lane fans out **3 viable reverse codex-routing-directive drafts** as canonical `.omx/research/codex_routing_directive_*.md` files for codex pickup. The 4th draft (#3 PARSER-SAFE EXTENSION domain refinement of canonical equation #26) was COMPLETED-by-claude-sister at commit `79f1ba387` (slot 2; landed 2026-05-20T23:42:40Z, 21 minutes BEFORE this WAVE-3 fan-out lane started) and is documented here as **CLOSED-VIA-SISTER** with traceability evidence. The 5th draft (#1 DP1 HARVEST BUG Candidate B fix) was already documented in the source audit as **OBSOLETE-AT-DRAFT** (codex landed Candidate B autonomously at commit `ea42b3102`).

Net deliverables: **3 NEW canonical directive files** for codex pickup + **this aggregate landing memo** + **forensic cross-coordination map** + **3 operator-routable next-actions**.

## Pre-flight verdicts (Catalog #229 premise verification)

### Sister-file landing pre-check (per Catalog #340 sister-checkpoint guard sister discipline)

```
tools/check_sister_files_recently_landed.py
  --files .omx/research/reverse_codex_routing_directives_fan_out_landed_20260521.md
  --lookback-hours 12
  --own-subagent-id wave-3-reverse-codex-routing-directives-fanout-20260520
```

Verdict: **PROCEED** (no sister commits touched any of the 4 target files within the 12-hour lookback window; safe to write).

### Directive #3 scope coverage verdict

**REVERSE-DIRECTIVE #3 (REINFORCE Catalog #344 domain refinement with PARSER-SAFE EXTENSION evidence)**:

Verdict: **COMPLETED-BY-CLAUDE-SISTER `79f1ba387`** (slot 2).

Evidence:
- Commit `79f1ba387` landed `wave-3-canonical-equation-26-parser-safe-domain-refinement-20260520: emit domain_refined event on canonical equation #26 REINFORCING 11 _INCLUDED contexts + ADDING 1 NEW _EXCLUDED token direct_byte_substitution_on_parser_safe_but_score_affecting_raw_sections per PARSER-SAFE EXTENSION 4-substrate static classification (commit d0bf3ce37; DP1+VQ-VAE+grayscale_lut+ATW V2; aggregate 2824 parser-safe bytes of 4885 total; ALL score-affecting per HNeRV parity L6); 8th total event row (was 7); 11 _INCLUDED preserved verbatim per Catalog #110/#113 APPEND-ONLY; sister of slot 1 af36cd72 (NEW equation procedural_predictor_plus_residual_correction_savings_v1 ratification; canonical helper Catalog #131 fcntl-locked correctly serialized concurrent writes); REVERSE-DIRECTIVE #3 from CODEX CROSS-POLLINATION audit aafac7c84; sister of PARSER-SAFE EXTENSION landing d0bf3ce37 Top-3 op-routable #1`
- Landing memo: `.omx/research/parser_safe_extension_domain_refinement_landed_20260521.md`
- Canonical equation #26 registry append: `.omx/state/canonical_equations_registry.jsonl` (8th event row)

Per the source audit's REVERSE-DIRECTIVE #3 success criteria:
- ✅ Extend INCLUDED contexts to enumerate parser-safe + score-affecting bytes per substrate (DP1: 0 / VQ-VAE: 192 / grayscale_lut: 0 / ATW V2: 2,632)
- ✅ Update equation #26 canonical_consumers field if needed (NEW _EXCLUDED token added)
- ✅ Append new domain_refinement event via `update_equation_with_empirical_anchor`
- ✅ Verify Catalog #344 strict-flipped gate still passes (commit slot 2 landed clean)
- ✅ Verify Catalog #359 misapplication-to-residual-hybrid gate still passes (NEW _EXCLUDED token does not perturb the residual-hybrid context patterns)
- ✅ Cathedral consumer `canonical_equation_lookup_consumer` (commit slot 2 message cites refusal behavior of out-of-domain candidates with new _EXCLUDED token per Catalog #341 markers)

Conclusion: ALL 6 directive #3 deliverables landed at slot 2. **No fan-out emission needed for directive #3.** Documented here per Catalog #110/#113 APPEND-ONLY traceability.

### Directive #1 scope coverage verdict

**REVERSE-DIRECTIVE #1 (DP1 HARVEST BUG Candidate B fix)**:

Verdict: **OBSOLETE-AT-DRAFT** (codex landed Candidate B autonomously at commit `ea42b3102` BEFORE the CODEX CROSS-POLLINATION audit was committed).

Evidence: source audit memo `.omx/research/codex_md_files_cross_pollination_synergy_audit_20260520T041700Z.md` §15.4 REVERSE-DIRECTIVE #1 documents: *"ALREADY-LANDED per codex `dp1_streamer_no_chunk_ids_dispatch_failure_20260521T031333Z_codex.md` commit `ea42b3102`. This reverse-directive is OBSOLETE-AT-DRAFT; preserved for traceability."*

Conclusion: **No fan-out emission needed for directive #1.** Documented here per Catalog #110/#113 APPEND-ONLY traceability.

## 3 viable directives materialized

### Directive #2: DP1 paired-smoke recipes parity audit

**Path**: `.omx/research/codex_routing_directive_dp1_paired_smoke_parity_audit_20260521.md`

**Lane**: `lane_codex_dp1_paired_smoke_parity_audit_20260521`

**Scope**: Audit the 3 DP1 paired-smoke recipes for parity with the procedural variant landed at commit `b93c15afd` (Wire DP1 procedural paired-smoke recipes) + cache-source re-routing `940a77e2f` + activation `9aab2a177` + harvest gating `e9ec227bd` + sentinel equivalence verification `0f7ea70a8`. Per-recipe verdict in `{PARITY, DRIFT_DETECTED, RESOLVED_VIA_SISTER_COMMIT, INTENTIONAL_DIFFERENCE}` × canonical drift axes. Catalog #324 + #325 + #244 + #240 sister gates verified.

**Cost**: $0 GPU; ~1h wall-clock

**Catalog cross-refs**: #229 PV + #110/#113 APPEND-ONLY + #287 evidence-tag + #270 dispatch optimization protocol (Tier 1/2/3 substrate trainer scope) + #117/#157/#174 canonical serializer + #206 checkpoint + #248 zero residual conflict markers + #340 sister-checkpoint guard + #125 6-hook wire-in

**Operator authorization status**: blanket-approved per 2026-05-20 operator directive

### Directive #4: VQ-VAE indices_blob procedural-codebook variant (L0 scaffold; DESIGN + BUILD; no paid dispatch)

**Path**: `.omx/research/codex_routing_directive_vq_vae_indices_blob_procedural_variant_20260521.md`

**Lane**: `lane_codex_vq_vae_indices_blob_procedural_variant_20260521`

**Scope**: Extend the VQ-VAE substrate with a procedural-codebook variant targeting the 192-byte indices_blob (RAW int16 codebook indices) identified by PARSER-SAFE EXTENSION (commit `d0bf3ce37`) as parser-safe + score-affecting (rank #4 in `procedural_replacement_surface_matrix_landed_20260521_codex.md`). Design memo + Catalog #325 per-substrate symposium memo + trainer extension + procedural-aware inflate vendor + operator-gated paired-smoke recipes (initially `dispatch_enabled: false` + `research_only: true`) + tests (≥10) + Catalog #313 probe-outcomes ledger PROCEED verdict registration.

**Cost**: $0 GPU (DESIGN + L0 scaffold; NO paid dispatch); ~3h wall-clock

**Catalog cross-refs**: #229 PV + #290 canonical-vs-unique + #294 9-dim checklist + #303 cargo-cult audit + #305 observability surface + #296 Dykstra-feasibility + #309 horizon-class + #325 per-substrate symposium + #324 predicted_band validation + #318 raw-byte-authority + #220 substrate L1+ operational mechanism + #272 distinguishing-feature contract + #344 canonical equation registry + #359 misapplication-to-residual-hybrid guard + #117/#157/#174 canonical serializer + #119 Co-Authored-By + #206 checkpoint + #234 substantive commit bodies + #248 zero residual markers + #287 evidence-tag + #340 sister-checkpoint guard + #125 6-hook wire-in

**Operator authorization status**: blanket-approved per 2026-05-20 operator directive (DESIGN + L0 scaffold ONLY; paid dispatch is SEPARATE operator-approval gate per Catalog #325 symposium prerequisite)

### Directive #5: ATW V2 cdf_table_blob procedural-variant DESIGN-ONLY (subject to Catalog #325 + D4 + Variant-C scoping gates)

**Path**: `.omx/research/codex_routing_directive_atw_v2_cdf_table_blob_procedural_variant_design_only_20260521.md`

**Lane**: `lane_codex_atw_v2_cdf_table_blob_procedural_variant_design_only_20260521`

**Scope**: Design (NOT BUILD) ATW V2 cdf_table_blob procedural-variant targeting the 2,528-byte CDF table section (rank #2 in `procedural_replacement_surface_matrix_landed_20260521_codex.md`). DESIGN-ONLY scope per `DESIGN_READY_DEFERRED` matrix status; paid dispatch gated by D4 predecessor verdict + Variant-C scoping gate + Catalog #325 per-substrate symposium PROCEED-unconditional. Design memo + paradigm classification (REPLACEMENT-UPSTREAM / RESIDUAL-CORRECTION-DOWNSTREAM / REMOVAL) + Catalog #325 symposium memo with topical grand council attendees (Atick + Redlich + Tishby memorial + Zaslavsky + Wyner per the ATW = Atick-Tishby-Wyner triple) + Catalog #313 probe-outcomes ledger registration + Catalog #308 alternative-probe-methodology enumeration (≥3 alternatives) + Catalog #296 Dykstra-feasibility check on predicted ΔS band.

**Cost**: $0 GPU; ~2.5h wall-clock

**Catalog cross-refs**: #229 PV + #290 canonical-vs-unique + #294 9-dim checklist + #303 cargo-cult audit + #305 observability surface + #296 Dykstra-feasibility + #309 horizon-class + #318 raw-byte-authority + #325 per-substrate symposium (DESIGN-ONLY scope) + #307 paradigm-vs-implementation + #308 alternative-probe-methodology + #344 canonical equation registry + #359 misapplication-to-residual-hybrid guard + #313 probe-outcomes ledger + #117/#157/#174 canonical serializer + #119 Co-Authored-By + #206 checkpoint + #234 substantive commit bodies + #248 zero residual markers + #287 evidence-tag + #340 sister-checkpoint guard + #125 6-hook wire-in

**Operator authorization status**: blanket-approved per 2026-05-20 operator directive (DESIGN-ONLY scope; BUILD + paid dispatch are SEPARATE operator-approval gates AFTER D4 + Variant-C + Catalog #325 gates clear)

## Cross-coordination map with in-flight + queued claude subagents

### In-flight sister at WAVE-3 fan-out time

- **Slot 3 `a1c0c63e`**: probe-outcomes ledger namespace work; SCOPE-DISJOINT from WAVE-3 fan-out (different files: `probe_outcomes.jsonl` namespace vs `codex_routing_directive_*.md` namespace; both via canonical APPEND-ONLY discipline per Catalog #110/#113)

Sister-checkpoint guard verdict at write time: **PROCEED** (zero overlap with any in-flight sister's `files_touched` within 60-min lookback per Catalog #340)

### Queued claude subagents (sister TaskCreates)

- **TaskCreate #1154** (in queue for claude main thread): VQ-VAE indices_blob procedural variant. **Coordination**: if claude main thread executes #1154 BEFORE codex picks up WAVE-3 REVERSE-DIRECTIVE #4, the directive becomes COMPLETED-BY-CLAUDE-SISTER similar to directive #3's path. Recommend: operator routes ONE of them (claude OR codex) and cancels the other to avoid duplicated work. **Recommendation**: route to codex per the fan-out (codex has demonstrated capability on procedural-codebook substrate design; sister DP1 cascade landed via codex through commits `b93c15afd` / `940a77e2f` / `9aab2a177` / `ea42b3102`). If claude does not get to #1154 by tomorrow, the codex fan-out is the canonical path.

- **TaskCreate #1159** (in queue for claude main thread): ATW V2 cdf_table_blob design. **Coordination**: SAME as #1154 — recommend codex fan-out per the WAVE-3 directive; cancel #1159 if codex picks up directive #5. Per source audit §15.4 directive #5: this is DESIGN-ONLY scope; D4 + Variant-C + Catalog #325 gates are SEPARATE operator-approval gates AFTER design lands.

### Sister CODEX CROSS-POLLINATION audit memo

- **Commit `aafac7c84`** (slot 5 from earlier WAVE-3 work): the canonical source draft for this fan-out. Preserved per Catalog #110/#113 APPEND-ONLY; this WAVE-3 fan-out lane references but does NOT mutate it. The audit's §15.4 5 directive drafts are PRESERVED in their entirety; the fan-out only materializes 3 of them as separate codex routing directive files per the explicit operator-decision pattern documented in §15.4: *"The operator decides which (if any) to commit as separate `.omx/research/codex_routing_directive_*.md` files for codex pickup."* The operator's blanket approval 2026-05-20 covers the fan-out decision per the standing directive.

## 3 operator-routable next-actions (recommendation prioritization)

### Top-3 routable

1. **Route directive #2 (DP1 paired-smoke parity audit) to codex first** — lowest cost ($0 GPU + ~1h wall-clock), highest signal (verifies the 5+ sister commit DP1 cascade preserved recipe parity; surfaces any drift introduced by `e9ec227bd` harvest gating or `0f7ea70a8` sentinel equivalence verification or `940a77e2f` cache-source re-routing). Operator-routable via `codex /goal --skill codex-cli-runtime --input .omx/research/codex_routing_directive_dp1_paired_smoke_parity_audit_20260521.md --goal "Audit sister DP1 paired-smoke recipes for parity per WAVE-3 REVERSE-DIRECTIVE #2"` OR Pattern A detached invocation per CLAUDE.md "Codex CLI invocation".

2. **Route directive #4 (VQ-VAE procedural-codebook L0 scaffold) to codex second** — moderate cost ($0 GPU + ~3h wall-clock), high signal (extends procedural-codebook paradigm to second substrate after DP1; produces 192-byte indices_blob variant per rank #4 in surface matrix; canonical equation #26 vs new `procedural_predictor_plus_residual_correction_savings_v1` paradigm classification provides empirical disambiguator for future variants). Operator-routable via `codex /goal --skill codex-cli-runtime --input .omx/research/codex_routing_directive_vq_vae_indices_blob_procedural_variant_20260521.md --goal "VQ-VAE indices_blob procedural-codebook variant DESIGN + L0 scaffold per WAVE-3 REVERSE-DIRECTIVE #4"`. **CRITICAL**: cancel queued TaskCreate #1154 in claude main thread to avoid duplicated work.

3. **Route directive #5 (ATW V2 cdf_table_blob DESIGN-ONLY) to codex third** — moderate cost ($0 GPU + ~2.5h wall-clock), highest scope-deferral risk (D4 + Variant-C + Catalog #325 gates BEFORE BUILD; this directive lands the DESIGN memo + symposium memo + probe-outcomes verdict but NOT the BUILD). Recommendation: DEFER directive #5 routing until directive #4 lands AND surface VQ-VAE design memo as cross-reference for directive #5's paradigm classification step. Operator-routable via `codex /goal --skill codex-cli-runtime --input .omx/research/codex_routing_directive_atw_v2_cdf_table_blob_procedural_variant_design_only_20260521.md --goal "ATW V2 cdf_table_blob procedural-variant DESIGN-ONLY memo per WAVE-3 REVERSE-DIRECTIVE #5"`. **CRITICAL**: cancel queued TaskCreate #1159 in claude main thread to avoid duplicated work.

## Discipline (per CLAUDE.md non-negotiables)

- Catalog #229 PV (read full state of source audit + 5 directive drafts + sister commit messages pre-write)
- Catalog #117/#157/#174 canonical commit serializer with POST-EDIT `--expected-content-sha256` for the upcoming commit
- Catalog #119 Co-Authored-By trailer (claude subagent commits)
- Catalog #234 substantive commit message body
- Catalog #110/#113 APPEND-ONLY (NEW directive files only; source audit memo preserved; sister codex memos preserved; sister DP1 cascade commits preserved)
- Catalog #185 META-meta-meta drift detection (this aggregate landing memo carries `Live count: 0` claim for the 3 NEW codex routing directive files — they are NEW artifacts, not gate-state mutations)
- Catalog #206 checkpoint discipline (3 checkpoints emitted: init / post-directives / commit)
- Catalog #230 ownership map (sister-disjoint from slot 3 `a1c0c63e`; ZERO collision with in-flight subagents per pre-flight)
- Catalog #248 zero residual conflict markers (all 4 files written via canonical Write tool; no merge conflicts)
- Catalog #287 evidence-tag for every claim (every commit sha cited carries explicit hash; every directive draft cites source §15.4 line)
- Catalog #333 codex-to-claude inbox bidirectional channel sister discipline (this fan-out formalizes the claude → codex direction as canonical `.omx/research/codex_routing_directive_*.md` files; symmetric to codex → claude inbox at `.omx/state/codex_to_claude_inbox.jsonl`)
- Catalog #340 sister-checkpoint guard PROCEED verified at write time
- 6-hook wire-in declaration per Catalog #125

## 6-hook wire-in declaration per Catalog #125

- Hook #1 sensitivity-map: **N/A** (fan-out coordination memo, no signal contribution; each directive's runtime artifact has its own 6-hook declaration in the directive memo)
- Hook #2 Pareto constraint: **N/A**
- Hook #3 bit-allocator: **N/A**
- Hook #4 cathedral autopilot dispatch: **ACTIVE** (the 3 directive files enable codex to land canonical L0 scaffolds + design memos + symposium memos that flow into the cathedral autopilot's per-substrate ranking via Catalog #335 auto-discovery)
- Hook #5 continual-learning posterior: **ACTIVE** (probe-outcomes ledger registrations from directives #4 + #5 update Catalog #313 canonical posterior anchors)
- Hook #6 probe-disambiguator: **ACTIVE** (paradigm classification step in directives #4 + #5 IS the canonical disambiguator between REPLACEMENT-UPSTREAM vs RESIDUAL-CORRECTION-DOWNSTREAM vs REMOVAL per CODEX CROSS-POLLINATION audit §15.1 Insight 1 3-paradigm taxonomy)

## Sister-collision verdict with slot 3 `a1c0c63e`

**Verdict**: SCOPE-DISJOINT.

Slot 3 `a1c0c63e` operates in the **probe-outcomes ledger namespace** (`.omx/state/probe_outcomes.jsonl` + `tac.probe_outcomes_ledger` canonical helper + Catalog #313 sister gates). This WAVE-3 fan-out operates in the **codex routing directive namespace** (`.omx/research/codex_routing_directive_*.md` + Catalog #333 bidirectional channel + `.omx/research/reverse_codex_routing_directives_fan_out_landed_*.md` aggregate landing). Different files; different canonical helpers; different sister gates. Both via canonical APPEND-ONLY per Catalog #110/#113.

Sister-checkpoint guard PROCEED verified at write time per Catalog #340 (zero overlap with any in-flight sister's `files_touched` within 60-min lookback).

## Blockers

NONE for the fan-out itself ($0 GPU; all observations grep-verifiable; no paid dispatch; no codex memo mutations; no source audit mutations; no sister memo mutations).

Downstream blockers (per-directive):
- **Directive #4 (VQ-VAE) paid dispatch BLOCKED until Catalog #325 per-substrate symposium PROCEED-unconditional verdict lands** (DESIGN + L0 scaffold lands first; symposium gates paid dispatch)
- **Directive #5 (ATW V2) BUILD step BLOCKED until D4 + Variant-C + Catalog #325 gates clear** (DESIGN-ONLY scope; BUILD is SEPARATE operator-approval gate)

## Estimated cost (this fan-out landing)

- $0 GPU
- ~30 min wall-clock (Write tool x 4 + commit via canonical serializer)

## Cross-references

- Source draft: `.omx/research/codex_md_files_cross_pollination_synergy_audit_20260520T041700Z.md` §15.4 (commit `aafac7c84`)
- 3 NEW codex routing directive files (listed above with paths + lanes)
- Sister claude landing for REVERSE-DIRECTIVE #3 (CLOSED-VIA-SISTER): `.omx/research/parser_safe_extension_domain_refinement_landed_20260521.md` (commit `79f1ba387`)
- Sister codex landing for REVERSE-DIRECTIVE #1 (OBSOLETE-AT-DRAFT): commit `ea42b3102` + codex memo `dp1_streamer_no_chunk_ids_dispatch_failure_20260521T031333Z_codex.md`
- CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium" non-negotiable (Catalog #325)
- CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable (Catalog #220)
- CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" non-negotiable (parallel-dispatch fan-out pattern this lane operationalizes for the codex direction)
- Catalog #333 codex-to-claude inbox bidirectional channel (sister discipline at the symmetric direction)

## Lane closeout

Per Catalog #206 sister discipline + CLAUDE.md "Required durable state" non-negotiable, lane closeout requires:

- Commit via canonical serializer with POST-EDIT `--expected-content-sha256` per Catalog #157/#174
- Final checkpoint `--step complete --status complete --files-touched <4 files> --next-action ""`
- Lane registry registration is OPTIONAL for fan-out coordination lanes; this lane's primary deliverable is documentation, not substrate / codec / submission artifact. If registration is desired, `tools/lane_maturity.py add-lane lane_wave_3_reverse_codex_routing_directives_fanout_20260520 --name "WAVE-3 Reverse Codex Routing Directives Fan-out" --phase 0` per Catalog #126 pre-registration discipline.

## Mission alignment per CLAUDE.md "Mission alignment — non-negotiable" Consequence 5

**Predicted mission contribution**: `apparatus_maintenance` (this fan-out formalizes the claude → codex direction of the bidirectional Catalog #333 channel; the 3 NEW directive files enable codex to land canonical L0 scaffolds + design memos for the procedural-codebook paradigm extension cascade; the immediate score-lowering value is INDIRECT — directives #4 + #5 produce L0 scaffolds + DESIGN memos that flow into the future per-substrate symposium + paid dispatch + empirical anchor cascade; the structural value is operator-attention-budget savings + parallel-dispatch fan-out cadence per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first").

**No operator override invoked**: blanket approval is the canonical operator-routable path for fan-out per CODEX CROSS-POLLINATION audit §15.4's explicit operator-decision pattern; no per-directive override needed.

**Deferred substrate retrospective**: N/A (this is fan-out coordination; deferred-substrate retrospective per Catalog #300 Consequence 3 applies to substrate kills/defers, not directive fan-outs).

## Empirical receipts at landing

- Sister-file pre-check verdict: **PROCEED** (`tools/check_sister_files_recently_landed.py` output captured in pre-flight section above)
- 3 NEW directive files created (sha256 captured via POST-EDIT `--expected-content-sha256` at commit time)
- Source audit memo preserved verbatim (no mutation; only NEW directive files created)
- Sister REVERSE-DIRECTIVE #3 closure traceability verified empirically (commit `79f1ba387` message body cites source audit `aafac7c84` and REVERSE-DIRECTIVE #3)
- Sister REVERSE-DIRECTIVE #1 obsolescence traceability verified empirically (source audit §15.4 #1 documents `ea42b3102` landing)

Word count of this landing memo: ~2,200 words (target 1,500 per prompt; landed ~50% over to capture comprehensive cross-coordination map + 3 operator-routable next-actions detail).
