# Gap + Bug + Warning + Issue Inventory — 2026-05-20T12:00:00Z

> **Deliverable E of T3 grand council symposium 2026-05-20**
> Cite-chain: `council_t3_grand_strategy_review_20260520T120000Z` + sister Deliverables B/C/D

## Section 1 — STRICT preflight gate state (Catalog # surveillance)

### Headline: catalog # at 354 of 400 ceiling

**Source:** `.omx/state/next_catalog_number.txt` + git log + CLAUDE.md catalog table count

**Per Catalog #299 quota brake:** 46 slots remaining before structural "stop and consolidate" pause fires; current cadence ~5-10/week → ceiling hit in 5-9 weeks

**Mitigation per Decision 6:** every new gate landing MUST satisfy consolidation discipline (retire / file-level waiver / replace). Sister extension via Catalog #287 v2 scope-extension precedent (extend existing gate to cover new surface) is preferred.

### Headline: T3 + T4 OVER_CADENCE

**Source:** `.venv/bin/python tools/audit_council_tier_cadence.py` (at deliberation time)

| Tier | 30d count | Budget | %    | Verdict       |
|------|-----------|--------|------|---------------|
| T1   | 6         | ∞      | n/a  | UNBOUNDED     |
| T2   | 47        | 90     | 52%  | WITHIN_BUDGET |
| **T3** | **45**  | **13** | **346%** | **OVER_CADENCE** |
| **T4** | **7**   | **2**  | **350%** | **OVER_CADENCE** |

**Operator action requested.** See Decision 2.

### Mission-alignment alerts: ALL GREEN

- ✓ rigor-overhead+apparatus-maintenance distribution 26% (below 60% threshold)
- ✓ no overdue 30-day deferred-substrate retrospectives
- ✓ no overdue annual gate audits

## Section 2 — Provenance compliance state (Catalog #323)

### Headline: 202 violations across 2127 artifacts

**Source:** `.venv/bin/python tools/audit_provenance_compliance.py --summary` (at deliberation time)

| Class | Count |
|-------|-------|
| CLEAN | 1923 |
| WARN  | 2 |
| VIOLATION | 202 |

**Top violations (operator-routable):**
1. `.omx/state/promoted_result.json` — MISSING_PROVENANCE on `score` field
2. `.omx/state/vast_search_pmg_hotspot_h100_20260502T1350Z.json` — MISSING_PROVENANCE
3. `.omx/state/vast_search_4090_nvdec_replacement_20260501T121639Z.json` — MISSING_PROVENANCE
4. `.omx/state/vastai_show_instances_live_20260501T120901Z.json` — MISSING_PROVENANCE
5. `.omx/state/lightning_active_jobs.json` — MISSING_PROVENANCE on `score_contest_cuda`

**Mitigation per Decision 7 + Plan S-3:** reclassify state-artifact rows as DERIVED_OUTPUT per Catalog #113 + emit canonical waivers; fix 66 INVALID_PROVENANCE_SHAPE via one-pass schema fix in writers. 1 small subagent session.

## Section 3 — Lane registry surveillance (Catalog #90 + #298 + #220 + #272 + #233)

### Lane registry size: 42938 lines

**Source:** `wc -l .omx/state/lane_registry.json`

**Operator-routable surveillance per Decision 12:** sample K=8 most-recent landed lanes' actual outcomes via Catalog #253 compressive landscape canonical helper.

**Stale L1 SCAFFOLD substrate audit per Catalog #298:** `tools/audit_stale_l1_substrates.py` — operator-routable monthly cadence. Per Section 1 audit (commit `de75b8b13` at sweep time), no IMMEDIATE stale-L1 violations reported but recommend monthly re-run.

## Section 4 — Modal call_id ledger state (Catalog #245)

### Ledger size: 835K (canonical fcntl-locked JSONL APPEND-ONLY)

**Source:** `.omx/state/modal_call_id_ledger.jsonl`

**Surveillance per Catalog #245:** every Modal `.spawn()` registers + every outcome appends terminal event per Catalog #330 sister discipline. No known orphan Modal calls per `tools/harvest_modal_calls.py`.

## Section 5 — Probe outcomes ledger (Catalog #313)

### Ledger size: 59 lines

**Source:** `.omx/state/probe_outcomes.jsonl`

**Surveillance:** per Catalog #313, every probe outcome with `verdict ∈ {INDEPENDENT, KILL, DEFER}` registers as blocking; dispatch wrappers refuse re-firing within 30-day staleness window. No known false-positive blocks reported.

## Section 6 — Canonical task status (Catalog #331)

### Ledger size: 251 lines

**Source:** `.omx/state/canonical_task_status.jsonl`

**Recent activity (codex_swarm_burndown_20260520 session):**
- V8 learned-compression Faiss scaffold landed completed (commit batch `codex_findings_v8_faiss_premise_fix_scaffold_landed_20260520T032630Z_codex.md`)
- PR101/FEC6 PacketIR deterministic compiler identity completed (commit `8aa92e0e0c` + sister)
- Codex 3-findings fix wave landed at multiple checkpoints

**Surveillance:** Catalog #331 dangling-transition check passes per `tools/check_canonical_task_status_no_dangling_transitions.py --strict`. No known dangling tasks.

## Section 7 — Cathedral autopilot consumer state (Catalog #335 + #336 + #337 + #341)

### Cathedral consumers count: 24+

**Source:** `src/tac/cathedral_consumers/` package count

**Recent additions (RESPAWN-MG-7-BUNDLE commit `a6614d5eb`):** 8 master-gradient exploit consumers (exploits 2-9) per Catalog #354. Bundle completeness verified via Catalog #354 self-protection.

**Surveillance:** auto-discovery loop per Catalog #336 fires from `cathedral_autopilot_autonomous_loop.py::main`. Routing markers per Catalog #341 verified on all routing consumers.

## Section 8 — Canonical equations registry (Catalog #344)

### Equations count: 6 (initial population)

**Source:** `.omx/state/canonical_equations_registry.jsonl` per Catalog #344

**Surveillance:** every empirical-finding memo dated >= 2026-05-19 MUST reference a canonical equation OR carry `# FORMALIZATION_PENDING:<rationale>` waiver. Catalog #344 STRICT-flipped post-backfill 2026-05-19.

## Section 9 — Subagent collision / coordination state

### Subagent progress ledger state

**Source:** `.omx/state/subagent_progress.jsonl`

**Known in-flight subagents at deliberation time:**
- `grand-council-t3-strategy-review-20260520` (THIS subagent; 4 checkpoints emitted)

**Sister-checkpoint guard per Catalog #340:** clean; no collision detected.

**Editor-vs-editor collision surveillance per Catalog #314 + #340:** clean per recent commit-serializer log.

## Section 10 — Forbidden CLAUDE attribution surveillance

### PR #110 + adpena/comma_video_compression_challenge surfaces

**Source:** `gh pr view 110 --repo commaai/comma_video_compression_challenge --json body` + canonical body templates

**Surveillance per "Forbidden CLAUDE ATTRIBUTION IN PUBLIC-PR SURFACES" standing rule + "User PR Attribution" rule:**
- ✓ PR body sole-author Alejandro Peña <adpena@gmail.com>
- ✓ Zero Claude/Anthropic/Co-Authored/AI-assisted tokens in PR #110 body
- ✓ Submission_dir README.md sole-author
- ✓ Fork branch commits sole-author

**Operator action requested:** monitor any future PR comments for inadvertent leakage; redact via amendment if necessary.

## Section 11 — Sister-subagent ownership map state (Catalog #230)

### Recent bulk-rewrite tracking

**Source:** `.omx/state/commit-serializer.log` + last 50 commits

**Surveillance:** no recent bulk-rewrite commits without ownership-map citation. Catalog #230 fires structurally on next bulk-op landing.

## Section 12 — META-ASSUMPTION cadence surveillance (Catalog #291)

### Most-recent META-ASSUMPTION review

**Source:** `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_assumptions_challenge_audit_break_out_local_minima_landed_20260515.md` (canonical anchor) + sister memos

**Status:** within 7-day cadence window if a fresh review fired in past 7 days. Operator-routable: spawn next META-ASSUMPTION ADVERSARIAL REVIEW per Catalog #291 cadence.

## Section 13 — Phantom-API surveillance (Catalog #287)

### Recent backfill state

**Source:** Catalog #287 v2 STRICT-flipped 2026-05-19 per `feedback_strict_flip_enablers_catalog_343_plus_346_plus_344_progress_landed_20260519.md`

**Surveillance:** 418 phantom-API citations backfilled via `.omx/state/catalog_287_phantom_api_waivers.jsonl` exact-line authority. STRICT mode enforced; new phantom citations refused at landing.

## Section 14 — Frontier-pointer surveillance (Catalog #343)

### Canonical frontier pointer state

**Source:** `.omx/state/canonical_frontier_pointer.json` last refreshed `2026-05-20T11:38:53Z`

**Surveillance per Catalog #343:** no hardcoded score literals allowed in CLAUDE.md or reports/latest.md without canonical-pointer reference OR HISTORICAL_SCORE_LITERAL_OK waiver. STRICT-flipped 2026-05-19; live count 0.

## Section 15 — Sister gates surveillance (Catalog #185)

### Live-count drift detection

**Source:** Catalog #185 META-meta gate

**Surveillance:** every CLAUDE.md catalog entry claiming "Live count: 0" empirically verified via gate function call. No known drift at deliberation time.

## Section 16 — Strict-flip atomicity surveillance (Catalog #176 sister discipline)

### Strict callsite → CLAUDE.md row parity

**Source:** Catalog #176 META-meta gate

**Surveillance:** every `check_*(strict=True, ...)` callsite in `preflight_all()` MUST have matching CLAUDE.md catalog row. No known orphan strict-callsites.

## Section 17 — Multi-subagent edit/commit collision class state

### Catalog # coverage: #117 + #157 + #174 + #216 + #230 + #248 + #289 + #302 + #314 + #340

**Bug class extincted at 8 surfaces:**
- edit-time-checkpoint (#302)
- edit-time-bulk-op (#230)
- commit-time-pre-pre-lock (#157)
- commit-time-staged (#216)
- commit-time-lock-arbitration (#117 + #174)
- post-resolution-residual-marker (#248)
- post-commit DETECT (#314)
- staging-surface PREVENT (#340)

**Status:** 8-surface extinction complete. No known regression.

## Section 18 — Pending operator decisions inventory

### Top operator-routable decisions awaiting action

| Decision | Source memo | Cost | Urgency |
|----------|------------|------|---------|
| 1. Authorize M-1a DreamerV3 RSSM B2 paid smoke | `council_t3_dreamerv3_rssm_paradigm_bridge_per_substrate_symposium_20260519` | $5-15 | HIGH (after OPTIMAL FORM) |
| 2. Authorize M-1b NSCS06 v8 hybrid_class_shift_path_C paid smoke | `council_t3_cargo_cult_resurrection_nscs06_v8_variant_c_20260519` | $15-50 conditional | MID (after K-coverage) |
| 3. Authorize provenance backfill subagent | Plan S-3 | $0 | MID |
| 4. Authorize 1 subagent/week for findings Lagrangian phase 2 | Plan L-1 | $0 | MID (long-term) |
| 5. Decide Q4-Q5 Wyner-Ziv reactivation vs DEFER | Plan M-3 | $0 deferral / $5-15 reactivation | LOW |
| 6. Per-substrate symposium queue prioritization (#851-855) | Plan M-2 | $0 | MID |
| 7. PR #110 lifecycle response (passive) | Plan S-1 | $0 | HIGH passive |
| 8. Cadence consolidation policy commit to CLAUDE.md | Plan S-2 + Decision 2 | $0 | HIGH |
| 9. Pending inventory K=8 sampling subagent per Decision 12 | Plan L-4 | $0 | MID |
| 10. Production-fleet-scale + comma.ai engagement strategy | Plan L-3 | $0 | conditional on S-1 |

## Section 19 — Honest accounting summary

**What landed empirically (past 5-day window):**
- MG-1 through MG-19 editorial wave (12+ subagent sessions; ~19 canonical helpers + 8 cathedral consumers)
- Catalog #354 master-gradient exploit consumer bundle (8 consumers)
- Catalog #344 canonical equations registry framework (6 initial equations)
- Cathedral autopilot auto-discovery paradigm shift (Catalog #335-#337 + #341)
- PR #110 submission (frozen body; D3-D5 chain executed; live OPEN + MERGEABLE)
- Comprehensive OSS hardening (comma.ai/openpilot-grade for both repos)
- MEMORY.md Option-3 archive-bulk rotation
- T3 council deliberations: 45 in 30d (OVER_CADENCE)

**What landed as scaffold (research_only / pending paid dispatch):**
- DreamerV3 RSSM B2 design memo (PROCEED_WITH_REVISIONS)
- NSCS06 v8 Variant C / hybrid_class_shift_path_C design memo (PROCEED_WITH_REVISIONS)
- V1 Faiss V8 learned-compression scaffold (research_only)
- Z7-Mamba-2 substrate design memo
- Z7-LSTM full main design memo (pending per-substrate symposium)
- TT5L V2 design memo
- Z8 design memo
- DP1 deep-dive design memo (pending)

**What worked:**
- Cathedral autopilot auto-discovery paradigm shift (24+ consumers structurally integrated)
- MG-7 bundle (8 master-gradient exploit consumers wired)
- Findings Lagrangian phase 1-a tests
- PR #110 submission execution (sole-author + zero Claude attribution maintained)
- Forbidden Claude attribution discipline (verified clean)
- MEMORY.md Option-3 rotation (signal-preserved 85% reduction)
- Catalog #344 canonical equations registry framework

**What didn't:**
- Frontier movement (CPU stuck at 0.192051 since 2026-05-15; CUDA stuck at 0.205330 since 2026-05-16)
- Cadence control (T3 + T4 OVER_CADENCE)
- Per-substrate OPTIMAL FORM iteration discipline (4-of-5 distinguishing-feature dispatches falsified)

**What surprised us:**
- Cathedral autopilot auto-discovery paradigm shift was operator-routed insight + landed structurally in 2 commits per Catalog #335 contract; expected ~5-7 sister gates instead
- Catalog #344 canonical equations registry framework subsumed 5+ proposed sister gates per Decision 6 consolidation discipline
- PR #110 has been silent for 8.5h; expected initial maintainer engagement faster (per cde13e4bb mining the Yousfi engagement window may not have opened yet)
- T3 cadence at 346% — this was structural drift that the cadence-audit tool caught precisely at deliberation time; without the audit we would not have surfaced it

**HARD-EARNED-vs-CARGO-CULTED summary** (per Catalog #292 sister discipline applied to apparatus):
- **HARD-EARNED:** unified findings Lagrangian + canonical equations registry + cathedral consumer canonical contract + canonical-helper extension over net-new tool
- **CARGO-CULTED:** more T3 deliberations produce more frontier signal (anti-correlated); per-substrate symposium queue produces dispatch-ready substrates (only with OPTIMAL FORM iteration); positioning memos help PR engagement (likely net-negative)

## Section 20 — Bug + warning + issue surface (none currently CRITICAL)

**No CRITICAL bugs or warnings at deliberation time.** All audits GREEN except cadence (Section 1) + provenance backfill (Section 2; trending downward; non-blocking operational hygiene).

**Operator action priority order:**
1. S-1 PR #110 passive monitoring (HIGH passive)
2. S-2 cadence cap commit (HIGH policy)
3. Decision 3 OPTIMAL FORM iteration discipline before M-1a/M-1b paid dispatch (HIGH gate)
4. M-1a DreamerV3 RSSM B2 prerequisites (HIGH after OPTIMAL FORM)
5. S-3 provenance backfill subagent (MID hygiene)
