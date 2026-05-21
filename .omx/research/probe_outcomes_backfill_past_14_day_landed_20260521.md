<!-- HISTORICAL_SCORE_LITERAL_OK:macos_cpu_advisory_probe_verdicts_no_contest_axis_score_claims_2026-05-21 -->
# Probe-outcomes ledger backfill past-14-day landed 2026-05-21

**Date (UTC):** 2026-05-21T04:50:00Z
**Lane:** `lane_wave_3_probe_outcomes_backfill_past_14_day_20260520`
**Subagent:** `wave-3-probe-outcomes-backfill-past-14-day-20260520`
**Scope source:** CODEX CROSS-POLLINATION audit `aafac7c84` §10.3 + §16 Top-5 #4
**Axis:** `[macOS-CPU advisory]` per CLAUDE.md "MPS auth eval is NOISE"; all backfilled rows carry `score_claim=false`, `promotion_eligible=false`, `ready_for_exact_eval_dispatch=false`.

## What landed

Six probe-outcome rows appended to `.omx/state/probe_outcomes.jsonl` via the canonical helper `tac.probe_outcomes_ledger.register_probe_outcome` (fcntl-locked APPEND-ONLY per Catalog #131 + #138 + #245 sister discipline). Ledger row count `82 -> 88`. All six anchors are HISTORICAL past-14-day codex + main-thread probe-verdict memos that the CODEX CROSS-POLLINATION audit §10.3 explicitly identified as missing from the canonical Catalog #313 surface despite each having a fully-landed source memo.

| # | Probe | Source commit | Verdict | blocker_status |
|---|---|---|---|---|
| 1 | magic-codec × DWT detail-subband procedural residual (pair #1) | `debbc5833` | `DEFER` | blocking |
| 2 | magic-codec sparse_packet_ir SRL1 × procedural-codebook null-byte residual (pair #2) | `a986efa99` + codex sister | `DEFER` | blocking |
| 3 | magic-codec procedural seed orthogonality (pair #4) | `d181a3a54` + codex sister | `PROCEED` | advisory |
| 4 | PR101 GOLD master-gradient-null-byte REMOVAL paradigm | `3dfb877c0` | `DEFER` | blocking |
| 5 | Parser-safe null-byte SUBSET smoke (fec6 frontier) | `e3e198c9f` | `DEFER` | blocking |
| 6 | Parser-safe methodology EXTENSION (4-substrate static classification) | `d0bf3ce37` | `PARTIAL` | advisory |

## Why this matters (signal preservation)

Before this backfill, four IMPLEMENTATION-LEVEL falsifications (pair #1 zscore 38.8, pair #2 zscore 101.18, null-byte REMOVAL H3_OPAQUE_TO_SCORER 3/3 inflate failures, parser-safe SUBSET structural-empty) plus two routing-clarifying verdicts (pair #4 raw-seed dominates 30/30, parser-safe EXTENSION 4-substrate static classification) sat in landing memos but were INVISIBLE to `tools/check_predecessor_probe_outcome.py` + `tac.probe_outcomes_ledger.latest_blocking_outcome_by_substrate`. Per CLAUDE.md "Subagent coherence-by-default": any future dispatch wrapper invoking these substrates would have observed `no_blocking_outcome` and proceeded to fire — re-running adjudicated probes the system had already answered. Net effect: $200-400 paid-GPU + 2-3 weeks BUILD investment risk per re-fired probe.

The Catalog #313 ledger is the structural extinction of THAT bug class. This backfill closes the §10.3 audit recommendation by ensuring each historical anchor is queryable by future subagents BEFORE they fire a redundant dispatch.

## Verdict semantics per Catalog #307 paradigm-vs-implementation discipline

All 6 verdicts are explicitly IMPLEMENTATION-LEVEL (pair #1, pair #2, null-byte REMOVAL, parser-safe SUBSET) or BOUNDARY-CHECK / METHODOLOGY-CLASSIFICATION (pair #4, parser-safe EXTENSION). NONE are paradigm-level kills. Per CLAUDE.md "Forbidden premature KILL without research exhaustion":

- **pair #1 + pair #2 DEFER** = procedural-codebook RESIDUAL-CORRECTION paradigm DEFERRED-pending-new-sister-equation `procedural_predictor_plus_residual_correction_savings_v1` (codex landed `procedural_predictor_residual_savings_equation_landed_20260521T010524Z_codex.md`). Canonical equation #26 `procedural_codebook_from_seed_compression_savings_v1` IN-DOMAIN scope refined to score-OPAQUE REPLACEMENT bytes only; sister equation will handle residual-hybrid contexts.
- **pair #4 PROCEED** = adopt routing rule "store procedural-codebook seed bytes RAW; route magic-codec ONLY to residual streams where there is an empirical byte stream to compress." Closed as paid-eval candidate.
- **null-byte REMOVAL DEFER** = H3_OPAQUE_TO_SCORER means the null-gradient bytes in fec6 are score-opaque AND parser-essential (PR101 magic header bit-essential at indices 0-7). Sister parser-safe SUBSET smoke `e3e198c9f` localizes WHY: 0/16,292 are parser-safe.
- **parser-safe SUBSET DEFER** = STRUCTURAL_EMPTY for current fec6 grammar; null-gradient signal remains valid for FUTURE archive designs exposing intermediate-transform regions by construction.
- **parser-safe EXTENSION PARTIAL** = methodology generalizes 4-substrate; all 4 carry parser-safe-but-score-affecting bytes (decoder side-information requiring co-trained procedural replacement, NOT direct substitution).

## Sister-DISJOINT verification with slot 2 (`a3632c82`)

Slot 2's scope is `.omx/state/canonical_equations_registry.jsonl` namespace (NEW canonical equations per Catalog #344). My scope is `.omx/state/probe_outcomes.jsonl` namespace (probe verdicts per Catalog #313). DIFFERENT JSONL files; DIFFERENT canonical helpers (`tac.canonical_equations.register_canonical_equation` vs `tac.probe_outcomes_ledger.register_probe_outcome`); ZERO file-overlap; ZERO function-call overlap. Sister-DISJOINT verified.

## Discipline footprint

Catalog #117 / #157 / #174 / #235 / #289 canonical serializer + POST-EDIT `--expected-content-sha256` + #119 Co-Authored-By trailer + #125 6-hook wire-in declaration + #185 META-meta-meta Live count post-flip drift detection + #186 catalog # claim via canonical serializer (no NEW catalog # claimed here) + #206 subagent crash-resume discipline (3 checkpoints emitted) + #229 PV (read 4 source memos + audit §10.3 + canonical helper signature pre-write) + #287 placeholder-rationale rejection (no waivers needed; ledger schema is structurally complete) + #313 probe-outcomes ledger (THIS LANDING) + #131 fcntl-locked JSONL APPEND-ONLY discipline + #138 fail-closed strict-load + #340 sister-checkpoint guard PROCEED + #110 + #113 APPEND-ONLY HISTORICAL_PROVENANCE (NO mutation of existing ledger rows; only NEW appends).

## 6-hook wire-in declaration per Catalog #125

- **#1 sensitivity-map:** N/A (canonical ledger backfill; no new sensitivity signal)
- **#2 Pareto constraint:** N/A (no Pareto-relevant rate/distortion deltas registered)
- **#3 bit-allocator:** N/A
- **#4 cathedral autopilot dispatch:** ACTIVE (future autopilot dispatch invocations against pair #1, pair #2, pair #4, null-byte REMOVAL, parser-safe SUBSET, parser-safe EXTENSION substrates now consult the canonical adjudicated verdicts via `latest_blocking_outcome_by_substrate`)
- **#5 continual-learning posterior:** ACTIVE PRIMARY (each backfilled row IS a posterior anchor for the Catalog #313 ledger; the entire purpose of this landing)
- **#6 probe-disambiguator:** ACTIVE (canonical ledger IS the disambiguator between "this probe has been adjudicated" vs "this probe is open for fresh dispatch")

## Operator-routable next-actions

1. **Cathedral-consumer wire-in audit**: verify that any future cathedral consumer touching `magic_codec_pair_*` / `pr101_gold_master_gradient_null_byte_removal_*` / `parser_safe_*` substrate tokens routes through `tac.probe_outcomes_ledger.latest_blocking_outcome_by_substrate(...)` before recommending dispatch. Catalog #313 + Catalog #335 sister bidirectional contract.
2. **Codex sister extensions**: codex landed `procedural_predictor_residual_savings_equation_landed_20260521T010524Z_codex.md` registering the NEW canonical equation `procedural_predictor_plus_residual_correction_savings_v1`; verify it carries an entry mapping pair #1 + pair #2 historical anchors as DOMAIN-MISFIT references (per Catalog #344 PERPETUAL_DOMAIN_REFINEMENT pattern).
3. **DP1 procedural paired-smoke recipe authoring**: per pair #2 + pair #4 cascade routing, the next frontier-moving work is DP1 procedural paired-smoke (Catalog #1098 territory). When authored, sister subagent SHOULD reference the 6 backfilled probe outcomes in design memo as PRIOR CONSTRAINTS.

## Blockers

None. All 6 registrations landed cleanly via canonical helper; ledger schema validated (88 rows, schema_version=1).
