# 12-Month Frustration Premortem and Pre-emptive Recommendations — 2026-05-16

**Subagent:** STRATEGIC FORESIGHT premortem
**Lane:** `lane_12_month_frustration_premortem_20260516`
**Horizon:** 2026-05-16 → 2027-05-16
**Cadence anchor:** Catalog #291 META-ASSUMPTION ADVERSARIAL REVIEW cycle (2026-05-16 event)
**Method:** Obituary-first reasoning — write the failure's headline; work backward to the preventable cause.
**Per Catalog #292 operating-within assumption disclosure:** "I am operating within the assumption that the contest, the operator, the Anthropic API, and the comma.ai upstream snapshot all continue to exist in materially recognizable form through 2027-05-16. If any of these assumptions falsifies (contest cancelled / operator burnout / Claude Code retired / upstream rotates scorer weights), large blocks of this premortem invert."

---

## Section 1 — Premortem framing and scope

It is 2027-05-16. The agent and the operator look back at the last twelve months. What does the obituary read? What got built, what rotted, what regret accumulated, what predictable disasters ambushed us?

Today (2026-05-16) the system has roughly the following surface:

- **2,441 lines of CLAUDE.md** with 41 NON-NEGOTIABLE markers, 22 FORBIDDEN-PATTERN entries, 22 HIGHEST EMPHASIS sections, 29 top-level `##` headings, and 191 numbered catalog gates (the most recent claimed is #297).
- **63,179 lines in `src/tac/preflight.py`** — the canonical strict-gate file. Average gate is ~150-200 LOC including AST helpers and dedicated test files. Catalog # numbering sits near 300; on the current cadence (~5-10 new gates per high-activity week) we are on a trajectory toward **catalog #500-#700** by 2027-05-16.
- **496+ lanes in `.omx/state/lane_registry.json`** (the WAVE-9 closure noted 496; subsequent waves likely pushed this past 700).
- **43 substrate trainer packages** under `src/tac/substrates/` (today) versus the **139 L1+ substrate entries** mentioned in the META-ASSUMPTION backfill audit. Substrate-trainer growth dominates LOC growth.
- **1,236 memory files** in the Claude memory directory; **850 of them are `feedback_*.md`** landings. MEMORY.md is already > 100K tokens and emits the explicit "WARNING: only part loaded" banner at session start.
- **1,606 `.omx/research/` ledger files** totaling 38 MB; **1,686 files under `.omx/state/`** totaling 487 MB (dominated by 2026-05-01 Lightning harvest JSONLs, individually 1.8-12.7 MB).
- **545 tools under `tools/`** (one-off + canonical helpers, no clear taxonomy boundary).
- **48 entries under `submissions/`** (including 13 `pr106_*_residual_sidecar/` family clones, multiple `_intake_` clones, and the canonical `exact_current/`).
- **249 GB under `experiments/results/`** — the dirty-disk DERIVED_OUTPUT trove. The 5,034 files under `.omx/` weigh 11 GB.
- **1,447 tests in `src/tac/tests/`** plus another 151 elsewhere. A single `pytest` run already costs several minutes when constrained to changed-scope and a long-running pass cannot complete in one operator-attention session.
- **Commit cadence: ~50 commits in the last 48 hours, 1,447 in the last week, 1,598 in the last two weeks** (subagent-attributed via serializer log). Two-week diff stats: 114 files, +10,757 / -1,250 LOC net.
- **`net GPU spend $0.10` for the 2026-05-12 close** — provider budget for paid dispatch is currently $0 across Vast.ai / Modal credits; Lightning free pool is the only no-cost lane. The frontier score on contest-CUDA is 0.193 (PR101 gold); on contest-CPU 0.195 (PR103 silver). PR107 apogee is 0.229 CUDA / 0.197 CPU.

These are not just metrics — they are the rotting substrate for every projection that follows. Twelve months of the current cadence, without structural change, produces a system whose growth has outpaced any single operator's or any single Claude session's ability to hold the whole thing in mind. The bottleneck is not "can we ship another substrate" — it is "can we still find anything six months from now."

---

## Section 2 — Failure category enumeration (15 categories)

For each: 12-month likely manifestation, frustration severity, probability, pre-emptive cost, recommended action.

### Category A — Catalog # exhaustion and gate-orchestration cost
**Operating-within assumption:** the cadence of "every bug surface gets a new STRICT preflight gate" continues at ~5-10 gates/week.

**12-month manifestation:** Catalog # crosses 600 by Q4 2026, 700+ by 2027-05-16. `preflight.py` swells past 120,000 LOC. Strict-mode `preflight_all()` execution time grows past 60 seconds even in `--scope dev` mode. The operator-authorize harness's 30-second budget breaks; dispatchers either start skipping preflight or get used to multi-minute pre-flights. New gates get added but reviewed only by their author. META-meta gates (#118, #159, #176, #185, #186, #289) themselves drift because nobody audits the META-meta surface. A future "phantom-row" incident (the Catalog #273-#278 phantom landings from 2026-05-15) becomes the rule, not the exception.

**Severity: CRITICAL. Probability: NEAR-CERTAIN. Cost to pre-empt: $0 (process change) + ~8 hours dev for the consolidation refactor.**

**Recommended action:** Introduce a **gate-consolidation discipline** before catalog #350: every new gate must either (a) be a META-meta gate that subsumes ≥3 sister-cases (one gate kills three bug classes), or (b) replace an existing gate that has been clean STRICT @ 0 for ≥30 days (retirement criterion). Add `tools/audit_gate_consolidation_opportunities.py` that ranks gates by (scope_overlap × consecutive_clean_days). Move the gate cadence from "additive forever" to "additive net of retirements." Also: pin a hard catalog # quota (e.g. "no new catalog # accepted without explicit operator approval after #400"). This forces consolidation.

---

### Category B — preflight.py file-size review-cost collapse
**Operating-within assumption:** the single 63K-LOC file remains the canonical strict-gate surface.

**12-month manifestation:** the file passes 100,000 LOC. A single subagent's edit takes 20+ minutes just to read the relevant context window. Concurrent subagent edits to preflight.py become the dominant commit-swap surface (the 2c957c31e + 8c9a5e7f incidents already showed the failure mode). The auto-commit hook stops being effective because the file consistently triggers the "file too big to review" path. New gates get added in non-canonical files because the canonical surface is too painful to edit.

**Severity: HIGH. Probability: LIKELY. Cost to pre-empt: ~6-10 hours refactor.**

**Recommended action:** Split `src/tac/preflight.py` into `src/tac/preflight/{core.py, gates_001_099.py, gates_100_199.py, gates_200_299.py, gates_meta.py, helpers.py}` with a single `preflight_all()` orchestrator that imports from the per-range modules. The split is mechanical (each gate is independent) but politically expensive — preserves Catalog #117/#157/#174 commit-serializer invariants by landing as ONE atomic commit, with explicit sister-subagent ownership map per Catalog #230 enforced.

---

### Category C — Lane registry size, search latency, schema drift
**Operating-within assumption:** lanes are append-only and `tools/lane_maturity.py` remains canonical.

**12-month manifestation:** registry passes 2,000 lanes. JSON parse cost noticeable in every preflight invocation (Catalog #90 `check_lane_registry_consistent` becomes the slowest gate). The 1 MB file size today becomes 5-10 MB. Cross-referencing a lane (e.g. "what lane registered the D4 fix?") requires manual `grep`. Schema additions over the 12 months (`distinguishing_feature_name`, `recipe_canary_status`, `target_modes`, `min_vram_gb`, `min_smoke_gpu`, `video_input_strategy`, `pyav_decode_strategy`, …) create per-row sparsity — fields exist only on lanes registered after their field was added. Queries that should be "find all L2+ substrates with operational mechanism declared" silently miss pre-2026-05-14 lanes that don't carry the field.

**Severity: HIGH. Probability: LIKELY. Cost to pre-empt: $0 (process discipline) + ~4 hours dev for the SQLite shadow.**

**Recommended action:** Land a `tools/lane_registry_to_sqlite.py` that materializes the JSON into a queryable SQLite database refreshed on every `lane_maturity.py mark`. The JSON remains the source of truth; SQLite is the read-side index. Add a `lane_registry_schema_version` field and a per-row `schema_version_at_creation` so future migrations are auditable. Also: institute a quarterly lane-registry archival cadence — terminal-status lanes older than 90 days move to `.omx/state/lane_registry_archive_<YYYYQQ>.json`.

---

### Category D — Memory file proliferation and MEMORY.md index rot
**Operating-within assumption:** every subagent lands a feedback memo and MEMORY.md indexes the top entries one-line.

**12-month manifestation:** memory directory exceeds 3,000 files. MEMORY.md grows past 1,000 indexed lines; only the top ~50 are read at session start because the 100K-token loader has long given up on the full file. The cross-reference graph rots — memo X references memo Y by name; Y gets renamed in a cleanup; X now points to nothing. The "session-warm context" the operator relies on for "we already learned this — see memo Z" silently degrades because Z has fallen out of the head-50 window. New subagents repeatedly rediscover lessons that the system already learned.

**Severity: HIGH. Probability: NEAR-CERTAIN. Cost to pre-empt: ~4 hours dev for indexer + retroactive cleanup.**

**Recommended action:** (1) Land a STRICT preflight gate that asserts every `feedback_*.md` referenced by MEMORY.md exists on disk (catches rename-rot). (2) Land `tools/memory_index_rotate.py` that quarterly bumps older entries to `MEMORY_ARCHIVE_<YYYYQQ>.md` while preserving full-text searchability via `MEMORY_SEARCH.md` (a flat list of `<filename>: <one-line description>` for grep). (3) Introduce a **semantic clustering pass**: every 90 days, group feedback memos by topic (substrate-bug-class / META-meta-gate / strategic-deliberation / operator-decision) and produce a single `MEMORY_CLUSTER_<YYYYQQ>.md` that compresses the cluster's learnings into 2-3 paragraphs. The detail stays in originals; the cluster is the new top-of-mind summary.

---

### Category E — Substrate proliferation outpacing dispatch capacity
**Operating-within assumption:** new substrates land as L1 SCAFFOLD continuously, and the dispatch wave is "next council decision."

**12-month manifestation:** substrate count crosses 200 by Q4 2026, 350+ by 2027-05-16. The vast majority are L1 SCAFFOLD with no contest-CUDA anchor. The "next dispatch decision" backlog grows to 50+ candidates; the operator's `next_experiments.md` becomes unusable. The cathedral autopilot ranker, designed to rank ~20 candidates, is over-fit on the 200+ scale. Every dispatch wave that DOES fire spends $20-100 chasing one substrate, leaving the other 199 untouched. The 0.193 frontier never moves because no candidate gets enough dispatch budget to actually train through a converged 5000+ epoch run. The "research-substrate trap" (8th forbidden pattern) becomes systemic.

**Severity: CRITICAL. Probability: NEAR-CERTAIN. Cost to pre-empt: $0 (operator-discipline) + ~6 hours dev for the substrate-pruning helper.**

**Recommended action:** Land a **substrate-retirement discipline**: any substrate at L1 SCAFFOLD that has not received a paid dispatch in 60 days is auto-marked `research_only=true` and removed from the cathedral autopilot ranking surface. Operator must explicitly opt-in via a 1-line `--reactivate-lane <id>` to bring it back. Pair with a **"new substrate requires retiring an existing one"** discipline (informal). Land `tools/audit_stale_l1_substrates.py` that produces the monthly retirement candidate list.

---

### Category F — Submissions directory hygiene and vendoring duplication
**Operating-within assumption:** every leaderboard PR replay or public-frontier intake spawns a new `submissions/<name>/` directory.

**12-month manifestation:** `submissions/` directory grows to 150+ entries (currently 48). Multiple intake clones of the same PR (e.g. `submissions/pr106_*_residual_sidecar/` already has 13 variants). Total `submissions/` size passes 30 GB. Many entries are stale leftovers from one-off bench scripts; nobody knows which `submissions/X/inflate.py` is canonical for replay vs which is a forensic snapshot. A single missing file in one of the 150 directories triggers Catalog #295's strict gate and refuses the entire repo's preflight. Vendored inflate.py copies drift from each other; bug fixes land in one and miss the other 12.

**Severity: MEDIUM. Probability: LIKELY. Cost to pre-empt: ~8 hours dev for the canonical-vs-vendored taxonomy.**

**Recommended action:** Move all non-canonical submissions to `submissions/_vendored/<source-pr>/...` with a top-level manifest listing canonical entries (`exact_current/`, `robust_current/`, current-frontier-shipper). Audit and dedup the 13 pr106 variants — pick one canonical, mark others as `_vendored/` snapshots with a `RELATIONSHIP_TO_CANONICAL: <delta>` README. Strict gate: any change to a vendored `inflate.py` requires `# VENDORED_INFLATE_EDIT_OK:<rationale>` waiver per Catalog #109 sister pattern.

---

### Category G — Test suite execution time scaling
**Operating-within assumption:** tests grow linearly with gates; pytest collection scans every test file.

**12-month manifestation:** test count crosses 5,000. Full `pytest` execution time passes 15 minutes; CI parallelization (when CI lands) only partially mitigates. Subagents skip the test gate on small commits ("I only edited one helper, no need for full pytest"); regressions slip in. The recursive adversarial review's 3-clean-pass gate becomes structurally hard to achieve because the test suite is slow enough that the gate-acquisition latency exceeds the operator's session-attention budget.

**Severity: HIGH. Probability: LIKELY. Cost to pre-empt: ~6 hours dev for test sharding.**

**Recommended action:** Introduce **pytest-distributed sharding** by gate-number band: tests for Catalog #1-#99, #100-#199, #200-#299, #META all run as parallel subprocess groups. Land `tools/run_tests_sharded.py` that fan-outs and aggregates. Pair with a **test-coverage marker discipline**: every test file declares which Catalog # it covers; the CI surface can target-test "just the gates touched by this commit" (changed-scope inference) and only run full-suite on weekly cadence.

---

### Category H — Provider lock-in, pricing, and credential rotation
**Operating-within assumption:** Modal / Vast.ai / Lightning continue to exist with current pricing and API surfaces.

**12-month manifestation:** at least one of the three providers either (a) significantly raises prices (50%+), (b) deprecates an API surface we depend on, (c) restricts free-tier access, or (d) goes bankrupt/acquired. Modal credits already exhausted (2026-04-15); Vast.ai account at $0 balance (2026-05-12); Lightning free pool is the single point of failure. The 11 Modal-specific dispatcher files, 545 tools/ scripts, and Catalog #153/#166/#244 gates are all tightly coupled to current Modal SDK. A Modal API change at the `fn.spawn()` surface breaks every dispatcher simultaneously. Secret rotation (Vast.ai keys, Modal tokens, Lightning auth) is undocumented; the operator has no runbook.

**Severity: CRITICAL. Probability: LIKELY. Cost to pre-empt: ~12-16 hours dev for the abstraction layer + ~2 hours documentation.**

**Recommended action:** (1) Land a **provider-abstraction layer** at `tac.deploy.providers.{modal,lightning,vastai,kaggle}` with a common `dispatch(trainer, recipe, gpu, max_seconds) -> DispatchHandle` interface. Modal-specific code consolidates behind one wall. (2) Land `docs/runbooks/credential_rotation.md` enumerating every provider's auth surface + rotation procedure. (3) Land a **monthly credential health check** workflow that verifies tokens are still valid and writes anchors to `.omx/state/credential_health.jsonl`.

---

### Category I — Anthropic API / Claude model deprecation
**Operating-within assumption:** Claude Opus 4.7 (1M context) remains generally available through 2027.

**12-month manifestation:** Anthropic deprecates Opus 4.7 in favor of Opus 5.x or Claude 5. The 1M-context flag becomes a paid premium tier. Rate limits tighten. A subagent prompt template that worked with 4.7 needs material reworking for 5.x. The shape of "checkpoint discipline" (Catalog #206 + #229) baked into prompts becomes obsolete. CLAUDE.md non-negotiables themselves are tuned to Claude 4.7's particular ways of forgetting and re-derive lessons; with Claude 5 the failure modes shift and some non-negotiables become vestigial while new bug classes (specific to 5.x) get no protection.

**Severity: HIGH. Probability: LIKELY. Cost to pre-empt: $0 (awareness) + ~4 hours dev for model-version pinning.**

**Recommended action:** (1) Add a `model_version` field to every subagent checkpoint and feedback memo so we can later distinguish "lesson learned under 4.7" vs "lesson learned under 5.x." (2) Add a CLAUDE.md non-negotiable noting that the existing FORBIDDEN_PATTERNS are calibrated against 4.7 behavior and need re-validation on model upgrade. (3) Land a `tools/claude_model_migration_check.py` skeleton that the operator runs the day Anthropic deprecates a model, and produces a per-non-negotiable "still applies / needs re-derivation" verdict.

---

### Category J — Single-operator bus-factor and hand-off readiness
**Operating-within assumption:** the operator (adpena) continues to be the sole human in the loop for all strategic and design-tradeoff decisions.

**12-month manifestation:** operator hits a 2-4 week gap (vacation, illness, life event, work pressure). When the operator returns, the in-flight session state is unreachable because there is no canonical "what would be productive to look at first" surface. The 1,236 memory files, 250+ in-flight L1 lanes, and the council deliberation backlog are unreadable without significant ramp-up time. If a third party (collaborator, postdoc, OSS contributor) needs to step in, they have no entry point — `CLAUDE.md` is for the agent, `README.md` is sparse, `PROGRAM.md` is 119 lines and 6 weeks stale (still names WILDE/SHIRAZ/GREEN as the architectural focus).

**Severity: CRITICAL. Probability: PLAUSIBLE-LIKELY. Cost to pre-empt: ~12-20 hours dev + writing.**

**Recommended action:** (1) Land `docs/HANDOFF.md` — a 1-page "if you are not adpena and need to understand the system, start here" with the canonical 5-minute orientation, the 10 most important non-negotiables, and the operator's current top-3 decisions awaiting resolution. (2) Land a `docs/SYSTEM_MAP.md` produced from `lane_registry.json` + `.omx/state/` + `submissions/` summary: the static surface diagram. (3) Update `PROGRAM.md` from the WILDE/SHIRAZ/GREEN era to current substrate-canvas era. (4) Add an `operator_return_brief.py` script that on demand produces "the 5-minute version of what happened while you were away" from commit log + lane registry deltas + cost-band posterior.

---

### Category K — Upstream contest repository drift
**Operating-within assumption:** `upstream/` snapshot is pinned and never changes; Yousfi / comma.ai don't change the scoring formula, dataset, or evaluator.

**12-month manifestation:** Yousfi rotates scorer weights to refresh the contest (mid-cycle). Or comma.ai changes the dataset video. Or `evaluate.py` adds a new component to the scoring formula. Any of these can silently invalidate every empirical anchor in `.omx/state/`, every score in MEMORY.md, every promotion-eligible lane in the registry. The forensic mismatch (old anchor on old weights vs new measurement on new weights) is hard to catch because nothing explicitly versions the upstream snapshot in our anchor records.

**Severity: CRITICAL. Probability: PLAUSIBLE. Cost to pre-empt: ~2-4 hours dev.**

**Recommended action:** (1) Stamp `upstream_snapshot_sha256` into every cost-band anchor, every contest-CUDA evidence record, every published score in MEMORY.md / reports. (2) Land `tools/check_upstream_snapshot_drift.py` that on each session start compares the current `upstream/` tree-hash to the most-recent anchor's hash and warns loudly if they differ. (3) Subscribe to or scrape the upstream repo for commits; alert on any change to `upstream/evaluate.py` or `upstream/modules.py`.

---

### Category L — Disk space, .omx/state JSONL growth, serializer log rot
**Operating-within assumption:** JSONL ledgers grow append-only and disk is "free."

**12-month manifestation:** `.omx/state/` passes 5 GB. The 2026-05-01 Lightning harvest JSONLs (already 12.7 MB each) become a class of "ancient artifacts that nobody dares delete but that nobody reads." The serializer log passes 50,000 entries; Catalog #117 / #157 / #174 / #206 / #289 gates that scan the log get slower (linear scan over 50K rows). `experiments/results/` (currently 249 GB) approaches 1 TB and saturates the operator's macOS SSD. The `.omx/` directory's 11 GB / 5,034 files becomes the dominant `git status` / `git diff` performance bottleneck even though most of it is gitignored.

**Severity: HIGH. Probability: NEAR-CERTAIN. Cost to pre-empt: $0 (operator-housekeeping) + ~4 hours dev for the archival helper.**

**Recommended action:** (1) Land `tools/archive_state_jsonl_quarterly.py` that moves JSONLs older than 90 days into `.omx/state/_archive/<YYYYQQ>/` (still searchable; just out of hot path). (2) Land `tools/audit_experiments_results_disk.py` that produces a monthly "top 20 largest dirs in `experiments/results/`" report + the canonical `gc_experiments_results.py` plan. (3) Audit Catalog #154 GC helper for performance — confirm linear scan over 249 GB completes in <30 sec or instrument it. (4) On the serializer log: cap log retention to last 90 days; archive older entries to a queryable parquet at `.omx/state/_archive/commit_serializer_<YYYYQQ>.parquet`.

---

### Category M — Council apparatus / sextet pact discipline drift
**Operating-within assumption:** the inner sextet (Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary) continues to be invoked rigorously for every council-grade decision.

**12-month manifestation:** the sextet ritual becomes performative. Memos start carrying the literal Catalog #292 section header but the per-member positions are 1-sentence rubber-stamps. The Assumption-Adversary role (added 2026-05-15) gets diluted into "Contrarian Lite." The grand reunion symposium (currently 116 KB single memo) becomes a quarterly event that nobody reads. New substrates ship with the assumption-classification section as a checkbox-compliance exercise rather than genuine interrogation. The 0.196-0.199 cluster — which the META-ASSUMPTION discipline was supposed to break — silently reappears as the 0.190-0.193 cluster a year from now.

**Severity: HIGH. Probability: LIKELY. Cost to pre-empt: $0 (process) + ~2 hours dev for the depth-of-deliberation auditor.**

**Recommended action:** Land `tools/audit_council_deliberation_depth.py` that scores recent council memos on (a) per-member position word-count, (b) presence of empirical receipts / paper citations, (c) explicit per-member assumption statement (Catalog #292), (d) recorded disagreements / votes. Produce a quarterly "council-discipline health" report. If depth scores trend down 3 quarters in a row, that itself is a signal to recharter the apparatus.

---

### Category N — Documentation rot and reports/latest.md staleness
**Operating-within assumption:** `reports/latest.md` is regenerated as part of session-close.

**12-month manifestation:** `reports/latest.md` is today already 4 days stale relative to its body content (R5-3 closure note from 2026-05-13 acknowledges it but does not regenerate). The lag becomes routine; nobody trusts the file. Other key docs (`PROGRAM.md` is 6 weeks behind reality already, `docs/superpowers/specs/2026-04-10-anti-drift-runtime-design.md` references frozen 4 weeks ago architecture) compound the staleness. The first time a new contributor or an outside reviewer looks at the public-facing docs they get a substantially false impression. The OSS v0.2.0-rc1 release becomes painful because every doc needs sanitization + freshness check.

**Severity: MEDIUM-HIGH. Probability: NEAR-CERTAIN (already happening). Cost to pre-empt: $0 (process) + ~2 hours dev for the staleness gate.**

**Recommended action:** Land a STRICT preflight gate that refuses `reports/latest.md` whose `last_refreshed_head` is more than 50 commits stale OR whose `last_refreshed_at` is more than 7 days stale, unless the file is regenerated. Land `tools/regenerate_reports_latest.py` that pulls from current state and writes the canonical structure. Add similar staleness gates for `PROGRAM.md` + key `docs/` runbooks.

---

### Category O — Frontier movement and contest-rule shifts
**Operating-within assumption:** the leaderboard is roughly static (0.193 PR101 gold; we sit at apogee 0.229).

**12-month manifestation:** Either (a) a competitor lands a paradigm-shift substrate (e.g. foundation-model-as-renderer, learned arithmetic coder, end-to-end joint codec) and the frontier moves to 0.10-0.15, leaving our entire research portfolio obsoleted; or (b) the contest rules change (Yousfi adds a new component, or the dataset is refreshed, or the time budget shrinks), invalidating our archive-grammar assumptions; or (c) the contest just ends (deadline lands), and we never converted apogee 0.229 into a sub-0.20 shippable.

**Severity: CRITICAL (depending on scenario). Probability: LIKELY for some manifestation. Cost to pre-empt: $0 (awareness) + dispatched discipline.**

**Recommended action:** (1) Implement the CLAUDE.md "Public frontier watch" non-negotiable RIGOROUSLY — automated polling of public PR list, alert on any new sub-0.193 claim, immediate intake + replay. (2) Pre-commit to a **shippability floor**: at any point we must have an archive + runtime ready to ship within 60 minutes that meets contest compliance, even if the score is mid-pack. The current state (no clean shippable closer to frontier than apogee 0.229) is fragile against deadline shock. (3) Treat the contest as "could end any week" and maintain submission escrow.

---

### Category P — Operator-decision backlog and decision-debt compounding
**Operating-within assumption:** operator-gated decisions get resolved roughly weekly.

**12-month manifestation:** decision backlog grows from current ~9 items (per `reports/latest.md` closure) to 30+ items. Many decisions interact (e.g. "Vast.ai balance reload" gates "sane_hnerv smoke dispatch" which gates "Wave 3 HNeRV-family build" which gates "TRADITION 2 substrate production targeting"). The interaction graph becomes unsolvable without the operator's full attention. Subagents start making "default" decisions that diverge from operator intent. Council deliberations land with "operator to choose" sections that pile up unread.

**Severity: HIGH. Probability: LIKELY. Cost to pre-empt: $0 (process) + ~3 hours dev for the decision queue helper.**

**Recommended action:** Land `tools/operator_decision_queue.py` that consolidates every `OPERATOR-GATED` decision across feedback memos, lane registry notes, and reports into a single dated JSONL. Surface the top-5 highest-value-blocked decisions at session start. Auto-flag decisions older than 14 days as "stale-pending" so the operator can either resolve, defer with a documented reason, or formally close.

---

## Section 3 — Top-10 highest-priority pre-emptive actions

Ranked by `(frustration_prevented × probability) / cost`.

| # | Action | Cost to land now | Frustration prevented | Severity × Prob | Type |
|---|---|---|---|---|---|
| 1 | **Land substrate-retirement discipline + `audit_stale_l1_substrates.py`** | $0 + ~6h | Prevents the 200+ stale L1 substrate trap; restores cathedral autopilot signal-to-noise | CRITICAL × NEAR-CERTAIN | Process + tool |
| 2 | **Stamp `upstream_snapshot_sha256` into every anchor + drift-check tool** | ~2-4h | Prevents the silent-invalidation-of-all-anchors disaster if Yousfi rotates scorer weights | CRITICAL × PLAUSIBLE | Tool + schema |
| 3 | **Provider abstraction layer (`tac.deploy.providers.*`) + credential rotation runbook** | ~12-16h + ~2h docs | Prevents the "Modal API change breaks every dispatcher simultaneously" cascade; documents secret rotation for bus-factor | CRITICAL × LIKELY | Refactor + docs |
| 4 | **`docs/HANDOFF.md` + updated `PROGRAM.md` + `docs/SYSTEM_MAP.md` + `operator_return_brief.py`** | ~12-20h | Closes the single-operator bus-factor risk; enables third-party collaboration; refreshes 6-week-stale public-facing docs | CRITICAL × PLAUSIBLE-LIKELY | Docs + tool |
| 5 | **Gate-consolidation discipline + retirement criterion + hard catalog # quota at #400** | $0 process + ~4h tool | Prevents Catalog # exhaustion + preflight.py review-cost collapse + phantom-row recurrence | CRITICAL × NEAR-CERTAIN | Process + tool |
| 6 | **Memory file rotation + cluster-summarization quarterly pass + reference-existence STRICT gate** | ~4h tool + ~2h backlog | Prevents memory-rot + cross-reference-rot; preserves "we already learned this" signal as scale grows | HIGH × NEAR-CERTAIN | Tool + gate |
| 7 | **Lane-registry SQLite shadow + quarterly archival + schema-version tracking** | ~4h tool | Prevents lane-registry search latency and schema drift across the 2,000-lane horizon | HIGH × LIKELY | Tool |
| 8 | **Reports / docs staleness STRICT preflight gate (Catalog #298 candidate)** | ~2h tool + ~1h gate | Prevents the "key docs silently rotted" failure already in progress (reports/latest.md) | MEDIUM-HIGH × NEAR-CERTAIN | Gate + tool |
| 9 | **Operator-decision queue helper + 14-day stale-flag + session-start surface** | ~3h tool | Prevents decision-debt compounding; surfaces interaction-graph deadlocks early | HIGH × LIKELY | Tool |
| 10 | **State JSONL quarterly archival + serializer log retention cap + experiments/results disk auditor** | ~4h tool | Prevents the disk-bloat / hot-path-slowdown failure mode; preserves Catalog #117/#157 gate performance | HIGH × NEAR-CERTAIN | Tool |

**Total estimated landing cost: ~60-90 hours of focused dev + ~$0 GPU spend.** Roughly 2-3 weeks of dedicated work or 6-8 weeks at 25% time.

---

## Section 4 — Pareto frontier: do everything vs do nothing

### Doing nothing (status-quo cadence)

By 2027-05-16, the realistic damage:
- **Catalog # north of 500**, `preflight.py` >100K LOC, gate-orchestration consuming >60 seconds per preflight, increasing skip-rate on small commits → bug class re-introduction.
- **Lane registry >1,500 lanes, mostly stale L1**; cathedral autopilot ranker producing noise → no dispatch wave fires the actually-promising substrate.
- **3,000+ memory files**, MEMORY.md unreadable, cross-reference graph 30% broken → repeated rediscovery of known lessons.
- **Provider lock-in incident** at least once (Modal pricing change or API deprecation) → 1-2 week emergency rebuild.
- **Operator burnout / 2-3 week gap** at least once → return-cost is multi-day re-orientation. If a third party tries to step in, they bounce off the docs and leave.
- **At least one contest-rule or upstream-snapshot change** silently invalidates 60%+ of empirical anchors; nobody notices until weeks later.
- **`reports/latest.md` is 8 weeks stale** the next time someone external looks at it; OSS release becomes a sanitization marathon.
- **Realistic frontier outcome:** still at apogee 0.229, no shippable sub-0.20 archive, contest deadline lands without escrow.

**Net damage estimate:** 200-400 hours of avoidable re-orientation, rediscovery, emergency rebuild, missed contest opportunities, plus ~$100-300 of wasted dispatch on the wrong substrates.

### Doing everything (all top-10)

Cost: ~60-90 focused dev hours over 6-8 weeks (parallelizable across sister subagents). Outcome: avoid the obituary; preserve the 12-month trajectory's optionality. Frees ~150-300 hours of avoided rework. Net win: ~100-200 hours saved + reduced cognitive load + a system that a third party can navigate.

### The optimal cut

Not all 10 are equally compressible. **Do items #1, #2, #4, #5, #6, #10** in the first 4 weeks (these have outsize compounding savings and / or close critical-path risks):

- **#1 substrate retirement** — restores autopilot signal
- **#2 upstream snapshot drift** — closes silent-invalidation catastrophe
- **#4 handoff / system map / PROGRAM refresh** — closes bus-factor
- **#5 gate consolidation discipline + quota** — bends the preflight curve
- **#6 memory rotation** — preserves "we already learned this" signal
- **#10 disk + serializer log archival** — preserves hot-path performance

**Skip / defer to Q3-Q4 2026:**

- **#3 provider abstraction** — high cost, deferred-OK if no provider incident in next 3 months; revisit on first incident
- **#7 lane-registry SQLite** — defer until registry hits 1,000 lanes (vs 700 today)
- **#8 reports staleness gate** — defer behind #4 (which forces the refresh anyway)
- **#9 operator-decision queue** — defer behind #4 (handoff already enumerates current decisions)

Optimal-cut total cost: ~30-45 hours. Captures ~70-80% of the avoidance value.

---

## Section 5 — Anti-patterns to avoid

Things the operator and Claude will be tempted to do that compound the problem. The "stop adding, start consolidating" line:

1. **Stop adding new STRICT preflight gates without retiring an existing one.** The Catalog # tally already exceeds the human ability to hold-in-mind. Every new gate should either replace an existing gate or subsume ≥3 existing gates. Hard cap at #400 unless explicit operator approval is recorded in `.omx/research/catalog_quota_exception_<YYYYMMDD>.md`.

2. **Stop adding new substrates without retiring one.** 43+ substrate trainer packages already; the ranker cannot prioritize 200+. New substrate adoption requires explicit retirement of an L1 SCAFFOLD that has been idle >60 days.

3. **Stop adding new councils.** Sextet pact + grand council + skunkworks + 20+ named seats is already the structural ceiling. Adding a "Compliance Council" or "Performance Council" splits attention. Use existing seats with specialized invocation.

4. **Stop adding `tools/<>.py` one-offs without a 90-day retirement plan.** 545 tools is already past the searchable threshold. Every new tool needs an entry in `tools/README.md` with a retirement criterion. Tools without a callsite in `tac.preflight`, `tools/operator_authorize.py`, or a feedback memo dated within 90 days are retirement candidates.

5. **Stop adding NEW memory file categories.** `feedback_*` is the canonical pattern; `project_*`, `feedback_grand_council_*`, `feedback_codex_*` are subcategories. Adding new patterns (`council_audit_*`, `decision_log_*`, etc.) fragments the index. Use existing patterns or extend MEMORY.md indexing tooling first.

6. **Stop adding new META-meta-meta gates (the Catalog #176 / #185 / #186 / #289 META-meta layer).** Each META-meta gate doubles operator cognitive load. The current layer is sufficient; investing in consolidation tooling beats adding more META-meta layers.

7. **Stop adding sister-subagent waves without explicit ownership maps.** Recent commit-swap incidents (8c9a5e7f, 2c957c31e) came from sister-subagent collisions. Catalog #230 + #157 + #174 protect at the commit surface, but the edit surface is still the source. Every multi-subagent wave needs the ownership map IN THE PROMPT, not "will sort out on review."

8. **Stop adding new operator-authorize wrappers.** 88 dispatch wrappers exist; Catalog #243 warn-only flagged them. Backfill the canonical routing first; do not add a 89th.

9. **Stop adding new research-only substrates** while 5 HIGH-RISK substrates per the META-ASSUMPTION backfill audit have not been unwound. Adding more research-only substrates inflates the L1 SCAFFOLD count which compounds Category E.

10. **Stop adding `feedback_*_LANDED_` memos that don't include the 5-line "what got retired or consolidated" section** — every landing should net-decrease repo entropy where possible. Pure-additive landings are the slow death.

---

## Section 6 — Operator-facing surprising findings (counterintuitive insights)

1. **The biggest 12-month risk is NOT contest-related — it's repository entropy.** The operator's mental model centers the frontier score (0.229 → 0.193). The premortem projection says repository entropy (catalog #, lane count, memory file proliferation, disk bloat) is more likely to be the cause of the next 6 months' frustration than any single substrate failing to converge. The 0.196 cluster might be unbreakable, but the system that's trying to break it becomes unmaintainable first.

2. **`reports/latest.md` is already a leading indicator and it is failing.** It is 4 days / 50+ commits stale per its own header. The system has the discipline to flag the staleness in the header but does not have the discipline to regenerate. This pattern will replicate across `PROGRAM.md`, `docs/HANDOFF.md` (when written), and every "session-warm summary" the operator counts on. The pre-emptive gate (Category N item) is high-leverage relative to its cost.

3. **The sextet pact and Assumption-Adversary seat (landed 2026-05-15) are at peak discipline RIGHT NOW.** Within 90 days, performative compliance will start eroding it. The Catalog #291/#292 gates protect the structural surface but the deliberation depth itself is uncatchable by gates — it's an audit-by-eyeball signal. Setting up the depth-audit tool (Category M item) at peak discipline (now) is much easier than retrofitting after the discipline has eroded.

4. **The 2026-05-12 cost-band posterior anchor was $0.10 net spend.** This is empirically the SMALLEST GPU envelope per actionable signal density we've ever achieved. The operator's instinct may be "we need more dispatch budget to break the frontier." The premortem projection says the actual scarce resource is `(operator attention × focused dev time)`, not GPU dollars. Reorienting the next session toward 30-45 hours of consolidation work (the optimal-cut from Section 4) will produce more 12-month-impact than another 10 paid dispatches.

5. **The "Race-mode rigor inversion" non-negotiable (CLAUDE.md line 7) is about to invert AGAINST us.** That rule was written for the May 4 2026 contest race window — a 4-hour event. The 12-month horizon has zero contest race windows (the contest deadline has passed; we're in a long-burn improvement loop). The rule's RACE-MODE flag (`.omx/state/RACE_MODE_ACTIVE.flag`) is dormant, but its psychological residue is still pushing "fan-out parallel actuator first, validate after" — exactly the wrong tradeoff for a no-deadline consolidation phase. Operator should explicitly stand down race-mode posture and embrace deliberation-first cadence.

---

## Section 7 — STRICT preflight gates to land NOW

### Catalog #298 candidate: `check_reports_latest_md_not_stale`

**Bug class:** key public-facing reports become stale relative to current state and silently mislead the next reader. Empirical anchor: `reports/latest.md` is 4 days stale + ~50 commits stale per its own header as of 2026-05-16.

**Scope:** `reports/latest.md` (and optionally `reports/*.md` matching `<canonical-report-name>.md` patterns). Refuses any state where:
- (a) the header's `last_refreshed_head` git sha is more than 50 commits behind `git rev-parse HEAD`, OR
- (b) the header's `last_refreshed_at` ISO timestamp is more than 7 days older than `datetime.utcnow()`,
UNLESS the file carries a same-line `# REPORT_STALENESS_OK:<rationale>` waiver in the first 30 lines (placeholder `<rationale>` literal rejected).

**Acceptance:** the canonical `tools/regenerate_reports_latest.py` (to be landed in the same commit batch) refreshes the header AND body. The header MUST include `last_refreshed_head`, `last_refreshed_at`, `regenerated_by` fields.

**Live count at landing:** 1 (the current `reports/latest.md`). Strict-flip atomic with the regeneration commit per CLAUDE.md "Strict-flip atomicity rule."

**Estimated landing cost:** ~3 hours (gate ~1h + regeneration tool ~2h).

**Sister of:** Catalog #113 (artifact lifecycle compliance — DERIVED_OUTPUT header requirement) + Catalog #185 (LIVE_COUNT drift detection — same META class). Together they extinct the "DERIVED_OUTPUT silently rotted into stale-but-still-cited evidence" failure mode.

### Catalog #299 candidate: `check_substrate_lane_l1_scaffold_not_stale_dispatch`

**Bug class:** L1 SCAFFOLD substrate lanes accumulate in the registry without dispatch, polluting the cathedral autopilot ranker surface and creating decision overload. Empirical anchor: the META-ASSUMPTION backfill audit's 5 HIGH-RISK substrates + the WAVE-9 closure's ~12-20 L1 SCAFFOLD substrates with no paid dispatch.

**Scope:** `.omx/state/lane_registry.json`. For every lane matching `lane_class` in {`substrate_*`} OR id-substring per Catalog #220 in-scope list, AND level >= 1, AND impl_complete gate truthy, refuses if:
- The lane has no `successful_dispatch` outcome in `.omx/state/cost_band_posterior.jsonl` AND
- The lane's `impl_complete` evidence timestamp (or `add-lane` timestamp from `lane_maturity_audit.log`) is more than 60 days old AND
- The lane does NOT carry `research_only=true` OR `lane_class=substrate_engineering` OR the same-line `# L1_STALE_DISPATCH_PENDING_OK:<rationale>` waiver (placeholder rejected).

**Acceptance:** the operator either dispatches the lane (the desired outcome), explicitly marks `research_only=true`, or waives with a rationale. The gate forces an explicit decision rather than letting the lane sit indefinitely.

**Live count at landing:** ESTIMATE 10-25 (the L1 SCAFFOLD substrate inventory from WAVE-9). Initial wire-in MUST be WARN-ONLY per CLAUDE.md "Strict-flip atomicity rule" because the operator-routed backfill sweep is itself the strict-flip atomic. Strict-flip after the sweep drives count to 0.

**Estimated landing cost:** ~4 hours (gate + dispatch-history join logic + WARN-ONLY initial commit + dispatcher integration with the cost-band posterior).

**Sister of:** Catalog #220 (substrate L1 scaffold operational mechanism) + Catalog #240 (recipe-vs-trainer state consistency) + Catalog #272 (distinguishing feature integration contract) + Catalog #233 (L1→L2 promotion canonical). Together they extinct the "L1 SCAFFOLD substrate accumulates indefinitely without dispatch resolution" failure mode that drives Category E.

---

## Final assessment

Twelve months from now, the failure category most likely to dominate the operator's frustration is NOT contest-related, NOT substrate-engineering, NOT provider-pricing. It is **repository entropy** — the cumulative weight of catalog #, lane count, memory file count, disk bloat, and stale-doc-rot, multiplied by the single-operator bus-factor.

The good news is that all 10 of the top-priority pre-emptive actions are within the operator's + Claude's existing capability surface. None require new providers, new model versions, new contest cooperation. The optimal cut (~30-45 hours of focused consolidation work) captures the majority of the avoidance value and can be parallelized across sister subagents.

The deeper good news is that the disciplines we've built — Catalog #229 (premise verification), #206 (checkpoint discipline), #230 (sister-subagent ownership maps), #291/#292 (META-ASSUMPTION cadence), #248 (no stash-pop conflict markers), #244 (canonical NVML block) — these are EXACTLY the tools the consolidation work needs. The system has the antibodies; it just needs to apply them to itself.

The thing we will most regret 12 months from now NOT having done today is: **stop adding, start consolidating, and write down the system for someone who is not us.**

---

*Generated by `subagent:12_month_premortem_20260516` per operator-orchestrated STRATEGIC FORESIGHT premortem. Per CLAUDE.md "Apples-to-apples evidence discipline" every projection tagged `[projection]`; every cost `[estimate]`. Per Catalog #229 7 premise verifications confirmed pre-edit (CLAUDE.md size + non-negotiable count, PROGRAM.md staleness, MEMORY.md size + index proliferation, lane_registry.json size + lane count, .omx/state disk usage + JSONL inventory, recent memo cadence + content sample, recent commit velocity + diff stats). Per Catalog #206 3 checkpoints filed. Per Catalog #230 disjoint scope (writes only to `.omx/research/` + Claude-memory landing memo). Per Catalog #292 operating-within assumption disclosed in Section 1 framing.*
