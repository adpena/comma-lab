---
title: "Cerebras knowledge-base mechanisms crossed against Pact's memory apparatus"
date: 2026-07-19
author: codex
research_only: true
lane_id: lane_cerebras_kb_crosswalk_566_20260719
review_status: "fresh-eyes-reviewed: 5 rounds; final 3 consecutive clean"
verdict_scope: "KNOWLEDGE-APPARATUS DESIGN ONLY; no launch, score, provider, or frontier authority"
pointer_delta: "UNMOVED — 0.1910828242 [contest-CPU Linux x86_64]"
source_status: VERIFIED_VIA_SOURCE_INSPECTION
---

# Verdict first

**ADOPT four bounded changes, not a wholesale Cerebras clone:** (P0) fail-closed source/completeness
manifests for graph and corpus builds, (P0) one fused lexical + semantic + graph ranking surface,
(P0) explicit freshness/tombstone/supersession policy, and (P1) a gold-query retrieval evaluation
suite. Pact is already materially stronger in typed provenance, named triggers, supersession edges,
equation-to-empirical-anchor recalibration, and DAG/DSL/equations/tasks structural governance.
Cerebras is stronger in source ingestion discipline and the query-time retrieval cascade.
The table's `ADOPT` rows are subrequirements or candidate strategies within those four bounded
families; the inverse section separately preserves Pact-native consumption controls.

The first P0 is not hypothetical. A fresh `graph_memory_recall.py --stats --json` in this linked
worktree produced **3,157 nodes / 4,856 edges**, while the read-only canonical-root cache contains
**9,704 nodes / 32,156 edges**. `src/tac/graph_memory/build.py::_memory_dir()` derives the Claude
memory slug from the linked-worktree path and `parse_memory_files()` silently returns zero when that
directory is absent. The cache publisher has no source-count manifest or collapse refusal. Labels:
both counts are **MEASURED (local filesystem snapshot, 2026-07-19)**; the worktree-path cause is
**VERIFIED_VIA_SOURCE_INSPECTION**. This is the live `index-partial-load` / goldfish failure class.

`research_only=true`: this memo changes no executable apparatus and creates no empirical score
anchor. Its named consumer is **MAIN landing review**, which should route the four accepted items to
owned implementation tasks. Triality legs are deliberately unchanged under `[no-triality]`; no
campaign lever or measured witness finding was created.

## Source custody and epistemic labels

- Primary source: Cerebras, [“How We Built Our Knowledge Base”](https://www.cerebras.ai/blog/how-we-built-our-knowledge-base), dated 2026-07-15. The detail route returned HTTP 500 during review, so the full first-party article was decoded from the official blog listing's embedded Sanity/RSC document at [cerebras.ai/blog?upage=5](https://www.cerebras.ai/blog?upage=5). Document id `d6088159-7525-4430-928f-3322c6e06b5b`; revision `Vp3FMfUz37QgrX4nooL6bB`; decoded JSON 40,992 bytes; SHA-256 `52826fdba8bf2a1f914b3e5b1bc01bf8f7c93ad39213483b6f7f3daa22aa2b6c`.
- **VERIFIED_VIA_SOURCE_INSPECTION** means the article says or describes the item. It does not mean this review reproduced Cerebras's system.
- Cerebras's “15,000 questions a day,” “within three months,” and qualitative accuracy claim are **COMPANY-REPORTED**, with no sampling frame, baseline, quality metric, confidence interval, latency, cost, or ablation in the post. They are not promoted to independent measurements here.
- Pact code/ledger counts below are **MEASURED (local snapshot, 2026-07-19)**. Design consequences are **DERIVED** and carry falsifiable gates.
- Review method follows `docs/operating_manual_craft_handoff.md`: re-derive from primary artifacts, label claims, and attack the conclusion before handoff (especially §§4–6).

## Ranked atomic crosswalk

The rank is Pact adoption priority, not the order in the article. Every row has one Cerebras
mechanism (or an explicitly disclosed omission), why it exists, the exact Pact surface, and a scoped
decision. `ALREADY-COVERED` does not mean identical; the final clause states where Pact is stronger.

| Rank | Cerebras mechanism and why | Pact first-hand surface | Decision, concrete action/gate, and incident | verdict_scope |
|---:|---|---|---|---|
| 1 | **Common ingestion contract with per-source fetch cadence.** Every source declares what data it supplies, how to connect, and how often to refresh; custom connectors emit common rows. This prevents a new silo per tool. | Graph build declares five source families in `src/tac/graph_memory/build.py:4-16`; `tools/corpus_query.py:33-35,202-228` declares seven stores. They do not emit a per-source health/watermark manifest. | **ADOPT.** Add `SourceSnapshot{name, canonical_root, units_seen, bytes_seen, latest_source_mtime, last_success, parser_version, required}` to both builders; atomically publish only with it. **Gate:** a linked worktree must reproduce canonical source counts within an explained append-only delta, and any previously nonzero required source becoming zero refuses cache replacement. Fixes the **MEASURED 9,704/32,156 → 3,157/4,856 partial-load collapse**. | `KNOWLEDGE-INDEX COMPLETENESS`; not a score verdict. |
| 2 | **Hybrid ranking per Slack thread:** exact full-text for error strings/flags/hosts, embeddings for paraphrase, inverse-document-frequency emphasis for rare tokens, and an age-decay view; fuse ranked lists. Each signal repairs a different miss class. | `tools/corpus_query.py:231-247` is deterministic lexical density/distinctness/recency; `tools/graph_memory_recall.py:3-21` is a separate reconstructive traversal. No common semantic rank or fusion layer is invoked by the recall-before-decide contract. | **ADOPT.** Make one typed `RecallEvidence` surface that runs lexical and graph paths plus an optional local embedding path, then reciprocal-rank fuses them. **Gate:** on a frozen gold-query set, fused top-10 must weakly dominate each constituent on supported-hit recall and never omit an exact identifier hit; latency and source coverage are recorded. Fixes **apparatus-writes-better-than-it-reads** and separated-path miss risk. | `PACT DURABLE-CORPUS RECALL`; embedding choice remains implementation-open. |
| 3 | **Freshness at two speeds:** Slack events update immediately; channels retain explicit freshness cadence; code sync reprocesses only changed chunks per commit. The why is avoiding stale answers without rebuilding everything. | `src/tac/graph_memory/__init__.py:89-121` rebuilds when cache is absent/stale by aggregate mtime. Canonical registries are append-only/latest-event-wins. There is no source-specific watermark surfaced with a hit. | **ADOPT.** Attach `observed_at`, source revision/content hash, freshness SLA, and stale reason to every evidence row; expose a source-specific incremental rebuild plan. **Gate:** mutate one source fixture and prove only its units change; withhold a hit once its SLA is exceeded unless the caller opts into stale evidence. Fixes silent stale-source answers and the current all-or-nothing cache test. | `INDEX FRESHNESS`; no claim that every source needs realtime ingestion. |
| 4 | **Retrieval evaluation is implied by adoption, not demonstrated.** The post reports 15,000 questions/day after three months and says distilled Slack embeddings improved accuracy “significantly,” but gives no before/after values, judge protocol, recall, citation support, staleness, latency, cost, or ablation. | Pact tests mechanics, schemas, and guards, but this review found no canonical gold-query benchmark spanning `corpus_query` + graph reconstruction + supersession. | **ADOPT.** Land a versioned query set containing exact IDs, paraphrases, chronology, superseded claims, cross-store synthesis, orphan findings, and partial-source failures. **Gate:** publish recall@k, supported-answer rate, stale/superseded-hit rate, source coverage, p50/p95 latency, and an ablation table for lexical/semantic/graph/rerank. Fixes repeated rediscovery without replacing it with unmeasured “better retrieval” claims. | `RETRIEVAL QUALITY`; Cerebras usage is adoption evidence only. |
| 5 | **Thread-level LLM distillation before embedding:** store a likely query, short summary, resolution, and systems/code references rather than embedding the raw transcript. This reduces length/noise and makes answers retrievable by intent. | Memory frontmatter provides `name`, `type`, `description`; `[[links]]`, sister, supersedes, file refs, equation refs, task refs, and FEED blocks form typed structure (`src/tac/graph_memory/build.py:83-148,181-235`). `MEMORY.md` is constrained under 17KB and established-findings clusters consolidate detail. | **ADOPT.** Bounded action: add a deterministic `retrieval_card` schema for durable memos—question/claim/verdict/scope/evidence refs/supersedes/consumer—generated or human-authored but content-hash bound. **Gate:** cards must improve gold-query recall without reducing exact-claim support; raw source remains the citation authority. Fixes prose-density and the 178KB index partial-load incident without trusting lossy summaries as truth. | `MEMO RETRIEVAL REPRESENTATION`; no LLM-authored verdict authority. |
| 6 | **Reciprocal-rank fusion:** add `weight/(60+rank)` across result lists, with article weight 1.0 and smoothing 60. It combines incomparable rank scales. | No shared rank-fusion implementation between `corpus_query` and graph memory. `corpus_query` uses its own scalar score; graph reconstruction uses seeds/traversal. | **ADOPT.** Treat it as a candidate, not a sacred constant: implement the formula behind a typed strategy enum alongside constituent-only modes. **Gate:** `k=60`, weights, and alternatives are selected only by the frozen evaluation suite; exact-ID preservation is hard. Fixes arbitrary cross-retriever score normalization. | `RANK-FUSION FORMULATION`; the Cerebras constants are source-reported, not Pact-optimal. |
| 7 | **Small reranker after broad retrieval:** score original query against candidates on a 0–10 scale and retain ten. This spends expensive reasoning only after cheap recall. | Costate digest invokes `corpus_query` under a two-second bounded path and retains top hits; it has no independent relevance/support rerank. | **ADOPT.** Sequence after ranks 1–4: add a local deterministic or pinned-model rerank adapter whose output includes model/hash/prompt version and leaves raw ranks visible. **Gate:** supported-answer recall@10 improves over fusion, exact identifiers are never dropped, and offline/no-model mode remains valid. Fixes high lexical scores for incidental term repetition observed in this review's top result. | `POST-RETRIEVAL ORDERING`; no network/model dependency may become mandatory. |
| 8 | **Deduplicate back to the source, cap results per file, and keep a diverse top 20.** The why is stopping one verbose file from crowding out other evidence. | `corpus_query` caps unit sizes and top results, but its returned list can contain many units from one source family. Graph reconstruction caps seeds/nodes/depth, not evidence diversity by source. | **ADOPT.** Add deterministic per-source and per-document caps plus a recorded diversity reason after fusion, before rerank. **Gate:** a synthetic dominant document cannot occupy more than the configured cap, while an exact-ID query may explicitly override the cap. Fixes giant-DAG/index crowd-out. | `RESULT-SET DIVERSITY`; caps must not suppress exact custody. |
| 9 | **Contextual expansion after ranking:** when a wiki chunk wins, add its two neighboring chunks. This restores context lost during chunking. | Graph neighbors are a first-class typed query and reconstructive traversal (`tools/graph_memory_recall.py:54-64,151-158`); Obsidian export preserves synthesized relations. | **ALREADY-COVERED.** Make the existing graph-neighbor expansion a mandatory post-fusion option and tag why each neighbor was included. Pact is stronger because edges are typed (`supersedes`, `produces`, `consumes`, `references`), not merely positional. | `Pact graph-backed documents`; positional neighbors still useful for unstructured docs. |
| 10 | **Planner chooses tools and sources from a compact project/source/capability index.** This avoids searching everything for every query. | `corpus_query --stores` and seven graph query tools expose explicit choices; recall-before-decide is enforced in `src/tac/subagent_contract.py:136-142`; costate digest invokes retrieval with “the corpus knows.” No learned/free-form planner is needed to access them. | **ALREADY-COVERED.** Wire the source/completeness manifest into the existing typed chooser rather than adding an opaque planner. Pact is stronger in deterministic bounded queries and enforceable `STORES CONSULTED`; Cerebras is stronger in automatic source selection. | `SINGLE-OPERATOR PACT PLANNING`; revisit planner learning only after gold-query evidence. |
| 11 | **Parallel fan-out to selected tools, common evidence records, then synthesis.** It lowers wall-clock and makes cross-source answers possible. | Corpus query already scans all selected stores in one bounded invocation; graph reconstruction assembles connected context. Their result schemas differ, so synthesis has no single custody envelope. | **ADOPT.** Keep the already-present bounded multi-store scan and add the common evidence schema. **Gate:** every synthesized claim links to one or more immutable evidence rows including store, ref, content hash, observed time, rank path, and supersession state. Fixes citation-by-memory and cross-tool evidence loss. | `READ-ONLY KNOWLEDGE QUERY`; no agent fan-out requirement. |
| 12 | **Direct MCP primitives stay narrow, stable, and as LLM-free as possible; the client orchestrates.** This keeps retrieval reproducible and reusable by humans and agents. | `tools/corpus_query.py` and seven typed graph tools are narrow CLI primitives with JSON output; the graph build is deterministic and has no RNG (`src/tac/graph_memory/build.py:16`). | **ALREADY-COVERED.** Preserve CLI/library interfaces and add the evidence/completeness schema below them. Pact is stronger in local/offline determinism and source-grounded typed edges; an MCP wrapper would add no current single-operator value. | `LOCAL PACT TOOLING`; protocol transport is not the capability. |
| 13 | **One Postgres table holds embeddings, raw summaries, and metadata across all sources.** This enables one query path and consistent filtering. | Pact deliberately uses source-of-truth files plus derived indexes: append-only JSONL registries, Markdown DAG/memory/deferrals, graph cache, and corpus iterators. | **ALREADY-COVERED.** Pact uses a federated common read model, not storage centralization. Do not migrate authority into one database; add one normalized `RecallEvidence` view. Pact is stronger because source artifacts remain auditable and reconstructable rather than becoming copies in a new authority store. | `PACT SINGLE-REPO AUTHORITY`; storage engine choice is non-authorizing. |
| 14 | **Raw Slack text becomes immediately keyword-searchable through a Postgres GIN index before semantic enrichment.** Exact search is available even if enrichment lags. | `corpus_query` reads authoritative text directly; `rg` remains the exact code/text fallback; semantic enrichment is not required for baseline recall. | **ALREADY-COVERED.** Preserve exact lexical availability as a hard fallback when graph/embedding/rerank stages fail. Pact is stronger in zero-ingestion direct-source search; Cerebras is stronger in event-driven indexing latency. | `LOCAL FILE CORPUS`; no Slack claim. |
| 15 | **Stable event IDs, acknowledgements, and deduplication on Slack Socket Mode.** This creates idempotent ingestion under retries. | Append-only registries use stable domain IDs/latest-event semantics and locked writes; canonical equations keep equation and anchor identifiers. Memory/DAG Markdown ingestion lacks a universal event ID. | **ADOPT.** Preserve the already-strong typed registry identities and add content-addressed unit IDs for Markdown indexing. **Gate:** ingesting identical bytes twice yields identical node/evidence IDs and no duplicate edges. Fixes repeated-block/duplicate-memo crowding. Pact remains stronger in domain-specific identities and provenance. | `MARKDOWN INDEX IDEMPOTENCE`; not a realtime connector design. |
| 16 | **On any Slack message, re-fetch the parent plus all replies and write the whole thread.** Thread is the atomic knowledge unit because message-level updates lose resolution context. | Pact treats a memory file or DAG FEED block as the unit; graph edges restore linked context. | **ALREADY-COVERED.** Keep file/FEED atomicity and add content-hash revision lineage. Pact is stronger where a FEED block carries verdict scope and provenance; no Slack thread exists. | `PACT MEMO/FEED UNIT`; Slack-specific mechanics excluded. |
| 17 | **Burst reconstruction:** consecutive messages by one author are joined and prefixed with the thread topic. This repairs chat fragmentation. | Durable Pact sources are already authored documents/blocks rather than chat-event streams. | **NOT-APPLICABLE.** The org-scale chat-fragment problem is outside the current single-operator durable-file corpus; `verdict_scope=PACT CURRENT AUTHORITATIVE STORES ONLY`. Reopen only if chat becomes an authority source. | `PACT CURRENT AUTHORITATIVE STORES ONLY`. |
| 18 | **Burst admission uses a weighted quality threshold:** its signals include rare-token IDF (the article gives 4.0), length (200 characters), and reaction-based social boost; qualifiers are stored with the thread. This controls chat noise. | Pact admission is typed/contractual (frontmatter, ledger schema, claim labels, task/lane ownership), not social engagement. | **NOT-APPLICABLE.** Reaction popularity is an org-chat heuristic and must not authorize Pact evidence; `verdict_scope=PACT EVIDENCE ADMISSION`. The reusable idea—store admission reasons—is already covered and should remain deterministic. | `PACT EVIDENCE ADMISSION`. |
| 19 | **Code ingestion uses CocoIndex after experiments, including repositories over 40GB.** The why is incremental scale without writing a bespoke indexer. | Pact code search is direct `rg`; the knowledge index primarily covers research, memory, equations, tasks, deferrals, and DAG. No evidence in this review shows code-search scale as the binding miss. | **NOT-APPLICABLE.** For now, org-scale/multi-repository semantic code indexing exceeds the current single-repository need; `verdict_scope=PACT CURRENT CODE-RETRIEVAL BOTTLENECK`. Reopen if the gold-query suite demonstrates lexical code misses or indexing latency. | `PACT CURRENT CODE-RETRIEVAL BOTTLENECK`. |
| 20 | **Language-specific regex split coarse-to-fine:** file/class/function/smaller blocks, permitting multi-level embeddings for one file. This preserves semantic boundaries. | Pact's DAG parser splits at FEED headers; memory uses file/frontmatter; JSONL uses rows. Source-code search is lexical. | **ALREADY-COVERED.** For current knowledge types, keep type-specific parsers rather than a universal token chunker. Pact is stronger because units map to domain objects; the code-specific class/function layer remains scoped out by row 19. | `DURABLE KNOWLEDGE STORES`; source-code embeddings excluded. |
| 21 | **Incremental code sync stores commit metadata and re-embeds/re-exports changed chunks only.** This bounds update cost and exposes revision. | Registries append events; graph cache checks aggregate source mtime and may rebuild everything. | **ADOPT.** As part of rank 3, persist per-unit source hash/revision and reuse unchanged derived rows. **Gate:** one changed fixture causes exactly its derived unit and incident edges to update, with an identical full rebuild hash. Fixes unnecessary rebuilds while preserving deterministic reconstruction. | `DERIVED INDEX MATERIALIZATION`; canonical sources stay unchanged. |
| 22 | **Repository onboarding is self-service configuration by pull request with path allow/deny rules.** Ownership and review happen through normal code review. | Pact stores are hard-coded in builders; canonical-doc registry finds all refs/content and tracks status/supersession. Lane/task ownership and serializer review govern change. | **ADOPT.** Reuse the already-strong review/ownership path, but put source declarations and allow/deny rules in a reviewed typed manifest. **Gate:** an undeclared source cannot silently enter the index; every declared source has owner and health. Pact remains stronger in serializer/MAIN landing review and typed canonical-doc supersession. | `INDEX SOURCE CONFIGURATION`; no organization-wide self-service promise. |
| 23 | **A custom source is a Python plugin plus a matching source entry emitting the common table shape.** The rest of the pipeline remains unchanged. | `_iter_store` and graph parsers are explicit source adapters but not a typed plugin registry. | **ADOPT.** Take the interface, not dynamic plugin loading: define a static typed adapter protocol registered in source control. **Gate:** contract tests run completeness, deterministic IDs, source hash, freshness, and malformed-row behavior against every adapter. Fixes per-source drift. | `IN-REPO STATIC ADAPTERS`; runtime third-party plugins excluded. |
| 24 | **Named projects are scoped bundles of sources; the same source may appear in several projects without duplication.** Searching everything degraded quality, so scope is a retrieval primitive. | `corpus_query --stores`, graph typed tools, lane IDs, task IDs, and topic/entity queries already scope recall without copying sources. | **ALREADY-COVERED.** Add named query profiles only if repeated store/topic sets appear; never duplicate canonical bytes. Pact is stronger in lane/task/topic typed scopes; Cerebras has a more polished user-facing project abstraction. | `SINGLE-OPERATOR QUERY SCOPING`. |
| 25 | **A default project is saved per user at onboarding and automatically scopes later queries.** This removes repeated setup. | Pact has one operator and session preflight derives current lane/state from canonical pointers; persisting a user default could hide cross-lane evidence. | **NOT-APPLICABLE.** Multi-user onboarding convenience is outside current single-operator scope; `verdict_scope=PACT SINGLE-OPERATOR SESSION DEFAULTS`. Explicit current lane/topic is safer than a hidden sticky project. | `PACT SINGLE-OPERATOR SESSION DEFAULTS`. |
| 26 | **Specialized query tools:** subsystem per-file summaries, unified search, Slack search, code `ripgrep`, recent pull requests, and `who_knows`. Different questions need different evidence. | Graph exposes time/keyword/entity/topic/decision/neighbor/supersession tools; corpus query covers research/equations/memory/DAG/council/tasks/docs; git/`rg` cover code/history. | **ALREADY-COVERED.** Add source health and normalized evidence rather than duplicating tools. Pact is stronger in decision/supersession/neighbor queries; Cerebras is stronger in human expertise discovery. | `PACT TOOL INVENTORY`; human-directory discovery excluded. |
| 27 | **Web UI runs planner/executor/synthesis and shows citations, caveats, and cross-source synthesis.** It makes evidence legible to non-tool users. | Pact operator surfaces are CLI/Markdown/costate digests; the memo/receipt culture requires evidence and caveats but has no dedicated KB UI. | **NOT-APPLICABLE.** For now, an org-facing UI does not fix the single-operator retrieval defects; `verdict_scope=PACT CURRENT KNOWLEDGE-CONSUMPTION BOTTLENECK`. Reopen after quality/completeness gates if the operator needs interactive exploration. | `PACT CURRENT KNOWLEDGE-CONSUMPTION BOTTLENECK`. |
| 28 | **Authentication, authorization, audit, and analytics are named as one of three KB responsibilities.** The post provides no schema, enforcement path, or measurements. | Repository permissions, lane ownership, serializer landing, source custody, and append-only audit ledgers are explicit. | **ALREADY-COVERED.** For local authority boundaries, Pact is stronger because review/custody are executable and artifact-linked; do not infer Cerebras implementation detail from one sentence. | `LOCAL REPOSITORY KNOWLEDGE`; enterprise identity/access excluded. |
| 29 | **Graph retrieval/linking is absent from the described Cerebras query pipeline.** It merges ranked retrieval and neighboring wiki context but does not describe entity/relation traversal or reconstruct-not-retrieve memory. | Task #411 explicitly reconstructs a graph over `[[links]]`, FEED refs, equations, tasks, deferrals, producers/consumers, sister, and supersedes; seven typed tools and Obsidian round-trip expose it (`tools/graph_memory_recall.py:3-21,54-76`). | **ALREADY-COVERED.** Keep graph as a co-equal constituent in fusion, never replace it with embeddings. Pact is stronger in relational reconstruction, but its completeness guard is weaker (rank 1). | `CEREBRAS POST DESCRIPTION`; absence in the post is not proof their organization has no graph elsewhere. |
| 30 | **Explicit supersession/tombstones and stale-claim refusal are absent from the post.** Age decay makes newer content rank higher but does not say an older contradicted claim is invalid. | Graph has `supersedes` edges and a typed supersession query; canonical docs have active/superseded/draft states and `superseded_by`; registries are latest-event-wins. | **ADOPT.** Use the already-present structural supersession to enforce retrieval currentness. **Gate:** a superseded fixture can appear only as historical provenance and can never synthesize as current without an explicit historical query. Fixes stale negative/verdict reuse. Pact remains stronger in typed supersession and verdict scope. | `RETRIEVAL-TIME CURRENTNESS`; absence limited to the article. |
| 31 | **Ownership/curation is mostly connector configuration by pull request plus project scoping.** The post does not specify claim owners, review cadence, contradictory evidence adjudication, or deletion/tombstone policy. | Lane registry, canonical tasks, named deferral triggers, curriculum candidate statuses, canonical-doc status, equation producers/consumers/anchors/recalibration, MAIN landing review, and consolidation debt assign lifecycle state. | **ALREADY-COVERED.** Pact is stronger in typed ownership and curation; the owed improvement is making these states visible to retrieval through the common evidence schema. | `CEREBRAS POST DESCRIPTION`; no claim about undocumented internal practice. |

## Pact apparatus inventory: exact surfaces and stronger guarantees

### Recall and reconstructed graph

- Task #411 is explicitly **reconstruct-not-retrieve**: graph memory parses memory frontmatter and
  `[[links]]`, DAG FEED blocks, canonical-equation producer/consumer relations, tasks, and deferrals,
  then assembles connected context (`tools/graph_memory_recall.py:3-11`). The increment-2 surface has
  seven typed query tools—time, keyword, entity, topic, decision, neighbors, supersedes—and a
  deterministic Obsidian export (`tools/graph_memory_recall.py:54-76`).
- The canonical-root graph cache is **MEASURED 9,704 nodes / 32,156 edges**. The isolated worktree
  rebuild is **MEASURED 3,157 / 4,856**. These are inventory counts, not evidence of retrieval
  quality. Their divergence is evidence of missing source-completeness custody.
- `src/tac/graph_memory/obsidian_export.py` round-trips synthesized edge types into linked Markdown.
  The source memory files already use frontmatter, `[[links]]`, sister, and supersession relations.
- The MRAgent premise is faithfully represented locally: reconstruct a query-specific connected
  subgraph instead of treating memory as flat chunks. No claim about the external paper beyond this
  locally implemented design is needed for this crosswalk.

### Retrieval-first and bounded recall

- Task #346's `tools/corpus_query.py` is one deterministic query over research, equations, external
  memory, DAG, council anchors, canonical tasks, and docs; the tool itself records that the root cause
  was “apparatus writes better than it reads.” `src/tac/subagent_contract.py:136-142` requires
  recall-before-decide and a `STORES CONSULTED` statement. `tools/costate_digest.py` invokes the query
  on active convening under a bounded two-second path and labels it “the corpus knows.”
- This review actually ran it, not merely cited the contract:

  `STORES CONSULTED: research=6,187; equations=749; memory=1,961; DAG=660; docs=95; tasks=182; council=0; truncated=false; wall_clock=1.442s.`

  These are **MEASURED iterator-unit counts in this worktree**, not unique-document counts and not a
  recall score.

### Memory cap, consolidation, and established findings

- External `MEMORY.md` declares the `<17KB` cap and routes detail to cluster files. The historical
  full index records a 178KB / 7.5×-over-limit index that only partially loaded, producing repeated
  known mistakes. `MEMORY_established_findings_cluster_20260717.md` preserves consolidated material
  verbatim with hooks and links intact rather than deleting signal.
- `tools/consolidation_debt.py` makes four debt components visible: dirty code pile, undispositioned
  landings, commits since consolidation, and a labeled heuristic ratio of research memos to
  system-intelligence commits. Its current worktree verdict is **MEASURED `OK`** (2 dirty files/49
  churn lines before this memo, 0 undispositioned landings, 15 stale commits, ratio 2.0 from 45
  memos/23 system-intelligence commits). This is health telemetry, not proof that findings were
  consumed.
- The sharper unresolved incident is disposition-versus-consumption: a first-hand memory audit found
  1,087 `codex_findings` memos and 151 reviewed arms but no proof that a named runtime/planner
  consumer absorbed every accepted finding. That is why the proposed evidence schema includes a
  named consumer rather than only `reviewed=true`.

### Typed lifecycle registries

- `tac.canonical_equations` makes laws typed and callable, with units, domain, validity, triggers,
  producers, consumers, and `EmpiricalAnchor` records carrying predicted/empirical values, residual,
  provenance, verification status, and recalibration. The append-only registry uses latest-event
  state. Current snapshot: **749 events / 361 unique equation IDs / 634 anchors** (MEASURED registry
  parse; counts are not quality claims).
- Catalog #533's `tools/canonical_doc_registry.py` searches content and all git references rather than
  agent-name globs, tracks active/superseded/draft state and `superseded_by`, and verifies canonical
  bytes. Current snapshot: **10 entries: 9 active, 1 superseded, 0 drafts**. This describes the
  mechanism, not a green current gate: focused verification in this branch found stale branch
  expectations and 17 duplicate-SOT violations (details below).
- `.omx/state/lane_registry.json` is the typed ownership/maturity surface; this research lane is L1,
  phase 0, `research_only=true`. `.omx/state/deferral_ledger.md` requires every open deferral to have
  an owner and named activation trigger. `tac.harness_failure_ledger` makes failures countable,
  causally classified, resolvable, and recurrence-rankable rather than burying them in logs.
- `tac.witness_dsl.curriculum_candidate_pool` provides append-only statuses such as armed,
  built-never-fired, needs-build, and reformulation-queue. Its state file is absent from this
  worktree but present in the canonical root with 26 rows / 22 candidate IDs. That split is another
  source-health fact the proposed manifest must expose; it is not interpreted as candidate state
  here.

### Structural consumption: quadrality, triality, tasks, and FEED trajectory

- The campaign has a four-layer stack: equations (invariant law) → gauge (chart/cost/selection) →
  DSL (typed executable program) → DAG (trajectory/work graph), stated in
  `src/tac/witness_dsl/gauge.py:1-18`. The concise triality is DAG = what happened, DSL = what to do,
  equations = why it works (`docs/triality_dag_dsl_equations_deepmath.md:24-31`). Canonical tasks add
  the shared where-is-it lifecycle surface.
- `tools/triality_drift_detector.py` is a stop hook, not prose etiquette: it detects per-leg and
  consumer/retrieval drift and requires `STORES CONSULTED` on relevant decision documents. DAG
  `FEED-*` blocks preserve measured trajectory and point into equations, tasks, files, and memory.
- This is stronger than an answer-facing citation UI: Pact attempts to make accepted findings change
  executable intent and future routing. The remaining weakness is that reviewed memos can still lack
  a proven consumer, and that the graph index can silently omit whole source families.

## Inverse crosswalk: what bit Pact that the Cerebras design does not address

These are not criticisms of Cerebras's system beyond the article. Every negative is scoped to what
the post describes.

| Pact incident/debt | Why the Cerebras article does not close it | Required Pact control | verdict_scope |
|---|---|---|---|
| **Graph partial-load across linked worktrees:** 9,704/32,156 canonical-root cache became 3,157/4,856 after a fresh worktree build. | Incremental sync is described, but no mandatory-source count floor, canonical-root identity, or publish refusal is specified. | Source manifest + canonical repository identity + collapse refusal + parity test. | `CEREBRAS POST DESCRIPTION ONLY`. |
| **Goldfish / writes-better-than-reads:** the intended graph degraded into a large Markdown folder until #411 reconstructed it. | Search quality is addressed; structural memory that reconstructs decisions, supersession, tasks, equations, and deferrals is not. | Keep graph reconstruction co-equal with retrieval and test connected-answer coverage. | `STRUCTURAL CAMPAIGN MEMORY`. |
| **Orphan memos and disposition ≠ consumption:** reviewed findings can have no named solver/planner/runtime consumer. | The post ends at user/agent answers; it does not describe outcome feedback into executable system state. | Evidence rows require `consumed_by` plus a consumer receipt; consolidation monitors unconsumed accepted findings. | `SYSTEM-INTELLIGENCE CONSUMPTION`. |
| **Triality/quadrality drift:** DAG claim, DSL intent, equation, gauge, and task state can disagree. | A common retrieval row is not a consistency proof among executable representations. | Preserve drift stop hooks and same-turn structural updates for apparatus-changing findings. | `PACT CAMPAIGN GOVERNANCE`. |
| **Authority/custody confusion:** a plausible summary or ancestor number can become a current claim. | Citations/caveats help, but the post does not specify artifact hash, axis, vehicle, command, or empirical-anchor custody. | Carry typed provenance and authority labels through every retrieval/synthesis row. | `CONTEST-GRADE CLAIM AUTHORITY`. |
| **Deferred/default-off work becomes forgotten.** | Project scope and age decay do not supply owner + named reactivation trigger + terminal state. | Keep deferral ledger and curriculum pool as typed sources; surface triggers in recall. | `PACT WORK-LIFECYCLE`. |
| **Canonical document duplication by filename/agent-name search.** | Unified storage can still ingest two authorities without detecting semantic duplication. | Keep Catalog #533 content/all-reference search and canonical status; expose it to retrieval. | `CANONICAL-DOCUMENT IDENTITY`. |
| **Memory-cap crisis and summary confidence laundering.** | LLM distillation reduces length but can discard caveats or promote a paraphrase to authority. | Hash-bind retrieval cards to raw sources; cards cannot be evidence authority; cap health is monitored. | `PACT MEMORY SUMMARIZATION`. |
| **Negative verdicts escape their instance and get reused as family-wide facts.** | Recency decay is not verdict scope or explicit supersession. | Every negative retains `verdict_scope`; retrieval filters current/historical/superseded states. | `NEGATIVE-VERDICT REUSE`. |
| **Run resumability, checkpoint custody, exact hardware axis, and score-pointer authority.** | These are experiment/contest governance, outside the described organizational KB. | Continue Pact's run, scorer, hardware-axis, and pointer controls unchanged. | `NOT-APPLICABLE TO CEREBRAS KB POST`; org KB vs contest execution. |

## Proposed landing order and falsifiable acceptance contract

1. **P0 — source health before smarter ranking.** Build the source snapshot schema, canonical-root
   resolution, and refuse-on-collapse publisher. Acceptance: canonical root and a temporary linked
   worktree produce identical source unit counts and identical graph hashes at the same git/external
   memory revision; remove one required source and both graph and corpus builds refuse publication
   with that source named.
2. **P0 — currentness.** Add per-unit source hash/revision, observed time, SLA, tombstone, and
   supersession state. Acceptance: a superseded fact is historical-only; a stale required source is
   visibly refused or explicitly opted into; one-file changes produce full-rebuild-equivalent output.
3. **P0 — fused recall.** Normalize lexical and graph results first; add semantic retrieval only with
   a pinned offline-capable adapter. Acceptance: exact IDs cannot regress, every hit exposes its rank
   path, and constituent/fused modes remain callable for disambiguation.
4. **P1 — evaluation and only then reranking.** Freeze gold queries and report before/after metrics,
   source coverage, staleness, latency, and ablations. The Cerebras `k=60`, weights, top-20, and
   top-10 are candidates, not imported truth.
5. **P1 — consumption closure.** A decision can be `reviewed` without being `consumed`; report both.
   Acceptance: every accepted system-changing finding has either a named consumer receipt or an
   explicit `research_only` blocker/deferral trigger.

## Focused verification receipt

Command:

```text
/Users/adpena/Projects/pact/.venv/bin/python -m pytest -q \
  src/tac/tests/test_graph_memory.py \
  src/tac/tests/test_graph_memory_increment2.py \
  src/tac/tests/test_graph_memory_lensed_recall.py \
  src/tac/tests/test_corpus_query.py \
  src/tac/tests/test_consolidation_debt.py \
  src/tac/tests/test_canonical_doc_registry_and_dup_sot_gate.py \
  src/tac/tests/test_harness_failure_ledger.py \
  src/tac/tests/test_triality_drift_detector.py
```

Result: **186 passed, 4 failed in 7.22s**. This research branch changed no implementation or tests.
The failures are retained as evidence, not smoothed into green:

1. `test_real_corpus_builds_structurally_valid` found zero `links` edges. This directly corroborates
   the missing external-memory source in the linked worktree.
2. `test_real_corpus_lensed_bounded_local_pool_for_cost` expected the broad real query to hit the
   800-node local-pool cap but saw only 174 nodes. This is consistent with the collapsed graph; the
   **causal attribution to source collapse is DERIVED**, not separately isolated by that test.
3. `test_lookup_finds_spec_v10_by_concept_not_name` expected the historical
   `claude/p0_521_spec_v10_capstone_20260717` branch but the current registry says `main`.
4. `test_live_repo_zero_violations` returned 17 duplicate-canonical-spec violations instead of zero.

The last two are **current canonical-doc gate/test drift, verdict_scope=THIS BRANCH SNAPSHOT**. They do
not falsify the registry design and were not repaired because this delegated lane is a research-only
Cerebras crosswalk. MAIN should disposition them separately rather than treating the entire focused
suite as green.

## Stops, custody, and pointer-delta honesty

- No provider call, job, training, evaluator, archive, score, paid dispatch, or pointer mutation was
  authorized or performed.
- Sacred directory `experiments/results/levelset_n600_witness_20260717T113932Z/` was not modified.
- This is a **research-only design crosswalk**, not a landing claim for the four changes.
- **Pointer delta: UNMOVED — `0.1910828242 [contest-CPU Linux x86_64]`.**
- MAIN must independently review the source extraction custody, the one-row-per-mechanism coverage,
  the 9,704/32,156 → 3,157/4,856 causal claim, every ADOPT gate, and every scoped negative before
  landing. MAIN—not this branch—owns merge authority and any follow-on task registration.
