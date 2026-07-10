# PAPERS-CHECKED — MRAgent: "Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents"

**Source:** https://github.com/Ji-shuo/MRAgent (operator-dropped 2026-07-10). Domain: long-term
conversational memory for LLM agents (LoCoMo + LongMemEval benchmarks). NOT the contest/witness/d_seg
problem — this is relevant to our **META recall apparatus**, not the witness line.

**STORES CONSULTED:** L83 (retrieval-first nexus root cause) · memory
`curriculum_candidate_pool_p0_orphan_class_20260710` + the MEMORY.md cap struggle this session · #346
(retrieval-first operating layer) · CLAUDE.md §OPERATOR PRIORITY anti-forgetfulness + §Results-must-become-
system-intelligence.

## What it is (measured from the repo)
Two-phase pipeline: (1) **Construction** — dialogue turns → self-contained sentences + keywords → an
in-memory GRAPH of {episode, topic, personal-fact} nodes. (2) **QA** — a tool-calling reasoning loop with
7 query tools (`query_conversation_time`, `query_event_keywords`, `query_personal_information`, …) that
RECONSTRUCTS an answer from the graph rather than retrieving chunks. OpenRouter multi-model wrapper;
LLM-as-judge eval.

## The one idea that MATTERS for us (the thesis, not the code)
**"Reconstruct, not retrieve" dissolves our MEMORY.md-cap problem.** Our recurring failure is: MEMORY.md
is a FLAT index that must FULLY load (<17KB) or it partial-loads and I forget (the goldfish problem I hit
~5× this session compressing it). MRAgent's thesis inverts that: don't load the whole store — RECONSTRUCT
the relevant context ON-DEMAND via tool-calls over a graph. If recall is reconstruct-on-demand, **the
flat-load cap stops being the binding constraint.** That is a genuine reframe of L83 ("apparatus WRITES
better than it READS") + the #346 retrieval-first nexus: the READ side becomes graph-traversal +
reconstruction, not grep-over-3000-files or load-the-whole-index.

## Honest scoping (what does NOT transfer)
- It's a CONVERSATIONAL-QA-benchmark system; our stores are richer + already partly graph-like (the DAG,
  triality `[[name]]` cross-links, the costate ledger). So its CONSTRUCTION phase (chat-log→graph) is
  largely N/A — we don't have a chat log, we have structured research stores.
- The RETRIEVAL phase (tool-calling reconstruction over an episode/topic/fact graph) is the transferable
  part — it maps onto #346's corpus-query + recall-push, but as typed graph-tools instead of grep.
- Adapting is REAL work (a graph over our memory files + a reconstruction tool-loop), and it is
  META-APPARATUS — subordinate to the witness line (per witness_line_priority binding). Bank, don't
  drop the witness work for it.

## Disposition
CANDIDATE reframe for #346 (retrieval-first nexus) + the L83 read-side. Specific applicable move: model
our memory files as an {episode/finding, topic, entity} graph (the `[[name]]` links already ARE edges) +
a reconstruct-on-demand recall tool, so MEMORY.md's flat cap stops gating recall. NOT witness-line;
operator-routable whether to pursue now (meta-apparatus improvement to my own recall) or bank. Verdict
scope: INSTANCE — one paper, thesis-useful, not yet built/measured here.
