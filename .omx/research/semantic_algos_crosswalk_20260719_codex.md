# Semantic Algos algorithm crosswalk — 2026-07-19

## Answer first

**Verdict: adopt zero upstream items.** At pinned upstream commit
`29da28ab5f7bdbe648b7780f8d862821437e6e81`, `semantic-algos` is a catalog of
17 natural-language reasoning procedures, six illustrative composition marks,
eight named composition examples, and 25 explicitly uninstalled composition
sketches. It is not an executable semantic classifier, corpus retriever, graph
algorithm, duplicate detector, or failure-memory implementation. Every
potentially relevant procedure is either already covered by a stronger
typed/local Pact surface or is not applicable to the five named apparatus
problems. A hosted embedding/API dependency would be a downgrade in
determinism, cost, latency, offline operation, and failure containment.

Pointer `0.1910828242 [contest-CPU Linux x86_64]` is **UNMOVED**. Disposition is
**MEANS-only**. This is a research-only, algorithms-only crosswalk. It launched
no training, scorer, evaluator, GPU, paid service, or submission; changed no
score or pointer; and did not touch
`experiments/results/levelset_n600_witness_20260717T113932Z/`.

The sibling `cerebras_kb_crosswalk_20260719` owns knowledge-base architecture.
This memo does not propose or duplicate a KB architecture.

## Authority, method, and evidence labels

The upstream repository was cloned from
[`kousun12/semantic-algos`](https://github.com/kousun12/semantic-algos/tree/29da28ab5f7bdbe648b7780f8d862821437e6e81),
pinned to the commit above, and all 20 tracked files were read. The inventory is
therefore source-derived rather than inferred from the README alone.

- **OBSERVED** means direct inspection of the pinned upstream or named Pact
  source.
- **MEASURED** means a direct count or focused test run in this session.
- **DERIVED** means the disposition follows from those inspected mechanisms.

The governing craft standard is
[`docs/operating_manual_craft_handoff.md`](../../docs/operating_manual_craft_handoff.md):
answer the real question, decompose work into checkable obligations, re-derive
from primary evidence, label claims, self-attack, and preserve literal blocker
scope. Those requirements are used as actual comparison criteria below, not as
a generic citation.

## What the upstream actually contains

**OBSERVED:** the commit has one `README.md`, 17 `SKILL.md` files, and two small
UI metadata YAML files for `joke` and `lyric`. It has no executable library,
tests, package/dependency manifest, command-line entry point, model, index,
cache, persistence schema, or runtime dispatcher. The README states MIT, but
the commit has no separate license file. Its Haskell-flavored composition is
explicitly described as a diagram, “not a runtime or new language.” The 25
named user-space programs are explicitly described as sketches that the
repository does not install.

The unit of reuse is therefore a prompt procedure executed by an agent. That
can be useful for human-facing ideation, but it does not supply the bounded,
typed, deterministic mechanism needed by the five Pact targets in this task.

## First-hand Pact baseline

| Named surface | Re-derived local mechanism and current evidence |
| --- | --- |
| fmtools SENSE #522 | `src/tac/fm_advisory.py` invokes the separately installed FM interpreter through a detached subprocess selected by `FM_ADVISORY_PYTHON`, `DASH_FM_PYTHON`, or the fixed local venv path. `classify` accepts only closed `anyOf` labels, uses a 512-entry process cache and a 25-second bound, and returns `None` on absence, timeout, malformed output, or failure. Four insertion points remain advisory: four regime labels with numeric annulus agreement, six event classes, high/medium/low duty relevance secondary to P8, and confound hints mapped to known IDs or `none`. `tools/costate_digest.py` renders no lines on absence. At `tools/triality_drift_detector.py:847`, `_FM_SCOPE_SCRIPT` runs detached and may only broaden a negative to family/paradigm; it can never authorize a score, block, launch, or actuation. **MEASURED prior live incident #522:** full FM-on digest took 12.8 seconds versus 1.6 seconds with session-start FM disabled, while producing the four expected advisory classes. |
| Graph memory #411 | `src/tac/graph_memory/model.py`, `build.py`, `recall.py`, and `query_tools.py` implement ten typed node kinds, eight edge kinds, deterministic parsing, seed matching, and bounded bidirectional expansion. Typed-query increment 2 exposes time-window, keyword, entity, topic, decision, neighbor, and supersession operations. Seed score combines capped term counts, distinct coverage, exact `#ref` bonus `+3`, and date bonus `+2`; at most four seeds expand both directions to depth two with `0.45` decay, multi-seed accumulation, stable tie-breaking, and a default 18-node result. Opt-in lensed recall adds shortest-path bridges and bounded local centrality/saddle candidates. **MEASURED current primary cache:** 9,704 valid JSONL nodes and 32,156 valid JSONL edges (including 28,047 `references`, 1,290 `links`, 1,201 `tagged`, 835 `consumes`, 733 `produces`, and 50 `sister` edges). This isolated sparse worktree rebuilds only 3,157 nodes/4,856 edges and is not used as the production-size claim. |
| Retrieval-first #346 | `tools/corpus_query.py` deterministically searches seven stores: research, equations, memory, DAG, council, tasks, and docs. It requires multi-term coverage for broad queries and scores distinct terms, total matches, density, and recency under per-store file/row and wall-time bounds. `tools/costate_digest.py::section_corpus_recall` exposes at most five “the corpus knows” lines from the top results, fails open, and never rebuilds graph memory at session start. No hosted embedding or network call is involved. |
| Anti-duplicate SoT #533 | `tools/canonical_doc_registry.py` has an explicit active/superseded/draft registry, concept tags, token-overlap lookup, all-ref `git grep`, and working-tree coverage. `src/tac/confound_gates.py::check_no_duplicate_canonical_spec_across_refs` inspects Markdown blobs across refs in one `git cat-file --batch`, extracts adjacent canonical vehicle families, and checks registry, filename, and worktree legs. The measured incident was concrete: a name glob missed an existing V10 spec on an unmerged branch and a duplicate was created. The landed guard has content-family coverage, but remains lexical/structural rather than semantic-paraphrase search and is WARN-only pending its strict flip. |
| Failure ledger / memory taxonomy | `src/tac/harness_failure_ledger.py` is append-only under `fcntl`, with five closed surfaces, measured/hypothesized/falsified causal states, four resolution states, four event types, state folding, preserved diagnoses, and deterministic unresolved-first / recurrence / recency ranking. The costate digest reader is schema-tolerant across historical class keys and terminal statuses. **MEASURED in this worktree:** 68 event rows, 20 distinct class buckets (including one legacy `?` bucket), three unresolved buckets, and eight recurrent buckets. This is persistent operational memory, not a one-shot narrative root-cause exercise. |

## Installed standard-library crosswalk — every upstream skill

Each row inventories the actual control mechanism in the pinned `SKILL.md`,
not just its one-line README gloss. `ALREADY-COVERED` and `NOT-APPLICABLE` are
negative adoption decisions, so every row states its exact `verdict_scope`.

| Upstream item | Actual mechanism | Pact crosswalk | Verdict |
| --- | --- | --- | --- |
| `five-whys` | Follow a flexible roughly-five-step causal, conceptual, incentive, or philosophical chain; keep each step concrete/testable; stop at speculation, circularity, or loss of utility; end with a root cause and next action. | The operating manual §§4–6 requires primary-evidence re-derivation and self-attack. The failure ledger stores typed causal status, history, recurrence, resolution, and landed guards across sessions. That is stronger than a transient prose chain because it preserves falsification and recurrence. | **ALREADY-COVERED.** `verdict_scope=UPSTREAM_PROMPT_PROCEDURE_X_PACT_FAILURE_CAUSAL_MEMORY`; no new executable taxonomy, ranking, or persistence is supplied. |
| `n-whys` | Run the why-chain to an exact caller-specified depth `n`, distinguish facts from hypotheses, then synthesize a principle and implications. | Exact-depth continuation conflicts with Pact’s fail-closed instruction to stop when evidence becomes speculative. Typed causal states and the operating manual already preserve the useful portion without manufacturing depth. | **NOT-APPLICABLE.** `verdict_scope=EXACT_DEPTH_PROMPT_CHAIN_X_EVIDENCE_BOUNDED_FAILURE_CLASSING`; it adds no safe failure-ledger primitive. |
| `first-principles-thinking` | List inherited defaults; separate necessities from custom; identify physical, economic, human, technical, logical, legal, and ethical primitives; rebuild; compare with the default; test. | The operating manual §4 requires re-derivation from primary artifacts, and §6 requires adversarial checks against requirements and edge cases. Pact’s local surfaces additionally bind decisions to code, typed ledgers, and tests. | **ALREADY-COVERED.** `verdict_scope=GENERIC_FIRST_PRINCIPLES_PROMPT_X_TASK_567_LOCAL_VERIFICATION`; upstream does not add a checkable implementation. |
| `assumption-audit` | Extract factual, causal, people, continuity, capability, and definitional assumptions; grade load, confidence, and testability; select the highest-load/lowest-confidence keystone; propose cheap tests and fallbacks. | The operating manual §§2, 4, 5, and 6 already require checkable obligations, primary evidence, confidence labels, and self-attack. Failure-ledger causal states preserve whether a premise is hypothesized, measured, or falsified. | **ALREADY-COVERED.** `verdict_scope=ASSUMPTION_PROMPT_X_CANONICAL_EVIDENCE_AND_FAILURE_STATE`; no measured incident requires another prose pass. |
| `dp-solve` | Define objective, state, and base cases; decompose overlapping subproblems; answer each once in a memo table with confidence/reuse; synthesize bottom-up; extract reusable lemmas. | Graph memory #411 is the concrete form: typed nodes/edges, deterministic seed scoring, memoized source parsing, bounded subgraph reconstruction, and typed queries at measured 9.7k/32.2k scale. Retrieval-first #346 supplies bounded store search when graph structure is unnecessary. | **ALREADY-COVERED.** `verdict_scope=UPSTREAM_DP_PROMPT_X_PACT_RETRIEVAL_RANKING_EXPANSION`; upstream has no graph, cache, parser, index, or ranking code. |
| `decision-matrix` | Compare 2–6 options on 4–8 criteria with weights summing to 100 and reasoned 1–10 cells; total scores; flip one weight for sensitivity; compare to intuition. | Pact’s duty and failure queues rank typed evidence with explicit operational priority and recurrence. Replacing evidence-bearing rank with subjective 1–10 cells would weaken custody; the upstream offers no integration to P8, failure events, or live incidents. | **NOT-APPLICABLE.** `verdict_scope=SUBJECTIVE_OPTION_MATRIX_X_TYPED_DUTY_AND_FAILURE_RANKING`; no apparatus integration or falsifiable live gate is supplied. |
| `inversion` | Enumerate 5–10 operational, silent, self-caused, and environmental failure modes; rank likelihood × damage; convert the strongest into guards; return to a forward plan. | The operating manual §6 requires self-attack. The failure ledger makes failures persistent and recurrence-ranked, while confound gates turn known bug classes into executable guards. This is stronger than a one-session ranked list. | **ALREADY-COVERED.** `verdict_scope=INVERSION_PROMPT_X_PERSISTENT_FAILURE_AND_GUARD_APPARATUS`; upstream supplies no guard or ledger hook. |
| `counterfactual` | Change one minimal plausible fact; propagate first- and second-order effects; model restoration/equilibrium forces; grade confidence; stop when speculative; classify contingent versus overdetermined. | This is a general reasoning aid, not a classifier, retriever, graph expansion rule, duplicate detector, or persistent failure primitive. No named live incident here is blocked on missing counterfactual prose. | **NOT-APPLICABLE.** `verdict_scope=COUNTERFACTUAL_NARRATIVE_X_FIVE_TASK_567_APPARATUS_SURFACES`; family usefulness outside this task remains open. |
| `ladder-of-abstraction` | Locate the current level; move down to concrete cases, across middle-level patterns, and up to principles; validate that the abstraction preserves causality; choose the action-changing level. | The operating manual §§1–5 already moves from literal ask through intended outcome to concrete obligations and scoped conclusions. Pact’s verdict-scope discipline prevents a formulation negative from becoming a family negative. | **ALREADY-COVERED.** `verdict_scope=ABSTRACTION_PROMPT_X_PACT_VERDICT_SCOPE_AND_CRAFT_HANDOFF`; there is no machine-readable classifier or scope gate upstream. |
| `analogy-transfer` | Abstract a concrete problem structurally; find 3–5 distant twins; name their mechanisms; translate 2–3 back; state mandatory disanalogies; propose a pilot. | It provides no source index, structural embedding, retrieval operation, or graph expansion rule. Cross-disciplinary paradigm research is also outside this Codex algorithms lane. | **NOT-APPLICABLE.** `verdict_scope=ANALOGY_IDEATION_PROMPT_X_PACT_RETRIEVAL_AND_GRAPH_MEMORY`; no claim is made about its usefulness in Claude-owned design research. |
| `question-forge` | Dispatch operational questions to direct answering; otherwise diagnose loaded premise, wrong level, false binary, displacement, wish, or shield; transform subject/level/time/inversion/value/dramatization; output one costly/generative question without answering it. | The operating manual §1 already requires literal ask, intended outcome, and standing context, and §7 requires answering the actual question first. Corpus recall supplies evidence before reframing. | **ALREADY-COVERED.** `verdict_scope=QUESTION_REFRAMING_PROMPT_X_PACT_SESSION_ROUTING`; local practice is stronger because it must still deliver the requested artifact. |
| `explanation-ladder` | Produce exactly five successive voices—high school, college, PhD, philosopher, “gigabrain”—each correcting and deepening the prior explanation. | This is a presentation form. It does not alter classing, retrieval, graph reconstruction, duplicate detection, or failure memory, and fixed personas add no custody. | **NOT-APPLICABLE.** `verdict_scope=FIVE_VOICE_EXPLANATION_FORMAT_X_TASK_567_APPARATUS`; no broader family negative. |
| `nietzche-ladder` | Pass a subject through Camel/inherited burden, Lion/negation, and Child/creative affirmation, with each stage responding to the last. | This is a philosophical presentation operator and has no mechanism on the named operational surfaces. The upstream path itself uses the spelling `nietzche-ladder`; this inventory preserves that source identity. | **NOT-APPLICABLE.** `verdict_scope=CAMEL_LION_CHILD_FORMAT_X_TASK_567_APPARATUS`; no broader philosophical-use verdict. |
| `golden-circle` | Separate why, how, and what; test alignment; rewrite inside-out from purpose to method to output. | Useful communication framing, but it supplies no deterministic classifier, evidence store, query, ranking, expansion, duplicate gate, or failure schema. | **NOT-APPLICABLE.** `verdict_scope=WHY_HOW_WHAT_FORMAT_X_TASK_567_APPARATUS`; communications uses remain open. |
| `parable` | Distill one tension; create a minimal world with one unexplained strange rule; embody dignified opposing stances; implicate the reader; end with an open image and no moral. | A literary output compiler is unrelated to the five apparatus surfaces and would hide rather than strengthen evidence custody. | **NOT-APPLICABLE.** `verdict_scope=LITERARY_PARABLE_OUTPUT_X_TASK_567_APPARATUS`; no claim about creative-writing value. |
| `lyric` | Choose one emotional contradiction, speaker/addressee/occasion, and image field; create a recurring kernel whose meaning changes; choose poem/song; turn and compress without moralizing. | A literary output compiler has no operational integration point here. Its only extra file is UI metadata, not an implementation. | **NOT-APPLICABLE.** `verdict_scope=LYRIC_OUTPUT_X_TASK_567_APPARATUS`; no claim about creative-writing value. |
| `joke` | Find a comic mismatch and target; establish Frames A/B and a hinge; choose one of six comic engines; compress setup and late punchline; test surprise, inevitability, and compression. | A humor output compiler does not improve any named classifier or memory surface. Its UI YAML adds no runtime reasoning code. | **NOT-APPLICABLE.** `verdict_scope=JOKE_OUTPUT_X_TASK_567_APPARATUS`; no claim about creative-writing value. |

## Composition-notation crosswalk — every upstream primitive

These six marks appear only in README diagrams. The upstream explicitly says
they are not necessarily valid Haskell and are not a runtime or language.

| Primitive | Upstream meaning | Verdict |
| --- | --- | --- |
| `>>>` | Pass one procedure’s result to the next. | **ALREADY-COVERED.** Pact already has executable Python composition plus typed DSL/DAG order. `verdict_scope=README_SEQUENCE_NOTATION_X_EXECUTABLE_PACT_COMPOSITION`; the glyph has no parser or validation. |
| `&&&` | Fan out multiple procedures over one input. | **ALREADY-COVERED.** Existing bounded query fan-out and typed parallel consumers are executable and testable. `verdict_scope=README_FANOUT_NOTATION_X_EXECUTABLE_PACT_COMPOSITION`; no runtime is offered. |
| `<|>` | Choose or fall back between routes. | **ALREADY-COVERED.** FM and digest surfaces already fail open through explicit bounded fallbacks, while authority paths fail closed where required. `verdict_scope=README_FALLBACK_NOTATION_X_TYPED_PACT_FAILURE_POLICY`; upstream cannot encode authority-specific behavior. |
| ``with`` | Attach options to an operator. | **ALREADY-COVERED.** Pact uses typed configuration and DSL compilation with provenance rather than unparsed option prose. `verdict_scope=README_CONFIGURATION_NOTATION_X_TYPED_DSL`; no parser or schema is supplied. |
| `map` | Apply a procedure to every result. | **NOT-APPLICABLE.** This generic higher-order notation contributes no new apparatus algorithm. `verdict_scope=README_MAP_GLYPH_X_TASK_567_SURFACES`. |
| `repeat` | Iterate or deepen a procedure a fixed number of times. | **NOT-APPLICABLE.** Blind fixed-depth repetition can violate evidence stop rules and adds no checkpoint or convergence mechanism. `verdict_scope=README_REPEAT_GLYPH_X_EVIDENCE_BOUNDED_PACT_LOOPS`. |

## README composition-example crosswalk — every named example

These names demonstrate the diagram notation; the upstream does not install
them as programs. They are separate from the later user-space sketch catalog.

| Example | Actual diagrammed composition | Verdict |
| --- | --- | --- |
| `work` | Run depth-eight `n-whys` on why people work, then assumption-audit the deepest claim. | **ALREADY-COVERED.** Evidence-bounded root-cause and assumption checks already live in the operating manual and failure state. `verdict_scope=README_WORK_EXAMPLE_X_TASK_567_CAUSAL_APPARATUS`; no program is shipped. |
| `desire` | Forge “what do I want?” and then run a depth-seven why-chain. | **ALREADY-COVERED.** Operating-manual ask routing covers the useful dispatch step without mandatory speculative depth. `verdict_scope=README_DESIRE_EXAMPLE_X_SESSION_QUESTION_ROUTING`; no program is shipped. |
| `promise` | Rebuild promising from first principles and invert the candidate rule. | **ALREADY-COVERED.** Primary-evidence derivation plus adversarial guard review is already required. `verdict_scope=README_PROMISE_EXAMPLE_X_CRAFT_HANDOFF`; no apparatus code is shipped. |
| `mouse` | Raise the abstraction of *If You Give a Mouse a Cookie* and transfer its repeating structure by analogy. | **NOT-APPLICABLE.** Literary structural analogy does not improve a named Task #567 surface. `verdict_scope=README_MOUSE_EXAMPLE_X_TASK_567_APPARATUS`. |
| `karamazov` | DP-decompose a novel, forge its most reused tension, and emit a parable. | **NOT-APPLICABLE.** It provides neither text parser nor graph/retrieval implementation and ends in literary output. `verdict_scope=README_KARAMAZOV_EXAMPLE_X_TASK_567_APPARATUS`. |
| `missing` | Forge a question about missing someone and emit a lyric. | **NOT-APPLICABLE.** Literary output only. `verdict_scope=README_MISSING_EXAMPLE_X_TASK_567_APPARATUS`. |
| `meetings` | Audit the alignment premise and convert the hidden contradiction into a joke. | **NOT-APPLICABLE.** Humor is not a duty/failure classifier or persistent dissent record. `verdict_scope=README_MEETINGS_EXAMPLE_X_TASK_567_APPARATUS`. |
| `deepAnalysis` | Forge a question; fan out first-principles, inversion, and analogy; synthesize; audit the result. | **ALREADY-COVERED as review procedure.** The craft manual already requires decomposition, independent re-derivation, self-attack, and final obligation checking. `verdict_scope=README_DEEP_ANALYSIS_EXAMPLE_X_PACT_REVIEW_PROTOCOL`; no fan-out runtime or synthesizer is shipped. |

## README-only program crosswalk — every uninstalled upstream pattern

These are source-declared design sketches, not installed skills. They are
enumerated to avoid silently treating examples as shipped algorithms.

| Program | Actual sketched mechanism | Verdict |
| --- | --- | --- |
| `omelas` | Forge a question, turn it into a parable, hide the question. | **NOT-APPLICABLE.** Hides intermediate evidence. `verdict_scope=UNINSTALLED_LITERARY_SKETCH_X_TASK_567_APPARATUS`. |
| `court-jester` | Audit a plan, turn the keystone contradiction into a joke, carry dissent in the punchline. | **NOT-APPLICABLE.** Dissent must remain explicit and inspectable. `verdict_scope=UNINSTALLED_HUMOR_DISSENT_SKETCH_X_CANONICAL_EVIDENCE`. |
| `sliding-doors` | Propagate both decision branches ten years and return two future-self letters without a recommendation. | **NOT-APPLICABLE.** No live apparatus decision needs narrative future letters. `verdict_scope=UNINSTALLED_COUNTERFACTUAL_LETTERS_X_TASK_567`. |
| `oracle` | Run a decision matrix, hide its scores inside a parable, use reader allegiance as a gut check. | **NOT-APPLICABLE.** Hidden scores oppose Pact custody. `verdict_scope=UNINSTALLED_HIDDEN_MATRIX_SKETCH_X_TYPED_DECISION_EVIDENCE`. |
| `cassandra` | Invert a plan, propagate the strongest failure, write a dated future post-mortem. | **ALREADY-COVERED.** Failure events, diagnoses, recurrence, and guards are stored directly. `verdict_scope=UNINSTALLED_FUTURE_POSTMORTEM_SKETCH_X_FAILURE_LEDGER`; prose adds no state. |
| `obituary` | Combine world-without-X counterfactual, Camel/Lion/Child, and eulogy. | **NOT-APPLICABLE.** Literary form only. `verdict_scope=UNINSTALLED_EULOGY_SKETCH_X_TASK_567`. |
| `borges` | Review an imaginary successful artifact from five years later, including consequences and imitators. | **NOT-APPLICABLE.** Speculative review supplies no observed incident or gate. `verdict_scope=UNINSTALLED_IMAGINARY_REVIEW_X_TASK_567`. |
| `chorus` | Give a plan-in-motion a voice that observes but cannot recommend or intervene. | **NOT-APPLICABLE.** The named advisory surface already has explicit non-authority semantics. `verdict_scope=UNINSTALLED_OBSERVER_VOICE_X_FM_ADVISORY`; no classifier is added. |
| `scheherazade` | End each session with the one live question and carry it forward. | **ALREADY-COVERED.** Exact checkpoint keys, `subagent_progress.jsonl`, next actions, and durable artifacts preserve more than a question. `verdict_scope=UNINSTALLED_SESSION_QUESTION_SKETCH_X_CRASH_RESUMABILITY`. |
| `rabbit-hole` | Alternate question-forge and n-whys until the question stops changing. | **NOT-APPLICABLE.** Textual fixed-point equality is neither an evidence convergence test nor bounded apparatus algorithm. `verdict_scope=UNINSTALLED_FIXED_POINT_QUESTION_SKETCH_X_TASK_567`. |
| `heist` | DP-decompose, run analogies on low-confidence subproblems, insert imported mechanisms, and resynthesize. | **ALREADY-COVERED for retrieval/decomposition only.** Graph memory and corpus query expose low-evidence substructure without inventing cross-domain mechanisms. `verdict_scope=UNINSTALLED_DP_ANALOGY_SKETCH_X_PACT_GRAPH_RETRIEVAL`; paradigm ideation remains outside this lane. |
| `chesterton` | Trace a convention’s cause, reduce to primitives, retain it if its constraint remains and remove it otherwise. | **ALREADY-COVERED.** Operating manual §§4–6 requires source re-derivation and adversarial necessity checks. `verdict_scope=UNINSTALLED_CONVENTION_AUDIT_SKETCH_X_CRAFT_HANDOFF`; no executable test is supplied. |
| `triage` | Inspect a question and route it to direct answer, causal analysis, option comparison, assumption test, or reframing. | **ALREADY-COVERED.** Operating manual §1 and current task routing separate literal ask, intended outcome, and standing context. `verdict_scope=UNINSTALLED_QUESTION_ROUTER_X_SESSION_CRAFT`; no classifier or dispatch implementation is supplied. |
| `peace-talks` | Run a decision matrix under each party’s weights and return the smallest explanatory weight differences. | **NOT-APPLICABLE.** No multi-party preference dispute exists on these five surfaces. `verdict_scope=UNINSTALLED_WEIGHT_DISPUTE_SKETCH_X_TASK_567`. |
| `turing-mirror` | Reconstruct an opponent’s why/how/what to their satisfaction, then audit oneself using their strongest premise. | **NOT-APPLICABLE.** No opponent-model apparatus requirement exists here. `verdict_scope=UNINSTALLED_STEELMAN_SKETCH_X_TASK_567`. |
| `dissent` | Preserve the strongest losing opinion as a document for future affected people. | **ALREADY-COVERED.** Append-only historical provenance, scoped negatives, and failure diagnoses preserve rejected evidence. `verdict_scope=UNINSTALLED_DISSENT_DOCUMENT_X_APPEND_ONLY_PROVENANCE`; no schema integration is added. |
| `devils-advocate` | Assign a bounded adversarial role to prosecute every supporting claim before canonization. | **ALREADY-COVERED.** Operating manual §6 and mandatory independent review require self-attack and literal blocker reporting. `verdict_scope=UNINSTALLED_ADVERSARIAL_ROLE_X_PACT_REVIEW_PROTOCOL`; no executable guard is supplied. |
| `truth-and-reconciliation` | Gather blame-free accounts under amnesty, close collection, then causally analyze the shared record. | **NOT-APPLICABLE.** Human reconciliation protocol, not an apparatus primitive. `verdict_scope=UNINSTALLED_RECONCILIATION_SKETCH_X_TASK_567`. |
| `federalist` | Derive one conclusion independently from economic, moral, and operational premises and inspect convergence. | **NOT-APPLICABLE.** No three-premise institutional decision is the blocker here. `verdict_scope=UNINSTALLED_MULTI_PREMISE_SKETCH_X_TASK_567`. |
| `rashomon` | Narrate every participant’s self-consistent view, separating invariant facts from interested differences. | **NOT-APPLICABLE.** No participant-narrative event is being adjudicated. `verdict_scope=UNINSTALLED_MULTIPERSPECTIVE_SKETCH_X_TASK_567`. |
| `aporia` | Forge questions until two sincere commitments contradict; return the contradiction unresolved. | **NOT-APPLICABLE.** The task requires operational dispositions, not unresolved philosophical contradiction. `verdict_scope=UNINSTALLED_APORIA_SKETCH_X_TASK_567`. |
| `veil` | Hide the chooser’s eventual role, evaluate policy from every position, then reveal the choice. | **NOT-APPLICABLE.** No distributive-policy selector is in scope. `verdict_scope=UNINSTALLED_VEIL_SKETCH_X_TASK_567`. |
| `via-negativa` | Define a subject by informative exclusions while leaving the positive space unstated. | **NOT-APPLICABLE.** Apparatus contracts require positive typed schemas and gates. `verdict_scope=UNINSTALLED_NEGATIVE_DEFINITION_SKETCH_X_TASK_567`. |
| `genealogy` | Trace a value through historical contests and ask who benefited from each formulation. | **NOT-APPLICABLE.** Historical value genealogy does not improve the named operational algorithms. `verdict_scope=UNINSTALLED_GENEALOGY_SKETCH_X_TASK_567`. |
| `epoché` | Bracket interpretation, record observations only, then restore interpretations with their assumptions. | **ALREADY-COVERED.** OBSERVED/MEASURED/DERIVED labels and premise/falsification states make this separation explicit. `verdict_scope=UNINSTALLED_BRACKETING_SKETCH_X_PACT_EVIDENCE_LABELS`; no persistence or validator is supplied. |

## Direct answers on the five target comparisons

### 1. Does upstream improve fmtools’ hardcoded regex/classing?

No. **DERIVED:** none of the 17 skills implements class extraction, a closed
taxonomy, regex replacement, parser, model call, or confidence-calibrated
dispatcher. `triage` is only an uninstalled natural-language sketch, and
`assumption-audit` is a prompt checklist. Integrating either would mean asking
an agent/LLM to interpret the text, which is less bounded than fmtools’ closed
`anyOf` contract and does not remove fmtools’ current interpreter dependency.

FM remains expensive enough to keep off the session-start spine (12.8 seconds
measured full versus 1.6 seconds gated), but the upstream offers no cheaper,
deterministic, offline classifier. `verdict_scope=SEMANTIC_ALGOS_COMMIT_29DA28A_X_FM_CLASSIFIER_REPLACEMENT`.
This is not a verdict that fmtools is optimal; it is a verdict that this
upstream does not supply the missing replacement.

### 2. Does upstream improve graph-memory retrieval/ranking/expansion?

No. `dp-solve` describes memoization in prose but has no source parser, typed
node/edge schema, seed matcher, ranking function, expansion bound, graph cache,
or query API. Graph memory already supplies all of those and operates on a
MEASURED 9,704-node/32,156-edge primary cache. `analogy-transfer` similarly has
no index from which to find its distant twins.

`verdict_scope=SEMANTIC_ALGOS_COMMIT_29DA28A_X_GRAPH_MEMORY_411_ALGORITHMS`.
The sibling KB architecture lane remains the owner of any storage-architecture
comparison.

### 3. Does upstream improve retrieval-first “the corpus knows”?

No. `tools/corpus_query.py` has explicit store coverage, bounded work, stable
scoring, multi-term coverage, recency, and fail-open rendering. Upstream has no
corpus connector or retrieval primitive at all. Calling an agent on every
query would be slower and less reproducible than deterministic local token
search.

`verdict_scope=SEMANTIC_ALGOS_COMMIT_29DA28A_X_RETRIEVAL_FIRST_346`.

### 4. Does upstream close semantic near-duplicate detection?

No. The local #533 apparatus has a real remaining gap: its registry tags,
token conjunction, filename/content-family extraction, and all-ref scan do not
constitute general paraphrase similarity. But upstream supplies no embedding,
semantic index, similarity metric, duplicate corpus, or threshold gate. A
hosted embedding API would be a downgrade because it introduces network and
provider availability, variable latency/cost, model-version drift, content
egress, and weaker offline reproducibility into a pre-create guard.

If the gap is pursued later, the integration point should remain
`tools/canonical_doc_registry.py lookup/check` and the gate should be
falsifiable on a fixed local fixture containing paraphrased duplicates plus
hard non-duplicates, with deterministic/offline execution and explicit false
positive/negative bounds. That is a local future requirement, **not an ADOPT
claim from this upstream**, because no upstream item implements any part of it.

`verdict_scope=SEMANTIC_ALGOS_COMMIT_29DA28A_X_SEMANTIC_NEAR_DUPLICATE_GATE`.

### 5. Does upstream improve failure-ledger taxonomy or duty ranking?

No. `five-whys`, `assumption-audit`, and `inversion` can shape a human analysis,
but they do not define a closed class schema, append event history, preserve
diagnoses, fold current state, record recurrence, or rank unresolved incidents.
The existing ledger and costate reader do. Upstream’s subjective decision
matrix would not be a sound substitute for P8/duty evidence or recurrence.

`verdict_scope=SEMANTIC_ALGOS_COMMIT_29DA28A_X_FAILURE_LEDGER_AND_DUTY_QUEUE`.

## Adoption-gate accounting

An `ADOPT` row required all three of: a concrete local integration point, a
falsifiable gate, and a measured incident/live pain that the upstream mechanism
actually remedies. There are measured pains—the 12.8-second FM path, the V10
duplicate incident, and recurrent failure classes—but the pinned upstream has
no executable mechanism that remedies any of them. Inventing an implementation
from the prose would be a new local design, not adoption of upstream code.

Accordingly:

- installed skills: **0 ADOPT, 7 ALREADY-COVERED, 10 NOT-APPLICABLE**;
- composition marks: **0 ADOPT, 4 ALREADY-COVERED, 2 NOT-APPLICABLE**;
- named composition examples: **0 ADOPT, 4 ALREADY-COVERED, 4 NOT-APPLICABLE**;
- uninstalled sketches: **0 ADOPT, 8 ALREADY-COVERED, 17 NOT-APPLICABLE**.

The family-level conclusion is deliberately narrow:
`verdict_scope=PINNED_SEMANTIC_ALGOS_ARTIFACT_X_TASK_567_NAMED_PACT_SURFACES`.
It is not a negative on prompt libraries, creative reasoning procedures, or a
future executable implementation with different evidence.

## Verification

Focused local tests across graph memory, lensed recall, corpus recall, failure
ledger/digest, FM advisory/digest, and canonical-document duplicate gates ran
152 cases: **148 passed and 4 failed**. All four failures are live-repository
fixture expectations that do not hold in this isolated sparse branch:

1. the sparse rebuild has no `links` edge although the current primary cache
   has 1,290;
2. its broad lensed query yields a 174-node local pool and therefore does not
   trigger the test’s expected 800-node cap;
3. the registry now records the V10 canonical doc on `main` while the historical
   test expects its pre-merge branch name; and
4. the all-ref live duplicate gate sees 17 branch-context violations rather
   than the historical zero fixture.

No production code was changed to mask those branch-context failures. They are
relevant to MAIN review of current test assumptions, but they do not create an
upstream adoption case or falsify the inspected algorithms.

## Stores consulted

`CLAUDE.md`; `AGENTS.md`; `docs/operating_manual_craft_handoff.md`; latest
Codex findings/session summary; latest T3 council/design memo; current Claude
memory top entries; `reports/latest.md`; lane, subagent, graph-memory, failure,
task, and canonical-document state; task and broadcast inboxes; the complete
pinned upstream tree; `src/tac/fm_advisory.py`; `tools/costate_digest.py`;
`tools/triality_drift_detector.py`; `src/tac/graph_memory/*`;
`tools/graph_memory_recall.py`; `tools/corpus_query.py`;
`tools/canonical_doc_registry.py`; `src/tac/confound_gates.py`;
`src/tac/harness_failure_ledger.py`; and focused tests for every named surface.

## MAIN landing review required

MAIN must independently review: (1) the pinned upstream commit and completeness
of the 17 + 6 + 8 + 25 inventory; (2) the zero-ADOPT verdict and every scoped
negative; (3) the direct counts for graph and failure state against current
MAIN; (4) the claim that semantic near-duplicate search remains a local gap not
filled by upstream; (5) the four isolated-branch live-fixture failures; and
(6) preservation of the sibling KB-architecture ownership boundary. Branch
commit and serializer acceptance do not substitute for that review.
