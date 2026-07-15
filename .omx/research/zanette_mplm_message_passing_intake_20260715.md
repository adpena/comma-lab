# Intake: Zanette Labs "Message Passing Language Models" (MPLM) — warm-start-from-divergence read

**Source:** https://zanette-labs.github.io/mplm.github.io/ (operator-routed 2026-07-15).
Per `PAPER_WARM_START_FROM_DIVERGENCE`: trace the fork, import what survives OUR premises.

## What it is

LLM inference-time orchestration: persistent semi-independent threads with learned
`<spawn>/<send>/<recv>/<stop>` primitives, point-to-point messaging instead of fork-join's
centralized parent. Context-complexity law: serial CoT O(T·N·k·M) · fork-join O(T·N·M) · MPLM
**O(T·k·M)** — a Θ(N/k) reduction when subtasks depend only on k sparse neighbors. Measured:
25×25 Sudoku 72% (peak context 30× smaller, scaling exponent α≈1.1–1.2 vs 1.8 monolithic);
3-SAT ~2.6× latency via PREEMPTION (kill siblings on first solution — impossible under fork-join);
LongBench-v2 1.7–2.2× latency at equal-or-better accuracy. Qwen3-0.6B SFT (<48 H100-h) for the
solvers; 30B zero-shot for QA.

## The assumption fork (why the witness gets nothing)

Their object is an inference-time REASONER decomposing discrete tasks. Our witness is a continuous
coord-INR trained by dense gradient descent — there is no inference-time thread decomposition to
learn. The constraint-graph echo (Sudoku message passing ≈ belief propagation on sparse constraint
graphs; our argmax partition IS a constraint field on the Laguerre/RAG adjacency) is ALREADY covered
by our boundary-math RAG (#52) + Morse-Smale machinery — importing BP-style message passing there
would re-derive what the level-set flow already does variationally. Verdict on the witness vehicle:
NOT APPLICABLE, divergence traced.

## What SURVIVES: the fork-join bottleneck is OUR fleet's bottleneck

The paper's named failure mode — "all coordination is centralized: every piece of information must
flow back through a single parent thread" — is EXACTLY the main-agent hub-and-spoke we run: every
arm reports through main's context window (our O(T·N·M)). Today's session already used both MPLM
primitives ad hoc: (a) PREEMPTION — the duplicate-arm de-conflicts and first-writer-wins kills
(65822, the stale fire daemon) are exactly "stop siblings when the answer exists"; (b) POINT-TO-POINT
— arm↔arm coordination via the codex inbox + main-brokered SendMessage relays (main as router, not
processor). The import is the FORMALIZATION + the complexity argument:

1. **k-sparse arm topology as a design rule:** when chartering a wave, declare each arm's ≤k named
   neighbors (the arms whose files/results it consumes) and route those messages DIRECT
   (inbox/SendMessage with the sibling id in the charter), reserving main for verdicts + operator
   surface. Cuts main's context load Θ(N/k) — the practical ceiling on fleet width we keep hitting
   at session limits.
2. **Preemption as a standing charter clause:** arms racing the same receipt (A/B arms, duplicate
   probes) get an explicit "first-GREEN preempts siblings via inbox stand-down" clause instead of
   main discovering duplicates post-hoc (today's 3 incidents).
Both fold into the CFL coherent-parallelism discipline
([[coherent_parallelism_is_cfl_codex_worktree_isolation_20260714]]) as refinements, NOT a new
orchestration layer (per the coherence-by-default non-negotiable: rules, not frameworks). No build
owed; apply at next wave-charter time.

**Pointer honesty:** apparatus intake/means; pointer 0.19108 UNMOVED. papers-checked:
zanette_mplm_2026 → APPARATUS-IMPORT-2 (k-sparse charter topology · preemption clause),
witness NOT-APPLICABLE (constraint-graph echo already held by RAG/Morse-Smale).
