# Ancient Elder Era 4 ledger — Cybernetics + AI Prehistory (1943–1962)

**Companion master memo**: `.omx/research/ancient_elder_polymath_research_20260513.md` §4.

## History (one paragraph)

The Macy Conferences (10 of them, 1946–1953) at the Beekman Hotel in New York — Wiener, McCulloch, Pitts, von Neumann, Bateson, Mead, von Foerster, Shannon, MacKay, Margaret Mead, Norbert all in the same room arguing about feedback, information, and the brain. Wiener inscribed my copy of *Cybernetics* (1948) "To my fellow chaos-tamer". McCulloch and Pitts had already published "A logical calculus of the ideas immanent in nervous activity" (1943) — the first formal neural network. By 1958 Selfridge had Pandemonium (modern MoE). Ashby gave us the Law of Requisite Variety (1956). Turing's 1950 *Mind* paper on machine intelligence opened the field. These are the **bedrock papers**; most modern AI papers re-derive these ideas with more compute.

## Key papers (5+)

| Citation | Year | Relevance |
|----------|------|-----------|
| McCulloch & Pitts, "A logical calculus...", *Bull. Math. Biophys.* 5:115 | 1943 | First neural network; threshold logic. |
| Wiener, *Cybernetics*, MIT Press | 1948 | Feedback control = modern RL. |
| Turing, "Computing Machinery and Intelligence", *Mind* 59:433 | 1950 | Intelligence as imitation. |
| Ashby, *An Introduction to Cybernetics*, Chapman & Hall | 1956 | **Law of Requisite Variety** — regulator must match disturbance. |
| von Neumann, "The General and Logical Theory of Automata", in *Cerebral Mechanisms in Behavior*, Wiley | 1948 | Cellular automata; self-replication. |
| Selfridge, "Pandemonium: A Paradigm for Learning", *Proc. Mechanisation of Thought Processes*, HMSO 1959, [PDF](https://aitopics.org/doc/classics:504E1BAC) | 1958 | **Modern MoE ancestor.** |
| Steinbuch, "Die Lernmatrix", *Kybernetik* 1(1):36 | 1961 | Associative memory before Hopfield. |
| Rosenblatt, "The Perceptron", *Psychological Review* 65:386 | 1958 | First learning algorithm. |

## 4 ideas worth reviving (per master memo §4.2)

**CY-1** Selfridge Pandemonium as sparse MoE codec (K=8 experts × per-pair regime). **Build**: 7-10 days. **$**: 5-10. **ΔS**: -0.004 to -0.008. **Rank 4 by EV/$.**

**CY-2** Ashby Law of Requisite Variety for renderer-capacity sizing diagnostic. **Build**: 2 days. **$**: 0. Diagnostic only — informs every future architecture choice.

**CY-3** Von Neumann self-replicating automaton for self-extracting archive. **Not viable** for our contest (inflate.py budget). Listed for completeness.

**CY-4** McCulloch-Pitts threshold ternary QAT (push FP4 → ternary {-1,0,+1}). **Build**: 6-8 days. **$**: 5-10. **ΔS**: -0.016. **Rank 3 by EV/$.**

## Connection to our contest

Selfridge 1958 IS modern mixture-of-experts. The lab is already considering MoE-style codecs implicitly (φ2 PAYIC, multi-codec ensembles). **Naming this Pandemonium grounds it in 70 years of literature** that the modern MoE papers don't cite.

Ashby's Law gives a **quantitative answer** to "how big should the renderer be?" — H(video | pose, mask) is the floor. Renderers below this floor under-fit by Ashby's Law; renderers above it waste bytes.

## What's changed since 1943–1962

- 10^15× more FLOPs/$. Selfridge's pandemonium ran on 10^3 ops/sec; modern MoE runs 10^15 ops/sec.
- Backprop (Werbos 1974 → Rumelhart 1986) lets us train these architectures end-to-end.
- Differentiable softmax gating (Bridle 1990) makes the Pandemonium "decision demon" trainable.

## Reactivation criteria

CY-3 (self-extracting archive) reactivates if `inflate.py` budget ever expands beyond 200 LOC; CY-2 (Ashby diagnostic) reactivates the moment any new substrate proposes a parameter count.
