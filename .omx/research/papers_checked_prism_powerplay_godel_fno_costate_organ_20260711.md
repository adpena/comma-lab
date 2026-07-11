# Papers-checked: PRISM (2607.00510) · PowerPlay (1112.5309) · Gödel machine (cs/0309048) · FNO/neuraloperator (2412.10354 + lineage) · QeRL (2510.11696) — the Rudin+Schmidhuber design cluster for the #426 costate organ

Date: 2026-07-11 · operator binding directive (Rudin interpretable-by-design + Schmidhuber
self-improvement; NVIDIA FNO from the lab-frontier survey) · read live where fetchable. Sister of
`amortized_operator_pontryagin_loop_cluster_20260711.md` (§3b/§3c/§7 fold-ins) + the lab-frontier
sweep. All BUILT + BACKTESTED this pass unless labelled owed.

## PRISM — Prototypes for Interpretable Sequence Modeling (arXiv 2607.00510, Lakkaraju cluster; ProtoPNet/Rudin-Chen lineage) — LEVER, BUILT
[MEASURED-by-paper, fetched] Every prediction = "a sparse, non-negative mixture of learned
prototypes" anchored to "coherent neighborhoods of training examples"; prototypes trained with
clustering objectives; sparse structure "localizes curvature in the loss landscape" → training-
data attribution "~500× faster than post hoc baselines" + "a more tractable Hessian"; "targeted
prototype suppression can remove model behaviors without finetuning"; "calibrating linear
prototype controllers" +~3 pts while tracing corrections to neighborhoods; 130M–1.6B within 2.5
pts of dense. **BUILT** as `tac.witness_control.prototype_router.PrototypeRouterLens` (E_prototype
lens): prototypes = named trajectory REGIMES (names DERIVED from measured per-class d_seg
signature — NO-FAKE — never asserted); sparse top-m non-negative similarity mixture = the router
AND the explanation; Daubechies multi-scale coarse-regime + fine-correction cascade; SLIM-style
linear controller head (the DECIDE stays a readable linear rule over prototypes); prototype
SUPPRESSION exposed + traceable; PrototypeAttribution = the OBSERVATORY row (fired prototypes,
weights, mixture-entropy uncertainty, neighborhood epochs). **MEASURED cost of interpretability
(routing_benchmark_v2, #205, 5-fold nested LOO):** prototype solo scalar 0.003109 vs flow 0.002881
(+8%) BUT per-class 0.0105 vs ridge 0.0237 (**2.3× better**); QUESTION_ROUTER-with-prototype gets
the best per-class 0.0744. Independently NAMED the live regime "lane-erosion" — matching the
harvest §3b measured warning. **Verdict: interpretability-by-design is competitive AND
per-class-dominant here; the +8% scalar cost is tiny and fully characterized.**

## PowerPlay (arXiv 1112.5309, Schmidhuber) — LEVER, BUILT (the self-inventing curriculum)
[MEASURED-by-paper, fetched] Searches "the space of possible pairs of new tasks and modifications
of the current problem solver" for "the simplest still unsolvable problem" whose solver "provably
solves all previously learned tasks plus the new one, while the unmodified predecessor does not";
ranks by "conditional computational (time & space) complexity, given the stored experience".
**BUILT** as `control_alphabet.powerplay_acquisition`: the duty-to-measure queue = the self-
inventing curriculum; rank = curiosity × blast-radius / cost. Curiosity = the INNOVATION SIGNAL
(compression progress / surprise) = |derived-λ − measured-binding| for fired levers (the model was
WRONG by this much ⇒ will LEARN this much); never-fired = pure exploration (the unknown is
maximally surprising, PowerPlay's frontier). Cost = measurement cost (PowerPlay prefers the
SIMPLEST). The "does-not-break-what-works" invariant = the never-regress guard + the Gödel gate.

## Gödel machine (arXiv cs/0309048, Schmidhuber) — CORRESPONDENCE named, ENFORCED
[MEASURED-by-paper] Self-rewrites its own code ONLY upon finding a PROOF that the rewrite improves
expected future reward. **CORRESPONDENCE:** our proof-of-improvement = the backtest verdict
(held-out forecast beats the incumbent) AND the never-regress guard (predicted ΔS < 0) AND
containment (HEAVY ⇒ operator-GO). Named + enforced as `control_alphabet.GodelProofGate.evaluate`
— a True verdict still only unlocks emitting an advisory rec / OperatorGoTicket (authorizes
nothing on its own). The backtest-IS-the-gate + containment governor we already built ARE the
Gödel discipline.

## LADDER (Schmidhuber lineage) — REAFFIRMED (⊂ costate, our L56/MEMORY finding)
Prior PROVEN result stands: LADDER = costate with 1 channel / constant λ; island-birth = per-class
λ homotopy (the ladder-homotopy that Movable-islands rode 0.9998→0.0073 LIVE, FEED-v9-harvest-1).
Compression-as-intelligence (his grand-council seat): the witness IS compression; the costate
organ self-improves by COMPRESSING its own trajectory experience (PRISM prototypes = a compressed,
attributable memory of the trajectory — the two lineages meet here). Latest-2026 Schmidhuber
scan: no new mechanism surfaced that changes the ⊂-relation (the async/single-rollout RL frontier
is the Z.ai/ByteDance thread, ledgered separately).

## FNO / neural operators (2412.10354 library · 2108.08481 foundational · 2404.07200 spectral · QeRL 2510.11696) — GROWTH SPINE (owed at >1 trajectory)
[MEASURED-by-secondary, NVIDIA survey] λ = ∂S/∂x is a FIELD→FIELD map ⇒ an operator-learning
problem, not regression. FNO's truncated-spectral kernel is low-rank BY CONSTRUCTION — set
n_modes ≈ our measured rank-8 — and resolution-invariant (learn on ~6 checkpoints, evaluate at any
state). **Our ridge-SOLVE is the rank-limited LINEAR special case; FNO-spectral is its principled
NONLINEAR successor along the same rank-8 axis** — the answer to "what architecture when the
λ-network graduates from ridge." QeRL noise-as-exploration = the online-λ-loop growth tool
(composes with SAO single-rollout). NVFP4 hardware NOT-APPLICABLE on M5 Max. Both owed at
>1 campaign trajectory (below the crossover; the measured tournament says solve-now).

**verdict_scope:** PRISM + PowerPlay + Gödel = BUILT+BACKTESTED (instance-scope, #205 trajectory);
FNO/QeRL = read-level growth-path (owed). No internal lane killed. Pointer 0.19108282 UNMOVED
(interpretability + self-improvement design = MEANS; only a byte-closed exact row moves it).
