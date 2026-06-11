# Reinforcement learning as a lane — honest fit-assessment (2026-06-11)

**Authority:** assessment / lane-scoping, `[research-signal]`, NON-PROMOTABLE. Frontier UNMOVED
0.19109982 [contest-CPU], 177,169 B. This records WHERE RL fits the contest problem so it is a
considered lane, not re-discovered. Trigger: operator "curious if RL might play a role or be a new
lane" + an X/twitter link (`@___harald___` status 2063081524799672572) that is **paywalled (HTTP 402)**
— assessed from the 2026 RL-kernel literature it surfaced (Dr. Kernel 2602.05885, Kernel-Smith
2603.28342, GPU Forecasters 2605.31464, Automated Kernel Generation 2601.15727) + first principles.

## The map (RL value is inversely proportional to having a gradient or a convex solver)

| Surface | RL fit | Why | Current method |
|---|---|---|---|
| Renderer training (synth + latents) | **NO** | gradients work (margin-polytope hinge surrogate + differentiable scorer); RL dominated by SGD | gradient descent (the pointer-mover) |
| Exact d_seg (argmax-flip, non-diff) | **PARTIAL** | non-diff, but the hinge already concentrates ~99.4% of gradient on the boundary band; marginal RL gain over surrogate is small | margin-polytope hinge |
| Byte / bit / mode allocation | **WEAK** | convex Lagrangian/Pareto already near-optimal; don't swap a working solver for a sample-hungry one | score-domain Lagrangian (THE LAW) |
| **Waterfilling atom-selection (sequential, commutator-aware)** | **STRONG** | a true MDP: state=archive, action=apply evaluator-action atom, reward=−ΔS under byte budget, non-commuting actions → RL/MCTS beats greedy | **commutator-aware GREEDY (task #30, PENDING)** |
| **Witness-program search (AlphaEvolve / FunSearch class)** | **STRONG (long-horizon)** | evolve the inflate-time witness generator against the exact-scorer reward = the GOAL's "proof-carrying evaluator-equivalent program compiler" | not built (the decade bet) |
| Metal megakernel authoring (the tweet) | **SECONDARY** | RL/evolutionary search could author the fused Metal megakernel (Dr. Kernel / Kernel-Smith style) | the secondary local-throughput lane (3 agents in flight) |

## The two RL angles that are actually ON the score DAG

1. **Upgrade the waterfiller from greedy → RL/MCTS (near-term, task #30).** The atom-selection loop is
   already specified as "commutator-aware greedy selection." Greedy is provably suboptimal when actions
   don't commute (applying atom A changes the ΔS of atom B). An RL/MCTS sequential optimizer against the
   *exact* ΔS is the principled upgrade. This is the highest-EV near-term RL move because the
   infrastructure (atom proposers, exact-ΔS loop) is the SAME — RL only replaces the *selection policy*.
   Gated behind a working base (B2) per the existing DAG; it does not need a new lane, it sharpens #30.
2. **Witness-program evolution (long-horizon, the GOAL's compiler).** FunSearch/AlphaEvolve evolve
   programs against a scalar reward; here the program is the inflate-time witness generator and the
   reward is −(compliant S). This is the decade-scale "evaluator-equivalent program compiler" the GOAL
   memo names. Big bet; aligns with the 10-year autonomous-research horizon, not this week.

## The hard part (the anti-reward-hacking guard — non-negotiable)

RL against the exact scorer is (a) **sample-expensive** — each reward = a scorer forward, or worse a
byte-closed `evaluate.py` eval — and (b) **reward-hackable** — a maximizer of −S finds
evaluator-corrupting / non-compliant witnesses (scorers-at-inflate-time, >30-min, hidden sidecars,
GT-decode tricks) unless compliance is baked INTO the reward. Our null-space / scorer-inverse work is
already "exploit the reward"; RL amplifies both the win and the cheating. **Therefore the reward MUST be
the exact COMPLIANT S (THE-LAW-screened, real `evaluate.py` semantics), never a proxy** — the
"proof-carrying" part of the program-compiler is the compliance certificate, not optional. This is the
same anti-reward-hacking discipline that governs the whole lab (no proxy scores, MPS never, GT decode
only via frame_utils.yuv420_to_rgb).

## Verdict (means/ends-aware, DAG-first)

RL is a **real lane**, correctly aimed at the **sequential discrete decisions where we currently
greedy-search** (the waterfiller, #30) and the **long-horizon program search** (the GOAL's compiler) —
NOT at renderer training (gradients win) and NOT at byte allocation (convex solver wins). The tweet's
likely topic (RL-authored kernels) attaches to the **secondary** megakernel lane, not the score DAG.
**Recorded as a candidate lane; NOT built this turn** — the imminent pointer-mover is the
capacity-de-risked paid n600 (gradient-trained), and the operator just reaffirmed DAG-first. RL's
near-term entry point is to sharpen the already-pending #30 waterfiller selection policy once a working
B2 base exists, with the exact compliant S as the only admissible reward.

**Sources:** [Dr. Kernel (RL for Triton)](https://arxiv.org/abs/2602.05885) · [Kernel-Smith](https://arxiv.org/pdf/2603.28342) · [GPU Forecasters](https://arxiv.org/abs/2605.31464) · [Automated Kernel Generation](https://arxiv.org/pdf/2601.15727). Tweet itself paywalled (HTTP 402).
