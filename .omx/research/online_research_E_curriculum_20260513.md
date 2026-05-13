# Online research ledger — Domain E: Curriculum learning bleeding edge

Per-paper notes; 8 entries.

---

## E.1 — Curriculum Learning (Bengio-Louradour-Collobert-Weston, ICML 2009; foundational)
- **Empirical claim**: Train on easy examples first, then hard. Faster convergence and better generalization on toy tasks + NLP.
- **Relevance**: Foundational. Our HNeRV training does an IMPLICIT curriculum via EMA decay and Lagrangian rho schedules.

## E.2 — Self-Paced Learning (Kumar-Packer-Koller, NeurIPS 2010)
- **Empirical claim**: Auto-select training examples by model loss; threshold tightens over training.
- **Relevance**: Direct application: select frames by current proxy loss; tighten as training progresses.

## E.3 — Neural Bellman-Ford Networks / NBFNet (Zhu-Zhang-Xhonneux-Tang, NeurIPS 2021)
- **arXiv**: https://arxiv.org/abs/2106.06935
- **Repo**: (search NBFNet)
- **Empirical claim**: GNN framework parameterizing generalized Bellman-Ford with INDICATOR/MESSAGE/AGGREGATE neural ops; SOTA on knowledge-graph link prediction.
- **Relevance**: TOP-10 actionable #10. The substrate for our shortest-path curriculum planner. The (hyperparam-state, training-step) graph is naturally a Bellman-Ford lattice.

## E.4 — Applying NBFNet to SSSP (2024)
- **Link**: https://www.scitepress.org/Papers/2024/124258/124258.pdf
- **Empirical claim**: Predecessor Prediction variant of NBFNet for shortest path.
- **Relevance**: Direct application of #3 to our curriculum-planner problem.

## E.5 — Population-Based Training (PBT; Jaderberg et al. 2017)
- **arXiv**: https://arxiv.org/abs/1711.09846
- **Empirical claim**: Bandit-like exploration of hyperparameters during training; copy-and-perturb winning configs.
- **Relevance**: Natural fit for our parallel-dispatch actuator + autopilot loop. Each parallel dispatch is one PBT agent.
- **Integration cost**: ~2 days dev to wire PBT-style copy-and-perturb into `tools/parallel_dispatch_top_k.py`.

## E.6 — Progressive Fourier Neural Representation / PFNR (Kim et al., ICLR 2024)
- **arXiv**: https://arxiv.org/abs/2306.11305
- **Repo**: https://github.com/ihaeyong/PFNR
- **Empirical claim**: Find Lottery Tickets in Fourier space; Fourier Sub-Network Operator (FSO) decomposes INR into sin+cos parts; sparsify per session for new-video addition; **lossless decoding of previous videos** via frozen weights.
- **Relevance**: TOP-20 EUREKA #16. Direct application: continual-learning curriculum where new training session never destroys old anchor.

## E.7 — Forget-free Winning Subnetworks for Video (2023-2024)
- **arXiv**: https://arxiv.org/html/2312.11973v3
- **Relevance**: Extension of PFNR; directly applicable to our multi-anchor preservation problem.

## E.8 — Hard-pair mining / Focal Loss (Lin et al. 2017)
- **Empirical claim**: Focal loss down-weights easy examples; effectively implicit curriculum.
- **Relevance**: Already implicit in our score-aware loss design via per-pair weighting.

## E.9 — Bilevel Optimization for Curriculum (various; 2023-2024)
- **Reference**: Various bilevel formulations of curriculum-as-meta-learning.
- **Relevance**: Frames curriculum as upper-level optimization over a lower-level training problem; matches our solver-as-living-thing CLAUDE.md non-negotiable.

## E.10 — Hessian-aware Curriculum (research direction; 2024)
- **Reference**: Hessian-trace as curriculum-difficulty signal; harder examples have higher Hessian curvature.
- **Relevance**: Combines with Sophia (Domain C #2) which already estimates diagonal Hessian.

---

## Curriculum-shortest-path planner design sketch (the operator-mentioned eureka)

Per CLAUDE.md "parallel-dispatch first" non-negotiable, the planner is a typed-atom emitter; the actuator (`tools/parallel_dispatch_top_k.py`) fans out dispatches.

Concrete planner:
1. Nodes = (rho, lr, batch-size, lagrangian-weights, EMA-decay, ...) tuples plus a "step-count" coordinate.
2. Edges = transitions (Δrho, Δlr, etc.). Edge cost = predicted distortion-cost minus predicted bytes-saved at that node.
3. Source node = current state. Sink = target (e.g., d_pose ≤ 1e-5 AND archive_bytes ≤ 75KB).
4. NBFNet learns edge cost from past `[contest-CUDA]` anchors (continual-learning posterior is the training signal).
5. Bellman-Ford traversal returns the cheapest predicted path.
6. Top-K paths feed `parallel_dispatch_top_k.py`.

The PROBE-DISAMBIGUATOR is the empirical anchor returned: posterior is updated, planner re-fits, next batch dispatched.

## Follow-up reads:
- Various 2024-2025 papers on curriculum-as-RL (HJB optimal control, see Domain F).
- "Soft-TransFormers for Continual Learning" (2024): https://arxiv.org/html/2411.16073v1
