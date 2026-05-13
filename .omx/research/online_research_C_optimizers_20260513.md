# Online research ledger — Domain C: Optimizer bleeding edge

Per-paper notes; 11 entries.

---

## C.1 — Muon (Keller Jordan, 2024)
- **Author**: Keller Jordan + collaborators
- **Blog**: https://kellerjordan.github.io/posts/muon/
- **Repo**: https://github.com/KellerJordan/Muon
- **NanoGPT speedrun stack**: https://github.com/KellerJordan/modded-nanogpt
- **Scalability tech report (Liu et al. 2025)**: https://arxiv.org/pdf/2502.16982
- **Empirical claim**: SGD-momentum + Newton-Schulz orthogonalization of 2D weights. **~35% speedup vs AdamW on NanoGPT**; **~2× compute-optimal efficiency** at scale. Used in CIFAR-10 + NanoGPT speed records.
- **Architecture**: Apply momentum to grad, orthogonalize, scale by aspect-ratio, apply. Hidden 2D weights only; AdamW for 1D + heads.
- **Relevance**: TOP-10 actionable #2. HNeRV-family conv weights ARE 2D hidden weights.
- **Integration cost**: ~1 day; Muon repo is small (~150 LOC).

## C.2 — Sophia (Liu et al., ICLR 2024)
- **Authors**: Hong Liu, Zhiyuan Li, David Hall, Percy Liang, Tengyu Ma (Stanford)
- **Venue**: ICLR 2024
- **arXiv**: https://arxiv.org/abs/2305.14342
- **Repo**: https://github.com/Liuhong99/Sophia
- **Empirical claim**: Diagonal-Hessian preconditioner; clipped update; **2× speedup vs Adam on 125M-1.5B GPT** in #steps + total compute + wall-clock.
- **Architecture**: Hessian estimate every K iters (~10 steps); element-wise clipping.
- **Relevance**: TOP-10 actionable #7. Alternative or complement to Muon.
- **Integration cost**: ~0.5 day drop-in.

## C.3 — Lion (Chen et al., ICLR 2023)
- **Authors**: Xiangning Chen et al. (Google Brain)
- **Venue**: NeurIPS 2023 + originally ICLR 2023
- **arXiv**: https://arxiv.org/abs/2302.06675
- **Repo**: https://github.com/lucidrains/lion-pytorch (community port)
- **Empirical claim**: Sign-momentum (no second moment); 2-15% wall-clock speedup vs AdamW; 2.3× compute saving on diffusion FID.
- **Theory**: Chen-Liang follow-up arXiv:2310.05898 "Lion Secretly Solves Constrained Optimization" — Lyapunov analysis.
- **Relevance**: Memory-efficient (no v_t), simple. Useful for memory-constrained substrate engineering.
- **Integration cost**: ~0.5 day; 1 file.

## C.4 — SOAP (Vyas-Janson et al., 2024)
- **Authors**: Nikhil Vyas, Depen Morwani, Rosie Zhao, Itai Shapira, David Brandfonbrener, Lucas Janson, Sham Kakade
- **arXiv**: https://arxiv.org/abs/2409.11321
- **Empirical claim**: Shampoo + Adam in Shampoo's preconditioner eigenbasis. **40% iteration reduction + 35% wall-clock save vs AdamW**; 20% over Shampoo.
- **Architecture**: Equivalence: Shampoo (1/2 power) = Adafactor in eigenbasis of Shampoo preconditioner.
- **Relevance**: Strong competitor to Muon + Sophia. Should be in the probe-disambiguator (master synthesis D-1).
- **Integration cost**: ~1 day.

## C.5 — Shampoo (Gupta-Koren-Singer, ICML 2018; revived 2024)
- **arXiv**: https://arxiv.org/abs/1802.09568
- **Empirical claim**: Kronecker-factored preconditioner; structure-aware. Used in Gemini-1.5 Flash 2024 training.
- **Relevance**: Reference for SOAP. Less practical as standalone for our small substrates.
- **Integration cost**: ~2 days.

## C.6 — SOAP+Muon (Vyas et al. 2025)
- **arXiv**: https://nikhilvyas.github.io/SOAP_Muon.pdf
- **Empirical claim**: Iterative whitening + Muon Newton-Schulz; speedups stack.
- **Relevance**: TOP-of-queue follow-up read after Muon baseline lands.
- **Integration cost**: ~1.5 days dev.

## C.7 — MuonBP (block-periodic Muon, 2024-2025)
- **OpenReview**: https://openreview.net/forum?id=mHouLSUQP5
- **Empirical claim**: Block-periodic orthogonalization → faster than vanilla Muon.
- **Relevance**: Micro-optimization of #1.
- **Integration cost**: ~1 day.

## C.8 — Muon Newton-Schulz Coefficient Optimization (Cesista 2025)
- **Blog**: https://leloykun.github.io/ponder/muon-opt-coeffs/
- **Empirical claim**: 1-2% extra efficiency by tuning Newton-Schulz coefficients.
- **Relevance**: Micro-opt; only matters at the wall-clock-bound frontier.

## C.9 — Tuddenham et al., 2022 (Orthogonal-SGDM, Muon ancestor)
- **Reference**: Mentioned in Keller Jordan's blog as the predecessor optimizer using SVD-orthogonalization.
- **Relevance**: Reference for Muon's theory.

## C.10 — AdaFactor (Shazeer-Stern, 2018)
- **Reference**: Memory-efficient Adam variant; used in Shampoo/SOAP analogies.
- **Relevance**: Memory-budget baseline.

## C.11 — Stochastic Gradient Langevin Dynamics (SGLD) — see Domain F
Cross-link: our scaffolded LangevinOptimizer (session insight #7) draws directly from SGLD literature.

---

## Optimizer probe-disambiguator design sketch (per CLAUDE.md D-1 surfaced)

To pick among Muon / Sophia / SOAP, build `tools/probe_optimizer_disambiguator.py`:
- Smoke-train T1-Balle for 200 epochs on each
- Compare (a) wall-clock to fixed loss, (b) final loss at fixed wall-clock, (c) memory footprint
- Run on **MPS-research-signal** axis (FREE) per CLAUDE.md MPS-research-signal rule
- Promote winner to next contest-CUDA dispatch

## Follow-up reads:
- "Understanding and Improving the Shampoo Optimizer via KL Minimization" (2025): https://arxiv.org/html/2509.03378v1
- "4-bit Shampoo for Memory-Efficient Network Training" (Wang et al., NeurIPS 2024): https://proceedings.neurips.cc/paper_files/paper/2024/file/e5b4633454cb2174779d294ccda02318-Paper-Conference.pdf
- Jeremy Bernstein "Deriving Muon": https://jeremybernste.in/writing/deriving-muon
- Tyler Romero "NanoGPT Speedrun Living Worklog": https://www.tylerromero.com/posts/nanogpt-speedrun-worklog/
