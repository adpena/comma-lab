# Online research ledger — Domain B: LoRA / DoRA / adapter literature (post-2023)

Per-paper notes; 12 entries. All claims `[literature-prediction]` or `[third-party-empirical:<paper>]`. Cross-link: master synthesis.

---

## B.1 — LoRA (Hu et al., ICLR 2022; foundational)
- **Authors**: Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft)
- **arXiv**: https://arxiv.org/abs/2106.09685
- **Empirical claim**: ΔW = BA decomposition with rank r ≪ min(d,k); ~10,000× fewer params at downstream-task perf parity.
- **Relevance**: foundation primitive for the entire family. Our $1-3 PR95 adapt lane is a LoRA application.
- **Integration cost**: already wired in PEFT; ~0.5 day to scaffold a per-frame LoRA on PR95 frozen base.

## B.2 — DoRA (Liu et al., ICML 2024 Oral)
- **Authors**: Shih-Yang Liu, Chien-Yi Wang, Hongxu Yin, Pavlo Molchanov, Yu-Chiang Frank Wang, Kwang-Ting Cheng, Min-Hung Chen (NVIDIA)
- **Venue**: ICML 2024 (Oral)
- **arXiv**: https://arxiv.org/abs/2402.09353
- **Repo**: https://github.com/NVlabs/DoRA
- **NVIDIA page**: https://research.nvidia.com/publication/2024-07_dora-weight-decomposed-low-rank-adaptation
- **Project page**: https://nbasyl.github.io/DoRA-project-page/
- **Empirical claim**: W = m · (V/||V||) decomposition; LoRA updates direction V, magnitude m trained separately. Outperforms LoRA on LLaMA/LLaVA/VL-BART. Merges back into pre-trained weight at inference.
- **Relevance**: TOP-10 actionable #1. Drop-in for our PR95 adapt lane.
- **Integration cost**: ~0.5 day.

## B.3 — PiSSA (Meng et al., NeurIPS 2024 Spotlight)
- **Authors**: Fanxu Meng, Zhaohui Wang, Muhan Zhang (Peking Univ.)
- **Venue**: NeurIPS 2024 Spotlight
- **arXiv**: https://arxiv.org/abs/2404.02948
- **Repo**: https://github.com/GraphPKU/PiSSA / https://github.com/MuLabPKU/PiSSA
- **Empirical claim**: Init A,B from principal SVD components of W; remainder frozen. Mistral-7B GSM8K **+5.16% vs LoRA** at same trainable budget. Init takes seconds via fast SVD.
- **Relevance**: TOP-10 actionable #8. Combines with DoRA naturally (PiSSA-DoRA).
- **Integration cost**: ~0.5 day; merged into HuggingFace PEFT.

## B.4 — QLoRA (Dettmers et al., NeurIPS 2023)
- **Authors**: Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, Luke Zettlemoyer (UW)
- **Venue**: NeurIPS 2023
- **arXiv**: https://arxiv.org/abs/2305.14314
- **Repo**: https://github.com/artidoro/qlora
- **Empirical claim**: 4-bit NormalFloat (NF4) + Double Quantization + Paged Optimizers. 65B param model on single 48GB GPU. Matches BFloat16 full-FT perf.
- **Relevance**: Critical for any 4-bit QAT adapter lane. NF4 init reduces quantization error vs INT4 ~1pp.
- **Integration cost**: bitsandbytes is the canonical library; ~0.5 day dev.

## B.5 — VeRA (Kopiczko et al., ICLR 2024)
- **Authors**: Dawid J. Kopiczko, Tijmen Blankevoort, Yuki Markus Asano (Univ. Amsterdam)
- **Venue**: ICLR 2024
- **arXiv**: https://arxiv.org/abs/2310.11454
- **Project**: https://dkopi.github.io/vera/
- **HF docs**: https://huggingface.co/docs/peft/en/package_reference/vera
- **Empirical claim**: Single pair of frozen random shared low-rank matrices + per-layer trainable scaling vectors d,b. **10× fewer params than LoRA at same perf**. On E2E with GPT2 Medium+Large, outperforms LoRA with 3-4× fewer params.
- **Relevance**: TOP-20 EUREKA #12. Aggressive parameter reduction — possibly fits sub-1KB adapter budgets.
- **Integration cost**: ~0.5 day; merged into PEFT.

## B.6 — LoRA+ (Hayou-Ghosh-Yu, ICML 2024)
- **Authors**: Soufiane Hayou, Nikhil Ghosh, Bin Yu (UC Berkeley)
- **Venue**: ICML 2024
- **arXiv**: https://arxiv.org/abs/2402.12354
- **Empirical claim**: Asymmetric LR for A vs B matrices in LoRA; well-chosen ratio. 2× speedup, 1-2% perf gain at same compute.
- **Relevance**: TOP-20 EUREKA #11. Free win — ~30 min config change.
- **Integration cost**: ~0.5 hour (single hyperparameter).

## B.7 — AdaLoRA (Zhang et al., ICLR 2023)
- **Authors**: Qingru Zhang, Minshuo Chen, Alexander Bukharin, Nikos Karampatziakis, Pengcheng He, Yu Cheng, Weizhu Chen, Tuo Zhao
- **Venue**: ICLR 2023
- **arXiv**: https://arxiv.org/abs/2303.10512
- **Repo**: https://github.com/QingruZhang/AdaLoRA
- **Empirical claim**: SVD parameterization of incremental updates; importance-aware rank allocation; better than LoRA at low budgets.
- **Relevance**: Adaptive-rank allocation natural fit for our bit-allocator hook. Especially relevant for non-uniform layer importance in HNeRV-family.
- **Integration cost**: ~1 day; in PEFT library.

## B.8 — LoftQ (Li et al., ICLR 2024)
- **Authors**: Yixiao Li, Yifan Yu, Chen Liang, Pengcheng He, Nikos Karampatziakis, Weizhu Chen, Tuo Zhao (Microsoft)
- **Venue**: ICLR 2024
- **arXiv**: https://arxiv.org/abs/2310.08659
- **MSR blog**: https://www.microsoft.com/en-us/research/blog/loftq-reimagining-llm-fine-tuning-with-smarter-initialization/
- **Empirical claim**: Alternating quantize-and-low-rank-fit init. Outperforms QLoRA at all precisions; closes 4-bit FT gap with full-precision FT.
- **Relevance**: TOP-20 EUREKA #17. Critical for 4-bit adapter regime — combines with PR95-adapter trailer lane.
- **Integration cost**: ~0.5 day.

## B.9 — LoRA-FA (Zhang et al., 2023)
- **arXiv**: https://arxiv.org/html/2308.03303v3
- **Empirical claim**: Freezes A matrix (memory-efficient); only B trained.
- **Relevance**: Memory-budget win; secondary if not memory-bound.
- **Integration cost**: ~0.5 day.

## B.10 — ReLoRA (Lialin et al., 2023)
- **arXiv**: https://arxiv.org/abs/2307.05695
- **Empirical claim**: Restart LoRA periodically + merge — effectively achieves higher rank via iterative low-rank training.
- **Relevance**: Compounding-rank trick; potentially useful for HNeRV-family which is parameter-light but training-step bound.
- **Integration cost**: ~1 day.

## B.11 — OFT / BOFT (orthogonal fine-tuning)
- **Reference**: Hashimoto et al. and follow-ups 2023-2024.
- **Empirical claim**: Constrains updates to orthogonal transforms; preserves pre-trained features better than LoRA.
- **Relevance**: Theoretically pleasing; preserves pre-trained spectrum. Unclear empirical gain on small INR-family models.
- **Integration cost**: ~1-2 days.

## B.12 — KronA (Edalati et al., 2022; CONV-LoRA variants)
- **Reference**: Kronecker-product adaptation; sister of OFT.
- **Empirical claim**: Kronecker factorization for parameter efficiency.
- **Relevance**: Particularly relevant for CONV layers in HNeRV-family substrates.
- **Integration cost**: ~1.5 days.

---

## Follow-up reads in domain B:
- Sebastian Raschka's DoRA-from-scratch tutorial: https://magazine.sebastianraschka.com/p/lora-and-dora-from-scratch
- "Improving LoRA: Implementing DoRA from Scratch" — pedagogical reference
- Hayou follow-up work (asymmetric LR theory)
- HuggingFace PEFT documentation: https://huggingface.co/docs/peft
- FinLoRA (financial LoRA documentation): https://finlora-docs.readthedocs.io/en/latest/lora_methods/qlora.html
