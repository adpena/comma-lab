---
name: Lottery Ticket Hypothesis — REVISED to DEPLOY-WORTHY at $200-500 budget (Lane J-IMP)
description: 2026-04-28 initial verdict RESEARCH-PARK was BUDGET-GATED ($25 cap). REVISED PM same day after user clarified budget is $200-500. Full IMP $40-80 NOW IN BUDGET. Jack-from-skunkworks ranked Lane J-IMP TOP-3 highest-EV ($25/60h, predicted [0.85, 1.00]). Compose with Lane Ω-V2 quantization: 88K → 9K active × 4 bits = 4.5KB renderer (vs 170KB current). Quantizr stopped at vanilla 5-stage QAT — open window.
type: project
originSessionId: 9518b12a-1bdd-4f5a-8ed1-c1def0bae30c
---
## Paper identity (verified)
- arXiv 1803.03635 (Frankle & Carbin 2019, ICLR best paper)
- "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks"
- Modern evolution surveyed: Frankle 2020 stabilization + 2021 SNIP critique + 2024 surveys

## Original claim
Dense randomly-initialized networks contain subnetworks that, trained in isolation, reach test accuracy comparable to original. Tested on LeNet/MNIST + Conv-2/4/6, VGG-19, ResNet-18 on CIFAR-10. Found tickets at 10-20% of original size matching dense accuracy.

## 2020 stabilization fix (Frankle 1912.05671 + 2003.03033)
Original IMP FAILED at ResNet-50/ImageNet scale. Fix = **rewind to early-training epoch** (0.1%-7% through training, NOT init). With rewinding, finds 80% sparsity ResNet-50 tickets matching ImageNet accuracy.

## 2021 critique (Frankle ICLR 2021, arXiv 2009.08576)
SNIP/GraSP/SynFlow pruning-at-init methods are STRUCTURALLY SUSPECT — randomly shuffling weights pruned per layer preserves or improves accuracy. Per-weight decisions don't matter; only per-layer ratios do. Cheap one-shot pruning ≈ random pruning.

## 2024 state
LTH is alive but refined. IMP genuinely finds higher-quality minima (smaller-volume better-generalizing basins) — not just "matching dense given equal compute." NO 2024 EVIDENCE FOR LTH AT SUB-100K PARAM SCALE. All canonical results are ≥1M params.

## Cost profile
- IMP to 80% sparsity: 5 cycles × 16h = 80h = $20
- IMP to 90% sparsity: 10 cycles × 16h = $40
- IMP to 95% sparsity (Quantizr-territory): 15 cycles × 16h = $60
- Hardware budget: $25 cap. Full IMP exceeds by 1.6-2.4×.

## Why NOT applicable (math)
- 80K params × FP4 = 40KB renderer.bin before brotli
- Sparse-CSR at our scale: 2B index + 0.5B FP4 per nonzero
- 80% sparsity store = 16K × 2.5B = 40KB → tied with dense
- Sparse export only wins at 95%+ sparsity → unreachable in budget
- Expected score gain at 95% sparsity (no quality loss): -0.027 rate points
- Same $40 on Lane Ω-V2 + Lane EBR delivers same wedge with proven mechanisms

## Per-lane scores (1-10)
- Lane Ω-V2 (per-element learnable bits): 3/10 — already a continuous-relaxation pruner
- Lane S (self-compress): 3/10 — same; self_compress.py:114 = `prune_mask = (bits < 0.5)`
- Lane W (hard-pair self-compress): 2/10 — orthogonal mechanism
- Lane F-V5 (hardware FP8): 5/10 — sparse + FP8 multiply IF storage codec exploits zero-runs (currently doesn't)
- Lane EBR (entropy bottleneck): 7/10 — entropy coder loves zero-heavy distributions
- Lane I (Cool-Chic): 2/10 — neural codec replaces renderer
- 5-stage QAT pipeline: 4/10 — could insert IMP as 6th stage but cost-prohibitive
- NEW LTH lane (Lane LT-CHEAP one-shot prune + sparse-CSR): 5/10 — see strategic note

## Lane LT-CHEAP design (if revisited)
One-shot global magnitude prune at sparsity sweep [50%, 70%, 85%]. Fine-tune 1500 steps each (LR rewinding 0.1× peak). Sparse-CSR export (uint16 indices + FP4 values). $2.10 / 4h on Vast.ai. Realistic predicted band [1.04, 1.07] — neutral. Pessimistic [1.10, 1.20]. Optimistic [1.00, 1.05].

## Strategic conclusion (anti-arbitrariness)
LTH at our 80K param scale is unproven (no 2024 evidence at sub-100K). LTH costs exceed budget. Our existing learnable-bit-alloc stack subsumes LTH's value proposition for fixed-budget rate attacks. Where LTH MIGHT matter: if Lane I (Cool-Chic) ships a new neural codec architecture not yet bit-allocation-trained, IMP-on-Cool-Chic could be a fast follow-up. Defer until Lane I lands.

## Cross-references
- `.omx/research/lane_g_v3_stacking_skunkworks_20260428.md` — wedge attribution showing rate is 44% lever
- `project_carve_evaluation_NOT_APPLICABLE_20260428` — sibling research evaluation
- `project_nitrobrew_evaluation_NOT_APPLICABLE_20260428` — sibling research evaluation
- `src/tac/self_compress.py:90-129` — already implements soft pruning
- `src/tac/learnable_bit_quant.py` — Lane Ω-V2 per-element learnable bits (Round 21 sign fix)
