# DAG FEED — Olmo Hybrid (arXiv 2604.03444) re-pointed to the costate-organ sequence arm

**UTC:** 2026-07-14T12:15Z · **Operator:** shared arXiv 2604.03444.
**Leg:** DAG (intake routing). NO equation/anchor (nothing MEASURED — a candidate reformulation, not a law).

## The paper
"Olmo Hybrid: From Theory to Practice and Back" (Merrill et al., AI2). Replaces transformer sliding-window
layers with **Gated DeltaNet** (gated linear-RNN) → a hybrid attention+recurrence 7B LM. Load-bearing
claim: hybrids express tasks **beyond both** attention and linear-RNNs (e.g. code execution / **state
tracking**), and this expressivity → **better pretraining scaling efficiency**.

## Re-pointing (full 12-object inventory, not nearest-salient)
- **Nearest-salient (WRONG):** "7B LM, not our domain → dismiss."
- **Best-fit object: #2 COSTATE ORGAN ROUTER**, specifically its **sequence arm** `C_gru_path` (lambda_net
  ARCHITECTURES) + the #344 Linear-NCDE trajectory model. The organ models the training trajectory as a
  temporal sequence of regime transitions (lane-erosion / mixed-Lane-Road / movable-island-unborn). That
  is a **state-tracking** problem — exactly the expressivity axis the paper says plain RNNs AND plain
  attention under-express. Gated DeltaNet is a candidate more-expressive replacement for the GRU path.
- Secondary touch: #12/#344 convergence-trajectory model (same sequence object).

## Recoverable signal (what transfers) + the NO-FAKE caveat (what does NOT)
- **Transfers (idea):** DeltaNet/hybrid as a **reformulation-queue** candidate for `C_gru_path`, motivated
  by the state-tracking expressivity result (regime-memory needs state, not just a soft-attention pool).
- **Does NOT transfer (number):** the 7B **scaling** result. Our organ has **n=1 / ~9-interval** data
  (#434 starvation). A 7B win is NOT evidence at our scale — ancestor rule + allergic-to-toys. A bigger
  sequence net likely OVERFITS the starved trajectory; the organ currently recommends `A_ridge_solve` (a
  linear solve) precisely because data is thin.
- **Verdict-scope:** candidate reformulation, INSTANCE-untested. NOT an adopt, NOT a family verdict.

## Routing action (gated, backtest-first — NO new subagent; $0 main-thread judgment)
1. Register `C_gru_path → gated_deltanet_hybrid` as a **reformulation-queue** entry on the organ's
   sequence arm in `lambda_net` ARCHITECTURES, **gated behind #434** (the synthetic-data starvation cure)
   — a more-expressive sequence net is only testable once the organ has enough trajectory data to fit it
   without overfitting.
2. **Backtest-before-adopt** (organ discipline): any new arch competes on held-out trajectory intervals
   vs the incumbent `A_ridge_solve` / `C_gru_path`, MEASURED, before it is recommended.
3. Broadcast to the live fleet so #434 (organ-data) sees the expressivity motivation as a design input.

**Pointer:** 0.19108 / 0.18804 UNMOVED. MEANS (apparatus / intake routing).
