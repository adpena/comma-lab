# n600 Modal cost math + scenario range (2026-06-11)

**Authority:** cost analysis, advisory. Modal pricing verified 2026-06 ([modal.com/pricing]; per-second).
Frontier UNMOVED 0.19109982. Trigger: operator "review modal cost math and provide range of scenarios."
NOTE: the MLX capstone is Apple-only → Modal runs the vendored PR95 TORCH trainer (P2). P1 (local MLX) = $0.

## Verified Modal GPU pricing (per-second → /hr)

| GPU | $/sec | $/hr |
|---|---|---|
| T4 | 0.000164 | 0.59 |
| L4 | 0.000222 | 0.80 |
| A10G | 0.000306 | 1.10 |
| A100-40GB | 0.000583 | 2.10 |
| A100-80GB | 0.000694 | 2.50 |
| H100 | 0.001097 | 3.95 |

New accounts: $30/mo free credits. Per-millisecond billing, no idle charge.

## The cost formula + the TWO uncertain inputs

`cost = epochs × (sec/epoch) × ($/sec)`. Both inputs are uncertain:
1. **Total epochs** (the SCIENCE): PR95 reference = 29,650; compressed = ~10k; minimal = ~3k (the fixed
   recipe descended 0.51→0.012 in 3 stages at n8 — convergence may be fast). Whether a COMPRESSED budget
   reaches the 5.6e-4 basin is the open scientific risk.
2. **Step-time** (the DOMINANT cost uncertainty, UNMEASURED): anchor = torch-CPU ~534 s/ep @ n600. GPU
   speedup for the memory-bound EfficientNet-B2 scorer (98% of the step) is unmeasured; nominal estimates
   T4 ~30 / A10G ~13 / A100-40GB ~7 s/ep, ±2× band. The decoder size (base_ch=20 vs 36) barely changes this
   — the FIXED scorer dominates.

## GPU-invariance

For a memory-bound workload, a faster GPU buys fewer hours at proportionally higher $/hr → total cost is
roughly INVARIANT across GPU classes for a given epoch budget (within ~20% at nominal step-times). ⟹ the GPU
choice is a WALL-CLOCK decision (pick the one you can saturate; A10G is the likely sweet spot for moderate
batch — good bandwidth, well-utilized, cheaper; A100 for shortest wall-clock; T4 if budget-pinched), NOT a
cost decision.

## Scenario table (training cost, nominal step-time)

| Epoch budget | T4 ($0.59) | A10G ($1.10) | A100-40GB ($2.10) |
|---|---|---|---|
| 3,000 (minimal/fast-converge) | ~$15 | ~$12 | ~$12 |
| 10,000 (compressed) | ~$49 | ~$40 | ~$41 |
| 29,650 (full PR95) | ~$146 | ~$118 | ~$121 |

Fast/slow band: ×0.5 at fast step-time, ×~2 at slow. → 10k epochs = **$20–100**; full PR95 = **$60–290**.
Add-on: paired CPU+CUDA exact evals ≈ **$2–5** (one-shot CUDA inflate+eval + 60–120 min CPU eval on a cheap
instance, per the dual-axis submission discipline).

## What the $100 buys (verdict)

- **Comfortable**: 3k–10k epochs at nominal step-time (~$12–49) — and per-arm, so $100 funds the de-risk +
  one real run, or MULTIPLE arms (base_ch=20 / 24 / untied).
- **Borderline-to-over**: the full 29,650-epoch PR95 reference (~$120–150 nominal, up to $290 slow) exceeds
  $100 — the epoch-budget-vs-basin risk in dollars: $100 ≈ a COMPRESSED budget.
- **$100 is NOT the binding constraint for a compressed run** — the science (basin in a compressed budget?)
  and the unmeasured step-time are.

## Recommendation

The single highest-leverage move is a **~$0.30 10-min step-time SMOKE** on the target GPU (the vendored PR95
torch trainer at base_ch=20, n600): it collapses the dominant step-time uncertainty, converts "$100 ≈ ??
epochs" into a hard number, AND yields an early descent-rate read (does the fixed recipe descend fast → fewer
epochs needed). Gate #9 in the readiness tracker — run it WITH the symposium, before the real spend. Then the
epoch budget (one big run vs several arms) is a measured decision, not a guess.

[modal.com/pricing]: https://modal.com/pricing
