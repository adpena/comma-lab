# Compute-heavy weight-TIED decode does NOT break the Cool-Chic 0.014 exact-d_seg wall — the wall is COMPUTE-INVARIANT (2026-06-11)

**Authority:** `[macOS-CPU advisory]` / `[research-signal]`, **NON-PROMOTABLE**, $0, **no paid dispatch, no
MPS**. Exact d_seg = torch-CPU argmax-disagreement through the frozen `upstream/modules.py` SegNet (live
AND EMA, agreement confirmed). The bar is the LOCAL d_seg wall, NOT a contest score. Frontier UNMOVED
**0.19109982 [contest-CPU]** — this turn moved no exact row; it is an MVP-first de-risk that REFUTES a
candidate paid path and SAVES the spend. Artifact:
`experiments/results/lane_compute_heavy_weight_tied_decoder_20260611/k_sweep_n8_35ep.{json,log}`; driver
`experiments/run_compute_heavy_weight_tied_decoder.py`.

## The operator insight under test (2026-06-11)

> "The compress time could TRAIN the inflate time as well as do a lot of other things."

This reframes the smaller-basis REFUTAL (`capacity_verdict_smaller_basis_by_rate_REFUTED_pivot_to_waterfiller_20260611.md`),
whose own named residual was: *"a qualitatively different smaller synth might break the wall."* The decisive
$0 question: a **compute-HEAVY, weight-EFFICIENT** inflate-time decoder gets its effective capacity from
inflate-time COMPUTE (shared weights × many passes), not stored params — so it could reach LOWER exact
d_seg than the cheap single-pass synth at the SAME-or-fewer stored bytes. The 30-min inflate budget is
95.5% unspent, so heavy deterministic decode is affordable.

## The variant (REUSE, not rebuild)

`WeightTiedRecurrentCarrier` subclasses the existing `CoolChicPairCarrier` (REUSED: latent grids + ARM
rate + frame1 delta + honest charged-byte accounting + the `reconstruct_pair` trainer contract). The
**only** override is the synthesis: the single-pass `1×1 conv → GELU → 1×1 conv` is replaced by a
**weight-TIED recurrent refinement**:

```
h = in_proj(feat)                     # 1×1, stored ONCE
for _ in range(K):                    # K shared passes = COMPUTE, not bytes
    h = h + block(h)                  # block (3×3 conv → GELU → 1×1 conv) stored ONCE; residual fixed-point
rgb = sigmoid(out_proj(h))            # 1×1, stored ONCE
```

Trained through the proven `ScoreAwareTrainer` (PR95 live-SegNet loss + EMA-warmup B4-fix + eval roundtrip
+ exact d_seg). Carrier held FIXED at the L2-wall grid config (6 grids @ 40×56, cpg=4, synth_hidden=16);
**only K (the number of inflate-time refinement passes) is swept.** n8, 35 epochs, hinge curriculum,
out_hw 96×128 — apples-to-apples with the 0.0140 wall measurement.

NO-FAKE: gradient flows to the tied block (grad ≈ 1.9e4–3.1e4 at step 0) AND the latent grids;
stored bytes counted honestly (the tied block is stored ONCE → K-INDEPENDENT, which the data confirms);
decode is inflate-LEGAL (deterministic, GT-free, scorer-free, self-contained; projected ≤30 min).

## The decisive measurement (the d_seg-vs-K curve)

| K (inflate passes) | exact d_seg (EMA) | EMA/live agree | stored weight B | total B | inflate~600pair | legal |
|---|---|---|---|---|---|---|
| 1 | 0.0186 | 4.8e-5 | 6474 | 8603 | ~2 s | ✓ |
| 2 | 0.0206 | 3.9e-4 | 6474 | 8598 | ~2 s | ✓ |
| **4** | **0.0158** (best) | 1.7e-4 | 6474 | 8594 | ~5 s | ✓ |
| 8 | 0.0539 | 1.6e-3 | 6474 | 8588 | ~13 s | ✓ |
| 16 | 0.4786 | 1.6e-3 | 6474 | 8588 | ~10 s | ✓ |

Reference: Cool-Chic wall **0.0140** (L2 arm, stored weight ~1290 B) · basin **5.6e-4** · corrected bar
0.0011–0.0017.

## The three decisive answers

1. **Does compute-heavy weight-tied decode break BELOW the 0.014 wall toward the 5.6e-4 basin at
   matched/fewer stored bytes? NO.** Best (K=4) = **0.0158**, ~11× ABOVE the bar and statistically TIED
   with the 0.0140 Cool-Chic wall — not a step toward the basin. And it costs MORE stored bytes (6474 B
   weight vs the wall arm's ~1290 B), so it is strictly DOMINATED by the cheap synth at the same d_seg.

2. **The d_seg-vs-K curve is a COMPUTE-INVARIANT wall, not capacity-from-compute.** d_seg does NOT fall
   monotonically as K rises. It hovers ~0.016–0.021 for low K, then **DEGRADES** at high K
   (K=8 → 0.054, K=16 → 0.479). The capacity-from-compute hypothesis predicts a monotone DESCENT with K;
   the data shows the opposite at the high-K end. The residual fixed-point refinement becomes
   optimization-unstable/divergent at K≥8 under this (n8/35ep) training budget — more shared passes did
   not buy more usable capacity; they made the loss landscape harder. Even granting that a longer/heavier
   training curriculum might tame the K=8/16 instability, the K=1→4 region (which IS well-conditioned,
   EMA/live agree ≤2e-4) shows NO descent toward the basin — it plateaus at the wall.

3. **Inflate stays cheap and legal across the whole sweep** (~2–13 s projected for 600 pairs vs the
   1800 s budget) — so the negative is NOT a budget artifact. Heavy compute WAS affordable; it simply did
   not convert into lower d_seg.

## The sharpened CRUX (why this hardens the capacity verdict)

The smaller-basis REFUTAL left ONE escape hatch: a qualitatively different synth getting capacity from
compute rather than stored params. This test closes it for the recurrent/unrolled-fixed-point class:
**inflate-time compute (shared weights × K passes) does NOT substitute for the missing representational
structure.** The d_seg wall at ~0.014–0.016 is invariant to how many times you apply a small shared
decoder. The capacity that the conv-HNeRV frontier (~178 KB, L20–L32 stack) uses to hold the 5.6e-4 basin
lives in DISTINCT stored weights with DISTINCT receptive structure — not in repeated application of a
shared compact block.

This is the operator's NO-FAKE rule working as intended: a $0 local measurement refuses to narrate
"capacity-from-compute" as a win when the exact d_seg says the wall is compute-invariant. The "compress
trains inflate" framing is TRUE as a mechanism (the trainer DID train the K-pass decoder end-to-end), but
the trained-richer-decoder does NOT reach lower d_seg here — the compute-richness did not become
representational capacity.

## Honest disposition + caveats (non-sycophantic)

- **Disposition: the wall is COMPUTE-INVARIANT → this HARDENS the capacity verdict.** It does NOT de-risk a
  paid compute-heavy-decoder retrain; if anything it argues AGAINST one for this architecture class. No
  operator paid-retrain flag is raised.
- **Scope of the negative (what is measured vs asserted):** this measures ONE compute-heavy class — a
  weight-tied residual recurrent refinement (the unrolled-fixed-point / deep-equilibrium-flavored
  candidate) at the n8/35ep budget. It does NOT exhaustively test the implicit coordinate-MLP (NeRF-style,
  few weights / many dense queries) candidate, which is a structurally different "capacity from compute"
  bet (per-pixel function evaluation rather than shared-block refinement). That remains a (lower-prior,
  burden-of-proof-now-high) open residual. But the recurrent/unrolled/DEQ family — the most natural
  reading of "shared weights × many passes" — is measured-negative.
- Per CLAUDE.md "Forbidden premature KILL": this is a measured negative on a specific implementation class,
  not a paradigm kill. Reactivation criteria: (a) a coordinate-MLP-implicit variant showing d_seg descent
  with query density at matched stored bytes; (b) the recurrent variant re-run with a longer/annealed
  curriculum that tames the K≥8 instability AND still shows descent (not just stability) toward the basin.

## The pivot (one crisp verdict, then move — per THE GOAL)

The compute-heavy weight-tied decode path walls. The live highest-EV path to a lower EXACT score is
UNCHANGED from the smaller-basis verdict: the **evaluator-action waterfiller / null-space composition on
the CURRENT frontier** — it operates on the archive that ALREADY holds the d_seg basin (it does NOT need
the capacity wall broken), shaving the exact score via argmax-equivalent / invisible / commutator-aware
atom edits (the path that produced the last real win, PR110 −0.000883). This turn refuted a path, saved a
candidate fake paid spend, and re-confirms the next unit should aim at the waterfiller, not at more
capacity-knob sweeps.

## 6-hook wire-in (Catalog #125)

1. sensitivity-map: N/A (a capacity diagnostic, not a per-byte sensitivity producer).
2. Pareto constraint: ACTIVE — adds the measured point "stored-bytes ↔ inflate-compute ↔ d_seg" to the
   capacity frontier: compute does NOT move the d_seg constraint inward for the recurrent class.
3. bit-allocator: N/A (no per-tensor importance change).
4. cathedral autopilot dispatch: N/A (non-promotable advisory; no archive-deployable artifact).
5. continual-learning posterior: the verdict (compute-invariant wall) is the reusable signal — folds into
   the capacity-verdict cluster as the closure of the "qualitatively different synth" residual.
6. probe-disambiguator: this IS the disambiguator between "capacity-from-compute" (d_seg falls with K) and
   "compute-invariant wall" (d_seg flat/worse with K) — verdict: compute-invariant.
