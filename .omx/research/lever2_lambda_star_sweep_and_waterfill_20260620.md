# Lever 2 — Ballé weight-entropy λ* sweep + waterfill-vs-uniform A/B (the deferred completion)

**UTC:** 20260620
**Authority:** `[contest-CPU advisory]` NON-PROMOTABLE. `$0`. NO paid dispatch. NO score claim. Pointer UNMOVED 0.19110.
**Live GREEN run (pid 20102, MPS, λ=0 default path) NOT touched** — its dir/flags untouched; on the byte-identical path (no `--weight-entropy-penalty-lambda` flag → λ=0 → penalty never built). All probes ran thread-capped (`OMP_NUM_THREADS=2`) and SEQUENTIALLY (one arm at a time) behind the GREEN run's CPU-authority eval thread, per the contention discipline.

## What this completes

The adversarial-review memo (`feedback_torch_vehicle_balle_adversarial_review_byteclose_20260620.md`, commit `fb4ed78d3`) RESOLVED the byte-close question (the order-0 H cut DOES translate to ~−22.6% shipped-archive bytes on the EMA shadow at ema_decay=0.9) but DEFERRED the λ* sweep + waterfill A/B `genuinely-deferred-because` the CPU was contended by the GREEN run. The directional anchor it set: **λ=50 / ema0.9 → −22.6% bytes BUT +0.038 d_seg → +3.80 score (WORSE)**. The open question: is there a smaller λ where bytes drop WITHOUT material d_seg harm?

## Harness

`experiments/probe_balle_byte_close_ab.py` — bit-shared init (same seed, same arch), differs ONLY by λ; builds the REAL vendored `codec.build_archive` (zigzag-int8 + brotli q11) archive and reads d_seg/d_pose from `best_meta.json`. Fast EMA (ema_decay=0.9) so the EMA shadow — the SHIPPED surface — tracks the low-entropy basin (the adversarial-review memo proved the default 0.999 EMA hides the effect on short runs). Synthetic scorer (RESEARCH-ONLY, no score claim): the d_seg values are synthetic, but the **byte-translation** and the **sign of the R/D trade** are what the probe measures, and they match the λ=50 anchor directionally.

Config: `--n-pairs 8 --epochs 80 --lr 3e-3 --ema-decay 0.9`. Tiny n + short epochs to stay light under GREEN contention.

## RESULT — uniform-λ sweep {0, 5, 15, 30} (the SHIPPED best/EMA-shadow archive)

| λ | best_total bytes | Δbytes | Δ% | d_seg | Δd_seg | Δd_pose | score | **Δscore** |
|---|---|---|---|---|---|---|---|---|
| 0 (control) | 79761 | +0 | +0.0% | 0.7689 | +0.0000 | +0.00000 | 78.687 | +0.000 |
| **5** | 69357 | **−10404** | **−13.0%** | 0.7979 | **+0.0289** | −0.00009 | 81.573 | **+2.886** |
| 15 | 79518 | −243 | −0.3% | 0.7994 | +0.0304 | +0.00088 | 81.732 | +3.045 |
| 30 | 79704 | −57 | −0.1% | 0.7994 | +0.0305 | +0.00152 | 81.737 | +3.050 |

### λ* VERDICT — **NO λ in {5, 15, 30} is net-positive.** Every arm raises the score (WORSE).

Two structural facts, both honest negatives:

1. **The d_seg cost is roughly CONSTANT (~+0.029 to +0.030) across all λ** — even the smallest λ=5 already pays nearly the full d_seg penalty. There is no low-λ regime where the penalty shapes weights "for free": the moment the term is active enough to matter, it has already perturbed the d_seg-critical weights.
2. **The byte win SHRINKS as λ grows** (λ=5: −10404; λ=15: −243; λ=30: −57). λ=5 is the BEST byte win AND the smallest d_seg cost — the exact opposite of a usable R/D knob (a usable knob would trade more bytes for more d_seg; here larger λ gives LESS byte win for the SAME d_seg cost — the larger λ over-concentrates and the brotli coder stops benefiting). So even the "best" point (λ=5) is net **+2.89 WORSE**.

This is consistent with the λ=50 anchor (+0.038 d_seg, +3.80 score). The conclusion holds across the whole swept range: **on this vehicle's vendored brotli codec, the weight-entropy penalty is a net-negative R/D actuator at every λ — the d_seg harm dominates the byte win.** (The synthetic d_seg is coarse; the real-loss magnitude would differ, but the SIGN — d_seg-cost-dominates — is the robust finding and matches the independent λ=50 anchor.)

## RESULT — waterfill-vs-uniform A/B at λ=5 (the best uniform point)

Same harness, same bit-shared λ=0 control (both arms' control = **79761 bytes**, confirming the A/B is fair). Waterfill = KKT reverse-water-fill `w_t ∝ byte_share_t/(sensitivity_t+eps)` (the Lever-4 `tensor_sensitivity_ema` protects d_seg-critical tensors; normalized to the same aggregate budget).

| arm (λ=5) | Δbytes | Δ% | Δd_seg | **Δscore** |
|---|---|---|---|---|
| **uniform** | −10404 | −13.0% | +0.0289 | **+2.886** (WORSE) |
| **waterfill** | −8444 | −10.6% | +0.0278 | **+2.772** (WORSE) |

### WATERFILL VERDICT — **waterfill is marginally better (Δscore +2.772 vs +2.886, by 0.114) but does NOT rescue the lever to net-positive.**

Waterfill works *directionally* — by protecting the high-sensitivity tensors and pushing rate onto big-byte / low-sensitivity tensors, it cuts the d_seg harm slightly (+0.0278 vs +0.0289, ~4% less) — but it also wins fewer bytes (−10.6% vs −13.0%) because it backs off the most-impactful (and most-sensitive) tensors. The two effects nearly cancel: waterfill's net ΔS is 0.114 better than uniform's, but **both arms are firmly net-negative (WORSE)**. Waterfill is the correct allocation *shape* but cannot overcome the fundamental problem — the weight-entropy penalty's d_seg harm dominates its byte win at every λ, with or without waterfill.

## Honest overall verdict

The Ballé weight-entropy lever is **mechanically real and NO-FAKE-clean** (λ=0 is byte-identical; the byte win is genuine on the shipped EMA archive). But the λ* sweep **closes it as a net-score win on this vehicle**: there is no λ in {5, 15, 30} (nor at the λ=50 anchor) where the shipped-archive byte reduction comes without a dominating d_seg cost. The lever is best left at its default (**λ=0, OFF**) — which is exactly where the live GREEN run is. The rate axis is better attacked by the byte-neutral d_seg-aware taper + waterfill-solved capacity reallocation (the small-basis rate-headroom finding), NOT by penalizing weight entropy at the cost of d_seg.

`research_only=true` for the lever as a score actuator; the flag remains a clean, tested, default-off R/D knob for any future operating point where the d_seg/byte trade flips (e.g. a vehicle whose codec is order-0-limited rather than brotli, where the H cut would translate more cheaply).

## 6-hook wire-in (Subagent coherence-by-default)

1. **Sensitivity-map** — ACTIVE: waterfill consumes the Lever-4 `tensor_sensitivity_ema` (∂S/∂w) to protect d_seg-critical tensors.
2. **Pareto constraint** — ACTIVE: the λ-sweep IS a measured R/D Pareto trace (Δbytes vs Δd_seg); the verdict (no net-positive point) is a Pareto-dominance result.
3. **Bit-allocator hook** — ACTIVE: the penalty + waterfill ARE a bit-allocator primitive; the verdict reroutes allocation away from weight-entropy toward the byte-neutral taper.
4. **Cathedral autopilot dispatch** — N/A: advisory, non-promotable; no archive-deployable dispatch (the lever stays OFF).
5. **Continual-learning posterior** — ACTIVE: this memo + the two JSON artifacts are the empirical anchors (λ-sweep + waterfill); they update the prior that the weight-entropy lever is net-negative on the brotli codec.
6. **Probe-disambiguator** — ACTIVE: `probe_balle_byte_close_ab.py` IS the disambiguator (uniform vs waterfill vs λ-sweep, all driven by the same harness).

## Artifacts

- `.omx/research/lever2_lambda_sweep_uniform_n8_20260620.json` (uniform sweep raw rows)
- `.omx/research/lever2_lambda5_waterfill_n8_20260620.json` (waterfill A/B raw rows)
- `experiments/probe_balle_byte_close_ab.py` (the harness; already committed `fb4ed78d3`)
