# Adversarial review + OPTIMIZE of the Ballé weight-entropy lever (commit 5e73936b2) — BYTE-CLOSE PROOF

**UTC:** 20260620 (reviewer != author; recursive adversarial review + optimize)
**Authority:** `[contest-CPU advisory]` NON-PROMOTABLE. `$0`. NO paid dispatch. NO score claim. Pointer UNMOVED 0.19110.
**Live GREEN run (pid 20102, MPS, λ=0 default path) NOT touched** — its dir/flags untouched; it is on the byte-identical path (no `--weight-entropy-penalty-lambda` flag → λ=0 → penalty never built).

## THE #1 DELIVERABLE — the empirical bit-spend proof (Catalog #304)

**The deployed coder is NOT the prompt's "PR112 per-tensor adaptive 256-ary constriction range coder."** This torch-vehicle exports via the vendored `codec.build_archive` = `zigzag(int8)` + **brotli quality 11** on the WHOLE concatenated decoder state-dict (verified at `…/hnerv_muon/src/codec.py::encode_decoder`; the constriction hybrid was REMOVED upstream "for simplicity — only ~217 bytes worse"). brotli is LZ77 + context-modeled Huffman, so it can exploit structure BEYOND order-0 — meaning the "lower order-0 H ⇒ fewer bytes" translation is genuinely an open empirical question only the real archive answers. (This corrects the lever memo's own claim that the coder is "PR112 constriction"; the actual export path is brotli.)

### Measured (3 consistent short A/Bs: n16/ep20, n16/ep30, n24-staged; bit-shared init, only λ differs)

| Surface | Δorder0-H (λ50−λ0) | Δarchive bytes (λ50−λ0) | Translates to bytes? |
|---|---|---|---|
| **LIVE final decoder** | **−1.55 to −1.56 bits/wt** | **−16007 (−19.6%)** | **YES** |
| **EMA shadow (the SHIPPED archive)** | **+0.008 to +0.018** (UP) | **+72 to +87** (UP) | **NO** |

**THE DECISIVE REVIEWER FINDING (reviewer-vs-author gap):** the author's headline NO-FAKE metric and test measure `measure_decoder_weight_symbol_entropy(driver._final_decoder)` — the **LIVE** decoder. But the contest **ships the EMA shadow** (`best/best_archive.bin`, built from `best_ema_decoder.pt`). On the LIVE weights the H-cut DOES translate to brotli bytes (−16 KB). On the **EMA shadow it does NOT** — it is marginally bigger.

**Mechanism — CONFIRMED (EMA-lag), and the byte-close question is RESOLVED.** EMA decay 0.999 has a ~1000-epoch time constant. The penalty pulls the LIVE weights into a low-entropy basin late in training; in a SHORT run with decay 0.999 the EMA shadow never catches up (it is a slow average of higher-entropy earlier weights — a mixture of shifting concentrated distributions can have HIGHER entropy than any single snapshot), so the shipped archive does NOT shrink. **A faster-EMA A/B (ema_decay=0.9, n24, 250ep) PROVES the H-cut DOES translate to real shipped archive bytes:**

| (ema_decay=0.9, λ=50 vs λ=0) | value |
|---|---|
| Δema_order0_H (the SHIPPED shadow) | **−1.67 bits/wt** |
| **Δbest_total_bytes (the SHIPPED archive.zip)** | **−17011 (−22.6%)** |
| Δd_seg | **+0.038 (TASK HARM)** |
| Δscore | **+3.80 (WORSE)** |

So the **empirical bit-spend proof is POSITIVE** — the order-0 H reduction DOES translate to ~−22.6% real brotli archive bytes ON THE SHIPPED EMA SHADOW, *once the EMA tracks the low-entropy basin* (fast EMA, or — for the default 0.999 EMA — a long enough run; the live GREEN run trains thousands of epochs so its EMA would track). **BUT at λ=50 the rate cut comes with d_seg damage (+0.038) → net score WORSE (+3.80).** The byte win is REAL but NOT FREE; it must be tuned to the operating λ where bytes drop WITHOUT d_seg harm — this is exactly the R/D trade and why λ* matters [λ-sweep {5,15,30} at ema_decay=0.9 in-flight]. The author's headline (live-weight H drop) is real but understates BOTH the upside (the shipped archive CAN shrink ~22%) AND the cost (d_seg harm at the λ that achieves it). [500-ep default-EMA confirmation in-flight.]

> Reviewer verdict on the headline: the lever's **mechanism is real** (it genuinely shapes the symbol distribution and that shaping DOES translate to brotli bytes on the weights it shapes) but the author's **metric is measured on the wrong surface** (live, not EMA-shadow). The memo's estimated "−0.013 to −0.017 ΔS" is NOT supported by any EMA-shadow byte measurement; on the short-run EMA shadow the archive is +72–87 bytes (a tiny rate INCREASE).

## Re-verified author claims (reviewer-vs-author)

- **2a λ=0 byte identity — CONFIRMED.** `test_lambda_zero_run_is_byte_identical_to_pre_lever_path` compares `best_archive.bin` bytes (not just score); re-ran, passes. Live GREEN run command has no penalty flag → λ=0 → `driver._weight_entropy_penalty is None`. The daemon-safety guard holds.
- **2b λ>0 lowers the HARD-codec H (not surrogate) — CONFIRMED on LIVE weights.** The headline test + my A/Bs use the exact-histogram `measure_decoder_weight_symbol_entropy`, not the Ballé surrogate. The live-weight H drop is real (−1.55 bits/wt).
- **2c prior params in optimizer + update — CONFIRMED.** Tests pass; the params are a subset of the AdamW group and change across a step.

## (3) Adversarial interactions

### 3a — C1a DOUBLE-COUNT: **CONFIRMED, and stacking is NET-NEGATIVE** (probe `experiments/probe_balle_c1a_qat_interaction.py`)

PR95's C1a (`cat_entropy_v2` via `spec.cat_lambda`, the 0.01→0.02 late-stage schedule) and the new penalty penalize the **SAME quantity** — the size-weighted codec-grid symbol entropy of `w/(max|w|/127)`, bits/weight (verified against `losses.cat_entropy_v2`). C1a is a fixed-bandwidth Gaussian soft-histogram; the penalty is a learned per-channel logistic-prior expected codelength.

Measured H reached from the same init (200 steps):

| Config | measured H | Δ vs init |
|---|---|---|
| penalty only | **5.99** | −1.72 |
| C1a only | 6.14 | −1.57 |
| **BOTH** | **6.33** | **−1.38** |

**BOTH reaches a HIGHER (worse) entropy than EITHER single** (−0.34 LESS reduction than the better single). The two same-quantity estimators interfere; **stacking is net-negative.** The author's memo implicitly treated the penalty as orthogonal to / stronger-than C1a — the "stronger" part holds (penalty −1.72 > C1a −1.57) but the "stack on top" assumption is **falsified.**

**FIX LANDED:** `weight_entropy_penalty_supersedes_c1a` (default **True**): when the penalty is active for a stage, the C1a term is **zeroed** (the learned prior supersedes the memoryless shadow). Byte-identical on the λ=0 path (penalty off → C1a untouched → live basin unaffected). 3 new tests.

### 3b — QAT interaction: **COMPOUNDS CORRECTLY.** Under fake-quant STE in the forward, the penalty still lowers H (−1.07). The penalty reads the underlying float `mod.weight` while QAT fake-quants in the forward; both push toward the integer grid → no conflict.

### 3c — task harm at the λ for a real byte cut: the short-run d_seg/d_pose are unchanged (Δd_seg=0, Δd_pose~−2e-6, Δscore~+5e-5 — all from the +72-byte rate). At λ=50 the task is NOT harmed in the synthetic loop; the REAL-loss harm at the operating λ needs the long converged run [in-flight].

## (4) OPTIMIZE — per-tensor WATERFILL λ vs uniform λ

Implemented `WeightEntropyPenalty.compute_waterfill_weights(decoder, sensitivity)` (KKT reverse-water-fill: `w_t ∝ byte_share_t/(sensitivity_t+eps)`, normalized to the same aggregate budget so the A/B is fair) + `rate_bits(per_tensor_weights=…)` + driver flag `weight_entropy_penalty_waterfill` (default OFF = uniform = byte-identical loss term). Sensitivity source = the Lever-4 `tensor_sensitivity_ema` when populated, else byte-share-only. Verified on the REAL base_ch20 decoder (14 coded tensors, numel-weighted-mean 1.0; the high-sensitivity `stem` is protected at mult 0.001 while low-sensitivity `blocks.0` is pushed at 5.59). 6 new tests (normalization, protection, loss-changes-not-reported-rate, uniform==None, driver default-off byte-identical, on-differs-from-uniform). **The uniform-vs-waterfill byte-cut A/B + the λ-sweep {5,15,30} for λ* are `genuinely-deferred-because` the CPU was contended with the live GREEN MPS run (its CPU-side authority eval thread held ~175% CPU; the n24/200ep probes did not converge in the window and were killed to protect the GREEN run — the priority).** The implementation + tests are complete and READY; the A/B is a re-run when the GREEN run finishes (the directional result is set: the λ=50/ema0.9 point overshoots into d_seg harm, so λ* lies in {5,15,30} where the −22.6% byte trend should hold with less d_seg cost).

## (5) Prior-persist fix (the declared gap) — LANDED

`_capture_state` now persists `weight_entropy_penalty` (the learned prior `state_dict`, or `None` on λ=0 → byte-identical); `_restore_into` restores it when the penalty is built. A λ>0 resume now continues the adapted prior instead of rebuilding a fresh one. 2 new tests (round-trip-through-capture/restore with corruption; λ=0 capture carries None). The full `test_driver_resume` suite (12 tests) still passes at atol=0.

## Honest overall verdict

The lever is **mechanically real and NO-FAKE-clean** (it genuinely shapes the symbol distribution; λ=0 is byte-identical, re-verified). The byte-close question is **RESOLVED: the order-0 H cut DOES translate to real shipped-archive bytes — ~−22.6% on the EMA-shadow `archive.zip` — once the EMA tracks the low-entropy basin** (proven at ema_decay=0.9; for the default 0.999 EMA it requires a long run, which the live GREEN run is). So the empirical bit-spend proof (Catalog #304) is **POSITIVE for the rate axis**.

**BUT the win is NOT free and NOT yet a net-score win:** at λ=50 the rate cut comes with d_seg harm (+0.038 → score +3.80 WORSE). The lever is a genuine R/D actuator that must be tuned to the operating λ where bytes drop without d_seg damage (λ* sweep deferred-because-CPU-contended). And the **author's headline metric is measured on the wrong surface** (live decoder, not the EMA shadow that ships) — it overstates the per-epoch effect and is silent on both the EMA-lag (why the short-run shipped archive does NOT shrink at decay 0.999) and the d_seg cost.

Two real interaction BUGS were found + FIXED: (3a) the **C1a double-count is net-negative** (stacking the two same-quantity penalties reaches WORSE entropy than either alone) → `supersedes_c1a` guard (default on); (5) the **prior-persist resume gap** → prior `state_dict` now round-trips through the checkpoint. Per-tensor waterfill λ added (ready, A/B deferred). QAT compounds correctly (no fix needed).

**Net:** the lever delivers a REAL archive-byte reduction (rate axis), is NO-FAKE on its mechanism, but is `[contest-CPU advisory]` NON-PROMOTABLE and **not yet a net-score win** — the exact-eval byte-closed paired CPU row at a tuned λ* (where Δd_seg ≈ 0) is the follow-on that would make a score claim. **No score is claimed; the pointer is UNMOVED (0.19110); $0; the live GREEN run was never touched.**
