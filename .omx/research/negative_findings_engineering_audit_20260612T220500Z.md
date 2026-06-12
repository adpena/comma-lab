# Negative-findings engineering audit — TRUE vs UNDER-POWERED (2026-06-12)

**Operator directive (binding):** *"review the engineering of all resulting in those negative
findings, audit and adversarial review"* + *"sometimes training time is necessary or capacity and
config for true signal"* + *"seems like it was so close — more nuanced and math-optimal may
actually be net positive."*

**Method.** For each negative I audited the ACTUAL code + measurement (not the memo), and
adversarially challenged: (1) real scorer / real inputs (not synthetic/proxy)? (2) apples-to-apples
(distortion fixed, only the lever varied)? (3) correct implementation (a bug can fake a null)?
(4) search/budget exhausted, or UNDER-POWERED (would training time / capacity / config flip it)?
(5) PARADIGM negative or IMPLEMENTATION negative (Catalog #307)? A negative is only trustworthy if
the engineering that produced it was correct AND adequately powered.

## VERDICT TABLE

| Finding | Engineering | Powered? | Verdict | Disposition |
|---|---|---|---|---|
| **D1 cross-pair latent dedup = 0 B** | CERTIFIED SOUND | exact (lossless coding) | **TRUE for these latents** | DEFER-with-auto-fire (basis-conditional) |
| **Lever-4 QAT byte-direction-only** | CERTIFIED SOUND (mechanism structural) | net magnitude under-powered | **TRUE mechanism** | superseded by variable codec |
| **Variable codec net +worse** | byte win CERTIFIED; net via CRUDE allocation | **UNDER-POWERED** (uniform 27/28 coarsen; no variable-grid training) | **RE-OPENED** | math-optimal waterfill (Partner B, $0) ∥ variable-grid QAT |
| **Cool-Chic AR-prior saturated** | CERTIFIED SOUND | **properly powered** (800 ep, 130× the smoke) | **TRUE negative** | latent-AR rate is saturated; not the lever |
| **Cool-Chic d_seg/pose wall** | sound, but | **UNDER-POWERED** ("monotonic, still descending") | **RE-OPENABLE** | more epochs / full-res / capacity (Track-B basis question) |
| **Witness seg-boundary sidecar NO-GO** | CERTIFIED SOUND | structural (flip-count) | **TRUE structural** | fold boundary correction INTO training, not a stored sidecar |

## DETAIL

### 1. D1 cross-pair latent dedup — CERTIFIED REAL NEGATIVE (apples-to-apples, search exhausted)
- **Quant parity:** `test_quant_matches_vendored_encode_decode_when_available` asserts
  `torch.equal(vend_recon, my_recon)` — D1's quantization reconstructs BIT-IDENTICAL latents to the
  REAL vendored codec (`import_vendored("codec")`). So the byte comparison varies ONLY the coding,
  never the distortion — the strongest possible apples-to-apples in a rate experiment.
- **Baseline real:** `vendored_B` is `brotli.compress(vend.encode_latents(lat), q=11)` on the REAL
  basin/arm/mps latents — the production encoder, not a reimpl.
- **Parse-back real:** full archive 89570→89570 byte-identical, latents bit-exact through the
  unified decoder.
- **NO-FAKE control:** −616 B real win on structured latents surviving parse-back (the apparatus
  isn't a no-op; `test_a_no_op_codec_would_fail_the_savings_assertion`).
- **Search:** exact-dedup (0 dups), VQ×3 (+1584 B), 2nd-order/motion (+1747 B), static range (+113 B),
  per-dim range (+4075 B). Only untried: adaptive context coder (~205 B ≈ −0.00014 ΔS, correctly
  deemed not worth it). Vendored is ~1.3% above the 0th-order symbol-entropy floor; 2nd-order +
  per-dim probes confirmed no conditional structure to exploit.
- **CONDITIONALITY (operator's lens):** the latents are a TRAINING OUTPUT; a different basis/capacity
  could yield cross-pair structure the (retained, auto-firing) apparatus would exploit. TRUE for
  THESE latents, not absolute. Correct DEFER, not KILL.

### 2. Lever-4 score-aware QAT — CERTIFIED REAL MECHANISM (structural, not budget-dependent)
- **A/B real:** `_train_arm` trains TWO arms from the SAME basin-EMA seed, same real-0.mkv slice,
  same RNG, both FULLY TRAINED, differing ONLY in `score_aware` (uniform-127 vs sensitivity grid),
  computing the real contest score. Honest caveat disclosed (12 pairs / 8 ep, advisory).
- **Mechanism is structural:** the vendored codec ALWAYS re-quantizes at 127, so a train-ONLY coarse
  grid is ERASED at deploy → the −3263 B snap washes to −7 B. This does NOT depend on the small
  budget — it's an architectural certainty. The fix is to make the variable grid the DEPLOYED grid
  (item 3).

### 3. Variable-level codec — byte win REAL, net-score UNDER-POWERED → RE-OPENED (the operator's catch)
- **Byte win certified:** `probe_variable_level_codec_byte_distortion.py` byte-closes 3 ways
  (vendored-127 / variable@score-aware / variable@uniform byte-identity guard), and the variable grid
  IS the deployed grid → byte −789..−1721 B SURVIVES inflate (unlike Lever-4).
- **Net measured +worse, but CRUDE:** the probe coarsened **27/28 tensors at a uniform
  `min_level_ratio`** → net advisory S Δ +0.001053 (ratio 0.5) / +0.006030 (ratio 0.75). The MORE
  aggressive arm had the SMALLER penalty — a tell that the uniform allocation is MIS-SHAPED (it
  coarsens pose-sensitive tensors it shouldn't).
- **TWO untried, under-powered axes (this is NOT a paradigm negative):**
  - **(a) math-optimal reverse-waterfill** — per-tensor n_levels set by the KKT condition
    (equalize dΔS/dByte using measured ‖∂S/∂w‖), pose-sensitive tensors held at 127, budget spent
    where distortion is cheapest. $0, no retrain. **Could flip net-positive** (operator's "so close /
    math-optimal may be net positive"). → **Partner B (a1e690f6) running.**
  - **(b) train the decoder at the variable grid** — QAT with the variable grid as the deployed grid
    (eval_roundtrip-style), so the decoder learns coarse-grid robustness and RECOVERS the distortion.
    Folds into the curriculum's QAT stages — the operator's "training time / config for true signal."
- **LEDGER CORRECTION:** I earlier framed ITEM B as a ready "−0.0005..−0.0011 win" — that was an
  OVERSTATEMENT (NO-FAKE). Net is +worse until (a) or (b). Corrected in the ledger + the
  `all-wins-substantial` memory. Do NOT count it in the stack yet.

### 4. Cool-Chic AR-prior — CERTIFIED REAL NEGATIVE (PROPERLY POWERED — the discipline worked)
- Step-1's "saturates at 6 epochs" was a HYPOTHESIS; the team TESTED it at **800 epochs (130×)** on
  the fixed latents (`cool_chic_ar_prior_training_feasibility_20260612T164116Z.md`):
  `disc_bits/elem 6.7192 → 6.6570 = 0.93% drop, non-monotone (±1% optimization noise on a SATURATED
  objective)`. This is exactly the operator's "test whether training time gives true signal" — and it
  was DONE. The AR prior is genuinely saturated; the latent-AR rate is not the lever. CLEAN.

### 5. Cool-Chic d_seg/pose wall — UNDER-POWERED (flagged honestly; re-openable)
- `cool_chic_param_at_dseg_basin_design_20260611.md`: the seg/pose are "UNDERTRAINED at the
  CPU-tractable epoch budget (NOT a basis limit)"; the d_seg curve is "monotonic and still
  descending — convergence (more epochs)." This is an explicit UNDER-POWERED flag. A separate
  "BASIS-SPECIFIC wall CONFIRMED" applies to the conv-HNeRV param↔d_seg relationship. Re-openable
  with more epochs / full-res / capacity — a Track-B basis question, not a closed door.

### 6. Witness seg-boundary sidecar — CERTIFIED REAL STRUCTURAL NEGATIVE
- Decisive reason is the absolute FLIP COUNT (~884 boundary-flips/pair → ~530K over 600 pairs) +
  46.4% round-trip survival (< 50% bar); "tightening τ does NOT help"; "flip-count economics dominate
  at every base operating point." Info-theoretic, not under-training. Verdict HYBRID: sidecar NO-GO,
  **fold the boundary correction INTO training** (the correct disposition — it becomes a d_seg
  training signal, which the distortion arm's seg-surrogate Lever 2 + margin-weight Lever 5 already do).

## SYNTHESIS (what the audit changes)
1. **The engineering behind the negatives is largely SOUND** — apples-to-apples (D1's `torch.equal`),
   real-scorer A/Bs (Lever-4), and the under-training hypothesis WAS tested where it mattered
   (Cool-Chic AR 800-ep). The team's NO-FAKE discipline held.
2. **The one real overstatement was MINE** — the ledger/memory framed the variable codec as a ready
   win; it is byte-real but net-+worse under the CRUDE allocation. CORRECTED.
3. **Two negatives are genuinely RE-OPENED under the operator's "training/config for true signal"
   lens:** the variable codec (→ math-optimal waterfill $0 ∥ variable-grid QAT) and the Cool-Chic
   d_seg/pose wall (→ more epochs / full-res / capacity, Track-B).
4. **Strategic:** latent rate is exhausted (D1) and latent-AR is saturated (Cool-Chic). The remaining
   RATE headroom is **decoder weights** via the math-optimal waterfill (if net-positive) or
   variable-grid training. The **binding lever remains the distortion arm** (d_seg/d_pose); the
   witness finding folds boundary correction INTO that training (Levers 2+5), where it belongs.

**Authority:** every number here is `[macOS-CPU advisory] / [contest-CPU advisory]` NON-PROMOTABLE
until `upstream/evaluate.py` on a byte-closed archive.
