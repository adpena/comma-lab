# DAG FEED — ddm_dc1 (2026-08-01) — the correction-stream LABEL cost · QA03's censored solver · the §B.5 sign

Parent memo: `.omx/research/ddm_dc1_correction_label_cost_and_qa03_censoring_20260801.md`
Receipt: `.omx/research/ddm_dc1_label_price_n600_20260801.json`
Equation: `ddm_dc1_correction_stream_label_cost_v1` (append-only sister; pp1 law UNMUTATED)
Axis: `[macOS-CPU advisory]` NON-PROMOTABLE · `score_claim=false` · scorer-free arm ·
pointer 0.1910828242 UNMOVED · own-vehicle frontier v4d 0.9639878 UNMOVED.

- **FEED-dc1-a [MEASURED]** correction-stream **LABEL** cost = **0.082–0.255 B/flip** (best of two
  coders) over ρ 2.2e-4..2.2e-2 = **0.28–0.88×** the blind 5-ary bound (log2(5)/8 = 0.2902410).
  ba31's *"label largely predictable from neighbouring labels"* **REFUTED as stated**: the
  neighbour-conditioned #307 `cls` stream buys only **1.14×** at the band edge, and a **GENERIC**
  LZMA raster control **beats** it at **6 of 7** densities (every ρ ≥ 5.6e-4). The rh1 lesson —
  *derived is not a synonym for correct* — reproduced on a second, independent surface.
  Controls: position reproduces pp1's registered curve to 4 dp at τ ∈ {0.008,0.02,0.05,0.1,0.2}
  with matching support sizes (cross-instrument check now enforced in code); random-label negative
  control lands on the blind bound; bit-exact flip+label decode verified on a leading subset.

- **FEED-dc1-b [MEASURED]** the rational-correction band's **lower edge moves 2.56×** once label is
  paid: ρ_c **5.02e-4 → 1.285e-3**. Band ≈ **[1.3e-3, 1e-2]**. Carrier design spec relaxes from
  "natively ≤ 5e-4" to "≤ ~1.3e-3" (PR130's 3e-4 rail still clears). The position-only crossing
  recomputes to 5.0146e-4 vs the registered 5.02e-4 — the canary, 0.11% apart.

- **FEED-dc1-c [MEASURED / category]** **QA03's 1.4518 B/flip is not a correction-stream price.**
  `sb1_seg_batch.py:242` edits token VALUES in place; `price_bytes()` re-encodes **all four archive
  sections**, so 2,709 B is a whole-ARCHIVE re-encode delta — entropy inflation of an existing
  stream. The receipt's own `byte_delta_note` defers the true shipping price to the **r7 SMEVR**
  coder (xi1 handoff). The pp1 law's `object` is *"correction/support stream position coding"*, so
  QA03 is out of domain before the LABEL exclusion is even reached. Recorded in the new law's
  `excluded` list so the category error is refused structurally. **Corollary:** ba31 §B.2's
  "≥51.6% is label + solver overhead" mis-attributes a real residue — QA03 has neither stream.

- **FEED-dc1-d [MEASURED]** **QA03's solver was CENSORED, not converged.** `--max-quanta` default
  **4** outranked the (correct, present) convergence test on **51/120 = 42.5%** of instances; those
  produced **64.7%** of realized flips (1,207/1,866); the step histogram **spikes 2.68×** at the cap
  (51 vs 19) instead of decaying. Stops at k<4 were voluntary, at k=4 forced — so **1,866 is a
  strict LOWER BOUND** on this formulation's converged yield, and `sb1`'s *"the −0.046…−0.138 band
  is NOT reachable by this formulation class"* was drawn on a censored solve. FIXED: default 32,
  per-instance `stop_reason` via `for/else`, receipt-level `n_cap_saturated`/`cap_saturated_frac`
  (the CLASS fix — censoring can no longer be invisible), + 10 scorer-free tests including a guard
  that fails if the default returns to 4. **Re-measure = ~30–60 min on ONE scorer slot, resumable**
  (only the 51 censored instances re-run). Not taken — this arm is scorer-free.
  Second defect NAMED, unbuilt: `_best_single_quantum` is greedy unit-step coordinate descent with
  no line search along the accepted direction — the *"damped-Newton line search"* label overstates
  the method; extending an accepted direction costs 1 eval vs 8 for a fresh probe round (8× cheaper
  descent), but it is a method change and must be raced. Third: `top_k=120` of ~648 non-zero cells.

- **FEED-dc1-e [DERIVED]** ba31 §B.3 re-priced with the measured label term. ja1/v4c base
  (508,639 flips, seg 0.431179 S): position-only 0.6702 → −0.204199; **position+label 0.8832 →
  −0.132040**; position+blind-label 0.9604 → −0.105900; uniform-bound+blind-label 1.2724 →
  **−0.000231 (break-even to 4 dp)**. burn-4 ep854: −0.178074 / **−0.110828** / −0.088182 /
  +0.004783. **The sign SURVIVES paying the label cost** (−0.132 S = **16.67% of the 0.7918468
  gap-to-bar**) but the magnitude falls **55%** from ba31's headline. Of ba31's quoted 0.264 S
  swing, only 0.072 S is a genuine correction to the price; the remaining 0.192 S is the gap to an
  object that is not a correction stream. **STILL the idealised full-residual slope, NOT a realized
  move** — QA03 addressed 0.367% of the residual.

- **FEED-dc1-f [ADJUDICATED]** ba31 §B.5 **does** carry a sign defect — **gr1's is not it.** gr1
  prices bytes **SAVED** per flip **INTRODUCED** (*"Water break-even = 1.273 bytes saved / flip
  introduced"*): below water = **LOSE**. of1/W1-COH prices bytes **SPENT** per flip **FIXED**
  (*"B/err = phase_bytes / flicker_flips_fixed"*): below water = **WIN**. gr1's DOMINATED verdict
  is **SOUND** and rests on its own realized JOINT axis (every candidate worse than the current
  point; strictly dominated by cell-granular drop at every byte budget), not on the water
  comparison. ba31 places both — plus QA03's third currency — in one `vs water 1.2731` column where
  "BELOW" means DOMINATED on one row and a −0.204 S WIN on another, then infers they are *"opposite
  ends of the same real line."* Invalid: two oppositely-oriented axes sharing units. Two further
  ba31 errors found: W1-COH described as *"called dominated"* when of1's verdict is CHANNEL
  ADMISSIBLE / open door; and gr1's row labelled *"token-granular **corrections**"* when gr1's
  candidates are **coarsenings**. Proposed fix (not applied — APPEND-ONLY): split §B.5 by sign
  convention and move QA03 out of both tables.

- **FEED-dc1-g [VERDICT]** aimed-correction family: **verdict_scope = FORMULATION** (two independent
  defects — a censored solver and an out-of-domain price — not one bad run; emphatically not FAMILY).
  Pool disposition `do_not_spend` was set on an undecomposed number that could not decide it and
  should not stand on that basis; nothing here earns `spend` either, because the **reach** question
  is untouched. Correct state: **`decidable_next`**, deciding measurement already named, costed and
  resumable (uncap QA03, re-run the 51 censored instances, one scorer slot).
  **Still-open reformulations:** (1) uncapped QA03; (2) direction-extending line search (8× cheaper
  per unit of descent); (3) top_k 120 → ~648; (4) the ru1 joint formulation over neighbour cells,
  never run; (5) label price on a **REAL** flip support rather than pp1's margin-thresholded proxy —
  the largest measurement gap here, since label cost depends on class *composition* and error sites
  are heavily class-skewed (Lane 25.72% vs Undrivable 0.10% error rate); (6) a base-class-excluding
  (4-ary) label coder, which would strictly beat this blind-to-base upper bound.
