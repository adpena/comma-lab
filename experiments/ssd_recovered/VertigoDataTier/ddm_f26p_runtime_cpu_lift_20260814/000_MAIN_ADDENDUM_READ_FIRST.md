# MAIN ADDENDUM to charter_ddm_f26p_runtime_cpu_lift_20260814 — READ BEFORE LEG A

**CRITICAL PRIOR RECALLED BY MAIN (do not re-discover): the CPU axis on this
lineage was ALREADY MEASURED on lc2, 2026-08-10.** Memo:
`.omx/research/ddm_lc2_adjudication_and_cpu_verdict_20260810.md` §2 (it is the
memo that "settles #998's open leg"). Measured facts:

- Call fc-01KZP70MKR3Z5B0XZG5BZK77GM, contest_cpu, locked venv, **8-core**/16GB
  Modal container: **inflate TIMEOUT at 1,800 s.** Token ANS decode 1,777.6 s
  for 600/600 (~3 s/pair, SEQUENTIAL) + render 180.8 s ≈ **1,958 s total**.
- Contest CPU runner is **4-core** — strictly slower than the measured 8-core.
- The raw decode WAS fully written (3,662,409,600 B) and harvested to
  `experiments/results/ddm_lc2_exact_row_20260810/harvest_cpu/` — the CPU
  decode COMPLETES and may serve your identity comparison for the lc2 sibling;
  MC36 is PR135/F26-lineage (a different decoder), so your Leg B identity run
  on the MC36 archive is still required.
- Verdict was `INSTANCE(lc2 decode speed)` with the reactivation NAMED:
  "a decode that fits 1800 s on 4 cores (e.g., parallel per-stream ANS decode
  or a faster token codec)."

## Consequences for your legs (charter amendments, binding)

1. **Leg C's budget verdict is nearly foreknown for the family**: sequential
   entropy decode dominates (~91% of CPU wall on lc2). Expect OVER. Do NOT
   spend long confirming a foreknown OVER at full n600 — measure a REPRESENTATIVE
   subset wall-clock (after the full-decode identity run of Leg B, which you
   need anyway), extrapolate honestly, and move your effort to Leg D.
2. **Leg D IS the main event, not a contingency**: the deliverable is the
   decode-speed engineering plan + first implementation increment — parallel
   per-stream ANS/HPAC decode (the streams are independent; 4 workers ≈ 4× if
   stream counts allow) and/or Rust lowering of the entropy-decode hot loop
   (full native grant; runtime-rs parity precedent). Target: total decode
   ≤ ~1,500 s projected on 4 contest cores (margin under 1,800). Derive the
   needed speedup factor from YOUR measured MC36 numbers, not lc2's.
3. **Honest upside framing**: the CPU-axis SCORE on this lineage is
   UNMEASURED (the decode never finished in budget) — unknown sign, do not
   quote PR102's −0.033 as an expectation; it is a different vehicle. The
   deliverable that matters is making the measurement BUYABLE.
4. Everything else in the charter stands (identity vs retained sha a41ca69d…,
   4-thread caps, payload law, MLX asset inventory Leg E, no Modal).
