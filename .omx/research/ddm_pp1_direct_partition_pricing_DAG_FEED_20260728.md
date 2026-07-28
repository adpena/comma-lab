# DAG FEED — ddm_pp1 direct-partition pricing + band-lemma registration (2026-07-28)

**Arm:** ddm_pp1 ($0 measurement, `[macOS-CPU advisory] NON-PROMOTABLE`). Pointer **0.1910828242
[contest-CPU] UNMOVED**. Isolated worktree off main @ cc4d520004 (ddm_ee1 merged, verified).

## FEED-pp1a (R1) — the direct partition is CHEAP; direct-explicit NOT dead but CONVERGES with implicit
- **MEASURED (real lossless coders, n600 GT `lstars`, bit-exact round-trip):** the direct partition codes
  to **173.6 KB lossless** (KT context-adaptive arithmetic, causal o8 spatial + prev-5 temporal context) /
  **172.2 KB lossy-optimal** (k=2 concession, S_partition 0.1154). Generic coders 330–660 KB (LZMA raster
  410, bz2 338, RLE 332, Paeth 499, per-class planes 660). Temporal-as-CONTEXT saves 33 KB over intra
  (206.9 → 173.6) — confirms ee1 C3 (temporal pays as conditioning, never predict-then-residual).
- **NO-FAKE authority:** the adaptive coder byte length = closed-form Dirichlet-multinomial (KT/Laplace)
  length to <0.01%, PROVEN on a 6-frame subset (coded/closed = 1.0000, bit-exact round-trip True) via the
  in-tree #307 `AdaptiveStream`. Full-n600 closed form is the amortized authority.
- **FALSIFIER (ee1, pre-registered): NOT REACHED.** Death needs lossless ≥350 KB AND lossy ≥250 KB;
  measured 173.6 / 172.2 KB — both far below. 173.6 KB is IN the 120–180 KB third-route band (close to the
  ECC external anchor ~150 KB; the +24 KB excess = ee1 §D.1's SegNet-argmax-noise caveat).
- **Composed arithmetic (HONEST):** explicit-partition route (173.6 KB + trained renderer 40 KB + pose 2 KB,
  native realization 3e-4) composes to **~0.189** — ABOVE the 0.172 bar, because the explicit context-arith
  partition is +57 KB vs PR130's learned tokens (117 KB). Generic-painter route is DEAD (~0.41; corrections
  = 421 KB support wall). **Convergence (ee1 C10): explicit 173.6 KB ≈ PR130 partition leg 177 KB — neither
  dominates.** Direct-explicit is NOT dead but does NOT beat the implicit carrier. **The binding constraint
  is REALIZATION (fd1/R2 slot), not partition coding** — this arm confirms partition coding is cheap.
- **R8 lane-dash sub-race:** #307 contour on the Lane field = 219.5 KB ≫ context-arith Lane attribution
  62.3 KB → the lane-dash dictionary shows NO byte win (context coder already captures lane structure).
- verdict_scope: FORMULATION (object `lstars`, these coder families). Closes ee1 C4 (unpriced direct-
  partition cell) + ar1's stream-1 UNMEASURED cell.

## FEED-pp1b (R3) — the correction-stream position-cost band lemma: CONFIRMED + REGISTERED
- **MEASURED ($0 falsifier):** position (support) coding price at 9 synthetic densities (margin-thresholded
  coherent + random incoherent). Uniform bound log2(1/ρ)/8 is an upper bound on the coherent cost at ALL 9
  points; **measured coherent water crossing ρ_c = 5.0e-4** (derived uniform ρ_u = 8.6e-4; context shifts
  the edge down ~1.7×). Random/incoherent OVERSHOOTS the bound and crosses higher (~1.5e-3). Cross-check: at
  fc1's 0.864% the curve interpolates ~0.44 B/err vs fc1 measured 0.413 — synthesis reproduces the anchor.
- **REGISTERED:** `ddm_pp1_correction_stream_position_band_v1` (evaluator + build + populate, advisory,
  score_claim=false; appended to canonical_equations_registry.jsonl). **Teeth:** corrections rational only
  for base error ρ ∈ ~[5e-4, 1e-2]; below ρ_c concede dominates; above ~1e-2 total support explodes.
  **Design spec sharpened: carrier native error ≤ ~5e-4 (ideally ≤3e-4) to ship NO correction stream.**

## NEXT (not this arm)
- The partition leg is priced-cheap and convergent → the live critical path is REALIZATION (fd1 family-d GN
  in description coords). No new partition-coder arm is warranted (falsifier settled). A learned prior on the
  partition tokens (the ~57 KB gap between explicit context-arith and PR130 tokens) is the only lever that
  would make the EXPLICIT route competitive — but it collapses into the implicit token+renderer (ee1 C10),
  which is fd1's slot.
