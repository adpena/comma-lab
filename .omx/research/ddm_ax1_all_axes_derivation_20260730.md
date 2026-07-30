# ddm_ax1 — all-axes optimization derivation vs the current vehicle (2026-07-30)

Operator-convened (07-30): *"dig deeper … optimize against our topology and deep math and geometry and
all dimensions upstream physics and photometrics and order of operations and dynamics and interactions."*
Derivation arm (task #789): every term derived by comparing the energy the math demands vs the forces in
the code. Consumers: burn-3 config · pj1 fork · vehicle revision. All numbers [macOS-CPU advisory];
labels MEASURED/DERIVED/CONJECTURE; pointer **0.1910828242 [contest-CPU] UNMOVED**.

## §0 PRE-REGISTERED f PREDICTION (committed BEFORE any pj1 result is visible; ddm_pj1 is live)

**Setup being predicted:** pj1 freezes the QA24 endpoint renderer (ema:: weights of
`stage_seg_trunk_tau_final.npz`) and fits ONLY the token field (cell-masked base+delta, quantized,
through R + uint8) to the C1 exact-solve frames (realized d_seg 1.52e-4). f = realized n600 d_seg of
the fitted state.

**Structural derivation (the key point):** against SOLVE-frame targets, target-infeasibility is ZERO by
construction — the targets are already-realized RGB (they exist and survive uint8/R). Therefore
**f measures PURE representation capacity** of (token grammar × frozen renderer), and the QA74
"25.58× gap, 96.1% attackable" typing (typed vs the teacher) splits as: (endpoint→f) = the
training/target gap (what QA75 distill can capture) vs (f→1.52e-4) = irreducible class capacity at
these weights (what only grammar/architecture changes reach).

**Mechanism decomposition (DERIVED from measured structure, with receipts):**
1. **Lane sub-cell microstructure — predicted 40-50% of f.** Lane = 38.7% of endpoint flips and 69.5×
   over its exact floor (renderer-REACH-limited, ledger QA81 row); dash birth/death lives at few-px
   scale inside 16×16-px cells with only 4 code channels/cell — ~1-2 boundary events/cell encodable;
   the known erasure class (error ∝ 1/persistence).
2. **Conv/pointwise expressiveness on curved boundaries — predicted 30-40% of f.** gelu pointwise +
   local conv stack (sg1 §4 lifted-trainer form, QA82 census row i) must place codim-1 boundaries at
   sub-cell precision from smooth token interpolation; curvature beyond local-linear interpolation
   residual concentrates in the rows-160-240 flip band (rowband law, 72.1%, registered
   `rowband_flip_mass_foveation_band_v1`).
3. **Quantization + uint8 floors — predicted 10-20% of f.** 16-level STE token lattice + #532 uint8
   range(A) breakage (Δ=62.74 vs 1.7e-13). Second-order because solve-target margins are
   uint8-feasible by construction — but the fit must HIT them through the quantized code.
4. **Masked cells — predicted ~5% of f.** sg1 §2: kept 384 cells carry **99.61% of flip mass**
   (top-|g|-sum ranking; dropped = sky rows 0-4 + hood rows 20-23, both measured-static classes);
   dropped-cell residual vs solve targets is correspondingly tiny.

**THE PREDICTION (pre-registered):** **f ∈ [7e-4, 2.0e-3], central 1.2e-3** — ≈4.4× below the QA24
endpoint (0.0052766), ≈8× above the teacher (1.52e-4). Fork mapping, pre-registered:
- f inside the band ⇒ **MIXED verdict**: QA75 distill captures ~4.4× (the map half); the remaining ~8×
  is class capacity — burn-3 must carry BOTH the distill target AND token-grammar upgrades (QA84
  rowband D8-in-flip-band + the §3/§4 derived levers below).
- f < 5e-4 ⇒ my capacity decomposition over-estimates; distill-dominant route; class is fine.
- f > 3e-3 ⇒ conv-expressiveness wall dominates; renderer-class change (vehicle revision) leads.
**Caveat (pre-registered):** if pj1's fit is unconverged at its wall-clock cap, its f is an UPPER bound;
comparison uses the loss-curve convergence status pj1 was chartered to report.

pointer 0.1910828242 [contest-CPU] UNMOVED
