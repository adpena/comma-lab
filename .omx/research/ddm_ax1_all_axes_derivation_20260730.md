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

---

## §1 TOPOLOGY
- **Recall (MEASURED):** Lane 38.7% of flips, 69.5× over exact floor (QA81 row); island/worldsheet
  birth-death events are the erasure class (error ∝ 1/persistence, L1 unification); rg3 class-birth
  productions closed the RG-coordinate residue; sg1 keep-mask covers 99.61% of flip mass.
- **Gap in the code:** the vehicle has NO term that protects low-persistence features — margin-weighted
  CE weights by margin, not by persistence; dashes (birth-death pairs) can be traded away for bulk-margin
  gain at equal loss.
- **DERIVED term:** persistence-aware Lane treatment = the cb1 explicit Lane band carrier composited
  IN-training (QA81, designed) so the trunk trains on the residual after the thin structure is painted —
  topology delegated to the carrier, texture to the trunk. Already designed; derivation here RATIFIES it
  as the topology-axis optimal (predict-then-innovate is exactly persistence-splitting).
- **Falsifier:** QA81 matched-budget A/B — no Lane-class d_seg win ⇒ instance-close.
- **Consumer:** burn-3 lever `qa81_lane_carrier_composite` (built stub, ph3 §10).

## §2 DEEP MATH
- **Recall (MEASURED):** rank-4 head + flip distance d=|m|/‖Δw‖ (segnet fractal); margin-Fisher custody
  (ms3/ms4); band lemma (pp1, LAW); ker(A) 80.67% resize nullity; 52% scorer-invisible render energy
  (#519/#520); product law c ×0.60/burn (gc9).
- **Gap:** (a) the loss weights pixels by margin but the QUANTIZER treats all token dims identically —
  quant noise is spent uniformly while flip risk is d=|m|/‖Δw‖-structured; (b) 52% of render energy is
  scorer-invisible yet the token code pays rate for it (SMEVR codes what the field IS, not what the
  scorer SEES).
- **DERIVED terms:** (a) margin-coupled token quantization — allocate effective token precision (dither
  scale / level count) per cell by the cell's aggregated flip-distance mass (NEW lever, §DSL);
  (b) range(A)-projected rendering target — project the render loss through the #580 projector so
  invisible energy is free to be whatever compresses best (gauge freedom AS a rate lever; ties to #519).
- **Falsifier:** (a) matched-byte A/B no d_seg win ⇒ instance-close; (b) projector-composed loss changes
  nothing at equal SMEVR bytes ⇒ the coder already absorbs the gauge (possible — SMEVR contexts may
  partially exploit it); measure, don't assume.
- **Consumer:** burn-3 levers `ax1_margin_coupled_token_quant` (new stub) + a gauge-projection loss row
  routed to the b2b config window.
- **What sets the ×0.60 slope (DERIVED, labeled):** each burn so far added one structure class to the
  grammar (burn-1: trained tokens; QA24: mask+margin-weights+rate-in-loss). Δlog c per burn ≈ the
  information the new structure class captures. Prediction: burn-3's slope is set by the LARGEST unclaimed
  structure class = the distill target (map) + Lane band (topology) — consistent with §0's split.

## §3 GEOMETRY
- **Recall (MEASURED):** op1 P2 — token grid STAYS in the image plane (98.806% of flip mass
  image-stationary; ξ does not place boundaries; >half the field has no BEV chart). Rowband law 72.1%
  rows 160-240 (registered). QA84 RowBandGrammar BUILT (D8 flip-band / D16-effective tied bulk, DOF 1248,
  SMEVR-closed, `--token-rowband-spec`).
- **Gap:** the QA24 endpoint ran uniform D16+drop50 — the flip band rows 160-240 gets the same cell
  resolution as the bulk despite carrying 72.1% of the mass.
- **DERIVED:** the D∝(flip-density)^-α allocation field (QA84's own derivation) — rowband D8-in-band is
  the separable optimum; quadtree pays only if in-band azimuthal sparsity is real. Nothing new to build:
  **burn-3 fires qa84_grammar_race_programs {uniform-D16-drop50 control, rowband-D8}** as pre-registered.
- **Falsifier:** matched counted-byte race — rowband no win ⇒ QA84 instance-close.
- **Consumer:** burn-3 token grammar (QA84 built lever). BEV re-litigating: CLOSED by op1 P2 (cited, not
  reopened).

## §4 ALL DIMENSIONS (class / frame / pair / cell / channel)
- **Recall (MEASURED):** frame_0 is structurally seg-free (scorer reads frame_1 for seg; frame_0 is pure
  pose surface — currently a ZEROS stub, d_pose 160.4 at the endpoint); 98.8% image-stationarity ⇒
  temporal deltas should be near-zero almost everywhere; SMEVR temporal contexts exploit this at CODING
  time (QA85 receipt) but training has NO delta-shrinkage force; lv1 solve-init MEASURED −34.4% @ ep4
  (adopted in QA24); v8 per-class carriers = QA81 (§1) for Lane, hood/sky already handled by mask.
- **Gaps + DERIVED terms:** (a) **delta group-sparsity** — energy demands near-zero deltas on the 98.8%
  stationary mass; code has no such force; add group-L1/L0-ish shrinkage on per-pair token deltas (rate
  falls at the SOURCE, compounding with SMEVR; NEW lever, §DSL). (b) **frame_0 carried-ξ warp** — replace
  the zeros stub with receiver-side warp of rendered frame_1 by the carried pose (rule-118 free, ZERO new
  tokens, zero seg risk by frame_0 seg-freedom) — the pose surface gets a physically-plausible photometric
  base for the terminal 6-eq solve instead of zeros (NEW lever, §DSL; distinct from QA39, which is
  token-CODING inter-prediction).
- **Falsifiers:** (a) shrinkage costs d_seg at matched bytes ⇒ weight too high — sweep is cheap;
  (b) warped frame_0 measures WORSE d_pose than zeros through the terminal solve ⇒ instance-close (the
  solve conditions on whatever base exists; zeros is a degenerate but unbiased base).
- **Consumer:** burn-3 levers `ax1_delta_group_sparsity`, `ax1_frame0_carried_warp` (new stubs).

## §5 UPSTREAM PHYSICS
- **Recall (MEASURED):** ξ carried (se(3) screw); op1: ξ predicts motion but cannot place boundaries;
  ground-plane homography custody in op1; camera geometry fixed.
- **Gap:** physics' remaining un-cashed value after op1's verdict is NOT boundary placement — it is
  (a) the frame_0 pose surface (§4b uses ξ physically) and (b) the delta PRIOR (ego-motion says WHERE
  image-stationarity breaks: the movable band + lane-corridor dash phase — exactly where deltas should
  concentrate).
- **DERIVED:** the delta-sparsity weight map (§4a) should be ξ-informed: relax shrinkage in the
  movable band / dash corridor, tighten in the static mass. Folded into §4a's lever as its weight field
  (no separate lever; same pool).
- **Falsifier:** uniform vs ξ-informed shrinkage at matched bytes.

## §6 PHOTOMETRICS
- **Recall:** QA80 margin-bounded photometric (built stub; band lemma makes it provably seg-flip-free);
  ea1 N3 made it a burn-3 REQUIREMENT (pose-legibility from birth); QA44/QA66 (a,b) rungs measured on
  the v4 line (−0.0134 S @ +150B, ja1 TOP row); chroma: SegNet reads RGB, PoseNet reads YUV6 —
  luma carries pose.
- **DERIVED:** the photometric budget lives in the margin-slack region (below flip distance) and in
  frame_0 (fully free). Composition: QA80 shapes frame_1's slack luma; §4b gives frame_0 the physical
  base. Together they dissolve L68 pre-emptively. Nothing new to build beyond §4b — QA80 already built.
- **Consumer:** burn-3 requires `qa80_margin_bounded_photometric` ON (per ea1 N3 derivation).

## §7 ORDER OF OPERATIONS
- **Recall (MEASURED):** compose order mask→tie→quantize→render→R→uint8 (57471fafaf law: mask BEFORE
  quantize); fc1 dependency-order lesson (solve → gauge-split → predict → quantize → truncate → entropy-code);
  event-driven schedule (#302/#686); lv1: solve-INIT before train (adopted).
- **DERIVED order for burn-3 (each adjacency reasoned):** solve-init tokens (map anchor) → carrier
  composite paints Lane (§1; residual defined BEFORE trunk training) → distill-dominant opening on solve
  targets (QA75; the map) annealing to margin-CE (the territory) → delta-shrinkage engages AFTER the
  base stabilizes (event: base-churn below threshold — shrinking deltas against a moving base is noise) →
  QA80 photometric engages at the #383-style conditioning gate → terminal pose solve LAST on the
  conditioned base (staging law, unchanged). One NEW ordering constraint derived: **delta-shrinkage
  after base-stability event** — the only adjacency not already encoded in a law.
- **Consumer:** burn-3 schedule spec (DDMEventContinuation family; b2b window).

## §8 DYNAMICS
- **Recall (MEASURED):** endpoint COUPLED_DESCENT, joint descent proven live (bytes 355,343→250,898
  while gate d_seg fell); ep_loss 0.528 final; EMA derived-decay law fired (0.99986667); QA86 w_rate
  derived-estimate 0.0768 vs live 0.05 (~65% of S-commensurate).
- **Gap + DERIVED:** the endpoint was still descending on BOTH axes at ep399 (COUPLED_DESCENT, no
  plateau event fired) ⇒ QA24 was schedule-truncated, not converged — part of the 0.00528-vs-prediction
  gap in §0 may be TRAINING-length, not capacity. Burn-3 length should be event-terminated (plateau
  detector), not epoch-capped; w_rate corrected to the derived 0.0768 (QA86 item d).
- **Falsifier:** if pj1's scorer-free fit (unlimited by the burn schedule) converges near the §0 band,
  training-length explains little; if pj1 blows past the band downward, length was a bigger share.
- **Consumer:** burn-3 schedule (event-terminated) + QA86 config corrections.

## §9 INTERACTIONS (the pools)
- **Pool A — token byte budget (COMPETING, one waterfill):** QA84 rowband allocation × §2a
  margin-coupled quant × §4a delta-shrinkage all spend/save the SAME counted token bytes → they enter
  ONE allocation race at matched SMEVR bytes, never stack-claimed additively (non-additive-pools LAW).
- **Pool B — the training target (composes with A):** QA75 distill changes WHAT is fit, orthogonal to
  how bytes are allocated; composes multiplicatively with Pool A in c.
- **Pool C — pose/photometric slack (orthogonal to A by band lemma):** QA80 + §4b frame_0 warp + terminal
  solve; provably no seg cost ⇒ genuinely composes.
- **Pool D — schedule/dynamics (§7/§8):** ordering + termination events; interacts with everything but
  spends no bytes.
- **v19b precedent (+0.0805 synergy) says measure Pool A JOINTLY, not per-lever.**

## §10 COMPOSED BURN-3 DERIVED STACK + ARITHMETIC
Stack: solve-init → QA81 carrier → QA75 distill-opening→margin-CE anneal → QA84 rowband ∥ §2a quant ∥
§4a shrinkage (ONE Pool-A race) → QA80 + §4b frame_0 warp → SMEVR + QA86 corrections (w_rate 0.0768,
EMA law, surrogate A/B) → event-terminated → terminal pose solve (QA43/6-eq).
**Arithmetic (DERIVED, gate-grade only via fidelity law):** central case = §0 f ~1.2e-3 reached via
distill + Pool-A cures pushing toward ru1's independently-derived corrected box (~6e-4 @ 130KB;
ρ_c 5.0e-4 corroborates the scale): seg 0.06 + rate 0.087 + pose 0.02-0.05 (solved on conditioned base)
→ **S ≈ 0.17-0.20 central** — AT the official bar 0.172141, not comfortably below; T_3 0.15 requires the
favorable tail of BOTH the distill capture and the Pool-A race. Product-law: c ≈ 6e-3-1.2e-2 vs needed
5.05e-3 — one further slope-holding burn beyond burn-3 remains the honest central expectation.
**verdict_scope note:** every negative implied above is instance/formulation-scoped as stated per row.

## OP-ROUTABLES
1. pj1 lands → compare f vs §0 band HONESTLY (convergence caveat) → gc9 fork table row.
2. Burn-3 config window consumes: §7 order + §9 pools + QA84/QA75/QA80/QA81/QA86 + the 4 new stubs.
3. Register `ax1_capacity_split_v1` as canonical equation ONLY after pj1 confirms/refutes the band
   (prediction → law needs the measured anchor).
4. Pool-A joint race harness = the one named build gap (extends qa84_grammar_race_programs to include
   quant-coupling + shrinkage arms at matched SMEVR bytes).

pointer 0.1910828242 [contest-CPU] UNMOVED
