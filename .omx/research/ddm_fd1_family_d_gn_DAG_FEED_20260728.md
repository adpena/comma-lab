# DAG FEED-fd1 — family-d GN/CG in description coordinates: S0 box-solve HOLD + the built engine

**Arm:** `ddm_fd1_20260728` (the gc5-adjudicated named build; rp1 CELLS-HOLD routing executed).
**Pointer 0.1910828242 [contest-CPU] UNMOVED.** All rows `[macOS-CPU frozen-scorer advisory]` /
`[macOS-MLX research-signal]`; `score_claim=false`.

**S0 — rp1's named next measurement CLOSED: the HOLD survives at the box-solve operating point.**
Receiver-closed inflate of the 277.7 MB box-solve archive reproduced r6cal custody bit-identically
(raw sha `32a773a2…`). Identical C1 probe (range-carrier zero-ker uint8 lift) on the real box-solve
frames, n600: **C1-vs-C0 cell-hold flip rate 3.757e-4 ≈ the GT band 3.630e-4; margin-absorption invariant
(165× vs 166× gap); C0 custody reproduces the box baseline 1.1600e-3 exactly; C1 vs lstars 1.2492e-3 =
1.077× C0; pose collateral +1.1%.** The rp1 GT-substrate caveat ("smaller margins would flip somewhat
more") is MEASURED CLOSED — the cell-hold band is operating-point-INVARIANT; the engine's realistic
realization-noise band is ~3.6–3.8e-4 per range-carrier uint8 realization.

**S1 — the family-d engine EXISTS and is a real solve (NO-FAKE #6):**
`FamilyDGaussNewtonEngineV1` — matrix-free damped Gauss-Newton/CG on the j2 lifted description DOF
through the exact linearization (paint → uint8-STE → fused-R → frozen SegNet); GGN `100·JᵀH_CE J/N`;
`mx.jvp` walls at the fused-R CustomKernel → quantum-scale central-secant JVP (ε=0.5, the same
linearization scale as j2's realized ±1-quantum secants) + exact reverse-mode transpose; ms4d metric
bundle loaded COMPLETE via the fail-closed loader (custody recorded per proposal; preconditioner =
measured Hutchinson Jacobi diagonal — explicit degradation, atlas dims do not index lift params).
Governed mode `--fd1-gn-window` in the launcher; v19 acceptance UNCHANGED; pose terminal (#383).
Smoke (block 447–450, 344 active params): 0.33 s/HVP, propose 2.0 s; full GN step reduces the exact
block objective 27.015 → 23.816 (−11.8%) measured through the STE forward — the second-order solve
moves the same objective the first-order engine descends, at negligible proposal cost.

**S2 — bounded governed GN window from the arbitrated W_joint: ZERO ACCEPTED STEPS (clean
instrument reading).** 2 GN steps × 3 multipliers = 6 realized candidates, all REJECTED by
unchanged v19; both steps classified `BLOCK_LOCALITY_OR_REALIZATION_GAP`. The solve is REAL
(block objective −7…−13% per proposal; CG res 0.63→0.325; 0.33 s/HVP; propose ~6 s of a ~1,547 s
step — 99.6% of wall is acceptance pricing). The failures, decomposed: (i) cross-pair seg transfer
ZERO — realized n600 d_seg bit-identical to baseline in 5/6 candidates (shared DOF tuned on a
4-pair block do not move the scored population through the uint8 staircase); (ii) pose collateral
+2.8…+13.1% prices every candidate out (pose leg = 72.8% of action at the pose-unsolved warm
start). Slope 0.000%/step vs ws3's measured first-order −0.078%/step (the −1.26%/ep charter
reference is gc5-B8-scoped ANCESTOR, cited-not-used). d_seg stayed at 0.0702 — 187× above the S0
band; NOT engine-scale saturation vs the band; ee1 band-lemma crossings not reachable. Peak RSS
12.03 GiB vs 15.75 projected. Receipt `s2_gn_window/fd1_gn_window_receipt.json`.

**SEAL: TYPED BLOCKER `BLOCKED_ZERO_ACCEPT_WINDOW_CAPACITY_ROUTED`** (fire owner MAIN; the
ticket's own `continue_while` is violated by the first measured window — a READY seal would be a
fake readiness claim). Routed to the gc5 capacity disambiguator TWO-RUNG ladder (steers 1+2 + ee1
C10): rung 1 = grow shared/cross-pair template DOF (+ #383-dual pose-null projector on the seg
step); rung 2 = token-grid + trained partition→pixel renderer (≤64 KB, scorer-in-loop; PR130
existence 40,252 B int4 → 2.97e-4 ≤ ρ_c = 5.0e-4 no-correction threshold, LESSONS-ONLY). pp1
context: partition leg 117–177 KB across three parametrizations, composed explicit S≈0.189 above
the 0.172 bar ⇒ REALIZATION is the campaign's binding constraint; this slot is the named
differentiator. Ticket custody resealed (launcher a2adf9d0 / engine 142b6ca1, semantic RFC8785
re-hashed); typed identity = the ws3 ticket (bit-faithful W_joint lineage).

**Artifacts:** memo `.omx/research/ddm_fd1_family_d_gn_description_engine_20260728.md` · engine
`src/tac/optimization/ddm_family_d_gn_description.py` · launcher mode in
`tools/launch_ddm_joint_descent.py` · S0 tool `tools/measure_ddm_rp1_rangeA_cell_probe.py` · SSD
`/Volumes/VertigoDataTier/pact/ddm_fd1_20260728/`.
