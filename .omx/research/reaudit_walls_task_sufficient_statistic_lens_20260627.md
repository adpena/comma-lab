# Adversarial Re-Audit of ALL Load-Bearing Negatives through the TASK-SUFFICIENT-STATISTIC Lens (2026-06-27)

**Trigger:** operator 2026-06-26 — "adversarial review against all negative results and findings and
wall interpretations and limitations" + "the representation and carrier and archive are all the thing
itself, task-sufficient statistic perspective" + "take as long as you want, for the science."
**Authority:** `[contest-CPU advisory]` NON-PROMOTABLE. Pointer UNMOVED contest-CPU **0.19110** (no exact
row moved — stated plainly; means != ends). $0 analysis (read DAG FEED-ca..co + ledgers + memory; no GPU).

**THE LENS:** a "wall" is REAL only if (a) it survives the realized-through-R + CPU-torch + contest-axis
foundation AND (b) no known artifact already beats it. Otherwise it is an ARTIFACT of one of:
**(A)** WRONG REPRESENTATION (storing pixels/appearance, not the task-sufficient statistic);
**(B)** MEASUREMENT ARTIFACT (EMA-shadow-lag 78×, MLX-GPU reduced-precision, field-vs-realized operator
mismatch, proxy-not-exact, MPS 23×); **(C)** SUB-OPTIMAL-FORM (un-tuned lr/w0/temp, default-flag bug,
collapse, under-training); **(D)** TREATING REPRESENTATION/CARRIER/ARCHIVE AS 3 SEPARATE LAYERS.

This CONVERGES WITH and EXTENDS the 4-lens re-founding (`reaudit_refounding_and_md_decoupling_20260626.md`:
"0 of 11 capacity-walls proven fundamental") and the strict scaling-law fit
(`scaling_law_dseg_capacity_fit_20260626.md`). It adds existence-proof cross-checks + a sub-0.19/0.15 EV rank.

## RE-AUDIT TABLE

| # | Wall / negative | Source | REAL or ARTIFACT(type) | Existence-proof cross-check | Reactivation / remeasure path |
|---|---|---|---|---|---|
| 1 | **d_seg 0.507/0.505 CE plateau** | FEED-ck, capstone_ema_shadow_lag | **ARTIFACT (B EMA-lag + C CE-loss/no-curriculum)** | MANY runs descend THROUGH it: muon_throughout 0.507→0.012; capstone_original_small_vq 0.507→0.0103 (49×); FIRE recipe 0.507→0.011; campaign 0.5073→0.0117 (43×); EMA-lag catch live 0.507→0.041 while EMA frozen | RE-OPENED — it IS the active witness path; always report live AND EMA + curriculum (tau_softplus→l7→Muon) |
| 2 | **Morse-Smale codec rate-dominated S~0.37; temporal NO-collapse (3.3%)** | FEED-cl (ada28d2d) | **REAL for STANDALONE per-frame MS (rate is the wall; temporal incoherence robust, 2 indep measurements)** — but NON-load-bearing for the witness (MS was never the carrier) | Witness/PR95 amortize ~177KB vs MS 444KB at equal d_seg → MS standalone correctly DOMINATED. Positive: d_seg **5.57e-4 at 740 B/frame** = geometric existence-proof (partition IS specifiable at capstone d_seg) | Salvage: MS = HARD-TAIL rare-class arc residual coder ON the witness (the sparse hard-pixel sidecar). Junction-TRACKER re-opens only if ≥5× temporal reduction on big regions |
| 3 | **Rate axis "exhausted" at 0.118** | FEED-cc, R4 | **ARTIFACT (A wrong-rep + D layers-separate)** — 0.118 = entropy of RGB-HNeRV (94% rendering weights) | base_ch20 byte-closes **89,628 B → rate 0.0597 (HALF frontier)** at d_seg 0.00256; FEED-cc curve B*~122KB | CAVEAT: cheap-rate AND low-d_seg do NOT coexist in the HNeRV basis (joint min S 0.193) — rate frees only JOINTLY with the basis change (task-space carrier 15–40KB) |
| 4 | **Capacity walls / params^-0.71 power law** | NCA fit, "2×2 capacity-limited" | **ARTIFACT (B+C)** — DEAD as universal law (fit from a COLLAPSED run d_seg 0.508; exponent unstable −0.71/−0.91/−1.52) | Clean converged fit alpha=**1.50**; bc36-converged reaches **5.6e-4** (same-N bc36 spread **17× by training state alone** = Michaud training-limited, not capacity) | RESIDUE (REAL): pure N-scaling in HNeRV basis caps at min S **0.193** (rate grows with N) — a DIFFERENT, real statement = WHY the basis-change is mandatory |
| 5 | **Pose collapse (amortized luma/texture carrier d_pose 2.67–12.66)** | adversarial_review_*:193/:30 | **ARTIFACT (A — reconstructing a STORED quantity)** | Stored-target sidecar d_pose **3.4e-5** (frontier) / bc36-converged 2.36e-5 / bc20 3.04e-4 = pose SOLVED (~1–2KB). Collapse was the flat-PALETTE pose-blind frame1 + luma-INR | Use RGB-render witness + stored-pose sidecar (already the capstone design); NEVER reconstruct pose from a texture INR. "Lever B dead" verdict was pre-RGB-render — superseded |
| 6 | **int5 Path-B cap S~0.49; prune+KD cliff 0.017–0.024** | R2 (scoped), reveng_pr95_prune | **ARTIFACT-of-METHOD (C/A retrofit-a-small-thing-from-a-big-trained-thing) — REAL only for the dominated config** | DECISIVE: from-scratch at SAME N beats it: bc20 from-scratch **0.00256** vs prune+KD bc20 **0.0239 = 9.3× better** at identical params → the "capacity cliff" is the prune/quantize SUBSPACE artifact, not a capacity wall | From-scratch small-basis (already the plan) / score-aware QAT-from-scratch, not retrofit of a converged FP teacher |
| 7a | **Level-set 587× R-survival "might not transfer"** (SDF field argmax ≠ realized SegNet(RGB).argmax) | FEED-cn/co (a7fcedee) | **GENUINELY OPEN — the ONE decisive unresolved crux** (NOT yet REAL or ARTIFACT) | n6 realized CE-only smoke plateaus 0.507 = zero transfer evidence yet; 587× is FIELD-level proxy only | $0 TRANSFER PROBE (fit phi → SegNet(R(RGB)).argmax flips vs argmax(R(phi))) BEFORE any GPU burn — the make-or-break gate |
| 7b | self-orient directional UNWIRED (crash if enabled) | FEED-cn/co | **FIXABLE BUG (C)** — the byte-closeable −48% lever is INACTIVE | self-orientation tangent cos **0.89** vs GT (FEED-ce) = lever IS byte-closeable; just not wired into the loop | Wire it (WIRE/curvelet); flip directional_byte_closeable=True |
| 7c | MLX-GPU verdict / EMA-dead / no-byte-close-forward | FEED-co | **(B) measurement + (D) layers-separate** | parent witness already uses numpy-fp32 CPU verdict + ema.shadow + one-codepath | numpy-fp32/cpu verdict; consume EMA shadow; ONE forward shared by train+inflate |
| 8a | **FEED-bj "isotropic witness non-viable 0.037"** | R1, FEED-bj | **ARTIFACT (B EMA-lag)** | live 0.0022 (17× lower than the lagging-EMA 0.037 reported) | already re-opened |
| 8b | **"directional −48% decisive lever" ranking** | R3 | **PROXY-AXIS unconfirmed (B)** — sister of 7a | the −48% rests on generator-argmax proxy, never realized-through-R | confirm realized in the n96 sweep (same measurement as 7a) |
| 8c | Decode-side floor (Lane A) | R1 | **REAL (1 of 4 that STAND)** — small fixed contribution | realized/contest-axis confirmed | accept as fixed cost |

## HIGHEST-EV RE-OPENS (ranked toward sub-0.19 / sub-0.15)

1. **$0 transfer probe (587× SDF→realized SegNet argmax) + directional −48% realized confirmation** (7a+7b+8b).
   Resolves whether the lower-d basis leg — the ONLY non-dominated sub-0.15 path — is real. $0, gates the GPU burn. The single highest-EV action (make-or-break for the whole capstone).
2. **From-scratch small-basis witness, FIXED recipe + lower-d basis (level-set/directional/step-native), realized-through-R, n96 basis×capacity response-surface → find B* → n600 → byte-close → contest-CPU exact.** The decisive pointer-mover; converts the re-opened d_seg/capacity/rate artifacts into a real curve.
3. **MD-Decoupling optimizer arm** (arXiv 2606.25971) — directly fixes the root cause of #1/#4/#6 artifacts (collapse, no-LR-transfer, warmup-dependence). Parallel ablation folded into #2 (promote only on a byte-closed exact row).
4. **MS hard-tail rare-class arc residual coder composed ON the witness** (#2 salvage) — the sparse lane/movable top-up the witness amortizes poorly; the doctrine's sparse hard-pixel sidecar.
5. **base_ch20 / bc36-converged byte-close → contest-CPU exact** — existence-grounded defensive bank + end-to-end harness test (bc36 ≈ 0.186–0.193; marginal vs 0.19110, a recode → bank only if measured below).

## ANY TRULY-REAL WALLS (the few that survive)

- **HNeRV/RGB-basis rate-d_seg trilemma: pure capacity scaling caps at min S ≈ 0.193** (strict 2-pt converged fit). REAL — but it is a wall AROUND the DOMINATED RGB path; it MOTIVATES (does not block) the task-space basis pivot.
- **Pose √-flatness** (`sqrt(10·d_pose)` flattens as d_pose→0). REAL but pose is SOLVED at 3.4e-5 → fixed ~0.018 cost, near floor, not a blocker.
- **Decode-side floor (Lane A)** + **MS standalone rate-dominance & temporal incoherence (3.3%)** — REAL for their dominated/standalone scopes; the latter correctly weakens the "seg=warp(pose)" fusion claim at the boundary.
- **THE RISK (open, not yet a wall):** if 7a FAILS realized, the best byte-closeable basis ever MEASURED is FINER 0.00138 → S~0.19–0.21 — i.e. 1.23× ABOVE the sub-0.19 goal. Then every measured basis caps near the frontier and a genuinely new basis is required. Sub-0.15 is UNPROVEN until a byte-closeable basis descends realized-through-R below ~1.1e-3.

## VERDICT

Of ~11 load-bearing capacity-walls + the major negatives: **~8–9 are ARTIFACT** (EMA-lag, collapse, CE-plateau, under-training, proxy-axis, MLX-precision, wrong-representation pose-collapse, prune/quantize-retrofit, mis-fit power-law, rate-of-wrong-representation, self-orient bug). **~2–3 are REAL but circle a DOMINATED path** (HNeRV rate-d_seg trilemma min 0.193; MS standalone; int5/prune for-that-config). **~1–2 are REAL fixed costs** (pose √-flatness ~0.018; decode-side floor). **EXACTLY ONE is genuinely OPEN and decisive** (the 587×/directional realized-transfer). The task-sufficient-statistic lens explains WHY: nearly every "wall" was measured on the WRONG OBJECT (pixels/appearance the scorer discards) or read through a corrupted axis — the trilemma and the floors were ARTIFACTS of representing the wrong thing and reading artifacts as floors, exactly the re-founding's finding. **The campaign was not walled; the vast majority of our "walled" space is artifact.** The honest residual: sub-0.15 has clean margin ON PAPER but is UNPROVEN until one $0 transfer probe + one realized-through-R byte-closeable basis descent confirm the lower-d basis actually beats ~1.1e-3. Bias the next slot to that probe, then the n96 basis×capacity burn. Pointer UNMOVED 0.19110.
