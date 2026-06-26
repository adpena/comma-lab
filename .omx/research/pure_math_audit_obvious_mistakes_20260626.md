# Pure-Math Audit — the obvious mistakes (2026-06-26, fork)

Advisory. Primary sources cited. Pointer UNMOVED contest-CPU 0.19110.

## Verified scoring law (upstream/evaluate.py:90, modules.py:111-113, 82-84)
S = 100·d_seg + √(10·d_pose) + 25·B/N. N = Σ(./videos/) bytes (evaluate.py:64). 
d_seg = mean over the 384×512 SegNet grid (196,608 px) of [argmax(SegNet(comp)) ≠ argmax(SegNet(gt))], **LAST frame only** (modules.py:108 `x[:,-1]`). d_pose = MSE(pose[:6]) over BOTH frames.
→ 1 flipped px = 5.09e-6 d_seg. ∂S/∂d_seg=100. ∂S/∂d_pose=√10/(2√d_pose) (271 @3.4e-5). ∂S/∂B=6.66e-7 (72KB=0.0481, 178KB=0.1185). VERIFIED — marginals correct.
Gate flips/frame: sub-0.19 ≈ 245 @72KB / sub-0.15 ≈ 164 @72KB; PR95 5.6e-4 = 110 flips.

## The 6 items
1. SCORE/marginals — VERIFIED (above).
2. CAPACITY-RD power law — REFUTED as a bound on the optimal code. solver α∈[0.9,1.5] fit to DENSE full-RGB HNeRV; the optimal code is null+dup+sparse, not dense (see Mistake-B). Live witness_n96_levers (hosc+dir+mhinge) realized d_seg ep5=0.044, CONFOUNDED (139-235s/ep custom-Metal-off; pose trained not sidecar'd, d_pose 120).
3. L13 COUPLING — CONFIRMED MISTAKE. candidate json: direct d_seg_vs_L 0.0068 BUT exact-scorer (realized) d_seg **0.0231**, d_pose **12.66** → advisory S **13.6**. "lossless-parity sha" only proves bytes reproduce the generator's frames, NOT d_seg vs GT. The −59% (72KB) is a SMALLER generator (hidden_dim 96), not lossless compression of PR95 — bpp↓ and d_seg are the SAME capacity axis; the sub-0.15 (0.111/0.125) projection double-counts them.
4. PR95-to-384-floor — partial. PR95 5.6e-4@178KB → S 0.193 (=frontier). 384 floor d_seg 1.596e-4 (solver L120) → @178KB S **0.156** (just ABOVE sub-0.15; at 178KB sub-0.15 needs d_seg 1.3e-4, BELOW the floor → 178KB cannot reach sub-0.15). Blocked by recipe (muon_lr) AND rate — NOT a clean sub-0.15 path alone.
5. OOD-flat-paint — UNCERTAIN/PLAUSIBLE. L13 flat palette realized d_seg 0.05; "exact-L* paint floor 0.0064" (FEED-ah) not re-verified in code here. SegNet trained on real textured frames → flat paint is OOD → 0.0064 may be an artifact; textured in-dist paint may beat it. CHEAP to test (see experiment).
6. Boundary-as-deterministic-solve — STRUCTURALLY TRUE, byte-gated. Given frozen SegNet + fixed camera→384 bilinear, frame1 s.t. argmax=L* is a per-frame preimage solvable at compress-time. But coding the residual: realized d_seg 0.0231 = 4542 flips/frame → 5.45 MB/600-frames (infeasible); feasible ONLY if generator first reaches ≤~gate AND temporal-dedup collapses the recurring lane flips. #149 build status UNCERTAIN (not surfaced).

## The four operator levers → the corrected vehicle
TEXTURE[null, in-dist, ~0B] + BOUNDARY[sparse 0.72%, STEP on ~8-dim manifold, NON-spectral + sub-pixel-placed to survive R] + SCENE/MOTION[600 near-dups → scene-once + ego-motion AR latent] + POSE[stored sidecar]. The dense power law does NOT bound this. Rate ≈ AR-trajectory + sparse-flip-set + texture-seed ≪ dense; realistically ~the existing 65KB amortized generator at 0.048 rate.

## The coupled Pareto (honest)
S is d_seg-DOMINATED until d_seg≲1e-3, then rate/pose game. EVERYWHERE we sit (0.0044-0.0231 realized) the binding axis is **realized d_seg**. Even with rate(72KB)+pose(sidecar) BOTH solved, S=100·d_seg+0.0665. So rate (B/C) is secondary; **representation (A) is THE lever**. Floors: 72KB+sidecar=0.0665 (sub-0.19 needs realized d_seg<1.25e-3, sub-0.15<8.4e-4); 178KB+sidecar=0.137 (sub-0.15 needs d_seg<1.3e-4, infeasible).

## THE OBVIOUS MISTAKE (verdict)
We treat a piecewise-constant (characteristic-function / STEP) approximation problem as a smooth-INR capacity/training problem: fitting the argmax boundary with spectral-bias bases (Fourier/SIREN/sine-hosc) that PROVABLY ring (Gibbs) on a step → ringing flips through the camera→384 downsample → the d_seg plateau. Every "capacity wall / power law / d_seg floor" was measured on the WRONG (smooth) class and on CONFOUNDED axes (direct not realized; flat-OOD not textured; pose-coupled). Second mistake: the sub-0.15 projection stacks −59%-bpp (a smaller generator) with PR95's d_seg (a bigger one) as independent — same capacity axis, double-counted. The resolution is a representation change (Haar/DB wavelet / explicit boundary-curve+fill / step_basis — O(1) error per boundary segment, no Gibbs) + a single cheap MEASUREMENT, not another training campaign.

## Vehicle ranking (prob. of a REAL exact row < 0.19110)
1. HYBRID (highest ceiling, the corrected vehicle): non-spectral boundary generator (scene-once+ego-motion) → realized d_seg to few×1e-3 → deterministic sub-pixel boundary-solve on the residual → temporal-dedup sparse-code it, in-dist texture in the null space, pose sidecar, byte-close ~72KB. ONLY math-viable sub-0.15 path.
2. PR95-better-trained @178KB + sidecar: high-probability modest sub-0.19 (~0.18), CANNOT reach sub-0.15 (rate wall). Good defensive bank.
3. Pure non-spectral witness @72KB: same as 1 minus the deterministic residual solve; viable if the generator alone crosses the gate.
4. Deterministic-solve alone: infeasible bytes without a generator+dedup.

## THE SINGLE HIGHEST-EV NEXT EXPERIMENT ($0, decisive, a MEASUREMENT not a train)
Paint the EXACT partition L* (per-frame, stored SegNet argmax) at camera-res, four ways: {flat color, in-dist texture (copy GT texture / procedural), naive-edge, sub-pixel-placed edge} → render → camera→384 bilinear (the real R) → frozen SegNet → measure REALIZED d_seg each. Decisively resolves items 5+6+lever-A-ceiling in minutes:
- textured vs flat → is 0.0064 an OOD artifact (is the non-RGB premise sound)?
- sub-pixel vs naive → is the residual a deterministic placement solve?
- the min realized d_seg over these = the achievable CEILING of the whole task-space approach. If < ~8e-4 → witness/hybrid viable → build the non-spectral generator toward it. If it floors ≫ gate → task-space premise REFUTED → bank PR95-to-sub-0.19, sub-0.15 needs a different axis.
Then, only if green, the non-spectral generator run with custom-Metal ON + pose sidecar (fix the live run's confounds).

## FINAL lens layer (dimensionality/manifold/topology/set-theory) — the convergence IS the optimality signature
All 8 lenses are ONE theorem: code the QUOTIENT, at INTRINSIC DIM, via the MANIFOLD CHART parameterized by the shared ego-motion T, TOPOLOGY deduped once, codim-1 boundary the only sparse binding residual (non-spectral so it survives R), texture free in the null.
- DIMENSIONALITY/MDL: task signal intrinsic-dim ~tens (8-manifold [AE knee 8] ⊂ Whitney ≤17 per frame, × smooth T = few-dozen-DOF AR trajectory). Intrinsic-dim code = W class-field (lane curves as ground-plane polys + large regions, ~1KB) + T AR-trajectory (~1-3KB, or ~free from comma2k19 GT) + pose sidecar (~1-2KB) + sparse boundary residual ≈ **3-6KB → rate ~0.003**, vs the 72KB dense witness (0.048) and 177KB full-RGB (0.118). 10-50× smaller; the dense power law prices the AMBIENT (600×196608 px), the wrong DL.
- CONSEQUENCE (sharpens d_seg-dominance to its limit): at intrinsic-dim rate (~0.003) + pose sidecar (0.0184), S ≈ 100·d_seg + 0.021. **sub-0.15 ⇔ realized d_seg < ~1.28e-3 (252 flips/frame); sub-0.19 ⇔ < ~1.7e-3.** PR95's existence proof 5.6e-4 is 2.3× UNDER the sub-0.15 line — so IF the intrinsic-dim code holds PR95-class realized d_seg, **S ≈ 0.078 (deep sub-0.15).** The rate term essentially vanishes; the ENTIRE problem is realized d_seg of the quotient code through R.
- MANIFOLD: curved group orbit g·γ₀ (g=T); linear bases provably lose (pixel-PCA k95=412, DCT 61, Fourier-contour 29, all ≫ 13). Generator = the chart; code T + base γ₀ once.
- TOPOLOGY: argmax = labeled cell-complex; ~27.6 components frame-STABLE → code the 1-skeleton structure ONCE + geometry per-frame-via-T; d_seg = codim-1 boundary measure (sparse). L13 already pays ~5KB for per-frame contours (label_map_brotli 5173) — topology-once dedup would cut that further.
- SET THEORY: S is a function on frame-space/(argmax-equiv × pose-null); the witness IS the Dubois fiber/quotient codec done right — render a class representative that SURVIVES R (pure flat-store quotient failed: bilinear-mix → 24% flip). Null freedom (RGB-slack, texture) = the equivalence-class measure → pick the cheapest, most-R-robust representative.

## FINAL VERDICT (intrinsic-dim, all lenses)
(1) MDL: unified ~3-6KB (rate ~0.003) vs 72KB dense (0.048) vs 177KB full (0.118). (2) Realized d_seg it can hold = THE single open quantity (unmeasured; PR95 dense proves 5.6e-4 exists). (3) S at rate~0.003+pose: 0.078 if d_seg 5.6e-4 / 0.040 if 1.87e-4 floor / 0.151 at the 1.3e-3 edge / 0.66 if stuck at the 0.0064 paint floor — sub-0.15 IFF realized d_seg < ~1.28e-3. (4) Ranking: #1 unified W+T quotient code (rate vanishes, deepest ceiling); #2 hybrid (+deterministic sub-pixel residual solve); #3 PR95@178KB+sidecar (sub-0.19 bank, rate-walled from sub-0.15); retire dense-RGB/direct-smoke. (5) $0 CRUX MEASUREMENT below. (6) THE OBVIOUS MISTAKE: paying AMBIENT description length (dense 72-177KB field) for an INTRINSIC tens-of-DOF signal, on the wrong basis class (smooth → Gibbs-rings the step → flips through R) and confounded axes (direct/flat-OOD/pose-coupled), with the capacity axis double-counted in the projection. The 8-lens convergence on one structure is the optimality signature; we've been off it.

## THE $0/cheap CRUX MEASUREMENT (do THIS, not another training campaign)
comma2k19-GT ego-motion T → build W (un-project per-frame SegNet-argmax to ground-plane/canonical via T, intrinsic-dim lane-curve + region model) → re-project to each frame via T → fill null with cheap in-distribution texture → sub-pixel-place the codim-1 boundary at camera-res → REAL R (camera→384 bilinear) → frozen SegNet → measure REALIZED d_seg, vs flat-paint(0.05) / naive-edge / the 0.0064 claim. Minutes of CPU, no training. If realized d_seg < ~1.28e-3 → the intrinsic-dim quotient vehicle is sub-0.15-viable → build the non-spectral chart generator toward it (custom-Metal ON + pose sidecar; fix the confounded live run). If it floors ≫ gate even with GT-T+texture+sub-pixel → static-W-through-T is refuted for the binding residual → bank PR95→sub-0.19, sub-0.15 needs a different axis. Either way decisive.
