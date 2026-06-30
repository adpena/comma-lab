# Canonical research index — POSE + CURRICULUM/SCHEDULING/OPTIMIZER/TRAINING-DYNAMICS

**Date:** 2026-06-29 · **Authority:** `[research-consolidation / advisory]` — $0 CPU/read-only, NO GPU, NO
MPS, NO training launch. **Pointer UNMOVED contest-CPU 0.19110** — only a byte-closed n600
`upstream/evaluate.py` row from the witness moves it. **MEANS≠ENDS:** this index is a MEANS (marshaling
measured signal so the from-scratch openpilot-seeded run launches at TRUE optimal form); it does not move
the pointer. NO-FAKE: every row cites an exact pointer + calibration tag; UNGROUNDED/ESTIMATE/SPECULATION
are labelled. **Mission framing:** operator 2026-06-29 "signal loss + rediscovery + STARTING LESS OPTIMAL
than we could with perfect recollection." This is the POSE + CURRICULUM/OPTIMIZER slice of the corpus sweep.

**Sister index slices / canon:** `curriculum_openpilot_seeded_deepmath_dsl_20260629T034000Z.md` (the deploy
config, afd0af1d/FEED-ln) · `per_stage_fractal_optimizer_priming_reheat_anneal_design_20260629.md` (reheat) ·
`pr95_segnet_stage_evolution_for_witness_curriculum_20260626.md` (stage evolution) ·
`anneal_optimal_math_geometry_calculus_20260613.md` (τ resonance) ·
memory `[[session-20260630-review-warpfix-lossless-exhausted-CURRENT]]` (the MEASURED TOOLBOX) ·
`[[different-stages-need-different-treatment...]]` · CLAUDE.md EMA / eval_roundtrip / OPTIMAL-FORM /
never-launch-non-resumable non-negotiables.

---

## 0. OPTIMAL-CONFIG CONTRIBUTION (the lead — the marshaled best for the from-scratch deploy)

### 0a. POSE (measured-optimal handling — pose is SOLVED via STORE, not train, not warp, not render)

| knob | OPTIMAL setting | why / cite |
|---|---|---|
| **mechanism** | **STORE the 6 PoseNet scalars/pair** (`src/tac/scorer_targets.py`) | scorer computes `MSE(PoseNet(gen)[:6], PoseNet(orig)[:6])`; store `PoseNet(orig)[:6]`. 600×6×fp16 = **7,200 B raw / <5 KB zlib**, d_pose≈0. GROUNDED (scorer_targets.py:9-13; DAG :612) |
| **render the pose?** | **NO — `--w-pose 0`** | pose rides the sidecar; the witness's ONLY controllable job is d_seg. GROUNDED (deepmath memo; CURRENT-STATE memory) |
| **stored pose: train it?** | **NO — FROZEN at inference** | Quantizr D5: a trainable stored pose drifts OFF the exact target the scorer measures against. GROUNDED (quantizr audit §2 D5) |
| **further codec (opt-in)** | **low-rank pose codec rank-4/511** (`PFL2`, default-OFF) | **2,563 B, MSE 2.7e-5 (≤ iid 2.9e-5), smaller than iid 3,088 B → Pareto-dominant, ~−0.0004 rate.** NOT naive rank-2/254 (net-NEGATIVE). MEASURED torch-CPU advisory (#140, `lowrank_pose_section_codec_landed_20260617.md`) |
| further codec (opt-in) | pose_from_embedding MLP ~1-2 KB | replaces optimized_poses.pt ~15KB; PARTIAL (DAG :612) |

> The from-scratch witness does NOT touch pose: store 6 scalars, freeze, `--w-pose 0`. The "pose collapse"
> (d_pose 0.06-0.34) was a *content-free-latent rendered-pose carrier* problem, NOT the stored sidecar — do
> NOT re-treat pose as open (§4 CONFLICTS). A RENDERED-pose vehicle is a *separate parallel arm* (§0c).

### 0b. CURRICULUM / OPTIMIZER (measured-optimal for the from-scratch openpilot-seeded witness)

SHORT, d_seg-only, NON-HNeRV coord-INR (~83K, ~61s/ep). The d_seg-CONDITIONING subset of PR95 ONLY — drop
all rate machinery (~14,500 ep) + the smooth stage.

| stage | setting (OPTIMAL) | measured d_seg dir | cite |
|---|---|---|---|
| **S0 seed** (FREE, 0 archive bytes) | `--structured-init` + `--lane-prior-phi1` (replace) | seed = Road↔Lane separatrix, residual **1.9e-5** | FEED-fs; deepmath Lens B/E |
| **S1 CE** (~200-400 ep; `ce_to≈300`) | full CE, high LR, EMA off/low | 0.01045→0.00643 (↓) | seg-evolution; MLX-port n600 DAG :532 |
| **S2 tau_softplus** [REHEAT] (~400-700 ep) | **`--tau-softplus-tau 0.3`**, mid LR, EMA 0.997 | →**0.00396 (THE primary drop, min)** | seg-evolution; anneal-memo (τ=0.3 = Δ_min reachability floor) |
| **S3 l7_softplus + margin engage** [REHEAT] (~300-600 ep) | l7 5× weight on margin<1.0 (l7_mult=4), renorm mean-1, wt under stop_grad, low LR | →0.00369 (slow new min) | seg-evolution; FEED-ln |
| **S4 Muon finisher** [REHEAT] (~200-400 ep) | **muon-lr 2e-3** (NOT 0.03), tau + render-temp FROZEN 0.05, reset-moments, muon-lr-floor-fix ON | **THE conditioning drop** (witness value PREDICTED ~6-9e-4, not yet measured) | deepmath Lens A; FEED-fi/fk; MUON_BITES_FROM_STAGE4 |
| **SKIP** | smooth (RAISES d_seg +6.8%), QAT/C1a/λ/σ (rate machinery) | smooth →0.00423 (↑) | seg-evolution; the witness rate story is byte-close of a tiny payload |

**Cross-stage knobs (OPTIMAL):**
- **softmax-temp (RENDER partition)** 1.0→0.05 throughout; **frozen 0.05 for the Muon finisher** (placement vs a stationary partition). This is a SEPARATE temp from `--tau-softplus-tau` (§4). FEED-fm.
- **REHEAT at EVERY stage transition** (the "different stages need different treatment" non-negotiable): `--stage-transition-rewarmup-epochs 8` (LR floor 0.1×→1×) + `--stage-transition-reset-moments` (zero stale AdamW 2nd-moments). MEASURED: floor 0.1×/8ep stable (n_skips=0); FULL restart 1.0× reproduces v3 destabilization → **PARTIAL restart, not full SGDR.** FEED-bu/fz.
- **EMA decay 0.997**, save the **SHADOW** (not live) as the inference ckpt, apply ONLY at eval with snapshot+restore (CLAUDE.md EMA non-negotiable). EMA-shadow-lag: 0.997 lags fast single-frame descent up to 78× (the "0.505 wall" was an export artifact, not capacity).
- **Stabilizers (NCA stack):** `--grad-clip 1.0` + `--spike-factor 5.0` (5×-median spike-guard) + per-boundary spike-guard RESET at each transition; **n_restarts≥2 keep-best expressed at the CAMPAIGN level** (ROLLBACK_BRANCH to best ckpt), NOT a trainer flag.
- **Adaptive stacking (#188):** `campaign.decide_next_stage` (PURE) → EXTEND (slope ≤ -1e-5, still descending, resume final) / ADVANCE (|slope| < 1e-6 plateau, stack next + reheat) / RERUN_NEW_CONFIG / ROLLBACK_BRANCH (final − best > 1e-5 → resume BEST + skip). window 300. EMIT-ONLY (containment). DETERMINISTIC.
- **Resumability (non-negotiable):** single recorded `--seed`, per-stage + ≤25-ep checkpoints, EMA-shadow, `--resume-from`-compatible.
- **DM1 (Stiefel-W + code-spectral-entropy):** byte-FREE conditioning lever; **DEMOTED to second-order** (PR collapses WHILE d_seg improves → not the binding d_seg cause). COMPOSE adaptively only if FiLM rank-collapse becomes binding; NOT in the opening.

### 0c. PARALLEL / ALTERNATIVE arms (NOT the from-scratch witness path; record so not lost)
- **Rendered-pose vehicle** (if a frame IS pose-scored): amortized-INR saliency-confined carrier (~22.5 KB, d_pose~0.006, closes palette wall ~900-2200×) + equimarginal pose-weight (at frontier ∂S/∂d_pose ≈ 86% of ∂S/∂d_seg) + pose-throttle every-k (reclaims ~51% epoch wall-clock, risks drift). NOT needed when `--w-pose 0`.
- **MD-Decoupling:** PARALLEL ablation arm only (stability real but under-steps d_seg at scale) — NOT the decisive run.
- **jump-to-Muon-early:** candidate (MUON_BITES_FROM_STAGE4 supports it); arbiter = the real stage-8 slope.

---

## 1. INDEX TABLE — POSE axis (deduplicated)

| # | finding | status | calibration | pointer |
|---|---|---|---|---|
| P1 | STORED-TARGET sidecar: store 6 PoseNet scalars/pair, d_pose≈0, 7,200B raw/<5KB zlib — the canonical pose SOLVE | GROUNDED, deployed | contest-fact | `src/tac/scorer_targets.py`; DAG :612 |
| P2 | Pose FROZEN at inference (trainable stored pose drifts off the exact target) | GROUNDED | quantizr-source | `quantizr_pose_implementation_audit_20260611T021200Z.md` §2 D5 |
| P3 | `--w-pose 0` in the witness: pose rides the sidecar, witness's only job = d_seg | GROUNDED, deployed | deepmath | curriculum-deepmath; CURRENT-STATE memory |
| P4 | low-rank pose codec #140 = **rank-4/511 Pareto-dominant** (2,563B, MSE 2.7e-5, −0.0004 rate); naive rank-2/254 net-NEGATIVE | MEASURED torch-CPU advisory | `[contest-CPU advisory]` | `lowrank_pose_section_codec_landed_20260617.md` + `pose_lowrank_CORRECTED_fidelity_20260617.json` |
| P5 | pose_from_embedding MLP ~1-2KB (replaces optimized_poses.pt ~15KB) | PARTIAL | advisory | `src/tac/pose_from_embedding.py`; DAG :612 |
| P6 | "pose collapse" (d_pose 0.06-0.34) = content-free-latent RENDERED carrier (no geometry to move), NOT the stored sidecar | GROUNDED (reconcile) | quantizr-source | quantizr audit §3 (D3 dominant) |
| P7 | amortized-INR pose CARRIER (saliency-confined PTNC): closes palette wall 12.658→~0.006 @ ~22.5KB; saliency cheaper at convergence 1.16-2.44× but slower; #57 overflow REFUTED | MEASURED contest-CPU advisory | `[contest-CPU advisory]` | `witness_L13_optimal_pose_carrier_result_20260621.md` |
| P8 | warp-carries-pose REFUTED for lossy: d_pose 190→12.6 only at d_pose-optimal cal which WRECKS d_seg (opposite homography scales) → pose stays on sidecar | MEASURED through-R advisory | `[CPU-torch advisory]` | FEED-lj (DAG :6807) |
| P9 | grok pose-warp: pose = FREE d_seg modulation for **Road** via STRATIFIED per-class warp (+15-17%; MyCar=identity, sky=rot-only) — dual-use d_pose+d_seg for the v2 DETERMINISTIC vehicle | MEASURED pre-R advisory | `[macOS advisory]` | `grok_pose_warp_dseg_test_20260629T181000Z.md` (FEED-ja) |
| P10 | pose operating point: at frontier (d_pose~3.4e-5) ∂S/∂d_pose ≈ 86% of ∂S/∂d_seg → equimarginal pose-weight matters for a RENDERED-pose vehicle | GROUNDED math | advisory | `sophisticated_pose_treatment_design_20260616T222900Z.md` |
| P11 | pose DESCENDS for free with training (ep50 0.0072→ep488 0.0002) → ruled out as a binding lever when rendered | MEASURED MLX | `[macOS-MLX]` | DAG :541 |
| P12 | adaptive pose-gradient controller / pose-throttle every-k reclaims ~51% epoch wall-clock, risks drift | BUILT default-OFF | advisory | `adaptive_pose_gradient_controller_20260616T194232Z.md` |
| P13 | FiLM-on-moving-frame-only (frame1), conv-residual-block injection, AdamW (NOT Muon) — Quantizr's d_pose 0.00051 @ 88K | GROUNDED quantizr-source | quantizr-source | quantizr audit §1 |

## 1b. INDEX TABLE — CURRICULUM / SCHEDULING / OPTIMIZER / DYNAMICS (deduplicated)

| # | finding | status | calibration | pointer |
|---|---|---|---|---|
| C1 | Witness SHORT curriculum S0 seed→S1 CE→S2 tau_softplus(0.3)→S3 l7→S4 Muon; SKIP smooth+QAT/C1a/λ/σ; ~1100-2100 ep vs 29650 | GROUNDED, deploy design | deepmath + MLX-port | `curriculum_openpilot_seeded_deepmath_dsl_20260629T034000Z.md`; `pr95_segnet_stage_evolution...20260626.md` |
| C2 | per-stage measured d_seg dirs: CE↓ 0.0104→0.0064 · tau_softplus →0.00396 (THE drop) · smooth →0.00423 (↑) · l7 →0.00369 · Muon=THE drop | MEASURED MLX-port n600 | `[macOS-MLX]` | DAG :532/:538/:637 |
| C3 | Muon = the finisher (spectral conditioner of ill-conditioned boundary-annulus valley; AdamW diagonal can't fix off-diagonal) | GROUNDED | deepmath | curriculum-deepmath Lens A; FEED-fk |
| C4 | **MUON_BITES_FROM_STAGE4**: Muon descends d_seg ~32% MORE than AdamW (gap −0.000340, widens monotone); AdamW grad-norm collapses on κ~19 Hessian → jump-to-Muon-early viable | MEASURED contest-CPU advisory | `[contest-CPU advisory]` | `muon_vs_adamw_from_stage4_convergence_arm_20260622.md` |
| C5 | **muon-lr = 2e-3** for the witness flat finisher (band 1e-3..2e-3, ceiling 5e-3); NOT 0.03 (6× too hot) | GROUNDED (witness) | deepmath measured band | curriculum-deepmath Lens A; FEED-fi/fl |
| C6 | muon-lr-floor-fix: Muon needs its OWN floor ratio (lr_floor_ratio/muon_lr), else never anneals to fine-polish | GROUNDED provenance | `[contest-CPU advisory]` | `decisive_run_161_muon_lr_floor_fix_resume_20260622.md` |
| C7 | τ is TWO temps: (a) `--tau-softplus-tau`=0.3 = SEG-SURROGATE = reachability floor Δ_min≈0.3; (b) softmax-temp 1.0→0.05 = RENDER anneal, frozen 0.05 for Muon | GROUNDED | deepmath + anneal | curriculum-deepmath Lens C; `anneal_optimal...20260613.md` |
| C8 | anneal-optimal: surrogate grad ∝(1/T)e^{-Δ/T}, peak T*=Δ; distribution-optimal = e^{-Δ/T}-WEIGHTED-MEAN fixed point (NOT median), clamp[0.3,0.6], Lever-5 τ slaved=T | MEASURED (median falsified slice-0) | `[contest-CPU advisory]` | `anneal_optimal...20260613.md` §6′ |
| C9 | REHEAT = stage-transition re-treatment: rewarmup floor 0.1×/8ep + reset-moments; PARTIAL restart (1.0× full restart re-destabilizes) | MEASURED | `[macOS-MLX]` | FEED-bu/fz; reheat design |
| C10 | margin-engage destabilization: margin stage at ep80 inherited base treatment (stale spike-guard ~8, base LR, temp cliff 0.1) → gnorm 648-772 → all batches skipped; FIX = anneal-in + re-warmup + recalibrate spike-guard + reset momentum | GROUNDED incident | measured | `[[different-stages-need-different-treatment...]]` |
| C11 | EMA 0.997, save SHADOW not live, apply at eval w/ snapshot+restore; EMA-shadow-lag up to 78× (the "0.505 wall" was an export artifact) | GROUNDED | CLAUDE.md non-neg | EMA non-negotiable; `[[capstone-ema-shadow-lag...]]` |
| C12 | NCA stabilizers: grad-clip 1.0 + spike-factor 5.0 (5×-median) + per-boundary reset; n_restarts≥2 keep-best at CAMPAIGN level (not a flag) | GROUNDED (trainer-gap table) | deepmath | curriculum-deepmath §3 |
| C13 | l7_softplus = margin-weight allocation lever (5× on margin<1.0); STAGE-conditioned (from-scratch STARVES interior; finetune re-allocates to annulus) | GROUNDED | measured | seg-evolution; `[[different-stages...]]` |
| C14 | structured-init S0 seed: `--structured-init` (static-core SDFs, SELF-DETECT roles) + `--lane-prior-phi1` (openpilot deg-3 centerline SDF, FREE 0 bytes); seed gives low-freq FREE → jump to high-freq annulus (NTK) | GROUNDED | FEED-fs | curriculum-deepmath Lens B/E |
| C15 | MD-Decoupling: WIRED in `train_witness_realized_through_R_mlx.py` (`--optimizer md`), NOT the level-set trainer; stability real (gnorm ≤376 vs AdamW 10869) but UNDER-STEPS d_seg → PARALLEL arm only | MEASURED CPU smoke | `[CPU-torch advisory]` | `md_decoupling_wirein_validation_cpu_smoke_20260627.md` |
| C16 | adaptive stacking #188: PURE `decide_next_stage` EXTEND/ADVANCE/RERUN/ROLLBACK; thresholds slope 1e-6/-1e-5/1e-5, window 300; emit-only, deterministic | BUILT | code | `src/tac/witness_dsl/campaign.py` |
| C17 | per-stage optimizer norm assignment (fractal): CE=Adam → tau/l7=Muon+SinkGD(FiLM-W) → muon_tail=Muon; MINIMAL = Stiefel-W(no-WD)+code-spectral-entropy+moment-reset+Muon-prime | DESIGN (MY-DESIGN, unproven) | design | `per_stage_fractal_optimizer_priming_reheat_anneal_design_20260629.md` |
| C18 | DM1 Stiefel-W+code-spectral-entropy: byte-free, PR(M)=PR(cov code) when WᵀW=I; DEMOTED 2nd-order (PR collapses WHILE d_seg improves → not the binding cause) | GROUNDED-but-demoted | deepmath | curriculum-deepmath Lens A; GR memo |
| C19 | PR95 8-stage forensic: 29650 ep = 3000 CE + 5650 tau + 1500 smooth + 500 QAT + 9000 C1a-L7 + 2000 λ + 3000 σ + 5000 Muon; seg_wt 100, pose_wt 1, pose √(10·MSE) const | GROUNDED forensic | intake | `pr95_8stage_curriculum_forensic_20260513.md` |
| C20 | DSL `curriculum_dsl.openpilot_seeded_opening` validates (==[]); never-invent (validate() refuses any flag not in real argparse); 57 tests green | BUILT | code | `src/tac/witness_dsl/curriculum_dsl.py` |

---

## 2. OPEN / HEADROOM (the named next-probes, NOT re-derive)

1. **Muon witness d_seg value NOT yet measured** (PREDICTED ~6-9e-4; the from-scratch run is what measures it). C2/C5.
2. **adaptive-τ (weighted-mean fixed point clamped [0.3,0.6])** NOT wired in the level-set trainer (trainer-gap; `--tau-softplus-tau` is a fixed scalar) — the named anneal headroom over fixed 0.3; #119 calibration. C8.
3. **curvelet multi-scale coarse→fine climb** (DM2 oriented basis, the −48% byte-free d_seg lever, Candès-Donoho cartoon-optimal): seed gives coarse low-freq FREE → climb finer curvelet bands on the boundary annulus → toward ~0.001. Research: `levelset_curvelet_witness_feasibility_20260627.md`.
4. **DM1 causation** (does fixing PR move d_seg?) — the per-stage-fractal §6 firewall smoke (A0/A1/A2/A3, PR-hold + d_seg-lift thresholds). C18.
5. **MD-Decoupling at scale (n96/n600):** needs own lr sweep + relaxed/disabled divergence detector + longer budget; promote only on a byte-closed exact row beating the AdamW manual-fix arm. C15.
6. **jump-to-Muon-early on the live run** (candidate from MUON_BITES_FROM_STAGE4; arbiter = the real stage-8 Muon slope). C4.
7. **saliency-confined pose-carrier edge at full-N/Muon scale** (advisory N≤12; does the 1.16-2.44× hold/grow?). P7.
8. **pose_from_embedding MLP further compression** (~1-2KB, PARTIAL #140 sister). P5.
9. **SinkGD-on-FiLM-W** (SPECULATION, 2nd-order to the structural cure; only if fixed SGDR + Stiefel insufficient). C17.

---

## 3. CONFLICTS / SUPERSEDED (resolved — do not re-litigate)

| topic | the conflict | RESOLUTION |
|---|---|---|
| **muon-lr** | 0.03 (recalled, UNGROUNDED) vs 2e-3 (witness deepmath, GROUNDED) vs 2e-4 (PR95 vendored) vs 3e-3 (muon-vs-adamw "higher") | **witness finisher = 2e-3** (FEED-fi band). 0.03 = from-scratch nanogpt convention (wrong regime). 2e-4 = PR95's 229K HNeRV (different model). 3e-3 = used only to surface the A/B contrast faster. C5. |
| **pose-collapse vs pose-solved** | "pose collapse d_pose 0.06-0.34" vs "pose SOLVED d_pose≈0" | both true, DIFFERENT mechanisms: collapse = content-free-latent RENDERED carrier (no geometry to move, D3); solved = STORED sidecar. The witness uses the sidecar (`--w-pose 0`). P6. |
| **low-rank pose codec ratio** | "rank-2 2.7× smaller" (the reopened claim) vs rank-4/511 | rank-2/254 is **net-NEGATIVE** (fidelity cost on √(10·d_pose) > byte save); the Pareto-dominant DEFAULT is **rank-4/511** (smaller AND lower MSE). P4 supersedes the old 2.7× line. |
| **MD-available vs trainer-gap** | "MD wired" vs "MD trainer-gap" | both true: WIRED in `train_witness_realized_through_R_mlx.py` (`--optimizer md`), trainer-GAP in the level-set trainer (no `--optimizer` flag). Use as a PARALLEL arm in the through-R trainer; the level-set reheat (rewarmup+reset-moments) IS the stable-transition mechanism there. C15/C9. |
| **τ single vs two** | "softmax-temp 1.0→0.05" appeared to contradict the anneal-memo's "miscalibrated cosine" | NOT a contradiction: TWO temps — seg-surrogate `--tau-softplus-tau`=0.3 (load-bearing, = Δ_min) vs render softmax-temp 1.0→0.05 (separate, frozen 0.05 for Muon). C7. |
| **smooth stage** | PR95 has it (1500 ep) | DROP for the witness — measured to RAISE d_seg +6.8% (transient); not a flag in the level-set `--curriculum`. C1/C2. |
| **warp dual-use** | grok "free dual-use warp" (P9) vs FEED-lj "REFUTED for lossy" (P8) | reconciled: warp is free dual-use for the v2 DETERMINISTIC vehicle (Road d_seg + d_pose), but for a LOSSY render d_seg & d_pose want OPPOSITE homography scales → pose stays on the stored sidecar; the warp's role = the v2 residual predictor, NOT a pose carrier. |

---

## 4. NO-FAKE ledger
- MEASURED rows cite the exact JSON/file/FEED + axis tag; the strongest optimizer anchor is C4
  (MUON_BITES_FROM_STAGE4, contest-CPU advisory, 6.8× the discrimination band) and C2 (per-stage d_seg, MLX-port).
- DESIGN/UNGROUNDED/SPECULATION explicitly labelled (C17 MY-DESIGN; muon-lr 0.03 UNGROUNDED; SinkGD SPECULATION).
- NOT claimed: no score moved; **pointer UNMOVED 0.19110**; this is a MEANS (marshaling for the launch), not a row.
- The optimal-config is a DESIGN recommendation; the verdict is the byte-closed n600 `upstream/evaluate.py`
  row at the composed θ* (opening + adaptively-stacked l7 + Muon). MEANS≠ENDS.
