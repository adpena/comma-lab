# Frontier pointer-move ledger (the exact-score scoreboard) — 2026-06-10

The single durable record of EXACT frontier-pointer moves. One row per attempt that reaches the
adjudication layer (`tac.optimization.scorer_quotient_candidate_row`). Only a row with
`pointer_update_eligible=True` (contest-tier `exact_evaluate`, recomputed ΔS<0) actually moves
`.omx/state/canonical_frontier_pointer.json`. Advisory/proxy rows are recorded for prioritization but
NEVER promote (the sub-0.15 firewall + "Frontier scores are pointer-only"). Lead every session report
with the latest pointer + whether it moved.

## Current exact pointer
| axis | score | archive sha | bytes | as-of |
|---|---|---|---|---|
| contest-CPU | **0.19109982** | `b46897267ded…` | 177,169 | 2026-06-10 (recoded-R3, defensive bank) |
| contest-CUDA | 0.20533003 | `9cb989cef519…` | 186,876 | 2026-05-16 (pr106) |

## Moves
| # | lever | candidate_kind | authority | ΔS | new pointer | innovation | decision |
|---|---|---|---|---|---|---|---|
| 0 | recoded-R3 (baseline) | requant/recode | contest-CPU + CUDA | (baseline) | 0.19109982 | defensive_bank (R1+R2 borrowed PR#112; fails Innovation Gate) | banked, submission-blocked on `constriction` allowlist |
| 1 | #64 lossless stack | — | — | **0.0** | unmoved | n/a | NO-OP: R1+R2+R3 already in base, S12 inapplicable to procedural carrier (NO-FAKE refused a no-op masquerade) |
| 2 | #72 lever-D margin-conditional residual | margin_residual | exact_cpu_advisory / exact_pair_scorer | **0.0** | unmoved | reusable rate-win | DEFER: RATE side WINS (0.856 B/flip < 1.27, overturns #51's 1.525 unconditional floor) but DISTORTION side DIES on receptive-field collateral (467 fixed / 2823 new-bad, net −2356; waterfill admits 0). True floor = collateral, not rate. Reactivate on contiguous-residual base (lever C / #73). |
| 3 | #73 Dykstra legal-frame feasibility | dykstra_feasible_frame | exact_cpu_advisory / exact_pair_scorer | **0.0** | unmoved | sharp geometric finding | scorer_effect (holds BOTH terms at frontier: d_seg 0.00057, d_pose 2.40e-5 in-tube, 4/4) but cheap-feasible set EMPTY at low byte (≥625KB/pair generic basis; <400KB the pose tube breaks). PROVES feasibility≠generation + the 177KB learned HNeRV basis IS the cheap-feasible representation. Reactivation = Dykstra with C=learned manifold = subsumed by #71. |

| 4 | #54 cross-pair pose corrector | cross_pair_corrector | exact_cpu_advisory / exact_pair_scorer | **0.0** | unmoved | paradigm proven, lever saturated | DEFER: cross-pair global-pool waterfilling PROVEN correct, but the frontier's FEC6 K=16 frame-0 selector is already per-pair POSE-OPTIMAL (0/42 improvable; constant-correction control +1.27e-3 worse → waterfiller load-bearing). Pose analog of #55. Region-allocator READY for a contiguous-residual lever-C base. |

| 5 | #71 Q* structural compression | structural_compression | exact_cpu_advisory / exact_pair_scorer | **0.0** | unmoved | original method (exact codec-grammar byte re-encoder + per-tensor exact-scorer cost/kB ablation + 0.000666/kB feasibility threshold) | DEFER: RATE side is REAL+LARGE (magnitude prune keep0.7 −20,741 B = sub-0.18, keep0.3 −68,657 B = sub-0.15 by rate alone) but DISTORTION DIES at 70–370× the feasibility threshold. SVD/low-rank INCREASES bytes +9–14% (dense weights r95≈78% full rank; break-even rank > r95; re-quant breaks brotli structure). Per-tensor exact-scorer ablation: cheapest tensor (blocks.1) +0.0485 score/kB vs the 0.000666/kB net-negative threshold = 73× over. Score-aware Taylor prune (|g·w|, diff pose+seg-KL) is WORSE than magnitude (keep0.6 ΔS +1.423 vs +1.309) — the learned weights are JOINTLY ENTANGLED, no score-irrelevant sparse subset exists. PROVES the 162 KB learned basis is at its DISTORTION-HOLDING FLOOR (minimal for the SCORE, not just a memorized point). Reactivation = score-domain RETRAINED smaller renderer (funded KD/QAT campaign) — the only path that relocates the floor; post-hoc compression of frozen weights is closed. |

| 6 | #74 distill to smaller learned student | structural_compression (distill) | exact_cpu_advisory / exact_pair_scorer | **0.0** | unmoved | original method (recon-primary KD onto the teacher's decoded frames + measured PoseNet pose-tube width curve) | DEFER: directly tests #71's reactivation (the "score-domain RETRAINED smaller renderer"). The teacher-frame KD BREAKS the #62 d_seg wall DIRECTIONALLY (80kb 2-pair student exact d_seg 0.2517 = HALF the constant floor 0.5069, where #62 argmax-CE-on-GT pinned AT the floor) — training on the teacher's already-argmax-correct frames gives the gradient #62 lacked. BUT exact d_pose = 189 (8M× the teacher's 2.3e-5, WORSE than the constant control 26): the proxy pose-MSE-to-teacher converged to 0.0005 yet the exact d_pose vs GT is catastrophic. ROOT CAUSE (measured pose-tube-width probe): to hold d_pose≈2.9e-5 the student must reproduce the teacher's frames to RMSE<3 (per-pixel error <±5/255); RMSE 5.8→11×, RMSE 23→780×. A small student decoding 384×512 reaches only RMSE~40-50 = far outside the tube = CAPACITY wall, not a training issue. Convergent with #62/#73/#71: the 177KB learned HNeRV basis is the cheap-tube-holding representation. Reactivation: (1) pose-frame decoupling — distill ONLY frame1 small + carry frame0 as a near-lossless residual (frame0 is SegNet-invisible, pure pose carrier, dominant failure: studF1+teachF0 d_pose 17 vs both 189); (2) PoseNet-FEATURE distill (match teacher FastViT activations, not pose-blind pixels); (3) #71 (compress the proven-tube-holding teacher) DOMINATES #74 (re-learn a basis that loses the tube) at this operating point. |

| 7 | #69 score-aware whole-tensor re-quant | requant | exact_cpu_advisory / exact_evaluate (600-pair) | **0.0** | unmoved | original (per-tensor frozen-scorer sensitivity rank, 120× spread) | NEGATIVE/DEFER: gentlest crush (4 tensors→6-bit, −18,868 B) costs Δscore +0.16020 = 13.8× the rate saving. The 6 cascaded PixelShuffle convs COMPOUND per-weight quant error multiplicatively → 6-bit flips argmax+pose. Frontier int8 weights ALREADY at the score-relevant floor — NO recon-only scorer-slack post-hoc. Reactivation: per-CHANNEL sensitivity OR score-aware QAT (re-quant without retrain can't relocate the decision boundary → #78 must be QAT-in-loop). |

## Convergent meta-finding (7 no-moves)
#64 (lossless exhausted) + #69 (whole-tensor re-quant: int8 already at the score floor, 6-conv cascade compounds quant error) + #72 (residual codes cheap @0.856 B/flip but collateral kills application) + #73
(generic-basis feasibility needs ≥625KB) + #71 (the LEARNED basis itself cannot be pruned/factored ~73× cheaper
than the rate it buys; SVD even costs bytes) + #54 (pose-selector saturation) + #74 (a fresh SMALLER learned
student distilled onto teacher frames breaks the #62 d_seg wall directionally but CANNOT hold the PoseNet tube —
the tube needs frame-RMSE<3 / ±5-per-255, a near-frontier-capacity wall) all confirm from SIX directions:
**the 162–177 KB learned HNeRV nonlinear basis IS the cheap-feasible representation for holding pose+seg
simultaneously, and it is at its DISTORTION-HOLDING FLOOR.** It is minimal for the SCORE — not an
over-parameterized memorized point. No post-hoc operation on the frozen frontier weights lowers the rate term
without an equal-or-larger distortion penalty; AND a fresh smaller architecture trained to mimic the teacher
loses the pose tube the teacher's full capacity holds. The remaining sub-frontier paths are: (a) SCORE-DOMAIN
RETRAINING that relocates the floor (distill/QAT against `α·B + β·d_seg + γ·√d_pose` directly, NOT
recon-to-teacher — see #75's harness-bug root cause below); (b) pose-frame decoupling (#74 reactivation: distill
frame1 small + carry frame0 as a near-lossless pose residual); (c) PoseNet-FEATURE distill (not pose-blind
pixels). All are funded long-training campaigns, not $0–$1 transforms. **#74 sharpens #71's reactivation: the
binding constraint is not the distillation but the PoseNet tube width — even a perfectly-trained smaller mimic
fails it; the score-domain Lagrangian (which spends capacity on the tube-relevant pixels) is the only retrain
that can both shrink bytes and hold the tube.**

## ROOT CAUSE PINPOINTED (#75 elephant + #68 fleet audit, 2026-06-10)
The whole d_seg≈0.50–0.71 plateau across ALL 30+ NeRV substrates (our killed/deferred fleet, the
0.196-0.199 cluster) is **ONE shared two-part bug in the shared MLX harness, NOT 30+ paradigm walls**:
(M-loss) `_shared/mlx_score_aware/bundle.py` defaults every SegNet/PoseNet objective weight to 0.0 →
"score-aware" runs were silently recon-MSE-only / scorer-blind (the #75 inert loop, located); (M-arch)
the default decoder is skip-free PixelShuffle+sin, missing PR95's bilinear-skip+HF-refine → mean-field
blur → argmax collapse (proof: can't memorize even ONE pair; grad-norm 5.6e9→8e-5 ill-conditioned). Our
eval is CORRECT (reproduced PR95's 0.19871 bit-exact). Per Catalog #307 these are IMPLEMENTATION
falsifications → **the deferred fleet REACTIVATES once #76 lands the working loop (both fixes).** The
capstone (#78) arch is specified: E-NeRV + bilinear-skip+HF-refine + FFNeRV-flow + fixed-codebook-VQ →
~42-74KB → sub-0.15. Critical path: #76 (loop) + #77 (optimizer) → #78 (capstone). #68 deliverable: commit `89e8829c6`.

## Pending movers (will append a row on landing, via the schema firewall)
- #69 score-aware Q* re-quant (rate) · ~~#71 Q* structural compression~~ (CLOSED post-hoc; reopens only as score-domain RETRAINING)
- #72 lever-D margin-conditional residual coder (d_seg) · #54 cross-pair waterfilled corrector (pose)
- #73 legal-frame Dykstra feasibility (realization) · #63 d_seg-loss hinge (gates the lever-C campaign)
- **NEW (the convergent next step): score-domain RETRAINED smaller renderer** — the only lever that moves the distortion floor #64/#71/#72/#73 all hit.

## Innovation-status note (per the Innovation Gate)
The current 0.19109982 frontier is a **defensive bank** (`defensive_bank=true`, `class_shift=false`,
`borrowed_substrate=true`, `submission_recommendation=hold_not_final`) — banked for readiness, NOT the
innovative submission. The original sub-0.15 submission must come from a class-shift mover (#73
feasibility / #63→lever-C / #71 structural) with `class_shift=true`.
