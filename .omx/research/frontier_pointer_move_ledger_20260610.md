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

| 6 | #74 distill to smaller learned student | structural_compression (distill) | exact_cpu_advisory / exact_pair_scorer | **0.0** | unmoved | original method (recon-primary KD onto the teacher's decoded frames + measured PoseNet pose-tube width curve) | CONTINUE / PROMISING (tests #71's reactivation "score-domain RETRAINED smaller renderer"). The recon-primary teacher-frame KD BREAKS the #62 argmax-CE-on-GT d_seg wall. **8-pair sweep (representative; supersedes a noisy 2-pair anchor): 40kb student → exact d_seg 3.44e-3 (6.4× teacher), d_pose 2.43e-3 (105× teacher), S 0.530, parity 1.0** — within ~1 order on seg, ~100× on pose, a DESCENDING RD curve NOT a wall. (The initial 80kb 2-pair anchor d_seg 0.25 / d_pose 189 was an unstable overfit outlier of the tiny tube + warm schedule — NOT representative.) **The curve is NON-MONOTONE: 60kb got WORSE (d_pose 0.0024→1.44, S 0.530→4.73) = the #57/#62 capacity-instability re-fires** — fixed-LR/fixed-schedule KD does NOT scale up; 40kb (smallest, most stable) is best. Mechanism bound (measured pose-tube-width probe): fully in-tube needs d_pose→2.9e-5 ⇒ frame-RMSE<3, so seg (within 1 order) descends faster than pose. NEXT GATE = funded STABILIZED long-train (per-size LR/grad-clip/EMA tuning + more epochs + full 600-pair + score-domain Lagrangian α·B+β·d_seg+γ·√d_pose) to keep bigger students in-basin + measure the d_pose asymptote — a LIVE campaign per "Long-burn score-lowering campaign default", not a DEFER. Accelerators: (1) pose-frame decoupling (distill frame1 small + frame0 near-lossless residual; frame0 SegNet-invisible pure pose carrier); (2) PoseNet-FEATURE distill (match teacher FastViT activations not pose-blind pixels); (3) composes with #71+#69. |

| 7 | #69 score-aware whole-tensor re-quant | requant | exact_cpu_advisory / exact_evaluate (600-pair) | **0.0** | unmoved | original (per-tensor frozen-scorer sensitivity rank, 120× spread) | NEGATIVE/DEFER: gentlest crush (4 tensors→6-bit, −18,868 B) costs Δscore +0.16020 = 13.8× the rate saving. The 6 cascaded PixelShuffle convs COMPOUND per-weight quant error multiplicatively → 6-bit flips argmax+pose. Frontier int8 weights ALREADY at the score-relevant floor — NO recon-only scorer-slack post-hoc. Reactivation: per-CHANNEL sensitivity OR score-aware QAT (re-quant without retrain can't relocate the decision boundary → #78 must be QAT-in-loop). |
| 8 | #79 archive-packaging + bit-packing | — (no candidate) | exact (evaluate.py source + on-disk bytes) | **0.0** | unmoved | n/a (packaging lever) | NEGATIVE / LEVER-CLOSED: `evaluate.py:63` counts ONLY `archive.zip`'s `.stat().st_size` (scripts NOT counted). The frontier zip is already at the container floor (single 1-char member `"x"`, STORED, 100 B overhead = 31+47+22, 0 padding) AND the 177,069 B payload is at the entropy floor (order-0 = 7.9990 bits/byte; zlib +61 / lzma +67 / brotli q11 +5 — every general coder makes it LARGER). Sub-byte bit-packing beats byte-alignment ONLY on non-entropy-coded streams; recoded-R3 already entropy-coded it. ZERO recoverable bytes ⟹ no `scorer_quotient_candidate_row` emitted. The rate term only falls with a SMALLER PAYLOAD (#78). Reactivation: re-run the entropy/lossless-recompression sweep on the capstone's FIRST byte-closed archive (pre-entropy-pass) — residual compressibility there IS free rate. Audit: `archive_packaging_byte_audit_20260610T224611Z.md`. |

| 9 | #52 lever-G engineered 0-byte correction | — (no candidate) | local CPU-torch advisory / exact_pair_scorer | **0.0** | unmoved | n/a (zero-byte distortion lever) | NEGATIVE / SUBCLASS-CLOSED: the GLOBAL-fixed-rule subclass (G1 per-channel offset, G3 low-pass blend) selects the IDENTITY on train (best_b=[0,0,0], α=0) — NO rule cuts held-out d_seg subject to the Δd_pose≤+1e-6 pose guard. Crux: bidirectional-symmetric boundary flips cancel under any global pixel op + the frontier ALREADY applies PR95-L28 channel postproc (at the global optimum). Zero archive bytes; pointer UNMOVED. DEFER not KILL: reactivation = a rule SPATIALLY conditioned on the renderer's OWN edge map (inflate-legal, no scorer), else route firepower to lever C (smaller amortizer). Measured: `lever_g_engineered_correction_smoke_20260610T095654Z.md` + `lever_g_result.json`. |

## Convergent meta-finding (8 post-hoc no-moves + #74 the live retrain exception)
#64 (lossless exhausted) + #69 (whole-tensor re-quant: int8 already at the score floor, 6-conv cascade compounds quant error) + #72 (residual codes cheap @0.856 B/flip but collateral kills application) + #73
(generic-basis feasibility needs ≥625KB) + #71 (the LEARNED basis itself cannot be pruned/factored ~73× cheaper
than the rate it buys; SVD even costs bytes) + #54 (pose-selector saturation) + #79 (PACKAGING: archive.zip at
the 100 B container floor + payload at the 7.999 bits/byte entropy floor — zero recoverable bytes) + #52 lever-G
(zero-byte global decode-rule: G1 offset + G3 lowpass select the IDENTITY; bidirectional-symmetric boundary
flips cancel + the L28 channel postproc is already applied) confirm from EIGHT directions that
**no $0–$1 POST-HOC operation on the frozen 162–177 KB frontier weights/bytes/decode lowers the score without an
equal-or-larger penalty** — the FROZEN carrier is at its post-hoc floor on rate AND distortion (minimal for the
SCORE, not an over-parameterized memorized point). **#74 is the EXCEPTION that opens the door: a fresh SMALLER
learned student distilled onto the teacher's frames is NOT post-hoc — it RELOCATES the floor.** The 8-pair sweep
shows a 40kb student reaching exact d_seg 3.44e-3 (6.4× teacher) / d_pose 2.43e-3 (105× teacher), S 0.530, a
DESCENDING RD curve — proving the distillation premise works and the #62 argmax-CE wall is broken (the noisy
2-pair anchor's d_seg 0.25/d_pose 189 was an unstable overfit outlier, NOT representative). The remaining
sub-frontier paths (now all under #74's umbrella) are FUNDED LONG-TRAIN campaigns, not transforms:
(a) score-domain Lagrangian retrain (`α·B + β·d_seg + γ·√d_pose` directly — spends capacity on the
tube-relevant pixels; see #75's harness-bug root cause below for why prior "score-aware" runs were inert);
(b) pose-frame decoupling (distill frame1 small + carry frame0 as a near-lossless pose residual); (c)
PoseNet-FEATURE distill (not pose-blind pixels). **Convergent conclusion: post-hoc compression of the frozen
basis is CLOSED (#64/#69/#71/#72/#73/#54), but score-domain RETRAINING of a smaller basis (#74) is OPEN and
PROMISING — the smaller learned student gets within an order of magnitude of the teacher's d_seg and the curve
is descending; the open question is the funded-long-train asymptote on the pose term.**

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

## ★ UNLOCK (#76, 2026-06-10) — the inert loop is FIXED; d_seg descends
NOT a pointer move (no exact-eval candidate yet) but the PREREQUISITE that unblocks the entire retraining
campaign. With the fixed loop (`tac.score_aware_loop`: direct CE through the LIVE frozen SegNet + GT-argmax
targets + eval-roundtrip STE + simple 100·seg+1·pose + AdamW + EMA), the live-render exact d_seg DESCENDS
**0.508 → 0.081 (84%)** in 8 CE epochs on real pairs — the 3000-epoch mean-field wall DISSOLVES. Root cause
(confirmed): (a) the harness used a learnable-student-head KL surrogate (NOT direct CE) because the ported
MLX SegNet NaN's in 2nd-order autograd; (b) a 6-term Lagrangian where pose dwarfed seg 170× → grad clipped
to noise. The operator's thesis CONFIRMED: the fleet-wide d_seg plateau was the broken loop, not the
paradigms. RESIDUAL LANDMINE: default EMA decay 0.999 lags so hard a 30-ep EMA-shadow showed NO descent
while the live render fell 84% — some past "no descent" verdicts may be EMA-shadow-lag artifacts (re-validate).
Next: #81 (re-test the pose-capacity-wall #74 on the working loop) → #82 (1:1 MLX port, parity-gated, fixes
the SegNet-NaN) → #78 (the capstone, on the working loop).

## ★ POSE VERDICT (#80, 2026-06-10) — pose is REAL geometry (unlike d_seg), but sub-0.15 doesn't need the tube
The pose-sensitive subspace is RANK-1 (eff dim ≈1.07; top singular = 99.97% energy; 6 dims read ~one
scalar projection; pose-null = 99.999% of pixel space). BUT the linear pose-null is TANGENT-only — error
confined to it caps d_pose at 1e-4–5e-2 (above the 2.9e-5 tube) and grows with RMSE via SECOND-ORDER
curvature. So the pose-capacity-wall (#74) is REAL GEOMETRY, not an inert-loop artifact (d_seg dissolved on
#76; pose does NOT). frame0 carries ~20× more pose debt than frame1. **RECONCILIATION (keeps sub-0.15 alive):
sub-0.15 does NOT need the pose TUBE (2.9e-5).** S=100·d_seg+√(10·d_pose)+25·B/D at a retrained small basis:
d_seg≈5.6e-4 (0.056, reachable via the #76 loop) + d_pose≈1e-4 (pose-term 0.0316, the GOOD end of the
tangent-null band) + rate≈0.04-0.06 (~60KB) = ≈0.13 < 0.15. The pose wall blocks matching the FRONTIER tube
cheaply; it does NOT block sub-0.15 (trade tube-precision for rate). OPEN ESCAPE (untested, highest-EV $0 pose
probe): second-order Gauss-Newton curvature-aware protection (leading eigenvectors of JᵀJ, not the linear
Jacobian rows) — the frontier's learned basis is the curvature-preserving rep.

## ★ FULL UNLOCK (#81 audit, 2026-06-10) — BOTH walls dissolve; decoder verified correct; Quantizr = the pose answer
(1) DECODER CORRECT: `pixel_shuffle_2x_nhwc` BIT-EXACT (0.0 drift) vs torch.nn.PixelShuffle(2); full
HNeRVDecoderMLX matches a from-scratch reference rel 2e-7. The decoder math was NOT the bug (now a regression
guard). (2) POSE-CAPACITY-WALL is FALSE: Quantizr (PR #55, 88K params) held d_pose 0.00051 by STORING the
6-dim GT pose explicitly (pose.npy.br) + FiLM-injecting it — he does NOT reconstruct pose from pixels. The
binding constraint was the pose REPRESENTATION (store+FiLM vs reconstruct), not capacity. #74's wall = FALSE.
(3) TRUST TABLE: SUSPECT (our inert loop) = #74/#62/B1/the d_seg≈0.50 fleet plateau; TRUST (frozen-frontier,
#75-verified eval) = #64/#69/#71/#72/#73/#54. (4) 9 config defects C1-C9 source-cited; #75's "B1 ran the
curriculum" CORRECTED (B1 ran 1 recon-MSE stage + a late-ramp dual-ascent seg weight under recon dominance +
AdamW clip + skip-free = the inert quartet). (5) NEW BLOCKER C8: use_bilinear_skip=True raises
NotImplementedError in archive export (mlx_renderer.py:7456) — a successful skip-on run CANNOT build a contest
archive until export+oracle-parity lands. THE CAPSTONE RECIPE (proven-by-existence): #76 working loop (d_seg)
+ store-6-pose-scalars+FiLM (Quantizr, pose) + #67 VQ free-inflate + #82 clean stack; build the C8 export.

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
