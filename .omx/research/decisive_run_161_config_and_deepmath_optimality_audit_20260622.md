# Decisive run #161 — full config + engineering + deep-math optimality audit (2026-06-22)

Operator: "one more deep pass for engineering and configs and deep math optimality." Authority:
`[contest-CPU advisory]`; pointer UNMOVED 0.19110; NO score claim. Run: `yousfi_r3_taper_marginhinge_e5`
(bc20, 600 pairs, MPS-gradient, CPU authority-eval), now stage3_v332_smooth ~ep 9,200. Pure code/config read
(zero CPU contention with the live run).

## A. Config / engineering audit — 9 axes (1 bug FIXED, 8 clean)

| # | axis | finding | verdict |
|---|---|---|---|
| 1 | curriculum | built by reading the VENDORED `stages.stageN.make_config` (epochs/LR/σ/λ/muon_lr/batch_size are the proven PR95 values); `total_epoch_budget=None` → full **29,650** | PR95-FAITHFUL ✓ |
| 2 | stage-8 Muon LR floor | was `--no-muon-lr-floor-fix` (BUG-B): Muon shared AdamW's floor → stuck 50% peak → finisher under-converges | **FIXED** (resume, verified True @ep575) |
| 3 | C1a coder-aware reg (stage 5) | `weight_entropy_penalty_lambda=0.0` (no flag) → `_penalty_active=False` → C1a NOT superseded | canonical ✓ |
| 4 | seg loss | `--seg-margin-hinge` → `seg_surrogate="margin_hinge"` across ALL stages (validated detector-informed d_seg lever); soft-cosine T<0.3 gradient-cliff therefore N/A | ✓ (see watch-item) |
| 5 | batch size | no `--batch-size` flag exists → curriculum-hardcoded (vendored B=8); all batch-coupled tuning (η√B, σ√B, EMA) internally consistent — NOT the B=64 mistuning risk | ✓ |
| 6 | EMA | decay 0.999; at ep 9,200 the shadow is hundreds of τ warmed → #85 short-run shadow-lag N/A; exact reads trustworthy | ✓ |
| 7 | pose throttle | `pose_grad_every_k=1` (no flag) → pose trains trunk every epoch = score-optimal (throttle is score-negative) | ✓ |
| 8 | weight-entropy penalty | OFF (λ=0) → no rate-penalty interaction with the d_seg finishers | ✓ |
| 9 | KD-warm-start on resume | `--kd-warm-start-dir` was used at the ORIGINAL fresh launch; on my resume the checkpoint exists → "resume always wins" → KD IGNORED → SAME trajectory continued, no re-priming | ✓ |

`defer_batch_sync=ON` (throughput only, proven bit-identical). score_aware_qat=False (vendored uniform QAT,
PR95-faithful). Net: the run is fully correct after the muon-fix.

## B. Deep-math optimality — the descent structure of the remaining curriculum

**The d_seg residual is conditioning-limited, not capacity-limited.** H_dseg ≈ (1/P)Σ ℓ''(m_p) j_p jᵀ_p,
κ-proxy ≈ 19, shallow boundary (66.5% of flips <0.5 logit). The optimizer hierarchy maps onto this exactly:

- **AdamW (stages 1–7) = a DIAGONAL preconditioner D⁻¹.** It rescales per-coordinate but does NOT decorrelate
  the off-diagonal coupling of H_dseg → on a κ≈19 Hessian its effective d_seg rate stays conditioning-gated →
  **power-law-slow**. This is precisely the observed trajectory (stage 2→3: d_seg 0.00220→0.00213, −3% over
  thousands of epochs). That slowness is the PREDICTED behavior of a diagonal preconditioner on a conditioned
  Hessian — **NOT a wall.**
- **Muon (stage 8) = a SPECTRAL preconditioner** via polar(M)=UVᵀ (Newton-Schulz) → whitens the update across
  the full coupled spectrum → convergence O(ln 1/ε) **independent of κ**. So the BULK of d_seg finishing is
  mathematically RESERVED for stage 8 — and the muon_lr_floor_fix (axis #2) is exactly what lets stage 8 anneal
  to the LR where that whitened descent polishes the shallow flips instead of bouncing at 50% peak.

**The two finishers + budget:** stage 5 (C1a-L7, 9000 ep, AdamW margin-hinge d_seg loss — a long power-law
descent, the single longest stage) + stage 8 (Muon, 5000 ep, the κ-buster). Combined 14,000 epochs of explicit
d_seg work, spectral finisher LAST.

**The quantitative ask (re-derived with the run's measured pose+rate 0.050+0.0584=0.108):**
- beat borrowed frontier 0.19110 → d_seg < (0.19110−0.108)/100 = **8.3e-4** (2.5× from 0.00213) → ln 2.5 = 0.92
- sub-0.15 (T_3) → d_seg < (0.15−0.108)/100 = **4.2e-4** (5.1×) → ln 5.1 = 1.63

In Muon's O(ln 1/ε) regime these are SMALL log-distances; 5000 Muon epochs at the corrected floor should clear
the frontier-beating 0.92 readily and put sub-0.15's 1.63 in reach — IF the conditioning analysis holds (the
live descent + the #85 EMA fix corroborate it). **The decisive empirical measurement is the stage-8 Muon d_seg
slope; the stage-5 C1a-L7 slope (~5h) is the leading indicator.**

**Residual risk the math flags (not new):** the shallow boundary is also QUANT-FRAGILE (the int5-cap finding) →
stage 4 QAT could nudge d_seg up before stages 5/8 recover it. But stage 4 here is the VENDORED uniform 127-level
QAT that PR95 itself used to reach 0.193 → proven recoverable by the later stages. Not a new risk.

## C. The one watch-item (a future A/B, NOT a mid-run intervention)
`margin_hinge` is applied across ALL stages. The recent int5 finding ("CE recovers d_seg where margin-hinge
ACTIVELY HARMS it at the coarse grid") suggests CE might dominate in the COARSE early stages while margin-hinge
wins in the FINE d_seg-finishing stages (5, 8). Caveats: that finding was at the int5 QUANTIZED grid (may not
transfer to full precision); and this run is already BELOW the plain-CE bc20 basin (0.00213 < 0.00260) with
margin-hinge. So this is a candidate per-stage-loss A/B for the NEXT vehicle, not a reason to perturb a healthy
descending run.

## NO-FAKE ledger
- MEASURED (code/config read, $0): curriculum vendored-faithful; C1a active; seg=margin_hinge all stages; B=8
  hardcoded; ema 0.999; pose_grad_every_k=1; weight-entropy OFF; muon_lr_floor_fix NOW True; KD ignored on resume.
- DERIVED: AdamW=diagonal→power-law-slow d_seg; Muon=spectral→O(ln 1/ε); finishing reserved for stages 5+8;
  frontier-beat = ln 2.5 log-distance, sub-0.15 = ln 5.1.
- NOT claimed: no score moved; the d_seg crossing is a PREDICTION to be measured at the stage-5/8 slopes, not a result.

Cross-refs: `decisive_run_161_muon_lr_floor_fix_resume_20260622.md` · `dseg_boundary_hessian_conditioning_20260621.md`
· `feedback_frontier_int5_score_aware_qat_finetune_path_b_caps_20260618.md` (the margin-hinge-vs-CE watch-item).
