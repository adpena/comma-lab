# Scorer numerical-dynamics LEVER MAP — exploit / signal / adapt / protect (2026-06-11)

**Subagent:** `scorer_numerics_dynamics_lever_map_20260611`. **Source:** operator 2026-06-11 — *"all of
this might point to dynamics we can exploit, or gain signal from and adapt to and protect against."*
**Evidence grade:** every number cited is `[macOS-CPU advisory]` / `[macOS-MLX research-signal]` /
`code_inspection` / `external_github_pr_comment` as tagged inline. **NO score claim, no promotion, no
dispatch.** This is the STRATEGY layer on top of the measured oracle geometry — a ranked lever map, NOT a
re-measurement. Pointer UNMOVED at **contest-CPU 0.19109982** (177,169 B, sha `b46897267ded…`, ABOVE T_1
→ GOAL UNSATISFIED). CUDA champion is a SEPARATE axis at 0.20533 (recode does not transfer).

## 0. The three numerical dynamics this map turns into levers

All three are already MEASURED (consumed, not re-derived):

- **D-LOCAL — local→contest directional drift (one-sided, calibratable).** Local macOS torch-CPU reads a
  slightly HIGHER S than the contest Linux-x86_64 CPU (the leaderboard authority). Anchor: PR107 macOS-CPU
  `0.19664189` vs Linux-x86_64-CPU `0.1966358879`, local worse by `+6e-6`
  (`feedback_local_advisory_is_conservative_upper_bound_vs_contest_20260611.md`). The DQS1/FEC6 trust-region
  calibration (`codex_findings_local_cpu_contest_drift_eureka_20260522T194925Z`) fits a stable-core bias of
  **+0.000010** (median, n=5 same-archive anchors), **SegNet-only** (`d_seg` higher by 1.0–1.2e-7; rate and
  PoseNet identical), guard band 0.000003. ⇒ local is a **conservative UPPER BOUND**: `contest_S ≈ local_S −
  ε`, ε≈1e-5 (in-class), one-sided, not a two-sided band.

- **D-AXIS — contest CPU↔CUDA gap (large, structured).** ~**0.033** (PR102 anchor): **70% pose / 30% seg /
  0% rate** (`cuda_cpu_pose_drift_mechanism_deep_dive_20260508`). Mechanisms (measured shares pending, but
  localized): (A) conv2d/depthwise reduction-order CPU-serial vs cuDNN-parallel; (B) GT decode PyAV-`yuv420_
  to_rgb` (CPU) vs NVDEC/DALI (CUDA) — ±1 LSB chroma-boundary divergence ×4 by the std=63.75 normalizer; (C)
  GELU tanh libm vs CUDA intrinsic; (D) BatchNorm1d flat reduction. **The leaderboard is the CPU axis** — so
  CPU is the optimize/predict/submit target; CUDA is its own axis, never inferred from CPU.

- **D-ARGMAX — reduction-order argmax fragility (the boundary set).** `d_seg = E[argmax(out1) ≠ argmax(out2)]`
  is INVARIANT to small logit perturbations except at the decision boundary, where they flip. The
  cross-hardware flip set is tiny and 100% boundary-localized: MLX-GPU vs torch-CPU = **243 flips / 19.66M
  pixels (0.00124%), EVERY flip at a SegNet boundary, mean top-2 margin 5.2e-5**
  (`mlx_scorer_port_drift_audit_20260611`); MLX-CPU vs torch-CPU = 2 flips (genuine fp32 ties). `d_pose` is
  NOT argmax — it is a noise-floor-limited MSE (at frontier `d_pose≈2.9e-5` the CUDA noise σ≈0.012 RMS is ~4×
  the pose signal). The margin field quantifies the static fragility: **4.83% of pixels are fragile at the
  2-logit threshold, 2.16% are boundary; 95% carry >2 logits of free room; 80.40% carry >5**
  (`segnet_margin_field_20260609.json` + lever-B target meta).

These three are the SAME physics (fp reduction order) read at three couplings: D-ARGMAX is the per-pixel
event; D-AXIS is its aggregate across the largest hardware change (+ the GT-decode term); D-LOCAL is its
aggregate across the smallest (macOS↔Linux, same CPU class), hence one-sided and tiny.

---

## 1. THE RANKED LEVER MAP (one row per lever, across the 4 modes)

EV is toward a LOWER **contest-CPU** score (the leaderboard) unless noted; PROTECT EV is risk-reduction.
"Finalize" = what the lever needs before it is actionable beyond advisory. "Composes-with" names the
existing artifact it stacks on WITHOUT duplicating.

| # | mode | dynamic | mechanism (one line) | EV toward lower contest-S | needs to finalize | composes-with (no dup) |
|---|---|---|---|---|---|---|
| **L1** | EXPLOIT | D-LOCAL | **Submit-decision rule**: a local advisory below (or within ε+guard of) the next threshold is HIGH-confidence contest-below → bank it / spend the paired exact eval, don't sit. | HIGH — converts measured candidates into banked frontier rows; the gate that turns lever-B's 0.120 advisory into a submission | the drift sibling's quantified ε per axis/class (use +1e-5 in-class for now) | `local_cpu_contest_drift.py` eureka trigger (#22); lever-B carrier campaign |
| **L2** | EXPLOIT | D-AXIS | **Optimize/predict against the CONTEST-CPU cell, not the macOS cell**: argmax-train + select candidates on the Linux-CPU equivalence class (the leaderboard), measured via paired exact eval, not the local cell. | MED-HIGH — prevents optimizing a phantom local optimum; the leaderboard is CPU so this is the correct target manifold | a few paired macOS↔Linux-CPU same-archive anchors to confirm the cell map (mostly already in eureka) | lever-B argmax objective; eureka calibration |
| **L3** | EXPLOIT | D-ARGMAX | **Spend bytes at the small-margin boundary set; hide free distortion in the certified-stable interior** (95% pixels >2 logits room, 80% >5). Carrier/coder must be RIGHT only at ~2–5% fragile pixels; everything else is rate headroom. | HIGH (rate-side) — the seg-axis dual of the resize-null basis; the reason lever-B's carrier is 2.54× smaller | nothing new (margin field already measured); the coder side is lever-D's open gate | `segnet_margin_field` (#54); lever-D margin-conditional coder (#72); lever-B 80.40%>5 interior budget |
| **L4** | SIGNAL | D-ARGMAX | **Cross-hardware argmax-DISAGREEMENT map = a FREE near-boundary-pixel detector** (macOS↔MLX-GPU↔[Linux/CUDA]): the 243-flip set IS the fragile boundary band, measured with ZERO scorer-margin call. Feed it to the sensitivity map / waterfiller as a decoder-free "where flips can happen" prior. | MED — sharpens the bit-allocator's fragile-pixel localization for free; reduces lever-D's position alphabet | one Linux-CPU↔CUDA flip map (the macOS/MLX leg is done) to confirm the boundary set transfers across the contest axes | `mlx_scorer_port_drift_audit` flip set; sensitivity map hook #1; lever-D waterfill |
| **L5** | SIGNAL | D-AXIS | **Per-layer drift = scorer-structure signal**: the layer-trace localizes WHICH ops (conv reduction / GELU / BN / GT-decode) carry the gap → tells the carrier which frequency/channel bands the scorer is numerically committed to (luma low-freq for pose, boundary for seg). | LOW-MED — informs ADAPT levers (L7/L8); a measurement that feeds, not moves, S | the spectral atlas right-sized re-run (deferred, lower priority than capstone daemon) | spectral atlas v2 (pose low-freq/horizontal, seg broadly weak); MLX layer trace |
| **L6** | ADAPT | D-LOCAL | **Directional calibration in the spend-triage rule**: project local→contest with `−ε` before ranking candidates for paid eval, so near-line candidates aren't wrongly killed. | MED — recovers candidates locally-just-above-threshold that cross on contest (the conservative-bias dividend) | the sibling's per-class ε (DEFER the exact value to it; +1e-5 in-class meanwhile) | `local_cpu_contest_drift.py`; the drift sibling (reference, don't re-derive) |
| **L7** | ADAPT | D-ARGMAX | **Train argmax-correct pixels with a cross-hardware-ROBUST MARGIN**: don't just match the macOS argmax — push the boundary pixels' top1−top2 margin past the measured cross-hardware logit drift (≥~0.1, the MLX-GPU max logit delta 0.096) so they DON'T flip wrong on Linux/CUDA. | MED-HIGH — converts a fragile local win into a contest-stable win; directly defuses the L-P1 risk | the contest-axis logit-drift magnitude at boundary pixels (MLX-GPU 0.096 is the local proxy; confirm on Linux/CUDA) | lever-B/-G margin-weighted hinge (already uses lever-G margin weighting); noise_std=0.012 calibration (exploit-4 of the cpu-cuda dive) |
| **L8** | ADAPT | D-AXIS | **Noise-injection at FastViT output matched to the CUDA pose noise floor (σ≈0.012 RMS)** during candidate training → pose robust to CUDA precision noise without harming CPU. | LOW-MED — CUDA-axis hedge; pose at frontier is already noise-floor-limited so upside is bounded | only relevant if the CUDA axis becomes the submission target; CPU is the leaderboard | eval_roundtrip/noise_std (CLAUDE.md); cpu-cuda dive exploit-4 |
| **L9** | PROTECT | D-ARGMAX | **Fragile-boundary-flip risk**: a pixel argmax-correct on macOS that flips WRONG on Linux/CUDA loses d_seg at the contest. DETECT via L4's cross-hardware flip map; AVOID via L7's robust margin; NEVER ship a carrier whose wins live inside the 5.2e-5-margin tie band. | risk-reduction HIGH — this is the failure mode that would make a local-sub-0.15 evaporate at the contest | the cross-hardware flip map (L4) as a pre-submission gate | L4 flip map; L7 robust margin; lever-B portability=argmax-parity contract |
| **L10** | PROTECT | D-AXIS | **d_pose-near-frontier recompute discipline**: at `d_pose≈3e-5` the score's `sqrt(10·d_pose)` term is in the noise-floor regime; never claim a pose win from a single axis, recompute S from components (the rounded field lies), require paired CPU+CUDA. | risk-reduction MED — prevents a phantom-pose promotion (the 100× GT-decode trap + noise-floor σ) | discipline only (already a CLAUDE.md non-negotiable; this names the numerical reason) | CLAUDE.md dual-axis + GT-decode-via-yuv420_to_rgb; cpu-cuda dive σ-fit |
| **L11** | PROTECT | D-LOCAL | **One-sided bias is in OUR favor — don't over-correct**: ε is +1e-5 (local worse), so a local crossing is MORE likely real, not less. The protect action is to NOT treat local-above-by-a-hair as a kill, and NOT add a two-sided margin that wastes the conservative dividend. | risk-reduction MED — prevents discarding real contest-crossers | the sibling confirms ε stays one-sided out-of-class (it widens to +1.4e-4/+2.9e-4 for mixed substrate — calibrate by class) | drift memory; eureka out-of-class caveat |

---

## 2. THE SINGLE MOST-EXPLOITABLE DYNAMIC

**D-ARGMAX (the small-margin boundary set), via L3.** It is the deepest exploit because it is not a
calibration trick — it is the structural reason the score is COMPRESSIBLE below the cluster: `d_seg` is an
argmax-rate, so it is invariant to the 80.40%->95% of pixel logit-energy that lives far from any boundary.
That certified-free interior is the seg-axis dual of the resize-null basis (#47's 80.67% pixel-energy null),
and it is exactly what makes lever-B's score-native carrier 2.54× smaller than the frontier decoder's
seg-share. Every byte the leaderboard cluster spends rendering interior pixel fidelity is wasted against an
argmax oracle. D-LOCAL (L1) is the highest-EV-this-week *decision* lever (it banks results), but D-ARGMAX is
the dynamic that produces the class shift.

---

## 3. TOP 3 LEVERS TO ACTUATE NEXT (toward sub-0.19 → sub-0.15) + concrete next step

Ranked by EV/cost toward a LOWER contest score. None duplicates existing work; each is the next step of a
confirmed lever.

1. **L1 — the submit-decision rule (banks lever-B).** Best EV/cost: it is the gate that converts the already
   -confirmed lever-B carrier (advisory S≈0.120, below T_3) into a banked exact frontier. **Next step:**
   byte-close the lever-B 70,452 B carrier into a legal `archive.zip` + run ONE paired contest-CPU+CUDA exact
   eval; apply the L1 rule (local−ε vs T_1) to decide submit. Reuse `lever_b_byte_close_exact_eval_readiness
   _20260611.md` (the readiness memo) + `local_cpu_contest_drift.py` eureka trigger; do NOT rebuild the
   carrier. This is the one lever where the EXACT POINTER can move now.

2. **L3 + L4 fused — boundary-targeted bit allocation with the free cross-hardware flip detector.** **Next
   step:** finish lever-D's margin-conditional coder (#72) with the position alphabet collapsed to L4's
   cross-hardware-flip ∪ low-margin set (decoder-free side info), and run its $0 conditional-entropy smoke to
   see if a non-trivial subset clears the 1.27 B/flip break-even with near-zero collateral. This is the seg
   rate-axis attack that stacks on lever-B's carrier. Reuse lever-D's harness + `segnet_margin_field`; do NOT
   re-measure the margin field.

3. **L7 — cross-hardware-robust margin training (protects the lever-B win at the contest).** **Next step:**
   add a hinge term to lever-B's training that pushes boundary-pixel top1−top2 margin past the measured
   cross-hardware logit drift (≥~0.1, anchored on the MLX-GPU 0.096 max logit delta), so the carrier's
   argmax-correctness is STABLE on Linux/CUDA, not just macOS. This converts the L9 risk into an L7
   guarantee BEFORE the L1 paired eval — cheap insurance that a local-sub-0.15 doesn't evaporate. Reuse the
   lever-B/-G margin-weighted hinge; add only the cross-hardware margin floor.

**Composition note (no duplication):** L1/L3/L7 all stack on the lever-B carrier (the confirmed score-native
substrate); L4 reuses the MLX drift-audit flip set as a free fragility prior into the existing sensitivity-map
hook #1; the null-space compiler (#47) and lever-G engineered corrections are REFERENCED as the orthogonal
rate-null (pixel-energy) and zero-byte-correction (DEFERRED at the global-fixed subclass) levers — this map
adds the ARGMAX-fragility + DRIFT-calibration dimension on top, it does not re-derive either.

## 4. Authority + firewall

Everything here is `[macOS-CPU advisory]` / `[macOS-MLX research-signal]` / `code_inspection` →
mechanism/strategy only. NO score row, NO promotion, NO dispatch, NO MPS. The contest CPU/CUDA exact
600-sample `evaluate.py` on 1:1 hardware is the only authority that closes any of these levers; D-LOCAL ε is
DEFERRED to the drift sibling (`local_to_contest_scorer_drift_ladder_and_correction_20260611.md`, not yet
landed) — use +1e-5 in-class as the interim conservative offset. The most-exploitable dynamic (D-ARGMAX) and
the top-3 actuation order are the deliverable; the next exact-pointer-moving unit is L1 (byte-close lever-B +
paired eval + submit-decision rule).
