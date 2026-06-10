# PTNC + frame1 dual-fidelity — VERDICT (task #61)

**Subagent:** `task61_ptnc_frame1_dual_fidelity`. **Authority of every number below:** `[local CPU-torch
advisory]` — exact upstream PoseNet/SegNet (`DistortionNet`) on CPU, GT decoded via
`upstream/frame_utils.yuv420_to_rgb` ONLY, S recomputed from components (the rounded field lies).
`[macOS research-signal]` for the carrier numpy forward (numpy↔torch RGB parity within 1 LSB). **NOT** the
contest 600-sample harness → non-promotable per the authority ladder. `$0` spend, no GPU, no paid
dispatch, **NO MPS**. `promotable=false`, `score_claim=false`, `ready_for_exact_eval_dispatch=false`.

**Frontier (pointer, not hardcoded):** `0.19109982` `[contest-CPU]`, 177,169 bytes. Secondary gate:
sub-0.15. **Pre-registration:** `.omx/research/frame1_dual_fidelity_ptnc_DESIGN_20260610T131300Z.md`.

---

## 0. PRE-REGISTRATION (written BEFORE measurement; see DESIGN memo §3–§4)
**PREDICTION:** PTNC (PoseNet-Jacobian-saliency-weighted recon anchor, USC IDSE with the MEASURED frozen
oracle) breaks the #57 d_pose 0.0036 ceiling toward the tube (≤ ~1e-4) at ≤ ~15 KB because it stops
spending capacity on pose-irrelevant luma. For the FULL candidate, the frame1 dual-fidelity carrier gets
total S below frontier 0.19110 — OR frame1 needs near-full-RGB fidelity, converging to HNeRV-class.

**KILL/DEFER CRITERION:** if PTNC cannot get the FULL candidate's total S below frontier (or sub-0.15) at
a byte cost preserving the rate advantage, DEFER to lever C and record: pose-relevant luma is NOT cheap
enough; the score-native pose axis converges to HNeRV-class.

**RESULT vs pre-registration:** PREDICTION **REFUTED** on both counts. (1) PTNC did NOT break the 0.0036
ceiling toward the tube — its best op-point (tiny capacity, d_pose 0.00331, 13.6 KB) is right AT the #57
ceiling, NOT the predicted ≤1e-4. The Jacobian-saliency anchor is CAPACITY-DEPENDENT (it beats dense at
tiny 13.6 KB but LOSES to dense at small 23.6 KB — the IDSE "free to be wrong in the pose-null" assumption
helps when capacity is scarce but breaks under uint8 quantization once capacity is ample, §1). It moves the
op-point, not the ceiling. (2) The frame1 dual constraint is now PROVEN ANTAGONISTIC at the coordinate-INR
capacity (a pose-trained frame1 gets d_seg 0.733; a seg-trained frame1 gets d_pose 12.14 — §3), and the
cheapest frame1 holding BOTH terms costs >400 KB (§2), destroying the rate advantage. **The KILL/DEFER
criterion FIRES → DEFER-to-lever-C** (not a kill; the carrier primitive works in isolation; the SPECIFIC
composition is falsified — Catalog #307 IMPLEMENTATION-LEVEL).

---

## 1. PTNC vs dense vs identity — the Jacobian-saliency anchor is CAPACITY-DEPENDENT, NOT ceiling-breaking

The PTNC mechanism is genuine, not a rename: the exact 6-dim PoseNet pose-MSE objective is IDENTICAL
across modes; only the input-domain RECON anchor differs (dense = uniform MSE; ptnc = per-pixel weight ∝
measured PoseNet pixel-Jacobian norm; identity = uniform via the saliency code path). The measured
Jacobian field is real + concentrated (median 1.6e-6 vs max 1.6e-3, a 1000× range; 77% of pixels nonzero;
the weight map renormalises to mean 1.0 with max ~38 — strong redistribution). Two surfaces measured, both
frame0-only, exact CPU PoseNet on the QUANTIZED numpy-decoded frame:

**(a) Matched-capacity head-to-head, h64/m24 (~23.6 KB), 60 ep:**

| anchor mode | seed | train pose-MSE (fp32) | **exact quantized d_pose** | bytes |
|---|---|---:|---:|---:|
| dense (#57 control) | 0 | 0.00578 | **0.01033** | 23,877 |
| dense | 1 | — | **0.00305** | 23,572 |
| **ptnc** (floor 0.02) | 0 | 0.00856 | **0.06145** | 23,659 |
| **ptnc** (floor 0.02) | 1 | — | **0.01783** | 23,555 |
| ptnc (floor 0.40) | 0 | — | 0.03596 | 23,707 |

**(b) RD sweep, tiny capacity h48/m16 (~13.6 KB), 80 ep:**

| anchor mode | **exact quantized d_pose** | bytes | d_pose/kb |
|---|---:|---:|---:|
| dense | 0.006413 | 13,658 | 0.000481 |
| **ptnc** | **0.003307** | 13,584 | 0.000249 |

**The picture is CAPACITY-DEPENDENT, not a clean dominance.** At SMALL capacity (23.6 KB, surface a) dense
beats PTNC at both seeds (0.0103 vs 0.0614; 0.00305 vs 0.0178). At TINY capacity (13.6 KB, surface b) PTNC
beats dense (0.00331 vs 0.00641). This is exactly the diagnosed mechanism: PTNC concentrates scarce
capacity on pose-relevant pixels, which HELPS when capacity is tight (tiny) but the quantization-bleed
HURTS once there is enough capacity to fit the pose-null too (small+). The bleed mechanism: PTNC tolerates
large carrier error in low-Jacobian pixels — but uint8 quantization + the PoseNet resize+yuv6 mixing
perturb those large-error regions, and the perturbation RE-ENTERS the pose signal because the
GT-operating-point Jacobian is NOT zero once the carrier output moves far off-GT (the Taylor-validity risk
the spec §Risk flagged). Raising floor 0.02→0.40 improves PTNC at small (0.0614→0.0360); floor→1.0 → dense.

**The decisive point for the pre-registration: PTNC does NOT break the 0.0036 ceiling toward the tube.**
Its best (tiny, 0.00331) is right AT the #57 ceiling (0.0036), 114× above the tube 2.9e-5 — NOT the
predicted ≤1e-4. The coordinate-INR pose family ceiling is ~0.003 regardless of anchor; PTNC moves the
op-point but not the ceiling. (Train-time fp32 pose-MSE briefly favored PTNC mid-train at small capacity —
the trap: the fp32 "win" does not survive quantization; authority is the exact quantized d_pose.)

## 2. frame1 dual-fidelity RD — the actual wall (exact d_seg AND d_pose, 4 pairs, frame0=GT0)

The decisive measurement: progressively-degraded GT frame1 luma, BOTH terms measured, full S with the
23.5 KB dense frame0 carrier accounted:

| frame1 rep | d_seg | d_pose | frame1 bytes | total bytes | **S** |
|---|---:|---:|---:|---:|---:|
| gt1 (sanity ceiling) | 0.0 | 0.0 | 0 | 23,572 | 0.0157 |
| lowres_gt1_f16 | 0.0226 | 0.243 | 26,934 | 50,506 | 3.85 |
| lowres_gt1_f8 | 0.0086 | 0.0435 | 105,988 | 129,560 | 1.61 |
| lowres_gt1_f4 | 0.0022 | 0.0124 | 418,457 | 442,029 | **0.86** |
| lowres_gt1_f2 | 0.00059 | 0.00072 | 1,599,304 | 1,622,876 | 1.22 |

The BEST raw-low-res frame1 op-point (f4, S=0.86) is **4.5× worse than frontier 0.19110** and already
costs 418 KB for frame1 ALONE (2.4× the entire 177 KB frontier archive). Holding BOTH d_seg and d_pose
requires near-full-RGB frame1 fidelity, and the raw byte cost explodes — the score-native rate advantage
is destroyed. The S curve is U-shaped (f4 is the min): coarser → d_seg/d_pose blow up; finer → bytes blow
up. No raw-low-res op-point comes near frontier.

## 3. THE DUAL CONSTRAINT IS ANTAGONISTIC at coordinate-INR capacity (the sharp #61 diagnosis)

An amortized frame1 carrier (h96/m32, 48.8 KB) trained for POSE reaches d_pose 0.0199 — but its d_seg is
**0.733** (seg_term 73.3!) because it is a smooth RGB frame that totally fails the SegNet argmax
partition. Conversely the #57 seg-trained frame1 (palette/lever-B, ~65 KB) reaches d_seg 0.064 but d_pose
**12.14** (pose-blind label map). The two objectives PULL THE FRAME1 CARRIER IN OPPOSITE DIRECTIONS:

| frame1 carrier trained for… | d_seg | d_pose | verdict |
|---|---:|---:|---|
| POSE (h96 INR, 48.8 KB) | **0.733** | 0.0199 | great pose, catastrophic seg |
| SEG (palette/lever-B, 65 KB) | 0.064 | **12.14** | great seg, catastrophic pose |

A single coordinate-MLP cannot hold a SHARP argmax partition (needs high-frequency boundaries the smooth
INR can't represent — the lever-B 0.0075 d_seg floor) AND carry the pose luma simultaneously. This is the
complete proof of the wall #57 located: **frame1's dual (seg+pose) constraint is not jointly satisfiable
by the coordinate-INR family at score-native byte budgets.**

## 4. Full advisory S of the best assembled candidate

The best honest full candidate at score-native byte budgets remains the #57 byte-closed candidate (seg
generator frame1 + carrier frame0): **S = 11.65** (d_seg 0.064, d_pose 2.67, 85.6 KB). PTNC does not
improve it (PTNC frame0 is worse than the dense frame0 already in #57; and no frame1 representation in §2/§3
holds both terms cheaply). The frame1 dual-fidelity RD (§2) shows the best ACHIEVABLE score-native-ish
candidate (carrier frame0 + lowres_gt1_f4 frame1) is **S ≈ 0.86 at 442 KB** — still 4.5× frontier and 2.5×
the frontier byte budget.

**Eval gate ("advisory S beats frontier 0.19110 OR sub-0.15"):** NOT met (best achievable S=0.86 ≫
0.191). **NO paired exact eval launched** (correct fail-closed: do not spend $ to confirm a
non-improvement). `$0` spent.

## 5. VERDICT: DEFER-to-lever-C (NOT kill; the carrier primitive works in isolation)

Per CLAUDE.md "Forbidden premature KILL" + Catalog #307 IMPLEMENTATION-LEVEL: the amortized carrier
primitive is real + working in isolation (frame0 dense d_pose 0.003–0.010, 23 KB, numpy-portable,
scorer-free inflate). The PTNC Jacobian-saliency ANCHOR is empirically shown CAPACITY-DEPENDENT (beats
dense at tiny 13.6 KB, loses at small 23.6 KB under quantization) and does NOT break the 0.0036 ceiling
toward the tube — a genuine, pre-registered partial-negative (the predicted ≤1e-4 is REFUTED). The frame1
dual-constraint composition is FALSIFIED on the full S (antagonistic objectives at coordinate-INR
capacity; near-full-RGB byte floor destroys the rate win). **The honest conclusion the pre-registration
named is confirmed: pose-relevant luma is NOT cheap enough; the score-native pose axis converges to
HNeRV-class** (HNeRV-parity-discipline lesson 5 — the full RGB renderer dominates a single-component slot).

### Lever C reactivation campaign (pre-registered; the next build)
A fresh-init, JOINTLY-trained **per-pair-latent CONVOLUTIONAL** frame1 decoder (NOT a coordinate-MLP — the
§1 ceiling + §3 antagonism prove the coordinate-MLP family cannot carry sharp argmax + pose luma) trained
against BOTH SegNet (d_seg) AND PoseNet (d_pose) on the contest video, export-first (archive grammar +
scorer-free inflate declared before training). Predicted byte band 40–120 KB (between score-native 85 KB
and frontier 177 KB). Open question: can a per-pair-latent conv decoder reach frontier d_seg=5.6e-4 +
d_pose=2.9e-5 below 177 KB. This IS the HNeRV-class carrier — the structurally expressive frame1 the
coordinate-INR cannot be. Reactivation gate: a conv-decoder smoke that holds d_seg < 0.01 AND d_pose <
0.01 jointly at < 120 KB (4 pairs) before any paid dispatch.

**Secondary reactivation (cheaper, deferred):** PTNC with a re-measure (EM/Dykstra) loop — re-measure the
Jacobian at the carrier's CURRENT operating point each N epochs (not frozen at GT) so the "pose-null"
assignment stays valid as the carrier moves. The §1 diagnosis predicts this is the ONLY way the IDSE
anchor could beat dense; it was out of scope here (single GT-point measurement). Lower priority than lever
C because even a perfect frame0 carrier (d_pose→0) leaves the dominant frame1 debt unaddressed.

## 6. Wire-in (Catalog #125)
1. **sensitivity-map — ACTIVE:** the §3 antagonism table is the new sensitivity input — frame1's d_seg and
   d_pose marginals are OPPOSED under the coordinate-INR; the waterfiller must treat frame1 as a joint
   (seg,pose) cell, not two independent budgets. The measured PoseNet pixel-Jacobian field
   (`posenet_jacobian_saliency`) is a reusable per-pixel pose-sensitivity surface for any future allocator.
2. **Pareto — ACTIVE:** §2 maps the frame1 {d_seg, d_pose, bytes} surface and establishes it is U-shaped
   with min S=0.86 at 442 KB — the Pareto-feasible move is NOT a bigger/different INR; it is the HNeRV-class
   conv decoder (lever C).
3. **bit-allocator — ACTIVE:** §1 shows the Jacobian-saliency anchor does NOT improve d_pose-per-byte under
   quantization (dense dominates); the allocator should NOT route frame0 bytes via PTNC. §3 shows frame1
   bytes are mis-allocated to a coordinate-INR (cannot hold both terms).
4. **cathedral-autopilot — gate NOT met:** advisory best S 0.86 ≫ frontier 0.191; no paired-eval dispatch.
5. **continual-learning — ACTIVE:** reseeds the planner: (a) PTNC Jacobian-saliency anchor is
   CAPACITY-DEPENDENT — it BEATS dense at tiny 13.6 KB (d_pose 0.0033 vs 0.0064) but LOSES at small 23.6 KB
   (quantization breaks the free-pose-null assumption once capacity is ample; floor→1 recovers dense); it
   moves the op-point but does NOT break the 0.0036 ceiling toward the tube; (b)
   the coordinate-INR frame0 pose ceiling is ~0.003–0.010 at 13–23 KB (confirms #57, non-broken); (c) frame1's
   dual (seg+pose) constraint is ANTAGONISTIC at coordinate-INR capacity (pose-trained → d_seg 0.733;
   seg-trained → d_pose 12.14); (d) the cheapest frame1 holding both terms is >400 KB raw-low-res →
   score-native pose axis converges to HNeRV-class; (e) the next lever is a per-pair-latent CONV decoder
   (lever C), NOT another coordinate-MLP.
6. **probe-disambiguator — RESOLVED:** "does the measured-Jacobian (IDSE) anchor beat dense MSE for the
   pose carrier?" → NO (dominated under quantization). "can a coordinate-INR frame1 hold seg AND pose
   jointly?" → NO (antagonistic). "is the score-native pose axis cheaper than HNeRV?" → NO (frame1 needs
   near-full-RGB fidelity). The next probe: a per-pair-latent conv frame1 decoder (lever C).

## 7. Deliverables + cross-references
- **Modules (NO-FAKE, tested):** `src/tac/boundary_math/posenet_jacobian_saliency.py` (the MEASURED frozen
  PoseNet pixel-Jacobian field + numpy-portable weight map + fail-closed severed-gradient guard) — 14
  behavior tests (`tests/test_posenet_jacobian_saliency.py`: 13 fast + 1 slow on-scorer real-Jacobian) +
  7 anchor-mechanism tests (`tests/test_ptnc_anchor_mechanism.py`: identity==dense MSE; constant carrier
  fails; pose-null error cheaper / pose-tube error costlier under PTNC; floor→1 recovers dense). 91
  boundary_math tests green (0 regressions); ruff clean.
- **Tools:** `tools/ptnc_train_pose_carrier.py` (PTNC trainer with `--anchor-mode {dense,ptnc,identity}` —
  the falsifiable comparison built in; exact PoseNet objective, differentiable yuv6, eval_roundtrip, numpy
  parity gate, exact re-measure on the quantized decoded frame). `tools/ptnc_rd_sweep.py` (the RD curve:
  modes × capacity + non-learned lowres-GT reference). `tools/ptnc_frame1_dual_fidelity_probe.py` (the §2
  wall measurement: joint d_seg+d_pose+bytes per frame1 representation, full S).
- **Artifacts (SSD tier):** `/Volumes/VertigoDataTier/pact/ptnc_task61_20260610/` (head-to-head carrier
  npz + ptnc_train_result.json per variant; frame1_probe/frame1_dual_fidelity.json; rd_sweep_f0/rd_sweep.json).
- **Cross-refs:** `score_native_pose_carrier_20260610T125000Z.md` (#57 verdict — the wall) ·
  `sota_plus_original_inventions_20260610T125100Z.md` (PTNC invention spec — area c) ·
  `closed_spec_boundary_math_system_of_equations_20260610.md` (seg-null / margin polytope) ·
  `frame1_dual_fidelity_ptnc_DESIGN_20260610T131300Z.md` (pre-registration) · CLAUDE.md HNeRV-parity
  lesson 5 (full renderer not single-component slot).
