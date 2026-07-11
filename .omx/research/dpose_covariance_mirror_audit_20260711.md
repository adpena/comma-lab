# d_pose-side MIRROR of the covariance-totality audit — the texture trunk's "lives-on-POSE" edge CLOSED

**Agent:** Fable crux-math subagent · **Date:** 2026-07-11 · **Cost:** $0 cached + one bounded
frozen-CPU-torch PoseNet probe (n=48 strided, single, no SegNet forward; live run pid 88030
untouched, verified 43% CPU healthy before/during). **Pointer 0.19108282 UNMOVED** — this is a LAW
refinement + an architecture verdict; it moves no score. All numbers
`[macOS-CPU research-signal / CPU-torch advisory]` `score_claim=false promotable=false`.
Per `docs/operating_manual_craft_handoff.md`: answer first, every number re-derived, labels
MEASURED/DERIVED/INFERRED out loud, own-round-1 review at bottom, verdicts scope-laddered.

**STORES CONSULTED:** `covariance_totality_texture_trunk_verdict_20260710.md` (the d_seg parent) ·
`einstein_pass_covariance_laws_20260710.py` (`witness_general_covariance_totality_v1`) · graph
recall "pose PoseNet jacobian cached frame0 inverse solve texture" → FEED-alldim + FEED-posesolve +
FEED-unitC (DAG) · `posenet_luma_chroma_sensitivity_asymmetry_v1` (equation; chroma-HF pose-null
DERIVED, luma 11.1×/plane, frame0 seg-free) · `verdict_frame0_chromahf_dofs_20260710.json` ·
`tools/pose_frame0_inverse_solve_probe.py` (#249: P-E existence 2.71e-7; image-space stores
RATE-PROHIBITIVE) · #206/#238/#245 (R1 dxi byte-close d_pose 0.001610 / 7.2KB) · L68/L86/L87
memory hooks · `.omx/tmp/covariance_audit/audit.py` (machinery reused verbatim for parity).

---

## ANSWER FIRST (the verdict, with scope)

**verdict_scope: FORMULATION** (texture-trunk-as-pose-carrier, under the measured byte economics
and the measured/existence-proved output-space steering channel).

**The texture trunk is DEAD-for-pose too — but by DOMINANCE, not absence.** The mirror is
genuinely ASYMMETRIC, and that asymmetry is the finding:

| | d_seg side (parent audit) | d_pose side (this audit) |
|---|---|---|
| Is the residual/signal photometric-HF? | **NO** — residual smooth, hf/lf 2e-4 | **YES** — pose readout is broadband/HF-legible (MEASURED below) |
| Why the trunk is dead | **ABSENCE** — no d_seg signal to carry | **DOMINANCE** — the signal is bias-dominated + reachable at 100–1000× fewer bytes through the 6-dim output-space channel |

- **CAPSTONE: DEFINITIVELY single covariant trunk on BOTH axes** (v7.5.2 form). **#395 fully
  dispositioned: DROP.** Reactivation criterion (pose side): the S1/S2 output-space per-pair solve
  (FEED-posesolve rungs A2/A2+) fails to reach ≤0.0011 at n600 through byte-close — and even then
  the trunk's calibrated HF ceiling (~1.3e-3, MEASURED below) barely matches R1's banked 1.61e-3,
  so reactivation would demand a NEW mechanism, not this one.
- **Joint intrinsic dim ≈ 10 (seg-8 + pose-adds-~2).** Pose's extra DOF should ride the dedicated
  low-dim ξ/steering channel (exactly S2's 6+k design and the banked 7.2KB dxi shape), NOT the
  trunk's FiLM mod vector. Mod/hidden-dim implications in §4.
- **NO-FAKE:** no recovered-d_pose is claimed; nothing here is byte-closed; pointer 0.19108282
  UNMOVED. Part B is n=48 (bounded by the containment rule, NOT decision-grade n600); the verdict's
  load-bearing byte-economics rest on n600-MEASURED anchors (R1 byte-close, #249), with Part B used
  for band-shape characterization + a ceiling.

---

## 1. The ξ-explained fraction, mirrored (MEASURED, n600, $0 cached)

Tool: `.omx/tmp/dpose_mirror_audit/mirror_partA.py` (imports the parent audit's exact machinery).
NO-FAKE parity: rank-8 cum-var **0.9321** reproduces the parent's 0.9316 (float32-Gram vs float64,
5e-4 agreement); EDT argmax-roundtrip asserted 0 px.

The prompt's step-1 as literally stated ("regress PoseNet-6 on gt_poses") is an **identity** —
`gt_poses` *is* the per-pair PoseNet-6 readout (R² = 1 by construction; DERIVED, and the Part-B
baseline re-verifies it: max|PoseNet(GT pair) − gt_poses| = 7.6e-6 raw, d_pose parity 2.6e-12).
The non-trivial mirror is: **is the pose readout a function of the covariant partition code the
single trunk already carries?**

| model (target = ξ_6, per-dim standardized) | lin CV R² (null) | kNN CV R² |
|---|---|---|
| code_8 | 0.228 (−0.03) | **0.329** |
| code_8 + phase | 0.237 (−0.10) | 0.315 |
| code_16 | 0.361 (−0.08) | **0.387** |
| phase only | 0.017 (−0.03) | — |

Compare the parent's forward direction (code←ξ): linear 0.223, kNN 0.392. **The overlap is
near-symmetric (~0.22 linear / ~0.33–0.39 nonlinear each way): the partition code and the pose
readout share one ego-motion core; neither is a function of the other.**

Per-pose-dim (kNN CV R² from code_8): dims 0/2/5 = **0.45 / 0.58 / 0.57** (the coarse ego dims —
the partition's homography orbit carries them); dims 3/4 = **0.03 / 0.02** (GT std 0.0096/0.0074,
the smallest-amplitude dims); dim 1 = 0.33. **Physical picture:** the fine rotation-rate dims move
the partition by *sub-argmax-quantum* amounts — they are pose DOF the partition literally cannot
see. CCA confirms: canonical correlations **[0.79, 0.65, 0.36, 0.31, 0.27, 0.05]** — two strong
shared directions, four weak.

## 2. The decisive residual: is pose photometric-texture-indexed? (MEASURED)

**(a) n600, $0:** per-pair pose residual (after code_8+phase, out-of-fold kNN; 68.5% of pose var
unexplained) vs per-pair GT luma-HF energy (camera-res, top-octave = residual after 4×4 box):
**Pearson −0.134, Spearman −0.089 ≈ 0.** The unexplained pose is NOT texture-indexed — it is the
fine-ego-DOF deficit of §1, not photometry.

**(b) n=48 bounded CPU-torch band-limit ladder** (`mirror_partB_posefreq.py`; Gaussian σ at camera
res on BOTH frames, uint8 re-quantized, exact `cpu_verdict_d_pose`; NO-FAKE baseline parity
2.6e-12; scorer plane = camera/2.28, so σ_cam≈2.3 ≙ one 384-plane pixel):

| σ_cam | d_pose mean (median) | √(10·d̄) | reading |
|---|---|---|---|
| 1 | **4.20e-3** (2.34e-3) | 0.205 | top camera octave — mostly ABOVE the scorer grid, survives via no-AA aliasing |
| 2 | 5.73e-2 (2.60e-2) | 0.757 | ≈ witness-work-res top octave |
| 4 | 5.58e-1 (4.21e-1) | 2.36 | mid-band (reproduces FEED-unitC blur-catastrophe) |
| 8 | 1.68 (1.48) | 4.09 | structure scale |
| σ4 f0-only / f1-only | 8.14e-2 / 8.89e-2 | — | frames symmetric (matches the 0.86× prior) |

**Unlike d_seg's residual, the pose readout IS exquisitely HF-legible — broadband with strong
fine-luma weight.** (Chroma-HF remains EXACTLY pose-null — derived, `frame_utils:65-72` box-average
— so "texture trunk pose job" could only ever mean the LUMA half.)

**(c) The attack on (b) — bias vs information** (`mirror_partB2_biasvar.py`, raw pose-6 readouts):

| σ | total MSE | constant-bias share | after 6+36-param linear-ξ calibration (in-sample) |
|---|---|---|---|
| 1 | 4.20e-3 | **62.7%** | **1.30e-3** |
| 2 | 5.73e-2 | **76.4%** | 9.55e-3 |

**63–76% of the band's pose effect is a CONSTANT readout bias** — correctable by 6 global scalars
(~24 bytes), not a per-pair texture job. After the trivial linear-ξ calibration, the genuinely
pair-specific pose information in the top octave has a ceiling of **~1.3e-3**.

## 3. Why DOMINANCE kills the pose job (the byte economics; anchored on n600-MEASURED rows)

The pose objective is **6 numbers per pair** (`modules.py:83-84`: MSE over `out['pose'][..., :6]`,
verified this audit re-read). What could a per-pair luma-HF texture trunk buy?

- Its calibrated pair-specific ceiling: **~1.3e-3** (MEASURED §2c) — the same order as the residual
  R1 *already banks without any texture* (**0.001610 n600 byte-close, 7.2KB dxi** — MEASURED,
  #238/#245), and 10–40× ABOVE the S2 output-space-steering target (~1e-4–3e-5, PREDICTED,
  FEED-posesolve) whose feasibility is existence-proved (#249 P-E **2.71e-7**, MEASURED).
- Its cost: dense per-pair HF luma. #249 MEASURED the image-space store class **RATE-PROHIBITIVE at
  n600**; the steering channel costs ~6–20 scalars/pair. **≥100–1000× byte disadvantage for an
  equal-or-worse Δd_pose.** Strictly dominated.
- The high HF sensitivity itself (the 11.1×-luma Jacobian, the steep σ-ladder) is not an argument
  FOR the trunk — it is exactly WHY the cheap steering channel works: large J means tiny, cheap,
  joint-descent-shaped perturbations move the 6-dim readout far (#249's LM solve mechanism).

**INFERRED (labeled):** the witness's current pose residual (1.6e-3 class) is a *steering/descent*
residual, not an HF-photometric deficit — supported by §2a (residual not texture-indexed) and by
the photometric-wall history (post-hoc dead, joint descent crosses), but not itself byte-closed.

## 4. The mod/hidden-dim question (the operator's explicit ask; MEASURED → DERIVED)

- **TwoNN local intrinsic dims (n600, MEASURED):** code_8 → 3.6 · code_16 → 4.4 · ξ_6 → 5.3 ·
  joint(code_8⊕ξ) → 5.7 · joint(code_16⊕ξ) → 6.8. **Pose adds ≈ +2.1–2.4 local DOF beyond the
  partition code** (consistent across both codebook widths; consistent with CCA's 2-strong/4-weak
  split and the per-dim table — dims 3/4 + part of 1 are partition-invisible).
- (TwoNN local dims sit below the linear/global rank-8 because curvature spreads a low-dim manifold
  across more linear dimensions; the established "intrinsic dim ~8" is the global-linear estimate.
  In that same accounting: **joint global dim ≈ 8 + 2 = 10**.)
- **Whitney sizing (DERIVED):** seg-only 2·8+1 = 17 (the standing mod-17–19 answer, #223/#299
  unchanged for a seg-only trunk). Joint 2·10+1 = **21**.
- **Concrete implication:** two admissible architectures —
  (a) pose conditioning rides the SAME FiLM mod vector → mod-dim must grow to **≥21** (mod-16
  under-embeds the joint object by ~5, mod-17–19 by 2–4);
  (b) trunk stays seg-covariant at **mod-17–19** and pose's **+2 DOF route through the dedicated
  low-dim dxi/steering channel** (6+k scalars/pair, the S2 shape; the banked 7.2KB dxi IS this
  channel's archive shape).
  **The measurement favors (b)**: the pose-only DOF are partition-invisible (sub-argmax-quantum),
  so spending trunk mod capacity on them buys d_seg nothing; and (b) is byte-cheaper. This is
  exactly the v7.5.2 conditioning-gate + banked-dxi design — now with a measured DOF count behind
  it. **Hidden-dim: size to rank-8 + gauge zero-modes; NO texture-trunk hidden budget on either
  axis.**

## 5. TRIALITY

- **equations:** third `EmpiricalAnchor` `dpose_photometric_band_mirror_audit_20260711` REGISTERED
  on `posenet_luma_chroma_sensitivity_asymmetry_v1`
  (`tac.canonical_equations.posenet_luma_chroma_asymmetry_20260710`; latest-row-wins re-populate
  done). No NEW equation — this refines the existing scorer-dimension-asymmetry law with the
  frequency-resolved + bias-decomposed pose leg.
- **DSL:** N/A-with-rationale — analysis audit; no new trainer flag (the levers it *supports* —
  conditioning-gate pose-finish + dxi channel — already exist; the lever it *kills* — texture
  trunk — is being dropped, and its DROP is owned by the v7.5.2 line).
- **DAG:** FEED-dpose-mirror appended to `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md`.

## OWN-ROUND-1 REVIEW (adversarial)

1. **Is the σ-ladder a domain-shift artifact rather than "HF carries pose"?** Partly — and that was
   measured, not hand-waved: 63–76% of the shift is constant bias (§2c). The verdict uses only the
   post-calibration pair-specific ceiling (1.3e-3). If anything the bias share makes the trunk
   *deader*.
2. **n=48, not n600 (Part B).** Bounded deliberately by the containment rule (live run owns the
   heavy slot; PoseNet probe kept single + small). Labeled everywhere; the DOMINANCE verdict's
   load-bearing numbers (R1 0.001610 byte-close; #249 existence + rate-prohibitive) are n600
   artifacts. Part B contributes band SHAPE + a ceiling; a 12× larger sample would not plausibly
   move a 100–1000× byte-dominance conclusion. Scope kept at FORMULATION accordingly.
3. **In-sample linear-ξ calibration (43 params, n=48) overfits.** Yes — so 1.3e-3 is, if anything,
   an UNDER-estimate of the irreducible ceiling... which is the direction that *favors* the trunk,
   and it still loses by 2–3 orders of magnitude on bytes. Robust.
4. **Is gt_poses-as-ξ circular?** gt_poses is the PoseNet readout (the scored object), not the true
   ego twist — the parent audit had the same lossy-proxy caveat. For THIS audit that is the correct
   basis: d_pose is defined against exactly this readout.
5. **The "pose collapse 2.67–12.66" conflation trap (prompt warning):** not repeated — no claim here
   touches the stored-sidecar-vs-carrier-composition history; the R1 number used is the
   byte-closed 0.001610, correctly attributed to joint-descent-trained sidecar-shaped bytes.
6. **Could the +2 DOF be PoseNet readout NOISE, not scene DOF?** Possible in part (readout noise
   inflates TwoNN of ξ_6). Even so, the sizing consequence is unchanged: those dims are
   partition-invisible either way, so they belong in the dedicated pose channel, not the trunk mod.
7. **Frame_0 nuance:** f0/f1 symmetric at σ4 (0.081/0.089) — the frame_0-as-pose-carrier DOF
   (FEED-unitC) is untouched by this verdict; it concerns WHERE cheap pose bytes live, not whether
   dense texture is the carrier.
