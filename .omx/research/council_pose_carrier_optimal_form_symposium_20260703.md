---
council_tier: T3
council_topic: pose-carrier optimal form (#250) on the CORRECTED 2026-07-03 findings
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Atick, Wyner, PR95Author, TimeTraveler]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: frontier_breaking
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "we just retracted a 'cheap pose' claim built on a linearized Jacobian argument; the new 'cheap pose' claim is ALSO a linearized Jacobian argument. Do NOT present it as more than a $0 gate."
council_assumption_adversary_verdict:
  - assumption: "the rank-6 PoseNet Jacobian basis is recomputable EXACTLY at decode from (frozen PoseNet, frame1, frame0_base)"
    classification: HARD-EARNED
    rationale: "all three inputs exist at decode; autodiff through the frozen net is deterministic (rule-118 generic algorithm). This is the load-bearing claim and it is sound."
  - assumption: "storing only the ~6-12-dim pose residual (not the full δframe0 image) reaches low d_pose at decode"
    classification: CARGO-CULTED-UNTIL-MEASURED
    rationale: "the min-norm solution LIVES in the 6-dim row space, but whether a FEW stored coefficients + a decode-recomputed basis + Gauss-Newton + uint8-STE actually lands ~0 d_pose is the exact thing the retracted claim got wrong at n=3. GATE it."
---

# GRAND COUNCIL OPTIMAL-FORM SYMPOSIUM — POSE CARRIER (#250)
## On the CORRECTED 2026-07-03 findings · pointer 0.19110 UNMOVED (MEANS)

**Confidence discipline (binding, post-whiplash):** this memo separates **SOLID** (measured/derived) from **LEAD**
(promising, unmeasured). No LEAD is a result. Nothing here moves the pointer until a byte-closed `evaluate.py`
row. The single most important output is a **$0 GATE**, not a claim.

---

## 1. What we actually KNOW (SOLID)
- **R1 store-nothing ξ: d_pose 0.0011 → contribution 0.105, ~0 pose rate.** The current shippable pose floor.
- **P-E existence proof (n=24 authority-confirmed):** a free frame0 (quant-aware LM-GN inverse solve) reaches
  d_pose **~1e-8** through the frozen CPU-torch PoseNet. Pose IS inverse-solvable to ~0 *in principle*.
- **Image-STORE carriers are rate-prohibitive (the correction):** storing the full per-pair δframe0 image is
  8.56 rate at 96×128 (n600). There is **no cheap image-STORE carrier below R1.** (This retracts the earlier
  "cheap coarse carrier nets 0.066" — a 200× n=3 rate-scaling error.)
- **The √ shape:** contribution = √(10·d_pose). Sweet spot d_pose ~1e-5 (0.010); below that, diminishing.
- **The rate constraint (hard):** a COUNTED per-pair payload costs `25·B·600/37.5M ≈ 4e-4·B` rate per byte.
  So ~0.010 rate ⇒ **~25 bytes/pair ≈ 6-12 scalars/pair.** *Cheap pose = ≤ ~12 stored scalars/pair.*

## 2. The deep-math reframe (Shannon LEAD · Dykstra CO-LEAD) — what the correction actually OPENS
Pose is an inverse problem on a FROZEN net: find frame0 s.t. `PoseNet(frame0, frame1)[:6] = ξ_target`.
Linearize about a base frame0: `J·δframe0 = r`, where `r = ξ_target − PoseNet(frame0_base)` (6-dim residual)
and `J = ∂PoseNet6/∂frame0` (rank ≤ 6). The **min-norm** correction is `δframe0 = J⁺·r` — it lives entirely in
the **6-dim row space of J** (6 basis frames).

**The key move (the synthesis the whole council converges on):** the 6 basis frames are DETERMINISTIC given
(frozen PoseNet, frame1, frame0_base) — **all available at decode** — so the decoder **recomputes the basis for
FREE** (rule-118: a generic algorithm, autodiff through the frozen net, is not counted). The archive stores
**only the ~6-12 coefficients** (the pose residual `r`, the *indirect-RD sufficient statistic*), ~24 bytes/pair,
rate ~0.010. Decode = recompute basis + a few Gauss-Newton steps → d_pose → ~0.

**This is why the correction is not a dead-end.** What was rate-prohibitive was storing the δframe0 *image*
(≈590k values). The Jacobian-coefficient realization stores the *6-dim statistic* and regenerates the basis
free — the "compile the generator, count only the video-derived payload" discipline applied to pose. The
retracted P-F conflated image-STORE (prohibitive) with coefficient-STORE + free-basis (cheap). **P-E is not
dead as a lever; it may be its cheapest legal form — UNMEASURED.**

## 3. The candidate levers (ranked; SOLID vs LEAD explicit)
| # | lever | mechanism | rate | d_pose | status |
|---|---|---|---|---|---|
| **L0** | R1 store-nothing (baseline) | store ξ (6), warp the task-space render | ~0 | 0.0011 | **SOLID** floor (0.105) |
| **L1** | **Jacobian-coefficient inverse-solve** | store 6-12-dim residual r; recompute rank-6 J basis FREE at decode; GN → target | ~0.010–0.020 | → ~0 (goal) | **LEAD — the decisive candidate** |
| L2 | P-D warp-a-REAL-keyframe | store cheap real texture + ξ; warp → pose-legible frame | ~0.03 | ~0.018? | LEAD (pose-space) |
| L3 | frame1 joint pose-legibility | add a pose term to the witness loss so frame1 is SegNet-good AND PoseNet-legible → lowers the 0.0011 floor at source | 0 | < 0.0011? | LEAD (structural; couples to #205) |

## 4. Coherence + synergy — the ONE-object view (why this fits, not bolts on)
- **se(3) screw dual-use (Chasles):** the SAME ξ that warps the partition for d_seg IS the pose target. L1's
  residual `r` is exactly the *correction* to the screw-warp's pose gap — one object, two reads.
- **Frozen-scorer Jacobian geometry — the synergy that makes pose FREE to compose:** seg⊥pose is 99.95% null
  (#206). The 6-dim pose correction lives in the pose row space, which is ~orthogonal to the SegNet argmax
  directions → **L1 composes with the d_seg witness WITHOUT perturbing d_seg.** Pose and seg are independent
  axes; fixing pose is a 0-d_seg-cost bolt-on. (This must be VERIFIED, not assumed — see the gate.)
- **rule-118 boundary (the legality that makes it cheap):** the Jacobian basis is a generic decode-time
  algorithm (FREE); the 6-dim `r` is the only counted, video-derived payload. This is the indirect-RD sufficient
  statistic — the theoretically minimal thing to store. **FIREWALL:** storing a per-pair *optimized image* as
  "code" would be the NO-FAKE #6/#8 eval-hack; storing the 6-dim `r` + recomputing a generic basis is legal.
- **Unified level-set flow:** pose is the ξ-facet; L1 is its cheap realization, composing orthogonally with the
  d_seg-facet (#205). The whole falls out of one variational object.

## 5. Research grounding (our own — no new dispatch needed to state this)
L1 is the convergence of shelved, already-built surfaces: **#157 exact-sensitivity KKT/reverse-waterfill**,
**#47 evaluator null-space compiler** (certified-invisibility basis), **#73 legal-frame Dykstra feasibility**,
**#140 low-rank pose-section codec**, **#206 pose FiLM/read-back**, **#193 se(3) Lie engine**. The VCM theory
layer (#151 indirect-RD/CEO) names `r` as the sufficient statistic. **One flagged EXTERNAL-research question**
(targeted, not a flurry): prior art on *Jacobian-subspace / gradient-domain residual coding with a
decoder-recomputed basis* in neural codecs (Ballé-family context models store latents, not decoder-recomputed
sensitivity bases — this may be genuinely novel; worth a $0 literature probe BEFORE claiming originality).

## 6. VERDICT + the decisive $0 GATE (the only thing that resolves L1)
**PROCEED_WITH_REVISIONS.** The image-store carrier is dead (correct); L1 (Jacobian-coefficient + free basis)
is the candidate cheapest legal pose optimum and is **UNMEASURED**. Do NOT build a carrier yet.

**THE $0 GATE (next pose action, n600, NO-FAKE, CPU-torch authority):**
store ONLY the ~6-12 Jacobian coefficients of `r` (NOT the δframe0 image); at "decode" recompute the rank-6
`J = ∂PoseNet6/∂frame0` basis from (frozen PoseNet, frame1, frame0_base=warp(frame1,ξ)); apply k Gauss-Newton
steps; measure the resulting **d_pose through the frozen authority**, all 600 pairs. Report:
1. d_pose vs coefficient-count K (K = 6, 8, 12, 18) and GN steps k — does K≈6-12 reach **≪ 0.0011** at ~0.01-0.02 rate?
2. the decode-side wall-clock (basis recompute + GN, ×600) — does it fit the **30-min budget**? (this is L1's real risk — the retracted claim was rate; L1's risk is decode COMPUTE.)
3. the **d_seg cost** of applying δframe0 (should be ~0 by seg⊥pose null — VERIFY).
4. GO (cheap legal pose lever exists → pose → ~free legally, re-opens "pose leaves the budget" HONESTLY) /
   NO-GO (coefficient store doesn't reach low d_pose OR decode too slow → R1's 0.105 is the floor → pursue L2/L3).

**NOTE:** the running n600 MLX `--pf-generic` sweep uses a DCT/low-rank basis — the WRONG basis (not
pose-sensitive). It does NOT answer L1; the **Jacobian basis** is the right one. So L1 needs its own probe.

## 7. Mission framing (honest, conservative)
Until L1's gate returns: **plan for pose = 0.105 (R1) as the conservative floor.** The binding fight remains
**d_seg + rate**, and **#205 is the pointer-mover** — unchanged by any of this. L1 is a $0, high-EV shot at
making pose a near-free line-item *legally*; it is the right next pose action, but it is a lead with a clear
gate, not a result. **Reactivation of L2/L3** if L1 NO-GOs. Dissent (Contrarian) preserved: L1 is another
linearized-Jacobian argument — measure before believing.

**Continual-learning:** append this deliberation to the council posterior; the corrected pose findings +
L1-synthesis + the $0 gate propagate to the DAG + #248 + [[project_pose_solved_screw_twist_dual_use_film_conditioned_sidecar]].

---

## 8. OPERATOR STEER 2026-07-03 — L3 ELEVATED TO PRIMARY: store-nothing goes cheaper IN TRAINING (0 payload), fold into #205
Operator: *"Can store-nothing go cheaper … deep math deeper than ever grand unified transcendental Understanding …
extreme iterated and optimal store nothing pose … in training levers that fall out for store nothing pose."*

**The deepest answer (Shannon/Dykstra/Atick/Assumption-Adversary converge): YES — store nothing at all; make the
render pose-legible IN THE SEGNET-NULL.** The render is ONE object read by two frozen nets. SegNet sees only
`argmax(frame1)` → WITHIN each cell all texture/chroma/sub-margin-luma is FREE (the 99.95% seg⊥pose null).
PoseNet reads the FULL YUV6 of both frames → it READS that within-cell structure. So paint **ξ-consistent
(optical-flow-consistent) pose-legible texture into the SegNet-null** → PoseNet reads the motion → **d_pose drops
at ~0 d_seg cost + 0 rate.** The se(3) screw fully dual-use: ONE ξ warps the partition (d_seg) AND its consistent
null-texture makes PoseNet read ξ (d_pose), store-nothing. R1 floored at 0.0011 ONLY because it trained with
`w_pose=0` — the null was never used for pose.

**The 0-rate in-training levers that fall out (fold ALL into #205):**
1. `w_pose>0` — pose loss that lives in the SegNet-null (orthogonal → lowers d_pose without moving d_seg).
2. FiLM-condition the render on ξ (#206) — render CONSUMES the twist; both frames ξ-consistent.
3. canonicalize-to-ground-frame + optimal per-pair ξ (#193) — shrink the rigid-warp residual (road ≈ planar homography).
4. ξ-consistent texture/chroma/sub-margin-luma in the null — chroma is TRIPLE-use (also a d_seg lever); within-cell luma below the argmax margin is free for SegNet, read by PoseNet.
One mechanism — pose-legibility painted into SegNet's null, ξ-consistent — several knobs.

**Why L3 DOMINATES L1/L2:** 0 rate (vs L1's ~0.01, L2's ~0.03) AND one object/one run (vs a bolt-on carrier) AND
it's already going to run as #205. So **L3 is the primary path; L1 (Jacobian-coefficient) is the $0 FALLBACK** if
store-nothing floors too high. **Confidence: mechanism SOLID (null real, PoseNet reads it); the achievable d_pose
FLOOR is UNMEASURED (R1 was w_pose=0) — #205 with w_pose>0 measures it.** Not a claim; a measurement #205 makes.

**REVISED VERDICT → #205 is the pose experiment too.** Configure #205 (the d_seg pointer-mover) with the store-
nothing-pose gauge (`w_pose>0` + FiLM-on-ξ + canonicalize + null-texture) → ONE run yields d_seg convergence AND
the optimal store-nothing d_pose, at 0 pose rate → the full S. **Disciplined launch sequence (respects the #205
OOM/crash history + CONTAINMENT):** (a) deep-math design of the store-nothing-pose config → (b) extreme-iterated
recursive-adversarial review → (c) #237 worktree→main reconcile → (d) memory-preflight + resumability + a short
measured-runnability n600 smoke at the REAL config (axis-9: measure peak RSS + scored quantities through byte-close
BEFORE the multi-hour burn) → (e) LAUNCH on operator GO. Steps (a)-(d) = $0/light prep. Cross-refs: #205, #206,
#227 (seg⊥pose decoupling MLX-port, engage now that w_pose>0), #241 (store-nothing carrier mode), #193, #237.
