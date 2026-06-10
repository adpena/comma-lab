# Legal-frame FEASIBILITY via Dykstra alternating projections — pre-registration + verdict (task #73)

**Subagent:** `task73_legal_frame_feasibility`. **Task #73** — the legal-frame FEASIBILITY solve
(the deepest under-attacked crux). **Evidence grade:** `[macOS-CPU advisory]` (frozen CPU-torch
SegNet/PoseNet, GT via `frame_utils.yuv420_to_rgb`, NEVER MPS). NO score claim, `promotable=false`,
`ready_for_exact_eval_dispatch=false` until a paired CPU+CUDA exact eval lands. $0 local first.
Frontier read from pointer: **0.19109982 [contest-CPU], 177,169 B**.

---

## 0. The pre-registration (written BEFORE building — this section is frozen)

### The crux (why this is different from the lever-B smoke #56/#57)
The lever-B smoke PROVED the **logit generator** mechanism (a tiny net hits the frozen SegNet's argmax
to d_seg 0.0073) — but it scores LOGITS, not FRAMES, and it WALLED on POSE (palette frame is pose-blind
→ d_pose 12.14; the INR generator hit a 0.0036 pose ceiling). The legal-frame task is NOT "generate a
frame that looks right" (generation, walled on pose). It is a **FEASIBILITY** problem: find ANY frame in
the set

    { F : argmax SegNet(F) = L*  (the SegNet argmax cell) }
  ∩ { F : ‖PoseNet(pair_F)[:6] − p*‖² ≤ τ_pose  (the pose tube) }
  ∩ { F : F − base ∈ cheap-encoding subspace  (resize-null + low-rank/sparse) }

cheapest. Feasibility = constraint satisfaction = **Dykstra alternating projections** onto three sets,
re-linearizing the nonlinear SegNet/PoseNet Jacobians each outer iteration (sequential convex feasibility).

### The method (frozen)
1. **Base frame** per pair: a CHEAP frame already in the cheap subspace — the **GT frame1** itself (the
   honest base; its perturbation in the cheap basis is what we store) OR a low-res-GT carrier. δ = F − base.
2. **Project onto the margin-cell (set A):** for each scored pixel `p`, each wrong class `c≠L*(p)`:
   `( J_{L*(p),p} − J_{c,p} )·δ ≥ −m_{p,c}` (the §4 polytope). The projection moves δ the minimum amount
   so every violated half-space (margin < γ) is satisfied — a per-pixel linear step along the SegNet
   input-Jacobian difference toward the runner-up class. (Reuse: `TorchSegNetJacobian` for the real
   per-pixel `∇_input(logit_target − logit_runnerup)`; the margin field from the frozen forward.)
3. **Project onto the pose-tube (set B):** move δ so PoseNet's 6 scored dims land within τ_pose — a step
   along the PoseNet input-Jacobian (the 6×N linear system, projected onto the GT pose). (Reuse:
   `compute_posenet_pixel_saliency` Jacobian path; here we need the SIGNED 6×N Jacobian, not just its norm —
   built fresh from the same differentiable-yuv6 backprop.)
4. **Project onto the cheap set (set C):** project δ onto resize-null ⊕ a low-rank/sparse basis so the
   stored perturbation stays cheap. (Reuse: `ResizeProjector.project_frame` / `evaluator_invisibility_basis`.)
5. **Dykstra correction terms** carried across A/B/C (true Dykstra, not naive cyclic POCS, so the iterate
   converges to the projection onto the intersection when the sets are convex linearizations).
6. **Re-linearize** A and B Jacobians at the new iterate each OUTER loop (SegNet/PoseNet are nonlinear);
   iterate to convergence (Δδ small or max outers).
7. **Measure** the converged frame's EXACT d_seg (popcount argmax-disagreement, frozen SegNet) + EXACT
   d_pose (frozen PoseNet, GT via yuv420_to_rgb) + the perturbation's encode bytes in the cheap basis.

### The pre-registered QUESTION
Does the Dykstra-projected legal frame hold **BOTH** d_seg≈0 **AND** pose-in-tube **SIMULTANEOUSLY** at
**LOW byte** — where the lever-B generators could not (palette pose-blind 12.14; INR pose-ceiling 0.0036)?

### The pre-registered PREDICTION
Projection should hold both terms BETTER than generation because feasibility does not force a single
smooth net to encode both a sharp partition boundary AND motion — it moves the GT frame the *minimum*
amount to satisfy each constraint, and the GT frame ALREADY holds the pose tube (d_pose=0 at GT) and is
ALREADY in the cell (d_seg=0 at GT). The non-trivial question is whether projecting δ onto the CHEAP set
(C) — which is what makes it a carrier, not a copy — keeps BOTH d_seg and d_pose small. If the cheap
projection of δ stays in cell∩tube → feasible cheap carrier exists → potential frontier move. If the
cheap projection KICKS the frame OUT of cell or tube and the A/B re-projections cannot pull it back at
low byte → the cheap-feasible set is empty at that byte → a sharp geometric finding (the carrier
genuinely needs more bytes than the cheap subspace provides).

### The pre-registered KILL criterion (honest geometric finding either way)
**KILL-FEASIBILITY-EMPTY:** if, after convergence, no projected frame holds d_seg ≤ 0.01 AND
d_pose ≤ ~5e-4 (pose-tube; the documented seg/pose crossover) at a perturbation byte-cost below the
frontier seg-share headroom — i.e. the projections do NOT converge to a cheap feasible point — REPORT
THAT as the honest finding: **the cheap-feasible set cell∩tube∩cheap is empty at low byte; the carrier
needs more bytes.** This is NOT a failure to report; it is the sharp geometric answer the score-native
carrier needed (it tells lever F the true byte floor of the legal frame).

### NO-FAKE commitments (frozen)
- The projections ACTUALLY move the frame toward each constraint on the REAL frozen scorer (class 1: no
  no-op that returns the base frame). Tests assert each projection step reduces the relevant constraint
  violation on the real SegNet/PoseNet.
- d_seg/d_pose are the EXACT frozen-scorer functionals (class 8), NOT a proxy. GT via yuv420_to_rgb,
  NEVER MPS.
- If cell∩tube∩cheap is empty at the cheap byte, report it honestly (do not declare a feasible frame
  that the exact scorer does not certify).

### Substrate-compute law
Frozen CPU-torch SegNet/PoseNet (NEVER MPS). GT decode via `frame_utils.yuv420_to_rgb` ONLY. Targets
on `/Volumes/VertigoDataTier/pact/...` (NO /tmp). Local $0 first; ≤$1 paired exact eval only if a cheap
feasible frame beats the frontier at ≤ frontier distortion.

---

## 1. VERDICT (filled after the smoke)

*(pending the build + smoke run)*
