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

## 1. VERDICT (filled after the smoke — 2026-06-10)

**HEADLINE (the pre-registered question answered):** Does the Dykstra-projected legal frame hold
BOTH d_seg AND pose at low byte? **PARTIAL-YES on the SCORER GRID, NO at LOW byte for the naive
cheap basis.** The exact operating points (`[macOS-CPU advisory]`, frozen CPU-torch SegNet/PoseNet,
GT via `frame_utils.yuv420_to_rgb`, NEVER MPS):

| start | d_seg | d_pose | delta bytes | held both? |
|---|---|---|---|---|
| GT frame1 (reference) | 0.00000 → 0.00001 | 0 → ~1e-11 | 14 (brotli floor) | **YES** |
| frontier comp PAIR (corrected) | 0.00057 | **2.40e-05** | 14 (delta=0 optimal) | **YES** |
| comp frame1 via cheap basis, full delta (rank=0, keep=1.0) | 0.00059 | 2.93e-05 | **625,475** | YES (but huge) |
| comp frame1, rank=64 + keep=0.5 | 0.00485 | **4.82e-02** | 395,415 | **NO (pose breaks)** |
| comp frame1, rank=32 + keep=0.25 | 0.01507 | 1.13e+00 | 198,812 | NO |
| comp frame1, rank=24 + keep=0.10 | 0.03678 | 1.18e+01 | 82,205 | NO |
| comp frame1, rank=8 + keep=0.02 | 0.03683 | 2.39e+01 | 18,677 | NO |

Per-pair smoke JSON: `experiments/results/task73_legal_frame_feasibility_20260610/smoke_4pairs_corrected.json`.
Byte-floor probe: `experiments/results/task73_legal_frame_feasibility_20260610/byte_floor_probe.py`.

### 1a. The corrected-pairing finding (a NO-FAKE class-8 catch the harness made)
The FIRST smoke had a **pairing bug** that the pre-registration's NO-FAKE class-8 commitment guards
against: it held PoseNet frame0 = GT and perturbed only frame1 = comp, manufacturing a phantom
mismatched pair PoseNet read as huge motion (d_pose ≈ 9.5–14.8). This is the SAME GT-decode/pairing
bug class documented in `pr110pp_r3_onhost_selector_verdict_20260610.md §0` (rgb24 inflated pose
~100×). Corrected: pair comp-frame0 with comp-frame1 (the REAL frontier pair). The corrected d_pose
of the frontier comp pair is **2.40e-05** — IN-tube (< the documented ~2.5e-4 seg/pose crossover).
Diagnostic receipt: GT pose6[0]=34.244; comp-pair pose6[0]=34.243 → d_pose 2.30e-5; the mismatched
(g0,comp1) pose6[0]=26.678 → d_pose 9.54 (the phantom). The verdict is reported on the corrected
measurement only.

### 1b. The deep geometric answer (what the FEASIBILITY framing revealed that generation could not)
1. **The pose tube is NOT binding at the frontier.** The lever-B GENERATORS walled on pose only
   because they synthesised a frame FROM SCRATCH (palette = zero motion → d_pose 12.14; INR could
   not encode motion past 0.0036). The frontier comp PAIR already encodes the motion correctly
   (d_pose 2.40e-05, in-tube). **Feasibility ≠ generation:** moving an already-correct frame the
   minimum amount holds pose where generating a frame from a pose-blind prior cannot. The wall was
   a property of generation, not of the legal-frame SET.
2. **The cheap-feasible set cell∩tube∩cheap is EMPTY at low byte for the naive low-rank/sparse
   basis.** The byte-floor probe is unambiguous: a FEASIBLE legal frame1 (via the comp0 anchor +
   delta) costs **≥ 625 KB per pair** in the raw low-rank/sparse/quantized basis; the MOMENT the
   delta is compressed below ~400 KB the **pose tube breaks first** (rank=64+keep=0.5 → d_pose
   0.048 ≫ 2.5e-4 while d_seg 0.00485 is still feasible). Pose is the BINDING constraint on the
   cheap basis — PoseNet's dim-0 ≈ 34.24 is a fine-spatial-structure motion aggregate that any
   low-rank/sparse spatial truncation perturbs catastrophically. **KILL-FEASIBILITY-EMPTY fired**
   for the naive basis (honest geometric finding, exactly as pre-registered).
3. **WHY the frontier uses an HNeRV decoder, derived not assumed.** The naive cheap basis (SVD +
   magnitude-sparse) cannot represent a feasible frame below ~400 KB/pair. The frontier's 177 KB
   WHOLE archive holds 600 feasible frames because its cheap subspace is the **learned HNeRV
   nonlinear basis** — that learned basis IS the cell∩tube∩cheap-occupying representation, and the
   frontier ALREADY occupies it. The Dykstra solve from the comp pair returns **delta=0 optimal**
   (1 iteration): there is no cheaper feasible delta than the frame the archive already stores. The
   carrier's true byte floor for a legal frame is the cost of a learned-basis legal frame ≈ what
   the frontier already pays.

### 1c. Pointer move
**NO MOVE.** No projected frame beats the frontier at ≤ frontier distortion: the feasible points are
either (a) the frontier comp pair itself (delta=0, already the frontier) or (b) far more expensive
than the frontier (≥ 395 KB/pair in the naive basis). No paired CPU+CUDA exact eval dispatched
(nothing beat the frontier to certify; `ready_for_exact_eval_dispatch=false`). $0 spent.

### 1d. What this tells the score-native carrier (the actionable hand-off)
- The legal-frame WALL is NOT pose-feasibility (the tube is reachable; the frontier is in it). It is
  the **byte cost of a learned basis that holds pose under compression**. The next carrier lever
  must compress within a basis that PRESERVES the fine spatial structure PoseNet integrates — i.e.
  a learned/score-aware basis (the HNeRV decoder, or a pose-Jacobian-aligned basis), NOT a
  generic low-rank/sparse spatial basis. This is consistent with the documented PR95-family
  HNeRV-decoder substrate and the seg/pose-marginal crossover discipline.
- The Dykstra feasibility solver is the right TOOL but the cheap-set C must be the learned basis to
  produce a frontier-relevant feasible point. Reactivation criterion: re-run the byte-floor probe
  with C = the HNeRV decoder's per-pair latent neighbourhood (project delta onto the decoder's
  reachable manifold), not SVD+sparse, and measure where pose breaks in THAT basis — that is the
  true legal-frame byte floor and the next candidate for a frontier move.

### 1e. Deliverables
- Solver: `src/tac/boundary_math/dykstra_legal_frame.py` (reuses the SegNet input-Jacobian margin
  half-spaces + PoseNet input-Jacobian + the differentiable-yuv6 patch; fail-closed on severed
  gradient). 22 behavior tests + 1 on-real-scorer slow test (all green) in
  `src/tac/boundary_math/tests/test_dykstra_legal_frame.py`.
- Smoke CLI: `tools/legal_frame_feasibility_smoke.py` (corrected apples-to-apples pairing).
- Evidence: `experiments/results/task73_legal_frame_feasibility_20260610/`.

