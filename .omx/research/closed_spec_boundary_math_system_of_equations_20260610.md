# The closed specification + the boundary-math system of equations (operator reframe, 2026-06-10)

**Operator, verbatim (six messages, one crux):** *"we can represent the boundary math then"* · *"we are
using wrong data structures and algorithms and solving for the wrong thing possibly"* · *"we can have a
system of equations"* · *"we know precisely what the math and pixels and boundaries look like and which
are contiguous and which are not and also have bitmask code and a ton of hard pair and sensitivity
analysis tools"* · *"evaluate.py is also a product of segnet and posenet with different characteristics
and we don't need human visual fidelity and perhaps need some kernel or core representation grammar
runtime or something; or otherwise maybe we can go the zig route and pursue a runtimeless [decoder]"* ·
*"exactly we have modules.py and evaluate.py and contest video and that's all the information we need."*

This memo is the foundational reframe of the offensive architecture. It unifies lever A (quotient
compiler), lever B (score-native carrier — now explained), and lever D (boundary coding), and it
specifies the carrier-runtime route. It supersedes the "search a rule family" framing of lever G as the
WRONG ALGORITHM (NO-FAKE class 6: search-masquerading-as-solver).

## 1. The problem is CLOSED and fully specified by three frozen objects
The optimal archive `A*` is a deterministic function of exactly:
- **M = `upstream/modules.py`**: SegNet `S: R^{H×W×3} → R^{384×512×5}` (reads frame1 ONLY, `x[:,-1,...]`);
  PoseNet `P: R^{2×H×W×3} → R^{12}` (first 6 dims scored).
- **E = `upstream/evaluate.py`**: `S(A) = 100·d_seg + sqrt(10·d_pose) + 25·|A|/D`, `D=37,545,489` (fixed).
- **V = `upstream/videos/0.mkv`**: 1200 frames → 600 pairs; defines the GT label maps
  `L*_t = argmax_c S_c(decode(V)_t.frame1)` and GT poses `p*_t = P(decode(V)_t)[:6]`.

"That's all the information we need" ⟹ `A* = argmin_A S(A)` is **fully determined** — a closed
deterministic optimization over a frozen oracle, NOT a learning problem. No training data, no human
prior, no generalization. Everything is derivable from {M, E, V}. (This is the Time-Traveler thesis made
literal: we have all the information we need to solve the problem space.)

## 2. evaluate.py is a composed oracle with DIFFERENT characteristics than its parts
The scored functionals are not `S` and `P` directly; they are:
- **d_seg = mean_pixels [ argmax S(F1) ≠ L* ]** — a **combinatorial / SET** functional on the argmax
  PARTITION. Piecewise-constant in F1; gradient zero almost everywhere, with deltas only at the argmax
  boundary. **This is why recon-MSE and smooth generators are the wrong tool** — they optimize a smooth
  surrogate over a discontinuous set functional, over-smoothing the 1D boundary that is the *entire*
  signal. The right mathematics is combinatorial/graph, on the partition.
- **d_pose = mean_pairs ‖P(pair)[:6] − p*‖²** — a **smooth, low-dim regression** (600×6 = 3600 scalars),
  pooled-before-sqrt ⟹ a single GLOBAL budget (a pose error on any pair trades 1:1 with any other).
Two different mathematics ⟹ **two different carriers**. Forcing both through one RGB renderer is the
category error behind the whole 0.19–0.20 cluster and our own 0.0075 generator floor (lever B).

## 3. The right data structures (operator: "wrong data structures and algorithms")
| Object | WRONG (current) | RIGHT (this reframe) |
|---|---|---|
| frame1 seg target | dense 384×512×3 RGB array | **region adjacency graph** of `L*` (connected components = contiguity) + **boundary contours** (chain-code / context-arithmetic) — O(boundary), not O(area) |
| d_seg measurement | argmax over dense logits, float compare | **bitmask popcount(XOR)** of 5 class masks (we have bitmask code: `lossless/transforms`, `stc_dasher`) |
| "is this pixel flippable / how cheaply" | searched per-pixel correction | **per-pixel margin-polytope free-budget** `b(p)=m(p)/‖g_p‖` from the measured SegNet margin field + Jacobian atlas |
| pose | per-frame RGB fidelity | 600×6 trajectory + the luma-motion tube polytope |

## 4. The system of equations (the SOLVE, not the search)
The evaluator-equivalence CELL for frame1 = { F : argmax S(F) = L* pixelwise }. First-order (Taylor about
GT — the operator's "Taylor combining everything"): for each pixel `p`, each wrong class `c ≠ L*(p)`:

    S_{L*(p)}(F;p) − S_c(F;p) ≥ 0
  ⟺ ( J_{L*(p),p} − J_{c,p} ) · δ  ≥  − m_{p,c}          (δ = F − GT, m_{p,c} = current margin ≥ 0)

A **system of linear inequalities** → a polytope `P` (the argmax cell). Interior pixels (large margin) =
slack = free to coarsen/quantize/region-fill. Boundary pixels (91% margin<0.5) = tight = the ONLY bits
we pay. The carrier is the **solution** to:

    minimize   description_length(A)   subject to   δ ∈ P (argmax cell)  ∧  pose(F) ∈ tube

solved by LP / weighted graph-cut / **MDL region-merge over the RAG** (merge/drop a region iff its
flip-debt `Σ over its boundary of (1/N)` < its contour byte-cost — a real combinatorial solve, not a
search). d_seg = 0 by construction for the kept partition; we trade tiny regions for bytes by SOLVING
the cut, not sweeping.

## 5. What we were solving WRONG (the category errors, named)
1. **recon-MSE** (every NeRV trainer): optimizes pixel fidelity — orthogonal to the scored functionals
   over the 80.67% resize-null subspace and over all region interiors; over-smooths the 1D boundary.
2. **lever B generator**: smooth RGB net cannot encode a sharp partition boundary → 0.0075 floor
   (now explained, not just measured).
3. **lever G as a searched rule family**: candidate-search wearing a solver's name (NO-FAKE class 6).
   The polytope SOLVES the minimum-perturbation correction per flip pixel in closed form.
4. **dense arrays everywhere**: O(area) data structures for an O(boundary) object.

## 6. The carrier-runtime: minimal grammar/kernel OR runtime-less (Zig) — operator's two routes
The inflate side need only EMIT a frame1 that lands in the argmax cell + the luma supporting the pose
tube. Human visual fidelity is irrelevant. Two routes:
- **(a) core representation grammar + tiny kernel/interpreter**: archive = a program in a small domain
  grammar {region-fill(label, contour), luma-field(...), pose(...)}; inflate interprets it. Compact;
  carries a small interpreter.
- **(b) runtime-less / native (Zig)**: a tiny static native rasterizer (no Python runtime) that fills
  regions from contours + lays the luma field. Smallest inflate footprint, fastest, contest-compliant if
  it obeys the runtime rules and loads no scorer. Archive bytes = the contour/region/pose/luma payload;
  the decoder is fixed rate-free code.
Either way the **scored object is stored directly** (partition contours + pose trajectory + minimal
luma), rasterized into the cell — vs storing a full RGB renderer. This is the class shift.

## 7. Why this is UNQUESTIONABLY original (Innovation Gate)
No leaderboard entry stores the SegNet **argmax partition as a contour-coded region adjacency graph,
solved against the SegNet margin polytope (a system of linear inequalities), with a runtime-less native
rasterizer**, treating d_seg as a combinatorial set functional and d_pose as a separate global smooth
budget. It is a new PROBLEM FORMULATION (task-conditioned MDL under a frozen composed oracle), not a new
codec — exactly lever A realized via the right data structures. The recoded-R3 borrowed-substrate
problem (NO-FAKE class 7) does not arise: every byte here is ours-original by construction.

## 8. Reuse map (SEARCH-FIRST, verified present — NO-FAKE: only claims the code honors)
- **connected-components + exact contest-unit region pricing**: `tac.analysis.hinerv_hard_region_miner`
  + `tac.substrates.hi_nerv.target_region_birth` (real: ranks worst connected target-class regions in
  contest score-debt units).
- **small-margin boundary-pixel identification + repair pricing**: `tac.optimization.frame1_seg_repair_atoms`
  (real: Class-3 repair atoms at small-margin boundary pixels — the per-pixel margin surface; we replace
  its *atom search* with the *polytope solve* but reuse the margin/pricing).
- **SegNet margin field + Jacobian atlas** (the polytope coefficients): `tac.optimization.evaluator_response_atlas`
  + the flip-map (#35/#36/#51).
- **bitmask transforms** (d_seg popcount, contour/RLE coding): `tac.lossless.transforms`,
  `tac.codecs.stc_dasher`.
- **MDL accounting scaffold** (partial; section-entropy only, scorer-conditioned layer is an explicit
  PROXY): `tac.analysis.scorer_conditional_mdl` — usable for byte accounting, NOT as the solver or
  authority.
- **invisibility basis / resize-null** (the free interior subspace): `tac.optimization.resize_null_preimage`
  + `evaluator_invisibility_basis` (#47/#49).
NEW pieces to build: the region adjacency graph, the boundary contour codec, the per-pixel
margin-polytope free-budget LP, the MDL region-merge cut solve, and the grammar/kernel-or-Zig rasterizer.

## 10. The waterfilling allocation layer — "right things in right places, optimized against evaluate.py"
(operator, 2026-06-10: *"the problem is how to do the right things in the right places mathematically
optimized against evaluate.py using waterfilling and all related techniques."*)

The boundary-math (§4) gives EXACT per-location marginals; waterfilling allocates every byte to the
location with the steepest score-reduction-per-byte, until the marginal equalizes at the **water level
λ\***. The key: evaluate.py's rate term FIXES the water level analytically.

**The water level is the rate price.** `S = 100·d_seg + sqrt(10·d_pose) + 25·B/D`. Each byte costs
`∂S/∂B = 25/D = 25/37,545,489 = 6.66e-7` score/byte. So **λ\* = 6.66e-7 score per byte** — an action pays
rent iff its distortion-score-reduction per byte exceeds λ\*. This IS the KKT stationarity condition of
evaluate.py: at the optimum every funded action has marginal = λ\* (the water level); under-water actions
are dropped, over-water actions are funded in steepest-first order.

**Per-axis marginals fed into the waterfiller:**
- **seg (linear, `100·d_seg`):** one flip is worth `100/N = 100/(600·196608) = 8.48e-7`. Fixing a flip
  pays rent iff it costs **< (100/N)/λ\* = 1.27 bytes/flip**. ← This is EXACTLY the naive-sidecar
  break-even; our measured 1.525 B/flip failed because it sat ABOVE the water level. ⟹ the MDL
  region-merge SOLVE (§4): **keep a region iff its boundary-contour bytes < (interior flips it avoids)·1.27**;
  else merge/drop it. ⟹ lever D (STC/UNIWARD contour coding below 1.27 B/flip) is the ONLY way seg-repair
  beats the water level — coding-theoretic efficiency is not optional, it is the gate.
- **pose (concave, `sqrt(10·d_pose)`, GLOBAL pool):** marginal `∂S/∂d_pose = 5/sqrt(10·d_pose)` GROWS as
  d_pose↓ (the documented crossover: below pose_avg≈2.5e-4 pose marginal exceeds seg's). Pose is a single
  global budget (pooled-before-sqrt ⟹ a pose move on any pair trades 1:1). Waterfill pose-correction
  bytes onto the pairs with steepest `Δd_pose/Δb` until `5/sqrt(10·d_pose)·(Δd_pose/Δb) = λ\*`.
- **rate (linear):** the price of water itself; the lossless floor (decoder 98.6% iid Shannon, latents
  MI=0) is the rate already paid.

**The allocator is the meta-Lagrangian/Pareto solver (already partly built: #28/#30/#36 "evaluator-action
waterfiller", the atlas engine, the V3 exact-ΔS waterfiller).** The reframe FEEDS it CLOSED-FORM
marginals from the polytope + contour codec + pose budget instead of swept estimates ⟹ the waterfilling
becomes a SOLVE (KKT: equalize marginals at λ\*), not a sweep. "Related techniques": Lagrangian/KKT
duality (λ\* is the dual variable), convex feasibility (Dykstra alternating projections onto the
seg/pose/rate constraint sets), optimal transport (the pose global-budget reallocation across pairs),
and the Pareto frontier (the achievable {d_seg, d_pose, B} surface).

## 9. Routing
- Lever G (running, `a4dbf7fd…`) → REDIRECTED from rule-family search to: build the boundary-math
  representation (RAG + contour + bitmask + margin-polytope free-budget) and the SOLVE (per-pixel LP +
  MDL region-merge), measured on the exact scorer. This is the offensive carrier's seg core.
- Lever F (info floor) → the **seg floor = boundary-contour entropy of L***, the **pose floor =
  trajectory entropy of p***; the closed spec means T_floor is computable from {M,E,V}. F consumes the
  contour entropy this build produces.
- Carrier-runtime (grammar/kernel vs runtime-less Zig) → design lane after the seg-core solve proves the
  contour-partition lands in the cell at target d_seg on the exact scorer.
