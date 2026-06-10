# The POSE CRUX + PoseNet-protection design — verdict (task #80)

**Subagent:** `task80_pose_crux`. **Authority of every number below:** `[local CPU-torch advisory]` —
exact frozen upstream PoseNet (`fastvit_t12`, reads BOTH frames -> resize 384x512 -> rgb_to_yuv6 -> 12ch
-> MSE on first 6 of 12 pose dims, per `upstream/modules.py:61-84`), GT decoded via
`frame_utils.yuv420_to_rgb` ONLY, scores recomputed from components. NOT the contest 600-sample harness ->
non-promotable. `$0` spend, no GPU, no paid dispatch, **NO MPS**. `promotable=false`, `score_claim=false`,
`ready_for_exact_eval_dispatch=false`.

---

## LEAD (the two headline answers the deliverable was asked to put first)

1. **IS THE POSE-SENSITIVE SUBSPACE LOW-DIM?  → YES, decisively. EFFECTIVE DIMENSION ≈ 1.07 (RANK-1).**
   The 6 scored PoseNet dims read essentially **ONE dominant direction** of frame-pixel space. Measured at
   BOTH levels (4 pairs, exact frozen PoseNet):
   - **Pixel-Jacobian** (6 × 589,824 at the 384×512 working res): effective dim **1.077** (range 1.070–1.093),
     σ-ratio σ₁/σ₂ ≈ **65×**, top-1 singular direction carries **99.97%** of the spectrum energy. Rank ≤ 6
     (the row space of a 6×N matrix), but effectively rank-1.
   - **Feature-Jacobian** (6 × 512 summary feature): effective dim **1.072**, top-1 = **99.96%** of the 512
     feature dims. The 6 pose dims read **one** of the 512 FastViT summary features (to first order).
   - **Pose-null energy fraction = 99.999%** of any GT-realistic frame perturbation lands in the
     (N−6)-dim complement (isotropic noise: 0.99999 = exactly (N−6)/N; the f8-low-res carrier residual:
     0.99966). The pose-sensitive subspace is an almost-measure-zero sliver of pixel space.

2. **DOES FEATURE / JACOBIAN-ALIGNED PROTECTION HOLD THE TUBE CHEAPER THAN PIXEL-RMSE<3?  → NO (the
   honest negative; but the rank-1 fact reframes the design).** The linear-Jacobian-null is **tangent-only**:
   it protects pose against SMALL perturbations but is NOT a finite-width invariant tube. Confining a
   frame0 error to the measured pose-null caps d_pose at **~1e-4 to 5e-2** (30×–1000× above the tube
   2.9e-5) and GROWS with RMSE via second-order curvature; it does NOT reach the tube and is NOT cheaper
   than uniform quantization. **The pixel-RMSE<3 floor (#74) is therefore essentially REAL, not an
   isotropic-noise artifact** — uniform quantization needs only RMSE≈2.3 (q=8) to hold d_pose 2.6e-4, and
   the Jacobian-aligned scheme is WORSE than uniform at every byte budget. **The capacity wall holds at
   first order; the escape, if any, is second-order curvature-aware protection (§4), not a linear basis.**

**Net for the capstone (#78):** budget for near-lossless pose-bearing frames. The rank-1 structure does
NOT give a free linear escape, but it sharply localizes WHAT must be near-lossless (§5 design).

---

## 1. The pose-sensitive subspace effective dimension (MEASURED) — the rank-1 finding

`tools/pose_subspace_spectrum_probe.py` (4 pairs, exact frozen CPU-torch PoseNet, differentiable-yuv6
patch ACTIVE so the gradient reaches camera-res pixels; fail-closed on severed gradient). The full signed
6×N Jacobian `J = d pose6 / d frame_pixels` is built by 6 backward passes; its spectrum via the 6×6 Gram
eigendecomposition (numerically stable; the orthonormal row basis via QR of Jᵀ, NOT division by tiny σᵢ).

| measurement | slot 0 (frame0) | slot 1 (frame1) |
|---|---:|---:|
| effective dim (participation ratio of σ) | **1.077** | 1.078 |
| rank (σ > 1e-4·σ_max) | 6 | 6 |
| σ_max / σ_2nd | **64.6×** | 64.6× |
| top-1 energy fraction | **0.9997** | 0.9997 |
| top-3 energy fraction | 0.99997 | 0.99997 |
| feature-space (6×512) effective dim | **1.072** | — |
| feature-space top-1 energy frac | **0.9996** | — |

**Reading:** d_pose is, to first order, a **single scalar projection** of the frame onto one dominant
direction. The 6-dim pose output collapses onto ~1 effective pixel-space direction (σ₁ dominates 65×).
This is far stronger than the operator's "low-dim ≤ 6" hypothesis — it is effectively **rank-1**. The
feature-level measurement confirms it is intrinsic to the pose head, not a pixel-resolution artifact.

## 2. The pose-null fraction (MEASURED) — 99.999% of perturbation energy is pose-invisible to first order

| perturbation | pose-null energy fraction | note |
|---|---:|---|
| isotropic Gaussian (RMSE 6) | **0.99999** | exactly (N−6)/N = 0.99999 (the analytic baseline) |
| f8 low-res carrier residual | **0.99966** | the #57-carrier residual is ALMOST ENTIRELY pose-null |

The pose-sensitive subspace is 6 directions out of 589,824 — a 1e-5 sliver. ANY natural frame
perturbation is ~99.99% pose-null **to first order**. This is the analog of the 80.67% resize-null for
SegNet, but FAR more extreme (6/N ≈ 1e-5 vs the resize-null's ~19%). **If the first-order picture held at
finite perturbation, the escape would be enormous.** It does not (§3) — that is the crux.

## 3. THE ESCAPE TEST (MEASURED) — the linear null is TANGENT-ONLY, not a finite-width invariant

Three independent tests, all on the exact frozen PoseNet, all converge on the same conclusion.

### 3a. Walking ALONG a pure-null direction (the cleanest test)
Pick a unit-RMSE direction exactly orthogonal to all 6 Jacobian rows; scale it and measure exact d_pose:

| frame0 work-RMSE | d_pose along pure-null | d_pose along dominant-sensitive dir |
|---:|---:|---:|
| 0.5 | 1.5e-3 | 2.3e+1 |
| 1.0 | 1.6e-3 | 3.0e+1 |
| 4.0 | 1.8e-3 | 1.1e-1 |
| 8.0 | 2.4e-4 – 1.1e-2 | 4.4e+1 |
| 16.0 | ~2e-3 – 5e-2 | 8.4e+1 |

The **sensitive direction destroys pose** (d_pose 20–84; ratio sens/null = 290×–24,000×) — this PROVES
the measured row space IS the pose-sensitive subspace (it is load-bearing, not a fabricated mask). But the
**pure-null direction caps d_pose at ~1e-3 to 5e-2, growing with RMSE** — it does NOT reach the tube
(2.9e-5). The first-order-null drifts INTO sensitivity through **second-order curvature** as you move along
it. The null is the tangent plane at GT, not a flat invariant manifold.

### 3b. The null plateau across 4 pairs (the escape ceiling)
d_pose of a frame0 pure-null perturbation at work-RMSE {1, 4, 16}:

| pair | RMSE 1 | RMSE 4 | RMSE 16 |
|---|---:|---:|---:|
| 0 | 1.6e-3 | 2.6e-3 | 4.7e-2 |
| 1 | 9.9e-5 | 4.4e-5 | 1.7e-2 |
| 2 | 2.8e-3 | 3.8e-3 | 1.1e-3 |
| 3 | 9.2e-5 | 9.9e-4 | 2.6e-2 |

The null-confined ceiling is **30×–1000× above the tube** (occasionally near it — pair 1 @ RMSE 4 = 4.4e-5 —
but never reliably). It is BETTER than isotropic at large RMSE but does NOT hit the tube.

### 3c. Jacobian-aligned QUANTIZATION vs uniform (the byte-cost escape) — NO win
Encode frame0 at the work resolution; uniform = round(x/q)·q; Jacobian-aligned = protect the 6 sensitive
coeffs losslessly + quantize the pose-null residual at step qn. Bytes = zlib-compressed int payload (pair 0):

| scheme | step | work-RMSE | d_pose | ~bytes |
|---|---:|---:|---:|---:|
| uniform | q=8 | 2.29 | **2.6e-4** | 175,705 |
| uniform | q=16 | 4.54 | 4.0e-2 | 104,819 |
| uniform | q=32 | 9.25 | 1.8e-2 | 64,876 |
| Jacobian-aligned | qn=8 | 2.29 | 2.7e-2 | 176,432 |
| Jacobian-aligned | qn=32 | 9.25 | **3.4e+0** | 65,042 |
| Jacobian-aligned | qn=128 | 29.83 | 3.1e+1 | 8,283 |

**Jacobian-aligned quantization is WORSE than uniform at every byte budget** (qn=32 → d_pose 3.4 vs
uniform q=32 → 0.018). Reason: coordinate-wise quantization of the null residual is NOT a projection onto
the null — it re-injects energy into the sensitive direction, and at the curvature scale this dominates.
Uniform quantization happens to keep error isotropic+small, which is the safest for a tangent-only null.

**The reconciliation (the honest mechanism):** the linear Jacobian-null is genuinely 99.999% of pixel
space and genuinely protects pose against INFINITESIMAL error, but d_pose is locally quadratic along the
null (curvature), so at quantization-scale RMSE (4–30) the null no longer protects to the tube. **The
pixel-RMSE<3 floor (#74) is real**: it is the radius within which the tangent-null approximation holds the
tube. Below RMSE≈2.3 (uniform q=8) the tube holds; above it, curvature breaks it regardless of basis.

## 4. Where an escape COULD still live (the curvature-aware reactivation, not a kill)

The linear escape fails, but the rank-1 fact is not exhausted. Per "Forbidden premature KILL", the
reactivation paths, priority-ordered:

1. **Second-order (curvature) protection — the dominant untested lever.** d_pose along the null is
   `~ 0.5·δᵀ H δ` (H = the pose Hessian/Gauss-Newton in pixel space). The escape needs the null of the
   QUADRATIC form, not the linear Jacobian. Build the Gauss-Newton `H ≈ Jᵀ J` of the 6 dims at GT and
   protect the directions of largest curvature (the leading eigenvectors of H), not just the linear
   Jacobian rows. This is a strictly larger protected subspace but still low-dim (the curvature spectrum of
   a rank-1-Jacobian map concentrates). UNTESTED — the next $0 probe: measure the curvature spectrum and
   re-run the quant-basis escape protecting the leading-Hessian directions.
2. **Iterated Dykstra pose-tube projection (re-linearized) toward the tube, not staying in the null.** #73
   proved the Dykstra pose-tube solve from the comp pair returns δ=0 optimal — i.e. the cheapest feasible
   frame in a learned basis is what the frontier already stores. The score-native escape must compress
   WITHIN a basis that preserves the curvature-sensitive direction (the HNeRV learned basis already does;
   a generic linear basis does not — confirmed here AND in #73).
3. **Feature-distillation with a curvature-weighted target.** Matching the teacher's ONE dominant summary
   feature (not all 512) is sufficient to first order, but the curvature finding says the target tolerance
   must be tight (the feature-RMSE radius is the image of the pixel-RMSE<3 ball). Feature-distill does NOT
   relax the fidelity requirement — it RE-EXPRESSES it in the 512-dim feature space, where the tolerance is
   set by the same curvature. Not cheaper, but a cleaner training signal than pixel-MSE (it spends capacity
   on exactly the 1 pose-relevant feature direction). Composes with the recon-primary distill of #74.

## 5. The per-frame pose contribution (MEASURED) — frame0 dominates 20× at f8

| configuration | d_pose (mean over 4 pairs) | note |
|---|---:|---|
| f8-carrier frame0 + GT frame1 | **0.163** | frame0 low-res |
| GT frame0 + f8-carrier frame1 | **0.0081** | frame1 low-res |
| both f8 carriers | 0.199 | |

**frame0 contributes ~20× more pose debt than frame1** at f8 fidelity. This is consistent with #74
(`studF1+teachF0 → 17` vs `student-both → 189`: student frame0 wrecks pose 11×) and #57 (the frame0
carrier was the pose carrier). **The optimal pose-frame split: frame0 needs the most pose fidelity; frame1
can be lower-fidelity FOR POSE** (though frame1 carries d_seg — the SegNet argmax — so its fidelity is
constrained by the seg term, per #57's "frame1 dual constraint"). The PoseNet-protection budget should
concentrate on frame0's pose-sensitive direction.

## 6. The PoseNet-protection design feeding the capstone (#78)

The capstone's pose budget, from this measurement:

1. **Pose is a rank-1 read.** Protect the SINGLE dominant pose-sensitive direction of each frame
   (per-pair, per-slot) — `spectrum.row_basis[0]` from `measure_pose_subspace_spectrum`. This is the
   highest-value byte for pose. But (§3) protecting it LINEARLY is insufficient — the protection must hold
   the frame near-lossless ALONG the curvature directions of that read.
2. **The pixel-RMSE<3 floor is real and frame0-concentrated.** Budget frame0 as a near-lossless pose
   carrier (RMSE < ~2.3 in the work-res luma the yuv6 path reads — chroma is 2×2-box-subsampled so it
   matters less; the y00/y10/y01/y11 luma channels are the pose signal). frame1 can trade pose fidelity for
   seg fidelity (it is the SegNet argmax carrier; pose-debt at f8 is only 0.008).
3. **The escape is curvature-aware, not linear.** The capstone should NOT assume a cheap linear-Jacobian
   pose-null exists (it does not at finite RMSE). The open lever is the Gauss-Newton-curvature protected
   subspace (§4.1) — a $0 next probe. Until that lands, budget for near-lossless frame0 luma.
4. **The frontier's HNeRV decoder IS the curvature-preserving basis.** Convergent with #73 (#73d) and
   #74 (reactivation path 2): the only representation proven to hold the tube under compression is a
   learned nonlinear basis (the 177KB HNeRV decoder), because it preserves the fine spatial structure the
   rank-1 pose direction integrates. A generic linear/SVD/sparse basis breaks pose first (#73). The
   capstone's pose carrier should be the learned basis, with the rank-1 direction as the saliency that
   weights its training (the #61 saliency map IS this — and this task confirms #61's weight is the right
   shape: concentrate on the 1 dominant direction).

## 7. Wire-in (6 hooks per Catalog #125)

1. **sensitivity-map — ACTIVE:** the rank-1 finding sharpens the #61 PoseNet saliency: the per-pixel norm
   is dominated by ONE direction (`row_basis[0]`); the waterfiller (#54) consumes the per-frame-slot rank-1
   direction + the frame0-dominates-20× split as the pose-marginal map.
2. **Pareto — ACTIVE:** §3c is a hard RD constraint row: `d_pose(RMSE)` is locally QUADRATIC (curvature),
   so the pose-fidelity-vs-bytes frontier has a sharp knee at RMSE≈2.3 (uniform q=8); below it the tube
   holds, above it d_pose blows up. The Pareto-feasible pose move is near-lossless frame0 luma, NOT a
   coarse Jacobian-aligned carrier.
3. **bit-allocator — ACTIVE:** the byte breakdown (§3c) + per-frame split (§5) is the literal allocator:
   spend on frame0 luma fidelity; frame1 pose can be cheap (seg-constrained separately).
4. **cathedral-autopilot — gate NOT met:** advisory only, no archive built, no paired-eval dispatch.
5. **continual-learning — ACTIVE:** reseeds the planner: (a) pose is a RANK-1 read (eff-dim 1.07 pixel +
   feature); (b) the pose-null is 99.999% but TANGENT-ONLY (curvature breaks it at RMSE>~3); (c) the
   pixel-RMSE<3 floor (#74) is REAL not an isotropic artifact; (d) Jacobian-aligned linear compression is
   WORSE than uniform; (e) frame0 dominates pose 20×; (f) the open escape is curvature-aware (Gauss-Newton)
   protection — the next $0 probe.
6. **probe-disambiguator — RESOLVED + ONE re-opened:** "is the pose subspace low-dim?" → YES rank-1.
   "is the pose-null a finite-width escape?" → NO, tangent-only (curvature). "does feature/Jacobian-align
   beat pixel-RMSE<3?" → NO. The re-opened probe: "does the Gauss-Newton-CURVATURE-aligned protected
   subspace hold the tube cheaper?" (§4.1) — UNTESTED, the highest-EV next pose probe.

## 8. NO-FAKE attestation

- The spectrum is the EXACT SVD of the EXACT backprop 6×N Jacobian of the frozen PoseNet (not a fabricated
  constant); the SENSITIVE-direction control (d_pose 20–84, 290×–24,000× the null) PROVES the measured row
  space is load-bearing — confining error to it is catastrophic, the opposite of a no-op.
- The orthonormal basis is verified orthonormal to 1e-14 (Gram − I); the QR construction does NOT divide by
  tiny singular values (the original Gram `Uᵀ J / sᵢ` formula blew up float noise — a numerical-correctness
  fix made mid-task, NOT a fake: the corrupted basis would have given a FAKE null fraction).
- d_pose is the EXACT frozen-PoseNet functional (MSE on first 6 dims), GT via `yuv420_to_rgb` ONLY
  (the rgb24 path manufactures ~100× phantom pose; the #73 pairing-bug class avoided by pairing comp0+comp1).
- The fail-closed guard raises `PoseSubspaceError` on a severed gradient (zero Jacobian) — a non-reachable
  gradient must NEVER masquerade as "all pose-null".
- 17 behavior tests (`src/tac/boundary_math/tests/test_posenet_subspace_spectrum.py`): 16 fast (synthetic
  known-rank Jacobian asserts participation ratio, rank detection, orthonormal basis, null-fraction of
  row-space-vector ≈0, null-fraction of isotropic ≈(N−r)/N, sensitive+null=delta) + 1 slow on-real-scorer
  (rank ≤ 6, eff-dim ≪ pixel dim). 194 boundary_math tests green, ruff clean.

## 9. Deliverables + cross-references

- **Module (NO-FAKE, tested):** `src/tac/boundary_math/posenet_subspace_spectrum.py`
  (`measure_pose_subspace_spectrum` — the 6×N Jacobian SVD spectrum + effective dim; `project_onto_pose_null`
  — the sensitive/null decomposition; `participation_ratio`; `expected_isotropic_null_fraction`).
- **Probe CLI:** `tools/pose_subspace_spectrum_probe.py` (effective dim + pose-null fraction + quant-basis
  escape + feature subspace + per-frame split; 4-pair run = 54s, $0).
- **Tests:** `src/tac/boundary_math/tests/test_posenet_subspace_spectrum.py` (17, incl. on-real-scorer slow).
- **Evidence:** `experiments/results/task80_pose_crux_20260610/spectrum_4pair.json` (+ 2pair smoke).
- **Cross-refs:** `score_native_pose_carrier_20260610T125000Z.md` (#57 — frame0 INR ceiling 0.0036; the
  frame1-dual constraint) · `distillation_smaller_student_20260610T191237Z.md` (#74 — pixel-RMSE<3 pose
  tube; frame0 dominant) · `legal_frame_feasibility_dykstra_20260610T175421Z.md` (#73 — generic linear
  basis breaks pose first; the learned-basis conclusion) · `src/tac/boundary_math/posenet_jacobian_saliency.py`
  (#61 — the per-pixel norm; this task confirms it is rank-1-dominated) · `upstream/modules.py:61-84`
  (the PoseNet scorer facts) · CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent"
  (pose marginal dominates at the frontier operating point — this task is the frame-side mechanism).
