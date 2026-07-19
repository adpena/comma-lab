# The v10 solve, flattened + factorized against upstream modules.py — the explicit Lagrangian + KKT structure (2026-07-19)

**Operator directive (verbatim):** *"It's all solvable, it's not intractable... It's all deep math and
engineering and geometry, optimized against upstream modules dot py lagrangian jacobian channels and
hyperplane, flattened and factorized."* This memo IS that derivation — the whole remaining game written
as ONE constrained program in the flattened coordinates the frozen-scorer factorization gives us.
Pointer `0.19108 [contest-CPU]` UNMOVED; this is the equations-leg (DERIVED, anchored to MEASURED
constants; no score claim). STORES CONSULTED: `frozen_scorer_exact_factorization_20260715` ·
`segnet_recursive_fractal_factorization_20260715` · `joint_seg_pose_rate` (band law) ·
`pose_plane_proximity_corollary_v1` · `yhat_rd_ladder_20260719_codex` · `null_subspace_rate_20260717`.

## 1. Coordinates (the flattening — all MEASURED facts)

Both scorers read the SAME plane: shared bilinear `A: R^(874×1164×3) → R^(384×512×3)` (SegNet reads
last-frame plane; PoseNet resizes FIRST with the same A then yuv6, `upstream/modules.py:71-75`). The
factor-2 lattice solve realizes ANY target plane exactly in uint8 camera space (n600: d_seg 9.66e-7,
all ULP-class; free at decode, rule-118). Therefore the ONLY optimization variables are the **plane
residuals** vs the source planes:

```
r_0, r_1 ∈ R^N,  N = 384·512·3 = 589,824 per frame,   ŷ_i = y_i + r_i
```

Camera-space realization, uint8 survival, and R-roundtrip walls are OUT of the problem (0 by
construction — the #548 measured fact). ker(A) (52% of camera energy, scorer-invisible) never enters:
we optimize on the plane and let the lattice solver pick any preimage — range-only byte counting is
already the law (anti-min-norm; the arbitrary-fill entropy waste is the measured 1.70MB/frame
rate-death of the direct payload).

## 2. The objective (rate — the only term that costs bytes)

```
R(r_0, r_1) = bytes( Q(r_0, r_1) vs free decode-side predictor )
```

The predictor (previously-decoded frame / generated fill / smooth interpolant of the stored
description) is generic code — FREE under rule 118. Only the residual description is counted. Proxy
for solving: weighted entropy/ℓ1 of quantized residuals; authority: actual brotli-Q11/zstd-19 bytes
(never the proxy alone).

## 3. The constraints (both scorers, first-order in r around the SOURCE plane)

**Seg (per-pixel hyperplanes — HIGH codimension).** The frozen head is EXACT rank-4 linear
(`segnet_head_rank4_linear_flipdist_v1`; SVs 3.128/2.154/2.025/1.796; Lane winner/rival normals up to
‖Δw‖≈4.01). For plane pixel p with cached winner c* and rival c′ (margins `m_p` cached in gt_n600):

```
g_{p,c′}(r_1)  =  q_{p,c′} · r_1  ≥  −m_{p,c′},      q_{p,c′} = ∇_ŷ [logit_{c*} − logit_{c′}](y_1)
```

The `q` vectors ARE the VJP-custody sidecars (unit hyperplane normals pulled back through the frozen
net — deliverable 1 of the running arm). Validity: first-order holds within the Lipschitz radius; the
band law `r_ch = min(r_max, d_feat/(3·Lip_local·|q_ch|))` is exactly the per-channel box
inner-approximation of this half-space system, and the f32 hard oracle stays the final authority (the
registered receiver-arithmetic law).

**Pose (ONE 6-dim ellipsoid — codimension SIX).** With `J_i = ∂P/∂ŷ_i |_(y_0,y_1) ∈ R^(6×N)` (the
frozen PoseNet Jacobian — deliverable 2 of the running arm):

```
h(r_0, r_1)  =  (1/6) ‖ J_0 r_0 + J_1 r_1 ‖²  ≤  τ_pose
```

## 4. The Lagrangian and the KKT theorem (why pose falls out — CODIMENSION, not luck)

```
L(r, λ, μ) = R(r) + Σ_{p,c′} λ_{p,c′} ( −q_{p,c′}·r_1 − m_{p,c′} ) + μ ( h(r) − τ_pose )
```

- **Stationarity:** ∂R/∂r = Σ λ q − (μ/3) Jᵀ(J r). Bytes flow along ACTIVE constraint normals — the
  duals λ ARE the costate field (the #247/#426 organ's object, now exact at this surface).
- **Complementary slackness (seg):** λ_{p,c′} > 0 only where the constraint is tight — the low-margin
  annulus. MEASURED corroboration: 96.9% of c2 residual flips sit in the annulus; the flip-prone set
  is a few % of pixels. Bytes concentrate exactly there; everywhere else the polytope is slack and the
  residual is pure predictor (free).
- **THE POSE THEOREM (the reframe, now derived):** the pose constraint restricts only the 6-dim
  subspace `rowspace(J) ⊂ R^(2N)` — codimension 6 out of 1,179,648. Its dual μ prices ONLY the
  component of r along Jᵀ; the orthogonal complement (dim 2N−6) is pose-free. With any sane predictor
  keeping ‖r‖ small, `h(r) = O(‖J r‖²)` starts ~0 ⇒ **μ = 0 at the optimum (pose inactive) unless the
  seg-driven residual has measurable projection onto rowspace(J)** — and even then, correcting it
  costs at most a rank-6 adjustment, O(6) numbers, not O(N) bytes. The measured rows are this theorem's
  instances: in-band planes → d_pose 5.35e-10..1.14e-9; the c2-witness plane (‖r‖ RMSE 25, huge
  Jᵀ-projection) → 63. Asymmetry in one line: **seg is ~10⁵ hyperplanes, pose is 6 numbers.**
- **Score-marginal closure:** dS = 100·d(d_seg) + (5/√(10·d_pose))·d(d_pose) + (25/37,545,489)·d(bytes).
  With μ=0 generically, the #536 KKT waterfill is effectively TWO-term (seg-band-scale vs bytes) — a
  scalar trade along the band law, with pose re-entering only through the rank-6 correction at
  τ_pose ~ 2.5e-4 (the crossover). This is why the waterfill looked "flat" at 1 admissible point: the
  curve IS low-dimensional; it needs band-scale sweeps (directed to the arm), not more pose points.

## 5. Channels (the per-channel anisotropy — MEASURED structure, free wins)

- Seg reads RGB fully; the band width per channel scales as 1/|q_ch| (anisotropic box).
- PoseNet chroma passes a 2×2 box filter (<2px chroma invisible, `frozen_scorer_exact_factorization`)
  ⇒ chroma rows of J are ~half-band ⇒ chroma residual is CHEAPER on the pose side by construction;
  luma (BT.601) carries the pose-sensitive component. The channel-split budget falls out of the same
  KKT system — no separate chroma heuristic needed.

## 6. What each running/landed piece supplies to this ONE program

| piece | role in the program |
|---|---|
| VJP-custody arm (running) | the constraint matrix: q rows (seg) + J (pose) + Lip field → first positive-band curve points |
| #548 ladder (landed) | objective calibration endpoints: 83,838B generator row · direct-plane rate-death · exactness-free realization |
| #543 receiver (landed) | the free decode: parse → expand description → lattice-realize; deterministic, scorer-free |
| f32 law (registered) | admissibility of every exactness claim (hard oracle or margin > κ·μ_ULP) |
| #536 waterfill | solves THIS KKT system on the measured curves (now knowing it is ~2-term) |
| #541 constructive solve | the production solver of this program: min R(r) s.t. polytope ∩ ellipsoid |

**Solvability verdict (the operator's point, now structural):** first-order in the residual, this is a
CONVEX program — linear inequalities ∩ one convex quadratic, minimized under a convex compressibility
surrogate — with the nonlinearity quarantined into (a) the Lipschitz validity radius (band law) and
(b) the final hard-oracle check. Nothing here is intractable; every term is a frozen, factorized,
measured object from upstream modules.py. The remaining work is engineering the constraint custody
(arm), sweeping the band scale (arm), and solving the LP/QP (existing machinery class).

Triality: equations-leg = this memo + `pose_plane_proximity_corollary_v1` + band law + rank-4 head law
· DAG = FEED-pose-falls-out (cross-ref this memo) · DSL = no new lever (design SoT for #541's
objective/constraints). launch_ready unchanged; MEANS toward the exact row.

---

## 2026-07-19 fresh-eyes verification corrections (append-only; original text above preserved)

Source: delegated adversarial verification
`.omx/research/spec_v10_reconciliation_and_kkt_verify_20260719_fable.md` (findings V-2..V-6; every check
re-derived from primary artifacts). The derivation's skeleton STANDS; the following statements above are
corrected in place of being re-cited:

1. **§4 "costs at most a rank-6 adjustment, O(6) numbers, not O(N) bytes" — over-strong as a BYTE claim
   (V-2).** The pose constraint restricts 6 DOF (sound). But a decode-side rank-6 expansion `δ = Jᵀc`
   would need the rows of `J` — PoseNet-derived (no scorers at inflate) AND evaluated at the source plane
   (video-derived ⇒ COUNTED at 6×N scale if shipped). The correction is therefore 6 DOF applied at ENCODE
   time; its BYTE cost through the description grammar is an OPEN EMPIRICAL quantity = the `bytes(τ_pose)`
   curve the VJP arm owes. Read "O(6) numbers" as a DOF statement only.
2. **§4 "μ = 0 at the optimum ... generically" — genericity is unearned (V-3).** Seg-driven residuals
   concentrate on the annulus (MEASURED 96.9%) and PoseNet Jacobian rows also concentrate on structured
   content — two edge-concentrated objects are not in general position. The near-source measured rows
   (5.35e-10..1.14e-9) are the trivial r→0 limit (they instantiate continuity, not J-orthogonality of
   seg-shaped residuals); the only measured large-r row (c2 witness, RMSE 25) had d_pose 63 — a huge
   J-projection. Pose inactivity at IN-BAND solutions is a PRE-REGISTERED FALSIFIABLE PREDICTION
   (`pose_plane_proximity_corollary_v1`), decided by the bindingness harvest — not a theorem instance.
3. **§3 "exactly the per-channel box inner-approximation of this half-space system" — drop "exactly"
   (V-4).** The implemented box budgets, per pixel, only that pixel's own 3 channels against its own
   hyperplane. The true constraint at p couples its receptive field (MEASURED ERF r50≈85px): simultaneous
   in-box perturbations of many pixels can accumulate at p beyond its diagonal budget. The box is the
   inner approximation of each single-pixel DIAGONAL slab; JOINT validity across the ~10⁵ overlapping
   constraints is measured only by the positive-band bindingness/repair-rate harvest; the frozen hard
   oracle stays the sole authority. Also: `Lip_local` is CONFIGURED custody, not yet measured — no
   positive-band operating point exists (the #549 run measured only the zero-band control), and pair-125
   (1 native-f32 flip at ZERO perturbation, d_seg 5.09e-6) shows the fp32 ULP-tie floor beneath the
   linear model at every band scale (the f32 admissibility law is load-bearing).
   Margin-scale corroboration (MEASURED, gt_n600): median margin 5.89, annulus fractions 2.67% < 1.0 /
   0.28% < 0.1 — bytes DO concentrate on a few % of pixels, as complementary slackness predicts.
4. **§6 "this is a CONVEX program" — one more quarantine item (V-5).** Besides (a) the Lipschitz radius
   and (b) the hard-oracle check, add (c): the description variables are quantized/integer (`Q(·)`,
   integer numerators) — the LP/QP claim holds for the continuous relaxation; the integer/repair leg is
   combinatorial and lives with the oracle.
5. **§4 "the #536 KKT waterfill is effectively TWO-term" — conditional, not settled (V-6).** 2-term
   PENDING the bindingness harvest (item 2); if pose binds it re-enters as a low-dimensional (rank-≤6)
   third term whose byte price is MEASURED per item 1, never assumed.

Everything else re-derived and VERIFIED (V-8): N/2N arithmetic · dS closure + 2.5e-4 crossover · SVs +
‖Δw‖ ≤ 4.01 · 96.9% source · ker(A)~52% · 1.70 MB/frame rate-death · 83,838 B rung · chroma 2×2-box.
Related arithmetic fix (V-1, lives in the law module / SPEC / DAG, not this memo): the slack from
5.35e-10 to the 2.5e-4 crossover is **~5.7 orders of magnitude**, not "~9 orders."
