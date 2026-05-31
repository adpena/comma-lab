# Joint P18/P19 gradient-driven water-filling solver — architecture

Date: 2026-05-31
Status: architecture / pre-empirical (no score claims; `[prediction]` bands only)
Lane (to register): `lane_joint_p18_p19_gradient_waterfill_solver_20260531`
Canonical equation (planned): `joint_segnet_posenet_gradient_waterfill_score_savings_v1`
  <!-- # FORMALIZATION_PENDING: pre-empirical architecture memo; equation registered
       with first real anchor when the M1 one-shot solver produces a paired CPU+CUDA
       measurement on a byte-closed archive. No predicted-vs-measured claim here. -->

## Operator source (verbatim, 2026-05-31)

1. "we can't forget about posenet as well"
2. "The new gradient/water-fill path should become joint P18/P19, not SegNet-only:
   SegNet gives the large argmax-flip surface, while PoseNet needs null-subset
   detection and Mahalanobis/AIL-style weighting so rate savings are not spent
   into pose-sensitive pairs."

This supersedes the implicit SegNet-only framing of the detector-informed
`recon_pixel_weight` lever validated in `#1587` (canonical equation
`detector_informed_recon_weight_d_seg_savings_v1`, 10 anchors). That lever is
now the **P18 term only** of a two-term joint weight.

## Why joint is a correctness requirement, not a nicety

Contest score: `S = 100·d_seg + sqrt(10·d_pose) + 25·B/N`, N = 37,545,489.

The water-fill allocates a finite rate budget B across the archive's degrees of
freedom (pixels / wavelet coefficients / per-pair latents). Optimal allocation
spends rate where it buys the most score reduction per byte. The score has TWO
distortion terms with very different marginal structure:

- **d_seg term** `100·d_seg`: d_seg = per-pixel argmax-flip RATE of SegNet.
  Marginal sensitivity is the (smooth surrogate) `|∂L_seg/∂x_i|` — the dense
  full-grid SegNet saliency (≈77% nonzero, NOT a thin boundary band; codex
  Finding 2 `codex_findings_z8_pixel_driver_and_segnet_grid_premise_20260531T153038Z`
  superseded the "interiors are free" premise). This is the LARGE surface.

- **d_pose term** `sqrt(10·d_pose)`: d_pose = Mahalanobis MSE on the first 6
  PoseNet (FastViT-T12) dims. Its marginal is
  `d(sqrt(10·d_pose))/d_pose = 5 / sqrt(10·d_pose)` — the **AIL gain** — which
  BLOWS UP as `d_pose → 0`. Per CLAUDE.md's operating-point table, at PR106's
  `pose_avg ≈ 3.4e-5` the pose marginal is **2.71× SegNet's**.

A SegNet-only weight map therefore pours saved rate into pose-sensitive
pixels/pairs and tanks d_pose — the exact PR97 anti-pattern (won SegNet 65%,
LOST 0.042 by trading pose away) and the mechanism behind my own Z8 result
where `d_pose = 0.52` dominated the score. **Pose is the binding axis at the
frontier; the lever must protect it.**

## The joint per-DOF weight (the unified gradient)

For degree-of-freedom (pixel / coefficient / per-pair-latent) `i`:

```
w_i  =  100 · |∂L_seg/∂x_i|              # P18: dense SegNet argmax-flip surrogate
        + (5 / sqrt(10·d_pose)) · ‖J_pose,i‖_{Σ⁻¹}   # P19: PoseNet Mahalanobis × AIL
```

- `J_pose,i = ∂(pose_6)/∂x_i` is the per-DOF PoseNet Jacobian (6 outputs).
- `‖·‖_{Σ⁻¹}` is the **Mahalanobis** norm under the contest's per-dim pose
  normalization (inverse-variance Σ⁻¹ = the std-whitening already applied by
  `upstream/modules.py` PoseNet head). Per-dim, not isotropic.
- `(5 / sqrt(10·d_pose))` is the **AIL gain** — the score-domain reweight that
  makes pose-sensitivity dominate as the frontier tightens. Compute `d_pose` at
  the current candidate (re-evaluated per relinearization step in M2/M3 below).

### Null-subset detection (P19 core — the safe-to-spend set)

```
NULL_pose = { i : ‖J_pose,i‖_{Σ⁻¹}  <  ε_null }
```

These DOFs do not move the 6 pose dims (PoseNet's ego-motion attention does not
weight them). They are **free to coarsen / drop** — rate savings come from here
WITHOUT hurting d_pose. The operator's "rate savings are not spent into
pose-sensitive pairs" is exactly: keep bits on high-`‖J_pose‖` DOFs, harvest
bytes from `NULL_pose`. Detection options (cheapest → most faithful):

- **N1 (MLX/numpy, $0)** finite-difference probe: perturb DOF `i`, measure the
  Σ⁻¹-weighted change in the 6 pose dims over the 600 pairs. Cheap per-block.
- **N2** autograd Jacobian-vector products through a differentiable PoseNet
  (`tac.differentiable_eval_roundtrip` + patched `rgb_to_yuv6`, per CLAUDE.md
  eval_roundtrip non-negotiable — the upstream YUV6 is `@torch.no_grad()` and
  severs the pose gradient otherwise).
- **N3** learned mask co-trained with the renderer (M3 driver below).

## The water-fill allocation (Lagrangian)

Minimize `Σ_i w_i · D_i(b_i)` s.t. `Σ_i b_i ≤ B`, with high-rate quantization
distortion `D_i(b_i) ≈ σ_i² · 2^{-2 b_i}`. KKT stationarity → reverse
water-filling:

```
b_i  =  [ ½ · log2( w_i · σ_i² / θ ) ]_+
```

θ is the **water level** (the single Lagrange multiplier), solved by bisection
to hit the byte budget B. High `w_i` → more bits (preserve). `NULL_pose` DOFs
(`w_i ≈ 0`) → `b_i = 0` (max coarsen / drop). This is the canonical
reverse-water-filling, now with a **joint seg+pose sensitivity weight**.

### Lagrangian redirect / Dykstra feasibility

The single-budget water level θ is the inner solve. The **outer** solve is the
meta-Lagrangian / Dykstra alternating-projection onto the intersection of three
feasible sets `{B ≤ B₀} ∩ {d_seg ≤ S₀} ∩ {d_pose ≤ P₀}` (Catalog #372
`invoke_dykstra_pareto_solver_on_candidates`). The per-axis dual variables that
come back from Dykstra ARE the per-axis tight constraints; the binding axis
(pose, at the frontier) gets the largest dual → its DOFs are protected first.
This wires the solver into hook #2 (Pareto) of the Catalog #125 6-hook contract.

## The gradient DRIVER — the operator's "backprop and adjustment across full space"

The missing op the operator flagged: a one-shot saliency×water-fill is a *local
linearization*. Three escalating drivers close the loop:

- **M1 — one-shot** (the validated lever; ready now): compute `w_i` once at the
  baseline render, allocate, quantize. Catalog #307 paradigm-level baseline.
- **M2 — iterative reweighted relinearization**: recompute `w_i` (both terms,
  including the AIL gain at the *current* d_pose) at each quantized state, re-
  solve θ, repeat to a fixed point. Handles the AIL non-linearity (pose marginal
  shifts as the candidate moves). This is the IRLS / EM flavor.
- **M3 — STE end-to-end (δS/δθ = 0)**: backprop the REAL score `S` (through the
  actual scorers, eval_roundtrip ON, differentiable YUV6) through a
  straight-through quantizer into the per-DOF precision parameters `b_i`
  themselves → the bit allocation becomes a *learned* variable optimized by
  gradient descent on the contest scorer. This is the unified action
  `δS_total/δθ = 0` (`feedback_unified_lagrangian_action_principle_GR_style`).
  P18 + P19 are then just two terms of one `S_total`; their joint balance is
  whatever the gradient finds — no hand-tuned mixing coefficient.

The escalation is the point: M1 proves the lever, M2 fixes the AIL non-linearity,
M3 removes the last hand-set knob (the seg/pose mix) by letting the real score
arbitrate. **MLX-first**: M1/M2/N1 are MLX/numpy $0; M3 trains MLX-local; only
the final paired CPU+CUDA ratification (Catalog #246) touches Modal (≤$20).

## P18 / P19 split (lane decomposition)

- **P18** = the SegNet argmax-flip surface = `recon_pixel_weight` channel CONTENT
  (`#1590`, in flight). The channel itself is generic `(B,1,H,W)` map +
  (planned) per-pair pose-weight vector — joint-ready. If `#1590` lands a
  SegNet-only *default*, the joint map is supplied by the solver; no channel
  re-architecture needed.
- **P19** = PoseNet null-subset detector (N1/N2/N3) + Mahalanobis/AIL per-pair /
  per-dim weight. NEW work; the solver wave owns it.
- **Joint** = `w_i` above + the water-fill + the Dykstra redirect = the `#1591`
  solver. NOT SegNet-only.

## Predicted ΔS band (pre-empirical) + Dykstra-feasibility

`[prediction]` only — no measured claim. Joint band is bounded BELOW the
SegNet-only band because P19 prevents the pose-axis regression that a SegNet-only
allocation incurs. Dykstra-feasibility check: the three-set intersection
`{B}∩{d_seg}∩{d_pose}` is non-empty by construction at the current frontier (the
frontier archive already satisfies all three); the solver searches the interior
toward lower B at fixed (d_seg, d_pose) — feasible region is the frontier's
sub-level set, so the projection converges (Boyd-Dattorro alternating
projection on convex sub-level sets). Predicted per-pair pose-protected rate
recovery: `[macOS-MLX research-signal]` band to be filled by the M1 $0 sweep;
NO `[contest-*]` claim until paired ratification.

## Observability surface

- Inspectable per layer: per-DOF `w_seg_i`, `w_pose_i`, `‖J_pose,i‖`, `NULL_pose`
  mask, allocated `b_i`, water level θ — all dumpable as a 384×512 (or per-pair)
  array per candidate.
- Decomposable per signal: `w_i` splits into the two named terms; the
  score-delta attributable to each is separable (ablate P19 → recover the
  SegNet-only baseline exactly).
- Diff-able across runs: M1 vs M2 vs M3 allocations diffed per-DOF.
- Queryable post-hoc: per-DOF JSONL ledger (`w_seg`, `w_pose`, `b`, in_null).
- Cite-able: each allocation tuple anchored to (archive sha, commit, scorer sha).
- Counterfactual-able: the byte-mutation no-op detector (Catalog #105/#139/#272)
  proves the reallocated bytes change the rendered frame.

## Cargo-cult audit per assumption

- "SegNet interiors are free" — CARGO-CULTED, already FALSIFIED (codex Finding 2);
  unwound: P18 uses the full-grid saliency, not a boundary band.
- "Pose can be ignored for rate allocation" — CARGO-CULTED (the bug this memo
  fixes); unwound: P19 null-subset + AIL.
- "One-shot saliency = optimal" — CARGO-CULTED (ignores AIL non-linearity);
  unwound: M2/M3 relinearization.
- "Isotropic pose sensitivity" — CARGO-CULTED (PoseNet dims are pre-whitened);
  unwound: Mahalanobis Σ⁻¹ per-dim norm = HARD-EARNED from `upstream/modules.py`.

## Canonical-vs-unique decision per layer

- water-fill bisection on θ: ADOPT_CANONICAL (standard reverse-water-filling).
- Dykstra redirect: ADOPT_CANONICAL (`tac.dykstra_pareto_solver`, Catalog #372).
- differentiable scorer path: ADOPT_CANONICAL (`tac.differentiable_eval_roundtrip`,
  eval_roundtrip non-negotiable).
- joint `w_i` weight + null-subset detector: FORK (unique to this solver; the
  two-term seg+pose+AIL weight is not in any existing helper).
- recon_pixel_weight channel: ADOPT_CANONICAL (codex's `#1590` shared channel),
  extended to carry the pose-weight term.

## horizon-class

`frontier_pursuit` — the lever targets the per-byte marginal at the current
frontier (pose-binding regime), not a plateau-adjacent rewrite.

## Z8 specialization — rate-binding regime (dead-zone PROTECTION, not repair-spend)

Operator correction, 2026-05-31 verbatim: *"the binding axis for Z8 is now
rate, and P18/P19 is the protection surface for lossy wavelet dead-zoning, not
a repair-spend toy."*

The general joint solver above is written for the contest-frontier operating
point (`pose_avg ≈ 3e-5`), where pose is the binding axis and `w_i` decides
where to SPEND a finite rate budget. **Z8 is at a different operating point**:
the 600-pair byte-closed advisory (`experiments/results/z8_600pair_byte_closed_contest_score_advisory/result.json`,
`[macOS-CPU advisory]` non-promotable) measured score **104.94** with rate term
`25·4.05 = 101.3` = **97% of the score** — the Mallat wavelet codec is
near-LOSSLESS (combined distortion 3.6, FAITHFUL render `d_seg=0.013`) paid for
with catastrophic rate (152 MB archive.zip, `rate=4.05`, 4× the 37.5 MB
denominator). **The binding axis is RATE, not pose** (the 64-pair probe's
pose-binding suspicion is REVERSED at byte-closed 600-pair scale).

At a rate-binding near-lossless codec the lever is **SUBTRACTIVE**, not
additive. The joint `w_i` is not a repair budget — it is the **dead-zone mask
threshold** for lossy wavelet detail-band quantization:

```
DEADZONE   = { atoms i : w_i < θ_deadzone }   # low joint weight → safe to zero
                                               #   (seg-flat AND pose-null)
PROTECT    = { atoms i : w_i ≥ θ_deadzone }   # high joint weight → keep precision
                                               #   (seg-boundary OR pose-sensitive)
```

Rate savings come from quantizing the DEADZONE detail coefficients to zero
(crush `rate 4.05 → <0.01`); the already-excellent distortion is preserved by
keeping bits on PROTECT. P19's null-subset `{i : ‖J_pose,i‖_{Σ⁻¹}≈0}` is
therefore "where it is safe to THROW BYTES AWAY," not "where to spend repair."
"Don't spend rate savings on pose-sensitive pairs" = keep PROTECT-set coefficients
at full precision; harvest bytes only from DEADZONE.

Codex landed this in HEAD (`d76356157` "Make Z8 P18/P19 waterfill rate-bound"):
`tac.optimization.joint_p18_p19_waterfill` now emits `rate_attack_deadzone_mask`
+ `distortion_protect_mask`, the Z8 joint driver `forbids SegNet-only
water-fill`, and the contract schema is
`select_low_joint_weight_atoms_for_wavelet_detail_band_deadzone`. This memo's
general solver and that Z8 specialization are the same `w_i`; the operating
point decides whether the lever spends (additive, contest-frontier) or
dead-zones (subtractive, Z8 rate-binding).

Z8 is **NOT** a near-term ratification candidate (546× from frontier). The path
is the gradient-driven rate/distortion co-optimization (#1591/#1592) attacking
the RATE axis first (30× larger than the distortion gap), NOT paired-CUDA
confirmation of the current near-lossless substrate.

## Wire-in / next steps (gated, MLX-first)

1. Let `#1590` land the generic `recon_pixel_weight` channel (P18 plumbing).
2. NEW `#1591`: implement the joint solver — N1 null-subset detector ($0 MLX) +
   joint `w_i` + M1 water-fill + Dykstra redirect. Sweep on the Z8 wavelet+WZ
   archive (now faithful) and the z5 archive when its render is classified.
3. Validate the P19 null-subset on a real contest-shaped render (the same gate
   codex named for `#1590`): perturb `NULL_pose` → d_pose unchanged; perturb its
   complement → d_pose moves. Two-gate ratification (PIXEL_CONSUMED + FAITHFUL).
4. Only after M1 shows a pose-protected rate recovery: M2, then M3, then a single
   paired CPU+CUDA ratification (Catalog #246, ≤$20) on the best byte-closed
   candidate. No PR; 0.189 gate holds.

---

## CORRECTION FOOTER — full-video joint backprop is the authority (APPEND-ONLY, 2026-05-31, per Catalog #110/#113 HISTORICAL_PROVENANCE)

Operator binding correction 2026-05-31 (verbatim, two messages):

> "doesn't repair and gradient and backprop and waterfill and all have to
> backprop against full video"

> "The mathematically correct object is full-video, joint backprop of the
> contest action: every repair/allocation/quantizer decision should ultimately
> see the whole video score functional, not a frozen local map. The practical
> implementation is a stochastic approximation to that: MLX VJP on
> frame/pair/region batches, STE through quantization/archive boundaries,
> repeated relinearization, and mandatory periodic full-video archive
> inflate/eval replay so minibatch gradients cannot drift into a local trick.
> That is exactly why I added the fresh-surface requirement to the Z8 iterative
> search. A "waterfill" map is only a local tangent plane; after each accepted
> archive mutation we need to recompute the P18/P19 surface and keep the
> full-video ledger as the authority path."

This SUPERSEDES any reading of §M1/M2/M3, §N1/N2/N3 null-subset detection,
§reverse water-fill, or §repair (DROP-MANY) as per-pair-isolated or
per-frame-isolated optimization. The binding is:

1. **The authority object is `S_full`.** The optimized functional is the
   full-video contest action
   `S_full = 100·d_seg(θ) + sqrt(10·d_pose(θ)) + 25·B(θ)/N`
   evaluated over ALL 1200 frames / 600 pairs / the ONE byte-closed archive.zip
   — never a per-pair or per-frame proxy. Every repair, allocation, and
   quantizer decision is a derivative of `S_full`.

2. **The joint `w_i` IS the full-video per-byte master gradient (not two
   hand-mixed maps).**
   `w_i = |∂S_full/∂x_i| = 100·|∂d_seg/∂x_i| + (5/sqrt(10·d_pose))·|∂d_pose/∂x_i| + (25/N)·∂B/∂x_i`.
   Because shared decoder / codebook / latent bytes decode into *every* frame,
   each component is a SUM over all frames `x_i` participates in. **P18** (the
   SegNet argmax-flip term `100·|∂d_seg/∂x_i|`) and **P19** (the PoseNet
   Mahalanobis/AIL null-subset term `(5/sqrt(10·d_pose))·||J_pose,i||_{Σ⁻¹}`) are
   the two COMPONENTS of ONE full-video master gradient — not two independent
   maps to blend by hand. This is the formal reason
   `tools/extract_master_gradient.py` + `.omx/state/master_gradient_anchors.jsonl`
   are the canonical full-video ledger; the solver consumes that ledger, it does
   not recompute a per-pair-local approximation of it.

3. **Practical stochastic approximation (the implementable surrogate of the
   ideal):**
   - **MLX VJP** (`mx.vjp` / value-and-grad) on frame / pair / region
     minibatches — an unbiased stochastic estimate of `∂S_full/∂θ`.
   - **STE** (straight-through estimator) through every quantization and
     archive-packing boundary so the gradient reaches the bytes the contest
     actually charges (`25·B/N`).
   - **Repeated relinearization** (M2 IRLS/EM) — the AIL gain
     `5/sqrt(10·d_pose)` and the byte-cost term are nonlinear, so a single
     tangent plane is wrong away from the current point; refit.
   - **Mandatory periodic full-video archive inflate/eval REPLAY.** The
     minibatch surrogate is trusted only as long as a periodic FULL-video
     `inflate.sh → upstream/evaluate.py` (the byte-closed archive, all frames)
     confirms the minibatch trajectory. The replay ledger is the anti-drift
     authority: a minibatch step that "improves" the proxy but the full-video
     replay does NOT confirm is a local trick (PR97-class single-axis
     cargo-cult) and is rejected.

4. **Fresh-surface requirement (the local-tangent-plane / trust-region
   discipline).** A waterfill / P18/P19 surface is a LOCAL TANGENT PLANE of
   `S_full` at the archive it was linearized at. After EACH accepted archive
   mutation the surface is STALE — recompute P18 (SegNet argmax-flip) and P19
   (PoseNet null-subset + Mahalanobis) on the NEW archive before the next
   decision. The full-video inflate/eval ledger, not the stale tangent plane, is
   the authority path. This is the same fresh-surface requirement the operator
   added to the Z8 iterative search.

5. **Forbidden cargo-cult: per-pair-local-linearization as the objective.**
   Treating a frozen per-pair gradient map as the optimization *target* (rather
   than as one minibatch sample of `∂S_full/∂θ` that must be periodically
   reconciled against the full-video replay) is the forbidden pattern. It drops
   the cross-frame coupling of shared bytes and mis-ranks which bytes to
   dead-zone vs protect. M1 / M2 / M3 + N1 / N2 / N3 + water-fill + repair ALL
   carry this contract.

### Consumer contract (binds `tac.optimization.joint_p18_p19_waterfill` + the #1592 solver)

- The gradients fed to `build_joint_p18_p19_waterfill_surface(...)` MUST be
  full-video master-gradient components (the SUM-over-frames form), not
  per-pair-isolated tangents.
- The returned surface is a LOCAL TANGENT PLANE: it carries
  `surface_is_local_tangent_plane=True` +
  `recompute_required_after_archive_mutation=True` + the
  `linearization_archive_sha` it was linearized at. The solver MUST recompute it
  after each accepted archive mutation
  (`surface_is_stale_for_archive(surface, new_archive_sha) == True`).
- The full-video inflate/eval REPLAY ledger is the authority; the surface is the
  local move *proposer* only — never the score authority. All surfaces remain
  `[macOS-MLX research-signal]` / non-promotable per Catalog #127/#192/#317/#323/#341
  until a paired CPU+CUDA full-video replay ratifies a byte-closed candidate.

---

## APPEND 2026-05-31 (operator directive) — deterministic nonsmooth bilevel optimal-control SUPERSEDES "better minibatches"

APPEND-ONLY per Catalog #110/#113. This SUPERSEDES the FRAMING of the prior
"Practical stochastic approximation" section: the stochastic/minibatch form is
demoted to **probe-ranking only, never authority**. The ideal object is not
"less-biased SGD" — it is a deterministic nonsmooth bilevel optimal-control
problem over the full video and archive.

### Operator directive (verbatim, binding)

> "The way to do better is to stop thinking of it as 'better minibatches' and
> formulate it as a deterministic nonsmooth bilevel optimal-control problem over
> the full video and archive."

### The ideal object

```
min over {renderer, state, quantizer, allocation, archive} decisions

  S = 100*d_seg(full_video)
    + sqrt(10*d_pose(full_video))
    + 25*archive_bytes/N

subject to:
  deterministic decoder/runtime
  legal archive grammar
  exact custody
  hardware/eval axis constraints
```

### The 8 bleeding-edge mandates (the upgrade over stochastic approximation)

1. **Full-video deterministic VJP — chunking is EXACT ACCUMULATION, not
   sampling.** `grad = sum(pair_chunk_grads)` over the WHOLE video. This is the
   full-batch gradient evaluated with memory scheduling; it is NOT a stochastic
   estimate. (Corrects the prior section's `mx.vjp on minibatches = unbiased
   stochastic estimate` framing: the chunk loop is memory scheduling of the full
   batch, not Monte-Carlo over pairs.)

2. **Adjoint / checkpointed full-sequence backprop.** For Z8 / Z7 / Dreamer /
   Mamba use scan/adjoint-style backprop through the recurrent / state-space
   dynamics. Gradient checkpointing is the canonical compute-for-memory move for
   long sequences: recompute activations during backward rather than dropping to
   random batches (less activation memory, some recompute cost).

3. **Implicit differentiation THROUGH the allocator (the FORBIDDEN-freeze
   correction).** Water-fill is NOT "compute sensitivity, freeze it, allocate."
   It is an inner optimization LAYER: `a*(θ) = argmin_a D(θ,a) + λ·R(a)`; then
   differentiate through the KKT / optimality conditions (OptNet / CVXPYlayers
   differentiable-optimization-layer lineage; implicit-function-theorem through
   the solver). The frozen-sensitivity threshold is a Phase-1 baseline ONLY; the
   canonical allocator is bilevel.

4. **Nonsmooth but score-aligned losses.** Soft KL / probability matching is
   WRONG near SegNet boundaries (the argmax-flip rate `d_seg` is nonsmooth).
   Use margin / subgradient objects: Crammer-Singer hinge, Fenchel-Young
   structured losses (connect convex losses ↔ structured prediction ↔ sparsity ↔
   generalized entropies ↔ margins), Clarke-subgradient treatment of argmax
   regions. (This is the SegNet sister of the existing `boundary_argmax_hinge`
   Z8 seg-lever finding — formalize it as Fenchel-Young.)

5. **Joint P18/P19 objective WITH interaction terms** (extends the surface):
   ```
   w_i = 100*|dL_seg/dx_i|
       + (5/sqrt(10*d_pose))*||J_pose,i||_{Sigma^-1}
       + interaction terms
   ```
   SegNet boundary flips and PoseNet Mahalanobis/null-subset behavior are
   optimized TOGETHER (second-order coupling), not summed as two frozen maps.

6. **Sequential convex / trust-region solve (the non-stochastic loop).**
   `full-video VJP → solve allocator/quantizer subproblem → materialize archive
   → exact inflate/eval → accept/reject → relinearize`. Minibatch/window
   gradients may RANK probes; they are NEVER authority. (This is the prior
   fresh-surface / mandatory full-video replay discipline, now named as the
   canonical SCP/trust-region outer loop.)

7. **Discrete archive decisions need mixed continuous/discrete math.**
   Quantization and bytes are discontinuous; `archive_bytes` is NOT a smooth
   loss. STE is a BRIDGE only — then CERTIFY with exact replay. For final coder
   decisions use DP, branch-and-bound, coordinate descent with EXACT byte deltas,
   or MILP/convex relaxations where tractable.

8. **The frontier is NOT "more SGD."** It is full-video variational
   optimization with implicit solver gradients, nonsmooth margin objectives,
   exact archive acceptance, and second-order interaction modeling.

### Contest mode vs production mode

```
contest mode:                         production mode:
  full-video resident MLX tensors       same math, but streaming/minibatch ALLOWED
  deterministic pair-chunk VJP          regularized across clips
    accumulation (exact full-batch)     validation-heldout behavior matters
  implicit-diff waterfill allocator     anti-overfit constraints on
  STE/prox through quantization           scorer-specific repairs
  exact archive replay after accepts
  NO stochastic authority
```

### Phased implementation roadmap (Carmack MVP-first; binds #1591/#1592/Z8 rate-attack)

- **Phase 1 (in-flight subagent `a797c158`):** full-video EXACT-ACCUMULATION
  VJP joint `w_i` → freeze-allocate dead-zone baseline → exact archive replay
  accept/reject via `tools/z8_600pair_byte_closed_contest_score_advisory.py`.
  This IS a valid Phase 1 of mandate 6 (SCP outer loop) with a frozen Phase-1
  inner solve. Honestly tag as Phase-1 baseline. The allocator interface must be
  shaped so Phase 2 can slot the implicit-diff layer in WITHOUT a rewrite.
- **Phase 2:** implicit-diff allocator — `a*(θ)=argmin_a D+λR` differentiated
  through KKT (OptNet/CVXPYlayers); replaces the frozen threshold (mandate 3).
- **Phase 3:** nonsmooth margin loss — Fenchel-Young / Crammer-Singer hinge /
  Clarke-subgradient for the SegNet argmax surface (mandate 4).
- **Phase 4:** adjoint/checkpointed full-sequence backprop for Z8/Z7/Dreamer/
  Mamba state-space dynamics (mandate 2).
- **Phase 5:** discrete archive coder via DP / branch-and-bound / exact-byte-
  delta coordinate descent (mandate 7); STE certified by exact replay.
- **Outer loop (all phases):** sequential-convex / trust-region accept/reject
  with full-video inflate/eval as sole authority (mandate 6).

### Canonical follow-ons (per Catalog #344)

- Register the deterministic-bilevel objective + the interaction-term joint
  `w_i` as canonical equations once Phase 1 lands its first empirical anchor
  (`FORMALIZATION_PENDING` until then).
- References for the eventual canonical-equation provenance: OptNet (Amos &
  Kolter 2017), CVXPYlayers (Agrawal et al. 2019), Fenchel-Young losses (Blondel
  et al. JMLR), gradient checkpointing (Chen et al. 2016 / HF docs), INR video /
  RNeRV long-training design, Geometric-Transformation-Embedded Mamba learned
  video compression.
